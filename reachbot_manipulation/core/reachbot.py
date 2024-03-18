"""Reachbot Pybullet implementation"""


from typing import Optional, Union

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
from reachbot_manipulation.utils.transformations import (
    transform_points,
    transform_point,
    invert_transform_mat,
    make_transform_mat,
)
from reachbot_manipulation.utils.rotations import quat_to_rmat
from reachbot_manipulation.utils.poses import tmat_to_pos_quat, pos_quat_to_tmat
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_TRANSFORMS,
    LOCAL_CABLE_POSITIONS,
    MIN_TENSION,
    MAX_TENSION,
    MAX_BOOM_SHOULDER_TORQUE,
)
from reachbot_manipulation.utils.python_utils import print_green
from reachbot_manipulation.control.cable_driven_parallel_robots import jacobian
from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.debug_visualizer import (
    visualize_line,
    remove_debug_objects,
)
from reachbot_manipulation.utils.dynamics import inertial_transformation
from reachbot_manipulation.geometry.points import cube_points
from reachbot_manipulation.optimization.tension_planner import TensionPlanner


class Reachbot:
    """Reachbot class for simulation in Pybullet

    Args:
        pose (npt.ArrayLike, optional): Initial pose of the robot when loaded. Defaults to
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) (At origin, pointed forward along x axis).
        kp (float, optional): Proportional gain for the cable PD controller. Defaults to 50.
        kv (float, optional): Derivative gain for the cable PD controller. Defaults to 50.
        kw (float, optional): Derivative gain on the body's velocity, if generating a moment with a boom-driven
            Reachbot. Defaults to 50
        boom_driven (bool, optional): Whether to use a boom-driven Reachbot (as opposed to cable-driven). Defaults to
            False (use cables)
        include_gripper (bool, optional): Whether to use a ReachBot model with an extra boom attached to the base to
            mimic how ReachBot might grasp an object. Defaults to False
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """

    NUM_CABLES = len(LOCAL_CABLE_POSITIONS)

    # EE-to-base
    # For the gripper version of reachbot
    EE_TRANSFORM = make_transform_mat(np.eye(3), [0, 0, -1.5])

    def __init__(
        self,
        pose: npt.ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        kp: float = 50,
        kv: float = 50,
        kw: float = 50,
        boom_driven: bool = False,
        include_gripper: bool = False,
        client: Optional[BulletClient] = None,
    ):
        self.client: pybullet = pybullet if client is None else client
        if not self.client.isConnected():
            raise ConnectionError(
                "Need to connect to pybullet before initializing the robot"
            )
        self.has_gripper = include_gripper
        self.urdf = (
            "reachbot_manipulation/assets/urdf/reachbot.urdf"
            if not include_gripper
            else "reachbot_manipulation/assets/urdf/reachbot_with_gripper.urdf"
        )
        self._dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.id = self.client.loadURDF(
            self.urdf,
            pose[:3],
            pose[3:],  # flags=pybullet.URDF_USE_INERTIA_FROM_FILE
        )
        self.recompute_inertial_properties()
        self.set_controller_gains(kp, kv, kw)
        self.boom_driven = boom_driven
        # Initializations
        self.cable_uids = [None] * self.NUM_CABLES
        self._num_attached = 0
        self._tensions = [0] * self.NUM_CABLES
        self._body_torque = np.zeros(3)
        self._attached_points = None
        self._max_site_tensions = MAX_TENSION * np.ones(self.NUM_CABLES)
        print_green("Reachbot is ready")

    def unload(self) -> None:
        """Remove the robot from the simulation"""
        self.client.removeBody(self.id)

    @property
    def pose(self) -> np.ndarray:
        """The current robot pose (position + XYZW quaternion) expressed in world frame

        Returns:
            np.ndarray: Position and quaternion, size (7,)
        """
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
        return np.concatenate([pos, orn])

    @property
    def tmat(self) -> np.ndarray:
        """The current robot pose in world frame, expressed as a transformation matrix

        Returns:
            np.ndarray: Transformation matrix (Robot to World), shape (4,4)
        """
        return pos_quat_to_tmat(self.pose)

    @property
    def position(self) -> np.ndarray:
        """Just the position component of the full pose

        Returns:
            np.ndarray: (3,) position vector
        """
        return self.pose[:3]

    @property
    def orientation(self) -> np.ndarray:
        """Just the quaternion component of the full pose

        Returns:
            np.ndarray: (4,) XYZW quaternion
        """
        return self.pose[3:]

    @property
    def rmat(self) -> np.ndarray:
        """The orientation of the robot expressed as a rotation matrix

        Returns:
            np.ndarray: Rotation matrix (Robot to World), shape (3,3)
        """
        return quat_to_rmat(self.orientation)

    @property
    def velocity(self) -> np.ndarray:
        """Linear velocity of the robot, with respect to the world frame xyz axes

        Returns:
            np.ndarray: [vx, vy, vz] linear velocities, shape (3,)
        """
        lin_vel, _ = self.client.getBaseVelocity(self.id)
        return np.array(lin_vel)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Angular velocity of the robot, about the world frame xyz axes

        Returns:
            np.ndarray: [wx, wy, wz] angular velocities, shape (3,)
        """
        _, ang_vel = self.client.getBaseVelocity(self.id)
        return np.array(ang_vel)

    @property
    def ee_pose(self) -> np.ndarray:
        """The current end-effector pose (position + XYZW quaternion) expressed in world frame

        Returns:
            np.ndarray: Position and quaternion, size (7,)
        """
        return tmat_to_pos_quat(self.ee_tmat)

    @property
    def ee_tmat(self) -> np.ndarray:
        """The current end-effector transformation matrix (EE-to-world)

        Returns:
            np.ndarray: Transformation matrix, shape (4, 4)
        """
        # Base-to-world @ EE-to-base
        return self.tmat @ self.EE_TRANSFORM

    @property
    def inertia(self) -> np.ndarray:
        """Body inertia tensor for the robot, shape (3, 3)"""
        return self._inertia

    @property
    def inv_inertia(self) -> np.ndarray:
        """Inverse of the robot's body inertia tensor, shape (3, 3)"""
        return self._inv_inertia

    @property
    def world_inertia(self) -> np.ndarray:
        """World-frame inertia tensor of the robot, shape (3, 3)

        This takes into account the current rotation of robot
        """
        R = self.rmat
        return R @ self.inertia @ R.T

    @property
    def world_inv_inertia(self) -> np.ndarray:
        """Inverse of the world-frame inertia tensor of the robot, shape (3, 3)

        This takes into account the current rotation of robot
        """
        R = self.rmat
        return R @ self.inv_inertia @ R.T

    @property
    def mass(self) -> float:
        """Mass of the robot"""
        return self._mass

    @property
    def local_com_position(self) -> np.ndarray:
        """Position of the center of mass of the robot w.r.t. the base, in local frame. Shape (3,)"""
        return self._local_com_position

    @property
    def world_com_position(self) -> np.ndarray:
        """Position of the center of mass of the robot, in world frame. Shape (3,)"""
        return transform_point(self.tmat, self.local_com_position)

    # TODO should this be updated to use the linear/angular momentums instead?
    @property
    def state_vector(self) -> np.ndarray:
        """The state vector x, such that x_dot = Ax + Bu

        We compose the state as [position, velocity, quaternion, angular velocity] ∈ R13
        """
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
        lin_vel, ang_vel = self.client.getBaseVelocity(self.id)
        return np.concatenate([pos, lin_vel, orn, ang_vel])

    def get_link_transform(self, link_index: int) -> np.ndarray:
        """Calculates the transformation matrix (w.r.t the world) for a specified link

        Args:
            link_index (int): Index of the link on the robot

        Returns:
            np.ndarray: Transformation matrix (link to world). Shape = (4,4)
        """
        # For reachbot, we only have one non-base link if we are using the gripper version
        if not self.has_gripper or self.has_gripper and link_index != 0:
            raise ValueError(f"Invalid link index: {link_index}")
        link_state = self.client.getLinkState(
            self.id, link_index, computeForwardKinematics=True
        )
        # First two link state values are linkWorldPosition, linkWorldOrientation
        # There are other state positions and orientations, but they're confusing. (TODO check on these)
        pos, quat = link_state[:2]
        return make_transform_mat(quat_to_rmat(quat), pos)

    def reset_to_ee_pose(self, pose: npt.ArrayLike) -> None:
        """Resets the position of the robot to achieve a target end-effector pose

        This will currently NOT adjust any of the joints in a "smart" way, it will just reset the position of the base
        given the current joint configuration

        Args:
            pose (npt.ArrayLike): Desired position + XYZW quaternion end-effector pose, shape (7,)
        """
        # Notation: EE: End effector, B: Base, W: World
        des_EE2W = pos_quat_to_tmat(pose)
        cur_B2W = pos_quat_to_tmat(self.pose)
        cur_EE2W = pos_quat_to_tmat(self.ee_pose)
        cur_W2EE = invert_transform_mat(cur_EE2W)
        cur_B2EE = cur_W2EE @ cur_B2W
        des_B2W = des_EE2W @ cur_B2EE
        self.reset_to_base_pose(tmat_to_pos_quat(des_B2W))

    def reset_to_base_pose(self, pose: npt.ArrayLike) -> None:
        """Resets the base of the robot to a target pose

        Args:
            pose (npt.ArrayLike): Desired position + XYZW quaternion pose of the robot's base, shape (7,)
        """
        self.client.resetBasePositionAndOrientation(self.id, pose[:3], pose[3:])

    @property
    def bounding_box(self) -> np.ndarray:
        """Current axis-aligned bounding box of the robot body, shape (2, 3)"""
        return np.array(self.client.getAABB(self.id, -1))

    def recompute_inertial_properties(self) -> None:
        """Calculate the inertial properties based on the current state of the robot in sim

        This is more accurate than the fixed values from the URDF, but it is fairly expensive to
        compute and should NOT be done on every simulation step.

        This will update the mass, inertia, inv_inertia, and center of mass
        """
        # Note: Mass will be fixed
        mass = 0.0
        inertia = np.zeros((3, 3))
        com = np.zeros(3)
        T_B2W = self.tmat  # Base to world
        links = (-1, 0) if self.has_gripper else (-1,)
        for link in links:
            link_info = pybullet.getDynamicsInfo(self.id, link)
            link_mass = link_info[0]
            link_inertia_diagonal = link_info[2]
            if link == -1:  # Separate handling for base link
                inertia += np.diag(link_inertia_diagonal)
                com += link_mass * T_B2W[:3, 3]
            else:
                T_L2W = self.get_link_transform(link)  # Link to world
                T_L2B = invert_transform_mat(T_B2W) @ T_L2W  # Link to base
                inertia += inertial_transformation(
                    link_mass, np.diag(link_inertia_diagonal), T_L2B
                )
                com += link_mass * T_L2W[:3, 3]
            mass += link_mass
        com /= mass
        self._local_com_position = T_B2W[:3, :3].T @ (com - T_B2W[:3, 3])
        self._mass = mass
        self._inertia = inertia
        self._inv_inertia = np.linalg.inv(inertia)

    @property
    def local_cable_positions(self) -> np.ndarray:
        """The cables' starting positions, in body frame. Shape (n_cables, 3)"""
        return LOCAL_CABLE_POSITIONS

    @property
    def cable_tmats(self) -> np.ndarray:
        """Transformation matrices for the cable starting points on the robot. "Cable to world"

        Returns:
            np.ndarray: Transformation matrices, shape (NUM_CABLES, 4, 4)
        """
        T_R2W = self.tmat
        # Robot to World @ Tendon to Robot = Tendon to World
        return np.array([T_R2W @ T_T2R for T_T2R in LOCAL_CABLE_TRANSFORMS])

    @property
    def cable_positions(self) -> np.ndarray:
        """Positions of the cable starting points on the robot, in world frame

        Returns:
            np.ndarray: Positions, shape (NUM_CABLES, 3)
        """
        return transform_points(self.tmat, self.local_cable_positions)

    # TODO update this to handle the case when not all cables are attached
    @property
    def cable_lengths(self) -> np.ndarray:
        """Lengths of all cables, shape (n_cables,)"""
        return np.linalg.norm(self.attached_points - self.cable_positions, axis=1)

    # TODO update this to handle the case when not all cables are attached
    @property
    def cable_speeds(self) -> np.ndarray:
        """Speeds of all cables, shape (n_cables)"""
        return self.jacobian @ np.concatenate(self.client.getBaseVelocity(self.id))

    # TODO update this to handle the case when not all cables are attached
    @property
    def cable_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        """Lengths and speeds of all cables, computed together to minimize calculations

        Returns:
            tuple[np.ndarray, np.ndarray]:
                np.ndarray: Cable lengths, shape (n_cables,)
                np.ndarray: Cable speeds, shape (n_cables,)
        """
        # lengths and speeds, computed together to minimize calculations
        start_pts = self.cable_positions
        end_pts = self.attached_points
        lengths = np.linalg.norm(end_pts - start_pts, axis=1)
        pos = self.client.getBasePositionAndOrientation(self.id)[0]
        J = jacobian(start_pts, end_pts, pos)
        vels = J @ np.concatenate(self.client.getBaseVelocity(self.id))
        return lengths, vels

    @property
    def attached_points(self) -> np.ndarray:
        """Locations of the ends of the cables, attached to the world. Shape (n_cables, 3)

        NOTE: This currently assumes we are NOT in the "end effector movement" phase where one cable is detached
        """
        try:
            return self._attached_points
        except AttributeError as e:
            raise AttributeError("Attached points have not been set") from e

    @property
    def jacobian(self) -> np.ndarray:
        """Jacobian mapping body velocities (linear/angular) to cable velocities. Shape (n_cables, 6)"""
        if self.attached_points is None:
            raise ValueError("Unable to determine Jacobian: Unknown attachment points")
        return jacobian(self.cable_positions, self.attached_points, self.position)

    @property
    def min_tension(self) -> float:
        """Minimum allowable cable tension, in Newtons"""
        return MIN_TENSION

    @property
    def max_tension(self) -> float:
        """Maximum allowable cable tension, in Newtons"""
        return MAX_TENSION

    @property
    def max_site_tensions(self) -> np.ndarray:
        """Maximum tension (in Newtons) on a per-site basis, shape (n_cables,)"""
        return self._max_site_tensions

    @max_site_tensions.setter
    def max_site_tensions(self, tensions: npt.ArrayLike):
        assert len(tensions == self.NUM_CABLES)
        self._max_site_tensions = np.clip(tensions, None, self.max_tension)

    @property
    def max_shoulder_torque(self) -> float:
        """Maximum torque magnitude at each shoulder. 0 if cable-driven"""
        if not self.boom_driven:
            return 0
        return MAX_BOOM_SHOULDER_TORQUE

    @property
    def max_body_torque(self) -> float:
        """Maximum net torque from all shoulder torques. 0 if cable-driven"""
        if not self.boom_driven:
            return 0
        return MAX_BOOM_SHOULDER_TORQUE * self.num_attached_cables

    @property
    def tensions(self) -> np.ndarray:
        try:
            return self._tensions
        except AttributeError as e:
            raise AttributeError("Tensions have not been set") from e

    def attach_to(
        self,
        points: Union[np.ndarray, list[Optional[np.ndarray]]],
        tensions: np.ndarray,
        max_site_tensions: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Connect Reachbot's cables to a set of grasp points

        Args:
            points (Union[np.ndarray, list[Optional[np.ndarray]]]): Grasp locations, shape (n_cables, 3) if all cables
                are attached, otherwise a list of points (length = n_cables) which may be None if not attached.
            tensions (np.ndarray): Tension force in each cable, shape (n_cables). If a cable is not attached, set the
                corresponding tension value to 0
            max_site_tensions (Optional[npt.ArrayLike]): If the limit surfaces are known for the sites being attached
                to, include a maximum force for each site here. This can correspond to some confidence interval on the
                site's true allowable pull force, for instance. Shape (n_cables,)
        """
        self.attached_mask = np.array([p is not None for p in points])
        if isinstance(points, np.ndarray):
            assert points.shape[0] == self.NUM_CABLES
            self._num_attached = self.NUM_CABLES
        else:
            assert len(points) == self.NUM_CABLES
            self._num_attached = sum(p is not None for p in points)
        self._attached_points = points
        self.set_nominal_tensions(tensions)
        self.set_nominal_lengths(self.cable_lengths)
        self.set_nominal_speeds(np.zeros(self.NUM_CABLES))
        if max_site_tensions is not None:
            self.max_site_tensions = max_site_tensions
        self.set_tensions(tensions)

    def detach_from(self, cable_ids: Union[int, npt.ArrayLike]):
        """Detach a cable or set of cables

        Args:
            cable_ids (Union[int, npt.ArrayLike]): Cable indices to detach from the environment.
        """
        cable_ids = np.ravel(cable_ids)
        for cid in cable_ids:
            remove_debug_objects(self.cable_uids[cid])
            # TODO this causes nans in the numpy array - decide if this is bad
            self._attached_points[cid] = None
            new_tensions = self.tensions
            new_tensions[cid] = 0
            self._tensions = new_tensions
            self._num_attached -= 1
            self.attached_mask[cid] = False

    @property
    def num_attached_cables(self) -> int:
        """Number of cables currently attached to the environment"""
        return self._num_attached

    @property
    def all_cables_attached(self) -> bool:
        """Whether or not all cables are connected to the environment"""
        return self.num_attached_cables == self.NUM_CABLES

    @property
    def is_detached(self) -> bool:
        """If no cables are connected to the environment"""
        return self.num_attached_cables == 0

    def set_tensions(self, tensions: npt.ArrayLike) -> None:
        """Set the tensions in each cable. This will ensure that the commanded tensions are within actuator limits

        Args:
            tensions (npt.ArrayLike): Tension force in each cable, shape (n_cables)
        """
        self._tensions = np.clip(
            np.ravel(tensions), self.min_tension, self.max_site_tensions
        )

    def set_torque(self, torque: npt.ArrayLike) -> None:
        """Sets the resultant torque on the body from the boom shoulders. Only valid for a boom-driven Reachbot

        Args:
            torque (npt.ArrayLike): World-frame torque, shape (3,)
        """
        if not self.boom_driven:
            print("Warning: Attempting to set the torque on a cable-driven reachbot")
            return
        torque_norm = np.linalg.norm(torque)
        max_magnitude = self.max_body_torque
        if torque_norm > max_magnitude:
            torque = (torque / torque_norm) * max_magnitude
        self._body_torque = torque

    def set_nominal_tensions(self, tensions: npt.ArrayLike):
        """Set the nominal tensions for the controller (i.e. the gravity offset tensions for a certain body pose)

        Args:
            tensions (npt.ArrayLike): Cable tensions, shape (n_cables,)
        """
        assert len(tensions) == self.NUM_CABLES
        self.nominal_tensions = np.ravel(tensions)

    def set_nominal_lengths(self, lengths: npt.ArrayLike):
        """Set the nominal cable lengths for the controller (i.e. the cable lengths for a certain body pose)

        Args:
            lengths (npt.ArrayLike): Cable lengths, shape (n_cables,)
        """
        assert len(lengths) == self.NUM_CABLES
        self.nominal_lengths = np.ravel(lengths)

    def set_nominal_speeds(self, speeds: npt.ArrayLike):
        """Set the nominal cable speeds for the controller (Typically 0, unless we're moving along a trajectory)

        Args:
            speeds (npt.ArrayLike): Cable speeds, shape (n_cables,)
        """
        assert len(speeds) == self.NUM_CABLES
        self.nominal_speeds = np.ravel(speeds)

    def get_controller_tensions(self) -> np.ndarray:
        """Use the PD position controller to determine the cable tensions based on the current errors

        Returns:
            np.ndarray: Cable tensions, shape (n_cables,)
        """
        lengths, vels = self.cable_dynamics
        length_error = lengths - self.nominal_lengths
        vel_error = vels - self.nominal_speeds
        return self.nominal_tensions + self.kp * length_error + self.kv * vel_error

    def get_controller_torque(self) -> np.ndarray:
        if not self.boom_driven:
            return np.zeros(3)
        # derivative control on angular velocity
        return -self.kw * self.angular_velocity

    def set_controller_gains(self, kp: float, kv: float, kw: float = 0):
        """Update the gains on the PD controller

        Args:
            kp (float): Position gain
            kv (float): Velocity gain
            kw (float, optional): Angular velocity derivative gain. Only if using a boom-driven ReachBot. Defaults to 0
        """
        assert kp >= 0 and kv >= 0 and kw >= 0
        self.kp = kp
        self.kv = kv
        self.kw = kw

    def _step_controller(self):
        """Updates the commanded tensions in the cables based on the PD position controller"""
        if self.is_detached:
            return
        self.set_tensions(self.get_controller_tensions())
        if self.boom_driven:
            self.set_torque(self.get_controller_torque())

    def _step_cable_visuals(self) -> None:
        """Update the visualization of the cables"""
        if self.is_detached:
            return
        # colors = tension_to_rgb(self.tensions, (self.min_tension, self.max_tension))
        colors = np.zeros((8, 3))
        uids = []
        start_points = self.cable_positions
        end_points = self.attached_points
        for i in range(self.NUM_CABLES):
            if end_points[i] is None:
                uids.append(None)
            else:
                uids.append(
                    visualize_line(
                        start_points[i],
                        end_points[i],
                        color=colors[i],
                        replace_uid=self.cable_uids[i],
                    )
                )
        self.cable_uids = uids

    def _step_forces(self):
        """Update the applied forces on the Reachbot body based on the current robot state and cable tensions"""
        if self.is_detached:
            return
        for direction, tension, pos in zip(
            self.cable_directions, self.tensions, self.cable_positions
        ):
            if (
                direction is None or tension == 0 or np.isnan(tension)
            ):  # Detached or no force
                continue
            self.client.applyExternalForce(
                self.id, -1, direction * tension, pos, self.client.WORLD_FRAME
            )
        if self.boom_driven:
            self.client.applyExternalTorque(
                self.id, -1, self._body_torque, pybullet.WORLD_FRAME
            )

    @property
    def cable_directions(self) -> Union[np.ndarray, list[Optional[np.ndarray]]]:
        """Directions of each cable. None if the cable is detached. Length = n_cables"""
        if isinstance(self.attached_points, np.ndarray):
            # All cables attached
            return normalize(self.attached_points - self.cable_positions)
        else:
            # Not necessarily all cables attached
            dirs = []
            for start_pt, end_pt in zip(self.cable_positions, self.attached_points):
                dirs.append(None if end_pt is None else normalize(end_pt - start_pt))
            return dirs

    def step(self, step_control: bool = True, step_sim: bool = True):
        """Updates the state of the robot and the cables

        Args:
            step_control (bool, optional): Whether or not to update the cable tensions based on the default ReachBot PD
                controller. This can be set to False if we want to bypass the default controller. Defaults to True
            step_sim (bool, optional): Whether or not to step the simulation. Defaults to True (generally, this should
                always be the case)
        """
        if step_control:
            self._step_controller()
        self._step_cable_visuals()
        self._step_forces()
        if step_sim:
            self.client.stepSimulation()


def _main():
    gravity = -3.71
    client = initialize_pybullet(gravity=-3.71, bg_color=(0.5, 0.5, 0.5))
    robot = Reachbot(boom_driven=False)
    attachment_pts = cube_points(sidelength=10)
    external_wrench = np.array([0, 0, robot.mass * gravity, 0, 0, 0])
    applied_wrench = -external_wrench
    tension_prob = TensionPlanner(
        robot.position,
        robot.orientation,
        robot.local_cable_positions,
        attachment_pts,
        applied_wrench,
        robot.max_tension,
        verbose=False,
    )
    tension_prob.solve()
    robot.attach_to(attachment_pts, tension_prob.optimal_tensions)
    # pid = visualize_points(robot.ee_pose[:3], (1, 0, 0))
    while True:
        # pid = visualize_points(robot.ee_pose[:3], (1, 0, 0), replace_uid=pid)
        robot.step()
        # time.sleep(1 / 120)


if __name__ == "__main__":
    _main()
