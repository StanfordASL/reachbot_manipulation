"""Optimization of the ReachBot boom tensions + body torque, after the booms have been attached to the environment

This employs a very simple understanding of how the added shoulder torques of booms (as compared with cables)
can aid in stabilizing the robot

This is primarily to check for feasiblility of a given pose when determining the workspaces of each type of robot
"""


from typing import Union, Optional

import numpy as np
import numpy.typing as npt
import cvxpy as cp

from reachbot_manipulation.utils.errors import OptimizationError
from reachbot_manipulation.utils.rotations import quat_to_rmat
from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.cvxpy_math import skew
from reachbot_manipulation.utils.grasp_utils import transform_wrench
from reachbot_manipulation.utils.cvxpy_utils import print_problem_info


class BoomWrenchOptimizationProblem:
    """Determine the tensions in the booms and net torque from the shoulders to apply a desired wrench

    This problem is DPP-parameterized so that it can be efficiently resolved when the pose of the robot changes

    Args:
        robot_pos (np.ndarray): Position of the robot's center of mass, shape (3,)
        robot_orn (np.ndarray): Orientation of the robot (XYZW quaternion), shape (4,)
        local_cable_positions (np.ndarray): Locations of where the booms originate from the robot, in the robot's
            body frame. Shape (n_booms, 3)
        attached_points (np.ndarray): Locations where the booms are attached to the world (world frame),
            shape (n_booms, 3)
        applied_wrench (npt.ArrayLike): Wrench to apply with the robot, shape (6,)
        max_tension (float): Maximum tension in the booms
        max_shoulder_torque (float): Maximum torque magnitude at each shoulder
        grasp_weights (Optional[np.ndarray]): Expected quality of each grasp, a value between 0 and 1 dictating
            the fraction of the max_tension that we expect to be able to achieve. Shape (n_booms,).
            Defaults to None (uniform weighting of 1)
        verbose (bool, optional): Whether to print info about the optimization problem after solving. Defaults to True.
    """

    def __init__(
        self,
        robot_pos: npt.ArrayLike,
        robot_orn: npt.ArrayLike,
        local_cable_positions: npt.ArrayLike,
        attached_points: npt.ArrayLike,
        applied_wrench: npt.ArrayLike,
        max_tension: float,
        max_shoulder_torque: float,
        grasp_weights: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        # Handle inputs
        self.offsets = np.atleast_2d(local_cable_positions)
        self.attached_points = np.atleast_2d(attached_points)
        self.desired_wrench_world_frame = np.ravel(applied_wrench)
        self.max_tension = max_tension
        self.max_shoulder_moment = max_shoulder_torque
        self.n_cables, self.dim = self.offsets.shape
        self.verbose = verbose
        if grasp_weights is None:
            grasp_weights = np.ones(self.n_cables)  # All good grasps
        elif np.any(grasp_weights > 1) or np.any(grasp_weights < 0):
            raise ValueError("Invalid grasp weights: Must be between 0 and 1")
        self.grasp_weights = grasp_weights
        # Set up variables and parameters
        self.desired_wrench_base_frame = cp.Parameter(6)
        self.directions_base_frame = cp.Parameter((self.n_cables, 3))
        self.tensions = cp.Variable(self.n_cables)
        self.torque = cp.Variable(3)
        # This assigns a value to the parameters
        self.update_robot_pose(robot_pos, robot_orn)
        # Construct the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    @property
    def objective(self) -> Union[cp.Minimize, cp.Maximize]:
        """Optimization objective"""
        # Minimize how close any given site is to its maximum tension, accounting for the expected grasp qualities
        return cp.Minimize(cp.max(self.tensions / self.max_per_cable_tension))

    @property
    def constraints(self) -> list[cp.Expression]:
        """Optimization constraints"""
        return [
            self.tensions <= self.max_per_cable_tension,
            self.tensions >= 0,
            cp.norm(self.torque) <= self.max_shoulder_moment * self.n_cables,
            self.applied_wrench_base_frame == self.desired_wrench_base_frame,
        ]

    def solve(self) -> None:
        """Solves the optimization problem"""
        self.prob.solve(solver=cp.ECOS)
        if self.prob.status != cp.OPTIMAL:
            raise OptimizationError(
                "Could not determine a static equilibrium solution for the given configuration.\n"
                + f"Problem status: {self.prob.status}"
            )
        if self.verbose:
            print_problem_info(self.prob)

    def update_robot_pose(self, pos: npt.ArrayLike, orn: npt.ArrayLike) -> None:
        """Update the position and orientation of the robot

        This will also update the values of the optimization parameters

        Args:
            pos (npt.ArrayLike): Position, shape (3,)
            orn (npt.ArrayLike): Orientation (XYZW quaternion), shape (4,)
        """
        R_B2W = quat_to_rmat(orn)
        R_W2B = R_B2W.T
        offsets_world_frame = self.offsets @ R_B2W.T
        start_points_world_frame = offsets_world_frame + pos
        self.directions_world_frame = normalize(
            self.attached_points - start_points_world_frame
        )
        # Update parameters
        self.desired_wrench_base_frame.value = transform_wrench(
            self.desired_wrench_world_frame, R_W2B
        )
        self.directions_base_frame.value = self.directions_world_frame @ R_W2B.T

    @property
    def max_per_cable_tension(self):
        """Maximum allowable tension in each cable, given the expected grasp qualities. Shape (n_cables,)"""
        return self.max_tension * self.grasp_weights

    @property
    def applied_wrench_base_frame(self) -> cp.Expression:
        """Resultant wrench from the cables, in the body frame. Shape (6,)"""
        wrench_components = []
        for i in range(self.n_cables):
            f = self.tensions[i] * self.directions_base_frame[i]
            r = self.offsets[i]
            wrench_components.append(cp.hstack([f, skew(r) @ f]))
        return cp.sum(wrench_components) + cp.hstack([(0, 0, 0), self.torque])

    @property
    def optimal_tensions(self) -> np.ndarray:
        """Optimal tension forces in each cable, shape (n_cables,)"""
        if self.tensions.value is None:
            raise ValueError("Cannot return the tensions, problem has not been solved")
        return self.tensions.value

    @property
    def optimal_forces(self) -> np.ndarray:
        """Optimal world-frame forces on the body from each cable, shape (n_cables, 3)"""
        return self.directions_world_frame * self.optimal_tensions.reshape(-1, 1)

    @property
    def optimal_torque(self) -> np.ndarray:
        if self.torque.value is None:
            raise ValueError("Cannot return the moment, problem has not been solved")
        return self.torque.value


def main():
    # pylint: disable=import-outside-toplevel
    from reachbot_manipulation.geometry.points import cube_points
    from reachbot_manipulation.core.reachbot import Reachbot
    from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
    from reachbot_manipulation.utils.debug_visualizer import visualize_points
    from reachbot_manipulation.config.reachbot_config import (
        LOCAL_CABLE_POSITIONS,
        MAX_BOOM_SHOULDER_TORQUE,
    )

    pos = (0, 0, 0)
    orn = (0, 0, 0, 1)
    attached_points = cube_points(sidelength=3)
    mass = 10.4
    gravity = -3.71
    gravity_wrench = np.array([0, 0, mass * gravity, 0, 0, 0])
    desired_wrench = -gravity_wrench
    max_tension = 30
    prob = BoomWrenchOptimizationProblem(
        pos,
        orn,
        LOCAL_CABLE_POSITIONS,
        attached_points,
        desired_wrench,
        max_tension,
        MAX_BOOM_SHOULDER_TORQUE,
    )
    prob.solve()
    opt_tensions = prob.optimal_tensions
    opt_moment = prob.optimal_torque
    print("Optimal tensions: ", opt_tensions)
    print("Optimal moment: ", opt_moment)
    client = initialize_pybullet(gravity=gravity)
    visualize_points(attached_points, (1, 0, 0))
    robot = Reachbot(np.concatenate([pos, orn]), boom_driven=True, client=client)
    robot.attach_to(attached_points, opt_tensions)
    robot.set_torque(opt_moment)
    robot.step()
    input()


if __name__ == "__main__":
    main()
