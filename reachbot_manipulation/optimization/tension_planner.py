"""Optimization of the ReachBot cable tensions, after the cables have been attached to the environment"""

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
from reachbot_manipulation.optimization.metrics import log_prob_of_success


class TensionPlanner:
    """Determine the tensions in the cables to apply a desired wrench with the robot's body

    This problem is DPP-parameterized so that it can be efficiently resolved when the pose of the robot changes

    Args:
        robot_pos (np.ndarray): Position of the robot's center of mass, shape (3,)
        robot_orn (np.ndarray): Orientation of the robot (XYZW quaternion), shape (4,)
        local_cable_positions (np.ndarray): Locations of where the cables originate from the robot, in the robot's
            body frame. Shape (n_cables, 3)
        attached_points (np.ndarray): Locations where the cables are attached to the world (world frame),
            shape (n_cables, 3)
        applied_wrench (npt.ArrayLike): Wrench to apply with the robot, shape (6,)
        max_tension (float): Maximum tension in the cables
        grasp_means (Optional[np.ndarray]): Mean pulling force for each grasp site, shape (n_cables,).
            Defaults to None (If no perception info is available, use the maximum tension as the mean).
        grasp_stdevs (Optional[np.ndarray]): Standard deviation of the pull force distribution for each grasp site,
            shape (n_cables,). Defaults to None (If no perception info is available, use a fraction of the max tension
            as the standard deviation)
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
        grasp_means: Optional[np.ndarray] = None,
        grasp_stdevs: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        # Handle inputs
        self.pose = np.concatenate([robot_pos, robot_orn])
        self.offsets = np.atleast_2d(local_cable_positions)
        self.attached_points = np.atleast_2d(attached_points)
        self.desired_wrench_world_frame = np.ravel(applied_wrench)
        self.max_tension = max_tension
        self.n_cables, self.dim = self.offsets.shape
        self.verbose = verbose
        if grasp_means is None:
            # Use a default distribution. Note that the standard dev is just chosen slightly arbitrarily
            grasp_means = max_tension * np.ones(self.n_cables)
            grasp_stdevs = (max_tension / 5) * np.ones(self.n_cables)
        else:
            grasp_means = np.ravel(grasp_means)
            grasp_stdevs = np.ravel(grasp_stdevs)
            # Check input
            if len(grasp_means) != self.n_cables or len(grasp_stdevs) != self.n_cables:
                raise ValueError(
                    "Invalid grasp site pull force distribution: Must provide mean and standard deviation for each site"
                )
            if np.any(grasp_means < 0) or np.any(grasp_stdevs < 0):
                raise ValueError(
                    "Invalid grasp site pull force distribution: Negative values encountered"
                )
        self.grasp_means = grasp_means
        self.grasp_stdevs = grasp_stdevs
        # Set up variables and parameters
        self.desired_wrench_base_frame = cp.Parameter(6)
        self.directions_base_frame = cp.Parameter((self.n_cables, 3))
        self.tensions = cp.Variable(self.n_cables)
        # This assigns a value to the parameters (wrench and direction)
        self.update_robot_pose(robot_pos, robot_orn)
        # Construct the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    @property
    def objective(self) -> Union[cp.Minimize, cp.Maximize]:
        """Optimization objective"""
        # Maximise the probability of success metric, based on the pull force distributions
        return cp.Maximize(
            log_prob_of_success(self.tensions, self.grasp_means, self.grasp_stdevs)
        )

    @property
    def constraints(self) -> list[cp.Expression]:
        """Optimization constraints"""
        return [
            self.tensions <= self.max_tension,
            self.tensions >= 0,
            self.applied_wrench_base_frame == self.desired_wrench_base_frame,
        ]

    def solve(self) -> None:
        """Solves the optimization problem"""
        self.prob.solve(solver=cp.ECOS)
        if self.prob.status != cp.OPTIMAL:
            raise OptimizationError(
                "Could not determine a solution for the given wrench and robot configuration.\n"
                + f"Problem status: {self.prob.status}"
            )
        if self.verbose:
            print_problem_info(self.prob)

    def update_robot_pose(self, pos: npt.ArrayLike, orn: npt.ArrayLike) -> None:
        """Update the position and orientation of the robot

        This will also update the values of the optimization parameters which depend on the pose of the robot
        (i.e. the base-frame wrench, and the base-frame directions)

        Args:
            pos (npt.ArrayLike): Position, shape (3,)
            orn (npt.ArrayLike): Orientation (XYZW quaternion), shape (4,)
        """
        self.pose = np.concatenate([pos, orn])
        assert self.pose.shape == (7,)
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

    def update_applied_wrench(self, wrench: npt.ArrayLike) -> None:
        """Updates the wrench applied by the robot

        Args:
            wrench (npt.ArrayLike): Desired world-frame applied wrench, shape (7,)
        """
        wrench = np.ravel(wrench)
        assert wrench.shape == (6,)
        self.desired_wrench_world_frame = wrench
        R_B2W = quat_to_rmat(self.pose[3:])
        R_W2B = R_B2W.T
        self.desired_wrench_base_frame.value = transform_wrench(
            self.desired_wrench_world_frame, R_W2B
        )

    @property
    def applied_wrench_base_frame(self) -> cp.Expression:
        """Resultant wrench from the cables, in the body frame. Shape (6,)"""
        wrench_components = []
        for i in range(self.n_cables):
            f = self.tensions[i] * self.directions_base_frame[i]
            r = self.offsets[i]
            wrench_components.append(cp.hstack([f, skew(r) @ f]))
        return cp.sum(wrench_components)

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


def main():
    # pylint: disable=import-outside-toplevel
    from reachbot_manipulation.geometry.points import cube_points
    from reachbot_manipulation.core.reachbot import Reachbot
    from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
    from reachbot_manipulation.utils.debug_visualizer import visualize_points
    from reachbot_manipulation.config.reachbot_config import LOCAL_CABLE_POSITIONS

    pos = (0, 0, 0)
    orn = (0, 0, 0, 1)
    attached_points = cube_points(sidelength=3)
    mass = 10.4
    gravity = -3.71
    gravity_wrench = np.array([0, 0, mass * gravity, 0, 0, 0])
    desired_wrench = -gravity_wrench
    max_tension = 30
    prob = TensionPlanner(
        pos, orn, LOCAL_CABLE_POSITIONS, attached_points, desired_wrench, max_tension
    )
    prob.solve()
    opt_tensions = prob.optimal_tensions
    print("Optimal tensions: ", opt_tensions)
    client = initialize_pybullet(gravity=gravity)
    visualize_points(attached_points, (1, 0, 0))
    robot = Reachbot(np.concatenate([pos, orn]), client=client)
    robot.attach_to(attached_points, opt_tensions)
    robot.step()
    input()


if __name__ == "__main__":
    main()
