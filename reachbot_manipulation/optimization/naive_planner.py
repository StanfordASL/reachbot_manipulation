"""Naive baseline planning methods for grasp site selection"""

from typing import Optional, Union

import numpy as np
import cvxpy as cp

from reachbot_manipulation.utils.cvxpy_math import skew
from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.cvxpy_utils import print_problem_info
from reachbot_manipulation.utils.errors import OptimizationError
from reachbot_manipulation.utils.rotations import quat_to_rmat
from reachbot_manipulation.optimization.stance_planner import (
    determine_directions,
    get_world_frame_cable_start_points,
)


class NaivePlanner:
    """Determine the configuration of cables that can apply a desired wrench, using a "naive" objective that doesn't
    consider uncertainty or disturbances

    The naive objective refers to extending the cables outwards to their nearest respective sites: This seems like a
    stable configuration at a first glance, but this tends to be very unstable against torque disturbances
    (though, good at managing forces)

    Args:
        robot_pos (np.ndarray): Position of the robot's center of mass, shape (3,)
        robot_orn (np.ndarray): Orientation of the robot (XYZW quaternion), shape (4,)
        local_cable_positions (np.ndarray): Locations of where the cables originate from the robot, in the robot's
            body frame. Shape (n_cables, 3)
        end_points (np.ndarray): Possible locations where the cables can be attached to the world (world frame),
            shape (n_grasp_sites, 3)
        local_cone_normals (np.ndarray): Nominal normal vectors for each cable (robot-frame), defining the central axes
            of the second-order cones, shape (n_cables, 3)
        cone_angle (float): Interior angle of the second-order cones, defining the allowable directions of the cables
        applied_wrench (np.ndarray): Nominal wrench to apply with the robot, shape (6,)
        max_tension (float): Maximum tension in the cables
        verbose (bool, optional): Whether to print info about the optimization problem after solving. Defaults to True
    """

    def __init__(
        self,
        robot_pos: np.ndarray,
        robot_orn: np.ndarray,
        local_cable_positions: np.ndarray,
        end_points: np.ndarray,
        local_cone_normals: np.ndarray,
        cone_angle: float,
        applied_wrench: np.ndarray,
        max_tension: float,
        verbose: bool = True,
    ):
        self.robot_pos = np.ravel(robot_pos)
        self.start_points = get_world_frame_cable_start_points(
            robot_pos, robot_orn, local_cable_positions
        )
        self.end_points = np.atleast_2d(end_points)
        R_B2W = quat_to_rmat(robot_orn)
        self.cone_normals = np.atleast_2d(local_cone_normals) @ R_B2W.T
        self.cone_angle = cone_angle
        self.max_tension = max_tension
        self.verbose = verbose
        self.n_cables = self.start_points.shape[0]
        self.n_sites = self.end_points.shape[0]

        # Parameterize
        self.applied_wrench = cp.Parameter(6)
        # Assign (and validate) initial values for the parameters
        self.update_wrench(applied_wrench)

        # Determine all possible directions of the cables
        self.directions = determine_directions(self.start_points, self.end_points)

        # Set up variables
        # Cables-to-sites assignments (1 if assigned, 0 if not)
        self.assignments = cp.Variable((self.n_cables, self.n_sites), boolean=True)
        # Tensions for each possible cable assignment, for each basis direction
        # Index via tensions[cable, site]
        self.tensions = cp.Variable((self.n_cables, self.n_sites))

        # Construct the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    @property
    def objective(self) -> Union[cp.Minimize, cp.Maximize]:
        """Optimization objective"""
        cable_dps = []
        for cable_idx in range(self.n_cables):
            for site_idx in range(self.n_sites):
                cable_dps.append(
                    self.assignments[cable_idx, site_idx]
                    * self.directions[cable_idx, site_idx]
                    @ self.cone_normals[cable_idx]
                )
        return cp.Maximize(cp.sum(cable_dps))

    @property
    def constraints(self) -> list[cp.Expression]:
        """Optimization constraints"""
        cons = []
        # Each cable must be assigned to one grasp point
        cons.append(cp.sum(self.assignments, axis=1) == 1)
        # Each grasp point must be assigned to at most one cable
        cons.append(cp.sum(self.assignments, axis=0) <= 1)
        cons.append(cp.sum(self.assignments, axis=0) >= 0)  # Is this needed??
        # Adhere to absolute tension limits
        cons.extend([ti <= self.max_tension for ti in self.tensions])
        cons.extend([ti >= 0 for ti in self.tensions])

        # We must be able to apply the desired wrench
        wrench_components = []
        for cable_idx in range(self.n_cables):
            for site_idx in range(self.n_sites):
                r = self.start_points[cable_idx] - self.robot_pos
                force = (
                    self.tensions[cable_idx, site_idx]
                    * self.directions[cable_idx, site_idx]
                )
                wrench_components.append(cp.hstack([force, skew(r) @ force]))
        cons.append(cp.sum(wrench_components) == self.applied_wrench)

        # Limit the maximum tension based on the cable assignment
        for cable_idx in range(self.n_cables):
            for site_idx in range(self.n_sites):
                cons.append(
                    self.tensions[cable_idx, site_idx]
                    <= self.max_tension * self.assignments[cable_idx, site_idx]
                )

        # Direction of the cable lies within allowable cone
        cos_theta = np.cos(self.cone_angle)
        for cable_idx in range(self.n_cables):
            for site_idx in range(self.n_sites):
                cons.append(
                    self.directions[cable_idx, site_idx]
                    @ self.cone_normals[cable_idx]
                    * self.assignments[cable_idx, site_idx]
                    >= cos_theta * self.assignments[cable_idx, site_idx]
                )
        return cons

    def solve(self) -> None:
        """Solve the optimization problem"""
        self.prob.solve(solver=cp.MOSEK)
        if self.prob.status != cp.OPTIMAL:
            raise OptimizationError(f"Non-optimal status: {self.prob.status}")
        if self.verbose:
            print_problem_info(self.prob)

    # Post-solve optimal values

    @property
    def optimal_assignments(self) -> list[Optional[int]]:
        """Optimal grasp site indices for each cable. None if that cable is unassigned. Length = n_cables"""

        # Boolean array of assignments. A[cable, site] == 1 if assigned, 0 if not
        A = self.assignments.value
        if A is None:
            raise ValueError("Optimization problem has not been solved yet")
        n_cables = A.shape[0]
        assignments = []
        for cable_idx in range(n_cables):
            # Note: use np.isclose because while the array should be purely boolean, it's usually stored as floats, and
            # we want to avoid any small floating point issues
            site_ids = np.flatnonzero(np.isclose(A[cable_idx], 1))
            if np.size(site_ids) == 0:  # Unassigned
                assignments.append(None)
            elif np.size(site_ids) == 1:  # Assigned to one site
                assignments.append(site_ids[0])
            else:  # More than one site. Should be impossible due to constraints
                raise RuntimeError(
                    "Somehow, a cable has been assigned to >1 grasp sites"
                )
        return assignments

    @property
    def optimal_sites(self) -> list[Optional[np.ndarray]]:
        """Optimal grasp sites (each shape (3,)) for each cable. None if that cable is unassigned. Length = n_cables"""
        sites = []
        for idx in self.optimal_assignments:
            if idx is None:
                sites.append(None)
            else:
                sites.append(self.end_points[idx])
        return sites

    @property
    def optimal_directions(self) -> list[Optional[np.ndarray]]:
        """Optimal directions (each shape (3,)) for each cable. None if that cable is unassigned. Length = n_cables"""
        dirs = []
        sites = self.optimal_sites
        for i in range(self.n_cables):
            if sites[i] is None:
                dirs.append(None)
            else:
                dirs.append(normalize(sites[i] - self.start_points[i]))
        return dirs

    # Parameter updates
    def update_wrench(self, new_wrench: np.ndarray) -> None:
        """Update the applied wrench parameter

        Args:
            new_wrench (np.ndarray): Applied wrench, shape (6,)
        """
        new_wrench = np.ravel(new_wrench)
        if new_wrench.shape != self.applied_wrench.shape:
            raise ValueError(
                "Wrench shape mismatch, cannot update the parameter\n"
                + f"Expected shape {self.applied_wrench.shape}, got {new_wrench.shape}"
            )
        self.applied_wrench.value = new_wrench


def main():
    # pylint: disable=import-outside-toplevel
    from reachbot_manipulation.core.reachbot import Reachbot
    from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
    from reachbot_manipulation.geometry.wrench_spaces import (
        cable_wrench_space,
        visualize_wrench_space,
    )
    from reachbot_manipulation.config.reachbot_config import (
        LOCAL_CABLE_POSITIONS,
        LOCAL_CONE_NORMALS,
    )
    from reachbot_manipulation.utils.debug_visualizer import visualize_points

    np.random.seed(0)

    # Set up problem parameters
    max_tension = 30
    n_sites = 40
    end_pts = 5 * normalize(np.random.randn(n_sites, 3))
    cone_angle = np.pi / 2
    applied_wrench = np.zeros(6)
    robot_pos = np.zeros(3)
    robot_orn = np.array([0, 0, 0, 1])
    world_frame_start_pts = get_world_frame_cable_start_points(
        robot_pos, robot_orn, LOCAL_CABLE_POSITIONS
    )
    # Solve
    print("Constructing the problem...")
    prob = NaivePlanner(
        robot_pos,
        robot_orn,
        LOCAL_CABLE_POSITIONS,
        end_pts,
        LOCAL_CONE_NORMALS,
        cone_angle,
        applied_wrench,
        max_tension,
    )
    print("Solving...")
    prob.solve()
    assignments = prob.optimal_assignments
    optimal_grasp_sites = prob.optimal_sites
    optimal_normals = prob.optimal_directions
    offsets = world_frame_start_pts - robot_pos
    wrench_space = cable_wrench_space(offsets, optimal_normals, max_tension, l_inf=True)
    print("Assignments: ", assignments)

    client = initialize_pybullet(
        gravity=0, bg_color=(1, 1, 1)
    )  # TODO make Mars grav?? or include in wrench
    visualize_points(end_pts, (1, 0, 0))
    robot = Reachbot(client=client)
    # TODO figure out the tensions later
    robot.attach_to(optimal_grasp_sites, np.ones(robot.NUM_CABLES))
    robot.step()
    visualize_wrench_space(wrench_space, desired_wrench=applied_wrench)
    input()


if __name__ == "__main__":
    main()
