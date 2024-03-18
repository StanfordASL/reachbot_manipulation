"""Mixed-integer optimization of the ReachBot booms/cables over a discrete set of grasp sites.

We can also optimize for multiple wrenches applied at different locations in the environment

For instance, in a pick-and-place task, we have 4 wrenches at 2 different poses:
- A static equilibrium wrench applied just prior to picking the object, at pose 1
- A holding wrench applied just after picking the object, at pose 1
- A static equilibrium wrench applied just prior to placing the object, at pose 2
- A static equilibroum wrench applied just after placing the object, at pose 2

And in any general task, we have >=2 wrenches at >=1 location. In the most simple scenario, we have:
- A static equilibrium wrench prior to the task
- A task wrench during the task
"""

from typing import Optional, Union

import numpy as np
import cvxpy as cp

from reachbot_manipulation.utils.cvxpy_math import skew
from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.cvxpy_utils import print_problem_info
from reachbot_manipulation.utils.errors import OptimizationError
from reachbot_manipulation.utils.rotations import quat_to_rmat
from reachbot_manipulation.utils.transformations import (
    make_transform_mat,
    transform_points,
)


class StancePlanner:
    """Determine the configuration of cables that can apply a set of wrenches at multiple locations, while maximizing a
    Ferrari-Canny-like metric that approximately reflects the minimum resisted wrench

    This problem is DPP-parameterized so that it can be efficiently resolved for different task definitions (basis +
    weightings) and grasp site qualities

    Args:
        robot_poses (Union[np.ndarray, list[np.ndarray]]): Position and XYZW quaternion orientations of the robot,
            shape (n_poses, 7)
        local_cable_positions (np.ndarray): Locations of where the cables originate from the robot, in the robot's
            body frame. Shape (n_cables, 3)
        end_points (np.ndarray): Possible locations where the cables can be attached to the world (world frame),
            shape (n_grasp_sites, 3)
        local_cone_normals (np.ndarray): Nominal normal vectors for each cable, defining the central axes of the
            second-order cones, in the robot frame. Shape (n_cables, 3)
        cone_angle (float): Interior angle of the second-order cones, defining the allowable directions of the cables
        applied_wrenches (Union[np.ndarray, list[np.ndarray]]): Wrenches to apply with the robot at each pose, shape
            (n_wrenches, 6,). Note n_wrenches == n_poses
        max_tension (float): Maximum tension in the cables
        task_basis (Optional[np.ndarray]): Task basis vectors in wrench space. These define the directions of interest
            for the wrench space, and can be specified to align with a desired task ellipsoid, for instance. Should be
            an orthonormal basis of 12 vectors in R6 (i.e. shape (12, 6)), since we include the positive and negative
            directions. Defaults to None (Use the standard +- unit basis)
        task_stdevs (Optional[np.ndarray]): Standard deviations of the task wrench distribution along the task basis
            vectors. A high standard deviation will increase the size of the wrench space in the corresponding
            direction. Defaults to None (uniform weighting)
        site_weights (Optional[np.ndarray]): Expected quality of each grasp site, a value between 0 and 1 dictating
            the fraction of the max_tension that we expect to be able to achieve at the site. Shape (n_grasp_sites,).
            Defaults to None (uniform weighting of 1)
        verbose (bool, optional): Whether to print info about the optimization problem after solving. Defaults to True
    """

    def __init__(
        self,
        robot_poses: Union[np.ndarray, list[np.ndarray]],
        local_cable_positions: np.ndarray,
        end_points: np.ndarray,
        local_cone_normals: np.ndarray,
        cone_angle: float,
        applied_wrenches: Union[np.ndarray, list[np.ndarray]],
        max_tension: float,
        task_basis: Optional[np.ndarray] = None,
        task_stdevs: Optional[np.ndarray] = None,
        site_weights: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        self.robot_poses = np.atleast_2d(robot_poses)
        assert self.robot_poses.shape[1] == 7
        self.n_poses = self.robot_poses.shape[0]
        self.applied_wrenches = np.atleast_2d(applied_wrenches)
        assert self.applied_wrenches.shape[1] == 6
        self.n_wrenches = self.applied_wrenches.shape[0]
        if self.n_wrenches != self.n_poses:
            raise ValueError(
                "The number of applied wrenches must match the number of poses. If the wrenches are "
                + "applied at the same pose, include that pose multiple times for each wrench"
            )

        rmats = [quat_to_rmat(self.robot_poses[i][3:]) for i in range(self.n_poses)]
        self.cone_normals = np.array([local_cone_normals @ R.T for R in rmats])
        self.start_points = np.array(
            [
                get_world_frame_cable_start_points(
                    self.robot_poses[i, :3],
                    self.robot_poses[i, 3:],
                    local_cable_positions,
                )
                for i in range(self.n_poses)
            ]
        )
        self.end_points = np.atleast_2d(end_points)
        self.cone_angle = cone_angle
        self.max_tension = max_tension
        self.verbose = verbose
        self.n_cables = local_cable_positions.shape[0]
        self.n_sites = self.end_points.shape[0]

        # Assign default values to unassigned inputs
        if task_basis is None:
            task_basis = np.vstack([np.eye(6), -np.eye(6)])  # +/- standard basis
        self.n_basis_vectors = task_basis.shape[0]  # == 12
        if task_stdevs is None:
            task_stdevs = np.ones(self.n_basis_vectors)  # Even weighting
        if site_weights is None:
            site_weights = np.ones(self.n_sites)  # All good locations

        # Parameterize
        self.basis = cp.Parameter((self.n_basis_vectors, 6))
        self.basis_weights = cp.Parameter(self.n_basis_vectors)
        self.site_weights = cp.Parameter(self.n_sites)
        # Assign (and validate) initial values for the parameters
        self.update_basis(task_basis)
        self.update_task_stdevs(task_stdevs)  # Updates basis weights
        self.update_site_weights(site_weights)

        # Determine all possible directions of the cables
        self.directions = np.array(
            [
                determine_directions(self.start_points[i], self.end_points)
                for i in range(self.n_poses)
            ]
        )

        # Set up variables
        # Cables-to-sites assignments (1 if assigned, 0 if not)
        self.assignments = cp.Variable((self.n_cables, self.n_sites), boolean=True)
        # Magnitudes of achievable wrenches in each basis direction
        self.task_scales = cp.Variable((self.n_wrenches, self.n_basis_vectors))
        # Tensions for each possible cable assignment, for each basis direction
        # Index via tensions[wrench][basis][cable, site]
        self.tensions = [
            [
                cp.Variable((self.n_cables, self.n_sites))
                for i in range(self.n_basis_vectors)
            ]
            for j in range(self.n_wrenches)
        ]

        # Construct the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    @property
    def objective(self) -> Union[cp.Minimize, cp.Maximize]:
        """Optimization objective"""
        least_resisted_task_wrench = cp.min(
            cp.hstack(
                [
                    cp.multiply(self.basis_weights, self.task_scales[i])
                    for i in range(self.n_wrenches)
                ]
            )
        )
        return cp.Maximize(least_resisted_task_wrench)

    @property
    def constraints(self) -> list[cp.Expression]:
        """Optimization constraints"""
        cons = []
        # Basis vector scalings must be positive
        cons.append(self.task_scales >= 0.0)
        # Each cable must be assigned to one grasp point
        cons.append(cp.sum(self.assignments, axis=1) == 1)
        # Each grasp point must be assigned to at most one cable
        cons.append(cp.sum(self.assignments, axis=0) <= 1)
        # Adhere to absolute tension limits
        for wrench_idx in range(self.n_wrenches):
            for basis_idx in range(self.n_basis_vectors):
                cons.append(self.tensions[wrench_idx][basis_idx] <= self.max_tension)
                cons.append(self.tensions[wrench_idx][basis_idx] >= 0)

        # Determine the wrenches along each basis direction
        for wrench_idx in range(self.n_wrenches):
            for basis_idx in range(self.n_basis_vectors):
                wrench_components = []
                for cable_idx in range(self.n_cables):
                    for site_idx in range(self.n_sites):
                        r = (
                            self.start_points[wrench_idx][cable_idx]
                            - self.robot_poses[wrench_idx][:3]
                        )
                        force = (
                            self.tensions[wrench_idx][basis_idx][cable_idx, site_idx]
                            * self.directions[wrench_idx][cable_idx][site_idx]
                        )
                        wrench_components.append(cp.hstack([force, skew(r) @ force]))
                cons.append(
                    cp.sum(wrench_components)
                    == self.applied_wrenches[wrench_idx]
                    + self.task_scales[wrench_idx][basis_idx] * self.basis[basis_idx]
                )

        # Limit the maximum tension based on the cable assignment and the expected grasp quality for the site
        for wrench_idx in range(self.n_wrenches):
            for basis_idx in range(self.n_basis_vectors):
                for cable_idx in range(self.n_cables):
                    for site_idx in range(self.n_sites):
                        cons.append(
                            self.tensions[wrench_idx][basis_idx][cable_idx, site_idx]
                            <= self.max_tension
                            * self.assignments[cable_idx, site_idx]
                            * self.site_weights[site_idx]
                        )

        # Direction of the cable lies within allowable cone
        cos_theta = np.cos(self.cone_angle)
        for wrench_idx in range(self.n_wrenches):
            for cable_idx in range(self.n_cables):
                for site_idx in range(self.n_sites):
                    cons.append(
                        self.directions[wrench_idx][cable_idx][site_idx]
                        @ self.cone_normals[wrench_idx][cable_idx]
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
    def optimal_directions(self) -> list[list[Optional[np.ndarray]]]:
        """Optimal directions (each shape (3,)) for each cable, for each pose. None if that cable is unassigned.
        "Shape" == (n_poses, n_cables, 3)"""
        sites = self.optimal_sites
        all_dirs = []
        for pose_idx in range(self.n_poses):
            dirs = []
            for cable_idx in range(self.n_cables):
                if sites[cable_idx] is None:
                    dirs.append(None)
                else:
                    dirs.append(
                        normalize(
                            sites[cable_idx] - self.start_points[pose_idx][cable_idx]
                        )
                    )
            all_dirs.append(dirs)
        return all_dirs

    # Parameter updates

    def update_basis(self, new_basis: np.ndarray) -> None:
        """Update the task basis directions parameter

        Args:
            new_basis (np.ndarray): Normalized task basis vectors in wrench space, shape (12, 6)
        """
        new_basis = np.asarray(new_basis)
        if new_basis.shape != self.basis.shape:
            raise ValueError(
                "Basis shape mismatch, cannot update the parameter\n"
                + f"Expected shape {self.basis.shape}, got {new_basis.shape}"
            )
        if not np.allclose(
            np.linalg.norm(new_basis, axis=-1), np.ones(self.n_basis_vectors)
        ):
            raise ValueError("Invalid basis: Must be normalized")
        self.basis.value = new_basis

    def update_task_stdevs(self, new_stdevs: np.ndarray) -> None:
        """Update the task wrench standard deviations

        This updates the basis_weights parameter, which is inversely proportional to the spread (i.e. standard
        deviation) of the task.

        Args:
            new_stdevs (np.ndarray): Standard deviations of the task wrench distribution along the task basis
                directions, shape (12,)
        """
        new_stdevs = np.ravel(new_stdevs)
        if np.any(new_stdevs <= 0):
            raise ValueError("Task standard deviations must be positive")
        # NOTE inversely-proportional relationship!
        new_weights = 1 / new_stdevs
        if new_weights.shape != self.basis_weights.shape:
            raise ValueError(
                "Basis weights shape mismatch, cannot update the parameter\n"
                + f"Expected shape {self.basis_weights.shape}, got {new_weights.shape}"
            )
        self.basis_weights.value = new_weights

    def update_site_weights(self, new_weights: np.ndarray) -> None:
        """Update the grasp site weighting parameter

        Args:
            new_weights (np.ndarray): Weights between (0, 1) representing a fraction of the max tension we expect to
                achieve from each grasp site. Shape (n_sites,)
        """
        new_weights = np.asarray(new_weights)
        if new_weights.shape != self.site_weights.shape:
            raise ValueError(
                "Site weights shape mismatch, cannot update the parameter\n"
                + f"Expected shape {self.site_weights.shape}, got {new_weights.shape}"
            )
        if np.any(new_weights > 1) or np.any(new_weights < 0):
            raise ValueError("Invalid site weights: All values must be between 0 and 1")
        self.site_weights.value = new_weights


# Helper functions


def determine_directions(
    start_points: np.ndarray, end_points: np.ndarray
) -> np.ndarray:
    """Determine all directions between two sets of points

    We can use this for preprocessing all of the possible grasp points into normalized cable directions

    Example: directions[start_point_index, end_point_index] = direction, shape (3,)

    Args:
        start_points (np.ndarray): Starting points (i.e. world-frame positions of the cable holes on the Reachbot body),
            shape (n_cables, 3)
        end_points (np.ndarray): Ending points (i.e. world-frame sites where the cables can attach to the cave),
            shape (n_sites, 3)

    Returns:
        np.ndarray: All directions, shape (n_cables, n_sites, 3)
    """
    # Handle inputs
    start_points = np.atleast_2d(start_points)
    end_points = np.atleast_2d(end_points)
    n_cables, dim = start_points.shape
    n_sites, dim_2 = end_points.shape
    assert dim_2 == dim
    # Create a grid of differences over a new axis, then normalize. Note shape: (n_cables, n_sites, 3)
    S = np.repeat(start_points.reshape(n_cables, 1, dim), n_sites, axis=1)
    E = np.repeat(end_points.reshape(1, n_sites, dim), n_cables, axis=0)
    return normalize(E - S)


def get_world_frame_cable_start_points(
    pos: np.ndarray, orn: np.ndarray, local_positions: np.ndarray
) -> np.ndarray:
    """Determine the world-frame locations where the cables originate, based on the robot's pose

    Args:
        pos (np.ndarray): Position of the robot, shape (3,)
        orn (np.ndarray): XYZW quaternion orientation of the robot, shape (4,)
        local_positions (np.ndarray): Locations where the cables originate from the robot, in the robot's local
            reference frame. Shape (n_cables, 3)

    Returns:
        np.ndarray: World-frame cable starting points, shape (n_cables, 3)
    """
    T_B2W = make_transform_mat(quat_to_rmat(orn), pos)  # Base to world
    return transform_points(T_B2W, local_positions)


def main():
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt
    from reachbot_manipulation.core.reachbot import Reachbot
    from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
    from reachbot_manipulation.geometry.wrench_spaces import (
        cable_wrench_space,
        visualize_wrench_space,
    )
    from reachbot_manipulation.config.reachbot_config import LOCAL_CABLE_POSITIONS
    from reachbot_manipulation.utils.debug_visualizer import visualize_points

    np.random.seed(0)

    # Set up problem parameters
    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [0.1, 0.2, 0.3, 0, 0, 0, 1]])
    n_poses = poses.shape[0]
    n_wrenches = n_poses
    applied_wrenches = 0.1 * np.random.rand(n_wrenches, 6)
    max_tension = 30
    n_sites = 10
    end_pts = 5 * normalize(np.random.randn(n_sites, 3))
    cone_angle = np.pi / 2
    world_frame_start_pts = np.array(
        [
            get_world_frame_cable_start_points(
                poses[i, :3], poses[i, 3:], LOCAL_CABLE_POSITIONS
            )
            for i in range(n_poses)
        ]
    )
    local_cone_normals = normalize(LOCAL_CABLE_POSITIONS)
    # Solve
    print("Constructing the problem...")
    prob = StancePlanner(
        poses,
        LOCAL_CABLE_POSITIONS,
        end_pts,
        local_cone_normals,
        cone_angle,
        applied_wrenches,
        max_tension,
        task_basis=None,
        task_stdevs=None,
        site_weights=None,
    )
    print("Solving...")
    prob.solve()
    assignments = prob.optimal_assignments
    optimal_grasp_sites = prob.optimal_sites
    optimal_normals = prob.optimal_directions
    offsets = [world_frame_start_pts[i] - poses[i][:3] for i in range(n_poses)]
    wrench_spaces = [
        cable_wrench_space(offsets[i], optimal_normals[i], max_tension, l_inf=True)
        for i in range(n_poses)
    ]
    print("Assignments: ", assignments)

    client = initialize_pybullet(gravity=0, bg_color=(1, 1, 1))
    visualize_points(end_pts, (1, 0, 0))
    robot = Reachbot(client=client)
    # NOTE ignoring the tension optimization here, just visualizing directions
    robot.attach_to(optimal_grasp_sites, np.ones(robot.NUM_CABLES))
    robot.step()
    for i in range(n_poses):
        visualize_wrench_space(
            wrench_spaces[i], desired_wrench=applied_wrenches[i], show=False
        )
    plt.show()
    input()


if __name__ == "__main__":
    main()
