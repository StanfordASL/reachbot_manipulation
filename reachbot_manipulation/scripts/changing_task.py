"""Visualizing the effect of changing the task standard deviation parameter in the grasp site optimization on the
optimal cable placement
"""

import numpy as np
from reachbot_manipulation.core.env import Environment, EnvConfig
from reachbot_manipulation.core.reachbot import Reachbot
from reachbot_manipulation.utils.debug_visualizer import visualize_lines
from reachbot_manipulation.utils.transformations import transform_points
from reachbot_manipulation.utils.poses import pos_quat_to_tmat
from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
from reachbot_manipulation.optimization.stance_planner import StancePlanner
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_POSITIONS,
    LOCAL_CONE_NORMALS,
)


RECOMPUTE = False
ASSIGNMENTS_WEIGHTED = [12, 26, 3, 13, 19, 18, 28, 29]
ASSIGNMENTS_UNWEIGHTED = [28, 22, 7, 1, 8, 2, 23, 29]


def main():
    rng = np.random.default_rng(0)
    env_config = EnvConfig(length=40, n_sites=30, rng=rng)
    env = Environment.from_config(env_config)

    # Set up problem parameters
    max_tension = 30
    end_pts = np.array([site.position for site in env.sites])
    cone_angle = np.pi / 2
    applied_wrench = np.array([0, 0, 0, 0, 0, 0])
    robot_pos = np.zeros(3)
    robot_orn = np.array([0, 0, 0, 1])
    pose = np.concatenate([robot_pos, robot_orn])

    # Solve the problem with a fully unweighted basis
    # +- standard basis
    task_basis = np.vstack([np.eye(6), -np.eye(6)])
    basis_weights = np.ones(12)
    site_weights = np.ones(env.num_sites)
    if RECOMPUTE:
        print("Constructing the problem...")
        prob = StancePlanner(
            pose,
            LOCAL_CABLE_POSITIONS,
            end_pts,
            LOCAL_CONE_NORMALS,
            cone_angle,
            applied_wrench,
            max_tension,
            task_basis,
            basis_weights,
            site_weights,
            verbose=False,
        )
        print("Solving...")
        prob.solve()
        assignments = prob.optimal_assignments
        print("Unweighted assignments: ", assignments)

        # Re-solve the problem with new weights / standard deviations
        new_task_stdevs = np.tile([1, 1, 3, 0.1, 0.1, 0.1], 2)
        prob.update_task_stdevs(new_task_stdevs)
        print("Re-solving...")
        prob.solve()
        new_assignments = prob.optimal_assignments
        print("Weighted assignments: ", new_assignments)
    else:
        assignments = ASSIGNMENTS_UNWEIGHTED
        new_assignments = ASSIGNMENTS_WEIGHTED

    start_points = transform_points(pos_quat_to_tmat(pose), LOCAL_CABLE_POSITIONS)
    end_points_unweighted = np.array([env.sites[i].position for i in assignments])
    end_points_weighted = np.array([env.sites[i].position for i in new_assignments])
    client = initialize_pybullet()
    robot = Reachbot(pose)
    env.visualize(client=client)
    lines_1 = visualize_lines(start_points, end_points_unweighted, (1, 0, 0))
    lines_2 = visualize_lines(start_points, end_points_weighted, (0, 0, 1))
    print("Red: unweighted")
    print("Blue: weighted towards the Z direction")
    input()


if __name__ == "__main__":
    main()
