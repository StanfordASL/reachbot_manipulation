"""An example of ReachBot connected to an optimal set of grasp points in the environment"""

import numpy as np
from reachbot_manipulation.core.env import Environment, EnvConfig
from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
from reachbot_manipulation.optimization.stance_planner import StancePlanner
from reachbot_manipulation.optimization.tension_planner import TensionPlanner
from reachbot_manipulation.core.reachbot import Reachbot
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_POSITIONS,
    LOCAL_CONE_NORMALS,
)

RECOMPUTE = False
PRECOMPUTED_ASSIGNMENTS = [4, 9, 12, 7, 15, 11, 14, 0]


def main():
    # Generate an environment
    rng = np.random.default_rng(0)
    env_config = EnvConfig(rng=rng)
    env = Environment.from_config(env_config)

    # Set up problem parameters
    max_tension = 30
    end_pts = np.array([site.position for site in env.sites])
    cone_angle = np.pi / 2
    mass = 10
    g = 3.71
    applied_wrench = np.array([0, 0, mass * g, 0, 0, 0])
    robot_pos = np.zeros(3)
    robot_orn = np.array([0, 0, 0, 1])
    pose = np.concatenate([robot_pos, robot_orn])
    task_basis = np.vstack([np.eye(6), -np.eye(6)])  # +- standard basis
    basis_weights = np.ones(12)  # Unweighted (ball metric)
    site_weights = np.ones(env.num_sites)  # Assumes all sites are good

    # Use the stance planner to solve for the optimal places to attach the ReachBot booms
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
        print("Assignments: ", assignments)
    else:
        assignments = PRECOMPUTED_ASSIGNMENTS
    attached_points = np.array([env.sites[i].position for i in assignments])

    # Determine the optimal tensions for the ReachBot stance
    tension_prob = TensionPlanner(
        robot_pos,
        robot_orn,
        LOCAL_CABLE_POSITIONS,
        attached_points,
        applied_wrench,
        max_tension,
    )
    tension_prob.solve()
    boom_tensions = tension_prob.optimal_tensions

    # Visualize/simulate in Pybullet
    client = initialize_pybullet(gravity=-g, bg_color=(1, 1, 1))
    env.visualize(client=client)
    robot = Reachbot(pose, client=client)
    robot.attach_to(attached_points, boom_tensions)
    robot.step()
    input("Press Enter to begin the interactive simulation")
    while True:
        robot.step()


if __name__ == "__main__":
    main()
