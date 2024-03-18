"""Simulating a cable detachment and the response from the PD controller on the cables"""

import time

import numpy as np
from reachbot_manipulation.core.reachbot import Reachbot
from reachbot_manipulation.core.env import Environment, EnvConfig
from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
from reachbot_manipulation.optimization.tension_planner import TensionPlanner
from reachbot_manipulation.optimization.stance_planner import StancePlanner
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_POSITIONS,
    LOCAL_CONE_NORMALS,
    MAX_TENSION,
)


RECOMPUTE_ASSIGNMENTS = False
PRECOMPUTED_ASSIGNMENTS = [10, 7, 13, 8, 2, 17, 5, 16]


def main():
    rng = np.random.default_rng(1)
    env_config = EnvConfig(length=40, n_sites=20, rng=rng)
    env = Environment.from_config(env_config)

    # Set up problem parameters
    max_tension = 30
    end_pts = np.array([site.position for site in env.sites])
    cone_angle = np.pi / 2
    g = 3.71
    mass = 10
    applied_wrench = np.array([0, 0, mass * g, 0, 0, 0])
    robot_pos = np.zeros(3)
    robot_orn = np.array([0, 0, 0, 1])
    pose = np.concatenate([robot_pos, robot_orn])
    if RECOMPUTE_ASSIGNMENTS:
        # Solve
        print("Constructing the problem...")
        prob = StancePlanner(
            pose,
            LOCAL_CABLE_POSITIONS,
            end_pts,
            LOCAL_CONE_NORMALS,
            cone_angle,
            applied_wrench,
            max_tension,
            task_basis=None,
            task_stdevs=None,
            site_weights=None,
            verbose=False,
        )
        print("Solving...")
        prob.solve()
        assignments = prob.optimal_assignments
        optimal_grasp_sites = prob.optimal_sites
    else:
        assignments = PRECOMPUTED_ASSIGNMENTS
        optimal_grasp_sites = end_pts[assignments]
    print("Assignments: ", assignments)

    tension_prob = TensionPlanner(
        robot_pos,
        robot_orn,
        LOCAL_CABLE_POSITIONS,
        optimal_grasp_sites,
        applied_wrench,
        MAX_TENSION,
        verbose=False,
    )
    tension_prob.solve()
    tensions = tension_prob.optimal_tensions
    client = initialize_pybullet(gravity=-g, bg_color=(1, 1, 1))
    env.visualize(client=client)
    robot = Reachbot(pose, client=client)
    robot.attach_to(optimal_grasp_sites, tensions)
    robot.step()
    detach_idx = 0
    secs_to_detach = 3
    print(f"Detaching in {secs_to_detach} seconds...")
    start_time = time.time()
    while time.time() - start_time < secs_to_detach:
        robot.step()
    robot.detach_from(detach_idx)
    print("Detached")
    while True:
        robot.step()


if __name__ == "__main__":
    main()
