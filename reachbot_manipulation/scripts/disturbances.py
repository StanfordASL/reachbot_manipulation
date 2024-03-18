"""Show the effects of disturbances on a reachbot in a naive/optimal stance, using either cables or booms"""

from datetime import datetime

import numpy as np

from reachbot_manipulation.core.env import Environment, EnvConfig
from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
from reachbot_manipulation.core.reachbot import Reachbot
from reachbot_manipulation.optimization.tension_planner import TensionPlanner
from reachbot_manipulation.optimization.stance_planner import StancePlanner
from reachbot_manipulation.optimization.naive_planner import NaivePlanner
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_POSITIONS,
    LOCAL_CONE_NORMALS,
)

RECOMPUTE: bool = False
PRECOMPUTED_OPT_ASSIGNMENTS: list[int] = [4, 7, 19, 10, 5, 11, 3, 14]
PRECOMPUTED_NAIVE_ASSIGNMENTS: list[int] = [7, 1, 4, 10, 11, 17, 8, 2]
USE_NAIVE: bool = True
USE_BOOMS: bool = False
RECORD: bool = False
SHOW_ENV: bool = False


def main():
    # Generate an environment
    rng = np.random.default_rng(0)
    env_config = EnvConfig(rng=rng)
    env = Environment.from_config(env_config)

    # Define nominal problem parameters
    pose = np.array([0, 0, 0, 0, 0, 0, 1])
    pos = pose[:3]
    orn = pose[3:]
    robot_mass = 10
    g = 3.71
    wrench = np.array([0, 0, robot_mass * g, 0, 0, 0])  # Static equilib
    max_tension = 30
    end_pts = np.array([site.position for site in env.sites])
    orn_uncert = np.pi / 12
    task_basis = np.vstack([np.eye(6), -np.eye(6)])  # +- standard basis
    basis_weights = np.ones(12)  # Unweighted (ball metric)
    site_weights = np.ones(env.num_sites)  # Assumes all sites are good
    cone_angle = np.pi / 2  # For naive planner

    # Solve the problem
    if RECOMPUTE:
        print("Constructing and solving the problem")
        if USE_NAIVE:
            prob = NaivePlanner(
                pos,
                orn,
                LOCAL_CABLE_POSITIONS,
                end_pts,
                LOCAL_CONE_NORMALS,
                cone_angle,
                wrench,
                max_tension,
                verbose=False,
            )
        else:
            prob = StancePlanner(
                pose,
                LOCAL_CABLE_POSITIONS,
                end_pts,
                LOCAL_CONE_NORMALS,
                cone_angle,
                wrench,
                max_tension,
                task_basis,
                basis_weights,
                site_weights,
                verbose=False,
            )
        prob.solve()
        assignments = prob.optimal_assignments
        print("Assignments: ", assignments)
    else:
        if USE_NAIVE:
            assignments = PRECOMPUTED_NAIVE_ASSIGNMENTS
        else:
            assignments = PRECOMPUTED_OPT_ASSIGNMENTS
    attached_points = np.array([env.sites[i].position for i in assignments])

    # Solve the nominal tensions for this stance
    tension_prob = TensionPlanner(
        pos,
        orn,
        LOCAL_CABLE_POSITIONS,
        attached_points,
        wrench,
        max_tension,
        verbose=False,
    )
    tension_prob.solve()
    tensions = tension_prob.optimal_tensions

    # Simulate
    client = initialize_pybullet(gravity=-g, bg_color=(0.5, 0.5, 0.5))
    client.resetDebugVisualizerCamera(3.20, 23.60, -13, (0, 0, 0))
    if SHOW_ENV:
        env_id, sites_id = env.visualize()
    robot = Reachbot(pose, client=client, kv=25, kp=25, kw=10)
    robot.attach_to(attached_points, tensions)
    robot.step()

    if RECORD:
        filename = (
            "artifacts/disturbances"
            + ("_boom" if USE_BOOMS else "_cable")
            + ("_naive" if USE_NAIVE else "_optimal")
            + f"_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
        )
        input("Press Enter to begin recording when ready")
        log_id = client.startStateLogging(client.STATE_LOGGING_VIDEO_MP4, filename)

    # Add a few frames before the disturbance
    for _ in range(100):
        robot.step()

    # Apply disturbance
    for _ in range(20):
        client.applyExternalForce(
            robot.id, -1, (0, -50, 0), (0, 0, 0), client.WORLD_FRAME
        )
        client.applyExternalTorque(robot.id, -1, (0, 0, 100), client.WORLD_FRAME)
        robot.step()

    for i in range(1100):
        robot.step()

    if RECORD:
        client.stopStateLogging(log_id)
    input("Press Enter to exit")
    client.disconnect()


if __name__ == "__main__":
    main()
