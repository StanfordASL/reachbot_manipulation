"""Showing that using the stance planner, we retain control over orientation at the nominal pose

This would not be feasible with the naive planner, since any change in orientation would lead to a configuration
where static equilibrium is not possible with a cable-driven robot
"""

from datetime import datetime

import numpy as np

from reachbot_manipulation.core.env import Environment, EnvConfig
from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
from reachbot_manipulation.core.reachbot import Reachbot
from reachbot_manipulation.optimization.tension_planner import TensionPlanner
from reachbot_manipulation.utils.rotations import Rx, Ry, Rz, rmat_to_quat
from reachbot_manipulation.trajectories.traj_planner import plan_traj
from reachbot_manipulation.utils.errors import OptimizationError
from reachbot_manipulation.optimization.stance_planner import StancePlanner
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_POSITIONS,
    LOCAL_CONE_NORMALS,
    MAX_TENSION,
)

RECOMPUTE: bool = False
PRECOMPUTED_ASSIGNMENTS: list[int] = [24, 7, 25, 4, 8, 11, 23, 20]
USE_OLD_METHOD: bool = True
RECORD: bool = False
SHOW_ENV: bool = False


def main():
    # Generate an environment
    rng = np.random.default_rng(0)
    env_config = EnvConfig(rng=rng, n_sites=30)
    env = Environment.from_config(env_config)

    # Define nominal problem parameters
    pose = np.array([0, 0, 0, 0, 0, 0, 1])
    pos = pose[:3]
    orn = pose[3:]
    robot_mass = 10
    g = 3.71
    wrench = np.array([0, 0, robot_mass * g, 0, 0, 0])  # Static equilib
    end_pts = np.array([site.position for site in env.sites])
    orn_uncert = np.pi / 12
    task_basis = np.vstack([np.eye(6), -np.eye(6)])  # +- standard basis
    basis_weights = np.ones(12)  # Unweighted (ball metric)
    site_weights = np.ones(env.num_sites)  # Assumes all sites are good
    cone_angle = np.pi / 2

    # Solve the problem
    if RECOMPUTE:
        print("Constructing and solving the problem")
        prob = StancePlanner(
            pose,
            LOCAL_CABLE_POSITIONS,
            end_pts,
            LOCAL_CONE_NORMALS,
            cone_angle,
            wrench,
            MAX_TENSION,
            task_basis,
            basis_weights,
            site_weights,
            verbose=False,
        )
        prob.solve()
        assignments = prob.optimal_assignments
        print("Assignments: ", assignments)
    else:
        assignments = PRECOMPUTED_ASSIGNMENTS
    attached_points = np.array([env.sites[i].position for i in assignments])

    # Solve the nominal tensions for this stance
    tension_prob = TensionPlanner(
        pos,
        orn,
        LOCAL_CABLE_POSITIONS,
        attached_points,
        wrench,
        MAX_TENSION,
        verbose=False,
    )
    tension_prob.solve()
    tensions = tension_prob.optimal_tensions

    new_orns = [
        rmat_to_quat(Rx(np.deg2rad(10))),
        rmat_to_quat(Rx(np.deg2rad(-10))),
        [0, 0, 0, 1],
        rmat_to_quat(Ry(np.deg2rad(-15))),
        rmat_to_quat(Ry(np.deg2rad(15))),
        [0, 0, 0, 1],
        rmat_to_quat(Rz(np.deg2rad(15))),
        rmat_to_quat(Rz(np.deg2rad(-15))),
        [0, 0, 0, 1],
    ]
    new_tensions = []
    for i, new_orn in enumerate(new_orns):
        tension_prob.update_robot_pose(pos, new_orn)
        try:
            tension_prob.solve()
            new_tensions.append(tension_prob.optimal_tensions)
        except OptimizationError as e:
            print(f"Orientation {i} infeasible")

    all_orns = [orn, *new_orns]
    all_tensions = [tensions, *new_tensions]
    traj_components = []
    dt = 1 / 240
    # Create trajectory to the perturbed orientations
    for i in range(len(all_orns) - 1):
        # HACK
        traj_components.append(
            plan_traj(
                pos,
                all_orns[i],
                pos,
                all_orns[i + 1],
                dt,
                attached_points,
                all_tensions[i],
                all_tensions[i + 1],
            )
        )

    # Simulate
    client = initialize_pybullet(gravity=-g, bg_color=(0.5, 0.5, 0.5))
    client.resetDebugVisualizerCamera(3.20, 23.60, -13, (0, 0, 0))
    if SHOW_ENV:
        env_id, sites_id = env.visualize()
    robot = Reachbot(pose, client=client, kv=25, kp=25, kw=10)
    robot.attach_to(attached_points, tensions)
    robot.step()

    if RECORD:
        input("Press Enter to begin recording when ready")
        filename = (
            f"artifacts/reorientation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
        )
        log_id = client.startStateLogging(client.STATE_LOGGING_VIDEO_MP4, filename)

    # Add a few frames before trying new orientations
    for _ in range(100):
        robot.step()
    # input("Press Enter to try new orns")
    for j, traj in enumerate(traj_components):
        for i in range(traj.num_timesteps):
            robot.set_tensions(traj.tensions[i])
            robot.set_nominal_lengths(traj.cable_lengths[i])
            robot.set_nominal_speeds(traj.cable_speeds[i])
            robot.step()
        # Add a few bonus frames when stopped
        for i in range(30):
            robot.step()
        # input("Press Enter to continue")

    if RECORD:
        client.stopStateLogging(log_id)
    input("Press Enter to exit")
    client.disconnect()


if __name__ == "__main__":
    main()
