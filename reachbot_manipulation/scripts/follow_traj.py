"""Example of Reachbot following a simple point-to-point trajectory"""

import numpy as np

from reachbot_manipulation.trajectories.traj_planner import plan_traj
from reachbot_manipulation.core.env import EnvConfig, Environment
from reachbot_manipulation.core.reachbot import Reachbot
from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
from reachbot_manipulation.optimization.tension_planner import TensionPlanner
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_POSITIONS,
    MAX_TENSION,
)


def main():
    # Define environment and attached points in the env
    rng = np.random.default_rng(1)
    env_config = EnvConfig(length=40, n_sites=20, rng=rng)
    env = Environment.from_config(env_config)
    assignments = [10, 7, 13, 8, 2, 17, 5, 16]
    attached_points = np.array([env.sites[i].position for i in assignments])

    # Define trajectory parameters
    p0 = np.zeros(3)
    pf = np.array([5, 0, 0])
    q0 = np.array([0, 0, 0, 1])
    qf = np.array([0, 0, 0, 1])
    dt = 1 / 240

    # Determine the initial cable tensions to hold Reachbot at static eq at the beginning and end of the traj
    mass = 10
    gravity = 3.71
    applied_wrench = np.array([0, 0, mass * gravity, 0, 0, 0])
    tension_prob = TensionPlanner(
        p0, q0, LOCAL_CABLE_POSITIONS, attached_points, applied_wrench, MAX_TENSION
    )
    tension_prob.solve()
    init_tensions = tension_prob.optimal_tensions
    tension_prob.update_robot_pose(pf, qf)
    tension_prob.solve()
    final_tensions = tension_prob.optimal_tensions
    print("Initial tensions: ", init_tensions)
    print("Final tensions: ", final_tensions)

    # Plan trajectory
    traj = plan_traj(p0, q0, pf, qf, dt, attached_points, init_tensions, final_tensions)

    # Simulate
    client = initialize_pybullet(gravity=-gravity, bg_color=(0.5, 0.5, 0.5))
    robot = Reachbot(np.concatenate([p0, q0]), client=client)
    robot.attach_to(attached_points, init_tensions, max_site_tensions=None)
    for i in range(traj.num_timesteps):
        robot.set_tensions(traj.tensions[i])
        robot.set_nominal_lengths(traj.cable_lengths[i])
        robot.set_nominal_speeds(traj.cable_speeds[i])
        robot.step()
    input("Press Enter to exit")
    client.disconnect()


if __name__ == "__main__":
    main()
