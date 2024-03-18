"""Demonstrating an example pick and place task"""

from datetime import datetime

import numpy as np

from reachbot_manipulation.core.env import Environment, EnvConfig
from reachbot_manipulation.utils.bullet_utils import initialize_pybullet
from reachbot_manipulation.optimization.stance_planner import StancePlanner
from reachbot_manipulation.optimization.tension_planner import TensionPlanner
from reachbot_manipulation.core.reachbot import Reachbot
from reachbot_manipulation.trajectories.traj_planner import plan_traj
from reachbot_manipulation.trajectories.trajectory import ReachbotTrajectory
from reachbot_manipulation.utils.bullet_utils import load_floor, load_rigid_object
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_POSITIONS,
    LOCAL_CONE_NORMALS,
)

RECOMPUTE = False
PRECOMPUTED_ASSIGNMENTS = [10, 1, 13, 16, 2, 17, 5, 8]
RECORD = False
SHOW_ENV = True


def main():
    # Generate an environment
    rng = np.random.default_rng(0)
    env_config = EnvConfig(rng=rng)
    env = Environment.from_config(env_config)

    # Sizes
    gripper_len = 1
    robot_size = 1
    object_size = 0.5
    env_radius = 10

    # Masses
    robot_mass = 10
    object_mass = 1
    system_mass = robot_mass + object_mass

    # Constants
    g = 3.71

    # Trajectory definition
    x0 = -2
    xf = 2
    y = 0
    z_grasp = -env_radius + object_size + gripper_len + robot_size / 2
    z_lift = z_grasp + 1
    orn = np.array([0, 0, 0, 1])
    pick_pose = np.array([x0, y, z_grasp, *orn])
    above_pick_pose = np.array([x0, y, z_lift, *orn])
    above_place_pose = np.array([xf, y, z_lift, *orn])
    place_pose = np.array([xf, y, z_grasp, *orn])

    # Wrenches
    robot_static_eq_wrench = np.array([0, 0, robot_mass * g, 0, 0, 0])
    system_static_eq_wrench = np.array([0, 0, system_mass * g, 0, 0, 0])

    pose_wrench_pairs = [
        [above_pick_pose, robot_static_eq_wrench],
        [pick_pose, robot_static_eq_wrench],
        [pick_pose, system_static_eq_wrench],
        [above_pick_pose, system_static_eq_wrench],
        [above_place_pose, system_static_eq_wrench],
        [place_pose, system_static_eq_wrench],
        [place_pose, robot_static_eq_wrench],
        [above_place_pose, robot_static_eq_wrench],
    ]

    # Planner inputs
    # HACK: in theory there would be more pose/wrench pairs but this subset will work
    poses = np.array([above_pick_pose, pick_pose, above_place_pose, place_pose])
    wrenches = np.array(
        [
            system_static_eq_wrench,
            system_static_eq_wrench,
            system_static_eq_wrench,
            system_static_eq_wrench,
        ]
    )
    max_tension = 100  # Increase for ease of computation
    end_pts = np.array([site.position for site in env.sites])
    task_basis = np.vstack([np.eye(6), -np.eye(6)])  # +- standard basis
    basis_weights = np.ones(12)  # Unweighted (ball metric)
    site_weights = np.ones(env.num_sites)  # Assumes all sites are good
    cone_angle = np.pi / 2

    # Solve the problem
    if RECOMPUTE:
        print("Constructing and solving the problem")
        prob = StancePlanner(
            poses,
            LOCAL_CABLE_POSITIONS,
            end_pts,
            LOCAL_CONE_NORMALS,
            cone_angle,
            wrenches,
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
        assignments = PRECOMPUTED_ASSIGNMENTS
    attached_points = np.array([env.sites[i].position for i in assignments])

    # Init tension prob
    tension_prob = TensionPlanner(
        poses[0, :3],
        poses[0, 3:],
        LOCAL_CABLE_POSITIONS,
        attached_points,
        wrenches[0],
        max_tension,
        verbose=False,
    )
    tensions = []
    for pose, wrench in pose_wrench_pairs:
        tension_prob.update_robot_pose(pose[:3], pose[3:])
        tension_prob.update_applied_wrench(wrench)
        tension_prob.solve()
        tensions.append(tension_prob.optimal_tensions)
    tensions = np.array(tensions)

    traj_components = []
    dt = 1 / 240
    for i in range(len(pose_wrench_pairs) - 1):
        pose_1 = pose_wrench_pairs[i][0]
        pose_2 = pose_wrench_pairs[i + 1][0]
        if np.allclose(pose_1, pose_2):
            # Fixed position, just vary tensions
            # Give it 3 seconds to adjust the tensions in the cables
            duration = 3
            n_timesteps = int(round(duration / dt))
            times = np.arange(0, n_timesteps) * dt
            traj_components.append(
                ReachbotTrajectory(
                    positions=np.ones((n_timesteps, 1)) * pose_1[:3],
                    quats=np.ones((n_timesteps, 1)) * pose_1[3:],
                    lin_vels=np.zeros((n_timesteps, 3)),
                    ang_vels=np.zeros((n_timesteps, 3)),
                    lin_accels=np.zeros((n_timesteps, 3)),
                    ang_accels=np.zeros((n_timesteps, 3)),
                    times=times,
                    attached_points=attached_points,
                    init_tensions=tensions[i],
                    final_tensions=tensions[i + 1],
                )
            )
        else:
            traj_components.append(
                plan_traj(
                    pose_1[:3],
                    pose_1[3:],
                    pose_2[:3],
                    pose_2[3:],
                    dt,
                    attached_points,
                    tensions[i],
                    tensions[i + 1],
                )
            )

    # Simulate
    client = initialize_pybullet(gravity=-g, bg_color=(0.5, 0.5, 0.5))
    if SHOW_ENV:
        env_id, sites_id = env.visualize()
    client.resetDebugVisualizerCamera(6.40, 10.40, -4.60, (0.38, -1.18, -6.97))
    floor_id = load_floor(z_pos=-env_radius, client=client)
    client.changeVisualShape(
        floor_id,
        -1,
        rgbaColor=[1, 1, 1, 0],
    )
    object_id = load_rigid_object(
        "cube.urdf",
        scale=object_size,
        pos=(x0, 0, -env_radius + object_size / 2),
        mass=object_mass,
    )
    robot = Reachbot(pose_wrench_pairs[0][0], client=client, kp=100)
    robot.attach_to(attached_points, tensions[0], max_site_tensions=None)
    if RECORD:
        input("Press Enter to begin recording")
        log_id = client.startStateLogging(
            client.STATE_LOGGING_VIDEO_MP4,
            f"artifacts/manip_task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4",
        )
    cid = None
    for j, traj in enumerate(traj_components):
        for i in range(traj.num_timesteps):
            robot.set_tensions(traj.tensions[i])
            robot.set_nominal_lengths(traj.cable_lengths[i])
            robot.set_nominal_speeds(traj.cable_speeds[i])
            robot.step()
        if j == 0:
            cid = client.createConstraint(
                robot.id,
                0,
                object_id,
                -1,
                client.JOINT_FIXED,
                (0, 0, 0),
                (0, 0, -0.5),
                (0, 0, 0.25),
            )
        if j == len(traj_components) - 2:
            client.removeConstraint(cid)
    if RECORD:
        client.stopStateLogging(log_id)
    input("Press Enter to exit")
    client.disconnect()


if __name__ == "__main__":
    main()
