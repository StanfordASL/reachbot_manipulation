"""Planning trajectories for ReachBot"""

import numpy as np
import numpy.typing as npt

from reachbot_manipulation.trajectories.trajectory import ReachbotTrajectory
from reachbot_manipulation.trajectories.quaternion_interpolation import (
    quaternion_interpolation_with_bcs,
)
from reachbot_manipulation.utils.quaternions import (
    quats_to_angular_velocities,
    quaternion_angular_error,
)
from reachbot_manipulation.trajectories.curve_utils import traj_from_curve
from reachbot_manipulation.trajectories.variable_time_curves import (
    free_final_time_bezier,
)
from reachbot_manipulation.config.reachbot_config import (
    SPEED_LIMIT,
    ACCEL_LIMIT,
    ANGULAR_SPEED_LIMIT,
)


def plan_traj(
    p0: npt.ArrayLike,
    q0: npt.ArrayLike,
    pf: npt.ArrayLike,
    qf: npt.ArrayLike,
    dt: float,
    attached_points: npt.ArrayLike,
    init_tensions: npt.ArrayLike,
    final_tensions: npt.ArrayLike,
) -> ReachbotTrajectory:
    """Generate a simple point-to-point trajectory for ReachBot to follow

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        q0 (npt.ArrayLike): Initial XYZW quaternion, shape (4,)
        pf (npt.ArrayLike): Final position, shape (3,)
        qf (npt.ArrayLike): Final XYZW quaternion, shape (4,)
        dt (float): Sampling period (seconds)
        attached_points (npt.ArrayLike): Grasp site positions where the ReachBot cables are attached, shape (n_cables,)
        init_tensions (npt.ArrayLike): Static equilibrium tensions at the initial pose, shape (n_cables,)
        final_tensions (npt.ArrayLike): Static equilibrium tensions at the final pose, shape (n_cables,)

    Returns:
        ReachbotTrajectory: The optimal trajectory plan
    """

    # Assume start and stop from rest
    v0 = np.zeros(3)
    w0 = np.zeros(3)
    a0 = np.zeros(3)
    dw0 = np.zeros(3)
    vf = np.zeros(3)
    wf = np.zeros(3)
    af = np.zeros(3)
    dwf = np.zeros(3)

    fixed_position = np.allclose(p0, pf)
    if fixed_position:
        # Use a heuristic to determine the final time of the orientation trajectory component
        duration = rotation_duration_heuristic(q0, qf)
        n_timesteps = int(round(duration / dt))
        times = np.arange(0, n_timesteps) * dt
        positions = np.ones((n_timesteps, 1)) * p0
        lin_vels = np.zeros((n_timesteps, 3))
        lin_accels = np.zeros((n_timesteps, 3))
    else:
        # Min-jerk position traj
        n_control_pts = 20
        # Initial guess at the duration of the trajectory
        tf_init = 2 * np.linalg.norm(np.subtract(pf, p0)) / SPEED_LIMIT
        curve = free_final_time_bezier(
            p0,
            pf,
            0,
            tf_init,
            n_control_pts,
            v0,
            vf,
            a0,
            af,
            None,
            SPEED_LIMIT,
            ACCEL_LIMIT,
        )
        duration = curve.b
        pos_traj = traj_from_curve(curve, dt)
        times = pos_traj.times
        n_timesteps = len(times)
        positions = pos_traj.positions
        lin_vels = pos_traj.linear_velocities
        lin_accels = pos_traj.linear_accels
    quats = quaternion_interpolation_with_bcs(
        q0, qf, w0, wf, dw0, dwf, duration, n_timesteps
    )
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return ReachbotTrajectory(
        positions=positions,
        quats=quats,
        lin_vels=lin_vels,
        ang_vels=omega,
        lin_accels=lin_accels,
        ang_accels=alpha,
        times=times,
        attached_points=attached_points,
        init_tensions=init_tensions,
        final_tensions=final_tensions,
    )


def rotation_duration_heuristic(q0: npt.ArrayLike, qf: npt.ArrayLike) -> float:
    """Calculate an estimate of how long a rotation will take

    Args:
        q0 (npt.ArrayLike): Initial XYZW quaternion, shape (4,)
        qf (npt.ArrayLike): Final XYZW quaternion, shape (4,)

    Returns:
        float: Time estimate, seconds
    """
    err = quaternion_angular_error(q0, qf)
    err_mag = np.linalg.norm(err)
    return err_mag / (0.5 * ANGULAR_SPEED_LIMIT)
