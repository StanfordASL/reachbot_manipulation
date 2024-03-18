"""Relationships between quaternions and angular velocity

Our quaternion / angular velocity conventions (currently) are:
- XYZW quaternions
- World-frame angular velocities

The matrices used by other sources differ because they might be using WXYZ quaternions or
they might define angular velocity in the body-fixed frame. For instance, Shuster uses XYZW
quaternions but defines angular velocity in body frame, and Khatib uses WXYZ quaternions
with angular velocities in world frame

In case we decide to use body-frame angular velocities in the future, the relevant equations
are included below (with the global frame equations as well for reference)
"""

from typing import Union

import numpy as np
import numpy.typing as npt


def world_frame_quat_deriv(q: npt.ArrayLike, omega_world: npt.ArrayLike) -> np.ndarray:
    """Quaternion derivative for a rotating body with a known WORLD-FRAME angular velocity

    Args:
        q (npt.ArrayLike): Current XYZW quaternion, shape (4,)
        omega_world (npt.ArrayLike): World-frame angular velocity (wx, wy, wz), shape (3,)

    Returns:
        np.ndarray: Quaternion derivative, shape (4,)
    """
    x, y, z, w = q
    GT = np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])
    return (1 / 2) * GT @ omega_world


def body_frame_quat_deriv(q: npt.ArrayLike, omega_body: npt.ArrayLike) -> np.ndarray:
    """Quaternion derivative for a rotating body with a known BODY-FRAME angular velocity

    Args:
        q (npt.ArrayLike): Current XYZW quaternion, shape (4,)
        omega_body (npt.ArrayLike): Body-frame angular velocity (w1, w2, w3), shape (3,)

    Returns:
        np.ndarray: Quaternion derivative, shape (4,)
    """
    x, y, z, w = q
    LT = np.array([[w, -z, y], [z, w, -x], [-y, x, w], [-x, -y, -z]])
    return (1 / 2) * LT @ omega_body


def world_frame_angular_error(q: npt.ArrayLike, q_des: npt.ArrayLike) -> np.ndarray:
    """Angular error vector between two orientations, defined in WORLD frame

    Args:
        q (npt.ArrayLike): Current XYZW quaternion, shape (4,)
        q_des (npt.ArrayLike): Desired XYZW quaternion, shape (4,)

    Returns:
        np.ndarray: Angular error, shape (3,)
    """
    x, y, z, w = q
    return 2 * np.array([[-w, z, -y, x], [-z, -w, x, y], [y, -x, -w, z]]) @ q_des


def body_frame_angular_error(q: npt.ArrayLike, q_des: npt.ArrayLike) -> np.ndarray:
    """Angular error vector between two orientations, defined in BODY frame

    Args:
        q (npt.ArrayLike): Current XYZW quaternion, shape (4,)
        q_des (npt.ArrayLike): Desired XYZW quaternion, shape (4,)

    Returns:
        np.ndarray: Angular error, shape (3,)
    """
    xd, yd, zd, wd = q_des
    return (
        2 * np.array([[wd, zd, -yd, -xd], [-zd, wd, xd, -yd], [yd, -xd, wd, -zd]]) @ q
    )


# Note about the angular velocities: The pytransform3d method seems more numerically stable, so I'm using that
# In some edge cases, this method below will result in a large erronrous "spike" in the angular velocity
# In reality this "spike" is somehow all of the angular velocities flipping briefly to all negative values
# Perhaps there is an ambiguity in the angular velocity vector and its negative at certain points (an artifact of
# quaternion double-cover?) Anyways, the pytransform3d method does not have these spikes


def body_frame_angular_velocities(
    quats: np.ndarray, dt: Union[float, npt.ArrayLike]
) -> np.ndarray:
    """Determines the BODY-frame angular velocities of a sequence of quaternions, for a given sampling time

    Args:
        quats (np.ndarray): Sequence of XYZW quaternions, shape (n, 4)
        dt (Union[float, np.ndarray]): Sampling time(s). If passing in an array of sampling times,
            this must be of length n

    Returns:
        np.ndarray: Body-frame angular velocities (w1, w2, w3), shape (n, 3)
    """
    xs = quats[:, 0]
    ys = quats[:, 1]
    zs = quats[:, 2]
    ws = quats[:, 3]
    n = quats.shape[0]  # Number of quaternions

    # If passing in an array if dts, check its shape first
    if np.ndim(dt) != 0 and len(dt) != n:
        raise ValueError(f"Invalid dt array length: {len(dt)}. Must be of length {n}")

    # This uses a new central differencing method to improve handling at start/end points
    dw = np.zeros((n, 3))
    # Handle the start
    dw[0, :] = np.array(
        [
            ws[0] * xs[1] - xs[0] * ws[1] - ys[0] * zs[1] + zs[0] * ys[1],
            ws[0] * ys[1] + xs[0] * zs[1] - ys[0] * ws[1] - zs[0] * xs[1],
            ws[0] * zs[1] - xs[0] * ys[1] + ys[0] * xs[1] - zs[0] * ws[1],
        ]
    )
    # Handle the end
    dw[-1, :] = np.array(
        [
            ws[-2] * xs[-1] - xs[-2] * ws[-1] - ys[-2] * zs[-1] + zs[-2] * ys[-1],
            ws[-2] * ys[-1] + xs[-2] * zs[-1] - ys[-2] * ws[-1] - zs[-2] * xs[-1],
            ws[-2] * zs[-1] - xs[-2] * ys[-1] + ys[-2] * xs[-1] - zs[-2] * ws[-1],
        ]
    )
    # Handle the middle range of quaternions
    # Multiply by a factor of 1/2 since the central difference covers 2 timesteps
    dw[1:-1, :] = (1 / 2) * np.column_stack(
        [
            ws[:-2] * xs[2:] - xs[:-2] * ws[2:] - ys[:-2] * zs[2:] + zs[:-2] * ys[2:],
            ws[:-2] * ys[2:] + xs[:-2] * zs[2:] - ys[:-2] * ws[2:] - zs[:-2] * xs[2:],
            ws[:-2] * zs[2:] - xs[:-2] * ys[2:] + ys[:-2] * xs[2:] - zs[:-2] * ws[2:],
        ]
    )
    # If dt is scalar, broadcasting is simple. If dt is an array of time deltas, adjust shape for broadcasting
    if np.ndim(dt) == 0:
        return 2.0 * dw / dt
    else:
        return 2.0 / (np.reshape(dt, (-1, 1)) * dw)


def world_frame_angular_velocities(
    quats: np.ndarray, dt: Union[float, npt.ArrayLike]
) -> np.ndarray:
    """Determines the WORLD-frame angular velocities of a sequence of quaternions, for a given sampling time

    Args:
        quats (np.ndarray): Sequence of XYZW quaternions, shape (n, 4)
        dt (Union[float, np.ndarray]): Sampling time(s). If passing in an array of sampling times,
            this must be of length n

    Returns:
        np.ndarray: World-frame angular velocities (w1, w2, w3), shape (n, 3)
    """
    xs = quats[:, 0]
    ys = quats[:, 1]
    zs = quats[:, 2]
    ws = quats[:, 3]
    n = quats.shape[0]  # Number of quaternions

    # If passing in an array if dts, check its shape first
    if np.ndim(dt) != 0 and len(dt) != n:
        raise ValueError(f"Invalid dt array length: {len(dt)}. Must be of length {n}")

    # This uses a new central differencing method to improve handling at start/end points
    dw = np.zeros((n, 3))
    # Handle the start

    dw[0, :] = np.array(
        [
            -ws[1] * xs[0] + xs[1] * ws[0] - ys[1] * zs[0] + zs[1] * ys[0],
            -ws[1] * ys[0] + xs[1] * zs[0] + ys[1] * ws[0] - zs[1] * xs[0],
            -ws[1] * zs[0] - xs[1] * ys[0] + ys[1] * xs[0] + zs[1] * ws[0],
        ]
    )
    # Handle the end
    dw[-1, :] = np.array(
        [
            -ws[-1] * xs[-2] + xs[-1] * ws[-2] - ys[-1] * zs[-2] + zs[-1] * ys[-2],
            -ws[-1] * ys[-2] + xs[-1] * zs[-2] + ys[-1] * ws[-2] - zs[-1] * xs[-2],
            -ws[-1] * zs[-2] - xs[-1] * ys[-2] + ys[-1] * xs[-2] + zs[-1] * ws[-2],
        ]
    )
    # Handle the middle range of quaternions
    # Multiply by a factor of 1/2 since the central difference covers 2 timesteps
    dw[1:-1, :] = (1 / 2) * np.column_stack(
        [
            -ws[2:] * xs[:-2] + xs[2:] * ws[:-2] - ys[2:] * zs[:-2] + zs[2:] * ys[:-2],
            -ws[2:] * ys[:-2] + xs[2:] * zs[:-2] + ys[2:] * ws[:-2] - zs[2:] * xs[:-2],
            -ws[2:] * zs[:-2] - xs[2:] * ys[:-2] + ys[2:] * xs[:-2] + zs[2:] * ws[:-2],
        ]
    )
    # If dt is scalar, broadcasting is simple. If dt is an array of time deltas, adjust shape for broadcasting
    if np.ndim(dt) == 0:
        return 2.0 * dw / dt
    else:
        return 2.0 / (np.reshape(dt, (-1, 1)) * dw)
