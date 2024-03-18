"""Determining an orientation trajectory between two quaternions with angular velocity boundary conditions

This is different from SLERP because SLERP will find the shortest path along the unit 4D hypersphere
(by traveling along a great circle), but this implies that we cannot dictate the direction of the angular
velocity vector during this interpolation

This method will instead create a 5th order polynomial to specify the boundary conditions on angular velocity
(and technically angular acceleration as well, since we only need a 3rd order polynomial to specify BCs on 
the first derivatives). 

There is no guarantee that the quaternions will be normalized, but we can do that after the interpolation process. 
The interpolation does not result in significant deviations (< 5%) from unit-norm quaternions, so this should not
significantly affect the orientation representation

See "Orientation Planning in Task Space using Quaternion Polynomials" DOI: 10.1109/ROBIO.2017.8324769 for more info
Note: this paper uses WXYZ quaternions
"""

import numpy as np
import numpy.typing as npt

from reachbot_manipulation.utils.quaternions import (
    quats_to_angular_velocities,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def quaternion_interpolation_with_bcs(
    qi: npt.ArrayLike,
    qf: npt.ArrayLike,
    wi: npt.ArrayLike,
    wf: npt.ArrayLike,
    dwi: npt.ArrayLike,
    dwf: npt.ArrayLike,
    duration: float,
    n: int,
) -> np.ndarray:
    """Generate a sequence of quaternions between two orientations with angular velocity boundary conditions

    - This implementation is a slightly modified version of "Orientation Planning in Task Space using Quaternion
      Polynomials" DOI: 10.1109/ROBIO.2017.8324769, Algorithm 1

    Args:
        qi (npt.ArrayLike): Initial XYZW quaternion, shape (4,)
        qf (npt.ArrayLike): Final XYZW quaternion, shape (4,)
        wi (npt.ArrayLike): Initial inertial-frame angular velocity, shape (3,)
        wf (npt.ArrayLike): Final inertial-frame angular velocity, shape (3,)
        dwi (npt.ArrayLike): Initial inertial-frame angular acceleration, shape (3,)
        dwf (npt.ArrayLike): Final inertial-frame angular acceleration, shape (3,)
        duration (float): Trajectory duration, seconds
        n (int): Number of timesteps

    Returns:
        np.ndarray: Sequence of XYZW quaternions, shape (n, 4)
    """
    # NOTE The algorithm in the paper uses WXYZ quaternions, so we'll need to convert back and forth
    qi = xyzw_to_wxyz(qi)
    qf = xyzw_to_wxyz(qf)
    wi = np.asarray(wi)
    wf = np.asarray(wf)
    dwi = np.asarray(dwi)
    dwf = np.asarray(dwf)
    # Ensure shortest path interpolation
    if _dot(qi, qf) < 0:
        qf = -qf
    # First and second derivatives of quaternion magnitude should be 0
    # (in theory quaternion norms should always be fixed at 1, but this interpolation
    # does not necessarily guarantee this. But, it generally stays within 5% of norm 1)
    dNi, ddNi, dNf, ddNf = (0, 0, 0, 0)
    # Get the first and second derivatives of the quaternion at the starting/ending points
    dqi = _get_dq(wi, dNi, qi)
    ddqi = _get_ddq(wi, dwi, dNi, ddNi, qi)
    dqf = _get_dq(wf, dNf, qf)
    ddqf = _get_ddq(wf, dwf, dNf, ddNf, qf)
    # Get the polynomial coefficients
    p = _fifth_order_quat_poly_coeffs(qi, qf, dqi, dqf, ddqi, ddqf, duration)
    # Linearly sample along this polynomial to get the interpolated quaternions
    taus = np.linspace(0, 1, n, endpoint=True)
    wxyz_quats = _interpolate_along_quat_poly(p, taus)
    # Convert back to XYZW for compatibility with the rest of the repository
    quats = wxyz_to_xyzw(wxyz_quats)
    # Normalize, since this process does not guarantee norm 1 quaternions
    quats /= np.linalg.norm(quats, axis=1).reshape(-1, 1)
    return quats


# Below are all helper functions based on the equations from the reference paper
# These are prefixed by an underscore to imply that they should NOT be imported outside this file
# Mainly, this is because the paper deals with WXYZ quaternions and I didn't want to convert
# all of the math for the intermediate steps


def _fifth_order_quat_poly_coeffs(
    qi: np.ndarray,
    qf: np.ndarray,
    dqi: np.ndarray,
    dqf: np.ndarray,
    ddqi: np.ndarray,
    ddqf: np.ndarray,
    T: float,
) -> np.ndarray:
    """Get the coefficients for a fifth-order polynomial in quaternion space

    See Equation 13 in the reference paper
    """
    return np.row_stack(
        [
            qi,
            3 * qi + dqi * T,
            (ddqi * T**2 + 6 * dqi * T + 12 * qi) / 2,
            qf,
            3 * qf - dqf * T,
            (ddqf * T**2 - 6 * dqf * T + 12 * qf) / 2,
        ]
    )


def _interpolate_along_quat_poly(p: np.ndarray, taus: np.ndarray):
    """Interpolate along a polynomial defined by coefficients p at percents tau

    See Equation 12 in the reference paper
    """
    # Convert taus to a column vector for proper broadcasting
    taus = np.atleast_2d(taus)
    if taus.shape[0] == 1:
        taus = taus.T
    # Output will be (n, 4) where n is the number of taus to interpolate at
    return (1 - taus) ** 3 * (p[0] + p[1] * taus + p[2] * taus**2) + (taus**3) * (
        p[3] + p[4] * (1 - taus) + p[5] * (1 - taus) ** 2
    )


def _get_dq(w: np.ndarray, dN: float, q: np.ndarray) -> np.ndarray:
    """Derivative of WXYZ quaternion (Equation 21)

    Args:
        w (np.ndarray): Inertial frame angular velocity, shape (3,)
        dN (float): Derivative of quaternion norm
        q (np.ndarray): Current WXYZ quaternion, shape (4,)

    Returns:
        np.ndarray: Quaternion derivative, shape (4,)
    """
    return _multiply(_pure((1 / 2) * w + dN), q)


def _get_ddq(
    w: np.ndarray, dw: np.ndarray, dN: float, ddN: float, q: np.ndarray
) -> np.ndarray:
    """Second derivative of WXYZ quaternion (Equation 22)

    Args:
        w (np.ndarray): Inertial frame angular velocity, shape (3,)
        dw (np.ndarray): Derivative of angular velocity, shape (3,)
        dN (float): Derivative of quaternion norm
        ddN (float): Second derivative of quaternion norm
        q (np.ndarray): Current WXYZ quaternion, shape (4,)

    Returns:
        np.ndarray: Quaternion derivative, shape (4,)
    """
    return _multiply(_pure((1 / 2) * dw + dN * w - (1 / 4) * _squared_norm(w) + ddN), q)


def _dot(q1, q2):
    """Dot product between two quaternions (Equation 4)"""
    return np.dot(q1, q2)


def _conj(q):
    """Conjugate of WXYZ quaternion (Between Eqns. 4/5)"""
    return np.array([1, -1, -1, -1]) * q


def _inv(q):
    """Inverse of a WXYZ quaternion (Between Eqns. 4/5)"""
    N = np.linalg.norm(q)
    return (1 / (N**2)) * _conj(q)


def _pure(v):
    """Pure WXYZ quaternion representation from vector component (Between Eqns. 7/8)"""
    return np.concatenate([[0], v])


def _multiply(q1, q2):
    """Multiplication of two WXYZ quaternions (Equation 3)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _squared_norm(v):
    """Squared vector norm"""
    return np.dot(v, v)


# UNUSED functions below
# These were based on the equations in the paper or implemented manually (such as some of the additional derivatives)
# Leaving these juse in case they are helpful in the future


def _get_q_from_curve(p, tau):
    # Equation 12
    return (1 - tau) ** 3 * (p[0] + p[1] * tau + p[2] * tau**2) + (tau**3) * (
        p[3] + p[4] * (1 - tau) + p[5] * (1 - tau) ** 2
    )


def _get_dq_from_curve(p, tau):
    # Derivative of equation 12 wrt time
    return (
        -3 * (1 - tau) ** 2 * (p[0] + p[1] * tau + p[2] * tau**2)
        + (1 - tau) ** 3 * (p[1] + 2 * p[2] * tau)
        + 3 * tau**2 * (p[3] + p[4] * (1 - tau) + p[5] * (1 - tau) ** 2)
        + tau**3 * (-p[4] - 2 * p[5] * (1 - tau))
    )


def _get_ddq_from_curve(p, tau):
    # Second derivative of equation 12 wrt time
    return (
        6 * (1 - tau) * (p[0] + p[1] * tau + p[2] * tau**2)
        - 6 * (1 - tau) ** 2 * (p[1] + 2 * p[2] * tau)
        + (1 - tau) ** 3 * 2 * p[2]
        + 6 * tau * (p[3] + p[4] * (1 - tau) + p[5] * (1 - tau) ** 2)
        + 6 * tau**2 * (-p[4] - 2 * p[5] * (1 - tau))
        + tau**3 * 2 * p[5]
    )


def _get_N(q):
    """Quaternion norm (should be 1 ideally)"""
    return np.linalg.norm(q)


def _get_w(q, dq):
    # Equation 6, part 2
    return (2 * dq * _inv(q))[1:]  # Index the vector part of pure quat


def _get_dw(q, dq, ddq):
    # Equation 7, part 2
    q_inv = _inv(q)
    return (2 * ddq * q_inv - 2 * (dq * q_inv) ** 2)[
        1:
    ]  # Index the vector part of pure quat


def _main():
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    # Calculate and plot an example interpolation
    ti = 0  # Initial time
    tf = 5  # Final time
    T = tf - ti  # Duration
    qi = np.array([0, 0, 0, 1])  # Initial quaternion
    wi = 0.1 * np.random.rand(3)  # Initial ang vel
    dwi = np.zeros(3)  # Initial ang accel
    qf = np.random.rand(4)  # Final quaternion (pre-normalization)
    qf /= np.linalg.norm(qf)  # Normalize
    wf = 0.1 * np.random.rand(3)  # Final angular velocity
    dwf = np.zeros(3)  # Final angular acceleration
    dt = 1 / 350  # Timestep (set to the pybullet physics timestep we're using)
    n = round(T / dt)  # Number of timesteps
    qs = quaternion_interpolation_with_bcs(qi, qf, wi, wf, dwi, dwf, T, n)
    ws = quats_to_angular_velocities(qs, dt)
    dws = np.gradient(ws, dt, axis=0)

    # Plot the quaternions, angular velocities, and angular accelerations
    fig = plt.figure()
    subfigs = fig.subfigures(1, 3)
    left = subfigs[0].subplots(1, 4)
    middle = subfigs[1].subplots(1, 3)
    right = subfigs[2].subplots(1, 3)
    x_axis = range(qs.shape[0])
    q_labels = ["qw", "qx", "qy", "qz"]
    w_labels = ["wx", "wy", "wz"]
    dw_labels = ["ax", "ay", "az"]
    x_label = "Time"
    for i, ax in enumerate(left):
        ax.plot(x_axis, qs[:, i])
        ax.set_title(q_labels[i])
        ax.set_xlabel(x_label)
    for i, ax in enumerate(middle):
        ax.plot(x_axis, ws[:, i])
        ax.set_title(w_labels[i])
        ax.set_xlabel(x_label)
    for i, ax in enumerate(right):
        ax.plot(x_axis, dws[:, i])
        ax.set_title(dw_labels[i])
        ax.set_xlabel(x_label)
    plt.show()


if __name__ == "__main__":
    _main()
