"""Functions for free-body dynamics, state-space control, and other tools

Note that any equations involving quaternions or angular velocity use our convention of XYZW quaternions and 
world-frame angular velocity definition (to match Pybullet)

Key equations:
F = m * a
T = I * alpha + omega x (I * omega) + m * r x a

Notation:
F: Force, in world frame
m: Mass
a: Linear acceleration of the body's center of mass, in world frame
T = Torque, in world frame
I = Inertia tensor of the body, in world frame
alpha: Angular acceleration of the body, in world frame
omega: Angular velocity of the body, in world frame
r: Position of the body's center of mass w.r.t the point of interest on the body, in world frame

Notes:
- If the point of interest is the center of mass of the body, the third term in the torque expression is 0
- For Reachbot, this point (the base frame) IS the center of mass (not considering the booms). So, this term is either
  exactly 0 or almost 0
"""

import numpy as np
import numpy.typing as npt


def inertial_transformation(
    mass: float, inertia: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Transform an inertia tensor defined for a local reference frame into a new reference frame

    Reference: Stanford ME320 Intro to Robotics course reader, chapter 5

    Args:
        mass (float): Mass of the object
        inertia (np.ndarray): Inertia tensor of the object, determined for its local frame. Shape (3, 3)
        transform (np.ndarray): "Local to desired" transformation matrix, shape (4, 4)

    Returns:
        np.ndarray: Transformed inertia tensor
    """
    p = transform[:3, 3]
    R = transform[:3, :3]
    # Parallel axis theorem for the translation component
    I = inertia + mass * (np.dot(p, p) * np.eye(3) - np.outer(p, p))
    # Similarity transform for the rotation component
    return R @ I @ R.T


def box_inertia(m: float, l: float, w: float, h: float) -> np.ndarray:
    """Inertia tensor for a solid, uniform-material box

    Args:
        m (float): Mass, kg
        l (float): Length (x-axis dimension), meters
        w (float): Width (y-axis dimension), meters
        h (float): Height (z-axis dimension), meters

    Returns:
        np.ndarray: Inertia tensor, shape (3, 3)
    """
    return (1 / 12) * m * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])


def state_matrix(
    q: npt.ArrayLike,
    w: npt.ArrayLike,
    inertia: np.ndarray,
    inv_inertia: np.ndarray,
) -> np.ndarray:
    """The A matrix, such that x_dot = Ax + Bu (+ gravity effects, disturbances, unmodeled effects)

    We assume that the state x = [position, velocity, quaternion, angular velocity] ∈ R13

    This is dependent on the current quaternion/angular velocity due to the nonlinearities
    in the orientation representation

    Args:
        q (npt.ArrayLike): XYZW quaternion, shape (4,)
        w (npt.ArrayLike): Angular velocity, shape (3,)
        inertia (np.ndarray): Inertia tensor (world frame), shape (3, 3)
        inv_inertia (np.ndarray): Inverse of the inertia tensor (world frame), shape (3, 3).

    Returns:
        np.ndarray: The A (state) matrix
    """

    if inv_inertia is None:
        inv_inertia = np.linalg.inv(inertia)
    A = np.zeros((13, 13))
    qx, qy, qz, qw = q
    w1, w2, w3 = w
    # Relationship between velocity / acceleration and current position / velocity
    A[:6, :6] = np.kron([[0, 1], [0, 0]], np.eye(3))
    # Relationship between quaternion derivative and current quaternion / angular velocity
    # NOTE: this is effectively a concatenation of the standard quaternion derivative matrices, HOWEVER, we multiply by
    # an extra factor of 1/2 here because of the way that this matrix multiplication A @ x will "double count" the
    # quaternion derivative effect, since we will have q_dot = f(w) @ q + f(q) @ w
    A[6:10, 6:] = (1 / 2) * np.array(
        [
            [0, -w3 / 2, w2 / 2, w1 / 2, qw / 2, qz / 2, -qy / 2],
            [w3 / 2, 0, -w1 / 2, w2 / 2, -qz / 2, qw / 2, qx / 2],
            [-w2 / 2, w1 / 2, 0, w3 / 2, qy / 2, -qx / 2, qw / 2],
            [-w1 / 2, -w2 / 2, -w3 / 2, 0, -qx / 2, -qy / 2, -qz / 2],
        ]
    )
    # Relationship between angular acceleration and angular velocity
    A[10:, 10:] = -1 * inv_inertia @ _jac_w_of_wxIw(inertia, w)
    return A


def basic_control_matrix(mass: float, inv_inertia: np.ndarray) -> np.ndarray:
    """The B matrix, such that x_dot = Ax + Bu (+ gravity effects, disturbances, unmodeled effects)

    We assume that the state x = [position, velocity, quaternion, angular velocity] ∈ R13
    and that the control u = [force, torque] ∈ R6

    Args:
        mass (float): Mass of the system
        inv_inertia (np.ndarray): Inverse of the inertia tensor (world frame), shape (3, 3)

    Returns:
        np.ndarray: The B (control) matrix, shape (13, 6)
    """
    B = np.zeros((13, 6))
    B[3:6, :3] = (1 / mass) * np.eye(3)
    B[10:, 3:] = inv_inertia
    return B


def _jac_w_of_wxIw(I: np.ndarray, w: npt.ArrayLike) -> np.ndarray:
    """Helper function. Computes the Jacobian of the expression (w x I w) with respect to w

    Derivation is via the MATLAB symbolic toolbox. e.g. jacobian(cross(w, I*w), [w])
    with w and I defined symbolically w.r.t their components

    Args:
        I (np.ndarray): Inertia tensor (world frame). Symmetric, positive semidefinite. Shape (3, 3)
        w (npt.ArrayLike): Angular velocity vector, shape (3,)

    Returns:
        np.ndarray: Jacobian matrix, shape (3, 3)
    """
    Ixx, Iyy, Izz = np.diag(I)
    Ixy = I[0, 1]
    Ixz = I[0, 2]
    Iyz = I[1, 2]
    w1, w2, w3 = w
    return np.array(
        [
            [
                Ixz * w2 - Ixy * w3,
                Ixz * w1 - Iyy * w3 + 2 * Iyz * w2 + Izz * w3,
                Izz * w2 - Iyy * w2 - 2 * Iyz * w3 - Ixy * w1,
            ],
            [
                Ixx * w3 - 2 * Ixz * w1 - Iyz * w2 - Izz * w3,
                Ixy * w3 - Iyz * w1,
                Ixx * w1 + Ixy * w2 + 2 * Ixz * w3 - Izz * w1,
            ],
            [
                2 * Ixy * w1 - Ixx * w2 + Iyy * w2 + Iyz * w3,
                Iyy * w1 - 2 * Ixy * w2 - Ixz * w3 - Ixx * w1,
                Iyz * w1 - Ixz * w2,
            ],
        ]
    )
