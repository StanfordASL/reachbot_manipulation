"""Dynamics and kinematics of cable-driven parallel robots (CDPRs)"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import cvxpy as cp
from scipy.linalg import null_space

from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.errors import OptimizationError, LinAlgError
from reachbot_manipulation.utils.python_utils import print_red, print_green
from reachbot_manipulation.geometry.points import (
    cube_points,
    rectangular_prism_points,
)
from reachbot_manipulation.utils.dynamics import basic_control_matrix, state_matrix
from reachbot_manipulation.utils.transformations import transform_points


# TODO should this use the robot's transformation matrix and transform the local start points?
def jacobian(
    start_pts: npt.ArrayLike, end_pts: npt.ArrayLike, robot_pos: npt.ArrayLike
) -> np.ndarray:
    """Jacobian for a cable-driven robot. Maps body twist to cable speeds

    Equation: L_dot = J @ t

    - L_dot: Derivatives of the lengths of each cable, shape (n_cables,)
    - J: Jacobian, shape (n_cables, 6)
    - t: Body twist: [velocity, angular velocity], shape (6,)

    All positions are defined in the WORLD FRAME

    Args:
        start_pts (npt.ArrayLike): World-frame positions of each cable's attachment point on the robot,
            shape (n_cables, 3)
        end_pts (npt.ArrayLike): World-frame positions of each cable's attachment point to the world,
            shape (n_cables, 3)
        robot_pos (npt.ArrayLike): World-frame position of the center of the robot, shape (3,)

    Returns:
        np.ndarray: Jacobian, shape (n_cables, 6)
    """
    # Unit vectors along the cables, pointing inwards towards the robot
    dirs = normalize(np.subtract(start_pts, end_pts))
    # Vectors pointing from the center of the robot body to where the cables originate from the body
    body_to_start_pts = np.subtract(start_pts, robot_pos)
    # Jacobian mapping body twist to velocities along cables. Shape (n_cables, 6)
    return np.column_stack((dirs, np.cross(body_to_start_pts, dirs)))


def cdpr_dynamics(
    pos: npt.ArrayLike,
    orn: npt.ArrayLike,
    vel: npt.ArrayLike,
    omega: npt.ArrayLike,
    start_pts: np.ndarray,
    end_pts: np.ndarray,
    tensions: npt.ArrayLike,
    mass: float,
    inertia: np.ndarray,
    inv_inertia: np.ndarray,
    gravity_vector: npt.ArrayLike,
) -> np.ndarray:
    """Determine the state derivative x_dot,
    assuming that the state x = [position, velocity, quaternion, angular velocity] ∈ R13

    This will include the gravity force, in addition to the standard rigid body dynamics

    Args:
        pos (npt.ArrayLike): Position, shape (3,)
        orn (npt.ArrayLike): Orientation (XYZW quaternion), shape (4,)
        vel (npt.ArrayLike): Linear velocity, shape (3,)
        omega (npt.ArrayLike): Angular velocity, shape (3,)
        start_pts (np.ndarray): World-frame positions of each cable's attachment point on the robot,
            shape (n_cables, 3)
        end_pts (np.ndarray): World-frame positions of each cable's attachment point to the world,
            shape (n_cables, 3)
        tensions (npt.ArrayLike): Cable tensions, shape (n_cables,)
        mass (float): Mass of the robot
        inertia (np.ndarray): Inertia tensor (world frame), shape (3, 3)
        inv_inertia (np.ndarray): Inverse of the inertia tensor (world frame), shape (3, 3).
        gravity_vector (npt.ArrayLike): Gravitational acceleration vector, shape (3,).
            i.e. (0, 0, -9.81) for Earth and (0, 0, -3.71) for Mars

    Returns:
        np.ndarray: State vector derivative: [velocity, acceleration, quaternion derivative, angular acceleration] ∈ R13
    """
    x = np.concatenate([pos, vel, orn, omega])  # State
    J = jacobian(start_pts, end_pts, pos)
    A = state_matrix(orn, omega, inertia, inv_inertia)
    # Note: We'll use the basic control matrix (rather than the cdpr-specific one)
    # because it's easier to add gravity to a wrench rather than cable tensions
    B = basic_control_matrix(mass, inv_inertia)
    wrench = tensions_to_wrench(tensions, J)
    # Account for gravity
    wrench += np.concatenate([np.multiply(mass, gravity_vector), np.zeros(3)])
    return A @ x + B @ wrench  # State derivative, x_dot


def cdpr_control_matrix(
    jacobian: np.ndarray, mass: float, inv_inertia: np.ndarray
) -> np.ndarray:
    """The B matrix, such that x_dot = Ax + Bu (+ gravity effects, disturbances, unmodeled effects),
    for a cable-driven robot

    We assume that the state x = [position, velocity, quaternion, angular velocity] ∈ R13
    and that the control u = [cable tensions] ∈ Rn (n = number of cables)

    Args:
        jacobian (np.ndarray): Jacobian mapping body twist to cable speeds, shape (n_cables, 6)
        mass (float): Mass of the system
        inv_inertia (np.ndarray): Inverse of the inertia tensor, shape (3, 3)

    Returns:
        np.ndarray: The B (control) matrix, shape (13, n)
    """
    # Since wrench = -jacobian.T @ tensions, we can simply compose this as a matrix multiplication with the
    # basic control matrix which assumes a wrench input
    return -1 * basic_control_matrix(mass, inv_inertia) @ jacobian.T


def cdpr_state_matrix(
    q: npt.ArrayLike,
    w: npt.ArrayLike,
    inertia: np.ndarray,
    inv_inertia: np.ndarray,
) -> np.ndarray:
    """The A matrix, such that x_dot = Ax + Bu (+ gravity effects, disturbances, unmodeled effects),
    for a cable-driven robot

    We assume that the state x = [position, velocity, quaternion, angular velocity] ∈ R13

    This is dependent on the current quaternion/angular velocity due to the nonlinearities
    in the orientation representation

    Args:
        q (npt.ArrayLike): XYZW quaternion, shape (4,)
        w (npt.ArrayLike): Angular velocity, shape (3,)
        inertia (np.ndarray): Inertia tensor, shape (3, 3)
        inv_inertia (np.ndarray): Inverse of the inertia tensor. This should be precomputed ahead of time for efficiency

    Returns:
        np.ndarray: The A (state) matrix
    """
    # This is the same state matrix as we would expect from a free-floating rigid body
    return state_matrix(q, w, inertia, inv_inertia)


def wrench_to_tensions(
    wrench: npt.ArrayLike, jacobian: np.ndarray, max_tension: Optional[float] = None
) -> np.ndarray:
    """Determine cable tensions to apply a given wrench

    Args:
        wrench (npt.ArrayLike): Wrench to apply with the robot's base, shape (6,)
        jacobian (np.ndarray): Jacobian defining the relationship between body vels and cable speed, shape (n_cables, 6)
        max_tension (Optional[float], optional): Maximum tension in the cables. Defaults to None (unconstrained).

    Raises:
        LinAlgError: Solution does not exist due to force closure issues
        OptimizationError: Solution does not exist due to the non-compression constraint

    Returns:
        np.ndarray: Cable tensions, shape (n_cables, )
    """
    J_pinv = np.linalg.pinv(jacobian).T
    N = null_space(jacobian.T)
    tau = -J_pinv @ wrench  # Nominal tensions, not including null space or constraints
    if N.size == 0:  # No nullspace
        if np.any(tau < 0):
            raise LinAlgError(
                "Unable to determine the cable tensions to support the wrench.\n"
                + "Jacobian has no null space component, and the nominal solution requires compression"
            )
        return tau
    c = cp.Variable(N.shape[1])  # Multipliers on null space components
    objective = cp.Minimize(cp.norm(c))  # TODO improve objective
    tensions = tau + N @ c
    constraints = [tensions >= 0]
    if max_tension is not None:
        constraints.append(tensions <= max_tension)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        raise OptimizationError(
            "Unable to determine the cable tensions to support the wrench.\n"
            + "While we have force closure, there are no solutions without compression"
        )
    return tensions.value


def check_jacobian(J: np.ndarray):
    if J.shape[-1] != 6:
        raise ValueError(f"Jacobian has an invalid shape: {J.shape}. Must be (n, 6)")
    if np.linalg.matrix_rank(J, tol=1e-12) < 6:
        raise LinAlgError("Jacobian is rank deficient!")


def tensions_to_wrench(tensions: npt.ArrayLike, jacobian: np.ndarray) -> np.ndarray:
    """Determine the wrench applied to the robot body from the cable tensions

    Args:
        tensions (npt.ArrayLike): Cable tensions, shape (n_cables,)
        jacobian (np.ndarray): Jacobian defining the relationship between body vels and cable speed, shape (n_cables, 6)

    Returns:
        np.ndarray: Wrench (Force, Torque), shape (6,)
    """
    try:
        return -jacobian.T @ tensions
    except ValueError as e:
        raise ValueError(
            "Invalid matrix multiplication: Check on the dimensions of the inputs.\n"
            + f"Got shapes: Tensions: {tensions.shape}, Jacobian: {jacobian.shape})"
        ) from e


def cable_speeds(
    jacobian: np.ndarray, velocity: npt.ArrayLike, omega: npt.ArrayLike
) -> np.ndarray:
    """Determine the cable speeds based on the linear/angular velocity of the CDPR robot body

    Args:
        jacobian (np.ndarray): Jacobian defining the relationship between body vels and cable speed, shape (n_cables, 6)
        velocity (npt.ArrayLike): Linear velocity of the robot, shape (3,)
        omega (npt.ArrayLike): Angular velocity of the robot, shape (3,)

    Returns:
        np.ndarray: Cable speeds, shape (n_cables,)
    """
    return jacobian @ np.concatenate([velocity, omega])


def get_cable_lengths(
    tmat: np.ndarray, local_start_pts: np.ndarray, grasp_points: np.ndarray
) -> np.ndarray:
    """Determine the lengths of the cables for a given CDPR configuration

    Args:
        tmat (np.ndarray): Base-to-world transformation matrix for the robot, shape (4, 4)
        local_start_pts (np.ndarray): Robot-frame locations of the cables on the robot, shape (n_cables, 3)
        grasp_points (np.ndarray): World-frame locations of where the cables are attached, shape (n_cables, 3)

    Returns:
        np.ndarray: Lengths of each cable, shape (n_cables,)
    """
    start_pts = transform_points(tmat, local_start_pts)
    return np.linalg.norm(grasp_points - start_pts, axis=1)


def _main():
    # Create an arbitrary jacobian that we know will be in force closure
    J = jacobian(
        cube_points(sidelength=1),
        rectangular_prism_points(sidelengths=(3, 4, 5)),
        (0, 0, 0),
    )
    try:
        check_jacobian(J)
    except Exception as e:
        print_red(e)
    wrench = np.random.rand(6)
    print("Wrench: ", wrench)
    tensions = wrench_to_tensions(wrench, J)
    print("Tensions: ", tensions)
    new_wrench = tensions_to_wrench(tensions, J)
    print("Calculated wrench: ", new_wrench)
    if np.allclose(wrench, new_wrench):
        print_green("Solution is valid")
    else:
        print_red("Solution is invalid")


if __name__ == "__main__":
    _main()
