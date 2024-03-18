"""Utilities for dexterous manipulation

References:
https://link.springer.com/article/10.1007/s10514-014-9402-3
https://groups.csail.mit.edu/robotics-center/public_papers/Dai15.pdf
CS237B
"""

import numpy as np
import numpy.typing as npt

from reachbot_manipulation.utils.math_utils import skew, normalize
from reachbot_manipulation.utils.rotations import axis_angle_to_rmat


def grasp_matrix(contact_pts: npt.ArrayLike) -> np.ndarray:
    """Generate the grasp matrix for a set of contact points

    This defines the relationship between applied forces and the resultant wrench on the body: W = G @ vec(F)
    where W is the resultant wrench, G is the grasp matrix, and vec(F) is the stacked vector of all of the forces

    All components are defined in the world frame

    From: https://groups.csail.mit.edu/robotics-center/public_papers/Dai15.pdf

    Args:
        contact_pts (npt.ArrayLike): Positions of the contacts w.r.t the center of the body, in world frame.
            Shape (n_points, 3)

    Returns:
        np.ndarray: Grasp matrix, shape (6, 3 * n_points)
    """
    contact_pts = np.atleast_2d(contact_pts)
    n_pts, dim = contact_pts.shape
    assert dim == 3
    return np.column_stack(
        [np.row_stack([np.eye(dim), skew(contact_pts[i])]) for i in range(n_pts)]
    )


def local_grasp_matrix(
    contact_pts: npt.ArrayLike, normals: npt.ArrayLike
) -> np.ndarray:
    """Generate the grasp matrix for forces which are defined *locally*, with respect to their surface normals

    Like the standard grasp matrix, this defines the relationship between applied forces and the resultant wrench.
    However, in this case, the forces are defined such that the z component of the force is aligned with their
    respective surface normal. This is useful for considering the local friction cone (with z as the center of cone)

    Args:
        contact_pts (npt.ArrayLike): Positions of the contacts w.r.t the center of the body, in world frame.
            Shape (n_points, 3)
        normals (npt.ArrayLike): Surface normal directions, shape (n_points, 3)

    Returns:
        np.ndarray: Grasp matrix, shape (6, 3 * n_points)
    """
    contact_pts = np.atleast_2d(contact_pts)
    normals = np.atleast_2d(normals)
    n_pts, dim = contact_pts.shape
    assert dim == 3
    rmats = np.array([grasp_normal_rmat(n) for n in normals])
    return np.column_stack(
        [
            np.row_stack([rmats[i], skew(contact_pts[i]) @ rmats[i]])
            for i in range(n_pts)
        ]
    )


def grasp_normal_rmat(normal: npt.ArrayLike) -> np.ndarray:
    """Determine the rotation matrix associated with a surface normal vector defined as (0, 0, 1) in the local frame

    i.e. "Local contact reference frame to World reference frame"

    For example, if R = grasp_normal_rmat(u), then R @ [0, 0, 1] == u

    Args:
        normal (npt.ArrayLike): Surface normal, shape (3,)

    Returns:
        np.ndarray: Rotation matrix, shape (3, 3)
    """
    # Compute (n x e1), (n x e2), and (n x e3), and choose the one with the largest magnitude to compute x
    normal = normalize(normal)
    nx = skew(normal)
    nx_mag = np.linalg.norm(nx, axis=0)
    idx = np.argmax(nx_mag)
    x = nx[:, idx] / nx_mag[idx]
    y = nx @ x  # z cross x
    return np.column_stack([x, y, normal])


def forces_to_wrench(
    contact_points: npt.ArrayLike, forces: npt.ArrayLike
) -> np.ndarray:
    """Converts a set of forces applied at contact points to their resultant wrenches

    Args:
        contact_points (npt.ArrayLike): Locations of the applied forces on the body, shape (n_pts, 3)
        forces (npt.ArrayLike): Force vectors, shape (n_pts, 3)

    Returns:
        np.ndarray: Applied wrenches, shape (6,)
    """
    contact_points = np.atleast_2d(contact_points)
    forces = np.atleast_2d(forces)
    if contact_points.shape[0] != forces.shape[0]:
        raise ValueError(
            "Mismatched numbers of forces and contacts\n"
            + f"Got {forces.shape[0]} forces and {contact_points.shape[0]} contacts"
        )
    # G @ F where G is is the grasp matrix and F is the stacked form of the forces
    return grasp_matrix(contact_points) @ np.ravel(forces)


def twist_to_contact_vels(
    contact_points: npt.ArrayLike, velocity: npt.ArrayLike, omega: npt.ArrayLike
) -> np.ndarray:
    """Determine the velocities of the grasped object's contact points, based on its velocity/angular velocity

    Args:
        contact_points (npt.ArrayLike): Locations of the applied forces on the body, shape (n_pts, 3)
        velocity (npt.ArrayLike): Body velocity, shape (3,)
        omega (npt.ArrayLike): Body angular velocity, shape (3,)

    Returns:
        np.ndarray: Contact point velocities, shape (n_pts, 3)
    """
    contact_points = np.atleast_2d(contact_points)
    twist = np.concatenate([velocity, omega])
    # G.T @ twist yields the stacked velocities, reshape to unstack
    return (grasp_matrix(contact_points).T @ twist).reshape(-1, 3)


def friction_pyramid(normal: npt.ArrayLike, mu: float, n_edges: int) -> np.ndarray:
    """Pyramidal approximation of a friction cone, with a discrete number of edges

    Based on the implementation in FastGrasp (fpokorny@kth.se)

    Args:
        normal (npt.ArrayLike): Surface normal vector, shape (3,)
        mu (float): Friction coefficient
        n_edges (int): Number of edges of the pyramid (Higher = closer to the actual cone)

    Returns:
        np.ndarray: Edges of the friction cone, shape (n_edges, 3)
    """
    e = np.array([0, 0, 1])
    normal = normalize(normal)
    axis = np.cross(e, normal)
    axis_norm = np.linalg.norm(axis)
    if axis_norm >= 1e-10:
        axis = axis / axis_norm
        R = axis_angle_to_rmat(axis, np.arccos(e.dot(normal)))
    elif np.dot(normal, axis) >= 0:  # Normal approx == (0, 0, 1)
        R = np.eye(3)
    else:  # Normal approx == (0, 0, -1)
        R = np.diag([1, -1, -1])
    forces = np.zeros((n_edges, 3))
    for i in range(n_edges):
        angle = 2 * np.pi * i / n_edges
        f = np.array([mu * np.cos(angle), mu * np.sin(angle), 1])
        forces[i] = R @ f
    return forces


def transform_wrench(wrench: npt.ArrayLike, R: np.ndarray) -> np.ndarray:
    """Transforms a wrench based on a reference frame rotation

    Example:
    >>> base_frame_wrench = transform_wrench(world_frame_wrench, R_world_to_base)

    Args:
        wrench (npt.ArrayLike): Wrench to transform, shape (6,)
        R (np.ndarray): Rotation matrix, shape (3, 3)

    Returns:
        np.ndarray: Transformed wrench, shape (6,)
    """
    return np.concatenate([R @ wrench[:3], R @ wrench[3:]])


def check_static_equilibrium(
    contact_points: npt.ArrayLike, forces: npt.ArrayLike, external_wrench: npt.ArrayLike
) -> bool:
    """Determine if a grasp or cable configuration is in static equilibrium with an external wrench

    All inputs must be defined in the same reference frame

    Args:
        contact_points (npt.ArrayLike): Grasp contact points on the body, or cable positions on the body.
            Shape (n_contacts, 3)
        forces (npt.ArrayLike): Applied forces at the contacts, shape (n_contacts, 3)
        external_wrench (npt.ArrayLike): Extermal wrench to support, shape (6,)

    Returns:
        bool: True if in static equilibrium, False otherwise
    """
    contact_points = np.atleast_2d(contact_points)
    forces = np.atleast_2d(forces)
    n = contact_points.shape[0]
    ws = np.zeros((n, 6))
    for i in range(n):
        r = contact_points[i]
        f = forces[i]
        ws[i] = np.concatenate([f, np.cross(r, f)])
    return np.allclose(np.sum(ws, axis=0) + external_wrench, np.zeros(6))


def _test_friction_pyramid():
    # pylint: disable=import-outside-toplevel
    import pybullet
    from reachbot_manipulation.utils.debug_visualizer import visualize_lines

    pybullet.connect(pybullet.GUI)
    n = 10
    start_pts = np.zeros((n, 3))
    end_pts = friction_pyramid((1, 1, 1), 1, n)
    visualize_lines(start_pts, end_pts, (1, 1, 1))
    input()


if __name__ == "__main__":
    _test_friction_pyramid()
