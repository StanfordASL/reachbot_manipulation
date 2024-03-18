"""Wrench spaces"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.geometry.convex_hulls import minkowski, hull_to_matrices
from reachbot_manipulation.geometry.plotting import plot_3d_hull
from reachbot_manipulation.utils.grasp_utils import (
    forces_to_wrench,
    friction_pyramid,
)
from reachbot_manipulation.optimization.inscribed_ellipsoid import (
    scale_ellipsoid_in_polyhedron,
)
from reachbot_manipulation.geometry.ellipsoids import plot_3d_ellipsoid


def grasp_wrench_space(
    contact_points: npt.ArrayLike,
    normals: npt.ArrayLike,
    mu: float,  # TODO Add option for mu as an array?
    n_edges: int,
    max_force: float = 1,
    l_inf: bool = False,
) -> ConvexHull:
    """Convex hull representing the space of applied wrenches for a dexterous grasp

    This implicitly assumes that the primitive forces (friction cone edges) are normalized to 1, so a rescaling
    is required if considering forces with a magnitude/limit != 1

    This also assumes that the object's body frame is the same as the world frame

    Args:
        contact_points (npt.ArrayLike): contact_points (npt.ArrayLike): Locations of the applied forces on the body,
            shape (n_points, 3)
        normals (npt.ArrayLike): Surface normal directions, shape (n_points, 3)
        mu (float): Friction coefficient
        n_edges (int): Number of edges of the friction pyramid approximant of the full friction cone
        max_force (float, optional): Maximum force magnitude applied at a contact point. Defaults to 1 N
        l_inf (bool, optional): Whether to compute the full L-infinity wrench space, using the Minkowski sum of all
            wrenches, as opposed to the L1 wrench space, using the convex hull of the wrenches. This is computationally
            intensive but more representative of the actual grasp space. Defaults to False.

    Returns:
        ConvexHull: Wrench space
    """

    n_contacts = len(contact_points)
    if len(normals) != n_contacts:
        raise ValueError("Number of normals must match the number of contacts")
    # Primitive forces: Shape: (n_contacts, n_edges, 3)
    # Primitive wrenches: Shape: (n_contacts, n_edges, 6)
    primitive_forces = [
        max_force * friction_pyramid(normal, mu, n_edges) for normal in normals
    ]
    primitive_wrenches = [np.zeros((n_edges, 6)) for _ in range(n_contacts)]
    for i in range(n_contacts):
        pyramid = primitive_forces[i]
        # Note: Each edge of the friction pyramid is applied at the same contact point
        for j in range(n_edges):
            primitive_wrenches[i][j, :] = forces_to_wrench(
                contact_points[i], pyramid[j]
            )
    if l_inf:
        # Include the 0 vector in each polytope
        polytopes = [np.row_stack([w, np.zeros(6)]) for w in primitive_wrenches]
        return minkowski(polytopes)
    # Include the 0 vector once
    wrenches = np.array(primitive_wrenches).reshape(-1, 6)
    return ConvexHull(np.row_stack([wrenches, np.zeros(6)]))


def cable_wrench_space(
    offsets: npt.ArrayLike,
    normals: npt.ArrayLike,
    max_force: float = 1,
    l_inf: bool = True,
) -> ConvexHull:
    """Determine the wrench space for a CDPR

    This is a simplified version of the grasp_wrench_space function as we don't need to calculate
    any additional forces for a friction cone/pyramid (since there is no friction)

    Args:
        offsets (npt.ArrayLike): World-frame offsets between the robot's center of mass and the location of the cables,
            (vectors from COM to each cable), shape (n_cables, 3)
        normals (npt.ArrayLike): Directions of the cables, pointing from the robot to where the cable is attached,
            shape (n_cables, 3)
        max_force (float, optional): Maximum amount of tensile force applied in a cable. Defaults to 1 N
        l_inf (bool, optional): Whether to compute the full L-infinity wrench space, using the Minkowski sum of all
            wrenches, as opposed to the L1 wrench space, using the convex hull of the wrenches. This is computationally
            intensive for a dexterous grasp, but not hard to compute for a CDPR, because the wrench polytopes are
            simpler. Defaults to True.

    Returns:
        ConvexHull: Wrench space
    """
    n_contacts = len(offsets)
    if len(normals) != n_contacts:
        raise ValueError("Number of normals must match the number of contacts")
    # Primitive forces: Shape: (n_contacts, 3)
    # Primitive wrenches: Shape: (n_contacts, 6)
    primitive_forces = max_force * normalize(normals)
    primitive_wrenches = [
        forces_to_wrench(offsets[i], primitive_forces[i]) for i in range(n_contacts)
    ]
    if l_inf:
        # Include the 0 vector in each polytopes
        polytopes = [np.row_stack([w, np.zeros(6)]) for w in primitive_wrenches]
        return minkowski(polytopes)
    # Include the 0 vector once
    primitive_wrenches.append(np.zeros(6))
    return ConvexHull(np.array(primitive_wrenches).reshape(-1, 6))


def visualize_wrench_space(
    hull: ConvexHull,
    ellipsoid: Optional[np.ndarray] = np.eye(6),
    desired_wrench: npt.ArrayLike = (0, 0, 0, 0, 0, 0),
    axs: Optional[tuple[plt.Axes, plt.Axes]] = None,
    show: bool = True,
) -> tuple[plt.Axes, plt.Axes]:
    """Visualize the wrench space by separately viewing the convex hulls of the force and torque space

    Args:
        hull (ConvexHull): Wrench space
        ellipsoid (Optional[np.ndarray]): Ellipsoid matrix A (quadratic form representation: x.T @ A @ x = b),
            shape (6, 6). None if plotting without a ball/ellipsoid is desired. Defaults to np.eye(6) (a ball)
        desired_wrench (npt.ArrayLike, optional): Desired wrench to use as the center of the largest inscribed ball.
            Defaults to (0, 0, 0, 0, 0, 0).
        axs (Optional[tuple[plt.Axes, plt.Axes]]): Two 3D axes for plotting the force and torque components. Defaults
            to None (create a new figure with two 3D axes)
        show (bool, optional): Whether to show the plot. Defaults to True.

    Returns:
        tuple[plt.Axes, plt.Axes]:
            plt.Axes: Plot of the convex hull of the force space
            plt.Axes: Plot of the convex hull of the torque space
    """
    vertices = hull.points[hull.vertices]
    force_hull = ConvexHull(vertices[:, :3])
    torque_hull = ConvexHull(vertices[:, 3:])

    if axs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    else:
        ax1, ax2 = axs
    ax1.set_title("Force")
    ax2.set_title("Torque")
    ax1 = plot_3d_hull(force_hull, ax=ax1, show=False)
    ax2 = plot_3d_hull(torque_hull, ax=ax2, show=False)

    if ellipsoid is not None:
        assert ellipsoid.shape == (6, 6)
        assert len(desired_wrench) == 6
        force_center = desired_wrench[:3]
        torque_center = desired_wrench[3:]
        force_A, force_b = hull_to_matrices(force_hull)
        torque_A, torque_b = hull_to_matrices(torque_hull)
        color = (1, 0, 0)
        force_ellipse = scale_ellipsoid_in_polyhedron(
            ellipsoid[:3, :3], force_A, force_b
        )
        torque_ellipse = scale_ellipsoid_in_polyhedron(
            ellipsoid[3:, 3:], torque_A, torque_b
        )
        plot_3d_ellipsoid(
            force_ellipse,
            center=force_center,
            face_color=color,
            edge_color=(0, 0, 0, 0),
            line_width=0,
            ax=ax1,
            show=False,
        )
        plot_3d_ellipsoid(
            torque_ellipse,
            center=torque_center,
            face_color=color,
            edge_color=(0, 0, 0, 0),
            line_width=0,
            ax=ax2,
            show=False,
        )

    ax1.set_xlabel("$F_x$ (N)")
    ax1.set_ylabel("$F_y$ (N)")
    ax1.set_zlabel("$F_z$ (N)")
    ax2.set_xlabel("$\\tau_x$ (Nm)")
    ax2.set_ylabel("$\\tau_y$ (Nm)")
    ax2.set_zlabel("$\\tau_z$ (Nm)")

    if show:
        plt.show()
    return ax1, ax2
