"""Tools for plotting geometric objects"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from reachbot_manipulation.utils.plotting import gca_3d, gca_2d


def plot_circle(
    center: npt.ArrayLike,
    radius: float,
    n: int = 50,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a circle

    Args:
        center (npt.ArrayLike): Center of the circle, shape (2,)
        radius (float): Radius of the circle
        n (int, optional): Number of points to use for plotting. Defaults to 50.
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_2d()
    ax.set_aspect("equal")
    thetas = np.linspace(0, 2 * np.pi, n)
    x = center[0] + radius * np.cos(thetas)
    y = center[1] + radius * np.sin(thetas)
    ax.plot(x, y)
    if show:
        plt.show()
    return ax


def plot_sphere(
    center: npt.ArrayLike,
    radius: float,
    n: int = 10,
    color: Union[str, npt.ArrayLike] = (1, 0, 0),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a sphere

    Args:
        center (npt.ArrayLike): Center of the sphere, shape (3,)
        radius (float): Radius of the sphere
        n (int, optional): Number of angular discretizations for plotting. Defaults to 10.
        color (Union[str, npt.ArrayLike], optional). Color of the sphere. Defaults to (1, 0, 0) (red)
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_3d()
    ax.set_aspect("equal")

    elevation_angles = np.linspace(0, np.pi, n)
    azimuth_angles = np.linspace(0, 2 * np.pi, n)

    sin_elevations = np.sin(elevation_angles)
    cos_elevations = np.cos(elevation_angles)

    sin_azimuths = np.sin(azimuth_angles)
    cos_azimuths = np.cos(azimuth_angles)

    X = center[0] + radius * np.outer(sin_elevations, sin_azimuths)
    Y = center[1] + radius * np.outer(sin_elevations, cos_azimuths)
    Z = center[2] + radius * np.outer(cos_elevations, np.ones_like(azimuth_angles))

    ax.plot_surface(X, Y, Z, color=color)
    if show:
        plt.show()
    return ax


# https://stackoverflow.com/questions/39822480/plotting-a-solid-cylinder-centered-on-a-plane-in-matplotlib
def plot_cylinder(
    center: npt.ArrayLike,
    radius: float,
    length: float,
    direction: npt.ArrayLike,
    hollow: bool = True,
    n: int = 50,
    color: Union[str, npt.ArrayLike] = (0, 0, 1, 1),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a 3D cylinder

    Args:
        center (npt.ArrayLike): Center of the cylinder, shape (3,)
        radius (float): Radius of the cylinder
        length (float): Length of the cylinder
        direction (npt.ArrayLike): Axis specifying the direction along the cylinder, shape (3,)
        hollow (bool, optional): Whether or not the cylinder has endcaps. Defaults to True (hollow, no endcaps).
        n (int, optional): Angular discretization. Defaults to 50.
        color (Union[str, npt.ArrayLike], optional): Matplotlib color spec for the surface. Defaults to (0, 0, 1, 1).
        ax (Optional[plt.Axes], optional): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    # Set up plotting axes
    if ax is None:
        ax = gca_3d()
    ax.set_aspect("equal")

    direction = np.asarray(direction)
    direction = direction / np.linalg.norm(direction)

    # Form orthonormal basis spanned by the direction and two orthogonal unit vectors
    nonparallel_unit_vec = np.array([1, 0, 0])
    if np.dot(direction, nonparallel_unit_vec) >= 1 - 1e-2:
        nonparallel_unit_vec = np.array([0, 1, 0])
    u_1 = np.cross(direction, nonparallel_unit_vec)
    u_1 = u_1 / np.linalg.norm(u_1)
    u_2 = np.cross(direction, u_1)

    # Generate coordinates for surface
    dist_lims = np.array([-length / 2, length / 2])
    thetas = np.linspace(0, 2 * np.pi, n)
    dist_lims_grid, theta_grid = np.meshgrid(dist_lims, thetas)
    X, Y, Z = [
        center[i]
        + direction[i] * dist_lims_grid
        + radius * np.sin(theta_grid) * u_1[i]
        + radius * np.cos(theta_grid) * u_2[i]
        for i in [0, 1, 2]
    ]
    ax.plot_surface(X, Y, Z, color=color)

    # Add end caps if desired
    if not hollow:
        r_lims = np.array([0, radius])
        r_lims_grid, theta_grid_2 = np.meshgrid(r_lims, thetas)
        # Bottom cap
        X2, Y2, Z2 = [
            center[i]
            - direction[i] * length / 2
            + r_lims_grid[i] * np.sin(theta_grid_2) * u_1[i]
            + r_lims_grid[i] * np.cos(theta_grid_2) * u_2[i]
            for i in [0, 1, 2]
        ]
        # Top cap
        X3, Y3, Z3 = [
            center[i]
            + direction[i] * length / 2
            + r_lims_grid[i] * np.sin(theta_grid_2) * u_1[i]
            + r_lims_grid[i] * np.cos(theta_grid_2) * u_2[i]
            for i in [0, 1, 2]
        ]
        ax.plot_surface(X2, Y2, Z2, color=color)
        ax.plot_surface(X3, Y3, Z3, color=color)

    if show:
        plt.show()
    return ax


def plot_task_polytope_3d_subset(
    center: npt.ArrayLike,
    basis: npt.ArrayLike,
    lengths: npt.ArrayLike,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a 3D subset of the basis-aligned task polytope (optimized for in the grasp site planner)

    This can either be the force component or a torque component of the optimized polytope, for instance

    Args:
        center (npt.ArrayLike): Center of the polytope (i.e. either the nominal force or torque), shape (3,)
        basis (npt.ArrayLike): Polytope basis vectors, shape (3, 3) or (6, 3) if both the positive and negative basis
            is specified
        lengths (npt.ArrayLike): Lengths of the polytope along each basis direction, shape (3,) or (6,)
        ax (Optional[plt.Axes], optional): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_3d()
    ax.set_aspect("equal")
    basis = np.atleast_2d(basis)
    lengths = np.ravel(lengths)
    center = np.ravel(center)
    assert center.shape[0] == 3
    # Consider the basis/lengths in *both* directions (corresponding to +/- basis directions)
    if basis.shape[0] == 3:
        basis = np.vstack([basis, -1 * basis])
    elif basis.shape[0] != 6:
        raise ValueError(
            "Invalid basis. Ensure that these correspond to a 3D subset of the polytope"
        )
    if lengths.shape[0] == 3:
        lengths = np.concatenate([lengths, lengths])
    elif lengths.shape[0] != 6:
        raise ValueError(
            "Invalid standard deviations. Ensure that these correspond to a 3D subset of the polytope"
        )
    pts = center + lengths.reshape(-1, 1) * basis
    return plot_3d_hull(ConvexHull(pts), ax, show=show)


def plot_3d_box(
    center: npt.ArrayLike,
    sidelengths: npt.ArrayLike,
    rotation: np.ndarray,
    color: Union[str, npt.ArrayLike] = (0, 1, 1, 0.2),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a 3D box

    Args:
        center (npt.ArrayLike): Center of the box, shape (3,)
        sidelengths (npt.ArrayLike): Lengths of the box's sides, shape (3,)
        rotation (np.ndarray): Rotation matrix defining the box's orientation, shape (3, 3)
        color (Union[str, npt.ArrayLike], optional): Matplotlib color spec for the faces of the box.
            Defaults to (0, 1, 1, 0.2) (slightly transparent cyan)
        ax (Optional[plt.Axes], optional): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The 3D plot
    """
    if ax is None:
        ax = gca_3d()
    ax.set_aspect("equal")
    l, w, h = sidelengths
    verts = (
        np.asarray(center)
        + np.array(
            [
                [l / 2, w / 2, h / 2],
                [l / 2, w / 2, -h / 2],
                [l / 2, -w / 2, h / 2],
                [l / 2, -w / 2, -h / 2],
                [-l / 2, w / 2, h / 2],
                [-l / 2, w / 2, -h / 2],
                [-l / 2, -w / 2, h / 2],
                [-l / 2, -w / 2, -h / 2],
            ]
        )
        @ rotation.T
    )
    idxs = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    faces = verts[idxs]
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=color,
            linewidths=1,
            edgecolors=(0, 0, 0, 0.5),
        )
    )
    if show:
        plt.show()
    return ax


def plot_3d_hull(
    hull: ConvexHull,
    ax: Optional[plt.Axes] = None,
    centered: bool = True,
    show: bool = True,
) -> plt.Axes:
    """Plots a 3D convex hull

    Args:
        hull (ConvexHull): Convex hull to plot
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        centered (bool, optional): Whether to center the plot on the hull. Defaults to True
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_3d()
    faces = hull.points[hull.simplices]  # Shape (n_simplices, 3, 3)
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors="cyan",  # TODO include alpha channel in rgba
            linewidths=0.75,
            edgecolors=(0, 0, 0, 0.3),
            alpha=0.2,
        )
    )
    # Center the plot on the hull
    if centered:
        lims = np.column_stack(
            [np.min(hull.points, axis=0), np.max(hull.points, axis=0)]
        )
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
    ax.set_aspect("equal")
    if show:
        plt.show()
    return ax


def plot_2d_hull(
    hull: ConvexHull,
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    show: bool = True,
    **plt_kwargs
) -> plt.Axes:
    """Plots a 2D convex hull

    Args:
        hull (ConvexHull): Convex hull to plot
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        color (str, optional): Matplotlib line color. Defaults to "k" (black)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_2d()
    ax.set_aspect("equal")
    edges = hull.points[hull.simplices]  # Shape (n_simplices, 2, 2)
    ax.plot(*edges.T, color=color, **plt_kwargs)
    if show:
        plt.show()
    return ax
