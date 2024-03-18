"""Ellipsoid computation and plotting"""

# TODO reorganize with the plotting file

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

from reachbot_manipulation.geometry.icosphere import icosphere
from reachbot_manipulation.utils.plotting import gca_3d


def get_2d_ellipse(A: np.ndarray, xTAx: float = 1, n_pts: int = 50) -> np.ndarray:
    """Gets points on an 2D ellipse defined by the quadratic form x.T @ A @ x = b

    Args:
        A (np.ndarray): Square, symmetric, PSD ellipse matrix, shape (2, 2)
        xTAx (float, optional): Value of the quadratic form. Defaults to 1.
        n_pts (int, optional): Number of points on the ellipse. Defaults to 50.

    Returns:
        np.ndarray: Ellipse (x, y) points, shape (n_pts, 2)
    """
    if A.shape != (2, 2):
        raise ValueError("Invalid A matrix: Must be of shape (2, 2)")
    # Get eigenvalues and eigenvectors
    d, V = np.linalg.eig(A)
    if np.any(d < 0):
        raise ValueError("Cannot plot the ellipse, matrix is not PSD")
    # This method needs eigenvalues sorted smallest to largest
    d = np.flip(d)
    V = np.fliplr(V)
    # Find the inverse square root of the A matrix
    B = V.T @ np.diag(1 / np.sqrt(d)) @ V
    # Map a circle of points to the ellipse via this inverse square root matrix
    thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    xy_circle = np.sqrt(xTAx) * np.row_stack([np.cos(thetas), np.sin(thetas)])
    return (B @ xy_circle).T


def plot_2d_ellipse(
    A: np.ndarray,
    xTAx: float = 1,
    n_pts: int = 50,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs
) -> plt.Axes:
    """Plot a 2D ellipse defined by the quadratic form x.T @ A @ x = b

    Args:
        A (np.ndarray): Square, symmetric, PSD ellipse matrix, shape (2, 2)
        xTAx (float, optional): Value of the quadratic form. Defaults to 1.
        n_pts (int, optional): Number of points on the ellipse. Defaults to 50.
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = plt.gca()
    # Plot ellipse defined by quadratic form
    xy_ellipse = get_2d_ellipse(A, xTAx, n_pts)
    ax.plot(xy_ellipse[:, 0], xy_ellipse[:, 1], **kwargs)
    ax.set_aspect("equal")
    if show:
        plt.show()
    return ax


def get_3d_ellipsoid(
    A: np.ndarray, xTAx: float = 1, point_density: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Gets points on an 3D ellipsoid defined by the quadratic form x.T @ A @ x = b

    Args:
        A (np.ndarray): Square, symmetric, PSD ellipse matrix, shape (3, 3)
        xTAx (float, optional): Value of the quadratic form. Defaults to 1.
        point_density (int, optional):  Number of subdivisions of space to create the mesh (>=1). Defaults to 3.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            np.ndarray: Vertices on the ellipsoid, shape (n_verts, 3)
            np.ndarray: Faces of the ellipsoid (sets of vertex indices defining triangular faces), shape (n_faces, 3)
    """
    if A.shape != (3, 3):
        raise ValueError("Invalid A matrix: Must be of shape (3, 3)")
    # Get eigenvalues and eigenvectors
    d, V = np.linalg.eig(A)
    if np.any(d < 0):
        raise ValueError("Cannot plot the ellipse, matrix is not PSD")
    # This method needs eigenvalues sorted smallest to largest
    d = np.flip(d)
    V = np.fliplr(V)
    # Find the inverse square root of the A matrix
    B = V.T @ np.diag(1 / np.sqrt(d)) @ V
    # Map a sphere of points/faces to the ellipse via this inverse square root matrix
    sphere_verts, sphere_faces = icosphere(point_density)
    sphere_verts *= np.sqrt(xTAx)
    return sphere_verts @ B.T, sphere_faces


def plot_3d_ellipsoid(
    A: np.ndarray,
    xTAx: float = 1,
    point_density: int = 3,
    center: npt.ArrayLike = (0, 0, 0),
    face_color: Union[npt.ArrayLike, str] = (1, 0, 0, 1),
    edge_color: Union[npt.ArrayLike, str] = (0, 0, 0, 1),
    line_width: float = 0.25,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a 3D ellipsoid (defined by the quadratic form x.T @ A @ x = b) in Matplotlib

    Args:
        A (np.ndarray): Square, symmetric, PSD ellipse matrix, shape (3, 3)
        xTAx (float, optional): Value of the quadratic form. Defaults to 1.
        point_density (int, optional): Number of subdivisions of space to create the mesh (>=1). Defaults to 3.
        center (npt.ArrayLike, optional): Center of the ellipsoid, shape (3,)
        face_color (Union[npt.ArrayLike, str]): Face color of the 3D mesh. Defaults to (1, 0, 0, 0.5).
        edge_color (Union[npt.ArrayLike, str]): Edge color of the lines around each face. Defaults to (0, 0, 0, 1).
        line_width (float, optional): Width of the lines around each face. Defaults to 0.25.
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    xyz_ellipse, faces = get_3d_ellipsoid(A, xTAx, point_density)
    return plot_3d_mesh(
        xyz_ellipse + center,
        faces,
        face_color,
        edge_color,
        line_width,
        ax=ax,
        show=show,
    )


def plot_3d_mesh(
    verts: npt.ArrayLike,
    faces: npt.ArrayLike,
    face_color: Union[npt.ArrayLike, str] = (1, 0, 0, 0.5),
    edge_color: Union[npt.ArrayLike, str] = (0, 0, 0, 1),
    line_width: float = 0.25,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a 3D mesh defined by vertices and faces in Matplotlib

    Args:
        verts (npt.ArrayLike): Vertices (a set of 3D points), shape (n_verts, 3)
        faces (npt.ArrayLike): Faces (indices of sets of 3 vertices), shape (n_faces, 3)
        face_color (Union[npt.ArrayLike, str]): Face color of the 3D mesh. Defaults to (1, 0, 0, 0.5).
        edge_color (Union[npt.ArrayLike, str]): Edge color of the lines around each face. Defaults to (0, 0, 0, 1).
        line_width (float, optional): Width of the lines around each face. Defaults to 0.25.
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_3d()
    ax.set_aspect("equal")
    verts = np.atleast_2d(verts)
    faces = np.atleast_2d(faces)
    # Note: adding the points improves how the plot automatically determines the axis limits
    ax.scatter(*verts.T, s=line_width, color=edge_color)
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
        verts[faces],
        shade=True,
        facecolors=face_color,
        edgecolors=edge_color,
        linewidth=line_width,
    )
    ax.add_collection3d(poly)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if show:
        plt.show()
    return ax


def main():
    A = np.diag(1 / np.square([1, 2, 3]))
    plot_3d_ellipsoid(A)


if __name__ == "__main__":
    main()
