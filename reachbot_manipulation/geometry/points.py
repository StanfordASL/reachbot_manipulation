"""Defining geometric configurations of points"""

import numpy as np
import numpy.typing as npt


def rectangular_prism_points(
    center: npt.ArrayLike = (0, 0, 0),
    sidelengths: npt.ArrayLike = (1, 1, 1),
    rotation: np.ndarray = np.eye(3),
) -> np.ndarray:
    """Construct a set of eight points defining the corners of a rectangular prism

    Args:
        center (npt.ArrayLike, optional): Center of the rectangular prism. Defaults to (0, 0, 0).
        sidelengths (npt.ArrayLike, optional): Lengths of the prism's sides. Defaults to (1, 1, 1).
        rotation (np.ndarray, optional): Rotation matrix defining orientation of the points. Defaults to np.eye(3).

    Returns:
        np.ndarray: Points, shape (8, 3)
    """
    l, w, h = sidelengths
    return (
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


def cube_points(
    center: npt.ArrayLike = (0, 0, 0),
    sidelength: float = 1,
    rotation: np.ndarray = np.eye(3),
) -> np.ndarray:
    """Construct a set of eight points defining the corners of a cube

    Args:
        center (npt.ArrayLike, optional): Center of the cube. Defaults to (0, 0, 0).
        sidelength (float, optional): Length of the cube's sides. Defaults to 1.
        rotation (np.ndarray, optional): Rotation matrix defining orientation of the points. Defaults to np.eye(3).

    Returns:
        np.ndarray: Points, shape (8, 3)
    """
    return (
        np.asarray(center)
        + np.array(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5],
            ]
        )
        @ rotation.T
        * sidelength
    )


def triangular_prism_points(
    center: npt.ArrayLike = (0, 0, 0),
    length: float = 1,
    radius: float = 1,
    rotation: np.ndarray = np.eye(3),
) -> np.ndarray:
    """Construct a set of six points defining the corners of a triangular prism

    With an identity rotation, the points are aligned so that the long axis of the triangular prism is along the Z axis
    and one tip of the triangles is along the Y axis

    Args:
        center (npt.ArrayLike, optional): Center of the triangular prism. Defaults to (0, 0, 0).
        length (float, optional): Length of the long axis of the triangular prism. Defaults to 1.
        radius (float, optional): Radius of the circumscribed circle defining the triangle. Defaults to 1.
        rotation (np.ndarray, optional): Rotation matrix defining orientation of the points. Defaults to np.eye(3).

    Returns:
        np.ndarray: Points, shape (6, 3)
    """
    xy = triangle_points(radius=radius)
    z = length / 2 * np.ones(3)
    pts = np.column_stack([np.row_stack([xy, xy]), np.concatenate([z, -z])])
    return np.asarray(center) + pts @ rotation.T


def skewed_rectangular_prism_points(
    center: npt.ArrayLike = (0, 0, 0),
    sidelengths: npt.ArrayLike = (1, 1, 1),
    rotation: np.ndarray = np.eye(3),
    skew_axis: str = "x",
) -> np.ndarray:
    """Construct a set of eight points defining the corners of a *skewed* rectangular prism

    "Skewed" meaning that there is a twist about the speficied axis

    Args:
        center (npt.ArrayLike, optional): Center of the rectangular prism. Defaults to (0, 0, 0).
        sidelengths (npt.ArrayLike, optional): Lengths of the prism's sides. Defaults to (1, 1, 1).
        rotation (np.ndarray, optional): Rotation matrix defining orientation of the points. Defaults to np.eye(3).
        skew_axis (str, optional): Axis about which to twist. "x", "y", or "z". Defaults to "x"

    Returns:
        np.ndarray: Points, shape (8, 3)
    """
    l, w, h = sidelengths
    skew_axis = skew_axis.lower()
    if skew_axis == "x":
        pts = np.array(
            [
                [l / 2, w / 2, h / 2],
                [l / 2, w / 2, -h / 2],
                [l / 2, -w / 2, h / 2],
                [l / 2, -w / 2, -h / 2],
                [-l / 2, h / 2, w / 2],
                [-l / 2, h / 2, -w / 2],
                [-l / 2, -h / 2, w / 2],
                [-l / 2, -h / 2, -w / 2],
            ]
        )
    elif skew_axis == "y":
        pts = np.array(
            [
                [l / 2, w / 2, h / 2],
                [l / 2, w / 2, -h / 2],
                [h / 2, -w / 2, l / 2],
                [h / 2, -w / 2, -l / 2],
                [-l / 2, w / 2, h / 2],
                [-l / 2, w / 2, -h / 2],
                [-h / 2, -w / 2, l / 2],
                [-h / 2, -w / 2, -l / 2],
            ]
        )
    elif skew_axis == "z":
        pts = np.array(
            [
                [l / 2, w / 2, h / 2],
                [w / 2, l / 2, -h / 2],
                [l / 2, -w / 2, h / 2],
                [w / 2, -l / 2, -h / 2],
                [-l / 2, w / 2, h / 2],
                [-w / 2, l / 2, -h / 2],
                [-l / 2, -w / 2, h / 2],
                [-w / 2, -l / 2, -h / 2],
            ]
        )
    else:
        raise ValueError("Skew axis not recognized")

    return np.asarray(center) + pts @ rotation.T


def skewed_triangular_prism_points(
    center: npt.ArrayLike = (0, 0, 0),
    length: float = 1,
    radius: float = 1,
    rotation: np.ndarray = np.eye(3),
) -> np.ndarray:
    """Construct a set of six points defining the corners of a *skewed* triangular prism

    "Skewed" meaning that there is a twist in the triangle about the z axis

    With an identity rotation, the points are aligned so that the long axis of the triangular prism is along the Z axis

    Args:
        center (npt.ArrayLike, optional): Center of the triangular prism. Defaults to (0, 0, 0).
        length (float, optional): Length of the long axis of the triangular prism. Defaults to 1.
        radius (float, optional): Radius of the circumscribed circle defining the triangle. Defaults to 1.
        rotation (np.ndarray, optional): Rotation matrix defining orientation of the points. Defaults to np.eye(3).

    Returns:
        np.ndarray: Points, shape (6, 3)
    """
    # 2D rotation matrix to rotate the XY coordinates of the bottom points
    R = np.array(
        [
            [np.cos(np.pi / 3), -np.sin(np.pi / 3)],
            [np.sin(np.pi / 3), np.cos(np.pi / 3)],
        ]
    )
    xy_top = triangle_points()
    xy_bot = xy_top @ R.T
    z = np.kron(np.array([1, -1]), length / 2 * np.ones(3))
    pts = np.column_stack([np.row_stack([xy_top, xy_bot]), z])
    return np.asarray(center) + pts @ rotation.T * radius


def tetrahedron_points(
    center: npt.ArrayLike = (0, 0, 0),
    scale: float = 1,
    rotation: np.ndarray = np.eye(3),
) -> np.ndarray:
    """Construct a set of four points defining the corners of a tetrahedron

    With an identity rotation, the points are aligned so that one tip of the tetrahedron is along the z axis, and one
    edge of the triangular base is parallel to the x axis

    Args:
        center (npt.ArrayLike, optional): Center of the tetrahedron. Defaults to (0, 0, 0).
        scale (float, optional): Scaling factor for resizing the tetrahedron. Defaults to 1.
        rotation (np.ndarray, optional): Rotation matrix defining orientation of the points. Defaults to np.eye(3).

    Returns:
        np.ndarray: Points, shape (4, 3)
    """
    return (
        np.asarray(center)
        + np.array(
            [
                [np.sqrt(8 / 9), 0, -1 / 3],
                [-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3],
                [-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3],
                [0, 0, 1],
            ]
        )
        @ rotation.T
        * scale
    )


def triangle_points(
    center: npt.ArrayLike = (0, 0),
    radius: float = 1,
    rotation: np.ndarray = np.eye(2),
) -> np.ndarray:
    """Construct a set of three 2D points defining the corners of an equilateral triangle

    With an identity rotation, the points are aligned so that one tip of the triangle is along the Y axis, and one
    the bottom edge is parallel to the x axis

    Args:
        center (npt.ArrayLike, optional): Center of the triangle. Defaults to (0, 0).
        radius (float, optional): Circumscribed radius. Defaults to 1.
        rotation (np.ndarray, optional): Rotation matrix defining orientation of the points. Defaults to np.eye(2).

    Returns:
        np.ndarray: Points, shape (3, 2)
    """
    return (
        np.asarray(center)
        + np.array(
            [
                [0, 1],
                [-np.cos(np.pi / 6), -np.sin(np.pi / 6)],
                [np.cos(np.pi / 6), -np.sin(np.pi / 6)],
            ]
        )
        @ rotation.T
        * radius
    )


# Based on https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate approximately evenly distributed points on a sphere

    Args:
        n (int): Number of points to generate

    Returns:
        np.ndarray: Points on the unit 2-sphere in 3D, shape (n, 3)
    """
    phi = np.pi * (np.sqrt(5.0) - 1.0)
    idxs = np.arange(n)
    ys = 1 - (idxs / (n - 1)) * 2
    radii = np.sqrt(1 - np.multiply(ys, ys))
    thetas = phi * idxs
    xs = np.multiply(np.cos(thetas), radii)
    zs = np.multiply(np.sin(thetas), radii)
    return np.column_stack((xs, ys, zs))


def _test_skewed_points():
    # pylint: disable=import-outside-toplevel
    import pybullet
    from reachbot_manipulation.utils.debug_visualizer import visualize_points

    pybullet.connect(pybullet.GUI)
    pts = skewed_rectangular_prism_points(
        sidelengths=(1, 3, 1), center=(0, 0, -1), skew_axis="z"
    )
    pts_2 = skewed_triangular_prism_points(center=(0, 0, 1))
    visualize_points(pts, (0, 0, 1))
    visualize_points(pts_2, (1, 0, 0))
    input()


if __name__ == "__main__":
    _test_skewed_points()
