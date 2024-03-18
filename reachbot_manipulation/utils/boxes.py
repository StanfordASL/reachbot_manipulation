"""Boxes and associated utility functions

These are currently used as the definition of the safe sets in the trajectory optimization

Based on: https://github.com/cvxgrp/fastpathplanning/blob/main/fastpathplanning/boxes.py
"""

from collections import defaultdict
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from reachbot_manipulation.utils.bullet_utils import create_box


class Box:
    """Representation of a box defined by lower/upper limits on its coordinates

    Unpackable as (lower, upper) = box

    Args:
        lower (npt.ArrayLike): Lower limits on the box coordinates, shape (box_dim,)
        upper (npt.ArrayLike): Upper limits on the box coordinates, shape (box_dim,)
    """

    def __init__(self, lower: npt.ArrayLike, upper: npt.ArrayLike):
        self.lower = np.ravel(lower).astype(np.float64)
        self.upper = np.ravel(upper).astype(np.float64)
        self._validate()
        self.center = (self.lower + self.upper) / 2
        self.dim = len(self.lower)

    def __iter__(self):
        return iter([self.lower, self.upper])

    def __str__(self):
        return f"Lower: {list(self.lower)}, Upper: {list(self.upper)}"

    def __repr__(self):
        return (
            f"{type(self).__name__}(lower={list(self.lower)}, upper={list(self.upper)})"
        )

    def _validate(self):
        if len(self.lower) != len(self.upper):
            raise ValueError("Invalid input dimensions")
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid inputs: Mismatched order of lower/upper points")


def expand_box(box: Box, distance: float) -> Box:
    """Increases the size of a box by a given distance

    Args:
        box (Box): The reference box
        distance (float): Amount to increase the size of the box in all dimensions

    Returns:
        Box: The increased-size box
    """
    return Box(box.lower - distance, box.upper + distance)


def contract_box(box: Box, distance: float) -> Box:
    """Reduces the size of a box by a given distance

    Args:
        box (Box): The reference box
        distance (float): Amount to decrease the size of the box in all dimensions

    Returns:
        Box: The reduced-size box
    """
    return Box(box.lower + distance, box.upper - distance)


def intersect_boxes(b1: Box, b2: Box) -> Box:
    """Calculate the intersection of two boxes

    Args:
        b1 (Box): First box
        b2 (Box): Second box

    Returns:
        Box: The intersection region
    """
    return Box(np.maximum(b1.lower, b2.lower), np.minimum(b1.upper, b2.upper))


def check_box_intersection(b1: Box, b2: Box) -> bool:
    """Evaluate if two boxes intersect or not

    Args:
        b1 (Box): First box
        b2 (Box): Second box

    Returns:
        bool: True if the boxes intersect, False if not
    """
    l = np.maximum(b1.lower, b2.lower)
    u = np.minimum(b1.upper, b2.upper)
    return np.all(u >= l)


def is_in_box(point: npt.ArrayLike, box: Box) -> bool:
    """Evaluate if a point lies within a box

    Args:
        point (npt.ArrayLike): Point to evaluate, shape (box_dim,)
        box (Box): Box to test

    Returns:
        bool: True if the point is inside the bounds of the box, False otherwise
    """
    assert np.size(point) == box.dim
    return np.all(point >= box.lower) and np.all(point <= box.upper)


def find_containing_box(
    point: npt.ArrayLike, boxes: Union[list[Box], npt.ArrayLike]
) -> Optional[int]:
    """Find the index of the first box which contains a certain point

    Args:
        point (npt.ArrayLike): Point to evaluate
        boxes (Union[list[Box], npt.ArrayLike]): Boxes to search. If an array, must be of shape (n_boxes, 2, box_dim)

    Returns:
        Optional[int]: Index of the first box which contains the point. None if the point is not in any box
    """
    for i, box in enumerate(boxes):
        lower, upper = box
        if np.all(point >= lower) and np.all(point <= upper):
            return i
    return None


def find_containing_box_name(
    point: npt.ArrayLike, boxes: dict[str, Box]
) -> Optional[str]:
    """Find the name of the first box which contains a certain point

    Args:
        point (npt.ArrayLike): Point to evaluate
        boxes (dict[str, Box]): Boxes to search. Key/value: (box name) -> box
    Returns:
        Optional[str]: Name of the first box which contains the point. None if the point is not in any box
    """
    for name, box in boxes.items():
        if np.all(point >= box.lower) and np.all(point <= box.upper):
            return name
    return None


def visualize_3D_box(
    box: Union[Box, npt.ArrayLike],
    padding: Optional[npt.ArrayLike] = None,
    rgba: npt.ArrayLike = (1, 0, 0, 0.5),
) -> int:
    """Visualize a box in Pybullet

    Args:
        box (Union[Box, npt.ArrayLike]): Box to visualize. If an array, must be of shape (1, 2, box_dim)
        padding (Optional[npt.ArrayLike]): If expanding (or contracting) the boxes by a certain amount, include the
            (x, y, z) padding distances here (shape (3,)). Defaults to None.
        rgba (npt.ArrayLike): Color of the box (RGB + alpha), shape (4,). Defaults to (1, 0, 0, 0.5).

    Returns:
        int: Pybullet ID of the box
    """
    lower, upper = box
    if padding is not None:
        lower -= padding
        upper += padding
    return create_box(
        pos=(lower + (upper - lower) / 2),  # Midpoint
        orn=(0, 0, 0, 1),
        mass=0,
        sidelengths=(upper - lower),
        use_collision=False,
        rgba=rgba,
    )


def plot_2D_box(
    box: Box, ax: Optional[plt.Axes] = None, show: bool = True, *args, **kwargs
) -> plt.Axes:
    """Plots the boundary of a 2D box

    Args:
        box (Box): 2D box to plot
        ax (Optional[plt.Axes]): If re-using existing plotting axes, include them here. Defaults to None.
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plotting axes
    """
    assert box.dim == 2
    if ax is None:
        ax = plt.gca()
    pts = np.array(
        [
            box.lower,
            [box.lower[0], box.upper[1]],
            box.upper,
            [box.upper[0], box.lower[1]],
            box.lower,
        ]
    )
    ax.plot(*pts.T, *args, **kwargs)
    if show:
        plt.show()
    return ax


def compute_graph(boxes: dict[str, Box]) -> dict[str, list[str]]:
    """Computes the graph between a set of boxes

    Returns:
        dict[str, list[str]]: Adjacency list / graph dictating safe paths within the boxes. Key/value pair is:
            (name of the box) -> (list of names of all neighbors of that box)
    """
    names = list(boxes.keys())
    n = len(names)
    adj = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if check_box_intersection(boxes[names[i]], boxes[names[j]]):
                adj[names[i]].append(names[j])
                adj[names[j]].append(names[i])
    return adj


def check_box_containment(
    bounding_box: Union[Box, npt.ArrayLike], safe_set: list[Box]
) -> bool:
    """Determine if the bounding box of an object is fully contained within a safe set

    Args:
        bounding_box (Union[Box, npt.ArrayLike]): Axis-aligned bounding box. If ArrayLike, must be of shape (2, 3)
            e.g. the lower and upper XYZ values defining the box
        safe_set (list[Box]): Boxes to check containment

    Returns:
        bool: True if the bounding box is contained in the safe set, False otherwise
    """
    return any(
        np.all(bounding_box[0] > box.lower) and np.all(bounding_box[1] < box.upper)
        for box in safe_set
    )
