"""CVXPY-compatible math operations"""


from typing import Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt


def cross(
    a: Union[npt.ArrayLike, cp.Variable, cp.Expression],
    b: Union[npt.ArrayLike, cp.Variable, cp.Expression],
) -> Union[np.ndarray, cp.Expression]:
    """Evaluate the cross product in R3: a x b

    Args:
        a (Union[npt.ArrayLike, cp.Variable, cp.Expression]): First vector, shape (3,)
        b (Union[npt.ArrayLike, cp.Variable, cp.Expression]): Second vector, shape (3,)

    Returns:
        Union[np.ndarray, cp.Expression]: Cross product result, shape (3,)
    """
    return (
        (a[1] * b[2] - a[2] * b[1]) * np.array([1, 0, 0])
        + (a[2] * b[0] - a[0] * b[2]) * np.array([0, 1, 0])
        + (a[0] * b[1] - a[1] * b[0]) * np.array([0, 0, 1])
    )


def skew(
    v: Union[npt.ArrayLike, cp.Variable, cp.Expression]
) -> Union[np.ndarray, cp.Expression]:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (Union[npt.ArrayLike, cp.Variable, cp.Expression]): Vector to convert, shape (3,)

    Returns:
        Union[np.ndarray, cp.Expression]: (3, 3) skew-symmetric matrix
    """
    return (
        v[0] * np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        + v[1] * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        + v[2] * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    )
