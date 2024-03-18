"""Detemining the maximum radius inscribe ball in a polyhedron

This is useful for grasp metrics similar to Ferrari-Canny
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import cvxpy as cp

from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.errors import OptimizationError
from reachbot_manipulation.optimization.cvxpy_problem import OptimizationProblem
from reachbot_manipulation.geometry.convex_hulls import polyhedron_hull
from reachbot_manipulation.geometry.plotting import (
    plot_2d_hull,
    plot_3d_hull,
    plot_circle,
    plot_sphere,
)


def largest_ball_in_polyhedron(
    A: np.ndarray, b: npt.ArrayLike, center: Optional[npt.ArrayLike] = None
) -> tuple[np.ndarray, float]:
    """Determine the largest ball inside a polyhedron defined by Ax <= b

    Args:
        A (np.ndarray): Array defining hyperplane normals, shape (n_planes, dim)
        b (npt.ArrayLike): Array defining hyperplane offset, shape (n_planes)
        center (Optional[npt.ArrayLike]): If fixing the center of the ball, include its position here.
            Defaults to None (determine the optimial center of the ball as well)

    Returns:
        tuple[np.ndarray, float]:
            np.ndarray: Center of the ball, shape (dim,)
            float: Radius of the ball
    """
    prob = InscribedBallProblem(A, b, center, verbose=False)
    prob.solve()
    return prob.optimal_center, prob.optimal_radius


# TODO: This is set up for resolves with the center as a parameter. Is it worthwhile to make a version with the
# polyhedron (A, b) parameterized?
class InscribedBallProblem(OptimizationProblem):
    """Determine the largest ball inside a polyhedron defined by Ax <= b

    Args:
        A (np.ndarray): Array defining hyperplane normals, shape (n_planes, dim)
        b (npt.ArrayLike): Array defining hyperplane offset, shape (n_planes)
        center (Optional[npt.ArrayLike]): If fixing the center of the ball, include its position here.
            Defaults to None (determine the optimal center of the ball as well)
        verbose (bool, optional): Whether to print info about the optimization problem after it is solved.
            Defaults to True
    """

    def __init__(
        self,
        A: np.ndarray,
        b: npt.ArrayLike,
        center: Optional[npt.ArrayLike] = None,
        verbose: bool = True,
    ):
        n_hyperplanes, dim = A.shape
        b = np.ravel(b)
        assert len(b) == n_hyperplanes
        self.variable_center = center is None
        if self.variable_center:
            center = cp.Variable(dim)
        else:
            center = np.ravel(center)
            assert len(center) == dim
            center = cp.Parameter(dim, value=center)

        self.center = center
        self.r = cp.Variable()
        self.A = A
        self.b = b
        # Construct the problem
        super().__init__(verbose)

    @property
    def objective(self) -> Union[cp.Maximize, cp.Minimize]:
        return cp.Maximize(self.r)

    @property
    def constraints(self) -> list[cp.Expression]:
        return [
            self.A @ self.center + self.r * cp.norm(self.A, axis=1) <= self.b,
            self.r >= 0,
        ]

    def solve(self) -> None:
        try:
            super().solve(solver=cp.ECOS)
        except OptimizationError as e:
            info = (
                "The polyhedron is likely poorly defined"
                if self.variable_center
                else "The polyhedron likely does not contain the center point"
            )
            raise OptimizationError(info) from e

    @property
    def optimal_center(self) -> np.ndarray:
        """Optimal center of the ball, shape (dim,)"""
        # Note: this is valid if center is a Variable or Parameter
        if self.center.value is None:
            raise ValueError("Cannot return the center, problem has not been solved")
        return self.center.value

    @property
    def optimal_radius(self) -> float:
        """Optimal radius of the ball"""
        if self.r.value is None:
            raise ValueError("Cannot return the radius, problem has not been solved")
        return self.r.value

    def update_center(self, new_center: npt.ArrayLike) -> None:
        """Update the center of the ball (if the center is parameterized, and not a variable)

        Args:
            new_center (npt.ArrayLike): New center, shape (dim,)
        """
        if self.variable_center:
            raise ValueError("Center is not a Parameter")
        new_center = np.ravel(new_center)
        if new_center.shape != self.center.shape:
            raise ValueError("Shape mismatch, cannot update the parameter")
        self.center.value = new_center


def _test_3d():
    # pylint: disable=import-outside-toplevel
    from reachbot_manipulation.geometry.points import cube_points

    A = normalize(cube_points())
    b = np.ones(A.shape[0])
    prob = InscribedBallProblem(A, b, center=(0, 0, 0))
    prob.solve()
    center = prob.optimal_center
    radius = prob.optimal_radius
    hull = polyhedron_hull(A, b)
    ax = plot_3d_hull(hull, show=False)
    plot_sphere(center, radius, ax=ax, show=True)


def _test_2d():
    n = 8
    thetas = np.linspace(0, 2 * ((n - 1) / n) * np.pi, n)
    x = np.cos(thetas)
    y = np.sin(thetas)
    A = np.column_stack([x, y])
    b = np.ones(A.shape[0])
    hull = polyhedron_hull(A, b)
    prob = InscribedBallProblem(A, b, center=(0, 0))
    prob.solve()
    center = prob.optimal_center
    radius = prob.optimal_radius
    ax = plot_2d_hull(hull, show=False)
    plot_circle(center, radius, ax=ax)


if __name__ == "__main__":
    _test_2d()
    _test_3d()
