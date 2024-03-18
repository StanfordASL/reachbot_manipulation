"""Determining the maximal vector in a convex hull vector space"""

# TODO add a minkowski/Linf version of this?
# If so, the sum(a) == 1 constraint will have to be changed to a <= 1
# and the scipy ConvexHull would be replaced with a Minkowski hull

# TODO rename the "a" and "b" variables

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import cvxpy as cp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from reachbot_manipulation.geometry.plotting import plot_2d_hull
from reachbot_manipulation.utils.errors import OptimizationError
from reachbot_manipulation.utils.cvxpy_utils import print_problem_info


class MaxDistanceInHullProblem:
    """Determine the maximum distances along a set of directions inside a convex hull

    Args:
        vector_space (np.ndarray): Points in/on the convex hull (may or may not be vertices of the hull),
            shape (n_points, dim)
        directions (np.ndarray): Directions to analyze, shape (n_directions, dim)
        center (Optional[np.ndarray]): Center point to measure from. Defaults to None (use the origin)
        verbose (bool, optional): Whether to print info about the optimization problem after solving.
            Defaults to True
    """

    def __init__(
        self,
        vector_space: np.ndarray,
        directions: np.ndarray,
        center: Optional[npt.ArrayLike] = None,
        verbose: bool = True,
    ):
        directions = np.atleast_2d(directions)
        self.vector_space = np.atleast_2d(vector_space)
        n, dim = directions.shape
        n_verts, dim_2 = self.vector_space.shape
        assert dim == dim_2
        if center is None:
            center = np.zeros(dim)
        else:
            center = np.atleast_2d(center)
            assert center.shape == (1, dim)
        self.verbose = verbose
        self.center = cp.Parameter(center.shape, value=center)
        self.directions = cp.Parameter(directions.shape, value=directions)
        self.a = cp.Variable((n, n_verts))
        self.b = cp.Variable((n, 1))
        self.prob = cp.Problem(self.objective, self.constraints)

    @property
    def objective(self) -> Union[cp.Minimize, cp.Maximize]:
        """Optimization objective"""
        return cp.Maximize(cp.sum(self.b))

    @property
    def constraints(self) -> list[cp.Expression]:
        """Optimization constraints"""
        return [
            self.a >= 0,
            self.b >= 0,
            cp.sum(self.a, axis=1) == 1,
            self.a @ self.vector_space
            == cp.multiply(self.b, self.directions) + self.center,
        ]

    def solve(self) -> None:
        """Solves the optimization problem"""
        self.prob.solve(solver=cp.ECOS)
        if self.prob.status != cp.OPTIMAL:
            raise OptimizationError(
                "Could not find a solution\n" + f"Problem status: {self.prob.status}"
            )
        if self.verbose:
            print_problem_info(self.prob)

    @property
    def optimal_distances(self) -> np.ndarray:
        """Optimal distances along each direction, shape (n_directions,)"""
        if self.b.value is None:
            raise ValueError("Cannot return the distances, problem has not been solved")
        return np.ravel(self.b.value)

    def update_center(self, new_center: npt.ArrayLike) -> None:
        """Update the center point parameter

        Args:
            new_center (npt.ArrayLike): New center point, shape (dim,)
        """
        new_center = np.atleast_2d(new_center)
        if new_center.shape != self.center.shape:
            raise ValueError("Shape mismatch, cannot update the parameter")
        self.center.value = new_center

    def update_directions(self, new_directions: npt.ArrayLike) -> None:
        """Update the directions parameter

        Args:
            new_directions (npt.ArrayLike): New directions, shape (n_directions, dim)
        """
        new_directions = np.atleast_2d(new_directions)
        if new_directions.shape != self.directions.shape:
            raise ValueError("Shape mismatch, cannot update the parameter")
        self.directions.value = new_directions


def main():
    np.random.seed(3)
    space = np.random.randn(10, 2)
    directions = np.row_stack([np.eye(2), -np.eye(2)])
    center = np.array([0, 0])
    prob = MaxDistanceInHullProblem(space, directions, center)
    prob.solve()
    distances = prob.optimal_distances
    vectors = distances.reshape(-1, 1) * directions
    ax = plot_2d_hull(ConvexHull(space), show=False)
    points = np.row_stack([[center, center + vi] for vi in vectors])
    plt.plot(*points.T)
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
