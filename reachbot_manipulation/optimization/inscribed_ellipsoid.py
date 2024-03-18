"""Maximum volume ellipsoids in polyhedra"""

# WORK IN PROGRESS

# TODO clean this up and implement it in a similar way as the inscribed ball

from typing import Union

import numpy as np
import numpy.typing as npt
import cvxpy as cp
import matplotlib.pyplot as plt

from reachbot_manipulation.optimization.cvxpy_problem import OptimizationProblem
from reachbot_manipulation.geometry.convex_hulls import polyhedron_hull
from reachbot_manipulation.geometry.plotting import plot_2d_hull
from reachbot_manipulation.geometry.ellipsoids import plot_2d_ellipse


class MaxVolumeEllipsoidInPolyhedron(OptimizationProblem):
    def __init__(
        self,
        A: np.ndarray,
        b: npt.ArrayLike,
        verbose: bool = True,
    ):
        self.A = A
        self.b = b
        self.m, self.n = A.shape
        self.B = cp.Variable((self.n, self.n))
        self.d = cp.Variable(self.n)
        self.scale = cp.Variable()
        # Construct the problem
        super().__init__(verbose)

    @property
    def objective(self) -> Union[cp.Maximize, cp.Minimize]:
        return cp.Maximize(cp.log_det(self.B))

    @property
    def constraints(self) -> list[cp.Expression]:
        return [
            self.A @ self.d + cp.norm(self.A @ self.B, axis=1) <= self.b,
        ]

    def solve(self) -> None:
        super().solve()

    @property
    def optimal_ellipse_params(self) -> tuple[np.ndarray, np.ndarray]:
        if self.B is None or self.d is None:
            raise ValueError("Cannot return the ellipse, problem has not been solved")
        return self.B.value, self.d.value


# https://math.stackexchange.com/questions/1092576/scale-ellipsoid-maximally-within-polyhedron
def scale_ellipsoid_in_polyhedron(ellipsoid: np.ndarray, A: np.ndarray, b: np.ndarray):
    B_inv = np.linalg.inv(ellipsoid)
    xTAx = np.min([(b[i] ** 2) / (A[i].T @ B_inv @ A[i]) for i in range(len(b))])
    return ellipsoid * (1 / xTAx)


def _test_scaled():
    np.random.seed(0)
    n = 8
    thetas = np.linspace(0, 2 * ((n - 1) / n) * np.pi, n)
    x = np.cos(thetas) + np.random.randn(n) * 0.1
    y = np.sin(thetas) + np.random.randn(n) * 0.1
    A = np.column_stack([x, y])
    b = np.ones(A.shape[0])
    hull = polyhedron_hull(A, b)
    B = np.diag([8, 1])
    new_ellipse = scale_ellipsoid_in_polyhedron(B, A, b)
    ax = plot_2d_hull(hull, show=False)
    plot_2d_ellipse(new_ellipse)
    plt.show()


def plot_affine_transform_ellipse(B, d):
    # Ellipse = {Bu + d | norm(u) <= 1}
    n = 100
    thetas = np.linspace(0, 2 * np.pi, n)
    x = np.cos(thetas)
    y = np.sin(thetas)
    pts = np.column_stack([x, y])
    transformed = pts @ B.T + d
    plt.plot(*transformed.T)


def _test_mve():
    np.random.seed(0)
    n = 8
    thetas = np.linspace(0, 2 * ((n - 1) / n) * np.pi, n)
    x = np.cos(thetas) + np.random.randn(n) * 0.1
    y = np.sin(thetas) + np.random.randn(n) * 0.1
    A = np.column_stack([x, y])
    b = np.ones(A.shape[0])
    hull = polyhedron_hull(A, b)
    prob = MaxVolumeEllipsoidInPolyhedron(A, b)
    prob.solve()
    ax = plot_2d_hull(hull, show=False)
    # plot_2d_ellipse(prob.B.value)
    B, d = prob.optimal_ellipse_params
    plot_affine_transform_ellipse(B, d)
    plt.show()


if __name__ == "__main__":
    _test_scaled()
    _test_mve()
    # _test_3d()
