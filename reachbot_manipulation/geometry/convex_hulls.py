"""Tools for operations with convex hulls"""

import itertools

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull


def hull_to_matrices(hull: ConvexHull) -> tuple[np.ndarray, np.ndarray]:
    """Determine the matrices A, b from a polyhedron convex hull such that Ax <= b

    Args:
        hull (ConvexHull): Convex hull defining a polyhedron

    Returns:
        tuple[np.ndarray, np.ndarray]:
            np.ndarray: A matrix, s.t. Ax <= b. Shape (m, n)
            np.ndarray: b array, s.t. Ax <= b. Shape (m,)
    """
    eqs = hull.equations
    return eqs[:, :-1], -1 * eqs[:, -1]


def polyhedron_hull(A: np.ndarray, b: np.ndarray) -> ConvexHull:
    """Convex hull of a polyhedron defined by Ax <= b

    Note: This can be inefficient to solve for high dimensions

    Args:
        A (np.ndarray): Array defining hyperplane normals, shape (n_planes, 3)
        b (np.ndarray): Array defining hyperplane offset, shape (n_planes)

    Returns:
        ConvexHull: Convex hull of the vertices of the polyhedron
    """
    n, dim = A.shape
    # Determine all possible intersections of hyperplanes and look for the corresponding vertex
    combos = itertools.combinations(range(n), dim)
    verts = []
    eps = 1e-12  # Tolerance
    for combo in combos:
        Ai = A[combo, :]
        bi = b[[combo]].reshape(-1)
        try:
            v = np.linalg.solve(Ai, bi)
        except np.linalg.LinAlgError:
            # This can occur if the Ai matrix is singular, in which case there is no vertex solution
            continue
        # Validate that this vertex solution is correct and  inside / on the boundary of the polyhedron
        is_vertex = np.linalg.norm(Ai @ v - bi) <= eps
        if not is_vertex:
            continue
        is_in_polyhedron = np.all(A @ v - b <= eps * np.ones_like(b))
        if is_in_polyhedron:
            verts.append(v)
    # Not all intersections of planes will be on the hull of the polyhedron, so use ConvexHull to manage this
    return ConvexHull(verts)


def intersect_polyhedra(
    A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the intersection of two polyhedra defined by Ax <= b and Cx <= d

    Args:
        A (np.ndarray): A, such that Ax <= b. Shape (m, n)
        b (np.ndarray): b, such that Ax <= b. Shape (m,)
        C (np.ndarray): C, such that Cx <= d. Shape (p, n)
        d (np.ndarray): d, such that Cx <= d. Shape (p,)

    Returns:
        tuple[np.ndarray, np.ndarray]: Intersection of the two polyhedra
            np.ndarray: F, such that Fx <= g. Shape (m + p, n)
            np.ndarray: g, such that Fx <= g. Shape (m + p)
    """
    return np.row_stack([A, C]), np.concatenate([b, d])


def minkowski(polytopes: list[np.ndarray], refine: bool = False) -> ConvexHull:
    """Determine the Minkowski sum between a set of polytopes

    P ⊕ Q = {x + y | x ∈ P, y ∈ Q}

    Args:
        polytopes (list[np.ndarray]): List of polytopes to sum. Polytopes are defined by a set
            of vectors defining a convex hull. Each polytope has shape (n_vectors_in_hull, dim),
            where n_vectors_in_hull can vary between polytopes, but dim is constant
        refine (bool, optional): Whether to refine the set of vectors in each polytope to ensure that it is a minimal
            set defining just the convex hull of the set. This can dramatically improve computation time, if the inputs
            are not guaranteed to be a minimal representation. Defaults to False

    Returns:
        ConvexHull: Minkowski sum, expressed as a convex hull
    """

    # Refine the vector sets if not necessarily a minimal representation of the convex hull
    if refine:
        new_polytopes = []
        for p in polytopes:
            hull = ConvexHull(p)
            new_polytopes.append(np.atleast_2d(hull.points[hull.vertices]))
        polytopes = new_polytopes
    else:
        polytopes = [np.atleast_2d(p) for p in polytopes]

    # Determine how many vectors define the convex hull for each polytope
    lens = [p.shape[0] for p in polytopes]
    # Form an array to enumerate all of the possible sums between the polytope vector sets
    # Each row will contain indices to select from each set:
    # i.e. combo_arr[i] = [3, 0, 1] means sum vector 3 from set 0, vector 0 from set 1, and vector 1 from set 2
    combo_grid = np.meshgrid(*[np.arange(l) for l in lens])
    combo_arr = np.column_stack([np.ravel(g) for g in combo_grid])
    dim = 1 if np.isscalar(polytopes[0][0]) else len(polytopes[0][0])
    n_sums, n_polytopes = combo_arr.shape
    sums = np.zeros((n_sums, dim))
    # Calculate all possible sums between the hull vectors
    for i in range(n_sums):
        sums[i] = np.sum(
            [polytopes[j][combo_arr[i, j]] for j in range(n_polytopes)], axis=0
        )
    return ConvexHull(sums)


def _test_minkowski():
    # pylint: disable=import-outside-toplevel
    from reachbot_manipulation.geometry.plotting import plot_2d_hull, plot_3d_hull
    from reachbot_manipulation.geometry.points import (
        tetrahedron_points,
        triangle_points,
    )

    # 2D Example
    set_1 = triangle_points()
    set_2 = triangle_points(rotation=np.diag([1, -1]))
    hull_1 = ConvexHull(set_1)
    hull_2 = ConvexHull(set_2)
    hull_3 = minkowski([set_1, set_2])
    ax = plot_2d_hull(hull_1, show=False)
    ax = plot_2d_hull(hull_2, ax=ax, show=False)
    ax = plot_2d_hull(hull_3, ax=ax, show=False)

    # 3D Example
    set_1 = tetrahedron_points()
    set_2 = tetrahedron_points(rotation=np.diag([1, -1, -1]))
    hull_1 = ConvexHull(set_1)
    hull_2 = ConvexHull(set_2)
    hull_3 = minkowski([set_1, set_2])
    ax = plot_3d_hull(hull_1, show=False)
    ax = plot_3d_hull(hull_2, ax=ax, show=False)
    ax = plot_3d_hull(hull_3, ax=ax, show=True)


def _test_intersection():
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt
    from reachbot_manipulation.geometry.plotting import plot_3d_hull
    from reachbot_manipulation.geometry.points import (
        tetrahedron_points,
        cube_points,
    )

    hull_1 = ConvexHull(tetrahedron_points())
    hull_2 = ConvexHull(cube_points())

    A1, b1 = hull_to_matrices(hull_1)
    A2, b2 = hull_to_matrices(hull_2)
    A3, b3 = intersect_polyhedra(A1, b1, A2, b2)
    hull_3 = polyhedron_hull(A3, b3)

    fig = plt.figure()
    ax_1 = fig.add_subplot(131, projection="3d")
    ax_2 = fig.add_subplot(132, projection="3d")
    ax_3 = fig.add_subplot(133, projection="3d")
    for ax in [ax_1, ax_2, ax_3]:
        ax.axes.set_xlim3d(-1, 1)
        ax.axes.set_ylim3d(-1, 1)
        ax.axes.set_zlim3d(-1, 1)
    plot_3d_hull(hull_1, ax_1, show=False)
    plot_3d_hull(hull_2, ax_2, show=False)
    plot_3d_hull(hull_3, ax_3, show=False)
    plt.show()


if __name__ == "__main__":
    _test_minkowski()
    _test_intersection()
