"""Metrics for determining the quality of a dexterous grasp"""


import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull

from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.geometry.convex_hulls import hull_to_matrices


def minimum_singular_value(grasp_matrix: np.ndarray) -> float:
    """Minimum singular value of the grasp matrix

    This is a quality measure that indicates how far the grasp configuration is from falling into a singular
    configuration (Li and Sastry 1988). 0 if singular

    Args:
        grasp_matrix (np.ndarray): Grasp matrix, shape (6, 3 * n_contacts)

    Returns:
        float: Minimum singular value
    """
    # Singular values (second svd output) are sorted fom largest to smallest in numpy
    return np.linalg.svd(grasp_matrix)[1][-1]


def grasp_ellipsoid_volume(grasp_matrix: np.ndarray) -> float:
    """Volume of the grasp ellipsoid (AKA: Product of singular values)

    This ellipsoid can be considered as a unit sphere in the contact force space, mapped into the wrench space by the
    grasp matrix.The global contribution of all the contact forces can be considered using the volume of this ellipsoid
    as the quality measure (Li and Sastry 1988)

    Args:
        grasp_matrix (np.ndarray): Grasp matrix, shape (6, 3 * n_contacts)

    Returns:
        float: Ellipsoid volume (or alternatively, product of singular values)
    """
    # Alternative method: np.sqrt(np.linalg.det(G @ G.T))
    return np.prod(np.linalg.svd(grasp_matrix)[1])


def grasp_isotropy_index(grasp_matrix: np.ndarray) -> float:
    """Ratio between the max/min singular values of the grasp matrix

    This criterion looks for a uniform contribution of the contact forces to the total wrench applied on the object,
    i.e. it tries to obtain an isotropic grasp where each applied contact force contributes to the object's internal
    forces in a similar way

    Args:
        grasp_matrix (np.ndarray): Grasp matrix, shape (6, 3 * n_contacts)

    Returns:
        float: Isotropy index
    """
    s = np.linalg.svd(grasp_matrix)[1]
    # Ratio between the min and max singular values
    # Note numpy sorts the singular values in descending order
    return s[-1] / s[0]


def grasp_wrench_space_volume(wrench_space: ConvexHull) -> float:
    """Volume of the grasp wrench space

    Args:
        wrench_space (ConvexHull): L1 or Linf grasp wrench space

    Returns:
        float: Volume
    """
    return wrench_space.volume


def ferrari_canny(wrench_space: ConvexHull) -> float:
    """Ferrari-Canny metric: Radius of the largest inscribed ball about the origin in the wrench space

    Args:
        wrench_space (ConvexHull): L1 or Linf wrench space

    Returns:
        float: Ball radius, if in force closure. If not in force closure, -1
    """
    # Scipy normalizes the A matrix, so the b vectors are the distances to the faces defined by the normals in A
    b = hull_to_matrices(wrench_space)[1]
    b_min = np.min(b)
    return b_min if b_min >= 0 else -1


def is_in_force_closure(wrench_space: ConvexHull) -> bool:
    """Determines whether a wrench space contains the origin, to see if it is in force closure

    Args:
        wrench_space (ConvexHull): L1 or Linf wrench space

    Returns:
        bool: True if in force closure, False otherwise
    """
    return ferrari_canny(wrench_space) > 0
