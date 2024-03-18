"""Assorted helper functions related to math operations / linear algebra"""

from typing import Union, Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def normalize(vec: npt.ArrayLike) -> np.ndarray:
    """Normalizes a vector to have magnitude 1

    If normalizing an array of vectors, each vector will have magnitude 1

    Args:
        vec (npt.ArrayLike): Input vector or array. Shape (dim,) or (n_vectors, dim)

    Returns:
        np.ndarray: Unit vector(s), shape (dim,) or (n_vectors, dim) (same shape as the input)
    """
    vec = np.atleast_1d(vec)
    # Single vector, shape (dim,)
    if np.ndim(vec) == 1:
        norm = np.linalg.norm(vec)
        if abs(norm) < 1e-12:
            raise ZeroDivisionError("Cannot normalize the vector, it has norm 0")
        return vec / norm
    # Array of vectors, shape (n, dim) or something more complex (take norm over last axis)
    else:
        norms = np.linalg.norm(vec, axis=-1)
        if np.any(np.abs(norms) <= 1e-12):
            raise ZeroDivisionError(
                "Cannot normalize the array, at least 1 vector has norm 0"
            )
        return vec / norms[..., np.newaxis]


def is_diagonal(array: npt.ArrayLike) -> bool:
    """Checks if an array is diagonal or not

    Args:
        array (npt.ArrayLike): The array to check

    Returns:
        bool: True if the array is diagonal, False otherwise
    """
    return np.array_equal(array, np.diag(np.diag(array)))


def safe_divide(
    a: npt.ArrayLike, b: npt.ArrayLike, fill: str = "original", tol: float = 1e-8
) -> npt.ArrayLike:
    """Calculates a/b while handling division by zero

    Args:
        a (npt.ArrayLike): Dividend
        b (npt.ArrayLike): Divisor (which may contain zeros)
        fill (str, optional): How to fill the result where the divisor = 0. One of "original" (a/0 = a),
            "zero" (a/0 = 0), "nan" (a/0 = nan), "inf" (a/0 = inf). Defaults to "original".
        tol (float, optional): Tolerance for determining float equality with 0. Defaults to 1e-8.

    Returns:
        npt.ArrayLike: Division result
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    fill = fill.lower()
    if fill == "original":
        out = a
    elif fill == "zero":
        out = np.zeros_like(a)
    elif fill == "nan":
        out = np.full(a.shape, np.nan)
    elif fill == "inf":
        # Account for sign of dividend for positive or negative inf
        out = np.array((np.sign(a) + (a == 0)) * np.inf)
    else:
        raise ValueError(f"Invalid fill type: {fill}")
    return np.divide(a, b, out=out, where=np.abs(b) >= tol)


def skew(v: npt.ArrayLike) -> np.ndarray:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (npt.ArrayLike): Vector to convert, shape (3,)

    Returns:
        np.ndarray: (3, 3) skew-symmetric matrix
    """
    v = np.ravel(v)
    if len(v) != 3:
        raise ValueError(f"Vector needs to be of length 3.\nGot: {v}")
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def unskew(S: npt.ArrayLike) -> np.ndarray:
    """Gives the vector associated with a skew-symmetric matrix S such that skew(vec) = S

    Args:
        S (npt.ArrayLike): Skew-symmetric 3x3 matrix

    Returns:
        np.ndarray: "Unskewed" vector, shape (3,)
    """
    S = np.asarray(S)
    if S.shape != (3, 3):
        raise ValueError(f"S must be a 3x3 matrix. Got shape: {S.shape}")
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def is_skew(S: np.ndarray) -> bool:
    """Checks that a square matrix is skew-symmetric

    Args:
        S (np.ndarray): Matrix to check

    Returns:
        bool: True if skew-symmetric, False otherwise
    """
    return np.allclose(S, -S.T)


def is_special_orthogonal(R: np.ndarray) -> bool:
    """Checks that a square matrix is in the special orthogonal group

    Args:
        R (np.ndarray): Matrix to check

    Returns:
        bool: True if special orthogonal (such as a rotation matrix), False otherwise
    """
    m, n = R.shape
    if m != n:
        return False
    return np.allclose(R @ R.T, np.eye(n)) and np.isclose(np.linalg.det(R), 1)


def spherical_vonmises_sampling(
    mu: npt.ArrayLike,
    kappa: Union[float, npt.ArrayLike],
    n_pts: int,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Samples points on a (generalized) sphere based on the von Mises distribution

    This is slightly more relevant to sampling on a sphere than a Gaussian because the
    distribution is circular in nature. Technically, the von Mises-Fisher distribution is for
    a sphere but Numpy doesn't seem to distinguish between these.

    Args:
        mu (npt.ArrayLike): Mean. Length defines the dimension of the sphere. Should be normalized
        kappa (Union[float, npt.ArrayLike]): Concentration parameter. This can be thought of as the
            "inverse of variance". Large kappa results in a distribution that approaches a gaussian;
            small kappa approaches a uniform distribution. As a quick not-at-all precise measure,
            kappa = 0: Uniform distribution around the sphere
            kappa = 5: Angular dispersion of ~90 degrees from the mean
            kappa = 10: Angular dispersion of ~60 degrees from the mean
            kappa = 50: Angular dispersion of ~15 degrees from the mean
        n_pts (int): Number of points to sample
        rng (Optional[Union[int, np.random.Generator]]): Random number generator. Defaults to None (use np.random)

    Returns:
        np.ndarray: Sampled points, shape (n_pts, dimension)
    """
    rng = np.random.default_rng(rng)
    mu = np.atleast_1d(mu)
    kappa = np.atleast_1d(kappa)
    sampled = rng.vonmises(mu, kappa, size=(n_pts, len(mu)))
    return normalize(sampled)


def orthonormal_basis_from_reference(v: npt.ArrayLike) -> np.ndarray:
    """Construct an orthonormal basis from a reference vector (or set of vectors), using Gram-Schmidt.

    The first vector in the basis will be parallel with the first reference vector, and following
    vectors in the basis will be constructed via their component orthogonal to the previous vectors

    The first vector in the basis MAY be the opposite direction as the first reference vector

    Args:
        v (npt.ArrayLike): Reference vector, or vectors. Shape (dim,) or (n_vectors, dim)

    Returns:
        np.ndarray: Orthonormal basis, shape (dim, dim).
    """
    # Convert to column vectors
    v = np.atleast_2d(v).T
    dim = v.shape[0]
    # Augment the vector or vectors with a known linearly independent spanning set
    A = np.column_stack([v, np.eye(dim)])
    # The Q matrix in QR is the result of the Gram-Schmidt process
    Q = np.linalg.qr(A)[0]
    # Convert back to row vectors for easier Numpy indexing
    return Q.T


def one_sided_confidence_bound(
    pct: float, mean: float, stdev: float, side="lower"
) -> float:
    """Determine a lower or upper one-sided confidence bound on a Gaussian distribution

    >>> x = one_sided_confidence_bound(0.9, 0, 1, "lower")
    -1.2815515655446004

    i.e. 90% of the distribution lies above this lower bound, x

    Args:
        pct (float): Percent confidence, between 0 and 1
        mean (float): Mean of the Gaussian
        stdev (float): Standard deviation of the Gaussian
        side (str, optional): Whether to compute a lower or upper bound. Defaults to "lower".

    Returns:
        float: The value x such that the given percent of the distribution lies above (for a lower bound)
            or below (for an upper bound) this x.
    """
    if not 0 < pct < 1:
        raise ValueError("Percent confidence must be between 0 and 1")
    if side not in {"lower", "upper"}:
        raise ValueError("Side must be 'lower' or 'upper'")
    if side == "lower":
        return norm.ppf(1 - pct, loc=mean, scale=stdev)
    else:  # Upper
        return norm.ppf(pct, loc=mean, scale=stdev)
