"""Helper functions associated with coordinate transformations"""

import numpy as np
import numpy.typing as npt
import pytransform3d.coordinates as pc


def cartesian_to_spherical(p: npt.ArrayLike) -> np.ndarray:
    """Converts a cartesian (xyz) coordinate to spherical (radius, elevation, azimuth)

    - Note that the elevation angle is traditionally measured w.r.t the +Z axis
    - Radius >= 0
    - Elevation is defined between 0 and pi
    - Azimuth is defined between -pi and pi

    Args:
        p (npt.ArrayLike): XYZ cartesian coordinate to translate into spherical, shape (3,)

    Returns:
        np.ndarray: Spherical coordinates (radius, elevation, azimuth), shape (3,)
    """
    if len(p) != 3:
        raise ValueError(f"Invalid coordinate size (should be 3).\nGot:{p}")
    # Convert to float to avoid a bug in pytransform3d with integer arrays
    return pc.spherical_from_cartesian(np.float64(p))
