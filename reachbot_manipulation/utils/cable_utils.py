"""Utilities for general modeling of cables

This does not involve the kinematics of cable-driven parallel robots. See control/ for more details there
"""

import numpy as np
import numpy.typing as npt


# Steel cable seems to have more of an "effective elastic modulus" due to the braiding. Normally steel is around 210 GPa
# but a general order-of-magnitude estimate of 100 GPa seems realistic for steel wire
# See "Mechanics model and its equation of wire rope based on elastic thin rod theory"
# Fishing line can be made from a few different materials and can be monofilament or braided. A few different sources
# seem to indicate that its elastic modulus is around 1.4 GPa. We'll assume we use nylon line
EFFECTIVE_ELASTIC_MODULI = {"steel": 100e9, "nylon": 1.4e9}


def cable_force(strain: npt.ArrayLike, radius: float, material: str) -> npt.ArrayLike:
    """Force induced by a strain on a cable. Positive strain (tension) only

    Args:
        strain (npt.ArrayLike): % elongation(s)
        radius (float): Radius of the cable, in meters
        material (str): Name of the cable material: i.e. "steel", "nylon"

    Returns:
        npt.Arraylike: Force magnitude(s), in Newtons
    """
    try:
        elastic_modulus = EFFECTIVE_ELASTIC_MODULI[material]
    except KeyError as e:
        raise ValueError(f"Unrecognized cable material: {material}") from e
    return np.pi * radius**2 * np.maximum(strain, 0) * elastic_modulus


def cable_strain(force: npt.ArrayLike, radius: float, material: str) -> npt.ArrayLike:
    """Strain induced by a force on a cable. Positive force (tension) only

    Args:
        force (npt.ArrayLike): Force(s) on a cable, in Newtons
        radius (float): Radius of the cable, in meters
        material (str): Name of the cable material: i.e. "steel", "nylon"

    Returns:
        npt.ArrayLike: % elongation(s)
    """
    try:
        elastic_modulus = EFFECTIVE_ELASTIC_MODULI[material]
    except KeyError as e:
        raise ValueError(f"Unrecognized cable material: {material}") from e
    return np.maximum(force, 0) / (np.pi * radius**2 * elastic_modulus)


def tension_to_rgb(tension: npt.ArrayLike, limits: tuple[float, float]) -> np.ndarray:
    """Convert tension force(s) to an RGB value to help visualize the scale within tensile limits

    Args:
        tension (npt.ArrayLike): Tension force(s), in Newtons
        limits (tuple[float, float]): (min, max) tensions, in Newtons

    Returns:
        np.ndarray: RGB values, shape (n_forces, 3)
    """
    tension = np.ravel(tension)
    n = tension.shape[0]
    tension_pct = (tension - limits[0]) / (limits[1] - limits[0])
    t = np.clip(tension_pct, 0, 1)
    return np.column_stack([t, 1 - t, np.zeros(n)])
