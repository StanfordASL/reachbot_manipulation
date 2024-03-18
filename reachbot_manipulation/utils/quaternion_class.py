"""Class for managing quaternion conventions

Usage examples:
q = Quaternion() # This will initialize it as empty
q.xyzw = [0.1, 0.2, 0.3, 0.4] # This will assign the values after initialization
q = Quaternion(xyzw=[0.1, 0.2, 0.3, 0.4]) # This will assign values at initialization
some_pytransform3d_function(q.wxyz) # Pass the wxyz data into modules that use this convention

NOTE
- This class has been separated from the quaternion and rotation files to prevent circular imports
  (Do not import either into this file, or else this issue will pop up again)
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from reachbot_manipulation.utils.math_utils import normalize


class Quaternion:
    """Quaternion class to handle the XYZW/WXYZ conventions with less confusion

    We will always default to using XYZW convention

    Args:
        xyzw (npt.ArrayLike, optional): Quaternions, in XYZW order. Defaults to None, in which
            case wxyz should be provided, or else the quaternion will be empty
        wxyz (npt.ArrayLike, optional): Quaternions, in WXYZ order. Defaults to None, in which
            case xyzw should be provided, or else the quaternion will be empty

    Raises:
        ValueError: if both XYZW and WXYZ inputs are provided at instantiation
    """

    def __init__(
        self, xyzw: Optional[npt.ArrayLike] = None, wxyz: Optional[npt.ArrayLike] = None
    ):
        if xyzw is not None and wxyz is not None:
            raise ValueError("Specify one of XYZW/WXYZ, not both")
        elif xyzw is not None:
            self.xyzw = xyzw
        elif wxyz is not None:
            self.wxyz = wxyz
        else:
            self._initialize_as_empty()

    def _check_if_loaded(self):
        vals = [self.x, self.y, self.z, self.w]
        if any(val is None for val in vals):
            raise ValueError(
                f"Quaternion has been initialized, but not set (value is {vals})"
            )

    def _check_quat(self, quat):
        quat = np.ravel(quat)
        if len(quat) != 4:
            raise ValueError(f"Invalid quaternion ({quat}):\nNot of length 4!")
        return normalize(quat)

    def _initialize_as_empty(self):
        self.x, self.y, self.z, self.w = [None, None, None, None]

    # TODO need to decide if this is needed
    def normalize(self):
        self.xyzw = normalize(self.xyzw)

    @property
    def xyzw(self):
        """Quaternion expressed in XYZW format. Shape = (4,)"""
        self._check_if_loaded()
        return np.array([self.x, self.y, self.z, self.w])

    @property
    def wxyz(self):
        """Quaternion expressed in WXYZ format. Shape = (4,)"""
        self._check_if_loaded()
        return np.array([self.w, self.x, self.y, self.z])

    @xyzw.setter
    def xyzw(self, xyzw: npt.ArrayLike):
        """Sets the quaternion based on an array in XYZW form"""
        xyzw = self._check_quat(xyzw)
        self.x, self.y, self.z, self.w = xyzw

    @wxyz.setter
    def wxyz(self, wxyz: npt.ArrayLike):
        """Sets the quaternion based on an array in WXYZ form"""
        wxyz = self._check_quat(wxyz)
        self.w, self.x, self.y, self.z = wxyz

    @property
    def conjugate(self) -> np.ndarray:
        """Conjugate of the XYZW quaternion"""
        self._check_if_loaded()
        return np.array([-self.x, -self.y, -self.z, self.w])
