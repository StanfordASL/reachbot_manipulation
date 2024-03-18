"""Geometry, inertial properties, and other configuration properties of the simplified Reachbot URDF model"""

import numpy as np

from reachbot_manipulation.utils.transformations import make_transform_mat
from reachbot_manipulation.geometry.points import rectangular_prism_points
from reachbot_manipulation.utils.math_utils import normalize

# Based on values from URDF
LENGTH = 1
WIDTH = 1
HEIGHT = 1

# Transformations between the body frame and the cable locations: "Cable-to-body"
# TODO determine if the rotation component is meaningful (placeholder identity rotation for now)
_R = np.eye(3)
LOCAL_CABLE_POSITIONS = rectangular_prism_points((0, 0, 0), (LENGTH, WIDTH, HEIGHT), _R)
LOCAL_CABLE_TRANSFORMS = [make_transform_mat(_R, p) for p in LOCAL_CABLE_POSITIONS]

# TODO decide if the normals need to be better aligned with the boom-reachable areas?
LOCAL_CONE_NORMALS = normalize(LOCAL_CABLE_POSITIONS)

MIN_TENSION = 0  # N
MAX_TENSION = 30  # N

MAX_REACH = 20  # meters

# TODO refine these values
SPEED_LIMIT = 1  # m/s
ACCEL_LIMIT = 0.1  # m/s^2
ANGULAR_SPEED_LIMIT = 0.5  # rad/s
ANGULAR_ACCEL_LIMIT = 0.3  # rad/s^2

# If using booms
MAX_BOOM_SHOULDER_TORQUE = 1  # Nm
