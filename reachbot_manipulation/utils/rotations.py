"""Helper functions associated with rotations

Notation for rotation matrices:
A
  R
B
describes the orientation of frame B with respect to frame A. AKA "B in A", or "B to A"
In code, this can be described as R_B_in_A or R_B2A

The columns of this matrix are the unit vectors of B expressed in the coordinates A
The rows of this matrix are the unit vectors of A expressed in the coordinates of B

Operator vs mapping example, with Rx(theta)
As an operator: Rx @ P will rotate point P by theta about the X axis
As a mapping: Gives R_B_in_A (R_B2A), where frame B is rotated theta radians about A's X axis

Composing rotations:
A       A     B     C
  R  =    R     R     R
D       B     C     D
e.g. R_D2A = R_B2A @ R_C2B @ R_D2C

Conventions:
- Euler angles are in XYZ convention
    - Euler angles correspond to "intrinsic" convention in pytransform3d
- Fixed angles are also XYZ convention
    - Fixed angles correspond to "extrinsic" convention in pytransform3d
- Quaternions are in XYZW
    - Who uses XYZW?
        - NASA
        - ROS
        - Bullet/Pybullet
    - Who uses WXYZ?
        - Pytransform3d
        - Mujoco
    - We'll need to convert XYZW -> WXYZ for pytransform3d

All angles are in radians
"""
# TODO
# - Decide if we want to get rid of the _xyz in the euler namings
# - make a new function for custom euler angle conventions
# - Make sure naming conventions are consistent
# - Finish any NotImplemented functions
# - Decide if the singularity exceptions from the old code should be re-included
# - Update the documentation anywhere pytransform3d changed something
# - Determine if the matrix checks should raise an exception or just return true/false
# - Decide if it's useful to add back in other conventions (like zyx)? - Removed for now
# - Make the angles inputs an array rather than multiple inputs
# - Make functions to add deltas to the orientation

from typing import Union

import numpy as np
import numpy.typing as npt
import pytransform3d.rotations as rt

from reachbot_manipulation.utils.quaternion_class import Quaternion

# Parameters to clarify meaning of pytransform3d inputs
_EULER = 0
_FIXED = 1
_X = 0
_Y = 1
_Z = 2


def Rx(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the X axis

    Args:
        theta (float): Rotation angle about the X axis (radians)

    Returns:
        np.ndarray: The associated (3,3) rotation matrix
    """
    return rt.matrix_from_euler([theta, 0, 0], _X, _Y, _Z, _EULER)


def Ry(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the Y axis

    Args:
        theta (float): Rotation angle about the Y axis (radians)

    Returns:
        np.ndarray: The associated (3,3) rotation matrix
    """
    return rt.matrix_from_euler([0, theta, 0], _X, _Y, _Z, _EULER)


def Rz(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the Z axis

    Args:
        theta (float): Rotation angle about the Z axis (radians)

    Returns:
        np.ndarray: The associated (3,3) rotation matrix
    """
    return rt.matrix_from_euler([0, 0, theta], _X, _Y, _Z, _EULER)


def euler_xyz_to_rmat(angles: npt.ArrayLike) -> np.ndarray:
    """Euler XYZ 3-angle rotation matrix

    Operations will be in the following order:
    1) Starting with frame A, rotate by the first angle about X_A to obtain intermediate frame B'
    2) Rotate by the second angle about Y_B' to obtain intermediate frame B''
    3) Rotate by the third angle about Z_B'' to obtain B

    Args:
        angles (npt.ArrayLike): Three angles (radians) corresponding to the XYZ rotations

    Returns:
        np.ndarray: R_B2A, where frame B is composed by three rotations in XYZ order starting from frame A
    """
    if len(angles) != 3:
        raise ValueError(f"Must provide 3 angles.\nGot:{angles}")
    return rt.matrix_from_euler(angles, _X, _Y, _Z, _EULER)


def fixed_xyz_to_rmat(angles: npt.ArrayLike) -> np.ndarray:
    """Fixed XYZ 3-angle rotation matrix

    Operations will be in the following order:
    1) Rotate by the first angle about X_A
    2) Rotate by the second angle about Y_A
    3) Rotate by the third angle about Z_A

    (This differs from Euler angles, as the rotation is not about the intermediate frames)

    Args:
        angles (npt.ArrayLike): Three angles (radians) corresponding to the XYZ rotations

    Returns:
        np.ndarray: R_B2A, where frame B is composed by three rotations in XYZ order starting from frame A
    """
    return rt.matrix_from_euler(angles, _X, _Y, _Z, _FIXED)


def rmat_to_axis_angle(rmat: np.ndarray) -> tuple[np.ndarray, float]:
    """Converts a rotation matrix into an axis-angle representation

    Args:
        rmat (np.ndarray): (3, 3) rotation matrix

    Returns:
        tuple[np.ndarray, float]:
            np.ndarray: Axis of rotation. Shape (3,)
            float: Rotation angle, in radians
    """
    axis_and_angle = rt.axis_angle_from_matrix(rmat)
    axis = axis_and_angle[:3]
    angle = axis_and_angle[3]
    return axis, angle


def axis_angle_to_rmat(axis: list[float], angle: float) -> np.ndarray:
    """Converts an axis-angle representation to a rotation matrix

    Args:
        axis (list[float]): Axis to rotate around: [x, y, z]
        angle (float): Angle to rotate (radians)

    Returns:
        np.ndarray: Rotation matrix equivalent for the axis-angle representation
    """
    axis_and_angle = np.concatenate([axis, [angle]])
    return rt.matrix_from_axis_angle(axis_and_angle)


def custom_euler_to_rmat(convention: str, angles: npt.ArrayLike) -> np.ndarray:
    """Converts euler angles of a specified convention (like 'zyx') to a rotation matrix

    Use this if a convention other than XYZ is needed

    Args:
        convention (str): Axis order for the three rotations: Must be some length=3
            permutation of 'xyz' - e.g. 'zyx', 'yxy', ...
        angles (npt.ArrayLike): Rotation angles, length = 3

    Raises:
        ValueError: If the convention or the angles are not of length 3, or if the
            convention does not strictly contain just "x", "y", or "z"

    Returns:
        np.ndarray: (3, 3) rotation matrix
    """
    if not len(convention) == 3:
        raise ValueError(f"Convention should be length 3.\nGot: {convention}")
    if not len(angles) == 3:
        raise ValueError(f"Angles should be length 3.\nGot: {angles}")
    convention = convention.lower()
    if not all(c in {"x", "y", "z"} for c in convention):
        raise ValueError(
            f"Convention must only include x, y, and z.\nGot: {convention}"
        )
    ax_to_ind = {"x": 0, "y": 1, "z": 2}
    inds = [ax_to_ind[convention[i]] for i in range(3)]
    return rt.matrix_from_euler(angles, *inds, _EULER)


def custom_fixed_to_rmat(convention: str, angles: npt.ArrayLike) -> np.ndarray:
    """Converts fixed angles of a specified convention (like 'zyx') to a rotation matrix

    Use this if a convention other than XYZ is needed

    Args:
        convention (str): Axis order for the three rotations: Must be some length=3
            permutation of 'xyz' - e.g. 'zyx', 'yxy', ...
        angles (npt.ArrayLike): Rotation angles, length = 3

    Raises:
        ValueError: If the convention or the angles are not of length 3, or if the
            convention does not strictly contain just "x", "y", or "z"

    Returns:
        np.ndarray: (3, 3) rotation matrix
    """
    return custom_euler_to_rmat(convention[::-1], angles[::-1])


def rmat_to_euler_xyz(rmat: np.ndarray) -> tuple[float, float, float]:
    """Converts a rotation matrix to Euler XYZ angles

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        tuple[float, float, float]: XYZ Euler angles
    """
    return rt.euler_from_matrix(rmat, _X, _Y, _Z, _EULER)


def rmat_to_fixed_xyz(rmat: np.ndarray) -> tuple[float, float, float]:
    """Converts a rotation matrix to Fixed XYZ angles

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        tuple[float, float, float]: XYZ Fixed angles
    """
    return rt.euler_from_matrix(rmat, _X, _Y, _Z, _FIXED)


def rmat_to_fixed_zyx(rmat: np.ndarray) -> tuple[float, float, float]:
    """Converts a rotation matrix to Fixed ZYX angles

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        tuple[float, float, float]: ZYX Fixed angles
    """
    return rt.euler_from_matrix(rmat, _Z, _Y, _X, _FIXED)


def check_rotation_mat(R: np.ndarray) -> bool:
    """Determines if a rotation matrix is valid (3x3, orthogonal, and determinant = 1)

    Args:
        R (np.ndarray): A rotation matrix

    Returns:
        bool: Whether or not R is a valid rotation matrix
    """
    try:
        rt.check_matrix(R, strict_check=True)
        return True
    except ValueError:
        return False


def rotate_point(rmat: np.ndarray, point: npt.ArrayLike):
    # Use rotation matrix as operator within a single frame
    # TODO
    raise NotImplementedError


def quat_to_rmat(quat: npt.ArrayLike) -> np.ndarray:
    """Converts XYZW quaternions to a rotation matrix

    Args:
        quat (npt.ArrayLike): XYZW quaternions

    Returns:
        np.ndarray: (3,3) rotation matrix
    """
    q = Quaternion(quat)
    return rt.matrix_from_quaternion(q.wxyz)


def rmat_to_quat(rmat: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix into XYZW quaternions

    NOTE: When computing a quaternion from the rotation matrix there is a sign ambiguity:
    q and -q represent the same rotation. (TODO: add a reference quaternion input to see
    which quaternion is closer, choose the sign accordingly?)

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        np.ndarray: XYZW quaternions
    """
    q = Quaternion()
    q.wxyz = rt.quaternion_from_matrix(rmat)
    return q.xyzw


def euler_xyz_to_quat(angles: npt.ArrayLike) -> np.ndarray:
    """Converts Euler XYZ angles to XYZW quaternions

    Args:
        angles (npt.ArrayLike): Angles (radians), length=3

    Returns:
        np.ndarray: XYZW quaternions
    """
    if len(angles) != 3:
        raise ValueError(f"Must provide 3 angles.\nGot: {angles}")
    q = Quaternion()
    q.wxyz = rt.quaternion_from_euler(angles, _X, _Y, _Z, _EULER)
    return q.xyzw


def fixed_xyz_to_quat(angles: npt.ArrayLike) -> np.ndarray:
    """Converts Fixed XYZ angles to XYZW quaternions

    Args:
        angles (npt.ArrayLike): Angles (radians), length=3

    Returns:
        np.ndarray: XYZW quaternions
    """
    if len(angles) != 3:
        raise ValueError(f"Must provide 3 angles.\nGot: {angles}")
    q = Quaternion()
    q.wxyz = rt.quaternion_from_euler(angles, _X, _Y, _Z, _FIXED)
    return q.xyzw


def quat_to_euler_xyz(quat: Union[Quaternion, npt.ArrayLike]) -> np.ndarray:
    """Converts XYZW quaternions to Euler XYZ angles

    Args:
        quat (Union[Quaternion, npt.ArrayLike]): Either a Quaternion object or
            an array of the XYZW quaternions (length = 4)

    Returns:
        np.ndarray: (3,) Euler XYZ angles
    """
    if not isinstance(quat, Quaternion):
        quat = Quaternion(xyzw=quat)
    return rt.euler_from_quaternion(quat.wxyz, _X, _Y, _Z, _EULER)


def quat_to_fixed_xyz(quat: Union[Quaternion, npt.ArrayLike]) -> np.ndarray:
    """Converts XYZW quaternions to Fixed XYZ angles

    Args:
        quat (Union[Quaternion, npt.ArrayLike]): Either a Quaternion object or
            an array of the XYZW quaternions (length = 4)

    Returns:
        np.ndarray: (3,) Fixed XYZ angles
    """
    if not isinstance(quat, Quaternion):
        quat = Quaternion(xyzw=quat)
    return rt.euler_from_quaternion(quat.wxyz, _X, _Y, _Z, _FIXED)


def axis_angle_between_two_vectors(
    v1: npt.ArrayLike, v2: npt.ArrayLike
) -> tuple[np.ndarray, float]:
    """Gives the axis/angle rotation that would rotate vector v1 to align with v2 (magnitude-independent)

    Args:
        v1 (npt.ArrayLike): (3,) Starting vector/direction
        v2 (npt.ArrayLike): (3,) Ending vector/direction

    Returns:
        tuple[np.ndarray, float]:
            np.ndarray: Axis of rotation. Shape (3,)
            float: Rotation angle, in radians
    """
    axis_and_angle = rt.axis_angle_from_two_directions(v1, v2)
    return axis_and_angle[:3], axis_and_angle[3]


def axis_angle_to_quat(axis: npt.ArrayLike, angle: float) -> np.ndarray:
    """Converts an axis/angle rotation representation to XYZW quaternion

    Args:
        axis (npt.ArrayLike): Axis of rotation. Shape (3,)
        angle (float): Rotation angle, in radians

    Returns:
        np.ndarray: (4,) XYZW quaternion
    """
    q = Quaternion()
    q.wxyz = rt.quaternion_from_axis_angle([*axis, angle])
    return q.xyzw


def quat_to_axis_angle(
    quat: Union[Quaternion, npt.ArrayLike]
) -> tuple[np.ndarray, float]:
    """Converts an XYZW quaternion to an axis/angle representation

    Args:
        quat (Union[Quaternion, npt.ArrayLike]): XYZW quaternion, shape (4,) if passing in an array

    Returns:
        tuple[np.ndarray, float]:
            np.ndarray: Axis of rotation. Shape (3,)
            float: Rotation angle, in radians
    """
    if not isinstance(quat, Quaternion):
        quat = Quaternion(xyzw=quat)
    axis_and_angle = rt.axis_angle_from_quaternion(quat.wxyz)
    axis = axis_and_angle[:3]
    angle = axis_and_angle[3]
    return axis, angle


def compact_axis_angle(axis: npt.ArrayLike, angle: float) -> np.ndarray:
    """Converts an axis/angle rotation into compact form (axis vector with a norm of the angle)

    Args:
        axis (npt.ArrayLike): Axis of rotation. Shape (3,)
        angle (float): Rotation angle, in radians

    Returns:
        np.ndarray: Compact axis/angle representation, shape (3,)
    """
    axis = np.asarray(axis)
    if np.size(axis) != 3 or np.ndim(angle) != 0:
        raise ValueError(f"Invalid axis/angle representation: {axis}, {angle}")
    return (angle / np.linalg.norm(axis)) * axis


def random_rmat() -> np.ndarray:
    """Generate a random rotation matrix

    Returns:
        np.ndarray: Rotation matrix, shape (3, 3)
    """
    q = np.random.rand(4)
    q /= np.linalg.norm(q)
    return quat_to_rmat(q)
