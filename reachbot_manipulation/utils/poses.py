"""Class/functions for managing poses and conversions between different representations

Our default representation of a pose will be position (xyz) + quaternion (xyzw)
"""
# TODO
# - Decide if we can eliminate any representations that won't be used
# - Add pos_fixed_xyz??
# - Is the Pose class useful?

from typing import Optional
from enum import Enum

import numpy as np
import numpy.typing as npt
import pytransform3d.trajectories as pt

from reachbot_manipulation.utils import rotations as rts
from reachbot_manipulation.utils import transformations as tfs
from reachbot_manipulation.utils import quaternions as qts


class Pose:
    """Class to handle conversions between different representations of a pose

    Args:
        pos_euler_xyz (npt.ArrayLike, optional): Position and Euler XYZ angles (length = 6).
            Defaults to None, in which case another representation should be provided
        pos_quat (npt.ArrayLike, optional): Position and XYZW quaternions (length = 7).
            Defaults to None, in which case another representation should be provided
        tmat (np.ndarray, optional): (4,4) transformation matrix.
            Defaults to None, in which case another representation should be provided

    Raises:
        ValueError: If either multiple inputs or no inputs are provided
    """

    class Convention(Enum):
        """Helper class to enumerate the different formats of a pose"""

        POS_EULER_XYZ = 1
        POS_QUAT = 2
        TMAT = 3

    def __init__(
        self,
        pos_euler_xyz: Optional[npt.ArrayLike] = None,
        pos_quat: Optional[npt.ArrayLike] = None,
        tmat: Optional[np.ndarray] = None,
    ):
        # Check to make sure that only one format of the pose was provided at init
        input_count = sum(val is not None for val in [pos_euler_xyz, pos_quat, tmat])
        if input_count > 1:
            raise ValueError(
                f"Too many inputs provided ({input_count}). Specify one type of pose"
            )
        if input_count == 0:
            # TODO: decide if it makes sense to allow for an "empty" initialization?
            # (Would there be a case where we'd want to assign a value later on?)
            raise ValueError("Must provide one pose input")

        # Initialize our protected variables
        self._pos_euler_xyz = None
        self._pos_quat = None
        self._tmat = None

        # Set the properties based on the input
        if pos_euler_xyz is not None:
            self._orig_convention = Pose.Convention.POS_EULER_XYZ
            self.pos_euler_xyz = pos_euler_xyz
        elif pos_quat is not None:
            self._orig_convention = Pose.Convention.POS_QUAT
            self.pos_quat = pos_quat
        elif tmat is not None:
            self._orig_convention = Pose.Convention.TMAT
            self.tmat = tmat

    @property
    def pos_euler_xyz(self):
        """Position and XYZ Euler angles, shape=(6,)"""
        if self._pos_euler_xyz is None:
            if self._orig_convention == Pose.Convention.POS_QUAT:
                self._pos_euler_xyz = pos_quat_to_pos_euler_xyz(self.pos_quat)
            elif self._orig_convention == Pose.Convention.TMAT:
                self._pos_euler_xyz = tmat_to_pos_euler_xyz(self.tmat)
        return self._pos_euler_xyz

    @pos_euler_xyz.setter
    def pos_euler_xyz(self, pose: npt.ArrayLike):
        """Updates the position/euler pose, and resets other stored representations"""
        if not check_pos_euler_xyz(pose):
            raise ValueError(
                f"Cannot set the pose, invalid position + euler format.\nGot: {pose}"
            )
        # Update the pose and our knowledge of the convention
        self._pos_euler_xyz = pose
        self._orig_convention = Pose.Convention.POS_EULER_XYZ
        # Reset the stored other conventions since we've updated the value
        self._pos_quat = None
        self._tmat = None

    @property
    def pos_quat(self):
        """Position and XYZW quaternions, shape=(7,)"""
        if self._pos_quat is None:
            if self._orig_convention == Pose.Convention.POS_EULER_XYZ:
                self._pos_quat = pos_euler_xyz_to_pos_quat(self.pos_euler_xyz)
            elif self._orig_convention == Pose.Convention.TMAT:
                self._pos_quat = tmat_to_pos_quat(self.tmat)
        return self._pos_quat

    @pos_quat.setter
    def pos_quat(self, pose: npt.ArrayLike):
        """Updates the position/quaternion pose, and resets other stored representations"""
        if not check_pos_quat(pose):
            raise ValueError(
                f"Cannot set the pose, invalid position + quaternion format.\nGot: {pose}"
            )
        # Update the pose and our knowledge of the convention
        self._pos_quat = pose
        self._orig_convention = Pose.Convention.POS_QUAT
        # Reset the stored other conventions since we've updated the value
        self._pos_euler_xyz = None
        self._tmat = None

    @property
    def tmat(self):
        """Transformation matrix, shape=(4,4)"""
        if self._tmat is None:
            if self._orig_convention == Pose.Convention.POS_EULER_XYZ:
                self._tmat = pos_euler_xyz_to_tmat(self.pos_euler_xyz)
            elif self._orig_convention == Pose.Convention.POS_QUAT:
                self._tmat = pos_quat_to_tmat(self.pos_quat)
        return self._tmat

    @tmat.setter
    def tmat(self, pose: npt.ArrayLike):
        """Updates the transformation matrix, and resets other stored representations"""
        if not tfs.check_transform_mat(pose):
            raise ValueError(
                f"Cannot set the pose, invalid transformation matrix format.\nGot: {pose}"
            )
        # Update the pose and our knowledge of the convention
        self._tmat = pose
        self._orig_convention = Pose.Convention.TMAT
        # Reset the stored other conventions since we've updated the value
        self._pos_euler_xyz = None
        self._pos_quat = None


class ArmPose(Pose):
    pass  # TODO


class RobotPose(Pose):
    pass  # TODO


# TODO still need to decide if these checks should error or just return bool
# Maybe make separate functions like strict_check_pos_euler_xyz?
# I didn't like the "might error, might return the same thing" that pytransform did
# TODO add other checks to these as well
def check_pos_euler_xyz(pose: npt.ArrayLike) -> bool:
    """Checks to see if a position + Euler XYZ pose is valid

    Args:
        pose (npt.ArrayLike): Position + Euler XYZ pose, shape = (6,)

    Returns:
        bool: Whether or not the pose is valid
    """
    return len(pose) == 6


def check_pos_quat(pose: npt.ArrayLike) -> bool:
    """Checks to see if a position + XYZW quaternion pose is valid

    Args:
        pose (npt.ArrayLike): Position + XYZW quaternion pose

    Returns:
        bool: Whether or not the pose is valid
    """
    return len(pose) == 7


def pos_euler_xyz_to_tmat(pose: npt.ArrayLike) -> np.ndarray:
    """Converts a position + Euler pose to a transformation matrix

    Args:
        pose (npt.ArrayLike): Position + Euler XYZ pose, shape = (6,)

    Returns:
        np.ndarray: Transformation matrix, shape (4,4)
    """
    if not check_pos_euler_xyz(pose):
        raise ValueError(f"Invalid position + euler pose.\nGot: {pose}")
    pos = pose[:3]
    orn = pose[3:]
    rmat = rts.euler_xyz_to_rmat(orn)
    return tfs.make_transform_mat(rmat, pos)


def pos_euler_xyz_to_pos_quat(pose: npt.ArrayLike) -> np.ndarray:
    """Converts a position + Euler pose to a position + XYZW quaternion pose

    Args:
        pose (npt.ArrayLike): Position + Euler XYZ pose, shape = (6,)

    Returns:
        np.ndarray: Position + XYZW quaternion pose, shape = (7,)
    """
    if not check_pos_euler_xyz(pose):
        raise ValueError(f"Invalid position + euler pose.\nGot: {pose}")
    pos = pose[:3]
    orn = pose[3:]
    quat = rts.euler_xyz_to_quat(orn)
    return np.array([*pos, *quat])


def tmat_to_pos_euler_xyz(tmat: np.ndarray) -> np.ndarray:
    """Converts a transformation matrix to a position + Euler pose

    Args:
        pose (npt.ArrayLike): Transformation matrix, shape (4,4)

    Returns:
        np.ndarray: Position + Euler XYZ pose, shape = (6,)
    """
    if not tfs.check_transform_mat(tmat):
        raise ValueError(f"Invalid transformation matrix.\nGot: {tmat}")
    rmat = tmat[:3, :3]
    pos = tmat[:3, 3]
    orn = rts.rmat_to_euler_xyz(rmat)
    return np.array([*pos, *orn])


def tmat_to_pos_quat(tmat: np.ndarray) -> np.ndarray:
    """Converts a transformation matrix to a position + XYZW quaternion pose

    Args:
        pose (npt.ArrayLike): Transformation matrix, shape (4,4)

    Returns:
        np.ndarray: Position + XYZW quaternion pose, shape = (7,)
    """
    if not tfs.check_transform_mat(tmat):
        raise ValueError(f"Invalid transformation matrix.\nGot: {tmat}")
    rmat = tmat[:3, :3]
    pos = tmat[:3, 3]
    quat = rts.rmat_to_quat(rmat)
    return np.array([*pos, *quat])


def pos_quat_to_tmat(pose: npt.ArrayLike) -> np.ndarray:
    """Converts a position + XYZW quaternion pose to a transformation matrix

    Args:
        pose (npt.ArrayLike): Position + XYZW quaternion pose, shape = (7,)

    Returns:
        np.ndarray: Transformation matrix, shape (4,4)
    """
    if not check_pos_quat(pose):
        raise ValueError(f"Invalid position + quaternion pose.\nGot: {pose}")
    pos = pose[:3]
    quat = pose[3:]
    rmat = rts.quat_to_rmat(quat)
    return tfs.make_transform_mat(rmat, pos)


def batched_pos_quats_to_tmats(poses: npt.ArrayLike) -> np.ndarray:
    """Converts a array of position + quaternion poses to an array of transformation matrices

    Args:
        poses (npt.ArrayLike): Position + quaternion poses, shape (n, 7)

    Returns:
        np.ndarray: Transformation matrices, shape (n, 4, 4)
    """
    # Assume poses is of shape (n, 7). If not, see if we can fix it, or raise an error
    poses = np.atleast_2d(poses)
    n_rows, n_cols = poses.shape
    if n_cols != 7 and n_rows == 7:
        print("Warning: you might have passed in the transpose of the pose array")
        poses = poses.T
    elif n_cols != 7 and n_rows != 7:
        raise ValueError(
            f"Invalid input shape: {poses.shape} Must be an array of position/quaternion poses"
        )
    # Convert XYZW poses to WXYZ for pytransform3d's quaternion convention
    wxyz_pqs = np.zeros_like(poses)
    wxyz_pqs[:, :3] = poses[:, :3]  # x, y, z
    wxyz_pqs[:, 3] = poses[:, -1]  # qw
    wxyz_pqs[:, 4:] = poses[:, 3:-1]  # qx, qy, qz
    # Use the batched conversion from pytransform3d since this is more efficient than a loop
    return pt.transforms_from_pqs(wxyz_pqs)


def pos_quat_to_pos_euler_xyz(pose: npt.ArrayLike) -> np.ndarray:
    """Converts a position + XYZW quaternion pose to a position + Euler pose

    Args:
        pose (npt.ArrayLike): Position + XYZW quaternion pose, shape = (7,)

    Returns:
        np.ndarray: Position + Euler XYZ pose, shape = (6,)
    """
    if not check_pos_quat(pose):
        raise ValueError(f"Invalid position + quaternion pose.\nGot: {pose}")
    pos = pose[:3]
    quat = pose[3:]
    orn = rts.quat_to_euler_xyz(quat)
    return np.array([*pos, *orn])


def add_global_pose_delta(pose: npt.ArrayLike, pose_delta: npt.ArrayLike) -> np.ndarray:
    """Adds a world-frame "delta" to a pose

    Args:
        pose (npt.ArrayLike): Original reference pose (position + quaternion), shape (7,)
        pose_delta (npt.ArrayLike): Delta to add to the pose (position + quaternion), shape (7,)

    Returns:
        np.ndarray: Position + quaternion pose with the delta applied, shape (7,)
    """
    if not check_pos_quat(pose) or not check_pos_quat(pose_delta):
        raise ValueError(
            f"Invalid inputs: Not position/quaternion form.\nGot: {pose}\nAnd: {pose_delta}"
        )
    new_pos = pose[:3] + pose_delta[:3]
    new_orn = qts.combine_quaternions(pose[3:], pose_delta[3:])
    return np.array([*new_pos, *new_orn])


def add_local_pose_delta(pose: npt.ArrayLike, pose_delta: npt.ArrayLike) -> np.ndarray:
    """Adds a local (robot)-frame "delta" to a pose

    Args:
        pose (npt.ArrayLike): Original reference pose (position + quaternion), shape (7,)
        pose_delta (npt.ArrayLike): Delta to add to the pose (position + quaternion), shape (7,)

    Returns:
        np.ndarray: Position + quaternion pose with the delta applied, shape (7,)
    """
    if not check_pos_quat(pose) or not check_pos_quat(pose_delta):
        raise ValueError(
            f"Invalid inputs: Not position/quaternion form.\nGot: {pose}\nAnd: {pose_delta}"
        )
    T_R2W = pos_quat_to_tmat(pose)  # Robot to world
    T_D2R = pos_quat_to_tmat(pose_delta)  # Delta to robot
    T_D2W = T_R2W @ T_D2R  # Delta to world
    return tmat_to_pos_quat(T_D2W)


# TODO
# Something like a rotational difference and a translational difference??? Idk
def distance_between_poses(pose1: Pose, pose2: Pose) -> np.ndarray:
    raise NotImplementedError


def pose_derivatives(
    poses: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the linear/angular first and second derivatives of a sequence of poses

    Args:
        poses (np.ndarray): Sequence of position + XYZW quaternion poses, shape (n, 7)
        dt (float): Timestep between poses, in seconds

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            np.ndarray: Linear velocities, shape (n, 3)
            np.ndarray: Angular velocities, shape (n, 3)
            np.ndarray: Linear accelerations, shape (n, 3)
            np.ndarray: Angular accelerations, shape (n, 3)
    """
    if poses.shape[-1] != 7:
        raise ValueError(
            f"Invalid pose array: must be shape (n, 7). Got: {poses.shape}"
        )
    positions = poses[:, :3]
    quaternions = poses[:, 3:]
    velocities = np.gradient(positions, dt, axis=0)
    accels = np.gradient(velocities, dt, axis=0)
    omegas = qts.quats_to_angular_velocities(quaternions, dt)
    alphas = np.gradient(omegas, dt, axis=0)
    return velocities, omegas, accels, alphas
