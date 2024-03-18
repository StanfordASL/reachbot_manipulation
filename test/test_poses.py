"""Test cases for pose representation conversions"""

import unittest

import numpy as np

from reachbot_manipulation.utils import poses
from reachbot_manipulation.utils.poses import Pose
from reachbot_manipulation.utils import rotations as rts
from reachbot_manipulation.utils import transformations as tfs


class PoseTest(unittest.TestCase):
    """Contains test cases to ensure that Pose instances work properly"""

    def test_conversions(self):
        angles = [0.1, 0.2, 0.3]
        R = rts.euler_xyz_to_rmat(angles)
        p = [4, 5, 6]
        q = rts.euler_xyz_to_quat(angles)
        T = tfs.make_transform_mat(R, p)
        pos_euler_xyz = np.array([*p, *angles])
        pos_quat = np.array([*p, *q])
        pose_1 = Pose(pos_euler_xyz=pos_euler_xyz)
        pose_2 = Pose(pos_quat=pos_quat)
        pose_3 = Pose(tmat=T)
        np.testing.assert_array_almost_equal(pose_1.pos_euler_xyz, pose_2.pos_euler_xyz)
        np.testing.assert_array_almost_equal(pose_2.pos_euler_xyz, pose_3.pos_euler_xyz)
        np.testing.assert_array_almost_equal(pose_1.pos_quat, pose_2.pos_quat)
        np.testing.assert_array_almost_equal(pose_2.pos_quat, pose_3.pos_quat)
        np.testing.assert_array_almost_equal(pose_1.tmat, pose_2.tmat)
        np.testing.assert_array_almost_equal(pose_2.tmat, pose_3.tmat)

    def test_reassignment(self):
        # Create a Pose with some initial pose
        angles = [0.1, 0.2, 0.3]
        p = [4, 5, 6]
        pos_euler_xyz = np.array([*p, *angles])
        pose = Pose(pos_euler_xyz=pos_euler_xyz)
        new_angles = [0.3, 0.2, 0.1]
        _ = pose.tmat  # Store a tmat calculation
        # Create a new pose and update the Pose object
        new_pos_euler_xyz = np.array([*p, *new_angles])
        pose.pos_euler_xyz = new_pos_euler_xyz
        new_tmat = pose.tmat
        # The pose.tmat should have been reset since we changed the value via another convention
        comparison_pos_euler_xyz = poses.tmat_to_pos_euler_xyz(new_tmat)
        np.testing.assert_array_almost_equal(
            new_pos_euler_xyz, comparison_pos_euler_xyz
        )

    def test_adding_delta(self):
        # Evaluate the deltas based on position/euler since it is easier to see if
        # the composition worked properly (since we don't need to add quaternions)
        pose_1 = poses.pos_euler_xyz_to_pos_quat([0, 0, 0, 0, 0, 0])
        pose_1_delta = poses.pos_euler_xyz_to_pos_quat([4, 5, 6, 0.1, 0.2, 0.3])
        new_pose_1 = poses.add_global_pose_delta(pose_1, pose_1_delta)
        np.testing.assert_array_almost_equal(
            new_pose_1, poses.pos_euler_xyz_to_pos_quat([4, 5, 6, 0.1, 0.2, 0.3])
        )

        # We should be able to evaluate the local delta in the same way as global
        # *for this example only*, because the initial pose is all 0
        pose_2 = poses.pos_euler_xyz_to_pos_quat([0, 0, 0, 0, 0, 0])
        pose_2_delta = poses.pos_euler_xyz_to_pos_quat([4, 5, 6, 0.1, 0.2, 0.3])
        new_pose_2 = poses.add_local_pose_delta(pose_2, pose_2_delta)
        np.testing.assert_array_almost_equal(
            new_pose_2, poses.pos_euler_xyz_to_pos_quat([4, 5, 6, 0.1, 0.2, 0.3])
        )

        # Now, do a non-trivial addition in the local frame
        pose_3 = poses.pos_euler_xyz_to_pos_quat([0, 0, 0, 0, 0, np.pi / 4])
        pose_3_delta = poses.pos_euler_xyz_to_pos_quat([1, 0, 0, 0, 0, 0])
        new_pose_3 = poses.add_local_pose_delta(pose_3, pose_3_delta)
        np.testing.assert_array_almost_equal(
            new_pose_3,
            poses.pos_euler_xyz_to_pos_quat(
                [np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0, 0, np.pi / 4]
            ),
        )


if __name__ == "__main__":
    unittest.main()
