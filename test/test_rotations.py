"""Test cases for rotations

All quaternion-associated tests will be in a separate test script because there's a lot to test there
"""
# TODO
# Add test cases for:
# - Quaternion distance
# - Quaternion interpolation
# - Pretty much a lot more quaternion stuff, there aren't many cases here

import unittest
import numpy as np

from reachbot_manipulation.utils import rotations as rts


class RotationsTest(unittest.TestCase):
    """Contains test cases for the rotations utility functions"""

    def test_simple_rmats(self):
        self.assertTrue(np.array_equal(rts.Rx(0), np.eye(3)))
        self.assertTrue(np.array_equal(rts.Ry(0), np.eye(3)))
        self.assertTrue(np.array_equal(rts.Rz(0), np.eye(3)))
        np.testing.assert_array_almost_equal(
            rts.Rx(np.deg2rad(90)), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        )
        np.testing.assert_array_almost_equal(
            rts.Ry(np.deg2rad(90)), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        )
        np.testing.assert_array_almost_equal(
            rts.Rz(np.deg2rad(90)), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        )

    def test_euler_angles(self):
        th_30 = np.deg2rad(30)
        # th_45 = np.deg2rad(45)
        # th_60 = np.deg2rad(60)
        # th_90 = np.deg2rad(90)
        th_180 = np.deg2rad(180)
        np.testing.assert_array_almost_equal(
            rts.euler_xyz_to_rmat([np.deg2rad(90), 0, 0]), rts.Rx(np.deg2rad(90))
        )
        np.testing.assert_array_almost_equal(
            rts.euler_xyz_to_rmat([0, np.deg2rad(90), 0]), rts.Ry(np.deg2rad(90))
        )
        np.testing.assert_array_almost_equal(
            rts.euler_xyz_to_rmat([0, 0, np.deg2rad(90)]), rts.Rz(np.deg2rad(90))
        )
        np.testing.assert_array_almost_equal(
            rts.euler_xyz_to_rmat([np.deg2rad(45), np.deg2rad(45), 0]),
            np.array(
                [
                    [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
                    [0.5, np.sqrt(2) / 2, -0.5],
                    [-0.5, np.sqrt(2) / 2, 0.5],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            rts.euler_xyz_to_rmat([th_30, th_30, th_30]),
            np.array(
                [
                    [0.7500000, -0.4330127, 0.5000000],
                    [0.6495190, 0.6250000, -0.4330127],
                    [-0.1250000, 0.6495190, 0.7500000],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            rts.euler_xyz_to_rmat([th_180, th_180, th_180]), np.eye(3)
        )

    def test_rmat_check(self):
        self.assertTrue(rts.check_rotation_mat(rts.Rx(1)))
        self.assertTrue(rts.check_rotation_mat(rts.euler_xyz_to_rmat([1, 2, 3])))
        self.assertFalse(rts.check_rotation_mat(np.random.rand(3, 3)))
        self.assertFalse(rts.check_rotation_mat(np.random.rand(4, 4)))
        self.assertFalse(rts.check_rotation_mat(np.random.rand(1, 3)))

    def test_rmat_angle_conversion(self):
        input_angles = [0.1, 0.2, 0.3]
        # Fixed XYZ
        R1 = rts.fixed_xyz_to_rmat(input_angles)
        output_angles_2 = rts.rmat_to_fixed_xyz(R1)
        np.testing.assert_array_almost_equal(input_angles, output_angles_2)
        # Euler XYZ
        R2 = rts.euler_xyz_to_rmat(input_angles)
        output_angles_3 = rts.rmat_to_euler_xyz(R2)
        np.testing.assert_array_almost_equal(input_angles, output_angles_3)

    def test_rmat_axis_angle_conversion(self):
        R_in = rts.euler_xyz_to_rmat([0.1, 0.2, 0.3])
        axis, angle = rts.rmat_to_axis_angle(R_in)
        R_out = rts.axis_angle_to_rmat(axis, angle)
        np.testing.assert_array_almost_equal(R_in, R_out)

    def test_other_conventions(self):
        angles = [0.1, 0.2, 0.3]
        np.testing.assert_array_almost_equal(
            rts.custom_euler_to_rmat("xyz", angles), rts.euler_xyz_to_rmat(angles)
        )
        np.testing.assert_array_almost_equal(
            rts.custom_fixed_to_rmat("xyz", angles), rts.fixed_xyz_to_rmat(angles)
        )
        np.testing.assert_array_almost_equal(
            rts.custom_euler_to_rmat("zyx", angles),
            rts.Rz(angles[0]) @ rts.Ry(angles[1]) @ rts.Rx(angles[2]),
        )
        np.testing.assert_array_almost_equal(
            rts.custom_fixed_to_rmat("zyx", angles),
            rts.Rx(angles[2]) @ rts.Ry(angles[1]) @ rts.Rz(angles[0]),
        )


if __name__ == "__main__":
    unittest.main()
