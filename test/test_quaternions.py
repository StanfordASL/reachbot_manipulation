"""Test cases for quaternions"""

# TODO
# - Quaternion distance
# - Quaternion interpolation

import unittest
import numpy as np

from reachbot_manipulation.utils import rotations as rts
from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.quaternion_class import Quaternion
from reachbot_manipulation.utils.quaternions import (
    quaternion_derivative,
    quats_to_angular_velocities,
    combine_quaternions,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
    quaternion_diff,
    quaternion_angular_error,
    random_quaternion,
)


class QuaternionTest(unittest.TestCase):
    """Contains test cases for the Quaternion class and all quaternion-associated functions"""

    def test_quaternion_class(self):
        q0 = normalize([0.1, 0.2, 0.3, 0.4])
        q = Quaternion(wxyz=q0)
        np.testing.assert_array_almost_equal(q0, q.wxyz, decimal=14)
        np.testing.assert_array_almost_equal([*q0[1:], q0[0]], q.xyzw, decimal=14)

    def test_quaternion_conversions(self):
        q = normalize([0.1, 0.2, 0.3, 0.4])
        axis, angle = rts.quat_to_axis_angle(q)
        np.testing.assert_array_almost_equal(rts.axis_angle_to_quat(axis, angle), q)
        rmat = rts.quat_to_rmat(q)
        np.testing.assert_array_almost_equal(rts.rmat_to_quat(rmat), q)
        euler_xyz = rts.quat_to_euler_xyz(q)
        np.testing.assert_array_almost_equal(rts.euler_xyz_to_quat(euler_xyz), q)
        fixed_xyz = rts.quat_to_fixed_xyz(q)
        np.testing.assert_array_almost_equal(rts.fixed_xyz_to_quat(fixed_xyz), q)

    def test_quaternion_combination(self):
        euler_1 = np.array([0, 0, 0])
        euler_2 = np.array([0.1, 0.2, 0.3])
        q1 = rts.euler_xyz_to_quat(euler_1)
        q2 = rts.euler_xyz_to_quat(euler_2)
        q3 = combine_quaternions(q1, q2)
        euler_3 = rts.quat_to_euler_xyz(q3)
        np.testing.assert_array_almost_equal(euler_3, euler_1 + euler_2)

    def test_quaternion_deriv(self):
        # Use a known angular velocity to propagate a quaternion
        # Then, see if we get back roughly the same angular velocity
        w = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        q1 = np.array([0, 0, 0, 1])
        deriv = quaternion_derivative(q1, w)
        q2 = normalize(q1 + deriv * dt)
        arr = np.row_stack([q1, q2])
        ang_vel = quats_to_angular_velocities(arr, dt)
        # The following check is very low-precision but this seems to be a general drawback
        # of this method that we're using
        # Test against the first timestep (2 in total)
        np.testing.assert_almost_equal(ang_vel[0], w, decimal=3)
        # Repeat the test starting from a quaternion other than the identity quat
        # This ensures that we have the correct local vs global sense of angular velocity
        q1 = random_quaternion()
        deriv = quaternion_derivative(q1, w)
        q2 = normalize(q1 + deriv * dt)
        arr = np.row_stack([q1, q2])
        ang_vel = quats_to_angular_velocities(arr, dt)
        np.testing.assert_almost_equal(ang_vel[0], w, decimal=3)

    def test_wxyz_xyzw_conversion(self):
        # Test 1-dimensional input (single quaternion)
        wxyz_1d = np.arange(4)
        xyzw_1d = wxyz_to_xyzw(wxyz_1d)
        np.testing.assert_array_equal(xyzw_to_wxyz(xyzw_1d), wxyz_1d)
        # Test 2-dimensional input (multiple quaternions)
        wxyz_2d = np.row_stack([wxyz_1d, wxyz_1d])
        xyzw_2d = wxyz_to_xyzw(wxyz_2d)
        np.testing.assert_array_equal(xyzw_to_wxyz(xyzw_2d), wxyz_2d)
        # Test invalid input
        invalid_quat = np.arange(5)
        with self.assertRaises(ValueError):
            wxyz_to_xyzw(invalid_quat)
        with self.assertRaises(ValueError):
            xyzw_to_wxyz(invalid_quat)

    def test_quaternion_diff(self):
        fixed_angles = np.array([0.01, 0.02, 0.03])
        q1 = np.array([0, 0, 0, 1])
        q2 = rts.fixed_xyz_to_quat(fixed_angles)
        q3 = quaternion_diff(q1, q2)
        np.testing.assert_array_almost_equal(q3, q2)
        # Test angular difference in compact axis-angle form
        ang_diff = quaternion_angular_error(q2, q1)
        caa = rts.compact_axis_angle(*rts.quat_to_axis_angle(q2))
        np.testing.assert_array_almost_equal(ang_diff, caa, decimal=4)
        # For small fixed angles, the error should be about the same
        np.testing.assert_array_almost_equal(ang_diff, fixed_angles, decimal=2)

    def test_quaternion_interpolation(self):
        # TODO. Need to think of a good way to evaluate this test case
        pass

    def test_quaternion_distance(self):
        # TODO. Need to think of a good way to evaluate this test case
        pass


if __name__ == "__main__":
    unittest.main()
