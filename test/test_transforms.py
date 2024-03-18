"""Test cases for transformations"""

import unittest
import numpy as np

from reachbot_manipulation.utils import rotations as rts
from reachbot_manipulation.utils import transformations as tfs


class TransformsTest(unittest.TestCase):
    """Contains test cases for the transformations utility functions"""

    def test_simple_transforms(self):
        np.testing.assert_array_equal(
            tfs.make_transform_mat(np.eye(3), np.zeros(3)), np.eye(4)
        )
        np.testing.assert_array_equal(
            tfs.make_transform_mat(
                np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), [1, 2, 3]
            ),
            np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]]),
        )

    def test_transform_inversion(self):
        rmat = rts.euler_xyz_to_rmat([1, 2, 3])
        trans = [1, 2, 3]
        T = tfs.make_transform_mat(rmat, trans)
        T_inv = tfs.invert_transform_mat(T)
        np.testing.assert_array_almost_equal(T @ T_inv, np.eye(4))
        np.testing.assert_array_almost_equal(T_inv @ T, np.eye(4))

    def test_transform_check(self):
        self.assertTrue(tfs.check_transform_mat(np.eye(4)))
        self.assertFalse(tfs.check_transform_mat(np.random.rand(4, 4)))
        self.assertFalse(tfs.check_transform_mat(np.random.rand(1, 4)))
        self.assertFalse(
            tfs.check_transform_mat(
                tfs.make_transform_mat(np.random.rand(3, 3), np.random.rand(3))
            )
        )
        self.assertTrue(
            tfs.check_transform_mat(
                tfs.make_transform_mat(rts.Rx(1), np.random.rand(3))
            )
        )

    def test_transform_points(self):
        np.random.seed(0)
        pts = np.random.rand(10, 3)
        T = tfs.make_transform_mat(rts.random_rmat(), np.random.rand(3))
        np.testing.assert_array_almost_equal(
            np.array([tfs.transform_point(T, p) for p in pts]),
            tfs.transform_points(T, pts),
        )

    def test_transform_point(self):
        R = rts.Rx(np.pi)
        t = np.array([1, 2, 3])
        T = tfs.make_transform_mat(R, t)
        # Rotation has no effect on the zero vector
        np.testing.assert_array_almost_equal(tfs.transform_point(T, [0, 0, 0]), t)
        # Translation only
        t = np.zeros(3)
        T = tfs.make_transform_mat(np.eye(3), t)
        np.testing.assert_array_almost_equal(
            tfs.transform_point(T, [1, 2, 3]), np.add([1, 2, 3], t)
        )
        # Rotation only
        R = rts.Rz(np.pi)
        p = np.array([1, 0, 0])
        t = np.zeros(3)
        T = tfs.make_transform_mat(R, t)
        np.testing.assert_array_almost_equal(tfs.transform_point(T, p), [-1, 0, 0])


if __name__ == "__main__":
    unittest.main()
