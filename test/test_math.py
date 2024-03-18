"""Test cases for math utilities"""

import unittest

import numpy as np

from reachbot_manipulation.utils.math_utils import normalize, is_diagonal, safe_divide


class MathTest(unittest.TestCase):
    """Contains test cases to validate math utility functions"""

    def test_normalize(self):
        a = np.random.rand(3)
        a = normalize(a)
        np.testing.assert_almost_equal(1.0, np.linalg.norm(a))
        a = np.random.rand(1)
        a = normalize(a)
        np.testing.assert_almost_equal(1.0, np.linalg.norm(a))
        with self.assertRaises(ZeroDivisionError):
            a = [0, 0, 0]
            normalize(a)

    def test_is_diagonal(self):
        a = np.diag([1, 2, 3])
        self.assertTrue(is_diagonal(a))
        a = np.random.rand(2, 2)
        self.assertFalse(is_diagonal(a))
        a = np.zeros((3, 3))
        self.assertTrue(is_diagonal(a))
        # Raises an error if it is a scalar
        with self.assertRaises(ValueError):
            self.assertTrue(is_diagonal(1))
        # But, if it is a 1x1 matrix, it should work
        a = np.array([1])
        self.assertTrue(is_diagonal(a))
        a = np.array([[1]])
        self.assertTrue(is_diagonal(a))

    def test_safe_divide(self):
        # Should work normally if there are no zeros
        a = [1, 2, 3]
        b = [1, 2, 3]
        c = safe_divide(a, b)
        # Test different fill methods
        np.testing.assert_array_equal(c, [1, 1, 1])
        b = [1, 2, 0]
        c = safe_divide(a, b, fill="original")
        np.testing.assert_array_equal(c, [1, 1, 3])
        c = safe_divide(a, b, fill="nan")
        np.testing.assert_array_equal(c, [1, 1, np.nan])
        c = safe_divide(a, b, fill="zero")
        np.testing.assert_array_equal(c, [1, 1, 0])
        c = safe_divide(a, b, fill="inf")
        np.testing.assert_array_equal(c, [1, 1, float("inf")])
        # Check that the "out" parameter did not actually modify a
        self.assertTrue(a == [1, 2, 3])
        # Test when the entire array is 0
        b = [0, 0, 0]
        c = safe_divide(a, b, fill="original")
        np.testing.assert_array_equal(c, a)
        # Test scalar case
        a = 1
        b = 2
        c = safe_divide(a, b)
        self.assertTrue(c == 0.5)
        b = 0
        c = safe_divide(a, b, fill="original")
        self.assertTrue(c == a)
        # Check negative inf
        a = -1
        b = 0
        c = safe_divide(a, b, fill="inf")
        self.assertTrue(c == -np.inf)


if __name__ == "__main__":
    unittest.main()
