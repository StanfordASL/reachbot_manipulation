"""Custom exceptions"""

from typing import Optional


class PybulletError(Exception):
    """Custom exception type for dealing with Pybullet-related issues

    Args:
        message (str): Info to display about the error
        code (int, optional): A pybullet return value indicating that an error occurred. Defaults to None.
    """

    def __init__(self, message: str, code: Optional[int] = None):
        # Store as class attribute to access the code and handle programmatically if needed
        self.code = code
        if code is not None:
            message += f"\nPybullet return value: {code}"
        super().__init__(message)


class OptimizationError(Exception):
    pass


class LinAlgError(Exception):
    pass
