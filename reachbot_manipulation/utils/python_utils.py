"""Utility functions broadly related to python as a whole"""

# TODO determine if this should be separated into more specific files - e.g. debug.py

from typing import Any
from enum import Enum


# TODO add @property wrapping as well
# (possibly buggy, see https://github.com/python/cpython/issues/89519)
class ExtendedEnum(Enum):
    """Add the ability to easily extract values of an Enum"""

    @classmethod
    def get_values(cls):
        """Values of members in the Enum"""
        return list(map(lambda c: c.value, cls))

    @classmethod
    def get_names(cls):
        """Names of members in the Enum"""
        return list(map(lambda c: c.name, cls))


def print_red(message: Any):
    """Helper function for printing in red text

    Args:
        message (Any): The message to print out in red
    """
    print(f"\033[31m{message}\033[0m")


def print_green(message: Any):
    """Helper function for printing in green text

    Args:
        message (Any): The message to print out in green
    """
    print(f"\033[32m{message}\033[0m")


def flatten(l: list[list]) -> list:
    """Flatten a list of lists into a single list

    Args:
        l (list[list]): List of lists to flaten

    Returns:
        list: Flattened list
    """
    return [item for sublist in l for item in sublist]
