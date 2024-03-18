"""Optimization metrics for Reachbot"""

from typing import Union

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def log_prob_of_success(
    force: Union[float, np.ndarray, cp.Variable, cp.Expression],
    mean: Union[float, np.ndarray],
    stdev: Union[float, np.ndarray],
) -> cp.Expression:
    """Robustness metric for a given stance, given the distribution of the pulling-force limit surface at each grasp

    This is the summed logarithm of the probability of success for each grasp. (A concave function)

    Based on "Motion Planning for a Climbing Robot with Stochastic Grasps"

    Args:
        force (Union[float, np.ndarray, cp.Variable, cp.Expression]): Pull force on each site
        mean (Union[float, np.ndarray]): Mean pull force for each site. Shape == shape(force)
        stdev (Union[float, np.ndarray]): Standard deviation of the pull force for each site. Shape == shape(force)

    Returns:
        cp.Expression: Scalar value representing the success metric to maximize
    """
    # Ensure uniformity in the inputs: Number of sites must match the number of distribution parameters
    if isinstance(force, float):
        n = 1
    elif isinstance(force, np.ndarray):
        n = len(force)
    elif isinstance(force, (cp.Variable, cp.Expression)):
        n = 1 if force.shape == () else force.shape[0]
    if n == 1:
        assert isinstance(mean, (float, int)) or len(mean) == 1
        assert isinstance(stdev, (float, int)) or len(stdev) == 1
    else:
        assert len(mean) == n
        assert len(stdev) == n
    # Evaluate the metric. Note that the cvxpy log_normcdf uses a close approximation to the true function
    return cp.sum(cp.log_normcdf((mean - force) / stdev))


def _test_log_prob_of_success():
    """Plot the metric and the associated probabilities, given a single site's pull force distribution"""
    # Create an arbitrary distribution and set of pull forces to test
    forces = np.linspace(5, 30, 20)
    mean = 20
    stdev = 5
    # Evaluate our success metric for each candidate pull force
    metrics = np.array([log_prob_of_success(f, mean, stdev).value for f in forces])
    # Exponentiate the metrics to back out the probabilities
    probs = np.exp(metrics)
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(forces, metrics)
    ax1.set_title(f"Success metric for a grasp site ~ N({mean}, {stdev})")
    ax1.set_xlabel("Pull force")
    ax1.set_ylabel("log(p(success))")
    ax2.plot(forces, probs)
    ax2.set_title(f"Success probabilities for a grasp site ~ N({mean}, {stdev})")
    ax2.set_xlabel("Pull force")
    ax2.set_ylabel("p(success)")
    plt.show()


def main():
    _test_log_prob_of_success()


if __name__ == "__main__":
    main()
