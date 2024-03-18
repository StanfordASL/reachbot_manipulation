"""Utilities for working with CVXPY problems"""

import cvxpy as cp


def print_problem_info(prob: cp.Problem) -> None:
    """Prints the stats/metrics from a CVXPY Problem

    Args:
        prob (cp.Problem): CVXPY Problem
    """
    # There are more attributes in the Problem but these are the most important
    print("status: ", prob.status)
    print("is_dcp: ", prob.is_dcp())
    print("is_dpp: ", prob.is_dpp())
    for key, value in prob.solver_stats.__dict__.items():
        print(key, ": ", value)
    for key, value in prob.size_metrics.__dict__.items():
        print(key, ": ", value)
