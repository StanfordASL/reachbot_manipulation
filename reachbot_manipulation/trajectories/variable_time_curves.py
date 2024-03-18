"""Generating trajectories with a free final time

We optimize using the following cost function: (integral of jerk l2 norm) + (time weight parameter) * (total duration)

The weighting on the time component of this cost can be adjusted based on what component matters more. In general, 
a value of 1e-4 appears to put the two components of the cost (jerk and time) on the same order of magnitude

Notes on the search method for the duration:
- We know that the jerk function is a quadratic, and if we add an affine factor based on the total duration
  of the trajectory, it will still be a quadratic. So, a version of quadratic fit search will work well here
- Depending on the weighting of this affine time component, the cost may look very linear. Even if this is the
  case, the search method as implemented will work well, because we will continually approach the boundary
  of feasibility until we stop within some tolerance
- Speaking of this feasibility boundary, this is the main difference between the implemented method and standard
  quadratic fit search. There is some T such that the trajectory is no longer feasible, given the constraints
  on velocity/accel/BCs..., and so in general, we want to solve for a T which is small, yet feasible. So, this
  search method incorporates this knowledge of this infeasible region for small time intervals.
- Ideally, we'd just be able to plug this into CVXPY (since it should just be a quadratic program with some 
  constraints anyways). However, I tried a bunch of formulations of the constraints and it didn't seem to be
  convex or DCP (often leading to either quadratic forms of two variables, or equality constraints on convex 
  functions). Maybe there is a better formulation out there...
"""

from typing import Optional, Callable, Any

import numpy as np
import numpy.typing as npt
import pybullet
import matplotlib.pyplot as plt

from reachbot_manipulation.trajectories.trajectory import plot_traj_constraints
from reachbot_manipulation.trajectories.bezier import BezierCurve, bezier_trajectory
from reachbot_manipulation.trajectories.curve_utils import traj_from_curve
from reachbot_manipulation.utils.boxes import Box
from reachbot_manipulation.utils.debug_visualizer import animate_path
from reachbot_manipulation.utils.errors import OptimizationError
from reachbot_manipulation.config.reachbot_config import SPEED_LIMIT, ACCEL_LIMIT


def left_quadratic_fit_search(
    f: Callable[[float], float | tuple[float, Any]],
    x_init: float,
    dx_tol: float,
    max_iters: int,
) -> tuple[float, float, list[Any]]:
    """A version of quadratic fit search that assumes we have an infeasible region for small x (x >= 0)

    Reference: Algorithms for Optimization (Kochenderfer), Algorithm 3.4

    Args:
        f (Callable[[float], float  |  tuple[float, Any]]): Univariate function to optimize, callable as f(x).
            The return must have the cost of the evaluation as the first output. Any additional outputs will be stored
            and the best will be returned at the end of the search
        x_init (float): Initial location to start the search
        dx_tol (float): Stopping tolerance on evaluation points: Terminate if the percent change between consecutive
            evaluation points is less than this tolerance
        max_iters (int): Maximum iterations of the algorithm (if the stopping tolerance is not achieved)

    Raises:
        OptimizationError: If no feasible solution is found in max_iters iterations

    Returns:
        tuple[float, float, list[Any]]:
            float: Best evaluation point x
            float: Cost of the function evaluation at the best x value
            list[Any]: Additional outputs of the function being optimized at the best x value. Empty list if there
                are no additional outputs
    """
    # Mutable dicts to keep track of the optimization process
    best = {"x": None, "cost": np.inf, "out": []}  # init
    log = {"iters": 0, "feasibility_bound": 0}  # init

    # Create wrapper around the function to handle if it has multiple outputs
    # Return will solely be the cost of the evaluation, but we store the other outputs
    # in the dictionaries as needed
    def _f(x: float) -> float:
        fx = f(x)
        log["iters"] += 1
        if isinstance(fx, tuple):
            cost, *out = fx
            # Out will by default be packed into a list
        else:
            cost = fx
            out = []
        # Check to see if this is the best so far - if so, update
        if cost <= best["cost"] and cost != np.inf:
            best["x"] = x
            best["cost"] = cost
            best["out"] = out
        return cost

    # Find the quadratic fit search interval (a, b, c) given an initial search location
    # This assumes that x is a positive value and that the only infeasible values occurs
    # when x is too small
    def _find_init_interval_from_guess(x: float):
        b = x
        yb = _f(b)
        if yb == np.inf:
            while yb == np.inf and log["iters"] <= max_iters - 1:
                log["feasibility_bound"] = max(b, log["feasibility_bound"])
                b *= 2
                yb = _f(b)
        a = (log["feasibility_bound"] + b) / 2
        ya = _f(a)
        if ya == np.inf:
            while ya == np.inf and log["iters"] <= max_iters - 1:
                log["feasibility_bound"] = max(a, log["feasibility_bound"])
                a = (a + b) / 2
                ya = _f(a)
        # we know c will be valid
        c = b + (b - a)
        yc = _f(c)
        return a, b, c, ya, yb, yc

    a, b, c, ya, yb, yc = _find_init_interval_from_guess(x_init)
    x_prev = None  # init
    while log["iters"] <= max_iters - 1:
        # Quadratic fit for the next search location
        x = (
            0.5
            * (ya * (b**2 - c**2) + yb * (c**2 - a**2) + yc * (a**2 - b**2))
            / (ya * (b - c) + yb * (c - a) + yc * (a - b))
        )
        # Handle if the fit location is known to be infeasible
        if x <= log["feasibility_bound"]:
            x = (log["feasibility_bound"] + a) / 2
        yx = _f(x)
        if yx == np.inf:  # Infeasible
            log["feasibility_bound"] = max(log["feasibility_bound"], x)
        else:
            # Standard quadratic fit update, with extra cases when x is not between a and c
            if x < a:
                if yx < ya:
                    a, ya = x, yx
            elif a <= x <= c:
                if x > b:
                    if yx > yb:
                        c, yc = x, yx
                    else:
                        a, ya, b, yb = b, yb, x, yx
                elif x < b:
                    if yx > yb:
                        a, ya = x, yx
                    else:
                        c, yc, b, yb = b, yb, x, yx
            else:  # x > c
                if yx < yc:
                    c, yc = x, yx
        # Termination criteria: if our evaluation point update has shrunk to within some tolerance
        if x_prev is not None and abs((x - x_prev) / x_prev) < dx_tol:
            break
        x_prev = x

    if best["x"] is None:
        raise OptimizationError("Unable to find a feasible solution")
    return best["x"], best["cost"], best["out"]


def free_final_time_bezier(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf_init: float,
    n_control_pts: int,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    box: Optional[Box] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    time_weight: float = 1e-4,
    timing_rtol: float = 0.01,
    max_iters: int = 15,
    debug: bool = False,
) -> BezierCurve:
    """Optimize a Bezier curve trajectory to balance minimizing jerk with minimizing the total duration

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        t0 (float): Starting time
        tf_init (float): Initial estimate of the final time
        n_control_pts (int): Number of control points for the Bezier curve. Must be >= to the number of constraints,
            and should not be too large (>15ish) as this can reduce optimization performance. 6-10 is usually good
        v0 (Optional[npt.ArrayLike]): Initial velocity, shape (3,). Defaults to None (unconstrained)
        vf (Optional[npt.ArrayLike]): Final velocity, shape (3,). Defaults to None (unconstrained)
        a0 (Optional[npt.ArrayLike]): Initial acceleration, shape (3,). Defaults to None (unconstrained)
        af (Optional[npt.ArrayLike]): Final acceleration, shape (3,). Defaults to None (unconstrained)
        box (Optional[Box]): Box constraint on (lower, upper) position bounds. Defaults to None (unconstrained)
        v_max (Optional[float]): Maximum L2 norm of the velocity. Defaults to None (unconstrained)
        a_max (Optional[float]): Maximum L2 norm of the acceleration. Defaults to None (unconstrained)
        time_weight (float, optional): Objective function weight corresponding to a linear penalty on the duration.
            Defaults to 1e-4. (this was observed to give duration roughly the same weighting as jerk)
        timing_rtol (float, optional): Tolerance on the free-final-time optimization. Defaults to 0.01
            (within 1% of the optimal time)
        max_iters (int, optional): Maximum number of iterations for the free-final-time optimization. Defaults to 15.
        debug (bool, optional): Whether to print/plot details on the free-final-time optimization. Defaults to False.

    Returns:
        BezierCurve: The optimal curve
    """
    curve_kwargs = dict(
        p0=p0,
        pf=pf,
        t0=t0,
        tf=tf_init,
        n_control_pts=n_control_pts,
        v0=v0,
        vf=vf,
        a0=a0,
        af=af,
        box=box,
        v_max=v_max,
        a_max=a_max,
        time_weight=time_weight,
    )

    if debug:
        # Keep track of the costs per time to plot afterwards
        costs_log: dict[float, float] = {}

    # Wrapper around the bezier trajectory function so that we can pop this into our quadratic fit search
    # method with the expected inputs/outputs, and handle when we can't solve for the curve
    # e.g. time as the input, and output the cost and the solved curve
    def _curve_wrapper(t: float) -> tuple[float, BezierCurve]:
        kwargs = curve_kwargs | {"tf": t}
        print("Evaluating time: ", t)
        try:
            curve, cost = bezier_trajectory(**kwargs)
        except OptimizationError:
            curve, cost = None, np.inf
        if debug:
            # Print info on the breakdown of the cost between jerk and time
            print(
                "Cost: ",
                cost,
                " Jerk: ",
                cost - time_weight * t,
                " Time: ",
                time_weight * t,
            )
            costs_log[t] = cost
        # The quadratic search assumes that cost is the first output
        return cost, curve

    t, cost, output = left_quadratic_fit_search(
        _curve_wrapper, tf_init, timing_rtol, max_iters
    )
    best_curve = output[0]
    if debug:
        print("Optimal time: ", t, " yields cost: ", cost)
        _plot_optimization_data(costs_log)

    return best_curve


def _plot_optimization_data(cost_log: dict[float, float], show: bool = True):
    """Helper function to plot the time optimization results when debugging

    Args:
        cost_log (dict[float, float]): Costs for each duration. Keys: times, Values: costs
        show (bool, optional): Whether or not to show the plot. Defaults to True.
    """
    fig = plt.figure()
    times, costs = zip(*cost_log.items())
    plt.subplot(1, 2, 1)
    plt.scatter(times, costs)
    sort_idxs = np.argsort(times)
    times_sorted = np.array(times)[sort_idxs]
    costs_sorted = np.array(costs)[sort_idxs]
    plt.plot(times_sorted, costs_sorted, "--")
    plt.xlabel("Time")
    plt.ylabel("Cost")
    plt.title("Cost vs duration")
    plt.subplot(1, 2, 2)
    plt.plot(range(len(costs)), costs)
    plt.scatter(range(len(costs)), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Convergence")
    if show:
        plt.show()


def _bezier_main():
    p0 = (0, 0, 0)
    pf = (1, 2, 3)
    t0 = 0
    tf_init = 30
    n_control_pts = 30
    dt = 0.1
    v0 = (0.3, 0.2, 0.1)
    vf = (0, 0, 0)
    a0 = (0, 0, 0)
    af = (0, 0, 0)
    print("Speed limit: ", SPEED_LIMIT)
    print("Accel limit: ", ACCEL_LIMIT)
    curve = free_final_time_bezier(
        p0,
        pf,
        t0,
        tf_init,
        n_control_pts,
        v0,
        vf,
        a0,
        af,
        None,
        SPEED_LIMIT,
        ACCEL_LIMIT,
        debug=True,
    )
    traj = traj_from_curve(curve, dt)
    traj.plot()
    plot_traj_constraints(traj, None, SPEED_LIMIT, ACCEL_LIMIT, None, None)
    pybullet.connect(pybullet.GUI)
    traj.visualize(30)
    animate_path(traj.positions, 5)
    input("Animation complete, press Enter to finish")


if __name__ == "__main__":
    _bezier_main()
