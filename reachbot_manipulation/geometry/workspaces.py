"""CDPR Workspaces

Definitions are based on "Workspace Classification and Quantification Calculations of Cable-Driven Parallel Robots"

This paper defines:
- Static Equilibrium Workspace (SEW)
- Wrench Closure Workspace (WCW)
- Wrench Feasible Workspace (WFW)
- Dynamic Workspace (DC)
- Collision Free Workspace (CFW)

We'll focus on the wrench feasible workspace because this incorporates constraints on the tensile forces from the cables
instead of assuming that infinite force can be applied.

We can also define a "static feasible workspace" as a special case of the wrench feasible workspace where the external
wrench that we're supporting is the gravitational force on the body
"""

# TODO check on the relationship between the position of the COM and the torque calculations... it might be ok, like
# we update the normals assuming the COM is always at the origin? The orientation of start points is confusing
# TODO see if there is a better paper from dexterous grasping that also talks about these workspaces

from typing import Optional

from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from reachbot_manipulation.utils.errors import OptimizationError
from reachbot_manipulation.geometry.plotting import plot_3d_hull, plot_2d_hull
from reachbot_manipulation.geometry.points import cube_points
from reachbot_manipulation.optimization.tension_planner import (
    TensionPlanner,
)
from reachbot_manipulation.optimization.boom_wrench_opt import (
    BoomWrenchOptimizationProblem,
)


def discretized_workspace(limits: npt.ArrayLike, spacing: npt.ArrayLike) -> np.ndarray:
    """Determine the coordinates evenly spanning a space within some bounds

    Example: discretized_workspace([(0, 1), (2, 3)], 0.5)
    >>> [[0.  2. ]
         [0.5 2. ]
         [1.  2. ]
         [0.  2.5]
         [0.5 2.5]
         [1.  2.5]
         [0.  3. ]
         [0.5 3. ]
         [1.  3. ]]

    Args:
        limits (npt.ArrayLike): (lower, upper) limits for each dimension. Shape (n_dimensions, 2)
        spacing (npt.ArrayLike): Approximate spacing between points along their respective axes. Float if using the same
            spacing for all dimensions, array if specifying these individually

    Returns:
        np.ndarray: Coordinates, shape (n_coords, dimension)
    """
    limits = np.atleast_2d(limits)  # Shape (n_dim, 2)
    n_dim, n_lim = limits.shape
    assert n_lim == 2
    if np.isscalar(spacing):
        spacing = spacing * np.ones(n_dim)
    else:
        spacing = np.ravel(spacing)
        assert len(spacing) == n_dim
    ranges = limits[:, 1] - limits[:, 0]
    assert np.all(ranges >= 0)
    ns = np.ceil(ranges / spacing).astype(int) + 1
    axes = [np.linspace(limits[i, 0], limits[i, 1], ns[i]) for i in range(n_dim)]
    grid = np.meshgrid(*axes)
    # Convert the meshgrid to coordinates
    return np.column_stack([np.ravel(g) for g in grid])


def static_equilibrium_workspace(
    workspace: npt.ArrayLike,
    mass: float,
    gravity: npt.ArrayLike,
    start_pts_base_frame: npt.ArrayLike,
    end_pts: npt.ArrayLike,
    orn: npt.ArrayLike,
    max_tension: float,
    max_shoulder_torque: float = 0,
):
    """Determine the coordinates within a workspace where the cable configuration (grasp) can support the robot in
    static equilibrium

    Args:
        workspace (npt.ArrayLike): Discretization of the robot's workspace. Shape (n_eval_coords, 3)
        mass (float): Mass of the robot/body
        gravity (npt.ArrayLike): Gravity vector, shape (3,). i.e. (0, 0, -3.71) for Mars
        start_pts_base_frame (npt.ArrayLike):  Positions of the contact points (cable starting points) on the
            body/robot, in the body's base (COM) frame. Shape (n_points, 3)
        end_pts (npt.ArrayLike): Locations of where the cables are attached to the surrounding environment,
            shape (n_cables, 3)
        orn (npt.ArrayLike): XYZW quaternion orientation of the body/robot, shape (4,)
        max_tension (float): Maximum tension from the cables (i.e. maximum normal force at the contacts), in Newtons
        max_shoulder_torque (float): Maximum torque applied at the shoulders in a boom-driven ReachBot. Defaults to
            0 (cable-driven)

    Returns:
        np.ndarray: All coordinates in the workspace where static equilibrium is possible, shape (n_coords, 3)
    """
    external_wrench = np.concatenate([np.multiply(mass, gravity), (0, 0, 0)])
    return wrench_feasible_workspace(
        workspace,
        external_wrench,
        start_pts_base_frame,
        end_pts,
        orn,
        max_tension,
        max_shoulder_torque,
    )


def wrench_feasible_workspace(
    workspace: npt.ArrayLike,
    external_wrench: npt.ArrayLike,
    start_pts_base_frame: npt.ArrayLike,
    end_pts: npt.ArrayLike,
    orn: npt.ArrayLike,
    max_tension: float,
    max_shoulder_torque: float = 0,
) -> np.ndarray:
    """Determine the coordinates within a workspace where the cable configuration (grasp) can support a certain external
    wrench, for a given orientation of the robot

    Args:
        workspace (npt.ArrayLike): Discretization of the robot's workspace. Shape (n_eval_coords, 3)
        external_wrench (npt.ArrayLike): External wrench to support, shape (6,)
        start_pts_base_frame (npt.ArrayLike):  Positions of the contact points (cable starting points) on the
            body/robot, in the body's base (COM) frame. Shape (n_points, 3)
        end_pts (npt.ArrayLike): Locations of where the cables are attached to the surrounding environment,
            shape (n_cables, 3)
        orn (npt.ArrayLike): XYZW quaternion orientation of the body/robot, shape (4,)
        max_tension (float): Maximum tension from the cables (i.e. maximum normal force at the contacts), in Newtons
        max_shoulder_torque (float): Maximum torque applied at the shoulders in a boom-driven ReachBot. Defaults to
            0 (cable-driven)

    Returns:
        np.ndarray: All coordinates in the workspace where the wrench can be supported, shape (n_coords, 3)
    """
    applied_wrench = -external_wrench
    if max_shoulder_torque == 0:  # Cables
        prob = TensionPlanner(
            workspace[0],
            orn,
            start_pts_base_frame,
            end_pts,
            applied_wrench,
            max_tension,
            verbose=False,
        )
    else:  # Booms
        assert max_shoulder_torque > 0
        prob = BoomWrenchOptimizationProblem(
            workspace[0],
            orn,
            start_pts_base_frame,
            end_pts,
            applied_wrench,
            max_tension,
            max_shoulder_torque,
            verbose=False,
        )

    valid_coords = []
    print("Calculating workspace...")
    for coord in tqdm(workspace):
        # Translate the robot based on the XYZ coordinate in the workspace
        prob.update_robot_pose(coord, orn)
        try:
            prob.solve()
            valid_coords.append(coord)
        except OptimizationError:
            continue
    return np.array(valid_coords)


def plot_workspace(
    coords: npt.ArrayLike,
    use_hull: bool = False,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot a set of points defining a workspace

    Args:
        coords (npt.ArrayLike): Points in the workspace, shape (n_points, dim = 2 or 3)
        use_hull (bool, optional): Whether to plot a convex hull of the points, as opposed to just a scatter plot. The
            convex hull may not be a valid representation of the true workspace. Defaults to False.
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Raises:
        NotImplementedError: If the dimension of the workspace is not 2D or 3D

    Returns:
        plt.Axes: The plot
    """
    coords = np.atleast_2d(coords)
    n_pts, dim = coords.shape

    if ax is None:
        if dim == 2:
            fig, ax = plt.subplots(1, 1)
        elif dim == 3:
            fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        else:
            raise NotImplementedError("Workspace can only be plotted for 2D and 3D")

    if use_hull:
        hull = ConvexHull(coords)
        if dim == 2:
            return plot_2d_hull(hull, ax=ax, show=show)
        else:  # dim == 3:
            return plot_3d_hull(hull, ax=ax, show=show)
    else:
        ax.scatter(*coords.T)
        if show:
            plt.show()
        return ax


def test_static_workspace():
    np.random.seed(0)
    limits = [(-5, 5), (-5, 5), (-5, 5)]
    spacing = 0.5
    workspace = discretized_workspace(limits, spacing)
    local_start_pts = cube_points()
    end_pts = cube_points(sidelength=10) + np.random.randn(8, 3)
    gravity = (0, 0, -3.71)
    max_tension = 30
    max_shoulder_moment = 1
    mass = 10
    orn = (0, 0, 0, 1)
    sew = static_equilibrium_workspace(
        workspace,
        mass,
        gravity,
        local_start_pts,
        end_pts,
        orn,
        max_tension,
        max_shoulder_moment,
    )
    plot_workspace(sew)
    plot_workspace(sew, use_hull=True)


if __name__ == "__main__":
    test_static_workspace()
