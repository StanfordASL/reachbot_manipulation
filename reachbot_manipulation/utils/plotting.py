"""Functions for plotting various data or debugging info"""

# TODO look into pytransform3d.plot_utils

from typing import Union, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def num_subplots_to_shape(n: int) -> tuple[int, int]:
    """Determines the best layout of a number of subplots within a larger figure

    Args:
        n (int): Number of subplots

    Returns:
        tuple[int, int]: Number of rows and columns for the subplot divisions
    """
    n_rows = int(np.sqrt(n))
    n_cols = n // n_rows + (n % n_rows > 0)
    assert n_rows * n_cols >= n
    return (n_rows, n_cols)


# Refer to:
# https://stackoverflow.com/questions/18344934/animate-a-rotating-3d-graph-in-matplotlibs
# https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
def animate_3d_plot(
    fig: plt.Figure,
    ax: Union[plt.Axes, list[plt.Axes]],
    filename: str,
    elevation_angle: float = 10,
    duration: float = 10,
    fps: int = 30,
    dpi: int = 200,
):
    """Animate a 3D plot (or plots) by rotating about the Z axis, and save it to a file

    Args:
        fig (plt.Figure): Figure containing the 3D plot
        ax (Union[plt.Axes, list[plt.Axes]]): 3D axes to rotate and animate. If a list, this will animate all of the
            provided axes together
        filename (str): Filename to save the animation to
        elevation_angle (float, optional): Elevation angle for viewing the plot as it rotates.
            Defaults to 10 (degrees)
        duration (float, optional): Animation duration. Defaults to 10 (seconds)
        fps (int, optional): Frames per second. Defaults to 30.
        dpi (int, optional): Dots per inch defining the resolution of the video. Defaults to 200
    """

    def _animate(i):
        if isinstance(ax, list):  # Animate multiple plots together
            for ax_i in ax:
                ax_i.view_init(elev=elevation_angle, azim=i)
        else:  # Just one plot to animate
            ax.view_init(elev=elevation_angle, azim=i)

    n_frames = round(duration * fps)
    interval = 1000 / fps
    anim = animation.FuncAnimation(fig, _animate, frames=n_frames, interval=interval)
    anim.save(filename, dpi=dpi, extra_args=["-vcodec", "libx264"])


def gca_3d() -> plt.Axes:
    """Gets the current matplotlib 3D axes, if one exists. If not, create a new figure/axis

    Returns:
        plt.Axes: 3D axes
    """
    if len(plt.get_fignums()) == 0:  # No existing figures
        _, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    else:
        ax = plt.gca()
        if ax.name != "3d":  # An axis exists, but it is not 3D
            _, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    return ax


def gca_2d() -> plt.Axes:
    """Gets the current matplotlib 2D axes, if one exists. If not, create a new figure/axis

    Returns:
        plt.Axes: 2D axes
    """
    if len(plt.get_fignums()) == 0:  # No existing figures
        _, ax = plt.subplots(1, 1)
    else:
        ax = plt.gca()
        if ax.name != "rectilinear":  # An axis exists, but it is not 2D
            _, ax = plt.subplots(1, 1)
    return ax


# TODO docstrings and typing
# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-a-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def plot_3d_arrows(
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Adds an arrow (or set of arrows) to a 3D plot

    Args:
        start (npt.ArrayLike): Arrow start point(s), shape (3,) or (n_arrows, 3)
        end (npt.ArrayLike): Arrow end point(s), shape (3,) or (n_arrows, 3)
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_3d()
    ax.set_aspect("equal")

    start = np.atleast_2d(start)
    end = np.atleast_2d(end)
    n_arrows = start.shape[0]
    assert start.shape[-1] == 3
    assert end.shape[0] == n_arrows

    # TODO make these inputs to the function
    arrow_props = dict(
        mutation_scale=20, arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0
    )
    for i in range(n_arrows):
        ax.add_artist(
            Arrow3D(
                [start[i, 0], end[i, 0]],
                [start[i, 1], end[i, 1]],
                [start[i, 2], end[i, 2]],
                **arrow_props
            )
        )
    if show:
        plt.show()
    return ax


def _test_plot_animation():
    # Generate a random plot to use as an example
    def _randrange(n, vmin, vmax):
        return (vmax - vmin) * np.random.rand(n) + vmin

    n = 100
    xx = _randrange(n, 23, 32)
    yy = _randrange(n, 0, 100)
    zz = _randrange(n, -50, -25)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xx, yy, zz, marker="o", s=20, c="goldenrod", alpha=0.6)

    # Animate and save the plot
    filename = "artifacts/test_animation.mp4"
    print("Animating the plot...")
    animate_3d_plot(fig, ax, filename)
    print("Done. File saved to ", filename)


if __name__ == "__main__":
    # _test_plot_animation()
    pass
