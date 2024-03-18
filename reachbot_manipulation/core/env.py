"""Environments for testing ReachBot, and models of grasp sites / limit surfaces"""

# TODO consider using scipy truncnorm instead of rejection sampling


from typing import Optional, Union
from dataclasses import dataclass

import pybullet
import numpy as np
import numpy.typing as npt
from scipy.stats import qmc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pybullet_utils.bullet_client import BulletClient

from reachbot_manipulation.utils.debug_visualizer import visualize_points
from reachbot_manipulation.utils.bullet_utils import load_visual_object
from reachbot_manipulation.utils.math_utils import (
    normalize,
    spherical_vonmises_sampling,
    one_sided_confidence_bound,
)
from reachbot_manipulation.utils.rotations import (
    axis_angle_between_two_vectors,
    axis_angle_to_rmat,
    quat_to_rmat,
    axis_angle_to_quat,
)
from reachbot_manipulation.geometry.plotting import plot_cylinder, plot_3d_box
from reachbot_manipulation.utils.plotting import gca_3d
from reachbot_manipulation.config.reachbot_config import (
    LOCAL_CABLE_POSITIONS,
    LENGTH,
    WIDTH,
    HEIGHT,
)


class EllipsoidalLimitSurface:
    """Modeling the limit surface pull-force distribution of a grasp site as an ellipsoid

    Based on the original model from "Motion Planning for a Climbing Robot with Stochastic Grasps"

    Calling the limit surface model will return the pull-force distribution along that direction,
    >>> lim_surf = EllipsoidalLimitSurface(...)
    >>> mean, stdev = lim_surf(direction)

    Args:
        normal_dir (npt.ArrayLike): Normal pulling direction of the grasp site, shape (3,)
        normal_force (float): Mean pulling force when pulling along the normal direction
        transverse_force (float): Mean pulling force when pulling along the transverse direction
        normal_stdev (float): Standard deviation of the pull-force distribution when along the normal direction
        transverse_stdev (float): Standard deviation of the pull-force distribution when along the transverse direction
    """

    def __init__(
        self,
        normal_dir: npt.ArrayLike,
        normal_force: float,
        transverse_force: float,
        normal_stdev: float,
        transverse_stdev: float,
    ):
        if normal_force < transverse_force:
            print(
                "Warning: Normal force should generally always be higher than transverse"
            )
        if normal_stdev > transverse_stdev:
            print(
                "Warning: Normal standard deviation should generally always be lower than transverse"
            )
        self.normal_dir = normalize(np.ravel(normal_dir))
        self.normal_force = normal_force
        self.transverse_force = transverse_force
        self.normal_stdev = normal_stdev
        self.transverse_stdev = transverse_stdev

    def __call__(self, pull_dir: npt.ArrayLike) -> tuple[float, float]:
        """Evaluates the limit surface force distribution for a given pull direction

        Args:
            pull_dir (npt.ArrayLike): Pulling direction, shape (3,). Note that this is a world-frame vector which points
                AWAY from the cave surface, not towards.

        Returns:
            tuple[float, float]: Tuple of:
                float: Mean pulling force
                float: Standard deviation in the pulling force
        """
        pull_dir = normalize(pull_dir)
        theta = np.arccos(np.clip(np.dot(pull_dir, self.normal_dir), -1, 1))
        # Ensure that we're pulling along the nominal direction of the grasp site
        if theta > np.pi / 2:
            # If not, assume that we get zero force from this site
            return 0, 1
        # Ellipsoidal model for the mean pull force
        mean = (
            self.normal_force
            * self.transverse_force
            / np.sqrt(
                self.transverse_force**2 * np.cos(theta) ** 2
                + self.normal_force**2 * np.sin(theta) ** 2
            )
        )
        # Linear model for the standard deviation
        stdev = self.normal_stdev + theta / (np.pi / 2) * (
            self.transverse_stdev - self.normal_stdev
        )
        return mean, stdev

    def pct_confidence_lower_bound_force(
        self, pull_dir: npt.ArrayLike, pct: float
    ) -> float:
        """Return the scalar force we can achieve from the grasp site with X% confidence

        Args:
            pull_dir (npt.ArrayLike): Pulling direction, shape (3,)
            pct (float): Percent confidence (for instance, 0.99 for 99% confidence bound)

        Returns:
            float: Pull force along the given direction
        """
        mean, stdev = self(pull_dir)
        return one_sided_confidence_bound(pct, mean, stdev, "lower")


class GraspSite:
    """A grasp site where ReachBot can place one of its booms

    Args:
        limit_surface (EllipsoidalLimitSurface): Pull-force distribution of the site
        position (npt.ArrayLike): Position of the grasp site in the environment, shape (3,)
    """

    def __init__(self, limit_surface: EllipsoidalLimitSurface, position: npt.ArrayLike):
        self.limit_surface = limit_surface
        self.position = np.asarray(position)
        if len(self.position) != 3:
            raise ValueError("Position must be a 3D location")


# TODO decide if we should use a site density rather than a number of sites
@dataclass
class EnvConfig:
    """Configuration of the environment for sampling a set of grasp sites

    This assumes a cylindrical approximation of the environment (such as a cave or lava tube),
    with ellipsoidal gaussian distributions for the limit surface model

    Args:
        center (npt.ArrayLike): Center of the cylindrical environment, shape (3,)
        radius (float): Radius of the cylindrical environment
        length (float): Length of the cylindrical environment
        direction (npt.ArrayLike): Parallel direction of the cylindrical environment, shape (3,)
        n_sites (int): Number of grasp sites in the environment
        mean_of_parallel_force_mean (float): Typical parallel pull force we expect from a grasp site
        mean_of_parallel_force_stdev (float): Typical spread in the parallel pull force for a grasp site
        stdev_of_parallel_force_mean (float): Spread in the parallel pull force across grasp sites
        stdev_of_parallel_force_stdev (float): Spread in the parallel pull force spread across grasp sites
        mean_of_transverse_force_mean (float): Typical transverse pull force we expect from a grasp site
        mean_of_transverse_force_stdev (float): Typical spread in the transverse pull force for a grasp site
        stdev_of_transverse_force_mean (float): Spread in the transverse pull force across grasp sites
        stdev_of_transverse_force_stdev (float): Spread in the transverse pull force spread across grasp sites
        stdev_orn_noise (float): Spread in the pointing direction for the grasp site direction, as compared with
            a nominal "inwards" direction
        rng (Optional[Union[int, np.random.Generator]]): Random number generator. Defaults to None (unseeded)
    """

    # Defining a cave / lava tube environment as a cylinder with some number of sampled grasp points
    center: npt.ArrayLike = (0, 0, 0)
    radius: float = 10  # meters
    length: float = 40  # meters
    direction: npt.ArrayLike = (1, 0, 0)  # Direction of the cylinder
    n_sites: int = 20
    # Defining distribution over parallel pull force distributions
    mean_of_parallel_force_mean: float = 40  # Newtons
    mean_of_parallel_force_stdev: float = 3
    stdev_of_parallel_force_mean: float = 3
    stdev_of_parallel_force_stdev: float = 1
    # Defining distribution over transverse pull force distributions
    mean_of_transverse_force_mean: float = 20
    mean_of_transverse_force_stdev: float = 5
    stdev_of_transverse_force_mean: float = 5
    stdev_of_transverse_force_stdev: float = 1
    # Defining noise distribution on nominally inward-pointing site orientation
    # This will be based on a spherical von mises distribution where kappa = 1/sigma**2
    stdev_orn_noise: float = 0.15
    # Seed the RNG if we want to generate the same environment every time
    rng: Optional[Union[int, np.random.Generator]] = None

    def __post_init__(self):
        assert len(self.center) == 3
        assert self.radius > 0
        assert self.length > 0
        assert len(self.direction) == 3
        assert self.n_sites > 0
        assert self.mean_of_parallel_force_mean >= 0
        assert self.mean_of_parallel_force_stdev >= 0
        assert self.stdev_of_parallel_force_mean >= 0
        assert self.stdev_of_parallel_force_stdev >= 0
        assert self.mean_of_transverse_force_mean >= 0
        assert self.mean_of_transverse_force_stdev >= 0
        assert self.stdev_of_transverse_force_mean >= 0
        assert self.stdev_of_transverse_force_stdev >= 0
        assert self.stdev_orn_noise >= 0


class Environment:
    """ReachBot environment, simulating a lava tube or Martian cave

    In general, the best way to automatically generate a new environment is via...

    >>> env = Environment.from_config(config)

    ... where the config specifies the sampling distributions for the environment parameters

    Args:
        grasp_sites (list[GraspSite]): Grasp sites, where ReachBot can place its booms. Contains info on its force
            distribution ans well as its position
        config (Optional[EnvConfig]): An input to store the config that generated the environment, via the from_config
            method. Defaults to None.
    """

    def __init__(
        self, grasp_sites: list[GraspSite], config: Optional[EnvConfig] = None
    ):
        self.sites = grasp_sites
        self.config = config
        self.num_sites = len(grasp_sites)
        # TODO decide if we should store arrays of the other data on the pull force distributions...
        self.positions = np.zeros((self.num_sites, 3))
        self.directions = np.zeros((self.num_sites, 3))
        self.forces_95_pct_confidence = np.zeros(self.num_sites)
        for i in range(self.num_sites):
            self.positions[i] = self.sites[i].position
            self.directions[i] = self.sites[i].limit_surface.normal_dir
            self.forces_95_pct_confidence[i] = self.sites[
                i
            ].limit_surface.pct_confidence_lower_bound_force(
                self.sites[i].limit_surface.normal_dir, 0.95
            )

    @classmethod
    def from_config(cls, config: EnvConfig) -> "Environment":
        """Constructs a new environment instance by randomly sampling sites based on the config parameters

        Args:
            config (EnvConfig): Config specifying information about the geometry of the environment as well as the
                distributions over the grasp sites for sampling.

        Returns:
            Environment: A new randomly-generated environment
        """

        rng = np.random.default_rng(config.rng)

        # Define a helper function to ensure that all sampled distribution values are positive w/ rejection sampling
        def _sampling_helper(mean: float, stdev: float, n: int):
            max_iter = 100  # Just to avoid infinite loop...
            sampled = rng.normal(mean, stdev, n)
            for _ in range(max_iter):
                neg_idxs = np.flatnonzero(sampled < 0)
                if neg_idxs.size == 0:
                    return sampled
                sampled[neg_idxs] = rng.normal(mean, stdev, neg_idxs.size)
            else:
                raise ValueError(f"Unable to sample within {max_iter} iterations")

        # Sample a bunch of grasp sites based on the distribution parameters in the config
        # Sample the locations of the grasp sites
        cyl_pts = cylindrical_halton_sampling(
            config.radius, config.length, config.n_sites, rng
        )
        R = axis_angle_to_rmat(
            *axis_angle_between_two_vectors((0, 0, 1), config.direction)
        )
        site_pts = cyl_pts @ R.T + config.center
        # Sample the pointing directions
        inward_dirs = cylinder_inward_normals(site_pts, (0, 0, 0), config.direction)
        site_dirs = np.array(
            [
                spherical_vonmises_sampling(
                    d, 1 / (config.stdev_orn_noise**2), 1, rng=rng
                )[0]
                for d in inward_dirs
            ]
        )
        # Sample the force distributions
        # Use rejection sampling in the chance that we sample a negative value for any of these
        parallel_force_means = _sampling_helper(
            config.mean_of_parallel_force_mean,
            config.stdev_of_parallel_force_mean,
            config.n_sites,
        )
        parallel_force_stdevs = _sampling_helper(
            config.mean_of_parallel_force_stdev,
            config.stdev_of_parallel_force_stdev,
            config.n_sites,
        )
        transverse_force_means = _sampling_helper(
            config.mean_of_transverse_force_mean,
            config.stdev_of_transverse_force_mean,
            config.n_sites,
        )
        transverse_force_stdevs = _sampling_helper(
            config.mean_of_transverse_force_stdev,
            config.stdev_of_transverse_force_stdev,
            config.n_sites,
        )
        # Validate the force distributions
        # Transverse force must be <= parallel force
        # Transverse stdev must be >= parallel stdev
        transverse_force_means = np.minimum(
            parallel_force_means, transverse_force_means
        )
        transverse_force_stdevs = np.maximum(
            parallel_force_stdevs, transverse_force_stdevs
        )
        # Construct the grasp sites
        sites = [
            GraspSite(
                EllipsoidalLimitSurface(
                    site_dirs[i],
                    parallel_force_means[i],
                    transverse_force_means[i],
                    parallel_force_stdevs[i],
                    transverse_force_stdevs[i],
                ),
                site_pts[i],
            )
            for i in range(config.n_sites)
        ]
        return cls(sites, config)

    def plot(
        self, color_sites: bool = True, ax: Optional[plt.Axes] = None, show: bool = True
    ) -> plt.Axes:
        """Plot the environment in a Matplotlib 3D plot

        Args:
            color_sites (bool, optional): Whether to color the grasp sites according to their quality. Defaults to True
            ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
            show (bool, optional): Whether to show the plot. Defaults to True

        Returns:
            plt.Axes: The plot
        """
        if ax is None:
            ax = gca_3d()
        ax.set_aspect("equal")

        # Color according to the quality of the grasp site
        if color_sites:
            color = self.forces_95_pct_confidence
            cmap = "rainbow_r"
        else:
            color = "k"
            cmap = None
        path_coll = ax.scatter(
            *self.positions.T,
            c=color,
            cmap=cmap,
            alpha=1,
        )
        if color_sites:
            plt.gcf().colorbar(path_coll)
        line_coll = Line3DCollection(
            [[p, p + n] for p, n in zip(self.positions, self.directions)],
            colors="k",
        )
        ax.add_collection(line_coll)
        if self.config is not None:
            ax = plot_cylinder(
                self.config.center,
                self.config.radius,
                self.config.length,
                self.config.direction,
                color=(0.9, 0.9, 0.9, 0.4),
                ax=ax,
                show=False,
            )
        if show:
            plt.show()
        return ax

    def visualize(self, client: Optional[BulletClient] = None) -> tuple[int, int]:
        """View the environment in Pybullet

        Args:
            client (BulletClient, optional): If connecting to multiple physics servers, include the client
                (the class instance, not just the ID) here. Defaults to None (use default connected client)

        Returns:
            tuple[int, int]:
                int: The Pybullet ID for the "cave" cylinder
                int: The Pybullet ID for the "grasp site" points
        """
        assert self.config is not None
        client: pybullet = pybullet if client is None else client
        connection_status = client.isConnected()
        # Bring up the Pybullet GUI if needed
        if not connection_status:
            client.connect(pybullet.GUI)
        # Construct grasp site points (colored according to quality)
        # Remap quality between 0 and 1 to use matplotlib colormap
        qualities = self.forces_95_pct_confidence
        qualities = qualities - np.min(qualities)
        qualities = qualities / np.max(qualities)
        cmap = plt.colormaps.get_cmap("RdYlGn")
        pts_colors = cmap(qualities)[:, :3]  # Remove alpha
        pts_id = visualize_points(
            [site.position for site in self.sites], pts_colors, client=client
        )
        # Construct cylinder cave
        cyl_rgba = (1.0, 0.85, 0.70, 0.8)
        cyl_scales = [
            self.config.radius,
            self.config.radius,
            self.config.length,
        ]
        cyl_orn = axis_angle_to_quat(
            *axis_angle_between_two_vectors([0, 0, 1], self.config.direction)
        )
        cyl_id = load_visual_object(
            "reachbot_manipulation/assets/meshes/hollow_cyl_inverted.obj",
            cyl_scales,
            self.config.center,
            cyl_orn,
            cyl_rgba,
            double_sided=False,
            client=client,
        )
        # Disconnect Pybullet if we originally weren't connected
        if not connection_status:
            input("Press Enter to disconnect Pybullet")
            client.disconnect()
        return cyl_id, pts_id

    def get_site_ids_in_range(
        self, position: npt.ArrayLike, distance: float
    ) -> np.ndarray:
        """Return the indices of the grasp sites in the environment that are within a given distance from a location

        Args:
            position (npt.ArrayLike): Location in the environment to measure from, shape (3,)
            distance (float): Distance, in meters

        Returns:
            np.ndarray: Indices of the grasp sites
        """
        dists = np.linalg.norm(self.positions - position, axis=-1)
        return np.flatnonzero(dists <= distance)

    def local_subset(self, position: npt.ArrayLike, distance: float) -> "Environment":
        """Constructs a new Environment instance, only containing the sites within a reachable distance from a
        certain reference point

        Args:
            position (npt.ArrayLike): Point to measure the distance about, shape (3,)
            distance (float): Maximum distance to consider, in meters

        Returns:
            Environment: A new Environment with fewer grasp sites
        """
        ids = self.get_site_ids_in_range(position, distance)
        return Environment([self.sites[i] for i in ids], config=self.config)


def cylinder_inward_normals(
    points: npt.ArrayLike, center_point: npt.ArrayLike, axis: npt.ArrayLike
) -> np.ndarray:
    """Determine the inward-facing normal direction(s) for a point or set of points on a cylinder

    Args:
        points (npt.ArrayLike): Points on a cylinder, shape (n_pts, 3)
        center_point (npt.ArrayLike): A point on the central axis of the cylinder, shape (3,)
        axis (npt.ArrayLike): The central axis of the cylinder, shape (3,)

    Returns:
        np.ndarray: Normal directions, shape (n_pts, 3)
    """
    points = np.atleast_2d(points)
    axis = normalize(axis)
    # Delta between the center point and the point on the cylinder surface
    a = points - center_point
    # Projection of the point onto the central axis
    b = np.dot(a, axis).reshape(-1, 1) * axis
    # Delta between the projected point on the axis, and the point on the surface
    c = a - b
    return normalize(-c)


def cylindrical_halton_sampling(
    radius: float,
    length: float,
    n_pts: int,
    rng: Optional[Union[int, np.random.Generator]] = None,
):
    """Sample points uniformly on the round surface of a cylinder in 3D, according to a Halton sequence.

    This assumes that the cylinder is centered at the origin, with its length parallel to the Z axis

    Args:
        radius (float): Radius of the cylinder
        length (float): Length of the cylinder
        n_pts (int): Number of points to sample
        rng (Optional[Union[int, np.random.Generator]]): Random number generator. Defaults to None (unseeded)

    Returns:
        np.ndarray: Sampled points, shape (n_pts, 3)
    """
    # Generate a 2D Halton sampling to map to the surface of the cylinder
    sampler = qmc.Halton(d=2, scramble=True, seed=rng)
    pts = sampler.random(n_pts)
    # The initial set of points will be on [0, 1] so rescale it to the dimensions of the cylinder
    pts = qmc.scale(pts, [0, -length / 2], [2 * np.pi, length / 2])
    # "wrap" the sheet of samples into a cylinder according to the radius
    x = radius * np.cos(pts[:, 0])
    y = radius * np.sin(pts[:, 0])
    z = pts[:, 1]
    return np.column_stack([x, y, z])


# TODO get typical values for density
def density_to_num_sites(density: float, radius: float, length: float) -> int:
    """Determine the number of grasp sites in a cylinder-like environment, for a given density (sites per surface area)

    Args:
        density (float): Density of grasp sites in the environmnent, in # per square meter
        radius (float): Radius of the cylindrical environment, in meters
        length (float): Length of the cylindrical environment, in meters

    Returns:
        int: Number of sites
    """
    area = 2 * np.pi * radius * length
    return round(density * area)


def plot_reachbot(
    pose: npt.ArrayLike,
    attached_points: Optional[npt.ArrayLike],
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a reachbot in the 3D environment

    Args:
        pose (npt.ArrayLike): Reachbot pose (position + XYZW quaternion), shape (7,)
        attached_points (Optional[npt.ArrayLike]): Locations in the environment where Reachbot's booms are attached,
            shape (n_booms, 3). If not attached, set this to None.
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The 3D plot
    """
    if ax is None:
        ax = gca_3d()
    ax.set_aspect("equal")
    pos = pose[:3]
    orn = pose[3:]
    rmat = quat_to_rmat(orn)
    # Slight hack: assumes a specific box dimension for the reachbot and its cables
    lwh = (LENGTH, WIDTH, HEIGHT)
    ax = plot_3d_box(pos, lwh, rmat, ax=ax, show=False)
    # If Reachbot is attached to the environment, also plot the lines for its booms
    if attached_points is not None:
        world_start_points = LOCAL_CABLE_POSITIONS @ rmat.T + pos
        lines = [[s, e] for s, e in zip(world_start_points, attached_points)]
        line_coll = Line3DCollection(lines, colors="k")
        ax.add_collection(line_coll)
    if show:
        plt.show()
    return ax


def _test_env():
    config = EnvConfig()
    env = Environment.from_config(config)
    env.plot()


def _test_env_local():
    # Constructing a subset of a larger environment based on what sites are reachable from a given position
    config = EnvConfig()
    env = Environment.from_config(config)
    print("Total number of sites in the full environment: ", env.num_sites)
    pos = np.zeros(3)
    dist = 20
    ids = env.get_site_ids_in_range(pos, dist)
    env_local = Environment([env.sites[i] for i in ids])
    print(
        "Number of sites in the local subset of the environment: ", env_local.num_sites
    )
    env_local.plot()


def _test_env_small():
    # Constructing a smaller-scale environment than what is specified in the default config
    config = EnvConfig(length=40, n_sites=20)
    env = Environment.from_config(config)
    env.plot()


def _test_env_with_reachbot():
    rng = np.random.default_rng(0)
    config = EnvConfig(length=40, n_sites=20, rng=rng)
    env = Environment.from_config(config)
    ax = env.plot(show=False)
    pose = np.array([0, 0, 0, 0, 0, 0, 1])
    attached_idxs = rng.choice(20, size=8, replace=False)
    attached_points = env.positions[attached_idxs]
    plot_reachbot(pose, attached_points, ax=ax, show=True)


def _test_env_viz():
    config = EnvConfig(length=40, n_sites=20)
    env = Environment.from_config(config)
    env.visualize()


def main():
    # _test_env_local()
    # _test_env_small()
    # _test_env()
    # _test_env_with_reachbot()
    _test_env_viz()


if __name__ == "__main__":
    main()
