"""Manipulation tasks

We can define these either as:
- A single wrench
- A single wrench with uncertainty (expressed as a gaussian)
- A set of wrenches with uncertainty (gaussian mixture model)
- An explicit time-series sequence of wrenches

When ReachBot plans a manipulation task, it observes the environment from a centrally-located position, where it can
identify all nearby grasp sites, and the area of interest for the task. In the case of a pick-and-place task, it can
estimate the pose required to grasp the object, and the mass of the object (yielding an estimated wrench required
to pick it up). However, these true values are uncertain due to perception noise, disturbances, and other factors.

So, this file generalizes a "task" as distributions about nominal wrenches/poses, to incorporate this uncertainty.

Note on reasonable variance values:
- Position variance: 0.05 gives a worst-case positional error of about 0.5 meters
- Orientation variance: 0.001 gives a worst-case angular error of about 10 degrees in each roll/pitch/yaw component
- Wrench variance: This is going to be task-dependent but for instance, a force variance of 1 gives a worst-case
  force error of about 3 N. Or, a variance of 0.1 gives a worst-case error of about 1 N. Likewise for torque, but
  the units will just be Nm
"""

from typing import Union, Any, Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm
import pytransform3d.batch_rotations as brt

from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.rotations import quat_to_rmat, rmat_to_fixed_xyz
from reachbot_manipulation.utils.quaternions import wxyz_to_xyzw


class SinglePoseTask:
    """Defining a task by a nominal pose/wrench pair, with some noise to model:

    - Uncertainty in the "true" wrench to achieve the task
    - Uncertainty in the true localization of the reachbot in the cave
    - Vibrations or other disturbances experienced while executing the task

    Args:
        pose (npt.ArrayLike): Nominal pose (position + XYZW quaternion) required to apply the wrench, shape (7,)
        wrench (npt.ArrayLike): Nominal wrench (force + torque) for the task, shape (6,)
        variance (npt.ArrayLike): Variances in each component of the wrench, shape (6,)
        pos_variance (npt.ArrayLike): Variances in each component of the ReachBot position, shape (3,)
        orn_variance (float): Variance in the angular error (radians^2)
        rng (Optional[Union[int, np.random.Generator]]): Random number generator. Defaults to None (unseeded)
    """

    def __init__(
        self,
        pose: npt.ArrayLike,
        wrench: npt.ArrayLike,
        wrench_variance: npt.ArrayLike,
        pos_variance: npt.ArrayLike,
        orn_variance: float,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.pose = np.ravel(pose)
        self.wrench = np.ravel(wrench)
        self.wrench_variance = np.ravel(wrench_variance)
        self.pos_variance = np.ravel(pos_variance)
        self.orn_variance = orn_variance
        assert self.pose.shape == (7,)
        assert self.wrench.shape == (6,)
        assert self.wrench_variance.shape == (6,)
        assert self.pos_variance.shape == (3,)
        assert isinstance(self.orn_variance, (float, int))
        self.wrench_cov = np.diag(self.wrench_variance)
        self.pos_cov = np.diag(self.pos_variance)
        self.rng = np.random.default_rng(rng)

    def sample_wrenches(self, n: int) -> np.ndarray:
        """Samples a set of wrenches from the distribution

        Args:
            n (int): Number of wrenches to sample

        Returns:
            np.ndarray: Wrenches, shape (n, 6)
        """
        return self.rng.multivariate_normal(self.wrench, self.wrench_cov, n)

    def sample_poses(self, n: int) -> np.ndarray:
        """Samples a set of poses from the distribution

        Args:
            n (int): Number of poses to sample

        Returns:
            np.ndarray: Poses, shape (n, 7)
        """
        positions = self.rng.multivariate_normal(self.pose[:3], self.pos_cov, n)
        orns = sample_orientations(
            self.pose[3:], np.sqrt(self.orn_variance), n, self.rng
        )
        return np.column_stack([positions, orns])

    def plot_wrench_distribution(
        self, n_samples: int = 1000, show: bool = True
    ) -> tuple[plt.Figure, Any]:
        """Samples wrenches about the nominal desired wrench according to the distribution, and plots a histogram
        for visualizing the expected range of force/torque noise

        Args:
            n_samples (int, optional): Number of wrenches to sample. Defaults to 1000.
            show (bool, optional): Whether or not to show the plot. Defaults to True.

        Returns:
            tuple[plt.Figure, Any]:
                plt.Figure: The figure
                Any: An array of Axes objects
        """
        fig, axs = plt.subplots(2, 3)
        samples = self.sample_wrenches(n_samples)
        deltas = samples - self.wrench
        titles = ["Δfx (N)", "Δfy (N)", "Δfz (N)", "ΔTx (Nm)", "ΔTy (Nm)", "ΔTz (Nm)"]
        # Plot histograms
        for i, ax in enumerate(axs.flat):
            ax.hist(deltas[:, i], density=True)
            ax.set_title(titles[i])
        # Plot the gaussian over the histogram
        for i, ax in enumerate(axs.flat):
            xvals = np.linspace(*ax.get_xlim(), 100, endpoint=True)
            ax.plot(xvals, norm.pdf(xvals, 0, np.sqrt(self.wrench_variance[i])))
        fig.suptitle(
            f"Wrench distribution about {np.array2string(self.wrench, precision=2)}"
        )
        if show:
            plt.show()
        return fig, axs

    def plot_pose_distribution(
        self, n_samples: int = 1000, show: bool = True
    ) -> tuple[plt.Figure, Any]:
        """Samples poses about the nominal target pose according to the distribution, and plots a histogram for
        visualizing the expected range of position/orientation noise

        Args:
            n_samples (int, optional): Number of poses to sample. Defaults to 1000.
            show (bool, optional): Whether or not to show the plot. Defaults to True.

        Returns:
            tuple[plt.Figure, Any]:
                plt.Figure: The figure
                Any: An array of Axes objects
        """
        fig, axs = plt.subplots(2, 3)
        pos_axs = axs[0, :]
        orn_axs = axs[1, :]
        poses = self.sample_poses(n_samples)
        orns = poses[:, 3:]
        titles = [
            "Δx (m)",
            "Δy (m)",
            "Δz (m)",
            "Δroll (deg)",
            "Δpitch (deg)",
            "Δyaw (deg)",
        ]
        pos_deltas = poses[:, :3] - self.pose[:3]
        nominal_quat = self.pose[3:]
        nominal_rmat = quat_to_rmat(nominal_quat)
        sampled_rmats = [quat_to_rmat(q) for q in orns]
        # R_A2B = R_B2W.T @ R_A2W
        rmat_deltas = [R.T @ nominal_rmat for R in sampled_rmats]
        rpy_deltas = np.array([rmat_to_fixed_xyz(rmat) for rmat in rmat_deltas])
        rpy_deltas_deg = np.rad2deg(rpy_deltas)
        for i, ax in enumerate(axs.flat):
            ax.set_title(titles[i])
        for i, ax in enumerate(pos_axs):
            ax.hist(pos_deltas[:, i], density=True)
        for i, ax in enumerate(orn_axs):
            ax.hist(rpy_deltas_deg[:, i], density=True)
        fig.suptitle(
            f"Pose distribution about {np.array2string(self.pose, precision=2)}"
        )
        if show:
            plt.show()
        return fig, axs


class MultiPoseTask:
    """Defining a task by a set of pose/wrench pairs, with some noise to model:

    - Uncertainty in the "true" wrench to achieve the task
    - Uncertainty in the true localization of the reachbot in the cave
    - Vibrations or other disturbances experienced while executing the task

    Args:
        poses (npt.ArrayLike): Nominal poses (position + XYZW quaternion) required to apply each wrench,
            shape (n_wrenches, 7)
        wrenches (npt.ArrayLike): Nominal wrenches (force + torque) for the task, shape (n_wrenches, 6)
        wrench_variances (npt.ArrayLike): Variances in each component of each wrench, shape (n_wrenches, 6)
        pos_variances (npt.ArrayLike): Variances in each component of each ReachBot position, shape (n_wrenches, 3)
        orn_variances (npt.ArrayLike): Variances in the angular error for each pose, shape (n_wrenches,)
        rng (Optional[Union[int, np.random.Generator]]): Random number generator. Defaults to None (unseeded)
    """

    def __init__(
        self,
        poses: npt.ArrayLike,
        wrenches: npt.ArrayLike,
        wrench_variances: npt.ArrayLike,
        pos_variances: npt.ArrayLike,
        orn_variances: npt.ArrayLike,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.poses = np.atleast_2d(poses)
        self.wrenches = np.atleast_2d(wrenches)
        self.wrench_variances = np.atleast_2d(wrench_variances)
        self.pos_variances = np.atleast_2d(pos_variances)
        self.orn_variances = np.ravel(orn_variances)
        self.n_wrenches = self.wrenches.shape[0]
        assert self.poses.shape == (self.n_wrenches, 7)
        assert self.wrenches.shape == (self.n_wrenches, 6)
        assert self.wrench_variances.shape == (self.n_wrenches, 6)
        assert self.pos_variances.shape == (self.n_wrenches, 3)
        assert self.orn_variances.shape == (self.n_wrenches,)
        self.wrench_covs = np.array([np.diag(vs) for vs in self.wrench_variances])
        self.pos_covs = np.array([np.diag(vs) for vs in self.pos_variances])
        self.rng = np.random.default_rng(rng)

    def sample_wrenches(self, n: int) -> np.ndarray:
        """Samples sets of wrenches from the distribution

        Indexing example:
        >>> wrenches = sample_wrenches(10) # shape (10, n_wrenches_in_task, 6)
        >>> first_set_of_sampled_task_wrenches = wrenches[0] # shape (n_wrenches_in_task, 6)

        Args:
            n (int): Number of sets of wrenches to sample

        Returns:
            np.ndarray: Wrench sets, shape (n, n_wrenches_in_task, 6)
        """
        return np.stack(
            [
                self.rng.multivariate_normal(self.wrenches[i], self.wrench_covs[i], n)
                for i in range(self.n_wrenches)
            ],
            axis=1,
        )

    def sample_poses(self, n: int) -> np.ndarray:
        """Samples sets of poses from the distribution

        Indexing example:
        >>> poses = sample_poses(10) # shape (10, n_poses_in_task, 6)
        >>> first_set_of_sampled_task_poses = poses[0] # shape (n_poses_in_task, 6)

        Args:
            n (int): Number of sets of poses to sample

        Returns:
            np.ndarray: Pose sets, shape (n, n_poses_in_task, 7)
        """
        poses = []
        orn_stdevs = np.sqrt(self.orn_variances)
        for i in range(self.n_wrenches):
            positions = self.rng.multivariate_normal(
                self.poses[i][:3], self.pos_covs[i], n
            )
            orns = sample_orientations(self.poses[i][3:], orn_stdevs[i], n, self.rng)
            poses.append(np.column_stack([positions, orns]))
        return np.stack(poses, axis=1)

    def plot_wrench_distributions(
        self, n_samples: int = 1000, show: bool = True
    ) -> plt.Figure:
        """Samples wrenches about the nominal desired wrench according to the distribution, and plots histograms
        for visualizing the expected range of force/torque noise for each wrench

        Args:
            n_samples (int, optional): Number of wrenches to sample. Defaults to 1000.
            show (bool, optional): Whether or not to show the plot. Defaults to True.

        Returns:
            plt.Figure: The matplotlib figures for each wrench in the task
        """
        samples = self.sample_wrenches(n_samples)
        fig = plt.figure()
        subfigs = fig.subfigures(1, self.n_wrenches)
        for wi in range(self.n_wrenches):
            # fig, axs = plt.subplots(2, 3)
            subfig = subfigs[wi]
            axs = subfig.subplots(2, 3)
            deltas = samples[:, wi] - self.wrenches[wi]
            titles = [
                "Δfx (N)",
                "Δfy (N)",
                "Δfz (N)",
                "ΔTx (Nm)",
                "ΔTy (Nm)",
                "ΔTz (Nm)",
            ]
            # Plot histograms
            for i, ax in enumerate(axs.flat):
                ax.hist(deltas[:, i], density=True)
                ax.set_title(titles[i])
            # Plot the gaussian over the histogram
            for i, ax in enumerate(axs.flat):
                xvals = np.linspace(*ax.get_xlim(), 100, endpoint=True)
                ax.plot(
                    xvals, norm.pdf(xvals, 0, np.sqrt(self.wrench_variances[wi, i]))
                )
            subfig.suptitle(
                f"Wrench (#{wi}): distribution about\n{np.array2string(self.wrenches[wi], precision=2)}"
            )
        if show:
            plt.show()
        return fig

    def plot_pose_distributions(
        self, n_samples: int = 1000, show: bool = True
    ) -> plt.Figure:
        """Samples poses about the nominal target pose according to the distribution, and plots a histogram for
        visualizing the expected range of position/orientation noise

        Args:
            n_samples (int, optional): Number of poses to sample. Defaults to 1000.
            show (bool, optional): Whether or not to show the plot. Defaults to True.

        Returns:
            plt.Figure: The matplotlib figures for each pose in the task
        """
        poses = self.sample_poses(n_samples)
        titles = [
            "Δx (m)",
            "Δy (m)",
            "Δz (m)",
            "Δroll (deg)",
            "Δpitch (deg)",
            "Δyaw (deg)",
        ]
        fig = plt.figure()
        subfigs = fig.subfigures(1, self.n_wrenches)
        for p in range(self.n_wrenches):
            # fig, axs = plt.subplots(2, 3)
            subfig = subfigs[p]
            axs = subfig.subplots(2, 3)
            pos_axs = axs[0, :]
            orn_axs = axs[1, :]
            pos_deltas = poses[:, p, :3] - self.poses[p, :3]
            nominal_quat = self.poses[p, 3:]
            nominal_rmat = quat_to_rmat(nominal_quat)
            sampled_rmats = [quat_to_rmat(q) for q in poses[:, p, 3:]]
            # R_A2B = R_B2W.T @ R_A2W
            rmat_deltas = [R.T @ nominal_rmat for R in sampled_rmats]
            rpy_deltas = np.array([rmat_to_fixed_xyz(rmat) for rmat in rmat_deltas])
            rpy_deltas_deg = np.rad2deg(rpy_deltas)
            for i, ax in enumerate(axs.flat):
                ax.set_title(titles[i])
            for i, ax in enumerate(pos_axs):
                ax.hist(pos_deltas[:, i], density=True)
            for i, ax in enumerate(orn_axs):
                ax.hist(rpy_deltas_deg[:, i], density=True)
            subfig.suptitle(
                f"Pose (#{p}) distribution about\n{np.array2string(self.poses[p], precision=2)}"
            )
        if show:
            plt.show()
        return fig


def sample_orientations(
    q: npt.ArrayLike,
    stdev: float,
    n: int,
    rng: Union[int, np.random.Generator, None] = None,
) -> np.ndarray:
    """Sample quaternions about a nominal orientation, according to a Gaussian-distributed angular error

    Args:
        q (npt.ArrayLike): Nominal orientation (XYZW quaternion), shape (4,)
        stdev (float): Standard deviation of the angular error, in radians
        n (int): Number of quaternions to sample
        rng (Union[int, np.random.Generator, None], optional): Random number generator. Defaults to None (unseeded)

    Returns:
        np.ndarray: XYZW quaternions, shape (n, 4)
    """
    # Edge case: if the stdev is 0 then just return a bunch of the nominal orientation
    if np.isclose(stdev, 0, atol=1e-10):
        return q * np.ones((n, 1))
    rng = np.random.default_rng(rng)
    R = quat_to_rmat(q)
    # Sample random axes
    axes = normalize(rng.normal(0, 1, size=(n, 3)))
    # Sample random angles according to variance (mean 0)
    angles = rng.normal(0, stdev, size=n)
    # Apply axis-angle transformation to the nominal orientation
    noise_rmats = brt.matrices_from_compact_axis_angles(angles.reshape(-1, 1) * axes)
    new_orn_rmats = noise_rmats @ R.T
    # Convert back to XYZW quaternions
    # NOTE: Using pytransform quaternion function which uses WXYZ not XYZW
    return wxyz_to_xyzw(brt.quaternions_from_matrices(new_orn_rmats))


def multivariate_gaussian_interpolation(
    mu_1: npt.ArrayLike,
    cov_1: npt.ArrayLike,
    mu_2: npt.ArrayLike,
    cov_2: npt.ArrayLike,
    t: npt.ArrayLike,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Sample a point interpolated between two multivariate gaussian distributions

    Args:
        mu_1 (npt.ArrayLike): Mean of the first distribution, shape (dim,)
        cov_1 (npt.ArrayLike): Covariance of the first distribution, shape (dim, dim)
        mu_2 (npt.ArrayLike): Mean of the second distribution, shape (dim,)
        cov_2 (npt.ArrayLike): Covariance of the second distribution, shape (dim, dim)
        t (npt.ArrayLike): Interpolation percentage
        rng (Optional[Union[int, np.random.Generator]]): Random number generator. Defaults to None (unseeded)

    Returns:
        np.ndarray: Sampled point, shape (dim,)
    """
    rng = np.random.default_rng(rng)
    cov_1 = np.atleast_2d(cov_1)
    cov_2 = np.atleast_2d(cov_2)
    if cov_1.shape[0] != cov_1.shape[1] or cov_2.shape[0] != cov_2.shape[1]:
        raise ValueError("Covariance matrices must be square")
    if isinstance(t, (float, int)):
        if not 0 <= t <= 1:
            raise ValueError("t must be between 0 and 1")
        pt_1 = rng.multivariate_normal(mu_1, cov_1)
        pt_2 = rng.multivariate_normal(mu_2, cov_2)
        return (1 - t) * pt_1 + t * pt_2
    elif isinstance(t, (list, np.ndarray)):
        t = np.asarray(t)
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError("t must be between 0 and 1")
        n = len(t)
        pts_1 = rng.multivariate_normal(mu_1, cov_1, n)
        pts_2 = rng.multivariate_normal(mu_2, cov_2, n)
        return (1 - t) * pts_1 + t * pts_2
    else:
        raise ValueError("Unexpected format for t")


def test_task():
    pose = [100, 200, 300, 0, 0, 0, 1]
    wrench = [0, 0, 5, 0, 0, 0]
    wrench_variance = [1, 1, 3, 0.1, 0.1, 0.1]
    pos_variance = [0.05, 0.05, 0.05]
    orn_variance = np.deg2rad(2) ** 2
    task = SinglePoseTask(pose, wrench, wrench_variance, pos_variance, orn_variance)
    task.plot_wrench_distribution(show=False)
    task.plot_pose_distribution(show=True)


def test_multi_wrench_task():
    # pylint: disable=import-outside-toplevel
    from reachbot_manipulation.utils.quaternions import random_quaternion

    rng = np.random.default_rng(0)
    n_wrenches = 3
    poses = np.array(
        [
            np.concatenate([rng.random(3), random_quaternion()])
            for _ in range(n_wrenches)
        ]
    )
    wrenches = rng.random((n_wrenches, 6))
    wrench_variances = rng.random((n_wrenches, 6))
    pos_variances = rng.random((n_wrenches, 3))
    orn_variances = (rng.random(n_wrenches) * np.deg2rad(3)) ** 2
    task = MultiPoseTask(
        poses, wrenches, wrench_variances, pos_variances, orn_variances
    )
    task.plot_wrench_distributions(show=False)
    task.plot_pose_distributions(show=True)


def main():
    # test_interp()
    # test_task()
    test_multi_wrench_task()


if __name__ == "__main__":
    main()
