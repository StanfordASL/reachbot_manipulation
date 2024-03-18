"""Trajectories

This was originally designed for a different project and then hacked together to suit ReachBot
"""

from typing import Optional, Union

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from reachbot_manipulation.utils.poses import batched_pos_quats_to_tmats
from reachbot_manipulation.utils.debug_visualizer import visualize_frame, visualize_path
from reachbot_manipulation.utils.boxes import Box
from reachbot_manipulation.config.reachbot_config import LOCAL_CABLE_POSITIONS
from reachbot_manipulation.utils.transformations import transform_points


class Trajectory:
    """Trajectory class: Keeps track of a sequence of poses/velocities/accels over a period of time

    - All arguments can be omitted as needed (for instance, a pose-only trajectory without velocities,
        or a trajectory without time information)
    - All positions/orientations/velocities... are assumed to be defined in world frame

    Args:
        positions (Optional[npt.ArrayLike]): Sequence of XYZ positions, shape (n, 3)
        quats (Optional[npt.ArrayLike]): Sequence of XYZW quaternions, shape (n, 4)
        lin_vels (Optional[npt.ArrayLike]): Sequence of (vx, vy, vz) linear velocities, shape (n, 3)
        ang_vels (Optional[npt.ArrayLike]): Sequence of (wx, wy, wz) angular velocities, shape (n, 3)
        lin_accels (Optional[npt.ArrayLike]): Sequence of (ax, ay, az) linear accelerations, shape (n, 3)
        ang_accels (Optional[npt.ArrayLike]): Sequence of (al_x, al_y, al_z) angular accelerations, shape (n, 3)
        times (Optional[npt.ArrayLike]): Times corresponding to each trajectory entry, shape (n)
    """

    def __init__(
        self,
        positions: Optional[npt.ArrayLike] = None,
        quats: Optional[npt.ArrayLike] = None,
        lin_vels: Optional[npt.ArrayLike] = None,
        ang_vels: Optional[npt.ArrayLike] = None,
        lin_accels: Optional[npt.ArrayLike] = None,
        ang_accels: Optional[npt.ArrayLike] = None,
        times: Optional[npt.ArrayLike] = None,
    ):
        self._positions = positions if positions is not None else []
        self._quats = quats if quats is not None else []
        self._lin_vels = lin_vels if lin_vels is not None else []
        self._ang_vels = ang_vels if ang_vels is not None else []
        self._lin_accels = lin_accels if lin_accels is not None else []
        self._ang_accels = ang_accels if ang_accels is not None else []
        self._times = times if times is not None else []
        self._poses = None  # Init
        self._tmats = None  # Init
        self._num_timesteps = None  # Init

    @property
    def positions(self) -> np.ndarray:
        return np.atleast_2d(self._positions)

    @property
    def quaternions(self) -> np.ndarray:
        return np.atleast_2d(self._quats)

    @property
    def linear_velocities(self) -> np.ndarray:
        return np.atleast_2d(self._lin_vels)

    @property
    def angular_velocities(self) -> np.ndarray:
        return np.atleast_2d(self._ang_vels)

    @property
    def linear_accels(self) -> np.ndarray:
        return np.atleast_2d(self._lin_accels)

    @property
    def angular_accels(self) -> np.ndarray:
        return np.atleast_2d(self._ang_accels)

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._times)

    @property
    def timestep(self) -> float | None:
        if np.size(self._times) == 0:
            return None
        return self._times[1] - self._times[0]

    @property
    def num_timesteps(self) -> int:
        if self._num_timesteps is None:
            if np.size(self.positions) > 0:
                self._num_timesteps = self.positions.shape[0]
            elif np.size(self.quaternions) > 0:
                self._num_timesteps = self.quaternions.shape[0]
        # If there is no position or orientation info, trajectory is empty (None)
        return self._num_timesteps

    @property
    def duration(self) -> float:
        return self._times[-1] - self._times[0]

    @property
    def poses(self) -> np.ndarray:
        """Pose array (position + xyzw quaternion), shape (n, 7)"""
        # if self._poses is not None:
        #     return self._poses  # Only calculate this once
        if self.positions.size == 0:
            raise ValueError("No position information available")
        if self.quaternions.size == 0:
            raise ValueError("No orientation information available")
        self._poses = np.column_stack([self.positions, self.quaternions])
        return self._poses

    @property
    def tmats(self) -> np.ndarray:
        """Poses expressed as transformation matrices, shape (n, 4, 4)"""
        # if self._tmats is not None:
        #     return self._tmats  # Only calculate this once
        self._tmats = batched_pos_quats_to_tmats(self.poses)
        return self._tmats

    @property
    def contains_pos_only(self) -> bool:
        """Whether the trajectory contains only position info"""
        return self.positions.size > 0 and self.quaternions.size == 0

    @property
    def contains_orn_only(self) -> bool:
        """Whether the trajectory contains only orientation info"""
        return self.positions.size == 0 and self.quaternions.size > 0

    @property
    def contains_pos_and_orn(self) -> bool:
        """Whether the trajectory contains both position and orientation info"""
        return self.positions.size > 0 and self.quaternions.size > 0

    @property
    def is_empty(self) -> bool:
        """Whether the trajectory contains no position/orientation info"""
        return self.positions.size == 0 and self.quaternions.size == 0

    def visualize(
        self,
        n: Optional[int] = None,
        size: float = 0.5,
        client: Optional[BulletClient] = None,
    ) -> list[int]:
        """View the trajectory in Pybullet

        Args:
            n (Optional[int]): Number of frames to plot, if plotting all of the frames is not desired.
                Defaults to None (plot all frames)
            size (float, optional): Length of the lines to plot for each frame. Defaults to 0.5 (this gives a good scale
                with respect to the dimensions of the robot)
            client (BulletClient, optional): If connecting to multiple physics servers, include the client
                (the class instance, not just the ID) here. Defaults to None (use default connected client)

        Returns:
            list[int]: Pybullet IDs for the lines drawn onto the GUI
        """
        client: pybullet = pybullet if client is None else client
        connection_status = client.isConnected()
        # Bring up the Pybullet GUI if needed
        if not connection_status:
            client.connect(pybullet.GUI)
        if self.contains_pos_and_orn:
            ids = visualize_traj(self, n, size, client=client)
        elif self.contains_pos_only:
            print("Trajectory only contains position info. Showing path instead")
            ids = visualize_path(self.positions, n, client=client)
        elif self.contains_orn_only:
            raise NotImplementedError(
                "Visualizing a sequence of purely orientations is not implemented yet"
            )
        else:  # Empty trajectory
            raise ValueError("No trajectory information to visualize")
        # Disconnect Pybullet if we originally weren't connected
        if not connection_status:
            input("Press Enter to disconnect Pybullet")
            client.disconnect()
        return ids

    def plot(self, show: bool = True) -> Figure:
        """Plot the trajectory components over time

        Args:
            show (bool, optional): Whether or not to display the plot. Defaults to True.

        Returns:
            Figure: Matplotlib figure containing the plots
        """
        return plot_traj(self, show=show)

    def get_segment(
        self, start_index: int, end_index: int, reset_time: bool = True
    ) -> "Trajectory":
        """Construct a trajectory segment from a larger trajectory

        Args:
            start_index (int): Starting index of the larger trajectory to extract the segment
            end_index (int): Ending index of the larger trajectory to extract the segment
            reset_time (bool): Whether to maintain the time association with the original trajectory,
                or reset the start time back to 0. Defaults to True (reset start time back to 0)

        Returns:
            Trajectory: A new trajectory representing a segment of the original trajectory
        """
        # TODO: add check for invalid slicing indices? Or just leave it up to numpy

        # Time needs to get handled differently because the trajectory may or may not have time info
        if np.size(self.times) == 0:  # No time info
            new_times = None
        else:
            new_times = self.times[start_index:end_index]
            if reset_time:
                new_times -= new_times[0]

        return Trajectory(
            self.positions[start_index:end_index],
            self.quaternions[start_index:end_index],
            self.linear_velocities[start_index:end_index],
            self.angular_velocities[start_index:end_index],
            self.linear_accels[start_index:end_index],
            self.angular_accels[start_index:end_index],
            new_times,
        )


class ReachbotTrajectory(Trajectory):
    """Trajectory for ReachBot

    This includes information on CDPR kinematics (cable lengths, derivatives, tensions), on top of the standard SE(3)
    positional info (+ derivatives) of the robot base

    Args:
        positions (npt.ArrayLike): Sequence of XYZ positions, shape (n, 3)
        quats (npt.ArrayLike): Sequence of XYZW quaternions, shape (n, 4)
        lin_vels (npt.ArrayLike): Sequence of (vx, vy, vz) linear velocities, shape (n, 3)
        ang_vels (npt.ArrayLike): Sequence of (wx, wy, wz) angular velocities, shape (n, 3)
        lin_accels (npt.ArrayLike): Sequence of (ax, ay, az) linear accelerations, shape (n, 3)
        ang_accels (npt.ArrayLike): Sequence of (al_x, al_y, al_z) angular accelerations, shape (n, 3)
        times (npt.ArrayLike): Times corresponding to each trajectory entry, shape (n)
        attached_points (npt.ArrayLike): Grasp site positions where the ReachBot cables are attached, shape (n_cables,)
        init_tensions (npt.ArrayLike): Static equilibrium tensions at the initial pose, shape (n_cables,)
        final_tensions (npt.ArrayLike): Static equilibrium tensions at the final pose, shape (n_cables,)
    """

    def __init__(
        self,
        positions: npt.ArrayLike,
        quats: npt.ArrayLike,
        lin_vels: npt.ArrayLike,
        ang_vels: npt.ArrayLike,
        lin_accels: npt.ArrayLike,
        ang_accels: npt.ArrayLike,
        times: npt.ArrayLike,
        attached_points: npt.ArrayLike,
        init_tensions: npt.ArrayLike,
        final_tensions: npt.ArrayLike,
    ):
        super().__init__(
            positions, quats, lin_vels, ang_vels, lin_accels, ang_accels, times
        )
        self.n_cables = LOCAL_CABLE_POSITIONS.shape[0]
        # Compute Reachbot cable lengths and speeds at each timestep
        self.cable_positions = np.array(
            [transform_points(T, LOCAL_CABLE_POSITIONS) for T in self.tmats]
        )
        self.attached_points = np.atleast_2d(attached_points)
        assert self.attached_points.shape == (self.n_cables, 3)
        self.cable_lengths = np.array(
            [
                np.linalg.norm(self.attached_points - self.cable_positions[i], axis=1)
                for i in range(self.num_timesteps)
            ]
        )
        dt = times[1] - times[0]  # HACK
        self.cable_speeds = np.gradient(self.cable_lengths, dt, axis=0)
        self.init_tensions = np.ravel(init_tensions)
        self.final_tensions = np.ravel(final_tensions)
        assert self.init_tensions.shape == (self.n_cables,)
        assert self.final_tensions.shape == (self.n_cables,)
        pcts = np.linspace(0, 1, self.num_timesteps)
        # Super simple interpolation of the tensions... Should ideally do something smarter here but this works
        self.tensions = self.init_tensions + pcts.reshape(-1, 1) @ (
            self.final_tensions - self.init_tensions
        ).reshape(1, -1)


# TODO see if we can incorporate a sequence of Boxes for the position constraints
# on a spline trajectory (rather than just a single Box constraint for a single curve)
def plot_traj_constraints(
    traj: Trajectory,
    pos_lims: Optional[Union[Box, npt.ArrayLike]] = None,
    max_vel: Optional[float] = None,
    max_accel: Optional[float] = None,
    max_omega: Optional[float] = None,
    max_alpha: Optional[float] = None,
    show: bool = True,
) -> Figure:
    """Plot trajectory info to visualize how it satisfies constraints

    Args:
        traj (Trajectory): Trajectory to plot
        pos_lims (Optional[Union[Box, npt.ArrayLike]]): Lower and upper limits on the XYZ position. Defaults to None.
        max_vel (Optional[float]): Maximum velocity magnitude. Defaults to None.
        max_accel (Optional[float]): Maximum acceleration magnitude. Defaults to None.
        max_omega (Optional[float]): Maximum angular velocity magnitude. Defaults to None.
        max_alpha (Optional[float]): Maximum angular acceleration magnitude. Defaults to None.
        show (bool, optional): Whether or not to display the plot. Defaults to True.

    Returns:
        Figure: The plot
    """
    fig = plt.figure()
    if traj.times is None or np.size(traj.times) == 0:
        x_axis = range(traj.num_timesteps)
        x_label = "Timesteps"
    else:
        x_axis = traj.times
        x_label = "Time, s"

    fmt = "k-"
    lim_fmt = "r--"
    # Top row is position info, bottom row is orientation info
    # Columns give derivative info
    subfigs = fig.subfigures(2, 3)
    # Position
    top_left = subfigs[0, 0].subplots(1, 3)
    if traj.positions.size > 0:
        for i, ax in enumerate(top_left):
            ax.plot(x_axis, traj.positions[:, i], fmt)
            ax.set_title(["x", "y", "z"][i])
            ax.set_xlabel(x_label)
        if pos_lims is not None:
            lower_pos_lim, upper_pos_lim = pos_lims
            for i, ax in enumerate(top_left):
                ax.plot(x_axis, lower_pos_lim[i] * np.ones_like(x_axis), lim_fmt)
                ax.plot(x_axis, upper_pos_lim[i] * np.ones_like(x_axis), lim_fmt)
    # Linear velocity
    if traj.linear_velocities.size > 0:
        top_middle = subfigs[0, 1].subplots(1, 1)
        top_middle.plot(x_axis, np.linalg.norm(traj.linear_velocities, axis=1), fmt)
        top_middle.set_title("||vel||")
        top_middle.set_xlabel(x_label)
        if max_vel is not None:
            top_middle.plot(x_axis, max_vel * np.ones_like(x_axis), lim_fmt)
    # Linear acceleration
    if traj.linear_accels.size > 0:
        top_right = subfigs[0, 2].subplots(1, 1)
        top_right.plot(x_axis, np.linalg.norm(traj.linear_accels, axis=1), fmt)
        top_right.set_title("||accel||")
        top_right.set_xlabel(x_label)
        if max_accel is not None:
            top_right.plot(x_axis, max_accel * np.ones_like(x_axis), lim_fmt)
    # Quaternions
    # These are unconstrained so it's the same plotting method as in the standard plot traj function
    bot_left = subfigs[1, 0].subplots(1, 4)
    if traj.quaternions.size > 0:
        _plot(
            bot_left, traj.quaternions, ["qx", "qy", "qz", "qw"], x_axis, x_label, fmt
        )
    # Angular velocity
    bot_middle = subfigs[1, 1].subplots(1, 1)
    if traj.angular_velocities.size > 0:
        bot_middle.plot(x_axis, np.linalg.norm(traj.angular_velocities, axis=1), fmt)
        bot_middle.set_title("||omega||")
        bot_middle.set_xlabel(x_label)
        if max_omega is not None:
            bot_middle.plot(x_axis, max_omega * np.ones_like(x_axis), lim_fmt)
    # Angular acceleration
    bot_right = subfigs[1, 2].subplots(1, 1)
    if traj.angular_accels.size > 0:
        bot_right.plot(x_axis, np.linalg.norm(traj.angular_accels, axis=1), fmt)
        bot_right.set_title("||alpha||")
        bot_right.set_xlabel(x_label)
        if max_alpha is not None:
            bot_right.plot(x_axis, max_alpha * np.ones_like(x_axis), lim_fmt)
    if show:
        plt.show()
    return fig


def plot_traj(traj: Trajectory, show: bool = True, fmt: str = "k-") -> Figure:
    """Plot the trajectory components over time

    Args:
        traj (Trajectory): The Trajectory object to plot
        show (bool, optional): Whether or not to display the plot. Defaults to True.
        fmt (str, optional): Matplotlib line specification. Defaults to "k-"

    Returns:
        Figure: Matplotlib figure containing the plots
    """

    # Indexing helper variables
    POS = 0
    ORN = 1
    LIN_VEL = 2
    ANG_VEL = 3
    LIN_ACCEL = 4
    ANG_ACCEL = 5

    labels = {
        POS: ["x", "y", "z"],
        ORN: ["qx", "qy", "qz", "qw"],
        LIN_VEL: ["vx", "vy", "vz"],
        ANG_VEL: ["wx", "wy", "wz"],
        LIN_ACCEL: ["ax", "ay", "az"],
        ANG_ACCEL: ["al_x", "al_y", "al_z"],
    }

    fig = plt.figure()
    if traj.times is None or np.size(traj.times) == 0:
        x_axis = range(traj.num_timesteps)
        x_label = "Timesteps"
    else:
        x_axis = traj.times
        x_label = "Time, s"
    # Top row is position info, bottom row is orientation info
    # Columns give derivative info
    subfigs = fig.subfigures(2, 3)
    top_left = subfigs[0, 0].subplots(1, 3)
    _plot(top_left, traj.positions, labels[POS], x_axis, x_label, fmt)
    top_middle = subfigs[0, 1].subplots(1, 3)
    _plot(top_middle, traj.linear_velocities, labels[LIN_VEL], x_axis, x_label, fmt)
    top_right = subfigs[0, 2].subplots(1, 3)
    _plot(top_right, traj.linear_accels, labels[LIN_ACCEL], x_axis, x_label, fmt)
    bot_left = subfigs[1, 0].subplots(1, 4)
    _plot(bot_left, traj.quaternions, labels[ORN], x_axis, x_label, fmt)
    bot_middle = subfigs[1, 1].subplots(1, 3)
    _plot(bot_middle, traj.angular_velocities, labels[ANG_VEL], x_axis, x_label, fmt)
    bot_right = subfigs[1, 2].subplots(1, 3)
    _plot(bot_right, traj.angular_accels, labels[ANG_ACCEL], x_axis, x_label, fmt)
    if show:
        plt.show()
    return fig


def _plot(
    axes: np.ndarray[plt.Axes],
    data: np.ndarray,
    labels: list[str],
    x_axis: np.ndarray,
    x_label: str,
    *args,
    **kwargs,
):
    """Helper function for plotting trajectory components

    Args:
        axes (np.ndarray[plt.Axes]): Matplotlib axes for subplots within a subfigure, length = n
        data (np.ndarray): Trajectory information to plot, shape (m, n) where m is the number of timesteps
            and n refers to the number of components of that trajectory info (for instance, position has
            data for x, y, and z, so n = 3)
        labels (list[str]): Labels for each of the components of the trajectory data, length = n
        x_axis (np.ndarray): X-axis data to plot the trajectory against, length = m
        x_label (str): Label for the x-axis (for instance, "Time" or "Steps")
    """
    # If the trajectory doesn't contain the info, don't plot it
    if np.size(data) == 0 or data is None:
        return
    # Number of components to plot (for instance, position: n = 3: x, y, z)
    n = data.shape[1]
    assert n == len(labels)
    assert n == len(axes)
    # Plot each component of the trajectory information on a separate axis
    for i, ax in enumerate(axes):
        ax.plot(x_axis, data[:, i], *args, **kwargs)
        ax.set_title(labels[i])
        ax.set_xlabel(x_label)


# TODO all of the plotting logic here is extremely similar to the single-plot method
# Figure out a way to simplify the code
def compare_trajs(
    traj_1: Trajectory,
    traj_2: Trajectory,
    show: bool = True,
    fmt_1: str = "k-",
    fmt_2: str = "b-",
) -> Figure:
    """Compares two trajectories by plotting them on the same axes

    Args:
        traj_1 (Trajectory): First trajectory to plot
        traj_2 (Trajectory): Second trajectory to plot
        show (bool, optional): . Defaults to True.
        fmt_1 (str, optional): Matplotlib line specification for the first traj. Defaults to "k-".
        fmt_2 (str, optional): Matplotlib line specification for the second traj. Defaults to "b-".

    Returns:
        Figure: Matplotlib figure containing the plots
    """
    # Indexing helper variables
    POS = 0
    ORN = 1
    LIN_VEL = 2
    ANG_VEL = 3
    LIN_ACCEL = 4
    ANG_ACCEL = 5

    labels = {
        POS: ["x", "y", "z"],
        ORN: ["qx", "qy", "qz", "qw"],
        LIN_VEL: ["vx", "vy", "vz"],
        ANG_VEL: ["wx", "wy", "wz"],
        LIN_ACCEL: ["ax", "ay", "az"],
        ANG_ACCEL: ["al_x", "al_y", "al_z"],
    }

    fig = plt.figure()
    # TODO this check is kinda weird right now
    # what happens if one has time info and the other doesn't??
    if traj_1.times is None or np.size(traj_1.times) == 0:
        x_axis_1 = range(traj_1.num_timesteps)
        x_label = "Timesteps"
    else:
        x_axis_1 = traj_1.times
        x_label = "Time, s"
    if traj_2.times is None or np.size(traj_2.times) == 0:
        x_axis_2 = range(traj_2.num_timesteps)
    else:
        x_axis_2 = traj_2.times
    # Top row is position info, bottom row is orientation info
    # Columns give derivative info
    # fmt: off
    subfigs = fig.subfigures(2, 3)
    top_left = subfigs[0, 0].subplots(1, 3)
    _plot(top_left, traj_1.positions, labels[POS], x_axis_1, x_label, fmt_1)
    _plot(top_left, traj_2.positions, labels[POS], x_axis_2, x_label, fmt_2)
    top_middle = subfigs[0, 1].subplots(1, 3)
    _plot(top_middle, traj_1.linear_velocities, labels[LIN_VEL], x_axis_1, x_label, fmt_1)
    _plot(top_middle, traj_2.linear_velocities, labels[LIN_VEL], x_axis_2, x_label, fmt_2)
    top_right = subfigs[0, 2].subplots(1, 3)
    _plot(top_right, traj_1.linear_accels, labels[LIN_ACCEL], x_axis_1, x_label, fmt_1)
    _plot(top_right, traj_2.linear_accels, labels[LIN_ACCEL], x_axis_2, x_label, fmt_2)
    bot_left = subfigs[1, 0].subplots(1, 4)
    _plot(bot_left, traj_1.quaternions, labels[ORN], x_axis_1, x_label, fmt_1)
    _plot(bot_left, traj_2.quaternions, labels[ORN], x_axis_2, x_label, fmt_2)
    bot_mid = subfigs[1, 1].subplots(1, 3)
    _plot(bot_mid, traj_1.angular_velocities, labels[ANG_VEL], x_axis_1, x_label, fmt_1)
    _plot(bot_mid, traj_2.angular_velocities, labels[ANG_VEL], x_axis_2, x_label, fmt_2)
    bot_right = subfigs[1, 2].subplots(1, 3)
    _plot(bot_right, traj_1.angular_accels, labels[ANG_ACCEL], x_axis_1, x_label, fmt_1)
    _plot(bot_right, traj_2.angular_accels, labels[ANG_ACCEL], x_axis_2, x_label, fmt_2)
    # fmt: on
    if show:
        plt.show()
    return fig


def visualize_traj(
    traj: Union[Trajectory, npt.ArrayLike],
    n: Optional[int] = None,
    size: float = 0.5,
    client: Optional[BulletClient] = None,
) -> list[int]:
    """Visualizes a trajectory's sequence of poses on the Pybullet GUI

    Args:
        traj (Union[Trajectory, npt.ArrayLike]): Trajectory to visualize (must contain at least
            position + orientation info), or an array of position + quaternion poses, shape (n, 7)
        n (Optional[int]): Number of frames to plot, if plotting all of the frames is not desired.
            Defaults to None (plot all frames)
        size (float, optional): Length of the lines to plot for each frame. Defaults to 0.5 (this gives a good scale
            with respect to the dimensions of the robot)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        list[int]: Pybullet IDs for the lines drawn onto the GUI
    """
    client: pybullet = pybullet if client is None else client
    if isinstance(traj, Trajectory):
        traj = traj.poses
    else:
        # If there is more information (velocity, time) in our array, only use the pose info
        traj = np.atleast_2d(traj)[:, :7]
    n_frames = traj.shape[0]
    # If desired, sample frames evenly across the trajectory to plot a subset
    if n is not None and n < n_frames:
        # This indexing ensures that the first and last frames are plotted
        idx = np.round(np.linspace(0, n_frames - 1, n, endpoint=True)).astype(int)
        traj = traj[idx, :]
    tmats = batched_pos_quats_to_tmats(traj)
    ids = []
    for i in range(tmats.shape[0]):
        ids += visualize_frame(tmats[i, :, :], size, client=client)
    return ids


def concatenate_trajs(traj_1: Trajectory, traj_2: Trajectory) -> Trajectory:
    """Combine two trajectories one after the other

    This will follow the first trajectory until its end, then follow the second one until it ends

    Args:
        traj_1 (Trajectory): First trajectory
        traj_2 (Trajectory): Second trajectory

    Returns:
        Trajectory: Combined trajectory
    """
    # Ensure continuity in time
    # TODO check for continuity in all other components?
    # TODO this assumes that both have time information
    dt = traj_1.times[-1] - traj_1.times[-2]
    if np.isclose(traj_2.times[0], 0):
        times = np.concatenate([traj_1.times, traj_2.times + traj_1.times[-1] + dt])
    elif np.isclose(traj_2.times[0], traj_1.times[-1] + dt):
        times = np.concatenate([traj_1.times, traj_2.times])
    return Trajectory(
        np.vstack([traj_1.positions, traj_2.positions]),
        np.vstack(
            [traj_1.quaternions, traj_2.quaternions],
        ),
        np.vstack([traj_1.linear_velocities, traj_2.linear_velocities]),
        np.vstack([traj_1.angular_velocities, traj_2.angular_velocities]),
        np.vstack([traj_1.linear_accels, traj_2.linear_accels]),
        np.vstack([traj_1.angular_accels, traj_2.angular_accels]),
        times,
    )
