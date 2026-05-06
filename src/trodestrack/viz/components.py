"""Reusable visualization components (artists) for video generation.

Each artist class manages a visual element (rat, LED, trail, etc.) and provides
an update() method to modify appearance for each video frame.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from jax import Array
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.text import Text
from matplotlib.transforms import Affine2D

from trodestrack.viz.styles import COLORS


class RatArtist:
    """Visualize rat body, orientation, and velocity.

    Represents the rat as a circle (meters) with a heading arrow and a velocity
    arrow. Methods update matplotlib artists in place for efficient blitting.
    """

    def __init__(self, ax: Axes, body_radius: float = 0.03):
        """Initialize rat artist.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to draw on.
        body_radius : float, default 0.03
            Body circle radius (m).
        """
        self.ax = ax
        self.body_radius = body_radius

        # Rat body: gray circle with white edge for visibility
        self.body = Circle(
            (0, 0),
            radius=body_radius,
            facecolor=COLORS["gray"],
            edgecolor="white",
            linewidth=2,
            alpha=0.9,  # Increased from 0.8 for prominence
            zorder=5,
        )
        ax.add_patch(self.body)

        # Heading arrow: large, prominent arrow indicating orientation
        # Use FancyArrowPatch with transforms for efficient updates
        arrow_length = body_radius * 1.8
        self.heading_arrow = FancyArrowPatch(
            (0, 0),
            (arrow_length, 0),
            mutation_scale=body_radius * 30,
            linewidth=1.5,
            edgecolor="white",
            facecolor="black",
            alpha=1.0,
            zorder=7,
        )
        ax.add_patch(self.heading_arrow)

        # Velocity vector: colored arrow showing motion direction and speed
        # Different color from heading to distinguish orientation vs motion
        # Initialize with same length as heading arrow
        self.velocity_arrow = FancyArrowPatch(
            (0, 0),
            (arrow_length, 0),
            mutation_scale=body_radius * 25,
            linewidth=1.0,
            edgecolor="white",
            facecolor=COLORS["purple"],
            alpha=0.7,
            zorder=6,
            visible=False,  # Hidden initially
        )
        ax.add_patch(self.velocity_arrow)

    def update(
        self, x: float, y: float, theta: float, vx: float = 0.0, vy: float = 0.0
    ) -> list[Any]:
        """Update rat position, orientation, and velocity.

        Parameters
        ----------
        x : float
            X position (m).
        y : float
            Y position (m).
        theta : float
            Heading angle (rad), 0 = +X axis, CCW positive.
        vx : float, default 0.0
            X velocity (m/s).
        vy : float, default 0.0
            Y velocity (m/s).

        Returns
        -------
        list[Any]
            Modified artists for blitting.
        """
        # Update body position
        self.body.center = (x, y)

        # Update heading arrow using transform (no recreation needed)
        T_heading = (
            Affine2D().rotate_around(0, 0, theta).translate(x, y) + self.ax.transData
        )
        self.heading_arrow.set_transform(T_heading)

        # Update velocity arrow using transform (scale by speed + rotate by velocity direction)
        speed = np.hypot(vx, vy)

        if speed > 0.01:  # Only show if moving > 1 cm/s
            vel_angle = np.arctan2(vy, vx)
            vel_scale = 4.0  # Amplification for visibility
            vel_length = speed * vel_scale

            # Scale arrow length relative to default, then rotate and translate
            arrow_length = self.body_radius * 1.8
            scale_factor = vel_length / arrow_length
            T_vel = (
                Affine2D()
                .scale(scale_factor, 1.0)
                .rotate_around(0, 0, vel_angle)
                .translate(x, y)
                + self.ax.transData
            )
            self.velocity_arrow.set_transform(T_vel)
            self.velocity_arrow.set_visible(True)
        else:
            # Hide when stationary
            self.velocity_arrow.set_visible(False)

        return [self.body, self.heading_arrow, self.velocity_arrow]


class LEDArtist:
    """Visualize LED detections with confidence indication.

    Shows LED position when visible (circle + confidence halo),
    or last known position with red X when dropped out.
    """

    def __init__(
        self,
        ax: Axes,
        led_id: int,
        color: str,
        marker_size: float = 0.008,
        show_residuals: bool = False,
    ):
        """Initialize LED artist.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to draw on.
        led_id : int
            LED identifier (1 or 2).
        color : str
            Color for this LED (e.g., COLORS["blue"]).
        marker_size : float, default 0.008
            Radius of LED marker (m).
        show_residuals : bool, default False
            Show residual lines from expected to observed position.
        """
        self.led_id = led_id
        self.color = color
        self.marker_size = marker_size
        self.show_residuals = show_residuals

        # LED marker: colored circle (reduced size for better hierarchy)
        (self.marker,) = ax.plot(
            [],
            [],
            "o",
            color=color,
            markersize=6,  # Reduced from 8
            markeredgewidth=1.0,  # Reduced from 1.5
            markeredgecolor="white",
            label=f"LED{led_id}",
            zorder=10,
        )

        # Confidence halo: transparent circle scaled by confidence
        self.halo = Circle(
            (0, 0), radius=marker_size * 2, color=color, alpha=0.0, zorder=9
        )
        ax.add_patch(self.halo)

        # Dropout marker: red X at last known position (subtle)
        (self.dropout_marker,) = ax.plot(
            [],
            [],
            "x",
            color=COLORS["red"],
            markersize=8,
            linewidth=2,
            alpha=0.7,
            zorder=11,
        )

        # Residual visualization: expected position (small cross) + residual line
        self.expected_marker: Line2D | None
        self.residual_line: Line2D | None
        if show_residuals:
            # Expected position marker (small cross)
            (self.expected_marker,) = ax.plot(
                [],
                [],
                "+",
                color=color,
                markersize=5,
                markeredgewidth=1.5,
                alpha=0.7,
                zorder=8,
            )
            # Residual line from expected to observed (de-emphasized gray dashed)
            (self.residual_line,) = ax.plot(
                [], [], "--", color=COLORS["gray"], linewidth=1.0, alpha=0.4, zorder=8
            )
        else:
            self.expected_marker = None
            self.residual_line = None

        # Track last known position for dropout marker. Initialize to
        # None so a session that opens with a dropout doesn't render a
        # phantom red X at the origin — the dropout marker should only
        # appear after at least one *visible* sample establishes a real
        # last-known position.
        self.last_x: float | None = None
        self.last_y: float | None = None

    def update(
        self,
        x: float,
        y: float,
        visible: bool,
        confidence: float = 1.0,
        x_expected: float | None = None,
        y_expected: float | None = None,
    ) -> list[Any]:
        """Update LED position, visibility, and confidence.

        Parameters
        ----------
        x : float
            Observed X (m).
        y : float
            Observed Y (m).
        visible : bool
            Whether LED is detected in this frame.
        confidence : float, default 1.0
            Detection confidence [0, 1] used for halo alpha.
        x_expected : float | None, optional
            Expected X from body model (for residuals).
        y_expected : float | None, optional
            Expected Y from body model (for residuals).

        Returns
        -------
        list[Any]
            Modified artists for blitting.
        """
        artists = [self.marker, self.halo, self.dropout_marker]

        if visible:
            # Show LED marker and confidence halo
            self.marker.set_data([x], [y])
            self.halo.center = (x, y)
            self.halo.set_alpha(confidence * 0.5)  # Scale alpha by confidence
            self.dropout_marker.set_data([], [])  # Hide dropout marker

            # Show residuals if enabled and expected position provided
            if (
                self.show_residuals
                and self.expected_marker is not None
                and self.residual_line is not None
                and x_expected is not None
                and y_expected is not None
            ):
                self.expected_marker.set_data([x_expected], [y_expected])
                self.residual_line.set_data([x_expected, x], [y_expected, y])
                artists.extend([self.expected_marker, self.residual_line])
            elif (
                self.show_residuals
                and self.expected_marker is not None
                and self.residual_line is not None
            ):
                # Hide residual if no expected position
                self.expected_marker.set_data([], [])
                self.residual_line.set_data([], [])
                artists.extend([self.expected_marker, self.residual_line])

            # Update last known position
            self.last_x, self.last_y = x, y
        else:
            # Hide LED marker and halo
            self.marker.set_data([], [])
            self.halo.set_alpha(0.0)

            # Show red X at last known position — but only if we've seen
            # at least one visible sample. Before the first detection
            # there is no meaningful "last known position" and rendering
            # at (0, 0) draws a phantom dropout in the arena corner.
            if self.last_x is not None and self.last_y is not None:
                self.dropout_marker.set_data([self.last_x], [self.last_y])
            else:
                self.dropout_marker.set_data([], [])

            # Hide residuals when dropped out
            if (
                self.show_residuals
                and self.expected_marker is not None
                and self.residual_line is not None
            ):
                self.expected_marker.set_data([], [])
                self.residual_line.set_data([], [])
                artists.extend([self.expected_marker, self.residual_line])

        return artists


class TrailArtist:
    """Fading trajectory trail behind rat.

    Uses LineCollection for efficient rendering of multi-segment trail
    with alpha gradient (old = transparent, new = opaque).
    """

    def __init__(
        self, ax: Axes, trail_length_s: float, fps: int, color: str | None = None
    ):
        """Initialize trail artist.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to draw on.
        trail_length_s : float
            Length of trail in seconds.
        fps : int
            Video frame rate (frames per second).
        color : str | None, optional
            Trail color (default: COLORS["blue"]).
        """
        if color is None:
            color = COLORS["blue"]

        self.trail_frames = int(trail_length_s * fps)
        self.positions: deque[list[float]] = deque(maxlen=self.trail_frames)
        self.color = color

        # LineCollection for efficient multi-segment rendering
        self.lines = LineCollection([], linewidths=1.5, colors=color, zorder=4)
        ax.add_collection(self.lines)

    def update(self, x: float, y: float) -> list[Any]:
        """Add new position and update fading trail.

        Parameters
        ----------
        x : float
            Current X position (m).
        y : float
            Current Y position (m).

        Returns
        -------
        list[Any]
            Modified artists for blitting.
        """
        self.positions.append([x, y])

        if len(self.positions) > 1:
            # Create line segments from consecutive positions
            segments = [
                [self.positions[i], self.positions[i + 1]]
                for i in range(len(self.positions) - 1)
            ]

            # Fade alpha from 0 (oldest) to 0.6 (newest) using RGBA per segment
            # COLORS["blue"] = "#2166AC" → RGB (0.133, 0.4, 0.675)
            alphas = np.linspace(0.0, 0.6, len(segments))
            colors = [(0.133, 0.4, 0.675, a) for a in alphas]

            self.lines.set_segments(segments)
            self.lines.set_colors(colors)

        return [self.lines]


class HUDArtist:
    """Heads-up display showing current state information.

    Displays time, speed, heading, and LED detection status in overlay text.
    """

    def __init__(self, ax: Axes):
        """Initialize HUD artist.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to draw on.
        """
        # State info only (time shown in progress bar)
        self.state_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.6,
                pad=0.3,
                edgecolor="none",
            ),
            zorder=20,
        )

    def update(self, t: float, state: dict[str, Any]) -> list[Text]:
        """Update HUD text with current state.

        Parameters
        ----------
        t : float
            Current time (s).
        state : dict[str, Any]
            Dictionary with state information:
            - speed: Speed in m/s
            - theta: Heading in radians
            - led1_visible: LED1 detection status
            - led2_visible: LED2 detection status

        Returns
        -------
        list[Text]
            Modified text artists.
        """
        # Format state info
        speed_ms = state.get("speed", 0.0)
        theta_rad = state.get("theta", 0.0)
        led1_vis = state.get("led1_visible", False)
        led2_vis = state.get("led2_visible", False)

        # Compact display with time, velocity, heading, LED status
        state_str = (
            f"t = {t:.2f} s\n"
            f"v = {speed_ms:.2f} m/s\n"
            f"θ = {np.rad2deg(theta_rad):5.1f}°\n"
            f"LED1: {'✓' if led1_vis else '✗'}\n"
            f"LED2: {'✓' if led2_vis else '✗'}"
        )

        self.state_text.set_text(state_str)

        return [self.state_text]


class EventMarkerArtist:
    """Highlight special events (LED swaps, long dropouts) with temporary banners.

    Shows colored text banner at top of display for short duration when
    events are detected.
    """

    def __init__(self, ax: Axes):
        """Initialize event marker artist.

        Args:
            ax: Matplotlib axes to draw on
        """
        # Event banner (top center)
        self.banner = ax.text(
            0.5,
            0.98,
            "",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="red",
            bbox=dict(boxstyle="round", facecolor=COLORS["yellow"], alpha=0.0, pad=0.5),
            zorder=25,
        )

        self.timer = 0  # Frames remaining to show banner
        self.banner_duration_frames = 60  # Show for 2 seconds at 30fps

    def update(self, events: dict[str, bool]) -> list[Text]:
        """Update event markers based on detected events.

        Args:
            events: Dictionary of event flags:
                - led_swap: True if LED swap detected this frame
                - long_dropout: True if long dropout ongoing

        Returns:
            List of modified text artists
        """
        # Check for new events
        if events.get("led_swap", False):
            self.banner.set_text("⚠ LED SWAP DETECTED")
            self.banner.set_color("red")
            bbox = self.banner.get_bbox_patch()
            if bbox is not None:
                bbox.set_facecolor(COLORS["yellow"])
            self.timer = self.banner_duration_frames

        elif events.get("long_dropout", False) and self.timer == 0:
            self.banner.set_text("⚠ LONG DROPOUT")
            self.banner.set_color("white")
            bbox = self.banner.get_bbox_patch()
            if bbox is not None:
                bbox.set_facecolor(COLORS["red"])
            self.timer = self.banner_duration_frames

        # Update banner visibility (fade out)
        if self.timer > 0:
            alpha = 0.9 * (self.timer / self.banner_duration_frames)
            bbox = self.banner.get_bbox_patch()
            if bbox is not None:
                bbox.set_alpha(alpha)
            self.timer -= 1
        else:
            self.banner.set_text("")
            bbox = self.banner.get_bbox_patch()
            if bbox is not None:
                bbox.set_alpha(0.0)

        return [self.banner]


class IMUPanelArtist:
    """Show recent IMU measurements in scrolling time series.

    Displays gyroscope and accelerometer measurements in separate subplots
    with a sliding time window.
    """

    def __init__(
        self,
        axes: list[Axes],
        window_s: float,
        fps: int,
        config: Any = None,
        gyro_ylim: tuple[float, float] | None = None,
        accel_ylim: tuple[float, float] | None = None,
    ):
        """Initialize IMU panel artist.

        Args:
            axes: List of 3 matplotlib axes [gyro, accel_x, accel_y]
            window_s: Time window to display in seconds
            fps: Video frame rate
            config: Simulation config (for reference lines)
            gyro_ylim: Fixed y-axis limits for gyro plot (rad/s). Default: (-5, 5)
            accel_ylim: Fixed y-axis limits for accel plots (m/s²). Default: (-15, 15)
        """
        self.ax_gyro, self.ax_accel_x, self.ax_accel_y = axes
        self.window_frames = int(window_s * fps)
        self.config = config

        # Fixed y-axis limits (symmetric around zero for intuitive interpretation)
        # Defaults based on typical rat locomotion: gyro ±3 rad/s, accel ±10 m/s²
        self.gyro_ylim = gyro_ylim if gyro_ylim is not None else (-3.0, 3.0)
        self.accel_ylim = accel_ylim if accel_ylim is not None else (-10.0, 10.0)

        # Data buffers
        self.time_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.gyro_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.accel_x_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.accel_y_buffer: deque[float] = deque(maxlen=self.window_frames)

        # Initialize line artists (measured)
        (self.gyro_line,) = self.ax_gyro.plot(
            [], [], "-", color=COLORS["blue"], linewidth=1
        )
        (self.accel_x_line,) = self.ax_accel_x.plot(
            [], [], "-", color=COLORS["red"], linewidth=1
        )
        (self.accel_y_line,) = self.ax_accel_y.plot(
            [], [], "-", color=COLORS["green"], linewidth=1
        )

        # Initialize truth line artists (ground truth overlays)
        (self.gyro_truth_line,) = self.ax_gyro.plot(
            [], [], "--", color="black", linewidth=1, alpha=0.6, label="truth"
        )
        (self.accel_x_truth_line,) = self.ax_accel_x.plot(
            [], [], "--", color="black", linewidth=1, alpha=0.6, label="truth"
        )
        (self.accel_y_truth_line,) = self.ax_accel_y.plot(
            [], [], "--", color="black", linewidth=1, alpha=0.6, label="truth"
        )

        # Zero reference lines
        self.ax_gyro.axhline(0, color=COLORS["gray"], linestyle="-", linewidth=0.5)
        self.ax_accel_x.axhline(0, color=COLORS["gray"], linestyle="-", linewidth=0.5)
        self.ax_accel_y.axhline(0, color=COLORS["gray"], linestyle="-", linewidth=0.5)

        # Set fixed y-axis limits
        self.ax_gyro.set_ylim(self.gyro_ylim)
        self.ax_accel_x.set_ylim(self.accel_ylim)
        self.ax_accel_y.set_ylim(self.accel_ylim)

        # Labels (explicit body-frame for EKF debugging)
        self.ax_gyro.set_ylabel("gyro\n(body rad/s)", fontsize=8)
        self.ax_accel_x.set_ylabel("accel X\n(body m/s²)", fontsize=8)
        self.ax_accel_y.set_ylabel("accel Y\n(body m/s²)", fontsize=8)
        self.ax_accel_y.set_xlabel("time (s)", fontsize=8)

        # Remove top spine, keep bottom for x-axis
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Add minor ticks at 0.5s intervals for temporal reference
        from matplotlib.ticker import MultipleLocator

        for ax in axes:
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.tick_params(which="minor", length=2, color="gray")

        # Only show x-ticks on bottom plot
        self.ax_gyro.set_xticklabels([])
        self.ax_accel_x.set_xticklabels([])

    def add_reference_bands(
        self, U_imu: np.ndarray, percentiles: tuple[float, float] = (10, 90)
    ) -> None:
        """Add shaded reference bands showing typical IMU ranges.

        Computes percentile-based bands for gyro and accelerometer channels
        and overlays them as low-alpha shaded regions on the IMU panel plots.

        Parameters
        ----------
        U_imu : np.ndarray
            IMU measurements (N, 3) containing [gyro, accel_x, accel_y] in
            standard units (rad/s for gyro, m/s² for accel).
        percentiles : tuple[float, float], default (10, 90)
            Low and high percentiles for band boundaries. Default shows
            10th to 90th percentile range.

        Returns
        -------
        None
            Modifies axes in-place by adding axhspan patches.
        """
        # Compute percentiles for each IMU channel
        gyro_low, gyro_high = np.percentile(U_imu[:, 0], percentiles)
        accel_x_low, accel_x_high = np.percentile(U_imu[:, 1], percentiles)
        accel_y_low, accel_y_high = np.percentile(U_imu[:, 2], percentiles)

        # Add shaded bands (subtle, low alpha)
        self.ax_gyro.axhspan(
            gyro_low, gyro_high, color=COLORS["blue"], alpha=0.1, zorder=1, linewidth=0
        )
        self.ax_accel_x.axhspan(
            accel_x_low,
            accel_x_high,
            color=COLORS["red"],
            alpha=0.1,
            zorder=1,
            linewidth=0,
        )
        self.ax_accel_y.axhspan(
            accel_y_low,
            accel_y_high,
            color=COLORS["green"],
            alpha=0.1,
            zorder=1,
            linewidth=0,
        )

    def update(
        self,
        t: float,
        imu_data: dict[str, float] | None = None,
        t_raw: np.ndarray | None = None,
        imu_raw: dict[str, np.ndarray] | None = None,
        imu_truth: dict[str, np.ndarray] | None = None,
    ) -> list[Any]:
        """Update IMU panel with new data.

        Supports two modes:
        1. Single sample mode: Pass imu_data (one point per frame, interpolated)
        2. High-rate mode: Pass t_raw + imu_raw (shows actual IMU sampling)

        Args:
            t: Current video time in seconds
            imu_data: (Mode 1) Single IMU sample dict with gyro, accel_x, accel_y
            t_raw: (Mode 2) Array of raw IMU timestamps within window
            imu_raw: (Mode 2) Dict with arrays: gyro, accel_x, accel_y
            imu_truth: Ground truth IMU values with arrays: yaw_rate, accel_x, accel_y

        Returns:
            List of modified artists
        """
        if t_raw is not None and imu_raw is not None:
            # High-rate mode: show all raw samples in window
            self.gyro_line.set_data(t_raw, imu_raw["gyro"])
            self.accel_x_line.set_data(t_raw, imu_raw["accel_x"])
            self.accel_y_line.set_data(t_raw, imu_raw["accel_y"])

            # Update truth overlays if provided
            if imu_truth is not None:
                self.gyro_truth_line.set_data(t_raw, imu_truth["yaw_rate"])
                self.accel_x_truth_line.set_data(t_raw, imu_truth["accel_x"])
                self.accel_y_truth_line.set_data(t_raw, imu_truth["accel_y"])
            else:
                # Clear truth lines if no truth data
                self.gyro_truth_line.set_data([], [])
                self.accel_x_truth_line.set_data([], [])
                self.accel_y_truth_line.set_data([], [])

            # Set x-axis limits based on actual data range
            if len(t_raw) > 0:
                self.ax_gyro.set_xlim(t_raw[0], t_raw[-1])
                self.ax_accel_x.set_xlim(t_raw[0], t_raw[-1])
                self.ax_accel_y.set_xlim(t_raw[0], t_raw[-1])
        else:
            # Single sample mode (legacy): buffer interpolated points
            if imu_data is not None:
                self.time_buffer.append(t)
                self.gyro_buffer.append(imu_data["gyro"])
                self.accel_x_buffer.append(imu_data["accel_x"])
                self.accel_y_buffer.append(imu_data["accel_y"])

            self.gyro_line.set_data(list(self.time_buffer), list(self.gyro_buffer))
            self.accel_x_line.set_data(
                list(self.time_buffer), list(self.accel_x_buffer)
            )
            self.accel_y_line.set_data(
                list(self.time_buffer), list(self.accel_y_buffer)
            )

            # Clear truth lines in single-sample mode (not supported)
            self.gyro_truth_line.set_data([], [])
            self.accel_x_truth_line.set_data([], [])
            self.accel_y_truth_line.set_data([], [])

            if len(self.time_buffer) > 1:
                t_min = self.time_buffer[0]
                t_max = self.time_buffer[-1]
                self.ax_gyro.set_xlim(t_min, t_max)
                self.ax_accel_x.set_xlim(t_min, t_max)
                self.ax_accel_y.set_xlim(t_min, t_max)

        return [
            self.gyro_line,
            self.accel_x_line,
            self.accel_y_line,
            self.gyro_truth_line,
            self.accel_x_truth_line,
            self.accel_y_truth_line,
        ]


class CameraPanelArtist:
    """Show camera detection status and confidence bars.

    Displays horizontal bars showing confidence for each LED,
    with different colors for visible/dropout states.
    """

    def __init__(self, ax: Axes):
        """Initialize camera panel artist.

        Args:
            ax: Matplotlib axes to draw on
        """
        self.ax = ax

        # Set up axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["LED1", "LED2"], fontsize=8)
        ax.set_xlabel("confidence", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Create bar artists (will update widths)
        self.led1_bar = ax.barh(0.5, 0, height=0.4, color=COLORS["blue"], alpha=0.6)[0]
        self.led2_bar = ax.barh(1.5, 0, height=0.4, color=COLORS["orange"], alpha=0.6)[
            0
        ]

        # Reference line at full confidence
        ax.axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=0.5)

        # Text labels for confidence values (will update)
        self.led1_text = ax.text(
            1.02, 0.5, "", va="center", ha="left", fontsize=7, family="monospace"
        )
        self.led2_text = ax.text(
            1.02, 1.5, "", va="center", ha="left", fontsize=7, family="monospace"
        )

        # Latency readout (camera observation lag)
        self.latency_text = ax.text(
            0.5, 2.2, "", va="center", ha="center", fontsize=7, family="monospace"
        )

    def update(
        self,
        led1_visible: bool,
        conf1: float,
        led2_visible: bool,
        conf2: float,
        latency_ms: float | None = None,
    ) -> list[Any]:
        """Update confidence bars and latency display.

        Args:
            led1_visible: LED1 detection status
            conf1: LED1 confidence [0, 1]
            led2_visible: LED2 detection status
            conf2: LED2 confidence [0, 1]
            latency_ms: Camera observation latency in milliseconds

        Returns:
            List of modified artists
        """
        # Update bar widths, colors, and text labels
        if led1_visible:
            self.led1_bar.set_width(conf1)
            self.led1_bar.set_color(COLORS["blue"])
            self.led1_bar.set_alpha(0.6)
            self.led1_text.set_text(f"{conf1:.2f}")
            self.led1_text.set_color("black")
        else:
            self.led1_bar.set_width(0.05)  # Small bar for dropout
            self.led1_bar.set_color(COLORS["red"])
            self.led1_bar.set_alpha(0.3)
            self.led1_text.set_text("(dropout)")
            self.led1_text.set_color(COLORS["red"])

        if led2_visible:
            self.led2_bar.set_width(conf2)
            self.led2_bar.set_color(COLORS["orange"])
            self.led2_bar.set_alpha(0.6)
            self.led2_text.set_text(f"{conf2:.2f}")
            self.led2_text.set_color("black")
        else:
            self.led2_bar.set_width(0.05)
            self.led2_bar.set_color(COLORS["red"])
            self.led2_bar.set_alpha(0.3)
            self.led2_text.set_text("(dropout)")
            self.led2_text.set_color(COLORS["red"])

        # Update latency display
        if latency_ms is not None:
            self.latency_text.set_text(f"latency: {latency_ms:.1f} ms")
        else:
            self.latency_text.set_text("")

        return [
            self.led1_bar,
            self.led2_bar,
            self.led1_text,
            self.led2_text,
            self.latency_text,
        ]


class ProgressBarArtist:
    """Show playback progress with event markers.

    Displays a thin progress bar showing current position in video timeline,
    with markers for key events (LED swaps, long dropouts).
    """

    def __init__(
        self,
        ax: Axes,
        duration_s: float,
        event_times: dict[str, list[float]] | None = None,
    ):
        """Initialize progress bar artist.

        Args:
            ax: Matplotlib axes to draw on
            duration_s: Total duration of video in seconds
            event_times: Dictionary mapping event types to lists of timestamps
        """
        self.ax = ax
        self.duration_s = duration_s

        # Set up axes (horizontal bar spanning 0-duration)
        ax.set_xlim(0, duration_s)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Background bar (full duration, light gray)
        self.background = ax.barh(
            0.5, duration_s, height=0.6, left=0, color=COLORS["light_gray"], alpha=0.3
        )[0]

        # Progress bar (current position, blue)
        self.progress_bar = ax.barh(
            0.5, 0, height=0.6, left=0, color=COLORS["blue"], alpha=0.6
        )[0]

        # Current time marker (vertical line)
        (self.time_marker,) = ax.plot(
            [0, 0], [0, 1], color="black", linewidth=2, zorder=10
        )

        # Event markers (vertical lines for swaps/dropouts)
        legend_handles = []
        if event_times:
            for event_type, times in event_times.items():
                if event_type == "led_swap":
                    color = COLORS["orange"]
                    marker = "v"
                    label = "LED swap"
                elif event_type == "long_dropout":
                    color = COLORS["red"]
                    marker = "x"
                    label = "Dropout"
                else:
                    continue

                # Plot markers and collect for legend
                for t in times:
                    ax.axvline(t, color=color, alpha=0.4, linewidth=1, linestyle="--")
                    ax.plot(t, 0.5, marker=marker, color=color, markersize=4, zorder=5)

                # Add legend handle (only once per event type)
                from matplotlib.lines import Line2D

                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        color="w",
                        markerfacecolor=color,
                        markersize=6,
                        label=label,
                    )
                )

        # Add legend if there are any events
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=6,
                frameon=True,
                framealpha=0.8,
            )

    def update(self, t: float) -> list[Any]:
        """Update progress bar to current time.

        Args:
            t: Current time in seconds

        Returns:
            List of modified artists
        """
        # Update progress bar width
        self.progress_bar.set_width(t)

        # Update time marker position
        self.time_marker.set_data([t, t], [0, 1])

        return [self.progress_bar, self.time_marker]


class FilterArtist:
    """Overlay filter predictions on arena view.

    Shows predicted position with uncertainty ellipse (95% confidence).
    """

    def __init__(self, ax: Axes):
        """Initialize filter artist.

        Args:
            ax: Matplotlib axes to draw on
        """
        self.ax = ax

        # Predicted position marker (hollow circle to distinguish from truth)
        (self.pred_marker,) = ax.plot(
            [],
            [],
            "o",
            color=COLORS["green"],
            markersize=8,
            fillstyle="none",
            linewidth=2,
            label="Filter Estimate",
            zorder=8,
        )

        # Uncertainty ellipse (95% confidence = 2.45-sigma for 2D)
        from matplotlib.patches import Ellipse

        self.uncertainty_ellipse = Ellipse(
            (0, 0),
            0,
            0,
            angle=0,
            facecolor=COLORS["green"],
            edgecolor=COLORS["green"],
            alpha=0.15,
            linewidth=1.5,
            linestyle="--",
            zorder=4,
        )
        ax.add_patch(self.uncertainty_ellipse)

    def update(self, x_pred: float, y_pred: float, P: np.ndarray | Array) -> list[Any]:
        """Update filter prediction and uncertainty.

        Parameters
        ----------
        x_pred : float
            Predicted x position in meters.
        y_pred : float
            Predicted y position in meters.
        P : np.ndarray or jax.Array
            2x2 position covariance matrix. The caller is responsible
            for selecting the position rows/cols from the full state
            covariance via the filter layout.

        Returns
        -------
        list[Any]
            List of modified artists.

        Raises
        ------
        ValueError
            If ``P`` is not shape ``(2, 2)``.
        """
        # Validate input shape
        P_np = np.asarray(P)
        if P_np.shape != (2, 2):
            raise ValueError(f"Expected P shape (2, 2), got {P_np.shape}")

        # Skip the overlay when the filter has diverged (non-finite mean
        # or covariance). Writing NaN/Inf into the marker position or
        # ellipse dimensions silently produces an invalid overlay rather
        # than a hard error — eigh can also return NaNs and propagate
        # non-finite width/height into the ellipse patch.
        if not (
            np.isfinite(x_pred) and np.isfinite(y_pred) and np.all(np.isfinite(P_np))
        ):
            self.pred_marker.set_data([], [])
            self.uncertainty_ellipse.width = 0.0
            self.uncertainty_ellipse.height = 0.0
            return [self.pred_marker, self.uncertainty_ellipse]

        # Update marker position
        self.pred_marker.set_data([x_pred], [y_pred])

        # Compute covariance ellipse (95% confidence for 2D: χ²(2, 0.05) = 5.991)
        P_pos = P_np
        eigenvalues, eigenvectors = np.linalg.eigh(P_pos)

        # Ensure eigenvalues are positive (numerical stability)
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        # Orientation angle from first eigenvector
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Width and height from eigenvalues (95% confidence)
        chi2_95 = 5.991
        width = 2 * np.sqrt(chi2_95 * eigenvalues[0])
        height = 2 * np.sqrt(chi2_95 * eigenvalues[1])

        # Update ellipse
        self.uncertainty_ellipse.center = (x_pred, y_pred)
        self.uncertainty_ellipse.width = width
        self.uncertainty_ellipse.height = height
        self.uncertainty_ellipse.angle = angle

        return [self.pred_marker, self.uncertainty_ellipse]


class ResidualPanelArtist:
    """Show measurement innovations (residuals) over time.

    Displays innovation for LED1 and LED2 positions in a scrolling time series.
    Innovation = observed - predicted (positive values indicate underestimation).
    """

    def __init__(self, ax: Axes, window_s: float, fps: int):
        """Initialize residual panel artist.

        Args:
            ax: Matplotlib axes to draw on
            window_s: Time window for scrolling display (seconds)
            fps: Video frame rate (for buffer sizing)
        """
        self.ax = ax
        self.window_s = window_s
        self.window_frames = int(window_s * fps)

        # Buffers for time series
        self.time_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.resid_led1_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.resid_led2_buffer: deque[float] = deque(maxlen=self.window_frames)

        # Initialize lines
        (self.line_led1,) = ax.plot(
            [],
            [],
            color=COLORS["blue"],
            linewidth=1.5,
            label="LED1 Residual",
            alpha=0.8,
        )
        (self.line_led2,) = ax.plot(
            [],
            [],
            color=COLORS["orange"],
            linewidth=1.5,
            label="LED2 Residual",
            alpha=0.8,
        )

        # Zero reference line
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

        # Styling
        ax.set_ylabel("Position Residual (cm)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_title(
            "Innovation Residuals", fontweight="normal", loc="left", fontsize=10
        )
        ax.grid(True, alpha=0.15)
        ax.legend(loc="upper right", fontsize=7)

        # Initial y-limits (will auto-scale)
        ax.set_ylim(-5, 5)

    def update(self, t: float, resid_led1: float, resid_led2: float) -> list[Any]:
        """Update residual time series.

        Args:
            t: Current time in seconds
            resid_led1: LED1 2D position residual L2 norm in cm (NaN if not visible)
            resid_led2: LED2 2D position residual L2 norm in cm (NaN if not visible)

        Returns:
            List of modified artists
        """
        # Add new samples to buffers
        self.time_buffer.append(t)
        self.resid_led1_buffer.append(resid_led1)
        self.resid_led2_buffer.append(resid_led2)

        # Update lines
        if len(self.time_buffer) > 0:
            time_array = np.array(self.time_buffer)
            self.line_led1.set_data(time_array, np.array(self.resid_led1_buffer))
            self.line_led2.set_data(time_array, np.array(self.resid_led2_buffer))

            # Auto-scale x-axis to show scrolling window
            self.ax.set_xlim(time_array[0], time_array[-1])

            # Auto-scale y-axis based on recent data (with some margin).
            # Use ``np.isfinite`` (not ``not np.isnan``): a diverged filter
            # can emit ±Inf residuals and matplotlib's set_ylim raises
            # "Axis limits cannot be NaN or Inf", aborting video render.
            all_resid = list(self.resid_led1_buffer) + list(self.resid_led2_buffer)
            valid_resid = [r for r in all_resid if np.isfinite(r)]
            if len(valid_resid) > 0:
                y_max = max(abs(min(valid_resid)), abs(max(valid_resid)))
                y_lim = max(y_max * 1.2, 1.0)  # At least ±1 cm
                self.ax.set_ylim(-y_lim, y_lim)

        return [self.line_led1, self.line_led2]


class StateErrorPanelArtist:
    """Show state estimation errors using small multiples (Tufte principle).

    Displays velocity and heading errors in a compact, information-dense layout
    with reference lines for PRD targets.
    """

    def __init__(self, ax_vel: Axes, ax_heading: Axes, window_s: float, fps: int):
        """Initialize state error panel artist.

        Args:
            ax_vel: Axes for velocity error (2D: vx, vy components)
            ax_heading: Axes for heading error
            window_s: Time window for scrolling display (seconds)
            fps: Video frame rate (for buffer sizing)
        """
        self.window_s = window_s
        self.window_frames = int(window_s * fps)

        # Velocity error panel (show both components for directional insight)
        self.ax_vel = ax_vel
        self.time_buffer_vel: deque[float] = deque(maxlen=self.window_frames)
        self.error_vx_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.error_vy_buffer: deque[float] = deque(maxlen=self.window_frames)

        (self.line_vx,) = ax_vel.plot(
            [], [], color=COLORS["red"], linewidth=1.5, label="vx error", alpha=0.8
        )
        (self.line_vy,) = ax_vel.plot(
            [], [], color=COLORS["green"], linewidth=1.5, label="vy error", alpha=0.8
        )

        # PRD target line: ±10 cm/s
        ax_vel.axhline(
            10,
            color="gray",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
            label="PRD: ±10 cm/s",
        )
        ax_vel.axhline(-10, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax_vel.axhline(0, color="black", linewidth=0.5, alpha=0.3)

        ax_vel.set_ylabel("Velocity Error (cm/s)", fontsize=8)
        ax_vel.set_xlabel("Time (s)", fontsize=8)
        ax_vel.set_title(
            "Velocity Estimation Error", fontweight="normal", loc="left", fontsize=9
        )
        ax_vel.grid(True, alpha=0.1, linewidth=0.5)
        ax_vel.legend(loc="upper right", fontsize=6, framealpha=0.9)
        ax_vel.set_ylim(-15, 15)

        # Heading error panel
        self.ax_heading = ax_heading
        self.time_buffer_heading: deque[float] = deque(maxlen=self.window_frames)
        self.error_heading_buffer: deque[float] = deque(maxlen=self.window_frames)

        (self.line_heading,) = ax_heading.plot(
            [],
            [],
            color=COLORS["purple"],
            linewidth=1.5,
            label="Heading error",
            alpha=0.8,
        )

        # PRD target line: ±7°
        ax_heading.axhline(
            7, color="gray", linewidth=1, linestyle="--", alpha=0.5, label="PRD: ±7°"
        )
        ax_heading.axhline(-7, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax_heading.axhline(0, color="black", linewidth=0.5, alpha=0.3)

        ax_heading.set_ylabel("Heading Error (deg)", fontsize=8)
        ax_heading.set_xlabel("Time (s)", fontsize=8)
        ax_heading.set_title(
            "Heading Estimation Error", fontweight="normal", loc="left", fontsize=9
        )
        ax_heading.grid(True, alpha=0.1, linewidth=0.5)
        ax_heading.legend(loc="upper right", fontsize=6, framealpha=0.9)
        ax_heading.set_ylim(-30, 30)  # Larger range to accommodate realistic errors

    def update(
        self,
        t: float,
        error_vx: float,
        error_vy: float,
        error_heading_deg: float,
    ) -> list[Any]:
        """Update state error time series.

        Args:
            t: Current time in seconds
            error_vx: X velocity error in cm/s (NaN if unavailable)
            error_vy: Y velocity error in cm/s (NaN if unavailable)
            error_heading_deg: Heading error in degrees (NaN if unavailable)

        Returns:
            List of modified artists
        """
        # Update velocity errors
        self.time_buffer_vel.append(t)
        self.error_vx_buffer.append(error_vx)
        self.error_vy_buffer.append(error_vy)

        if len(self.time_buffer_vel) > 0:
            time_array = np.array(self.time_buffer_vel)
            self.line_vx.set_data(time_array, np.array(self.error_vx_buffer))
            self.line_vy.set_data(time_array, np.array(self.error_vy_buffer))
            self.ax_vel.set_xlim(time_array[0], time_array[-1])

        # Update heading error
        self.time_buffer_heading.append(t)
        self.error_heading_buffer.append(error_heading_deg)

        if len(self.time_buffer_heading) > 0:
            time_array = np.array(self.time_buffer_heading)
            self.line_heading.set_data(time_array, np.array(self.error_heading_buffer))
            self.ax_heading.set_xlim(time_array[0], time_array[-1])

        return [self.line_vx, self.line_vy, self.line_heading]


class BiasEstimatePanelArtist:
    """Show learned IMU bias estimates over time.

    Displays gyro and accelerometer bias evolution to verify filter learning.
    Following Gelman's advice: show uncertainty bands, not just point estimates.
    """

    def __init__(self, ax: Axes, window_s: float, fps: int):
        """Initialize bias estimate panel artist.

        Args:
            ax: Matplotlib axes to draw on
            window_s: Time window for scrolling display (seconds)
            fps: Video frame rate (for buffer sizing)
        """
        self.ax = ax
        self.window_s = window_s
        self.window_frames = int(window_s * fps)

        # Buffers
        self.time_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.gyro_bias_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.accel_bias_x_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.accel_bias_y_buffer: deque[float] = deque(maxlen=self.window_frames)

        # Lines (use distinct colors for each bias)
        (self.line_gyro,) = ax.plot(
            [],
            [],
            color=COLORS["blue"],
            linewidth=1.5,
            label="Gyro bias (rad/s)",
            alpha=0.8,
        )
        (self.line_ax,) = ax.plot(
            [],
            [],
            color=COLORS["red"],
            linewidth=1.5,
            label="Accel X bias (m/s²)",
            alpha=0.8,
        )
        (self.line_ay,) = ax.plot(
            [],
            [],
            color=COLORS["green"],
            linewidth=1.5,
            label="Accel Y bias (m/s²)",
            alpha=0.8,
        )

        # Zero reference
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

        # Styling (minimal, Tufte-inspired)
        ax.set_ylabel("Bias Estimate", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title("Learned IMU Biases", fontweight="normal", loc="left", fontsize=9)
        ax.grid(True, alpha=0.1, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
        ax.set_ylim(-0.1, 0.1)  # Will auto-scale

    def update(
        self,
        t: float,
        gyro_bias: float,
        accel_bias_x: float,
        accel_bias_y: float,
    ) -> list[Any]:
        """Update bias estimate time series.

        Args:
            t: Current time in seconds
            gyro_bias: Gyro z-axis bias estimate in rad/s
            accel_bias_x: Accel x-axis bias estimate in m/s²
            accel_bias_y: Accel y-axis bias estimate in m/s²

        Returns:
            List of modified artists
        """
        # Add new samples
        self.time_buffer.append(t)
        self.gyro_bias_buffer.append(gyro_bias)
        self.accel_bias_x_buffer.append(accel_bias_x)
        self.accel_bias_y_buffer.append(accel_bias_y)

        # Update lines
        if len(self.time_buffer) > 0:
            time_array = np.array(self.time_buffer)
            self.line_gyro.set_data(time_array, np.array(self.gyro_bias_buffer))
            self.line_ax.set_data(time_array, np.array(self.accel_bias_x_buffer))
            self.line_ay.set_data(time_array, np.array(self.accel_bias_y_buffer))

            # Auto-scale x-axis
            self.ax.set_xlim(time_array[0], time_array[-1])

            # Auto-scale y-axis based on data range. Filter out non-finite
            # samples — a diverged filter can emit ±Inf bias estimates and
            # matplotlib's set_ylim raises on Inf, aborting video render.
            all_biases = [
                b
                for b in (
                    list(self.gyro_bias_buffer)
                    + list(self.accel_bias_x_buffer)
                    + list(self.accel_bias_y_buffer)
                )
                if np.isfinite(b)
            ]
            if len(all_biases) > 0:
                y_max = max(abs(min(all_biases)), abs(max(all_biases)))
                y_lim = max(y_max * 1.2, 0.01)  # At least ±0.01
                self.ax.set_ylim(-y_lim, y_lim)

        return [self.line_gyro, self.line_ax, self.line_ay]


class NEESPanelArtist:
    """Show NEES (filter consistency metric) with chi-squared bounds.

    NEES should stay within chi-squared confidence bounds for a consistent filter.
    Following Heer's principle: show data quality metrics prominently.
    """

    def __init__(self, ax: Axes, window_s: float, fps: int, state_dim: int = 2):
        """Initialize NEES panel artist.

        Args:
            ax: Matplotlib axes to draw on
            window_s: Time window for scrolling display (seconds)
            fps: Video frame rate (for buffer sizing)
            state_dim: Dimension of state for NEES (typically 2 for position)
        """
        self.ax = ax
        self.window_s = window_s
        self.window_frames = int(window_s * fps)
        self.state_dim = state_dim

        # Buffers
        self.time_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.nees_buffer: deque[float] = deque(maxlen=self.window_frames)

        # NEES line
        (self.line_nees,) = ax.plot(
            [], [], color=COLORS["purple"], linewidth=2, label="NEES", alpha=0.9
        )

        # Chi-squared 95% confidence bounds (from scipy)
        from scipy.stats import chi2  # type: ignore[import]

        self.chi2_lower = chi2.ppf(0.025, df=state_dim)
        self.chi2_upper = chi2.ppf(0.975, df=state_dim)
        self.chi2_mean = state_dim

        # Reference lines (Tufte: use subtle colors for reference)
        ax.axhline(
            self.chi2_mean,
            color="gray",
            linewidth=1.5,
            linestyle="-",
            alpha=0.5,
            label=f"Expected: {self.chi2_mean:.1f}",
        )
        ax.axhline(
            self.chi2_lower,
            color="red",
            linewidth=1,
            linestyle="--",
            alpha=0.4,
            label=f"95% CI: [{self.chi2_lower:.1f}, {self.chi2_upper:.1f}]",
        )
        ax.axhline(self.chi2_upper, color="red", linewidth=1, linestyle="--", alpha=0.4)

        # Fill between bounds (visual emphasis on acceptable range)
        # Use axhspan instead of fill_between for horizontal band
        ax.axhspan(
            self.chi2_lower,
            self.chi2_upper,
            color="green",
            alpha=0.05,
            label="Acceptable range",
            zorder=0,
        )

        # Styling
        ax.set_ylabel(f"NEES ({state_dim}-D)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title(
            "Filter Consistency (NEES)", fontweight="normal", loc="left", fontsize=9
        )
        ax.grid(True, alpha=0.1, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
        ax.set_ylim(0, max(self.chi2_upper * 1.5, 10))

    def update(self, t: float, nees: float) -> list[Any]:
        """Update NEES time series.

        Args:
            t: Current time in seconds
            nees: NEES value (NaN if unavailable)

        Returns:
            List of modified artists
        """
        # Add new sample
        self.time_buffer.append(t)
        self.nees_buffer.append(nees)

        # Update line
        if len(self.time_buffer) > 0:
            time_array = np.array(self.time_buffer)
            nees_array = np.array(self.nees_buffer)

            # Filter to *finite* samples (not just non-NaN). A diverged
            # filter or near-singular covariance can emit ±Inf NEES, which
            # downstream matplotlib set_ylim rejects with "Axis limits
            # cannot be NaN or Inf" and aborts video render. The NEES line
            # is stroked through only the finite samples (the other
            # diagnostic panels pass NaN straight to set_data, which
            # matplotlib renders as a gap; NEES filters first to keep its
            # line continuous). When the rolling window contains no
            # finite samples we must explicitly clear the line —
            # otherwise the previously-rendered NEES value stays visible
            # during dropout / divergence windows and gives a misleading
            # "consistency reading" for periods when NEES is unavailable.
            valid_mask = np.isfinite(nees_array)
            if np.any(valid_mask):
                self.line_nees.set_data(time_array[valid_mask], nees_array[valid_mask])
            else:
                self.line_nees.set_data([], [])

            # Auto-scale x-axis
            self.ax.set_xlim(time_array[0], time_array[-1])

            # Auto-scale y-axis based on data (but keep bounds visible)
            valid_nees = nees_array[valid_mask]
            if len(valid_nees) > 0:
                y_max = max(np.max(valid_nees), self.chi2_upper)
                self.ax.set_ylim(0, y_max * 1.2)

        return [self.line_nees]
