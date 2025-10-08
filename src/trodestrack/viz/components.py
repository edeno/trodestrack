"""Reusable visualization components (artists) for video generation.

Each artist class manages a visual element (rat, LED, trail, etc.) and provides
an update() method to modify appearance for each video frame.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.text import Text
from matplotlib.transforms import Affine2D

from trodestrack.viz.styles import COLORS


class RatArtist:
    """Visualize rat body, orientation, and velocity.

    The rat is represented as a circle (body) with a heading arrow and
    velocity vector for clear motion indication.
    """

    def __init__(self, ax: Axes, body_radius: float = 0.03):
        """Initialize rat artist.

        Args:
            ax: Matplotlib axes to draw on
            body_radius: Radius of rat body circle in meters (default: 3cm)
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

        Args:
            x: X position in meters
            y: Y position in meters
            theta: Heading angle in radians (0 = +X axis, counterclockwise)
            vx: X velocity in m/s (optional, for velocity arrow)
            vy: Y velocity in m/s (optional, for velocity arrow)

        Returns:
            List of modified artists for blitting
        """
        # Update body position
        self.body.center = (x, y)

        # Update heading arrow using transform (no recreation needed)
        T_heading = Affine2D().rotate_around(0, 0, theta).translate(x, y) + self.ax.transData
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
                Affine2D().scale(scale_factor, 1.0).rotate_around(0, 0, vel_angle).translate(x, y)
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

    def __init__(self, ax: Axes, led_id: int, color: str, marker_size: float = 0.008):
        """Initialize LED artist.

        Args:
            ax: Matplotlib axes to draw on
            led_id: LED identifier (1 or 2)
            color: Color for this LED (e.g., COLORS["blue"])
            marker_size: Radius of LED marker in meters
        """
        self.led_id = led_id
        self.color = color
        self.marker_size = marker_size

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
        self.halo = Circle((0, 0), radius=marker_size * 2, color=color, alpha=0.0, zorder=9)
        ax.add_patch(self.halo)

        # Dropout marker: red X at last known position
        (self.dropout_marker,) = ax.plot(
            [], [], "x", color=COLORS["red"], markersize=12, linewidth=2, zorder=11
        )

        # Track last known position for dropout marker
        self.last_x = 0.0
        self.last_y = 0.0

    def update(self, x: float, y: float, visible: bool, confidence: float = 1.0) -> list[Any]:
        """Update LED position, visibility, and confidence.

        Args:
            x: X position in meters
            y: Y position in meters
            visible: Whether LED is detected in this frame
            confidence: Detection confidence [0, 1] (used for halo alpha)

        Returns:
            List of modified artists for blitting
        """
        if visible:
            # Show LED marker and confidence halo
            self.marker.set_data([x], [y])
            self.halo.center = (x, y)
            self.halo.set_alpha(confidence * 0.5)  # Scale alpha by confidence
            self.dropout_marker.set_data([], [])  # Hide dropout marker

            # Update last known position
            self.last_x, self.last_y = x, y
        else:
            # Hide LED marker and halo
            self.marker.set_data([], [])
            self.halo.set_alpha(0.0)

            # Show red X at last known position
            self.dropout_marker.set_data([self.last_x], [self.last_y])

        return [self.marker, self.halo, self.dropout_marker]


class TrailArtist:
    """Fading trajectory trail behind rat.

    Uses LineCollection for efficient rendering of multi-segment trail
    with alpha gradient (old = transparent, new = opaque).
    """

    def __init__(self, ax: Axes, trail_length_s: float, fps: int, color: str = None):
        """Initialize trail artist.

        Args:
            ax: Matplotlib axes to draw on
            trail_length_s: Length of trail in seconds
            fps: Video frame rate (frames per second)
            color: Trail color (default: COLORS["blue"])
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

        Args:
            x: Current X position in meters
            y: Current Y position in meters

        Returns:
            List of modified artists for blitting
        """
        self.positions.append([x, y])

        if len(self.positions) > 1:
            # Create line segments from consecutive positions
            segments = [
                [self.positions[i], self.positions[i + 1]] for i in range(len(self.positions) - 1)
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

        Args:
            ax: Matplotlib axes to draw on
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
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, pad=0.3, edgecolor="none"),
            zorder=20,
        )

    def update(self, t: float, state: dict[str, Any]) -> list[Text]:
        """Update HUD text with current state.

        Args:
            t: Current time in seconds (unused - time shown in progress bar)
            state: Dictionary with state information:
                - speed: Speed in m/s
                - theta: Heading in radians
                - led1_visible: LED1 detection status
                - led2_visible: LED2 detection status

        Returns:
            List of modified text artists
        """
        # Format state info (time and confidence removed - shown elsewhere)
        speed_ms = state.get("speed", 0.0)
        theta_rad = state.get("theta", 0.0)
        led1_vis = state.get("led1_visible", False)
        led2_vis = state.get("led2_visible", False)

        # Compact display with reduced precision
        state_str = (
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
            self.banner.get_bbox_patch().set_facecolor(COLORS["yellow"])
            self.timer = self.banner_duration_frames

        elif events.get("long_dropout", False) and self.timer == 0:
            self.banner.set_text("⚠ LONG DROPOUT")
            self.banner.set_color("white")
            self.banner.get_bbox_patch().set_facecolor(COLORS["red"])
            self.timer = self.banner_duration_frames

        # Update banner visibility (fade out)
        if self.timer > 0:
            alpha = 0.9 * (self.timer / self.banner_duration_frames)
            self.banner.get_bbox_patch().set_alpha(alpha)
            self.timer -= 1
        else:
            self.banner.set_text("")
            self.banner.get_bbox_patch().set_alpha(0.0)

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
        self.gyro_ylim = gyro_ylim if gyro_ylim is not None else (-5.0, 5.0)
        self.accel_ylim = accel_ylim if accel_ylim is not None else (-15.0, 15.0)

        # Data buffers
        self.time_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.gyro_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.accel_x_buffer: deque[float] = deque(maxlen=self.window_frames)
        self.accel_y_buffer: deque[float] = deque(maxlen=self.window_frames)

        # Initialize line artists
        (self.gyro_line,) = self.ax_gyro.plot([], [], "-", color=COLORS["blue"], linewidth=1)
        (self.accel_x_line,) = self.ax_accel_x.plot([], [], "-", color=COLORS["red"], linewidth=1)
        (self.accel_y_line,) = self.ax_accel_y.plot([], [], "-", color=COLORS["green"], linewidth=1)

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

    def add_reference_bands(self, U_imu: np.ndarray, percentiles: tuple[float, float] = (10, 90)):
        """Add shaded reference bands showing typical IMU ranges.

        Args:
            U_imu: (N, 3) array of IMU measurements [gyro, accel_x, accel_y]
            percentiles: Tuple of (low, high) percentiles for bands (default: 10th-90th)
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

        Returns:
            List of modified artists
        """
        if t_raw is not None and imu_raw is not None:
            # High-rate mode: show all raw samples in window
            self.gyro_line.set_data(t_raw, imu_raw["gyro"])
            self.accel_x_line.set_data(t_raw, imu_raw["accel_x"])
            self.accel_y_line.set_data(t_raw, imu_raw["accel_y"])

            # Set x-axis limits based on actual data range
            if len(t_raw) > 0:
                self.ax_gyro.set_xlim(t_raw[0], t_raw[-1])
                self.ax_accel_x.set_xlim(t_raw[0], t_raw[-1])
                self.ax_accel_y.set_xlim(t_raw[0], t_raw[-1])
        else:
            # Single sample mode (legacy): buffer interpolated points
            self.time_buffer.append(t)
            self.gyro_buffer.append(imu_data["gyro"])
            self.accel_x_buffer.append(imu_data["accel_x"])
            self.accel_y_buffer.append(imu_data["accel_y"])

            self.gyro_line.set_data(list(self.time_buffer), list(self.gyro_buffer))
            self.accel_x_line.set_data(list(self.time_buffer), list(self.accel_x_buffer))
            self.accel_y_line.set_data(list(self.time_buffer), list(self.accel_y_buffer))

            if len(self.time_buffer) > 1:
                t_min = self.time_buffer[0]
                t_max = self.time_buffer[-1]
                self.ax_gyro.set_xlim(t_min, t_max)
                self.ax_accel_x.set_xlim(t_min, t_max)
                self.ax_accel_y.set_xlim(t_min, t_max)

        return [self.gyro_line, self.accel_x_line, self.accel_y_line]


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
        self.led2_bar = ax.barh(1.5, 0, height=0.4, color=COLORS["orange"], alpha=0.6)[0]

        # Reference line at full confidence
        ax.axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=0.5)

        # Text labels for confidence values (will update)
        self.led1_text = ax.text(
            1.02, 0.5, "", va="center", ha="left", fontsize=7, family="monospace"
        )
        self.led2_text = ax.text(
            1.02, 1.5, "", va="center", ha="left", fontsize=7, family="monospace"
        )

    def update(
        self, led1_visible: bool, conf1: float, led2_visible: bool, conf2: float
    ) -> list[Any]:
        """Update confidence bars.

        Args:
            led1_visible: LED1 detection status
            conf1: LED1 confidence [0, 1]
            led2_visible: LED2 detection status
            conf2: LED2 confidence [0, 1]

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

        return [self.led1_bar, self.led2_bar, self.led1_text, self.led2_text]


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
        self.progress_bar = ax.barh(0.5, 0, height=0.6, left=0, color=COLORS["blue"], alpha=0.6)[0]

        # Current time marker (vertical line)
        (self.time_marker,) = ax.plot([0, 0], [0, 1], color="black", linewidth=2, zorder=10)

        # Event markers (vertical lines for swaps/dropouts)
        if event_times:
            for event_type, times in event_times.items():
                if event_type == "led_swap":
                    color = COLORS["orange"]
                    marker = "v"
                elif event_type == "long_dropout":
                    color = COLORS["red"]
                    marker = "x"
                else:
                    continue

                for t in times:
                    ax.axvline(t, color=color, alpha=0.4, linewidth=1, linestyle="--")
                    ax.plot(t, 0.5, marker=marker, color=color, markersize=4, zorder=5)

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
