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
from matplotlib.patches import Circle, Wedge
from matplotlib.text import Text

from trodestrack.viz.styles import COLORS


class RatArtist:
    """Visualize rat body, orientation, and scale.

    The rat is represented as a circle (body) with a wedge (nose) indicating
    heading direction.
    """

    def __init__(self, ax: Axes, body_radius: float = 0.03):
        """Initialize rat artist.

        Args:
            ax: Matplotlib axes to draw on
            body_radius: Radius of rat body circle in meters (default: 3cm)
        """
        self.body_radius = body_radius

        # Rat body: gray circle
        self.body = Circle((0, 0), radius=body_radius, color=COLORS["gray"], alpha=0.7, zorder=5)
        ax.add_patch(self.body)

        # Nose: black wedge indicating heading (±30° cone)
        self.nose = Wedge(
            (0, 0),
            r=body_radius,
            theta1=-30,
            theta2=30,
            color="black",
            alpha=0.9,
            zorder=6,
        )
        ax.add_patch(self.nose)

    def update(self, x: float, y: float, theta: float) -> list[Any]:
        """Update rat position and orientation.

        Args:
            x: X position in meters
            y: Y position in meters
            theta: Heading angle in radians (0 = +X axis, counterclockwise)

        Returns:
            List of modified artists for blitting
        """
        self.body.center = (x, y)
        self.nose.center = (x, y)

        # Convert theta to degrees for wedge
        theta_deg = np.rad2deg(theta)
        self.nose.theta1 = theta_deg - 30
        self.nose.theta2 = theta_deg + 30

        return [self.body, self.nose]


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

        # LED marker: colored circle
        (self.marker,) = ax.plot(
            [],
            [],
            "o",
            color=color,
            markersize=8,
            markeredgewidth=1.5,
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

            # Fade alpha from 0 (oldest) to 0.6 (newest)
            alphas = np.linspace(0.0, 0.6, len(segments))

            self.lines.set_segments(segments)
            self.lines.set_alpha(alphas)

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
        # Time display (top-left, large)
        self.time_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            fontweight="bold",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3),
            zorder=20,
        )

        # State info (below time, smaller)
        self.state_text = ax.text(
            0.02,
            0.91,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, pad=0.3),
            zorder=20,
        )

    def update(self, t: float, state: dict[str, Any]) -> list[Text]:
        """Update HUD text with current state.

        Args:
            t: Current time in seconds
            state: Dictionary with state information:
                - speed: Speed in m/s
                - theta: Heading in radians
                - led1_visible: LED1 detection status
                - led2_visible: LED2 detection status
                - conf1: LED1 confidence [0, 1]
                - conf2: LED2 confidence [0, 1]

        Returns:
            List of modified text artists
        """
        time_str = f"t = {t:.2f}s"

        # Format state info
        speed_ms = state.get("speed", 0.0)
        theta_rad = state.get("theta", 0.0)
        led1_vis = state.get("led1_visible", False)
        led2_vis = state.get("led2_visible", False)
        conf1 = state.get("conf1", 0.0)
        conf2 = state.get("conf2", 0.0)

        state_str = (
            f"v = {speed_ms:.3f} m/s\n"
            f"θ = {np.rad2deg(theta_rad):6.1f}°\n"
            f"LED1: {'✓' if led1_vis else '✗'} ({conf1:.2f})\n"
            f"LED2: {'✓' if led2_vis else '✗'} ({conf2:.2f})"
        )

        self.time_text.set_text(time_str)
        self.state_text.set_text(state_str)

        return [self.time_text, self.state_text]


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
    ):
        """Initialize IMU panel artist.

        Args:
            axes: List of 3 matplotlib axes [gyro, accel_x, accel_y]
            window_s: Time window to display in seconds
            fps: Video frame rate
            config: Simulation config (for reference lines)
        """
        self.ax_gyro, self.ax_accel_x, self.ax_accel_y = axes
        self.window_frames = int(window_s * fps)
        self.config = config

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

        # Labels
        self.ax_gyro.set_ylabel("gyro\n(rad/s)", fontsize=8)
        self.ax_accel_x.set_ylabel("accel X\n(m/s²)", fontsize=8)
        self.ax_accel_y.set_ylabel("accel Y\n(m/s²)", fontsize=8)
        self.ax_accel_y.set_xlabel("time (s)", fontsize=8)

        # Remove top spine, keep bottom for x-axis
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Only show x-ticks on bottom plot
        self.ax_gyro.set_xticklabels([])
        self.ax_accel_x.set_xticklabels([])

    def update(self, t: float, imu_data: dict[str, float]) -> list[Any]:
        """Add new IMU sample and scroll window.

        Args:
            t: Current time in seconds
            imu_data: Dictionary with IMU measurements:
                - gyro: Gyroscope measurement (rad/s)
                - accel_x: Accelerometer X (m/s²)
                - accel_y: Accelerometer Y (m/s²)

        Returns:
            List of modified artists
        """
        # Append new data
        self.time_buffer.append(t)
        self.gyro_buffer.append(imu_data["gyro"])
        self.accel_x_buffer.append(imu_data["accel_x"])
        self.accel_y_buffer.append(imu_data["accel_y"])

        # Update line data
        self.gyro_line.set_data(list(self.time_buffer), list(self.gyro_buffer))
        self.accel_x_line.set_data(list(self.time_buffer), list(self.accel_x_buffer))
        self.accel_y_line.set_data(list(self.time_buffer), list(self.accel_y_buffer))

        # Auto-scale axes to show window
        if len(self.time_buffer) > 1:
            t_min = self.time_buffer[0]
            t_max = self.time_buffer[-1]
            self.ax_gyro.set_xlim(t_min, t_max)

            # Auto-scale y-axes with padding
            if len(self.gyro_buffer) > 0:
                gyro_arr = np.array(list(self.gyro_buffer))
                gyro_range = np.ptp(gyro_arr)
                gyro_pad = max(0.1, gyro_range * 0.1)
                self.ax_gyro.set_ylim(gyro_arr.min() - gyro_pad, gyro_arr.max() + gyro_pad)

            if len(self.accel_x_buffer) > 0:
                accel_x_arr = np.array(list(self.accel_x_buffer))
                accel_x_range = np.ptp(accel_x_arr)
                accel_x_pad = max(0.5, accel_x_range * 0.1)
                self.ax_accel_x.set_ylim(
                    accel_x_arr.min() - accel_x_pad, accel_x_arr.max() + accel_x_pad
                )

            if len(self.accel_y_buffer) > 0:
                accel_y_arr = np.array(list(self.accel_y_buffer))
                accel_y_range = np.ptp(accel_y_arr)
                accel_y_pad = max(0.5, accel_y_range * 0.1)
                self.ax_accel_y.set_ylim(
                    accel_y_arr.min() - accel_y_pad, accel_y_arr.max() + accel_y_pad
                )

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
        # Update bar widths and colors
        if led1_visible:
            self.led1_bar.set_width(conf1)
            self.led1_bar.set_color(COLORS["blue"])
            self.led1_bar.set_alpha(0.6)
        else:
            self.led1_bar.set_width(0.05)  # Small bar for dropout
            self.led1_bar.set_color(COLORS["red"])
            self.led1_bar.set_alpha(0.3)

        if led2_visible:
            self.led2_bar.set_width(conf2)
            self.led2_bar.set_color(COLORS["orange"])
            self.led2_bar.set_alpha(0.6)
        else:
            self.led2_bar.set_width(0.05)
            self.led2_bar.set_color(COLORS["red"])
            self.led2_bar.set_alpha(0.3)

        return [self.led1_bar, self.led2_bar]
