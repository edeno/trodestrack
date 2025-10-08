"""Main video generation API for diagnostic playback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from trodestrack.sim.utils import SimOut
from trodestrack.viz.components import (
    CameraPanelArtist,
    HUDArtist,
    IMUPanelArtist,
    LEDArtist,
    ProgressBarArtist,
    RatArtist,
    TrailArtist,
)
from trodestrack.viz.styles import COLORS, apply_tufte_style
from trodestrack.viz.utils import prepare_video_data


def create_diagnostic_video(
    sim_data: SimOut,
    output_path: str | Path,
    fps: int = 30,
    speedup: float = 1.0,
    time_window_s: float = 2.0,
    trail_length_s: float = 1.5,
    dpi: int = 100,
    codec: str = "h264",
    bitrate: int = 2000,
    return_animation: bool = False,
) -> Path | tuple[Path, Any, Any]:
    """Generate diagnostic video from simulation data.

    Creates multi-panel video showing:
    - Arena view (rat, LEDs, trail, HUD)
    - IMU time series (gyro, accel X, accel Y)
    - Camera status (confidence bars)

    Args:
        sim_data: Simulation output dictionary from simulate_rat_imu()
        output_path: Output video file path (e.g., "debug.mp4")
        fps: Video frame rate (frames per second)
        speedup: Playback speed multiplier (>1 faster, <1 slower)
        time_window_s: Time window for scrolling plots (seconds)
        trail_length_s: Length of fading trail (seconds)
        dpi: Figure resolution (dots per inch)
        codec: Video codec (default: "h264")
        bitrate: Video bitrate in kbps
        return_animation: If True, return (path, anim, fig) for testing (default: False)

    Returns:
        Path to created video file, or (path, anim, fig) if return_animation=True

    Example:
        >>> from trodestrack.sim.rat_imu import simulate_rat_imu, RatIMUSimConfig
        >>> config = RatIMUSimConfig(duration_s=10.0, led_swap_prob=0.15)
        >>> sim = simulate_rat_imu(config, seed=42)
        >>> create_diagnostic_video(sim, "debug.mp4", fps=30)
        PosixPath('debug.mp4')
    """
    output_path = Path(output_path)

    # Apply Tufte style
    apply_tufte_style()

    # Prepare interpolated data at video frame times
    print(f"Preparing video data (fps={fps}, speedup={speedup:.1f}x)...")
    video_data = prepare_video_data(sim_data, fps=fps, speedup=speedup)
    n_frames = video_data["n_frames"]
    print(f"  {n_frames} frames to render ({n_frames/fps:.1f}s video)")

    # Create figure with 16:9 aspect ratio (standard for video)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(
        5,
        2,
        figure=fig,
        hspace=0.35,
        wspace=0.3,
        top=0.90,  # Leave room for title and legend
        bottom=0.05,
        left=0.08,
        right=0.95,
        height_ratios=[1, 1, 1, 0.35, 0.25],  # Compact camera + progress bars
    )

    # Left column: Arena view (spans 3 rows)
    ax_arena = fig.add_subplot(gs[:3, 0])
    ax_arena.set_aspect("equal")
    ax_arena.set_xlabel("x (m)", fontsize=9)
    ax_arena.set_ylabel("y (m)", fontsize=9)
    ax_arena.set_title("Arena View", fontweight="normal", loc="left", fontsize=10)

    # Enable grid for spatial reference (subtle, 0.1m or 0.2m spacing)
    ax_arena.grid(True, which="major", alpha=0.15, linewidth=0.5, linestyle="-")
    ax_arena.set_axisbelow(True)  # Grid behind data

    # Right column: IMU plots (3 rows)
    ax_gyro = fig.add_subplot(gs[0, 1])
    ax_accel_x = fig.add_subplot(gs[1, 1], sharex=ax_gyro)
    ax_accel_y = fig.add_subplot(gs[2, 1], sharex=ax_gyro)

    ax_gyro.set_title("IMU Measurements", fontweight="normal", loc="left", fontsize=10)

    # Row 4: Camera status
    ax_camera = fig.add_subplot(gs[3, :])
    ax_camera.set_title("Camera Status", fontweight="normal", loc="left", fontsize=10)

    # Row 5: Progress bar
    ax_progress = fig.add_subplot(gs[4, :])

    # Extract config for convenience
    config = sim_data["config"]

    # Set arena bounds based on data
    X_truth = video_data["X_truth"]
    x_min, x_max = X_truth[:, 0].min(), X_truth[:, 0].max()
    y_min, y_max = X_truth[:, 1].min(), X_truth[:, 1].max()

    # Ensure minimum span to avoid over-zoom on tight/stationary paths
    min_span = 0.3  # meters
    x_range = max(x_max - x_min, min_span)
    y_range = max(y_max - y_min, min_span)

    # Center the span if range was expanded
    if x_max - x_min < min_span:
        x_center = (x_min + x_max) / 2
        x_min = x_center - min_span / 2
        x_max = x_center + min_span / 2
    if y_max - y_min < min_span:
        y_center = (y_min + y_max) / 2
        y_min = y_center - min_span / 2
        y_max = y_center + min_span / 2

    # Add 5% padding (reduced from 10% for better data-ink ratio)
    padding = 0.05
    ax_arena.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax_arena.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

    # Add scale bar in lower-left corner (10 cm reference)

    scale_length = 0.1  # 10 cm
    scale_x = x_min + 0.05 * x_range  # 5% from left edge
    scale_y = y_min + 0.05 * y_range  # 5% from bottom edge

    # Scale bar line
    ax_arena.plot(
        [scale_x, scale_x + scale_length],
        [scale_y, scale_y],
        color="black",
        linewidth=2,
        solid_capstyle="butt",
        zorder=15,
    )
    # End caps
    ax_arena.plot(
        [scale_x, scale_x],
        [scale_y - 0.01, scale_y + 0.01],
        color="black",
        linewidth=1.5,
        zorder=15,
    )
    ax_arena.plot(
        [scale_x + scale_length, scale_x + scale_length],
        [scale_y - 0.01, scale_y + 0.01],
        color="black",
        linewidth=1.5,
        zorder=15,
    )
    # Label
    ax_arena.text(
        scale_x + scale_length / 2,
        scale_y - 0.02,
        "10 cm",
        ha="center",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        zorder=15,
    )

    # Initialize artists
    print("Initializing artists...")
    rat = RatArtist(ax_arena)
    led1 = LEDArtist(ax_arena, led_id=1, color=COLORS["blue"], show_residuals=True)
    led2 = LEDArtist(ax_arena, led_id=2, color=COLORS["orange"], show_residuals=True)
    trail = TrailArtist(ax_arena, trail_length_s=trail_length_s, fps=fps)
    hud = HUDArtist(ax_arena)
    imu_panel = IMUPanelArtist(
        [ax_gyro, ax_accel_x, ax_accel_y],
        window_s=time_window_s,
        fps=fps,
        config=sim_data["config"],
    )

    # Add IMU reference bands (10th-90th percentile) for context
    imu_panel.add_reference_bands(sim_data["U_imu"], percentiles=(10, 90))

    camera_panel = CameraPanelArtist(ax_camera)

    # Pre-compute event times for progress bar markers
    print("Detecting events...")
    event_times = {"led_swap": [], "long_dropout": []}

    for frame_idx in range(n_frames):
        t = video_data["t_video"][frame_idx]
        cam_idx = video_data["cam_idx"][frame_idx]

        led1_pos = sim_data["Z_cam_led1"][cam_idx]
        led2_pos = sim_data["Z_cam_led2"][cam_idx]
        led1_visible = sim_data["mask_led1"][cam_idx]
        led2_visible = sim_data["mask_led2"][cam_idx]

        # LED swap detection (spacing deviation + vector direction)
        if led1_visible and led2_visible:
            # Check 1: Spacing deviation (catches occlusion-induced swaps)
            spacing = np.linalg.norm(led1_pos - led2_pos)
            expected_spacing = np.linalg.norm(config.led1_offset_body - config.led2_offset_body)
            spacing_anomaly = abs(spacing - expected_spacing) > 0.5 * expected_spacing

            # Check 2: LED vector direction (catches reflection/labeling swaps)
            # Get body state for this frame
            state = video_data["X_truth"][frame_idx]
            theta = state[4]  # heading angle
            body_x_axis = np.array([np.cos(theta), np.sin(theta)])
            led_vector = led1_pos - led2_pos  # Vector from LED2 to LED1

            # If LED1 is front and LED2 is back, dot product should be positive
            # (led_vector should point in same direction as body X-axis)
            dot_product = np.dot(led_vector, body_x_axis)
            direction_anomaly = dot_product < 0  # Vector points backward = swap

            if spacing_anomaly or direction_anomaly:
                event_times["led_swap"].append(t)

        # Long dropout detection
        if not (led1_visible or led2_visible):
            # Only mark start of dropout sequence (avoid many markers)
            if frame_idx == 0 or (
                sim_data["mask_led1"][video_data["cam_idx"][frame_idx - 1]]
                or sim_data["mask_led2"][video_data["cam_idx"][frame_idx - 1]]
            ):
                event_times["long_dropout"].append(t)

    # Debounce event markers (avoid clusters from jitter)
    min_spacing = 0.3  # seconds
    for event_type in event_times:
        times = np.array(event_times[event_type])
        if len(times) > 0:
            # Keep first event, then only events ≥ min_spacing from previous kept
            debounced = [times[0]]
            for t in times[1:]:
                if t - debounced[-1] >= min_spacing:
                    debounced.append(t)
            event_times[event_type] = debounced

    print(
        f"  Found {len(event_times['led_swap'])} LED swaps, "
        f"{len(event_times['long_dropout'])} dropout sequences (debounced)"
    )

    # Progress bar (shows timeline with current position and event markers)
    progress_bar = ProgressBarArtist(
        ax_progress, duration_s=config.duration_s, event_times=event_times
    )

    # Add legend entries for heading and velocity
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["blue"],
            markersize=8,
            label="LED1 (front)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["orange"],
            markersize=8,
            label="LED2 (back)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=3,
            marker=">",
            markersize=8,
            label="Heading (orientation)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["purple"],
            linewidth=2,
            marker=">",
            markersize=7,
            label="Velocity (0.25 m/s = 1 cm)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["blue"],
            linewidth=1.5,
            alpha=0.5,
            label="Path trail",
        ),
    ]
    # Overall title (includes frame rate for temporal context)
    title_str = (
        f"Diagnostic Video | {config.duration_s:.0f}s simulation | "
        f"{speedup:.1f}x speed | {fps} fps"
    )
    fig.suptitle(title_str, fontsize=11, fontweight="normal", y=0.98)

    # Place legend in top right corner (right column)
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.92),  # Top right corner of figure
        ncol=1,  # Vertical layout
        fontsize=6,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    # Pre-compute LED offsets (avoid hasattr checks per frame)
    config = sim_data["config"]
    led1_offset = getattr(config, "led1_offset_body", None)
    led2_offset = None
    if hasattr(config, "led2_offset_body") and getattr(config, "use_second_led", False):
        led2_offset = config.led2_offset_body

    # Animation update function
    def update_frame(frame_idx: int) -> None:
        """Update all artists for a single frame.

        Note: Component update() methods return artist lists for blitting, but we ignore
        those returns since blit=False. With blitting disabled, matplotlib redraws the
        entire figure each frame, so we don't need to track individual artist changes.
        """
        # Get data for this frame
        t = video_data["t_video"][frame_idx]
        state = video_data["X_truth"][frame_idx]  # [x, y, vx, vy, θ]
        imu = video_data["U_imu"][frame_idx]  # [gyro, accel_x, accel_y]
        cam_idx = video_data["cam_idx"][frame_idx]

        # Extract state components
        x, y, vx, vy, theta = state
        speed = np.hypot(vx, vy)

        # Get camera data for this frame
        led1_pos = sim_data["Z_cam_led1"][cam_idx]
        led2_pos = sim_data["Z_cam_led2"][cam_idx]
        led1_visible = sim_data["mask_led1"][cam_idx]
        led2_visible = sim_data["mask_led2"][cam_idx]
        conf1 = sim_data["confidence_led1"][cam_idx]
        conf2 = sim_data["confidence_led2"][cam_idx]

        # Compute expected LED positions from body model (for residuals)
        # LED positions = body position + rotated LED offset
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        if led1_offset is not None:
            led1_expected_x = x + cos_th * led1_offset[0] - sin_th * led1_offset[1]
            led1_expected_y = y + sin_th * led1_offset[0] + cos_th * led1_offset[1]
        else:
            led1_expected_x, led1_expected_y = None, None

        if led2_offset is not None:
            led2_expected_x = x + cos_th * led2_offset[0] - sin_th * led2_offset[1]
            led2_expected_y = y + sin_th * led2_offset[0] + cos_th * led2_offset[1]
        else:
            led2_expected_x, led2_expected_y = None, None

        # Update artists (events now shown in progress bar)
        rat.update(x, y, theta, vx, vy)
        led1.update(
            led1_pos[0] if not np.isnan(led1_pos[0]) else x,
            led1_pos[1] if not np.isnan(led1_pos[1]) else y,
            led1_visible,
            conf1,
            led1_expected_x,
            led1_expected_y,
        )
        led2.update(
            led2_pos[0] if not np.isnan(led2_pos[0]) else x,
            led2_pos[1] if not np.isnan(led2_pos[1]) else y,
            led2_visible,
            conf2,
            led2_expected_x,
            led2_expected_y,
        )
        trail.update(x, y)

        hud_state = {
            "speed": speed,
            "theta": theta,
            "vx": vx,
            "vy": vy,
            "led1_visible": led1_visible,
            "led2_visible": led2_visible,
            "conf1": conf1,
            "conf2": conf2,
        }
        hud.update(t, hud_state)

        # IMU panel: show raw high-rate data within window
        # Find all IMU samples in the time window [t - window_s, t]
        t_window_start = t - time_window_s
        imu_mask = (sim_data["t_imu"] >= t_window_start) & (sim_data["t_imu"] <= t)
        t_imu_window = sim_data["t_imu"][imu_mask]
        U_imu_window = sim_data["U_imu"][imu_mask]

        if len(t_imu_window) > 0:
            imu_raw = {
                "gyro": U_imu_window[:, 0],
                "accel_x": U_imu_window[:, 1],
                "accel_y": U_imu_window[:, 2],
            }

            # Extract ground truth for truth overlay (expected IMU values)
            yaw_rate_truth_window = sim_data["yaw_rate_truth"][imu_mask]
            accel_body_truth_window = sim_data["accel_body_truth"][imu_mask]
            imu_truth = {
                "yaw_rate": yaw_rate_truth_window,
                "accel_x": accel_body_truth_window[:, 0],
                "accel_y": accel_body_truth_window[:, 1],
            }

            imu_panel.update(t, t_raw=t_imu_window, imu_raw=imu_raw, imu_truth=imu_truth)
        else:
            # Fallback to interpolated single sample if no raw data in window
            imu_dict = {"gyro": imu[0], "accel_x": imu[1], "accel_y": imu[2]}
            imu_panel.update(t, imu_data=imu_dict)

        # Camera panel with latency readout
        # Compute latency: t_cam_obs - t_cam_exp (observation - exposure lag)
        t_cam_exp = sim_data["t_cam_exp"][cam_idx]
        t_cam_obs = sim_data["t_cam_obs"][cam_idx]
        latency_ms = (t_cam_obs - t_cam_exp) * 1000  # Convert to milliseconds

        camera_panel.update(led1_visible, conf1, led2_visible, conf2, latency_ms)

        # Progress bar
        progress_bar.update(t)

    # Create animation
    print("Rendering animation...")
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,  # Disabled: transforms on FancyArrowPatch don't work with blitting
        repeat=False,
    )

    # Save video with codec fallback chain
    print(f"Encoding video to {output_path}...")
    from matplotlib.animation import FFMpegWriter

    # Try multiple codecs in order of preference
    codec_fallbacks = ["libx264", "h264", "mpeg4"]
    if codec not in codec_fallbacks:
        codec_fallbacks.insert(0, codec)  # User-specified codec first

    video_saved = False
    for try_codec in codec_fallbacks:
        try:
            writer = FFMpegWriter(fps=fps, codec=try_codec, bitrate=bitrate)
            anim.save(str(output_path), writer=writer, dpi=dpi)
            print(f"✓ Video saved: {output_path} (codec: {try_codec})")
            video_saved = True
            break
        except Exception as e:
            print(f"  Codec '{try_codec}' failed: {e}")
            continue

    if not video_saved:
        print("✗ All video codecs failed, falling back to GIF...")
        # Fallback to GIF
        output_path = output_path.with_suffix(".gif")
        anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
        print(f"✓ GIF saved: {output_path}")

    # Return animation/figure for testing if requested, otherwise just path
    if return_animation:
        return output_path, anim, fig
    else:
        plt.close(fig)
        return output_path
