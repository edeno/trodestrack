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
    EventMarkerArtist,
    HUDArtist,
    IMUPanelArtist,
    LEDArtist,
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
) -> Path:
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

    Returns:
        Path to created video file

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

    # Create figure with multi-panel layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(
        4, 2, figure=fig, hspace=0.35, wspace=0.3, top=0.95, bottom=0.05, left=0.08, right=0.95
    )

    # Left column: Arena view (spans 3 rows)
    ax_arena = fig.add_subplot(gs[:3, 0])
    ax_arena.set_aspect("equal")
    ax_arena.set_xlabel("x (m)", fontsize=9)
    ax_arena.set_ylabel("y (m)", fontsize=9)
    ax_arena.set_title("Arena View", fontweight="normal", loc="left", fontsize=10)

    # Right column: IMU plots (3 rows)
    ax_gyro = fig.add_subplot(gs[0, 1])
    ax_accel_x = fig.add_subplot(gs[1, 1], sharex=ax_gyro)
    ax_accel_y = fig.add_subplot(gs[2, 1], sharex=ax_gyro)

    ax_gyro.set_title("IMU Measurements", fontweight="normal", loc="left", fontsize=10)

    # Bottom row: Camera status
    ax_camera = fig.add_subplot(gs[3, :])
    ax_camera.set_title("Camera Status", fontweight="normal", loc="left", fontsize=10)

    # Set arena bounds based on data
    X_truth = video_data["X_truth"]
    x_min, x_max = X_truth[:, 0].min(), X_truth[:, 0].max()
    y_min, y_max = X_truth[:, 1].min(), X_truth[:, 1].max()

    # Add 10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    ax_arena.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax_arena.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

    # Initialize artists
    print("Initializing artists...")
    rat = RatArtist(ax_arena)
    led1 = LEDArtist(ax_arena, led_id=1, color=COLORS["blue"])
    led2 = LEDArtist(ax_arena, led_id=2, color=COLORS["orange"])
    trail = TrailArtist(ax_arena, trail_length_s=trail_length_s, fps=fps)
    hud = HUDArtist(ax_arena)
    events = EventMarkerArtist(ax_arena)
    imu_panel = IMUPanelArtist(
        [ax_gyro, ax_accel_x, ax_accel_y],
        window_s=time_window_s,
        fps=fps,
        config=sim_data["config"],
    )
    camera_panel = CameraPanelArtist(ax_camera)

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
            label="Velocity (motion)",
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
    ax_arena.legend(
        handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.9
    )

    # Overall title
    config = sim_data["config"]
    title_str = f"Diagnostic Video | {config.duration_s:.0f}s simulation | {speedup:.1f}x speed"
    fig.suptitle(title_str, fontsize=11, fontweight="normal")

    # Collect all artists that need to be returned for blitting
    all_artists = []

    # Arena artists
    all_artists.extend([rat.body, rat.heading_arrow, rat.velocity_arrow])
    all_artists.extend([led1.marker, led1.halo, led1.dropout_marker])
    all_artists.extend([led2.marker, led2.halo, led2.dropout_marker])
    all_artists.append(trail.lines)
    all_artists.extend([hud.time_text, hud.state_text])
    all_artists.append(events.banner)

    # IMU artists
    all_artists.extend([imu_panel.gyro_line, imu_panel.accel_x_line, imu_panel.accel_y_line])

    # Camera artists
    all_artists.extend([camera_panel.led1_bar, camera_panel.led2_bar])

    # Animation update function
    def update_frame(frame_idx: int) -> list[Any]:
        """Update all artists for a single frame."""
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

        # Detect events
        # LED swap: check if spacing deviates significantly (heuristic)
        led_swap = False
        if led1_visible and led2_visible:
            spacing = np.linalg.norm(led1_pos - led2_pos)
            expected_spacing = np.linalg.norm(config.led1_offset_body - config.led2_offset_body)
            # Consider swap if spacing differs by >50%
            if abs(spacing - expected_spacing) > 0.5 * expected_spacing:
                led_swap = True

        # Long dropout: consecutive frames without either LED
        long_dropout = not (led1_visible or led2_visible)

        # Update artists
        rat.update(x, y, theta, vx, vy)
        led1.update(
            led1_pos[0] if not np.isnan(led1_pos[0]) else x,
            led1_pos[1] if not np.isnan(led1_pos[1]) else y,
            led1_visible,
            conf1,
        )
        led2.update(
            led2_pos[0] if not np.isnan(led2_pos[0]) else x,
            led2_pos[1] if not np.isnan(led2_pos[1]) else y,
            led2_visible,
            conf2,
        )
        trail.update(x, y)

        hud_state = {
            "speed": speed,
            "theta": theta,
            "led1_visible": led1_visible,
            "led2_visible": led2_visible,
            "conf1": conf1,
            "conf2": conf2,
        }
        hud.update(t, hud_state)

        event_dict = {"led_swap": led_swap, "long_dropout": long_dropout}
        events.update(event_dict)

        # IMU panel
        imu_dict = {"gyro": imu[0], "accel_x": imu[1], "accel_y": imu[2]}
        imu_panel.update(t, imu_dict)

        # Camera panel
        camera_panel.update(led1_visible, conf1, led2_visible, conf2)

        return all_artists

    # Create animation
    print("Rendering animation...")
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=n_frames,
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )

    # Save video
    print(f"Encoding video to {output_path}...")
    try:
        # Try ffmpeg writer
        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(fps=fps, codec=codec, bitrate=bitrate)
        anim.save(str(output_path), writer=writer, dpi=dpi)
        print(f"✓ Video saved: {output_path}")
    except Exception as e:
        print(f"✗ FFmpeg encoding failed: {e}")
        print("  Falling back to pillow (GIF)...")

        # Fallback to GIF
        output_path = output_path.with_suffix(".gif")
        anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
        print(f"✓ GIF saved: {output_path}")

    plt.close(fig)

    return output_path
