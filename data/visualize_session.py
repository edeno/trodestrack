"""Visualize Arthur session with video overlay and IMU data.

This script creates a synchronized video showing:
- Video frames with LED positions overlaid
- Gyroscope trace (heading rate)
- Accelerometer traces (X and Y)
- Timeline indicator

All data is properly aligned using timestamps.

Design principles (Raymond Hettinger style):
- Small, composable functions
- Clear names that reveal intent
- Separation of concerns
- Pure functions where possible
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle

from load_arthur_session import (
    SessionData,
    convert_meters_to_pixels,
    get_video_info,
    load_arthur_session,
    load_video_frame,
)


def find_nearest_index(timestamps: np.ndarray, target_time: float) -> int:
    """Find index of timestamp nearest to target time.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamps in seconds
    target_time : float
        Target time to find

    Returns
    -------
    int
        Index of nearest timestamp
    """
    return int(np.argmin(np.abs(timestamps - target_time)))


def extract_time_window(
    timestamps: np.ndarray, data: np.ndarray, center_time: float, window_s: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract data within a time window around center time.

    Parameters
    ----------
    timestamps : np.ndarray
        Data timestamps [N]
    data : np.ndarray
        Data array [N] or [N × D]
    center_time : float
        Center of time window in seconds
    window_s : float
        Total window duration in seconds

    Returns
    -------
    t_window : np.ndarray
        Timestamps in window
    data_window : np.ndarray
        Data in window
    """
    half_window = window_s / 2.0
    mask = (timestamps >= center_time - half_window) & (timestamps <= center_time + half_window)
    return timestamps[mask], data[mask]


def setup_figure() -> Tuple[plt.Figure, dict]:
    """Create figure layout for video visualization.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    axes : dict
        Dictionary of axis objects with keys: 'video', 'gyro', 'accel'
    """
    fig = plt.figure(figsize=(12, 8))

    # Create grid layout
    gs = fig.add_gridspec(
        3, 2, hspace=0.3, wspace=0.3, left=0.08, right=0.95, top=0.95, bottom=0.08
    )

    # Video takes left column
    ax_video = fig.add_subplot(gs[:, 0])
    ax_video.set_aspect("equal")
    ax_video.axis("off")
    ax_video.set_title("Video with Tracked LEDs", fontsize=12, fontweight="bold")

    # IMU plots on right
    ax_gyro = fig.add_subplot(gs[0, 1])
    ax_gyro.set_ylabel("Gyro Z (deg/s)", fontsize=10)
    ax_gyro.grid(True, alpha=0.3)
    ax_gyro.set_title("Angular Rate", fontsize=10, fontweight="bold")

    ax_accel = fig.add_subplot(gs[1:, 1])
    ax_accel.set_xlabel("Time (s)", fontsize=10)
    ax_accel.set_ylabel("Acceleration (m/s²)", fontsize=10)
    ax_accel.grid(True, alpha=0.3)
    ax_accel.set_title("Accelerometer", fontsize=10, fontweight="bold")

    axes = {"video": ax_video, "gyro": ax_gyro, "accel": ax_accel}

    return fig, axes


def create_video_overlay(
    video_path: str,
    data: SessionData,
    output_path: str,
    start_time: float = 0.0,
    duration: float = 10.0,
    fps: float = 30.0,
    imu_window_s: float = 2.0,
    led_marker_size: float = 8.0,
    dpi: int = 100,
) -> None:
    """Create video with position overlay and IMU data visualization.

    Parameters
    ----------
    video_path : str
        Path to input video file
    data : SessionData
        Loaded session data from load_arthur_session()
    output_path : str
        Path for output video file
    start_time : float, optional
        Start time in seconds (default: 0.0)
    duration : float, optional
        Duration of output video in seconds (default: 10.0)
    fps : float, optional
        Frame rate of output video (default: 30.0)
    imu_window_s : float, optional
        Time window for IMU plots in seconds (default: 2.0)
    led_marker_size : float, optional
        Size of LED markers in pixels (default: 8.0)
    dpi : int, optional
        Output resolution (default: 100)

    Notes
    -----
    Requires opencv-python and ffmpeg for video output.
    Output file format determined by extension (e.g., .mp4, .avi).
    """
    # Get video metadata
    video_info = get_video_info(video_path)
    if not video_info:
        raise RuntimeError("Could not read video info. Is opencv-python installed?")

    print("=" * 80)
    print("CREATING VIDEO VISUALIZATION")
    print("=" * 80)
    print(f"\nInput video: {video_path}")
    print(f"  Resolution: {video_info['width']}×{video_info['height']}")
    print(f"  Frame rate: {video_info['fps']:.1f} fps")
    print(f"  Duration: {video_info['duration_s']:.1f} s")
    print(f"\nOutput: {output_path}")
    print(f"  Start time: {start_time:.1f} s")
    print(f"  Duration: {duration:.1f} s")
    print(f"  Output fps: {fps:.1f}")

    # Validate time range
    end_time = start_time + duration
    max_time = min(data.t_cam[-1], video_info["duration_s"])
    if end_time > max_time:
        print(f"\nWarning: Requested end time {end_time:.1f}s exceeds data ({max_time:.1f}s)")
        end_time = max_time
        duration = end_time - start_time
        print(f"  Adjusted duration to {duration:.1f}s")

    # Convert positions to pixels
    led1_pixels = convert_meters_to_pixels(data.Z_cam_led1, data.meters_per_pixel)
    led2_pixels = convert_meters_to_pixels(data.Z_cam_led2, data.meters_per_pixel)

    # Convert gyro to deg/s for display
    gyro_deg_s = data.U_imu[:, 0] * 180 / np.pi
    accel_x = data.U_imu[:, 1]
    accel_y = data.U_imu[:, 2]

    # Setup figure
    fig, axes = setup_figure()

    # Initialize plot elements
    video_frame = axes["video"].imshow(
        np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8)
    )
    led1_circle = Circle((0, 0), led_marker_size, color="red", alpha=0.7, label="LED1 (back)")
    led2_circle = Circle((0, 0), led_marker_size, color="cyan", alpha=0.7, label="LED2 (front)")
    axes["video"].add_patch(led1_circle)
    axes["video"].add_patch(led2_circle)
    axes["video"].legend(loc="upper right", fontsize=8)

    # IMU plot elements
    (gyro_line,) = axes["gyro"].plot([], [], "b-", linewidth=1.5, label="Gyro Z")
    gyro_marker = axes["gyro"].axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)

    (accel_x_line,) = axes["accel"].plot([], [], "r-", linewidth=1.5, label="Accel X")
    (accel_y_line,) = axes["accel"].plot([], [], "g-", linewidth=1.5, label="Accel Y")
    accel_marker = axes["accel"].axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes["accel"].legend(loc="upper right", fontsize=8)

    # Time display
    time_text = axes["video"].text(
        0.02,
        0.98,
        "",
        transform=axes["video"].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def init():
        """Initialize animation."""
        return (
            video_frame,
            led1_circle,
            led2_circle,
            gyro_line,
            gyro_marker,
            accel_x_line,
            accel_y_line,
            accel_marker,
            time_text,
        )

    def update(frame_num: int):
        """Update frame.

        Parameters
        ----------
        frame_num : int
            Frame number in output video
        """
        # Compute current time
        current_time = start_time + frame_num / fps

        # Load video frame
        video_frame_idx = int(current_time * video_info["fps"])
        frame = load_video_frame(video_path, video_frame_idx)
        if frame is not None:
            video_frame.set_data(frame)

        # Find nearest camera measurement
        cam_idx = find_nearest_index(data.t_cam, current_time)
        led1_pos = led1_pixels[cam_idx]
        led2_pos = led2_pixels[cam_idx]

        # Update LED positions
        led1_circle.center = (led1_pos[0], led1_pos[1])
        led2_circle.center = (led2_pos[0], led2_pos[1])

        # Extract IMU data in time window
        t_gyro, gyro_window = extract_time_window(
            data.t_imu, gyro_deg_s, current_time, imu_window_s
        )
        t_accel_x, accel_x_window = extract_time_window(
            data.t_imu, accel_x, current_time, imu_window_s
        )
        t_accel_y, accel_y_window = extract_time_window(
            data.t_imu, accel_y, current_time, imu_window_s
        )

        # Update gyro plot
        gyro_line.set_data(t_gyro, gyro_window)
        axes["gyro"].set_xlim(current_time - imu_window_s / 2, current_time + imu_window_s / 2)
        if len(gyro_window) > 0:
            y_margin = max(20, np.ptp(gyro_window) * 0.2)
            axes["gyro"].set_ylim(gyro_window.min() - y_margin, gyro_window.max() + y_margin)
        gyro_marker.set_xdata([current_time, current_time])

        # Update accelerometer plot
        accel_x_line.set_data(t_accel_x, accel_x_window)
        accel_y_line.set_data(t_accel_y, accel_y_window)
        axes["accel"].set_xlim(current_time - imu_window_s / 2, current_time + imu_window_s / 2)
        if len(accel_x_window) > 0 and len(accel_y_window) > 0:
            all_accel = np.concatenate([accel_x_window, accel_y_window])
            y_margin = max(2, np.ptp(all_accel) * 0.2)
            axes["accel"].set_ylim(all_accel.min() - y_margin, all_accel.max() + y_margin)
        accel_marker.set_xdata([current_time, current_time])

        # Update time display
        time_text.set_text(f"t = {current_time:.2f} s")

        return (
            video_frame,
            led1_circle,
            led2_circle,
            gyro_line,
            gyro_marker,
            accel_x_line,
            accel_y_line,
            accel_marker,
            time_text,
        )

    # Create animation
    n_frames = int(duration * fps)
    print(f"\nGenerating {n_frames} frames...")

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=True
    )

    # Save video
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_path, writer=writer, dpi=dpi)

    plt.close(fig)

    print(f"\n✓ Video saved to: {output_path}")
    print("=" * 80)


def main():
    """Example usage: create visualization video."""
    script_dir = Path(__file__).parent

    # Load data
    print("Loading session data...")
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220314_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220314_imu_info.parquet"),
        meters_per_pixel=0.0022,
        verbose=False,
    )
    print(f"✓ Loaded {len(data.t_cam):,} frames and {len(data.t_imu):,} IMU samples\n")

    # Create visualization
    video_path = script_dir / "20220314_arthur_02_r1.mp4"
    output_path = "/tmp/arthur_visualization.mp4"

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    create_video_overlay(
        video_path=str(video_path),
        data=data,
        output_path=output_path,
        start_time=60.0,  # Start at 1 minute
        duration=10.0,  # 10 second clip
        fps=30.0,
        imu_window_s=2.0,  # Show 2s of IMU data
        led_marker_size=10.0,
        dpi=100,
    )

    print(f"\n✓ Complete! View output: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
