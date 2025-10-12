"""Visualize complete IMU sensor data with video overlay.

This script shows ALL sensor data from the Arthur session:
- 3D Gyroscope (angular rates around X, Y, Z axes)
- 3D Accelerometer (linear acceleration along X, Y, Z axes)
- Video with LED tracking overlay

Layout is intuitive, grouping by sensor type with physically meaningful axes.

Design principles (Raymond Hettinger style):
- Small, composable functions
- Clear names that reveal intent
- Separation of concerns
- Pure functions where possible
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# SpikeGadgets headstage hardware specifications
GYRO_SCALE = 0.061  # deg/s per LSB (±2000 deg/s range, 16-bit)
ACCEL_SCALE = 0.000061  # g per LSB (±2g range, 16-bit)
GRAVITY = 9.80665  # m/s²
DEG_TO_RAD = np.pi / 180.0


def load_all_imu_data(imu_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load complete 3D IMU data with unit conversion.

    Parameters
    ----------
    imu_file : str
        Path to IMU parquet file

    Returns
    -------
    timestamps : np.ndarray
        Unix timestamps (float) [N]
    gyro_rad_s : np.ndarray
        Gyroscope in rad/s [N × 3]: [X, Y, Z]
    accel_m_s2 : np.ndarray
        Accelerometer in m/s² [N × 3]: [X, Y, Z]
    """
    df = pd.read_parquet(imu_file)

    # Get timestamps and convert to relative time
    timestamps = df.index.values
    t_start = timestamps[0]
    timestamps_rel = timestamps - t_start

    # Remove sample-and-hold: find where any axis changes
    all_values = np.column_stack([df[col].values for col in df.columns])
    changes = np.where(np.any(np.diff(all_values, axis=0) != 0, axis=1))[0] + 1
    unique_idx = np.concatenate([[0], changes])

    timestamps_rel = timestamps_rel[unique_idx]

    # Load and convert gyro (all 3 axes)
    gyro_raw = np.column_stack(
        [df["Headstage_GyroX"].values, df["Headstage_GyroY"].values, df["Headstage_GyroZ"].values]
    )
    gyro_raw = gyro_raw[unique_idx]
    gyro_deg_s = gyro_raw * GYRO_SCALE
    gyro_rad_s = gyro_deg_s * DEG_TO_RAD

    # Load and convert accel (all 3 axes)
    accel_raw = np.column_stack(
        [
            df["Headstage_AccelX"].values,
            df["Headstage_AccelY"].values,
            df["Headstage_AccelZ"].values,
        ]
    )
    accel_raw = accel_raw[unique_idx]
    accel_g = accel_raw * ACCEL_SCALE
    accel_m_s2 = accel_g * GRAVITY

    return timestamps_rel, gyro_rad_s, accel_m_s2


def load_camera_data(
    position_file: str, meters_per_pixel: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load camera tracking data.

    Parameters
    ----------
    position_file : str
        Path to position parquet file
    meters_per_pixel : float
        Conversion factor

    Returns
    -------
    timestamps : np.ndarray
        Relative timestamps in seconds [N]
    led1_pixels : np.ndarray
        LED1 positions in pixels [N × 2]
    led2_pixels : np.ndarray
        LED2 positions in pixels [N × 2]
    """
    df = pd.read_parquet(position_file)

    timestamps = df.index.values
    t_start = timestamps[0]
    timestamps_rel = timestamps - t_start

    led1_pixels = df[["xloc", "yloc"]].values
    led2_pixels = df[["xloc2", "yloc2"]].values

    return timestamps_rel, led1_pixels, led2_pixels


def find_nearest_index(timestamps: np.ndarray, target_time: float) -> int:
    """Find index of timestamp nearest to target time."""
    return int(np.argmin(np.abs(timestamps - target_time)))


def extract_time_window(
    timestamps: np.ndarray, data: np.ndarray, center_time: float, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract data within a time window around center time."""
    half_window = window_s / 2.0
    mask = (timestamps >= center_time - half_window) & (timestamps <= center_time + half_window)
    return timestamps[mask], data[mask]


def load_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Load a single video frame as RGB."""
    if not HAS_CV2:
        return None

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    if not HAS_CV2:
        return None

    cap = cv2.VideoCapture(video_path)
    try:
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_s": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
    finally:
        cap.release()


def setup_figure(
    gyro_ylim: tuple[float, float] = (-200.0, 200.0),
    accel_ylim: tuple[float, float] = (-15.0, 15.0),
) -> tuple[plt.Figure, dict]:
    """Create comprehensive figure layout showing all sensors.

    Layout:
    - Left: Video with LED overlay
    - Right top: 3D Gyroscope (X, Y, Z angular rates)
    - Right bottom: 3D Accelerometer (X, Y, Z acceleration)

    Parameters
    ----------
    gyro_ylim : Tuple[float, float]
        Y-axis limits for gyroscope (deg/s)
    accel_ylim : Tuple[float, float]
        Y-axis limits for accelerometer (m/s²)

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    axes : dict
        Dictionary with keys: 'video', 'gyro', 'accel'
    """
    fig = plt.figure(figsize=(14, 8))

    # Create grid: video on left, sensors stacked on right
    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.3,
        wspace=0.25,
        left=0.07,
        right=0.97,
        top=0.95,
        bottom=0.08,
        width_ratios=[1.2, 1],
    )

    # Video (left column, spans both rows)
    ax_video = fig.add_subplot(gs[:, 0])
    ax_video.set_aspect("equal")
    ax_video.axis("off")
    ax_video.set_title("Video with LED Tracking", fontsize=12, fontweight="bold", pad=10)

    # Gyroscope (right top)
    ax_gyro = fig.add_subplot(gs[0, 1])
    ax_gyro.set_ylabel("Angular Rate (deg/s)", fontsize=10)
    ax_gyro.set_ylim(gyro_ylim)
    ax_gyro.grid(True, alpha=0.3, linewidth=0.5)
    ax_gyro.set_title("3-Axis Gyroscope", fontsize=11, fontweight="bold")
    ax_gyro.tick_params(labelsize=9)

    # Accelerometer (right bottom)
    ax_accel = fig.add_subplot(gs[1, 1])
    ax_accel.set_xlabel("Time (s)", fontsize=10)
    ax_accel.set_ylabel("Acceleration (m/s²)", fontsize=10)
    ax_accel.set_ylim(accel_ylim)
    ax_accel.grid(True, alpha=0.3, linewidth=0.5)
    ax_accel.set_title("3-Axis Accelerometer", fontsize=11, fontweight="bold")
    ax_accel.tick_params(labelsize=9)

    # Add reference lines
    ax_accel.axhline(GRAVITY, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="1g")
    ax_accel.axhline(-GRAVITY, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    axes = {"video": ax_video, "gyro": ax_gyro, "accel": ax_accel}

    return fig, axes


def create_comprehensive_video(
    video_path: str,
    position_file: str,
    imu_file: str,
    output_path: str,
    meters_per_pixel: float = 0.0022,
    start_time: float = 0.0,
    duration: float = 10.0,
    fps: float = 30.0,
    imu_window_s: float = 2.0,
    led_marker_size: float = 10.0,
    dpi: int = 100,
    gyro_ylim: tuple[float, float] = (-200.0, 200.0),
    accel_ylim: tuple[float, float] = (-15.0, 15.0),
) -> None:
    """Create video showing all sensor data with intuitive layout.

    Parameters
    ----------
    video_path : str
        Path to input video
    position_file : str
        Path to position parquet
    imu_file : str
        Path to IMU parquet
    output_path : str
        Output video path
    meters_per_pixel : float
        Conversion factor (default: 0.0022)
    start_time : float
        Start time in seconds (default: 0)
    duration : float
        Duration in seconds (default: 10)
    fps : float
        Output frame rate (default: 30)
    imu_window_s : float
        Time window for IMU plots (default: 2s)
    led_marker_size : float
        LED marker radius in pixels (default: 10)
    dpi : int
        Output resolution (default: 100)
    gyro_ylim : Tuple[float, float]
        Gyro Y-axis limits in deg/s (default: ±200)
    accel_ylim : Tuple[float, float]
        Accel Y-axis limits in m/s² (default: ±15)

    Notes
    -----
    Shows complete 6-axis IMU data:
    - Gyro X, Y, Z (roll, pitch, yaw rates)
    - Accel X, Y, Z (including gravity component)
    """
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load data
    print("\nLoading IMU data (3D gyro + 3D accel)...")
    t_imu, gyro_rad_s, accel_m_s2 = load_all_imu_data(imu_file)
    gyro_deg_s = gyro_rad_s * 180 / np.pi
    print(f"✓ Loaded {len(t_imu):,} IMU samples at ~{1 / np.median(np.diff(t_imu)):.1f} Hz")

    print("\nLoading camera tracking data...")
    t_cam, led1_pixels, led2_pixels = load_camera_data(position_file, meters_per_pixel)
    print(f"✓ Loaded {len(t_cam):,} camera frames at ~{1 / np.median(np.diff(t_cam)):.1f} Hz")

    print("\nLoading video metadata...")
    video_info = get_video_info(video_path)
    if not video_info:
        raise RuntimeError("opencv-python required for video")
    print(f"✓ Video: {video_info['width']}×{video_info['height']} @ {video_info['fps']:.1f} fps")

    # Validate time range
    end_time = start_time + duration
    max_time = min(t_cam[-1], t_imu[-1], video_info["duration_s"])
    if end_time > max_time:
        print(f"\n⚠ Warning: Requested {end_time:.1f}s exceeds data ({max_time:.1f}s)")
        duration = max_time - start_time
        end_time = max_time

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)
    print(f"\nOutput: {output_path}")
    print(f"  Time range: {start_time:.1f}s to {end_time:.1f}s ({duration:.1f}s)")
    print(f"  Output fps: {fps:.1f}")
    print(f"  IMU window: {imu_window_s:.1f}s")

    # Setup figure
    fig, axes = setup_figure(gyro_ylim, accel_ylim)

    # Initialize video display
    video_frame = axes["video"].imshow(
        np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8)
    )
    led1_circle = Circle((0, 0), led_marker_size, color="#FF4444", alpha=0.8, label="LED1 (rear)")
    led2_circle = Circle((0, 0), led_marker_size, color="#44FFFF", alpha=0.8, label="LED2 (front)")
    axes["video"].add_patch(led1_circle)
    axes["video"].add_patch(led2_circle)
    axes["video"].legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Initialize gyro plot (3 axes)
    colors_gyro = {"X": "#E74C3C", "Y": "#2ECC71", "Z": "#3498DB"}  # Red, Green, Blue
    gyro_lines = {}
    for _i, axis in enumerate(["X", "Y", "Z"]):
        (line,) = axes["gyro"].plot(
            [], [], color=colors_gyro[axis], linewidth=1.8, label=f"Gyro {axis}", alpha=0.9
        )
        gyro_lines[axis] = line
    gyro_marker = axes["gyro"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
    axes["gyro"].legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Initialize accel plot (3 axes)
    colors_accel = {"X": "#E74C3C", "Y": "#2ECC71", "Z": "#3498DB"}  # Red, Green, Blue
    accel_lines = {}
    for _i, axis in enumerate(["X", "Y", "Z"]):
        (line,) = axes["accel"].plot(
            [], [], color=colors_accel[axis], linewidth=1.8, label=f"Accel {axis}", alpha=0.9
        )
        accel_lines[axis] = line
    accel_marker = axes["accel"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
    axes["accel"].legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)

    # Time display on video
    time_text = axes["video"].text(
        0.02,
        0.98,
        "",
        transform=axes["video"].transAxes,
        fontsize=13,
        verticalalignment="top",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="black"),
    )

    def init():
        """Initialize animation."""
        artists = [video_frame, led1_circle, led2_circle, gyro_marker, accel_marker, time_text]
        artists.extend(gyro_lines.values())
        artists.extend(accel_lines.values())
        return artists

    def update(frame_num: int):
        """Update animation frame."""
        current_time = start_time + frame_num / fps

        # Load and display video frame
        video_frame_idx = int(current_time * video_info["fps"])
        frame = load_video_frame(video_path, video_frame_idx)
        if frame is not None:
            video_frame.set_data(frame)

        # Update LED positions
        cam_idx = find_nearest_index(t_cam, current_time)
        led1_circle.center = tuple(led1_pixels[cam_idx])
        led2_circle.center = tuple(led2_pixels[cam_idx])

        # Extract gyro data in window (all 3 axes)
        for i, axis in enumerate(["X", "Y", "Z"]):
            t_window, data_window = extract_time_window(
                t_imu, gyro_deg_s[:, i], current_time, imu_window_s
            )
            gyro_lines[axis].set_data(t_window, data_window)

        # Extract accel data in window (all 3 axes)
        for i, axis in enumerate(["X", "Y", "Z"]):
            t_window, data_window = extract_time_window(
                t_imu, accel_m_s2[:, i], current_time, imu_window_s
            )
            accel_lines[axis].set_data(t_window, data_window)

        # Update time axes
        xlim = (current_time - imu_window_s / 2, current_time + imu_window_s / 2)
        axes["gyro"].set_xlim(xlim)
        axes["gyro"].set_ylim(gyro_ylim)
        gyro_marker.set_xdata([current_time, current_time])

        axes["accel"].set_xlim(xlim)
        axes["accel"].set_ylim(accel_ylim)
        accel_marker.set_xdata([current_time, current_time])

        # Update time display
        time_text.set_text(f"t = {current_time:.2f} s")

        artists = [video_frame, led1_circle, led2_circle, gyro_marker, accel_marker, time_text]
        artists.extend(gyro_lines.values())
        artists.extend(accel_lines.values())
        return artists

    # Create animation
    n_frames = int(duration * fps)
    print(f"\nGenerating {n_frames} frames...")

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=True
    )

    # Save
    writer = FFMpegWriter(fps=fps, bitrate=6000)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"\n✓ Video saved to: {output_path}")
    print("=" * 80)


def main():
    """Generate comprehensive sensor visualization."""
    script_dir = Path(__file__).parent

    video_path = script_dir / "20220314_arthur_02_r1.mp4"
    position_file = script_dir / "arthur20220314_position_info.parquet"
    imu_file = script_dir / "arthur20220314_imu_info.parquet"
    output_path = script_dir / "arthur_all_sensors.mp4"

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    create_comprehensive_video(
        video_path=str(video_path),
        position_file=str(position_file),
        imu_file=str(imu_file),
        output_path=str(output_path),
        meters_per_pixel=0.0022,
        start_time=60.0,  # Start at 1 minute
        duration=10.0,  # 10 second clip
        fps=30.0,
        imu_window_s=2.0,  # 2s rolling window
        led_marker_size=10.0,
        dpi=100,
        gyro_ylim=(-200.0, 200.0),  # ±200 deg/s
        accel_ylim=(-15.0, 15.0),  # ±15 m/s² (covers ±1.5g)
    )

    print(f"\n✓ Complete! View output: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
