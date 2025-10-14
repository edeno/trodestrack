"""Visualize Arthur session with filter overlay and IMU data.

This script creates a synchronized video showing:
- Video frames with tracked LED positions and filter trajectory
- Filter state estimates (position, velocity, heading)
- 3D IMU data (gyroscope and accelerometer)
- Comparison between measured and estimated states

Design principles:
- Clear separation of data, rendering, and animation concerns
- Pedagogical presentation of filter behavior
- Proper time synchronization across all data streams
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # faster headless rendering
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_arthur_session import (
    SessionData,
    convert_meters_to_pixels,
    get_video_info,
    load_arthur_session,
    load_video_frame,
)
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle


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


def setup_figure() -> tuple[plt.Figure, dict]:
    """Create figure layout for filter visualization.

    Layout:
    - Top left: Video with trajectory overlay
    - Top right: IMU data (gyroscope + accelerometer)
    - Bottom left: Position and velocity estimates
    - Bottom right: Heading and uncertainty

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    axes : dict
        Dictionary of axis objects
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid layout
    gs = fig.add_gridspec(
        3, 3, hspace=0.35, wspace=0.35, left=0.05, right=0.97, top=0.95, bottom=0.05
    )

    # Video (top-left, tall)
    ax_video = fig.add_subplot(gs[:2, :2])
    ax_video.set_aspect("equal")
    ax_video.axis("off")
    ax_video.set_title("Video with Filter Overlay", fontsize=12, fontweight="bold", pad=10)

    # IMU plots (right column)
    ax_gyro = fig.add_subplot(gs[0, 2])
    ax_gyro.set_ylabel("Gyro (deg/s)", fontsize=9)
    ax_gyro.grid(True, alpha=0.3)
    ax_gyro.set_title("Gyroscope (3-axis)", fontsize=10, fontweight="bold")
    ax_gyro.tick_params(labelsize=8)

    ax_accel = fig.add_subplot(gs[1, 2])
    ax_accel.set_ylabel("Accel (m/s²)", fontsize=9)
    ax_accel.grid(True, alpha=0.3)
    ax_accel.set_title("Accelerometer (3-axis)", fontsize=10, fontweight="bold")
    ax_accel.tick_params(labelsize=8)

    # State estimates (bottom row)
    ax_pos_vel = fig.add_subplot(gs[2, 0])
    ax_pos_vel.set_xlabel("Time (s)", fontsize=9)
    ax_pos_vel.set_ylabel("Speed (cm/s)", fontsize=9)
    ax_pos_vel.grid(True, alpha=0.3)
    ax_pos_vel.set_title("Velocity Magnitude", fontsize=10, fontweight="bold")
    ax_pos_vel.tick_params(labelsize=8)

    ax_heading = fig.add_subplot(gs[2, 1])
    ax_heading.set_xlabel("Time (s)", fontsize=9)
    ax_heading.set_ylabel("Heading (deg)", fontsize=9)
    ax_heading.grid(True, alpha=0.3)
    ax_heading.set_title("Heading Estimate", fontsize=10, fontweight="bold")
    ax_heading.tick_params(labelsize=8)

    ax_uncertainty = fig.add_subplot(gs[2, 2])
    ax_uncertainty.set_xlabel("Time (s)", fontsize=9)
    ax_uncertainty.set_ylabel("Position std (cm)", fontsize=9)
    ax_uncertainty.grid(True, alpha=0.3)
    ax_uncertainty.set_title("Position Uncertainty", fontsize=10, fontweight="bold")
    ax_uncertainty.tick_params(labelsize=8)

    axes = {
        "video": ax_video,
        "gyro": ax_gyro,
        "accel": ax_accel,
        "pos_vel": ax_pos_vel,
        "heading": ax_heading,
        "uncertainty": ax_uncertainty,
    }

    return fig, axes


def create_filter_overlay_video(
    video_path: str,
    data: SessionData,
    filter_result,
    t_filter: np.ndarray,
    position_df: pd.DataFrame,
    output_path: str,
    start_time: float = 0.0,
    duration: float = 10.0,
    fps: float = 30.0,
    imu_window_s: float = 2.0,
    state_window_s: float = 5.0,
    led_marker_size: float = 8.0,
    trajectory_length: int = 90,  # frames to show in trajectory
    dpi: int = 100,
) -> None:
    """Create video with filter overlay and IMU/state visualization.

    Parameters
    ----------
    video_path : str
        Path to input video file
    data : SessionData
        Loaded session data from load_arthur_session()
    filter_result
        Filter result from extended_kalman_filter()
    t_filter : np.ndarray
        Filter timestamps (typically same as data.t_cam)
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
    state_window_s : float, optional
        Time window for state plots in seconds (default: 5.0)
    led_marker_size : float, optional
        Size of LED markers in pixels (default: 8.0)
    trajectory_length : int, optional
        Number of past frames to show in trajectory (default: 90 = 3s at 30fps)
    dpi : int, optional
        Output resolution (default: 100)

    Notes
    -----
    Requires opencv-python and ffmpeg for video output.
    """
    # Get video metadata
    video_info = get_video_info(video_path)
    if not video_info:
        raise RuntimeError("Could not read video info. Is opencv-python installed?")

    print("=" * 80)
    print("CREATING FILTER OVERLAY VIDEO")
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

    # Extract filter states
    try:
        X_filter = np.array(filter_result.filtered_means)  # [T × n_state]
        P_filter = np.array(filter_result.filtered_covariances)  # [T × n_state × n_state]
    except AttributeError:
        X_filter = np.array(filter_result.smoothed_means)  # [T × n_state]
        P_filter = np.array(filter_result.smoothed_covariances)  # [T × n_state × n_state]

    # Extract state components (assuming LAYOUT_2D_CAM_3D_IMU: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az])
    pos_filter = X_filter[:, :2]  # [T × 2]
    vel_filter = X_filter[:, 2:5]  # [T × 3] (vx, vy, vz)
    heading_filter = X_filter[:, 5]  # [T]

    # Compute position uncertainty (std of x, y)
    pos_std = np.sqrt(P_filter[:, 0, 0] + P_filter[:, 1, 1])  # [T]

    # Compute velocity magnitude
    vel_mag = np.sqrt(vel_filter[:, 0] ** 2 + vel_filter[:, 1] ** 2)  # [T] (2D velocity)

    # Convert positions to pixels for overlay
    pos_filter_pixels = convert_meters_to_pixels(pos_filter, data.meters_per_pixel)
    led1_pixels = convert_meters_to_pixels(data.Z_cam_led1, data.meters_per_pixel)
    led2_pixels = convert_meters_to_pixels(data.Z_cam_led2, data.meters_per_pixel)

    # Convert IMU data for display
    # 3D IMU mode
    gyro_x = data.U_imu[:, 0] * 180 / np.pi
    gyro_y = data.U_imu[:, 1] * 180 / np.pi
    gyro_z = data.U_imu[:, 2] * 180 / np.pi
    accel_x = data.U_imu[:, 3]
    accel_y = data.U_imu[:, 4]
    accel_z = data.U_imu[:, 5]

    # Setup figure
    # Precompute per-frame indices and cache video frames
    n_frames = int(duration * fps)
    frame_times = start_time + np.arange(n_frames) / fps
    # nearest filter idx for each animation frame
    frame_to_filter_idx = np.clip(
        np.searchsorted(t_filter, frame_times, side="left"), 0, len(t_filter) - 1
    )
    # nearest camera (LED) idx for each animation frame
    frame_to_cam_idx = np.clip(
        np.searchsorted(data.t_cam, frame_times, side="left"), 0, len(data.t_cam) - 1
    )
    # map to actual video frame indices via position_df
    pos_vid_inds = position_df["video_frame_ind"].to_numpy()
    frame_to_video_ind = pos_vid_inds[frame_to_filter_idx]

    # --- Parallel frame caching ---
    # Decode all required frames concurrently to minimize I/O latency.
    def _read_one(video_idx: int):
        return load_video_frame(video_path, int(video_idx))

    cached_frames = [None] * n_frames
    max_workers = min(8, (os.cpu_count() or 4))  # modest default
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_read_one, int(vf)): i for i, vf in enumerate(frame_to_video_ind)}
        for fut in as_completed(futures):
            i = futures[fut]
            cached_frames[i] = fut.result()

    fig, axes = setup_figure()

    # Initialize video frame
    video_frame = axes["video"].imshow(
        np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8)
    )

    # Initialize trajectory line
    (trajectory_line,) = axes["video"].plot(
        [], [], "c-", linewidth=1.5, alpha=0.6, label="Filter trajectory"
    )

    # Initialize LED circles
    led1_circle = Circle((0, 0), led_marker_size, color="red", alpha=0.5, label="LED1 (back)")
    led2_circle = Circle((0, 0), led_marker_size, color="yellow", alpha=0.5, label="LED2 (front)")
    axes["video"].add_patch(led1_circle)
    axes["video"].add_patch(led2_circle)

    # Initialize filter position marker
    filter_circle = Circle(
        (0, 0),
        led_marker_size * 1.2,
        color="cyan",
        alpha=0.8,
        linewidth=2,
        fill=False,
        label="Filter estimate",
    )
    axes["video"].add_patch(filter_circle)

    # Initialize heading indicator as a simple line (blit-friendly)
    (heading_line_video,) = axes["video"].plot(
        [], [], color="cyan", linewidth=2, alpha=0.7, label="Heading"
    )

    # Legend for video
    axes["video"].legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Initialize IMU plots
    (gyro_x_line,) = axes["gyro"].plot([], [], "r-", linewidth=1.2, label="X", alpha=0.8)
    (gyro_y_line,) = axes["gyro"].plot([], [], "g-", linewidth=1.2, label="Y", alpha=0.8)
    (gyro_z_line,) = axes["gyro"].plot([], [], "b-", linewidth=1.2, label="Z", alpha=0.8)
    gyro_marker = axes["gyro"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    axes["gyro"].legend(loc="upper right", fontsize=7)

    (accel_x_line,) = axes["accel"].plot([], [], "r-", linewidth=1.2, label="X", alpha=0.8)
    (accel_y_line,) = axes["accel"].plot([], [], "g-", linewidth=1.2, label="Y", alpha=0.8)
    (accel_z_line,) = axes["accel"].plot([], [], "b-", linewidth=1.2, label="Z", alpha=0.8)
    accel_marker = axes["accel"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    axes["accel"].legend(loc="upper right", fontsize=7)

    # Initialize state plots
    (vel_line,) = axes["pos_vel"].plot([], [], "b-", linewidth=1.5, label="Speed")
    vel_marker = axes["pos_vel"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    (heading_line,) = axes["heading"].plot([], [], "m-", linewidth=1.5, label="Heading")
    heading_marker = axes["heading"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    (unc_line,) = axes["uncertainty"].plot([], [], "orange", linewidth=1.5, label="Position σ")
    unc_marker = axes["uncertainty"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    # Time display
    time_text = axes["video"].text(
        0.02,
        0.98,
        "",
        transform=axes["video"].transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    def init():
        """Initialize animation."""
        return (
            video_frame,
            trajectory_line,
            led1_circle,
            led2_circle,
            filter_circle,
            heading_line_video,
            gyro_x_line,
            gyro_y_line,
            gyro_z_line,
            gyro_marker,
            accel_x_line,
            accel_y_line,
            accel_z_line,
            accel_marker,
            vel_line,
            vel_marker,
            heading_line,
            heading_marker,
            unc_line,
            unc_marker,
            time_text,
        )

    def update(frame_num: int):
        """Update frame."""
        # Compute current time and precomputed indices
        current_time = frame_times[frame_num]
        filter_idx = frame_to_filter_idx[frame_num]
        cam_idx = frame_to_cam_idx[frame_num]

        # Use cached video frame
        frame = cached_frames[frame_num]
        if frame is not None:
            video_frame.set_data(frame)

        current_pos = pos_filter_pixels[filter_idx]
        current_heading = heading_filter[filter_idx]

        # Update trajectory (show past N frames)
        traj_start = max(0, filter_idx - trajectory_length)
        traj_x = pos_filter_pixels[traj_start : filter_idx + 1, 0]
        traj_y = pos_filter_pixels[traj_start : filter_idx + 1, 1]
        trajectory_line.set_data(traj_x, traj_y)

        # Find nearest camera measurements
        led1_pos = led1_pixels[cam_idx]
        led2_pos = led2_pixels[cam_idx]

        # Update LED positions
        led1_circle.center = (led1_pos[0], led1_pos[1])
        led2_circle.center = (led2_pos[0], led2_pos[1])

        # Update filter position
        filter_circle.center = (current_pos[0], current_pos[1])

        # Update heading line (blit-friendly)
        arrow_length = 40  # pixels
        arrow_heading = current_heading + np.pi  # keep your 180° correction
        dx = arrow_length * np.cos(arrow_heading)
        dy = arrow_length * np.sin(arrow_heading)
        heading_line_video.set_data(
            [current_pos[0], current_pos[0] + dx], [current_pos[1], current_pos[1] + dy]
        )

        # Extract IMU data in time window
        t_gyro_x, gyro_x_window = extract_time_window(
            data.t_imu, gyro_x, current_time, imu_window_s
        )
        t_gyro_y, gyro_y_window = extract_time_window(
            data.t_imu, gyro_y, current_time, imu_window_s
        )
        t_gyro_z, gyro_z_window = extract_time_window(
            data.t_imu, gyro_z, current_time, imu_window_s
        )

        t_accel_x, accel_x_window = extract_time_window(
            data.t_imu, accel_x, current_time, imu_window_s
        )
        t_accel_y, accel_y_window = extract_time_window(
            data.t_imu, accel_y, current_time, imu_window_s
        )
        t_accel_z, accel_z_window = extract_time_window(
            data.t_imu, accel_z, current_time, imu_window_s
        )

        # Update gyro plots
        gyro_x_line.set_data(t_gyro_x, gyro_x_window)
        gyro_y_line.set_data(t_gyro_y, gyro_y_window)
        gyro_z_line.set_data(t_gyro_z, gyro_z_window)
        axes["gyro"].set_xlim(current_time - imu_window_s / 2, current_time + imu_window_s / 2)
        axes["gyro"].set_ylim(-200, 200)
        gyro_marker.set_xdata([current_time, current_time])

        # Update accelerometer plots
        accel_x_line.set_data(t_accel_x, accel_x_window)
        accel_y_line.set_data(t_accel_y, accel_y_window)
        accel_z_line.set_data(t_accel_z, accel_z_window)
        axes["accel"].set_xlim(current_time - imu_window_s / 2, current_time + imu_window_s / 2)
        axes["accel"].set_ylim(-15, 15)
        accel_marker.set_xdata([current_time, current_time])

        # Extract filter states in time window
        t_state, vel_window = extract_time_window(
            t_filter, vel_mag * 100, current_time, state_window_s
        )  # cm/s
        _, heading_window = extract_time_window(
            t_filter, heading_filter * 180 / np.pi, current_time, state_window_s
        )  # deg
        _, unc_window = extract_time_window(
            t_filter, pos_std * 100, current_time, state_window_s
        )  # cm

        # Update velocity plot
        vel_line.set_data(t_state, vel_window)
        axes["pos_vel"].set_xlim(
            current_time - state_window_s / 2, current_time + state_window_s / 2
        )
        axes["pos_vel"].set_ylim(
            0, max(100, np.max(vel_window) * 1.2) if len(vel_window) > 0 else 100
        )
        vel_marker.set_xdata([current_time, current_time])

        # Update heading plot
        heading_line.set_data(t_state, heading_window)
        axes["heading"].set_xlim(
            current_time - state_window_s / 2, current_time + state_window_s / 2
        )
        axes["heading"].set_ylim(-180, 180)
        heading_marker.set_xdata([current_time, current_time])

        # Update uncertainty plot
        unc_line.set_data(t_state, unc_window)
        axes["uncertainty"].set_xlim(
            current_time - state_window_s / 2, current_time + state_window_s / 2
        )
        axes["uncertainty"].set_ylim(
            0, max(2, np.max(unc_window) * 1.2) if len(unc_window) > 0 else 2
        )
        unc_marker.set_xdata([current_time, current_time])

        # Update time display
        time_text.set_text(f"t = {current_time:.2f} s")

        return (
            video_frame,
            trajectory_line,
            led1_circle,
            led2_circle,
            filter_circle,
            heading_line_video,
            gyro_x_line,
            gyro_y_line,
            gyro_z_line,
            gyro_marker,
            accel_x_line,
            accel_y_line,
            accel_z_line,
            accel_marker,
            vel_line,
            vel_marker,
            heading_line,
            heading_marker,
            unc_line,
            unc_marker,
            time_text,
        )

    # Create animation
    print(f"\nGenerating {n_frames} frames...")

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=True
    )

    # Save video
    writer = FFMpegWriter(fps=fps, bitrate=8000)
    anim.save(output_path, writer=writer, dpi=dpi)

    plt.close(fig)

    print(f"\n✓ Video saved to: {output_path}")
    print("=" * 80)


def main(smoother: bool = False) -> int:
    """Example usage: create filter overlay video."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
    from trodestrack.runtime.offline import rts_smoother

    script_dir = Path(__file__).parent

    # Load data
    print("Loading session data...")
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220324_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220324_imu_info.parquet"),
        imu_mode="3d",
        meters_per_pixel=0.0022,
        verbose=False,
    )
    print(f"✓ Loaded {len(data.t_cam):,} frames and {len(data.t_imu):,} IMU samples\n")

    # Run filter
    print("Running Extended Kalman Filter with 3D IMU...")
    ekf_config = EKFConfig(
        state_mode="2d_cam_3d_imu",  # 10D state: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
        process_noise_pos=0.10,
        process_noise_vel=2.0,
        process_noise_gyro_bias=2e-6,
        process_noise_accel_bias=2e-4,
        measurement_noise_pos=0.02**2,
        use_heading_measurement=True,
        led_distance_tolerance=0.2,
        adaptive_heading_noise=True,
        damping_coeff=0.1,
        num_iter=2,
        led_distance=data.led_distance,
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.99,
    )

    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=data.t_imu,
        U_imu=data.U_imu,
        t_cam=data.t_cam,
        Z_cam_led1=data.Z_cam_led1,
        Z_cam_led2=data.Z_cam_led2,
        mask_cam=data.mask_cam,
    )
    print(f"✓ Filter complete: {len(result.filtered_means):,} timesteps\n")

    if smoother:
        print("Running RTS smoother...")
        result = rts_smoother(
            result, ekf_config, t_imu=data.t_imu, U_imu=data.U_imu, t_cam=data.t_cam
        )
        print(f"✓ Smoothing complete: {len(result.smoothed_means):,} timesteps\n")

    # Create visualization
    video_path = script_dir / "20220324_arthur_02_r1.mp4"
    output_path = script_dir / "arthur_filter_overlay.mp4"

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Load position dataframe for video_frame_ind
    position_df = pd.read_parquet(script_dir / "arthur20220324_position_info.parquet")

    create_filter_overlay_video(
        video_path=str(video_path),
        data=data,
        filter_result=result,
        t_filter=data.t_cam,  # Filter runs at camera timestamps
        position_df=position_df,
        output_path=str(output_path),
        start_time=120.0,  # Start at 2 minutes
        duration=10.0,  # 10 minute clip
        fps=30.0,
        imu_window_s=2.0,  # Show 2s of IMU data
        state_window_s=5.0,  # Show 5s of state estimates
        led_marker_size=10.0,
        trajectory_length=90,  # 3 seconds of trajectory at 30fps
        dpi=100,
    )

    print(f"\n✓ Complete! View output: {output_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize EKF results with uncertainty overlays.")
    p.add_argument(
        "--smooth",
        action="store_true",
        help="Run RTS smoother (if available) before visualization.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit(main(args.smooth))
