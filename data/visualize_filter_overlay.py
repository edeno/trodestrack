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
import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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
    render_mode: str = "single_process",  # "single_process" or "parallel_png"
    max_workers: int | None = None,  # for parallel mode
    ffmpeg_threads: int | None = None,  # for ffmpeg encoding
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

    # Compute position uncertainty (Euclidean std, including covariance)
    pos_std = np.sqrt(np.trace(P_filter[:, :2, :2], axis1=1, axis2=2))

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
    print(f"\nGenerating {n_frames} frames...")
    frame_times = start_time + np.arange(n_frames) / fps
    frame_to_filter_idx = np.clip(
        np.searchsorted(t_filter, frame_times, side="left"), 0, len(t_filter) - 1
    )
    frame_to_cam_idx = np.clip(
        np.searchsorted(data.t_cam, frame_times, side="left"), 0, len(data.t_cam) - 1
    )
    pos_vid_inds = position_df["video_frame_ind"].to_numpy()
    frame_to_video_ind = pos_vid_inds[frame_to_filter_idx]

    # Branch early if using process-parallel renderer
    if render_mode == "parallel_png":
        _render_parallel_png(
            video_path=video_path,
            output_path=output_path,
            n_frames=n_frames,
            fps=fps,
            dpi=dpi,
            max_workers=max_workers,
            frame_times=frame_times,
            frame_to_filter_idx=frame_to_filter_idx,
            frame_to_cam_idx=frame_to_cam_idx,
            frame_to_video_ind=frame_to_video_ind,
            video_info=video_info,
            data=data,
            pos_filter_pixels=pos_filter_pixels,
            heading_filter=heading_filter,
            led1_pixels=led1_pixels,
            led2_pixels=led2_pixels,
            vel_mag=vel_mag,
            pos_std=pos_std,
            imu_window_s=imu_window_s,
            state_window_s=state_window_s,
            trajectory_length=trajectory_length,
            led_marker_size=led_marker_size,
        )
        print(f"\n✓ Video saved to: {output_path}")
        return  # done

    # --- Parallel frame caching ---
    # Decode all required frames concurrently to minimize I/O latency.
    print(f"\nCaching {n_frames} video frames for animation...")
    t0 = time.time()  # for progress timing

    # Threaded frame loading for animation mode
    def _read_one(video_idx: int):
        return load_video_frame(video_path, int(video_idx))

    cached_frames = [None] * n_frames
    max_workers = min(8, (os.cpu_count() or 4))  # modest default
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_read_one, int(vf)): i for i, vf in enumerate(frame_to_video_ind)}
        for fut in as_completed(futures):
            i = futures[fut]
            cached_frames[i] = fut.result()

    # Setup figure
    fig, axes = setup_figure()

    # Initialize video frame
    video_frame = axes["video"].imshow(
        np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8),
        animated=True,
    )

    # Initialize trajectory line
    (trajectory_line,) = axes["video"].plot(
        [], [], "c-", linewidth=1.5, alpha=0.6, label="Filter trajectory"
    )
    trajectory_line.set_animated(True)

    # Initialize LED circles
    led1_circle = Circle((0, 0), led_marker_size, color="red", alpha=0.5, label="LED1 (back)")
    led2_circle = Circle((0, 0), led_marker_size, color="yellow", alpha=0.5, label="LED2 (front)")
    axes["video"].add_patch(led1_circle)
    axes["video"].add_patch(led2_circle)
    led1_circle.set_animated(True)
    led2_circle.set_animated(True)

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
    filter_circle.set_animated(True)

    # Initialize heading indicator as a simple line (blit-friendly)
    (heading_line_video,) = axes["video"].plot(
        [], [], color="cyan", linewidth=2, alpha=0.7, label="Heading"
    )
    heading_line_video.set_animated(True)

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
    t0 = time.time()  # for progress timing

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=True
    )

    # Save video
    # writer = FFMpegWriter(fps=fps, bitrate=8000)
    # anim.save(output_path, writer=writer, dpi=dpi)
    # Enable multi-threaded ffmpeg encode
    if ffmpeg_threads is None:
        ffmpeg_threads = min(4, os.cpu_count() or 2)
    writer = FFMpegWriter(
        fps=fps,
        bitrate=4000,
        extra_args=["-threads", str(ffmpeg_threads), "-pix_fmt", "yuv420p"],
    )
    anim.save(output_path, writer=writer, dpi=max(72, dpi))

    plt.close(fig)
    time_elapsed = time.time() - t0
    print(f"✓ Video generation complete in {time_elapsed:.1f}s.")
    print(f"\n✓ Video saved to: {output_path}")
    print("=" * 80)


def _render_parallel_png(
    *,
    video_path: str,
    output_path: str,
    n_frames: int,
    fps: float,
    dpi: int,
    max_workers: int | None,
    frame_times: np.ndarray,
    frame_to_filter_idx: np.ndarray,
    frame_to_cam_idx: np.ndarray,
    frame_to_video_ind: np.ndarray,
    video_info: dict,
    data: SessionData,
    pos_filter_pixels: np.ndarray,
    heading_filter: np.ndarray,
    led1_pixels: np.ndarray,
    led2_pixels: np.ndarray,
    vel_mag: np.ndarray,
    pos_std: np.ndarray,
    imu_window_s: float,
    state_window_s: float,
    trajectory_length: int,
    led_marker_size: float,
) -> None:
    """
    Render frames in multiple processes to a temp directory, then stitch with ffmpeg.
    Each worker creates its own figure and calls a local per-frame render (no shared artists).
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH — please install it.")
    # Choose workers
    if max_workers is None:
        max_workers = max(2, (os.cpu_count() or 4) // 2)
    print(f"\n[parallel] Using {max_workers} workers")

    # Temp dir for PNG frames
    tmpdir = tempfile.mkdtemp(prefix="overlay_frames_")
    pat = os.path.join(tmpdir, "frame_%06d.png")
    print(f"[parallel] Writing frames to {tmpdir}")

    # Partition frames into roughly-equal chunks
    chunks = []
    chunk_size = math.ceil(n_frames / max_workers)
    for s in range(0, n_frames, chunk_size):
        e = min(n_frames, s + chunk_size)
        chunks.append((s, e))

    # Render chunks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for s, e in chunks:
            futures.append(
                ex.submit(
                    _render_chunk,
                    video_path,
                    video_info,
                    s,
                    e,
                    fps,
                    dpi,
                    pat,
                    frame_times[s:e],
                    frame_to_filter_idx[s:e],
                    frame_to_cam_idx[s:e],
                    frame_to_video_ind[s:e],
                    # immutable arrays (slices / views) below are serialized once per worker
                    data.t_imu,
                    data.U_imu,
                    data.t_cam,
                    pos_filter_pixels,
                    heading_filter,
                    led1_pixels,
                    led2_pixels,
                    vel_mag,
                    pos_std,
                    imu_window_s,
                    state_window_s,
                    trajectory_length,
                    led_marker_size,
                )
            )
        # wait for completion / raise early on error
        for fut in futures:
            fut.result()

    # Stitch with ffmpeg (multi-threaded)
    _ffmpeg_stitch(pat, output_path, fps, bitrate=4000, threads=max_workers)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


def _render_chunk(
    video_path,
    video_info,
    start_f,
    end_f,
    fps,
    dpi,
    pattern,
    frame_times_chunk,
    fidx_chunk,
    cidx_chunk,
    vidx_chunk,
    t_imu,
    U_imu,
    t_cam,
    pos_filter_pixels,
    heading_filter,
    led1_pixels,
    led2_pixels,
    vel_mag,
    pos_std,
    imu_window_s,
    state_window_s,
    trajectory_length,
    led_marker_size,
):
    """
    Worker: build a fresh figure and render frames [start_f, end_f) into PNGs named by `pattern`.
    """
    # Local imports (worker)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Helper: load one frame (keeps your existing loader)
    def _read_frame(video_idx: int):
        return load_video_frame(video_path, int(video_idx))

    # Create figure/axes and artists fresh in this process
    fig, axes = setup_figure()
    video_frame = axes["video"].imshow(
        np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8)
    )
    (trajectory_line,) = axes["video"].plot(
        [], [], "c-", linewidth=1.5, alpha=0.6, label="Filter trajectory"
    )
    led1_circle = Circle((0, 0), led_marker_size, color="red", alpha=0.5, label="LED1 (back)")
    led2_circle = Circle((0, 0), led_marker_size, color="yellow", alpha=0.5, label="LED2 (front)")
    axes["video"].add_patch(led1_circle)
    axes["video"].add_patch(led2_circle)
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
    (heading_line_video,) = axes["video"].plot(
        [], [], color="cyan", linewidth=2, alpha=0.7, label="Heading"
    )
    axes["video"].legend(loc="upper right", fontsize=9, framealpha=0.9)

    (gyro_x_line,) = axes["gyro"].plot([], [], "r-", linewidth=1.2, label="X", alpha=0.8)
    (gyro_y_line,) = axes["gyro"].plot([], [], "g-", linewidth=1.2, label="Y", alpha=0.8)
    (gyro_z_line,) = axes["gyro"].plot([], [], "b-", linewidth=1.2, label="Z", alpha=0.8)
    gyro_marker = axes["gyro"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    (accel_x_line,) = axes["accel"].plot([], [], "r-", linewidth=1.2, label="X", alpha=0.8)
    (accel_y_line,) = axes["accel"].plot([], [], "g-", linewidth=1.2, label="Y", alpha=0.8)
    (accel_z_line,) = axes["accel"].plot([], [], "b-", linewidth=1.2, label="Z", alpha=0.8)
    accel_marker = axes["accel"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    (vel_line,) = axes["pos_vel"].plot([], [], "b-", linewidth=1.5, label="Speed")
    vel_marker = axes["pos_vel"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    (heading_line_ax,) = axes["heading"].plot([], [], "m-", linewidth=1.5, label="Heading")
    heading_marker = axes["heading"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )
    (unc_line,) = axes["uncertainty"].plot([], [], "orange", linewidth=1.5, label="Position σ")
    unc_marker = axes["uncertainty"].axvline(
        0, color="black", linestyle="--", linewidth=1.5, alpha=0.5
    )

    time_text = axes["video"].text(
        0.02,
        0.98,
        "",
        transform=axes["video"].transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Local IMU conversions (reuse your logic)
    gyro_x = U_imu[:, 0] * 180 / np.pi
    gyro_y = U_imu[:, 1] * 180 / np.pi
    gyro_z = U_imu[:, 2] * 180 / np.pi
    accel_x = U_imu[:, 3]
    accel_y = U_imu[:, 4]
    accel_z = U_imu[:, 5]

    # Render each frame in this chunk
    for i in range(start_f, end_f):
        local_i = i - start_f
        current_time = frame_times_chunk[local_i]
        filter_idx = fidx_chunk[local_i]
        cam_idx = cidx_chunk[local_i]
        video_idx = int(vidx_chunk[local_i])

        # Video image
        frame = _read_frame(video_idx)
        if frame is not None:
            video_frame.set_data(frame)

        # Position + heading
        current_pos = pos_filter_pixels[filter_idx]
        current_heading = heading_filter[filter_idx]
        traj_start = max(0, filter_idx - trajectory_length)
        traj_x = pos_filter_pixels[traj_start : filter_idx + 1, 0]
        traj_y = pos_filter_pixels[traj_start : filter_idx + 1, 1]
        trajectory_line.set_data(traj_x, traj_y)
        filter_circle.center = (current_pos[0], current_pos[1])
        led1_pos = led1_pixels[cam_idx]
        led2_pos = led2_pixels[cam_idx]
        led1_circle.center = (led1_pos[0], led1_pos[1])
        led2_circle.center = (led2_pos[0], led2_pos[1])
        # Heading ray (keep 180° correction)
        arrow_length = 40
        arrow_heading = current_heading + np.pi
        dx = arrow_length * np.cos(arrow_heading)
        dy = arrow_length * np.sin(arrow_heading)
        heading_line_video.set_data(
            [current_pos[0], current_pos[0] + dx], [current_pos[1], current_pos[1] + dy]
        )

        # IMU windows

        def _win(ts, arr, win, current_time=current_time):
            half = win / 2.0
            m = (ts >= current_time - half) & (ts <= current_time + half)
            return ts[m], arr[m]

        t_gyro_x, gx = _win(t_imu, gyro_x, imu_window_s)
        t_gyro_y, gy = _win(t_imu, gyro_y, imu_window_s)
        t_gyro_z, gz = _win(t_imu, gyro_z, imu_window_s)
        t_accel_x, ax = _win(t_imu, accel_x, imu_window_s)
        t_accel_y, ay = _win(t_imu, accel_y, imu_window_s)
        t_accel_z, az = _win(t_imu, accel_z, imu_window_s)
        # Update IMU plots
        gyro_x_line.set_data(t_gyro_x, gx)
        gyro_y_line.set_data(t_gyro_y, gy)
        gyro_z_line.set_data(t_gyro_z, gz)
        axes["gyro"].set_xlim(current_time - imu_window_s / 2, current_time + imu_window_s / 2)
        axes["gyro"].set_ylim(-200, 200)
        gyro_marker.set_xdata([current_time, current_time])
        accel_x_line.set_data(t_accel_x, ax)
        accel_y_line.set_data(t_accel_y, ay)
        accel_z_line.set_data(t_accel_z, az)
        axes["accel"].set_xlim(current_time - imu_window_s / 2, current_time + imu_window_s / 2)
        axes["accel"].set_ylim(-15, 15)
        accel_marker.set_xdata([current_time, current_time])

        # State windows
        # vel_mag in cm/s, heading in deg, pos std in cm
        def _win2(ts, arr, win, current_time=current_time):
            half = win / 2.0
            m = (ts >= current_time - half) & (ts <= current_time + half)
            return ts[m], arr[m]

        t_state, vel_w = _win2(t_cam, vel_mag * 100, state_window_s)
        _, heading_w = _win2(t_cam, heading_filter * 180 / np.pi, state_window_s)
        _, unc_w = _win2(t_cam, pos_std * 100, state_window_s)

        vel_line.set_data(t_state, vel_w)
        axes["pos_vel"].set_xlim(
            current_time - state_window_s / 2, current_time + state_window_s / 2
        )
        axes["pos_vel"].set_ylim(0, max(100, (np.max(vel_w) * 1.2) if len(vel_w) else 100))
        vel_marker.set_xdata([current_time, current_time])

        heading_line_ax.set_data(t_state, heading_w)
        axes["heading"].set_xlim(
            current_time - state_window_s / 2, current_time + state_window_s / 2
        )
        axes["heading"].set_ylim(-180, 180)
        heading_marker.set_xdata([current_time, current_time])

        unc_line.set_data(t_state, unc_w)
        axes["uncertainty"].set_xlim(
            current_time - state_window_s / 2, current_time + state_window_s / 2
        )
        axes["uncertainty"].set_ylim(0, max(2, (np.max(unc_w) * 1.2) if len(unc_w) else 2))
        unc_marker.set_xdata([current_time, current_time])

        # Time label
        time_text.set_text(f"t = {current_time:.2f} s")

        # Save PNG for this frame index (global numbering)
        plt.tight_layout()
        out_path = pattern % (i + 1)  # 1-based for ffmpeg %06d
        fig.canvas.draw()
        fig.savefig(out_path, dpi=max(72, dpi))

    plt.close(fig)


def _ffmpeg_stitch(
    pattern: str, output_path: str, fps: float, bitrate: int = 4000, threads: int = 4
):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(int(fps)),
        "-b:v",
        str(bitrate * 1000),
        "-threads",
        str(max(1, threads)),
        output_path,
    ]
    print("[ffmpeg]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(smoother: bool = False, render_mode: str = "single_process") -> int:
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
    t0 = time.time()
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
    time_elapsed = time.time() - t0
    print(f"✓ Filter complete in {time_elapsed:.1f}s: {len(result.filtered_means):,} timesteps\n")

    if smoother:
        print("Running RTS smoother...")
        t0 = time.time()
        result = rts_smoother(
            result, ekf_config, t_imu=data.t_imu, U_imu=data.U_imu, t_cam=data.t_cam
        )
        time_elapsed = time.time() - t0
        print(
            f"✓ Smoothing complete in {time_elapsed:.1f}s: {len(result.smoothed_means):,} timesteps\n"
        )

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
        ffmpeg_threads=4,  # Set encoding threads separately
        render_mode=render_mode,
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
    p.add_argument(
        "--render_mode",
        choices=["single_process", "parallel_png"],
        default="single_process",
        help="Choose the rendering mode for the visualization.",
    )
    return p.parse_args()


if __name__ == "__main__":
    import sys

    args = _parse_args()
    sys.exit(main(args.smooth, args.render_mode))
