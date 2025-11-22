"""Create video with camera tracking and raw IMU - using proven code from visualize_filter_overlay.py.

This version properly adapts the working visualization code, removing only the
filter-specific panels (velocity, heading, uncertainty) while keeping the
working IMU trace display.
"""

import sys
from pathlib import Path

# Add data directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))

import matplotlib

matplotlib.use("Agg")
import math
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_arthur_session import (
    convert_meters_to_pixels,
    get_video_info,
    load_arthur_session,
    load_video_frame,
)
from matplotlib.patches import Circle


def setup_figure():
    """Create 3-panel figure: video (left), gyro (top-right), accel (bottom-right)."""
    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2, hspace=0.3, wspace=0.3, left=0.05, right=0.97, top=0.95, bottom=0.05
    )

    # Video takes left side (both rows)
    ax_video = fig.add_subplot(gs[:, 0])
    ax_video.set_aspect("equal")
    ax_video.axis("off")
    ax_video.set_title("Camera Tracking", fontsize=14, fontweight="bold", pad=10)

    # Gyroscope top right
    ax_gyro = fig.add_subplot(gs[0, 1])
    ax_gyro.set_ylabel("Gyro (deg/s)", fontsize=10)
    ax_gyro.grid(True, alpha=0.3)
    ax_gyro.set_title("Gyroscope (3-axis)", fontsize=12, fontweight="bold")
    ax_gyro.tick_params(labelsize=9)

    # Accelerometer bottom right
    ax_accel = fig.add_subplot(gs[1, 1])
    ax_accel.set_ylabel("Accel (m/s²)", fontsize=10)
    ax_accel.set_xlabel("Time (s)", fontsize=10)
    ax_accel.grid(True, alpha=0.3)
    ax_accel.set_title("Accelerometer (3-axis)", fontsize=12, fontweight="bold")
    ax_accel.tick_params(labelsize=9)

    return fig, {"video": ax_video, "gyro": ax_gyro, "accel": ax_accel}


def render_chunk(
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
    pos_filter_px,
    led1_px,
    led2_px,
    imu_window_s,
    trajectory_length,
    led_marker_size,
):
    """Render a chunk of frames in parallel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = setup_figure()

    # Video image placeholder
    video_frame = axes["video"].imshow(
        np.zeros((video_info["height"], video_info["width"], 3), dtype=np.uint8)
    )

    # LED markers and trajectory
    (trajectory_line,) = axes["video"].plot(
        [], [], "c-", linewidth=1.5, alpha=0.6, label="Trajectory"
    )
    led1_circle = Circle((0, 0), led_marker_size, color="red", alpha=0.7, label="LED1 (back)")
    led2_circle = Circle((0, 0), led_marker_size, color="yellow", alpha=0.7, label="LED2 (front)")
    axes["video"].add_patch(led1_circle)
    axes["video"].add_patch(led2_circle)
    axes["video"].legend(loc="upper right", fontsize=10, framealpha=0.9)

    # IMU plot lines
    (gyro_x_line,) = axes["gyro"].plot([], [], "-", linewidth=1.5, label="X", alpha=0.8)
    (gyro_y_line,) = axes["gyro"].plot([], [], "-", linewidth=1.5, label="Y", alpha=0.8)
    (gyro_z_line,) = axes["gyro"].plot([], [], "-", linewidth=1.5, label="Z", alpha=0.8)
    gyro_marker = axes["gyro"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    axes["gyro"].legend(loc="upper right", fontsize=9)

    (accel_x_line,) = axes["accel"].plot([], [], "-", linewidth=1.5, label="X", alpha=0.8)
    (accel_y_line,) = axes["accel"].plot([], [], "-", linewidth=1.5, label="Y", alpha=0.8)
    (accel_z_line,) = axes["accel"].plot([], [], "-", linewidth=1.5, label="Z", alpha=0.8)
    accel_marker = axes["accel"].axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    axes["accel"].legend(loc="upper right", fontsize=9)

    time_text = axes["video"].text(
        0.02,
        0.98,
        "",
        transform=axes["video"].transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Convert IMU to deg/s for gyro (already in rad/s, need deg/s)
    gyro = np.degrees(U_imu[:, :3])
    accel = U_imu[:, 3:6]

    for i in range(start_f, end_f):
        j = i - start_f
        t = frame_times_chunk[j]
        fi = fidx_chunk[j]
        ci = cidx_chunk[j]
        vi = int(vidx_chunk[j])

        # Load video frame
        frame = load_video_frame(video_path, vi)
        if frame is not None:
            video_frame.set_data(frame)

        # Update LED positions
        led1_circle.center = tuple(led1_px[ci])
        led2_circle.center = tuple(led2_px[ci])

        # Trajectory
        t0 = max(0, fi - trajectory_length)
        traj = pos_filter_px[t0 : fi + 1]
        trajectory_line.set_data(traj[:, 0], traj[:, 1])

        # Update timestamp
        time_text.set_text(f"t = {t:.2f}s")

        # IMU window
        half = imu_window_s / 2.0
        m = (t_imu >= t - half) & (t_imu <= t + half)
        tx = t_imu[m]
        gx, gy, gz = gyro[m].T if m.any() else (np.array([]),) * 3
        ax, ay, az = accel[m].T if m.any() else (np.array([]),) * 3

        # Update gyro
        gyro_x_line.set_data(tx, gx)
        gyro_y_line.set_data(tx, gy)
        gyro_z_line.set_data(tx, gz)
        axes["gyro"].set_xlim(t - half, t + half)
        axes["gyro"].set_ylim(-200, 200)
        gyro_marker.set_xdata([t, t])

        # Update accel
        accel_x_line.set_data(tx, ax)
        accel_y_line.set_data(tx, ay)
        accel_z_line.set_data(tx, az)
        axes["accel"].set_xlim(t - half, t + half)
        axes["accel"].set_ylim(-15, 15)
        accel_marker.set_xdata([t, t])

        # Save frame
        out_path = pattern % (i + 1)
        fig.savefig(out_path, dpi=max(72, dpi))

    plt.close(fig)


def main():
    """Create simple camera + IMU video."""
    script_dir = Path(__file__).parent
    video_dir = script_dir.parent / "videos"
    data_dir = script_dir.parent.parent.parent / "data"

    video_path = str(video_dir / "20220324_arthur_02_r1_trimmed.mp4")
    position_file = str(data_dir / "arthur20220324_position_info.parquet")
    imu_file = str(data_dir / "arthur20220324_imu_info.parquet")
    output_path = str(video_dir / "20220324_arthur_02_r1_trimmed_camera_imu.mp4")

    print("=" * 80)
    print("Creating Camera + IMU Video")
    print("=" * 80)

    # Load session data using the proven loader
    print("\nLoading session data...")
    data = load_arthur_session(
        position_file=position_file,
        imu_file=imu_file,
        imu_mode="3d",
        meters_per_pixel=0.0022,
        verbose=True,
    )

    # Load position dataframe
    position_df = pd.read_parquet(position_file)

    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        raise RuntimeError("Could not read video info")

    print(f"\nVideo: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f} fps")
    print(f"Duration: {video_info['duration_s']:.1f}s")

    # Configuration
    start_time = 0.0
    duration = 90.0
    fps = 30.0
    imu_window_s = 2.0
    trajectory_length = 90
    led_marker_size = 10.0
    dpi = 100

    # Frame indexing (copied from visualize_filter_overlay.py)
    end_time = start_time + duration
    max_time = min(data.t_cam[-1], video_info["duration_s"])
    if end_time > max_time:
        end_time = max_time
        duration = max(0.0, end_time - start_time)

    n_frames = int(duration * fps)
    frame_times = start_time + (np.arange(n_frames) / fps)

    # Use camera positions directly (no filter)
    pos_filter_px = convert_meters_to_pixels(data.Z_cam_led1, data.meters_per_pixel)
    led1_px = convert_meters_to_pixels(data.Z_cam_led1, data.meters_per_pixel)
    led2_px = convert_meters_to_pixels(data.Z_cam_led2, data.meters_per_pixel)

    # Nearest indices
    fidx = np.searchsorted(data.t_cam, frame_times)
    fidx = np.clip(fidx, 0, len(data.t_cam) - 1)
    cidx = fidx.copy()

    pos_vid_inds = position_df["video_frame_ind"].to_numpy()
    vidx = pos_vid_inds[fidx]

    print(f"\nRendering {n_frames:,} frames using parallel PNG mode...")

    # Parallel rendering
    max_workers = 4
    tmpdir = tempfile.mkdtemp(prefix="camera_imu_")
    pattern = str(Path(tmpdir) / "frame_%06d.png")

    print(f"  Temp directory: {tmpdir}")

    try:
        # Partition frames
        chunk_size = math.ceil(n_frames / max_workers)
        chunks = [(s, min(n_frames, s + chunk_size)) for s in range(0, n_frames, chunk_size)]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    render_chunk,
                    video_path,
                    video_info,
                    s,
                    e,
                    fps,
                    dpi,
                    pattern,
                    frame_times[s:e],
                    fidx[s:e],
                    cidx[s:e],
                    vidx[s:e],
                    data.t_imu,
                    data.U_imu,
                    data.t_cam,
                    pos_filter_px,
                    led1_px,
                    led2_px,
                    imu_window_s,
                    trajectory_length,
                    led_marker_size,
                )
                for (s, e) in chunks
            ]
            for fut in futs:
                fut.result()

        # Stitch with ffmpeg
        print("\n  Stitching frames with ffmpeg...")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-framerate",
            str(int(fps)),
            "-i",
            pattern,
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(int(fps)),
            "-b:v",
            "4000k",
            output_path,
        ]
        subprocess.run(cmd, check=True)

        print(f"\n✓ Video created: {output_path}")
        file_size = Path(output_path).stat().st_size / 1024**2
        print(f"  Size: {file_size:.1f} MB")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())

    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
