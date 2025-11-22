"""Create simple video with camera tracking + raw IMU traces only.

Shows:
- Video with LED1 (back/red) and LED2 (front/yellow) positions
- Raw 3-axis gyroscope traces
- Raw 3-axis accelerometer traces

No filter estimates, no heading, no velocity, no uncertainty.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


# SpikeGadgets hardware specs
GYRO_SCALE = 0.061  # deg/s per LSB
ACCEL_SCALE = 0.000061  # g per LSB
GRAVITY = 9.80665  # m/s²


def get_video_info(video_path: str) -> dict:
    """Get video file metadata."""
    cap = cv2.VideoCapture(video_path)
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        return info
    finally:
        cap.release()


def load_imu_data(imu_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and process IMU data.

    Returns
    -------
    t_imu : np.ndarray
        Timestamps in seconds (relative)
    gyro_deg : np.ndarray
        Gyroscope data in deg/s [N x 3]
    accel_mps2 : np.ndarray
        Accelerometer data in m/s² [N x 3]
    """
    print(f"Loading IMU data from {imu_file.name}...")
    imu_df = pd.read_parquet(imu_file)
    print(f"  Raw samples: {len(imu_df):,}")

    # Remove sample-and-hold repeats
    gyro_z_raw = imu_df["Headstage_GyroZ"].values
    changes = np.where(np.diff(gyro_z_raw) != 0)[0] + 1
    unique_indices = np.concatenate([[0], changes])
    imu_unique = imu_df.iloc[unique_indices]
    print(
        f"  Unique samples: {len(imu_unique):,} (removed {len(imu_df) / len(imu_unique):.1f}x repeats)"
    )

    # Extract timestamps (convert to relative time in seconds)
    t_imu_unix = imu_unique.index.values.astype(np.int64)
    t_imu = (t_imu_unix - t_imu_unix[0]) / 1e9  # nanoseconds to seconds

    # Convert gyroscope (raw -> deg/s)
    gyro_deg = np.column_stack(
        [
            imu_unique["Headstage_GyroX"].values * GYRO_SCALE,
            imu_unique["Headstage_GyroY"].values * GYRO_SCALE,
            imu_unique["Headstage_GyroZ"].values * GYRO_SCALE,
        ]
    )

    # Convert accelerometer (raw -> g -> m/s²)
    accel_mps2 = np.column_stack(
        [
            imu_unique["Headstage_AccelX"].values * ACCEL_SCALE * GRAVITY,
            imu_unique["Headstage_AccelY"].values * ACCEL_SCALE * GRAVITY,
            imu_unique["Headstage_AccelZ"].values * ACCEL_SCALE * GRAVITY,
        ]
    )

    print(f"  Duration: {t_imu[-1]:.1f}s ({t_imu[-1] / 60:.1f} min)")
    print(f"  Sampling rate: ~{len(t_imu) / t_imu[-1]:.1f} Hz")

    return t_imu, gyro_deg, accel_mps2


def setup_figure() -> tuple[plt.Figure, dict]:
    """Create 3-panel figure layout: video (left), gyro (top-right), accel (bottom-right)."""
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
    ax_gyro.set_ylabel("Angular Velocity (deg/s)", fontsize=10)
    ax_gyro.set_xlabel("Time (s)", fontsize=10)
    ax_gyro.grid(True, alpha=0.3)
    ax_gyro.set_title("Gyroscope (3-axis)", fontsize=12, fontweight="bold")
    ax_gyro.tick_params(labelsize=9)

    # Accelerometer bottom right
    ax_accel = fig.add_subplot(gs[1, 1])
    ax_accel.set_ylabel("Acceleration (m/s²)", fontsize=10)
    ax_accel.set_xlabel("Time (s)", fontsize=10)
    ax_accel.grid(True, alpha=0.3)
    ax_accel.set_title("Accelerometer (3-axis)", fontsize=12, fontweight="bold")
    ax_accel.tick_params(labelsize=9)

    return fig, {
        "video": ax_video,
        "gyro": ax_gyro,
        "accel": ax_accel,
    }


def create_camera_imu_video(
    video_path: Path,
    position_file: Path,
    imu_file: Path,
    output_path: Path,
    led_radius: int = 8,
    trajectory_length: int = 90,
    imu_window_s: float = 2.0,
    dpi: int = 100,
) -> None:
    """Create video with camera positions and raw IMU traces.

    Parameters
    ----------
    video_path : Path
        Path to input video
    position_file : Path
        Path to position parquet file
    imu_file : Path
        Path to IMU parquet file
    output_path : Path
        Path for output video
    led_radius : int
        Radius of LED circles in pixels
    trajectory_length : int
        Number of frames to show in trajectory trail
    imu_window_s : float
        Time window for IMU plots (seconds)
    dpi : int
        Output resolution
    """
    print("=" * 80)
    print("Creating Camera + IMU Video (Simple)")
    print("=" * 80)

    # Load position data
    print(f"\nLoading position data from {position_file.name}...")
    pos_df = pd.read_parquet(position_file)
    led1_pixels = pos_df[["xloc", "yloc"]].values
    led2_pixels = pos_df[["xloc2", "yloc2"]].values
    pos_timestamps = (
        pos_df.index.values - pos_df.index.values[0]
    ) / 1e9  # ns to seconds
    print(f"  Loaded {len(pos_df):,} frames")

    # Load IMU data
    t_imu, gyro_deg, accel_mps2 = load_imu_data(imu_file)

    # Get video info
    print(f"\nOpening video {video_path.name}...")
    video_info = get_video_info(str(video_path))
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Duration: {video_info['duration_s']:.1f}s")

    # Align video frames with position data
    video_frame_inds = pos_df["video_frame_ind"].values
    n_frames = min(len(video_frame_inds), video_info["frame_count"])
    print(f"\nProcessing {n_frames:,} frames...")

    # Setup matplotlib figure
    fig, axes = setup_figure()

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    # Create temporary directory for frames
    tmpdir = tempfile.mkdtemp(prefix="camera_imu_")
    frame_pattern = str(Path(tmpdir) / "frame_%06d.png")
    print(f"  Rendering to {tmpdir}")

    try:
        for i in range(n_frames):
            # Read video frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_frame_inds[i]))
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time = pos_timestamps[i]

            # Clear axes
            axes["video"].clear()
            axes["gyro"].clear()
            axes["accel"].clear()

            # === VIDEO PANEL ===
            axes["video"].imshow(frame_rgb)
            axes["video"].set_aspect("equal")
            axes["video"].axis("off")
            axes["video"].set_title(
                "Camera Tracking", fontsize=14, fontweight="bold", pad=10
            )

            # Draw trajectory trails (fading)
            start_idx = max(0, i - trajectory_length)
            for j in range(start_idx, i):
                alpha = 0.3 + 0.7 * (j - start_idx) / max(1, trajectory_length)
                # LED1 trajectory (red)
                axes["video"].plot(
                    led1_pixels[j : j + 2, 0],
                    led1_pixels[j : j + 2, 1],
                    "r-",
                    linewidth=1.5,
                    alpha=alpha,
                )
                # LED2 trajectory (yellow)
                axes["video"].plot(
                    led2_pixels[j : j + 2, 0],
                    led2_pixels[j : j + 2, 1],
                    color="#FFFF00",
                    linewidth=1.5,
                    alpha=alpha,
                )

            # Draw current LED positions
            axes["video"].plot(
                led1_pixels[i, 0],
                led1_pixels[i, 1],
                "o",
                color="red",
                markersize=led_radius,
                markeredgewidth=2,
                markeredgecolor="white",
                label="LED1 (back)",
            )
            axes["video"].plot(
                led2_pixels[i, 0],
                led2_pixels[i, 1],
                "o",
                color="#FFFF00",
                markersize=led_radius,
                markeredgewidth=2,
                markeredgecolor="white",
                label="LED2 (front)",
            )

            # Add timestamp and legend
            axes["video"].text(
                0.02,
                0.98,
                f"t = {current_time:.2f}s",
                transform=axes["video"].transAxes,
                fontsize=12,
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )
            axes["video"].legend(loc="upper right", fontsize=10, framealpha=0.9)

            # === IMU PANELS ===
            half_window = imu_window_s / 2.0
            mask = (t_imu >= current_time - half_window) & (
                t_imu <= current_time + half_window
            )
            t_window = t_imu[mask]
            gyro_window = gyro_deg[mask]
            accel_window = accel_mps2[mask]

            # Gyroscope
            if len(t_window) > 0:
                axes["gyro"].plot(
                    t_window,
                    gyro_window[:, 0],
                    "-",
                    linewidth=1.5,
                    label="X",
                    alpha=0.8,
                    color="#1f77b4",
                )
                axes["gyro"].plot(
                    t_window,
                    gyro_window[:, 1],
                    "-",
                    linewidth=1.5,
                    label="Y",
                    alpha=0.8,
                    color="#ff7f0e",
                )
                axes["gyro"].plot(
                    t_window,
                    gyro_window[:, 2],
                    "-",
                    linewidth=1.5,
                    label="Z",
                    alpha=0.8,
                    color="#2ca02c",
                )
                axes["gyro"].axvline(
                    current_time,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.5,
                )

            axes["gyro"].set_xlim(
                current_time - half_window, current_time + half_window
            )
            axes["gyro"].set_ylim(-200, 200)
            axes["gyro"].set_ylabel("Angular Velocity (deg/s)", fontsize=10)
            axes["gyro"].set_xlabel("Time (s)", fontsize=10)
            axes["gyro"].grid(True, alpha=0.3)
            axes["gyro"].set_title("Gyroscope (3-axis)", fontsize=12, fontweight="bold")
            axes["gyro"].legend(loc="upper right", fontsize=9, framealpha=0.9)
            axes["gyro"].tick_params(labelsize=9)

            # Accelerometer
            if len(t_window) > 0:
                axes["accel"].plot(
                    t_window,
                    accel_window[:, 0],
                    "-",
                    linewidth=1.5,
                    label="X",
                    alpha=0.8,
                    color="#1f77b4",
                )
                axes["accel"].plot(
                    t_window,
                    accel_window[:, 1],
                    "-",
                    linewidth=1.5,
                    label="Y",
                    alpha=0.8,
                    color="#ff7f0e",
                )
                axes["accel"].plot(
                    t_window,
                    accel_window[:, 2],
                    "-",
                    linewidth=1.5,
                    label="Z",
                    alpha=0.8,
                    color="#2ca02c",
                )
                axes["accel"].axvline(
                    current_time,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.5,
                )

            axes["accel"].set_xlim(
                current_time - half_window, current_time + half_window
            )
            axes["accel"].set_ylim(-15, 15)
            axes["accel"].set_ylabel("Acceleration (m/s²)", fontsize=10)
            axes["accel"].set_xlabel("Time (s)", fontsize=10)
            axes["accel"].grid(True, alpha=0.3)
            axes["accel"].set_title(
                "Accelerometer (3-axis)", fontsize=12, fontweight="bold"
            )
            axes["accel"].legend(loc="upper right", fontsize=9, framealpha=0.9)
            axes["accel"].tick_params(labelsize=9)

            # Save frame (pad to ensure dimensions divisible by 2 for H.264)
            out_frame_path = frame_pattern % (i + 1)
            fig.savefig(out_frame_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

            if (i + 1) % 100 == 0:
                print(
                    f"    {i + 1:,} / {n_frames:,} frames ({100 * (i + 1) / n_frames:.1f}%)"
                )

        cap.release()
        plt.close(fig)

        # Stitch frames with ffmpeg
        print("\n  Stitching with ffmpeg...")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(int(video_info["fps"])),
            "-i",
            frame_pattern,
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure dimensions divisible by 2
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(int(video_info["fps"])),
            "-b:v",
            "4000k",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

        print(f"\n✓ Video created: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("=" * 80)


def main():
    """Create camera + IMU video for trimmed session."""
    # Paths
    script_dir = Path(__file__).parent
    video_dir = script_dir.parent / "videos"
    data_dir = script_dir.parent.parent.parent / "data"

    video_path = video_dir / "20220324_arthur_02_r1_trimmed.mp4"
    position_file = data_dir / "arthur20220324_position_info.parquet"
    imu_file = data_dir / "arthur20220324_imu_info.parquet"
    output_path = video_dir / "20220324_arthur_02_r1_trimmed_camera_imu.mp4"

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    if not position_file.exists():
        print(f"Error: Position file not found: {position_file}")
        return 1

    if not imu_file.exists():
        print(f"Error: IMU file not found: {imu_file}")
        return 1

    create_camera_imu_video(
        video_path=video_path,
        position_file=position_file,
        imu_file=imu_file,
        output_path=output_path,
        led_radius=8,
        trajectory_length=90,  # ~3 seconds at 30fps
        imu_window_s=2.0,  # 2-second IMU window
        dpi=100,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
