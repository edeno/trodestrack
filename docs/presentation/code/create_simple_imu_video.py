"""Create video with camera tracking and raw IMU - simplified version using existing code."""

import sys
from pathlib import Path

# Add data directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))

import numpy as np
import pandas as pd
from load_arthur_session import load_arthur_session
from visualize_filter_overlay import (
    OverlayStyle,
    RenderConfig,
    create_filter_overlay_video,
)


def create_simple_camera_imu_video():
    """Create video showing raw camera tracking + IMU for trimmed video."""

    # Paths
    script_dir = Path(__file__).parent
    video_dir = script_dir.parent / "videos"
    data_dir = script_dir.parent.parent.parent / "data"

    video_path = str(video_dir / "20220324_arthur_02_r1_trimmed.mp4")
    position_file = str(data_dir / "arthur20220324_position_info.parquet")
    imu_file = str(data_dir / "arthur20220324_imu_info.parquet")
    output_path = str(video_dir / "20220324_arthur_02_r1_trimmed_imu.mp4")

    print("Loading session data...")
    data = load_arthur_session(
        position_file=position_file,
        imu_file=imu_file,
        imu_mode="3d",
        meters_per_pixel=0.0022,
        verbose=True,
    )

    print("\nCreating simple camera + IMU video for trimmed clip...")

    # Load position dataframe for video frame indices
    position_df = pd.read_parquet(position_file)

    # Create a minimal "filter result" that just passes through camera positions
    # This is a hack to reuse the existing visualization code
    class SimpleResult:
        def __init__(self, data, position_df):
            # Use camera timestamps
            n = len(data.t_cam)

            # State: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
            # We'll just use camera positions and zeros for the rest
            self.filtered_means = np.zeros((n, 10))
            self.filtered_means[:, :2] = (
                data.Z_cam_led1
            )  # Use LED1 position as "filter" output

            # Compute simple velocities from positions
            dt = np.diff(data.t_cam, prepend=data.t_cam[0])
            dt = np.where(dt > 0, dt, 1e-6)
            vel = (
                np.diff(data.Z_cam_led1, axis=0, prepend=data.Z_cam_led1[0:1])
                / dt[:, None]
            )
            self.filtered_means[:, 2:4] = vel  # vx, vy

            # Compute heading from LED1 to LED2
            diff = data.Z_cam_led2 - data.Z_cam_led1
            self.filtered_means[:, 5] = np.arctan2(diff[:, 1], diff[:, 0])

            # Minimal covariances (small, so uncertainty plots won't dominate)
            self.filtered_covariances = np.tile(np.eye(10) * 1e-6, (n, 1, 1))

    result = SimpleResult(data, position_df)

    # Configuration
    style = OverlayStyle(
        led_marker_size=10.0,
        trajectory_length=90,  # 3 seconds at 30fps
        heading_arrow_px=40.0,
        heading_flip=True,
        show_uncertainty_circle=False,
        invert_yaxis=False,
    )

    config = RenderConfig(
        start_time=0.0,
        duration=90.0,  # Full trimmed video
        fps=30.0,
        imu_window_s=2.0,
        state_window_s=5.0,
        dpi=100,
        render_mode="parallel_png",  # Faster for long videos
        max_workers=4,
        ffmpeg_threads=4,
        bitrate_kbps=4000,
        style=style,
    )

    create_filter_overlay_video(
        video_path=video_path,
        data=data,
        filter_result=result,
        t_filter=data.t_cam,
        position_df=position_df,
        output_path=output_path,
        config=config,
    )

    print(f"\n✓ Complete! Output: {output_path}")


if __name__ == "__main__":
    create_simple_camera_imu_video()
