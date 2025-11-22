"""Create video showing EKF + RTS smoother with comprehensive diagnostics.

Shows:
- Video overlay with smoother estimates
- Gyroscope (3-axis)
- Accelerometer (3-axis)
- Speed with ±2σ confidence intervals
- Heading with ±2σ confidence intervals
- Position uncertainty

Note: This uses the existing visualize_filter_overlay.py infrastructure,
which already supports both EKF (filtered_means) and RTS (smoothed_means).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.runtime.offline import rts_smoother

# Add parent directories to path for imports
script_dir = Path(__file__).parent.resolve()
repo_root = (
    script_dir.parent.parent.parent
)  # code -> presentation -> docs -> trodestrack
data_dir = repo_root / "data"
sys.path.insert(0, str(data_dir))
sys.path.insert(0, str(repo_root / "src"))


from load_arthur_session import get_video_info, load_arthur_session
from visualize_filter_overlay import (
    OverlayStyle,
    RenderConfig,
    create_filter_overlay_video,
)


def main():
    """Generate EKF + RTS smoother video."""

    # Paths
    video_path = script_dir.parent / "videos" / "20220324_arthur_02_r1_trimmed.mp4"
    position_file = data_dir / "arthur20220324_position_info.parquet"
    imu_file = data_dir / "arthur20220324_imu_info.parquet"
    output_path = (
        script_dir.parent / "videos" / "20220324_arthur_02_r1_trimmed_ekf_smoother.mp4"
    )

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    print("=" * 80)
    print("Creating EKF + RTS Smoother Video")
    print("=" * 80)

    # Load session data
    print("\nLoading session data…")
    meters_per_pixel = 0.0022
    data = load_arthur_session(
        position_file=str(position_file),
        imu_file=str(imu_file),
        imu_mode="3d",
        meters_per_pixel=meters_per_pixel,
        verbose=True,
    )
    print(
        f"✓ Loaded {len(data.t_cam):,} camera frames and {len(data.t_imu):,} IMU samples"
    )

    # Run EKF
    print("\nRunning Extended Kalman Filter (3D IMU)…")
    t0 = time.time()
    pixel_std_px = 3.0
    cam_sigma_m = meters_per_pixel * pixel_std_px  # ≈ 0.0044 m
    cam_var_m2 = cam_sigma_m**2  # ≈ 1.94e-5 m^2
    ekf_config = EKFConfig(
        state_mode="2d_cam_3d_imu",
        # Camera measurement noise (per axis)
        measurement_noise_pos=cam_var_m2,  # derive from pixels→meters
        # Heading pseudo-measurement (from LED spacing)
        use_heading_measurement=True,
        led_distance=0.0475,  # set to your rig (meters)
        led_distance_tolerance=0.4,
        adaptive_heading_noise=True,
        measurement_noise_heading=(0.06) ** 2,
        # Process (model) diffusion
        process_noise_pos=1e-4,  # m^2/s, modest growth during gaps
        process_noise_vel=2.5e-3,  # (m/s)^2/s
        process_noise_heading=5e-3,  # rad^2/s (~0.8°/s 1σ)
        process_noise_gyro_bias=1e-6,  # ((rad/s)^2)/s
        process_noise_accel_bias=1e-4,  # ((m/s^2)^2)/s
        # Robust gating
        use_mahalanobis_gating=False,
        mahalanobis_threshold_prob=0.999,
        # Vision blackout behavior
        adaptive_q_during_dropout=True,
        dropout_q_pos_multiplier=1.5,
        dropout_q_vel_multiplier=1.5,
        dropout_q_bias_multiplier=0.5,
        freeze_bias_during_blackout=True,
        reduce_imu_noise_during_blackout=True,
        blackout_imu_noise_scale=0.3,
        # Zero velocity updates
        enable_zupt=True,
        zupt_velocity_threshold=0.04,
        zupt_measurement_noise=(0.01) ** 2,
    )

    ekf_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=data.t_imu,
        U_imu=data.U_imu,
        t_cam=data.t_cam,
        Z_cam_led1=data.Z_cam_led1,
        Z_cam_led2=data.Z_cam_led2,
        mask_cam=data.mask_cam,
    )
    print(
        f"✓ EKF complete in {time.time() - t0:.1f}s: {len(ekf_result.filtered_means):,} timesteps"
    )

    # Run RTS smoother
    print("\nRunning RTS smoother (backward pass)…")
    t1 = time.time()
    smoother_result = rts_smoother(
        ekf_result,
        ekf_config,
        t_imu=data.t_imu,
        U_imu=data.U_imu,
        t_cam=data.t_cam,
        mask_cam=data.mask_cam,
    )
    print(
        f"✓ RTS smoother complete in {time.time() - t1:.1f}s: {len(smoother_result.smoothed_means):,} timesteps"
    )

    # Load position dataframe
    position_df = pd.read_parquet(position_file)

    # Configure rendering
    style = OverlayStyle(
        led_marker_size=10.0,
        trajectory_length=90,
        heading_arrow_px=40.0,
        heading_flip=True,
        show_uncertainty_circle=False,
        invert_yaxis=False,
    )

    # Get video info to determine duration
    video_info = get_video_info(str(video_path))
    video_duration = video_info["duration_s"]

    print(f"\nVideo duration: {video_duration:.1f}s")
    print("Rendering 10-second clip at 30 fps using parallel PNG rendering")

    config = RenderConfig(
        start_time=10.0,  # Start at 10s to skip initial setup
        duration=1.5 * 60.0,  # Just 10 seconds for testing
        fps=30.0,
        imu_window_s=2.0,
        state_window_s=5.0,
        dpi=100,
        render_mode="parallel_png",  # Use parallel rendering for speed
        max_workers=8,
        ffmpeg_threads=4,
        bitrate_kbps=4000,
        style=style,
    )

    print("\nGenerating video…")
    create_filter_overlay_video(
        video_path=str(video_path),
        data=data,
        filter_result=smoother_result,  # Pass smoother result
        t_filter=data.t_cam,
        position_df=position_df,
        output_path=str(output_path),
        config=config,
    )

    print(f"\n{'=' * 80}")
    print("✓ Complete! Video saved to:")
    print(f"  {output_path}")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())
    sys.exit(main())
