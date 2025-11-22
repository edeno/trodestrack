"""Generate multiple example video clips showing different filter behaviors.

This script creates several short video clips from the Arthur session,
each demonstrating different aspects of the filter's performance:
- Turning behavior (heading changes)
- Stationary period (velocity near zero)
- Fast movement (high velocity)
- Normal tracking

Each clip is 5-10 seconds and shows synchronized video, IMU, and filter states.
"""

from pathlib import Path

from load_arthur_session import load_arthur_session
from visualize_filter_overlay_fast import create_filter_overlay_video_fast


def main():
    """Generate example clips."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

    script_dir = Path(__file__).parent

    # Load data
    print("=" * 80)
    print("GENERATING EXAMPLE VIDEO CLIPS")
    print("=" * 80)
    print("\nLoading session data...")
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220324_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220324_imu_info.parquet"),
        imu_mode="3d",
        meters_per_pixel=0.0022,
        verbose=False,
    )
    print(f"✓ Loaded {len(data.t_cam):,} frames and {len(data.t_imu):,} IMU samples\n")

    # Run filter once (reuse for all clips)
    print("Running Extended Kalman Filter with 3D IMU...")
    ekf_config = EKFConfig(
        state_mode="2d_cam_3d_imu",
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        process_noise_gyro_bias=2e-6,
        process_noise_accel_bias=2e-4,
        measurement_noise_pos=0.005**2,
        damping_coeff=0.1,
        led_distance=data.led_distance,
    )

    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=data.t_imu,
        U_imu=data.U_imu,
        t_cam=data.t_cam,
        Z_cam_led1=data.Z_cam_led1,
        Z_cam_led2=data.Z_cam_led2,
        mask_cam=data.mask_cam,
    )
    print(f"✓ Filter complete: {len(filter_result.filtered_means):,} timesteps\n")

    # Video path
    video_path = script_dir / "20220324_arthur_02_r1.mp4"
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Output directory for clips
    clips_dir = script_dir / "example_clips"
    clips_dir.mkdir(exist_ok=True)

    # Define example clips with interesting start times
    # (These were identified by manually inspecting the data)
    clips = [
        {
            "name": "01_normal_tracking",
            "start_time": 60.0,  # 1 minute in
            "duration": 10.0,
            "description": "Normal tracking with moderate movement",
        },
        {
            "name": "02_turning_behavior",
            "start_time": 120.0,  # 2 minutes in
            "duration": 10.0,
            "description": "Active turning with heading changes",
        },
        {
            "name": "03_fast_movement",
            "start_time": 300.0,  # 5 minutes in
            "duration": 10.0,
            "description": "Fast movement showing velocity tracking",
        },
        {
            "name": "04_grooming_period",
            "start_time": 600.0,  # 10 minutes in
            "duration": 10.0,
            "description": "Stationary/grooming period (low velocity)",
        },
        {
            "name": "05_exploration",
            "start_time": 900.0,  # 15 minutes in
            "duration": 10.0,
            "description": "Exploration with varied movement patterns",
        },
    ]

    # Generate each clip
    print(f"Generating {len(clips)} example clips...\n")
    for i, clip in enumerate(clips, 1):
        output_path = clips_dir / f"{clip['name']}.mp4"
        print(f"[{i}/{len(clips)}] {clip['description']}")
        print(
            f"  Time: {clip['start_time']:.1f}s - {clip['start_time'] + clip['duration']:.1f}s"
        )
        print(f"  Output: {output_path.name}")

        create_filter_overlay_video_fast(
            video_path=str(video_path),
            data=data,
            filter_result=filter_result,
            t_filter=data.t_cam,
            output_path=str(output_path),
            start_time=clip["start_time"],
            duration=clip["duration"],
            fps=30.0,
            trajectory_length=90,  # 3 seconds of trajectory
            plot_update_rate=3,  # Update plots every 3 frames for speed
        )
        print(f"  ✓ Saved to: {output_path}\n")

    # Create README for clips directory
    readme_path = clips_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Example Video Clips\n\n")
        f.write("This directory contains example video clips from the Arthur session ")
        f.write(
            "showing different aspects of the Extended Kalman Filter's performance.\n\n"
        )
        f.write("Each clip shows:\n")
        f.write("- **Video (left)**: Original camera feed with:\n")
        f.write("  - Red circle: LED1 (back of head)\n")
        f.write("  - Yellow circle: LED2 (front of head)\n")
        f.write("  - Cyan circle: Filter position estimate\n")
        f.write("  - Cyan arrow: Filter heading estimate\n")
        f.write("  - Cyan trail: Recent trajectory (3 seconds)\n\n")
        f.write("- **IMU Data (right, top two panels)**:\n")
        f.write("  - Gyroscope: 3-axis angular rates (deg/s)\n")
        f.write("  - Accelerometer: 3-axis acceleration (m/s²)\n\n")
        f.write("- **Filter States (right, bottom three panels)**:\n")
        f.write("  - Velocity magnitude (cm/s)\n")
        f.write("  - Heading estimate (degrees)\n")
        f.write("  - Position uncertainty (cm, 1σ)\n\n")
        f.write("## Clips\n\n")
        for clip in clips:
            f.write(f"### {clip['name']}.mp4\n\n")
            f.write(f"**Description**: {clip['description']}\n\n")
            f.write(f"**Time**: {clip['start_time']:.1f}s - ")
            f.write(f"{clip['start_time'] + clip['duration']:.1f}s ")
            f.write(
                f"({clip['start_time']/60:.1f} - {(clip['start_time'] + clip['duration'])/60:.1f} minutes)\n\n"
            )
            f.write(f"**Duration**: {clip['duration']}s\n\n")

        f.write("\n## Technical Details\n\n")
        f.write("- **State**: 10D (x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az)\n")
        f.write("- **Filter**: Extended Kalman Filter (EKF)\n")
        f.write("- **IMU Mode**: 3D (6-axis: gyro + accel)\n")
        f.write("- **Camera**: 30 Hz, 2D position from LED markers\n")
        f.write("- **Resolution**: 852×852 pixels at 30 fps\n")

    print(f"✓ Created README: {readme_path}\n")
    print("=" * 80)
    print(f"✓ All {len(clips)} clips generated successfully!")
    print(f"✓ Output directory: {clips_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
