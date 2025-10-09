"""Generate diagnostic videos for EKF on different simulation scenarios.

This script creates animated visualizations showing:
1. Stationary scenario - EKF rejecting IMU drift
2. Constant velocity scenario - EKF tracking linear motion
3. Circular motion scenario - EKF tracking rotation with gyro bias learning

Usage:
    uv run python examples/04_ekf_scenarios_video.py

Output:
    Saves MP4 videos to current directory for each scenario.
"""

from __future__ import annotations

import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)
from trodestrack.viz.video import create_diagnostic_video


def main() -> None:
    """Generate diagnostic videos for each EKF scenario."""
    print("=" * 70)
    print("EKF Scenario Videos")
    print("=" * 70)

    # Common configuration
    sim_config = SimpleSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_std=0.01,
        accel_bias_std=0.05,
        cam_noise_std=0.005,
        cam_dropout_prob=0.1,
    )

    ekf_config = EKFConfig(
        process_noise_pos=0.01**2 / 0.005,
        process_noise_vel=0.1**2 / 0.005,
        process_noise_heading=0.01**2 / 0.005,
        process_noise_gyro_bias=1e-6 / 0.005,
        process_noise_accel_bias=1e-4 / 0.005,
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.05**2,
        imu_gyro_noise_density=0.001,
        imu_accel_noise_density=0.05,
        damping_coeff=0.5,
        led_distance=0.04,  # 4 cm LED spacing
    )

    # Scenario 1: Stationary
    print("\n" + "=" * 70)
    print("1. Stationary Scenario")
    print("=" * 70)
    print("Running simulation...")
    sim_stationary = simulate_stationary(sim_config, position=np.array([1.0, 1.0]), seed=42)

    print("Running EKF...")
    result_stationary = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_stationary["t_imu"],
        U_imu=sim_stationary["U_imu"],
        t_cam=sim_stationary["t_cam_exp"],
        Z_cam_led1=sim_stationary["Z_cam_led1"],
        Z_cam_led2=sim_stationary["Z_cam_led2"],
        mask_cam=sim_stationary["mask_cam"],
    )

    print("Generating video...")
    from pathlib import Path

    out = Path("diagnostics/videos/ekf_stationary.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    output_path = create_diagnostic_video(
        sim_data=sim_stationary,
        output_path=str(out),
        filter_results=result_stationary,
        led_distance=ekf_config.led_distance,
        fps=30,
        speedup=1.0,
    )
    print(f"✓ Saved: {output_path}")

    # Scenario 2: Constant Velocity
    print("\n" + "=" * 70)
    print("2. Constant Velocity Scenario")
    print("=" * 70)
    print("Running simulation...")
    sim_const_vel = simulate_constant_velocity(
        sim_config,
        initial_position=np.array([0.5, 0.5]),
        velocity=np.array([0.3, 0.0]),
        seed=42,
    )

    print("Running EKF...")
    result_const_vel = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_const_vel["t_imu"],
        U_imu=sim_const_vel["U_imu"],
        t_cam=sim_const_vel["t_cam_exp"],
        Z_cam_led1=sim_const_vel["Z_cam_led1"],
        Z_cam_led2=sim_const_vel["Z_cam_led2"],
        mask_cam=sim_const_vel["mask_cam"],
    )

    print("Generating video...")
    out = Path("diagnostics/videos/ekf_constant_velocity.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    output_path = create_diagnostic_video(
        sim_data=sim_const_vel,
        output_path=str(out),
        filter_results=result_const_vel,
        led_distance=ekf_config.led_distance,
        fps=30,
        speedup=1.0,
    )
    print(f"✓ Saved: {output_path}")

    # Scenario 3: Circular Motion (longer duration for bias convergence demo)
    print("\n" + "=" * 70)
    print("3. Circular Motion Scenario")
    print("=" * 70)
    print("Running simulation...")
    # Use longer duration to demonstrate gyro bias learning
    circular_config = SimpleSimConfig(
        duration_s=20.0,  # 2x longer for bias convergence demo
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_std=0.01,
        accel_bias_std=0.05,
        cam_noise_std=0.005,
        cam_dropout_prob=0.1,
    )
    sim_circular = simulate_circular(
        circular_config,
        center=np.array([1.0, 1.0]),
        radius=0.5,
        angular_velocity=0.5,
        seed=42,
    )

    print("Running EKF...")
    result_circular = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_circular["t_imu"],
        U_imu=sim_circular["U_imu"],
        t_cam=sim_circular["t_cam_exp"],
        Z_cam_led1=sim_circular["Z_cam_led1"],
        Z_cam_led2=sim_circular["Z_cam_led2"],
        mask_cam=sim_circular["mask_cam"],
    )

    print("Generating video...")
    out = Path("diagnostics/videos/ekf_circular.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    output_path = create_diagnostic_video(
        sim_data=sim_circular,
        output_path=str(out),
        filter_results=result_circular,
        led_distance=ekf_config.led_distance,
        fps=30,
        speedup=1.0,
    )
    print(f"✓ Saved: {output_path}")

    print("\n" + "=" * 70)
    print("✓ All videos generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
