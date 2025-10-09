"""Generate diagnostic video for EKF on realistic rat IMU simulation (Tier 3).

This script demonstrates the EKF performance on a simulation with:
- Ornstein-Uhlenbeck motion dynamics (realistic rat movement)
- IMU tilt, drag, and bias random walks
- LED occlusions, swaps, and confidence scaling
- Correlated camera dropouts

Usage:
    uv run python examples/05_ekf_rat_imu_video.py

Output:
    Saves MP4 video showing EKF tracking performance on realistic simulation.
"""

from __future__ import annotations

import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.qa.metrics import compute_nees, compute_position_rmse
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.utils import interp_angle
from trodestrack.viz.video import create_diagnostic_video


def main() -> None:
    """Generate diagnostic video for EKF on realistic rat IMU simulation."""
    print("=" * 70)
    print("EKF on Realistic Rat IMU Simulation (Tier 3)")
    print("=" * 70)

    # Realistic simulation configuration with robustness features
    sim_config = RatIMUSimConfig(
        duration_s=15.0,
        fs_imu=200.0,
        fs_cam=30.0,
        # OU motion dynamics parameters
        tau_yaw_rate=0.8,  # Yaw rate correlation time (s)
        sigma_yaw_rate=np.deg2rad(60.0),  # Yaw rate noise (rad/s / √s)
        tau_a_fwd=0.7,  # Forward accel correlation time (s)
        sigma_a_fwd=1.0,  # Forward accel noise (m/s² / √s)
        tau_a_lat=0.5,  # Lateral accel correlation time (s)
        sigma_a_lat=0.5,  # Lateral accel noise (m/s² / √s)
        # IMU parameters
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_rw_density=0.01,
        accel_bias_rw_density=0.05,
        # LED configuration
        use_second_led=True,
        led1_offset_body=np.array([0.025, 0.0]),  # Front LED
        led2_offset_body=np.array([-0.025, 0.0]),  # Back LED
        led_swap_prob=0.05,  # 5% swap rate
        # Camera parameters
        cam_sigma_m=0.005,  # 0.5 cm
        cam_dropout_prob=0.15,  # 15% dropout
        cam_dropout_correlation=0.75,  # Correlated dropouts
        use_confidence=True,
        confidence_base=0.85,
    )

    print("\nRunning realistic simulation...")
    sim = simulate_rat_imu(sim_config, seed=42)

    print(f"  Generated {len(sim['t_imu'])} IMU samples")
    print(f"  Camera: {len(sim['t_cam_exp'])} frames")
    print(
        f"  Valid: {sim['mask_cam'].sum()}/{len(sim['mask_cam'])} "
        f"({100*sim['mask_cam'].mean():.1f}%)"
    )

    # EKF configuration
    ekf_config = EKFConfig(
        # Process noise rates (will be multiplied by dt in filter)
        process_noise_pos=0.01**2 / 0.005,  # (m)²/s
        process_noise_vel=0.1**2 / 0.005,  # (m/s)²/s
        process_noise_heading=0.01**2 / 0.005,  # (rad)²/s
        process_noise_gyro_bias=1e-6 / 0.005,  # (rad/s)²/s
        process_noise_accel_bias=1e-4 / 0.005,  # (m/s²)²/s
        # Measurement noise (match simulation)
        measurement_noise_pos=0.005**2,  # (m)² = (0.5 cm)²
        measurement_noise_heading=0.05**2,  # (rad)²
        # IMU noise (match simulation)
        imu_gyro_noise_density=0.001,  # rad/s/√Hz
        imu_accel_noise_density=0.05,  # m/s²/√Hz
        # Velocity damping coefficient
        damping_coeff=0.5,  # 1/s
        # LED configuration
        led_distance=0.05,  # 5 cm front-back spacing
    )

    print("\nRunning EKF...")
    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Compute performance metrics
    print("\nComputing performance metrics...")
    # Interpolate ground truth at camera times (linear for pos/vel, angle-aware for heading)
    t_imu = sim["t_imu"]
    t_cam = sim["t_cam_exp"]
    X_truth = sim["X_truth"]

    X_truth_at_cam = np.column_stack(
        [
            np.interp(t_cam, t_imu, X_truth[:, 0]),  # x
            np.interp(t_cam, t_imu, X_truth[:, 1]),  # y
            np.interp(t_cam, t_imu, X_truth[:, 2]),  # vx
            np.interp(t_cam, t_imu, X_truth[:, 3]),  # vy
            interp_angle(t_cam, t_imu, X_truth[:, 4]),  # θ (angle-aware)
        ]
    )
    X_est = np.array(result.filtered_means)
    P_est = np.array(result.filtered_covariances)

    # Position RMSE
    pos_rmse = compute_position_rmse(
        X_truth_at_cam[:, :2] * 100,  # m to cm
        X_est[:, :2] * 100,  # m to cm
    )

    # Velocity error
    vel_err = np.linalg.norm(X_est[:, 2:4] - X_truth_at_cam[:, 2:4], axis=1) * 100  # m/s to cm/s
    vel_rmse = np.sqrt(np.mean(vel_err**2))

    # Heading error
    def wrap_angle(theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

    head_err = np.degrees(wrap_angle(X_est[:, 4] - X_truth_at_cam[:, 4]))
    head_rmse = np.sqrt(np.mean(head_err**2))

    # NEES for position
    nees = compute_nees(
        states_true=X_truth_at_cam[:, :2],
        states_est=X_est[:, :2],
        covariances_est=P_est[:, :2, :2],
    )
    mean_nees = np.mean(nees)

    # Innovation statistics (measurement residuals)
    # Compute predicted LED positions from predicted state
    X_pred = np.array(result.predicted_means)
    innovations_led1 = []
    innovations_led2 = []

    for i in range(len(t_cam)):
        if sim["mask_led1"][i]:
            px_pred, py_pred, theta_pred = X_pred[i, 0], X_pred[i, 1], X_pred[i, 4]
            dx = 0.5 * ekf_config.led_distance * np.cos(theta_pred)
            dy = 0.5 * ekf_config.led_distance * np.sin(theta_pred)
            led1_pred = np.array([px_pred - dx, py_pred - dy])
            innov = np.linalg.norm(sim["Z_cam_led1"][i] - led1_pred) * 100  # cm
            innovations_led1.append(innov)

        if sim["mask_led2"][i]:
            px_pred, py_pred, theta_pred = X_pred[i, 0], X_pred[i, 1], X_pred[i, 4]
            dx = 0.5 * ekf_config.led_distance * np.cos(theta_pred)
            dy = 0.5 * ekf_config.led_distance * np.sin(theta_pred)
            led2_pred = np.array([px_pred + dx, py_pred + dy])
            innov = np.linalg.norm(sim["Z_cam_led2"][i] - led2_pred) * 100  # cm
            innovations_led2.append(innov)

    all_innovations = innovations_led1 + innovations_led2
    mean_innov = np.mean(all_innovations) if all_innovations else 0.0
    std_innov = np.std(all_innovations) if all_innovations else 0.0

    # Dropout analysis
    dropout_lengths = []
    current_dropout = 0
    for i in range(len(sim["mask_cam"])):
        if not sim["mask_cam"][i]:
            current_dropout += 1
        elif current_dropout > 0:
            dropout_lengths.append(current_dropout)
            current_dropout = 0
    if current_dropout > 0:
        dropout_lengths.append(current_dropout)

    max_dropout = max(dropout_lengths) if dropout_lengths else 0
    n_dropouts = len(dropout_lengths)

    print("\n" + "=" * 70)
    print("Performance Metrics")
    print("=" * 70)
    print(f"Position RMSE: {pos_rmse:.2f} cm (PRD target: ≤2 cm)")
    print(f"Velocity RMSE: {vel_rmse:.2f} cm/s (PRD target: ≤10 cm/s)")
    print(f"Heading RMSE: {head_rmse:.2f}° (PRD target: ≤7°)")
    print(f"Mean NEES: {mean_nees:.2f} (ideal: 2.0 for 2D position)")
    print(f"Marginal Log-Likelihood: {result.marginal_loglik:.2f}")
    print()
    print("Innovation (residual) stats:")
    print(f"  Mean: {mean_innov:.2f} cm")
    print(f"  Std:  {std_innov:.2f} cm")
    print()
    print("Dropout analysis:")
    print(f"  Sequences: {n_dropouts}")
    print(f"  Max length: {max_dropout} frames")
    print("=" * 70)

    print("\nGenerating video...")
    from pathlib import Path

    out = Path("diagnostics/videos/ekf_rat_imu.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    output_path = create_diagnostic_video(
        sim_data=sim,
        output_path=str(out),
        filter_results=result,
        led_distance=ekf_config.led_distance,
        fps=30,
        speedup=1.0,
    )

    print("\n" + "=" * 70)
    print(f"✓ Video saved: {output_path}")
    print(f"  Duration: {sim_config.duration_s:.1f}s")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
