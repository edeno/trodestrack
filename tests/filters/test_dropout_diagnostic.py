"""Diagnostic script for dropout drift issue.

This is a diagnostic visualization script, not a test suite.
Run directly with: python -m tests.filters.test_dropout_diagnostic
"""

import matplotlib.pyplot as plt
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu


def main():
    """Run dropout diagnostic and generate visualization."""
    # Generate simulation with dropout
    config = RatIMUSimConfig(
        duration_s=15.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_rw_density=0.0001,
        accel_bias_rw_density=0.001,
        cam_sigma_m=0.005,
        use_second_led=True,
    )
    sim_data = simulate_rat_imu(config=config, seed=42)

    # Inject 5s dropout from t=5s to t=10s
    dropout_start_idx = np.searchsorted(sim_data["t_cam_exp"], 5.0)
    dropout_end_idx = np.searchsorted(sim_data["t_cam_exp"], 10.0)

    mask_with_dropout = sim_data["mask_cam"].copy()
    mask_with_dropout[dropout_start_idx:dropout_end_idx] = False

    # Run EKF with standard config
    ekf_config = EKFConfig(
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        process_noise_heading=0.02,
        process_noise_gyro_bias=2e-6,
        process_noise_accel_bias=2e-4,
        measurement_noise_pos=0.005**2,
        imu_gyro_noise_density=0.001,
        imu_accel_noise_density=0.05,
        damping_coeff=0.5,
        led_distance=0.04,
        use_heading_measurement=True,
    )

    sim_data_dropout = sim_data.copy()
    sim_data_dropout["mask_cam"] = mask_with_dropout

    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_data["t_imu"],
        U_imu=sim_data["U_imu"],
        t_cam=sim_data["t_cam_exp"],
        Z_cam_led1=sim_data["Z_cam_led1"],
        Z_cam_led2=sim_data["Z_cam_led2"],
        mask_cam=mask_with_dropout,
    )

    # Resolve bias indices via the actual filter layout — under the default
    # 10D 2d_cam_3d_imu layout, b_gz is at column 6 and accel-bias columns
    # are 7..9; hardcoding 5 / 6:8 silently plotted heading and gyro bias.
    layout = get_layout(ekf_config.state_mode)
    gyro_bias_idx = layout.bias_gyro_idx[0]
    accel_bias_x_idx = layout.bias_accel_idx[0]
    accel_bias_y_idx = layout.bias_accel_idx[1]
    accel_bias_slice = list(layout.bias_accel_idx)[:2]

    # Interpolate truth
    t_truth = sim_data["t_imu"]
    X_truth = sim_data["X_truth"]

    pos_truth = np.column_stack(
        [
            np.interp(sim_data["t_cam_exp"], t_truth, X_truth[:, 0]),
            np.interp(sim_data["t_cam_exp"], t_truth, X_truth[:, 1]),
        ]
    )

    # Plot trajectory during dropout
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position trajectory
    ax = axes[0, 0]
    ax.plot(pos_truth[:, 0], pos_truth[:, 1], "k-", label="Ground Truth", linewidth=2)
    ax.plot(
        result.filtered_means[:, 0],
        result.filtered_means[:, 1],
        "b-",
        label="EKF Estimate",
        linewidth=2,
    )

    # Highlight dropout region
    dropout_truth = pos_truth[dropout_start_idx:dropout_end_idx]
    dropout_est = result.filtered_means[dropout_start_idx:dropout_end_idx, :2]

    ax.plot(
        dropout_truth[:, 0],
        dropout_truth[:, 1],
        "r-",
        linewidth=3,
        label="Dropout (Truth)",
    )
    ax.plot(
        dropout_est[:, 0], dropout_est[:, 1], "m-", linewidth=3, label="Dropout (EKF)"
    )

    ax.scatter(
        [dropout_truth[0, 0]],
        [dropout_truth[0, 1]],
        c="green",
        s=100,
        marker="o",
        label="Dropout Start",
    )
    ax.scatter(
        [dropout_truth[-1, 0]],
        [dropout_truth[-1, 1]],
        c="orange",
        s=100,
        marker="s",
        label="Dropout End",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Position Trajectory")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    # Position error over time
    ax = axes[0, 1]
    pos_error = (
        np.linalg.norm(pos_truth - result.filtered_means[:, :2], axis=1) * 100
    )  # cm
    ax.plot(sim_data["t_cam_exp"], pos_error, "b-", linewidth=2)
    ax.axvspan(5.0, 10.0, color="red", alpha=0.2, label="Dropout")
    ax.axhline(15.0, color="red", linestyle="--", label="Target limit (15 cm)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (cm)")
    ax.set_title("Position Error Over Time")
    ax.legend()
    ax.grid(True)

    # Bias estimates
    ax = axes[1, 0]
    ax.plot(
        sim_data["t_cam_exp"],
        result.filtered_means[:, gyro_bias_idx],
        label="Gyro Bias (rad/s)",
    )
    ax.plot(
        sim_data["t_cam_exp"],
        result.filtered_means[:, accel_bias_x_idx],
        label="Accel X Bias (m/s²)",
    )
    ax.plot(
        sim_data["t_cam_exp"],
        result.filtered_means[:, accel_bias_y_idx],
        label="Accel Y Bias (m/s²)",
    )
    ax.axvspan(5.0, 10.0, color="red", alpha=0.2, label="Dropout")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bias Estimate")
    ax.set_title("IMU Bias Estimates")
    ax.legend()
    ax.grid(True)

    # Velocity error
    ax = axes[1, 1]
    vel_truth = np.column_stack(
        [
            np.interp(sim_data["t_cam_exp"], t_truth, X_truth[:, 2]),
            np.interp(sim_data["t_cam_exp"], t_truth, X_truth[:, 3]),
        ]
    )
    vel_error = (
        np.linalg.norm(vel_truth - result.filtered_means[:, 2:4], axis=1) * 100
    )  # cm/s
    ax.plot(sim_data["t_cam_exp"], vel_error, "b-", linewidth=2)
    ax.axvspan(5.0, 10.0, color="red", alpha=0.2, label="Dropout")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity Error (cm/s)")
    ax.set_title("Velocity Error Over Time")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("/tmp/dropout_diagnostic.png", dpi=150)
    print("\nDiagnostic plot saved to /tmp/dropout_diagnostic.png")

    # Print detailed stats
    print(f"\nDropout region: t={5.0}s to t={10.0}s")
    print(f"Position at dropout start (truth): {dropout_truth[0]}")
    print(f"Position at dropout start (EKF):   {dropout_est[0]}")
    print(f"Position at dropout end (truth):   {dropout_truth[-1]}")
    print(f"Position at dropout end (EKF):     {dropout_est[-1]}")

    drift_truth = np.linalg.norm(dropout_truth[-1] - dropout_truth[0]) * 100
    drift_ekf = np.linalg.norm(dropout_est[-1] - dropout_est[0]) * 100
    drift_error = np.linalg.norm(dropout_est[-1] - dropout_truth[-1]) * 100

    print(f"\nActual animal motion during dropout: {drift_truth:.3f} cm")
    print(f"EKF estimated motion during dropout: {drift_ekf:.3f} cm")
    print(f"Drift ERROR (EKF vs truth at end):   {drift_error:.3f} cm")

    # Check bias drift via layout indices, not the hardcoded 6:8 / 5 that
    # mixed accel-bias and gyro-bias columns under the default 10D layout.
    bias_accel_start = result.filtered_means[dropout_start_idx, accel_bias_slice]
    bias_accel_end = result.filtered_means[dropout_end_idx, accel_bias_slice]
    bias_drift = np.linalg.norm(bias_accel_end - bias_accel_start)
    print(f"\nAccelerometer bias drift during dropout: {bias_drift:.6f} m/s²")
    print(
        f"Gyro bias start: "
        f"{result.filtered_means[dropout_start_idx, gyro_bias_idx]:.6f} rad/s"
    )
    print(
        f"Gyro bias end:   "
        f"{result.filtered_means[dropout_end_idx, gyro_bias_idx]:.6f} rad/s"
    )


if __name__ == "__main__":
    main()
