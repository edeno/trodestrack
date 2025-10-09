"""Generate position and heading error plots for 5s dropout scenario.

Creates a clean 2-panel figure showing:
- Position error time series (filter vs smoothed)
- Heading error time series (filter vs smoothed)

Usage:
    uv run python examples/07_dropout_error_plots.py

Output:
    Saves PNG: output/dropout_error_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.qa.metrics import compute_dropout_drift
from trodestrack.runtime.offline import rts_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.utils import interp_angle
from trodestrack.viz.styles import COLORS, apply_tufte_style

# Output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Generate error comparison plots for filter vs IEKS smoother."""
    print("=" * 70)
    print("Filter vs IEKS Smoother: Error Comparison Plots")
    print("=" * 70)

    # Simulation configuration (same as test)
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
        imu_tilt_roll_deg=0.0,
        imu_tilt_pitch_deg=0.0,
    )

    print("\nRunning simulation...")
    sim_data = simulate_rat_imu(config=config, seed=42)

    # Inject 5s dropout from t=5s to t=10s
    dropout_start_idx = np.searchsorted(sim_data["t_cam_exp"], 5.0)
    dropout_end_idx = np.searchsorted(sim_data["t_cam_exp"], 10.0)

    mask_with_dropout = sim_data["mask_cam"].copy()
    mask_with_dropout[dropout_start_idx:dropout_end_idx] = False

    sim_data_dropout = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in sim_data.items()}
    sim_data_dropout["mask_cam"] = mask_with_dropout

    # Clear LEDs during blackout
    for key in ("Z_cam_led1", "Z_cam_led2"):
        if key in sim_data_dropout:
            arr = sim_data_dropout[key].copy()
            arr[dropout_start_idx:dropout_end_idx] = float("nan")
            sim_data_dropout[key] = arr

    for key in ("mask_led1", "mask_led2"):
        if key in sim_data_dropout:
            arr = sim_data_dropout[key].copy()
            arr[dropout_start_idx:dropout_end_idx] = False
            sim_data_dropout[key] = arr

    # EKF configuration
    ekf_config_override = {
        "damping_coeff": 0.4,
        "freeze_bias_during_blackout": True,
        "reduce_imu_noise_during_blackout": True,
        "blackout_imu_noise_scale": 0.5,
    }
    ekf_params = dict(
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
    ekf_params.update(ekf_config_override)
    ekf_config = EKFConfig(**ekf_params)

    print("Running EKF filter...")
    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_data_dropout["t_imu"],
        U_imu=sim_data_dropout["U_imu"],
        t_cam=sim_data_dropout["t_cam_exp"],
        Z_cam_led1=sim_data_dropout["Z_cam_led1"],
        Z_cam_led2=sim_data_dropout["Z_cam_led2"],
        mask_cam=mask_with_dropout,
    )

    print("Running IEKS smoother (iter=2, blackout-aware Q/R)...")
    smoother_result = rts_smoother(
        filter_result=filter_result,
        ekf_config=ekf_config,
        t_imu=sim_data_dropout["t_imu"],
        U_imu=sim_data_dropout["U_imu"],
        t_cam=sim_data_dropout["t_cam_exp"],
        num_iter=2,
        mask_cam=mask_with_dropout,
    )

    # Compute drift metrics
    drift_result_filter = compute_dropout_drift(
        positions=filter_result.filtered_means[:, :2],
        valid_mask=mask_with_dropout,
        t=sim_data["t_cam_exp"],
        min_duration_s=4.5,
    )

    drift_result_smooth = compute_dropout_drift(
        positions=smoother_result.smoothed_means[:, :2],
        valid_mask=mask_with_dropout,
        t=sim_data["t_cam_exp"],
        min_duration_s=4.5,
    )

    drift_filter_m = drift_result_filter["drift_m"]
    drift_smooth_m = drift_result_smooth["drift_m"]

    print("\nDrift Results:")
    print(f"  Filter:   {drift_filter_m:.4f} m")
    print(f"  Smoothed: {drift_smooth_m:.4f} m")
    print(f"  Improvement: {drift_filter_m / drift_smooth_m:.2f}× reduction")

    # Interpolate truth
    t_truth = sim_data["t_imu"]
    X_truth = sim_data["X_truth"]
    t_cam = sim_data["t_cam_exp"]

    pos_truth = np.column_stack(
        [
            np.interp(t_cam, t_truth, X_truth[:, 0]),
            np.interp(t_cam, t_truth, X_truth[:, 1]),
        ]
    )
    heading_truth = interp_angle(t_cam, t_truth, X_truth[:, 4])

    # Extract estimates
    pos_filter = np.array(filter_result.filtered_means[:, :2])
    pos_smooth = np.array(smoother_result.smoothed_means[:, :2])
    heading_filter = np.array(filter_result.filtered_means[:, 4])
    heading_smooth = np.array(smoother_result.smoothed_means[:, 4])

    # Compute errors
    pos_err_filter = np.linalg.norm(pos_filter - pos_truth, axis=1)
    pos_err_smooth = np.linalg.norm(pos_smooth - pos_truth, axis=1)

    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    heading_err_filter = np.abs(angle_diff(heading_filter, heading_truth))
    heading_err_smooth = np.abs(angle_diff(heading_smooth, heading_truth))

    # Create figure
    print("\nGenerating error plots...")
    apply_tufte_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("5s Vision Dropout: Filter vs IEKS Smoother", fontsize=14, fontweight="bold")

    # Position error panel
    ax1.plot(
        t_cam,
        pos_err_filter,
        color=COLORS["red"],
        linewidth=2.5,
        label=f"Filter (drift: {drift_filter_m:.2f}m)",
    )
    ax1.plot(
        t_cam,
        pos_err_smooth,
        color=COLORS["blue"],
        linewidth=2.5,
        label=f"Smoothed (drift: {drift_smooth_m:.2f}m, {drift_filter_m/drift_smooth_m:.2f}× improvement)",
    )
    ax1.axvspan(5.0, 10.0, alpha=0.15, color="gray", label="Vision blackout")
    ax1.set_ylabel("Position Error (m)", fontsize=11)
    ax1.set_title("Position Error", fontsize=12, pad=10)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Heading error panel
    ax2.plot(
        t_cam,
        np.rad2deg(heading_err_filter),
        color=COLORS["red"],
        linewidth=2.5,
        label="Filter",
    )
    ax2.plot(
        t_cam,
        np.rad2deg(heading_err_smooth),
        color=COLORS["blue"],
        linewidth=2.5,
        label="Smoothed",
    )
    ax2.axvspan(5.0, 10.0, alpha=0.15, color="gray", label="Vision blackout")
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Heading Error (deg)", fontsize=11)
    ax2.set_title("Heading Error", fontsize=12, pad=10)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "dropout_error_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Figure saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
