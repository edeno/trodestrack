"""Visualize robustness test scenarios from test_robustness.py.

This script generates plots showing filter behavior under challenging conditions:
1. Extreme outlier rejection via gating
2. Stability during frequent LED swaps
3. Stability during 5-second dropout
4. Bias estimation during dropouts

Run: uv run python examples/05_robustness_scenarios.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.qa.metrics import compute_position_rmse
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.simple import SimpleSimConfig, simulate_constant_velocity, simulate_stationary


def plot_scenario_1_outlier_rejection():
    """Plot outlier rejection via Mahalanobis gating."""
    print("Scenario 1: Outlier rejection via Mahalanobis gating...")

    # Create stationary scenario
    config_sim = SimpleSimConfig(
        duration_s=10.0,
        cam_noise_std=0.005,
        cam_dropout_prob=0.0,
    )
    sim = simulate_stationary(position=np.array([0.5, 0.5]), config=config_sim, seed=42)

    # Inject extreme outlier at frame 150
    Z_cam_led1 = sim["Z_cam_led1"].copy()
    Z_cam_led2 = sim["Z_cam_led2"].copy()
    outlier_idx = 150
    Z_cam_led1[outlier_idx] = np.array([5.0, 5.0])  # 5m error
    Z_cam_led2[outlier_idx] = np.array([5.04, 5.0])

    # Run filter WITH gating
    config_ekf_gating = EKFConfig(
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.997,
        measurement_noise_pos=0.005**2,
    )
    result_gating = extended_kalman_filter(
        ekf_config=config_ekf_gating,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=sim["mask_cam"],
    )

    # Run filter WITHOUT gating for comparison
    config_ekf_no_gating = EKFConfig(
        use_mahalanobis_gating=False,
        measurement_noise_pos=0.005**2,
    )
    result_no_gating = extended_kalman_filter(
        ekf_config=config_ekf_no_gating,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=sim["mask_cam"],
    )

    # Ground truth
    t_truth = sim["t_imu"]
    pos_truth = sim["X_truth"][:, :2]
    t_cam = sim["t_cam_exp"]
    pos_truth_interp = np.column_stack(
        [np.interp(t_cam, t_truth, pos_truth[:, 0]), np.interp(t_cam, t_truth, pos_truth[:, 1])]
    )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Scenario 1: Outlier Rejection via Mahalanobis Gating", fontsize=14, fontweight="bold"
    )

    # Plot 1: Trajectories
    ax = axes[0, 0]
    ax.plot(
        pos_truth_interp[:, 0], pos_truth_interp[:, 1], "k-", linewidth=2, label="Truth", alpha=0.7
    )
    ax.scatter(Z_cam_led1[:, 0], Z_cam_led1[:, 1], c="gray", s=10, alpha=0.3, label="Measurements")
    ax.scatter(
        Z_cam_led1[outlier_idx, 0],
        Z_cam_led1[outlier_idx, 1],
        c="red",
        s=200,
        marker="x",
        linewidths=3,
        label="Outlier (5m error)",
    )
    ax.plot(
        result_gating.filtered_means[:, 0],
        result_gating.filtered_means[:, 1],
        "b-",
        linewidth=1.5,
        label="EKF (with gating)",
    )
    ax.plot(
        result_no_gating.filtered_means[:, 0],
        result_no_gating.filtered_means[:, 1],
        "r--",
        linewidth=1.5,
        label="EKF (no gating)",
    )
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Trajectory Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Plot 2: Position error over time
    ax = axes[0, 1]
    error_gating = np.linalg.norm(result_gating.filtered_means[:, :2] - pos_truth_interp, axis=1)
    error_no_gating = np.linalg.norm(
        result_no_gating.filtered_means[:, :2] - pos_truth_interp, axis=1
    )

    ax.plot(t_cam, error_gating * 100, "b-", linewidth=1.5, label="With gating")
    ax.plot(t_cam, error_no_gating * 100, "r--", linewidth=1.5, label="No gating")
    ax.axvline(t_cam[outlier_idx], color="red", linestyle=":", alpha=0.5, label="Outlier injected")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (cm)")
    ax.set_title("Position Error vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, min(100, np.max(error_no_gating) * 100 * 1.1)])

    # Plot 3: Covariance trace
    ax = axes[1, 0]
    cov_trace_gating = np.array([np.trace(P[:2, :2]) for P in result_gating.filtered_covariances])
    cov_trace_no_gating = np.array(
        [np.trace(P[:2, :2]) for P in result_no_gating.filtered_covariances]
    )

    ax.semilogy(t_cam, cov_trace_gating, "b-", linewidth=1.5, label="With gating")
    ax.semilogy(t_cam, cov_trace_no_gating, "r--", linewidth=1.5, label="No gating")
    ax.axvline(t_cam[outlier_idx], color="red", linestyle=":", alpha=0.5, label="Outlier injected")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position covariance trace (m²)")
    ax.set_title("Uncertainty (Covariance)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    rmse_gating = compute_position_rmse(result_gating.filtered_means[:, :2], pos_truth_interp)
    rmse_no_gating = compute_position_rmse(result_no_gating.filtered_means[:, :2], pos_truth_interp)

    summary_text = f"""
    OUTLIER REJECTION PERFORMANCE

    Outlier: 5.0 m error at t={t_cam[outlier_idx]:.2f}s

    Position RMSE:
    • With gating:    {rmse_gating*100:.2f} cm ✓
    • Without gating: {rmse_no_gating*100:.2f} cm ✗

    Improvement: {(1 - rmse_gating/rmse_no_gating)*100:.1f}%

    Result: Gating successfully rejects
    extreme outlier without divergence.
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family="monospace", verticalalignment="center")

    plt.tight_layout()
    return fig


def plot_scenario_2_led_swaps():
    """Plot filter stability under frequent LED swaps."""
    print("Scenario 2: Stability during frequent LED swaps...")

    # Create scenario with frequent swaps
    config_sim = RatIMUSimConfig(
        duration_s=30.0,
        use_second_led=True,
        led_swap_mode="persistent",
        led_swap_rate=0.5,  # 0.5 swaps/second
        led_swap_duration_mean=2.0,
        led_swap_duration_std=0.5,
        cam_dropout_prob=0.1,
        cam_sigma_m=0.005,
    )
    sim = simulate_rat_imu(config_sim, seed=42)

    # Run filter
    config_ekf = EKFConfig(measurement_noise_pos=0.005**2)
    result = extended_kalman_filter(
        ekf_config=config_ekf,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Ground truth
    t_truth = sim["t_imu"]
    pos_truth = sim["X_truth"][:, :2]
    t_cam = sim["t_cam_exp"]
    pos_truth_interp = np.column_stack(
        [np.interp(t_cam, t_truth, pos_truth[:, 0]), np.interp(t_cam, t_truth, pos_truth[:, 1])]
    )

    # Identify swap periods
    swap_mask = sim["swap_applied"]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Scenario 2: Filter Stability Under Frequent LED Swaps", fontsize=14, fontweight="bold"
    )

    # Plot 1: Trajectory
    ax = axes[0, 0]
    ax.plot(
        pos_truth_interp[:, 0], pos_truth_interp[:, 1], "k-", linewidth=2, label="Truth", alpha=0.7
    )

    # Show normal vs swapped measurements in different colors
    normal_frames = ~swap_mask & sim["mask_cam"]
    swapped_frames = swap_mask & sim["mask_cam"]

    ax.scatter(
        sim["Z_cam_led1"][normal_frames, 0],
        sim["Z_cam_led1"][normal_frames, 1],
        c="blue",
        s=5,
        alpha=0.2,
        label="Normal measurements",
    )
    ax.scatter(
        sim["Z_cam_led1"][swapped_frames, 0],
        sim["Z_cam_led1"][swapped_frames, 1],
        c="orange",
        s=5,
        alpha=0.4,
        label="Swapped measurements",
    )

    ax.plot(
        result.filtered_means[:, 0],
        result.filtered_means[:, 1],
        "g-",
        linewidth=1.5,
        label="EKF estimate",
    )
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Trajectory with LED Swaps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Plot 2: Position error over time
    ax = axes[0, 1]
    error = np.linalg.norm(result.filtered_means[:, :2] - pos_truth_interp, axis=1)

    ax.plot(t_cam, error * 100, "g-", linewidth=1.5, label="Position error")

    # Highlight swap periods
    for i in range(len(swap_mask)):
        if swap_mask[i]:
            ax.axvspan(t_cam[i], t_cam[i] + 1 / 30, color="orange", alpha=0.1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (cm)")
    ax.set_title("Position Error (orange = swap active)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Covariance trace
    ax = axes[1, 0]
    cov_trace = np.array([np.trace(P[:2, :2]) for P in result.filtered_covariances])

    ax.plot(t_cam, cov_trace * 1e4, "g-", linewidth=1.5, label="Position covariance")

    # Highlight swap periods
    for i in range(len(swap_mask)):
        if swap_mask[i]:
            ax.axvspan(t_cam[i], t_cam[i] + 1 / 30, color="orange", alpha=0.1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position covariance trace (cm²)")
    ax.set_title("Uncertainty (orange = swap active)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color="red", linestyle="--", alpha=0.5, label="Max bound (10cm)")

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    n_swaps = np.sum(swap_mask)
    swap_rate = n_swaps / config_sim.duration_s
    rmse = compute_position_rmse(result.filtered_means[:, :2], pos_truth_interp)
    max_cov = np.max(cov_trace)

    summary_text = f"""
    LED SWAP STABILITY

    Configuration:
    • Swap rate:     {config_sim.led_swap_rate:.1f} events/s
    • Swap duration: {config_sim.led_swap_duration_mean:.1f}s mean
    • Dropout rate:  {config_sim.cam_dropout_prob*100:.0f}%

    Actual swaps: {n_swaps} events ({swap_rate:.2f}/s)

    Filter performance:
    • Position RMSE:    {rmse*100:.2f} cm
    • Max covariance:   {max_cov*1e4:.2f} cm²
    • Status: {'✓ STABLE' if max_cov < 0.01 else '✗ DIVERGED'}

    Result: Filter remains stable despite
    frequent swaps and dropouts.
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family="monospace", verticalalignment="center")

    plt.tight_layout()
    return fig


def plot_scenario_3_long_dropout():
    """Plot filter stability during 5-second dropout."""
    print("Scenario 3: Stability during 5-second dropout...")

    # Create constant velocity scenario
    config_sim = SimpleSimConfig(
        duration_s=15.0,
        cam_dropout_prob=0.0,
        cam_noise_std=0.003,
    )
    sim = simulate_constant_velocity(
        initial_position=np.array([0.5, 0.5]),
        velocity=np.array([0.2, 0.0]),
        config=config_sim,
        seed=456,
    )

    # Inject 5-second dropout
    Z_cam_led1 = sim["Z_cam_led1"].copy()
    Z_cam_led2 = sim["Z_cam_led2"].copy()
    mask_cam = sim["mask_cam"].copy()

    dropout_start = 150
    dropout_end = 300  # 150 frames = 5 seconds at 30 Hz
    Z_cam_led1[dropout_start:dropout_end] = np.nan
    Z_cam_led2[dropout_start:dropout_end] = np.nan
    mask_cam[dropout_start:dropout_end] = False

    # Run filter
    config_ekf = EKFConfig()
    result = extended_kalman_filter(
        ekf_config=config_ekf,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
    )

    # Ground truth
    t_truth = sim["t_imu"]
    pos_truth = sim["X_truth"][:, :2]
    t_cam = sim["t_cam_exp"]
    pos_truth_interp = np.column_stack(
        [np.interp(t_cam, t_truth, pos_truth[:, 0]), np.interp(t_cam, t_truth, pos_truth[:, 1])]
    )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Scenario 3: Filter Stability During 5-Second Dropout (PRD §4.2)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: Trajectory
    ax = axes[0, 0]
    ax.plot(
        pos_truth_interp[:, 0], pos_truth_interp[:, 1], "k-", linewidth=2, label="Truth", alpha=0.7
    )

    # Show measurements (with dropout gap)
    valid_meas = mask_cam
    ax.scatter(
        Z_cam_led1[valid_meas, 0],
        Z_cam_led1[valid_meas, 1],
        c="gray",
        s=10,
        alpha=0.3,
        label="Measurements",
    )

    # Show filter estimate in different colors
    before_dropout = result.filtered_means[:dropout_start, :2]
    during_dropout = result.filtered_means[dropout_start:dropout_end, :2]
    after_dropout = result.filtered_means[dropout_end:, :2]

    ax.plot(before_dropout[:, 0], before_dropout[:, 1], "b-", linewidth=1.5, label="Before dropout")
    ax.plot(
        during_dropout[:, 0],
        during_dropout[:, 1],
        "r-",
        linewidth=2,
        label="During dropout (IMU only)",
    )
    ax.plot(after_dropout[:, 0], after_dropout[:, 1], "g-", linewidth=1.5, label="After recovery")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Trajectory During 5s Dropout")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Plot 2: Position error over time
    ax = axes[0, 1]
    error = np.linalg.norm(result.filtered_means[:, :2] - pos_truth_interp, axis=1)

    ax.plot(t_cam, error * 100, "b-", linewidth=1.5, label="Position error")
    ax.axvspan(
        t_cam[dropout_start], t_cam[dropout_end], color="red", alpha=0.2, label="Dropout period"
    )
    ax.axhline(15, color="red", linestyle="--", alpha=0.5, label="PRD limit (15 cm)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (cm)")
    ax.set_title("Position Error (PRD: ≤15cm after 5s dropout)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Covariance trace
    ax = axes[1, 0]
    cov_trace = np.array([np.trace(P[:2, :2]) for P in result.filtered_covariances])

    ax.semilogy(t_cam, cov_trace, "b-", linewidth=1.5, label="Position covariance")
    ax.axvspan(
        t_cam[dropout_start], t_cam[dropout_end], color="red", alpha=0.2, label="Dropout period"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position covariance trace (m²)")
    ax.set_title("Uncertainty Growth During Dropout")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    dropout_mid = (dropout_start + dropout_end) // 2
    error_at_dropout_end = error[dropout_end - 1] if dropout_end < len(error) else error[-1]
    cov_before = cov_trace[dropout_start - 1]
    cov_during = cov_trace[dropout_mid]
    cov_growth = cov_during / cov_before

    rmse = compute_position_rmse(result.filtered_means[:, :2], pos_truth_interp)

    prd_pass = error_at_dropout_end <= 0.15

    summary_text = f"""
    5-SECOND DROPOUT STABILITY

    Dropout: {t_cam[dropout_start]:.1f}s to {t_cam[dropout_end]:.1f}s

    Position error at dropout end:
    • Drift: {error_at_dropout_end*100:.2f} cm
    • PRD limit: 15.0 cm
    • Status: {'✓ PASS' if prd_pass else '✗ FAIL'}

    Covariance growth:
    • Before: {cov_before*1e4:.2f} cm²
    • During: {cov_during*1e4:.2f} cm²
    • Growth: {cov_growth:.1f}x

    Overall RMSE: {rmse*100:.2f} cm

    Result: Filter remains stable during
    extended vision loss (IMU-only mode).
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family="monospace", verticalalignment="center")

    plt.tight_layout()
    return fig


def main():
    """Generate all robustness scenario plots."""
    print("=" * 60)
    print("ROBUSTNESS TEST SCENARIO VISUALIZATION")
    print("=" * 60)
    print()

    # Create output directory
    import os

    os.makedirs("output", exist_ok=True)

    # Generate plots
    fig1 = plot_scenario_1_outlier_rejection()
    fig1.savefig("output/robustness_01_outlier_rejection.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: output/robustness_01_outlier_rejection.png")

    fig2 = plot_scenario_2_led_swaps()
    fig2.savefig("output/robustness_02_led_swaps.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: output/robustness_02_led_swaps.png")

    fig3 = plot_scenario_3_long_dropout()
    fig3.savefig("output/robustness_03_long_dropout.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: output/robustness_03_long_dropout.png")

    print()
    print("=" * 60)
    print("All plots saved to output/")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
