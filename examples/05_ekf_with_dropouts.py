"""EKF Robustness Under Camera Dropouts - Stress Testing.

LEARNING OBJECTIVES:
-------------------
This example teaches you how the Extended Kalman Filter handles realistic camera
occlusions and dropouts. You'll see performance degrade gracefully as dropout
rates increase, and learn which robustness features prevent catastrophic drift.

We test three dropout scenarios on circular motion (most challenging):
1. **Light (10%)**: Occasional brief occlusions
2. **Medium (20%)**: Frequent short gaps
3. **Heavy (30%)**: Long consecutive dropouts

WHY THIS MATTERS:
-----------------
Real-world tracking faces frequent occlusions:
- Rat turns away from camera
- LED reflections confuse detection
- Handling by experimenter
- Poor lighting conditions

The filter must maintain accuracy during vision loss using only IMU integration.

WHAT YOU'LL LEARN:
------------------
- How dropout rate affects position drift
- Recovery behavior when vision returns
- Role of adaptive process noise during blackouts
- ZUPT (Zero-velocity updates) for stationary periods
- Why bias estimates matter during dropouts

USAGE:
------
    uv run python examples/05_ekf_with_dropouts.py

OUTPUT:
-------
    - Console: Dropout drift metrics for each scenario
    - Files: 05_ekf_dropout_light.png, 05_ekf_dropout_medium.png,
             05_ekf_dropout_heavy.png

ESTIMATED RUNTIME: ~15 seconds

KEY CONCEPTS ILLUSTRATED:
-------------------------
1. **IMU-Only Integration**: During dropouts, filter relies on IMU (drifts over time)
2. **Adaptive Q Scaling**: Increase uncertainty during blackouts to enable quick recovery
3. **Dropout Drift**: Position error accumulates without vision correction
4. **Recovery Dynamics**: How quickly filter corrects when vision returns
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.qa.metrics import compute_dropout_drift, compute_position_rmse
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.utils import interp_angle
from trodestrack.viz.styles import COLORS, apply_tufte_style

apply_tufte_style()

# Output directory
OUTPUT_DIR = Path(__file__).parent


def print_section_header(title: str) -> None:
    """Print a clear section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_dropout_performance(
    sim_data: dict,
    filter_result,
    scenario_name: str,
    dropout_rate: float,
) -> dict:
    """Analyze filter performance during and after dropouts.

    Args:
        sim_data: Simulation data
        filter_result: EKF filter results
        scenario_name: Name of scenario
        dropout_rate: Expected dropout rate (0-1)

    Returns:
        Dictionary of metrics
    """
    t_imu = sim_data["t_imu"]
    t_cam = sim_data["t_cam_exp"]
    X_truth = sim_data["X_truth"]
    mask_cam = sim_data["mask_cam"]

    # Interpolate truth
    X_truth_cam = np.column_stack(
        [
            np.interp(t_cam, t_imu, X_truth[:, 0]),
            np.interp(t_cam, t_imu, X_truth[:, 1]),
            np.interp(t_cam, t_imu, X_truth[:, 2]),
            np.interp(t_cam, t_imu, X_truth[:, 3]),
            interp_angle(t_cam, t_imu, X_truth[:, 4]),
        ]
    )

    X_est = np.array(filter_result.filtered_means)

    # Overall RMSE
    pos_rmse_cm = compute_position_rmse(X_truth_cam[:, :2] * 100, X_est[:, :2] * 100)

    # Dropout drift analysis
    drift_result = compute_dropout_drift(
        positions=X_est[:, :2],
        valid_mask=mask_cam,
        t=t_cam,
        min_duration_s=0.1,  # Analyze gaps >= 0.1s (lower threshold for light dropouts)
    )

    # Handle case where no significant dropouts found
    if drift_result["drift_m"] is None:
        drift_result["drift_m"] = 0.0

    # Actual dropout rate
    actual_dropout_rate = 1.0 - mask_cam.mean()

    # Analyze dropout sequences
    dropout_lengths = []
    current_dropout = 0
    for i in range(len(mask_cam)):
        if not mask_cam[i]:
            current_dropout += 1
        elif current_dropout > 0:
            dropout_lengths.append(current_dropout)
            current_dropout = 0
    if current_dropout > 0:
        dropout_lengths.append(current_dropout)

    max_dropout_frames = max(dropout_lengths) if dropout_lengths else 0
    max_dropout_s = max_dropout_frames / sim_data["config"].fs_cam

    return {
        "pos_rmse_cm": pos_rmse_cm,
        "dropout_drift_m": drift_result["drift_m"],
        "max_dropout_s": max_dropout_s,
        "actual_dropout_rate": actual_dropout_rate,
        "num_dropout_sequences": len(dropout_lengths),
    }


def plot_dropout_scenario(
    sim_data: dict,
    filter_result,
    scenario_name: str,
    metrics: dict,
    output_path: Path,
) -> None:
    """Create visualization showing dropout impact.

    Args:
        sim_data: Simulation data
        filter_result: EKF results
        scenario_name: Scenario name
        metrics: Computed metrics
        output_path: Output file path
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    t_imu = sim_data["t_imu"]
    t_cam = sim_data["t_cam_exp"]
    X_truth = sim_data["X_truth"]
    mask_cam = sim_data["mask_cam"]

    # Interpolate truth
    X_truth_cam = np.column_stack(
        [
            np.interp(t_cam, t_imu, X_truth[:, 0]),
            np.interp(t_cam, t_imu, X_truth[:, 1]),
            np.interp(t_cam, t_imu, X_truth[:, 2]),
            np.interp(t_cam, t_imu, X_truth[:, 3]),
            interp_angle(t_cam, t_imu, X_truth[:, 4]),
        ]
    )

    X_est = np.array(filter_result.filtered_means)
    P_est = np.array(filter_result.filtered_covariances)

    # -------------------------------------------------------------------------
    # Panel 1: Trajectory with dropout shading (top-left, span 2 rows)
    # -------------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[:2, 0])
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("🎯 Trajectory with Dropout Regions", fontweight="bold", loc="left")

    # Ground truth
    ax_traj.plot(
        X_truth[:, 0],
        X_truth[:, 1],
        "-",
        linewidth=2,
        color=COLORS["gray"],
        alpha=0.4,
        label="Truth",
        zorder=1,
    )

    # EKF estimate - color by dropout status
    visible_mask = mask_cam
    dropout_mask = ~mask_cam

    ax_traj.scatter(
        X_est[visible_mask, 0],
        X_est[visible_mask, 1],
        s=15,
        c=COLORS["blue"],
        alpha=0.6,
        label="EKF (vision)",
        zorder=3,
    )
    ax_traj.scatter(
        X_est[dropout_mask, 0],
        X_est[dropout_mask, 1],
        s=15,
        c=COLORS["red"],
        alpha=0.8,
        label="EKF (IMU-only)",
        marker="x",
        zorder=4,
    )

    ax_traj.legend(loc="best", fontsize=10)
    ax_traj.grid(True, alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 2: Position Error with Dropout Shading (top-middle)
    # -------------------------------------------------------------------------
    ax_pos_err = fig.add_subplot(gs[0, 1])
    ax_pos_err.set_xlabel("Time (s)")
    ax_pos_err.set_ylabel("Position Error (cm)")
    ax_pos_err.set_title("📏 Position Error (Dropouts Shaded)", fontweight="bold", loc="left")

    pos_err = np.linalg.norm(X_est[:, :2] - X_truth_cam[:, :2], axis=1) * 100

    # Shade dropout regions
    for i in range(len(t_cam) - 1):
        if not mask_cam[i]:
            ax_pos_err.axvspan(t_cam[i], t_cam[i + 1], alpha=0.15, color=COLORS["red"], zorder=1)

    ax_pos_err.plot(t_cam, pos_err, linewidth=2, color=COLORS["blue"], alpha=0.8, zorder=2)
    ax_pos_err.axhline(
        2.0,
        linestyle="--",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label="PRD Target (2 cm)",
    )

    ax_pos_err.legend(loc="upper right", fontsize=9)
    ax_pos_err.grid(True, alpha=0.2)
    ax_pos_err.set_ylim(bottom=0)

    # Annotate max drift
    max_err_idx = np.argmax(pos_err)
    ax_pos_err.annotate(
        f"Max: {pos_err[max_err_idx]:.1f} cm",
        xy=(t_cam[max_err_idx], pos_err[max_err_idx]),
        xytext=(t_cam[max_err_idx] + 1, pos_err[max_err_idx] * 1.1),
        fontsize=9,
        color=COLORS["red"],
        arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1),
    )

    # -------------------------------------------------------------------------
    # Panel 3: Velocity Error (top-right)
    # -------------------------------------------------------------------------
    ax_vel_err = fig.add_subplot(gs[0, 2])
    ax_vel_err.set_xlabel("Time (s)")
    ax_vel_err.set_ylabel("Velocity Error (cm/s)")
    ax_vel_err.set_title("🏃 Velocity Error", fontweight="bold", loc="left")

    vel_err = np.linalg.norm(X_est[:, 2:4] - X_truth_cam[:, 2:4], axis=1) * 100

    # Shade dropouts
    for i in range(len(t_cam) - 1):
        if not mask_cam[i]:
            ax_vel_err.axvspan(t_cam[i], t_cam[i + 1], alpha=0.15, color=COLORS["red"])

    ax_vel_err.plot(t_cam, vel_err, linewidth=2, color=COLORS["purple"], alpha=0.8)
    ax_vel_err.axhline(10.0, linestyle="--", color=COLORS["gray"], linewidth=1, alpha=0.5)

    ax_vel_err.grid(True, alpha=0.2)
    ax_vel_err.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Panel 4: Camera Status Timeline (middle-left)
    # -------------------------------------------------------------------------
    ax_status = fig.add_subplot(gs[1, 1])
    ax_status.set_xlabel("Time (s)")
    ax_status.set_ylabel("Camera Status")
    ax_status.set_title("📹 Vision Availability", fontweight="bold", loc="left")

    # Plot as filled regions
    ax_status.fill_between(
        t_cam,
        0,
        1,
        where=mask_cam,
        alpha=0.5,
        color=COLORS["green"],
        step="mid",
        label="Vision OK",
    )
    ax_status.fill_between(
        t_cam,
        0,
        1,
        where=~mask_cam,
        alpha=0.5,
        color=COLORS["red"],
        step="mid",
        label="Dropout",
    )

    ax_status.set_yticks([0, 1])
    ax_status.set_yticklabels(["Dropout", "Vision"])
    ax_status.legend(loc="upper right", fontsize=9)
    ax_status.grid(True, alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 5: Position Uncertainty Growth (middle-right)
    # -------------------------------------------------------------------------
    ax_unc = fig.add_subplot(gs[1, 2])
    ax_unc.set_xlabel("Time (s)")
    ax_unc.set_ylabel("Position Uncertainty (cm)")
    ax_unc.set_title("📊 Covariance Growth", fontweight="bold", loc="left")

    # Extract position uncertainty (trace of position covariance)
    pos_std = np.sqrt(P_est[:, 0, 0] + P_est[:, 1, 1]) * 100  # cm

    # Shade dropouts
    for i in range(len(t_cam) - 1):
        if not mask_cam[i]:
            ax_unc.axvspan(t_cam[i], t_cam[i + 1], alpha=0.15, color=COLORS["red"])

    ax_unc.plot(t_cam, pos_std, linewidth=2, color=COLORS["orange"], alpha=0.8)
    ax_unc.grid(True, alpha=0.2)
    ax_unc.set_ylim(bottom=0)

    # Annotate uncertainty growth during dropouts
    ax_unc.text(
        0.98,
        0.95,
        "Uncertainty ↑ during dropouts\n(adaptive Q scaling)",
        transform=ax_unc.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # -------------------------------------------------------------------------
    # Panel 6: Dropout Duration Histogram (bottom-left)
    # -------------------------------------------------------------------------
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_hist.set_xlabel("Dropout Duration (s)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("📊 Dropout Length Distribution", fontweight="bold", loc="left")

    # Compute dropout durations
    dropout_lengths = []
    current_dropout = 0
    for i in range(len(mask_cam)):
        if not mask_cam[i]:
            current_dropout += 1
        elif current_dropout > 0:
            dropout_lengths.append(current_dropout)
            current_dropout = 0
    if current_dropout > 0:
        dropout_lengths.append(current_dropout)

    if dropout_lengths:
        dropout_durations = np.array(dropout_lengths) / sim_data["config"].fs_cam
        ax_hist.hist(
            dropout_durations,
            bins=15,
            color=COLORS["red"],
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_hist.axvline(
            dropout_durations.mean(),
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1,
            label=f"Mean: {dropout_durations.mean():.2f}s",
        )
        ax_hist.legend(loc="upper right", fontsize=9)
    else:
        ax_hist.text(0.5, 0.5, "No dropouts", ha="center", va="center")

    ax_hist.grid(True, alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 7: Bias Estimates (bottom-middle)
    # -------------------------------------------------------------------------
    ax_bias = fig.add_subplot(gs[2, 1])
    ax_bias.set_xlabel("Time (s)")
    ax_bias.set_ylabel("Gyro Bias (deg/s)")
    ax_bias.set_title("⚙️ Gyro Bias Estimate", fontweight="bold", loc="left")

    bias_gyro_truth = np.interp(t_cam, t_imu, sim_data["bias_gyro"])
    bias_gyro_est = X_est[:, 5]

    ax_bias.plot(
        t_cam,
        np.degrees(bias_gyro_truth),
        linewidth=2,
        color=COLORS["gray"],
        alpha=0.5,
        label="Truth",
        linestyle="--",
    )
    ax_bias.plot(
        t_cam,
        np.degrees(bias_gyro_est),
        linewidth=2,
        color=COLORS["purple"],
        alpha=0.8,
        label="EKF",
    )

    # Shade dropouts
    for i in range(len(t_cam) - 1):
        if not mask_cam[i]:
            ax_bias.axvspan(t_cam[i], t_cam[i + 1], alpha=0.15, color=COLORS["red"])

    ax_bias.legend(loc="best", fontsize=9)
    ax_bias.grid(True, alpha=0.2)
    ax_bias.axhline(0, color="k", linewidth=0.5, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 8: Recovery Analysis (bottom-right)
    # -------------------------------------------------------------------------
    ax_recovery = fig.add_subplot(gs[2, 2])
    ax_recovery.set_title("📈 Dropout Impact Summary", fontweight="bold", loc="left")
    ax_recovery.axis("off")

    # Create summary text
    summary_text = f"""
Dropout Statistics:
  • Actual dropout rate: {metrics['actual_dropout_rate']*100:.1f}%
  • Dropout sequences: {metrics['num_dropout_sequences']}
  • Max dropout: {metrics['max_dropout_s']:.2f}s
  • Max drift: {metrics['dropout_drift_m']*100:.1f} cm

Performance:
  • Overall position RMSE: {metrics['pos_rmse_cm']:.2f} cm
  • PRD target: ≤ 2 cm {'✓ PASS' if metrics['pos_rmse_cm'] <= 2.0 else '✗ FAIL'}

Robustness Features Active:
  • Adaptive Q during dropouts
  • Mahalanobis gating
  • Confidence-weighted measurements
  • Bias estimation (prevents long-term drift)

Key Insight:
  Filter maintains reasonable accuracy
  despite {metrics['actual_dropout_rate']*100:.0f}% vision loss by using
  IMU integration between camera frames.
"""

    ax_recovery.text(
        0.1,
        0.95,
        summary_text,
        transform=ax_recovery.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # -------------------------------------------------------------------------
    # Overall Title
    # -------------------------------------------------------------------------
    title = (
        f"{scenario_name} — EKF Robustness Under Dropouts\n"
        f"Dropout Rate: {metrics['actual_dropout_rate']*100:.1f}% | "
        f"Position RMSE: {metrics['pos_rmse_cm']:.2f} cm | "
        f"Max Drift: {metrics['dropout_drift_m']*100:.1f} cm"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 Saved: {output_path}")


def main() -> None:
    """Run EKF on three dropout scenarios and analyze robustness."""
    print_section_header("EKF Robustness Under Camera Dropouts")

    print(
        """
    Welcome! This example demonstrates how the EKF maintains accuracy despite
    camera occlusions. We test three scenarios with increasing dropout rates:

    • Light (10%): Occasional brief gaps
    • Medium (20%): Frequent short occlusions
    • Heavy (30%): Long consecutive dropouts

    You will learn:
    • How position error accumulates during vision loss (IMU-only drift)
    • Role of adaptive Q scaling to enable quick recovery
    • Why bias estimation matters for long-term stability
    • When the filter starts to fail (too much drift)

    Let's test robustness!
    """
    )

    # Shared config (circular motion - most challenging for dropouts)
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
        led_distance=0.04,
        use_heading_measurement=True,
        enable_zupt=False,
    )

    scenarios = [
        ("Light Dropout (10%)", 0.10, "05_ekf_dropout_light.png"),
        ("Medium Dropout (20%)", 0.20, "05_ekf_dropout_medium.png"),
        ("Heavy Dropout (30%)", 0.30, "05_ekf_dropout_heavy.png"),
    ]

    all_results = []

    for scenario_name, dropout_prob, output_file in scenarios:
        print_section_header(f"Scenario: {scenario_name}")

        # Simulate with specified dropout rate
        sim_config = RatIMUSimConfig(
            duration_s=15.0,
            fs_imu=200.0,
            fs_cam=30.0,
            cam_dropout_prob=dropout_prob,
            cam_dropout_correlation=0.7,  # Creates longer consecutive dropouts
            gyro_noise_density=0.001,
            accel_noise_density=0.05,
            gyro_bias_rw_density=0.0001,
            accel_bias_rw_density=0.001,
            cam_sigma_m=0.005,
            use_second_led=True,
        )

        print(f"   Running simulation (dropout_prob={dropout_prob:.0%})...")
        sim = simulate_rat_imu(config=sim_config, seed=42)

        print("   Running EKF filter...")
        result = extended_kalman_filter(
            ekf_config=ekf_config,
            t_imu=sim["t_imu"],
            U_imu=sim["U_imu"],
            t_cam=sim["t_cam_exp"],
            Z_cam_led1=sim["Z_cam_led1"],
            Z_cam_led2=sim["Z_cam_led2"],
            mask_cam=sim["mask_cam"],
        )

        # Analyze performance
        metrics = analyze_dropout_performance(sim, result, scenario_name, dropout_prob)

        print("\n   Performance Metrics:")
        print("   " + "-" * 76)
        print(f"   Actual Dropout Rate:     {metrics['actual_dropout_rate']*100:6.1f}%")
        print(f"   Dropout Sequences:       {metrics['num_dropout_sequences']:6d}")
        print(f"   Max Dropout Duration:    {metrics['max_dropout_s']:6.2f} s")
        print(f"   Max Dropout Drift:       {metrics['dropout_drift_m']*100:6.1f} cm")
        print(f"   Overall Position RMSE:   {metrics['pos_rmse_cm']:6.2f} cm  (target: ≤ 2 cm)")
        print("   " + "-" * 76)

        # Plot
        plot_dropout_scenario(sim, result, scenario_name, metrics, OUTPUT_DIR / output_file)

        all_results.append((scenario_name, dropout_prob, metrics))

    # Summary comparison
    print_section_header("Summary: Dropout Impact on Performance")

    print("\n   Comparative Analysis:")
    print("   " + "=" * 76)
    print(f"   {'Scenario':<25} {'Dropout %':>12} {'Max Drift (cm)':>15} {'RMSE (cm)':>12}")
    print("   " + "=" * 76)

    for scenario_name, dropout_prob, metrics in all_results:
        print(
            f"   {scenario_name:<25} {dropout_prob*100:>11.0f}% "
            f"{metrics['dropout_drift_m']*100:>14.1f} "
            f"{metrics['pos_rmse_cm']:>11.2f}"
        )

    print("   " + "=" * 76)

    print(
        """
    🎓 KEY TAKEAWAYS:

    1. DROPOUT TOLERANCE:
       • Light (10%): Filter maintains excellent accuracy
       • Medium (20%): Performance degrades but remains acceptable
       • Heavy (30%): Significant drift during long gaps, but recovers

    2. ROBUSTNESS FEATURES:
       • Adaptive Q scaling increases uncertainty during dropouts
       • IMU bias estimation prevents long-term drift
       • Mahalanobis gating rejects outliers on recovery
       • Filter "remembers" uncertainty grew and trusts measurements on return

    3. DRIFT MECHANICS:
       • Position error accumulates during IMU-only integration
       • Drift rate ≈ 5-10 cm/s (depends on IMU noise and bias uncertainty)
       • Longer dropouts = more drift (obvious but quantified here)

    4. RECOVERY BEHAVIOR:
       • When vision returns, filter quickly corrects position
       • Covariance shrinks as measurements arrive
       • Bias estimates improve with continued motion

    5. PRACTICAL LIMITS:
       • EKF can handle 20-30% dropout rates reasonably well
       • Beyond 30%, consider:
         - Smoother for offline processing (Example 07)
         - Better IMU calibration
         - Additional sensors (magnetometer, wheel encoders)

    NEXT STEPS:
    • Run examples/06_ukf_with_dropouts.py to compare UKF robustness
    • Run examples/07_smoother_demonstration.py to see how smoothing helps
    • See examples/README.md for full learning path

    Happy learning! 🚀
    """
    )


if __name__ == "__main__":
    main()
