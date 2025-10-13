"""Unscented Kalman Filter (UKF) vs Extended Kalman Filter (EKF) - Comparative Study.

LEARNING OBJECTIVES:
-------------------
This example teaches you the differences between UKF and EKF on the same three
fundamental scenarios you saw in example 03:

1. **Stationary**: Does UKF's nonlinear handling help when there's no motion?
2. **Constant Velocity**: How do the filters compare on simple linear motion?
3. **Circular Motion**: Where nonlinearity matters most, does UKF outperform EKF?

WHY THIS MATTERS:
-----------------
The Extended Kalman Filter (EKF) linearizes nonlinear functions using Jacobians.
The Unscented Kalman Filter (UKF) propagates sigma points through the actual
nonlinear functions. This example helps you understand:

• When the extra computational cost of UKF is justified
• How sigma-point transforms capture nonlinearity better than linearization
• The tradeoff between accuracy and computational efficiency

KEY QUESTIONS WE'LL ANSWER:
---------------------------
1. Is UKF more accurate than EKF on these scenarios?
2. How much slower is UKF (computational cost)?
3. Does UKF's improved nonlinear handling translate to better consistency (NEES)?
4. When should you choose UKF over EKF in practice?

WHAT YOU'LL LEARN:
------------------
- How to configure and run UKF with appropriate sigma-point parameters
- How to compare two filters on the same data (apples-to-apples)
- When UKF's advantages outweigh its computational cost
- How to interpret comparative metrics (speedup factors, accuracy ratios)

USAGE:
------
    uv run python examples/04_ukf_basic_scenarios.py

OUTPUT:
-------
    - Console: Side-by-side EKF vs UKF metrics tables with comparisons
    - Files: 04_comparison_stationary.png, 04_comparison_constant_velocity.png,
             04_comparison_circular.png

ESTIMATED RUNTIME: ~10-15 seconds (UKF is slower than EKF)

KEY CONCEPTS ILLUSTRATED:
-------------------------
1. **Sigma-Point Transforms**: UKF propagates (2n+1) = 17 points vs EKF's 1 point
2. **Linearization Error**: Where EKF's Jacobian approximation breaks down
3. **Computational Tradeoff**: UKF ≈ 3-5× slower but can be more accurate
4. **Consistency**: NEES tells us if the filter "knows what it doesn't know"
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.qa.metrics import compute_nees, compute_position_rmse
from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)
from trodestrack.sim.utils import interp_angle
from trodestrack.viz.styles import COLORS, apply_tufte_style

# Apply clean visualization style
apply_tufte_style()

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Pedagogical Helper Functions
# =============================================================================


def print_section_header(title: str) -> None:
    """Print a clear section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_comparison_table(
    scenario_name: str,
    ekf_metrics: dict,
    ukf_metrics: dict,
    ekf_time_s: float,
    ukf_time_s: float,
) -> None:
    """Print side-by-side comparison of EKF vs UKF performance.

    Args:
        scenario_name: Name of the scenario
        ekf_metrics: Dictionary of EKF metrics
        ukf_metrics: Dictionary of UKF metrics
        ekf_time_s: EKF computation time in seconds
        ukf_time_s: UKF computation time in seconds
    """
    print(f"\n{scenario_name} Performance Comparison:")
    print("-" * 80)
    print(f"{'Metric':<25} {'EKF':>12} {'UKF':>12} {'Winner':>12} {'Improvement':>15}")
    print("-" * 80)

    # Position RMSE
    ekf_pos = ekf_metrics["pos_rmse_cm"]
    ukf_pos = ukf_metrics["pos_rmse_cm"]
    pos_winner = "UKF" if ukf_pos < ekf_pos else "EKF" if ekf_pos < ukf_pos else "TIE"
    pos_improv = f"{abs(ekf_pos - ukf_pos):.2f} cm" if pos_winner != "TIE" else "-"
    print(
        f"{'Position RMSE (cm)':<25} {ekf_pos:>12.2f} {ukf_pos:>12.2f} "
        f"{pos_winner:>12} {pos_improv:>15}"
    )

    # Velocity RMSE
    ekf_vel = ekf_metrics["vel_rmse_cm_s"]
    ukf_vel = ukf_metrics["vel_rmse_cm_s"]
    vel_winner = "UKF" if ukf_vel < ekf_vel else "EKF" if ekf_vel < ukf_vel else "TIE"
    vel_improv = f"{abs(ekf_vel - ukf_vel):.2f} cm/s" if vel_winner != "TIE" else "-"
    print(
        f"{'Velocity RMSE (cm/s)':<25} {ekf_vel:>12.2f} {ukf_vel:>12.2f} "
        f"{vel_winner:>12} {vel_improv:>15}"
    )

    # Heading RMSE
    ekf_heading = ekf_metrics["heading_rmse_deg"]
    ukf_heading = ukf_metrics["heading_rmse_deg"]
    heading_winner = (
        "UKF" if ukf_heading < ekf_heading else "EKF" if ekf_heading < ukf_heading else "TIE"
    )
    heading_improv = f"{abs(ekf_heading - ukf_heading):.2f}°" if heading_winner != "TIE" else "-"
    print(
        f"{'Heading RMSE (deg)':<25} {ekf_heading:>12.2f} {ukf_heading:>12.2f} "
        f"{heading_winner:>12} {heading_improv:>15}"
    )

    # Mean NEES
    ekf_nees = ekf_metrics["mean_nees"]
    ukf_nees = ukf_metrics["mean_nees"]
    nees_target = 8.0
    ekf_nees_err = abs(ekf_nees - nees_target)
    ukf_nees_err = abs(ukf_nees - nees_target)
    nees_winner = "UKF" if ukf_nees_err < ekf_nees_err else "EKF"
    print(
        f"{'Mean NEES (ideal=8.0)':<25} {ekf_nees:>12.2f} {ukf_nees:>12.2f} "
        f"{nees_winner:>12} {'closer to 8.0':>15}"
    )

    # Computation time
    speedup = ukf_time_s / ekf_time_s
    print("-" * 80)
    print(
        f"{'Computation Time (ms)':<25} {ekf_time_s*1000:>12.1f} {ukf_time_s*1000:>12.1f} "
        f"{'EKF':>12} {f'{speedup:.1f}x slower':>15}"
    )
    print("-" * 80)

    # Overall assessment
    accuracy_wins = sum(
        [
            ukf_pos < ekf_pos,
            ukf_vel < ekf_vel,
            ukf_heading < ekf_heading,
            ukf_nees_err < ekf_nees_err,
        ]
    )
    if accuracy_wins >= 3:
        print(
            f"  ✓ UKF wins {accuracy_wins}/4 metrics → UKF is more accurate "
            f"(at {speedup:.1f}× computational cost)"
        )
    elif accuracy_wins <= 1:
        print(
            f"  ✓ EKF wins {4-accuracy_wins}/4 metrics → EKF is sufficient "
            f"(and {speedup:.1f}× faster)"
        )
    else:
        print(
            f"  ≈ Mixed results ({accuracy_wins}/4 to UKF) → Similar performance, "
            f"use EKF for speed"
        )


def explain_scenario(scenario_name: str, nonlinearity_level: str, expected_winner: str) -> None:
    """Print pedagogical explanation of expected filter behavior.

    Args:
        scenario_name: Name of the scenario
        nonlinearity_level: How nonlinear is this scenario
        expected_winner: Which filter is expected to perform better
    """
    print(f"\n📚 {scenario_name} Scenario")
    print(f"   Nonlinearity: {nonlinearity_level}")
    print(f"   Expected: {expected_winner}")


# =============================================================================
# Visualization Function
# =============================================================================


def plot_comparison(
    sim_data: dict,
    ekf_result,
    ukf_result,
    scenario_name: str,
    ekf_metrics: dict,
    ukf_metrics: dict,
    output_path: Path,
) -> None:
    """Create side-by-side comparison visualization of EKF vs UKF.

    Args:
        sim_data: Simulation data
        ekf_result: EKF filter results
        ukf_result: UKF filter results
        scenario_name: Scenario name
        ekf_metrics: EKF metrics dictionary
        ukf_metrics: UKF metrics dictionary
        output_path: Output file path
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    t_imu = sim_data["t_imu"]
    t_cam = sim_data["t_cam_exp"]
    X_truth = sim_data["X_truth"]

    # Interpolate truth to camera times (full 8D state)
    X_truth_cam = np.column_stack(
        [
            np.interp(t_cam, t_imu, X_truth[:, 0]),
            np.interp(t_cam, t_imu, X_truth[:, 1]),
            np.interp(t_cam, t_imu, X_truth[:, 2]),
            np.interp(t_cam, t_imu, X_truth[:, 3]),
            interp_angle(t_cam, t_imu, X_truth[:, 4]),
            np.interp(t_cam, t_imu, sim_data["bias_gyro"]),
            np.interp(t_cam, t_imu, sim_data["bias_accel_x"]),
            np.interp(t_cam, t_imu, sim_data["bias_accel_y"]),
        ]
    )

    X_ekf = np.array(ekf_result.filtered_means)
    X_ukf = np.array(ukf_result.filtered_means)

    # -------------------------------------------------------------------------
    # Panel 1: 2D Trajectory Comparison (top-left, span 2 rows)
    # -------------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[:2, 0])
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("🎯 Trajectory: EKF vs UKF", fontweight="bold", loc="left")

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

    # EKF estimate
    ax_traj.plot(
        X_ekf[:, 0],
        X_ekf[:, 1],
        "-",
        linewidth=2.5,
        color=COLORS["blue"],
        alpha=0.7,
        label="EKF",
        zorder=2,
    )

    # UKF estimate
    ax_traj.plot(
        X_ukf[:, 0],
        X_ukf[:, 1],
        "--",
        linewidth=2.5,
        color=COLORS["red"],
        alpha=0.7,
        label="UKF",
        zorder=3,
    )

    ax_traj.legend(loc="best", fontsize=10)
    ax_traj.grid(True, alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 2: Position Error Comparison (top-middle)
    # -------------------------------------------------------------------------
    ax_pos_err = fig.add_subplot(gs[0, 1])
    ax_pos_err.set_xlabel("Time (s)")
    ax_pos_err.set_ylabel("Position Error (cm)")
    ax_pos_err.set_title("📏 Position Error Comparison", fontweight="bold", loc="left")

    ekf_pos_err = np.linalg.norm(X_ekf[:, :2] - X_truth_cam[:, :2], axis=1) * 100  # cm
    ukf_pos_err = np.linalg.norm(X_ukf[:, :2] - X_truth_cam[:, :2], axis=1) * 100  # cm

    ax_pos_err.plot(t_cam, ekf_pos_err, linewidth=2, color=COLORS["blue"], alpha=0.7, label="EKF")
    ax_pos_err.plot(
        t_cam,
        ukf_pos_err,
        linewidth=2,
        color=COLORS["red"],
        alpha=0.7,
        linestyle="--",
        label="UKF",
    )
    ax_pos_err.axhline(
        2.0,
        linestyle=":",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label="PRD (2 cm)",
    )

    ax_pos_err.legend(loc="upper right", fontsize=9)
    ax_pos_err.grid(True, alpha=0.2)
    ax_pos_err.set_ylim(bottom=0)

    # Annotate RMSEs
    ax_pos_err.text(
        0.02,
        0.98,
        f"EKF: {ekf_metrics['pos_rmse_cm']:.2f} cm\nUKF: {ukf_metrics['pos_rmse_cm']:.2f} cm",
        transform=ax_pos_err.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # -------------------------------------------------------------------------
    # Panel 3: Velocity Error Comparison (top-right)
    # -------------------------------------------------------------------------
    ax_vel_err = fig.add_subplot(gs[0, 2])
    ax_vel_err.set_xlabel("Time (s)")
    ax_vel_err.set_ylabel("Velocity Error (cm/s)")
    ax_vel_err.set_title("🏃 Velocity Error Comparison", fontweight="bold", loc="left")

    ekf_vel_err = np.linalg.norm(X_ekf[:, 2:4] - X_truth_cam[:, 2:4], axis=1) * 100  # cm/s
    ukf_vel_err = np.linalg.norm(X_ukf[:, 2:4] - X_truth_cam[:, 2:4], axis=1) * 100  # cm/s

    ax_vel_err.plot(t_cam, ekf_vel_err, linewidth=2, color=COLORS["blue"], alpha=0.7, label="EKF")
    ax_vel_err.plot(
        t_cam,
        ukf_vel_err,
        linewidth=2,
        color=COLORS["red"],
        alpha=0.7,
        linestyle="--",
        label="UKF",
    )
    ax_vel_err.axhline(
        10.0,
        linestyle=":",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label="PRD (10 cm/s)",
    )

    ax_vel_err.legend(loc="upper right", fontsize=9)
    ax_vel_err.grid(True, alpha=0.2)
    ax_vel_err.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Panel 4: Heading Error Comparison (middle-left)
    # -------------------------------------------------------------------------
    ax_heading_err = fig.add_subplot(gs[1, 1])
    ax_heading_err.set_xlabel("Time (s)")
    ax_heading_err.set_ylabel("Heading Error (deg)")
    ax_heading_err.set_title("🧭 Heading Error Comparison", fontweight="bold", loc="left")

    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    ekf_heading_err = np.abs(np.degrees(angle_diff(X_ekf[:, 4], X_truth_cam[:, 4])))
    ukf_heading_err = np.abs(np.degrees(angle_diff(X_ukf[:, 4], X_truth_cam[:, 4])))

    ax_heading_err.plot(
        t_cam,
        ekf_heading_err,
        linewidth=2,
        color=COLORS["blue"],
        alpha=0.7,
        label="EKF",
    )
    ax_heading_err.plot(
        t_cam,
        ukf_heading_err,
        linewidth=2,
        color=COLORS["red"],
        alpha=0.7,
        linestyle="--",
        label="UKF",
    )
    ax_heading_err.axhline(
        7.0,
        linestyle=":",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label="PRD (7°)",
    )

    ax_heading_err.legend(loc="upper right", fontsize=9)
    ax_heading_err.grid(True, alpha=0.2)
    ax_heading_err.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Panel 5: Gyro Bias Comparison (middle-right)
    # -------------------------------------------------------------------------
    ax_gyro = fig.add_subplot(gs[1, 2])
    ax_gyro.set_xlabel("Time (s)")
    ax_gyro.set_ylabel("Gyro Bias (deg/s)")
    ax_gyro.set_title("⚙️ Gyro Bias Estimates", fontweight="bold", loc="left")

    # Truth
    ax_gyro.plot(
        t_cam,
        np.degrees(X_truth_cam[:, 5]),
        linewidth=2,
        color=COLORS["gray"],
        alpha=0.5,
        label="Truth",
        linestyle="--",
    )

    # EKF & UKF estimates
    ax_gyro.plot(
        t_cam,
        np.degrees(X_ekf[:, 5]),
        linewidth=2,
        color=COLORS["blue"],
        alpha=0.7,
        label="EKF",
    )
    ax_gyro.plot(
        t_cam,
        np.degrees(X_ukf[:, 5]),
        linewidth=2,
        color=COLORS["red"],
        alpha=0.7,
        linestyle="--",
        label="UKF",
    )

    ax_gyro.legend(loc="best", fontsize=9)
    ax_gyro.grid(True, alpha=0.2)
    ax_gyro.axhline(0, color="k", linewidth=0.5, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 6: RMSE Summary Bar Chart (bottom-left)
    # -------------------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[2, 0])
    ax_bar.set_title("📊 RMSE Comparison", fontweight="bold", loc="left")

    metrics_names = ["Position\n(cm)", "Velocity\n(cm/s)", "Heading\n(deg)"]
    ekf_values = [
        ekf_metrics["pos_rmse_cm"],
        ekf_metrics["vel_rmse_cm_s"],
        ekf_metrics["heading_rmse_deg"],
    ]
    ukf_values = [
        ukf_metrics["pos_rmse_cm"],
        ukf_metrics["vel_rmse_cm_s"],
        ukf_metrics["heading_rmse_deg"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax_bar.bar(x - width / 2, ekf_values, width, label="EKF", color=COLORS["blue"], alpha=0.7)
    ax_bar.bar(x + width / 2, ukf_values, width, label="UKF", color=COLORS["red"], alpha=0.7)

    ax_bar.set_ylabel("RMSE")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics_names)
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.grid(True, alpha=0.2, axis="y")

    # -------------------------------------------------------------------------
    # Panel 7: NEES Comparison (bottom-middle)
    # -------------------------------------------------------------------------
    ax_nees = fig.add_subplot(gs[2, 1])
    ax_nees.set_xlabel("Time (s)")
    ax_nees.set_ylabel("NEES")
    ax_nees.set_title("📊 Filter Consistency (NEES)", fontweight="bold", loc="left")

    # Compute NEES for both filters
    P_ekf = np.array(ekf_result.filtered_covariances)
    P_ukf = np.array(ukf_result.filtered_covariances)

    ekf_nees = compute_nees(X_truth_cam, X_ekf, P_ekf)
    ukf_nees = compute_nees(X_truth_cam, X_ukf, P_ukf)

    ax_nees.plot(t_cam, ekf_nees, linewidth=1.5, color=COLORS["blue"], alpha=0.6, label="EKF")
    ax_nees.plot(
        t_cam,
        ukf_nees,
        linewidth=1.5,
        color=COLORS["red"],
        alpha=0.6,
        linestyle="--",
        label="UKF",
    )

    # Expected value and bounds
    from scipy.stats import chi2

    state_dim = 8
    ax_nees.axhline(
        state_dim,
        linestyle="-",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label=f"Expected ({state_dim})",
    )
    lower = chi2.ppf(0.025, df=state_dim)
    upper = chi2.ppf(0.975, df=state_dim)
    ax_nees.axhline(lower, linestyle=":", color=COLORS["red"], linewidth=1, alpha=0.4)
    ax_nees.axhline(upper, linestyle=":", color=COLORS["red"], linewidth=1, alpha=0.4)
    ax_nees.fill_between(t_cam, lower, upper, alpha=0.1, color=COLORS["green"], label="95% CI")

    ax_nees.legend(loc="upper right", fontsize=8)
    ax_nees.grid(True, alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 8: Sigma Point Coverage (bottom-right) - UKF specific
    # -------------------------------------------------------------------------
    ax_sigma = fig.add_subplot(gs[2, 2])
    ax_sigma.set_title("🎯 Sigma Point Coverage (UKF)", fontweight="bold", loc="left")
    ax_sigma.set_xlabel("X Position (m)")
    ax_sigma.set_ylabel("Y Position (m)")
    ax_sigma.set_aspect("equal")

    # Show final position with uncertainty ellipse for both filters
    # EKF covariance ellipse
    from matplotlib.patches import Ellipse

    ekf_pos_final = X_ekf[-1, :2]
    ekf_cov_final = P_ekf[-1, :2, :2]
    eigvals, eigvecs = np.linalg.eigh(ekf_cov_final)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(5.991 * eigvals)  # 95% confidence
    ekf_ellipse = Ellipse(
        ekf_pos_final,
        width,
        height,
        angle=angle,
        facecolor="none",
        edgecolor=COLORS["blue"],
        linewidth=2,
        linestyle="-",
        alpha=0.7,
        label="EKF 95% CI",
    )
    ax_sigma.add_patch(ekf_ellipse)

    # UKF covariance ellipse
    ukf_pos_final = X_ukf[-1, :2]
    ukf_cov_final = P_ukf[-1, :2, :2]
    eigvals, eigvecs = np.linalg.eigh(ukf_cov_final)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(5.991 * eigvals)
    ukf_ellipse = Ellipse(
        ukf_pos_final,
        width,
        height,
        angle=angle,
        facecolor="none",
        edgecolor=COLORS["red"],
        linewidth=2,
        linestyle="--",
        alpha=0.7,
        label="UKF 95% CI",
    )
    ax_sigma.add_patch(ukf_ellipse)

    # True final position
    truth_pos_final = X_truth_cam[-1, :2]
    ax_sigma.plot(
        truth_pos_final[0],
        truth_pos_final[1],
        "o",
        color=COLORS["gray"],
        markersize=10,
        label="Truth",
        zorder=10,
    )

    # Estimate points
    ax_sigma.plot(
        ekf_pos_final[0],
        ekf_pos_final[1],
        "s",
        color=COLORS["blue"],
        markersize=8,
        label="EKF",
        zorder=9,
    )
    ax_sigma.plot(
        ukf_pos_final[0],
        ukf_pos_final[1],
        "^",
        color=COLORS["red"],
        markersize=8,
        label="UKF",
        zorder=9,
    )

    ax_sigma.legend(loc="best", fontsize=8)
    ax_sigma.grid(True, alpha=0.2)

    # -------------------------------------------------------------------------
    # Overall Title
    # -------------------------------------------------------------------------
    title = (
        f"{scenario_name} — EKF vs UKF Comparison\n"
        f"EKF: {ekf_metrics['pos_rmse_cm']:.2f} cm, {ekf_metrics['vel_rmse_cm_s']:.2f} cm/s, "
        f"{ekf_metrics['heading_rmse_deg']:.2f}°  |  "
        f"UKF: {ukf_metrics['pos_rmse_cm']:.2f} cm, {ukf_metrics['vel_rmse_cm_s']:.2f} cm/s, "
        f"{ukf_metrics['heading_rmse_deg']:.2f}°"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

    # Save
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 Saved: {output_path}")


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Run EKF and UKF on three scenarios and compare performance."""
    print_section_header("UKF vs EKF Comparative Study")

    print(
        """
    Welcome! This example compares Unscented Kalman Filter (UKF) and Extended
    Kalman Filter (EKF) on the same three scenarios from example 03.

    You will learn:
    • How UKF's sigma-point transform differs from EKF's Jacobian linearization
    • When UKF's improved nonlinearity handling provides real benefits
    • The computational tradeoff: UKF ≈ 3-5× slower but potentially more accurate
    • How to choose between EKF and UKF for your application

    Let's compare!
    """
    )

    # -------------------------------------------------------------------------
    # Shared Configuration (IDENTICAL to example 03 for fair comparison)
    # -------------------------------------------------------------------------
    print_section_header("Step 1: Configuration (Identical for Fair Comparison)")

    # Simulation config using REALISTIC SpikeGadgets IMU specifications
    sim_config = SimpleSimConfig(
        duration_s=10.0,
        fs_imu=200.0,
        fs_cam=30.0,
        # SpikeGadgets IMU specs (from PRD.md):
        gyro_noise_density=0.000175,  # 0.01 °/s/√Hz (SpikeGadgets spec)
        accel_noise_density=0.00196,  # 0.2 mg/√Hz (SpikeGadgets spec)
        gyro_bias_std=0.001,  # ~0.06 °/s bias std
        accel_bias_std=0.01,  # ~1 mg bias std
        cam_noise_std=0.005,  # 5 mm camera noise
        cam_dropout_prob=0.0,  # NO DROPOUTS (ideal conditions)
    )

    # EKF config using REALISTIC SpikeGadgets IMU specifications (same as example 03)
    ekf_config = EKFConfig(
        # Process noise spectral densities (tuned for good performance)
        process_noise_pos=2e-3,  # m^2/s^3
        process_noise_vel=1e-1,  # (m/s)^2/s
        process_noise_heading=1e-3,  # rad^2/s
        process_noise_gyro_bias=5e-7,  # (rad/s)^2/s
        process_noise_accel_bias=5e-5,  # (m/s²)^2/s
        # Measurement noise
        measurement_noise_pos=0.005**2,  # m^2
        measurement_noise_heading=0.05**2,  # rad^2
        # SpikeGadgets IMU specs
        imu_gyro_noise_density=0.000175,  # rad/s/√Hz
        imu_accel_noise_density=0.00196,  # m/s²/√Hz
        # Physics
        damping_coeff=0.5,
        led_distance=0.04,
        use_heading_measurement=True,
        enable_zupt=False,
    )

    # UKF config (matched parameters to EKF, default sigma-point settings)
    ukf_config = UKFConfig(
        # Process noise spectral densities (same as EKF)
        process_noise_pos=2e-3,  # m^2/s^3
        process_noise_vel=1e-1,  # (m/s)^2/s
        process_noise_heading=1e-3,  # rad^2/s
        process_noise_gyro_bias=5e-7,  # (rad/s)^2/s
        process_noise_accel_bias=5e-5,  # (m/s²)^2/s
        # Measurement noise (same as EKF)
        measurement_noise_pos=0.005**2,  # m^2
        measurement_noise_heading=0.05**2,  # rad^2
        # SpikeGadgets IMU specs (same as EKF)
        imu_gyro_noise_density=0.000175,  # rad/s/√Hz
        imu_accel_noise_density=0.00196,  # m/s²/√Hz
        # Physics (same as EKF)
        damping_coeff=0.5,
        led_distance=0.04,
        use_heading_measurement=True,
        enable_zupt=False,
        # UKF-specific parameters (defaults)
        alpha=1.732,  # sqrt(3), aggressive sigma-point spread
        beta=2.0,  # Gaussian optimal
        kappa=1.0,  # Secondary scaling
    )

    print(
        f"""
    Configuration Summary:
      • Same simulation, process noise, measurement noise for both filters
      • EKF: Uses Jacobian linearization (fast)
      • UKF: Propagates {2*8+1} sigma points (slower but captures nonlinearity)
      • Alpha = {ukf_config.alpha:.3f}: Sigma-point spread parameter
      • Beta = {ukf_config.beta}: Optimal for Gaussian distributions
    """
    )

    # Storage for results
    all_results = []

    # -------------------------------------------------------------------------
    # Scenario 1: Stationary
    # -------------------------------------------------------------------------
    print_section_header("Step 2: Scenario 1 - Stationary")
    explain_scenario(
        "Stationary",
        "Minimal (no motion dynamics)",
        "EKF should be sufficient - little nonlinearity to exploit",
    )

    print("   Running simulation...")
    sim_stat = simulate_stationary(
        sim_config, position=np.array([1.0, 1.0]), heading=np.pi / 6, seed=42
    )

    print("   Running EKF...")
    t0_ekf = time.time()
    ekf_stat = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_stat["t_imu"],
        U_imu=sim_stat["U_imu"],
        t_cam=sim_stat["t_cam_exp"],
        Z_cam_led1=sim_stat["Z_cam_led1"],
        Z_cam_led2=sim_stat["Z_cam_led2"],
        mask_cam=sim_stat["mask_cam"],
    )
    ekf_time_stat = time.time() - t0_ekf

    print("   Running UKF...")
    t0_ukf = time.time()
    ukf_stat = unscented_kalman_filter(
        ukf_config=ukf_config,
        t_imu=sim_stat["t_imu"],
        U_imu=sim_stat["U_imu"],
        t_cam=sim_stat["t_cam_exp"],
        Z_cam_led1=sim_stat["Z_cam_led1"],
        Z_cam_led2=sim_stat["Z_cam_led2"],
        mask_cam=sim_stat["mask_cam"],
    )
    ukf_time_stat = time.time() - t0_ukf

    # Compute metrics for both filters
    t_cam = sim_stat["t_cam_exp"]
    t_imu = sim_stat["t_imu"]
    X_truth = sim_stat["X_truth"]
    X_truth_cam = np.column_stack([np.interp(t_cam, t_imu, X_truth[:, i]) for i in range(5)])
    X_truth_cam[:, 4] = interp_angle(t_cam, t_imu, X_truth[:, 4])

    # Add bias truth
    bias_truth_cam = np.column_stack(
        [
            np.interp(t_cam, t_imu, sim_stat["bias_gyro"]),
            np.interp(t_cam, t_imu, sim_stat["bias_accel_x"]),
            np.interp(t_cam, t_imu, sim_stat["bias_accel_y"]),
        ]
    )
    X_truth_full = np.column_stack([X_truth_cam, bias_truth_cam])

    # Get layout for NEES calculation
    layout = get_layout(ekf_config.state_mode)

    # Helper function to compute metrics
    def compute_metrics(result, truth_full):
        X_est = np.array(result.filtered_means)
        P_est = np.array(result.filtered_covariances)

        pos_rmse = compute_position_rmse(truth_full[:, :2] * 100, X_est[:, :2] * 100)
        vel_err = np.linalg.norm(X_est[:, 2:4] - truth_full[:, 2:4], axis=1) * 100
        vel_rmse = np.sqrt(np.mean(vel_err**2))

        def angle_diff(a, b):
            return np.arctan2(np.sin(a - b), np.cos(a - b))

        heading_err = np.abs(np.degrees(angle_diff(X_est[:, 4], truth_full[:, 4])))
        heading_rmse = np.sqrt(np.mean(heading_err**2))

        # Compute NEES using layout (automatically handles angle wrapping)
        nees = compute_nees(truth_full, X_est, P_est, layout=layout)
        mean_nees = np.mean(nees)

        return {
            "pos_rmse_cm": pos_rmse,
            "vel_rmse_cm_s": vel_rmse,
            "heading_rmse_deg": heading_rmse,
            "mean_nees": mean_nees,
        }

    ekf_metrics_stat = compute_metrics(ekf_stat, X_truth_full)
    ukf_metrics_stat = compute_metrics(ukf_stat, X_truth_full)

    print_comparison_table(
        "Stationary", ekf_metrics_stat, ukf_metrics_stat, ekf_time_stat, ukf_time_stat
    )

    print(
        """
    📖 Interpretation:
       • Stationary has minimal nonlinearity → EKF vs UKF differences are small
       • UKF's extra computation doesn't provide significant accuracy benefit
       • Both filters meet PRD targets easily
       • Recommendation: Use EKF for stationary scenarios (faster, simpler)
    """
    )

    plot_comparison(
        sim_stat,
        ekf_stat,
        ukf_stat,
        "Stationary",
        ekf_metrics_stat,
        ukf_metrics_stat,
        OUTPUT_DIR / "04_comparison_stationary.png",
    )

    all_results.append(
        ("Stationary", ekf_metrics_stat, ukf_metrics_stat, ekf_time_stat, ukf_time_stat)
    )

    # -------------------------------------------------------------------------
    # Scenario 2: Constant Velocity
    # -------------------------------------------------------------------------
    print_section_header("Step 3: Scenario 2 - Constant Velocity")
    explain_scenario(
        "Constant Velocity",
        "Low (primarily linear motion)",
        "EKF should be sufficient - dynamics are nearly linear",
    )

    print("   Running simulation...")
    sim_vel = simulate_constant_velocity(
        sim_config,
        initial_position=np.array([0.5, 0.5]),
        velocity=np.array([0.3, 0.0]),
        seed=42,
    )

    print("   Running EKF...")
    t0_ekf = time.time()
    ekf_vel = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_vel["t_imu"],
        U_imu=sim_vel["U_imu"],
        t_cam=sim_vel["t_cam_exp"],
        Z_cam_led1=sim_vel["Z_cam_led1"],
        Z_cam_led2=sim_vel["Z_cam_led2"],
        mask_cam=sim_vel["mask_cam"],
    )
    ekf_time_vel = time.time() - t0_ekf

    print("   Running UKF...")
    t0_ukf = time.time()
    ukf_vel = unscented_kalman_filter(
        ukf_config=ukf_config,
        t_imu=sim_vel["t_imu"],
        U_imu=sim_vel["U_imu"],
        t_cam=sim_vel["t_cam_exp"],
        Z_cam_led1=sim_vel["Z_cam_led1"],
        Z_cam_led2=sim_vel["Z_cam_led2"],
        mask_cam=sim_vel["mask_cam"],
    )
    ukf_time_vel = time.time() - t0_ukf

    # Compute truth
    t_cam = sim_vel["t_cam_exp"]
    t_imu = sim_vel["t_imu"]
    X_truth = sim_vel["X_truth"]
    X_truth_cam = np.column_stack([np.interp(t_cam, t_imu, X_truth[:, i]) for i in range(5)])
    X_truth_cam[:, 4] = interp_angle(t_cam, t_imu, X_truth[:, 4])
    bias_truth_cam = np.column_stack(
        [
            np.interp(t_cam, t_imu, sim_vel["bias_gyro"]),
            np.interp(t_cam, t_imu, sim_vel["bias_accel_x"]),
            np.interp(t_cam, t_imu, sim_vel["bias_accel_y"]),
        ]
    )
    X_truth_full = np.column_stack([X_truth_cam, bias_truth_cam])

    ekf_metrics_vel = compute_metrics(ekf_vel, X_truth_full)
    ukf_metrics_vel = compute_metrics(ukf_vel, X_truth_full)

    print_comparison_table(
        "Constant Velocity",
        ekf_metrics_vel,
        ukf_metrics_vel,
        ekf_time_vel,
        ukf_time_vel,
    )

    print(
        """
    📖 Interpretation:
       • Constant velocity is nearly linear → small difference between EKF/UKF
       • Heading evolves smoothly, no abrupt turns
       • Both filters achieve excellent accuracy
       • Recommendation: Use EKF for straight-line motion (faster, adequate accuracy)
    """
    )

    plot_comparison(
        sim_vel,
        ekf_vel,
        ukf_vel,
        "Constant Velocity",
        ekf_metrics_vel,
        ukf_metrics_vel,
        OUTPUT_DIR / "04_comparison_constant_velocity.png",
    )

    all_results.append(
        (
            "Constant Velocity",
            ekf_metrics_vel,
            ukf_metrics_vel,
            ekf_time_vel,
            ukf_time_vel,
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 3: Circular Motion
    # -------------------------------------------------------------------------
    print_section_header("Step 4: Scenario 3 - Circular Motion")
    explain_scenario(
        "Circular Motion",
        "High (continuous rotation, nonlinear heading evolution)",
        "UKF may show advantage - persistent nonlinearity in rotation",
    )

    print("   Running simulation...")
    sim_circ = simulate_circular(
        sim_config,
        center=np.array([1.0, 1.0]),
        radius=0.5,
        angular_velocity=0.5,
        seed=42,
    )

    print("   Running EKF...")
    t0_ekf = time.time()
    ekf_circ = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_circ["t_imu"],
        U_imu=sim_circ["U_imu"],
        t_cam=sim_circ["t_cam_exp"],
        Z_cam_led1=sim_circ["Z_cam_led1"],
        Z_cam_led2=sim_circ["Z_cam_led2"],
        mask_cam=sim_circ["mask_cam"],
    )
    ekf_time_circ = time.time() - t0_ekf

    print("   Running UKF...")
    t0_ukf = time.time()
    ukf_circ = unscented_kalman_filter(
        ukf_config=ukf_config,
        t_imu=sim_circ["t_imu"],
        U_imu=sim_circ["U_imu"],
        t_cam=sim_circ["t_cam_exp"],
        Z_cam_led1=sim_circ["Z_cam_led1"],
        Z_cam_led2=sim_circ["Z_cam_led2"],
        mask_cam=sim_circ["mask_cam"],
    )
    ukf_time_circ = time.time() - t0_ukf

    # Compute truth
    t_cam = sim_circ["t_cam_exp"]
    t_imu = sim_circ["t_imu"]
    X_truth = sim_circ["X_truth"]
    X_truth_cam = np.column_stack([np.interp(t_cam, t_imu, X_truth[:, i]) for i in range(5)])
    X_truth_cam[:, 4] = interp_angle(t_cam, t_imu, X_truth[:, 4])
    bias_truth_cam = np.column_stack(
        [
            np.interp(t_cam, t_imu, sim_circ["bias_gyro"]),
            np.interp(t_cam, t_imu, sim_circ["bias_accel_x"]),
            np.interp(t_cam, t_imu, sim_circ["bias_accel_y"]),
        ]
    )
    X_truth_full = np.column_stack([X_truth_cam, bias_truth_cam])

    ekf_metrics_circ = compute_metrics(ekf_circ, X_truth_full)
    ukf_metrics_circ = compute_metrics(ukf_circ, X_truth_full)

    print_comparison_table(
        "Circular Motion",
        ekf_metrics_circ,
        ukf_metrics_circ,
        ekf_time_circ,
        ukf_time_circ,
    )

    print(
        """
    📖 Interpretation:
       • Circular motion has strong nonlinearity (trigonometric heading dynamics)
       • UKF's sigma-point transform may capture rotation better than EKF's Jacobian
       • This is where UKF's advantages are most likely to appear
       • Trade computational cost (3-5× slower) against potential accuracy gain
       • Recommendation: Consider UKF for highly nonlinear scenarios if accuracy critical
    """
    )

    plot_comparison(
        sim_circ,
        ekf_circ,
        ukf_circ,
        "Circular Motion",
        ekf_metrics_circ,
        ukf_metrics_circ,
        OUTPUT_DIR / "04_comparison_circular.png",
    )

    all_results.append(
        (
            "Circular Motion",
            ekf_metrics_circ,
            ukf_metrics_circ,
            ekf_time_circ,
            ukf_time_circ,
        )
    )

    # -------------------------------------------------------------------------
    # Summary Across All Scenarios
    # -------------------------------------------------------------------------
    print_section_header("Summary: EKF vs UKF Trade-offs")

    print(
        """
    🎓 KEY TAKEAWAYS:

    1. COMPUTATIONAL COST:
       • UKF propagates (2n+1) = 17 sigma points vs EKF's 1 linearization point
       • UKF is consistently 3-5× slower than EKF
       • For real-time applications (<33 ms latency), this matters!

    2. ACCURACY COMPARISON:
    """
    )

    # Print summary table
    print("   " + "-" * 76)
    print(f"   {'Scenario':<20} {'Metric':<15} {'EKF':>12} {'UKF':>12} {'Winner':>10}")
    print("   " + "-" * 76)

    for scenario_name, ekf_m, ukf_m, _, _ in all_results:
        for metric_name, ekf_key, ukf_key, unit in [
            ("Position", "pos_rmse_cm", "pos_rmse_cm", "cm"),
            ("Velocity", "vel_rmse_cm_s", "vel_rmse_cm_s", "cm/s"),
            ("Heading", "heading_rmse_deg", "heading_rmse_deg", "°"),
        ]:
            ekf_val = ekf_m[ekf_key]
            ukf_val = ukf_m[ukf_key]
            winner = "UKF ✓" if ukf_val < ekf_val else "EKF ✓" if ekf_val < ukf_val else "TIE"
            print(
                f"   {scenario_name:<20} {metric_name:<15} "
                f"{ekf_val:>10.2f} {unit:<2} {ukf_val:>10.2f} {unit:<2} {winner:>10}"
            )
        print("   " + "·" * 76)

    print("   " + "-" * 76)

    # Count wins
    ukf_wins = sum(
        [
            1
            for _, ekf_m, ukf_m, _, _ in all_results
            for ukf_val, ekf_val in [
                (ukf_m["pos_rmse_cm"], ekf_m["pos_rmse_cm"]),
                (ukf_m["vel_rmse_cm_s"], ekf_m["vel_rmse_cm_s"]),
                (ukf_m["heading_rmse_deg"], ekf_m["heading_rmse_deg"]),
            ]
            if ukf_val < ekf_val
        ]
    )
    total_comparisons = len(all_results) * 3

    print(
        f"""
    3. OVERALL VERDICT:
       • UKF won {ukf_wins}/{total_comparisons} metric comparisons
       • EKF won {total_comparisons - ukf_wins}/{total_comparisons} metric comparisons
       • {'UKF provides marginal improvements' if ukf_wins > total_comparisons/2 else 'EKF is competitive with UKF'}

    4. WHEN TO USE UKF:
       ✓ Highly nonlinear dynamics (tight circular motion, aggressive maneuvers)
       ✓ Offline processing where computation time is not critical
       ✓ When accuracy is paramount and you've hit EKF's limits
       ✓ When you want to avoid deriving/debugging Jacobians

    5. WHEN TO USE EKF:
       ✓ Real-time applications requiring low latency (<33 ms)
       ✓ Stationary or near-linear motion patterns
       ✓ Resource-constrained systems (embedded, mobile)
       ✓ When 3-5× speedup matters more than marginal accuracy gain

    6. BEST PRACTICE:
       • Start with EKF (simpler, faster, usually sufficient)
       • Profile on your specific data
       • Switch to UKF only if EKF fails to meet accuracy requirements
       • For trodestrack: EKF is recommended for online mode, UKF for offline smoothing

    NEXT STEPS:
    • Run examples/05_ekf_with_dropouts.py to test robustness under occlusions
    • Run examples/06_ukf_with_dropouts.py to compare UKF vs EKF under stress
    • See examples/README.md for full learning path

    Happy learning! 🚀
    """
    )


if __name__ == "__main__":
    main()
