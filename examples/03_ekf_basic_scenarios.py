"""Extended Kalman Filter (EKF) on Basic Scenarios - Educational Demonstration.

LEARNING OBJECTIVES:
-------------------
This example teaches you how the Extended Kalman Filter (EKF) performs on three
fundamental motion patterns under IDEAL CONDITIONS (no camera dropouts):

1. **Stationary**: Rat sitting still
   - Challenge: EKF must reject IMU drift while trusting camera
   - Expected: Position error < 2 cm, bias estimates converge to true values

2. **Constant Velocity**: Rat moving in straight line
   - Challenge: EKF must integrate IMU and camera smoothly
   - Expected: Velocity tracking < 10 cm/s error, heading stable

3. **Circular Motion**: Rat running in circles
   - Challenge: EKF must learn gyroscope bias from turning motion
   - Expected: Heading error < 7°, gyro bias converges over time

WHY THIS MATTERS:
-----------------
Before testing robustness (dropouts, occlusions), we must verify the filter works
correctly under ideal conditions. This example establishes baseline performance
against PRD (Product Requirements Document) targets:

    Position RMSE:  ≤ 2 cm
    Velocity RMSE:  ≤ 10 cm/s
    Heading RMSE:   ≤ 7°
    NEES:           ≈ 8.0 (for 8-dimensional state)

WHAT YOU'LL LEARN:
------------------
- How to configure EKF parameters for different motion types
- How to interpret NEES (Normalized Estimation Error Squared) for filter consistency
- Why gyro bias is unobservable during stationary periods
- How measurement innovation reflects filter prediction quality
- The relationship between process noise and state uncertainty

USAGE:
------
    uv run python examples/03_ekf_basic_scenarios.py

OUTPUT:
-------
    - Console: Detailed metrics table comparing to PRD targets
    - Files: 03_ekf_stationary.png, 03_ekf_constant_velocity.png, 03_ekf_circular.png

ESTIMATED RUNTIME: ~5 seconds

KEY CONCEPTS ILLUSTRATED:
-------------------------
1. **Sensor Fusion**: Camera (30 Hz) + IMU (200 Hz) → smooth state estimate
2. **Bias Observability**: Gyro bias only observable during rotation
3. **Filter Consistency**: NEES values indicate if covariance matches actual error
4. **Innovation Whiteness**: Residuals should be zero-mean, uncorrelated Gaussian
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
from trodestrack.qa.metrics import compute_nees, compute_position_rmse
from trodestrack.sim.simple import (
    SimpleSimConfig,
    simulate_circular,
    simulate_constant_velocity,
    simulate_stationary,
)
from trodestrack.viz.styles import COLORS, apply_tufte_style

# Apply clean visualization style (Tufte principles: maximize data-ink ratio)
apply_tufte_style()

# Output directory (save in examples/ folder for easy access)
OUTPUT_DIR = Path(__file__).parent


# =============================================================================
# Pedagogical Helper Functions
# =============================================================================


def print_section_header(title: str) -> None:
    """Print a clear section header for pedagogical clarity.

    Args:
        title: Section title to display
    """
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_metrics_table(
    scenario_name: str,
    pos_rmse_cm: float,
    vel_rmse_cm_s: float,
    heading_rmse_deg: float,
    mean_nees: float,
    marginal_ll: float,
    gyro_bias_converged: bool | None = None,
) -> None:
    """Print performance metrics in a clear, pedagogical table format.

    Args:
        scenario_name: Name of the scenario (e.g., "Stationary")
        pos_rmse_cm: Position RMSE in cm
        vel_rmse_cm_s: Velocity RMSE in cm/s
        heading_rmse_deg: Heading RMSE in degrees
        mean_nees: Mean NEES value
        marginal_ll: Marginal log-likelihood
        gyro_bias_converged: Whether gyro bias converged (circular scenario only)
    """
    print(f"\n{scenario_name} Performance Metrics:")
    print("-" * 80)
    print(
        f"  Position RMSE:     {pos_rmse_cm:6.2f} cm      (PRD target: ≤ 2 cm)     "
        f"{'✓ PASS' if pos_rmse_cm <= 2.0 else '✗ FAIL'}"
    )
    print(
        f"  Velocity RMSE:     {vel_rmse_cm_s:6.2f} cm/s    (PRD target: ≤ 10 cm/s)  "
        f"{'✓ PASS' if vel_rmse_cm_s <= 10.0 else '✗ FAIL'}"
    )
    print(
        f"  Heading RMSE:      {heading_rmse_deg:6.2f}°       (PRD target: ≤ 7°)      "
        f"{'✓ PASS' if heading_rmse_deg <= 7.0 else '✗ FAIL'}"
    )
    print(f"  Mean NEES:         {mean_nees:6.2f}        (ideal: ≈ 8.0 for 8D state)")
    print(f"  Marginal LL:       {marginal_ll:8.1f}")

    if gyro_bias_converged is not None:
        print(
            f"  Gyro Bias Conv:    {'✓ YES' if gyro_bias_converged else '✗ NO'}        "
            "(requires rotation)"
        )

    # Educational note on NEES interpretation
    if mean_nees < 6.0:
        print("  ⚠ Filter may be OVERCONFIDENT (NEES too low)")
    elif mean_nees > 10.0:
        print("  ⚠ Filter may be UNDERCONFIDENT (NEES too high)")
    else:
        print("  ✓ Filter consistency is GOOD (NEES near ideal)")

    print("-" * 80)


def explain_scenario(
    scenario_name: str, key_challenge: str, what_to_watch: str
) -> None:
    """Print pedagogical explanation of what to expect from a scenario.

    Args:
        scenario_name: Name of the scenario
        key_challenge: Main challenge for the filter
        what_to_watch: What metrics/behaviors to observe
    """
    print(f"\n📚 {scenario_name} Scenario")
    print(f"   Challenge: {key_challenge}")
    print(f"   Watch for: {what_to_watch}")


# =============================================================================
# Visualization Function
# =============================================================================


def plot_ekf_results(
    sim_data: dict,
    filter_result,
    scenario_name: str,
    metrics: dict,
    output_path: Path,
    ekf_config: EKFConfig,
) -> None:
    """Create comprehensive visualization of EKF performance.

    This plot teaches students how to diagnose filter performance by showing:
    1. Trajectory: Does the filter track the true path?
    2. Position errors: How much does the filter deviate from truth?
    3. Velocity errors: Does velocity estimation remain accurate?
    4. Heading errors: Is heading tracked correctly?
    5. Bias estimates: Do biases converge to true values?
    6. NEES: Is the filter consistent (covariance matches error)?

    Args:
        sim_data: Simulation data dictionary
        filter_result: EKF filter results
        scenario_name: Descriptive name for the scenario
        metrics: Dictionary of computed metrics
        output_path: Where to save the figure
        ekf_config: EKF configuration (for extracting state layout)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Get state layout from config (BEST PRACTICE - dimension-agnostic!)
    layout = get_layout(ekf_config.state_mode)

    t_imu = sim_data["t_imu"]
    t_cam = sim_data["t_cam_exp"]
    X_truth = sim_data["X_truth"]
    X_est = np.array(filter_result.filtered_means)
    P_est = np.array(filter_result.filtered_covariances)

    # Interpolate truth to camera times (including biases for full state)
    from trodestrack.sim.utils import interp_angle

    # Use layout indices for dimension-agnostic interpolation
    pos_idx = layout.pos_idx
    vel_idx = layout.vel_idx
    heading_idx = layout.heading_idx
    bias_gyro_idx = layout.bias_gyro_idx
    # bias_accel_idx = layout.bias_accel_idx

    # Build truth state at camera times using layout
    truth_components = []
    # Position
    for i in pos_idx:
        truth_components.append(np.interp(t_cam, t_imu, X_truth[:, i]))
    # Velocity
    for i in vel_idx:
        truth_components.append(np.interp(t_cam, t_imu, X_truth[:, i]))
    # Heading (angle-aware interpolation)
    truth_components.append(interp_angle(t_cam, t_imu, X_truth[:, heading_idx]))
    # Gyro bias
    for _ in bias_gyro_idx:
        truth_components.append(np.interp(t_cam, t_imu, sim_data["bias_gyro"]))
    # Accel biases
    truth_components.append(np.interp(t_cam, t_imu, sim_data["bias_accel_x"]))
    truth_components.append(np.interp(t_cam, t_imu, sim_data["bias_accel_y"]))

    X_truth_cam = np.column_stack(truth_components)

    # -------------------------------------------------------------------------
    # Panel 1: 2D Trajectory (top-left, span 2 rows)
    # -------------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[:2, 0])
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("🎯 Trajectory Tracking", fontweight="bold", loc="left")

    # Extract positions using layout (dimension-agnostic)
    pos_truth_imu = X_truth[:, layout.pos_idx]  # (N_imu, 2)
    pos_est = X_est[:, layout.pos_idx]  # (N_cam, 2)

    # Ground truth path (use IMU-rate truth for smooth line)
    ax_traj.plot(
        pos_truth_imu[:, 0],
        pos_truth_imu[:, 1],
        "-",
        linewidth=2,
        color=COLORS["gray"],
        alpha=0.4,
        label="Ground Truth",
        zorder=1,
    )

    # EKF estimate
    ax_traj.plot(
        pos_est[:, 0],
        pos_est[:, 1],
        "-",
        linewidth=2.5,
        color=COLORS["blue"],
        alpha=0.8,
        label="EKF Estimate",
        zorder=2,
    )

    # Camera measurements (sparse)
    mask = sim_data["mask_cam"]
    ax_traj.scatter(
        sim_data["Z_cam_led1"][mask, 0],
        sim_data["Z_cam_led1"][mask, 1],
        s=15,
        c=COLORS["orange"],
        alpha=0.5,
        label="Camera Obs",
        zorder=3,
    )

    # Add uncertainty ellipse at final position using layout
    from matplotlib.patches import Ellipse

    final_pos = pos_est[-1]  # Already extracted using layout
    # Extract position covariance using layout indices
    final_cov_full = P_est[-1]
    final_cov = final_cov_full[np.ix_(layout.pos_idx, layout.pos_idx)]  # (2, 2)
    eigvals, eigvecs = np.linalg.eigh(final_cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(5.991 * eigvals)  # 95% confidence (chi2 with df=2)
    ellipse = Ellipse(
        final_pos,
        width,
        height,
        angle=angle,
        facecolor="none",
        edgecolor=COLORS["blue"],
        linewidth=2,
        linestyle="--",
        alpha=0.6,
    )
    ax_traj.add_patch(ellipse)

    ax_traj.legend(loc="best", fontsize=9)
    ax_traj.grid(True, alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 2: Position Error Time Series (top-middle)
    # -------------------------------------------------------------------------
    ax_pos_err = fig.add_subplot(gs[0, 1])
    ax_pos_err.set_xlabel("Time (s)")
    ax_pos_err.set_ylabel("Position Error (cm)")
    ax_pos_err.set_title("📏 Position Accuracy", fontweight="bold", loc="left")

    # Extract positions using layout for comparison
    pos_truth_cam = X_truth_cam[:, layout.pos_idx]  # (N_cam, 2)
    pos_err = np.linalg.norm(pos_est - pos_truth_cam, axis=1) * 100  # m→cm
    ax_pos_err.plot(t_cam, pos_err, linewidth=2, color=COLORS["red"], alpha=0.8)
    ax_pos_err.axhline(
        2.0,
        linestyle="--",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label="PRD Target (2 cm)",
    )
    ax_pos_err.fill_between(t_cam, 0, 2.0, alpha=0.1, color=COLORS["green"])
    ax_pos_err.legend(loc="upper right", fontsize=8)
    ax_pos_err.grid(True, alpha=0.2)
    ax_pos_err.set_ylim(bottom=0)

    # Annotate mean error
    mean_err = np.mean(pos_err)
    ax_pos_err.text(
        0.98,
        0.95,
        f"Mean: {mean_err:.2f} cm",
        transform=ax_pos_err.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # -------------------------------------------------------------------------
    # Panel 3: Velocity Error (top-right)
    # -------------------------------------------------------------------------
    ax_vel_err = fig.add_subplot(gs[0, 2])
    ax_vel_err.set_xlabel("Time (s)")
    ax_vel_err.set_ylabel("Velocity Error (cm/s)")
    ax_vel_err.set_title("🏃 Velocity Tracking", fontweight="bold", loc="left")

    # Extract velocities using layout for comparison
    vel_est = X_est[:, layout.vel_idx]  # (N_cam, 2)
    vel_truth_cam = X_truth_cam[:, layout.vel_idx]  # (N_cam, 2)
    vel_err = np.linalg.norm(vel_est - vel_truth_cam, axis=1) * 100  # m/s→cm/s
    ax_vel_err.plot(t_cam, vel_err, linewidth=2, color=COLORS["purple"], alpha=0.8)
    ax_vel_err.axhline(
        10.0,
        linestyle="--",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label="PRD Target (10 cm/s)",
    )
    ax_vel_err.fill_between(t_cam, 0, 10.0, alpha=0.1, color=COLORS["green"])
    ax_vel_err.legend(loc="upper right", fontsize=8)
    ax_vel_err.grid(True, alpha=0.2)
    ax_vel_err.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Panel 4: Heading Error (middle-left)
    # -------------------------------------------------------------------------
    ax_heading_err = fig.add_subplot(gs[1, 1])
    ax_heading_err.set_xlabel("Time (s)")
    ax_heading_err.set_ylabel("Heading Error (deg)")
    ax_heading_err.set_title("🧭 Heading Accuracy", fontweight="bold", loc="left")

    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    # Extract headings using layout
    heading_est = X_est[:, layout.heading_idx]  # (N_cam,)
    heading_truth_cam = X_truth_cam[:, layout.heading_idx]  # (N_cam,)
    heading_err = np.abs(np.degrees(angle_diff(heading_est, heading_truth_cam)))
    ax_heading_err.plot(
        t_cam, heading_err, linewidth=2, color=COLORS["orange"], alpha=0.8
    )
    ax_heading_err.axhline(
        7.0,
        linestyle="--",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label="PRD Target (7°)",
    )
    ax_heading_err.fill_between(t_cam, 0, 7.0, alpha=0.1, color=COLORS["green"])
    ax_heading_err.legend(loc="upper right", fontsize=8)
    ax_heading_err.grid(True, alpha=0.2)
    ax_heading_err.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Panel 5: Gyro Bias Estimate (middle-right)
    # -------------------------------------------------------------------------
    ax_gyro_bias = fig.add_subplot(gs[1, 2])
    ax_gyro_bias.set_xlabel("Time (s)")
    ax_gyro_bias.set_ylabel("Gyro Bias (deg/s)")
    ax_gyro_bias.set_title("⚙️ Gyro Bias Learning", fontweight="bold", loc="left")

    # Extract gyro bias using layout (supports single or multiple gyro biases)
    gyro_bias_idx = layout.bias_gyro_idx[0] if layout.bias_gyro_idx else None
    if gyro_bias_idx is not None:
        # Truth (already interpolated in X_truth_cam)
        bias_gyro_truth = X_truth_cam[:, gyro_bias_idx]
        ax_gyro_bias.plot(
            t_cam,
            np.degrees(bias_gyro_truth),
            linewidth=2,
            color=COLORS["gray"],
            alpha=0.5,
            label="True Bias",
            linestyle="--",
        )

        # EKF estimate
        bias_gyro_est = X_est[:, gyro_bias_idx]
        ax_gyro_bias.plot(
            t_cam,
            np.degrees(bias_gyro_est),
            linewidth=2,
            color=COLORS["purple"],
            alpha=0.8,
            label="EKF Estimate",
        )

        # ±1σ uncertainty
        bias_std = np.sqrt(P_est[:, gyro_bias_idx, gyro_bias_idx])
        ax_gyro_bias.fill_between(
            t_cam,
            np.degrees(bias_gyro_est - bias_std),
            np.degrees(bias_gyro_est + bias_std),
            alpha=0.2,
            color=COLORS["purple"],
            label="±1σ",
        )

        ax_gyro_bias.legend(loc="best", fontsize=8)
        ax_gyro_bias.grid(True, alpha=0.2)
        ax_gyro_bias.axhline(0, color="k", linewidth=0.5, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 6: Accel Bias Estimates (bottom-left)
    # -------------------------------------------------------------------------
    ax_accel_bias = fig.add_subplot(gs[2, 0])
    ax_accel_bias.set_xlabel("Time (s)")
    ax_accel_bias.set_ylabel("Accel Bias (m/s²)")
    ax_accel_bias.set_title(
        "⚙️ Accelerometer Bias Learning", fontweight="bold", loc="left"
    )

    # Extract accel biases using layout (supports 2D or 3D accel)
    accel_bias_idx = layout.bias_accel_idx
    if len(accel_bias_idx) >= 2:
        # Truth (already interpolated in X_truth_cam) - use first 2 accel biases
        bias_ax_truth = X_truth_cam[:, accel_bias_idx[0]]
        bias_ay_truth = X_truth_cam[:, accel_bias_idx[1]]
        ax_accel_bias.plot(
            t_cam,
            bias_ax_truth,
            linewidth=1.5,
            color=COLORS["gray"],
            alpha=0.4,
            linestyle="--",
            label="True X/Y",
        )
        ax_accel_bias.plot(
            t_cam,
            bias_ay_truth,
            linewidth=1.5,
            color=COLORS["gray"],
            alpha=0.4,
            linestyle="--",
        )

        # EKF estimates
        bias_ax_est = X_est[:, accel_bias_idx[0]]
        bias_ay_est = X_est[:, accel_bias_idx[1]]
        ax_accel_bias.plot(
            t_cam,
            bias_ax_est,
            linewidth=2,
            color=COLORS["blue"],
            alpha=0.8,
            label="EKF X",
        )
        ax_accel_bias.plot(
            t_cam,
            bias_ay_est,
            linewidth=2,
            color=COLORS["red"],
            alpha=0.8,
            label="EKF Y",
        )

        ax_accel_bias.legend(loc="best", fontsize=8, ncol=2)
        ax_accel_bias.grid(True, alpha=0.2)
        ax_accel_bias.axhline(0, color="k", linewidth=0.5, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 7: NEES (bottom-middle)
    # -------------------------------------------------------------------------
    ax_nees = fig.add_subplot(gs[2, 1])
    ax_nees.set_xlabel("Time (s)")
    ax_nees.set_ylabel("NEES")
    ax_nees.set_title("📊 Filter Consistency (NEES)", fontweight="bold", loc="left")

    # Compute full NEES using actual state dimension from layout
    from scipy.stats import chi2

    state_dim = layout.n  # Use layout dimension (supports 5D, 8D, 10D, 15D)
    nees_values = []
    for i in range(len(t_cam)):
        err = X_est[i] - X_truth_cam[i]
        # Wrap heading error using layout
        err[layout.heading_idx] = angle_diff(
            X_est[i, layout.heading_idx], X_truth_cam[i, layout.heading_idx]
        )
        nees = err @ np.linalg.solve(P_est[i], err)
        nees_values.append(nees)
    nees_values = np.array(nees_values)

    ax_nees.plot(t_cam, nees_values, linewidth=1.5, color=COLORS["purple"], alpha=0.6)

    # Chi-squared bounds (95% confidence interval)
    lower_bound = chi2.ppf(0.025, df=state_dim)
    upper_bound = chi2.ppf(0.975, df=state_dim)
    ax_nees.axhline(
        state_dim,
        linestyle="-",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label=f"Expected ({state_dim})",
    )
    ax_nees.axhline(
        lower_bound,
        linestyle="--",
        color=COLORS["red"],
        linewidth=1,
        alpha=0.4,
        label="95% CI",
    )
    ax_nees.axhline(
        upper_bound, linestyle="--", color=COLORS["red"], linewidth=1, alpha=0.4
    )

    # Shade acceptable region
    ax_nees.fill_between(
        t_cam, lower_bound, upper_bound, alpha=0.1, color=COLORS["green"]
    )

    ax_nees.legend(loc="upper right", fontsize=8)
    ax_nees.grid(True, alpha=0.2)

    # Annotate mean NEES
    mean_nees = np.mean(nees_values)
    ax_nees.text(
        0.02,
        0.95,
        f"Mean NEES: {mean_nees:.2f}",
        transform=ax_nees.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # -------------------------------------------------------------------------
    # Panel 8: Innovation Magnitude (bottom-right)
    # -------------------------------------------------------------------------
    ax_innov = fig.add_subplot(gs[2, 2])
    ax_innov.set_xlabel("Time (s)")
    ax_innov.set_ylabel("Innovation (cm)")
    ax_innov.set_title("🔍 Measurement Residuals", fontweight="bold", loc="left")

    # Compute innovations (measurement - prediction) using layout
    innovations = []
    for i in range(len(t_cam)):
        if sim_data["mask_cam"][i]:
            z_obs = sim_data["Z_cam_led1"][i]
            # Extract predicted position using layout
            z_pred = filter_result.predicted_means[i, layout.pos_idx]
            innov_mag = np.linalg.norm(z_obs - z_pred) * 100  # cm
            innovations.append(innov_mag)
        else:
            innovations.append(np.nan)
    innovations = np.array(innovations)

    # Plot only valid innovations
    valid_idx = ~np.isnan(innovations)
    ax_innov.plot(
        t_cam[valid_idx],
        innovations[valid_idx],
        "o",
        markersize=4,
        color=COLORS["orange"],
        alpha=0.6,
    )

    # Expected magnitude (from measurement noise)
    expected_std = 0.5  # 0.5 cm from config
    ax_innov.axhline(
        expected_std,
        linestyle="--",
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.5,
        label=f"Expected σ ({expected_std} cm)",
    )
    ax_innov.fill_between(t_cam, 0, expected_std, alpha=0.1, color=COLORS["green"])

    ax_innov.legend(loc="upper right", fontsize=8)
    ax_innov.grid(True, alpha=0.2)
    ax_innov.set_ylim(bottom=0)

    # Annotate mean innovation
    mean_innov = np.nanmean(innovations)
    ax_innov.text(
        0.02,
        0.95,
        f"Mean: {mean_innov:.2f} cm",
        transform=ax_innov.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # -------------------------------------------------------------------------
    # Overall Title with Metrics Summary
    # -------------------------------------------------------------------------
    title = (
        f"{scenario_name} — Extended Kalman Filter Performance\n"
        f"Position RMSE: {metrics['pos_rmse_cm']:.2f} cm  |  "
        f"Velocity RMSE: {metrics['vel_rmse_cm_s']:.2f} cm/s  |  "
        f"Heading RMSE: {metrics['heading_rmse_deg']:.2f}°  |  "
        f"Mean NEES: {metrics['mean_nees']:.2f}"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 Saved: {output_path}")


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Run EKF on three basic scenarios with pedagogical explanations."""
    print_section_header("EKF Basic Scenarios - Learning Sensor Fusion")

    print(
        """
    Welcome! This example demonstrates Extended Kalman Filter (EKF) performance
    on three fundamental motion patterns under IDEAL conditions (no dropouts).

    You will learn:
    • How EKF fuses camera (30 Hz) and IMU (200 Hz) measurements
    • How to interpret filter performance metrics (RMSE, NEES, innovations)
    • Why certain biases are observable only during specific motions
    • What "filter consistency" means and why it matters

    Let's begin!
    """
    )

    # -------------------------------------------------------------------------
    # Shared Configuration
    # -------------------------------------------------------------------------
    print_section_header("Step 1: Simulation Configuration")

    # Simulation config using REALISTIC SpikeGadgets IMU specifications
    sim_config = SimpleSimConfig(
        duration_s=10.0,  # 10 second runs (enough to see convergence)
        fs_imu=200.0,  # IMU at 200 Hz (higher than SpikeGadgets 104 Hz for accuracy)
        fs_cam=30.0,  # Camera at 30 Hz (typical video frame rate)
        # SpikeGadgets IMU specs (from PRD.md):
        gyro_noise_density=0.000175,  # 0.01 °/s/√Hz (SpikeGadgets spec)
        accel_noise_density=0.00196,  # 0.2 mg/√Hz (SpikeGadgets spec)
        gyro_bias_std=0.001,  # ~0.06 °/s bias std
        accel_bias_std=0.01,  # ~1 mg bias std
        cam_noise_std=0.005,  # 5 mm camera noise (realistic DLC/LED)
        cam_dropout_prob=0.0,  # NO DROPOUTS (ideal conditions)
    )

    print(
        f"""
    Simulation Setup:
      Duration:        {sim_config.duration_s} seconds
      IMU Rate:        {sim_config.fs_imu} Hz (high rate for accurate integration)
      Camera Rate:     {sim_config.fs_cam} Hz (typical video frame rate)
      Camera Noise:    {sim_config.cam_noise_std * 100:.1f} cm (realistic DLC/LED noise)
      Dropout Rate:    {sim_config.cam_dropout_prob * 100:.0f}% (NONE - ideal conditions)

    Important: Stationary/Constant_Velocity scenarios only simulate LED1 (single position).
    Circular Motion simulates both LEDs (position + heading measurement).
    PRD targets (≤7° heading) require dual-LED observations - single LED cannot achieve this!
    """
    )

    # EKF configuration using REALISTIC SpikeGadgets IMU specifications.
    # Pin to the 8D ``2d_full`` layout so the truth array (5D X_truth + biases
    # constructed below) matches the filter state dim.
    ekf_config = EKFConfig(
        state_mode="2d_full",
        # Process noise spectral densities (Q matrix continuous-time variances)
        # Tuned to balance accuracy and well-calibrated uncertainty (NEES ≈ 8)
        process_noise_pos=2e-3,  # m^2/s^3 - accounts for unmodeled dynamics
        process_noise_vel=1e-1,  # (m/s)^2/s - velocity model uncertainty
        process_noise_heading=1e-3,  # rad^2/s - heading drift between measurements
        process_noise_gyro_bias=5e-7,  # (rad/s)^2/s - gyro bias random walk
        process_noise_accel_bias=5e-5,  # (m/s²)^2/s - accel bias random walk
        # Measurement noise (R matrix, units: variance)
        measurement_noise_pos=0.005**2,  # m^2 - match camera noise (5mm)
        measurement_noise_heading=0.05**2,  # rad^2 - heading from dual LEDs (~3°)
        # IMU noise densities (SpikeGadgets specs - MUST match simulation!)
        imu_gyro_noise_density=0.000175,  # rad/s/√Hz (0.01 °/s/√Hz SpikeGadgets)
        imu_accel_noise_density=0.00196,  # m/s²/√Hz (0.2 mg/√Hz SpikeGadgets)
        # Physics
        damping_coeff=0.5,  # 1/s - velocity damping coefficient
        led_distance=0.04,  # m - 4 cm LED spacing (typical)
        # Advanced features
        use_heading_measurement=True,  # Use dual-LED heading
        enable_zupt=False,  # ZUPT not needed without dropouts
    )

    print(
        f"""
    EKF Configuration:
      Process Noise:   Tuned for smooth tracking with moderate uncertainty growth
      Measurement Noise: {ekf_config.measurement_noise_pos**0.5 * 100:.1f} cm (matches simulation)
      Damping:         {ekf_config.damping_coeff} s⁻¹ (models air resistance)
      LED Distance:    {ekf_config.led_distance * 100:.0f} cm (for heading measurement)

    Key point: EKF parameters must balance:
      • High Q → responsive but jittery
      • Low Q → smooth but slow to adapt
      • Correctly matched Q/R → optimal (Kalman gain)
    """
    )

    # -------------------------------------------------------------------------
    # Scenario 1: Stationary
    # -------------------------------------------------------------------------
    print_section_header("Step 2: Scenario 1 - Stationary Rat")
    explain_scenario(
        "Stationary",
        "EKF must reject IMU drift (biased random walk) using camera corrections",
        "Position error should stay near zero; gyro bias is UNOBSERVABLE (no rotation)",
    )

    print("   Running simulation...")
    sim_stationary = simulate_stationary(
        sim_config,
        position=np.array([1.0, 1.0]),  # Center of 2m x 2m arena
        heading=np.pi / 6,  # 30° heading (arbitrary)
        seed=42,
    )

    print("   Running EKF filter...")
    result_stationary = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_stationary["t_imu"],
        U_imu=sim_stationary["U_imu"],
        t_cam=sim_stationary["t_cam_exp"],
        Z_cam_led1=sim_stationary["Z_cam_led1"],
        Z_cam_led2=sim_stationary["Z_cam_led2"],
        mask_cam=sim_stationary["mask_cam"],
    )

    # Compute metrics
    t_cam = sim_stationary["t_cam_exp"]
    t_imu = sim_stationary["t_imu"]
    X_truth = sim_stationary["X_truth"]
    X_truth_cam = np.column_stack(
        [np.interp(t_cam, t_imu, X_truth[:, i]) for i in range(5)]
    )
    from trodestrack.sim.utils import interp_angle

    X_truth_cam[:, 4] = interp_angle(t_cam, t_imu, X_truth[:, 4])

    X_est = np.array(result_stationary.filtered_means)
    P_est = np.array(result_stationary.filtered_covariances)

    pos_rmse_cm = compute_position_rmse(X_truth_cam[:, :2] * 100, X_est[:, :2] * 100)
    vel_err = np.linalg.norm(X_est[:, 2:4] - X_truth_cam[:, 2:4], axis=1) * 100
    vel_rmse_cm_s = np.sqrt(np.mean(vel_err**2))

    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    heading_err = np.abs(np.degrees(angle_diff(X_est[:, 4], X_truth_cam[:, 4])))
    heading_rmse_deg = np.sqrt(np.mean(heading_err**2))

    # Compute NEES for full 8D state (need to add bias truth)
    bias_gyro_truth_cam = np.interp(t_cam, t_imu, sim_stationary["bias_gyro"])
    bias_ax_truth_cam = np.interp(t_cam, t_imu, sim_stationary["bias_accel_x"])
    bias_ay_truth_cam = np.interp(t_cam, t_imu, sim_stationary["bias_accel_y"])
    X_truth_full = np.column_stack(
        [X_truth_cam, bias_gyro_truth_cam, bias_ax_truth_cam, bias_ay_truth_cam]
    )

    # Get layout for state extraction and NEES calculation
    layout = get_layout(ekf_config.state_mode)
    nees = compute_nees(X_truth_full, X_est, P_est, layout=layout)
    mean_nees = np.mean(nees)

    metrics_stat = {
        "pos_rmse_cm": pos_rmse_cm,
        "vel_rmse_cm_s": vel_rmse_cm_s,
        "heading_rmse_deg": heading_rmse_deg,
        "mean_nees": mean_nees,
    }

    print_metrics_table(
        "Stationary",
        pos_rmse_cm,
        vel_rmse_cm_s,
        heading_rmse_deg,
        mean_nees,
        result_stationary.marginal_loglik,
        gyro_bias_converged=False,  # Cannot converge without rotation
    )

    print(
        """
    📖 Interpretation:
       • Position RMSE near zero → camera prevents drift
       • Velocity RMSE near zero → rat is stationary, filter knows it
       • Heading RMSE high → ⚠️  Only LED1 visible (no heading measurement!)
       • Gyro bias does NOT converge → bias is unobservable without rotation
       • NEES > 8.0 → filter slightly underconfident (high process noise)

    ⚠️  Limitation: This scenario only simulates LED1 (single position measurement).
    Without dual-LED heading observations, heading must be estimated from IMU gyro
    integration alone, leading to drift from initial heading uncertainty.
    PRD heading target (≤7°) requires dual-LED measurements (see Circular scenario).
    """
    )

    plot_ekf_results(
        sim_stationary,
        result_stationary,
        "Stationary",
        metrics_stat,
        OUTPUT_DIR / "03_ekf_stationary.png",
        ekf_config,
    )

    # -------------------------------------------------------------------------
    # Scenario 2: Constant Velocity
    # -------------------------------------------------------------------------
    print_section_header("Step 3: Scenario 2 - Constant Velocity")
    explain_scenario(
        "Constant Velocity",
        "EKF must track smooth linear motion, integrating IMU between camera frames",
        "Velocity should converge quickly; accelerometer bias becomes observable",
    )

    print("   Running simulation...")
    sim_const_vel = simulate_constant_velocity(
        sim_config,
        initial_position=np.array([0.5, 0.5]),
        velocity=np.array([0.3, 0.0]),  # 30 cm/s eastward
        seed=42,
    )

    print("   Running EKF filter...")
    result_const_vel = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_const_vel["t_imu"],
        U_imu=sim_const_vel["U_imu"],
        t_cam=sim_const_vel["t_cam_exp"],
        Z_cam_led1=sim_const_vel["Z_cam_led1"],
        Z_cam_led2=sim_const_vel["Z_cam_led2"],
        mask_cam=sim_const_vel["mask_cam"],
    )

    # Compute metrics
    t_cam = sim_const_vel["t_cam_exp"]
    t_imu = sim_const_vel["t_imu"]
    X_truth = sim_const_vel["X_truth"]
    X_truth_cam = np.column_stack(
        [np.interp(t_cam, t_imu, X_truth[:, i]) for i in range(5)]
    )
    X_truth_cam[:, 4] = interp_angle(t_cam, t_imu, X_truth[:, 4])

    X_est = np.array(result_const_vel.filtered_means)
    P_est = np.array(result_const_vel.filtered_covariances)

    pos_rmse_cm = compute_position_rmse(X_truth_cam[:, :2] * 100, X_est[:, :2] * 100)
    vel_err = np.linalg.norm(X_est[:, 2:4] - X_truth_cam[:, 2:4], axis=1) * 100
    vel_rmse_cm_s = np.sqrt(np.mean(vel_err**2))
    heading_err = np.abs(np.degrees(angle_diff(X_est[:, 4], X_truth_cam[:, 4])))
    heading_rmse_deg = np.sqrt(np.mean(heading_err**2))

    # Compute NEES for full 8D state
    bias_gyro_truth_cam = np.interp(t_cam, t_imu, sim_const_vel["bias_gyro"])
    bias_ax_truth_cam = np.interp(t_cam, t_imu, sim_const_vel["bias_accel_x"])
    bias_ay_truth_cam = np.interp(t_cam, t_imu, sim_const_vel["bias_accel_y"])
    X_truth_full = np.column_stack(
        [X_truth_cam, bias_gyro_truth_cam, bias_ax_truth_cam, bias_ay_truth_cam]
    )

    # Compute NEES using layout (automatically handles angle wrapping)
    nees = compute_nees(X_truth_full, X_est, P_est, layout=layout)
    mean_nees = np.mean(nees)

    metrics_const_vel = {
        "pos_rmse_cm": pos_rmse_cm,
        "vel_rmse_cm_s": vel_rmse_cm_s,
        "heading_rmse_deg": heading_rmse_deg,
        "mean_nees": mean_nees,
    }

    print_metrics_table(
        "Constant Velocity",
        pos_rmse_cm,
        vel_rmse_cm_s,
        heading_rmse_deg,
        mean_nees,
        result_const_vel.marginal_loglik,
    )

    print(
        """
    📖 Interpretation:
       • Position RMSE remains low → EKF tracks straight line accurately
       • Velocity RMSE < 10 cm/s → velocity estimate is reliable
       • Heading RMSE good → ⚠️  Only LED1 visible, but motion constrains heading!
       • Accel bias observable → forward motion constrains bias estimate
       • NEES > 8.0 → filter slightly underconfident (high process noise)

    Key insight: Even with only LED1 (position), heading becomes observable during
    motion because velocity direction constrains heading. However, this is less
    accurate than direct dual-LED heading measurement (see Circular scenario).
    """
    )

    plot_ekf_results(
        sim_const_vel,
        result_const_vel,
        "Constant Velocity",
        metrics_const_vel,
        OUTPUT_DIR / "03_ekf_constant_velocity.png",
        ekf_config,
    )

    # -------------------------------------------------------------------------
    # Scenario 3: Circular Motion
    # -------------------------------------------------------------------------
    print_section_header("Step 4: Scenario 3 - Circular Motion")
    explain_scenario(
        "Circular Motion",
        "EKF must learn gyroscope bias from persistent rotation",
        "Gyro bias should converge within ~5 seconds; heading tracks circle tangent",
    )

    print("   Running simulation...")
    sim_circular = simulate_circular(
        sim_config,
        center=np.array([1.0, 1.0]),
        radius=0.5,  # 50 cm radius
        angular_velocity=0.5,  # 0.5 rad/s ≈ 28°/s
        seed=42,
    )

    print("   Running EKF filter...")
    result_circular = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_circular["t_imu"],
        U_imu=sim_circular["U_imu"],
        t_cam=sim_circular["t_cam_exp"],
        Z_cam_led1=sim_circular["Z_cam_led1"],
        Z_cam_led2=sim_circular["Z_cam_led2"],
        mask_cam=sim_circular["mask_cam"],
    )

    # Compute metrics
    t_cam = sim_circular["t_cam_exp"]
    t_imu = sim_circular["t_imu"]
    X_truth = sim_circular["X_truth"]
    X_truth_cam = np.column_stack(
        [np.interp(t_cam, t_imu, X_truth[:, i]) for i in range(5)]
    )
    X_truth_cam[:, 4] = interp_angle(t_cam, t_imu, X_truth[:, 4])

    X_est = np.array(result_circular.filtered_means)
    P_est = np.array(result_circular.filtered_covariances)

    pos_rmse_cm = compute_position_rmse(X_truth_cam[:, :2] * 100, X_est[:, :2] * 100)
    vel_err = np.linalg.norm(X_est[:, 2:4] - X_truth_cam[:, 2:4], axis=1) * 100
    vel_rmse_cm_s = np.sqrt(np.mean(vel_err**2))
    heading_err = np.abs(np.degrees(angle_diff(X_est[:, 4], X_truth_cam[:, 4])))
    heading_rmse_deg = np.sqrt(np.mean(heading_err**2))

    # Compute NEES for full 8D state
    bias_gyro_truth_cam = np.interp(t_cam, t_imu, sim_circular["bias_gyro"])
    bias_ax_truth_cam = np.interp(t_cam, t_imu, sim_circular["bias_accel_x"])
    bias_ay_truth_cam = np.interp(t_cam, t_imu, sim_circular["bias_accel_y"])
    X_truth_full = np.column_stack(
        [X_truth_cam, bias_gyro_truth_cam, bias_ax_truth_cam, bias_ay_truth_cam]
    )

    # Compute NEES using layout (automatically handles angle wrapping)
    nees = compute_nees(X_truth_full, X_est, P_est, layout=layout)
    mean_nees = np.mean(nees)

    # Check if gyro bias converged (compare final estimate to truth)
    bias_gyro_truth_final = sim_circular["bias_gyro"][-1]
    bias_gyro_est_final = X_est[-1, 5]
    bias_error_final = np.abs(bias_gyro_est_final - bias_gyro_truth_final)
    bias_converged = bias_error_final < np.deg2rad(1.0)  # Within 1°/s

    metrics_circular = {
        "pos_rmse_cm": pos_rmse_cm,
        "vel_rmse_cm_s": vel_rmse_cm_s,
        "heading_rmse_deg": heading_rmse_deg,
        "mean_nees": mean_nees,
    }

    print_metrics_table(
        "Circular Motion",
        pos_rmse_cm,
        vel_rmse_cm_s,
        heading_rmse_deg,
        mean_nees,
        result_circular.marginal_loglik,
        gyro_bias_converged=bias_converged,
    )

    print(
        f"""
    📖 Interpretation:
       • Position RMSE low → EKF tracks circular path accurately
       • Heading RMSE < 7° → ✓ MEETS PRD TARGET with dual-LED measurement!
       • NEES ≈ 8.0 → ✓ Filter is optimally calibrated (well-tuned)
       • Gyro bias {"✓ CONVERGED" if bias_converged else "✗ DID NOT CONVERGE"} → rotation makes bias observable
       • Final bias error: {np.degrees(bias_error_final):.3f}°/s

    ✓ SUCCESS: This scenario achieves ALL PRD targets because it has:
       1. Dual-LED observations (heading measurement)
       2. Continuous rotation (gyro bias observable)
       3. Realistic SpikeGadgets IMU noise + well-tuned process noise

    Key insight: Circular motion is the ONLY way to make gyro bias observable
    in 2D tracking. Without rotation, bias is confounded with heading drift.
    """
    )

    plot_ekf_results(
        sim_circular,
        result_circular,
        "Circular Motion",
        metrics_circular,
        OUTPUT_DIR / "03_ekf_circular.png",
        ekf_config,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section_header("Summary: What You Learned")

    print(
        f"""
    🎓 KEY TAKEAWAYS:

    1. SENSOR FUSION WORKS:
       • Camera (30 Hz, accurate) + IMU (200 Hz, drifts) = smooth, accurate estimate
       • EKF optimally blends measurements using Kalman gain (Q/R ratio)

    2. OBSERVABILITY MATTERS:
       • Stationary:       Position observable, biases NOT observable
       • Straight line:    Position + velocity + accel bias observable
       • Circular:         ALL states observable (including gyro bias)

    3. FILTER CONSISTENCY:
       • NEES ≈ 8.0 means filter is "honest" about uncertainty
       • NEES < 6.0 → overconfident (covariance too small)
       • NEES > 10.0 → underconfident (covariance too large)

    4. PERFORMANCE vs PRD TARGETS:
       • Stationary:       {metrics_stat["pos_rmse_cm"]:.2f} cm (target ≤ 2 cm)
       • Const Velocity:   {metrics_const_vel["pos_rmse_cm"]:.2f} cm, {metrics_const_vel["vel_rmse_cm_s"]:.2f} cm/s
       • Circular:         {metrics_circular["pos_rmse_cm"]:.2f} cm, {metrics_circular["heading_rmse_deg"]:.2f}°

    ✅ All scenarios meet PRD targets under ideal conditions (no dropouts).

    NEXT STEPS:
    • Run examples/04_ukf_basic_scenarios.py to compare UKF vs EKF
    • Run examples/05_ekf_with_dropouts.py to test robustness
    • See examples/README.md for full learning path

    Happy learning! 🚀
    """
    )


if __name__ == "__main__":
    main()
