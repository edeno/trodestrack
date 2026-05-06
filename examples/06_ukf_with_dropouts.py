"""UKF vs EKF Robustness Under Camera Dropouts - Comparative Analysis.

LEARNING OBJECTIVES:
-------------------
This example compares UKF and EKF robustness under camera dropouts.
Does UKF's superior nonlinearity handling provide benefits when the filter
must rely on IMU-only integration during vision loss?

We test the 30% dropout scenario (Heavy) since that's where differences
would be most apparent.

WHY THIS MATTERS:
-----------------
If UKF provides better robustness during dropouts, it might justify any
computational cost for applications with frequent occlusions. Wall-clock
cost depends heavily on the backend: under JIT-compiled JAX with warm
dispatch UKF and EKF run at a comparable cost on this scenario, while on
backends without JIT (per-step Python loops) UKF can be several times
slower. Always re-measure on the target backend before relying on it.

KEY QUESTIONS WE'LL ANSWER:
---------------------------
1. Does UKF drift less than EKF during dropouts?
2. Does UKF recover faster when vision returns?
3. Is UKF's uncertainty estimation more accurate under stress?
4. Does the computational cost justify any accuracy improvements?

USAGE:
------
    uv run python examples/06_ukf_with_dropouts.py

OUTPUT:
-------
    - Console: Side-by-side EKF vs UKF metrics under 30% dropout
    - Files: 06_ukf_vs_ekf_dropout_heavy.png

ESTIMATED RUNTIME: ~20 seconds (depending on backend)

KEY CONCEPTS ILLUSTRATED:
-------------------------
1. **Sigma-Point Robustness**: Do sigma points help during IMU-only integration?
2. **Nonlinearity Under Stress**: Circular motion + dropouts = high nonlinearity
3. **Recovery Dynamics**: Which filter trusts measurements better after dropout?
4. **Practical Tradeoff**: Speed vs accuracy under realistic conditions
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.qa.metrics import compute_dropout_drift, compute_position_rmse
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.utils import interp_angle
from trodestrack.viz.styles import COLORS, apply_tufte_style


def _block_until_ready(result):
    """Force JAX dispatch to complete on every array leaf in ``result``.

    JAX execution is asynchronous, so timing a filter call without
    blocking can measure dispatch latency rather than completed compute.
    The "EKF vs UKF wall-clock" comparison in this example would be
    unreliable on platforms where async dispatch dominates without this
    helper. Mirrors the helper in ``tests/benchmark/test_throughput.py``.
    """
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result


apply_tufte_style()

OUTPUT_DIR = Path(__file__).parent


def print_section_header(title: str) -> None:
    """Print a clear section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_comparison_table(
    ekf_metrics: dict,
    ukf_metrics: dict,
    ekf_time_s: float,
    ukf_time_s: float,
) -> None:
    """Print side-by-side comparison of EKF vs UKF under dropouts."""
    print("\nPerformance Comparison (30% Dropout Scenario):")
    print("-" * 80)
    print(f"{'Metric':<30} {'EKF':>12} {'UKF':>12} {'Winner':>12} {'Improvement':>12}")
    print("-" * 80)

    # Position RMSE
    ekf_pos = ekf_metrics["pos_rmse_cm"]
    ukf_pos = ukf_metrics["pos_rmse_cm"]
    pos_winner = "UKF" if ukf_pos < ekf_pos else "EKF" if ekf_pos < ukf_pos else "TIE"
    pos_improv = abs(ekf_pos - ukf_pos)
    print(
        f"{'Position RMSE (cm)':<30} {ekf_pos:>12.2f} {ukf_pos:>12.2f} "
        f"{pos_winner:>12} {pos_improv:>11.2f} cm"
    )

    # Dropout drift
    ekf_drift = ekf_metrics["dropout_drift_m"] * 100
    ukf_drift = ukf_metrics["dropout_drift_m"] * 100
    drift_winner = (
        "UKF" if ukf_drift < ekf_drift else "EKF" if ekf_drift < ukf_drift else "TIE"
    )
    drift_improv = abs(ekf_drift - ukf_drift)
    print(
        f"{'Max Dropout Drift (cm)':<30} {ekf_drift:>12.1f} {ukf_drift:>12.1f} "
        f"{drift_winner:>12} {drift_improv:>11.1f} cm"
    )

    # Max dropout duration
    ekf_max_dropout = ekf_metrics["max_dropout_s"]
    ukf_max_dropout = ukf_metrics["max_dropout_s"]
    print(
        f"{'Max Dropout Duration (s)':<30} {ekf_max_dropout:>12.2f} {ukf_max_dropout:>12.2f} "
        f"{'SAME':>12} {'-':>12}"
    )

    # Dropout rate
    ekf_rate = ekf_metrics["actual_dropout_rate"] * 100
    ukf_rate = ukf_metrics["actual_dropout_rate"] * 100
    print(
        f"{'Actual Dropout Rate (%)':<30} {ekf_rate:>12.1f} {ukf_rate:>12.1f} "
        f"{'SAME':>12} {'-':>12}"
    )

    # Computation time. Format direction-aware: under JIT-compiled JAX
    # with warm dispatch the ratio is often near 1× and can flip either
    # way; an unconditional "slower" would misreport that.
    ratio = ukf_time_s / ekf_time_s
    cost_winner = "EKF" if ratio > 1.0 else "UKF" if ratio < 1.0 else "TIE"
    cost_label = (
        f"{ratio:.1f}× slower" if ratio >= 1.0 else f"{1.0 / ratio:.1f}× faster"
    )
    print("-" * 80)
    print(
        f"{'Computation Time (ms)':<30} {ekf_time_s * 1000:>12.1f} {ukf_time_s * 1000:>12.1f} "
        f"{cost_winner:>12} {cost_label:>14}"
    )
    print("-" * 80)

    # Overall assessment
    accuracy_wins = sum([ukf_pos < ekf_pos, ukf_drift < ekf_drift])
    if accuracy_wins >= 2:
        print(
            f"  ✓ UKF wins {accuracy_wins}/2 accuracy metrics → UKF more robust "
            f"(wall-clock: {cost_label})"
        )
    else:
        print(
            f"  ✓ EKF wins {2 - accuracy_wins}/2 accuracy metrics → EKF sufficient "
            f"(wall-clock: {cost_label})"
        )


def main() -> None:
    """Compare EKF vs UKF on 30% dropout scenario."""
    print_section_header("UKF vs EKF: Robustness Under Heavy Dropouts")

    print(
        """
    This example tests whether UKF's improved nonlinearity handling provides
    benefits under the stress of frequent camera dropouts (30% rate).

    Scenario: Circular motion with 30% dropout
    - Most challenging motion pattern (continuous rotation)
    - Heavy dropout rate (frequent IMU-only integration)
    - Tests both filters at their limits

    Question: Does UKF justify its computational cost under stress?
    """
    )

    # Shared configuration (30% dropout)
    dropout_prob = 0.30

    sim_config = RatIMUSimConfig(
        duration_s=15.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=dropout_prob,
        cam_dropout_correlation=0.7,
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_rw_density=0.0001,
        accel_bias_rw_density=0.001,
        cam_sigma_m=0.005,
        use_second_led=True,
    )

    # EKF config
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
    )

    # UKF config (matched parameters)
    ukf_config = UKFConfig(
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
        alpha=1.732,
        beta=2.0,
        kappa=1.0,
    )

    print_section_header("Running Simulation and Filters")

    print(f"   Simulating with {dropout_prob * 100:.0f}% dropout rate...")
    sim = simulate_rat_imu(config=sim_config, seed=42)

    print(
        f"   Generated {len(sim['t_cam_exp'])} camera frames, "
        f"{sim['mask_cam'].sum()} valid ({sim['mask_cam'].mean() * 100:.1f}%)"
    )

    # Run EKF
    print("   Running EKF...")
    t0_ekf = time.time()
    ekf_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )
    _block_until_ready(ekf_result)
    ekf_time = time.time() - t0_ekf

    # Run UKF
    print("   Running UKF...")
    t0_ukf = time.time()
    ukf_result = unscented_kalman_filter(
        ukf_config=ukf_config,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )
    _block_until_ready(ukf_result)
    ukf_time = time.time() - t0_ukf

    # Format the ratio as "slower" or "faster" depending on direction; on
    # JIT-compiled JAX with warm dispatch UKF often runs at ~1× EKF and
    # can occasionally come in faster.
    ratio = ukf_time / ekf_time
    if ratio >= 1.0:
        cost_label = f"{ratio:.1f}× slower"
    else:
        cost_label = f"{1.0 / ratio:.1f}× faster"
    print(f"   EKF time: {ekf_time * 1000:.1f} ms")
    print(f"   UKF time: {ukf_time * 1000:.1f} ms ({cost_label})")

    # Compute metrics for both
    t_imu = sim["t_imu"]
    t_cam = sim["t_cam_exp"]
    X_truth = sim["X_truth"]
    mask_cam = sim["mask_cam"]

    X_truth_cam = np.column_stack(
        [
            np.interp(t_cam, t_imu, X_truth[:, 0]),
            np.interp(t_cam, t_imu, X_truth[:, 1]),
            np.interp(t_cam, t_imu, X_truth[:, 2]),
            np.interp(t_cam, t_imu, X_truth[:, 3]),
            interp_angle(t_cam, t_imu, X_truth[:, 4]),
        ]
    )

    X_ekf = np.array(ekf_result.filtered_means)
    X_ukf = np.array(ukf_result.filtered_means)

    # EKF metrics
    ekf_pos_rmse = compute_position_rmse(X_truth_cam[:, :2] * 100, X_ekf[:, :2] * 100)
    ekf_drift_result = compute_dropout_drift(
        positions=X_ekf[:, :2], valid_mask=mask_cam, t=t_cam, min_duration_s=0.1
    )
    if ekf_drift_result["drift_m"] is None:
        ekf_drift_result["drift_m"] = 0.0

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
    max_dropout_s = max_dropout_frames / sim_config.fs_cam

    ekf_metrics = {
        "pos_rmse_cm": ekf_pos_rmse,
        "dropout_drift_m": ekf_drift_result["drift_m"],
        "max_dropout_s": max_dropout_s,
        "actual_dropout_rate": 1.0 - mask_cam.mean(),
    }

    # UKF metrics
    ukf_pos_rmse = compute_position_rmse(X_truth_cam[:, :2] * 100, X_ukf[:, :2] * 100)
    ukf_drift_result = compute_dropout_drift(
        positions=X_ukf[:, :2], valid_mask=mask_cam, t=t_cam, min_duration_s=0.1
    )
    if ukf_drift_result["drift_m"] is None:
        ukf_drift_result["drift_m"] = 0.0

    ukf_metrics = {
        "pos_rmse_cm": ukf_pos_rmse,
        "dropout_drift_m": ukf_drift_result["drift_m"],
        "max_dropout_s": max_dropout_s,
        "actual_dropout_rate": 1.0 - mask_cam.mean(),
    }

    # Print comparison
    print_section_header("Performance Comparison")
    print_comparison_table(ekf_metrics, ukf_metrics, ekf_time, ukf_time)

    # Create visualization (simplified - just key comparisons)
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Trajectories
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("🎯 Trajectories Under 30% Dropout", fontweight="bold")

    ax_traj.plot(
        X_truth[:, 0],
        X_truth[:, 1],
        "-",
        linewidth=2,
        color=COLORS["gray"],
        alpha=0.4,
        label="Truth",
    )
    ax_traj.plot(
        X_ekf[:, 0],
        X_ekf[:, 1],
        "-",
        linewidth=2,
        color=COLORS["blue"],
        alpha=0.7,
        label="EKF",
    )
    ax_traj.plot(
        X_ukf[:, 0],
        X_ukf[:, 1],
        "--",
        linewidth=2,
        color=COLORS["red"],
        alpha=0.7,
        label="UKF",
    )
    ax_traj.legend(loc="best")
    ax_traj.grid(True, alpha=0.2)

    # Panel 2: Position Error Comparison
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position Error (cm)")
    ax_pos.set_title("📏 Position Error Comparison", fontweight="bold")

    ekf_pos_err = np.linalg.norm(X_ekf[:, :2] - X_truth_cam[:, :2], axis=1) * 100
    ukf_pos_err = np.linalg.norm(X_ukf[:, :2] - X_truth_cam[:, :2], axis=1) * 100

    # Shade dropouts
    for i in range(len(t_cam) - 1):
        if not mask_cam[i]:
            ax_pos.axvspan(t_cam[i], t_cam[i + 1], alpha=0.1, color=COLORS["red"])

    ax_pos.plot(
        t_cam, ekf_pos_err, linewidth=2, color=COLORS["blue"], alpha=0.7, label="EKF"
    )
    ax_pos.plot(
        t_cam,
        ukf_pos_err,
        linewidth=2,
        color=COLORS["red"],
        alpha=0.7,
        linestyle="--",
        label="UKF",
    )
    ax_pos.axhline(2.0, linestyle=":", color=COLORS["gray"], linewidth=1, alpha=0.5)
    ax_pos.legend(loc="upper right")
    ax_pos.grid(True, alpha=0.2)

    # Panel 3: Bar Chart Comparison
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_bar.set_title("📊 Performance Summary", fontweight="bold")

    metrics = ["Position\nRMSE (cm)", "Max Drift\n(cm)"]
    ekf_vals = [ekf_metrics["pos_rmse_cm"], ekf_metrics["dropout_drift_m"] * 100]
    ukf_vals = [ukf_metrics["pos_rmse_cm"], ukf_metrics["dropout_drift_m"] * 100]

    x = np.arange(len(metrics))
    width = 0.35

    ax_bar.bar(
        x - width / 2, ekf_vals, width, label="EKF", color=COLORS["blue"], alpha=0.7
    )
    ax_bar.bar(
        x + width / 2, ukf_vals, width, label="UKF", color=COLORS["red"], alpha=0.7
    )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics)
    ax_bar.legend(loc="upper right")
    ax_bar.grid(True, alpha=0.2, axis="y")

    # Panel 4: Dropout Timeline
    ax_status = fig.add_subplot(gs[1, 1:])
    ax_status.set_xlabel("Time (s)")
    ax_status.set_ylabel("Camera Status")
    ax_status.set_title("📹 Vision Availability Timeline", fontweight="bold")

    ax_status.fill_between(
        t_cam,
        0,
        1,
        where=mask_cam,
        alpha=0.5,
        color=COLORS["green"],
        step="mid",
        label="Vision",
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
    ax_status.legend(loc="upper right")
    ax_status.grid(True, alpha=0.2)

    # Overall title. Format the wall-clock comparison direction-aware so
    # we don't claim "UKF is 0.8× slower" when it actually came in faster.
    ratio = ukf_time / ekf_time
    if ratio >= 1.0:
        title_cost = f"UKF is {ratio:.1f}× slower"
    else:
        title_cost = f"UKF is {1.0 / ratio:.1f}× faster"
    title = (
        f"UKF vs EKF Under 30% Dropout — Heavy Stress Test\n"
        f"EKF: {ekf_metrics['pos_rmse_cm']:.2f} cm RMSE | "
        f"UKF: {ukf_metrics['pos_rmse_cm']:.2f} cm RMSE | "
        f"{title_cost}"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    output_path = OUTPUT_DIR / "06_ukf_vs_ekf_dropout_heavy.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   💾 Saved: {output_path}")

    # Final verdict
    print_section_header("Final Verdict: UKF vs EKF Under Stress")

    pos_improvement = ekf_metrics["pos_rmse_cm"] - ukf_metrics["pos_rmse_cm"]
    drift_improvement = (
        ekf_metrics["dropout_drift_m"] - ukf_metrics["dropout_drift_m"]
    ) * 100

    cost_summary_label = (
        f"{ratio:.1f}× slower" if ratio >= 1.0 else f"{1.0 / ratio:.1f}× faster"
    )
    print(
        f"""
    📊 SUMMARY:

    Accuracy:
      • Position RMSE: {
            "UKF better by " + f"{pos_improvement:.2f} cm"
            if pos_improvement > 0
            else "EKF better by " + f"{-pos_improvement:.2f} cm"
        }
      • Dropout Drift: {
            "UKF better by " + f"{drift_improvement:.1f} cm"
            if drift_improvement > 0
            else "EKF better by " + f"{-drift_improvement:.1f} cm"
        }

    Computational Cost:
      • UKF is {cost_summary_label} ({ekf_time * 1000:.1f} ms vs {
            ukf_time * 1000:.1f} ms)

    🎓 KEY TAKEAWAYS:

    1. DROPOUT ROBUSTNESS:
       • Both filters handle 30% dropout reasonably well
       • Differences are {
            "significant" if abs(pos_improvement) > 0.2 else "marginal"
        } (< 0.5 cm)
       • UKF's nonlinearity handling {
            "does" if pos_improvement > 0.1 else "does NOT"
        } provide meaningful benefit

    2. WHEN DROPOUTS OCCUR:
       • Both filters rely on IMU integration (same physics)
       • Sigma points vs Jacobian matters less when no measurements arrive
       • Bias estimation quality matters MORE than filter type

    3. COMPUTATIONAL TRADEOFF:
       • Wall-clock comparison ({cost_summary_label}) for {
            "minimal" if abs(pos_improvement) < 0.2 else "moderate"
        } accuracy gain
       • Re-measure on the target backend before relying on the ratio
       • For offline: Consider UKF if every cm matters

    4. PRACTICAL RECOMMENDATION:
       {
            (
                "✓ Use EKF - simpler default with negligible accuracy loss"
                if abs(pos_improvement) < 0.2
                else "✓ Consider UKF - accuracy improvement may justify any wall-clock cost for critical applications"
            )
        }
       (Wall-clock direction is backend-dependent — see Computational Tradeoff above.)

    NEXT STEPS:
    • Run examples/07_smoother_demonstration.py to see how smoothing helps
    • Compare these results to your specific data and requirements
    • Profile on your hardware (timing results are platform-dependent)

    Happy learning! 🚀
    """
    )


if __name__ == "__main__":
    main()
