"""
Generate visual for Slide 16: Smoother Comparison

Shows 2-panel comparison during dropout:
- Left: EKF forward-only filter (real-time)
- Right: RTS smoother (forward + backward, offline)

Demonstrates the power of offline smoothing: RTS uses future measurements
in the backward pass to improve estimates during camera dropouts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
from trodestrack.runtime.offline import rts_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
OUTPUT_DIR = Path(__file__).parent.parent / "visuals"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette
BLUE = "#2E86AB"
ORANGE = "#F77F00"
GREEN = "#06A77D"
RED = "#D62828"
GRAY = "#6C757D"


def generate_slide16():
    """2-panel smoother comparison: EKF vs RTS (both use IMU during dropout)"""

    # Generate simulation with controlled dropout
    np.random.seed(42)
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=104.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,  # Start with no dropout, create artificial dropout
        use_second_led=True,
    )
    sim = simulate_rat_imu(config)

    # Create artificial dropout period from t=12s to t=17s (5 seconds)
    t_cam = sim["t_cam_exp"]
    dropout_start = 12.0
    dropout_end = 17.0
    dropout_mask = (t_cam >= dropout_start) & (t_cam <= dropout_end)

    # Modify mask to create dropout
    mask_cam = sim["mask_cam"].copy()
    mask_cam[dropout_mask] = False

    # Get ground truth at camera times
    t_imu = sim["t_imu"]
    cam_indices = np.searchsorted(t_imu, t_cam)
    cam_indices = np.clip(cam_indices, 0, len(t_imu) - 1)
    pos_truth = sim["X_truth"][cam_indices, :2]

    # Run EKF
    ekf_config = EKFConfig()
    ekf_result = extended_kalman_filter(
        ekf_config,
        sim["t_imu"],
        sim["U_imu"],
        t_cam,
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        mask_cam,
    )

    # Run RTS smoother
    rts_result = rts_smoother(
        ekf_result,
        ekf_config,
        sim["t_imu"],
        sim["U_imu"],
        t_cam,
        mask_cam=mask_cam,
    )

    # Extract positions
    layout = get_layout("2d_full")
    pos_ekf = np.array(ekf_result.filtered_means[:, layout.pos_idx])
    pos_rts = np.array(rts_result.smoothed_means[:, layout.pos_idx])

    # Calculate drift at end of dropout period
    dropout_end_idx = np.where(t_cam >= dropout_end)[0][0]
    drift_ekf = np.linalg.norm(pos_ekf[dropout_end_idx] - pos_truth[dropout_end_idx])
    drift_rts = np.linalg.norm(pos_rts[dropout_end_idx] - pos_truth[dropout_end_idx])

    # Create 2-panel figure with subplot_mosaic - each panel gets a legend space
    mosaic = [["panel1", "leg1", "panel2", "leg2"]]

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(16, 7),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4, 1, 4, 1]},
    )

    axes = [axd["panel1"], axd["panel2"]]
    leg_axes = [axd["leg1"], axd["leg2"]]

    # Turn off axes for legend panels
    for ax in leg_axes:
        ax.axis("off")

    # Define zoom window around dropout period
    zoom_start = dropout_start - 2.0
    zoom_end = dropout_end + 2.0
    zoom_mask = (t_cam >= zoom_start) & (t_cam <= zoom_end)
    dropout_start_idx = np.argmin(np.abs(t_cam - dropout_start))

    # Panel 1: EKF (forward-only)
    ax1 = axes[0]
    ax1.plot(
        pos_truth[zoom_mask, 0],
        pos_truth[zoom_mask, 1],
        linewidth=4.5,
        color=BLUE,
        linestyle="--",
        alpha=0.7,
        label="Ground truth",
        zorder=1,
    )
    ax1.plot(
        pos_ekf[zoom_mask, 0],
        pos_ekf[zoom_mask, 1],
        linewidth=3,
        color=GREEN,
        alpha=0.8,
        label="EKF (forward only)",
        zorder=2,
    )

    # Highlight dropout region
    ax1.plot(
        pos_ekf[dropout_mask, 0],
        pos_ekf[dropout_mask, 1],
        linewidth=5,
        color=GREEN,
        alpha=0.4,
        zorder=3,
    )

    # Mark dropout start/end
    ax1.scatter(
        pos_ekf[dropout_start_idx, 0],
        pos_ekf[dropout_start_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
    )
    ax1.scatter(
        pos_ekf[dropout_end_idx, 0],
        pos_ekf[dropout_end_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
    )

    ax1.set_xlabel("X (m)", fontsize=16, weight="bold", labelpad=10)
    ax1.set_ylabel("Y (m)", fontsize=16, weight="bold", labelpad=10)
    ax1.set_title(
        f"EKF Filter (Real-Time)\nDrift: {drift_ekf:.3f}m",
        fontsize=18,
        weight="bold",
        color=GREEN,
    )
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Create legend in dedicated space
    handles1, labels1 = ax1.get_legend_handles_labels()
    leg_axes[0].legend(
        handles1,
        labels1,
        loc="center left",
        fontsize=13,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
    )

    # Panel 2: RTS Smoother
    ax2 = axes[1]
    ax2.plot(
        pos_truth[zoom_mask, 0],
        pos_truth[zoom_mask, 1],
        linewidth=4.5,
        color=BLUE,
        linestyle="--",
        alpha=0.7,
        label="Ground truth",
        zorder=1,
    )
    ax2.plot(
        pos_rts[zoom_mask, 0],
        pos_rts[zoom_mask, 1],
        linewidth=3,
        color=BLUE,
        alpha=0.9,
        label="RTS smoother",
        zorder=2,
    )

    # Highlight dropout region
    ax2.plot(
        pos_rts[dropout_mask, 0],
        pos_rts[dropout_mask, 1],
        linewidth=5,
        color=BLUE,
        alpha=0.4,
        zorder=3,
    )

    # Mark dropout start/end
    ax2.scatter(
        pos_rts[dropout_start_idx, 0],
        pos_rts[dropout_start_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
        label="Camera measurement",
    )
    ax2.scatter(
        pos_rts[dropout_end_idx, 0],
        pos_rts[dropout_end_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
        label="Camera measurement",
    )

    ax2.set_xlabel("X (m)", fontsize=16, weight="bold", labelpad=10)
    ax2.set_ylabel("Y (m)", fontsize=16, weight="bold", labelpad=10)
    ax2.set_title(
        f"RTS Smoother (Offline)\nDrift: {drift_rts:.3f}m",
        fontsize=18,
        weight="bold",
        color=BLUE,
        pad=10,
    )
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Create legend in dedicated space
    handles2, labels2 = ax2.get_legend_handles_labels()
    leg_axes[1].legend(
        handles2,
        labels2,
        loc="center left",
        fontsize=13,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
    )

    # Match axis limits across both panels
    all_x = np.concatenate(
        [
            pos_truth[zoom_mask, 0],
            pos_ekf[zoom_mask, 0],
            pos_rts[zoom_mask, 0],
        ]
    )
    all_y = np.concatenate(
        [
            pos_truth[zoom_mask, 1],
            pos_ekf[zoom_mask, 1],
            pos_rts[zoom_mask, 1],
        ]
    )
    x_margin = (all_x.max() - all_x.min()) * 0.15
    y_margin = (all_y.max() - all_y.min()) * 0.15

    for ax in axes:
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    # Overall title
    fig.suptitle(
        "Offline Smoothing: Backward Pass Uses Future Measurements",
        fontsize=22,
        weight="bold",
        y=1.02,
    )

    # Key insight - compare EKF vs RTS only
    improvement_pct = (drift_ekf - drift_rts) / drift_ekf * 100

    fig.text(
        0.5,
        -0.09,
        f"Key Insight: RTS smoother reduces drift by {improvement_pct:.0f}% vs EKF by using future camera measurements!\n"
        f"Both methods use IMU during dropout, but RTS backward pass retroactively corrects estimates.\n"
        f"Use RTS smoother offline for best accuracy (EKF: {drift_ekf:.3f}m → RTS: {drift_rts:.3f}m)",
        fontsize=18,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=BLUE, alpha=0.1),
    )
    output_path = OUTPUT_DIR / "slide16_smoother_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Print statistics
    print(f"  EKF drift: {drift_ekf:.3f} m")
    print(f"  RTS drift: {drift_rts:.3f} m ({improvement_pct:.1f}% improvement)")
    print("  Both methods use IMU during dropout - RTS adds backward pass")


if __name__ == "__main__":
    print("Generating Slide 16: Smoother Comparison...")
    print()
    generate_slide16()
    print()
    print("✅ Slide 16 visual generated!")
    print(f"   Output: {OUTPUT_DIR / 'slide16_smoother_comparison.png'}")
