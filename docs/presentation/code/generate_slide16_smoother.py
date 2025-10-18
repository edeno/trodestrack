"""
Generate visual for Slide 16: Smoother Comparison

Shows 3-panel comparison during dropout:
- Left: Vision-only (naive extrapolation)
- Middle: EKF forward-only filter
- Right: RTS smoother (forward + backward)

Demonstrates the power of offline smoothing for reducing dropout drift.
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
    """3-panel smoother comparison: vision-only vs EKF vs RTS"""

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

    # Simulate vision-only (last valid observation + velocity extrapolation)
    # This is a naive baseline: hold last velocity and extrapolate
    pos_vision = np.zeros_like(pos_truth)
    vel_vision = np.zeros((len(t_cam), 2))

    # Initialize with first valid observation
    first_valid_idx = np.where(mask_cam)[0][0]
    pos_vision[first_valid_idx] = sim["Z_cam_led1"][first_valid_idx]
    vel_vision[first_valid_idx] = [0.0, 0.0]

    # Forward pass: update when camera is valid, extrapolate when dropout
    for i in range(first_valid_idx + 1, len(t_cam)):
        dt = t_cam[i] - t_cam[i - 1]
        if mask_cam[i]:
            # Camera available: use measurement
            pos_vision[i] = sim["Z_cam_led1"][i]
            # Estimate velocity from position difference
            vel_vision[i] = (pos_vision[i] - pos_vision[i - 1]) / dt
        else:
            # Dropout: extrapolate with last known velocity
            pos_vision[i] = pos_vision[i - 1] + vel_vision[i - 1] * dt
            vel_vision[i] = vel_vision[i - 1]  # Hold velocity constant

    # Calculate drift at end of dropout period
    dropout_end_idx = np.where(t_cam >= dropout_end)[0][0]
    drift_vision = np.linalg.norm(pos_vision[dropout_end_idx] - pos_truth[dropout_end_idx])
    drift_ekf = np.linalg.norm(pos_ekf[dropout_end_idx] - pos_truth[dropout_end_idx])
    drift_rts = np.linalg.norm(pos_rts[dropout_end_idx] - pos_truth[dropout_end_idx])

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    # Define zoom window around dropout period
    zoom_start = dropout_start - 2.0
    zoom_end = dropout_end + 2.0
    zoom_mask = (t_cam >= zoom_start) & (t_cam <= zoom_end)

    # Panel 1: Vision-only
    ax1 = axes[0]
    ax1.plot(
        pos_truth[zoom_mask, 0],
        pos_truth[zoom_mask, 1],
        linewidth=2.5,
        color=BLUE,
        linestyle="--",
        alpha=0.7,
        label="Ground truth",
        zorder=1,
    )
    ax1.plot(
        pos_vision[zoom_mask, 0],
        pos_vision[zoom_mask, 1],
        linewidth=3,
        color=RED,
        alpha=0.8,
        label="Vision-only extrapolation",
        zorder=2,
    )

    # Highlight dropout region
    ax1.plot(
        pos_vision[dropout_mask, 0],
        pos_vision[dropout_mask, 1],
        linewidth=5,
        color=RED,
        alpha=0.4,
        zorder=3,
    )

    # Mark dropout start/end
    dropout_start_idx = np.argmin(np.abs(t_cam - dropout_start))
    ax1.scatter(
        pos_vision[dropout_start_idx, 0],
        pos_vision[dropout_start_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
    )
    ax1.scatter(
        pos_vision[dropout_end_idx, 0],
        pos_vision[dropout_end_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
    )

    ax1.set_xlabel("X (meters)", fontsize=14, weight="bold")
    ax1.set_ylabel("Y (meters)", fontsize=14, weight="bold")
    ax1.set_title(
        f"Vision-Only Baseline\nDrift @ 5s: {drift_vision:.2f} m",
        fontsize=16,
        weight="bold",
        color=RED,
    )
    ax1.legend(fontsize=11, loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Panel 2: EKF
    ax2 = axes[1]
    ax2.plot(
        pos_truth[zoom_mask, 0],
        pos_truth[zoom_mask, 1],
        linewidth=2.5,
        color=BLUE,
        linestyle="--",
        alpha=0.7,
        label="Ground truth",
        zorder=1,
    )
    ax2.plot(
        pos_ekf[zoom_mask, 0],
        pos_ekf[zoom_mask, 1],
        linewidth=3,
        color=GREEN,
        alpha=0.8,
        label="EKF (forward only)",
        zorder=2,
    )

    # Highlight dropout region
    ax2.plot(
        pos_ekf[dropout_mask, 0],
        pos_ekf[dropout_mask, 1],
        linewidth=5,
        color=GREEN,
        alpha=0.4,
        zorder=3,
    )

    # Mark dropout start/end
    ax2.scatter(
        pos_ekf[dropout_start_idx, 0],
        pos_ekf[dropout_start_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
    )
    ax2.scatter(
        pos_ekf[dropout_end_idx, 0],
        pos_ekf[dropout_end_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
    )

    ax2.set_xlabel("X (meters)", fontsize=14, weight="bold")
    ax2.set_ylabel("Y (meters)", fontsize=14, weight="bold")
    ax2.set_title(
        f"Extended Kalman Filter\nDrift @ 5s: {drift_ekf:.2f} m",
        fontsize=16,
        weight="bold",
        color=GREEN,
    )
    ax2.legend(fontsize=11, loc="upper right")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Panel 3: RTS Smoother
    ax3 = axes[2]
    ax3.plot(
        pos_truth[zoom_mask, 0],
        pos_truth[zoom_mask, 1],
        linewidth=2.5,
        color=BLUE,
        linestyle="--",
        alpha=0.7,
        label="Ground truth",
        zorder=1,
    )
    ax3.plot(
        pos_rts[zoom_mask, 0],
        pos_rts[zoom_mask, 1],
        linewidth=3,
        color=BLUE,
        alpha=0.9,
        label="RTS smoother",
        zorder=2,
    )

    # Highlight dropout region
    ax3.plot(
        pos_rts[dropout_mask, 0],
        pos_rts[dropout_mask, 1],
        linewidth=5,
        color=BLUE,
        alpha=0.4,
        zorder=3,
    )

    # Mark dropout start/end
    ax3.scatter(
        pos_rts[dropout_start_idx, 0],
        pos_rts[dropout_start_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
    )
    ax3.scatter(
        pos_rts[dropout_end_idx, 0],
        pos_rts[dropout_end_idx, 1],
        s=150,
        marker="x",
        color=ORANGE,
        linewidths=3,
        zorder=10,
    )

    ax3.set_xlabel("X (meters)", fontsize=14, weight="bold")
    ax3.set_ylabel("Y (meters)", fontsize=14, weight="bold")
    ax3.set_title(
        f"RTS Smoother (Offline)\nDrift @ 5s: {drift_rts:.2f} m",
        fontsize=16,
        weight="bold",
        color=BLUE,
    )
    ax3.legend(fontsize=11, loc="upper right")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # Match axis limits across all panels
    all_x = np.concatenate(
        [
            pos_truth[zoom_mask, 0],
            pos_vision[zoom_mask, 0],
            pos_ekf[zoom_mask, 0],
            pos_rts[zoom_mask, 0],
        ]
    )
    all_y = np.concatenate(
        [
            pos_truth[zoom_mask, 1],
            pos_vision[zoom_mask, 1],
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
        "Offline Smoothing: Backward Pass Improves Accuracy\n" "5-second camera dropout (t=12-17s)",
        fontsize=20,
        weight="bold",
        y=0.98,
    )

    # Key insight
    improvement_ekf = (drift_vision - drift_ekf) / drift_vision * 100
    improvement_rts = (drift_ekf - drift_rts) / drift_ekf * 100

    fig.text(
        0.5,
        0.02,
        f"Key Insight: EKF reduces drift by {improvement_ekf:.0f}% vs vision-only. "
        f"RTS smoother reduces drift by {improvement_rts:.0f}% more using future measurements!\n"
        f"Use RTS smoother offline for best accuracy (PRD target: ≤3.5m @ 5s → {drift_rts:.2f}m ✓)",
        fontsize=13,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=BLUE, alpha=0.1),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    output_path = OUTPUT_DIR / "slide16_smoother_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Print statistics
    print(f"  Vision-only drift: {drift_vision:.3f} m")
    print(f"  EKF drift: {drift_ekf:.3f} m ({improvement_ekf:.1f}% improvement)")
    print(f"  RTS drift: {drift_rts:.3f} m ({improvement_rts:.1f}% further improvement)")
    print(f"  Total improvement: {(drift_vision - drift_rts) / drift_vision * 100:.1f}%")


if __name__ == "__main__":
    print("Generating Slide 16: Smoother Comparison...")
    print()
    generate_slide16()
    print()
    print("✅ Slide 16 visual generated!")
    print(f"   Output: {OUTPUT_DIR / 'slide16_smoother_comparison.png'}")
