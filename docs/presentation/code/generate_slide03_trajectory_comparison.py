"""
Generate visual for Slide 3: Trajectory Comparison

Shows ground truth vs noisy vision-only observations with dropout gaps.
Demonstrates the core problem that TrodesTrack solves.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def generate_slide03():
    """Ground truth vs noisy camera observations with gaps"""

    # Generate realistic simulation with dropout
    np.random.seed(42)
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=104.0,
        fs_cam=30.0,
        cam_dropout_prob=0.30,  # 30% dropout to show gaps clearly
        use_second_led=True,
    )
    sim = simulate_rat_imu(config)

    # Extract data
    t_cam = sim["t_cam_exp"]
    t_imu = sim["t_imu"]

    # Ground truth at camera times
    cam_indices = np.searchsorted(t_imu, t_cam)
    cam_indices = np.clip(cam_indices, 0, len(t_imu) - 1)
    pos_truth = sim["X_truth"][cam_indices, :2]

    # Noisy camera observations (LED1)
    Z_cam = sim["Z_cam_led1"]
    mask_led1 = sim["mask_led1"]  # LED1 validity mask

    # Valid observations only
    valid_obs = mask_led1
    pos_observed = Z_cam[valid_obs, :]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left panel: Ground truth (smooth, complete)
    ax1 = axes[0]
    ax1.plot(
        pos_truth[:, 0], pos_truth[:, 1], linewidth=3, color=BLUE, label="Ground truth", zorder=10
    )

    # Mark start and end
    ax1.scatter(
        pos_truth[0, 0],
        pos_truth[0, 1],
        s=200,
        marker="o",
        color=GREEN,
        edgecolor="black",
        linewidth=2,
        zorder=20,
        label="Start",
    )
    ax1.scatter(
        pos_truth[-1, 0],
        pos_truth[-1, 1],
        s=200,
        marker="s",
        color=ORANGE,
        edgecolor="black",
        linewidth=2,
        zorder=20,
        label="End",
    )

    ax1.set_xlabel("X (meters)", fontsize=16, weight="bold")
    ax1.set_ylabel("Y (meters)", fontsize=16, weight="bold")
    ax1.set_title(
        "Ground Truth Trajectory\n(What we want to estimate)",
        fontsize=18,
        weight="bold",
        color=BLUE,
    )
    ax1.legend(fontsize=14, loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right panel: Noisy observations with gaps
    ax2 = axes[1]

    # Show gaps by plotting ground truth in background (faint)
    ax2.plot(
        pos_truth[:, 0],
        pos_truth[:, 1],
        linewidth=1,
        color=GRAY,
        alpha=0.2,
        linestyle="--",
        label="Hidden truth",
    )

    # Plot observed points only (with noise, gaps)
    ax2.scatter(
        pos_observed[:, 0],
        pos_observed[:, 1],
        s=30,
        color=RED,
        alpha=0.6,
        label="Camera observations",
        zorder=10,
    )

    # Highlight dropout regions by finding gaps
    # Find consecutive observations
    obs_indices = np.where(valid_obs)[0]
    gaps = np.diff(obs_indices)
    gap_starts = obs_indices[:-1][gaps > 1]  # Indices where gaps start
    gap_ends = obs_indices[1:][gaps > 1]  # Indices where gaps end

    # Mark first few gaps with annotations
    for _, (start_idx, end_idx) in enumerate(zip(gap_starts[:3], gap_ends[:3], strict=False)):
        # Get positions at gap boundaries
        pos_start = pos_truth[start_idx, :]
        pos_end = pos_truth[end_idx, :]
        gap_duration = t_cam[end_idx] - t_cam[start_idx]

        # Draw arrow spanning the gap
        ax2.annotate(
            "",
            xy=pos_end,
            xytext=pos_start,
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.5, alpha=0.7),
        )

        # Label the gap
        mid_pos = (pos_start + pos_end) / 2
        ax2.text(
            mid_pos[0],
            mid_pos[1] + 0.3,
            f"Gap: {gap_duration:.1f}s",
            fontsize=11,
            weight="bold",
            color=ORANGE,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor=ORANGE),
        )

    # Mark start and end
    ax2.scatter(
        pos_observed[0, 0],
        pos_observed[0, 1],
        s=200,
        marker="o",
        color=GREEN,
        edgecolor="black",
        linewidth=2,
        zorder=20,
        label="Start",
    )
    ax2.scatter(
        pos_observed[-1, 0],
        pos_observed[-1, 1],
        s=200,
        marker="s",
        color=ORANGE,
        edgecolor="black",
        linewidth=2,
        zorder=20,
        label="End",
    )

    ax2.set_xlabel("X (meters)", fontsize=16, weight="bold")
    ax2.set_ylabel("Y (meters)", fontsize=16, weight="bold")
    ax2.set_title(
        "Vision-Only Observations\n(Noisy + Missing Data)", fontsize=18, weight="bold", color=RED
    )
    ax2.legend(fontsize=14, loc="upper right")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Make axes equal for fair comparison
    all_x = pos_truth[:, 0]
    all_y = pos_truth[:, 1]
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1

    for ax in axes:
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    # Overall title
    fig.suptitle(
        "The Tracking Problem: Noisy, Incomplete Camera Observations",
        fontsize=22,
        weight="bold",
        y=0.98,
    )

    # Key statistics
    dropout_rate = 1 - valid_obs.sum() / len(valid_obs)
    num_gaps = len(gap_starts)
    mean_gap = np.mean(
        [t_cam[end] - t_cam[start] for start, end in zip(gap_starts, gap_ends, strict=False)]
    )

    fig.text(
        0.5,
        0.02,
        f"Challenge: {dropout_rate*100:.0f}% dropout rate • "
        f"{num_gaps} gaps • Average gap: {mean_gap:.1f}s • "
        f"Camera noise ±2cm • How do we fill in the gaps?",
        fontsize=14,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=ORANGE, alpha=0.2),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    output_path = OUTPUT_DIR / "slide03_trajectory_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Print statistics
    print(f"  Dropout rate: {dropout_rate*100:.1f}%")
    print(f"  Number of gaps: {num_gaps}")
    print(f"  Mean gap duration: {mean_gap:.2f} s")
    print(f"  Valid observations: {valid_obs.sum()}/{len(valid_obs)}")


if __name__ == "__main__":
    print("Generating Slide 3: Trajectory Comparison...")
    print()
    generate_slide03()
    print()
    print("✅ Slide 3 visual generated!")
    print(f"   Output: {OUTPUT_DIR / 'slide03_trajectory_comparison.png'}")
