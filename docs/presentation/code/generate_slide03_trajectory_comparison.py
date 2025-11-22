"""
Generate visual for Slide 3: Trajectory Comparison

Shows ground truth vs noisy vision-only observations with dropout gaps.
Demonstrates the core problem that TrodesTrack solves.

PRESENTATION OPTIMIZED:
- 16:9 aspect ratio (10" × 5.625")
- Large fonts (title 32pt, labels 24pt)
- Bold lines (4pt width)
- No overlapping elements
"""

# Import presentation utilities
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

sys.path.insert(0, str(Path(__file__).parent))
from presentation_utils import (
    COLORS,
    add_title,
    clean_axis,
    plot_trajectory,
    save_presentation_figure,
    scatter_points,
)

OUTPUT_DIR = Path(__file__).parent.parent / "visuals"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    mask_led1 = sim["mask_led1"]

    # Valid observations only
    valid_obs = mask_led1
    pos_observed = Z_cam[valid_obs, :]

    # Create figure with mosaic layout for better legend control
    # Layout: Two main panels + two legend panels on right
    mosaic = """
    AL
    BR
    """

    _fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 5.625),  # Wider for legend space
        dpi=150,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [5, 0.8], "height_ratios": [1, 1]},
    )

    axes = [axd["A"], axd["B"]]
    legend_axes = [axd["L"], axd["R"]]

    # Left panel: Ground truth (smooth, complete)
    ax1 = axes[0]
    plot_trajectory(
        ax1,
        pos_truth[:, 0],
        pos_truth[:, 1],
        color=COLORS["blue"],
        linewidth=4,
        label="Ground truth",
    )

    # Mark start and end (larger markers for visibility)
    scatter_points(
        ax1,
        [pos_truth[0, 0]],
        [pos_truth[0, 1]],
        color=COLORS["green"],
        size=300,
        label="Start",
    )
    scatter_points(
        ax1,
        [pos_truth[-1, 0]],
        [pos_truth[-1, 1]],
        color=COLORS["orange"],
        size=300,
        label="End",
    )

    # Title with more padding to avoid overlap with suptitle
    add_title(ax1, "Ground Truth", fontsize=24, color=COLORS["blue"], pad=20)
    ax1.set_xlabel("X position (m)", fontsize=20, fontweight="bold", labelpad=12)
    ax1.set_ylabel("Y position (m)", fontsize=20, fontweight="bold", labelpad=12)
    ax1.tick_params(labelsize=16)  # Larger tick labels
    ax1.set_aspect("equal")
    clean_axis(ax1, grid=False)

    # Move legend to dedicated panel
    legend_axes[0].axis("off")
    handles1, labels1 = ax1.get_legend_handles_labels()
    legend_axes[0].legend(
        handles1, labels1, loc="center left", fontsize=18, frameon=False
    )

    # Right panel: Noisy observations with gaps
    ax2 = axes[1]

    # Show hidden truth in background (very faint)
    ax2.plot(
        pos_truth[:, 0],
        pos_truth[:, 1],
        linewidth=2,
        color=COLORS["gray"],
        alpha=0.2,
        linestyle=":",
        zorder=1,
    )

    # Plot observed points only (with noise, gaps)
    scatter_points(
        ax2,
        pos_observed[:, 0],
        pos_observed[:, 1],
        color=COLORS["red"],
        size=50,
        alpha=0.6,
        label="Camera obs",
    )

    # Find gaps and annotate ONLY the longest one (avoid overlap)
    obs_indices = np.where(valid_obs)[0]
    gaps = np.diff(obs_indices)
    gap_starts = obs_indices[:-1][gaps > 1]
    gap_ends = obs_indices[1:][gaps > 1]

    # Find the LONGEST gap only (minimal annotation)
    if len(gap_starts) > 0:
        gap_durations = [
            t_cam[end] - t_cam[start]
            for start, end in zip(gap_starts, gap_ends, strict=False)
        ]
        longest_gap_idx = np.argmax(gap_durations)

        start_idx = gap_starts[longest_gap_idx]
        end_idx = gap_ends[longest_gap_idx]
        pos_start = pos_truth[start_idx, :]
        pos_end = pos_truth[end_idx, :]
        gap_duration = t_cam[end_idx] - t_cam[start_idx]

        # Draw arrow spanning the gap
        ax2.annotate(
            "",
            xy=pos_end,
            xytext=pos_start,
            arrowprops=dict(arrowstyle="<->", color=COLORS["orange"], lw=4, alpha=0.9),
            zorder=50,
        )

        # Label the gap - positioned carefully to avoid data overlap
        mid_pos = (pos_start + pos_end) / 2
        # Place label ABOVE the trajectory if possible
        y_offset = 0.5 if mid_pos[1] < pos_truth[:, 1].max() - 0.3 else -0.3

        ax2.text(
            mid_pos[0],
            mid_pos[1] + y_offset,
            f"{gap_duration:.1f}s gap",
            fontsize=18,
            fontweight="bold",
            color=COLORS["orange"],
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="white",
                alpha=0.95,
                edgecolor=COLORS["orange"],
                linewidth=3,
            ),
            zorder=100,
        )

    # Mark start and end
    scatter_points(
        ax2,
        [pos_observed[0, 0]],
        [pos_observed[0, 1]],
        color=COLORS["green"],
        size=300,
    )
    scatter_points(
        ax2,
        [pos_observed[-1, 0]],
        [pos_observed[-1, 1]],
        color=COLORS["orange"],
        size=300,
    )

    # Shorter title to avoid overlap
    add_title(ax2, "Camera Observations", fontsize=24, color=COLORS["red"], pad=20)
    ax2.set_xlabel("X position (m)", fontsize=20, fontweight="bold", labelpad=12)
    ax2.set_ylabel("")  # No y-label on right panel (shared with left)
    ax2.tick_params(labelsize=16)  # Larger tick labels
    ax2.set_aspect("equal")
    clean_axis(ax2, grid=False)

    # Move legend to dedicated panel
    legend_axes[1].axis("off")
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend_axes[1].legend(
        handles2, labels2, loc="center left", fontsize=18, frameon=False
    )

    # Make axes equal for fair comparison
    all_x = pos_truth[:, 0]
    all_y = pos_truth[:, 1]
    x_margin = (all_x.max() - all_x.min()) * 0.15  # Larger margin to prevent overlap
    y_margin = (all_y.max() - all_y.min()) * 0.15

    for ax in axes:
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    # No suptitle - panel titles are sufficient and avoid overlap
    # Slide context (PowerPoint slide title) will provide "The Tracking Problem"
    # Panel titles "Ground Truth" and "Camera Observations" are self-explanatory

    # Save
    output_path = OUTPUT_DIR / "slide03_trajectory_comparison.png"
    save_presentation_figure(output_path)
    plt.close()

    # Print statistics
    dropout_rate = 1 - valid_obs.sum() / len(valid_obs)
    num_gaps = len(gap_starts)
    print(f"  Dropout rate: {dropout_rate * 100:.1f}%")
    print(f"  Number of gaps: {num_gaps}")


if __name__ == "__main__":
    print("Generating Slide 3: Trajectory Comparison...")
    generate_slide03()
    print("✅ Slide 3 visual generated!")
