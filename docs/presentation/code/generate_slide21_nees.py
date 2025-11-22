"""
Generate visual for Slide 21: NEES Histogram

Shows NEES distribution vs χ² theoretical distribution.
Demonstrates quality assurance metrics for filter tuning.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
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


def generate_slide21():
    """NEES histogram vs χ² theoretical distribution"""

    # Generate realistic 60-second simulation
    np.random.seed(42)
    config = RatIMUSimConfig(
        duration_s=60.0,
        fs_imu=104.0,
        fs_cam=30.0,
        cam_dropout_prob=0.10,  # 10% realistic dropout
        use_second_led=True,
    )
    sim = simulate_rat_imu(config)

    # Run EKF
    ekf_config = EKFConfig()
    result = extended_kalman_filter(
        ekf_config,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        sim["mask_cam"],
    )

    # Get layout
    layout = get_layout("2d_full")

    # Downsample truth to camera rate
    t_cam = sim["t_cam_exp"]
    t_imu = sim["t_imu"]
    cam_indices = np.searchsorted(t_imu, t_cam)
    cam_indices = np.clip(cam_indices, 0, len(t_imu) - 1)
    X_truth_cam = sim["X_truth"][cam_indices]

    # Compute NEES for position + velocity + heading (5D subset)
    # State is [x, y, vx, vy, θ, b_gz, b_ax, b_ay] (8D)
    # We evaluate NEES on [x, y, vx, vy, θ] (5D) since biases aren't directly measurable
    state_indices = list(layout.pos_idx) + list(layout.vel_idx) + [layout.heading_idx]
    n_dims = len(state_indices)

    # Compute errors and NEES
    errors = np.array(
        result.filtered_means[:, state_indices] - X_truth_cam[:, state_indices]
    )

    # Handle angle wrapping for heading (last index in state_indices)
    errors[:, 4] = np.arctan2(np.sin(errors[:, 4]), np.cos(errors[:, 4]))

    # Compute NEES manually
    nees_values = []
    for i in range(len(errors)):
        error = errors[i]
        cov = result.filtered_covariances[i][np.ix_(state_indices, state_indices)]

        # NEES = e^T P^{-1} e
        try:
            cov_inv = np.linalg.inv(cov)
            nees = error @ cov_inv @ error
            nees_values.append(nees)
        except np.linalg.LinAlgError:
            # Singular covariance, skip
            pass

    nees_values = np.array(nees_values)

    # Create figure with subplot_mosaic - main plot with dedicated legend/stats area
    mosaic = [["main", "legend"]]

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 5.625),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [3, 1]},
    )

    ax = axd["main"]
    ax_legend = axd["legend"]

    # Plot histogram of NEES values
    bins = np.linspace(0, 20, 50)
    _counts, _bin_edges, _patches = ax.hist(
        nees_values,
        bins=bins,
        density=True,
        alpha=0.7,
        color=BLUE,
        edgecolor="black",
        linewidth=3.5,
        label=f"Observed NEES (n={len(nees_values)})",
    )

    # Overlay χ² theoretical distribution
    x = np.linspace(0, 20, 200)
    chi2_pdf = stats.chi2.pdf(x, df=n_dims)
    ax.plot(
        x,
        chi2_pdf,
        linewidth=3,
        color=RED,
        label=f"χ² distribution (df={n_dims})",
        zorder=10,
    )

    # Add 95% confidence interval
    chi2_lower = stats.chi2.ppf(0.025, df=n_dims)
    chi2_upper = stats.chi2.ppf(0.975, df=n_dims)

    ax.axvline(
        chi2_lower,
        color=ORANGE,
        linestyle="--",
        linewidth=4,
        alpha=0.7,
        label=f"95% CI: [{chi2_lower:.1f}, {chi2_upper:.1f}]",
    )
    ax.axvline(chi2_upper, color=ORANGE, linestyle="--", linewidth=4, alpha=0.7)

    # Shade the acceptance region
    ax.axvspan(
        chi2_lower, chi2_upper, alpha=0.1, color=GREEN, label="Acceptance region"
    )

    # Add mean NEES line
    mean_nees = np.mean(nees_values)
    ax.axvline(
        mean_nees,
        color=GREEN,
        linestyle="-",
        linewidth=3,
        label=f"Mean NEES: {mean_nees:.2f}",
    )

    # Color-code mean NEES annotation
    if chi2_lower <= mean_nees <= chi2_upper:
        mean_color = GREEN
        status = "✓ Well-calibrated"
    else:
        mean_color = RED
        status = "✗ Miscalibrated"

    ax.text(
        mean_nees,
        ax.get_ylim()[1] * 0.9,
        f"{mean_nees:.2f}\n{status}",
        fontsize=18,
        weight="bold",
        color=mean_color,
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            alpha=0.9,
            edgecolor=mean_color,
            linewidth=4,
        ),
    )

    # Expected mean for χ² is df
    expected_mean = n_dims
    ax.axvline(
        expected_mean,
        color=RED,
        linestyle=":",
        linewidth=4,
        alpha=0.7,
        label=f"Expected mean: {expected_mean}",
    )

    # Labels and title
    ax.set_xlabel("NEES Value", fontsize=18, weight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontsize=18, weight="bold", labelpad=10)
    ax.set_title(
        "Normalized Estimation Error Squared (NEES) Test",
        fontsize=20,
        weight="bold",
        pad=10,
    )
    # No legend on main plot - will be in dedicated panel
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)

    # Dedicated legend/stats panel
    ax_legend.axis("off")

    # Get handles and labels from main plot
    handles, labels = ax.get_legend_handles_labels()

    # Create legend in dedicated space
    ax_legend.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=12,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        title="Legend",
        title_fontsize=14,
    )

    # Interpretation guide - moved to smaller stats box to avoid overlap
    interpretation = (
        f"Mean ≈ {n_dims}: Perfect calibration | "
        f"Mean < {n_dims}: Overconfident | "
        f"Mean > {n_dims}: Underconfident"
    )

    fig.text(
        0.5,
        0.02,
        interpretation,
        fontsize=13,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=GRAY, alpha=0.1),
    )

    # Statistics summary - in legend panel below legend
    percent_in_ci = (
        100
        * np.sum((nees_values >= chi2_lower) & (nees_values <= chi2_upper))
        / len(nees_values)
    )

    stats_text = (
        f"Statistics:\n\n"
        f"Mean: {mean_nees:.2f}\n"
        f"Median: {np.median(nees_values):.2f}\n"
        f"Std: {np.std(nees_values):.2f}\n\n"
        f"In 95% CI:\n{percent_in_ci:.0f}%\n\n"
        f"Status:\n{status}"
    )

    # Place stats box in legend panel
    ax_legend.text(
        0.5,
        0.35,
        stats_text,
        transform=ax_legend.transAxes,
        fontsize=12,
        weight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            alpha=0.9,
            edgecolor=mean_color,
            linewidth=2,
        ),
    )
    output_path = OUTPUT_DIR / "slide21_nees_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Print statistics
    print(f"  Mean NEES: {mean_nees:.2f} (expected: {n_dims})")
    print(f"  Median NEES: {np.median(nees_values):.2f}")
    print(f"  Std NEES: {np.std(nees_values):.2f}")
    print(f"  95% CI: [{chi2_lower:.2f}, {chi2_upper:.2f}]")
    print(f"  Percent in CI: {percent_in_ci:.1f}%")
    print(f"  Status: {status}")


if __name__ == "__main__":
    print("Generating Slide 21: NEES Histogram...")
    print()
    generate_slide21()
    print()
    print("✅ Slide 21 visual generated!")
    print(f"   Output: {OUTPUT_DIR / 'slide21_nees_histogram.png'}")
