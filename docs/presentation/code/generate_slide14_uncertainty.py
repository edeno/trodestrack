"""
Generate visual for Slide 14: Uncertainty Evolution

Shows covariance ellipses growing during dropout and shrinking on recovery.
Demonstrates how the Kalman filter tracks uncertainty over time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

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


def plot_covariance_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Plot a covariance ellipse at n_std standard deviations.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    mean : array (2,)
        Mean position [x, y]
    cov : array (2, 2)
        Covariance matrix
    n_std : float
        Number of standard deviations for ellipse
    **kwargs : dict
        Additional arguments for Ellipse patch
    """
    # Eigendecomposition
    vals, vecs = np.linalg.eigh(cov)

    # Angle of ellipse
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # Width and height (scale by chi-squared 95% confidence for 2D)
    # For 95% confidence in 2D: chi2.ppf(0.95, 2) = 5.991
    width, height = 2 * n_std * np.sqrt(vals)

    # Create ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def generate_slide14():
    """Uncertainty evolution with covariance ellipses"""

    # Generate simulation with controlled dropout
    np.random.seed(42)
    config = RatIMUSimConfig(
        duration_s=20.0,
        fs_imu=104.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,  # Start with no dropout, we'll create artificial dropout
        use_second_led=True,
    )
    sim = simulate_rat_imu(config)

    # Create artificial dropout period from t=8s to t=13s (5 seconds)
    t_cam = sim["t_cam_exp"]
    dropout_start = 8.0
    dropout_end = 13.0
    dropout_mask = (t_cam >= dropout_start) & (t_cam <= dropout_end)

    # Modify the mask to create dropout
    mask_cam = sim["mask_cam"].copy()
    mask_cam[dropout_mask] = False

    # Run EKF
    ekf_config = EKFConfig()
    result = extended_kalman_filter(
        ekf_config,
        sim["t_imu"],
        sim["U_imu"],
        t_cam,
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        mask_cam,
    )

    # Get layout
    layout = get_layout("2d_full")

    # Extract position estimates and covariances
    pos_est = np.array(result.filtered_means[:, layout.pos_idx])
    pos_cov = np.array(result.filtered_covariances[:, layout.pos_idx, :][:, :, layout.pos_idx])

    # Downsample truth to camera rate
    t_imu = sim["t_imu"]
    cam_indices = np.searchsorted(t_imu, t_cam)
    cam_indices = np.clip(cam_indices, 0, len(t_imu) - 1)
    pos_truth = sim["X_truth"][cam_indices, :2]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot ground truth trajectory
    ax.plot(
        pos_truth[:, 0],
        pos_truth[:, 1],
        linewidth=2,
        color=BLUE,
        linestyle="--",
        alpha=0.5,
        label="Ground truth",
        zorder=1,
    )

    # Plot estimated trajectory
    ax.plot(
        pos_est[:, 0], pos_est[:, 1], linewidth=2.5, color=GREEN, label="EKF estimate", zorder=2
    )

    # Plot covariance ellipses at key moments
    # Sample ellipses every 0.5 seconds to avoid clutter
    ellipse_interval = int(0.5 / (t_cam[1] - t_cam[0]))  # Every 0.5 seconds
    ellipse_indices = np.arange(0, len(t_cam), ellipse_interval)

    # Determine color based on whether camera is active
    for idx in ellipse_indices:
        mean = pos_est[idx]
        cov = pos_cov[idx]
        t = t_cam[idx]

        # Color-code: green when camera active, red during dropout
        if dropout_start <= t <= dropout_end:
            # During dropout - red, more opaque
            color = RED
            alpha = 0.4
            edge_width = 1.5
        else:
            # Camera active - green, transparent
            color = GREEN
            alpha = 0.15
            edge_width = 1.0

        plot_covariance_ellipse(
            ax,
            mean,
            cov,
            n_std=2.0,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=edge_width,
            zorder=3,
        )

    # Highlight key moments with larger ellipses and labels
    key_moments = [
        (1.0, "Initialized\n(small uncertainty)", GREEN),
        (9.0, "Dropout starts\n(uncertainty growing)", ORANGE),
        (13.0, "Camera returns\n(uncertainty shrinks)", GREEN),
    ]

    for t_key, label, color in key_moments:
        # Find closest index
        idx = np.argmin(np.abs(t_cam - t_key))
        mean = pos_est[idx]
        cov = pos_cov[idx]

        # Plot a highlighted ellipse
        plot_covariance_ellipse(
            ax,
            mean,
            cov,
            n_std=2.0,
            facecolor="none",
            edgecolor=color,
            alpha=0.9,
            linewidth=3,
            linestyle="--",
            zorder=10,
        )

        # Add annotation with arrow
        ax.annotate(
            label,
            xy=mean,
            xytext=(mean[0] + 0.8, mean[1] + 0.8),
            fontsize=13,
            weight="bold",
            color=color,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2
            ),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=2.5),
            zorder=20,
        )

    # Mark dropout region on trajectory
    dropout_indices = np.where(dropout_mask)[0]
    if len(dropout_indices) > 0:
        dropout_pos = pos_est[dropout_indices]
        ax.plot(
            dropout_pos[:, 0],
            dropout_pos[:, 1],
            linewidth=4,
            color=RED,
            alpha=0.6,
            label="Dropout period (5s)",
            zorder=4,
        )

    # Mark start and end
    ax.scatter(
        pos_est[0, 0],
        pos_est[0, 1],
        s=200,
        marker="o",
        color=GREEN,
        edgecolor="black",
        linewidth=2,
        zorder=30,
        label="Start",
    )
    ax.scatter(
        pos_est[-1, 0],
        pos_est[-1, 1],
        s=200,
        marker="s",
        color=BLUE,
        edgecolor="black",
        linewidth=2,
        zorder=30,
        label="End",
    )

    # Labels and title
    ax.set_xlabel("X (meters)", fontsize=16, weight="bold")
    ax.set_ylabel("Y (meters)", fontsize=16, weight="bold")
    ax.set_title(
        "Uncertainty Evolution: Covariance Ellipses (±2σ)\n"
        "Growing during dropout, shrinking when camera returns",
        fontsize=20,
        weight="bold",
    )
    ax.legend(fontsize=13, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Add time annotations along trajectory
    time_markers = [0, 5, 10, 15, 20]
    for t_mark in time_markers:
        idx = np.argmin(np.abs(t_cam - t_mark))
        pos = pos_est[idx]
        ax.text(
            pos[0],
            pos[1],
            f"{t_mark}s",
            fontsize=10,
            color=GRAY,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", alpha=0.8, edgecolor=GRAY),
        )

    # Key insight
    fig.text(
        0.5,
        0.02,
        "Key Insight: Kalman filter tracks both state AND uncertainty. "
        "Uncertainty grows during dropout (no measurements), "
        "shrinks when camera returns (measurements correct prediction).",
        fontsize=13,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=BLUE, alpha=0.1),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    output_path = OUTPUT_DIR / "slide14_uncertainty.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Print statistics
    # Compute uncertainty magnitude (trace of position covariance)
    uncertainty_trace = np.array([np.trace(cov) for cov in pos_cov])
    idx_before = np.argmin(np.abs(t_cam - (dropout_start - 1)))
    idx_during = np.argmin(np.abs(t_cam - ((dropout_start + dropout_end) / 2)))
    idx_after = np.argmin(np.abs(t_cam - (dropout_end + 1)))

    print("  Uncertainty (trace of pos cov):")
    print(f"    Before dropout: {uncertainty_trace[idx_before]:.6f} m²")
    print(f"    During dropout: {uncertainty_trace[idx_during]:.6f} m²")
    print(f"    After recovery: {uncertainty_trace[idx_after]:.6f} m²")
    print(
        f"    Growth factor: {uncertainty_trace[idx_during] / uncertainty_trace[idx_before]:.1f}×"
    )


if __name__ == "__main__":
    print("Generating Slide 14: Uncertainty Evolution...")
    print()
    generate_slide14()
    print()
    print("✅ Slide 14 visual generated!")
    print(f"   Output: {OUTPUT_DIR / 'slide14_uncertainty.png'}")
