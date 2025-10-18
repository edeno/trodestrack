"""
Generate Slide 2: Camera Failure Modes Grid

Shows 2×2 grid demonstrating common vision-only tracking failures:
1. LED occlusion (rat near wall)
2. LED reflection (shiny surface)
3. Motion blur (fast movement)
4. Dim lighting (low confidence)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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


def generate_slide02():
    """
    Create 2×2 grid showing camera tracking failure modes.

    Each panel shows a different failure scenario with simulated data.
    """

    from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

    # Generate 4 simulations with different failure modes
    np.random.seed(42)

    # Simulation configs for different failure modes
    configs = {
        "occlusion": RatIMUSimConfig(
            duration_s=10.0,
            cam_dropout_prob=0.5,  # High dropout (simulates occlusion)
            use_second_led=True,
        ),
        "reflection": RatIMUSimConfig(
            duration_s=10.0,
            cam_dropout_prob=0.1,
            led_wall_reflection_prob=0.4,  # High reflection probability
            use_second_led=True,
        ),
        "blur": RatIMUSimConfig(
            duration_s=10.0,
            cam_dropout_prob=0.3,  # More dropout (simulates blur)
            use_second_led=True,
        ),
        "dim": RatIMUSimConfig(
            duration_s=10.0,
            cam_dropout_prob=0.2,
            use_confidence=True,  # Enable confidence tracking
            confidence_base=0.6,  # Lower base confidence (dim lighting)
            use_second_led=True,
        ),
    }

    # Generate simulations
    sims = {}
    for name, config in configs.items():
        sims[name] = simulate_rat_imu(config)

    # Create 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    failure_modes = [
        {
            "name": "occlusion",
            "title": "LED Occlusion",
            "description": "Rat near wall or object\nblocks camera view",
            "color": RED,
            "show_gaps": True,
        },
        {
            "name": "reflection",
            "title": "LED Reflection",
            "description": "Shiny surfaces create\nfalse LED detections",
            "color": ORANGE,
            "show_gaps": False,
        },
        {
            "name": "blur",
            "title": "Motion Blur",
            "description": "Fast movements exceed\ncamera shutter speed",
            "color": "#9B59B6",  # Purple
            "show_gaps": True,
        },
        {
            "name": "dim",
            "title": "Dim Lighting",
            "description": "Low illumination reduces\ndetection confidence",
            "color": GRAY,
            "show_gaps": True,
        },
    ]

    for ax, mode in zip(axes, failure_modes, strict=False):
        sim = sims[mode["name"]]

        # Extract data
        t_cam = sim["t_cam_exp"]
        Z_cam = sim["Z_cam_led1"]  # LED1 position observations
        mask_cam = sim["mask_cam"]
        confidence = sim["confidence_led1"]

        # Get ground truth at camera times
        t_imu = sim["t_imu"]
        X_truth = sim["X_truth"]
        cam_indices = np.searchsorted(t_imu, t_cam)
        cam_indices = np.clip(cam_indices, 0, len(t_imu) - 1)
        pos_truth = X_truth[cam_indices, :2]

        # Plot ground truth
        ax.plot(
            pos_truth[:, 0],
            pos_truth[:, 1],
            linewidth=3,
            color=BLUE,
            alpha=0.7,
            linestyle="--",
            label="Ground truth",
            zorder=1,
        )

        # Plot observations
        if mode["show_gaps"]:
            # Show gaps explicitly
            valid_obs = Z_cam[mask_cam]

            ax.scatter(
                valid_obs[:, 0],
                valid_obs[:, 1],
                s=80,
                c=GREEN,
                alpha=0.6,
                marker="o",
                label="Valid detections",
                zorder=3,
                edgecolors="black",
                linewidths=1,
            )

            # Mark gaps with X
            ax.scatter(
                pos_truth[~mask_cam, 0],
                pos_truth[~mask_cam, 1],
                s=150,
                c=RED,
                alpha=0.5,
                marker="x",
                linewidths=3,
                label="Dropout",
                zorder=4,
            )
        else:
            # For reflection, show outliers
            # Create some artificial outliers for visualization
            outlier_mask = (
                sim["led_reflection_applied"]
                if "led_reflection_applied" in sim
                else np.zeros(len(mask_cam), dtype=bool)
            )

            normal_obs = Z_cam[mask_cam & ~outlier_mask]
            outlier_obs = Z_cam[mask_cam & outlier_mask]

            ax.scatter(
                normal_obs[:, 0],
                normal_obs[:, 1],
                s=80,
                c=GREEN,
                alpha=0.6,
                marker="o",
                label="Valid detections",
                zorder=3,
                edgecolors="black",
                linewidths=1,
            )

            if len(outlier_obs) > 0:
                ax.scatter(
                    outlier_obs[:, 0],
                    outlier_obs[:, 1],
                    s=120,
                    c=ORANGE,
                    alpha=0.7,
                    marker="*",
                    label="Reflection outliers",
                    zorder=4,
                    edgecolors="black",
                    linewidths=1.5,
                )

        # Add arena boundary (simulated)
        arena_bounds = Rectangle(
            (pos_truth[:, 0].min() - 0.1, pos_truth[:, 1].min() - 0.1),
            pos_truth[:, 0].max() - pos_truth[:, 0].min() + 0.2,
            pos_truth[:, 1].max() - pos_truth[:, 1].min() + 0.2,
            fill=False,
            edgecolor=GRAY,
            linewidth=2,
            linestyle=":",
            alpha=0.5,
        )
        ax.add_patch(arena_bounds)

        # Calculate dropout statistics
        dropout_pct = (1 - mask_cam.mean()) * 100
        avg_confidence = confidence[mask_cam].mean() if mask_cam.any() else 0.0

        # Add statistics box
        stats_text = f"Dropout: {dropout_pct:.1f}%\n" f"Avg confidence: {avg_confidence:.2f}"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.8),
            weight="bold",
        )

        # Title and description
        ax.set_title(
            f"{mode['title']}\n{mode['description']}",
            fontsize=16,
            weight="bold",
            color=mode["color"],
            pad=15,
        )

        ax.set_xlabel("X (meters)", fontsize=13, weight="bold")
        ax.set_ylabel("Y (meters)", fontsize=13, weight="bold")
        ax.legend(fontsize=11, loc="lower right")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        "The Behavioral Tracking Challenge\n" "Why Vision-Only Tracking Fails in Real Experiments",
        fontsize=24,
        weight="bold",
        y=0.995,
    )

    # Key insight
    fig.text(
        0.5,
        0.005,
        "Common problems: Occlusions (40-60%), Reflections (10-30%), Motion blur (20-40%), "
        "Dim lighting (30-50%) → Need sensor fusion!",
        fontsize=14,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=1.0", facecolor=RED, alpha=0.15),
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_path = OUTPUT_DIR / "slide02_failure_modes.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating Slide 2: Camera Failure Modes Grid...")
    generate_slide02()
    print("✅ Slide 2 complete!")
