"""
Generate visuals for IMU physics slides (5A, 5B, 5C)

Slide 5A: Accelerometer physics (specific force, not acceleration)
Slide 5B: Gyroscope physics (angular velocity and drift)
Slide 5C: Correction challenges (before/after bias correction)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

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

# ============================================================================
# SLIDE 5A: What Does an Accelerometer REALLY Measure?
# ============================================================================


def generate_slide05a():
    """Accelerometer physics: specific force vs acceleration"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.625), constrained_layout=True)

    scenarios = [
        {
            "title": "Stationary\n(at rest)",
            "acceleration": np.array([0, 0]),
            "gravity": np.array([0, -9.81]),
            "reading": np.array([0, +9.81]),  # Specific force = a - g
            "label": "Accelerometer reads:\n[0, 0, +9.81] m/s²",
            "description": "Spring compressed\nby gravity",
        },
        {
            "title": "Free Fall\n(dropping)",
            "acceleration": np.array([0, -9.81]),
            "gravity": np.array([0, -9.81]),
            "reading": np.array([0, 0]),  # a - g = 0
            "label": "Accelerometer reads:\n[0, 0, 0] m/s²",
            "description": "Zero g!\n(Einstein equivalence)",
        },
        {
            "title": "Accelerating Right\n(on maze)",
            "acceleration": np.array([+4.0, 0]),
            "gravity": np.array([0, -9.81]),
            "reading": np.array([+4.0, +9.81]),
            "label": "Accelerometer reads:\n[+4.0, 0, +9.81] m/s²",
            "description": "Motion + gravity\nmixed together",
        },
    ]

    for ax, scenario in zip(axes, scenarios, strict=False):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw IMU chip (simplified)
        chip = Rectangle((-0.3, -0.15), 0.6, 0.3, facecolor=GRAY, edgecolor="black", linewidth=4)
        ax.add_patch(chip)

        # Draw spring-mass system inside (conceptual)
        mass = Circle((0, 0), 0.08, facecolor=ORANGE, edgecolor="black", linewidth=3.5)
        ax.add_patch(mass)

        # Draw acceleration vector (true motion)
        if np.linalg.norm(scenario["acceleration"]) > 0:
            acc_scale = 0.1
            ax.arrow(
                0,
                0,
                scenario["acceleration"][0] * acc_scale,
                scenario["acceleration"][1] * acc_scale,
                head_width=0.1,
                head_length=0.08,
                fc=BLUE,
                ec=BLUE,
                linewidth=4,
                label="True acceleration",
            )
            ax.text(
                scenario["acceleration"][0] * acc_scale * 1.5,
                scenario["acceleration"][1] * acc_scale * 1.5,
                "a",
                fontsize=24,
                color=BLUE,
                weight="bold",
                ha="center",
                va="center",
            )

        # Draw gravity vector (always down)
        grav_scale = 0.08
        ax.arrow(
            0,
            0,
            scenario["gravity"][0] * grav_scale,
            scenario["gravity"][1] * grav_scale,
            head_width=0.1,
            head_length=0.08,
            fc=RED,
            ec=RED,
            linewidth=4,
            linestyle="--",
            label="Gravity",
        )
        ax.text(
            scenario["gravity"][0] * grav_scale - 0.25,
            scenario["gravity"][1] * grav_scale,
            "g",
            fontsize=24,
            color=RED,
            weight="bold",
            ha="center",
            va="center",
        )

        # Title
        ax.text(0, 1.2, scenario["title"], fontsize=28, weight="bold", ha="center", va="top")

        # Reading (what accelerometer outputs) - use transAxes for panel-relative position
        ax.text(
            0.5,
            -0.2,
            scenario["label"],
            fontsize=20,
            ha="center",
            va="top",
            transform=ax.transAxes,  # Panel-relative coordinates
            bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN, alpha=0.3),
        )

        # Description
        ax.text(
            0.5,
            -0.45,
            scenario["description"],
            fontsize=16,
            style="italic",
            ha="center",
            va="top",
            transform=ax.transAxes,  # Panel-relative coordinates
            color=GRAY,
        )

    # Key insight (moved higher to avoid cutting off)
    fig.text(
        0.5,
        0.00,
        "Key: When IMU tilted, gravity contaminates X/Y axes → Need correction!",
        fontsize=16,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=ORANGE, alpha=0.2),
    )
    output_path = OUTPUT_DIR / "slide05a_accelerometer_physics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# SLIDE 5B: What Does a Gyroscope REALLY Measure?
# ============================================================================


def generate_slide05b():
    """Gyroscope physics: SPLIT INTO CLEAN SINGLE-FOCUS FIGURE

    Focus: Just the drift plot - one clear message
    """
    # Use subplot_mosaic to reserve space for legend on the right
    mosaic = [["main", "legend"]]
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4, 1]},
    )
    ax = axd["main"]
    ax_legend = axd["legend"]
    ax_legend.axis("off")

    # Simulate gyro drift
    dt = 0.01
    duration = 10.0
    t = np.arange(0, duration, dt)

    # True heading (rat turns then stops)
    true_omega = np.zeros_like(t)
    true_omega[t < 3] = 30.0  # Turn for 3 seconds at 30 °/s
    true_heading = np.cumsum(true_omega) * dt  # Integration

    # Gyro with bias (simulated)
    gyro_bias = 5.0  # 5 °/s constant bias
    gyro_noise = np.random.randn(len(t)) * 0.5  # Small white noise
    measured_omega = true_omega + gyro_bias + gyro_noise
    measured_heading = np.cumsum(measured_omega) * dt

    # Plot with BOLD, VISIBLE lines
    ax.plot(t, true_heading, linewidth=5, color=BLUE, label="True heading", zorder=10)
    ax.plot(
        t,
        measured_heading,
        linewidth=5,
        color=RED,
        linestyle="--",
        label="Gyro (with bias)",
        zorder=10,
    )

    # Highlight drift (subtle)
    ax.fill_between(t, true_heading, measured_heading, alpha=0.2, color=RED, zorder=1)

    # Annotate final error - simplified and repositioned
    final_error = measured_heading[-1] - true_heading[-1]
    ax.annotate(
        f"Drift:\n{final_error:.0f}°",
        xy=(t[-1], measured_heading[-1]),
        xytext=(7, 80),
        fontsize=18,
        weight="bold",
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, linewidth=3),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=RED, linewidth=2),
        zorder=100,
    )

    # Clean labels - reduced font sizes
    ax.set_xlabel("Time (s)", fontsize=18, weight="bold", labelpad=10)
    ax.set_ylabel("Heading (°)", fontsize=18, weight="bold", labelpad=10)

    # Extract legend and place in dedicated panel
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(
        handles,
        labels,
        loc="center left",
        fontsize=16,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor=RED,
        title="Legend",
        title_fontsize=18,
    )

    ax.grid(True, alpha=0.2, linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 135)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(labelsize=14, width=2, length=6)

    # Simple, CLEAR title (one message) - reduced font size
    ax.set_title(
        "Gyroscope Bias Causes Unbounded Drift",
        fontsize=24,
        weight="bold",
    )

    # Simplified caption - reduced font size
    fig.text(
        0.5,
        -0.05,
        "Integration: θ = θ + ∫ω dt  •  Bias ~1-5°/s → ±50° error after 10s",
        fontsize=14,
        ha="center",
        style="italic",
        color=GRAY,
    )

    output_path = OUTPUT_DIR / "slide05b_gyroscope_physics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# SLIDE 5C: The Correction Challenge
# ============================================================================


def generate_slide05c():
    """Before/after bias correction comparison"""
    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
    from trodestrack.models.state_layout import get_layout
    from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

    # Generate realistic simulation
    np.random.seed(42)
    config = RatIMUSimConfig(
        duration_s=30.0,
        fs_imu=104.0,
        fs_cam=30.0,
        gyro_bias_rw_density=np.deg2rad(0.03),  # 10× larger bias drift for demo
        accel_bias_rw_density=0.05,  # 10× larger bias drift for demo
        cam_dropout_prob=0.0,  # No dropout for cleaner comparison
        use_second_led=True,
    )
    sim = simulate_rat_imu(config)

    # Run EKF with bias estimation
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

    # Simulate "raw integration" without bias correction
    # (Integrate IMU naively)
    layout = get_layout("2d_full")

    # For visualization: integrate gyro with bias, accel with bias
    t_imu = sim["t_imu"]
    U_imu = sim["U_imu"]

    # Naive integration (no bias removal)
    naive_heading = np.zeros(len(t_imu))
    naive_vel = np.zeros((len(t_imu), 2))
    naive_pos = np.zeros((len(t_imu), 2))

    heading = 0.0
    vel = np.array([0.0, 0.0])
    pos = sim["X_truth"][0, :2]  # Start at true position

    for i in range(1, len(t_imu)):
        dt = t_imu[i] - t_imu[i - 1]

        # Gyro integration (raw, includes bias)
        omega_z = U_imu[i, 2]  # Gyro Z
        heading += omega_z * dt
        naive_heading[i] = heading

        # Accel integration (raw, includes bias and gravity contamination)
        accel_x = U_imu[i, 0]  # Accel X
        accel_y = U_imu[i, 1]  # Accel Y

        # Rotate to world frame (using current heading estimate)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        accel_world = np.array(
            [
                cos_h * accel_x - sin_h * accel_y,
                sin_h * accel_x + cos_h * accel_y,
            ]
        )

        vel += accel_world * dt
        pos += vel * dt

        naive_vel[i] = vel
        naive_pos[i] = pos

    # Downsample to camera rate for fair comparison
    cam_indices = np.searchsorted(t_imu, sim["t_cam_exp"])
    cam_indices = np.clip(cam_indices, 0, len(t_imu) - 1)

    naive_pos_cam = naive_pos[cam_indices]

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.625), constrained_layout=True)

    # Left: Raw integration (with bias)
    ax1 = axes[0]
    ax1.plot(
        naive_pos_cam[:, 0],
        naive_pos_cam[:, 1],
        linewidth=4,
        color=RED,
        alpha=0.7,
        label="Naive integration",
    )

    # Downsample X_truth to camera rate
    X_truth_cam = sim["X_truth"][cam_indices]
    ax1.plot(
        X_truth_cam[:, 0],
        X_truth_cam[:, 1],
        linewidth=4,
        color=BLUE,
        linestyle="--",
        label="Ground truth",
    )

    # Highlight divergence
    ax1.arrow(
        naive_pos_cam[-1, 0],
        naive_pos_cam[-1, 1],
        0,
        0,
        head_width=0.3,
        head_length=0.2,
        fc=RED,
        ec=RED,
        linewidth=0,
    )

    # Calculate drift
    drift = np.linalg.norm(naive_pos_cam[-1] - X_truth_cam[-1, :2])
    ax1.text(
        naive_pos_cam[-1, 0],
        naive_pos_cam[-1, 1] + 0.5,
        f"Drift: {drift:.1f} m",
        fontsize=24,
        weight="bold",
        color=RED,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
    )

    ax1.set_xlabel("X (meters)", fontsize=20, weight="bold", labelpad=10)
    ax1.set_ylabel("Y (meters)", fontsize=20, weight="bold", labelpad=10)
    ax1.set_title(
        "Before: Raw IMU Integration\n(No Bias Correction)",
        fontsize=20,
        weight="bold",
        color=RED,
        pad=15,
    )
    ax1.legend(
        fontsize=16,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor=RED,
    )
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right: EKF with bias estimation
    ax2 = axes[1]
    ekf_pos = result.filtered_means[:, layout.pos_idx]
    ax2.plot(
        ekf_pos[:, 0],
        ekf_pos[:, 1],
        linewidth=4,
        color=GREEN,
        alpha=0.8,
        label="EKF (bias-corrected)",
    )
    ax2.plot(
        X_truth_cam[:, 0],
        X_truth_cam[:, 1],
        linewidth=4,
        color=BLUE,
        linestyle="--",
        label="Ground truth",
    )

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.sum((ekf_pos - X_truth_cam[:, :2]) ** 2, axis=1)))
    ax2.text(
        ekf_pos[-1, 0],
        ekf_pos[-1, 1] + 0.5,
        f"RMSE: {rmse*100:.1f} cm",
        fontsize=24,
        weight="bold",
        color=GREEN,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
    )

    ax2.set_xlabel("X (meters)", fontsize=20, weight="bold", labelpad=10)
    ax2.set_ylabel("Y (meters)", fontsize=20, weight="bold", labelpad=10)
    ax2.set_title(
        "After: Kalman Filter\n(Bias Estimated in State)",
        fontsize=20,
        weight="bold",
        color=GREEN,
        pad=15,
    )
    ax2.legend(
        fontsize=16,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor=GREEN,
    )
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Make axes equal for fair comparison
    all_x = np.concatenate([naive_pos_cam[:, 0], ekf_pos[:, 0], X_truth_cam[:, 0]])
    all_y = np.concatenate([naive_pos_cam[:, 1], ekf_pos[:, 1], X_truth_cam[:, 1]])
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1

    for ax in axes:
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    # Overall title
    fig.suptitle(
        "The Correction Challenge: 4 Corrections Needed\n"
        "Gravity removal • Gyro bias • Accel bias • Frame alignment",
        fontsize=32,
        weight="bold",
    )

    # Key insight
    fig.text(
        0.5,
        0.02,
        "Kalman filter estimates biases in state vector [x, y, vₓ, vᵧ, θ, b_gz, b_ax, b_ay] → "
        f"Reduces drift by {drift/rmse:.0f}×!",
        fontsize=13,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=GREEN, alpha=0.2),
    )
    output_path = OUTPUT_DIR / "slide05c_bias_correction.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Generating IMU physics visuals for slides 5A, 5B, 5C...")
    print()

    print("[1/3] Slide 5A: Accelerometer physics...")
    generate_slide05a()

    print("[2/3] Slide 5B: Gyroscope physics...")
    generate_slide05b()

    print("[3/3] Slide 5C: Bias correction comparison...")
    generate_slide05c()

    print()
    print("✅ All IMU physics visuals generated!")
    print(f"   Output directory: {OUTPUT_DIR}")
