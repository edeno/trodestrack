"""
Generate visuals for IMU physics slides (5A, 5B, 5C)

Slide 5A: Accelerometer physics (specific force, not acceleration)
Slide 5B: Gyroscope physics (angular velocity and drift)
Slide 5C: Correction challenges (before/after bias correction)
"""

from pathlib import Path

import matplotlib.patches as mpatches
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
            "acceleration": np.array([+2.0, 0]),
            "gravity": np.array([0, -9.81]),
            "reading": np.array([+2.0, +9.81]),
            "label": "Accelerometer reads:\n[+2.0, 0, +9.81] m/s²",
            "description": "Motion + gravity\nmixed together",
        },
    ]

    for ax, scenario in zip(axes, scenarios, strict=False):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw IMU chip (simplified)
        chip = Rectangle((-0.3, -0.15), 0.6, 0.3, facecolor=GRAY, edgecolor="black", linewidth=2)
        ax.add_patch(chip)

        # Draw spring-mass system inside (conceptual)
        mass = Circle((0, 0), 0.08, facecolor=ORANGE, edgecolor="black", linewidth=1.5)
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
                linewidth=2,
                label="True acceleration",
            )
            ax.text(
                scenario["acceleration"][0] * acc_scale * 1.5,
                scenario["acceleration"][1] * acc_scale * 1.5,
                "a",
                fontsize=16,
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
            linewidth=2,
            linestyle="--",
            label="Gravity",
        )
        ax.text(
            scenario["gravity"][0] * grav_scale - 0.25,
            scenario["gravity"][1] * grav_scale,
            "g",
            fontsize=16,
            color=RED,
            weight="bold",
            ha="center",
            va="center",
        )

        # Title
        ax.text(0, 1.2, scenario["title"], fontsize=18, weight="bold", ha="center", va="top")

        # Reading (what accelerometer outputs)
        ax.text(
            0,
            -0.7,
            scenario["label"],
            fontsize=14,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=GREEN, alpha=0.3),
        )

        # Description
        ax.text(
            0,
            -1.15,
            scenario["description"],
            fontsize=12,
            style="italic",
            ha="center",
            va="top",
            color=GRAY,
        )

    # Overall title
    fig.suptitle(
        "What Does an Accelerometer REALLY Measure?\n"
        "Specific Force: f = a - g (not acceleration!)",
        fontsize=22,
        weight="bold",
        y=0.98,
    )

    # Key insight
    fig.text(
        0.5,
        0.02,
        "Key Challenge: When IMU is tilted, gravity contaminates X/Y axes → Need correction!",
        fontsize=14,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=ORANGE, alpha=0.2),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path = OUTPUT_DIR / "slide05a_accelerometer_physics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# SLIDE 5B: What Does a Gyroscope REALLY Measure?
# ============================================================================


def generate_slide05b():
    """Gyroscope physics: angular velocity and drift"""
    fig = plt.figure(figsize=(16, 8))

    # Left panel: Gyroscope measurement concept
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # Draw rat (top-down view, simplified)
    rat_body = mpatches.Ellipse(
        (0, 0), 1.2, 0.6, angle=30, facecolor=GRAY, edgecolor="black", linewidth=2
    )
    ax1.add_patch(rat_body)

    # Draw headstage
    headstage = Rectangle((-0.15, 0.25), 0.3, 0.2, facecolor=BLUE, edgecolor="black", linewidth=2)
    ax1.add_patch(headstage)
    ax1.text(0, 0.35, "IMU", fontsize=10, weight="bold", ha="center", va="center", color="white")

    # Draw rotation arrow (counterclockwise)
    from matplotlib.patches import FancyArrowPatch

    rotation_arrow = FancyArrowPatch(
        (0.7, 0.7),
        (-0.5, 0.9),
        arrowstyle="->,head_width=0.4,head_length=0.4",
        color=ORANGE,
        linewidth=3,
        connectionstyle="arc3,rad=0.5",
    )
    ax1.add_patch(rotation_arrow)
    ax1.text(-0.1, 1.1, "ω = +30 °/s", fontsize=16, weight="bold", color=ORANGE, ha="center")

    # Draw Z-axis (pointing up through headstage)
    ax1.arrow(0, 0.45, 0, 0.5, head_width=0.1, head_length=0.08, fc=GREEN, ec=GREEN, linewidth=2.5)
    ax1.text(0.15, 0.95, "Z-axis\n(yaw)", fontsize=12, color=GREEN, weight="bold", ha="left")

    # Labels
    ax1.text(0, -1.3, "Top-Down View: Rat Turning Left", fontsize=16, weight="bold", ha="center")
    ax1.text(
        0,
        -1.55,
        "Gyroscope measures rotation rate around Z-axis",
        fontsize=12,
        style="italic",
        ha="center",
        color=GRAY,
    )

    # Right panel: Integration and drift
    ax2 = plt.subplot(1, 2, 2)

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

    # Plot
    ax2.plot(t, true_heading, linewidth=3, color=BLUE, label="True heading")
    ax2.plot(
        t,
        measured_heading,
        linewidth=2,
        color=RED,
        linestyle="--",
        label="Gyro integration (with bias)",
    )

    # Highlight drift
    ax2.fill_between(
        t, true_heading, measured_heading, alpha=0.3, color=RED, label="Accumulated drift"
    )

    # Annotate final error
    final_error = measured_heading[-1] - true_heading[-1]
    ax2.annotate(
        f"Drift: {final_error:.0f}°\n(after 10 sec)",
        xy=(t[-1], measured_heading[-1]),
        xytext=(7, measured_heading[-1] + 30),
        fontsize=14,
        weight="bold",
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, linewidth=2),
    )

    ax2.set_xlabel("Time (s)", fontsize=14, weight="bold")
    ax2.set_ylabel("Heading (degrees)", fontsize=14, weight="bold")
    ax2.set_title("Integration: θ = θ₀ + ∫ω dt", fontsize=16, weight="bold")
    ax2.legend(fontsize=12, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)

    # Overall title
    fig.suptitle(
        "What Does a Gyroscope REALLY Measure?\n"
        "Angular Velocity (°/s) → Bias causes unbounded drift!",
        fontsize=22,
        weight="bold",
    )

    # Key specs
    fig.text(
        0.5,
        0.02,
        "SpikeGadgets Specs: Noise 0.01 °/s/√Hz, Bias ~1-5 °/s → "
        "Without camera correction: ±50° error after 10 seconds!",
        fontsize=13,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=ORANGE, alpha=0.2),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Raw integration (with bias)
    ax1 = axes[0]
    ax1.plot(
        naive_pos_cam[:, 0],
        naive_pos_cam[:, 1],
        linewidth=2,
        color=RED,
        alpha=0.7,
        label="Naive integration",
    )

    # Downsample X_truth to camera rate
    X_truth_cam = sim["X_truth"][cam_indices]
    ax1.plot(
        X_truth_cam[:, 0],
        X_truth_cam[:, 1],
        linewidth=2,
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
        fontsize=14,
        weight="bold",
        color=RED,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    ax1.set_xlabel("X (meters)", fontsize=14, weight="bold")
    ax1.set_ylabel("Y (meters)", fontsize=14, weight="bold")
    ax1.set_title(
        "Before: Raw IMU Integration\n(No Bias Correction)", fontsize=16, weight="bold", color=RED
    )
    ax1.legend(fontsize=12, loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right: EKF with bias estimation
    ax2 = axes[1]
    ekf_pos = result.filtered_means[:, layout.pos_idx]
    ax2.plot(
        ekf_pos[:, 0],
        ekf_pos[:, 1],
        linewidth=2,
        color=GREEN,
        alpha=0.8,
        label="EKF (bias-corrected)",
    )
    ax2.plot(
        X_truth_cam[:, 0],
        X_truth_cam[:, 1],
        linewidth=2,
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
        fontsize=14,
        weight="bold",
        color=GREEN,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    ax2.set_xlabel("X (meters)", fontsize=14, weight="bold")
    ax2.set_ylabel("Y (meters)", fontsize=14, weight="bold")
    ax2.set_title(
        "After: Kalman Filter\n(Bias Estimated in State)", fontsize=16, weight="bold", color=GREEN
    )
    ax2.legend(fontsize=12, loc="upper right")
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
        fontsize=20,
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

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
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
