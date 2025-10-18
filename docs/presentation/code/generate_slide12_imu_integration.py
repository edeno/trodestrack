"""
Generate Slide 12: IMU Integration Visualization

Shows how IMU measurements are integrated between camera frames.
Demonstrates the predict step of the Kalman filter.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def generate_slide12():
    """
    Create visualization showing IMU integration between camera frames.

    Shows:
    1. Timeline with camera frames and IMU samples
    2. Gyro/accel measurements at each IMU time
    3. Integration process (θ += ω*dt, v += a*dt, x += v*dt)
    4. State evolution from frame k to frame k+1
    """

    from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

    # Generate short simulation with circular motion (easy to visualize)
    np.random.seed(42)
    config = RatIMUSimConfig(
        duration_s=0.5,  # Just 0.5 seconds
        fs_imu=104.0,  # SpikeGadgets rate
        fs_cam=30.0,  # Camera rate (~33ms between frames)
        cam_dropout_prob=0.0,
        use_second_led=True,
    )
    sim = simulate_rat_imu(config)

    # Extract data
    t_imu = sim["t_imu"]
    U_imu = sim["U_imu"]  # [accel_x, accel_y, gyro_z]
    t_cam = sim["t_cam_exp"]
    X_truth = sim["X_truth"]

    # Find two consecutive camera frames
    cam_idx_k = 3  # Pick 4th camera frame
    cam_idx_k1 = 4  # 5th camera frame

    t_k = t_cam[cam_idx_k]
    t_k1 = t_cam[cam_idx_k1]

    # Find IMU samples between these two camera frames
    imu_mask = (t_imu >= t_k) & (t_imu < t_k1)
    t_imu_between = t_imu[imu_mask]
    U_imu_between = U_imu[imu_mask]

    # Get ground truth states at camera frames
    imu_idx_k = np.searchsorted(t_imu, t_k)
    imu_idx_k1 = np.searchsorted(t_imu, t_k1)
    X_k = X_truth[imu_idx_k]
    X_k1 = X_truth[imu_idx_k1]

    # Get trajectory segment
    traj_mask = (t_imu >= t_k) & (t_imu <= t_k1)
    traj_segment = X_truth[traj_mask, :2]

    # Create figure with three panels
    fig = plt.figure(figsize=(18, 10))

    # ========================================================================
    # Panel 1 (Top): Timeline showing camera frames and IMU samples
    # ========================================================================
    ax1 = plt.subplot(3, 1, 1)

    # Draw camera frame markers
    for i, tc in enumerate([t_k, t_k1]):
        ax1.axvline(tc, color=BLUE, linewidth=3, alpha=0.7)
        label = f"Camera Frame {cam_idx_k if i == 0 else cam_idx_k1}"
        ax1.text(
            tc,
            1.5,
            label,
            ha="center",
            fontsize=14,
            weight="bold",
            color=BLUE,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

    # Draw IMU sample markers
    for i, ti in enumerate(t_imu_between):
        ax1.axvline(ti, color=ORANGE, linewidth=1.5, alpha=0.5, linestyle="--")
        if i == 0:  # Label first one
            ax1.text(
                ti,
                -0.5,
                "IMU samples\n@104 Hz",
                ha="center",
                fontsize=11,
                style="italic",
                color=ORANGE,
            )

    # Highlight integration interval
    ax1.axhspan(
        -1,
        2,
        xmin=(t_k - t_k) / (t_k1 - t_k),
        xmax=(t_k1 - t_k) / (t_k1 - t_k),
        alpha=0.15,
        color=GREEN,
    )

    # Labels
    ax1.text(
        (t_k + t_k1) / 2,
        0.5,
        f"Δt = {(t_k1 - t_k)*1000:.1f} ms\n({len(t_imu_between)} IMU samples)",
        ha="center",
        fontsize=16,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=GREEN, alpha=0.3),
    )

    ax1.set_xlim(t_k - 0.005, t_k1 + 0.005)
    ax1.set_ylim(-1, 2)
    ax1.set_yticks([])
    ax1.set_xlabel("Time (seconds)", fontsize=14, weight="bold")
    ax1.set_title("Timeline: IMU Pre-Integration Between Camera Frames", fontsize=18, weight="bold")
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 2 (Middle): IMU measurements
    # ========================================================================
    ax2 = plt.subplot(3, 1, 2)

    # Plot gyro Z
    ax2_gyro = ax2
    ax2_gyro.plot(
        t_imu_between,
        np.rad2deg(U_imu_between[:, 2]),
        "o-",
        color=RED,
        linewidth=2,
        markersize=6,
        label="Gyro Z (°/s)",
    )
    ax2_gyro.axhline(0, color=GRAY, linewidth=1, linestyle="--", alpha=0.5)
    ax2_gyro.set_ylabel("Gyro Z (°/s)", fontsize=12, weight="bold", color=RED)
    ax2_gyro.tick_params(axis="y", labelcolor=RED)
    ax2_gyro.set_xlim(t_k - 0.005, t_k1 + 0.005)
    ax2_gyro.grid(True, alpha=0.3)
    ax2_gyro.legend(loc="upper left", fontsize=11)

    # Plot accel X, Y on second y-axis
    ax2_accel = ax2.twinx()
    ax2_accel.plot(
        t_imu_between,
        U_imu_between[:, 0],
        "s-",
        color=GREEN,
        linewidth=2,
        markersize=5,
        alpha=0.7,
        label="Accel X (m/s²)",
    )
    ax2_accel.plot(
        t_imu_between,
        U_imu_between[:, 1],
        "^-",
        color=BLUE,
        linewidth=2,
        markersize=5,
        alpha=0.7,
        label="Accel Y (m/s²)",
    )
    ax2_accel.axhline(0, color=GRAY, linewidth=1, linestyle="--", alpha=0.5)
    ax2_accel.set_ylabel("Accel X, Y (m/s²)", fontsize=12, weight="bold", color=GREEN)
    ax2_accel.tick_params(axis="y", labelcolor=GREEN)
    ax2_accel.legend(loc="upper right", fontsize=11)

    ax2.set_xlabel("Time (seconds)", fontsize=14, weight="bold")
    ax2.set_title("IMU Measurements During Integration Interval", fontsize=18, weight="bold")

    # ========================================================================
    # Panel 3 (Bottom): Spatial trajectory with integration equations
    # ========================================================================
    ax3 = plt.subplot(3, 1, 3)

    # Plot trajectory segment
    ax3.plot(
        traj_segment[:, 0],
        traj_segment[:, 1],
        "o-",
        color=ORANGE,
        linewidth=3,
        markersize=8,
        alpha=0.8,
        label="Trajectory",
    )

    # Mark start and end positions
    ax3.plot(X_k[0], X_k[1], "o", color=BLUE, markersize=20, label=f"Frame {cam_idx_k}", zorder=10)
    ax3.plot(
        X_k1[0], X_k1[1], "s", color=GREEN, markersize=20, label=f"Frame {cam_idx_k1}", zorder=10
    )

    # Draw velocity vector at start
    vel_scale = 0.1
    ax3.arrow(
        X_k[0],
        X_k[1],
        X_k[2] * vel_scale,
        X_k[3] * vel_scale,
        head_width=0.02,
        head_length=0.01,
        fc=RED,
        ec=RED,
        linewidth=2,
        zorder=5,
        length_includes_head=True,
    )
    ax3.text(
        X_k[0] + X_k[2] * vel_scale * 0.5,
        X_k[1] + X_k[3] * vel_scale * 0.5 + 0.03,
        "v",
        fontsize=16,
        weight="bold",
        color=RED,
        ha="center",
    )

    # Draw heading vector at start
    heading_scale = 0.08
    theta_k = X_k[4]  # X_truth is [x, y, vx, vy, theta]
    ax3.arrow(
        X_k[0],
        X_k[1],
        heading_scale * np.cos(theta_k),
        heading_scale * np.sin(theta_k),
        head_width=0.015,
        head_length=0.01,
        fc=GRAY,
        ec=GRAY,
        linewidth=2,
        zorder=5,
        length_includes_head=True,
        alpha=0.7,
    )
    ax3.text(
        X_k[0] + heading_scale * np.cos(theta_k) * 1.5,
        X_k[1] + heading_scale * np.sin(theta_k) * 1.5,
        "θ",
        fontsize=16,
        weight="bold",
        color=GRAY,
        ha="center",
    )

    # Add integration equations
    equations = [
        r"$\theta_{k+1} = \theta_k + \int_{t_k}^{t_{k+1}} \omega_z \, dt$",
        r"$v_{k+1} = v_k + \int_{t_k}^{t_{k+1}} R(\theta) \cdot a \, dt$",
        r"$x_{k+1} = x_k + \int_{t_k}^{t_{k+1}} v \, dt$",
    ]

    eq_text = "\n".join(equations)
    ax3.text(
        0.02,
        0.98,
        eq_text,
        transform=ax3.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="lightyellow", alpha=0.9),
        family="monospace",
    )

    # Show actual state changes
    delta_x = X_k1[0] - X_k[0]
    delta_y = X_k1[1] - X_k[1]
    delta_theta = np.rad2deg(X_k1[4] - X_k[4])

    stats_text = (
        f"State Changes:\n"
        f"Δx = {delta_x*100:.2f} cm\n"
        f"Δy = {delta_y*100:.2f} cm\n"
        f"Δθ = {delta_theta:.2f}°"
    )

    ax3.text(
        0.98,
        0.98,
        stats_text,
        transform=ax3.transAxes,
        fontsize=13,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=GREEN, alpha=0.3),
        weight="bold",
    )

    ax3.set_xlabel("X (meters)", fontsize=14, weight="bold")
    ax3.set_ylabel("Y (meters)", fontsize=14, weight="bold")
    ax3.set_title("Spatial View: State Propagation via IMU Integration", fontsize=18, weight="bold")
    ax3.legend(fontsize=12, loc="lower left")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        "The Predict Step: IMU Pre-Integration Between Camera Frames\n"
        "How TrodesTrack propagates state estimates using high-rate IMU data",
        fontsize=22,
        weight="bold",
        y=0.995,
    )

    # Key insight
    fig.text(
        0.5,
        0.005,
        f"Key: {len(t_imu_between)} IMU samples @ 104 Hz provide smooth state "
        f"propagation between camera frames @ 30 Hz → No motion blur!",
        fontsize=13,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=ORANGE, alpha=0.2),
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_path = OUTPUT_DIR / "slide12_imu_integration.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating Slide 12: IMU Integration Visualization...")
    generate_slide12()
    print("✅ Slide 12 complete!")
