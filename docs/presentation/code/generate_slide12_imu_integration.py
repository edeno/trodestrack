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
    U_imu = sim["U_imu"]  # (T_imu, 3) = [omega_z, f_x, f_y]; see rat_imu.py
    t_cam = sim["t_cam_exp"]
    X_truth = sim["X_truth"]

    # Find two consecutive camera frames to show realistic IMU sampling
    # At 30 Hz camera (~33ms) and 104 Hz IMU (~9.6ms), we get 3-4 IMU samples between frames
    cam_idx_k = 3  # Pick 4th camera frame
    cam_idx_k1 = 4  # 5th camera frame (consecutive frames)

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

    # Create figure with subplot_mosaic for precise layout control
    # Layout: 3 rows (timeline, measurements, spatial+equations+legend)
    mosaic = [
        ["timeline", "timeline", "timeline", "timeline"],
        ["measurements", "measurements", "measurements", "measurements"],
        [
            "spatial",
            "spatial",
            "legend",
            "equations",
        ],
    ]

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 5.625),
        constrained_layout=True,
        gridspec_kw={
            "width_ratios": [1.5, 1.5, 0.6, 0.6],
            "height_ratios": [1, 1, 3],
            "hspace": 0.3,
            "wspace": 0.35,
        },
    )

    ax1 = axd["timeline"]
    ax2 = axd["measurements"]
    ax3 = axd["spatial"]
    ax_eq = axd["equations"]
    ax_legend = axd["legend"]

    # ========================================================================
    # Panel 1 (Top): Timeline showing camera frames and IMU samples
    # ========================================================================

    # Draw camera frame markers - simplified labels
    for i, tc in enumerate([t_k, t_k1]):
        ax1.axvline(tc, color=BLUE, linewidth=3, alpha=0.7)
        label = f"Frame {cam_idx_k if i == 0 else cam_idx_k1}"
        ax1.text(
            tc,
            1.3,
            label,
            ha="center",
            fontsize=12,
            weight="bold",
            color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    # Draw IMU sample markers - no text labels (legend will explain)
    for ti in t_imu_between:
        ax1.axvline(ti, color=ORANGE, linewidth=2, alpha=0.4, linestyle="--")

    # Highlight integration interval
    ax1.axhspan(
        -1,
        2,
        xmin=(t_k - t_k) / (t_k1 - t_k),
        xmax=(t_k1 - t_k) / (t_k1 - t_k),
        alpha=0.15,
        color=GREEN,
    )

    # Simplified interval label
    ax1.text(
        (t_k + t_k1) / 2,
        0.0,
        f"Δt={(t_k1 - t_k) * 1000:.1f}ms\n({len(t_imu_between)} samples)",
        ha="center",
        fontsize=11,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN, alpha=0.25),
    )

    ax1.set_xlim(t_k - 0.005, t_k1 + 0.005)
    ax1.set_ylim(-1, 2)
    ax1.set_yticks([])

    ax1.set_title(
        "Timeline: IMU Pre-Integration Between Camera Frames",
        fontsize=14,
        weight="bold",
        pad=8,
    )
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 2 (Middle): IMU measurements
    # ========================================================================
    # Plot gyro Z (column 0 in [omega_z, f_x, f_y])
    ax2_gyro = ax2
    ax2_gyro.plot(
        t_imu_between,
        np.rad2deg(U_imu_between[:, 0]),
        "o-",
        color=RED,
        linewidth=4,
        markersize=6,
        label="Gyro Z (°/s)",
    )
    ax2_gyro.axhline(0, color=GRAY, linewidth=3, linestyle="--", alpha=0.5)
    ax2_gyro.set_ylabel(
        "Gyro Z (°/s)", fontsize=13, weight="bold", color=RED, labelpad=8
    )
    ax2_gyro.tick_params(axis="y", labelcolor=RED, labelsize=10)
    ax2_gyro.set_xlim(t_k - 0.005, t_k1 + 0.005)
    ax2_gyro.grid(True, alpha=0.3)
    ax2_gyro.legend(
        loc="upper left", fontsize=10, frameon=True, fancybox=False, framealpha=0.95
    )

    # Plot accel X, Y on second y-axis (columns 1 and 2 in [omega_z, f_x, f_y])
    ax2_accel = ax2.twinx()
    ax2_accel.plot(
        t_imu_between,
        U_imu_between[:, 1],
        "s-",
        color=GREEN,
        linewidth=4,
        markersize=5,
        alpha=0.7,
        label="Accel X (m/s²)",
    )
    ax2_accel.plot(
        t_imu_between,
        U_imu_between[:, 2],
        "^-",
        color=BLUE,
        linewidth=4,
        markersize=5,
        alpha=0.7,
        label="Accel Y (m/s²)",
    )
    ax2_accel.axhline(0, color=GRAY, linewidth=3, linestyle="--", alpha=0.5)
    ax2_accel.set_ylabel(
        "Accel X, Y (m/s²)", fontsize=13, weight="bold", color=GREEN, labelpad=8
    )
    ax2_accel.tick_params(axis="y", labelcolor=GREEN, labelsize=10)
    ax2_accel.legend(
        loc="upper right", fontsize=10, frameon=True, fancybox=False, framealpha=0.95
    )

    ax2.set_xlabel("Time (s)", fontsize=14, weight="bold")

    # ========================================================================
    # Panel 3 (Bottom): Spatial trajectory
    # ========================================================================
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
    ax3.plot(
        X_k[0],
        X_k[1],
        "o",
        color=BLUE,
        markersize=20,
        label=f"Frame {cam_idx_k}",
        zorder=10,
    )
    ax3.plot(
        X_k1[0],
        X_k1[1],
        "s",
        color=GREEN,
        markersize=20,
        label=f"Frame {cam_idx_k1}",
        zorder=10,
    )

    # Draw velocity vector - RED ARROW
    vel_scale = 0.08
    _vel_arrow = ax3.arrow(
        X_k[0],
        X_k[1],
        X_k[2] * vel_scale,
        X_k[3] * vel_scale,
        head_width=0.008,
        head_length=0.005,
        fc=RED,
        ec=RED,
        linewidth=1,
        zorder=5,
        length_includes_head=True,
        label="Velocity (red)",
    )

    # Draw heading vector - GRAY ARROW (shorter than velocity)
    heading_scale = 0.05  # Make heading arrow shorter
    theta_k = X_k[4]  # X_truth is [x, y, vx, vy, theta]
    _heading_arrow = ax3.arrow(
        X_k[0],
        X_k[1],
        heading_scale * np.cos(theta_k),
        heading_scale * np.sin(theta_k),
        head_width=0.008,
        head_length=0.005,
        fc=GRAY,
        ec=GRAY,
        linewidth=1,
        zorder=5,
        length_includes_head=True,
        label="Heading (gray)",
    )

    # Equations panel - dedicated space for integration math
    ax_eq.axis("off")  # Turn off axis for equations panel

    equations = [
        "Integration\nEqs:",
        "",
        r"$\theta_{k+1} = \theta_k +$",
        r"$\int \omega_z dt$",
        "",
        r"$v_{k+1} = v_k +$",
        r"$\int R(\theta)a dt$",
        "",
        r"$x_{k+1} = x_k +$",
        r"$\int v dt$",
    ]

    eq_text = "\n".join(equations)
    ax_eq.text(
        0.5,
        0.5,
        eq_text,
        transform=ax_eq.transAxes,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="lightyellow",
            alpha=0.9,
            edgecolor=ORANGE,
            linewidth=1.5,
        ),
        family="monospace",
    )

    # Show actual state changes - moved to title area to avoid overlap
    delta_x = X_k1[0] - X_k[0]
    delta_y = X_k1[1] - X_k[1]
    delta_theta = np.rad2deg(X_k1[4] - X_k[4])

    # Compact stats in title
    stats_str = (
        f"Δx={delta_x * 100:.1f}cm, Δy={delta_y * 100:.1f}cm, Δθ={delta_theta:.1f}°"
    )

    ax3.set_xlabel("X (m)", fontsize=13, weight="bold", labelpad=8)
    ax3.set_ylabel("Y (m)", fontsize=13, weight="bold", labelpad=8)
    ax3.set_title(
        f"Spatial: State Propagation\n{stats_str}",
        fontsize=12,
        weight="bold",
        pad=8,
    )
    ax3.tick_params(labelsize=10)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # Legend Panel (Bottom-Right): Dedicated legend subplot
    # ========================================================================
    ax_legend.axis("off")

    # Collect handles and labels from the spatial plot
    handles, labels = ax3.get_legend_handles_labels()

    # Create legend in dedicated subplot
    legend = ax_legend.legend(
        handles,
        labels,
        loc="center",
        fontsize=11,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor=GRAY,
        title="Legend",
        title_fontsize=12,
    )
    legend.get_title().set_weight("bold")

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
    output_path = OUTPUT_DIR / "slide12_imu_integration.png"
    plt.savefig(output_path, dpi=150, facecolor="white")
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating Slide 12: IMU Integration Visualization...")
    generate_slide12()
    print("✅ Slide 12 complete!")
