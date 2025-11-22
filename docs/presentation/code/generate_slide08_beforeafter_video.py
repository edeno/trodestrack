"""
Generate Slide 8: Before/After Split-Screen Video

Creates a 10-second split-screen video comparing:
- Left panel: Vision-only extrapolation (jumps, huge uncertainty during dropout)
- Right panel: Sensor fusion (smooth, bounded uncertainty)

Shows 5-second dropout in the middle of the session.
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
OUTPUT_DIR = Path(__file__).parent.parent / "videos"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette
BLUE = "#2E86AB"
ORANGE = "#F77F00"
GREEN = "#06A77D"
RED = "#D62828"
GRAY = "#6C757D"
PURPLE = "#9B59B6"  # For IMU direction (different from camera's orange)


def plot_covariance_ellipse(ax, mean, cov, color, alpha=0.3, n_std=2):
    """Plot covariance ellipse for 2D position"""
    # Extract 2×2 position covariance
    cov_2d = cov[:2, :2]

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Ellipse dimensions (n_std standard deviations)
    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(
        xy=mean[:2],
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=4,
    )
    ax.add_patch(ellipse)


def plot_heading_arrow(ax, pos, heading, color, length=0.15, label=None):
    """Plot heading direction arrow"""
    dx = length * np.cos(heading)
    dy = length * np.sin(heading)
    ax.arrow(
        pos[0],
        pos[1],
        dx,
        dy,
        head_width=0.05,
        head_length=0.04,
        fc=color,
        ec=color,
        linewidth=3,
        alpha=0.9,
        length_includes_head=True,
        label=label,
    )


def naive_vision_only_tracking(t_cam, Z_cam_led1, Z_cam_led2, mask_cam):
    """
    Naive vision-only tracking: just use observations when available,
    extrapolate with last known velocity during dropout.
    Also computes velocity direction (not body heading, just motion direction).
    """
    n_frames = len(t_cam)
    pos = np.zeros((n_frames, 2))
    cov = np.zeros((n_frames, 2, 2))
    velocity_direction = np.zeros(n_frames)

    # Initialize with first valid observation
    first_valid = np.where(mask_cam)[0][0]
    pos[first_valid] = Z_cam_led1[first_valid]
    cov[first_valid] = np.eye(2) * 0.005**2  # Small initial uncertainty
    velocity_direction[first_valid] = 0.0

    last_valid_pos = pos[first_valid].copy()
    last_valid_vel = np.array([0.0, 0.0])
    last_valid_dir = 0.0

    for i in range(first_valid + 1, n_frames):
        dt = t_cam[i] - t_cam[i - 1]

        if mask_cam[i]:
            # Use observation
            pos[i] = Z_cam_led1[i]
            cov[i] = np.eye(2) * 0.005**2

            # Update velocity estimate and direction
            last_valid_vel = (pos[i] - last_valid_pos) / (t_cam[i] - t_cam[i - 1])
            if np.linalg.norm(last_valid_vel) > 0.01:  # Only update if moving
                velocity_direction[i] = np.arctan2(last_valid_vel[1], last_valid_vel[0])
                last_valid_dir = velocity_direction[i]
            else:
                velocity_direction[i] = last_valid_dir
            last_valid_pos = pos[i].copy()
        else:
            # Extrapolate with constant velocity
            pos[i] = pos[i - 1] + last_valid_vel * dt

            # Direction stays at last known value during dropout (no gyro!)
            velocity_direction[i] = last_valid_dir

            # Uncertainty grows quadratically during dropout
            cov[i] = cov[i - 1] + np.eye(2) * (0.5 * dt) ** 2  # Very pessimistic growth

    return pos, cov, velocity_direction


def generate_slide08():
    """
    Create split-screen video comparing vision-only vs sensor fusion.

    SPATIAL LAYOUT PLAN (2-panel horizontal, 12"×5.625"):
    ┌──────────────────────────┬────┬─────────────────────────┬────┐
    │ LEFT PANEL               │Leg │ RIGHT PANEL             │Leg │
    │ "Vision-Only Tracking"   │end │ "Sensor Fusion (EKF)"  │end │
    │ (red title, 32pt)        │    │ (green title, 32pt)    │    │
    │                          │Err │                         │Err │
    │ [Trajectory plot]        │or  │ [Trajectory plot]      │or  │
    │ - Ground truth (blue)    │    │ - Ground truth (blue)  │    │
    │ - Vision-only (red)      │2σ  │ - EKF (green)          │2σ  │
    │ - Current pos (circle)   │    │ - Current pos (circle) │    │
    │ - Uncertainty ellipse    │    │ - Uncertainty ellipse  │    │
    ├──────────────────────────┴────┴─────────────────────────┴────┤
    │ Time: X.XXs | Status (centered text box at bottom)          │
    └──────────────────────────────────────────────────────────────┘

    OVERLAP PREVENTION:
    - Use constrained_layout=True (MANDATORY)
    - Figure size 12"×5.625" (wider to accommodate legends + stats on right)
    - NO suptitle - status moved to bottom center to avoid title collision
    - Panel titles at 32pt with pad=15 (plenty of clearance at top)
    - Legends OUTSIDE RIGHT: bbox_to_anchor=(1.02, 0.95) at top of right margin
    - Error stats OUTSIDE RIGHT: x=1.02, y=0.6 (below legends, stacked vertically)
    - Status text at very bottom: fig.text(0.5, 0.005)
    - labelpad=10 on all axis labels
    """

    from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
    from trodestrack.models.state_layout import get_layout
    from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

    # Generate simulation with 5-second dropout in middle
    np.random.seed(42)

    # Create 10-second simulation
    config = RatIMUSimConfig(
        duration_s=10.0,
        fs_imu=104.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,
        use_second_led=True,
        imu_tilt_roll_deg=0.0,
        imu_tilt_pitch_deg=0.0,
        gyro_bias_rw_density=np.deg2rad(0.0005),
        accel_bias_rw_density=0.001,
    )

    sim = simulate_rat_imu(config)

    # Manually create 1-second dropout in middle
    t_cam = sim["t_cam_exp"]
    mask_cam = sim["mask_cam"].copy()

    # Find middle 1 seconds
    dropout_start = 2.5  # Start dropout at 2.5s
    dropout_end = 3.5  # End dropout at 3.5s
    dropout_mask = (t_cam >= dropout_start) & (t_cam <= dropout_end)
    mask_cam[dropout_mask] = False

    # Run EKF with sensor fusion
    ekf_config = EKFConfig()

    result = extended_kalman_filter(
        ekf_config,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        mask_cam,
    )

    # Run naive vision-only tracking (with velocity direction from camera)
    vision_pos, vision_cov, vision_dir = naive_vision_only_tracking(
        t_cam, sim["Z_cam_led1"], sim["Z_cam_led2"], mask_cam
    )

    # Get ground truth
    layout = get_layout("2d_full")
    t_imu = sim["t_imu"]
    X_truth = sim["X_truth"]
    cam_indices = np.searchsorted(t_imu, t_cam)
    cam_indices = np.clip(cam_indices, 0, len(t_imu) - 1)
    pos_truth = X_truth[cam_indices, :2]

    # EKF results (sensor fusion: camera + IMU)
    ekf_pos = result.filtered_means[:, layout.pos_idx]

    # Extract 2×2 position covariances for each time step
    ekf_cov = np.array(
        [
            result.filtered_covariances[i][np.ix_(layout.pos_idx, layout.pos_idx)]
            for i in range(len(result.filtered_means))
        ]
    )

    # IMU-only prediction from EKF propagator (state before camera update)
    imu_pred = np.array(result.predicted_means)
    imu_pred_pos = imu_pred[:, layout.pos_idx]
    imu_pred_vel = imu_pred[:, layout.vel_idx]
    imu_vel_dir = np.arctan2(imu_pred_vel[:, 1], imu_pred_vel[:, 0])

    # Create figure with 2 panels (side by side)
    # MANDATORY: Use constrained_layout=True to prevent overlaps
    # Wider figure (12" instead of 10") to accommodate error stats on right side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.625), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # Set up axes limits (equal for both panels)
    x_min = pos_truth[:, 0].min() - 0.2
    x_max = pos_truth[:, 0].max() + 0.2
    y_min = pos_truth[:, 1].min() - 0.2
    y_max = pos_truth[:, 1].max() + 0.2

    for ax, title, color in zip(
        axes,
        ["Vision-Only Tracking", "Sensor Fusion (EKF)"],
        [RED, GREEN],
        strict=False,
    ):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # MANDATORY: Use labelpad=10 for axis labels
        ax.set_xlabel("X (meters)", fontsize=24, weight="bold", labelpad=10)
        ax.set_ylabel("Y (meters)", fontsize=24, weight="bold", labelpad=10)
        ax.set_title(title, fontsize=32, weight="bold", color=color, pad=15)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Add status text at very bottom center (below legends)
    status_text = fig.text(
        0.5,
        0.005,
        "",
        ha="center",
        va="bottom",
        fontsize=18,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="gray"
        ),
    )

    # Animation function
    def init():
        """Initialize animation"""
        return []

    def animate(frame):
        """Update animation frame"""
        # Clear previous plots
        for ax in axes:
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # MANDATORY: Use labelpad=10 for axis labels
            ax.set_xlabel("X (meters)", fontsize=24, weight="bold", labelpad=10)
            ax.set_ylabel("Y (meters)", fontsize=24, weight="bold", labelpad=10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        # Current time
        t_now = t_cam[frame]
        in_dropout = not mask_cam[frame]

        # Update status text at bottom with time and dropout status
        dropout_text = "⚠ CAMERA DROPOUT" if in_dropout else "✓ Camera Active"
        dropout_color = RED if in_dropout else GREEN
        status_text.set_text(f"Time: {t_now:.2f}s  |  {dropout_text}")
        status_text.set_color(dropout_color)

        # LEFT PANEL: Vision-only
        ax_left = axes[0]
        ax_left.set_title(
            "Vision-Only\nTracking", fontsize=32, weight="bold", color=RED, pad=15
        )

        # Plot ground truth (past trajectory)
        ax_left.plot(
            pos_truth[: frame + 1, 0],
            pos_truth[: frame + 1, 1],
            linewidth=3,
            color=BLUE,
            alpha=0.5,
            linestyle="--",
            label="Ground truth",
        )

        # Plot vision-only estimate (past trajectory)
        ax_left.plot(
            vision_pos[: frame + 1, 0],
            vision_pos[: frame + 1, 1],
            linewidth=3,
            color=RED,
            alpha=0.8,
            label="Vision-only",
        )

        # Plot current position
        ax_left.plot(
            vision_pos[frame, 0],
            vision_pos[frame, 1],
            "o",
            color=RED,
            markersize=15,
            zorder=10,
        )

        # Plot uncertainty ellipse
        plot_covariance_ellipse(
            ax_left, vision_pos[frame], vision_cov[frame], RED, alpha=0.3, n_std=2
        )

        # Plot velocity direction arrow (camera-based)
        plot_heading_arrow(
            ax_left,
            vision_pos[frame],
            vision_dir[frame],
            ORANGE,
            length=0.15,
            label="Camera direction",
        )

        # Calculate current error - place on RIGHT OUTSIDE below legend
        vision_error = np.linalg.norm(vision_pos[frame] - pos_truth[frame])
        ax_left.text(
            1.01,
            0.50,
            f"Error:\n{vision_error * 100:.1f} cm\n\n2σ:\n{2 * np.sqrt(vision_cov[frame, 0, 0]) * 100:.1f} cm",
            transform=ax_left.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.95,
                edgecolor=RED,
                linewidth=1.5,
            ),
            weight="bold",
            color=RED,
        )

        # MANDATORY: Place legend OUTSIDE on the right to avoid covering plot
        ax_left.legend(
            fontsize=14,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor=RED,
            facecolor="white",
            borderpad=0.3,
            labelspacing=0.2,
            handlelength=1.5,
        )

        # RIGHT PANEL: Sensor fusion
        ax_right = axes[1]
        ax_right.set_title(
            "Sensor Fusion\nTracking", fontsize=32, weight="bold", color=GREEN, pad=15
        )

        # Plot ground truth (past trajectory)
        ax_right.plot(
            pos_truth[: frame + 1, 0],
            pos_truth[: frame + 1, 1],
            linewidth=3,
            color=BLUE,
            alpha=0.5,
            linestyle="--",
            label="Ground truth",
        )

        # Plot EKF estimate (past trajectory)
        ax_right.plot(
            ekf_pos[: frame + 1, 0],
            ekf_pos[: frame + 1, 1],
            linewidth=3,
            color=GREEN,
            alpha=0.8,
            label="EKF",
        )

        # Plot camera prediction (vision/extrapolation)
        ax_right.plot(
            vision_pos[: frame + 1, 0],
            vision_pos[: frame + 1, 1],
            linewidth=2,
            color=ORANGE,
            alpha=0.8,
            linestyle="--",
            label="Camera pred.",
        )

        # Plot IMU-only dead-reckoned prediction (state prior to camera update)
        ax_right.plot(
            imu_pred_pos[: frame + 1, 0],
            imu_pred_pos[: frame + 1, 1],
            linewidth=2,
            color=PURPLE,
            alpha=0.7,
            linestyle=":",
            label="IMU-only",
        )

        # Plot current fused position
        ax_right.plot(
            ekf_pos[frame, 0],
            ekf_pos[frame, 1],
            "o",
            color=GREEN,
            markersize=15,
            zorder=10,
        )

        # Plot uncertainty ellipse
        plot_covariance_ellipse(
            ax_right, ekf_pos[frame], ekf_cov[frame], GREEN, alpha=0.3, n_std=2
        )

        # Plot direction arrows for camera and IMU predictions
        plot_heading_arrow(
            ax_right,
            vision_pos[frame],
            vision_dir[frame],
            ORANGE,
            length=0.15,
            label="Camera pred. dir.",
        )
        plot_heading_arrow(
            ax_right,
            imu_pred_pos[frame],
            imu_vel_dir[frame],
            PURPLE,
            length=0.12,
            label="IMU vel. dir.",
        )

        # Calculate current error - place on RIGHT OUTSIDE below legend
        ekf_error = np.linalg.norm(ekf_pos[frame] - pos_truth[frame])
        ax_right.text(
            1.01,
            0.50,
            f"Error:\n{ekf_error * 100:.1f} cm\n\n2σ:\n{2 * np.sqrt(ekf_cov[frame, 0, 0]) * 100:.1f} cm",
            transform=ax_right.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.95,
                edgecolor=GREEN,
                linewidth=1.5,
            ),
            weight="bold",
            color=GREEN,
        )

        # MANDATORY: Place legend OUTSIDE on the right to avoid covering plot
        handles, labels = ax_right.get_legend_handles_labels()
        uniq = dict(zip(labels, handles, strict=True))
        ax_right.legend(
            uniq.values(),
            uniq.keys(),
            fontsize=14,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor=GREEN,
            facecolor="white",
            borderpad=0.3,
            labelspacing=0.2,
            handlelength=1.5,
        )

        return []

    # Create animation
    # Use every 3rd frame (10 Hz) to reduce file size
    frames = list(range(0, len(t_cam), 3))
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=frames,
        interval=100,  # 100ms per frame = 10 FPS
        blit=False,
        repeat=True,
    )

    # MANDATORY: Check for warnings before saving
    print("Validating figure layout (checking for warnings)...")
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Render a test frame to check for layout warnings
        animate(0)
        fig.canvas.draw()

        if w:
            print(f"⚠ Found {len(w)} warning(s) during validation:")
            for warning in w:
                print(f"  - {warning.category.__name__}: {warning.message}")
        else:
            print("✓ No layout warnings detected")

    # Save video
    output_path = OUTPUT_DIR / "slide08_beforeafter.mp4"
    print(f"Saving video to {output_path} (this may take 2-3 minutes)...")

    # Use FFmpeg writer
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=10, bitrate=2000, codec="libx264")

    anim.save(str(output_path), writer=writer, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating Slide 8: Before/After Split-Screen Video...")
    print("This will take 2-3 minutes...")
    generate_slide08()
    print("✅ Slide 8 complete!")
