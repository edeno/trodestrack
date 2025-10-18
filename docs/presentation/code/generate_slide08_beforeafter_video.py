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


def naive_vision_only_tracking(t_cam, Z_cam, mask_cam):
    """
    Naive vision-only tracking: just use observations when available,
    extrapolate with last known velocity during dropout.
    """
    n_frames = len(t_cam)
    pos = np.zeros((n_frames, 2))
    cov = np.zeros((n_frames, 2, 2))

    # Initialize with first valid observation
    first_valid = np.where(mask_cam)[0][0]
    pos[first_valid] = Z_cam[first_valid]
    cov[first_valid] = np.eye(2) * 0.005**2  # Small initial uncertainty

    last_valid_pos = pos[first_valid].copy()
    last_valid_vel = np.array([0.0, 0.0])

    for i in range(first_valid + 1, n_frames):
        dt = t_cam[i] - t_cam[i - 1]

        if mask_cam[i]:
            # Use observation
            pos[i] = Z_cam[i]
            cov[i] = np.eye(2) * 0.005**2

            # Update velocity estimate
            last_valid_vel = (pos[i] - last_valid_pos) / (t_cam[i] - t_cam[i - 1])
            last_valid_pos = pos[i].copy()
        else:
            # Extrapolate with constant velocity
            pos[i] = pos[i - 1] + last_valid_vel * dt

            # Uncertainty grows quadratically during dropout
            cov[i] = cov[i - 1] + np.eye(2) * (0.5 * dt) ** 2  # Very pessimistic growth

    return pos, cov


def generate_slide08():
    """
    Create split-screen video comparing vision-only vs sensor fusion.
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

    # Manually create 5-second dropout in middle (frames 150-300, roughly 5s @ 30Hz)
    t_cam = sim["t_cam_exp"]
    mask_cam = sim["mask_cam"].copy()

    # Find middle 5 seconds
    dropout_start = 2.5  # Start dropout at 2.5s
    dropout_end = 7.5  # End dropout at 7.5s
    dropout_mask = (t_cam >= dropout_start) & (t_cam <= dropout_end)
    mask_cam[dropout_mask] = False

    # Run EKF with sensor fusion
    ekf_config = EKFConfig(
        adaptive_q_during_dropout=True,
        dropout_q_pos_multiplier=2.0,
        dropout_q_vel_multiplier=2.0,
        dropout_q_bias_multiplier=0.0,
        freeze_bias_during_blackout=True,
        reduce_imu_noise_during_blackout=True,
        blackout_imu_noise_scale=0.1,
        use_heading_measurement=True,
    )

    result = extended_kalman_filter(
        ekf_config,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        mask_cam,
    )

    # Run naive vision-only tracking
    vision_pos, vision_cov = naive_vision_only_tracking(t_cam, sim["Z_cam_led1"], mask_cam)

    # Get ground truth
    layout = get_layout("2d_full")
    t_imu = sim["t_imu"]
    X_truth = sim["X_truth"]
    cam_indices = np.searchsorted(t_imu, t_cam)
    cam_indices = np.clip(cam_indices, 0, len(t_imu) - 1)
    pos_truth = X_truth[cam_indices, :2]

    # EKF results
    ekf_pos = result.filtered_means[:, layout.pos_idx]
    # Extract 2×2 position covariances for each time step
    ekf_cov = np.array(
        [
            result.filtered_covariances[i][np.ix_(layout.pos_idx, layout.pos_idx)]
            for i in range(len(result.filtered_means))
        ]
    )

    # Create figure with 2 panels (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=3.0)

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
        ax.set_xlabel("X (meters)", fontsize=24, weight="bold")
        ax.set_ylabel("Y (meters)", fontsize=24, weight="bold")
        ax.set_title(title, fontsize=32, weight="bold", color=color, pad=15)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Add overall title
    title_text = fig.suptitle("", fontsize=24, weight="bold", y=0.98)

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
            ax.set_xlabel("X (meters)", fontsize=24, weight="bold")
            ax.set_ylabel("Y (meters)", fontsize=24, weight="bold")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        # Current time
        t_now = t_cam[frame]
        in_dropout = not mask_cam[frame]

        # Update title with time and dropout status
        dropout_text = "⚠ CAMERA DROPOUT" if in_dropout else "Camera Active"
        dropout_color = RED if in_dropout else GREEN
        title_text.set_text(
            f"Before & After: Vision-Only vs Sensor Fusion\n"
            f"Time: {t_now:.2f}s  |  {dropout_text}"
        )
        title_text.set_color(dropout_color)

        # LEFT PANEL: Vision-only
        ax_left = axes[0]
        ax_left.set_title("Vision-Only Tracking", fontsize=32, weight="bold", color=RED, pad=15)

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
            vision_pos[frame, 0], vision_pos[frame, 1], "o", color=RED, markersize=15, zorder=10
        )

        # Plot uncertainty ellipse
        plot_covariance_ellipse(
            ax_left, vision_pos[frame], vision_cov[frame], RED, alpha=0.3, n_std=2
        )

        # Calculate current error
        vision_error = np.linalg.norm(vision_pos[frame] - pos_truth[frame])
        ax_left.text(
            0.02,
            0.98,
            f"Position Error: {vision_error*100:.1f} cm\n"
            f"2σ Uncertainty: {2*np.sqrt(vision_cov[frame, 0, 0])*100:.1f} cm",
            transform=ax_left.transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9),
            weight="bold",
        )

        ax_left.legend(fontsize=13, loc="lower right")

        # RIGHT PANEL: Sensor fusion
        ax_right = axes[1]
        ax_right.set_title("Sensor Fusion (EKF)", fontsize=32, weight="bold", color=GREEN, pad=15)

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

        # Plot current position
        ax_right.plot(
            ekf_pos[frame, 0], ekf_pos[frame, 1], "o", color=GREEN, markersize=15, zorder=10
        )

        # Plot uncertainty ellipse
        plot_covariance_ellipse(ax_right, ekf_pos[frame], ekf_cov[frame], GREEN, alpha=0.3, n_std=2)

        # Calculate current error
        ekf_error = np.linalg.norm(ekf_pos[frame] - pos_truth[frame])
        ax_right.text(
            0.02,
            0.98,
            f"Position Error: {ekf_error*100:.1f} cm\n"
            f"2σ Uncertainty: {2*np.sqrt(ekf_cov[frame, 0, 0])*100:.1f} cm",
            transform=ax_right.transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9),
            weight="bold",
        )

        ax_right.legend(fontsize=13, loc="lower right")

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
