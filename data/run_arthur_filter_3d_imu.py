"""Run trodestrack EKF on Arthur session data with 3D IMU.

This script demonstrates running the Extended Kalman Filter on real data using
the full 6-axis IMU (gyro X/Y/Z + accel X/Y/Z) with the LAYOUT_2D_CAM_3D_IMU
state layout (10D state: x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az).

The key difference from the standard 2D setup:
- IMU input: [N × 6] instead of [N × 3]
- State layout: "2d_cam_3d_imu" (10D) instead of "2d_full" (8D)
- Accel bias: 3D (b_ax, b_ay, b_az) instead of 2D (b_ax, b_ay)
- Velocity: 3D (vx, vy, vz) instead of 2D (vx, vy)

This enables:
- Gravity compensation using all 3 accel axes
- Detection of vertical motion (rearing, jumping) via vz
- Better bias estimation with 3D accel data

Usage:
    cd data/
    uv run python run_arthur_filter_3d_imu.py

Output:
    - Console: Filter statistics and performance metrics
    - arthur_filter_results/: Directory with plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from load_arthur_session import load_arthur_session

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.viz.styles import COLORS, apply_tufte_style

# Apply clean visualization style
apply_tufte_style()


def main():
    """Run EKF on Arthur data with 3D IMU and visualize results."""
    print("=" * 80)
    print("ARTHUR SESSION: EKF WITH 3D IMU (LAYOUT_2D_CAM_3D_IMU)")
    print("=" * 80)

    # =========================================================================
    # 1. Load Data
    # =========================================================================
    script_dir = Path(__file__).parent
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220314_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220314_imu_info.parquet"),
        imu_mode="3d",  # Load all 6 IMU axes
        verbose=True,
    )

    print("\n✓ Loaded session:")
    print(f"  Duration: {data.t_cam[-1]:.1f} s ({data.t_cam[-1]/60:.1f} min)")
    print(f"  Camera: {len(data.t_cam):,} frames at {data.fs_cam:.1f} Hz")
    print(f"  IMU: {len(data.t_imu):,} samples at {data.fs_imu:.1f} Hz")
    print(f"  IMU shape: {data.U_imu.shape} (6 axes: gyro X/Y/Z, accel X/Y/Z)")

    # =========================================================================
    # 2. Configure Filter
    # =========================================================================
    print("\nConfiguring EKF with LAYOUT_2D_CAM_3D_IMU (10D state)...")

    ekf_config = EKFConfig(
        # State layout
        state_mode="2d_cam_3d_imu",  # 10D: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
        # Process noise (tuned for rat tracking)
        process_noise_pos=0.02,  # Position diffusion
        process_noise_vel=2.0,  # Velocity diffusion
        process_noise_heading=0.02,  # Heading diffusion
        process_noise_gyro_bias=2e-6,  # Gyro bias random walk
        process_noise_accel_bias=2e-4,  # Accel bias random walk
        # Measurement noise
        measurement_noise_pos=0.005**2,  # Camera position noise (5mm std)
        measurement_noise_heading=0.05**2,  # LED heading noise (~3 deg std)
        # IMU noise densities
        imu_gyro_noise_density=0.0001,  # rad/s/√Hz
        imu_accel_noise_density=0.005,  # m/s²/√Hz
        # Dynamics
        damping_coeff=0.1,  # Velocity damping
        led_distance=data.led_distance,  # LED separation distance
        # Features
        use_heading_measurement=True,  # Use LED heading
        use_mahalanobis_gating=False,  # Disable gating (data is clean)
        enable_zupt=False,  # Disable ZUPT for now
    )

    print(f"  State mode: {ekf_config.state_mode}")
    print(f"  LED distance: {ekf_config.led_distance*100:.2f} cm")
    print(f"  Heading measurement: {ekf_config.use_heading_measurement}")
    print(f"  ZUPT: {ekf_config.enable_zupt}")

    # =========================================================================
    # 3. Run Filter
    # =========================================================================
    print("\nRunning EKF...")

    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=data.t_imu,
        U_imu=data.U_imu,  # [N × 6] for 3D IMU
        t_cam=data.t_cam,
        Z_cam_led1=data.Z_cam_led1,
        Z_cam_led2=data.Z_cam_led2,
        mask_cam=data.mask_cam,
    )

    print("✓ Filter complete!")
    print(f"  Filtered states: {result.filtered_means.shape}")
    print(f"  Marginal log-likelihood: {result.marginal_loglik:.1f}")

    # =========================================================================
    # 4. Extract and Visualize Results
    # =========================================================================
    print("\nGenerating visualizations...")

    # Extract states
    x_est = np.array(result.filtered_means)  # [N_cam × 10]
    # P_est = np.array(result.filtered_covariances)  # [N_cam × 10 × 10]  # For future uncertainty plots
    t_cam = np.array(data.t_cam)

    # State indices for LAYOUT_2D_CAM_3D_IMU:
    # [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
    #  0  1  2   3   4  5  6    7    8    9
    pos_x = x_est[:, 0]
    pos_y = x_est[:, 1]
    vel_x = x_est[:, 2]
    vel_y = x_est[:, 3]
    vel_z = x_est[:, 4]  # NEW: vertical velocity
    heading = x_est[:, 5]
    bias_gz = x_est[:, 6]
    bias_ax = x_est[:, 7]
    bias_ay = x_est[:, 8]
    bias_az = x_est[:, 9]  # NEW: vertical accel bias

    # Camera measurements
    led1_x = data.Z_cam_led1[:, 0]
    led1_y = data.Z_cam_led1[:, 1]
    led2_x = data.Z_cam_led2[:, 0]
    led2_y = data.Z_cam_led2[:, 1]
    mid_x = (led1_x + led2_x) / 2
    mid_y = (led1_y + led2_y) / 2

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Trajectory
    ax_traj = fig.add_subplot(gs[0, :])
    ax_traj.plot(
        mid_x, mid_y, ".", color=COLORS["gray"], alpha=0.2, markersize=1, label="Camera (midpoint)"
    )
    ax_traj.plot(pos_x, pos_y, "-", color=COLORS["blue"], linewidth=1.5, label="EKF estimate")
    ax_traj.set_xlabel("X position (m)")
    ax_traj.set_ylabel("Y position (m)")
    ax_traj.set_title("2D Trajectory (10D State with 3D IMU)", fontweight="bold")
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect("equal")

    # Row 2: Velocities
    ax_vel_xy = fig.add_subplot(gs[1, 0])
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    ax_vel_xy.plot(t_cam, vel_mag, "-", color=COLORS["blue"], linewidth=1)
    ax_vel_xy.set_xlabel("Time (s)")
    ax_vel_xy.set_ylabel("Horizontal speed (m/s)")
    ax_vel_xy.set_title("Horizontal Speed", fontweight="bold")
    ax_vel_xy.grid(True, alpha=0.3)

    ax_vel_z = fig.add_subplot(gs[1, 1])
    ax_vel_z.plot(t_cam, vel_z, "-", color=COLORS["green"], linewidth=1)
    ax_vel_z.set_xlabel("Time (s)")
    ax_vel_z.set_ylabel("Vertical velocity (m/s)")
    ax_vel_z.set_title("Vertical Velocity (NEW!)", fontweight="bold")
    ax_vel_z.grid(True, alpha=0.3)
    ax_vel_z.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=0.5)

    ax_heading = fig.add_subplot(gs[1, 2])
    ax_heading.plot(t_cam, heading * 180 / np.pi, "-", color=COLORS["purple"], linewidth=1)
    ax_heading.set_xlabel("Time (s)")
    ax_heading.set_ylabel("Heading (deg)")
    ax_heading.set_title("Heading Angle", fontweight="bold")
    ax_heading.grid(True, alpha=0.3)

    # Row 3: Biases
    ax_bias_gz = fig.add_subplot(gs[2, 0])
    ax_bias_gz.plot(t_cam, bias_gz * 180 / np.pi, "-", color=COLORS["red"], linewidth=1)
    ax_bias_gz.set_xlabel("Time (s)")
    ax_bias_gz.set_ylabel("Gyro Z bias (deg/s)")
    ax_bias_gz.set_title("Gyro Bias (yaw rate)", fontweight="bold")
    ax_bias_gz.grid(True, alpha=0.3)
    ax_bias_gz.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=0.5)

    ax_bias_axy = fig.add_subplot(gs[2, 1])
    ax_bias_axy.plot(t_cam, bias_ax, "-", color=COLORS["orange"], linewidth=1, label="b_ax")
    ax_bias_axy.plot(t_cam, bias_ay, "-", color=COLORS["blue"], linewidth=1, label="b_ay")
    ax_bias_axy.set_xlabel("Time (s)")
    ax_bias_axy.set_ylabel("Accel bias (m/s²)")
    ax_bias_axy.set_title("Horizontal Accel Biases", fontweight="bold")
    ax_bias_axy.legend()
    ax_bias_axy.grid(True, alpha=0.3)
    ax_bias_axy.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=0.5)

    ax_bias_az = fig.add_subplot(gs[2, 2])
    ax_bias_az.plot(t_cam, bias_az, "-", color=COLORS["green"], linewidth=1)
    ax_bias_az.set_xlabel("Time (s)")
    ax_bias_az.set_ylabel("Accel Z bias (m/s²)")
    ax_bias_az.set_title("Vertical Accel Bias (NEW!)", fontweight="bold")
    ax_bias_az.grid(True, alpha=0.3)
    ax_bias_az.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=0.5)

    plt.suptitle(
        "Arthur Session: EKF with 3D IMU (2D Pos + 3D IMU)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # Save
    save_path = script_dir / "arthur_3d_imu_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved plot: {save_path}")

    plt.close(fig)

    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
