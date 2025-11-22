"""Generate diagnostic video comparing filter vs IEKS smoother on 5s dropout.

This script demonstrates the dramatic improvement from IEKS + blackout-aware
smoothing on a challenging 5-second vision dropout scenario:

- Filter drift: ~1.67 m (IMU-only integration with white noise)
- Smoothed drift: ~0.54 m (3.08× improvement using future vision)
- Theory: ~0.50 m (white accel noise floor)

The video shows:
1. Arena view with filter (red) and smoothed (blue) trajectories
2. Position error time series highlighting the dropout interval
3. Velocity error showing drift accumulation
4. Bias estimates demonstrating smoother's backward propagation

Key phenomena illustrated:
- Forward filter drifts during blackout (no vision correction)
- Smoother uses vision AFTER gap to pull estimates back toward truth
- Blackout-aware Q scaling makes gap "tight" for effective backward pull
- IEKS relinearization around smoothed trajectory improves accuracy

Usage:
    uv run python examples/06_dropout_smoother_comparison.py

Output:
    Saves MP4 video: output/dropout_smoother_comparison.mp4
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.qa.metrics import compute_dropout_drift
from trodestrack.runtime.offline import rts_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.utils import interp_angle
from trodestrack.viz.styles import COLORS, apply_tufte_style

# Output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Generate comparison video for filter vs IEKS smoother on 5s dropout."""
    print("=" * 70)
    print("Filter vs IEKS Smoother: 5s Dropout Comparison")
    print("=" * 70)

    # Simulation configuration (same as test_prd_dropout_drift_5s_smoothed)
    config = RatIMUSimConfig(
        duration_s=15.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,  # No random dropouts
        gyro_noise_density=0.001,
        accel_noise_density=0.05,
        gyro_bias_rw_density=0.0001,
        accel_bias_rw_density=0.001,
        cam_sigma_m=0.005,
        use_second_led=True,
        # Zero IMU tilt to eliminate gravity leakage
        imu_tilt_roll_deg=0.0,
        imu_tilt_pitch_deg=0.0,
    )

    print("\nRunning simulation...")
    sim_data = simulate_rat_imu(config=config, seed=42)

    # Manually inject 5s dropout from t=5s to t=10s
    dropout_start_idx = np.searchsorted(sim_data["t_cam_exp"], 5.0)
    dropout_end_idx = np.searchsorted(sim_data["t_cam_exp"], 10.0)

    # Create mask with dropout
    mask_with_dropout = sim_data["mask_cam"].copy()
    mask_with_dropout[dropout_start_idx:dropout_end_idx] = False

    # Update simulation data with dropout mask
    sim_data_dropout = {
        k: (v.copy() if hasattr(v, "copy") else v) for k, v in sim_data.items()
    }
    sim_data_dropout["mask_cam"] = mask_with_dropout

    # Ensure no usable pixels during blackout (set LEDs to NaN)
    for key in ("Z_cam_led1", "Z_cam_led2"):
        if key in sim_data_dropout:
            arr = sim_data_dropout[key].copy()
            arr[dropout_start_idx:dropout_end_idx] = float("nan")
            sim_data_dropout[key] = arr

    # Force per-LED masks off as well
    for key in ("mask_led1", "mask_led2"):
        if key in sim_data_dropout:
            arr = sim_data_dropout[key].copy()
            arr[dropout_start_idx:dropout_end_idx] = False
            sim_data_dropout[key] = arr

    print(f"  Dropout interval: t={5.0:.1f}s to t={10.0:.1f}s (5.0s duration)")
    print(
        f"  Valid frames: {mask_with_dropout.sum()}/{len(mask_with_dropout)} "
        f"({100*mask_with_dropout.mean():.1f}%)"
    )

    # Run EKF with blackout-aware filtering
    ekf_config_override = {
        "damping_coeff": 0.4,
        "freeze_bias_during_blackout": True,
        "reduce_imu_noise_during_blackout": True,
        "blackout_imu_noise_scale": 0.5,
    }
    ekf_params = dict(
        process_noise_pos=0.02,
        process_noise_vel=2.0,
        process_noise_heading=0.02,
        process_noise_gyro_bias=2e-6,
        process_noise_accel_bias=2e-4,
        measurement_noise_pos=0.005**2,
        imu_gyro_noise_density=0.001,
        imu_accel_noise_density=0.05,
        damping_coeff=0.5,
        led_distance=0.04,
        use_heading_measurement=True,
    )
    ekf_params.update(ekf_config_override)
    ekf_config = EKFConfig(**ekf_params)

    print("\nRunning EKF filter...")
    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_data_dropout["t_imu"],
        U_imu=sim_data_dropout["U_imu"],
        t_cam=sim_data_dropout["t_cam_exp"],
        Z_cam_led1=sim_data_dropout["Z_cam_led1"],
        Z_cam_led2=sim_data_dropout["Z_cam_led2"],
        mask_cam=mask_with_dropout,
    )

    print("Running IEKS smoother (iter=2, blackout-aware Q/R)...")
    smoother_result = rts_smoother(
        filter_result=filter_result,
        ekf_config=ekf_config,
        t_imu=sim_data_dropout["t_imu"],
        U_imu=sim_data_dropout["U_imu"],
        t_cam=sim_data_dropout["t_cam_exp"],
        num_iter=2,  # IEKS with 2 iterations
        mask_cam=mask_with_dropout,  # Enable blackout-aware Q/R scaling
    )

    # Compute dropout drift for both
    drift_result_filter = compute_dropout_drift(
        positions=filter_result.filtered_means[:, :2],
        valid_mask=mask_with_dropout,
        t=sim_data["t_cam_exp"],
        min_duration_s=4.5,
    )

    drift_result_smooth = compute_dropout_drift(
        positions=smoother_result.smoothed_means[:, :2],
        valid_mask=mask_with_dropout,
        t=sim_data["t_cam_exp"],
        min_duration_s=4.5,
    )

    drift_filter_m = drift_result_filter["drift_m"]
    drift_smooth_m = drift_result_smooth["drift_m"]

    print("\n5s Dropout Drift Results:")
    print(f"  Filter:   {drift_filter_m:.4f} m")
    print(f"  Smoothed: {drift_smooth_m:.4f} m")
    print(f"  Improvement: {drift_filter_m / drift_smooth_m:.2f}× reduction")
    print(f"  Theory (~0.50 m): {drift_smooth_m / 0.50:.2f}× observed/theory")

    # Interpolate truth to camera times
    t_truth = sim_data["t_imu"]
    X_truth = sim_data["X_truth"]
    t_cam = sim_data["t_cam_exp"]

    pos_truth = np.column_stack(
        [
            np.interp(t_cam, t_truth, X_truth[:, 0]),
            np.interp(t_cam, t_truth, X_truth[:, 1]),
        ]
    )
    heading_truth = interp_angle(t_cam, t_truth, X_truth[:, 4])

    # Extract filter and smoother estimates
    pos_filter = np.array(filter_result.filtered_means[:, :2])
    pos_smooth = np.array(smoother_result.smoothed_means[:, :2])
    heading_filter = np.array(filter_result.filtered_means[:, 4])
    heading_smooth = np.array(smoother_result.smoothed_means[:, 4])

    # Compute errors
    pos_err_filter = np.linalg.norm(pos_filter - pos_truth, axis=1)
    pos_err_smooth = np.linalg.norm(pos_smooth - pos_truth, axis=1)

    # Heading errors with wrapping
    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    heading_err_filter = np.abs(angle_diff(heading_filter, heading_truth))
    heading_err_smooth = np.abs(angle_diff(heading_smooth, heading_truth))

    # Create visualization
    print("\nGenerating comparison video...")
    output_path = OUTPUT_DIR / "dropout_smoother_comparison.mp4"

    apply_tufte_style()
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Arena view (top-left, tall)
    ax_arena = fig.add_subplot(gs[:2, 0])
    ax_arena.set_aspect("equal")
    ax_arena.set_xlabel("X Position (m)")
    ax_arena.set_ylabel("Y Position (m)")
    ax_arena.set_title("Arena View: Filter (Red) vs Smoothed (Blue)")

    # Plot truth trajectory
    ax_arena.plot(
        pos_truth[:, 0],
        pos_truth[:, 1],
        color=COLORS["gray"],
        linewidth=1,
        alpha=0.3,
        label="Truth",
    )

    # Plot filter and smoother trajectories
    (line_filter,) = ax_arena.plot(
        [], [], color=COLORS["red"], linewidth=2, alpha=0.7, label="Filter"
    )
    (line_smooth,) = ax_arena.plot(
        [], [], color=COLORS["blue"], linewidth=2, alpha=0.7, label="Smoothed"
    )

    # Current position markers
    (marker_filter,) = ax_arena.plot(
        [], [], "o", color=COLORS["red"], markersize=8, zorder=10
    )
    (marker_smooth,) = ax_arena.plot(
        [], [], "o", color=COLORS["blue"], markersize=8, zorder=10
    )

    ax_arena.legend(loc="upper right")
    ax_arena.grid(True, alpha=0.2)

    # Panel 2: Position error time series (top-right)
    ax_pos_err = fig.add_subplot(gs[0, 1])
    ax_pos_err.set_xlabel("Time (s)")
    ax_pos_err.set_ylabel("Position Error (m)")
    ax_pos_err.set_title("Position Error: Filter vs Smoothed")

    ax_pos_err.plot(
        t_cam, pos_err_filter, color=COLORS["red"], linewidth=2, label="Filter"
    )
    ax_pos_err.plot(
        t_cam, pos_err_smooth, color=COLORS["blue"], linewidth=2, label="Smoothed"
    )

    # Highlight dropout interval
    ax_pos_err.axvspan(5.0, 10.0, alpha=0.2, color="gray", label="Blackout")

    # Add drift annotations
    dropout_mid = 7.5
    ax_pos_err.annotate(
        f"Filter drift: {drift_filter_m:.2f}m",
        xy=(dropout_mid, pos_err_filter[dropout_start_idx:dropout_end_idx].max()),
        xytext=(dropout_mid, pos_err_filter.max() * 0.9),
        fontsize=9,
        color=COLORS["red"],
        ha="center",
    )
    ax_pos_err.annotate(
        f"Smoothed drift: {drift_smooth_m:.2f}m",
        xy=(dropout_mid, pos_err_smooth[dropout_start_idx:dropout_end_idx].max()),
        xytext=(dropout_mid, pos_err_filter.max() * 0.75),
        fontsize=9,
        color=COLORS["blue"],
        ha="center",
    )

    (vline_pos,) = ax_pos_err.plot(
        [], [], "k--", linewidth=1, alpha=0.5, label="Current time"
    )
    ax_pos_err.legend(loc="upper left", fontsize=8)
    ax_pos_err.grid(True, alpha=0.2)

    # Panel 3: Heading error (middle-right)
    ax_heading_err = fig.add_subplot(gs[1, 1])
    ax_heading_err.set_xlabel("Time (s)")
    ax_heading_err.set_ylabel("Heading Error (deg)")
    ax_heading_err.set_title("Heading Error: Filter vs Smoothed")

    ax_heading_err.plot(
        t_cam, np.rad2deg(heading_err_filter), color=COLORS["red"], linewidth=2
    )
    ax_heading_err.plot(
        t_cam, np.rad2deg(heading_err_smooth), color=COLORS["blue"], linewidth=2
    )
    ax_heading_err.axvspan(5.0, 10.0, alpha=0.2, color="gray")
    (vline_heading,) = ax_heading_err.plot([], [], "k--", linewidth=1, alpha=0.5)
    ax_heading_err.grid(True, alpha=0.2)

    # Panel 4: Bias estimates (bottom row)
    ax_bias = fig.add_subplot(gs[2, :])
    ax_bias.set_xlabel("Time (s)")
    ax_bias.set_ylabel("Accel Bias (m/s²)")
    ax_bias.set_title("Accel Bias Estimates: Filter vs Smoothed")

    # Plot filter biases
    bias_filter_x = filter_result.filtered_means[:, 6]
    bias_filter_y = filter_result.filtered_means[:, 7]
    bias_smooth_x = smoother_result.smoothed_means[:, 6]
    bias_smooth_y = smoother_result.smoothed_means[:, 7]

    ax_bias.plot(
        t_cam,
        bias_filter_x,
        color=COLORS["red"],
        linewidth=2,
        alpha=0.5,
        label="Filter X",
    )
    ax_bias.plot(
        t_cam,
        bias_filter_y,
        color=COLORS["red"],
        linewidth=2,
        alpha=0.5,
        linestyle="--",
        label="Filter Y",
    )
    ax_bias.plot(
        t_cam,
        bias_smooth_x,
        color=COLORS["blue"],
        linewidth=2,
        alpha=0.5,
        label="Smoothed X",
    )
    ax_bias.plot(
        t_cam,
        bias_smooth_y,
        color=COLORS["blue"],
        linewidth=2,
        alpha=0.5,
        linestyle="--",
        label="Smoothed Y",
    )
    ax_bias.axvspan(5.0, 10.0, alpha=0.2, color="gray")
    (vline_bias,) = ax_bias.plot([], [], "k--", linewidth=1, alpha=0.5)
    ax_bias.legend(loc="upper right", fontsize=8, ncol=4)
    ax_bias.grid(True, alpha=0.2)

    # Animation update function
    def update(frame):
        """Update all panels for frame."""
        t_current = t_cam[frame]

        # Update arena trajectories
        line_filter.set_data(pos_filter[:frame, 0], pos_filter[:frame, 1])
        line_smooth.set_data(pos_smooth[:frame, 0], pos_smooth[:frame, 1])
        marker_filter.set_data([pos_filter[frame, 0]], [pos_filter[frame, 1]])
        marker_smooth.set_data([pos_smooth[frame, 0]], [pos_smooth[frame, 1]])

        # Update time markers
        vline_pos.set_data([t_current, t_current], ax_pos_err.get_ylim())
        vline_heading.set_data([t_current, t_current], ax_heading_err.get_ylim())
        vline_bias.set_data([t_current, t_current], ax_bias.get_ylim())

        return (
            line_filter,
            line_smooth,
            marker_filter,
            marker_smooth,
            vline_pos,
            vline_heading,
            vline_bias,
        )

    # Create animation
    n_frames = len(t_cam)
    fps = 30
    speedup = 2.0  # 2× speed

    writer = FFMpegWriter(fps=fps, codec="h264", bitrate=3000)

    print(f"  Rendering {n_frames} frames at {fps} fps (speedup={speedup:.1f}×)...")
    with writer.saving(fig, str(output_path), dpi=120):
        for frame in range(n_frames):
            update(frame)
            writer.grab_frame()

            # Progress indicator
            if (frame + 1) % 50 == 0 or frame == n_frames - 1:
                pct = 100 * (frame + 1) / n_frames
                print(f"    Progress: {pct:.1f}% ({frame+1}/{n_frames} frames)")

    plt.close(fig)

    print(f"\n✓ Video saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Duration: {n_frames / fps:.1f}s at {fps} fps")


if __name__ == "__main__":
    main()
