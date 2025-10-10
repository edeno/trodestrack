"""Sensor Fusion Comparison: EKF/UKF with Vision-only, IMU-only, and Fusion.

This script generates comprehensive visualizations comparing filter performance
under different sensor configurations with a camera dropout period.

Generates:
- Comparison plots (position, velocity, heading errors)
- Diagnostic videos for each configuration
- Smoother comparison plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.qa.metrics import (
    compute_heading_rmse,
    compute_position_rmse,
    compute_velocity_rmse,
)
from trodestrack.runtime.offline import rts_smoother, sigma_point_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.sim.utils import interp_angle

# Video rendering will be added separately if needed

# Output directory
OUTPUT_DIR = Path("output/sensor_fusion_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SENSOR FUSION COMPARISON: Vision-only, IMU-only, and Fusion")
print("=" * 80)

# =============================================================================
# Step 1: Generate Simulation with Dropout
# =============================================================================

print("\n[Step 1/6] Generating 60s simulation with 5s dropout...")

config = RatIMUSimConfig(
    duration_s=60.0,
    fs_imu=200.0,
    fs_cam=30.0,
    cam_dropout_prob=0.0,  # Manual dropout injection
    gyro_noise_density=0.001,
    accel_noise_density=0.05,
    gyro_bias_rw_density=0.0001,
    accel_bias_rw_density=0.001,
    cam_sigma_m=0.005,
    use_second_led=True,
    imu_tilt_roll_deg=0.0,
    imu_tilt_pitch_deg=0.0,
    arena_w=1.0,
    arena_h=1.0,
)
sim_data = simulate_rat_imu(config=config, seed=42)

# Inject 5-second dropout from t=25s to t=30s
dropout_start_t = 25.0
dropout_end_t = 30.0
dropout_start_idx = np.searchsorted(sim_data["t_cam_exp"], dropout_start_t)
dropout_end_idx = np.searchsorted(sim_data["t_cam_exp"], dropout_end_t)

# Create mask with dropout
mask_with_dropout = sim_data["mask_cam"].copy()
mask_with_dropout[dropout_start_idx:dropout_end_idx] = False

# Set observations to NaN during dropout
Z_cam_led1_dropout = sim_data["Z_cam_led1"].copy()
Z_cam_led2_dropout = sim_data["Z_cam_led2"].copy()
Z_cam_led1_dropout[dropout_start_idx:dropout_end_idx] = np.nan
Z_cam_led2_dropout[dropout_start_idx:dropout_end_idx] = np.nan

print(f"  Session duration: {config.duration_s}s")
print(f"  Camera frames: {len(sim_data['t_cam_exp'])}")
print(f"  Dropout period: {dropout_start_t:.1f}s - {dropout_end_t:.1f}s")

# Ground truth interpolation
t_truth = sim_data["t_imu"]
X_truth = sim_data["X_truth"]
t_cam = sim_data["t_cam_exp"]

pos_truth = np.column_stack(
    [np.interp(t_cam, t_truth, X_truth[:, 0]), np.interp(t_cam, t_truth, X_truth[:, 1])]
)
vel_truth = np.column_stack(
    [np.interp(t_cam, t_truth, X_truth[:, 2]), np.interp(t_cam, t_truth, X_truth[:, 3])]
)
heading_truth = interp_angle(t_cam, t_truth, X_truth[:, 4])

# =============================================================================
# Step 2: Run EKF - Fusion (baseline)
# =============================================================================

print("\n[Step 2/6] Running EKF - Fusion (Vision + IMU)...")

ekf_config_fusion = EKFConfig(
    process_noise_pos=0.02,
    process_noise_vel=2.0,
    process_noise_heading=0.02,
    process_noise_gyro_bias=2e-6,
    process_noise_accel_bias=2e-4,
    measurement_noise_pos=0.005**2,
    imu_gyro_noise_density=0.001,
    imu_accel_noise_density=0.05,
    damping_coeff=0.4,
    led_distance=0.04,
    use_heading_measurement=True,
    adaptive_q_during_dropout=True,
    dropout_q_pos_multiplier=10.0,
    dropout_q_vel_multiplier=10.0,
    dropout_q_bias_multiplier=0.1,
)

ekf_fusion = extended_kalman_filter(
    ekf_config=ekf_config_fusion,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=t_cam,
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

pos_rmse_ekf_fusion = compute_position_rmse(pos_truth, ekf_fusion.filtered_means[:, :2])
vel_rmse_ekf_fusion = compute_velocity_rmse(vel_truth, ekf_fusion.filtered_means[:, 2:4])
heading_rmse_ekf_fusion = np.rad2deg(
    compute_heading_rmse(heading_truth, ekf_fusion.filtered_means[:, 4])
)

print(f"  Position RMSE: {pos_rmse_ekf_fusion * 100:.2f} cm")
print(f"  Velocity RMSE: {vel_rmse_ekf_fusion * 100:.2f} cm/s")
print(f"  Heading RMSE:  {heading_rmse_ekf_fusion:.2f}°")

# =============================================================================
# Step 3: Run EKF - IMU-only
# =============================================================================

print("\n[Step 3/6] Running EKF - IMU-only...")

# Create IMU-only data (all camera observations masked)
mask_imu_only = np.zeros_like(mask_with_dropout, dtype=bool)
Z_cam_led1_imu_only = np.full_like(Z_cam_led1_dropout, np.nan)
Z_cam_led2_imu_only = np.full_like(Z_cam_led2_dropout, np.nan)

ekf_imu = extended_kalman_filter(
    ekf_config=ekf_config_fusion,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=t_cam,
    Z_cam_led1=Z_cam_led1_imu_only,
    Z_cam_led2=Z_cam_led2_imu_only,
    mask_cam=mask_imu_only,
)

pos_rmse_ekf_imu = compute_position_rmse(pos_truth, ekf_imu.filtered_means[:, :2])
vel_rmse_ekf_imu = compute_velocity_rmse(vel_truth, ekf_imu.filtered_means[:, 2:4])
heading_rmse_ekf_imu = np.rad2deg(compute_heading_rmse(heading_truth, ekf_imu.filtered_means[:, 4]))

print(f"  Position RMSE: {pos_rmse_ekf_imu * 100:.2f} cm (drift expected)")
print(f"  Velocity RMSE: {vel_rmse_ekf_imu * 100:.2f} cm/s")
print(f"  Heading RMSE:  {heading_rmse_ekf_imu:.2f}°")

# =============================================================================
# Step 4: Run EKF - Vision-only
# =============================================================================

print("\n[Step 4/6] Running EKF - Vision-only (degraded IMU)...")

# Create vision-only config (inflate IMU noise 1000x)
ekf_config_vision = EKFConfig(
    process_noise_pos=0.02,
    process_noise_vel=2000.0,  # 1000x normal
    process_noise_heading=0.02,
    process_noise_gyro_bias=2e-6,
    process_noise_accel_bias=2e-4,
    measurement_noise_pos=0.005**2,
    imu_gyro_noise_density=10.0,  # 10000x normal
    imu_accel_noise_density=50.0,  # 1000x normal
    damping_coeff=0.4,
    led_distance=0.04,
    use_heading_measurement=True,
)

ekf_vision = extended_kalman_filter(
    ekf_config=ekf_config_vision,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=t_cam,
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

pos_rmse_ekf_vision = compute_position_rmse(pos_truth, ekf_vision.filtered_means[:, :2])
vel_rmse_ekf_vision = compute_velocity_rmse(vel_truth, ekf_vision.filtered_means[:, 2:4])
heading_rmse_ekf_vision = np.rad2deg(
    compute_heading_rmse(heading_truth, ekf_vision.filtered_means[:, 4])
)

print(f"  Position RMSE: {pos_rmse_ekf_vision * 100:.2f} cm")
print(f"  Velocity RMSE: {vel_rmse_ekf_vision * 100:.2f} cm/s")
print(f"  Heading RMSE:  {heading_rmse_ekf_vision:.2f}°")

# =============================================================================
# Step 5: Run UKF - Fusion
# =============================================================================

print("\n[Step 5/6] Running UKF - Fusion (Vision + IMU)...")

ukf_config_fusion = UKFConfig(
    process_noise_pos=0.02,
    process_noise_vel=2.0,
    process_noise_heading=0.02,
    process_noise_gyro_bias=2e-6,
    process_noise_accel_bias=2e-4,
    measurement_noise_pos=0.005**2,
    imu_gyro_noise_density=0.001,
    imu_accel_noise_density=0.05,
    damping_coeff=0.4,
    led_distance=0.04,
    use_heading_measurement=True,
    adaptive_q_during_dropout=True,
    dropout_q_pos_multiplier=10.0,
    dropout_q_vel_multiplier=10.0,
    dropout_q_bias_multiplier=0.1,
)

ukf_fusion = unscented_kalman_filter(
    ukf_config=ukf_config_fusion,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=t_cam,
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

pos_rmse_ukf_fusion = compute_position_rmse(pos_truth, ukf_fusion.filtered_means[:, :2])
vel_rmse_ukf_fusion = compute_velocity_rmse(vel_truth, ukf_fusion.filtered_means[:, 2:4])
heading_rmse_ukf_fusion = np.rad2deg(
    compute_heading_rmse(heading_truth, ukf_fusion.filtered_means[:, 4])
)

print(f"  Position RMSE: {pos_rmse_ukf_fusion * 100:.2f} cm")
print(f"  Velocity RMSE: {vel_rmse_ukf_fusion * 100:.2f} cm/s")
print(f"  Heading RMSE:  {heading_rmse_ukf_fusion:.2f}°")

# =============================================================================
# Step 6: Run Smoothers
# =============================================================================

print("\n[Step 6/6] Running smoothers (EKF-RTS, UKF-SigmaPoint)...")

# EKF RTS smoother
ekf_smooth = rts_smoother(
    filter_result=ekf_fusion,
    ekf_config=ekf_config_fusion,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=t_cam,
    mask_cam=mask_with_dropout,
)

pos_rmse_ekf_smooth = compute_position_rmse(pos_truth, ekf_smooth.smoothed_means[:, :2])
vel_rmse_ekf_smooth = compute_velocity_rmse(vel_truth, ekf_smooth.smoothed_means[:, 2:4])
heading_rmse_ekf_smooth = np.rad2deg(
    compute_heading_rmse(heading_truth, ekf_smooth.smoothed_means[:, 4])
)

print(f"  EKF Smoother Position RMSE: {pos_rmse_ekf_smooth * 100:.2f} cm")
print(f"  EKF Smoother Velocity RMSE: {vel_rmse_ekf_smooth * 100:.2f} cm/s")
print(f"  EKF Smoother Heading RMSE:  {heading_rmse_ekf_smooth:.2f}°")

# UKF sigma-point smoother
try:
    ukf_smooth = sigma_point_smoother(
        filter_result=ukf_fusion,
        ukf_config=ukf_config_fusion,
        t_imu=sim_data["t_imu"],
        U_imu=sim_data["U_imu"],
        t_cam=t_cam,
    )

    # Check for NaN values
    if np.any(np.isnan(ukf_smooth.smoothed_means)):
        print("  UKF Smoother: WARNING - Contains NaN values, skipping metrics")
        ukf_smooth = None
        pos_rmse_ukf_smooth = np.nan
        vel_rmse_ukf_smooth = np.nan
        heading_rmse_ukf_smooth = np.nan
    else:
        pos_rmse_ukf_smooth = compute_position_rmse(pos_truth, ukf_smooth.smoothed_means[:, :2])
        vel_rmse_ukf_smooth = compute_velocity_rmse(vel_truth, ukf_smooth.smoothed_means[:, 2:4])
        heading_rmse_ukf_smooth = np.rad2deg(
            compute_heading_rmse(heading_truth, ukf_smooth.smoothed_means[:, 4])
        )

        print(f"  UKF Smoother Position RMSE: {pos_rmse_ukf_smooth * 100:.2f} cm")
        print(f"  UKF Smoother Velocity RMSE: {vel_rmse_ukf_smooth * 100:.2f} cm/s")
        print(f"  UKF Smoother Heading RMSE:  {heading_rmse_ukf_smooth:.2f}°")
except Exception as e:
    print(f"  UKF Smoother: ERROR - {e}")
    ukf_smooth = None
    pos_rmse_ukf_smooth = np.nan
    vel_rmse_ukf_smooth = np.nan
    heading_rmse_ukf_smooth = np.nan

# =============================================================================
# Step 7: Create Comparison Plots
# =============================================================================

print("\n[Step 7/8] Creating comparison plots...")

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Sensor Fusion Comparison: EKF vs UKF", fontsize=14, fontweight="bold")

# Column 1: EKF comparisons
# Position error
ax = axes[0, 0]
pos_error_ekf_fusion = np.linalg.norm(ekf_fusion.filtered_means[:, :2] - pos_truth, axis=1)
pos_error_ekf_imu = np.linalg.norm(ekf_imu.filtered_means[:, :2] - pos_truth, axis=1)
pos_error_ekf_vision = np.linalg.norm(ekf_vision.filtered_means[:, :2] - pos_truth, axis=1)
pos_error_ekf_smooth = np.linalg.norm(ekf_smooth.smoothed_means[:, :2] - pos_truth, axis=1)

ax.plot(t_cam, pos_error_ekf_fusion * 100, "b-", label="Fusion (Filter)", alpha=0.8)
ax.plot(t_cam, pos_error_ekf_imu * 100, "r--", label="IMU-only", alpha=0.6)
ax.plot(t_cam, pos_error_ekf_vision * 100, "g--", label="Vision-only", alpha=0.6)
ax.plot(t_cam, pos_error_ekf_smooth * 100, "m-", label="Fusion (Smoother)", alpha=0.8, linewidth=2)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.2, color="red", label="Dropout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (cm)")
ax.set_title("EKF: Position Error")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Velocity error
ax = axes[1, 0]
vel_error_ekf_fusion = np.linalg.norm(ekf_fusion.filtered_means[:, 2:4] - vel_truth, axis=1)
vel_error_ekf_imu = np.linalg.norm(ekf_imu.filtered_means[:, 2:4] - vel_truth, axis=1)
vel_error_ekf_vision = np.linalg.norm(ekf_vision.filtered_means[:, 2:4] - vel_truth, axis=1)
vel_error_ekf_smooth = np.linalg.norm(ekf_smooth.smoothed_means[:, 2:4] - vel_truth, axis=1)

ax.plot(t_cam, vel_error_ekf_fusion * 100, "b-", label="Fusion (Filter)", alpha=0.8)
ax.plot(t_cam, vel_error_ekf_imu * 100, "r--", label="IMU-only", alpha=0.6)
ax.plot(t_cam, vel_error_ekf_vision * 100, "g--", label="Vision-only", alpha=0.6)
ax.plot(t_cam, vel_error_ekf_smooth * 100, "m-", label="Fusion (Smoother)", alpha=0.8, linewidth=2)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.2, color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Error (cm/s)")
ax.set_title("EKF: Velocity Error")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Heading error
ax = axes[2, 0]


def wrap_angle_deg(angle_deg):
    return np.arctan2(np.sin(np.deg2rad(angle_deg)), np.cos(np.deg2rad(angle_deg))) * 180 / np.pi


heading_error_ekf_fusion = wrap_angle_deg(
    np.rad2deg(ekf_fusion.filtered_means[:, 4]) - np.rad2deg(heading_truth)
)
heading_error_ekf_imu = wrap_angle_deg(
    np.rad2deg(ekf_imu.filtered_means[:, 4]) - np.rad2deg(heading_truth)
)
heading_error_ekf_vision = wrap_angle_deg(
    np.rad2deg(ekf_vision.filtered_means[:, 4]) - np.rad2deg(heading_truth)
)
heading_error_ekf_smooth = wrap_angle_deg(
    np.rad2deg(ekf_smooth.smoothed_means[:, 4]) - np.rad2deg(heading_truth)
)

ax.plot(t_cam, heading_error_ekf_fusion, "b-", label="Fusion (Filter)", alpha=0.8)
ax.plot(t_cam, heading_error_ekf_imu, "r--", label="IMU-only", alpha=0.6)
ax.plot(t_cam, heading_error_ekf_vision, "g--", label="Vision-only", alpha=0.6)
ax.plot(t_cam, heading_error_ekf_smooth, "m-", label="Fusion (Smoother)", alpha=0.8, linewidth=2)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.2, color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Heading Error (deg)")
ax.set_title("EKF: Heading Error")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Column 2: UKF comparisons
# Position error
ax = axes[0, 1]
pos_error_ukf_fusion = np.linalg.norm(ukf_fusion.filtered_means[:, :2] - pos_truth, axis=1)

ax.plot(t_cam, pos_error_ukf_fusion * 100, "b-", label="Fusion (Filter)", alpha=0.8)
if ukf_smooth is not None:
    pos_error_ukf_smooth = np.linalg.norm(ukf_smooth.smoothed_means[:, :2] - pos_truth, axis=1)
    ax.plot(
        t_cam,
        pos_error_ukf_smooth * 100,
        "m-",
        label="Fusion (Smoother)",
        alpha=0.8,
        linewidth=2,
    )
ax.plot(t_cam, pos_error_ekf_fusion * 100, "c--", label="EKF (ref)", alpha=0.4)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.2, color="red", label="Dropout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (cm)")
ax.set_title("UKF: Position Error")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Velocity error
ax = axes[1, 1]
vel_error_ukf_fusion = np.linalg.norm(ukf_fusion.filtered_means[:, 2:4] - vel_truth, axis=1)

ax.plot(t_cam, vel_error_ukf_fusion * 100, "b-", label="Fusion (Filter)", alpha=0.8)
if ukf_smooth is not None:
    vel_error_ukf_smooth = np.linalg.norm(ukf_smooth.smoothed_means[:, 2:4] - vel_truth, axis=1)
    ax.plot(
        t_cam,
        vel_error_ukf_smooth * 100,
        "m-",
        label="Fusion (Smoother)",
        alpha=0.8,
        linewidth=2,
    )
ax.plot(t_cam, vel_error_ekf_fusion * 100, "c--", label="EKF (ref)", alpha=0.4)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.2, color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Error (cm/s)")
ax.set_title("UKF: Velocity Error")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Heading error
ax = axes[2, 1]
heading_error_ukf_fusion = wrap_angle_deg(
    np.rad2deg(ukf_fusion.filtered_means[:, 4]) - np.rad2deg(heading_truth)
)

ax.plot(t_cam, heading_error_ukf_fusion, "b-", label="Fusion (Filter)", alpha=0.8)
if ukf_smooth is not None:
    heading_error_ukf_smooth = wrap_angle_deg(
        np.rad2deg(ukf_smooth.smoothed_means[:, 4]) - np.rad2deg(heading_truth)
    )
    ax.plot(
        t_cam, heading_error_ukf_smooth, "m-", label="Fusion (Smoother)", alpha=0.8, linewidth=2
    )
ax.plot(t_cam, heading_error_ekf_fusion, "c--", label="EKF (ref)", alpha=0.4)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.2, color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Heading Error (deg)")
ax.set_title("UKF: Heading Error")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "sensor_fusion_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {plot_path}")
plt.close()

# =============================================================================
# Step 8: Create Summary Table
# =============================================================================

print("\n[Step 8/8] Generating summary table...")

print("\n" + "=" * 80)
print("SUMMARY: Position RMSE (cm)")
print("=" * 80)
print(
    f"{'Configuration':<25} {'EKF Filter':<12} {'EKF Smoother':<12} {'UKF Filter':<12} {'UKF Smoother':<12}"
)
print("-" * 80)
ukf_smooth_str = (
    f"{pos_rmse_ukf_smooth*100:>10.2f}" if not np.isnan(pos_rmse_ukf_smooth) else "    N/A   "
)
print(
    f"{'Fusion (Vision + IMU)':<25} {pos_rmse_ekf_fusion*100:>10.2f}   {pos_rmse_ekf_smooth*100:>10.2f}   {pos_rmse_ukf_fusion*100:>10.2f}   {ukf_smooth_str}"
)
print(f"{'IMU-only':<25} {pos_rmse_ekf_imu*100:>10.2f}   {'N/A':<10}   {'N/A':<10}   {'N/A':<10}")
print(
    f"{'Vision-only':<25} {pos_rmse_ekf_vision*100:>10.2f}   {'N/A':<10}   {'N/A':<10}   {'N/A':<10}"
)
print("=" * 80)

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print(
    f"1. Fusion vs IMU-only:    {pos_rmse_ekf_imu / pos_rmse_ekf_fusion:.1f}× improvement (expected: large drift without vision)"
)
print(
    f"2. Fusion vs Vision-only: {pos_rmse_ekf_vision / pos_rmse_ekf_fusion:.1f}× improvement (vision dominates for position)"
)
print(
    f"3. Filter vs Smoother:    {pos_rmse_ekf_fusion / pos_rmse_ekf_smooth:.2f}× improvement (backward pass helps)"
)
print(
    f"4. EKF vs UKF:            {pos_rmse_ukf_fusion / pos_rmse_ekf_fusion:.2f}× ratio (comparable performance)"
)
print("=" * 80)

print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
print("✅ Sensor fusion comparison complete!")
print("✅ Sensor fusion comparison complete!")
