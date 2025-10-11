"""Debug: Why does Fusion have worse RMSE than Vision-only?

Investigates why fusion (8.46 cm) has worse position RMSE than vision-only (4.12 cm).
Hypothesis: IMU noise/bias is degrading position estimates during fusion.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.qa.metrics import compute_position_rmse
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

OUTPUT_DIR = Path("output/debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DEBUG: Why Fusion > Vision-only RMSE?")
print("=" * 80)

# Generate simulation
config = RatIMUSimConfig(
    duration_s=60.0,
    fs_imu=200.0,
    fs_cam=30.0,
    cam_dropout_prob=0.0,
    gyro_noise_density=0.001,
    accel_noise_density=0.05,
    gyro_bias_rw_density=0.0001,
    accel_bias_rw_density=0.001,
    cam_sigma_m=0.005,
    use_second_led=True,
    imu_tilt_roll_deg=0.0,
    imu_tilt_pitch_deg=0.0,
)
sim_data = simulate_rat_imu(config=config, seed=42)

# Inject dropout
dropout_start_t = 25.0
dropout_end_t = 30.0
dropout_start_idx = np.searchsorted(sim_data["t_cam_exp"], dropout_start_t)
dropout_end_idx = np.searchsorted(sim_data["t_cam_exp"], dropout_end_t)

mask_with_dropout = sim_data["mask_cam"].copy()
mask_with_dropout[dropout_start_idx:dropout_end_idx] = False

Z_cam_led1_dropout = sim_data["Z_cam_led1"].copy()
Z_cam_led2_dropout = sim_data["Z_cam_led2"].copy()
Z_cam_led1_dropout[dropout_start_idx:dropout_end_idx] = np.nan
Z_cam_led2_dropout[dropout_start_idx:dropout_end_idx] = np.nan

t_cam = sim_data["t_cam_exp"]
t_truth = sim_data["t_imu"]
X_truth = sim_data["X_truth"]

pos_truth = np.column_stack(
    [np.interp(t_cam, t_truth, X_truth[:, 0]), np.interp(t_cam, t_truth, X_truth[:, 1])]
)

# Run Fusion
print("\n[1/4] Running Fusion EKF...")
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

# Run Vision-only
print("\n[2/4] Running Vision-only EKF...")
ekf_config_vision = EKFConfig(
    process_noise_pos=0.02,
    process_noise_vel=2000.0,  # 1000x inflated
    process_noise_heading=0.02,
    process_noise_gyro_bias=2e-6,
    process_noise_accel_bias=2e-4,
    measurement_noise_pos=0.005**2,
    imu_gyro_noise_density=10.0,  # 10000x inflated
    imu_accel_noise_density=50.0,  # 1000x inflated
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

# Compute RMSE
print("\n[3/4] Computing RMSE...")

# Check which frames are valid (non-NaN)
valid_fusion = np.isfinite(ekf_fusion.filtered_means[:, 0]) & np.isfinite(
    ekf_fusion.filtered_means[:, 1]
)
valid_vision = np.isfinite(ekf_vision.filtered_means[:, 0]) & np.isfinite(
    ekf_vision.filtered_means[:, 1]
)

print(f"  Fusion: {valid_fusion.sum()}/{len(valid_fusion)} frames valid")
print(f"  Vision: {valid_vision.sum()}/{len(valid_vision)} frames valid")

# Overall RMSE
try:
    rmse_fusion = compute_position_rmse(pos_truth, ekf_fusion.filtered_means[:, :2])
    print(f"  Fusion RMSE (all valid frames): {rmse_fusion * 100:.2f} cm")
except ValueError as e:
    print(f"  Fusion RMSE: ERROR - {e}")
    rmse_fusion = np.nan

try:
    rmse_vision = compute_position_rmse(pos_truth, ekf_vision.filtered_means[:, :2])
    print(f"  Vision RMSE (all valid frames): {rmse_vision * 100:.2f} cm")
except ValueError as e:
    print(f"  Vision RMSE: ERROR - {e}")
    rmse_vision = np.nan

# RMSE excluding dropout period
mask_no_dropout = mask_with_dropout.copy()
try:
    rmse_fusion_no_dropout = compute_position_rmse(
        pos_truth, ekf_fusion.filtered_means[:, :2], mask=mask_no_dropout
    )
    print(f"  Fusion RMSE (excluding dropout): {rmse_fusion_no_dropout * 100:.2f} cm")
except ValueError as e:
    print(f"  Fusion RMSE (no dropout): ERROR - {e}")
    rmse_fusion_no_dropout = np.nan

try:
    rmse_vision_no_dropout = compute_position_rmse(
        pos_truth, ekf_vision.filtered_means[:, :2], mask=mask_no_dropout
    )
    print(f"  Vision RMSE (excluding dropout): {rmse_vision_no_dropout * 100:.2f} cm")
except ValueError as e:
    print(f"  Vision RMSE (no dropout): ERROR - {e}")
    rmse_vision_no_dropout = np.nan

# Plot position errors
print("\n[4/4] Creating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Position error over time
ax = axes[0, 0]
pos_error_fusion = np.linalg.norm(ekf_fusion.filtered_means[:, :2] - pos_truth, axis=1)
pos_error_vision = np.linalg.norm(ekf_vision.filtered_means[:, :2] - pos_truth, axis=1)

ax.plot(t_cam, pos_error_fusion * 100, "b-", linewidth=2, label="Fusion", alpha=0.8)
ax.plot(t_cam, pos_error_vision * 100, "g--", linewidth=2, label="Vision-only", alpha=0.6)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red", label="Dropout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (cm)")
ax.set_title("Position Error: Fusion vs Vision-only")
ax.set_ylim([0, 30])
ax.legend()
ax.grid(True, alpha=0.3)

# Position error (excluding dropout) - zoomed in
ax = axes[0, 1]
mask_plot = mask_with_dropout
ax.plot(
    t_cam[mask_plot],
    pos_error_fusion[mask_plot] * 100,
    "b-",
    linewidth=2,
    label="Fusion",
    alpha=0.8,
)
ax.plot(
    t_cam[mask_plot],
    pos_error_vision[mask_plot] * 100,
    "g--",
    linewidth=2,
    label="Vision-only",
    alpha=0.6,
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (cm)")
ax.set_title("Position Error (Dropout Frames Excluded)")
ax.legend()
ax.grid(True, alpha=0.3)

# Histogram of errors (excluding dropout)
ax = axes[1, 0]
ax.hist(pos_error_fusion[mask_plot] * 100, bins=50, alpha=0.6, label="Fusion", color="blue")
ax.hist(pos_error_vision[mask_plot] * 100, bins=50, alpha=0.6, label="Vision-only", color="green")
ax.set_xlabel("Position Error (cm)")
ax.set_ylabel("Count")
ax.set_title("Error Distribution (Excluding Dropout)")
ax.legend()
ax.grid(True, alpha=0.3)

# Valid frame counts
ax = axes[1, 1]
categories = ["Fusion\n(valid)", "Vision\n(valid)"]
valid_counts = [valid_fusion.sum(), valid_vision.sum()]
colors = ["blue", "green"]
ax.bar(categories, valid_counts, color=colors, alpha=0.6)
ax.axhline(
    len(valid_fusion),
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Total frames ({len(valid_fusion)})",
)
ax.set_ylabel("Number of Valid Frames")
ax.set_title("Valid Frames After Dropout")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = OUTPUT_DIR / "fusion_vs_vision_debug.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {plot_path}")
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("1. Valid frames:")
print(
    f"   Fusion: {valid_fusion.sum()}/{len(valid_fusion)} ({valid_fusion.sum()/len(valid_fusion)*100:.1f}%)"
)
print(
    f"   Vision: {valid_vision.sum()}/{len(valid_vision)} ({valid_vision.sum()/len(valid_vision)*100:.1f}%)"
)
print()
print("2. RMSE comparison:")
print(f"   Fusion (all):      {rmse_fusion * 100:.2f} cm")
print(f"   Vision (all):      {rmse_vision * 100:.2f} cm")
print(f"   Fusion (no dropout): {rmse_fusion_no_dropout * 100:.2f} cm")
print(f"   Vision (no dropout): {rmse_vision_no_dropout * 100:.2f} cm")
print()
print("3. Why is fusion worse?")
if not np.isnan(rmse_fusion_no_dropout) and not np.isnan(rmse_vision_no_dropout):
    if rmse_fusion_no_dropout < rmse_vision_no_dropout:
        print("   CORRECTED: Fusion is actually BETTER when excluding dropout!")
        print(
            f"   Fusion: {rmse_fusion_no_dropout * 100:.2f} cm < Vision: {rmse_vision_no_dropout * 100:.2f} cm"
        )
    else:
        print("   Even excluding dropout, fusion is worse.")
        print("   Likely cause: IMU noise/bias integration during propagation")
else:
    print("   Cannot compare due to NaN values")
print("=" * 80)
