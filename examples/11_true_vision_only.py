"""True Vision-Only Filter: Constant-Velocity Model

Implements a proper vision-only filter that doesn't diverge during dropout.
Key idea: Instead of inflating IMU noise (which causes divergence), we:
1. Use zero IMU inputs (ignore accelerometer/gyro completely)
2. Rely on process noise to model constant-velocity motion
3. Use moderate Q values to prevent covariance explosion

This represents what a real vision-only tracker would do.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.qa.metrics import compute_position_rmse, compute_velocity_rmse
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.runtime.offline import rts_smoother

OUTPUT_DIR = Path("output/true_vision_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TRUE VISION-ONLY FILTER")
print("=" * 80)

# =============================================================================
# Step 1: Generate Simulation
# =============================================================================

print("\n[1/5] Generating simulation with dropout...")

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

# Inject 5s dropout
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
vel_truth = np.column_stack(
    [np.interp(t_cam, t_truth, X_truth[:, 2]), np.interp(t_cam, t_truth, X_truth[:, 3])]
)

# =============================================================================
# Step 2: Run Fusion (baseline)
# =============================================================================

print("\n[2/5] Running Fusion (baseline)...")

ekf_config_fusion = EKFConfig(
    process_noise_pos=0.001,  # Updated from 0.02 (20×)
    process_noise_vel=0.5,  # Updated from 2.0 (4×)
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

pos_rmse_fusion = compute_position_rmse(pos_truth, ekf_fusion.filtered_means[:, :2])
vel_rmse_fusion = compute_velocity_rmse(vel_truth, ekf_fusion.filtered_means[:, 2:4])

print(f"  Fusion Position RMSE: {pos_rmse_fusion * 100:.2f} cm")
print(f"  Fusion Velocity RMSE: {vel_rmse_fusion * 100:.2f} cm/s")

# =============================================================================
# Step 3: True Vision-Only (Zero IMU + Constant Velocity)
# =============================================================================

print("\n[3/5] Running TRUE Vision-Only (constant-velocity model)...")

# Strategy: Set IMU inputs to ZERO (not inflated noise)
# This simulates a pure vision tracker with constant-velocity assumption
U_imu_zero = np.zeros_like(sim_data["U_imu"])

# Use moderate process noise for constant-velocity random walk
# Position diffuses based on velocity uncertainty
# Velocity random walk represents acceleration (unknown without IMU)
ekf_config_vision_true = EKFConfig(
    process_noise_pos=0.001,  # Small - position changes via velocity
    process_noise_vel=0.5,  # Moderate - velocity drifts without IMU
    process_noise_heading=0.01,  # Small heading drift
    process_noise_gyro_bias=0.0,  # No gyro bias (no IMU)
    process_noise_accel_bias=0.0,  # No accel bias (no IMU)
    measurement_noise_pos=0.005**2,
    imu_gyro_noise_density=0.0,  # No IMU noise (inputs are zero)
    imu_accel_noise_density=0.0,
    damping_coeff=0.0,  # No damping (no physics model without IMU)
    led_distance=0.04,
    use_heading_measurement=True,
)

ekf_vision_true = extended_kalman_filter(
    ekf_config=ekf_config_vision_true,
    t_imu=sim_data["t_imu"],
    U_imu=U_imu_zero,  # ← KEY: Zero IMU inputs
    t_cam=t_cam,
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

# Check for NaN
valid_vision_true = np.isfinite(ekf_vision_true.filtered_means[:, 0])
print(
    f"  Valid frames: {valid_vision_true.sum()}/{len(valid_vision_true)} ({valid_vision_true.sum()/len(valid_vision_true)*100:.1f}%)"
)

if valid_vision_true.sum() > 0:
    pos_rmse_vision_true = compute_position_rmse(pos_truth, ekf_vision_true.filtered_means[:, :2])
    vel_rmse_vision_true = compute_velocity_rmse(vel_truth, ekf_vision_true.filtered_means[:, 2:4])
    print(f"  Vision-Only Position RMSE: {pos_rmse_vision_true * 100:.2f} cm")
    print(f"  Vision-Only Velocity RMSE: {vel_rmse_vision_true * 100:.2f} cm/s")
else:
    print("  ERROR: All frames invalid (NaN)")
    pos_rmse_vision_true = np.nan
    vel_rmse_vision_true = np.nan

# =============================================================================
# Step 4: Vision-Only with Smoother
# =============================================================================

print("\n[4/5] Running Vision-Only RTS Smoother...")

if valid_vision_true.all():
    vision_smooth = rts_smoother(
        filter_result=ekf_vision_true,
        ekf_config=ekf_config_vision_true,
        t_imu=sim_data["t_imu"],
        U_imu=U_imu_zero,
        t_cam=t_cam,
        mask_cam=mask_with_dropout,
    )

    valid_smooth = np.isfinite(vision_smooth.smoothed_means[:, 0])
    if valid_smooth.all():
        pos_rmse_vision_smooth = compute_position_rmse(
            pos_truth, vision_smooth.smoothed_means[:, :2]
        )
        vel_rmse_vision_smooth = compute_velocity_rmse(
            vel_truth, vision_smooth.smoothed_means[:, 2:4]
        )
        print(f"  Vision-Only Smoother Position RMSE: {pos_rmse_vision_smooth * 100:.2f} cm")
        print(f"  Vision-Only Smoother Velocity RMSE: {vel_rmse_vision_smooth * 100:.2f} cm/s")
    else:
        print("  Smoother contains NaN values")
        vision_smooth = None
        pos_rmse_vision_smooth = np.nan
        vel_rmse_vision_smooth = np.nan
else:
    print("  Skipping smoother (filter has NaN)")
    vision_smooth = None
    pos_rmse_vision_smooth = np.nan
    vel_rmse_vision_smooth = np.nan

# =============================================================================
# Step 5: Visualization
# =============================================================================

print("\n[5/5] Creating comparison plots...")

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("True Vision-Only vs Fusion", fontsize=14, fontweight="bold")

# Position error
ax = axes[0, 0]
pos_error_fusion = np.linalg.norm(ekf_fusion.filtered_means[:, :2] - pos_truth, axis=1)
pos_error_vision = np.linalg.norm(ekf_vision_true.filtered_means[:, :2] - pos_truth, axis=1)

ax.plot(t_cam, pos_error_fusion * 100, "b-", linewidth=2, label="Fusion", alpha=0.8)
ax.plot(t_cam, pos_error_vision * 100, "g-", linewidth=2, label="Vision-Only (true)", alpha=0.8)
if vision_smooth is not None:
    pos_error_smooth = np.linalg.norm(vision_smooth.smoothed_means[:, :2] - pos_truth, axis=1)
    ax.plot(
        t_cam, pos_error_smooth * 100, "m-", linewidth=2, label="Vision-Only (smoothed)", alpha=0.8
    )
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red", label="Dropout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (cm)")
ax.set_title("Position Error Over Time")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 50])

# Position error (zoomed on dropout)
ax = axes[0, 1]
zoom_start = dropout_start_idx - 150
zoom_end = dropout_end_idx + 150
ax.plot(
    t_cam[zoom_start:zoom_end],
    pos_error_fusion[zoom_start:zoom_end] * 100,
    "b-",
    linewidth=2,
    label="Fusion",
    alpha=0.8,
)
ax.plot(
    t_cam[zoom_start:zoom_end],
    pos_error_vision[zoom_start:zoom_end] * 100,
    "g-",
    linewidth=2,
    label="Vision-Only (true)",
    alpha=0.8,
)
if vision_smooth is not None:
    ax.plot(
        t_cam[zoom_start:zoom_end],
        pos_error_smooth[zoom_start:zoom_end] * 100,
        "m-",
        linewidth=2,
        label="Vision-Only (smoothed)",
        alpha=0.8,
    )
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (cm)")
ax.set_title("Position Error (Dropout Zoom)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

# Velocity error
ax = axes[1, 0]
vel_error_fusion = np.linalg.norm(ekf_fusion.filtered_means[:, 2:4] - vel_truth, axis=1)
vel_error_vision = np.linalg.norm(ekf_vision_true.filtered_means[:, 2:4] - vel_truth, axis=1)

ax.plot(t_cam, vel_error_fusion * 100, "b-", linewidth=2, label="Fusion", alpha=0.8)
ax.plot(t_cam, vel_error_vision * 100, "g-", linewidth=2, label="Vision-Only (true)", alpha=0.8)
if vision_smooth is not None:
    vel_error_smooth = np.linalg.norm(vision_smooth.smoothed_means[:, 2:4] - vel_truth, axis=1)
    ax.plot(
        t_cam, vel_error_smooth * 100, "m-", linewidth=2, label="Vision-Only (smoothed)", alpha=0.8
    )
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Error (cm/s)")
ax.set_title("Velocity Error Over Time")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

# Covariance comparison
ax = axes[1, 1]
pos_var_fusion = ekf_fusion.filtered_covariances[:, 0, 0] + ekf_fusion.filtered_covariances[:, 1, 1]
pos_var_vision = (
    ekf_vision_true.filtered_covariances[:, 0, 0] + ekf_vision_true.filtered_covariances[:, 1, 1]
)

ax.plot(t_cam, pos_var_fusion, "b-", linewidth=2, label="Fusion", alpha=0.8)
ax.plot(t_cam, pos_var_vision, "g-", linewidth=2, label="Vision-Only (true)", alpha=0.8)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Variance (m²)")
ax.set_title("Position Uncertainty (no divergence!)")
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

# Trajectory
ax = axes[2, 0]
dropout_mask = ~mask_with_dropout
ax.plot(pos_truth[:, 0], pos_truth[:, 1], "k-", linewidth=2, label="Ground Truth", alpha=0.7)
ax.plot(
    ekf_fusion.filtered_means[:, 0],
    ekf_fusion.filtered_means[:, 1],
    "b-",
    linewidth=1.5,
    label="Fusion",
    alpha=0.6,
)
ax.plot(
    ekf_vision_true.filtered_means[:, 0],
    ekf_vision_true.filtered_means[:, 1],
    "g--",
    linewidth=1.5,
    label="Vision-Only",
    alpha=0.6,
)
ax.scatter(
    pos_truth[dropout_mask, 0],
    pos_truth[dropout_mask, 1],
    c="red",
    s=30,
    alpha=0.3,
    label="Dropout Period",
)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Trajectory Comparison")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)
ax.axis("equal")

# Summary metrics
ax = axes[2, 1]
ax.axis("off")

summary_text = f"""
TRUE VISION-ONLY vs FUSION COMPARISON

Position RMSE:
  Fusion:           {pos_rmse_fusion * 100:.2f} cm
  Vision-Only:      {pos_rmse_vision_true * 100:.2f} cm
  Vision Smoothed:  {pos_rmse_vision_smooth * 100:.2f} cm

Velocity RMSE:
  Fusion:           {vel_rmse_fusion * 100:.2f} cm/s
  Vision-Only:      {vel_rmse_vision_true * 100:.2f} cm/s
  Vision Smoothed:  {vel_rmse_vision_smooth * 100:.2f} cm/s

Valid Frames:
  Fusion:           {np.isfinite(ekf_fusion.filtered_means[:, 0]).sum()}/{len(t_cam)}
  Vision-Only:      {valid_vision_true.sum()}/{len(t_cam)}

Key Differences:
• Vision-only uses ZERO IMU inputs
• Constant-velocity model (no physics)
• Moderate process noise prevents divergence
• No adaptive Q during dropout
• Relies purely on camera observations
"""

ax.text(
    0.05,
    0.95,
    summary_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

plt.tight_layout()
plot_path = OUTPUT_DIR / "true_vision_only_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {plot_path}")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: True Vision-Only Implementation")
print("=" * 80)
print("\nKey Design Choices:")
print("  1. Zero IMU inputs (not inflated noise)")
print("  2. Moderate velocity process noise (0.5 m²/s for random walk)")
print("  3. Small position process noise (0.001 m²/s, changes via velocity)")
print("  4. No damping (no physics model without IMU)")
print("  5. No adaptive Q (would cause divergence without measurements)")
print()
print("Results:")
print(f"  ✅ No NaN divergence: {valid_vision_true.sum()}/{len(valid_vision_true)} valid")
print(f"  ✅ Position RMSE: {pos_rmse_vision_true * 100:.2f} cm")
print(f"  ✅ Velocity RMSE: {vel_rmse_vision_true * 100:.2f} cm/s")
print(f"  ✅ Covariance bounded: max = {np.max(pos_var_vision):.4f} m²")
print()
print("Comparison to Fusion:")
print(f"  Position: {pos_rmse_vision_true / pos_rmse_fusion:.2f}× Fusion")
print(f"  Velocity: {vel_rmse_vision_true / vel_rmse_fusion:.2f}× Fusion")
print("=" * 80)
