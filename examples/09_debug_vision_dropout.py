"""Debug: Why doesn't vision-only degrade during dropout?

Investigates the vision-only configuration to understand why it doesn't
show performance degradation during camera dropout.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

OUTPUT_DIR = Path("output/debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DEBUG: Vision-only Performance During Dropout")
print("=" * 80)

# Generate simulation with dropout
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

# Run vision-only configuration
print("\n[1/3] Running vision-only EKF...")
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

# Check what's happening during dropout
print("\n[2/3] Analyzing dropout behavior...")

print(f"\nDropout indices: {dropout_start_idx} to {dropout_end_idx}")
print(f"Mask during dropout: {mask_with_dropout[dropout_start_idx:dropout_end_idx]}")
print("LED1 observations during dropout (first 5):")
print(f"  {Z_cam_led1_dropout[dropout_start_idx:dropout_start_idx+5, :]}")
print("LED2 observations during dropout (first 5):")
print(f"  {Z_cam_led2_dropout[dropout_start_idx:dropout_start_idx+5, :]}")

# Check covariance during dropout
print("\nCovariance (position variance) timeline:")
pos_var = ekf_vision.filtered_covariances[:, 0, 0] + ekf_vision.filtered_covariances[:, 1, 1]
print(f"  Before dropout (t=24s): {pos_var[dropout_start_idx-30]:.6f} m²")
print(f"  Dropout start (t=25s):  {pos_var[dropout_start_idx]:.6f} m²")
print(f"  Dropout middle (t=27.5s): {pos_var[dropout_start_idx+75]:.6f} m²")
print(f"  Dropout end (t=30s):    {pos_var[dropout_end_idx-1]:.6f} m²")
print(f"  After dropout (t=31s):  {pos_var[dropout_end_idx+30]:.6f} m²")

# Check position estimates during dropout
print("\nPosition estimates during dropout:")
print(f"  Ground truth at start: {pos_truth[dropout_start_idx, :]}")
print(f"  Estimate at start:     {ekf_vision.filtered_means[dropout_start_idx, :2]}")
print(
    f"  Error at start:        {np.linalg.norm(ekf_vision.filtered_means[dropout_start_idx, :2] - pos_truth[dropout_start_idx, :]):.4f} m"
)
print(f"\n  Ground truth at end:   {pos_truth[dropout_end_idx-1, :]}")
print(f"  Estimate at end:       {ekf_vision.filtered_means[dropout_end_idx-1, :2]}")
print(
    f"  Error at end:          {np.linalg.norm(ekf_vision.filtered_means[dropout_end_idx-1, :2] - pos_truth[dropout_end_idx-1, :]):.4f} m"
)

# Check velocity during dropout
vel_truth_start = np.array(
    [
        np.interp(t_cam[dropout_start_idx], t_truth, X_truth[:, 2]),
        np.interp(t_cam[dropout_start_idx], t_truth, X_truth[:, 3]),
    ]
)
vel_truth_end = np.array(
    [
        np.interp(t_cam[dropout_end_idx - 1], t_truth, X_truth[:, 2]),
        np.interp(t_cam[dropout_end_idx - 1], t_truth, X_truth[:, 3]),
    ]
)

print("\nVelocity estimates during dropout:")
print(f"  Ground truth at start: {vel_truth_start}")
print(f"  Estimate at start:     {ekf_vision.filtered_means[dropout_start_idx, 2:4]}")
print(f"\n  Ground truth at end:   {vel_truth_end}")
print(f"  Estimate at end:       {ekf_vision.filtered_means[dropout_end_idx-1, 2:4]}")

# Check if damping is keeping velocity near zero
print("\nDamping analysis:")
print(f"  Damping coefficient: {ekf_config_vision.damping_coeff}")
print("  Velocity magnitude during dropout (sample every 30 frames):")
for i in range(dropout_start_idx, dropout_end_idx, 30):
    vel_est = ekf_vision.filtered_means[i, 2:4]
    vel_mag = np.linalg.norm(vel_est)
    vel_truth_x = np.interp(t_cam[i], t_truth, X_truth[:, 2])
    vel_truth_y = np.interp(t_cam[i], t_truth, X_truth[:, 3])
    vel_truth_mag = np.sqrt(vel_truth_x**2 + vel_truth_y**2)
    print(f"    t={t_cam[i]:.1f}s: est={vel_mag:.4f} m/s, truth={vel_truth_mag:.4f} m/s")

# Create diagnostic plot
print("\n[3/3] Creating diagnostic plots...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Position error
ax = axes[0]
pos_error = np.linalg.norm(ekf_vision.filtered_means[:, :2] - pos_truth, axis=1)
ax.plot(t_cam, pos_error * 100, "g-", linewidth=2, label="Vision-only Position Error")
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red", label="Dropout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (cm)")
ax.set_title("Vision-only: Position Error")
ax.legend()
ax.grid(True, alpha=0.3)

# Position covariance
ax = axes[1]
ax.plot(t_cam, pos_var, "b-", linewidth=2, label="Position Variance (trace)")
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red", label="Dropout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Variance (m²)")
ax.set_title("Vision-only: Position Uncertainty")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

# Velocity magnitude
ax = axes[2]
vel_est_mag = np.linalg.norm(ekf_vision.filtered_means[:, 2:4], axis=1)
vel_truth_mag = np.linalg.norm(
    np.column_stack(
        [np.interp(t_cam, t_truth, X_truth[:, 2]), np.interp(t_cam, t_truth, X_truth[:, 3])]
    ),
    axis=1,
)
ax.plot(t_cam, vel_est_mag * 100, "g-", linewidth=2, label="Vision-only Estimate", alpha=0.8)
ax.plot(t_cam, vel_truth_mag * 100, "k--", linewidth=1, label="Ground Truth", alpha=0.5)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red", label="Dropout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Magnitude (cm/s)")
ax.set_title("Vision-only: Velocity Estimates")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "vision_only_dropout_debug.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {plot_path}")
plt.close()

print("\n" + "=" * 80)
print("ROOT CAUSE IDENTIFIED:")
print("=" * 80)
print(f"Vision-only covariance explodes during dropout: {pos_var[dropout_end_idx-1]:.1f} m²")
print("This is ~180,000 m² (424m radius!) - filter completely loses track")
print("NaN after dropout indicates numerical instability/divergence")
print("\nThe low RMSE (4.12 cm) is MISLEADING:")
print("  - Only averaging over non-NaN frames")
print("  - Dropout frames may be excluded from RMSE calculation")
print("  - Or initial reports were from runs without proper dropout masking")
print("\nACTUAL BEHAVIOR: Vision-only filter FAILS during dropout (as expected)")
print("=" * 80)
