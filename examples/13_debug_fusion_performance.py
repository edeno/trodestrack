"""Debug Fusion Filter Underperformance

Investigates why the fusion filter is performing worse than expected,
particularly compared to vision-only after smoothing.
"""

import numpy as np
from pathlib import Path

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.runtime.offline import rts_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.qa.metrics import compute_position_rmse

OUTPUT_DIR = Path("output/fusion_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def interpolate_truth_to_cam(sim_data):
    """Interpolate ground truth to camera times."""
    X_truth = sim_data["X_truth"]
    return np.column_stack(
        [
            np.interp(sim_data["t_cam_exp"], sim_data["t_imu"], X_truth[:, 0]),
            np.interp(sim_data["t_cam_exp"], sim_data["t_imu"], X_truth[:, 1]),
        ]
    )


print("=" * 80)
print("DEBUGGING FUSION FILTER PERFORMANCE")
print("=" * 80)

# =============================================================================
# Step 1: Generate Simulation
# =============================================================================

print("\n[1/5] Generating 60s simulation with 5s dropout...")

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

# Inject 5s dropout from t=25s to t=30s
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

print(f"  Simulation: {config.duration_s}s")
print(f"  Dropout: {dropout_start_t}s - {dropout_end_t}s")

# =============================================================================
# Step 2: Test Multiple Fusion Configurations
# =============================================================================

print("\n[2/5] Testing multiple fusion configurations...")

configs = {
    "Production (NEW)": EKFConfig(
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
        adaptive_q_during_dropout=False,  # DISABLED - prevents covariance explosion
        dropout_q_pos_multiplier=10.0,
        dropout_q_vel_multiplier=10.0,
        dropout_q_bias_multiplier=0.1,
    ),
    "Production (OLD)": EKFConfig(
        process_noise_pos=0.02,  # Old value
        process_noise_vel=2.0,  # Old value
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
    ),
    "No Adaptive Q": EKFConfig(
        process_noise_pos=0.001,  # Use new tuning
        process_noise_vel=0.5,
        process_noise_heading=0.02,
        process_noise_gyro_bias=2e-6,
        process_noise_accel_bias=2e-4,
        measurement_noise_pos=0.005**2,
        imu_gyro_noise_density=0.001,
        imu_accel_noise_density=0.05,
        damping_coeff=0.4,
        led_distance=0.04,
        use_heading_measurement=True,
        adaptive_q_during_dropout=False,  # Disable adaptive Q
    ),
    "Lower Adaptive Multipliers": EKFConfig(
        process_noise_pos=0.001,
        process_noise_vel=0.5,
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
        dropout_q_pos_multiplier=5.0,  # Reduced from 10×
        dropout_q_vel_multiplier=5.0,  # Reduced from 10×
        dropout_q_bias_multiplier=0.5,  # Increased from 0.1×
    ),
}

results = {}

# Get ground truth at camera times (once for all configs)
pos_true_interp = interpolate_truth_to_cam(sim_data)

for name, ekf_config in configs.items():
    print(f"\n  Testing: {name}")
    ekf_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=sim_data["t_imu"],
        U_imu=sim_data["U_imu"],
        t_cam=sim_data["t_cam_exp"],
        Z_cam_led1=Z_cam_led1_dropout,
        Z_cam_led2=Z_cam_led2_dropout,
        mask_cam=mask_with_dropout,
    )

    # Compute RMSE (filter)
    rmse_filter = compute_position_rmse(
        positions_true=pos_true_interp,
        positions_est=ekf_result.filtered_means[:, :2],
    )

    # Compute RMSE (smoother)
    smoothed = rts_smoother(
        filter_result=ekf_result,
        ekf_config=ekf_config,
        t_imu=sim_data["t_imu"],
        U_imu=sim_data["U_imu"],
        t_cam=sim_data["t_cam_exp"],
        mask_cam=mask_with_dropout,
    )

    # Check if smoother produced valid results
    n_valid_smoother = np.isfinite(smoothed.smoothed_means[:, 0]).sum()
    if n_valid_smoother > 0:
        rmse_smoother = compute_position_rmse(
            positions_true=pos_true_interp,
            positions_est=smoothed.smoothed_means[:, :2],
        )
    else:
        rmse_smoother = np.nan

    # Check for NaN
    n_valid_filter = np.isfinite(ekf_result.filtered_means[:, 0]).sum()
    n_valid_smoother = np.isfinite(smoothed.smoothed_means[:, 0]).sum()
    n_total = len(ekf_result.filtered_means)

    results[name] = {
        "rmse_filter": rmse_filter,
        "rmse_smoother": rmse_smoother,
        "valid_filter_pct": 100 * n_valid_filter / n_total,
        "valid_smoother_pct": 100 * n_valid_smoother / n_total,
        "ekf_result": ekf_result,
        "smoothed": smoothed,
    }

    print(f"    Filter RMSE: {rmse_filter*100:.2f} cm ({n_valid_filter}/{n_total} valid)")
    print(f"    Smoother RMSE: {rmse_smoother*100:.2f} cm ({n_valid_smoother}/{n_total} valid)")

# =============================================================================
# Step 3: Compare with True Vision-Only
# =============================================================================

print("\n[3/5] Running true vision-only baseline...")

U_imu_zero = np.zeros_like(sim_data["U_imu"])

ekf_config_vision = EKFConfig(
    process_noise_pos=0.001,
    process_noise_vel=0.5,
    process_noise_heading=0.01,
    process_noise_gyro_bias=0.0,
    process_noise_accel_bias=0.0,
    measurement_noise_pos=0.005**2,
    imu_gyro_noise_density=0.0,
    imu_accel_noise_density=0.0,
    damping_coeff=0.0,
    led_distance=0.04,
    use_heading_measurement=True,
)

ekf_vision = extended_kalman_filter(
    ekf_config=ekf_config_vision,
    t_imu=sim_data["t_imu"],
    U_imu=U_imu_zero,
    t_cam=sim_data["t_cam_exp"],
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

rmse_vision_filter = compute_position_rmse(
    positions_true=pos_true_interp,
    positions_est=ekf_vision.filtered_means[:, :2],
)

smoothed_vision = rts_smoother(
    filter_result=ekf_vision,
    ekf_config=ekf_config_vision,
    t_imu=sim_data["t_imu"],
    U_imu=U_imu_zero,
    t_cam=sim_data["t_cam_exp"],
    mask_cam=mask_with_dropout,
)

rmse_vision_smoother = compute_position_rmse(
    positions_true=pos_true_interp,
    positions_est=smoothed_vision.smoothed_means[:, :2],
)

n_valid_vision_filter = np.isfinite(ekf_vision.filtered_means[:, 0]).sum()
n_valid_vision_smoother = np.isfinite(smoothed_vision.smoothed_means[:, 0]).sum()

print(
    f"  Vision-Only Filter RMSE: {rmse_vision_filter*100:.2f} cm ({n_valid_vision_filter}/{n_total} valid)"
)
print(
    f"  Vision-Only Smoother RMSE: {rmse_vision_smoother*100:.2f} cm ({n_valid_vision_smoother}/{n_total} valid)"
)

# =============================================================================
# Step 4: Analyze Error Statistics
# =============================================================================

print("\n[4/5] Analyzing error statistics...")

prod_result = results["Production (NEW)"]["ekf_result"]

# Compute position errors over time
pos_errors = pos_true_interp - prod_result.filtered_means[:, :2]
pos_errors_norm = np.linalg.norm(pos_errors, axis=1)

# Before dropout
pre_dropout_mask = sim_data["t_cam_exp"] < dropout_start_t
mean_error_pre = np.mean(pos_errors_norm[pre_dropout_mask])
std_error_pre = np.std(pos_errors_norm[pre_dropout_mask])

# During dropout
dropout_mask = (sim_data["t_cam_exp"] >= dropout_start_t) & (sim_data["t_cam_exp"] <= dropout_end_t)
mean_error_dropout = np.mean(pos_errors_norm[dropout_mask])
max_error_dropout = np.max(pos_errors_norm[dropout_mask])

# After dropout
post_dropout_mask = sim_data["t_cam_exp"] > dropout_end_t
mean_error_post = np.mean(pos_errors_norm[post_dropout_mask])

print("\n  Position Error Statistics (fusion):")
print(f"    Before dropout: {mean_error_pre*100:.2f} ± {std_error_pre*100:.2f} cm")
print(f"    During dropout: {mean_error_dropout*100:.2f} cm (max: {max_error_dropout*100:.2f} cm)")
print(f"    After dropout: {mean_error_post*100:.2f} cm")

# =============================================================================
# Step 5: Analyze Covariance Evolution
# =============================================================================

print("\n[5/5] Analyzing covariance evolution...")

P_trace = np.array(
    [np.trace(P[:2, :2]) for P in prod_result.filtered_covariances]
)  # Position covariance trace
P_trace_vision = np.array([np.trace(P[:2, :2]) for P in ekf_vision.filtered_covariances])

# Before dropout
pre_dropout_idx = sim_data["t_cam_exp"] < dropout_start_t
mean_P_pre = np.mean(P_trace[pre_dropout_idx])
mean_P_vision_pre = np.mean(P_trace_vision[pre_dropout_idx])

# During dropout
dropout_mask = (sim_data["t_cam_exp"] >= dropout_start_t) & (sim_data["t_cam_exp"] <= dropout_end_t)
max_P_dropout = np.max(P_trace[dropout_mask])
max_P_vision_dropout = np.max(P_trace_vision[dropout_mask])

# After dropout
post_dropout_mask = sim_data["t_cam_exp"] > dropout_end_t
mean_P_post = np.mean(P_trace[post_dropout_mask])
mean_P_vision_post = np.mean(P_trace_vision[post_dropout_mask])

print("\n  Fusion Covariance (tr(P_pos)):")
print(f"    Before dropout: {mean_P_pre*1e4:.2f} cm²")
print(f"    Peak during dropout: {max_P_dropout*1e4:.2f} cm²")
print(f"    After dropout: {mean_P_post*1e4:.2f} cm²")

print("\n  Vision-Only Covariance (tr(P_pos)):")
print(f"    Before dropout: {mean_P_vision_pre*1e4:.2f} cm²")
print(f"    Peak during dropout: {max_P_vision_dropout*1e4:.2f} cm²")
print(f"    After dropout: {mean_P_vision_post*1e4:.2f} cm²")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n📊 Configuration Comparison (Filter RMSE):")
for name, res in results.items():
    print(f"  {name:25s}: {res['rmse_filter']*100:6.2f} cm ({res['valid_filter_pct']:.0f}% valid)")
print(
    f"  {'Vision-Only (baseline)':25s}: {rmse_vision_filter*100:6.2f} cm ({100*n_valid_vision_filter/n_total:.0f}% valid)"
)

print("\n📊 Configuration Comparison (Smoother RMSE):")
for name, res in results.items():
    print(
        f"  {name:25s}: {res['rmse_smoother']*100:6.2f} cm ({res['valid_smoother_pct']:.0f}% valid)"
    )
print(
    f"  {'Vision-Only (baseline)':25s}: {rmse_vision_smoother*100:6.2f} cm ({100*n_valid_vision_smoother/n_total:.0f}% valid)"
)

print("\n🔍 Key Findings:")
best_config = min(results.items(), key=lambda x: x[1]["rmse_filter"])
print(f"  • Best fusion config: {best_config[0]} ({best_config[1]['rmse_filter']*100:.2f} cm)")

if rmse_vision_smoother < best_config[1]["rmse_smoother"]:
    print("  • Vision-only smoother STILL beats best fusion smoother!")
    print(
        f"    Vision: {rmse_vision_smoother*100:.2f} cm vs Fusion: {best_config[1]['rmse_smoother']*100:.2f} cm"
    )
    print("  • Possible reasons:")
    print("    1. IMU biases not converging properly")
    print("    2. Process noise tuning issue (Q too high)")
    print("    3. IMU drift corrupting state during dropout")
else:
    print("  • Fusion smoother now beats vision-only!")
    print(
        f"    Fusion: {best_config[1]['rmse_smoother']*100:.2f} cm vs Vision: {rmse_vision_smoother*100:.2f} cm"
    )

print("=" * 80)
