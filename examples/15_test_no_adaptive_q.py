"""Test Impact of Disabling Adaptive Q

Compares fusion performance with and without adaptive Q during dropout.
Goal: See if disabling adaptive Q prevents covariance explosion and improves smoother.
"""

import numpy as np
from pathlib import Path

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.runtime.offline import rts_smoother
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.qa.metrics import compute_position_rmse

OUTPUT_DIR = Path("output/no_adaptive_q")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TESTING: ADAPTIVE Q ON vs OFF")
print("=" * 80)

# =============================================================================
# Step 1: Generate Simulation
# =============================================================================

print("\n[1/4] Generating 60s simulation with 5s dropout...")

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

print(f"  Duration: {config.duration_s}s")
print(f"  Dropout: {dropout_start_t}s - {dropout_end_t}s")


# Ground truth
def interpolate_truth_to_cam(sim_data):
    X_truth = sim_data["X_truth"]
    return np.column_stack(
        [
            np.interp(sim_data["t_cam_exp"], sim_data["t_imu"], X_truth[:, 0]),
            np.interp(sim_data["t_cam_exp"], sim_data["t_imu"], X_truth[:, 1]),
        ]
    )


pos_true_interp = interpolate_truth_to_cam(sim_data)

# =============================================================================
# Step 2: Run WITH Adaptive Q
# =============================================================================

print("\n[2/4] Running fusion WITH adaptive Q (10× multiplier)...")

ekf_config_with_adaptive = EKFConfig(
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
    adaptive_q_during_dropout=True,  # ON
    dropout_q_pos_multiplier=10.0,
    dropout_q_vel_multiplier=10.0,
    dropout_q_bias_multiplier=0.1,
)

ekf_with_adaptive = extended_kalman_filter(
    ekf_config=ekf_config_with_adaptive,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=sim_data["t_cam_exp"],
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

rmse_filter_with = compute_position_rmse(
    positions_true=pos_true_interp,
    positions_est=ekf_with_adaptive.filtered_means[:, :2],
)

smoothed_with = rts_smoother(
    filter_result=ekf_with_adaptive,
    ekf_config=ekf_config_with_adaptive,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=sim_data["t_cam_exp"],
    mask_cam=mask_with_dropout,
)

rmse_smoother_with = compute_position_rmse(
    positions_true=pos_true_interp,
    positions_est=smoothed_with.smoothed_means[:, :2],
)

# Covariance stats
P_trace_with = np.array([np.trace(P[:2, :2]) for P in ekf_with_adaptive.filtered_covariances])
dropout_mask = (sim_data["t_cam_exp"] >= dropout_start_t) & (sim_data["t_cam_exp"] <= dropout_end_t)
max_P_with = np.max(P_trace_with[dropout_mask])

print(f"  Filter RMSE: {rmse_filter_with*100:.2f} cm")
print(f"  Smoother RMSE: {rmse_smoother_with*100:.2f} cm")
print(f"  Max covariance during dropout: {max_P_with*1e4:.2f} cm²")

# =============================================================================
# Step 3: Run WITHOUT Adaptive Q
# =============================================================================

print("\n[3/4] Running fusion WITHOUT adaptive Q (disabled)...")

ekf_config_no_adaptive = EKFConfig(
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
    adaptive_q_during_dropout=False,  # OFF
)

ekf_no_adaptive = extended_kalman_filter(
    ekf_config=ekf_config_no_adaptive,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=sim_data["t_cam_exp"],
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

rmse_filter_no = compute_position_rmse(
    positions_true=pos_true_interp,
    positions_est=ekf_no_adaptive.filtered_means[:, :2],
)

smoothed_no = rts_smoother(
    filter_result=ekf_no_adaptive,
    ekf_config=ekf_config_no_adaptive,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=sim_data["t_cam_exp"],
    mask_cam=mask_with_dropout,
)

rmse_smoother_no = compute_position_rmse(
    positions_true=pos_true_interp,
    positions_est=smoothed_no.smoothed_means[:, :2],
)

# Covariance stats
P_trace_no = np.array([np.trace(P[:2, :2]) for P in ekf_no_adaptive.filtered_covariances])
max_P_no = np.max(P_trace_no[dropout_mask])

print(f"  Filter RMSE: {rmse_filter_no*100:.2f} cm")
print(f"  Smoother RMSE: {rmse_smoother_no*100:.2f} cm")
print(f"  Max covariance during dropout: {max_P_no*1e4:.2f} cm²")

# =============================================================================
# Step 4: Compare Results
# =============================================================================

print("\n[4/4] Analyzing impact...")

# Position errors during dropout
pos_err_with = np.linalg.norm(pos_true_interp - ekf_with_adaptive.filtered_means[:, :2], axis=1)
pos_err_no = np.linalg.norm(pos_true_interp - ekf_no_adaptive.filtered_means[:, :2], axis=1)

mean_err_dropout_with = np.mean(pos_err_with[dropout_mask])
mean_err_dropout_no = np.mean(pos_err_no[dropout_mask])

print("\n" + "=" * 80)
print("SUMMARY: ADAPTIVE Q IMPACT")
print("=" * 80)

print("\n📊 Performance Comparison:")
print("\n  Filter RMSE:")
print(f"    WITH adaptive Q:    {rmse_filter_with*100:.2f} cm")
print(f"    WITHOUT adaptive Q: {rmse_filter_no*100:.2f} cm")
print(f"    Difference:         {(rmse_filter_no - rmse_filter_with)*100:+.2f} cm")

print("\n  Smoother RMSE:")
print(f"    WITH adaptive Q:    {rmse_smoother_with*100:.2f} cm")
print(f"    WITHOUT adaptive Q: {rmse_smoother_no*100:.2f} cm")
print(f"    Difference:         {(rmse_smoother_no - rmse_smoother_with)*100:+.2f} cm")

print("\n📈 Covariance During Dropout:")
print(f"    WITH adaptive Q:    {max_P_with*1e4:,.2f} cm²")
print(f"    WITHOUT adaptive Q: {max_P_no*1e4:,.2f} cm²")
print(f"    Reduction:          {max_P_with/max_P_no:.1f}× smaller without adaptive Q")

print("\n🎯 Drift During Dropout:")
print(f"    WITH adaptive Q:    {mean_err_dropout_with*100:.2f} cm")
print(f"    WITHOUT adaptive Q: {mean_err_dropout_no*100:.2f} cm")
print(f"    Difference:         {(mean_err_dropout_no - mean_err_dropout_with)*100:+.2f} cm")

print("\n💡 Key Findings:")

# Check if smoother improved
if rmse_smoother_no < rmse_smoother_with:
    improvement_pct = (rmse_smoother_with - rmse_smoother_no) / rmse_smoother_with * 100
    print(f"  ✅ Smoother IMPROVED by {improvement_pct:.1f}% without adaptive Q!")
else:
    degradation_pct = (rmse_smoother_no - rmse_smoother_with) / rmse_smoother_with * 100
    print(f"  ❌ Smoother DEGRADED by {degradation_pct:.1f}% without adaptive Q")

# Check covariance explosion
if max_P_no < max_P_with * 0.1:
    print("  ✅ Covariance explosion PREVENTED (10× reduction)")
elif max_P_no < max_P_with * 0.5:
    print("  ⚠️  Covariance growth REDUCED but still significant")
else:
    print("  ❌ Covariance still explodes even without adaptive Q")

# Check dropout drift
if mean_err_dropout_no > mean_err_dropout_with * 1.5:
    print("  ⚠️  WARNING: Dropout drift increased significantly without adaptive Q")
    print("     Filter may be over-confident during dropout (underestimates uncertainty)")
elif mean_err_dropout_no > mean_err_dropout_with * 1.2:
    print("  ⚠️  Dropout drift slightly worse without adaptive Q")
else:
    print("  ✅ Dropout drift comparable or better")

print("\n📋 Recommendation:")
if rmse_smoother_no < rmse_smoother_with and mean_err_dropout_no < mean_err_dropout_with * 1.3:
    print("  ✅ DISABLE adaptive Q - improves smoother without harming filter")
    print("  ✅ Update production config: adaptive_q_during_dropout=False")
elif rmse_smoother_no < rmse_smoother_with:
    print("  ⚠️  Disabling adaptive Q improves smoother but increases dropout drift")
    print("  💡 Consider compromise: reduce multiplier to 5× instead of disabling")
else:
    print("  ❌ Keep adaptive Q - disabling doesn't help")
    print("  💡 Problem lies elsewhere (bias drift, model mismatch, etc.)")

print("=" * 80)
