"""Create Diagnostic Videos: Fusion, Vision-Only, and IMU-Only

Generates side-by-side comparison videos showing filter behavior
during camera dropout periods.
"""

import numpy as np
from pathlib import Path

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.viz.video import create_diagnostic_video

OUTPUT_DIR = Path("output/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CREATING DIAGNOSTIC VIDEOS")
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

# Update sim_data for video generation
sim_data_dropout = dict(sim_data)
sim_data_dropout["mask_cam"] = mask_with_dropout
sim_data_dropout["Z_cam_led1"] = Z_cam_led1_dropout
sim_data_dropout["Z_cam_led2"] = Z_cam_led2_dropout

print(f"  Simulation: {config.duration_s}s")
print(f"  Dropout: {dropout_start_t}s - {dropout_end_t}s")
print(f"  Camera frames: {len(sim_data['t_cam_exp'])}")

# =============================================================================
# Step 2: Run Fusion Filter
# =============================================================================

print("\n[2/4] Running Fusion Filter...")

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
    t_cam=sim_data_dropout["t_cam_exp"],
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

print("  Fusion filter complete")

# =============================================================================
# Step 3: Run Vision-Only Filter
# =============================================================================

print("\n[3/4] Running Vision-Only Filter...")

# Use zero IMU inputs for true vision-only
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
    t_cam=sim_data_dropout["t_cam_exp"],
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

print("  Vision-only filter complete")

# =============================================================================
# Step 4: Create Videos
# =============================================================================

print("\n[4/4] Rendering videos...")

# Video 1: Fusion (Vision + IMU)
print("\n  [4a] Rendering Fusion video...")
video_fusion = OUTPUT_DIR / "fusion_with_dropout.mp4"
create_diagnostic_video(
    sim_data=sim_data_dropout,
    output_path=video_fusion,
    filter_results=ekf_fusion,
    fps=30,
    speedup=2.0,  # 2x speed for faster viewing
    time_window_s=3.0,
    trail_length_s=2.0,
    dpi=120,
)
print(f"    ✅ Saved: {video_fusion}")

# Video 2: Vision-Only
print("\n  [4b] Rendering Vision-Only video...")
# For vision-only video, we need to use the zero-IMU sim data
sim_data_vision = dict(sim_data_dropout)
sim_data_vision["U_imu"] = U_imu_zero  # Show zero IMU in plots

video_vision = OUTPUT_DIR / "vision_only_with_dropout.mp4"
create_diagnostic_video(
    sim_data=sim_data_vision,
    output_path=video_vision,
    filter_results=ekf_vision,
    fps=30,
    speedup=2.0,
    time_window_s=3.0,
    trail_length_s=2.0,
    dpi=120,
)
print(f"    ✅ Saved: {video_vision}")

# Video 3: Ground Truth Only (no filter, for reference)
print("\n  [4c] Rendering Ground Truth video...")
video_truth = OUTPUT_DIR / "ground_truth_with_dropout.mp4"
create_diagnostic_video(
    sim_data=sim_data_dropout,
    output_path=video_truth,
    filter_results=None,  # No filter overlay
    fps=30,
    speedup=2.0,
    time_window_s=3.0,
    trail_length_s=2.0,
    dpi=120,
)
print(f"    ✅ Saved: {video_truth}")

print("\n" + "=" * 80)
print("VIDEO GENERATION COMPLETE")
print("=" * 80)
print(f"\nGenerated 3 videos in: {OUTPUT_DIR}/")
print(f"  1. {video_fusion.name} - Fusion (Vision + IMU)")
print(f"  2. {video_vision.name} - Vision-Only (constant-velocity)")
print(f"  3. {video_truth.name} - Ground Truth (reference)")
print()
print("Video Details:")
print("  Duration: 30 seconds (60s sim @ 2× speed)")
print("  Frame rate: 30 fps")
print("  Resolution: 1200×900 @ 120 DPI")
print()
print("What to look for:")
print("  • Dropout period: RED shaded region (t=12.5-15s in video)")
print("  • Fusion: Small drift during dropout, quick recovery")
print("  • Vision-only: Larger drift during dropout, slower recovery")
print("  • Camera confidence: Goes to zero during dropout")
print("  • Position uncertainty ellipse: Grows during dropout")
print("=" * 80)
