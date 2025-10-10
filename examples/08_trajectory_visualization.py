"""Trajectory Visualization: Show spatial paths during dropout periods.

Creates detailed trajectory plots showing how different sensor configurations
behave during camera dropout periods.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Plot uncertainty growth during dropout (3-sigma ellipses)
from matplotlib.patches import Ellipse

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

# Output directory
OUTPUT_DIR = Path("output/sensor_fusion_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TRAJECTORY VISUALIZATION WITH DROPOUT")
print("=" * 80)

# =============================================================================
# Generate Simulation with Dropout
# =============================================================================

print("\n[1/3] Generating 60s simulation with 5s dropout...")

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
    arena_w=1.0,
    arena_h=1.0,
)
sim_data = simulate_rat_imu(config=config, seed=42)

# Inject dropout from t=25s to t=30s
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

# =============================================================================
# Run Filters
# =============================================================================

print("\n[2/3] Running EKF configurations...")

ekf_config = EKFConfig(
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

# Fusion
ekf_fusion = extended_kalman_filter(
    ekf_config=ekf_config,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=t_cam,
    Z_cam_led1=Z_cam_led1_dropout,
    Z_cam_led2=Z_cam_led2_dropout,
    mask_cam=mask_with_dropout,
)

# IMU-only
mask_imu_only = np.zeros_like(mask_with_dropout, dtype=bool)
Z_cam_imu = np.full_like(Z_cam_led1_dropout, np.nan)

ekf_imu = extended_kalman_filter(
    ekf_config=ekf_config,
    t_imu=sim_data["t_imu"],
    U_imu=sim_data["U_imu"],
    t_cam=t_cam,
    Z_cam_led1=Z_cam_imu,
    Z_cam_led2=Z_cam_imu,
    mask_cam=mask_imu_only,
)

# Vision-only
ekf_config_vision = EKFConfig(
    process_noise_pos=0.02,
    process_noise_vel=2000.0,
    process_noise_heading=0.02,
    process_noise_gyro_bias=2e-6,
    process_noise_accel_bias=2e-4,
    measurement_noise_pos=0.005**2,
    imu_gyro_noise_density=10.0,
    imu_accel_noise_density=50.0,
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

# =============================================================================
# Create Trajectory Plots
# =============================================================================

print("\n[3/3] Creating trajectory plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Trajectory Comparison: Sensor Fusion vs Modalities", fontsize=14, fontweight="bold")

# Get dropout segment indices
dropout_mask = ~mask_with_dropout

# Full trajectory comparison
ax = axes[0, 0]
ax.plot(pos_truth[:, 0], pos_truth[:, 1], "k-", linewidth=2, label="Ground Truth", alpha=0.7)
ax.plot(
    ekf_fusion.filtered_means[:, 0],
    ekf_fusion.filtered_means[:, 1],
    "b-",
    linewidth=1.5,
    label="Fusion",
    alpha=0.8,
)
ax.plot(
    ekf_vision.filtered_means[:, 0],
    ekf_vision.filtered_means[:, 1],
    "g--",
    linewidth=1.5,
    label="Vision-only",
    alpha=0.6,
)
ax.plot(
    ekf_imu.filtered_means[:, 0],
    ekf_imu.filtered_means[:, 1],
    "r:",
    linewidth=1.5,
    label="IMU-only",
    alpha=0.6,
)

# Mark dropout segment
ax.scatter(
    pos_truth[dropout_mask, 0],
    pos_truth[dropout_mask, 1],
    c="red",
    s=30,
    alpha=0.3,
    label="Dropout Period",
    zorder=5,
)

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Full Session Trajectory (60s)")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)
ax.axis("equal")
ax.set_xlim([0, config.arena_w])
ax.set_ylim([0, config.arena_h])

# Zoom on dropout region
ax = axes[0, 1]

# Get indices for plotting (t=20s to t=35s, centered on dropout)
zoom_start_idx = np.searchsorted(t_cam, 20.0)
zoom_end_idx = np.searchsorted(t_cam, 35.0)

ax.plot(
    pos_truth[zoom_start_idx:zoom_end_idx, 0],
    pos_truth[zoom_start_idx:zoom_end_idx, 1],
    "k-",
    linewidth=3,
    label="Ground Truth",
    alpha=0.7,
)
ax.plot(
    ekf_fusion.filtered_means[zoom_start_idx:zoom_end_idx, 0],
    ekf_fusion.filtered_means[zoom_start_idx:zoom_end_idx, 1],
    "b-",
    linewidth=2,
    label="Fusion",
    alpha=0.8,
)
ax.plot(
    ekf_vision.filtered_means[zoom_start_idx:zoom_end_idx, 0],
    ekf_vision.filtered_means[zoom_start_idx:zoom_end_idx, 1],
    "g--",
    linewidth=2,
    label="Vision-only",
    alpha=0.6,
)

# Mark start and end of dropout
ax.scatter(
    [pos_truth[dropout_start_idx, 0]],
    [pos_truth[dropout_start_idx, 1]],
    c="red",
    s=150,
    marker="o",
    label="Dropout Start",
    zorder=10,
    edgecolors="black",
    linewidths=2,
)
ax.scatter(
    [pos_truth[dropout_end_idx - 1, 0]],
    [pos_truth[dropout_end_idx - 1, 1]],
    c="orange",
    s=150,
    marker="s",
    label="Dropout End",
    zorder=10,
    edgecolors="black",
    linewidths=2,
)

# Highlight dropout segment
ax.scatter(
    pos_truth[dropout_mask, 0],
    pos_truth[dropout_mask, 1],
    c="red",
    s=50,
    alpha=0.2,
    zorder=5,
)

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Dropout Region Detail (t=20-35s)")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)
ax.axis("equal")

# Uncertainty ellipses during dropout
ax = axes[1, 0]


def plot_covariance_ellipse(ax, pos, cov, n_std=3.0, **kwargs):
    """Plot covariance ellipse at position with n-std confidence."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


# Plot trajectory with uncertainty at key points
times_to_plot = [
    dropout_start_idx - 10,
    dropout_start_idx,
    dropout_start_idx + 50,
    dropout_end_idx - 1,
    dropout_end_idx + 10,
]
labels = ["Before", "Start", "Middle", "End", "After"]
colors = ["green", "orange", "red", "orange", "green"]

ax.plot(
    pos_truth[zoom_start_idx:zoom_end_idx, 0],
    pos_truth[zoom_start_idx:zoom_end_idx, 1],
    "k-",
    linewidth=2,
    label="Ground Truth",
    alpha=0.5,
)
ax.plot(
    ekf_fusion.filtered_means[zoom_start_idx:zoom_end_idx, 0],
    ekf_fusion.filtered_means[zoom_start_idx:zoom_end_idx, 1],
    "b-",
    linewidth=1.5,
    label="Fusion",
    alpha=0.6,
)

for i, (idx, label, color) in enumerate(zip(times_to_plot, labels, colors)):
    pos_est = ekf_fusion.filtered_means[idx, :2]
    cov_pos = ekf_fusion.filtered_covariances[idx, :2, :2]

    plot_covariance_ellipse(
        ax, pos_est, cov_pos, n_std=3.0, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2
    )

    ax.scatter([pos_est[0]], [pos_est[1]], c=color, s=80, zorder=10, edgecolors="black")

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Fusion: Uncertainty Growth (3σ ellipses)")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)
ax.axis("equal")

# Position error over time with dropout highlighted
ax = axes[1, 1]

pos_error_fusion = np.linalg.norm(ekf_fusion.filtered_means[:, :2] - pos_truth, axis=1)
pos_error_vision = np.linalg.norm(ekf_vision.filtered_means[:, :2] - pos_truth, axis=1)

ax.plot(t_cam, pos_error_fusion * 100, "b-", linewidth=2, label="Fusion", alpha=0.8)
ax.plot(t_cam, pos_error_vision * 100, "g--", linewidth=2, label="Vision-only", alpha=0.6)
ax.axvspan(dropout_start_t, dropout_end_t, alpha=0.3, color="red", label="Dropout Period")
ax.axhline(2.0, color="purple", linestyle="--", linewidth=1.5, label="PRD Limit (2 cm)", alpha=0.7)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (cm)")
ax.set_title("Position Error vs Time")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 20])

plt.tight_layout()
plot_path = OUTPUT_DIR / "trajectory_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {plot_path}")
plt.close()

print("\n✅ Trajectory visualization complete!")
print("\n✅ Trajectory visualization complete!")
print("\n✅ Trajectory visualization complete!")
