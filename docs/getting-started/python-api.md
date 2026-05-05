# Python API Guide

This guide covers the core API patterns for using TrodesTrack programmatically.

## Core Workflow

The typical TrodesTrack workflow is:

1. **Load/Generate Data** - Simulation or real sensor data
2. **Configure Filter** - Set up EKF or UKF parameters
3. **Run Filter** - Process data through the Kalman filter
4. **Extract Results** - Use state layouts for dimension-agnostic access
5. **Analyze/Visualize** - Generate reports or custom plots

## Data Format

TrodesTrack expects data in a dictionary format with specific keys:

```python
sim = {
    # Timestamps
    't_imu': np.array(...),          # (M,) IMU timestamps in seconds
    't_cam_exp': np.array(...),      # (N,) camera exposure times in seconds

    # Camera measurements
    'Z_cam_led1': np.array(...),     # (N, 2) LED1 positions [x, y] in meters
    'Z_cam_led2': np.array(...),     # (N, 2) LED2 positions [x, y] in meters
    'mask_cam': np.array(...),       # (N,) boolean mask for valid frames

    # IMU measurements. Channel count depends on the configured state mode:
    #   (M, 3) [omega_z, f_x, f_y]                    — non-quaternion layouts;
    #                                                    works with the default
    #                                                    "2d_cam_3d_imu" as a
    #                                                    degenerate path (vz idle).
    #   (M, 4) [omega_z, f_x, f_y, f_z]               — "2d_cam_3d_imu" with 3D velocity.
    #   (M, 6) [omega_x, omega_y, omega_z, f_x, f_y, f_z] — quaternion-orientation
    #                                                       layouts (e.g.
    #                                                       "3d_cam_6dof_imu").
    'U_imu': np.array(...),

    # Optional: ground truth for validation (at IMU rate)
    'X_truth': np.array(...),        # (M, 5) true state [x, y, vx, vy, theta]
}
```

## Simulation

### Simple Simulations

For testing and validation, use analytic simulations:

```python
from trodestrack.sim.simple import (
    simulate_stationary,
    simulate_constant_velocity,
    simulate_circular,
    SimpleSimConfig,
)

# Configure simulation. ``seed`` is an argument of each simulate_* call,
# not a SimpleSimConfig field.
config = SimpleSimConfig(
    duration_s=10.0,
    fs_cam=30.0,          # Camera sampling rate (Hz)
    fs_imu=200.0,         # IMU sampling rate (Hz)
)

# Stationary rat at (0.5, 0.5) meters
sim = simulate_stationary(position=[0.5, 0.5], config=config, seed=42)

# Moving at constant velocity
sim = simulate_constant_velocity(
    velocity=[0.2, 0.0],  # m/s
    config=config,
    seed=42,
)

# Circular motion
sim = simulate_circular(config=config, seed=42)
```

### Realistic Rat IMU Simulation

For production-like testing:

```python
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

config = RatIMUSimConfig(
    duration_s=30.0,                  # 30 second session
    fs_imu=104.0,                     # SpikeGadgets hardware IMU rate (Hz)
    fs_cam=30.0,                      # Camera frame rate (Hz)
    arena_w=1.0,                      # Arena width  (x-axis, meters)
    arena_h=1.0,                      # Arena height (y-axis, meters)

    # Camera artifacts
    cam_dropout_prob=0.1,             # 10% frame dropout rate
)

# ``seed`` is an argument of ``simulate_rat_imu``, not the config.
sim = simulate_rat_imu(config, seed=42)
```

## Filter Configuration

### EKF Configuration

```python
from trodestrack.models.ekf import EKFConfig

cfg = EKFConfig(
    # State mode (determines state dimension; default is 2d_cam_3d_imu)
    state_mode="2d_cam_3d_imu",          # 10D: [x,y,vx,vy,vz,θ,b_gz,b_ax,b_ay,b_az]

    # Process noise
    process_noise_pos=1e-4,              # m^2/s
    process_noise_vel=5e-3,              # m^2/s^3
    process_noise_heading=5e-4,          # rad^2/s
    process_noise_gyro_bias=5e-8,        # rad^2/s^3
    process_noise_accel_bias=2e-5,       # m^2/s^5

    # Measurement noise
    measurement_noise_pos=0.01**2,       # m^2 (1 cm)
    measurement_noise_heading=0.05**2,   # rad^2 (~3 deg)

    # Dynamics
    damping_coeff=0.2,                   # 1/s (velocity decay)

    # Robustness features
    use_mahalanobis_gating=True,         # Reject outliers (3σ)
    mahalanobis_threshold_prob=0.997,    # 3-sigma gate

    enable_zupt=True,                    # Zero-velocity updates
    zupt_velocity_threshold=0.02,        # m/s

    # Adaptive noise during dropout
    adaptive_q_during_dropout=True,
    dropout_q_pos_multiplier=2.0,        # see docs/TUNING.md for the full set
)
```

### UKF Configuration

```python
from trodestrack.models.ukf import UKFConfig

# UKF uses same base parameters plus sigma-point settings.
# Defaults: alpha=sqrt(3)≈1.732, beta=2.0, kappa=1.0 (UKFConfig.aggressive()).
# Note: alpha must be large enough that (n + λ) = α² (n + κ) > 1e-3
# (UKFConfig validates and rejects degenerate spreads like alpha=1e-3).
cfg = UKFConfig(
    state_mode="2d_full",
    # ... same parameters as EKFConfig ...

    # UKF-specific (use the validated aggressive preset)
    alpha=1.732,   # sqrt(3): wide sigma-point spread
    beta=2.0,      # Prior knowledge (2.0 for Gaussian)
    kappa=1.0,     # Secondary scaling parameter
)
```

## Running Filters

### Extended Kalman Filter

```python
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig

cfg = EKFConfig()
filter_args = (
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)
result = extended_kalman_filter(*filter_args)

# Result is an EKFResult NamedTuple. The state dimension follows the
# layout for cfg.state_mode (default "2d_cam_3d_imu" -> 10D state).
print(f"Filtered means shape: {result.filtered_means.shape}")       # (N, layout.n)
print(f"Filtered covariances shape: {result.filtered_covariances.shape}")  # (N, layout.n, layout.n)
```

### Unscented Kalman Filter

```python
from trodestrack.models.ukf import unscented_kalman_filter, UKFConfig

cfg = UKFConfig()
result = unscented_kalman_filter(
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)
```

### RTS Smoother (Offline)

For offline analysis, the smoother uses future observations:

```python
from trodestrack.runtime.offline import rts_smoother

forward_result = extended_kalman_filter(*filter_args)
smoothed_result = rts_smoother(
    forward_result, cfg, sim["t_imu"], sim["U_imu"], sim["t_cam_exp"]
)
```

## State Layouts

**Always use state layouts** for dimension-agnostic code:

```python
from trodestrack.models.state_layout import get_layout

# Get layout from config
layout = get_layout(cfg.state_mode)

# Extract states (works with any state dimension!)
positions = result.filtered_means[:, layout.pos_idx]
velocities = result.filtered_means[:, layout.vel_idx]
headings = result.filtered_means[:, layout.heading_idx]

# Extract covariances
P = result.filtered_covariances
pos_cov = P[:, layout.pos_idx, :][:, :, layout.pos_idx]  # (N, 2, 2)
pos_std = np.sqrt(np.diagonal(pos_cov, axis1=1, axis2=2))  # (N, 2)
```

### Available State Modes

| Mode | Dim | State Vector | Use Case |
|------|-----|--------------|----------|
| `"2d_cam_3d_imu"` | 10D | [x, y, vx, vy, vz, theta, b_gz, b_ax, b_ay, b_az] | **Default**: 2D camera + 3D accel |
| `"2d_full"` | 8D | [x, y, vx, vy, theta, b_gz, b_ax, b_ay] | Standard 2D sensor fusion |
| `"vision_only"` | 5D | [x, y, vx, vy, theta] | Camera-driven tracking; APIs still require placeholder IMU arrays |
| `"2d_cam_6dof_imu_orientation"` | 14D | [x, y, vx, vy, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az] | Experimental: 2D camera + 6-DOF IMU with quaternion orientation |
| `"3d_euler"` | 15D | [x, y, z, vx, vy, vz, roll, pitch, yaw, b_gx, b_gy, b_gz, b_ax, b_ay, b_az] | Full 3D tracking with Euler-angle orientation |
| `"3d_quat"` | 16D | [x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az] | Full 3D tracking with quaternion orientation |
| `"3d_cam_6dof_imu"` | 16D | same as `"3d_quat"` | **Required** by the experimental `extended_kalman_filter_3d` entry point (3D LED observations + 6-channel IMU) |

See [State Layouts](../user-guide/state-layouts.md) for complete documentation.

## QA and Visualization

### Generate QA Report

```python
import numpy as np
from trodestrack.models.state_layout import get_layout
from trodestrack.qa.metrics import compute_nees
from trodestrack.qa.report import generate_qa_report

layout = get_layout(cfg.state_mode)

# Align ground truth (IMU rate, 5D [x, y, vx, vy, theta]) to camera frames.
X_truth_at_cam = np.array(
    [sim["X_truth"][np.argmin(np.abs(sim["t_imu"] - t_c))] for t_c in sim["t_cam_exp"]]
)
filtered = np.asarray(result.filtered_means)
filtered_cov = np.asarray(result.filtered_covariances)

pos_idx = list(layout.pos_idx)
vel_idx_2d = list(layout.vel_idx)[:2]  # X_truth has only vx, vy
heading_col = int(layout.heading_idx)

# Position-only NEES (state_dim=2): expected mean ~ 2 for a consistent filter.
nees = compute_nees(
    states_true=X_truth_at_cam[:, :2],
    states_est=filtered[:, pos_idx],
    covariances_est=filtered_cov[np.ix_(np.arange(filtered.shape[0]), pos_idx, pos_idx)],
)
generate_qa_report(
    pdf_path="qa_report.pdf",
    t=sim["t_cam_exp"],
    positions_true=X_truth_at_cam[:, :2],
    positions_est=filtered[:, pos_idx],
    velocities_true=X_truth_at_cam[:, 2:4],
    velocities_est=filtered[:, vel_idx_2d],
    headings_true=X_truth_at_cam[:, 4],
    headings_est=filtered[:, heading_col],
    nees=nees,
    state_dim=2,
)
```

### Compute Metrics

```python
from trodestrack.qa.metrics import (
    compute_position_rmse,
    compute_nees,
)

# Position RMSE (uses the aligned X_truth_at_cam from above).
# Signature is compute_position_rmse(positions_true, positions_est, ...) —
# the truth array goes first.
pos_rmse = compute_position_rmse(
    X_truth_at_cam[:, :2],
    np.asarray(result.filtered_means[:, :2]),
)
print(f"Position RMSE: {pos_rmse * 100:.2f} cm")

# NEES (filter consistency, position only)
nees = compute_nees(
    states_true=X_truth_at_cam[:, :2],
    states_est=np.asarray(result.filtered_means[:, :2]),
    covariances_est=np.asarray(result.filtered_covariances[:, :2, :2]),
)
print(f"Mean NEES: {nees.mean():.2f} (expected ~ 2 for position-only NEES)")
```

### Create Diagnostic Video

```python
from trodestrack.viz.video import create_diagnostic_video

create_diagnostic_video(
    sim,                       # SimOut from simulate_rat_imu
    "diagnostics.mp4",         # output_path (positional)
    filter_results=result,     # optional EKFResult overlay
    fps=30,
    speedup=2.0,               # 2x playback speed
)
```

## Error Handling

TrodesTrack raises informative errors:

```python
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout

# EKFConfig itself does not validate state_mode at construction time;
# the lookup happens inside get_layout(...) (and inside the filter call
# path), which raises KeyError for unknown modes.
cfg = EKFConfig(state_mode="invalid")  # no error here
try:
    layout = get_layout(cfg.state_mode)
except KeyError as e:
    print(f"Error: {e}")  # "'invalid' is not a registered state_mode"

# Data shape mismatch. extended_kalman_filter takes positional arrays
# (ekf_config, t_imu, U_imu, t_cam, Z_cam_led1, Z_cam_led2, mask_cam, ...)
# and validates IMU shape via validate_imu_input_shape — passing a
# wrong-shaped IMU array raises ValueError; passing the wrong number of
# positional args raises TypeError before the shape check runs.
try:
    cfg = EKFConfig(state_mode="2d_full")
    bad_U_imu = np.zeros((100, 7))  # 7 channels is not a valid layout
    result = extended_kalman_filter(
        cfg,
        sim["t_imu"], bad_U_imu, sim["t_cam_exp"],
        sim["Z_cam_led1"], sim["Z_cam_led2"], sim["mask_cam"],
    )
except ValueError as e:
    print(f"Error: {e}")  # validate_imu_input_shape message
```

## Performance Tips

1. **JIT Compilation**: First call is slow (compilation), subsequent calls are fast
2. **Batch Processing**: Process multiple sessions without recompiling
3. **GPU Acceleration**: Use `jax.config.update('jax_platform_name', 'gpu')` for large sessions
4. **State Mode**: Use `"vision_only"` (5D) if you don't need bias estimation

## Next Steps

- **[State Layouts](../user-guide/state-layouts.md)**: Deep dive into dimension-agnostic coding
- **[Tuning Guide](../TUNING.md)**: Optimize filter parameters
- **[Troubleshooting Guide](../TROUBLESHOOTING.md)**: Common filter failures and solutions
