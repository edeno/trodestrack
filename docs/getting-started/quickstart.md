# Quick Start

Get up and running with TrodesTrack in 3 minutes.

## Prerequisites

Make sure you have [installed TrodesTrack](installation.md).

## 1. Generate and Filter Synthetic Data

The fastest way to understand TrodesTrack is to run the EKF example:

```bash
# Run EKF on basic scenarios (stationary, constant velocity, circular)
uv run python examples/03_ekf_basic_scenarios.py
```

This generates 3 diagnostic PNGs showing filter performance, bias convergence, and NEES consistency checks.

!!! tip "Key Insight"
    Gyro bias is only observable during rotation! This is why the circular motion scenario shows bias convergence while stationary and straight-line motion do not.

## 2. Compare EKF vs UKF

```bash
uv run python examples/04_ukf_basic_scenarios.py
```

Compares sigma-point (UKF) vs Jacobian (EKF) approaches.

!!! note "Verdict"
    EKF wins 6/9 metrics (UKF: 3/9). Wall-clock cost is comparable under JIT-compiled JAX with warm dispatch on these scenarios; on backends without JIT (per-step Python loops) UKF can be several times slower. **Start with EKF and re-measure on your target backend.**

## 3. Test Dropout Robustness

```bash
# Test with camera dropouts
uv run python examples/05_ekf_with_dropouts.py
```

Simulates 10%, 20%, and 30% camera dropout to stress-test IMU-only periods.

## 4. Use Smoothing for Offline Analysis

```bash
uv run python examples/07_smoother_demonstration.py
```

Shows how backward RTS smoothing achieves **3x drift reduction** on 5-second dropout by using future observations.

## 5. Generate QA Reports

```bash
uv run python examples/08_qa_report_generation.py
```

Creates a publication-quality PDF with all PRD metrics, NEES/NIS checks, and time series plots.

## 6. Run Real Data From a YAML Config

For SpikeGadgets/Trodes-style real data, put input paths and preprocessing settings in a session YAML:

```yaml
inputs:
  format: spikegadgets_trodes
  imu_file: path/to/imu.parquet
  position_file: path/to/position.parquet
camera:
  meters_per_pixel: 0.0022
filter:
  state_mode: 2d_cam_6dof_imu_orientation
  enable_experimental_accel_translation: false
  use_gravity_orientation_update: true
  use_mahalanobis_gating: false
outputs:
  output_dir: runs/session_001
led_identity:
  mode: auto
```

Then run either CLI path:

```bash
uv run trodestrack online --config session.yaml
uv run trodestrack smooth --config session.yaml
```

Config-driven real-data runs write `session_diagnostics.json` plus IMU calibration reports when available. IMU-fused real-data runs also compute a vision-only safety baseline by default, so they may take about twice as long as a single filter pass. For tilted headstages, prefer `2d_cam_6dof_imu_orientation` with `enable_experimental_accel_translation: false` until accelerometer-driven translation has passed the safety check; use `filter.state_mode: vision_only` only as a baseline or fallback.

See `examples/session_spikegadgets_trodes.yaml` for a template with the common real-data options included.

!!! note "Vision-only gating"
    Config-driven `state_mode: vision_only` runs default Mahalanobis gating off so large but valid camera motion is not mistaken for an outlier. Set `filter.use_mahalanobis_gating: true` only when you have tuned process and measurement noise for your camera data.

## Python API Quick Reference

### Generate Synthetic Data

```python
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

# Default config matches SpikeGadgets hardware (104 Hz IMU, realistic noise)
config = RatIMUSimConfig(duration_s=10.0)
sim = simulate_rat_imu(config, seed=42)  # seed is an arg of simulate_rat_imu
```

### Run EKF Filter

```python
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig

# Run filter with default configuration
cfg = EKFConfig()
result = extended_kalman_filter(
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)

# Access results — use layout indices, never hardcoded slices (see warning below).
from trodestrack.models.state_layout import get_layout
layout = get_layout(cfg.state_mode)
positions = result.filtered_means[:, layout.pos_idx]   # (N, 2)
covariances = result.filtered_covariances              # (N, layout.n, layout.n)
```

### Use State Layouts (Recommended!)

```python
# Get layout from config (dimension-agnostic!) — already imported above.

# Extract states using layout indices.
# Defaults: state_mode="2d_cam_3d_imu" -> layout.n=10, layout.vel_idx=(2,3,4) (vx, vy, vz).
positions = result.filtered_means[:, layout.pos_idx]      # (N, 2)
velocities = result.filtered_means[:, layout.vel_idx]     # (N, len(layout.vel_idx))
headings = result.filtered_means[:, layout.heading_idx]   # (N,) for scalar-heading layouts
```

!!! warning "Avoid Hardcoded Indices"
    Always use `layout.pos_idx` instead of `[:, 0:2]`. Hardcoded indices break when switching state modes (5D, 8D, 10D, 14D, 15D, 16D).

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

# Headings: scalar-2D-heading layouts (vision_only / 2d_full / 2d_cam_3d_imu /
# 2d_cam_6dof_imu_orientation) expose heading as a scalar yaw angle and the
# QA report's heading panels can plot ``filtered[:, heading_col]`` directly.
# Tuple-heading layouts (``3d_euler``: 3-tuple Euler; ``3d_quat`` /
# ``3d_cam_6dof_imu``: 4-tuple scalar-first quaternion) need an explicit
# yaw extraction; see ``trodestrack.models.filter_common.state_yaw`` for
# the reference implementation. ``layout.has_heading_2d`` distinguishes
# the two cases.
if layout.has_heading_2d:
    # Scalar yaw — heading_idx may be a Python int or a 1-tuple.
    heading_col = (
        int(layout.heading_idx)
        if isinstance(layout.heading_idx, int)
        else int(layout.heading_idx[0])
    )
    headings_est_for_report = filtered[:, heading_col]
else:
    raise NotImplementedError(
        f"This QA snippet plots scalar yaw; layout heading_idx="
        f"{layout.heading_idx!r} (state_mode={cfg.state_mode}) returns a "
        "tuple. Extract yaw via ``state_yaw`` per sample before plotting."
    )

# Position-only NEES (state_dim=2): expected mean ~ 2 for a consistent filter.
nees = compute_nees(
    states_true=X_truth_at_cam[:, :2],
    states_est=filtered[:, pos_idx],
    covariances_est=filtered_cov[np.ix_(np.arange(filtered.shape[0]), pos_idx, pos_idx)],
)
generate_qa_report(
    pdf_path="report.pdf",
    t=sim["t_cam_exp"],
    positions_true=X_truth_at_cam[:, :2],
    positions_est=filtered[:, pos_idx],
    velocities_true=X_truth_at_cam[:, 2:4],
    velocities_est=filtered[:, vel_idx_2d],
    headings_true=X_truth_at_cam[:, 4],
    headings_est=headings_est_for_report,
    nees=nees,
    state_dim=2,
)
```

## Next Steps

- **[Python API Guide](python-api.md)**: Deeper dive into the API
- **[State Layouts](../user-guide/state-layouts.md)**: Write dimension-agnostic code
- **[Examples](../examples/index.md)**: Explore all pedagogical examples
- **[Tuning Guide](../user-guide/tuning.md)**: Optimize filter parameters
