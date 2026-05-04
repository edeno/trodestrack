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
    EKF wins 5/9 metrics and is 1-5x faster. **Start with EKF!**

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

## Python API Quick Reference

### Generate Synthetic Data

```python
from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

# Default config matches SpikeGadgets hardware (104 Hz IMU, realistic noise)
config = RatIMUSimConfig(duration_s=10.0, seed=42)
sim = simulate_rat_imu(config)
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

# Access results
positions = result.filtered_means[:, :2]  # (N, 2) positions
covariances = result.filtered_covariances  # (N, layout.n, layout.n)
```

### Use State Layouts (Recommended!)

```python
from trodestrack.models.state_layout import get_layout

# Get layout from config (dimension-agnostic!)
layout = get_layout(cfg.state_mode)

# Extract states using layout indices.
# Defaults: state_mode="2d_cam_3d_imu" -> layout.n=10, layout.vel_idx=(2,3,4) (vx, vy, vz).
positions = result.filtered_means[:, layout.pos_idx]      # (N, 2)
velocities = result.filtered_means[:, layout.vel_idx]     # (N, len(layout.vel_idx))
headings = result.filtered_means[:, layout.heading_idx]   # (N,) for scalar-heading layouts
```

!!! warning "Avoid Hardcoded Indices"
    Always use `layout.pos_idx` instead of `[:, 0:2]`. Hardcoded indices break when switching state modes (5D, 8D, 10D, 15D).

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
    pdf_path="report.pdf",
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

## Next Steps

- **[Python API Guide](python-api.md)**: Deeper dive into the API
- **[State Layouts](../user-guide/state-layouts.md)**: Write dimension-agnostic code
- **[Examples](../examples/index.md)**: Explore all pedagogical examples
- **[Tuning Guide](../user-guide/tuning.md)**: Optimize filter parameters
