# User Guide

This section provides detailed guidance for using TrodesTrack effectively in your research.

## Topics

<div class="grid cards" markdown>

-   :material-vector-arrange-above:{ .lg .middle } **State Layouts**

    ---

    Write dimension-agnostic code that works with any state mode (5D, 8D, 10D, 14D, 15D, 16D)

    [:octicons-arrow-right-24: State Layouts](state-layouts.md)

-   :material-tune:{ .lg .middle } **Filter Tuning**

    ---

    Systematic approach to parameter selection using NEES-based diagnostics

    [:octicons-arrow-right-24: Tuning Guide](tuning.md)

-   :material-wrench:{ .lg .middle } **Troubleshooting**

    ---

    Common filter failures, diagnostic steps, and solutions

    [:octicons-arrow-right-24: Troubleshooting](troubleshooting.md)

</div>

## Key Concepts

### State Estimation

TrodesTrack uses Kalman filtering to estimate the "state" of the animal:

- **Position** (x, y): Where the animal is in the arena
- **Velocity** (vx, vy): How fast and which direction it's moving
- **Heading** (theta): Which way the animal is facing
- **Biases**: IMU sensor drift corrections

The filter maintains both:

- **State estimate** (mean): Best guess of current state
- **Uncertainty** (covariance): How confident we are in the estimate

### Sensor Fusion

TrodesTrack combines two complementary sensor types:

| Sensor | Rate | Strengths | Weaknesses |
|--------|------|-----------|------------|
| Camera | 30 Hz | Absolute position | Occlusions, jitter |
| IMU | 100 Hz | High rate, no dropout | Drift over time |

The Kalman filter optimally weights these sensors based on their uncertainties.

### Filter Consistency

A well-tuned filter is "consistent" when its uncertainty estimates match actual errors:

- **NEES < 1**: Underconfident (covariance too large)
- **NEES ~ 2**: Well-tuned (position-only NEES, ``state_dim=2``)
- **NEES > 4**: Overconfident (covariance too small)

See the [Tuning Guide](tuning.md) for how to achieve consistency.

## Best Practices

1. **Always use state layouts** - Never hardcode state indices
2. **Start with default parameters** - They work for most scenarios
3. **Check NEES first** - Filter consistency is more important than RMSE
4. **Use smoothing for offline analysis** - It corrects for filter overconfidence
5. **Generate QA reports** - Document your parameter choices

## Hardware Requirements

### IMU Data

TrodesTrack consumes IMU data from 6-axis hardware (3-axis gyro +
3-axis accelerometer). The number of channels passed to the filter
depends on the configured ``state_mode``:

- **3 channels** ``[ω_z, f_x, f_y]`` — works with any non-quaternion layout
  (``"2d_full"``, ``"vision_only"``, and the default ``"2d_cam_3d_imu"``).
  In ``"2d_cam_3d_imu"`` this is a degenerate path: ``f_z`` is unobserved
  and ``vz`` stays idle (see `tests/filters/test_imu_shape_validation.py`).
- **4 channels** ``[ω_z, f_x, f_y, f_z]`` — recommended for the default
  ``"2d_cam_3d_imu"`` layout when you want 3D velocity. Select these
  columns from your raw 6-axis stream.
- **6 channels** ``[ω_x, ω_y, ω_z, f_x, f_y, f_z]`` — required for
  quaternion-orientation layouts (``"2d_cam_6dof_imu_orientation"``,
  ``"3d_cam_6dof_imu"``).

Units must be SI (rad/s for gyro, m/s² for accelerometer), not raw
sensor counts. ``trodestrack.models.filter_common.validate_imu_input_shape``
enforces the layout-specific channel rules at filter entry.

### Camera Data

- LED or keypoint positions in **meters** (not pixels)
- Timestamps in **seconds**
- Boolean mask for valid detections

## Common Workflows

### Online (Forward-Only) Tracking

The `trodestrack online` CLI command is **forward-filter-only** (no
backward smoothing): it loads the full IMU / camera / LED arrays from
disk, runs the EKF in a single forward pass via `extended_kalman_filter`,
and writes filtered means + covariances to `--output-dir`. It is *not*
streaming — there is no per-frame ingest loop. "Online" here means
"forward-only / suitable for online use offline" rather than "incremental
real-time".

True frame-by-frame ingest would require driving `predict_step` /
`update_step` from `trodestrack.models.filter_common` directly; that is
not exposed as a CLI today. See `uv run trodestrack online --help` for
the batch CLI.

### Offline Analysis

```python
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig
from trodestrack.runtime.offline import rts_smoother

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
smoothed = rts_smoother(
    result, cfg, sim["t_imu"], sim["U_imu"], sim["t_cam_exp"]
)
# See README.md for the full QA-report invocation via generate_qa_report.
```

### Parameter Search

```python
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.state_layout import get_layout
from trodestrack.qa.metrics import compute_nees

# `sim` is the SimOut produced upstream (e.g. by simulate_rat_imu).
X_truth_at_cam = np.array(
    [sim["X_truth"][np.argmin(np.abs(sim["t_imu"] - t_c))] for t_c in sim["t_cam_exp"]]
)
truth_pos = X_truth_at_cam[:, :2]  # X_truth is laid out [x, y, vx, vy, theta]

results = []
for q_pos in [0.01, 0.02, 0.05, 0.1]:
    cfg = EKFConfig(process_noise_pos=q_pos)
    layout = get_layout(cfg.state_mode)
    pos_idx = list(layout.pos_idx)[:2]  # x, y columns under the active layout
    result = extended_kalman_filter(
        cfg,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        sim["mask_cam"],
    )
    means = np.asarray(result.filtered_means)
    covs = np.asarray(result.filtered_covariances)
    pos_means = means[:, pos_idx]
    pos_covs = covs[np.ix_(np.arange(means.shape[0]), pos_idx, pos_idx)]
    nees = compute_nees(
        states_true=truth_pos,
        states_est=pos_means,
        covariances_est=pos_covs,
    )
    results.append((q_pos, float(nees.mean())))

best_q = min(results, key=lambda r: abs(r[1] - 2.0))  # Target = state_dim
```
