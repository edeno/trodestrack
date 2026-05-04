# User Guide

This section provides detailed guidance for using TrodesTrack effectively in your research.

## Topics

<div class="grid cards" markdown>

-   :material-vector-arrange-above:{ .lg .middle } **State Layouts**

    ---

    Write dimension-agnostic code that works with any state mode (5D, 8D, 10D, 15D)

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

- **NEES < 6**: Underconfident (covariance too large)
- **NEES ~ 8**: Well-tuned (for 8D state)
- **NEES > 10**: Overconfident (covariance too small)

See the [Tuning Guide](tuning.md) for how to achieve consistency.

## Best Practices

1. **Always use state layouts** - Never hardcode state indices
2. **Start with default parameters** - They work for most scenarios
3. **Check NEES first** - Filter consistency is more important than RMSE
4. **Use smoothing for offline analysis** - It corrects for filter overconfidence
5. **Generate QA reports** - Document your parameter choices

## Hardware Requirements

### IMU Data

TrodesTrack expects 6-axis IMU data:

- **Gyroscope**: Angular velocity (rad/s)
- **Accelerometer**: Linear acceleration (m/s^2)

Units must be SI (not raw sensor counts).

### Camera Data

- LED or keypoint positions in **meters** (not pixels)
- Timestamps in **seconds**
- Boolean mask for valid detections

## Common Workflows

### Online Tracking

For incremental, frame-by-frame use, run the `trodestrack online` CLI
command (`uv run trodestrack online --help` for usage). It wraps the EKF
predict/update primitives for streaming inputs.

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
from trodestrack.qa.metrics import compute_nees

X_truth_at_cam = np.array(
    [sim["X_truth"][np.argmin(np.abs(sim["t_imu"] - t_c))] for t_c in sim["t_cam_exp"]]
)
truth_pos = X_truth_at_cam[:, :2]

results = []
for q_pos in [0.01, 0.02, 0.05, 0.1]:
    cfg = EKFConfig(process_noise_pos=q_pos)
    result = extended_kalman_filter(
        cfg,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        sim["mask_cam"],
    )
    nees = compute_nees(
        states_true=truth_pos,
        states_est=np.asarray(result.filtered_means[:, :2]),
        covariances_est=np.asarray(result.filtered_covariances[:, :2, :2]),
    )
    results.append((q_pos, float(nees.mean())))

best_q = min(results, key=lambda r: abs(r[1] - 8.0))
```
