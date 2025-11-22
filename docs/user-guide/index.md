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

- **NEES < 6**: Overconfident (covariance too small)
- **NEES ~ 8**: Well-tuned (for 8D state)
- **NEES > 10**: Underconfident (covariance too large)

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

### Real-Time Tracking

```python
# Initialize filter
cfg = EKFConfig()
x, P = ekf_initialize_state(sim, cfg)

# Process each frame
for frame in data_stream:
    x, P = ekf_update_step(x, P, frame, cfg)
    yield x[:2]  # Position estimate
```

### Offline Analysis

```python
# Run forward filter
fwd = extended_kalman_filter(cfg, sim)

# Run backward smoother
smoothed = rts_smoother(fwd, cfg, sim)

# Generate report
generate_filter_report(smoothed, output_path="report.pdf")
```

### Parameter Search

```python
# Grid search over process noise
results = []
for q_pos in [0.01, 0.02, 0.05, 0.1]:
    cfg = EKFConfig(process_noise_pos=q_pos)
    result = extended_kalman_filter(cfg, sim)
    nees = compute_nees(result, sim['x_truth']).mean()
    results.append((q_pos, nees))

# Find best parameter
best_q = min(results, key=lambda x: abs(x[1] - 8.0))
```
