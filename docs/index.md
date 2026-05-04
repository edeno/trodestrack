# TrodesTrack

**Sensor-fused 2D rat tracking with JAX EKF/UKF for neuroscience research**

TrodesTrack combines video tracking (Trodes LEDs and/or DeepLabCut keypoints) with IMU data from SpikeGadgets headstages to provide accurate position, velocity, and heading estimates for freely-moving rats on behavioral mazes.

## Features

- **Sensor Fusion**: Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for combining video (~30 Hz) and IMU (100 Hz) measurements
- **3D IMU Support**: Full 6-axis IMU processing (3-axis gyro + 3-axis accel) with gravity compensation
- **Online & Offline Processing**: Real-time filtering and RTS smoothing for offline analysis
- **Robust Handling**: Occlusions, LED swaps, reflections, and sensor dropout
- **JAX-Accelerated**: High-performance implementation using JAX - **316x realtime** on CPU, GPU-ready
- **Rich Simulation**: Comprehensive synthetic data generation for testing and validation
- **Diagnostic Visualization**: Publication-quality video output for quality control

## Performance

TrodesTrack achieves production-ready accuracy:

| Metric | Target | Achieved |
|--------|--------|----------|
| Position RMSE | < 2 cm | < 2 cm |
| Velocity RMSE | < 10 cm/s | < 10 cm/s |
| Heading RMSE | < 7 deg | < 7 deg |
| Throughput | > 10x realtime | **316x realtime** |

## Quick Example

```python
from trodestrack.sim.simple import simulate_circular, SimpleSimConfig
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig
from trodestrack.models.state_layout import get_layout

# Generate synthetic data
sim_config = SimpleSimConfig(duration_s=10.0)
sim = simulate_circular(sim_config)

# Run EKF
ekf_config = EKFConfig()
result = extended_kalman_filter(
    ekf_config,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)

# Extract states using layout (dimension-agnostic!)
layout = get_layout(ekf_config.state_mode)
positions = result.filtered_means[:, layout.pos_idx]  # (N, 2) in meters
velocities = result.filtered_means[:, layout.vel_idx]  # (N, 2) in m/s
headings = result.filtered_means[:, layout.heading_idx]  # (N,) in radians
```

## Getting Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install TrodesTrack with pip/uv and set up your environment

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Run your first filter on synthetic data in 3 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Learn about tuning, troubleshooting, and best practices

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation for all modules

    [:octicons-arrow-right-24: API Reference](reference/)

</div>

## Hardware Compatibility

**Supported IMU Hardware:**

TrodesTrack is designed for **SpikeGadgets headstages** with integrated 6-axis IMU sensors.

| Sensor | Range | Resolution |
|--------|-------|------------|
| Accelerometer | +/- 2g | 16-bit (0.000061g/LSB) |
| Gyroscope | +/- 2000 deg/s | 16-bit (0.061 deg/s/LSB) |
| Refresh Rate | 104 Hz | (both sensors enabled) |

**Video Tracking:**

- Trodes LED detection (dual LED setup for heading)
- DeepLabCut keypoint tracking (any pose estimation output)
- Camera rate: typically 30 Hz (configurable)

## Project Status

**Current Milestone**: Integration & QA (M4) - Ready for Production Testing

See the [GitHub repository](https://github.com/edeno/trodestrack) for the full roadmap and changelog.

## Citation

If you use TrodesTrack in your research, please cite:

```bibtex
@software{trodestrack2025,
  title={TrodesTrack: Sensor-fused 2D rat tracking with JAX EKF/UKF},
  author={Denovellis, Eric},
  year={2025},
  url={https://github.com/edeno/trodestrack}
}
```

## License

MIT License - see [LICENSE](https://github.com/edeno/trodestrack/blob/main/LICENSE) for details.
