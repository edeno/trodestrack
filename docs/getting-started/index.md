# Getting Started

This section will help you get up and running with TrodesTrack quickly.

## Overview

TrodesTrack is a sensor-fusion tracking system that combines:

- **Video tracking** (Trodes LEDs or DeepLabCut keypoints) at ~30 Hz
- **IMU data** (gyroscope + accelerometer) at ~100 Hz

Using Extended Kalman Filters (EKF) or Unscented Kalman Filters (UKF), TrodesTrack produces accurate estimates of:

- Position (x, y) in meters
- Velocity (vx, vy) in m/s
- Heading (theta) in radians
- IMU biases (gyro + accel) for drift compensation

## Learning Path

We recommend following this progression:

1. **[Installation](installation.md)** - Set up your environment
2. **[Quick Start](quickstart.md)** - Run your first filter in 3 minutes
3. **[Python API](python-api.md)** - Understand the core API patterns
4. **[State Layouts](../user-guide/state-layouts.md)** - Learn dimension-agnostic coding
5. **[Examples](../examples/index.md)** - Explore pedagogical examples

## Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- For GPU acceleration: CUDA-compatible GPU with JAX GPU support

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Install TrodesTrack**

    ---

    [:octicons-arrow-right-24: Installation Guide](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Run Your First Filter**

    ---

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

</div>
