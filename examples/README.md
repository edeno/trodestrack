# trodestrack Examples

This directory contains examples demonstrating the `trodestrack` simulator and filter capabilities.

## Simulation Examples

### 01_basic_simulation.py

Comprehensive demonstration of the rat IMU + vision simulator with four scenarios:

**Example 1: Basic Simulation**
- Single LED at body center
- No confidence modeling (uniform noise)
- 10-second duration @ 200 Hz IMU, 30 Hz camera
- 15% dropout rate
- Output: `01_basic_sim.png`

**Example 2: Two-LED Simulation**
- Front/back LEDs (6 cm separation) for heading measurements
- Demonstrates body-frame to world-frame LED positioning
- 10% dropout rate
- Output: `02_two_led_sim.png`

**Example 3: Confidence-Enabled Simulation**
- Confidence scores correlated with dropouts
- Confidence-scaled measurement noise
- 20% dropout rate (more challenging)
- Two LEDs enabled
- Output: `03_confidence_sim.png`

**Example 4: Noise Validation**
- Longer 60-second run for statistical analysis
- Validates measurement noise characteristics
- Compares measured vs. true IMU signals (gyro, accel)
- Compares camera position measurements vs. truth
- Output: `04_noise_validation.png`

### Running the Examples

```bash
# From repository root
uv run python examples/01_basic_simulation.py
```

This will:
1. Run all four simulation scenarios
2. Generate publication-quality plots
3. Print summary statistics
4. Save PNG files to `examples/`

To display plots interactively, uncomment `plt.show()` at the end of the script.

### Plot Descriptions

Each overview plot contains:
- **Row 1, Left**: 2D trajectory in arena with camera observations color-coded by time
- **Row 1, Right**: Speed over time (showing velocity clipping)
- **Row 2**: Position time series (x, y) with truth + observations
- **Row 3**: IMU measurements (gyro, accel X, accel Y) with ground truth overlaid
- **Row 4**: Bias random walks and confidence/dropout patterns

Noise validation plots show:
- Measurement errors over time
- Error distributions (histograms)
- 2D spatial error patterns for camera

### Key Features Demonstrated

✅ **Physics**
- Ornstein-Uhlenbeck smooth motion
- Second-order position integration
- Velocity damping and speed clipping
- Arena boundary reflections

✅ **IMU Modeling**
- Body-frame gyroscope (yaw rate)
- Body-frame accelerometers (2D planar)
- White noise + bias random walks
- Ground truth channels for validation

✅ **Camera Modeling**
- LED positions in body frame → world frame
- Timestamp jitter and latency (exposure vs. arrival)
- Angle-aware interpolation (no ±π jumps)
- Frame dropouts
- Optional confidence scores

✅ **Validation**
- Measurement error statistics
- Noise distribution checks
- Bias drift analysis

## Expected Output Statistics

From Example 4 (60s simulation):

```
IMU Measurement Noise (std):
  Gyroscope:      ~0.007 rad/s ≈ 0.4 °/s
  Accelerometer:  ~0.42 m/s²

Camera Measurement Noise:
  2D position:    ~3.2 mm std
  Dropout rate:   ~10%

Bias Random Walks:
  Gyro:          ~0.03 °/s peak-to-peak
  Accel:         ~0.04 m/s² peak-to-peak
```

These match the configured noise densities and random walk parameters.

## Next Steps

- Use simulated data to test EKF/UKF implementations
- Tune filter parameters using NEES diagnostics
- Benchmark filter throughput and accuracy
- Test robustness to occlusions and dropouts
