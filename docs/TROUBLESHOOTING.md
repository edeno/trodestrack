# TrodesTrack Troubleshooting Guide

**Common filter failures, diagnostic steps, and solutions**

This guide helps you identify and fix problems with Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) tracking performance.

---

## 📋 Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Symptom Index](#symptom-index)
3. [Common Problems](#common-problems)
4. [Data Quality Issues](#data-quality-issues)
5. [Filter Divergence](#filter-diverges-to-infinity)
6. [Performance Issues](#slow-performance)
7. [Advanced Debugging](#advanced-debugging)

---

## Quick Diagnostic Checklist

Run through these checks first before diving into specific problems:

### ✅ Data Sanity Checks

```python
# Check for NaN/Inf in inputs.
# IMU samples must always be finite (the filter does not mask IMU rows).
assert np.isfinite(sim['U_imu']).all(), "IMU contains NaN/Inf"

# LED arrays may contain NaNs by design — that's how the simulator and
# the filter mark dropped or partially-occluded LEDs. Only require
# finite values where the camera mask says the frame is valid AND that
# specific LED row is finite (the filter's measurement model handles
# single-LED frames; see tests/filters/test_ekf_partial_observations.py).
mask = sim['mask_cam']
led1_valid = mask & np.isfinite(sim['Z_cam_led1']).all(axis=1)
led2_valid = mask & np.isfinite(sim['Z_cam_led2']).all(axis=1)
assert np.isfinite(sim['Z_cam_led1'][led1_valid]).all()
assert np.isfinite(sim['Z_cam_led2'][led2_valid]).all()
assert led1_valid.any() or led2_valid.any(), "no valid LED frames"

# Check units (common mistake: pixels instead of meters)
led_positions = sim['Z_cam_led1'][led1_valid]
assert led_positions.max() < 10.0, "LED positions likely in pixels, not meters"

# Check time alignment
assert len(sim['t_cam_exp']) == len(sim['Z_cam_led1']), "Time/measurement mismatch"
assert sim['t_imu'].min() <= sim['t_cam_exp'].min(), "IMU starts after camera"
assert sim['t_imu'].max() >= sim['t_cam_exp'].max(), "IMU ends before camera"
```

### ✅ Configuration Sanity Checks

```python
cfg = EKFConfig()

# Check process noise (should be positive)
assert cfg.process_noise_pos > 0, "Process noise must be positive"
assert cfg.process_noise_vel > 0, "Process noise must be positive"

# Check measurement noise (should be small)
assert cfg.measurement_noise_pos < 0.1, "Measurement noise too large (>10cm²)"

# Check damping (should be reasonable)
assert 0.0 <= cfg.damping_coeff <= 5.0, "Damping coefficient out of range"
```

### ✅ Generate QA Report

```bash
uv run python examples/08_qa_report_generation.py
```

**Check these metrics first:**
1. **NEES histogram**: Should be centered around 2.0 (position-only NEES; expected mean equals state_dim)
2. **Position RMSE**: Should be < 10 cm (ideally < 2 cm)
3. **Innovation statistics**: Should be zero-mean
4. **Trajectory plot**: Check for discontinuities or divergence

---

## Symptom Index

Click on your symptom to jump to the solution:

- [Filter diverges to infinity](#filter-diverges-to-infinity)
- [Position estimate jitters/oscillates](#position-estimate-jitters)
- [Heading estimate drifts continuously](#heading-drift)
- [Filter lags behind true motion](#filter-lags-behind-motion)
- [Position jumps at LED swaps](#led-swap-jumps)
- [Large errors during occlusions](#occlusion-drift)
- [Stationary rat drifts](#stationary-drift)
- [Covariance grows unbounded](#covariance-explosion)
- [Filter runs extremely slowly](#slow-performance)
- [NEES inconsistent (< 1 or > 4 for position-only NEES)](#nees-inconsistency)
- [NaN/Inf in filter output](#naninf-output)

---

## Common Problems

### Filter Diverges to Infinity

**Symptom:** Position estimates grow without bound (> 100 m).

#### Diagnostic Steps

1. **Check input units:**
   ```python
   # LED positions should be in meters, not pixels
   print(f"LED1 max: {sim['Z_cam_led1'].max():.2f} m")
   # Should be < 2 m for typical arena
   ```

2. **Check for NaN/Inf in inputs:**
   ```python
   assert np.isfinite(sim['U_imu']).all(), "IMU contains invalid values"
   ```

3. **Check initial state:**
   ```python
   import jax.numpy as jnp
   from trodestrack.models.filter_common import initialize_state

   layout = get_layout(cfg.state_mode)
   dt_cam = float(np.mean(np.diff(sim["t_cam_exp"])))
   state = initialize_state(
       sim["Z_cam_led1"],
       sim["Z_cam_led2"],
       sim["mask_cam"],
       dt_cam,
       cfg.led_distance,
       layout=layout,
   )
   pos = state.mean[jnp.array(layout.pos_idx)]
   vel = state.mean[jnp.array(layout.vel_idx)]
   print(f"Initial position: {pos}")  # Should be in arena
   print(f"Initial velocity: {vel}")  # Should be < 2 m/s
   ```

#### Solutions

**If LED positions are in pixels:**
```python
# Convert to meters using scale factor
scale_m_per_pixel = 0.001  # Example: 1 mm per pixel
sim['Z_cam_led1'] = sim['Z_cam_led1'] * scale_m_per_pixel
sim['Z_cam_led2'] = sim['Z_cam_led2'] * scale_m_per_pixel
```

**If IMU units are wrong:**
```python
# Check conversion from raw to SI units (see PRD Section 5)
# Gyro: raw * 0.061 * (π/180) = rad/s
# Accel: raw * 0.000061 * 9.80665 = m/s²
```

**If covariance initialization is too small:**
```python
from trodestrack.models.ekf import EKFState
from trodestrack.models.filter_common import initialize_state
from trodestrack.models.state_layout import get_layout

layout = get_layout(cfg.state_mode)
dt_cam = float(np.mean(np.diff(sim["t_cam_exp"])))
state = initialize_state(
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
    dt_cam,
    cfg.led_distance,
    layout=layout,
)
inflated = EKFState(mean=state.mean, cov=state.cov * 10.0)
result = extended_kalman_filter(
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
    initial_state=inflated,
)
```

---

### Position Estimate Jitters

**Symptom:** Position oscillates frame-to-frame instead of smooth tracking.

#### Diagnostic Steps

1. **Check measurement noise:**
   ```python
   # Plot raw LED observations
   plt.plot(sim['Z_cam_led1'][:, 0], label='LED1 x')
   plt.show()
   # If raw data is jittery, increase measurement_noise_pos
   ```

2. **Check NEES:**
   - If NEES < 1.0: Filter is underconfident (covariance too large) → Kalman gain too high → trusts noisy measurements too much

#### Solutions

**Option 1: Increase measurement noise (trust camera less)**
```python
cfg = EKFConfig(
    measurement_noise_pos=0.02**2  # Increase from 0.01² (2 cm vs 1 cm)
)
```

**Option 2: Decrease process noise (smoother motion model)**
```python
cfg = EKFConfig(
    process_noise_pos=5e-5,   # Decrease from 1e-4 (current default)
    process_noise_vel=2.5e-3  # Decrease from 5e-3 (current default)
)
```

**Option 3: Use smoother for offline analysis**
```python
from trodestrack.runtime.offline import rts_smoother

# Smoother eliminates jitter by using future observations
smoothed = rts_smoother(
    result, cfg, sim["t_imu"], sim["U_imu"], sim["t_cam_exp"]
)
```

**Option 4: Enable Mahalanobis gating (reject outliers)**
```python
cfg = EKFConfig(
    use_mahalanobis_gating=True,
    mahalanobis_threshold_prob=0.997
)
```

---

### Heading Drift

**Symptom:** Heading estimate diverges from true heading, especially during straight-line motion.

#### Root Cause

**Gyro bias is unobservable during straight-line motion.** The filter cannot distinguish:
- Constant heading with zero bias
- Drifting heading with non-zero bias

#### Diagnostic Steps

1. **Check motion type:**
   ```python
   # X_truth columns: [x, y, vx, vy, theta] at IMU rate.
   X_truth = sim['X_truth']
   vel_angle = np.arctan2(X_truth[:, 3], X_truth[:, 2])  # vy, vx
   plt.plot(X_truth[:, 4], label='True heading')
   plt.plot(vel_angle, label='Velocity heading')
   # If these match → straight-line motion → bias unobservable
   ```

2. **Check bias convergence:**
   ```python
   # Plot gyro bias estimate (use layout indexing — index varies by state_mode)
   layout = get_layout(cfg.state_mode)
   gyro_bias = result.filtered_means[:, list(layout.bias_gyro_idx)]
   plt.plot(gyro_bias)
   # Should converge to truth during rotation, drift during straight line
   ```

#### Solutions

**Option 1: Enable heading pseudo-measurements from dual LEDs**
```python
cfg = EKFConfig(
    use_heading_measurement=True,     # Extract heading from LED orientation
    led_distance=None,                # Auto-detect spacing
    led_distance_tolerance=0.3,       # Reject if spacing > 30% off
    adaptive_heading_noise=True       # Scale noise with geometry quality
)
```

**Requirements:** LEDs must be rigidly attached with spacing ≥ 3 cm.

**Option 2: Use velocity heading when confidence is high**
```python
# Built-in: When only one LED is visible, filter uses velocity heading
# No configuration needed, but only helps during motion
```

**Option 3: Increase gyro bias process noise (allow faster adaptation)**
```python
cfg = EKFConfig(
    process_noise_gyro_bias=5e-7  # Increase from 5e-8 (current default)
)
```

**Trade-off:** Faster adaptation → more noise in bias estimate.

**Option 4: Use smoother (offline only)**
```python
# Smoother can correct past heading using future observations
smoothed = rts_smoother(
    result, cfg, sim["t_imu"], sim["U_imu"], sim["t_cam_exp"]
)
```

---

### Filter Lags Behind Motion

**Symptom:** Position estimate is behind true position, especially during fast motion.

#### Diagnostic Steps

1. **Check velocity RMSE:**
   - If velocity RMSE is high → filter can't track acceleration changes

2. **Check NEES:**
   - If NEES > 4.0 → filter is overconfident (covariance too small) → Kalman gain too low → trusts model too much

#### Solutions

**Option 1: Increase velocity process noise**
```python
cfg = EKFConfig(
    process_noise_vel=2e-2  # Increase from 5e-3 (current default)
)
```

**Option 2: Decrease damping coefficient**
```python
cfg = EKFConfig(
    damping_coeff=0.1  # Decrease from 0.2 (current default; less friction)
)
```

**Option 3: Increase measurement trust**
```python
cfg = EKFConfig(
    measurement_noise_pos=0.005**2  # Decrease from 0.01² (trust camera more)
)
```

**Option 4: Use IEKF (iterated EKF)**
```python
cfg = EKFConfig(
    num_iter=3  # Iterate update step for better linearization
)
```

**Cost:** 3× slower per update.

---

### LED Swap Jumps

**Symptom:** Position estimate jumps discontinuously when front/back LEDs are swapped.

#### Root Cause

LED swap detection failed → filter treats swapped LEDs as true observation → heading flips 180°.

#### Diagnostic Steps

1. **Check LED spacing:**
   ```python
   distances = np.linalg.norm(sim['Z_cam_led2'] - sim['Z_cam_led1'], axis=1)
   print(f"Median LED spacing: {np.nanmedian(distances):.3f} m")
   # Should be consistent (e.g., 4 cm ± 5 mm)
   ```

2. **Check for swap events:**
   ```python
   # Plot heading discontinuities (use layout — index varies by state_mode)
   layout = get_layout(cfg.state_mode)
   heading_diff = np.diff(result.filtered_means[:, layout.heading_idx])
   swap_frames = np.where(np.abs(heading_diff) > 2.0)[0]  # > 114° jump
   print(f"Detected {len(swap_frames)} swap events")
   ```

#### Solutions

**Option 1: Enable heading measurement (prevents swaps)**
```python
cfg = EKFConfig(
    use_heading_measurement=True,     # Constrains heading to LED orientation
    led_distance_tolerance=0.3        # Rejects inconsistent spacing
)
```

**Option 2: Enable Mahalanobis gating (rejects swapped observations)**
```python
cfg = EKFConfig(
    use_mahalanobis_gating=True,
    mahalanobis_threshold_prob=0.997  # Tight gate
)
```

**Option 3: Manual swap correction in preprocessing**
```python
# Detect and correct swaps before filtering
swap_mask = detect_led_swaps(sim)
sim['Z_cam_led1'][swap_mask], sim['Z_cam_led2'][swap_mask] = \
    sim['Z_cam_led2'][swap_mask], sim['Z_cam_led1'][swap_mask]
```

**Option 4: Use mixture update (advanced, not yet implemented)**
```python
# Future feature: Test both hypotheses (swap vs no-swap) and select best
```

---

### Occlusion Drift

**Symptom:** Large position errors during camera dropouts (occlusions).

#### Expected Behavior

**PRD Target:** ≤ 3.5 m drift after 5 seconds of occlusion (95th percentile over realistic sessions; see `tests/integration/test_prd_session.py::test_5s_dropout_drift_integration` and `qa.metrics.compute_dropout_drift`).

**Note:** The simulator typically achieves ~11 cm in this scenario; the 3.5 m target is the relaxed PRD requirement that accommodates consumer-grade IMU drift in worst-case sessions.

#### Diagnostic Steps

1. **Check dropout duration:** (no dedicated helper ships; compute inline.)
   ```python
   import numpy as np

   mask = np.asarray(sim["mask_cam"], dtype=bool)
   # Run-length encode the False (dropped) runs.
   gaps = []
   length = 0
   for valid in mask:
       if not valid:
           length += 1
       elif length:
           gaps.append(length)
           length = 0
   if length:
       gaps.append(length)
   if gaps:
       max_gap = max(gaps)
       fs_cam = 1.0 / np.mean(np.diff(sim["t_cam_exp"]))
       print(f"Max dropout: {max_gap} frames (~{max_gap / fs_cam:.1f}s)")
   else:
       print("No dropouts in this session.")
   ```

2. **Check IMU-only drift:**
   ```python
   # During dropout, filter relies on IMU only
   # Check if IMU bias estimates are converged before dropout
   layout = get_layout(cfg.state_mode)
   bias_idx = list(layout.bias_gyro_idx) + list(layout.bias_accel_idx)
   bias_at_dropout = result.filtered_means[dropout_start, bias_idx]
   ```

#### Solutions

**Option 1: Enable adaptive process noise during dropout**
```python
cfg = EKFConfig(
    adaptive_q_during_dropout=True,      # Inflate uncertainty during blackout
    dropout_q_pos_multiplier=10.0,       # Position uncertainty 10×
    dropout_q_vel_multiplier=10.0,       # Velocity uncertainty 10×
    dropout_q_bias_multiplier=0.1        # Slow bias drift
)
```

**Effect:** Filter becomes more uncertain → smoother can correct with future data.

**Option 2: Use smoother (offline only)**
```python
# Smoother corrects dropout drift using future observations
smoothed = rts_smoother(
    result, cfg, sim["t_imu"], sim["U_imu"], sim["t_cam_exp"]
)
```

**Example:** 5s dropout drift reduces from ~2 m (filter) to ~0.5 m (smoother).

**Option 3: Increase IMU noise densities (if IMU is low quality)**
```python
cfg = EKFConfig(
    imu_gyro_noise_density=5e-4,    # Increase from 0.00017453 (SpikeGadgets default)
    imu_accel_noise_density=5e-3,   # Increase from 0.00196133 (SpikeGadgets default)
)
```

**Option 4: Pre-converge biases before critical period**
```python
# Ensure rat rotates for ≥ 5 seconds before occlusion
# This allows gyro bias to converge
```

---

### Stationary Drift

**Symptom:** Position estimate drifts when rat is not moving.

#### Root Cause

IMU noise accumulates during stationary periods → small velocity estimates → position drift.

#### Diagnostic Steps

1. **Check velocity magnitude during stationary period:**
   ```python
   velocity_mag = np.linalg.norm(result.filtered_means[:, 2:4], axis=1)
   stationary_mask = velocity_mag < 0.05  # < 5 cm/s
   drift = np.linalg.norm(np.diff(result.filtered_means[stationary_mask, :2], axis=0), axis=1)
   print(f"Mean drift during stationary: {drift.mean():.4f} m/frame")
   ```

#### Solutions

**Option 1: Enable ZUPT (zero-velocity update)**
```python
cfg = EKFConfig(
    enable_zupt=True,                      # Detect stationary periods
    zupt_velocity_threshold=0.05,          # Trigger if |v| < 5 cm/s
    zupt_measurement_noise=0.01**2         # Trust v=0 with 1 cm/s noise
)
```

**Effect:** When velocity drops below threshold, filter applies `v = 0` constraint.

**Option 2: Increase velocity damping**
```python
cfg = EKFConfig(
    damping_coeff=0.5  # Increase from 0.2 (current default; faster velocity decay)
)
```

**Option 3: Decrease velocity process noise**
```python
cfg = EKFConfig(
    process_noise_vel=2.5e-3  # Decrease from 5e-3 (current default; trust low-velocity model)
)
```

---

### Covariance Explosion

**Symptom:** Filter covariance grows unbounded (diagonal elements > 100 m²).

#### Diagnostic Steps

1. **Check process noise:**
   ```python
   print(f"Position Q: {cfg.process_noise_pos}")  # Should be < 1.0
   print(f"Velocity Q: {cfg.process_noise_vel}")  # Should be < 10.0
   ```

2. **Check for measurement dropout:**
   ```python
   dropout_rate = 1.0 - sim['mask_cam'].mean()
   print(f"Camera dropout rate: {dropout_rate * 100:.1f}%")
   # If > 50%, covariance will grow
   ```

#### Solutions

**Option 1: Decrease process noise**
```python
cfg = EKFConfig(
    process_noise_pos=5e-5,   # Decrease from 1e-4 (current default)
    process_noise_vel=2.5e-3  # Decrease from 5e-3 (current default)
)
```

**Option 2: Enable adaptive Q (inflation during dropout only)**
```python
cfg = EKFConfig(
    adaptive_q_during_dropout=True,      # Only inflate during blackout
    dropout_q_pos_multiplier=5.0,        # Moderate inflation (was 10×)
)
```

**Option 3: Check for filter divergence**
```python
# If covariance explodes AND position diverges → see "Filter Divergence"
assert np.isfinite(result.filtered_covariances).all(), "Covariance contains NaN/Inf"
```

---

### Slow Performance

**Symptom:** Filter takes > 1 second per frame (expected: ~0.4 ms per frame).

#### Diagnostic Steps

1. **Check JIT compilation:**
   ```python
   # First call is slow (compilation), second call is fast
   import time

   def _run():
       return extended_kalman_filter(
           cfg,
           sim["t_imu"],
           sim["U_imu"],
           sim["t_cam_exp"],
           sim["Z_cam_led1"],
           sim["Z_cam_led2"],
           sim["mask_cam"],
       )

   start = time.time()
   _run()  # First call: ~5 seconds (compilation)
   print(f"First call: {time.time() - start:.2f}s")

   start = time.time()
   _run()  # Second call: ~0.01 seconds (cached trace)
   print(f"Second call: {time.time() - start:.2f}s")
   ```

2. **Check iteration count (IEKF):**
   ```python
   print(f"IEKF iterations: {cfg.num_iter}")  # Should be 1 for standard EKF
   ```

3. **Check IMU sample count:**
   ```python
   print(f"IMU samples: {len(sim['U_imu'])}")  # Should be < 100k for 30-min session
   ```

#### Solutions

**Option 1: Use EKF instead of UKF**
```python
# UKF is 1-5× slower due to sigma-point transforms
from trodestrack.models.ekf import extended_kalman_filter
# Instead of:
# from trodestrack.models.ukf import unscented_kalman_filter
```

**Option 2: Reduce IEKF iterations**
```python
cfg = EKFConfig(num_iter=1)  # Standard EKF (no iteration)
```

**Option 3: Downsample IMU data**
```python
# Note: SpikeGadgets hardware refreshes at 104 Hz with sample-and-hold
# Output appears as ~20-30 kHz but contains only 104 Hz unique data
# Preprocessing removes sample-and-hold repeats → ~100 Hz effective rate
# Synthetic simulations typically use 200 Hz for benchmarking
```

**Option 4: Use GPU acceleration**
```python
import jax
# Force GPU device
jax.config.update('jax_platform_name', 'gpu')
```

**Expected speedup:** 5-10× on GPU for long sessions.

---

### NEES Inconsistency

**Symptom:** NEES histogram is not centered around 2.0 (position-only NEES; expected mean equals state_dim).

See **[Filter Tuning](tuning.md)** for detailed parameter tuning guidance.

**Quick fixes:**

- **NEES > 4.0** (overconfident, P too small) → Increase `process_noise_pos` by 2-5× OR increase `measurement_noise_pos` by 2×
- **NEES < 1.0** (underconfident, P too large) → Decrease `process_noise_pos` by 2× OR decrease `measurement_noise_pos` by 2×

---

### NaN/Inf Output

**Symptom:** Filter returns NaN or Inf in state estimates or covariances.

#### Diagnostic Steps

1. **Check inputs:**
   ```python
   # IMU must always be finite. LEDs may contain NaNs by design (dropouts /
   # single-LED frames are valid inputs); only require finiteness on rows
   # the camera mask marks valid AND that have finite coordinates.
   assert np.isfinite(sim['U_imu']).all()
   mask = sim['mask_cam']
   led1_valid = mask & np.isfinite(sim['Z_cam_led1']).all(axis=1)
   led2_valid = mask & np.isfinite(sim['Z_cam_led2']).all(axis=1)
   assert np.isfinite(sim['Z_cam_led1'][led1_valid]).all()
   assert np.isfinite(sim['Z_cam_led2'][led2_valid]).all()
   ```

2. **Check for extreme values:**
   ```python
   print(f"Max LED position: {np.nanmax(sim['Z_cam_led1'])}")  # Should be < 10 m
   print(f"Max IMU gyro: {np.nanmax(sim['U_imu'][:, 0])}")     # Should be < 100 rad/s
   ```

3. **Check covariance:** (use the filter's own initialization output for `P0`.)
   ```python
   from trodestrack.models.filter_common import initialize_state
   from trodestrack.models.state_layout import get_layout

   layout = get_layout(cfg.state_mode)
   dt_cam = float(np.mean(np.diff(sim["t_cam_exp"])))
   state0 = initialize_state(
       sim["Z_cam_led1"], sim["Z_cam_led2"], sim["mask_cam"],
       dt_cam, cfg.led_distance, layout=layout,
   )
   P0 = np.asarray(state0.cov)
   # NaN in covariance indicates numerical instability
   assert np.isfinite(P0).all(), "P0 contains NaN/Inf"
   print(f"Initial covariance condition number: {np.linalg.cond(P0):.2e}")
   # Should be < 1e10
   ```

#### Solutions

**Option 1: Pass NaNs through; the filter handles them.**
```python
# The camera measurement model treats NaN-containing LED rows as
# missing observations and falls back to single-LED or prediction-only
# updates as appropriate. You don't need to overwrite NaNs with dummy
# values — see tests/filters/test_ekf_partial_observations.py.
# If a frame is *entirely* unusable (both LEDs NaN), set mask_cam=False
# for that frame so the filter skips the camera update altogether:
both_invalid = (
    ~np.isfinite(sim['Z_cam_led1']).all(axis=1)
    & ~np.isfinite(sim['Z_cam_led2']).all(axis=1)
)
sim['mask_cam'][both_invalid] = False
```

**Option 2: Increase numerical stability**
```python
# Joseph-form covariance updates (already enabled by default)
# If still unstable, increase regularization:
# (Not exposed in config, but internal diagonal_boost=1e-9)
```

**Option 3: Check for division by zero**
```python
# Ensure LED spacing > 0
if cfg.led_distance is not None:
    assert cfg.led_distance > 0.001, "LED spacing too small"
```

---

## Data Quality Issues

### LED Detection Failures

**Symptom:** Many missing or low-confidence LED detections.

#### Solutions

1. **Increase measurement noise to account for uncertainty:**
   ```python
   cfg = EKFConfig(measurement_noise_pos=0.02**2)  # 2 cm noise
   ```

2. **Enable Mahalanobis gating to reject spurious detections:**
   ```python
   cfg = EKFConfig(use_mahalanobis_gating=True)
   ```

3. **Check camera calibration:**
   - Verify homography matrix (if using pixel→meter conversion)
   - Check lens distortion correction

### IMU Saturation

**Symptom:** IMU readings clipped at maximum sensor range.

#### Diagnostic Steps

```python
# Check for saturation (depends on sensor range)
gyro_max = np.abs(sim['U_imu'][:, 0]).max()
accel_max = np.abs(sim['U_imu'][:, 1:3]).max()
print(f"Max gyro: {gyro_max:.2f} rad/s")   # Typical limit: 2000°/s = 35 rad/s
print(f"Max accel: {accel_max:.2f} m/s²")  # Typical limit: 16g = 157 m/s²
```

#### Solutions

1. **Use higher-range IMU** (hardware change)
2. **Increase IMU noise densities to model uncertainty:**
   ```python
   cfg = EKFConfig(
       imu_gyro_noise_density=0.001,    # 10× increase
       imu_accel_noise_density=0.05     # 10× increase
   )
   ```

### Time Synchronization Issues

**Symptom:** Filter performance varies with random seed (should be deterministic).

#### Diagnostic Steps

```python
# Check timestamp alignment
dt_cam = np.diff(sim['t_cam_exp'])
print(f"Camera frame intervals: min={dt_cam.min():.4f}, max={dt_cam.max():.4f}s")
# Should be consistent (~0.033s for 30 Hz)

dt_imu = np.diff(sim['t_imu'])
print(f"IMU sample intervals: min={dt_imu.min():.6f}, max={dt_imu.max():.6f}s")
# Should be consistent (~0.005s for 200 Hz)
```

#### Solutions

1. **Verify hardware sync** (SpikeGadgets provides hardware timestamp)
2. **Interpolate missing timestamps** (if dropped frames)
3. **Resample IMU to uniform rate** (if jittered sampling)

---

## Advanced Debugging

### Enable Verbose Logging

```python
# Not yet implemented, but planned:
# cfg = EKFConfig(verbose=True, log_level='DEBUG')
```

### Visualize Innovation Statistics

```python
from trodestrack.viz.video import create_diagnostic_video

# Generate video with 9-panel diagnostics
create_diagnostic_video(
    sim,                            # SimOut
    "debug_video.mp4",              # output_path
    filter_results=result,          # optional EKFResult overlay
    state_mode=cfg.state_mode,      # required when filter_results is set; aligns
                                    # heading/bias panels with the actual layout
    fps=30,
    speedup=1.0,                    # real-time playback
)
```

**Check these panels:**
1. **Trajectory**: Discontinuities indicate swaps or divergence
2. **Position error**: Should be < 5 cm most of the time
3. **Velocity error**: Should be < 20 cm/s most of the time
4. **Heading error**: Should be < 10° most of the time
5. **NEES**: Should stay in [1, 4] envelope (position-only NEES, ``state_dim=2``)
6. **Bias estimates**: Should converge during rotation

### Run Integration Tests

```bash
# Run PRD acceptance tests
uv run pytest tests/integration/test_prd_session.py -v
```

**Expected results:**
- Position RMSE ≤ 2 cm ✓
- Velocity RMSE ≤ 10 cm/s ✓
- Heading RMSE ≤ 7° ✓
- Throughput ≥ 10× realtime ✓

### Compare EKF vs UKF

```python
# Run both filters on same data
from trodestrack.models.ekf import extended_kalman_filter
from trodestrack.models.ukf import unscented_kalman_filter

filter_args = (
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)
result_ekf = extended_kalman_filter(*filter_args)
result_ukf = unscented_kalman_filter(*filter_args)

X_truth_at_cam = np.array(
    [sim["X_truth"][np.argmin(np.abs(sim["t_imu"] - t_c))] for t_c in sim["t_cam_exp"]]
)
truth_pos = X_truth_at_cam[:, :2]
# Signature is compute_position_rmse(positions_true, positions_est, ...).
rmse_ekf = compute_position_rmse(
    truth_pos, np.asarray(result_ekf.filtered_means[:, :2])
)
rmse_ukf = compute_position_rmse(
    truth_pos, np.asarray(result_ukf.filtered_means[:, :2])
)
print(f"EKF RMSE: {rmse_ekf*100:.2f} cm")
print(f"UKF RMSE: {rmse_ukf*100:.2f} cm")
```

**If UKF is much better:** Linearization error is significant → consider IEKF.

---

## Getting Help

### Before Opening an Issue

1. **Generate QA report** and attach to issue
2. **Run diagnostic checklist** from top of this guide
3. **Include configuration** (copy-paste `EKFConfig` settings)
4. **Describe data source** (synthetic sim, real data, sensor specs)

### Provide Minimal Reproducible Example

```python
# Minimal failing example
from trodestrack.sim.simple import simulate_stationary, SimpleSimConfig
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig

config_sim = SimpleSimConfig(duration_s=5.0)
sim = simulate_stationary(position=[0.5, 0.5], config=config_sim, seed=42)

cfg = EKFConfig()  # Add your custom settings here
result = extended_kalman_filter(
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)

# Describe the problem here
print(f"Problem: Position RMSE is {compute_position_rmse(...):.4f} m (expected < 0.02)")
```

### Useful Resources

- **[Filter Tuning](tuning.md)** - Parameter selection guide
- **[Examples](../examples/index.md)** - Educational examples with expected outputs
- **[GitHub Issues](https://github.com/edeno/trodestrack/issues)** - Report bugs

---

**Still stuck?** Open an issue with your QA report and we'll help debug!
