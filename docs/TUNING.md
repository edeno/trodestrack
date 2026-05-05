# TrodesTrack Filter Tuning Guide

**A systematic approach to parameter selection using NEES-based diagnostics**

This guide helps you tune Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) parameters for optimal tracking performance on your specific dataset.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Filter Consistency](#understanding-filter-consistency)
3. [Core Parameters](#core-parameters)
4. [Diagnostic Workflow](#diagnostic-workflow)
5. [Common Tuning Scenarios](#common-tuning-scenarios)
6. [Advanced Features](#advanced-features)
7. [Parameter Reference](#parameter-reference)

---

## Quick Start

### The 5-Minute Tuning Checklist

1. **Run with default parameters** on your data
2. **Check NEES** (Normalized Estimation Error Squared) in the QA report
3. **If NEES > 4.0** → filter is overconfident (covariance too small) → increase process noise Q or measurement noise R
4. **If NEES < 1.0** → filter is underconfident (covariance too large) → decrease process noise or measurement noise
5. **Iterate** until NEES ≈ 2.0 (position-only NEES, ``state_dim=2``)

> *NEES thresholds in this guide are for the position-only NEES the QA
> snippets compute (chi-square with 2 dof, mean = 2). For full-state NEES
> with ``state_dim=D``, the expected mean is D and the well-tuned interval
> is the chi-square 95% CI for D dof. Compute it with
> ``scipy.stats.chi2.ppf([0.025, 0.975], df=D)`` (e.g. D=8 → [2.18, 17.54]);
> do not scale the 2-dof interval linearly — chi-square quantiles do not
> scale with D.*

### Generate Your First QA Report

```python
from trodestrack.models.ekf import extended_kalman_filter, EKFConfig
from trodestrack.models.state_layout import get_layout
from trodestrack.qa.metrics import compute_nees
from trodestrack.qa.report import generate_qa_report

# Load your data into the dict shape this guide uses below: keys
# ``t_imu``, ``U_imu``, ``t_cam_exp``, ``Z_cam_led1``, ``Z_cam_led2``,
# ``mask_cam`` (and ``X_truth`` if available). Native loaders for Trodes,
# DeepLabCut, and SpikeGadgets are tracked under "In Progress" in
# README.md; until they ship, build the dict yourself from your own
# pipeline, e.g. via ``simulate_rat_imu`` for synthetic data.
# sim = load_my_data(...)

cfg = EKFConfig()
layout = get_layout(cfg.state_mode)
result = extended_kalman_filter(
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)
import numpy as np

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
    pdf_path="tuning_report.pdf",
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

**Key metrics to check:**
- **NEES panel**: Should be centered around 2.0 with most samples in [1.0, 4.0] for position-only NEES
- **Position RMSE**: Target ≤ 2 cm (PRD requirement)
- **Velocity RMSE**: Target ≤ 10 cm/s
- **Heading RMSE**: Target ≤ 7°
- **Innovation statistics**: Should be zero-mean with small variance

---

## Understanding Filter Consistency

### What is NEES?

**Normalized Estimation Error Squared (NEES)** measures whether your filter's uncertainty estimates are honest:

```
NEES = eₖᵀ Pₖ⁻¹ eₖ
```

Where:
- `eₖ = x̂ₖ - xₖ` is the estimation error
- `Pₖ` is the filter's covariance estimate

**For position-only NEES (state_dim=2, what the QA snippets compute):**
- **Expected NEES**: 2.0 (equal to state dimension)
- **95% CI**: [0.05, 7.38] (from χ²(2, 0.025) to χ²(2, 0.975))
- **Ideal range**: [1.0, 4.0] (informal guideline for "well-tuned")

For NEES on a higher-dimensional aligned subset (e.g., the 5D `[x, y, vx, vy, θ]`
slice that matches `X_truth`), the expected mean is **D** (the state dimension)
but the 95% CI must be looked up from the chi-square distribution — quantiles
do not scale linearly with D. Compute the bounds as
``scipy.stats.chi2.ppf([0.025, 0.975], df=D)`` (e.g. D=5 → [0.83, 12.83],
D=8 → [2.18, 17.54]).

### Interpreting NEES

| NEES Value (state_dim=2) | Diagnosis | Action |
|--------------------------|-----------|--------|
| < 1.0 | **Underconfident** (P too large) | Decrease Q or R |
| 1.0 - 4.0 | **Well-tuned** ✓ | No action needed |
| > 4.0 | **Overconfident** (P too small) | Increase Q (process noise) or R (measurement noise) |
| Highly variable | **Inconsistent** | Check for outliers, tune gating |

### Why NEES Matters

- **NEES ≈ 2.0** → filter covariance matches actual error (position-only NEES)
- **NEES < 1.0** → covariance too large → filter underutilizes measurements
- **NEES > 4.0** → covariance too small → smoother may overshoot and propagate errors

**Pro Tip:** Slightly underconfident (NEES ≈ 1.5-2.0) is safer than overconfident for real data.

---

## Core Parameters

### Process Noise (Q Matrix)

Controls how much the filter trusts the dynamics model vs measurements.

```python
@dataclass
class FilterCoreConfig:
    # Position random walk (m²/s)
    process_noise_pos: float = 1e-4  # DEFAULT

    # Velocity random walk (m²/s³)
    process_noise_vel: float = 5e-3  # DEFAULT

    # Heading random walk (rad²/s)
    process_noise_heading: float = 5e-4  # DEFAULT

    # Gyro bias random walk (rad²/s³)
    process_noise_gyro_bias: float = 5e-8  # DEFAULT

    # Accel bias random walk (m²/s⁵)
    process_noise_accel_bias: float = 2e-5  # DEFAULT
```

**When to adjust:**
- **Increase Q** if NEES > 4.0 (filter too confident; covariance too small)
- **Decrease Q** if NEES < 1.0 (filter too uncertain; covariance too large)
- **Increase `process_noise_vel`** if velocity estimates lag true motion
- **Increase `process_noise_gyro_bias`** if heading drifts during rotation

### Measurement Noise (R Matrix)

Controls how much the filter trusts the camera observations.

```python
    # Position measurement variance (m²)
    measurement_noise_pos: float = 0.01**2  # (1 cm)² = 1e-4

    # Heading measurement variance (rad²)
    measurement_noise_heading: float = 0.05**2  # (~3°)² = 2.5e-3
```

**When to adjust:**
- **Increase R** if camera tracking is noisy (jittery LED detections)
- **Decrease R** if camera tracking is very precise (controlled lab setup)
- **Scale with DLC confidence**: built-in scaling is `R = base_R / clip(confidence, clip_min, 1.0)` (linear, not squared) — see `confidence_to_R_diagonal` in `filter_common.py`

### IMU Noise Densities

Model the IMU sensor's intrinsic noise characteristics.

```python
    # Gyro noise density (rad/s/√Hz) — SpikeGadgets product manual (0.01°/s/√Hz)
    imu_gyro_noise_density: float = 0.00017453  # DEFAULT

    # Accel noise density (m/s²/√Hz) — SpikeGadgets product manual (0.2 mg/√Hz)
    imu_accel_noise_density: float = 0.00196133  # DEFAULT
```

**When to adjust:**
- **Increase** if IMU data is noisy (cheap sensor, poor attachment)
- **Decrease** if IMU data is clean (high-quality sensor, rigid mounting)
- **Use sensor datasheet** values if available (recommended)

**Gravity-frame convention:** `imu_gravity_body` is in world-frame
coordinates. The calibration helper `estimate_accel_gravity_body` reports a
sensor-frame stationary accelerometer reading; if the IMU mount is tilted,
rotate that vector from body/sensor frame into the world frame before using it
as `imu_gravity_body`.

### Velocity Damping

Models air drag and friction.

```python
    # Damping coefficient λ (1/s)
    damping_coeff: float = 0.2  # DEFAULT
```

**Interpretation:**
- `λ = 0.2` → velocity decays to ~82% in 1 second without acceleration (current default; light drag for low-friction floors)
- `λ = 0.5` → velocity decays to 60% in 1 second
- `λ = 1.0` → velocity decays to 37% in 1 second (high drag)
- `λ = 0.0` → no drag (unrealistic for animals)

**When to adjust:**
- **Increase λ** if animal stops quickly (high friction surface)
- **Decrease λ** if animal glides (smooth surface, low friction)

---

## Diagnostic Workflow

### Step 1: Baseline Run

Run filter with **default parameters** and generate QA report:

```bash
uv run python examples/08_qa_report_generation.py
```

### Step 2: Examine NEES Histogram

Open `tuning_report.pdf` and check the **NEES histogram** panel:

- **Most samples in [1.0, 4.0]?** → Well-tuned ✓
- **Peak > 4.0?** → Overconfident → Go to Step 3
- **Peak < 1.0?** → Underconfident → Go to Step 4

### Step 3: Fix Overconfidence (NEES > 4.0)

**Problem:** Filter covariance is too small.

**Solutions (try in order):**

1. **Increase position process noise** by 2-5×
   ```python
   cfg = EKFConfig(process_noise_pos=5e-4)  # Was 1e-4
   ```

2. **Increase velocity process noise** by 2×
   ```python
   cfg = EKFConfig(process_noise_vel=1e-2)  # Was 5e-3
   ```

3. **Increase IMU noise densities** by 2×
   ```python
   cfg = EKFConfig(
       imu_gyro_noise_density=3.5e-4,   # Was 0.00017453
       imu_accel_noise_density=3.9e-3,  # Was 0.00196133
   )
   ```

**Re-run and check NEES.** Repeat until NEES ≈ 2.0.

### Step 4: Fix Underconfidence (NEES < 1.0)

**Problem:** Filter covariance is too large.

**Solutions (try in order):**

1. **Decrease position process noise** by 2×
   ```python
   cfg = EKFConfig(process_noise_pos=5e-5)  # Was 1e-4
   ```

2. **Increase measurement trust** (decrease R) by 2×
   ```python
   cfg = EKFConfig(measurement_noise_pos=0.005**2)  # Was 0.01**2
   ```

3. **Decrease IMU noise densities** by 2×
   ```python
   cfg = EKFConfig(
       imu_gyro_noise_density=8.7e-5,    # Was 0.00017453
       imu_accel_noise_density=9.8e-4,   # Was 0.00196133
   )
   ```

**Re-run and check NEES.** Repeat until NEES ≈ 2.0.

### Step 5: Check RMSE vs PRD Targets

Once NEES is in range, verify accuracy:

| Metric | Target | How to Check |
|--------|--------|--------------|
| Position RMSE | ≤ 2 cm | Summary table in PDF |
| Velocity RMSE | ≤ 10 cm/s | Summary table in PDF |
| Heading RMSE | ≤ 7° | Summary table in PDF |

**If RMSE targets not met:**
- Check for **outliers** → Enable Mahalanobis gating (see below)
- Check for **bias convergence** → Inspect bias estimate plots
- Check for **LED swaps** → Inspect trajectory plot for discontinuities

---

## Common Tuning Scenarios

### Scenario 1: Camera Occlusions

**Symptom:** Large position errors during occlusions, NEES spikes during blackouts.

**Solution:** Enable adaptive process noise during dropout.

```python
cfg = EKFConfig(
    adaptive_q_during_dropout=True,        # Enable adaptation
    dropout_q_pos_multiplier=10.0,         # Inflate position uncertainty 10×
    dropout_q_vel_multiplier=10.0,         # Inflate velocity uncertainty 10×
    dropout_q_bias_multiplier=0.1          # Slow bias drift (optional)
)
```

**Effect:** Filter becomes more uncertain during blackouts → smoother can correct with future data.

### Scenario 2: Poor LED Tracking

**Symptom:** Jittery position estimates, frequent outliers, high NIS.

**Solution:** Increase measurement noise and enable gating.

```python
cfg = EKFConfig(
    measurement_noise_pos=0.02**2,         # Increase from 0.01² default (1 cm → 2 cm)
    use_mahalanobis_gating=True,           # Reject outliers
    mahalanobis_threshold_prob=0.99        # Reject top 1% (tighter than 0.997)
)
```

**Effect:** Filter trusts measurements less, rejects extreme outliers.

### Scenario 3: Heading Drift

**Symptom:** Heading estimate drifts away from true heading during straight-line motion.

**Solution:** Enable heading pseudo-measurements from dual LEDs.

```python
cfg = EKFConfig(
    use_heading_measurement=True,          # Extract heading from LED pair
    led_distance=0.04,                     # Auto-detect if None
    led_distance_tolerance=0.3,            # Reject if spacing > 30% off
    adaptive_heading_noise=True            # Increase R for poor geometry
)
```

**Effect:** Heading is constrained by LED orientation, reducing drift.

**Important:** Only use if LED spacing is reliable (≥ 3 cm, rigid attachment).

### Scenario 4: Stationary Drift

**Symptom:** Position estimate drifts during stationary periods (rat not moving).

**Solution:** Enable zero-velocity updates (ZUPT).

```python
cfg = EKFConfig(
    enable_zupt=True,                      # Enable ZUPT
    zupt_velocity_threshold=0.05,          # Trigger if |v| < 5 cm/s
    zupt_measurement_noise=0.01**2         # Trust velocity=0 with 1 cm/s noise
)
```

**Effect:** When velocity drops below threshold, filter applies v=0 constraint.

### Scenario 5: Fast, Erratic Motion

**Symptom:** Filter lags behind true motion, velocity RMSE high, NEES > 4.0.

**Solution:** Increase velocity process noise.

```python
cfg = EKFConfig(
    process_noise_vel=2e-2,                # Increase from 5e-3 default
    damping_coeff=0.1                      # Decrease damping (animal doesn't stop quickly)
)
```

**Effect:** Filter tracks rapid velocity changes more aggressively.

---

## Advanced Features

### Mahalanobis Gating (Outlier Rejection)

**Purpose:** Reject measurements that are statistically inconsistent with state estimate.

**When to use:**
- LED reflections off walls
- Occasional tracking errors (DLC artifacts)
- LED swaps not automatically resolved

**Configuration:**

```python
cfg = EKFConfig(
    use_mahalanobis_gating=True,
    mahalanobis_threshold_prob=0.997       # Reject measurements beyond 3σ
)
```

**Threshold probabilities:**
- `0.95` → Reject ~5% of measurements (loose gate)
- `0.99` → Reject ~1% of measurements (moderate gate)
- `0.997` → Reject ~0.3% of measurements (tight gate, **recommended**)

**Diagnostic:** Check **NIS (Normalized Innovation Squared)** in QA report. If many samples exceed threshold, increase `measurement_noise_pos`.

### Adaptive Measurement Noise

**Purpose:** Scale measurement trust based on DLC confidence or LED geometry quality.

**Built-in scaling:** linear (not squared) reciprocal of clipped DLC
confidence, implemented by `confidence_to_R_diagonal` in
`trodestrack.models.filter_common`:
```python
R_scaled = R_base / clip(confidence, clip_min, 1.0)
```

**Additional heading noise scaling:**

```python
cfg = EKFConfig(
    adaptive_heading_noise=True            # Scale heading R with LED geometry
)
```

**Effect:** Poor LED geometry (near-collinear) → higher heading noise → less trust.

### Iterated EKF (IEKF)

**Purpose:** Improve linearization accuracy for highly nonlinear measurements (e.g., heading).

```python
cfg = EKFConfig(num_iter=3)  # DEFAULT: 1 (standard EKF)
```

**Cost:** 3× slower per update.

**When to use:**
- Large heading corrections (> 30°)
- Rapid rotations with sparse camera updates
- UKF too slow but EKF linearization insufficient

---

## Parameter Reference

### Complete FilterCoreConfig

```python
from trodestrack.models.ekf import EKFConfig

cfg = EKFConfig(
    # --- Process Noise ---
    process_noise_pos=1e-4,                # Position random walk (m²/s)
    process_noise_vel=5e-3,                # Velocity random walk (m²/s³)
    process_noise_heading=5e-4,            # Heading random walk (rad²/s)
    process_noise_gyro_bias=5e-8,          # Gyro bias random walk (rad²/s³)
    process_noise_accel_bias=2e-5,         # Accel bias random walk (m²/s⁵)

    # --- Measurement Noise ---
    measurement_noise_pos=0.01**2,         # Position measurement variance (m²)
    measurement_noise_heading=0.05**2,     # Heading measurement variance (rad²)

    # --- IMU Noise (SpikeGadgets product manual) ---
    imu_gyro_noise_density=0.00017453,     # Gyro noise density (rad/s/√Hz, 0.01°/s/√Hz)
    imu_accel_noise_density=0.00196133,    # Accel noise density (m/s²/√Hz, 0.2 mg/√Hz)
    imu_gravity_body=(0.0, 0.0, 9.81),     # World-frame gravity vector (m/s²)

    # --- Dynamics ---
    damping_coeff=0.2,                     # Velocity damping λ (1/s)
    led_distance=0.04,                     # LED spacing (m); set to None to auto-detect

    # --- Mahalanobis Gating ---
    use_mahalanobis_gating=True,           # Default: reject outliers (3σ)
    mahalanobis_threshold_prob=0.997,      # Reject beyond 3σ

    # --- Heading Measurement ---
    use_heading_measurement=True,          # Use dual-LED heading constraint
    led_distance_tolerance=0.3,            # Reject if spacing > 30% off
    adaptive_heading_noise=True,           # Scale noise with geometry quality

    # --- Adaptive Q During Dropout ---
    adaptive_q_during_dropout=True,        # Inflate Q during camera blackout
    dropout_q_pos_multiplier=2.0,          # Position uncertainty multiplier
    dropout_q_vel_multiplier=2.0,          # Velocity uncertainty multiplier
    dropout_q_bias_multiplier=0.5,         # Bias drift multiplier
    freeze_bias_during_blackout=True,      # Hold bias estimates during dropout
    reduce_imu_noise_during_blackout=True, # Tighten IMU noise during dropout
    blackout_imu_noise_scale=0.3,          # IMU-noise scale during dropout

    # --- 6-DOF Orientation (used only by quaternion layouts) ---
    enable_experimental_accel_translation=False,   # Off by default in 2D modes
    use_gravity_orientation_update=True,           # Gated gravity-direction update
    gravity_orientation_measurement_noise=0.05**2,
    gravity_accel_magnitude_tolerance_m_s2=0.5,
    gravity_gyro_norm_threshold_rad_s=0.2,

    # --- Zero-Velocity Update (ZUPT) ---
    enable_zupt=True,                      # Enable stationary detection
    zupt_velocity_threshold=0.02,          # Trigger if |v| < 2 cm/s
    zupt_measurement_noise=0.01**2,        # ZUPT measurement noise (m²/s²)

    # --- Advanced ---
    num_iter=1,                            # IEKF iterations (1 = standard EKF)
    state_mode="2d_cam_3d_imu",            # Default 10D layout
)
```

### Quick Reference: What to Tune First

| Issue | Parameter | Typical Change |
|-------|-----------|----------------|
| NEES > 4.0 | `process_noise_pos` | Increase 2-5× |
| NEES < 1.0 | `process_noise_pos` | Decrease 2× |
| Velocity lag | `process_noise_vel` | Increase 2× |
| Heading drift | `use_heading_measurement` | Set to `True` |
| Stationary drift | `enable_zupt` | Set to `True` |
| Jittery tracking | `measurement_noise_pos` | Increase 2× |
| Occlusion drift | `adaptive_q_during_dropout` | Set to `True` |
| Frequent outliers | `use_mahalanobis_gating` | Set to `True` |

---

## Best Practices

1. **Always start with default parameters** → Generate baseline QA report
2. **Tune one parameter at a time** → Isolate effects
3. **Use NEES as primary metric** → Accuracy follows consistency
4. **Validate on multiple sessions** → Avoid overfitting to one dataset
5. **Document final parameters** → Save config YAML for reproducibility
6. **Re-tune if hardware changes** → New IMU, new camera, new LED attachment
7. **Use smoother for offline analysis** → Corrects filter overconfidence

---

## Troubleshooting

### NEES Won't Converge

**Possible causes:**
1. **Unmodeled dynamics** → Try UKF instead of EKF
2. **LED swaps** → Enable heading measurement or manual swap correction
3. **Time sync issues** → Check IMU/camera timestamp alignment
4. **Sensor calibration** → Verify IMU units (see PRD Section 5)

### Position Accurate but NEES High

**Diagnosis:** Covariance is too small (overconfident); the filter's
reported uncertainty is tighter than the actual error magnitude warrants.

**Solution:** Increase process noise (Q) or measurement noise (R) so the
covariance grows to match observed error.

### Position Inaccurate but NEES Low

**Diagnosis:** Systematic error (bias, calibration, modeling error).

**Solutions:**
- Check IMU bias convergence plots
- Verify camera calibration (homography)
- Check LED distance auto-detection

### Filter Diverges

**Diagnosis:** Numerical instability or severe model mismatch.

**Solutions:**
1. Check input data for `NaN` or `Inf` values
2. Ensure camera measurements are in meters (not pixels)
3. Increase process noise by 10× as sanity check
4. Enable Joseph-form updates (built-in, but verify)

---

## Further Reading

- **[Examples](../examples/index.md)** - See Examples 03-04 for NEES interpretation
- **[Troubleshooting](troubleshooting.md)** - Detailed failure mode analysis
- **Särkkä (2013)** - "Bayesian Filtering and Smoothing" (textbook reference)

---

**Questions?** Open an issue on GitHub with your QA report and parameter configuration.
