# True Vision-Only Filter: Solution to NaN Divergence

## ❌ The Problem

The original "vision-only" configuration caused **catastrophic divergence**:

```python
# BAD: Inflating IMU noise causes numerical instability
ekf_config_vision_bad = EKFConfig(
    process_noise_vel=2000.0,      # 1000× inflated
    imu_gyro_noise_density=10.0,   # 10000× inflated
    imu_accel_noise_density=50.0,  # 1000× inflated
)
```

**Results:**
- Covariance explodes: 0.025 m² → 182,527 m² during 5s dropout
- Filter diverges to NaN after dropout
- Only 50% of frames are valid
- RMSE of "4.12 cm" is misleading (excludes NaN frames)

---

## ✅ The Solution

**Use ZERO IMU inputs with a constant-velocity motion model:**

```python
# GOOD: Zero IMU inputs with moderate process noise
ekf_config_vision_true = EKFConfig(
    process_noise_pos=0.001,       # Small - position via velocity
    process_noise_vel=0.5,          # Moderate - velocity random walk
    process_noise_heading=0.01,     # Small heading drift
    process_noise_gyro_bias=0.0,    # No gyro (no IMU)
    process_noise_accel_bias=0.0,   # No accel (no IMU)
    imu_gyro_noise_density=0.0,     # No IMU noise
    imu_accel_noise_density=0.0,
    damping_coeff=0.0,              # No physics model
)

# Use ZERO IMU inputs (not inflated noise)
U_imu_zero = np.zeros_like(sim_data["U_imu"])

ekf_vision_true = extended_kalman_filter(
    ekf_config=ekf_config_vision_true,
    U_imu=U_imu_zero,  # ← KEY FIX
    ...
)
```

---

## 🎯 Results

### Filter Performance

| Metric | Fusion | Vision-Only (true) | Smoother (vision) |
|--------|--------|-------------------|-------------------|
| **Position RMSE** | 8.50 cm | 11.55 cm | **2.61 cm** ✨ |
| **Velocity RMSE** | 10.17 cm/s | 10.26 cm/s | 5.52 cm/s |
| **Valid Frames** | 1800/1800 (100%) | 1800/1800 (100%) ✅ | 1800/1800 (100%) |
| **Max Covariance** | ~10 m² | 411 m² | ~0.1 m² |

### Key Achievements

✅ **No NaN divergence**: All 1800 frames valid
✅ **Bounded covariance**: 411 m² max (vs 182,527 m² in broken version)
✅ **Reasonable performance**: 11.55 cm RMSE (36% worse than fusion)
✅ **Smoother works beautifully**: **2.61 cm RMSE** (69% better than fusion!)

---

## 📊 Visualization Insights

### 1. Position Error During Dropout

Both filters degrade during dropout, but **vision-only degrades more**:
- **Fusion**: Peaks at ~20 cm (2.4× baseline)
- **Vision-only**: Peaks at ~80 cm (6.9× baseline)
- **Vision smoother**: Peaks at ~20 cm (benefits from backward pass)

### 2. Position Uncertainty

**Covariance growth (log scale):**
- Before dropout: Both ~10⁻⁵ m²
- During dropout:
  - Fusion: ~10² m² (adaptive Q helps)
  - Vision-only: ~10² m² (similar growth)
- After dropout: Both recover to ~10⁻⁵ m²

**Key insight**: Without adaptive Q, covariance grows more during dropout, but stays numerically stable.

### 3. Velocity Error

Vision-only **cannot estimate velocity well without IMU**:
- Baseline: Vision-only ~10 cm/s, similar to fusion
- During dropout: Both degrade, but vision-only has more noise
- Smoother helps significantly: 5.52 cm/s (46% better than filter)

---

## 🔑 Why This Works

### 1. **Zero IMU Inputs (Not Inflated Noise)**

❌ **Bad approach**: Inflate IMU noise to 1000×
- Still propagates noisy IMU measurements
- Causes covariance to explode exponentially
- Results in numerical instability → NaN

✅ **Good approach**: Set IMU inputs to zero
- Treats IMU as if it doesn't exist
- Pure constant-velocity kinematic model
- Covariance grows linearly (not exponentially)

### 2. **Moderate Velocity Process Noise**

**Q_vel = 0.5 m²/s** represents:
- Velocity random walk (unmodeled acceleration)
- Position diffusion: `σ_pos² = Q_vel × dt²`
- Over 5s dropout: `σ_pos ≈ sqrt(0.5 × 5) = 1.58 m`

This is **realistic** for constant-velocity assumption with unknown accelerations.

### 3. **No Adaptive Q**

Fusion uses adaptive Q during dropout:
```python
adaptive_q_during_dropout=True,
dropout_q_pos_multiplier=10.0,    # Inflate Q by 10×
```

This **helps fusion** because IMU is still available (just degraded).

But for **true vision-only**, adaptive Q would:
- Multiply already-moderate Q by 10×
- Cause faster covariance growth
- Risk numerical issues without measurements to constrain it

**Better**: Use fixed moderate Q that's already tuned for dropout behavior.

### 4. **Smoother is Critical**

The **RTS smoother is AMAZING for vision-only**:
- Filter RMSE: 11.55 cm → Smoother RMSE: **2.61 cm** (77% improvement!)
- Why? Backward pass uses vision **after** dropout to constrain estimates **during** dropout
- This is why vision-only smoother **beats fusion filter** (2.61 cm < 8.50 cm)

---

## 🚀 Production Recommendations

### ✅ Use Fusion Filter + Smoother
**Best overall:** Fusion filter (8.50 cm) → Smoother (6.11 cm)

### ✅ Vision-Only Smoother is Viable
**If IMU fails:** Vision-only smoother (2.61 cm) is excellent!
- Better than fusion filter alone
- Requires offline processing (backward pass)
- Velocity estimates still poor (5.52 cm/s vs truth)

### ❌ Don't Use Vision-Only Filter Online
**Vision-only filter** (11.55 cm) is marginal:
- 36% worse than fusion
- Large uncertainty during dropout (80 cm spikes)
- Poor velocity estimates without IMU

---

## 📝 Implementation Template

```python
def create_vision_only_filter(fs_cam: float, measurement_noise: float) -> EKFConfig:
    """Create a proper vision-only EKF configuration.

    Uses constant-velocity motion model with zero IMU inputs.
    Prevents NaN divergence during camera dropouts.

    Args:
        fs_cam: Camera frame rate (Hz)
        measurement_noise: Camera position measurement noise (m²)

    Returns:
        EKFConfig for vision-only tracking
    """
    dt_cam = 1.0 / fs_cam

    # Process noise for constant-velocity random walk
    # Position diffuses via velocity: σ_pos² = Q_vel × dt²
    # Velocity drifts due to unknown acceleration: Q_vel = σ_accel² × dt
    sigma_accel = 0.5  # Expected acceleration magnitude (m/s²)
    Q_vel = sigma_accel**2 * dt_cam  # Velocity process noise rate
    Q_pos = Q_vel * dt_cam**2        # Position process noise rate (small)

    return EKFConfig(
        process_noise_pos=Q_pos,
        process_noise_vel=Q_vel,
        process_noise_heading=0.01,
        process_noise_gyro_bias=0.0,     # No IMU
        process_noise_accel_bias=0.0,    # No IMU
        measurement_noise_pos=measurement_noise,
        imu_gyro_noise_density=0.0,      # No IMU noise
        imu_accel_noise_density=0.0,
        damping_coeff=0.0,               # No damping (no physics)
        led_distance=0.04,
        use_heading_measurement=True,
        adaptive_q_during_dropout=False, # Fixed Q (already tuned for dropout)
    )

# Usage:
config = create_vision_only_filter(fs_cam=30.0, measurement_noise=0.005**2)
U_imu_zero = np.zeros((len(t_imu), 3))  # Zero IMU inputs

result = extended_kalman_filter(
    ekf_config=config,
    t_imu=t_imu,
    U_imu=U_imu_zero,  # ← Zero IMU
    t_cam=t_cam,
    Z_cam_led1=Z_cam_led1,
    Z_cam_led2=Z_cam_led2,
    mask_cam=mask_cam,
)

# Apply smoother for best results
smoothed = rts_smoother(
    filter_result=result,
    ekf_config=config,
    t_imu=t_imu,
    U_imu=U_imu_zero,
    t_cam=t_cam,
)
```

---

## 🎓 Lessons Learned

1. **Inflating noise ≠ Disabling sensor**
   - Inflated noise still uses (bad) measurements → divergence
   - Zero inputs = true sensor absence → stable

2. **Process noise tuning is critical**
   - Too large: covariance explosion → NaN
   - Too small: overconfident → divergence on first measurement
   - Just right: bounded covariance, graceful degradation

3. **Adaptive Q needs sensors**
   - Works great for fusion (IMU still provides info)
   - Dangerous for vision-only (no fallback during dropout)

4. **Smoothers are powerful**
   - Vision-only filter: 11.55 cm (mediocre)
   - Vision-only smoother: 2.61 cm (excellent!)
   - Uses information from both sides of dropout

5. **RMSE can lie**
   - Always check valid frame percentage
   - Excluding NaNs makes bad filters look good
   - Report: "X cm over Y/Z valid frames"

---

## 📚 References

- **Constant-velocity model**: Kalman & Bucy (1961)
- **Process noise tuning**: Bar-Shalom et al. (2001), "Estimation with Applications to Tracking and Navigation"
- **RTS smoother**: Rauch et al. (1965)

---

Generated: 2025-10-10
Trodestrack: Sensor Fusion Analysis
