# trodestrack Extensibility Analysis

**Date:** 2025-10-10
**Analyst:** Claude Code
**Question:** How extensible is the code to (1) full 3D tracking, (2) 2D camera + 3D accelerometer, (3) IMU-only, (4) vision-only?

---

## Executive Summary

**Overall Rating:** ⭐⭐⭐⭐☆ (4/5) - **Very Extensible**

The trodestrack codebase is **well-architected for extensibility**, with key abstractions that support multiple sensor configurations and state dimensions. The recent shared filter core refactor (filter_common.py) eliminated hard-coded 8D assumptions in critical paths. However, some tactical changes are needed for each scenario.

**Extension Difficulty:**

1. **Vision-only:** ⭐⭐⭐⭐⭐ (Easiest - 2-4 hours)
2. **IMU-only:** ⭐⭐⭐⭐☆ (Easy - 4-8 hours)
3. **2D camera + 3D accel:** ⭐⭐⭐☆☆ (Moderate - 1-2 weeks)
4. **Full 3D:** ⭐⭐☆☆☆ (Complex - 3-4 weeks)

---

## Architecture Analysis

### ✅ Extensibility Strengths

#### 1. **Dimension-Agnostic Smoothers** (P0.4 from REVIEW.md - COMPLETED)

```python
# runtime/offline.py:63-110
def build_Q_rate(config, n: int) -> jnp.ndarray:
    """Build process noise rate matrix Q_rate for arbitrary state dimension.

    For standard 8D state: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    For future 3D state: [x, y, z, vx, vy, vz, roll, pitch, yaw, biases...]
    """
    if n == 8:
        # Standard 2D mapping
    else:
        # Generic fallback (uniform noise)
```

**Impact:** RTS and sigma-point smoothers already support arbitrary state dimensions (4, 6, 8, 10, 12 tested).

#### 2. **Shared Core Abstractions**

```python
# models/filter_common.py:52-56
class FilterState(NamedTuple):
    """Kalman filter state comprising mean vector and covariance matrix."""
    mean: jnp.ndarray  # (n,) - NO HARDCODED DIMENSION
    cov: jnp.ndarray   # (n, n)
```

**Impact:** State representation is dimension-agnostic, making n=12 (3D) straightforward.

#### 3. **Modular Measurement Functions**

```python
# models/filter_common.py:146-154
def measurement_function(state: jnp.ndarray, led_distance: float) -> jnp.ndarray:
    """Project state into dual-LED measurement space."""
    px, py = state[0], state[1]  # Only uses first 2 dims
    theta = state[4]              # Only uses heading dim
    # ... computes 4D LED measurement
```

**Impact:** Measurement function only accesses position and heading, agnostic to velocity or 3D structure.

#### 4. **Pluggable Dynamics**

```python
# models/filter_common.py:109-143
def dynamics_function(state: jnp.ndarray, imu: jnp.ndarray, dt: float, damping: float):
    """Integrate constant-acceleration dynamics with linear damping."""
    px, py, vx, vy, theta, b_gz, b_ax, b_ay = state
    omega_z, fx, fy = imu
    # ... 2D-specific integration
```

**Impact:** Dynamics function is isolated—easy to swap for 3D version.

---

### ⚠️ Extensibility Limitations

#### 1. **Hard-Coded 8D Assumptions in Initialization**

```python
# models/filter_common.py:216-227
mean_init = jnp.array([
    pos_init[0], pos_init[1],    # x, y (ok for 3D, just add z)
    vel_init[0], vel_init[1],    # vx, vy (need vz)
    heading_init,                 # θ (need roll, pitch)
    0.0, 0.0, 0.0,               # b_gz, b_ax, b_ay (need 3D biases)
])
```

**Impact:** `initialize_state()` needs separate 3D version or parameterization.

#### 2. **2D Rotation in Dynamics**

```python
# models/filter_common.py:121-124
cos_theta = jnp.cos(theta)
sin_theta = jnp.sin(theta)
rotation = jnp.array([[cos_theta, -sin_theta],
                       [sin_theta, cos_theta]])  # 2D rotation only
accel_world = rotation @ accel_body
```

**Impact:** 3D needs full 3×3 rotation matrix (SO(3)) from quaternion or Euler angles.

#### 3. **Hard-Coded IMU Input Dimensions**

```python
# models/filter_common.py:115
omega_z, fx, fy = imu  # Assumes 3-element IMU: [ω_z, f_x, f_y]
```

**Impact:** 3D IMU has 6 elements: `[ω_x, ω_y, ω_z, f_x, f_y, f_z]`

#### 4. **LED Measurement Model (2D-Only)**

```python
# models/filter_common.py:146-154
def measurement_function(state, led_distance):
    px, py = state[0], state[1]  # 2D position only
    theta = state[4]              # 2D heading only
    dx = 0.5 * led_distance * jnp.cos(theta)
    dy = 0.5 * led_distance * jnp.sin(theta)
    return jnp.array([px - dx, py - dy, px + dx, py + dy])  # 4D (x,y for 2 LEDs)
```

**Impact:** 3D camera would measure `[x, y, z]` for each LED (6D measurement).

#### 5. **Q_rate Fallback for Non-8D States**

```python
# runtime/offline.py:105-110
else:
    # For non-standard dimensions, use uniform noise
    # TODO(P1): For future 3D/custom states, accept explicit noise vector
    return jnp.diag(jnp.full(n, config.process_noise_pos))
```

**Impact:** Non-8D states get uniform noise (too low for velocity/angular states).

---

## Scenario 1: Vision-Only (No IMU)

**Difficulty:** ⭐⭐⭐⭐⭐ (Easiest)
**Effort:** 2-4 hours
**State:** `[x, y, vx, vy, θ]` (5D, no biases)

### What Works Out-of-the-Box

- ✅ Smoothers already support n=5
- ✅ Measurement function only uses position/heading
- ✅ EKF/UKF prediction can skip IMU pre-integration

### Required Changes

#### 1. **New Dynamics (No IMU Integration)**

```python
def dynamics_function_vision_only(
    state: jnp.ndarray,
    dt: float,
    damping: float
) -> jnp.ndarray:
    """Constant-velocity dynamics with damping (no IMU)."""
    px, py, vx, vy, theta = state

    # Constant velocity with damping
    vel_next = jnp.array([vx, vy]) * (1 - damping * dt)
    pos_next = jnp.array([px, py]) + vel_next * dt
    theta_next = theta  # No gyro, heading is latent

    return jnp.array([
        pos_next[0], pos_next[1],
        vel_next[0], vel_next[1],
        theta_next
    ])
```

#### 2. **New Initialization (5D)**

```python
def initialize_state_vision_only(...) -> FilterState:
    """Initialize 5D state [x, y, vx, vy, θ] from LED observations."""
    # Same logic as current, just drop bias dims
    mean_init = jnp.array([
        pos_init[0], pos_init[1],
        vel_init[0], vel_init[1],
        heading_init
    ])

    cov_init = jnp.diag(jnp.array([
        0.01**2, 0.01**2,  # position
        0.1**2, 0.1**2,     # velocity
        (jnp.pi/4)**2       # heading
    ]))

    return FilterState(mean=mean_init, cov=cov_init)
```

#### 3. **Q_rate for 5D State**

```python
# In build_Q_rate():
if n == 5:
    return jnp.diag(jnp.array([
        config.process_noise_pos,     # x
        config.process_noise_pos,     # y
        config.process_noise_vel,     # vx
        config.process_noise_vel,     # vy
        config.process_noise_heading  # θ
    ]))
```

#### 4. **Prediction Step (Skip IMU Integration)**

```python
# In EKF predict_step():
if imu_samples is None:  # Vision-only mode
    # Single dynamics step (no IMU pre-integration)
    m_pred = dynamics_function_vision_only(m_in, dt_cam, config.damping_coeff)

    # Compute Jacobian F for 5D state
    F = compute_jacobian_5d(m_in, dt_cam, config.damping_coeff)

    # Process noise (5×5)
    Q_rate = build_Q_rate(config, n=5)
    Q = Q_rate * dt_cam

    P_pred = F @ P_in @ F.T + Q
```

### Testing

```python
def test_ekf_vision_only():
    """Test EKF with vision-only (no IMU)."""
    sim = simulate_constant_velocity(duration_s=10.0, seed=42)

    config = EKFConfig(mode="vision_only")  # New config flag
    result = extended_kalman_filter(config, **sim, imu_samples=None)

    # Check 5D state
    assert result.filtered_means.shape == (len(sim["t_cam"]), 5)

    # Position should still be accurate (camera is good)
    pos_rmse = compute_position_rmse(sim["positions_true"], result.filtered_means[:, :2])
    assert pos_rmse < 0.01  # Better than 1cm (no IMU drift!)
```

**Pros:**

- Simpler model (5D vs 8D)
- No IMU bias drift issues
- Faster computation (fewer states)

**Cons:**

- No velocity smoothing during camera dropouts (dead reckoning fails)
- Heading is latent (less observable without gyro)

---

## Scenario 2: IMU-Only (No Camera)

**Difficulty:** ⭐⭐⭐⭐☆ (Easy)
**Effort:** 4-8 hours
**State:** `[x, y, vx, vy, θ, b_gz, b_ax, b_ay]` (8D, same as current)

### What Works Out-of-the-Box

- ✅ State dimension unchanged (8D)
- ✅ Dynamics function unchanged (IMU pre-integration)
- ✅ All infrastructure works

### Required Changes

#### 1. **New Measurement Function (Zero-Velocity Pseudo-Measurement)**

IMU-only tracking relies on:

- **Zero-velocity updates (ZUPT)** during stationary periods
- **Pseudo-measurements** of expected behavior (e.g., bounded velocity)

```python
def measurement_function_imu_only(state: jnp.ndarray) -> jnp.ndarray:
    """Pseudo-measurement: velocity magnitude should be reasonable."""
    vx, vy = state[2], state[3]
    v_mag = jnp.sqrt(vx**2 + vy**2)

    # Return velocity magnitude as 1D measurement
    return jnp.array([v_mag])

def measurement_jacobian_imu_only(state: jnp.ndarray) -> jnp.ndarray:
    """H for velocity magnitude measurement."""
    vx, vy = state[2], state[3]
    v_mag = jnp.sqrt(vx**2 + vy**2) + 1e-8  # Avoid singularity

    H = jnp.zeros((1, 8))
    H = H.at[0, 2].set(vx / v_mag)  # ∂v_mag/∂vx
    H = H.at[0, 3].set(vy / v_mag)  # ∂v_mag/∂vy
    return H
```

#### 2. **Update Step (No Camera)**

```python
# In EKF update_step():
if mode == "imu_only":
    # Option A: ZUPT only (stationary detection)
    state, ll_zupt = update_zupt(state, config)

    # Option B: Velocity magnitude constraint
    z_obs = jnp.array([0.3])  # Expected velocity ~30 cm/s
    H = measurement_jacobian_imu_only(state.mean)
    # ... standard EKF update
```

#### 3. **Initialization (No Camera Observations)**

```python
def initialize_state_imu_only() -> FilterState:
    """Initialize 8D state without camera observations."""
    mean_init = jnp.array([
        0.0, 0.0,      # Start at origin (unknown position)
        0.0, 0.0,      # Start stationary (unknown velocity)
        0.0,           # Unknown heading
        0.0, 0.0, 0.0  # Zero bias initial guess
    ])

    cov_init = jnp.diag(jnp.array([
        10.0**2, 10.0**2,  # Large position uncertainty (no camera!)
        0.5**2, 0.5**2,     # Moderate velocity uncertainty
        jnp.pi**2,          # Large heading uncertainty
        0.1**2, 1.0**2, 1.0**2  # Bias uncertainties
    ]))

    return FilterState(mean=mean_init, cov=cov_init)
```

### Challenges

**Drift:** Without vision corrections, IMU-only tracking drifts due to:

1. **Bias estimation errors** - Accelerometer/gyro biases slowly drift
2. **Integration errors** - Double integration of acceleration → position error grows quadratically
3. **Heading uncertainty** - Gyro drift causes heading errors, which corrupt velocity rotation

**Mitigation Strategies:**

- **ZUPT (Zero-Velocity Updates):** Already implemented! Essential for IMU-only.
- **Magnetic heading:** Use magnetometer to constrain yaw (PRD §15 roadmap)
- **Arena bounds:** Use geometric constraints to limit drift
- **Stride detection:** For walking animals, use periodic gait patterns

### Testing

```python
def test_ekf_imu_only_with_zupt():
    """Test IMU-only tracking with ZUPT."""
    # Simulate rat that pauses frequently (realistic behavior)
    sim = simulate_rat_with_pauses(duration_s=30.0, pause_prob=0.3, seed=42)

    config = EKFConfig(
        mode="imu_only",
        enable_zupt=True,             # Critical for IMU-only
        zupt_velocity_threshold=0.05   # 5 cm/s
    )

    result = extended_kalman_filter(config, **sim, Z_cam_led1=None, Z_cam_led2=None)

    # IMU-only drift will be large without ZUPT
    pos_rmse = compute_position_rmse(sim["positions_true"], result.filtered_means[:, :2])

    # Expect ~1m drift over 30s (vs 10cm with camera)
    assert pos_rmse < 1.0  # Acceptable for IMU-only

    # Heading should be better (gyro bias converges)
    heading_rmse = compute_heading_rmse(sim["heading_true"], result.filtered_means[:, 4])
    assert heading_rmse < np.deg2rad(20)  # ±20° acceptable
```

**Pros:**

- High-rate tracking (1+ kHz possible)
- Works in complete darkness or occlusions
- Smooth trajectories (no camera frame rate limit)

**Cons:**

- Position drift (1-10 m over 1 minute)
- Requires frequent ZUPT opportunities
- Heading accumulates errors

---

## Scenario 3: 2D Camera + 3D Accelerometer

**Difficulty:** ⭐⭐⭐☆☆ (Moderate)
**Effort:** 1-2 weeks
**State:** `[x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]` (10D)

### Motivation

Rat tracking with:

- **2D camera** (overhead view, measures x, y)
- **3D IMU** (accelerometer measures [f_x, f_y, f_z], gyro measures ω_z only)
- Useful for detecting **rearing** (rat stands on hind legs, z-velocity ≠ 0)

### Required Changes

#### 1. **New State Representation (10D)**

```python
# State: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
#        [0, 1,  2,  3,  4, 5,    6,    7,    8,    9]
```

#### 2. **Dynamics Function (3D Acceleration, 2D Position)**

```python
def dynamics_function_2d_cam_3d_imu(
    state: jnp.ndarray,
    imu: jnp.ndarray,
    dt: float,
    damping: float
) -> jnp.ndarray:
    """2D position with 3D velocity and acceleration."""
    px, py, vx, vy, vz, theta, b_gz, b_ax, b_ay, b_az = state
    omega_z, fx, fy, fz = imu  # 4-element IMU

    # Unbiased IMU
    omega_z_unbiased = omega_z - b_gz
    accel_body = jnp.array([fx - b_ax, fy - b_ay, fz - b_az])  # 3D

    # 2D rotation for heading
    theta_next = theta + omega_z_unbiased * dt
    cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)

    # Rotate horizontal acceleration to world frame (2D)
    accel_world_xy = jnp.array([
        cos_theta * accel_body[0] - sin_theta * accel_body[1],
        sin_theta * accel_body[0] + cos_theta * accel_body[1]
    ])

    # Vertical acceleration (no rotation, gravity-subtracted in preprocessing)
    accel_world_z = accel_body[2]

    # Update velocities (3D)
    vel = jnp.array([vx, vy, vz])
    vel_next = vel + jnp.array([
        accel_world_xy[0] * dt,
        accel_world_xy[1] * dt,
        accel_world_z * dt
    ]) - damping * vel * dt

    # Update position (2D only, vz doesn't integrate to position)
    pos = jnp.array([px, py])
    pos_next = pos + vel[:2] * dt + 0.5 * accel_world_xy * dt**2

    return jnp.array([
        pos_next[0], pos_next[1],      # x, y
        vel_next[0], vel_next[1], vel_next[2],  # vx, vy, vz
        theta_next,                     # θ
        b_gz, b_ax, b_ay, b_az         # biases
    ])
```

#### 3. **Measurement Function (2D Camera, No Z)**

```python
def measurement_function_2d_cam_3d_imu(state: jnp.ndarray, led_distance: float) -> jnp.ndarray:
    """Project state into 2D LED measurements (ignore vz)."""
    px, py = state[0], state[1]  # 2D position
    theta = state[5]              # Heading (index shifted due to vz)

    dx = 0.5 * led_distance * jnp.cos(theta)
    dy = 0.5 * led_distance * jnp.sin(theta)

    return jnp.array([px - dx, py - dy, px + dx, py + dy])  # Still 4D (2 LEDs × 2D)
```

#### 4. **Observability Analysis**

**Observable States:**

- x, y (from camera)
- vx, vy (from camera + accel_xy)
- θ (from LEDs + gyro)
- b_gz (from gyro + heading)
- b_ax, b_ay (from accel_xy + position)

**Weakly Observable:**

- vz (from accel_z only, no position measurement)
- b_az (couples with vz, hard to separate)

**Implications:**

- vz and b_az uncertainties will be large (~0.5 m/s, 0.5 m/s²)
- Need ZUPT or vertical constraint to improve observability
- Useful for detecting **qualitative** vertical motion (rearing), not precise z-velocity

#### 5. **Q_rate for 10D State**

```python
# In build_Q_rate():
if n == 10:
    return jnp.diag(jnp.array([
        config.process_noise_pos,      # x
        config.process_noise_pos,      # y
        config.process_noise_vel,      # vx
        config.process_noise_vel,      # vy
        config.process_noise_vel * 2,  # vz (less observable, more noise)
        config.process_noise_heading,  # θ
        config.process_noise_gyro_bias,  # b_gz
        config.process_noise_accel_bias, # b_ax
        config.process_noise_accel_bias, # b_ay
        config.process_noise_accel_bias * 2  # b_az (less observable)
    ]))
```

### Testing

```python
def test_ekf_2d_cam_3d_imu_rearing_detection():
    """Test 2D camera + 3D IMU for vertical motion detection."""
    # Simulate rat that rears up periodically
    sim = simulate_rat_with_rearing(duration_s=30.0, seed=42)
    # sim["U_imu"] is now 4D: [omega_z, fx, fy, fz]
    # sim["Z_cam"] is still 4D: [led1_x, led1_y, led2_x, led2_y]

    config = EKFConfig(mode="2d_cam_3d_imu")
    result = extended_kalman_filter(config, **sim)

    # Check 10D state
    assert result.filtered_means.shape == (len(sim["t_cam"]), 10)

    # Horizontal position should be accurate
    pos_rmse = compute_position_rmse(sim["positions_true"][:, :2], result.filtered_means[:, :2])
    assert pos_rmse < 0.02  # 2cm target

    # Vertical velocity should detect rearing (qualitative)
    vz_filtered = result.filtered_means[:, 4]
    rearing_frames = sim["is_rearing"]

    # During rearing, vz should be positive (moving up)
    assert np.mean(vz_filtered[rearing_frames]) > 0.1  # >10 cm/s upward

    # During normal locomotion, vz should be near zero
    normal_frames = ~rearing_frames
    assert np.abs(np.mean(vz_filtered[normal_frames])) < 0.05  # <5 cm/s
```

**Pros:**

- Detects vertical motion (rearing, jumping)
- Better velocity estimates (3D accel helps)
- Still has camera for position

**Cons:**

- vz and b_az poorly observable (large uncertainties)
- More complex state (10D vs 8D)
- Requires 3D IMU simulation

---

## Scenario 4: Full 3D Tracking

**Difficulty:** ⭐⭐☆☆☆ (Complex)
**Effort:** 3-4 weeks
**State:** `[x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]` (16D with quaternion) or 15D with Euler angles

### Motivation

Full 6-DOF pose tracking:

- **3D camera** (stereo or depth camera, measures x, y, z)
- **6-axis IMU** (3-axis gyro + 3-axis accel)
- **Orientation:** Roll, pitch, yaw (or quaternion)

### Challenges

#### 1. **Orientation Representation**

**Option A: Euler Angles (Roll, Pitch, Yaw)**

```python
# State: [x, y, z, vx, vy, vz, roll, pitch, yaw, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]
# Dimension: 15D
```

**Pros:**

- Intuitive
- Fewer states (15D vs 16D)

**Cons:**

- **Gimbal lock** at pitch = ±90°
- Singularities in Jacobian
- Angle wrapping complications

**Option B: Quaternion**

```python
# State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]
# Dimension: 16D
```

**Pros:**

- No gimbal lock
- Smooth, well-defined dynamics
- Preferred for aerospace/robotics

**Cons:**

- More states (16D)
- Normalization constraint (|q| = 1)
- Less intuitive

**Recommendation:** Use **quaternion** for filter, convert to Euler for output/visualization.

#### 2. **Dynamics Function (3D Kinematics)**

```python
def dynamics_function_3d(
    state: jnp.ndarray,
    imu: jnp.ndarray,
    dt: float,
    damping: float
) -> jnp.ndarray:
    """Full 3D dynamics with quaternion orientation."""
    # State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    bias_gyro = state[10:13]
    bias_accel = state[13:16]

    # IMU: [ω_x, ω_y, ω_z, f_x, f_y, f_z]
    omega_body = imu[0:3] - bias_gyro
    accel_body = imu[3:6] - bias_accel

    # Quaternion derivative: q̇ = 0.5 * q ⊗ ω_body
    quat_dot = 0.5 * quaternion_multiply(quat, jnp.array([0, *omega_body]))
    quat_next = quat + quat_dot * dt
    quat_next = quat_next / jnp.linalg.norm(quat_next)  # Normalize

    # Rotate acceleration to world frame
    R_body_to_world = quaternion_to_rotation_matrix(quat)
    accel_world = R_body_to_world @ accel_body

    # Integrate velocity and position
    vel_next = vel + accel_world * dt - damping * vel * dt
    pos_next = pos + vel * dt + 0.5 * accel_world * dt**2

    return jnp.concatenate([
        pos_next,      # x, y, z
        vel_next,      # vx, vy, vz
        quat_next,     # qw, qx, qy, qz
        bias_gyro,     # b_gx, b_gy, b_gz
        bias_accel     # b_ax, b_ay, b_az
    ])
```

#### 3. **Measurement Function (3D Camera)**

```python
def measurement_function_3d(state: jnp.ndarray, led_distance: float) -> jnp.ndarray:
    """Project state into 3D LED measurements (stereo camera)."""
    pos = state[0:3]  # [x, y, z]
    quat = state[6:10]

    # Rotate LED offset vector from body to world frame
    R = quaternion_to_rotation_matrix(quat)
    led_offset_body = jnp.array([led_distance / 2, 0, 0])  # LED along x-axis in body frame

    led1_world = pos - R @ led_offset_body
    led2_world = pos + R @ led_offset_body

    return jnp.array([*led1_world, *led2_world])  # 6D measurement
```

#### 4. **Jacobians (Much More Complex)**

EKF requires Jacobians for 16D state:

- **F (dynamics Jacobian):** 16×16 matrix with quaternion kinematics
- **H (measurement Jacobian):** 6×16 matrix with rotation derivatives

**Computational Cost:**

- Quaternion derivatives are nonlinear → more complex Jacobians
- UKF may be preferable (no Jacobians needed, but 33 sigma points for 16D state)

#### 5. **Observability (Much Harder)**

**Observable from 3D camera + 6-axis IMU:**

- ✅ Position (x, y, z) - from camera
- ✅ Velocity (vx, vy, vz) - from camera + accel
- ✅ Orientation (roll, pitch, yaw) - from LEDs + gyro + accel (gravity vector)
- ⚠️ Gyro biases (b_gx, b_gy, b_gz) - partially observable from orientation drift
- ⚠️ Accel biases (b_ax, b_ay, b_az) - weakly observable (couples with gravity errors)

**Unobservable Modes:**

- Yaw bias and magnetometer heading (unless magnetometer added - PRD §15)
- Horizontal accel biases during straight-line motion

#### 6. **Gravity Handling (Critical)**

In 3D, **gravity** is always present and couples with orientation:

- **Roll/Pitch** affect measured gravity direction
- **Accel bias** corrupts gravity estimates
- Need to estimate **IMU-to-body frame alignment** (tilt correction)

**Approaches:**

1. **Gravity vector state augmentation:** Add g_world to state (19D)
2. **Pre-calibration:** Estimate tilt from stationary data
3. **MARG filter:** Use magnetometer + accel + gyro (common in IMUs)

#### 7. **Magnetometer Integration (PRD §15 Roadmap)**

To improve yaw observability:

```python
def measurement_function_3d_with_mag(state: jnp.ndarray) -> jnp.ndarray:
    """Add magnetometer heading measurement."""
    quat = state[6:10]

    # Expected magnetic field direction in body frame
    R = quaternion_to_rotation_matrix(quat)
    mag_world = jnp.array([1, 0, 0])  # North (assuming alignment)
    mag_body_expected = R.T @ mag_world

    return mag_body_expected  # 3D measurement
```

### Required Architectural Changes

#### 1. **Separate 3D Dynamics Module**

```python
# Create: src/trodestrack/models/dynamics_3d.py
def dynamics_function_quaternion(state, imu, dt, damping):
    """Full 3D dynamics with quaternion."""

def quaternion_multiply(q1, q2):
    """Hamilton product of quaternions."""

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3×3 rotation matrix."""

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion."""
```

#### 2. **3D Initialization**

```python
def initialize_state_3d(
    led1_obs: jnp.ndarray,  # (N, 3)
    led2_obs: jnp.ndarray,  # (N, 3)
    imu_static: jnp.ndarray,  # Stationary IMU for gravity calibration
) -> FilterState:
    """Initialize 16D state from 3D observations."""

    # Position from LEDs
    pos_init = (led1_obs[0] + led2_obs[0]) / 2

    # Orientation from gravity vector (roll, pitch from accel)
    accel_static = imu_static[3:6]
    roll_init, pitch_init = estimate_roll_pitch_from_gravity(accel_static)
    yaw_init = estimate_yaw_from_leds(led1_obs[0], led2_obs[0])

    quat_init = euler_to_quaternion(roll_init, pitch_init, yaw_init)

    mean_init = jnp.concatenate([
        pos_init,                      # x, y, z
        jnp.zeros(3),                  # vx, vy, vz
        quat_init,                     # qw, qx, qy, qz
        jnp.zeros(3),                  # b_gx, b_gy, b_gz
        jnp.zeros(3)                   # b_ax, b_ay, b_az
    ])

    # 16×16 covariance (large for orientation/biases)
    cov_init = jnp.diag(jnp.array([
        0.01, 0.01, 0.01,              # position (1cm)
        0.1, 0.1, 0.1,                 # velocity (10 cm/s)
        0.1, 0.1, 0.1, 0.1,            # quaternion (0.1 rad ≈ 6°)
        0.01, 0.01, 0.01,              # gyro bias (0.01 rad/s)
        0.1, 0.1, 0.1                  # accel bias (0.1 m/s²)
    ]) ** 2)

    return FilterState(mean=mean_init, cov=cov_init)
```

#### 3. **Q_rate for 16D State**

```python
# In build_Q_rate():
if n == 16:
    return jnp.diag(jnp.array([
        config.process_noise_pos,      # x, y, z
        config.process_noise_pos,
        config.process_noise_pos,
        config.process_noise_vel,      # vx, vy, vz
        config.process_noise_vel,
        config.process_noise_vel,
        config.process_noise_quat,     # qw, qx, qy, qz (NEW CONFIG PARAM)
        config.process_noise_quat,
        config.process_noise_quat,
        config.process_noise_quat,
        config.process_noise_gyro_bias,  # b_gx, b_gy, b_gz
        config.process_noise_gyro_bias,
        config.process_noise_gyro_bias,
        config.process_noise_accel_bias, # b_ax, b_ay, b_az
        config.process_noise_accel_bias,
        config.process_noise_accel_bias
    ]))
```

#### 4. **New Config Parameters**

```python
@dataclass
class FilterCoreConfig:
    # ... existing params ...

    # 3D-specific params
    process_noise_quat: float = 0.01      # Quaternion process noise (rad²/s)
    measurement_noise_pos_3d: float = 0.01**2  # 3D camera noise (m²)
    use_magnetometer: bool = False
    mag_declination: float = 0.0          # Magnetic declination (rad)
```

### Testing

```python
def test_ekf_full_3d():
    """Test full 3D EKF with quaternion state."""
    # Simulate 3D motion (helix trajectory)
    sim = simulate_rat_3d_helix(duration_s=30.0, seed=42)
    # sim["U_imu"]: (N_imu, 6) [ω_x, ω_y, ω_z, f_x, f_y, f_z]
    # sim["Z_cam"]: (N_cam, 6) [led1_x, led1_y, led1_z, led2_x, led2_y, led2_z]

    config = EKFConfig(mode="full_3d")
    result = extended_kalman_filter(config, **sim)

    # Check 16D state
    assert result.filtered_means.shape == (len(sim["t_cam"]), 16)

    # Position (3D)
    pos_rmse = compute_position_rmse_3d(sim["positions_true"], result.filtered_means[:, :3])
    assert pos_rmse < 0.03  # 3cm target (3D is harder)

    # Orientation (convert quaternion to Euler)
    quat_filtered = result.filtered_means[:, 6:10]
    euler_filtered = quaternion_to_euler(quat_filtered)

    roll_rmse = compute_angle_rmse(sim["roll_true"], euler_filtered[:, 0])
    pitch_rmse = compute_angle_rmse(sim["pitch_true"], euler_filtered[:, 1])
    yaw_rmse = compute_angle_rmse(sim["yaw_true"], euler_filtered[:, 2])

    assert roll_rmse < np.deg2rad(10)   # ±10° roll
    assert pitch_rmse < np.deg2rad(10)  # ±10° pitch
    assert yaw_rmse < np.deg2rad(15)    # ±15° yaw (less observable without mag)
```

---

## Recommendations for Extensibility

### Immediate Actions (For Any Extension)

#### 1. **Parameterize State Dimension**

```python
# Add to FilterCoreConfig:
@dataclass
class FilterCoreConfig:
    state_mode: str = "2d_full"  # "2d_full", "vision_only", "imu_only", "2d_cam_3d_imu", "3d_full"

    @property
    def state_dim(self) -> int:
        """Return state dimension based on mode."""
        return {
            "2d_full": 8,          # [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
            "vision_only": 5,      # [x, y, vx, vy, θ]
            "imu_only": 8,         # Same as 2d_full
            "2d_cam_3d_imu": 10,   # [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
            "3d_full": 16          # [x,y,z, vx,vy,vz, qw,qx,qy,qz, b_gx,b_gy,b_gz, b_ax,b_ay,b_az]
        }[self.state_mode]
```

#### 2. **Dynamics Dispatch**

```python
def get_dynamics_function(mode: str):
    """Return appropriate dynamics function for mode."""
    return {
        "2d_full": dynamics_function_2d,
        "vision_only": dynamics_function_vision_only,
        "imu_only": dynamics_function_2d,  # Same as 2d_full
        "2d_cam_3d_imu": dynamics_function_2d_cam_3d_imu,
        "3d_full": dynamics_function_3d_quaternion
    }[mode]
```

#### 3. **Measurement Dispatch**

```python
def get_measurement_function(mode: str):
    """Return appropriate measurement function for mode."""
    return {
        "2d_full": measurement_function_2d,
        "vision_only": measurement_function_2d,
        "imu_only": measurement_function_imu_only,  # ZUPT or velocity constraint
        "2d_cam_3d_imu": measurement_function_2d,   # Still 2D camera
        "3d_full": measurement_function_3d
    }[mode]
```

#### 4. **Complete Q_rate Mapping**

```python
# In build_Q_rate():
def build_Q_rate(config: FilterCoreConfig, n: int) -> jnp.ndarray:
    """Build Q_rate for any state dimension."""

    if n == 5:  # Vision-only
        return jnp.diag(jnp.array([...]))

    elif n == 8:  # 2D full or IMU-only
        return jnp.diag(jnp.array([...]))

    elif n == 10:  # 2D camera + 3D IMU
        return jnp.diag(jnp.array([...]))

    elif n == 16:  # Full 3D
        return jnp.diag(jnp.array([...]))

    else:
        raise ValueError(f"Unsupported state dimension: {n}")
```

### Long-Term Architectural Improvements

#### 1. **Abstract Base Class for Dynamics**

```python
# src/trodestrack/models/dynamics_base.py
from abc import ABC, abstractmethod

class DynamicsModel(ABC):
    """Abstract base class for dynamics models."""

    @abstractmethod
    def integrate(self, state: jnp.ndarray, imu: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Integrate dynamics one timestep."""
        pass

    @abstractmethod
    def jacobian(self, state: jnp.ndarray, imu: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Compute dynamics Jacobian F."""
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

class Dynamics2D(DynamicsModel):
    """2D dynamics with heading."""
    state_dim = 8

    def integrate(self, state, imu, dt):
        return dynamics_function_2d(state, imu, dt, self.damping)

    def jacobian(self, state, imu, dt):
        # ... compute F
        pass

class Dynamics3DQuaternion(DynamicsModel):
    """3D dynamics with quaternion orientation."""
    state_dim = 16

    def integrate(self, state, imu, dt):
        return dynamics_function_3d_quaternion(state, imu, dt, self.damping)

    def jacobian(self, state, imu, dt):
        # ... compute F (more complex)
        pass
```

#### 2. **Plugin System for Custom Dynamics**

```python
# Allow users to define custom dynamics
class CustomRatDynamics(DynamicsModel):
    """User-defined dynamics for specific experiment."""
    state_dim = 12  # Custom state

    def integrate(self, state, imu, dt):
        # User's custom integration
        pass

# Use in filter
config = EKFConfig(dynamics_model=CustomRatDynamics())
```

---

## Summary Table

| Scenario | Difficulty | Effort | State Dim | Key Changes | Pros | Cons |
|----------|-----------|--------|-----------|-------------|------|------|
| **Vision-Only** | ⭐⭐⭐⭐⭐ | 2-4 hours | 5D | Skip IMU integration, new dynamics | Simple, no IMU drift | No dead reckoning |
| **IMU-Only** | ⭐⭐⭐⭐☆ | 4-8 hours | 8D | ZUPT, pseudo-measurements | High-rate, works in dark | Large drift, needs ZUPT |
| **2D Cam + 3D IMU** | ⭐⭐⭐☆☆ | 1-2 weeks | 10D | 3D accel, vz state | Detect vertical motion | vz poorly observable |
| **Full 3D** | ⭐⭐☆☆☆ | 3-4 weeks | 16D | Quaternion, 3D camera, gravity handling | Full 6-DOF pose | Complex Jacobians, quaternion normalization |

---

## Conclusion

**The trodestrack codebase is well-positioned for extensibility:**

✅ **Easy Extensions (1-2 weeks):**

- Vision-only (drop IMU)
- IMU-only (add ZUPT, pseudo-measurements)

✅ **Moderate Extensions (2-4 weeks):**

- 2D camera + 3D accelerometer (detect vertical motion)

⚠️ **Complex Extensions (1-2 months):**

- Full 3D tracking (quaternion, 3D camera, gravity handling, magnetometer)

**Key Enablers:**

1. Dimension-agnostic smoothers (P0.4 completed)
2. Shared filter core (clean abstractions)
3. Modular dynamics/measurement functions
4. Extensible Q_rate with TODO for 3D

**Recommended Next Steps:**

1. Fix P0 critical issues (CR-1, CR-2)
2. Add `state_mode` parameter to config
3. Implement vision-only mode (easiest proof-of-concept)
4. Test on synthetic data before real experiments

**PRD §15 Roadmap is Realistic:**
The architecture supports the 3D roadmap, but expect 3-6 months of focused effort for production-quality 3D tracking.
