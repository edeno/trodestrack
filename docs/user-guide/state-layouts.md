# State Layouts

TrodesTrack uses an explicit **state layout system** to eliminate hardcoded dimension assumptions and support multiple tracking modes (5D, 8D, 10D, 14D, 15D, and 16D states).

!!! tip "Best Practice"
    Always use state layouts instead of magic indices like `[:, 0:2]`.

## Why State Layouts?

Consider this common pattern:

```python
# BAD: Hardcoded indices (fragile!)
positions = result.filtered_means[:, 0:2]
velocities = result.filtered_means[:, 2:4]
heading = result.filtered_means[:, 4]
```

This code breaks when switching state modes:

- `"vision_only"` (5D): Position at [0:2], velocity at [2:4], heading at [4]
- `"2d_full"` (8D): Same, but with 3 extra bias states
- `"3d_euler"` (15D): Position at [0:3], velocity at [3:6], completely different layout!

**State layouts solve this:**

```python
# GOOD: Dimension-agnostic (robust!)
layout = get_layout(cfg.state_mode)
positions = result.filtered_means[:, layout.pos_idx]
velocities = result.filtered_means[:, layout.vel_idx]
heading = result.filtered_means[:, layout.heading_idx]
```

## Using State Layouts

### Basic Usage

```python
from trodestrack.models.state_layout import get_layout
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

# Run filter
cfg = EKFConfig(state_mode="2d_full")
result = extended_kalman_filter(
    cfg,
    sim["t_imu"],
    sim["U_imu"],
    sim["t_cam_exp"],
    sim["Z_cam_led1"],
    sim["Z_cam_led2"],
    sim["mask_cam"],
)

# Get layout from config
layout = get_layout(cfg.state_mode)

# Extract states
positions = result.filtered_means[:, layout.pos_idx]      # (N, 2) meters
velocities = result.filtered_means[:, layout.vel_idx]     # (N, 2) m/s
headings = result.filtered_means[:, layout.heading_idx]   # (N,) radians
```

### Extracting Covariances

```python
# Full covariance matrix
P = result.filtered_covariances  # (N, 8, 8) for "2d_full"

# Position covariance (2x2 submatrix)
pos_cov = P[:, layout.pos_idx, :][:, :, layout.pos_idx]  # (N, 2, 2)

# Position uncertainty (standard deviation)
pos_std = np.sqrt(np.diagonal(pos_cov, axis1=1, axis2=2))  # (N, 2)
```

### Plotting with Uncertainty

```python
import matplotlib.pyplot as plt

# Get position and uncertainty
positions = result.filtered_means[:, layout.pos_idx]
pos_std = np.sqrt(np.diagonal(
    result.filtered_covariances[:, layout.pos_idx, :][:, :, layout.pos_idx],
    axis1=1, axis2=2
))

# Plot with confidence bands
t = sim['t_cam_exp']
plt.plot(t, positions[:, 0], label='x')
plt.fill_between(t,
                 positions[:, 0] - 2*pos_std[:, 0],
                 positions[:, 0] + 2*pos_std[:, 0],
                 alpha=0.3, label='+/- 2 sigma')
plt.xlabel('Time (s)')
plt.ylabel('X Position (m)')
plt.legend()
plt.show()
```

## Available State Modes

### `"2d_full"` (8D)

Standard sensor fusion with camera and IMU.

```
State vector: [x, y, vx, vy, theta, b_gz, b_ax, b_ay]

Indices:
- pos_idx: (0, 1)         - Position (x, y) in meters
- vel_idx: (2, 3)         - Velocity (vx, vy) in m/s
- heading_idx: 4          - Heading (theta) in radians
- bias_gyro_idx: (5,)     - Gyro-Z bias in rad/s
- bias_accel_idx: (6, 7)  - Accel-XY bias in m/s^2
```

**Use when:** You have both camera and IMU data.

### `"vision_only"` (5D)

Camera-only tracking without bias estimation.

```
State vector: [x, y, vx, vy, theta]

Indices:
- pos_idx: (0, 1)
- vel_idx: (2, 3)
- heading_idx: 4
- bias_gyro_idx: ()       # empty tuple — no biases
- bias_accel_idx: ()      # empty tuple — no biases
```

**Use when:** You only have camera data, or want faster processing.

### `"2d_cam_3d_imu"` (10D) — Default

The default `state_mode` for `EKFConfig`/`UKFConfig` (see
`FilterCoreConfig.state_mode`). 2D camera with 3D accelerometer for detecting
rearing behavior.

```
State vector: [x, y, vx, vy, vz, theta, b_gz, b_ax, b_ay, b_az]

Indices:
- pos_idx: (0, 1)
- vel_idx: (2, 3, 4)         # Note: includes vz!
- heading_idx: 5
- bias_gyro_idx: (6,)
- bias_accel_idx: (7, 8, 9)
```

**Use when:** You want to detect vertical motion (rearing, jumping).

### `"3d_euler"` (15D)

Full 3D tracking with Euler angles.

```
State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

Indices:
- pos_idx: (0, 1, 2)
- vel_idx: (3, 4, 5)
- heading_idx: (6, 7, 8)   # tuple covers (roll, pitch, yaw)
- bias_gyro_idx: (9, 10, 11)
- bias_accel_idx: (12, 13, 14)
```

**Use when:** You need full 3D pose estimation.

### `"3d_quat"` (16D)

Full 3D tracking with quaternions (avoids gimbal lock).

```
State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

Indices:
- pos_idx: (0, 1, 2)
- vel_idx: (3, 4, 5)
- heading_idx: (6, 7, 8, 9)   # 4-tuple = quaternion (qw, qx, qy, qz)
- bias_gyro_idx: (10, 11, 12)
- bias_accel_idx: (13, 14, 15)
```

**Use when:** You need 3D tracking without gimbal lock issues.

### `"3d_cam_6dof_imu"` (16D)

Experimental full 3D camera + 6-DOF IMU EKF mode. This mode uses the same
state layout as `"3d_quat"` and is exposed as a separate name so callers can
opt into the 3D camera filter path explicitly.

```
State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

Indices:
- pos_idx: (0, 1, 2)
- vel_idx: (3, 4, 5)
- heading_idx: (6, 7, 8, 9)   # 4-tuple = quaternion (qw, qx, qy, qz)
- bias_gyro_idx: (10, 11, 12)
- bias_accel_idx: (13, 14, 15)
```

**Use when:** You have 3D LED observations and want to call
`extended_kalman_filter_3d`.

The 3D EKF expects 6-channel IMU inputs `[gx, gy, gz, ax, ay, az]`, 3D LED
observations shaped `(n_time, n_leds, 3)`, and body-frame LED offsets shaped
`(n_leds, 3)`. The experimental path supports camera IEKF iterations,
Mahalanobis gating, and ZUPT. Translational accelerometer integration is
controlled by `enable_experimental_accel_translation`; leave it off for
camera/gyro-only comparisons and turn it on for accel-enabled 3D validation.
It currently has synthetic coverage only; treat it as experimental until
representative real 3D data passes dropout and bias-recovery checks.

## 3D Camera Measurement Convention

The experimental 3D camera measurement model is `Camera3DPositionModel`, used
by the `extended_kalman_filter_3d` entry point.

Inputs:

```python
Z_cam_leds.shape == (n_time, n_leds, 3)
mask_cam_leds.shape == (n_time, n_leds)
led_offsets_body.shape == (n_leds, 3)
```

- `Z_cam_leds[t, i]` is LED `i` observed in world coordinates `[x, y, z]`.
- `mask_cam_leds[t, i]` marks whole-LED visibility.
- Individual missing coordinates may also be represented with `NaN`; those
  coordinates are ignored independently.
- `led_offsets_body[i]` is the fixed LED offset in the body/headstage frame.
- Predictions use the 3D position and scalar-first body-to-world quaternion
  from the `"3d_cam_6dof_imu"` state layout.

The model keeps a fixed flattened measurement shape of `n_leds * 3`. Missing
LEDs or coordinates receive large measurement variance and zero residual, so
future JAX filter loops can keep static shapes while ignoring partial
observations.

## Layout Object Reference

The `StateLayout` object provides:

```python
layout = get_layout("2d_full")

# Dimensions (derived from index tuples)
layout.n                  # 8 (total state dimension)
len(layout.pos_idx)       # 2 (position dimension)
len(layout.vel_idx)       # 2 (velocity dimension)

# Indices for slicing (all are tuples; cast to list/jnp.array as needed)
layout.pos_idx            # (0, 1)
layout.vel_idx            # (2, 3)
layout.heading_idx        # 4 (int for 2D scalar heading)
layout.bias_gyro_idx      # (5,) — single-element tuple for 2D
layout.bias_accel_idx     # (6, 7)

# Boolean queries
layout.has_biases                   # True
layout.has_quaternion_orientation   # False (4-element heading_idx tuple = quat layout)
layout.has_heading_2d               # True (scalar int heading_idx = 2D layout)
```

## Writing Dimension-Agnostic Code

### Pattern 1: Function that Works with Any State Mode

```python
def compute_speed(result, cfg):
    """Compute speed from filter result (works with any state mode)."""
    layout = get_layout(cfg.state_mode)
    velocities = result.filtered_means[:, layout.vel_idx]
    return np.linalg.norm(velocities, axis=1)
```

### Pattern 2: Conditional Logic Based on Layout

```python
def extract_biases(result, cfg):
    """Extract biases if available."""
    layout = get_layout(cfg.state_mode)

    if not layout.has_biases:
        return None

    biases = {}
    if layout.bias_gyro_idx:  # tuple is non-empty
        biases['gyro'] = result.filtered_means[:, list(layout.bias_gyro_idx)]
    if layout.bias_accel_idx:
        biases['accel'] = result.filtered_means[:, list(layout.bias_accel_idx)]

    return biases
```

### Pattern 3: Generic Plotting Function

```python
def plot_position_with_uncertainty(result, cfg, ax=None):
    """Plot position with uncertainty bands (any state mode)."""
    import matplotlib.pyplot as plt

    layout = get_layout(cfg.state_mode)
    pos = result.filtered_means[:, layout.pos_idx]
    P = result.filtered_covariances
    pos_cov = P[:, layout.pos_idx, :][:, :, layout.pos_idx]
    pos_std = np.sqrt(np.diagonal(pos_cov, axis1=1, axis2=2))

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(pos[:, 0], pos[:, 1], 'b-', label='Position')

    # Add uncertainty ellipses at intervals
    for i in range(0, len(pos), len(pos)//10):
        ellipse = create_confidence_ellipse(pos[i], pos_cov[i])
        ax.add_patch(ellipse)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.axis('equal')

    return ax
```

## Common Mistakes

### Mistake 1: Hardcoded Indices

```python
# WRONG: Breaks when state mode changes
positions = result[:, 0:2]

# CORRECT: Use layout
layout = get_layout(cfg.state_mode)
positions = result[:, layout.pos_idx]
```

### Mistake 2: Assuming 2D

```python
# WRONG: Fails for 3D modes
vx, vy = result[:, 2], result[:, 3]

# CORRECT: Handle any dimension
layout = get_layout(cfg.state_mode)
velocities = result[:, layout.vel_idx]  # (N, 2) or (N, 3)
```

### Mistake 3: Ignoring Optional Indices

```python
# WRONG: Crashes if biases not present
gyro_bias = result[:, 5]

# CORRECT: Check first (bias_gyro_idx is an empty tuple when no biases exist)
layout = get_layout(cfg.state_mode)
if layout.bias_gyro_idx:
    gyro_bias = result[:, list(layout.bias_gyro_idx)]
else:
    gyro_bias = None
```

## See Also

- [src/trodestrack/models/state_layout.py](https://github.com/edeno/trodestrack/blob/main/src/trodestrack/models/state_layout.py) - Full implementation
- [Example 03b](../examples/index.md) - Using state layouts in practice
