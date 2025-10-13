# Arthur Session Data (2022-03-24)

Real trodestrack data from rat "Arthur" recorded on March 24, 2022.

## Files

- **`arthur20220324_position_info.parquet`** - Camera tracking data (73,450 frames, 40 min)
- **`arthur20220324_imu_info.parquet`** - IMU sensor data (72.5M samples with sample-and-hold)
- **`20220324_arthur_02_r1.mp4`** - Video recording (279 MB)
- **`load_arthur_session.py`** - Data loader with unit conversion and preprocessing
- **`visualize_session.py`** - Create video with LED tracking and 2D IMU (gyro Z, accel X/Y)
- **`visualize_all_sensors.py`** - Create video showing complete 6-axis IMU data (all gyro + accel axes)

## Data Format

### Timestamps

- **Format**: Unix timestamps (float64) as pandas DataFrame index
- **Example**: 1648163575.123 (seconds since epoch)
- **Usage**: Convert to relative time (subtract start time) for trodestrack EKF

### Camera Data

- **Frames**: 73,450
- **Rate**: 30.4 Hz
- **Duration**: 2416.5 seconds (40.3 minutes)
- **Columns**: `xloc`, `yloc`, `xloc2`, `yloc2`, `video_frame_ind`, `HWframeCount`, `HWTimestamp`
- **Units**: Pixels (raw 16-bit integers)
- **Conversion**: `position_m = pixel_value × 0.0022` (user-provided)
- **LED separation**: 21.6 pixels = 4.75 cm

### IMU Data

- **Total samples**: 72,521,877 (with sample-and-hold repeats)
- **Unique samples**: ~252,334 (true data points)
- **Hardware refresh rate**: 104 Hz (per SpikeGadgets specification)
- **Effective rate**: ~100 Hz after removing sample-and-hold duplicates
- **Nominal output rate**: ~20-30 kHz (sample-and-hold repeats from 104 Hz source)
- **Duration**: 2417.4 seconds (40.3 minutes)
- **Columns**: `Headstage_GyroX/Y/Z`, `Headstage_AccelX/Y/Z`
- **Units**: RAW integers (16-bit)
- **Time overlap**: 100% with camera data

## Unit Conversions (SpikeGadgets Headstage)

**Official Hardware Specifications** (source: [SpikeGadgets Product Manual](https://spikegadgets.com/documentation/)):

- 3-axis accelerometer: ±2g range, 16-bit signed integers
- 3-axis gyroscope: ±2000 deg/s range, 16-bit signed integers
- Sensor refresh rate: 104 Hz (when both sensors enabled)
- Internal sampling: 500 Hz per sensor (both enabled), 1 kHz (single sensor)
- Output behavior: Sample-and-hold repeats expand 104 Hz data to nominal 20-30 kHz

### Gyroscope (±2000 deg/s range)

```python
GYRO_SCALE = 0.061  # deg/s per LSB (2000/32767)
gyro_deg_s = raw_value * 0.061
gyro_rad_s = gyro_deg_s * (np.pi / 180)
```

### Accelerometer (±2g range)

```python
ACCEL_SCALE = 0.000061  # g per LSB (2/32767)
accel_g = raw_value * 0.000061
accel_m_s2 = accel_g * 9.80665
```

### Camera (user-provided)

```python
METERS_PER_PIXEL = 0.0022
position_m = pixel_value * 0.0022
```

## Data Validation

### ✓ Unit Conversions Verified

- 3D accelerometer magnitude: **9.67 m/s²** (expected ~9.81 m/s²)
- Gyro range: ±800 deg/s (within ±2000 deg/s spec)
- LED separation: 4.75 cm (consistent throughout)

### ✓ Time Synchronization

- Camera: [0.772, 2416.5] seconds (relative)
- IMU: [0.000, 2417.4] seconds (relative)
- Overlap: 2416.5 seconds (100% of camera data)

### ✓ Tracking Quality

- Both LEDs visible: 100% of frames
- No dropouts or missing data
- Consecutive video frames [8, 73457]

## Understanding the IMU Axes

### SpikeGadgets Coordinate System

The SpikeGadgets headstage uses a right-handed coordinate system as shown in the [official documentation](https://spikegadgets.com/documentation/). The physical orientation of the axes depends on how the headstage is mounted on the animal.

**Note:** The axis labels (X, Y, Z) are hardware conventions. The actual physical directions (forward/back, left/right, up/down) depend on the mounting orientation.

### How We Determined Axis Orientation

To determine which physical direction each axis corresponds to in our data, we analyzed using three methods:

**Method 1: Gravity Detection**

- Measured mean accelerometer values across the entire session
- The axis with the largest magnitude mean is aligned with gravity (vertical)
- Result: **Accel Z = -8.55 m/s²** → Z-axis points upward in body frame

**Method 2: Variance Analysis**

- Compared standard deviation of gyro axes to find most active rotations
- Gyro X: σ = 55.4 deg/s, Gyro Y: σ = 39.4 deg/s, Gyro Z: σ = 54.7 deg/s
- Result: X and Z are most active (roll and yaw), Y less active (pitch)

**Method 3: Physical Constraint**

- For an upright headstage, yaw rotation must be around the vertical axis
- Therefore: **Gyro Z measures yaw** (matches vertical axis)

Run [analyze_imu_orientation.py](analyze_imu_orientation.py) to see the complete analysis.

### Physical Orientation (Body Frame)

Based on data analysis, the SpikeGadgets headstage axes correspond to:

- **X-axis**: Left-right (mediolateral) - horizontal plane
- **Y-axis**: Forward-backward (anteroposterior) - horizontal plane
- **Z-axis**: Up-down (dorsoventral) - **vertical, aligned with gravity**

### Gyroscope: Angular Rates (Rotation)

The 3-axis gyroscope measures **angular velocity** (how fast the rat is rotating):

- **Gyro X (roll rate)**: Rotation around X-axis → tilting left/right (like an airplane rolling)
- **Gyro Y (pitch rate)**: Rotation around Y-axis → nodding up/down (like pitching forward/back)
- **Gyro Z (yaw rate)**: Rotation around Z-axis → **turning left/right (heading change)** ← **used by 2D tracking**

### Accelerometer: Linear Acceleration + Gravity

The 3-axis accelerometer measures **specific force** (linear acceleration + gravity):

- **Accel X**: Acceleration in X direction + gravity component from tilt
- **Accel Y**: Acceleration in Y direction + gravity component from tilt
- **Accel Z**: Acceleration in Z direction + **~9.81 m/s² from gravity** (when upright)

**Key Insight**: When the rat is upright and stationary, the accelerometer reads:

- X ≈ 0, Y ≈ 0, Z ≈ -9.81 m/s² (gravity pulls "down" in body frame)
- When tilted, gravity components appear in X and Y axes too

### Why 2D Tracking Uses Only a Subset

**Trodestrack's 2D tracking assumes:**

1. The rat moves primarily on a horizontal plane (maze floor)
2. The headstage stays approximately upright (small roll/pitch variations)
3. Only **yaw (heading)** matters for 2D position tracking

**Therefore, 2D tracking uses:**

- **Gyro Z** → Measures yaw rate (turning left/right on the floor)
- **Accel X, Y** → Approximate horizontal plane acceleration (after gravity removal)
- **Ignores: Gyro X, Y, Accel Z** → Roll, pitch, and vertical motion not tracked

## Critical Limitation: 3D IMU

**⚠️ The current trodestrack EKF expects 2D accelerometer input (X, Y only), but this data contains 3D accelerometer measurements (X, Y, Z) with gravity primarily in the Z-axis (mean -8.55 m/s²).**

This means trodestrack cannot properly process this data without modification. See below for solutions.

### Accelerometer Breakdown

- **AccelX**: mean=0.78 m/s², std=1.70 m/s² (motion + gravity component from tilt)
- **AccelY**: mean=2.83 m/s², std=3.00 m/s² (motion + gravity component from tilt)
- **AccelZ**: mean=-8.55 m/s², std=1.42 m/s² (mostly gravity, ~87% of 9.81 m/s²)
- **3D magnitude**: 9.67 m/s² ✓

The headstage is mounted approximately upright with gravity primarily in the -Z direction, but with some tilt causing gravity components in X and Y.

## Usage

### Quick Start

```python
from load_arthur_session import load_arthur_session

# Load and preprocess all data
data = load_arthur_session(
    position_file="arthur20220324_position_info.parquet",
    imu_file="arthur20220324_imu_info.parquet",
    meters_per_pixel=0.0022,
    verbose=True
)

# Access preprocessed data
print(f"Duration: {data.t_cam[-1]:.1f} s")
print(f"IMU rate: {data.fs_imu:.1f} Hz")
print(f"Camera rate: {data.fs_cam:.1f} Hz")
print(f"LED separation: {data.led_distance*100:.2f} cm")

# Data is ready for trodestrack EKF
# data.t_imu, data.t_cam: timestamps (s, relative)
# data.U_imu: [N×3] IMU inputs (rad/s, m/s², m/s²)
# data.Z_cam_led1, data.Z_cam_led2: [N×2] LED positions (m)
# data.mask_cam: [N] validity mask
```

### What the Loader Does

The `load_arthur_session()` function handles all preprocessing:

1. **Removes sample-and-hold**: Extracts 158,271 unique samples from 45M repeated values
2. **Converts timestamps**: Unix time → relative time (start from 0)
3. **Estimates sampling rates**: 100 Hz (IMU), 30.4 Hz (camera)
4. **Converts units to SI**:
   - Gyro: raw → deg/s → rad/s
   - Accel: raw → g → m/s²
   - Camera: pixels → meters
5. **Validates data quality**: Checks 3D accel magnitude ≈ 9.81 m/s²
6. **Computes LED separation**: Median distance between LEDs

### Run the Loader Standalone

```bash
cd data/
uv run python load_arthur_session.py
```

This runs validation checks and prints a summary of the loaded data.

### Create Visualization Videos

#### Option 1: Filter Overlay (Recommended - Shows Filter Performance)

Overlays Extended Kalman Filter estimates on the actual video with synchronized IMU data:

```bash
cd data/
uv run python visualize_filter_overlay.py
```

Creates `arthur_filter_overlay.mp4` showing:

- **Video (left)**: Original camera feed with filter overlay
  - Red circle: LED1 (back of head, measured)
  - Yellow circle: LED2 (front of head, measured)
  - Cyan circle: Filter position estimate
  - Cyan arrow: Filter heading estimate
  - Cyan trail: Recent trajectory (3 seconds)
- **IMU Data (right, top)**:
  - 3-axis gyroscope (deg/s) - angular rates
  - 3-axis accelerometer (m/s²) - linear acceleration + gravity
- **Filter States (right, bottom)**:
  - Velocity magnitude (cm/s)
  - Heading estimate (degrees)
  - Position uncertainty (cm, 1σ)

To generate multiple example clips showing different behaviors:

```bash
cd data/
uv run python generate_example_clips.py
```

This creates 5 clips in `example_clips/` directory, each demonstrating different filter scenarios (turning, fast movement, stationary, etc.).

#### Option 2: 2D IMU (Trodestrack-compatible subset)

Shows only the axes used by trodestrack's 2D EKF (gyro Z, accel X/Y):

```bash
cd data/
uv run python visualize_session.py
```

Creates `arthur_visualization.mp4` with:

- Video with LED tracking (red=rear, cyan=front)
- Gyro Z (yaw rate, used for heading)
- Accel X/Y (horizontal plane motion)

#### Option 3: Complete 6-Axis IMU

Shows ALL sensor data from the headstage:

```bash
cd data/
uv run python visualize_all_sensors.py
```

Creates `arthur_all_sensors.mp4` with:

- Video with LED tracking
- **3D Gyroscope**: X (roll rate), Y (pitch rate), Z (yaw rate)
- **3D Accelerometer**: X, Y, Z (including gravity in Z-axis)

This reveals the complete sensor picture and shows why 2D tracking only uses a subset.

#### Customization

Both scripts support customization:

```python
# 2D version (trodestrack subset)
from visualize_session import create_video_overlay
from load_arthur_session import load_arthur_session

data = load_arthur_session("position.parquet", "imu.parquet")
create_video_overlay(
    video_path="video.mp4",
    data=data,
    output_path="output.mp4",
    start_time=60.0,
    duration=10.0,
    fps=30.0,
    gyro_ylim=(-200, 200),    # deg/s
    accel_ylim=(-15, 15)      # m/s²
)

# Full 6-axis version
from visualize_all_sensors import create_comprehensive_video

create_comprehensive_video(
    video_path="video.mp4",
    position_file="position.parquet",
    imu_file="imu.parquet",
    output_path="output.mp4",
    start_time=60.0,
    duration=10.0,
    fps=30.0,
    gyro_ylim=(-200, 200),    # deg/s
    accel_ylim=(-15, 15)      # m/s² (shows gravity!)
)
```

## 3D IMU Support (✓ Implemented)

**Status**: ✅ Trodestrack now supports 3D IMU data!

The data loader and EKF have been updated to handle full 6-axis IMU (3-axis gyro + 3-axis accel):

### Using 3D IMU Mode

```python
from load_arthur_session import load_arthur_session
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter

# Load data with 3D IMU mode
data = load_arthur_session(
    position_file="arthur20220324_position_info.parquet",
    imu_file="arthur20220324_imu_info.parquet",
    imu_mode="3d",  # Use full 6-axis IMU
    meters_per_pixel=0.0022,
)

# Configure EKF for 2D camera + 3D IMU
ekf_config = EKFConfig(
    state_mode="2d_cam_3d_imu",  # 10D state
    process_noise_pos=0.02,
    process_noise_vel=2.0,
    process_noise_gyro_bias=2e-6,
    process_noise_accel_bias=2e-4,
    measurement_noise_pos=0.005**2,
    damping_coeff=0.1,
    led_distance=data.led_distance,
)

# Run filter
result = extended_kalman_filter(
    ekf_config=ekf_config,
    t_imu=data.t_imu,
    U_imu=data.U_imu,  # [N × 6] for 3D mode
    t_cam=data.t_cam,
    Z_cam_led1=data.Z_cam_led1,
    Z_cam_led2=data.Z_cam_led2,
    mask_cam=data.mask_cam,
)
```

### State Layout (2D Camera + 3D IMU)

The 10D state vector contains:

- `[0:2]` - Position: x, y (meters)
- `[2:5]` - Velocity: vx, vy, vz (m/s) - **includes vertical velocity!**
- `[5]` - Heading: θ (radians)
- `[6]` - Gyro bias: b_gz (rad/s)
- `[7:10]` - Accel bias: b_ax, b_ay, b_az (m/s²) - **includes vertical accel bias!**

### Key Features

- **Gravity compensation**: Uses all 3 accel axes for better bias estimation
- **Vertical motion detection**: vz state can detect rearing, jumping, etc.
- **Improved robustness**: Better handling of headstage tilt and orientation changes
- **Backward compatible**: Set `imu_mode="2d"` for legacy behavior

See `benchmark_3d_imu_vs_vision_only.py` for performance comparison and `visualize_filter_overlay.py` to visualize filter behavior with 3D IMU data.

## Questions?

For issues or questions about this dataset, see the main trodestrack documentation or open an issue on GitHub.

---

**Last Updated**: 2025-10-12
**Dataset Version**: 2.0
**Status**: ✅ Ready to use with trodestrack (3D IMU supported)
