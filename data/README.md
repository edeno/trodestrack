# Arthur Session Data (2022-03-14)

Real trodestrack data from rat "Arthur" recorded on March 14, 2022.

## Files

- **`arthur20220314_position_info.parquet`** - Camera tracking data (46,056 frames, 25 min)
- **`arthur20220314_imu_info.parquet`** - IMU sensor data (45.5M samples with sample-and-hold)
- **`20220314_arthur_02_r1.mp4`** - Video recording (253 MB)
- **`load_arthur_session.py`** - Data loader with unit conversion and preprocessing

## Data Format

### Timestamps
- **Format**: Unix timestamps (float64) as pandas DataFrame index
- **Example**: 1647365966.563 (seconds since epoch)
- **Usage**: Convert to relative time (subtract start time) for trodestrack EKF

### Camera Data
- **Frames**: 46,056
- **Rate**: 30.4 Hz
- **Duration**: 1515.2 seconds (25.3 minutes)
- **Columns**: `xloc`, `yloc`, `xloc2`, `yloc2`, `video_frame_ind`, `HWframeCount`, `HWTimestamp`
- **Units**: Pixels (raw 16-bit integers)
- **Conversion**: `position_m = pixel_value × 0.0022` (user-provided)
- **LED separation**: 23.1 pixels = 5.08 cm

### IMU Data
- **Total samples**: 45,480,614 (with sample-and-hold repeats)
- **Unique samples**: ~158,271 (true data points)
- **True rate**: 100 Hz (nominal 20 kHz with ~287× sample-and-hold)
- **Duration**: 1516.0 seconds (25.3 minutes)
- **Columns**: `Headstage_GyroX/Y/Z`, `Headstage_AccelX/Y/Z`
- **Units**: RAW integers (16-bit)
- **Time overlap**: 100% with camera data

## Unit Conversions (SpikeGadgets Headstage)

### Gyroscope (±2000 deg/s range)
```python
GYRO_SCALE = 0.061  # deg/s per LSB
gyro_deg_s = raw_value * 0.061
gyro_rad_s = gyro_deg_s * (np.pi / 180)
```

### Accelerometer (±2g range)
```python
ACCEL_SCALE = 0.000061  # g per LSB
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
- LED separation: 5.08 cm (consistent throughout)

### ✓ Time Synchronization
- Camera: [0.772, 1515.944] seconds (relative)
- IMU: [0.000, 1516.012] seconds (relative)
- Overlap: 1515.2 seconds (100% of camera data)

### ✓ Tracking Quality
- Both LEDs visible: 100% of frames
- No dropouts or missing data
- Consecutive video frames [8, 46063]

## Critical Limitation: 3D IMU

**⚠️ The current trodestrack EKF expects 2D accelerometer input (X, Y only), but this data contains 3D accelerometer measurements (X, Y, Z) with gravity primarily in the Z-axis (mean -8.55 m/s²).**

This means trodestrack cannot properly process this data without modification. See below for solutions.

### Accelerometer Breakdown
- **AccelX**: mean=0.78 m/s², std=1.70 m/s² (motion + gravity component)
- **AccelY**: mean=2.83 m/s², std=3.00 m/s² (motion + gravity component)
- **AccelZ**: mean=-8.55 m/s², std=1.42 m/s² (mostly gravity)
- **3D magnitude**: 9.67 m/s² ✓

The headstage is mounted approximately upright with gravity primarily in the -Z direction.

## Usage

### Quick Start

```python
from load_arthur_session import load_arthur_session

# Load and preprocess all data
data = load_arthur_session(
    position_file="arthur20220314_position_info.parquet",
    imu_file="arthur20220314_imu_info.parquet",
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

## Next Steps

Before this data can be used with trodestrack, choose one approach:

### Option 1: Quick Fix (Constant Tilt Correction)
Estimate headstage tilt from mean accelerometer values and project X/Y onto horizontal plane:
```python
# Estimate tilt from mean accel
accel_mean = [accel_x.mean(), accel_y.mean(), accel_z.mean()]
tilt_angle = np.arccos(-accel_z.mean() / 9.81)  # Angle from vertical

# Project X/Y to horizontal plane (simplified)
# This assumes constant tilt throughout session
```

### Option 2: Proper Fix (Extend EKF to 3D)
Extend trodestrack EKF to handle 3D IMU:
- Add pitch/roll to state vector
- Update process model for 3D dynamics
- Update measurement model for 3D accelerometer
- Handle gravity vector properly

### Option 3: Use Only Gyro + Camera
Disable accelerometer inputs and rely on gyro + camera only:
- Heading from gyro integration + LED pair
- Position and velocity from camera only
- Less robust during camera dropouts

## Questions?

For issues or questions about this dataset, see the main trodestrack documentation or open an issue on GitHub.

---

**Last Updated**: 2025-10-11
**Dataset Version**: 1.0
**Status**: ⚠️ Requires 3D IMU support in trodestrack
