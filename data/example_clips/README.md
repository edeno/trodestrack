# Example Video Clips

This directory contains example video clips from the Arthur session showing different aspects of the Extended Kalman Filter's performance.

Each clip shows:
- **Video (left)**: Original camera feed with:
  - Red circle: LED1 (back of head)
  - Yellow circle: LED2 (front of head)
  - Cyan circle: Filter position estimate
  - Cyan arrow: Filter heading estimate
  - Cyan trail: Recent trajectory (3 seconds)

- **IMU Data (right, top two panels)**:
  - Gyroscope: 3-axis angular rates (deg/s)
  - Accelerometer: 3-axis acceleration (m/s²)

- **Filter States (right, bottom three panels)**:
  - Velocity magnitude (cm/s)
  - Heading estimate (degrees)
  - Position uncertainty (cm, 1σ)

## Clips

### 01_normal_tracking.mp4

**Description**: Normal tracking with moderate movement

**Time**: 60.0s - 70.0s (1.0 - 1.2 minutes)

**Duration**: 10.0s

### 02_turning_behavior.mp4

**Description**: Active turning with heading changes

**Time**: 120.0s - 130.0s (2.0 - 2.2 minutes)

**Duration**: 10.0s

### 03_fast_movement.mp4

**Description**: Fast movement showing velocity tracking

**Time**: 300.0s - 310.0s (5.0 - 5.2 minutes)

**Duration**: 10.0s

### 04_grooming_period.mp4

**Description**: Stationary/grooming period (low velocity)

**Time**: 600.0s - 610.0s (10.0 - 10.2 minutes)

**Duration**: 10.0s

### 05_exploration.mp4

**Description**: Exploration with varied movement patterns

**Time**: 900.0s - 910.0s (15.0 - 15.2 minutes)

**Duration**: 10.0s


## Technical Details

- **State**: 10D (x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az)
- **Filter**: Extended Kalman Filter (EKF)
- **IMU Mode**: 3D (6-axis: gyro + accel)
- **Camera**: 30 Hz, 2D position from LED markers
- **Resolution**: 852×852 pixels at 30 fps
