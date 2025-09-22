"""Physical and numerical constants for trodestrack package."""

import numpy as np


# Physical constants
STANDARD_GRAVITY_MS2 = 9.80665  # Standard gravitational acceleration in m/s²
PI_RADIANS = np.pi  # π radians
DEGREES_TO_RADIANS = np.pi / 180.0  # Conversion factor from degrees to radians
RADIANS_TO_DEGREES = 180.0 / np.pi  # Conversion factor from radians to degrees

# SpikeGadgets hardware constants
SPIKEGADGETS_DEFAULT_CLOCK_RATE = 30000.0  # Default SpikeGadgets clock rate in Hz
SPIKEGADGETS_ACCEL_SCALE_FACTOR = 0.000061  # Raw accelerometer to g conversion
SPIKEGADGETS_GYRO_SCALE_FACTOR = 0.061  # Raw gyroscope to deg/s conversion

# Binary format constants
SPIKEGADGETS_TIMESTAMP_BYTES = 4  # uint32 timestamp
SPIKEGADGETS_IMU_VALUE_BYTES = 2  # int16 IMU values
SPIKEGADGETS_BASIC_RECORD_SIZE = 16  # 4 + 6*2 bytes (timestamp + 6 IMU values)
SPIKEGADGETS_MAG_RECORD_SIZE = 22  # 4 + 9*2 bytes (timestamp + 9 IMU+mag values)

# Coordinate system constants
SPATIAL_DIMENSIONS = 2  # Number of spatial dimensions for 2D tracking
IMU_AXES = 3  # Number of IMU axes (x, y, z)
ARENA_DIMENSIONS = 2  # Arena has width and height

# Default thresholds and tolerances
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Default confidence threshold for detections
DEFAULT_SYNC_TOLERANCE_MS = 1.0  # Default synchronization tolerance in milliseconds
DEFAULT_SYNC_TOLERANCE_S = DEFAULT_SYNC_TOLERANCE_MS / 1000.0  # In seconds
DEFAULT_ALIGNMENT_ERROR_S = 0.01  # Default maximum alignment error in seconds

# Video processing constants
DEFAULT_VIDEO_FPS = 30.0  # Default video frame rate
DEFAULT_CM_TO_PIXELS = 2.0  # Default conversion from cm to pixels
DEFAULT_PIXEL_PER_CM = 10.0  # Default pixels per cm for ruler scale

# Trajectory simulation constants
ARENA_BOUNDARY_MARGIN_CM = 10.0  # Margin from arena edge in cm
MIN_INITIAL_SPEED_CM_S = 5.0  # Minimum initial speed in cm/s
MAX_INITIAL_SPEED_CM_S = 20.0  # Maximum initial speed in cm/s
TURN_ANGLE_STD_RAD = 0.5  # Standard deviation for random turns in radians
ACCELERATION_DIVISOR = 10.0  # Divisor for random acceleration magnitude
MIN_VELOCITY_THRESHOLD = 1e-6  # Minimum velocity to avoid division by zero

# Noise and confidence defaults
MIN_OCCLUSION_CONFIDENCE = 0.05  # Minimum confidence during occlusions
ZERO_CONFIDENCE = 0.0  # Zero confidence for dropouts
DEFAULT_CONFIDENCE = 1.0  # Default confidence when not provided
INVALID_POSITION = 0.0  # Position value for invalid/NaN data

# Unit conversion factors
CM_TO_M = 100.0  # Conversion from cm to meters
MS_TO_S = 1000.0  # Conversion from milliseconds to seconds

# Sampling and alignment constants
SAMPLE_FRAMES_FOR_SYNC = 100  # Number of frames to sample for synchronization check
MIN_FRAMES_FOR_ANALYSIS = 2  # Minimum frames needed for analysis
YAML_INDENT = 2  # YAML file indentation

# Chi-squared critical values
CHI2_P001_DF2 = 9.21  # Chi-squared critical value for p=0.01, df=2 (Mahalanobis gating)

# Mathematical constants for bias modeling
BIAS_MODELS_AXES = 3  # Number of axes for bias time series (x, y, z)

# Alignment and synchronization constants
DEFAULT_SYNC_SAMPLE_FRAMES = 100  # Number of frames to sample for synchronization check
DEFAULT_ALIGNMENT_MAX_ERROR_S = 0.01  # Default maximum alignment error in seconds (10 ms)
DEFAULT_SYNC_TOLERANCE_S = 0.001  # Default synchronization tolerance in seconds (1 ms)

# Clock correction constants
DEFAULT_DRIFT_RATE = 0.0  # Default clock drift rate (no drift)

# Time division constants
TIME_DIVISOR = 2.0  # Divisor for half-distance calculations