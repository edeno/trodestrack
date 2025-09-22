"""IMU processing module for trodestrack.

This module provides tools for IMU data preprocessing and pre-integration
for sensor fusion with video tracking data.

Main components:
- preintegration: JAX-based IMU pre-integration between camera frames
- utils: Utility functions for bias estimation, alignment, and quality checks
"""

from .preintegration import (
    IMUPreintegrationResult,
    PreintegrationState,
    preintegrate_imu_scan,
    preintegrate_between_frames,
    convert_spikegadgets_to_preintegration_units,
    rotation_matrix_2d,
)

from .utils import (
    remove_gravity_estimate,
    detect_imu_misalignment,
    estimate_gyroscope_bias,
    compute_imu_alignment_matrix,
    validate_imu_data_quality,
)

__all__ = [
    # Pre-integration
    "IMUPreintegrationResult",
    "PreintegrationState",
    "preintegrate_imu_scan",
    "preintegrate_between_frames",
    "convert_spikegadgets_to_preintegration_units",
    "rotation_matrix_2d",
    # Utilities
    "remove_gravity_estimate",
    "detect_imu_misalignment",
    "estimate_gyroscope_bias",
    "compute_imu_alignment_matrix",
    "validate_imu_data_quality",
]