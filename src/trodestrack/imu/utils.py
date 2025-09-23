"""IMU utility functions for preprocessing and alignment."""

import logging
from typing import Any, Tuple

import numpy as np

from ..constants import STANDARD_GRAVITY_MS2

logger = logging.getLogger(__name__)


def remove_gravity_estimate(
    accel_data: np.ndarray, axis: int = 2, window_size: int = 1000
) -> np.ndarray:
    """Remove gravity component from accelerometer data using rolling mean.

    Parameters
    ----------
    accel_data : np.ndarray, shape (n_samples, 3)
        Raw accelerometer data in m/s²
    axis : int, optional
        Axis to use for gravity estimate (typically z-axis = 2). Default is 2.
    window_size : int, optional
        Window size for rolling mean gravity estimate. Default is 1000.

    Returns
    -------
    np.ndarray, shape (n_samples, 3)
        Accelerometer data with gravity component removed

    Notes
    -----
    This is a simple gravity removal that assumes the IMU is relatively
    stationary on average. For better gravity removal, use the full
    orientation estimate from the filter.
    """
    if accel_data.shape[1] != 3:
        raise ValueError("Accelerometer data must have 3 columns")

    accel_degrav = accel_data.copy()

    # Estimate gravity from rolling mean of specified axis
    if len(accel_data) >= window_size:
        # Use pandas-style rolling mean for gravity estimate
        try:
            import pandas as pd

            gravity_series = pd.Series(accel_data[:, axis])
            gravity_estimate = (
                gravity_series.rolling(window=window_size, center=True, min_periods=1).mean().values
            )
        except ImportError:
            # Fallback to simple uniform filter
            from scipy.ndimage import uniform_filter1d

            gravity_estimate = uniform_filter1d(
                accel_data[:, axis], size=window_size, mode="nearest"
            )
    else:
        # Use global mean for short sequences
        gravity_estimate = np.full(len(accel_data), np.mean(accel_data[:, axis]))

    # Remove gravity estimate from the specified axis
    accel_degrav[:, axis] -= gravity_estimate

    return accel_degrav


def detect_imu_misalignment(
    accel_data: np.ndarray, expected_gravity: float = STANDARD_GRAVITY_MS2, tolerance: float = 0.2
) -> Tuple[bool, float]:
    """Detect potential IMU misalignment by checking gravity magnitude.

    Parameters
    ----------
    accel_data : np.ndarray, shape (n_samples, 3)
        Accelerometer data in m/s²
    expected_gravity : float, optional
        Expected gravity magnitude in m/s². Default is 9.80665.
    tolerance : float, optional
        Tolerance for gravity magnitude check. Default is 0.2 (20%).

    Returns
    -------
    is_misaligned : bool
        True if IMU appears significantly misaligned
    gravity_error : float
        Relative error in gravity magnitude

    Notes
    -----
    This function checks if the mean magnitude of acceleration is close
    to expected gravity when the animal is relatively stationary.
    """
    if accel_data.shape[1] != 3:
        raise ValueError("Accelerometer data must have 3 columns")

    if len(accel_data) == 0:
        return False, 0.0

    # Calculate magnitude of acceleration vectors
    accel_magnitude = np.linalg.norm(accel_data, axis=1)

    # Use median to be robust to motion artifacts
    median_magnitude = np.median(accel_magnitude)

    # Calculate relative error
    gravity_error = abs(median_magnitude - expected_gravity) / expected_gravity

    is_misaligned = gravity_error > tolerance

    if is_misaligned:
        logger.warning(
            "IMU may be misaligned: gravity magnitude %.3f m/s² "
            "(expected %.3f m/s², error %.1f%%)",
            median_magnitude, expected_gravity, gravity_error * 100
        )

    return is_misaligned, gravity_error


def estimate_gyroscope_bias(
    gyro_data: np.ndarray, stationary_threshold: float = 0.1, min_samples: int = 100
) -> Tuple[np.ndarray, bool]:
    """Estimate gyroscope bias from stationary periods.

    Parameters
    ----------
    gyro_data : np.ndarray, shape (n_samples, 3)
        Gyroscope data in rad/s
    stationary_threshold : float, optional
        Threshold for detecting stationary periods (rad/s). Default is 0.1.
    min_samples : int, optional
        Minimum samples needed for bias estimate. Default is 100.

    Returns
    -------
    bias_estimate : np.ndarray, shape (3,)
        Estimated gyroscope bias [bx, by, bz] in rad/s
    is_reliable : bool
        True if bias estimate is considered reliable

    Notes
    -----
    This function identifies periods where all gyroscope axes are below
    the threshold and estimates bias from the mean during these periods.
    """
    if gyro_data.shape[1] != 3:
        raise ValueError("Gyroscope data must have 3 columns")

    if len(gyro_data) == 0:
        return np.zeros(3), False

    # Find stationary periods (all axes below threshold)
    gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
    stationary_mask = gyro_magnitude < stationary_threshold

    stationary_samples = np.sum(stationary_mask)

    if stationary_samples < min_samples:
        logger.warning(
            "Insufficient stationary samples for bias estimation: %d < %d",
            stationary_samples, min_samples
        )
        # Return mean of all data as fallback
        return np.mean(gyro_data, axis=0), False

    # Estimate bias from stationary periods
    bias_estimate = np.mean(gyro_data[stationary_mask], axis=0)
    is_reliable = True

    return bias_estimate, is_reliable


def compute_imu_alignment_matrix(gravity_vector: np.ndarray, target_axis: int = 2) -> np.ndarray:
    """Compute rotation matrix to align IMU gravity vector with target axis.

    Parameters
    ----------
    gravity_vector : np.ndarray, shape (3,)
        Observed gravity vector in IMU frame
    target_axis : int, optional
        Target axis for gravity alignment (0=x, 1=y, 2=z). Default is 2 (z).

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix to align gravity with target axis

    Notes
    -----
    This computes the rotation matrix that aligns the observed gravity
    vector with the specified target axis. This can be used to correct
    for IMU mounting orientation.
    """
    if len(gravity_vector) != 3:
        raise ValueError("Gravity vector must have 3 components")

    # Normalize gravity vector
    gravity_unit = gravity_vector / np.linalg.norm(gravity_vector)

    # Target unit vector
    target_unit = np.zeros(3)
    target_unit[target_axis] = 1.0

    # Compute rotation axis (cross product)
    rotation_axis = np.cross(gravity_unit, target_unit)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    # Handle parallel vectors (no rotation needed)
    if rotation_axis_norm < 1e-6:
        # Check if vectors are aligned or anti-aligned
        dot_product = np.dot(gravity_unit, target_unit)
        if dot_product > 0:
            # Already aligned
            return np.eye(3)
        else:
            # Anti-aligned, need 180-degree rotation
            # Find a perpendicular axis
            if abs(gravity_unit[0]) < 0.9:
                perp_axis = np.array([1, 0, 0])
            else:
                perp_axis = np.array([0, 1, 0])
            rotation_axis = np.cross(gravity_unit, perp_axis)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

    # Normalize rotation axis
    rotation_axis = rotation_axis / rotation_axis_norm

    # Compute rotation angle
    cos_angle = np.dot(gravity_unit, target_unit)
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    # Rodrigues' rotation formula
    K = np.array(
        [
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ]
    )

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


def validate_imu_data_quality(
    accel_data: np.ndarray, gyro_data: np.ndarray, timestamps: np.ndarray, sampling_rate: float
) -> dict[str, Any]:
    """Validate IMU data quality and return diagnostic information.

    Parameters
    ----------
    accel_data : np.ndarray, shape (n_samples, 3)
        Accelerometer data in m/s²
    gyro_data : np.ndarray, shape (n_samples, 3)
        Gyroscope data in rad/s
    timestamps : np.ndarray, shape (n_samples,)
        Sample timestamps in seconds
    sampling_rate : float
        Expected sampling rate in Hz

    Returns
    -------
    dict
        Diagnostic information about data quality

    Notes
    -----
    This function performs various quality checks on IMU data including:
    - Timing consistency
    - Data range validation
    - Gravity alignment check
    - Bias estimation
    """
    diagnostics: dict[str, Any] = {}

    # Check data shapes
    n_samples = len(timestamps)
    diagnostics["n_samples"] = n_samples
    diagnostics["duration_s"] = timestamps[-1] - timestamps[0] if n_samples > 1 else 0.0

    # Check timing consistency
    if n_samples > 1:
        dt_expected = 1.0 / sampling_rate
        dt_actual = np.median(np.diff(timestamps))
        dt_error = abs(dt_actual - dt_expected) / dt_expected
        diagnostics["timing_error_percent"] = dt_error * 100

        # Check for timing gaps
        dt_gaps = np.diff(timestamps)
        large_gaps = dt_gaps > 2 * dt_expected
        diagnostics["timing_gaps_count"] = np.sum(large_gaps)
        diagnostics["max_gap_s"] = np.max(dt_gaps)

    # Check data ranges
    diagnostics["accel_range_ms2"] = [np.min(accel_data), np.max(accel_data)]
    diagnostics["gyro_range_rad_s"] = [np.min(gyro_data), np.max(gyro_data)]

    # Check for saturated samples
    accel_saturated = np.any(np.abs(accel_data) > 150)  # Reasonable accel limit
    gyro_saturated = np.any(np.abs(gyro_data) > 35)  # Reasonable gyro limit (2000 deg/s)
    diagnostics["accel_saturated"] = accel_saturated
    diagnostics["gyro_saturated"] = gyro_saturated

    # Gravity alignment check
    is_misaligned, gravity_error = detect_imu_misalignment(accel_data)
    diagnostics["gravity_misaligned"] = is_misaligned
    diagnostics["gravity_error_percent"] = gravity_error * 100

    # Bias estimation
    gyro_bias, bias_reliable = estimate_gyroscope_bias(gyro_data)
    diagnostics["gyro_bias_rad_s"] = gyro_bias
    diagnostics["gyro_bias_reliable"] = bias_reliable

    # Overall quality assessment
    issues = []
    if diagnostics.get("timing_error_percent", 0) > 1:
        issues.append("timing_inconsistent")
    if diagnostics.get("timing_gaps_count", 0) > 0:
        issues.append("timing_gaps")
    if accel_saturated:
        issues.append("accel_saturated")
    if gyro_saturated:
        issues.append("gyro_saturated")
    if is_misaligned:
        issues.append("gravity_misaligned")
    if not bias_reliable:
        issues.append("unreliable_bias")

    diagnostics["quality_issues"] = issues
    diagnostics["quality_good"] = len(issues) == 0

    return diagnostics
