"""Synthetic IMU data generation."""

import numpy as np
from typing import Dict
from ..io.spikegadgets import SpikeGadgetsIMUData
from .config import SimConfig
from ..constants import (
    STANDARD_GRAVITY_MS2,
    SPIKEGADGETS_ACCEL_SCALE_FACTOR_G_PER_LSB,
    SPIKEGADGETS_GYRO_SCALE_FACTOR_DEGPS_PER_LSB,
    CM_TO_M,
    IMU_AXES,
)


def generate_synthetic_imu(
    ground_truth: Dict[str, np.ndarray], config: SimConfig
) -> SpikeGadgetsIMUData:
    """Generate synthetic IMU data from ground truth trajectory.

    Parameters
    ----------
    ground_truth : dict of str to np.ndarray
        Dictionary containing ground truth trajectory data with keys:
        - "timestamps" : (n_samples,) array of time points in seconds
        - "positions" : (n_samples, 2) array of x,y positions in cm
        - "velocities" : (n_samples, 2) array of vx,vy velocities in cm/s
        - "headings" : (n_samples,) array of heading angles in radians
    config : SimConfig
        Simulation configuration parameters

    Returns
    -------
    SpikeGadgetsIMUData
        Synthetic IMU measurements with realistic noise, bias drift, and misalignment

    Notes
    -----
    Simulates accelerometer and gyroscope data with:
    - Time-varying bias following random walk
    - White noise on measurements
    - Optional IMU-to-body misalignment
    - Realistic sampling at config.imu_rate Hz
    """
    # Set random seed for reproducibility
    np.random.seed(config.seed)

    # Extract ground truth data
    gt_timestamps = ground_truth["timestamps"]
    gt_velocities = ground_truth["velocities"]  # [N, 2] in cm/s
    gt_headings = ground_truth["headings"]  # [N] in radians

    # Generate IMU timestamps at high rate
    n_samples = int(config.duration * config.imu_rate)
    imu_timestamps = np.linspace(0, config.duration, n_samples)

    # Interpolate ground truth to IMU timestamps
    imu_velocities = np.column_stack(
        [
            np.interp(imu_timestamps, gt_timestamps, gt_velocities[:, 0]),
            np.interp(imu_timestamps, gt_timestamps, gt_velocities[:, 1]),
        ]
    )

    imu_headings = np.interp(imu_timestamps, gt_timestamps, gt_headings)

    # Compute true accelerations and angular velocities
    dt = 1.0 / config.imu_rate

    # Linear accelerations (cm/s² -> m/s²)
    accel_x = np.gradient(imu_velocities[:, 0]) / (dt * CM_TO_M)  # Convert cm/s² to m/s²
    accel_y = np.gradient(imu_velocities[:, 1]) / (dt * CM_TO_M)

    # Angular velocity (rad/s)
    omega_z = np.gradient(imu_headings) / dt

    # Generate biases with drift
    accel_bias_x, accel_bias_y, accel_bias_z = _generate_bias_time_series(
        n_samples, dt, config.imu.accel_bias_std, config.imu.bias_drift_std
    )
    gyro_bias_x, gyro_bias_y, gyro_bias_z = _generate_bias_time_series(
        n_samples, dt, config.imu.gyro_bias_std, config.imu.bias_drift_std
    )

    # Apply IMU misalignment (small rotation around z-axis)
    misalign_rad = np.radians(config.imu.misalignment_deg)
    cos_theta, sin_theta = np.cos(misalign_rad), np.sin(misalign_rad)

    # True accelerations in body frame (including gravity)
    # Rotate horizontal accelerations by heading angle
    cos_heading, sin_heading = np.cos(imu_headings), np.sin(imu_headings)

    # Transform to body frame (x forward, y right, z down)
    accel_body_x = accel_x * cos_heading + accel_y * sin_heading
    accel_body_y = -accel_x * sin_heading + accel_y * cos_heading
    accel_body_z = np.full_like(accel_body_x, STANDARD_GRAVITY_MS2)  # Gravity in z (down)

    # Apply misalignment
    accel_body_x_mis = accel_body_x * cos_theta - accel_body_y * sin_theta
    accel_body_y_mis = accel_body_x * sin_theta + accel_body_y * cos_theta
    accel_body_z_mis = accel_body_z  # Z-axis unchanged for small misalignment

    # Convert to g units and add biases and noise
    accel_g_x = (
        (accel_body_x_mis / STANDARD_GRAVITY_MS2)
        + accel_bias_x
        + np.random.normal(0, config.imu.accel_noise_std, n_samples)
    )
    accel_g_y = (
        (accel_body_y_mis / STANDARD_GRAVITY_MS2)
        + accel_bias_y
        + np.random.normal(0, config.imu.accel_noise_std, n_samples)
    )
    accel_g_z = (
        (accel_body_z_mis / STANDARD_GRAVITY_MS2)
        + accel_bias_z
        + np.random.normal(0, config.imu.accel_noise_std, n_samples)
    )

    # Gyroscope data (deg/s) with biases and noise
    gyro_deg_x = gyro_bias_x + np.random.normal(0, config.imu.gyro_noise_std, n_samples)
    gyro_deg_y = gyro_bias_y + np.random.normal(0, config.imu.gyro_noise_std, n_samples)
    gyro_deg_z = (
        np.degrees(omega_z)
        + gyro_bias_z
        + np.random.normal(0, config.imu.gyro_noise_std, n_samples)
    )

    # Convert to raw units (using SpikeGadgets conversion factors)
    accel_raw = np.column_stack(
        [accel_g_x / SPIKEGADGETS_ACCEL_SCALE_FACTOR_G_PER_LSB, accel_g_y / SPIKEGADGETS_ACCEL_SCALE_FACTOR_G_PER_LSB, accel_g_z / SPIKEGADGETS_ACCEL_SCALE_FACTOR_G_PER_LSB]  # g to raw
    ).astype(np.int32)

    gyro_raw = np.column_stack(
        [gyro_deg_x / SPIKEGADGETS_GYRO_SCALE_FACTOR_DEGPS_PER_LSB, gyro_deg_y / SPIKEGADGETS_GYRO_SCALE_FACTOR_DEGPS_PER_LSB, gyro_deg_z / SPIKEGADGETS_GYRO_SCALE_FACTOR_DEGPS_PER_LSB]  # deg/s to raw
    ).astype(np.int32)

    # Store true biases in metadata for testing
    metadata = {
        "true_accel_bias": np.column_stack([accel_bias_x, accel_bias_y, accel_bias_z]),
        "true_gyro_bias": np.column_stack([gyro_bias_x, gyro_bias_y, gyro_bias_z]),
        "true_omega_z": omega_z,
        "true_accel_body": np.column_stack(
            [accel_body_x_mis, accel_body_y_mis, accel_body_z_mis]
        ),
        "misalignment_deg": config.imu.misalignment_deg,
        "noise_std_accel": config.imu.accel_noise_std,
        "noise_std_gyro": config.imu.gyro_noise_std,
    }

    return SpikeGadgetsIMUData(
        timestamps=imu_timestamps,
        accel_raw=accel_raw,
        gyro_raw=gyro_raw,
        sampling_rate=config.imu_rate,
        metadata=metadata,
    )


def _generate_bias_time_series(
    n_samples: int, dt: float, initial_std: float, drift_std: float
) -> tuple:
    """Generate bias time series for 3-axis sensor with random walk drift.

    Parameters
    ----------
    n_samples : int
        Number of time samples
    dt : float
        Time step between samples in seconds
    initial_std : float
        Standard deviation of initial bias values
    drift_std : float
        Standard deviation of drift per second (sqrt of PSD)

    Returns
    -------
    tuple of np.ndarray
        Three arrays (bias_x, bias_y, bias_z) each of shape (n_samples,)
        representing time-varying bias for each axis

    Notes
    -----
    Implements random walk model: bias[k+1] = bias[k] + noise * sqrt(dt)
    where noise ~ N(0, drift_std).
    """
    # Initial biases
    bias_x = np.random.normal(0, initial_std)
    bias_y = np.random.normal(0, initial_std)
    bias_z = np.random.normal(0, initial_std)

    # Generate random walk drift vectorized
    drift_noise_std = drift_std * np.sqrt(dt)

    # Generate all random walk increments at once
    increments = np.random.normal(0, drift_noise_std, (n_samples - 1, IMU_AXES))

    # Initialize bias series with initial values
    bias_series = np.zeros((n_samples, IMU_AXES))
    bias_series[0] = [bias_x, bias_y, bias_z]

    # Vectorized cumulative sum for random walk
    bias_series[1:] = np.cumsum(increments, axis=0)
    bias_series = bias_series + bias_series[0]  # Add initial bias

    return bias_series[:, 0], bias_series[:, 1], bias_series[:, 2]
