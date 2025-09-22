"""Synthetic IMU data generation."""

import numpy as np
from typing import Dict, Any
from ..io.spikegadgets import SpikeGadgetsIMUData
from .config import SimConfig


def generate_synthetic_imu(ground_truth: Dict[str, np.ndarray], config: SimConfig) -> SpikeGadgetsIMUData:
    """Generate synthetic IMU data from ground truth trajectory.

    Args:
        ground_truth: Dictionary containing ground truth trajectory data
        config: Simulation configuration

    Returns:
        SpikeGadgetsIMUData with synthetic IMU measurements
    """
    # Set random seed for reproducibility
    np.random.seed(config.seed)

    # Extract ground truth data
    gt_timestamps = ground_truth["timestamps"]
    gt_positions = ground_truth["positions"]  # [N, 2] in cm
    gt_velocities = ground_truth["velocities"]  # [N, 2] in cm/s
    gt_headings = ground_truth["headings"]  # [N] in radians

    # Generate IMU timestamps at high rate
    n_samples = int(config.duration * config.imu_rate)
    imu_timestamps = np.linspace(0, config.duration, n_samples)

    # Interpolate ground truth to IMU timestamps
    imu_positions = np.column_stack([
        np.interp(imu_timestamps, gt_timestamps, gt_positions[:, 0]),
        np.interp(imu_timestamps, gt_timestamps, gt_positions[:, 1])
    ])

    imu_velocities = np.column_stack([
        np.interp(imu_timestamps, gt_timestamps, gt_velocities[:, 0]),
        np.interp(imu_timestamps, gt_timestamps, gt_velocities[:, 1])
    ])

    imu_headings = np.interp(imu_timestamps, gt_timestamps, gt_headings)

    # Compute true accelerations and angular velocities
    dt = 1.0 / config.imu_rate

    # Linear accelerations (cm/s² -> m/s²)
    accel_x = np.gradient(imu_velocities[:, 0]) / (dt * 100)  # Convert cm/s² to m/s²
    accel_y = np.gradient(imu_velocities[:, 1]) / (dt * 100)

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
    accel_body_z = np.full_like(accel_body_x, 9.80665)  # Gravity in z (down)

    # Apply misalignment
    accel_body_x_mis = accel_body_x * cos_theta - accel_body_y * sin_theta
    accel_body_y_mis = accel_body_x * sin_theta + accel_body_y * cos_theta
    accel_body_z_mis = accel_body_z  # Z-axis unchanged for small misalignment

    # Convert to g units and add biases and noise
    accel_g_x = (accel_body_x_mis / 9.80665) + accel_bias_x + np.random.normal(0, config.imu.accel_noise_std, n_samples)
    accel_g_y = (accel_body_y_mis / 9.80665) + accel_bias_y + np.random.normal(0, config.imu.accel_noise_std, n_samples)
    accel_g_z = (accel_body_z_mis / 9.80665) + accel_bias_z + np.random.normal(0, config.imu.accel_noise_std, n_samples)

    # Gyroscope data (deg/s) with biases and noise
    gyro_deg_x = gyro_bias_x + np.random.normal(0, config.imu.gyro_noise_std, n_samples)
    gyro_deg_y = gyro_bias_y + np.random.normal(0, config.imu.gyro_noise_std, n_samples)
    gyro_deg_z = np.degrees(omega_z) + gyro_bias_z + np.random.normal(0, config.imu.gyro_noise_std, n_samples)

    # Convert to raw units (using SpikeGadgets conversion factors)
    accel_raw = np.column_stack([
        accel_g_x / 0.000061,  # g to raw
        accel_g_y / 0.000061,
        accel_g_z / 0.000061
    ]).astype(np.int32)

    gyro_raw = np.column_stack([
        gyro_deg_x / 0.061,  # deg/s to raw
        gyro_deg_y / 0.061,
        gyro_deg_z / 0.061
    ]).astype(np.int32)

    # Store true biases in metadata for testing
    metadata = {
        "true_accel_bias": np.column_stack([accel_bias_x, accel_bias_y, accel_bias_z]),
        "true_gyro_bias": np.column_stack([gyro_bias_x, gyro_bias_y, gyro_bias_z]),
        "true_omega_z": omega_z,
        "true_accel_body": np.column_stack([accel_body_x_mis, accel_body_y_mis, accel_body_z_mis]),
        "misalignment_deg": config.imu.misalignment_deg,
        "noise_std_accel": config.imu.accel_noise_std,
        "noise_std_gyro": config.imu.gyro_noise_std
    }

    return SpikeGadgetsIMUData(
        timestamps=imu_timestamps,
        accel_raw=accel_raw,
        gyro_raw=gyro_raw,
        sampling_rate=config.imu_rate,
        metadata=metadata
    )


def _generate_bias_time_series(n_samples: int, dt: float, initial_std: float, drift_std: float) -> tuple:
    """Generate bias time series for 3-axis sensor with random walk drift.

    Args:
        n_samples: Number of samples
        dt: Time step in seconds
        initial_std: Standard deviation of initial bias
        drift_std: Standard deviation of drift per second

    Returns:
        Tuple of (bias_x, bias_y, bias_z) arrays
    """
    # Initial biases
    bias_x = np.random.normal(0, initial_std)
    bias_y = np.random.normal(0, initial_std)
    bias_z = np.random.normal(0, initial_std)

    # Generate random walk drift
    drift_noise_std = drift_std * np.sqrt(dt)

    bias_x_series = np.zeros(n_samples)
    bias_y_series = np.zeros(n_samples)
    bias_z_series = np.zeros(n_samples)

    bias_x_series[0] = bias_x
    bias_y_series[0] = bias_y
    bias_z_series[0] = bias_z

    for i in range(1, n_samples):
        bias_x_series[i] = bias_x_series[i-1] + np.random.normal(0, drift_noise_std)
        bias_y_series[i] = bias_y_series[i-1] + np.random.normal(0, drift_noise_std)
        bias_z_series[i] = bias_z_series[i-1] + np.random.normal(0, drift_noise_std)

    return bias_x_series, bias_y_series, bias_z_series