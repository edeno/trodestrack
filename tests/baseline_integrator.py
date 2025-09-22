"""Deterministic baseline integrator for IMU pre-integration validation.

This module provides a simple, deterministic trapezoidal integration method
for validating the JAX pre-integration implementation. The baseline uses
pure NumPy operations for transparency and debugging.
"""

from typing import NamedTuple, Optional

import numpy as np


class BaselineIntegrationResult(NamedTuple):
    """Result of baseline IMU integration.

    Attributes
    ----------
    delta_position : np.ndarray, shape (2,)
        Change in position [dx, dy] in cm
    delta_velocity : np.ndarray, shape (2,)
        Change in velocity [dvx, dvy] in cm/s
    delta_heading : float
        Change in heading (delta_theta) in radians, wrapped to [-π, π]
    dt : float
        Total integration time in seconds
    n_samples : int
        Number of IMU samples integrated
    trajectory : dict, optional
        Full trajectory arrays for debugging
    """

    delta_position: np.ndarray
    delta_velocity: np.ndarray
    delta_heading: float
    dt: float
    n_samples: int
    trajectory: Optional[dict] = None


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π] range.

    Parameters
    ----------
    angle : float
        Angle in radians

    Returns
    -------
    float
        Wrapped angle in [-π, π]
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def baseline_trapezoidal_integration(
    imu_data: np.ndarray,
    timestamps: np.ndarray,
    initial_heading: float = 0.0,
    initial_velocity: Optional[np.ndarray] = None,
    gyro_bias: float = 0.0,
    accel_bias: Optional[np.ndarray] = None,
    damping_lambda: float = 0.0,
    return_trajectory: bool = False,
) -> BaselineIntegrationResult:
    """Deterministic trapezoidal integration of IMU data.

    This provides a simple, transparent baseline for validating the JAX
    implementation. Uses trapezoidal rule for all integrations.

    Parameters
    ----------
    imu_data : np.ndarray, shape (n_samples, 6)
        IMU measurements: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        Units: accel in m/s², gyro in rad/s
    timestamps : np.ndarray, shape (n_samples,)
        Sample timestamps in seconds
    initial_heading : float, optional
        Initial heading in radians. Default is 0.0.
    initial_velocity : np.ndarray, shape (2,), optional
        Initial velocity [vx, vy] in cm/s. Default is [0, 0].
    gyro_bias : float, optional
        Gyroscope z-axis bias in rad/s. Default is 0.0.
    accel_bias : np.ndarray, shape (2,), optional
        Accelerometer bias [bx, by] in m/s². Default is [0, 0].
    damping_lambda : float, optional
        Velocity damping factor. Default is 0.0 (no damping).
    return_trajectory : bool, optional
        Whether to return full trajectory for debugging. Default is False.

    Returns
    -------
    BaselineIntegrationResult
        Integration result with deltas and metadata
    """
    # Validate inputs
    if imu_data.shape[1] != 6:
        raise ValueError(f"IMU data must have 6 columns, got {imu_data.shape[1]}")
    if len(timestamps) != len(imu_data):
        raise ValueError(
            f"Timestamp length {len(timestamps)} doesn't match IMU data length {len(imu_data)}"
        )

    n_samples = len(timestamps)
    if n_samples < 2:
        # Return zero deltas for insufficient data
        return BaselineIntegrationResult(
            delta_position=np.zeros(2),
            delta_velocity=np.zeros(2),
            delta_heading=0.0,
            dt=0.0 if n_samples == 0 else timestamps[-1] - timestamps[0],
            n_samples=n_samples,
        )

    # Set defaults
    if initial_velocity is None:
        initial_velocity = np.zeros(2)
    if accel_bias is None:
        accel_bias = np.zeros(2)

    # Convert initial velocity from cm/s to m/s for consistency
    initial_velocity_ms = np.array(initial_velocity) / 100.0

    # Extract IMU components
    accel_x = imu_data[:, 0]  # m/s²
    accel_y = imu_data[:, 1]  # m/s²
    gyro_z = imu_data[:, 5]  # rad/s

    # Apply bias correction
    accel_x_corrected = accel_x - accel_bias[0]
    accel_y_corrected = accel_y - accel_bias[1]
    gyro_z_corrected = gyro_z - gyro_bias

    # Initialize trajectory arrays
    if return_trajectory:
        trajectory = {
            "timestamps": timestamps.copy(),
            "position": np.zeros((n_samples, 2)),
            "velocity": np.zeros((n_samples, 2)),
            "heading": np.zeros(n_samples),
            "accel_world": np.zeros((n_samples, 2)),
        }
    else:
        trajectory = None

    # Initialize state
    position = np.zeros(2)  # Start at origin (m)
    velocity = initial_velocity_ms.copy()  # m/s
    heading = initial_heading  # rad

    if return_trajectory:
        trajectory["position"][0] = position * 100.0  # Convert to cm for output
        trajectory["velocity"][0] = velocity * 100.0  # Convert to cm/s for output
        trajectory["heading"][0] = heading

    # Trapezoidal integration step by step
    for i in range(1, n_samples):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            continue  # Skip non-positive time steps

        # Previous state
        prev_heading = heading
        prev_velocity = velocity.copy()

        # Integrate heading using trapezoidal rule
        avg_gyro = 0.5 * (gyro_z_corrected[i - 1] + gyro_z_corrected[i])
        heading += avg_gyro * dt

        # Wrap heading to [-π, π]
        heading = wrap_angle(heading)

        # Rotate accelerations to world frame (use average heading)
        avg_heading = wrap_angle(0.5 * (prev_heading + heading))
        cos_h = np.cos(avg_heading)
        sin_h = np.sin(avg_heading)

        # Transform accelerations at both time points
        accel_world_prev = np.array(
            [
                cos_h * accel_x_corrected[i - 1] - sin_h * accel_y_corrected[i - 1],
                sin_h * accel_x_corrected[i - 1] + cos_h * accel_y_corrected[i - 1],
            ]
        )

        accel_world_curr = np.array(
            [
                cos_h * accel_x_corrected[i] - sin_h * accel_y_corrected[i],
                sin_h * accel_x_corrected[i] + cos_h * accel_y_corrected[i],
            ]
        )

        # Apply velocity damping to previous velocity
        damping_prev = -damping_lambda * prev_velocity
        damping_curr = -damping_lambda * velocity  # Will be updated

        # Integrate velocity using trapezoidal rule with damping
        avg_accel = 0.5 * (accel_world_prev + accel_world_curr)
        avg_damping = 0.5 * (damping_prev + damping_curr)

        # Update velocity
        velocity += (avg_accel + avg_damping) * dt

        # Integrate position using trapezoidal rule
        avg_velocity = 0.5 * (prev_velocity + velocity)
        position += avg_velocity * dt

        # Store trajectory if requested
        if return_trajectory:
            trajectory["position"][i] = position * 100.0  # Convert to cm
            trajectory["velocity"][i] = velocity * 100.0  # Convert to cm/s
            trajectory["heading"][i] = heading
            trajectory["accel_world"][i] = avg_accel

    # Compute deltas
    delta_position_cm = position * 100.0  # Convert m to cm
    delta_velocity_cm = (velocity - initial_velocity_ms) * 100.0  # Convert m/s to cm/s
    delta_heading_wrapped = wrap_angle(heading - initial_heading)

    total_time = timestamps[-1] - timestamps[0]

    return BaselineIntegrationResult(
        delta_position=delta_position_cm,
        delta_velocity=delta_velocity_cm,
        delta_heading=delta_heading_wrapped,
        dt=total_time,
        n_samples=n_samples,
        trajectory=trajectory,
    )


def compare_integration_methods(
    imu_data: np.ndarray,
    timestamps: np.ndarray,
    initial_heading: float = 0.0,
    initial_velocity: Optional[np.ndarray] = None,
    gyro_bias: float = 0.0,
    accel_bias: Optional[np.ndarray] = None,
    damping_lambda: float = 0.0,
    atol_position: float = 0.01,  # 1 cm
    atol_velocity: float = 0.1,  # 0.1 cm/s
    atol_heading: float = 0.01,  # ~0.6 degrees
) -> dict:
    """Compare baseline trapezoidal vs JAX scan integration.

    Parameters
    ----------
    imu_data : np.ndarray
        IMU data for comparison
    timestamps : np.ndarray
        Sample timestamps
    initial_heading : float, optional
        Initial heading in radians
    initial_velocity : np.ndarray, optional
        Initial velocity in cm/s
    gyro_bias : float, optional
        Gyroscope bias in rad/s
    accel_bias : np.ndarray, optional
        Accelerometer bias in m/s²
    damping_lambda : float, optional
        Velocity damping factor
    atol_position : float, optional
        Absolute tolerance for position comparison (cm)
    atol_velocity : float, optional
        Absolute tolerance for velocity comparison (cm/s)
    atol_heading : float, optional
        Absolute tolerance for heading comparison (rad)

    Returns
    -------
    dict
        Comparison results with errors and pass/fail status
    """
    import jax.numpy as jnp

    from trodestrack.imu.preintegration import preintegrate_imu_scan

    # Baseline integration
    baseline_result = baseline_trapezoidal_integration(
        imu_data,
        timestamps,
        initial_heading=initial_heading,
        initial_velocity=initial_velocity,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        damping_lambda=damping_lambda,
        return_trajectory=False,
    )

    # JAX integration
    jax_result = preintegrate_imu_scan(
        jnp.array(imu_data),
        jnp.array(timestamps),
        initial_heading=initial_heading,
        initial_velocity=jnp.array(initial_velocity) if initial_velocity is not None else None,
        gyro_bias=gyro_bias,
        accel_bias=jnp.array(accel_bias) if accel_bias is not None else None,
        damping_lambda=damping_lambda,
    )

    # Compute errors
    pos_error = np.linalg.norm(np.array(jax_result.delta_position) - baseline_result.delta_position)
    vel_error = np.linalg.norm(np.array(jax_result.delta_velocity) - baseline_result.delta_velocity)

    # Handle angle difference properly
    heading_diff = wrap_angle(float(jax_result.delta_heading) - baseline_result.delta_heading)
    heading_error = abs(heading_diff)

    # Check tolerances
    pos_pass = pos_error <= atol_position
    vel_pass = vel_error <= atol_velocity
    heading_pass = heading_error <= atol_heading

    return {
        "baseline_result": baseline_result,
        "jax_result": jax_result,
        "errors": {"position": pos_error, "velocity": vel_error, "heading": heading_error},
        "tolerances": {
            "position": atol_position,
            "velocity": atol_velocity,
            "heading": atol_heading,
        },
        "passed": {
            "position": pos_pass,
            "velocity": vel_pass,
            "heading": heading_pass,
            "all": pos_pass and vel_pass and heading_pass,
        },
    }
