"""IMU pre-integration using JAX for sensor fusion with camera frames.

This module implements efficient pre-integration of IMU measurements between
sparse camera frames using JAX scan operations. The implementation follows
the mathematical model defined in the PRD:

- Δθ = ∫(ω_z − b_gz) dt (heading change)
- Δv = ∫R(θ)(a − b_a) dt − λ∫v dt (velocity change with damping)
- Position prediction using integrated velocity and acceleration

Key features:
- JAX-compiled for performance
- Bias compensation for gyroscope and accelerometer
- Optional velocity damping (λ term)
- Robust integration with configurable step size
"""

from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike


class IMUPreintegrationResult(NamedTuple):
    """Result of IMU pre-integration between two time points.

    Attributes
    ----------
    delta_position : jnp.ndarray, shape (2,)
        Change in position [dx, dy] in cm
    delta_velocity : jnp.ndarray, shape (2,)
        Change in velocity [dvx, dvy] in cm/s
    delta_heading : float
        Change in heading (delta_theta) in radians
    dt : float
        Total integration time in seconds
    n_samples : int
        Number of IMU samples integrated
    """

    delta_position: Array
    delta_velocity: Array
    delta_heading: ArrayLike
    dt: ArrayLike
    n_samples: int


class PreintegrationState(NamedTuple):
    """Internal state during pre-integration scan.

    Attributes
    ----------
    position : jnp.ndarray, shape (2,)
        Current integrated position [x, y] relative to start
    velocity : jnp.ndarray, shape (2,)
        Current integrated velocity [vx, vy]
    heading : float
        Current integrated heading (theta)
    time : float
        Current time since integration start
    """

    position: Array
    velocity: Array
    heading: float
    time: float


@jax.jit
def wrap_angle_jax(angle: float) -> float:
    """Wrap angle to [-π, π] range using JAX operations.

    Parameters
    ----------
    angle : float
        Angle in radians

    Returns
    -------
    float
        Wrapped angle in [-π, π]
    """
    return ((angle + jnp.pi) % (2 * jnp.pi)) - jnp.pi


@jax.jit
def rotation_matrix_2d(theta: float) -> Array:
    """Create 2D rotation matrix from heading angle.

    Parameters
    ----------
    theta : float
        Heading angle in radians

    Returns
    -------
    jnp.ndarray, shape (2, 2)
        2D rotation matrix
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=jnp.float64)


@jax.jit
def preintegration_step(
    state: PreintegrationState,
    imu_sample: ArrayLike,
    gyro_bias: ArrayLike,
    accel_bias: ArrayLike,
    damping_lambda: float,
    dt: float,
) -> PreintegrationState:
    """Single step of IMU pre-integration.

    Parameters
    ----------
    state : PreintegrationState
        Current integration state
    imu_sample : chex.Array, shape (6,)
        IMU sample: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        Note: Only accel_x, accel_y, gyro_z are used for 2D tracking
    gyro_bias : float
        Gyroscope z-axis bias in rad/s
    accel_bias : jnp.ndarray, shape (2,)
        Accelerometer bias [bx, by] in m/s²
    damping_lambda : float
        Velocity damping factor (0 = no damping)
    dt : float
        Integration step size in seconds

    Returns
    -------
    PreintegrationState
        Updated integration state
    """
    # Extract IMU measurements (convert to proper units if needed)
    accel_xy = imu_sample[:2]  # Already in m/s² from preprocessing
    gyro_z = imu_sample[5]  # Already in rad/s from preprocessing

    # Apply bias correction
    omega_z_corrected = gyro_z - gyro_bias
    accel_corrected = accel_xy - accel_bias

    # Update heading using corrected gyroscope
    new_heading = wrap_angle_jax(state.heading + omega_z_corrected * dt)

    # Rotate accelerometer measurements to world frame
    # Use heading at middle of interval for better accuracy
    mid_heading = wrap_angle_jax(state.heading + 0.5 * omega_z_corrected * dt)
    R = rotation_matrix_2d(mid_heading)
    accel_world = R @ accel_corrected

    # Apply velocity damping: dv/dt = accel - λ*v
    velocity_damping = -damping_lambda * state.velocity
    total_acceleration = accel_world + velocity_damping

    # Update velocity with acceleration
    new_velocity = state.velocity + total_acceleration * dt

    # Update position using average velocity over interval
    avg_velocity = 0.5 * (state.velocity + new_velocity)
    new_position = state.position + avg_velocity * dt

    # Update time
    new_time = state.time + dt

    return PreintegrationState(
        position=new_position, velocity=new_velocity, heading=new_heading, time=new_time
    )


def preintegrate_imu_scan(
    imu_data: ArrayLike,
    timestamps: ArrayLike,
    initial_heading: float = 0.0,
    initial_velocity: Optional[ArrayLike] = None,
    gyro_bias: ArrayLike = 0.0,
    accel_bias: Optional[ArrayLike] = None,
    damping_lambda: ArrayLike = 0.0,
) -> IMUPreintegrationResult:
    """Pre-integrate IMU measurements using JAX scan.

    Parameters
    ----------
    imu_data : jnp.ndarray, shape (n_samples, 6)
        IMU measurements: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        Units: accel in m/s², gyro in rad/s
    timestamps : jnp.ndarray, shape (n_samples,)
        Sample timestamps in seconds
    initial_heading : float, optional
        Initial heading in radians. Default is 0.0.
    initial_velocity : jnp.ndarray, shape (2,), optional
        Initial velocity [vx, vy] in cm/s. Default is [0, 0].
    gyro_bias : float, optional
        Gyroscope z-axis bias in rad/s. Default is 0.0.
    accel_bias : jnp.ndarray, shape (2,), optional
        Accelerometer bias [bx, by] in m/s². Default is [0, 0].
    damping_lambda : float, optional
        Velocity damping factor. Default is 0.0 (no damping).

    Returns
    -------
    IMUPreintegrationResult
        Pre-integration result with deltas and metadata

    Notes
    -----
    This function uses jax.lax.scan for efficient computation. The integration
    assumes uniform time steps, calculated from the timestamp array.
    """
    # Ensure 64-bit precision for inputs
    imu_data = jnp.asarray(imu_data, dtype=jnp.float64)
    timestamps = jnp.asarray(timestamps, dtype=jnp.float64)

    # Validate inputs
    chex.assert_rank(imu_data, 2)
    chex.assert_rank(timestamps, 1)
    chex.assert_shape(imu_data, (None, 6))
    chex.assert_shape(timestamps, (imu_data.shape[0],))

    n_samples = len(timestamps)
    if n_samples < 2:
        # Return zero deltas for insufficient data
        return IMUPreintegrationResult(
            delta_position=jnp.zeros(2, dtype=jnp.float64),
            delta_velocity=jnp.zeros(2, dtype=jnp.float64),
            delta_heading=0.0,
            dt=0.0,
            n_samples=n_samples,
        )

    # Set defaults
    if initial_velocity is None:
        initial_velocity = jnp.zeros(2, dtype=jnp.float64)
    if accel_bias is None:
        accel_bias = jnp.zeros(2, dtype=jnp.float64)

    # Ensure 64-bit precision
    initial_velocity = jnp.asarray(initial_velocity, dtype=jnp.float64)
    accel_bias = jnp.asarray(accel_bias, dtype=jnp.float64)

    # Convert initial velocity from cm/s to m/s for internal consistency
    # NOTE: All internal calculations use SI units (m, m/s, m/s²)
    # External interface expects cm for position and cm/s for velocity
    VELOCITY_CONVERSION = 100.0  # cm/s to m/s
    initial_velocity_ms = initial_velocity / VELOCITY_CONVERSION

    # Calculate time steps
    dts = jnp.diff(timestamps)

    # Initialize state
    initial_state = PreintegrationState(
        position=jnp.zeros(2, dtype=jnp.float64),  # Start at origin
        velocity=initial_velocity_ms,
        heading=float(initial_heading),
        time=0.0,
    )

    # Define scan function
    def scan_fn(state, inputs):
        imu_sample, dt = inputs
        new_state = preintegration_step(
            state, imu_sample, gyro_bias, accel_bias, damping_lambda, dt
        )
        return new_state, None

    # Run scan over IMU samples (skip first sample since we use diffs)
    scan_inputs = (imu_data[1:], dts)
    final_state, _ = lax.scan(scan_fn, initial_state, scan_inputs)

    # Convert results back to cm for position/velocity (external interface units)
    POSITION_CONVERSION = 100.0  # m to cm
    delta_position_cm = final_state.position * POSITION_CONVERSION
    delta_velocity_cm = (final_state.velocity - initial_velocity_ms) * VELOCITY_CONVERSION

    total_time = timestamps[-1] - timestamps[0]

    return IMUPreintegrationResult(
        delta_position=delta_position_cm,
        delta_velocity=delta_velocity_cm,
        delta_heading=wrap_angle_jax(final_state.heading - initial_heading),
        dt=total_time,
        n_samples=n_samples,
    )


def preintegrate_between_frames(
    imu_data: ArrayLike,
    timestamps: ArrayLike,
    start_time: float,
    end_time: float,
    initial_heading: float = 0.0,
    initial_velocity: Optional[ArrayLike] = None,
    gyro_bias: float = 0.0,
    accel_bias: Optional[ArrayLike] = None,
    damping_lambda: float = 0.0,
) -> IMUPreintegrationResult:
    """Pre-integrate IMU data between two specific time points.

    This is the main interface for pre-integrating IMU measurements between
    camera frames for sensor fusion.

    Parameters
    ----------
    imu_data : jnp.ndarray, shape (n_samples, 6)
        Full IMU measurement array
    timestamps : jnp.ndarray, shape (n_samples,)
        Full timestamp array
    start_time : float
        Start time for integration in seconds
    end_time : float
        End time for integration in seconds
    initial_heading : float, optional
        Initial heading at start_time in radians
    initial_velocity : jnp.ndarray, shape (2,), optional
        Initial velocity at start_time in cm/s
    gyro_bias : float, optional
        Gyroscope z-axis bias in rad/s
    accel_bias : jnp.ndarray, shape (2,), optional
        Accelerometer bias in m/s²
    damping_lambda : float, optional
        Velocity damping factor

    Returns
    -------
    IMUPreintegrationResult
        Pre-integration result for the specified time interval

    Raises
    ------
    ValueError
        If time range is invalid or contains no IMU samples
    """
    if end_time <= start_time:
        raise ValueError(f"Invalid time range: start={start_time}, end={end_time}")

    # Find IMU samples in time range
    mask = (timestamps >= start_time) & (timestamps <= end_time)
    indices = jnp.where(mask)[0]

    if len(indices) < 2:
        # Not enough samples for integration
        dt = end_time - start_time
        return IMUPreintegrationResult(
            delta_position=jnp.zeros(2),
            delta_velocity=jnp.zeros(2),
            delta_heading=0.0,
            dt=dt,
            n_samples=len(indices),
        )

    # Extract relevant data
    subset_imu = imu_data[indices]
    subset_timestamps = timestamps[indices]

    # Perform pre-integration on subset
    return preintegrate_imu_scan(
        subset_imu,
        subset_timestamps,
        initial_heading=initial_heading,
        initial_velocity=initial_velocity,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        damping_lambda=damping_lambda,
    )


def convert_spikegadgets_to_preintegration_units(
    accel_ms2: ArrayLike, gyro_rad_s: ArrayLike
) -> Array:
    """Convert SpikeGadgets IMU data to pre-integration units.

    Parameters
    ----------
    accel_ms2 : jnp.ndarray, shape (n_samples, 3)
        Accelerometer data in m/s² (from SpikeGadgetsIMUData.get_accel_ms2())
    gyro_rad_s : jnp.ndarray, shape (n_samples, 3)
        Gyroscope data in rad/s (from SpikeGadgetsIMUData.get_gyro_rad_s())

    Returns
    -------
    jnp.ndarray, shape (n_samples, 6)
        Combined IMU array for pre-integration: [ax, ay, az, gx, gy, gz]
    """
    chex.assert_shape(accel_ms2, (None, 3))
    chex.assert_shape(gyro_rad_s, (None, 3))
    chex.assert_equal_shape([accel_ms2, gyro_rad_s])

    # Combine accelerometer and gyroscope data
    imu_combined = jnp.concatenate([accel_ms2, gyro_rad_s], axis=1)
    return imu_combined
