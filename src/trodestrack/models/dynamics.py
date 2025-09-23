"""Dynamics models for state prediction with IMU integration.

This module implements the prediction step of the EKF/UKF, including:
- State prediction with IMU pre-integration
- Jacobian computation for EKF linearization
- Process noise modeling
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from .state import State2D, state_to_array, array_to_state



@jax.jit
def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π] range using JAX operations.

    Args:
        angle: Angle in radians

    Returns:
        Wrapped angle in [-π, π]
    """
    # Use jnp.remainder for consistent wrapping
    return jnp.remainder(angle + jnp.pi, 2 * jnp.pi) - jnp.pi


@jax.jit
def rotation_matrix_2d(theta: float) -> jnp.ndarray:
    """Create 2D rotation matrix from heading angle.

    Args:
        theta: Heading angle in radians

    Returns:
        2D rotation matrix (2x2)
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=jnp.float64)


def _check_damping_stability(dt: float, velocity_damping: float) -> None:
    """Check that velocity damping is numerically stable.

    For stability, we need: 0 <= damping_factor = (1 - λ*dt) <= 1
    This requires: 0 <= λ*dt <= 1, or λ <= 1/dt

    Args:
        dt: Time step (seconds)
        velocity_damping: Damping coefficient λ (1/s)

    Raises:
        ValueError: If damping would cause instability
    """
    max_stable_damping = 1.0 / dt
    if velocity_damping > max_stable_damping:
        raise ValueError(
            f"Velocity damping λ={velocity_damping:.3f} too large for dt={dt:.4f}s. "
            f"Maximum stable damping: λ <= {max_stable_damping:.3f}"
        )
    if velocity_damping < 0:
        raise ValueError(f"Velocity damping λ={velocity_damping:.3f} must be non-negative")


def predict_state(
    state: State2D,
    dt: float,
    accel: jnp.ndarray,
    gyro: jnp.ndarray,
    velocity_damping: float = 0.0,
) -> State2D:
    """Predict state forward using IMU measurements.

    Implements the dynamics model from the PRD:
    - Position: x_{k+1} = x_k + v_k * dt + 0.5 * a_corrected * dt²
    - Velocity: v_{k+1} = v_k + a_corrected * dt - λ * v_k * dt
    - Heading: θ_{k+1} = θ_k + (ω_z - b_gz) * dt
    - Biases: modeled as random walks (unchanged in prediction)

    Args:
        state: Current state
        dt: Time step (seconds)
        accel: Accelerometer measurement [ax, ay] in m/s²
        gyro: Gyroscope measurement [gz] in rad/s
        velocity_damping: Velocity damping coefficient λ (1/s)

    Returns:
        Predicted state
    """
    # Check damping stability
    _check_damping_stability(dt, velocity_damping)

    # Bias-corrected measurements
    accel_corrected = accel - jnp.array([state.b_ax, state.b_ay])
    gyro_corrected = gyro[0] - state.b_gz

    # Rotate acceleration from IMU/body frame to world frame using current heading
    R = rotation_matrix_2d(state.theta)
    accel_world = R @ accel_corrected

    # Convert acceleration to cm/s² for consistency with position units
    accel_corrected_cm = accel_world * 100.0

    # Current velocity
    velocity = jnp.array([state.vx, state.vy])

    # Apply velocity damping: v_damped = v * (1 - λ*dt)
    damping_factor = 1.0 - velocity_damping * dt
    velocity_damped = velocity * damping_factor

    # Velocity update: v_{k+1} = v_damped + a * dt
    velocity_new = velocity_damped + accel_corrected_cm * dt

    # Position update: x_{k+1} = x_k + v_k * dt + 0.5 * a * dt²
    position = jnp.array([state.x, state.y])
    position_new = position + velocity * dt + 0.5 * accel_corrected_cm * dt**2

    # Heading update: θ_{k+1} = wrap(θ_k + ω * dt)
    heading_new = wrap_angle(state.theta + gyro_corrected * dt)

    # Biases remain unchanged (random walk model - noise added in process noise)
    return State2D(
        x=float(position_new[0]),
        y=float(position_new[1]),
        vx=float(velocity_new[0]),
        vy=float(velocity_new[1]),
        theta=float(heading_new),
        b_gz=state.b_gz,
        b_ax=state.b_ax,
        b_ay=state.b_ay,
    )


def predict_covariance(
    covariance: jnp.ndarray,
    state_array: jnp.ndarray,
    dt: float,
    accel: jnp.ndarray,
    gyro: jnp.ndarray,
    velocity_damping: float,
    process_noise: jnp.ndarray,
) -> jnp.ndarray:
    """Predict covariance matrix using linearized dynamics.

    Args:
        covariance: Current covariance matrix (8x8)
        state_array: Current state as array
        dt: Time step
        accel: Accelerometer measurement
        gyro: Gyroscope measurement
        velocity_damping: Velocity damping coefficient
        process_noise: Process noise matrix Q

    Returns:
        Predicted covariance: F * P * F^T + Q
    """
    # Compute state transition Jacobian
    F = compute_state_jacobian(state_array, dt, accel, gyro, velocity_damping)

    # Predict covariance: P_{k+1|k} = F * P_k * F^T + Q
    predicted_cov = F @ covariance @ F.T + process_noise

    return predicted_cov


@jax.jit
def compute_state_jacobian(
    state_array: jnp.ndarray,
    dt: float,
    accel: jnp.ndarray,
    gyro: jnp.ndarray,
    velocity_damping: float,
) -> jnp.ndarray:
    """Compute Jacobian of dynamics function for EKF linearization.

    Uses JAX automatic differentiation to compute ∂f/∂x where f is the
    dynamics function.

    Args:
        state_array: State vector [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
        dt: Time step
        accel: Accelerometer measurement
        gyro: Gyroscope measurement
        velocity_damping: Velocity damping coefficient

    Returns:
        8x8 Jacobian matrix
    """

    def dynamics_function(x: jnp.ndarray) -> jnp.ndarray:
        """Dynamics function for Jacobian computation."""
        # Extract state components
        pos = x[:2]
        vel = x[2:4]
        theta = x[4]
        b_gz = x[5]
        b_ax = x[6]
        b_ay = x[7]

        # Bias-corrected measurements
        accel_corrected = accel - jnp.array([b_ax, b_ay])
        gyro_corrected = gyro[0] - b_gz

        # Rotate acceleration from IMU/body frame to world frame
        R = rotation_matrix_2d(theta)
        accel_world = R @ accel_corrected

        # Convert to cm/s²
        accel_corrected_cm = accel_world * 100.0

        # Apply damping
        damping_factor = 1.0 - velocity_damping * dt
        vel_damped = vel * damping_factor

        # Update dynamics
        vel_new = vel_damped + accel_corrected_cm * dt
        pos_new = pos + vel * dt + 0.5 * accel_corrected_cm * dt**2
        theta_new = wrap_angle(theta + gyro_corrected * dt)

        # Biases unchanged
        return jnp.array([
            pos_new[0], pos_new[1],
            vel_new[0], vel_new[1],
            theta_new, b_gz, b_ax, b_ay
        ])

    # Compute Jacobian using automatic differentiation
    jacobian = jax.jacfwd(dynamics_function)(state_array)
    return jacobian


def compute_process_noise(
    dt: float,
    accel_noise_std: float,
    gyro_noise_std: float,
    bias_drift_std: float,
) -> jnp.ndarray:
    """Compute process noise matrix Q for the dynamics model.

    Process noise accounts for:
    - IMU measurement noise propagated through integration
    - Random walk of IMU biases
    - Model uncertainties

    Args:
        dt: Time step (seconds)
        accel_noise_std: Accelerometer noise standard deviation (m/s²)
        gyro_noise_std: Gyroscope noise standard deviation (rad/s)
        bias_drift_std: Bias random walk standard deviation (per √s)

    Returns:
        8x8 process noise matrix Q
    """
    # Convert accelerometer noise to cm/s²
    accel_noise_cm = accel_noise_std * 100.0

    # Position noise from accelerometer (double integration of white noise)
    # For continuous white noise: Var(Δx) = (1/3) * σ_a² * dt³ (not dt⁴)
    # This is the correct Van Loan discrete-time process noise for double integrator
    pos_noise_var = (1.0/3.0) * accel_noise_cm**2 * dt**3

    # Velocity noise from accelerometer (single integration of white noise)
    # For continuous white noise: Var(Δv) = σ_a² * dt (not dt²)
    vel_noise_var = accel_noise_cm**2 * dt

    # Heading noise from gyroscope (single integration of white noise)
    # For continuous white noise: Var(Δθ) = σ_ω² * dt (not dt²)
    heading_noise_var = gyro_noise_std**2 * dt

    # Bias drift (random walk)
    # Var(Δb) = σ_drift² * dt
    bias_noise_var = bias_drift_std**2 * dt

    # Construct diagonal process noise matrix
    Q = jnp.diag(jnp.array([
        pos_noise_var,      # x position
        pos_noise_var,      # y position
        vel_noise_var,      # vx velocity
        vel_noise_var,      # vy velocity
        heading_noise_var,  # θ heading
        bias_noise_var,     # b_gz gyro bias
        bias_noise_var,     # b_ax accel bias
        bias_noise_var,     # b_ay accel bias
    ]))

    # Add cross-correlation between position and velocity (from acceleration)
    # For continuous white noise: Cov(Δx, Δv) = (1/2) * σ_a² * dt² (not dt³)
    pos_vel_cov = 0.5 * accel_noise_cm**2 * dt**2

    # Set cross-correlation terms
    Q = Q.at[0, 2].set(pos_vel_cov)  # Cov(x, vx)
    Q = Q.at[2, 0].set(pos_vel_cov)  # Cov(vx, x)
    Q = Q.at[1, 3].set(pos_vel_cov)  # Cov(y, vy)
    Q = Q.at[3, 1].set(pos_vel_cov)  # Cov(vy, y)

    return Q