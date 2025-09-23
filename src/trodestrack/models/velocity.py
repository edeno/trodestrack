"""Velocity pseudo-measurements for enhanced state estimation.

This module implements velocity constraints that can be applied when
vision tracking is reliable:
- Velocity estimation from position sequences
- Velocity pseudo-measurements for Kalman filtering
- Confidence-based gating for velocity constraints
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._solvers import kalman_gain
from .state import State2D, array_to_state, state_to_array


def velocity_measurement(state: State2D) -> jnp.ndarray:
    """Extract velocity measurement from state.

    Args:
        state: Current state

    Returns:
        Velocity measurement [vx, vy] in cm/s
    """
    return jnp.array([state.vx, state.vy])


@jax.jit
def compute_velocity_jacobian(state_array: jnp.ndarray) -> jnp.ndarray:
    """Compute Jacobian of velocity measurement function.

    Args:
        state_array: State vector [x, y, vx, vy, θ, b_gz, b_ax, b_ay]

    Returns:
        2x8 Jacobian matrix for velocity measurement
    """
    # Velocity measurement: h(x) = [vx, vy]
    # Jacobian: ∂h/∂x = [[0, 0, 1, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0]]
    H = jnp.zeros((2, 8))
    H = H.at[0, 2].set(1.0)  # ∂z_vx/∂vx = 1
    H = H.at[1, 3].set(1.0)  # ∂z_vy/∂vy = 1

    return H


def estimate_velocity_from_positions(
    positions: jnp.ndarray,
    timestamps: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate velocity from sequence of positions using least squares.

    Args:
        positions: Array of shape (n, 2) with positions in cm
        timestamps: Array of shape (n,) with timestamps in seconds

    Returns:
        Estimated velocity [vx, vy] in cm/s
    """
    n_points = len(positions)

    if n_points < 2:
        return jnp.zeros(2)

    if n_points == 2:
        # Simple finite difference
        dt = timestamps[1] - timestamps[0]
        velocity = (positions[1] - positions[0]) / dt
        return velocity

    # Least squares fit: position = velocity * (t - t0) + position0
    # Rearrange: position - position0 = velocity * (t - t0)
    t0 = timestamps[0]
    pos0 = positions[0]

    # Design matrix A: each row is [t - t0] for each timestamp
    A = (timestamps - t0).reshape(-1, 1)

    # Target matrix B: each row is [pos - pos0] for each position
    B = positions - pos0

    # Solve A * velocity = B for velocity (2D)
    # velocity = (A^T A)^-1 A^T B
    AtA = A.T @ A
    AtB = A.T @ B

    # Handle singular case (all timestamps equal)
    if AtA[0, 0] < 1e-12:
        return jnp.zeros(2)

    velocity = AtB / AtA[0, 0]  # AtA is 1x1, AtB is 1x2
    velocity = velocity.flatten()

    return velocity


def create_velocity_noise(
    velocity_noise_std: float,
    confidence: float,
) -> jnp.ndarray:
    """Create velocity measurement noise matrix.

    Args:
        velocity_noise_std: Velocity measurement noise std dev (cm/s)
        confidence: Average confidence of position measurements used

    Returns:
        2x2 velocity noise covariance matrix
    """
    # Scale noise by confidence (lower confidence = higher velocity noise)
    scaled_velocity_std = velocity_noise_std / confidence
    velocity_var = scaled_velocity_std**2

    # Diagonal covariance matrix
    R = jnp.diag(jnp.array([velocity_var, velocity_var]))

    return R


def should_use_velocity_constraint(
    avg_confidence: float,
    position_residual_norm: float,
    min_confidence: float = 0.8,
    max_position_residual: float = 2.0,
) -> bool:
    """Determine whether to apply velocity pseudo-measurement.

    Velocity constraints are only applied when position tracking is reliable
    to avoid corrupting the filter with bad velocity estimates.

    Args:
        avg_confidence: Average confidence of recent position measurements
        position_residual_norm: Norm of recent position residuals (cm)
        min_confidence: Minimum confidence threshold
        max_position_residual: Maximum position residual threshold (cm)

    Returns:
        True if velocity constraint should be applied
    """
    confidence_ok = avg_confidence >= min_confidence
    residual_ok = position_residual_norm <= max_position_residual

    return confidence_ok and residual_ok


def velocity_pseudo_measurement_update(
    state: State2D,
    state_covariance: jnp.ndarray,
    observed_velocity: jnp.ndarray,
    velocity_noise: jnp.ndarray,
) -> Tuple[State2D, jnp.ndarray]:
    """Apply velocity pseudo-measurement update to state.

    This implements a standard Kalman filter update using velocity as
    the measurement. Only the velocity components of the state are updated.

    Args:
        state: Current state estimate
        state_covariance: Current state covariance (8x8)
        observed_velocity: Observed velocity [vx, vy] from position tracking
        velocity_noise: Velocity measurement noise covariance (2x2)

    Returns:
        Tuple of (updated_state, updated_covariance)
    """
    # Convert state to array for linear algebra
    state_array = state_to_array(state)

    # Measurement function and Jacobian
    H = compute_velocity_jacobian(state_array)
    predicted_velocity = velocity_measurement(state)

    # Innovation (measurement residual)
    innovation = observed_velocity - predicted_velocity

    # Innovation covariance: S = H P H^T + R
    innovation_cov = H @ state_covariance @ H.T + velocity_noise

    # Kalman gain: K = P H^T S^-1
    K = kalman_gain(state_covariance, H, velocity_noise)

    # State update: x+ = x- + K * innovation
    updated_state_array = state_array + K @ innovation

    # Covariance update: P+ = (I - K H) P-
    I = jnp.eye(8)
    updated_covariance = (I - K @ H) @ state_covariance

    # Convert back to State2D
    updated_state = array_to_state(updated_state_array)

    return updated_state, updated_covariance


def compute_velocity_from_recent_positions(
    positions: jnp.ndarray,
    timestamps: jnp.ndarray,
    confidences: jnp.ndarray,
    window_size: int = 5,
    min_confidence: float = 0.7,
) -> Tuple[Array, Array, bool]:
    """Compute velocity from recent position measurements.

    Args:
        positions: Recent positions array (n, 2)
        timestamps: Recent timestamps array (n,)
        confidences: Recent confidence array (n,)
        window_size: Maximum number of recent frames to use
        min_confidence: Minimum confidence for valid measurements

    Returns:
        Tuple of (velocity, avg_confidence, is_valid)
    """
    # Limit to window size
    n_frames = min(len(positions), window_size)
    positions = positions[-n_frames:]
    timestamps = timestamps[-n_frames:]
    confidences = confidences[-n_frames:]

    # Filter by confidence
    valid_mask = confidences >= min_confidence
    if jnp.sum(valid_mask) < 2:
        # Need at least 2 valid points for velocity estimation
        return jnp.zeros(2), jnp.array(0.0), False

    valid_positions = positions[valid_mask]
    valid_timestamps = timestamps[valid_mask]
    valid_confidences = confidences[valid_mask]

    # Estimate velocity
    velocity = estimate_velocity_from_positions(valid_positions, valid_timestamps)

    # Compute average confidence
    avg_confidence = jnp.mean(valid_confidences)

    return velocity, avg_confidence, True
