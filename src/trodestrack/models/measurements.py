"""Measurement models for position and heading observations.

This module implements the observation models for the EKF/UKF:
- Position measurements from video tracking (DLC/Trodes)
- Heading measurements from LED pairs
- Measurement noise modeling with confidence scaling
- Measurement validation and gating
"""

from typing import Optional

import jax
import jax.numpy as jnp

from .state import State2D

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


def position_measurement(state: State2D) -> jnp.ndarray:
    """Extract position measurement from state.

    Args:
        state: Current state

    Returns:
        Position measurement [x, y] in cm
    """
    return jnp.array([state.x, state.y])


def heading_measurement(led_front: jnp.ndarray, led_back: jnp.ndarray) -> float:
    """Compute heading from LED pair.

    Args:
        led_front: Front LED position [x, y] in pixels
        led_back: Back LED position [x, y] in pixels

    Returns:
        Heading angle in radians (from back to front LED)
    """
    # Vector from back to front LED
    led_vector = led_front - led_back

    # Compute heading angle
    heading = jnp.arctan2(led_vector[1], led_vector[0])

    return heading


@jax.jit
def compute_position_jacobian(state_array: jnp.ndarray) -> jnp.ndarray:
    """Compute Jacobian of position measurement function.

    Args:
        state_array: State vector [x, y, vx, vy, θ, b_gz, b_ax, b_ay]

    Returns:
        2x8 Jacobian matrix for position measurement
    """
    # Position measurement: h(x) = [x, y]
    # Jacobian: ∂h/∂x = [[1, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 1, 0, 0, 0, 0, 0, 0]]
    H = jnp.zeros((2, 8))
    H = H.at[0, 0].set(1.0)  # ∂z_x/∂x = 1
    H = H.at[1, 1].set(1.0)  # ∂z_y/∂y = 1

    return H


@jax.jit
def compute_heading_jacobian(state_array: jnp.ndarray) -> jnp.ndarray:
    """Compute Jacobian of heading measurement function.

    Args:
        state_array: State vector [x, y, vx, vy, θ, b_gz, b_ax, b_ay]

    Returns:
        1x8 Jacobian matrix for heading measurement
    """
    # Heading measurement: h(x) = θ
    # Jacobian: ∂h/∂x = [0, 0, 0, 0, 1, 0, 0, 0]
    H = jnp.zeros((1, 8))
    H = H.at[0, 4].set(1.0)  # ∂z_θ/∂θ = 1

    return H


def validate_led_measurement(
    led_front: jnp.ndarray,
    led_back: jnp.ndarray,
    expected_spacing: float,
    tolerance: float,
) -> bool:
    """Validate LED measurement based on expected spacing.

    Args:
        led_front: Front LED position [x, y]
        led_back: Back LED position [x, y]
        expected_spacing: Expected distance between LEDs (pixels)
        tolerance: Tolerance for spacing validation (pixels)

    Returns:
        True if LED measurement is valid, False otherwise
    """
    # Compute actual spacing
    actual_spacing = jnp.linalg.norm(led_front - led_back)

    # Check if within tolerance
    return jnp.abs(actual_spacing - expected_spacing) <= tolerance


def create_measurement_noise(
    position_noise_std: float,
    confidence: float,
    has_heading: bool,
    heading_noise_std: Optional[float] = None,
) -> jnp.ndarray:
    """Create measurement noise matrix R.

    Args:
        position_noise_std: Position measurement noise std dev (cm)
        confidence: Detection confidence [0, 1] (scales position noise)
        has_heading: Whether heading measurement is available
        heading_noise_std: Heading measurement noise std dev (radians)

    Returns:
        Measurement noise matrix R
    """
    # Scale position noise by confidence (lower confidence = higher noise)
    scaled_position_std = position_noise_std / confidence
    position_var = scaled_position_std ** 2

    if has_heading:
        if heading_noise_std is None:
            raise ValueError("heading_noise_std required when has_heading=True")

        # 3x3 matrix for [x, y, θ]
        heading_var = heading_noise_std ** 2
        R = jnp.diag(jnp.array([position_var, position_var, heading_var]))
    else:
        # 2x2 matrix for [x, y] only
        R = jnp.diag(jnp.array([position_var, position_var]))

    return R


def create_combined_measurement(
    position: jnp.ndarray,
    heading: Optional[float] = None,
) -> jnp.ndarray:
    """Create combined measurement vector.

    Args:
        position: Position measurement [x, y] in cm
        heading: Optional heading measurement in radians

    Returns:
        Combined measurement vector
    """
    if heading is not None:
        return jnp.concatenate([position, jnp.array([heading])])
    else:
        return position


def create_combined_jacobian(
    state_array: jnp.ndarray,
    has_heading: bool,
) -> jnp.ndarray:
    """Create combined measurement Jacobian matrix.

    Args:
        state_array: State vector
        has_heading: Whether heading measurement is available

    Returns:
        Combined Jacobian matrix
    """
    H_pos = compute_position_jacobian(state_array)

    if has_heading:
        H_heading = compute_heading_jacobian(state_array)
        H = jnp.vstack([H_pos, H_heading])
    else:
        H = H_pos

    return H