"""State representation for 2D tracking with IMU integration.

This module implements the 8-dimensional state vector as specified in the PRD:
x_k = [x, y, vx, vy, θ, b_gz, b_ax, b_ay]^T

Where:
- x, y: position in cm
- vx, vy: velocity in cm/s
- θ: heading in radians
- b_gz: gyroscope z-axis bias in rad/s
- b_ax, b_ay: accelerometer x,y biases in m/s²
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from pydantic import BaseModel, ConfigDict

# State dimension as per PRD
STATE_DIM: int = 8


class State2D(BaseModel):
    """2D state representation with IMU biases.

    Attributes:
        x: Position x-coordinate (cm)
        y: Position y-coordinate (cm)
        vx: Velocity x-component (cm/s)
        vy: Velocity y-component (cm/s)
        theta: Heading angle (radians)
        b_gz: Gyroscope z-axis bias (rad/s)
        b_ax: Accelerometer x-axis bias (m/s²)
        b_ay: Accelerometer y-axis bias (m/s²)
    """

    x: ArrayLike
    y: ArrayLike
    vx: ArrayLike
    vy: ArrayLike
    theta: ArrayLike
    b_gz: ArrayLike
    b_ax: ArrayLike
    b_ay: ArrayLike

    model_config = ConfigDict(
        frozen=True,  # Immutable state representation
        arbitrary_types_allowed=True,  # Allow JAX arrays and ArrayLike types
    )


def state_to_array(state: State2D, dtype=None) -> Array:
    """Convert State2D to JAX array.

    Args:
        state: State2D instance
        dtype: Optional dtype override (defaults to JAX default based on global x64 policy)

    Returns:
        8-dimensional JAX array [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    """
    return jnp.array(
        [state.x, state.y, state.vx, state.vy, state.theta, state.b_gz, state.b_ax, state.b_ay],
        dtype=dtype,  # Let JAX use its default dtype (respects global x64 policy)
    )


def array_to_state(arr: ArrayLike) -> State2D:
    """Convert JAX array to State2D.

    Args:
        arr: 8-dimensional array [x, y, vx, vy, θ, b_gz, b_ax, b_ay]

    Returns:
        State2D instance

    Raises:
        ValueError: If array is not 8-dimensional
    """
    if arr.shape != (8,):
        raise ValueError(f"Expected 8-dimensional array, got shape {arr.shape}")

    return State2D(
        x=arr[0],
        y=arr[1],
        vx=arr[2],
        vy=arr[3],
        theta=arr[4],
        b_gz=arr[5],
        b_ax=arr[6],
        b_ay=arr[7],
    )


def create_initial_state(
    positions: ArrayLike,
    timestamps: ArrayLike,
    confidences: ArrayLike,
    homography: ArrayLike,
    led_front: Optional[ArrayLike] = None,
    led_back: Optional[ArrayLike] = None,
) -> Tuple[State2D, Array]:
    """Estimate initial state from early video frames.

    Args:
        positions: Array of shape (n, 2) with pixel positions
        timestamps: Array of shape (n,) with frame timestamps
        confidences: Array of shape (n,) with detection confidences
        homography: 3x3 homography matrix for pixel → cm conversion
        led_front: Optional front LED position for heading estimation
        led_back: Optional back LED position for heading estimation

    Returns:
        Tuple of (initial_state, initial_covariance_matrix)

    Notes:
        - Position: Average of first few valid frames (converted to cm)
        - Velocity: Finite difference from early frames
        - Heading: From LED pair if available, else velocity direction
        - Biases: Initialize to zero
        - Covariance: Conservative uncertainties as per PRD
    """
    # Convert pixel positions to cm using homography
    n_frames = positions.shape[0]

    # Apply homography transformation
    # Convert to homogeneous coordinates
    pos_homogeneous = jnp.concatenate([positions, jnp.ones((n_frames, 1))], axis=1)  # Shape: (n, 3)

    # Transform to cm coordinates
    pos_cm_homogeneous = (homography @ pos_homogeneous.T).T  # Shape: (n, 3)
    pos_cm = pos_cm_homogeneous[:, :2] / pos_cm_homogeneous[:, 2:3]  # Normalize

    # Estimate position as weighted average of early frames
    weights = confidences / jnp.sum(confidences)
    position_cm = jnp.sum(pos_cm * weights.reshape(-1, 1), axis=0)

    # Estimate velocity from finite differences
    if n_frames >= 2:
        # Use first two frames for velocity estimate
        dt = timestamps[1] - timestamps[0]
        velocity_cm_s = (pos_cm[1] - pos_cm[0]) / dt
    else:
        # Single frame - assume zero velocity
        velocity_cm_s = jnp.zeros(2)

    # Estimate heading
    if led_front is not None and led_back is not None:
        # Use LED pair direction (in pixels, then convert orientation)
        led_vector = led_front - led_back
        heading = jnp.arctan2(led_vector[1], led_vector[0])
    else:
        # Fallback to velocity direction
        heading = jnp.arctan2(velocity_cm_s[1], velocity_cm_s[0])

    # Create initial state
    initial_state = State2D(
        x=float(position_cm[0]),
        y=float(position_cm[1]),
        vx=float(velocity_cm_s[0]),
        vy=float(velocity_cm_s[1]),
        theta=float(heading),
        b_gz=0.0,  # Initialize biases to zero
        b_ax=0.0,
        b_ay=0.0,
    )

    # Create initial covariance matrix with conservative uncertainties
    # Values based on PRD guidance: low for position, high for velocity, medium for heading
    initial_cov = jnp.diag(
        jnp.array(
            [
                1.0,  # x position uncertainty (cm²)
                1.0,  # y position uncertainty (cm²)
                100.0,  # vx velocity uncertainty (cm²/s²)
                100.0,  # vy velocity uncertainty (cm²/s²)
                0.25,  # θ heading uncertainty (rad²) ≈ 28 degrees std
                0.01,  # b_gz gyro bias uncertainty (rad²/s²)
                0.01,  # b_ax accel bias uncertainty (m²/s⁴)
                0.01,  # b_ay accel bias uncertainty (m²/s⁴)
            ]
        )
    )

    return initial_state, initial_cov
