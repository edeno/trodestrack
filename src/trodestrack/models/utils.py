"""Shared utilities for state-space models (EKF, UKF, etc.)."""

from __future__ import annotations

import jax.numpy as jnp


def build_G_matrix(theta: float, dt: float) -> jnp.ndarray:
    """Build IMU input noise propagation matrix (G) for standard 8-state model.

    Constructs the Jacobian ∂f/∂u that maps IMU measurement noise
    into state space for process noise covariance calculation:
        Q_total = Q_process + G @ Q_imu @ G.T

    Args:
        theta: Current heading angle in radians
        dt: Time step in seconds

    Returns:
        G matrix, shape (8, 3) mapping IMU noise [ω_z, f_x, f_y] to state

    State vector: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    Input vector: [ω_z, f_x, f_y]

    Dynamics dependencies:
        - θ_{k+1} = θ_k + (ω_z - b_gz) * dt
          → ∂θ/∂ω_z = dt

        - v_{k+1} = v_k + R(θ)(f - b_a) * dt
          → ∂v/∂f = R(θ) * dt

        - p_{k+1} = p_k + v * dt + 0.5 * R(θ)(f - b_a) * dt²
          → ∂p/∂f = R(θ) * 0.5 * dt²

    where R(θ) is the 2D rotation matrix.

    Example:
        >>> import jax.numpy as jnp
        >>> G = build_G_matrix(theta=0.0, dt=0.005)
        >>> G.shape
        (8, 3)
        >>> G[4, 0]  # θ depends on ω_z with gain dt
        0.005
        >>> G[0:2, 1:3]  # position depends on force with gain 0.5*dt²
        array([[1.25e-05, 0.00e+00],
               [0.00e+00, 1.25e-05]])
    """
    # 2D rotation matrix R(θ)
    c, s = jnp.cos(theta), jnp.sin(theta)
    R_2d = jnp.array([[c, -s], [s, c]])

    # Initialize G matrix: state (8) × input (3)
    # Rows: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    # Cols: [ω_z, f_x, f_y]
    G = jnp.zeros((8, 3))

    # Heading depends on gyro: ∂θ/∂ω_z = dt
    G = G.at[4, 0].set(dt)

    # Velocity depends on accelerometer via rotation: ∂v/∂f = R(θ) * dt
    G = G.at[2:4, 1:3].set(R_2d * dt)

    # Position depends on accelerometer: ∂p/∂f = R(θ) * 0.5 * dt²
    G = G.at[0:2, 1:3].set(R_2d * (0.5 * dt * dt))

    return G


def build_G_matrix_generic(
    n: int,
    theta: float,
    dt: float,
    *,
    pos_idx: tuple[int, int] = (0, 1),
    vel_idx: tuple[int, int] = (2, 3),
    theta_idx: int = 4,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Generic IMU input noise mapping G for arbitrary state layouts.

    - Places ∂θ/∂ω_z = dt at theta_idx
    - Places ∂v/∂f = R(θ)·dt at vel_idx
    - Places ∂p/∂f = R(θ)·0.5·dt² at pos_idx
    Missing indices (out of bounds) are ignored.
    """
    G = jnp.zeros((n, 3), dtype=dtype)
    c, s = jnp.cos(theta), jnp.sin(theta)
    R_2d = jnp.array([[c, -s], [s, c]], dtype=dtype)

    # Heading
    if 0 <= theta_idx < n:
        G = G.at[theta_idx, 0].set(dt)

    # Velocity
    vx_i, vy_i = vel_idx
    if 0 <= vx_i < n and 0 <= vy_i < n:
        G = G.at[vx_i : vy_i + 1, 1:3].set(R_2d * dt)

    # Position
    px_i, py_i = pos_idx
    if 0 <= px_i < n and 0 <= py_i < n:
        G = G.at[px_i : py_i + 1, 1:3].set(R_2d * (0.5 * dt * dt))

    return G
