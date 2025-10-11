from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp

from trodestrack.models.state_layout import LAYOUT_REGISTRY


def build_Q_rate(config: Any, n: int, dtype=jnp.float32) -> jnp.ndarray:
    """Diagonal process noise-rate matrix Q_rate (dimension-agnostic).

    Parameters
    ----------
    config : Any
        Filter configuration with ``process_noise_*`` parameters.
    n : int
        State dimension.
    dtype : jnp.dtype, default jnp.float32
        Array dtype.

    Returns
    -------
    jnp.ndarray
        Diagonal matrix Q_rate (n, n) with entries mapped via StateLayout.

    Notes
    -----
    Supported layouts include 5D (vision-only), 8D (2D full), 10D (2D cam + 3D IMU),
    etc. If ``n`` does not match any known layout, falls back to a uniform
    diagonal with ``process_noise_pos``.
    """
    # Try to find matching layout
    layout = None
    for mode_layout in LAYOUT_REGISTRY.values():
        if mode_layout.n == n:
            layout = mode_layout
            break

    if layout is None:
        # Fallback for unknown dimensions
        return jnp.diag(jnp.full(n, config.process_noise_pos, dtype=dtype))

    # Build diagonal using layout
    diag = jnp.zeros(n, dtype=dtype)

    # Position noise
    for idx in layout.pos_idx:
        diag = diag.at[idx].set(config.process_noise_pos)

    # Velocity noise
    for idx in layout.vel_idx:
        diag = diag.at[idx].set(config.process_noise_vel)

    # Heading/orientation noise
    if layout.has_heading_2d:
        diag = diag.at[layout.heading_idx].set(config.process_noise_heading)
    else:
        # 3D orientation (Euler or quaternion)
        for idx in layout.heading_idx:
            diag = diag.at[idx].set(config.process_noise_heading)

    # Gyro bias noise
    for idx in layout.bias_gyro_idx:
        diag = diag.at[idx].set(config.process_noise_gyro_bias)

    # Accel bias noise
    for idx in layout.bias_accel_idx:
        diag = diag.at[idx].set(config.process_noise_accel_bias)

    return jnp.diag(diag)


def build_input_noise_cov(config: Any, dt: float, dtype=jnp.float32) -> jnp.ndarray:
    """IMU input noise covariance from noise densities.

    Parameters
    ----------
    config : Any
        Filter configuration with IMU noise densities.
    dt : float
        Sample period (s).
    dtype : jnp.dtype, default jnp.float32
        Array dtype.

    Returns
    -------
    jnp.ndarray
        Qu (3, 3) for [ω_z(rad/s), f_x(m/s^2), f_y(m/s^2)].
    """
    sg = (config.imu_gyro_noise_density * jnp.sqrt(dt)) ** 2
    sa = (config.imu_accel_noise_density * jnp.sqrt(dt)) ** 2
    return jnp.diag(jnp.array([sg, sa, sa], dtype=dtype))


def assemble_Q(
    config: Any,
    theta: float,
    dt: float,
    n: int,
    *,
    has_vision: bool,
    dtype=jnp.float32,
    G_override: Optional[jnp.ndarray] = None,
    Qu_override: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Assemble total process noise matrix.

    Parameters
    ----------
    config : Any
        Filter configuration.
    theta : float
        Heading angle (rad), used for IMU input mapping.
    dt : float
        Time step (s).
    n : int
        State dimension.
    has_vision : bool
        Whether vision is available (affects dropout scaling).
    dtype : jnp.dtype, default jnp.float32
        Array dtype.
    G_override : jnp.ndarray | None, optional
        Optional precomputed input mapping matrix G (n, 3).
    Qu_override : jnp.ndarray | None, optional
        Optional IMU input covariance (3, 3).

    Returns
    -------
    jnp.ndarray
        Total process noise Q (n, n), symmetrized.

    Notes
    -----
    Q = Q_rate·dt (+ blackout scaling) + G Qu Gᵀ.
    Applies optional blackout scaling for pos/vel/bias components and IMU input
    noise reduction, and can freeze biases by zeroing corresponding rows/cols.
    """
    # Base random-walk diffusion (time-scaled)
    Q_rate = build_Q_rate(config, n, dtype=dtype) * jnp.asarray(dt, dtype=dtype)

    # Apply adaptive Q scaling during dropout (dimension-agnostic)
    if getattr(config, "adaptive_q_during_dropout", False):
        # Find layout for this dimension
        layout = None
        for mode_layout in LAYOUT_REGISTRY.values():
            if mode_layout.n == n:
                layout = mode_layout
                break

        if layout is not None:
            one = jnp.asarray(1.0, dtype=dtype)
            pos_mult = jnp.asarray(getattr(config, "dropout_q_pos_multiplier", 1.0), dtype=dtype)
            vel_mult = jnp.asarray(getattr(config, "dropout_q_vel_multiplier", 1.0), dtype=dtype)
            bias_mult = jnp.asarray(getattr(config, "dropout_q_bias_multiplier", 1.0), dtype=dtype)

            # Apply only during blackout
            pos_mult = jnp.asarray(jnp.where(has_vision, one, pos_mult), dtype=dtype)
            vel_mult = jnp.asarray(jnp.where(has_vision, one, vel_mult), dtype=dtype)
            bias_mult = jnp.asarray(jnp.where(has_vision, one, bias_mult), dtype=dtype)

            # Scale position diagonal elements
            for idx in layout.pos_idx:
                Q_rate = Q_rate.at[idx, idx].set(Q_rate[idx, idx] * pos_mult)

            # Scale velocity diagonal elements
            for idx in layout.vel_idx:
                Q_rate = Q_rate.at[idx, idx].set(Q_rate[idx, idx] * vel_mult)

            # Scale bias diagonal elements
            for idx in list(layout.bias_gyro_idx) + list(layout.bias_accel_idx):
                Q_rate = Q_rate.at[idx, idx].set(Q_rate[idx, idx] * bias_mult)

    # IMU input noise mapping
    Qu = Qu_override if Qu_override is not None else build_input_noise_cov(config, dt, dtype=dtype)
    if getattr(config, "reduce_imu_noise_during_blackout", False):
        scale = jnp.asarray(getattr(config, "blackout_imu_noise_scale", 1.0), dtype=dtype)
        Qu = Qu * jnp.where(has_vision, jnp.asarray(1.0, dtype=dtype), scale)

    # Build G matrix using layout information
    if G_override is not None:
        Q = Q_rate + G_override @ Qu @ G_override.T
    else:
        # Find layout for this dimension to build appropriate G matrix
        layout = None
        for mode_layout in LAYOUT_REGISTRY.values():
            if mode_layout.n == n:
                layout = mode_layout
                break

        if layout is not None and layout.has_biases:
            # Use layout to build dimension-agnostic G matrix
            from trodestrack.models.utils import build_G_matrix_generic

            # For 2D heading, use scalar heading_idx; for 3D, skip IMU mapping for now
            if layout.has_heading_2d:
                G = build_G_matrix_generic(
                    n,
                    jnp.asarray(theta, dtype=dtype),
                    jnp.asarray(dt, dtype=dtype),
                    pos_idx=layout.pos_idx[:2],
                    vel_idx=layout.vel_idx[:2],
                    theta_idx=layout.heading_idx,
                    dtype=dtype,
                )
                Q = Q_rate + G @ Qu @ G.T
            else:
                # 3D orientation: no simple IMU mapping yet, use diffusion only
                Q = Q_rate
        else:
            # No layout or no biases: diffusion only
            Q = Q_rate

    # Optionally freeze bias random walks during dropout (dimension-agnostic)
    if getattr(config, "freeze_bias_during_blackout", False):
        # Find layout for this dimension
        layout = None
        for mode_layout in LAYOUT_REGISTRY.values():
            if mode_layout.n == n:
                layout = mode_layout
                break

        if layout is not None and layout.has_biases:
            # Zero bias rows/cols during blackout in a JAX-safe way
            freeze_factor = jnp.where(
                has_vision, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
            )
            bias_indices = list(layout.bias_gyro_idx) + list(layout.bias_accel_idx)
            for idx in bias_indices:
                Q = Q.at[idx, :].set(Q[idx, :] * freeze_factor)
                Q = Q.at[:, idx].set(Q[:, idx] * freeze_factor)

    # Symmetrize for numerical hygiene
    return 0.5 * (Q + Q.T)
