from __future__ import annotations

import jax.numpy as jnp

from trodestrack.models.utils import build_G_matrix
from typing import Any


def build_Q_rate(config: Any, n: int, dtype=jnp.float32) -> jnp.ndarray:
    """Diagonal process noise rate Q_rate for standard 8D (and fallback for others).

    Standard 8D mapping: [x, y, vx, vy, θ, b_gz, b_ax, b_ay].
    For non-8D, returns uniform diag using position noise as a conservative fallback.
    """
    if n == 8:
        diag = jnp.array(
            [
                config.process_noise_pos,  # x
                config.process_noise_pos,  # y
                config.process_noise_vel,  # vx
                config.process_noise_vel,  # vy
                config.process_noise_heading,  # θ
                config.process_noise_gyro_bias,  # b_gz
                config.process_noise_accel_bias,  # b_ax
                config.process_noise_accel_bias,  # b_ay
            ],
            dtype=dtype,
        )
        return jnp.diag(diag)
    # Fallback for other dims
    return jnp.diag(jnp.full(n, config.process_noise_pos, dtype=dtype))


def build_input_noise_cov(config: Any, dt: float, dtype=jnp.float32) -> jnp.ndarray:
    """IMU input noise covariance Qu from density params (per-sample variance)."""
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
) -> jnp.ndarray:
    """Assemble total process noise Q = Q_rate*dt (+ blackout scaling) + G Qu G^T.

    - Applies adaptive Q scaling during vision dropout (pos/vel/bias multipliers)
    - Applies reduced IMU input noise during dropout when configured
    - Optionally freezes bias RW during dropout by zeroing bias rows/cols
    - Returns symmetrized matrix for numerical safety
    """
    # Base random-walk diffusion (time-scaled)
    Q_rate = build_Q_rate(config, n, dtype=dtype) * jnp.asarray(dt, dtype=dtype)

    if n == 8:
        # Compute multipliers in a JAX-safe way
        one = jnp.asarray(1.0, dtype=dtype)
        pos_mult = one
        vel_mult = one
        bias_mult = one
        if getattr(config, "adaptive_q_during_dropout", False):
            pos_mult = jnp.asarray(getattr(config, "dropout_q_pos_multiplier", 1.0), dtype=dtype)
            vel_mult = jnp.asarray(getattr(config, "dropout_q_vel_multiplier", 1.0), dtype=dtype)
            bias_mult = jnp.asarray(getattr(config, "dropout_q_bias_multiplier", 1.0), dtype=dtype)
            # Apply only during blackout
            pos_mult = jnp.asarray(jnp.where(has_vision, one, pos_mult), dtype=dtype)
            vel_mult = jnp.asarray(jnp.where(has_vision, one, vel_mult), dtype=dtype)
            bias_mult = jnp.asarray(jnp.where(has_vision, one, bias_mult), dtype=dtype)

        # Scale diagonal blocks
        Q_rate = Q_rate.at[0, 0].set(Q_rate[0, 0] * pos_mult)
        Q_rate = Q_rate.at[1, 1].set(Q_rate[1, 1] * pos_mult)
        Q_rate = Q_rate.at[2, 2].set(Q_rate[2, 2] * vel_mult)
        Q_rate = Q_rate.at[3, 3].set(Q_rate[3, 3] * vel_mult)
        Q_rate = Q_rate.at[5, 5].set(Q_rate[5, 5] * bias_mult)
        Q_rate = Q_rate.at[6, 6].set(Q_rate[6, 6] * bias_mult)
        Q_rate = Q_rate.at[7, 7].set(Q_rate[7, 7] * bias_mult)

    # IMU input noise mapping
    Qu = build_input_noise_cov(config, dt, dtype=dtype)
    if getattr(config, "reduce_imu_noise_during_blackout", False):
        scale = jnp.asarray(getattr(config, "blackout_imu_noise_scale", 1.0), dtype=dtype)
        Qu = Qu * jnp.where(has_vision, jnp.asarray(1.0, dtype=dtype), scale)

    # For standard 8D, use known G(theta, dt); otherwise, zero (no IMU mapping known)
    if n == 8:
        G = build_G_matrix(jnp.asarray(theta, dtype=dtype), jnp.asarray(dt, dtype=dtype))
        Q = Q_rate + G @ Qu @ G.T
    else:
        Q = Q_rate

    # Optionally freeze bias random walks during dropout
    if (n == 8) and getattr(config, "freeze_bias_during_blackout", False):
        # Zero bias rows/cols during blackout in a JAX-safe way
        freeze_factor = jnp.where(
            has_vision, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
        )
        for idx in (5, 6, 7):
            Q = Q.at[idx, :].set(Q[idx, :] * freeze_factor)
            Q = Q.at[:, idx].set(Q[:, idx] * freeze_factor)

    # Symmetrize for numerical hygiene
    return 0.5 * (Q + Q.T)
