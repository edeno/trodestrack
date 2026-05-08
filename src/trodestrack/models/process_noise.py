from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
from jax import Array

from trodestrack.models.filter_common import build_G_matrix_generic, symmetrize
from trodestrack.models.quaternion import rotate_vector_body_to_world
from trodestrack.models.state_layout import (
    LAYOUT_REGISTRY,
    StateLayout,
    get_heading_index,
)


def _get_layout_for_dimension(n: int) -> StateLayout | None:
    """Find layout matching state dimension n.

    Parameters
    ----------
    n : int
        State dimension.

    Returns
    -------
    StateLayout | None
        Matching layout, or None if no layout matches.
    """
    for mode_layout in LAYOUT_REGISTRY.values():
        if mode_layout.n == n:
            return mode_layout
    return None


def build_Q_rate(config: Any, n: int, dtype: Any = jnp.float32) -> jnp.ndarray:
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
    Supported layouts include 5D (vision-only), 8D (2D full), 10D (2D cam +
    3D IMU), etc. If ``n`` does not match any known layout, falls back to a
    uniform diagonal filled with ``process_noise_pos``.

    This permissive fallback is **intentional** for state-dimension
    extensibility -- see ``tests/runtime/test_offline_state_dim.py`` -- so
    that the smoother infrastructure can be exercised with experimental state
    layouts that have not yet been registered. Filter/smoother entrypoints
    validate ``state_mode`` against the registry; hitting this branch is only
    possible via low-level calls with a non-registry dimension.
    """
    layout = _get_layout_for_dimension(n)

    if layout is None:
        # Extensibility fallback (see docstring note above).
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
        heading_indices = cast(tuple[int, ...], layout.heading_idx)
        for idx in heading_indices:
            diag = diag.at[idx].set(config.process_noise_heading)

    # Gyro bias noise
    for idx in layout.bias_gyro_idx:
        diag = diag.at[idx].set(config.process_noise_gyro_bias)

    # Accel bias noise
    for idx in layout.bias_accel_idx:
        diag = diag.at[idx].set(config.process_noise_accel_bias)

    return jnp.diag(diag)


def build_input_noise_cov(
    config: Any,
    dt: float | Array,
    n_accel: int = 2,
    dtype: Any = jnp.float32,
    n_gyro: int = 1,
) -> jnp.ndarray:
    """IMU input noise covariance from noise densities.

    Parameters
    ----------
    config : Any
        Filter configuration with IMU noise densities.
    dt : float
        Sample period (s).
    n_accel : int, default 2
        Number of accelerometer axes (2 for 2D, 3 for 3D).
    dtype : jnp.dtype, default jnp.float32
        Array dtype.
    n_gyro : int, default 1
        Number of gyroscope axes (1 for yaw-only, 3 for quaternion orientation).

    Returns
    -------
    jnp.ndarray
        Qu (n_gyro+n_accel, n_gyro+n_accel), for example
        [ω_z, f_x, f_y] or [ω_x, ω_y, ω_z, f_x, f_y, f_z].

    Raises
    ------
    ValueError
        If n_accel is not 2 or 3, or n_gyro is not 1 or 3.

    Notes
    -----
    All accelerometer axes are assumed to have the same noise density.
    """
    if n_accel not in (2, 3):
        raise ValueError(f"n_accel must be 2 or 3, got {n_accel}")
    if n_gyro not in (1, 3):
        raise ValueError(f"n_gyro must be 1 or 3, got {n_gyro}")

    dt_arr = jnp.asarray(dt, dtype=dtype)
    sg = config.imu_gyro_noise_density**2 / dt_arr
    sa = config.imu_accel_noise_density**2 / dt_arr

    # Build diagonal: gyro axes first, then accelerometer axes.
    diag = jnp.concatenate(
        [
            jnp.full(n_gyro, sg, dtype=dtype),
            jnp.full(n_accel, sa, dtype=dtype),
        ]
    )
    Qu = jnp.diag(diag)
    return symmetrize(Qu)


def assemble_Q(
    config: Any,
    theta: float | Array,
    dt: float | Array,
    n: int,
    *,
    has_vision: bool | Array,
    dtype=jnp.float32,
    G_override: jnp.ndarray | None = None,
    Qu_override: jnp.ndarray | None = None,
    orientation_quaternion: jnp.ndarray | None = None,
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
    orientation_quaternion : jnp.ndarray | None, optional
        Scalar-first body-to-world quaternion used to map full 3-axis
        accelerometer input noise for quaternion layouts. If omitted, yaw-only
        mapping is used as a conservative planar approximation.

    Returns
    -------
    jnp.ndarray
        Total process noise Q (n, n), symmetrized.

    Notes
    -----
    Q = Q_rate·dt (+ blackout scaling) + G Qu Gᵀ.

    Applies optional blackout scaling for pos/vel/bias components and IMU input
    noise reduction, and can freeze biases by zeroing corresponding rows/cols.

    **Dropout Behavior:**
    - ``reduce_imu_noise_during_blackout=True``: Scales IMU input noise Qu by
      ``blackout_imu_noise_scale`` (typically < 1) during vision gaps, reducing
      noise driving the G Qu Gᵀ term. This prevents IMU noise from inflating
      uncertainty when vision is unavailable.
    - ``freeze_bias_during_blackout=True``: Zeros bias random walks during
      vision gaps to prevent bias drift when biases are unobservable.
    - Combine both flags to stabilize filter behavior during extended occlusions.
    """
    # Find layout for this dimension (used for adaptive Q, n_accel, G matrix, bias freeze)
    layout: StateLayout | None = _get_layout_for_dimension(n)

    # Determine number of accelerometer axes from layout
    # Infer from bias count: 2 bias terms (b_ax, b_ay) → 2D accel
    #                       3 bias terms (b_ax, b_ay, b_az) → 3D accel
    # Default to 2 for backward compatibility (2D tracking with 2D accel)
    n_accel = 2
    n_gyro = 1
    if layout is not None:
        n_accel = len(layout.bias_accel_idx) if len(layout.bias_accel_idx) > 0 else 2
        if layout.has_quaternion_orientation:
            n_gyro = 3

    # Base random-walk diffusion (time-scaled)
    Q_rate = build_Q_rate(config, n, dtype=dtype) * jnp.asarray(dt, dtype=dtype)

    # Apply adaptive Q scaling during dropout (dimension-agnostic)
    if getattr(config, "adaptive_q_during_dropout", False):
        if layout is not None:
            one = jnp.asarray(1.0, dtype=dtype)
            pos_mult = jnp.asarray(
                getattr(config, "dropout_q_pos_multiplier", 1.0), dtype=dtype
            )
            vel_mult = jnp.asarray(
                getattr(config, "dropout_q_vel_multiplier", 1.0), dtype=dtype
            )
            bias_mult = jnp.asarray(
                getattr(config, "dropout_q_bias_multiplier", 1.0), dtype=dtype
            )

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
    Qu = (
        Qu_override
        if Qu_override is not None
        else build_input_noise_cov(
            config, dt, n_accel=n_accel, dtype=dtype, n_gyro=n_gyro
        )
    )
    if getattr(config, "reduce_imu_noise_during_blackout", False):
        scale = jnp.asarray(
            getattr(config, "blackout_imu_noise_scale", 1.0), dtype=dtype
        )
        Qu = Qu * jnp.where(has_vision, jnp.asarray(1.0, dtype=dtype), scale)

    # Build G matrix using layout information
    if G_override is not None:
        Q = Q_rate + G_override @ Qu @ G_override.T
    else:
        if layout is not None and layout.has_biases:
            # Use layout to build dimension-agnostic G matrix
            # For 2D heading, use scalar heading_idx; for 3D, skip IMU mapping for now
            if layout.has_heading_2d:
                pos_pair = (layout.pos_idx[0], layout.pos_idx[1])
                # Use full velocity tuple (2D or 3D depending on layout)
                vel_tuple = cast(
                    tuple[int, int] | tuple[int, int, int], tuple(layout.vel_idx)
                )
                theta_idx = get_heading_index(layout)
                G = build_G_matrix_generic(
                    n,
                    jnp.asarray(theta, dtype=dtype),
                    jnp.asarray(dt, dtype=dtype),
                    pos_idx=pos_pair,
                    vel_idx=vel_tuple,
                    theta_idx=theta_idx,
                    n_accel=n_accel,
                    dtype=dtype,
                )
                Q = Q_rate + G @ Qu @ G.T
            elif layout.has_quaternion_orientation:
                G = jnp.zeros((n, n_gyro + n_accel), dtype=dtype)
                quat_indices = cast(tuple[int, int, int, int], layout.heading_idx)
                half_dt = 0.5 * jnp.asarray(dt, dtype=dtype)
                # Gyro input noise is applied to the vector quaternion
                # components. Independent diffusion on qw remains in Q_rate as
                # a conservative prototype covariance term; normalization keeps
                # the state on the unit-quaternion manifold after prediction.
                for row, col in zip(quat_indices[1:], range(3), strict=True):
                    G = G.at[row, col].set(half_dt)

                if getattr(config, "enable_experimental_accel_translation", False):
                    accel_col = n_gyro
                    dt_arr = jnp.asarray(dt, dtype=dtype)
                    half_dt2 = 0.5 * dt_arr**2
                    if orientation_quaternion is None:
                        c, s = jnp.cos(theta), jnp.sin(theta)
                        rotation_world_from_body = jnp.array(
                            [
                                [c, -s, 0.0],
                                [s, c, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=dtype,
                        )
                    else:
                        rotation_world_from_body = rotate_vector_body_to_world(
                            orientation_quaternion,
                            jnp.eye(3, dtype=dtype),
                        ).T

                    for dim, (pos_i, vel_i) in enumerate(
                        zip(layout.pos_idx, layout.vel_idx, strict=True)
                    ):
                        for axis in range(3):
                            coeff = rotation_world_from_body[dim, axis]
                            G = G.at[vel_i, accel_col + axis].set(dt_arr * coeff)
                            G = G.at[pos_i, accel_col + axis].set(half_dt2 * coeff)

                Q = Q_rate + G @ Qu @ G.T
            else:
                # 3D Euler orientation: no simple IMU mapping yet, use diffusion only
                Q = Q_rate
        else:
            # No layout or no biases: diffusion only
            Q = Q_rate

    # Optionally freeze bias random walks during dropout (dimension-agnostic)
    if getattr(config, "freeze_bias_during_blackout", False):
        # Reuse layout from earlier (already looked up at top of function)
        if layout is not None and layout.has_biases:
            # Vectorized bias freeze: mask rows/cols without Python loop
            freeze_factor = jnp.where(
                has_vision, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
            )
            bias_indices = jnp.array(
                list(layout.bias_gyro_idx) + list(layout.bias_accel_idx),
                dtype=jnp.int32,
            )
            n = Q.shape[0]
            row_mask = jnp.ones((n,), dtype=dtype).at[bias_indices].set(freeze_factor)
            Q = Q * row_mask[:, None] * row_mask[None, :]

    return symmetrize(Q)
