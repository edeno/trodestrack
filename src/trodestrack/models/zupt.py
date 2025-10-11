from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from trodestrack.models.state_layout import LAYOUT_REGISTRY


def H_vel(n: int, dtype=jnp.float32) -> jnp.ndarray:
    """Selector matrix picking [vx, vy] from the state.

    Parameters
    ----------
    n : int
        State dimension.
    dtype : jnp.dtype, default jnp.float32
        Array dtype.

    Returns
    -------
    jnp.ndarray
        H matrix (2, n) selecting the first two velocity components.

    Notes
    -----
    Uses StateLayout to find indices where available; falls back to legacy
    indices [2, 3] if no matching layout is found.
    """
    # Find layout for this dimension
    layout = None
    for mode_layout in LAYOUT_REGISTRY.values():
        if mode_layout.n == n:
            layout = mode_layout
            break

    H = jnp.zeros((2, n), dtype=dtype)

    if layout is not None and len(layout.vel_idx) >= 2:
        # Use layout's velocity indices (first two components for 2D ZUPT)
        vx_idx, vy_idx = layout.vel_idx[0], layout.vel_idx[1]
        H = H.at[0, vx_idx].set(1.0)
        H = H.at[1, vy_idx].set(1.0)
    else:
        # Fallback to legacy indices [2, 3] for unknown layouts
        H = H.at[0, 2].set(1.0)
        H = H.at[1, 3].set(1.0)

    return H


def zupt_model(config: object, state_mean: jnp.ndarray, n: int, dtype=jnp.float32):
    """Branchless zero-velocity measurement model (H, R, y).

    Parameters
    ----------
    config : object
        Filter configuration with ZUPT settings.
    state_mean : jnp.ndarray
        Current state estimate (n,).
    n : int
        State dimension.
    dtype : jnp.dtype, default jnp.float32
        Array dtype.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(H, R, innovation)`` where H is (2, n), R is (2, 2) ((m/s)^2), and
        innovation is (2,) = −[vx, vy].

    Notes
    -----
    Stationarity check uses ``lax.select`` to avoid branching and respects
    ``enable_zupt``. ``R`` is small when stationary and 1e6 otherwise.
    """
    # Find layout for this dimension to get velocity indices
    layout = None
    for mode_layout in LAYOUT_REGISTRY.values():
        if mode_layout.n == n:
            layout = mode_layout
            break

    if layout is not None and len(layout.vel_idx) >= 2:
        vx_idx, vy_idx = layout.vel_idx[0], layout.vel_idx[1]
        vx, vy = state_mean[vx_idx], state_mean[vy_idx]
    else:
        # Fallback to legacy indices [2, 3]
        vx, vy = state_mean[2], state_mean[3]

    v_mag = jnp.sqrt(vx**2 + vy**2)

    is_stationary = (v_mag < getattr(config, "zupt_velocity_threshold", 0.05)) & getattr(
        config, "enable_zupt", False
    )

    R_scalar = lax.select(
        is_stationary,
        jnp.asarray(getattr(config, "zupt_measurement_noise", 0.01**2), dtype=dtype),
        jnp.asarray(1e6, dtype=dtype),
    )
    R = jnp.diag(jnp.array([R_scalar, R_scalar], dtype=dtype))

    H = H_vel(n, dtype=dtype)
    innovation = -H @ state_mean

    return H, R, innovation
