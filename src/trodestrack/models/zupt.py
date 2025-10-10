from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def H_vel(n: int, dtype=jnp.float32) -> jnp.ndarray:
    """Build a 2xN selector that picks [vx, vy] from the state.

    - Row 0 selects index 2 (vx)
    - Row 1 selects index 3 (vy)
    """
    H = jnp.zeros((2, n), dtype=dtype)
    H = H.at[0, 2].set(1.0)
    H = H.at[1, 3].set(1.0)
    return H


def zupt_model(config: object, state_mean: jnp.ndarray, n: int, dtype=jnp.float32):
    """Construct a branchless zero-velocity measurement model (H, R, y).

    - Innovation y = -[vx, vy]
    - H selects the velocity components
    - R is set to `zupt_measurement_noise` when stationary and very large otherwise
    - Stationarity check is branchless (lax.select) and respects `enable_zupt` flag
    """
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
