"""Regression tests against silent layout fallbacks in ZUPT.

``update_zupt`` historically assumed "unknown state dimension → use 2d_full",
silently producing semantically wrong updates (wrong velocity indices, wrong
measurement Jacobian) whenever the state dimension didn't match any
registered layout. ZUPT's semantics depend on specific layout indices, so
there is no valid fallback -- it must raise at ingress.

``build_Q_rate`` has a similar unknown-dimension fallback, but that fallback
is intentional for state-layout extensibility (see
``tests/runtime/test_offline_state_dim.py`` -- it exercises the smoother
with experimental 3-, 7-, 9-dim states). We therefore keep ``build_Q_rate``
permissive and only guard ``update_zupt``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig
from trodestrack.models.filter_common import FilterState, update_zupt

# Dimensions that do NOT correspond to any entry in LAYOUT_REGISTRY.
# 5 = vision_only, 8 = 2d_full, 10 = 2d_cam_3d_imu, 15 = 3d_euler, 16 = 3d_quat
UNKNOWN_DIMS = [3, 7, 9, 11, 13]


@pytest.mark.parametrize("n", UNKNOWN_DIMS)
def test_update_zupt_raises_for_unknown_state_dimension(n: int):
    """ZUPT must not silently fall back to the 2d_full layout.

    Prior behavior: ``update_zupt`` picked ``get_layout("2d_full")`` whenever
    the state dimension didn't match any registered layout. That meant a
    state-layout wiring bug elsewhere produced numerically finite but
    semantically wrong ZUPT updates (wrong velocity indices, wrong
    measurement Jacobian). The function must now raise at ingress.
    """
    state = FilterState(mean=jnp.zeros(n), cov=jnp.eye(n) * 0.01)
    config = EKFConfig(enable_zupt=True, state_mode="2d_full")
    with pytest.raises(ValueError, match="state dimension"):
        update_zupt(state, config)


def test_update_zupt_still_runs_for_registered_dimensions():
    """ZUPT must still work for dimensions that match a registered layout."""
    n = 8  # 2d_full
    state = FilterState(mean=jnp.zeros(n), cov=jnp.eye(n) * 0.01)
    config = EKFConfig(enable_zupt=True, state_mode="2d_full")
    posterior, log_lik = update_zupt(state, config)
    assert posterior.mean.shape == (n,)
    assert posterior.cov.shape == (n, n)
    assert np.all(np.isfinite(np.asarray(posterior.cov)))
    assert np.isfinite(float(log_lik))
