"""Regression tests against silent layout fallbacks in ZUPT.

``update_zupt`` historically picked a layout from ``LAYOUT_REGISTRY`` by
matching ``state.dim`` to ``layout.n``. Two layouts that share a
dimension would silently shadow each other; an unknown dimension fell
back to ``2d_full`` and produced semantically wrong updates (wrong
velocity indices, wrong measurement Jacobian).

``update_zupt`` now requires an explicit ``layout`` keyword argument and
validates that ``state.dim == layout.n``. ZUPT's semantics depend on
specific layout indices, so there is no valid fallback -- it must raise
at ingress.

``build_Q_rate`` has a similar unknown-dimension fallback, but that
fallback is intentional for state-layout extensibility (see
``tests/runtime/test_offline_state_dim.py`` -- it exercises the smoother
with experimental 3-, 7-, 9-dim states). We therefore keep
``build_Q_rate`` permissive and only guard ``update_zupt``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig
from trodestrack.models.filter_common import FilterState, update_zupt
from trodestrack.models.state_layout import get_layout

# Dimensions that do NOT correspond to any entry in LAYOUT_REGISTRY.
# 5 = vision_only, 8 = 2d_full, 10 = 2d_cam_3d_imu, 15 = 3d_euler, 16 = 3d_quat
UNKNOWN_DIMS = [3, 7, 9, 11, 13]


@pytest.mark.parametrize("n", UNKNOWN_DIMS)
def test_update_zupt_raises_when_state_dim_mismatches_layout(n: int):
    """ZUPT must not silently accept a layout that doesn't match the state.

    Prior behavior: ``update_zupt`` looked up the layout by state
    dimension via a registry walk, so a state-layout wiring bug
    elsewhere produced numerically finite but semantically wrong ZUPT
    updates. The function now requires an explicit ``layout`` and
    raises if its dimension does not match the state.
    """
    state = FilterState(mean=jnp.zeros(n), cov=jnp.eye(n) * 0.01)
    config = EKFConfig(enable_zupt=True, state_mode="2d_full")
    layout = get_layout("2d_full")  # n=8
    with pytest.raises(ValueError, match="state has dim"):
        update_zupt(state, config, layout=layout)


def test_update_zupt_requires_explicit_layout_keyword():
    """Calling without the ``layout`` keyword is a ``TypeError``."""
    state = FilterState(mean=jnp.zeros(8), cov=jnp.eye(8) * 0.01)
    config = EKFConfig(enable_zupt=True, state_mode="2d_full")
    with pytest.raises(TypeError):
        update_zupt(state, config)  # type: ignore[call-arg]


def test_update_zupt_still_runs_for_matching_layout():
    """ZUPT must still work when ``layout.n`` matches the state dimension."""
    n = 8  # 2d_full
    state = FilterState(mean=jnp.zeros(n), cov=jnp.eye(n) * 0.01)
    config = EKFConfig(enable_zupt=True, state_mode="2d_full")
    posterior, log_lik = update_zupt(state, config, layout=get_layout("2d_full"))
    assert posterior.mean.shape == (n,)
    assert posterior.cov.shape == (n, n)
    assert np.all(np.isfinite(np.asarray(posterior.cov)))
    assert np.isfinite(float(log_lik))
