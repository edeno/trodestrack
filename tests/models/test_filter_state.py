"""Tests for ``FilterState.create`` validating constructor.

``FilterState`` is a plain ``NamedTuple`` so raw construction stays
permissive (the filter cores build instances inside JIT-compiled scan
bodies where validation can't trace). ``FilterState.create`` is the
host-side construction path that validates shape, finiteness, symmetry,
and positive-definiteness up front so invalid priors fail with a clear
``ValueError`` instead of crashing deep in a JAX trace.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.filter_common import FilterState
from trodestrack.models.state_layout import get_layout


def test_filter_state_create_accepts_valid_input() -> None:
    state = FilterState.create(jnp.zeros(2), jnp.eye(2))
    assert isinstance(state, FilterState)
    assert state.mean.shape == (2,)
    assert state.cov.shape == (2, 2)


def test_filter_state_create_rejects_wrong_shape() -> None:
    # Mean (5,) vs covariance (4, 4) — shape mismatch must raise.
    with pytest.raises(ValueError, match=r"FilterState\.cov must have shape"):
        FilterState.create(jnp.zeros(5), jnp.eye(4))


def test_filter_state_create_rejects_non_psd_cov() -> None:
    # Diagonal with a negative entry is symmetric but not PSD.
    cov = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match=r"strictly positive definite"):
        FilterState.create(jnp.zeros(2), cov)


def test_filter_state_create_rejects_non_symmetric_cov() -> None:
    cov = jnp.array([[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError, match=r"symmetric"):
        FilterState.create(jnp.zeros(2), cov)


def test_filter_state_create_rejects_non_finite_mean() -> None:
    mean = jnp.array([0.0, jnp.nan])
    with pytest.raises(ValueError, match=r"FilterState\.mean contains non-finite"):
        FilterState.create(mean, jnp.eye(2))


def test_filter_state_create_validates_layout_dim_when_provided() -> None:
    # 5-D mean against an 8-D layout (the default 2d_full layout has n=8).
    layout = get_layout("2d_full")
    assert layout.n == 8
    with pytest.raises(ValueError, match=r"layout requires"):
        FilterState.create(jnp.zeros(5), jnp.eye(5), layout=layout)


def test_filter_state_create_accepts_matching_layout() -> None:
    layout = get_layout("2d_full")
    state = FilterState.create(
        jnp.zeros(layout.n),
        jnp.eye(layout.n),
        layout=layout,
    )
    assert state.mean.shape == (layout.n,)


def test_filter_state_create_rejects_non_one_dimensional_mean() -> None:
    # A 2-D "mean" should be rejected up front.
    bad_mean = np.zeros((2, 2))
    with pytest.raises(ValueError, match=r"FilterState\.mean must be 1-D"):
        FilterState.create(bad_mean, jnp.eye(2))
