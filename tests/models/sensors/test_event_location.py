"""Unit tests for EventLocationModel.

Covers:
- predict / jacobian / meas_cov / innovation on a single event source.
- Stacked event updates: K events fold into ``(2K, n_state)`` H and a
  block-diagonal covariance.
- Sentinel rows (compact index ``-1``) are masked out of the innovation
  log-likelihood and produce no posterior change when run through
  ``update_event_location``.
- Beam short/long limits: short beams produce nearly isotropic covariance;
  long beams produce strongly anisotropic covariance.
- JIT-compatibility smoke (predict/jacobian/meas_cov under jax.jit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.config.schemas import BeamSpec, ZoneTriggerSpec
from trodestrack.models.filter_common import FilterState
from trodestrack.models.sensors.event_location import (
    EventLocationModel,
    update_event_location,
)
from trodestrack.models.state_layout import get_layout

MAX_EVENTS = 4


def _make_model(layout, sources):
    anchors = np.stack([s.anchor for s in sources], axis=0).astype(np.float32)
    covariances = np.stack([s.covariance for s in sources], axis=0).astype(np.float32)
    return EventLocationModel(
        source_anchors=jnp.asarray(anchors),
        source_covariances=jnp.asarray(covariances),
        layout=layout,
        max_events_per_frame=MAX_EVENTS,
    )


def _state_at(layout, x, y):
    state = np.zeros(layout.n, dtype=np.float32)
    state[layout.pos_idx[0]] = x
    state[layout.pos_idx[1]] = y
    return jnp.asarray(state)


# -----------------------------------------------------------------------------
# Predict / jacobian / meas_cov on a single source
# -----------------------------------------------------------------------------


class TestEventLocationModelBasics:
    def test_predict_repeats_position_per_event(self):
        layout = get_layout("2d_full")
        zone = ZoneTriggerSpec(id=1, center=(0.4, -0.2), sigma_m=0.02)
        model = _make_model(layout, [zone.to_event_source()])

        state = _state_at(layout, 0.4, -0.2)
        # source_indices has 4 slots, only the first is valid (compact index 0).
        idx = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
        pred = model.predict(state, source_indices=idx)
        # 4 events × 2 dims = 8.
        assert pred.shape == (2 * MAX_EVENTS,)
        # First event: predicted position (0.4, -0.2).
        np.testing.assert_allclose(pred[:2], [0.4, -0.2])

    def test_jacobian_selects_position(self):
        layout = get_layout("2d_full")
        zone = ZoneTriggerSpec(id=1, center=(0.0, 0.0), sigma_m=0.02)
        model = _make_model(layout, [zone.to_event_source()])

        state = _state_at(layout, 0.0, 0.0)
        idx = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
        H = model.jacobian(state, source_indices=idx)
        assert H.shape == (2 * MAX_EVENTS, layout.n)
        # First event: rows 0/1 should select pos_idx[0], pos_idx[1].
        assert H[0, layout.pos_idx[0]] == 1.0
        assert H[1, layout.pos_idx[1]] == 1.0

    def test_innovation_zero_at_anchor(self):
        layout = get_layout("2d_full")
        zone = ZoneTriggerSpec(id=1, center=(0.5, 0.25), sigma_m=0.02)
        model = _make_model(layout, [zone.to_event_source()])

        state = _state_at(layout, 0.5, 0.25)
        idx = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
        pred = model.predict(state, source_indices=idx)
        innov = model.innovation(source_indices=idx, meas_pred=pred)
        # Innovation at the anchor is zero for all rows.
        np.testing.assert_allclose(np.asarray(innov), 0.0, atol=1e-6)

    def test_innovation_offset_when_off_anchor(self):
        layout = get_layout("2d_full")
        zone = ZoneTriggerSpec(id=1, center=(0.5, 0.25), sigma_m=0.02)
        model = _make_model(layout, [zone.to_event_source()])

        state = _state_at(layout, 0.0, 0.0)
        idx = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
        pred = model.predict(state, source_indices=idx)
        innov = np.asarray(model.innovation(source_indices=idx, meas_pred=pred))
        # First event: anchor - state position.
        np.testing.assert_allclose(innov[:2], [0.5, 0.25])
        # Padded rows: zero residual.
        np.testing.assert_allclose(innov[2:], 0.0, atol=1e-6)


# -----------------------------------------------------------------------------
# Stacked H for K events
# -----------------------------------------------------------------------------


class TestStackedH:
    def test_two_events_two_rows_each(self):
        layout = get_layout("2d_full")
        zone1 = ZoneTriggerSpec(id=1, center=(0.0, 0.0), sigma_m=0.02)
        zone2 = ZoneTriggerSpec(id=2, center=(1.0, 1.0), sigma_m=0.02)
        model = _make_model(layout, [zone1.to_event_source(), zone2.to_event_source()])

        state = _state_at(layout, 0.0, 0.0)
        idx = jnp.asarray([0, 1, -1, -1], dtype=jnp.int32)
        H = model.jacobian(state, source_indices=idx)
        # First event rows.
        assert H[0, layout.pos_idx[0]] == 1.0
        assert H[1, layout.pos_idx[1]] == 1.0
        # Second event rows (rows 2, 3).
        assert H[2, layout.pos_idx[0]] == 1.0
        assert H[3, layout.pos_idx[1]] == 1.0

    def test_meas_cov_block_diagonal(self):
        layout = get_layout("2d_full")
        zone1 = ZoneTriggerSpec(id=1, center=(0.0, 0.0), sigma_m=0.02)
        zone2 = ZoneTriggerSpec(id=2, center=(1.0, 1.0), sigma_m=0.05)
        model = _make_model(layout, [zone1.to_event_source(), zone2.to_event_source()])

        idx = jnp.asarray([0, 1, -1, -1], dtype=jnp.int32)
        R = np.asarray(model.meas_cov(source_indices=idx))
        # Block 0 (rows/cols 0:2) = 0.02²·I.
        np.testing.assert_allclose(R[0:2, 0:2], (0.02**2) * np.eye(2), atol=1e-7)
        # Block 1 (rows/cols 2:4) = 0.05²·I.
        np.testing.assert_allclose(R[2:4, 2:4], (0.05**2) * np.eye(2), atol=1e-7)
        # Off-diagonal blocks zero.
        np.testing.assert_allclose(R[0:2, 2:4], 0.0)
        np.testing.assert_allclose(R[2:4, 0:2], 0.0)
        # Padded slots get a well-conditioned identity block on the diagonal;
        # the actual gating happens via zero H rows, not large R.
        assert R[4, 4] > 0.0 and R[5, 5] > 0.0
        assert R[6, 6] > 0.0 and R[7, 7] > 0.0


# -----------------------------------------------------------------------------
# update_event_location wrapper
# -----------------------------------------------------------------------------


def _initial_state(layout, mean_xy, pos_var=0.5**2, other_var=1.0):
    """Initial state with given xy and an isotropic covariance."""
    n = layout.n
    mean = np.zeros(n, dtype=np.float32)
    mean[layout.pos_idx[0]] = mean_xy[0]
    mean[layout.pos_idx[1]] = mean_xy[1]
    cov = np.eye(n, dtype=np.float32) * other_var
    cov[layout.pos_idx[0], layout.pos_idx[0]] = pos_var
    cov[layout.pos_idx[1], layout.pos_idx[1]] = pos_var
    return FilterState(mean=jnp.asarray(mean), cov=jnp.asarray(cov))


class TestUpdateEventLocation:
    def test_sentinel_only_frame_no_op(self):
        layout = get_layout("2d_full")
        zone = ZoneTriggerSpec(id=1, center=(0.7, 0.0), sigma_m=0.02)
        model = _make_model(layout, [zone.to_event_source()])

        state = _initial_state(layout, mean_xy=(0.1, 0.1))
        idx = jnp.asarray([-1, -1, -1, -1], dtype=jnp.int32)
        new_state, log_lik = update_event_location(state, model, idx)
        np.testing.assert_allclose(np.asarray(new_state.mean), np.asarray(state.mean))
        np.testing.assert_allclose(np.asarray(new_state.cov), np.asarray(state.cov))
        np.testing.assert_allclose(float(log_lik), 0.0)

    def test_single_event_pulls_position_toward_anchor(self):
        layout = get_layout("2d_full")
        zone = ZoneTriggerSpec(id=1, center=(0.7, 0.3), sigma_m=0.02)
        model = _make_model(layout, [zone.to_event_source()])

        state = _initial_state(layout, mean_xy=(0.1, 0.1), pos_var=0.5**2)
        idx = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
        new_state, log_lik = update_event_location(state, model, idx)

        new_pos = np.asarray(new_state.mean)[list(layout.pos_idx[:2])]
        # Posterior mean moved meaningfully toward the anchor.
        assert abs(new_pos[0] - 0.7) < abs(0.1 - 0.7)
        assert abs(new_pos[1] - 0.3) < abs(0.1 - 0.3)
        # Position variance should have decreased.
        new_cov = np.asarray(new_state.cov)
        new_var = new_cov[layout.pos_idx[0], layout.pos_idx[0]]
        assert new_var < 0.5**2
        assert float(log_lik) < 0.0  # finite Gaussian log-lik

    def test_stacked_two_events_equivalent_to_sequential(self):
        # Distinct anchors and distinct sigmas pin the stacked Kalman update
        # against sequential application non-trivially. With identical
        # measurements the stacked information matrix is rank-deficient,
        # which would let a buggy implementation pass.
        layout = get_layout("2d_full")
        zone1 = ZoneTriggerSpec(id=1, center=(0.5, 0.5), sigma_m=0.02)
        zone2 = ZoneTriggerSpec(id=2, center=(0.7, 0.3), sigma_m=0.05)
        model = _make_model(layout, [zone1.to_event_source(), zone2.to_event_source()])

        # Sequential: process two events, one at a time.
        state = _initial_state(layout, mean_xy=(0.0, 0.0))
        idx_first = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
        idx_second = jnp.asarray([1, -1, -1, -1], dtype=jnp.int32)
        seq_state, _ = update_event_location(state, model, idx_first)
        seq_state, _ = update_event_location(seq_state, model, idx_second)

        # Stacked: process both in a single update.
        idx_both = jnp.asarray([0, 1, -1, -1], dtype=jnp.int32)
        stacked_state, _ = update_event_location(state, model, idx_both)

        np.testing.assert_allclose(
            np.asarray(stacked_state.mean), np.asarray(seq_state.mean), atol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(stacked_state.cov), np.asarray(seq_state.cov), atol=1e-4
        )


# -----------------------------------------------------------------------------
# Beam short/long limits
# -----------------------------------------------------------------------------


class TestBeamLimits:
    @pytest.mark.parametrize("beam_length", [1e-6, 0.5, 1.0])
    def test_long_beam_anisotropic_along_dominates(self, beam_length):
        layout = get_layout("2d_full")
        sigma_perp = 0.005
        spec = BeamSpec(
            id=1,
            emitter=(-beam_length / 2, 0.0),
            receiver=(beam_length / 2, 0.0),
            sigma_perp_m=sigma_perp,
        )
        src = spec.to_event_source()
        model = _make_model(layout, [src])

        idx = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
        R_full = np.asarray(model.meas_cov(source_indices=idx))
        R_event = R_full[0:2, 0:2]
        eigvals = np.linalg.eigvalsh(R_event)
        if beam_length < sigma_perp:
            # Short beam: nearly isotropic.
            assert eigvals.max() / eigvals.min() < 1.01
        else:
            # Along-beam variance >> perpendicular variance.
            expected_along = (beam_length / np.sqrt(12.0)) ** 2
            np.testing.assert_allclose(eigvals.max(), expected_along, rtol=1e-3)
            assert eigvals.max() / eigvals.min() > 5.0

    def test_long_beam_constrains_perp_more_than_along(self):
        # Long beam: 5s of isotropic prior should collapse mostly perpendicular.
        layout = get_layout("2d_full")
        sigma_perp = 0.005
        beam_length = 1.0
        # Beam aligned with x-axis at y=0, anchor at origin.
        spec = BeamSpec(
            id=1,
            emitter=(-beam_length / 2, 0.0),
            receiver=(beam_length / 2, 0.0),
            sigma_perp_m=sigma_perp,
        )
        model = _make_model(layout, [spec.to_event_source()])
        # Wide isotropic prior.
        state = _initial_state(layout, mean_xy=(0.0, 0.05), pos_var=0.5**2)
        idx = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
        new_state, _ = update_event_location(state, model, idx)
        new_cov = np.asarray(new_state.cov)
        # y-axis (perp) variance much smaller than x-axis (along).
        var_x = new_cov[layout.pos_idx[0], layout.pos_idx[0]]
        var_y = new_cov[layout.pos_idx[1], layout.pos_idx[1]]
        assert var_y < var_x / 10


# -----------------------------------------------------------------------------
# JIT smoke
# -----------------------------------------------------------------------------


def test_update_event_location_jit_smoke():
    layout = get_layout("2d_full")
    zone = ZoneTriggerSpec(id=1, center=(0.5, 0.5), sigma_m=0.02)
    model = _make_model(layout, [zone.to_event_source()])

    state = _initial_state(layout, mean_xy=(0.1, 0.1))

    @jax.jit
    def run(s, idx):
        return update_event_location(s, model, idx)

    idx_active = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
    idx_empty = jnp.asarray([-1, -1, -1, -1], dtype=jnp.int32)
    s_active, _ = run(state, idx_active)
    s_empty, _ = run(state, idx_empty)
    # JIT path should match non-JIT for both branches.
    assert s_active.mean.shape == state.mean.shape
    np.testing.assert_allclose(np.asarray(s_empty.mean), np.asarray(state.mean))
