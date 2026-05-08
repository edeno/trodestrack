"""TTL event-location measurement model and stacked Kalman update.

Every TTL event sensor (beam break, zone trigger, RFID reader) collapses to
the same measurement: a 2D point fix at a known anchor with an anisotropic
2x2 measurement covariance. ``EventLocationModel`` consumes resolved
``(anchor, R)`` pairs for each configured source plus a per-frame padded
array of compact source indices. Per camera frame, ``update_event_location``
folds the (up to ``MAX_EVENTS_PER_FRAME``) active events into one block
update; padded slots use a large R and a masked log-likelihood so they do
not perturb the posterior.

The intentional asymmetry to the existing ``MeasurementModel`` protocol:
that protocol is frame-indexed with a fixed ``meas_dim``, while event
sources carry per-frame variable source IDs and a variable number of
events. This wrapper deliberately exposes the per-event ``source_indices``
keyword instead of forcing the protocol shape.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from trodestrack.models.filter_common import (
    FilterState,
    joseph_update,
    psd_solve,
    symmetrize,
)
from trodestrack.models.state_layout import StateLayout

PAD_SENTINEL: int = -1
# Sentinel events get well-conditioned R blocks (identity) AND zero H rows;
# the zero H is what truly gates them out of the gain. Using small R keeps
# the relative diagonal boost in ``psd_solve`` stable when valid R is tiny.
SENTINEL_R_SCALAR: float = 1.0


class EventLocationModel:
    """2D-position measurement model for TTL event sources.

    Parameters
    ----------
    source_anchors : array_like, shape (n_sources, 2)
        World-frame anchor (meters) for each configured source.
    source_covariances : array_like, shape (n_sources, 2, 2)
        World-frame measurement covariance for each source.
    layout : StateLayout
        State index mapping; the model uses ``layout.pos_idx[:2]`` as the
        position selector.
    max_events_per_frame : int
        Static pad width for the per-frame ``source_indices`` array.
    dtype : jnp.dtype, default jnp.float32
        Array dtype.

    Notes
    -----
    - Every event contributes 2 rows to the stacked H/innovation, so the
      stacked dimension is ``2 * max_events_per_frame``.
    - Compact source indices ``-1`` mark padded slots; their R block is
      ``LARGE_R_SCALAR * I`` and the log-likelihood is masked to zero.
    """

    def __init__(
        self,
        source_anchors: jnp.ndarray,
        source_covariances: jnp.ndarray,
        layout: StateLayout,
        max_events_per_frame: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        anchors = jnp.asarray(source_anchors, dtype=dtype)
        covariances = jnp.asarray(source_covariances, dtype=dtype)
        if anchors.ndim != 2 or anchors.shape[1] != 2:
            raise ValueError(
                f"source_anchors must have shape (n_sources, 2); got {anchors.shape}."
            )
        if covariances.shape != (anchors.shape[0], 2, 2):
            raise ValueError(
                "source_covariances must have shape (n_sources, 2, 2) matching "
                f"source_anchors; got {covariances.shape} for "
                f"n_sources={anchors.shape[0]}."
            )
        if max_events_per_frame < 1:
            raise ValueError(
                f"max_events_per_frame must be >= 1; got {max_events_per_frame}."
            )
        # Reject non-finite or non-PSD covariances at the boundary so a
        # bad spec doesn't silently poison every event update through
        # NaN propagation in psd_solve.
        np_cov = np.asarray(covariances)
        if not np.all(np.isfinite(np_cov)):
            raise ValueError("source_covariances contains non-finite values.")
        for i, R in enumerate(np_cov):
            eigvals = np.linalg.eigvalsh(0.5 * (R + R.T))
            if eigvals.min() <= 0.0:
                raise ValueError(
                    f"source_covariances[{i}] is not positive-definite "
                    f"(min eigenvalue {eigvals.min():.3e})."
                )

        self.anchors = anchors
        self.covariances = covariances
        self.layout = layout
        self.max_events_per_frame = int(max_events_per_frame)
        self.dtype = dtype

    @property
    def meas_dim_per_event(self) -> int:
        """Always 2 — every event is a 2D position measurement."""
        return 2

    @property
    def stacked_meas_dim(self) -> int:
        """Stacked measurement dimension across all padded slots."""
        return 2 * self.max_events_per_frame

    # ------------------------------------------------------------------
    # Per-frame predict / jacobian / meas_cov / innovation
    # ------------------------------------------------------------------

    def _pos_selector(self, n: int) -> jnp.ndarray:
        """Return a (2, n) selector for ``layout.pos_idx[:2]``."""
        H_pos = jnp.zeros((2, n), dtype=self.dtype)
        H_pos = H_pos.at[0, self.layout.pos_idx[0]].set(1.0)
        H_pos = H_pos.at[1, self.layout.pos_idx[1]].set(1.0)
        return H_pos

    def _state_position(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        pos_idx = jnp.array(self.layout.pos_idx[:2], dtype=jnp.int32)
        return state_mean[pos_idx].astype(self.dtype)

    def predict(
        self,
        state_mean: jnp.ndarray,
        *,
        source_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """Stacked prediction h(x) = position repeated per event.

        Returns shape ``(2 * max_events_per_frame,)``.
        """
        del source_indices  # h(x) does not depend on which source is active
        pos = self._state_position(state_mean)
        return jnp.tile(pos, self.max_events_per_frame)

    def jacobian(
        self,
        state_mean: jnp.ndarray,
        *,
        source_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """Stacked Jacobian. Sentinel slots get zero rows so they do not pull."""
        n = state_mean.shape[0]
        H_pos = self._pos_selector(n)  # (2, n)
        K = self.max_events_per_frame
        H_full = jnp.tile(H_pos, (K, 1))  # (2K, n)
        # Zero out rows for sentinel events. Combined with sentinel innov=0
        # and well-conditioned sentinel R, this guarantees the Kalman gain
        # for sentinel rows is exactly zero.
        row_mask = self.valid_event_mask(source_indices)
        return H_full * row_mask[:, None].astype(self.dtype)

    def meas_cov(self, *, source_indices: jnp.ndarray) -> jnp.ndarray:
        """Block-diagonal R; sentinel slots use a well-conditioned identity block."""
        valid = source_indices >= 0
        clamped = jnp.maximum(source_indices, 0)
        per_event_R = self.covariances[clamped]  # (K, 2, 2)
        sentinel_R = jnp.eye(2, dtype=self.dtype) * jnp.asarray(
            SENTINEL_R_SCALAR, dtype=self.dtype
        )
        per_event_R = jnp.where(valid[:, None, None], per_event_R, sentinel_R)
        # Build a block-diagonal (2K, 2K) via index scatter. K is static.
        K = self.max_events_per_frame
        block = jnp.zeros((2 * K, 2 * K), dtype=self.dtype)
        for k in range(K):
            block = block.at[2 * k : 2 * k + 2, 2 * k : 2 * k + 2].set(per_event_R[k])
        return block

    def innovation(
        self,
        *,
        source_indices: jnp.ndarray,
        meas_pred: jnp.ndarray,
    ) -> jnp.ndarray:
        """Innovation z - h(x) for K stacked events with padded zero-residual rows."""
        valid = source_indices >= 0
        clamped = jnp.maximum(source_indices, 0)
        per_event_anchor = self.anchors[clamped]  # (K, 2)
        meas_pred_2d = meas_pred.reshape(self.max_events_per_frame, 2)
        per_event_z = jnp.where(
            valid[:, None],
            per_event_anchor,
            meas_pred_2d,  # zero residual on sentinel rows
        )
        innov_2d = per_event_z - meas_pred_2d
        return innov_2d.reshape(self.stacked_meas_dim)

    def valid_event_mask(self, source_indices: jnp.ndarray) -> jnp.ndarray:
        """Per-row validity mask of length ``2K`` (each event contributes 2 rows)."""
        valid = source_indices >= 0
        return jnp.repeat(valid, 2)


# ---------------------------------------------------------------------------
# Update wrapper used by the EKF scan body
# ---------------------------------------------------------------------------


def update_event_location(
    state: FilterState,
    model: EventLocationModel,
    source_indices: jnp.ndarray,
) -> tuple[FilterState, jnp.ndarray]:
    """Apply the stacked TTL event-location update for one camera frame.

    Parameters
    ----------
    state : FilterState
        Predicted state (after IMU propagation, camera, heading, ZUPT updates).
    model : EventLocationModel
        Resolved event-source geometry.
    source_indices : jnp.ndarray, shape (max_events_per_frame,), int
        Compact source indices for the events active in this frame; padded
        with ``-1``.

    Returns
    -------
    state_post : FilterState
        Updated state. If no events are valid this frame, ``state`` is
        returned unchanged and the log-likelihood is exactly 0.0.
    log_lik : jnp.ndarray
        Marginal log-likelihood of the (masked) event measurements.

    Notes
    -----
    - When all entries of ``source_indices`` are ``-1`` the wrapper short-
      circuits via ``lax.cond`` and pays zero update cost.
    - Sentinel rows are masked out of the log-likelihood explicitly so that
      "events config present but no events" is bitwise identical to "no
      ttl_events config".
    """
    valid_mask = source_indices >= 0
    any_valid = jnp.any(valid_mask)

    def no_update(_) -> tuple[FilterState, jnp.ndarray]:
        return state, jnp.asarray(0.0, dtype=state.mean.dtype)

    def do_update(_) -> tuple[FilterState, jnp.ndarray]:
        m_prior, P_prior = state.mean, state.cov

        meas_pred = model.predict(m_prior, source_indices=source_indices)
        H = model.jacobian(m_prior, source_indices=source_indices)
        R = model.meas_cov(source_indices=source_indices)
        innov = model.innovation(source_indices=source_indices, meas_pred=meas_pred)

        S = symmetrize(H @ P_prior @ H.T + R)
        K_gain = psd_solve(S, H @ P_prior).T
        m_post = m_prior + K_gain @ innov
        P_post = joseph_update(P_prior, K_gain, H, R)

        # Masked log-likelihood: project innovation/S onto valid rows. Padded
        # rows have huge R already; the explicit mask makes the
        # zero-event-with-events-config case bitwise equal to the
        # no-events-config case.
        row_mask = model.valid_event_mask(source_indices)
        active_outer = row_mask[:, None] & row_mask[None, :]
        eye = jnp.eye(S.shape[0], dtype=S.dtype)
        S_masked = jnp.where(active_outer, S, eye)
        innov_masked = jnp.where(row_mask, innov, 0.0)
        n_active = jnp.sum(row_mask.astype(state.mean.dtype))
        solved = psd_solve(S_masked, innov_masked)
        sign, logdet = jnp.linalg.slogdet(symmetrize(S_masked))
        logdet = jnp.where(sign > 0, logdet, jnp.asarray(0.0, dtype=state.mean.dtype))
        log_lik = -0.5 * (
            n_active * jnp.log(jnp.asarray(2.0 * np.pi, dtype=state.mean.dtype))
            + logdet
            + jnp.dot(innov_masked, solved)
        )

        return FilterState(mean=m_post, cov=P_post), log_lik

    return lax.cond(any_valid, do_update, no_update, operand=None)
