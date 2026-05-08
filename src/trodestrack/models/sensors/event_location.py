"""TTL event-location measurement model and stacked Kalman update.

Every TTL event sensor (beam break, zone trigger, RFID reader) collapses to
the same measurement: a 2D point fix at a known anchor with an anisotropic
2x2 measurement covariance. ``EventLocationModel`` consumes resolved
``(anchor, R)`` pairs for each configured source plus a per-frame padded
array of compact source indices. Per camera frame, ``update_event_location``
folds the (up to ``max_events_per_frame``) active events into one block
update; padded slots get zero H rows + identity R so they contribute
nothing to the gain, and the log-likelihood is masked to valid rows only.

The intentional asymmetry to the existing ``MeasurementModel`` protocol:
that protocol is frame-indexed with a fixed ``meas_dim``, while event
sources carry per-frame variable source IDs and a variable number of
events. This wrapper deliberately exposes the per-event ``source_indices``
keyword instead of forcing the protocol shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import block_diag

from trodestrack.models.filter_common import (
    FilterState,
    gaussian_log_likelihood_masked,
    joseph_update,
    psd_solve,
    symmetrize,
)
from trodestrack.models.state_layout import StateLayout

PAD_SENTINEL: int = -1
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
        # Reject non-finite or non-PSD covariances at the boundary so a bad
        # spec does not silently poison every event update through NaN
        # propagation in psd_solve. Skip under jax.jit (the public
        # extended_kalman_filter validated the source arrays before tracing).
        if not isinstance(covariances, jax.core.Tracer):
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

        # Cache invariants computed once per filter run instead of per frame.
        self._pos_idx = jnp.asarray(layout.pos_idx[:2], dtype=jnp.int32)
        H_pos = jnp.zeros((2, layout.n), dtype=dtype)
        H_pos = H_pos.at[0, layout.pos_idx[0]].set(1.0)
        H_pos = H_pos.at[1, layout.pos_idx[1]].set(1.0)
        self._H_pos = H_pos
        self._H_full = jnp.tile(H_pos, (self.max_events_per_frame, 1))
        self._sentinel_R = jnp.eye(2, dtype=dtype) * jnp.asarray(
            SENTINEL_R_SCALAR, dtype=dtype
        )

    @property
    def meas_dim_per_event(self) -> int:
        return 2

    @property
    def stacked_meas_dim(self) -> int:
        return 2 * self.max_events_per_frame

    def _state_position(self, state_mean: jnp.ndarray) -> jnp.ndarray:
        return state_mean[self._pos_idx].astype(self.dtype)

    def predict(
        self,
        state_mean: jnp.ndarray,
        *,
        source_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        del source_indices  # h(x) does not depend on which source is active
        pos = self._state_position(state_mean)
        return jnp.tile(pos, self.max_events_per_frame)

    def jacobian(
        self,
        state_mean: jnp.ndarray,
        *,
        source_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        del state_mean  # H_pos is shape-independent of state_mean values
        # Sentinel rows are zeroed: combined with sentinel innov=0 and the
        # well-conditioned sentinel R block, this guarantees the Kalman
        # gain on those rows is exactly zero.
        row_mask = self.valid_event_mask(source_indices)
        return self._H_full * row_mask[:, None].astype(self.dtype)

    def meas_cov(self, *, source_indices: jnp.ndarray) -> jnp.ndarray:
        valid = source_indices >= 0
        clamped = jnp.maximum(source_indices, 0)
        per_event_R = self.covariances[clamped]  # (K, 2, 2)
        per_event_R = jnp.where(valid[:, None, None], per_event_R, self._sentinel_R)
        return block_diag(*per_event_R)

    def innovation(
        self,
        *,
        source_indices: jnp.ndarray,
        meas_pred: jnp.ndarray,
    ) -> jnp.ndarray:
        valid = source_indices >= 0
        clamped = jnp.maximum(source_indices, 0)
        per_event_anchor = self.anchors[clamped]  # (K, 2)
        meas_pred_2d = meas_pred.reshape(self.max_events_per_frame, 2)
        # Sentinel rows: anchor = predicted position so the residual is zero.
        per_event_z = jnp.where(valid[:, None], per_event_anchor, meas_pred_2d)
        return (per_event_z - meas_pred_2d).reshape(self.stacked_meas_dim)

    def valid_event_mask(self, source_indices: jnp.ndarray) -> jnp.ndarray:
        valid = source_indices >= 0
        return jnp.repeat(valid, 2)


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

        # Mask the log-likelihood explicitly so "events config present but
        # no events fire" is bitwise equal to "no events config".
        row_mask = model.valid_event_mask(source_indices)
        log_lik = gaussian_log_likelihood_masked(innov, S, row_mask)

        return FilterState(mean=m_post, cov=P_post), log_lik

    return lax.cond(any_valid, do_update, no_update, operand=None)
