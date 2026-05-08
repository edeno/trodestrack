"""TTL event-location measurement model and stacked Kalman update.

Every TTL event sensor (beam break, zone trigger, RFID reader) collapses to
the same measurement: a 2D point fix at a known anchor with an anisotropic
2x2 measurement covariance. ``EventLocationModel`` consumes resolved
``(anchor, covariance)`` pairs for each configured source plus a per-frame
padded array of compact source indices. Per camera frame, ``update_event_location``
folds the (up to ``max_events_per_frame``) active events into one block
update; padded slots get zero H rows + identity covariance blocks so they
contribute nothing to the gain, and the log-likelihood is masked to valid
rows only.

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

# Must match ``trodestrack.io.ttl_events.PAD_SENTINEL``. Defined locally to
# keep the model-layer free of an io-layer import dependency; the value is a
# single -1 literal that changes only if the entire padding convention does.
PAD_SENTINEL: int = -1
SENTINEL_R_SCALAR: float = 1.0
NO_EVENT_PAD_WIDTH: int = 1


def _coerce_event_floats(arr, *, name: str) -> np.ndarray:
    """Coerce a public-API event float array to ``float64``.

    Rejects bool, complex, object, string, datetime, and other dtypes
    that would silently coerce through ``np.asarray(..., dtype=float)``
    (probe: bool ``True`` → ``1.0``, string ``"0.5"`` → ``0.5``,
    complex ``1+2j`` → ``1.0`` with imaginary part discarded).
    """
    raw = np.asarray(arr)
    if not (
        np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)
    ):
        raise ValueError(
            f"{name} must be a real integer or float array; got "
            f"dtype={raw.dtype!r}. Bool, complex, object, and string "
            "dtypes are rejected to avoid silent coercion."
        )
    return raw.astype(float, copy=False)


def _empty_event_channel(
    n_cam: int, pad_width: int = NO_EVENT_PAD_WIDTH
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """No-op event channel: one dummy source, all-sentinel indices."""
    return (
        jnp.zeros((1, 2), dtype=jnp.float32),
        jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32), (1, 2, 2)),
        jnp.full((n_cam, pad_width), -1, dtype=jnp.int32),
        pad_width,
    )


def resolve_event_inputs(
    event_source_anchors: np.ndarray | None,
    event_source_covariances: np.ndarray | None,
    event_indices_per_frame: np.ndarray | None,
    *,
    n_cam: int,
    func_name: str = "kalman_filter",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Validate optional event arguments and return JAX-ready dense arrays.

    Shared between ``extended_kalman_filter`` and
    ``unscented_kalman_filter``. If all three arguments are ``None`` the
    wrapper builds a no-op event channel (one dummy source, all-sentinel
    indices). If any one is provided, all three are required, validated
    for dtype/shape/range, and returned as JAX arrays.

    The validation here is what protects the JIT'd core: ``EventLocationModel``
    skips its host-side PSD check when arrays are JAX tracers, so this
    function must reject malformed inputs before tracing.
    """
    provided = (
        event_source_anchors is not None,
        event_source_covariances is not None,
        event_indices_per_frame is not None,
    )
    if any(provided) and not all(provided):
        raise ValueError(
            f"{func_name} event channel: event_source_anchors, "
            "event_source_covariances, and event_indices_per_frame must be "
            "provided together (all None disables the channel)."
        )
    if not any(provided):
        return _empty_event_channel(n_cam)

    anchors = _coerce_event_floats(event_source_anchors, name="event_source_anchors")
    covariances = _coerce_event_floats(
        event_source_covariances, name="event_source_covariances"
    )

    raw_indices = np.asarray(event_indices_per_frame)
    if np.issubdtype(raw_indices.dtype, np.integer):
        # ``np.uint64(2**64 - 1).astype(int64) == -1``, which would silently
        # match the padded-sentinel convention and drop a real event row.
        if np.issubdtype(raw_indices.dtype, np.unsignedinteger) and raw_indices.size:
            int64_max = np.iinfo(np.int64).max
            overflow = raw_indices > np.uint64(int64_max)
            if overflow.any():
                bad = sorted({int(x) for x in raw_indices[overflow][:5]})
                raise ValueError(
                    "event_indices_per_frame contains values above the "
                    f"signed int64 range (max {int64_max}); these would "
                    "wrap to negative ids and silently match the padded "
                    f"sentinel. Got entries like {bad}."
                )
        indices = raw_indices.astype(np.int64, copy=False)
    elif np.issubdtype(raw_indices.dtype, np.floating):
        bad_idx = ~np.isfinite(raw_indices) | (raw_indices != np.floor(raw_indices))
        if bad_idx.any():
            raise ValueError(
                "event_indices_per_frame must contain integer compact "
                "source indices; got non-integer entries like "
                f"{sorted({float(x) for x in raw_indices[bad_idx][:5]})}."
            )
        indices = raw_indices.astype(np.int64)
    else:
        raise ValueError(
            "event_indices_per_frame must be an integer or float array; "
            f"got dtype={raw_indices.dtype!r}. Bool, object, and string "
            "dtypes are rejected to avoid silent coercion."
        )

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(
            f"event_source_anchors must have shape (n_sources, 2); got {anchors.shape}."
        )
    n_sources = anchors.shape[0]
    if covariances.shape != (n_sources, 2, 2):
        raise ValueError(
            "event_source_covariances must have shape (n_sources, 2, 2) "
            f"matching event_source_anchors; got {covariances.shape} for "
            f"n_sources={n_sources}."
        )
    if not np.all(np.isfinite(anchors)) or not np.all(np.isfinite(covariances)):
        raise ValueError(
            "event_source_anchors / event_source_covariances must be finite."
        )
    # The stricter ``EventLocationModel`` PSD check is skipped under jax.jit,
    # so direct callers (not just YAML-driven sessions) must validate
    # symmetry and positive-definiteness here. Otherwise an asymmetric or
    # negative-definite R silently yields NaN log-likelihoods and posteriors.
    sym = 0.5 * (covariances + covariances.transpose(0, 2, 1))
    asymmetry = np.max(np.abs(covariances - sym), axis=(1, 2))
    if asymmetry.size and asymmetry.max() > 1e-9:
        bad = int(np.argmax(asymmetry))
        raise ValueError(
            f"event_source_covariances[{bad}] is not symmetric "
            f"(asymmetry {asymmetry[bad]:.3e}); 2x2 covariances must be PSD."
        )
    for i, R in enumerate(sym):
        eigvals = np.linalg.eigvalsh(R)
        if eigvals.min() <= 0.0:
            raise ValueError(
                f"event_source_covariances[{i}] is not positive-definite "
                f"(min eigenvalue {eigvals.min():.3e})."
            )
    if indices.ndim != 2 or indices.shape[0] != n_cam:
        raise ValueError(
            "event_indices_per_frame must have shape (len(t_cam), "
            f"max_events_per_frame); got {indices.shape} for n_cam={n_cam}."
        )
    valid_idx = indices >= 0
    if valid_idx.any() and (indices[valid_idx].max() >= n_sources):
        raise ValueError(
            "event_indices_per_frame contains a compact source index out of "
            f"range [0, {n_sources}); got max "
            f"{int(indices[valid_idx].max())}."
        )
    if (indices[~valid_idx] != -1).any():
        raise ValueError(
            "event_indices_per_frame padded entries must be exactly -1; "
            "negative values other than -1 are not allowed."
        )
    if n_sources == 0:
        return _empty_event_channel(n_cam)

    return (
        jnp.asarray(anchors, dtype=jnp.float32),
        jnp.asarray(covariances, dtype=jnp.float32),
        jnp.asarray(indices, dtype=jnp.int32),
        int(indices.shape[1]),
    )


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
        # Direct callers (tests, custom pipelines) skip
        # ``resolve_event_inputs``, so guard the constructor too. Under
        # jax.jit the inputs are tracers and this branch is skipped — the
        # public filter wrappers run ``resolve_event_inputs`` before tracing
        # to cover that path. The two checks together are defense-in-depth.
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
