"""Ingest helpers for TTL event sensors (beam break, zone trigger, RFID).

The events parquet schema is uniform across source types::

    columns: [time(s, float), source_id(int), edge("rise" | "fall")]

Source-type metadata lives in the YAML config; the loader and per-frame
indexer here only care about ``source_id`` and the active edge configured for
each source. ``per_frame_event_indices`` returns a dense
``(n_cam, max_events_per_frame)`` array of compact source indices (or ``-1``
for padded slots) that the EKF event-update path consumes inside a JIT'd
``lax.scan``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

EDGE_NAME_TO_INT: dict[str, int] = {"fall": 0, "rise": 1}
PAD_SENTINEL: int = -1

_INT64_MIN: int = int(np.iinfo(np.int64).min)
_INT64_MAX: int = int(np.iinfo(np.int64).max)


def _coerce_source_ids(raw: np.ndarray, events_file: Path) -> np.ndarray:
    """Validate and coerce a ``source_id`` column to ``int64``.

    Rejects non-integer floats, non-finite values, and integers outside
    the signed int64 range. Routing an out-of-range source ID would
    silently map the event to a wrong-but-valid configured source.
    """
    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger) and raw.size:
            overflow = raw > np.uint64(_INT64_MAX)
            if overflow.any():
                bad = sorted({int(x) for x in raw[overflow][:5]})
                raise ValueError(
                    f"{events_file} contains source_id value(s) above the "
                    f"signed int64 range (max {_INT64_MAX}); the "
                    f"trodestrack event channel cannot represent them. "
                    f"Got entries like {bad}."
                )
        return raw.astype(np.int64, copy=False)

    if not np.issubdtype(raw.dtype, np.floating):
        # Strings, booleans, object, etc. would silently coerce to int via
        # ``np.asarray(..., dtype=float)`` ("1" → 1, True → 1), violating
        # the documented integer schema. Reject loudly.
        raise ValueError(
            f"{events_file} source_id column must be an integer or float "
            f"dtype; got dtype={raw.dtype!r}. The documented schema is "
            "``source_id (int)``."
        )

    raw_float = np.asarray(raw, dtype=float)
    bad = ~np.isfinite(raw_float) | (raw_float != np.floor(raw_float))
    if bad.any():
        raise ValueError(
            f"{events_file} contains non-integer source_id value(s); "
            "expected integers but got entries like "
            f"{sorted({float(x) for x in raw_float[bad][:5]})}."
        )
    # Float64 cannot represent ``_INT64_MAX = 2^63 - 1`` exactly — it
    # rounds up to ``2^63``. Reject any float at or above ``2^63``
    # (saturates on cast) or strictly below ``-2^63`` (== _INT64_MIN, which
    # *is* exactly representable). Using the float bound ``2.0**63`` keeps
    # the check consistent with what the int64 cast can actually produce.
    upper_bound = 2.0**63
    out_of_range = (raw_float >= upper_bound) | (raw_float < -upper_bound)
    if out_of_range.any():
        raise ValueError(
            f"{events_file} contains source_id value(s) outside the signed "
            f"int64 range [{_INT64_MIN}, {_INT64_MAX}]; the trodestrack "
            "event channel cannot represent them. Got entries like "
            f"{sorted({float(x) for x in raw_float[out_of_range][:5]})}."
        )
    return raw_float.astype(np.int64)


def load_ttl_events(
    events_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a TTL events parquet into time / source_id / edge arrays.

    Parameters
    ----------
    events_file : Path
        Parquet file with columns ``time`` (float seconds), ``source_id``
        (int), and ``edge`` (string ``"rise"`` or ``"fall"``).

    Returns
    -------
    t_evt : np.ndarray, shape (n_evt,)
        Event timestamps in seconds.
    source_id : np.ndarray, shape (n_evt,)
        User-facing source IDs (arbitrary ints; not yet compacted).
    edge : np.ndarray, shape (n_evt,)
        Encoded edge: ``0`` for ``"fall"``, ``1`` for ``"rise"``.
    """

    df = pd.read_parquet(events_file)
    missing = [col for col in ("time", "source_id", "edge") if col not in df.columns]
    if missing:
        raise ValueError(
            f"{events_file} is missing required column(s): {', '.join(missing)}."
        )

    raw_time = df["time"].to_numpy()
    if not np.issubdtype(raw_time.dtype, np.number) or np.issubdtype(
        raw_time.dtype, np.bool_
    ):
        raise ValueError(
            f"{events_file} time column must be a numeric (integer or "
            f"float) dtype; got dtype={raw_time.dtype!r}. The documented "
            "schema is ``time (s, float)``."
        )
    t_evt = raw_time.astype(float, copy=False)
    edge_str = df["edge"].to_numpy()

    source_id = _coerce_source_ids(df["source_id"].to_numpy(), events_file)

    if t_evt.size == 0:
        return t_evt, source_id, np.zeros(0, dtype=int)

    # Reject non-finite timestamps explicitly; otherwise downstream
    # ``np.searchsorted`` quietly buckets NaN past the end and ±inf to
    # frame 0 / n_cam, dropping the event with no diagnostic.
    non_finite = ~np.isfinite(t_evt)
    if non_finite.any():
        raise ValueError(
            f"{events_file} contains {int(non_finite.sum())} non-finite "
            "timestamp(s); expected finite float seconds."
        )

    edge = np.empty(edge_str.shape, dtype=int)
    for name, value in EDGE_NAME_TO_INT.items():
        edge[edge_str == name] = value
    unknown_mask = ~np.isin(edge_str, list(EDGE_NAME_TO_INT.keys()))
    if unknown_mask.any():
        bad = sorted({str(x) for x in edge_str[unknown_mask]})
        raise ValueError(
            f"{events_file} contains unknown edge value(s) {bad}; "
            "expected 'rise' or 'fall'."
        )
    return t_evt, source_id, edge


def per_frame_event_indices(
    t_evt: np.ndarray,
    source_id: np.ndarray,
    edge: np.ndarray,
    t_cam: np.ndarray,
    *,
    source_active_edges: Mapping[int, int],
    source_id_to_index: Mapping[int, int],
    max_events_per_frame: int,
) -> np.ndarray:
    """Bucket TTL events into camera frames as compact source indices.

    Each event is assigned to the camera frame ``k`` whose interval
    ``(t_cam[k-1], t_cam[k]]`` contains the event time. Events before
    ``t_cam[0]`` are dropped (no enclosing frame); events after
    ``t_cam[-1]`` are likewise dropped.

    Parameters
    ----------
    t_evt, source_id, edge : np.ndarray, shape (n_evt,)
        Event arrays from :func:`load_ttl_events`.
    t_cam : np.ndarray, shape (n_cam,)
        Camera-frame timestamps.
    source_active_edges : Mapping[int, int]
        Maps user-facing ``source_id`` to its active edge code (0=fall, 1=rise).
        Events whose edge does not match the active edge for their source are
        silently dropped.
    source_id_to_index : Mapping[int, int]
        Maps user-facing ``source_id`` to a dense compact index in
        ``[0, n_sources)``.
    max_events_per_frame : int
        Static pad limit; raises ``ValueError`` if any frame exceeds it.

    Returns
    -------
    np.ndarray, shape (n_cam, max_events_per_frame)
        Compact source indices, padded with ``-1``.
    """

    n_cam = int(t_cam.shape[0])
    out = np.full((n_cam, max_events_per_frame), PAD_SENTINEL, dtype=np.int32)
    if t_evt.size == 0:
        return out

    unknown = sorted({int(s) for s in source_id if int(s) not in source_id_to_index})
    if unknown:
        raise ValueError(
            f"events file references unknown source id(s) {unknown}; "
            "configure them under ttl_events.{beams,zone_triggers,rfid_readers}."
        )

    active_edges = np.asarray(
        [source_active_edges[int(s)] for s in source_id], dtype=int
    )
    keep = edge == active_edges
    t_evt = t_evt[keep]
    source_id = source_id[keep]
    if t_evt.size == 0:
        return out

    # searchsorted side='left' yields smallest k with t_cam[k] >= t_evt,
    # placing each event in the (t_cam[k-1], t_cam[k]] bucket.
    frame_idx = np.searchsorted(t_cam, t_evt, side="left")
    valid = (frame_idx >= 1) & (frame_idx < n_cam)
    frame_idx = frame_idx[valid]
    source_id = source_id[valid]
    if frame_idx.size == 0:
        return out

    # Compute each event's slot inside its camera frame via stable sort,
    # then a per-frame "rank from group start" using searchsorted.
    order = np.argsort(frame_idx, kind="stable")
    sorted_frames = frame_idx[order]
    sorted_sids = source_id[order]
    group_start = np.searchsorted(sorted_frames, sorted_frames, side="left")
    slot = np.arange(sorted_frames.size) - group_start
    if slot.max() >= max_events_per_frame:
        overflow_frame = int(sorted_frames[slot.argmax()])
        raise ValueError(
            f"camera frame {overflow_frame} has more than "
            f"max_events_per_frame={max_events_per_frame} TTL events; "
            "raise the pad limit or debounce noisy sources."
        )
    compact = np.fromiter(
        (source_id_to_index[int(s)] for s in sorted_sids),
        dtype=np.int32,
        count=sorted_sids.size,
    )
    out[sorted_frames, slot] = compact
    return out
