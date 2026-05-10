"""``from_behavioral_events`` — NWB DIO TimeSeries → TTL bridge.

Pure container-layer code — does not import ``pynwb`` at module
load. BehavioralEvents / TimeSeries attributes are read duck-typed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trodestrack.config.schemas import NWBDIOToTTLConfig
from trodestrack.io.nwb._shared import eager_array
from trodestrack.io.ttl_events import EDGE_NAME_TO_INT


def from_behavioral_events(
    events: Any,
    dio_cfg: NWBDIOToTTLConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(t_evt, source_id, edge)`` arrays from NWB DIO TimeSeries.

    Accepts trodestrack-canonical input shapes:

    - A ``BehavioralEvents`` container (path-loader source — pulled
      from ``processing["behavior"]["behavioral_events"]``).
    - A ``dict[str, TimeSeries]`` keyed by the event-name. Spyglass's
      ``(DIOEvents & key).fetch_nwb()`` returns a ``list[dict]`` with
      ``dio_event_name`` / ``dio`` columns; Spyglass's ``make()``
      assembles the dict shape (``{row["dio_event_name"]: row["dio"]
      for row in ...}``) before calling here. trodestrack's API is
      Spyglass-agnostic.

    DIO encoding: ``data`` is ``int8`` 0/1 where each value is
    already a transition. The integer values match the parquet
    path's ``EDGE_NAME_TO_INT`` mapping (``1`` = rise, ``0`` = fall),
    so the array can flow into ``per_frame_event_indices`` unchanged
    — see the assert below for the explicit pinning.

    The ``trodes_to_nwb`` writer strips the initial level on the
    write side (``trodes_to_nwb/spike_gadgets_raw_io.py:1348``
    returns ``dio_change_times[1:], change_dir_trim[1:]`` and
    ``trodes_to_nwb/convert_dios.py:97`` writes those directly into
    the NWB ``TimeSeries``), so the loader must NOT drop ``data[0]``
    again — every sample on disk is a real edge.

    Eager numpy materialization: the returned arrays are independent
    of the source IO so the caller may close it (or let
    ``fetch_nwb``'s underlying file close) without breaking
    downstream code. The result is sorted by ``t_evt`` so it satisfies
    the same monotonic-time contract ``load_ttl_events`` produces.
    """

    # The on-disk int8 values must match the trodestrack-canonical
    # edge encoding the parquet path produces; pin it so a future
    # ``EDGE_NAME_TO_INT`` flip can't silently mis-interpret NWB DIO.
    assert EDGE_NAME_TO_INT == {"fall": 0, "rise": 1}

    # Normalize input: BehavioralEvents has ``time_series``;
    # raw dict is direct.
    ts_dict = getattr(events, "time_series", None)
    if ts_dict is None:
        if not isinstance(events, dict):
            raise TypeError(
                "from_behavioral_events expected a BehavioralEvents "
                "container or dict[str, TimeSeries]; got "
                f"{type(events).__name__}."
            )
        ts_dict = events

    available = list(ts_dict.keys())
    missing = [name for name in dio_cfg.name_to_source_id if name not in ts_dict]
    if missing:
        raise ValueError(
            f"NWB DIO TimeSeries missing for name_to_source_id keys "
            f"{missing}. Available TimeSeries names: {available}."
        )

    t_parts: list[np.ndarray] = []
    sid_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    for name, source_id in dio_cfg.name_to_source_id.items():
        ts = ts_dict[name]
        data = eager_array(ts.data)
        timestamps = eager_array(ts.timestamps, dtype=float)
        if data.shape != timestamps.shape:
            raise ValueError(
                f"DIO TimeSeries {name!r} data shape {data.shape} does "
                f"not match timestamps shape {timestamps.shape}."
            )
        if data.size == 0:
            continue
        # Validate dtype *before* casting — otherwise float values
        # like ``[0.2, 1.8]`` silently truncate to ``[0, 1]`` and
        # pass the 0/1 membership check.
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError(
                f"DIO TimeSeries {name!r} has non-integer dtype "
                f"{data.dtype!r}; the writer encoding "
                "(trodes_to_nwb/convert_dios.py:97) is int8 0/1 "
                "transitions only. Float / object / bool inputs are "
                "rejected to avoid silent coercion."
            )
        if not np.isin(data, (0, 1)).all():
            unique = sorted({int(x) for x in data.tolist()})
            raise ValueError(
                f"DIO TimeSeries {name!r} contains non-{{0, 1}} values "
                f"{unique}; the writer encoding documented at "
                "trodes_to_nwb/convert_dios.py:97 is int8 0/1 "
                "transitions only."
            )
        # Reject non-finite timestamps (matching the parquet TTL
        # path at ``ttl_events.py:132``); NaN/inf would otherwise
        # propagate into ``t_evt`` and confuse the downstream
        # ``np.searchsorted`` bucketing.
        if not np.all(np.isfinite(timestamps)):
            n_bad = int((~np.isfinite(timestamps)).sum())
            raise ValueError(
                f"DIO TimeSeries {name!r} contains {n_bad} non-finite "
                "timestamp(s); expected finite float seconds (parity "
                "with the parquet TTL events validation at "
                "ttl_events.py:132)."
            )
        edges = data.astype(int, copy=False)
        t_parts.append(timestamps)
        edge_parts.append(edges)
        sid_parts.append(np.full(edges.size, source_id, dtype=np.int64))

    if not t_parts:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=int),
        )

    t_evt = np.concatenate(t_parts)
    source_id = np.concatenate(sid_parts)
    edge = np.concatenate(edge_parts)
    # Sort by time so ``per_frame_event_indices`` sees the same
    # monotonic ordering ``load_ttl_events`` produces from a parquet.
    order = np.argsort(t_evt, kind="stable")
    return t_evt[order], source_id[order], edge[order]
