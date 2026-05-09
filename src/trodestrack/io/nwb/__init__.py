"""Public NWB position / IMU / DIO loader.

Two-layer design (Spyglass integration seam):

- **Container layer**: ``from_position_container`` /
  ``from_pose_estimation_container`` accept already-loaded pynwb
  containers. They do **not** import ``pynwb`` at module load —
  attributes are read duck-typed so a Spyglass ``make()`` (which has
  already imported ``pynwb`` to call ``fetch_nwb``) can call them
  directly without trodestrack carrying a hard ``pynwb`` dependency.
- **Path layer**: ``load_nwb_session`` lazy-imports ``pynwb``, opens
  the file with ``NWBHDF5IO``, walks containers, and delegates to the
  container-layer entries. Used by ``inputs.format=nwb`` direct-NWB
  users.

All container entries materialize numpy arrays before returning so
the caller may close its IO handle (or let ``fetch_nwb``'s underlying
file close) without breaking downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from trodestrack.config.schemas import (
    IMUConfig,
    NWBConfig,
    NWBDIOToTTLConfig,
    NWBLEDSourceConfig,
)
from trodestrack.io.imu_parquet import convert_imu_columns_to_si
from trodestrack.io.loaders._shared import PositionPixels
from trodestrack.io.ttl_events import EDGE_NAME_TO_INT

if TYPE_CHECKING:
    # Imported for type-checking only; runtime never hits this branch.
    import pynwb


__all__ = [
    "NWBSessionExtras",
    "from_analog_container",
    "from_behavioral_events",
    "from_pose_estimation_container",
    "from_position_container",
    "load_nwb_session",
]


@dataclass(frozen=True)
class NWBSessionExtras:
    """Optional extras the NWB loader pulls alongside position data.

    ``imu`` is a ``(t_imu_unix, U_full)`` pair from the NWB analog
    group. ``dio_events`` is a ``(t_evt, source_id, edge)`` triple
    from the NWB DIO bridge — same shape as
    ``trodestrack.io.ttl_events.load_ttl_events`` returns, so the
    downstream ``per_frame_event_indices`` indexer accepts it
    unchanged.
    """

    imu: tuple[np.ndarray, np.ndarray] | None = None
    dio_events: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Container-layer entry points (public; do NOT import pynwb).
# ---------------------------------------------------------------------


def from_position_container(
    position: pynwb.behavior.Position,
    cfg: NWBLEDSourceConfig,
) -> PositionPixels:
    """Convert a Trodes-style ``Position`` container into ``PositionPixels``.

    Reads the named SpatialSeries pair (``led_0_series_{epoch}`` /
    ``led_1_series_{epoch}`` are the writer defaults from
    ``trodes_to_nwb/convert_position.py:1067-1079``); pulls ``data``
    and ``timestamps``; populates ``coords_meters_per_pixel`` from
    each series' ``conversion`` attribute under the conversion=1.0
    sentinel rule.

    For ``cfg.tracking_geometry`` of ``single_led1`` / ``single_led2``
    the container must hold exactly one SpatialSeries (or the
    matching ``led{1,2}_series_name`` must be set). The unobserved
    LED is filled with NaN so the EKF/UKF observation model still
    sees an LED-pair-shaped input.

    Eager numpy materialization: the returned arrays are independent
    of the source ``Position`` container, so the caller may close the
    underlying ``NWBHDF5IO`` after this returns.
    """

    if cfg.tracking_geometry != "dual_led":
        return _from_position_container_single(position, cfg)

    series1, series2 = _resolve_trodes_series(position, cfg)

    led1, led2, t_cam, scale1, scale2 = _read_trodes_pair(series1, series2)

    # The two series are written with the same ``conversion`` per
    # writer convention; if a future writer diverges, we use the
    # smaller (more conservative) value and surface the mismatch in
    # diagnostics.
    coords_mpp = _resolve_coords_mpp([series1, series2], [scale1, scale2])

    return PositionPixels(
        led1_pixels=led1,
        led2_pixels=led2,
        t_cam=t_cam,
        confidence=None,
        coords_meters_per_pixel=coords_mpp,
        diagnostics={
            "format": "nwb",
            "container": "trodes_position",
            "tracking_geometry": "dual_led",
            "led1_series_name": series1.name,
            "led2_series_name": series2.name,
            "reference_frame": getattr(series1, "reference_frame", None),
            "led1_unit": getattr(series1, "unit", None),
            "led2_unit": getattr(series2, "unit", None),
            "led1_conversion": float(getattr(series1, "conversion", 1.0)),
            "led2_conversion": float(getattr(series2, "conversion", 1.0)),
        },
    )


def _from_position_container_single(
    position: Any, cfg: NWBLEDSourceConfig
) -> PositionPixels:
    """Single-LED Trodes-Position branch of ``from_position_container``."""

    series = _resolve_single_trodes_series(position, cfg)
    data = _eager_array(series.data, dtype=float)
    t_cam = _eager_array(series.timestamps, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"SpatialSeries {series.name!r} data shape {data.shape} is "
            "not (n, 2). Native 3-D coordinates are not supported in "
            "v1 of the NWB loader."
        )
    if t_cam.shape != (data.shape[0],):
        raise ValueError(
            f"SpatialSeries {series.name!r} timestamps shape "
            f"{t_cam.shape} does not match data length {data.shape[0]}."
        )

    conversion = float(getattr(series, "conversion", 1.0))
    coords_mpp = _resolve_coords_mpp_single(series, conversion)
    missing = np.full_like(data, np.nan)
    if cfg.tracking_geometry == "single_led1":
        led1, led2 = data, missing
    else:
        led1, led2 = missing, data

    return PositionPixels(
        led1_pixels=led1,
        led2_pixels=led2,
        t_cam=t_cam,
        confidence=None,
        coords_meters_per_pixel=coords_mpp,
        diagnostics={
            "format": "nwb",
            "container": "trodes_position",
            "tracking_geometry": cfg.tracking_geometry,
            "observed_series_name": series.name,
            "reference_frame": getattr(series, "reference_frame", None),
            "observed_unit": getattr(series, "unit", None),
            "observed_conversion": conversion,
        },
    )


def from_pose_estimation_container(
    pose: Any,
    cfg: NWBLEDSourceConfig,
) -> PositionPixels:
    """Convert an ``ndx_pose.PoseEstimation`` into ``PositionPixels``.

    Reads the named ``PoseEstimationSeries`` for LED1/LED2, applies
    the ``likelihood_threshold`` NaN-mask, and surfaces
    ``source_software`` / ``scorer`` / schema-version diagnostics.

    Reads attributes via fallback so an NWB file written with
    ``ndx-pose`` installed (typed-attribute names
    ``confidence_definition`` / ``source_software_version``) and one
    read without ``ndx-pose`` (on-disk names ``definition`` /
    ``version``) both work.

    For ``cfg.tracking_geometry`` of ``single_led1`` / ``single_led2``
    only the matching ``led{1,2}_bodypart`` is required; the
    unobserved LED is filled with NaN and its confidence column is
    set to a finite neutral value (the all-NaN coordinates already
    drop those frames out of the EKF observation, so the confidence
    value is informational only).

    The Skeleton chain (v0.2.x ``processing["behavior"]["Skeletons"]``
    vs v0.1.x inline ``nodes`` on the ``PoseEstimation``) affects
    diagnostics only — position data comes from
    ``pose.pose_estimation_series[bodypart]`` which works in both
    schemas.
    """

    if cfg.tracking_geometry != "dual_led":
        return _from_pose_estimation_container_single(pose, cfg)

    if cfg.led1_bodypart is None or cfg.led2_bodypart is None:
        raise ValueError(
            "inputs.nwb.led_source.led1_bodypart and led2_bodypart are "
            "required when reading from a PoseEstimation container."
        )

    series_dict = pose.pose_estimation_series
    available = list(series_dict.keys())
    for name in (cfg.led1_bodypart, cfg.led2_bodypart):
        if name not in series_dict:
            raise ValueError(
                f"PoseEstimation series {name!r} not found. Available: {available}."
            )

    series1 = series_dict[cfg.led1_bodypart]
    series2 = series_dict[cfg.led2_bodypart]

    led1, led2, t_cam, scale1, scale2 = _read_trodes_pair(series1, series2)

    conf1 = _eager_array(series1.confidence, dtype=float)
    conf2 = _eager_array(series2.confidence, dtype=float)
    led1[conf1 < cfg.likelihood_threshold] = np.nan
    led2[conf2 < cfg.likelihood_threshold] = np.nan

    confidence = np.column_stack([conf1, conf1, conf2, conf2])
    coords_mpp = _resolve_coords_mpp([series1, series2], [scale1, scale2])

    schema_version = _detect_pose_schema_version(pose)
    confidence_definition = getattr(series1, "confidence_definition", None) or getattr(
        series1, "definition", None
    )
    source_software_version = getattr(pose, "source_software_version", None) or getattr(
        pose, "version", None
    )

    return PositionPixels(
        led1_pixels=led1,
        led2_pixels=led2,
        t_cam=t_cam,
        confidence=confidence,
        coords_meters_per_pixel=coords_mpp,
        diagnostics={
            "format": "nwb",
            "container": "ndx_pose",
            "tracking_geometry": "dual_led",
            "led1_bodypart": cfg.led1_bodypart,
            "led2_bodypart": cfg.led2_bodypart,
            "ndx_pose_schema_version": schema_version,
            "source_software": getattr(pose, "source_software", None),
            "source_software_version": source_software_version,
            "confidence_definition": confidence_definition,
            "scorer": getattr(pose, "scorer", None),
            "led1_kept_fraction": float(np.mean(conf1 >= cfg.likelihood_threshold)),
            "led2_kept_fraction": float(np.mean(conf2 >= cfg.likelihood_threshold)),
        },
    )


def _from_pose_estimation_container_single(
    pose: Any, cfg: NWBLEDSourceConfig
) -> PositionPixels:
    """Single-LED PoseEstimation branch of ``from_pose_estimation_container``."""

    if cfg.tracking_geometry == "single_led1":
        observed_name = cfg.led1_bodypart
        observed_field = "led1_bodypart"
    else:
        observed_name = cfg.led2_bodypart
        observed_field = "led2_bodypart"
    if observed_name is None:
        raise ValueError(
            f"inputs.nwb.led_source.{observed_field} is required when "
            f"tracking_geometry={cfg.tracking_geometry!r}."
        )

    series_dict = pose.pose_estimation_series
    if observed_name not in series_dict:
        raise ValueError(
            f"PoseEstimation series {observed_name!r} not found. "
            f"Available: {list(series_dict.keys())}."
        )
    series = series_dict[observed_name]

    data = _eager_array(series.data, dtype=float)
    t_cam = _eager_array(series.timestamps, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"PoseEstimationSeries {series.name!r} data shape {data.shape} is "
            "not (n, 2). Native 3-D coordinates are not supported in "
            "v1 of the NWB loader."
        )
    if t_cam.shape != (data.shape[0],):
        raise ValueError(
            f"PoseEstimationSeries {series.name!r} timestamps shape "
            f"{t_cam.shape} does not match data length {data.shape[0]}."
        )

    conf = _eager_array(series.confidence, dtype=float)
    data[conf < cfg.likelihood_threshold] = np.nan

    missing = np.full_like(data, np.nan)
    neutral_conf = np.ones_like(conf)
    if cfg.tracking_geometry == "single_led1":
        led1, led2 = data, missing
        confidence = np.column_stack([conf, conf, neutral_conf, neutral_conf])
    else:
        led1, led2 = missing, data
        confidence = np.column_stack([neutral_conf, neutral_conf, conf, conf])

    conversion = float(getattr(series, "conversion", 1.0))
    coords_mpp = _resolve_coords_mpp_single(series, conversion)
    schema_version = _detect_pose_schema_version(pose)
    confidence_definition = getattr(series, "confidence_definition", None) or getattr(
        series, "definition", None
    )
    source_software_version = getattr(pose, "source_software_version", None) or getattr(
        pose, "version", None
    )

    return PositionPixels(
        led1_pixels=led1,
        led2_pixels=led2,
        t_cam=t_cam,
        confidence=confidence,
        coords_meters_per_pixel=coords_mpp,
        diagnostics={
            "format": "nwb",
            "container": "ndx_pose",
            "tracking_geometry": cfg.tracking_geometry,
            "observed_bodypart": observed_name,
            "ndx_pose_schema_version": schema_version,
            "source_software": getattr(pose, "source_software", None),
            "source_software_version": source_software_version,
            "confidence_definition": confidence_definition,
            "scorer": getattr(pose, "scorer", None),
            "observed_kept_fraction": float(np.mean(conf >= cfg.likelihood_threshold)),
        },
    )


def from_analog_container(
    analog_ts: Any,
    imu_cfg: IMUConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Trodes-style analog ``TimeSeries`` to ``(t_imu, U_full)``.

    The TimeSeries is the one ``trodes_to_nwb.convert_analog`` writes
    at ``processing/analog/analog/analog`` — channel ids live in the
    ``description`` attribute as a triple-space-separated string
    (per ``trodes_to_nwb.convert_analog.__merge_row_description``);
    data is ``(n_samples, n_channels)`` int16-stacked ECU + headstage
    columns.

    Channel selection: each ``imu_cfg.axis_map`` value (e.g.
    ``"Headstage_GyroX"``) must resolve to exactly one column. SI
    conversion shares ``convert_imu_columns_to_si`` with the parquet
    path, guaranteeing bit-identical output for matched inputs.

    Eager numpy materialization: the returned arrays are independent
    of the source ``TimeSeries`` so the caller may close the
    underlying ``NWBHDF5IO`` (or let ``fetch_nwb``'s file close)
    after this returns.
    """

    # Resolve channel ids from description before any data read so we
    # can column-slice the underlying h5py dataset to just the IMU
    # axes we need. The Trodes analog group can hold tens of channels
    # (ECU + headstage) at the recording sample rate; eagerly reading
    # the whole array would balloon to GB-scale for long sessions
    # only to discard ~94% of it.
    description = str(getattr(analog_ts, "description", "") or "")
    channel_ids = _parse_channel_ids(description)
    name_to_col = {name: i for i, name in enumerate(channel_ids)}
    missing: list[tuple[str, str]] = []
    axis_to_col: dict[str, int] = {}
    for axis, mapped_name in imu_cfg.axis_map.items():
        if mapped_name not in name_to_col:
            missing.append((axis, mapped_name))
            continue
        axis_to_col[axis] = name_to_col[mapped_name]
    if missing:
        raise ValueError(
            f"NWB analog channel IDs missing axis_map entries: "
            f"{missing}. Available channel IDs: {channel_ids}. "
            "Update inputs.imu (axis_map) to match the channel names "
            "the writer recorded."
        )

    timestamps = _eager_array(analog_ts.timestamps, dtype=float)
    # Slice only the columns we need. h5py supports sorted-int fancy
    # indexing on the second axis; sort the unique columns, slice,
    # then unscramble per axis.
    unique_cols = sorted(set(axis_to_col.values()))
    col_to_local = {col: i for i, col in enumerate(unique_cols)}
    raw_dataset = analog_ts.data
    if hasattr(raw_dataset, "shape") and len(raw_dataset.shape) != 2:
        raise ValueError(
            f"Analog TimeSeries data must be 2-D (n_samples, n_channels); "
            f"got shape {raw_dataset.shape}."
        )
    if hasattr(raw_dataset, "shape") and raw_dataset.shape[1] != len(channel_ids):
        raise ValueError(
            f"Analog TimeSeries channel-id count from description "
            f"({len(channel_ids)}: {channel_ids}) does not match data "
            f"shape {raw_dataset.shape}. The description must list "
            "exactly one id per data column (per "
            "trodes_to_nwb.convert_analog.__merge_row_description)."
        )
    sliced = np.asarray(raw_dataset[:, unique_cols])
    if timestamps.shape != (sliced.shape[0],):
        raise ValueError(
            f"Analog TimeSeries timestamps shape {timestamps.shape} does "
            f"not match data length {sliced.shape[0]}."
        )

    raw_columns = {
        axis: sliced[:, col_to_local[col]] for axis, col in axis_to_col.items()
    }
    U_full = convert_imu_columns_to_si(raw_columns, imu_cfg)
    return timestamps, U_full


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
        data = _eager_array(ts.data)
        timestamps = _eager_array(ts.timestamps, dtype=float)
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


def _parse_channel_ids(description: str) -> list[str]:
    """Split a channel-id description string back into a list.

    ``trodes_to_nwb.convert_analog.__merge_row_description`` writes
    ``"   ".join(ids) + "   "`` (triple-space separator with a
    trailing ``"   "``). Splitting on the triple-space yields the
    original ids plus a trailing empty string; filter empties.
    """

    return [part for part in description.split("   ") if part]


# ---------------------------------------------------------------------
# Path-based wrapper (lazy-imports pynwb).
# ---------------------------------------------------------------------


def load_nwb_session(
    cfg: NWBConfig,
    *,
    imu_cfg: IMUConfig | None = None,
) -> tuple[PositionPixels, NWBSessionExtras]:
    """Open an NWB file at ``cfg.nwb_file``, pick LED container by
    neurodata type, and delegate to a container-layer entry.

    When ``imu_cfg`` is provided and the file has the standard
    ``processing/analog/analog/analog`` TimeSeries (per
    ``trodes_to_nwb.convert_analog._NWB_ANALOG_DATA_PATH``), the
    extras include the SI-converted ``(t_imu, U_full)`` pair via
    ``from_analog_container``. Without ``imu_cfg`` we don't know
    which channels to select, so the analog container is left
    unread.

    Lazy-imports ``pynwb`` so ``import trodestrack.io.nwb`` does not
    force the ``[nwb]`` extra to be installed.
    """

    try:
        import pynwb  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "inputs.format='nwb' requires the [nwb] extra. "
            "Install with: uv pip install 'trodestrack[nwb]'."
        ) from e

    with pynwb.NWBHDF5IO(str(cfg.nwb_file), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        led_container = _detect_led_container(nwbfile, cfg.led_source.container)
        ndt = _neurodata_type(led_container)
        if ndt == "Position":
            pixels = from_position_container(led_container, cfg.led_source)
        elif ndt == "PoseEstimation":
            pixels = from_pose_estimation_container(led_container, cfg.led_source)
        else:
            raise ValueError(
                f"Detected LED container has unsupported neurodata_type "
                f"{ndt!r}. Expected 'Position' or 'PoseEstimation'."
            )
        imu_pair: tuple[np.ndarray, np.ndarray] | None = None
        diagnostics: dict[str, Any] = {}
        if imu_cfg is not None:
            analog_ts = _find_analog_timeseries(nwbfile)
            if analog_ts is not None:
                t_imu, U_full = from_analog_container(analog_ts, imu_cfg)
                imu_pair = (t_imu, U_full)
                diagnostics["imu_source"] = "nwb_analog"
                diagnostics["imu_samples"] = int(t_imu.size)
                diagnostics["imu_channel_count"] = int(U_full.shape[1])
            else:
                diagnostics["imu_source"] = None
        dio_events: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if cfg.dio_to_ttl is not None:
            events_container = _find_behavioral_events(nwbfile)
            if events_container is None:
                raise ValueError(
                    f"inputs.nwb.dio_to_ttl is configured but "
                    f"{cfg.nwb_file} has no "
                    "processing['behavior']['behavioral_events'] container."
                )
            dio_events = from_behavioral_events(events_container, cfg.dio_to_ttl)
            diagnostics["dio_event_count"] = int(dio_events[0].size)
            diagnostics["dio_source"] = "nwb_behavioral_events"
        extras = NWBSessionExtras(
            imu=imu_pair, dio_events=dio_events, diagnostics=diagnostics
        )
        return pixels, extras
    # NWBHDF5IO closed here; pixels/extras hold numpy arrays only.


def _find_behavioral_events(nwbfile: Any) -> Any | None:
    """Locate ``processing/behavior/behavioral_events`` if present.

    The Trodes writer creates a ``BehavioralEvents`` container named
    ``"behavioral_events"`` under ``processing["behavior"]`` (per
    ``trodes_to_nwb/convert_dios.py``). Returns the container or
    ``None`` when absent.
    """

    processing = getattr(nwbfile, "processing", None) or {}
    if "behavior" not in processing:
        return None
    behavior = processing["behavior"]
    interfaces = getattr(behavior, "data_interfaces", {})
    if "behavioral_events" not in interfaces:
        return None
    return interfaces["behavioral_events"]


def _find_analog_timeseries(nwbfile: Any) -> Any | None:
    """Locate ``processing/analog/analog/analog`` if present.

    The Trodes writer creates a ``processing["analog"]`` module with a
    ``BehavioralEvents`` container named ``"analog"`` whose
    ``time_series`` dict has a single entry also named ``"analog"``
    (per ``trodes_to_nwb.convert_analog.add_analog_data``). The full
    HDF5 path is ``processing/analog/analog/analog/data``.

    Returns the TimeSeries or ``None`` when the file has no analog
    group.
    """

    processing = getattr(nwbfile, "processing", None) or {}
    if "analog" not in processing:
        return None
    module = processing["analog"]
    interfaces = getattr(module, "data_interfaces", {})
    if "analog" not in interfaces:
        return None
    container = interfaces["analog"]
    time_series = getattr(container, "time_series", None)
    if time_series is None or "analog" not in time_series:
        return None
    return time_series["analog"]


# ---------------------------------------------------------------------
# Helpers (private; pynwb-free for the container-extraction paths).
# ---------------------------------------------------------------------


def _eager_array(dataset: Any, *, dtype: Any = None) -> np.ndarray:
    """Slice a pynwb / h5py dataset into a numpy array.

    The ``[...]`` indexing materializes the underlying h5py dataset
    into RAM, breaking the dependency on the source IO handle. The
    asarray pass enforces a concrete dtype.
    """

    arr = np.asarray(dataset[...])
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _read_trodes_pair(
    series1: Any, series2: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Pull ``(data, timestamps, conversion)`` for two SpatialSeries.

    Returns ``(led1_xy, led2_xy, t_cam_seconds, conv1, conv2)`` where
    LED arrays are ``(n, 2)``.

    Both SpatialSeries' timestamps must agree (1:1 in length and
    bitwise equal); silently using LED1's clock for LED2 samples
    would let a mis-paired write surface as a fused-but-misaligned
    trajectory.
    """

    led1 = _eager_array(series1.data, dtype=float)
    led2 = _eager_array(series2.data, dtype=float)
    t1 = _eager_array(series1.timestamps, dtype=float)
    t2 = _eager_array(series2.timestamps, dtype=float)
    if led1.ndim != 2 or led1.shape[1] != 2:
        raise ValueError(
            f"SpatialSeries {series1.name!r} data shape {led1.shape} is "
            "not (n, 2). Native 3-D coordinates are not supported in "
            "v1 of the NWB loader."
        )
    if led2.shape != led1.shape:
        raise ValueError(
            f"LED1 / LED2 SpatialSeries shapes mismatch "
            f"({led1.shape} vs {led2.shape}); the writer should keep "
            "them aligned 1:1."
        )
    if t1.shape != (led1.shape[0],):
        raise ValueError(
            f"SpatialSeries {series1.name!r} timestamps shape "
            f"{t1.shape} does not match data length {led1.shape[0]}."
        )
    if t2.shape != t1.shape:
        raise ValueError(
            f"LED1 / LED2 timestamps length mismatch "
            f"({series1.name!r}: {t1.shape}; "
            f"{series2.name!r}: {t2.shape}). The writer must align "
            "the two series 1:1 — re-export the session or supply the "
            "two SpatialSeries that actually share a clock."
        )
    if not np.array_equal(t1, t2):
        max_diff = float(np.max(np.abs(t1 - t2)))
        raise ValueError(
            f"LED1 / LED2 timestamps differ between {series1.name!r} "
            f"and {series2.name!r} (max abs difference: {max_diff:.6g} "
            "s). The two series must share a clock; the loader will "
            "not silently pair LED2 samples with LED1's time base."
        )
    conv1 = float(getattr(series1, "conversion", 1.0))
    conv2 = float(getattr(series2, "conversion", 1.0))
    return led1, led2, t1, conv1, conv2


def _resolve_coords_mpp(serieses: list[Any], conversions: list[float]) -> float | None:
    """Apply the conversion=1.0 sentinel rule to a series pair.

    - ``unit == "pixels"`` and ``conversion == 1.0`` → ``None``
      (no calibration baked in; YAML / homography path takes over).
    - ``unit == "meters"`` → use ``conversion`` as the scalar.
    - Otherwise (e.g., ``unit == "pixels"`` with ``conversion != 1.0``):
      use the conversion. Treat as already-calibrated to meters.

    Both series must resolve to the same scalar. Disagreement (one
    LED is the sentinel "no calibration", the other is calibrated; or
    both are calibrated with different scales) is rejected — silently
    picking LED1 would let a mis-paired write produce a fused-but-
    miscalibrated trajectory at the same fault surface as the
    timestamp-mismatch guard above.
    """

    def resolve_one(series: Any, conversion: float) -> float | None:
        unit = getattr(series, "unit", "pixels")
        if unit == "pixels" and conversion == 1.0:
            return None
        return conversion

    resolved1 = resolve_one(serieses[0], conversions[0])
    resolved2 = resolve_one(serieses[1], conversions[1])
    if resolved1 != resolved2:
        raise ValueError(
            f"LED1 / LED2 calibration mismatch: {serieses[0].name!r} "
            f"resolves to coords_meters_per_pixel={resolved1!r} "
            f"(unit={getattr(serieses[0], 'unit', 'pixels')!r}, "
            f"conversion={conversions[0]}); "
            f"{serieses[1].name!r} resolves to {resolved2!r} "
            f"(unit={getattr(serieses[1], 'unit', 'pixels')!r}, "
            f"conversion={conversions[1]}). The two series must share "
            "a calibration — silently picking one would produce a "
            "fused-but-miscalibrated trajectory."
        )
    return resolved1


def _resolve_coords_mpp_single(series: Any, conversion: float) -> float | None:
    """Single-series counterpart to ``_resolve_coords_mpp``.

    Same conversion=1.0 / ``unit`` sentinel rule applied to one
    SpatialSeries; no pair-mismatch check because there is no pair.
    """

    unit = getattr(series, "unit", "pixels")
    if unit == "pixels" and conversion == 1.0:
        return None
    return conversion


def _resolve_single_trodes_series(position: Any, cfg: NWBLEDSourceConfig) -> Any:
    """Pick the single observed SpatialSeries for ``single_led1`` /
    ``single_led2`` from a ``Position`` container.

    Resolution order:

    1. If ``cfg.led{1,2}_series_name`` (matching ``tracking_geometry``)
       is set, use it by name.
    2. Otherwise auto-detect: the container must hold exactly one
       SpatialSeries — ambiguity (two or more) is rejected because
       the loader cannot know which point is the physical LED.
    """

    series_dict = position.spatial_series
    available = list(series_dict.keys())

    if cfg.tracking_geometry == "single_led1":
        named = cfg.led1_series_name
        named_field = "led1_series_name"
    else:
        named = cfg.led2_series_name
        named_field = "led2_series_name"

    if named is not None:
        if named not in series_dict:
            raise ValueError(
                f"SpatialSeries {named!r} not found in Position "
                f"container. Available: {available}."
            )
        return series_dict[named]

    if len(available) == 1:
        return series_dict[available[0]]

    raise ValueError(
        f"tracking_geometry={cfg.tracking_geometry!r} requires the "
        "Position container to hold exactly one SpatialSeries (or "
        f"set inputs.nwb.led_source.{named_field}). Found {available!r}."
    )


def _resolve_trodes_series(position: Any, cfg: NWBLEDSourceConfig) -> tuple[Any, Any]:
    """Pick LED1/LED2 SpatialSeries from a ``Position`` container.

    Resolution order:

    1. If ``cfg.led1_series_name`` / ``cfg.led2_series_name`` are
       both set, use them by name.
    2. Otherwise, walk ``position.spatial_series`` looking for the
       writer defaults (``led_0_series_*`` / ``led_1_series_*``
       prefix from ``convert_position.py:1067-1079``).
    3. If exactly two SpatialSeries exist (any names), use them in
       sorted order.
    4. Otherwise, raise with the available names listed.
    """

    series_dict = position.spatial_series
    available = list(series_dict.keys())

    if cfg.led1_series_name is not None and cfg.led2_series_name is not None:
        for name in (cfg.led1_series_name, cfg.led2_series_name):
            if name not in series_dict:
                raise ValueError(
                    f"SpatialSeries {name!r} not found in Position "
                    f"container. Available: {available}."
                )
        return series_dict[cfg.led1_series_name], series_dict[cfg.led2_series_name]

    led0 = sorted(name for name in available if name.startswith("led_0_series"))
    led1 = sorted(name for name in available if name.startswith("led_1_series"))
    if led0 and led1:
        return series_dict[led0[0]], series_dict[led1[0]]

    if len(available) == 2:
        names_sorted = sorted(available)
        return series_dict[names_sorted[0]], series_dict[names_sorted[1]]

    raise ValueError(
        f"Could not auto-detect LED1/LED2 SpatialSeries in the "
        f"Position container. Found {available!r}. Set "
        "inputs.nwb.led_source.led1_series_name and led2_series_name "
        "explicitly."
    )


def _neurodata_type(container: Any) -> str:
    """Return the NWB neurodata-type string for a container.

    Uses the ``neurodata_type`` attribute pynwb sets on every typed
    container; falls back to the class name when reading
    dynamic-spec containers without their typed Python class.
    """

    return str(getattr(container, "neurodata_type", type(container).__name__))


def _detect_led_container(nwbfile: Any, container_choice: str) -> Any:
    """Walk ``processing["behavior"]`` and return the LED container.

    Auto-detection matches on ``neurodata_type`` (``Position`` vs
    ``PoseEstimation``), not on container name — writer-default names
    aren't guarantees.
    """

    behavior = (
        nwbfile.processing.get("behavior") if hasattr(nwbfile, "processing") else None
    )
    if behavior is None:
        raise ValueError(
            "NWB file has no processing['behavior'] module; cannot "
            "locate position data."
        )

    candidates: list[tuple[str, str, Any]] = []
    for name, di in behavior.data_interfaces.items():
        ndt = _neurodata_type(di)
        if ndt == "Position":
            candidates.append(("trodes_position", name, di))
        elif ndt == "PoseEstimation":
            candidates.append(("ndx_pose", name, di))

    if container_choice == "auto":
        if not candidates:
            raise ValueError(
                "No Position or PoseEstimation container found under "
                "processing['behavior']."
            )
        kinds = {kind for kind, _, _ in candidates}
        if len(kinds) > 1:
            kind_to_names = {
                kind: [name for k, name, _ in candidates if k == kind] for kind in kinds
            }
            raise ValueError(
                "Multiple LED container types found "
                f"({kind_to_names}); set inputs.nwb.led_source.container "
                "to 'trodes_position' or 'ndx_pose' to disambiguate."
            )
        if len(candidates) > 1:
            names = [name for _, name, _ in candidates]
            raise ValueError(
                "Multiple containers of the same type found "
                f"({names}); set inputs.nwb.led_source.container and "
                "the matching series-name / bodypart fields to pick "
                "the right one."
            )
        return candidates[0][2]

    target_kind = container_choice
    matching = [di for kind, _, di in candidates if kind == target_kind]
    if not matching:
        kinds_found = [kind for kind, _, _ in candidates]
        raise ValueError(
            f"Requested container={container_choice!r} but only "
            f"{kinds_found} found under processing['behavior']."
        )
    if len(matching) > 1:
        names = [name for kind, name, _ in candidates if kind == target_kind]
        raise ValueError(
            f"Multiple {container_choice!r} containers found ({names}); "
            "the loader currently picks the first one — name them in "
            "led_source to disambiguate."
        )
    return matching[0]


def _detect_pose_schema_version(pose: Any) -> str:
    """Best-effort ndx-pose schema-version detection.

    v0.2.x stores nodes/edges via a ``Skeleton`` linked from the
    parent processing module's ``Skeletons`` container. v0.1.x stores
    ``nodes`` / ``edges`` inline on the ``PoseEstimation`` itself.

    This is diagnostic-only — position data is read from
    ``pose.pose_estimation_series[bodypart]`` which works in both
    schemas regardless of detection result.
    """

    # A real Skeleton object exposes ``nodes`` with content; a broken
    # link from a v0.1.x file may surface a stub ``skeleton`` whose
    # attributes are missing or empty, so check for actual node data.
    skeleton = getattr(pose, "skeleton", None)
    if skeleton is not None:
        skeleton_nodes = getattr(skeleton, "nodes", None)
        if skeleton_nodes is not None and len(np.asarray(skeleton_nodes)) > 0:
            return "v0.2.x"
    inline_nodes = getattr(pose, "nodes", None)
    if inline_nodes is not None and len(np.asarray(inline_nodes)) > 0:
        return "v0.1.x"
    return "unknown"
