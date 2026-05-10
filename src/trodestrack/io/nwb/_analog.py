"""``from_analog_container`` — Trodes-style analog IMU TimeSeries reader.

Pure container-layer code — does not import ``pynwb`` at module
load. TimeSeries attributes are read duck-typed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trodestrack.config.schemas import IMUConfig
from trodestrack.io.imu_parquet import convert_imu_columns_to_si
from trodestrack.io.nwb._shared import eager_array


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

    timestamps = eager_array(analog_ts.timestamps, dtype=float)
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


def _parse_channel_ids(description: str) -> list[str]:
    """Split a channel-id description string back into a list.

    ``trodes_to_nwb.convert_analog.__merge_row_description`` writes
    ``"   ".join(ids) + "   "`` (triple-space separator with a
    trailing ``"   "``). Splitting on the triple-space yields the
    original ids plus a trailing empty string; filter empties.
    """

    return [part for part in description.split("   ") if part]
