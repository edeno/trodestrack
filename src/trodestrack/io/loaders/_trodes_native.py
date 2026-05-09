"""Native Trodes ``.videoPositionTracking`` + PTP-timestamps loader.

Reads SpikeGadgets / Trodes binaries directly without going through
the parquet conversion step. v1 of this loader supports **PTP-synced
timestamps only** (``*.videoTimeStamps.cameraHWSync``); non-PTP
variants (``cameraHWFrameCount``, plain ``videoTimeStamps``) require
clock-stitching logic that is out of scope.

The Trodes binary parser (header + record-array parse, PTP timestamp
processing, sample-count clock dedup) is adapted from the MIT-licensed
``trodes_to_nwb`` (Loren Frank Lab, 2023) at commit
``eae7abae1a005f3a5c1fbac7bc93282063d71d63``,
``src/trodes_to_nwb/convert_position.py``. The non-PTP branch and
unused helpers are intentionally not vendored.
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label

from trodestrack.io.loaders._shared import PositionPixels

logger = logging.getLogger(__name__)

NANOSECONDS_PER_SECOND = 1e9
DEFAULT_MIN_PTP_PAUSE_S = 0.4
DEFAULT_MAX_PTP_PAUSE_S = 2.0


def load_trodes_native_position(
    position_tracking_file: Path,
    camera_timestamps_file: Path,
) -> PositionPixels:
    """Read PTP-synced Trodes position binaries into a ``PositionPixels``.

    Parameters
    ----------
    position_tracking_file
        Path to the ``*.videoPositionTracking`` binary (online-tracker
        ``(xloc, yloc, xloc2, yloc2)`` quad).
    camera_timestamps_file
        Path to the ``*.videoTimeStamps.cameraHWSync`` PTP-timestamp
        binary. The filename suffix is checked first (heuristic); the
        authoritative gate is the ``HWTimestamp`` column.

    Raises
    ------
    ValueError
        If the timestamps filename or the timestamps file's
        ``Fields`` declaration does not match the PTP variant.
    """

    _check_ptp_filename(camera_timestamps_file)

    timestamps_file = _read_trodes_datafile(camera_timestamps_file)
    timestamps_dtype_names = timestamps_file["data"].dtype.names or ()
    if "HWTimestamp" not in timestamps_dtype_names:
        raise ValueError(
            f"camera_timestamps_file={camera_timestamps_file} has the "
            "*.videoTimeStamps.cameraHWSync suffix but is missing the "
            f"required 'HWTimestamp' column (Fields="
            f"{timestamps_file.get('fields')!r}). The filename is a "
            "heuristic; the column is the truth — only PTP-synced files "
            "carry HWTimestamp. v1 of trodes_native supports PTP only."
        )

    initial_frame_count = len(timestamps_file["data"])
    video_timestamps = _get_position_timestamps_ptp_joined(
        camera_timestamps_file, position_tracking_file
    )

    if not {"xloc", "yloc", "xloc2", "yloc2"}.issubset(video_timestamps.columns):
        raise ValueError(
            f"position_tracking_file={position_tracking_file} is missing "
            "one or more of the required columns (xloc, yloc, xloc2, "
            "yloc2)."
        )

    led1 = video_timestamps[["xloc", "yloc"]].to_numpy(dtype=float)
    led2 = video_timestamps[["xloc2", "yloc2"]].to_numpy(dtype=float)

    # Trodes online tracker emits zero rows when LEDs are lost. Mark
    # those as NaN so downstream NaN-aware code (mask_cam, EKF
    # observation gating) handles them correctly.
    led1[(led1 == 0).all(axis=1)] = np.nan
    led2[(led2 == 0).all(axis=1)] = np.nan

    t_cam = video_timestamps.index.to_numpy(dtype=float)
    sample_rate_hz = (
        float(1.0 / np.median(np.diff(t_cam))) if len(t_cam) > 1 else float("nan")
    )

    return PositionPixels(
        led1_pixels=led1,
        led2_pixels=led2,
        t_cam=t_cam,
        confidence=None,
        coords_meters_per_pixel=None,
        diagnostics={
            "format": "trodes_native",
            "ptp_pause_frames_removed": int(initial_frame_count - len(t_cam)),
            "frame_count": len(t_cam),
            "sample_rate_hz": sample_rate_hz,
            "position_tracking_file": str(position_tracking_file),
            "camera_timestamps_file": str(camera_timestamps_file),
        },
    )


def _check_ptp_filename(camera_timestamps_file: Path) -> None:
    """Reject non-PTP ``videoTimeStamps`` variants up front.

    The Trodes regex
    (``trodes/python/trodes/trodes_data.py:719``) matches both PTP
    (``cameraHWSync``) and non-PTP (``cameraHWFrameCount`` / plain
    ``videoTimeStamps``) suffixes; v1 of this loader only accepts the
    PTP variant.
    """

    name = camera_timestamps_file.name
    if name.endswith(".videoTimeStamps.cameraHWSync"):
        return
    if name.endswith(".videoTimeStamps.cameraHWFrameCount") or name.endswith(
        ".videoTimeStamps"
    ):
        raise ValueError(
            f"camera_timestamps_file={camera_timestamps_file} is a "
            "non-PTP timestamps variant. v1 of the trodes_native loader "
            "supports PTP-synced timestamps only "
            "(*.videoTimeStamps.cameraHWSync). Convert the session via "
            "trodes_to_nwb and use inputs.format='nwb', or wait for a "
            "future loader version."
        )
    raise ValueError(
        f"camera_timestamps_file={camera_timestamps_file} does not have "
        "a recognized Trodes videoTimeStamps suffix; expected "
        "*.videoTimeStamps.cameraHWSync."
    )


# ----------------------------------------------------------------------
# Vendored Trodes binary parser (PTP branch only).
# Adapted from trodes_to_nwb @ eae7abae src/trodes_to_nwb/convert_position.py
# ----------------------------------------------------------------------


def _parse_dtype(fieldstr: str) -> np.dtype:
    """Parse a Trodes ``Fields`` header string into a numpy dtype.

    The header line looks like ``<time uint32><HWframeCount uint32>
    <HWTimestamp uint64>``; this returns the matching record dtype.
    """

    sep = " ".join(
        fieldstr.replace("><", " ").replace(">", " ").replace("<", " ").split()
    ).split()
    typearr: list[tuple[str, type, int]] = []
    for i in range(0, len(sep), 2):
        fieldname = sep[i]
        repeats = 1
        ftype = "uint32"
        if "*" in sep[i + 1]:
            temptypes = re.split(r"\*", sep[i + 1])
            ftype = temptypes[temptypes[0].isdigit()]
            repeats = int(temptypes[temptypes[1].isdigit()])
        else:
            ftype = sep[i + 1]
        try:
            fieldtype = getattr(np, ftype)
        except AttributeError as err:
            raise AttributeError(f"{ftype} is not a valid field type.") from err
        typearr.append((str(fieldname), fieldtype, repeats))
    return np.dtype(typearr)


def _read_trodes_datafile(filename: Path) -> dict[str, Any]:
    """Read a Trodes ``<Start settings>...<End settings>`` binary.

    Returns a dict with the parsed header (lowercased keys) plus a
    ``data`` numpy structured array.
    """

    with open(filename, "rb") as file:
        first = file.readline().decode().strip()
        if first != "<Start settings>":
            raise ValueError(
                f"{filename} is not a Trodes binary: expected first "
                f"line '<Start settings>', got {first!r}."
            )
        fields_text: dict[str, Any] = {}
        for raw_line in file:
            line = raw_line.decode().strip()
            if line == "<End settings>":
                break
            settings_name, _, setting = line.partition(": ")
            fields_text[settings_name.lower()] = setting
        else:
            raise ValueError(
                f"{filename} settings block not terminated by '<End settings>'."
            )
        try:
            fields_text["data"] = np.fromfile(
                file, dtype=_parse_dtype(fields_text["fields"])
            )
        except KeyError as err:
            raise ValueError(
                f"{filename} header missing required 'Fields' declaration."
            ) from err
        return fields_text


def _convert_datafile_to_pandas(datafile: dict[str, Any]) -> pd.DataFrame:
    """Convert the structured-array ``data`` to a flat DataFrame."""

    return pd.DataFrame(
        {key: np.squeeze(datafile["data"][key]) for key in datafile["data"].dtype.names}
    )


def _detect_repeat_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Boolean mask of timestamps where the current value <= previous."""

    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=bool)
    return np.insert(timestamps[:-1] >= timestamps[1:], 0, False)


def _find_large_frame_jumps(
    frame_count: np.ndarray, min_frame_jump: int = 15
) -> np.ndarray:
    """Boolean mask of frames where the count jumps by ``> min_frame_jump``."""

    return np.insert(np.diff(np.asarray(frame_count)) > min_frame_jump, 0, False)


def _detect_trodes_time_repeats_or_frame_jumps(
    trodes_time: np.ndarray, frame_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find non-repeating Trodes time labels (drop disconnect/jump segments)."""

    is_repeat = _detect_repeat_timestamps(np.asarray(trodes_time))
    is_repeat = np.logical_or(is_repeat, _find_large_frame_jumps(frame_count))

    repeat_labels = label(is_repeat)[0]
    repeat_label_ids, repeat_label_counts = np.unique(repeat_labels, return_counts=True)
    keep_mask = np.logical_and(repeat_label_ids != 0, repeat_label_counts > 2)
    keep_repeat_ids = repeat_label_ids[keep_mask]
    is_repeat[~np.isin(repeat_labels, keep_repeat_ids)] = False

    non_repeat_labels = label(~is_repeat)[0]
    non_repeat_label_ids = np.unique(non_repeat_labels)
    non_repeat_label_ids = non_repeat_label_ids[non_repeat_label_ids != 0]
    return non_repeat_labels, non_repeat_label_ids


def _ptp_pause_index(timestamps_s: np.ndarray) -> int:
    """Index just after the PTP timing pause (used to drop initial frames).

    The Trodes acquisition emits a 0.4-2.0s pause near the start of
    every PTP-synced session; frames before the pause are dropped.
    """

    head = np.asarray(timestamps_s[:100])
    diffs = np.diff(head)
    pauses = np.nonzero(
        np.logical_and(diffs > DEFAULT_MIN_PTP_PAUSE_S, diffs < DEFAULT_MAX_PTP_PAUSE_S)
    )[0]
    if pauses.size == 0:
        return 0
    return int(pauses[0]) + 1


def _get_position_timestamps_ptp(video_timestamps: pd.DataFrame) -> pd.DataFrame:
    """Convert ``HWTimestamp`` (ns) to a PTP-time (s) index and trim the pause."""

    if "HWTimestamp" not in video_timestamps.columns:
        raise ValueError(
            "'HWTimestamp' column missing from input DataFrame for PTP processing."
        )
    ptp_systime = np.asarray(video_timestamps["HWTimestamp"])
    ptp_seconds = pd.Index(ptp_systime / NANOSECONDS_PER_SECOND, name="time")
    if datetime.datetime.fromtimestamp(ptp_seconds[0]).year < 2000:
        logger.warning(
            "PTP timestamps correspond to a time earlier than 2000; "
            "this may be due to a PTP clock reset."
        )
    video_timestamps = video_timestamps.drop(
        columns=["HWframeCount", "HWTimestamp"]
    ).set_index(ptp_seconds)
    pause_ind = _ptp_pause_index(video_timestamps.index.to_numpy())
    return video_timestamps.iloc[pause_ind:]


def _get_position_timestamps_ptp_joined(
    timestamps_file: Path, position_tracking_file: Path
) -> pd.DataFrame:
    """Read both files, dedup the Trodes sample-count clock, join, PTP-stitch.

    Mirrors the PTP branch of ``trodes_to_nwb.convert_position.get_position_timestamps``
    inline so the reader stays self-contained.
    """

    datafile = _read_trodes_datafile(timestamps_file)
    video_timestamps = (
        _convert_datafile_to_pandas(datafile)
        .set_index("PosTimestamp")
        .rename(columns={"frameCount": "HWframeCount"})
    )
    # AVT cameras: HWframeCount is uint16 and wraps; unwrap to int32.
    video_timestamps["HWframeCount"] = np.unwrap(
        video_timestamps["HWframeCount"].astype(np.int32),
        period=np.iinfo(np.uint16).max,
    )
    video_timestamps["video_frame_ind"] = np.arange(len(video_timestamps))

    non_repeat_labels, _ = _detect_trodes_time_repeats_or_frame_jumps(
        video_timestamps.index.to_numpy(),
        video_timestamps["HWframeCount"].to_numpy(),
    )
    video_timestamps["non_repeat_timestamp_labels"] = non_repeat_labels
    video_timestamps = video_timestamps.loc[
        video_timestamps["non_repeat_timestamp_labels"] > 0
    ]

    position_tracking = _convert_datafile_to_pandas(
        _read_trodes_datafile(position_tracking_file)
    ).set_index("time")
    is_repeat = _detect_repeat_timestamps(position_tracking.index.to_numpy())
    position_tracking = position_tracking.iloc[~is_repeat]

    video_timestamps = pd.merge(
        video_timestamps,
        position_tracking,
        right_index=True,
        left_index=True,
        how="left",
    )
    return _get_position_timestamps_ptp(video_timestamps)
