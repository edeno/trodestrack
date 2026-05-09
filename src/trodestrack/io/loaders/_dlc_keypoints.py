"""DeepLabCut HDF5 keypoint loader.

Reads DLC ``df_with_missing`` HDF5 + sibling ``_meta.pickle`` directly
without going through an intermediate parquet conversion. v1 supports
single-animal projects only; multi-animal MultiIndex layouts are
rejected up front.

Saver compatibility:

- **PyTorch** saver writes ``frame_dimensions=(w, h)`` and includes a
  ``pytorch-config`` key in the metadata
  (``deeplabcut/pose_estimation_pytorch/apis/videos.py:865-883``).
- **TF / legacy** saver writes ``frame_dimensions=(ny, nx)`` (i.e.,
  ``(h, w)``) and includes a ``DLC-model-config file`` key
  (``deeplabcut/pose_estimation_tensorflow/predict_videos.py:1053-1069``).

The loader detects the saver via metadata-key presence and normalizes
``frame_dimensions`` to ``(width, height)`` before downstream consumers
see it.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from trodestrack.io.loaders._shared import PositionPixels
from trodestrack.io.loaders._trodes_native import (
    NANOSECONDS_PER_SECOND,
    _read_trodes_datafile,
)

logger = logging.getLogger(__name__)


def load_dlc_keypoints_position(
    h5_file: Path,
    led1_bodypart: str,
    led2_bodypart: str,
    *,
    likelihood_threshold: float = 0.6,
    timestamps_source: Literal[
        "meta_pickle", "trodes_hw_sync", "timestamp_file"
    ] = "meta_pickle",
    camera_timestamps_file: Path | None = None,
    timestamp_file: Path | None = None,
    apply_crop_offset: bool = True,
) -> PositionPixels:
    """Read a DLC ``df_with_missing`` HDF5 into a ``PositionPixels``.

    Parameters
    ----------
    h5_file
        Path to ``<video_stem>DLC_*.h5``.
    led1_bodypart, led2_bodypart
        Bodypart names (DLC ``bodyparts`` MultiIndex level) to use as
        LED1 and LED2.
    likelihood_threshold
        Per-frame likelihood gate; rows below threshold are NaN'd in
        ``led{1,2}_pixels`` and the per-LED columns of ``confidence``.
        Default 0.6 matches DLC's ``pcutoff`` default
        (``deeplabcut/create_project/new.py:243``).
    timestamps_source
        Where camera frame timestamps come from:
        ``"meta_pickle"`` synthesizes from the sibling ``_meta.pickle``
        (``np.arange(nframes) / fps``); ``"trodes_hw_sync"`` reads
        ``HWTimestamp`` from a Trodes ``*.videoTimeStamps.cameraHWSync``
        binary; ``"timestamp_file"`` reads a 1-D float-seconds array
        via ``np.loadtxt``.
    camera_timestamps_file
        Required when ``timestamps_source='trodes_hw_sync'``.
    timestamp_file
        Required when ``timestamps_source='timestamp_file'``.
    apply_crop_offset
        When True, add the crop ``(x_min, y_min)`` from the meta
        pickle's ``cropping_parameters`` so output coords are in the
        original video's pixel frame, not the cropped frame.

    Raises
    ------
    ImportError
        If the ``[dlc]`` extra is not installed (``pandas.read_hdf``
        needs PyTables).
    ValueError
        If the HDF5 carries a multi-animal MultiIndex (``individuals``
        level present), the configured bodyparts aren't found, or the
        timestamps_source can't produce a vector of the right length.
    """

    df = _read_dlc_h5(h5_file)
    _reject_multi_animal(df, h5_file=h5_file)
    df = df.droplevel("scorer", axis=1)

    led1_xy, led1_lik = _extract_led(df, led1_bodypart, h5_file=h5_file)
    led2_xy, led2_lik = _extract_led(df, led2_bodypart, h5_file=h5_file)

    led1_pixels = led1_xy.copy()
    led2_pixels = led2_xy.copy()
    led1_pixels[led1_lik < likelihood_threshold] = np.nan
    led2_pixels[led2_lik < likelihood_threshold] = np.nan

    metadata = _read_meta_pickle(h5_file)
    crop_params = metadata.get("cropping_parameters")
    crop_offset = (0.0, 0.0)
    if apply_crop_offset and crop_params is not None and len(crop_params) == 4:
        x_min = float(crop_params[0])
        y_min = float(crop_params[2])
        if x_min != 0.0 or y_min != 0.0:
            led1_pixels[:, 0] += x_min
            led1_pixels[:, 1] += y_min
            led2_pixels[:, 0] += x_min
            led2_pixels[:, 1] += y_min
            crop_offset = (x_min, y_min)

    n_frames = len(df)
    t_cam = _resolve_timestamps(
        timestamps_source,
        n_frames,
        meta=metadata,
        camera_timestamps_file=camera_timestamps_file,
        timestamp_file=timestamp_file,
    )

    # Match the existing _load_leds confidence layout: (n, 4) with
    # per-LED likelihoods replicated across x/y.
    confidence = np.column_stack([led1_lik, led1_lik, led2_lik, led2_lik])

    frame_dimensions = _normalize_frame_dimensions(metadata)
    saver = _detect_saver(metadata)

    led1_kept = float(np.mean(led1_lik >= likelihood_threshold))
    led2_kept = float(np.mean(led2_lik >= likelihood_threshold))

    return PositionPixels(
        led1_pixels=led1_pixels,
        led2_pixels=led2_pixels,
        t_cam=t_cam,
        confidence=confidence,
        frame_dimensions=frame_dimensions,
        coords_meters_per_pixel=None,
        diagnostics={
            "format": "dlc_keypoints",
            "saver": saver,
            "scorer": metadata.get("Scorer"),
            "fps": metadata.get("fps"),
            "nframes": metadata.get("nframes"),
            "frame_dimensions": frame_dimensions,
            "crop_offset_applied": crop_offset,
            "led1_kept_fraction": led1_kept,
            "led2_kept_fraction": led2_kept,
            "h5_file": str(h5_file),
            "timestamps_source": timestamps_source,
        },
    )


def _read_dlc_h5(h5_file: Path) -> pd.DataFrame:
    """Read a DLC ``df_with_missing`` HDF5, surfacing a clear ImportError
    when the ``[dlc]`` extra (PyTables) is missing."""

    try:
        result = pd.read_hdf(h5_file, key="df_with_missing")
    except ImportError as e:
        raise ImportError(
            "inputs.format='dlc_keypoints' requires the [dlc] extra. "
            "Install with: uv pip install 'trodestrack[dlc]'."
        ) from e
    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"{h5_file} key='df_with_missing' must be a DataFrame; "
            f"got {type(result).__name__}."
        )
    return result


def _reject_multi_animal(df: pd.DataFrame, *, h5_file: Path) -> None:
    """Multi-animal projects carry an ``individuals`` MultiIndex level
    (``deeplabcut/pose_estimation_pytorch/apis/videos.py:741-744``).
    Reject up front rather than silently slicing individual 0."""

    levels = list(df.columns.names)
    if "individuals" in levels:
        raise ValueError(
            f"{h5_file} is a multi-animal DLC project (``individuals`` "
            f"MultiIndex level present, levels={levels}). v1 of "
            "dlc_keypoints supports single-animal projects only; "
            "export an individual to a single-animal HDF5 or wait for "
            "a future loader version."
        )


def _extract_led(
    df: pd.DataFrame, bodypart: str, *, h5_file: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(xy, likelihood)`` for a named bodypart from a
    ``(bodyparts, coords)``-indexed DataFrame."""

    bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
    if bodypart not in bodyparts:
        raise ValueError(
            f"{h5_file} bodypart={bodypart!r} not found. Available: {bodyparts}."
        )
    sub = df.xs(bodypart, level="bodyparts", axis=1)
    coords = sub.columns.get_level_values("coords").tolist()
    for required in ("x", "y", "likelihood"):
        if required not in coords:
            raise ValueError(
                f"{h5_file} bodypart={bodypart!r} missing required "
                f"coord {required!r} (have {coords})."
            )
    xy = sub[["x", "y"]].to_numpy(dtype=float)
    likelihood = sub["likelihood"].to_numpy(dtype=float)
    return xy, likelihood


def _read_meta_pickle(h5_file: Path) -> dict[str, Any]:
    """Read the sibling ``<h5_stem>_meta.pickle`` and unwrap the
    standard ``{"data": {...}}`` envelope.

    PyTorch and TF savers both wrap in ``data`` (per
    ``predict_videos.py:1069`` and ``videos.py:883``). The unwrap is
    forgiving: if a future saver omits the wrapper, we accept a
    flat dict.
    """

    pickle_path = h5_file.with_name(h5_file.stem + "_meta.pickle")
    if not pickle_path.exists():
        # Returning an empty dict lets downstream code surface its own
        # specific errors (e.g., "fps required for meta_pickle source").
        return {}
    with pickle_path.open("rb") as f:
        loaded = pickle.load(f)
    if (
        isinstance(loaded, dict)
        and "data" in loaded
        and isinstance(loaded["data"], dict)
    ):
        return loaded["data"]
    return loaded if isinstance(loaded, dict) else {}


def _detect_saver(metadata: dict[str, Any]) -> str:
    """Detect the DLC saver variant from metadata key presence.

    PyTorch metadata includes ``pytorch-config``; TF includes
    ``DLC-model-config file``. Falls back to ``"unknown"`` so the
    diagnostic surfaces the gap without breaking the loader.
    """

    if "pytorch-config" in metadata:
        return "pytorch"
    if "DLC-model-config file" in metadata:
        return "tensorflow"
    return "unknown"


def _normalize_frame_dimensions(
    metadata: dict[str, Any],
) -> tuple[int, int] | None:
    """Normalize ``frame_dimensions`` to ``(width, height)``.

    PyTorch writes ``(w, h)`` directly; TF writes ``(ny, nx)`` =
    ``(h, w)`` and we swap.
    """

    raw = metadata.get("frame_dimensions")
    if raw is None or len(raw) != 2:
        return None
    if _detect_saver(metadata) == "tensorflow":
        h, w = raw
        return int(w), int(h)
    w, h = raw
    return int(w), int(h)


def _resolve_timestamps(
    source: Literal["meta_pickle", "trodes_hw_sync", "timestamp_file"],
    n_frames: int,
    *,
    meta: dict[str, Any],
    camera_timestamps_file: Path | None,
    timestamp_file: Path | None,
) -> np.ndarray:
    if source == "meta_pickle":
        fps = meta.get("fps")
        if fps is None or fps <= 0:
            raise ValueError(
                "dlc_keypoints.timestamps_source='meta_pickle' requires "
                "a positive 'fps' value in the sibling ``_meta.pickle`` "
                f"(got {fps!r}). Provide ``timestamps_source: "
                "trodes_hw_sync`` (with camera_timestamps_file) or "
                "``timestamps_source: timestamp_file`` instead."
            )
        return np.arange(n_frames, dtype=float) / float(fps)

    if source == "trodes_hw_sync":
        assert camera_timestamps_file is not None  # schema validates
        datafile = _read_trodes_datafile(camera_timestamps_file)
        dtype_names = datafile["data"].dtype.names or ()
        if "HWTimestamp" not in dtype_names:
            raise ValueError(
                f"camera_timestamps_file={camera_timestamps_file} is "
                "missing the required 'HWTimestamp' column for "
                "PTP-synced timestamps."
            )
        hw_ns = np.asarray(np.squeeze(datafile["data"]["HWTimestamp"]))
        t = hw_ns.astype(np.float64) / NANOSECONDS_PER_SECOND
        if t.size != n_frames:
            raise ValueError(
                f"camera_timestamps_file has {t.size} timestamps but "
                f"the DLC HDF5 has {n_frames} frames. The two must "
                "agree 1:1 (DLC processes the whole recorded video; "
                "if the recording has a PTP pause, run DLC on the "
                "whole clip rather than a post-pause subset)."
            )
        return t

    assert source == "timestamp_file"
    assert timestamp_file is not None  # schema validates
    t = np.loadtxt(timestamp_file, dtype=float).reshape(-1)
    if t.size != n_frames:
        raise ValueError(
            f"timestamp_file={timestamp_file} has {t.size} entries "
            f"but the DLC HDF5 has {n_frames} frames."
        )
    return t
