"""Cross-cutting helpers for the NWB loader modules.

These helpers are pynwb-free at module load — every NWB attribute is
read duck-typed so a Spyglass ``make()`` callsite can use them
without trodestrack carrying a hard ``pynwb`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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


def eager_array(dataset: Any, *, dtype: Any = None) -> np.ndarray:
    """Slice a pynwb / h5py dataset into a numpy array.

    The ``[...]`` indexing materializes the underlying h5py dataset
    into RAM, breaking the dependency on the source IO handle. The
    asarray pass enforces a concrete dtype.
    """

    arr = np.asarray(dataset[...])
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def neurodata_type(container: Any) -> str:
    """Return the NWB neurodata-type string for a container.

    Uses the ``neurodata_type`` attribute pynwb sets on every typed
    container; falls back to the class name when reading
    dynamic-spec containers without their typed Python class.
    """

    return str(getattr(container, "neurodata_type", type(container).__name__))


def detect_led_container(nwbfile: Any, container_choice: str) -> Any:
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
        ndt = neurodata_type(di)
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


def read_trodes_pair(
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

    led1 = eager_array(series1.data, dtype=float)
    led2 = eager_array(series2.data, dtype=float)
    t1 = eager_array(series1.timestamps, dtype=float)
    t2 = eager_array(series2.timestamps, dtype=float)
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


def resolve_coords_mpp(serieses: list[Any], conversions: list[float]) -> float | None:
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


def resolve_coords_mpp_single(series: Any, conversion: float) -> float | None:
    """Single-series counterpart to ``resolve_coords_mpp``.

    Same conversion=1.0 / ``unit`` sentinel rule applied to one
    SpatialSeries; no pair-mismatch check because there is no pair.
    """

    unit = getattr(series, "unit", "pixels")
    if unit == "pixels" and conversion == 1.0:
        return None
    return conversion
