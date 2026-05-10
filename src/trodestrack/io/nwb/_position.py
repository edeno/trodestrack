"""``from_position_container`` and Trodes-Position helpers.

Pure container-layer code — does not import ``pynwb`` at module
load. NWB attributes are read duck-typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trodestrack.config.schemas import NWBLEDSourceConfig
from trodestrack.io.loaders._shared import PositionPixels
from trodestrack.io.nwb._shared import (
    eager_array,
    read_trodes_pair,
    resolve_coords_mpp,
    resolve_coords_mpp_single,
)

if TYPE_CHECKING:
    import pynwb


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

    led1, led2, t_cam, scale1, scale2 = read_trodes_pair(series1, series2)

    # The two series are written with the same ``conversion`` per
    # writer convention; if a future writer diverges, we use the
    # smaller (more conservative) value and surface the mismatch in
    # diagnostics.
    coords_mpp = resolve_coords_mpp([series1, series2], [scale1, scale2])

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
    data = eager_array(series.data, dtype=float)
    t_cam = eager_array(series.timestamps, dtype=float)
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
    coords_mpp = resolve_coords_mpp_single(series, conversion)
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
