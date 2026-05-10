"""Loader-private intermediate types shared across format readers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PositionPixels:
    """Loader-private intermediate carrying pixel-space LED coordinates.

    Every format reader returns one of these before meter conversion.
    ``pixels_to_meters`` (in ``src/trodestrack/io/pixel_to_meters.py``)
    is the single point that applies the camera calibration.

    Parameters
    ----------
    led1_pixels
        ``(n_cam, 2)`` float array of LED1 (x, y) pixel coordinates.
        NaN rows mark frames where LED1 is invalid / not detected.
    led2_pixels
        Optional second-LED ``(n_cam, 2)`` array. ``None`` for
        single-LED formats; NaN rows otherwise.
    t_cam
        ``(n_cam,)`` camera timestamps in seconds.
    confidence
        Optional ``(n_cam, 4)`` array laid out as
        ``[c1x, c1y, c2x, c2y]`` (per-LED replicated across x/y to
        match the existing ``conf_cam`` shape consumers expect).
    frame_dimensions
        Optional ``(width, height)`` tuple in pixels when the source
        records it.
    diagnostics
        Source-specific metadata the format reader wants to surface in
        the session diagnostics block (kept distinct from the
        ``PreparedSession.diagnostics`` produced after meter
        conversion).
    coords_meters_per_pixel
        Calibration the reader recovered from the source file. ``None``
        means "no calibration baked into the source"; the conversion
        helper falls back to ``CameraConfig.meters_per_pixel``. NWB
        readers populate this from the per-series ``conversion`` only
        when ``unit == "meters"`` or ``conversion != 1.0``; the
        ``pixels`` + ``conversion=1.0`` sentinel case leaves it
        ``None``. The reader does *not* pre-multiply pixel data by
        this value; ``pixels_to_meters`` is the single point of
        conversion.
    """

    led1_pixels: np.ndarray
    led2_pixels: np.ndarray | None
    t_cam: np.ndarray
    confidence: np.ndarray | None = None
    frame_dimensions: tuple[int, int] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    coords_meters_per_pixel: float | None = None
