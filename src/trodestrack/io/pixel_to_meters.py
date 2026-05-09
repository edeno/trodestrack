"""Single point of pixel→meter conversion shared across loaders."""

from __future__ import annotations

import numpy as np

from trodestrack.config.schemas import CameraConfig
from trodestrack.io.loaders import PositionPixels


def pixels_to_meters(
    pixels: PositionPixels,
    camera_config: CameraConfig,
    *,
    meters_per_pixel_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply camera calibration to pixel-space LED coordinates.

    Resolution order for the scale factor (the geom-homography plan
    will add a homography branch later, same module path):

    1. ``meters_per_pixel_override`` (NWB knob — wins; for post-hoc
       re-calibration).
    2. ``pixels.coords_meters_per_pixel`` (file-side, e.g. NWB
       per-series ``conversion`` when the file is calibrated).
    3. ``camera_config.meters_per_pixel`` (the YAML default).

    Without this precedence, NWB-calibrated data would silently
    re-scale by the YAML's ``meters_per_pixel`` default
    (``CameraConfig.meters_per_pixel`` has a non-null default of
    0.0022), so we track the file-side value explicitly.

    Returns
    -------
    led1_meters : (n_cam, 2) float array
    led2_meters : (n_cam, 2) float array or ``None`` when the source
        only carries one LED.
    confidence : (n_cam, 4) float array or ``None``; passed through
        unchanged from ``pixels.confidence``.
    """

    if meters_per_pixel_override is not None:
        scale = meters_per_pixel_override
    elif pixels.coords_meters_per_pixel is not None:
        scale = pixels.coords_meters_per_pixel
    else:
        scale = camera_config.meters_per_pixel

    led1_meters = np.asarray(pixels.led1_pixels, dtype=float) * scale
    if pixels.led2_pixels is None:
        led2_meters: np.ndarray | None = None
    else:
        led2_meters = np.asarray(pixels.led2_pixels, dtype=float) * scale
    return led1_meters, led2_meters, pixels.confidence
