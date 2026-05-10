"""Unit tests for the pixel→meter calibration helper."""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.config.schemas import CameraConfig
from trodestrack.io.loaders import PositionPixels
from trodestrack.io.pixel_to_meters import pixels_to_meters


def _make_pixels(
    coords_meters_per_pixel: float | None = None,
    led2: bool = True,
    confidence: bool = False,
) -> PositionPixels:
    led1_pixels = np.array([[100.0, 200.0], [110.0, 210.0]], dtype=float)
    led2_pixels: np.ndarray | None = None
    if led2:
        led2_pixels = np.array([[105.0, 205.0], [115.0, 215.0]], dtype=float)
    conf: np.ndarray | None = None
    if confidence:
        conf = np.array([[0.9, 0.9, 0.8, 0.8], [0.7, 0.7, 0.6, 0.6]], dtype=float)
    return PositionPixels(
        led1_pixels=led1_pixels,
        led2_pixels=led2_pixels,
        t_cam=np.array([0.0, 0.033], dtype=float),
        confidence=conf,
        coords_meters_per_pixel=coords_meters_per_pixel,
    )


def test_override_wins_over_file_side_and_yaml() -> None:
    """``meters_per_pixel_override`` takes precedence over both file-side
    and YAML scales."""

    pixels = _make_pixels(coords_meters_per_pixel=0.005)
    cam = CameraConfig(meters_per_pixel=0.0022)
    led1, led2, _ = pixels_to_meters(pixels, cam, meters_per_pixel_override=0.001)

    assert pixels.led2_pixels is not None and led2 is not None
    np.testing.assert_allclose(led1, pixels.led1_pixels * 0.001)
    np.testing.assert_allclose(led2, pixels.led2_pixels * 0.001)


def test_file_side_wins_when_no_override() -> None:
    """``coords_meters_per_pixel`` from the source wins over the YAML
    default when no override is provided."""

    pixels = _make_pixels(coords_meters_per_pixel=0.005)
    cam = CameraConfig(meters_per_pixel=0.0022)
    led1, led2, _ = pixels_to_meters(pixels, cam)

    assert pixels.led2_pixels is not None and led2 is not None
    np.testing.assert_allclose(led1, pixels.led1_pixels * 0.005)
    np.testing.assert_allclose(led2, pixels.led2_pixels * 0.005)


def test_yaml_default_wins_when_file_side_is_none() -> None:
    """``CameraConfig.meters_per_pixel`` is the fallback when neither
    override nor file-side scale is provided."""

    pixels = _make_pixels(coords_meters_per_pixel=None)
    cam = CameraConfig(meters_per_pixel=0.0022)
    led1, led2, _ = pixels_to_meters(pixels, cam)

    assert pixels.led2_pixels is not None and led2 is not None
    np.testing.assert_allclose(led1, pixels.led1_pixels * 0.0022)
    np.testing.assert_allclose(led2, pixels.led2_pixels * 0.0022)


def test_led2_none_passes_through() -> None:
    """Single-LED sources (``led2_pixels=None``) propagate through
    unchanged."""

    pixels = _make_pixels(led2=False)
    cam = CameraConfig(meters_per_pixel=0.0022)
    led1, led2, _ = pixels_to_meters(pixels, cam)

    assert led2 is None
    np.testing.assert_allclose(led1, pixels.led1_pixels * 0.0022)


def test_confidence_passes_through_unchanged() -> None:
    """The conversion does not touch the per-LED confidence array."""

    pixels = _make_pixels(confidence=True)
    cam = CameraConfig(meters_per_pixel=0.0022)
    _, _, conf = pixels_to_meters(pixels, cam)

    assert conf is pixels.confidence


@pytest.mark.parametrize("scale", [0.001, 0.0022, 0.005])
def test_round_trip_reverses_with_scale(scale: float) -> None:
    """Dividing the converted output by the scale recovers the input
    pixel coordinates exactly (single-precision tolerance)."""

    pixels = _make_pixels()
    cam = CameraConfig(meters_per_pixel=scale)
    led1, _, _ = pixels_to_meters(pixels, cam)

    np.testing.assert_allclose(led1 / scale, pixels.led1_pixels)
