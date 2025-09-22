"""Geometry and calibration utilities for trodestrack."""

from .arena import (
    ArenaValidator,
    check_arena_bounds,
    clip_to_arena_bounds,
    get_arena_center,
)
from .homography import (
    HomographyTransform,
    compute_homography_from_corners,
    transform_points_cm_to_pixel,
    transform_points_pixel_to_cm,
)

__all__ = [
    "HomographyTransform",
    "compute_homography_from_corners",
    "transform_points_pixel_to_cm",
    "transform_points_cm_to_pixel",
    "ArenaValidator",
    "check_arena_bounds",
    "clip_to_arena_bounds",
    "get_arena_center",
]
