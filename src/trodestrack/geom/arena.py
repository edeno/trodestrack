"""Arena bounds validation and coordinate clipping utilities."""

from typing import Tuple

import numpy as np


class ArenaValidator:
    """Validator for checking if coordinates are within arena bounds."""

    def __init__(self, bounds: Tuple[float, float, float, float]):
        """
        Initialize arena validator.

        Args:
            bounds: Arena bounds as (x_min, y_min, x_max, y_max) in cm
        """
        self.x_min, self.y_min, self.x_max, self.y_max = bounds

        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Invalid arena bounds: min values must be less than max values")

    def is_inside(self, point: np.ndarray) -> bool:
        """
        Check if a single point is inside arena bounds.

        Args:
            point: Array of shape (2,) with (x, y) coordinates

        Returns:
            True if point is inside bounds, False otherwise
        """
        x, y = point
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    def is_inside_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Check if multiple points are inside arena bounds.

        Args:
            points: Array of shape (N, 2) with (x, y) coordinates

        Returns:
            Boolean array of shape (N,) indicating which points are inside
        """
        return check_arena_bounds(points, (self.x_min, self.y_min, self.x_max, self.y_max))

    def clip_to_bounds(self, points: np.ndarray) -> np.ndarray:
        """
        Clip points to arena boundaries.

        Args:
            points: Array of shape (N, 2) with (x, y) coordinates

        Returns:
            Array of shape (N, 2) with clipped coordinates
        """
        return clip_to_arena_bounds(points, (self.x_min, self.y_min, self.x_max, self.y_max))

    def get_center(self) -> np.ndarray:
        """
        Get the center point of the arena.

        Returns:
            Array of shape (2,) with arena center coordinates
        """
        return get_arena_center((self.x_min, self.y_min, self.x_max, self.y_max))

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get arena bounds.

        Returns:
            Arena bounds as (x_min, y_min, x_max, y_max)
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)


def check_arena_bounds(
    points: np.ndarray, bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Check if points are within arena bounds.

    Args:
        points: Array of shape (N, 2) with (x, y) coordinates
        bounds: Arena bounds as (x_min, y_min, x_max, y_max)

    Returns:
        Boolean array of shape (N,) indicating which points are inside bounds
    """
    x_min, y_min, x_max, y_max = bounds

    x_inside = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    y_inside = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)

    return x_inside & y_inside


def clip_to_arena_bounds(
    points: np.ndarray, bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Clip coordinates to arena bounds.

    Args:
        points: Array of shape (N, 2) with (x, y) coordinates
        bounds: Arena bounds as (x_min, y_min, x_max, y_max)

    Returns:
        Array of shape (N, 2) with coordinates clipped to bounds
    """
    x_min, y_min, x_max, y_max = bounds

    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], x_min, x_max)
    clipped[:, 1] = np.clip(clipped[:, 1], y_min, y_max)

    return clipped


def get_arena_center(bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Compute the center point of the arena.

    Args:
        bounds: Arena bounds as (x_min, y_min, x_max, y_max)

    Returns:
        Array of shape (2,) with center coordinates
    """
    x_min, y_min, x_max, y_max = bounds
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    return np.array([center_x, center_y])