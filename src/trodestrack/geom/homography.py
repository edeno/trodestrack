"""Homography computation and coordinate transformation utilities."""

from typing import List, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

# Enable 64-bit precision in JAX for accurate homography computations
jax.config.update("jax_enable_x64", True)


class HomographyTransform:
    """Class for applying homography transformations between pixel and cm coordinates."""

    def __init__(self, homography_matrix: Union[List[List[float]], np.ndarray, jnp.ndarray]):
        """
        Initialize homography transform.

        Args:
            homography_matrix: 3x3 homography matrix for pixel->cm transformation

        Raises:
            ValueError: If matrix is not 3x3
        """
        # Convert to numpy array for validation
        matrix = np.array(homography_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3")

        # Store as JAX array for computation (use float64 for precision)
        self.H = jnp.array(matrix, dtype=jnp.float64)
        self.H_inv = jnp.linalg.inv(self.H)

    def pixel_to_cm(self, points_pixel: np.ndarray) -> np.ndarray:
        """
        Transform points from pixel coordinates to cm coordinates.

        Args:
            points_pixel: Array of shape (N, 2) with pixel coordinates

        Returns:
            Array of shape (N, 2) with cm coordinates
        """
        return np.array(_transform_points_jax(jnp.array(points_pixel), self.H))

    def cm_to_pixel(self, points_cm: np.ndarray) -> np.ndarray:
        """
        Transform points from cm coordinates to pixel coordinates.

        Args:
            points_cm: Array of shape (N, 2) with cm coordinates

        Returns:
            Array of shape (N, 2) with pixel coordinates
        """
        return np.array(_transform_points_jax(jnp.array(points_cm), self.H_inv))


@jit
def _transform_points_jax(points: jnp.ndarray, H: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-compiled homography transformation of points.

    Args:
        points: Array of shape (N, 2) with 2D coordinates
        H: 3x3 homography matrix

    Returns:
        Transformed points of shape (N, 2)
    """
    # Convert to homogeneous coordinates (N, 3)
    ones = jnp.ones((points.shape[0], 1))
    points_homogeneous = jnp.concatenate([points, ones], axis=1)

    # Apply transformation: (3, N) = (3, 3) @ (3, N)
    transformed_homogeneous = H @ points_homogeneous.T

    # Convert back to 2D coordinates (divide by w)
    transformed_2d = transformed_homogeneous[:2] / transformed_homogeneous[2]

    return transformed_2d.T


def compute_homography_from_corners(
    pixel_corners: np.ndarray, cm_corners: np.ndarray
) -> np.ndarray:
    """
    Compute homography matrix from corresponding corner points.

    Uses the direct linear transformation (DLT) method to solve for
    the homography matrix H such that: cm_point = H @ pixel_point

    Args:
        pixel_corners: Array of shape (4, 2) with pixel coordinates of arena corners
        cm_corners: Array of shape (4, 2) with corresponding cm coordinates

    Returns:
        3x3 homography matrix

    Raises:
        ValueError: If corners are degenerate (collinear) or insufficient
    """
    if pixel_corners.shape != (4, 2) or cm_corners.shape != (4, 2):
        raise ValueError("Exactly 4 corner points required for homography computation")

    # Build the A matrix for the DLT algorithm
    # Each point pair contributes 2 rows to the system Ah = 0
    A = []
    for (px, py), (cx, cy) in zip(pixel_corners, cm_corners):
        # First row: -px*cx - py*cy - cx + u*px + v*py + u = 0
        # Second row: -px*cy - py*cy - cy + u*px + v*py + v = 0
        # Where H = [[a, b, c], [d, e, f], [g, h, 1]] (normalized so H[2,2] = 1)
        # This gives us: cx = (a*px + b*py + c) / (g*px + h*py + 1)
        #               cy = (d*px + e*py + f) / (g*px + h*py + 1)

        # Rearranging: (a*px + b*py + c) - cx*(g*px + h*py + 1) = 0
        #             (d*px + e*py + f) - cy*(g*px + h*py + 1) = 0

        # First equation: a*px + b*py + c - cx*g*px - cx*h*py - cx = 0
        A.append([px, py, 1, 0, 0, 0, -cx * px, -cx * py, -cx])

        # Second equation: d*px + e*py + f - cy*g*px - cy*h*py - cy = 0
        A.append([0, 0, 0, px, py, 1, -cy * px, -cy * py, -cy])

    A = np.array(A)

    # Solve Ah = 0 using SVD
    try:
        U, s, Vt = np.linalg.svd(A)

        # Check for degenerate configuration (smallest singular value too small)
        if s[-1] < 1e-10:
            raise ValueError("Cannot compute homography from given corners (degenerate configuration)")

        h = Vt[-1]  # Last row of V^T (corresponding to smallest singular value)

        # Reshape to 3x3 matrix
        H = h.reshape(3, 3)

        # Check if H[2, 2] is close to zero (another degenerate case)
        if abs(H[2, 2]) < 1e-10:
            raise ValueError("Cannot compute homography from given corners (degenerate configuration)")

        # Normalize so that H[2, 2] = 1
        H = H / H[2, 2]

        return H

    except np.linalg.LinAlgError as e:
        raise ValueError("Cannot compute homography from given corners (degenerate configuration)") from e


def transform_points_pixel_to_cm(
    points_pixel: np.ndarray, homography_matrix: List[List[float]]
) -> np.ndarray:
    """
    Standalone function to transform points from pixels to cm.

    Args:
        points_pixel: Array of shape (N, 2) with pixel coordinates
        homography_matrix: 3x3 homography matrix as nested list

    Returns:
        Array of shape (N, 2) with cm coordinates
    """
    transform = HomographyTransform(homography_matrix)
    return transform.pixel_to_cm(points_pixel)


def transform_points_cm_to_pixel(
    points_cm: np.ndarray, homography_matrix: List[List[float]]
) -> np.ndarray:
    """
    Standalone function to transform points from cm to pixels.

    Args:
        points_cm: Array of shape (N, 2) with cm coordinates
        homography_matrix: 3x3 homography matrix as nested list

    Returns:
        Array of shape (N, 2) with pixel coordinates
    """
    transform = HomographyTransform(homography_matrix)
    return transform.cm_to_pixel(points_cm)