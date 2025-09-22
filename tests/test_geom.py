"""Tests for geometry and coordinate transformation utilities."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from trodestrack.geom.homography import (
    HomographyTransform,
    compute_homography_from_corners,
    transform_points_pixel_to_cm,
    transform_points_cm_to_pixel,
)
from trodestrack.geom.arena import (
    ArenaValidator,
    check_arena_bounds,
    clip_to_arena_bounds,
    get_arena_center,
)


class TestHomographyTransform:
    """Test homography transformation utilities."""

    def test_identity_transform_roundtrip(self):
        """Test that identity matrix preserves coordinates."""
        # Identity homography matrix
        identity_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        transform = HomographyTransform(identity_matrix)

        # Test points in pixels
        pixel_points = np.array([[100.0, 150.0], [200.0, 250.0], [300.0, 350.0]])

        # Roundtrip: pixel -> cm -> pixel
        cm_points = transform.pixel_to_cm(pixel_points)
        roundtrip_pixels = transform.cm_to_pixel(cm_points)

        np.testing.assert_allclose(pixel_points, roundtrip_pixels, rtol=1e-6)

    def test_simple_scale_transform(self):
        """Test homography with simple scaling (2 pixels per cm)."""
        # Scale by 0.5 (1 cm = 2 pixels)
        scale_matrix = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        transform = HomographyTransform(scale_matrix)

        pixel_points = np.array([[100.0, 200.0]])
        expected_cm = np.array([[50.0, 100.0]])

        cm_points = transform.pixel_to_cm(pixel_points)
        np.testing.assert_allclose(cm_points, expected_cm, rtol=1e-6)

    def test_translation_transform(self):
        """Test homography with translation offset."""
        # Translation by (10, 20) cm
        translation_matrix = [[1.0, 0.0, -10.0], [0.0, 1.0, -20.0], [0.0, 0.0, 1.0]]
        transform = HomographyTransform(translation_matrix)

        pixel_points = np.array([[0.0, 0.0]])
        expected_cm = np.array([[-10.0, -20.0]])

        cm_points = transform.pixel_to_cm(pixel_points)
        np.testing.assert_allclose(cm_points, expected_cm, rtol=1e-6)

    @given(st.lists(st.lists(st.floats(-1000, 1000), min_size=2, max_size=2), min_size=1, max_size=10))
    def test_homography_roundtrip_property(self, points):
        """Property test: any valid transform should roundtrip."""
        # Use a generic transformation matrix (scale + rotation + translation)
        matrix = [[0.5, 0.1, -50.0], [-0.1, 0.5, -100.0], [0.0, 0.0, 1.0]]
        transform = HomographyTransform(matrix)

        pixel_points = np.array(points)
        cm_points = transform.pixel_to_cm(pixel_points)
        roundtrip_pixels = transform.cm_to_pixel(cm_points)

        np.testing.assert_allclose(pixel_points, roundtrip_pixels, rtol=1e-4, atol=1e-10)

    def test_batch_transform(self):
        """Test batch transformation of multiple points."""
        matrix = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        transform = HomographyTransform(matrix)

        # Multiple points
        pixel_points = np.array([[100.0, 200.0], [150.0, 250.0], [50.0, 75.0]])
        expected_cm = np.array([[50.0, 100.0], [75.0, 125.0], [25.0, 37.5]])

        cm_points = transform.pixel_to_cm(pixel_points)
        np.testing.assert_allclose(cm_points, expected_cm, rtol=1e-6)

    def test_invalid_matrix_shape(self):
        """Test error handling for invalid matrix shapes."""
        with pytest.raises(ValueError, match="Homography matrix must be 3x3"):
            HomographyTransform([[1.0, 0.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="Homography matrix must be 3x3"):
            HomographyTransform([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


class TestHomographyComputation:
    """Test homography matrix computation from corner points."""

    def test_compute_from_four_corners(self):
        """Test computing homography from known arena corners."""
        # Arena corners in pixels (typical camera view)
        pixel_corners = np.array([[100, 150], [500, 160], [480, 400], [120, 390]])

        # Corresponding real-world corners in cm (rectangular arena)
        cm_corners = np.array([[0, 0], [200, 0], [200, 150], [0, 150]])

        # Should compute a valid 3x3 homography matrix
        H = compute_homography_from_corners(pixel_corners, cm_corners)

        assert H.shape == (3, 3)
        assert not np.isnan(H).any()
        assert not np.isinf(H).any()

        # Test that computed homography transforms corners correctly
        transform = HomographyTransform(H.tolist())
        transformed_cm = transform.pixel_to_cm(pixel_corners)

        # Should be close to expected cm_corners (very high precision expected)
        np.testing.assert_allclose(transformed_cm, cm_corners, rtol=1e-2, atol=1e-9)

    def test_degenerate_corners(self):
        """Test error handling for degenerate corner configurations."""
        # Collinear points (degenerate case)
        pixel_corners = np.array([[0, 0], [100, 100], [200, 200], [300, 300]])
        cm_corners = np.array([[0, 0], [50, 50], [100, 100], [150, 150]])

        with pytest.raises(ValueError, match="Cannot compute homography"):
            compute_homography_from_corners(pixel_corners, cm_corners)


class TestArenaValidator:
    """Test arena bounds validation utilities."""

    def test_arena_validator_initialization(self):
        """Test ArenaValidator initialization."""
        bounds = (0.0, 0.0, 200.0, 150.0)  # x_min, y_min, x_max, y_max
        validator = ArenaValidator(bounds)

        assert validator.x_min == 0.0
        assert validator.y_min == 0.0
        assert validator.x_max == 200.0
        assert validator.y_max == 150.0

    def test_point_inside_arena(self):
        """Test detection of points inside arena bounds."""
        bounds = (0.0, 0.0, 200.0, 150.0)
        validator = ArenaValidator(bounds)

        # Points inside arena
        inside_points = np.array([[100.0, 75.0], [50.0, 50.0], [150.0, 100.0]])

        for point in inside_points:
            assert validator.is_inside(point)

    def test_point_outside_arena(self):
        """Test detection of points outside arena bounds."""
        bounds = (0.0, 0.0, 200.0, 150.0)
        validator = ArenaValidator(bounds)

        # Points outside arena
        outside_points = np.array([[-10.0, 75.0], [250.0, 75.0], [100.0, -20.0], [100.0, 200.0]])

        for point in outside_points:
            assert not validator.is_inside(point)

    def test_clip_to_arena_bounds(self):
        """Test clipping coordinates to arena bounds."""
        bounds = (0.0, 0.0, 200.0, 150.0)

        # Points that need clipping
        points = np.array([[-10.0, 75.0], [250.0, 75.0], [100.0, -20.0], [100.0, 200.0]])
        expected_clipped = np.array([[0.0, 75.0], [200.0, 75.0], [100.0, 0.0], [100.0, 150.0]])

        clipped = clip_to_arena_bounds(points, bounds)
        np.testing.assert_allclose(clipped, expected_clipped, rtol=1e-6)

    def test_get_arena_center(self):
        """Test computation of arena center."""
        bounds = (10.0, 20.0, 210.0, 170.0)
        expected_center = np.array([110.0, 95.0])  # (10+210)/2, (20+170)/2

        center = get_arena_center(bounds)
        np.testing.assert_allclose(center, expected_center, rtol=1e-6)

    @given(st.floats(-1000, 1000), st.floats(-1000, 1000), st.floats(1, 1000), st.floats(1, 1000))
    def test_arena_bounds_property(self, x_min, y_min, width, height):
        """Property test: clipped points should always be inside bounds."""
        x_max = x_min + width
        y_max = y_min + height
        bounds = (x_min, y_min, x_max, y_max)

        # Generate random points that might be outside
        points = np.array([[x_min - 100, y_min - 100], [x_max + 100, y_max + 100]])
        clipped = clip_to_arena_bounds(points, bounds)

        # All clipped points should be within bounds
        assert np.all(clipped[:, 0] >= x_min)
        assert np.all(clipped[:, 0] <= x_max)
        assert np.all(clipped[:, 1] >= y_min)
        assert np.all(clipped[:, 1] <= y_max)


class TestCoordinateTransformFunctions:
    """Test standalone coordinate transformation functions."""

    def test_transform_points_pixel_to_cm(self):
        """Test pixel-to-cm transformation function."""
        matrix = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        points_pixel = np.array([[100.0, 200.0]])
        expected_cm = np.array([[50.0, 100.0]])

        cm_points = transform_points_pixel_to_cm(points_pixel, matrix)
        np.testing.assert_allclose(cm_points, expected_cm, rtol=1e-6)

    def test_transform_points_cm_to_pixel(self):
        """Test cm-to-pixel transformation function."""
        matrix = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        points_cm = np.array([[50.0, 100.0]])
        expected_pixel = np.array([[100.0, 200.0]])

        pixel_points = transform_points_cm_to_pixel(points_cm, matrix)
        np.testing.assert_allclose(pixel_points, expected_pixel, rtol=1e-6)

    def test_check_arena_bounds_function(self):
        """Test standalone arena bounds checking function."""
        bounds = (0.0, 0.0, 200.0, 150.0)

        inside_points = np.array([[100.0, 75.0]])
        outside_points = np.array([[250.0, 75.0]])

        assert check_arena_bounds(inside_points, bounds).all()
        assert not check_arena_bounds(outside_points, bounds).any()