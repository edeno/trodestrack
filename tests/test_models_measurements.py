"""Tests for measurement models."""

import jax.numpy as jnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from trodestrack.models.measurements import (
    compute_heading_jacobian,
    compute_position_jacobian,
    create_measurement_noise,
    heading_measurement,
    position_measurement,
    validate_led_measurement,
)
from trodestrack.models.state import State2D


class TestPositionMeasurement:
    """Test position measurement model."""

    def test_position_measurement_basic(self):
        """Test basic position measurement."""
        state = State2D(
            x=100.0, y=200.0, vx=10.0, vy=5.0, theta=0.5, b_gz=0.01, b_ax=0.02, b_ay=0.01
        )

        # Expected measurement is just the position
        expected = jnp.array([100.0, 200.0])
        measured = position_measurement(state)

        np.testing.assert_allclose(measured, expected, rtol=1e-15)

    def test_position_jacobian(self):
        """Test position measurement Jacobian."""
        state_array = jnp.array([10.0, 20.0, 1.0, 2.0, 0.3, 0.01, 0.02, 0.01])

        H = compute_position_jacobian(state_array)

        # Should be 2x8 matrix
        assert H.shape == (2, 8)

        # Position measurement depends only on position states
        expected_H = jnp.zeros((2, 8))
        expected_H = expected_H.at[0, 0].set(1.0)  # ∂z_x/∂x = 1
        expected_H = expected_H.at[1, 1].set(1.0)  # ∂z_y/∂y = 1

        np.testing.assert_allclose(H, expected_H, rtol=1e-15)

    @given(
        x=st.floats(-100, 100, allow_subnormal=False),
        y=st.floats(-100, 100, allow_subnormal=False),
    )
    def test_position_measurement_property(self, x, y):
        """Property test: position measurement extracts position."""
        state = State2D(x=x, y=y, vx=0.0, vy=0.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0)

        measured = position_measurement(state)

        np.testing.assert_allclose(measured, [x, y], rtol=1e-15)


class TestHeadingMeasurement:
    """Test heading measurement model."""

    def test_heading_measurement_from_leds(self):
        """Test heading measurement from LED pair."""
        # LED positions in pixels
        led_front = jnp.array([105.0, 200.0])
        led_back = jnp.array([95.0, 200.0])

        # Expected heading (front - back vector)
        expected_heading = jnp.arctan2(0.0, 10.0)  # 0 radians (pointing right)

        measured = heading_measurement(led_front, led_back)

        np.testing.assert_allclose(measured, expected_heading, rtol=1e-12)

    def test_heading_measurement_vertical(self):
        """Test heading measurement for vertical orientation."""
        led_front = jnp.array([100.0, 210.0])  # Front LED above
        led_back = jnp.array([100.0, 190.0])  # Back LED below

        expected_heading = jnp.arctan2(20.0, 0.0)  # π/2 radians (pointing up)

        measured = heading_measurement(led_front, led_back)

        np.testing.assert_allclose(measured, expected_heading, rtol=1e-12)

    def test_heading_measurement_diagonal(self):
        """Test heading measurement for diagonal orientation."""
        led_front = jnp.array([110.0, 210.0])  # Front LED up-right
        led_back = jnp.array([90.0, 190.0])  # Back LED down-left

        # Vector from back to front: [20, 20]
        expected_heading = jnp.arctan2(20.0, 20.0)  # π/4 radians (45°)

        measured = heading_measurement(led_front, led_back)

        np.testing.assert_allclose(measured, expected_heading, rtol=1e-12)

    def test_heading_jacobian_zero_state(self):
        """Test heading measurement Jacobian at zero state."""
        state_array = jnp.zeros(8)

        H = compute_heading_jacobian(state_array)

        # Should be 1x8 matrix
        assert H.shape == (1, 8)

        # Heading measurement depends only on heading state
        expected_H = jnp.zeros((1, 8))
        expected_H = expected_H.at[0, 4].set(1.0)  # ∂z_θ/∂θ = 1

        np.testing.assert_allclose(H, expected_H, rtol=1e-15)

    @given(
        theta=st.floats(-np.pi, np.pi, allow_subnormal=False),
    )
    def test_heading_jacobian_property(self, theta):
        """Property test: heading Jacobian should be identity for heading state."""
        state_array = jnp.array([0.0, 0.0, 0.0, 0.0, theta, 0.0, 0.0, 0.0])

        H = compute_heading_jacobian(state_array)

        # Should extract heading only
        expected_H = jnp.zeros((1, 8))
        expected_H = expected_H.at[0, 4].set(1.0)

        np.testing.assert_allclose(H, expected_H, rtol=1e-15)


class TestLEDValidation:
    """Test LED measurement validation."""

    def test_validate_led_measurement_good(self):
        """Test validation of good LED measurement."""
        led_front = jnp.array([105.0, 200.0])
        led_back = jnp.array([95.0, 200.0])
        expected_spacing = 10.0  # pixels
        tolerance = 2.0  # pixels

        is_valid = validate_led_measurement(led_front, led_back, expected_spacing, tolerance)

        assert is_valid

    def test_validate_led_measurement_too_close(self):
        """Test validation rejects LEDs that are too close."""
        led_front = jnp.array([103.0, 200.0])
        led_back = jnp.array([97.0, 200.0])  # Only 6 pixels apart
        expected_spacing = 10.0
        tolerance = 2.0

        is_valid = validate_led_measurement(led_front, led_back, expected_spacing, tolerance)

        assert not is_valid

    def test_validate_led_measurement_too_far(self):
        """Test validation rejects LEDs that are too far."""
        led_front = jnp.array([108.0, 200.0])
        led_back = jnp.array([92.0, 200.0])  # 16 pixels apart
        expected_spacing = 10.0
        tolerance = 2.0

        is_valid = validate_led_measurement(led_front, led_back, expected_spacing, tolerance)

        assert not is_valid

    def test_validate_led_measurement_diagonal(self):
        """Test validation with diagonal LED arrangement."""
        # 3-4-5 triangle: 3 and 4 sides, hypotenuse is 5
        led_front = jnp.array([103.0, 204.0])
        led_back = jnp.array([100.0, 200.0])
        distance = jnp.sqrt(9 + 16)  # sqrt(3² + 4²) = 5
        expected_spacing = 5.0
        tolerance = 0.5

        is_valid = validate_led_measurement(led_front, led_back, expected_spacing, tolerance)

        assert is_valid

    @given(
        spacing=st.floats(5.0, 50.0, allow_subnormal=False),
        tolerance=st.floats(0.1, 5.0, allow_subnormal=False),
        angle=st.floats(-np.pi, np.pi, allow_subnormal=False),
    )
    def test_validate_led_measurement_property(self, spacing, tolerance, angle):
        """Property test: exact spacing should always validate."""
        # Create LEDs at exact expected spacing
        led_front = jnp.array([spacing * jnp.cos(angle), spacing * jnp.sin(angle)])
        led_back = jnp.array([0.0, 0.0])

        is_valid = validate_led_measurement(led_front, led_back, spacing, tolerance)

        assert is_valid


class TestMeasurementNoise:
    """Test measurement noise computation."""

    def test_create_measurement_noise_position_only(self):
        """Test measurement noise for position-only measurements."""
        position_noise_std = 2.0  # cm
        confidence = 0.95

        R = create_measurement_noise(
            position_noise_std=position_noise_std, confidence=confidence, has_heading=False
        )

        # Should be 2x2 for position only
        assert R.shape == (2, 2)

        # Should be diagonal
        assert jnp.allclose(R, jnp.diag(jnp.diag(R)))

        # Noise should be scaled by confidence
        expected_var = (position_noise_std / confidence) ** 2
        np.testing.assert_allclose(jnp.diag(R), [expected_var, expected_var], rtol=1e-12)

    def test_create_measurement_noise_with_heading(self):
        """Test measurement noise with heading measurement."""
        position_noise_std = 2.0
        heading_noise_std = 0.1  # radians
        confidence = 0.9

        R = create_measurement_noise(
            position_noise_std=position_noise_std,
            confidence=confidence,
            has_heading=True,
            heading_noise_std=heading_noise_std,
        )

        # Should be 3x3 for position + heading
        assert R.shape == (3, 3)

        # Should be diagonal
        assert jnp.allclose(R, jnp.diag(jnp.diag(R)))

        # Position noise scaled by confidence, heading noise constant
        expected_pos_var = (position_noise_std / confidence) ** 2
        expected_heading_var = heading_noise_std**2

        expected_diag = jnp.array([expected_pos_var, expected_pos_var, expected_heading_var])
        np.testing.assert_allclose(jnp.diag(R), expected_diag, rtol=1e-12)

    def test_create_measurement_noise_confidence_scaling(self):
        """Test that measurement noise scales correctly with confidence."""
        position_noise_std = 1.0
        confidence_high = 0.95
        confidence_low = 0.5

        R_high = create_measurement_noise(position_noise_std, confidence_high, False)
        R_low = create_measurement_noise(position_noise_std, confidence_low, False)

        # Lower confidence should result in higher noise
        assert jnp.all(jnp.diag(R_low) > jnp.diag(R_high))

        # Ratio should match inverse confidence ratio squared (higher confidence = lower noise)
        expected_ratio = (confidence_low / confidence_high) ** 2
        actual_ratio = jnp.diag(R_high)[0] / jnp.diag(R_low)[0]
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-12)
