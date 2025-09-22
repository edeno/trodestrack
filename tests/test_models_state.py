"""Tests for state representation and utilities."""

import numpy as np
import pytest
import jax.numpy as jnp
from hypothesis import given, strategies as st

from trodestrack.models.state import (
    State2D,
    create_initial_state,
    state_to_array,
    array_to_state,
    STATE_DIM,
)


class TestState2D:
    """Test the 2D state representation."""

    def test_state_dimension(self):
        """State should be 8-dimensional as per PRD."""
        assert STATE_DIM == 8

    def test_state_construction(self):
        """Test basic state construction."""
        state = State2D(
            x=1.0,
            y=2.0,
            vx=0.5,
            vy=-0.3,
            theta=0.785,  # 45 degrees
            b_gz=0.01,
            b_ax=0.02,
            b_ay=-0.01,
        )

        assert state.x == 1.0
        assert state.y == 2.0
        assert state.vx == 0.5
        assert state.vy == -0.3
        assert state.theta == 0.785
        assert state.b_gz == 0.01
        assert state.b_ax == 0.02
        assert state.b_ay == -0.01

    def test_state_array_conversion(self):
        """Test conversion between state and array representations."""
        state = State2D(
            x=1.0, y=2.0, vx=0.5, vy=-0.3,
            theta=0.785, b_gz=0.01, b_ax=0.02, b_ay=-0.01
        )

        # Convert to array
        arr = state_to_array(state)
        assert arr.shape == (8,)
        np.testing.assert_allclose(
            arr, [1.0, 2.0, 0.5, -0.3, 0.785, 0.01, 0.02, -0.01], rtol=1e-15
        )

        # Convert back to state
        recovered = array_to_state(arr)
        assert recovered.x == state.x
        assert recovered.y == state.y
        assert recovered.vx == state.vx
        assert recovered.vy == state.vy
        assert recovered.theta == state.theta
        assert recovered.b_gz == state.b_gz
        assert recovered.b_ax == state.b_ax
        assert recovered.b_ay == state.b_ay

    @given(
        x=st.floats(-10, 10, allow_subnormal=False),
        y=st.floats(-10, 10, allow_subnormal=False),
        vx=st.floats(-5, 5, allow_subnormal=False),
        vy=st.floats(-5, 5, allow_subnormal=False),
        theta=st.floats(-np.pi, np.pi, allow_subnormal=False),
        b_gz=st.floats(-0.1, 0.1, allow_subnormal=False),
        b_ax=st.floats(-0.1, 0.1, allow_subnormal=False),
        b_ay=st.floats(-0.1, 0.1, allow_subnormal=False),
    )
    def test_state_array_roundtrip(self, x, y, vx, vy, theta, b_gz, b_ax, b_ay):
        """Property test: state -> array -> state roundtrip preserves values."""
        original = State2D(
            x=x, y=y, vx=vx, vy=vy,
            theta=theta, b_gz=b_gz, b_ax=b_ax, b_ay=b_ay
        )

        arr = state_to_array(original)
        recovered = array_to_state(arr)

        np.testing.assert_allclose([recovered.x, recovered.y, recovered.vx, recovered.vy,
                                   recovered.theta, recovered.b_gz, recovered.b_ax, recovered.b_ay],
                                  [original.x, original.y, original.vx, original.vy,
                                   original.theta, original.b_gz, original.b_ax, original.b_ay],
                                  rtol=1e-14, atol=1e-14)

    def test_jax_compatibility(self):
        """Test that state arrays work with JAX."""
        state = State2D(x=1.0, y=2.0, vx=0.5, vy=-0.3,
                       theta=0.785, b_gz=0.01, b_ax=0.02, b_ay=-0.01)

        arr = state_to_array(state)
        jax_arr = jnp.array(arr)

        # Should be able to do JAX operations
        doubled = 2 * jax_arr
        assert doubled.shape == (8,)
        np.testing.assert_array_equal(doubled, 2 * arr)


class TestInitialState:
    """Test initial state estimation utilities."""

    def test_create_initial_state_basic(self):
        """Test basic initial state creation."""
        # Mock some early video frames
        positions = jnp.array([[100.0, 200.0], [102.0, 201.0], [104.0, 202.0]])  # pixels
        timestamps = jnp.array([0.0, 0.033, 0.066])  # 30 Hz
        confidences = jnp.array([0.95, 0.97, 0.93])

        # Mock LED pair for heading (front-back separation in pixels)
        led_front = jnp.array([105.0, 203.0])
        led_back = jnp.array([103.0, 201.0])

        # Mock homography (identity for simplicity)
        homography = jnp.eye(3)

        state, covariance = create_initial_state(
            positions=positions,
            timestamps=timestamps,
            confidences=confidences,
            led_front=led_front,
            led_back=led_back,
            homography=homography,
        )

        # Position should be weighted average of early frames (in cm after homography)
        weights = confidences / jnp.sum(confidences)
        expected_x = jnp.sum(positions[:, 0] * weights)
        expected_y = jnp.sum(positions[:, 1] * weights)
        np.testing.assert_allclose([state.x, state.y], [expected_x, expected_y], rtol=1e-6)

        # Velocity should be estimated from finite differences
        dt = timestamps[1] - timestamps[0]
        expected_vx = (positions[1, 0] - positions[0, 0]) / dt
        expected_vy = (positions[1, 1] - positions[0, 1]) / dt
        np.testing.assert_allclose([state.vx, state.vy], [expected_vx, expected_vy], rtol=1e-6)

        # Heading should be from LED vector
        led_vector = led_front - led_back
        expected_theta = jnp.arctan2(led_vector[1], led_vector[0])
        np.testing.assert_allclose(state.theta, expected_theta, rtol=1e-6)

        # Biases should start at zero
        assert state.b_gz == 0.0
        assert state.b_ax == 0.0
        assert state.b_ay == 0.0

        # Covariance should be positive definite
        assert covariance.shape == (8, 8)
        eigenvals = jnp.linalg.eigvals(covariance)
        assert jnp.all(eigenvals > 0), "Covariance should be positive definite"

    def test_create_initial_state_missing_leds(self):
        """Test fallback when LED heading is unavailable."""
        positions = jnp.array([[100.0, 200.0], [102.0, 201.0], [104.0, 202.0]])
        timestamps = jnp.array([0.0, 0.033, 0.066])
        confidences = jnp.array([0.95, 0.97, 0.93])
        homography = jnp.eye(3)

        # No LED data provided
        state, covariance = create_initial_state(
            positions=positions,
            timestamps=timestamps,
            confidences=confidences,
            led_front=None,
            led_back=None,
            homography=homography,
        )

        # Should fall back to velocity heading
        dt = timestamps[1] - timestamps[0]
        vel_x = (positions[1, 0] - positions[0, 0]) / dt
        vel_y = (positions[1, 1] - positions[0, 1]) / dt
        expected_theta = jnp.arctan2(vel_y, vel_x)

        np.testing.assert_allclose(state.theta, expected_theta, rtol=1e-6)