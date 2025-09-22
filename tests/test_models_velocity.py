"""Tests for velocity pseudo-measurements."""

import numpy as np
import pytest
import jax.numpy as jnp
from hypothesis import given, strategies as st, settings

from trodestrack.models.velocity import (
    velocity_measurement,
    compute_velocity_jacobian,
    create_velocity_noise,
    estimate_velocity_from_positions,
    velocity_pseudo_measurement_update,
    should_use_velocity_constraint,
)
from trodestrack.models.state import State2D, state_to_array


class TestVelocityMeasurement:
    """Test velocity measurement model."""

    def test_velocity_measurement_basic(self):
        """Test basic velocity measurement extraction."""
        state = State2D(
            x=10.0, y=20.0, vx=5.0, vy=-3.0, theta=0.5,
            b_gz=0.01, b_ax=0.02, b_ay=0.01
        )

        # Expected measurement is just the velocity
        expected = jnp.array([5.0, -3.0])
        measured = velocity_measurement(state)

        np.testing.assert_allclose(measured, expected, rtol=1e-15)

    def test_velocity_jacobian(self):
        """Test velocity measurement Jacobian."""
        state_array = jnp.array([10.0, 20.0, 5.0, -3.0, 0.5, 0.01, 0.02, 0.01])

        H = compute_velocity_jacobian(state_array)

        # Should be 2x8 matrix
        assert H.shape == (2, 8)

        # Velocity measurement depends only on velocity states
        expected_H = jnp.zeros((2, 8))
        expected_H = expected_H.at[0, 2].set(1.0)  # ∂z_vx/∂vx = 1
        expected_H = expected_H.at[1, 3].set(1.0)  # ∂z_vy/∂vy = 1

        np.testing.assert_allclose(H, expected_H, rtol=1e-15)

    @given(
        vx=st.floats(-100, 100, allow_subnormal=False),
        vy=st.floats(-100, 100, allow_subnormal=False),
    )
    def test_velocity_measurement_property(self, vx, vy):
        """Property test: velocity measurement extracts velocity."""
        state = State2D(
            x=0.0, y=0.0, vx=vx, vy=vy, theta=0.0,
            b_gz=0.0, b_ax=0.0, b_ay=0.0
        )

        measured = velocity_measurement(state)

        np.testing.assert_allclose(measured, [vx, vy], rtol=1e-15)


class TestVelocityEstimation:
    """Test velocity estimation from position sequences."""

    def test_estimate_velocity_constant_motion(self):
        """Test velocity estimation for constant motion."""
        # Constant velocity: 10 cm/s in x, -5 cm/s in y
        dt = 0.033  # 30 Hz
        positions = jnp.array([
            [0.0, 0.0],
            [10.0 * dt, -5.0 * dt],
            [20.0 * dt, -10.0 * dt],
        ])
        timestamps = jnp.array([0.0, dt, 2 * dt])

        estimated_vel = estimate_velocity_from_positions(positions, timestamps)

        expected_vel = jnp.array([10.0, -5.0])
        np.testing.assert_allclose(estimated_vel, expected_vel, rtol=1e-10)

    def test_estimate_velocity_noisy_positions(self):
        """Test velocity estimation with noisy positions."""
        dt = 0.1
        # True velocity: [2, 3] cm/s
        true_positions = jnp.array([
            [0.0, 0.0],
            [0.2, 0.3],
            [0.4, 0.6],
            [0.6, 0.9],
            [0.8, 1.2],
        ])

        # Add small amount of noise
        noise = jnp.array([
            [0.01, -0.01],
            [0.02, 0.01],
            [-0.01, 0.02],
            [0.01, -0.01],
            [-0.02, 0.01],
        ])
        noisy_positions = true_positions + noise
        timestamps = jnp.arange(5) * dt

        estimated_vel = estimate_velocity_from_positions(noisy_positions, timestamps)

        # Should be close to true velocity despite noise
        expected_vel = jnp.array([2.0, 3.0])
        np.testing.assert_allclose(estimated_vel, expected_vel, rtol=0.1)

    def test_estimate_velocity_two_points(self):
        """Test velocity estimation with minimum two points."""
        positions = jnp.array([[0.0, 0.0], [1.0, 2.0]])
        timestamps = jnp.array([0.0, 0.5])

        estimated_vel = estimate_velocity_from_positions(positions, timestamps)

        # Should be simple finite difference
        expected_vel = jnp.array([2.0, 4.0])  # (1-0)/(0.5-0) = 2, (2-0)/(0.5-0) = 4
        np.testing.assert_allclose(estimated_vel, expected_vel, rtol=1e-12)

    @given(
        dt=st.floats(0.01, 0.2, allow_subnormal=False),
        vx=st.floats(-50, 50, allow_subnormal=False, allow_infinity=False),
        vy=st.floats(-50, 50, allow_subnormal=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_estimate_velocity_property(self, dt, vx, vy):
        """Property test: perfect constant motion should recover exact velocity."""
        # Generate perfect constant motion
        n_points = 5
        timestamps = jnp.arange(n_points) * dt
        positions = jnp.column_stack([vx * timestamps, vy * timestamps])

        estimated_vel = estimate_velocity_from_positions(positions, timestamps)

        np.testing.assert_allclose(estimated_vel, [vx, vy], rtol=1e-10, atol=1e-12)


class TestVelocityNoise:
    """Test velocity measurement noise computation."""

    def test_create_velocity_noise_basic(self):
        """Test basic velocity noise creation."""
        velocity_noise_std = 5.0  # cm/s
        confidence = 0.9

        R = create_velocity_noise(velocity_noise_std, confidence)

        # Should be 2x2 diagonal matrix
        assert R.shape == (2, 2)
        assert jnp.allclose(R, jnp.diag(jnp.diag(R)))

        # Noise should be scaled by confidence
        expected_var = (velocity_noise_std / confidence) ** 2
        np.testing.assert_allclose(jnp.diag(R), [expected_var, expected_var], rtol=1e-12)

    def test_create_velocity_noise_confidence_scaling(self):
        """Test velocity noise confidence scaling."""
        velocity_noise_std = 1.0
        confidence_high = 0.95
        confidence_low = 0.5

        R_high = create_velocity_noise(velocity_noise_std, confidence_high)
        R_low = create_velocity_noise(velocity_noise_std, confidence_low)

        # Lower confidence should result in higher noise
        assert jnp.all(jnp.diag(R_low) > jnp.diag(R_high))

        # Ratio should match inverse confidence ratio squared
        expected_ratio = (confidence_low / confidence_high) ** 2
        actual_ratio = jnp.diag(R_high)[0] / jnp.diag(R_low)[0]
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-12)


class TestVelocityConstraints:
    """Test velocity constraint logic."""

    def test_should_use_velocity_constraint_good_conditions(self):
        """Test velocity constraint decision with good conditions."""
        avg_confidence = 0.9
        position_residual_norm = 1.0  # Small residual
        min_confidence = 0.8
        max_position_residual = 2.0

        should_use = should_use_velocity_constraint(
            avg_confidence, position_residual_norm, min_confidence, max_position_residual
        )

        assert should_use

    def test_should_use_velocity_constraint_low_confidence(self):
        """Test velocity constraint rejected due to low confidence."""
        avg_confidence = 0.7  # Below threshold
        position_residual_norm = 1.0
        min_confidence = 0.8
        max_position_residual = 2.0

        should_use = should_use_velocity_constraint(
            avg_confidence, position_residual_norm, min_confidence, max_position_residual
        )

        assert not should_use

    def test_should_use_velocity_constraint_large_residual(self):
        """Test velocity constraint rejected due to large position residual."""
        avg_confidence = 0.9
        position_residual_norm = 3.0  # Above threshold
        min_confidence = 0.8
        max_position_residual = 2.0

        should_use = should_use_velocity_constraint(
            avg_confidence, position_residual_norm, min_confidence, max_position_residual
        )

        assert not should_use

    @given(
        avg_confidence=st.floats(0.5, 1.0, allow_subnormal=False),
        position_residual=st.floats(0.1, 5.0, allow_subnormal=False),
    )
    def test_should_use_velocity_constraint_property(self, avg_confidence, position_residual):
        """Property test: velocity constraint logic should be consistent."""
        min_confidence = 0.8
        max_residual = 2.0

        should_use = should_use_velocity_constraint(
            avg_confidence, position_residual, min_confidence, max_residual
        )

        # Should use only if both conditions are met
        expected = (avg_confidence >= min_confidence) and (position_residual <= max_residual)
        assert should_use == expected


class TestVelocityPseudoMeasurementUpdate:
    """Test velocity pseudo-measurement update functionality."""

    def test_velocity_pseudo_measurement_update_simple(self):
        """Test velocity pseudo-measurement with simple case."""
        # Current state with some velocity
        state = State2D(
            x=10.0, y=20.0, vx=5.0, vy=3.0, theta=0.0,
            b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        state_cov = jnp.eye(8) * 0.1  # Small covariance

        # Observed velocity from recent positions
        observed_velocity = jnp.array([4.8, 3.2])  # Close to state velocity
        velocity_noise = jnp.eye(2) * 0.01  # Low noise

        updated_state, updated_cov = velocity_pseudo_measurement_update(
            state, state_cov, observed_velocity, velocity_noise
        )

        # Updated velocity should be closer to observed velocity
        # (Kalman gain will blend state and measurement)
        assert abs(updated_state.vx - observed_velocity[0]) < abs(state.vx - observed_velocity[0])
        assert abs(updated_state.vy - observed_velocity[1]) < abs(state.vy - observed_velocity[1])

        # Updated covariance should be smaller (more certain)
        assert jnp.all(jnp.diag(updated_cov)[2:4] <= jnp.diag(state_cov)[2:4])

    def test_velocity_pseudo_measurement_update_no_change_positions(self):
        """Test that position and other states are unchanged."""
        state = State2D(
            x=10.0, y=20.0, vx=5.0, vy=3.0, theta=1.5,
            b_gz=0.01, b_ax=0.02, b_ay=0.01
        )
        state_cov = jnp.eye(8) * 0.1

        observed_velocity = jnp.array([4.0, 2.0])
        velocity_noise = jnp.eye(2) * 0.1

        updated_state, updated_cov = velocity_pseudo_measurement_update(
            state, state_cov, observed_velocity, velocity_noise
        )

        # Position, heading, and biases should be unchanged
        assert updated_state.x == state.x
        assert updated_state.y == state.y
        assert updated_state.theta == state.theta
        assert updated_state.b_gz == state.b_gz
        assert updated_state.b_ax == state.b_ax
        assert updated_state.b_ay == state.b_ay

    def test_velocity_pseudo_measurement_high_noise(self):
        """Test velocity update with high measurement noise."""
        state = State2D(
            x=0.0, y=0.0, vx=10.0, vy=5.0, theta=0.0,
            b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        state_cov = jnp.eye(8) * 0.01  # Low state uncertainty

        observed_velocity = jnp.array([0.0, 0.0])  # Very different from state
        velocity_noise = jnp.eye(2) * 100.0  # High measurement noise

        updated_state, updated_cov = velocity_pseudo_measurement_update(
            state, state_cov, observed_velocity, velocity_noise
        )

        # With high measurement noise, state should change very little
        np.testing.assert_allclose([updated_state.vx, updated_state.vy], [10.0, 5.0], rtol=0.1)