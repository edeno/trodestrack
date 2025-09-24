"""Tests for dynamics models and prediction step."""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from trodestrack.models.dynamics import (
    compute_process_noise,
    compute_state_jacobian,
    predict_state,
)
from trodestrack.models.state import State2D


class TestDynamics:
    """Test the dynamics model for state prediction."""

    def test_predict_state_no_motion(self):
        """Test prediction when no motion occurs."""
        # Initial state at rest
        state = State2D(x=1.0, y=2.0, vx=0.0, vy=0.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0)

        # No IMU data (zero acceleration, zero angular velocity)
        dt = 0.033  # 30 Hz frame rate
        accel = jnp.zeros(2)  # No acceleration
        gyro = jnp.array([0.0])  # No rotation
        velocity_damping = 0.0  # No damping

        predicted = predict_state(state, dt, accel, gyro, velocity_damping)

        # Should remain at same position and velocity
        np.testing.assert_allclose([predicted.x, predicted.y], [1.0, 2.0], rtol=1e-14)
        np.testing.assert_allclose([predicted.vx, predicted.vy], [0.0, 0.0], rtol=1e-14)
        np.testing.assert_allclose(predicted.theta, 0.0, rtol=1e-14)

        # Biases should remain unchanged (random walk with no noise added here)
        np.testing.assert_allclose(
            [predicted.b_gz, predicted.b_ax, predicted.b_ay], [0.0, 0.0, 0.0], rtol=1e-14
        )

    def test_predict_state_constant_velocity(self):
        """Test prediction with constant velocity."""
        state = State2D(
            x=0.0,
            y=0.0,
            vx=10.0,
            vy=5.0,
            theta=0.0,  # 10 cm/s in x, 5 cm/s in y
            b_gz=0.0,
            b_ax=0.0,
            b_ay=0.0,
        )

        dt = 0.1  # 100 ms
        accel = jnp.zeros(2)  # No acceleration
        gyro = jnp.array([0.0])  # No rotation
        velocity_damping = 0.0

        predicted = predict_state(state, dt, accel, gyro, velocity_damping)

        # Position should update based on velocity
        expected_x = 0.0 + 10.0 * 0.1  # 1.0 cm
        expected_y = 0.0 + 5.0 * 0.1  # 0.5 cm
        np.testing.assert_allclose([predicted.x, predicted.y], [expected_x, expected_y], rtol=1e-12)

        # Velocity should remain unchanged
        np.testing.assert_allclose([predicted.vx, predicted.vy], [10.0, 5.0], rtol=1e-14)

    def test_predict_state_with_acceleration(self):
        """Test prediction with acceleration input."""
        state = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0)

        dt = 0.1
        accel = jnp.array([1.0, 0.5])  # 1 m/s² in x, 0.5 m/s² in y
        gyro = jnp.array([0.0])
        velocity_damping = 0.0

        predicted = predict_state(state, dt, accel, gyro, velocity_damping)

        # Convert acceleration to cm/s²
        accel_cm = accel * 100.0  # 100 cm/s² in x, 50 cm/s² in y

        # Velocity should update: v = v0 + a*dt
        expected_vx = 0.0 + accel_cm[0] * dt  # 10 cm/s
        expected_vy = 0.0 + accel_cm[1] * dt  # 5 cm/s
        np.testing.assert_allclose(
            [predicted.vx, predicted.vy], [expected_vx, expected_vy], rtol=1e-12
        )

        # Position should update: x = x0 + v0*dt + 0.5*a*dt²
        expected_x = 0.0 + 0.0 * dt + 0.5 * accel_cm[0] * dt**2  # 0.5 cm
        expected_y = 0.0 + 0.0 * dt + 0.5 * accel_cm[1] * dt**2  # 0.25 cm
        np.testing.assert_allclose([predicted.x, predicted.y], [expected_x, expected_y], rtol=1e-12)

    def test_predict_state_with_rotation(self):
        """Test prediction with angular velocity."""
        state = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0)

        dt = 0.1
        accel = jnp.zeros(2)
        gyro = jnp.array([np.pi])  # π rad/s angular velocity
        velocity_damping = 0.0

        predicted = predict_state(state, dt, accel, gyro, velocity_damping)

        # Heading should update: θ = θ0 + ω*dt
        expected_theta = 0.0 + np.pi * dt  # 0.1π radians
        np.testing.assert_allclose(predicted.theta, expected_theta, rtol=1e-12)

    def test_predict_state_with_bias_compensation(self):
        """Test prediction with IMU bias compensation."""
        state = State2D(
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            theta=0.0,
            b_gz=0.1,
            b_ax=0.05,
            b_ay=0.02,  # Non-zero biases
        )

        dt = 0.1
        accel = jnp.array([1.0, 0.5])  # Raw acceleration
        gyro = jnp.array([1.0])  # Raw angular velocity
        velocity_damping = 0.0

        predicted = predict_state(state, dt, accel, gyro, velocity_damping)

        # Bias-corrected measurements
        corrected_gyro = gyro[0] - state.b_gz

        # Check angular velocity correction
        expected_theta = 0.0 + corrected_gyro * dt
        np.testing.assert_allclose(predicted.theta, expected_theta, rtol=1e-12)

    def test_predict_state_with_velocity_damping(self):
        """Test prediction with velocity damping."""
        state = State2D(x=0.0, y=0.0, vx=10.0, vy=5.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0)

        dt = 0.1
        accel = jnp.zeros(2)
        gyro = jnp.array([0.0])
        velocity_damping = 0.1  # 10% damping per second

        predicted = predict_state(state, dt, accel, gyro, velocity_damping)

        # Velocity should be damped: v = v * (1 - λ*dt)
        damping_factor = 1.0 - velocity_damping * dt  # 0.99
        expected_vx = 10.0 * damping_factor  # 9.9 cm/s
        expected_vy = 5.0 * damping_factor  # 4.95 cm/s
        np.testing.assert_allclose(
            [predicted.vx, predicted.vy], [expected_vx, expected_vy], rtol=1e-12
        )

    @given(
        dt=st.floats(0.001, 0.1, allow_subnormal=False),
        vx=st.floats(-100, 100, allow_subnormal=False),
        vy=st.floats(-100, 100, allow_subnormal=False),
    )
    def test_predict_state_position_integration(self, dt, vx, vy):
        """Property test: position integration should be consistent."""
        state = State2D(x=0.0, y=0.0, vx=vx, vy=vy, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0)

        accel = jnp.zeros(2)
        gyro = jnp.array([0.0])
        velocity_damping = 0.0

        predicted = predict_state(state, dt, accel, gyro, velocity_damping)

        # Position should be: x = x0 + vx*dt
        expected_x = 0.0 + vx * dt
        expected_y = 0.0 + vy * dt

        np.testing.assert_allclose(
            [predicted.x, predicted.y], [expected_x, expected_y], rtol=1e-10, atol=1e-12
        )

    def test_damping_stability_check(self):
        """Test that damping stability check catches unstable parameters."""
        state = State2D(x=0.0, y=0.0, vx=10.0, vy=5.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0)
        accel = jnp.zeros(2)
        gyro = jnp.array([0.0])

        # Test stable damping (should work)
        dt = 0.1  # 100ms
        stable_damping = 5.0  # λ = 5 < 1/0.1 = 10
        predicted = predict_state(state, dt, accel, gyro, stable_damping)
        assert predicted.vx < state.vx  # Velocity should decrease

        # Test unstable damping (should raise error)
        unstable_damping = 15.0  # λ = 15 > 1/0.1 = 10
        with pytest.raises(ValueError, match="too large for dt"):
            predict_state(state, dt, accel, gyro, unstable_damping)

        # Test negative damping (should raise error)
        with pytest.raises(ValueError, match="must be non-negative"):
            predict_state(state, dt, accel, gyro, -1.0)


class TestJacobians:
    """Test Jacobian computation for EKF."""

    def test_compute_state_jacobian_identity_case(self):
        """Test Jacobian computation for simple case."""
        state_array = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Zero state
        dt = 0.1
        accel = jnp.zeros(2)
        gyro = jnp.array([0.0])
        velocity_damping = 0.0

        F = compute_state_jacobian(state_array, dt, accel, gyro, velocity_damping)

        # Should be 8x8 matrix
        assert F.shape == (8, 8)

        # For zero inputs, many elements should be identity-like
        # Position depends on velocity: ∂x/∂vx = dt, ∂y/∂vy = dt
        np.testing.assert_allclose(F[0, 2], dt, rtol=1e-12)  # ∂x/∂vx
        np.testing.assert_allclose(F[1, 3], dt, rtol=1e-12)  # ∂y/∂vy

        # Heading integrates gyro: ∂θ/∂θ = 1
        np.testing.assert_allclose(F[4, 4], 1.0, rtol=1e-12)

    def test_compute_state_jacobian_dimensions(self):
        """Test that Jacobian has correct dimensions."""
        state_array = jnp.ones(8)
        dt = 0.033
        accel = jnp.array([0.1, 0.2])
        gyro = jnp.array([0.05])
        velocity_damping = 0.01

        F = compute_state_jacobian(state_array, dt, accel, gyro, velocity_damping)

        assert F.shape == (8, 8)
        assert jnp.isfinite(F).all(), "Jacobian should contain only finite values"


class TestProcessNoise:
    """Test process noise computation."""

    def test_compute_process_noise_dimensions(self):
        """Test process noise matrix dimensions and structure."""
        dt = 0.033
        accel_noise_std = 0.1  # m/s²
        gyro_noise_std = 0.01  # rad/s
        bias_drift_std = 0.001  # bias random walk

        Q = compute_process_noise(dt, accel_noise_std, gyro_noise_std, bias_drift_std)

        assert Q.shape == (8, 8)
        assert jnp.isfinite(Q).all(), "Process noise should contain only finite values"

        # Should be positive semi-definite
        eigenvals = jnp.linalg.eigvals(Q)
        assert jnp.all(eigenvals >= -1e-10), "Process noise should be positive semi-definite"

    def test_compute_process_noise_scaling(self):
        """Test that process noise scales correctly with dt."""
        dt1 = 0.01
        dt2 = 0.02
        accel_noise_std = 0.1
        gyro_noise_std = 0.01
        bias_drift_std = 0.001

        Q1 = compute_process_noise(dt1, accel_noise_std, gyro_noise_std, bias_drift_std)
        Q2 = compute_process_noise(dt2, accel_noise_std, gyro_noise_std, bias_drift_std)

        # Process noise should scale with dt (for continuous-time white noise)
        # Some elements scale with dt, others with dt²
        # Bias covariances scale with dt
        np.testing.assert_allclose(Q2[5, 5], Q1[5, 5] * 2, rtol=1e-10)  # Gyro bias
        np.testing.assert_allclose(Q2[6, 6], Q1[6, 6] * 2, rtol=1e-10)  # Accel bias x
        np.testing.assert_allclose(Q2[7, 7], Q1[7, 7] * 2, rtol=1e-10)  # Accel bias y
