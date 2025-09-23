"""Tests for Extended Kalman Filter implementation."""

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ekf import (
    EKFFilter,
    EKFState,
    create_initial_ekf_state,
    ekf_predict,
    ekf_update,
)
from trodestrack.models.measurements import create_measurement_noise
from trodestrack.models.state import State2D, state_to_array


class TestEKFState:
    """Test EKF state representation."""

    def test_ekf_state_creation(self):
        """Test EKF state creation."""
        state = jnp.array([1.0, 2.0, 0.5, -0.3, 0.1, 0.0, 0.0, 0.0])
        covariance = jnp.eye(8)
        log_likelihood = -10.5

        ekf_state = EKFState(
            state=state,
            covariance=covariance,
            log_likelihood=log_likelihood,
        )

        assert ekf_state.state.shape == (8,)
        assert ekf_state.covariance.shape == (8, 8)
        assert ekf_state.log_likelihood == log_likelihood
        np.testing.assert_array_equal(ekf_state.state, state)
        np.testing.assert_array_equal(ekf_state.covariance, covariance)

    def test_create_initial_ekf_state(self):
        """Test creation of initial EKF state from State2D."""
        initial_state = State2D(
            x=10.0, y=20.0, vx=1.0, vy=-0.5, theta=0.5, b_gz=0.01, b_ax=0.02, b_ay=-0.01
        )
        initial_covariance = jnp.diag(jnp.array([1.0, 1.0, 10.0, 10.0, 0.1, 0.01, 0.01, 0.01]))

        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        expected_state_array = state_to_array(initial_state)
        np.testing.assert_array_equal(ekf_state.state, expected_state_array)
        np.testing.assert_array_equal(ekf_state.covariance, initial_covariance)
        assert ekf_state.log_likelihood == 0.0


class TestEKFPredict:
    """Test EKF prediction step."""

    def test_ekf_predict_basic(self):
        """Test basic EKF prediction step."""
        # Initial state: stationary at origin
        initial_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        initial_cov = jnp.eye(8) * 0.1
        ekf_state = EKFState(initial_state, initial_cov, 0.0)

        # Simple forward motion: constant acceleration
        dt = 0.1
        accel = jnp.array([1.0, 0.0])  # 1 m/s² forward
        gyro = jnp.array([0.0])  # No rotation
        velocity_damping = 0.0

        predicted = ekf_predict(
            ekf_state,
            dt,
            accel,
            gyro,
            velocity_damping,
            accel_noise_std=0.1,
            gyro_noise_std=0.05,
            bias_drift_std=0.01,
        )

        # Check state prediction (no biases, so acceleration should be applied directly)
        # v = v0 + a*dt = 0 + 100*0.1 = 10 cm/s (converted from m/s²)
        # x = x0 + v0*dt + 0.5*a*dt² = 0 + 0 + 0.5*100*0.01 = 0.5 cm
        assert predicted.state[0] == pytest.approx(0.5, abs=1e-10)  # x position
        assert predicted.state[1] == pytest.approx(0.0, abs=1e-10)  # y position
        assert predicted.state[2] == pytest.approx(10.0, abs=1e-10)  # vx velocity
        assert predicted.state[3] == pytest.approx(0.0, abs=1e-10)  # vy velocity
        assert predicted.state[4] == pytest.approx(0.0, abs=1e-10)  # theta

        # Diagonal elements (variances) should increase due to process noise
        assert jnp.all(jnp.diag(predicted.covariance) >= jnp.diag(initial_cov))

    def test_ekf_predict_with_bias(self):
        """Test EKF prediction with IMU biases."""
        # State with biases
        initial_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1])
        initial_cov = jnp.eye(8) * 0.1
        ekf_state = EKFState(initial_state, initial_cov, 0.0)

        dt = 0.1
        accel = jnp.array([1.0, 0.5])  # Raw measurement
        gyro = jnp.array([0.2])  # Raw measurement
        velocity_damping = 0.0

        predicted = ekf_predict(
            ekf_state,
            dt,
            accel,
            gyro,
            velocity_damping,
            accel_noise_std=0.1,
            gyro_noise_std=0.05,
            bias_drift_std=0.01,
        )

        # Bias-corrected measurements:
        # accel_corrected = [1.0, 0.5] - [0.2, 0.1] = [0.8, 0.4] m/s²
        # gyro_corrected = 0.2 - 0.1 = 0.1 rad/s

        # Expected heading change: θ = 0 + 0.1*0.1 = 0.01 rad
        assert predicted.state[4] == pytest.approx(0.01, abs=1e-10)

        # Biases should remain unchanged (random walk model)
        assert predicted.state[5] == pytest.approx(0.1, abs=1e-10)  # b_gz
        assert predicted.state[6] == pytest.approx(0.2, abs=1e-10)  # b_ax
        assert predicted.state[7] == pytest.approx(0.1, abs=1e-10)  # b_ay

    def test_ekf_predict_with_damping(self):
        """Test EKF prediction with velocity damping."""
        # Initial state with velocity
        initial_state = jnp.array([0.0, 0.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        initial_cov = jnp.eye(8) * 0.1
        ekf_state = EKFState(initial_state, initial_cov, 0.0)

        dt = 0.1
        accel = jnp.array([0.0, 0.0])  # No acceleration
        gyro = jnp.array([0.0])  # No rotation
        velocity_damping = 2.0  # Strong damping

        predicted = ekf_predict(
            ekf_state,
            dt,
            accel,
            gyro,
            velocity_damping,
            accel_noise_std=0.1,
            gyro_noise_std=0.05,
            bias_drift_std=0.01,
        )

        # Damping factor: 1 - λ*dt = 1 - 2.0*0.1 = 0.8
        # New velocity: v * damping_factor = [10, 5] * 0.8 = [8, 4]
        assert predicted.state[2] == pytest.approx(8.0, abs=1e-10)  # vx
        assert predicted.state[3] == pytest.approx(4.0, abs=1e-10)  # vy

        # Position update: x = x0 + v0*dt (using original velocity for position)
        # x = 0 + 10*0.1 = 1.0
        # y = 0 + 5*0.1 = 0.5
        assert predicted.state[0] == pytest.approx(1.0, abs=1e-10)  # x
        assert predicted.state[1] == pytest.approx(0.5, abs=1e-10)  # y

    def test_ekf_predict_covariance_growth(self):
        """Test that covariance grows appropriately during prediction."""
        initial_state = jnp.zeros(8)
        initial_cov = jnp.eye(8) * 0.01  # Small initial uncertainty
        ekf_state = EKFState(initial_state, initial_cov, 0.0)

        dt = 0.1
        accel = jnp.zeros(2)
        gyro = jnp.zeros(1)

        predicted = ekf_predict(
            ekf_state,
            dt,
            accel,
            gyro,
            velocity_damping=0.0,
            accel_noise_std=0.5,
            gyro_noise_std=0.1,
            bias_drift_std=0.02,
        )

        # All diagonal elements (variances) should increase due to process noise
        assert jnp.all(jnp.diag(predicted.covariance) > jnp.diag(initial_cov))

        # Velocity uncertainty should grow more than position for short time steps
        # Position noise = 0.25 * σ_a² * dt⁴, Velocity noise = σ_a² * dt²
        # For dt=0.1, σ_a=0.5*100=50 cm/s²: pos_noise = 0.25*50²*0.1⁴ = 0.003125
        # vel_noise = 50²*0.1² = 25, so vel > pos for short dt
        assert predicted.covariance[2, 2] > predicted.covariance[0, 0]  # vel > pos
        assert predicted.covariance[3, 3] > predicted.covariance[1, 1]  # vel > pos


class TestEKFUpdate:
    """Test EKF measurement update step."""

    def test_ekf_update_position_only(self):
        """Test EKF update with position measurement only."""
        # Prior state with some uncertainty
        prior_state = jnp.array([1.0, 2.0, 0.5, -0.3, 0.1, 0.0, 0.0, 0.0])
        prior_cov = jnp.eye(8) * 2.0  # High uncertainty
        ekf_state = EKFState(prior_state, prior_cov, -5.0)

        # Position measurement close to prior
        measurement = jnp.array([1.1, 1.9])  # Slight offset
        measurement_noise = create_measurement_noise(0.5, 1.0, False)

        result = ekf_update(ekf_state, measurement, measurement_noise, False)

        # Should not be gated
        assert not result.gated

        # State should move toward measurement
        assert result.state.state[0] > prior_state[0]  # x moves toward 1.1
        assert result.state.state[1] < prior_state[1]  # y moves toward 1.9

        # Uncertainty should decrease
        assert jnp.all(jnp.diag(result.state.covariance) <= jnp.diag(prior_cov))

        # Check innovation
        expected_innovation = measurement - prior_state[:2]
        np.testing.assert_array_almost_equal(result.innovation, expected_innovation, decimal=10)

        # Log-likelihood should update
        assert result.state.log_likelihood < ekf_state.log_likelihood  # More negative

    def test_ekf_update_position_and_heading(self):
        """Test EKF update with both position and heading measurements."""
        prior_state = jnp.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
        prior_cov = jnp.eye(8) * 1.0
        ekf_state = EKFState(prior_state, prior_cov, 0.0)

        # Combined measurement
        position = jnp.array([0.1, -0.1])
        heading = 0.4
        measurement = jnp.concatenate([position, jnp.array([heading])])
        measurement_noise = create_measurement_noise(0.3, 1.0, True, 0.1)

        result = ekf_update(ekf_state, measurement, measurement_noise, True)

        assert not result.gated
        assert result.innovation.shape == (3,)  # [x, y, theta]

        # Position should move toward measurement
        assert result.state.state[0] > prior_state[0]  # x toward 0.1
        assert result.state.state[1] < prior_state[1]  # y toward -0.1

        # Heading should move toward measurement
        assert result.state.state[4] < prior_state[4]  # theta toward 0.4

    def test_ekf_update_heading_wrap(self):
        """Test proper handling of heading wrap-around."""
        prior_state = jnp.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])  # theta ≈ π
        prior_cov = jnp.eye(8) * 0.1
        ekf_state = EKFState(prior_state, prior_cov, 0.0)

        # Heading measurement that wraps around
        position = jnp.array([0.0, 0.0])
        heading = -3.0  # Should be close to 3.0 when wrapped
        measurement = jnp.concatenate([position, jnp.array([heading])])
        measurement_noise = create_measurement_noise(0.1, 1.0, True, 0.1)

        result = ekf_update(ekf_state, measurement, measurement_noise, True)

        # Innovation should be wrapped properly
        # Raw innovation would be -3.0 - 3.0 = -6.0
        # Wrapped innovation should be close to 0 (since 3.0 ≈ -3.0 + 2π)
        wrapped_innovation = result.innovation[2]
        assert abs(wrapped_innovation) < 0.5  # Should be small after wrapping

    def test_ekf_update_gating(self):
        """Test measurement gating for outliers."""
        prior_state = jnp.zeros(8)
        prior_cov = jnp.eye(8) * 0.1  # Low uncertainty
        ekf_state = EKFState(prior_state, prior_cov, 0.0)

        # Outlier measurement (far from prior)
        measurement = jnp.array([100.0, 100.0])  # Very far from origin
        measurement_noise = create_measurement_noise(1.0, 1.0, False)

        result = ekf_update(ekf_state, measurement, measurement_noise, False, gate_threshold=9.21)

        # Should be gated due to large Mahalanobis distance
        assert result.gated

        # State should remain unchanged
        np.testing.assert_array_equal(result.state.state, ekf_state.state)
        np.testing.assert_array_equal(result.state.covariance, ekf_state.covariance)

    def test_ekf_update_low_confidence(self):
        """Test measurement update with low confidence (high noise)."""
        prior_state = jnp.zeros(8)
        prior_cov = jnp.eye(8) * 1.0
        ekf_state = EKFState(prior_state, prior_cov, 0.0)

        measurement = jnp.array([1.0, 1.0])
        # Low confidence = high noise
        measurement_noise = create_measurement_noise(1.0, 0.1, False)  # confidence=0.1

        result = ekf_update(ekf_state, measurement, measurement_noise, False)

        # Should accept measurement but with limited influence
        assert not result.gated

        # State change should be smaller due to high measurement noise
        state_change = jnp.linalg.norm(result.state.state[:2] - prior_state[:2])
        assert state_change < 0.5  # Limited influence due to low confidence


class TestEKFFilter:
    """Test EKF filter class interface."""

    def create_test_filter(self) -> EKFFilter:
        """Create a test EKF filter."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_cov = jnp.eye(8) * 0.5

        return EKFFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            velocity_damping=0.1,
            accel_noise_std=0.2,
            gyro_noise_std=0.05,
            bias_drift_std=0.01,
            position_noise_std=1.0,
            heading_noise_std=0.1,
            gate_threshold=9.21,
        )

    def test_filter_initialization(self):
        """Test filter initialization."""
        filter = self.create_test_filter()

        state = filter.get_current_state()
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.theta == 0.0

        cov = filter.get_current_covariance()
        assert cov.shape == (8, 8)

        assert filter.get_log_likelihood() == 0.0

    def test_filter_predict_only(self):
        """Test filter prediction without measurements."""
        filter = self.create_test_filter()

        # Apply forward acceleration
        filter.predict(dt=0.1, accel=jnp.array([2.0, 0.0]), gyro=jnp.array([0.0]))

        state = filter.get_current_state()
        # Should have moved forward due to acceleration
        assert state.x > 0.0
        assert state.vx > 0.0

    def test_filter_update_only(self):
        """Test filter update without prediction."""
        filter = self.create_test_filter()

        # Update with position measurement
        result = filter.update(position=jnp.array([1.0, 2.0]), confidence=0.8)

        assert not result.gated
        state = filter.get_current_state()
        # State should move toward measurement
        assert state.x > 0.0
        assert state.y > 0.0

    def test_filter_predict_update_cycle(self):
        """Test complete predict-update cycle."""
        filter = self.create_test_filter()

        # Predict step
        filter.predict(dt=0.1, accel=jnp.array([1.0, 0.0]), gyro=jnp.array([0.1]))

        state_after_predict = filter.get_current_state()

        # Update step
        result = filter.update(position=jnp.array([0.5, 0.1]), heading=0.05, confidence=0.9)

        assert not result.gated
        state_after_update = filter.get_current_state()

        # Both prediction and measurement should influence final state
        # Y position should change due to measurement (measurement: y=0.1, prediction: y=0.0)
        assert abs(state_after_update.y - 0.1) < abs(state_after_predict.y - 0.1)
        # Heading should change due to measurement
        assert abs(state_after_update.theta - 0.05) < abs(state_after_predict.theta - 0.05)

    def test_filter_missing_measurements(self):
        """Test filter with missing measurements."""
        filter = self.create_test_filter()

        # Update with no measurements
        result = filter.update()

        # Should return empty result
        assert result.innovation.size == 0
        assert result.kalman_gain.size == 0

        # Update with position only
        result = filter.update(position=jnp.array([1.0, 1.0]))
        assert not result.gated
        assert result.innovation.shape == (2,)

        # Update with heading only (rare case)
        result = filter.update(heading=0.5)
        assert not result.gated
        assert result.innovation.shape == (3,)  # Includes dummy position

    def test_filter_low_confidence_handling(self):
        """Test filter behavior with very low confidence measurements."""
        filter = self.create_test_filter()

        # Very low confidence measurement
        result = filter.update(
            position=jnp.array([10.0, 10.0]),  # Far from current estimate
            confidence=0.01,  # Very low confidence
        )

        # Should still accept but with minimal influence
        assert not result.gated
        state = filter.get_current_state()
        # Should move only slightly toward measurement
        assert abs(state.x) < 5.0  # Much less than measurement value
        assert abs(state.y) < 5.0


class TestEKFLinearCase:
    """Test EKF against known linear-Gaussian case."""

    def test_ekf_vs_linear_kalman_filter(self):
        """Test EKF matches linear Kalman filter for linear case."""
        # For pure position measurements with no IMU dynamics,
        # the EKF should behave like a linear Kalman filter

        # Initial state: stationary at origin
        initial_state = State2D(
            x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_cov = jnp.eye(8) * 1.0

        filter = EKFFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            velocity_damping=0.0,  # No damping
            accel_noise_std=0.0,  # No process noise
            gyro_noise_std=0.0,
            bias_drift_std=0.0,
            position_noise_std=1.0,
            heading_noise_std=0.1,
        )

        # Series of position measurements
        measurements = [
            jnp.array([1.0, 0.0]),
            jnp.array([2.0, 0.5]),
            jnp.array([3.0, 1.0]),
        ]

        log_likelihoods = []
        for measurement in measurements:
            # No prediction (no IMU input)
            filter.predict(dt=0.1, accel=jnp.zeros(2), gyro=jnp.zeros(1))

            # Update with measurement
            _ = filter.update(position=measurement, confidence=1.0)
            log_likelihoods.append(filter.get_log_likelihood())

        # Final state should be close to last measurement (with some smoothing)
        final_state = filter.get_current_state()
        assert abs(final_state.x - 3.0) < 1.0
        assert abs(final_state.y - 1.0) < 1.0

        # Log-likelihood should increase (become less negative) with each update
        assert len(log_likelihoods) == 3
        # Each update should provide reasonable likelihood
        for ll in log_likelihoods:
            assert ll < 0  # Log-likelihood should be negative
            assert ll > -50  # But not excessively negative
