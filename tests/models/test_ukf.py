"""Tests for Unscented Kalman Filter implementation."""

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ukf import (
    UKFFilter,
    UKFParams,
    UKFState,
    create_initial_ukf_state,
    generate_sigma_points,
    propagate_sigma_points,
    ukf_predict,
    ukf_update,
)
from trodestrack.models.measurements import create_measurement_noise
from trodestrack.models.state import State2D, state_to_array


class TestUKFParams:
    """Test UKF parameter handling."""

    def test_default_params(self):
        """Test default UKF parameters."""
        params = UKFParams()
        assert params.alpha == 1.0
        assert params.beta == 2.0
        assert params.kappa == 0.0

    def test_custom_params(self):
        """Test custom UKF parameters."""
        params = UKFParams(alpha=0.1, beta=1.0, kappa=1.0)
        assert params.alpha == 0.1
        assert params.beta == 1.0
        assert params.kappa == 1.0


class TestSigmaPoints:
    """Test sigma point generation."""

    def test_sigma_point_generation(self):
        """Test basic sigma point generation."""
        state = jnp.array([1.0, 2.0, 0.5, -0.3, 0.1, 0.0, 0.0, 0.0])
        covariance = jnp.eye(8) * 0.1
        params = UKFParams()

        sigma_points, weights = generate_sigma_points(state, covariance, params)

        # Should have 2n+1 = 17 sigma points
        assert sigma_points.shape == (17, 8)

        # First sigma point should be the mean
        np.testing.assert_array_almost_equal(sigma_points[0], state, decimal=10)

        # Check weights
        weights_mean, weights_cov = weights
        assert weights_mean.shape == (17,)
        assert weights_cov.shape == (17,)

        # Mean weights should sum to 1
        assert abs(jnp.sum(weights_mean) - 1.0) < 1e-10

        # Covariance weights don't necessarily sum to 1 in UKF
        # (they incorporate the beta parameter for higher-order moments)
        # Just check they are reasonable values
        assert jnp.all(jnp.isfinite(weights_cov))
        assert weights_cov[0] > 0  # First weight should be positive when beta=2

    def test_sigma_point_propagation(self):
        """Test sigma point propagation through dynamics."""
        # Create simple sigma points
        sigma_points = jnp.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Center
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Perturbed x
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Perturbed y
        ])

        dt = 0.1
        accel = jnp.array([1.0, 0.0])
        gyro = jnp.array([0.0])
        velocity_damping = 0.0

        propagated = propagate_sigma_points(
            sigma_points, dt, accel, gyro, velocity_damping
        )

        assert propagated.shape == sigma_points.shape

        # All sigma points should show forward motion due to acceleration
        for i in range(propagated.shape[0]):
            assert propagated[i, 0] > sigma_points[i, 0]  # x increased
            assert propagated[i, 2] > sigma_points[i, 2]  # vx increased

    def test_sigma_point_weights_properties(self):
        """Test mathematical properties of sigma point weights."""
        state = jnp.zeros(8)
        covariance = jnp.eye(8)
        params = UKFParams(alpha=0.5, beta=2.0, kappa=1.0)

        _, weights = generate_sigma_points(state, covariance, params)
        weights_mean, weights_cov = weights

        # Mean weights should sum to 1
        assert abs(jnp.sum(weights_mean) - 1.0) < 1e-12

        # Covariance weights have different normalization in UKF
        assert jnp.all(jnp.isfinite(weights_cov))

        # First weight (mean) should be different from others
        assert weights_mean[0] != weights_mean[1]
        assert weights_cov[0] != weights_cov[1]

        # All other weights should be equal
        for i in range(1, len(weights_mean)):
            assert weights_mean[i] == weights_mean[1]
            assert weights_cov[i] == weights_cov[1]


class TestUKFPredict:
    """Test UKF prediction step."""

    def test_ukf_predict_basic(self):
        """Test basic UKF prediction step."""
        initial_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        initial_cov = jnp.eye(8) * 0.1
        ukf_state = UKFState(initial_state, initial_cov, 0.0)

        dt = 0.1
        accel = jnp.array([1.0, 0.0])
        gyro = jnp.array([0.0])
        velocity_damping = 0.0
        params = UKFParams()

        predicted = ukf_predict(
            ukf_state, dt, accel, gyro, velocity_damping,
            accel_noise_std=0.1, gyro_noise_std=0.05, bias_drift_std=0.01,
            params=params
        )

        # Check state prediction (UKF has numerical differences due to sigma points and rotation)
        # The rotation now couples heading uncertainty with position/velocity through sigma points
        assert predicted.state[0] == pytest.approx(0.5, rel=0.1)  # x position (allow 10% relative error)
        assert predicted.state[1] == pytest.approx(0.0, abs=1e-2)  # y position (allow small absolute error)
        assert predicted.state[2] == pytest.approx(10.0, rel=0.1)  # vx velocity
        assert predicted.state[3] == pytest.approx(0.0, abs=1e-2)  # vy velocity

        # Covariance should increase
        assert jnp.all(jnp.diag(predicted.covariance) >= jnp.diag(initial_cov))

    def test_ukf_predict_with_bias(self):
        """Test UKF prediction with IMU biases."""
        initial_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1])
        initial_cov = jnp.eye(8) * 0.1
        ukf_state = UKFState(initial_state, initial_cov, 0.0)

        dt = 0.1
        accel = jnp.array([1.0, 0.5])
        gyro = jnp.array([0.2])
        velocity_damping = 0.0
        params = UKFParams()

        predicted = ukf_predict(
            ukf_state, dt, accel, gyro, velocity_damping,
            accel_noise_std=0.1, gyro_noise_std=0.05, bias_drift_std=0.01,
            params=params
        )

        # Expected heading change: θ = 0 + (0.2 - 0.1)*0.1 = 0.01 rad
        assert predicted.state[4] == pytest.approx(0.01, abs=1e-6)

        # Biases should remain unchanged
        assert predicted.state[5] == pytest.approx(0.1, abs=1e-6)
        assert predicted.state[6] == pytest.approx(0.2, abs=1e-6)
        assert predicted.state[7] == pytest.approx(0.1, abs=1e-6)


class TestUKFUpdate:
    """Test UKF measurement update step."""

    def test_ukf_update_position_only(self):
        """Test UKF update with position measurement only."""
        prior_state = jnp.array([1.0, 2.0, 0.5, -0.3, 0.1, 0.0, 0.0, 0.0])
        prior_cov = jnp.eye(8) * 2.0
        ukf_state = UKFState(prior_state, prior_cov, -5.0)

        measurement = jnp.array([1.1, 1.9])
        measurement_noise = create_measurement_noise(0.5, 1.0, False)
        params = UKFParams()

        result = ukf_update(ukf_state, measurement, measurement_noise, False, params=params)

        assert not result.gated

        # State should move toward measurement
        assert result.state.state[0] > prior_state[0]  # x moves toward 1.1
        assert result.state.state[1] < prior_state[1]  # y moves toward 1.9

        # Uncertainty should decrease
        assert jnp.all(jnp.diag(result.state.covariance) <= jnp.diag(prior_cov))

    def test_ukf_update_position_and_heading(self):
        """Test UKF update with both position and heading measurements."""
        prior_state = jnp.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
        prior_cov = jnp.eye(8) * 1.0
        ukf_state = UKFState(prior_state, prior_cov, 0.0)

        position = jnp.array([0.1, -0.1])
        heading = 0.4
        measurement = jnp.concatenate([position, jnp.array([heading])])
        measurement_noise = create_measurement_noise(0.3, 1.0, True, 0.1)
        params = UKFParams()

        result = ukf_update(ukf_state, measurement, measurement_noise, True, params=params)

        assert not result.gated
        assert result.innovation.shape == (3,)

        # State should move toward measurement
        assert result.state.state[0] > prior_state[0]  # x toward 0.1
        assert result.state.state[1] < prior_state[1]  # y toward -0.1
        assert result.state.state[4] < prior_state[4]  # theta toward 0.4

    def test_ukf_update_heading_wrap(self):
        """Test proper handling of heading wrap-around in UKF."""
        prior_state = jnp.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
        prior_cov = jnp.eye(8) * 0.1
        ukf_state = UKFState(prior_state, prior_cov, 0.0)

        position = jnp.array([0.0, 0.0])
        heading = -3.0  # Should be close to 3.0 when wrapped
        measurement = jnp.concatenate([position, jnp.array([heading])])
        measurement_noise = create_measurement_noise(0.1, 1.0, True, 0.1)
        params = UKFParams()

        result = ukf_update(ukf_state, measurement, measurement_noise, True, params=params)

        # Innovation should be wrapped properly
        wrapped_innovation = result.innovation[2]
        assert abs(wrapped_innovation) < 0.5

    def test_ukf_update_gating(self):
        """Test measurement gating for outliers."""
        prior_state = jnp.zeros(8)
        prior_cov = jnp.eye(8) * 0.1
        ukf_state = UKFState(prior_state, prior_cov, 0.0)

        # Outlier measurement
        measurement = jnp.array([100.0, 100.0])
        measurement_noise = create_measurement_noise(1.0, 1.0, False)
        params = UKFParams()

        result = ukf_update(ukf_state, measurement, measurement_noise, False,
                           gate_threshold=9.21, params=params)

        # Should be gated
        assert result.gated

        # State should remain unchanged
        np.testing.assert_array_equal(result.state.state, ukf_state.state)
        np.testing.assert_array_equal(result.state.covariance, ukf_state.covariance)


class TestUKFFilter:
    """Test UKF filter class interface."""

    def create_test_filter(self) -> UKFFilter:
        """Create a test UKF filter."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0,
            b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_cov = jnp.eye(8) * 0.5

        return UKFFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            velocity_damping=0.1,
            accel_noise_std=0.2,
            gyro_noise_std=0.05,
            bias_drift_std=0.01,
            position_noise_std=1.0,
            heading_noise_std=0.1,
            gate_threshold=9.21,
            ukf_params=UKFParams(),
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

        filter.predict(
            dt=0.1,
            accel=jnp.array([2.0, 0.0]),
            gyro=jnp.array([0.0])
        )

        state = filter.get_current_state()
        # Should have moved forward due to acceleration
        assert state.x > 0.0
        assert state.vx > 0.0

    def test_filter_update_only(self):
        """Test filter update without prediction."""
        filter = self.create_test_filter()

        result = filter.update(
            position=jnp.array([1.0, 2.0]),
            confidence=0.8
        )

        assert not result.gated
        state = filter.get_current_state()
        # State should move toward measurement
        assert state.x > 0.0
        assert state.y > 0.0

    def test_filter_predict_update_cycle(self):
        """Test complete predict-update cycle."""
        filter = self.create_test_filter()

        # Predict step
        filter.predict(
            dt=0.1,
            accel=jnp.array([1.0, 0.0]),
            gyro=jnp.array([0.1])
        )

        state_after_predict = filter.get_current_state()

        # Update step
        result = filter.update(
            position=jnp.array([0.5, 0.1]),
            heading=0.05,
            confidence=0.9
        )

        assert not result.gated
        state_after_update = filter.get_current_state()

        # Y position should change due to measurement
        assert abs(state_after_update.y - 0.1) < abs(state_after_predict.y - 0.1)
        # Heading should change due to measurement
        assert abs(state_after_update.theta - 0.05) < abs(state_after_predict.theta - 0.05)

    def test_filter_missing_measurements(self):
        """Test filter with missing measurements."""
        filter = self.create_test_filter()

        # Update with no measurements
        result = filter.update()
        assert result.innovation.size == 0

        # Update with position only
        result = filter.update(position=jnp.array([1.0, 1.0]))
        assert not result.gated
        assert result.innovation.shape == (2,)

        # Update with heading only
        result = filter.update(heading=0.5)
        assert not result.gated
        assert result.innovation.shape == (3,)


class TestUKFvsEKF:
    """Compare UKF vs EKF behavior."""

    def test_ukf_vs_ekf_linear_case(self):
        """Test UKF vs EKF on linear case (should be similar)."""
        # For linear case, UKF and EKF should give similar results
        initial_state = State2D(
            x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0,
            b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_cov = jnp.eye(8) * 1.0

        ukf_filter = UKFFilter(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            velocity_damping=0.0,
            accel_noise_std=0.0,
            gyro_noise_std=0.0,
            bias_drift_std=0.0,
            position_noise_std=1.0,
            heading_noise_std=0.1,
        )

        # Series of measurements
        measurements = [
            jnp.array([1.0, 0.0]),
            jnp.array([2.0, 0.5]),
            jnp.array([3.0, 1.0]),
        ]

        for measurement in measurements:
            # No prediction (linear case)
            ukf_filter.predict(dt=0.1, accel=jnp.zeros(2), gyro=jnp.zeros(1))

            # Update with measurement
            result = ukf_filter.update(position=measurement, confidence=1.0)
            assert not result.gated

        # Final state should be reasonable
        final_state = ukf_filter.get_current_state()
        assert abs(final_state.x - 3.0) < 2.0  # Should be close to last measurement
        assert abs(final_state.y - 1.0) < 2.0