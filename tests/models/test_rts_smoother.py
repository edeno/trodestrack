"""Tests for RTS smoother implementation."""

import pytest
import jax.numpy as jnp
import numpy as np

from trodestrack.models.rts_smoother import (
    rts_backward_step,
    rts_smooth,
    RTSSmoother,
    ForwardPassData,
    compute_smoothing_improvement,
)
from trodestrack.models.ekf import EKFState, EKFResult
from trodestrack.models.state import State2D


class TestRTSBackwardStep:
    """Test individual RTS backward step computation."""

    def test_identity_case(self):
        """Test RTS step when next state equals prediction (no correction needed)."""
        # Setup simple 2D case
        n_dim = 2
        x_f = jnp.array([1.0, 2.0])
        P_f = jnp.eye(n_dim) * 0.1
        x_p_next = jnp.array([1.1, 2.1])  # Predicted next state
        P_p_next = jnp.eye(n_dim) * 0.2
        x_s_next = x_p_next  # Smoothed next state equals prediction

        x_s, P_s = rts_backward_step(x_s_next, P_p_next, x_f, P_f, x_p_next, P_p_next)

        # When smoothed next state equals prediction, smoothed current should equal filtered
        np.testing.assert_allclose(x_s, x_f, rtol=1e-6)
        np.testing.assert_allclose(P_s, P_f, rtol=1e-6)

    def test_correction_case(self):
        """Test RTS step when next smoothed state differs from prediction."""
        n_dim = 2
        x_f = jnp.array([1.0, 2.0])
        P_f = jnp.eye(n_dim) * 0.1
        x_p_next = jnp.array([1.1, 2.1])  # Predicted next state
        P_p_next = jnp.eye(n_dim) * 0.2
        x_s_next = jnp.array([1.05, 2.05])  # Smoothed next state (corrected)
        P_s_next = jnp.eye(n_dim) * 0.15

        x_s, P_s = rts_backward_step(x_s_next, P_s_next, x_f, P_f, x_p_next, P_p_next)

        # Smoothed state should be between filtered and prediction correction
        assert jnp.all(jnp.abs(x_s - x_f) > 0)  # Should be different from filtered
        assert not jnp.allclose(P_s, P_f, rtol=1e-10)  # Covariance should be modified

    def test_numerical_stability(self):
        """Test RTS step with near-singular covariance matrices."""
        n_dim = 8  # Full state dimension
        x_f = jnp.zeros(n_dim)
        P_f = jnp.eye(n_dim) * 1e-6  # Very small covariance
        x_p_next = jnp.ones(n_dim) * 0.01
        P_p_next = jnp.eye(n_dim) * 1e-5  # Slightly larger
        x_s_next = jnp.ones(n_dim) * 0.005
        P_s_next = jnp.eye(n_dim) * 1e-5

        # Should not raise exception
        x_s, P_s = rts_backward_step(x_s_next, P_s_next, x_f, P_f, x_p_next, P_p_next)

        # Results should be finite
        assert jnp.all(jnp.isfinite(x_s))
        assert jnp.all(jnp.isfinite(P_s))


class TestRTSSmooth:
    """Test full RTS smoothing algorithm."""

    def test_empty_sequence(self):
        """Test RTS smoothing with empty input."""
        forward_data = ForwardPassData(
            filtered_states=[],
            filtered_covariances=[],
            predicted_states=[],
            predicted_covariances=[],
            log_likelihood=0.0,
        )

        result = rts_smooth(forward_data)

        assert len(result.smoothed_states) == 0
        assert len(result.smoothed_covariances) == 0
        assert result.log_likelihood == 0.0

    def test_single_timestep(self):
        """Test RTS smoothing with single measurement."""
        n_dim = 8
        state = jnp.array([1.0, 2.0, 0.1, 0.2, 0.5, 0.01, 0.02, 0.03])
        covariance = jnp.eye(n_dim) * 0.1

        forward_data = ForwardPassData(
            filtered_states=[state],
            filtered_covariances=[covariance],
            predicted_states=[state],  # No next prediction needed
            predicted_covariances=[covariance],
            log_likelihood=10.0,
        )

        result = rts_smooth(forward_data)

        # Single timestep should remain unchanged
        assert len(result.smoothed_states) == 1
        np.testing.assert_allclose(result.smoothed_states[0], state, rtol=1e-10)
        np.testing.assert_allclose(result.smoothed_covariances[0], covariance, rtol=1e-10)
        assert result.log_likelihood == 10.0

    def test_two_timestep_sequence(self):
        """Test RTS smoothing with two timesteps."""
        n_dim = 8

        # First timestep (filtered)
        x1_f = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        P1_f = jnp.eye(n_dim) * 0.1

        # Second timestep (predicted from first, then filtered)
        x2_p = jnp.array([0.1, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Prediction
        P2_p = jnp.eye(n_dim) * 0.2
        x2_f = jnp.array([0.12, 0.12, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0])  # After measurement
        P2_f = jnp.eye(n_dim) * 0.15

        forward_data = ForwardPassData(
            filtered_states=[x1_f, x2_f],
            filtered_covariances=[P1_f, P2_f],
            predicted_states=[x1_f, x2_p],  # First is initial, second is prediction
            predicted_covariances=[P1_f, P2_p],
            log_likelihood=25.0,
        )

        result = rts_smooth(forward_data)

        assert len(result.smoothed_states) == 2
        assert len(result.smoothed_covariances) == 2

        # Final timestep should remain unchanged
        np.testing.assert_allclose(result.smoothed_states[1], x2_f, rtol=1e-10)

        # First timestep should be modified (improved by future information)
        x1_s = result.smoothed_states[0]
        assert not jnp.allclose(x1_s, x1_f, rtol=1e-6)  # Should be different

    def test_monotonic_improvement_property(self):
        """Test that smoothing should generally improve estimates on clean data."""
        n_dim = 8
        n_steps = 5

        # Create synthetic sequence with consistent motion
        filtered_states = []
        filtered_covariances = []
        predicted_states = []
        predicted_covariances = []

        # Initialize
        x = jnp.array([0.0, 0.0, 2.0, 1.0, 0.1, 0.01, 0.02, 0.01])  # Constant velocity
        P = jnp.eye(n_dim) * 0.1

        for i in range(n_steps):
            # Add some noise to simulate measurement updates
            x_filtered = x + jnp.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * np.random.normal(0, 0.1)
            P_filtered = P * (1 + 0.1 * np.random.uniform(0.8, 1.2))

            filtered_states.append(x_filtered)
            filtered_covariances.append(P_filtered)

            # Predict next step
            if i < n_steps - 1:
                dt = 0.1
                x_next = x_filtered.copy()
                x_next = x_next.at[0].add(x_filtered[2] * dt)  # x += vx * dt
                x_next = x_next.at[1].add(x_filtered[3] * dt)  # y += vy * dt
                predicted_states.append(x_next)
                predicted_covariances.append(P_filtered * 1.1)
                x = x_next
            else:
                predicted_states.append(x_filtered)
                predicted_covariances.append(P_filtered)

        forward_data = ForwardPassData(
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances,
            predicted_states=predicted_states,
            predicted_covariances=predicted_covariances,
            log_likelihood=50.0,
        )

        result = rts_smooth(forward_data)

        # Check that all outputs are finite and reasonable
        for i, (x_s, P_s) in enumerate(zip(result.smoothed_states, result.smoothed_covariances)):
            assert jnp.all(jnp.isfinite(x_s)), f"Non-finite smoothed state at step {i}"
            assert jnp.all(jnp.isfinite(P_s)), f"Non-finite smoothed covariance at step {i}"
            assert jnp.all(jnp.diag(P_s) > 0), f"Non-positive diagonal covariance at step {i}"


class TestRTSSmoother:
    """Test RTSSmoother class interface."""

    def test_initialization(self):
        """Test RTSSmoother initialization with parameters."""
        smoother = RTSSmoother(
            velocity_damping=0.2,
            accel_noise_std=1.0,
            gyro_noise_std=0.2,
            bias_drift_std=0.02,
        )

        assert smoother.velocity_damping == 0.2
        assert smoother.accel_noise_std == 1.0
        assert smoother.gyro_noise_std == 0.2
        assert smoother.bias_drift_std == 0.02

    def test_collect_forward_data_mismatch(self):
        """Test error handling for mismatched input lengths."""
        smoother = RTSSmoother()

        # Create mismatched inputs
        ekf_results = [create_dummy_ekf_result() for _ in range(3)]
        prediction_data = [(jnp.zeros(8), jnp.eye(8)) for _ in range(2)]  # Different length

        with pytest.raises(ValueError, match="Mismatch between EKF results"):
            smoother.collect_forward_data(ekf_results, prediction_data)

    def test_collect_forward_data_success(self):
        """Test successful collection of forward pass data."""
        smoother = RTSSmoother()

        # Create matching inputs
        ekf_results = [create_dummy_ekf_result() for _ in range(3)]
        prediction_data = [(jnp.zeros(8), jnp.eye(8)) for _ in range(3)]

        forward_data = smoother.collect_forward_data(ekf_results, prediction_data)

        assert len(forward_data.filtered_states) == 3
        assert len(forward_data.filtered_covariances) == 3
        assert len(forward_data.predicted_states) == 3
        assert len(forward_data.predicted_covariances) == 3

    def test_smooth_sequence_integration(self):
        """Test full smoother workflow."""
        smoother = RTSSmoother()

        # Create simple forward data
        n_steps = 3
        filtered_states = [jnp.array([i, i, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) for i in range(n_steps)]
        filtered_covariances = [jnp.eye(8) * 0.1 for _ in range(n_steps)]
        predicted_states = filtered_states.copy()
        predicted_covariances = filtered_covariances.copy()

        forward_data = ForwardPassData(
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances,
            predicted_states=predicted_states,
            predicted_covariances=predicted_covariances,
            log_likelihood=30.0,
        )

        result = smoother.smooth_sequence(forward_data)

        assert len(result.smoothed_states) == n_steps
        assert len(result.smoothed_covariances) == n_steps
        assert result.log_likelihood == 30.0


class TestSmoothingImprovement:
    """Test smoothing improvement computation."""

    def test_perfect_case(self):
        """Test improvement computation when smoothed equals ground truth."""
        n_steps = 5
        n_dim = 8

        # Ground truth
        ground_truth = [jnp.array([i, i, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) for i in range(n_steps)]

        # Filtered has some error
        filtered_states = [gt + jnp.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for gt in ground_truth]

        # Smoothed is perfect
        smoothed_states = ground_truth.copy()

        filt_rmse, smooth_rmse, improvement = compute_smoothing_improvement(
            filtered_states, smoothed_states, ground_truth
        )

        assert filt_rmse > 0
        assert smooth_rmse == pytest.approx(0.0, abs=1e-10)
        assert improvement == pytest.approx(100.0, rel=1e-3)

    def test_partial_improvement(self):
        """Test improvement computation with partial error reduction."""
        n_steps = 10
        ground_truth = [jnp.array([i, i*0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]) for i in range(n_steps)]

        # Filtered has 10cm error
        filtered_states = [gt + jnp.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for gt in ground_truth]

        # Smoothed has 5cm error
        smoothed_states = [gt + jnp.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for gt in ground_truth]

        filt_rmse, smooth_rmse, improvement = compute_smoothing_improvement(
            filtered_states, smoothed_states, ground_truth
        )

        assert filt_rmse == pytest.approx(10.0, abs=0.1)
        assert smooth_rmse == pytest.approx(5.0, abs=0.1)
        assert improvement == pytest.approx(50.0, rel=1e-2)

    def test_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        filtered = [jnp.zeros(8) for _ in range(3)]
        smoothed = [jnp.zeros(8) for _ in range(2)]  # Different length
        truth = [jnp.zeros(8) for _ in range(3)]

        with pytest.raises(ValueError, match="All state lists must have the same length"):
            compute_smoothing_improvement(filtered, smoothed, truth)


def create_dummy_ekf_result():
    """Create a dummy EKF result for testing."""
    state = jnp.zeros(8)
    covariance = jnp.eye(8)
    ekf_state = EKFState(
        state=state,
        covariance=covariance,
        log_likelihood=10.0,
    )

    return EKFResult(
        state=ekf_state,
        innovation=jnp.array([0.1, 0.1]),
        innovation_covariance=jnp.eye(2) * 0.1,
        kalman_gain=jnp.zeros((8, 2)),
        gated=False,
    )