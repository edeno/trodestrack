"""Tests for cached EKF implementation."""

import jax.numpy as jnp
import numpy as np

from trodestrack.models.cached_ekf import (
    CachedEKFFilter,
    efficient_rts_smooth_with_cache,
    compute_cache_efficiency_stats,
)
from trodestrack.models.ekf import EKFState, create_initial_ekf_state
from trodestrack.models.state import State2D


class TestCachedEKFFilter:
    """Test cached EKF filter implementation."""

    def test_initialization(self):
        """Test cached EKF filter initialization."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            velocity_damping=0.2,
            enable_caching=True,
        )

        assert cached_filter.velocity_damping == 0.2
        assert cached_filter.enable_caching is True
        assert cached_filter.step_count == 0
        assert len(cached_filter.cached_computations.state_jacobians) == 0

    def test_predict_with_caching(self):
        """Test prediction step with caching enabled."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        dt = 0.1
        accel = jnp.array([0.1, 0.2])  # m/s²
        gyro = jnp.array([0.05])  # rad/s

        predicted_state, cached_data = cached_filter.predict_with_caching(dt, accel, gyro)

        # Check predicted state is valid
        assert predicted_state.state.shape == (8,)
        assert predicted_state.covariance.shape == (8, 8)
        assert jnp.all(jnp.isfinite(predicted_state.state))
        assert jnp.all(jnp.isfinite(predicted_state.covariance))

        # Check cached data
        assert "state_jacobian" in cached_data
        assert "process_noise" in cached_data
        assert "predicted_covariance" in cached_data
        assert cached_data["state_jacobian"].shape == (8, 8)
        assert cached_data["process_noise"].shape == (8, 8)

        # Check that data was cached
        assert len(cached_filter.cached_computations.state_jacobians) == 1
        assert len(cached_filter.cached_computations.predicted_covariances) == 1

    def test_update_with_caching_position_only(self):
        """Test update step with position measurement and caching."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        # Create predicted state for update
        predicted_state = EKFState(
            state=jnp.array([1.0, 2.0, 1.0, 1.0, 0.1, 0.0, 0.0, 0.0]),
            covariance=jnp.eye(8) * 0.2,
            log_likelihood=0.0,
        )

        position = jnp.array([1.1, 2.1])  # cm
        confidence = 0.8

        result, cached_data = cached_filter.update_with_caching(
            predicted_state, position=position, confidence=confidence
        )

        # Check result
        assert not result.gated
        assert result.innovation.shape == (2,)  # Position only
        assert result.innovation_covariance.shape == (2, 2)

        # Check cached data
        assert "innovation_covariance" in cached_data
        assert len(cached_filter.cached_computations.innovation_covariances) == 1

    def test_update_with_caching_no_measurement(self):
        """Test update step with no measurements."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        predicted_state = EKFState(
            state=jnp.array([1.0, 2.0, 1.0, 1.0, 0.1, 0.0, 0.0, 0.0]),
            covariance=jnp.eye(8) * 0.2,
            log_likelihood=0.0,
        )

        result, cached_data = cached_filter.update_with_caching(predicted_state)

        # Should return no-update result
        assert not result.gated
        assert result.innovation.shape == (0,)
        assert result.innovation_covariance.shape[1] == 0  # Check second dimension

        # Check cached data
        assert cached_data["measurement_jacobian"] is None
        assert cached_data["innovation_covariance"] is None

    def test_step_with_caching(self):
        """Test complete EKF step with caching."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        dt = 0.1
        accel = jnp.array([0.1, 0.2])
        gyro = jnp.array([0.05])
        position = jnp.array([0.1, 0.1])
        heading = 0.05
        confidence = 0.9

        result, cached_data = cached_filter.step_with_caching(
            dt, accel, gyro, position, heading, confidence
        )

        # Check that step was successful
        assert not result.gated
        assert cached_filter.step_count == 1

        # Check all cached data is present
        expected_keys = [
            "state_jacobian",
            "process_noise",
            "predicted_covariance",
            "measurement_jacobian",
            "innovation_covariance",
        ]
        for key in expected_keys:
            assert key in cached_data

        # Check cached computations were stored
        assert len(cached_filter.cached_computations.state_jacobians) == 1
        assert len(cached_filter.cached_computations.predicted_covariances) == 1

    def test_multiple_steps_caching(self):
        """Test caching over multiple steps."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        n_steps = 5
        dt = 0.1

        for i in range(n_steps):
            accel = jnp.array([0.1, 0.2])
            gyro = jnp.array([0.05])
            position = jnp.array([i * 0.1, i * 0.1])
            confidence = 0.8

            result, cached_data = cached_filter.step_with_caching(
                dt, accel, gyro, position=position, confidence=confidence
            )

            assert not result.gated
            assert cached_filter.step_count == i + 1

        # Check that all steps were cached
        assert len(cached_filter.cached_computations.state_jacobians) == n_steps
        assert len(cached_filter.cached_computations.predicted_covariances) == n_steps

    def test_cache_retrieval(self):
        """Test retrieval of cached computations."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        # Perform a step to generate cached data
        dt = 0.1
        accel = jnp.array([0.1, 0.2])
        gyro = jnp.array([0.05])
        position = jnp.array([0.1, 0.1])

        cached_filter.step_with_caching(dt, accel, gyro, position=position)

        # Test cache retrieval
        jacobian = cached_filter.get_cached_jacobian(0)
        assert jacobian is not None
        assert jacobian.shape == (8, 8)

        pred_cov = cached_filter.get_cached_predicted_covariance(0)
        assert pred_cov is not None
        assert pred_cov.shape == (8, 8)

        # Test out-of-bounds access
        assert cached_filter.get_cached_jacobian(10) is None
        assert cached_filter.get_cached_predicted_covariance(10) is None

    def test_caching_disabled(self):
        """Test filter behavior with caching disabled."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=False,  # Disabled
        )

        dt = 0.1
        accel = jnp.array([0.1, 0.2])
        gyro = jnp.array([0.05])

        result, cached_data = cached_filter.step_with_caching(dt, accel, gyro)

        # Filter should work but no caching
        assert cached_filter.step_count == 1
        assert len(cached_filter.cached_computations.state_jacobians) == 0
        assert cached_filter.get_cached_jacobian(0) is None

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        # Perform multiple steps
        for i in range(3):
            dt = 0.1
            accel = jnp.array([0.1, 0.2])
            gyro = jnp.array([0.05])
            cached_filter.step_with_caching(dt, accel, gyro)

        # Verify cache has data
        assert cached_filter.step_count == 3
        assert len(cached_filter.cached_computations.state_jacobians) == 3

        # Clear cache
        cached_filter.clear_cache()

        # Verify cache is empty
        assert cached_filter.step_count == 0
        assert len(cached_filter.cached_computations.state_jacobians) == 0
        assert cached_filter.get_cached_jacobian(0) is None


class TestEfficientRTSSmooth:
    """Test efficient RTS smoothing with cached data."""

    def test_empty_sequence(self):
        """Test RTS smoothing with empty input."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        smoothed_states, smoothed_covariances = efficient_rts_smooth_with_cache(cached_filter, [])

        assert len(smoothed_states) == 0
        assert len(smoothed_covariances) == 0

    def test_single_step_sequence(self):
        """Test RTS smoothing with single step."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        # Generate single EKF result
        dt = 0.1
        accel = jnp.array([0.1, 0.2])
        gyro = jnp.array([0.05])
        result, _ = cached_filter.step_with_caching(dt, accel, gyro)

        smoothed_states, smoothed_covariances = efficient_rts_smooth_with_cache(
            cached_filter, [result]
        )

        assert len(smoothed_states) == 1
        assert len(smoothed_covariances) == 1
        # Single step should remain unchanged
        np.testing.assert_allclose(smoothed_states[0], result.state.state, rtol=1e-10)

    def test_multi_step_sequence(self):
        """Test RTS smoothing with multiple steps and cached data."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=2.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        # Generate sequence of EKF results
        results = []
        n_steps = 5
        dt = 0.1

        for i in range(n_steps):
            accel = jnp.array([0.1, 0.2])
            gyro = jnp.array([0.05])
            position = jnp.array([i * 0.2, i * 0.1])  # Measurements with some error
            confidence = 0.8

            result, _ = cached_filter.step_with_caching(
                dt, accel, gyro, position=position, confidence=confidence
            )
            results.append(result)

        # Perform efficient RTS smoothing
        smoothed_states, smoothed_covariances = efficient_rts_smooth_with_cache(
            cached_filter, results
        )

        assert len(smoothed_states) == n_steps
        assert len(smoothed_covariances) == n_steps

        # Check that all results are finite
        for i, (x_s, P_s) in enumerate(zip(smoothed_states, smoothed_covariances)):
            assert jnp.all(jnp.isfinite(x_s)), f"Non-finite smoothed state at step {i}"
            assert jnp.all(jnp.isfinite(P_s)), f"Non-finite smoothed covariance at step {i}"
            assert x_s.shape == (8,)
            assert P_s.shape == (8, 8)

        # Final step should be unchanged (no future information)
        np.testing.assert_allclose(smoothed_states[-1], results[-1].state.state, rtol=1e-10)


class TestCacheEfficiencyStats:
    """Test cache efficiency computation."""

    def test_empty_cache_stats(self):
        """Test efficiency stats with empty cache."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        stats = compute_cache_efficiency_stats(cached_filter)

        assert stats["total_steps"] == 0
        assert stats["cached_jacobians"] == 0
        assert stats["cached_covariances"] == 0
        assert stats["jacobian_cache_rate"] == 0.0
        assert stats["covariance_cache_rate"] == 0.0

    def test_full_cache_stats(self):
        """Test efficiency stats with full cache utilization."""
        initial_state = State2D(
            x=0.0, y=0.0, vx=1.0, vy=1.0, theta=0.0, b_gz=0.0, b_ax=0.0, b_ay=0.0
        )
        initial_covariance = jnp.eye(8) * 0.1
        ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        cached_filter = CachedEKFFilter(
            initial_state=ekf_state,
            enable_caching=True,
        )

        # Perform multiple steps
        n_steps = 10
        for i in range(n_steps):
            dt = 0.1
            accel = jnp.array([0.1, 0.2])
            gyro = jnp.array([0.05])
            cached_filter.step_with_caching(dt, accel, gyro)

        stats = compute_cache_efficiency_stats(cached_filter)

        assert stats["total_steps"] == n_steps
        assert stats["cached_jacobians"] == n_steps
        assert stats["cached_covariances"] == n_steps
        assert stats["jacobian_cache_rate"] == 1.0
        assert stats["covariance_cache_rate"] == 1.0
        assert stats["estimated_cache_memory_bytes"] > 0
        assert stats["estimated_cache_memory_mb"] > 0
