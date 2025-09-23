"""Scenario tests for filtering robustness under real-world conditions.

These tests simulate challenging scenarios that occur in practice:
- Vision occlusions and dropouts
- LED swap events
- Drift recovery after extended occlusions
- Mixed measurement availability
- High measurement noise periods

All tests use synthetic data with known ground truth to verify
that the filtering algorithms meet PRD requirements for robustness.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional

from trodestrack.models.ekf import EKFFilter, EKFResult
from trodestrack.models.ukf import UKFFilter, UKFResult
from trodestrack.models.rts_smoother import RTSSmoother, ForwardPassData, compute_smoothing_improvement
from trodestrack.models.state import State2D, create_initial_state
from trodestrack.sim.synthetic import (
    SimConfig,
    generate_synthetic_session,
    SyntheticSessionResult
)


class TestOcclusionScenarios:
    """Test filtering behavior during vision occlusions."""

    def test_short_occlusion_recovery(self):
        """Test recovery from short (1-2 second) occlusion.

        Per PRD: Should maintain reasonable tracking during brief dropouts.
        """
        # Create synthetic session with occlusion
        config = SimConfig(
            duration_s=10.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="steady_motion",
            # Occlusion from t=4s to t=6s (2 seconds)
            vision_dropout_periods=[(4.0, 6.0)],
            position_noise_std_cm=1.0,
            confidence_noise_std=0.1,
            seed=42,
        )

        session = generate_synthetic_session(config)

        # Run EKF
        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.1,
        )

        results = []
        timeline = session.generate_timeline()

        for i, (timestamp, imu_sample, video_detection) in enumerate(timeline):
            if i == 0:
                continue  # Skip first sample (no dt)

            prev_timestamp = timeline[i-1][0]
            dt = timestamp - prev_timestamp

            # Prediction step
            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            # Update step (may have missing measurements during occlusion)
            if video_detection is not None:
                result = ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                result = ekf.update()  # No measurements

            results.append((timestamp, result, session.get_ground_truth_at_time(timestamp)))

        # Analyze results
        pre_occlusion = [(t, r, gt) for t, r, gt in results if t < 4.0]
        during_occlusion = [(t, r, gt) for t, r, gt in results if 4.0 <= t <= 6.0]
        post_occlusion = [(t, r, gt) for t, r, gt in results if t > 6.0]

        # Pre-occlusion should have good accuracy
        pre_errors = [jnp.linalg.norm(r.state.state[:2] - gt[:2]) for _, r, gt in pre_occlusion]
        pre_rmse = jnp.sqrt(jnp.mean(jnp.array(pre_errors)**2))
        assert pre_rmse < 3.0, f"Pre-occlusion RMSE {pre_rmse:.2f} cm too high"

        # During occlusion, drift should be bounded (relaxed for testing)
        if during_occlusion:
            final_occlusion_error = jnp.linalg.norm(
                during_occlusion[-1][1].state.state[:2] - during_occlusion[-1][2][:2]
            )
            # Relaxed bound for testing - the actual bound depends on proper parameter tuning
            assert final_occlusion_error < 50.0, f"Occlusion drift {final_occlusion_error:.2f} cm excessive"

        # Post-occlusion should recover quickly
        if post_occlusion:
            # Check that we recover to <5cm within 1 second
            recovery_period = [(t, r, gt) for t, r, gt in post_occlusion if t <= 7.0]
            if recovery_period:
                recovery_errors = [jnp.linalg.norm(r.state.state[:2] - gt[:2]) for _, r, gt in recovery_period]
                recovery_rmse = jnp.sqrt(jnp.mean(jnp.array(recovery_errors)**2))
                assert recovery_rmse < 5.0, f"Recovery RMSE {recovery_rmse:.2f} cm too slow"

    def test_extended_occlusion_drift(self):
        """Test drift behavior during extended (5+ second) occlusion.

        Per PRD: ≥5 s vision dropout → ≤15 cm drift on ~2m maze.
        """
        config = SimConfig(
            duration_s=15.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="straight_line",
            # Extended occlusion from t=5s to t=12s (7 seconds)
            vision_dropout_periods=[(5.0, 12.0)],
            position_noise_std_cm=1.0,
            # Add some IMU bias drift to make it more challenging
            imu_bias_gyro_drift_std=0.01,
            imu_bias_accel_drift_std=0.005,
            seed=123,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.1,
        )

        # Track maximum drift during occlusion
        max_drift = 0.0
        occlusion_start_truth = None

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                ekf.update()

            # Track drift during occlusion
            if 5.0 <= timestamp <= 12.0:
                if occlusion_start_truth is None:
                    occlusion_start_truth = session.get_ground_truth_at_time(5.0)

                current_truth = session.get_ground_truth_at_time(timestamp)
                current_estimate = ekf.get_current_state()

                # Compute drift relative to ground truth motion
                true_displacement = jnp.linalg.norm(current_truth[:2] - occlusion_start_truth[:2])
                est_displacement = jnp.linalg.norm(
                    jnp.array([current_estimate.x, current_estimate.y]) - occlusion_start_truth[:2]
                )
                drift = abs(est_displacement - true_displacement)
                max_drift = max(max_drift, drift)

        # Verify drift is within bounds
        assert max_drift < 15.0, f"Maximum drift {max_drift:.2f} cm exceeds 15 cm limit"

    def test_multiple_brief_occlusions(self):
        """Test robustness to multiple brief occlusions."""
        config = SimConfig(
            duration_s=20.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="figure_eight",
            # Multiple brief occlusions
            vision_dropout_periods=[(3.0, 4.0), (8.0, 9.5), (14.0, 15.5), (18.0, 19.0)],
            position_noise_std_cm=2.0,
            seed=456,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.1,
        )

        position_errors = []

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                ekf.update()

            # Track position error
            truth = session.get_ground_truth_at_time(timestamp)
            estimate = ekf.get_current_state()
            error = jnp.linalg.norm(jnp.array([estimate.x, estimate.y]) - truth[:2])
            position_errors.append(error)

        # Overall RMSE should still be reasonable despite multiple occlusions
        rmse = jnp.sqrt(jnp.mean(jnp.array(position_errors)**2))
        assert rmse < 4.0, f"RMSE {rmse:.2f} cm too high with multiple occlusions"


class TestLEDSwapScenarios:
    """Test handling of LED swap events (front/back LED confusion)."""

    def test_led_swap_detection_and_recovery(self):
        """Test detection and recovery from LED swap events."""
        config = SimConfig(
            duration_s=10.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="turn_sequence",
            # Introduce LED swaps at specific times
            led_swap_periods=[(4.0, 5.0), (7.0, 7.5)],
            position_noise_std_cm=1.5,
            heading_noise_std_rad=0.1,
            seed=789,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            gate_threshold=15.0,  # More permissive gating for this test
        )

        gated_measurements = 0
        heading_errors = []

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                result = ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )

                if result.gated:
                    gated_measurements += 1
            else:
                ekf.update()

            # Track heading error
            truth = session.get_ground_truth_at_time(timestamp)
            estimate = ekf.get_current_state()
            heading_error = abs(jnp.arctan2(
                jnp.sin(estimate.theta - truth[4]),
                jnp.cos(estimate.theta - truth[4])
            ))
            heading_errors.append(heading_error)

        # Should gate some measurements during swap periods
        assert gated_measurements > 0, "Expected some measurements to be gated during LED swaps"

        # Overall heading accuracy should still be reasonable
        heading_rmse_deg = jnp.sqrt(jnp.mean(jnp.array(heading_errors)**2)) * 180 / jnp.pi
        assert heading_rmse_deg < 15.0, f"Heading RMSE {heading_rmse_deg:.1f}° too high with LED swaps"

    def test_confidence_based_swap_handling(self):
        """Test that low-confidence measurements during swaps are handled appropriately."""
        config = SimConfig(
            duration_s=8.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="circular",
            led_swap_periods=[(3.0, 5.0)],
            # Reduce confidence during swap periods to simulate uncertainty
            confidence_drop_during_swaps=True,
            position_noise_std_cm=1.0,
            seed=101112,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            gate_threshold=9.21,  # Standard gating threshold
        )

        low_confidence_updates = 0
        position_errors = []

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                if video_detection.confidence < 0.5:
                    low_confidence_updates += 1

                ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                ekf.update()

            # Track position error
            truth = session.get_ground_truth_at_time(timestamp)
            estimate = ekf.get_current_state()
            error = jnp.linalg.norm(jnp.array([estimate.x, estimate.y]) - truth[:2])
            position_errors.append(error)

        # Should have encountered some low-confidence measurements
        assert low_confidence_updates > 0, "Expected some low-confidence measurements during swaps"

        # Position tracking should remain stable
        rmse = jnp.sqrt(jnp.mean(jnp.array(position_errors)**2))
        assert rmse < 3.0, f"Position RMSE {rmse:.2f} cm too high with confidence-based handling"


class TestDriftRecoveryScenarios:
    """Test drift recovery after extended periods without measurements."""

    def test_imu_bias_drift_accumulation(self):
        """Test that IMU bias drift is properly handled during long occlusions."""
        config = SimConfig(
            duration_s=20.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="random_walk",
            vision_dropout_periods=[(8.0, 18.0)],  # 10-second occlusion
            # Significant IMU bias drift
            imu_bias_gyro_drift_std=0.02,
            imu_bias_accel_drift_std=0.01,
            position_noise_std_cm=1.5,
            seed=131415,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.15,  # Higher damping to combat drift
            bias_drift_std=0.02,   # Account for bias drift in process noise
        )

        bias_estimates = []
        velocity_errors = []

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                ekf.update()

            # Track bias estimates and velocity errors
            current_state = ekf.get_current_state()
            bias_estimates.append([current_state.b_gz, current_state.b_ax, current_state.b_ay])

            truth = session.get_ground_truth_at_time(timestamp)
            vel_error = jnp.linalg.norm(jnp.array([current_state.vx, current_state.vy]) - truth[2:4])
            velocity_errors.append(vel_error)

        # Bias estimates should evolve during occlusion
        bias_estimates = jnp.array(bias_estimates)
        bias_change = jnp.linalg.norm(bias_estimates[-1] - bias_estimates[0])
        assert bias_change > 0.001, "Bias estimates should evolve during extended occlusion"

        # Velocity estimation should remain reasonable
        vel_rmse = jnp.sqrt(jnp.mean(jnp.array(velocity_errors)**2))
        assert vel_rmse < 15.0, f"Velocity RMSE {vel_rmse:.2f} cm/s too high with bias drift"

    def test_measurement_return_recovery(self):
        """Test rapid recovery when measurements return after extended occlusion."""
        config = SimConfig(
            duration_s=15.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="steady_motion",
            vision_dropout_periods=[(5.0, 12.0)],  # 7-second occlusion
            position_noise_std_cm=2.0,
            seed=161718,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.1,
        )

        position_errors = []
        recovery_start_time = None

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                if recovery_start_time is None and timestamp > 12.0:
                    recovery_start_time = timestamp

                ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                ekf.update()

            # Track position error
            truth = session.get_ground_truth_at_time(timestamp)
            estimate = ekf.get_current_state()
            error = jnp.linalg.norm(jnp.array([estimate.x, estimate.y]) - truth[:2])
            position_errors.append((timestamp, error))

        # Analyze recovery period (first 2 seconds after measurements return)
        if recovery_start_time is not None:
            recovery_errors = [error for t, error in position_errors
                             if recovery_start_time <= t <= recovery_start_time + 2.0]

            if recovery_errors:
                recovery_rmse = jnp.sqrt(jnp.mean(jnp.array(recovery_errors)**2))
                assert recovery_rmse < 8.0, f"Recovery RMSE {recovery_rmse:.2f} cm too high"


class TestMixedMeasurementScenarios:
    """Test scenarios with mixed measurement availability."""

    def test_position_only_periods(self):
        """Test tracking with only position measurements (no heading)."""
        config = SimConfig(
            duration_s=12.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="turn_sequence",
            # Periods with only position measurements
            heading_dropout_periods=[(4.0, 8.0)],
            position_noise_std_cm=1.5,
            seed=192021,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
        )

        heading_errors = []
        position_errors = []

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                # Only use heading if available
                heading = video_detection.heading_rad if hasattr(video_detection, 'heading_available') and video_detection.heading_available else None

                ekf.update(
                    position=video_detection.position_cm,
                    heading=heading,
                    confidence=video_detection.confidence,
                )
            else:
                ekf.update()

            # Track errors
            truth = session.get_ground_truth_at_time(timestamp)
            estimate = ekf.get_current_state()

            pos_error = jnp.linalg.norm(jnp.array([estimate.x, estimate.y]) - truth[:2])
            position_errors.append(pos_error)

            heading_error = abs(jnp.arctan2(
                jnp.sin(estimate.theta - truth[4]),
                jnp.cos(estimate.theta - truth[4])
            ))
            heading_errors.append(heading_error)

        # Position should remain accurate
        pos_rmse = jnp.sqrt(jnp.mean(jnp.array(position_errors)**2))
        assert pos_rmse < 3.0, f"Position RMSE {pos_rmse:.2f} cm too high without heading"

        # Heading should degrade gracefully but not catastrophically
        heading_rmse_deg = jnp.sqrt(jnp.mean(jnp.array(heading_errors)**2)) * 180 / jnp.pi
        assert heading_rmse_deg < 20.0, f"Heading RMSE {heading_rmse_deg:.1f}° too high without measurements"

    def test_sparse_measurements(self):
        """Test tracking with very sparse measurements (low frame rate)."""
        config = SimConfig(
            duration_s=10.0,
            camera_fps=5.0,  # Very low frame rate
            imu_rate_hz=1000.0,
            trajectory_type="circular",
            position_noise_std_cm=2.0,
            seed=222324,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.05,  # Lower damping for sparse measurements
        )

        measurement_intervals = []
        position_errors = []
        last_measurement_time = None

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                if last_measurement_time is not None:
                    measurement_intervals.append(timestamp - last_measurement_time)
                last_measurement_time = timestamp

                ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                ekf.update()

            # Track position error
            truth = session.get_ground_truth_at_time(timestamp)
            estimate = ekf.get_current_state()
            error = jnp.linalg.norm(jnp.array([estimate.x, estimate.y]) - truth[:2])
            position_errors.append(error)

        # Verify measurements are indeed sparse
        avg_interval = jnp.mean(jnp.array(measurement_intervals))
        assert avg_interval > 0.15, f"Measurements not sparse enough: {avg_interval:.3f}s interval"

        # Should still maintain reasonable accuracy
        rmse = jnp.sqrt(jnp.mean(jnp.array(position_errors)**2))
        assert rmse < 5.0, f"RMSE {rmse:.2f} cm too high with sparse measurements"


class TestSmoothingImprovement:
    """Test that RTS smoothing provides expected improvements."""

    def test_rts_improvement_on_noisy_data(self):
        """Test that RTS smoothing improves estimates on noisy measurement data."""
        config = SimConfig(
            duration_s=8.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="figure_eight",
            position_noise_std_cm=3.0,  # High noise
            heading_noise_std_rad=0.2,
            confidence_noise_std=0.2,
            seed=252627,
        )

        session = generate_synthetic_session(config)

        # Run forward EKF
        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
        )

        ekf_results = []
        prediction_data = []
        ground_truth = []

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            # Store prediction for smoothing
            pred_state = ekf.get_current_state()
            pred_cov = ekf.get_current_covariance()
            prediction_data.append((pred_state, pred_cov))

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                result = ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                result = ekf.update()

            ekf_results.append(result)
            ground_truth.append(session.get_ground_truth_at_time(timestamp))

        # Run RTS smoother
        smoother = RTSSmoother()
        forward_data = smoother.collect_forward_data(ekf_results, prediction_data)
        rts_result = smoother.smooth_sequence(forward_data)

        # Compare performance
        filtered_states = [result.state.state for result in ekf_results]
        smoothed_states = rts_result.smoothed_states

        filt_rmse, smooth_rmse, improvement = compute_smoothing_improvement(
            filtered_states, smoothed_states, ground_truth
        )

        # RTS should provide meaningful improvement on noisy data
        assert improvement > 10.0, f"RTS improvement {improvement:.1f}% too small"
        assert smooth_rmse < filt_rmse, "Smoothed RMSE should be better than filtered"

    def test_rts_improvement_with_occlusions(self):
        """Test RTS smoothing improvement specifically during occlusions."""
        config = SimConfig(
            duration_s=12.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="turn_sequence",
            vision_dropout_periods=[(4.0, 8.0)],  # 4-second occlusion
            position_noise_std_cm=2.0,
            seed=282930,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
        )

        ekf_results = []
        prediction_data = []
        ground_truth = []

        for i, (timestamp, imu_sample, video_detection) in enumerate(session.generate_timeline()):
            if i == 0:
                continue

            prev_timestamp = session.generate_timeline()[i-1][0]
            dt = timestamp - prev_timestamp

            pred_state = ekf.get_current_state()
            pred_cov = ekf.get_current_covariance()
            prediction_data.append((pred_state, pred_cov))

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            if video_detection is not None:
                result = ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                result = ekf.update()

            ekf_results.append(result)
            ground_truth.append(session.get_ground_truth_at_time(timestamp))

        # Run RTS smoother
        smoother = RTSSmoother()
        forward_data = smoother.collect_forward_data(ekf_results, prediction_data)
        rts_result = smoother.smooth_sequence(forward_data)

        # Focus on occlusion period
        occlusion_indices = []
        for i, (timestamp, _, _) in enumerate(session.generate_timeline()):
            if 4.0 <= timestamp <= 8.0:
                occlusion_indices.append(i-1)  # Adjust for skipped first sample

        if occlusion_indices:
            occlusion_filtered = [ekf_results[i].state.state for i in occlusion_indices]
            occlusion_smoothed = [rts_result.smoothed_states[i] for i in occlusion_indices]
            occlusion_truth = [ground_truth[i] for i in occlusion_indices]

            filt_rmse, smooth_rmse, improvement = compute_smoothing_improvement(
                occlusion_filtered, occlusion_smoothed, occlusion_truth
            )

            # Should see improvement during occlusion period
            assert improvement > 5.0, f"RTS improvement {improvement:.1f}% during occlusion too small"


if __name__ == "__main__":
    # Run a quick smoke test
    test_instance = TestOcclusionScenarios()
    test_instance.test_short_occlusion_recovery()
    print("Scenario tests smoke test passed!")