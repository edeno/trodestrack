"""Benchmark tests for RTS smoother performance vs EKF.

This module tests the PRD requirement that RTS smoother should improve
RMSE by ≥20% vs EKF on synthetic "twitchy" sessions with rapid motion changes.
"""

import jax.numpy as jnp
import numpy as np

import trodestrack.sim.synthetic as synthetic_module
from trodestrack.models.ekf import EKFFilter
from trodestrack.models.rts_smoother import RTSSmoother, compute_smoothing_improvement
from trodestrack.sim.synthetic import SimConfig, generate_synthetic_session


class TestRTSBenchmark:
    """Test RTS smoother performance improvements."""

    def test_rts_improvement_twitchy_session(self):
        """Test that RTS improves RMSE ≥20% on synthetic twitchy session.

        Per PRD: RTS smoother improves RMSE ≥20% vs EKF on synthetic "twitchy" session.
        """
        # Create "twitchy" session with rapid motion changes
        config = SimConfig(
            duration_s=12.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="twitchy_motion",  # Will create rapid direction changes
            position_noise_std_cm=2.5,  # Moderate noise to make smoothing beneficial
            heading_noise_std_rad=0.15,
            confidence_noise_std=0.2,
            # Add some brief occlusions to make smoothing more valuable
            vision_dropout_periods=[(3.0, 3.5), (7.0, 7.5), (10.0, 10.3)],
            seed=987654,
        )

        session = generate_synthetic_session(config)

        # Run forward EKF pass
        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.1,
            position_noise_std=2.5,
            heading_noise_std=0.15,
        )

        ekf_results = []
        prediction_data = []
        ground_truth_states = []
        timeline = session.generate_timeline()

        for i, (timestamp, imu_sample, video_detection) in enumerate(timeline):
            if i == 0:
                continue  # Skip first sample

            prev_timestamp = timeline[i - 1][0]
            dt = timestamp - prev_timestamp

            # EKF prediction step
            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            # Store predicted state after prediction for RTS
            pred_state_array = ekf.get_current_state()
            pred_covariance = ekf.get_current_covariance()
            prediction_data.append(
                (
                    jnp.array(
                        [
                            pred_state_array.x,
                            pred_state_array.y,
                            pred_state_array.vx,
                            pred_state_array.vy,
                            pred_state_array.theta,
                            pred_state_array.b_gz,
                            pred_state_array.b_ax,
                            pred_state_array.b_ay,
                        ]
                    ),
                    pred_covariance,
                )
            )

            # EKF update step
            if video_detection is not None:
                result = ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                result = ekf.update()

            ekf_results.append(result)
            ground_truth_states.append(session.get_ground_truth_at_time(timestamp))

        # Run RTS smoother
        smoother = RTSSmoother(
            velocity_damping=0.1,
            accel_noise_std=0.5,
            gyro_noise_std=0.1,
            bias_drift_std=0.01,
        )

        forward_data = smoother.collect_forward_data(ekf_results, prediction_data)
        rts_result = smoother.smooth_sequence(forward_data)

        # Compare performance
        filtered_states = [result.state.state for result in ekf_results]
        smoothed_states = rts_result.smoothed_states

        filt_rmse, smooth_rmse, improvement_pct = compute_smoothing_improvement(
            filtered_states, smoothed_states, ground_truth_states
        )

        print("\nRTS Benchmark Results:")
        print(f"EKF RMSE: {filt_rmse:.2f} cm")
        print(f"RTS RMSE: {smooth_rmse:.2f} cm")
        print(f"Improvement: {improvement_pct:.1f}%")

        # Verify PRD requirement: ≥20% improvement
        assert improvement_pct >= 20.0, (
            f"RTS improvement {improvement_pct:.1f}% below 20% requirement. "
            f"EKF: {filt_rmse:.2f} cm, RTS: {smooth_rmse:.2f} cm"
        )

        # Additional sanity checks
        assert smooth_rmse < filt_rmse, "Smoothed RMSE should be better than filtered"
        assert smooth_rmse > 0, "Smoothed RMSE should be positive"
        assert filt_rmse < 20.0, f"EKF RMSE {filt_rmse:.2f} cm unexpectedly high"

    def test_rts_improvement_noisy_session(self):
        """Test RTS improvement on high-noise session."""
        config = SimConfig(
            duration_s=8.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="figure_eight",
            position_noise_std_cm=4.0,  # High noise
            heading_noise_std_rad=0.25,
            confidence_noise_std=0.3,
            seed=123456,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.08,
            position_noise_std=4.0,
            heading_noise_std=0.25,
        )

        ekf_results = []
        prediction_data = []
        ground_truth_states = []
        timeline = session.generate_timeline()

        for i, (timestamp, imu_sample, video_detection) in enumerate(timeline):
            if i == 0:
                continue

            prev_timestamp = timeline[i - 1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            pred_state_array = ekf.get_current_state()
            pred_covariance = ekf.get_current_covariance()
            prediction_data.append(
                (
                    jnp.array(
                        [
                            pred_state_array.x,
                            pred_state_array.y,
                            pred_state_array.vx,
                            pred_state_array.vy,
                            pred_state_array.theta,
                            pred_state_array.b_gz,
                            pred_state_array.b_ax,
                            pred_state_array.b_ay,
                        ]
                    ),
                    pred_covariance,
                )
            )

            if video_detection is not None:
                result = ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                result = ekf.update()

            ekf_results.append(result)
            ground_truth_states.append(session.get_ground_truth_at_time(timestamp))

        # Run RTS smoother
        smoother = RTSSmoother()
        forward_data = smoother.collect_forward_data(ekf_results, prediction_data)
        rts_result = smoother.smooth_sequence(forward_data)

        # Compare performance
        filtered_states = [result.state.state for result in ekf_results]
        smoothed_states = rts_result.smoothed_states

        filt_rmse, smooth_rmse, improvement_pct = compute_smoothing_improvement(
            filtered_states, smoothed_states, ground_truth_states
        )

        print("\nHigh-Noise RTS Results:")
        print(f"EKF RMSE: {filt_rmse:.2f} cm")
        print(f"RTS RMSE: {smooth_rmse:.2f} cm")
        print(f"Improvement: {improvement_pct:.1f}%")

        # Should see significant improvement on noisy data
        assert (
            improvement_pct >= 15.0
        ), f"RTS improvement {improvement_pct:.1f}% too small on noisy data"

    def test_rts_improvement_with_occlusions(self):
        """Test RTS improvement specifically during and after occlusions."""
        config = SimConfig(
            duration_s=15.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="circular",
            position_noise_std_cm=2.0,
            heading_noise_std_rad=0.1,
            # Extended occlusion to test smoothing benefits
            vision_dropout_periods=[(6.0, 10.0)],  # 4-second occlusion
            seed=789123,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.12,
        )

        ekf_results = []
        prediction_data = []
        ground_truth_states = []
        timeline = session.generate_timeline()

        for i, (timestamp, imu_sample, video_detection) in enumerate(timeline):
            if i == 0:
                continue

            prev_timestamp = timeline[i - 1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            pred_state_array = ekf.get_current_state()
            pred_covariance = ekf.get_current_covariance()
            prediction_data.append(
                (
                    jnp.array(
                        [
                            pred_state_array.x,
                            pred_state_array.y,
                            pred_state_array.vx,
                            pred_state_array.vy,
                            pred_state_array.theta,
                            pred_state_array.b_gz,
                            pred_state_array.b_ax,
                            pred_state_array.b_ay,
                        ]
                    ),
                    pred_covariance,
                )
            )

            if video_detection is not None:
                result = ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                result = ekf.update()

            ekf_results.append(result)
            ground_truth_states.append(session.get_ground_truth_at_time(timestamp))

        # Run RTS smoother
        smoother = RTSSmoother()
        forward_data = smoother.collect_forward_data(ekf_results, prediction_data)
        rts_result = smoother.smooth_sequence(forward_data)

        # Analyze improvement specifically during occlusion period
        occlusion_indices = []
        for i, (timestamp, _, _) in enumerate(timeline[1:]):  # Skip first
            if 6.0 <= timestamp <= 10.0:
                occlusion_indices.append(i)

        if occlusion_indices:
            occlusion_filtered = [ekf_results[i].state.state for i in occlusion_indices]
            occlusion_smoothed = [rts_result.smoothed_states[i] for i in occlusion_indices]
            occlusion_truth = [ground_truth_states[i] for i in occlusion_indices]

            occ_filt_rmse, occ_smooth_rmse, occ_improvement = compute_smoothing_improvement(
                occlusion_filtered, occlusion_smoothed, occlusion_truth
            )

            print("\nOcclusion Period RTS Results:")
            print(f"EKF RMSE (occlusion): {occ_filt_rmse:.2f} cm")
            print(f"RTS RMSE (occlusion): {occ_smooth_rmse:.2f} cm")
            print(f"Improvement (occlusion): {occ_improvement:.1f}%")

            # Should see improvement during occlusion
            assert (
                occ_improvement >= 10.0
            ), f"RTS improvement {occ_improvement:.1f}% too small during occlusion"

        # Overall improvement
        filtered_states = [result.state.state for result in ekf_results]
        smoothed_states = rts_result.smoothed_states

        filt_rmse, smooth_rmse, improvement_pct = compute_smoothing_improvement(
            filtered_states, smoothed_states, ground_truth_states
        )

        print("\nOverall RTS Results:")
        print(f"EKF RMSE: {filt_rmse:.2f} cm")
        print(f"RTS RMSE: {smooth_rmse:.2f} cm")
        print(f"Improvement: {improvement_pct:.1f}%")

        assert (
            improvement_pct >= 12.0
        ), f"Overall RTS improvement {improvement_pct:.1f}% too small with occlusions"

    def test_rts_velocity_improvement(self):
        """Test that RTS also improves velocity estimates."""
        config = SimConfig(
            duration_s=10.0,
            camera_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="turn_sequence",
            position_noise_std_cm=2.0,
            heading_noise_std_rad=0.12,
            seed=456789,
        )

        session = generate_synthetic_session(config)

        ekf = EKFFilter(
            initial_state=session.initial_state,
            initial_covariance=jnp.eye(8) * 0.1,
            velocity_damping=0.1,
        )

        ekf_results = []
        prediction_data = []
        ground_truth_states = []
        timeline = session.generate_timeline()

        for i, (timestamp, imu_sample, video_detection) in enumerate(timeline):
            if i == 0:
                continue

            prev_timestamp = timeline[i - 1][0]
            dt = timestamp - prev_timestamp

            ekf.predict(dt, imu_sample.accel, imu_sample.gyro)

            pred_state_array = ekf.get_current_state()
            pred_covariance = ekf.get_current_covariance()
            prediction_data.append(
                (
                    jnp.array(
                        [
                            pred_state_array.x,
                            pred_state_array.y,
                            pred_state_array.vx,
                            pred_state_array.vy,
                            pred_state_array.theta,
                            pred_state_array.b_gz,
                            pred_state_array.b_ax,
                            pred_state_array.b_ay,
                        ]
                    ),
                    pred_covariance,
                )
            )

            if video_detection is not None:
                result = ekf.update(
                    position=video_detection.position_cm,
                    heading=video_detection.heading_rad,
                    confidence=video_detection.confidence,
                )
            else:
                result = ekf.update()

            ekf_results.append(result)
            ground_truth_states.append(session.get_ground_truth_at_time(timestamp))

        # Run RTS smoother
        smoother = RTSSmoother()
        forward_data = smoother.collect_forward_data(ekf_results, prediction_data)
        rts_result = smoother.smooth_sequence(forward_data)

        # Compute velocity RMSE
        filtered_vel_errors = []
        smoothed_vel_errors = []

        for i, gt in enumerate(ground_truth_states):
            # EKF velocity error
            ekf_vel = ekf_results[i].state.state[2:4]  # [vx, vy]
            gt_vel = gt[2:4]
            filt_vel_error = jnp.linalg.norm(ekf_vel - gt_vel)
            filtered_vel_errors.append(filt_vel_error)

            # RTS velocity error
            rts_vel = rts_result.smoothed_states[i][2:4]
            smooth_vel_error = jnp.linalg.norm(rts_vel - gt_vel)
            smoothed_vel_errors.append(smooth_vel_error)

        filt_vel_rmse = jnp.sqrt(jnp.mean(jnp.array(filtered_vel_errors) ** 2))
        smooth_vel_rmse = jnp.sqrt(jnp.mean(jnp.array(smoothed_vel_errors) ** 2))
        vel_improvement = (filt_vel_rmse - smooth_vel_rmse) / filt_vel_rmse * 100.0

        print("\nVelocity Estimation Results:")
        print(f"EKF velocity RMSE: {filt_vel_rmse:.2f} cm/s")
        print(f"RTS velocity RMSE: {smooth_vel_rmse:.2f} cm/s")
        print(f"Velocity improvement: {vel_improvement:.1f}%")

        # Should see some improvement in velocity estimates
        assert vel_improvement >= 5.0, f"RTS velocity improvement {vel_improvement:.1f}% too small"


# Extend synthetic data generator to support "twitchy" motion
def _generate_twitchy_trajectory(self):
    """Generate trajectory with rapid direction changes (twitchy motion)."""
    dt = 1.0 / self.config.camera_fps
    n_steps = int(self.config.duration_s * self.config.camera_fps)

    self.timestamps = np.linspace(0, self.config.duration_s, n_steps)
    self.ground_truth_positions = np.zeros((n_steps, 2))
    self.ground_truth_velocities = np.zeros((n_steps, 2))
    self.ground_truth_headings = np.zeros(n_steps)

    # Twitchy motion with frequent direction changes
    pos = np.array([0.0, 0.0])
    vel = np.array([25.0, 0.0])
    heading = 0.0

    for i in range(n_steps):
        self.ground_truth_positions[i] = pos
        self.ground_truth_velocities[i] = vel
        self.ground_truth_headings[i] = heading

        # Random direction changes every 0.5-1.5 seconds
        if i > 0 and self.rng.random() < 0.02:  # ~2% chance per frame
            # Sharp turn
            turn_angle = self.rng.uniform(-np.pi / 2, np.pi / 2)
            heading += turn_angle

            # Speed variation
            speed = self.rng.uniform(15.0, 35.0)
            vel = speed * np.array([np.cos(heading), np.sin(heading)])

        pos += vel * dt


# Monkey patch the trajectory generation to support twitchy motion


original_generate_trajectory = synthetic_module.SyntheticSessionResult._generate_trajectory


def patched_generate_trajectory(self):
    if self.config.trajectory_type == "twitchy_motion":
        _generate_twitchy_trajectory(self)
    else:
        original_generate_trajectory(self)


synthetic_module.SyntheticSessionResult._generate_trajectory = patched_generate_trajectory


if __name__ == "__main__":
    # Run the benchmark test
    test_instance = TestRTSBenchmark()
    test_instance.test_rts_improvement_twitchy_session()
    print("RTS benchmark test passed!")
