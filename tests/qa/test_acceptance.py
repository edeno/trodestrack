"""
Acceptance tests for QA metrics on synthetic datasets with PRD thresholds.

These tests validate that the filtering and smoothing algorithms meet the
Project Requirements Document (PRD) performance criteria on synthetic data.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from trodestrack.sim.synthetic import SimConfig
from trodestrack.runtime.offline import smooth_session
from trodestrack.config.schemas import SessionConfig, FilterConfig, OutputConfig
from trodestrack.qa.metrics import (
    compute_rmse,
    compute_nees,
    compute_occlusion_drift,
    evaluate_prd_compliance,
)


# Placeholder for future implementation
class SyntheticDataGenerator:
    """Placeholder class for future synthetic data generation."""

    def __init__(self, config):
        self.config = config

    def generate_session(self):
        """Placeholder method."""
        raise NotImplementedError("SyntheticDataGenerator not implemented yet")


@pytest.mark.skip(
    reason="SyntheticDataGenerator not implemented - placeholder for future acceptance tests"
)
class TestPRDAcceptance:
    """Test PRD compliance on synthetic datasets."""

    @pytest.fixture
    def prd_compliant_config(self) -> SimConfig:
        """Create simulation config expected to meet PRD requirements."""
        return SimConfig(
            duration_s=30.0,
            video_fps=30.0,
            imu_rate_hz=1000.0,  # Reduced from full rate for testing efficiency
            trajectory_type="mixed",
            # Low noise for PRD compliance
            accel_noise_std=0.02,  # m/s^2 (reduced from default)
            gyro_noise_std=0.005,  # rad/s (reduced)
            video_position_noise_std=0.5,  # cm (reduced)
            video_confidence_noise_std=0.02,
            # Minimal bias drift
            accel_bias_std=0.001,  # m/s^2
            gyro_bias_std=0.0005,  # rad/s
            # Short occlusions for testing
            occlusion_probability=0.1,
            occlusion_duration_range=(1.0, 3.0),  # 1-3 second occlusions
            seed=42,  # Fixed seed for reproducibility
        )

    @pytest.fixture
    def challenging_config(self) -> SimConfig:
        """Create more challenging simulation config."""
        return SimConfig(
            duration_s=60.0,
            video_fps=30.0,
            imu_rate_hz=1000.0,
            trajectory_type="twitchy",
            # Higher noise - near PRD limits
            accel_noise_std=0.05,  # m/s^2
            gyro_noise_std=0.01,  # rad/s
            video_position_noise_std=1.0,  # cm
            video_confidence_noise_std=0.05,
            # More bias drift
            accel_bias_std=0.002,  # m/s^2
            gyro_bias_std=0.001,  # rad/s
            # Longer occlusions
            occlusion_probability=0.15,
            occlusion_duration_range=(2.0, 5.0),  # 2-5 second occlusions
            seed=123,
        )

    @pytest.mark.skip(reason="SyntheticDataGenerator not implemented")
    def test_ekf_prd_compliance_clean_data(self, prd_compliant_config, tmp_path):
        """Test EKF meets PRD requirements on clean synthetic data."""
        # Generate synthetic session
        generator = SyntheticDataGenerator(prd_compliant_config)
        session = generator.generate_session()

        # Save to temporary files
        video_file = tmp_path / "video.npz"
        imu_file = tmp_path / "imu.npz"
        session.save_npz(video_file, imu_file)

        # Create session config for EKF
        session_config = SessionConfig(
            video_file=video_file,
            imu_file=imu_file,
            filter=FilterConfig(
                filter_type="ekf",
                process_noise_std={
                    "velocity_x": 0.5,  # cm/s
                    "velocity_y": 0.5,
                    "heading": 0.1,  # rad
                    "gyro_bias": 0.001,  # rad/s
                    "accel_bias_x": 0.01,  # m/s^2
                    "accel_bias_y": 0.01,
                },
                measurement_noise_std={
                    "position": 1.0,  # cm
                    "heading": 0.1,  # rad
                },
                velocity_damping=0.1,
            ),
            output=OutputConfig(
                output_dir=tmp_path / "output",
                save_states=True,
                save_plots=False,
            ),
        )

        # Run offline smoothing
        result = smooth_session(session_config)

        # Extract ground truth for comparison
        ground_truth_states = session.true_states

        # Compute RMSE metrics
        rmse_metrics = compute_rmse(
            result.smoothed_states,
            ground_truth_states,
        )

        # Compute NEES for filter consistency
        nees_metrics = compute_nees(
            result.filtered_states,
            ground_truth_states,
            result.filtered_covariances,
        )

        # Compute occlusion drift
        occlusion_mask = session.video_data.confidence < 0.1  # Low confidence = occlusion
        drift_metrics = compute_occlusion_drift(
            result.smoothed_states,
            ground_truth_states,
            occlusion_mask,
            dt=1.0 / session.metadata["video_fps"],
        )

        # Combine all metrics
        all_metrics = {**rmse_metrics, **nees_metrics, **drift_metrics}

        # Evaluate PRD compliance
        compliance = evaluate_prd_compliance(all_metrics)

        # Assert PRD compliance
        assert compliance["position_rmse_ok"], (
            f"Position RMSE {rmse_metrics['position_rmse_cm']:.2f} cm exceeds "
            f"PRD threshold of 2.0 cm"
        )

        assert compliance["velocity_rmse_ok"], (
            f"Velocity RMSE {rmse_metrics['velocity_rmse_cm_s']:.2f} cm/s exceeds "
            f"PRD threshold of 10.0 cm/s"
        )

        assert compliance["heading_rmse_ok"], (
            f"Heading RMSE {rmse_metrics['heading_rmse_deg']:.2f}° exceeds "
            f"PRD threshold of 7.0°"
        )

        if drift_metrics["num_occlusions"] > 0:
            assert compliance["occlusion_drift_ok"], (
                f"Max occlusion drift {drift_metrics['max_drift_cm']:.2f} cm exceeds "
                f"PRD threshold of 15.0 cm"
            )

        # Filter consistency checks (NEES should be close to expected value)
        nees_ratio = nees_metrics["nees_consistency_ratio"]
        assert (
            0.7 <= nees_ratio <= 1.5
        ), f"NEES consistency ratio {nees_ratio:.3f} indicates poorly calibrated filter"

        # Overall compliance
        assert compliance["overall_prd_compliant"], "Overall PRD compliance failed"

    @pytest.mark.skip(reason="SyntheticDataGenerator not implemented")
    def test_rts_smoother_improvement(self, prd_compliant_config, tmp_path):
        """Test RTS smoother provides significant improvement over EKF."""
        # Generate synthetic session
        generator = SyntheticDataGenerator(prd_compliant_config)
        session = generator.generate_session()

        # Save to temporary files
        video_file = tmp_path / "video.npz"
        imu_file = tmp_path / "imu.npz"
        session.save_npz(video_file, imu_file)

        # Create session config
        session_config = SessionConfig(
            video_file=video_file,
            imu_file=imu_file,
            filter=FilterConfig(filter_type="ekf"),
            output=OutputConfig(
                output_dir=tmp_path / "output",
                save_states=True,
                save_plots=False,
            ),
        )

        # Run offline smoothing
        result = smooth_session(session_config)
        ground_truth_states = session.true_states

        # Compute RMSE for filtered and smoothed estimates
        filtered_rmse = compute_rmse(result.filtered_states, ground_truth_states)
        smoothed_rmse = compute_rmse(result.smoothed_states, ground_truth_states)

        # RTS smoother should improve position RMSE
        improvement = filtered_rmse["position_rmse_cm"] - smoothed_rmse["position_rmse_cm"]
        improvement_percent = 100 * improvement / filtered_rmse["position_rmse_cm"]

        # Assert significant improvement (at least 10% for clean data)
        assert improvement > 0, "RTS smoother should improve upon EKF estimates"
        assert (
            improvement_percent >= 10.0
        ), f"RTS smoother improvement {improvement_percent:.1f}% below expected minimum of 10%"

    @pytest.mark.skip(reason="SyntheticDataGenerator not implemented")
    def test_occlusion_robustness(self, challenging_config, tmp_path):
        """Test robustness during challenging occlusion periods."""
        # Modify config for more challenging occlusions
        challenging_config.occlusion_probability = 0.2
        challenging_config.occlusion_duration_range = (3.0, 6.0)

        # Generate synthetic session
        generator = SyntheticDataGenerator(challenging_config)
        session = generator.generate_session()

        # Save to temporary files
        video_file = tmp_path / "video.npz"
        imu_file = tmp_path / "imu.npz"
        session.save_npz(video_file, imu_file)

        # Create session config with robust settings
        session_config = SessionConfig(
            video_file=video_file,
            imu_file=imu_file,
            filter=FilterConfig(
                filter_type="ekf",
                process_noise_std={
                    "velocity_x": 1.0,  # Higher for robustness
                    "velocity_y": 1.0,
                    "heading": 0.2,
                    "gyro_bias": 0.002,
                    "accel_bias_x": 0.02,
                    "accel_bias_y": 0.02,
                },
                velocity_damping=0.05,  # Lighter damping during occlusions
            ),
            output=OutputConfig(
                output_dir=tmp_path / "output",
                save_states=True,
                save_plots=False,
            ),
        )

        # Run offline smoothing
        result = smooth_session(session_config)
        ground_truth_states = session.true_states

        # Analyze occlusion periods
        occlusion_mask = session.video_data.confidence < 0.1
        drift_metrics = compute_occlusion_drift(
            result.smoothed_states,
            ground_truth_states,
            occlusion_mask,
            dt=1.0 / session.metadata["video_fps"],
            max_drift_duration=7.0,  # Analyze up to 7s occlusions
        )

        # Assert reasonable drift bounds even for challenging conditions
        if drift_metrics["num_occlusions"] > 0:
            # Allow higher drift for challenging conditions, but still bounded
            max_acceptable_drift = 20.0  # cm (relaxed from PRD 15cm for challenging test)
            assert drift_metrics["max_drift_cm"] <= max_acceptable_drift, (
                f"Max drift {drift_metrics['max_drift_cm']:.2f} cm exceeds "
                f"acceptable limit of {max_acceptable_drift} cm for challenging conditions"
            )

            # Mean drift should be much better
            assert (
                drift_metrics["mean_drift_cm"] <= 10.0
            ), f"Mean drift {drift_metrics['mean_drift_cm']:.2f} cm too high"

    @pytest.mark.skip(reason="SyntheticDataGenerator not implemented")
    def test_filter_consistency_nees(self, prd_compliant_config, tmp_path):
        """Test filter consistency using NEES analysis."""
        # Generate synthetic session
        generator = SyntheticDataGenerator(prd_compliant_config)
        session = generator.generate_session()

        # Save to temporary files
        video_file = tmp_path / "video.npz"
        imu_file = tmp_path / "imu.npz"
        session.save_npz(video_file, imu_file)

        # Create session config
        session_config = SessionConfig(
            video_file=video_file,
            imu_file=imu_file,
            filter=FilterConfig(filter_type="ekf"),
            output=OutputConfig(
                output_dir=tmp_path / "output",
                save_states=True,
                save_plots=False,
            ),
        )

        # Run offline smoothing
        result = smooth_session(session_config)
        ground_truth_states = session.true_states

        # Compute NEES metrics
        full_nees = compute_nees(
            result.filtered_states,
            ground_truth_states,
            result.filtered_covariances,
        )

        # Test full state NEES consistency
        nees_ratio = full_nees["nees_consistency_ratio"]

        # Filter should be reasonably well-calibrated (within 30% of ideal)
        assert 0.7 <= nees_ratio <= 1.3, (
            f"NEES ratio {nees_ratio:.3f} indicates poor filter calibration " f"(expected 0.7-1.3)"
        )

        # NEES values should not have excessive outliers
        nees_values = full_nees["nees_values"]
        outlier_threshold = 30.0  # Conservative threshold for 8-DOF chi-squared
        outlier_fraction = np.mean(nees_values > outlier_threshold)

        assert outlier_fraction <= 0.05, (
            f"Too many NEES outliers: {outlier_fraction*100:.1f}% exceed threshold "
            f"(expected ≤5%)"
        )

    @pytest.mark.skip(reason="SyntheticDataGenerator not implemented")
    @pytest.mark.parametrize("filter_type", ["ekf", "ukf"])
    def test_filter_comparison_prd(self, filter_type, prd_compliant_config, tmp_path):
        """Test both EKF and UKF meet PRD requirements."""
        # Generate synthetic session
        generator = SyntheticDataGenerator(prd_compliant_config)
        session = generator.generate_session()

        # Save to temporary files
        video_file = tmp_path / "video.npz"
        imu_file = tmp_path / "imu.npz"
        session.save_npz(video_file, imu_file)

        # Create session config
        session_config = SessionConfig(
            video_file=video_file,
            imu_file=imu_file,
            filter=FilterConfig(filter_type=filter_type),
            output=OutputConfig(
                output_dir=tmp_path / "output",
                save_states=True,
                save_plots=False,
            ),
        )

        # Run offline smoothing
        result = smooth_session(session_config)
        ground_truth_states = session.true_states

        # Compute metrics
        rmse_metrics = compute_rmse(result.smoothed_states, ground_truth_states)

        # Evaluate PRD compliance
        compliance = evaluate_prd_compliance(rmse_metrics)

        # Both filters should meet PRD requirements on clean data
        assert compliance["position_rmse_ok"], (
            f"{filter_type.upper()} position RMSE {rmse_metrics['position_rmse_cm']:.2f} cm "
            f"exceeds PRD threshold"
        )

        assert compliance["velocity_rmse_ok"], (
            f"{filter_type.upper()} velocity RMSE {rmse_metrics['velocity_rmse_cm_s']:.2f} cm/s "
            f"exceeds PRD threshold"
        )

        assert compliance["heading_rmse_ok"], (
            f"{filter_type.upper()} heading RMSE {rmse_metrics['heading_rmse_deg']:.2f}° "
            f"exceeds PRD threshold"
        )


class TestQADiagnostics:
    """Test QA diagnostic capabilities."""

    def test_prd_threshold_validation(self):
        """Test PRD threshold evaluation with known metrics."""
        # Test passing metrics
        passing_metrics = {
            "position_rmse_cm": 1.5,
            "velocity_rmse_cm_s": 8.0,
            "heading_rmse_deg": 5.0,
            "max_drift_cm": 12.0,
        }

        compliance = evaluate_prd_compliance(passing_metrics)
        assert compliance["overall_prd_compliant"]
        assert all(compliance.values())

        # Test failing metrics
        failing_metrics = {
            "position_rmse_cm": 3.0,  # Exceeds 2.0
            "velocity_rmse_cm_s": 12.0,  # Exceeds 10.0
            "heading_rmse_deg": 9.0,  # Exceeds 7.0
            "max_drift_cm": 20.0,  # Exceeds 15.0
        }

        compliance = evaluate_prd_compliance(failing_metrics)
        assert not compliance["overall_prd_compliant"]
        assert not any(
            [
                compliance["position_rmse_ok"],
                compliance["velocity_rmse_ok"],
                compliance["heading_rmse_ok"],
                compliance["occlusion_drift_ok"],
            ]
        )

    def test_drift_analysis_empty_occlusions(self):
        """Test occlusion drift analysis with no occlusions."""
        # Create simple synthetic states
        n_steps = 100
        states = jnp.zeros((n_steps, 8))
        occlusion_mask = jnp.zeros(n_steps, dtype=bool)  # No occlusions

        drift_metrics = compute_occlusion_drift(states, states, occlusion_mask)

        assert drift_metrics["num_occlusions"] == 0
        assert drift_metrics["mean_drift_cm"] == 0.0
        assert drift_metrics["max_drift_cm"] == 0.0

    def test_metrics_with_missing_data(self):
        """Test metrics computation with partially missing data."""
        n_steps = 50

        # Create test data with some variation
        estimated_states = jnp.ones((n_steps, 8)) * 1.0
        ground_truth_states = jnp.ones((n_steps, 8)) * 1.1  # Small offset

        # Mask for valid data (exclude first and last 10 timesteps)
        mask = jnp.arange(n_steps)
        mask = (mask >= 10) & (mask < 40)

        rmse_metrics = compute_rmse(estimated_states, ground_truth_states, mask)

        # Should compute metrics only on valid data
        assert "position_rmse_cm" in rmse_metrics
        assert rmse_metrics["position_rmse_cm"] > 0  # Should detect the offset
        assert rmse_metrics["position_rmse_cm"] < 1.0  # But be reasonably small
