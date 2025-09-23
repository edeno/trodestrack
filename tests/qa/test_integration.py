"""
Integration tests for QA system with simplified synthetic data.

These tests verify that the QA system works end-to-end with simple
synthetic data to validate core functionality.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from trodestrack.qa.metrics import (
    compute_rmse, compute_nees, compute_occlusion_drift, evaluate_prd_compliance
)
from trodestrack.qa.report import generate_qa_report


class TestQAIntegration:
    """Integration tests for QA system functionality."""

    def test_qa_report_generation_basic(self, tmp_path):
        """Test basic QA report generation with synthetic data."""
        # Create simple synthetic data
        n_steps = 200
        dt = 1.0 / 30.0  # 30 Hz
        timestamps = jnp.arange(n_steps) * dt

        # Ground truth: circular motion
        radius = 50.0  # cm
        angular_vel = 0.5  # rad/s
        angles = angular_vel * timestamps

        ground_truth_states = jnp.zeros((n_steps, 8))
        ground_truth_states = ground_truth_states.at[:, 0].set(radius * jnp.cos(angles))  # x
        ground_truth_states = ground_truth_states.at[:, 1].set(radius * jnp.sin(angles))  # y
        ground_truth_states = ground_truth_states.at[:, 2].set(-radius * angular_vel * jnp.sin(angles))  # vx
        ground_truth_states = ground_truth_states.at[:, 3].set(radius * angular_vel * jnp.cos(angles))   # vy
        ground_truth_states = ground_truth_states.at[:, 4].set(angles + jnp.pi/2)  # heading

        # Estimated states: ground truth + small noise
        rng = np.random.RandomState(42)
        noise_std = jnp.array([0.5, 0.5, 1.0, 1.0, 0.05, 0.001, 0.001, 0.001])  # Different noise for each component
        noise = rng.randn(n_steps, 8) * noise_std
        estimated_states = ground_truth_states + jnp.array(noise)

        # Covariances: identity matrices scaled appropriately
        base_cov = jnp.diag(jnp.array([1.0, 1.0, 2.0, 2.0, 0.1, 0.01, 0.001, 0.001]))  # Position, velocity, heading, biases
        covariances = jnp.tile(base_cov, (n_steps, 1, 1))

        # Create occlusion mask (some dropout periods)
        occlusion_mask = jnp.zeros(n_steps, dtype=bool)
        occlusion_mask = occlusion_mask.at[50:70].set(True)   # 20 frame occlusion
        occlusion_mask = occlusion_mask.at[150:170].set(True) # Another occlusion

        # Generate QA report
        result = generate_qa_report(
            estimated_states=estimated_states,
            ground_truth_states=ground_truth_states,
            covariances=covariances,
            output_dir=tmp_path,
            session_name="integration_test",
            timestamps=timestamps,
            occlusion_mask=occlusion_mask,
        )

        # Verify report structure
        assert "metrics" in result
        assert "plots" in result
        assert "data_summary" in result
        assert "summary_report" in result

        # Check metrics
        metrics = result["metrics"]
        assert "position_rmse_cm" in metrics
        assert "velocity_rmse_cm_s" in metrics
        assert "heading_rmse_deg" in metrics
        assert "nees_mean" in metrics
        assert "overall_prd_compliant" in metrics

        # Should pass PRD requirements with small synthetic noise
        assert metrics["position_rmse_cm"] < 2.0
        assert metrics["velocity_rmse_cm_s"] < 10.0
        assert metrics["heading_rmse_deg"] < 7.0

        # Check plots were generated
        plots = result["plots"]
        assert "trajectory" in plots
        assert "velocity_heading" in plots
        assert "nees" in plots
        assert "bias_traces" in plots

        # Verify files exist
        for plot_path in plots.values():
            assert Path(plot_path).exists()

        # Check occlusion analysis
        assert "num_occlusions" in metrics
        assert metrics["num_occlusions"] == 2  # Should find our two occlusion periods

    def test_prd_compliance_evaluation(self):
        """Test PRD compliance evaluation with known metrics."""
        # Test case 1: All metrics pass
        passing_metrics = {
            "position_rmse_cm": 1.5,
            "velocity_rmse_cm_s": 8.0,
            "heading_rmse_deg": 5.0,
            "max_drift_cm": 10.0,
        }

        compliance = evaluate_prd_compliance(passing_metrics)
        assert compliance["overall_prd_compliant"]
        assert compliance["position_rmse_ok"]
        assert compliance["velocity_rmse_ok"]
        assert compliance["heading_rmse_ok"]
        assert compliance["occlusion_drift_ok"]

        # Test case 2: Some metrics fail
        failing_metrics = {
            "position_rmse_cm": 3.0,    # Fails (> 2.0)
            "velocity_rmse_cm_s": 15.0, # Fails (> 10.0)
            "heading_rmse_deg": 4.0,    # Passes
            "max_drift_cm": 8.0,        # Passes
        }

        compliance = evaluate_prd_compliance(failing_metrics)
        assert not compliance["overall_prd_compliant"]
        assert not compliance["position_rmse_ok"]
        assert not compliance["velocity_rmse_ok"]
        assert compliance["heading_rmse_ok"]
        assert compliance["occlusion_drift_ok"]

    def test_metrics_consistency(self):
        """Test that metrics are computed consistently."""
        n_steps = 100

        # Perfect estimates
        states = jnp.ones((n_steps, 8))
        covariances = jnp.tile(jnp.eye(8), (n_steps, 1, 1))

        # RMSE with perfect estimates should be zero
        rmse = compute_rmse(states, states)
        assert rmse["position_rmse_cm"] == pytest.approx(0.0, abs=1e-6)
        assert rmse["velocity_rmse_cm_s"] == pytest.approx(0.0, abs=1e-6)
        assert rmse["heading_rmse_deg"] == pytest.approx(0.0, abs=1e-6)

        # NEES with perfect estimates and identity covariance
        nees = compute_nees(states, states, covariances)
        assert nees["nees_mean"] == pytest.approx(0.0, abs=1e-6)
        assert nees["nees_consistency_ratio"] == pytest.approx(0.0, abs=1e-6)

    def test_occlusion_drift_analysis(self):
        """Test occlusion drift analysis with controlled data."""
        n_steps = 150

        # Ground truth: stationary
        gt_states = jnp.zeros((n_steps, 8))

        # Estimated: drift during occlusion
        est_states = jnp.zeros((n_steps, 8))

        # Simulate drift during frames 50-80
        occlusion_mask = jnp.zeros(n_steps, dtype=bool)
        occlusion_mask = occlusion_mask.at[50:80].set(True)

        # Linear drift from 0 to 15 cm during occlusion
        for i in range(50, 80):
            progress = (i - 50) / (80 - 50)
            est_states = est_states.at[i, 0].set(progress * 15.0)

        # Continue with 15 cm offset after occlusion
        est_states = est_states.at[80:, 0].set(15.0)

        drift_metrics = compute_occlusion_drift(
            est_states, gt_states, occlusion_mask, dt=1.0/30.0
        )

        assert drift_metrics["num_occlusions"] == 1
        assert drift_metrics["max_drift_cm"] == pytest.approx(15.0, abs=0.5)
        assert drift_metrics["mean_drift_cm"] == pytest.approx(15.0, abs=0.5)

    def test_angle_wrapping_in_metrics(self):
        """Test that angle wrapping is handled correctly in RMSE."""
        n_steps = 10

        # Ground truth: 0 radians
        gt_states = jnp.zeros((n_steps, 8))

        # Estimated: angles near 2π (should wrap to small errors)
        est_states = jnp.zeros((n_steps, 8))
        est_states = est_states.at[:, 4].set(2 * jnp.pi - 0.1)  # Just under 2π

        rmse = compute_rmse(est_states, gt_states)

        # Should wrap to small error (~0.1 rad = ~5.7°), not large error (~354°)
        assert rmse["heading_rmse_deg"] < 10.0  # Should be ~5.7 degrees, not ~354

    def test_qa_system_robustness(self):
        """Test QA system handles edge cases robustly."""
        n_steps = 50

        # Test with minimal data
        small_states = jnp.zeros((n_steps, 8))
        small_covariances = jnp.tile(jnp.eye(8) * 0.01, (n_steps, 1, 1))

        rmse = compute_rmse(small_states, small_states)
        nees = compute_nees(small_states, small_states, small_covariances)

        # Should handle small datasets without errors
        assert isinstance(rmse["position_rmse_cm"], float)
        assert isinstance(nees["nees_mean"], float)

        # Test with missing occlusions
        no_occlusion_mask = jnp.zeros(n_steps, dtype=bool)
        drift = compute_occlusion_drift(small_states, small_states, no_occlusion_mask)

        assert drift["num_occlusions"] == 0
        assert drift["mean_drift_cm"] == 0.0

    def test_nees_filter_calibration_detection(self):
        """Test NEES can detect filter miscalibration."""
        n_steps = 100

        # Simulate filter with wrong uncertainty estimates
        gt_states = jnp.zeros((n_steps, 8))

        # Small estimation errors
        est_states = jnp.zeros((n_steps, 8))
        est_states = est_states.at[:, 0].set(0.1)  # 0.1 cm error

        # Case 1: Overconfident filter (covariance too small)
        small_covariances = jnp.tile(jnp.eye(8) * 0.001, (n_steps, 1, 1))  # Very small
        nees_overconfident = compute_nees(est_states, gt_states, small_covariances)

        # Should have high NEES (filter too confident)
        assert nees_overconfident["nees_consistency_ratio"] > 1.2

        # Case 2: Underconfident filter (covariance too large)
        large_covariances = jnp.tile(jnp.eye(8) * 10.0, (n_steps, 1, 1))  # Much larger
        nees_underconfident = compute_nees(est_states, gt_states, large_covariances)

        # Should have low NEES (filter not confident enough)
        assert nees_underconfident["nees_consistency_ratio"] < 0.2