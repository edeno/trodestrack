"""
Unit tests for QA metrics computation.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from trodestrack.qa.metrics import (
    compute_rmse,
    compute_nees,
    compute_position_nees,
    compute_occlusion_drift,
    evaluate_prd_compliance,
)


class TestRMSEMetrics:
    """Test RMSE computation functions."""

    def test_perfect_estimates(self):
        """Test RMSE with perfect estimates (should be zero)."""
        n_steps = 100
        states = jnp.ones((n_steps, 8))

        rmse = compute_rmse(states, states)

        assert rmse["position_rmse_cm"] == pytest.approx(0.0, abs=1e-6)
        assert rmse["velocity_rmse_cm_s"] == pytest.approx(0.0, abs=1e-6)
        assert rmse["heading_rmse_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_known_offset_rmse(self):
        """Test RMSE with known constant offset."""
        n_steps = 100

        # Ground truth: all zeros
        gt_states = jnp.zeros((n_steps, 8))

        # Estimated: constant offset
        est_states = jnp.zeros((n_steps, 8))
        est_states = est_states.at[:, 0].set(3.0)  # 3 cm offset in x
        est_states = est_states.at[:, 1].set(4.0)  # 4 cm offset in y
        est_states = est_states.at[:, 2].set(5.0)  # 5 cm/s offset in vx
        est_states = est_states.at[:, 3].set(12.0)  # 12 cm/s offset in vy
        est_states = est_states.at[:, 4].set(np.pi/6)  # 30 degree offset in heading

        rmse = compute_rmse(est_states, gt_states)

        # Position RMSE should be sqrt(3^2 + 4^2) = 5.0
        assert rmse["position_rmse_cm"] == pytest.approx(5.0, abs=1e-6)

        # Velocity RMSE should be sqrt(5^2 + 12^2) = 13.0
        assert rmse["velocity_rmse_cm_s"] == pytest.approx(13.0, abs=1e-6)

        # Heading RMSE should be 30 degrees
        assert rmse["heading_rmse_deg"] == pytest.approx(30.0, abs=1e-6)

    def test_angle_wrapping_rmse(self):
        """Test RMSE correctly handles angle wrapping."""
        n_steps = 100

        # Ground truth: 0 radians
        gt_states = jnp.zeros((n_steps, 8))

        # Estimated: just over π (should wrap to small negative angle)
        est_states = jnp.zeros((n_steps, 8))
        est_states = est_states.at[:, 4].set(np.pi + 0.1)  # Slightly over π

        rmse = compute_rmse(est_states, gt_states)

        # Should wrap to -π + 0.1, giving error magnitude of π - 0.1
        expected_error_rad = np.pi - 0.1
        expected_error_deg = expected_error_rad * 180.0 / np.pi

        assert rmse["heading_rmse_deg"] == pytest.approx(expected_error_deg, abs=1e-3)

    def test_rmse_with_mask(self):
        """Test RMSE computation with validity mask."""
        n_steps = 100

        gt_states = jnp.zeros((n_steps, 8))

        # First half: perfect estimates, second half: large errors
        est_states = jnp.zeros((n_steps, 8))
        est_states = est_states.at[50:, 0].set(10.0)  # Large error in second half

        # Mask to include only first half (perfect estimates)
        mask = jnp.arange(n_steps) < 50

        rmse = compute_rmse(est_states, gt_states, mask)

        # Should be zero since we masked out the errors
        assert rmse["position_rmse_cm"] == pytest.approx(0.0, abs=1e-6)

        # Without mask, should have significant error
        rmse_no_mask = compute_rmse(est_states, gt_states)
        assert rmse_no_mask["position_rmse_cm"] > 5.0


class TestNEESMetrics:
    """Test NEES computation functions."""

    def test_nees_perfect_filter(self):
        """Test NEES with perfectly calibrated filter (identity covariance)."""
        n_steps = 100

        # Ground truth: zeros
        gt_states = jnp.zeros((n_steps, 8))

        # Estimated: small random errors
        rng = np.random.RandomState(42)
        errors = rng.randn(n_steps, 8) * 0.1
        est_states = jnp.array(errors)

        # Covariances: identity (matches error magnitude)
        covariances = jnp.tile(jnp.eye(8) * 0.01, (n_steps, 1, 1))  # 0.1^2 = 0.01

        nees = compute_nees(est_states, gt_states, covariances)

        # NEES should be close to 8 (state dimension) for well-calibrated filter
        assert nees["nees_mean"] == pytest.approx(8.0, abs=2.0)
        assert nees["nees_consistency_ratio"] == pytest.approx(1.0, abs=0.25)

    def test_nees_underconfident_filter(self):
        """Test NEES with underconfident filter (too large covariances)."""
        n_steps = 100

        gt_states = jnp.zeros((n_steps, 8))

        # Small errors
        rng = np.random.RandomState(42)
        errors = rng.randn(n_steps, 8) * 0.1
        est_states = jnp.array(errors)

        # Large covariances (10x actual error variance)
        covariances = jnp.tile(jnp.eye(8) * 0.1, (n_steps, 1, 1))

        nees = compute_nees(est_states, gt_states, covariances)

        # NEES should be much less than 8 (underconfident)
        assert nees["nees_consistency_ratio"] < 0.5

    def test_position_nees_subset(self):
        """Test position-only NEES computation."""
        n_steps = 50

        gt_states = jnp.zeros((n_steps, 8))

        # Position errors only
        est_states = jnp.zeros((n_steps, 8))
        est_states = est_states.at[:, 0].set(0.1)  # x error
        est_states = est_states.at[:, 1].set(0.1)  # y error

        # Full covariances
        covariances = jnp.tile(jnp.eye(8) * 0.01, (n_steps, 1, 1))

        pos_nees = compute_position_nees(est_states, gt_states, covariances)

        # Position NEES should be close to 2 (position dimension)
        assert pos_nees["position_nees_mean"] == pytest.approx(2.0, abs=1.0)
        assert pos_nees["position_nees_expected"] == 2.0


class TestOcclusionDrift:
    """Test occlusion drift analysis."""

    def test_no_occlusions(self):
        """Test drift analysis with no occlusions."""
        n_steps = 100
        states = jnp.zeros((n_steps, 8))
        occlusion_mask = jnp.zeros(n_steps, dtype=bool)

        drift = compute_occlusion_drift(states, states, occlusion_mask)

        assert drift["num_occlusions"] == 0
        assert drift["mean_drift_cm"] == 0.0

    def test_single_occlusion_no_drift(self):
        """Test single occlusion with no drift."""
        n_steps = 100
        states = jnp.zeros((n_steps, 8))

        # Single occlusion in middle
        occlusion_mask = jnp.zeros(n_steps, dtype=bool)
        occlusion_mask = occlusion_mask.at[40:60].set(True)  # 20 frame occlusion

        drift = compute_occlusion_drift(states, states, occlusion_mask, dt=1.0/30.0)

        assert drift["num_occlusions"] == 1
        assert drift["mean_drift_cm"] == pytest.approx(0.0, abs=1e-6)

    def test_single_occlusion_with_drift(self):
        """Test single occlusion with known drift."""
        n_steps = 100

        # Ground truth: stationary
        gt_states = jnp.zeros((n_steps, 8))

        # Estimated: drift during occlusion
        est_states = jnp.zeros((n_steps, 8))
        # Add 10 cm drift in x during occlusion (frames 40-60)
        drift_frames = jnp.arange(40, 60)
        for i, frame in enumerate(drift_frames):
            est_states = est_states.at[frame, 0].set(i * 0.5)  # 0.5 cm per frame

        # Continue drift after occlusion ends
        est_states = est_states.at[60:, 0].set(10.0)  # Final 10 cm offset

        # Occlusion during drift period
        occlusion_mask = jnp.zeros(n_steps, dtype=bool)
        occlusion_mask = occlusion_mask.at[40:60].set(True)

        drift = compute_occlusion_drift(est_states, gt_states, occlusion_mask, dt=1.0/30.0)

        assert drift["num_occlusions"] == 1
        assert drift["mean_drift_cm"] == pytest.approx(10.0, abs=0.5)

    def test_multiple_occlusions(self):
        """Test multiple occlusion segments."""
        n_steps = 150

        gt_states = jnp.zeros((n_steps, 8))
        est_states = jnp.zeros((n_steps, 8))

        # Two occlusion periods with different drifts
        occlusion_mask = jnp.zeros(n_steps, dtype=bool)
        occlusion_mask = occlusion_mask.at[20:40].set(True)   # First occlusion: frames 20-39
        occlusion_mask = occlusion_mask.at[80:100].set(True)  # Second occlusion: frames 80-99

        # Simulate drift during first occlusion (from 0 to 5 cm during frames 20-39)
        for i in range(20, 40):
            progress = (i - 20) / (40 - 20)  # 0.0 to 1.0
            est_states = est_states.at[i, 0].set(progress * 5.0)

        # Continue with 5 cm error until second occlusion
        est_states = est_states.at[40:80, 0].set(5.0)

        # Simulate additional drift during second occlusion (from 5 to 8 cm during frames 80-99)
        for i in range(80, 100):
            progress = (i - 80) / (100 - 80)  # 0.0 to 1.0
            est_states = est_states.at[i, 0].set(5.0 + progress * 3.0)

        # Continue with 8 cm error after second occlusion
        est_states = est_states.at[100:, 0].set(8.0)

        drift = compute_occlusion_drift(est_states, gt_states, occlusion_mask, dt=1.0/30.0)

        assert drift["num_occlusions"] == 2
        # First occlusion should have ~5 cm drift, second should have ~3 cm drift
        assert drift["max_drift_cm"] >= 3.0  # At least one occlusion should have significant drift

    def test_short_occlusions_filtered(self):
        """Test that very short occlusions are filtered out."""
        n_steps = 100

        states = jnp.zeros((n_steps, 8))

        # Very short occlusion (2 frames)
        occlusion_mask = jnp.zeros(n_steps, dtype=bool)
        occlusion_mask = occlusion_mask.at[50:52].set(True)

        drift = compute_occlusion_drift(states, states, occlusion_mask, dt=1.0/30.0)

        # Should be filtered out as too short
        assert drift["num_occlusions"] == 0


class TestPRDCompliance:
    """Test PRD compliance evaluation."""

    def test_all_passing_metrics(self):
        """Test PRD evaluation with all passing metrics."""
        metrics = {
            "position_rmse_cm": 1.5,
            "velocity_rmse_cm_s": 8.0,
            "heading_rmse_deg": 5.0,
            "max_drift_cm": 10.0,
        }

        compliance = evaluate_prd_compliance(metrics)

        assert compliance["position_rmse_ok"]
        assert compliance["velocity_rmse_ok"]
        assert compliance["heading_rmse_ok"]
        assert compliance["occlusion_drift_ok"]
        assert compliance["overall_prd_compliant"]

    def test_some_failing_metrics(self):
        """Test PRD evaluation with some failing metrics."""
        metrics = {
            "position_rmse_cm": 2.5,  # Fails (> 2.0)
            "velocity_rmse_cm_s": 8.0,  # Passes
            "heading_rmse_deg": 5.0,   # Passes
            "max_drift_cm": 20.0,      # Fails (> 15.0)
        }

        compliance = evaluate_prd_compliance(metrics)

        assert not compliance["position_rmse_ok"]
        assert compliance["velocity_rmse_ok"]
        assert compliance["heading_rmse_ok"]
        assert not compliance["occlusion_drift_ok"]
        assert not compliance["overall_prd_compliant"]

    def test_missing_metrics(self):
        """Test PRD evaluation with missing metrics."""
        metrics = {
            "position_rmse_cm": 1.5,
            # Missing other metrics
        }

        compliance = evaluate_prd_compliance(metrics)

        assert compliance["position_rmse_ok"]
        # Missing metrics shouldn't affect overall if present ones pass
        assert "velocity_rmse_ok" not in compliance
        assert "overall_prd_compliant" in compliance

    def test_boundary_values(self):
        """Test PRD evaluation at exact threshold boundaries."""
        metrics = {
            "position_rmse_cm": 2.0,      # Exactly at threshold
            "velocity_rmse_cm_s": 10.0,   # Exactly at threshold
            "heading_rmse_deg": 7.0,      # Exactly at threshold
            "max_drift_cm": 15.0,         # Exactly at threshold
        }

        compliance = evaluate_prd_compliance(metrics)

        # At threshold should pass
        assert compliance["position_rmse_ok"]
        assert compliance["velocity_rmse_ok"]
        assert compliance["heading_rmse_ok"]
        assert compliance["occlusion_drift_ok"]
        assert compliance["overall_prd_compliant"]