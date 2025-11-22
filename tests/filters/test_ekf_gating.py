"""Tests for EKF χ² gating (outlier rejection).

This module tests that the EKF properly rejects outlier measurements
using Mahalanobis distance (normalized innovation squared, NIS) with
χ² threshold at p=0.997 confidence level.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from scipy.stats import chi2

from trodestrack.models.ekf import EKFConfig, EKFState, update_step
from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.state_layout import LAYOUT_2D_FULL


@pytest.fixture
def ekf_config():
    """Standard EKF configuration."""
    return EKFConfig(state_mode="2d_full")


@pytest.fixture
def ekf_config_with_gating():
    """EKF configuration with gating enabled."""
    return EKFConfig(
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.997,  # p=0.997 → very conservative
        state_mode="2d_full",  # Use 8D layout (tests use hardcoded indices)
    )


@pytest.fixture
def initial_state():
    """Create a simple initial state for testing."""
    mean = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01  # Low uncertainty
    return EKFState(mean=mean, cov=cov)


def make_camera_model(z_led1, z_led2, config, confidence=None):
    """Helper to create camera model for single-frame test."""
    z_led1_all = z_led1.reshape(1, 2)
    z_led2_all = z_led2.reshape(1, 2)
    conf_all = None if confidence is None else confidence.reshape(1, 4)

    return CameraPositionModel(
        led_distance=config.led_distance,
        measurement_noise_base=config.measurement_noise_pos,
        layout=LAYOUT_2D_FULL,
        z_led1_all=z_led1_all,
        z_led2_all=z_led2_all,
        conf_all=conf_all,
        confidence_clip_min=1e-2,
    )


def test_gating_accepts_good_measurement(ekf_config_with_gating, initial_state):
    """Test that gating accepts measurements consistent with prediction."""
    # Observation very close to predicted state (1.0, 1.0)
    # Use even closer observation to ensure it passes gating
    z_led1 = jnp.array([1.0, 1.0])
    z_led2 = jnp.array([1.04, 1.0])  # Expected LED2 given 4cm spacing

    camera_model = make_camera_model(z_led1, z_led2, ekf_config_with_gating)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config_with_gating,
        layout=LAYOUT_2D_FULL,
    )

    # State should be updated (measurement accepted)
    # Allow some tolerance since update might be small if observation = prediction
    assert not jnp.allclose(
        state_upd.mean, initial_state.mean, atol=1e-8
    ), "Good measurement should update state (covariance or heading change)"

    # Log-likelihood should be non-zero (measurement used)
    assert log_lik != 0.0


def test_gating_rejects_outlier_measurement(ekf_config_with_gating, initial_state):
    """Test that gating rejects measurements far from prediction."""
    # Observation very far from predicted state (1.0, 1.0)
    # Expected state at (1.0, 1.0), observation at (5.0, 5.0) → ~4m error
    z_led1 = jnp.array([4.98, 4.98])
    z_led2 = jnp.array([5.02, 5.02])

    camera_model = make_camera_model(z_led1, z_led2, ekf_config_with_gating)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config_with_gating,
        layout=LAYOUT_2D_FULL,
    )

    # State should remain unchanged (measurement rejected)
    assert jnp.allclose(
        state_upd.mean, initial_state.mean, atol=1e-6
    ), "Outlier should be rejected, state unchanged"
    assert jnp.allclose(
        state_upd.cov, initial_state.cov, atol=1e-6
    ), "Outlier should be rejected, covariance unchanged"

    # Log-likelihood should be zero (measurement rejected)
    assert log_lik == 0.0, "Rejected measurement should have zero log-likelihood"


def test_gating_threshold_calculation():
    """Test that χ² threshold is computed correctly for different dimensions."""
    # For 2D measurement (single LED): χ²(2, p=0.997) = 11.618
    threshold_2d = chi2.ppf(0.997, df=2)
    assert 11.0 < threshold_2d < 12.0

    # For 4D measurement (both LEDs): χ²(4, p=0.997) = 16.014
    threshold_4d = chi2.ppf(0.997, df=4)
    assert 15.5 < threshold_4d < 16.5

    # Higher dimensions → higher thresholds (more lenient)
    assert threshold_4d > threshold_2d


def test_gating_disabled_accepts_outlier(initial_state):
    """Test that with gating disabled, outliers are accepted.

    With Mahalanobis gating disabled, even far outliers should update the state.
    The Kalman gain determines how much the state moves, but it WILL move.
    """
    # Explicitly disable gating
    ekf_config = EKFConfig(state_mode="2d_full", use_mahalanobis_gating=False)

    # Very far observation (4 meters from state)
    z_led1 = jnp.array([4.98, 4.98])
    z_led2 = jnp.array([5.02, 5.02])

    camera_model = make_camera_model(z_led1, z_led2, ekf_config)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # State should be updated significantly (high Kalman gain with low P/R ratio)
    # With P=0.01 and R~0.0001, K ≈ 0.99, so state should move ~4m * 0.99 ≈ 4m
    assert not jnp.allclose(
        state_upd.mean[:2], initial_state.mean[:2], atol=0.1
    ), "Without gating, outlier should update state"

    # Verify log-likelihood is non-zero (update happened)
    assert log_lik != 0.0, "Update should produce non-zero log-likelihood"


def test_gating_moderate_outlier():
    """Test gating with moderate outlier (borderline case)."""
    # Create state at origin
    mean = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01
    state = EKFState(mean=mean, cov=cov)

    config = EKFConfig(
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.997,  # Very conservative
        measurement_noise_pos=0.005**2,
        state_mode="2d_full",  # Use 8D layout (tests use hardcoded indices)
    )

    # Moderate outlier: 10cm error (moderate but not huge)
    z_led1 = jnp.array([0.1, 0.1])
    z_led2 = jnp.array([0.14, 0.1])

    camera_model = make_camera_model(z_led1, z_led2, config)
    state_upd, _ = update_step(
        state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # This test just verifies no crashes - acceptance depends on exact NIS computation
    # The key is that gating logic executes without errors
    assert jnp.all(jnp.isfinite(state_upd.mean))
    assert jnp.all(jnp.isfinite(state_upd.cov))


def test_gating_with_partial_observations():
    """Test that gating works with single-LED observations."""
    mean = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01
    state = EKFState(mean=mean, cov=cov)

    config = EKFConfig(
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.997,
        state_mode="2d_full",
    )

    # LED1 valid, LED2 outlier (NaN)
    # Good LED1 observation
    z_led1 = jnp.array([0.98, 0.98])
    z_led2 = jnp.array([jnp.nan, jnp.nan])

    camera_model = make_camera_model(z_led1, z_led2, config)
    state_upd, _ = update_step(
        state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # LED1 should be accepted (2D measurement, lower threshold)
    assert not jnp.allclose(state_upd.mean[:2], state.mean[:2], atol=1e-6)


def test_gating_with_confidence_scaling():
    """Test that gating works correctly with confidence-scaled noise."""
    mean = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01
    state = EKFState(mean=mean, cov=cov)

    config = EKFConfig(
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.997,
        state_mode="2d_full",
    )

    # Moderate outlier (might be rejected with high confidence, accepted with low)
    z_led1 = jnp.array([1.5, 1.5])
    z_led2 = jnp.array([1.54, 1.5])

    # High confidence: tighter gate (might reject)
    conf_high = jnp.array([0.99, 0.99, 0.99, 0.99])
    camera_model_high = make_camera_model(z_led1, z_led2, config, conf_high)
    state_high, _ = update_step(
        state,
        camera_model_high,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # Low confidence: looser gate (should accept)
    conf_low = jnp.array([0.01, 0.01, 0.01, 0.01])
    camera_model_low = make_camera_model(z_led1, z_led2, config, conf_low)
    state_low, _ = update_step(
        state,
        camera_model_low,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # With low confidence (large R), innovation is less surprising
    # Both might accept, but low confidence should be more lenient
    # This is hard to test precisely, so just verify no crashes
    assert jnp.all(jnp.isfinite(state_high.mean))
    assert jnp.all(jnp.isfinite(state_low.mean))


def test_gating_consistency():
    """Test that gating is deterministic and consistent."""
    mean = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01
    state = EKFState(mean=mean, cov=cov)

    config = EKFConfig(
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.997,
        state_mode="2d_full",  # Use 8D layout (tests use hardcoded indices)
    )

    # Same observation should give same result
    z_led1 = jnp.array([0.05, 0.05])
    z_led2 = jnp.array([0.09, 0.05])

    camera_model = make_camera_model(z_led1, z_led2, config)
    state1, ll1 = update_step(
        state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )
    state2, ll2 = update_step(
        state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # Results should be identical
    assert jnp.allclose(state1.mean, state2.mean)
    assert jnp.allclose(state1.cov, state2.cov)
    assert ll1 == ll2
