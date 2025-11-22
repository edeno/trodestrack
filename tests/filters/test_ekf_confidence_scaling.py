"""Tests for EKF confidence-scaled measurement noise.

This module tests that the EKF properly scales measurement noise
based on DLC confidence scores. Higher confidence should lead to
lower measurement noise and thus stronger updates.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from trodestrack.models.ekf import EKFConfig, EKFState, update_step
from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.state_layout import LAYOUT_2D_FULL


@pytest.fixture
def ekf_config():
    """Standard EKF configuration."""
    return EKFConfig()


@pytest.fixture
def initial_state():
    """Create a simple initial state for testing."""
    mean = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.1  # Higher initial uncertainty
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


def test_high_confidence_reduces_covariance_more(ekf_config, initial_state):
    """Test that high confidence observations reduce uncertainty more."""
    # Observation far from current estimate
    z_led1 = jnp.array([0.5, 0.5])
    z_led2 = jnp.array([0.54, 0.5])

    # High confidence update
    conf_high = jnp.array([0.99, 0.99, 0.99, 0.99])
    camera_model_high = make_camera_model(z_led1, z_led2, ekf_config, conf_high)
    state_high, _ = update_step(
        initial_state,
        camera_model_high,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Low confidence update
    conf_low = jnp.array([0.1, 0.1, 0.1, 0.1])
    camera_model_low = make_camera_model(z_led1, z_led2, ekf_config, conf_low)
    state_low, _ = update_step(
        initial_state,
        camera_model_low,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # High confidence should reduce position covariance more
    cov_high = jnp.trace(state_high.cov[:2, :2])
    cov_low = jnp.trace(state_low.cov[:2, :2])

    assert cov_high < cov_low, (
        f"High confidence covariance {cov_high:.6f} should be less than "
        f"low confidence {cov_low:.6f}"
    )


def test_high_confidence_pulls_mean_more(ekf_config, initial_state):
    """Test that high confidence observations pull mean more toward measurement."""
    # Observation far from current estimate (initial is at [1, 1])
    z_led1 = jnp.array([0.5, 0.5])
    z_led2 = jnp.array([0.54, 0.5])

    # High confidence update
    conf_high = jnp.array([0.99, 0.99, 0.99, 0.99])
    camera_model_high = make_camera_model(z_led1, z_led2, ekf_config, conf_high)
    state_high, _ = update_step(
        initial_state,
        camera_model_high,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Low confidence update
    conf_low = jnp.array([0.1, 0.1, 0.1, 0.1])
    camera_model_low = make_camera_model(z_led1, z_led2, ekf_config, conf_low)
    state_low, _ = update_step(
        initial_state,
        camera_model_low,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # High confidence should move mean closer to observation
    # Target midpoint: (0.52, 0.5)
    dist_high = jnp.linalg.norm(state_high.mean[:2] - jnp.array([0.52, 0.5]))
    dist_low = jnp.linalg.norm(state_low.mean[:2] - jnp.array([0.52, 0.5]))

    assert (
        dist_high < dist_low
    ), f"High confidence should be closer to observation: {dist_high:.6f} vs {dist_low:.6f}"


def test_zero_confidence_smaller_update_than_high(ekf_config, initial_state):
    """Test that zero confidence leads to smaller update than high confidence."""
    z_led1 = jnp.array([0.5, 0.5])
    z_led2 = jnp.array([0.54, 0.5])

    # Zero confidence (will be clipped to minimum 1e-2)
    conf_zero = jnp.array([0.0, 0.0, 0.0, 0.0])
    camera_model_zero = make_camera_model(z_led1, z_led2, ekf_config, conf_zero)
    state_zero, _ = update_step(
        initial_state,
        camera_model_zero,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # High confidence
    conf_high = jnp.array([0.99, 0.99, 0.99, 0.99])
    camera_model_high = make_camera_model(z_led1, z_led2, ekf_config, conf_high)
    state_high, _ = update_step(
        initial_state,
        camera_model_high,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Zero confidence update should be smaller than high confidence
    # (measured by distance moved from initial state)
    dist_zero = jnp.linalg.norm(state_zero.mean[:2] - initial_state.mean[:2])
    dist_high = jnp.linalg.norm(state_high.mean[:2] - initial_state.mean[:2])

    assert dist_zero < dist_high, (
        f"Zero confidence update {dist_zero:.6f} should be smaller than "
        f"high confidence {dist_high:.6f}"
    )


def test_confidence_per_led_covariance(ekf_config, initial_state):
    """Test that confidence affects covariance reduction per LED."""
    z_led1 = jnp.array([0.5, 0.5])
    z_led2 = jnp.array([0.54, 0.5])

    # Both LEDs high confidence
    conf_both_high = jnp.array([0.99, 0.99, 0.99, 0.99])
    camera_model_both = make_camera_model(z_led1, z_led2, ekf_config, conf_both_high)
    state_both, _ = update_step(
        initial_state,
        camera_model_both,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # LED1 high, LED2 low
    conf_led1_high = jnp.array([0.99, 0.99, 0.01, 0.01])
    camera_model_led1 = make_camera_model(z_led1, z_led2, ekf_config, conf_led1_high)
    state_led1, _ = update_step(
        initial_state,
        camera_model_led1,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Both high confidence should reduce covariance more than one high
    cov_both = jnp.trace(state_both.cov[:2, :2])
    cov_led1 = jnp.trace(state_led1.cov[:2, :2])

    assert (
        cov_both < cov_led1
    ), f"Both-high covariance {cov_both:.6f} should be < LED1-only {cov_led1:.6f}"


def test_default_confidence_is_high(ekf_config, initial_state):
    """Test that default confidence (when not specified) behaves as high confidence."""
    z_led1 = jnp.array([0.5, 0.5])
    z_led2 = jnp.array([0.54, 0.5])

    # Update without confidence parameter (should default to 1.0)
    camera_model_default = make_camera_model(
        z_led1, z_led2, ekf_config, confidence=None
    )
    state_default, _ = update_step(
        initial_state,
        camera_model_default,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Update with explicit high confidence
    conf_high = jnp.array([1.0, 1.0, 1.0, 1.0])
    camera_model_explicit = make_camera_model(z_led1, z_led2, ekf_config, conf_high)
    state_explicit, _ = update_step(
        initial_state,
        camera_model_explicit,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Should be identical
    assert jnp.allclose(state_default.mean, state_explicit.mean, atol=1e-6)
    assert jnp.allclose(state_default.cov, state_explicit.cov, atol=1e-6)


def test_confidence_clipping():
    """Test that confidence is clipped to reasonable range [min, 1.0]."""
    config = EKFConfig()
    mean = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.1
    state = EKFState(mean=mean, cov=cov)

    z_led1 = jnp.array([0.9, 0.9])
    z_led2 = jnp.array([0.94, 0.9])

    # Test very small confidence (should be clipped)
    conf_tiny = jnp.array([1e-10, 1e-10, 1e-10, 1e-10])
    camera_model_tiny = make_camera_model(z_led1, z_led2, config, conf_tiny)
    state_tiny, _ = update_step(
        state,
        camera_model_tiny,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # Test slightly negative confidence (invalid but should be clipped)
    conf_neg = jnp.array([-0.1, -0.1, -0.1, -0.1])
    camera_model_neg = make_camera_model(z_led1, z_led2, config, conf_neg)
    state_neg, _ = update_step(
        state,
        camera_model_neg,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # Both should produce valid states (no NaN, no extreme values)
    assert jnp.all(jnp.isfinite(state_tiny.mean))
    assert jnp.all(jnp.isfinite(state_tiny.cov))
    assert jnp.all(jnp.isfinite(state_neg.mean))
    assert jnp.all(jnp.isfinite(state_neg.cov))

    # Clipped values should behave similarly (both at minimum)
    assert jnp.allclose(state_tiny.mean, state_neg.mean, atol=1e-4)


def test_confidence_affects_log_likelihood(ekf_config, initial_state):
    """Test that confidence affects the computed log-likelihood."""
    z_led1 = jnp.array([0.95, 0.95])  # Close to initial state
    z_led2 = jnp.array([0.99, 0.95])

    # High confidence: small R → higher likelihood for good match
    conf_high = jnp.array([0.99, 0.99, 0.99, 0.99])
    camera_model_high = make_camera_model(z_led1, z_led2, ekf_config, conf_high)
    _, ll_high = update_step(
        initial_state,
        camera_model_high,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Low confidence: large R → lower likelihood sensitivity
    conf_low = jnp.array([0.1, 0.1, 0.1, 0.1])
    camera_model_low = make_camera_model(z_led1, z_led2, ekf_config, conf_low)
    _, ll_low = update_step(
        initial_state,
        camera_model_low,
        frame_idx=0,
        observation_is_valid=True,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Log-likelihoods should differ
    assert not jnp.allclose(
        ll_high, ll_low, atol=0.1
    ), f"Confidence should affect log-likelihood: {ll_high:.4f} vs {ll_low:.4f}"
