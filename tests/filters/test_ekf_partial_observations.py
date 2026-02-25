"""Tests for EKF handling of partial LED observations.

This module tests that the EKF correctly handles different LED availability scenarios:
- Both LEDs available (4D measurement)
- Only LED1 available (2D measurement)
- Only LED2 available (2D measurement)
- No LEDs available (skip update)

The key requirement from PR_FIX_PLAN.md is to remove the 1e10 masking hack
and instead use shape-specific updates (2D vs 4D).
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
    mean = jnp.array([1.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01
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


def test_update_both_leds_valid(ekf_config, initial_state):
    """Test update when both LEDs are valid (4D measurement)."""
    # Both LEDs observed
    z_led1 = jnp.array([0.98, 1.0])
    z_led2 = jnp.array([1.02, 1.0])
    mask = True

    camera_model = make_camera_model(z_led1, z_led2, ekf_config)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=mask,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Position should be updated toward observation
    assert jnp.allclose(state_upd.mean[0], 1.0, atol=0.05)
    assert jnp.allclose(state_upd.mean[1], 1.0, atol=0.05)

    # Covariance should decrease (measurement reduces uncertainty)
    assert jnp.all(jnp.diag(state_upd.cov) <= jnp.diag(initial_state.cov))

    # Log-likelihood should be finite
    # Note: log-likelihood can be positive for high-probability observations
    # (when data is very close to prediction with small measurement noise)
    assert jnp.isfinite(log_lik)


def test_update_only_led1_valid(ekf_config, initial_state):
    """Test update when only LED1 is valid (2D measurement)."""
    # Only LED1 observed (LED2 is NaN)
    z_led1 = jnp.array([0.98, 1.0])
    z_led2 = jnp.array([jnp.nan, jnp.nan])
    mask = True

    camera_model = make_camera_model(z_led1, z_led2, ekf_config)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=mask,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Position should be updated toward LED1
    # But less confident than dual-LED case
    assert jnp.allclose(state_upd.mean[0], 0.98, atol=0.05)
    assert jnp.allclose(state_upd.mean[1], 1.0, atol=0.05)

    # Covariance should still decrease but less than dual-LED
    assert jnp.all(jnp.diag(state_upd.cov) <= jnp.diag(initial_state.cov))

    # Log-likelihood should be finite
    assert jnp.isfinite(log_lik)


def test_update_only_led2_valid(ekf_config, initial_state):
    """Test update when only LED2 is valid (2D measurement)."""
    # Only LED2 observed (LED1 is NaN)
    z_led1 = jnp.array([jnp.nan, jnp.nan])
    z_led2 = jnp.array([1.02, 1.0])
    mask = True

    camera_model = make_camera_model(z_led1, z_led2, ekf_config)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=mask,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Position should be updated toward LED2
    assert jnp.allclose(state_upd.mean[0], 1.02, atol=0.05)
    assert jnp.allclose(state_upd.mean[1], 1.0, atol=0.05)

    # Covariance should still decrease
    assert jnp.all(jnp.diag(state_upd.cov) <= jnp.diag(initial_state.cov))

    # Log-likelihood should be finite
    assert jnp.isfinite(log_lik)


def test_update_no_leds_valid(ekf_config, initial_state):
    """Test that update is skipped when no LEDs are valid."""
    # No LEDs observed
    z_led1 = jnp.array([jnp.nan, jnp.nan])
    z_led2 = jnp.array([jnp.nan, jnp.nan])
    mask = True

    camera_model = make_camera_model(z_led1, z_led2, ekf_config)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=mask,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # State should remain unchanged
    assert jnp.allclose(state_upd.mean, initial_state.mean)
    assert jnp.allclose(state_upd.cov, initial_state.cov)

    # Log-likelihood should be zero (no measurement)
    assert log_lik == 0.0


def test_update_mask_false(ekf_config, initial_state):
    """Test that update is skipped when mask is False."""
    # Valid observations but mask is False
    z_led1 = jnp.array([0.98, 1.0])
    z_led2 = jnp.array([1.02, 1.0])
    mask = False

    camera_model = make_camera_model(z_led1, z_led2, ekf_config)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=mask,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # State should remain unchanged
    assert jnp.allclose(state_upd.mean, initial_state.mean)
    assert jnp.allclose(state_upd.cov, initial_state.cov)

    # Log-likelihood should be zero
    assert log_lik == 0.0


def test_update_covariance_reduction_dual_vs_single():
    """Test that dual-LED reduces covariance more than single-LED."""
    config = EKFConfig()
    mean = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.1  # Higher initial uncertainty
    state = EKFState(mean=mean, cov=cov)

    # Dual-LED observation
    z_led1_dual = jnp.array([0.98, 1.0])
    z_led2_dual = jnp.array([1.02, 1.0])
    camera_model_dual = make_camera_model(z_led1_dual, z_led2_dual, config)
    state_dual, _ = update_step(
        state,
        camera_model_dual,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # Single-LED observation (LED1 only)
    z_led1_single = jnp.array([1.0, 1.0])
    z_led2_single = jnp.array([jnp.nan, jnp.nan])
    camera_model_single = make_camera_model(z_led1_single, z_led2_single, config)
    state_single, _ = update_step(
        state,
        camera_model_single,
        frame_idx=0,
        observation_is_valid=True,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # Dual-LED should reduce position covariance more
    dual_pos_var = jnp.trace(state_dual.cov[:2, :2])
    single_pos_var = jnp.trace(state_single.cov[:2, :2])

    assert dual_pos_var < single_pos_var, (
        f"Dual-LED variance {dual_pos_var} should be less than single-LED variance {single_pos_var}"
    )


def test_update_no_extreme_artifacts(ekf_config, initial_state):
    """Test that the covariance doesn't have extreme numerical artifacts.

    This test verifies that while we use masking (1e6 instead of 1e10),
    the covariance remains reasonable and PSD.
    """
    # Only LED1 valid
    z_led1 = jnp.array([1.0, 1.0])
    z_led2 = jnp.array([jnp.nan, jnp.nan])
    mask = True

    camera_model = make_camera_model(z_led1, z_led2, ekf_config)
    state_upd, _ = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=mask,
        config=ekf_config,
        layout=LAYOUT_2D_FULL,
    )

    # Check that no covariance element is unreasonably large (< 1000, not 1e10)
    max_cov = jnp.max(jnp.abs(state_upd.cov))
    assert max_cov < 1000.0, (
        f"Max covariance {max_cov} suggests extreme numerical artifacts"
    )

    # Check that covariance is still PSD (all eigenvalues positive)
    eigenvalues = jnp.linalg.eigvalsh(state_upd.cov)
    assert jnp.all(eigenvalues > -1e-6), (
        f"Covariance should be PSD, got eigenvalues {eigenvalues}"
    )
