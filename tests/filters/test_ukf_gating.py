"""Tests for UKF χ² gating (outlier rejection)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.state_layout import LAYOUT_2D_FULL
from trodestrack.models.ukf import UKFConfig, UKFState, update_step


@pytest.fixture
def ukf_config():
    """Standard UKF configuration with gating enabled."""
    return UKFConfig(
        use_mahalanobis_gating=True,
        mahalanobis_threshold_prob=0.997,
    )


@pytest.fixture
def ukf_config_no_gating():
    """UKF configuration with gating disabled (baseline)."""
    return UKFConfig(use_mahalanobis_gating=False)


@pytest.fixture
def initial_state():
    """Create a simple initial state near [1, 1]."""
    mean = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01
    return UKFState(mean=mean, cov=cov)


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


def test_ukf_gating_rejects_outlier(ukf_config, initial_state):
    """Mahalanobis gating should reject extreme outliers."""
    z_led1 = jnp.array([5.0, 5.0])
    z_led2 = jnp.array([5.04, 5.0])

    camera_model = make_camera_model(z_led1, z_led2, ukf_config)
    state_upd, log_lik = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=ukf_config,
    )

    # Expect rejection: state and covariance unchanged, zero log-likelihood
    assert jnp.allclose(state_upd.mean, initial_state.mean, atol=1e-6)
    assert jnp.allclose(state_upd.cov, initial_state.cov, atol=1e-6)
    assert log_lik == 0.0


def test_ukf_gating_disabled_accepts_outlier(ukf_config_no_gating, initial_state):
    """Without gating, the same outlier should be incorporated."""
    z_led1 = jnp.array([5.0, 5.0])
    z_led2 = jnp.array([5.04, 5.0])

    camera_model = make_camera_model(z_led1, z_led2, ukf_config_no_gating)
    state_upd, _ = update_step(
        initial_state,
        camera_model,
        frame_idx=0,
        observation_is_valid=True,
        config=ukf_config_no_gating,
    )

    assert not jnp.allclose(state_upd.mean, initial_state.mean, atol=1e-3)
