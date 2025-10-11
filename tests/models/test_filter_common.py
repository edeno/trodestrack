"""Tests for shared filter configuration and state utilities."""

from __future__ import annotations

import jax.numpy as jnp

from trodestrack.models.ekf import EKFConfig, EKFState
from trodestrack.models.filter_common import (
    FilterCoreConfig,
    FilterState,
    initialize_state,
    update_zupt,
)
from trodestrack.models.ukf import UKFConfig, UKFState


def test_filter_core_config_defaults_match_existing_configs() -> None:
    core = FilterCoreConfig()
    ekf = EKFConfig()
    ukf = UKFConfig()

    shared_fields = [
        "process_noise_pos",
        "process_noise_vel",
        "process_noise_heading",
        "process_noise_gyro_bias",
        "process_noise_accel_bias",
        "measurement_noise_pos",
        "measurement_noise_heading",
        "imu_gyro_noise_density",
        "imu_accel_noise_density",
        "damping_coeff",
        "led_distance",
        "use_mahalanobis_gating",
        "mahalanobis_threshold_prob",
        "use_heading_measurement",
        "led_distance_tolerance",
        "adaptive_heading_noise",
        "adaptive_q_during_dropout",
        "dropout_q_pos_multiplier",
        "dropout_q_vel_multiplier",
        "dropout_q_bias_multiplier",
        "freeze_bias_during_blackout",
        "reduce_imu_noise_during_blackout",
        "blackout_imu_noise_scale",
        "enable_zupt",
        "zupt_velocity_threshold",
        "zupt_measurement_noise",
    ]

    for field in shared_fields:
        assert getattr(core, field) == getattr(
            ekf, field
        ), f"{field} mismatch between FilterCoreConfig and EKFConfig"
        assert getattr(core, field) == getattr(
            ukf, field
        ), f"{field} mismatch between FilterCoreConfig and UKFConfig"


def test_filter_state_namedtuple_parity() -> None:
    zeros = jnp.zeros(8)
    identity = jnp.eye(8)

    base_state = FilterState(mean=zeros, cov=identity)
    ekf_state = EKFState(zeros, identity)
    ukf_state = UKFState(zeros, identity)

    assert isinstance(ekf_state, FilterState)
    assert isinstance(ukf_state, FilterState)
    assert ekf_state == base_state
    assert ukf_state == base_state


def test_initialize_state_returns_filter_state() -> None:
    led1 = jnp.array([[0.0, 0.0], [0.04, 0.0]])
    led2 = jnp.array([[0.04, 0.0], [0.08, 0.0]])
    mask = jnp.array([True, True])

    state = initialize_state(led1, led2, mask, dt_cam=1.0 / 30.0, led_distance=0.04)

    assert isinstance(state, FilterState)
    assert state.mean.shape == (8,)
    assert state.cov.shape == (8, 8)


def test_update_zupt_accepts_filter_configs() -> None:
    zeros = jnp.zeros(8)
    identity = jnp.eye(8)
    state = FilterState(mean=zeros, cov=identity)

    ekf_config = EKFConfig(enable_zupt=True, zupt_velocity_threshold=0.05)
    state_ekf, log_lik_ekf = update_zupt(state, ekf_config)
    assert isinstance(state_ekf, FilterState)
    assert bool(jnp.isfinite(log_lik_ekf))

    ukf_config = UKFConfig(enable_zupt=True, zupt_velocity_threshold=0.05)
    state_ukf, log_lik_ukf = update_zupt(state, ukf_config)
    assert isinstance(state_ukf, FilterState)
    assert bool(jnp.isfinite(log_lik_ukf))
