"""Tests for shared filter configuration and state utilities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, EKFState
from trodestrack.models.filter_common import (
    FilterCoreConfig,
    FilterState,
    gravity_compensate,
    initialize_state,
    rotate_body_accel_to_world,
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
        "imu_gravity_body",
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
        assert getattr(core, field) == getattr(ekf, field), (
            f"{field} mismatch between FilterCoreConfig and EKFConfig"
        )
        assert getattr(core, field) == getattr(ukf, field), (
            f"{field} mismatch between FilterCoreConfig and UKFConfig"
        )


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


def test_initialize_state_ignores_partial_coordinate_nan_leds() -> None:
    """Initialization should not treat one finite LED coordinate as a 2D position."""
    led1 = jnp.array([[0.0, jnp.nan], [1.0, 2.0]])
    led2 = jnp.array([[jnp.nan, 0.0], [1.04, 2.0]])
    mask = jnp.array([True, True])

    state = initialize_state(led1, led2, mask, dt_cam=1.0 / 30.0, led_distance=0.04)

    np.testing.assert_allclose(state.mean[:2], [1.02, 2.0], atol=1e-6)
    np.testing.assert_allclose(state.mean[4], 0.0, atol=1e-6)


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


def test_update_zupt_disabled_is_exact_noop_with_large_covariance() -> None:
    mean = jnp.array([0.0, 0.0, 10.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 1e6
    state = FilterState(mean=mean, cov=cov)

    posterior, log_lik = update_zupt(
        state,
        EKFConfig(enable_zupt=False, state_mode="2d_full"),
    )

    np.testing.assert_allclose(posterior.mean, mean)
    np.testing.assert_allclose(posterior.cov, cov)
    assert float(log_lik) == 0.0


def test_update_zupt_inactive_is_exact_noop_with_large_covariance() -> None:
    mean = jnp.zeros(8)
    cov = jnp.eye(8) * 1e6
    state = FilterState(mean=mean, cov=cov)

    posterior, log_lik = update_zupt(
        state,
        EKFConfig(enable_zupt=True, zupt_velocity_threshold=0.05, state_mode="2d_full"),
        active=False,
    )

    np.testing.assert_allclose(posterior.mean, mean)
    np.testing.assert_allclose(posterior.cov, cov)
    assert float(log_lik) == 0.0


def test_update_zupt_moving_state_is_exact_noop_with_large_covariance() -> None:
    mean = jnp.array([0.0, 0.0, 10.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 1e6
    state = FilterState(mean=mean, cov=cov)

    posterior, log_lik = update_zupt(
        state,
        EKFConfig(
            enable_zupt=True,
            zupt_velocity_threshold=0.02,
            state_mode="2d_full",
        ),
    )

    np.testing.assert_allclose(posterior.mean, mean)
    np.testing.assert_allclose(posterior.cov, cov)
    assert float(log_lik) == 0.0


# =============================================================================
# Tests for 3D IMU → 2D Pose Helpers (M5)
# =============================================================================


def test_rotate_body_accel_to_world_zero_heading() -> None:
    """Test rotation at zero heading (body frame aligned with world frame)."""
    accel_body = jnp.array([1.0, 0.0, 0.0])
    yaw_heading = 0.0

    accel_world = rotate_body_accel_to_world(accel_body, yaw_heading)

    # At zero heading, body frame aligns with world frame
    np.testing.assert_allclose(accel_world, [1.0, 0.0, 0.0], atol=1e-6)


def test_rotate_body_accel_to_world_90_degrees() -> None:
    """Test rotation at 90 degrees heading."""
    accel_body = jnp.array([1.0, 0.0, 0.0])
    yaw_heading = jnp.pi / 2  # 90 degrees

    accel_world = rotate_body_accel_to_world(accel_body, yaw_heading)

    # At 90° heading, forward (x) in body → left (y) in world
    np.testing.assert_allclose(accel_world, [0.0, 1.0, 0.0], atol=1e-6)


def test_rotate_body_accel_to_world_preserves_vertical() -> None:
    """Test that vertical component (z-axis) is preserved."""
    accel_body = jnp.array([0.0, 0.0, 9.81])
    yaw_heading = jnp.pi / 4  # 45 degrees

    accel_world = rotate_body_accel_to_world(accel_body, yaw_heading)

    # Z-axis should be unchanged (only yaw rotation)
    np.testing.assert_allclose(accel_world[2], 9.81, atol=1e-6)
    # X and Y should remain zero (no acceleration in horizontal plane)
    np.testing.assert_allclose(accel_world[0:2], [0.0, 0.0], atol=1e-6)


def test_rotate_body_accel_to_world_full_3d() -> None:
    """Test rotation with non-zero components in all axes."""
    accel_body = jnp.array([1.0, 2.0, 3.0])
    yaw_heading = jnp.pi / 6  # 30 degrees

    accel_world = rotate_body_accel_to_world(accel_body, yaw_heading)

    # Verify shape and finite values
    assert accel_world.shape == (3,)
    assert jnp.all(jnp.isfinite(accel_world))

    # Verify rotation only affects x-y plane (z unchanged)
    np.testing.assert_allclose(accel_world[2], 3.0, atol=1e-6)

    # Verify x-y plane rotation is correct (R_z(30°) @ [1, 2])
    cos_30 = jnp.cos(jnp.pi / 6)
    sin_30 = jnp.sin(jnp.pi / 6)
    expected_x = cos_30 * 1.0 - sin_30 * 2.0
    expected_y = sin_30 * 1.0 + cos_30 * 2.0
    np.testing.assert_allclose(accel_world[0], expected_x, atol=1e-6)
    np.testing.assert_allclose(accel_world[1], expected_y, atol=1e-6)


def test_rotate_body_accel_to_world_returns_jax_array() -> None:
    """Test that output is a JAX array."""
    accel_body = jnp.array([1.0, 0.0, 0.0])
    yaw_heading = 0.0

    accel_world = rotate_body_accel_to_world(accel_body, yaw_heading)

    assert isinstance(accel_world, jnp.ndarray)


def test_gravity_compensate_removes_gravity() -> None:
    """Test that gravity compensation removes standard gravity."""
    accel_world = jnp.array([0.0, 0.0, 9.81])

    accel_compensated = gravity_compensate(accel_world, g=9.81)

    # Pure gravity should be removed
    np.testing.assert_allclose(accel_compensated, [0.0, 0.0, 0.0], atol=1e-6)


def test_gravity_compensate_preserves_horizontal() -> None:
    """Test that horizontal acceleration is preserved."""
    accel_world = jnp.array([1.0, 2.0, 9.81])

    accel_compensated = gravity_compensate(accel_world, g=9.81)

    # Horizontal components unchanged, vertical reduced by g
    np.testing.assert_allclose(accel_compensated, [1.0, 2.0, 0.0], atol=1e-6)


def test_gravity_compensate_with_nonstandard_g() -> None:
    """Test gravity compensation with non-standard gravity value."""
    accel_world = jnp.array([0.0, 0.0, 10.0])
    g_custom = 10.0

    accel_compensated = gravity_compensate(accel_world, g=g_custom)

    np.testing.assert_allclose(accel_compensated, [0.0, 0.0, 0.0], atol=1e-6)


def test_gravity_compensate_default_g() -> None:
    """Test that default gravity value is 9.81 m/s²."""
    accel_world = jnp.array([0.0, 0.0, 9.81])

    accel_compensated = gravity_compensate(accel_world)

    # Should use default g=9.81
    np.testing.assert_allclose(accel_compensated, [0.0, 0.0, 0.0], atol=1e-6)


def test_gravity_compensate_returns_jax_array() -> None:
    """Test that output is a JAX array."""
    accel_world = jnp.array([0.0, 0.0, 9.81])

    accel_compensated = gravity_compensate(accel_world)

    assert isinstance(accel_compensated, jnp.ndarray)


def test_filter_core_config_bool_fields_require_strict_bool() -> None:
    """Bool toggles in FilterCoreConfig must reject non-bool values.

    Plain truthiness silently accepts ``"False"`` (truthy), integers,
    and lists. The values later flow into JAX boolean ops (``&`` /
    ``lax.cond``) and either silently take the wrong branch or crash
    deep in JAX with ``TypeError: unsupported operand type(s) for &:
    'str' and 'jaxlib.xla_extension.ArrayImpl'``. CLI / YAML / env
    loaders are an obvious source — require ``bool`` exactly.
    """
    bool_fields = (
        "use_mahalanobis_gating",
        "use_heading_measurement",
        "adaptive_heading_noise",
        "adaptive_q_during_dropout",
        "freeze_bias_during_blackout",
        "reduce_imu_noise_during_blackout",
        "enable_experimental_accel_translation",
        "use_gravity_orientation_update",
        "enable_zupt",
    )
    for fname in bool_fields:
        for bad in ("False", "True", 1, 0, [1]):
            with pytest.raises(ValueError, match=rf"{fname} must be a Python"):
                FilterCoreConfig(**{fname: bad})

    # Subclasses inherit the gate.
    with pytest.raises(ValueError, match=r"use_heading_measurement must be a Python"):
        EKFConfig(use_heading_measurement="False")
    with pytest.raises(ValueError, match=r"use_heading_measurement must be a Python"):
        UKFConfig(use_heading_measurement="False")

    # True / False still accepted on the parent and the subclasses.
    FilterCoreConfig(use_heading_measurement=True)
    FilterCoreConfig(use_heading_measurement=False)
    EKFConfig(use_heading_measurement=True)
    UKFConfig(use_heading_measurement=False)
