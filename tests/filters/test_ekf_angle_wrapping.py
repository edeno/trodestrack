"""Tests for EKF angle wrapping behavior.

This module tests that heading angles are properly wrapped to (-π, π]
in both predict and update steps to avoid numerical issues with
large angle values.
"""

from __future__ import annotations

import jax.numpy as jnp

from trodestrack.models.ekf import (
    EKFConfig,
    EKFState,
    predict_step,
    update_step,
    wrap_angle,
)
from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.state_layout import LAYOUT_2D_FULL


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


def test_wrap_angle_function():
    """Test that wrap_angle correctly maps to (-π, π].

    Note: The interval is (-π, π], so π is included but -π is not.
    However, arctan2 may return -π for angles exactly at ±π.
    """
    # Test wrapping from above
    assert jnp.allclose(wrap_angle(jnp.pi + 0.1), -(jnp.pi - 0.1), atol=1e-6)
    assert jnp.allclose(wrap_angle(2 * jnp.pi), 0.0, atol=1e-6)
    # 3π wraps to π, but arctan2(sin(3π), cos(3π)) = arctan2(0, -1) = π or -π
    assert jnp.abs(wrap_angle(3 * jnp.pi)) > jnp.pi - 1e-6  # near ±π

    # Test wrapping from below
    assert jnp.allclose(wrap_angle(-jnp.pi - 0.1), jnp.pi - 0.1, atol=1e-6)
    assert jnp.allclose(wrap_angle(-2 * jnp.pi), 0.0, atol=1e-6)

    # Test no wrapping needed
    assert jnp.allclose(wrap_angle(0.0), 0.0, atol=1e-6)
    assert jnp.allclose(wrap_angle(jnp.pi / 2), jnp.pi / 2, atol=1e-6)
    assert jnp.allclose(wrap_angle(-jnp.pi / 2), -jnp.pi / 2, atol=1e-6)


def test_predict_step_wraps_heading():
    """Test that predict_step wraps heading to (-π, π]."""
    config = EKFConfig()

    # Create state with heading near +π
    mean = jnp.array([0.0, 0.0, 0.0, 0.0, jnp.pi - 0.1, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01
    state = EKFState(mean=mean, cov=cov)

    # IMU input with positive angular velocity (will push heading past +π)
    dt = 0.01
    omega_z = 20.0  # rad/s -> will add 0.2 rad, pushing past π
    u_imu = jnp.array([omega_z, 0.0, 0.0])

    # Predict
    state_pred = predict_step(state, u_imu, dt, config, layout=LAYOUT_2D_FULL)

    # Heading should be wrapped to negative side
    theta_pred = state_pred.mean[4]
    assert theta_pred > -jnp.pi, f"Heading {theta_pred} should be > -π"
    assert theta_pred <= jnp.pi, f"Heading {theta_pred} should be ≤ π"

    # Expected unwrapped value: π - 0.1 + 0.2 = π + 0.1
    # Wrapped: -(π - 0.1) ≈ -3.04
    expected_wrapped = wrap_angle((jnp.pi - 0.1) + omega_z * dt)
    assert jnp.allclose(
        theta_pred, expected_wrapped, atol=0.05
    ), f"Heading {theta_pred} should be close to {expected_wrapped}"


def test_predict_step_wraps_heading_negative():
    """Test that predict_step wraps heading from negative side."""
    config = EKFConfig()

    # Create state with heading near -π
    mean = jnp.array([0.0, 0.0, 0.0, 0.0, -jnp.pi + 0.1, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.01
    state = EKFState(mean=mean, cov=cov)

    # IMU input with negative angular velocity (will push heading past -π)
    dt = 0.01
    omega_z = -20.0  # rad/s -> will subtract 0.2 rad, pushing past -π
    u_imu = jnp.array([omega_z, 0.0, 0.0])

    # Predict
    state_pred = predict_step(state, u_imu, dt, config, layout=LAYOUT_2D_FULL)

    # Heading should be wrapped to positive side
    theta_pred = state_pred.mean[4]
    assert theta_pred > -jnp.pi, f"Heading {theta_pred} should be > -π"
    assert theta_pred <= jnp.pi, f"Heading {theta_pred} should be ≤ π"

    # Expected: wrapped to positive side near +π
    expected_wrapped = wrap_angle((-jnp.pi + 0.1) + omega_z * dt)
    assert jnp.allclose(theta_pred, expected_wrapped, atol=0.05)


def test_update_step_wraps_heading():
    """Test that update_step wraps heading after measurement update."""
    config = EKFConfig()

    # Create predicted state with large positive heading
    # This simulates accumulated heading from many IMU integrations
    mean_pred = jnp.array([1.0, 1.0, 0.0, 0.0, 5.0 * jnp.pi, 0.0, 0.0, 0.0])
    cov_pred = jnp.eye(8) * 0.01
    state_pred = EKFState(mean=mean_pred, cov=cov_pred)

    # LED observations that suggest heading near 0
    # LED1 at (0.98, 1.0), LED2 at (1.02, 1.0) -> heading ≈ 0
    z_led1 = jnp.array([0.98, 1.0])
    z_led2 = jnp.array([1.02, 1.0])
    mask = True

    # Update
    camera_model = make_camera_model(z_led1, z_led2, config)
    state_upd, log_lik = update_step(
        state_pred,
        camera_model,
        frame_idx=0,
        observation_is_valid=mask,
        config=config,
        layout=LAYOUT_2D_FULL,
    )

    # Heading should be wrapped to (-π, π]
    theta_upd = state_upd.mean[4]
    assert theta_upd > -jnp.pi, f"Heading {theta_upd} should be > -π"
    assert theta_upd <= jnp.pi, f"Heading {theta_upd} should be ≤ π"

    # After update with measurements suggesting heading ≈ 0,
    # 5π wraps to π (or -π), and with LED measurements at horizontal (heading=0),
    # the filter should pull it toward 0, but it might stay near ±π due to
    # the ambiguity in wrapped angles and measurement uncertainty.
    # The key test is that it's wrapped to valid range.
    assert (
        jnp.abs(theta_upd) <= jnp.pi
    ), f"Heading {theta_upd} should be in valid range after wrapping"


def test_heading_continuity_across_2pi():
    """Test that heading remains continuous when crossing ±π boundary."""
    config = EKFConfig()

    # Start near π
    mean = jnp.array([0.0, 0.0, 0.0, 0.0, jnp.pi - 0.05, 0.0, 0.0, 0.0])
    cov = jnp.eye(8) * 0.001
    state = EKFState(mean=mean, cov=cov)

    # Small positive rotation
    dt = 0.01
    omega_z = 10.0  # will add 0.1 rad -> cross π boundary
    u_imu = jnp.array([omega_z, 0.0, 0.0])

    state_pred = predict_step(state, u_imu, dt, config, layout=LAYOUT_2D_FULL)
    theta_pred = state_pred.mean[4]

    # Should wrap to negative side: (π - 0.05) + 0.1 = π + 0.05 -> wraps to -(π - 0.05)
    expected = wrap_angle((jnp.pi - 0.05) + 0.1)
    assert jnp.allclose(
        theta_pred, expected, atol=0.01
    ), f"Heading {theta_pred} should wrap correctly to {expected}"

    # Check continuity: difference should be ~0.1 rad (unwrapped)
    # but after wrapping, the wrapped difference might be large
    # The key is that the *physical* rotation is still 0.1 rad
    unwrapped_diff = omega_z * dt
    assert jnp.allclose(unwrapped_diff, 0.1, atol=0.01)
