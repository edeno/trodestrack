"""Tests for the 3D LED camera measurement model."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from trodestrack.models.filter_common import (
    FilterState,
    joseph_update,
    normalize_state_orientation,
    psd_solve,
)
from trodestrack.models.quaternion import (
    quaternion_from_rotation_vector,
    rotate_vector_body_to_world,
)
from trodestrack.models.sensors import Camera3DPositionModel
from trodestrack.models.state_layout import get_layout


def _state_from_pose(position: jnp.ndarray, quat: jnp.ndarray) -> jnp.ndarray:
    layout = get_layout("3d_quat")
    state = jnp.zeros(layout.n)
    state = state.at[jnp.array(layout.pos_idx)].set(position)
    state = state.at[jnp.array(layout.heading_idx)].set(quat)
    return state


def _ekf_update(
    state: FilterState,
    model: Camera3DPositionModel,
    frame_idx: int,
) -> FilterState:
    mean, cov = state
    meas_pred = model.predict(mean)
    innovation = model.innovation(frame_idx, meas_pred)
    H = model.jacobian(mean, frame_idx)
    R = model.meas_cov(frame_idx)
    S = H @ cov @ H.T + R
    K = psd_solve(S, H @ cov).T
    mean_upd = mean + K @ innovation
    mean_upd = normalize_state_orientation(mean_upd, model.layout)
    cov_upd = joseph_update(cov, K, H, R)
    return FilterState(mean=mean_upd, cov=cov_upd)


def test_camera_3d_prediction_matches_known_pose_and_offsets() -> None:
    layout = get_layout("3d_quat")
    led_offsets = jnp.array(
        [
            [-0.02, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ]
    )
    position = jnp.array([1.0, 2.0, 0.5])
    quat = quaternion_from_rotation_vector(jnp.array([0.0, 0.0, jnp.pi / 2.0]))
    state = _state_from_pose(position, quat)
    expected = position[None, :] + rotate_vector_body_to_world(quat, led_offsets)
    model = Camera3DPositionModel(
        led_offsets_body=led_offsets,
        measurement_noise_base=1e-4,
        layout=layout,
        z_leds_all=expected[None, :, :],
    )

    prediction = model.predict(state)

    np.testing.assert_allclose(
        np.asarray(prediction),
        np.asarray(expected.reshape(-1)),
        atol=1e-6,
    )
    assert model.meas_dim == 6


def test_camera_3d_missing_coordinates_are_ignored_without_nan_residuals() -> None:
    layout = get_layout("3d_quat")
    led_offsets = jnp.array(
        [
            [-0.02, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ]
    )
    state = _state_from_pose(
        jnp.array([1.0, 2.0, 0.5]),
        quaternion_from_rotation_vector(jnp.array([0.0, 0.0, 0.0])),
    )
    z_leds = jnp.array(
        [
            [
                [1.1, jnp.nan, 0.5],
                [9.0, 9.0, 9.0],
            ]
        ]
    )
    mask_leds = jnp.array([[True, False]])
    model = Camera3DPositionModel(
        led_offsets_body=led_offsets,
        measurement_noise_base=1e-4,
        layout=layout,
        z_leds_all=z_leds,
        mask_leds_all=mask_leds,
    )

    meas_pred = model.predict(state)
    innovation = model.innovation(0, meas_pred)
    H = model.jacobian(state, 0)
    R_diag = jnp.diag(model.meas_cov(0))
    valid = model.valid_coordinates(0)

    assert np.isfinite(np.asarray(innovation)).all()
    assert np.asarray(valid).tolist() == [True, False, True, False, False, False]
    np.testing.assert_allclose(np.asarray(innovation[~valid]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(H[~valid]), 0.0, atol=1e-12)
    assert np.all(np.asarray(R_diag[~valid]) == model.invalid_measurement_noise)
    assert np.all(np.asarray(R_diag[valid]) < model.invalid_measurement_noise)


def test_camera_3d_all_missing_frame_does_not_change_covariance() -> None:
    layout = get_layout("3d_quat")
    led_offsets = jnp.array(
        [
            [-0.02, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ]
    )
    state_mean = _state_from_pose(
        jnp.array([0.0, 0.0, 0.0]),
        quaternion_from_rotation_vector(jnp.array([0.0, 0.0, 0.0])),
    )
    state = FilterState(mean=state_mean, cov=jnp.eye(layout.n) * 1e9)
    model = Camera3DPositionModel(
        led_offsets_body=led_offsets,
        measurement_noise_base=1e-4,
        layout=layout,
        z_leds_all=jnp.full((1, 2, 3), jnp.nan),
        mask_leds_all=jnp.array([[False, False]]),
    )

    updated = _ekf_update(state, model, frame_idx=0)

    np.testing.assert_allclose(
        np.asarray(updated.mean),
        np.asarray(state.mean),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(updated.cov),
        np.asarray(state.cov),
        rtol=1e-6,
        atol=1e-6,
    )


def test_camera_3d_update_recovers_low_noise_synthetic_pose() -> None:
    layout = get_layout("3d_quat")
    led_offsets = jnp.array(
        [
            [-0.03, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.0, 0.025, 0.02],
        ]
    )
    true_position = jnp.array([1.0, -0.5, 0.25])
    true_quat = quaternion_from_rotation_vector(jnp.array([0.08, -0.06, 0.12]))
    true_state = _state_from_pose(true_position, true_quat)
    z_leds = (
        true_position[None, :] + rotate_vector_body_to_world(true_quat, led_offsets)
    )[None, :, :]
    model = Camera3DPositionModel(
        led_offsets_body=led_offsets,
        measurement_noise_base=1e-8,
        layout=layout,
        z_leds_all=z_leds,
    )
    initial_state = _state_from_pose(
        jnp.array([0.85, -0.35, 0.35]),
        quaternion_from_rotation_vector(jnp.array([0.0, 0.0, 0.0])),
    )
    state = FilterState(mean=initial_state, cov=jnp.eye(layout.n) * 0.25)

    initial_error = jnp.linalg.norm(
        model.predict(state.mean) - model.predict(true_state)
    )
    for _ in range(6):
        state = _ekf_update(state, model, frame_idx=0)
    final_error = jnp.linalg.norm(model.predict(state.mean) - model.predict(true_state))

    assert float(final_error) < 1e-3
    assert float(final_error) < 1e-3 * float(initial_error)
    np.testing.assert_allclose(
        np.asarray(state.mean[jnp.array(layout.pos_idx)]),
        np.asarray(true_position),
        atol=1e-3,
    )
