from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.filter_common import initialize_state, measurement_function
from trodestrack.models.state_layout import get_layout
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter


def test_initialize_state_vision_only_layout() -> None:
    layout = get_layout("vision_only")

    # Two frames, one valid, one invalid
    led1 = jnp.array([[0.1, 0.2], [jnp.nan, jnp.nan]], dtype=jnp.float32)
    led2 = jnp.array([[0.14, 0.2], [jnp.nan, jnp.nan]], dtype=jnp.float32)
    mask = jnp.array([True, False])

    state = initialize_state(
        led1, led2, mask, dt_cam=jnp.array(1 / 30.0), led_distance=0.04, layout=layout
    )

    # Expect 5D state for vision_only
    assert state.mean.shape[0] == layout.n == 5

    # Position mapped to layout indices
    assert np.isfinite(np.array(state.mean[layout.pos_idx[0]])).all()
    assert np.isfinite(np.array(state.mean[layout.pos_idx[1]])).all()


def test_measurement_function_respects_layout_indices() -> None:
    layout = get_layout("2d_cam_3d_imu")

    # Build a state with known position and heading at layout indices
    n = layout.n
    m = jnp.zeros(n)
    px_i, py_i = layout.pos_idx[0], layout.pos_idx[1]
    h_i = layout.heading_idx  # type: ignore[assignment]
    m = m.at[px_i].set(1.0)
    m = m.at[py_i].set(2.0)
    m = m.at[h_i].set(0.0)  # cos=1, sin=0

    z = measurement_function(m, 0.04, layout)
    # LED positions should be symmetric around (x,y) along x-axis
    assert np.allclose(np.array(z[0]), 1.0 - 0.02, atol=1e-6)
    assert np.allclose(np.array(z[2]), 1.0 + 0.02, atol=1e-6)
    assert np.allclose(np.array(z[1]), 2.0, atol=1e-6)
    assert np.allclose(np.array(z[3]), 2.0, atol=1e-6)


def test_ekf_smoke_vision_only_shapes() -> None:
    ekf_config = EKFConfig(
        state_mode="vision_only", led_distance=0.04, use_heading_measurement=False
    )

    # Small 3-frame scenario
    t_cam = np.array([0.0, 0.0333, 0.0666], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.0666, 15, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 3), dtype=np.float32)

    # Simple forward motion in camera
    Z1 = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    mask = np.array([True, True, True])

    filter_result = extended_kalman_filter(
        ekf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )

    # Shapes reflect 5D state
    assert filter_result.filtered_means.shape == (3, get_layout("vision_only").n)
    assert filter_result.predicted_means.shape == (3, get_layout("vision_only").n)


def test_ukf_smoke_2d_cam_3d_imu_shapes() -> None:
    ukf_config = UKFConfig(
        state_mode="2d_cam_3d_imu", led_distance=0.04, use_heading_measurement=False
    )

    # Small 3-frame scenario
    t_cam = np.array([0.0, 0.0333, 0.0666], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.0666, 15, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 3), dtype=np.float32)

    # Simple forward motion in camera
    Z1 = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    mask = np.array([True, True, True])

    filter_result = unscented_kalman_filter(
        ukf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )

    # Shapes reflect 10D state
    assert filter_result.filtered_means.shape == (3, get_layout("2d_cam_3d_imu").n)
    assert filter_result.predicted_means.shape == (3, get_layout("2d_cam_3d_imu").n)
