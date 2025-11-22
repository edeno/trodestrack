from __future__ import annotations

import numpy as np

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.runtime.offline import rts_smoother, sigma_point_smoother


def _tiny_synthetic_sequence():
    # Three camera frames, evenly spaced; IMU at higher rate
    t_cam = np.array([0.0, 0.0333, 0.0666], dtype=np.float32)
    t_imu = np.linspace(0.0, 0.0666, 15, dtype=np.float32)
    U_imu = np.zeros((t_imu.shape[0], 3), dtype=np.float32)
    # Straight motion along x, LEDs 4 cm apart on x-axis
    Z1 = np.array([[0.00, 0.0], [0.01, 0.0], [0.02, 0.0]], dtype=np.float32)
    Z2 = Z1 + np.array([0.04, 0.0], dtype=np.float32)
    mask = np.array([True, True, True])
    return t_cam, t_imu, U_imu, Z1, Z2, mask


def test_rts_smoother_smoke_vision_only_layout():
    t_cam, t_imu, U_imu, Z1, Z2, mask = _tiny_synthetic_sequence()

    ekf_config = EKFConfig(
        state_mode="vision_only", led_distance=0.04, use_heading_measurement=False
    )
    filter_result = extended_kalman_filter(
        ekf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )
    smoother_result = rts_smoother(filter_result, ekf_config, t_imu, U_imu, t_cam)

    assert filter_result.filtered_means.shape[1] == 5
    assert smoother_result.smoothed_means.shape[1] == 5


def test_sigma_point_smoother_smoke_2d_cam_3d_imu_layout():
    t_cam, t_imu, U_imu, Z1, Z2, mask = _tiny_synthetic_sequence()

    ukf_config = UKFConfig(
        state_mode="2d_cam_3d_imu", led_distance=0.04, use_heading_measurement=False
    )
    filter_result = unscented_kalman_filter(
        ukf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )
    smoother_result = sigma_point_smoother(
        filter_result, ukf_config, t_imu, U_imu, t_cam
    )

    assert filter_result.filtered_means.shape[1] == 10
    assert smoother_result.smoothed_means.shape[1] == 10


def test_ukf_layout_no_hardcoded_8d():
    # Ensure UKF works with 10D without indexing errors.
    t_cam, t_imu, U_imu, Z1, Z2, mask = _tiny_synthetic_sequence()
    ukf_config = UKFConfig(
        state_mode="2d_cam_3d_imu", led_distance=0.04, use_heading_measurement=True
    )
    filter_result = unscented_kalman_filter(
        ukf_config, t_imu, U_imu, t_cam, Z1, Z2, mask, initial_state=None, conf_cam=None
    )
    assert filter_result.filtered_means.shape == (3, 10)
    assert filter_result.predicted_covariances.shape == (3, 10, 10)
