"""3D quaternion UKF rejection-contract tests.

Parallel to ``test_ekf_3d_analytic.py`` but locks in the *current*
behavior: the UKF entry point rejects every quaternion-orientation state
mode (``3d_quat`` and ``3d_cam_6dof_imu``) with ``NotImplementedError``
because sigma-point quaternion mean/covariance handling is not yet
implemented (see ``unscented_kalman_filter`` docstring and
``tests/filters/test_imu_shape_validation.py``).

The rejection happens at the very top of ``unscented_kalman_filter``
(before any input tensor is inspected — see ``ukf.py`` ~line 1039), so
each test just needs zero-filled IMU/camera arrays of the right shape.
We deliberately do *not* call ``simulate_3d`` here: a ~20 s sim per
test would be ~100 s of wasted CI time. Once the UKF gains 3D
quaternion support, swap ``_make_unused_inputs`` for ``simulate_3d``
and replace ``pytest.raises`` with analytic-bound assertions matching
the EKF tolerances (up to ~20% looser for sigma-point spread). The
scenario docstrings below preserve the analytic intent so the future
tests inherit the right parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter

# =============================================================================
# Helpers
# =============================================================================


def _make_ukf_3d_config() -> UKFConfig:
    """3D-quaternion UKF config (currently rejected at the entry point).

    Mirror of the EKF 3D config in ``test_ekf_3d_analytic.py`` so the
    downstream call site exercises the same shape contract once the UKF
    learns to handle quaternion orientation.
    """

    return UKFConfig(
        state_mode="3d_cam_6dof_imu",
        measurement_noise_pos=5e-3**2,
        process_noise_gyro_bias=1e-7,
        enable_experimental_accel_translation=True,
        enable_zupt=False,
        use_gravity_orientation_update=True,
        use_mahalanobis_gating=False,
    )


def _make_unused_inputs() -> dict:
    """Minimal IMU/camera arrays with the right shapes for the UKF entry point.

    Returns zero-filled tensors sized for a tiny 6-DOF IMU + 3D camera
    session (50 IMU samples, 15 camera frames). The entry point raises
    ``NotImplementedError`` before any value is consumed, so the
    contents are irrelevant — only the shapes matter, and even those
    aren't validated before the raise. Replace with ``simulate_3d(...)``
    when the UKF gains 3D quaternion support.
    """

    n_imu, n_cam = 50, 15
    return {
        "t_imu": np.linspace(0.0, 0.5, n_imu, dtype=np.float64),
        "U_imu": np.zeros((n_imu, 6), dtype=np.float64),
        "t_cam": np.linspace(0.0, 0.5, n_cam, dtype=np.float64),
        "Z_cam_led1": np.zeros((n_cam, 3), dtype=np.float64),
        "Z_cam_led2": np.zeros((n_cam, 3), dtype=np.float64),
        "mask_cam": np.ones(n_cam, dtype=bool),
    }


def _assert_ukf_3d_rejects() -> None:
    """Invoke the UKF with the 3D-quaternion config and assert it rejects."""

    config = _make_ukf_3d_config()
    inp = _make_unused_inputs()
    with pytest.raises(NotImplementedError, match="quaternion"):
        unscented_kalman_filter(
            config,
            inp["t_imu"],
            inp["U_imu"],
            inp["t_cam"],
            inp["Z_cam_led1"],
            inp["Z_cam_led2"],
            inp["mask_cam"],
        )


# =============================================================================
# Tests
# =============================================================================
#
# Each test corresponds 1:1 with an analytic test in test_ekf_3d_analytic.py
# (stationary pitch/roll, yaw-only bias observability, 5 s dropout drift,
# 4D NEES consistency, perfect-input idempotence). When the UKF gains 3D
# quaternion support, each test should switch from a rejection contract
# to a real analytic-bound assertion with the parameters captured in its
# docstring.


def test_ukf_3d_stationary_pitch_roll_rejects() -> None:
    """Stationary pitch=10° roll=5°, 30 s @ 100 Hz IMU.

    Future analytic bound: pitch/roll within 2° of truth, quaternion
    norm 1 ± 1e-6 (matching the EKF test, up to ~20% looser for
    sigma-point spread).
    """
    _assert_ukf_3d_rejects()


def test_ukf_3d_yaw_only_motion_rejects() -> None:
    """Yaw-only at 30 deg/s with gyro bias z=0.05, 30 s @ 100 Hz IMU.

    Future analytic bound: gyro-bias-z RMSE under the EKF target after
    the bias-observability window, ~20% looser than the EKF.
    """
    _assert_ukf_3d_rejects()


def test_ukf_3d_5s_dropout_rejects() -> None:
    """Yaw-only 30 s session with a 5 s camera dropout window (t=[10,15] s).

    Future analytic bound: drift ≤ 0.18 m (~20% looser than the EKF's
    0.15 m to account for sigma-point spread under no observations).
    """
    _assert_ukf_3d_rejects()


def test_ukf_3d_nees_consistency_rejects() -> None:
    """4D NEES session: yaw-only 30 s with small gyro bias and noise.

    Future analytic bound: NEES in [1.0, 9.6] (~20% looser top end
    than the EKF's [1.0, 8.0]).
    """
    _assert_ukf_3d_rejects()


def test_ukf_3d_perfect_input_rejects() -> None:
    """Perfect-input idempotence: stationary 10 s, all noise sources zero.

    Future analytic bound: posterior matches truth with ``rtol=1.2e-5``
    (~20% looser than the EKF's ``1e-5`` to account for sigma-point spread).
    """
    _assert_ukf_3d_rejects()
