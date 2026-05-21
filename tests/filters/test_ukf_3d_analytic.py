"""3D quaternion UKF analytic-correctness tests.

Parallel to ``test_ekf_3d_analytic.py``. Each test exercises the same
scenario from ``simulate_3d_session`` but routes the call through
``unscented_kalman_filter``.

Current behavior: the UKF entry point rejects every quaternion-orientation
state mode (``3d_quat`` and ``3d_cam_6dof_imu``) with ``NotImplementedError``
because sigma-point quaternion mean/covariance handling is not yet
implemented (see ``unscented_kalman_filter`` docstring and
``tests/filters/test_imu_shape_validation.py``). These tests lock that
contract in: each one runs the same simulator setup as its EKF
counterpart and asserts the UKF raises ``NotImplementedError`` with a
quaternion-related message rather than silently downgrading or producing
wrong output. When the UKF gains 3D quaternion support, these stubs will
fail and force real analytic bounds (matching the EKF tolerances, up to
~20% looser to account for sigma-point spread) to be written.

The simulator calls themselves still allocate the full 6-DOF IMU + 3D
LED tensors and the resulting fixtures end up shaped the way the
post-UKF-3D tests will need, so this file stays usable as the skeleton
once the UKF gains 3D quaternion support — only the call site swaps from
``pytest.raises`` to an analytic-bound assertion.
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


def _run_ukf_3d_stub(sim) -> None:
    """Invoke ``unscented_kalman_filter`` with 3D-quaternion config.

    The 2D ``Z_cam_led*`` arguments are passed through unchanged because
    the entry point rejects the configuration before it inspects the
    measurement tensors — so any shape works for the rejection contract.
    """

    config = _make_ukf_3d_config()
    unscented_kalman_filter(
        config,
        sim.t_imu,
        sim.U_imu,
        sim.t_cam,
        sim.Z_cam_led1,
        sim.Z_cam_led2,
        sim.mask_cam,
    )


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.slow
def test_ukf_3d_stationary_pitch_roll_recovers_gravity_orientation(simulate_3d) -> None:
    """Stationary pitch/roll session: UKF currently rejects 3D quaternion modes."""

    sim = simulate_3d(
        seed=0,
        duration_s=30.0,
        fs_imu=100.0,
        motion="stationary",
        init_pitch_deg=10.0,
        init_roll_deg=5.0,
    )
    with pytest.raises(NotImplementedError, match="quaternion"):
        _run_ukf_3d_stub(sim)


@pytest.mark.slow
def test_ukf_3d_yaw_only_motion_converges_gyro_bias_z(simulate_3d) -> None:
    """Yaw-only bias-observability session: UKF currently rejects 3D quaternion modes."""

    sim = simulate_3d(
        seed=0,
        duration_s=30.0,
        fs_imu=100.0,
        motion="yaw_only",
        yaw_rate_dps=30.0,
        init_pitch_deg=2.0,
        init_roll_deg=1.0,
        gyro_bias_xyz=(0.0, 0.0, 0.05),
        gyro_noise_std=5e-4,
        accel_noise_std=5e-3,
    )
    with pytest.raises(NotImplementedError, match="quaternion"):
        _run_ukf_3d_stub(sim)


@pytest.mark.slow
def test_ukf_3d_5s_dropout_drift_under_acceptance_target(simulate_3d) -> None:
    """5 s dropout session: UKF currently rejects 3D quaternion modes.

    Once the UKF supports 3D quaternion, the analytic bound should be
    drift ≤ 0.18 m (~20% looser than the EKF's 0.15 m bound to account
    for sigma-point spread under no observations).
    """

    sim = simulate_3d(
        seed=0,
        duration_s=30.0,
        fs_imu=100.0,
        motion="yaw_only",
        yaw_rate_dps=10.0,
        init_pitch_deg=2.0,
        init_roll_deg=1.0,
        gyro_bias_xyz=(0.001, -0.001, 0.005),
        gyro_noise_std=5e-4,
        accel_noise_std=5e-3,
        dropout_window_s=(10.0, 15.0),
    )
    with pytest.raises(NotImplementedError, match="quaternion"):
        _run_ukf_3d_stub(sim)


@pytest.mark.slow
def test_ukf_3d_nees_consistency_on_4d_state(simulate_3d) -> None:
    """4D NEES session: UKF currently rejects 3D quaternion modes.

    Once supported, NEES bound should be [1.0, 9.6] (~20% looser top end
    than the EKF's [1.0, 8.0]). TODO: Replace this stub with a real NEES
    assertion when ``unscented_kalman_filter`` learns sigma-point
    quaternion handling.
    """

    sim = simulate_3d(
        seed=0,
        duration_s=30.0,
        fs_imu=100.0,
        motion="yaw_only",
        yaw_rate_dps=10.0,
        init_pitch_deg=2.0,
        init_roll_deg=1.0,
        gyro_bias_xyz=(0.001, -0.001, 0.005),
        gyro_noise_std=5e-4,
        accel_noise_std=5e-3,
    )
    with pytest.raises(NotImplementedError, match="quaternion"):
        _run_ukf_3d_stub(sim)


@pytest.mark.slow
def test_ukf_3d_perfect_input_no_drift(simulate_3d) -> None:
    """Perfect-input idempotence session: UKF currently rejects 3D quaternion modes.

    Once supported, the posterior-vs-truth tolerance should be ``rtol=1.2e-5``
    (~20% looser than the EKF's ``1e-5`` to account for sigma-point spread).
    """

    sim = simulate_3d(
        seed=42,
        duration_s=10.0,
        fs_imu=100.0,
        motion="perfect",
        init_pitch_deg=10.0,
        init_roll_deg=5.0,
    )
    # Touch ``sim`` so ruff doesn't flag the parameter as unused once the
    # entry-point check is replaced with a real filter call.
    assert np.isfinite(sim.U_imu).all()
    with pytest.raises(NotImplementedError, match="quaternion"):
        _run_ukf_3d_stub(sim)
