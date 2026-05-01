"""Tests for IMU input shape validation at filter/smoother entrypoints.

Rationale
---------
Before this guardrail, passing a 6-channel IMU array [ω_x, ω_y, ω_z, f_x, f_y, f_z]
(e.g. from ``load_arthur_session(mode='3d')``) into a filter that expects the
4-channel form [ω_z, f_x, f_y, f_z] would silently misinterpret channels --
``dynamics_function`` at filter_common.py:530 auto-detects "3D" as
``imu.shape[0] >= 4`` and then indexes ``imu[1], imu[2], imu[3]``, turning the
6-channel loader output into [ω_y, ω_z, f_x] instead of the intended
[f_x, f_y, f_z]. The filter then produces numerically plausible but wrong
estimates.

These tests pin the validator behaviour: correct shapes pass through; wrong
shapes raise a ``ValueError`` whose message tells the user exactly what the
expected channel layout is and (for the 6-channel case) how to reduce the
loader output.
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.runtime.offline import rts_smoother, sigma_point_smoother
from trodestrack.sim.simple import SimpleSimConfig, simulate_circular

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_2d():
    """Short 2D simulation producing a 3-channel IMU array."""
    sim_cfg = SimpleSimConfig(
        duration_s=1.0, fs_cam=30.0, fs_imu=200.0, cam_dropout_prob=0.0
    )
    return simulate_circular(config=sim_cfg, radius=0.5, seed=7)


# -----------------------------------------------------------------------------
# EKF entrypoint
# -----------------------------------------------------------------------------


def _run_ekf(sim, state_mode, U_imu):
    cfg = EKFConfig(state_mode=state_mode)
    return extended_kalman_filter(
        ekf_config=cfg,
        t_imu=sim["t_imu"],
        U_imu=U_imu,
        t_cam=sim["t_cam_obs"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )


def test_ekf_accepts_correct_3_channel_imu_for_2d_layout(sim_2d):
    """Happy path: 3-channel IMU with 2d_full layout must succeed."""
    result = _run_ekf(sim_2d, "2d_full", sim_2d["U_imu"])
    assert np.all(np.isfinite(np.asarray(result.filtered_means)))


def test_ekf_rejects_6_channel_imu_for_2d_layout(sim_2d):
    """Passing a 6-channel IMU into a 2D layout must raise, not silently run.

    This is the loader-output shape from ``load_arthur_session(mode='3d')``.
    """
    u = sim_2d["U_imu"]  # (N, 3)
    # Fabricate a 6-channel array of shape (N, 6)
    u6 = np.concatenate([u, u], axis=1)
    with pytest.raises(ValueError, match="6 channels"):
        _run_ekf(sim_2d, "2d_full", u6)


def test_ekf_rejects_6_channel_imu_for_3d_imu_layout(sim_2d):
    """Passing a 6-channel IMU into the 2d_cam_3d_imu layout must raise.

    This is the critical real-data bug: the 3D IMU layout expects 4 channels
    [ω_z, f_x, f_y, f_z] but the loader outputs 6 channels; without this guard
    the dynamics silently reads [ω_y, ω_z, f_x] as [f_x, f_y, f_z].
    """
    u = sim_2d["U_imu"]
    u6 = np.concatenate([u, u], axis=1)
    with pytest.raises(ValueError, match="6 channels"):
        _run_ekf(sim_2d, "2d_cam_3d_imu", u6)


def test_ekf_accepts_3_channel_imu_for_3d_imu_layout(sim_2d):
    """Degenerate but supported: 3-ch IMU + 3D-velocity layout runs the 2D
    dynamics branch and leaves ``vz`` idle. This must not raise."""
    result = _run_ekf(sim_2d, "2d_cam_3d_imu", sim_2d["U_imu"])
    assert np.all(np.isfinite(np.asarray(result.filtered_means)))


def test_ekf_rejects_4_channel_imu_for_2d_layout(sim_2d):
    """4-channel IMU with a 2D-velocity layout would silently drop ``f_z``."""
    u = sim_2d["U_imu"]  # (N, 3)
    u4 = np.concatenate([u, u[:, :1]], axis=1)  # fake 4th channel
    with pytest.raises(ValueError, match="f_z would be silently dropped"):
        _run_ekf(sim_2d, "2d_full", u4)


def test_ekf_rejects_1d_imu(sim_2d):
    """U_imu must be 2D (N, n_channels); a 1D array must raise."""
    u1d = sim_2d["U_imu"][:, 0]  # (N,)
    with pytest.raises(ValueError, match="2-D array"):
        _run_ekf(sim_2d, "2d_full", u1d)


def test_ekf_accepts_6_channel_imu_for_6dof_orientation_layout(sim_2d):
    """Quaternion orientation mode consumes full 6-axis IMU samples."""
    u = sim_2d["U_imu"]
    u6 = np.column_stack(
        [
            np.zeros((u.shape[0], 2)),
            u[:, 0],
            u[:, 1],
            u[:, 2],
            np.full(u.shape[0], 9.81),
        ]
    )

    result = _run_ekf(sim_2d, "2d_cam_6dof_imu_orientation", u6)

    assert result.filtered_means.shape[1] == 14
    assert np.all(np.isfinite(np.asarray(result.filtered_means)))


def test_ekf_rejects_3_channel_imu_for_6dof_orientation_layout(sim_2d):
    with pytest.raises(ValueError, match="requires 6-channel IMU"):
        _run_ekf(sim_2d, "2d_cam_6dof_imu_orientation", sim_2d["U_imu"])


# -----------------------------------------------------------------------------
# UKF entrypoint
# -----------------------------------------------------------------------------


def _run_ukf(sim, state_mode, U_imu):
    cfg = UKFConfig(state_mode=state_mode)
    return unscented_kalman_filter(
        ukf_config=cfg,
        t_imu=sim["t_imu"],
        U_imu=U_imu,
        t_cam=sim["t_cam_obs"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )


def test_ukf_accepts_correct_3_channel_imu_for_2d_layout(sim_2d):
    result = _run_ukf(sim_2d, "2d_full", sim_2d["U_imu"])
    assert np.all(np.isfinite(np.asarray(result.filtered_means)))


def test_ukf_rejects_6_channel_imu_for_2d_layout(sim_2d):
    u = sim_2d["U_imu"]
    u6 = np.concatenate([u, u], axis=1)
    with pytest.raises(ValueError, match="6 channels"):
        _run_ukf(sim_2d, "2d_full", u6)


def test_ukf_rejects_6_channel_imu_for_3d_imu_layout(sim_2d):
    u = sim_2d["U_imu"]
    u6 = np.concatenate([u, u], axis=1)
    with pytest.raises(ValueError, match="6 channels"):
        _run_ukf(sim_2d, "2d_cam_3d_imu", u6)


def test_ukf_rejects_6dof_orientation_layout_until_quaternion_mean_exists(sim_2d):
    u = sim_2d["U_imu"]
    u6 = np.column_stack(
        [
            np.zeros((u.shape[0], 2)),
            u[:, 0],
            u[:, 1],
            u[:, 2],
            np.full(u.shape[0], 9.81),
        ]
    )

    with pytest.raises(NotImplementedError, match="quaternion"):
        _run_ukf(sim_2d, "2d_cam_6dof_imu_orientation", u6)


# -----------------------------------------------------------------------------
# Error-message content
# -----------------------------------------------------------------------------


def test_error_message_names_entrypoint_and_hints_loader(sim_2d):
    """Error must mention the filter function and give a loader-output hint."""
    u6 = np.concatenate([sim_2d["U_imu"], sim_2d["U_imu"]], axis=1)
    with pytest.raises(ValueError) as excinfo:
        _run_ekf(sim_2d, "2d_cam_3d_imu", u6)
    msg = str(excinfo.value)
    # Expected channel count + naming of channels
    assert "4" in msg and "ω_z" in msg and "f_z" in msg
    # Loader-output hint for the 6-channel case (the bug scenario)
    assert "load_arthur_session" in msg or "6 channels" in msg


# -----------------------------------------------------------------------------
# Smoother entrypoints
# -----------------------------------------------------------------------------


def test_rts_smoother_rejects_wrong_imu_shape(sim_2d):
    """RTS smoother (EKF) must also validate IMU shape."""
    cfg = EKFConfig(state_mode="2d_full")
    good_filter = extended_kalman_filter(
        ekf_config=cfg,
        t_imu=sim_2d["t_imu"],
        U_imu=sim_2d["U_imu"],
        t_cam=sim_2d["t_cam_obs"],
        Z_cam_led1=sim_2d["Z_cam_led1"],
        Z_cam_led2=sim_2d["Z_cam_led2"],
        mask_cam=sim_2d["mask_cam"],
    )
    u6 = np.concatenate([sim_2d["U_imu"], sim_2d["U_imu"]], axis=1)
    with pytest.raises(ValueError, match="6 channels"):
        rts_smoother(
            filter_result=good_filter,
            ekf_config=cfg,
            t_imu=sim_2d["t_imu"],
            U_imu=u6,
            t_cam=sim_2d["t_cam_obs"],
        )


def test_sigma_smoother_rejects_wrong_imu_shape(sim_2d):
    """Sigma-point smoother (UKF) must also validate IMU shape."""
    cfg = UKFConfig(state_mode="2d_full")
    good_filter = unscented_kalman_filter(
        ukf_config=cfg,
        t_imu=sim_2d["t_imu"],
        U_imu=sim_2d["U_imu"],
        t_cam=sim_2d["t_cam_obs"],
        Z_cam_led1=sim_2d["Z_cam_led1"],
        Z_cam_led2=sim_2d["Z_cam_led2"],
        mask_cam=sim_2d["mask_cam"],
    )
    u6 = np.concatenate([sim_2d["U_imu"], sim_2d["U_imu"]], axis=1)
    with pytest.raises(ValueError, match="6 channels"):
        sigma_point_smoother(
            filter_result=good_filter,
            ukf_config=cfg,
            t_imu=sim_2d["t_imu"],
            U_imu=u6,
            t_cam=sim_2d["t_cam_obs"],
        )
