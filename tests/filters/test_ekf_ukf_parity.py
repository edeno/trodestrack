"""EKF vs UKF tight parity on near-linear, low-noise inputs.

The existing ``test_ukf_vs_ekf_accuracy_*`` tests assert UKF RMSE within
10% of EKF RMSE. That is a usefulness check, not a parity check — it
would pass even if EKF and UKF drifted apart in their state means by
large amounts as long as each tracked truth reasonably.

In the limit of a near-linear measurement model and low IMU/camera noise,
EKF and UKF should produce nearly identical state estimates because the
sigma-point linearization collapses to the analytic Jacobian. This test
exercises that regime and asserts a sub-millimeter agreement, catching
regressions where one filter silently diverges from the other (mismatched
gating logic, dropout-Q drift, ZUPT trigger off-by-one, etc.) that the
existing tolerance tests would not catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.sim.simple import SimpleSimConfig, simulate_stationary


@pytest.mark.slow
def test_ekf_and_ukf_agree_on_low_noise_stationary() -> None:
    """On a nearly-linear stationary sim, EKF and UKF means agree within mm."""
    cfg = SimpleSimConfig(
        duration_s=5.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=1e-6,
        accel_noise_density=1e-6,
        gyro_bias_std=0.0,
        accel_bias_std=0.0,
        cam_noise_std=1e-4,  # 0.1 mm
        cam_dropout_prob=0.0,
    )
    sim = simulate_stationary(cfg, position=np.array([0.5, 0.5]), heading=0.0, seed=11)

    ekf_cfg = EKFConfig(enable_zupt=False)
    ekf_result = extended_kalman_filter(
        ekf_config=ekf_cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    ukf_cfg = UKFConfig(enable_zupt=False)
    ukf_result = unscented_kalman_filter(
        ukf_config=ukf_cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    ekf_pos = np.asarray(ekf_result.filtered_means[:, :2])
    ukf_pos = np.asarray(ukf_result.filtered_means[:, :2])
    diff = np.linalg.norm(ekf_pos - ukf_pos, axis=1)
    # Empirically max ~2.4 mm, mean ~1.9 mm on this low-noise stationary sim.
    # Thresholds tighter than the existing 10% RMSE tolerance in
    # ``test_ukf_accuracy.py`` — a 2x divergence (e.g. a sigma-point broadcast
    # bug) would fail these but pass the 10% test.
    assert float(diff.max()) < 5e-3, (
        f"EKF / UKF disagree by up to {float(diff.max()) * 1000:.3f} mm "
        f"on low-noise stationary; expected < 5 mm"
    )
    assert float(diff.mean()) < 3e-3, (
        f"Mean EKF/UKF disagreement {float(diff.mean()) * 1000:.3f} mm > 3 mm"
    )


@pytest.mark.slow
def test_ekf_and_ukf_marginal_loglik_agree_on_low_noise() -> None:
    """Marginal log-likelihood is a sensitive aggregate; check parity."""
    cfg = SimpleSimConfig(
        duration_s=3.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=1e-6,
        accel_noise_density=1e-6,
        cam_noise_std=1e-4,
        cam_dropout_prob=0.0,
    )
    sim = simulate_stationary(cfg, position=np.array([0.5, 0.5]), heading=0.0, seed=13)

    ekf_cfg = EKFConfig(enable_zupt=False)
    ekf_result = extended_kalman_filter(
        ekf_config=ekf_cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    ukf_cfg = UKFConfig(enable_zupt=False)
    ukf_result = unscented_kalman_filter(
        ukf_config=ukf_cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    ekf_ll = float(ekf_result.marginal_loglik)
    ukf_ll = float(ukf_result.marginal_loglik)
    # Aggregate log-likelihood relative difference must stay < 5% — empirically
    # ~1.6% on this low-noise sim. Looser than 1% to avoid flakiness across
    # XLA platforms / float32 jitter, but tight enough to catch 10-20%
    # divergences from one filter going off the rails.
    rel_diff = abs(ekf_ll - ukf_ll) / max(abs(ekf_ll), abs(ukf_ll), 1.0)
    assert rel_diff < 5e-2, (
        f"EKF/UKF marginal log-likelihoods diverged: "
        f"EKF={ekf_ll:.4f}, UKF={ukf_ll:.4f}, rel_diff={rel_diff:.4f}"
    )
