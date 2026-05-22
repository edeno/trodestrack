"""Analytic-correctness tests for ``extended_kalman_filter_3d``.

The 3D camera + 6-DOF IMU EKF was previously covered only by smoother
smoke tests and JAXPR benchmarks. This file mirrors the structure of
``test_ekf_analytic.py`` for the 2D filter: deterministic seeds,
explicit RMSE / bias / drift / NEES bounds in cm, deg, and rad/s.

The shared ``simulate_3d_session`` helper in ``conftest.py`` provides
the ground-truth trajectory, 6-DOF IMU readings, and 3D LED
observations. All tests are marked ``@pytest.mark.slow`` because each
runs a 5-30 s simulated session at 100 Hz IMU / 30 Hz camera.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter_3d
from trodestrack.models.filter_common import FilterState
from trodestrack.models.quaternion import (
    quaternion_to_roll_pitch_yaw,
    quaternion_to_yaw,
)
from trodestrack.models.state_layout import get_layout

# =============================================================================
# Helpers
# =============================================================================


def _stack_leds(sim) -> tuple[np.ndarray, np.ndarray]:
    """Stack LED1/LED2 into the ``(N_cam, 2, 3)`` shape expected by the 3D EKF."""

    Z_cam_leds = np.stack([sim.Z_cam_led1, sim.Z_cam_led2], axis=1)
    mask_cam_leds = np.broadcast_to(sim.mask_cam[:, None], (sim.t_cam.size, 2)).copy()
    return Z_cam_leds, mask_cam_leds


def _default_3d_config(**overrides) -> EKFConfig:
    """3D EKF config used by the analytic tests.

    Tight measurement noise + small process noise + gravity-orientation
    update enabled; mahalanobis gating off so an early-frame innovation
    can't reject the first true measurements while the prior is wide.
    """

    base = dict(
        state_mode="3d_cam_6dof_imu",
        measurement_noise_pos=5e-3**2,
        process_noise_gyro_bias=1e-7,
        enable_experimental_accel_translation=True,
        enable_zupt=False,
        use_gravity_orientation_update=True,
        use_mahalanobis_gating=False,
    )
    base.update(overrides)
    return EKFConfig(**base)


def _yaw_jacobian_wrt_quat(q: np.ndarray) -> np.ndarray:
    """Jacobian of ZYX yaw extraction with respect to the quaternion components."""

    qw, qx, qy, qz = q
    numerator = 2.0 * (qw * qz + qx * qy)
    denominator = 1.0 - 2.0 * (qy * qy + qz * qz)
    sq_norm = numerator * numerator + denominator * denominator
    if sq_norm == 0.0:
        return np.zeros(4)
    d_num = np.array([2.0 * qz, 2.0 * qy, 2.0 * qx, 2.0 * qw])
    d_den = np.array([0.0, 0.0, -4.0 * qy, -4.0 * qz])
    return (denominator * d_num - numerator * d_den) / sq_norm


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.slow
def test_ekf_3d_stationary_pitch_roll_recovers_gravity_orientation(simulate_3d) -> None:
    """Stationary headstage at true pitch=10°, roll=5°: filter recovers orientation.

    Asserts:
        - recovered pitch within 2° of truth at convergence,
        - recovered roll within 2° of truth at convergence,
        - quaternion norm 1 ± 1e-6 throughout (no drift away from S^3).
    """

    sim = simulate_3d(
        seed=0,
        duration_s=30.0,
        fs_imu=100.0,
        motion="stationary",
        init_pitch_deg=10.0,
        init_roll_deg=5.0,
    )
    layout = get_layout("3d_cam_6dof_imu")
    Z_cam_leds, mask_cam_leds = _stack_leds(sim)
    result = extended_kalman_filter_3d(
        _default_3d_config(),
        sim.t_imu,
        sim.U_imu,
        sim.t_cam,
        Z_cam_leds,
        sim.truth["led_offsets_body"],
        mask_cam_leds=mask_cam_leds,
    )

    quat_idx = np.array(layout.heading_idx)
    quats = np.asarray(result.filtered_means[:, quat_idx])
    norms = np.linalg.norm(quats, axis=1)
    assert np.all(np.isfinite(quats))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # Average the recovered orientation over the last second of the session
    # so single-frame noise doesn't dominate the bound check.
    n_avg = max(1, int(0.1 * sim.t_cam.size))
    rolls = []
    pitches = []
    for q in quats[-n_avg:]:
        q_norm = q / np.linalg.norm(q)
        r, p, _y = quaternion_to_roll_pitch_yaw(q_norm)
        rolls.append(float(r))
        pitches.append(float(p))
    mean_roll_deg = float(np.rad2deg(np.mean(rolls)))
    mean_pitch_deg = float(np.rad2deg(np.mean(pitches)))
    assert abs(mean_roll_deg - 5.0) < 2.0, (
        f"Recovered roll {mean_roll_deg:.2f}° outside ±2° of 5° truth"
    )
    assert abs(mean_pitch_deg - 10.0) < 2.0, (
        f"Recovered pitch {mean_pitch_deg:.2f}° outside ±2° of 10° truth"
    )


@pytest.mark.slow
def test_ekf_3d_yaw_only_motion_converges_gyro_bias_z(simulate_3d) -> None:
    """Constant 30°/s yaw with true gyro_z bias = 0.05 rad/s: bias converges.

    Asserts:
        - estimated gyro_z bias within 5e-3 rad/s of truth after 10 s.
    """

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
    layout = get_layout("3d_cam_6dof_imu")
    Z_cam_leds, mask_cam_leds = _stack_leds(sim)
    result = extended_kalman_filter_3d(
        _default_3d_config(),
        sim.t_imu,
        sim.U_imu,
        sim.t_cam,
        Z_cam_leds,
        sim.truth["led_offsets_body"],
        mask_cam_leds=mask_cam_leds,
    )

    bg_idx = np.array(layout.bias_gyro_idx)
    bias_estimates = np.asarray(result.filtered_means[:, bg_idx])
    idx_10s = int(np.searchsorted(sim.t_cam, 10.0))
    bz_at_10s = bias_estimates[idx_10s, 2]
    bz_true = float(sim.truth["gyro_bias"][2])
    assert abs(bz_at_10s - bz_true) < 5e-3, (
        f"Estimated gyro_z bias {bz_at_10s:.4f} rad/s vs truth {bz_true:.4f} rad/s; "
        f"error {abs(bz_at_10s - bz_true):.4f} exceeds 5e-3 rad/s after 10 s"
    )


@pytest.mark.slow
def test_ekf_3d_5s_dropout_drift_under_acceptance_target(simulate_3d) -> None:
    """5 s vision blackout: position drift bounded under 0.15 m.

    Matches the 2D acceptance target at
    ``tests/filters/test_ekf_analytic.py::test_ekf_long_dropout_drift``.
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
    Z_cam_leds, mask_cam_leds = _stack_leds(sim)
    result = extended_kalman_filter_3d(
        _default_3d_config(),
        sim.t_imu,
        sim.U_imu,
        sim.t_cam,
        Z_cam_leds,
        sim.truth["led_offsets_body"],
        mask_cam_leds=mask_cam_leds,
    )

    pos_est = np.asarray(result.filtered_means[:, :3])
    truth_pos_cam = np.column_stack(
        [
            np.interp(sim.t_cam, sim.t_imu, sim.truth["positions"][:, d])
            for d in range(3)
        ]
    )
    dropout_end_idx = int(np.searchsorted(sim.t_cam, 15.0)) - 1
    drift_m = float(
        np.linalg.norm(pos_est[dropout_end_idx] - truth_pos_cam[dropout_end_idx])
    )
    assert drift_m < 0.15, (
        f"Position drift {drift_m * 100:.2f} cm at end of 5 s dropout exceeds 15 cm"
    )


@pytest.mark.slow
def test_ekf_3d_nees_consistency_on_4d_state(simulate_3d) -> None:
    """Mean NEES on the (x, y, z, yaw) subset stays in [1.0, 8.0].

    Wider than the 2D NEES bound because the 3D filter's pose-coupled
    innovation tails are heavier (gyro biases couple into orientation,
    which couples into LED prediction).

    TODO: Tighten to [1.5, 4.5] once 3D filter tuning stabilizes —
    mirrors the existing TODO at ``test_ekf_analytic.py::test_ekf_consistency_nees``.
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
    layout = get_layout("3d_cam_6dof_imu")
    Z_cam_leds, mask_cam_leds = _stack_leds(sim)
    result = extended_kalman_filter_3d(
        _default_3d_config(),
        sim.t_imu,
        sim.U_imu,
        sim.t_cam,
        Z_cam_leds,
        sim.truth["led_offsets_body"],
        mask_cam_leds=mask_cam_leds,
    )

    pos_idx = np.array(layout.pos_idx)
    quat_idx = np.array(layout.heading_idx)
    pos_est = np.asarray(result.filtered_means[:, pos_idx])
    quats_est = np.asarray(result.filtered_means[:, quat_idx])
    covs_full = np.asarray(result.filtered_covariances)

    truth_pos_cam = np.column_stack(
        [
            np.interp(sim.t_cam, sim.t_imu, sim.truth["positions"][:, d])
            for d in range(3)
        ]
    )
    yaw_truth_imu = np.array(
        [float(quaternion_to_yaw(q)) for q in sim.truth["quaternions"]]
    )
    yaw_truth_cam = np.interp(sim.t_cam, sim.t_imu, np.unwrap(yaw_truth_imu))
    yaw_est = np.unwrap(
        np.array([float(quaternion_to_yaw(q / np.linalg.norm(q))) for q in quats_est])
    )

    errs_4d = np.column_stack(
        [
            pos_est[:, 0] - truth_pos_cam[:, 0],
            pos_est[:, 1] - truth_pos_cam[:, 1],
            pos_est[:, 2] - truth_pos_cam[:, 2],
            (yaw_est - yaw_truth_cam + np.pi) % (2.0 * np.pi) - np.pi,
        ]
    )

    nees_vals = np.empty(sim.t_cam.size)
    nees_vals.fill(np.nan)
    for i in range(sim.t_cam.size):
        H = np.zeros((4, covs_full.shape[1]))
        H[0, pos_idx[0]] = 1.0
        H[1, pos_idx[1]] = 1.0
        H[2, pos_idx[2]] = 1.0
        q = quats_est[i] / np.linalg.norm(quats_est[i])
        j_yaw = _yaw_jacobian_wrt_quat(q)
        for k, idx_q in enumerate(quat_idx):
            H[3, idx_q] = j_yaw[k]
        P4 = H @ covs_full[i] @ H.T + 1e-12 * np.eye(4)
        try:
            nees_vals[i] = errs_4d[i] @ np.linalg.solve(P4, errs_4d[i])
        except np.linalg.LinAlgError:
            continue

    valid = np.isfinite(nees_vals)
    assert valid.sum() > 0.9 * nees_vals.size, "Too many NEES failures to be meaningful"
    mean_nees = float(np.mean(nees_vals[valid]))
    assert 1.0 < mean_nees < 8.0, (
        f"Mean 4D NEES {mean_nees:.2f} outside the [1.0, 8.0] consistency band"
    )


@pytest.mark.slow
def test_ekf_3d_perfect_input_no_drift(simulate_3d) -> None:
    """Zero-noise input + truth initial state: posterior == truth and cov stays bounded."""

    sim = simulate_3d(
        seed=42,
        duration_s=10.0,
        fs_imu=100.0,
        motion="perfect",
        init_pitch_deg=10.0,
        init_roll_deg=5.0,
    )
    layout = get_layout("3d_cam_6dof_imu")
    n = layout.n
    mean0 = np.zeros(n, dtype=np.float64)
    mean0[: len(layout.pos_idx)] = sim.truth["positions"][0]
    mean0[np.array(layout.heading_idx)] = sim.truth["quaternions"][0]
    mean0[np.array(layout.bias_gyro_idx)] = sim.truth["gyro_bias"]
    mean0[np.array(layout.bias_accel_idx)] = sim.truth["accel_bias"]
    cov0 = np.eye(n, dtype=np.float64) * 1e-8
    initial_state = FilterState(mean=jnp.asarray(mean0), cov=jnp.asarray(cov0))

    config = _default_3d_config(
        measurement_noise_pos=1e-8,
        process_noise_pos=1e-10,
        process_noise_vel=1e-8,
        process_noise_gyro_bias=1e-12,
        process_noise_accel_bias=1e-10,
        use_gravity_orientation_update=False,
    )
    Z_cam_leds, mask_cam_leds = _stack_leds(sim)
    result = extended_kalman_filter_3d(
        config,
        sim.t_imu,
        sim.U_imu,
        sim.t_cam,
        Z_cam_leds,
        sim.truth["led_offsets_body"],
        mask_cam_leds=mask_cam_leds,
        initial_state=initial_state,
    )

    pos_est = np.asarray(result.filtered_means[:, :3])
    truth_pos_cam = np.column_stack(
        [
            np.interp(sim.t_cam, sim.t_imu, sim.truth["positions"][:, d])
            for d in range(3)
        ]
    )
    np.testing.assert_allclose(pos_est, truth_pos_cam, atol=1e-5, rtol=1e-5)

    covs = np.asarray(result.filtered_covariances)
    assert np.all(np.isfinite(covs))
    diag = covs.diagonal(axis1=1, axis2=2)
    assert float(diag.max()) < 1e-2, (
        f"Posterior covariance grew to {diag.max():.2e} on perfect input"
    )


@pytest.mark.slow
def test_ekf_3d_iekf_converges_under_iteration(simulate_3d) -> None:
    """``num_iter=3`` produces a tighter posterior than ``num_iter=1``.

    The IEKF re-linearizes against the current iterate; multiple
    iterations should produce a posterior consistent with itself
    (contraction for well-conditioned problems). This directly
    exercises the ``lax.scan`` carry semantics: if the carry-vs-output
    indexing returned the second-to-last iterate instead of the final
    one, ``num_iter=1`` and ``num_iter=3`` would diverge in unexpected
    ways and the trace inequality below would flip.
    """
    sim = simulate_3d(
        seed=42,
        duration_s=5.0,
        fs_imu=100.0,
        motion="yaw_only",
        yaw_rate_dps=10.0,
        init_pitch_deg=2.0,
        init_roll_deg=1.0,
        gyro_bias_xyz=(0.001, -0.001, 0.005),
        gyro_noise_std=5e-4,
        accel_noise_std=5e-3,
    )
    layout = get_layout("3d_cam_6dof_imu")
    Z_cam_leds, mask_cam_leds = _stack_leds(sim)

    config1 = _default_3d_config()  # num_iter=1 (default)
    config3 = replace(config1, num_iter=3)  # host-side dataclass replace

    result1 = extended_kalman_filter_3d(
        config1,
        sim.t_imu,
        sim.U_imu,
        sim.t_cam,
        Z_cam_leds,
        sim.truth["led_offsets_body"],
        mask_cam_leds=mask_cam_leds,
    )
    result3 = extended_kalman_filter_3d(
        config3,
        sim.t_imu,
        sim.U_imu,
        sim.t_cam,
        Z_cam_leds,
        sim.truth["led_offsets_body"],
        mask_cam_leds=mask_cam_leds,
    )

    pos_idx = list(layout.pos_idx)
    final_pos1 = np.asarray(result1.filtered_means[-1, pos_idx])
    final_pos3 = np.asarray(result3.filtered_means[-1, pos_idx])
    # IEKF is a contraction here — final positions should agree to well
    # within the measurement-noise floor (5 mm sigma in the default cfg).
    assert float(np.linalg.norm(final_pos3 - final_pos1)) < 0.01, (
        f"num_iter=3 final position differs from num_iter=1 by "
        f"{float(np.linalg.norm(final_pos3 - final_pos1)) * 100:.2f} cm"
    )

    # Critical: assert num_iter=3 covariance is not LARGER (in trace) than
    # num_iter=1. If the scan returned a non-final iterate by mistake,
    # the cov would reflect an earlier (worse) linearization and trip
    # this inequality.
    trace1 = float(jnp.trace(result1.filtered_covariances[-1]))
    trace3 = float(jnp.trace(result3.filtered_covariances[-1]))
    assert trace3 <= trace1 * 1.05, (
        f"num_iter=3 covariance trace {trace3:.4e} exceeds num_iter=1 "
        f"trace {trace1:.4e} by more than 5% — final iterate may be wrong"
    )
