import numpy as np
import jax.numpy as jnp

import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.runtime.offline import rts_smoother, sigma_point_smoother
from trodestrack.sim.simple import SimpleSimConfig, simulate_circular


def reference_Q(cfg, theta: float, dt: float, has_vision: bool) -> jnp.ndarray:
    dtype = jnp.float32
    # Base diag diffusion
    q_px = jnp.asarray(cfg.process_noise_pos * dt, dtype=dtype)
    q_py = jnp.asarray(cfg.process_noise_pos * dt, dtype=dtype)
    q_vx = jnp.asarray(cfg.process_noise_vel * dt, dtype=dtype)
    q_vy = jnp.asarray(cfg.process_noise_vel * dt, dtype=dtype)
    q_th = jnp.asarray(cfg.process_noise_heading * dt, dtype=dtype)
    q_bg = jnp.asarray(cfg.process_noise_gyro_bias * dt, dtype=dtype)
    q_bax = jnp.asarray(cfg.process_noise_accel_bias * dt, dtype=dtype)
    q_bay = jnp.asarray(cfg.process_noise_accel_bias * dt, dtype=dtype)
    if getattr(cfg, "adaptive_q_during_dropout", False) and (not has_vision):
        q_px = q_px * cfg.dropout_q_pos_multiplier
        q_py = q_py * cfg.dropout_q_pos_multiplier
        q_vx = q_vx * cfg.dropout_q_vel_multiplier
        q_vy = q_vy * cfg.dropout_q_vel_multiplier
        q_bg = q_bg * cfg.dropout_q_bias_multiplier
        q_bax = q_bax * cfg.dropout_q_bias_multiplier
        q_bay = q_bay * cfg.dropout_q_bias_multiplier
    Q_proc = jnp.diag(jnp.array([q_px, q_py, q_vx, q_vy, q_th, q_bg, q_bax, q_bay], dtype=dtype))
    # IMU input mapping
    std_w = cfg.imu_gyro_noise_density * np.sqrt(dt)
    std_f = cfg.imu_accel_noise_density * np.sqrt(dt)
    Qu = jnp.diag(jnp.array([std_w**2, std_f**2, std_f**2], dtype=dtype))
    if getattr(cfg, "reduce_imu_noise_during_blackout", False) and (not has_vision):
        Qu = Qu * cfg.blackout_imu_noise_scale
    c, s = jnp.cos(theta), jnp.sin(theta)
    G = jnp.zeros((8, 3), dtype=dtype)
    G = G.at[2, 1].set(dt * c).at[2, 2].set(-dt * s)
    G = G.at[3, 1].set(dt * s).at[3, 2].set(dt * c)
    G = G.at[4, 0].set(dt)
    Q = Q_proc + G @ Qu @ G.T
    if getattr(cfg, "freeze_bias_during_blackout", False) and (not has_vision):
        for idx in (5, 6, 7):
            Q = Q.at[idx, :].set(0.0)
            Q = Q.at[:, idx].set(0.0)
    return 0.5 * (Q + Q.T)


@pytest.mark.slow
@pytest.mark.parametrize("dropout_prob", [0.0, 0.4])
def test_rts_smoother_parity(monkeypatch, dropout_prob):
    """Parity check between new assemble_Q smoother and reference Q for EKF RTS.

    Runtime (observed locally): ~6.2 s
    """
    # Simulate circular motion to exercise heading and bias dynamics
    sim_cfg = SimpleSimConfig(
        duration_s=3.0, fs_imu=200.0, fs_cam=30.0, cam_dropout_prob=dropout_prob
    )
    sim = simulate_circular(sim_cfg, seed=7)

    cfg = EKFConfig(
        use_mahalanobis_gating=True,
        adaptive_q_during_dropout=True,
        reduce_imu_noise_during_blackout=True,
        freeze_bias_during_blackout=True,
    )

    filt = extended_kalman_filter(
        ekf_config=cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # New smoother (assemble_Q)
    sm_new = rts_smoother(
        filter_result=filt,
        ekf_config=cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        mask_cam=sim["mask_cam"],
    )

    # Old-equivalent: monkeypatch assemble_Q to reference
    import trodestrack.models.process_noise as pn

    def assemble_Q_ref(config, theta, dt, n, *, has_vision, dtype=jnp.float32):
        assert n == 8
        return reference_Q(config, theta, dt, has_vision).astype(dtype)

    monkeypatch.setattr(pn, "assemble_Q", assemble_Q_ref)

    sm_old = rts_smoother(
        filter_result=filt,
        ekf_config=cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        mask_cam=sim["mask_cam"],
    )

    assert jnp.allclose(sm_new.smoothed_means, sm_old.smoothed_means, rtol=1e-5, atol=1e-7)
    assert jnp.allclose(
        sm_new.smoothed_covariances, sm_old.smoothed_covariances, rtol=1e-5, atol=1e-7
    )


@pytest.mark.slow
@pytest.mark.parametrize("dropout_prob", [0.0, 0.4])
def test_sigma_point_smoother_parity(monkeypatch, dropout_prob):
    """Parity check between new assemble_Q smoother and reference Q for UKF sigma-point smoother.

    Runtime (observed locally): ~6.3 s
    """
    sim_cfg = SimpleSimConfig(
        duration_s=3.0, fs_imu=200.0, fs_cam=30.0, cam_dropout_prob=dropout_prob
    )
    sim = simulate_circular(sim_cfg, seed=11)

    cfg = UKFConfig(
        use_mahalanobis_gating=True,
        adaptive_q_during_dropout=True,
        reduce_imu_noise_during_blackout=True,
        freeze_bias_during_blackout=True,
    )

    filt = unscented_kalman_filter(
        ukf_config=cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    sm_new = sigma_point_smoother(
        filter_result=filt,
        ukf_config=cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        mask_cam=sim["mask_cam"],
    )

    import trodestrack.models.process_noise as pn

    def assemble_Q_ref(config, theta, dt, n, *, has_vision, dtype=jnp.float32):
        assert n == 8
        return reference_Q(config, theta, dt, has_vision).astype(dtype)

    monkeypatch.setattr(pn, "assemble_Q", assemble_Q_ref)

    sm_old = sigma_point_smoother(
        filter_result=filt,
        ukf_config=cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        mask_cam=sim["mask_cam"],
    )

    assert jnp.allclose(sm_new.smoothed_means, sm_old.smoothed_means, rtol=1e-5, atol=1e-7)
    assert jnp.allclose(
        sm_new.smoothed_covariances, sm_old.smoothed_covariances, rtol=1e-5, atol=1e-7
    )
