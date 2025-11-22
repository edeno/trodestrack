import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.simple import SimpleSimConfig, simulate_constant_velocity


def reference_Q(
    cfg: EKFConfig, theta: float, dt: float, has_vision: bool
) -> jnp.ndarray:
    """Pre-refactor EKF Q assembly (used for parity checks)."""
    dtype = jnp.float32

    # Base diagonal diffusion scaled by dt
    q_px = jnp.asarray(cfg.process_noise_pos * dt, dtype=dtype)
    q_py = jnp.asarray(cfg.process_noise_pos * dt, dtype=dtype)
    q_vx = jnp.asarray(cfg.process_noise_vel * dt, dtype=dtype)
    q_vy = jnp.asarray(cfg.process_noise_vel * dt, dtype=dtype)
    q_th = jnp.asarray(cfg.process_noise_heading * dt, dtype=dtype)
    q_bg = jnp.asarray(cfg.process_noise_gyro_bias * dt, dtype=dtype)
    q_bax = jnp.asarray(cfg.process_noise_accel_bias * dt, dtype=dtype)
    q_bay = jnp.asarray(cfg.process_noise_accel_bias * dt, dtype=dtype)

    if cfg.adaptive_q_during_dropout and (not has_vision):
        q_px = q_px * cfg.dropout_q_pos_multiplier
        q_py = q_py * cfg.dropout_q_pos_multiplier
        q_vx = q_vx * cfg.dropout_q_vel_multiplier
        q_vy = q_vy * cfg.dropout_q_vel_multiplier
        q_bg = q_bg * cfg.dropout_q_bias_multiplier
        q_bax = q_bax * cfg.dropout_q_bias_multiplier
        q_bay = q_bay * cfg.dropout_q_bias_multiplier

    Q_proc = jnp.diag(
        jnp.array([q_px, q_py, q_vx, q_vy, q_th, q_bg, q_bax, q_bay], dtype=dtype)
    )

    # IMU input noise mapped into state via G
    std_w = cfg.imu_gyro_noise_density * np.sqrt(dt)
    std_f = cfg.imu_accel_noise_density * np.sqrt(dt)
    Qu = jnp.diag(jnp.array([std_w**2, std_f**2, std_f**2], dtype=dtype))

    if cfg.reduce_imu_noise_during_blackout and (not has_vision):
        Qu = Qu * cfg.blackout_imu_noise_scale

    c, s = jnp.cos(theta), jnp.sin(theta)
    G = jnp.zeros((8, 3), dtype=dtype)
    # ax, ay → vx, vy
    G = G.at[2, 1].set(dt * c).at[2, 2].set(-dt * s)
    G = G.at[3, 1].set(dt * s).at[3, 2].set(dt * c)
    # gz → heading
    G = G.at[4, 0].set(dt)

    Q = Q_proc + G @ Qu @ G.T

    if cfg.freeze_bias_during_blackout and (not has_vision):
        for idx in (5, 6, 7):
            Q = Q.at[idx, :].set(0.0)
            Q = Q.at[:, idx].set(0.0)

    return 0.5 * (Q + Q.T)


@pytest.mark.parametrize(
    "duration_s, vx, vy, dropout_prob",
    [
        (2.0, 0.2, 0.1, 0.0),  # no dropout
        (2.0, 0.2, 0.1, 0.3),  # with dropout
    ],
)
def test_ekf_parity_before_after_q_refactor(
    monkeypatch, duration_s, vx, vy, dropout_prob
):
    # Simulate a short constant-velocity scenario
    sim_cfg = SimpleSimConfig(
        duration_s=duration_s,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=dropout_prob,
    )
    sim = simulate_constant_velocity(
        sim_cfg,
        initial_position=np.array([0.0, 0.0]),
        velocity=np.array([vx, vy]),
        seed=123,
    )

    ekf_cfg = EKFConfig(
        use_mahalanobis_gating=True,
        adaptive_q_during_dropout=True,
        reduce_imu_noise_during_blackout=True,
        freeze_bias_during_blackout=True,
    )

    # Run NEW path (assemble_Q)
    res_new = extended_kalman_filter(
        ekf_config=ekf_cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Monkeypatch assemble_Q to use reference_Q and run OLD-equivalent path
    import trodestrack.models.process_noise as pn

    def assemble_Q_ref(config, theta, dt, n, *, has_vision, dtype=jnp.float32):
        assert n == 8
        return reference_Q(config, theta, dt, has_vision).astype(dtype)

    monkeypatch.setattr(pn, "assemble_Q", assemble_Q_ref)

    res_old = extended_kalman_filter(
        ekf_config=ekf_cfg,
        t_imu=sim["t_imu"],
        U_imu=sim["U_imu"],
        t_cam=sim["t_cam_exp"],
        Z_cam_led1=sim["Z_cam_led1"],
        Z_cam_led2=sim["Z_cam_led2"],
        mask_cam=sim["mask_cam"],
    )

    # Compare filtered and predicted trajectories
    atol = 1e-7
    rtol = 1e-5

    assert jnp.allclose(
        res_new.filtered_means, res_old.filtered_means, rtol=rtol, atol=atol
    )
    assert jnp.allclose(
        res_new.predicted_means, res_old.predicted_means, rtol=rtol, atol=atol
    )
    assert jnp.allclose(
        res_new.filtered_covariances, res_old.filtered_covariances, rtol=rtol, atol=atol
    )
    assert jnp.allclose(
        res_new.predicted_covariances,
        res_old.predicted_covariances,
        rtol=rtol,
        atol=atol,
    )
