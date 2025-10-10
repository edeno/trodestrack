import jax.numpy as jnp
import numpy as np

from trodestrack.models.ekf import EKFConfig


def reference_Q(cfg: EKFConfig, theta: float, dt: float, has_vision: bool) -> jnp.ndarray:
    """Compute process noise Q using the existing EKF-style logic as reference.

    Mirrors the logic in ekf.predict_step for the 8D state:
    - Time-scaled diagonal process diffusion (pos/vel/heading/biases)
    - Optional blackout scaling for pos/vel/bias RW terms
    - IMU input noise mapped via G(theta, dt) with optional blackout scaling
    - Optional bias freeze during blackout (zero-out bias blocks)
    """
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

    Q_proc = jnp.diag(jnp.array([q_px, q_py, q_vx, q_vy, q_th, q_bg, q_bax, q_bay], dtype=dtype))

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

    # Symmetrize for safety
    return 0.5 * (Q + Q.T)


def test_assemble_Q_matches_reference_when_vision_present():
    from trodestrack.models.process_noise import assemble_Q

    cfg = EKFConfig(
        adaptive_q_during_dropout=True,
        reduce_imu_noise_during_blackout=True,
        freeze_bias_during_blackout=True,
    )
    theta = 0.7
    dt = 1.0 / 200.0
    has_vision = True

    Q_ref = reference_Q(cfg, theta, dt, has_vision)
    Q_new = assemble_Q(cfg, theta, dt, n=8, has_vision=has_vision, dtype=jnp.float32)

    assert jnp.allclose(Q_new, Q_ref, rtol=1e-6, atol=1e-8)


def test_assemble_Q_matches_reference_in_blackout_with_bias_freeze():
    from trodestrack.models.process_noise import assemble_Q

    cfg = EKFConfig(
        adaptive_q_during_dropout=True,
        reduce_imu_noise_during_blackout=True,
        freeze_bias_during_blackout=True,
        dropout_q_pos_multiplier=10.0,
        dropout_q_vel_multiplier=10.0,
        dropout_q_bias_multiplier=0.1,
        blackout_imu_noise_scale=0.5,
    )

    theta = -1.1
    dt = 0.005
    has_vision = False

    Q_ref = reference_Q(cfg, theta, dt, has_vision)
    Q_new = assemble_Q(cfg, theta, dt, n=8, has_vision=has_vision, dtype=jnp.float32)

    assert jnp.allclose(Q_new, Q_ref, rtol=1e-6, atol=1e-8)


def test_assemble_Q_zeroes_bias_blocks_when_freeze_enabled():
    from trodestrack.models.process_noise import assemble_Q

    cfg = EKFConfig(
        freeze_bias_during_blackout=True,
        adaptive_q_during_dropout=False,
        reduce_imu_noise_during_blackout=False,
    )
    Q = assemble_Q(cfg, theta=0.0, dt=0.01, n=8, has_vision=False, dtype=jnp.float32)

    # Bias indices [5,6,7] rows/cols should be zero
    for idx in (5, 6, 7):
        assert jnp.allclose(Q[idx, :], 0.0)
        assert jnp.allclose(Q[:, idx], 0.0)
