import jax.numpy as jnp
import numpy as np

from trodestrack.models.ekf import EKFConfig


def reference_Q(
    cfg: EKFConfig, theta: float, dt: float, has_vision: bool
) -> jnp.ndarray:
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


def test_build_input_noise_cov_supports_3d_accel():
    """Test that build_input_noise_cov can generate 4x4 Qu for 3D accel."""
    from trodestrack.models.process_noise import build_input_noise_cov

    cfg = EKFConfig()
    dt = 0.005

    # Request 4D input noise for [ω_z, f_x, f_y, f_z]
    Qu = build_input_noise_cov(cfg, dt, n_accel=3, dtype=jnp.float32)

    # Should be 4x4 (1 gyro + 3 accel)
    assert Qu.shape == (4, 4)

    # Diagonal structure expected
    assert jnp.allclose(Qu, jnp.diag(jnp.diag(Qu)), atol=1e-8)

    # Gyro noise
    sg = (cfg.imu_gyro_noise_density * np.sqrt(dt)) ** 2
    assert jnp.allclose(Qu[0, 0], sg, rtol=1e-6)

    # Accel noise (all 3 axes should be equal)
    sa = (cfg.imu_accel_noise_density * np.sqrt(dt)) ** 2
    assert jnp.allclose(Qu[1, 1], sa, rtol=1e-6)
    assert jnp.allclose(Qu[2, 2], sa, rtol=1e-6)
    assert jnp.allclose(Qu[3, 3], sa, rtol=1e-6)


def test_assemble_Q_for_10d_state_with_3d_accel():
    """Test assemble_Q() with 10D state (2D pos + 3D vel + 3D accel bias)."""
    from trodestrack.models.process_noise import assemble_Q

    cfg = EKFConfig(
        adaptive_q_during_dropout=False,
        reduce_imu_noise_during_blackout=False,
        freeze_bias_during_blackout=False,
    )
    theta = 0.5
    dt = 0.005
    has_vision = True

    # 10D state: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
    Q = assemble_Q(cfg, theta, dt, n=10, has_vision=has_vision, dtype=jnp.float32)

    # Check shape
    assert Q.shape == (10, 10)

    # Check symmetry
    assert jnp.allclose(Q, Q.T, atol=1e-8)

    # Check PSD (all eigenvalues non-negative)
    eigvals = jnp.linalg.eigvalsh(Q)
    assert jnp.all(eigvals >= -1e-8)

    # Verify diagonal entries are positive (process noise should be non-zero)
    diag_entries = jnp.diag(Q)
    assert jnp.all(diag_entries > 0)


def test_assemble_Q_10d_with_blackout_scaling():
    """Test that 10D state respects blackout scaling for all 3 accel bias terms."""
    from trodestrack.models.process_noise import assemble_Q

    cfg = EKFConfig(
        adaptive_q_during_dropout=True,
        dropout_q_pos_multiplier=10.0,
        dropout_q_vel_multiplier=10.0,
        dropout_q_bias_multiplier=0.1,
        reduce_imu_noise_during_blackout=True,
        blackout_imu_noise_scale=0.5,
    )
    theta = 0.0
    dt = 0.01

    Q_vision = assemble_Q(cfg, theta, dt, n=10, has_vision=True, dtype=jnp.float32)
    Q_blackout = assemble_Q(cfg, theta, dt, n=10, has_vision=False, dtype=jnp.float32)

    # Position noise should increase during blackout (indices 0, 1)
    assert Q_blackout[0, 0] > Q_vision[0, 0] * 5  # ~10x multiplier
    assert Q_blackout[1, 1] > Q_vision[1, 1] * 5

    # Velocity noise should increase during blackout (indices 2, 3, 4)
    assert Q_blackout[2, 2] > Q_vision[2, 2] * 5
    assert Q_blackout[3, 3] > Q_vision[3, 3] * 5
    assert Q_blackout[4, 4] > Q_vision[4, 4] * 5  # vz

    # Bias noise should decrease during blackout (indices 6, 7, 8, 9)
    # Note: bias indices are [6: b_gz, 7-9: b_ax, b_ay, b_az]
    assert Q_blackout[6, 6] < Q_vision[6, 6]  # b_gz
    assert Q_blackout[7, 7] < Q_vision[7, 7]  # b_ax
    assert Q_blackout[8, 8] < Q_vision[8, 8]  # b_ay
    assert Q_blackout[9, 9] < Q_vision[9, 9]  # b_az (NEW!)


def test_assemble_Q_10d_freezes_all_biases_during_blackout():
    """Test that freeze_bias_during_blackout zeros all 4 bias terms in 10D state."""
    from trodestrack.models.process_noise import assemble_Q

    cfg = EKFConfig(
        freeze_bias_during_blackout=True,
        adaptive_q_during_dropout=False,
        reduce_imu_noise_during_blackout=False,
    )
    Q = assemble_Q(cfg, theta=0.0, dt=0.01, n=10, has_vision=False, dtype=jnp.float32)

    # Bias indices in 10D: [6: b_gz, 7: b_ax, 8: b_ay, 9: b_az]
    # All bias rows/cols should be zero during blackout
    for idx in (6, 7, 8, 9):
        assert jnp.allclose(Q[idx, :], 0.0, atol=1e-10)
        assert jnp.allclose(Q[:, idx], 0.0, atol=1e-10)
