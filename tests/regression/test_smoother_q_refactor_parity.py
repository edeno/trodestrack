from typing import Any

import jax.numpy as jnp
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.runtime.offline import rts_smoother, sigma_point_smoother
from trodestrack.sim.simple import SimpleSimConfig, simulate_circular


def reference_Q(config: Any, theta: float, dt: float, has_vision: bool) -> jnp.ndarray:
    dtype = jnp.float32
    # Base diag diffusion
    q_px = jnp.asarray(config.process_noise_pos * dt, dtype=dtype)
    q_py = jnp.asarray(config.process_noise_pos * dt, dtype=dtype)
    q_vx = jnp.asarray(config.process_noise_vel * dt, dtype=dtype)
    q_vy = jnp.asarray(config.process_noise_vel * dt, dtype=dtype)
    q_th = jnp.asarray(config.process_noise_heading * dt, dtype=dtype)
    q_bg = jnp.asarray(config.process_noise_gyro_bias * dt, dtype=dtype)
    q_bax = jnp.asarray(config.process_noise_accel_bias * dt, dtype=dtype)
    q_bay = jnp.asarray(config.process_noise_accel_bias * dt, dtype=dtype)
    if getattr(config, "adaptive_q_during_dropout", False):
        one = jnp.asarray(1.0, dtype=dtype)
        pos_mult = jnp.where(has_vision, one, config.dropout_q_pos_multiplier)
        vel_mult = jnp.where(has_vision, one, config.dropout_q_vel_multiplier)
        bias_mult = jnp.where(has_vision, one, config.dropout_q_bias_multiplier)
        q_px = q_px * pos_mult
        q_py = q_py * pos_mult
        q_vx = q_vx * vel_mult
        q_vy = q_vy * vel_mult
        q_bg = q_bg * bias_mult
        q_bax = q_bax * bias_mult
        q_bay = q_bay * bias_mult
    Q_proc = jnp.diag(
        jnp.array([q_px, q_py, q_vx, q_vy, q_th, q_bg, q_bax, q_bay], dtype=dtype)
    )
    # IMU input mapping
    dt_arr = jnp.asarray(dt, dtype=dtype)
    sg = config.imu_gyro_noise_density**2 / dt_arr
    sa = config.imu_accel_noise_density**2 / dt_arr
    Qu = jnp.diag(jnp.array([sg, sa, sa], dtype=dtype))
    if getattr(config, "reduce_imu_noise_during_blackout", False):
        imu_mult = jnp.where(has_vision, 1.0, config.blackout_imu_noise_scale)
        Qu = Qu * jnp.asarray(imu_mult, dtype=dtype)
    c, s = jnp.cos(theta), jnp.sin(theta)
    G = jnp.zeros((8, 3), dtype=dtype)
    G = G.at[2, 1].set(dt * c).at[2, 2].set(-dt * s)
    G = G.at[3, 1].set(dt * s).at[3, 2].set(dt * c)
    G = G.at[4, 0].set(dt)
    Q = Q_proc + G @ Qu @ G.T
    if getattr(config, "freeze_bias_during_blackout", False):
        freeze_factor = jnp.where(
            has_vision, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
        )
        bias_indices = jnp.array([5, 6, 7], dtype=jnp.int32)
        row_mask = jnp.ones((8,), dtype=dtype).at[bias_indices].set(freeze_factor)
        Q = Q * row_mask[:, None] * row_mask[None, :]
    return 0.5 * (Q + Q.T)


@pytest.mark.slow
@pytest.mark.parametrize("dropout_prob", [0.0, 0.4])
def test_rts_smoother_parity(monkeypatch, dropout_prob):
    """Parity check between new assemble_Q smoother and reference Q for EKF RTS.

    Runtime (observed locally): ~6.2 s
    """
    # Simulate circular motion to exercise heading and bias dynamics
    simulation_config = SimpleSimConfig(
        duration_s=3.0, fs_imu=200.0, fs_cam=30.0, cam_dropout_prob=dropout_prob
    )
    simulation_data = simulate_circular(simulation_config, seed=7)

    ekf_config = EKFConfig(
        state_mode="2d_full",
        use_mahalanobis_gating=True,
        adaptive_q_during_dropout=True,
        reduce_imu_noise_during_blackout=True,
        freeze_bias_during_blackout=True,
    )

    filter_result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=simulation_data["t_imu"],
        U_imu=simulation_data["U_imu"],
        t_cam=simulation_data["t_cam_exp"],
        Z_cam_led1=simulation_data["Z_cam_led1"],
        Z_cam_led2=simulation_data["Z_cam_led2"],
        mask_cam=simulation_data["mask_cam"],
    )

    # New smoother (assemble_Q)
    new_smoother_result = rts_smoother(
        filter_result=filter_result,
        ekf_config=ekf_config,
        t_imu=simulation_data["t_imu"],
        U_imu=simulation_data["U_imu"],
        t_cam=simulation_data["t_cam_exp"],
        mask_cam=simulation_data["mask_cam"],
    )

    # Old-equivalent: monkeypatch the offline module's bound symbol, not
    # process_noise.assemble_Q. runtime/offline.py imports assemble_Q directly,
    # so patching the source module would leave the traced smoother untouched.
    import trodestrack.runtime.offline as offline_module

    def assemble_Q_ref(
        config,
        theta,
        dt,
        n,
        *,
        has_vision,
        dtype=jnp.float32,
        orientation_quaternion=None,
    ):
        assert n == 8
        return reference_Q(config, theta, dt, has_vision).astype(dtype)

    monkeypatch.setattr(offline_module, "assemble_Q", assemble_Q_ref)
    offline_module._rts_smoother_jit.clear_cache()
    try:
        reference_smoother_result = rts_smoother(
            filter_result=filter_result,
            ekf_config=ekf_config,
            t_imu=simulation_data["t_imu"],
            U_imu=simulation_data["U_imu"],
            t_cam=simulation_data["t_cam_exp"],
            mask_cam=simulation_data["mask_cam"],
        )
    finally:
        offline_module._rts_smoother_jit.clear_cache()

    assert jnp.allclose(
        new_smoother_result.smoothed_means,
        reference_smoother_result.smoothed_means,
        rtol=1e-5,
        atol=1e-7,
    )
    assert jnp.allclose(
        new_smoother_result.smoothed_covariances,
        reference_smoother_result.smoothed_covariances,
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.slow
@pytest.mark.parametrize("dropout_prob", [0.0, 0.4])
def test_sigma_point_smoother_parity(monkeypatch, dropout_prob):
    """Parity check between new assemble_Q smoother and reference Q for UKF sigma-point smoother.

    Runtime (observed locally): ~6.3 s
    """
    simulation_config = SimpleSimConfig(
        duration_s=3.0, fs_imu=200.0, fs_cam=30.0, cam_dropout_prob=dropout_prob
    )
    simulation_data = simulate_circular(simulation_config, seed=11)

    ukf_config = UKFConfig(
        state_mode="2d_full",
        use_mahalanobis_gating=True,
        adaptive_q_during_dropout=True,
        reduce_imu_noise_during_blackout=True,
        freeze_bias_during_blackout=True,
    )

    filter_result = unscented_kalman_filter(
        ukf_config=ukf_config,
        t_imu=simulation_data["t_imu"],
        U_imu=simulation_data["U_imu"],
        t_cam=simulation_data["t_cam_exp"],
        Z_cam_led1=simulation_data["Z_cam_led1"],
        Z_cam_led2=simulation_data["Z_cam_led2"],
        mask_cam=simulation_data["mask_cam"],
    )

    new_smoother_result = sigma_point_smoother(
        filter_result=filter_result,
        ukf_config=ukf_config,
        t_imu=simulation_data["t_imu"],
        U_imu=simulation_data["U_imu"],
        t_cam=simulation_data["t_cam_exp"],
        mask_cam=simulation_data["mask_cam"],
    )

    import trodestrack.runtime.offline as offline_module

    def assemble_Q_ref(
        config,
        theta,
        dt,
        n,
        *,
        has_vision,
        dtype=jnp.float32,
        orientation_quaternion=None,
    ):
        assert n == 8
        return reference_Q(config, theta, dt, has_vision).astype(dtype)

    monkeypatch.setattr(offline_module, "assemble_Q", assemble_Q_ref)
    offline_module._sigma_point_smoother_jit.clear_cache()
    try:
        reference_smoother_result = sigma_point_smoother(
            filter_result=filter_result,
            ukf_config=ukf_config,
            t_imu=simulation_data["t_imu"],
            U_imu=simulation_data["U_imu"],
            t_cam=simulation_data["t_cam_exp"],
            mask_cam=simulation_data["mask_cam"],
        )
    finally:
        offline_module._sigma_point_smoother_jit.clear_cache()

    assert jnp.allclose(
        new_smoother_result.smoothed_means,
        reference_smoother_result.smoothed_means,
        rtol=1e-5,
        atol=5e-7,
    )
    assert jnp.allclose(
        new_smoother_result.smoothed_covariances,
        reference_smoother_result.smoothed_covariances,
        rtol=1e-5,
        atol=1e-7,
    )
