"""Shared configuration, state containers, and helpers for Kalman filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve


@dataclass
class FilterCoreConfig:
    """Parameters common to EKF and UKF implementations."""

    process_noise_pos: float = 0.02
    process_noise_vel: float = 2.0
    process_noise_heading: float = 0.02
    process_noise_gyro_bias: float = 2e-6
    process_noise_accel_bias: float = 2e-4

    measurement_noise_pos: float = 0.005**2
    measurement_noise_heading: float = 0.05**2

    imu_gyro_noise_density: float = 0.0001
    imu_accel_noise_density: float = 0.005

    damping_coeff: float = 0.5
    led_distance: float | None = 0.04

    use_mahalanobis_gating: bool = False  # Disable by default, enable in production
    mahalanobis_threshold_prob: float = 0.997  # Reject ~0.3% of measurements (3σ)

    use_heading_measurement: bool = False
    led_distance_tolerance: float = 0.3
    adaptive_heading_noise: bool = True

    adaptive_q_during_dropout: bool = True
    dropout_q_pos_multiplier: float = 10.0
    dropout_q_vel_multiplier: float = 10.0
    dropout_q_bias_multiplier: float = 0.1
    freeze_bias_during_blackout: bool = False
    reduce_imu_noise_during_blackout: bool = False
    blackout_imu_noise_scale: float = 0.5

    enable_zupt: bool = False
    zupt_velocity_threshold: float = 0.05
    zupt_measurement_noise: float = 0.01**2


class FilterState(NamedTuple):
    """Kalman filter state comprising mean vector and covariance matrix."""

    mean: jnp.ndarray
    cov: jnp.ndarray


def symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    """Enforce numerical symmetry on a covariance matrix."""

    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def psd_solve(matrix: jnp.ndarray, rhs: jnp.ndarray, diagonal_boost: float = 1e-9) -> jnp.ndarray:
    """Solve matrix @ x = rhs for PSD matrices via Cholesky factorisation."""

    stabilized = symmetrize(matrix) + diagonal_boost * jnp.eye(matrix.shape[-1])
    chol, lower = cho_factor(stabilized, lower=True)
    return cho_solve((chol, lower), rhs)


def joseph_update(
    cov_prior: jnp.ndarray,
    gain: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
) -> jnp.ndarray:
    """Joseph-form covariance update that preserves PSD and symmetry."""

    n = cov_prior.shape[0]
    identity = jnp.eye(n)
    I_minus_KH = identity - gain @ H
    return symmetrize(I_minus_KH @ cov_prior @ I_minus_KH.T + gain @ R @ gain.T)


def wrap_angle(theta: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to (-π, π] to avoid discontinuities."""

    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def chi2_threshold(dof: int, prob: float) -> jnp.ndarray:
    """Closed-form χ² thresholds for common degrees of freedom and probabilities.

    Supports dof ∈ {2, 4} (single-LED and dual-LED measurements) and
    prob ∈ {0.95, 0.975, 0.99, 0.997} (common gating thresholds).

    Args:
        dof: Degrees of freedom (2 or 4)
        prob: Probability threshold (0.95, 0.975, 0.99, or 0.997)

    Returns:
        χ² threshold value for the given (dof, prob) pair

    Reference:
        Chi-squared distribution quantiles from scipy.stats.chi2.ppf()
        - dof=2: Mahalanobis distance² for single LED (x, y)
        - dof=4: Mahalanobis distance² for dual LEDs (x1, y1, x2, y2)

    Example:
        >>> chi2_threshold(2, 0.997)  # ~3σ for single LED
        11.618
        >>> chi2_threshold(4, 0.95)   # 95% CI for dual LEDs
        9.488
    """
    # Thresholds for dof=2 (single LED)
    threshold_2 = lax.select(
        jnp.abs(prob - 0.997) < 0.001,
        11.618,  # 99.7% (3σ) - very tight gate
        lax.select(
            jnp.abs(prob - 0.99) < 0.001,
            9.210,  # 99% - tight gate
            lax.select(
                jnp.abs(prob - 0.975) < 0.001,
                7.378,  # 97.5% (2.5σ) - moderate gate
                5.991,  # 95% (2σ) - loose gate (default fallback)
            ),
        ),
    )

    # Thresholds for dof=4 (dual LEDs)
    threshold_4 = lax.select(
        jnp.abs(prob - 0.997) < 0.001,
        16.014,  # 99.7% (3σ) - very tight gate
        lax.select(
            jnp.abs(prob - 0.99) < 0.001,
            13.277,  # 99% - tight gate
            lax.select(
                jnp.abs(prob - 0.975) < 0.001,
                11.143,  # 97.5% (2.5σ) - moderate gate
                9.488,  # 95% (2σ) - loose gate (default fallback)
            ),
        ),
    )

    return lax.select(dof == 2, threshold_2, threshold_4)


def dynamics_function(
    state: jnp.ndarray, imu: jnp.ndarray, dt: float, damping: float
) -> jnp.ndarray:
    """Integrate constant-acceleration dynamics with linear damping."""

    px, py, vx, vy, theta, b_gz, b_ax, b_ay = state
    omega_z, fx, fy = imu

    omega_z_unbiased = omega_z - b_gz
    accel_body = jnp.array([fx - b_ax, fy - b_ay])

    theta_next = theta + omega_z_unbiased * dt
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    rotation = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    accel_world = rotation @ accel_body

    vel = jnp.array([vx, vy])
    vel_next = vel + accel_world * dt - damping * vel * dt

    pos = jnp.array([px, py])
    pos_next = pos + vel * dt + 0.5 * accel_world * dt**2 - 0.5 * damping * vel * dt**2

    return jnp.array(
        [
            pos_next[0],
            pos_next[1],
            vel_next[0],
            vel_next[1],
            theta_next,
            b_gz,
            b_ax,
            b_ay,
        ]
    )


def measurement_function(state: jnp.ndarray, led_distance: float) -> jnp.ndarray:
    """Project state into dual-LED measurement space."""

    px = state[0]
    py = state[1]
    theta = state[4]
    dx = 0.5 * led_distance * jnp.cos(theta)
    dy = 0.5 * led_distance * jnp.sin(theta)
    return jnp.array([px - dx, py - dy, px + dx, py + dy])


def make_led_selector(only_led1: bool, only_led2: bool) -> jnp.ndarray:
    """Create 2×4 selector matrix for single-LED observations.

    Extracts the active 2D subspace from 4D measurement space.
    Measurement layout: [led1_x, led1_y, led2_x, led2_y]

    Args:
        only_led1: True if only LED1 is valid
        only_led2: True if only LED2 is valid

    Returns:
        M: 2×4 selector matrix
            - LED1-only: rows [1,0,0,0] and [0,1,0,0]
            - LED2-only: rows [0,0,1,0] and [0,0,0,1]

    Examples:
        >>> M = make_led_selector(only_led1=True, only_led2=False)
        >>> M.shape
        (2, 4)
        >>> M @ jnp.array([x1, y1, x2, y2])  # Extracts [x1, y1]
    """
    # LED1 selector: picks first 2 dimensions
    M_led1 = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    # LED2 selector: picks last 2 dimensions
    M_led2 = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    # Select based on which LED is valid
    return lax.select(only_led1, M_led1, M_led2)


def apply_lifted_inverse(
    S4: jnp.ndarray,
    w4: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> jnp.ndarray:
    """Apply effective inverse S⁻¹ to 4D vector with lifted subspace operator.

    This computes x = S_eff⁻¹ @ w where S_eff is either:
    - Full 4×4 innovation covariance (both LEDs valid)
    - Lifted 2×2 subspace (only one LED valid)

    The key insight: compute in active subspace, then lift back to 4D.
    This avoids large variance hacks while keeping static shapes for JAX.

    Args:
        S4: Innovation covariance (4, 4)
        w4: Vector to multiply (4,)
        both_leds: True if both LEDs are valid
        only_led1: True if only LED1 is valid
        only_led2: True if only LED2 is valid

    Returns:
        x4: Result of S_eff⁻¹ @ w4 (4,) with static shape

    Algorithm:
        - Both LEDs: x4 = solve(S4, w4)
        - Single LED: x4 = M2ᵀ @ solve(M2 @ S4 @ M2ᵀ, M2 @ w4)
          where M2 is 2×4 selector for active LED

    References:
        - Matrix cookbook: subspace projections
        - Lifted Kalman filtering for partial observations

    Examples:
        >>> S = jnp.eye(4) * 0.01  # Innovation covariance
        >>> innov = jnp.array([0.1, 0.2, nan, nan])  # Only LED1 valid
        >>> x = apply_lifted_inverse(S, innov, False, True, False)
        >>> x[:2]  # Should be ~[10, 20] (scaled by 1/0.01)
        >>> x[2:]  # Should be [0, 0] (lifted from 2D subspace)
    """
    # 4D path: both LEDs valid
    x4_full = psd_solve(S4, w4)

    # 2D path: single LED valid
    M2 = make_led_selector(only_led1, only_led2)  # (2, 4)
    S2 = M2 @ S4 @ M2.T  # (2, 2) - subspace innovation covariance
    w2 = M2 @ w4  # (2,) - project to subspace
    x2 = psd_solve(S2, w2)  # (2,) - solve in subspace
    x4_lifted = M2.T @ x2  # (4,) - lift back to 4D

    # Select based on LED validity (both branches return same shape)
    return lax.select(both_leds, x4_full, x4_lifted)


def initialize_state(
    led1_obs: jnp.ndarray,
    led2_obs: jnp.ndarray,
    mask: jnp.ndarray,
    dt_cam: float | jnp.ndarray,
    led_distance: float = 0.04,
) -> FilterState:
    """Bootstrap filter state from early LED observations.

    Args:
        led1_obs: LED1 observations (N, 2)
        led2_obs: LED2 observations (N, 2)
        mask: Observation validity mask (N,)
        dt_cam: Camera frame interval in seconds (can be float or JAX scalar for JIT)
        led_distance: LED spacing in meters

    Returns:
        Initial filter state with mean and covariance
    """

    valid_indices = jnp.where(mask)[0]
    first_valid = valid_indices[0] if len(valid_indices) > 0 else 0

    led1_valid = jnp.isfinite(led1_obs[first_valid, 0])
    led2_valid = jnp.isfinite(led2_obs[first_valid, 0])

    pos_led1 = led1_obs[first_valid]
    pos_led2 = led2_obs[first_valid]

    pos_init = jnp.where(
        led1_valid & led2_valid,
        (pos_led1 + pos_led2) / 2.0,
        jnp.where(
            led1_valid,
            pos_led1,
            jnp.where(led2_valid, pos_led2, jnp.array([0.0, 0.0])),
        ),
    )

    def compute_velocity() -> jnp.ndarray:
        idx1 = valid_indices[0]
        idx2 = valid_indices[1]
        dt = (idx2 - idx1) * dt_cam

        led1_1, led2_1 = led1_obs[idx1], led2_obs[idx1]
        led1_2, led2_2 = led1_obs[idx2], led2_obs[idx2]

        led1_1_valid = jnp.isfinite(led1_1[0])
        led2_1_valid = jnp.isfinite(led2_1[0])
        led1_2_valid = jnp.isfinite(led1_2[0])
        led2_2_valid = jnp.isfinite(led2_2[0])

        pos1 = jnp.where(
            led1_1_valid & led2_1_valid,
            (led1_1 + led2_1) / 2.0,
            jnp.where(led1_1_valid, led1_1, led2_1),
        )
        pos2 = jnp.where(
            led1_2_valid & led2_2_valid,
            (led1_2 + led2_2) / 2.0,
            jnp.where(led1_2_valid, led1_2, led2_2),
        )
        return (pos2 - pos1) / dt

    vel_init = jnp.where(len(valid_indices) >= 2, compute_velocity(), jnp.zeros(2))

    led_vec = pos_led2 - pos_led1
    heading_from_leds = jnp.arctan2(led_vec[1], led_vec[0])
    heading_init = jnp.where(led1_valid & led2_valid, heading_from_leds, 0.0)

    mean_init = jnp.array(
        [
            pos_init[0],
            pos_init[1],
            vel_init[0],
            vel_init[1],
            heading_init,
            0.0,
            0.0,
            0.0,
        ]
    )

    heading_std = jnp.where(
        led1_valid & led2_valid,
        jnp.pi / 4,
        jnp.pi / 2,
    )

    cov_init = jnp.diag(
        jnp.array(
            [
                0.01**2,
                0.01**2,
                0.1**2,
                0.1**2,
                heading_std**2,
                0.05**2,
                0.1**2,
                0.1**2,
            ]
        )
    )

    return FilterState(mean=mean_init, cov=cov_init)


def update_zupt(
    state: FilterState,
    config: FilterCoreConfig,
) -> tuple[FilterState, jnp.ndarray]:
    """Apply zero-velocity pseudo-measurement when stationary."""

    mean, cov = state
    vx, vy = mean[2], mean[3]
    v_mag = jnp.sqrt(vx**2 + vy**2)

    is_stationary = (v_mag < config.zupt_velocity_threshold) & config.enable_zupt

    R_scalar = lax.select(is_stationary, config.zupt_measurement_noise, 1e6)
    R = jnp.diag(jnp.array([R_scalar, R_scalar]))

    innovation = -mean[2:4]
    H = jnp.zeros((2, 8))
    H = H.at[0, 2].set(1.0)
    H = H.at[1, 3].set(1.0)

    S = H @ cov @ H.T + R
    K = psd_solve(S, H @ cov).T

    mean_updated = mean + K @ innovation
    cov_updated = joseph_update(cov, K, H, R)

    log_det = jnp.linalg.slogdet(S)[1]
    innov_quad = innovation @ psd_solve(S, innovation)
    log_likelihood = -0.5 * (2 * jnp.log(2 * jnp.pi) + log_det + innov_quad)
    log_likelihood = lax.select(is_stationary, log_likelihood, jnp.array(0.0))

    return FilterState(mean=mean_updated, cov=cov_updated), log_likelihood
