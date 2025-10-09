"""Extended Kalman Filter (EKF) for sensor-fused rat tracking.

This module implements a 2D EKF with 8-state model:
    x_k = [x, y, vx, vy, θ, b_gz, b_ax, b_ay]^T

Where:
    - (x, y): Position in meters
    - (vx, vy): Velocity in m/s
    - θ: Heading angle in radians
    - b_gz: Gyroscope z-axis bias in rad/s
    - b_ax, b_ay: Accelerometer x, y biases in m/s²

The filter fuses:
    - High-rate IMU measurements (gyro, accel) at ~200 Hz
    - Low-rate camera observations (LED positions) at ~30 Hz

Key features:
    - IMU pre-integration between camera frames
    - Velocity damping to model drag
    - Dual-LED position and heading measurements
    - Mahalanobis gating for outlier rejection (future)
    - RTS smoother for offline processing (future)

References:
    - PRD.md Section 6: Mathematical Model
    - Dynamax inference_ekf.py
    - Särkkä (2013) "Bayesian Filtering and Smoothing"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, lax


# =============================================================================
# Configuration & State
# =============================================================================


@dataclass
class EKFConfig:
    """Extended Kalman Filter configuration.

    Process noise RATES (Q matrix diagonal, variance per unit time):
        These are RATES (variance/second), NOT per-step variances.
        They are multiplied by dt in predict_step to produce step variances.

        For typical IMU dt = 0.005s (200 Hz), the defaults produce:
            - position: 0.02 m²/s × 0.005s = 1e-4 m² per step (1 cm std)
            - velocity: 2.0 (m/s)²/s × 0.005s = 0.01 (m/s)² per step (10 cm/s std)
            - heading: 0.02 rad²/s × 0.005s = 1e-4 rad² per step (0.01 rad std)

        To convert from desired per-step variance q_step at dt_typical:
            q_rate = q_step / dt_typical

        Example: For 1 cm std position at 200 Hz:
            q_step = (0.01 m)² = 1e-4 m²
            dt = 0.005s
            q_rate = 1e-4 / 0.005 = 0.02 m²/s

    Measurement noise (R matrix diagonal):
        measurement_noise_pos: Camera position noise (m²)
        measurement_noise_heading: LED heading noise (rad²)

    IMU noise densities:
        imu_gyro_noise_density: Gyroscope noise density (rad/s/√Hz)
        imu_accel_noise_density: Accelerometer noise density (m/s²/√Hz)

    Dynamics parameters:
        damping_coeff: Velocity damping coefficient λ (1/s)
        led_distance: Front-back LED spacing (m)

    Filter parameters:
        num_iter: Number of IEKF iterations (1=standard EKF, >1=iterated EKF)
    """

    # Process noise RATES (variance/second, will be scaled by dt in filter)
    # Defaults produce reasonable step variances @ 200 Hz IMU
    process_noise_pos: float = 0.02  # m²/s → 1cm std @ 200Hz
    process_noise_vel: float = 2.0  # (m/s)²/s → 10cm/s std @ 200Hz
    process_noise_heading: float = 0.02  # rad²/s → 0.01rad std @ 200Hz
    process_noise_gyro_bias: float = 2e-4  # (rad/s)²/s
    process_noise_accel_bias: float = 0.02  # (m/s²)²/s

    # Measurement noise
    measurement_noise_pos: float = 0.005**2  # (0.5 cm)²
    measurement_noise_heading: float = 0.05**2  # (~3 deg)²

    # IMU noise
    imu_gyro_noise_density: float = 0.001  # rad/s/√Hz
    imu_accel_noise_density: float = 0.05  # m/s²/√Hz

    # Dynamics
    damping_coeff: float = 0.5  # 1/s
    led_distance: float = 0.04  # 4 cm

    # Filter
    num_iter: int = 1  # Standard EKF (set >1 for IEKF)


class EKFState(NamedTuple):
    """EKF state representation.

    Attributes:
        mean: State mean [x, y, vx, vy, θ, b_gz, b_ax, b_ay] (8,)
        cov: State covariance (8, 8)
    """

    mean: jnp.ndarray  # (8,)
    cov: jnp.ndarray  # (8, 8)


class EKFResult(NamedTuple):
    """EKF filtering result.

    Attributes:
        filtered_means: Filtered state means at camera times (N_cam, 8)
        filtered_covariances: Filtered covariances at camera times (N_cam, 8, 8)
        predicted_means: Predicted state means at camera times (N_cam, 8)
        predicted_covariances: Predicted covariances at camera times (N_cam, 8, 8)
        marginal_loglik: Marginal log-likelihood of observations
    """

    filtered_means: jnp.ndarray  # (N_cam, 8)
    filtered_covariances: jnp.ndarray  # (N_cam, 8, 8)
    predicted_means: jnp.ndarray  # (N_cam, 8)
    predicted_covariances: jnp.ndarray  # (N_cam, 8, 8)
    marginal_loglik: float


# =============================================================================
# Utility Functions
# =============================================================================


def symmetrize(A: jnp.ndarray) -> jnp.ndarray:
    """Symmetrize a matrix to maintain numerical stability.

    Args:
        A: Square matrix or batch of matrices (..., n, n)

    Returns:
        Symmetrized matrix (A + A^T) / 2

    Note:
        Uses jnp.swapaxes instead of .T to support batched operations.
    """
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def psd_solve(A: jnp.ndarray, b: jnp.ndarray, diagonal_boost: float = 1e-9) -> jnp.ndarray:
    """Solve linear system Ax = b for positive semi-definite A.

    Uses Cholesky decomposition for numerical stability.

    Args:
        A: PSD matrix (n, n)
        b: Right-hand side (n, m) or (n,)
        diagonal_boost: Small value added to diagonal for numerical stability

    Returns:
        Solution x such that Ax = b
    """
    from jax.scipy.linalg import cho_factor, cho_solve

    A_stable = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A_stable, lower=True)
    x = cho_solve((L, lower), b)
    return x


def wrap_angle(theta: jnp.ndarray) -> jnp.ndarray:
    """Wrap angle to (-π, π].

    Args:
        theta: Angle in radians

    Returns:
        Wrapped angle in (-π, π]
    """
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def gaussian_log_likelihood(innovation: jnp.ndarray, covariance: jnp.ndarray) -> jnp.ndarray:
    """Compute Gaussian log-likelihood of innovation.

    Computes log p(y | mu, Sigma) where y ~ N(mu, Sigma)
    and innovation = y - mu.

    Args:
        innovation: Innovation vector (k,)
        covariance: Innovation covariance (k, k)

    Returns:
        Log-likelihood (scalar)

    Formula:
        log_prob = -0.5 * (k*log(2π) + log(det(S)) + v^T S^{-1} v)
    """
    k = innovation.shape[0]

    # Log determinant using Cholesky (more stable than det)
    sign, logdet = jnp.linalg.slogdet(covariance)

    # Mahalanobis distance: v^T S^{-1} v
    # psd_solve computes S^{-1} @ v, then we dot with v
    S_inv_v = psd_solve(covariance, innovation)
    mahal = jnp.dot(innovation, S_inv_v)

    # Gaussian log-likelihood
    log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + logdet + mahal)

    return log_prob


# =============================================================================
# State Initialization
# =============================================================================


def initialize_state(
    led1_obs: jnp.ndarray,
    led2_obs: jnp.ndarray,
    mask: jnp.ndarray,
    dt_cam: float,
    led_distance: float = 0.04,
) -> EKFState:
    """Initialize EKF state from first few camera observations.

    Estimates initial position, velocity, and heading from LED observations.
    Biases are initialized to zero with high uncertainty.

    Args:
        led1_obs: LED1 observations (N, 2) in meters
        led2_obs: LED2 observations (N, 2) in meters
        mask: Validity mask (N,) - union of LED1 and LED2
        dt_cam: Camera timestep in seconds
        led_distance: LED spacing in meters

    Returns:
        Initial EKF state with mean and covariance

    Example:
        >>> led1 = jnp.array([[1.0, 2.0], [1.1, 2.0]])
        >>> led2 = jnp.array([[1.04, 2.0], [1.14, 2.0]])
        >>> mask = jnp.array([True, True])
        >>> state = initialize_state(led1, led2, mask, dt_cam=1/30.0)
        >>> state.mean.shape
        (8,)
    """
    # Get first valid observation
    valid_indices = jnp.where(mask)[0]
    first_valid = valid_indices[0] if len(valid_indices) > 0 else 0

    # Check which LEDs are valid
    led1_valid = jnp.isfinite(led1_obs[first_valid, 0])
    led2_valid = jnp.isfinite(led2_obs[first_valid, 0])

    # Position: midpoint of LEDs (or single LED if one is missing)
    pos_led1 = led1_obs[first_valid]
    pos_led2 = led2_obs[first_valid]

    # Position: midpoint of LEDs (or single LED if one is missing, or origin if both invalid)
    pos_init = jnp.where(
        led1_valid & led2_valid,
        (pos_led1 + pos_led2) / 2.0,  # Both valid: use midpoint
        jnp.where(
            led1_valid,
            pos_led1,  # Only LED1 valid
            jnp.where(
                led2_valid,
                pos_led2,  # Only LED2 valid
                jnp.array([0.0, 0.0]),  # Neither valid: fallback to origin
            ),
        ),
    )

    # Heading: from LED vector if both available, else zero
    led_vec = pos_led2 - pos_led1
    heading_from_leds = jnp.arctan2(led_vec[1], led_vec[0])
    heading_init = jnp.where(led1_valid & led2_valid, heading_from_leds, 0.0)

    # Velocity: estimate from first few valid frames
    def compute_velocity():
        """Compute initial velocity from first two valid observations."""
        idx1 = valid_indices[0]
        idx2 = valid_indices[1]
        dt = (idx2 - idx1) * dt_cam

        # Get positions, handling missing LEDs
        led1_1, led2_1 = led1_obs[idx1], led2_obs[idx1]
        led1_2, led2_2 = led1_obs[idx2], led2_obs[idx2]

        led1_1_valid = jnp.isfinite(led1_1[0])
        led2_1_valid = jnp.isfinite(led2_1[0])
        led1_2_valid = jnp.isfinite(led1_2[0])
        led2_2_valid = jnp.isfinite(led2_2[0])

        # Position 1
        pos1 = jnp.where(
            led1_1_valid & led2_1_valid,
            (led1_1 + led2_1) / 2.0,
            jnp.where(led1_1_valid, led1_1, led2_1),
        )

        # Position 2
        pos2 = jnp.where(
            led1_2_valid & led2_2_valid,
            (led1_2 + led2_2) / 2.0,
            jnp.where(led1_2_valid, led1_2, led2_2),
        )

        return (pos2 - pos1) / dt

    vel_init = jnp.where(
        len(valid_indices) >= 2,
        compute_velocity(),
        jnp.zeros(2),
    )

    # Initial mean: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    mean_init = jnp.array(
        [
            pos_init[0],
            pos_init[1],
            vel_init[0],
            vel_init[1],
            heading_init,
            0.0,  # Gyro bias
            0.0,  # Accel x bias
            0.0,  # Accel y bias
        ]
    )

    # Initial covariance (diagonal)
    cov_init = jnp.diag(
        jnp.array(
            [
                0.01**2,  # x: 1 cm (0.01 m) std
                0.01**2,  # y: 1 cm (0.01 m) std
                0.1**2,  # vx: 10 cm/s (0.1 m/s) std
                0.1**2,  # vy: 10 cm/s (0.1 m/s) std
                0.1**2,  # θ: ~6 deg std
                0.05**2,  # b_gz: 0.05 rad/s std
                0.1**2,  # b_ax: 0.1 m/s² std
                0.1**2,  # b_ay: 0.1 m/s² std
            ]
        )
    )

    return EKFState(mean=mean_init, cov=cov_init)


# =============================================================================
# Dynamics & Measurement Models
# =============================================================================


def dynamics_function(
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
    damping: float,
) -> jnp.ndarray:
    """Dynamics function: propagate state using IMU measurements.

    State: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    Input: [ω_z, f_x, f_y] (gyro and specific force)

    Dynamics:
        θ_{k+1} = θ_k + (ω_z - b_gz) * dt
        v_{k+1} = v_k + R(θ) * (f - b_a) * dt - λ * v_k * dt
        p_{k+1} = p_k + v_k * dt + 0.5 * a_k * dt²
        b_{k+1} = b_k  (biases are random walks with process noise)

    Args:
        x: State (8,)
        u: IMU input [ω_z, f_x, f_y] (3,)
        dt: Timestep
        damping: Damping coefficient λ

    Returns:
        Next state (8,)
    """
    # Unpack state
    px, py, vx, vy, theta, b_gz, b_ax, b_ay = x

    # Unpack IMU
    omega_z, fx, fy = u

    # Remove biases
    omega_z_unbiased = omega_z - b_gz
    f_unbiased = jnp.array([fx - b_ax, fy - b_ay])

    # Update heading
    theta_next = theta + omega_z_unbiased * dt

    # Rotate specific force to world frame
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[c, -s], [s, c]])
    accel_world = R @ f_unbiased

    # Update velocity (with damping)
    vel = jnp.array([vx, vy])
    vel_next = vel + accel_world * dt - damping * vel * dt

    # Update position
    pos = jnp.array([px, py])
    pos_next = pos + vel * dt + 0.5 * accel_world * dt**2

    # Biases remain constant (process noise added separately in Q)
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


def measurement_function(x: jnp.ndarray, led_distance: float) -> jnp.ndarray:
    """Measurement function: predict LED observations from state.

    Predicts LED1 and LED2 positions based on center position and heading.

    LED1 is behind (−led_distance/2 along heading)
    LED2 is front (+led_distance/2 along heading)

    Args:
        x: State (8,)
        led_distance: LED spacing in meters

    Returns:
        Predicted measurements [led1_x, led1_y, led2_x, led2_y] (4,)
    """
    px, py, vx, vy, theta, *_ = x

    # LED offset vector in body frame
    dx = 0.5 * led_distance * jnp.cos(theta)
    dy = 0.5 * led_distance * jnp.sin(theta)

    # LED positions
    led1_x = px - dx
    led1_y = py - dy
    led2_x = px + dx
    led2_y = py + dy

    return jnp.array([led1_x, led1_y, led2_x, led2_y])


# =============================================================================
# EKF Predict & Update Steps
# =============================================================================


def predict_step(
    state: EKFState,
    u_imu: jnp.ndarray,
    dt_imu: float,
    config: EKFConfig,
) -> EKFState:
    """EKF prediction step using IMU measurement.

    Args:
        state: Current state
        u_imu: IMU measurement [ω_z, f_x, f_y]
        dt_imu: IMU timestep
        config: EKF configuration

    Returns:
        Predicted state
    """
    m, P = state.mean, state.cov

    # Dynamics function with fixed parameters
    def f(x):
        return dynamics_function(x, u_imu, dt_imu, config.damping_coeff)

    # Jacobian
    F = jacfwd(f)
    F_x = F(m)

    # Predict mean
    m_pred = f(m)

    # Time-scaled process noise for random walks and kinematic diffusion
    # Biases: random walk ~ q_b * dt
    q_bg = config.process_noise_gyro_bias * dt_imu
    q_bax = config.process_noise_accel_bias * dt_imu
    q_bay = config.process_noise_accel_bias * dt_imu

    # Kinematics: simple dt scaling (could use dt² for position if needed)
    q_px = config.process_noise_pos * dt_imu
    q_py = config.process_noise_pos * dt_imu
    q_vx = config.process_noise_vel * dt_imu
    q_vy = config.process_noise_vel * dt_imu
    q_th = config.process_noise_heading * dt_imu

    Q_proc = jnp.diag(jnp.array([q_px, q_py, q_vx, q_vy, q_th, q_bg, q_bax, q_bay]))

    # IMU input noise mapped into state via linearization
    # Input noise standard deviations from densities
    std_w = config.imu_gyro_noise_density * jnp.sqrt(dt_imu)
    std_f = config.imu_accel_noise_density * jnp.sqrt(dt_imu)
    Q_u = jnp.diag(jnp.array([std_w**2, std_f**2, std_f**2]))

    # G is ∂f/∂u evaluated at (m, u)
    # For our dynamics:
    #   θ_{k+1} = θ_k + (ω_z - b_gz) * dt  →  ∂θ/∂ω_z = dt
    #   v_{k+1} = v_k + R(θ)(f - b_a) * dt  →  ∂v/∂f = R(θ) * dt
    #   p_{k+1} = p_k + v_k * dt + 0.5 * R(θ)(f - b_a) * dt²  →  ∂p/∂f = R(θ) * 0.5 * dt²
    theta = m[4]
    c, s = jnp.cos(theta), jnp.sin(theta)
    R_2d = jnp.array([[c, -s], [s, c]])

    # Build G matrix: state (8) × input (3)
    # Rows: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    # Cols: [ω_z, f_x, f_y]
    G = jnp.zeros((8, 3))
    G = G.at[4, 0].set(dt_imu)  # θ depends on ω_z
    G = G.at[2:4, 1:3].set(R_2d * dt_imu)  # velocity depends on force
    G = G.at[0:2, 1:3].set(R_2d * (0.5 * dt_imu * dt_imu))  # position depends on force

    # Total process noise: kinematic diffusion + IMU input noise
    Q = Q_proc + G @ Q_u @ G.T

    # Predict covariance
    P_pred = F_x @ P @ F_x.T + Q
    P_pred = symmetrize(P_pred)

    return EKFState(mean=m_pred, cov=P_pred)


def update_step(
    state: EKFState,
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    mask: bool,
    config: EKFConfig,
) -> tuple[EKFState, float]:
    """EKF measurement update step using camera observations.

    Supports iterated EKF (IEKF) via config.num_iter parameter.

    Args:
        state: Predicted state
        z_led1: LED1 observation [x, y] in meters
        z_led2: LED2 observation [x, y] in meters
        mask: Observation validity flag
        config: EKF configuration

    Returns:
        Tuple of (updated_state, log_likelihood)
    """
    m_pred, P_pred = state.mean, state.cov

    # If no valid observation, return prediction unchanged with zero log-likelihood
    def no_update(m, P):
        return EKFState(mean=m, cov=P), 0.0

    # If valid observation, perform update
    def do_update(m, P):
        # Check which LEDs are valid
        led1_valid = jnp.isfinite(z_led1[0])
        led2_valid = jnp.isfinite(z_led2[0])

        # Build observation vector and mask
        # Only include valid measurements
        z_obs_full = jnp.concatenate([z_led1, z_led2])
        obs_mask = jnp.array(
            [
                led1_valid,
                led1_valid,
                led2_valid,
                led2_valid,
            ]
        )

        # If no valid observations, return prediction with zero log-likelihood
        def no_leds_update(m_in, P_in):
            return EKFState(mean=m_in, cov=P_in), 0.0

        # If at least one LED valid, perform update (with optional IEKF)
        def do_leds_update(m_in, P_in):
            # Measurement function
            def h(x):
                return measurement_function(x, config.led_distance)

            # Jacobian
            H = jacfwd(h)

            # IEKF: Iterate re-linearization around posterior
            def iekf_step(carry, _):
                """Single IEKF iteration: re-linearize and update."""
                m_iter, P_iter = carry

                # Re-compute Jacobian at current estimate
                H_x = H(m_iter)
                z_pred = h(m_iter)

                # Innovation (full 4-D)
                innov_full = z_obs_full - z_pred

                # Mask invalid rows: zero them out and give large variance
                innov = jnp.where(obs_mask, innov_full, 0.0)
                H_masked = jnp.where(obs_mask[:, None], H_x, 0.0)
                R_diag = jnp.where(obs_mask, config.measurement_noise_pos, 1e10)
                R = jnp.diag(R_diag)

                # Innovation covariance
                S = H_masked @ P_iter @ H_masked.T + R

                # Kalman gain
                K = psd_solve(S, H_masked @ P_iter).T

                # Update mean
                m_upd = m_iter + K @ innov

                # Update covariance (Joseph form)
                I_KH = jnp.eye(8) - K @ H_masked
                P_upd = I_KH @ P_iter @ I_KH.T + K @ R @ K.T
                P_upd = symmetrize(P_upd)

                return (m_upd, P_upd), (S, innov, obs_mask)

            # Run IEKF iterations
            carry_init = (m_in, P_in)
            (m_final, P_final), (S_all, innov_all, mask_all) = lax.scan(
                iekf_step, carry_init, jnp.arange(config.num_iter)
            )

            # Extract final (last) iteration values
            S_final = S_all[-1]
            innov_final = innov_all[-1]
            mask_final = mask_all[-1]

            # Compute log-likelihood using only valid dimensions
            # NOTE: This uses diagonal approximation (assumes independent dimensions)
            # which is reasonable for partial observations but slightly optimistic
            # for full covariance coupling. For exact likelihood on masked observations,
            # use submatrix extraction, but that's not JAX-compatible without dynamic shapes.
            log_lik_per_dim = -0.5 * (
                jnp.log(2 * jnp.pi)
                + jnp.log(jnp.diag(S_final))
                + innov_final**2 / jnp.diag(S_final)
            )
            log_lik = jnp.where(mask_final, log_lik_per_dim, 0.0).sum()

            return EKFState(mean=m_final, cov=P_final), log_lik

        # Conditional update based on whether we have any valid LEDs
        return lax.cond(
            led1_valid | led2_valid,
            do_leds_update,
            no_leds_update,
            m,
            P,
        )

    # Conditional update based on mask
    return lax.cond(mask, do_update, no_update, m_pred, P_pred)


# =============================================================================
# Main EKF Filter
# =============================================================================


def extended_kalman_filter(
    ekf_config: EKFConfig,
    t_imu: np.ndarray,
    U_imu: np.ndarray,
    t_cam: np.ndarray,
    Z_cam_led1: np.ndarray,
    Z_cam_led2: np.ndarray,
    mask_cam: np.ndarray,
    initial_state: EKFState | None = None,
) -> EKFResult:
    """Run Extended Kalman Filter on full trajectory.

    Processes IMU data at high rate and updates with camera observations.

    Algorithm:
        1. Initialize state from camera observations
        2. For each camera frame:
            a. Predict using IMU between previous and current frame
            b. Update with camera observation (if valid)
        3. Return filtered estimates at camera times

    Args:
        ekf_config: EKF configuration
        t_imu: IMU timestamps (N_imu,)
        U_imu: IMU measurements [ω_z, f_x, f_y] (N_imu, 3)
        t_cam: Camera timestamps (N_cam,)
        Z_cam_led1: LED1 observations (N_cam, 2) in meters
        Z_cam_led2: LED2 observations (N_cam, 2) in meters
        mask_cam: Camera validity mask (N_cam,)
        initial_state: Optional initial state (if None, auto-initialize)

    Returns:
        EKF filtering result with states at camera times
    """
    # Convert to JAX arrays
    t_imu_jax = jnp.array(t_imu)
    U_imu_jax = jnp.array(U_imu)
    t_cam_jax = jnp.array(t_cam)
    Z_cam_led1_jax = jnp.array(Z_cam_led1)
    Z_cam_led2_jax = jnp.array(Z_cam_led2)
    mask_cam_jax = jnp.array(mask_cam)

    # Initialize state
    if initial_state is None:
        initial_state = initialize_state(
            Z_cam_led1_jax,
            Z_cam_led2_jax,
            mask_cam_jax,
            dt_cam=float(jnp.mean(jnp.diff(t_cam_jax))),
            led_distance=ekf_config.led_distance,
        )

    n_cam = len(t_cam)

    # Precompute IMU indices for each camera interval
    # For efficient scanning, we create a fixed-size index array with padding
    max_imu_per_frame = int(jnp.ceil((t_imu_jax[-1] - t_imu_jax[0]) / len(t_cam) * 2)) + 10

    # Compute mean IMU timestep for fallback when imu_idx == 0
    dt_imu_mean = float(jnp.mean(jnp.diff(t_imu_jax)))

    def compute_imu_index_arrays():
        """Build padded index arrays for IMU samples between camera frames."""
        all_indices = []
        for i in range(n_cam):
            if i == 0:
                # First frame: no IMU propagation
                indices = jnp.full(max_imu_per_frame, -1, dtype=jnp.int32)
            else:
                t_prev = t_cam_jax[i - 1]
                t_current = t_cam_jax[i]
                # Find IMU samples in (t_prev, t_current]
                mask = (t_imu_jax > t_prev) & (t_imu_jax <= t_current)
                valid_indices = jnp.where(mask, size=max_imu_per_frame, fill_value=-1)[0]
                indices = valid_indices
            all_indices.append(indices)
        return jnp.array(all_indices)

    imu_index_arrays = compute_imu_index_arrays()

    def filter_step(carry, t_idx):
        """Single filtering step at camera frame t_idx."""
        state_prev, log_lik_accum = carry

        # Propagate using IMU samples in this segment
        def propagate_from_prev(state_in):
            """Propagate from previous camera frame to current."""
            # Get IMU indices for this interval
            imu_indices = imu_index_arrays[t_idx]

            # Predict forward using each IMU sample
            def propagate_imu(state, imu_idx):
                """Propagate state with single IMU measurement."""
                # Skip invalid indices
                is_valid = imu_idx >= 0

                def do_propagate(s):
                    # Get IMU sample and timestep
                    u = U_imu_jax[imu_idx]
                    # Compute dt (use mean when at first index)
                    dt = lax.cond(
                        imu_idx > 0,
                        lambda: t_imu_jax[imu_idx] - t_imu_jax[imu_idx - 1],
                        lambda: jnp.array(dt_imu_mean),
                    )
                    return predict_step(s, u, dt, ekf_config)

                def no_propagate(s):
                    return s

                return lax.cond(is_valid, do_propagate, no_propagate, state), None

            state_out, _ = lax.scan(propagate_imu, state_in, imu_indices)
            return state_out

        def no_propagate(state_in):
            """First frame: no IMU propagation."""
            return state_in

        # Use lax.cond to handle first frame
        state_pred = lax.cond(t_idx == 0, no_propagate, propagate_from_prev, state_prev)

        # Measurement update (now returns state and log-likelihood)
        state_filt, log_lik_k = update_step(
            state_pred,
            Z_cam_led1_jax[t_idx],
            Z_cam_led2_jax[t_idx],
            mask_cam_jax[t_idx],
            ekf_config,
        )

        # Store outputs
        outputs = {
            "filtered_mean": state_filt.mean,
            "filtered_cov": state_filt.cov,
            "predicted_mean": state_pred.mean,
            "predicted_cov": state_pred.cov,
        }

        # Update carry with accumulated log-likelihood
        carry = (state_filt, log_lik_accum + log_lik_k)

        return carry, outputs

    # Run filter over all camera frames
    carry_init = (initial_state, 0.0)
    (_, log_lik_total), outputs = lax.scan(filter_step, carry_init, jnp.arange(n_cam))

    return EKFResult(
        filtered_means=outputs["filtered_mean"],
        filtered_covariances=outputs["filtered_cov"],
        predicted_means=outputs["predicted_mean"],
        predicted_covariances=outputs["predicted_cov"],
        marginal_loglik=float(log_lik_total),
    )
