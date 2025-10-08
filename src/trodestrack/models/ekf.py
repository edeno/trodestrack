"""Extended Kalman Filter (EKF) for sensor-fused rat tracking.

This module implements a 2D EKF with 8-state model:
    x_k = [x, y, vx, vy, θ, b_gz, b_ax, b_ay]^T

Where:
    - (x, y): Position in cm
    - (vx, vy): Velocity in cm/s
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

    Process noise (Q matrix diagonal):
        process_noise_pos: Position process noise (m²)
        process_noise_vel: Velocity process noise (m/s)²
        process_noise_heading: Heading process noise (rad²)
        process_noise_gyro_bias: Gyro bias random walk (rad/s)²
        process_noise_accel_bias: Accel bias random walk (m/s²)²

    Measurement noise (R matrix diagonal):
        measurement_noise_pos: Camera position noise (m²)
        measurement_noise_heading: LED heading noise (rad²)

    IMU noise densities:
        imu_gyro_noise_density: Gyroscope noise density (rad/s/√Hz)
        imu_accel_noise_density: Accelerometer noise density (m/s²/√Hz)

    Dynamics parameters:
        damping_coeff: Velocity damping coefficient λ (1/s)
        led_distance: Front-back LED spacing (m)
    """

    # Process noise
    process_noise_pos: float = 0.01**2  # (1 cm)²
    process_noise_vel: float = 0.1**2  # (10 cm/s)²
    process_noise_heading: float = 0.01**2  # (0.01 rad)²
    process_noise_gyro_bias: float = 1e-6  # Very slow drift
    process_noise_accel_bias: float = 1e-4  # Slow drift

    # Measurement noise
    measurement_noise_pos: float = 0.005**2  # (0.5 cm)²
    measurement_noise_heading: float = 0.05**2  # (~3 deg)²

    # IMU noise
    imu_gyro_noise_density: float = 0.001  # rad/s/√Hz
    imu_accel_noise_density: float = 0.05  # m/s²/√Hz

    # Dynamics
    damping_coeff: float = 0.5  # 1/s
    led_distance: float = 0.04  # 4 cm


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
        A: Square matrix

    Returns:
        Symmetrized matrix (A + A^T) / 2
    """
    return 0.5 * (A + A.T)


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
                0.01**2,  # x: 1 cm std
                0.01**2,  # y: 1 cm std
                0.1**2,  # vx: 10 cm/s std
                0.1**2,  # vy: 10 cm/s std
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

    # Process noise (Q matrix)
    Q = jnp.diag(
        jnp.array(
            [
                config.process_noise_pos,
                config.process_noise_pos,
                config.process_noise_vel,
                config.process_noise_vel,
                config.process_noise_heading,
                config.process_noise_gyro_bias,
                config.process_noise_accel_bias,
                config.process_noise_accel_bias,
            ]
        )
    )

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
) -> EKFState:
    """EKF measurement update step using camera observations.

    Args:
        state: Predicted state
        z_led1: LED1 observation [x, y] in meters
        z_led2: LED2 observation [x, y] in meters
        mask: Observation validity flag
        config: EKF configuration

    Returns:
        Updated (filtered) state
    """
    m_pred, P_pred = state.mean, state.cov

    # If no valid observation, return prediction unchanged
    def no_update(m, P):
        return EKFState(mean=m, cov=P)

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

        # If no valid observations, return prediction
        def no_leds_update(m_in, P_in):
            return EKFState(mean=m_in, cov=P_in)

        # If at least one LED valid, perform update
        def do_leds_update(m_in, P_in):
            # Measurement function
            def h(x):
                return measurement_function(x, config.led_distance)

            # Jacobian
            H = jacfwd(h)
            H_x = H(m_in)

            # Predicted measurement
            z_pred = h(m_in)

            # Innovation (only for valid measurements)
            innov_full = z_obs_full - z_pred

            # For simplicity, use a fixed-size innovation with masked-out invalid values
            # Set invalid innovations to zero (won't affect update due to zero rows in H)
            innov = jnp.where(obs_mask, innov_full, 0.0)

            # Mask out invalid rows in H
            H_masked = jnp.where(obs_mask[:, None], H_x, 0.0)

            # Measurement noise (larger for invalid measurements to avoid singularity)
            R_diag = jnp.where(
                obs_mask,
                config.measurement_noise_pos,
                1e6,  # Large noise for invalid measurements
            )
            R = jnp.diag(R_diag)

            # Innovation covariance
            S = H_masked @ P_in @ H_masked.T + R

            # Kalman gain
            K = psd_solve(S, H_masked @ P_in).T

            # Update mean
            m_upd = m_in + K @ innov

            # Update covariance (Joseph form for numerical stability)
            I_KH = jnp.eye(8) - K @ H_masked
            P_upd = I_KH @ P_in @ I_KH.T + K @ R @ K.T
            P_upd = symmetrize(P_upd)

            return EKFState(mean=m_upd, cov=P_upd)

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

    def filter_step(carry, t_idx):
        """Single filtering step at camera frame t_idx."""
        state_prev, _ = carry

        # Current camera time
        t_current = t_cam_jax[t_idx]

        # Propagate using all IMU samples between frames
        def propagate_from_prev(state_in):
            """Propagate from previous camera frame to current."""
            t_prev = t_cam_jax[t_idx - 1]

            # Find IMU indices in this interval
            imu_mask = (t_imu_jax > t_prev) & (t_imu_jax <= t_current)
            imu_indices = jnp.where(imu_mask, size=len(t_imu_jax), fill_value=-1)[0]

            # Predict forward using each IMU sample
            def propagate_imu(state, imu_idx):
                """Propagate state with single IMU measurement."""
                # Skip invalid indices
                is_valid = imu_idx >= 0

                def do_propagate(s):
                    # Get IMU sample and timestep
                    u = U_imu_jax[imu_idx]
                    # Compute dt using conditional to avoid negative index
                    dt = lax.cond(
                        imu_idx > 0,
                        lambda: t_imu_jax[imu_idx] - t_imu_jax[imu_idx - 1],
                        lambda: jnp.array(0.005),  # Default 200 Hz
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

        # Measurement update
        state_filt = update_step(
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

        # Update carry
        carry = (state_filt, t_idx)

        return carry, outputs

    # Run filter over all camera frames
    carry_init = (initial_state, 0)
    _, outputs = lax.scan(filter_step, carry_init, jnp.arange(n_cam))

    return EKFResult(
        filtered_means=outputs["filtered_mean"],
        filtered_covariances=outputs["filtered_cov"],
        predicted_means=outputs["predicted_mean"],
        predicted_covariances=outputs["predicted_cov"],
        marginal_loglik=0.0,  # TODO: implement log-likelihood computation
    )
