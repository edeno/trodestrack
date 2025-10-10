"""Unscented Kalman Filter (UKF) for sensor-fused rat tracking.

This module implements a 2D UKF with 8-state model:
    x_k = [x, y, vx, vy, θ, b_gz, b_ax, b_ay]^T

Where:
    - (x, y): Position in meters
    - (vx, vy): Velocity in m/s
    - θ: Heading angle in radians
    - b_gz: Gyroscope z-axis bias in rad/s
    - b_ax, b_ay: Accelerometer x, y biases in m/s²

The UKF uses sigma-point transforms to handle nonlinearity without Jacobians.
It propagates (2n+1) = 17 sigma points through nonlinear dynamics and measurements.

Key advantages over EKF:
    - No Jacobian computation required
    - Better handling of strong nonlinearities
    - 2nd-order accuracy for Gaussian distributions

Key differences from EKF:
    - Uses unscented transform instead of linearization
    - Propagates sigma points through actual nonlinear functions
    - Slightly higher computational cost but often more accurate

References:
    - PRD.md Section 6: Mathematical Model
    - Dynamax inference_ukf.py
    - Särkkä (2013) "Bayesian Filtering and Smoothing", Algorithm 5.14
    - Julier & Uhlmann (1997) "New extension of the Kalman filter"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax, vmap

from trodestrack.models.utils import build_G_matrix

from trodestrack.models.ekf import (
    dynamics_function,
    initialize_state,
    measurement_function,
    psd_solve,
    symmetrize,
)

# =============================================================================
# Configuration & State
# =============================================================================


@dataclass
class UKFConfig:
    """Unscented Kalman Filter configuration.

    Inherits process noise and measurement noise settings from EKF.
    Adds UKF-specific hyperparameters for sigma-point generation.

    Process noise RATES (Q matrix diagonal, variance per unit time):
        Same as EKF - see ekf.py EKFConfig for details.

    Measurement noise (R matrix diagonal):
        measurement_noise_pos: Camera position noise (m²)
        measurement_noise_heading: LED heading noise (rad²)

    IMU noise densities:
        imu_gyro_noise_density: Gyroscope noise density (rad/s/√Hz)
        imu_accel_noise_density: Accelerometer noise density (m/s²/√Hz)

    Dynamics parameters:
        damping_coeff: Velocity damping coefficient λ (1/s)
        led_distance: Front-back LED spacing (m)

    UKF hyperparameters:
        alpha: Spread of sigma points (typically 1e-3 to 1)
            - Smaller values keep points closer to mean
            - Larger values spread points further out
            - Typical: 1e-3 (conservative) or sqrt(3) (standard)

        beta: Prior knowledge about distribution (optimal=2 for Gaussian)
            - Incorporates higher-order moments
            - beta=2 is optimal for Gaussian priors

        kappa: Secondary scaling parameter
            - Typically 0 or 3-n (where n=state dimension)
            - kappa=0 common choice for moderate dimensions
            - kappa=3-n ensures positive semi-definite covariance

    Lambda computation:
        lambda = alpha² * (n + kappa) - n
        This determines sigma-point spread: sqrt((n + lambda) * P)
    """

    # Process noise RATES (variance/second, will be scaled by dt in filter)
    process_noise_pos: float = 0.02  # m²/s → 1cm std @ 200Hz
    process_noise_vel: float = 2.0  # (m/s)²/s → 10cm/s std @ 200Hz
    process_noise_heading: float = 0.02  # rad²/s → 0.01rad std @ 200Hz
    # Bias process noise
    process_noise_gyro_bias: float = 2e-6  # (rad/s)²/s → slow drift
    process_noise_accel_bias: float = 2e-4  # (m/s²)²/s → slow drift

    # Measurement noise
    measurement_noise_pos: float = 0.005**2  # (0.5 cm)²
    measurement_noise_heading: float = 0.05**2  # (~3 deg)²

    # IMU noise densities
    imu_gyro_noise_density: float = 0.0001  # rad/s/√Hz
    imu_accel_noise_density: float = 0.005  # m/s²/√Hz

    # Dynamics
    damping_coeff: float = 0.5  # 1/s
    led_distance: float | None = 0.04  # 4 cm (None = auto-detect from data)

    # Heading pseudo-measurement from LED pair (feature parity with EKF)
    use_heading_measurement: bool = False  # Enable heading observation from LED vector
    led_distance_tolerance: float = 0.3  # ±30% tolerance for LED spacing gating
    adaptive_heading_noise: bool = True  # Scale R_heading by baseline geometry

    # UKF hyperparameters (defaults from dynamax/sbitzer UKF-exposed)
    alpha: float = 1.732  # sqrt(3), Sigma-point spread
    beta: float = 2.0  # Prior knowledge (2 = Gaussian optimal)
    kappa: float = 1.0  # Secondary scaling


class UKFState(NamedTuple):
    """UKF state representation.

    Identical to EKF state structure for compatibility.

    Attributes:
        mean: State mean [x, y, vx, vy, θ, b_gz, b_ax, b_ay] (8,)
        cov: State covariance (8, 8)
    """

    mean: jnp.ndarray  # (8,)
    cov: jnp.ndarray  # (8, 8)


class UKFResult(NamedTuple):
    """UKF filtering result.

    Attributes:
        filtered_means: Filtered state means at camera times (N_cam, 8)
        filtered_covariances: Filtered covariances at camera times (N_cam, 8, 8)
        predicted_means: Predicted state means at camera times (N_cam, 8)
        predicted_covariances: Predicted covariances at camera times (N_cam, 8, 8)
        marginal_loglik: Marginal log-likelihood of observations
        estimated_led_distance: Auto-detected LED spacing (m), None if explicit
    """

    filtered_means: jnp.ndarray  # (N_cam, 8)
    filtered_covariances: jnp.ndarray  # (N_cam, 8, 8)
    predicted_means: jnp.ndarray  # (N_cam, 8)
    predicted_covariances: jnp.ndarray  # (N_cam, 8, 8)
    marginal_loglik: float
    estimated_led_distance: float | None


# =============================================================================
# Sigma-Point Generation
# =============================================================================


def compute_sigma_points(mean: jnp.ndarray, cov: jnp.ndarray, n: int, lamb: float) -> jnp.ndarray:
    """Compute (2n+1) sigma points for unscented transform.

    Generates symmetric sigma points around mean using Cholesky decomposition
    of the covariance matrix scaled by (n + lambda).

    Args:
        mean: State mean (n,)
        cov: State covariance (n, n) - must be positive definite
        n: State dimension
        lamb: Lambda parameter from UKF hyperparameters

    Returns:
        sigma_points: (2n+1, n) array where:
            - sigma_points[0] = mean (central point)
            - sigma_points[1:n+1] = mean + sqrt((n+λ)·P) columns
            - sigma_points[n+1:] = mean - sqrt((n+λ)·P) columns

    Example:
        For 8-D state, generates 17 sigma points:
        >>> mean = jnp.zeros(8)
        >>> cov = jnp.eye(8)
        >>> sigmas = compute_sigma_points(mean, cov, 8, 1.0)
        >>> sigmas.shape
        (17, 8)

    Note:
        Uses Cholesky decomposition for numerical stability.
        If cov is not positive definite, Cholesky will raise an error.
    """
    # Compute Cholesky decomposition: P = L @ L.T
    # Add small regularization for numerical stability
    cov_reg = symmetrize(cov) + 1e-9 * jnp.eye(n)
    # Scale by sqrt(n + lambda) for sigma-point spread
    chol = jnp.linalg.cholesky(cov_reg)
    distances = jnp.sqrt(n + lamb) * chol

    # Generate positive-direction sigma points: mean + columns of distances
    sigma_plus = jnp.array([mean + distances[:, i] for i in range(n)])

    # Generate negative-direction sigma points: mean - columns of distances
    sigma_minus = jnp.array([mean - distances[:, i] for i in range(n)])

    # Stack: [mean, sigma_plus (n points), sigma_minus (n points)]
    return jnp.concatenate((jnp.array([mean]), sigma_plus, sigma_minus))


def compute_weights(
    n: int, alpha: float, beta: float, lamb: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute weights for reconstructing mean and covariance from sigma points.

    Weights differ for mean vs covariance reconstruction to incorporate
    higher-order moment information (via beta parameter).

    Args:
        n: State dimension
        alpha: Spread parameter
        beta: Prior knowledge parameter (2 = Gaussian optimal)
        lamb: Lambda = alpha² * (n + kappa) - n

    Returns:
        Tuple of (w_mean, w_cov) where:
            w_mean: (2n+1,) weights for computing mean
            w_cov: (2n+1,) weights for computing covariance

    Formulas (Särkkä 5.77):
        w_mean[0] = lambda / (n + lambda)
        w_mean[i] = 1 / (2 * (n + lambda))  for i > 0

        w_cov[0] = lambda / (n + lambda) + (1 - alpha² + beta)
        w_cov[i] = 1 / (2 * (n + lambda))  for i > 0

    Note:
        Different weights for mean/covariance allow incorporating
        higher-order information without additional sigma points.
    """
    factor = 1.0 / (2.0 * (n + lamb))

    # Mean weights: symmetric except for central point
    w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))

    # Covariance weights: central point incorporates beta correction
    w_cov = jnp.concatenate(
        (
            jnp.array([lamb / (n + lamb) + (1.0 - alpha**2 + beta)]),
            jnp.ones(2 * n) * factor,
        )
    )

    return w_mean, w_cov


# =============================================================================
# Helper Functions
# =============================================================================


def _outer_product_batch(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute batched outer products for covariance reconstruction.

    Args:
        x: (N, n) array
        y: (N, m) array

    Returns:
        outer_products: (N, n, m) array where result[i] = x[i][:, None] @ y[i][None, :]
    """
    return vmap(lambda a, b: jnp.atleast_2d(a).T @ jnp.atleast_2d(b), 0, 0)(x, y)


def gaussian_log_likelihood_ukf(innovation: jnp.ndarray, covariance: jnp.ndarray) -> jnp.ndarray:
    """Compute Gaussian log-likelihood of innovation with numerical stability.

    Stability features:
        - Adds small jitter to covariance diagonal for near-singular matrices
        - Checks sign from slogdet to detect numerical issues
        - Uses Cholesky decomposition when feasible via psd_solve

    Args:
        innovation: Innovation vector (k,)
        covariance: Innovation covariance (k, k)

    Returns:
        Log-likelihood (scalar)

    Formula:
        log_prob = -0.5 * (k*log(2π) + log(det(S)) + v^T S^{-1} v)

    Notes:
        - If determinant is negative or zero (numerical error), adds jitter
        - Jitter = 1e-8 * trace(S) / k added to diagonal
        - This prevents divergence for near-singular covariances
    """
    k = innovation.shape[0]

    # Add small jitter to diagonal for numerical stability
    # Scale by mean diagonal value to be adaptive
    jitter = 1e-8 * jnp.trace(covariance) / k
    S_stable = covariance + jnp.eye(k) * jitter

    # Log determinant using slogdet (more stable than det)
    sign, logdet = jnp.linalg.slogdet(S_stable)

    # Check for numerical issues (sign should be +1 for PSD matrix)
    # If sign <= 0, increase jitter and recompute
    def add_more_jitter():
        jitter_large = 1e-6 * jnp.trace(covariance) / k
        S_jittered = covariance + jnp.eye(k) * jitter_large
        sign_j, logdet_j = jnp.linalg.slogdet(S_jittered)
        return logdet_j

    # Use original logdet if sign is positive, otherwise use jittered version
    logdet_safe = lax.cond(sign > 0, lambda: logdet, add_more_jitter)

    # Mahalanobis distance: v^T S^{-1} v
    S_inv_v = psd_solve(S_stable, innovation)
    mahal = jnp.dot(innovation, S_inv_v)

    # Gaussian log-likelihood
    log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + logdet_safe + mahal)

    return log_prob


# =============================================================================
# UKF Predict & Update Steps
# =============================================================================


def predict_step(
    state: UKFState,
    u_imu: jnp.ndarray,
    dt_imu: float,
    config: UKFConfig,
) -> UKFState:
    """UKF prediction step using IMU measurement via unscented transform.

    Args:
        state: Current state
        u_imu: IMU measurement [ω_z, f_x, f_y]
        dt_imu: IMU timestep
        config: UKF configuration

    Returns:
        Predicted state with mean and covariance

    Algorithm:
        1. Compute lambda and weights from hyperparameters
        2. Generate sigma points around current mean
        3. Propagate each sigma point through dynamics
        4. Reconstruct predicted mean and covariance from transformed points
        5. Add process noise Q
    """
    m, P = state.mean, state.cov
    n = len(m)  # State dimension (8)

    # Compute UKF parameters
    lamb = config.alpha**2 * (n + config.kappa) - n
    w_mean, w_cov = compute_weights(n, config.alpha, config.beta, lamb)

    # Generate sigma points (2n+1 = 17 points)
    sigmas = compute_sigma_points(m, P, n, lamb)

    # Propagate sigma points through dynamics
    def f(x):
        return dynamics_function(x, u_imu, dt_imu, config.damping_coeff)

    sigmas_prop = vmap(f)(sigmas)  # (17, 8)

    # Reconstruct predicted mean (weighted sum)
    m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)

    # Reconstruct predicted covariance
    # Cov = Σ w_cov[i] * (sigma_prop[i] - m_pred) * (sigma_prop[i] - m_pred)^T
    deviations = sigmas_prop - m_pred
    P_pred = jnp.tensordot(w_cov, _outer_product_batch(deviations, deviations), axes=1)

    # Add process noise Q (time-scaled)
    q_bg = config.process_noise_gyro_bias * dt_imu
    q_bax = config.process_noise_accel_bias * dt_imu
    q_bay = config.process_noise_accel_bias * dt_imu
    q_px = config.process_noise_pos * dt_imu
    q_py = config.process_noise_pos * dt_imu
    q_vx = config.process_noise_vel * dt_imu
    q_vy = config.process_noise_vel * dt_imu
    q_th = config.process_noise_heading * dt_imu

    Q_proc = jnp.diag(jnp.array([q_px, q_py, q_vx, q_vy, q_th, q_bg, q_bax, q_bay]))

    # IMU input noise (same as EKF)
    std_w = config.imu_gyro_noise_density * jnp.sqrt(dt_imu)
    std_f = config.imu_accel_noise_density * jnp.sqrt(dt_imu)
    Q_u = jnp.diag(jnp.array([std_w**2, std_f**2, std_f**2]))

    # Build G matrix for input noise propagation (shared utility with EKF)
    theta = m[4]
    G = build_G_matrix(theta, dt_imu)

    Q = Q_proc + G @ Q_u @ G.T
    P_pred = P_pred + Q

    # Symmetrize for numerical stability
    P_pred = symmetrize(P_pred)

    return UKFState(mean=m_pred, cov=P_pred)


def update_step(
    state: UKFState,
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    mask: bool,
    config: UKFConfig,
    confidence: jnp.ndarray | None = None,
) -> tuple[UKFState, float]:
    """UKF measurement update step using camera observations via unscented transform.

    Args:
        state: Predicted state
        z_led1: LED1 observation [x, y] in meters
        z_led2: LED2 observation [x, y] in meters
        mask: Observation validity flag
        config: UKF configuration
        confidence: Confidence scores [led1_x, led1_y, led2_x, led2_y] (4,)
            Range: [0, 1], where 1.0 = high confidence
            If None, defaults to 1.0 (high confidence)
            Measurement noise is scaled as: R_eff = R_base / clip(conf, min, 1.0)
            NOTE: Current implementation uses diagonal R approximation (see issue)

    Returns:
        Tuple of (updated_state, log_likelihood)

    Algorithm:
        1. Generate sigma points around predicted mean
        2. Transform sigma points through measurement function
        3. Reconstruct predicted observation mean and covariance
        4. Compute cross-covariance between state and observations
        5. Kalman update using cross-covariance
    """
    m_pred, P_pred = state.mean, state.cov

    # Process confidence scores (default to high confidence if not provided)
    if confidence is None:
        conf = jnp.ones(4)
    else:
        # Clip confidence to [1e-2, 1.0] to prevent numerical issues
        conf = jnp.clip(confidence, 1e-2, 1.0)

    # If no valid observation, return prediction unchanged
    def no_update(m, P):
        return UKFState(mean=m, cov=P), 0.0

    # If valid observation, perform update
    def do_update(m, P):
        # Check which LEDs are valid
        led1_valid = jnp.isfinite(z_led1[0])
        led2_valid = jnp.isfinite(z_led2[0])

        # Build observation vector
        z_obs_full = jnp.concatenate([z_led1, z_led2])
        obs_mask = jnp.array([led1_valid, led1_valid, led2_valid, led2_valid])

        # If no valid LEDs, skip update
        def no_leds_update(m_in, P_in):
            return UKFState(mean=m_in, cov=P_in), 0.0

        # If at least one LED valid, perform unscented update
        def do_leds_update(m_in, P_in):
            n = len(m_in)

            # Compute UKF parameters
            lamb = config.alpha**2 * (n + config.kappa) - n
            w_mean, w_cov = compute_weights(n, config.alpha, config.beta, lamb)

            # Generate sigma points
            sigmas = compute_sigma_points(m_in, P_in, n, lamb)

            # Transform sigma points through measurement function
            def h(x):
                return measurement_function(x, config.led_distance)

            sigmas_meas = vmap(h)(sigmas)  # (17, 4)

            # Reconstruct predicted observation
            z_pred = jnp.tensordot(w_mean, sigmas_meas, axes=1)

            # Compute innovation covariance S
            meas_deviations = sigmas_meas - z_pred
            S = jnp.tensordot(w_cov, _outer_product_batch(meas_deviations, meas_deviations), axes=1)

            # Add measurement noise R (masked for invalid observations)
            # Scale R by confidence: R_eff = R_base / conf (lower conf → larger R)
            R_base = config.measurement_noise_pos
            R_conf_scaled = R_base / conf  # Element-wise scaling
            R_diag = jnp.where(obs_mask, R_conf_scaled, 1e10)
            R = jnp.diag(R_diag)
            S = S + R

            # Compute cross-covariance between state and observations
            state_deviations = sigmas - m_in
            P_cross = jnp.tensordot(
                w_cov, _outer_product_batch(state_deviations, meas_deviations), axes=1
            )

            # Innovation (with masking)
            innov_full = z_obs_full - z_pred
            innov = jnp.where(obs_mask, innov_full, 0.0)

            # Kalman gain
            K = psd_solve(S, P_cross.T).T

            # Update mean
            m_upd = m_in + K @ innov

            # Update covariance using UKF's native form
            # UKF: P⁺ = P - K S K^T (where S includes R implicitly)
            # This differs from EKF's Joseph form but achieves similar stability
            # because the unscented transform captures cross-correlations exactly
            # without linearization, making the K S K^T term naturally stable
            P_upd = P_in - K @ S @ K.T
            P_upd = symmetrize(P_upd)

            # Compute log-likelihood (only for valid dimensions)
            # Use diagonal approximation for masked observations
            log_lik_per_dim = -0.5 * (
                jnp.log(2 * jnp.pi) + jnp.log(jnp.diag(S)) + innov_full**2 / jnp.diag(S)
            )
            log_lik = jnp.where(obs_mask, log_lik_per_dim, 0.0).sum()

            return UKFState(mean=m_upd, cov=P_upd), log_lik

        # Conditional update based on LED availability
        return lax.cond(
            led1_valid | led2_valid,
            do_leds_update,
            no_leds_update,
            m,
            P,
        )

    # Conditional update based on mask
    return lax.cond(mask, do_update, no_update, m_pred, P_pred)


def update_heading(
    state: UKFState,
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    config: UKFConfig,
) -> tuple[UKFState, float]:
    """Apply 1D heading pseudo-measurement update from LED pair (UKF version).

    Sequential update after position update. Uses large-R gating pattern
    for JAX compatibility (no branching), same as EKF.

    Args:
        state: Current state (after position update)
        z_led1: LED1 observation (2,) in meters
        z_led2: LED2 observation (2,) in meters
        config: UKF configuration

    Returns:
        Updated state and heading measurement log-likelihood

    Algorithm:
        1. Compute heading observation: θ_obs = arctan2(dy, dx)
        2. Check validity: both LEDs visible + spacing within tolerance
        3. Gate via large R: R = R_base (valid) or R = 1e6 (invalid)
        4. Apply 1D unscented update
        5. Wrap heading after update

    Note:
        Always performs update (JAX-friendly). Invalid observations are
        gated via R=1e6 → K≈0 → no actual update.
    """
    from trodestrack.models.ekf import wrap_angle

    m, P = state.mean, state.cov

    # Check LED validity
    led1_valid = jnp.isfinite(z_led1).all()
    led2_valid = jnp.isfinite(z_led2).all()
    both_leds = led1_valid & led2_valid

    # Compute heading observation (always compute, gate via R)
    dx = z_led2[0] - z_led1[0]
    dy = z_led2[1] - z_led1[1]
    heading_obs = jnp.arctan2(dy, dx)

    # Check LED spacing validity
    obs_spacing = jnp.sqrt(dx**2 + dy**2)

    # Determine expected spacing (use config value)
    expected_spacing = config.led_distance

    # Spacing ratio and tolerance check (handle NaN safely)
    spacing_ratio = obs_spacing / expected_spacing
    spacing_valid = jnp.isfinite(spacing_ratio) & (
        (spacing_ratio > (1 - config.led_distance_tolerance))
        & (spacing_ratio < (1 + config.led_distance_tolerance))
    )

    # Overall validity: both LEDs + spacing OK + feature enabled
    use_heading = config.use_heading_measurement & both_leds & spacing_valid

    # Base heading measurement noise
    R_base = config.measurement_noise_heading

    # Adaptive noise scaling (if enabled and spacing is valid)
    # Clip obs_spacing to avoid division by zero/NaN
    obs_spacing_safe = jnp.where(
        jnp.isfinite(obs_spacing) & (obs_spacing > 0.001), obs_spacing, expected_spacing
    )
    R_heading_adapted = lax.cond(
        config.adaptive_heading_noise,
        lambda: R_base * (expected_spacing / obs_spacing_safe) ** 2,
        lambda: R_base,
    )

    # Gate via large R (JAX-friendly: no branching)
    # Valid: R ≈ 0.05² → strong update
    # Invalid: R = 1e6 → K ≈ 0 → no update
    R_heading = lax.select(use_heading, R_heading_adapted, 1e6)

    # 1D unscented heading update
    # For 1D measurement, we can use a simplified unscented transform
    n = len(m)
    lamb = config.alpha**2 * (n + config.kappa) - n
    w_mean, w_cov = compute_weights(n, config.alpha, config.beta, lamb)

    # Generate sigma points
    sigmas = compute_sigma_points(m, P, n, lamb)

    # Transform sigma points through 1D heading measurement function
    # h(x) = x[4] (heading component)
    sigmas_heading = sigmas[:, 4]  # (2n+1,)

    # Predicted heading
    h_pred = jnp.dot(w_mean, sigmas_heading)

    # Innovation with angle wrapping (replace NaN with 0 for gated case)
    innov_raw = wrap_angle(heading_obs - h_pred)
    innov = jnp.where(jnp.isfinite(innov_raw), innov_raw, 0.0)

    # Innovation covariance (1D)
    heading_deviations = sigmas_heading - h_pred
    S = jnp.dot(w_cov, heading_deviations**2) + R_heading

    # Cross-covariance between state and heading measurement
    # state_deviations: (2n+1, n), heading_deviations: (2n+1,)
    # P_cross = sum_i w_cov[i] * state_dev[i, :] * heading_dev[i]
    state_deviations = sigmas - m  # (2n+1, n)
    weighted_products = state_deviations * heading_deviations[:, None]  # (2n+1, n)
    P_cross = jnp.dot(w_cov, weighted_products)  # (n,)

    # Kalman gain (nx1)
    K = P_cross / S

    # Update mean
    m_upd = m + K * innov

    # Wrap heading after update
    m_upd = m_upd.at[4].set(wrap_angle(m_upd[4]))

    # Update covariance using UKF's native form (1D measurement)
    # UKF: P⁺ = P - K S K^T where S = σ² + R
    # The unscented transform ensures numerical stability naturally
    P_upd = P - jnp.outer(K, K) * S
    P_upd = symmetrize(P_upd)

    # Log-likelihood
    log_lik = -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S) + innov**2 / S)

    return UKFState(m_upd, P_upd), log_lik


# =============================================================================
# Main UKF Filter
# =============================================================================


def unscented_kalman_filter(
    ukf_config: UKFConfig,
    t_imu: np.ndarray,
    U_imu: np.ndarray,
    t_cam: np.ndarray,
    Z_cam_led1: np.ndarray,
    Z_cam_led2: np.ndarray,
    mask_cam: np.ndarray,
    initial_state: UKFState | None = None,
    conf_cam: np.ndarray | None = None,
) -> UKFResult:
    """Run Unscented Kalman Filter on full trajectory.

    Processes IMU data at high rate and updates with camera observations.
    Uses sigma-point transforms for prediction and measurement updates.

    Algorithm:
        1. Initialize state from camera observations (reuse EKF initialization)
        2. For each camera frame:
            a. Predict using IMU between previous and current frame (unscented)
            b. Update with camera observation (unscented)
        3. Return filtered estimates at camera times

    Args:
        ukf_config: UKF configuration
        t_imu: IMU timestamps (N_imu,)
        U_imu: IMU measurements [ω_z, f_x, f_y] (N_imu, 3)
        t_cam: Camera timestamps (N_cam,)
        Z_cam_led1: LED1 observations (N_cam, 2) in meters
        Z_cam_led2: LED2 observations (N_cam, 2) in meters
        mask_cam: Camera validity mask (N_cam,)
        initial_state: Optional initial state (if None, auto-initialize)
        conf_cam: Camera confidence scores (N_cam, 4) for [led1_x, led1_y, led2_x, led2_y]
            Range: [0, 1], where 1.0 = high confidence
            If None, defaults to 1.0 (high confidence, backward compatible)
            Measurement noise is scaled as: R_eff = R_base / clip(conf, min, 1.0)
            PRD requirement: "DLC confidence → measurement noise scaling" (Section 13)

    Returns:
        UKF filtering result with states at camera times

    Note:
        Reuses EKF's initialize_state for compatibility.
        UKF and EKF should produce similar initial states.
    """
    # Convert to JAX arrays
    t_imu_jax = jnp.array(t_imu)
    U_imu_jax = jnp.array(U_imu)
    t_cam_jax = jnp.array(t_cam)
    Z_cam_led1_jax = jnp.array(Z_cam_led1)
    Z_cam_led2_jax = jnp.array(Z_cam_led2)
    mask_cam_jax = jnp.array(mask_cam)
    conf_cam_jax = None if conf_cam is None else jnp.array(conf_cam)

    # Auto-detect LED spacing if not specified
    # Store estimated value to return in result (immutability: do NOT mutate config)
    from trodestrack.models.ekf import estimate_led_spacing

    estimated_led_distance: float | None = None
    config_for_filter: UKFConfig

    if ukf_config.led_distance is None:
        estimated_led_distance = estimate_led_spacing(Z_cam_led1_jax, Z_cam_led2_jax, mask_cam_jax)
        # Create new config with estimated spacing (do NOT mutate original)
        config_dict = {k: v for k, v in ukf_config.__dict__.items()}
        config_dict["led_distance"] = estimated_led_distance
        config_for_filter = UKFConfig(**config_dict)
    else:
        # Use original config as-is
        config_for_filter = ukf_config

    # Initialize state (reuse EKF initialization)
    if initial_state is None:
        ekf_init = initialize_state(
            Z_cam_led1_jax,
            Z_cam_led2_jax,
            mask_cam_jax,
            dt_cam=float(jnp.mean(jnp.diff(t_cam_jax))),
            led_distance=config_for_filter.led_distance,  # type: ignore[arg-type]
        )
        initial_state = UKFState(mean=ekf_init.mean, cov=ekf_init.cov)

    n_cam = len(t_cam)

    # Precompute IMU indices for each camera interval
    # Compute exact maximum per-frame count once (on CPU/NumPy) for robust padding
    cuts = np.searchsorted(t_imu, t_cam)
    counts = np.diff(np.r_[0, cuts])
    max_imu_per_frame = int(counts.max())

    # Compute mean IMU timestep for fallback
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
                    return predict_step(s, u, dt, config_for_filter)

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

        # Position measurement update (returns state and log-likelihood)
        state_after_pos, log_lik_pos = update_step(
            state_pred,
            Z_cam_led1_jax[t_idx],
            Z_cam_led2_jax[t_idx],
            mask_cam_jax[t_idx],
            config_for_filter,
            None if conf_cam_jax is None else conf_cam_jax[t_idx],
        )

        # Heading measurement update (sequential after position)
        # Only applied if use_heading_measurement=True (gated via large R otherwise)
        state_filt, log_lik_heading = update_heading(
            state_after_pos,
            Z_cam_led1_jax[t_idx],
            Z_cam_led2_jax[t_idx],
            config_for_filter,
        )

        # Total log-likelihood (position + heading)
        log_lik_k = log_lik_pos + log_lik_heading

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

    return UKFResult(
        filtered_means=outputs["filtered_mean"],
        filtered_covariances=outputs["filtered_cov"],
        predicted_means=outputs["predicted_mean"],
        predicted_covariances=outputs["predicted_cov"],
        marginal_loglik=float(log_lik_total),
        estimated_led_distance=estimated_led_distance,
    )
