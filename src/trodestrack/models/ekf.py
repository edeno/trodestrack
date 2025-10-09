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

from trodestrack.models.utils import build_G_matrix


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
    # Bias process noise: very slow drift (hours-scale)
    # Reduced 100x from initial values to allow convergence
    process_noise_gyro_bias: float = 2e-6  # (rad/s)²/s → slow drift
    process_noise_accel_bias: float = 2e-4  # (m/s²)²/s → slow drift

    # Measurement noise
    measurement_noise_pos: float = 0.005**2  # (0.5 cm)²
    measurement_noise_heading: float = 0.05**2  # (~3 deg)²

    # IMU noise densities (reduced 10x for better bias observability)
    # At 400 Hz: gyro_std ≈ 0.002 rad/s, accel_std ≈ 0.1 m/s²
    imu_gyro_noise_density: float = 0.0001  # rad/s/√Hz (was 0.001)
    imu_accel_noise_density: float = 0.005  # m/s²/√Hz (was 0.05)

    # Dynamics
    damping_coeff: float = 0.5  # 1/s
    led_distance: float | None = 0.04  # 4 cm (None = auto-detect from data)

    # Filter
    num_iter: int = 1  # Standard EKF (set >1 for IEKF)

    # Outlier rejection via Mahalanobis distance gating
    use_mahalanobis_gating: bool = False  # Enable χ² gating
    mahalanobis_threshold_prob: float = 0.997  # p-value for χ² threshold (conservative)

    # Heading pseudo-measurement from LED pair
    use_heading_measurement: bool = False  # Enable heading observation from LED vector
    led_distance_tolerance: float = 0.3  # ±30% tolerance for LED spacing gating
    adaptive_heading_noise: bool = True  # Scale R_heading by baseline geometry

    # Blackout-aware process noise (P0 mitigation for camera dropouts)
    freeze_bias_during_blackout: bool = False  # Set bias Q=0 when no vision
    reduce_imu_noise_during_blackout: bool = False  # Reduce input noise when no vision
    blackout_imu_noise_scale: float = 0.5  # Scale factor for IMU noise (0.25-0.5 recommended)


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
        estimated_led_distance: Auto-detected LED spacing (m), None if explicit
    """

    filtered_means: jnp.ndarray  # (N_cam, 8)
    filtered_covariances: jnp.ndarray  # (N_cam, 8, 8)
    predicted_means: jnp.ndarray  # (N_cam, 8)
    predicted_covariances: jnp.ndarray  # (N_cam, 8, 8)
    marginal_loglik: float
    estimated_led_distance: float | None


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


def wrap_angle(theta: jnp.ndarray) -> jnp.ndarray:
    """Wrap angle to (-π, π].

    Args:
        theta: Angle in radians

    Returns:
        Wrapped angle in (-π, π]
    """
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def joseph_update(
    P: jnp.ndarray,
    K: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
) -> jnp.ndarray:
    """Joseph form covariance update for numerical stability.

    The Joseph form ensures covariance remains positive semi-definite (PSD)
    and symmetric even under numerical errors. It is more stable than the
    standard form P⁺ = (I - KH)P, especially for near-singular covariances.

    Joseph form:
        P⁺ = (I - KH)P(I - KH)ᵀ + KRKᵀ

    This form explicitly preserves:
        1. Symmetry: P⁺ = (P⁺)ᵀ
        2. Positive semi-definiteness: P⁺ ⪰ 0
        3. Numerical stability for ill-conditioned problems

    Args:
        P: Prior covariance (n, n)
        K: Kalman gain (n, m)
        H: Measurement Jacobian (m, n)
        R: Measurement noise covariance (m, m)

    Returns:
        P_upd: Updated covariance (n, n), guaranteed PSD and symmetric

    References:
        - Bierman (1977) "Factorization Methods for Discrete Sequential Estimation"
        - Särkkä (2013) "Bayesian Filtering and Smoothing", Section 5.3
        - Bar-Shalom et al. (2001) "Estimation with Applications to Tracking"
          Section 5.2.3

    Notes:
        - Computationally more expensive than standard form (3 matrix products vs 1)
        - Worth the cost for numerical stability, especially with:
          * Near-singular covariances
          * High-gain updates (small R)
          * Long sequences where errors accumulate
    """
    n = P.shape[0]
    identity = jnp.eye(n)

    # I - KH
    I_KH = identity - K @ H

    # Joseph form: (I - KH) P (I - KH)^T + K R K^T
    P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T

    # Enforce symmetry (should be symmetric but numerical errors can accumulate)
    P_upd = symmetrize(P_upd)

    return P_upd


def estimate_led_spacing(
    Z_cam_led1: jnp.ndarray,
    Z_cam_led2: jnp.ndarray,
    mask_cam: jnp.ndarray,
) -> float:
    """Estimate LED spacing from camera observations.

    Computes median distance between LED1 and LED2 across all valid
    dual-LED observations.

    Args:
        Z_cam_led1: LED1 observations (N_cam, 2) in meters
        Z_cam_led2: LED2 observations (N_cam, 2) in meters
        mask_cam: Camera validity mask (N_cam,)

    Returns:
        Median LED spacing in meters

    Note:
        Returns 0.04 m (4 cm) as fallback if no valid dual-LED observations.
    """
    # Find frames where both LEDs are visible
    led1_valid = jnp.isfinite(Z_cam_led1).all(axis=1)
    led2_valid = jnp.isfinite(Z_cam_led2).all(axis=1)
    both_valid = led1_valid & led2_valid & mask_cam

    # Compute distances for valid frames
    distances = jnp.linalg.norm(Z_cam_led2 - Z_cam_led1, axis=1)

    # Median of valid distances
    valid_distances = jnp.where(both_valid, distances, jnp.nan)

    # Use nanmedian, with fallback if all NaN
    median_spacing = jnp.nanmedian(valid_distances)

    # Fallback to 4 cm if no valid observations
    return float(jnp.where(jnp.isnan(median_spacing), 0.04, median_spacing))


def gaussian_log_likelihood(innovation: jnp.ndarray, covariance: jnp.ndarray) -> jnp.ndarray:
    """Compute Gaussian log-likelihood of innovation with numerical stability.

    Computes log p(y | mu, Sigma) where y ~ N(mu, Sigma)
    and innovation = y - mu.

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
    # psd_solve computes S^{-1} @ v, then we dot with v
    S_inv_v = psd_solve(S_stable, innovation)
    mahal = jnp.dot(innovation, S_inv_v)

    # Gaussian log-likelihood
    log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + logdet_safe + mahal)

    return log_prob


def chi2_threshold(dof: int, prob: float) -> jnp.ndarray:
    """Compute χ² threshold for given degrees of freedom and probability.

    Args:
        dof: Degrees of freedom (measurement dimensionality)
        prob: Probability level (e.g., 0.997 for 99.7% confidence)

    Returns:
        threshold: χ² threshold value

    Examples:
        >>> chi2_threshold(2, 0.997)  # 2D measurement, p=0.997
        11.618
        >>> chi2_threshold(4, 0.997)  # 4D measurement, p=0.997
        16.014

    Notes:
        Uses analytical values for common cases (dof=2,4 and prob=0.95,0.99,0.997).
        JAX-compatible using lax.select for branching.
    """
    # Analytical values for common cases
    # Source: scipy.stats.chi2.ppf(prob, dof)

    # For dof=2
    threshold_2_95 = 5.991
    threshold_2_99 = 9.210
    threshold_2_997 = 11.618  # Corrected from scipy

    # For dof=4
    threshold_4_95 = 9.488
    threshold_4_99 = 13.277
    threshold_4_997 = 16.014  # Corrected from scipy

    # Select threshold based on dof (2 or 4) and prob
    # First select based on prob
    is_997 = jnp.abs(prob - 0.997) < 0.01
    is_99 = jnp.abs(prob - 0.99) < 0.01

    # Threshold for dof=2
    threshold_2 = lax.select(
        is_997, threshold_2_997, lax.select(is_99, threshold_2_99, threshold_2_95)
    )

    # Threshold for dof=4
    threshold_4 = lax.select(
        is_997, threshold_4_997, lax.select(is_99, threshold_4_99, threshold_4_95)
    )

    # Select based on dof
    return lax.select(dof == 2, threshold_2, threshold_4)


def compute_nis_and_loglik(
    innov4: jnp.ndarray,
    S4: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute exact NIS and log-likelihood for 2D or 4D measurement.

    Uses Cholesky decomposition for numerical stability and computes
    statistics in the active measurement subspace (2D or 4D).

    Args:
        innov4: Innovation vector (4,) in full measurement space
        S4: Innovation covariance (4, 4)
        both_leds: True if both LEDs valid (4D measurement)
        only_led1: True if only LED1 valid (2D measurement)
        only_led2: True if only LED2 valid (2D measurement)

    Returns:
        Tuple of (nis, log_likelihood):
            - nis: Normalized Innovation Squared (scalar)
            - log_likelihood: Gaussian log-likelihood (scalar)

    Notes:
        - 4D case: Uses full innovation covariance
        - 2D case: Projects to active subspace via selector matrix
        - Both return exact statistics (no diagonal approximation)
        - NIS follows χ²(k) where k=2 or 4 depending on measurement dim
    """
    from jax.scipy.linalg import cho_solve

    # 4D branch: both LEDs valid
    def compute_4d():
        L4 = jnp.linalg.cholesky(S4 + 1e-9 * jnp.eye(4))
        x4 = cho_solve((L4, True), innov4)
        nis = jnp.dot(innov4, x4)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L4)))
        loglik = -0.5 * (logdet + nis + 4 * jnp.log(2 * jnp.pi))
        return nis, loglik

    # 2D branch: single LED valid
    def compute_2d():
        M2 = make_led_selector(only_led1, only_led2)  # (2, 4)
        S2 = M2 @ S4 @ M2.T  # (2, 2)
        innov2 = M2 @ innov4  # (2,)

        L2 = jnp.linalg.cholesky(S2 + 1e-9 * jnp.eye(2))
        x2 = cho_solve((L2, True), innov2)
        nis = jnp.dot(innov2, x2)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L2)))
        loglik = -0.5 * (logdet + nis + 2 * jnp.log(2 * jnp.pi))
        return nis, loglik

    # Select based on LED validity
    return lax.cond(both_leds, compute_4d, compute_2d)


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

    # Heading: from LED vector if both available, else zero (with high uncertainty)
    led_vec = pos_led2 - pos_led1
    heading_from_leds = jnp.arctan2(led_vec[1], led_vec[0])
    # Note: Velocity-based heading from 2 frames is very noisy (~170° error possible)
    # Better to start at zero with high uncertainty and let filter learn from IMU
    heading_init = jnp.where(led1_valid & led2_valid, heading_from_leds, 0.0)

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
    # Use large heading uncertainty to allow filter to learn from dynamics
    # Even with dual LEDs, heading from short baseline (4 cm) is noisy
    heading_std = jnp.where(
        led1_valid & led2_valid,
        jnp.pi / 4,  # ~45° even with dual LEDs (let filter refine from IMU)
        jnp.pi / 2,  # ~90° when no heading info (let IMU determine)
    )

    cov_init = jnp.diag(
        jnp.array(
            [
                0.01**2,  # x: 1 cm (0.01 m) std
                0.01**2,  # y: 1 cm (0.01 m) std
                0.1**2,  # vx: 10 cm/s (0.1 m/s) std
                0.1**2,  # vy: 10 cm/s (0.1 m/s) std
                heading_std**2,  # θ: adaptive based on availability
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
        p_{k+1} = p_k + v_k * dt + 0.5 * a_k * dt² - 0.5 * λ * v_k * dt²
        b_{k+1} = b_k  (biases are random walks with process noise)

    The position update includes a second-order damping correction term
    (-0.5 * λ * v * dt²) for consistency with the velocity damping term.
    This correction is small at high sampling rates (200-400 Hz) but
    improves accuracy at larger dt.

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

    # Update position (with consistent damping correction)
    # Second-order damping term: -0.5 * λ * v * dt²
    pos = jnp.array([px, py])
    pos_next = pos + vel * dt + 0.5 * accel_world * dt**2 - 0.5 * damping * vel * dt**2

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
    has_vision: bool = True,
) -> EKFState:
    """EKF prediction step using IMU measurement.

    Args:
        state: Current state
        u_imu: IMU measurement [ω_z, f_x, f_y]
        dt_imu: IMU timestep
        config: EKF configuration
        has_vision: Whether camera measurements are available (for blackout-aware Q)

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

    # Wrap heading angle to (-π, π] to prevent numerical issues
    m_pred = m_pred.at[4].set(wrap_angle(m_pred[4]))

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

    # Blackout-aware process noise: reduce IMU input noise when no vision
    # (P0 mitigation - reduces t^3/2 white noise growth during dropouts)
    # Use lax.cond instead of Python if to ensure both branches are traced
    imu_noise_scale = lax.cond(
        config.reduce_imu_noise_during_blackout,
        lambda: lax.select(has_vision, 1.0, config.blackout_imu_noise_scale),
        lambda: 1.0,
    )
    Q_u = Q_u * imu_noise_scale

    # G is ∂f/∂u: IMU input noise propagation matrix
    # Maps IMU measurement noise [ω_z, f_x, f_y] into state space
    theta = m[4]
    G = build_G_matrix(theta, dt_imu)

    # Total process noise: kinematic diffusion + IMU input noise
    Q = Q_proc + G @ Q_u @ G.T

    # Blackout-aware process noise: freeze bias Q when no vision
    # (P0 mitigation - prevents unobservable bias from random walking)
    # Use lax.cond to ensure both branches are traced
    bias_scale = lax.cond(
        config.freeze_bias_during_blackout,
        lambda: lax.select(has_vision, 1.0, 0.0),
        lambda: 1.0,
    )
    Q = Q.at[5, 5].set(Q[5, 5] * bias_scale)  # gyro bias
    Q = Q.at[6, 6].set(Q[6, 6] * bias_scale)  # accel_x bias
    Q = Q.at[7, 7].set(Q[7, 7] * bias_scale)  # accel_y bias

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
    confidence: jnp.ndarray | None = None,
) -> tuple[EKFState, float]:
    """EKF measurement update step using camera observations.

    Supports iterated EKF (IEKF) via config.num_iter parameter.
    Supports confidence-scaled measurement noise (e.g., from DLC).

    Args:
        state: Predicted state
        z_led1: LED1 observation [x, y] in meters
        z_led2: LED2 observation [x, y] in meters
        mask: Observation validity flag
        config: EKF configuration
        confidence: Confidence scores [led1_x, led1_y, led2_x, led2_y] (4,)
            Range: [0, 1], where 1.0 = high confidence
            If None, defaults to 1.0 (high confidence)
            Measurement noise is scaled as: R_eff = R_base / clip(conf, min, 1.0)

    Returns:
        Tuple of (updated_state, log_likelihood)

    Notes:
        Confidence scaling follows: R_i = R_base / conf_i (clipped to [min, 1.0])
        - High confidence (→1.0): R ≈ R_base (trust measurement)
        - Low confidence (→0): R → large (distrust measurement)
        - Minimum confidence prevents R → ∞
    """
    m_pred, P_pred = state.mean, state.cov

    # Process confidence scores
    # Default to high confidence if not provided
    if confidence is None:
        conf = jnp.ones(4)
    else:
        # Clip confidence to [1e-2, 1.0] to prevent numerical issues
        # - Upper bound: 1.0 (perfect confidence)
        # - Lower bound: 1e-2 (prevents R → ∞)
        conf = jnp.clip(confidence, 1e-2, 1.0)

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
                """Single IEKF iteration using lifted subspace operator.

                Uses exact 2D/4D mathematics without variance hacks:
                - 4D path: both LEDs valid
                - 2D path: single LED valid, lifted via selector matrix

                All arrays maintain static 4D shapes for JAX compatibility.
                """
                m_iter, P_iter = carry

                # Re-compute Jacobian at current estimate
                H4 = H(m_iter)  # (4, 8)
                z_pred_4 = h(m_iter)  # (4,)

                # Innovation in full 4D space
                # Zero out invalid LED components to avoid NaN propagation
                innov_4_raw = z_obs_full - z_pred_4  # (4,)
                innov_4 = jnp.where(obs_mask, innov_4_raw, 0.0)  # Zero invalid components

                # Confidence-scaled measurement noise
                # R_i = R_base / conf_i for each dimension
                # Higher confidence → smaller R → trust measurement more
                R_base = config.measurement_noise_pos
                R_diag = R_base / conf  # Shape (4,)
                R4 = jnp.diag(R_diag)

                # Innovation covariance (always 4×4)
                S4 = H4 @ P_iter @ H4.T + R4

                # Lifted inverse operator: v = S_eff⁻¹ @ innov_4
                # This automatically handles 2D/4D based on LED validity
                both_leds = led1_valid & led2_valid
                only_led1 = led1_valid & (~led2_valid)
                only_led2 = (~led1_valid) & led2_valid

                v = apply_lifted_inverse(S4, innov_4, both_leds, only_led1, only_led2)

                # Kalman update without forming K explicitly
                # δx = (P H^T) @ v
                PH_t = P_iter @ H4.T  # (8, 4)
                delta_x = PH_t @ v  # (8,)

                # Update mean
                m_upd = m_iter + delta_x

                # Wrap heading angle to (-π, π] after update
                m_upd = m_upd.at[4].set(wrap_angle(m_upd[4]))

                # Joseph form covariance update via alternative formulation
                # P⁺ = P - PH^T S^{-1} HP (equivalent to (I - KH)P(I - KH)^T + KRK^T)
                # This formulation works naturally with the lifted subspace operator
                HP = H4 @ P_iter  # (4, 8)

                def apply_inv_to_col(col_idx):
                    """Apply S_eff⁻¹ to column of HP."""
                    col = HP[:, col_idx]
                    return apply_lifted_inverse(S4, col, both_leds, only_led1, only_led2)

                # Stack inverse-transformed columns
                inv_S_HP = jnp.stack([apply_inv_to_col(i) for i in range(8)], axis=1)  # (4, 8)

                # Complete Joseph form
                PH_t_inv_S_HP = PH_t @ inv_S_HP  # (8, 8)
                P_upd = P_iter - PH_t_inv_S_HP
                P_upd = symmetrize(P_upd)

                return (m_upd, P_upd), (S4, innov_4, both_leds, only_led1, only_led2)

            # Run IEKF iterations
            carry_init = (m_in, P_in)
            (m_final, P_final), (S_all, innov_all, both_all, led1_all, led2_all) = lax.scan(
                iekf_step, carry_init, jnp.arange(config.num_iter)
            )

            # Extract final (last) iteration values
            S_final = S_all[-1]
            innov_final = innov_all[-1]
            both_final = both_all[-1]
            led1_final = led1_all[-1]
            led2_final = led2_all[-1]

            # Compute exact NIS and log-likelihood using lifted subspace operator
            # No diagonal approximation - uses correct dimensionality (2D or 4D)
            nis, log_lik = compute_nis_and_loglik(
                innov_final, S_final, both_final, led1_final, led2_final
            )

            # Mahalanobis gating: reject if NIS exceeds χ² threshold
            # Always compute gating decision (use lax.cond, not Python if)
            def apply_gating():
                """Apply Mahalanobis gating."""
                # Determine measurement dimensionality for threshold
                dof = lax.cond(
                    both_final,
                    lambda: 4,
                    lambda: 2,  # Single LED
                )
                threshold = chi2_threshold(dof, config.mahalanobis_threshold_prob)

                # Gate: accept if NIS < threshold, reject otherwise
                def accept_measurement():
                    return EKFState(mean=m_final, cov=P_final), log_lik

                def reject_measurement():
                    # Return prediction unchanged with zero log-likelihood
                    return EKFState(mean=m_in, cov=P_in), 0.0

                return lax.cond(nis < threshold, accept_measurement, reject_measurement)

            def no_gating():
                """Return update without gating."""
                return EKFState(mean=m_final, cov=P_final), log_lik

            # Use lax.cond to select gating vs no-gating path
            return lax.cond(
                config.use_mahalanobis_gating,
                apply_gating,
                no_gating,
            )

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


def update_heading(
    state: EKFState,
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    config: EKFConfig,
) -> tuple[EKFState, jnp.ndarray]:
    """Apply 1D heading pseudo-measurement update from LED pair.

    Sequential update after position update. Uses large-R gating pattern
    for JAX compatibility (no branching).

    Args:
        state: Current state (after position update)
        z_led1: LED1 observation (2,) in meters
        z_led2: LED2 observation (2,) in meters
        config: EKF configuration

    Returns:
        Updated state and heading measurement log-likelihood

    Algorithm:
        1. Compute heading observation: θ_obs = arctan2(dy, dx)
        2. Check validity: both LEDs visible + spacing within tolerance
        3. Gate via large R: R = R_base (valid) or R = 1e6 (invalid)
        4. Apply 1D Kalman update with Joseph form
        5. Wrap heading after update

    Note:
        Always performs update (JAX-friendly). Invalid observations are
        gated via R=1e6 → K≈0 → no actual update.
    """
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

    # Determine expected spacing (use config value, which may be auto-detected)
    expected_spacing = config.led_distance if config.led_distance is not None else 0.04

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

    # 1D heading update
    # Measurement function: h(x) = x[4] (heading)
    h_pred = m[4]

    # Innovation with angle wrapping (replace NaN with 0 for gated case)
    innov_raw = wrap_angle(heading_obs - h_pred)
    innov = jnp.where(jnp.isfinite(innov_raw), innov_raw, 0.0)

    # Jacobian: H = [0, 0, 0, 0, 1, 0, 0, 0]
    # Shape (1, 8) for proper matrix operations
    H = jnp.zeros((1, 8))
    H = H.at[0, 4].set(1.0)

    # Innovation covariance (scalar, but treat as 1x1 matrix)
    S = H @ P @ H.T + jnp.array([[R_heading]])

    # Kalman gain (8, 1)
    K = psd_solve(S, H @ P).T

    # Mean update
    m_upd = m + (K @ jnp.array([[innov]])).ravel()
    m_upd = m_upd.at[4].set(wrap_angle(m_upd[4]))  # Wrap after update

    # Covariance update using Joseph form
    R_mat = jnp.array([[R_heading]])
    P_upd = joseph_update(P, K, H, R_mat)

    # Log-likelihood (only meaningful if heading was used)
    # For gated observations (R=1e6), this will be near zero
    S_scalar = S[0, 0]  # Extract scalar from 1x1 matrix
    log_lik = -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S_scalar) + innov**2 / S_scalar)

    # Zero out log-likelihood if not used (for cleaner accounting)
    log_lik = lax.select(use_heading, log_lik, 0.0)

    return EKFState(m_upd, P_upd), log_lik


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

    # Auto-detect LED spacing if not specified
    # Store estimated value to return in result (immutability: do NOT mutate config)
    estimated_led_distance: float | None = None
    config_for_filter: EKFConfig

    if ekf_config.led_distance is None:
        estimated_led_distance = estimate_led_spacing(Z_cam_led1_jax, Z_cam_led2_jax, mask_cam_jax)
        # Create new config with estimated spacing (do NOT mutate original)
        config_dict = {k: v for k, v in ekf_config.__dict__.items()}
        config_dict["led_distance"] = estimated_led_distance
        config_for_filter = EKFConfig(**config_dict)
    else:
        # Use original config as-is
        config_for_filter = ekf_config

    # Initialize state
    if initial_state is None:
        initial_state = initialize_state(
            Z_cam_led1_jax,
            Z_cam_led2_jax,
            mask_cam_jax,
            dt_cam=float(jnp.mean(jnp.diff(t_cam_jax))),
            led_distance=config_for_filter.led_distance,  # type: ignore[arg-type]
        )

    n_cam = len(t_cam)

    # Precompute IMU indices for each camera interval
    # For efficient scanning, we create a fixed-size index array with padding
    # Compute exact maximum per-frame count once (on CPU/NumPy) for robust padding
    cuts = np.searchsorted(t_imu, t_cam)
    counts = np.diff(np.r_[0, cuts])
    max_imu_per_frame = int(counts.max())

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

        # Check if we have vision at this timestep (for blackout-aware Q)
        has_vision_t = mask_cam_jax[t_idx]

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
                    return predict_step(s, u, dt, config_for_filter, has_vision_t)

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
        )

        # Heading measurement update (sequential after position)
        # Only applied if use_heading_measurement=True (gated via large R otherwise)
        state_filt, log_lik_heading = update_heading(
            state_after_pos,
            Z_cam_led1_jax[t_idx],
            Z_cam_led2_jax[t_idx],
            config_for_filter,
        )

        # Total log-likelihood for this frame
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

    return EKFResult(
        filtered_means=outputs["filtered_mean"],
        filtered_covariances=outputs["filtered_cov"],
        predicted_means=outputs["predicted_mean"],
        predicted_covariances=outputs["predicted_cov"],
        marginal_loglik=float(log_lik_total),
        estimated_led_distance=estimated_led_distance,
    )
