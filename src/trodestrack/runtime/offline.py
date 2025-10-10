"""Offline smoothing for sensor-fused rat tracking.

This module implements RTS (Rauch-Tung-Striebel) smoothing for post-processing:
    - RTS smoother for EKF (extended_kalman_smoother)
    - Sigma-point smoother for UKF (unscented_kalman_smoother)

The smoothers run backwards from the final filtered estimate to refine all
state estimates using future observations. This produces lower-variance
estimates than forward filtering alone.

References:
    - PRD.md Section 12: Algorithms & Implementation Notes
    - Särkkä (2013) "Bayesian Filtering and Smoothing", Algorithm 8.2
    - Dynamax inference_ekf.py, inference_ukf.py
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, lax

from trodestrack.models.ekf import EKFConfig, EKFResult
from trodestrack.models.filter_common import (
    dynamics_function,
    psd_solve,
    symmetrize,
)
from jax import vmap

from trodestrack.models.ukf import UKFConfig, UKFResult
from trodestrack.models.process_noise import assemble_Q

# =============================================================================
# Smoother Result Types
# =============================================================================


class SmootherResult(NamedTuple):
    """Smoother result (both EKF and UKF).

    Attributes:
        smoothed_means: Smoothed state means at camera times (N_cam, n)
        smoothed_covariances: Smoothed covariances at camera times (N_cam, n, n)
        marginal_loglik: Marginal log-likelihood from filter

    where n is the state dimension (8 for standard 2D, 12 for future 3D, etc.)
    """

    smoothed_means: jnp.ndarray  # (N_cam, n)
    smoothed_covariances: jnp.ndarray  # (N_cam, n, n)
    marginal_loglik: float


# =============================================================================
# Helper Functions
# =============================================================================


def build_Q_rate(config: EKFConfig | UKFConfig, n: int) -> jnp.ndarray:
    """Build process noise rate matrix Q_rate for arbitrary state dimension.

    For standard 8D state: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    For future 3D state: [x, y, z, vx, vy, vz, roll, pitch, yaw, biases...]

    Args:
        config: EKF or UKF configuration containing process noise parameters
        n: State dimension

    Returns:
        Diagonal Q_rate matrix (n, n)

    Note:
        This function maps config noise parameters to state dimensions.
        Standard mapping for 8D state:
            - [0:2] = position (x, y) → process_noise_pos
            - [2:4] = velocity (vx, vy) → process_noise_vel
            - [4] = heading (θ) → process_noise_heading
            - [5] = gyro bias (b_gz) → process_noise_gyro_bias
            - [6:8] = accel bias (b_ax, b_ay) → process_noise_accel_bias

        For non-standard dimensions, falls back to uniform noise = process_noise_pos.
    """
    # For standard 8D state, use the known mapping
    if n == 8:
        return jnp.diag(
            jnp.array(
                [
                    config.process_noise_pos,  # x
                    config.process_noise_pos,  # y
                    config.process_noise_vel,  # vx
                    config.process_noise_vel,  # vy
                    config.process_noise_heading,  # θ
                    config.process_noise_gyro_bias,  # b_gz
                    config.process_noise_accel_bias,  # b_ax
                    config.process_noise_accel_bias,  # b_ay
                ]
            )
        )
    else:
        # For non-standard dimensions, use uniform noise
        # TODO(P1): For future 3D/custom states, accept explicit noise vector
        # or implement convention (first n/2 = position, second n/2 = velocity, etc.)
        # Current fallback is conservative but may be too tight for velocity/angular states.
        # Using process_noise_pos (0.02 m²/s) for ALL dimensions makes velocity states
        # ~100x less noisy than they should be (process_noise_vel = 2.0 (m/s)²/s).
        return jnp.diag(jnp.full(n, config.process_noise_pos))


# =============================================================================
# RTS Smoother for EKF
# =============================================================================


def rts_smoother(
    filter_result: EKFResult,
    ekf_config: EKFConfig,
    t_imu: np.ndarray,
    U_imu: np.ndarray,
    t_cam: np.ndarray,
    num_iter: int = 1,
    mask_cam: np.ndarray | None = None,
) -> SmootherResult:
    """Run RTS (Rauch-Tung-Striebel) smoother on EKF filter output.

    The RTS smoother performs a backward pass over the filtered estimates,
    incorporating information from future measurements to refine each state
    estimate. This produces smoothed estimates with lower variance than
    the forward filter alone.

    Supports iterative smoothing (IEKS): relinearize around previous smoothed
    trajectory for improved handling of nonlinearities.

    Algorithm (Särkkä 2013, Algorithm 8.2 + iterative extension):
        For iteration i = 1...num_iter:
            For k = N-1, ..., 1:
                1. Predict forward from k to k+1 (around x_smooth^(i-1)): m_pred, P_pred
                2. Compute smoother gain: G_k = P_k @ F_k^T @ P_pred^{-1}
                3. Smooth state: m_smooth[k] = m_filt[k] + G_k @ (m_smooth[k+1] - m_pred)
                4. Smooth cov: P_smooth[k] = P_filt[k] + G_k @ (P_smooth[k+1] - P_pred) @ G_k^T

    Args:
        filter_result: Output from extended_kalman_filter()
        ekf_config: EKF configuration
        t_imu: IMU timestamps (N_imu,) - needed to compute prediction
        U_imu: IMU measurements [ω_z, f_x, f_y] (N_imu, 3)
        t_cam: Camera timestamps (N_cam,)
        num_iter: Number of IEKS iterations (default 1 = standard RTS)
        mask_cam: Camera mask (N_cam,) - if provided, applies blackout-aware noise scaling

    Returns:
        SmootherResult with smoothed means and covariances

    Note:
        The smoother runs backward in time, starting from the last filtered
        estimate (which equals the last smoothed estimate by definition).
        State dimension is derived from filter_result.filtered_means.shape[1].

        Blackout-aware Q/R scaling (when mask_cam is provided):
        - During vision blackouts, reduces accel bias RW noise and IMU input noise
        - Helps tighten how hard post-gap vision "pulls" backward through gaps
    """
    # Convert to JAX arrays
    t_imu_jax = jnp.array(t_imu)
    U_imu_jax = jnp.array(U_imu)
    t_cam_jax = jnp.array(t_cam)

    # Extract filter outputs and derive state dimension from data
    filtered_means = filter_result.filtered_means  # (N_cam, n)
    filtered_covs = filter_result.filtered_covariances  # (N_cam, n, n)
    n_cam = len(t_cam)
    n = filtered_means.shape[1]  # Derive state dimension from data

    # Compute mean IMU dt for fallback
    dt_imu_mean = float(jnp.mean(jnp.diff(t_imu_jax)))

    # Precompute IMU indices for each camera interval (same as filter)
    # Compute exact maximum per-frame count once (on CPU/NumPy) for robust padding
    cuts = np.searchsorted(t_imu, t_cam)
    counts = np.diff(np.r_[0, cuts])
    max_imu_per_frame = int(counts.max())

    def compute_imu_index_arrays():
        """Build padded index arrays for IMU samples between camera frames.

        IMPORTANT: This is a HOST-SIDE precomputation, NOT JIT-traced.
        The Python loop runs on CPU before smoother execution, producing static
        index arrays baked into the JIT-compiled smoother.

        Returns:
            jnp.ndarray: (n_cam, max_imu_per_frame) array of IMU indices
                where -1 indicates padding (invalid index)
        """
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

    # Convert mask_cam to JAX if provided
    mask_cam_jax = jnp.array(mask_cam) if mask_cam is not None else None

    # Compute Jacobian of dynamics
    def f(x, u, dt):
        return dynamics_function(x, u, dt, ekf_config.damping_coeff)

    F_jac = jacfwd(f, argnums=0)

    def predict_between_frames(
        t_idx: int,
        x_k: jnp.ndarray,
        P_k: jnp.ndarray,
        x_k_lin: jnp.ndarray,
    ):
        """Predict from frame t_idx to t_idx+1 using IMU.

        This replicates the filter's prediction logic to get m_pred, P_pred, and G.

        Args:
            t_idx: Time index
            x_k: State at time k (for RTS update)
            P_k: Covariance at time k
            x_k_lin: Linearization point trajectory at time k (for IEKS)

        Returns: (m_pred, P_pred, G) where G is the smoother gain matrix.
        """
        # Get IMU indices for interval [t_idx, t_idx+1)
        imu_indices = imu_index_arrays[t_idx + 1]

        # Blackout-aware noise scaling
        # Check if either frame k or k+1 is in blackout
        in_blackout = (mask_cam_jax is not None) and (
            (~mask_cam_jax[t_idx]) | (~mask_cam_jax[t_idx + 1])
        )

        def propagate_one_imu(carry, imu_idx):
            """Propagate through one IMU sample."""
            x_in, P_in, F_accum, x_lin_in = carry

            # Skip invalid indices
            is_valid = imu_idx >= 0

            def do_propagate(state_cov_F_lin):
                x_s, P_s, F_prev, x_lin_s = state_cov_F_lin
                # Get IMU sample and dt
                u = U_imu_jax[imu_idx]
                dt = lax.cond(
                    imu_idx > 0,
                    lambda: t_imu_jax[imu_idx] - t_imu_jax[imu_idx - 1],
                    lambda: jnp.array(dt_imu_mean),
                )

                # Predict mean (propagate actual state)
                x_pred = f(x_s, u, dt)

                # Compute Jacobian around linearization point (IEKS)
                F_k = F_jac(x_lin_s, u, dt)

                dtype = x_s.dtype

                if n == 8:
                    q_px = jnp.asarray(ekf_config.process_noise_pos * dt, dtype=dtype)
                    q_py = jnp.asarray(ekf_config.process_noise_pos * dt, dtype=dtype)
                    q_vx = jnp.asarray(ekf_config.process_noise_vel * dt, dtype=dtype)
                    q_vy = jnp.asarray(ekf_config.process_noise_vel * dt, dtype=dtype)
                    q_th = jnp.asarray(ekf_config.process_noise_heading * dt, dtype=dtype)
                    q_bg = jnp.asarray(ekf_config.process_noise_gyro_bias * dt, dtype=dtype)
                    q_bax = jnp.asarray(ekf_config.process_noise_accel_bias * dt, dtype=dtype)
                    q_bay = jnp.asarray(ekf_config.process_noise_accel_bias * dt, dtype=dtype)

                    if ekf_config.adaptive_q_during_dropout:
                        pos_scale = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ekf_config.dropout_q_pos_multiplier, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                        vel_scale = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ekf_config.dropout_q_vel_multiplier, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                        bias_scale_adaptive = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ekf_config.dropout_q_bias_multiplier, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                    else:
                        pos_scale = jnp.asarray(1.0, dtype=dtype)
                        vel_scale = jnp.asarray(1.0, dtype=dtype)
                        bias_scale_adaptive = jnp.asarray(1.0, dtype=dtype)

                    q_px = q_px * pos_scale
                    q_py = q_py * pos_scale
                    q_vx = q_vx * vel_scale
                    q_vy = q_vy * vel_scale
                    q_bg = q_bg * bias_scale_adaptive
                    q_bax = q_bax * bias_scale_adaptive
                    q_bay = q_bay * bias_scale_adaptive

                    Q_k = jnp.diag(
                        jnp.array(
                            [q_px, q_py, q_vx, q_vy, q_th, q_bg, q_bax, q_bay],
                            dtype=dtype,
                        )
                    )
                else:
                    Q_rate_base = build_Q_rate(ekf_config, n)
                    q_diag = jnp.diag(Q_rate_base) * dt
                    if ekf_config.adaptive_q_during_dropout:
                        scale = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ekf_config.dropout_q_pos_multiplier, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                        q_diag = q_diag * scale
                    Q_k = jnp.diag(q_diag.astype(dtype))

                # Predict covariance
                if n == 8:
                    # IMU input noise mapping (reuse EKF logic for blackout-aware scaling)
                    std_w = jnp.asarray(
                        ekf_config.imu_gyro_noise_density * jnp.sqrt(dt), dtype=dtype
                    )
                    std_f = jnp.asarray(
                        ekf_config.imu_accel_noise_density * jnp.sqrt(dt), dtype=dtype
                    )
                    Q_u = jnp.diag(jnp.array([std_w**2, std_f**2, std_f**2], dtype=dtype))

                    if ekf_config.reduce_imu_noise_during_blackout:
                        imu_scale = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ekf_config.blackout_imu_noise_scale, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                        Q_u = Q_u * imu_scale

                    theta = x_s[4]
                    Q_total = assemble_Q(
                        ekf_config,
                        theta=theta,
                        dt=dt,
                        n=n,
                        has_vision=~in_blackout,
                        dtype=dtype,
                    )
                else:
                    Q_total = Q_k

                P_pred = F_k @ P_s @ F_k.T + Q_total
                P_pred = symmetrize(P_pred)

                # Accumulate Jacobian: F_total = F_new @ F_prev
                F_new = F_k @ F_prev

                # Propagate linearization trajectory
                x_lin_pred = f(x_lin_s, u, dt)

                return x_pred, P_pred, F_new, x_lin_pred

            def no_propagate(state_cov_F_lin):
                return state_cov_F_lin

            return (
                lax.cond(is_valid, do_propagate, no_propagate, (x_in, P_in, F_accum, x_lin_in)),
                None,
            )

        # Scan through all IMU samples in this interval
        # Initialize with identity Jacobian (dimension n)
        F_init = jnp.eye(n)
        (x_pred, P_pred, F_total, _), _ = lax.scan(
            propagate_one_imu, (x_k, P_k, F_init, x_k_lin), imu_indices
        )

        # Compute smoother gain: G = P_k @ F_total^T @ P_pred^{-1}
        G = psd_solve(P_pred, F_total @ P_k).T

        return x_pred, P_pred, G

    def smoother_step(carry, args):
        """Single backward smoothing step.

        Args:
            carry: (smoothed_mean_next, smoothed_cov_next) at time k+1
            args: (t, filtered_mean_k, filtered_cov_k, lin_mean_k) at time k

        Returns:
            Updated carry and smoothed estimates at time k
        """
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov, lin_mean = args

        # Predict from k to k+1 and get smoother gain
        # Linearize around lin_mean (IEKS) but update using filtered_mean (RTS)
        m_pred, P_pred, G = predict_between_frames(t, filtered_mean, filtered_cov, lin_mean)

        # Smooth mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - P_pred) @ G.T
        smoothed_cov = symmetrize(smoothed_cov)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    def run_one_rts_iteration(lin_means: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run one RTS backward pass.

        Args:
            lin_means: Linearization trajectory (N_cam, n)

        Returns:
            smoothed_means, smoothed_covs: Updated estimates
        """
        # Initial condition: smoothed[N-1] = filtered[N-1]
        _, (smoothed_means_iter, smoothed_covs_iter) = lax.scan(
            smoother_step,
            (filtered_means[-1], filtered_covs[-1]),
            (
                jnp.arange(n_cam - 1),
                filtered_means[:-1],
                filtered_covs[:-1],
                lin_means[:-1],
            ),
            reverse=True,
        )

        # Concatenate with final frame (smoothed[-1] = filtered[-1])
        smoothed_means_iter = jnp.vstack([smoothed_means_iter, filtered_means[-1][None, ...]])
        smoothed_covs_iter = jnp.vstack([smoothed_covs_iter, filtered_covs[-1][None, ...]])

        return smoothed_means_iter, smoothed_covs_iter

    # Iterative EKS (IEKS): relinearize around previous smoothed trajectory
    # Initialize linearization trajectory with filtered estimates
    lin_means = filtered_means

    for iter_idx in range(num_iter):
        smoothed_means, smoothed_covs = run_one_rts_iteration(lin_means)

        # Update linearization trajectory for next iteration
        lin_means = smoothed_means

        # Optional: check convergence (early stopping if RMSE change < 1%)
        # Not implemented here to keep JAX-friendly (would need conditional break)

    return SmootherResult(
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        marginal_loglik=filter_result.marginal_loglik,
    )


# =============================================================================
# UKF Helper Functions (from dynamax)
# =============================================================================


def _compute_sigma_points(m: jnp.ndarray, P: jnp.ndarray, n: int, lamb: float) -> jnp.ndarray:
    """Generate sigma points for unscented transform.

    Args:
        m: Mean (n,)
        P: Covariance (n, n)
        n: State dimension
        lamb: UKF lambda parameter

    Returns:
        Sigma points (2n+1, n)
    """
    # Regularize covariance for Cholesky
    eps = 1e-8
    P_reg = P + eps * jnp.eye(n)

    # Compute Cholesky decomposition
    L = jnp.linalg.cholesky(P_reg)
    scale = jnp.sqrt(n + lamb)

    # Generate sigma points
    sigmas = [m]  # Mean point
    for i in range(n):
        sigmas.append(m + scale * L[:, i])  # Positive direction
        sigmas.append(m - scale * L[:, i])  # Negative direction

    return jnp.array(sigmas)


def _outer_product(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute outer product a ⊗ b."""
    return jnp.outer(a, b)


# =============================================================================
# Sigma-Point Smoother for UKF
# =============================================================================


def sigma_point_smoother(
    filter_result: UKFResult,
    ukf_config: UKFConfig,
    t_imu: np.ndarray,
    U_imu: np.ndarray,
    t_cam: np.ndarray,
    mask_cam: np.ndarray | None = None,
) -> SmootherResult:
    """Run sigma-point (RTS-like) smoother on UKF filter output.

    The sigma-point smoother uses the unscented transform to predict forward
    and then applies the RTS backward smoothing equations. This avoids
    computing Jacobians while still achieving accurate smoothing.

    Algorithm (Särkkä 2013, modified for sigma points):
        For k = N-1, ..., 1:
            1. Predict forward using sigma points: m_pred, P_pred, cross_cov
            2. Compute smoother gain: G_k = cross_cov @ P_pred^{-1}
            3. Smooth state: m_smooth[k] = m_filt[k] + G_k @ (m_smooth[k+1] - m_pred)
            4. Smooth cov: P_smooth[k] = P_filt[k] + G_k @ (P_smooth[k+1] - P_pred) @ G_k^T

    Args:
        filter_result: Output from unscented_kalman_filter()
        ukf_config: UKF configuration
        t_imu: IMU timestamps (N_imu,)
        U_imu: IMU measurements [ω_z, f_x, f_y] (N_imu, 3)
        t_cam: Camera timestamps (N_cam,)
        mask_cam: Camera mask (N_cam,) - if provided, applies blackout-aware noise scaling

    Returns:
        SmootherResult with smoothed means and covariances

    Note:
        Uses unscented transform for prediction to compute cross-covariance
        between filtered[k] and predicted[k+1], which is needed for the gain.
        State dimension is derived from filter_result.filtered_means.shape[1].

        Blackout-aware Q/R scaling (when mask_cam is provided):
        - During vision blackouts, reduces accel bias RW noise and IMU input noise
        - Helps tighten how hard post-gap vision "pulls" backward through gaps
        - Mirrors EKF RTS smoother behavior for consistency
    """
    # Convert to JAX arrays
    t_imu_jax = jnp.array(t_imu)
    U_imu_jax = jnp.array(U_imu)
    t_cam_jax = jnp.array(t_cam)

    # Convert mask_cam to JAX if provided
    mask_cam_jax = jnp.array(mask_cam) if mask_cam is not None else None

    # Extract filter outputs and derive state dimension from data
    filtered_means = filter_result.filtered_means  # (N_cam, n)
    filtered_covs = filter_result.filtered_covariances  # (N_cam, n, n)
    n_cam = len(t_cam)
    n = filtered_means.shape[1]  # Derive state dimension from data

    # Compute UKF sigma-point weights (dimension-dependent)
    alpha = ukf_config.alpha
    beta = ukf_config.beta
    kappa = ukf_config.kappa
    lamb = alpha**2 * (n + kappa) - n

    # Weights (Julier & Uhlmann)
    w_mean = jnp.concatenate(
        [jnp.array([lamb / (n + lamb)]), jnp.full(2 * n, 1.0 / (2 * (n + lamb)))]
    )
    w_cov_0 = lamb / (n + lamb) + (1 - alpha**2 + beta)
    w_cov = jnp.concatenate([jnp.array([w_cov_0]), jnp.full(2 * n, 1.0 / (2 * (n + lamb)))])

    # Compute mean IMU dt
    dt_imu_mean = float(jnp.mean(jnp.diff(t_imu_jax)))

    # Precompute IMU indices
    # Compute exact maximum per-frame count once (on CPU/NumPy) for robust padding
    cuts = np.searchsorted(t_imu, t_cam)
    counts = np.diff(np.r_[0, cuts])
    max_imu_per_frame = int(counts.max())

    def compute_imu_index_arrays():
        """Build padded index arrays for IMU samples between camera frames.

        IMPORTANT: This is a HOST-SIDE precomputation, NOT JIT-traced.
        The Python loop runs on CPU before smoother execution, producing static
        index arrays baked into the JIT-compiled smoother.

        Returns:
            jnp.ndarray: (n_cam, max_imu_per_frame) array of IMU indices
                where -1 indicates padding (invalid index)
        """
        all_indices = []
        for i in range(n_cam):
            if i == 0:
                indices = jnp.full(max_imu_per_frame, -1, dtype=jnp.int32)
            else:
                t_prev = t_cam_jax[i - 1]
                t_current = t_cam_jax[i]
                mask = (t_imu_jax > t_prev) & (t_imu_jax <= t_current)
                valid_indices = jnp.where(mask, size=max_imu_per_frame, fill_value=-1)[0]
                indices = valid_indices
            all_indices.append(indices)
        return jnp.array(all_indices)

    imu_index_arrays = compute_imu_index_arrays()

    def f(x, u, dt):
        return dynamics_function(x, u, dt, ukf_config.damping_coeff)

    # Process noise rates (derived from state dimension)
    Q_rate = build_Q_rate(ukf_config, n)

    def predict_between_frames_sigma(
        t_idx: int, x_k: jnp.ndarray, P_k: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predict from frame t_idx to t_idx+1 using sigma points.

        Returns:
            m_pred: Predicted mean (n,)
            P_pred: Predicted covariance (n, n)
            S_cross: Cross-covariance P(x_k, x_{k+1}) (n, n)
        """
        imu_indices = imu_index_arrays[t_idx + 1]

        # Blackout-aware noise scaling (mirrors EKF RTS smoother)
        # Check if either frame k or k+1 is in blackout
        in_blackout = (mask_cam_jax is not None) and (
            (~mask_cam_jax[t_idx]) | (~mask_cam_jax[t_idx + 1])
        )

        # Compute cross-covariance between filtered[k] and predicted[k+1]
        # by propagating sigma points through all IMU steps.
        # This correctly captures the linearization of the composed dynamics.

        def propagate_one_imu(carry, imu_idx):
            x_in, P_in = carry
            is_valid = imu_idx >= 0

            def do_propagate(state_cov):
                x_s, P_s = state_cov
                u = U_imu_jax[imu_idx]
                dt = lax.cond(
                    imu_idx > 0,
                    lambda: t_imu_jax[imu_idx] - t_imu_jax[imu_idx - 1],
                    lambda: jnp.array(dt_imu_mean),
                )

                dtype = x_s.dtype

                # Build Q_k with adaptive dropout scaling (Knob 1)
                if n == 8:
                    q_px = jnp.asarray(ukf_config.process_noise_pos * dt, dtype=dtype)
                    q_py = jnp.asarray(ukf_config.process_noise_pos * dt, dtype=dtype)
                    q_vx = jnp.asarray(ukf_config.process_noise_vel * dt, dtype=dtype)
                    q_vy = jnp.asarray(ukf_config.process_noise_vel * dt, dtype=dtype)
                    q_th = jnp.asarray(ukf_config.process_noise_heading * dt, dtype=dtype)
                    q_bg = jnp.asarray(ukf_config.process_noise_gyro_bias * dt, dtype=dtype)
                    q_bax = jnp.asarray(ukf_config.process_noise_accel_bias * dt, dtype=dtype)
                    q_bay = jnp.asarray(ukf_config.process_noise_accel_bias * dt, dtype=dtype)

                    # Knob 1: Adaptive Q during dropout (scales process noise)
                    if ukf_config.adaptive_q_during_dropout:
                        pos_scale = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ukf_config.dropout_q_pos_multiplier, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                        vel_scale = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ukf_config.dropout_q_vel_multiplier, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                        bias_scale_adaptive = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ukf_config.dropout_q_bias_multiplier, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                    else:
                        pos_scale = jnp.asarray(1.0, dtype=dtype)
                        vel_scale = jnp.asarray(1.0, dtype=dtype)
                        bias_scale_adaptive = jnp.asarray(1.0, dtype=dtype)

                    q_px = q_px * pos_scale
                    q_py = q_py * pos_scale
                    q_vx = q_vx * vel_scale
                    q_vy = q_vy * vel_scale
                    q_bg = q_bg * bias_scale_adaptive
                    q_bax = q_bax * bias_scale_adaptive
                    q_bay = q_bay * bias_scale_adaptive

                    Q_k = jnp.diag(
                        jnp.array([q_px, q_py, q_vx, q_vy, q_th, q_bg, q_bax, q_bay], dtype=dtype)
                    )
                else:
                    Q_k = Q_rate * dt
                    if ukf_config.adaptive_q_during_dropout:
                        scale = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ukf_config.dropout_q_pos_multiplier, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                        Q_k = Q_k * scale

                # IMU input noise mapping (matches EKF RTS smoother)
                if n == 8:
                    std_w = jnp.asarray(
                        ukf_config.imu_gyro_noise_density * jnp.sqrt(dt), dtype=dtype
                    )
                    std_f = jnp.asarray(
                        ukf_config.imu_accel_noise_density * jnp.sqrt(dt), dtype=dtype
                    )
                    Q_u = jnp.diag(jnp.array([std_w**2, std_f**2, std_f**2], dtype=dtype))

                    # Knob 2: Reduce IMU noise during blackout
                    if ukf_config.reduce_imu_noise_during_blackout:
                        imu_scale = lax.cond(
                            in_blackout,
                            lambda: jnp.asarray(ukf_config.blackout_imu_noise_scale, dtype=dtype),
                            lambda: jnp.asarray(1.0, dtype=dtype),
                        )
                        Q_u = Q_u * imu_scale

                    theta = x_s[4]
                    Q_total = assemble_Q(
                        ukf_config,
                        theta=theta,
                        dt=dt,
                        n=n,
                        has_vision=~in_blackout,
                        dtype=dtype,
                    )
                else:
                    Q_total = Q_k

                # Generate sigma points
                sigmas = _compute_sigma_points(x_s, P_s, n, lamb)

                # Propagate sigma points
                def prop_fn(x):
                    return f(x, u, dt)

                sigmas_prop = vmap(prop_fn)(sigmas)

                # Compute predicted mean
                m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)

                # Compute predicted covariance
                deviations = sigmas_prop - m_pred
                P_pred = jnp.tensordot(
                    w_cov,
                    vmap(_outer_product, in_axes=(0, 0))(deviations, deviations),
                    axes=1,
                )
                P_pred = P_pred + Q_total

                return m_pred, P_pred

            def no_propagate(state_cov):
                return state_cov

            return lax.cond(is_valid, do_propagate, no_propagate, (x_in, P_in)), None

        # Propagate through all IMU samples
        (x_pred, P_pred), _ = lax.scan(propagate_one_imu, (x_k, P_k), imu_indices)

        # Compute cross-covariance for smoother gain
        # We do one more sigma-point transform from x_k to x_pred
        # This gives us the cross-covariance P(x_k, x_pred)
        sigmas_k = _compute_sigma_points(x_k, P_k, n, lamb)

        # Propagate these sigma points through all IMU steps
        def propagate_sigma_through_all_imu(sigma_start):
            def prop_one(x_in, imu_idx):
                is_valid = imu_idx >= 0

                def do_prop(x):
                    u = U_imu_jax[imu_idx]
                    dt = lax.cond(
                        imu_idx > 0,
                        lambda: t_imu_jax[imu_idx] - t_imu_jax[imu_idx - 1],
                        lambda: jnp.array(dt_imu_mean),
                    )
                    return f(x, u, dt)

                return lax.cond(is_valid, do_prop, lambda x: x, x_in), None

            x_out, _ = lax.scan(prop_one, sigma_start, imu_indices)
            return x_out

        sigmas_pred = vmap(propagate_sigma_through_all_imu)(sigmas_k)

        # Compute cross-covariance: P(x_k, x_pred) = Σ w_cov[i] * (sigma_k[i] - m_k) * (sigma_pred[i] - m_pred)^T
        dev_k = sigmas_k - x_k
        dev_pred = sigmas_pred - x_pred
        S_cross = jnp.tensordot(
            w_cov, vmap(_outer_product, in_axes=(0, 0))(dev_k, dev_pred), axes=1
        )

        return x_pred, P_pred, S_cross

    def smoother_step(carry, args):
        """Single backward smoothing step."""
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Predict from k to k+1
        m_pred, P_pred, S_cross = predict_between_frames_sigma(t, filtered_mean, filtered_cov)

        # Compute smoother gain: G = S_cross @ P_pred^{-1}
        G = psd_solve(P_pred, S_cross.T).T

        # Smooth mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - P_pred) @ G.T
        smoothed_cov = symmetrize(smoothed_cov)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run smoother backward
    _, (smoothed_means, smoothed_covs) = lax.scan(
        smoother_step,
        (filtered_means[-1], filtered_covs[-1]),
        (jnp.arange(n_cam - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    # Concatenate with final frame
    smoothed_means = jnp.vstack([smoothed_means, filtered_means[-1][None, ...]])
    smoothed_covs = jnp.vstack([smoothed_covs, filtered_covs[-1][None, ...]])

    return SmootherResult(
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        marginal_loglik=filter_result.marginal_loglik,
    )
