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
from jax import jacfwd, lax, vmap

from trodestrack.models.ekf import EKFConfig, EKFResult
from trodestrack.models.filter_common import (
    compute_imu_index_arrays,
    dynamics_function,
    psd_solve,
    symmetrize,
)
from trodestrack.models.process_noise import assemble_Q
from trodestrack.models.ukf import UKFConfig, UKFResult

# =============================================================================
# Smoother Result Types
# =============================================================================


class SmootherResult(NamedTuple):
    """Smoother result (both EKF and UKF).

    Attributes
    ----------
    smoothed_means : jnp.ndarray
        Smoothed state means at camera times (N_cam, n).
    smoothed_covariances : jnp.ndarray
        Smoothed covariances at camera times (N_cam, n, n).
    marginal_loglik : float
        Marginal log-likelihood from filter.

    Notes
    -----
    n is the state dimension (8 for standard 2D, 10+ for extended layouts).
    """

    smoothed_means: jnp.ndarray  # (N_cam, n)
    smoothed_covariances: jnp.ndarray  # (N_cam, n, n)
    marginal_loglik: float


# =============================================================================
# Helper Functions
# =============================================================================


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
    """Run RTS (Rauch-Tung-Striebel) smoother on EKF output.

    Parameters
    ----------
    filter_result : EKFResult
        Output from :func:`trodestrack.models.ekf.extended_kalman_filter`.
    ekf_config : EKFConfig
        EKF configuration (for dynamics and Q assembly).
    t_imu : np.ndarray
        IMU timestamps (N_imu,) in seconds.
    U_imu : np.ndarray
        IMU measurements [ω_z(rad/s), f_x(m/s^2), f_y(m/s^2)] (N_imu, 3).
    t_cam : np.ndarray
        Camera timestamps (N_cam,) in seconds.
    num_iter : int, default 1
        Number of IEKS iterations; 1 yields standard RTS.
    mask_cam : np.ndarray | None, optional
        Camera validity mask (N_cam,). If provided, applies blackout-aware noise scaling.

    Returns
    -------
    SmootherResult
        Smoothed means and covariances at camera times; log-likelihood copied
        from the forward EKF pass.
    """
    # Convert to JAX arrays
    t_imu_jax = jnp.array(t_imu)
    U_imu_jax = jnp.array(U_imu)

    # Extract filter outputs and derive state dimension from data
    filtered_means = filter_result.filtered_means  # (N_cam, n)
    filtered_covs = filter_result.filtered_covariances  # (N_cam, n, n)
    n_cam = len(t_cam)
    n = filtered_means.shape[1]  # Derive state dimension from data

    # Compute mean IMU dt for fallback
    dt_imu_mean = float(jnp.mean(jnp.diff(t_imu_jax)))

    # Precompute IMU index arrays (host-side, using shared utility)
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    # Convert mask_cam to JAX if provided
    mask_cam_jax = jnp.array(mask_cam) if mask_cam is not None else None

    # Resolve state layout once for this smoother run
    from trodestrack.models.state_layout import get_heading_index, get_layout

    layout = get_layout(ekf_config.state_mode)

    # Compute Jacobian of dynamics
    def f(x, u, dt):
        return dynamics_function(x, u, dt, ekf_config.damping_coeff, layout)

    F_jac = jacfwd(f, argnums=0)

    def predict_between_frames(
        t_idx: int,
        x_k: jnp.ndarray,
        P_k: jnp.ndarray,
        x_k_lin: jnp.ndarray,
    ):
        """Predict from frame t_idx to t_idx+1 using IMU.

        Parameters
        ----------
        t_idx : int
            Time index k.
        x_k : jnp.ndarray
            State at time k (n,).
        P_k : jnp.ndarray
            Covariance at time k (n, n).
        x_k_lin : jnp.ndarray
            Linearization point at time k (for IEKS) (n,).

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            ``(m_pred, P_pred, G)`` with shapes (n,), (n, n), (n, n).
        """
        # Get IMU indices for interval [t_idx, t_idx+1)
        imu_indices = imu_index_arrays[t_idx + 1]

        # Blackout-aware noise scaling
        # Use target-frame rule to match forward filter behavior
        # Apply blackout scaling based on vision availability at target frame (t_idx+1)
        in_blackout = (mask_cam_jax is not None) and (~mask_cam_jax[t_idx + 1])

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
                h_idx = get_heading_index(layout)
                theta = x_s[h_idx] if n > h_idx else jnp.asarray(0.0, dtype=dtype)
                Q_total = assemble_Q(
                    ekf_config,
                    theta=theta,
                    dt=dt,
                    n=n,
                    has_vision=jnp.logical_not(in_blackout),
                    dtype=dtype,
                )

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

        Parameters
        ----------
        carry : tuple[jnp.ndarray, jnp.ndarray]
            ``(smoothed_mean_next, smoothed_cov_next)`` at time k+1.
        args : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            ``(t, filtered_mean_k, filtered_cov_k, lin_mean_k)`` at time k.

        Returns
        -------
        tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
            Updated carry and smoothed estimates at time k.
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

        Parameters
        ----------
        lin_means : jnp.ndarray
            Linearization trajectory (N_cam, n).

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            ``(smoothed_means, smoothed_covs)`` each with shapes (N_cam, n) and (N_cam, n, n).
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

    for _iter_idx in range(num_iter):
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

    Parameters
    ----------
    m : jnp.ndarray
        Mean (n,).
    P : jnp.ndarray
        Covariance (n, n).
    n : int
        State dimension.
    lamb : float
        UKF lambda parameter.

    Returns
    -------
    jnp.ndarray
        Sigma points (2n+1, n).
    """
    # Regularize covariance for Cholesky
    P_reg = symmetrize(P)

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
    """Run sigma-point (RTS-like) smoother on UKF output.

    Parameters
    ----------
    filter_result : UKFResult
        Output from :func:`trodestrack.models.ukf.unscented_kalman_filter`.
    ukf_config : UKFConfig
        UKF configuration (for dynamics and Q assembly).
    t_imu : np.ndarray
        IMU timestamps (N_imu,) in seconds.
    U_imu : np.ndarray
        IMU measurements [ω_z(rad/s), f_x(m/s^2), f_y(m/s^2)] (N_imu, 3).
    t_cam : np.ndarray
        Camera timestamps (N_cam,) in seconds.
    mask_cam : np.ndarray | None, optional
        Camera validity mask (N_cam,). If provided, applies blackout-aware noise scaling.

    Returns
    -------
    SmootherResult
        Smoothed means and covariances at camera times; log-likelihood copied
        from the forward UKF pass.
    Notes
    -----
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

    # Precompute IMU index arrays (host-side, using shared utility)
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    # Resolve state layout once for this smoother run
    from trodestrack.models.state_layout import get_heading_index, get_layout

    layout = get_layout(ukf_config.state_mode)

    def f(x, u, dt):
        return dynamics_function(x, u, dt, ukf_config.damping_coeff, layout)

    # Process noise assembly handled via assemble_Q per step

    def predict_between_frames_sigma(
        t_idx: int, x_k: jnp.ndarray, P_k: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predict from frame t_idx to t_idx+1 using sigma points.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            m_pred (n,), P_pred (n, n), S_cross = P(x_k, x_{k+1}) (n, n).
        """
        imu_indices = imu_index_arrays[t_idx + 1]

        # Blackout-aware noise scaling (mirrors EKF RTS smoother)
        # Use target-frame rule to match forward filter behavior
        # Apply blackout scaling based on vision availability at target frame (t_idx+1)
        in_blackout = (mask_cam_jax is not None) and (~mask_cam_jax[t_idx + 1])

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
                # Predict covariance using shared assemble_Q
                h_idx = get_heading_index(layout)
                theta = x_s[h_idx] if n > h_idx else jnp.asarray(0.0, dtype=dtype)
                Q_total = assemble_Q(
                    ukf_config,
                    theta=theta,
                    dt=dt,
                    n=n,
                    has_vision=jnp.logical_not(in_blackout),
                    dtype=dtype,
                )

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
