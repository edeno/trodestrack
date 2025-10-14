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

import jax
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
from trodestrack.models.state_layout import StateLayout, get_heading_index, get_layout
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


RTS_SMOOTHER_STATIC_ARGNAMES = ("layout", "ekf_config", "num_iter")
# Donate filtered_means (arg 1) and filtered_covs (arg 2) to enable buffer reuse
# in scan carry iterations. These arrays are large (N_cam, n) and (N_cam, n, n)
# and are never used after smoother returns.
RTS_SMOOTHER_DONATE_ARGNUMS: tuple[int, ...] = (1, 2)


# =============================================================================
# Helper Functions
# =============================================================================


def _rts_smoother_impl(
    lin_means_init: jnp.ndarray,
    filtered_means: jnp.ndarray,
    filtered_covs: jnp.ndarray,
    t_imu_jax: jnp.ndarray,
    U_imu_jax: jnp.ndarray,
    mask_cam_jax: jnp.ndarray,
    mask_is_provided: bool,
    imu_index_arrays: jnp.ndarray,
    dt_imu_mean: jnp.ndarray,
    *,
    num_iter: int,
    ekf_config: EKFConfig,
    layout: StateLayout,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Core RTS smoother staged under ``jax.jit``."""
    n_cam = filtered_means.shape[0]
    n = filtered_means.shape[1]

    def f(x, u, dt):
        return dynamics_function(x, u, dt, ekf_config.damping_coeff, layout)

    F_jac = jacfwd(f, argnums=0)
    has_mask = jnp.asarray(mask_is_provided, dtype=bool)
    h_idx = get_heading_index(layout)
    has_heading = h_idx < n

    def predict_between_frames(
        t_idx: int,
        x_k: jnp.ndarray,
        P_k: jnp.ndarray,
        x_k_lin: jnp.ndarray,
    ):
        imu_indices = imu_index_arrays[t_idx + 1]
        in_blackout = has_mask & (~mask_cam_jax[t_idx + 1])

        def propagate_one_imu(carry, imu_idx):
            x_in, P_in, F_accum, x_lin_in = carry
            is_valid = imu_idx >= 0

            def do_propagate(state_cov_F_lin):
                x_s, P_s, F_prev, x_lin_s = state_cov_F_lin
                u = U_imu_jax[imu_idx]
                dt = lax.cond(
                    imu_idx > 0,
                    lambda: t_imu_jax[imu_idx] - t_imu_jax[imu_idx - 1],
                    lambda: dt_imu_mean,
                )

                x_pred = f(x_s, u, dt)
                F_k = F_jac(x_lin_s, u, dt)

                dtype = x_pred.dtype
                theta = x_pred[h_idx] if has_heading else jnp.asarray(0.0, dtype=dtype)
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
                F_new = F_k @ F_prev
                x_lin_pred = f(x_lin_s, u, dt)

                return (x_pred, P_pred, F_new, x_lin_pred), None

            def no_propagate(state_cov_F_lin):
                return state_cov_F_lin, None

            return lax.cond(
                is_valid,
                do_propagate,
                no_propagate,
                (x_in, P_in, F_accum, x_lin_in),
            )

        F_init = jnp.eye(n)
        (x_pred, P_pred, F_total, _), _ = lax.scan(
            propagate_one_imu, (x_k, P_k, F_init, x_k_lin), imu_indices
        )

        G = psd_solve(P_pred, F_total @ P_k).T
        return x_pred, P_pred, G

    def smoother_step(carry, args):
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov, lin_mean = args

        m_pred, P_pred, G = predict_between_frames(t, filtered_mean, filtered_cov, lin_mean)

        # Correct for angle wrapping in heading (if present in layout)
        h_idx = get_heading_index(layout)
        resid = smoothed_mean_next - m_pred
        resid = resid.at[h_idx].set(jnp.arctan2(jnp.sin(resid[h_idx]), jnp.cos(resid[h_idx])))

        smoothed_mean = filtered_mean + G @ resid
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - P_pred) @ G.T
        smoothed_cov = symmetrize(smoothed_cov)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    def run_one_iteration(lin_means_current: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        (_, (smoothed_means_iter, smoothed_covs_iter)) = lax.scan(
            smoother_step,
            (filtered_means[-1], filtered_covs[-1]),
            (
                jnp.arange(n_cam - 1),
                filtered_means[:-1],
                filtered_covs[:-1],
                lin_means_current[:-1],
            ),
            reverse=True,
        )

        smoothed_means_iter = jnp.vstack([smoothed_means_iter, filtered_means[-1][None, ...]])
        smoothed_covs_iter = jnp.vstack([smoothed_covs_iter, filtered_covs[-1][None, ...]])

        return smoothed_means_iter, smoothed_covs_iter

    lin_means = lin_means_init
    smoothed_means = filtered_means
    smoothed_covs = filtered_covs

    for _ in range(num_iter):
        smoothed_means, smoothed_covs = run_one_iteration(lin_means)
        lin_means = smoothed_means

    return smoothed_means, smoothed_covs


_rts_smoother_jit = jax.jit(
    _rts_smoother_impl,
    static_argnames=RTS_SMOOTHER_STATIC_ARGNAMES,
    donate_argnums=RTS_SMOOTHER_DONATE_ARGNUMS,
)


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

    # Copy filtered arrays before donation to avoid invalidating filter_result
    # Buffer donation enables efficient reuse through scan iterations inside JIT
    filtered_means = filter_result.filtered_means.copy()
    filtered_covs = filter_result.filtered_covariances.copy()

    dt_imu_mean = jnp.mean(jnp.diff(t_imu_jax))
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    mask_is_provided = mask_cam is not None
    if mask_is_provided:
        mask_cam_jax = jnp.array(mask_cam, dtype=bool)
    else:
        mask_cam_jax = jnp.ones(filtered_means.shape[0], dtype=bool)

    layout = get_layout(ekf_config.state_mode)

    lin_scratch = filtered_means.copy()

    smoothed_means, smoothed_covs = _rts_smoother_jit(
        lin_scratch,
        filtered_means,
        filtered_covs,
        t_imu_jax,
        U_imu_jax,
        mask_cam_jax,
        mask_is_provided,
        imu_index_arrays,
        dt_imu_mean,
        num_iter=num_iter,
        ekf_config=ekf_config,
        layout=layout,
    )

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

SIGMA_POINT_SMOOTHER_STATIC_ARGNAMES = ("layout", "ukf_config")
# Donate filtered_means (arg 0) and filtered_covs (arg 1) to enable buffer reuse
# in scan carry iterations. These arrays are large (N_cam, n) and (N_cam, n, n)
# and are never used after smoother returns.
SIGMA_POINT_SMOOTHER_DONATE_ARGNUMS: tuple[int, ...] = (0, 1)


def _sigma_point_smoother_impl(
    filtered_means: jnp.ndarray,
    filtered_covs: jnp.ndarray,
    t_imu_jax: jnp.ndarray,
    U_imu_jax: jnp.ndarray,
    t_cam_jax: jnp.ndarray,
    imu_index_arrays: jnp.ndarray,
    dt_imu_mean: jnp.ndarray,
    mask_cam_jax: jnp.ndarray | None,
    w_mean: jnp.ndarray,
    w_cov: jnp.ndarray,
    lamb: float,
    *,
    layout: StateLayout,
    ukf_config: UKFConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Core sigma-point smoother implementation staged under ``jax.jit``.

    This is the JIT-compiled inner loop. Call via the public wrapper
    :func:`sigma_point_smoother` which handles preprocessing.
    """
    n_cam = filtered_means.shape[0]
    n = filtered_means.shape[1]

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
        h_idx = get_heading_index(layout)
        has_heading = h_idx < n

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

                # Generate sigma points
                sigmas = _compute_sigma_points(x_s, P_s, n, lamb)

                # Propagate sigma points
                def prop_fn(x):
                    return f(x, u, dt)

                sigmas_prop = vmap(prop_fn)(sigmas)

                # Compute predicted mean
                m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)

                dtype = m_pred.dtype
                theta = m_pred[h_idx] if has_heading else jnp.asarray(0.0, dtype=dtype)
                Q_total = assemble_Q(
                    ukf_config,
                    theta=theta,
                    dt=dt,
                    n=n,
                    has_vision=jnp.logical_not(in_blackout),
                    dtype=dtype,
                )

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

        # Correct for angle wrapping in heading (if present in layout)
        h_idx = get_heading_index(layout)
        resid = smoothed_mean_next - m_pred
        resid = resid.at[h_idx].set(jnp.arctan2(jnp.sin(resid[h_idx]), jnp.cos(resid[h_idx])))

        # Smooth mean and covariance
        smoothed_mean = filtered_mean + G @ resid
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

    return smoothed_means, smoothed_covs


_sigma_point_smoother_jit = jax.jit(
    _sigma_point_smoother_impl,
    static_argnames=SIGMA_POINT_SMOOTHER_STATIC_ARGNAMES,
    donate_argnums=SIGMA_POINT_SMOOTHER_DONATE_ARGNUMS,
)


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
        IMU measurements [ω_z(rad/s), f_x(m/s^2), f_y(m/s^2)] (N_imu, 3 or 6).
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
    t_cam_jax = jnp.array(t_cam)

    # Convert mask_cam to JAX if provided
    mask_cam_jax = jnp.array(mask_cam) if mask_cam is not None else None

    # Copy filtered arrays before donation to avoid invalidating filter_result
    # Buffer donation enables efficient reuse through scan iterations inside JIT
    filtered_means = filter_result.filtered_means.copy()  # (N_cam, n)
    filtered_covs = filter_result.filtered_covariances.copy()  # (N_cam, n, n)
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
    dt_imu_mean = jnp.mean(jnp.diff(t_imu_jax))

    # Precompute IMU index arrays (host-side, using shared utility)
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    # Resolve state layout once for this smoother run
    layout = get_layout(ukf_config.state_mode)

    # Call JIT-compiled implementation
    smoothed_means, smoothed_covs = _sigma_point_smoother_jit(
        filtered_means,
        filtered_covs,
        t_imu_jax,
        U_imu_jax,
        t_cam_jax,
        imu_index_arrays,
        dt_imu_mean,
        mask_cam_jax,
        w_mean,
        w_cov,
        lamb,
        layout=layout,
        ukf_config=ukf_config,
    )

    return SmootherResult(
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        marginal_loglik=filter_result.marginal_loglik,
    )
