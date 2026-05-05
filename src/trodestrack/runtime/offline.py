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

from trodestrack.models.ekf import EKF3DResult, EKFConfig, EKFResult
from trodestrack.models.filter_common import (
    build_quaternion_transition_jacobian,
    compute_imu_index_arrays,
    dynamics_function,
    normalize_state_orientation,
    psd_solve,
    state_yaw,
    symmetrize,
    validate_imu_input_shape,
    validate_timestamps,
    wrap_angle,
)

# Note: full camera-shape validator (validate_camera_input_shapes) is used
# only by the EKF/UKF entrypoints, which take LED + mask + conf arrays. The
# smoothers only get t_cam + mask_cam, validated inline below.
from trodestrack.models.process_noise import assemble_Q
from trodestrack.models.state_layout import StateLayout, get_heading_index, get_layout
from trodestrack.models.ukf import UKFConfig, UKFResult, compute_sigma_points

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
    n is the state dimension (e.g. 8 for 2d_full, 5 for vision_only,
    10 for 2d_cam_3d_imu).  See ``StateLayout`` for all supported modes.
    """

    smoothed_means: jnp.ndarray  # (N_cam, n)
    smoothed_covariances: jnp.ndarray  # (N_cam, n, n)
    marginal_loglik: float


RTS_SMOOTHER_STATIC_ARGNAMES = ("layout", "ekf_config", "num_iter")
# Donate lin_means_init (arg 0) and filtered_covs (arg 2) to enable buffer
# reuse in scan carry iterations. These arrays are copied by the wrapper and
# are never used after smoother returns.
RTS_SMOOTHER_DONATE_ARGNUMS: tuple[int, ...] = (0, 2)


# =============================================================================
# Helper Functions
# =============================================================================


def _transition_mean_and_jacobian(
    state_mean: jnp.ndarray,
    linearization_mean: jnp.ndarray,
    u_imu: jnp.ndarray,
    dt_imu: jnp.ndarray,
    *,
    ekf_config: EKFConfig,
    layout: StateLayout,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict one IMU step and return a transition Jacobian.

    Quaternion layouts use the same first-order Jacobian structure as the EKF
    prediction step to avoid autodiff through the zero-rotation quaternion
    exponential. Scalar-heading layouts keep the existing autodiff path.
    """

    def f(x):
        return dynamics_function(
            x,
            u_imu,
            dt_imu,
            ekf_config.damping_coeff,
            layout,
            gravity_body=ekf_config.imu_gravity_body,
            enable_experimental_accel_translation=(
                ekf_config.enable_experimental_accel_translation
            ),
        )

    mean_pred = f(state_mean)

    if not layout.has_quaternion_orientation:
        return mean_pred, jacfwd(f)(linearization_mean)

    linearization_pred = f(linearization_mean)
    F_x = build_quaternion_transition_jacobian(
        linearization_mean,
        linearization_pred,
        dt_imu,
        ekf_config.damping_coeff,
        layout,
        u_imu=u_imu,
        enable_experimental_accel_translation=(
            ekf_config.enable_experimental_accel_translation
        ),
    )

    return normalize_state_orientation(mean_pred, layout), F_x


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
    has_mask = jnp.asarray(mask_is_provided, dtype=bool)
    has_scalar_heading = layout.has_heading_2d

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

                x_pred, F_k = _transition_mean_and_jacobian(
                    x_s,
                    x_lin_s,
                    u,
                    dt,
                    ekf_config=ekf_config,
                    layout=layout,
                )

                dtype = x_pred.dtype
                theta = (
                    x_pred[get_heading_index(layout)]
                    if has_scalar_heading
                    else state_yaw(x_pred, layout)
                )
                Q_total = assemble_Q(
                    ekf_config,
                    theta=theta,
                    dt=dt,
                    n=n,
                    has_vision=jnp.logical_not(in_blackout),
                    dtype=dtype,
                    orientation_quaternion=(
                        x_pred[jnp.array(layout.heading_idx, dtype=jnp.int32)]
                        if layout.has_quaternion_orientation
                        else None
                    ),
                )

                P_pred = F_k @ P_s @ F_k.T + Q_total
                P_pred = symmetrize(P_pred)
                F_new = F_k @ F_prev
                x_lin_pred, _ = _transition_mean_and_jacobian(
                    x_lin_s,
                    x_lin_s,
                    u,
                    dt,
                    ekf_config=ekf_config,
                    layout=layout,
                )

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

        m_pred, P_pred, G = predict_between_frames(
            t, filtered_mean, filtered_cov, lin_mean
        )

        smoothed_mean_next_aligned = smoothed_mean_next
        if layout.has_heading_2d:
            h_idx = get_heading_index(layout)
            resid = smoothed_mean_next_aligned - m_pred
            resid = resid.at[h_idx].set(wrap_angle(resid[h_idx]))
        elif layout.has_quaternion_orientation:
            quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
            sign = jnp.where(
                jnp.dot(smoothed_mean_next[quat_idx], m_pred[quat_idx]) < 0.0,
                jnp.asarray(-1.0, dtype=smoothed_mean_next.dtype),
                jnp.asarray(1.0, dtype=smoothed_mean_next.dtype),
            )
            smoothed_mean_next_aligned = smoothed_mean_next.at[quat_idx].set(
                sign * smoothed_mean_next[quat_idx]
            )
            resid = smoothed_mean_next_aligned - m_pred
        else:
            resid = smoothed_mean_next_aligned - m_pred

        smoothed_mean = filtered_mean + G @ resid
        smoothed_mean = normalize_state_orientation(smoothed_mean, layout)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - P_pred) @ G.T
        smoothed_cov = symmetrize(smoothed_cov)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    def run_one_iteration(
        lin_means_current: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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

        smoothed_means_iter = jnp.vstack(
            [smoothed_means_iter, filtered_means[-1][None, ...]]
        )
        smoothed_covs_iter = jnp.vstack(
            [smoothed_covs_iter, filtered_covs[-1][None, ...]]
        )

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
    filter_result: EKFResult | EKF3DResult,
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
    filter_result : EKFResult or EKF3DResult
        Output from :func:`trodestrack.models.ekf.extended_kalman_filter` or
        :func:`trodestrack.models.ekf.extended_kalman_filter_3d`.
    ekf_config : EKFConfig
        EKF configuration (for dynamics and Q assembly).
    t_imu : np.ndarray
        IMU timestamps (N_imu,) in seconds.
    U_imu : np.ndarray
        IMU measurements. Shape depends on the state layout:
        - (N_imu, 3) ``[ω_z, f_x, f_y]`` for non-quaternion layouts
          (``"2d_full"``, ``"vision_only"``, and the default
          ``"2d_cam_3d_imu"`` as a degenerate path that leaves ``vz`` idle);
        - (N_imu, 4) ``[ω_z, f_x, f_y, f_z]`` for the default
          ``"2d_cam_3d_imu"`` layout when 3D-velocity dynamics are desired;
        - (N_imu, 6) ``[ω_x, ω_y, ω_z, f_x, f_y, f_z]`` for
          quaternion-orientation layouts (e.g. ``"3d_cam_6dof_imu"``).
        Channel-vs-layout compatibility is enforced by
        :func:`trodestrack.models.filter_common.validate_imu_input_shape`.
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
    # Reject num_iter < 1 explicitly. The inner ``for _ in range(num_iter)``
    # loop never runs for 0 / negative values and the wrapper would
    # otherwise return the filtered result as if smoothing succeeded.
    # Mirrors EKFConfig.num_iter validation.
    if not isinstance(num_iter, int) or num_iter < 1:
        raise ValueError(
            "num_iter must be an integer >= 1 (1 = standard RTS, "
            f">1 = IEKS iterations); got {num_iter!r}."
        )

    # Reject non-finite filter_result entries up front. The smoother runs
    # IMU pre-integration around these means and would otherwise propagate
    # NaN/inf through every backward pass and report a "successful"
    # smoothed result.
    fmeans = np.asarray(filter_result.filtered_means)
    fcovs = np.asarray(filter_result.filtered_covariances)
    if not np.all(np.isfinite(fmeans)):
        n_bad = int(np.sum(~np.isfinite(fmeans)))
        raise ValueError(
            f"rts_smoother: filter_result.filtered_means contains {n_bad} "
            "non-finite value(s) (NaN/inf); the smoother cannot recover "
            "from a corrupted forward pass."
        )
    if not np.all(np.isfinite(fcovs)):
        n_bad = int(np.sum(~np.isfinite(fcovs)))
        raise ValueError(
            f"rts_smoother: filter_result.filtered_covariances contains "
            f"{n_bad} non-finite value(s) (NaN/inf)."
        )

    # Validate IMU input shape early so silent channel mismatches fail loudly.
    validate_imu_input_shape(
        U_imu,
        get_layout(ekf_config.state_mode),
        t_imu=t_imu,
        func_name="rts_smoother",
    )

    # Reject non-finite / non-monotonic timestamps so np.diff(t_imu) and
    # the smoother's IMU pre-integration don't silently propagate NaN.
    validate_timestamps(t_imu, name="t_imu", func_name="rts_smoother", min_size=2)
    validate_timestamps(t_cam, name="t_cam", func_name="rts_smoother")

    # Validate t_cam / mask_cam alignment with the filter result. JAX
    # indexing silently clamps a too-short mask_cam to its last in-range
    # value, marking every later frame with that stale flag. Catch the
    # length mismatch at the entry point.
    n_cam = int(filter_result.filtered_means.shape[0])
    t_cam_arr = np.asarray(t_cam)
    if t_cam_arr.ndim != 1 or t_cam_arr.shape[0] != n_cam:
        raise ValueError(
            f"rts_smoother: t_cam must have shape ({n_cam},) to match "
            f"filter_result.filtered_means, got {t_cam_arr.shape}."
        )
    if mask_cam is not None:
        mask_arr = np.asarray(mask_cam)
        if mask_arr.shape != (n_cam,):
            raise ValueError(
                f"rts_smoother: mask_cam must have shape ({n_cam},) to match "
                f"t_cam / filter_result, got {mask_arr.shape}."
            )

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


# =============================================================================
# Sigma-Point Smoother for UKF
# =============================================================================

SIGMA_POINT_SMOOTHER_STATIC_ARGNAMES = ("mask_is_provided", "layout", "ukf_config")
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
    mask_cam_jax: jnp.ndarray,
    w_mean: jnp.ndarray,
    w_cov: jnp.ndarray,
    lamb: float,
    *,
    mask_is_provided: bool,
    layout: StateLayout,
    ukf_config: UKFConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Core sigma-point smoother implementation staged under ``jax.jit``.

    This is the JIT-compiled inner loop. Call via the public wrapper
    :func:`sigma_point_smoother` which handles preprocessing.
    """
    n_cam = filtered_means.shape[0]
    n = filtered_means.shape[1]
    has_mask = jnp.asarray(mask_is_provided, dtype=bool)

    def f(x, u, dt):
        return dynamics_function(
            x,
            u,
            dt,
            ukf_config.damping_coeff,
            layout,
            gravity_body=ukf_config.imu_gravity_body,
        )

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
        in_blackout = has_mask & (~mask_cam_jax[t_idx + 1])

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
                sigmas = compute_sigma_points(x_s, P_s, n, lamb)

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
                    vmap(jnp.outer, in_axes=(0, 0))(deviations, deviations),
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
        sigmas_k = compute_sigma_points(x_k, P_k, n, lamb)

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
            w_cov, vmap(jnp.outer, in_axes=(0, 0))(dev_k, dev_pred), axes=1
        )

        return x_pred, P_pred, S_cross

    def smoother_step(carry, args):
        """Single backward smoothing step."""
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Predict from k to k+1
        m_pred, P_pred, S_cross = predict_between_frames_sigma(
            t, filtered_mean, filtered_cov
        )

        # Compute smoother gain: G = S_cross @ P_pred^{-1}
        G = psd_solve(P_pred, S_cross.T).T

        h_idx = get_heading_index(layout)
        resid = smoothed_mean_next - m_pred
        resid = resid.at[h_idx].set(wrap_angle(resid[h_idx]))

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
        IMU measurements. Either 3 channels ``[ω_z, f_x, f_y]`` for 2D layouts
        or 4 channels ``[ω_z, f_x, f_y, f_z]`` for 3D-velocity layouts
        (e.g. ``2d_cam_3d_imu``).
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
    # Reject non-finite filter_result entries up front. The smoother runs
    # IMU pre-integration around these means and would otherwise propagate
    # NaN/inf through every backward pass and report a "successful"
    # smoothed result.
    fmeans = np.asarray(filter_result.filtered_means)
    fcovs = np.asarray(filter_result.filtered_covariances)
    if not np.all(np.isfinite(fmeans)):
        n_bad = int(np.sum(~np.isfinite(fmeans)))
        raise ValueError(
            f"sigma_point_smoother: filter_result.filtered_means contains "
            f"{n_bad} non-finite value(s) (NaN/inf); the smoother cannot "
            "recover from a corrupted forward pass."
        )
    if not np.all(np.isfinite(fcovs)):
        n_bad = int(np.sum(~np.isfinite(fcovs)))
        raise ValueError(
            f"sigma_point_smoother: filter_result.filtered_covariances "
            f"contains {n_bad} non-finite value(s) (NaN/inf)."
        )

    # Validate IMU input shape early so silent channel mismatches fail loudly.
    validate_imu_input_shape(
        U_imu,
        get_layout(ukf_config.state_mode),
        t_imu=t_imu,
        func_name="sigma_point_smoother",
    )

    # Reject non-finite / non-monotonic timestamps so np.diff(t_imu) and
    # the sigma-point smoother's IMU pre-integration don't silently
    # propagate NaN.
    validate_timestamps(
        t_imu, name="t_imu", func_name="sigma_point_smoother", min_size=2
    )
    validate_timestamps(t_cam, name="t_cam", func_name="sigma_point_smoother")

    # Validate t_cam / mask_cam alignment with the filter result so a
    # too-short mask_cam doesn't silently reuse its last in-range value
    # for every later frame (JAX out-of-bounds indexing clamps).
    n_cam = int(filter_result.filtered_means.shape[0])
    t_cam_arr_check = np.asarray(t_cam)
    if t_cam_arr_check.ndim != 1 or t_cam_arr_check.shape[0] != n_cam:
        raise ValueError(
            f"sigma_point_smoother: t_cam must have shape ({n_cam},) to "
            f"match filter_result, got {t_cam_arr_check.shape}."
        )
    if mask_cam is not None:
        mask_arr_check = np.asarray(mask_cam)
        if mask_arr_check.shape != (n_cam,):
            raise ValueError(
                f"sigma_point_smoother: mask_cam must have shape ({n_cam},) "
                f"to match t_cam / filter_result, got {mask_arr_check.shape}."
            )

    # Convert to JAX arrays
    t_imu_jax = jnp.array(t_imu)
    U_imu_jax = jnp.array(U_imu)
    t_cam_jax = jnp.array(t_cam)

    # Copy filtered arrays before donation to avoid invalidating filter_result
    # Buffer donation enables efficient reuse through scan iterations inside JIT
    filtered_means = filter_result.filtered_means.copy()  # (N_cam, n)
    filtered_covs = filter_result.filtered_covariances.copy()  # (N_cam, n, n)

    # Convert mask_cam to JAX (sentinel all-True when not provided, matching EKF smoother)
    mask_is_provided = mask_cam is not None
    if mask_is_provided:
        mask_cam_jax = jnp.array(mask_cam, dtype=bool)
    else:
        mask_cam_jax = jnp.ones(filtered_means.shape[0], dtype=bool)
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
    w_cov = jnp.concatenate(
        [jnp.array([w_cov_0]), jnp.full(2 * n, 1.0 / (2 * (n + lamb)))]
    )

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
        mask_is_provided=mask_is_provided,
        layout=layout,
        ukf_config=ukf_config,
    )

    return SmootherResult(
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        marginal_loglik=filter_result.marginal_loglik,
    )
