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

from trodestrack.models.filter_common import (
    FilterCoreConfig,
    FilterState,
    chi2_threshold,
    compute_imu_index_arrays,
    dynamics_function,
    initialize_state,
    symmetrize,
    update_zupt,
    wrap_angle,
)
from trodestrack.models.filter_update import ukf_projected_update
from trodestrack.models.process_noise import assemble_Q
from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.sensors.heading_pseudo import HeadingPseudoModel
from trodestrack.models.state_layout import StateLayout, get_heading_index, get_layout

# =============================================================================
# Configuration & State
# =============================================================================


@dataclass
class UKFConfig(FilterCoreConfig):
    """Unscented Kalman filter configuration extending FilterCoreConfig.

    Parameters
    ----------
    alpha : float, default 1.732
        Sigma-point spread parameter. Smaller (e.g., 1e-3) is more conservative;
        larger (≈√3) captures stronger nonlinearities.
    beta : float, default 2.0
        Prior knowledge parameter (2 is optimal for Gaussian priors).
    kappa : float, default 1.0
        Secondary scaling parameter.

    Notes
    -----
    Use ``UKFConfig.conservative()`` for numerically safer presets, or
    ``UKFConfig.aggressive()`` for stronger nonlinearity capture.
    """

    alpha: float = 1.732  # sqrt(3), Sigma-point spread
    beta: float = 2.0  # Prior knowledge (2 = Gaussian optimal)
    kappa: float = 1.0  # Secondary scaling

    @classmethod
    def conservative(cls, **kwargs) -> UKFConfig:
        """Conservative UKF preset with small alpha for numerical stability.

        Uses alpha=1e-3 so sigma points stay very close to the mean, minimizing
        risk of numerical issues with non-PSD covariances or extreme nonlinearities.
        Recommended for:
            - High-rate IMU (>200 Hz) with small dt
            - Initial development/debugging
            - When encountering numerical instabilities

        Parameters
        ----------
        **kwargs
            Override any default parameters.

        Returns
        -------
        UKFConfig
            Preset with alpha=1e-3, beta=2.0, kappa=0.0.

        Example:
            >>> config = UKFConfig.conservative(use_mahalanobis_gating=True)
        """
        return cls(alpha=1e-3, beta=2.0, kappa=0.0, **kwargs)

    @classmethod
    def aggressive(cls, **kwargs) -> UKFConfig:
        """Aggressive UKF preset with large alpha for capturing nonlinearity.

        Uses alpha=sqrt(3) to spread sigma points widely, better approximating
        the distribution after nonlinear transformations. May induce numerical
        strain for poorly-conditioned covariances.
        Recommended for:
            - Low-rate IMU or long dt
            - Strong nonlinearities (fast rotations)
            - Well-tuned noise parameters

        Parameters
        ----------
        **kwargs
            Override any default parameters.

        Returns
        -------
        UKFConfig
            Preset with alpha=sqrt(3), beta=2.0, kappa=1.0.

        Example:
            >>> config = UKFConfig.aggressive(use_heading_measurement=True)
        """
        return cls(alpha=1.732, beta=2.0, kappa=1.0, **kwargs)


UKFState = FilterState


class UKFResult(NamedTuple):
    """UKF filtering result.

    Attributes
    ----------
    filtered_means : jnp.ndarray
        Filtered state means at camera times (N_cam, n).
    filtered_covariances : jnp.ndarray
        Filtered covariances at camera times (N_cam, n, n).
    predicted_means : jnp.ndarray
        Predicted state means at camera times (N_cam, n).
    predicted_covariances : jnp.ndarray
        Predicted covariances at camera times (N_cam, n, n).
    marginal_loglik : float
        Sum of per-frame Gaussian log-likelihoods.
    estimated_led_distance : float | None
        Auto-detected LED spacing (m), or None if explicitly provided.
    """

    filtered_means: jnp.ndarray  # (N_cam, n)
    filtered_covariances: jnp.ndarray  # (N_cam, n, n)
    predicted_means: jnp.ndarray  # (N_cam, n)
    predicted_covariances: jnp.ndarray  # (N_cam, n, n)
    marginal_loglik: float
    estimated_led_distance: float | None


# =============================================================================
# Sigma-Point Generation
# =============================================================================


def compute_sigma_points(mean: jnp.ndarray, cov: jnp.ndarray, n: int, lamb: float) -> jnp.ndarray:
    """Compute (2n+1) sigma points for the unscented transform.

    Parameters
    ----------
    mean : jnp.ndarray
        State mean (n,).
    cov : jnp.ndarray
        State covariance (n, n), PSD.
    n : int
        State dimension.
    lamb : float
        UKF lambda parameter (α²(n+κ) − n).

    Returns
    -------
    jnp.ndarray
        Sigma points (2n+1, n) with central point at index 0.
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
    """Compute unscented weights for mean and covariance.

    Parameters
    ----------
    n : int
        State dimension.
    alpha : float
        Spread parameter.
    beta : float
        Prior knowledge parameter.
    lamb : float
        Lambda = α²(n + κ) − n.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(w_mean, w_cov)`` each of shape (2n+1,).

    Notes
    -----
    Särkkä Eq. (5.77):
    w₀^m = λ/(n+λ),  wᵢ^m = 1/(2(n+λ)),
    w₀^c = λ/(n+λ) + (1 − α² + β),  wᵢ^c = 1/(2(n+λ)).
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
# NOTE: make_led_selector() is now imported from filter_common.py (shared with EKF)


def _outer_product_batch(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Batched outer products for covariance reconstruction.

    Parameters
    ----------
    x : jnp.ndarray
        Array (N, n).
    y : jnp.ndarray
        Array (N, m).

    Returns
    -------
    jnp.ndarray
        Outer products (N, n, m) with ``result[i] = x[i][:, None] @ y[i][None, :]``.
    """
    return vmap(lambda a, b: jnp.atleast_2d(a).T @ jnp.atleast_2d(b), 0, 0)(x, y)


# =============================================================================
# UKF Predict & Update Steps
# =============================================================================


def predict_step(
    state: UKFState,
    u_imu: jnp.ndarray,
    dt_imu: float,
    config: UKFConfig,
    has_vision: bool = True,
    *,
    layout: StateLayout,
) -> UKFState:
    """UKF prediction step using IMU via unscented transform.

    Parameters
    ----------
    state : UKFState
        Current state.
    u_imu : jnp.ndarray
        IMU input [ω_z(rad/s), f_x(m/s^2), f_y(m/s^2)] (3,).
    dt_imu : float
        IMU timestep (s).
    config : UKFConfig
        UKF configuration.
    has_vision : bool, default True
        Whether vision is available at the current frame (for Q scaling).
    layout : StateLayout
        State index mapping.

    Returns
    -------
    UKFState
        Predicted state with mean and covariance.
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
        return dynamics_function(x, u_imu, dt_imu, config.damping_coeff, layout)

    sigmas_prop = vmap(f)(sigmas)  # (17, 8)

    # Reconstruct predicted mean (weighted sum)
    m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)

    # Reconstruct predicted covariance
    # Cov = Σ w_cov[i] * (sigma_prop[i] - m_pred) * (sigma_prop[i] - m_pred)^T
    deviations = sigmas_prop - m_pred
    P_pred = jnp.tensordot(w_cov, _outer_product_batch(deviations, deviations), axes=1)

    # Add process noise Q using shared assembly for parity with EKF/smoothers
    dtype = m.dtype
    h_idx = get_heading_index(layout)
    Q = assemble_Q(
        config,
        theta=m[h_idx],
        dt=dt_imu,
        n=n,
        has_vision=has_vision,
        dtype=dtype,
    )

    P_pred = P_pred + Q

    # Symmetrize for numerical stability
    P_pred = symmetrize(P_pred)

    return UKFState(mean=m_pred, cov=P_pred)


# =============================================================================
# Update Step Helpers
# =============================================================================


def update_step(
    state: UKFState,
    camera_model: CameraPositionModel,
    frame_idx: int,
    observation_is_valid: bool,
    config: UKFConfig,
) -> tuple[UKFState, float]:
    """UKF measurement update using camera model.

    Parameters
    ----------
    state : UKFState
        Predicted state.
    camera_model : CameraPositionModel
        Camera position measurement model with preallocated frame data.
    frame_idx : int
        Current frame index.
    observation_is_valid : bool
        Observation validity flag.
    config : UKFConfig
        UKF configuration.

    Returns
    -------
    tuple[UKFState, float]
        Updated state and log-likelihood.

    Notes
    -----
    Uses the MeasurementModel protocol and generic ukf_projected_update() primitive
    to handle 4D→2D projection for single-LED observations.
    """
    m_pred, P_pred = state.mean, state.cov

    # If no valid observation, return prediction unchanged
    def no_update(m, P):
        return UKFState(mean=m, cov=P), 0.0

    # If valid observation, perform update
    def do_update(m, P):
        # Get LED validity from camera model
        both_leds, only_led1, only_led2, _ = camera_model.subspace(frame_idx)
        led1_valid = both_leds | only_led1
        led2_valid = both_leds | only_led2

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

            # Transform sigma points through camera model prediction
            sigmas_meas = vmap(camera_model.predict)(sigmas)  # (17, 4)

            # Reconstruct predicted observation
            z_pred = jnp.tensordot(w_mean, sigmas_meas, axes=1)

            # Compute innovation covariance S
            meas_deviations = sigmas_meas - z_pred
            S = jnp.tensordot(w_cov, _outer_product_batch(meas_deviations, meas_deviations), axes=1)

            # Add measurement noise R from camera model (confidence-scaled)
            R = camera_model.meas_cov(frame_idx)
            S = S + R

            # Compute cross-covariance between state and observations
            state_deviations = sigmas - m_in
            P_cross = jnp.tensordot(
                w_cov, _outer_product_batch(state_deviations, meas_deviations), axes=1
            )

            # Get innovation from camera model (handles NaN → zero residual)
            innovation = camera_model.innovation(frame_idx, z_pred)

            # Call generic UKF projected update primitive
            state_upd, nis, log_lik = ukf_projected_update(
                UKFState(mean=m_in, cov=P_in),
                innovation,
                S,
                P_cross,
                both_leds,
                only_led1,
                only_led2,
            )

            # Mahalanobis gating
            def apply_gating():
                """Apply Mahalanobis gating to reject outliers."""
                dof = lax.cond(both_leds, lambda: 4, lambda: 2)
                threshold = chi2_threshold(dof, config.mahalanobis_threshold_prob)

                def accept():
                    return state_upd, log_lik

                def reject():
                    return UKFState(mean=m_in, cov=P_in), 0.0

                nis_safe = jnp.where(jnp.isfinite(nis), nis, jnp.inf)
                return lax.cond(nis_safe < threshold, accept, reject)

            def skip_gating():
                return state_upd, log_lik

            return lax.cond(config.use_mahalanobis_gating, apply_gating, skip_gating)

        # Conditional update based on LED availability
        return lax.cond(led1_valid | led2_valid, do_leds_update, no_leds_update, m, P)

    # Conditional update based on validity flag
    return lax.cond(observation_is_valid, do_update, no_update, m_pred, P_pred)


def update_heading(
    state: UKFState,
    heading_model: HeadingPseudoModel,
    frame_idx: int,
    observation_is_valid: bool,
    config: UKFConfig,
    *,
    layout: StateLayout,
) -> tuple[UKFState, float]:
    """Apply 1D heading pseudo-measurement update using heading model.

    Parameters
    ----------
    state : UKFState
        Current state (after position update).
    heading_model : HeadingPseudoModel
        Heading measurement model with preallocated frame data.
    frame_idx : int
        Current frame index.
    observation_is_valid : bool
        Camera validity flag (False skips update entirely).
    config : UKFConfig
        UKF configuration (for UKF parameters alpha, beta, kappa).
    layout : StateLayout
        State index mapping.

    Returns
    -------
    tuple[UKFState, float]
        Updated state and heading measurement log-likelihood (scalar).
    """
    observation_flag = jnp.asarray(observation_is_valid, dtype=bool)

    def no_update(state_in: UKFState) -> tuple[UKFState, jnp.ndarray]:
        zero = jnp.array(0.0, dtype=state_in.mean.dtype)
        return state_in, zero

    def do_update(state_in: UKFState) -> tuple[UKFState, jnp.ndarray]:
        m, P = state_in.mean, state_in.cov

        # Get measurement covariance from model (already gates invalid)
        R_mat = heading_model.meas_cov(frame_idx)  # (1, 1)
        R_heading = R_mat[0, 0]

        # 1D unscented heading update
        n = len(m)
        lamb = config.alpha**2 * (n + config.kappa) - n
        w_mean, w_cov = compute_weights(n, config.alpha, config.beta, lamb)

        # Generate sigma points
        sigmas = compute_sigma_points(m, P, n, lamb)

        # Transform sigma points through heading model
        h_idx = get_heading_index(layout)
        sigmas_heading = sigmas[:, h_idx]  # (2n+1,)

        # Predicted heading
        h_pred = jnp.dot(w_mean, sigmas_heading)

        # Get innovation from model (already angle-wrapped)
        innovation_vec = heading_model.innovation(frame_idx, jnp.array([h_pred]))
        innov = innovation_vec[0]

        # Innovation covariance (1D)
        heading_deviations = sigmas_heading - h_pred
        S = jnp.dot(w_cov, heading_deviations**2) + R_heading

        # Cross-covariance between state and heading measurement
        state_deviations = sigmas - m  # (2n+1, n)
        weighted_products = state_deviations * heading_deviations[:, None]  # (2n+1, n)
        P_cross = jnp.dot(w_cov, weighted_products)  # (n,)

        # Kalman gain (n,)
        K = P_cross / S

        # Update mean
        m_upd = m + K * innov

        # Wrap heading after update
        m_upd = m_upd.at[h_idx].set(wrap_angle(m_upd[h_idx]))

        # Update covariance
        P_upd = P - jnp.outer(K, K) * S
        P_upd = symmetrize(P_upd)

        # Log-likelihood
        log_lik = -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S) + innov**2 / S)

        # Zero out log-likelihood if R is huge (gated)
        use_heading = R_heading < 1e5
        log_lik = lax.select(use_heading, log_lik, jnp.array(0.0, dtype=log_lik.dtype))

        return UKFState(m_upd, P_upd), log_lik

    return lax.cond(observation_flag, do_update, no_update, state)


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
    """Run Unscented Kalman Filter on a full trajectory.

    Parameters
    ----------
    ukf_config : UKFConfig
        UKF configuration.
    t_imu : np.ndarray
        IMU timestamps (N_imu,) in seconds.
    U_imu : np.ndarray
        IMU measurements [ω_z(rad/s), f_x(m/s^2), f_y(m/s^2)] (N_imu, 3).
    t_cam : np.ndarray
        Camera timestamps (N_cam,) in seconds.
    Z_cam_led1 : np.ndarray
        LED1 positions (N_cam, 2) in meters.
    Z_cam_led2 : np.ndarray
        LED2 positions (N_cam, 2) in meters.
    mask_cam : np.ndarray
        Camera validity mask (N_cam,), boolean.
    initial_state : UKFState | None, optional
        Optional initial state (auto-initialized if None).
    conf_cam : np.ndarray | None, optional
        Confidence scores (N_cam, 4) for [x1,y1,x2,y2] in [0, 1] for per-dimension
        R scaling.

    Returns
    -------
    UKFResult
        Filtered and predicted states at camera times, and log-likelihood.
    """
    # Convert to JAX arrays
    t_imu_jax = jnp.array(t_imu)
    U_imu_jax = jnp.array(U_imu)
    t_cam_jax = jnp.array(t_cam)
    Z_cam_led1_jax = jnp.array(Z_cam_led1)
    Z_cam_led2_jax = jnp.array(Z_cam_led2)
    mask_cam_jax = jnp.array(mask_cam)
    # Precompute clipped confidences device-side for stable shapes
    conf_cam_jax = None if conf_cam is None else jnp.clip(jnp.array(conf_cam), 1e-2, 1.0)

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
            dt_cam=jnp.mean(jnp.diff(t_cam_jax)),  # Keep as JAX scalar for JIT compatibility
            led_distance=config_for_filter.led_distance,  # type: ignore[arg-type]
            layout=get_layout(config_for_filter.state_mode),
        )
        initial_state = UKFState(mean=ekf_init.mean, cov=ekf_init.cov)

    n_cam = len(t_cam)

    # Resolve state layout once for this run
    layout = get_layout(config_for_filter.state_mode)

    # Compute mean IMU timestep for fallback
    dt_imu_mean = jnp.mean(jnp.diff(t_imu_jax))  # Keep as JAX scalar for JIT compatibility

    # Precompute IMU index arrays (host-side, using shared utility)
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    # Instantiate measurement models with preallocated arrays
    camera_model = CameraPositionModel(
        led_distance=config_for_filter.led_distance,  # type: ignore[arg-type]
        measurement_noise_base=config_for_filter.measurement_noise_pos,
        layout=layout,
        z_led1_all=Z_cam_led1_jax,
        z_led2_all=Z_cam_led2_jax,
        conf_all=conf_cam_jax,
        confidence_clip_min=1e-2,
    )

    heading_model = HeadingPseudoModel(
        config=config_for_filter,
        layout=layout,
        z_led1_all=Z_cam_led1_jax,
        z_led2_all=Z_cam_led2_jax,
    )

    def filter_step(carry, t_idx):
        """Single filtering step at camera frame t_idx."""
        state_prev, log_lik_accum = carry

        # Check if we have vision at this timestep (for blackout-aware Q scaling)
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
                    return predict_step(s, u, dt, config_for_filter, has_vision_t, layout=layout)

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
            camera_model,
            t_idx,
            mask_cam_jax[t_idx],
            config_for_filter,
        )

        # Heading measurement update (sequential after position)
        # Only applied if use_heading_measurement=True (gated via large R otherwise)
        state_after_heading, log_lik_heading = update_heading(
            state_after_pos,
            heading_model,
            t_idx,
            mask_cam_jax[t_idx],
            config_for_filter,
            layout=layout,
        )

        # Zero-velocity update (reuse shared implementation for parity)
        state_after_zupt, log_lik_zupt = update_zupt(
            state_after_heading,
            config_for_filter,
        )
        state_filt = UKFState(
            mean=state_after_zupt.mean,
            cov=state_after_zupt.cov,
        )

        # Total log-likelihood (position + heading + ZUPT)
        log_lik_k = log_lik_pos + log_lik_heading + log_lik_zupt

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
