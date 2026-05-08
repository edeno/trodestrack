"""Unscented Kalman Filter (UKF) for sensor-fused rat tracking.

The UKF uses sigma-point transforms to handle nonlinearity without
Jacobians. It propagates (2n+1) sigma points through nonlinear dynamics
and measurements, where ``n`` is the state dimension selected by
``UKFConfig.state_mode`` (resolved via ``StateLayout``). For example,
the legacy 8D ``2d_full`` layout uses 17 sigma points; the default 10D
``2d_cam_3d_imu`` layout uses 21. Quaternion-orientation layouts are
rejected by the UKF.

Key advantages over EKF:
    - No Jacobian computation required
    - Better handling of strong nonlinearities
    - 2nd-order accuracy for Gaussian distributions

Key differences from EKF:
    - Uses unscented transform instead of linearization
    - Propagates sigma points through actual nonlinear functions
    - Slightly higher computational cost but often more accurate

References:
    - Särkkä (2013) "Bayesian Filtering and Smoothing", Algorithm 5.14
    - Julier & Uhlmann (1997) "New extension of the Kalman filter"
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax, tree_util, vmap

from trodestrack.models.filter_common import (
    FilterCoreConfig,
    FilterState,
    camera_stationary_zupt_gate_2d,
    chi2_threshold,
    compute_imu_index_arrays,
    dynamics_function,
    estimate_led_spacing,
    imu_stationary_zupt_gate,
    initialize_state,
    symmetrize,
    update_zupt,
    update_zupt_visual_context,
    validate_camera_input_shapes,
    validate_imu_input_shape,
    validate_initial_state,
    validate_timestamps,
    wrap_angle,
)
from trodestrack.models.filter_update import ukf_projected_update
from trodestrack.models.process_noise import assemble_Q
from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.sensors.event_location import (
    EventLocationModel,
    resolve_event_inputs,
    update_event_location,
)
from trodestrack.models.sensors.heading_pseudo import HeadingPseudoModel
from trodestrack.models.state_layout import StateLayout, get_heading_index, get_layout

# =============================================================================
# Configuration & State
# =============================================================================


@dataclass(frozen=True)
class UKFConfig(FilterCoreConfig):
    """Unscented Kalman filter configuration extending FilterCoreConfig.

    Parameters
    ----------
    alpha : float, default 1.732
        Sigma-point spread parameter. Larger (≈√3) captures stronger
        nonlinearities; smaller spreads concentrate the sigma points near
        the mean, but values that drive ``(n + λ) = α²(n + κ)`` below
        ``_MIN_N_PLUS_LAMBDA = 1e-2`` are rejected by ``__post_init__``
        because the central weight ``w_mean[0] = λ/(n+λ)`` blows up and
        covariance reconstruction loses precision. With ``kappa=0`` and
        ``n=10``, for example, ``alpha=1e-3`` resolves to
        ``(n+λ) ≈ 1e-5`` and is rejected.
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

    # Floor on (n + λ) = α²(n + κ). Below this, sigma-point weights
    # w_mean[0] = λ/(n+λ) exceed ≈100 in magnitude and covariance
    # reconstruction loses precision to catastrophic cancellation. Example:
    # with α=1e-3, κ=0, n=10, (n+λ) ≈ 1e-5 and w_mean[0] ≈ -1e6.
    _MIN_N_PLUS_LAMBDA: ClassVar[float] = 1e-2

    def __post_init__(self) -> None:
        """Validate UKF scaling parameters don't produce degenerate weights."""
        # Parent-class validation (mahalanobis_threshold_prob etc.)
        super().__post_init__()

        # Reject non-finite scaling parameters before computing (n + λ): a
        # NaN here passes the < _MIN_N_PLUS_LAMBDA inequality below and
        # propagates through compute_weights into every sigma-point
        # update, returning NaN means / loglik with no other warning.
        # alpha must additionally be > 0 (it appears squared but a 0
        # collapses (n + λ) to -n, yielding negative weights everywhere).
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(
                "UKFConfig.alpha must be a finite strictly-positive value; "
                f"got {self.alpha!r}."
            )
        if not np.isfinite(self.beta):
            raise ValueError(
                "UKFConfig.beta must be a finite value (typically 2.0 for "
                f"Gaussian priors); got {self.beta!r}."
            )
        if not np.isfinite(self.kappa):
            raise ValueError(
                "UKFConfig.kappa must be a finite secondary-scaling value; "
                f"got {self.kappa!r}."
            )

        # Need the layout dimension to check (n + λ); imported here to avoid a
        # top-level cycle (state_layout is pulled in by the shared config).
        from trodestrack.models.state_layout import get_layout

        layout = get_layout(self.state_mode)
        n = layout.n
        n_plus_lambda = self.alpha**2 * (n + self.kappa)
        if n_plus_lambda < self._MIN_N_PLUS_LAMBDA:
            # The diagnostic message references (n + κ) and (n + λ) in
            # denominators. Guard those: kappa <= -n hits the degenerate-
            # weight gate too, but a literal division would raise
            # ZeroDivisionError before reaching the actual ValueError. In
            # that case ``w_mean[0]`` blows up to ±∞ rather than a finite
            # number, and the only fix is to pick kappa > -n.
            if (n + self.kappa) <= 0:
                raise ValueError(
                    "UKFConfig would produce degenerate sigma-point weights "
                    f"for state_mode='{self.state_mode}' (n={n}): "
                    f"(n + κ) = {n + self.kappa!r} ≤ 0, so "
                    f"(n + λ) = α²(n + κ) = {n_plus_lambda!r} is also "
                    "non-positive and the central weight w_mean[0] = "
                    "λ/(n+λ) is unbounded. Use kappa > "
                    f"{-n} (or raise it past -n) to keep the unscented "
                    "weights well defined."
                )
            min_alpha = (self._MIN_N_PLUS_LAMBDA / (n + self.kappa)) ** 0.5
            raise ValueError(
                f"UKFConfig would produce degenerate sigma-point weights for "
                f"state_mode='{self.state_mode}' (n={n}): "
                f"(n + λ) = α²(n + κ) = {n_plus_lambda:.3e} < "
                f"{self._MIN_N_PLUS_LAMBDA:.0e}. With these values, "
                f"w_mean[0] = λ/(n+λ) ≈ {-n / n_plus_lambda:.1e}, which causes "
                f"catastrophic cancellation when reconstructing covariance. "
                f"Use alpha ≥ {min_alpha:.3f} (or raise kappa)."
            )

    @classmethod
    def conservative(cls, **kwargs) -> UKFConfig:
        """Conservative UKF preset with moderate spread for numerical stability.

        Uses alpha=0.5 and kappa=0, giving (n + λ) = 0.25·n. Sigma points stay
        closer to the mean than the default, which dampens nonlinearity
        distortion, while keeping weights at O(1) magnitude (w_mean[0] ≈ -3 at
        n=10) so covariance reconstruction stays stable.

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
            Preset with alpha=0.5, beta=2.0, kappa=0.0.

        Example:
            >>> config = UKFConfig.conservative(use_mahalanobis_gating=True)

        Note
        ----
        Prior versions of this preset used alpha=1e-3 which produces extreme
        weights (O(10⁶)) at state dimension n=10 and is rejected at runtime by
        the __post_init__ guard.
        """
        return cls(alpha=0.5, beta=2.0, kappa=0.0, **kwargs)

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


tree_util.register_pytree_node_class(UKFConfig)


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
    usable_vision_mask : jnp.ndarray | None
        Boolean mask of camera frames with at least one finite LED observation.
    """

    filtered_means: jnp.ndarray  # (N_cam, n)
    filtered_covariances: jnp.ndarray  # (N_cam, n, n)
    predicted_means: jnp.ndarray  # (N_cam, n)
    predicted_covariances: jnp.ndarray  # (N_cam, n, n)
    marginal_loglik: float
    estimated_led_distance: float | None
    usable_vision_mask: jnp.ndarray | None = None


# =============================================================================
# Sigma-Point Generation
# =============================================================================


def compute_sigma_points(
    mean: jnp.ndarray, cov: jnp.ndarray, n: int, lamb: float
) -> jnp.ndarray:
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

    Notes
    -----
    Jitter scales with average diagonal (trace/n) to handle covariances
    with varying scales. Fixed 1e-9 jitter can fail for tight/degenerate
    covariances in real animal motion data.
    """
    # Compute Cholesky decomposition: P = L @ L.T
    # Scale jitter by average diagonal for numerical stability
    # This prevents failures with tight covariances (e.g., high-confidence position updates)
    avg_variance = jnp.trace(cov) / n
    eps = 1e-8 * avg_variance
    # Ensure minimum jitter for numerical stability even with tiny covariances
    eps = jnp.maximum(eps, jnp.asarray(1e-12, dtype=cov.dtype))
    cov_reg = symmetrize(cov) + eps * jnp.eye(n, dtype=cov.dtype)
    chol = jnp.linalg.cholesky(cov_reg)
    # Scale by sqrt(n + lambda) for sigma-point spread
    spread = jnp.sqrt(n + lamb)
    distances = spread * chol  # (n, n)

    # Generate sigma points via broadcasting (vectorized)
    # sigma_pm = [+distances columns, -distances columns] (n, 2n)
    sigma_pm = jnp.concatenate((distances, -distances), axis=1)
    # Broadcast mean across all 2n sigma points: mean[:, None] + sigma_pm
    sigmas = jnp.concatenate(
        (mean[:, None], mean[:, None] + sigma_pm), axis=1
    ).T  # (2n+1, n)

    return sigmas


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
    n = len(
        m
    )  # State dimension, resolved from `layout` (8 for 2d_full, 10 for 2d_cam_3d_imu, etc.)

    # Compute UKF parameters
    lamb = config.alpha**2 * (n + config.kappa) - n
    w_mean, w_cov = compute_weights(n, config.alpha, config.beta, lamb)

    # Generate sigma points (2n+1 — 17 for 8D 2d_full, 21 for the default
    # 10D 2d_cam_3d_imu, etc.; n is resolved from the active StateLayout).
    sigmas = compute_sigma_points(m, P, n, lamb)

    # Propagate sigma points through dynamics
    def f(x):
        return dynamics_function(
            x,
            u_imu,
            dt_imu,
            config.damping_coeff,
            layout,
            gravity_body=config.imu_gravity_body,
            enable_experimental_accel_translation=(
                config.enable_experimental_accel_translation
            ),
        )

    sigmas_prop = vmap(f)(sigmas)  # (17, 8)

    # Reconstruct predicted mean (weighted sum)
    # Note: For heading dimension, we need circular mean, but it's handled below
    m_pred = jnp.tensordot(w_mean, sigmas_prop, axes=1)

    # Fix heading component using circular mean
    # Arithmetic mean fails for angles when sigma points straddle 0°/360°
    h_idx = get_heading_index(layout)
    sigmas_heading_prop = sigmas_prop[:, h_idx]
    sin_weighted = w_mean @ jnp.sin(sigmas_heading_prop)
    cos_weighted = w_mean @ jnp.cos(sigmas_heading_prop)
    m_pred = m_pred.at[h_idx].set(jnp.arctan2(sin_weighted, cos_weighted))

    # Reconstruct predicted covariance
    # Cov = Σ w_cov[i] * (sigma_prop[i] - m_pred) * (sigma_prop[i] - m_pred)^T
    deviations = sigmas_prop - m_pred

    # Wrap heading deviations to [-π, π] for correct covariance computation
    # Without this, heading deviations can be 358° instead of 2°
    heading_deviations = wrap_angle(deviations[:, h_idx])
    deviations = deviations.at[:, h_idx].set(heading_deviations)

    P_pred = jnp.tensordot(w_cov, _outer_product_batch(deviations, deviations), axes=1)

    # Add process noise Q using shared assembly for parity with EKF/smoothers
    # Use predicted heading θ⁺ for tighter alignment between dynamics and Q
    dtype = m.dtype
    Q = assemble_Q(
        config,
        theta=m_pred[h_idx],  # Use predicted heading, not current
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
            S = jnp.tensordot(
                w_cov, _outer_product_batch(meas_deviations, meas_deviations), axes=1
            )

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

            # S is the full 4×4 covariance before projection; ukf_projected_update
            # handles exact 2D statistics for single-LED cases.
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
        use_heading = heading_model.use_measurement(frame_idx)

        def apply_update(state_valid: UKFState) -> tuple[UKFState, jnp.ndarray]:
            m, P = state_valid.mean, state_valid.cov

            # Get measurement covariance from model
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

            # Predicted heading using circular mean (not arithmetic mean!)
            # For angles, we must use: atan2(Σ w_i sin(θ_i), Σ w_i cos(θ_i))
            # Arithmetic mean fails when sigma points straddle 0°/360°
            sin_weighted = w_mean @ jnp.sin(sigmas_heading)
            cos_weighted = w_mean @ jnp.cos(sigmas_heading)
            h_pred = jnp.arctan2(sin_weighted, cos_weighted)

            # Get innovation from model (already angle-wrapped)
            innovation_vec = heading_model.innovation(frame_idx, jnp.array([h_pred]))
            innov = innovation_vec[0]

            # Innovation covariance (1D)
            # Wrap heading deviations to (-π, π] for correct angular variance.
            # h_pred is the circular mean (atan2, always wrapped), but sigmas_heading
            # may be on the opposite side of the ±π boundary. Without this wrap, near
            # the wrap boundary the unwrapped deviations are ~2π instead of ~0 and
            # S is inflated by ~(2π)², collapsing the Kalman gain toward zero.
            # This mirrors the wrap applied in predict_step (see around line 362).
            heading_deviations = wrap_angle(sigmas_heading - h_pred)
            S = jnp.dot(w_cov, heading_deviations**2) + R_heading

            # Cross-covariance between state and heading measurement
            state_deviations = sigmas - m  # (2n+1, n)
            weighted_products = state_deviations * heading_deviations[:, None]
            P_cross = w_cov @ weighted_products  # (n,)

            # Kalman gain (n,)
            K = P_cross / S

            # Update mean
            m_upd = m + K * innov

            # Wrap heading after update
            m_upd = m_upd.at[h_idx].set(wrap_angle(m_upd[h_idx]))

            # Update covariance using explicit Joseph form for the scalar heading
            # measurement. The measurement Jacobian is H = e_h^T.
            n_state = P.shape[0]
            I_minus_KH = jnp.eye(n_state, dtype=P.dtype).at[:, h_idx].add(-K)
            P_upd = I_minus_KH @ P @ I_minus_KH.T + jnp.outer(K, K) * R_heading
            P_upd = symmetrize(P_upd)

            # Log-likelihood
            log_lik = -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S) + innov**2 / S)

            return UKFState(m_upd, P_upd), log_lik

        return lax.cond(use_heading, apply_update, no_update, state_in)

    return lax.cond(observation_flag, do_update, no_update, state)


# =============================================================================
# JIT Configuration (mirror EKF pattern)
# =============================================================================

UNSCENTED_KALMAN_FILTER_STATIC_ARGNAMES = (
    "layout",
    "config_for_filter",
    "max_events_per_frame",
)
# Buffer donation not beneficial: input shapes (N_cam, 2) don't match output shapes (N_cam, n)
# XLA cannot reuse donated buffers when shapes differ. Donation only helps when input
# buffers can be reused for outputs of matching shape/dtype.
UNSCENTED_KALMAN_FILTER_DONATE_ARGNUMS: tuple[int, ...] = ()


def _unscented_kalman_filter_impl(
    initial_state: UKFState,
    initial_zupt_context: jnp.ndarray,
    t_imu_jax: jnp.ndarray,
    U_imu_jax: jnp.ndarray,
    t_cam_jax: jnp.ndarray,
    Z_cam_led1_jax: jnp.ndarray,
    Z_cam_led2_jax: jnp.ndarray,
    mask_cam_jax: jnp.ndarray,
    conf_cam_jax: jnp.ndarray | None,
    event_source_anchors_jax: jnp.ndarray,
    event_source_covariances_jax: jnp.ndarray,
    event_indices_per_frame_jax: jnp.ndarray,
    imu_index_arrays: jnp.ndarray,
    dt_imu_mean: jnp.ndarray,
    *,
    config_for_filter: UKFConfig,
    layout: StateLayout,
    max_events_per_frame: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Core UKF implementation staged under ``jax.jit``."""
    n_cam = int(t_cam_jax.shape[0])

    # led_distance guaranteed non-None by caller assert (see unscented_kalman_filter)
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

    event_model = EventLocationModel(
        source_anchors=event_source_anchors_jax,
        source_covariances=event_source_covariances_jax,
        layout=layout,
        max_events_per_frame=max_events_per_frame,
        dtype=initial_state.mean.dtype,
    )

    def filter_step(carry, t_idx):
        """Single filtering step at camera frame t_idx."""
        (
            state_prev,
            log_lik_accum,
            has_seen_vision_prev,
            stationary_context_prev,
            stationary_context_age_prev,
        ) = carry

        # Check if this frame has usable vision for blackout-aware Q scaling.
        both_leds, only_led1, only_led2, _ = camera_model.subspace(t_idx)
        frame_has_led = mask_cam_jax[t_idx] & (both_leds | only_led1 | only_led2)

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
                    return predict_step(
                        s, u, dt, config_for_filter, frame_has_led, layout=layout
                    )

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

        has_seen_vision_next = has_seen_vision_prev | frame_has_led
        zupt_imu_indices = imu_index_arrays[t_idx]
        zupt_imu_samples = U_imu_jax[jnp.maximum(zupt_imu_indices, 0)]
        zupt_imu_valid = zupt_imu_indices >= 0
        imu_stationary = imu_stationary_zupt_gate(
            zupt_imu_samples,
            zupt_imu_valid,
            config_for_filter,
            layout,
        )
        visual_speed_valid, visual_stationary = camera_stationary_zupt_gate_2d(
            t_cam_jax,
            Z_cam_led1_jax,
            Z_cam_led2_jax,
            mask_cam_jax,
            t_idx,
            config_for_filter,
        )
        stationary_context_next, stationary_context_age_next = (
            update_zupt_visual_context(
                visual_speed_valid,
                visual_stationary,
                stationary_context_prev,
                stationary_context_age_prev,
                config_for_filter,
            )
        )

        # Zero-velocity update (reuse shared implementation for parity)
        state_after_zupt, log_lik_zupt = update_zupt(
            state_after_heading,
            config_for_filter,
            active=has_seen_vision_next & imu_stationary & stationary_context_next,
        )

        event_source_indices = event_indices_per_frame_jax[t_idx]
        state_after_events, log_lik_event = update_event_location(
            state_after_zupt,
            event_model,
            event_source_indices,
        )
        state_filt = UKFState(
            mean=state_after_events.mean,
            cov=state_after_events.cov,
        )

        log_lik_k = log_lik_pos + log_lik_heading + log_lik_zupt + log_lik_event

        # Store outputs
        outputs = {
            "filtered_mean": state_filt.mean,
            "filtered_cov": state_filt.cov,
            "predicted_mean": state_pred.mean,
            "predicted_cov": state_pred.cov,
            "usable_vision": frame_has_led,
        }

        # Update carry with accumulated log-likelihood
        carry = (
            state_filt,
            log_lik_accum + log_lik_k,
            has_seen_vision_next,
            stationary_context_next,
            stationary_context_age_next,
        )

        return carry, outputs

    # Run filter over all camera frames
    carry_init = (
        initial_state,
        jnp.array(0.0, dtype=initial_state.mean.dtype),
        jnp.asarray(initial_zupt_context, dtype=bool),
        jnp.asarray(False, dtype=bool),
        jnp.asarray(
            config_for_filter.zupt_visual_context_hold_frames + 1,
            dtype=jnp.int32,
        ),
    )
    (_, log_lik_total, _, _, _), outputs = lax.scan(
        filter_step, carry_init, jnp.arange(n_cam)
    )

    return (
        outputs["filtered_mean"],
        outputs["filtered_cov"],
        outputs["predicted_mean"],
        outputs["predicted_cov"],
        log_lik_total,
        outputs["usable_vision"],
    )


_unscented_kalman_filter_jit = jax.jit(
    _unscented_kalman_filter_impl,
    static_argnames=UNSCENTED_KALMAN_FILTER_STATIC_ARGNAMES,
    donate_argnums=UNSCENTED_KALMAN_FILTER_DONATE_ARGNUMS,
)


# =============================================================================
# Public API
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
    event_source_anchors: np.ndarray | None = None,
    event_source_covariances: np.ndarray | None = None,
    event_indices_per_frame: np.ndarray | None = None,
) -> UKFResult:
    """Run Unscented Kalman Filter on a full trajectory.

    Parameters
    ----------
    ukf_config : UKFConfig
        UKF configuration.
    t_imu : np.ndarray
        IMU timestamps (N_imu,) in seconds.
    U_imu : np.ndarray
        IMU measurements ``(N_imu, K)``. The accepted channel count
        ``K`` is layout-dependent and enforced by
        :func:`trodestrack.models.filter_common.validate_imu_input_shape`:

        - ``K = 3``: ``[ω_z, f_x, f_y]`` for ``vision_only``, ``2d_full``,
          and ``2d_cam_3d_imu`` (degenerate, vz idle).
        - ``K = 4``: ``[ω_z, f_x, f_y, f_z]`` for ``2d_cam_3d_imu`` with
          3D vertical velocity active.

        The UKF does not accept quaternion-orientation layouts, so the
        6-channel form used by ``2d_cam_6dof_imu_orientation`` /
        ``3d_cam_6dof_imu`` is rejected here — use the EKF entry point
        for those modes. Units: rad/s for gyro components, m/s² for
        specific-force.
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
    layout = get_layout(ukf_config.state_mode)
    if layout.has_quaternion_orientation:
        raise NotImplementedError(
            "unscented_kalman_filter does not yet support quaternion-orientation "
            "state modes because sigma-point quaternion mean/covariance handling "
            "is not implemented. Use extended_kalman_filter for "
            "'2d_cam_6dof_imu_orientation'."
        )

    # Validate IMU input shape early so silent channel mismatches fail loudly.
    validate_imu_input_shape(
        U_imu,
        layout,
        t_imu=t_imu,
        func_name="unscented_kalman_filter",
    )

    # Reject non-finite / non-monotonic timestamps. Without this, NaN
    # entries propagate through compute_imu_index_arrays / dt and the
    # filter silently returns NaN means.
    validate_timestamps(
        t_imu, name="t_imu", func_name="unscented_kalman_filter", min_size=2
    )
    validate_timestamps(t_cam, name="t_cam", func_name="unscented_kalman_filter")

    # Validate camera-aligned arrays match len(t_cam). JAX out-of-bounds
    # indexing silently clamps, so a too-short Z_cam_led* / mask_cam /
    # conf_cam would otherwise reuse the last in-range row for every
    # later frame and the filter would return finite-but-wrong outputs.
    validate_camera_input_shapes(
        t_cam,
        Z_cam_led1,
        Z_cam_led2,
        mask_cam,
        conf_cam=conf_cam,
        func_name="unscented_kalman_filter",
    )

    # Convert to JAX arrays
    t_imu_jax = jnp.array(t_imu)
    U_imu_jax = jnp.array(U_imu)
    t_cam_jax = jnp.array(t_cam)
    Z_cam_led1_jax = jnp.array(Z_cam_led1)
    Z_cam_led2_jax = jnp.array(Z_cam_led2)
    mask_cam_jax = jnp.array(mask_cam)
    # Precompute clipped confidences device-side for stable shapes
    conf_cam_jax = (
        None if conf_cam is None else jnp.clip(jnp.array(conf_cam), 1e-2, 1.0)
    )

    # Auto-detect LED spacing if not specified
    # Store estimated value to return in result (immutability: do NOT mutate config)
    estimated_led_distance: float | None = None
    config_for_filter: UKFConfig

    if ukf_config.led_distance is None:
        estimated_led_distance = estimate_led_spacing(
            Z_cam_led1_jax, Z_cam_led2_jax, mask_cam_jax
        )
        # Create new config with estimated spacing (do NOT mutate original)
        config_for_filter = replace(ukf_config, led_distance=estimated_led_distance)
    else:
        # Use original config as-is
        config_for_filter = ukf_config

    # Initialize state (reuse EKF initialization)
    assert config_for_filter.led_distance is not None, (
        "led_distance must be set or auto-detected before filter"
    )
    initial_zupt_context = initial_state is not None
    if initial_state is None:
        ekf_init = initialize_state(
            Z_cam_led1_jax,
            Z_cam_led2_jax,
            mask_cam_jax,
            dt_cam=jnp.mean(
                jnp.diff(t_cam_jax)
            ),  # Keep as JAX scalar for JIT compatibility
            led_distance=config_for_filter.led_distance,
            layout=get_layout(config_for_filter.state_mode),
        )
        initial_state = UKFState(mean=ekf_init.mean, cov=ekf_init.cov)
    else:
        validate_initial_state(
            initial_state,
            get_layout(config_for_filter.state_mode),
            func_name="unscented_kalman_filter",
        )

    # Resolve state layout once for this run
    layout = get_layout(config_for_filter.state_mode)

    # Compute mean IMU timestep for fallback
    dt_imu_mean = jnp.mean(
        jnp.diff(t_imu_jax)
    )  # Keep as JAX scalar for JIT compatibility

    # Precompute IMU index arrays (host-side, using shared utility)
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    (
        event_source_anchors_jax,
        event_source_covariances_jax,
        event_indices_per_frame_jax,
        max_events_per_frame,
    ) = resolve_event_inputs(
        event_source_anchors,
        event_source_covariances,
        event_indices_per_frame,
        n_cam=int(t_cam_jax.shape[0]),
        func_name="unscented_kalman_filter",
    )

    # Call JIT-compiled implementation
    (
        filtered_means,
        filtered_covs,
        predicted_means,
        predicted_covs,
        log_lik_total,
        usable_vision_mask,
    ) = _unscented_kalman_filter_jit(
        initial_state,
        jnp.asarray(initial_zupt_context, dtype=bool),
        t_imu_jax,
        U_imu_jax,
        t_cam_jax,
        Z_cam_led1_jax,
        Z_cam_led2_jax,
        mask_cam_jax,
        conf_cam_jax,
        event_source_anchors_jax,
        event_source_covariances_jax,
        event_indices_per_frame_jax,
        imu_index_arrays,
        dt_imu_mean,
        config_for_filter=config_for_filter,
        layout=layout,
        max_events_per_frame=max_events_per_frame,
    )

    return UKFResult(
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        predicted_means=predicted_means,
        predicted_covariances=predicted_covs,
        marginal_loglik=float(log_lik_total),
        estimated_led_distance=estimated_led_distance,
        usable_vision_mask=usable_vision_mask,
    )
