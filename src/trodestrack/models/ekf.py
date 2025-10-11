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
    - Mahalanobis gating for outlier rejection
    - RTS smoother for offline processing (see runtime/offline.py)

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
from jax import jacfwd, lax, vmap

from trodestrack.models.filter_common import (
    FilterCoreConfig,
    FilterState,
    apply_lifted_inverse,
    chi2_threshold,
    compute_nis_and_loglik,
    dynamics_function,
    initialize_state,
    joseph_update,
    measurement_function,
    prepare_heading_measurement,
    psd_solve,
    symmetrize,
    update_zupt,
    wrap_angle,
)
from trodestrack.models.filter_utils import compute_imu_index_arrays
from trodestrack.models.process_noise import assemble_Q
from trodestrack.models.state_layout import StateLayout, get_heading_index, get_layout

# =============================================================================
# Configuration & State
# =============================================================================


@dataclass
class EKFConfig(FilterCoreConfig):
    """EKF configuration extending the shared FilterCoreConfig."""

    num_iter: int = 1


EKFState = FilterState


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

    filtered_means: jnp.ndarray  # (N_cam, n)
    filtered_covariances: jnp.ndarray  # (N_cam, n, n)
    predicted_means: jnp.ndarray  # (N_cam, n)
    predicted_covariances: jnp.ndarray  # (N_cam, n, n)
    marginal_loglik: float
    estimated_led_distance: float | None


# =============================================================================
# Utility Functions
# =============================================================================
# NOTE: make_led_selector() and apply_lifted_inverse() are now imported
# from filter_common.py to share with UKF


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


def predict_step(
    state: EKFState,
    u_imu: jnp.ndarray,
    dt_imu: float,
    config: EKFConfig,
    has_vision: bool = True,
    *,
    layout: StateLayout,
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
        return dynamics_function(x, u_imu, dt_imu, config.damping_coeff, layout)

    # Jacobian
    F = jacfwd(f)
    F_x = F(m)

    # Predict mean
    m_pred = f(m)

    # Wrap heading angle to (-π, π] to prevent numerical issues
    h_idx = get_heading_index(layout)
    m_pred = m_pred.at[h_idx].set(wrap_angle(m_pred[h_idx]))

    # Assemble process noise using shared helper
    Q = assemble_Q(
        config,
        theta=m[h_idx],
        dt=dt_imu,
        n=m.shape[0],
        has_vision=has_vision,
        dtype=m.dtype,
    )

    # Predict covariance
    P_pred = F_x @ P @ F_x.T + Q
    P_pred = symmetrize(P_pred)

    return EKFState(mean=m_pred, cov=P_pred)


# =============================================================================
# Update Step Helpers
# =============================================================================


def _prepare_camera_observations(
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    pre_z_obs_full: jnp.ndarray | None,
    pre_led1_valid: bool | None,
    pre_led2_valid: bool | None,
) -> tuple[jnp.ndarray, jnp.ndarray, bool, bool]:
    """Prepare camera observations for update step.

    Checks LED validity and constructs observation vector and mask.

    Args:
        z_led1: LED1 observation [x, y]
        z_led2: LED2 observation [x, y]
        pre_z_obs_full: Precomputed concatenated observations (optional)
        pre_led1_valid: Precomputed LED1 validity (optional)
        pre_led2_valid: Precomputed LED2 validity (optional)

    Returns:
        Tuple of (z_obs_full, obs_mask, led1_valid, led2_valid)
            - z_obs_full: (4,) concatenated observations
            - obs_mask: (4,) validity mask
            - led1_valid: LED1 validity flag
            - led2_valid: LED2 validity flag
    """
    # Check which LEDs are valid (use precomputed if provided)
    led1_valid = pre_led1_valid if pre_led1_valid is not None else jnp.isfinite(z_led1[0])
    led2_valid = pre_led2_valid if pre_led2_valid is not None else jnp.isfinite(z_led2[0])

    # Build observation vector (use precomputed if provided)
    z_obs_full = pre_z_obs_full if pre_z_obs_full is not None else jnp.concatenate([z_led1, z_led2])

    # Construct observation mask
    obs_mask = jnp.array([led1_valid, led1_valid, led2_valid, led2_valid])

    return z_obs_full, obs_mask, led1_valid, led2_valid


def _compute_lifted_joseph_covariance(
    P_iter: jnp.ndarray,
    H4: jnp.ndarray,
    S4: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> jnp.ndarray:
    """Compute Joseph-form covariance update using lifted subspace operator.

    Uses the alternative formulation:
        P⁺ = P - PH^T S^{-1} HP

    which is equivalent to the standard Joseph form but works naturally with
    the lifted subspace operator.

    Args:
        P_iter: Current covariance (n, n)
        H4: Measurement Jacobian (4, n)
        S4: Innovation covariance (4, 4)
        both_leds: Both LEDs valid flag
        only_led1: Only LED1 valid flag
        only_led2: Only LED2 valid flag

    Returns:
        Updated covariance (n, n)
    """
    HP = H4 @ P_iter  # (4, n)
    PH_t = P_iter @ H4.T  # (n, 4)

    # Vectorized application of lifted inverse to columns of HP
    # Uses vmap for cleaner, more JAX-idiomatic code
    def apply_inv_to_col(col: jnp.ndarray) -> jnp.ndarray:
        """Apply S_eff⁻¹ to a column vector."""
        return apply_lifted_inverse(S4, col, both_leds, only_led1, only_led2)

    # Vectorize over columns (axis 1)
    inv_S_HP = vmap(apply_inv_to_col, in_axes=1, out_axes=1)(HP)  # (4, n)

    # Complete Joseph form
    PH_t_inv_S_HP = PH_t @ inv_S_HP  # (n, n)
    P_upd = P_iter - PH_t_inv_S_HP
    P_upd = symmetrize(P_upd)

    return P_upd


def update_step(
    state: EKFState,
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    mask: bool,
    config: EKFConfig,
    confidence: jnp.ndarray | None = None,
    *,
    pre_z_obs_full: jnp.ndarray | None = None,
    pre_conf: jnp.ndarray | None = None,
    pre_led1_valid: bool | None = None,
    pre_led2_valid: bool | None = None,
    layout: StateLayout,
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

    # Confidence→R scaling via shared helper
    from trodestrack.models.filter_common import confidence_to_R_diagonal

    # If no valid observation, return prediction unchanged with zero log-likelihood
    def no_update(m, P):
        return EKFState(mean=m, cov=P), 0.0

    # If valid observation, perform update
    def do_update(m, P):
        # Prepare camera observations using helper
        z_obs_full, obs_mask, led1_valid, led2_valid = _prepare_camera_observations(
            z_led1, z_led2, pre_z_obs_full, pre_led1_valid, pre_led2_valid
        )

        # If no valid observations, return prediction with zero log-likelihood
        def no_leds_update(m_in, P_in):
            return EKFState(mean=m_in, cov=P_in), 0.0

        # If at least one LED valid, perform update (with optional IEKF)
        def do_leds_update(m_in, P_in):
            # Measurement function
            def h(x):
                return measurement_function(x, config.led_distance, layout)

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
                H4 = H(m_iter)  # (4, n)
                z_pred_4 = h(m_iter)  # (4,)

                # Innovation in full 4D space
                # Zero out invalid LED components to avoid NaN propagation
                innov_4_raw = z_obs_full - z_pred_4  # (4,)
                innov_4 = jnp.where(obs_mask, innov_4_raw, 0.0)  # Zero invalid components

                # Confidence-scaled measurement noise
                # R_i = R_base / conf_i for each dimension (shared helper)
                conf_arg = pre_conf if pre_conf is not None else confidence
                R_diag = confidence_to_R_diagonal(
                    conf_arg, base=config.measurement_noise_pos, size=4
                )
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
                PH_t = P_iter @ H4.T  # (n, 4)
                delta_x = PH_t @ v  # (n,)

                # Update mean
                m_upd = m_iter + delta_x

                # Wrap heading angle to (-π, π] after update
                h_idx_local = get_heading_index(get_layout(config.state_mode))
                m_upd = m_upd.at[h_idx_local].set(wrap_angle(m_upd[h_idx_local]))

                # Joseph form covariance update using helper (vectorized with vmap)
                P_upd = _compute_lifted_joseph_covariance(
                    P_iter, H4, S4, both_leds, only_led1, only_led2
                )

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
    mask: bool,
) -> tuple[EKFState, jnp.ndarray]:
    """Apply 1D heading pseudo-measurement update from LED pair.

    Sequential update after position update. Uses large-R gating pattern
    for JAX compatibility (no branching) and respects the camera mask to
    prevent updates during vision dropouts.

    Args:
        state: Current state (after position update)
        z_led1: LED1 observation (2,) in meters
        z_led2: LED2 observation (2,) in meters
        config: EKF configuration
        mask: Camera validity flag (False skips update entirely)

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
    mask_bool = jnp.asarray(mask, dtype=bool)

    def no_update(state_in: EKFState) -> tuple[EKFState, jnp.ndarray]:
        return state_in, jnp.array(0.0, dtype=state_in.mean.dtype)

    def do_update(state_in: EKFState) -> tuple[EKFState, jnp.ndarray]:
        m, P = state_in.mean, state_in.cov

        # Prepare heading measurement (shared preprocessing with UKF)
        heading_obs, R_heading, use_heading = prepare_heading_measurement(z_led1, z_led2, config)

        # 1D heading update
        # Measurement function: h(x) = x[h_idx] (heading)
        h_idx = get_heading_index(get_layout(config.state_mode))
        h_pred = m[h_idx]

        # Innovation with angle wrapping (replace NaN with 0 for gated case)
        innov_raw = wrap_angle(heading_obs - h_pred)
        innov = jnp.where(jnp.isfinite(innov_raw), innov_raw, 0.0)

        # Jacobian row for heading index
        n = m.shape[0]
        H = jnp.zeros((1, n))
        H = H.at[0, h_idx].set(1.0)

        # Innovation covariance (scalar, but treat as 1x1 matrix)
        S = H @ P @ H.T + jnp.array([[R_heading]])

        # Kalman gain (n, 1)
        K = psd_solve(S, H @ P).T

        # Mean update
        m_upd = m + (K @ jnp.array([[innov]])).ravel()
        m_upd = m_upd.at[h_idx].set(wrap_angle(m_upd[h_idx]))  # Wrap after update

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

    return lax.cond(mask_bool, do_update, no_update, state)


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
    conf_cam: np.ndarray | None = None,
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
        conf_cam: Camera confidence scores (N_cam, 4) for [led1_x, led1_y, led2_x, led2_y]
            Range: [0, 1], where 1.0 = high confidence
            If None, defaults to 1.0 (high confidence, backward compatible)
            Measurement noise is scaled as: R_eff = R_base / clip(conf, min, 1.0)
            PRD requirement: "DLC confidence → measurement noise scaling" (Section 13)

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
    # Precompute clipped confidences device-side for stable shapes
    conf_cam_jax = None if conf_cam is None else jnp.clip(jnp.array(conf_cam), 1e-2, 1.0)

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
            dt_cam=jnp.mean(jnp.diff(t_cam_jax)),  # Keep as JAX scalar for JIT compatibility
            led_distance=config_for_filter.led_distance,  # type: ignore[arg-type]
            layout=get_layout(config_for_filter.state_mode),
        )

    n_cam = len(t_cam)

    # Resolve state layout once for this run
    layout = get_layout(config_for_filter.state_mode)

    # Compute mean IMU timestep for fallback when imu_idx == 0
    dt_imu_mean = jnp.mean(jnp.diff(t_imu_jax))  # Keep as JAX scalar for JIT compatibility

    # Precompute IMU index arrays (host-side, using shared utility)
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    # Precompute device-friendly measurement inputs per frame
    led1_valid_arr = jnp.isfinite(Z_cam_led1_jax[:, 0])
    led2_valid_arr = jnp.isfinite(Z_cam_led2_jax[:, 0])
    z_obs_full_arr = jnp.concatenate([Z_cam_led1_jax, Z_cam_led2_jax], axis=1)  # (N_cam, 4)
    conf4_arr = None if conf_cam_jax is None else conf_cam_jax  # already clipped if provided

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
            Z_cam_led1_jax[t_idx],
            Z_cam_led2_jax[t_idx],
            mask_cam_jax[t_idx],
            config_for_filter,
            None if conf_cam_jax is None else conf_cam_jax[t_idx],
            pre_z_obs_full=z_obs_full_arr[t_idx],
            pre_conf=None if conf4_arr is None else conf4_arr[t_idx],
            pre_led1_valid=led1_valid_arr[t_idx],
            pre_led2_valid=led2_valid_arr[t_idx],
            layout=layout,
        )

        # Heading measurement update (sequential after position)
        # Only applied if use_heading_measurement=True (gated via large R otherwise)
        state_after_heading, log_lik_heading = update_heading(
            state_after_pos,
            Z_cam_led1_jax[t_idx],
            Z_cam_led2_jax[t_idx],
            config_for_filter,
            mask_cam_jax[t_idx],
        )

        # Zero-velocity update (ZUPT) for stationary detection (sequential after heading)
        # Only applied if enable_zupt=True and velocity < threshold (gated via large R otherwise)
        state_filt, log_lik_zupt = update_zupt(
            state_after_heading,
            config_for_filter,
        )

        # Total log-likelihood for this frame
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

    return EKFResult(
        filtered_means=outputs["filtered_mean"],
        filtered_covariances=outputs["filtered_cov"],
        predicted_means=outputs["predicted_mean"],
        predicted_covariances=outputs["predicted_cov"],
        marginal_loglik=float(log_lik_total),
        estimated_led_distance=estimated_led_distance,
    )
