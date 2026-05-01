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

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, lax, tree_util

from trodestrack.models.filter_common import (
    FilterCoreConfig,
    FilterState,
    chi2_threshold,
    compute_imu_index_arrays,
    dynamics_function,
    estimate_led_spacing,
    initialize_state,
    joseph_update,
    normalize_state_orientation,
    psd_solve,
    state_yaw,
    symmetrize,
    update_zupt,
    validate_imu_input_shape,
    wrap_angle,
)
from trodestrack.models.filter_update import ekf_projected_update
from trodestrack.models.process_noise import assemble_Q
from trodestrack.models.quaternion import rotate_vector_world_to_body
from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.sensors.heading_pseudo import (
    HEADING_GATE_THRESHOLD,
    HeadingPseudoModel,
)
from trodestrack.models.state_layout import StateLayout, get_heading_index, get_layout

# =============================================================================
# Configuration & State
# =============================================================================


@dataclass(frozen=True)
class EKFConfig(FilterCoreConfig):
    """EKF configuration extending the shared FilterCoreConfig.

    Notes
    -----
    Inherits all parameters from :class:`FilterCoreConfig`. Adds
    ``num_iter`` for iterated EKF (IEKF).

    Parameters
    ----------
    num_iter : int, default 1
        Number of inner IEKF iterations per measurement update.
    """

    num_iter: int = 1


tree_util.register_pytree_node_class(EKFConfig)


EKFState = FilterState


class EKFResult(NamedTuple):
    """EKF filtering result.

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


class EKFComputationResult(NamedTuple):
    """Internal EKF outputs produced by the JIT implementation."""

    filtered_means: jnp.ndarray
    filtered_covariances: jnp.ndarray
    predicted_means: jnp.ndarray
    predicted_covariances: jnp.ndarray
    marginal_loglik: jnp.ndarray


EXTENDED_KALMAN_FILTER_STATIC_ARGNAMES = ("layout", "config_for_filter")
# Buffer donation not beneficial: input shapes (N_cam, 2) don't match output shapes (N_cam, n)
# XLA cannot reuse donated buffers when shapes differ. Donation only helps when input
# buffers can be reused for outputs of matching shape/dtype.
EXTENDED_KALMAN_FILTER_DONATE_ARGNUMS: tuple[int, ...] = ()


def _extended_kalman_filter_impl(
    initial_state: EKFState,
    t_imu_jax: jnp.ndarray,
    U_imu_jax: jnp.ndarray,
    t_cam_jax: jnp.ndarray,
    Z_cam_led1_jax: jnp.ndarray,
    Z_cam_led2_jax: jnp.ndarray,
    mask_cam_jax: jnp.ndarray,
    conf_cam_jax: jnp.ndarray | None,
    imu_index_arrays: jnp.ndarray,
    dt_imu_mean: jnp.ndarray,
    *,
    config_for_filter: EKFConfig,
    layout: StateLayout,
) -> EKFComputationResult:
    """Core EKF implementation staged under ``jax.jit``."""
    n_cam = int(t_cam_jax.shape[0])

    # led_distance guaranteed non-None by caller assert (see extended_kalman_filter)
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
        has_vision_t = mask_cam_jax[t_idx]

        def propagate_from_prev(state_in):
            imu_indices = imu_index_arrays[t_idx]

            def propagate_imu(state, imu_idx):
                is_valid = imu_idx >= 0

                def do_propagate(s):
                    u = U_imu_jax[imu_idx]
                    dt = lax.cond(
                        imu_idx > 0,
                        lambda: t_imu_jax[imu_idx] - t_imu_jax[imu_idx - 1],
                        lambda: dt_imu_mean,
                    )
                    return predict_step(
                        s, u, dt, config_for_filter, has_vision_t, layout=layout
                    )

                def no_propagate(s):
                    return s

                return lax.cond(is_valid, do_propagate, no_propagate, state), None

            state_out, _ = lax.scan(propagate_imu, state_in, imu_indices)
            return state_out

        def no_propagate(state_in):
            return state_in

        state_pred = lax.cond(t_idx == 0, no_propagate, propagate_from_prev, state_prev)

        state_after_pos, log_lik_pos = update_step(
            state_pred,
            camera_model,
            t_idx,
            mask_cam_jax[t_idx],
            config_for_filter,
            layout=layout,
        )

        state_after_heading, log_lik_heading = update_heading(
            state_after_pos,
            heading_model,
            t_idx,
            mask_cam_jax[t_idx],
            layout=layout,
        )

        state_filt, log_lik_zupt = update_zupt(
            state_after_heading,
            config_for_filter,
        )

        log_lik_k = log_lik_pos + log_lik_heading + log_lik_zupt

        outputs = {
            "filtered_mean": state_filt.mean,
            "filtered_cov": state_filt.cov,
            "predicted_mean": state_pred.mean,
            "predicted_cov": state_pred.cov,
        }

        carry_next = (state_filt, log_lik_accum + log_lik_k)
        return carry_next, outputs

    carry_init = (
        initial_state,
        jnp.asarray(0.0, dtype=initial_state.mean.dtype),
    )

    (_, log_lik_total), outputs = lax.scan(filter_step, carry_init, jnp.arange(n_cam))

    return EKFComputationResult(
        filtered_means=outputs["filtered_mean"],
        filtered_covariances=outputs["filtered_cov"],
        predicted_means=outputs["predicted_mean"],
        predicted_covariances=outputs["predicted_cov"],
        marginal_loglik=log_lik_total,
    )


_extended_kalman_filter_jit = jax.jit(
    _extended_kalman_filter_impl,
    static_argnames=EXTENDED_KALMAN_FILTER_STATIC_ARGNAMES,
    donate_argnums=EXTENDED_KALMAN_FILTER_DONATE_ARGNUMS,
)


# =============================================================================
# Utility Functions
# =============================================================================


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

    Parameters
    ----------
    state : EKFState
        Current state (mean (n,), cov (n, n)).
    u_imu : jnp.ndarray
        IMU input [ω_z(rad/s), f_x(m/s^2), f_y(m/s^2)] (3,).
    dt_imu : float
        IMU timestep (s).
    config : EKFConfig
        EKF configuration.
    has_vision : bool, default True
        Whether a camera measurement is available at the current frame (for
        blackout-aware Q scaling).
    layout : StateLayout
        State index mapping.

    Returns
    -------
    EKFState
        Predicted state.
    """
    m, P = state.mean, state.cov

    # Dynamics function with fixed parameters
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

    # Predict mean
    m_pred = f(m)

    if layout.has_quaternion_orientation:
        # Autodiff through the quaternion exponential is singular at exactly
        # zero rotation. Use a conservative transition Jacobian for this
        # experimental branch: constant-velocity x/y dynamics plus identity
        # orientation/bias propagation. Quaternion process noise is still
        # injected through Q below.
        F_x = jnp.eye(m.shape[0], dtype=m.dtype)
        F_x = F_x.at[layout.pos_idx[0], layout.vel_idx[0]].set(dt_imu)
        F_x = F_x.at[layout.pos_idx[1], layout.vel_idx[1]].set(dt_imu)
    else:
        # Jacobian
        F = jacfwd(f)
        F_x = F(m)

    # Wrap scalar heading or normalize quaternion orientation.
    if layout.has_heading_2d:
        h_idx = get_heading_index(layout)
        m_pred = m_pred.at[h_idx].set(wrap_angle(m_pred[h_idx]))
        q_heading = m_pred[h_idx]
    else:
        m_pred = normalize_state_orientation(m_pred, layout)
        q_heading = state_yaw(m_pred, layout)

    # Assemble process noise using shared helper
    # Use predicted heading θ⁺ for tighter alignment between dynamics and Q
    Q = assemble_Q(
        config,
        theta=q_heading,  # Use predicted heading, not current
        dt=dt_imu,
        n=m.shape[0],
        has_vision=has_vision,
        dtype=m.dtype,
    )

    # Predict covariance
    P_pred = F_x @ P @ F_x.T + Q
    P_pred = symmetrize(P_pred)
    predicted_state = EKFState(mean=m_pred, cov=P_pred)

    if layout.has_quaternion_orientation:
        predicted_state = _apply_gravity_orientation_update(
            predicted_state,
            u_imu,
            config,
            layout=layout,
        )

    return predicted_state


def _gravity_direction_prediction(
    state_mean: jnp.ndarray,
    config: EKFConfig,
    *,
    layout: StateLayout,
) -> jnp.ndarray:
    """Predict stationary accelerometer direction from quaternion orientation."""

    quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
    gravity_magnitude = jnp.linalg.norm(
        jnp.asarray(config.imu_gravity_body, dtype=state_mean.dtype)
    )
    gravity_world = jnp.array(
        [0.0, 0.0, gravity_magnitude],
        dtype=state_mean.dtype,
    )
    gravity_body = rotate_vector_world_to_body(state_mean[quat_idx], gravity_world)
    return gravity_body / jnp.maximum(
        jnp.linalg.norm(gravity_body),
        jnp.asarray(1e-6, dtype=state_mean.dtype),
    )


def _apply_gravity_orientation_update(
    state: EKFState,
    u_imu: jnp.ndarray,
    config: EKFConfig,
    *,
    layout: StateLayout,
) -> EKFState:
    """Apply gated accelerometer gravity-direction update for quaternion layouts."""

    m = state.mean
    accel = u_imu[3:6]
    gyro = u_imu[:3]
    accel_bias = jnp.array(
        [
            m[layout.bias_accel_idx[0]],
            m[layout.bias_accel_idx[1]],
            m[layout.bias_accel_idx[2]],
        ],
        dtype=m.dtype,
    )
    accel_unbiased = accel - accel_bias
    accel_norm = jnp.linalg.norm(accel_unbiased)
    gravity_magnitude = jnp.linalg.norm(
        jnp.asarray(config.imu_gravity_body, dtype=m.dtype)
    )
    accel_direction = accel_unbiased / jnp.maximum(
        accel_norm,
        jnp.asarray(1e-6, dtype=m.dtype),
    )
    gyro_norm = jnp.linalg.norm(gyro)
    use_update = (
        jnp.asarray(config.use_gravity_orientation_update)
        & jnp.isfinite(accel_unbiased).all()
        & jnp.isfinite(gyro).all()
        & (accel_norm > 1e-6)
        & (
            jnp.abs(accel_norm - gravity_magnitude)
            <= config.gravity_accel_magnitude_tolerance_m_s2
        )
        & (gyro_norm <= config.gravity_gyro_norm_threshold_rad_s)
    )

    def do_update(state_in: EKFState) -> EKFState:
        mean, cov = state_in.mean, state_in.cov
        z_pred = _gravity_direction_prediction(mean, config, layout=layout)
        H = jacfwd(lambda x: _gravity_direction_prediction(x, config, layout=layout))(
            mean
        )
        innovation = accel_direction - z_pred
        R = jnp.eye(3, dtype=mean.dtype) * jnp.asarray(
            config.gravity_orientation_measurement_noise,
            dtype=mean.dtype,
        )
        S = H @ cov @ H.T + R
        K = psd_solve(S, H @ cov).T
        mean_upd = mean + K @ innovation
        mean_upd = normalize_state_orientation(mean_upd, layout)
        cov_upd = joseph_update(cov, K, H, R)
        return EKFState(mean=mean_upd, cov=cov_upd)

    return lax.cond(use_update, do_update, lambda state_in: state_in, state)


# =============================================================================
# Update Step Helpers
# =============================================================================


def update_step(
    state: EKFState,
    camera_model: CameraPositionModel,
    frame_idx: int,
    observation_is_valid: bool,
    config: EKFConfig,
    *,
    layout: StateLayout,
) -> tuple[EKFState, float]:
    """EKF measurement update step using camera model.

    Parameters
    ----------
    state : EKFState
        Predicted state.
    camera_model : CameraPositionModel
        Camera position measurement model with preallocated frame data.
    frame_idx : int
        Current frame index.
    observation_is_valid : bool
        Observation validity flag.
    config : EKFConfig
        EKF configuration (used for IEKF iterations and gating).
    layout : StateLayout
        State index mapping.

    Returns
    -------
    tuple[EKFState, float]
        Updated state and log-likelihood.

    Notes
    -----
    Uses the MeasurementModel protocol and generic ekf_projected_update() primitive
    to handle 4D→2D projection for single-LED observations. Supports iterated EKF
    (IEKF) via config.num_iter parameter.
    """
    m_pred, P_pred = state.mean, state.cov

    # If no valid observation, return prediction unchanged with zero log-likelihood
    def no_update(m, P):
        return EKFState(mean=m, cov=P), 0.0

    # If valid observation, perform update
    def do_update(m, P):
        # Get LED validity from camera model
        both_leds, only_led1, only_led2, _ = camera_model.subspace(frame_idx)
        led1_valid = both_leds | only_led1
        led2_valid = both_leds | only_led2

        # If no valid observations, return prediction with zero log-likelihood
        def no_leds_update(m_in, P_in):
            return EKFState(mean=m_in, cov=P_in), 0.0

        # If at least one LED valid, perform update (with optional IEKF)
        def do_leds_update(m_in, P_in):
            # IEKF: Iterate re-linearization around posterior
            def iekf_step(carry, _):
                """Single IEKF iteration using generic ekf_projected_update() primitive."""
                m_iter, P_iter = carry
                state_iter = EKFState(mean=m_iter, cov=P_iter)

                # Get measurement prediction from camera model
                meas_pred = camera_model.predict(m_iter)

                # Get Jacobian from camera model
                H = camera_model.jacobian(m_iter)

                # Get innovation from camera model (handles NaN → zero residual)
                innovation = camera_model.innovation(frame_idx, meas_pred)

                # Get measurement covariance from camera model (confidence-scaled)
                R = camera_model.meas_cov(frame_idx)
                R_diag = jnp.diag(R)

                # Call generic EKF projected update primitive
                state_upd, nis, log_lik = ekf_projected_update(
                    state_iter, innovation, H, R_diag, both_leds, only_led1, only_led2
                )

                # Wrap scalar heading or normalize quaternion after update.
                if layout.has_heading_2d:
                    h_idx = get_heading_index(layout)
                    m_upd = state_upd.mean.at[h_idx].set(
                        wrap_angle(state_upd.mean[h_idx])
                    )
                else:
                    m_upd = normalize_state_orientation(state_upd.mean, layout)
                state_upd = EKFState(mean=m_upd, cov=state_upd.cov)

                return (state_upd.mean, state_upd.cov), (nis, log_lik, both_leds)

            # Run IEKF iterations
            carry_init = (m_in, P_in)
            (m_final, P_final), (nis_all, log_lik_all, both_all) = lax.scan(
                iekf_step, carry_init, jnp.arange(config.num_iter)
            )

            # Extract final (last) iteration values
            nis_final = nis_all[-1]
            log_lik_final = log_lik_all[-1]
            both_final = both_all[-1]

            # Mahalanobis gating: reject if NIS exceeds χ² threshold
            def apply_gating():
                """Apply Mahalanobis gating."""
                # Determine measurement dimensionality for threshold
                dof = lax.cond(both_final, lambda: 4, lambda: 2)
                threshold = chi2_threshold(dof, config.mahalanobis_threshold_prob)

                # Gate: accept if NIS < threshold, reject otherwise
                def accept_measurement():
                    return EKFState(mean=m_final, cov=P_final), log_lik_final

                def reject_measurement():
                    # Return prediction unchanged with zero log-likelihood
                    return EKFState(mean=m_in, cov=P_in), 0.0

                return lax.cond(
                    nis_final < threshold, accept_measurement, reject_measurement
                )

            def no_gating():
                """Return update without gating."""
                return EKFState(mean=m_final, cov=P_final), log_lik_final

            # Use lax.cond to select gating vs no-gating path
            return lax.cond(config.use_mahalanobis_gating, apply_gating, no_gating)

        # Conditional update based on whether we have any valid LEDs
        return lax.cond(led1_valid | led2_valid, do_leds_update, no_leds_update, m, P)

    # Conditional update based on validity flag
    return lax.cond(observation_is_valid, do_update, no_update, m_pred, P_pred)


def update_heading(
    state: EKFState,
    heading_model: HeadingPseudoModel,
    frame_idx: int,
    observation_is_valid: bool,
    *,
    layout: StateLayout,
) -> tuple[EKFState, jnp.ndarray]:
    """Apply 1D heading pseudo-measurement update using heading model.

    Parameters
    ----------
    state : EKFState
        Current state (after position update).
    heading_model : HeadingPseudoModel
        Heading measurement model with preallocated frame data.
    frame_idx : int
        Current frame index.
    observation_is_valid : bool
        Camera validity flag (False skips update entirely).
    layout : StateLayout
        State index mapping.

    Returns
    -------
    tuple[EKFState, jnp.ndarray]
        Updated state and heading measurement log-likelihood (scalar).

    Notes
    -----
    Uses large-R gating via heading_model.meas_cov(): invalid observations
    yield R=1e6 so K≈0, avoiding branching in JAX while preventing spurious updates.
    """
    observation_flag = jnp.asarray(observation_is_valid, dtype=bool)

    def no_update(state_in: EKFState) -> tuple[EKFState, jnp.ndarray]:
        return state_in, jnp.array(0.0, dtype=state_in.mean.dtype)

    def do_update(state_in: EKFState) -> tuple[EKFState, jnp.ndarray]:
        m, P = state_in.mean, state_in.cov

        # Get heading prediction from model
        h_pred_vec = heading_model.predict(m)  # (1,)

        # Get innovation from model (already angle-wrapped)
        innovation_vec = heading_model.innovation(frame_idx, h_pred_vec)  # (1,)
        innov = innovation_vec[0]

        # Get measurement covariance from model (already gates invalid)
        R_mat = heading_model.meas_cov(frame_idx)  # (1, 1)
        R_heading = R_mat[0, 0]

        # Get Jacobian from model
        H = heading_model.jacobian(m)  # (1, n)

        # Innovation covariance (1x1 matrix)
        S = H @ P @ H.T + R_mat

        # Kalman gain (n, 1)
        K = psd_solve(S, H @ P).T

        # Mean update
        m_upd = m + (K @ jnp.array([[innov]])).ravel()

        # Wrap scalar heading or normalize quaternion after update
        if layout.has_heading_2d:
            h_idx = get_heading_index(layout)
            m_upd = m_upd.at[h_idx].set(wrap_angle(m_upd[h_idx]))
        else:
            m_upd = normalize_state_orientation(m_upd, layout)

        # Covariance update using Joseph form
        P_upd = joseph_update(P, K, H, R_mat)

        # Log-likelihood
        # For gated observations (R=1e6), this will be near zero
        S_scalar = S[0, 0]
        log_lik = -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(S_scalar) + innov**2 / S_scalar)

        # Zero out log-likelihood if R is huge (gated)
        use_heading = R_heading < HEADING_GATE_THRESHOLD
        log_lik = lax.select(
            use_heading,
            log_lik,
            jnp.array(0.0, dtype=log_lik.dtype),
        )

        return EKFState(m_upd, P_upd), log_lik

    return lax.cond(observation_flag, do_update, no_update, state)


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
    """Run Extended Kalman Filter on a full trajectory.

    Parameters
    ----------
    ekf_config : EKFConfig
        EKF configuration.
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
    initial_state : EKFState | None, optional
        Optional initial state (auto-initialized if None).
    conf_cam : np.ndarray | None, optional
        Confidence scores (N_cam, 4) for [x1,y1,x2,y2] in [0, 1]. If provided,
        measurement noise is scaled per-dimension.

    Returns
    -------
    EKFResult
        Filtered and predicted states at camera times, and log-likelihood.
    """
    # Validate IMU input shape early so silent channel mismatches fail loudly.
    validate_imu_input_shape(
        U_imu,
        get_layout(ekf_config.state_mode),
        func_name="extended_kalman_filter",
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
    config_for_filter: EKFConfig

    if ekf_config.led_distance is None:
        estimated_led_distance = estimate_led_spacing(
            Z_cam_led1_jax, Z_cam_led2_jax, mask_cam_jax
        )
        # Create new config with estimated spacing (do NOT mutate original)
        config_for_filter = replace(ekf_config, led_distance=estimated_led_distance)
    else:
        # Use original config as-is
        config_for_filter = ekf_config

    # Initialize state
    assert config_for_filter.led_distance is not None, (
        "led_distance must be set or auto-detected before filter"
    )
    if initial_state is None:
        initial_state = initialize_state(
            Z_cam_led1_jax,
            Z_cam_led2_jax,
            mask_cam_jax,
            dt_cam=jnp.mean(
                jnp.diff(t_cam_jax)
            ),  # Keep as JAX scalar for JIT compatibility
            led_distance=config_for_filter.led_distance,
            layout=get_layout(config_for_filter.state_mode),
        )

    # Resolve state layout once for this run
    layout = get_layout(config_for_filter.state_mode)

    # Compute mean IMU timestep for fallback when imu_idx == 0
    dt_imu_mean = jnp.mean(
        jnp.diff(t_imu_jax)
    )  # Keep as JAX scalar for JIT compatibility

    # Precompute IMU index arrays (host-side, using shared utility)
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    computation = _extended_kalman_filter_jit(
        initial_state,
        t_imu_jax,
        U_imu_jax,
        t_cam_jax,
        Z_cam_led1_jax,
        Z_cam_led2_jax,
        mask_cam_jax,
        conf_cam_jax,
        imu_index_arrays,
        dt_imu_mean,
        config_for_filter=config_for_filter,
        layout=layout,
    )

    return EKFResult(
        filtered_means=computation.filtered_means,
        filtered_covariances=computation.filtered_covariances,
        predicted_means=computation.predicted_means,
        predicted_covariances=computation.predicted_covariances,
        marginal_loglik=float(computation.marginal_loglik),
        estimated_led_distance=estimated_led_distance,
    )
