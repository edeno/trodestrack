"""Extended Kalman Filter (EKF) for sensor-fused tracking.

The production 2D EKF state vector is selected by ``EKFConfig.state_mode``;
the default is the 10D ``2d_cam_3d_imu`` layout (see
``trodestrack.models.state_layout``):

    x_k = [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]^T   (default, 10D)

The legacy 8D ``2d_full`` layout is also registered:

    x_k = [x, y, vx, vy, θ, b_gz, b_ax, b_ay]^T              (8D)

Where, common to both:
    - (x, y): Position in meters
    - (vx, vy): Velocity in m/s
    - θ: Heading angle in radians
    - b_gz: Gyroscope z-axis bias in rad/s
    - b_ax, b_ay: Accelerometer x, y biases in m/s²
    - 10D adds vz (vertical velocity) and b_az (vertical accel bias) for
      3D-accel IMU support.

State dimension is dimension-agnostic throughout the filter; per-component
indices are resolved via ``StateLayout`` rather than hardcoded.

The filter fuses:
    - High-rate IMU measurements (gyro, accel) at ~200 Hz
    - Low-rate camera observations (LED positions) at ~30 Hz

Key features:
    - IMU pre-integration between camera frames
    - Velocity damping to model drag
    - Dual-LED position and heading measurements
    - Experimental 3D camera + quaternion 6-DOF IMU filtering
    - Mahalanobis gating for outlier rejection
    - RTS smoother for offline processing (see runtime/offline.py)

References:
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
    build_quaternion_transition_jacobian,
    camera_stationary_zupt_gate_2d,
    camera_stationary_zupt_gate_3d,
    chi2_threshold,
    compute_imu_index_arrays,
    dynamics_function,
    estimate_led_spacing,
    imu_stationary_zupt_gate,
    initialize_state,
    joseph_update,
    normalize_state_orientation,
    psd_solve,
    state_yaw,
    symmetrize,
    update_zupt,
    update_zupt_visual_context,
    validate_camera_3d_input_shapes,
    validate_camera_input_shapes,
    validate_imu_input_shape,
    validate_initial_state,
    validate_timestamps,
    wrap_angle,
)
from trodestrack.models.filter_update import ekf_projected_update
from trodestrack.models.process_noise import assemble_Q
from trodestrack.models.quaternion import rotate_vector_world_to_body
from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.sensors.camera_position_3d import Camera3DPositionModel
from trodestrack.models.sensors.event_location import (
    EventLocationModel,
    update_event_location,
)
from trodestrack.models.sensors.heading_pseudo import HeadingPseudoModel
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
        Number of inner IEKF iterations per measurement update. Must be
        ``>= 1``; ``num_iter=0`` is rejected because the 2D path indexes
        ``nis_all[-1]`` after a zero-length scan (IndexError) and the 3D
        path silently skips the camera update entirely.
    """

    num_iter: int = 1

    def __post_init__(self) -> None:
        # Run shared (FilterCoreConfig) validation first.
        super().__post_init__()
        if not isinstance(self.num_iter, int) or self.num_iter < 1:
            raise ValueError(
                "num_iter must be an integer >= 1 (1 = standard EKF, "
                f">1 = IEKF iterations); got {self.num_iter!r}."
            )


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


class EKF3DResult(NamedTuple):
    """EKF filtering result for 3D camera + 6-DOF IMU mode."""

    filtered_means: jnp.ndarray
    filtered_covariances: jnp.ndarray
    predicted_means: jnp.ndarray
    predicted_covariances: jnp.ndarray
    marginal_loglik: float
    usable_vision_mask: jnp.ndarray | None = None


class EKF3DComputationResult(NamedTuple):
    """Internal 3D EKF outputs produced by the traceable core."""

    filtered_means: jnp.ndarray
    filtered_covariances: jnp.ndarray
    predicted_means: jnp.ndarray
    predicted_covariances: jnp.ndarray
    marginal_loglik: jnp.ndarray
    usable_vision_mask: jnp.ndarray


class EKFComputationResult(NamedTuple):
    """Internal EKF outputs produced by the JIT implementation."""

    filtered_means: jnp.ndarray
    filtered_covariances: jnp.ndarray
    predicted_means: jnp.ndarray
    predicted_covariances: jnp.ndarray
    marginal_loglik: jnp.ndarray
    usable_vision_mask: jnp.ndarray


EXTENDED_KALMAN_FILTER_STATIC_ARGNAMES = (
    "layout",
    "config_for_filter",
    "max_events_per_frame",
)
# Buffer donation not beneficial: input shapes (N_cam, 2) don't match output shapes (N_cam, n)
# XLA cannot reuse donated buffers when shapes differ. Donation only helps when input
# buffers can be reused for outputs of matching shape/dtype.
EXTENDED_KALMAN_FILTER_DONATE_ARGNUMS: tuple[int, ...] = ()
EXTENDED_KALMAN_FILTER_3D_STATIC_ARGNAMES = ("layout", "config_for_filter")
EXTENDED_KALMAN_FILTER_3D_DONATE_ARGNUMS: tuple[int, ...] = ()


def _extended_kalman_filter_impl(
    initial_state: EKFState,
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
    config_for_filter: EKFConfig,
    layout: StateLayout,
    max_events_per_frame: int,
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
        both_leds, only_led1, only_led2, _ = camera_model.subspace(t_idx)
        frame_has_led = mask_cam_jax[t_idx] & (both_leds | only_led1 | only_led2)

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
                        s, u, dt, config_for_filter, frame_has_led, layout=layout
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
        state_after_zupt, log_lik_zupt = update_zupt(
            state_after_heading,
            config_for_filter,
            active=has_seen_vision_next & imu_stationary & stationary_context_next,
        )

        event_source_indices = event_indices_per_frame_jax[t_idx]
        state_filt, log_lik_event = update_event_location(
            state_after_zupt,
            event_model,
            event_source_indices,
        )

        log_lik_k = log_lik_pos + log_lik_heading + log_lik_zupt + log_lik_event

        outputs = {
            "filtered_mean": state_filt.mean,
            "filtered_cov": state_filt.cov,
            "predicted_mean": state_pred.mean,
            "predicted_cov": state_pred.cov,
            "usable_vision": frame_has_led,
        }

        carry_next = (
            state_filt,
            log_lik_accum + log_lik_k,
            has_seen_vision_next,
            stationary_context_next,
            stationary_context_age_next,
        )
        return carry_next, outputs

    carry_init = (
        initial_state,
        jnp.asarray(0.0, dtype=initial_state.mean.dtype),
        jnp.asarray(initial_zupt_context, dtype=bool),
        jnp.asarray(False, dtype=bool),
        jnp.asarray(
            config_for_filter.zupt_visual_context_hold_frames + 1, dtype=jnp.int32
        ),
    )

    (_, log_lik_total, _, _, _), outputs = lax.scan(
        filter_step, carry_init, jnp.arange(n_cam)
    )

    return EKFComputationResult(
        filtered_means=outputs["filtered_mean"],
        filtered_covariances=outputs["filtered_cov"],
        predicted_means=outputs["predicted_mean"],
        predicted_covariances=outputs["predicted_cov"],
        marginal_loglik=log_lik_total,
        usable_vision_mask=outputs["usable_vision"],
    )


_extended_kalman_filter_jit = jax.jit(
    _extended_kalman_filter_impl,
    static_argnames=EXTENDED_KALMAN_FILTER_STATIC_ARGNAMES,
    donate_argnums=EXTENDED_KALMAN_FILTER_DONATE_ARGNUMS,
)


# Default static pad width when ``ttl_events`` is not configured. The padded
# arrays are still threaded through the JIT'd core (filled with -1) so the
# scan body has a single shape regardless of whether events are configured.
_DEFAULT_NO_EVENT_MAX_PER_FRAME: int = 1


def _resolve_event_inputs(
    event_source_anchors: np.ndarray | None,
    event_source_covariances: np.ndarray | None,
    event_indices_per_frame: np.ndarray | None,
    *,
    n_cam: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Validate optional event arguments and return JAX-ready dense arrays.

    If all three arguments are ``None`` the wrapper builds an empty no-op
    event channel (one padded slot per frame, all sentinel ``-1``). If any
    one is provided, all three are required.
    """
    provided = [
        event_source_anchors is not None,
        event_source_covariances is not None,
        event_indices_per_frame is not None,
    ]
    if any(provided) and not all(provided):
        raise ValueError(
            "extended_kalman_filter event channel: event_source_anchors, "
            "event_source_covariances, and event_indices_per_frame must be "
            "provided together (all None disables the channel)."
        )
    if not any(provided):
        anchors_jax = jnp.zeros((1, 2), dtype=jnp.float32)
        covariances_jax = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32), (1, 2, 2))
        indices_jax = jnp.full(
            (n_cam, _DEFAULT_NO_EVENT_MAX_PER_FRAME), -1, dtype=jnp.int32
        )
        return (
            anchors_jax,
            covariances_jax,
            indices_jax,
            _DEFAULT_NO_EVENT_MAX_PER_FRAME,
        )

    assert event_source_anchors is not None
    assert event_source_covariances is not None
    assert event_indices_per_frame is not None
    anchors = np.asarray(event_source_anchors, dtype=float)
    covariances = np.asarray(event_source_covariances, dtype=float)
    indices = np.asarray(event_indices_per_frame, dtype=int)

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(
            f"event_source_anchors must have shape (n_sources, 2); got {anchors.shape}."
        )
    n_sources = anchors.shape[0]
    if covariances.shape != (n_sources, 2, 2):
        raise ValueError(
            "event_source_covariances must have shape (n_sources, 2, 2) "
            f"matching event_source_anchors; got {covariances.shape} for "
            f"n_sources={n_sources}."
        )
    if not np.all(np.isfinite(anchors)) or not np.all(np.isfinite(covariances)):
        raise ValueError(
            "event_source_anchors / event_source_covariances must be finite."
        )
    if indices.ndim != 2 or indices.shape[0] != n_cam:
        raise ValueError(
            "event_indices_per_frame must have shape (len(t_cam), "
            f"max_events_per_frame); got {indices.shape} for n_cam={n_cam}."
        )
    valid_idx = indices >= 0
    if valid_idx.any() and (indices[valid_idx].max() >= n_sources):
        raise ValueError(
            "event_indices_per_frame contains a compact source index out of "
            f"range [0, {n_sources}); got max "
            f"{int(indices[valid_idx].max())}."
        )
    if (indices[~valid_idx] != -1).any():
        raise ValueError(
            "event_indices_per_frame padded entries must be exactly -1; "
            "negative values other than -1 are not allowed."
        )
    if n_sources == 0:
        # Avoid an empty-axis ``EventLocationModel`` by collapsing to the
        # no-events fallback (still validated above for shape consistency).
        anchors_jax = jnp.zeros((1, 2), dtype=jnp.float32)
        covariances_jax = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32), (1, 2, 2))
        indices_jax = jnp.full(indices.shape, -1, dtype=jnp.int32)
        return anchors_jax, covariances_jax, indices_jax, indices.shape[1]

    return (
        jnp.asarray(anchors, dtype=jnp.float32),
        jnp.asarray(covariances, dtype=jnp.float32),
        jnp.asarray(indices, dtype=jnp.int32),
        int(indices.shape[1]),
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
        # zero rotation. Use a first-order transition Jacobian for this
        # experimental branch so IMU biases can become observable from later
        # camera updates without tracing through the quaternion exponential.
        F_x = build_quaternion_transition_jacobian(
            m,
            m_pred,
            dt_imu,
            config.damping_coeff,
            layout,
            u_imu=u_imu,
            enable_experimental_accel_translation=(
                config.enable_experimental_accel_translation
            ),
        )
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
        orientation_quaternion=(
            m_pred[jnp.array(layout.heading_idx, dtype=jnp.int32)]
            if layout.has_quaternion_orientation
            else None
        ),
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
    """Predict stationary accelerometer direction from quaternion orientation.

    ``imu_gravity_body`` is interpreted as the calibrated world-frame gravity
    vector used by the 3D filter. Preserve non-vertical x/y components so the
    gravity pseudo-update matches translational gravity compensation.
    """

    quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
    gravity_world = jnp.asarray(config.imu_gravity_body, dtype=state_mean.dtype)
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
    gyro_bias = jnp.array(
        [
            m[layout.bias_gyro_idx[0]],
            m[layout.bias_gyro_idx[1]],
            m[layout.bias_gyro_idx[2]],
        ],
        dtype=m.dtype,
    )
    accel_bias = jnp.array(
        [
            m[layout.bias_accel_idx[0]],
            m[layout.bias_accel_idx[1]],
            m[layout.bias_accel_idx[2]],
        ],
        dtype=m.dtype,
    )
    gyro_unbiased = gyro - gyro_bias
    accel_unbiased = accel - accel_bias
    accel_norm = jnp.linalg.norm(accel_unbiased)
    gravity_magnitude = jnp.linalg.norm(
        jnp.asarray(config.imu_gravity_body, dtype=m.dtype)
    )
    accel_direction = accel_unbiased / jnp.maximum(
        accel_norm,
        jnp.asarray(1e-6, dtype=m.dtype),
    )
    gyro_norm = jnp.linalg.norm(gyro_unbiased)
    use_update = (
        jnp.asarray(config.use_gravity_orientation_update)
        & jnp.isfinite(accel_unbiased).all()
        & jnp.isfinite(gyro_unbiased).all()
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
            # IEKF: re-linearize at the current iterate while updating the
            # same FIXED prior every step. The previous implementation re-
            # fed each posterior as a fresh prior, which is sequential
            # filtering — running it ``num_iter`` times against a zero-
            # residual measurement shrank position variance by 1/num_iter
            # with no mean change. The fixed-prior corrected-innovation
            # pattern below matches Sarkka (2013, Algorithm 5.7) and the
            # 3D path's ``_camera_3d_linear_update``.
            prior_state = EKFState(mean=m_in, cov=P_in)

            def iekf_step(carry, _):
                """One IEKF iteration: linearize at m_iter, update from prior."""
                m_iter, _P_iter_unused = carry

                # Linearize at the current iterate.
                meas_pred = camera_model.predict(m_iter)
                H = camera_model.jacobian(m_iter)

                # Innovation z - h(m_iter), NaN-zeroed by camera_model.
                innovation = camera_model.innovation(frame_idx, meas_pred)

                # Confidence-scaled R diagonal.
                R_diag = jnp.diag(camera_model.meas_cov(frame_idx))

                # Corrected innovation:
                #   y_corr = (z - h(m_iter)) + H @ (m_iter - m_prior)
                # When m_iter == m_prior this reduces to a standard EKF
                # update; for subsequent IEKF iterations it shifts the
                # innovation so the same prior absorbs the linearization
                # change instead of the posterior of the previous step.
                corrected_innovation = innovation + H @ (m_iter - prior_state.mean)

                # Update the fixed prior with the corrected innovation.
                state_upd, nis, log_lik = ekf_projected_update(
                    prior_state,
                    corrected_innovation,
                    H,
                    R_diag,
                    both_leds,
                    only_led1,
                    only_led2,
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
    Invalid heading geometry is skipped exactly. Large-R gating alone is not a
    no-op when heading covariance is large.
    """
    observation_flag = jnp.asarray(observation_is_valid, dtype=bool)

    def no_update(state_in: EKFState) -> tuple[EKFState, jnp.ndarray]:
        return state_in, jnp.array(0.0, dtype=state_in.mean.dtype)

    def do_update(state_in: EKFState) -> tuple[EKFState, jnp.ndarray]:
        use_heading = heading_model.use_measurement(frame_idx)

        def apply_update(state_valid: EKFState) -> tuple[EKFState, jnp.ndarray]:
            m, P = state_valid.mean, state_valid.cov

            # Get heading prediction from model
            h_pred_vec = heading_model.predict(m)  # (1,)

            # Get innovation from model (already angle-wrapped)
            innovation_vec = heading_model.innovation(frame_idx, h_pred_vec)  # (1,)
            innov = innovation_vec[0]

            # Get measurement covariance from model
            R_mat = heading_model.meas_cov(frame_idx)  # (1, 1)

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
            S_scalar = S[0, 0]
            log_lik = -0.5 * (
                jnp.log(2 * jnp.pi) + jnp.log(S_scalar) + innov**2 / S_scalar
            )

            return EKFState(m_upd, P_upd), log_lik

        return lax.cond(use_heading, apply_update, no_update, state_in)

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
    event_source_anchors: np.ndarray | None = None,
    event_source_covariances: np.ndarray | None = None,
    event_indices_per_frame: np.ndarray | None = None,
) -> EKFResult:
    """Run Extended Kalman Filter on a full trajectory.

    Parameters
    ----------
    ekf_config : EKFConfig
        EKF configuration.
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
        - ``K = 6``: ``[ω_x, ω_y, ω_z, f_x, f_y, f_z]`` for the
          quaternion-orientation layouts ``2d_cam_6dof_imu_orientation``
          and the experimental ``3d_cam_6dof_imu``.

        Units: rad/s for gyro components, m/s² for specific-force.
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
    layout = get_layout(ekf_config.state_mode)

    # Reject 3D state layouts at the 2D entry point. Without this guard,
    # the 2D `extended_kalman_filter` builds the 2D `CameraPositionModel`
    # (which assumes (N, 2) LED arrays) on top of a 3D state and can return
    # a finite-but-wrong result, e.g. for the `3d_quat` alias.
    if layout.spatial_dim == 3:
        raise ValueError(
            f"state_mode={ekf_config.state_mode!r} is a 3D state layout and "
            "cannot be run through the 2D `extended_kalman_filter`, which "
            "uses 2D LED arrays and the 2D `CameraPositionModel`. Use "
            "`extended_kalman_filter_3d` for `3d_cam_6dof_imu`; other 3D "
            "layouts are registered for custom analyses / future filter work."
        )

    # Validate IMU input shape early so silent channel mismatches fail loudly.
    validate_imu_input_shape(
        U_imu,
        layout,
        t_imu=t_imu,
        func_name="extended_kalman_filter",
    )

    # Validate timestamps (finite + strictly increasing). The CLI already
    # enforces this; replicate at the Python API boundary so callers using
    # the function directly get the same guard. Without this, NaN / inf
    # / decreasing entries propagate through np.diff into compute_imu_-
    # index_arrays and IMU pre-integration and silently produce NaN
    # filtered_means.
    validate_timestamps(
        t_imu, name="t_imu", func_name="extended_kalman_filter", min_size=2
    )
    validate_timestamps(t_cam, name="t_cam", func_name="extended_kalman_filter")

    # Validate camera-aligned arrays match len(t_cam). Without this, JAX
    # indexing silently clamps a too-short Z_cam_led* / mask_cam / conf_cam
    # to its last in-range row, reusing a stale frame for every later step
    # and returning finite-but-wrong outputs.
    validate_camera_input_shapes(
        t_cam,
        Z_cam_led1,
        Z_cam_led2,
        mask_cam,
        conf_cam=conf_cam,
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

    (
        event_source_anchors_jax,
        event_source_covariances_jax,
        event_indices_per_frame_jax,
        max_events_per_frame,
    ) = _resolve_event_inputs(
        event_source_anchors,
        event_source_covariances,
        event_indices_per_frame,
        n_cam=int(t_cam_jax.shape[0]),
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
    initial_zupt_context = initial_state is not None
    if initial_state is None:
        initial_state = initialize_state(
            Z_cam_led1_jax,
            Z_cam_led2_jax,
            mask_cam_jax,
            dt_cam=jnp.mean(
                jnp.diff(t_cam_jax)
            ),  # Keep as JAX scalar for JIT compatibility
            led_distance=config_for_filter.led_distance,
            layout=layout,
        )
    else:
        # Validate caller-supplied initial state against the active layout.
        # A wrong-dimension mean / cov silently propagated through the
        # JIT'd core and emitted plausible-looking but layout-mismatched
        # filtered_means; non-finite entries poisoned every downstream
        # state.
        validate_initial_state(
            initial_state,
            layout,
            func_name="extended_kalman_filter",
        )

    # Compute mean IMU timestep for fallback when imu_idx == 0
    dt_imu_mean = jnp.mean(
        jnp.diff(t_imu_jax)
    )  # Keep as JAX scalar for JIT compatibility

    # Precompute IMU index arrays (host-side, using shared utility)
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)

    computation = _extended_kalman_filter_jit(
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

    return EKFResult(
        filtered_means=computation.filtered_means,
        filtered_covariances=computation.filtered_covariances,
        predicted_means=computation.predicted_means,
        predicted_covariances=computation.predicted_covariances,
        marginal_loglik=float(computation.marginal_loglik),
        estimated_led_distance=estimated_led_distance,
        usable_vision_mask=computation.usable_vision_mask,
    )


def extended_kalman_filter_3d(
    ekf_config: EKFConfig,
    t_imu: np.ndarray,
    U_imu: np.ndarray,
    t_cam: np.ndarray,
    Z_cam_leds: np.ndarray,
    led_offsets_body: np.ndarray,
    mask_cam_leds: np.ndarray | None = None,
    initial_state: EKFState | None = None,
    conf_cam: np.ndarray | None = None,
) -> EKF3DResult:
    """Run the experimental 3D camera + 6-DOF IMU EKF.

    This is the first full-3D filter path. It intentionally lives beside the
    existing 2D entry point because the camera measurement shape is different:
    ``Z_cam_leds`` is ``(n_cam, n_leds, 3)`` and ``led_offsets_body`` is
    ``(n_leds, 3)``.

    Notes
    -----
    The state mode must be ``"3d_cam_6dof_imu"``. Accelerometer translation is
    controlled by ``enable_experimental_accel_translation``; leave it disabled
    for camera/gyro-only comparisons and enable it explicitly for accel-enabled
    validation.
    """
    if ekf_config.state_mode != "3d_cam_6dof_imu":
        raise ValueError(
            "extended_kalman_filter_3d requires "
            "EKFConfig(state_mode='3d_cam_6dof_imu')."
        )

    layout = get_layout(ekf_config.state_mode)
    validate_imu_input_shape(
        U_imu,
        layout,
        t_imu=t_imu,
        func_name="extended_kalman_filter_3d",
    )

    # Reject non-finite / non-monotonic timestamps before any filtering,
    # for the same reason as the 2D path: dt derivation feeds compute_-
    # imu_index_arrays and IMU pre-integration and silently propagates
    # NaN / negative dt otherwise.
    validate_timestamps(
        t_imu, name="t_imu", func_name="extended_kalman_filter_3d", min_size=2
    )
    validate_timestamps(t_cam, name="t_cam", func_name="extended_kalman_filter_3d")

    # Validate the 3D camera-aligned arrays match len(t_cam) and the LED
    # offset count. Without this, JAX out-of-bounds indexing silently
    # clamps a too-short Z_cam_leds / mask_cam_leds / conf_cam to its
    # last in-range row and the filter returns finite-but-wrong outputs.
    validate_camera_3d_input_shapes(
        t_cam,
        Z_cam_leds,
        led_offsets_body,
        mask_cam_leds=mask_cam_leds,
        conf_cam=conf_cam,
        func_name="extended_kalman_filter_3d",
    )

    config_for_filter = ekf_config

    t_imu_jax = jnp.asarray(t_imu)
    U_imu_jax = jnp.asarray(U_imu)
    t_cam_jax = jnp.asarray(t_cam)
    Z_cam_leds_jax = jnp.asarray(Z_cam_leds)
    led_offsets_body_jax = jnp.asarray(led_offsets_body)
    mask_cam_leds_jax = (
        None if mask_cam_leds is None else jnp.asarray(mask_cam_leds, dtype=bool)
    )
    conf_cam_jax = (
        None if conf_cam is None else jnp.clip(jnp.asarray(conf_cam), 1e-2, 1.0)
    )

    initial_zupt_context = initial_state is not None
    if initial_state is None:
        initial_state = _initialize_3d_state(
            Z_cam_leds_jax,
            led_offsets_body_jax,
            layout=layout,
            mask_cam_leds=mask_cam_leds_jax,
        )
    else:
        validate_initial_state(
            initial_state, layout, func_name="extended_kalman_filter_3d"
        )

    # Uses NumPy interval construction, so keep it outside the traceable core.
    # The padded result has static shape for the nested IMU ``lax.scan``.
    imu_index_arrays = compute_imu_index_arrays(t_imu, t_cam)
    computation = _extended_kalman_filter_3d_jit(
        config_for_filter,
        jnp.asarray(initial_zupt_context, dtype=bool),
        t_imu_jax,
        U_imu_jax,
        t_cam_jax,
        Z_cam_leds_jax,
        led_offsets_body_jax,
        mask_cam_leds_jax,
        conf_cam_jax,
        initial_state,
        imu_index_arrays,
        layout=layout,
    )

    return EKF3DResult(
        filtered_means=computation.filtered_means,
        filtered_covariances=computation.filtered_covariances,
        predicted_means=computation.predicted_means,
        predicted_covariances=computation.predicted_covariances,
        marginal_loglik=float(computation.marginal_loglik),
        usable_vision_mask=computation.usable_vision_mask,
    )


def _extended_kalman_filter_3d_core(
    config_for_filter: EKFConfig,
    initial_zupt_context: jnp.ndarray,
    t_imu_jax: jnp.ndarray,
    U_imu_jax: jnp.ndarray,
    t_cam_jax: jnp.ndarray,
    Z_cam_leds_jax: jnp.ndarray,
    led_offsets_body_jax: jnp.ndarray,
    mask_cam_leds_jax: jnp.ndarray | None,
    conf_cam_jax: jnp.ndarray | None,
    initial_state: EKFState,
    imu_index_arrays: jnp.ndarray,
    *,
    layout: StateLayout,
) -> EKF3DComputationResult:
    """Traceable 3D EKF implementation returning JAX arrays and scalar.

    The public wrapper stages this through ``_extended_kalman_filter_3d_jit``
    and converts only the final log-likelihood scalar back to Python ``float``.
    Callers that need a pure-JAX result can use this core or the private JIT
    helper directly in tests and benchmarks.
    """
    camera_model = Camera3DPositionModel(
        led_offsets_body=led_offsets_body_jax,
        measurement_noise_base=config_for_filter.measurement_noise_pos,
        layout=layout,
        z_leds_all=Z_cam_leds_jax,
        mask_leds_all=mask_cam_leds_jax,
        conf_all=conf_cam_jax,
    )
    dt_imu_mean = jnp.mean(jnp.diff(t_imu_jax))
    chi2_thresholds = _chi2_threshold_table(
        camera_model.meas_dim,
        config_for_filter.mahalanobis_threshold_prob,
        dtype=initial_state.mean.dtype,
    )

    def filter_step(carry, t_idx):
        (
            state_prev,
            marginal_loglik_prev,
            has_seen_vision_prev,
            stationary_context_prev,
            stationary_context_age_prev,
        ) = carry
        has_vision = jnp.any(camera_model.valid_coordinates(t_idx))
        has_seen_vision_next = has_seen_vision_prev | has_vision

        def propagate_from_prev(state_in: EKFState) -> EKFState:
            imu_indices = imu_index_arrays[t_idx]

            def propagate_imu(state_imu: EKFState, imu_idx: jnp.ndarray):
                def do_propagate(state_valid: EKFState) -> EKFState:
                    dt = lax.cond(
                        imu_idx > 0,
                        lambda: t_imu_jax[imu_idx] - t_imu_jax[imu_idx - 1],
                        lambda: dt_imu_mean,
                    )
                    return predict_step(
                        state_valid,
                        U_imu_jax[imu_idx],
                        dt,
                        config_for_filter,
                        has_vision=has_vision,
                        layout=layout,
                    )

                state_out = lax.cond(
                    imu_idx >= 0,
                    do_propagate,
                    lambda state_invalid: state_invalid,
                    state_imu,
                )
                return state_out, None

            state_out, _ = lax.scan(propagate_imu, state_in, imu_indices)
            return state_out

        state_pred = lax.cond(
            t_idx == 0,
            lambda state_in: state_in,
            propagate_from_prev,
            state_prev,
        )
        predicted_mean = state_pred.mean
        predicted_covariance = state_pred.cov

        state_cam, log_lik_camera = _update_camera_3d_scanned(
            state_pred,
            camera_model,
            t_idx,
            layout,
            config_for_filter,
            chi2_thresholds,
        )
        zupt_imu_indices = imu_index_arrays[t_idx]
        zupt_imu_samples = U_imu_jax[jnp.maximum(zupt_imu_indices, 0)]
        zupt_imu_valid = zupt_imu_indices >= 0
        imu_stationary = imu_stationary_zupt_gate(
            zupt_imu_samples,
            zupt_imu_valid,
            config_for_filter,
            layout,
        )
        zupt_camera_lookback = jnp.asarray(
            config_for_filter.zupt_camera_stationary_window_frames,
            dtype=t_idx.dtype,
        )
        prev_idx = jnp.maximum(t_idx - zupt_camera_lookback, 0)
        visual_speed_valid, visual_stationary = camera_stationary_zupt_gate_3d(
            t_cam_jax,
            Z_cam_leds_jax,
            camera_model.valid_coordinates(t_idx),
            camera_model.valid_coordinates(prev_idx),
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
        state_filt, log_lik_zupt = update_zupt(
            state_cam,
            config_for_filter,
            active=has_seen_vision_next & imu_stationary & stationary_context_next,
        )
        marginal_loglik = marginal_loglik_prev + log_lik_camera + log_lik_zupt
        outputs = (
            state_filt.mean,
            state_filt.cov,
            predicted_mean,
            predicted_covariance,
            has_vision,
        )
        return (
            state_filt,
            marginal_loglik,
            has_seen_vision_next,
            stationary_context_next,
            stationary_context_age_next,
        ), outputs

    (final_state, marginal_loglik, _, _, _), scan_outputs = lax.scan(
        filter_step,
        (
            initial_state,
            jnp.asarray(0.0, dtype=initial_state.mean.dtype),
            jnp.asarray(initial_zupt_context, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(
                config_for_filter.zupt_visual_context_hold_frames + 1,
                dtype=jnp.int32,
            ),
        ),
        jnp.arange(t_cam_jax.shape[0], dtype=jnp.int32),
    )
    del final_state
    (
        filtered_means,
        filtered_covariances,
        predicted_means,
        predicted_covariances,
        usable_vision_mask,
    ) = scan_outputs

    return EKF3DComputationResult(
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        marginal_loglik=marginal_loglik,
        usable_vision_mask=usable_vision_mask,
    )


_extended_kalman_filter_3d_jit = jax.jit(
    _extended_kalman_filter_3d_core,
    static_argnames=EXTENDED_KALMAN_FILTER_3D_STATIC_ARGNAMES,
    donate_argnums=EXTENDED_KALMAN_FILTER_3D_DONATE_ARGNUMS,
)


def _initialize_3d_state(
    Z_cam_leds: jnp.ndarray,
    led_offsets_body: jnp.ndarray,
    *,
    layout: StateLayout,
    mask_cam_leds: jnp.ndarray | None,
) -> EKFState:
    """Initialize a 3D quaternion state from the first usable camera frame."""
    valid_leds = jnp.isfinite(Z_cam_leds).all(axis=2)
    if mask_cam_leds is not None:
        valid_leds = valid_leds & mask_cam_leds

    counts = jnp.sum(valid_leds, axis=1)
    frame_idx = int(np.argmax(np.asarray(counts > 0)))
    has_valid = bool(np.asarray(counts[frame_idx] > 0))

    mean = jnp.zeros(layout.n)
    quat_idx = jnp.array(layout.heading_idx, dtype=jnp.int32)
    pos_idx = jnp.array(layout.pos_idx, dtype=jnp.int32)
    mean = mean.at[quat_idx].set(jnp.array([1.0, 0.0, 0.0, 0.0]))

    if has_valid:
        frame_valid = valid_leds[frame_idx]
        denom = jnp.maximum(jnp.sum(frame_valid), 1)
        centroid_world = (
            jnp.sum(
                jnp.where(frame_valid[:, None], Z_cam_leds[frame_idx], 0.0),
                axis=0,
            )
            / denom
        )
        centroid_body = (
            jnp.sum(
                jnp.where(frame_valid[:, None], led_offsets_body, 0.0),
                axis=0,
            )
            / denom
        )
        mean = mean.at[pos_idx].set(centroid_world - centroid_body)

    # Position / velocity prior tightness depends on whether we actually
    # observed an LED in this frame. With ``has_valid`` the centroid is a
    # good initial guess (5 cm std). Without it the mean is the arbitrary
    # origin, and the prior must advertise that arbitrariness via wide
    # variance — otherwise downstream camera updates anchor against a
    # falsely-confident origin (verified: pre-fix returned origin with
    # 0.05² position variance even when no LEDs were observed).
    pos_var = 0.05**2 if has_valid else 10.0**2
    vel_var = 0.2**2 if has_valid else 1.0**2

    cov = jnp.eye(layout.n) * 1.0
    for idx in layout.pos_idx:
        cov = cov.at[idx, idx].set(pos_var)
    for idx in layout.vel_idx:
        cov = cov.at[idx, idx].set(vel_var)
    for idx in layout.heading_idx:
        cov = cov.at[idx, idx].set(0.5**2)
    for idx in layout.bias_gyro_idx:
        cov = cov.at[idx, idx].set(0.05**2)
    for idx in layout.bias_accel_idx:
        cov = cov.at[idx, idx].set(0.1**2)
    return EKFState(mean=mean, cov=cov)


def _update_camera_3d_scanned(
    state: EKFState,
    camera_model: Camera3DPositionModel,
    frame_idx: jnp.ndarray,
    layout: StateLayout,
    config: EKFConfig,
    chi2_thresholds: jnp.ndarray,
) -> tuple[EKFState, jnp.ndarray]:
    """Apply one JAX-scan-compatible gated 3D LED camera EKF update."""
    active_mask = camera_model.valid_coordinates(frame_idx)
    dof = jnp.sum(active_mask.astype(jnp.int32))

    def no_update(_: None) -> tuple[EKFState, jnp.ndarray]:
        return state, jnp.asarray(0.0, dtype=state.mean.dtype)

    def do_update(_: None) -> tuple[EKFState, jnp.ndarray]:
        state_iter = state
        innovation = jnp.zeros(camera_model.meas_dim, dtype=state.mean.dtype)
        S = jnp.eye(camera_model.meas_dim, dtype=state.mean.dtype)
        for _ in range(config.num_iter):
            state_iter, innovation, S = _camera_3d_linear_update(
                state,
                state_iter.mean,
                camera_model,
                frame_idx,
                layout,
            )

        log_lik = _gaussian_log_likelihood_masked(innovation, S, active_mask)
        if config.use_mahalanobis_gating:
            nis = _mahalanobis_distance_masked(innovation, S, active_mask)
            accept_update = nis < chi2_thresholds[dof]
        else:
            accept_update = jnp.asarray(True)
        return lax.cond(
            accept_update,
            lambda _: (state_iter, log_lik),
            lambda _: (state, jnp.asarray(0.0, dtype=state.mean.dtype)),
            operand=None,
        )

    return lax.cond(dof > 0, do_update, no_update, operand=None)


def _camera_3d_linear_update(
    prior_state: EKFState,
    linearization_mean: jnp.ndarray,
    camera_model: Camera3DPositionModel,
    frame_idx: int,
    layout: StateLayout,
) -> tuple[EKFState, jnp.ndarray, jnp.ndarray]:
    """Apply one IEKF linearized 3D camera update from the fixed prior."""
    mean_prior, cov_prior = prior_state.mean, prior_state.cov
    meas_pred = camera_model.predict(linearization_mean)
    innovation = camera_model.innovation(frame_idx, meas_pred)
    H = camera_model.jacobian(linearization_mean, frame_idx)
    R = camera_model.meas_cov(frame_idx)
    corrected_innovation = innovation + H @ (linearization_mean - mean_prior)
    S = H @ cov_prior @ H.T + R
    K = psd_solve(S, H @ cov_prior).T
    mean_upd = mean_prior + K @ corrected_innovation
    mean_upd = normalize_state_orientation(mean_upd, layout)
    cov_upd = joseph_update(cov_prior, K, H, R)
    return EKFState(mean=mean_upd, cov=cov_upd), corrected_innovation, S


def _masked_measurement_system(
    innovation: jnp.ndarray,
    covariance: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return fixed-size innovation/covariance with inactive rows as identity."""
    active = active_mask.astype(bool)
    active_outer = active[:, None] & active[None, :]
    eye = jnp.eye(covariance.shape[0], dtype=covariance.dtype)
    covariance_masked = jnp.where(active_outer, covariance, eye)
    innovation_masked = jnp.where(active, innovation, 0.0)
    return innovation_masked, covariance_masked


def _mahalanobis_distance_masked(
    innovation: jnp.ndarray,
    covariance: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Mahalanobis distance on active measurement coordinates using masks."""
    innovation_masked, covariance_masked = _masked_measurement_system(
        innovation,
        covariance,
        active_mask,
    )
    solved = psd_solve(covariance_masked, innovation_masked)
    return jnp.dot(innovation_masked, solved)


def _chi2_threshold_table(
    max_dof: int,
    prob: float,
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Return chi-square thresholds indexed by dynamic active dimension."""
    from scipy.stats import chi2

    values = [0.0] + [float(chi2.ppf(prob, df=dof)) for dof in range(1, max_dof + 1)]
    return jnp.asarray(values, dtype=dtype)


def _gaussian_log_likelihood_masked(
    innovation: jnp.ndarray,
    covariance: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Gaussian log likelihood on active measurement coordinates using masks."""
    innovation_masked, covariance_masked = _masked_measurement_system(
        innovation,
        covariance,
        active_mask,
    )
    dim = jnp.sum(active_mask.astype(innovation.dtype))
    solved = psd_solve(covariance_masked, innovation_masked)
    sign, logdet = jnp.linalg.slogdet(symmetrize(covariance_masked))
    logdet = jnp.where(sign > 0, logdet, jnp.asarray(0.0, dtype=innovation.dtype))
    return -0.5 * (
        dim * jnp.log(jnp.asarray(2.0 * np.pi, dtype=innovation.dtype))
        + logdet
        + jnp.dot(innovation_masked, solved)
    )
