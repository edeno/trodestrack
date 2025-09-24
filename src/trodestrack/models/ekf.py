"""Extended Kalman Filter implementation for 2D tracking.

This module implements the EKF algorithm for online state estimation, featuring:
- JAX-compiled prediction and update steps
- Robust measurement handling with gating
- Efficient Jacobian computation via automatic differentiation
- Support for missing measurements and occlusions
- Functional interface for lax.scan forward pass
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import dataclass
from jax import Array
from jax.typing import ArrayLike

from ._solvers import kalman_gain, mahalanobis_distance
from .gating import chi_squared_threshold
from .dynamics import (
    compute_process_noise,
    predict_covariance,
    rotation_matrix_2d,
    wrap_angle,
)
from .measurements import (
    _create_position_heading_jacobian,
    _create_position_jacobian,
    create_measurement_noise,
)
from .state import State2D, array_to_state, state_to_array


class EKFState(NamedTuple):
    """EKF state representation.

    Attributes:
        state: Current state estimate
        covariance: State covariance matrix (8x8)
        log_likelihood: Cumulative log-likelihood
    """

    state: ArrayLike  # 8-dimensional state vector
    covariance: ArrayLike  # 8x8 covariance matrix
    log_likelihood: float


class EKFResult(NamedTuple):
    """Result from EKF update step.

    Attributes:
        state: Updated EKF state
        innovation: Measurement innovation (residual)
        innovation_covariance: Innovation covariance matrix
        kalman_gain: Kalman gain matrix
        gated: Whether measurement was gated (rejected)
    """

    state: EKFState
    innovation: ArrayLike
    innovation_covariance: ArrayLike
    kalman_gain: ArrayLike
    gated: ArrayLike


class EkfCarry(NamedTuple):
    """Carry state for lax.scan EKF forward pass.

    Attributes:
        x: Current state estimate (8-dimensional)
        P: Current covariance matrix (8x8)
    """

    x: ArrayLike
    P: ArrayLike


class EkfOutputs(NamedTuple):
    """Outputs from EKF step for lax.scan.

    Attributes:
        x_filt: Filtered state estimate
        P_filt: Filtered covariance matrix
        x_pred: Predicted state estimate (before update)
        P_pred: Predicted covariance matrix (before update)
    """

    x_filt: ArrayLike
    P_filt: ArrayLike
    x_pred: ArrayLike
    P_pred: ArrayLike


# Type alias for EKF step input
EkfInput = Tuple[Dict[str, Any], float, Optional[ArrayLike], Dict[str, Any]]


@dataclass
class EkfScanInputs:
    """PyTree dataclass for EKF scan inputs.

    This provides a clean, functional structure for scan inputs that is
    JIT-cache friendly and avoids large tuples with NaN padding.

    Attributes:
        measurements: Measurement data (positions, headings, confidences, validity masks)
        imu_data: IMU measurements [ax, ay, gz] for each frame
        time_deltas: Time differences between frames
        filter_params: Filter configuration parameters
    """

    # Measurement data
    positions: ArrayLike  # (n_frames, 2) - [x, y] positions
    headings: ArrayLike  # (n_frames,) - heading angles
    confidences: ArrayLike  # (n_frames,) - measurement confidences
    position_valid: ArrayLike  # (n_frames,) - True if position is valid
    heading_valid: ArrayLike  # (n_frames,) - True if heading is valid

    # IMU and timing
    imu_blocks: ArrayLike  # (n_frames, 3) - [ax, ay, gz] for each frame
    dt: ArrayLike  # (n_frames,) - time deltas

    # Filter configuration (constant values)
    velocity_damping: float
    accel_noise_std: float
    gyro_noise_std: float
    bias_drift_std: float
    position_noise_std: float
    heading_noise_std: float
    gate_threshold: float


@jax.jit
def ekf_predict(
    ekf_state: EKFState,
    dt: float,
    accel: ArrayLike,
    gyro: ArrayLike,
    velocity_damping: float,
    accel_noise_std: float,
    gyro_noise_std: float,
    bias_drift_std: float,
) -> EKFState:
    """EKF prediction step.

    Args:
        ekf_state: Current EKF state
        dt: Time step (seconds)
        accel: Accelerometer measurement [ax, ay] (m/s²)
        gyro: Gyroscope measurement [gz] (rad/s)
        velocity_damping: Velocity damping coefficient λ
        accel_noise_std: Accelerometer noise std dev (m/s²)
        gyro_noise_std: Gyroscope noise std dev (rad/s)
        bias_drift_std: Bias drift std dev (per √s)

    Returns:
        Predicted EKF state
    """
    # Use pure JAX array operations instead of State2D conversion
    predicted_state = _predict_state_jax(ekf_state.state, dt, accel, gyro, velocity_damping)

    # Predict covariance using linearized dynamics
    process_noise = compute_process_noise(dt, accel_noise_std, gyro_noise_std, bias_drift_std)

    predicted_covariance = predict_covariance(
        ekf_state.covariance,
        ekf_state.state,
        dt,
        accel,
        gyro,
        velocity_damping,
        process_noise,
    )

    return EKFState(
        state=predicted_state,
        covariance=predicted_covariance,
        log_likelihood=ekf_state.log_likelihood,
    )


@jax.jit
def _predict_state_jax(
    state_array: ArrayLike,
    dt: float,
    accel: ArrayLike,
    gyro: ArrayLike,
    velocity_damping: float,
) -> Array:
    """JAX-optimized state prediction function.

    Args:
        state_array: State vector [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
        dt: Time step
        accel: Accelerometer measurement [ax, ay]
        gyro: Gyroscope measurement [gz]
        velocity_damping: Velocity damping coefficient

    Returns:
        Predicted state vector
    """
    # Extract state components
    pos = state_array[:2]
    vel = state_array[2:4]
    theta = state_array[4]
    b_gz = state_array[5]
    b_ax = state_array[6]
    b_ay = state_array[7]

    # Bias-corrected measurements
    accel_corrected = accel - jnp.array([b_ax, b_ay])
    gyro_corrected = gyro[0] - b_gz

    # Rotate acceleration from IMU/body frame to world frame
    R = rotation_matrix_2d(theta)
    accel_world = R @ accel_corrected

    # Convert acceleration to cm/s² for consistency with position units
    accel_corrected_cm = accel_world * 100.0

    # Apply velocity damping: v_damped = v * (1 - λ*dt)
    damping_factor = 1.0 - velocity_damping * dt
    vel_damped = vel * damping_factor

    # Velocity update: v_{k+1} = v_damped + a * dt
    vel_new = vel_damped + accel_corrected_cm * dt

    # Position update: x_{k+1} = x_k + v_k * dt + 0.5 * a * dt²
    pos_new = pos + vel * dt + 0.5 * accel_corrected_cm * dt**2

    # Heading update: θ_{k+1} = wrap(θ_k + ω * dt)
    theta_new = wrap_angle(theta + gyro_corrected * dt)

    # Biases remain unchanged (random walk model)
    return jnp.array([pos_new[0], pos_new[1], vel_new[0], vel_new[1], theta_new, b_gz, b_ax, b_ay])


def ekf_update(
    ekf_state: EKFState,
    measurement: ArrayLike,
    measurement_noise: ArrayLike,
    has_heading: bool,
    gate_threshold: Optional[float] = None,  # Auto-computed based on DoF if None
) -> EKFResult:
    """EKF measurement update step.

    Args:
        ekf_state: Predicted EKF state
        measurement: Measurement vector (position + optional heading)
        measurement_noise: Measurement noise covariance matrix R
        has_heading: Whether measurement includes heading
        gate_threshold: Chi-squared threshold for gating (auto-computed if None)

    Returns:
        EKF update result
    """
    # Auto-compute threshold based on degrees of freedom if not provided
    if gate_threshold is None:
        dof = 3 if has_heading else 2
        gate_threshold = chi_squared_threshold(dof, p_value=0.01)

    # Use specialized functions for each case to avoid conditional logic in JAX
    # Note: This is acceptable since the branching happens at the Python level (not in JIT)
    # and avoids recompilation issues
    if has_heading:
        return _ekf_update_position_heading(
            ekf_state, measurement, measurement_noise, gate_threshold
        )
    else:
        return _ekf_update_position_only(ekf_state, measurement, measurement_noise, gate_threshold)


@jax.jit
def _ekf_update_position_only(
    ekf_state: EKFState,
    measurement: ArrayLike,
    measurement_noise: ArrayLike,
    gate_threshold: float,
) -> EKFResult:
    """JAX-compiled EKF update for position-only measurements."""
    # Compute measurement Jacobian (position only)
    H = _create_position_jacobian(ekf_state.state)

    # Predicted measurement (position only)
    predicted_measurement = ekf_state.state[:2]

    # Innovation (measurement residual)
    innovation = measurement - predicted_measurement

    # Innovation covariance: S = H * P * H^T + R
    innovation_covariance = H @ ekf_state.covariance @ H.T + measurement_noise

    # Mahalanobis gating using safe solve
    mahalanobis_dist = mahalanobis_distance(innovation, innovation_covariance)
    gated = mahalanobis_dist > gate_threshold

    # Kalman gain: K = P * H^T * S^{-1} using safe solve
    K = kalman_gain(ekf_state.covariance, H, measurement_noise)

    # State update: x = x + K * innovation (only if not gated)
    updated_state = jnp.where(gated, ekf_state.state, ekf_state.state + K @ innovation)

    # Covariance update using Joseph form for numerical stability: P = (I - K*H) @ P @ (I - K*H)^T + K @ R @ K^T
    identity = jnp.eye(8)
    I_KH = identity - K @ H
    joseph_covariance = I_KH @ ekf_state.covariance @ I_KH.T + K @ measurement_noise @ K.T
    updated_covariance = jnp.where(gated, ekf_state.covariance, joseph_covariance)

    # Log-likelihood update (only if not gated)
    log_det_S = jnp.linalg.slogdet(innovation_covariance)[1]
    measurement_dim = 2
    log_likelihood_update = -0.5 * (
        measurement_dim * jnp.log(2 * jnp.pi) + log_det_S + mahalanobis_dist
    )

    updated_log_likelihood = jnp.where(
        gated, ekf_state.log_likelihood, ekf_state.log_likelihood + log_likelihood_update
    )

    updated_ekf_state = EKFState(
        state=updated_state,
        covariance=updated_covariance,
        log_likelihood=updated_log_likelihood,
    )

    return EKFResult(
        state=updated_ekf_state,
        innovation=innovation,
        innovation_covariance=innovation_covariance,
        kalman_gain=K,
        gated=gated,
    )


@jax.jit
def _ekf_update_position_heading(
    ekf_state: EKFState,
    measurement: ArrayLike,
    measurement_noise: ArrayLike,
    gate_threshold: float,
) -> EKFResult:
    """JAX-compiled EKF update for position + heading measurements."""
    # Compute measurement Jacobian (position + heading)
    H = _create_position_heading_jacobian(ekf_state.state)

    # Predicted measurement (position + heading)
    predicted_measurement = jnp.concatenate(
        [ekf_state.state[:2], jnp.array([ekf_state.state[4]])]  # position [x, y]  # heading [θ]
    )

    # Innovation (measurement residual)
    innovation = measurement - predicted_measurement

    # Wrap heading innovation to [-π, π]
    wrapped_heading_innov = jnp.arctan2(jnp.sin(innovation[2]), jnp.cos(innovation[2]))
    innovation = innovation.at[2].set(wrapped_heading_innov)

    # Innovation covariance: S = H * P * H^T + R
    innovation_covariance = H @ ekf_state.covariance @ H.T + measurement_noise

    # Mahalanobis gating
    mahalanobis_dist = mahalanobis_distance(innovation, innovation_covariance)
    gated = mahalanobis_dist > gate_threshold

    # Kalman gain: K = P * H^T * S^{-1}
    K = kalman_gain(ekf_state.covariance, H, measurement_noise)

    # State update: x = x + K * innovation (only if not gated)
    updated_state = jnp.where(gated, ekf_state.state, ekf_state.state + K @ innovation)

    # Covariance update using Joseph form for numerical stability: P = (I - K*H) @ P @ (I - K*H)^T + K @ R @ K^T
    identity = jnp.eye(8)
    I_KH = identity - K @ H
    joseph_covariance = I_KH @ ekf_state.covariance @ I_KH.T + K @ measurement_noise @ K.T
    updated_covariance = jnp.where(gated, ekf_state.covariance, joseph_covariance)

    # Log-likelihood update (only if not gated)
    log_det_S = jnp.linalg.slogdet(innovation_covariance)[1]
    measurement_dim = 3
    log_likelihood_update = -0.5 * (
        measurement_dim * jnp.log(2.0 * jnp.pi) + log_det_S + mahalanobis_dist
    )

    updated_log_likelihood = jnp.where(
        gated, ekf_state.log_likelihood, ekf_state.log_likelihood + log_likelihood_update
    )

    updated_ekf_state = EKFState(
        state=updated_state,
        covariance=updated_covariance,
        log_likelihood=updated_log_likelihood,
    )

    return EKFResult(
        state=updated_ekf_state,
        innovation=innovation,
        innovation_covariance=innovation_covariance,
        kalman_gain=K,
        gated=gated,
    )


def create_initial_ekf_state(
    initial_state: State2D,
    initial_covariance: jnp.ndarray,
) -> EKFState:
    """Create initial EKF state from State2D and covariance.

    Args:
        initial_state: Initial state estimate
        initial_covariance: Initial covariance matrix

    Returns:
        Initial EKF state
    """
    return EKFState(
        state=state_to_array(initial_state),
        covariance=initial_covariance,
        log_likelihood=0.0,
    )


@jax.jit
def ekf_step(carry: EkfCarry, inp: EkfInput) -> Tuple[EkfCarry, EkfOutputs]:
    """Functional EKF step for lax.scan forward pass.

    Args:
        carry: Current EKF state (x, P)
        inp: Input tuple (meas_struct, dt, imu_block, filter_cfg)

    Returns:
        Tuple of (new_carry, outputs) where:
        - new_carry: Updated EKF state after filtering
        - outputs: Filtered and predicted states/covariances
    """
    x, P = carry
    meas_struct, dt, imu_block, filter_cfg = inp

    # Extract filter configuration
    velocity_damping = filter_cfg.get("velocity_damping", 0.1)
    accel_noise_std = filter_cfg.get("accel_noise_std", 0.5)
    gyro_noise_std = filter_cfg.get("gyro_noise_std", 0.1)
    bias_drift_std = filter_cfg.get("bias_drift_std", 0.01)
    position_noise_std = filter_cfg.get("position_noise_std", 1.0)
    heading_noise_std = filter_cfg.get("heading_noise_std", 0.1)
    gate_threshold = filter_cfg.get("gate_threshold", 9.21)

    # Prediction step
    # Extract IMU measurements (imu_block is guaranteed to be a valid array in our pipeline)
    # If missing data is passed, it will be zeros which is handled correctly
    accel = imu_block[:2]  # [ax, ay]
    gyro = imu_block[2:]  # [gz]

    # Predict state using existing JAX function
    x_pred = _predict_state_jax(x, dt, accel, gyro, velocity_damping)

    # Predict covariance using linearized dynamics
    process_noise = compute_process_noise(dt, accel_noise_std, gyro_noise_std, bias_drift_std)
    P_pred = predict_covariance(P, x, dt, accel, gyro, velocity_damping, process_noise)

    # Measurement update step
    # Extract measurements from meas_struct
    position = meas_struct.get("position")
    heading = meas_struct.get("heading")
    confidence = meas_struct.get("confidence", 1.0)

    # Create EKF state for measurement update
    pred_ekf_state = EKFState(state=x_pred, covariance=P_pred, log_likelihood=0.0)

    # Skip update if no measurements available
    if position is None and heading is None:
        # No measurement update - return prediction
        x_filt = x_pred
        P_filt = P_pred
    else:
        # Create measurement vector and noise
        if position is not None and heading is not None:
            measurement = jnp.concatenate([position, jnp.array([heading])])
            has_heading = True
        elif position is not None:
            measurement = position
            has_heading = False
        else:
            # Heading-only case (rare)
            dummy_position = x_pred[:2]
            measurement = jnp.concatenate([dummy_position, jnp.array([heading])])
            has_heading = True
            confidence = 0.01  # Very low confidence for dummy position

        # Create measurement noise matrix
        measurement_noise = create_measurement_noise(
            position_noise_std,
            confidence,
            has_heading,
            heading_noise_std if has_heading else None,
        )

        # Perform update
        result = ekf_update(
            pred_ekf_state,
            measurement,
            measurement_noise,
            has_heading,
            gate_threshold,
        )

        # Extract filtered state and covariance
        x_filt = result.state.state
        P_filt = result.state.covariance

    # Create outputs
    outputs = EkfOutputs(
        x_filt=x_filt,
        P_filt=P_filt,
        x_pred=x_pred,
        P_pred=P_pred,
    )

    # Create new carry state
    new_carry = EkfCarry(x=x_filt, P=P_filt)

    return new_carry, outputs


# JAX-compatible measurement structure for lax.scan
class MeasurementArrays(NamedTuple):
    """Structured measurement arrays for JAX lax.scan compatibility.

    All measurements use NaN to indicate missing values, and masks indicate validity.
    """

    positions: ArrayLike  # Shape (n_frames, 2) - [x, y] positions
    headings: ArrayLike  # Shape (n_frames,) - heading angles
    confidences: ArrayLike  # Shape (n_frames,) - confidence values
    position_mask: ArrayLike  # Shape (n_frames,) - True if position valid
    heading_mask: ArrayLike  # Shape (n_frames,) - True if heading valid


@jax.jit
def ekf_step_functional(
    carry: EkfCarry,
    scan_inputs: EkfScanInputs,
    frame_idx: int,
) -> Tuple[EkfCarry, EkfOutputs]:
    """Functional EKF step using PyTree inputs.

    Args:
        carry: Current EKF state (x, P)
        scan_inputs: PyTree dataclass containing all scan inputs
        frame_idx: Current frame index

    Returns:
        Tuple of (new_carry, outputs)
    """
    x, P = carry

    # Extract frame-specific data
    position = scan_inputs.positions[frame_idx]
    heading = scan_inputs.headings[frame_idx]
    confidence = scan_inputs.confidences[frame_idx]
    pos_valid = scan_inputs.position_valid[frame_idx]
    head_valid = scan_inputs.heading_valid[frame_idx]
    imu_block = scan_inputs.imu_blocks[frame_idx]
    dt = scan_inputs.dt[frame_idx]

    # Prediction step
    accel = imu_block[:2]  # [ax, ay]
    gyro = imu_block[2:]  # [gz]

    x_pred = _predict_state_jax(x, dt, accel, gyro, scan_inputs.velocity_damping)

    # Predict covariance using linearized dynamics
    process_noise = compute_process_noise(
        dt, scan_inputs.accel_noise_std, scan_inputs.gyro_noise_std, scan_inputs.bias_drift_std
    )
    P_pred = predict_covariance(P, x, dt, accel, gyro, scan_inputs.velocity_damping, process_noise)

    # Measurement update
    x_filt, P_filt = jax.lax.cond(
        pos_valid,
        lambda: _functional_measurement_update(
            x_pred, P_pred, position, heading, confidence, pos_valid, head_valid, scan_inputs
        ),
        lambda: (x_pred, P_pred),  # No update if no position measurement
    )

    # Create outputs
    outputs = EkfOutputs(x_filt=x_filt, P_filt=P_filt, x_pred=x_pred, P_pred=P_pred)
    new_carry = EkfCarry(x=x_filt, P=P_filt)

    return new_carry, outputs


@jax.jit
def _functional_measurement_update(
    x_pred: ArrayLike,
    P_pred: ArrayLike,
    position: ArrayLike,
    heading: float,
    confidence: float,
    pos_valid: bool,
    head_valid: bool,
    scan_inputs: EkfScanInputs,
) -> Tuple[Array, Array]:
    """Perform measurement update step."""
    # Create measurement vector - always use 3D format [x, y, heading]
    measurement = jnp.array(
        [
            position[0],  # x position
            position[1],  # y position
            jnp.where(head_valid, heading, x_pred[4]),  # heading (or prediction if invalid)
        ]
    )

    # Create measurement noise - large noise for invalid measurements
    c = jnp.clip(confidence, 1e-3, 1.0)
    pos_noise_var = (scan_inputs.position_noise_std / c) ** 2
    noise_diag = jnp.array(
        [
            pos_noise_var,  # x position noise
            pos_noise_var,  # y position noise
            jnp.where(head_valid, scan_inputs.heading_noise_std**2, 1e6),  # heading noise
        ]
    )
    R = jnp.diag(noise_diag)

    # Measurement Jacobian for [x, y, heading]
    H = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x position
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y position
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # heading
        ]
    )

    # Predicted measurement
    h_pred = H @ x_pred

    # Innovation with angle wrapping
    innovation = measurement - h_pred
    innovation = innovation.at[2].set(wrap_angle(innovation[2]))

    # Innovation covariance and Kalman gain
    S = H @ P_pred @ H.T + R
    K = kalman_gain(P_pred, H, R)

    # State and covariance updates
    x_update = x_pred + K @ innovation
    I_KH = jnp.eye(8) - K @ H
    P_update = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

    return x_update, P_update


def create_functional_scan_inputs(
    positions: ArrayLike,
    headings: ArrayLike,
    confidences: ArrayLike,
    position_valid: ArrayLike,
    heading_valid: ArrayLike,
    imu_blocks: ArrayLike,
    dt: ArrayLike,
    velocity_damping: float,
    accel_noise_std: float,
    gyro_noise_std: float,
    bias_drift_std: float,
    position_noise_std: float,
    heading_noise_std: float,
    gate_threshold: float,
) -> EkfScanInputs:
    """Create PyTree scan inputs from arrays.

    Args:
        positions: Position measurements (n_frames, 2)
        headings: Heading measurements (n_frames,)
        confidences: Measurement confidences (n_frames,)
        position_valid: Position validity mask (n_frames,)
        heading_valid: Heading validity mask (n_frames,)
        imu_blocks: IMU measurements (n_frames, 3)
        dt: Time deltas (n_frames,)
        velocity_damping: Velocity damping coefficient
        accel_noise_std: Accelerometer noise std
        gyro_noise_std: Gyroscope noise std
        bias_drift_std: Bias drift std
        position_noise_std: Position noise std
        heading_noise_std: Heading noise std
        gate_threshold: Gating threshold

    Returns:
        EkfScanInputs PyTree dataclass
    """
    return EkfScanInputs(
        positions=positions,
        headings=headings,
        confidences=confidences,
        position_valid=position_valid,
        heading_valid=heading_valid,
        imu_blocks=imu_blocks,
        dt=dt,
        velocity_damping=velocity_damping,
        accel_noise_std=accel_noise_std,
        gyro_noise_std=gyro_noise_std,
        bias_drift_std=bias_drift_std,
        position_noise_std=position_noise_std,
        heading_noise_std=heading_noise_std,
        gate_threshold=gate_threshold,
    )


@jax.jit
def ekf_step_pytree(
    carry: EkfCarry,
    scan_input: Tuple[
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
) -> Tuple[EkfCarry, EkfOutputs]:
    """Functional EKF step that accepts frame-wise PyTree inputs for lax.scan.

    Args:
        carry: Current EKF state (x, P)
        scan_input: Tuple of frame-wise inputs (position, heading, confidence, pos_valid,
                   head_valid, imu_block, dt, + filter params)

    Returns:
        Tuple of (new_carry, outputs)
    """
    x, P = carry
    (
        position,
        heading,
        confidence,
        pos_valid,
        head_valid,
        imu_block,
        dt,
        velocity_damping,
        accel_noise_std,
        gyro_noise_std,
        bias_drift_std,
        position_noise_std,
        heading_noise_std,
        gate_threshold,
    ) = scan_input

    # Prediction step
    accel = imu_block[:2]  # [ax, ay]
    gyro = imu_block[2:]  # [gz]

    x_pred = _predict_state_jax(x, dt, accel, gyro, velocity_damping)

    # Predict covariance using linearized dynamics
    process_noise = compute_process_noise(dt, accel_noise_std, gyro_noise_std, bias_drift_std)
    P_pred = predict_covariance(P, x, dt, accel, gyro, velocity_damping, process_noise)

    # Measurement update
    x_filt, P_filt = jax.lax.cond(
        pos_valid,
        lambda: _pytree_measurement_update(
            x_pred,
            P_pred,
            position,
            heading,
            confidence,
            head_valid,
            position_noise_std,
            heading_noise_std,
        ),
        lambda: (x_pred, P_pred),  # No update if no position measurement
    )

    # Create outputs
    outputs = EkfOutputs(x_filt=x_filt, P_filt=P_filt, x_pred=x_pred, P_pred=P_pred)
    new_carry = EkfCarry(x=x_filt, P=P_filt)

    return new_carry, outputs


@jax.jit
def _pytree_measurement_update(
    x_pred: ArrayLike,
    P_pred: ArrayLike,
    position: ArrayLike,
    heading: float,
    confidence: float,
    head_valid: bool,
    position_noise_std: float,
    heading_noise_std: float,
) -> Tuple[Array, Array]:
    """Perform measurement update step for PyTree version."""
    # Create measurement vector - always use 3D format [x, y, heading]
    measurement = jnp.array(
        [
            position[0],  # x position
            position[1],  # y position
            jnp.where(head_valid, heading, x_pred[4]),  # heading (or prediction if invalid)
        ]
    )

    # Create measurement noise - large noise for invalid measurements
    c = jnp.clip(confidence, 1e-3, 1.0)
    pos_noise_var = (position_noise_std / c) ** 2
    noise_diag = jnp.array(
        [
            pos_noise_var,  # x position noise
            pos_noise_var,  # y position noise
            jnp.where(head_valid, heading_noise_std**2, 1e6),  # heading noise
        ]
    )
    R = jnp.diag(noise_diag)

    # Measurement Jacobian for [x, y, heading]
    H = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x position
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y position
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # heading
        ]
    )

    # Predicted measurement
    h_pred = H @ x_pred

    # Innovation with angle wrapping
    innovation = measurement - h_pred
    innovation = innovation.at[2].set(wrap_angle(innovation[2]))

    # Innovation covariance and Kalman gain
    S = H @ P_pred @ H.T + R
    K = kalman_gain(P_pred, H, R)

    # State and covariance updates
    x_update = x_pred + K @ innovation
    I_KH = jnp.eye(8) - K @ H
    P_update = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

    return x_update, P_update


def ekf_step_arrays(
    carry: EkfCarry,
    inp: Tuple[
        ArrayLike,
        float,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
) -> Tuple[EkfCarry, EkfOutputs]:
    """JAX-compatible EKF step for lax.scan using structured arrays.

    NOTE: Not JIT-compiled to allow for static argument optimization at call site.
    Use ekf_step_arrays_pure() for pure JIT-compiled version or
    ekf_step_arrays_optimized() for version with static filter parameters.
    """
    return ekf_step_arrays_pure(carry, inp)


def create_ekf_step_arrays_optimized(
    velocity_damping: float,
    accel_noise_std: float,
    gyro_noise_std: float,
    bias_drift_std: float,
    position_noise_std: float,
    heading_noise_std: float,
    gate_threshold: float,
):
    """Create optimized EKF step function with static filter parameters.

    This creates a JIT-compiled function with filter parameters as static arguments,
    providing optimal performance by eliminating redundant parameter passing.

    Args:
        velocity_damping: Velocity damping coefficient
        accel_noise_std: Accelerometer noise std
        gyro_noise_std: Gyroscope noise std
        bias_drift_std: Bias drift std
        position_noise_std: Position noise std
        heading_noise_std: Heading noise std
        gate_threshold: Gating threshold

    Returns:
        JIT-compiled EKF step function with static parameters
    """

    @jax.jit
    def ekf_step_arrays_optimized(
        carry: EkfCarry,
        inp: Tuple[
            jnp.ndarray, float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
        ],
    ) -> Tuple[EkfCarry, EkfOutputs]:
        """Optimized EKF step with static filter parameters.

        Args:
            carry: Current EKF state (x, P)
            inp: Input tuple (position, dt, imu_block, heading, confidence, pos_mask, head_mask)

        Returns:
            Tuple of (new_carry, outputs)
        """
        x, P = carry
        position, dt, imu_block, heading, confidence, pos_mask, head_mask = inp

        # Prediction step
        accel = imu_block[:2]  # [ax, ay]
        gyro = imu_block[2:]  # [gz]

        x_pred = _predict_state_jax(x, dt, accel, gyro, velocity_damping)

        # Predict covariance using linearized dynamics
        process_noise = compute_process_noise(dt, accel_noise_std, gyro_noise_std, bias_drift_std)
        P_pred = predict_covariance(P, x, dt, accel, gyro, velocity_damping, process_noise)

        # Default to prediction (no measurement update)
        x_filt = x_pred
        P_filt = P_pred

        # Check if we have any valid measurements
        has_position = pos_mask
        has_heading = head_mask

        # Create a full measurement vector (always 3 elements: [x, y, heading])
        measurement = jnp.array(
            [
                jnp.where(has_position, position[0], x_pred[0]),  # x position
                jnp.where(has_position, position[1], x_pred[1]),  # y position
                jnp.where(has_heading, heading, x_pred[4]),  # heading
            ]
        )

        # Apply measurement update only if we have position measurements
        def apply_measurement_update():
            # Create measurement noise - scale by confidence
            c = jnp.clip(confidence, 1e-3, 1.0)
            pos_noise_var = (position_noise_std / c) ** 2

            # Always use 3D measurement format: [x, y, heading]
            noise_diag = jnp.array(
                [
                    jnp.where(has_position, pos_noise_var, 1e6),  # Large noise for missing position
                    jnp.where(has_position, pos_noise_var, 1e6),
                    jnp.where(
                        has_heading, heading_noise_std**2, 1e6
                    ),  # Large noise for missing heading
                ]
            )
            measurement_noise_matrix = jnp.diag(noise_diag)

            # Measurement function: h(x) = [x[0], x[1], x[4]] (position + heading)
            H = jnp.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x position
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y position
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # heading
                ]
            )

            # Predicted measurement
            h_pred = H @ x_pred  # [x, y, theta]

            # Innovation (residual)
            innovation = measurement - h_pred

            # Wrap heading innovation to [-π, π]
            innovation = innovation.at[2].set(wrap_angle(innovation[2]))

            # Innovation covariance
            S = H @ P_pred @ H.T + measurement_noise_matrix

            # Kalman gain using stable solver
            K = kalman_gain(P_pred, H, measurement_noise_matrix)

            # State update
            x_update = x_pred + K @ innovation

            # Covariance update (Joseph form for numerical stability)
            I_KH = jnp.eye(8) - K @ H
            P_update = I_KH @ P_pred @ I_KH.T + K @ measurement_noise_matrix @ K.T

            return x_update, P_update

        def no_measurement_update():
            return x_pred, P_pred

        # Use conditional execution for JAX compatibility
        x_filt, P_filt = jax.lax.cond(has_position, apply_measurement_update, no_measurement_update)

        # Create outputs
        outputs = EkfOutputs(
            x_filt=x_filt,
            P_filt=P_filt,
            x_pred=x_pred,
            P_pred=P_pred,
        )

        # Create new carry state
        new_carry = EkfCarry(x=x_filt, P=P_filt)

        return new_carry, outputs

    return ekf_step_arrays_optimized


@jax.jit
def ekf_step_arrays_pure(
    carry: EkfCarry,
    inp: Tuple[
        ArrayLike,
        float,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
) -> Tuple[EkfCarry, EkfOutputs]:
    """Pure JAX-compiled EKF step for lax.scan using structured arrays.

    This is the core computational kernel that should be JIT-compiled.
    All parameters are explicit to enable optimal caching and avoid closures.

    Args:
        carry: Current EKF state (x, P)
        inp: Input tuple (position, dt, imu_block, heading, confidence, pos_mask, head_mask,
                         velocity_damping, accel_noise_std, gyro_noise_std, bias_drift_std,
                         position_noise_std, heading_noise_std, gate_threshold)

    Returns:
        Tuple of (new_carry, outputs)
    """
    x, P = carry
    (
        position,
        dt,
        imu_block,
        heading,
        confidence,
        pos_mask,
        head_mask,
        velocity_damping,
        accel_noise_std,
        gyro_noise_std,
        bias_drift_std,
        position_noise_std,
        heading_noise_std,
        gate_threshold,
    ) = inp

    # Prediction step
    # Extract IMU measurements
    accel = imu_block[:2]  # [ax, ay]
    gyro = imu_block[2:]  # [gz]

    # Predict state using existing JAX function
    x_pred = _predict_state_jax(x, dt, accel, gyro, velocity_damping)

    # Predict covariance using linearized dynamics
    process_noise = compute_process_noise(dt, accel_noise_std, gyro_noise_std, bias_drift_std)
    P_pred = predict_covariance(P, x, dt, accel, gyro, velocity_damping, process_noise)

    # Default to prediction (no measurement update)
    x_filt = x_pred
    P_filt = P_pred

    # Check if we have any valid measurements
    has_position = pos_mask
    has_heading = head_mask

    # Perform measurement update only if we have measurements
    # For simplicity, we'll handle position-only and position+heading cases
    # and skip the more complex heading-only case for now

    # Create a full measurement vector (always 3 elements: [x, y, heading])
    # Use the actual values if available, otherwise use state prediction as placeholder
    measurement = jnp.array(
        [
            jnp.where(has_position, position[0], x_pred[0]),  # x position
            jnp.where(has_position, position[1], x_pred[1]),  # y position
            jnp.where(has_heading, heading, x_pred[4]),  # heading
        ]
    )

    # Apply measurement update only if we have position measurements
    def apply_measurement_update():
        # Create measurement noise - scale by confidence
        c = jnp.clip(confidence, 1e-3, 1.0)
        pos_noise_var = (position_noise_std / c) ** 2

        # Always use 3D measurement format: [x, y, heading]
        # For missing measurements, noise is made very large to minimize impact
        noise_diag = jnp.array(
            [
                jnp.where(has_position, pos_noise_var, 1e6),  # Large noise for missing position
                jnp.where(has_position, pos_noise_var, 1e6),
                jnp.where(
                    has_heading, heading_noise_std**2, 1e6
                ),  # Large noise for missing heading
            ]
        )
        measurement_noise_matrix = jnp.diag(noise_diag)

        # Measurement function: h(x) = [x[0], x[1], x[4]] (position + heading)
        H = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x position
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y position
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # heading
            ]
        )

        # Predicted measurement
        h_pred = H @ x_pred  # [x, y, theta]

        # Innovation (residual)
        innovation = measurement - h_pred

        # Wrap heading innovation to [-π, π]
        innovation = innovation.at[2].set(wrap_angle(innovation[2]))

        # Innovation covariance
        S = H @ P_pred @ H.T + measurement_noise_matrix

        # Kalman gain using stable solver
        K = kalman_gain(P_pred, H, measurement_noise_matrix)

        # State update
        x_update = x_pred + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = jnp.eye(8) - K @ H
        P_update = I_KH @ P_pred @ I_KH.T + K @ measurement_noise_matrix @ K.T

        return x_update, P_update

    def no_measurement_update():
        return x_pred, P_pred

    # Use conditional execution for JAX compatibility
    x_filt, P_filt = jax.lax.cond(has_position, apply_measurement_update, no_measurement_update)

    # Create outputs
    outputs = EkfOutputs(
        x_filt=x_filt,
        P_filt=P_filt,
        x_pred=x_pred,
        P_pred=P_pred,
    )

    # Create new carry state
    new_carry = EkfCarry(x=x_filt, P=P_filt)

    return new_carry, outputs


class EKFFilter:
    """Extended Kalman Filter for 2D tracking.

    This class provides a stateful interface to the EKF algorithm with
    configuration management and measurement processing.
    """

    def __init__(
        self,
        initial_state: State2D,
        initial_covariance: ArrayLike,
        velocity_damping: ArrayLike = 0.1,
        accel_noise_std: ArrayLike = 0.5,
        gyro_noise_std: ArrayLike = 0.1,
        bias_drift_std: ArrayLike = 0.01,
        position_noise_std: ArrayLike = 1.0,
        heading_noise_std: ArrayLike = 0.1,
        gate_threshold: ArrayLike = 9.21,
    ):
        """Initialize EKF filter.

        Args:
            initial_state: Initial state estimate
            initial_covariance: Initial covariance matrix
            velocity_damping: Velocity damping coefficient λ
            accel_noise_std: Accelerometer noise std dev (m/s²)
            gyro_noise_std: Gyroscope noise std dev (rad/s)
            bias_drift_std: Bias drift std dev (per √s)
            position_noise_std: Position measurement noise std dev (cm)
            heading_noise_std: Heading measurement noise std dev (rad)
            gate_threshold: Chi-squared threshold for measurement gating
        """
        self.ekf_state = create_initial_ekf_state(initial_state, initial_covariance)

        # Process noise parameters
        self.velocity_damping = velocity_damping
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.bias_drift_std = bias_drift_std

        # Measurement noise parameters
        self.position_noise_std = position_noise_std
        self.heading_noise_std = heading_noise_std

        # Gating threshold
        self.gate_threshold = gate_threshold

    def predict(
        self,
        dt: float,
        accel: ArrayLike,
        gyro: ArrayLike,
    ) -> None:
        """Perform prediction step.

        Args:
            dt: Time step (seconds)
            accel: Accelerometer measurement [ax, ay] (m/s²)
            gyro: Gyroscope measurement [gz] (rad/s)
        """
        self.ekf_state = ekf_predict(
            self.ekf_state,
            dt,
            accel,
            gyro,
            self.velocity_damping,
            self.accel_noise_std,
            self.gyro_noise_std,
            self.bias_drift_std,
        )

    def update(
        self,
        position: Optional[ArrayLike] = None,
        heading: Optional[float] = None,
        confidence: float = 1.0,
    ) -> EKFResult:
        """Perform measurement update step.

        Args:
            position: Position measurement [x, y] in cm (None if missing)
            heading: Heading measurement in radians (None if missing)
            confidence: Detection confidence [0, 1]

        Returns:
            EKF update result
        """
        # Skip update if no measurements available
        if position is None and heading is None:
            # Return no-update result
            return EKFResult(
                state=self.ekf_state,
                innovation=jnp.array([]),
                innovation_covariance=jnp.array([[]]),
                kalman_gain=jnp.array([[]]),
                gated=False,
            )

        # Create measurement vector
        if position is not None and heading is not None:
            measurement = jnp.concatenate([position, jnp.array([heading])])
            has_heading = True
        elif position is not None:
            measurement = position
            has_heading = False
        else:
            # Heading-only measurement (rare case)
            # Create dummy position measurement with high noise
            dummy_position = self.ekf_state.state[:2]
            measurement = jnp.concatenate([dummy_position, jnp.array([heading])])
            has_heading = True
            confidence = 0.01  # Very low confidence for dummy position

        # Create measurement noise matrix
        measurement_noise = create_measurement_noise(
            self.position_noise_std,
            confidence,
            has_heading,
            self.heading_noise_std if has_heading else None,
        )

        # Perform update
        result = ekf_update(
            self.ekf_state,
            measurement,
            measurement_noise,
            has_heading,
            self.gate_threshold,
        )

        # Update internal state if not gated
        if not result.gated:
            self.ekf_state = result.state

        return result

    def get_current_state(self) -> State2D:
        """Get current state estimate as State2D object.

        Returns:
            Current state estimate
        """
        return array_to_state(self.ekf_state.state)

    def get_current_covariance(self) -> Array:
        """Get current covariance matrix.

        Returns:
            Current covariance matrix
        """
        return self.ekf_state.covariance

    def get_log_likelihood(self) -> float:
        """Get cumulative log-likelihood.

        Returns:
            Cumulative log-likelihood
        """
        return float(self.ekf_state.log_likelihood)
