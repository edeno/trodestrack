"""Rauch-Tung-Striebel (RTS) smoother implementation for 2D tracking.

This module implements the RTS backward-pass smoothing algorithm to optimize
state estimates by incorporating future measurements. The smoother provides
improved accuracy over forward-only filtering by using information from the
entire time series.

Key features:
- JAX-compiled backward pass for performance
- Compatible with EKF forward pass results
- Handles missing measurements and gated observations
- Provides uncertainty estimates for smoothed states
"""

from typing import List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from ._solvers import safe_solve
from .ekf import EKFResult


class RTSResult(NamedTuple):
    """Result from RTS smoothing operation.

    Attributes:
        smoothed_states: JAX array of smoothed state estimates [N, state_dim]
        smoothed_covariances: JAX array of smoothed covariance matrices [N, state_dim, state_dim]
        log_likelihood: Total log-likelihood of the sequence
    """

    smoothed_states: jnp.ndarray
    smoothed_covariances: jnp.ndarray
    log_likelihood: float


class ForwardPassData(NamedTuple):
    """Data from forward pass needed for RTS smoothing.

    Attributes:
        filtered_states: JAX array of states after measurement updates [N, state_dim]
        filtered_covariances: JAX array of covariances after measurement updates [N, state_dim, state_dim]
        predicted_states: JAX array of states after prediction steps [N, state_dim]
        predicted_covariances: JAX array of covariances after prediction steps [N, state_dim, state_dim]
        log_likelihood: Cumulative log-likelihood
    """

    filtered_states: jnp.ndarray
    filtered_covariances: jnp.ndarray
    predicted_states: jnp.ndarray
    predicted_covariances: jnp.ndarray
    log_likelihood: float


@jax.jit
def rts_backward_step(
    x_s_next: jnp.ndarray,
    P_s_next: jnp.ndarray,
    x_f: jnp.ndarray,
    P_f: jnp.ndarray,
    x_p_next: jnp.ndarray,
    P_p_next: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single RTS backward step.

    Computes the smoothed estimate at time k given:
    - Smoothed estimate at time k+1
    - Filtered estimate at time k
    - Predicted estimate at time k+1

    Args:
        x_s_next: Smoothed state at time k+1
        P_s_next: Smoothed covariance at time k+1
        x_f: Filtered state at time k
        P_f: Filtered covariance at time k
        x_p_next: Predicted state at time k+1 (from k)
        P_p_next: Predicted covariance at time k+1 (from k)

    Returns:
        Tuple of (smoothed_state, smoothed_covariance) at time k
    """
    # Compute gain matrix: G_k = P_f_k @ F_k^T @ P_p_{k+1}^{-1}
    # Since we don't store F_k explicitly, use the relationship:
    # P_p_{k+1} = F_k @ P_f_k @ F_k^T + Q_k
    # We can solve for the gain using the predicted covariance

    # Smoother gain: G = P_f @ P_p_next^{-1} using safe solve
    G = safe_solve(P_p_next, P_f.T).T

    # Smoothed state: x_s_k = x_f_k + G_k @ (x_s_{k+1} - x_p_{k+1})
    x_s = x_f + G @ (x_s_next - x_p_next)

    # Smoothed covariance: P_s_k = P_f_k + G_k @ (P_s_{k+1} - P_p_{k+1}) @ G_k^T
    P_s = P_f + G @ (P_s_next - P_p_next) @ G.T

    return x_s, P_s


def rts_smooth(
    forward_data: ForwardPassData,
) -> RTSResult:
    """Perform RTS smoothing on forward pass results.

    The RTS smoother runs a backward pass through the filtered estimates,
    incorporating information from future measurements to improve past estimates.

    Args:
        forward_data: Results from forward filtering pass

    Returns:
        RTSResult with smoothed states and covariances
    """
    N = forward_data.filtered_states.shape[0]

    if N == 0:
        # Return empty arrays with correct shapes
        state_dim = 8  # trodestrack uses 8-dimensional state
        return RTSResult(
            smoothed_states=jnp.array([]).reshape(0, state_dim),
            smoothed_covariances=jnp.array([]).reshape(0, state_dim, state_dim),
            log_likelihood=forward_data.log_likelihood
        )

    # Data is already JAX arrays from ForwardPassData
    filtered_states = forward_data.filtered_states
    filtered_covariances = forward_data.filtered_covariances
    predicted_states = forward_data.predicted_states
    predicted_covariances = forward_data.predicted_covariances

    # Initialize output arrays (JAX-compatible)
    smoothed_states = jnp.zeros_like(filtered_states)
    smoothed_covariances = jnp.zeros_like(filtered_covariances)

    # Initialize backward pass with final filtered estimate
    smoothed_states = smoothed_states.at[N - 1].set(filtered_states[N - 1])
    smoothed_covariances = smoothed_covariances.at[N - 1].set(filtered_covariances[N - 1])

    # Backward pass: smooth from k = N-2 down to 0 using lax.scan
    def backward_step_fn(carry, inputs):
        """Single backward step for lax.scan."""
        x_s_next, P_s_next = carry
        x_f, P_f, x_p_next, P_p_next = inputs

        # Perform backward step
        x_s, P_s = rts_backward_step(x_s_next, P_s_next, x_f, P_f, x_p_next, P_p_next)

        return (x_s, P_s), (x_s, P_s)

    # Prepare inputs for scan (forward order, reverse=True will handle backward iteration)
    scan_inputs = (
        filtered_states[:-1],  # x_f from 0 to N-2
        filtered_covariances[:-1],  # P_f from 0 to N-2
        predicted_states[1:],  # x_p_next from 1 to N-1
        predicted_covariances[1:],  # P_p_next from 1 to N-1
    )

    # Initial carry state (final filtered estimate)
    init_carry = (smoothed_states[N - 1], smoothed_covariances[N - 1])

    # Run backward scan using reverse=True (much cleaner!)
    final_carry, backward_outputs = lax.scan(
        backward_step_fn, init_carry, scan_inputs, reverse=True
    )

    # Extract results (already in correct forward order due to reverse=True)
    backward_states, backward_covariances = backward_outputs
    smoothed_states = smoothed_states.at[:-1].set(backward_states)
    smoothed_covariances = smoothed_covariances.at[:-1].set(backward_covariances)

    return RTSResult(
        smoothed_states=smoothed_states,
        smoothed_covariances=smoothed_covariances,
        log_likelihood=forward_data.log_likelihood,
    )


class RTSSmoother:
    """RTS smoother for batch processing of measurement sequences.

    This class provides a high-level interface for RTS smoothing, managing
    the forward filtering pass and backward smoothing pass automatically.
    """

    def __init__(
        self,
        velocity_damping: float = 0.1,
        accel_noise_std: float = 0.5,
        gyro_noise_std: float = 0.1,
        bias_drift_std: float = 0.01,
    ):
        """Initialize RTS smoother.

        Args:
            velocity_damping: Velocity damping coefficient λ
            accel_noise_std: Accelerometer noise std dev (m/s²)
            gyro_noise_std: Gyroscope noise std dev (rad/s)
            bias_drift_std: Bias drift std dev (per √s)
        """
        self.velocity_damping = velocity_damping
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.bias_drift_std = bias_drift_std

    def collect_forward_data(
        self,
        ekf_results: List[EKFResult],
        prediction_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    ) -> ForwardPassData:
        """Collect data from forward pass for smoothing.

        Args:
            ekf_results: Results from EKF forward pass
            prediction_data: List of (predicted_state, predicted_covariance) tuples

        Returns:
            ForwardPassData suitable for RTS smoothing
        """
        if len(ekf_results) != len(prediction_data):
            raise ValueError(
                f"Mismatch between EKF results ({len(ekf_results)}) "
                f"and prediction data ({len(prediction_data)})"
            )

        filtered_states = [result.state.state for result in ekf_results]
        filtered_covariances = [result.state.covariance for result in ekf_results]
        predicted_states = [pred[0] for pred in prediction_data]
        predicted_covariances = [pred[1] for pred in prediction_data]

        # Total log-likelihood is from the final EKF result
        log_likelihood = ekf_results[-1].state.log_likelihood if ekf_results else 0.0

        return ForwardPassData(
            filtered_states=jnp.array(filtered_states),
            filtered_covariances=jnp.array(filtered_covariances),
            predicted_states=jnp.array(predicted_states),
            predicted_covariances=jnp.array(predicted_covariances),
            log_likelihood=log_likelihood,
        )

    def smooth_sequence(
        self,
        forward_data: ForwardPassData,
    ) -> RTSResult:
        """Smooth a sequence of filtered estimates.

        Args:
            forward_data: Data from forward filtering pass

        Returns:
            RTSResult with smoothed estimates
        """
        return rts_smooth(forward_data)


def compute_smoothing_improvement(
    filtered_states: List[jnp.ndarray],
    smoothed_states: List[jnp.ndarray],
    ground_truth: List[jnp.ndarray],
) -> Tuple[float, float, float]:
    """Compute RMSE improvement from smoothing.

    Args:
        filtered_states: Forward-only filtered states
        smoothed_states: RTS smoothed states
        ground_truth: True states for comparison

    Returns:
        Tuple of (filtered_rmse, smoothed_rmse, improvement_pct)
    """
    if len(filtered_states) != len(smoothed_states) or len(filtered_states) != len(ground_truth):
        raise ValueError("All state lists must have the same length")

    # Compute position RMSE for filtered estimates
    filtered_errors = []
    for filt, truth in zip(filtered_states, ground_truth):
        pos_error = jnp.linalg.norm(filt[:2] - truth[:2])  # Position only
        filtered_errors.append(pos_error)

    # Compute position RMSE for smoothed estimates
    smoothed_errors = []
    for smooth, truth in zip(smoothed_states, ground_truth):
        pos_error = jnp.linalg.norm(smooth[:2] - truth[:2])  # Position only
        smoothed_errors.append(pos_error)

    filtered_rmse = jnp.sqrt(jnp.mean(jnp.array(filtered_errors) ** 2))
    smoothed_rmse = jnp.sqrt(jnp.mean(jnp.array(smoothed_errors) ** 2))

    improvement_pct = (filtered_rmse - smoothed_rmse) / filtered_rmse * 100.0

    return float(filtered_rmse), float(smoothed_rmse), float(improvement_pct)
