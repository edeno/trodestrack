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
from jax import Array, lax
from jax.typing import ArrayLike

from ._solvers import safe_solve, _symmetrize_and_stabilize
from .ekf import EKFResult


class RTSResult(NamedTuple):
    """Result from RTS smoothing operation.

    Attributes:
        smoothed_states: JAX array of smoothed state estimates [N, state_dim]
        smoothed_covariances: JAX array of smoothed covariance matrices [N, state_dim, state_dim]
        log_likelihood: Total log-likelihood of the sequence
    """

    smoothed_states: Array
    smoothed_covariances: Array
    log_likelihood: float


class ForwardPassData(NamedTuple):
    """Data from forward pass needed for RTS smoothing.

    Attributes:
        filtered_states: JAX array of states after measurement updates [N, state_dim]
        filtered_covariances: JAX array of covariances after measurement updates [N, state_dim, state_dim]
        predicted_states: JAX array of states after prediction steps [N, state_dim]
        predicted_covariances: JAX array of covariances after prediction steps [N, state_dim, state_dim]
        transition_matrices: JAX array of state transition matrices [N, state_dim, state_dim]
        log_likelihood: Cumulative log-likelihood
    """

    filtered_states: Array
    filtered_covariances: Array
    predicted_states: Array
    predicted_covariances: Array
    transition_matrices: Array
    log_likelihood: float


@jax.jit
def rts_backward_step(
    x_s_next: ArrayLike,
    P_s_next: ArrayLike,
    x_f: ArrayLike,
    P_f: ArrayLike,
    x_p_next: ArrayLike,
    P_p_next: ArrayLike,
    F: ArrayLike,
) -> Tuple[Array, Array]:
    """Single RTS backward step.

    Computes the smoothed estimate at time k given:
    - Smoothed estimate at time k+1
    - Filtered estimate at time k
    - Predicted estimate at time k+1
    - Transition matrix F from k to k+1

    Args:
        x_s_next: Smoothed state at time k+1
        P_s_next: Smoothed covariance at time k+1
        x_f: Filtered state at time k
        P_f: Filtered covariance at time k
        x_p_next: Predicted state at time k+1 (from k)
        P_p_next: Predicted covariance at time k+1 (from k)
        F: State transition matrix from k to k+1

    Returns:
        Tuple of (smoothed_state, smoothed_covariance) at time k
    """
    # Correct smoother gain formula: G_k = P_f_k @ F_k^T @ P_p_{k+1}^{-1}
    # This is equivalent to: G = (F @ P_f)^T @ P_p_next^{-1} = P_f @ F^T @ P_p_next^{-1}
    G = safe_solve(P_p_next, (F @ P_f).T).T

    # Smoothed state: x_s_k = x_f_k + G_k @ (x_s_{k+1} - x_p_{k+1})
    x_s = x_f + G @ (x_s_next - x_p_next)

    # Smoothed covariance: P_s_k = P_f_k + G_k @ (P_s_{k+1} - P_p_{k+1}) @ G_k^T
    # Add symmetrization for numerical stability (Joseph-form equivalent for smoothing)
    P_s = _symmetrize_and_stabilize(
        P_f + G @ (P_s_next - P_p_next) @ G.T
    )

    return x_s, P_s


def rts_smooth(
    forward_data: ForwardPassData,
) -> RTSResult:
    """Perform RTS smoothing on forward pass results.

    This is a wrapper around the pure JIT-compiled rts_smooth_pure function.
    Use rts_smooth_pure for optimal performance when calling repeatedly.
    """
    return rts_smooth_pure(
        forward_data.filtered_states,
        forward_data.filtered_covariances,
        forward_data.predicted_states,
        forward_data.predicted_covariances,
        forward_data.transition_matrices,
        forward_data.log_likelihood,
    )


def rts_smooth_pure(
    filtered_states: ArrayLike,
    filtered_covariances: ArrayLike,
    predicted_states: ArrayLike,
    predicted_covariances: ArrayLike,
    transition_matrices: ArrayLike,
    log_likelihood: float,
) -> RTSResult:
    """Pure RTS smoothing implementation with optimal JIT compilation.

    Handles empty input case at Python level to avoid JIT shape issues,
    then delegates to JIT-compiled implementation for computational work.

    Args:
        filtered_states: Filtered state estimates (N, 8)
        filtered_covariances: Filtered covariances (N, 8, 8)
        predicted_states: Predicted state estimates (N, 8)
        predicted_covariances: Predicted covariances (N, 8, 8)
        transition_matrices: State transition matrices (N, 8, 8)
        log_likelihood: Forward pass log-likelihood

    Returns:
        RTSResult with smoothed states and covariances
    """
    N = filtered_states.shape[0]

    # Handle empty input at Python level to avoid JIT shape compatibility issues
    if N == 0:
        state_dim = 8  # trodestrack uses 8-dimensional state
        return RTSResult(
            smoothed_states=jnp.array([]).reshape(0, state_dim),
            smoothed_covariances=jnp.array([]).reshape(0, state_dim, state_dim),
            log_likelihood=log_likelihood,
        )

    # Delegate to JIT-compiled implementation for non-empty case
    return _rts_smooth_impl(
        filtered_states,
        filtered_covariances,
        predicted_states,
        predicted_covariances,
        transition_matrices,
        log_likelihood,
    )


@jax.jit
def _rts_smooth_impl(
    filtered_states: ArrayLike,
    filtered_covariances: ArrayLike,
    predicted_states: ArrayLike,
    predicted_covariances: ArrayLike,
    transition_matrices: ArrayLike,
    log_likelihood: float,
) -> RTSResult:
    """Internal JAX-compiled RTS smoothing implementation.

    Args:
        filtered_states: Filtered state estimates (N, 8)
        filtered_covariances: Filtered covariances (N, 8, 8)
        predicted_states: Predicted state estimates (N, 8)
        predicted_covariances: Predicted covariances (N, 8, 8)
        transition_matrices: State transition matrices (N, 8, 8)
        log_likelihood: Forward pass log-likelihood

    Returns:
        RTSResult with smoothed states and covariances
    """
    N = filtered_states.shape[0]

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
        x_f, P_f, x_p_next, P_p_next, F = inputs

        # Perform backward step with transition matrix
        x_s, P_s = rts_backward_step(x_s_next, P_s_next, x_f, P_f, x_p_next, P_p_next, F)

        return (x_s, P_s), (x_s, P_s)

    # Prepare inputs for scan (forward order, reverse=True will handle backward iteration)
    scan_inputs = (
        filtered_states[:-1],  # x_f from 0 to N-2
        filtered_covariances[:-1],  # P_f from 0 to N-2
        predicted_states[1:],  # x_p_next from 1 to N-1
        predicted_covariances[1:],  # P_p_next from 1 to N-1
        transition_matrices[:-1],  # F from 0 to N-2 (transition from k to k+1)
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
        log_likelihood=log_likelihood,
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
        prediction_data: List[Tuple[ArrayLike, ArrayLike]],
        transition_matrices: List[ArrayLike],
    ) -> ForwardPassData:
        """Collect data from forward pass for smoothing.

        Args:
            ekf_results: Results from EKF forward pass
            prediction_data: List of (predicted_state, predicted_covariance) tuples
            transition_matrices: List of state transition matrices F_k

        Returns:
            ForwardPassData suitable for RTS smoothing
        """
        if len(ekf_results) != len(prediction_data):
            raise ValueError(
                f"Mismatch between EKF results ({len(ekf_results)}) "
                f"and prediction data ({len(prediction_data)})"
            )

        if len(transition_matrices) != len(ekf_results):
            raise ValueError(
                f"Mismatch between transition matrices ({len(transition_matrices)}) "
                f"and EKF results ({len(ekf_results)})"
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
            transition_matrices=jnp.array(transition_matrices),
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
    filtered_states: List[ArrayLike],
    smoothed_states: List[ArrayLike],
    ground_truth: List[ArrayLike],
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
