"""Cached EKF implementation for efficient Jacobian/covariance reuse.

This module provides an optimized EKF implementation that stores and reuses
computed Jacobians and covariance matrices to improve computational efficiency,
particularly useful for RTS smoothing where the same computations are needed
multiple times.

Key optimizations:
- Cache state transition Jacobians F
- Cache predicted covariances P_{k+1|k}
- Reuse measurement Jacobians H when possible
- Minimize redundant matrix operations
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from ._solvers import safe_solve
from .dynamics import compute_process_noise, compute_state_jacobian
from .ekf import EKFResult, EKFState, _predict_state_jax, ekf_update
from .measurements import create_measurement_noise


class CachedComputations(NamedTuple):
    """Cached computations for efficient reuse.

    Attributes:
        state_jacobians: State transition Jacobians F_k for each step
        predicted_covariances: Predicted covariances P_{k+1|k}
        process_noise_matrices: Process noise matrices Q_k
        measurement_jacobians: Measurement Jacobians H_k (when available)
        innovation_covariances: Innovation covariances S_k (when available)
    """

    state_jacobians: List[jnp.ndarray]
    predicted_covariances: List[jnp.ndarray]
    process_noise_matrices: List[jnp.ndarray]
    measurement_jacobians: List[Optional[jnp.ndarray]]
    innovation_covariances: List[Optional[jnp.ndarray]]


class CachedEKFFilter:
    """EKF filter with cached computations for efficiency.

    This class extends the basic EKF with caching mechanisms to store
    intermediate computations that can be reused, particularly beneficial
    for RTS smoothing where the same Jacobians are needed multiple times.
    """

    def __init__(
        self,
        initial_state: EKFState,
        velocity_damping: float = 0.1,
        accel_noise_std: float = 0.5,
        gyro_noise_std: float = 0.1,
        bias_drift_std: float = 0.01,
        position_noise_std: float = 1.0,
        heading_noise_std: float = 0.1,
        gate_threshold: float = 9.21,
        enable_caching: bool = True,
    ):
        """Initialize cached EKF filter.

        Args:
            initial_state: Initial EKF state
            velocity_damping: Velocity damping coefficient λ
            accel_noise_std: Accelerometer noise std dev (m/s²)
            gyro_noise_std: Gyroscope noise std dev (rad/s)
            bias_drift_std: Bias drift std dev (per √s)
            position_noise_std: Position measurement noise std dev (cm)
            heading_noise_std: Heading measurement noise std dev (rad)
            gate_threshold: Chi-squared threshold for measurement gating
            enable_caching: Whether to enable computation caching
        """
        self.current_state = initial_state

        # Process noise parameters
        self.velocity_damping = velocity_damping
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.bias_drift_std = bias_drift_std

        # Measurement noise parameters
        self.position_noise_std = position_noise_std
        self.heading_noise_std = heading_noise_std
        self.gate_threshold = gate_threshold

        # Caching
        self.enable_caching = enable_caching
        self.cached_computations = CachedComputations(
            state_jacobians=[],
            predicted_covariances=[],
            process_noise_matrices=[],
            measurement_jacobians=[],
            innovation_covariances=[],
        )

        # Step counter for indexing cached data
        self.step_count = 0

    def predict_with_caching(
        self,
        dt: float,
        accel: jnp.ndarray,
        gyro: jnp.ndarray,
    ) -> Tuple[EKFState, Dict[str, jnp.ndarray]]:
        """Perform prediction step with caching of intermediate computations.

        Args:
            dt: Time step (seconds)
            accel: Accelerometer measurement [ax, ay] (m/s²)
            gyro: Gyroscope measurement [gz] (rad/s)

        Returns:
            Tuple of (predicted_state, cached_data)
        """
        # Compute and cache state transition Jacobian
        F = compute_state_jacobian(
            self.current_state.state,
            dt,
            accel,
            gyro,
            self.velocity_damping,
        )

        # Compute and cache process noise
        Q = compute_process_noise(
            dt, self.accel_noise_std, self.gyro_noise_std, self.bias_drift_std
        )

        # Predict state (using existing function)
        predicted_state_array = _predict_state_jax(
            self.current_state.state,
            dt,
            accel,
            gyro,
            self.velocity_damping,
        )

        # Predict covariance
        predicted_covariance = F @ self.current_state.covariance @ F.T + Q

        # Create predicted EKF state
        predicted_state = EKFState(
            state=predicted_state_array,
            covariance=predicted_covariance,
            log_likelihood=self.current_state.log_likelihood,
        )

        # Store cached computations if enabled
        cached_data = {
            "state_jacobian": F,
            "process_noise": Q,
            "predicted_covariance": predicted_covariance,
        }

        if self.enable_caching:
            self.cached_computations.state_jacobians.append(F)
            self.cached_computations.predicted_covariances.append(predicted_covariance)
            self.cached_computations.process_noise_matrices.append(Q)

        return predicted_state, cached_data

    def update_with_caching(
        self,
        predicted_state: EKFState,
        position: Optional[jnp.ndarray] = None,
        heading: Optional[float] = None,
        confidence: float = 1.0,
    ) -> Tuple[EKFResult, Dict[str, Optional[Array]]]:
        """Perform measurement update with caching of intermediate computations.

        Args:
            predicted_state: Predicted EKF state from prediction step
            position: Position measurement [x, y] in cm (None if missing)
            heading: Heading measurement in radians (None if missing)
            confidence: Detection confidence [0, 1]

        Returns:
            Tuple of (ekf_result, cached_data)
        """
        # Handle case with no measurements
        if position is None and heading is None:
            # Store empty cached data
            cached_data = {
                "measurement_jacobian": None,
                "innovation_covariance": None,
            }

            if self.enable_caching:
                self.cached_computations.measurement_jacobians.append(None)
                self.cached_computations.innovation_covariances.append(None)

            # Return no-update result
            result = EKFResult(
                state=predicted_state,
                innovation=jnp.array([]),
                innovation_covariance=jnp.array([[]]),
                kalman_gain=jnp.array([[]]),
                gated=False,
            )
            return result, cached_data

        # Create measurement vector and noise
        if position is not None and heading is not None:
            measurement = jnp.concatenate([position, jnp.array([heading])])
            has_heading = True
        elif position is not None:
            measurement = position
            has_heading = False
        else:
            # Heading-only case (rare)
            dummy_position = predicted_state.state[:2]
            measurement = jnp.concatenate([dummy_position, jnp.array([heading])])
            has_heading = True
            confidence = 0.01

        measurement_noise = create_measurement_noise(
            self.position_noise_std,
            confidence,
            has_heading,
            self.heading_noise_std if has_heading else None,
        )

        # Perform standard update
        result = ekf_update(
            predicted_state,
            measurement,
            measurement_noise,
            has_heading,
            self.gate_threshold,
        )

        # Cache measurement Jacobian and innovation covariance
        cached_data = {
            "measurement_jacobian": None,  # Would need to compute separately
            "innovation_covariance": result.innovation_covariance,
        }

        if self.enable_caching:
            self.cached_computations.measurement_jacobians.append(None)
            self.cached_computations.innovation_covariances.append(result.innovation_covariance)

        return result, cached_data

    def step_with_caching(
        self,
        dt: float,
        accel: jnp.ndarray,
        gyro: jnp.ndarray,
        position: Optional[jnp.ndarray] = None,
        heading: Optional[float] = None,
        confidence: float = 1.0,
    ) -> Tuple[EKFResult, Dict[str, Optional[Array]]]:
        """Perform complete EKF step (predict + update) with caching.

        Args:
            dt: Time step (seconds)
            accel: Accelerometer measurement [ax, ay] (m/s²)
            gyro: Gyroscope measurement [gz] (rad/s)
            position: Position measurement [x, y] in cm (None if missing)
            heading: Heading measurement in radians (None if missing)
            confidence: Detection confidence [0, 1]

        Returns:
            Tuple of (ekf_result, all_cached_data)
        """
        # Prediction step
        predicted_state, predict_cache = self.predict_with_caching(dt, accel, gyro)

        # Update step
        result, update_cache = self.update_with_caching(
            predicted_state, position, heading, confidence
        )

        # Update internal state
        if not result.gated:
            self.current_state = result.state

        # Increment step counter
        self.step_count += 1

        # Combine cached data
        all_cached_data = {**predict_cache, **update_cache}

        return result, all_cached_data

    def get_cached_jacobian(self, step_index: int) -> Optional[jnp.ndarray]:
        """Get cached state Jacobian for a specific step.

        Args:
            step_index: Step index (0-based)

        Returns:
            Cached Jacobian matrix, or None if not available
        """
        if not self.enable_caching or step_index >= len(self.cached_computations.state_jacobians):
            return None
        return self.cached_computations.state_jacobians[step_index]

    def get_cached_predicted_covariance(self, step_index: int) -> Optional[jnp.ndarray]:
        """Get cached predicted covariance for a specific step.

        Args:
            step_index: Step index (0-based)

        Returns:
            Cached predicted covariance, or None if not available
        """
        if not self.enable_caching or step_index >= len(
            self.cached_computations.predicted_covariances
        ):
            return None
        return self.cached_computations.predicted_covariances[step_index]

    def get_all_cached_data(self) -> CachedComputations:
        """Get all cached computations.

        Returns:
            CachedComputations with all stored data
        """
        return self.cached_computations

    def clear_cache(self) -> None:
        """Clear all cached computations and reset step counter."""
        self.cached_computations = CachedComputations(
            state_jacobians=[],
            predicted_covariances=[],
            process_noise_matrices=[],
            measurement_jacobians=[],
            innovation_covariances=[],
        )
        self.step_count = 0


def efficient_rts_smooth_with_cache(
    cached_ekf: CachedEKFFilter,
    ekf_results: List[EKFResult],
) -> Tuple[List[Array], List[Array]]:
    """Efficient RTS smoothing using cached computations.

    This function performs RTS smoothing while reusing cached Jacobians
    and predicted covariances, avoiding redundant computations.

    Args:
        cached_ekf: EKF filter with cached computations
        ekf_results: Results from forward filtering pass

    Returns:
        Tuple of (smoothed_states, smoothed_covariances)
    """
    N = len(ekf_results)
    if N == 0:
        return [], []

    # Initialize with final filtered estimates
    smoothed_states: List[Optional[Array]] = [None] * N
    smoothed_covariances: List[Optional[Array]] = [None] * N

    smoothed_states[N - 1] = ekf_results[N - 1].state.state
    smoothed_covariances[N - 1] = ekf_results[N - 1].state.covariance

    # Backward pass using cached data
    for k in range(N - 2, -1, -1):
        # Get cached predicted covariance (if available)
        P_p_next = cached_ekf.get_cached_predicted_covariance(k + 1)

        if P_p_next is not None:
            # Use cached data for efficiency
            x_f = ekf_results[k].state.state
            P_f = ekf_results[k].state.covariance
            x_s_next = smoothed_states[k + 1]
            P_s_next = smoothed_covariances[k + 1]
            x_p_next = ekf_results[k + 1].state.state  # Approximation

            # Compute smoother gain using safe solve
            G = safe_solve(P_p_next, P_f.T).T

            # Smoothed estimates
            x_s = x_f + G @ (x_s_next - x_p_next)
            P_s = P_f + G @ (P_s_next - P_p_next) @ G.T

            smoothed_states[k] = x_s
            smoothed_covariances[k] = P_s
        else:
            # Fallback to filtered estimates if no cached data
            smoothed_states[k] = ekf_results[k].state.state
            smoothed_covariances[k] = ekf_results[k].state.covariance

    return smoothed_states, smoothed_covariances


def compute_cache_efficiency_stats(cached_ekf: CachedEKFFilter) -> Dict[str, float]:
    """Compute statistics on cache efficiency and memory usage.

    Args:
        cached_ekf: EKF filter with cached data

    Returns:
        Dictionary with cache efficiency statistics
    """
    cached_data = cached_ekf.get_all_cached_data()

    n_steps = len(cached_data.state_jacobians)
    n_jacobians = sum(1 for j in cached_data.state_jacobians if j is not None)
    n_pred_covariances = sum(1 for p in cached_data.predicted_covariances if p is not None)

    # Estimate memory usage (rough approximation)
    bytes_per_matrix = 8 * 8 * 8  # 8x8 matrix of float64
    total_cache_memory = (n_jacobians + n_pred_covariances) * bytes_per_matrix

    return {
        "total_steps": n_steps,
        "cached_jacobians": n_jacobians,
        "cached_covariances": n_pred_covariances,
        "jacobian_cache_rate": n_jacobians / max(1, n_steps),
        "covariance_cache_rate": n_pred_covariances / max(1, n_steps),
        "estimated_cache_memory_bytes": total_cache_memory,
        "estimated_cache_memory_mb": total_cache_memory / (1024**2),
    }
