"""Unscented Kalman Filter implementation for 2D tracking.

This module implements the UKF algorithm for offline smoothing, featuring:
- JAX-compiled sigma point generation and propagation
- Robust measurement handling with gating
- Support for missing measurements and occlusions
- Enhanced nonlinear handling compared to EKF
"""

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky

from ._solvers import _symmetrize_and_stabilize, mahalanobis_distance, safe_solve
from .dynamics import compute_process_noise, rotation_matrix_2d, wrap_angle
from .measurements import create_measurement_noise
from .state import State2D, array_to_state, state_to_array


class UKFState(NamedTuple):
    """UKF state representation.

    Attributes:
        state: Current state estimate
        covariance: State covariance matrix (8x8)
        log_likelihood: Cumulative log-likelihood
    """

    state: jnp.ndarray  # 8-dimensional state vector
    covariance: jnp.ndarray  # 8x8 covariance matrix
    log_likelihood: float


class UKFResult(NamedTuple):
    """Result from UKF update step.

    Attributes:
        state: Updated UKF state
        innovation: Measurement innovation (residual)
        innovation_covariance: Innovation covariance matrix
        kalman_gain: Kalman gain matrix
        gated: Whether measurement was gated (rejected)
    """

    state: UKFState
    innovation: jnp.ndarray
    innovation_covariance: jnp.ndarray
    kalman_gain: jnp.ndarray
    gated: bool


class UKFParams(NamedTuple):
    """UKF algorithm parameters.

    Attributes:
        alpha: Spread parameter (controls sigma point spread) [0.001, 1]
        beta: Distribution parameter (incorporates prior knowledge) (2 for Gaussian)
        kappa: Secondary spread parameter (typically 0 or 3-n)
    """

    alpha: float = 1.0  # Use unscented transform without scaling for stability
    beta: float = 2.0
    kappa: float = 0.0


@jax.jit
def generate_sigma_points(
    state: jnp.ndarray,
    covariance: jnp.ndarray,
    params: UKFParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate sigma points and weights for UKF.

    Args:
        state: Mean state vector (n,)
        covariance: Covariance matrix (n, n)
        params: UKF parameters

    Returns:
        Tuple of (sigma_points, weights) where:
        - sigma_points: Array of shape (2n+1, n)
        - weights: Array of shape (2n+1,) with mean and covariance weights
    """
    n = state.shape[0]  # State dimension
    lambda_ = params.alpha**2 * (n + params.kappa) - n

    # Compute square root of scaled covariance matrix with PSD hygiene
    scaled_cov = (n + lambda_) * covariance
    scaled_cov_stable = _symmetrize_and_stabilize(scaled_cov, jitter=1e-12)
    sqrt_matrix = cholesky(scaled_cov_stable, lower=True)

    # Generate sigma points vectorized (no Python loops)
    sigma_points = jnp.zeros((2 * n + 1, n))

    # Central sigma point
    sigma_points = sigma_points.at[0].set(state)

    # Positive and negative sigma points (vectorized)
    # Positive sigma points: state + sqrt_matrix[:, i] for each column i
    positive_points = state[None, :] + sqrt_matrix.T  # (n, n) matrix
    sigma_points = sigma_points.at[1 : n + 1].set(positive_points)

    # Negative sigma points: state - sqrt_matrix[:, i] for each column i
    negative_points = state[None, :] - sqrt_matrix.T  # (n, n) matrix
    sigma_points = sigma_points.at[n + 1 : 2 * n + 1].set(negative_points)

    # Compute weights according to standard UKF formulation
    # For mean weights
    weight_mean_0 = lambda_ / (n + lambda_)
    weight_others = 1.0 / (2.0 * (n + lambda_))

    # For covariance weights
    weight_cov_0 = weight_mean_0 + (1.0 - params.alpha**2 + params.beta)

    # Create weight arrays
    weights_mean = jnp.concatenate([jnp.array([weight_mean_0]), jnp.full(2 * n, weight_others)])

    weights_cov = jnp.concatenate([jnp.array([weight_cov_0]), jnp.full(2 * n, weight_others)])

    return sigma_points, (weights_mean, weights_cov)


@jax.jit
def propagate_sigma_points(
    sigma_points: jnp.ndarray,
    dt: float,
    accel: jnp.ndarray,
    gyro: jnp.ndarray,
    velocity_damping: float,
) -> jnp.ndarray:
    """Propagate sigma points through dynamics function.

    Args:
        sigma_points: Input sigma points (2n+1, n)
        dt: Time step
        accel: Accelerometer measurement
        gyro: Gyroscope measurement
        velocity_damping: Velocity damping coefficient

    Returns:
        Propagated sigma points (2n+1, n)
    """

    def dynamics_function(x: jnp.ndarray) -> jnp.ndarray:
        """Dynamics function for sigma point propagation."""
        # Extract state components
        pos = x[:2]
        vel = x[2:4]
        theta = x[4]
        b_gz = x[5]
        b_ax = x[6]
        b_ay = x[7]

        # Bias-corrected measurements
        accel_corrected = accel - jnp.array([b_ax, b_ay])
        gyro_corrected = gyro[0] - b_gz

        # Rotate acceleration from IMU/body frame to world frame
        R = rotation_matrix_2d(theta)
        accel_world = R @ accel_corrected

        # Convert to cm/s²
        accel_corrected_cm = accel_world * 100.0

        # Apply damping
        damping_factor = 1.0 - velocity_damping * dt
        vel_damped = vel * damping_factor

        # Update dynamics
        vel_new = vel_damped + accel_corrected_cm * dt
        pos_new = pos + vel * dt + 0.5 * accel_corrected_cm * dt**2
        theta_new = wrap_angle(theta + gyro_corrected * dt)

        # Biases unchanged
        return jnp.array(
            [pos_new[0], pos_new[1], vel_new[0], vel_new[1], theta_new, b_gz, b_ax, b_ay]
        )

    # Apply dynamics to each sigma point
    return jax.vmap(dynamics_function)(sigma_points)


@jax.jit
def predict_from_sigma_points(
    propagated_points: jnp.ndarray,
    weights: Tuple[jnp.ndarray, jnp.ndarray],
    process_noise: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute predicted mean and covariance from propagated sigma points.

    Args:
        propagated_points: Propagated sigma points (2n+1, n)
        weights: Tuple of (mean_weights, covariance_weights)
        process_noise: Process noise matrix Q

    Returns:
        Tuple of (predicted_mean, predicted_covariance)
    """
    weights_mean, weights_cov = weights

    # Predicted mean
    predicted_mean = jnp.sum(weights_mean[:, None] * propagated_points, axis=0)

    # Predicted covariance
    centered_points = propagated_points - predicted_mean[None, :]
    predicted_cov = jnp.sum(
        weights_cov[:, None, None] * centered_points[:, :, None] * centered_points[:, None, :],
        axis=0,
    )
    predicted_cov += process_noise

    return predicted_mean, predicted_cov


@jax.jit
def ukf_predict(
    ukf_state: UKFState,
    dt: float,
    accel: jnp.ndarray,
    gyro: jnp.ndarray,
    velocity_damping: float,
    accel_noise_std: float,
    gyro_noise_std: float,
    bias_drift_std: float,
    params: UKFParams,
) -> UKFState:
    """UKF prediction step.

    Args:
        ukf_state: Current UKF state
        dt: Time step (seconds)
        accel: Accelerometer measurement [ax, ay] (m/s²)
        gyro: Gyroscope measurement [gz] (rad/s)
        velocity_damping: Velocity damping coefficient λ
        accel_noise_std: Accelerometer noise std dev (m/s²)
        gyro_noise_std: Gyroscope noise std dev (rad/s)
        bias_drift_std: Bias drift std dev (per √s)
        params: UKF parameters

    Returns:
        Predicted UKF state
    """
    # Generate sigma points
    sigma_points, weights = generate_sigma_points(ukf_state.state, ukf_state.covariance, params)

    # Propagate through dynamics
    propagated_points = propagate_sigma_points(sigma_points, dt, accel, gyro, velocity_damping)

    # Compute process noise
    process_noise = compute_process_noise(dt, accel_noise_std, gyro_noise_std, bias_drift_std)

    # Predict mean and covariance
    predicted_mean, predicted_cov = predict_from_sigma_points(
        propagated_points, weights, process_noise
    )

    return UKFState(
        state=predicted_mean,
        covariance=predicted_cov,
        log_likelihood=ukf_state.log_likelihood,
    )


@jax.jit
def measurement_sigma_points_position(
    sigma_points: jnp.ndarray,
) -> jnp.ndarray:
    """Transform sigma points through position measurement function.

    Args:
        sigma_points: State sigma points (2n+1, n)

    Returns:
        Measurement sigma points (2n+1, 2)
    """
    return sigma_points[:, :2]  # Extract position components


@jax.jit
def measurement_sigma_points_position_heading(
    sigma_points: jnp.ndarray,
) -> jnp.ndarray:
    """Transform sigma points through position + heading measurement function.

    Args:
        sigma_points: State sigma points (2n+1, n)

    Returns:
        Measurement sigma points (2n+1, 3)
    """
    position = sigma_points[:, :2]  # x, y
    heading = sigma_points[:, 4:5]  # theta
    return jnp.concatenate([position, heading], axis=1)


@jax.jit
def _ukf_update_position_only(
    ukf_state: UKFState,
    measurement: jnp.ndarray,
    measurement_noise: jnp.ndarray,
    gate_threshold: float,
    params: UKFParams,
) -> UKFResult:
    """UKF update for position-only measurements."""
    # Generate sigma points
    sigma_points, weights = generate_sigma_points(ukf_state.state, ukf_state.covariance, params)

    # Transform sigma points through measurement function
    measurement_points = measurement_sigma_points_position(sigma_points)

    weights_mean, weights_cov = weights

    # Predicted measurement
    predicted_measurement = jnp.sum(weights_mean[:, None] * measurement_points, axis=0)

    # Innovation
    innovation = measurement - predicted_measurement

    # Innovation covariance
    centered_measurement_points = measurement_points - predicted_measurement[None, :]
    innovation_cov = jnp.sum(
        weights_cov[:, None, None]
        * centered_measurement_points[:, :, None]
        * centered_measurement_points[:, None, :],
        axis=0,
    )
    innovation_cov += measurement_noise

    # Cross-covariance
    centered_state_points = sigma_points - ukf_state.state[None, :]
    cross_cov = jnp.sum(
        weights_cov[:, None, None]
        * centered_state_points[:, :, None]
        * centered_measurement_points[:, None, :],
        axis=0,
    )

    # Mahalanobis gating
    mahalanobis_dist = mahalanobis_distance(innovation, innovation_cov)
    gated = mahalanobis_dist > gate_threshold

    # Kalman gain
    K = safe_solve(innovation_cov, cross_cov.T).T

    # State and covariance update (conditional on gating)
    updated_state = jnp.where(gated, ukf_state.state, ukf_state.state + K @ innovation)

    updated_covariance = jnp.where(
        gated, ukf_state.covariance, ukf_state.covariance - K @ innovation_cov @ K.T
    )

    # Log-likelihood update
    log_det_S = jnp.linalg.slogdet(innovation_cov)[1]
    measurement_dim = 2
    log_likelihood_update = -0.5 * (
        measurement_dim * jnp.log(2 * jnp.pi) + log_det_S + mahalanobis_dist
    )

    updated_log_likelihood = jnp.where(
        gated, ukf_state.log_likelihood, ukf_state.log_likelihood + log_likelihood_update
    )

    updated_ukf_state = UKFState(
        state=updated_state,
        covariance=updated_covariance,
        log_likelihood=updated_log_likelihood,
    )

    return UKFResult(
        state=updated_ukf_state,
        innovation=innovation,
        innovation_covariance=innovation_cov,
        kalman_gain=K,
        gated=gated,
    )


@jax.jit
def _ukf_update_position_heading(
    ukf_state: UKFState,
    measurement: jnp.ndarray,
    measurement_noise: jnp.ndarray,
    gate_threshold: float,
    params: UKFParams,
) -> UKFResult:
    """UKF update for position + heading measurements."""
    # Generate sigma points
    sigma_points, weights = generate_sigma_points(ukf_state.state, ukf_state.covariance, params)

    # Transform sigma points through measurement function
    measurement_points = measurement_sigma_points_position_heading(sigma_points)

    weights_mean, weights_cov = weights

    # Predicted measurement
    predicted_measurement = jnp.sum(weights_mean[:, None] * measurement_points, axis=0)

    # Innovation with heading wrapping
    innovation = measurement - predicted_measurement
    wrapped_heading_innov = jnp.arctan2(jnp.sin(innovation[2]), jnp.cos(innovation[2]))
    innovation = innovation.at[2].set(wrapped_heading_innov)

    # Innovation covariance
    # Handle angle wrapping for measurement points
    centered_measurement_points = measurement_points - predicted_measurement[None, :]
    # Wrap heading differences
    wrapped_heading_diffs = jnp.arctan2(
        jnp.sin(centered_measurement_points[:, 2]), jnp.cos(centered_measurement_points[:, 2])
    )
    centered_measurement_points = centered_measurement_points.at[:, 2].set(wrapped_heading_diffs)

    innovation_cov = jnp.sum(
        weights_cov[:, None, None]
        * centered_measurement_points[:, :, None]
        * centered_measurement_points[:, None, :],
        axis=0,
    )
    innovation_cov += measurement_noise

    # Cross-covariance
    centered_state_points = sigma_points - ukf_state.state[None, :]
    cross_cov = jnp.sum(
        weights_cov[:, None, None]
        * centered_state_points[:, :, None]
        * centered_measurement_points[:, None, :],
        axis=0,
    )

    # Mahalanobis gating
    mahalanobis_dist = mahalanobis_distance(innovation, innovation_cov)
    gated = mahalanobis_dist > gate_threshold

    # Kalman gain
    K = safe_solve(innovation_cov, cross_cov.T).T

    # State and covariance update (conditional on gating)
    updated_state = jnp.where(gated, ukf_state.state, ukf_state.state + K @ innovation)

    updated_covariance = jnp.where(
        gated, ukf_state.covariance, ukf_state.covariance - K @ innovation_cov @ K.T
    )

    # Log-likelihood update
    log_det_S = jnp.linalg.slogdet(innovation_cov)[1]
    measurement_dim = 3
    log_likelihood_update = -0.5 * (
        measurement_dim * jnp.log(2 * jnp.pi) + log_det_S + mahalanobis_dist
    )

    updated_log_likelihood = jnp.where(
        gated, ukf_state.log_likelihood, ukf_state.log_likelihood + log_likelihood_update
    )

    updated_ukf_state = UKFState(
        state=updated_state,
        covariance=updated_covariance,
        log_likelihood=updated_log_likelihood,
    )

    return UKFResult(
        state=updated_ukf_state,
        innovation=innovation,
        innovation_covariance=innovation_cov,
        kalman_gain=K,
        gated=gated,
    )


def ukf_update(
    ukf_state: UKFState,
    measurement: jnp.ndarray,
    measurement_noise: jnp.ndarray,
    has_heading: bool,
    gate_threshold: float = 9.21,
    params: UKFParams = UKFParams(),
) -> UKFResult:
    """UKF measurement update step.

    Args:
        ukf_state: Predicted UKF state
        measurement: Measurement vector (position + optional heading)
        measurement_noise: Measurement noise covariance matrix R
        has_heading: Whether measurement includes heading
        gate_threshold: Chi-squared threshold for gating
        params: UKF parameters

    Returns:
        UKF update result
    """
    if has_heading:
        return _ukf_update_position_heading(
            ukf_state, measurement, measurement_noise, gate_threshold, params
        )
    else:
        return _ukf_update_position_only(
            ukf_state, measurement, measurement_noise, gate_threshold, params
        )


def create_initial_ukf_state(
    initial_state: State2D,
    initial_covariance: jnp.ndarray,
) -> UKFState:
    """Create initial UKF state from State2D and covariance.

    Args:
        initial_state: Initial state estimate
        initial_covariance: Initial covariance matrix

    Returns:
        Initial UKF state
    """
    return UKFState(
        state=state_to_array(initial_state),
        covariance=initial_covariance,
        log_likelihood=0.0,
    )


class UKFFilter:
    """Unscented Kalman Filter for 2D tracking.

    This class provides a stateful interface to the UKF algorithm with
    configuration management and measurement processing.
    """

    def __init__(
        self,
        initial_state: State2D,
        initial_covariance: jnp.ndarray,
        velocity_damping: float = 0.1,
        accel_noise_std: float = 0.5,
        gyro_noise_std: float = 0.1,
        bias_drift_std: float = 0.01,
        position_noise_std: float = 1.0,
        heading_noise_std: float = 0.1,
        gate_threshold: float = 9.21,
        ukf_params: UKFParams = UKFParams(),
    ):
        """Initialize UKF filter.

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
            ukf_params: UKF algorithm parameters
        """
        self.ukf_state = create_initial_ukf_state(initial_state, initial_covariance)

        # Process noise parameters
        self.velocity_damping = velocity_damping
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.bias_drift_std = bias_drift_std

        # Measurement noise parameters
        self.position_noise_std = position_noise_std
        self.heading_noise_std = heading_noise_std

        # Algorithm parameters
        self.gate_threshold = gate_threshold
        self.ukf_params = ukf_params

    def predict(
        self,
        dt: float,
        accel: jnp.ndarray,
        gyro: jnp.ndarray,
    ) -> None:
        """Perform prediction step.

        Args:
            dt: Time step (seconds)
            accel: Accelerometer measurement [ax, ay] (m/s²)
            gyro: Gyroscope measurement [gz] (rad/s)
        """
        self.ukf_state = ukf_predict(
            self.ukf_state,
            dt,
            accel,
            gyro,
            self.velocity_damping,
            self.accel_noise_std,
            self.gyro_noise_std,
            self.bias_drift_std,
            self.ukf_params,
        )

    def update(
        self,
        position: Optional[jnp.ndarray] = None,
        heading: Optional[float] = None,
        confidence: float = 1.0,
    ) -> UKFResult:
        """Perform measurement update step.

        Args:
            position: Position measurement [x, y] in cm (None if missing)
            heading: Heading measurement in radians (None if missing)
            confidence: Detection confidence [0, 1]

        Returns:
            UKF update result
        """
        # Skip update if no measurements available
        if position is None and heading is None:
            # Return no-update result
            return UKFResult(
                state=self.ukf_state,
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
            dummy_position = self.ukf_state.state[:2]
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
        result = ukf_update(
            self.ukf_state,
            measurement,
            measurement_noise,
            has_heading,
            self.gate_threshold,
            self.ukf_params,
        )

        # Update internal state if not gated
        if not result.gated:
            self.ukf_state = result.state

        return result

    def get_current_state(self) -> State2D:
        """Get current state estimate as State2D object.

        Returns:
            Current state estimate
        """
        return array_to_state(self.ukf_state.state)

    def get_current_covariance(self) -> jnp.ndarray:
        """Get current covariance matrix.

        Returns:
            Current covariance matrix
        """
        return self.ukf_state.covariance

    def get_log_likelihood(self) -> float:
        """Get cumulative log-likelihood.

        Returns:
            Cumulative log-likelihood
        """
        return float(self.ukf_state.log_likelihood)
