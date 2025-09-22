"""Mahalanobis gating and measurement masking for outlier rejection.

This module implements robust measurement processing:
- Mahalanobis distance computation for outlier detection
- Chi-squared gating for measurement validation
- Measurement masking for handling missing/invalid data
- Confidence-based filtering
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from scipy.stats import chi2

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


def mahalanobis_distance(
    residual: jnp.ndarray,
    covariance: jnp.ndarray,
) -> float:
    """Compute Mahalanobis distance of residual.

    Args:
        residual: Measurement residual (z - h(x))
        covariance: Residual covariance matrix (H @ P @ H^T + R)

    Returns:
        Mahalanobis distance
    """
    # Handle potential numerical issues with covariance
    try:
        # Compute Cholesky decomposition for stable inversion
        L = jnp.linalg.cholesky(covariance + 1e-12 * jnp.eye(covariance.shape[0]))

        # Solve L @ y = residual for y
        y = jnp.linalg.solve(L, residual)

        # Mahalanobis distance = ||y||
        distance = jnp.linalg.norm(y)

    except jnp.linalg.LinAlgError:
        # Fallback to pseudoinverse for singular matrices
        try:
            cov_inv = jnp.linalg.pinv(covariance)
            distance = jnp.sqrt(residual.T @ cov_inv @ residual)
        except:
            # Ultimate fallback: treat as identity covariance
            distance = jnp.linalg.norm(residual)

    return distance


def mahalanobis_gate(
    residual: jnp.ndarray,
    covariance: jnp.ndarray,
    threshold: float,
) -> bool:
    """Apply Mahalanobis gating to measurement.

    Args:
        residual: Measurement residual
        covariance: Residual covariance matrix
        threshold: Chi-squared gating threshold

    Returns:
        True if measurement passes gate, False if rejected
    """
    distance = mahalanobis_distance(residual, covariance)
    return distance <= threshold


def chi_squared_threshold(dof: int, p_value: float = 0.05) -> float:
    """Compute chi-squared threshold for gating.

    Args:
        dof: Degrees of freedom (measurement dimension)
        p_value: P-value for rejection (default: 0.05 for 95% confidence)

    Returns:
        Chi-squared threshold value
    """
    # Use scipy for accurate chi-squared quantile
    return chi2.ppf(1 - p_value, dof)


def create_measurement_mask(
    measurements: jnp.ndarray,
    confidences: jnp.ndarray,
    min_confidence: float,
) -> jnp.ndarray:
    """Create boolean mask for valid measurements.

    Args:
        measurements: Measurement values
        confidences: Measurement confidences [0, 1]
        min_confidence: Minimum confidence threshold

    Returns:
        Boolean mask (True = valid, False = invalid)
    """
    # Check for finite values
    finite_mask = jnp.isfinite(measurements)

    # Check confidence threshold
    confidence_mask = confidences >= min_confidence

    # Combined mask
    mask = finite_mask & confidence_mask

    return mask


def apply_measurement_mask(
    measurements: jnp.ndarray,
    covariance: jnp.ndarray,
    jacobian: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply measurement mask to filter out invalid measurements.

    Args:
        measurements: Full measurement vector
        covariance: Full measurement covariance matrix
        jacobian: Full measurement Jacobian matrix
        mask: Boolean mask for valid measurements

    Returns:
        Tuple of (masked_measurements, masked_covariance, masked_jacobian)
    """
    # Extract valid measurements
    valid_indices = jnp.where(mask)[0]

    if len(valid_indices) == 0:
        # No valid measurements - return empty arrays
        n_states = jacobian.shape[1]
        return (
            jnp.array([]),
            jnp.zeros((0, 0)),
            jnp.zeros((0, n_states))
        )

    # Filter measurements
    masked_measurements = measurements[valid_indices]

    # Filter covariance matrix (select rows and columns)
    masked_covariance = covariance[jnp.ix_(valid_indices, valid_indices)]

    # Filter Jacobian (select rows)
    masked_jacobian = jacobian[valid_indices, :]

    return masked_measurements, masked_covariance, masked_jacobian


@jax.jit
def compute_innovation_covariance(
    measurement_jacobian: jnp.ndarray,
    state_covariance: jnp.ndarray,
    measurement_noise: jnp.ndarray,
) -> jnp.ndarray:
    """Compute innovation covariance S = H @ P @ H^T + R.

    Args:
        measurement_jacobian: Measurement Jacobian H
        state_covariance: State covariance P
        measurement_noise: Measurement noise covariance R

    Returns:
        Innovation covariance matrix S
    """
    return measurement_jacobian @ state_covariance @ measurement_jacobian.T + measurement_noise


def validate_and_gate_measurement(
    measurement: jnp.ndarray,
    predicted_measurement: jnp.ndarray,
    innovation_covariance: jnp.ndarray,
    confidence: float,
    min_confidence: float = 0.5,
    gating_threshold: float = None,
    measurement_dim: int = None,
) -> Tuple[bool, float]:
    """Validate and gate a single measurement.

    Args:
        measurement: Observed measurement
        predicted_measurement: Predicted measurement h(x)
        innovation_covariance: Innovation covariance S
        confidence: Measurement confidence [0, 1]
        min_confidence: Minimum confidence threshold
        gating_threshold: Chi-squared gating threshold (computed if None)
        measurement_dim: Measurement dimension for threshold computation

    Returns:
        Tuple of (is_valid, mahalanobis_distance)
    """
    # Check confidence threshold
    if confidence < min_confidence:
        return False, jnp.inf

    # Check for finite values
    if not jnp.all(jnp.isfinite(measurement)):
        return False, jnp.inf

    # Compute residual
    residual = measurement - predicted_measurement

    # Compute Mahalanobis distance
    distance = mahalanobis_distance(residual, innovation_covariance)

    # Apply gating
    if gating_threshold is None:
        if measurement_dim is None:
            measurement_dim = len(measurement)
        gating_threshold = chi_squared_threshold(measurement_dim, p_value=0.05)

    is_valid = distance <= gating_threshold

    return is_valid, distance