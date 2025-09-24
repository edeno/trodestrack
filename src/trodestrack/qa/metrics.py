"""
Quality assurance metrics for state estimation evaluation.

This module provides comprehensive metrics for evaluating tracking performance,
including RMSE, NEES, and specialized drift analysis for occlusion periods.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from typing import Dict, Optional, Tuple, Union


def compute_rmse(
    estimated_states: jnp.ndarray,
    ground_truth_states: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute Root Mean Square Error (RMSE) for position, velocity, and heading.

    Args:
        estimated_states: Shape (N, 8) array of estimated states [x, y, vx, vy, theta, ...]
        ground_truth_states: Shape (N, 8) array of ground truth states
        mask: Optional shape (N,) boolean mask for valid timesteps

    Returns:
        Dictionary with RMSE values for position, velocity, and heading
    """
    if mask is not None:
        estimated_states = estimated_states[mask]
        ground_truth_states = ground_truth_states[mask]

    # Extract components
    est_pos = estimated_states[:, :2]  # x, y
    gt_pos = ground_truth_states[:, :2]

    est_vel = estimated_states[:, 2:4]  # vx, vy
    gt_vel = ground_truth_states[:, 2:4]

    est_heading = estimated_states[:, 4]  # theta
    gt_heading = ground_truth_states[:, 4]

    # Position RMSE (cm)
    pos_error = jnp.linalg.norm(est_pos - gt_pos, axis=1)
    rmse_position = float(jnp.sqrt(jnp.mean(pos_error**2)))

    # Velocity RMSE (cm/s)
    vel_error = jnp.linalg.norm(est_vel - gt_vel, axis=1)
    rmse_velocity = float(jnp.sqrt(jnp.mean(vel_error**2)))

    # Heading RMSE (degrees) - handle angle wrapping
    heading_error = _wrap_angle_difference(est_heading - gt_heading)
    rmse_heading = float(jnp.sqrt(jnp.mean(heading_error**2)) * 180.0 / jnp.pi)

    return {
        "position_rmse_cm": rmse_position,
        "velocity_rmse_cm_s": rmse_velocity,
        "heading_rmse_deg": rmse_heading,
    }


def compute_nees(
    estimated_states: jnp.ndarray,
    ground_truth_states: jnp.ndarray,
    covariances: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute Normalized Estimation Error Squared (NEES) for filter consistency.

    NEES measures filter consistency by comparing the estimation error magnitude
    to the filter's predicted uncertainty. For a consistent filter, NEES should
    follow a chi-squared distribution with DOF equal to state dimension.

    Args:
        estimated_states: Shape (N, 8) array of estimated states
        ground_truth_states: Shape (N, 8) array of ground truth states
        covariances: Shape (N, 8, 8) array of covariance matrices
        mask: Optional shape (N,) boolean mask for valid timesteps

    Returns:
        Dictionary with NEES statistics and consistency metrics
    """
    if mask is not None:
        estimated_states = estimated_states[mask]
        ground_truth_states = ground_truth_states[mask]
        covariances = covariances[mask]

    # Compute estimation errors
    errors = estimated_states - ground_truth_states

    # Handle heading angle wrapping
    errors = errors.at[:, 4].set(_wrap_angle_difference(errors[:, 4]))

    # Compute NEES for each timestep using vectorized JAX operations
    nees_array = _compute_nees_vectorized(errors, covariances)

    # Expected NEES for 8-DOF state
    expected_nees = 8.0

    return {
        "nees_mean": float(jnp.mean(nees_array)),
        "nees_std": float(jnp.std(nees_array)),
        "nees_expected": expected_nees,
        "nees_consistency_ratio": float(jnp.mean(nees_array) / expected_nees),
        "nees_values": np.array(nees_array),  # Return as numpy for plotting
    }


@jax.jit
def _compute_nees_vectorized(errors: jnp.ndarray, covariances: jnp.ndarray) -> jnp.ndarray:
    """Vectorized NEES computation using JAX lax.scan.

    This replaces the Python for loop with JAX operations for better
    performance and GPU compatibility.

    Args:
        errors: Estimation errors array (n_timesteps, state_dim)
        covariances: Covariance matrices array (n_timesteps, state_dim, state_dim)

    Returns:
        NEES values array (n_timesteps,)
    """

    def nees_step(carry, inputs):
        """Single step for computing NEES."""
        error, cov = inputs

        # NEES = e^T * P^{-1} * e
        # Use pseudoinverse for numerical stability
        cov_inv = jnp.linalg.pinv(cov)
        nees_i = error.T @ cov_inv @ error

        return carry, nees_i

    # Run lax.scan to compute all NEES values
    _, nees_values = lax.scan(nees_step, None, (errors, covariances))

    return nees_values


def compute_position_nees(
    estimated_states: jnp.ndarray,
    ground_truth_states: jnp.ndarray,
    covariances: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute NEES for position components only (2-DOF).

    Args:
        estimated_states: Shape (N, 8) array of estimated states
        ground_truth_states: Shape (N, 8) array of ground truth states
        covariances: Shape (N, 8, 8) array of covariance matrices
        mask: Optional shape (N,) boolean mask for valid timesteps

    Returns:
        Dictionary with position NEES statistics
    """
    if mask is not None:
        estimated_states = estimated_states[mask]
        ground_truth_states = ground_truth_states[mask]
        covariances = covariances[mask]

    # Extract position components
    pos_errors = estimated_states[:, :2] - ground_truth_states[:, :2]
    pos_covariances = covariances[:, :2, :2]

    # Compute position NEES
    nees_values = []
    for i in range(len(pos_errors)):
        error = pos_errors[i]
        cov = pos_covariances[i]

        cov_inv = jnp.linalg.pinv(cov)
        nees_i = float(error.T @ cov_inv @ error)
        nees_values.append(nees_i)

    nees_array = jnp.array(nees_values)
    expected_nees = 2.0  # 2-DOF for position

    return {
        "position_nees_mean": float(jnp.mean(nees_array)),
        "position_nees_std": float(jnp.std(nees_array)),
        "position_nees_expected": expected_nees,
        "position_nees_consistency_ratio": float(jnp.mean(nees_array) / expected_nees),
        "position_nees_values": np.array(nees_values),
    }


def compute_occlusion_drift(
    estimated_states: jnp.ndarray,
    ground_truth_states: jnp.ndarray,
    occlusion_mask: jnp.ndarray,
    dt: float = 1.0 / 30.0,
    max_drift_duration: float = 7.0,
) -> Dict[str, float]:
    """
    Compute position drift during occlusion periods.

    Measures how much the estimated position drifts from ground truth
    during periods when visual measurements are unavailable.

    Args:
        estimated_states: Shape (N, 8) array of estimated states
        ground_truth_states: Shape (N, 8) array of ground truth states
        occlusion_mask: Shape (N,) boolean mask where True = occluded
        dt: Time step between frames (default: 1/30 s)
        max_drift_duration: Maximum occlusion duration to analyze (seconds)

    Returns:
        Dictionary with occlusion drift metrics
    """
    max_frames = int(max_drift_duration / dt)

    # Find occlusion segments
    occlusion_segments = _find_segments(occlusion_mask)

    drift_results = []
    for start_idx, end_idx in occlusion_segments:
        duration_frames = end_idx - start_idx
        duration_seconds = duration_frames * dt

        # Skip short occlusions or very long ones
        if duration_frames < 3 or duration_frames > max_frames:
            continue

        # Position at start and end of occlusion
        pos_start_est = estimated_states[start_idx, :2]
        pos_end_est = estimated_states[end_idx - 1, :2]

        pos_start_gt = ground_truth_states[start_idx, :2]
        pos_end_gt = ground_truth_states[end_idx - 1, :2]

        # Drift = change in position error during occlusion
        error_start = jnp.linalg.norm(pos_start_est - pos_start_gt)
        error_end = jnp.linalg.norm(pos_end_est - pos_end_gt)

        drift = float(error_end - error_start)
        drift_rate = drift / duration_seconds  # cm/s

        drift_results.append(
            {
                "duration_s": duration_seconds,
                "drift_cm": drift,
                "drift_rate_cm_s": drift_rate,
                "final_error_cm": float(error_end),
            }
        )

    if not drift_results:
        return {
            "num_occlusions": 0,
            "mean_drift_cm": 0.0,
            "max_drift_cm": 0.0,
            "mean_drift_rate_cm_s": 0.0,
        }

    drifts = [r["drift_cm"] for r in drift_results]
    drift_rates = [r["drift_rate_cm_s"] for r in drift_results]

    return {
        "num_occlusions": len(drift_results),
        "mean_drift_cm": float(np.mean(drifts)),
        "max_drift_cm": float(np.max(drifts)),
        "std_drift_cm": float(np.std(drifts)),
        "mean_drift_rate_cm_s": float(np.mean(drift_rates)),
        "occlusion_details": drift_results,
    }


def evaluate_prd_compliance(metrics: Dict[str, Union[float, Dict]]) -> Dict[str, bool]:
    """
    Evaluate compliance with PRD (Project Requirements Document) thresholds.

    PRD Requirements:
    - Position RMSE ≤ 2 cm
    - Velocity RMSE ≤ 10 cm/s
    - Heading RMSE ≤ 7°
    - Occlusion drift ≤ 15 cm after 5-7s dropout

    Args:
        metrics: Dictionary containing computed metrics

    Returns:
        Dictionary with boolean compliance flags
    """
    compliance = {}

    # Position RMSE compliance
    if "position_rmse_cm" in metrics:
        compliance["position_rmse_ok"] = metrics["position_rmse_cm"] <= 2.0

    # Velocity RMSE compliance
    if "velocity_rmse_cm_s" in metrics:
        compliance["velocity_rmse_ok"] = metrics["velocity_rmse_cm_s"] <= 10.0

    # Heading RMSE compliance
    if "heading_rmse_deg" in metrics:
        compliance["heading_rmse_ok"] = metrics["heading_rmse_deg"] <= 7.0

    # Occlusion drift compliance
    if "max_drift_cm" in metrics:
        compliance["occlusion_drift_ok"] = metrics["max_drift_cm"] <= 15.0

    # Overall compliance
    compliance["overall_prd_compliant"] = all(compliance.values())

    return compliance


def _wrap_angle_difference(angle_diff: jnp.ndarray) -> jnp.ndarray:
    """Wrap angle difference to [-π, π] range using shortest angular distance."""
    # This ensures we get the shortest angular distance, not just the wrapped angle
    wrapped = jnp.remainder(angle_diff + jnp.pi, 2 * jnp.pi) - jnp.pi
    return wrapped


def _find_segments(mask: jnp.ndarray) -> list[Tuple[int, int]]:
    """
    Find contiguous segments where mask is True.

    Returns:
        List of (start_idx, end_idx) tuples for each segment
    """
    mask_np = np.array(mask, dtype=bool)

    # Find transitions
    diff = np.diff(np.concatenate(([False], mask_np, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return list(zip(starts, ends))
