"""Quality assurance metrics for tracking performance evaluation.

This module provides metrics to validate filter accuracy and consistency against
PRD requirements:
- Position RMSE <= 2 cm
- Velocity RMSE <= 10 cm/s
- Heading error <= 7 degrees

Additionally provides NEES (Normalized Estimation Error Squared) for filter
consistency checks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_position_rmse(
    positions_true: NDArray[np.float64],
    positions_est: NDArray[np.float64],
) -> float:
    """Compute root mean square error for 2D position estimates.

    Args:
        positions_true: Ground truth positions, shape (N, 2) in cm
        positions_est: Estimated positions, shape (N, 2) in cm

    Returns:
        RMSE in cm

    Example:
        >>> true_pos = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> est_pos = np.array([[0.1, 0.1], [1.1, 1.1]])
        >>> rmse = compute_position_rmse(true_pos, est_pos)
        >>> print(f"{rmse:.2f} cm")
        0.14 cm
    """
    if positions_true.shape != positions_est.shape:
        raise ValueError(
            f"Shape mismatch: true {positions_true.shape} vs est {positions_est.shape}"
        )

    if positions_true.shape[1] != 2:
        raise ValueError(f"Expected 2D positions, got shape {positions_true.shape}")

    errors = positions_true - positions_est
    squared_errors = np.sum(errors**2, axis=1)  # Euclidean distance squared per timestep
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return float(rmse)


def compute_velocity_rmse(
    velocities_true: NDArray[np.float64],
    velocities_est: NDArray[np.float64],
) -> float:
    """Compute root mean square error for 2D velocity estimates.

    Args:
        velocities_true: Ground truth velocities, shape (N, 2) in cm/s
        velocities_est: Estimated velocities, shape (N, 2) in cm/s

    Returns:
        RMSE in cm/s

    Example:
        >>> true_vel = np.array([[10.0, 0.0], [10.0, 0.0]])
        >>> est_vel = np.array([[10.5, 0.2], [10.3, -0.1]])
        >>> rmse = compute_velocity_rmse(true_vel, est_vel)
        >>> print(f"{rmse:.2f} cm/s")
        0.44 cm/s
    """
    if velocities_true.shape != velocities_est.shape:
        raise ValueError(
            f"Shape mismatch: true {velocities_true.shape} vs est {velocities_est.shape}"
        )

    if velocities_true.shape[1] != 2:
        raise ValueError(f"Expected 2D velocities, got shape {velocities_true.shape}")

    errors = velocities_true - velocities_est
    squared_errors = np.sum(errors**2, axis=1)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return float(rmse)


def compute_heading_error(
    headings_true: NDArray[np.float64],
    headings_est: NDArray[np.float64],
) -> float:
    """Compute mean absolute heading error with proper angle wrapping.

    Args:
        headings_true: Ground truth headings, shape (N,) in radians
        headings_est: Estimated headings, shape (N,) in radians

    Returns:
        Mean absolute error in degrees

    Example:
        >>> true_heading = np.array([0.0, np.pi/2, np.pi])
        >>> est_heading = np.array([0.1, np.pi/2 + 0.1, np.pi - 0.1])
        >>> mae = compute_heading_error(true_heading, est_heading)
        >>> print(f"{mae:.2f} deg")
        5.73 deg
    """
    if headings_true.shape != headings_est.shape:
        raise ValueError(f"Shape mismatch: true {headings_true.shape} vs est {headings_est.shape}")

    if headings_true.ndim != 1:
        raise ValueError(f"Expected 1D headings, got shape {headings_true.shape}")

    # Compute wrapped difference: map to (-π, π]
    diff = headings_true - headings_est
    diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))

    # Mean absolute error in radians
    mae_rad = np.mean(np.abs(diff_wrapped))

    # Convert to degrees
    mae_deg = np.rad2deg(mae_rad)

    return float(mae_deg)


def compute_nees(
    states_true: NDArray[np.float64],
    states_est: NDArray[np.float64],
    covariances_est: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Normalized Estimation Error Squared (NEES) for consistency check.

    NEES measures whether estimation errors are consistent with reported covariances.
    For a consistent filter, NEES should follow a chi-squared distribution with
    degrees of freedom equal to state dimension.

    Args:
        states_true: Ground truth states, shape (N, D)
        states_est: Estimated states, shape (N, D)
        covariances_est: Estimated covariances, shape (N, D, D)

    Returns:
        NEES values per timestep, shape (N,)

    Example:
        >>> true_state = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> est_state = np.array([[0.1, 0.1], [1.1, 1.1]])
        >>> cov = np.stack([np.eye(2) * 0.1, np.eye(2) * 0.1])
        >>> nees = compute_nees(true_state, est_state, cov)
        >>> print(f"NEES: {nees}")
        NEES: [0.2 0.2]

    Notes:
        For a D-dimensional state, NEES ~ chi^2(D) if filter is consistent.
        - 95% confidence interval for D=5: [1.145, 11.07]
        - 95% confidence interval for D=8: [2.733, 15.51]

        If NEES is consistently outside this range, the filter is either:
        - Over-confident (NEES too high): covariance underestimated
        - Under-confident (NEES too low): covariance overestimated
    """
    if states_true.shape != states_est.shape:
        raise ValueError(f"Shape mismatch: true {states_true.shape} vs est {states_est.shape}")

    N, D = states_true.shape

    if covariances_est.shape != (N, D, D):
        raise ValueError(
            f"Covariance shape mismatch: expected ({N}, {D}, {D}), " f"got {covariances_est.shape}"
        )

    nees = np.zeros(N)

    for i in range(N):
        error = states_true[i] - states_est[i]
        cov = covariances_est[i]

        # NEES = e^T * P^{-1} * e
        # Use solve instead of inv for numerical stability
        try:
            cov_inv_error = np.linalg.solve(cov, error)
            nees[i] = error @ cov_inv_error
        except np.linalg.LinAlgError:
            # Singular covariance - filter is broken
            nees[i] = np.inf

    return nees


def compute_nees_stats(nees: NDArray[np.float64], state_dim: int) -> dict[str, float]:
    """Compute summary statistics for NEES consistency check.

    Args:
        nees: NEES values per timestep, shape (N,)
        state_dim: Dimension of state (degrees of freedom for chi-squared)

    Returns:
        Dictionary with keys:
        - mean: Mean NEES (should be ~state_dim for consistent filter)
        - std: Standard deviation
        - min: Minimum NEES
        - max: Maximum NEES
        - chi2_lower_95: Lower 95% confidence bound for chi^2(state_dim)
        - chi2_upper_95: Upper 95% confidence bound for chi^2(state_dim)
        - pct_in_bounds: Percentage of NEES values within 95% CI

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> nees = np.random.chisquare(df=5, size=100)
        >>> stats = compute_nees_stats(nees, state_dim=5)
        >>> # Mean should be approximately state_dim for consistent filter
        >>> 4.0 < stats['mean'] < 6.0
        True
        >>> # Most samples should be within 95% confidence bounds
        >>> stats['pct_in_bounds'] > 90.0
        True
    """
    from scipy.stats import chi2

    # Chi-squared 95% confidence interval
    lower = chi2.ppf(0.025, df=state_dim)
    upper = chi2.ppf(0.975, df=state_dim)

    in_bounds = np.sum((nees >= lower) & (nees <= upper))
    pct_in_bounds = 100.0 * in_bounds / len(nees)

    return {
        "mean": float(np.mean(nees)),
        "std": float(np.std(nees)),
        "min": float(np.min(nees)),
        "max": float(np.max(nees)),
        "chi2_lower_95": float(lower),
        "chi2_upper_95": float(upper),
        "pct_in_bounds": float(pct_in_bounds),
    }
