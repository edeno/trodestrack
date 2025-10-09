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


def compute_heading_rmse(
    headings_true: NDArray[np.float64],
    headings_est: NDArray[np.float64],
) -> float:
    """Compute root mean square heading error with proper angle wrapping.

    Args:
        headings_true: Ground truth headings, shape (N,) in radians
        headings_est: Estimated headings, shape (N,) in radians

    Returns:
        Root mean square error in radians

    Example:
        >>> true_heading = np.array([0.0, np.pi/2, np.pi])
        >>> est_heading = np.array([0.1, np.pi/2 + 0.1, np.pi - 0.1])
        >>> rmse = compute_heading_rmse(true_heading, est_heading)
        >>> print(f"{rmse:.4f} rad ({np.rad2deg(rmse):.2f} deg)")
        0.1000 rad (5.73 deg)
    """
    if headings_true.shape != headings_est.shape:
        raise ValueError(f"Shape mismatch: true {headings_true.shape} vs est {headings_est.shape}")

    if headings_true.ndim != 1:
        raise ValueError(f"Expected 1D headings, got shape {headings_true.shape}")

    # Compute wrapped difference: map to (-π, π]
    diff = headings_true - headings_est
    diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))

    # Root mean square error in radians
    rmse_rad = np.sqrt(np.mean(diff_wrapped**2))

    return float(rmse_rad)


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


def compute_nis(
    innovations: NDArray[np.float64],
    innovation_covariances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Normalized Innovation Squared (NIS) for measurement consistency check.

    NIS measures whether measurement innovations (residuals) are consistent with
    predicted innovation covariances. For a consistent filter, NIS should follow
    a chi-squared distribution with degrees of freedom equal to measurement dimension.

    Args:
        innovations: Measurement innovations (residuals), shape (N, M)
        innovation_covariances: Innovation covariances, shape (N, M, M)

    Returns:
        NIS values per timestep, shape (N,)

    Example:
        >>> innovations = np.array([[0.1, 0.1], [0.2, -0.1]])
        >>> cov = np.stack([np.eye(2) * 0.1, np.eye(2) * 0.1])
        >>> nis = compute_nis(innovations, cov)
        >>> print(f"NIS: {nis}")
        NIS: [0.2 0.5]

    Notes:
        For an M-dimensional measurement, NIS ~ chi^2(M) if filter is consistent.
        - 95% confidence interval for M=2: [0.051, 7.378]
        - 95% confidence interval for M=4: [0.484, 11.143]

        If NIS is consistently outside this range, the filter is either:
        - Over-confident (NIS too high): measurement noise R underestimated
        - Under-confident (NIS too low): measurement noise R overestimated
    """
    if innovations.shape[0] != innovation_covariances.shape[0]:
        raise ValueError(
            f"Shape mismatch: innovations {innovations.shape} vs "
            f"covariances {innovation_covariances.shape}"
        )

    N, M = innovations.shape

    if innovation_covariances.shape != (N, M, M):
        raise ValueError(
            f"Innovation covariance shape mismatch: expected ({N}, {M}, {M}), "
            f"got {innovation_covariances.shape}"
        )

    nis = np.zeros(N)

    for i in range(N):
        innov = innovations[i]
        cov = innovation_covariances[i]

        # NIS = r^T * S^{-1} * r
        # Use solve instead of inv for numerical stability
        try:
            cov_inv_innov = np.linalg.solve(cov, innov)
            nis[i] = innov @ cov_inv_innov
        except np.linalg.LinAlgError:
            # Singular covariance - filter is broken
            nis[i] = np.inf

    return nis


def compute_nis_stats(nis: NDArray[np.float64], measurement_dim: int) -> dict[str, float]:
    """Compute summary statistics for NIS consistency check.

    Args:
        nis: NIS values per timestep, shape (N,)
        measurement_dim: Dimension of measurement (degrees of freedom for chi-squared)

    Returns:
        Dictionary with keys:
        - mean: Mean NIS (should be ~measurement_dim for consistent filter)
        - std: Standard deviation
        - min: Minimum NIS
        - max: Maximum NIS
        - chi2_lower_95: Lower 95% confidence bound for chi^2(measurement_dim)
        - chi2_upper_95: Upper 95% confidence bound for chi^2(measurement_dim)
        - pct_in_bounds: Percentage of NIS values within 95% CI

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> nis = np.random.chisquare(df=4, size=100)
        >>> stats = compute_nis_stats(nis, measurement_dim=4)
        >>> # Mean should be approximately measurement_dim for consistent filter
        >>> 3.0 < stats['mean'] < 5.0
        True
        >>> # Most samples should be within 95% confidence bounds
        >>> stats['pct_in_bounds'] > 90.0
        True
    """
    from scipy.stats import chi2

    # Chi-squared 95% confidence interval
    lower = chi2.ppf(0.025, df=measurement_dim)
    upper = chi2.ppf(0.975, df=measurement_dim)

    in_bounds = np.sum((nis >= lower) & (nis <= upper))
    pct_in_bounds = 100.0 * in_bounds / len(nis)

    return {
        "mean": float(np.mean(nis)),
        "std": float(np.std(nis)),
        "min": float(np.min(nis)),
        "max": float(np.max(nis)),
        "chi2_lower_95": float(lower),
        "chi2_upper_95": float(upper),
        "pct_in_bounds": float(pct_in_bounds),
    }


def compute_residual_autocorrelation(
    residuals: NDArray[np.float64],
    max_lag: int = 10,
) -> NDArray[np.float64]:
    """Compute autocorrelation function (ACF) of residuals to check whiteness.

    For a well-tuned filter, residuals should be white noise (uncorrelated over time).
    Non-zero autocorrelation indicates:
    - Process noise Q too small (under-modeling dynamics)
    - Timing offset between sensors
    - Unmodeled dynamics or correlations

    Args:
        residuals: Residual time series, shape (N,) or (N, M) for multivariate
        max_lag: Maximum lag to compute (default: 10)

    Returns:
        Autocorrelation values for lags 0 to max_lag, shape (max_lag+1,) or (M, max_lag+1)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> white_noise = np.random.randn(100)
        >>> acf = compute_residual_autocorrelation(white_noise, max_lag=5)
        >>> # Lag 0 should be 1.0 (perfect correlation with self)
        >>> np.isclose(acf[0], 1.0)
        True
        >>> # Higher lags should be near zero for white noise
        >>> np.all(np.abs(acf[1:]) < 0.3)  # Loose bound for small sample
        True

    Notes:
        Interpretation:
        - ACF[0] = 1.0 always (correlation with self)
        - ACF[k] ≈ 0 for k > 0 indicates whiteness
        - Significant ACF[1] indicates lag-1 correlation (most common issue)
        - 95% confidence bounds: ± 1.96 / sqrt(N) for large N
    """
    if residuals.ndim == 1:
        # Univariate residuals
        N = len(residuals)
        mean = np.mean(residuals)
        var = np.var(residuals, ddof=1)

        if var == 0:
            # Constant residuals (degenerate case)
            return np.zeros(max_lag + 1)

        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[lag] = 1.0
            else:
                # Autocorrelation at lag k: E[(X_t - μ)(X_{t+k} - μ)] / σ²
                cov = np.mean((residuals[: N - lag] - mean) * (residuals[lag:] - mean))
                acf[lag] = cov / var

        return acf

    else:
        # Multivariate residuals: compute ACF for each dimension
        M = residuals.shape[1]
        acf_all = np.zeros((M, max_lag + 1))
        for m in range(M):
            acf_all[m] = compute_residual_autocorrelation(residuals[:, m], max_lag)
        return acf_all


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
