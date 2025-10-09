"""Quality assurance metrics for tracking performance evaluation.

This module provides metrics to validate filter accuracy and consistency against
PRD requirements (all in SI units):
- Position RMSE <= 0.02 m (2 cm)
- Velocity RMSE <= 0.10 m/s (10 cm/s)
- Heading error <= 0.122 rad (7 degrees)

Additionally provides NEES (Normalized Estimation Error Squared) for filter
consistency checks.

All functions use SI units (meters, m/s, radians) for inputs and outputs.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray


def compute_position_rmse(
    positions_true: NDArray[np.float64],
    positions_est: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Compute root mean square error for 2D position estimates.

    Args:
        positions_true: Ground truth positions, shape (N, 2) in meters
        positions_est: Estimated positions, shape (N, 2) in meters
        mask: Optional validity mask, shape (N,). Only valid (True) entries used.

    Returns:
        RMSE in meters

    Example:
        >>> true_pos = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> est_pos = np.array([[0.1, 0.1], [1.1, 1.1]])
        >>> rmse = compute_position_rmse(true_pos, est_pos)
        >>> print(f"{rmse:.4f} m")
        0.1414 m

        >>> # With mask
        >>> mask = np.array([True, False])  # Ignore second sample
        >>> rmse_masked = compute_position_rmse(true_pos, est_pos, mask=mask)
        >>> print(f"{rmse_masked:.4f} m")
        0.1414 m
    """
    if positions_true.shape != positions_est.shape:
        raise ValueError(
            f"Shape mismatch: true {positions_true.shape} vs est {positions_est.shape}"
        )

    if positions_true.shape[1] != 2:
        raise ValueError(f"Expected 2D positions, got shape {positions_true.shape}")

    # Build validity mask: finite values + optional user mask
    valid = np.isfinite(positions_true).all(axis=1) & np.isfinite(positions_est).all(axis=1)
    if mask is not None:
        if mask.shape[0] != positions_true.shape[0]:
            raise ValueError(
                f"Mask shape {mask.shape} incompatible with positions {positions_true.shape}"
            )
        valid &= mask

    if not np.any(valid):
        raise ValueError("No valid samples remaining after masking and NaN filtering")

    errors = positions_true[valid] - positions_est[valid]
    squared_errors = np.sum(errors**2, axis=1)  # Euclidean distance squared per timestep
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return float(rmse)


def compute_velocity_rmse(
    velocities_true: NDArray[np.float64],
    velocities_est: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Compute root mean square error for 2D velocity estimates.

    Args:
        velocities_true: Ground truth velocities, shape (N, 2) in m/s
        velocities_est: Estimated velocities, shape (N, 2) in m/s
        mask: Optional validity mask, shape (N,). Only valid (True) entries used.

    Returns:
        RMSE in m/s

    Example:
        >>> true_vel = np.array([[0.10, 0.0], [0.10, 0.0]])
        >>> est_vel = np.array([[0.105, 0.002], [0.103, -0.001]])
        >>> rmse = compute_velocity_rmse(true_vel, est_vel)
        >>> print(f"{rmse:.4f} m/s")
        0.0044 m/s
    """
    if velocities_true.shape != velocities_est.shape:
        raise ValueError(
            f"Shape mismatch: true {velocities_true.shape} vs est {velocities_est.shape}"
        )

    if velocities_true.shape[1] != 2:
        raise ValueError(f"Expected 2D velocities, got shape {velocities_true.shape}")

    # Build validity mask: finite values + optional user mask
    valid = np.isfinite(velocities_true).all(axis=1) & np.isfinite(velocities_est).all(axis=1)
    if mask is not None:
        if mask.shape[0] != velocities_true.shape[0]:
            raise ValueError(
                f"Mask shape {mask.shape} incompatible with velocities {velocities_true.shape}"
            )
        valid &= mask

    if not np.any(valid):
        raise ValueError("No valid samples remaining after masking and NaN filtering")

    errors = velocities_true[valid] - velocities_est[valid]
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
        Mean absolute error in radians

    Example:
        >>> true_heading = np.array([0.0, np.pi/2, np.pi])
        >>> est_heading = np.array([0.1, np.pi/2 + 0.1, np.pi - 0.1])
        >>> mae = compute_heading_error(true_heading, est_heading)
        >>> print(f"{mae:.4f} rad ({np.rad2deg(mae):.2f} deg)")
        0.1000 rad (5.73 deg)
    """
    if headings_true.shape != headings_est.shape:
        raise ValueError(f"Shape mismatch: true {headings_true.shape} vs est {headings_est.shape}")

    if headings_true.ndim != 1:
        raise ValueError(f"Expected 1D headings, got shape {headings_true.shape}")

    # Compute wrapped difference: map to (-π, π]
    diff = headings_true - headings_est
    diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))

    # Mean absolute error in radians (SI unit)
    mae_rad = np.mean(np.abs(diff_wrapped))

    return float(mae_rad)


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
        # Use Cholesky + triangular solves for stability on near-PSD matrices
        try:
            # Compute L such that L @ L.T = cov
            L = np.linalg.cholesky(cov)
            # Solve L @ y = error for y
            y = scipy.linalg.solve_triangular(L, error, lower=True)
            # NEES = ||y||^2 = e^T @ inv(L @ L.T) @ e
            nees[i] = y @ y
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
        # Use Cholesky + triangular solves for stability on near-PSD matrices
        try:
            # Compute L such that L @ L.T = cov
            L = np.linalg.cholesky(cov)
            # Solve L @ y = innov for y
            y = scipy.linalg.solve_triangular(L, innov, lower=True)
            # NIS = ||y||^2 = r^T @ inv(L @ L.T) @ r
            nis[i] = y @ y
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
            # Constant residuals (degenerate case): ACF[0]=1, rest NaN
            # This preserves identity at lag 0 but indicates "no information"
            return np.concatenate(([1.0], np.full(max_lag, np.nan)))

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


def chi2_ci95(df: int) -> tuple[float, float]:
    """Compute 95% confidence interval for chi-squared distribution.

    Args:
        df: Degrees of freedom (measurement/state dimensionality)

    Returns:
        Tuple of (lower_bound, upper_bound) for 95% CI

    Example:
        >>> lower, upper = chi2_ci95(df=2)
        >>> print(f"95% CI for χ²(2): [{lower:.3f}, {upper:.3f}]")
        95% CI for χ²(2): [0.051, 7.378]

        >>> lower, upper = chi2_ci95(df=4)
        >>> print(f"95% CI for χ²(4): [{lower:.3f}, {upper:.3f}]")
        95% CI for χ²(4): [0.484, 11.143]

    Notes:
        Common use cases:
        - df=2: Position-only updates (x, y)
        - df=4: Dual-LED updates (x1, y1, x2, y2)
        - df=5: Position + velocity (x, y, vx, vy, θ)
        - df=8: Full state (x, y, vx, vy, θ, b_gz, b_ax, b_ay)

        For NEES/NIS consistency checks, approximately 95% of values should
        fall within this interval if the filter is well-calibrated.
    """
    from scipy.stats import chi2

    lower = float(chi2.ppf(0.025, df=df))
    upper = float(chi2.ppf(0.975, df=df))

    return lower, upper


def compute_dropout_drift(
    positions: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    t: NDArray[np.float64],
    min_duration_s: float = 5.0,
) -> dict[str, float | None]:
    """Compute position drift during first contiguous dropout block.

    Measures how far the filter drifts during camera occlusion, which is a
    critical PRD requirement: drift should be ≤0.15 m (15 cm) after 5s dropout.

    Args:
        positions: Estimated positions over time, shape (N, 2) in meters
        valid_mask: Camera validity mask, shape (N,). False = dropout
        t: Timestamps, shape (N,) in seconds
        min_duration_s: Minimum dropout duration to analyze (default: 5.0s)

    Returns:
        Dictionary with keys:
        - drift_m: Euclidean drift from start to end of dropout in meters (None if no dropout found)
        - duration_s: Duration of dropout in seconds (None if no dropout found)
        - start_idx: Index where dropout starts (None if no dropout found)
        - end_idx: Index where dropout ends (None if no dropout found)

    Example:
        >>> # Simulate 10s trajectory with 5s dropout at t=3-8s
        >>> t = np.linspace(0, 10, 100)
        >>> positions = np.column_stack([t * 0.1, np.zeros_like(t)])  # Moving at 0.1 m/s
        >>> mask = (t < 3.0) | (t >= 8.0)  # Dropout from 3-8s
        >>> result = compute_dropout_drift(positions, mask, t, min_duration_s=4.0)
        >>> # Drift should be ~0.5 m (5s * 0.1 m/s)
        >>> 0.4 < result['drift_m'] < 0.6
        True
        >>> np.isclose(result['duration_s'], 5.0, atol=0.1)
        True

    Notes:
        PRD Acceptance Criteria (§4.2):
        - After 5s camera dropout, IMU-only drift should be ≤0.15 m (15 cm)

        This function identifies the FIRST contiguous dropout block that
        exceeds min_duration_s and measures drift from block start to end.
    """
    if positions.shape[0] != valid_mask.shape[0] or positions.shape[0] != t.shape[0]:
        raise ValueError(
            f"Shape mismatch: positions {positions.shape}, "
            f"mask {valid_mask.shape}, time {t.shape}"
        )

    # Find contiguous dropout blocks
    dropout = ~valid_mask
    diff = np.diff(dropout.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]  # Dropout begins
    ends = np.where(diff == -1)[0]  # Dropout ends

    # Find first block with duration >= min_duration_s
    for start_idx, end_idx in zip(starts, ends):
        duration = t[end_idx - 1] - t[start_idx]
        if duration >= min_duration_s:
            # Measure drift from start to end
            pos_start = positions[start_idx]
            pos_end = positions[end_idx - 1]
            drift = np.linalg.norm(pos_end - pos_start)

            return {
                "drift_m": float(drift),
                "duration_s": float(duration),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
            }

    # No qualifying dropout found
    return {
        "drift_m": None,
        "duration_s": None,
        "start_idx": None,
        "end_idx": None,
    }
