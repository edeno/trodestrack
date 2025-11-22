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

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

if TYPE_CHECKING:
    from trodestrack.models.state_layout import StateLayout

from trodestrack.models.state_layout import get_heading_index


def compute_position_rmse(
    positions_true: NDArray[np.float64],
    positions_est: NDArray[np.float64],
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Compute root mean square error for 2D position estimates.

    Parameters
    ----------
    positions_true : NDArray[np.float64]
        Ground truth positions (N, 2) in meters.
    positions_est : NDArray[np.float64]
        Estimated positions (N, 2) in meters.
    valid_mask : NDArray[np.bool_] | None, optional
        Optional validity mask (N,). Only True entries used.

    Returns
    -------
    float
        RMSE in meters.

    Example:
        >>> true_pos = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> est_pos = np.array([[0.1, 0.1], [1.1, 1.1]])
        >>> rmse = compute_position_rmse(true_pos, est_pos)
        >>> print(f"{rmse:.4f} m")
        0.1414 m

        >>> # With validity mask
        >>> valid_mask = np.array([True, False])  # Ignore second sample
        >>> rmse_masked = compute_position_rmse(true_pos, est_pos, valid_mask=valid_mask)
        >>> print(f"{rmse_masked:.4f} m")
        0.1414 m
    """
    if positions_true.shape != positions_est.shape:
        raise ValueError(
            f"Shape mismatch: true {positions_true.shape} vs est {positions_est.shape}"
        )

    if positions_true.shape[1] != 2:
        raise ValueError(f"Expected 2D positions, got shape {positions_true.shape}")

    # Build validity mask: finite values + optional user-provided mask
    valid = np.isfinite(positions_true).all(axis=1) & np.isfinite(positions_est).all(
        axis=1
    )
    if valid_mask is not None:
        if valid_mask.shape[0] != positions_true.shape[0]:
            raise ValueError(
                f"Mask shape {valid_mask.shape} incompatible with positions {positions_true.shape}"
            )
        valid &= valid_mask

    if not np.any(valid):
        raise ValueError("No valid samples remaining after masking and NaN filtering")

    errors = positions_true[valid] - positions_est[valid]
    squared_errors = np.sum(
        errors**2, axis=1
    )  # Euclidean distance squared per timestep
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return float(rmse)


def compute_velocity_rmse(
    velocities_true: NDArray[np.float64],
    velocities_est: NDArray[np.float64],
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Compute root mean square error for 2D velocity estimates.

    Parameters
    ----------
    velocities_true : NDArray[np.float64]
        Ground truth velocities (N, 2) in m/s.
    velocities_est : NDArray[np.float64]
        Estimated velocities (N, 2) in m/s.
    valid_mask : NDArray[np.bool_] | None, optional
        Optional validity mask (N,). Only True entries used.

    Returns
    -------
    float
        RMSE in m/s.

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

    # Build validity mask: finite values + optional user-provided mask
    valid = np.isfinite(velocities_true).all(axis=1) & np.isfinite(velocities_est).all(
        axis=1
    )
    if valid_mask is not None:
        if valid_mask.shape[0] != velocities_true.shape[0]:
            raise ValueError(
                f"Mask shape {valid_mask.shape} incompatible with velocities {velocities_true.shape}"
            )
        valid &= valid_mask

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

    Parameters
    ----------
    headings_true : NDArray[np.float64]
        Ground truth headings (N,) in radians.
    headings_est : NDArray[np.float64]
        Estimated headings (N,) in radians.

    Returns
    -------
    float
        Mean absolute error in radians.

    Example:
        >>> true_heading = np.array([0.0, np.pi / 2, np.pi])
        >>> est_heading = np.array([0.1, np.pi / 2 + 0.1, np.pi - 0.1])
        >>> mae = compute_heading_error(true_heading, est_heading)
        >>> print(f"{mae:.4f} rad ({np.rad2deg(mae):.2f} deg)")
        0.1000 rad (5.73 deg)
    """
    if headings_true.shape != headings_est.shape:
        raise ValueError(
            f"Shape mismatch: true {headings_true.shape} vs est {headings_est.shape}"
        )

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

    Parameters
    ----------
    headings_true : NDArray[np.float64]
        Ground truth headings (N,) in radians.
    headings_est : NDArray[np.float64]
        Estimated headings (N,) in radians.

    Returns
    -------
    float
        Root mean square error in radians.

    Example:
        >>> true_heading = np.array([0.0, np.pi / 2, np.pi])
        >>> est_heading = np.array([0.1, np.pi / 2 + 0.1, np.pi - 0.1])
        >>> rmse = compute_heading_rmse(true_heading, est_heading)
        >>> print(f"{rmse:.4f} rad ({np.rad2deg(rmse):.2f} deg)")
        0.1000 rad (5.73 deg)
    """
    if headings_true.shape != headings_est.shape:
        raise ValueError(
            f"Shape mismatch: true {headings_true.shape} vs est {headings_est.shape}"
        )

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
    layout: StateLayout | None = None,
    heading_idx: int | None = None,
) -> NDArray[np.float64]:
    """Compute Normalized Estimation Error Squared (NEES) for consistency check.

    NEES measures whether estimation errors are consistent with reported covariances.
    For a consistent filter, NEES should follow a chi-squared distribution with
    degrees of freedom equal to state dimension.

    Parameters
    ----------
    states_true : NDArray[np.float64]
        Ground truth states (N, D).
    states_est : NDArray[np.float64]
        Estimated states (N, D).
    covariances_est : NDArray[np.float64]
        Estimated covariances (N, D, D).
    layout : StateLayout or None, optional
        State layout describing index mapping. If provided, automatically
        extracts the heading index for proper angle wrapping. This is the
        recommended approach as it prevents errors from forgetting to specify
        heading_idx manually. Default: None.
    heading_idx : int or None, optional
        **Deprecated**: Use `layout` parameter instead. Index of heading state
        (in radians) that requires angle wrapping. If provided, the heading error
        will be wrapped to [-π, π] before computing NEES. Default: None.

    Returns
    -------
    NDArray[np.float64]
        NEES values per timestep (N,).

    Example:
        >>> # Basic usage (no angles)
        >>> true_state = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> est_state = np.array([[0.1, 0.1], [1.1, 1.1]])
        >>> cov = np.stack([np.eye(2) * 0.1, np.eye(2) * 0.1])
        >>> nees = compute_nees(true_state, est_state, cov)
        >>> print(f"NEES: {nees}")
        NEES: [0.2 0.2]

        >>> # With state layout (recommended for states with heading)
        >>> from trodestrack.models.state_layout import get_layout
        >>> layout = get_layout("2d_full")
        >>> nees = compute_nees(X_truth, X_est, P_est, layout=layout)

        >>> # Legacy approach (deprecated, but still supported)
        >>> nees = compute_nees(X_truth, X_est, P_est, heading_idx=4)

    Notes:
        For a D-dimensional state, NEES ~ chi^2(D) if filter is consistent.
        - 95% confidence interval for D=5: [1.145, 11.07]
        - 95% confidence interval for D=8: [2.733, 15.51]

        If NEES is consistently outside this range, the filter is either:
        - Over-confident (NEES too high): covariance underestimated
        - Under-confident (NEES too low): covariance overestimated

        **Important**: For states containing heading/orientation angles, always
        pass the ``layout`` parameter to ensure proper angle wrapping. Without this,
        NEES values will be incorrectly large when angles wrap through 0°/360°.
    """
    if states_true.shape != states_est.shape:
        raise ValueError(
            f"Shape mismatch: true {states_true.shape} vs est {states_est.shape}"
        )

    N, D = states_true.shape

    if covariances_est.shape != (N, D, D):
        raise ValueError(
            f"Covariance shape mismatch: expected ({N}, {D}, {D}), got {covariances_est.shape}"
        )

    # Extract heading index from layout if provided
    if layout is not None:
        if layout.has_heading_2d:
            heading_idx = get_heading_index(layout)
        # For 3D orientations, we don't currently support angle wrapping
        # (would need quaternion or Euler angle handling)

    nees = np.zeros(N)

    for i in range(N):
        error = states_true[i] - states_est[i]

        # Apply angle wrapping if heading_idx is specified
        if heading_idx is not None:
            # Wrap heading error to [-π, π]
            error[heading_idx] = np.arctan2(
                np.sin(states_true[i, heading_idx] - states_est[i, heading_idx]),
                np.cos(states_true[i, heading_idx] - states_est[i, heading_idx]),
            )

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

    Parameters
    ----------
    innovations : NDArray[np.float64]
        Measurement innovations (N, M).
    innovation_covariances : NDArray[np.float64]
        Innovation covariances (N, M, M).

    Returns
    -------
    NDArray[np.float64]
        NIS values per timestep (N,).

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


def compute_nis_stats(
    nis: NDArray[np.float64],
    measurement_dim: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Summary statistics for NIS consistency check.

    Parameters
    ----------
    nis : NDArray[np.float64]
        NIS values per timestep (N,).
    measurement_dim : int
        Measurement dimension (degrees of freedom for χ²).
    confidence : float, default 0.95
        Confidence level for χ² bounds.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - mean: Mean NIS (should be ~measurement_dim for consistent filter)
        - std: Standard deviation
        - min: Minimum NIS
        - max: Maximum NIS
        - chi2_lower: Lower confidence bound for chi^2(measurement_dim)
        - chi2_upper: Upper confidence bound for chi^2(measurement_dim)
        - pct_in_bounds: Percentage of NIS values within confidence interval
        - confidence: Confidence level used (for reference)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> nis = np.random.chisquare(df=4, size=100)
        >>> stats = compute_nis_stats(nis, measurement_dim=4, confidence=0.95)
        >>> # Mean should be approximately measurement_dim for consistent filter
        >>> 3.0 < stats["mean"] < 5.0
        True
        >>> # Most samples should be within 95% confidence bounds
        >>> stats["pct_in_bounds"] > 90.0
        True
    """
    lower, upper = chi2_bounds(df=measurement_dim, confidence=confidence)
    pct_in_bounds = (
        within_envelope(nis, df=measurement_dim, confidence=confidence) * 100.0
    )

    return {
        "mean": float(np.mean(nis)),
        "std": float(np.std(nis)),
        "min": float(np.min(nis)),
        "max": float(np.max(nis)),
        "chi2_lower": float(lower),
        "chi2_upper": float(upper),
        "pct_in_bounds": float(pct_in_bounds),
        "confidence": float(confidence),
    }


def compute_residual_autocorrelation(
    residuals: NDArray[np.float64],
    max_lag: int = 10,
) -> NDArray[np.float64]:
    """Autocorrelation function (ACF) of residuals to check whiteness.

    For a well-tuned filter, residuals should be white noise (uncorrelated over time).
    Non-zero autocorrelation indicates:
    - Process noise Q too small (under-modeling dynamics)
    - Timing offset between sensors
    - Unmodeled dynamics or correlations

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Residual time series, shape (N,) or (N, M) for multivariate.
    max_lag : int, default 10
        Maximum lag to compute.

    Returns
    -------
    NDArray[np.float64]
        Autocorrelation values for lags 0..max_lag, shape (max_lag+1,) or (M, max_lag+1).

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


def compute_nees_stats(
    nees: NDArray[np.float64],
    state_dim: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute summary statistics for NEES consistency check.

    Parameters
    ----------
    nees : NDArray[np.float64]
        NEES values per timestep (N,).
    state_dim : int
        State dimension (degrees of freedom for χ²).
    confidence : float, default 0.95
        Confidence level for χ² bounds.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - mean: Mean NEES (should be ~state_dim for consistent filter)
        - std: Standard deviation
        - min: Minimum NEES
        - max: Maximum NEES
        - chi2_lower: Lower confidence bound for chi^2(state_dim)
        - chi2_upper: Upper confidence bound for chi^2(state_dim)
        - pct_in_bounds: Percentage of NEES values within confidence interval
        - confidence: Confidence level used (for reference)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> nees = np.random.chisquare(df=5, size=100)
        >>> stats = compute_nees_stats(nees, state_dim=5, confidence=0.95)
        >>> # Mean should be approximately state_dim for consistent filter
        >>> 4.0 < stats["mean"] < 6.0
        True
        >>> # Most samples should be within 95% confidence bounds
        >>> stats["pct_in_bounds"] > 90.0
        True
    """
    lower, upper = chi2_bounds(df=state_dim, confidence=confidence)
    pct_in_bounds = within_envelope(nees, df=state_dim, confidence=confidence) * 100.0

    return {
        "mean": float(np.mean(nees)),
        "std": float(np.std(nees)),
        "min": float(np.min(nees)),
        "max": float(np.max(nees)),
        "chi2_lower": float(lower),
        "chi2_upper": float(upper),
        "pct_in_bounds": float(pct_in_bounds),
        "confidence": float(confidence),
    }


def chi2_bounds(df: int, confidence: float = 0.95) -> tuple[float, float]:
    """Confidence interval for chi-squared distribution.

    Parameters
    ----------
    df : int
        Degrees of freedom (measurement/state dimensionality).
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) for the confidence interval.

    Example:
        >>> lower, upper = chi2_bounds(df=2, confidence=0.95)
        >>> print(f"95% CI for χ²(2): [{lower:.3f}, {upper:.3f}]")
        95% CI for χ²(2): [0.051, 7.378]

        >>> lower, upper = chi2_bounds(df=4, confidence=0.99)
        >>> print(f"99% CI for χ²(4): [{lower:.3f}, {upper:.3f}]")
        99% CI for χ²(4): [0.297, 14.860]

    Notes:
        Common use cases:
        - df=2: Position-only updates (x, y)
        - df=4: Dual-LED updates (x1, y1, x2, y2)
        - df=5: Position + velocity (x, y, vx, vy, θ)
        - df=8: Full state (x, y, vx, vy, θ, b_gz, b_ax, b_ay)

        For NEES/NIS consistency checks, approximately `confidence*100`% of
        values should fall within this interval if the filter is well-calibrated.
    """
    from scipy.stats import chi2

    alpha = 1.0 - confidence
    lower = float(chi2.ppf(alpha / 2, df=df))
    upper = float(chi2.ppf(1.0 - alpha / 2, df=df))

    return lower, upper


def chi2_ci95(df: int) -> tuple[float, float]:
    """Compute 95% confidence interval for chi-squared distribution.

    This is a convenience wrapper around chi2_bounds() with confidence=0.95.

    Parameters
    ----------
    df : int
        Degrees of freedom (measurement/state dimensionality).

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) for 95% CI.

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
    return chi2_bounds(df=df, confidence=0.95)


def within_envelope(
    values: NDArray[np.float64],
    df: int,
    confidence: float = 0.95,
) -> float:
    """Percentage of values within χ² confidence envelope.

    Parameters
    ----------
    values : NDArray[np.float64]
        Values distributed approximately as χ² (e.g., NEES or NIS) (N,).
    df : int
        Degrees of freedom (measurement/state dimensionality).
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        Fraction in [0.0, 1.0] within the envelope.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> nees_values = np.random.chisquare(df=4, size=1000)
        >>> pct = within_envelope(nees_values, df=4, confidence=0.95)
        >>> print(f"{pct * 100:.1f}% within 95% envelope")
        94.8% within 95% envelope

        >>> # With 99% confidence, more values should be within bounds
        >>> pct_99 = within_envelope(nees_values, df=4, confidence=0.99)
        >>> print(f"{pct_99 * 100:.1f}% within 99% envelope")
        99.1% within 99% envelope

    Notes:
        For a well-calibrated filter, approximately `confidence*100`% of NEES
        or NIS values should fall within the chi-squared confidence envelope.
        Significant deviations indicate filter miscalibration:
        - Too many outside upper bound → underconfident filter (P too large)
        - Too many outside lower bound → overconfident filter (P too small)
    """
    lower, upper = chi2_bounds(df=df, confidence=confidence)
    within_bounds = (values >= lower) & (values <= upper)
    return float(np.mean(within_bounds))


def compute_dropout_drift(
    positions: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    t: NDArray[np.float64],
    min_duration_s: float = 5.0,
) -> dict[str, float | None]:
    """Position drift during first contiguous dropout block.

    Measures how far the filter drifts during camera occlusion, which is a
    critical PRD requirement: drift should be ≤0.20 m (20 cm) after 5s dropout
    (updated from 0.15 m based on physical IMU drift limits ~3 cm/s).

    Parameters
    ----------
    positions : NDArray[np.float64]
        Estimated positions (N, 2) in meters.
    valid_mask : NDArray[np.bool_]
        Camera validity mask (N,), False indicates dropout.
    t : NDArray[np.float64]
        Timestamps (N,) in seconds.
    min_duration_s : float, default 5.0
        Minimum dropout duration to analyze (s).

    Returns
    -------
    dict[str, float | None]
        Dict with keys: 'drift_m', 'duration_s', 'start_idx', 'end_idx'.

    Example:
        >>> # Simulate 10s trajectory with 5s dropout at t=3-8s
        >>> t = np.linspace(0, 10, 100)
        >>> positions = np.column_stack([t * 0.1, np.zeros_like(t)])  # Moving at 0.1 m/s
        >>> valid_mask = (t < 3.0) | (t >= 8.0)  # Dropout from 3-8s
        >>> result = compute_dropout_drift(positions, valid_mask, t, min_duration_s=4.0)
        >>> # Drift should be ~0.5 m (5s * 0.1 m/s)
        >>> 0.4 < result["drift_m"] < 0.6
        True
        >>> np.isclose(result["duration_s"], 5.0, atol=0.1)
        True

    Notes:
        PRD Acceptance Criteria (§4.2, updated):
        - After 5s camera dropout, IMU-only drift should be ≤0.20 m (20 cm)
        - Previous 0.15 m requirement was at physical limits (~3 cm/s drift rate)

        This function identifies the FIRST contiguous dropout block that
        exceeds min_duration_s and measures drift from block start to end.
    """
    if positions.shape[0] != valid_mask.shape[0] or positions.shape[0] != t.shape[0]:
        raise ValueError(
            f"Shape mismatch: positions {positions.shape}, mask {valid_mask.shape}, time {t.shape}"
        )

    # Find contiguous dropout blocks
    dropout = ~valid_mask
    diff = np.diff(dropout.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]  # Dropout begins
    ends = np.where(diff == -1)[0]  # Dropout ends

    # Find first block with duration >= min_duration_s
    for start_idx, end_idx in zip(starts, ends, strict=False):
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
