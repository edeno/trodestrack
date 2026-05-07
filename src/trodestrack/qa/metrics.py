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
        >>> rmse_masked = compute_position_rmse(
        ...     true_pos, est_pos, valid_mask=valid_mask
        ... )
        >>> print(f"{rmse_masked:.4f} m")
        0.1414 m
    """
    if positions_true.shape != positions_est.shape:
        raise ValueError(
            f"Shape mismatch: true {positions_true.shape} vs est {positions_est.shape}"
        )

    # Check ndim before indexing shape[1] — a 1-D input would otherwise
    # raise IndexError instead of the documented ValueError.
    if positions_true.ndim != 2 or positions_true.shape[1] != 2:
        raise ValueError(
            f"Expected positions of shape (N, 2); got shape {positions_true.shape}."
        )

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

    # Check ndim before indexing shape[1] — a 1-D input would otherwise
    # raise IndexError instead of the documented ValueError.
    if velocities_true.ndim != 2 or velocities_true.shape[1] != 2:
        raise ValueError(
            f"Expected velocities of shape (N, 2); got shape {velocities_true.shape}."
        )

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

    # Mean absolute error in radians (SI unit). Drop non-finite samples so
    # a single NaN/inf in either input does not poison the summary —
    # compute_position_rmse and compute_velocity_rmse already filter NaNs,
    # and the QA report relies on that contract for both surfaces. Match
    # those helpers and raise if no finite samples remain so the QA
    # report can't silently embed a NaN heading metric.
    finite = np.isfinite(diff_wrapped)
    if not np.any(finite):
        raise ValueError(
            "No valid samples remaining after NaN filtering in heading "
            "MAE; both inputs must share at least one finite-paired sample."
        )
    mae_rad = np.mean(np.abs(diff_wrapped[finite]))

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

    # Root mean square error in radians. Drop non-finite samples so a
    # single NaN/inf in either input does not poison the summary; match
    # compute_position_rmse / compute_velocity_rmse and raise if no
    # finite-paired sample remains so the QA report can't silently embed
    # a NaN heading RMSE.
    finite = np.isfinite(diff_wrapped)
    if not np.any(finite):
        raise ValueError(
            "No valid samples remaining after NaN filtering in heading "
            "RMSE; both inputs must share at least one finite-paired sample."
        )
    rmse_rad = np.sqrt(np.mean(diff_wrapped[finite] ** 2))

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
        For a D-dimensional state, NEES ~ chi^2(D) if the filter is consistent.
        Central 95% intervals (chi2.ppf at 0.025 / 0.975), matching the
        helper in :func:`get_chi2_confidence_interval`:

        - D=5:  [0.8312, 12.8325]
        - D=8:  [2.1797, 17.5345]
        - D=10: [3.2470, 20.4832]
        - D=14: [5.6287, 26.1189]
        - D=15: [6.2621, 27.4884]
        - D=16: [6.9077, 28.8454]

        If NEES is consistently outside this range, the filter is either:
        - Over-confident (NEES too high): covariance underestimated
        - Under-confident (NEES too low): covariance overestimated

        **Heading / orientation handling.** When ``layout`` describes a
        scalar 2D heading (``layout.has_heading_2d``), this function
        wraps the heading-component residual to ``[-π, π]`` so the NEES
        is not inflated by 0°/360° wraparound. For 3D-orientation
        layouts (Euler tuples or quaternions, i.e. ``layout.heading_idx``
        is a 3- or 4-tuple), no orientation residual handling is
        applied — the orientation components enter the residual
        unwrapped, which is generally only meaningful when truth and
        estimate are referenced to the same parameterization without
        sign flips. Treat 3D NEES from this helper as an approximate
        diagnostic, not a calibrated chi-square test, and prefer
        per-component diagnostics for 3D orientation.
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


def _validate_finite_1d_samples(values: NDArray[np.float64], name: str) -> None:
    """Reject empty or non-finite-only sample arrays for QA stats helpers.

    Common precondition for ``compute_nees_stats`` / ``compute_nis_stats``
    / ``within_envelope`` — np.mean / std / min / max return NaN/inf on
    these inputs, and the QA report would silently embed a misleading
    summary.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must have at least one sample, got an empty array.")
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError(
            f"{name} contains no finite samples (all NaN/inf); cannot summarize."
        )


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
    _validate_finite_1d_samples(nis, name="nis")
    nis_finite = np.asarray(nis)[np.isfinite(nis)]

    lower, upper = chi2_bounds(df=measurement_dim, confidence=confidence)
    pct_in_bounds = (
        within_envelope(nis_finite, df=measurement_dim, confidence=confidence) * 100.0
    )

    return {
        "mean": float(np.mean(nis_finite)),
        "std": float(np.std(nis_finite)),
        "min": float(np.min(nis_finite)),
        "max": float(np.max(nis_finite)),
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
    if not isinstance(max_lag, int) or max_lag < 0:
        raise ValueError(f"max_lag must be a non-negative integer; got {max_lag!r}.")

    if residuals.ndim == 1:
        # Univariate residuals
        N = len(residuals)
        # Need at least max_lag + 2 samples so the slices residuals[:N-lag]
        # and residuals[lag:] are both non-empty *and* the variance is well
        # defined (var uses ddof=1). Without this, lag=N produces an empty
        # slice (NaN) and lag>N raises an opaque broadcasting error.
        if max_lag >= N:
            raise ValueError(
                f"max_lag ({max_lag}) must be < N ({N}); residuals have "
                "too few samples to compute autocorrelation at the "
                "requested lag. Reduce max_lag or supply a longer series."
            )
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
    _validate_finite_1d_samples(nees, name="nees")
    nees_finite = np.asarray(nees)[np.isfinite(nees)]

    lower, upper = chi2_bounds(df=state_dim, confidence=confidence)
    pct_in_bounds = (
        within_envelope(nees_finite, df=state_dim, confidence=confidence) * 100.0
    )

    return {
        "mean": float(np.mean(nees_finite)),
        "std": float(np.std(nees_finite)),
        "min": float(np.min(nees_finite)),
        "max": float(np.max(nees_finite)),
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
    # scipy.stats.chi2.ppf returns NaN for df <= 0, which the QA report
    # would otherwise embed as nan chi2_lower / chi2_upper / 0% in_bounds
    # entries. Reject up front with a precise message.
    if not isinstance(df, (int, np.integer)) or df < 1:
        raise ValueError(
            f"df must be a positive integer (degrees of freedom); got {df!r}."
        )
    if not np.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValueError(
            f"confidence must be a finite value in (0, 1); got {confidence!r}."
        )

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
        - Too many outside upper bound → overconfident filter (P too small);
          actual error² is larger than the covariance reports
        - Too many outside lower bound → underconfident filter (P too large);
          actual error² is smaller than the covariance reports
    """
    _validate_finite_1d_samples(values, name="values")
    values_finite = np.asarray(values)[np.isfinite(values)]

    lower, upper = chi2_bounds(df=df, confidence=confidence)
    within_bounds = (values_finite >= lower) & (values_finite <= upper)
    return float(np.mean(within_bounds))


def compute_dropout_drift(
    positions: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    t: NDArray[np.float64],
    min_duration_s: float = 5.0,
) -> dict[str, float | None]:
    """Position drift during first contiguous dropout block.

    Measures how far the filter drifts during camera occlusion, which is a
    critical PRD requirement: drift should be ≤3.5 m after 5s dropout
    on production hardware (consumer-grade IMU, 95th percentile bound).
    In simulation the filter achieves ~11 cm; see ``test_ekf_long_dropout_drift``.

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
        >>> positions = np.column_stack(
        ...     [t * 0.1, np.zeros_like(t)]
        ... )  # Moving at 0.1 m/s
        >>> valid_mask = (t < 3.0) | (t >= 8.0)  # Dropout from 3-8s
        >>> result = compute_dropout_drift(positions, valid_mask, t, min_duration_s=4.0)
        >>> # Drift should be ~0.5 m (5s * 0.1 m/s)
        >>> 0.4 < result["drift_m"] < 0.6
        True
        >>> np.isclose(result["duration_s"], 5.0, atol=0.1)
        True

    Notes:
        PRD Acceptance Criteria (§4.2, updated):
        - After 5s camera dropout, IMU-only drift should be ≤3.5 m
          (production hardware bound; simulation achieves ≤15 cm)
        - Previous 0.15-0.20 m requirements were at physical limits of
          consumer-grade IMUs (~3 cm/s drift rate)

        This function identifies the FIRST contiguous dropout block that
        exceeds min_duration_s and measures drift from block start to end.
    """
    valid_mask_arr = np.asarray(valid_mask)
    # Reject non-1D masks. ``(N, 1)`` (a common shape coming out of
    # column-vector loading or one-hot conversion) silently passed the
    # ``shape[0]`` check, then ``np.diff(..., axis=-1)`` operated along
    # the wrong axis and the function returned "no qualifying dropout"
    # for what was actually a real dropout — masking a PRD-relevant
    # drift failure. Mirror the stricter contract used by the plotting
    # layer (``qa.plots._validate_optional_bool_mask``) here.
    if valid_mask_arr.ndim != 1:
        raise ValueError(
            f"valid_mask must be 1-D (N,); got shape {valid_mask_arr.shape}."
        )
    if (
        positions.shape[0] != valid_mask_arr.shape[0]
        or positions.shape[0] != t.shape[0]
    ):
        raise ValueError(
            f"Shape mismatch: positions {positions.shape}, mask {valid_mask_arr.shape}, time {t.shape}"
        )

    # Find contiguous dropout blocks
    dropout = ~valid_mask_arr
    diff = np.diff(dropout.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]  # Dropout begins (first invalid sample)
    ends = np.where(diff == -1)[0]  # Dropout ends (first valid sample after, exclusive)

    # Each contiguous dropout block covers samples [start_idx, end_idx).
    # The duration of the *interval covered by those samples* is
    # ``t[end_idx] - t[start_idx]`` when end_idx < N (the block ends when
    # the next valid sample arrives). For example, a 150-frame dropout at
    # 30 Hz spanning samples [s, s+150) has duration t[s+150] - t[s] = 5.0 s,
    # whereas ``t[end_idx - 1] - t[start_idx]`` only covers 149 sample
    # intervals (≈4.967 s) and would silently fail a min_duration_s=5.0
    # threshold for an exactly-5 s block.
    #
    # If the dropout extends to the end of the trace (end_idx == N) there
    # is no "next valid sample"; extrapolate by adding the local sample
    # period (median dt over the block, or the global median if the block
    # is too short for a local estimate).
    n = len(t)
    for start_idx, end_idx in zip(starts, ends, strict=False):
        if end_idx < n:
            duration = float(t[end_idx] - t[start_idx])
        elif end_idx - start_idx >= 2:
            local_dt = float(np.median(np.diff(t[start_idx:end_idx])))
            duration = float(t[end_idx - 1] - t[start_idx] + local_dt)
        elif n >= 2:
            global_dt = float(np.median(np.diff(t)))
            duration = global_dt
        else:
            duration = 0.0

        if duration >= min_duration_s:
            # Measure drift from start to end (last in-block sample).
            pos_start = positions[start_idx]
            pos_end = positions[end_idx - 1]
            drift = np.linalg.norm(pos_end - pos_start)

            return {
                "drift_m": float(drift),
                "duration_s": duration,
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
