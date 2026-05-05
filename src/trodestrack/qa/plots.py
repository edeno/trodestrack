"""Quality assurance plotting utilities for filter diagnostics.

This module provides functions to visualize filter performance metrics:
- Residual time series with confidence bands
- Position and velocity error plots
- NEES/NIS histograms with chi-squared bounds
- Covariance ellipses for uncertainty visualization

All plots follow Tufte/Gelman principles (minimal chartjunk, maximum data-ink ratio).
"""

from __future__ import annotations

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from numpy.typing import NDArray

from trodestrack.models.state_layout import StateLayout, get_heading_index
from trodestrack.qa.metrics import chi2_bounds
from trodestrack.viz.styles import COLORS, apply_tufte_style


def _validate_time_axis(
    t: NDArray[np.floating], name: str = "t"
) -> NDArray[np.float64]:
    """Reject non-1D time vectors used as plotting x-axes.

    Matplotlib raises an opaque "x and y must have same first dimension"
    when plot is fed a (N, 1) ``t``; this guard names the contract.
    """

    arr = np.asarray(t)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D (N,) in seconds; got shape {arr.shape}.")
    return arr


def _validate_optional_bool_mask(
    mask: NDArray[np.bool_] | None,
    expected_len: int,
    name: str = "valid_mask",
) -> NDArray[np.bool_] | None:
    """Reject malformed valid_mask arrays before they index plotted arrays."""

    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.shape != (expected_len,):
        raise ValueError(f"{name} must have shape ({expected_len},); got {arr.shape}.")
    if arr.dtype != np.bool_ and not (
        np.issubdtype(arr.dtype, np.integer) and np.all(np.isin(arr, (0, 1)))
    ):
        raise ValueError(
            f"{name} must be boolean or 0/1 integer; got dtype {arr.dtype!r}."
        )
    return arr.astype(bool)


def plot_residuals(
    t: NDArray[np.float64],
    residuals: NDArray[np.float64],
    ylabel: str = "Residuals",
    confidence_std: float | None = None,
    dim_labels: list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot residual time series with optional confidence bands.

    Residuals are measurement innovations (observed − predicted). For a well-tuned
    filter, residuals should be zero-mean white noise within confidence bounds.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time vector (N,) in seconds.
    residuals : NDArray[np.float64]
        Residuals over time (N, D) where D is dimensionality.
    ylabel : str, default "Residuals"
        Y-axis label.
    confidence_std : float | None, optional
        If provided, plot ±confidence_std bands (e.g., 0.01 for ±1 cm).
    dim_labels : list[str] | None, optional
        Custom labels for each dimension (default auto-labels).

    Returns
    -------
    tuple[Figure, list[Axes]]
        Matplotlib Figure and list of Axes (length D).

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 100)
        >>> residuals = np.random.randn(100, 2) * 0.01  # 1cm noise
        >>> fig, axes = plot_residuals(t, residuals, confidence_std=0.01)
        >>> # Save or display
        >>> fig.savefig("residuals.png", dpi=150, bbox_inches="tight")
        >>> plt.close(fig)

    Notes:
        - Zero-mean residuals indicate unbiased filter
        - Residuals within ±2σ bands ~95% of time indicates correct R tuning
        - Correlated residuals (visible patterns) indicate Q too small or timing issues
    """
    t = _validate_time_axis(t, name="t")
    residuals_arr = np.asarray(residuals)
    if residuals_arr.ndim != 2:
        raise ValueError(
            f"residuals must be 2D (N, D); got shape {residuals_arr.shape}."
        )
    if t.shape[0] != residuals_arr.shape[0]:
        raise ValueError(
            f"Shape mismatch: time {t.shape} vs residuals {residuals_arr.shape}"
        )
    residuals = residuals_arr

    apply_tufte_style()

    _N, D = residuals.shape

    # Default dimension labels for 2D position residuals
    if dim_labels is None:
        if D == 2:
            dim_labels = ["X (m)", "Y (m)"]
        elif D == 4:
            dim_labels = ["LED1 X (m)", "LED1 Y (m)", "LED2 X (m)", "LED2 Y (m)"]
        else:
            dim_labels = [f"Dim {i + 1}" for i in range(D)]
    elif len(dim_labels) != D:
        raise ValueError(
            f"dim_labels must have length D={D} (one per residual column); "
            f"got len={len(dim_labels)}."
        )

    # Create subplots: one per dimension, stacked vertically
    fig, axes = plt.subplots(
        D, 1, figsize=(8, 2 * D), sharex=True, constrained_layout=True
    )

    # Handle single dimension case (axes is not a list)
    if D == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Plot residuals as line
        ax.plot(t, residuals[:, i], color=COLORS["blue"], linewidth=0.8, alpha=0.7)

        # Plot zero line (reference)
        ax.axhline(0, color=COLORS["gray"], linewidth=0.5, linestyle="--", alpha=0.5)

        # Plot confidence bands if requested
        if confidence_std is not None:
            ax.axhspan(
                -confidence_std,
                confidence_std,
                color=COLORS["light_gray"],
                alpha=0.3,
                label=f"±{confidence_std:.3f} m",
            )

        # Labels
        ax.set_ylabel(dim_labels[i])
        if i == D - 1:  # Only bottom subplot gets x-label
            ax.set_xlabel("Time (s)")

        # Legend only if confidence bands are shown
        if confidence_std is not None and i == 0:
            ax.legend(loc="upper right", fontsize=8)

    return fig, axes


def plot_position_error(
    t: NDArray[np.float64],
    positions_true: NDArray[np.float64],
    positions_est: NDArray[np.float64],
    valid_mask: NDArray[np.bool_] | None = None,
    prd_threshold_m: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot Euclidean position error over time.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time vector (N,) in seconds.
    positions_true : NDArray[np.float64]
        Ground truth positions (N, 2) in meters.
    positions_est : NDArray[np.float64]
        Estimated positions (N, 2) in meters.
    valid_mask : NDArray[np.bool_] | None, optional
        Optional validity mask (N,). Only valid (True) entries plotted.
    prd_threshold_m : float | None, optional
        If provided, plot PRD requirement threshold (e.g., 0.02 for 2 cm).

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes.

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 100)
        >>> pos_true = np.column_stack([t * 0.1, np.zeros(100)])
        >>> pos_est = pos_true + np.random.randn(100, 2) * 0.01
        >>> fig, ax = plot_position_error(t, pos_true, pos_est, prd_threshold_m=0.02)
        >>> plt.close(fig)

    Notes:
        PRD requirement: position error ≤ 0.02 m (2 cm)
    """
    t = _validate_time_axis(t, name="t")
    pt = np.asarray(positions_true)
    pe = np.asarray(positions_est)
    if pt.shape != pe.shape:
        raise ValueError(f"Shape mismatch: true {pt.shape} vs est {pe.shape}")
    if pt.ndim != 2 or pt.shape[1] != 2:
        raise ValueError(
            f"positions_true / positions_est must have shape (N, 2); got {pt.shape}."
        )
    if pt.shape[0] != t.shape[0]:
        raise ValueError(f"Shape mismatch: positions {pt.shape} vs time {t.shape}.")
    valid_mask = _validate_optional_bool_mask(valid_mask, t.shape[0])

    apply_tufte_style()

    # Compute Euclidean error
    errors = pt - pe
    euclidean_error = np.linalg.norm(errors, axis=1)

    # Apply validity mask if provided
    if valid_mask is not None:
        t_plot = t[valid_mask]
        error_plot = euclidean_error[valid_mask]
    else:
        t_plot = t
        error_plot = euclidean_error

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    # Plot error
    ax.plot(t_plot, error_plot, color=COLORS["blue"], linewidth=1.0, label="Error")

    # Plot PRD threshold if provided
    if prd_threshold_m is not None:
        ax.axhline(
            prd_threshold_m,
            color=COLORS["red"],
            linewidth=1.0,
            linestyle="--",
            label=f"PRD threshold ({prd_threshold_m * 100:.0f} cm)",
        )

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.legend(loc="upper right")

    return fig, ax


def plot_velocity_error(
    t: NDArray[np.float64],
    velocities_true: NDArray[np.float64],
    velocities_est: NDArray[np.float64],
    valid_mask: NDArray[np.bool_] | None = None,
) -> tuple[Figure, Axes]:
    """Plot Euclidean velocity error over time.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time vector (N,) in seconds.
    velocities_true : NDArray[np.float64]
        Ground truth velocities (N, 2) in m/s.
    velocities_est : NDArray[np.float64]
        Estimated velocities (N, 2) in m/s.
    valid_mask : NDArray[np.bool_] | None, optional
        Optional validity mask (N,). Only valid (True) entries plotted.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes.

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 100)
        >>> vel_true = np.column_stack([np.ones(100) * 0.1, np.zeros(100)])
        >>> vel_est = vel_true + np.random.randn(100, 2) * 0.01
        >>> fig, ax = plot_velocity_error(t, vel_true, vel_est)
        >>> plt.close(fig)

    Notes:
        PRD requirement: velocity error ≤ 0.10 m/s (10 cm/s)
    """
    t = _validate_time_axis(t, name="t")
    vt = np.asarray(velocities_true)
    ve = np.asarray(velocities_est)
    if vt.shape != ve.shape:
        raise ValueError(f"Shape mismatch: true {vt.shape} vs est {ve.shape}")
    if vt.ndim != 2 or vt.shape[1] != 2:
        raise ValueError(
            f"velocities_true / velocities_est must have shape (N, 2); got {vt.shape}."
        )
    if vt.shape[0] != t.shape[0]:
        raise ValueError(f"Shape mismatch: velocities {vt.shape} vs time {t.shape}.")
    valid_mask = _validate_optional_bool_mask(valid_mask, t.shape[0])

    apply_tufte_style()

    # Compute Euclidean error
    errors = vt - ve
    euclidean_error = np.linalg.norm(errors, axis=1)

    # Apply validity mask if provided
    if valid_mask is not None:
        t_plot = t[valid_mask]
        error_plot = euclidean_error[valid_mask]
    else:
        t_plot = t
        error_plot = euclidean_error

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    # Plot error
    ax.plot(
        t_plot,
        error_plot,
        color=COLORS["purple"],
        linewidth=1.0,
        label="Velocity Error",
    )

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity Error (m/s)")
    ax.legend(loc="upper right")

    return fig, ax


def plot_heading_error(
    t: NDArray[np.float64],
    headings_true: NDArray[np.float64],
    headings_est: NDArray[np.float64],
    valid_mask: NDArray[np.bool_] | None = None,
    prd_threshold_deg: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot heading error over time with proper angle wrapping.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time vector (N,) in seconds.
    headings_true : NDArray[np.float64]
        Ground truth headings (N,) in radians.
    headings_est : NDArray[np.float64]
        Estimated headings (N,) in radians.
    valid_mask : NDArray[np.bool_] | None, optional
        Optional validity mask (N,). Only valid (True) entries plotted.
    prd_threshold_deg : float | None, optional
        If provided, plot PRD requirement threshold (degrees), e.g., 7.0.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes.

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 100)
        >>> heading_true = np.linspace(0, 2 * np.pi, 100)
        >>> heading_est = heading_true + np.random.randn(100) * 0.1
        >>> fig, ax = plot_heading_error(
        ...     t, heading_true, heading_est, prd_threshold_deg=7.0
        ... )
        >>> plt.close(fig)

    Notes:
        PRD requirement: heading error ≤ 7.0 degrees
        Angle wrapping ensures errors are in [-π, π] range.
    """
    if headings_true.shape != headings_est.shape:
        raise ValueError(
            f"Shape mismatch: true {headings_true.shape} vs est {headings_est.shape}"
        )

    apply_tufte_style()

    # Compute wrapped heading error (in [-π, π])
    errors = headings_true - headings_est
    errors_wrapped = np.arctan2(np.sin(errors), np.cos(errors))
    errors_deg = np.rad2deg(np.abs(errors_wrapped))

    # Apply validity mask if provided
    if valid_mask is not None:
        t_plot = t[valid_mask]
        error_plot = errors_deg[valid_mask]
    else:
        t_plot = t
        error_plot = errors_deg

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    # Plot error
    ax.plot(
        t_plot, error_plot, color=COLORS["orange"], linewidth=1.0, label="Heading Error"
    )

    # Plot PRD threshold if provided
    if prd_threshold_deg is not None:
        ax.axhline(
            prd_threshold_deg,
            color=COLORS["red"],
            linewidth=1.0,
            linestyle="--",
            label=f"PRD threshold ({prd_threshold_deg:.0f}°)",
        )

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading Error (degrees)")
    ax.legend(loc="upper right")

    return fig, ax


def plot_nees_histogram(
    nees: NDArray[np.float64],
    state_dim: int,
    confidence: float = 0.95,
) -> tuple[Figure, Axes]:
    """Plot NEES histogram with chi-squared confidence bounds.

    NEES (Normalized Estimation Error Squared) should follow χ²(state_dim) for
    a consistent filter. Bounds show expected range for the given confidence.

    Parameters
    ----------
    nees : NDArray[np.float64]
        NEES values (N,).
    state_dim : int
        State dimensionality (degrees of freedom for χ²).
    confidence : float, default 0.95
        Confidence level for χ² bounds.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> nees = np.random.chisquare(df=8, size=500)
        >>> fig, ax = plot_nees_histogram(nees, state_dim=8, confidence=0.95)
        >>> plt.close(fig)

    Notes:
        - Mean NEES should be approximately equal to state_dim
        - ~95% of NEES values should fall within χ²(state_dim, 0.95) bounds
        - NEES consistently above upper bound → filter overconfident (P too small)
        - NEES consistently below lower bound → filter underconfident (P too large)
    """
    if nees.ndim != 1:
        raise ValueError(f"Expected 1D NEES array, got shape {nees.shape}")

    apply_tufte_style()

    # Compute chi-squared bounds
    lower, upper = chi2_bounds(df=state_dim, confidence=confidence)

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    # Plot histogram
    ax.hist(
        nees,
        bins=30,
        color=COLORS["blue"],
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5,
        label=f"NEES (n={len(nees)})",
    )

    # Plot chi-squared bounds
    ax.axvline(
        lower,
        color=COLORS["red"],
        linewidth=1.5,
        linestyle="--",
        label=f"χ²({state_dim}, {confidence:.0%}) bounds",
    )
    ax.axvline(upper, color=COLORS["red"], linewidth=1.5, linestyle="--")

    # Plot mean
    mean_nees = float(np.mean(nees))
    ax.axvline(
        mean_nees,
        color=COLORS["gray"],
        linewidth=1.0,
        linestyle="-",
        label=f"Mean = {mean_nees:.2f}",
    )

    # Labels
    ax.set_xlabel("NEES")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    return fig, ax


def plot_nis_histogram(
    nis: NDArray[np.float64],
    measurement_dim: int,
    confidence: float = 0.95,
) -> tuple[Figure, Axes]:
    """Plot NIS histogram with chi-squared confidence bounds.

    NIS (Normalized Innovation Squared) should follow χ²(measurement_dim) for
    a consistent filter. Bounds show expected range for the given confidence.

    Parameters
    ----------
    nis : NDArray[np.float64]
        NIS values (N,).
    measurement_dim : int
        Measurement dimensionality (degrees of freedom for χ²).
    confidence : float, default 0.95
        Confidence level for χ² bounds.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> nis = np.random.chisquare(df=4, size=500)
        >>> fig, ax = plot_nis_histogram(nis, measurement_dim=4, confidence=0.95)
        >>> plt.close(fig)

    Notes:
        - Mean NIS should be approximately equal to measurement_dim
        - ~95% of NIS values should fall within χ²(measurement_dim, 0.95) bounds
        - NIS consistently above upper bound → measurement noise R underestimated
        - NIS consistently below lower bound → measurement noise R overestimated
    """
    if nis.ndim != 1:
        raise ValueError(f"Expected 1D NIS array, got shape {nis.shape}")

    apply_tufte_style()

    # Compute chi-squared bounds
    lower, upper = chi2_bounds(df=measurement_dim, confidence=confidence)

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    # Plot histogram
    ax.hist(
        nis,
        bins=30,
        color=COLORS["orange"],
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5,
        label=f"NIS (n={len(nis)})",
    )

    # Plot chi-squared bounds
    ax.axvline(
        lower,
        color=COLORS["red"],
        linewidth=1.5,
        linestyle="--",
        label=f"χ²({measurement_dim}, {confidence:.0%}) bounds",
    )
    ax.axvline(upper, color=COLORS["red"], linewidth=1.5, linestyle="--")

    # Plot mean
    mean_nis = float(np.mean(nis))
    ax.axvline(
        mean_nis,
        color=COLORS["gray"],
        linewidth=1.0,
        linestyle="-",
        label=f"Mean = {mean_nis:.2f}",
    )

    # Labels
    ax.set_xlabel("NIS")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    return fig, ax


def plot_covariance_ellipse(
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
    n_std: list[float] | None = None,
    trajectory: NDArray[np.float64] | None = None,
    color: str = "blue",
    alpha: float = 0.3,
) -> tuple[Figure, Axes]:
    """Plot 2D covariance ellipse with optional trajectory overlay.

    Visualizes position uncertainty as ellipses at 1σ, 2σ, 3σ levels (configurable).

    Parameters
    ----------
    mean : NDArray[np.float64]
        Mean position (2,) in meters.
    cov : NDArray[np.float64]
        Covariance matrix (2, 2) in m².
    n_std : list[float] | None, optional
        Sigma levels to plot; default [1, 2, 3].
    trajectory : NDArray[np.float64] | None, optional
        Optional trajectory to overlay (N, 2) in meters.
    color : str, default "blue"
        Ellipse color (matplotlib color name or hex).
    alpha : float, default 0.3
        Ellipse face transparency.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes.

    Example:
        >>> import numpy as np
        >>> mean = np.array([0.5, 0.5])
        >>> cov = np.array([[0.01, 0.005], [0.005, 0.02]])  # Correlated
        >>> fig, ax = plot_covariance_ellipse(mean, cov, n_std=[1, 2])
        >>> plt.close(fig)

    Notes:
        - Ellipse orientation shows correlation structure
        - 1σ ellipse contains ~39% of probability mass (2D Gaussian)
        - 2σ ellipse contains ~86% of probability mass
        - 3σ ellipse contains ~99% of probability mass

    Raises
    ------
    ValueError
        If mean or cov are not 2D, or if cov is singular.
    """
    if mean.shape != (2,):
        raise ValueError(f"Expected 2D mean, got shape {mean.shape}")

    if cov.shape != (2, 2):
        raise ValueError(f"Expected 2×2 covariance, got shape {cov.shape}")

    if n_std is None:
        n_std = [1, 2, 3]

    apply_tufte_style()

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    # Plot trajectory if provided
    if trajectory is not None:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=COLORS["gray"],
            linewidth=0.8,
            alpha=0.5,
            label="Trajectory",
        )

    # Compute eigenvalues and eigenvectors for ellipse orientation
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Check for singular covariance
    if np.any(eigenvalues <= 0):
        raise ValueError(
            f"Singular or negative covariance matrix (eigenvalues: {eigenvalues})"
        )

    # Ellipse angle (orientation of major axis)
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = np.rad2deg(angle_rad)

    # Plot ellipses at each sigma level
    color_rgba = matplotlib.colors.to_rgba(
        color if color in COLORS else COLORS.get(color, color)
    )

    for i, n in enumerate(n_std):
        # Ellipse width and height (2 * n * sqrt(eigenvalue))
        width = 2 * n * np.sqrt(eigenvalues[0])
        height = 2 * n * np.sqrt(eigenvalues[1])

        # Create ellipse patch
        ellipse = Ellipse(
            xy=tuple(mean),
            width=width,
            height=height,
            angle=angle_deg,
            facecolor=color_rgba,
            edgecolor=color_rgba[:3] + (1.0,),  # Solid edge
            alpha=alpha / (i + 1),  # Fade outer ellipses
            linewidth=1.0,
            label=f"{n}σ" if i == 0 else None,
        )
        ax.add_patch(ellipse)

    # Plot mean
    ax.plot(
        mean[0],
        mean[1],
        "o",
        color=COLORS["red"],
        markersize=6,
        label="Mean",
        zorder=10,
    )

    # Labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")

    # Auto-scale to show ellipses
    # Extend limits to 4σ to ensure 3σ ellipse is visible
    margin = 4 * np.sqrt(max(eigenvalues))
    ax.set_xlim(mean[0] - margin, mean[0] + margin)
    ax.set_ylim(mean[1] - margin, mean[1] + margin)

    return fig, ax


def plot_heading_consistency_from_leds(
    t: NDArray[np.float64],
    filtered_means: NDArray[np.float64],
    Z_cam_led1: NDArray[np.float64],
    Z_cam_led2: NDArray[np.float64],
    mask_cam: NDArray[np.bool_],
    *,
    layout: StateLayout,
    title: str | None = None,
) -> tuple[Figure, list[Axes], dict[str, float]]:
    """Compare filter heading vs. camera LED-derived heading over time.

    Computes camera heading as atan2(y2 - y1, x2 - x1) when both LEDs are valid,
    and compares it to the filter's heading state. Plots time series and the
    wrapped angular error, and returns basic summary statistics.

    Parameters
    ----------
    t : ndarray (N,)
        Time vector in seconds (camera timeline).
    filtered_means : ndarray (N, n)
        Filtered state means at camera times.
    Z_cam_led1 : ndarray (N, 2)
        LED1 camera observations in meters.
    Z_cam_led2 : ndarray (N, 2)
        LED2 camera observations in meters.
    mask_cam : ndarray (N,)
        Camera validity mask.
    layout : StateLayout
        State layout (used to locate heading index).
    title : str | None
        Optional figure title.

    Returns
    -------
    (fig, axes, stats)
        Figure, list of Axes [heading_trace_ax, error_ax], and a stats dict with:
        - 'count_valid': number of valid frames
        - 'mean_abs_err_deg': mean absolute angular error (deg)
        - 'median_abs_err_deg': median absolute angular error (deg)
        - 'circular_std_deg': circular standard deviation (deg)

    Notes
    -----
    Use this to verify pixel→meter mapping orientation. If camera Y is not
    oriented upward in world coordinates (e.g., image coordinates with Y down),
    the LED-derived heading may be flipped or rotated relative to the filter
    heading. Consistent tracking requires that both share the same world axes.
    """
    apply_tufte_style()

    # Convert to numpy arrays
    t = np.asarray(t)
    m = np.asarray(filtered_means)
    led1 = np.asarray(Z_cam_led1)
    led2 = np.asarray(Z_cam_led2)
    cam_mask = np.asarray(mask_cam).astype(bool)

    # Heading indices and vectors
    h_idx = get_heading_index(layout)
    theta_filt = m[:, h_idx]

    # LED-derived heading where both LEDs are finite and camera mask is true
    both_finite = np.isfinite(led1).all(axis=1) & np.isfinite(led2).all(axis=1)
    valid = cam_mask & both_finite

    d = led2 - led1
    theta_cam = np.arctan2(d[:, 1], d[:, 0])

    # Wrapped error (in [-pi, pi]) computed only on valid frames
    def wrap(a: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.arctan2(np.sin(a), np.cos(a))

    err = wrap(theta_cam - theta_filt)
    err_deg = np.rad2deg(np.abs(err[valid]))

    # Summary stats (valid frames only)
    count_valid = int(np.sum(valid))
    mean_abs_err = float(np.mean(err_deg)) if count_valid > 0 else float("nan")
    median_abs_err = float(np.median(err_deg)) if count_valid > 0 else float("nan")
    # Circular std over valid frames
    if count_valid > 0:
        e_valid = err[valid]
        R = np.hypot(np.mean(np.cos(e_valid)), np.mean(np.sin(e_valid)))
        circ_std = np.sqrt(-2.0 * np.log(max(R, 1e-12)))  # radians
        circular_std_deg = float(np.rad2deg(circ_std))
    else:
        circular_std_deg = float("nan")

    stats = {
        "count_valid": count_valid,
        "mean_abs_err_deg": mean_abs_err,
        "median_abs_err_deg": median_abs_err,
        "circular_std_deg": circular_std_deg,
    }

    # Build figure
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)

    # Top: headings over time (valid frames)
    axes[0].plot(
        t[valid],
        wrap(theta_cam[valid]),
        label="Camera LED heading",
        color=COLORS["orange"],
        linewidth=1.2,
    )
    axes[0].plot(
        t[valid],
        wrap(theta_filt[valid]),
        label="Filtered heading",
        color=COLORS["green"],
        linewidth=1.2,
    )
    axes[0].set_ylabel("Heading (rad)")
    axes[0].legend(loc="upper right")
    if title:
        axes[0].set_title(title)

    # Bottom: absolute error in degrees (valid frames)
    axes[1].plot(
        t[valid], err_deg, color=COLORS["red"], linewidth=1.0, label="|Δθ| (deg)"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Abs error (deg)")
    axes[1].legend(loc="upper right")

    # Annotate stats
    txt = (
        f"n={count_valid}  mean={mean_abs_err:.2f}°  median={median_abs_err:.2f}°  "
        f"circ-std={circular_std_deg:.2f}°"
    )
    axes[1].text(
        0.01,
        0.98,
        txt,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=COLORS["gray"],
    )

    return fig, list(axes), stats
