"""Quality assurance plotting utilities for filter diagnostics.

This module provides functions to visualize filter performance metrics:
- Residual time series with confidence bands
- Position and velocity error plots
- NEES/NIS histograms with chi-squared bounds
- Covariance ellipses for uncertainty visualization

All plots follow Tufte/Gelman principles (minimal chartjunk, maximum data-ink ratio).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from numpy.typing import NDArray

from trodestrack.qa.metrics import chi2_bounds
from trodestrack.viz.styles import COLORS, apply_tufte_style


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
    if t.shape[0] != residuals.shape[0]:
        raise ValueError(f"Shape mismatch: time {t.shape} vs residuals {residuals.shape}")

    apply_tufte_style()

    N, D = residuals.shape

    # Default dimension labels for 2D position residuals
    if dim_labels is None:
        if D == 2:
            dim_labels = ["X (m)", "Y (m)"]
        elif D == 4:
            dim_labels = ["LED1 X (m)", "LED1 Y (m)", "LED2 X (m)", "LED2 Y (m)"]
        else:
            dim_labels = [f"Dim {i + 1}" for i in range(D)]

    # Create subplots: one per dimension, stacked vertically
    fig, axes = plt.subplots(D, 1, figsize=(8, 2 * D), sharex=True, constrained_layout=True)

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
    mask: NDArray[np.bool_] | None = None,
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
    mask : NDArray[np.bool_] | None, optional
        Optional validity mask (N,). Only valid (True) entries plotted.
    prd_threshold_m : float | None, optional
        If provided, plot PRD requirement threshold (e.g., 0.02 for 2 cm).

    Returns:
        Tuple of (figure, axes)

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
    if positions_true.shape != positions_est.shape:
        raise ValueError(
            f"Shape mismatch: true {positions_true.shape} vs est {positions_est.shape}"
        )

    apply_tufte_style()

    # Compute Euclidean error
    errors = positions_true - positions_est
    euclidean_error = np.linalg.norm(errors, axis=1)

    # Apply mask if provided
    if mask is not None:
        t_plot = t[mask]
        error_plot = euclidean_error[mask]
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
    mask: NDArray[np.bool_] | None = None,
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
    mask : NDArray[np.bool_] | None, optional
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
    if velocities_true.shape != velocities_est.shape:
        raise ValueError(
            f"Shape mismatch: true {velocities_true.shape} vs est {velocities_est.shape}"
        )

    apply_tufte_style()

    # Compute Euclidean error
    errors = velocities_true - velocities_est
    euclidean_error = np.linalg.norm(errors, axis=1)

    # Apply mask if provided
    if mask is not None:
        t_plot = t[mask]
        error_plot = euclidean_error[mask]
    else:
        t_plot = t
        error_plot = euclidean_error

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    # Plot error
    ax.plot(t_plot, error_plot, color=COLORS["purple"], linewidth=1.0, label="Velocity Error")

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity Error (m/s)")
    ax.legend(loc="upper right")

    return fig, ax


def plot_heading_error(
    t: NDArray[np.float64],
    headings_true: NDArray[np.float64],
    headings_est: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
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
    mask : NDArray[np.bool_] | None, optional
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
        >>> heading_true = np.linspace(0, 2*np.pi, 100)
        >>> heading_est = heading_true + np.random.randn(100) * 0.1
        >>> fig, ax = plot_heading_error(t, heading_true, heading_est, prd_threshold_deg=7.0)
        >>> plt.close(fig)

    Notes:
        PRD requirement: heading error ≤ 7.0 degrees
        Angle wrapping ensures errors are in [-π, π] range.
    """
    if headings_true.shape != headings_est.shape:
        raise ValueError(f"Shape mismatch: true {headings_true.shape} vs est {headings_est.shape}")

    apply_tufte_style()

    # Compute wrapped heading error (in [-π, π])
    errors = headings_true - headings_est
    errors_wrapped = np.arctan2(np.sin(errors), np.cos(errors))
    errors_deg = np.rad2deg(np.abs(errors_wrapped))

    # Apply mask if provided
    if mask is not None:
        t_plot = t[mask]
        error_plot = errors_deg[mask]
    else:
        t_plot = t
        error_plot = errors_deg

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    # Plot error
    ax.plot(t_plot, error_plot, color=COLORS["orange"], linewidth=1.0, label="Heading Error")

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

    Raises:
        ValueError: If mean or cov are not 2D, or if cov is singular
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
        raise ValueError(f"Singular or negative covariance matrix (eigenvalues: {eigenvalues})")

    # Ellipse angle (orientation of major axis)
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = np.rad2deg(angle_rad)

    # Plot ellipses at each sigma level
    color_rgba = plt.matplotlib.colors.to_rgba(
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
    ax.plot(mean[0], mean[1], "o", color=COLORS["red"], markersize=6, label="Mean", zorder=10)

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
