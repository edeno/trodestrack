"""Quality assurance metrics and plotting for tracking performance evaluation."""

from trodestrack.qa.metrics import (
    chi2_bounds,
    chi2_ci95,
    compute_heading_error,
    compute_nees,
    compute_nees_stats,
    compute_nis,
    compute_nis_stats,
    compute_position_rmse,
    compute_residual_autocorrelation,
    compute_velocity_rmse,
    within_envelope,
)
from trodestrack.qa.plots import (
    plot_covariance_ellipse,
    plot_nees_histogram,
    plot_nis_histogram,
    plot_position_error,
    plot_residuals,
    plot_velocity_error,
)

__all__ = [
    # Metrics
    "compute_position_rmse",
    "compute_velocity_rmse",
    "compute_heading_error",
    "compute_nees",
    "compute_nees_stats",
    "compute_nis",
    "compute_nis_stats",
    "compute_residual_autocorrelation",
    "chi2_bounds",
    "chi2_ci95",
    "within_envelope",
    # Plots
    "plot_residuals",
    "plot_position_error",
    "plot_velocity_error",
    "plot_nees_histogram",
    "plot_nis_histogram",
    "plot_covariance_ellipse",
]
