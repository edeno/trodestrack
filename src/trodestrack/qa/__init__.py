"""Quality assurance metrics for tracking performance evaluation."""

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

__all__ = [
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
]
