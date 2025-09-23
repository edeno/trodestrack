"""
Quality assurance module for tracking analysis.

This module provides comprehensive QA capabilities including metrics computation,
visualization, logging, and report generation for evaluating tracking performance.
"""

from .metrics import (
    compute_rmse,
    compute_nees,
    compute_position_nees,
    compute_occlusion_drift,
    evaluate_prd_compliance,
)

from .plots import (
    plot_trajectory_comparison,
    plot_velocity_and_heading,
    plot_nees_analysis,
    plot_bias_traces,
    plot_measurement_residuals,
)

from .logging import QALogger, create_qa_session

from .report import QAReportGenerator, generate_qa_report

__all__ = [
    # Metrics
    "compute_rmse",
    "compute_nees",
    "compute_position_nees",
    "compute_occlusion_drift",
    "evaluate_prd_compliance",
    # Plots
    "plot_trajectory_comparison",
    "plot_velocity_and_heading",
    "plot_nees_analysis",
    "plot_bias_traces",
    "plot_measurement_residuals",
    # Logging
    "QALogger",
    "create_qa_session",
    # Report generation
    "QAReportGenerator",
    "generate_qa_report",
]