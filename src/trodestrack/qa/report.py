"""
Comprehensive QA report generation for tracking analysis.

This module provides end-to-end report generation capabilities, combining
metrics computation, visualization, and structured output for complete
quality assurance analysis of tracking sessions.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .metrics import (
    compute_rmse, compute_nees, compute_position_nees,
    compute_occlusion_drift, evaluate_prd_compliance
)
from .plots import (
    plot_trajectory_comparison, plot_velocity_and_heading,
    plot_nees_analysis, plot_bias_traces, plot_measurement_residuals
)
from .logging import QALogger


class QAReportGenerator:
    """
    Comprehensive QA report generator for tracking analysis.

    Combines metrics computation, visualization, and structured logging
    to provide complete quality assurance analysis and reporting.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        session_name: Optional[str] = None,
    ):
        """
        Initialize QA report generator.

        Args:
            output_dir: Directory for output files and reports
            session_name: Optional session name for organizing outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize QA logger
        self.logger = QALogger(output_dir, session_name or "qa_report")

        # Storage for analysis results
        self.metrics = {}
        self.plots = {}
        self.data_summary = {}

    def analyze_tracking_session(
        self,
        estimated_states: jnp.ndarray,
        ground_truth_states: jnp.ndarray,
        covariances: jnp.ndarray,
        timestamps: Optional[jnp.ndarray] = None,
        occlusion_mask: Optional[jnp.ndarray] = None,
        residuals: Optional[Dict[str, jnp.ndarray]] = None,
        measurement_validity: Optional[Dict[str, jnp.ndarray]] = None,
        arena_bounds: Optional[Tuple[float, float, float, float]] = None,
        filter_parameters: Optional[Dict] = None,
    ) -> Dict:
        """
        Perform comprehensive QA analysis of a tracking session.

        Args:
            estimated_states: Shape (N, 8) estimated states
            ground_truth_states: Shape (N, 8) ground truth states
            covariances: Shape (N, 8, 8) covariance matrices
            timestamps: Optional timestamps
            occlusion_mask: Optional occlusion mask
            residuals: Optional measurement residuals
            measurement_validity: Optional measurement validity masks
            arena_bounds: Optional arena bounds for plotting
            filter_parameters: Optional filter configuration for logging

        Returns:
            Dictionary with comprehensive analysis results
        """
        self.logger.logger.info("Starting comprehensive QA analysis")

        # Log input data hashes for reproducibility
        self.logger.log_data_hash("estimated_states", estimated_states)
        self.logger.log_data_hash("ground_truth_states", ground_truth_states)
        self.logger.log_data_hash("covariances", covariances)

        if filter_parameters:
            self.logger.log_parameters(filter_parameters)

        # Store data summary
        self.data_summary = {
            "num_timesteps": len(estimated_states),
            "state_dimension": estimated_states.shape[1],
            "duration_s": float(timestamps[-1] - timestamps[0]) if timestamps is not None else None,
            "occlusion_fraction": float(jnp.mean(occlusion_mask)) if occlusion_mask is not None else 0.0,
        }

        # 1. Compute RMSE metrics
        self.logger.logger.info("Computing RMSE metrics")
        rmse_metrics = compute_rmse(estimated_states, ground_truth_states)
        self.metrics.update(rmse_metrics)

        # 2. Compute NEES consistency metrics
        self.logger.logger.info("Computing NEES consistency metrics")
        nees_metrics = compute_nees(estimated_states, ground_truth_states, covariances)
        self.metrics.update(nees_metrics)

        position_nees_metrics = compute_position_nees(
            estimated_states, ground_truth_states, covariances
        )
        self.metrics.update(position_nees_metrics)

        # 3. Occlusion drift analysis
        if occlusion_mask is not None:
            self.logger.logger.info("Computing occlusion drift metrics")
            dt = 1.0/30.0  # Default to 30 Hz
            if timestamps is not None and len(timestamps) > 1:
                dt = float((timestamps[1] - timestamps[0]))

            drift_metrics = compute_occlusion_drift(
                estimated_states, ground_truth_states, occlusion_mask, dt
            )
            self.metrics.update(drift_metrics)

        # 4. PRD compliance evaluation
        self.logger.logger.info("Evaluating PRD compliance")
        compliance = evaluate_prd_compliance(self.metrics)
        self.metrics.update(compliance)

        # Log all computed metrics
        self.logger.log_metrics(self.metrics)

        # 5. Generate visualization plots
        self.logger.logger.info("Generating visualization plots")
        self._generate_plots(
            estimated_states, ground_truth_states, covariances,
            timestamps, occlusion_mask, residuals, measurement_validity,
            arena_bounds
        )

        # 6. Save data artifacts
        self.logger.logger.info("Saving data artifacts")
        self._save_data_artifacts(
            estimated_states, covariances, timestamps, residuals
        )

        # 7. Generate summary report
        self.logger.logger.info("Generating summary report")
        summary_report = self._generate_summary_report()

        analysis_results = {
            "metrics": self.metrics,
            "data_summary": self.data_summary,
            "plots": self.plots,
            "summary_report": summary_report,
            "output_dir": str(self.output_dir),
        }

        self.logger.logger.info("QA analysis completed successfully")
        return analysis_results

    def _generate_plots(
        self,
        estimated_states: jnp.ndarray,
        ground_truth_states: jnp.ndarray,
        covariances: jnp.ndarray,
        timestamps: Optional[jnp.ndarray],
        occlusion_mask: Optional[jnp.ndarray],
        residuals: Optional[Dict[str, jnp.ndarray]],
        measurement_validity: Optional[Dict[str, jnp.ndarray]],
        arena_bounds: Optional[Tuple[float, float, float, float]],
    ) -> None:
        """Generate all QA plots and save to output directory."""

        # 1. Trajectory comparison
        trajectory_path = self.output_dir / f"{self.logger.session_name}_trajectory.png"
        fig_traj = plot_trajectory_comparison(
            estimated_states, ground_truth_states, timestamps,
            occlusion_mask, arena_bounds,
            title=f"Trajectory Analysis - {self.logger.session_name}",
            save_path=trajectory_path
        )
        plt.close(fig_traj)
        self.plots["trajectory"] = str(trajectory_path)
        self.logger.save_artifact("trajectory_plot", trajectory_path,
                                "2D trajectory comparison with error analysis")

        # 2. Velocity and heading analysis
        velocity_path = self.output_dir / f"{self.logger.session_name}_velocity_heading.png"
        fig_vel = plot_velocity_and_heading(
            estimated_states, ground_truth_states, timestamps, occlusion_mask,
            title=f"Velocity & Heading Analysis - {self.logger.session_name}",
            save_path=velocity_path
        )
        plt.close(fig_vel)
        self.plots["velocity_heading"] = str(velocity_path)
        self.logger.save_artifact("velocity_heading_plot", velocity_path,
                                "Velocity and heading error analysis")

        # 3. NEES consistency analysis
        nees_path = self.output_dir / f"{self.logger.session_name}_nees.png"
        fig_nees = plot_nees_analysis(
            estimated_states, ground_truth_states, covariances, timestamps,
            title=f"NEES Consistency Analysis - {self.logger.session_name}",
            save_path=nees_path
        )
        plt.close(fig_nees)
        self.plots["nees"] = str(nees_path)
        self.logger.save_artifact("nees_plot", nees_path,
                                "NEES filter consistency analysis")

        # 4. Bias traces
        bias_path = self.output_dir / f"{self.logger.session_name}_bias_traces.png"
        fig_bias = plot_bias_traces(
            estimated_states, ground_truth_states, timestamps,
            title=f"IMU Bias Estimates - {self.logger.session_name}",
            save_path=bias_path
        )
        plt.close(fig_bias)
        self.plots["bias_traces"] = str(bias_path)
        self.logger.save_artifact("bias_traces_plot", bias_path,
                                "IMU bias estimation over time")

        # 5. Measurement residuals (if available)
        if residuals:
            residuals_path = self.output_dir / f"{self.logger.session_name}_residuals.png"
            fig_res = plot_measurement_residuals(
                residuals, timestamps, measurement_validity,
                title=f"Measurement Residuals - {self.logger.session_name}",
                save_path=residuals_path
            )
            plt.close(fig_res)
            self.plots["residuals"] = str(residuals_path)
            self.logger.save_artifact("residuals_plot", residuals_path,
                                    "Measurement residual analysis")

    def _save_data_artifacts(
        self,
        estimated_states: jnp.ndarray,
        covariances: jnp.ndarray,
        timestamps: Optional[jnp.ndarray],
        residuals: Optional[Dict[str, jnp.ndarray]],
    ) -> None:
        """Save data artifacts in structured formats."""

        try:
            # Save states to parquet
            states_path = self.logger.save_states_parquet(
                estimated_states, covariances, timestamps
            )

            # Save residuals to parquet (if available)
            if residuals:
                residuals_path = self.logger.save_residuals_parquet(
                    residuals, timestamps
                )

        except ImportError:
            self.logger.logger.warning("pandas not available - skipping parquet export")

        # Always save metrics as JSON
        metrics_path = self.output_dir / f"{self.logger.session_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert JAX arrays to lists for JSON serialization
            json_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, (jnp.ndarray, np.ndarray)):
                    json_metrics[key] = value.tolist()
                else:
                    json_metrics[key] = value

            json.dump(json_metrics, f, indent=2, default=str)

        self.logger.save_artifact("metrics_json", metrics_path,
                                "Computed QA metrics in JSON format")

    def _generate_summary_report(self) -> str:
        """Generate comprehensive text summary report."""

        # Get logger summary
        base_summary = self.logger.create_summary_report()

        # Add detailed analysis
        detailed_lines = [
            "",
            "Detailed Analysis:",
            "=" * 40,
            "",
        ]

        # Data summary
        detailed_lines.extend([
            "Data Summary:",
            f"  Duration: {self.data_summary.get('duration_s', 'N/A')} seconds",
            f"  Timesteps: {self.data_summary.get('num_timesteps', 'N/A')}",
            f"  State dimension: {self.data_summary.get('state_dimension', 'N/A')}",
            f"  Occlusion fraction: {self.data_summary.get('occlusion_fraction', 0.0):.1%}",
            "",
        ])

        # Performance vs PRD requirements
        detailed_lines.extend([
            "PRD Compliance Assessment:",
            "-" * 30,
        ])

        # Position RMSE
        pos_rmse = self.metrics.get("position_rmse_cm", None)
        if pos_rmse is not None:
            status = "✓ PASS" if pos_rmse <= 2.0 else "✗ FAIL"
            detailed_lines.append(f"  Position RMSE: {pos_rmse:.2f} cm (≤2.0 cm) {status}")

        # Velocity RMSE
        vel_rmse = self.metrics.get("velocity_rmse_cm_s", None)
        if vel_rmse is not None:
            status = "✓ PASS" if vel_rmse <= 10.0 else "✗ FAIL"
            detailed_lines.append(f"  Velocity RMSE: {vel_rmse:.2f} cm/s (≤10.0 cm/s) {status}")

        # Heading RMSE
        head_rmse = self.metrics.get("heading_rmse_deg", None)
        if head_rmse is not None:
            status = "✓ PASS" if head_rmse <= 7.0 else "✗ FAIL"
            detailed_lines.append(f"  Heading RMSE: {head_rmse:.2f}° (≤7.0°) {status}")

        # Occlusion drift
        max_drift = self.metrics.get("max_drift_cm", None)
        if max_drift is not None:
            status = "✓ PASS" if max_drift <= 15.0 else "✗ FAIL"
            detailed_lines.append(f"  Max occlusion drift: {max_drift:.2f} cm (≤15.0 cm) {status}")

        detailed_lines.append("")

        # NEES consistency
        if "nees_consistency_ratio" in self.metrics:
            ratio = self.metrics["nees_consistency_ratio"]
            detailed_lines.extend([
                "Filter Consistency (NEES):",
                "-" * 25,
                f"  Full state NEES ratio: {ratio:.3f} (ideal: 1.0)",
            ])

            if ratio < 0.8:
                detailed_lines.append("    → Filter overconfident (uncertainty too small)")
            elif ratio > 1.2:
                detailed_lines.append("    → Filter underconfident (uncertainty too large)")
            else:
                detailed_lines.append("    → Filter well-calibrated ✓")

        if "position_nees_consistency_ratio" in self.metrics:
            pos_ratio = self.metrics["position_nees_consistency_ratio"]
            detailed_lines.append(f"  Position NEES ratio: {pos_ratio:.3f} (ideal: 1.0)")

        detailed_lines.append("")

        # Occlusion analysis
        if "num_occlusions" in self.metrics:
            num_occ = self.metrics["num_occlusions"]
            mean_drift = self.metrics.get("mean_drift_cm", 0.0)
            max_drift = self.metrics.get("max_drift_cm", 0.0)

            detailed_lines.extend([
                "Occlusion Analysis:",
                "-" * 18,
                f"  Number of occlusions: {num_occ}",
                f"  Mean drift: {mean_drift:.2f} cm",
                f"  Max drift: {max_drift:.2f} cm",
                "",
            ])

        # Generated outputs
        detailed_lines.extend([
            "Generated Outputs:",
            "-" * 18,
        ])

        for plot_name, plot_path in self.plots.items():
            filename = Path(plot_path).name
            detailed_lines.append(f"  {plot_name}: {filename}")

        # Combine reports
        full_report = base_summary + "\n" + "\n".join(detailed_lines)

        # Save to file
        report_path = self.output_dir / f"{self.logger.session_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(full_report)

        self.logger.save_artifact("summary_report", report_path,
                                "Comprehensive text summary report")

        return full_report


def generate_qa_report(
    estimated_states: jnp.ndarray,
    ground_truth_states: jnp.ndarray,
    covariances: jnp.ndarray,
    output_dir: Union[str, Path],
    session_name: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Generate comprehensive QA report for tracking analysis.

    Convenience function that creates a QAReportGenerator and runs
    complete analysis with all default settings.

    Args:
        estimated_states: Shape (N, 8) estimated states
        ground_truth_states: Shape (N, 8) ground truth states
        covariances: Shape (N, 8, 8) covariance matrices
        output_dir: Directory for output files
        session_name: Optional session name
        **kwargs: Additional arguments for analyze_tracking_session

    Returns:
        Dictionary with comprehensive analysis results
    """
    generator = QAReportGenerator(output_dir, session_name)

    return generator.analyze_tracking_session(
        estimated_states, ground_truth_states, covariances, **kwargs
    )