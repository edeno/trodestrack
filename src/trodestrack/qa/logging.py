"""
Structured logging and data persistence for QA analysis.

This module provides comprehensive logging capabilities for tracking analysis,
including JSON metadata logging, parquet state persistence, and diagnostic
data export for reproducible analysis.
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import jax.numpy as jnp

# Optional dependencies for data persistence
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class QALogger:
    """
    Structured logger for tracking analysis with metadata and artifacts.

    Provides JSON logging, artifact management, and reproducibility tracking
    for comprehensive quality assurance workflows.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        session_name: str,
        log_level: int = logging.INFO,
    ):
        """
        Initialize QA logger with output directory and session metadata.

        Args:
            output_dir: Directory for output files and logs
            session_name: Unique name for this analysis session
            log_level: Python logging level (default: INFO)
        """
        self.output_dir = Path(output_dir)
        self.session_name = session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up structured logging
        self.log_file = self.output_dir / f"{session_name}_qa.log"
        self.json_file = self.output_dir / f"{session_name}_metadata.json"

        # Configure Python logger
        self.logger = logging.getLogger(f"trodestrack.qa.{session_name}")
        self.logger.setLevel(log_level)

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Initialize metadata
        self.metadata = {
            "session_name": session_name,
            "timestamp": datetime.now().isoformat(),
            "trodestrack_version": "0.1.0",  # TODO: Get from package
            "output_dir": str(self.output_dir),
            "artifacts": {},
            "metrics": {},
            "parameters": {},
            "data_hashes": {},
        }

        self.logger.info(f"QA session '{session_name}' initialized in {self.output_dir}")

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log analysis parameters for reproducibility.

        Args:
            parameters: Dictionary of analysis parameters
        """
        self.metadata["parameters"].update(parameters)
        self.logger.info(f"Logged parameters: {list(parameters.keys())}")

    def log_data_hash(self, data_name: str, data: Union[np.ndarray, jnp.ndarray]) -> str:
        """
        Compute and log hash of input data for reproducibility tracking.

        Args:
            data_name: Name/identifier for the data
            data: Input data array

        Returns:
            SHA-256 hash of the data
        """
        # Convert to numpy if needed
        if hasattr(data, "numpy"):
            data_np = np.array(data)
        else:
            data_np = np.array(data)

        # Compute hash
        data_bytes = data_np.tobytes()
        data_hash = hashlib.sha256(data_bytes).hexdigest()

        self.metadata["data_hashes"][data_name] = {
            "hash": data_hash,
            "shape": data_np.shape,
            "dtype": str(data_np.dtype),
        }

        self.logger.info(f"Logged data hash for '{data_name}': {data_hash[:16]}...")
        return data_hash

    def log_metrics(self, metrics: Dict[str, Union[float, int, dict]]) -> None:
        """
        Log computed QA metrics.

        Args:
            metrics: Dictionary of computed metrics
        """
        self.metadata["metrics"].update(metrics)

        # Log key metrics to console
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                self.logger.info(f"Metric {key}: {value:.4f}")
            elif isinstance(value, dict) and "overall_prd_compliant" in value:
                self.logger.info(f"PRD Compliance: {value}")

    def save_states_parquet(
        self,
        states: jnp.ndarray,
        covariances: jnp.ndarray,
        timestamps: Optional[jnp.ndarray] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save estimated states and covariances to parquet format.

        Args:
            states: Shape (N, 8) estimated states
            covariances: Shape (N, 8, 8) covariance matrices
            timestamps: Optional timestamps
            filename: Optional custom filename

        Returns:
            Path to saved parquet file
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for parquet export")

        if filename is None:
            filename = f"{self.session_name}_states.parquet"

        output_path = self.output_dir / filename

        # Convert JAX arrays to numpy
        states_np = np.array(states)
        covariances_np = np.array(covariances)

        # Create DataFrame with state components
        data = {
            "x_cm": states_np[:, 0],
            "y_cm": states_np[:, 1],
            "vx_cm_s": states_np[:, 2],
            "vy_cm_s": states_np[:, 3],
            "theta_rad": states_np[:, 4],
            "bias_gz_rad_s": states_np[:, 5],
            "bias_ax_m_s2": states_np[:, 6],
            "bias_ay_m_s2": states_np[:, 7],
        }

        if timestamps is not None:
            data["timestamp_s"] = np.array(timestamps)

        # Add covariance diagonal (uncertainties)
        for i in range(8):
            data[f"var_{i}"] = covariances_np[:, i, i]

        # Add some key off-diagonal covariances
        data["cov_x_y"] = covariances_np[:, 0, 1]
        data["cov_vx_vy"] = covariances_np[:, 2, 3]
        data["cov_x_vx"] = covariances_np[:, 0, 2]
        data["cov_y_vy"] = covariances_np[:, 1, 3]

        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)

        # Log artifact
        self.metadata["artifacts"]["states_parquet"] = str(output_path)
        self.logger.info(f"Saved states to parquet: {output_path}")

        return output_path

    def save_residuals_parquet(
        self,
        residuals: Dict[str, jnp.ndarray],
        timestamps: Optional[jnp.ndarray] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save measurement residuals to parquet format.

        Args:
            residuals: Dictionary with residual arrays
            timestamps: Optional timestamps
            filename: Optional custom filename

        Returns:
            Path to saved parquet file
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for parquet export")

        if filename is None:
            filename = f"{self.session_name}_residuals.parquet"

        output_path = self.output_dir / filename

        # Build data dictionary
        data = {}

        if timestamps is not None:
            data["timestamp_s"] = np.array(timestamps)

        # Position residuals
        if "position" in residuals:
            pos_res = np.array(residuals["position"])
            if pos_res.ndim == 2 and pos_res.shape[1] >= 2:
                data["pos_residual_x_cm"] = pos_res[:, 0]
                data["pos_residual_y_cm"] = pos_res[:, 1]
                data["pos_residual_mag_cm"] = np.linalg.norm(pos_res, axis=1)
            else:
                data["pos_residual_cm"] = pos_res

        # Heading residuals
        if "heading" in residuals:
            heading_res = np.array(residuals["heading"])
            data["heading_residual_rad"] = heading_res
            data["heading_residual_deg"] = np.degrees(heading_res)

        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)

        # Log artifact
        self.metadata["artifacts"]["residuals_parquet"] = str(output_path)
        self.logger.info(f"Saved residuals to parquet: {output_path}")

        return output_path

    def save_artifact(
        self,
        artifact_name: str,
        file_path: Union[str, Path],
        description: Optional[str] = None,
    ) -> None:
        """
        Register an artifact (plot, data file, etc.) in the metadata.

        Args:
            artifact_name: Unique name for the artifact
            file_path: Path to the artifact file
            description: Optional description of the artifact
        """
        artifact_info = {
            "path": str(file_path),
            "created": datetime.now().isoformat(),
        }

        if description:
            artifact_info["description"] = description

        self.metadata["artifacts"][artifact_name] = artifact_info
        self.logger.info(f"Registered artifact '{artifact_name}': {file_path}")

    def save_metadata(self) -> Path:
        """
        Save metadata to JSON file.

        Returns:
            Path to saved JSON file
        """
        # Update timestamp
        self.metadata["completed"] = datetime.now().isoformat()

        with open(self.json_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        self.logger.info(f"Saved metadata to: {self.json_file}")
        return self.json_file

    def create_summary_report(self) -> str:
        """
        Create a text summary of the QA session.

        Returns:
            Summary report as string
        """
        report_lines = [
            "Trodestrack QA Session Report",
            "=" * 40,
            f"Session: {self.session_name}",
            f"Completed: {self.metadata.get('completed', 'In Progress')}",
            f"Output Directory: {self.output_dir}",
            "",
        ]

        # Parameters
        if self.metadata["parameters"]:
            report_lines.extend(
                [
                    "Parameters:",
                    "-" * 20,
                ]
            )
            for key, value in self.metadata["parameters"].items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")

        # Metrics
        if self.metadata["metrics"]:
            report_lines.extend(
                [
                    "Key Metrics:",
                    "-" * 20,
                ]
            )

            # RMSE metrics
            for key in ["position_rmse_cm", "velocity_rmse_cm_s", "heading_rmse_deg"]:
                if key in self.metadata["metrics"]:
                    value = self.metadata["metrics"][key]
                    report_lines.append(f"  {key}: {value:.3f}")

            # NEES consistency
            for key in ["nees_consistency_ratio", "position_nees_consistency_ratio"]:
                if key in self.metadata["metrics"]:
                    value = self.metadata["metrics"][key]
                    report_lines.append(f"  {key}: {value:.3f}")

            # PRD compliance
            if "overall_prd_compliant" in self.metadata["metrics"]:
                compliant = self.metadata["metrics"]["overall_prd_compliant"]
                status = "✓ PASS" if compliant else "✗ FAIL"
                report_lines.append(f"  PRD Compliance: {status}")

            report_lines.append("")

        # Artifacts
        if self.metadata["artifacts"]:
            report_lines.extend(
                [
                    "Generated Artifacts:",
                    "-" * 20,
                ]
            )
            for name, info in self.metadata["artifacts"].items():
                if isinstance(info, dict) and "path" in info:
                    path = Path(info["path"]).name
                    report_lines.append(f"  {name}: {path}")
                else:
                    report_lines.append(f"  {name}: {info}")

        return "\n".join(report_lines)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save metadata."""
        self.save_metadata()


def create_qa_session(
    output_dir: Union[str, Path], session_name: Optional[str] = None, **kwargs
) -> QALogger:
    """
    Create a new QA logging session with automatic session naming.

    Args:
        output_dir: Directory for output files
        session_name: Optional session name (auto-generated if None)
        **kwargs: Additional arguments for QALogger

    Returns:
        QALogger instance
    """
    if session_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"qa_session_{timestamp}"

    return QALogger(output_dir, session_name, **kwargs)
