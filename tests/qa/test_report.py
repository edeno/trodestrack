"""Tests for QA report generation.

Following TDD: These tests are written BEFORE implementation to define the API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from trodestrack.qa.report import generate_qa_report


class TestGenerateQAReport:
    """Test PDF report generation with metrics and plots."""

    def test_basic_report_creation(self) -> None:
        """Test that basic report is created with minimal inputs."""
        # Arrange: Create synthetic filter results
        np.random.seed(42)
        N = 200
        t = np.linspace(0, 10, N)

        # Ground truth and estimates
        pos_true = np.column_stack([t * 0.1, np.zeros(N)])
        pos_est = pos_true + np.random.randn(N, 2) * 0.01
        vel_true = np.column_stack([np.ones(N) * 0.1, np.zeros(N)])
        vel_est = vel_true + np.random.randn(N, 2) * 0.01
        heading_true = np.zeros(N)
        heading_est = np.random.randn(N) * 0.1

        # NEES values
        nees = np.random.chisquare(df=8, size=N)

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Act: Generate report
            generate_qa_report(
                pdf_path=pdf_path,
                t=t,
                positions_true=pos_true,
                positions_est=pos_est,
                velocities_true=vel_true,
                velocities_est=vel_est,
                headings_true=heading_true,
                headings_est=heading_est,
                nees=nees,
                state_dim=8,
            )

            # Assert: PDF file exists and has content
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0

        finally:
            # Cleanup
            if pdf_path.exists():
                pdf_path.unlink()

    def test_report_with_optional_parameters(self) -> None:
        """Test report generation with optional parameters (NIS, covariances, config)."""
        # Arrange
        np.random.seed(42)
        N = 200
        t = np.linspace(0, 10, N)

        pos_true = np.column_stack([t * 0.1, np.zeros(N)])
        pos_est = pos_true + np.random.randn(N, 2) * 0.01
        vel_true = np.column_stack([np.ones(N) * 0.1, np.zeros(N)])
        vel_est = vel_true + np.random.randn(N, 2) * 0.01
        heading_true = np.zeros(N)
        heading_est = np.random.randn(N) * 0.1

        nees = np.random.chisquare(df=8, size=N)
        nis = np.random.chisquare(df=4, size=N)  # Optional

        # Optional config dictionary
        config = {
            "filter_type": "EKF",
            "process_noise_pos": 0.02,
            "measurement_noise": 0.005**2,
            "imu_gyro_noise_density": 0.001,
        }

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Act
            generate_qa_report(
                pdf_path=pdf_path,
                t=t,
                positions_true=pos_true,
                positions_est=pos_est,
                velocities_true=vel_true,
                velocities_est=vel_est,
                headings_true=heading_true,
                headings_est=heading_est,
                nees=nees,
                state_dim=8,
                nis=nis,
                measurement_dim=4,
                config=config,
            )

            # Assert
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0

        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    def test_report_with_trajectory_plot(self) -> None:
        """Test that report includes 2D trajectory plot."""
        # Arrange: Create circular trajectory
        np.random.seed(42)
        N = 200
        t = np.linspace(0, 10, N)
        theta_traj = t * 0.5  # Angular position

        pos_true = np.column_stack([np.cos(theta_traj) * 0.3, np.sin(theta_traj) * 0.3])
        pos_est = pos_true + np.random.randn(N, 2) * 0.01
        vel_true = np.column_stack([-np.sin(theta_traj) * 0.15, np.cos(theta_traj) * 0.15])
        vel_est = vel_true + np.random.randn(N, 2) * 0.01
        heading_true = theta_traj
        heading_est = heading_true + np.random.randn(N) * 0.1
        nees = np.random.chisquare(df=8, size=N)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Act
            generate_qa_report(
                pdf_path=pdf_path,
                t=t,
                positions_true=pos_true,
                positions_est=pos_est,
                velocities_true=vel_true,
                velocities_est=vel_est,
                headings_true=heading_true,
                headings_est=heading_est,
                nees=nees,
                state_dim=8,
            )

            # Assert: PDF created
            assert pdf_path.exists()

        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    def test_report_validation_shape_mismatch(self) -> None:
        """Test that shape mismatches raise ValueError."""
        # Arrange: Mismatched array lengths
        t = np.linspace(0, 10, 100)
        pos_true = np.random.randn(100, 2)
        pos_est = np.random.randn(50, 2)  # Wrong length
        vel_true = np.random.randn(100, 2)
        vel_est = np.random.randn(100, 2)
        heading_true = np.zeros(100)
        heading_est = np.zeros(100)
        nees = np.random.randn(100)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Act/Assert: Should raise ValueError
            with pytest.raises(ValueError, match="Shape mismatch"):
                generate_qa_report(
                    pdf_path=pdf_path,
                    t=t,
                    positions_true=pos_true,
                    positions_est=pos_est,
                    velocities_true=vel_true,
                    velocities_est=vel_est,
                    headings_true=heading_true,
                    headings_est=heading_est,
                    nees=nees,
                    state_dim=8,
                )

        finally:
            # Cleanup in case file was created
            if pdf_path.exists():
                pdf_path.unlink()

    def test_report_pdf_path_validation(self) -> None:
        """Test that invalid PDF path raises error."""
        # Arrange
        t = np.linspace(0, 10, 100)
        pos_true = np.random.randn(100, 2)
        pos_est = np.random.randn(100, 2)
        vel_true = np.random.randn(100, 2)
        vel_est = np.random.randn(100, 2)
        heading_true = np.zeros(100)
        heading_est = np.zeros(100)
        nees = np.random.randn(100)

        # Invalid path (directory that doesn't exist)
        pdf_path = Path("/nonexistent/directory/report.pdf")

        # Act/Assert
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            generate_qa_report(
                pdf_path=pdf_path,
                t=t,
                positions_true=pos_true,
                positions_est=pos_est,
                velocities_true=vel_true,
                velocities_est=vel_est,
                headings_true=heading_true,
                headings_est=heading_est,
                nees=nees,
                state_dim=8,
            )

    def test_report_includes_summary_statistics(self) -> None:
        """Test that report includes summary statistics (RMSE, NEES stats)."""
        # This test verifies the report contains expected sections
        # We can't easily validate PDF contents without parsing it,
        # but we can ensure the function completes without error
        # and produces a non-empty file

        np.random.seed(42)
        N = 200
        t = np.linspace(0, 10, N)

        pos_true = np.column_stack([t * 0.1, np.zeros(N)])
        pos_est = pos_true + np.random.randn(N, 2) * 0.01
        vel_true = np.column_stack([np.ones(N) * 0.1, np.zeros(N)])
        vel_est = vel_true + np.random.randn(N, 2) * 0.01
        heading_true = np.zeros(N)
        heading_est = np.random.randn(N) * 0.1
        nees = np.random.chisquare(df=8, size=N)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Act
            generate_qa_report(
                pdf_path=pdf_path,
                t=t,
                positions_true=pos_true,
                positions_est=pos_est,
                velocities_true=vel_true,
                velocities_est=vel_est,
                headings_true=heading_true,
                headings_est=heading_est,
                nees=nees,
                state_dim=8,
            )

            # Assert: PDF is reasonably sized (contains plots and text)
            # A minimal PDF with plots should be at least 50KB
            assert pdf_path.stat().st_size > 50000

        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    def test_report_with_title(self) -> None:
        """Test report generation with custom title."""
        np.random.seed(42)
        N = 100
        t = np.linspace(0, 10, N)

        pos_true = np.random.randn(N, 2)
        pos_est = pos_true + np.random.randn(N, 2) * 0.01
        vel_true = np.random.randn(N, 2)
        vel_est = vel_true + np.random.randn(N, 2) * 0.01
        heading_true = np.zeros(N)
        heading_est = np.zeros(N)
        nees = np.random.chisquare(df=8, size=N)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Act: Generate with custom title
            generate_qa_report(
                pdf_path=pdf_path,
                t=t,
                positions_true=pos_true,
                positions_est=pos_est,
                velocities_true=vel_true,
                velocities_est=vel_est,
                headings_true=heading_true,
                headings_est=heading_est,
                nees=nees,
                state_dim=8,
                title="Custom Test Report",
            )

            # Assert
            assert pdf_path.exists()

        finally:
            if pdf_path.exists():
                pdf_path.unlink()
