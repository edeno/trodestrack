"""Tests for CLI report command.

Following TDD: These tests are written BEFORE implementation to define the API.

The report command should:
1. Accept a run directory with filter results
2. Generate a PDF report at the specified path
3. Handle missing/invalid inputs gracefully
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def mock_run_directory(tmp_path: Path) -> Path:
    """Create a mock run directory with filter results.

    This simulates the output of a filter run with all required files:
    - timestamps.npy
    - positions_true.npy
    - positions_est.npy
    - velocities_true.npy
    - velocities_est.npy
    - headings_true.npy
    - headings_est.npy
    - nees.npy
    - state_dim.txt (contains the state dimension)
    """
    run_dir = tmp_path / "run1"
    run_dir.mkdir()

    # Create synthetic data
    np.random.seed(42)
    N = 200
    t = np.linspace(0, 10, N)

    # Save required files
    np.save(run_dir / "timestamps.npy", t)
    np.save(run_dir / "positions_true.npy", np.column_stack([t * 0.1, np.zeros(N)]))
    np.save(
        run_dir / "positions_est.npy",
        np.column_stack([t * 0.1, np.zeros(N)]) + np.random.randn(N, 2) * 0.01,
    )
    np.save(
        run_dir / "velocities_true.npy",
        np.column_stack([np.ones(N) * 0.1, np.zeros(N)]),
    )
    np.save(
        run_dir / "velocities_est.npy",
        np.column_stack([np.ones(N) * 0.1, np.zeros(N)]) + np.random.randn(N, 2) * 0.01,
    )
    np.save(run_dir / "headings_true.npy", np.zeros(N))
    np.save(run_dir / "headings_est.npy", np.random.randn(N) * 0.1)
    np.save(run_dir / "nees.npy", np.random.chisquare(df=8, size=N))

    # Write state dimension
    (run_dir / "state_dim.txt").write_text("8")

    return run_dir


class TestReportCommand:
    """Test the 'trodestrack report' CLI command."""

    def test_report_command_basic(self, mock_run_directory: Path) -> None:
        """Test basic report generation from CLI."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Run CLI command
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "trodestrack",
                    "report",
                    "--run",
                    str(mock_run_directory),
                    "--pdf",
                    str(pdf_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Assert: Command succeeded
            assert result.returncode == 0
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0

        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    def test_report_command_missing_run_directory(self) -> None:
        """Test error handling when run directory doesn't exist."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Run CLI command with non-existent run directory
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "trodestrack",
                    "report",
                    "--run",
                    "/nonexistent/path",
                    "--pdf",
                    str(pdf_path),
                ],
                capture_output=True,
                text=True,
            )

            # Assert: Command failed with clear error message
            assert result.returncode != 0
            assert (
                "does not exist" in result.stderr.lower()
                or "not found" in result.stderr.lower()
            )

        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    def test_report_command_help(self) -> None:
        """Test that help message is displayed."""
        result = subprocess.run(
            ["uv", "run", "trodestrack", "report", "--help"],
            capture_output=True,
            text=True,
        )

        # Assert: Help message contains expected content
        assert result.returncode == 0
        assert "report" in result.stdout.lower()
        assert "--run" in result.stdout
        assert "--pdf" in result.stdout

    def test_report_command_missing_required_files(self, tmp_path: Path) -> None:
        """Test error handling when required files are missing from run directory."""
        # Create incomplete run directory (missing some required files)
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()

        # Only create timestamps.npy (missing all other files)
        np.save(run_dir / "timestamps.npy", np.linspace(0, 10, 200))

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            # Run CLI command
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "trodestrack",
                    "report",
                    "--run",
                    str(run_dir),
                    "--pdf",
                    str(pdf_path),
                ],
                capture_output=True,
                text=True,
            )

            # Assert: Command failed with clear error message about missing files
            assert result.returncode != 0
            assert (
                "missing" in result.stderr.lower()
                or "not found" in result.stderr.lower()
                or "required" in result.stderr.lower()
            )

        finally:
            if pdf_path.exists():
                pdf_path.unlink()
