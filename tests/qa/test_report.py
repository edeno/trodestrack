"""Tests for QA report generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from trodestrack.qa.report import (
    TARGET_HEADING_MAE_DEG,
    TARGET_POSITION_RMSE_M,
    TARGET_VELOCITY_RMSE_MS,
    _create_summary_page,
    _create_trajectory_plot,
    generate_qa_report,
)
from trodestrack.viz.styles import COLORS


def _summary_verdict_text(fig: plt.Figure) -> str:
    """Return the verdict banner text from a summary-page Figure.

    The verdict is set via ``fig.text(...)`` at fixed coordinates (0.5, 0.93)
    above the title and renders the PASS/FAIL block. Inspecting ``fig.texts``
    directly avoids matplotlib's PDF Type-3 font / glyph-encoding fragility
    that previously required ``pypdf`` and a ``/uniXXXXXXXX`` glyph decoder
    just to recover the same string.
    """
    for txt in fig.texts:
        body = txt.get_text()
        if body.startswith("RESULT:"):
            return body
    raise AssertionError(
        "Summary figure did not contain a 'RESULT:' verdict banner; "
        f"found texts: {[t.get_text() for t in fig.texts]}"
    )


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
        vel_true = np.column_stack(
            [-np.sin(theta_traj) * 0.15, np.cos(theta_traj) * 0.15]
        )
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


class TestSummaryVerdictBanner:
    """The summary page leads with a PASS/FAIL verdict against project targets.

    Verdict tests inspect the Figure directly instead of round-tripping
    through PDF. The previous approach needed a ``/uniXXXXXXXX`` glyph
    decoder and a ``pdf.fonttype = 42`` opt-in fixture to survive
    matplotlib's Type-3 font subsetting — both eliminated by reading
    ``fig.texts`` from the in-memory summary page.
    """

    @staticmethod
    def _make_summary(
        *, pos_rmse: float, vel_rmse: float, heading_mae_rad: float
    ) -> plt.Figure:
        nees_stats = {
            "mean": 8.0,
            "std": 1.0,
            "min": 4.0,
            "max": 14.0,
            "chi2_lower": 2.18,
            "chi2_upper": 17.5,
            "pct_in_bounds": 95.0,
            "confidence": 0.95,
        }
        return _create_summary_page(
            title="Verdict-banner test",
            pos_rmse=pos_rmse,
            vel_rmse=vel_rmse,
            heading_mae=heading_mae_rad,
            heading_rmse=heading_mae_rad,
            nees_stats=nees_stats,
            nis_stats=None,
            config=None,
        )

    def test_summary_page_shows_pass_when_all_metrics_inside_targets(self) -> None:
        """All metrics inside their targets => PASS banner."""
        fig = self._make_summary(
            pos_rmse=TARGET_POSITION_RMSE_M / 2.0,
            vel_rmse=TARGET_VELOCITY_RMSE_MS / 2.0,
            heading_mae_rad=float(np.deg2rad(TARGET_HEADING_MAE_DEG / 2.0)),
        )
        try:
            verdict = _summary_verdict_text(fig)
            assert verdict == "RESULT: PASS", verdict
        finally:
            plt.close(fig)

    def test_summary_page_shows_fail_naming_position(self) -> None:
        """A failing position metric => FAIL banner naming only 'position'."""
        fig = self._make_summary(
            pos_rmse=TARGET_POSITION_RMSE_M * 5.0,
            vel_rmse=TARGET_VELOCITY_RMSE_MS / 2.0,
            heading_mae_rad=float(np.deg2rad(TARGET_HEADING_MAE_DEG / 2.0)),
        )
        try:
            verdict = _summary_verdict_text(fig)
            assert verdict.startswith("RESULT: FAIL"), verdict
            assert "position" in verdict, verdict
            assert "velocity" not in verdict, verdict
            assert "heading" not in verdict, verdict
        finally:
            plt.close(fig)

    def test_summary_page_shows_fail_naming_all_failing_metrics(self) -> None:
        """Multiple failing metrics => FAIL banner names each of them."""
        fig = self._make_summary(
            pos_rmse=TARGET_POSITION_RMSE_M * 5.0,
            vel_rmse=TARGET_VELOCITY_RMSE_MS * 5.0,
            heading_mae_rad=float(np.deg2rad(TARGET_HEADING_MAE_DEG * 5.0)),
        )
        try:
            verdict = _summary_verdict_text(fig)
            assert verdict.startswith("RESULT: FAIL"), verdict
            assert "position" in verdict, verdict
            assert "velocity" in verdict, verdict
            assert "heading" in verdict, verdict
        finally:
            plt.close(fig)


class TestTrajectoryMarkers:
    """The trajectory start/end markers must use the Okabe-Ito colorblind pair."""

    def test_qa_report_start_end_markers_are_okabe_ito(self) -> None:
        """Start/end markers should use Okabe-Ito blue/orange, not green/red."""
        np.random.seed(0)
        N = 50
        positions_true = np.column_stack([np.linspace(0.0, 1.0, N), np.zeros(N)])
        positions_est = positions_true + np.random.randn(N, 2) * 0.001

        fig = _create_trajectory_plot(positions_true, positions_est)
        try:
            ax = fig.axes[0]
            # Find the two scatter PathCollections (start + end markers).
            scatters = ax.collections
            assert len(scatters) >= 2, (
                f"Expected start/end scatter markers; got {len(scatters)} collections"
            )
            # Collect the face colors of the scatter markers as hex strings.
            from matplotlib.colors import to_hex

            face_hexes = {
                to_hex(scatter.get_facecolor()[0]).lower() for scatter in scatters[:2]
            }
            expected = {
                COLORS["okabe_ito_blue"].lower(),
                COLORS["okabe_ito_orange"].lower(),
            }
            assert face_hexes == expected, (
                f"Marker colors {face_hexes} did not match Okabe-Ito pair {expected}"
            )
            # Defensively guard against regressing to the old green/red pair.
            forbidden = {COLORS["green"].lower(), COLORS["red"].lower()}
            assert not (face_hexes & forbidden), (
                f"Marker colors must not include the legacy green/red pair: {face_hexes}"
            )
        finally:
            plt.close(fig)
