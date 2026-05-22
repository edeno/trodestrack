"""Tests for QA report generation.

Following TDD: These tests are written BEFORE implementation to define the API.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from trodestrack.qa.report import (
    TARGET_HEADING_MAE_DEG,
    TARGET_POSITION_RMSE_M,
    TARGET_VELOCITY_RMSE_MS,
    _create_trajectory_plot,
    generate_qa_report,
)
from trodestrack.viz.styles import COLORS

_UNI_GLYPH = re.compile(r"/uni([0-9A-Fa-f]{4,8})")


def _decode_pdf_glyphs(text: str) -> str:
    """Decode ``/uniXXXXXXXX`` matplotlib glyph references back to Unicode.

    Matplotlib on Windows (and any platform when using non-default fonts)
    encodes characters as ``/uniHHHHHHHH`` glyph references rather than
    direct Unicode in the PDF text stream. pypdf returns those references
    verbatim, so substring searches like ``"RESULT: PASS" in extracted``
    fail even though the text is structurally present.
    """
    return _UNI_GLYPH.sub(lambda m: chr(int(m.group(1), 16)), text)


def _extract_pdf_text(pdf_path: Path) -> str:
    """Return PDF text content, preferring pypdf and falling back to raw bytes.

    Matplotlib PDFs typically keep text streams uncompressed, so a raw-byte
    search works in practice when pypdf isn't installed. The fallback mirrors
    the pattern used in ``tests/cli/test_report_command.py``.

    The ``/uniXXXXXXXX`` glyph references emitted by matplotlib on Windows
    are decoded back to Unicode so callers can do plain substring searches.
    """
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError:
        pypdf = None  # type: ignore[assignment]
    if pypdf is not None:
        reader = pypdf.PdfReader(str(pdf_path))
        raw = "".join(page.extract_text() or "" for page in reader.pages)
        return _decode_pdf_glyphs(raw)
    return pdf_path.read_bytes().decode("latin-1", errors="replace")


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
    """The summary page leads with a PASS/FAIL verdict against project targets."""

    @staticmethod
    def _render(
        pdf_path: Path,
        *,
        pos_offset: float,
        vel_offset: float,
        heading_offset_rad: float,
    ) -> None:
        """Generate a report with deterministic per-metric error magnitudes."""
        np.random.seed(0)
        N = 200
        t = np.linspace(0.0, 10.0, N)
        pos_true = np.column_stack([t * 0.1, np.zeros(N)])
        # Constant offset in x => RMSE == abs(offset) exactly.
        pos_est = pos_true + np.array([pos_offset, 0.0])
        vel_true = np.column_stack([np.ones(N) * 0.1, np.zeros(N)])
        vel_est = vel_true + np.array([vel_offset, 0.0])
        heading_true = np.zeros(N)
        heading_est = np.full(N, heading_offset_rad)
        nees = np.random.chisquare(df=8, size=N)
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

    def test_qa_report_summary_page_shows_pass_for_passing_metrics(self) -> None:
        """All metrics inside their targets should render a PASS banner."""
        # Use half the target as the per-metric offset so each RMSE/MAE is
        # comfortably under the threshold (and the test follows if the
        # constants move).
        pos_offset = TARGET_POSITION_RMSE_M / 2.0
        vel_offset = TARGET_VELOCITY_RMSE_MS / 2.0
        heading_offset_rad = float(np.deg2rad(TARGET_HEADING_MAE_DEG / 2.0))

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            self._render(
                pdf_path,
                pos_offset=pos_offset,
                vel_offset=vel_offset,
                heading_offset_rad=heading_offset_rad,
            )
            text = _extract_pdf_text(pdf_path)
            assert "RESULT: PASS" in text, text[:500]
        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    def test_qa_report_summary_page_shows_fail_with_failing_metric_names(
        self,
    ) -> None:
        """A failing position metric must produce a FAIL banner naming position."""
        # Position offset clearly exceeds the target; velocity and heading stay
        # inside theirs so the verdict cites only "position".
        pos_offset = TARGET_POSITION_RMSE_M * 5.0
        vel_offset = TARGET_VELOCITY_RMSE_MS / 2.0
        heading_offset_rad = float(np.deg2rad(TARGET_HEADING_MAE_DEG / 2.0))

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            self._render(
                pdf_path,
                pos_offset=pos_offset,
                vel_offset=vel_offset,
                heading_offset_rad=heading_offset_rad,
            )
            text = _extract_pdf_text(pdf_path)
            assert "RESULT: FAIL" in text, text[:500]
            assert "position" in text, text[:500]
        finally:
            if pdf_path.exists():
                pdf_path.unlink()


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
