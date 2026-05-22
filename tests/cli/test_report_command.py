"""Tests for the ``trodestrack report`` CLI command.

Covers the happy path (qa_inputs/ → PDF), the validation raise branches
in ``load_run_data`` (missing files, bad shapes, non-finite/non-monotonic
timestamps, bad state_dim or measurement_dim), and the optional custom
``--title`` rendering path.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trodestrack import main
from trodestrack.cli.report import load_run_data


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

    def test_report_command_nis_without_measurement_dim_errors(
        self, mock_run_directory: Path
    ) -> None:
        """``trodestrack report`` must reject ``nis.npy`` without
        ``measurement_dim.txt`` instead of silently defaulting df=4.

        The chi-square consistency bounds and "% in bounds" depend on
        ``measurement_dim`` (df=2 for position-only NIS, df=4 for
        dual-LED, etc.). Silently defaulting to 4 produced a
        successful run with potentially-wrong bounds — masking real
        consistency violations on non-dual-LED layouts.
        """
        np.save(
            mock_run_directory / "nis.npy",
            np.random.chisquare(df=2, size=200),
        )
        assert not (mock_run_directory / "measurement_dim.txt").exists()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
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
            )
            assert result.returncode != 0, (
                f"Expected non-zero exit on missing measurement_dim.txt, "
                f"got {result.returncode}; stderr={result.stderr!r}"
            )
            assert "measurement_dim.txt" in result.stderr, (
                f"Expected stderr to mention measurement_dim.txt; got {result.stderr!r}"
            )
            # Report should NOT have been written.
            assert pdf_path.stat().st_size == 0
        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    def test_report_command_nis_with_measurement_dim_succeeds(
        self, mock_run_directory: Path
    ) -> None:
        """Providing ``measurement_dim.txt`` alongside ``nis.npy`` must work."""
        np.save(
            mock_run_directory / "nis.npy",
            np.random.chisquare(df=2, size=200),
        )
        (mock_run_directory / "measurement_dim.txt").write_text("2")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
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
            )
            assert result.returncode == 0, (
                f"Expected zero exit with measurement_dim=2; stderr={result.stderr!r}"
            )
            assert pdf_path.exists() and pdf_path.stat().st_size > 0
        finally:
            if pdf_path.exists():
                pdf_path.unlink()


# In-process tests (patch sys.argv + call main directly) exercise the
# real cli.report module and so contribute to coverage; the subprocess
# tests above do not, because pytest-cov can't see into a child uv run.


def test_report_command_renders_pdf_from_qa_inputs_directory(
    tmp_path: Path, build_qa_inputs_dir
) -> None:
    """End-to-end: a qa_inputs/ directory renders a non-empty PDF.

    Covers the full ``cli.report.run_report_command`` happy path —
    loader -> ``generate_qa_report`` -> file on disk — plus the
    ``load_run_data`` return contract used by the report renderer.
    """
    n = 100
    qa_dir = build_qa_inputs_dir(tmp_path, n=n)
    pdf_path = tmp_path / "report.pdf"

    with patch(
        "sys.argv",
        [
            "trodestrack",
            "report",
            "--run",
            str(qa_dir),
            "--pdf",
            str(pdf_path),
        ],
    ):
        main()

    assert pdf_path.exists(), f"Report PDF was not written to {pdf_path}"
    assert pdf_path.stat().st_size > 1024, (
        f"Report PDF is suspiciously small ({pdf_path.stat().st_size} bytes); "
        "expected a multi-page report > 1 KB."
    )

    data = load_run_data(qa_dir)
    expected_keys = {
        "t",
        "positions_true",
        "positions_est",
        "velocities_true",
        "velocities_est",
        "headings_true",
        "headings_est",
        "nees",
        "state_dim",
    }
    assert expected_keys.issubset(data.keys()), (
        f"load_run_data missing keys: {expected_keys - set(data.keys())}"
    )


def test_report_command_raises_friendly_error_when_required_file_missing(
    tmp_path: Path, build_qa_inputs_dir, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing ``positions_true.npy`` must produce exit 1 with a clear stderr."""
    qa_dir = build_qa_inputs_dir(tmp_path, n=100)
    (qa_dir / "positions_true.npy").unlink()
    pdf_path = tmp_path / "report.pdf"

    with (
        patch(
            "sys.argv",
            [
                "trodestrack",
                "report",
                "--run",
                str(qa_dir),
                "--pdf",
                str(pdf_path),
            ],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1, (
        f"Expected exit code 1 on missing file; got {exc_info.value.code}"
    )
    captured = capsys.readouterr()
    stderr_lower = captured.err.lower()
    assert "missing" in stderr_lower or "not found" in stderr_lower, (
        f"Expected stderr to mention 'missing' / 'not found'; got {captured.err!r}"
    )
    assert "positions_true.npy" in captured.err, (
        f"Expected stderr to name the absent file; got {captured.err!r}"
    )
    assert not pdf_path.exists(), "PDF should not be written when loader fails"


def test_report_command_validates_array_shape_consistency(
    tmp_path: Path, build_qa_inputs_dir, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mismatched ``positions_est`` shape must yield a clean error.

    Regression guard: the loader's explicit shape checks should fire
    before any broadcasting / indexing produces a cryptic
    ``IndexError`` or ``ValueError: shapes ... not aligned`` from deep
    inside the report renderer.
    """
    n = 100
    qa_dir = build_qa_inputs_dir(tmp_path, n=n)
    # Overwrite positions_est with an (N+1, 2) array so it disagrees
    # with positions_true / timestamps; the loader should flag this.
    bad_positions_est = np.zeros((n + 1, 2))
    np.save(qa_dir / "positions_est.npy", bad_positions_est)
    pdf_path = tmp_path / "report.pdf"

    with (
        patch(
            "sys.argv",
            [
                "trodestrack",
                "report",
                "--run",
                str(qa_dir),
                "--pdf",
                str(pdf_path),
            ],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "positions_est" in captured.err, (
        f"Expected stderr to name the offending array; got {captured.err!r}"
    )
    assert "shape" in captured.err.lower() or "expected" in captured.err.lower(), (
        f"Expected stderr to describe a shape mismatch; got {captured.err!r}"
    )
    assert not pdf_path.exists()


def test_load_run_data_returns_expected_keys_and_shapes(
    tmp_path: Path, build_qa_inputs_dir
) -> None:
    """Direct unit test of ``load_run_data``'s return contract.

    Also drops in an optional ``nis.npy`` + ``measurement_dim.txt``
    pair so the optional-NIS branch of the loader is exercised
    alongside the required nine-file core.
    """
    n = 100
    qa_dir = build_qa_inputs_dir(tmp_path, n=n)
    # Exercise the optional-NIS branch with a matching-shape series
    # and a valid measurement dimension (dual-LED -> df=4).
    np.save(qa_dir / "nis.npy", np.random.default_rng(0).chisquare(df=4, size=n))
    (qa_dir / "measurement_dim.txt").write_text("4")

    data = load_run_data(qa_dir)

    expected_keys = {
        "t",
        "positions_true",
        "positions_est",
        "velocities_true",
        "velocities_est",
        "headings_true",
        "headings_est",
        "nees",
        "state_dim",
    }
    assert set(data.keys()) >= expected_keys

    assert data["t"].shape == (n,)
    assert data["positions_true"].shape == (n, 2)
    assert data["positions_est"].shape == (n, 2)
    assert data["velocities_true"].shape == (n, 2)
    assert data["velocities_est"].shape == (n, 2)
    assert data["headings_true"].shape == (n,)
    assert data["headings_est"].shape == (n,)
    assert data["nees"].shape == (n,)
    assert isinstance(data["state_dim"], int)
    assert data["state_dim"] >= 1

    # Optional NIS metadata loaded too.
    assert data["nis"].shape == (n,)
    assert data["measurement_dim"] == 4


def test_report_command_with_custom_title_appears_in_pdf(
    tmp_path: Path, build_qa_inputs_dir
) -> None:
    """``--title`` must propagate through to the rendered PDF."""
    title = "Session 2024-10-11"
    qa_dir = build_qa_inputs_dir(tmp_path, n=100)
    pdf_path = tmp_path / "report.pdf"

    with patch(
        "sys.argv",
        [
            "trodestrack",
            "report",
            "--run",
            str(qa_dir),
            "--pdf",
            str(pdf_path),
            "--title",
            title,
        ],
    ):
        main()

    assert pdf_path.exists() and pdf_path.stat().st_size > 1024

    # Prefer pypdf for a robust text-layer check; fall back to a raw
    # byte search (uncompressed text streams in matplotlib PDFs make
    # this work in practice, but it isn't guaranteed).
    try:
        import pypdf
    except ImportError:
        pypdf = None

    if pypdf is not None:
        import re as _re

        reader = pypdf.PdfReader(str(pdf_path))
        extracted = "".join(page.extract_text() or "" for page in reader.pages)
        # Matplotlib on Windows emits text as /uniXXXXXXXX glyph references
        # that pypdf does not auto-decode to Unicode. Decode them so the
        # plain substring check works cross-platform.
        extracted = _re.sub(
            r"/uni([0-9A-Fa-f]{4,8})",
            lambda m: chr(int(m.group(1), 16)),
            extracted,
        )
        assert title in extracted, (
            f"Title {title!r} not found in extracted PDF text; "
            f"first 500 chars: {extracted[:500]!r}"
        )
    else:
        pdf_bytes = pdf_path.read_bytes()
        assert pdf_bytes.find(title.encode("utf-8")) != -1, (
            f"Title bytes {title!r} not found in raw PDF; "
            "install pypdf for a more robust check."
        )


def _bad_state_dim(qa_dir: Path) -> None:
    (qa_dir / "state_dim.txt").write_text("not-an-int")


def _zero_state_dim(qa_dir: Path) -> None:
    (qa_dir / "state_dim.txt").write_text("0")


def _non_finite_timestamps(qa_dir: Path) -> None:
    t = np.linspace(0.0, 1.0, 100)
    t[5] = np.nan
    np.save(qa_dir / "timestamps.npy", t)


def _non_monotonic_timestamps(qa_dir: Path) -> None:
    t = np.linspace(0.0, 1.0, 100)
    t[10] = t[9]
    np.save(qa_dir / "timestamps.npy", t)


def _two_d_timestamps(qa_dir: Path) -> None:
    np.save(qa_dir / "timestamps.npy", np.zeros((100, 2)))


def _too_short_timestamps(qa_dir: Path) -> None:
    np.save(qa_dir / "timestamps.npy", np.array([0.0]))


def _bad_nis_shape(qa_dir: Path) -> None:
    np.save(qa_dir / "nis.npy", np.zeros(50))


def _bad_measurement_dim(qa_dir: Path) -> None:
    np.save(qa_dir / "nis.npy", np.zeros(100))
    (qa_dir / "measurement_dim.txt").write_text("not-an-int")


def _zero_measurement_dim(qa_dir: Path) -> None:
    np.save(qa_dir / "nis.npy", np.zeros(100))
    (qa_dir / "measurement_dim.txt").write_text("0")


def _bad_velocities_true_shape(qa_dir: Path) -> None:
    np.save(qa_dir / "velocities_true.npy", np.zeros((50, 2)))


def _bad_velocities_est_shape(qa_dir: Path) -> None:
    np.save(qa_dir / "velocities_est.npy", np.zeros((50, 2)))


def _bad_headings_true_shape(qa_dir: Path) -> None:
    np.save(qa_dir / "headings_true.npy", np.zeros(50))


def _bad_headings_est_shape(qa_dir: Path) -> None:
    np.save(qa_dir / "headings_est.npy", np.zeros(50))


def _bad_nees_shape(qa_dir: Path) -> None:
    np.save(qa_dir / "nees.npy", np.zeros(50))


def _bad_positions_true_shape(qa_dir: Path) -> None:
    np.save(qa_dir / "positions_true.npy", np.zeros((50, 2)))


@pytest.mark.parametrize(
    "corrupt,match",
    [
        (_bad_state_dim, "state_dim.txt must contain an integer"),
        (_zero_state_dim, "state_dim must be a positive integer"),
        (_non_finite_timestamps, "non-finite"),
        (_non_monotonic_timestamps, "strictly increasing"),
        (_two_d_timestamps, "1D array"),
        (_too_short_timestamps, "at least two samples"),
        (_bad_nis_shape, "nis has shape"),
        (_bad_measurement_dim, "measurement_dim.txt must contain an integer"),
        (_zero_measurement_dim, "measurement_dim must be a positive integer"),
        (_bad_positions_true_shape, "positions_true has shape"),
        (_bad_velocities_true_shape, "velocities_true has shape"),
        (_bad_velocities_est_shape, "velocities_est has shape"),
        (_bad_headings_true_shape, "headings_true has shape"),
        (_bad_headings_est_shape, "headings_est has shape"),
        (_bad_nees_shape, "nees has shape"),
    ],
)
def test_load_run_data_validates_inputs(
    tmp_path: Path, build_qa_inputs_dir, corrupt, match
) -> None:
    """``load_run_data`` raises on the documented corruption modes.

    Pure validation-path coverage for the ``raise`` branches in
    ``cli.report.load_run_data`` that the happy-path tests do not exercise.
    """
    qa_dir = build_qa_inputs_dir(tmp_path, n=100)
    corrupt(qa_dir)
    with pytest.raises(ValueError, match=match):
        load_run_data(qa_dir)


def test_load_run_data_rejects_missing_run_dir(tmp_path: Path) -> None:
    """Nonexistent run directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_run_data(tmp_path / "does_not_exist")


def test_load_run_data_rejects_file_as_run_dir(tmp_path: Path) -> None:
    """Path that is a file (not a directory) raises NotADirectoryError."""
    file_path = tmp_path / "regular_file.txt"
    file_path.write_text("hello")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        load_run_data(file_path)


def _build_smooth_output_dir(
    tmp_path: Path, *, n: int = 80, state_mode: str = "2d_cam_3d_imu"
) -> tuple[Path, Path, Path]:
    """Synthesize a filter/smooth output directory plus ground-truth files.

    Builds ``run_dir/{smoothed,filtered}_means.txt`` and their flat
    covariance siblings under the named ``state_mode``, with a positive-
    definite covariance series and a smooth circular ground-truth
    trajectory. Returns the run dir plus paths to the gt position and
    heading text files.
    """
    from trodestrack.models.state_layout import get_layout

    run_dir = tmp_path / "smooth_run"
    run_dir.mkdir()

    layout = get_layout(state_mode)
    n_state = layout.n
    rng = np.random.default_rng(0)

    t_axis = np.linspace(0.0, n / 30.0, n)
    radius = 0.3
    angular_velocity = 0.4
    theta = t_axis * angular_velocity
    positions_true = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    headings_true = theta + np.pi / 2

    means = np.zeros((n, n_state))
    pos_idx = list(layout.pos_idx)
    vel_idx = list(layout.vel_idx[:2])
    head_idx = int(layout.heading_idx)  # only 2D layouts are exercised here
    means[:, pos_idx] = positions_true + rng.standard_normal((n, 2)) * 0.005
    means[:, vel_idx] = (
        np.column_stack(
            [
                -radius * angular_velocity * np.sin(theta),
                radius * angular_velocity * np.cos(theta),
            ]
        )
        + rng.standard_normal((n, 2)) * 0.01
    )
    means[:, head_idx] = headings_true + rng.standard_normal(n) * 0.05

    # Diagonal covariance avoids any PSD pathology and gives compute_nees
    # well-defined chi-squared(3) samples on the (x, y, heading) sub-state.
    cov_diag = np.full(n_state, 1e-3)
    covs = np.broadcast_to(np.diag(cov_diag), (n, n_state, n_state)).copy()

    np.savetxt(run_dir / "smoothed_means.txt", means)
    np.savetxt(
        run_dir / "smoothed_covariances.txt",
        covs.reshape(n, n_state * n_state),
    )
    np.savetxt(run_dir / "filtered_means.txt", means)
    np.savetxt(
        run_dir / "filtered_covariances.txt",
        covs.reshape(n, n_state * n_state),
    )
    (run_dir / "metadata.txt").write_text(
        "trodestrack smooth - test fixture\n"
        + "=" * 40
        + "\n\n"
        + "Filter Configuration (effective values):\n"
        + f"  State mode: {state_mode}\n"
    )

    gt_pos_path = tmp_path / "truth_pos.txt"
    gt_head_path = tmp_path / "truth_head.txt"
    np.savetxt(gt_pos_path, positions_true)
    np.savetxt(gt_head_path, headings_true)

    return run_dir, gt_pos_path, gt_head_path


def test_report_from_run_synthesizes_qa_inputs_from_smooth_output(
    tmp_path: Path,
) -> None:
    """``trodestrack report --from-run`` must render a PDF directly from
    a filter/smooth output directory once ground truth is supplied."""
    run_dir, gt_pos_path, gt_head_path = _build_smooth_output_dir(tmp_path, n=80)
    pdf_path = tmp_path / "report.pdf"

    with patch(
        "sys.argv",
        [
            "trodestrack",
            "report",
            "--from-run",
            str(run_dir),
            "--ground-truth-positions",
            str(gt_pos_path),
            "--ground-truth-headings",
            str(gt_head_path),
            "--pdf",
            str(pdf_path),
        ],
    ):
        main()

    assert pdf_path.exists(), f"Report PDF was not written to {pdf_path}"
    assert pdf_path.stat().st_size > 1024, (
        f"Report PDF is suspiciously small ({pdf_path.stat().st_size} bytes); "
        "expected a multi-page report > 1 KB."
    )


def test_report_from_run_requires_ground_truth_args(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--from-run`` without ground-truth flags must exit 1 with a clear error."""
    run_dir, _, _ = _build_smooth_output_dir(tmp_path, n=80)
    pdf_path = tmp_path / "report.pdf"

    with (
        patch(
            "sys.argv",
            [
                "trodestrack",
                "report",
                "--from-run",
                str(run_dir),
                "--pdf",
                str(pdf_path),
            ],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "--ground-truth-positions" in captured.err, (
        f"Expected stderr to name the missing flag; got {captured.err!r}"
    )
    assert "--ground-truth-headings" in captured.err, (
        f"Expected stderr to name the missing flag; got {captured.err!r}"
    )
    assert not pdf_path.exists()
