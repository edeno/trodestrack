"""Regression tests for ``filter`` / ``smooth`` CLI ValueError handling.

The ``trodestrack filter`` and ``trodestrack smooth`` subcommands
construct ``EKFConfig`` and run the EKF / RTS smoother. A bad numeric
flag (e.g. ``--process-noise-pos -1`` or ``--num-iter 0``) used to
bubble out as a raw Python traceback because, unlike
``trodestrack report``, neither command wrapped its body in a
``ValueError`` handler. These tests exercise the wrapper added in
:func:`trodestrack.cli.filter.run_filter` and
:func:`trodestrack.cli.smooth.run_smooth`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trodestrack import main
from trodestrack.cli.utils import friendly_cli_errors


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def synthetic_inputs(temp_dir):
    """Minimal valid input files so the CLI reaches the EKF stage."""
    input_dir = temp_dir / "input"
    input_dir.mkdir()

    duration_s = 1.0
    fs_imu = 200.0
    fs_cam = 30.0

    n_imu = int(duration_s * fs_imu)
    t_imu = np.linspace(0.0, duration_s, n_imu)
    U_imu = np.zeros((n_imu, 3))

    n_cam = int(duration_s * fs_cam)
    t_cam = np.linspace(0.0, duration_s, n_cam)
    Z_cam_led1 = np.tile([0.5, 0.5], (n_cam, 1))
    Z_cam_led2 = np.tile([0.54, 0.5], (n_cam, 1))
    mask_cam = np.ones(n_cam, dtype=int)

    np.savetxt(input_dir / "t_imu.txt", t_imu)
    np.savetxt(input_dir / "U_imu.txt", U_imu)
    np.savetxt(input_dir / "t_cam.txt", t_cam)
    np.savetxt(input_dir / "Z_cam_led1.txt", Z_cam_led1)
    np.savetxt(input_dir / "Z_cam_led2.txt", Z_cam_led2)
    np.savetxt(input_dir / "mask_cam.txt", mask_cam, fmt="%d")
    return input_dir


def _filter_argv(input_dir: Path, output_dir: Path, *extra: str) -> list[str]:
    return [
        "trodestrack",
        "filter",
        "--imu-timestamps",
        str(input_dir / "t_imu.txt"),
        "--imu-measurements",
        str(input_dir / "U_imu.txt"),
        "--camera-timestamps",
        str(input_dir / "t_cam.txt"),
        "--led1-positions",
        str(input_dir / "Z_cam_led1.txt"),
        "--led2-positions",
        str(input_dir / "Z_cam_led2.txt"),
        "--output-dir",
        str(output_dir),
        *extra,
    ]


def _smooth_argv(input_dir: Path, output_dir: Path, *extra: str) -> list[str]:
    return [
        "trodestrack",
        "smooth",
        "--imu-timestamps",
        str(input_dir / "t_imu.txt"),
        "--imu-measurements",
        str(input_dir / "U_imu.txt"),
        "--camera-timestamps",
        str(input_dir / "t_cam.txt"),
        "--led1-positions",
        str(input_dir / "Z_cam_led1.txt"),
        "--led2-positions",
        str(input_dir / "Z_cam_led2.txt"),
        "--output-dir",
        str(output_dir),
        *extra,
    ]


def test_filter_negative_process_noise_prints_error_not_traceback(
    synthetic_inputs, temp_dir, capsys
) -> None:
    """``--process-noise-pos -1`` must exit 1 with ``Error:``, no traceback.

    The probe in the finding: ``EKFConfig`` validation rejects
    negative process noise via ``ValueError``. Without the wrapper,
    the raw exception propagated and Python printed a traceback
    before the message. The wrapper added to ``run_filter`` should
    convert this to a clean ``Error: ... must be ...`` line.
    """
    output_dir = temp_dir / "run"
    argv = _filter_argv(synthetic_inputs, output_dir, "--process-noise-pos", "-1")
    with patch("sys.argv", argv), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Error:" in captured.err, (
        f"Expected 'Error:' prefix in stderr, got: {captured.err!r}"
    )
    assert "Traceback" not in captured.err, (
        f"Expected no traceback in stderr, got: {captured.err!r}"
    )


@pytest.mark.parametrize("debug_value", ["1", "true", "yes", "TRUE", "Yes"])
def test_friendly_cli_errors_reraises_when_debug_env_set(
    monkeypatch, debug_value: str
) -> None:
    """``TRODESTRACK_DEBUG`` (truthy values) should let exceptions propagate.

    The wrapper normally converts ``ValueError`` (and friends) into a
    stderr line + ``sys.exit(1)``. When the debug env var is set to a
    truthy value the wrapper should instead re-raise so users get a full
    traceback for bug reports.
    """
    monkeypatch.setenv("TRODESTRACK_DEBUG", debug_value)

    @friendly_cli_errors
    def boom() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError, match="x"):
        boom()


def test_friendly_cli_errors_includes_exception_type_in_unexpected(
    capsys,
) -> None:
    """Unexpected errors should name the exception class and the debug hint.

    A ``KeyError`` is not in the (FileNotFoundError, NotADirectoryError,
    ValueError) friendly set so it hits the generic ``Exception`` branch.
    The stderr message must include ``KeyError`` (so bug reports identify
    the failure mode) and the ``TRODESTRACK_DEBUG=1`` hint.
    """

    @friendly_cli_errors
    def boom() -> None:
        raise KeyError("missing")

    with pytest.raises(SystemExit) as exc_info:
        boom()
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "KeyError" in captured.err, (
        f"Expected 'KeyError' in stderr, got: {captured.err!r}"
    )
    assert "TRODESTRACK_DEBUG=1" in captured.err, (
        f"Expected 'TRODESTRACK_DEBUG=1' hint in stderr, got: {captured.err!r}"
    )


def test_smooth_zero_num_iter_prints_error_not_traceback(
    synthetic_inputs, temp_dir, capsys
) -> None:
    """``--num-iter 0`` must exit 1 with ``Error:``, no traceback.

    The probe in the finding: ``rts_smoother`` validates ``num_iter
    >= 1``. The forward filter still runs; without the wrapper the
    smoother's ``ValueError`` bubbled past argparse and dumped a
    traceback. The wrapper added to ``run_smooth`` should print a
    clean ``Error: ...`` line instead.
    """
    output_dir = temp_dir / "run"
    argv = _smooth_argv(synthetic_inputs, output_dir, "--num-iter", "0")
    with patch("sys.argv", argv), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Error:" in captured.err, (
        f"Expected 'Error:' prefix in stderr, got: {captured.err!r}"
    )
    assert "Traceback" not in captured.err, (
        f"Expected no traceback in stderr, got: {captured.err!r}"
    )
