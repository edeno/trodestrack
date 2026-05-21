"""Shared fixtures for ``tests/cli/``.

The ``_build_qa_inputs_dir`` helper synthesizes the directory layout
that ``cli.report.load_run_data`` consumes — paired ground-truth and
estimated trajectories plus NEES — so report-command tests don't have
to hand-roll the same nine-file fixture each time. The layout mirrors
``examples/08_qa_report_generation.py`` and the contract documented in
``cli/report.py::load_run_data``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _build_qa_inputs_dir(tmp_path: Path, n: int = 100) -> Path:
    """Create a ``qa_inputs/`` directory under ``tmp_path``.

    Writes deterministic arrays in the layout consumed by
    ``cli.report.load_run_data``: a 1D ``timestamps.npy``, paired
    ``(N, 2)`` position/velocity arrays, ``(N,)`` headings and NEES,
    and a ``state_dim.txt`` with the integer state dimensionality.

    Parameters
    ----------
    tmp_path : Path
        Pytest tmp_path-style directory to create ``qa_inputs/`` inside.
    n : int, optional
        Number of camera samples. Default 100 keeps the report fast
        while satisfying the loader's ``size >= 2`` requirement on
        ``timestamps.npy`` and giving the QA plots enough data to render.

    Returns
    -------
    Path
        Path to the created ``qa_inputs/`` directory.
    """
    rng = np.random.default_rng(42)
    qa_dir = tmp_path / "qa_inputs"
    qa_dir.mkdir()

    duration = n / 30.0  # ~30 Hz camera rate
    t = np.linspace(0.0, duration, n)

    # Circular trajectory mirrors examples/08_qa_report_generation.py
    angular_velocity = 0.3
    radius = 0.3
    theta = t * angular_velocity
    positions_true = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    velocities_true = np.column_stack(
        [
            -radius * angular_velocity * np.sin(theta),
            radius * angular_velocity * np.cos(theta),
        ]
    )
    headings_true = theta + np.pi / 2

    positions_est = positions_true + rng.standard_normal((n, 2)) * 0.01
    velocities_est = velocities_true + rng.standard_normal((n, 2)) * 0.02
    headings_est = headings_true + rng.standard_normal(n) * np.deg2rad(3)

    # state_dim=10 matches the default 2d_cam_3d_imu state used by
    # the smooth/online commands; load_run_data uses it as the NEES
    # chi-squared df, so a positive integer is the only contract.
    state_dim = 10
    nees = rng.chisquare(df=state_dim, size=n)

    np.save(qa_dir / "timestamps.npy", t)
    np.save(qa_dir / "positions_true.npy", positions_true)
    np.save(qa_dir / "positions_est.npy", positions_est)
    np.save(qa_dir / "velocities_true.npy", velocities_true)
    np.save(qa_dir / "velocities_est.npy", velocities_est)
    np.save(qa_dir / "headings_true.npy", headings_true)
    np.save(qa_dir / "headings_est.npy", headings_est)
    np.save(qa_dir / "nees.npy", nees)
    (qa_dir / "state_dim.txt").write_text(str(state_dim))

    return qa_dir


@pytest.fixture
def build_qa_inputs_dir():
    """Expose ``_build_qa_inputs_dir`` to tests as a callable fixture.

    Using a fixture (rather than a cross-module import like
    ``from tests.cli.conftest import _build_qa_inputs_dir``) is the
    idiomatic pytest pattern and avoids depending on whether
    ``tests/`` is on ``sys.path``.
    """
    return _build_qa_inputs_dir
