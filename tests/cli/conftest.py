"""Shared fixtures for ``tests/cli/``.

The ``build_qa_inputs_dir`` fixture synthesizes the directory layout
that ``cli.report.load_run_data`` consumes — paired ground-truth and
estimated trajectories plus NEES — so report-command tests don't have
to hand-roll the same nine-file fixture each time. The layout mirrors
``examples/08_qa_report_generation.py`` and the contract documented in
``cli/report.py::load_run_data``.

The ``assert_outputs_are_finite_and_psd`` fixture is the shared
finiteness + symmetry + positive-definiteness check that the smooth
and filter command tests both use against ``smoothed_*.txt`` /
``filtered_*.txt`` outputs.
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
    # the smooth/filter commands; load_run_data uses it as the NEES
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


def _assert_outputs_are_finite_and_psd(
    means_path: Path, cov_path: Path, *, rtol: float = 1e-8
) -> None:
    """Shared finiteness + symmetry + PD check for smooth/filter outputs.

    Loads the flat covariance file ``(n_cam, state_dim**2)``, reshapes
    to ``(n_cam, state_dim, state_dim)``, then asserts:
      * all entries finite (no NaN/Inf in means or covariances),
      * symmetric within ``rtol * max(|covs|, 1)`` on every frame,
      * positive-definite (smallest eigenvalue > 0) on every frame.
    """

    means = np.loadtxt(means_path)
    flat_covs = np.loadtxt(cov_path)
    n_cam = means.shape[0]
    state_dim = means.shape[1]
    assert flat_covs.shape == (n_cam, state_dim * state_dim)
    covs = flat_covs.reshape(n_cam, state_dim, state_dim)

    assert np.all(np.isfinite(means)), "means contain NaN/Inf"
    assert np.all(np.isfinite(covs)), "covariances contain NaN/Inf"

    # Symmetry within rtol on every frame.
    asym = np.abs(covs - np.swapaxes(covs, 1, 2))
    max_sym_violation = float(asym.max())
    cov_scale = float(np.abs(covs).max())
    assert max_sym_violation <= rtol * max(cov_scale, 1.0), (
        f"covariance asymmetry {max_sym_violation:.3e} exceeds "
        f"rtol*scale={rtol * cov_scale:.3e}"
    )

    # Positive definiteness (smallest eigenvalue > 0 on every frame).
    sym_covs = 0.5 * (covs + np.swapaxes(covs, 1, 2))
    eigs = np.linalg.eigvalsh(sym_covs)
    min_eig = float(eigs.min())
    assert min_eig > 0.0, f"covariance not PD: min eigenvalue {min_eig:.3e}"


@pytest.fixture
def assert_outputs_are_finite_and_psd():
    """Expose ``_assert_outputs_are_finite_and_psd`` as a callable fixture."""
    return _assert_outputs_are_finite_and_psd


def _smooth_filter_io_args(input_dir: Path, output_dir: Path) -> list[str]:
    """Build the standard input/output flag list for ``smooth`` / ``filter``.

    Both subcommands take the same six input flags (IMU timestamps,
    IMU measurements, camera timestamps, two LED position files,
    camera mask) and a single output-dir flag. The smooth and filter
    tests reproduce this 14-element list verbatim, so factor it out.
    """
    return [
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
        "--camera-mask",
        str(input_dir / "mask_cam.txt"),
        "--output-dir",
        str(output_dir),
    ]


@pytest.fixture
def smooth_filter_io_args():
    """Expose ``_smooth_filter_io_args`` as a callable fixture."""
    return _smooth_filter_io_args
