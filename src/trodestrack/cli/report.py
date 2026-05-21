"""CLI command for generating QA reports from QA-prepared run directories.

This module provides the 'trodestrack report' command that reads a directory
of QA-prepared arrays (paired ground-truth and estimated trajectories plus
NEES/NIS series) and generates a comprehensive PDF report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trodestrack.cli.utils import friendly_cli_errors
from trodestrack.qa.report import generate_qa_report


def add_report_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the 'report' subcommand to the argument parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object to add the report command to.
    """
    parser = subparsers.add_parser(
        "report",
        help="Generate PDF QA report from a QA-prepared run directory",
        description=(
            "Generate a comprehensive PDF report with accuracy metrics, "
            "consistency checks, and diagnostic plots from a QA-prepared "
            "run directory containing paired ground-truth and estimated "
            "trajectories (positions_true.npy, positions_est.npy, etc.) "
            "and NEES/NIS series. Filter outputs from `trodestrack filter` "
            "or `trodestrack smooth` are not in this format directly — "
            "ground-truth arrays must be supplied alongside the estimates."
        ),
    )

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to QA-prepared run directory (paired ground-truth and "
        "estimated arrays plus NEES/NIS .npy files; see command description)",
        metavar="DIR",
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to output PDF report",
        metavar="FILE",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the report (default: auto-generated from run directory)",
    )

    parser.set_defaults(func=run_report_command)


def load_run_data(run_dir: Path) -> dict[str, NDArray | int]:
    """Load filter results from a run directory.

    Parameters
    ----------
    run_dir : Path
        Path to run directory containing .npy files.

    Returns
    -------
    dict
        Dictionary containing loaded arrays and metadata.

    Raises
    ------
    FileNotFoundError
        If any required files are missing.
    ValueError
        If data shapes are inconsistent.
    """
    required_files = [
        "timestamps.npy",
        "positions_true.npy",
        "positions_est.npy",
        "velocities_true.npy",
        "velocities_est.npy",
        "headings_true.npy",
        "headings_est.npy",
        "nees.npy",
        "state_dim.txt",
    ]

    # Check that run directory exists
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    if not run_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {run_dir}")

    # Check that all required files exist
    missing_files = []
    for filename in required_files:
        file_path = run_dir / filename
        if not file_path.exists():
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {run_dir}: {', '.join(missing_files)}"
        )

    # Load data
    data = {
        "t": np.load(run_dir / "timestamps.npy"),
        "positions_true": np.load(run_dir / "positions_true.npy"),
        "positions_est": np.load(run_dir / "positions_est.npy"),
        "velocities_true": np.load(run_dir / "velocities_true.npy"),
        "velocities_est": np.load(run_dir / "velocities_est.npy"),
        "headings_true": np.load(run_dir / "headings_true.npy"),
        "headings_est": np.load(run_dir / "headings_est.npy"),
        "nees": np.load(run_dir / "nees.npy"),
    }

    # Validate the time axis. generate_qa_report enforces this too, but
    # surfacing the failure at the loader names the offending file
    # (timestamps.npy) directly rather than the in-memory array.
    t_arr = data["t"]
    if t_arr.ndim != 1:
        raise ValueError(
            f"timestamps.npy must contain a 1D array; got shape {t_arr.shape}."
        )
    if t_arr.size < 2:
        raise ValueError(
            f"timestamps.npy must have at least two samples; got shape {t_arr.shape}."
        )
    if not np.all(np.isfinite(t_arr)):
        n_bad = int(np.sum(~np.isfinite(t_arr)))
        raise ValueError(
            f"timestamps.npy contains {n_bad} non-finite value(s) (NaN/inf); "
            "QA-report timestamps must be finite seconds."
        )
    if not np.all(np.diff(t_arr) > 0):
        first_bad = int(np.argmax(np.diff(t_arr) <= 0))
        raise ValueError(
            "timestamps.npy must be strictly increasing; first non-increasing "
            f"step at index {first_bad + 1} (t[{first_bad}]={t_arr[first_bad]!r}, "
            f"t[{first_bad + 1}]={t_arr[first_bad + 1]!r})."
        )

    # Load state dimension. Must be a positive integer — chi2_bounds /
    # compute_nees_stats use it as the chi-squared df, and df <= 0 makes
    # scipy.stats.chi2.ppf return NaN bounds that would silently embed in
    # the QA report.
    state_dim_text = (run_dir / "state_dim.txt").read_text().strip()
    try:
        state_dim = int(state_dim_text)
    except ValueError as e:
        raise ValueError(
            f"state_dim.txt must contain an integer; got {state_dim_text!r}."
        ) from e
    if state_dim < 1:
        raise ValueError(
            "state_dim must be a positive integer (degrees of freedom for "
            f"NEES chi-squared bounds); got {state_dim} from state_dim.txt."
        )
    data["state_dim"] = state_dim

    # Validate data shapes for consistency
    N = len(data["t"])
    if data["positions_true"].shape != (N, 2):
        raise ValueError(
            f"positions_true has shape {data['positions_true'].shape}, expected ({N}, 2)"
        )
    if data["positions_est"].shape != (N, 2):
        raise ValueError(
            f"positions_est has shape {data['positions_est'].shape}, expected ({N}, 2)"
        )
    if data["velocities_true"].shape != (N, 2):
        raise ValueError(
            f"velocities_true has shape {data['velocities_true'].shape}, expected ({N}, 2)"
        )
    if data["velocities_est"].shape != (N, 2):
        raise ValueError(
            f"velocities_est has shape {data['velocities_est'].shape}, expected ({N}, 2)"
        )
    if data["headings_true"].shape != (N,):
        raise ValueError(
            f"headings_true has shape {data['headings_true'].shape}, expected ({N},)"
        )
    if data["headings_est"].shape != (N,):
        raise ValueError(
            f"headings_est has shape {data['headings_est'].shape}, expected ({N},)"
        )
    if data["nees"].shape != (N,):
        raise ValueError(f"nees has shape {data['nees'].shape}, expected ({N},)")

    # Load optional NIS if available. Shape must match the trajectory
    # length so the report's NIS summary aligns with the same frames as
    # positions/headings; otherwise consistency stats would be attached
    # to the wrong sample set.
    nis_path = run_dir / "nis.npy"
    if nis_path.exists():
        nis = np.load(nis_path)
        if nis.shape != (N,):
            raise ValueError(
                f"nis has shape {nis.shape}, expected ({N},) to match the "
                "trajectory time base."
            )
        data["nis"] = nis
        # Load measurement dimension if NIS exists. Same positive-integer
        # contract as state_dim — used as df for the NIS chi-squared bounds.
        meas_dim_path = run_dir / "measurement_dim.txt"
        if meas_dim_path.exists():
            meas_dim_text = meas_dim_path.read_text().strip()
            try:
                meas_dim = int(meas_dim_text)
            except ValueError as e:
                raise ValueError(
                    "measurement_dim.txt must contain an integer; got "
                    f"{meas_dim_text!r}."
                ) from e
            if meas_dim < 1:
                raise ValueError(
                    "measurement_dim must be a positive integer (degrees of "
                    "freedom for NIS chi-squared bounds); got "
                    f"{meas_dim} from measurement_dim.txt."
                )
            data["measurement_dim"] = meas_dim

    return data


@friendly_cli_errors
def run_report_command(args: argparse.Namespace) -> None:
    """Execute the report command.

    The :func:`friendly_cli_errors` decorator surfaces
    ``FileNotFoundError`` / ``NotADirectoryError`` / ``ValueError``
    from the loader and report-generation paths as ``Error: ...``
    stderr lines.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # Load data from run directory
    data = load_run_data(args.run)

    # Prepare report arguments
    report_args = {
        "pdf_path": args.pdf,
        "t": data["t"],
        "positions_true": data["positions_true"],
        "positions_est": data["positions_est"],
        "velocities_true": data["velocities_true"],
        "velocities_est": data["velocities_est"],
        "headings_true": data["headings_true"],
        "headings_est": data["headings_est"],
        "nees": data["nees"],
        "state_dim": data["state_dim"],
    }

    # Add optional parameters if available. ``measurement_dim`` is
    # required when NIS is present because chi-square consistency
    # bounds (and "% in bounds") depend on it: position-only NIS is
    # df=2, dual-LED NIS is df=4, and other layouts are possible.
    # Silently defaulting to 4 produced a successful exit code with
    # potentially-wrong bounds. Force the user to be explicit.
    if "nis" in data:
        report_args["nis"] = data["nis"]
        if "measurement_dim" not in data:
            raise ValueError(
                "nis.npy is present in the input directory but "
                "measurement_dim.txt is missing. The chi-square "
                "consistency bounds depend on the measurement "
                "dimensionality (df=2 for position-only, df=4 for "
                "dual-LED, etc.) and cannot be guessed from nis.npy. "
                "Add measurement_dim.txt with the correct integer "
                "(e.g. ``echo 4 > measurement_dim.txt``) or remove "
                "nis.npy from the input directory."
            )
        report_args["measurement_dim"] = data["measurement_dim"]

    if args.title is not None:
        report_args["title"] = args.title

    # Generate report
    generate_qa_report(**report_args)

    print(f"Report generated successfully: {args.pdf}")
