"""CLI command for generating QA reports from filter results.

This module provides the 'trodestrack report' command that reads filter outputs
from a run directory and generates a comprehensive PDF report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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
        help="Generate PDF quality assurance report from filter results",
        description=(
            "Generate a comprehensive PDF report with accuracy metrics, "
            "consistency checks, and diagnostic plots from filter results."
        ),
    )

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to run directory containing filter results",
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

    # Load state dimension
    state_dim_text = (run_dir / "state_dim.txt").read_text().strip()
    data["state_dim"] = int(state_dim_text)

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

    # Load optional NIS if available
    nis_path = run_dir / "nis.npy"
    if nis_path.exists():
        data["nis"] = np.load(nis_path)
        # Load measurement dimension if NIS exists
        meas_dim_path = run_dir / "measurement_dim.txt"
        if meas_dim_path.exists():
            meas_dim_text = meas_dim_path.read_text().strip()
            data["measurement_dim"] = int(meas_dim_text)

    return data


def run_report_command(args: argparse.Namespace) -> None:
    """Execute the report command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    try:
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

        # Add optional parameters if available
        if "nis" in data:
            report_args["nis"] = data["nis"]
            if "measurement_dim" not in data:
                print(
                    "Warning: nis.npy found but measurement_dim.txt missing, "
                    "defaulting to measurement_dim=4",
                    file=sys.stderr,
                )
            report_args["measurement_dim"] = data.get("measurement_dim", 4)

        if args.title is not None:
            report_args["title"] = args.title

        # Generate report
        generate_qa_report(**report_args)

        print(f"Report generated successfully: {args.pdf}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotADirectoryError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
