"""CLI command for generating QA reports from QA-prepared run directories.

This module provides the 'trodestrack report' command that reads a directory
of QA-prepared arrays (paired ground-truth and estimated trajectories plus
NEES/NIS series) and generates a comprehensive PDF report.

It also exposes the ``--from-run`` mode that bridges filter/smooth output
directories (``filtered_means.txt`` / ``smoothed_means.txt`` plus
``metadata.txt``) to the report's QA input contract, given ground-truth
position and heading arrays from the user.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trodestrack.cli.utils import friendly_cli_errors
from trodestrack.models.state_layout import (
    LAYOUT_REGISTRY,
    StateLayout,
    get_layout,
)
from trodestrack.qa.metrics import compute_nees
from trodestrack.qa.report import generate_qa_report

# Fallback when ``metadata.txt`` is missing or does not record ``State mode``.
# Matches the ``filter`` / ``smooth`` default in cli/_defaults.py and
# applies to the run directories those subcommands produce.
_DEFAULT_STATE_MODE = "2d_cam_3d_imu"


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
            "and NEES/NIS series. To consume a filter/smooth output "
            "directory directly, use --from-run with --ground-truth-positions "
            "and --ground-truth-headings."
        ),
    )

    parser.add_argument(
        "--run",
        type=Path,
        required=False,
        help="Path to QA-prepared run directory (paired ground-truth and "
        "estimated arrays plus NEES/NIS .npy files; see command description). "
        "Mutually exclusive with --from-run.",
        metavar="DIR",
    )

    parser.add_argument(
        "--from-run",
        type=Path,
        required=False,
        help="Path to a trodestrack filter/smooth output directory. "
        "Requires --ground-truth-positions and --ground-truth-headings.",
        metavar="DIR",
    )

    parser.add_argument(
        "--ground-truth-positions",
        type=Path,
        required=False,
        help="Path to ground-truth positions .npy or .txt file (N, 2). "
        "Required when --from-run is used.",
        metavar="FILE",
    )

    parser.add_argument(
        "--ground-truth-headings",
        type=Path,
        required=False,
        help="Path to ground-truth headings .npy or .txt file (N,). "
        "Required when --from-run is used.",
        metavar="FILE",
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


def _load_array_auto(path: Path, name: str) -> NDArray:
    """Load a ground-truth array, auto-detecting ``.npy`` vs ``.txt``.

    Parameters
    ----------
    path : Path
        Path to the ground-truth file. ``.npy`` is loaded with
        :func:`numpy.load`; anything else is loaded with
        :func:`numpy.loadtxt`.
    name : str
        Human-readable name used in the error message.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return np.loadtxt(path)


def _read_state_mode_from_metadata(run_dir: Path) -> str:
    """Parse ``metadata.txt`` for the ``State mode:`` line.

    Returns
    -------
    str
        The recorded state mode (e.g. ``"2d_cam_3d_imu"``) when found.
        Falls back to :data:`_DEFAULT_STATE_MODE` when ``metadata.txt``
        is absent or does not include a ``State mode:`` line.
    """
    metadata_path = run_dir / "metadata.txt"
    if not metadata_path.exists():
        return _DEFAULT_STATE_MODE
    text = metadata_path.read_text()
    match = re.search(r"State mode:\s*(\S+)", text)
    if match is None:
        return _DEFAULT_STATE_MODE
    return match.group(1)


def _build_qa_inputs_from_run(
    run_dir: Path,
    gt_pos_path: Path,
    gt_head_path: Path,
    state_layout: StateLayout | None = None,
) -> dict[str, NDArray | int]:
    """Synthesize the ``load_run_data`` return dict from a filter/smooth run.

    Reads ``smoothed_means.txt`` (preferred) or ``filtered_means.txt`` from
    ``run_dir`` along with the matching ``*_covariances.txt`` (flat
    ``(N, D*D)`` layout), uses ``state_layout`` to extract the position,
    velocity, and 2D-heading sub-state, loads ground-truth positions and
    headings from the user-supplied files, and computes a NEES series
    against the ``(x, y, θ)`` sub-state.

    Parameters
    ----------
    run_dir : Path
        Directory written by ``trodestrack filter`` / ``trodestrack smooth``.
    gt_pos_path : Path
        Ground-truth positions file (``.npy`` or ``.txt``); shape ``(N, 2)``.
    gt_head_path : Path
        Ground-truth headings file (``.npy`` or ``.txt``); shape ``(N,)``.
    state_layout : StateLayout or None, optional
        Layout describing how to index the state vector. When ``None``,
        the layout is read from ``run_dir/metadata.txt`` (or falls back
        to the ``2d_cam_3d_imu`` default).

    Returns
    -------
    dict
        Same shape as :func:`load_run_data`: keys ``t``, ``positions_true``,
        ``positions_est``, ``velocities_true``, ``velocities_est``,
        ``headings_true``, ``headings_est``, ``nees``, ``state_dim``.

    Raises
    ------
    FileNotFoundError
        If ``run_dir`` is missing, or no ``smoothed_means.txt`` /
        ``filtered_means.txt`` exists inside it, or the covariance file
        next to the means is missing, or the ground-truth files are
        absent.
    ValueError
        If the layout does not use a 2D scalar heading, or the loaded
        arrays disagree on length / shape.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {run_dir}")

    if state_layout is None:
        mode = _read_state_mode_from_metadata(run_dir)
        if mode not in LAYOUT_REGISTRY:
            raise ValueError(
                f"State mode {mode!r} from {run_dir}/metadata.txt is not a "
                f"registered layout. Known modes: {sorted(LAYOUT_REGISTRY)}."
            )
        state_layout = get_layout(mode)

    if not state_layout.has_heading_2d:
        raise ValueError(
            "--from-run currently supports only 2D-heading state layouts "
            f"(got {state_layout.n}D state with non-scalar heading_idx="
            f"{state_layout.heading_idx!r}). For 3D-orientation runs, "
            "construct a qa_inputs/ directory manually and use --run."
        )

    smoothed = run_dir / "smoothed_means.txt"
    filtered = run_dir / "filtered_means.txt"
    if smoothed.exists():
        means_path = smoothed
        cov_path = run_dir / "smoothed_covariances.txt"
    elif filtered.exists():
        means_path = filtered
        cov_path = run_dir / "filtered_covariances.txt"
    else:
        raise FileNotFoundError(
            f"Neither smoothed_means.txt nor filtered_means.txt found in "
            f"{run_dir}. Run `trodestrack filter` or `trodestrack smooth` "
            "first, or pass --run with a QA-prepared directory."
        )
    if not cov_path.exists():
        raise FileNotFoundError(
            f"{cov_path.name} not found in {run_dir} alongside "
            f"{means_path.name}. NEES requires the covariance series."
        )

    means = np.loadtxt(means_path)
    if means.ndim != 2:
        raise ValueError(
            f"{means_path.name} must be 2D (N, D); got shape {means.shape}."
        )
    n_cam, n_state = means.shape
    if n_state != state_layout.n:
        raise ValueError(
            f"{means_path.name} has state dimension {n_state}, but layout "
            f"{state_layout.n}D was selected. Pass a matching --state-mode "
            "or correct metadata.txt."
        )

    flat_covs = np.loadtxt(cov_path)
    if flat_covs.shape != (n_cam, n_state * n_state):
        raise ValueError(
            f"{cov_path.name} has shape {flat_covs.shape}, expected "
            f"({n_cam}, {n_state * n_state}) to match the means."
        )
    covs = flat_covs.reshape(n_cam, n_state, n_state)

    pos_idx = np.asarray(state_layout.pos_idx, dtype=int)
    # For 3D-IMU layouts, vel_idx includes vz; the QA report consumes
    # 2D velocities, so take the first two components (vx, vy).
    vel_idx = np.asarray(state_layout.vel_idx[:2], dtype=int)
    heading_idx = int(state_layout.heading_idx)  # type: ignore[arg-type]

    positions_est = means[:, pos_idx]
    velocities_est = means[:, vel_idx]
    headings_est = means[:, heading_idx]

    positions_true = np.asarray(
        _load_array_auto(gt_pos_path, "Ground-truth positions"), dtype=float
    )
    headings_true = np.asarray(
        _load_array_auto(gt_head_path, "Ground-truth headings"), dtype=float
    )

    if positions_true.shape != (n_cam, 2):
        raise ValueError(
            f"Ground-truth positions has shape {positions_true.shape}, "
            f"expected ({n_cam}, 2) to match {means_path.name}."
        )
    if headings_true.shape != (n_cam,):
        raise ValueError(
            f"Ground-truth headings has shape {headings_true.shape}, "
            f"expected ({n_cam},) to match {means_path.name}."
        )

    # Synthesize timestamps. The filter/smooth output does not record the
    # camera timestamps next to the means, so reconstruct a uniform axis
    # from the row count — the QA report uses t only as a monotonic plot
    # axis, not for any time-derivative computation.
    t = np.arange(n_cam, dtype=float)

    # Ground-truth velocities default to the estimate (no separate gt
    # source available here). The velocity-RMSE figure will then read as
    # zero, but the velocity panel still renders for inspection. Users
    # who want true velocity comparison should prepare a qa_inputs/
    # directory with separate velocities_true.npy and use --run.
    velocities_true = np.asarray(velocities_est, dtype=float).copy()

    # NEES on the (x, y, θ) sub-state. Build a sub-covariance by
    # selecting the matching rows/columns; angle wrapping on the
    # heading residual is handled inside compute_nees via the
    # explicit heading-idx kwarg (third sub-state dimension).
    sub_idx = np.concatenate([pos_idx, np.array([heading_idx], dtype=int)])
    sub_states_est = means[:, sub_idx]
    sub_states_true = np.column_stack([positions_true, headings_true])
    sub_covs = covs[:, sub_idx[:, None], sub_idx[None, :]]
    nees = compute_nees(
        sub_states_true,
        sub_states_est,
        sub_covs,
        heading_idx=2,
    )

    return {
        "t": t,
        "positions_true": positions_true,
        "positions_est": np.asarray(positions_est, dtype=float),
        "velocities_true": velocities_true,
        "velocities_est": np.asarray(velocities_est, dtype=float),
        "headings_true": headings_true,
        "headings_est": np.asarray(headings_est, dtype=float),
        "nees": np.asarray(nees, dtype=float),
        "state_dim": 3,
    }


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
    from_run = getattr(args, "from_run", None)
    run = getattr(args, "run", None)

    if from_run is not None and run is not None:
        raise ValueError(
            "--run and --from-run are mutually exclusive. Pass --run for a "
            "QA-prepared directory, or --from-run for a filter/smooth "
            "output directory plus --ground-truth-positions / "
            "--ground-truth-headings."
        )
    if from_run is None and run is None:
        raise ValueError(
            "Either --run (QA-prepared directory) or --from-run "
            "(filter/smooth output directory + ground-truth files) is "
            "required."
        )

    if from_run is not None:
        gt_pos = getattr(args, "ground_truth_positions", None)
        gt_head = getattr(args, "ground_truth_headings", None)
        missing = []
        if gt_pos is None:
            missing.append("--ground-truth-positions")
        if gt_head is None:
            missing.append("--ground-truth-headings")
        if missing:
            raise ValueError(
                f"--from-run requires {' and '.join(missing)}. The bridge "
                "needs ground-truth arrays to compute NEES and accuracy "
                "metrics against the filter/smooth estimates."
            )
        data = _build_qa_inputs_from_run(from_run, gt_pos, gt_head)
    else:
        # Load data from QA-prepared run directory
        data = load_run_data(run)

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
