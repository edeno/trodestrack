"""Shared utility functions for CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load_data_file(
    path: Path, name: str, expected_shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """Load a data file with validation.

    Parameters
    ----------
    path : Path
        Path to data file.
    name : str
        Descriptive name for error messages.
    expected_shape : tuple[int, ...] | None, optional
        Expected shape (None to skip validation).

    Returns
    -------
    np.ndarray
        Loaded numpy array.

    Raises
    ------
    SystemExit
        If file doesn't exist or has wrong shape.

    Examples
    --------
    >>> from pathlib import Path
    >>> import numpy as np
    >>> # Load IMU timestamps (N,)
    >>> t_imu = load_data_file(Path("t_imu.txt"), "IMU timestamps")
    >>> # Load camera positions (N, 2) with shape validation
    >>> z_led1 = load_data_file(
    ...     Path("z_led1.txt"), "LED1 positions", expected_shape=(100, 2)
    ... )
    """
    if not path.exists():
        print(f"Error: {name} file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = np.loadtxt(path)
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if expected_shape is not None and data.shape != expected_shape:
        print(
            f"Error: {name} has shape {data.shape}, expected {expected_shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    return data


def validate_monotonic_timestamps(t: np.ndarray, name: str) -> None:
    """Reject timestamp arrays that are non-finite or not strictly increasing.

    The filter uses ``np.diff(t)`` to derive sample periods and feeds them
    into ``compute_imu_index_arrays`` and IMU pre-integration. Non-finite
    or decreasing timestamps would produce NaN/negative dt and silently
    poison the filter outputs.

    Exits with status 1 on failure so the CLI fails loudly rather than
    producing bad estimates downstream.

    Parameters
    ----------
    t : np.ndarray
        Timestamp array (N,) in seconds.
    name : str
        Descriptive name for error messages (e.g. "IMU timestamps").
    """
    if t.ndim != 1:
        print(
            f"Error: {name} must be 1D, got shape {t.shape}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not np.all(np.isfinite(t)):
        n_bad = int(np.sum(~np.isfinite(t)))
        print(
            f"Error: {name} contains {n_bad} non-finite value(s) (NaN/inf); "
            "timestamps must be finite seconds.",
            file=sys.stderr,
        )
        sys.exit(1)
    diffs = np.diff(t)
    if t.size >= 2 and not np.all(diffs > 0):
        first_bad = int(np.argmax(diffs <= 0))
        print(
            f"Error: {name} must be strictly increasing; first non-increasing "
            f"step at index {first_bad + 1} (t[{first_bad}]={t[first_bad]!r}, "
            f"t[{first_bad + 1}]={t[first_bad + 1]!r}, dt={diffs[first_bad]!r}).",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_finite_array(a: np.ndarray, name: str) -> None:
    """Reject arrays that contain NaN or inf.

    The IMU integration path does not have a per-sample mask (unlike the
    camera path, which uses ``mask_cam``). A single NaN/inf row in
    ``U_imu`` propagates through ``np.diff`` / pre-integration and
    contaminates every downstream state estimate. This guard makes that
    failure loud at CLI entry.

    Exits with status 1 on failure.

    Parameters
    ----------
    a : np.ndarray
        Array to check.
    name : str
        Descriptive name for error messages (e.g. "IMU measurements").
    """
    if not np.all(np.isfinite(a)):
        bad_mask = ~np.isfinite(a)
        n_bad = int(bad_mask.sum())
        # Report the first offending row to help users locate the problem.
        if a.ndim == 1:
            first_bad = int(np.argmax(bad_mask))
        else:
            first_bad = int(np.argmax(bad_mask.any(axis=tuple(range(1, a.ndim)))))
        print(
            f"Error: {name} contains {n_bad} non-finite value(s) (NaN/inf); "
            f"first offending row at index {first_bad}. The CLI does not "
            "support partial-sample masking for this input.",
            file=sys.stderr,
        )
        sys.exit(1)
