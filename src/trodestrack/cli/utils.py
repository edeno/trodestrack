"""Shared utility functions for CLI commands."""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., None])


def friendly_cli_errors(func: F) -> F:
    """Convert library exceptions into ``Error: <msg>`` + ``sys.exit(1)``.

    Wraps a CLI subcommand body so that ``FileNotFoundError`` /
    ``NotADirectoryError`` / ``ValueError`` (and any other unexpected
    ``Exception``) raised by downstream library code surface as a
    clean stderr line rather than a Python traceback. The traceback
    still appears for ``KeyboardInterrupt`` / ``SystemExit`` so users
    can interrupt a long run normally and inline ``sys.exit(1)``
    callers (e.g. shape checks in ``cli/utils.load_data_file``) keep
    their own error path.

    Notes
    -----
    Generic ``Exception`` is caught so unexpected failures (e.g. a
    bug in a downstream module) still exit with a friendly
    ``Unexpected error (<ExceptionClass>): ...`` line. The exception
    class name is included so users filing bug reports can identify
    whether the failure was a ``KeyError``, ``AttributeError``, etc.

    Set ``TRODESTRACK_DEBUG=1`` (or ``true``/``yes``) in the
    environment to re-raise exceptions with a full Python traceback
    instead of converting them to a stderr line and exiting. The
    "Unexpected error" message also surfaces this hint so users hit
    by an unfamiliar failure know how to get the traceback.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        debug = os.environ.get("TRODESTRACK_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            return func(*args, **kwargs)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            if debug:
                raise
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            if debug:
                raise
            print(
                f"Unexpected error ({type(e).__name__}): {e}\n"
                "Re-run with TRODESTRACK_DEBUG=1 for a full traceback.",
                file=sys.stderr,
            )
            sys.exit(1)

    return wrapper  # type: ignore[return-value]


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
        print(
            f"Error loading {name} from {path}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
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
    # Require at least two samples — downstream code derives the sample
    # period via ``np.diff(t)`` (mean used for rate reporting) and
    # indexes ``t[-1] - t[0]`` for the session duration. A 0- or 1-sample
    # stream silently produces ``NaN`` rates / raw IndexError instead of
    # a clear CLI error.
    if t.size < 2:
        print(
            f"Error: {name} must contain at least two samples; got "
            f"size {t.size}. The filter derives the sample period from "
            f"``np.diff(t)`` and the session duration from ``t[-1] - t[0]``.",
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
    if not np.all(diffs > 0):
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


def require_cli_inputs(args: object, names: tuple[str, ...], *, command: str) -> None:
    """Require legacy per-file CLI args when ``--config`` is not used."""

    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        print(
            f"Error: trodestrack {command} requires either --config or all "
            f"required input flags. Missing: {flags}.",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_camera_mask(mask_raw: np.ndarray, n_cam: int) -> np.ndarray:
    """Validate a CLI camera mask and return a boolean array."""

    if mask_raw.shape != (n_cam,):
        print(
            f"Error: Camera mask shape {mask_raw.shape} doesn't match (n_cam={n_cam},)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not np.all(np.isfinite(mask_raw)):
        print(
            "Error: --camera-mask contains non-finite values (NaN/inf); "
            "expected only 0 or 1.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not np.all(np.isin(mask_raw, (0, 1))):
        bad = mask_raw[~np.isin(mask_raw, (0, 1))]
        print(
            f"Error: --camera-mask must contain only 0 or 1; found "
            f"{len(bad)} other value(s) (e.g. {bad[:5].tolist()}).",
            file=sys.stderr,
        )
        sys.exit(1)
    return mask_raw.astype(bool)
