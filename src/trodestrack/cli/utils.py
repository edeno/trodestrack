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
