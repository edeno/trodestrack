"""Trodes LED output data loader."""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class TrodesLEDData:
    """Container for Trodes LED tracking data."""

    def __init__(
        self,
        timestamps: np.ndarray,
        front_led: np.ndarray,
        back_led: np.ndarray,
        front_confidence: np.ndarray,
        back_confidence: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Trodes LED data container.

        Args:
            timestamps: Frame timestamps (seconds)
            front_led: Front LED positions (N, 2) array [x, y] in pixels
            back_led: Back LED positions (N, 2) array [x, y] in pixels
            front_confidence: Front LED detection confidence [0-1]
            back_confidence: Back LED detection confidence [0-1]
            metadata: Optional metadata dictionary
        """
        self.timestamps = np.asarray(timestamps)
        self.front_led = np.asarray(front_led)
        self.back_led = np.asarray(back_led)
        self.front_confidence = np.asarray(front_confidence)
        self.back_confidence = np.asarray(back_confidence)
        self.metadata = metadata or {}

        # Validate shapes
        n_frames = len(self.timestamps)
        if not all(
            len(arr) == n_frames
            for arr in [
                self.front_led,
                self.back_led,
                self.front_confidence,
                self.back_confidence,
            ]
        ):
            raise ValueError("All arrays must have the same length")

        if self.front_led.shape != (n_frames, 2):
            raise ValueError("LED position arrays must be (N, 2)")
        if self.back_led.shape != (n_frames, 2):
            raise ValueError("LED position arrays must be (N, 2)")

    @property
    def n_frames(self) -> int:
        """Number of frames in the dataset."""
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        """Duration of the recording in seconds."""
        if self.n_frames == 0:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def fps(self) -> float:
        """Estimated frames per second."""
        if self.n_frames <= 1:
            return 0.0
        return (self.n_frames - 1) / self.duration

    def get_valid_frames(self, confidence_threshold: float = 0.5) -> np.ndarray:
        """Get boolean mask for frames with sufficient LED confidence.

        Args:
            confidence_threshold: Minimum confidence for both LEDs

        Returns:
            Boolean array indicating valid frames
        """
        return (self.front_confidence >= confidence_threshold) & (
            self.back_confidence >= confidence_threshold
        )

    def get_led_vector(self) -> np.ndarray:
        """Get vector from back LED to front LED.

        Returns:
            Array of shape (N, 2) with LED vectors [dx, dy]
        """
        return self.front_led - self.back_led

    def get_heading(self) -> np.ndarray:
        """Get heading angle from LED vector.

        Returns:
            Array of heading angles in radians
        """
        led_vector = self.get_led_vector()
        return np.arctan2(led_vector[:, 1], led_vector[:, 0])

    def get_position(self) -> np.ndarray:
        """Get center position between LEDs.

        Returns:
            Array of shape (N, 2) with center positions [x, y]
        """
        return (self.front_led + self.back_led) / 2


def load_trodes_led_csv(file_path: Path) -> TrodesLEDData:
    """Load Trodes LED tracking data from CSV file.

    Expected CSV format:
    - timestamp, front_x, front_y, front_conf, back_x, back_y, back_conf
    - OR timestamp, front_x, front_y, back_x, back_y (confidence defaults to 1.0)

    Args:
        file_path: Path to Trodes LED CSV file

    Returns:
        TrodesLEDData container

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Trodes LED file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Check for required columns
    required_cols = ["timestamp", "front_x", "front_y", "back_x", "back_y"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract data
    timestamps = df["timestamp"].values
    front_led = df[["front_x", "front_y"]].values
    back_led = df[["back_x", "back_y"]].values

    # Handle confidence columns
    if "front_conf" in df.columns and "back_conf" in df.columns:
        front_confidence = df["front_conf"].values
        back_confidence = df["back_conf"].values
    else:
        # Default confidence to 1.0 if not provided
        front_confidence = np.ones(len(df))
        back_confidence = np.ones(len(df))
        warnings.warn("No confidence columns found, defaulting to 1.0")

    # Handle NaN values by setting confidence to 0
    front_nan_mask = np.isnan(front_led).any(axis=1)
    back_nan_mask = np.isnan(back_led).any(axis=1)

    front_confidence[front_nan_mask] = 0.0
    back_confidence[back_nan_mask] = 0.0

    # Set NaN positions to (0, 0) for easier handling
    front_led[front_nan_mask] = 0.0
    back_led[back_nan_mask] = 0.0

    # Create metadata
    metadata = {
        "file_path": str(file_path),
        "n_frames": len(df),
        "columns": list(df.columns),
        "has_confidence": "front_conf" in df.columns,
    }

    return TrodesLEDData(
        timestamps=timestamps,
        front_led=front_led,
        back_led=back_led,
        front_confidence=front_confidence,
        back_confidence=back_confidence,
        metadata=metadata,
    )


def load_trodes_led_h5(file_path: Path) -> TrodesLEDData:
    """Load Trodes LED tracking data from HDF5 file.

    Args:
        file_path: Path to Trodes LED HDF5 file

    Returns:
        TrodesLEDData container

    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If h5py not available
        ValueError: If HDF5 format is invalid
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 support. Install with: pip install h5py")

    if not file_path.exists():
        raise FileNotFoundError(f"Trodes LED file not found: {file_path}")

    try:
        with h5py.File(file_path, "r") as f:
            # Expected structure: /led_tracking/{timestamps, front_led, back_led, confidence}
            if "led_tracking" not in f:
                raise ValueError("HDF5 file missing 'led_tracking' group")

            group = f["led_tracking"]

            timestamps = group["timestamps"][:]
            front_led = group["front_led"][:]
            back_led = group["back_led"][:]

            # Handle optional confidence data
            if "front_conf" in group and "back_conf" in group:
                front_confidence = group["front_conf"][:]
                back_confidence = group["back_conf"][:]
            else:
                front_confidence = np.ones(len(timestamps))
                back_confidence = np.ones(len(timestamps))
                warnings.warn("No confidence data found in HDF5, defaulting to 1.0")

            # Create metadata
            metadata = {
                "file_path": str(file_path),
                "n_frames": len(timestamps),
                "format": "hdf5",
                "has_confidence": "front_conf" in group,
            }

            return TrodesLEDData(
                timestamps=timestamps,
                front_led=front_led,
                back_led=back_led,
                front_confidence=front_confidence,
                back_confidence=back_confidence,
                metadata=metadata,
            )

    except Exception as e:
        raise ValueError(f"Failed to read HDF5 file: {e}")


def load_trodes_led(file_path: Path) -> TrodesLEDData:
    """Load Trodes LED data with automatic format detection.

    Args:
        file_path: Path to Trodes LED file (.csv or .h5)

    Returns:
        TrodesLEDData container
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        return load_trodes_led_csv(file_path)
    elif file_path.suffix.lower() in [".h5", ".hdf5"]:
        return load_trodes_led_h5(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
