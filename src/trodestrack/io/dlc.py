"""DeepLabCut keypoint data loader."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings


class DLCKeypointData:
    """Container for DeepLabCut keypoint tracking data."""

    def __init__(
        self,
        timestamps: np.ndarray,
        keypoints: Dict[str, np.ndarray],
        confidences: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize DLC keypoint data container.

        Args:
            timestamps: Frame timestamps (seconds)
            keypoints: Dict mapping keypoint names to (N, 2) position arrays [x, y]
            confidences: Dict mapping keypoint names to (N,) confidence arrays
            metadata: Optional metadata dictionary
        """
        self.timestamps = np.asarray(timestamps)
        self.keypoints = {name: np.asarray(pos) for name, pos in keypoints.items()}
        self.confidences = {
            name: np.asarray(conf) for name, conf in confidences.items()
        }
        self.metadata = metadata or {}

        # Validate shapes
        n_frames = len(self.timestamps)
        for name in self.keypoints:
            if len(self.keypoints[name]) != n_frames:
                raise ValueError(f"Keypoint {name} length mismatch")
            if self.keypoints[name].shape != (n_frames, 2):
                raise ValueError(f"Keypoint {name} must be (N, 2) array")
            if len(self.confidences[name]) != n_frames:
                raise ValueError(f"Confidence {name} length mismatch")

    @property
    def n_frames(self) -> int:
        """Number of frames in the dataset."""
        return len(self.timestamps)

    @property
    def keypoint_names(self) -> List[str]:
        """List of keypoint names."""
        return list(self.keypoints.keys())

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

    def get_valid_frames(
        self,
        keypoint_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Get boolean mask for frames with sufficient keypoint confidence.

        Args:
            keypoint_names: Keypoints to check (default: all)
            confidence_threshold: Minimum confidence threshold

        Returns:
            Boolean array indicating valid frames
        """
        if keypoint_names is None:
            keypoint_names = self.keypoint_names

        valid = np.ones(self.n_frames, dtype=bool)
        for name in keypoint_names:
            if name in self.confidences:
                valid &= self.confidences[name] >= confidence_threshold

        return valid

    def get_keypoint(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get keypoint positions and confidences.

        Args:
            name: Keypoint name

        Returns:
            Tuple of (positions, confidences)
        """
        if name not in self.keypoints:
            raise KeyError(f"Keypoint '{name}' not found")

        return self.keypoints[name], self.confidences[name]

    def get_led_data(
        self, front_keypoint: str = "front", back_keypoint: str = "back"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract LED-like data from specific keypoints.

        Args:
            front_keypoint: Name of front LED keypoint
            back_keypoint: Name of back LED keypoint

        Returns:
            Tuple of (front_positions, back_positions, front_conf, back_conf)
        """
        if front_keypoint not in self.keypoints:
            raise KeyError(f"Front keypoint '{front_keypoint}' not found")
        if back_keypoint not in self.keypoints:
            raise KeyError(f"Back keypoint '{back_keypoint}' not found")

        front_pos = self.keypoints[front_keypoint]
        back_pos = self.keypoints[back_keypoint]
        front_conf = self.confidences[front_keypoint]
        back_conf = self.confidences[back_keypoint]

        return front_pos, back_pos, front_conf, back_conf

    def get_heading_from_keypoints(
        self, front_keypoint: str = "front", back_keypoint: str = "back"
    ) -> np.ndarray:
        """Get heading angle from keypoint vector.

        Args:
            front_keypoint: Name of front keypoint
            back_keypoint: Name of back keypoint

        Returns:
            Array of heading angles in radians
        """
        front_pos, back_pos, _, _ = self.get_led_data(front_keypoint, back_keypoint)
        led_vector = front_pos - back_pos
        return np.arctan2(led_vector[:, 1], led_vector[:, 0])

    def get_center_position(
        self, front_keypoint: str = "front", back_keypoint: str = "back"
    ) -> np.ndarray:
        """Get center position between two keypoints.

        Args:
            front_keypoint: Name of front keypoint
            back_keypoint: Name of back keypoint

        Returns:
            Array of shape (N, 2) with center positions [x, y]
        """
        front_pos, back_pos, _, _ = self.get_led_data(front_keypoint, back_keypoint)
        return (front_pos + back_pos) / 2


def load_dlc_csv(file_path: Path) -> DLCKeypointData:
    """Load DeepLabCut data from CSV file.

    Expected CSV format from DLC analysis:
    - MultiIndex columns: (scorer, bodypart, coords)
    - Where coords are 'x', 'y', 'likelihood'

    Args:
        file_path: Path to DLC CSV file

    Returns:
        DLCKeypointData container

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"DLC file not found: {file_path}")

    try:
        # Read CSV with multi-level header
        df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
    except Exception as e:
        raise ValueError(f"Failed to read DLC CSV file: {e}")

    # Get scorer name (first level of columns)
    scorer = df.columns.levels[0][0]

    # Extract bodyparts (second level)
    bodyparts = df.columns.levels[1]

    # Generate timestamps if not provided
    if df.index.name == "frame":
        # Assume frame numbers, need to convert to timestamps
        # This requires fps information - use metadata or assume 30 fps
        fps = 30.0  # Default assumption
        timestamps = df.index.values / fps
        warnings.warn(f"No timestamp column found, assuming {fps} fps")
    else:
        timestamps = df.index.values

    # Extract keypoint data
    keypoints = {}
    confidences = {}

    for bodypart in bodyparts:
        try:
            # Get x, y coordinates and likelihood
            x_col = (scorer, bodypart, "x")
            y_col = (scorer, bodypart, "y")
            likelihood_col = (scorer, bodypart, "likelihood")

            if all(col in df.columns for col in [x_col, y_col, likelihood_col]):
                x_vals = df[x_col].values
                y_vals = df[y_col].values
                likelihood_vals = df[likelihood_col].values

                # Combine x, y into position array
                positions = np.column_stack([x_vals, y_vals])

                keypoints[bodypart] = positions
                confidences[bodypart] = likelihood_vals

        except KeyError:
            warnings.warn(f"Could not extract data for bodypart: {bodypart}")
            continue

    if not keypoints:
        raise ValueError("No valid keypoints found in DLC file")

    # Create metadata
    metadata = {
        "file_path": str(file_path),
        "n_frames": len(df),
        "scorer": scorer,
        "bodyparts": list(bodyparts),
        "format": "dlc_csv",
    }

    return DLCKeypointData(
        timestamps=timestamps,
        keypoints=keypoints,
        confidences=confidences,
        metadata=metadata,
    )


def load_dlc_h5(file_path: Path) -> DLCKeypointData:
    """Load DeepLabCut data from HDF5 file.

    Args:
        file_path: Path to DLC HDF5 file

    Returns:
        DLCKeypointData container

    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If h5py not available
        ValueError: If HDF5 format is invalid
    """
    # h5py is required by pandas for HDF5 support
    import importlib.util
    if importlib.util.find_spec("h5py") is None:
        raise ImportError(
            "h5py required for HDF5 support. Install with: pip install h5py"
        )

    if not file_path.exists():
        raise FileNotFoundError(f"DLC file not found: {file_path}")

    try:
        # Load data using pandas HDFStore format
        df = pd.read_hdf(file_path)

        # Process similar to CSV but from HDF5
        if isinstance(df.columns, pd.MultiIndex):
            return _process_dlc_dataframe(df, file_path)
        else:
            raise ValueError("HDF5 file does not have expected MultiIndex columns")

    except Exception as e:
        raise ValueError(f"Failed to read DLC HDF5 file: {e}")


def load_dlc_pickle(file_path: Path) -> DLCKeypointData:
    """Load DeepLabCut data from pickle file.

    Args:
        file_path: Path to DLC pickle file

    Returns:
        DLCKeypointData container

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If pickle format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"DLC file not found: {file_path}")

    try:
        df = pd.read_pickle(file_path)
        return _process_dlc_dataframe(df, file_path)
    except Exception as e:
        raise ValueError(f"Failed to read DLC pickle file: {e}")


def _process_dlc_dataframe(df: pd.DataFrame, file_path: Path) -> DLCKeypointData:
    """Process DLC DataFrame with MultiIndex columns.

    Args:
        df: DLC DataFrame with MultiIndex columns
        file_path: Original file path for metadata

    Returns:
        DLCKeypointData container
    """
    # Get scorer name (first level of columns)
    scorer = df.columns.levels[0][0]

    # Extract bodyparts (second level)
    bodyparts = df.columns.levels[1]

    # Generate timestamps
    if df.index.name == "frame":
        fps = 30.0  # Default assumption
        timestamps = df.index.values / fps
        warnings.warn(f"No timestamp column found, assuming {fps} fps")
    else:
        timestamps = df.index.values

    # Extract keypoint data
    keypoints = {}
    confidences = {}

    for bodypart in bodyparts:
        try:
            # Get x, y coordinates and likelihood
            x_col = (scorer, bodypart, "x")
            y_col = (scorer, bodypart, "y")
            likelihood_col = (scorer, bodypart, "likelihood")

            if all(col in df.columns for col in [x_col, y_col, likelihood_col]):
                x_vals = df[x_col].values
                y_vals = df[y_col].values
                likelihood_vals = df[likelihood_col].values

                # Handle NaN values by setting confidence to 0
                nan_mask = np.isnan(x_vals) | np.isnan(y_vals)
                likelihood_vals[nan_mask] = 0.0

                # Set NaN positions to (0, 0)
                x_vals[np.isnan(x_vals)] = 0.0
                y_vals[np.isnan(y_vals)] = 0.0

                # Combine x, y into position array
                positions = np.column_stack([x_vals, y_vals])

                keypoints[bodypart] = positions
                confidences[bodypart] = likelihood_vals

        except KeyError:
            warnings.warn(f"Could not extract data for bodypart: {bodypart}")
            continue

    if not keypoints:
        raise ValueError("No valid keypoints found in DLC file")

    # Create metadata
    metadata = {
        "file_path": str(file_path),
        "n_frames": len(df),
        "scorer": scorer,
        "bodyparts": list(bodyparts),
        "format": "dlc_processed",
    }

    return DLCKeypointData(
        timestamps=timestamps,
        keypoints=keypoints,
        confidences=confidences,
        metadata=metadata,
    )


def load_dlc(file_path: Path) -> DLCKeypointData:
    """Load DeepLabCut data with automatic format detection.

    Args:
        file_path: Path to DLC file (.csv, .h5, or .pkl)

    Returns:
        DLCKeypointData container
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        return load_dlc_csv(file_path)
    elif file_path.suffix.lower() in [".h5", ".hdf5"]:
        return load_dlc_h5(file_path)
    elif file_path.suffix.lower() in [".pkl", ".pickle"]:
        return load_dlc_pickle(file_path)
    else:
        raise ValueError(f"Unsupported DLC file format: {file_path.suffix}")
