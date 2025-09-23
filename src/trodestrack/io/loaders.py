"""Unified data loaders for trodestrack runtime APIs.

This module provides high-level loader functions that abstract away the details
of different data formats and return standardized dictionaries for use by
the runtime APIs.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def load_video_detections(file_path: Path) -> Dict[str, Any]:
    """Load video detection data from various formats.

    Args:
        file_path: Path to video detection file

    Returns:
        Dictionary with standardized video data:
        - 'timestamps': Frame timestamps (N,) array
        - 'positions': Position data (N, 2) array [x, y]
        - 'confidences': Detection confidence (N,) array [0-1]
        - 'headings': Optional heading data (N,) array (None if not available)
        - 'metadata': Optional metadata dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not recognized or data invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Video detection file not found: {file_path}")

    logger.info(f"Loading video detections from: {file_path}")

    # Determine format from file extension
    if file_path.suffix.lower() == ".npz":
        return _load_video_npz(file_path)
    elif file_path.suffix.lower() in [".csv", ".txt"]:
        return _load_video_csv(file_path)
    elif file_path.suffix.lower() == ".h5":
        return _load_video_dlc_h5(file_path)
    else:
        # Try to auto-detect format
        logger.warning(
            f"Unknown video file extension {file_path.suffix}, attempting auto-detection"
        )
        return _auto_detect_video_format(file_path)


def load_imu_data(file_path: Path) -> Dict[str, Any]:
    """Load IMU data from various formats.

    Args:
        file_path: Path to IMU data file

    Returns:
        Dictionary with standardized IMU data:
        - 'timestamps': Sample timestamps (N,) array
        - 'data': IMU measurements (N, 6) array [ax, ay, az, gx, gy, gz]
        - 'sampling_rate': Sampling rate in Hz
        - 'metadata': Optional metadata dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not recognized or data invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"IMU data file not found: {file_path}")

    logger.info(f"Loading IMU data from: {file_path}")

    # Determine format from file extension
    if file_path.suffix.lower() == ".npz":
        return _load_imu_npz(file_path)
    elif file_path.suffix.lower() in [".csv", ".txt"]:
        return _load_imu_csv(file_path)
    elif file_path.suffix.lower() == ".rec":
        return _load_imu_spikegadgets(file_path)
    else:
        # Try to auto-detect format
        logger.warning(f"Unknown IMU file extension {file_path.suffix}, attempting auto-detection")
        return _auto_detect_imu_format(file_path)


def _load_video_npz(file_path: Path) -> Dict[str, Any]:
    """Load video data from NPZ format."""
    data = np.load(file_path)

    required_keys = ["timestamps", "positions"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in video NPZ file")

    # Convert to JAX arrays for better performance
    result = {
        "timestamps": jnp.array(data["timestamps"]),
        "positions": jnp.array(data["positions"]),
        "confidences": jnp.array(data.get("confidences", np.ones(len(data["timestamps"])))),
        "headings": jnp.array(data["headings"]) if data.get("headings") is not None else None,
        "metadata": {"format": "npz", "source_file": str(file_path)},
    }

    # Validate data shapes
    n_frames = len(result["timestamps"])
    if result["positions"].shape != (n_frames, 2):
        raise ValueError(
            f"Invalid positions shape: expected ({n_frames}, 2), got {result['positions'].shape}"
        )

    return result


def _load_video_csv(file_path: Path) -> Dict[str, Any]:
    """Load video data from CSV format."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required to load CSV files. Install with: uv add pandas")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}")

    # Try to detect column structure
    timestamp_col = None
    x_col = None
    y_col = None
    confidence_col = None
    heading_col = None

    # Common column name patterns
    for col in df.columns:
        col_lower = col.lower()
        if "time" in col_lower or "frame" in col_lower:
            timestamp_col = col
        elif col_lower in ["x", "pos_x", "position_x"]:
            x_col = col
        elif col_lower in ["y", "pos_y", "position_y"]:
            y_col = col
        elif "confidence" in col_lower or "likelihood" in col_lower:
            confidence_col = col
        elif "heading" in col_lower or "angle" in col_lower or "theta" in col_lower:
            heading_col = col

    if timestamp_col is None:
        # Use frame index as timestamp
        timestamps = np.arange(len(df)) / 30.0  # Assume 30 FPS
        logger.warning("No timestamp column found, using frame index at 30 FPS")
    else:
        timestamps = df[timestamp_col].values

    if x_col is None or y_col is None:
        raise ValueError("Could not find x and y position columns in CSV")

    positions = np.column_stack([df[x_col].values, df[y_col].values])
    confidences = df[confidence_col].values if confidence_col else np.ones(len(df))
    headings = df[heading_col].values if heading_col else None

    # Convert to JAX arrays for better performance
    return {
        "timestamps": jnp.array(timestamps),
        "positions": jnp.array(positions),
        "confidences": jnp.array(confidences),
        "headings": jnp.array(headings) if headings is not None else None,
        "metadata": {"format": "csv", "source_file": str(file_path)},
    }


def _load_video_dlc_h5(file_path: Path) -> Dict[str, Any]:
    """Load video data from DeepLabCut H5 format."""
    from .dlc import load_dlc_h5

    dlc_data = load_dlc_h5(file_path)

    # Convert DLC data to standard format
    # Use first available keypoint as position
    keypoint_names = list(dlc_data.keypoints.keys())
    if not keypoint_names:
        raise ValueError("No keypoints found in DLC file")

    primary_keypoint = keypoint_names[0]
    logger.info(f"Using keypoint '{primary_keypoint}' as primary position")

    # Convert to JAX arrays for better performance
    return {
        "timestamps": jnp.array(dlc_data.timestamps),
        "positions": jnp.array(dlc_data.keypoints[primary_keypoint]),
        "confidences": jnp.array(dlc_data.confidences[primary_keypoint]),
        "headings": None,  # Could compute from multiple keypoints if available
        "metadata": {
            "format": "dlc_h5",
            "source_file": str(file_path),
            "keypoint_names": keypoint_names,
            "primary_keypoint": primary_keypoint,
        },
    }


def _load_imu_npz(file_path: Path) -> Dict[str, Any]:
    """Load IMU data from NPZ format."""
    data = np.load(file_path)

    required_keys = ["timestamps", "data"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in IMU NPZ file")

    # Get sampling rate
    if "sampling_rate" in data:
        sampling_rate = float(data["sampling_rate"])
    else:
        # Estimate from timestamps
        dt = np.median(np.diff(data["timestamps"]))
        sampling_rate = 1.0 / dt
        logger.warning(
            "No sampling_rate in NPZ file, estimated %.1f Hz from timestamps",
            sampling_rate
        )

    imu_data = data["data"]
    if imu_data.shape[1] != 6:
        raise ValueError(
            f"IMU data must have 6 columns [ax, ay, az, gx, gy, gz], got {imu_data.shape[1]}"
        )

    # Convert to JAX arrays for better performance
    return {
        "timestamps": jnp.array(data["timestamps"]),
        "data": jnp.array(imu_data),
        "sampling_rate": sampling_rate,
        "metadata": {"format": "npz", "source_file": str(file_path)},
    }


def _load_imu_csv(file_path: Path) -> Dict[str, Any]:
    """Load IMU data from CSV format."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required to load CSV files. Install with: uv add pandas")

    df = pd.read_csv(file_path)

    # Look for standard IMU columns
    timestamp_col = None
    accel_cols = []
    gyro_cols = []

    for col in df.columns:
        col_lower = col.lower()
        if "time" in col_lower:
            timestamp_col = col
        elif any(x in col_lower for x in ["accel_x", "ax", "acc_x"]):
            accel_cols.append(col)
        elif any(x in col_lower for x in ["accel_y", "ay", "acc_y"]):
            accel_cols.append(col)
        elif any(x in col_lower for x in ["accel_z", "az", "acc_z"]):
            accel_cols.append(col)
        elif any(x in col_lower for x in ["gyro_x", "gx", "gyr_x"]):
            gyro_cols.append(col)
        elif any(x in col_lower for x in ["gyro_y", "gy", "gyr_y"]):
            gyro_cols.append(col)
        elif any(x in col_lower for x in ["gyro_z", "gz", "gyr_z"]):
            gyro_cols.append(col)

    if len(accel_cols) != 3 or len(gyro_cols) != 3:
        raise ValueError("Could not find 3 accelerometer and 3 gyroscope columns in CSV")

    timestamps = df[timestamp_col].values if timestamp_col else np.arange(len(df)) / 1000.0

    # Combine IMU data
    imu_data = np.column_stack(
        [
            df[accel_cols[0]].values,  # ax
            df[accel_cols[1]].values,  # ay
            df[accel_cols[2]].values,  # az
            df[gyro_cols[0]].values,  # gx
            df[gyro_cols[1]].values,  # gy
            df[gyro_cols[2]].values,  # gz
        ]
    )

    # Estimate sampling rate
    dt = np.median(np.diff(timestamps))
    sampling_rate = 1.0 / dt

    # Convert to JAX arrays for better performance
    return {
        "timestamps": jnp.array(timestamps),
        "data": jnp.array(imu_data),
        "sampling_rate": sampling_rate,
        "metadata": {"format": "csv", "source_file": str(file_path)},
    }


def _load_imu_spikegadgets(file_path: Path) -> Dict[str, Any]:
    """Load IMU data from SpikeGadgets format."""
    from .spikegadgets import load_spikegadgets

    imu_data = load_spikegadgets(file_path)

    # Convert to standard format
    accel_ms2 = imu_data.get_accel_ms2()
    gyro_rad_s = imu_data.get_gyro_rad_s()

    # Combine into 6-column format
    combined_data = np.column_stack([accel_ms2, gyro_rad_s])

    # Convert to JAX arrays for better performance
    return {
        "timestamps": jnp.array(imu_data.timestamps),
        "data": jnp.array(combined_data),
        "sampling_rate": imu_data.sampling_rate,
        "metadata": {
            "format": "spikegadgets",
            "source_file": str(file_path),
            "original_sampling_rate": imu_data.sampling_rate,
        },
    }


def _auto_detect_video_format(file_path: Path) -> Dict[str, Any]:
    """Attempt to auto-detect video format."""
    # Try formats in order of likelihood
    for loader in [_load_video_npz, _load_video_csv]:
        try:
            return loader(file_path)
        except Exception as e:
            logger.debug(f"Failed to load with {loader.__name__}: {e}")
            continue

    raise ValueError(f"Could not auto-detect format for video file: {file_path}")


def _auto_detect_imu_format(file_path: Path) -> Dict[str, Any]:
    """Attempt to auto-detect IMU format."""
    # Try formats in order of likelihood
    for loader in [_load_imu_npz, _load_imu_csv]:
        try:
            return loader(file_path)
        except Exception as e:
            logger.debug(f"Failed to load with {loader.__name__}: {e}")
            continue

    raise ValueError(f"Could not auto-detect format for IMU file: {file_path}")


def validate_video_data(data: Dict[str, Any]) -> None:
    """Validate loaded video data structure."""
    required_keys = ["timestamps", "positions", "confidences"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    n_frames = len(data["timestamps"])

    if data["positions"].shape != (n_frames, 2):
        raise ValueError(f"Invalid positions shape: expected ({n_frames}, 2)")

    if len(data["confidences"]) != n_frames:
        raise ValueError(f"Confidences length mismatch: expected {n_frames}")

    if data["headings"] is not None and len(data["headings"]) != n_frames:
        raise ValueError(f"Headings length mismatch: expected {n_frames}")


def validate_imu_data(data: Dict[str, Any]) -> None:
    """Validate loaded IMU data structure."""
    required_keys = ["timestamps", "data", "sampling_rate"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    n_samples = len(data["timestamps"])

    if data["data"].shape != (n_samples, 6):
        raise ValueError(f"Invalid IMU data shape: expected ({n_samples}, 6)")

    if data["sampling_rate"] <= 0:
        raise ValueError("Sampling rate must be positive")
