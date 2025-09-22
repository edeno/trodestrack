"""Timestamp alignment utilities for synchronizing video and IMU data."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
from ..constants import (
    DEFAULT_SYNC_SAMPLE_FRAMES,
    DEFAULT_ALIGNMENT_MAX_ERROR_S,
    DEFAULT_SYNC_TOLERANCE_S,
    DEFAULT_DRIFT_RATE,
)


def align_timestamps(
    video_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
    method: str = "nearest",
    max_gap: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align video frames with IMU samples using various interpolation methods.

    Parameters
    ----------
    video_timestamps : np.ndarray, shape (n_video,)
        Video frame timestamps in seconds
    imu_timestamps : np.ndarray, shape (n_imu,)
        IMU sample timestamps in seconds
    method : str, optional
        Alignment method, one of {"nearest", "interpolate", "subsample"}.
        Default is "nearest".
    max_gap : float, optional
        Maximum allowed time gap for alignment in seconds.
        Frames/samples with larger gaps are excluded. Default is None (no limit).

    Returns
    -------
    video_indices : np.ndarray, shape (n_aligned,)
        Indices of aligned video frames
    imu_indices : np.ndarray, shape (n_aligned,)
        Indices of aligned IMU samples

    Raises
    ------
    ValueError
        If alignment method is invalid or timestamp arrays are empty

    Notes
    -----
    The "nearest" method uses vectorized nearest neighbor search with O(n_video * n_imu)
    complexity. For large datasets, consider subsampling first.
    """
    if len(video_timestamps) == 0 or len(imu_timestamps) == 0:
        raise ValueError("Cannot align empty timestamp arrays")

    if method == "nearest":
        return _align_nearest(video_timestamps, imu_timestamps, max_gap)
    elif method == "interpolate":
        return _align_interpolate(video_timestamps, imu_timestamps, max_gap)
    elif method == "subsample":
        return _align_subsample(video_timestamps, imu_timestamps, max_gap)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _align_nearest(
    video_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
    max_gap: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align by finding nearest IMU sample for each video frame.

    Parameters
    ----------
    video_timestamps : np.ndarray, shape (n_video,)
        Video frame timestamps in seconds
    imu_timestamps : np.ndarray, shape (n_imu,)
        IMU sample timestamps in seconds
    max_gap : float, optional
        Maximum allowed gap between video and IMU timestamps in seconds.
        Default is None (no limit).

    Returns
    -------
    video_indices : np.ndarray, shape (n_aligned,)
        Indices of aligned video frames
    imu_indices : np.ndarray, shape (n_aligned,)
        Indices of aligned IMU samples

    Notes
    -----
    Uses vectorized nearest neighbor search with O(n_video * n_imu) complexity.
    """
    # Vectorized nearest neighbor search
    video_timestamps = np.asarray(video_timestamps)
    imu_timestamps = np.asarray(imu_timestamps)

    # Compute all pairwise distances using broadcasting
    time_diffs = np.abs(video_timestamps[:, np.newaxis] - imu_timestamps[np.newaxis, :])

    # Find nearest IMU indices for all video timestamps
    nearest_imu_indices = np.argmin(time_diffs, axis=1)
    min_gaps = time_diffs[np.arange(len(video_timestamps)), nearest_imu_indices]

    # Apply gap constraint
    if max_gap is not None:
        valid_mask = min_gaps <= max_gap
        video_indices = np.where(valid_mask)[0]
        imu_indices = nearest_imu_indices[valid_mask]
    else:
        video_indices = np.arange(len(video_timestamps))
        imu_indices = nearest_imu_indices

    return video_indices, imu_indices


def _align_interpolate(
    video_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
    max_gap: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align by interpolating IMU data to video timestamps.

    Parameters
    ----------
    video_timestamps : np.ndarray, shape (n_video,)
        Video frame timestamps in seconds
    imu_timestamps : np.ndarray, shape (n_imu,)
        IMU sample timestamps in seconds
    max_gap : float, optional
        Maximum allowed gap for extrapolation in seconds

    Returns
    -------
    video_indices : np.ndarray, shape (n_aligned,)
        Indices of aligned video frames
    imu_interpolation_info : np.ndarray, shape (n_aligned,)
        IMU interpolation information

    Notes
    -----
    This returns video indices and interpolation weights for IMU data.
    The actual interpolation should be done by the caller.
    Currently falls back to nearest neighbor alignment.
    """
    # For now, implement as nearest neighbor with interpolation info
    # Full interpolation requires knowing the IMU data shape
    warnings.warn("Interpolation method not fully implemented, using nearest neighbor")
    return _align_nearest(video_timestamps, imu_timestamps, max_gap)


def _align_subsample(
    video_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
    max_gap: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align by subsampling IMU to video rate.

    Parameters
    ----------
    video_timestamps : np.ndarray, shape (n_video,)
        Video frame timestamps in seconds
    imu_timestamps : np.ndarray, shape (n_imu,)
        IMU sample timestamps in seconds
    max_gap : float, optional
        Maximum allowed gap between aligned timestamps in seconds

    Returns
    -------
    video_indices : np.ndarray, shape (n_aligned,)
        Indices of aligned video frames
    imu_indices : np.ndarray, shape (n_aligned,)
        Indices of subsampled IMU samples

    Notes
    -----
    Estimates video rate and subsamples IMU accordingly.
    Falls back to nearest neighbor if IMU rate is too low.
    """
    # Estimate video frame rate
    if len(video_timestamps) > 1:
        video_dt = np.median(np.diff(video_timestamps))
        video_rate = 1.0 / video_dt
    else:
        raise ValueError("Need at least 2 video frames for subsampling")

    # Estimate IMU rate
    if len(imu_timestamps) > 1:
        imu_dt = np.median(np.diff(imu_timestamps))
        imu_rate = 1.0 / imu_dt
    else:
        raise ValueError("Need at least 2 IMU samples for subsampling")

    # Calculate decimation factor
    decimation_factor = int(imu_rate / video_rate)

    if decimation_factor < 1:
        warnings.warn("IMU rate lower than video rate, using nearest neighbor")
        return _align_nearest(video_timestamps, imu_timestamps, max_gap)

    # Subsample IMU timestamps
    subsampled_imu_indices = np.arange(0, len(imu_timestamps), decimation_factor)
    subsampled_imu_timestamps = imu_timestamps[subsampled_imu_indices]

    # Align subsampled IMU with video
    video_indices, aligned_imu_sub_indices = _align_nearest(
        video_timestamps, subsampled_imu_timestamps, max_gap
    )

    # Map back to original IMU indices
    imu_indices = subsampled_imu_indices[aligned_imu_sub_indices]

    return video_indices, imu_indices


def check_timestamp_synchronization(
    video_timestamps: np.ndarray, imu_timestamps: np.ndarray, tolerance: float = DEFAULT_SYNC_TOLERANCE_S
) -> Dict[str, Any]:
    """Check if video and IMU timestamps are hardware synchronized.

    Parameters
    ----------
    video_timestamps : np.ndarray, shape (n_video,)
        Video frame timestamps in seconds
    imu_timestamps : np.ndarray, shape (n_imu,)
        IMU sample timestamps in seconds
    tolerance : float, optional
        Tolerance for synchronization check in seconds. Default is DEFAULT_SYNC_TOLERANCE_S (1 ms).

    Returns
    -------
    dict
        Synchronization analysis results containing:
        - "is_synchronized" : bool
        - "overlap_duration" : float, duration of overlapping timestamps
        - "alignment_error_stats" : dict with mean, std, max alignment errors
        - "clock_offset_estimate" : float, estimated constant offset

    Notes
    -----
    Uses alignment error statistics to determine if timestamps appear to originate
    from the same hardware clock (e.g., SpikeGadgets synchronization).
    """
    # Find overlapping time range
    video_start, video_end = video_timestamps[0], video_timestamps[-1]
    imu_start, imu_end = imu_timestamps[0], imu_timestamps[-1]

    overlap_start = max(video_start, imu_start)
    overlap_end = min(video_end, imu_end)
    overlap_duration = overlap_end - overlap_start

    # Check if there's sufficient overlap
    if overlap_duration <= 0:
        return {
            "synchronized": False,
            "reason": "No temporal overlap",
            "video_range": (video_start, video_end),
            "imu_range": (imu_start, imu_end),
            "overlap_duration": 0.0,
        }

    # Sample alignment quality in overlap region
    video_mask = (video_timestamps >= overlap_start) & (video_timestamps <= overlap_end)
    overlap_video_times = video_timestamps[video_mask]

    if len(overlap_video_times) < 2:
        return {
            "synchronized": False,
            "reason": "Insufficient video frames in overlap",
            "overlap_duration": overlap_duration,
        }

    # Find nearest IMU samples for video frames
    alignment_errors = []
    for video_time in overlap_video_times[:DEFAULT_SYNC_SAMPLE_FRAMES]:  # Sample first N frames
        time_diffs = np.abs(imu_timestamps - video_time)
        min_error = np.min(time_diffs)
        alignment_errors.append(min_error)

    alignment_errors = np.array(alignment_errors)
    mean_error = np.mean(alignment_errors)
    max_error = np.max(alignment_errors)

    # Determine if synchronized
    is_synchronized = bool(max_error < tolerance)

    return {
        "synchronized": is_synchronized,
        "mean_alignment_error": mean_error,
        "max_alignment_error": max_error,
        "tolerance": tolerance,
        "overlap_duration": overlap_duration,
        "video_range": (video_start, video_end),
        "imu_range": (imu_start, imu_end),
        "n_video_frames": len(video_timestamps),
        "n_imu_samples": len(imu_timestamps),
    }


def estimate_clock_offset(
    video_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
    method: str = "cross_correlation",
) -> float:
    """Estimate clock offset between video and IMU timestamps.

    Args:
        video_timestamps: Video frame timestamps
        imu_timestamps: IMU sample timestamps
        method: Method for offset estimation

    Returns:
        Estimated time offset (IMU_time = video_time + offset)
    """
    if method == "cross_correlation":
        return _estimate_offset_xcorr(video_timestamps, imu_timestamps)
    elif method == "first_sample":
        return imu_timestamps[0] - video_timestamps[0]
    else:
        raise ValueError(f"Unknown offset estimation method: {method}")


def _estimate_offset_xcorr(
    video_timestamps: np.ndarray, imu_timestamps: np.ndarray
) -> float:
    """Estimate offset using cross-correlation of timestamp derivatives.

    Args:
        video_timestamps: Video frame timestamps
        imu_timestamps: IMU sample timestamps

    Returns:
        Estimated time offset
    """
    # For now, use simple first-sample method
    # Full cross-correlation requires more sophisticated signal processing
    warnings.warn("Cross-correlation method not implemented, using first sample")
    return imu_timestamps[0] - video_timestamps[0]


def apply_clock_correction(
    timestamps: np.ndarray, offset: float, drift_rate: float = DEFAULT_DRIFT_RATE
) -> np.ndarray:
    """Apply clock offset and drift correction to timestamps.

    Args:
        timestamps: Original timestamps
        offset: Time offset to add
        drift_rate: Clock drift rate (fractional change per second)

    Returns:
        Corrected timestamps
    """
    corrected = timestamps + offset

    if drift_rate != 0.0:
        # Apply drift correction: t_corrected = t * (1 + drift_rate * t)
        corrected = corrected * (1.0 + drift_rate * corrected)

    return corrected


def validate_alignment(
    video_indices: np.ndarray,
    imu_indices: np.ndarray,
    video_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
    max_error: float = DEFAULT_ALIGNMENT_MAX_ERROR_S,
) -> Dict[str, Any]:
    """Validate timestamp alignment quality.

    Args:
        video_indices: Aligned video frame indices
        imu_indices: Aligned IMU sample indices
        video_timestamps: Original video timestamps
        imu_timestamps: Original IMU timestamps
        max_error: Maximum acceptable alignment error (seconds)

    Returns:
        Dictionary with validation results
    """
    if len(video_indices) != len(imu_indices):
        raise ValueError("Video and IMU index arrays must have same length")

    if len(video_indices) == 0:
        return {"valid": False, "reason": "No aligned samples", "n_aligned": 0}

    # Calculate alignment errors
    aligned_video_times = video_timestamps[video_indices]
    aligned_imu_times = imu_timestamps[imu_indices]
    errors = np.abs(aligned_video_times - aligned_imu_times)

    # Statistics
    mean_error = np.mean(errors)
    max_error_actual = np.max(errors)
    std_error = np.std(errors)

    # Validation
    is_valid = bool(max_error_actual <= max_error)

    return {
        "valid": is_valid,
        "n_aligned": len(video_indices),
        "mean_error": mean_error,
        "max_error": max_error_actual,
        "std_error": std_error,
        "error_threshold": max_error,
        "alignment_rate": len(video_indices) / len(video_timestamps),
    }
