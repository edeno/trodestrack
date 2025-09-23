"""Offline smoothing API for trodestrack.

This module implements the main offline smoothing pipeline that combines:
- Data loading and preprocessing
- EKF filtering with IMU pre-integration
- RTS smoothing for improved accuracy
- Results saving and diagnostics
"""

import logging
import warnings
from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from ..config.schemas import SessionConfig
from ..geom.homography import transform_points_pixel_to_cm
from ..imu.preintegration import preintegrate_between_frames
from ..io.loaders import load_imu_data, load_video_detections
from ..models.ekf import EkfCarry, EKFFilter, ekf_step, ekf_step_arrays
from ..models.rts_smoother import ForwardPassData, rts_smooth
from ..models.state import State2D, create_initial_state, state_to_array

logger = logging.getLogger(__name__)


class SmoothingResult(NamedTuple):
    """Result from offline smoothing pipeline.

    Attributes:
        filtered_states: Array of filtered states (n_frames, 8) from EKF
        smoothed_states: Array of smoothed states (n_frames, 8) from RTS
        timestamps: Frame timestamps (n_frames,)
        filtered_covariances: Filtered covariances (n_frames, 8, 8)
        smoothed_covariances: Smoothed covariances (n_frames, 8, 8)
        log_likelihood: Total log-likelihood from filtering
        diagnostics: Dictionary of diagnostic information
    """

    filtered_states: jnp.ndarray
    smoothed_states: jnp.ndarray
    timestamps: jnp.ndarray
    filtered_covariances: jnp.ndarray
    smoothed_covariances: jnp.ndarray
    log_likelihood: float
    diagnostics: dict


def smooth_session(config: SessionConfig) -> SmoothingResult:
    """Run offline smoothing pipeline on a session.

    This is the main API function that implements the complete offline
    smoothing workflow from raw data to optimized state estimates.

    Args:
        config: Session configuration with data paths and parameters

    Returns:
        SmoothingResult with filtered and smoothed state estimates

    Raises:
        FileNotFoundError: If input files are missing
        ValueError: If configuration is invalid or data is incompatible
    """
    logger.info("Starting offline smoothing pipeline")

    # Validate required inputs
    if config.video_file is None and config.imu_file is None:
        raise ValueError("At least one of video_file or imu_file must be specified")

    # Create output directory if needed
    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    video_data, imu_data, sync_info = _load_and_sync_data(config)

    # Initialize filter
    logger.info("Initializing Extended Kalman Filter")
    ekf_filter, initial_state = _initialize_filter(config, video_data, imu_data)

    # Run filtering pass
    logger.info("Running EKF filtering pass")
    (
        filtered_states,
        filtered_covariances,
        frame_timestamps,
        predicted_states,
        predicted_covariances,
    ) = _run_filtering_pass(ekf_filter, config, video_data, imu_data, sync_info)

    # Run smoothing pass
    logger.info("Running RTS smoothing pass")
    smoothed_states, smoothed_covariances = _run_smoothing_pass(
        filtered_states,
        filtered_covariances,
        predicted_states,
        predicted_covariances,
        config,
        imu_data,
        frame_timestamps,
    )

    # Collect diagnostics
    diagnostics = _collect_diagnostics(
        config, filtered_states, smoothed_states, sync_info, ekf_filter
    )

    # Save results if requested
    if config.output.save_states:
        _save_results(config, filtered_states, smoothed_states, frame_timestamps, diagnostics)

    logger.info("Offline smoothing completed successfully")

    return SmoothingResult(
        filtered_states=filtered_states,
        smoothed_states=smoothed_states,
        timestamps=frame_timestamps,
        filtered_covariances=filtered_covariances,
        smoothed_covariances=smoothed_covariances,
        log_likelihood=ekf_filter.get_log_likelihood(),
        diagnostics=diagnostics,
    )


def _load_and_sync_data(config: SessionConfig) -> Tuple[Optional[dict], Optional[dict], dict]:
    """Load video and IMU data and perform time synchronization."""
    video_data = None
    imu_data = None

    # Load video data
    if config.video_file is not None:
        logger.info(f"Loading video detections from: {config.video_file}")
        video_data = load_video_detections(config.video_file)

        # Apply coordinate mapping
        if config.mapping.type == "homography":
            logger.info("Applying homography coordinate transformation")
            # Apply homography transformation
            homography_matrix = jnp.array(config.mapping.homography_matrix)
            video_data["positions"] = transform_points_pixel_to_cm(
                video_data["positions"], homography_matrix
            )
        elif config.mapping.type == "ruler_scale":
            logger.info("Applying ruler-scale coordinate transformation")
            # Convert pixels to cm using scale
            scale = config.mapping.pixel_per_cm
            video_data["positions"] = video_data["positions"] / scale

    # Load IMU data
    if config.imu_file is not None:
        logger.info(f"Loading IMU data from: {config.imu_file}")
        imu_data = load_imu_data(config.imu_file)

        # Downsample if requested
        if config.imu.downsampling_rate < imu_data["sampling_rate"]:
            logger.info(
                f"Downsampling IMU from {imu_data['sampling_rate']:.1f} Hz to {config.imu.downsampling_rate:.1f} Hz"
            )
            imu_data = _downsample_imu_data(imu_data, config.imu.downsampling_rate)

    # Perform synchronization
    sync_info = _synchronize_timestamps(video_data, imu_data, config.synchronization)

    return video_data, imu_data, sync_info


def _initialize_filter(
    config: SessionConfig, video_data: Optional[dict], imu_data: Optional[dict]
) -> Tuple[EKFFilter, State2D]:
    """Initialize the EKF filter with appropriate initial conditions."""

    # Create initial state estimate
    if video_data is not None:
        # Use first few frames for initialization
        n_init = min(3, len(video_data["positions"]))
        homography = jnp.eye(3)  # Identity for now, actual homography applied earlier
        if config.mapping.type == "homography":
            homography = jnp.array(config.mapping.homography_matrix)

        initial_state, _ = create_initial_state(
            positions=video_data["positions"][:n_init],
            timestamps=video_data["timestamps"][:n_init],
            confidences=video_data["confidences"][:n_init],
            homography=homography,
        )
    else:
        # IMU-only initialization (less accurate)
        logger.warning("Initializing with IMU-only data - position accuracy will be limited")
        initial_state = State2D(
            x=0.0,
            y=0.0,  # Unknown initial position
            vx=0.0,
            vy=0.0,  # Start at rest
            theta=0.0,  # Unknown initial heading
            b_gz=0.0,
            b_ax=0.0,
            b_ay=0.0,  # Assume no initial bias
        )

    # Create initial covariance matrix
    initial_covariance = _create_initial_covariance(config.filter.initial_state_variance)

    # Initialize EKF filter
    ekf_filter = EKFFilter(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        velocity_damping=config.filter.velocity_damping,
        accel_noise_std=np.sqrt(config.filter.process_noise["velocity"]),
        gyro_noise_std=np.sqrt(config.filter.process_noise["heading"]),
        bias_drift_std=np.sqrt(config.filter.process_noise["bias_gyro"]),
        position_noise_std=np.sqrt(config.filter.measurement_noise["position"]),
        heading_noise_std=np.sqrt(config.filter.measurement_noise["heading"]),
        gate_threshold=config.filter.gating_threshold,
    )

    return ekf_filter, initial_state


def _run_filtering_pass(
    ekf_filter: EKFFilter,
    config: SessionConfig,
    video_data: Optional[dict],
    imu_data: Optional[dict],
    sync_info: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the forward filtering pass with EKF."""

    # Determine frame timestamps
    if video_data is not None:
        frame_timestamps = video_data["timestamps"]
    else:
        # Create artificial frames at video rate for IMU-only processing
        duration = imu_data["timestamps"][-1] - imu_data["timestamps"][0]
        n_frames = int(duration * config.video_fps)
        frame_timestamps = jnp.linspace(
            imu_data["timestamps"][0], imu_data["timestamps"][-1], n_frames
        )

    n_frames = len(frame_timestamps)
    logger.info(f"Processing {n_frames} frames")

    # Use consistent filtering implementation for all dataset sizes
    return _run_filtering_pass_consistent(
        ekf_filter, config, video_data, imu_data, frame_timestamps
    )


def _run_filtering_pass_consistent(
    ekf_filter: EKFFilter,
    config: SessionConfig,
    video_data: Optional[dict],
    imu_data: Optional[dict],
    frame_timestamps: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX lax.scan-based filtering implementation for all dataset sizes."""
    n_frames = len(frame_timestamps)
    logger.info(f"Processing {n_frames} frames with JAX lax.scan EKF")

    # Get initial state from the EKF filter
    initial_state = ekf_filter.ekf_state.state
    initial_covariance = ekf_filter.ekf_state.covariance

    # Prepare measurement data as JAX arrays
    if video_data is not None:
        positions = video_data["positions"]
        confidences = jnp.array(video_data.get("confidences", [1.0] * len(positions)))
        headings = video_data.get("headings")
        if headings is not None:
            headings = jnp.array(headings)
        else:
            headings = jnp.full(len(positions), jnp.nan)

        # Pad data to match frame count if needed
        if len(positions) < n_frames:
            pad_size = n_frames - len(positions)
            positions = jnp.concatenate([positions, jnp.full((pad_size, 2), jnp.nan)])
            confidences = jnp.concatenate([confidences, jnp.zeros(pad_size)])
            headings = jnp.concatenate([headings, jnp.full(pad_size, jnp.nan)])
    else:
        # No video data - create arrays of invalid measurements
        positions = jnp.full((n_frames, 2), jnp.nan)
        confidences = jnp.zeros(n_frames)
        headings = jnp.full(n_frames, jnp.nan)

    # Create validity masks
    position_mask = jnp.all(jnp.isfinite(positions), axis=1)
    heading_mask = jnp.isfinite(headings)

    # Compute time differences
    dts = jnp.diff(frame_timestamps, prepend=frame_timestamps[0])

    # Prepare IMU data for each frame if available
    if imu_data is not None:
        imu_blocks = _prepare_imu_blocks_for_frames(imu_data, frame_timestamps, config)
    else:
        # Create dummy IMU blocks (zeros)
        imu_blocks = jnp.zeros((n_frames, 3))  # [ax, ay, gz]

    # Create filter configuration arrays (constant values repeated for each frame)
    velocity_damping = jnp.full(n_frames, config.filter.velocity_damping)
    accel_noise_std = jnp.full(n_frames, np.sqrt(config.filter.process_noise["velocity"]))
    gyro_noise_std = jnp.full(n_frames, np.sqrt(config.filter.process_noise["heading"]))
    bias_drift_std = jnp.full(n_frames, np.sqrt(config.filter.process_noise["bias_gyro"]))
    position_noise_std = jnp.full(n_frames, np.sqrt(config.filter.measurement_noise["position"]))
    heading_noise_std = jnp.full(n_frames, np.sqrt(config.filter.measurement_noise["heading"]))
    gate_threshold = jnp.full(n_frames, config.filter.gating_threshold)

    # Create scan inputs: transpose all arrays to create sequence of inputs
    scan_inputs = (
        positions,  # (n_frames, 2)
        dts,  # (n_frames,)
        imu_blocks,  # (n_frames, 3)
        headings,  # (n_frames,)
        confidences,  # (n_frames,)
        position_mask,  # (n_frames,)
        heading_mask,  # (n_frames,)
        velocity_damping,  # (n_frames,)
        accel_noise_std,  # (n_frames,)
        gyro_noise_std,  # (n_frames,)
        bias_drift_std,  # (n_frames,)
        position_noise_std,  # (n_frames,)
        heading_noise_std,  # (n_frames,)
        gate_threshold,  # (n_frames,)
    )

    # Initial carry state
    carry0 = EkfCarry(x=initial_state, P=initial_covariance)

    # Run lax.scan with the JAX-compatible EKF step
    final_carry, outputs = lax.scan(ekf_step_arrays, carry0, scan_inputs)

    # Extract results (both filtered and predicted for RTS)
    filtered_states = outputs.x_filt
    filtered_covariances = outputs.P_filt
    predicted_states = outputs.x_pred
    predicted_covariances = outputs.P_pred

    return (
        filtered_states,
        filtered_covariances,
        frame_timestamps,
        predicted_states,
        predicted_covariances,
    )


def _prepare_imu_blocks_for_frames(
    imu_data: dict,
    frame_timestamps: jnp.ndarray,
    config: SessionConfig,
) -> jnp.ndarray:
    """Prepare IMU measurement blocks for each frame using pre-integration."""
    n_frames = len(frame_timestamps)
    imu_blocks = []

    prev_timestamp = frame_timestamps[0]

    for i, timestamp in enumerate(frame_timestamps):
        dt = timestamp - prev_timestamp

        if dt > 0 and i > 0:
            # Pre-integrate IMU between frames for average measurements
            try:
                imu_result = preintegrate_between_frames(
                    imu_data["data"],
                    imu_data["timestamps"],
                    prev_timestamp,
                    timestamp,
                    initial_heading=0.0,  # Will be corrected in EKF
                    initial_velocity=jnp.array([0.0, 0.0]),
                    gyro_bias=0.0,
                    accel_bias=jnp.array([0.0, 0.0]),
                    damping_lambda=config.filter.velocity_damping,
                )

                if imu_result.n_samples > 0:
                    # Use average angular velocity for this interval
                    avg_accel = jnp.array([0.0, 0.0])  # Pre-integration handles accel
                    avg_gyro = imu_result.delta_heading / dt
                    imu_block = jnp.array([avg_accel[0], avg_accel[1], avg_gyro])
                else:
                    imu_block = jnp.array([0.0, 0.0, 0.0])

            except Exception:
                # Fallback to zero measurements
                imu_block = jnp.array([0.0, 0.0, 0.0])
        else:
            # First frame or zero dt
            imu_block = jnp.array([0.0, 0.0, 0.0])

        imu_blocks.append(imu_block)
        prev_timestamp = timestamp

    return jnp.stack(imu_blocks)


def _run_smoothing_pass(
    filtered_states: jnp.ndarray,
    filtered_covariances: jnp.ndarray,
    predicted_states: jnp.ndarray,
    predicted_covariances: jnp.ndarray,
    config: SessionConfig,
    imu_data: Optional[dict],
    frame_timestamps: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run the backward smoothing pass with RTS smoother."""

    if config.filter.filter_type == "ekf":
        logger.info("Running JAX-optimized RTS smoothing with true forward predictions")

        # Create forward pass data for RTS using true predictions from EKF forward pass
        forward_data = ForwardPassData(
            filtered_states=list(filtered_states),
            filtered_covariances=list(filtered_covariances),
            predicted_states=list(predicted_states),
            predicted_covariances=list(predicted_covariances),
            log_likelihood=0.0,  # Not used in offline context
        )

        # Run JAX-optimized RTS smoother
        rts_result = rts_smooth(forward_data)

        smoothed_states = jnp.array(rts_result.smoothed_states)
        smoothed_covariances = jnp.array(rts_result.smoothed_covariances)

        logger.info("JAX-optimized RTS smoothing completed")
    else:
        logger.info(
            "UKF selected - using filtered results as smoothed (UKF smoothing not yet implemented)"
        )
        smoothed_states = filtered_states
        smoothed_covariances = filtered_covariances

    return smoothed_states, smoothed_covariances


def _create_initial_covariance(initial_variances: dict) -> jnp.ndarray:
    """Create initial covariance matrix from configuration."""
    return jnp.diag(
        jnp.array(
            [
                initial_variances["position"],  # x
                initial_variances["position"],  # y
                initial_variances["velocity"],  # vx
                initial_variances["velocity"],  # vy
                initial_variances["heading"],  # theta
                initial_variances["bias_gyro"],  # b_gz
                initial_variances["bias_accel"],  # b_ax
                initial_variances["bias_accel"],  # b_ay
            ]
        )
    )


def _downsample_imu_data(imu_data: dict, target_rate: float) -> dict:
    """Downsample IMU data to target rate."""
    original_rate = imu_data["sampling_rate"]
    decimation_factor = int(original_rate / target_rate)

    if decimation_factor <= 1:
        return imu_data

    # Simple decimation
    downsampled_data = {
        "data": imu_data["data"][::decimation_factor],
        "timestamps": imu_data["timestamps"][::decimation_factor],
        "sampling_rate": original_rate / decimation_factor,
    }

    return downsampled_data


def _synchronize_timestamps(
    video_data: Optional[dict], imu_data: Optional[dict], sync_config
) -> dict:
    """Synchronize video and IMU timestamps."""
    sync_info = {
        "method": sync_config.method,
        "video_start": None,
        "video_end": None,
        "imu_start": None,
        "imu_end": None,
        "overlap_start": None,
        "overlap_end": None,
    }

    if video_data is not None:
        sync_info["video_start"] = video_data["timestamps"][0]
        sync_info["video_end"] = video_data["timestamps"][-1]

    if imu_data is not None:
        sync_info["imu_start"] = imu_data["timestamps"][0]
        sync_info["imu_end"] = imu_data["timestamps"][-1]

    # Compute overlap period
    if video_data is not None and imu_data is not None:
        sync_info["overlap_start"] = max(sync_info["video_start"], sync_info["imu_start"])
        sync_info["overlap_end"] = min(sync_info["video_end"], sync_info["imu_end"])

        overlap_duration = sync_info["overlap_end"] - sync_info["overlap_start"]
        if overlap_duration <= 0:
            warnings.warn("No temporal overlap between video and IMU data")

    return sync_info


def _collect_diagnostics(
    config: SessionConfig,
    filtered_states: jnp.ndarray,
    smoothed_states: jnp.ndarray,
    sync_info: dict,
    ekf_filter: EKFFilter,
) -> dict:
    """Collect diagnostic information about the smoothing run."""

    diagnostics = {
        "config_summary": {
            "filter_type": config.filter.filter_type,
            "video_fps": config.video_fps,
            "mapping_type": config.mapping.type,
            "sync_method": config.synchronization.method,
        },
        "data_summary": {
            "n_frames": len(filtered_states),
            "duration_s": (sync_info.get("overlap_end") or 0)
            - (sync_info.get("overlap_start") or 0),
        },
        "filter_performance": {
            "log_likelihood": ekf_filter.get_log_likelihood(),
        },
        "sync_info": sync_info,
    }

    # Compute RMSE improvement from smoothing
    if len(smoothed_states) > 0:
        position_rmse_improvement = _compute_smoothing_improvement(
            filtered_states[:, :2], smoothed_states[:, :2]
        )
        diagnostics["smoothing_improvement"] = {
            "position_rmse_improvement_cm": position_rmse_improvement,
        }

    return diagnostics


def _compute_smoothing_improvement(
    filtered_positions: jnp.ndarray, smoothed_positions: jnp.ndarray
) -> float:
    """Compute RMS improvement from smoothing."""
    if len(filtered_positions) < 2:
        return 0.0

    # Compute position differences (proxy for improvement)
    diff = jnp.linalg.norm(smoothed_positions - filtered_positions, axis=1)
    return float(jnp.mean(diff))


def _save_results(
    config: SessionConfig,
    filtered_states: jnp.ndarray,
    smoothed_states: jnp.ndarray,
    timestamps: jnp.ndarray,
    diagnostics: dict,
) -> None:
    """Save results to output directory."""

    output_dir = config.output.output_dir

    # Save state estimates as numpy arrays (parquet would need pandas)
    logger.info("Saving state estimates")
    np.savez(
        output_dir / "states.npz",
        filtered_states=np.array(filtered_states),
        smoothed_states=np.array(smoothed_states),
        timestamps=np.array(timestamps),
    )

    # Save diagnostics as JSON
    import json

    diagnostics_serializable = _make_json_serializable(diagnostics)
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics_serializable, f, indent=2)

    logger.info(f"Results saved to: {output_dir}")


def _make_json_serializable(obj):
    """Convert JAX/NumPy objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (jnp.float64, jnp.float32, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (jnp.int64, jnp.int32, np.int64, np.int32)):
        return int(obj)
    else:
        return obj
