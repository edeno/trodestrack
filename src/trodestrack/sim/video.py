"""Synthetic video data generation."""

import numpy as np
from typing import Dict, Any
from ..io.trodes import TrodesLEDData
from .config import SimConfig


def generate_synthetic_video(ground_truth: Dict[str, np.ndarray], config: SimConfig) -> TrodesLEDData:
    """Generate synthetic video data from ground truth trajectory.

    Args:
        ground_truth: Dictionary containing ground truth trajectory data
        config: Simulation configuration

    Returns:
        TrodesLEDData with synthetic video measurements
    """
    # Set random seed for reproducibility
    np.random.seed(config.seed + 1)  # Different seed than IMU

    # Extract ground truth data
    gt_timestamps = ground_truth["timestamps"]
    gt_positions = ground_truth["positions"]  # [N, 2] in cm
    gt_headings = ground_truth["headings"]  # [N] in radians

    # Generate video timestamps
    n_frames = int(config.duration * config.video_fps)
    video_timestamps = np.linspace(0, config.duration, n_frames)

    # Interpolate ground truth to video timestamps
    video_positions = np.column_stack([
        np.interp(video_timestamps, gt_timestamps, gt_positions[:, 0]),
        np.interp(video_timestamps, gt_timestamps, gt_positions[:, 1])
    ])
    video_headings = np.interp(video_timestamps, gt_timestamps, gt_headings)

    # Convert positions from cm to pixels (assume 1 cm = 2 pixels for now)
    cm_to_pixels = 2.0
    video_positions_px = video_positions * cm_to_pixels

    # Generate LED positions based on heading and front-back distance
    led_distance_px = config.led.front_back_distance
    half_distance = led_distance_px / 2.0

    # Front LED is ahead in heading direction, back LED is behind
    cos_heading = np.cos(video_headings)
    sin_heading = np.sin(video_headings)

    front_led_offset = np.column_stack([
        half_distance * cos_heading,
        half_distance * sin_heading
    ])
    back_led_offset = -front_led_offset

    # True LED positions (no noise yet)
    front_led_true = video_positions_px + front_led_offset
    back_led_true = video_positions_px + back_led_offset

    # Generate confidence values
    base_confidence = np.random.uniform(
        config.video.confidence_min,
        config.video.confidence_max,
        n_frames
    )

    # Initialize output arrays
    front_led = np.copy(front_led_true)
    back_led = np.copy(back_led_true)
    front_confidence = np.copy(base_confidence)
    back_confidence = np.copy(base_confidence)

    # Apply noise based on confidence (lower confidence = more noise) - vectorized
    noise_scale_front = config.video.position_noise_std * (1.0 / front_confidence)
    noise_scale_back = config.video.position_noise_std * (1.0 / back_confidence)

    # Generate all noise at once
    front_noise = np.random.normal(0, 1, (n_frames, 2)) * noise_scale_front[:, np.newaxis]
    back_noise = np.random.normal(0, 1, (n_frames, 2)) * noise_scale_back[:, np.newaxis]

    front_led += front_noise
    back_led += back_noise

    # Apply occlusions
    occlusion_frames = _generate_occlusions(
        n_frames, config.video.occlusion_probability,
        config.video.occlusion_duration_mean, config.video_fps
    )

    # Reduce confidence during occlusions (but keep above minimum)
    occlusion_confidence = max(config.video.confidence_min, 0.05)
    front_confidence[occlusion_frames] = occlusion_confidence
    back_confidence[occlusion_frames] = occlusion_confidence

    # Apply LED swaps occasionally
    swap_frames = np.random.random(n_frames) < config.video.led_swap_probability
    if np.any(swap_frames):
        # Swap front and back LEDs at these frames
        temp_front = front_led[swap_frames].copy()
        front_led[swap_frames] = back_led[swap_frames]
        back_led[swap_frames] = temp_front

        # Also swap confidences
        temp_front_conf = front_confidence[swap_frames].copy()
        front_confidence[swap_frames] = back_confidence[swap_frames]
        back_confidence[swap_frames] = temp_front_conf

    # Apply frame dropouts
    dropout_frames = np.random.random(n_frames) < config.video.dropout_probability
    front_confidence[dropout_frames] = 0.0
    back_confidence[dropout_frames] = 0.0
    # Set positions to NaN for dropped frames
    front_led[dropout_frames] = np.nan
    back_led[dropout_frames] = np.nan

    # Store metadata for testing
    metadata = {
        "true_front_led": front_led_true,
        "true_back_led": back_led_true,
        "true_positions_cm": video_positions,
        "true_headings": video_headings,
        "occlusion_frames": occlusion_frames,
        "swap_frames": swap_frames,
        "dropout_frames": dropout_frames,
        "cm_to_pixels": cm_to_pixels,
        "led_distance_px": led_distance_px
    }

    return TrodesLEDData(
        timestamps=video_timestamps,
        front_led=front_led,
        back_led=back_led,
        front_confidence=front_confidence,
        back_confidence=back_confidence,
        metadata=metadata
    )


def _generate_occlusions(n_frames: int, occlusion_prob: float,
                        duration_mean: float, fps: float) -> np.ndarray:
    """Generate occlusion mask with realistic duration.

    Args:
        n_frames: Total number of frames
        occlusion_prob: Probability of starting an occlusion per frame
        duration_mean: Mean occlusion duration in seconds
        fps: Video frame rate

    Returns:
        Boolean array indicating occluded frames
    """
    occlusion_mask = np.zeros(n_frames, dtype=bool)

    # Mean duration in frames
    duration_frames = int(duration_mean * fps)

    i = 0
    while i < n_frames:
        # Check if we start an occlusion
        if np.random.random() < occlusion_prob:
            # Generate occlusion duration (exponential distribution)
            duration = max(1, int(np.random.exponential(duration_frames)))
            end_frame = min(i + duration, n_frames)
            occlusion_mask[i:end_frame] = True
            i = end_frame
        else:
            i += 1

    return occlusion_mask