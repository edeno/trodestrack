"""Utilities for video generation: frame interpolation, time sync."""

from __future__ import annotations

import numpy as np

from trodestrack.sim.utils import SimOut, interp_angle


def prepare_video_data(
    sim_data: SimOut, fps: int = 30, speedup: float = 1.0
) -> dict[str, np.ndarray | int]:
    """Interpolate simulation data to video frame times.

    Handles different sampling rates:
    - IMU: typically 200+ Hz
    - Camera: typically 30 Hz
    - Video: target fps (e.g., 30 fps)

    Args:
        sim_data: Simulation output dictionary
        fps: Target video frame rate (frames per second)
        speedup: Playback speed multiplier (>1 = faster, <1 = slower)

    Returns:
        Dictionary with interpolated data at video frame times:
            t_video: (n_frames,) Video frame timestamps
            X_truth: (n_frames, 5) Ground truth state [x, y, vx, vy, θ]
            U_imu: (n_frames, 3) IMU measurements [gyro, accel_x, accel_y]
            bias_gyro: (n_frames,) Gyro bias
            bias_accel_x: (n_frames,) Accel X bias
            bias_accel_y: (n_frames,) Accel Y bias
            cam_idx: (n_frames,) Indices into camera arrays (nearest-neighbor)
            fps: Target fps
            n_frames: Total number of frames

    Note:
        - Position/velocity: linear interpolation
        - Heading: angle-aware interpolation (wraps at ±π)
        - Camera events (dropouts, swaps): nearest-neighbor (discrete)
    """
    # Determine video timeline using arange (not linspace) to avoid off-by-one
    # linspace includes endpoint, giving n_frames-1 intervals → wrong fps
    t_start = 0.0
    t_end = float(sim_data["t_imu"][-1])
    dt = speedup / fps  # Time step per frame
    t_video = np.arange(t_start, t_end + 1e-9, dt)
    n_frames = len(t_video)

    # Interpolate IMU measurements (linear)
    U_imu = np.column_stack(
        [np.interp(t_video, sim_data["t_imu"], sim_data["U_imu"][:, i]) for i in range(3)]
    )

    # Interpolate biases (linear)
    bias_gyro = np.interp(t_video, sim_data["t_imu"], sim_data["bias_gyro"])
    bias_accel_x = np.interp(t_video, sim_data["t_imu"], sim_data["bias_accel_x"])
    bias_accel_y = np.interp(t_video, sim_data["t_imu"], sim_data["bias_accel_y"])

    # Interpolate ground truth state
    # Position and velocity: linear interpolation
    X_truth = np.column_stack(
        [np.interp(t_video, sim_data["t_imu"], sim_data["X_truth"][:, i]) for i in range(4)]
        + [
            # Heading: angle-aware interpolation
            interp_angle(t_video, sim_data["t_imu"], sim_data["X_truth"][:, 4])
        ]
    )

    # Camera data: true nearest-neighbor for discrete events
    # Find nearest camera frame for each video frame (not just previous)
    idx = np.searchsorted(sim_data["t_cam_exp"], t_video)
    idx0 = np.clip(idx - 1, 0, len(sim_data["t_cam_exp"]) - 1)
    idx1 = np.clip(idx, 0, len(sim_data["t_cam_exp"]) - 1)
    left = sim_data["t_cam_exp"][idx0]
    right = sim_data["t_cam_exp"][idx1]
    cam_idx = np.where(np.abs(t_video - left) <= np.abs(right - t_video), idx0, idx1)

    return {
        "t_video": t_video,
        "X_truth": X_truth,
        "U_imu": U_imu,
        "bias_gyro": bias_gyro,
        "bias_accel_x": bias_accel_x,
        "bias_accel_y": bias_accel_y,
        "cam_idx": cam_idx,
        "fps": fps,
        "n_frames": n_frames,
    }
