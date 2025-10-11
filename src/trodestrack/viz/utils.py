"""Utilities for video generation: frame interpolation, time sync."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from trodestrack.sim.utils import SimOut, interp_angle


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


class VideoData(TypedDict):
    t_video: FloatArray
    X_truth: FloatArray
    U_imu: FloatArray
    bias_gyro: FloatArray
    bias_accel_x: FloatArray
    bias_accel_y: FloatArray
    cam_idx: IntArray
    fps: int
    n_frames: int


def prepare_video_data(sim_data: SimOut, fps: int = 30, speedup: float = 1.0) -> VideoData:
    """Interpolate simulation data to video frame times.

    Handles differing sampling rates for IMU (e.g., 200 Hz), camera (e.g., 30 Hz),
    and target video frame rate.

    Parameters
    ----------
    sim_data : SimOut
        Simulation output dictionary (from sim module).
    fps : int, default 30
        Target video frame rate (frames per second).
    speedup : float, default 1.0
        Playback speed multiplier (>1 = faster, <1 = slower).

    Returns
    -------
    dict
        Interpolated data at video frame times with keys:
        - t_video: (n_frames,) timestamps
        - X_truth: (n_frames, 5) [x, y, vx, vy, θ]
        - U_imu: (n_frames, 3) [ω_z, a_x, a_y]
        - bias_gyro, bias_accel_x, bias_accel_y: (n_frames,)
        - cam_idx: (n_frames,) nearest camera indices
        - fps: int, n_frames: int

    Notes
    -----
    - Position/velocity: linear interpolation
    - Heading: unwrap → interp → rewrap
    - Camera events (dropouts, swaps): nearest-neighbor
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
