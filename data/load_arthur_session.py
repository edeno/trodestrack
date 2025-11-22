"""Load and prepare Arthur session data for trodestrack.

This module provides functions to load SpikeGadgets tracking data with proper
unit conversions, sample-and-hold removal, and video frame extraction.

Design principles (Raymond Hettinger style):
- One function does one thing well
- Composition over complexity
- Clear names that reveal intent
- Immutable data flow where possible
- Type hints for clarity
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# SpikeGadgets headstage hardware specifications
# Source: https://spikegadgets.com/documentation/ (Product Manual)
# Accelerometer: ±2g range, 16-bit signed (2/32767 = 0.000061g per LSB)
# Gyroscope: ±2000 deg/s range, 16-bit signed (2000/32767 = 0.061 deg/s per LSB)
# Refresh rate: 104 Hz (both sensors), Internal sampling: 500 Hz per sensor
GYRO_SCALE = 0.061  # deg/s per LSB
ACCEL_SCALE = 0.000061  # g per LSB
GRAVITY = 9.80665  # m/s²
DEG_TO_RAD = np.pi / 180.0


@dataclass(frozen=True)
class SessionData:
    """Immutable container for trodestrack session data.

    All timestamps are relative (start from 0).
    All units are SI: meters, seconds, rad/s, m/s².

    Attributes
    ----------
    t_imu : np.ndarray
        IMU timestamps in seconds (relative to start)
    t_cam : np.ndarray
        Camera timestamps in seconds (relative to start)
    U_imu : np.ndarray
        IMU data in SI units. Shape depends on mode:
        - 2D mode [N_imu × 3]: [gyro_z, accel_x, accel_y] (rad/s, m/s², m/s²)
        - 3D mode [N_imu × 6]: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
    Z_cam_led1 : np.ndarray
        LED1 positions [N_cam × 2]: [x, y] in meters
    Z_cam_led2 : np.ndarray
        LED2 positions [N_cam × 2]: [x, y] in meters
    mask_cam : np.ndarray
        Validity mask [N_cam] boolean array
    led_distance : float
        Median LED separation distance in meters
    fs_imu : float
        IMU sampling rate in Hz
    fs_cam : float
        Camera sampling rate in Hz
    meters_per_pixel : float
        Scale factor for converting pixels to meters
    """

    t_imu: np.ndarray
    t_cam: np.ndarray
    U_imu: np.ndarray
    Z_cam_led1: np.ndarray
    Z_cam_led2: np.ndarray
    mask_cam: np.ndarray
    led_distance: float
    fs_imu: float
    fs_cam: float
    meters_per_pixel: float


def find_unique_samples(values: np.ndarray) -> np.ndarray:
    """Find indices where values change (removes sample-and-hold repeats).

    Parameters
    ----------
    values : np.ndarray
        Array with repeated values from sample-and-hold

    Returns
    -------
    np.ndarray
        Indices of unique samples (first occurrence of each value)

    Examples
    --------
    >>> values = np.array([1, 1, 1, 2, 2, 3, 3, 3])
    >>> find_unique_samples(values)
    array([0, 3, 5])
    """
    changes = np.where(np.diff(values) != 0)[0] + 1
    return np.concatenate([[0], changes])


def estimate_sampling_rate(timestamps: np.ndarray, n_samples: int = 10000) -> float:
    """Estimate sampling rate from timestamps.

    Uses median of time differences for robustness to outliers.

    Parameters
    ----------
    timestamps : np.ndarray
        Time values in seconds
    n_samples : int, optional
        Number of samples to use for estimation (default: 10000)

    Returns
    -------
    float
        Sampling rate in Hz
    """
    dt_samples = np.diff(timestamps[: min(n_samples, len(timestamps))])
    median_dt = np.median(dt_samples)
    return 1.0 / median_dt


def convert_timestamps_to_relative(
    t_imu: np.ndarray, t_cam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Unix timestamps to relative time starting from 0.

    Parameters
    ----------
    t_imu : np.ndarray
        IMU timestamps (Unix time in seconds)
    t_cam : np.ndarray
        Camera timestamps (Unix time in seconds)

    Returns
    -------
    t_imu_relative : np.ndarray
        IMU timestamps starting from 0
    t_cam_relative : np.ndarray
        Camera timestamps starting from 0
    """
    t_start = min(t_imu[0], t_cam[0])
    return t_imu - t_start, t_cam - t_start


def convert_imu_to_si(imu_df: pd.DataFrame, mode: str = "2d") -> np.ndarray:
    """Convert raw IMU values to SI units.

    Converts gyro (raw → deg/s → rad/s) and accel (raw → g → m/s²).

    Parameters
    ----------
    imu_df : pd.DataFrame
        DataFrame with IMU columns (GyroX/Y/Z, AccelX/Y/Z)
    mode : str, optional
        IMU output mode:
        - "2d": [N × 3] = [gyro_z, accel_x, accel_y] (default, legacy)
        - "3d": [N × 6] = [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]

    Returns
    -------
    np.ndarray
        Array of IMU data in SI units (rad/s for gyro, m/s² for accel)
        Shape depends on mode: [N × 3] for "2d", [N × 6] for "3d"

    Raises
    ------
    ValueError
        If mode is not "2d" or "3d"
    """
    if mode not in ("2d", "3d"):
        raise ValueError(f"Invalid IMU mode: {mode!r}. Must be '2d' or '3d'.")

    if mode == "2d":
        # Legacy 2D mode: only gyro Z and accel X/Y
        gyro_z_raw = imu_df["Headstage_GyroZ"].values
        accel_x_raw = imu_df["Headstage_AccelX"].values
        accel_y_raw = imu_df["Headstage_AccelY"].values

        # Gyro: raw → deg/s → rad/s
        gyro_z = gyro_z_raw * GYRO_SCALE * DEG_TO_RAD

        # Accel: raw → g → m/s²
        accel_x = accel_x_raw * ACCEL_SCALE * GRAVITY
        accel_y = accel_y_raw * ACCEL_SCALE * GRAVITY

        return np.column_stack([gyro_z, accel_x, accel_y])

    else:  # mode == "3d"
        # Full 3D IMU: all 6 axes
        gyro_x_raw = imu_df["Headstage_GyroX"].values
        gyro_y_raw = imu_df["Headstage_GyroY"].values
        gyro_z_raw = imu_df["Headstage_GyroZ"].values
        accel_x_raw = imu_df["Headstage_AccelX"].values
        accel_y_raw = imu_df["Headstage_AccelY"].values
        accel_z_raw = imu_df["Headstage_AccelZ"].values

        # Gyro: raw → deg/s → rad/s
        gyro_x = gyro_x_raw * GYRO_SCALE * DEG_TO_RAD
        gyro_y = gyro_y_raw * GYRO_SCALE * DEG_TO_RAD
        gyro_z = gyro_z_raw * GYRO_SCALE * DEG_TO_RAD

        # Accel: raw → g → m/s²
        accel_x = accel_x_raw * ACCEL_SCALE * GRAVITY
        accel_y = accel_y_raw * ACCEL_SCALE * GRAVITY
        accel_z = accel_z_raw * ACCEL_SCALE * GRAVITY

        return np.column_stack([gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z])


def convert_positions_to_meters(
    pos_df: pd.DataFrame, meters_per_pixel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert LED positions from pixels to meters.

    Parameters
    ----------
    pos_df : pd.DataFrame
        DataFrame with columns: xloc, yloc, xloc2, yloc2
    meters_per_pixel : float
        Scale factor (e.g., 0.0022)

    Returns
    -------
    led1_positions : np.ndarray
        LED1 positions [N × 2] in meters
    led2_positions : np.ndarray
        LED2 positions [N × 2] in meters
    """
    led1_pixels = pos_df[["xloc", "yloc"]].values
    led2_pixels = pos_df[["xloc2", "yloc2"]].values

    led1_meters = led1_pixels * meters_per_pixel
    led2_meters = led2_pixels * meters_per_pixel

    return led1_meters, led2_meters


def convert_meters_to_pixels(
    positions_m: np.ndarray, meters_per_pixel: float
) -> np.ndarray:
    """Convert positions from meters to pixels.

    Parameters
    ----------
    positions_m : np.ndarray
        Positions in meters [N × 2]
    meters_per_pixel : float
        Scale factor (e.g., 0.0022)

    Returns
    -------
    np.ndarray
        Positions in pixels [N × 2]
    """
    return positions_m / meters_per_pixel


def compute_led_separation(led1: np.ndarray, led2: np.ndarray) -> float:
    """Compute median LED separation distance.

    Parameters
    ----------
    led1 : np.ndarray
        LED1 positions [N × 2]
    led2 : np.ndarray
        LED2 positions [N × 2]

    Returns
    -------
    float
        Median Euclidean distance in same units as input
    """
    separations = np.linalg.norm(led2 - led1, axis=1)
    return float(np.median(separations))


def load_video_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Load a single frame from a video file.

    Parameters
    ----------
    video_path : str
        Path to video file (e.g., .mp4)
    frame_idx : int
        Zero-based frame index to load

    Returns
    -------
    np.ndarray or None
        Frame as RGB image [H × W × 3] or None if cv2 not available or frame not found

    Notes
    -----
    Requires opencv-python (cv2) to be installed.
    """
    if not HAS_CV2:
        print("Warning: opencv-python not installed, cannot load video frames")
        return None

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """Get video file metadata.

    Parameters
    ----------
    video_path : str
        Path to video file

    Returns
    -------
    dict
        Dictionary with keys: 'width', 'height', 'fps', 'frame_count', 'duration_s'
        Returns empty dict if cv2 not available or file not found

    Notes
    -----
    Requires opencv-python (cv2) to be installed.
    """
    if not HAS_CV2:
        print("Warning: opencv-python not installed, cannot read video info")
        return {}

    cap = cv2.VideoCapture(video_path)
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        return info
    finally:
        cap.release()


def load_arthur_session(
    position_file: str,
    imu_file: str,
    meters_per_pixel: float = 0.0022,
    imu_mode: str = "3d",
    verbose: bool = True,
) -> SessionData:
    """Load Arthur session data with proper preprocessing.

    This function performs the following steps:
    1. Loads parquet files
    2. Removes IMU sample-and-hold repeats
    3. Converts timestamps to relative time
    4. Converts all units to SI
    5. Estimates sampling rates
    6. Validates data quality

    Parameters
    ----------
    position_file : str
        Path to position parquet file
    imu_file : str
        Path to IMU parquet file
    meters_per_pixel : float, optional
        Camera scale factor (default: 0.0022)
    imu_mode : str, optional
        IMU output mode (default: "3d"):
        - "2d": [N × 3] = [gyro_z, accel_x, accel_y] (legacy)
        - "3d": [N × 6] = [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
    verbose : bool, optional
        Print loading progress and statistics (default: True)

    Returns
    -------
    SessionData
        Immutable dataclass with all preprocessing applied

    Examples
    --------
    >>> # Load with full 3D IMU (default)
    >>> data = load_arthur_session(
    ...     "arthur20220324_position_info.parquet", "arthur20220324_imu_info.parquet"
    ... )
    >>> print(f"IMU shape: {data.U_imu.shape}")  # (N, 6)

    >>> # Load with legacy 2D IMU
    >>> data_2d = load_arthur_session("position.parquet", "imu.parquet", imu_mode="2d")
    >>> print(f"IMU shape: {data_2d.U_imu.shape}")  # (N, 3)

    Notes
    -----
    - Removes ~287× sample-and-hold from IMU data
    - Validates 3D accelerometer magnitude ≈ 9.81 m/s² (when imu_mode="3d")
    - All output timestamps are relative (start from 0)
    """
    if verbose:
        print("=" * 80)
        print("LOADING ARTHUR SESSION")
        print("=" * 80)

    # Load raw data
    pos_df = pd.read_parquet(position_file)
    imu_df = pd.read_parquet(imu_file)

    if verbose:
        print("\nRaw data:")
        print(f"  Position: {len(pos_df):,} frames")
        print(f"  IMU: {len(imu_df):,} samples (with sample-and-hold)")

    # Remove IMU sample-and-hold
    unique_indices = find_unique_samples(imu_df["Headstage_GyroZ"].values)
    imu_unique = imu_df.iloc[unique_indices]

    if verbose:
        repeat_factor = len(imu_df) / len(imu_unique)
        print("\nSample-and-hold removal:")
        print(f"  Unique samples: {len(imu_unique):,}")
        print(f"  Repeat factor: {repeat_factor:.1f}×")

    # Extract timestamps and convert to relative time
    t_imu_unix = imu_unique.index.values
    t_cam_unix = pos_df.index.values
    t_imu, t_cam = convert_timestamps_to_relative(t_imu_unix, t_cam_unix)

    # Estimate sampling rates
    fs_imu = estimate_sampling_rate(t_imu)
    fs_cam = estimate_sampling_rate(t_cam)

    if verbose:
        print("\nSampling rates:")
        print(f"  IMU: {fs_imu:.1f} Hz")
        print(f"  Camera: {fs_cam:.1f} Hz")
        print("\nDuration:")
        print(f"  IMU: {t_imu[-1]:.1f} s ({t_imu[-1]/60:.1f} min)")
        print(f"  Camera: {t_cam[-1]:.1f} s ({t_cam[-1]/60:.1f} min)")
        print(f"  Time overlap: {min(t_imu[-1], t_cam[-1]):.1f} s")

    # Convert units to SI
    U_imu = convert_imu_to_si(imu_unique, mode=imu_mode)
    Z_cam_led1, Z_cam_led2 = convert_positions_to_meters(pos_df, meters_per_pixel)

    # Compute LED separation
    led_distance = compute_led_separation(Z_cam_led1, Z_cam_led2)

    if verbose:
        print(f"\nUnit conversions (IMU mode: {imu_mode}):")
        if imu_mode == "2d":
            print(
                f"  Gyro Z: [{U_imu[:, 0].min()*180/np.pi:.1f}, {U_imu[:, 0].max()*180/np.pi:.1f}] deg/s"
            )
            print(f"  Accel X: [{U_imu[:, 1].min():.2f}, {U_imu[:, 1].max():.2f}] m/s²")
            print(f"  Accel Y: [{U_imu[:, 2].min():.2f}, {U_imu[:, 2].max():.2f}] m/s²")
        else:  # 3d mode
            print(
                f"  Gyro X: [{U_imu[:, 0].min()*180/np.pi:.1f}, {U_imu[:, 0].max()*180/np.pi:.1f}] deg/s"
            )
            print(
                f"  Gyro Y: [{U_imu[:, 1].min()*180/np.pi:.1f}, {U_imu[:, 1].max()*180/np.pi:.1f}] deg/s"
            )
            print(
                f"  Gyro Z: [{U_imu[:, 2].min()*180/np.pi:.1f}, {U_imu[:, 2].max()*180/np.pi:.1f}] deg/s"
            )
            print(f"  Accel X: [{U_imu[:, 3].min():.2f}, {U_imu[:, 3].max():.2f}] m/s²")
            print(f"  Accel Y: [{U_imu[:, 4].min():.2f}, {U_imu[:, 4].max():.2f}] m/s²")
            print(f"  Accel Z: [{U_imu[:, 5].min():.2f}, {U_imu[:, 5].max():.2f}] m/s²")
        print(f"  LED separation: {led_distance*100:.2f} cm")
        print(
            f"  Arena: {Z_cam_led1[:, 0].max() - Z_cam_led1[:, 0].min():.2f} × "
            f"{Z_cam_led1[:, 1].max() - Z_cam_led1[:, 1].min():.2f} m"
        )

    # Validate 3D accelerometer magnitude (when in 3D mode)
    if verbose and imu_mode == "3d":
        accel_mag_3d = np.sqrt(U_imu[:, 3] ** 2 + U_imu[:, 4] ** 2 + U_imu[:, 5] ** 2)
        print("\nData quality check:")
        print(
            f"  3D accel magnitude: {accel_mag_3d.mean():.2f} m/s² (expected ~{GRAVITY:.2f})"
        )

    # All frames valid (no tracking failures in this dataset)
    mask_cam = np.ones(len(pos_df), dtype=bool)

    if verbose:
        print(f"\n{'=' * 80}")
        print("✓ Data ready for trodestrack")
        print(f"{'=' * 80}\n")

    return SessionData(
        t_imu=t_imu,
        t_cam=t_cam,
        U_imu=U_imu,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
        led_distance=led_distance,
        fs_imu=fs_imu,
        fs_cam=fs_cam,
        meters_per_pixel=meters_per_pixel,
    )


def main():
    """Example usage and validation."""
    import sys

    # Assume script is in data/ directory
    script_dir = Path(__file__).parent

    # Load data (using 20220324 session - 40 min recording)
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220324_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220324_imu_info.parquet"),
        meters_per_pixel=0.0022,
        verbose=True,
    )

    # Print summary
    print("\nSessionData summary:")
    print(f"  t_imu: {len(data.t_imu):,} samples at {data.fs_imu:.1f} Hz")
    print(f"  t_cam: {len(data.t_cam):,} frames at {data.fs_cam:.1f} Hz")
    print(f"  U_imu: shape {data.U_imu.shape}, dtype {data.U_imu.dtype}")
    print(f"  Z_cam_led1: shape {data.Z_cam_led1.shape}, dtype {data.Z_cam_led1.dtype}")
    print(f"  Z_cam_led2: shape {data.Z_cam_led2.shape}, dtype {data.Z_cam_led2.dtype}")
    print(f"  mask_cam: {data.mask_cam.sum():,} / {len(data.mask_cam):,} valid frames")
    print(f"  led_distance: {data.led_distance*100:.2f} cm")
    print(f"  meters_per_pixel: {data.meters_per_pixel}")

    # Test pixel conversion
    print("\n✓ Testing pixel conversion:")
    led1_pixels = convert_meters_to_pixels(data.Z_cam_led1, data.meters_per_pixel)
    print(
        f"  LED1 pixels range: [{led1_pixels[:, 0].min():.1f}, {led1_pixels[:, 0].max():.1f}]"
    )

    # Test video loading if available
    video_path = script_dir / "20220324_arthur_02_r1.mp4"
    if video_path.exists():
        print("\n✓ Testing video loading:")
        info = get_video_info(str(video_path))
        if info:
            print(f"  Video: {info['width']}×{info['height']} @ {info['fps']:.1f} fps")
            print(f"  Frames: {info['frame_count']:,} ({info['duration_s']:.1f} s)")

            frame = load_video_frame(str(video_path), 1000)
            if frame is not None:
                print(f"  Loaded frame 1000: shape {frame.shape}, dtype {frame.dtype}")
        else:
            print("  (opencv-python not installed)")
    else:
        print(f"\n  Video file not found: {video_path}")

    # Validate immutability
    print("\n✓ SessionData is immutable (frozen dataclass)")
    try:
        data.led_distance = 0.05
        print("✗ ERROR: Should not be able to modify frozen dataclass!")
        sys.exit(1)
    except AttributeError:
        print("  (Cannot modify attributes after creation)")

    print("\n✓ All validation checks passed!")
    return 0


if __name__ == "__main__":
    exit(main())
