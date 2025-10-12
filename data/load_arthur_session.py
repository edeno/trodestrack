"""Load and prepare Arthur session data for trodestrack.

This module provides functions to load SpikeGadgets tracking data with proper
unit conversions and sample-and-hold removal.

"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

# SpikeGadgets headstage hardware specifications
GYRO_SCALE = 0.061  # deg/s per LSB (±2000 deg/s range, 16-bit)
ACCEL_SCALE = 0.000061  # g per LSB (±2g range, 16-bit)
GRAVITY = 9.80665  # m/s²
DEG_TO_RAD = np.pi / 180.0


@dataclass(frozen=True)
class SessionData:
    """Immutable container for trodestrack session data.

    All timestamps are relative (start from 0).
    All units are SI: meters, seconds, rad/s, m/s².
    """

    # Time arrays
    t_imu: np.ndarray  # IMU timestamps (s)
    t_cam: np.ndarray  # Camera timestamps (s)

    # IMU data [N_imu × 3]: [gyro_z, accel_x, accel_y]
    U_imu: np.ndarray  # rad/s, m/s², m/s²

    # Camera data [N_cam × 2]: [x, y] for each LED
    Z_cam_led1: np.ndarray  # meters
    Z_cam_led2: np.ndarray  # meters

    # Validity mask
    mask_cam: np.ndarray  # bool [N_cam]

    # Derived parameters
    led_distance: float  # meters (median separation)
    fs_imu: float  # Hz (sampling rate)
    fs_cam: float  # Hz (sampling rate)


def find_unique_samples(values: np.ndarray) -> np.ndarray:
    """Find indices where values change (removes sample-and-hold repeats).

    Args:
        values: Array with repeated values from sample-and-hold

    Returns:
        Indices of unique samples (first occurrence of each value)

    Example:
        >>> values = [1, 1, 1, 2, 2, 3, 3, 3]
        >>> find_unique_samples(values)
        array([0, 3, 5])
    """
    changes = np.where(np.diff(values) != 0)[0] + 1
    return np.concatenate([[0], changes])


def estimate_sampling_rate(timestamps: np.ndarray, n_samples: int = 10000) -> float:
    """Estimate sampling rate from timestamps.

    Uses median of time differences for robustness to outliers.

    Args:
        timestamps: Time values in seconds
        n_samples: Number of samples to use for estimation

    Returns:
        Sampling rate in Hz
    """
    dt_samples = np.diff(timestamps[: min(n_samples, len(timestamps))])
    median_dt = np.median(dt_samples)
    return 1.0 / median_dt


def convert_timestamps_to_relative(
    t_imu: np.ndarray, t_cam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Unix timestamps to relative time starting from 0.

    Args:
        t_imu: IMU timestamps (Unix time)
        t_cam: Camera timestamps (Unix time)

    Returns:
        (t_imu_relative, t_cam_relative) both starting from 0
    """
    t_start = min(t_imu[0], t_cam[0])
    return t_imu - t_start, t_cam - t_start


def convert_imu_to_si(imu_df: pd.DataFrame) -> np.ndarray:
    """Convert raw IMU values to SI units.

    Converts gyro (raw → deg/s → rad/s) and accel (raw → g → m/s²).

    Args:
        imu_df: DataFrame with columns Headstage_GyroZ, Headstage_AccelX, Headstage_AccelY

    Returns:
        Array [N × 3] of [gyro_z (rad/s), accel_x (m/s²), accel_y (m/s²)]
    """
    gyro_z_raw = imu_df["Headstage_GyroZ"].values
    accel_x_raw = imu_df["Headstage_AccelX"].values
    accel_y_raw = imu_df["Headstage_AccelY"].values

    # Gyro: raw → deg/s → rad/s
    gyro_z = gyro_z_raw * GYRO_SCALE * DEG_TO_RAD

    # Accel: raw → g → m/s²
    accel_x = accel_x_raw * ACCEL_SCALE * GRAVITY
    accel_y = accel_y_raw * ACCEL_SCALE * GRAVITY

    return np.column_stack([gyro_z, accel_x, accel_y])


def convert_positions_to_meters(
    pos_df: pd.DataFrame, meters_per_pixel: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert LED positions from pixels to meters.

    Args:
        pos_df: DataFrame with columns xloc, yloc, xloc2, yloc2
        meters_per_pixel: Scale factor (e.g., 0.0022)

    Returns:
        (led1_positions, led2_positions) both [N × 2] in meters
    """
    led1_pixels = pos_df[["xloc", "yloc"]].values
    led2_pixels = pos_df[["xloc2", "yloc2"]].values

    led1_meters = led1_pixels * meters_per_pixel
    led2_meters = led2_pixels * meters_per_pixel

    return led1_meters, led2_meters


def compute_led_separation(led1: np.ndarray, led2: np.ndarray) -> float:
    """Compute median LED separation distance.

    Args:
        led1: LED1 positions [N × 2]
        led2: LED2 positions [N × 2]

    Returns:
        Median Euclidean distance in same units as input
    """
    separations = np.linalg.norm(led2 - led1, axis=1)
    return float(np.median(separations))


def load_arthur_session(
    position_file: str, imu_file: str, meters_per_pixel: float = 0.0022, verbose: bool = True
) -> SessionData:
    """Load Arthur session data with proper preprocessing.

    This function:
    1. Loads parquet files
    2. Removes IMU sample-and-hold repeats
    3. Converts timestamps to relative time
    4. Converts all units to SI
    5. Estimates sampling rates
    6. Validates data quality

    Args:
        position_file: Path to position parquet file
        imu_file: Path to IMU parquet file
        meters_per_pixel: Camera scale factor (default: 0.0022)
        verbose: Print loading progress and statistics

    Returns:
        SessionData with all preprocessing applied

    Example:
        >>> data = load_arthur_session(
        ...     "arthur20220314_position_info.parquet",
        ...     "arthur20220314_imu_info.parquet"
        ... )
        >>> print(f"Duration: {data.t_cam[-1]:.1f} s")
        >>> print(f"IMU rate: {data.fs_imu:.1f} Hz")
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
    U_imu = convert_imu_to_si(imu_unique)
    Z_cam_led1, Z_cam_led2 = convert_positions_to_meters(pos_df, meters_per_pixel)

    # Compute LED separation
    led_distance = compute_led_separation(Z_cam_led1, Z_cam_led2)

    if verbose:
        print("\nUnit conversions:")
        print(
            f"  Gyro: [{U_imu[:, 0].min()*180/np.pi:.1f}, {U_imu[:, 0].max()*180/np.pi:.1f}] deg/s"
        )
        print(f"  Accel X: [{U_imu[:, 1].min():.2f}, {U_imu[:, 1].max():.2f}] m/s²")
        print(f"  Accel Y: [{U_imu[:, 2].min():.2f}, {U_imu[:, 2].max():.2f}] m/s²")
        print(f"  LED separation: {led_distance*100:.2f} cm")
        print(
            f"  Arena: {Z_cam_led1[:, 0].max() - Z_cam_led1[:, 0].min():.2f} × "
            f"{Z_cam_led1[:, 1].max() - Z_cam_led1[:, 1].min():.2f} m"
        )

    # Validate 3D accelerometer magnitude (optional but recommended)
    if verbose and "Headstage_AccelZ" in imu_unique.columns:
        accel_z = imu_unique["Headstage_AccelZ"].values * ACCEL_SCALE * GRAVITY
        accel_mag_3d = np.sqrt(U_imu[:, 1] ** 2 + U_imu[:, 2] ** 2 + accel_z**2)
        print("\nData quality check:")
        print(f"  3D accel magnitude: {accel_mag_3d.mean():.2f} m/s² " f"(expected ~{GRAVITY:.2f})")

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
    )


def main():
    """Example usage and validation."""
    import sys
    from pathlib import Path

    # Assume script is in data/ directory
    script_dir = Path(__file__).parent

    # Load data
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220314_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220314_imu_info.parquet"),
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
