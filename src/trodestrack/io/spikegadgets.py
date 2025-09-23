"""SpikeGadgets IMU data loader."""

import logging
import struct
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..constants import (
    DEGREES_TO_RADIANS,
    SPIKEGADGETS_ACCEL_SCALE_FACTOR_G_PER_LSB,
    SPIKEGADGETS_BASIC_RECORD_SIZE,
    SPIKEGADGETS_DEFAULT_CLOCK_RATE_HZ,
    SPIKEGADGETS_MAG_RECORD_SIZE,
    STANDARD_GRAVITY_MS2,
)

logger = logging.getLogger(__name__)


class SpikeGadgetsIMUData:
    """Container for SpikeGadgets IMU data."""

    def __init__(
        self,
        timestamps: np.ndarray,
        accel_raw: np.ndarray,
        gyro_raw: np.ndarray,
        mag_raw: Optional[np.ndarray] = None,
        sampling_rate: float = SPIKEGADGETS_DEFAULT_CLOCK_RATE_HZ,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SpikeGadgets IMU data container.

        Parameters
        ----------
        timestamps : np.ndarray, shape (n_samples,)
            Sample timestamps in seconds
        accel_raw : np.ndarray, shape (n_samples, 3)
            Raw accelerometer data [x, y, z] in raw counts
        gyro_raw : np.ndarray, shape (n_samples, 3)
            Raw gyroscope data [x, y, z] in raw counts
        mag_raw : np.ndarray, shape (n_samples, 3), optional
            Raw magnetometer data [x, y, z] in raw counts. Default is None.
        sampling_rate : float, optional
            IMU sampling rate in Hz. Default is 30000.0.
        metadata : dict, optional
            Optional metadata dictionary
        """
        self.timestamps = np.asarray(timestamps)
        self.accel_raw = np.asarray(accel_raw)
        self.gyro_raw = np.asarray(gyro_raw)
        self.mag_raw = np.asarray(mag_raw) if mag_raw is not None else None
        self.sampling_rate = sampling_rate
        self.metadata = metadata or {}

        # Validate shapes
        n_samples = len(self.timestamps)
        if len(self.accel_raw) != n_samples:
            raise ValueError("Accelerometer data length mismatch")
        if len(self.gyro_raw) != n_samples:
            raise ValueError("Gyroscope data length mismatch")
        if self.mag_raw is not None and len(self.mag_raw) != n_samples:
            raise ValueError("Magnetometer data length mismatch")

        if self.accel_raw.shape != (n_samples, 3):
            raise ValueError("Accelerometer data must be (N, 3) array")
        if self.gyro_raw.shape != (n_samples, 3):
            raise ValueError("Gyroscope data must be (N, 3) array")
        if self.mag_raw is not None and self.mag_raw.shape != (n_samples, 3):
            raise ValueError("Magnetometer data must be (N, 3) array")

    @property
    def n_samples(self) -> int:
        """Number of IMU samples."""
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        """Duration of the recording in seconds."""
        if self.n_samples == 0:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def has_magnetometer(self) -> bool:
        """Whether magnetometer data is available."""
        return self.mag_raw is not None

    def get_accel_g(
        self, scale_factor: float = SPIKEGADGETS_ACCEL_SCALE_FACTOR_G_PER_LSB
    ) -> np.ndarray:
        """Convert raw accelerometer measurements to g units.

        Parameters
        ----------
        scale_factor : float, optional
            Conversion factor from raw counts to g. Default is 0.000061 g/count
            for typical SpikeGadgets configuration.

        Returns
        -------
        np.ndarray, shape (n_samples, 3)
            Accelerometer measurements in g units [ax, ay, az]

        Notes
        -----
        Standard Earth gravity is approximately 9.80665 m/s².
        """
        return self.accel_raw * scale_factor

    def get_accel_ms2(self, scale_factor: float = 0.000061) -> np.ndarray:
        """Convert raw accelerometer measurements to m/s² units.

        Parameters
        ----------
        scale_factor : float, optional
            Conversion factor from raw counts to g. Default is 0.000061 g/count
            for typical SpikeGadgets configuration.

        Returns
        -------
        np.ndarray, shape (n_samples, 3)
            Accelerometer measurements in m/s² units [ax, ay, az]

        Notes
        -----
        Conversion: raw * scale_factor * 9.80665 (standard gravity).
        """
        g_to_ms2 = STANDARD_GRAVITY_MS2
        return self.get_accel_g(scale_factor) * g_to_ms2

    def get_gyro_deg_s(self, scale_factor: float = 0.061) -> np.ndarray:
        """Convert raw gyroscope measurements to degrees/second.

        Parameters
        ----------
        scale_factor : float, optional
            Conversion factor from raw counts to deg/s. Default is 0.061 deg/s/count
            for typical SpikeGadgets configuration.

        Returns
        -------
        np.ndarray, shape (n_samples, 3)
            Gyroscope measurements in degrees/second [ωx, ωy, ωz]
        """
        return self.gyro_raw * scale_factor

    def get_gyro_rad_s(self, scale_factor: float = 0.061) -> np.ndarray:
        """Convert raw gyroscope measurements to radians/second.

        Parameters
        ----------
        scale_factor : float, optional
            Conversion factor from raw counts to deg/s. Default is 0.061 deg/s/count
            for typical SpikeGadgets configuration.

        Returns
        -------
        np.ndarray, shape (n_samples, 3)
            Gyroscope measurements in radians/second [ωx, ωy, ωz]

        Notes
        -----
        Conversion: raw * scale_factor * π/180 (degrees to radians).
        """
        deg_to_rad = DEGREES_TO_RADIANS
        return self.get_gyro_deg_s(scale_factor) * deg_to_rad

    def downsample(self, target_rate: float) -> "SpikeGadgetsIMUData":
        """Downsample IMU data to target rate.

        Parameters
        ----------
        target_rate : float
            Target sampling rate in Hz

        Returns
        -------
        SpikeGadgetsIMUData
            New SpikeGadgetsIMUData instance with downsampled data

        Notes
        -----
        Uses simple decimation by integer factor. Updates metadata
        with decimation information.
        """
        if target_rate >= self.sampling_rate:
            logger.warning("Target rate >= current rate, returning original data")
            return self

        # Calculate decimation factor
        decimation_factor = int(self.sampling_rate / target_rate)

        # Downsample by taking every nth sample
        indices = np.arange(0, self.n_samples, decimation_factor)

        downsampled_timestamps = self.timestamps[indices]
        downsampled_accel = self.accel_raw[indices]
        downsampled_gyro = self.gyro_raw[indices]
        downsampled_mag = self.mag_raw[indices] if self.has_magnetometer else None

        # Update metadata
        new_metadata = self.metadata.copy()
        new_metadata.update(
            {
                "original_rate": self.sampling_rate,
                "decimation_factor": decimation_factor,
                "effective_rate": self.sampling_rate / decimation_factor,
            }
        )

        return SpikeGadgetsIMUData(
            timestamps=downsampled_timestamps,
            accel_raw=downsampled_accel,
            gyro_raw=downsampled_gyro,
            mag_raw=downsampled_mag,
            sampling_rate=target_rate,
            metadata=new_metadata,
        )

    def get_time_range(self, start_time: float, end_time: float) -> "SpikeGadgetsIMUData":
        """Extract data within specified time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            New SpikeGadgetsIMUData instance with time-sliced data
        """
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise ValueError("No data found in specified time range")

        sliced_timestamps = self.timestamps[indices]
        sliced_accel = self.accel_raw[indices]
        sliced_gyro = self.gyro_raw[indices]
        sliced_mag = self.mag_raw[indices] if self.has_magnetometer else None

        # Update metadata
        new_metadata = self.metadata.copy()
        new_metadata.update(
            {"time_slice": (start_time, end_time), "original_n_samples": self.n_samples}
        )

        return SpikeGadgetsIMUData(
            timestamps=sliced_timestamps,
            accel_raw=sliced_accel,
            gyro_raw=sliced_gyro,
            mag_raw=sliced_mag,
            sampling_rate=self.sampling_rate,
            metadata=new_metadata,
        )


# Constants for data validation
INT16_MIN = -32768
INT16_MAX = 32767
SATURATION_THRESHOLD = 0.95  # Warn if >95% of range is used


def _validate_imu_data_ranges(
    accel_data: np.ndarray, gyro_data: np.ndarray, mag_data: Optional[np.ndarray] = None
) -> None:
    """Validate IMU data ranges and check for saturation/overflow.

    Parameters
    ----------
    accel_data : np.ndarray, shape (n_samples, 3)
        Raw accelerometer data
    gyro_data : np.ndarray, shape (n_samples, 3)
        Raw gyroscope data
    mag_data : np.ndarray, shape (n_samples, 3), optional
        Raw magnetometer data

    Raises
    ------
    ValueError
        If data contains values outside int16 range

    Warns
    -----
    UserWarning
        If data appears saturated (near int16 limits)
    """

    def check_saturation(data: np.ndarray, name: str) -> None:
        """Check for saturation in sensor data."""
        min_val, max_val = np.min(data), np.max(data)

        # Check for overflow
        if min_val < INT16_MIN or max_val > INT16_MAX:
            raise ValueError(
                f"{name} data contains values outside int16 range: [{min_val}, {max_val}]"
            )

        # Check for saturation
        saturation_min = INT16_MIN * SATURATION_THRESHOLD
        saturation_max = INT16_MAX * SATURATION_THRESHOLD

        if min_val <= saturation_min or max_val >= saturation_max:
            saturated_samples = np.sum((data <= saturation_min) | (data >= saturation_max))
            logger.warning(
                "%s data may be saturated: %d samples near int16 limits "
                "(range: [%d, %d], limits: [%d, %d])",
                name, saturated_samples, min_val, max_val, INT16_MIN, INT16_MAX
            )

    check_saturation(accel_data, "Accelerometer")
    check_saturation(gyro_data, "Gyroscope")
    if mag_data is not None:
        check_saturation(mag_data, "Magnetometer")


def load_spikegadgets_binary(file_path: Path) -> SpikeGadgetsIMUData:
    """Load SpikeGadgets IMU data from binary file.

    Expected binary format:
    - 4-byte timestamp (uint32)
    - 6 x 2-byte IMU values (int16): accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    - Optional: 3 x 2-byte magnetometer values (int16): mag_x, mag_y, mag_z

    Args:
        file_path: Path to SpikeGadgets binary file

    Returns:
        SpikeGadgetsIMUData container

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If binary format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"SpikeGadgets file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            data = f.read()

        # Determine record size based on file size
        # Basic format: 4 bytes timestamp + 6*2 bytes IMU = 16 bytes
        # With magnetometer: 4 bytes timestamp + 9*2 bytes IMU+mag = 22 bytes

        basic_record_size = SPIKEGADGETS_BASIC_RECORD_SIZE
        mag_record_size = SPIKEGADGETS_MAG_RECORD_SIZE

        if len(data) % basic_record_size == 0:
            record_size = basic_record_size
            has_mag = False
        elif len(data) % mag_record_size == 0:
            record_size = mag_record_size
            has_mag = True
        else:
            # Provide detailed error message about file format mismatch
            file_size = len(data)
            basic_remainder = file_size % basic_record_size
            mag_remainder = file_size % mag_record_size
            raise ValueError(
                f"Binary file size ({file_size} bytes) doesn't match expected record format. "
                f"Remainder: {basic_remainder} bytes for basic format (expected 0), "
                f"{mag_remainder} bytes for magnetometer format (expected 0). "
                f"Expected record sizes: {basic_record_size} or {mag_record_size} bytes."
            )

        n_records = len(data) // record_size

        # Validate file integrity
        if n_records == 0:
            raise ValueError(
                f"File too small: contains 0 complete records (file size: {len(data)} bytes, record size: {record_size} bytes)"
            )

        if len(data) != n_records * record_size:
            raise ValueError(
                f"Corrupt IMU file: size not exact multiple of record size (file: {len(data)} bytes, expected: {n_records * record_size} bytes)"
            )

        # Unpack binary data vectorized
        if has_mag:
            # Unpack all records at once: timestamp (uint32) + 9 IMU values (int16)
            format_str = f"<{n_records}I{n_records * 9}h"
            values = struct.unpack(format_str, data)

            # Reshape into records
            timestamps = np.array(values[:n_records], dtype=np.uint32)
            imu_values = np.array(values[n_records:], dtype=np.int16).reshape(n_records, 9)

            accel_data = imu_values[:, 0:3]
            gyro_data = imu_values[:, 3:6]
            mag_data = imu_values[:, 6:9]
        else:
            # Unpack all records at once: timestamp (uint32) + 6 IMU values (int16)
            format_str = f"<{n_records}I{n_records * 6}h"
            values = struct.unpack(format_str, data)

            # Reshape into records
            timestamps = np.array(values[:n_records], dtype=np.uint32)
            imu_values = np.array(values[n_records:], dtype=np.int16).reshape(n_records, 6)

            accel_data = imu_values[:, 0:3]
            gyro_data = imu_values[:, 3:6]
            mag_data = None

        # Validate data ranges and check for potential int16 overflow
        _validate_imu_data_ranges(accel_data, gyro_data, mag_data if has_mag else None)

        # Convert to numpy arrays with explicit dtype
        timestamps = timestamps.astype(np.float64)
        accel_raw = accel_data.astype(np.float64)
        gyro_raw = gyro_data.astype(np.float64)
        mag_raw = mag_data.astype(np.float64) if has_mag else None

        # Convert timestamps from SpikeGadgets units to seconds
        # Use default SpikeGadgets clock rate
        clock_rate = SPIKEGADGETS_DEFAULT_CLOCK_RATE_HZ
        timestamps = timestamps / clock_rate

        # Create metadata
        metadata = {
            "file_path": str(file_path),
            "n_samples": n_records,
            "format": "spikegadgets_binary",
            "has_magnetometer": has_mag,
            "clock_rate": clock_rate,
            "record_size": record_size,
        }

        return SpikeGadgetsIMUData(
            timestamps=timestamps,
            accel_raw=accel_raw,
            gyro_raw=gyro_raw,
            mag_raw=mag_raw,
            sampling_rate=clock_rate,
            metadata=metadata,
        )

    except struct.error as e:
        raise ValueError(f"Binary format error in SpikeGadgets file: {e}")
    except MemoryError as e:
        raise ValueError(f"File too large to load into memory: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read SpikeGadgets binary file: {e}")


def load_spikegadgets_csv(file_path: Path) -> SpikeGadgetsIMUData:
    """Load SpikeGadgets IMU data from CSV file.

    Expected CSV format:
    - timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    - Optional: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z

    Args:
        file_path: Path to SpikeGadgets CSV file

    Returns:
        SpikeGadgetsIMUData container

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"SpikeGadgets file not found: {file_path}")

    try:
        import pandas as pd

        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Check for required columns
    required_cols = [
        "timestamp",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract data
    timestamps = df["timestamp"].values
    accel_raw = df[["accel_x", "accel_y", "accel_z"]].values
    gyro_raw = df[["gyro_x", "gyro_y", "gyro_z"]].values

    # Check for magnetometer data
    mag_cols = ["mag_x", "mag_y", "mag_z"]
    if all(col in df.columns for col in mag_cols):
        mag_raw = df[mag_cols].values
        has_mag = True
    else:
        mag_raw = None
        has_mag = False

    # Estimate sampling rate
    if len(timestamps) > 1:
        dt = np.median(np.diff(timestamps))
        sampling_rate = float(1.0 / dt)
    else:
        sampling_rate = 30000.0  # Default assumption

    # Create metadata
    metadata = {
        "file_path": str(file_path),
        "n_samples": len(df),
        "format": "spikegadgets_csv",
        "has_magnetometer": has_mag,
        "estimated_rate": sampling_rate,
        "columns": list(df.columns),
    }

    return SpikeGadgetsIMUData(
        timestamps=timestamps,
        accel_raw=accel_raw,
        gyro_raw=gyro_raw,
        mag_raw=mag_raw,
        sampling_rate=sampling_rate,
        metadata=metadata,
    )


def load_spikegadgets(file_path: Path) -> SpikeGadgetsIMUData:
    """Load SpikeGadgets IMU data with automatic format detection.

    Args:
        file_path: Path to SpikeGadgets file (.bin or .csv)

    Returns:
        SpikeGadgetsIMUData container
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        return load_spikegadgets_csv(file_path)
    elif file_path.suffix.lower() in [".bin", ".dat"]:
        return load_spikegadgets_binary(file_path)
    else:
        raise ValueError(f"Unsupported SpikeGadgets file format: {file_path.suffix}")
