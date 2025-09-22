"""Tests for data I/O loaders."""

import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from trodestrack.io.alignment import (
    align_timestamps,
    check_timestamp_synchronization,
    estimate_clock_offset,
    validate_alignment,
)
from trodestrack.io.dlc import DLCKeypointData, load_dlc_csv
from trodestrack.io.spikegadgets import (
    SpikeGadgetsIMUData,
    load_spikegadgets_binary,
    load_spikegadgets_csv,
)
from trodestrack.io.trodes import TrodesLEDData, load_trodes_led_csv


class TestTrodesLEDData:
    """Test Trodes LED data container."""

    def test_initialization(self):
        """Test TrodesLEDData initialization."""
        timestamps = np.array([0.0, 0.033, 0.066])
        front_led = np.array([[100, 200], [110, 210], [120, 220]])
        back_led = np.array([[80, 180], [90, 190], [100, 200]])
        front_conf = np.array([0.9, 0.8, 0.7])
        back_conf = np.array([0.85, 0.75, 0.65])

        data = TrodesLEDData(
            timestamps=timestamps,
            front_led=front_led,
            back_led=back_led,
            front_confidence=front_conf,
            back_confidence=back_conf,
        )

        assert data.n_frames == 3
        assert data.fps == pytest.approx(30.3, rel=0.1)
        np.testing.assert_array_equal(data.timestamps, timestamps)

    def test_validation_errors(self):
        """Test validation errors in TrodesLEDData."""
        timestamps = np.array([0.0, 0.033])
        front_led = np.array([[100, 200]])  # Wrong length
        back_led = np.array([[80, 180], [90, 190]])
        front_conf = np.array([0.9, 0.8])
        back_conf = np.array([0.85, 0.75])

        with pytest.raises(ValueError, match="same length"):
            TrodesLEDData(
                timestamps=timestamps,
                front_led=front_led,
                back_led=back_led,
                front_confidence=front_conf,
                back_confidence=back_conf,
            )

    def test_led_vector_and_heading(self):
        """Test LED vector and heading calculations."""
        timestamps = np.array([0.0, 0.033])
        front_led = np.array([[1.0, 0.0], [0.0, 1.0]])
        back_led = np.array([[0.0, 0.0], [0.0, 0.0]])
        front_conf = np.array([1.0, 1.0])
        back_conf = np.array([1.0, 1.0])

        data = TrodesLEDData(
            timestamps=timestamps,
            front_led=front_led,
            back_led=back_led,
            front_confidence=front_conf,
            back_confidence=back_conf,
        )

        vectors = data.get_led_vector()
        expected_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(vectors, expected_vectors)

        headings = data.get_heading()
        expected_headings = np.array([0.0, np.pi / 2])
        np.testing.assert_array_almost_equal(headings, expected_headings)

    def test_valid_frames(self):
        """Test valid frame detection."""
        timestamps = np.array([0.0, 0.033, 0.066])
        front_led = np.array([[100, 200], [110, 210], [120, 220]])
        back_led = np.array([[80, 180], [90, 190], [100, 200]])
        front_conf = np.array([0.9, 0.3, 0.7])  # Middle frame below threshold
        back_conf = np.array([0.85, 0.75, 0.65])

        data = TrodesLEDData(
            timestamps=timestamps,
            front_led=front_led,
            back_led=back_led,
            front_confidence=front_conf,
            back_confidence=back_conf,
        )

        valid_frames = data.get_valid_frames(confidence_threshold=0.5)
        expected_valid = np.array([True, False, True])
        np.testing.assert_array_equal(valid_frames, expected_valid)


class TestTrodesLoader:
    """Test Trodes data loading functions."""

    def test_load_csv_basic(self):
        """Test loading basic Trodes CSV file."""
        with TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_trodes.csv"

            # Create test CSV
            data = {
                "timestamp": [0.0, 0.033, 0.066],
                "front_x": [100, 110, 120],
                "front_y": [200, 210, 220],
                "back_x": [80, 90, 100],
                "back_y": [180, 190, 200],
                "front_conf": [0.9, 0.8, 0.7],
                "back_conf": [0.85, 0.75, 0.65],
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)

            # Load data
            led_data = load_trodes_led_csv(csv_file)

            assert led_data.n_frames == 3
            assert led_data.metadata["has_confidence"] is True
            np.testing.assert_array_equal(led_data.timestamps, [0.0, 0.033, 0.066])

    def test_load_csv_no_confidence(self):
        """Test loading CSV without confidence columns."""
        with TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_trodes_no_conf.csv"

            # Create test CSV without confidence
            data = {
                "timestamp": [0.0, 0.033, 0.066],
                "front_x": [100, 110, 120],
                "front_y": [200, 210, 220],
                "back_x": [80, 90, 100],
                "back_y": [180, 190, 200],
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)

            # Load data
            with pytest.warns(UserWarning, match="No confidence columns found"):
                led_data = load_trodes_led_csv(csv_file)

            assert led_data.metadata["has_confidence"] is False
            np.testing.assert_array_equal(led_data.front_confidence, [1.0, 1.0, 1.0])

    def test_load_csv_with_nan(self):
        """Test loading CSV with NaN values."""
        with TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_trodes_nan.csv"

            # Create test CSV with NaN
            data = {
                "timestamp": [0.0, 0.033, 0.066],
                "front_x": [100, np.nan, 120],
                "front_y": [200, np.nan, 220],
                "back_x": [80, 90, 100],
                "back_y": [180, 190, 200],
                "front_conf": [0.9, 0.8, 0.7],
                "back_conf": [0.85, 0.75, 0.65],
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)

            # Load data
            led_data = load_trodes_led_csv(csv_file)

            # Check that NaN position gets 0 confidence
            assert led_data.front_confidence[1] == 0.0
            assert led_data.front_led[1, 0] == 0.0
            assert led_data.front_led[1, 1] == 0.0


class TestDLCKeypointData:
    """Test DLC keypoint data container."""

    def test_initialization(self):
        """Test DLCKeypointData initialization."""
        timestamps = np.array([0.0, 0.033, 0.066])
        keypoints = {
            "nose": np.array([[100, 200], [110, 210], [120, 220]]),
            "tail": np.array([[80, 180], [90, 190], [100, 200]]),
        }
        confidences = {
            "nose": np.array([0.9, 0.8, 0.7]),
            "tail": np.array([0.85, 0.75, 0.65]),
        }

        data = DLCKeypointData(timestamps=timestamps, keypoints=keypoints, confidences=confidences)

        assert data.n_frames == 3
        assert set(data.keypoint_names) == {"nose", "tail"}
        assert data.fps == pytest.approx(30.3, rel=0.1)

    def test_get_keypoint(self):
        """Test keypoint retrieval."""
        timestamps = np.array([0.0, 0.033])
        keypoints = {"nose": np.array([[100, 200], [110, 210]])}
        confidences = {"nose": np.array([0.9, 0.8])}

        data = DLCKeypointData(timestamps=timestamps, keypoints=keypoints, confidences=confidences)

        pos, conf = data.get_keypoint("nose")
        np.testing.assert_array_equal(pos, keypoints["nose"])
        np.testing.assert_array_equal(conf, confidences["nose"])

        with pytest.raises(KeyError):
            data.get_keypoint("nonexistent")

    def test_led_data_extraction(self):
        """Test extracting LED-like data from keypoints."""
        timestamps = np.array([0.0, 0.033])
        keypoints = {
            "front": np.array([[100, 200], [110, 210]]),
            "back": np.array([[80, 180], [90, 190]]),
        }
        confidences = {"front": np.array([0.9, 0.8]), "back": np.array([0.85, 0.75])}

        data = DLCKeypointData(timestamps=timestamps, keypoints=keypoints, confidences=confidences)

        front_pos, back_pos, front_conf, back_conf = data.get_led_data("front", "back")

        np.testing.assert_array_equal(front_pos, keypoints["front"])
        np.testing.assert_array_equal(back_pos, keypoints["back"])


class TestSpikeGadgetsIMUData:
    """Test SpikeGadgets IMU data container."""

    def test_initialization(self):
        """Test SpikeGadgetsIMUData initialization."""
        timestamps = np.array([0.0, 1 / 30000, 2 / 30000])
        accel_raw = np.array([[1000, 2000, 3000], [1100, 2100, 3100], [1200, 2200, 3200]])
        gyro_raw = np.array([[100, 200, 300], [110, 210, 310], [120, 220, 320]])

        data = SpikeGadgetsIMUData(
            timestamps=timestamps,
            accel_raw=accel_raw,
            gyro_raw=gyro_raw,
            sampling_rate=30000.0,
        )

        assert data.n_samples == 3
        assert data.sampling_rate == 30000.0
        assert not data.has_magnetometer

    def test_unit_conversions(self):
        """Test IMU unit conversions."""
        timestamps = np.array([0.0, 1 / 30000])
        accel_raw = np.array([[16384, 0, 0], [0, 16384, 0]])  # 1g in each direction
        gyro_raw = np.array([[16384, 0, 0], [0, 16384, 0]])  # Some rotation

        data = SpikeGadgetsIMUData(timestamps=timestamps, accel_raw=accel_raw, gyro_raw=gyro_raw)

        # Test accelerometer conversion
        accel_g = data.get_accel_g()
        expected_g = accel_raw * 0.000061
        np.testing.assert_array_almost_equal(accel_g, expected_g)

        accel_ms2 = data.get_accel_ms2()
        expected_ms2 = expected_g * 9.80665
        np.testing.assert_array_almost_equal(accel_ms2, expected_ms2)

        # Test gyroscope conversion
        gyro_deg = data.get_gyro_deg_s()
        expected_deg = gyro_raw * 0.061
        np.testing.assert_array_almost_equal(gyro_deg, expected_deg)

    def test_downsampling(self):
        """Test IMU downsampling."""
        timestamps = np.linspace(0, 1, 30000)  # 1 second at 30kHz
        accel_raw = np.random.randint(-32768, 32767, (30000, 3))
        gyro_raw = np.random.randint(-32768, 32767, (30000, 3))

        data = SpikeGadgetsIMUData(
            timestamps=timestamps,
            accel_raw=accel_raw,
            gyro_raw=gyro_raw,
            sampling_rate=30000.0,
        )

        # Downsample to 1kHz
        downsampled = data.downsample(1000.0)

        assert downsampled.sampling_rate == 1000.0
        assert downsampled.n_samples == 1000
        assert "decimation_factor" in downsampled.metadata


class TestSpikeGadgetsLoaders:
    """Test SpikeGadgets data loading functions."""

    def test_load_csv(self):
        """Test loading SpikeGadgets CSV file."""
        with TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_imu.csv"

            # Create test CSV
            data = {
                "timestamp": [0.0, 1 / 30000, 2 / 30000],
                "accel_x": [1000, 1100, 1200],
                "accel_y": [2000, 2100, 2200],
                "accel_z": [3000, 3100, 3200],
                "gyro_x": [100, 110, 120],
                "gyro_y": [200, 210, 220],
                "gyro_z": [300, 310, 320],
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)

            # Load data
            imu_data = load_spikegadgets_csv(csv_file)

            assert imu_data.n_samples == 3
            assert not imu_data.has_magnetometer
            np.testing.assert_array_equal(imu_data.accel_raw[:, 0], [1000, 1100, 1200])

    def test_load_binary_basic(self):
        """Test loading basic SpikeGadgets binary file."""
        with TemporaryDirectory() as temp_dir:
            bin_file = Path(temp_dir) / "test_imu.bin"

            # Create test binary data (without magnetometer)
            with open(bin_file, "wb") as f:
                for i in range(3):
                    timestamp = i  # Raw timestamp
                    accel = [1000 + i * 100, 2000 + i * 100, 3000 + i * 100]
                    gyro = [100 + i * 10, 200 + i * 10, 300 + i * 10]

                    # Pack: timestamp (uint32) + 6 IMU values (int16)
                    record = struct.pack("<I6h", timestamp, *accel, *gyro)
                    f.write(record)

            # Load data
            imu_data = load_spikegadgets_binary(bin_file)

            assert imu_data.n_samples == 3
            assert not imu_data.has_magnetometer
            assert imu_data.metadata["record_size"] == 16


class TestTimestampAlignment:
    """Test timestamp alignment utilities."""

    def test_align_nearest(self):
        """Test nearest neighbor alignment."""
        video_timestamps = np.array([0.0, 0.033, 0.066, 0.100])
        imu_timestamps = np.linspace(0, 0.1, 3001)  # 30kHz for 0.1s

        video_indices, imu_indices = align_timestamps(
            video_timestamps, imu_timestamps, method="nearest"
        )

        assert len(video_indices) == len(imu_indices)
        assert len(video_indices) == 4  # All video frames should align

        # Check that alignment is reasonable
        aligned_video_times = video_timestamps[video_indices]
        aligned_imu_times = imu_timestamps[imu_indices]
        max_error = np.max(np.abs(aligned_video_times - aligned_imu_times))
        assert max_error < 0.001  # Should be very close

    def test_align_with_gap_constraint(self):
        """Test alignment with maximum gap constraint."""
        video_timestamps = np.array([0.0, 0.033, 0.066, 0.200])  # Large gap at end
        imu_timestamps = np.linspace(0, 0.1, 3001)  # IMU stops at 0.1s

        video_indices, imu_indices = align_timestamps(
            video_timestamps, imu_timestamps, method="nearest", max_gap=0.01
        )

        # Last video frame should be rejected due to gap
        assert len(video_indices) == 3
        np.testing.assert_array_equal(video_indices, [0, 1, 2])

    def test_synchronization_check(self):
        """Test timestamp synchronization checking."""
        # Well-synchronized case
        video_timestamps = np.array([0.0, 0.033, 0.066])
        imu_timestamps = np.linspace(0, 0.1, 3001)

        sync_result = check_timestamp_synchronization(
            video_timestamps, imu_timestamps, tolerance=0.001
        )

        assert sync_result["synchronized"] is True
        assert sync_result["overlap_duration"] > 0.06

        # Poorly synchronized case
        video_timestamps_offset = video_timestamps + 1.0  # 1 second offset
        sync_result_bad = check_timestamp_synchronization(
            video_timestamps_offset, imu_timestamps, tolerance=0.001
        )

        assert sync_result_bad["synchronized"] is False

    def test_clock_offset_estimation(self):
        """Test clock offset estimation."""
        base_timestamps = np.array([0.0, 0.033, 0.066])
        offset = 0.1

        video_timestamps = base_timestamps
        imu_timestamps = base_timestamps + offset

        estimated_offset = estimate_clock_offset(
            video_timestamps, imu_timestamps, method="first_sample"
        )

        assert estimated_offset == pytest.approx(offset, abs=1e-6)

    def test_validate_alignment(self):
        """Test alignment validation."""
        video_timestamps = np.array([0.0, 0.033, 0.066])
        imu_timestamps = np.array([0.001, 0.034, 0.067])  # Small offsets

        video_indices = np.array([0, 1, 2])
        imu_indices = np.array([0, 1, 2])

        validation = validate_alignment(
            video_indices, imu_indices, video_timestamps, imu_timestamps, max_error=0.01
        )

        assert validation["valid"] is True
        assert validation["n_aligned"] == 3
        assert validation["max_error"] < 0.01
