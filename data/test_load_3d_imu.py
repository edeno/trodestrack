"""Test loading Arthur session with full 3D IMU data.

This test validates that we can load all 6 IMU axes (gyro X/Y/Z, accel X/Y/Z)
from the Arthur session data for use with LAYOUT_2D_CAM_3D_IMU.

Design: Test-Driven Development
- Create test FIRST to define expected behavior
- Run test to verify it FAILS (data loader doesn't support 3D yet)
- Then implement the feature
- Verify test PASSES
"""

from pathlib import Path

import numpy as np
import pytest
from load_arthur_session import (
    GRAVITY,
    load_arthur_session,
)


class TestLoad3DIMU:
    """Test suite for loading full 3D IMU data."""

    @pytest.fixture
    def data_dir(self) -> Path:
        """Return path to data directory."""
        return Path(__file__).parent

    @pytest.fixture
    def position_file(self, data_dir: Path) -> str:
        """Return path to position parquet file."""
        fpath = data_dir / "arthur20220324_position_info.parquet"
        if not fpath.exists():
            pytest.skip(f"Position file not found: {fpath}")
        return str(fpath)

    @pytest.fixture
    def imu_file(self, data_dir: Path) -> str:
        """Return path to IMU parquet file."""
        fpath = data_dir / "arthur20220324_imu_info.parquet"
        if not fpath.exists():
            pytest.skip(f"IMU file not found: {fpath}")
        return str(fpath)

    def test_load_3d_imu_shape(self, position_file: str, imu_file: str):
        """Test that 3D IMU mode returns correct shape [N × 6]."""
        data = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="3d",
            verbose=False,
        )

        # Should have 6 columns: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
        assert (
            data.U_imu.shape[1] == 6
        ), f"Expected 6 IMU channels for 3D mode, got {data.U_imu.shape[1]}"

        # Should have same number of samples as before
        assert len(data.U_imu) == len(data.t_imu)

    def test_3d_accelerometer_magnitude(self, position_file: str, imu_file: str):
        """Test that 3D accelerometer magnitude ≈ gravity (9.81 m/s²)."""
        data = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="3d",
            verbose=False,
        )

        # Extract 3D accelerometer (columns 3, 4, 5)
        accel_x = data.U_imu[:, 3]
        accel_y = data.U_imu[:, 4]
        accel_z = data.U_imu[:, 5]

        # Compute magnitude
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Mean magnitude should be close to gravity
        mean_mag = np.mean(accel_mag)
        assert (
            9.5 < mean_mag < 10.1
        ), f"3D accel magnitude {mean_mag:.2f} m/s² not close to gravity (9.81 m/s²)"

    def test_gravity_in_z_axis(self, position_file: str, imu_file: str):
        """Test that gravity is primarily in the Z-axis (upright headstage)."""
        data = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="3d",
            verbose=False,
        )

        # Extract accelerometer means
        accel_x_mean = np.mean(data.U_imu[:, 3])
        accel_y_mean = np.mean(data.U_imu[:, 4])
        accel_z_mean = np.mean(data.U_imu[:, 5])

        # Z-axis should have largest magnitude (gravity)
        abs_means = np.abs([accel_x_mean, accel_y_mean, accel_z_mean])
        gravity_axis = np.argmax(abs_means)

        assert gravity_axis == 2, (
            f"Expected gravity in Z-axis (index 2), but found in axis {gravity_axis}\n"
            f"  Accel X mean: {accel_x_mean:.2f} m/s²\n"
            f"  Accel Y mean: {accel_y_mean:.2f} m/s²\n"
            f"  Accel Z mean: {accel_z_mean:.2f} m/s²"
        )

        # Z should be negative (pointing up in body frame)
        assert (
            accel_z_mean < 0
        ), f"Expected negative Z accel (upward), got {accel_z_mean:.2f} m/s²"

    def test_gyro_units_radians_per_second(self, position_file: str, imu_file: str):
        """Test that gyro output is in rad/s with reasonable range."""
        data = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="3d",
            verbose=False,
        )

        # Extract gyro data (columns 0, 1, 2)
        gyro_x = data.U_imu[:, 0]
        gyro_y = data.U_imu[:, 1]
        gyro_z = data.U_imu[:, 2]

        # Convert to deg/s for validation
        gyro_x_deg = gyro_x * 180 / np.pi
        gyro_y_deg = gyro_y * 180 / np.pi
        gyro_z_deg = gyro_z * 180 / np.pi

        # All axes should be within ±2000 deg/s (hardware spec)
        assert np.abs(gyro_x_deg).max() < 2000, "Gyro X exceeds hardware range"
        assert np.abs(gyro_y_deg).max() < 2000, "Gyro Y exceeds hardware range"
        assert np.abs(gyro_z_deg).max() < 2000, "Gyro Z exceeds hardware range"

        # Should have some non-zero variance (rat is moving)
        assert np.std(gyro_z_deg) > 1.0, "Gyro Z variance too low (rat not moving?)"

    def test_accel_units_meters_per_second_squared(
        self, position_file: str, imu_file: str
    ):
        """Test that accel output is in m/s² with reasonable range."""
        data = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="3d",
            verbose=False,
        )

        # Extract accel data
        accel_x = data.U_imu[:, 3]
        accel_y = data.U_imu[:, 4]
        accel_z = data.U_imu[:, 5]

        # Should be within ±2g (hardware spec: ±2g range)
        max_accel = 2 * GRAVITY
        assert np.abs(accel_x).max() < max_accel, "Accel X exceeds ±2g range"
        assert np.abs(accel_y).max() < max_accel, "Accel Y exceeds ±2g range"
        assert np.abs(accel_z).max() < max_accel, "Accel Z exceeds ±2g range"

    def test_backward_compatibility_2d_mode(self, position_file: str, imu_file: str):
        """Test that default 2D mode still works (backward compatibility)."""
        data_2d = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="2d",
            verbose=False,
        )

        # Should have 3 columns: [gyro_z, accel_x, accel_y]
        assert (
            data_2d.U_imu.shape[1] == 3
        ), f"Expected 3 IMU channels for 2D mode, got {data_2d.U_imu.shape[1]}"

    def test_3d_vs_2d_consistency(self, position_file: str, imu_file: str):
        """Test that 3D mode's subset matches 2D mode output."""
        data_2d = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="2d",
            verbose=False,
        )

        data_3d = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="3d",
            verbose=False,
        )

        # Extract overlapping channels from 3D mode
        # 2D: [gyro_z, accel_x, accel_y]
        # 3D: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
        # So 3D[:, [2, 3, 4]] should match 2D[:, [0, 1, 2]]

        gyro_z_2d = data_2d.U_imu[:, 0]
        accel_x_2d = data_2d.U_imu[:, 1]
        accel_y_2d = data_2d.U_imu[:, 2]

        gyro_z_3d = data_3d.U_imu[:, 2]
        accel_x_3d = data_3d.U_imu[:, 3]
        accel_y_3d = data_3d.U_imu[:, 4]

        # Should be identical (same conversion, same source data)
        np.testing.assert_allclose(gyro_z_2d, gyro_z_3d, rtol=1e-10)
        np.testing.assert_allclose(accel_x_2d, accel_x_3d, rtol=1e-10)
        np.testing.assert_allclose(accel_y_2d, accel_y_3d, rtol=1e-10)

    def test_immutability(self, position_file: str, imu_file: str):
        """Test that SessionData remains immutable (frozen dataclass)."""
        data = load_arthur_session(
            position_file=position_file,
            imu_file=imu_file,
            imu_mode="3d",
            verbose=False,
        )

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            data.led_distance = 0.05

        # Should not be able to add new attributes
        with pytest.raises(AttributeError):
            data.new_field = "test"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
