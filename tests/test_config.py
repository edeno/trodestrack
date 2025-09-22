"""Tests for configuration system."""

import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from trodestrack.config.loader import create_default_config, load_config, save_config
from trodestrack.config.schemas import (
    FilterConfig,
    IMUConfig,
    LEDConfig,
    MappingConfig,
    OutputConfig,
    SessionConfig,
)


class TestSchemas:
    """Test configuration schemas."""

    def test_mapping_config_homography(self):
        """Test homography mapping configuration."""
        config = MappingConfig(
            type="homography",
            homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        assert config.type == "homography"
        assert len(config.homography_matrix) == 3

    def test_mapping_config_ruler_scale(self):
        """Test ruler scale mapping configuration."""
        config = MappingConfig(type="ruler_scale", pixel_per_cm=10.5)
        assert config.type == "ruler_scale"
        assert config.pixel_per_cm == 10.5

    def test_mapping_config_validation_errors(self):
        """Test mapping config validation errors."""
        # Missing homography matrix for homography type
        with pytest.raises(ValueError, match="homography_matrix required"):
            MappingConfig(type="homography")

        # Missing pixel_per_cm for ruler_scale type
        with pytest.raises(ValueError, match="pixel_per_cm required"):
            MappingConfig(type="ruler_scale")

        # Invalid homography matrix size
        with pytest.raises(ValueError, match="must be 3x3"):
            MappingConfig(type="homography", homography_matrix=[[1.0, 0.0], [0.0, 1.0]])

    def test_filter_config_defaults(self):
        """Test filter configuration defaults."""
        config = FilterConfig()
        assert config.filter_type == "ekf"
        assert config.gating_threshold == 9.21
        assert "position" in config.process_noise
        assert "velocity" in config.process_noise

    def test_led_config_validation(self):
        """Test LED configuration validation."""
        config = LEDConfig(front_back_distance_cm=2.5)
        assert config.front_back_distance_cm == 2.5
        assert config.swap_policy == "wrapped_residual"

        # Test confidence bounds
        with pytest.raises(ValueError):
            LEDConfig(front_back_distance_cm=2.5, confidence_threshold=1.5)

    def test_imu_config_defaults(self):
        """Test IMU configuration defaults."""
        config = IMUConfig()
        assert config.downsampling_rate == 1000.0
        assert config.accel_scale == 0.000061
        assert config.gyro_scale == 0.061

    def test_output_config(self):
        """Test output configuration."""
        with TemporaryDirectory() as temp_dir:
            config = OutputConfig(output_dir=Path(temp_dir))
            assert config.output_dir == Path(temp_dir)
            assert config.save_states is True
            assert config.log_level == "INFO"


class TestSessionConfig:
    """Test session configuration."""

    def test_minimal_session_config(self):
        """Test minimal valid session configuration."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy files
            video_file = temp_path / "video.csv"
            imu_file = temp_path / "imu.bin"
            video_file.touch()
            imu_file.touch()

            config = SessionConfig(
                video_file=video_file,
                imu_file=imu_file,
                mapping=MappingConfig(type="ruler_scale", pixel_per_cm=10.0),
                output=OutputConfig(output_dir=temp_path / "output"),
            )

            assert config.video_file == video_file
            assert config.imu_file == imu_file
            assert config.video_fps == 30.0
            assert config.filter.filter_type == "ekf"

    def test_session_config_file_validation(self):
        """Test file existence validation."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Non-existent files should raise validation errors
            with pytest.raises(ValueError, match="Video file does not exist"):
                SessionConfig(
                    video_file=temp_path / "nonexistent.csv",
                    imu_file=temp_path / "imu.bin",
                    mapping=MappingConfig(type="ruler_scale", pixel_per_cm=10.0),
                    output=OutputConfig(output_dir=temp_path),
                )


class TestConfigLoader:
    """Test configuration loading and saving."""

    def test_save_and_load_config(self):
        """Test round-trip config save and load."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy files
            video_file = temp_path / "video.csv"
            imu_file = temp_path / "imu.bin"
            video_file.touch()
            imu_file.touch()

            # Create config
            original_config = SessionConfig(
                video_file=video_file,
                imu_file=imu_file,
                video_fps=25.0,
                mapping=MappingConfig(type="ruler_scale", pixel_per_cm=15.0),
                output=OutputConfig(output_dir=temp_path / "output"),
                led=LEDConfig(front_back_distance_cm=3.0),
            )

            # Save config
            config_file = temp_path / "config.yaml"
            save_config(original_config, config_file)

            # Load config
            loaded_config = load_config(config_file)

            # Verify round-trip (compare resolved paths since loader resolves them)
            assert loaded_config.video_file == original_config.video_file.resolve()
            assert loaded_config.imu_file == original_config.imu_file.resolve()
            assert loaded_config.output.output_dir == original_config.output.output_dir.resolve()
            assert loaded_config.video_fps == original_config.video_fps
            assert loaded_config.mapping.pixel_per_cm == original_config.mapping.pixel_per_cm
            assert (
                loaded_config.led.front_back_distance_cm
                == original_config.led.front_back_distance_cm
            )

    def test_load_nonexistent_config(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_create_default_config(self):
        """Test creating default configuration."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            video_file = temp_path / "video.csv"
            imu_file = temp_path / "imu.bin"
            output_dir = temp_path / "output"

            # Create files for validation
            video_file.touch()
            imu_file.touch()

            # Test homography default
            config = create_default_config(
                video_file=video_file,
                imu_file=imu_file,
                output_dir=output_dir,
                mapping_type="homography",
            )

            assert config.mapping.type == "homography"
            assert config.mapping.homography_matrix is not None

            # Test ruler scale default
            config = create_default_config(
                video_file=video_file,
                imu_file=imu_file,
                output_dir=output_dir,
                mapping_type="ruler_scale",
            )

            assert config.mapping.type == "ruler_scale"
            assert config.mapping.pixel_per_cm == 10.0

    def test_yaml_config_format(self):
        """Test YAML configuration file format."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy files
            video_file = temp_path / "video.csv"
            imu_file = temp_path / "imu.bin"
            video_file.touch()
            imu_file.touch()

            # Create and save config
            config = create_default_config(
                video_file=video_file,
                imu_file=imu_file,
                output_dir=temp_path / "output",
            )

            config_file = temp_path / "config.yaml"
            save_config(config, config_file)

            # Verify YAML structure
            with open(config_file, "r") as f:
                yaml_data = yaml.safe_load(f)

            assert "video_file" in yaml_data
            assert "imu_file" in yaml_data
            assert "mapping" in yaml_data
            assert "filter" in yaml_data
            assert "output" in yaml_data
            assert yaml_data["video_fps"] == 30.0
