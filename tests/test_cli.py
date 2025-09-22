"""Tests for CLI interface."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from trodestrack.cli.main import (
    create_parser,
    main,
    cmd_smooth,
    cmd_online,
    cmd_report,
    cmd_calib_homography
)
from trodestrack.config.loader import create_default_config


class TestCLIParser:
    """Test CLI argument parsing."""

    def test_parser_creation(self):
        """Test parser creates all expected subcommands."""
        parser = create_parser()

        # Test help doesn't raise
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

    def test_smooth_command_args(self):
        """Test smooth command argument parsing."""
        parser = create_parser()

        args = parser.parse_args(["smooth", "--config", "session.yaml"])
        assert args.command == "smooth"
        assert args.config == Path("session.yaml")
        assert args.output is None

        args = parser.parse_args(["smooth", "--config", "session.yaml", "--output", "results/"])
        assert args.output == Path("results/")

    def test_online_command_args(self):
        """Test online command argument parsing."""
        parser = create_parser()

        args = parser.parse_args(["online", "--config", "session.yaml"])
        assert args.command == "online"
        assert args.config == Path("session.yaml")

    def test_report_command_args(self):
        """Test report command argument parsing."""
        parser = create_parser()

        args = parser.parse_args(["report", "--run-dir", "results/"])
        assert args.command == "report"
        assert args.run_dir == Path("results/")
        assert args.output is None

    def test_calib_command_args(self):
        """Test calibration command argument parsing."""
        parser = create_parser()

        args = parser.parse_args(["calib-homography", "--video", "frame.png"])
        assert args.command == "calib-homography"
        assert args.video == Path("frame.png")
        assert args.output is None

    def test_missing_required_args(self):
        """Test parser handles missing required arguments."""
        parser = create_parser()

        # Missing config for smooth
        with pytest.raises(SystemExit):
            parser.parse_args(["smooth"])

        # Missing run-dir for report
        with pytest.raises(SystemExit):
            parser.parse_args(["report"])


class TestCLICommands:
    """Test CLI command implementations."""

    def test_cmd_smooth_success(self):
        """Test smooth command with valid config."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy files and config
            video_file = temp_path / "video.csv"
            imu_file = temp_path / "imu.bin"
            video_file.touch()
            imu_file.touch()

            config = create_default_config(
                video_file=video_file,
                imu_file=imu_file,
                output_dir=temp_path / "output"
            )

            config_file = temp_path / "config.yaml"
            from trodestrack.config.loader import save_config
            save_config(config, config_file)

            # Mock arguments
            args = MagicMock()
            args.config = config_file
            args.output = None

            # Should succeed but not implement actual processing yet
            result = cmd_smooth(args)
            assert result == 0

    def test_cmd_smooth_invalid_config(self):
        """Test smooth command with invalid config."""
        args = MagicMock()
        args.config = Path("/nonexistent/config.yaml")
        args.output = None

        result = cmd_smooth(args)
        assert result == 1

    def test_cmd_online_success(self):
        """Test online command with valid config."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy files and config
            video_file = temp_path / "video.csv"
            imu_file = temp_path / "imu.bin"
            video_file.touch()
            imu_file.touch()

            config = create_default_config(
                video_file=video_file,
                imu_file=imu_file,
                output_dir=temp_path / "output"
            )

            config_file = temp_path / "config.yaml"
            from trodestrack.config.loader import save_config
            save_config(config, config_file)

            args = MagicMock()
            args.config = config_file

            result = cmd_online(args)
            assert result == 0

    def test_cmd_report_missing_dir(self):
        """Test report command with missing run directory."""
        args = MagicMock()
        args.run_dir = Path("/nonexistent/results")
        args.output = None

        result = cmd_report(args)
        assert result == 1

    def test_cmd_calib_missing_video(self):
        """Test calibration command with missing video file."""
        args = MagicMock()
        args.video = Path("/nonexistent/video.mp4")
        args.output = None

        result = cmd_calib_homography(args)
        assert result == 1

    def test_main_integration(self):
        """Test main function integration."""
        # Test invalid command - argparse will raise SystemExit
        with pytest.raises(SystemExit):
            main(["invalid-command"])

        # Test help
        with pytest.raises(SystemExit):
            main(["--help"])


class TestCLIIntegration:
    """Test CLI integration with configuration system."""

    def test_end_to_end_config_workflow(self):
        """Test complete config creation and CLI usage."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy input files
            video_file = temp_path / "video.csv"
            imu_file = temp_path / "imu.bin"
            video_file.touch()
            imu_file.touch()

            # Create configuration
            config = create_default_config(
                video_file=video_file,
                imu_file=imu_file,
                output_dir=temp_path / "output"
            )

            # Save configuration
            config_file = temp_path / "session.yaml"
            from trodestrack.config.loader import save_config
            save_config(config, config_file)

            # Test CLI can load and use the config
            result = main(["smooth", "--config", str(config_file)])
            assert result == 0  # Should succeed (even though not implemented)

    def test_cli_output_override(self):
        """Test CLI output directory override."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy files and config
            video_file = temp_path / "video.csv"
            imu_file = temp_path / "imu.bin"
            video_file.touch()
            imu_file.touch()

            config = create_default_config(
                video_file=video_file,
                imu_file=imu_file,
                output_dir=temp_path / "original_output"
            )

            config_file = temp_path / "config.yaml"
            from trodestrack.config.loader import save_config
            save_config(config, config_file)

            # Test output override
            override_output = temp_path / "override_output"
            result = main(["smooth", "--config", str(config_file), "--output", str(override_output)])
            assert result == 0