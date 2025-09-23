"""Tests for offline smoothing runtime API."""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import jax.numpy as jnp

from trodestrack.config.schemas import (
    SessionConfig,
    FilterConfig,
    OutputConfig,
    MappingConfig,
    IMUConfig,
    SynchronizationConfig,
)
from trodestrack.runtime.offline import smooth_session, SmoothingResult
from trodestrack.sim.generator import generate_synthetic_session


class TestOfflineSmoothing:
    """Test offline smoothing pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def basic_config(self, temp_output_dir):
        """Create basic session configuration."""
        return SessionConfig(
            video_file=None,  # Will be set by tests
            imu_file=None,  # Will be set by tests
            video_fps=30.0,
            mapping=MappingConfig(
                type="ruler_scale",
                pixel_per_cm=4.0,
            ),
            filter=FilterConfig(
                filter_type="ekf",
                velocity_damping=0.1,
            ),
            imu=IMUConfig(),
            synchronization=SynchronizationConfig(),
            output=OutputConfig(
                output_dir=temp_output_dir,
                save_states=True,
                save_residuals=False,
                save_plots=False,
            ),
        )

    def test_synthetic_session_smoothing(self, basic_config):
        """Test smoothing on synthetic data."""
        # Generate synthetic session
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create synthetic data files
            video_file = tmpdir / "video.npz"
            imu_file = tmpdir / "imu.npz"

            session_data = generate_synthetic_session(
                duration_s=10.0,
                video_fps=30.0,
                imu_rate=1000.0,
                arena_size_cm=(200, 150),
                trajectory_type="figure_eight",
                video_noise_cm=0.5,
                imu_noise_scale=1.0,
                occlusion_probability=0.0,  # No occlusions for basic test
                random_seed=42,
            )

            # Save synthetic data
            np.savez(
                video_file,
                timestamps=session_data.video_timestamps,
                positions=session_data.video_positions,
                confidences=session_data.video_confidences,
            )

            np.savez(
                imu_file,
                timestamps=session_data.imu_timestamps,
                data=session_data.imu_data,
                sampling_rate=session_data.imu_rate,
            )

            # Update config with file paths
            basic_config.video_file = video_file
            basic_config.imu_file = imu_file

            # Run smoothing
            result = smooth_session(basic_config)

            # Validate results
            assert isinstance(result, SmoothingResult)
            assert len(result.filtered_states) > 0
            assert len(result.smoothed_states) == len(result.filtered_states)
            assert len(result.timestamps) == len(result.filtered_states)
            assert result.filtered_covariances.shape == (len(result.filtered_states), 8, 8)
            assert result.smoothed_covariances.shape == (len(result.smoothed_states), 8, 8)

            # Check that results are finite
            assert jnp.all(jnp.isfinite(result.filtered_states))
            assert jnp.all(jnp.isfinite(result.smoothed_states))

            # Check diagnostic information
            assert "config_summary" in result.diagnostics
            assert "data_summary" in result.diagnostics
            assert result.diagnostics["data_summary"]["n_frames"] > 0

    def test_video_only_smoothing(self, basic_config):
        """Test smoothing with video-only data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create simple video data
            timestamps = np.linspace(0, 5.0, 150)  # 30 FPS for 5 seconds
            positions = np.column_stack(
                [
                    50 + 30 * np.cos(timestamps),  # Circular motion
                    50 + 30 * np.sin(timestamps),
                ]
            )
            confidences = np.ones(len(timestamps))

            video_file = tmpdir / "video_only.npz"
            np.savez(
                video_file,
                timestamps=timestamps,
                positions=positions,
                confidences=confidences,
            )

            # Update config
            basic_config.video_file = video_file
            basic_config.imu_file = None

            # Run smoothing
            result = smooth_session(basic_config)

            # Validate results
            assert len(result.filtered_states) == len(timestamps)
            assert jnp.all(jnp.isfinite(result.filtered_states))

            # Position should be close to measurements (since no IMU drift)
            estimated_positions = result.smoothed_states[:, :2]
            position_errors = jnp.linalg.norm(estimated_positions - positions, axis=1)
            assert jnp.mean(position_errors) < 5.0  # Should be within 5 cm on average

    def test_imu_only_smoothing(self, basic_config):
        """Test smoothing with IMU-only data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create simple IMU data
            duration = 5.0
            imu_rate = 1000.0
            n_samples = int(duration * imu_rate)

            timestamps = np.linspace(0, duration, n_samples)
            # Simulate constant velocity motion with noise
            imu_data = np.zeros((n_samples, 6))
            imu_data[:, 0] = np.random.normal(0, 0.1, n_samples)  # accel_x noise
            imu_data[:, 1] = np.random.normal(0, 0.1, n_samples)  # accel_y noise
            imu_data[:, 5] = np.random.normal(0, 0.01, n_samples)  # gyro_z noise

            imu_file = tmpdir / "imu_only.npz"
            np.savez(
                imu_file,
                timestamps=timestamps,
                data=imu_data,
                sampling_rate=imu_rate,
            )

            # Update config
            basic_config.video_file = None
            basic_config.imu_file = imu_file

            # Run smoothing
            result = smooth_session(basic_config)

            # Validate results - should run without error
            assert len(result.filtered_states) > 0
            assert jnp.all(jnp.isfinite(result.filtered_states))

    def test_output_files_created(self, basic_config):
        """Test that output files are created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal video data
            timestamps = np.linspace(0, 2.0, 60)
            positions = np.column_stack(
                [
                    np.linspace(0, 10, 60),
                    np.zeros(60),
                ]
            )

            video_file = tmpdir / "video.npz"
            np.savez(
                video_file,
                timestamps=timestamps,
                positions=positions,
                confidences=np.ones(60),
            )

            basic_config.video_file = video_file

            # Run smoothing
            result = smooth_session(basic_config)

            # Check output files
            output_dir = basic_config.output.output_dir
            assert (output_dir / "states.npz").exists()
            assert (output_dir / "diagnostics.json").exists()

            # Validate saved data
            saved_data = np.load(output_dir / "states.npz")
            assert "filtered_states" in saved_data
            assert "smoothed_states" in saved_data
            assert "timestamps" in saved_data
            assert len(saved_data["filtered_states"]) == len(timestamps)

    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configuration with no input files
            config = SessionConfig(
                video_file=None,
                imu_file=None,
                video_fps=30.0,
                mapping=MappingConfig(type="ruler_scale", pixel_per_cm=4.0),
                output=OutputConfig(output_dir=Path(tmpdir)),
            )

            with pytest.raises(ValueError, match="At least one of video_file or imu_file"):
                smooth_session(config)

    def test_missing_input_files(self, basic_config):
        """Test handling of missing input files."""
        # Set non-existent file paths
        basic_config.video_file = Path("/nonexistent/video.npz")
        basic_config.imu_file = Path("/nonexistent/imu.npz")

        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            smooth_session(basic_config)

    def test_configuration_validation(self, temp_output_dir):
        """Test that configuration validation works properly."""
        # Valid configuration
        config = SessionConfig(
            video_file=None,
            imu_file=None,
            video_fps=30.0,
            mapping=MappingConfig(type="ruler_scale", pixel_per_cm=4.0),
            output=OutputConfig(output_dir=temp_output_dir),
        )

        # Should not raise during creation
        assert config.video_fps == 30.0
        assert config.mapping.type == "ruler_scale"

    def test_filter_types(self, basic_config):
        """Test different filter configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create simple video data
            timestamps = np.linspace(0, 2.0, 60)
            positions = np.zeros((60, 2))

            video_file = tmpdir / "video.npz"
            np.savez(
                video_file,
                timestamps=timestamps,
                positions=positions,
                confidences=np.ones(60),
            )

            basic_config.video_file = video_file

            # Test EKF
            basic_config.filter.filter_type = "ekf"
            result_ekf = smooth_session(basic_config)
            assert len(result_ekf.filtered_states) > 0

            # Test UKF
            basic_config.filter.filter_type = "ukf"
            result_ukf = smooth_session(basic_config)
            assert len(result_ukf.filtered_states) > 0

            # Both should produce valid results
            assert jnp.all(jnp.isfinite(result_ekf.filtered_states))
            assert jnp.all(jnp.isfinite(result_ukf.filtered_states))


class TestSmoothingResultValidation:
    """Test SmoothingResult structure and validation."""

    def test_smoothing_result_structure(self):
        """Test that SmoothingResult has correct structure."""
        # Create mock result
        n_frames = 100
        result = SmoothingResult(
            filtered_states=jnp.zeros((n_frames, 8)),
            smoothed_states=jnp.zeros((n_frames, 8)),
            timestamps=jnp.linspace(0, 10, n_frames),
            filtered_covariances=jnp.zeros((n_frames, 8, 8)),
            smoothed_covariances=jnp.zeros((n_frames, 8, 8)),
            log_likelihood=-123.45,
            diagnostics={"test": "value"},
        )

        # Validate fields
        assert result.filtered_states.shape == (n_frames, 8)
        assert result.smoothed_states.shape == (n_frames, 8)
        assert result.timestamps.shape == (n_frames,)
        assert result.filtered_covariances.shape == (n_frames, 8, 8)
        assert result.smoothed_covariances.shape == (n_frames, 8, 8)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.diagnostics, dict)
