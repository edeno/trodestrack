"""Smoke tests for runtime APIs to verify basic functionality."""

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
from trodestrack.runtime.offline import smooth_session
from trodestrack.runtime.online import OnlineTracker, TrackingFrame


class TestRuntimeSmoke:
    """Smoke tests for runtime APIs."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def minimal_config(self, temp_output_dir):
        """Create minimal session configuration."""
        return SessionConfig(
            video_file=None,
            imu_file=None,
            video_fps=30.0,
            mapping=MappingConfig(type="ruler_scale", pixel_per_cm=4.0),
            filter=FilterConfig(filter_type="ekf"),
            imu=IMUConfig(),
            synchronization=SynchronizationConfig(),
            output=OutputConfig(
                output_dir=temp_output_dir,
                save_states=False,  # Skip saving for smoke tests
                save_residuals=False,
                save_plots=False,
            ),
        )

    def test_online_tracker_basic(self, minimal_config):
        """Test basic OnlineTracker functionality."""
        tracker = OnlineTracker(minimal_config)

        # Initialize tracker
        tracker.initialize(
            initial_position=jnp.array([50.0, 50.0]),
            initial_heading=0.0,
        )

        # Process a simple frame
        frame = TrackingFrame(
            timestamp=1.0,
            position=jnp.array([51.0, 50.0]),
            heading=None,
            confidence=1.0,
            imu_measurements=[],
        )

        result = tracker.process_frame(frame)

        # Verify result structure
        assert result.state is not None
        assert result.covariance.shape == (8, 8)
        assert result.timestamp == 1.0
        assert result.processing_time_ms > 0

    def test_offline_smoothing_minimal(self, minimal_config):
        """Test offline smoothing with minimal video data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal video data
            timestamps = np.array([0.0, 1.0, 2.0])
            positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
            confidences = np.array([1.0, 1.0, 1.0])

            video_file = tmpdir / "video.npz"
            np.savez(
                video_file,
                timestamps=timestamps,
                positions=positions,
                confidences=confidences,
            )

            minimal_config.video_file = video_file

            # Run smoothing - should not crash
            result = smooth_session(minimal_config)

            # Verify basic structure
            assert len(result.filtered_states) == 3
            assert len(result.smoothed_states) == 3
            assert len(result.timestamps) == 3

    def test_import_all_runtime_components(self):
        """Test that all runtime components can be imported."""
        # Test imports
        from trodestrack.runtime import (
            smooth_session,
            SmoothingResult,
            OnlineTracker,
            StreamingTracker,
            TrackingFrame,
            TrackingResult,
        )

        # Verify they are callable/instantiable
        assert callable(smooth_session)
        assert hasattr(SmoothingResult, '_fields')  # NamedTuple
        assert callable(OnlineTracker)
        assert callable(StreamingTracker)
        assert hasattr(TrackingFrame, '_fields')    # NamedTuple
        assert hasattr(TrackingResult, '_fields')   # NamedTuple

    def test_config_validation_passes(self, temp_output_dir):
        """Test that valid configurations pass validation."""
        # Should not raise any exceptions
        config = SessionConfig(
            video_file=None,
            imu_file=None,
            video_fps=30.0,
            mapping=MappingConfig(type="ruler_scale", pixel_per_cm=4.0),
            filter=FilterConfig(filter_type="ekf"),
            output=OutputConfig(output_dir=temp_output_dir),
        )

        assert config.video_fps == 30.0
        assert config.filter.filter_type == "ekf"