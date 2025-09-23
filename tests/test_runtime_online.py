"""Tests for online tracking runtime API."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.config.schemas import (
    FilterConfig,
    IMUConfig,
    MappingConfig,
    OutputConfig,
    SessionConfig,
    SynchronizationConfig,
)
from trodestrack.models.state import State2D
from trodestrack.runtime.online import (
    OnlineTracker,
    StreamingTracker,
    TrackingFrame,
    TrackingResult,
)


class TestOnlineTracker:
    """Test OnlineTracker class."""

    @pytest.fixture
    def basic_config(self):
        """Create basic session configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SessionConfig(
                video_file=None,
                imu_file=None,
                video_fps=30.0,
                mapping=MappingConfig(type="ruler_scale", pixel_per_cm=4.0),
                filter=FilterConfig(filter_type="ekf", velocity_damping=0.1),
                imu=IMUConfig(),
                synchronization=SynchronizationConfig(),
                output=OutputConfig(
                    output_dir=Path(tmpdir),
                    save_states=True,
                    save_residuals=False,
                    save_plots=False,
                ),
            )

    @pytest.fixture
    def tracker(self, basic_config):
        """Create initialized tracker."""
        tracker = OnlineTracker(basic_config)
        tracker.initialize(
            initial_position=jnp.array([50.0, 50.0]),
            initial_heading=0.0,
        )
        return tracker

    def test_tracker_initialization(self, basic_config):
        """Test tracker initialization."""
        tracker = OnlineTracker(basic_config)

        # Should not be initialized initially
        assert not tracker._initialized
        assert tracker.get_current_state() is None

        # Initialize tracker
        initial_pos = jnp.array([10.0, 20.0])
        initial_heading = np.pi / 4

        tracker.initialize(initial_pos, initial_heading)

        # Should be initialized now
        assert tracker._initialized
        state = tracker.get_current_state()
        assert state is not None
        assert np.isclose(state.x, 10.0)
        assert np.isclose(state.y, 20.0)
        assert np.isclose(state.theta, np.pi / 4)

    def test_process_frame_position_only(self, tracker):
        """Test processing frame with position measurement only."""
        frame = TrackingFrame(
            timestamp=1.0,
            position=jnp.array([55.0, 48.0]),
            heading=None,
            confidence=0.9,
            imu_measurements=[],
        )

        result = tracker.process_frame(frame)

        # Validate result
        assert isinstance(result, TrackingResult)
        assert isinstance(result.state, State2D)
        assert result.timestamp == 1.0
        assert result.processing_time_ms > 0
        assert not result.gated  # Should not be gated with good measurement

        # State should be updated toward measurement
        assert abs(result.state.x - 55.0) < 5.0  # Should be closer to measurement
        assert abs(result.state.y - 48.0) < 5.0

    def test_process_frame_with_heading(self, tracker):
        """Test processing frame with position and heading."""
        frame = TrackingFrame(
            timestamp=1.0,
            position=jnp.array([52.0, 53.0]),
            heading=np.pi / 6,
            confidence=0.8,
            imu_measurements=[],
        )

        result = tracker.process_frame(frame)

        # Should process both position and heading
        assert not result.gated
        assert len(result.innovation) == 3  # [x, y, theta] innovation

    def test_process_frame_with_imu(self, tracker):
        """Test processing frame with IMU measurements."""
        # Create IMU measurements
        imu_measurements = [
            (jnp.array([0.1, 0.0, 9.8]), jnp.array([0.0, 0.0, 0.05]), 0.5),
            (jnp.array([0.0, 0.1, 9.8]), jnp.array([0.0, 0.0, 0.02]), 0.7),
        ]

        frame = TrackingFrame(
            timestamp=1.0,
            position=jnp.array([51.0, 51.0]),
            heading=None,
            confidence=0.9,
            imu_measurements=imu_measurements,
        )

        result = tracker.process_frame(frame)

        # Should successfully process with IMU
        assert isinstance(result.state, State2D)
        assert result.processing_time_ms > 0

    def test_process_frame_missing_measurement(self, tracker):
        """Test processing frame with no measurements."""
        frame = TrackingFrame(
            timestamp=1.0,
            position=None,
            heading=None,
            confidence=0.0,
            imu_measurements=[],
        )

        result = tracker.process_frame(frame)

        # Should still return valid result (prediction only)
        assert isinstance(result.state, State2D)
        assert len(result.innovation) == 0  # No measurement update

    def test_auto_initialization(self, basic_config):
        """Test automatic initialization from first frame."""
        tracker = OnlineTracker(basic_config)

        # Process frame without manual initialization
        frame = TrackingFrame(
            timestamp=0.0,
            position=jnp.array([25.0, 35.0]),
            heading=np.pi / 3,
            confidence=1.0,
            imu_measurements=[],
        )

        result = tracker.process_frame(frame)

        # Should auto-initialize and process
        assert tracker._initialized
        assert isinstance(result.state, State2D)
        assert abs(result.state.x - 25.0) < 1.0
        assert abs(result.state.y - 35.0) < 1.0

    def test_uninitialized_processing_error(self, basic_config):
        """Test error when processing without initialization."""
        tracker = OnlineTracker(basic_config)

        frame = TrackingFrame(
            timestamp=1.0,
            position=None,  # No position for auto-init
            heading=None,
            confidence=0.0,
            imu_measurements=[],
        )

        with pytest.raises(RuntimeError, match="Tracker not initialized"):
            tracker.process_frame(frame)

    def test_performance_statistics(self, tracker):
        """Test performance statistics collection."""
        # Process several frames
        for i in range(10):
            frame = TrackingFrame(
                timestamp=float(i),
                position=jnp.array([50.0 + i, 50.0]),
                heading=None,
                confidence=0.9,
                imu_measurements=[],
            )
            tracker.process_frame(frame)

        stats = tracker.get_performance_stats()

        # Check statistics
        assert stats["frame_count"] == 10
        assert "avg_processing_time_ms" in stats
        assert "max_processing_time_ms" in stats
        assert stats["avg_processing_time_ms"] > 0

    def test_tracker_reset(self, tracker):
        """Test tracker reset functionality."""
        # Process a frame
        frame = TrackingFrame(
            timestamp=1.0,
            position=jnp.array([60.0, 40.0]),
            heading=None,
            confidence=1.0,
            imu_measurements=[],
        )
        tracker.process_frame(frame)

        # Reset tracker
        tracker.reset()

        # Should be uninitialized
        assert not tracker._initialized
        assert tracker.get_current_state() is None
        assert tracker._frame_count == 0

    def test_imu_buffer_management(self, basic_config):
        """Test IMU buffer functionality."""
        tracker = OnlineTracker(basic_config)
        tracker.initialize(jnp.array([0.0, 0.0]), 0.0)

        # Add IMU measurements to buffer
        tracker.add_imu_measurement(jnp.array([0.1, 0.0, 9.8]), jnp.array([0.0, 0.0, 0.01]), 0.5)

        # Buffer should contain measurement
        assert len(tracker._imu_buffer) == 1

        # Process frame - should use buffered IMU
        frame = TrackingFrame(
            timestamp=1.0,
            position=jnp.array([1.0, 0.0]),
            heading=None,
            confidence=1.0,
            imu_measurements=[],
        )

        result = tracker.process_frame(frame)
        assert isinstance(result.state, State2D)


class TestStreamingTracker:
    """Test StreamingTracker class."""

    @pytest.fixture
    def basic_config(self):
        """Create basic session configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SessionConfig(
                video_file=None,
                imu_file=None,
                video_fps=30.0,
                mapping=MappingConfig(type="ruler_scale", pixel_per_cm=4.0),
                filter=FilterConfig(filter_type="ekf"),
                imu=IMUConfig(),
                synchronization=SynchronizationConfig(),
                output=OutputConfig(
                    output_dir=Path(tmpdir),
                    save_states=False,
                ),
            )

    def test_streaming_tracker_video_only(self, basic_config):
        """Test streaming tracker with video-only data."""
        tracker = StreamingTracker(basic_config)

        # Create simple video data
        timestamps = np.linspace(0, 2.0, 60)  # 30 FPS
        positions = np.column_stack(
            [
                20 + 10 * np.cos(timestamps),
                20 + 10 * np.sin(timestamps),
            ]
        )
        confidences = np.ones(len(timestamps))

        video_data = {
            "timestamps": timestamps,
            "positions": positions,
            "confidences": confidences,
        }

        # Process data
        results = tracker.process_data_streams(video_data=video_data)

        # Validate results
        assert len(results) == len(timestamps)
        assert all(isinstance(r, TrackingResult) for r in results)

        # Check state estimates
        states, result_timestamps = tracker.get_state_estimates()
        assert states.shape == (len(timestamps), 8)
        assert len(result_timestamps) == len(timestamps)

    def test_streaming_tracker_with_imu(self, basic_config):
        """Test streaming tracker with video and IMU data."""
        tracker = StreamingTracker(basic_config)

        # Create synchronized video and IMU data
        video_timestamps = np.linspace(0, 1.0, 30)
        video_positions = np.column_stack(
            [
                np.linspace(0, 10, 30),
                np.zeros(30),
            ]
        )

        imu_timestamps = np.linspace(0, 1.0, 1000)
        imu_data = np.zeros((1000, 6))
        imu_data[:, 0] = 0.1  # Constant acceleration
        imu_data[:, 5] = 0.01  # Small gyro noise

        video_data = {
            "timestamps": video_timestamps,
            "positions": video_positions,
            "confidences": np.ones(30),
        }

        imu_data_dict = {
            "timestamps": imu_timestamps,
            "data": imu_data,
        }

        # Process data
        results = tracker.process_data_streams(video_data, imu_data_dict)

        # Should successfully process all frames
        assert len(results) == 30
        assert all(isinstance(r, TrackingResult) for r in results)

    def test_performance_summary(self, basic_config):
        """Test performance summary generation."""
        tracker = StreamingTracker(basic_config)

        # Create minimal data
        video_data = {
            "timestamps": np.array([0.0, 1.0, 2.0]),
            "positions": np.array([[0, 0], [1, 0], [2, 0]]),
            "confidences": np.array([1.0, 0.5, 1.0]),  # One low confidence
        }

        # Process data
        results = tracker.process_data_streams(video_data=video_data)

        # Get performance summary
        summary = tracker.get_performance_summary()

        # Check summary contents
        assert "total_frames" in summary
        assert "gating_rate" in summary
        assert "avg_processing_time_ms" in summary
        assert summary["total_frames"] == 3

    def test_no_data_error(self, basic_config):
        """Test error handling with no input data."""
        tracker = StreamingTracker(basic_config)

        with pytest.raises(ValueError, match="At least one data stream"):
            tracker.process_data_streams()

    def test_empty_data_handling(self, basic_config):
        """Test handling of empty data streams."""
        tracker = StreamingTracker(basic_config)

        # Empty video data
        video_data = {
            "timestamps": np.array([]),
            "positions": np.array([]).reshape(0, 2),
            "confidences": np.array([]),
        }

        results = tracker.process_data_streams(video_data=video_data)

        # Should handle gracefully
        assert len(results) == 0


class TestTrackingDataStructures:
    """Test tracking data structures."""

    def test_tracking_frame_creation(self):
        """Test TrackingFrame creation and validation."""
        frame = TrackingFrame(
            timestamp=5.0,
            position=jnp.array([10.0, 20.0]),
            heading=np.pi / 4,
            confidence=0.8,
            imu_measurements=[
                (jnp.array([0, 0, 9.8]), jnp.array([0, 0, 0]), 4.9),
                (jnp.array([0, 0, 9.8]), jnp.array([0, 0, 0]), 5.0),
            ],
        )

        assert frame.timestamp == 5.0
        assert jnp.allclose(frame.position, jnp.array([10.0, 20.0]))
        assert frame.heading == np.pi / 4
        assert frame.confidence == 0.8
        assert len(frame.imu_measurements) == 2

    def test_tracking_result_creation(self):
        """Test TrackingResult creation."""
        state = State2D(x=1, y=2, vx=3, vy=4, theta=5, b_gz=6, b_ax=7, b_ay=8)
        covariance = jnp.eye(8)

        result = TrackingResult(
            state=state,
            covariance=covariance,
            timestamp=10.0,
            processing_time_ms=5.5,
            innovation=jnp.array([0.1, 0.2]),
            gated=False,
            log_likelihood=-50.0,
        )

        assert result.state == state
        assert jnp.allclose(result.covariance, jnp.eye(8))
        assert result.timestamp == 10.0
        assert result.processing_time_ms == 5.5
        assert not result.gated
        assert result.log_likelihood == -50.0

    def test_tracking_frame_with_missing_measurements(self):
        """Test TrackingFrame with missing measurements."""
        frame = TrackingFrame(
            timestamp=1.0,
            position=None,  # Missing position
            heading=None,  # Missing heading
            confidence=0.0,
            imu_measurements=[],
        )

        assert frame.position is None
        assert frame.heading is None
        assert frame.confidence == 0.0
        assert len(frame.imu_measurements) == 0
