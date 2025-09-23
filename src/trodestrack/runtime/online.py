"""Online tracking API for trodestrack.

This module implements real-time state estimation for live tracking scenarios.
It provides a stateful tracker interface that can process measurements frame by frame
with low latency for real-time applications.
"""

import logging
import time
from typing import NamedTuple, Optional, Tuple, List
from collections import deque

import jax.numpy as jnp
import numpy as np

from ..config.schemas import SessionConfig
from ..models.state import State2D, create_initial_state, state_to_array, array_to_state
from ..models.ekf import EKFFilter
from ..models.measurements import create_measurement_noise
from ..imu.preintegration import preintegrate_imu_scan

logger = logging.getLogger(__name__)


class TrackingFrame(NamedTuple):
    """Single frame of tracking data.

    Attributes:
        timestamp: Frame timestamp in seconds
        position: Position measurement [x, y] in cm (None if missing)
        heading: Heading measurement in radians (None if missing)
        confidence: Detection confidence [0, 1]
        imu_measurements: IMU data since last frame [(accel, gyro, timestamp), ...]
    """
    timestamp: float
    position: Optional[jnp.ndarray]
    heading: Optional[float]
    confidence: float
    imu_measurements: List[Tuple[jnp.ndarray, jnp.ndarray, float]]


class TrackingResult(NamedTuple):
    """Result from processing a single tracking frame.

    Attributes:
        state: Current state estimate
        covariance: State covariance matrix (8x8)
        timestamp: Frame timestamp
        processing_time_ms: Time taken to process frame (milliseconds)
        innovation: Measurement innovation (empty if no measurement update)
        gated: Whether measurement was rejected by gating
        log_likelihood: Cumulative log-likelihood
    """
    state: State2D
    covariance: jnp.ndarray
    timestamp: float
    processing_time_ms: float
    innovation: jnp.ndarray
    gated: bool
    log_likelihood: float


class OnlineTracker:
    """Real-time tracking interface for live sensor fusion.

    This class maintains filter state and processes incoming measurements
    frame-by-frame for real-time tracking applications.
    """

    def __init__(self, config: SessionConfig):
        """Initialize online tracker.

        Args:
            config: Session configuration
        """
        self.config = config
        self._setup_logging()

        # Initialize filter (will be set up when first frame arrives)
        self._ekf_filter: Optional[EKFFilter] = None
        self._initialized = False

        # Timing and diagnostics
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._last_timestamp: Optional[float] = None

        # IMU buffer for accumulating measurements between frames
        self._imu_buffer: deque = deque(maxlen=10000)  # Ring buffer for recent IMU

        # Performance tracking
        self._processing_times: deque = deque(maxlen=1000)

        logger.info("OnlineTracker initialized")

    def _setup_logging(self) -> None:
        """Configure logging for online tracker."""
        log_level = getattr(logging, self.config.output.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

    def initialize(self, initial_position: jnp.ndarray, initial_heading: float = 0.0) -> None:
        """Initialize tracker with known starting conditions.

        Args:
            initial_position: Initial position [x, y] in cm
            initial_heading: Initial heading in radians
        """
        # Create initial state
        initial_state = State2D(
            x=float(initial_position[0]),
            y=float(initial_position[1]),
            vx=0.0,  # Start at rest
            vy=0.0,
            theta=float(initial_heading),
            b_gz=0.0,  # Assume no initial biases
            b_ax=0.0,
            b_ay=0.0,
        )

        # Create initial covariance matrix
        initial_covariance = self._create_initial_covariance()

        # Initialize EKF filter
        self._ekf_filter = EKFFilter(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            velocity_damping=self.config.filter.velocity_damping,
            accel_noise_std=np.sqrt(self.config.filter.process_noise["velocity"]),
            gyro_noise_std=np.sqrt(self.config.filter.process_noise["heading"]),
            bias_drift_std=np.sqrt(self.config.filter.process_noise["bias_gyro"]),
            position_noise_std=np.sqrt(self.config.filter.measurement_noise["position"]),
            heading_noise_std=np.sqrt(self.config.filter.measurement_noise["heading"]),
            gate_threshold=self.config.filter.gating_threshold,
        )

        self._initialized = True
        self._start_time = time.time()

        logger.info(f"Tracker initialized at position ({initial_position[0]:.1f}, {initial_position[1]:.1f}) cm, "
                   f"heading {np.degrees(initial_heading):.1f}°")

    def add_imu_measurement(self, accel: jnp.ndarray, gyro: jnp.ndarray, timestamp: float) -> None:
        """Add IMU measurement to buffer.

        Args:
            accel: Accelerometer measurement [ax, ay, az] in m/s²
            gyro: Gyroscope measurement [gx, gy, gz] in rad/s
            timestamp: Measurement timestamp in seconds
        """
        # Store in buffer for processing with next frame
        self._imu_buffer.append((accel, gyro, timestamp))

    def process_frame(self, frame: TrackingFrame) -> TrackingResult:
        """Process a single tracking frame.

        Args:
            frame: Frame data with measurements and timing

        Returns:
            TrackingResult with updated state estimate

        Raises:
            RuntimeError: If tracker not initialized
        """
        start_time = time.perf_counter()

        if not self._initialized:
            # Auto-initialize if we have position measurement
            if frame.position is not None:
                logger.info("Auto-initializing tracker from first frame")
                self.initialize(frame.position, frame.heading or 0.0)
            else:
                raise RuntimeError("Tracker not initialized - call initialize() or provide position in first frame")

        # Handle timing
        dt = 0.0
        if self._last_timestamp is not None:
            dt = frame.timestamp - self._last_timestamp

        # Collect IMU measurements since last frame
        imu_measurements = list(frame.imu_measurements)

        # Add any buffered IMU measurements in time range
        if self._last_timestamp is not None:
            buffered_imu = [
                (accel, gyro, ts) for accel, gyro, ts in self._imu_buffer
                if self._last_timestamp <= ts <= frame.timestamp
            ]
            imu_measurements.extend(buffered_imu)

        # Sort by timestamp
        imu_measurements.sort(key=lambda x: x[2])

        # Prediction step with IMU pre-integration
        innovation = jnp.array([])
        gated = False

        if dt > 0 and len(imu_measurements) > 0:
            # Perform IMU pre-integration
            imu_data, imu_timestamps = self._prepare_imu_data(imu_measurements)

            if len(imu_data) > 1:
                # Get current state for pre-integration
                current_state = self._ekf_filter.get_current_state()

                # Compute average IMU measurements for prediction
                avg_accel = jnp.mean(imu_data[:, :2], axis=0)  # [ax, ay]
                avg_gyro = jnp.array([jnp.mean(imu_data[:, 5])])  # [gz]

                # Run prediction step
                self._ekf_filter.predict(dt, avg_accel, avg_gyro)

        # Update step with visual measurements
        if frame.position is not None or frame.heading is not None:
            result = self._ekf_filter.update(
                position=frame.position,
                heading=frame.heading,
                confidence=frame.confidence,
            )
            innovation = result.innovation
            gated = result.gated

        # Get final state
        final_state = self._ekf_filter.get_current_state()
        final_covariance = self._ekf_filter.get_current_covariance()
        log_likelihood = self._ekf_filter.get_log_likelihood()

        # Update tracking state
        self._frame_count += 1
        self._last_timestamp = frame.timestamp

        # Timing
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000.0
        self._processing_times.append(processing_time_ms)

        # Log progress periodically
        if self._frame_count % 100 == 0:
            avg_time = np.mean(list(self._processing_times))
            logger.info(f"Processed {self._frame_count} frames, avg processing time: {avg_time:.2f} ms")

        return TrackingResult(
            state=final_state,
            covariance=final_covariance,
            timestamp=frame.timestamp,
            processing_time_ms=processing_time_ms,
            innovation=innovation,
            gated=gated,
            log_likelihood=log_likelihood,
        )

    def get_current_state(self) -> Optional[State2D]:
        """Get current state estimate.

        Returns:
            Current state or None if not initialized
        """
        if self._ekf_filter is None:
            return None
        return self._ekf_filter.get_current_state()

    def get_performance_stats(self) -> dict:
        """Get performance statistics.

        Returns:
            Dictionary with timing and processing statistics
        """
        if not self._processing_times:
            return {}

        processing_times = list(self._processing_times)

        stats = {
            'frame_count': self._frame_count,
            'avg_processing_time_ms': np.mean(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'std_processing_time_ms': np.std(processing_times),
        }

        if self._start_time is not None:
            elapsed_time = time.time() - self._start_time
            stats['elapsed_time_s'] = elapsed_time
            if elapsed_time > 0:
                stats['frames_per_second'] = self._frame_count / elapsed_time

        return stats

    def reset(self) -> None:
        """Reset tracker to uninitialized state."""
        self._ekf_filter = None
        self._initialized = False
        self._frame_count = 0
        self._start_time = None
        self._last_timestamp = None
        self._imu_buffer.clear()
        self._processing_times.clear()
        logger.info("Tracker reset")

    def _create_initial_covariance(self) -> jnp.ndarray:
        """Create initial covariance matrix from configuration."""
        variances = self.config.filter.initial_state_variance
        return jnp.diag(jnp.array([
            variances["position"],  # x
            variances["position"],  # y
            variances["velocity"],  # vx
            variances["velocity"],  # vy
            variances["heading"],   # theta
            variances["bias_gyro"], # b_gz
            variances["bias_accel"], # b_ax
            variances["bias_accel"], # b_ay
        ]))

    def _prepare_imu_data(self, imu_measurements: List[Tuple[jnp.ndarray, jnp.ndarray, float]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare IMU measurements for processing.

        Args:
            imu_measurements: List of (accel, gyro, timestamp) tuples

        Returns:
            Tuple of (imu_data_array, timestamps_array)
        """
        if not imu_measurements:
            return jnp.array([]).reshape(0, 6), jnp.array([])

        # Convert to arrays
        imu_data = []
        timestamps = []

        for accel, gyro, timestamp in imu_measurements:
            # Combine accel (3D) and gyro (3D) into 6D measurement
            measurement = jnp.concatenate([accel[:3], gyro[:3]])
            imu_data.append(measurement)
            timestamps.append(timestamp)

        return jnp.array(imu_data), jnp.array(timestamps)


class StreamingTracker:
    """Higher-level interface for streaming data processing.

    This class provides convenience methods for processing streams of
    sensor data from files or live sources.
    """

    def __init__(self, config: SessionConfig):
        """Initialize streaming tracker.

        Args:
            config: Session configuration
        """
        self.config = config
        self.tracker = OnlineTracker(config)
        self.results: List[TrackingResult] = []

    def process_data_streams(self, video_data: Optional[dict] = None,
                           imu_data: Optional[dict] = None) -> List[TrackingResult]:
        """Process complete data streams.

        Args:
            video_data: Video detection data
            imu_data: IMU measurement data

        Returns:
            List of tracking results for each frame
        """
        logger.info("Processing data streams")

        if video_data is None and imu_data is None:
            raise ValueError("At least one data stream must be provided")

        # Initialize from first video frame if available
        if video_data is not None and len(video_data['positions']) > 0:
            first_position = video_data['positions'][0]
            first_heading = video_data.get('headings', [0.0])[0] or 0.0
            if jnp.all(jnp.isfinite(first_position)):
                self.tracker.initialize(first_position, first_heading)

        # Process frame by frame
        if video_data is not None:
            n_frames = len(video_data['timestamps'])

            for i in range(n_frames):
                # Create tracking frame
                timestamp = video_data['timestamps'][i]
                position = video_data['positions'][i] if jnp.all(jnp.isfinite(video_data['positions'][i])) else None
                heading = video_data.get('headings', [None] * n_frames)[i]
                confidence = video_data.get('confidences', [1.0] * n_frames)[i]

                # Get IMU measurements for this time interval
                imu_measurements = []
                if imu_data is not None and i > 0:
                    prev_timestamp = video_data['timestamps'][i-1]
                    # Find IMU samples in time range
                    mask = (imu_data['timestamps'] >= prev_timestamp) & (imu_data['timestamps'] <= timestamp)
                    if jnp.any(mask):
                        for j in jnp.where(mask)[0]:
                            imu_sample = imu_data['data'][j]
                            accel = imu_sample[:3]
                            gyro = imu_sample[3:6]
                            imu_ts = imu_data['timestamps'][j]
                            imu_measurements.append((accel, gyro, imu_ts))

                frame = TrackingFrame(
                    timestamp=timestamp,
                    position=position,
                    heading=heading,
                    confidence=confidence,
                    imu_measurements=imu_measurements,
                )

                # Process frame
                result = self.tracker.process_frame(frame)
                self.results.append(result)

        logger.info(f"Processed {len(self.results)} frames")
        return self.results

    def get_state_estimates(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get state estimates and timestamps from processed results.

        Returns:
            Tuple of (states_array, timestamps_array)
        """
        if not self.results:
            return jnp.array([]).reshape(0, 8), jnp.array([])

        states = [state_to_array(result.state) for result in self.results]
        timestamps = [result.timestamp for result in self.results]

        return jnp.array(states), jnp.array(timestamps)

    def get_performance_summary(self) -> dict:
        """Get comprehensive performance summary.

        Returns:
            Dictionary with performance metrics
        """
        tracker_stats = self.tracker.get_performance_stats()

        if self.results:
            processing_times = [result.processing_time_ms for result in self.results]
            innovations = [jnp.linalg.norm(result.innovation) if len(result.innovation) > 0 else 0.0
                          for result in self.results]
            gated_count = sum(1 for result in self.results if result.gated)

            summary = {
                **tracker_stats,
                'total_frames': len(self.results),
                'gated_measurements': gated_count,
                'gating_rate': gated_count / len(self.results),
                'avg_innovation_norm': np.mean(innovations),
                'final_log_likelihood': self.results[-1].log_likelihood,
            }
        else:
            summary = tracker_stats

        return summary