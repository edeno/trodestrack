"""Simplified synthetic data generator for scenario testing.

This module provides a minimal synthetic data generator for testing
filtering scenarios without needing the full sim module implementation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from ..models.state import State2D


@dataclass
class SimConfig:
    """Simplified configuration for synthetic session generation."""

    duration_s: float = 10.0
    camera_fps: float = 30.0
    imu_rate_hz: float = 1000.0
    trajectory_type: str = "steady_motion"
    vision_dropout_periods: List[Tuple[float, float]] = None
    heading_dropout_periods: List[Tuple[float, float]] = None
    led_swap_periods: List[Tuple[float, float]] = None
    position_noise_std_cm: float = 1.0
    heading_noise_std_rad: float = 0.05
    confidence_noise_std: float = 0.1
    confidence_drop_during_swaps: bool = False
    imu_bias_gyro_drift_std: float = 0.001
    imu_bias_accel_drift_std: float = 0.0005
    seed: int = 42

    def __post_init__(self):
        if self.vision_dropout_periods is None:
            self.vision_dropout_periods = []
        if self.heading_dropout_periods is None:
            self.heading_dropout_periods = []
        if self.led_swap_periods is None:
            self.led_swap_periods = []


@dataclass
class IMUSample:
    """IMU measurement sample."""

    accel: jnp.ndarray  # [ax, ay] in m/s²
    gyro: jnp.ndarray  # [gz] in rad/s


@dataclass
class VideoDetection:
    """Video detection result."""

    position_cm: jnp.ndarray  # [x, y] in cm
    heading_rad: float  # heading in radians
    confidence: float  # detection confidence [0, 1]


class SyntheticSessionResult:
    """Result from synthetic session generation."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Generate ground truth trajectory
        self._generate_trajectory()

        # Create initial state
        self.initial_state = State2D(
            x=self.ground_truth_positions[0, 0],
            y=self.ground_truth_positions[0, 1],
            vx=self.ground_truth_velocities[0, 0],
            vy=self.ground_truth_velocities[0, 1],
            theta=self.ground_truth_headings[0],
            b_gz=0.0,
            b_ax=0.0,
            b_ay=0.0,
        )

    def _generate_trajectory(self):
        """Generate ground truth trajectory based on type."""
        dt = 1.0 / self.config.camera_fps
        n_steps = int(self.config.duration_s * self.config.camera_fps)

        self.timestamps = np.linspace(0, self.config.duration_s, n_steps)
        self.ground_truth_positions = np.zeros((n_steps, 2))
        self.ground_truth_velocities = np.zeros((n_steps, 2))
        self.ground_truth_headings = np.zeros(n_steps)

        if self.config.trajectory_type == "steady_motion":
            # Constant velocity motion
            vel = np.array([20.0, 10.0])  # cm/s
            for i in range(n_steps):
                self.ground_truth_positions[i] = vel * self.timestamps[i]
                self.ground_truth_velocities[i] = vel
                self.ground_truth_headings[i] = np.arctan2(vel[1], vel[0])

        elif self.config.trajectory_type == "circular":
            # Circular motion
            radius = 50.0  # cm
            angular_vel = 0.5  # rad/s
            for i in range(n_steps):
                angle = angular_vel * self.timestamps[i]
                self.ground_truth_positions[i] = radius * np.array([np.cos(angle), np.sin(angle)])
                self.ground_truth_velocities[i] = (
                    radius * angular_vel * np.array([-np.sin(angle), np.cos(angle)])
                )
                self.ground_truth_headings[i] = angle + np.pi / 2

        elif self.config.trajectory_type == "figure_eight":
            # Figure-8 motion
            scale = 30.0
            for i in range(n_steps):
                t = self.timestamps[i] * 0.3
                x = scale * np.sin(t)
                y = scale * np.sin(t) * np.cos(t)
                self.ground_truth_positions[i] = np.array([x, y])

                # Compute velocity via finite differences
                if i > 0:
                    vel = (self.ground_truth_positions[i] - self.ground_truth_positions[i - 1]) / dt
                    self.ground_truth_velocities[i] = vel
                    self.ground_truth_headings[i] = np.arctan2(vel[1], vel[0])

        elif self.config.trajectory_type == "turn_sequence":
            # Sequence of turns
            pos = np.array([0.0, 0.0])
            vel = np.array([25.0, 0.0])
            heading = 0.0

            for i in range(n_steps):
                self.ground_truth_positions[i] = pos
                self.ground_truth_velocities[i] = vel
                self.ground_truth_headings[i] = heading

                # Add random turns
                if i % 90 == 0 and i > 0:  # Turn every 3 seconds
                    turn_angle = self.rng.uniform(-np.pi / 3, np.pi / 3)
                    heading += turn_angle
                    speed = np.linalg.norm(vel)
                    vel = speed * np.array([np.cos(heading), np.sin(heading)])

                pos += vel * dt

        elif self.config.trajectory_type == "straight_line":
            # Straight line motion
            vel = np.array([30.0, 15.0])
            for i in range(n_steps):
                self.ground_truth_positions[i] = vel * self.timestamps[i]
                self.ground_truth_velocities[i] = vel
                self.ground_truth_headings[i] = np.arctan2(vel[1], vel[0])

        else:
            # Random walk fallback
            pos = np.array([0.0, 0.0])
            vel = np.array([0.0, 0.0])

            for i in range(n_steps):
                # Random acceleration
                accel = self.rng.normal(0, 5.0, 2)  # cm/s²
                vel += accel * dt

                # Speed limit
                speed = np.linalg.norm(vel)
                if speed > 50.0:
                    vel = vel / speed * 50.0

                pos += vel * dt

                self.ground_truth_positions[i] = pos
                self.ground_truth_velocities[i] = vel
                self.ground_truth_headings[i] = np.arctan2(vel[1], vel[0]) if speed > 1.0 else 0.0

    def get_ground_truth_at_time(self, t: float) -> jnp.ndarray:
        """Get ground truth state at given time."""
        idx = int(t * self.config.camera_fps)
        idx = min(idx, len(self.ground_truth_positions) - 1)

        return jnp.array(
            [
                self.ground_truth_positions[idx, 0],  # x
                self.ground_truth_positions[idx, 1],  # y
                self.ground_truth_velocities[idx, 0],  # vx
                self.ground_truth_velocities[idx, 1],  # vy
                self.ground_truth_headings[idx],  # theta
                0.0,
                0.0,
                0.0,  # biases
            ]
        )

    def generate_timeline(self) -> List[Tuple[float, IMUSample, Optional[VideoDetection]]]:
        """Generate timeline of IMU and video measurements."""
        imu_dt = 1.0 / self.config.imu_rate_hz
        video_dt = 1.0 / self.config.camera_fps

        timeline = []
        current_time = 0.0
        video_idx = 0

        while current_time < self.config.duration_s:
            # Generate IMU sample (always available)
            imu_sample = self._generate_imu_sample(current_time)

            # Generate video detection (may be missing)
            video_detection = None
            if abs(current_time - video_idx * video_dt) < imu_dt / 2:
                video_detection = self._generate_video_detection(current_time, video_idx)
                video_idx += 1

            timeline.append((current_time, imu_sample, video_detection))
            current_time += imu_dt

        return timeline

    def _generate_imu_sample(self, t: float) -> IMUSample:
        """Generate IMU sample at given time."""
        # Get ground truth acceleration and angular velocity
        truth = self.get_ground_truth_at_time(t)

        # Simple finite difference for acceleration
        dt = 0.01
        truth_next = self.get_ground_truth_at_time(t + dt)
        accel_true = (truth_next[2:4] - truth[2:4]) / dt / 100.0  # Convert cm/s² to m/s²

        # Angular velocity from heading derivative
        heading_next = truth_next[4]
        heading_current = truth[4]
        gyro_true = (heading_next - heading_current) / dt

        # Add noise
        accel_noise = self.rng.normal(0, 0.1, 2)  # m/s²
        gyro_noise = self.rng.normal(0, 0.05)  # rad/s

        return IMUSample(
            accel=jnp.array(accel_true + accel_noise), gyro=jnp.array([gyro_true + gyro_noise])
        )

    def _generate_video_detection(self, t: float, frame_idx: int) -> Optional[VideoDetection]:
        """Generate video detection at given time."""
        # Check for vision dropouts
        for start, end in self.config.vision_dropout_periods:
            if start <= t <= end:
                return None

        truth = self.get_ground_truth_at_time(t)

        # Add position noise
        pos_noise = self.rng.normal(0, self.config.position_noise_std_cm, 2)
        position = truth[:2] + pos_noise

        # Add heading noise
        heading_noise = self.rng.normal(0, self.config.heading_noise_std_rad)
        heading = truth[4] + heading_noise

        # Check for heading dropouts
        for start, end in self.config.heading_dropout_periods:
            if start <= t <= end:
                heading = None
                break

        # Check for LED swaps
        for start, end in self.config.led_swap_periods:
            if start <= t <= end:
                if heading is not None:
                    heading += np.pi  # 180-degree error
                break

        # Generate confidence
        confidence = self.rng.uniform(0.7, 1.0)

        # Drop confidence during swaps if configured
        if self.config.confidence_drop_during_swaps:
            for start, end in self.config.led_swap_periods:
                if start <= t <= end:
                    confidence = self.rng.uniform(0.2, 0.5)
                    break

        # Add confidence noise
        confidence += self.rng.normal(0, self.config.confidence_noise_std)
        confidence = np.clip(confidence, 0.0, 1.0)

        return VideoDetection(
            position_cm=jnp.array(position),
            heading_rad=heading if heading is not None else 0.0,
            confidence=confidence,
        )


def generate_synthetic_session(config: SimConfig) -> SyntheticSessionResult:
    """Generate a synthetic session for testing."""
    return SyntheticSessionResult(config)
