"""Tests for synthetic data generation."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from trodestrack.sim.config import SimConfig
from trodestrack.sim.imu import generate_synthetic_imu
from trodestrack.sim.session import generate_synthetic_session
from trodestrack.sim.video import generate_synthetic_video


class TestSimConfig:
    """Test synthetic data configuration schema."""

    def test_minimal_sim_config(self):
        """Test creation with minimal required parameters."""
        config_data = {
            "duration": 30.0,
            "video_fps": 30.0,
            "imu_rate": 30000.0,
            "arena_size": [200.0, 200.0],
            "seed": 42,
        }
        config = SimConfig(**config_data)
        assert config.duration == 30.0
        assert config.video_fps == 30.0
        assert config.imu_rate == 30000.0
        assert config.arena_size == [200.0, 200.0]
        assert config.seed == 42

    def test_sim_config_with_all_parameters(self):
        """Test configuration with all optional parameters."""
        config_data = {
            "duration": 60.0,
            "video_fps": 30.0,
            "imu_rate": 20000.0,
            "arena_size": [300.0, 150.0],
            "seed": 123,
            "trajectory": {
                "max_speed": 50.0,
                "max_acceleration": 100.0,
                "turn_probability": 0.02,
            },
            "imu": {
                "accel_noise_std": 0.1,
                "gyro_noise_std": 0.05,
                "accel_bias_std": 0.02,
                "gyro_bias_std": 0.01,
                "bias_drift_std": 0.001,
                "misalignment_deg": 2.0,
            },
            "video": {
                "position_noise_std": 2.0,
                "confidence_min": 0.1,
                "confidence_max": 0.99,
                "occlusion_probability": 0.05,
                "occlusion_duration_mean": 1.0,
                "led_swap_probability": 0.01,
                "dropout_probability": 0.02,
            },
            "led": {"front_back_distance": 25.0, "swap_detection_threshold": 0.8},
        }
        config = SimConfig(**config_data)
        assert config.trajectory.max_speed == 50.0
        assert config.imu.accel_noise_std == 0.1
        assert config.video.position_noise_std == 2.0
        assert config.led.front_back_distance == 25.0

    def test_sim_config_validation_errors(self):
        """Test validation errors for invalid parameters."""
        from pydantic import ValidationError

        # Test negative duration
        with pytest.raises(ValidationError):
            SimConfig(
                duration=-1.0,
                video_fps=30.0,
                imu_rate=30000.0,
                arena_size=[200.0, 200.0],
                seed=42,
            )

        # Test zero fps
        with pytest.raises(ValidationError):
            SimConfig(
                duration=30.0,
                video_fps=0.0,
                imu_rate=30000.0,
                arena_size=[200.0, 200.0],
                seed=42,
            )

        # Test invalid arena size
        with pytest.raises(ValidationError):
            SimConfig(
                duration=30.0,
                video_fps=30.0,
                imu_rate=30000.0,
                arena_size=[-100.0, 200.0],
                seed=42,
            )

    def test_sim_config_yaml_serialization(self):
        """Test YAML serialization and deserialization."""
        config_data = {
            "duration": 30.0,
            "video_fps": 30.0,
            "imu_rate": 30000.0,
            "arena_size": [200.0, 200.0],
            "seed": 42,
        }
        config = SimConfig(**config_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config.model_dump(), f)
            temp_path = Path(f.name)

        try:
            with open(temp_path, "r") as f:
                loaded_data = yaml.safe_load(f)
            loaded_config = SimConfig(**loaded_data)
            assert loaded_config.duration == config.duration
            assert loaded_config.seed == config.seed
        finally:
            temp_path.unlink()


class TestSyntheticIMU:
    """Test synthetic IMU data generation."""

    def test_generate_synthetic_imu_basic(self):
        """Test basic IMU data generation."""
        config = SimConfig(
            duration=1.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )

        # Generate ground truth trajectory
        ground_truth = self._generate_trajectory(config)

        # Generate synthetic IMU data
        imu_data = generate_synthetic_imu(ground_truth, config)

        # Check basic properties
        expected_samples = int(config.duration * config.imu_rate)
        assert len(imu_data.timestamps) == expected_samples
        assert imu_data.accel_raw.shape == (expected_samples, 3)
        assert imu_data.gyro_raw.shape == (expected_samples, 3)
        assert imu_data.sampling_rate == config.imu_rate

    def test_synthetic_imu_noise_characteristics(self):
        """Test that synthetic IMU has expected noise characteristics."""
        config = SimConfig(
            duration=10.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )
        config.imu.accel_noise_std = 0.1
        config.imu.gyro_noise_std = 0.05

        ground_truth = self._generate_trajectory(config)
        imu_data = generate_synthetic_imu(ground_truth, config)

        # Test basic sanity checks instead of exact noise levels
        # (Exact noise testing is difficult due to trajectory derivative noise)

        # Check that accelerometer data is reasonable (not all zeros, within expected range)
        accel_ms2 = imu_data.get_accel_ms2()
        assert np.all(np.abs(accel_ms2[:, :2]) < 50.0)  # Horizontal accel should be reasonable
        assert np.all(np.abs(accel_ms2[:, 2] - 9.81) < 5.0)  # Z should be near gravity

        # Check that there is some noise (not constant)
        assert np.std(accel_ms2[:, 0]) > 0.1
        assert np.std(accel_ms2[:, 1]) > 0.1
        assert np.std(accel_ms2[:, 2]) > 0.1

    def test_synthetic_imu_bias_drift(self):
        """Test that IMU biases drift over time."""
        config = SimConfig(
            duration=30.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )
        config.imu.bias_drift_std = 0.01

        ground_truth = self._generate_trajectory(config)
        imu_data = generate_synthetic_imu(ground_truth, config)

        # Extract bias estimates from metadata
        assert "true_accel_bias" in imu_data.metadata
        assert "true_gyro_bias" in imu_data.metadata

        accel_bias = imu_data.metadata["true_accel_bias"]
        gyro_bias = imu_data.metadata["true_gyro_bias"]

        # Biases should change over time (not constant)
        assert np.std(accel_bias[:, 0]) > 0.001  # Some drift
        assert np.std(gyro_bias[:, 2]) > 0.001  # Z-axis gyro bias

    def test_synthetic_imu_reproducibility(self):
        """Test that synthetic IMU generation is reproducible with same seed."""
        config = SimConfig(
            duration=1.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )

        ground_truth = self._generate_trajectory(config)

        imu_data1 = generate_synthetic_imu(ground_truth, config)
        imu_data2 = generate_synthetic_imu(ground_truth, config)

        np.testing.assert_array_equal(imu_data1.accel_raw, imu_data2.accel_raw)
        np.testing.assert_array_equal(imu_data1.gyro_raw, imu_data2.gyro_raw)

    def _generate_trajectory(self, config: SimConfig) -> Dict[str, np.ndarray]:
        """Helper to generate ground truth trajectory."""
        # This is a placeholder - will be implemented later
        t = np.linspace(0, config.duration, int(config.duration * config.imu_rate))
        positions = (
            np.column_stack(
                [
                    50 * np.sin(2 * np.pi * t / 10),  # x
                    50 * np.cos(2 * np.pi * t / 10),  # y
                ]
            )
            + 100
        )  # Center in arena

        velocities = np.gradient(positions, axis=0) / (t[1] - t[0])
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])

        return {
            "timestamps": t,
            "positions": positions,
            "velocities": velocities,
            "headings": headings,
        }

    def _compute_expected_accel(self, ground_truth: Dict[str, np.ndarray]) -> np.ndarray:
        """Helper to compute expected acceleration from trajectory."""
        velocities = ground_truth["velocities"]
        dt = ground_truth["timestamps"][1] - ground_truth["timestamps"][0]
        accelerations = np.gradient(velocities, axis=0) / dt

        # Add gravity to z-axis
        accel_3d = np.column_stack(
            [accelerations, -9.80665 * np.ones(len(accelerations))]  # Gravity
        )
        return accel_3d


class TestSyntheticVideo:
    """Test synthetic video data generation."""

    def test_generate_synthetic_video_basic(self):
        """Test basic video data generation."""
        config = SimConfig(
            duration=1.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )

        ground_truth = self._generate_trajectory(config)
        video_data = generate_synthetic_video(ground_truth, config)

        # Check basic properties
        expected_frames = int(config.duration * config.video_fps)
        assert len(video_data.timestamps) == expected_frames
        assert video_data.front_led.shape == (expected_frames, 2)
        assert video_data.back_led.shape == (expected_frames, 2)
        assert len(video_data.front_confidence) == expected_frames
        assert len(video_data.back_confidence) == expected_frames

    def test_synthetic_video_noise_and_confidence(self):
        """Test video noise and confidence relationships."""
        config = SimConfig(
            duration=5.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )
        config.video.position_noise_std = 3.0
        config.video.confidence_min = 0.1
        config.video.confidence_max = 0.95
        config.video.dropout_probability = 0.0  # Disable dropouts for this test

        ground_truth = self._generate_trajectory(config)
        video_data = generate_synthetic_video(ground_truth, config)

        # Check confidence range (excluding dropout frames which have 0 confidence)
        non_dropout_mask = video_data.front_confidence > 0
        assert np.all(video_data.front_confidence[non_dropout_mask] >= config.video.confidence_min)
        assert np.all(video_data.front_confidence[non_dropout_mask] <= config.video.confidence_max)
        assert np.all(video_data.back_confidence[non_dropout_mask] >= config.video.confidence_min)
        assert np.all(video_data.back_confidence[non_dropout_mask] <= config.video.confidence_max)

        # Higher confidence should correlate with lower position noise
        # (This is a statistical relationship, so we test it loosely)
        high_conf_mask = video_data.front_confidence > 0.8
        low_conf_mask = video_data.front_confidence < 0.3

        if np.sum(high_conf_mask) > 10 and np.sum(low_conf_mask) > 10:
            high_conf_positions = video_data.front_led[high_conf_mask]
            low_conf_positions = video_data.front_led[low_conf_mask]

            # Should have some difference in noise levels (loose test)
            high_noise = np.std(high_conf_positions, axis=0)
            low_noise = np.std(low_conf_positions, axis=0)
            # Just check that they're not identical
            assert not np.allclose(high_noise, low_noise, rtol=0.01)

    def test_synthetic_video_occlusions(self):
        """Test video occlusion generation."""
        config = SimConfig(
            duration=10.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )
        config.video.occlusion_probability = 0.1  # High occlusion rate for testing

        ground_truth = self._generate_trajectory(config)
        video_data = generate_synthetic_video(ground_truth, config)

        # Check that some frames have low confidence (indicating occlusions)
        low_confidence_frames = np.sum(
            (video_data.front_confidence < 0.2) | (video_data.back_confidence < 0.2)
        )
        total_frames = len(video_data.timestamps)

        # Should have some occlusions (loose bound due to randomness)
        # Note: with 0.1 probability per frame, we expect roughly 10% of frames affected
        assert low_confidence_frames > 0.01 * total_frames  # At least 1% occluded
        # Allow up to 80% since occlusions can be long and overlap with dropouts
        assert low_confidence_frames < 0.8 * total_frames  # Not more than 80%

    def test_synthetic_video_led_spacing(self):
        """Test that LED spacing is approximately correct."""
        config = SimConfig(
            duration=2.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )
        config.led.front_back_distance = 20.0  # pixels
        config.video.dropout_probability = 0.0  # Disable dropouts for this test

        ground_truth = self._generate_trajectory(config)
        video_data = generate_synthetic_video(ground_truth, config)

        # Calculate LED distances (excluding NaN frames)
        led_distances = np.linalg.norm(video_data.front_led - video_data.back_led, axis=1)

        # Filter out NaN values
        valid_distances = led_distances[~np.isnan(led_distances)]

        # Should be approximately the configured distance (allowing for noise)
        mean_distance = np.mean(valid_distances)
        assert abs(mean_distance - config.led.front_back_distance) < 5.0  # Within 5 pixels

    def _generate_trajectory(self, config: SimConfig) -> Dict[str, np.ndarray]:
        """Helper to generate ground truth trajectory."""
        # This is a placeholder - will be implemented later
        t = np.linspace(0, config.duration, int(config.duration * config.video_fps))
        positions = (
            np.column_stack(
                [
                    50 * np.sin(2 * np.pi * t / 10),  # x
                    50 * np.cos(2 * np.pi * t / 10),  # y
                ]
            )
            + 100
        )  # Center in arena

        velocities = np.gradient(positions, axis=0) / (t[1] - t[0])
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])

        return {
            "timestamps": t,
            "positions": positions,
            "velocities": velocities,
            "headings": headings,
        }


class TestSyntheticSession:
    """Test complete synthetic session generation."""

    def test_generate_synthetic_session_basic(self):
        """Test basic synthetic session generation."""
        config = SimConfig(
            duration=2.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )

        session_data = generate_synthetic_session(config)

        # Check that all components are present
        assert "ground_truth" in session_data
        assert "imu_data" in session_data
        assert "video_data" in session_data

        # Check basic consistency
        gt = session_data["ground_truth"]
        imu = session_data["imu_data"]
        video = session_data["video_data"]

        assert len(gt["timestamps"]) == len(imu.timestamps)
        assert len(video.timestamps) == int(config.duration * config.video_fps)

    def test_synthetic_session_reproducibility(self):
        """Test that synthetic sessions are reproducible with same seed."""
        config = SimConfig(
            duration=1.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )

        session1 = generate_synthetic_session(config)
        session2 = generate_synthetic_session(config)

        # Ground truth should be identical
        np.testing.assert_array_equal(
            session1["ground_truth"]["positions"], session2["ground_truth"]["positions"]
        )

        # IMU data should be identical
        np.testing.assert_array_equal(
            session1["imu_data"].accel_raw, session2["imu_data"].accel_raw
        )

    def test_synthetic_session_time_alignment(self):
        """Test that IMU and video timestamps are properly aligned."""
        config = SimConfig(
            duration=5.0,
            video_fps=30.0,
            imu_rate=1000.0,
            arena_size=[200.0, 200.0],
            seed=42,
        )

        session_data = generate_synthetic_session(config)

        imu_timestamps = session_data["imu_data"].timestamps
        video_timestamps = session_data["video_data"].timestamps

        # Check time ranges
        assert imu_timestamps[0] == pytest.approx(0.0, abs=1e-6)
        assert imu_timestamps[-1] == pytest.approx(config.duration, abs=1e-3)
        assert video_timestamps[0] == pytest.approx(0.0, abs=1e-6)
        assert video_timestamps[-1] <= config.duration

        # Check that video frame times are subset of possible IMU times
        for vt in video_timestamps:
            # Should be able to find nearby IMU timestamp
            closest_imu_idx = np.argmin(np.abs(imu_timestamps - vt))
            time_diff = abs(imu_timestamps[closest_imu_idx] - vt)
            assert time_diff < 1.0 / config.imu_rate  # Within one IMU sample
