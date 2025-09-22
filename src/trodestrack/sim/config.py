"""Configuration schemas for synthetic data generation."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TrajectoryConfig(BaseModel):
    """Configuration for synthetic trajectory generation."""
    max_speed: float = Field(default=40.0, gt=0, description="Maximum speed in cm/s")
    max_acceleration: float = Field(default=80.0, gt=0, description="Maximum acceleration in cm/s²")
    turn_probability: float = Field(default=0.01, ge=0, le=1, description="Probability of direction change per frame")


class IMUConfig(BaseModel):
    """Configuration for synthetic IMU data generation."""
    accel_noise_std: float = Field(default=0.05, ge=0, description="Accelerometer noise standard deviation (g)")
    gyro_noise_std: float = Field(default=0.02, ge=0, description="Gyroscope noise standard deviation (deg/s)")
    accel_bias_std: float = Field(default=0.01, ge=0, description="Accelerometer bias standard deviation (g)")
    gyro_bias_std: float = Field(default=0.005, ge=0, description="Gyroscope bias standard deviation (deg/s)")
    bias_drift_std: float = Field(default=0.0005, ge=0, description="Bias drift standard deviation per second")
    misalignment_deg: float = Field(default=1.0, ge=0, le=45, description="IMU misalignment in degrees")


class VideoConfig(BaseModel):
    """Configuration for synthetic video data generation."""
    position_noise_std: float = Field(default=1.5, ge=0, description="Position noise standard deviation (pixels)")
    confidence_min: float = Field(default=0.05, ge=0, le=1, description="Minimum detection confidence")
    confidence_max: float = Field(default=0.98, ge=0, le=1, description="Maximum detection confidence")
    occlusion_probability: float = Field(default=0.02, ge=0, le=1, description="Probability of occlusion per frame")
    occlusion_duration_mean: float = Field(default=0.5, gt=0, description="Mean occlusion duration in seconds")
    led_swap_probability: float = Field(default=0.005, ge=0, le=1, description="Probability of LED swap per frame")
    dropout_probability: float = Field(default=0.01, ge=0, le=1, description="Probability of frame dropout")

    @field_validator('confidence_max')
    @classmethod
    def confidence_max_greater_than_min(cls, v, info):
        if 'confidence_min' in info.data and v <= info.data['confidence_min']:
            raise ValueError('confidence_max must be greater than confidence_min')
        return v


class LEDConfig(BaseModel):
    """Configuration for LED positioning and behavior."""
    front_back_distance: float = Field(default=20.0, gt=0, description="Distance between front and back LEDs (pixels)")
    swap_detection_threshold: float = Field(default=0.9, ge=0, le=1, description="Threshold for detecting LED swaps")


class SimConfig(BaseModel):
    """Configuration for synthetic session generation."""

    # Required parameters
    duration: float = Field(gt=0, description="Session duration in seconds")
    video_fps: float = Field(gt=0, description="Video frame rate in Hz")
    imu_rate: float = Field(gt=0, description="IMU sampling rate in Hz")
    arena_size: List[float] = Field(description="Arena dimensions [width, height] in cm")
    seed: int = Field(description="Random seed for reproducibility")

    # Optional nested configurations
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    imu: IMUConfig = Field(default_factory=IMUConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    led: LEDConfig = Field(default_factory=LEDConfig)

    @field_validator('duration')
    @classmethod
    def duration_positive(cls, v):
        if v <= 0:
            raise ValueError('Duration must be positive')
        return v

    @field_validator('video_fps')
    @classmethod
    def video_fps_positive(cls, v):
        if v <= 0:
            raise ValueError('Video FPS must be positive')
        return v

    @field_validator('imu_rate')
    @classmethod
    def imu_rate_positive(cls, v):
        if v <= 0:
            raise ValueError('IMU rate must be positive')
        return v

    @field_validator('arena_size')
    @classmethod
    def arena_size_positive(cls, v):
        if len(v) != 2:
            raise ValueError('Arena size must have exactly 2 dimensions')
        if any(dim <= 0 for dim in v):
            raise ValueError('Arena size must have positive dimensions')
        return v

    @field_validator('seed')
    @classmethod
    def seed_non_negative(cls, v):
        if v < 0:
            raise ValueError('Seed must be non-negative')
        return v