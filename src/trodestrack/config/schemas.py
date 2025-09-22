"""Configuration schemas using Pydantic for trodestrack."""

from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class IMUConfig(BaseModel):
    """Configuration for IMU processing."""

    downsampling_rate: float = Field(
        default=1000.0,
        description="Target IMU rate for processing (Hz)",
        gt=0.0
    )
    alignment_correction: Optional[List[List[float]]] = Field(
        default=None,
        description="3x3 rotation matrix for IMU-to-body frame alignment"
    )
    accel_scale: float = Field(
        default=0.000061,
        description="Raw accelerometer to g conversion factor"
    )
    gyro_scale: float = Field(
        default=0.061,
        description="Raw gyroscope to deg/s conversion factor"
    )


class LEDConfig(BaseModel):
    """Configuration for LED tracking."""

    front_back_distance_cm: float = Field(
        description="Expected distance between front and back LEDs (cm)",
        gt=0.0
    )
    swap_policy: str = Field(
        default="wrapped_residual",
        description="Policy for handling LED swaps",
        pattern="^(wrapped_residual|mixture_update|disabled)$"
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence for LED detections",
        ge=0.0,
        le=1.0
    )


class MappingConfig(BaseModel):
    """Configuration for pixel-to-world coordinate mapping."""

    type: str = Field(
        description="Mapping type",
        pattern="^(homography|ruler_scale)$"
    )
    homography_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="3x3 homography matrix for pixel->cm transformation"
    )
    pixel_per_cm: Optional[float] = Field(
        default=None,
        description="Ruler-based scale (pixels per cm)",
        gt=0.0
    )
    arena_bounds: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="Arena bounds in cm (x_min, y_min, x_max, y_max)"
    )

    @field_validator('homography_matrix')
    @classmethod
    def validate_homography_matrix(cls, v):
        if v is not None:
            if len(v) != 3 or any(len(row) != 3 for row in v):
                raise ValueError("Homography matrix must be 3x3")
        return v

    @model_validator(mode='after')
    def validate_mapping_consistency(self):
        if self.type == 'ruler_scale' and self.pixel_per_cm is None:
            raise ValueError("pixel_per_cm required when type is 'ruler_scale'")
        if self.type == 'homography' and self.homography_matrix is None:
            raise ValueError("homography_matrix required when type is 'homography'")
        return self


class FilterConfig(BaseModel):
    """Configuration for Kalman filtering."""

    filter_type: str = Field(
        default="ekf",
        description="Filter type to use",
        pattern="^(ekf|ukf)$"
    )
    process_noise: dict = Field(
        default_factory=lambda: {
            "position": 0.01,
            "velocity": 0.1,
            "heading": 0.01,
            "bias_gyro": 1e-6,
            "bias_accel": 1e-4
        },
        description="Process noise variances for state components"
    )
    measurement_noise: dict = Field(
        default_factory=lambda: {
            "position": 1.0,
            "heading": 0.1
        },
        description="Measurement noise variances"
    )
    gating_threshold: float = Field(
        default=9.21,
        description="Mahalanobis distance threshold for outlier rejection (χ² p=0.01, df=2)",
        gt=0.0
    )
    velocity_damping: float = Field(
        default=0.0,
        description="Velocity damping coefficient (λ in dynamics)",
        ge=0.0
    )
    initial_state_variance: dict = Field(
        default_factory=lambda: {
            "position": 1.0,
            "velocity": 10.0,
            "heading": 0.1,
            "bias_gyro": 1.0,
            "bias_accel": 1.0
        },
        description="Initial state covariance diagonal values"
    )


class OutputConfig(BaseModel):
    """Configuration for output files and logging."""

    output_dir: Path = Field(
        description="Directory for output files"
    )
    save_states: bool = Field(
        default=True,
        description="Save state estimates to parquet"
    )
    save_residuals: bool = Field(
        default=True,
        description="Save measurement residuals and diagnostics"
    )
    save_plots: bool = Field(
        default=True,
        description="Generate and save diagnostic plots"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$"
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )


class SessionConfig(BaseModel):
    """Top-level configuration for a tracking session."""

    # Input files
    video_file: Optional[Path] = Field(
        default=None,
        description="Path to video detection file (Trodes LED or DLC)"
    )
    imu_file: Optional[Path] = Field(
        default=None,
        description="Path to IMU data file (SpikeGadgets format)"
    )

    # Frame rate and timing
    video_fps: float = Field(
        default=30.0,
        description="Video frame rate (Hz)",
        gt=0.0
    )

    # Configuration components
    mapping: MappingConfig = Field(
        description="Coordinate mapping configuration"
    )
    filter: FilterConfig = Field(
        default_factory=FilterConfig,
        description="Kalman filter configuration"
    )
    led: Optional[LEDConfig] = Field(
        default=None,
        description="LED tracking configuration (if using LEDs)"
    )
    imu: IMUConfig = Field(
        default_factory=IMUConfig,
        description="IMU processing configuration"
    )
    output: OutputConfig = Field(
        description="Output configuration"
    )

    @field_validator('video_file')
    @classmethod
    def validate_video_file_exists(cls, v):
        if v is not None and not v.exists():
            raise ValueError(f"Video file does not exist: {v}")
        return v

    @field_validator('imu_file')
    @classmethod
    def validate_imu_file_exists(cls, v):
        if v is not None and not v.exists():
            raise ValueError(f"IMU file does not exist: {v}")
        return v

    @field_validator('led')
    @classmethod
    def validate_led_config(cls, v):
        # LED config is optional, but if provided, should be valid
        return v

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"  # Reject unknown fields
    )