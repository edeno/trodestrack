"""
Rat IMU + Vision Simulator

Generates synthetic ground-truth trajectories with realistic IMU measurements
(gyroscope, accelerometer) and camera observations (LED positions with confidence).

Features:
- 2D planar motion with heading (5D state: x, y, vx, vy, θ)
- Ornstein-Uhlenbeck processes for smooth, realistic motion
- IMU measurements matching SpikeGadgets hardware (default 104 Hz sensor refresh rate)
- Camera observations at video rate (default 30 Hz)
- Optional second LED for heading measurements
- Optional confidence scores with correlation to dropouts/occlusions
- Proper IMU physics: specific force (f = a - g), gravity with tilt, bias random walks, white noise
- Arena boundary reflections with energy loss

Coordinate Frames:
- World: right-handed, x=horizontal, y=vertical, θ=heading (0=right, CCW+)
- Body: attached to rat, x=forward, y=left, z=up (right-handed)
- IMU: aligned with body (small misalignment optionally configurable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields

import numpy as np

from .utils import (
    SimOut,
    confidence_to_noise_scale,
    density_to_sample_std,
    interp_angle,
    ou_step,
    rw_step,
    wrap_angle,
)

# -----------------------------------------------------------------------------
# Gravity & IMU-specific utilities (not in utils.py)
# -----------------------------------------------------------------------------


def _make_strictly_increasing_within_bounds(
    times: np.ndarray,
    *,
    lower: float,
    upper: float,
    min_step: float,
) -> np.ndarray:
    """Separate sorted timestamps while keeping them inside fixed bounds."""
    out = np.clip(np.asarray(times, dtype=float), lower, upper).copy()
    if out.size <= 1:
        return out

    span = upper - lower
    if span <= 0:
        raise ValueError("timestamp bounds must have positive span")

    step = min(float(min_step), span / (out.size - 1))
    if step <= 0:
        step = span / (out.size - 1)

    for idx in range(1, out.size):
        out[idx] = max(out[idx], out[idx - 1] + step)

    if out[-1] > upper:
        out[-1] = upper
        for idx in range(out.size - 2, -1, -1):
            out[idx] = min(out[idx], out[idx + 1] - step)

    if out[0] < lower:
        out[0] = lower
        for idx in range(1, out.size):
            out[idx] = max(out[idx], out[idx - 1] + step)

    return out


def compute_gravity_in_tilted_frame(
    tilt_roll_rad: float, tilt_pitch_rad: float, gravity: float
) -> tuple[float, float]:
    """Gravity components in IMU frame with small roll/pitch tilt.

    In 2D planar motion (yaw only), the IMU is nominally level (z-up).
    Small mounting errors cause roll/pitch tilt, projecting gravity
    into the x-y (horizontal) plane.

    3D rotation sequence (ZYX Euler): Yaw → Pitch → Roll
    For planar motion with yaw handled separately, we only need pitch/roll tilt.

    Gravity in world frame: g_world = [0, 0, -g]
    After pitch (θ_p) and roll (θ_r) rotations:
        g_x = g * sin(θ_p)
        g_y = -g * sin(θ_r) * cos(θ_p)

    Parameters
    ----------
    tilt_roll_rad : float
        Roll tilt angle (rad) about x-axis.
    tilt_pitch_rad : float
        Pitch tilt angle (rad) about y-axis.
    gravity : float
        Gravity magnitude g (m/s^2).

    Returns
    -------
    tuple[float, float]
        (g_x, g_y) components (m/s^2) in the tilted IMU x-y plane.

    Notes
    -----
    For small angles: g_x ≈ g θ_p, g_y ≈ −g θ_r cos(θ_p).
    """
    cos_p = np.cos(tilt_pitch_rad)
    sin_p = np.sin(tilt_pitch_rad)
    sin_r = np.sin(tilt_roll_rad)

    g_x = gravity * sin_p
    g_y = -gravity * sin_r * cos_p

    return g_x, g_y


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Module-level constants for dataclass defaults (RUF009 compliance)
# These avoid function calls in dataclass field defaults
_GYRO_NOISE_DENSITY = np.deg2rad(0.01)  # 0.01 °/s/√Hz (SpikeGadgets spec)
_GYRO_BIAS_RW_DENSITY = np.deg2rad(0.003)  # rad/s / √s
_SIGMA_YAW_RATE = np.deg2rad(60.0)  # rad/s / √s


@dataclass
class RatIMUSimConfig:
    """
    Configuration for rat IMU + vision simulation.

    Durations and Rates:
        duration_s: Total simulation duration in seconds
        fs_imu: IMU sampling rate in Hz
        fs_cam: Camera sampling rate in Hz

    Arena (meters):
        arena_w: Arena width (x-axis)
        arena_h: Arena height (y-axis)

    Camera Model:
        cam_sigma_m: Camera measurement noise std per axis (meters)
        cam_dropout_prob: Probability of frame dropout (0-1)
        cam_dropout_correlation: Correlation between LED1/LED2 dropouts (0=independent, 1=same)
        cam_latency_s: Constant latency applied to timestamps (seconds)
        cam_jitter_s: Timestamp jitter std (seconds)
        use_confidence: Enable confidence score generation
        confidence_base: Base confidence when detection is good (0-1)
        confidence_dropout_decay: Confidence multiplier near dropouts

    LED Configuration:
        led1_offset_body: Position of LED1 in body frame [x, y] meters
        use_second_led: Enable second LED for heading measurements
        led2_offset_body: Position of LED2 in body frame [x, y] meters

    LED Swap Configuration:
        led_swap_mode: Swap behavior mode ("per_frame" or "persistent")
            - "per_frame": Independent per-frame swaps (legacy behavior)
            - "persistent": Event-based swaps that persist for a duration
        led_swap_prob: Probability of swapping LED1/LED2 labels per visible frame (0-1)
            Used in "per_frame" mode only
        led_swap_rate: Mean number of swap events per second (persistent mode)
            Example: 0.5 = one swap every 2 seconds on average
            Units: events/second
        led_swap_duration_mean: Mean duration of each swap event (seconds)
            Example: 1.0 = swaps last 1 second on average
            Units: seconds
        led_swap_duration_std: Standard deviation of swap duration (seconds)
            Example: 0.3 = 0.3 second variability in duration
            Units: seconds
        led_wall_reflection_prob: Probability of LED reflection artifacts near walls (0-1)
            Example: 0.3 = 30% chance of mirrored detection when near wall
            Range: [0, 1]
            Units: probability (0 = disabled, 1 = always reflect when near wall)
        led_wall_reflection_distance: Distance threshold from wall (m) within which reflections can occur
            Example: 0.2 = reflections possible within 20cm of arena boundaries
            Range: [0, inf), typically 0.1-0.5m
            Units: meters (distance from rat center to nearest wall)

    IMU Noise (densities per √Hz for white noise):
        gyro_noise_density: Gyroscope white noise (rad/s / √Hz)
        accel_noise_density: Accelerometer white noise (m/s² / √Hz)

    IMU Bias Random Walks (densities per √s):
        gyro_bias_rw_density: Gyroscope bias random walk (rad/s / √s)
        accel_bias_rw_density: Accelerometer bias random walk (m/s² / √s)

    IMU Mounting and Gravity:
        imu_tilt_roll_deg: Small constant roll misalignment (degrees)
        imu_tilt_pitch_deg: Small constant pitch misalignment (degrees)
        gravity: Gravitational acceleration magnitude (m/s²)

    Motion Model (Ornstein-Uhlenbeck parameters):
        tau_yaw_rate: Time constant for yaw rate (seconds)
        sigma_yaw_rate: Noise intensity for yaw rate (rad/s / √s)
        tau_a_fwd: Time constant for forward acceleration (seconds)
        sigma_a_fwd: Noise intensity for forward accel (m/s² / √s)
        tau_a_lat: Time constant for lateral acceleration (seconds)
        sigma_a_lat: Noise intensity for lateral accel (m/s² / √s)

    Physical Constraints:
        vel_drag: Linear velocity damping coefficient (1/s) [DEPRECATED: use drag_fwd/drag_lat]
        drag_fwd: Forward drag coefficient in body frame (1/s)
        drag_lat: Lateral drag coefficient in body frame (1/s)
        speed_clip: Maximum speed clipping threshold (m/s)

    Note on drag:
        - Anisotropic drag: drag_fwd (streamlined) < drag_lat (sideways sliding)
        - Applied in body frame: drag rotates with animal heading
        - For backward compatibility: if only vel_drag is set, drag_fwd = drag_lat = vel_drag
        - If drag_fwd/drag_lat are set, they override vel_drag

    Initial State:
        m0: Initial state mean [x, y, vx, vy, θ]
        P0: Initial state covariance (5×5)
    """

    # Durations / rates
    duration_s: float = 60.0
    fs_imu: float = 104.0  # SpikeGadgets hardware sensor refresh rate
    fs_cam: float = 30.0

    # Arena (meters)
    arena_w: float = 2.0
    arena_h: float = 2.0

    # Camera model
    cam_sigma_m: float = 0.005  # 5 mm std noise per axis
    cam_dropout_prob: float = 0.10
    cam_dropout_correlation: float = (
        0.8  # Correlation between LED1/LED2 dropouts (0=independent, 1=identical)
    )
    cam_latency_s: float = 0.05
    cam_jitter_s: float = 0.005
    use_confidence: bool = False
    confidence_base: float = 0.95
    confidence_dropout_decay: float = 0.3

    # LED configuration
    # Convention: LED1 is rear, LED2 is front
    # Body frame: x=forward, y=left, z=up (right-handed)
    # LED vector (LED2 - LED1) points forward along heading
    led1_offset_body: np.ndarray = field(
        default_factory=lambda: np.array([-0.02, 0.0])
    )  # Rear LED (2cm behind center)
    use_second_led: bool = False
    led2_offset_body: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.0])
    )  # Front LED (2cm ahead of center)
    # LED swap configuration
    led_swap_mode: str = "per_frame"  # "per_frame" or "persistent"
    led_swap_prob: float = (
        0.0  # Probability of swapping LED1/LED2 labels per frame (per_frame mode)
    )
    led_swap_rate: float = 0.5  # Mean swap events per second (persistent mode)
    led_swap_duration_mean: float = (
        1.0  # Mean duration of swap event in seconds (persistent mode)
    )
    led_swap_duration_std: float = (
        0.3  # Std dev of swap duration in seconds (persistent mode)
    )

    led_wall_reflection_prob: float = (
        0.0  # Probability of LED reflection artifacts near walls (0-1)
    )
    led_wall_reflection_distance: float = (
        0.2  # Distance from wall (m) within which reflections can occur
    )

    # IMU white noise densities (per √Hz) - SpikeGadgets specifications
    gyro_noise_density: float = _GYRO_NOISE_DENSITY  # 0.01 °/s/√Hz
    accel_noise_density: float = 0.00196133  # 0.2 mg/√Hz = 0.0002g * 9.80665

    # IMU bias random-walk densities (per √s)
    gyro_bias_rw_density: float = _GYRO_BIAS_RW_DENSITY  # rad/s / √s
    accel_bias_rw_density: float = 0.005  # m/s² / √s

    # IMU mounting/tilt (small constant roll/pitch misalignment)
    imu_tilt_roll_deg: float = 3.0  # Roll tilt in degrees
    imu_tilt_pitch_deg: float = 2.0  # Pitch tilt in degrees
    gravity: float = 9.80665  # m/s² (Earth gravity)

    # Motion model (OU parameters)
    tau_yaw_rate: float = 0.8  # s
    sigma_yaw_rate: float = _SIGMA_YAW_RATE  # 60 °/s / √s
    tau_a_fwd: float = 0.7  # s
    sigma_a_fwd: float = 1.0  # m/s² / √s
    tau_a_lat: float = 0.5  # s
    sigma_a_lat: float = 0.5  # m/s² / √s

    # Physical damping / limits
    vel_drag: float = 0.4  # 1/s (deprecated, use drag_fwd/drag_lat)
    drag_fwd: float | None = None  # Forward drag in body frame (1/s)
    drag_lat: float | None = None  # Lateral drag in body frame (1/s)
    speed_clip: float = 1.5  # m/s

    # Initial state (truth)
    m0: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    P0: np.ndarray = field(
        default_factory=lambda: np.diag([0.01, 0.01, 0.05, 0.05, np.deg2rad(5.0) ** 2])
    )

    def __post_init__(self):
        """Validate configuration parameters."""
        import warnings

        # Reject non-finite scalar parameters up front. The downstream
        # checks use bare comparisons like ``<= 0`` / ``< 0`` which NaN
        # silently passes (NaN compares False to both), and the simulator
        # then emits non-finite IMU and/or truth arrays. Listing the
        # numeric fields explicitly so a future field addition trips
        # AttributeError rather than silently dropping out of the gate.
        finite_scalar_fields = (
            "duration_s",
            "fs_imu",
            "fs_cam",
            "arena_w",
            "arena_h",
            "cam_dropout_prob",
            "cam_dropout_correlation",
            "cam_jitter_s",
            "cam_latency_s",
            "led_swap_prob",
            "led_swap_rate",
            "led_swap_duration_mean",
            "led_swap_duration_std",
            "led_wall_reflection_prob",
            "led_wall_reflection_distance",
            "confidence_base",
            "confidence_dropout_decay",
            "speed_clip",
            "gravity",
            "imu_tilt_roll_deg",
            "imu_tilt_pitch_deg",
            "cam_sigma_m",
            "gyro_noise_density",
            "accel_noise_density",
            "gyro_bias_rw_density",
            "accel_bias_rw_density",
            "tau_yaw_rate",
            "tau_a_fwd",
            "tau_a_lat",
            "sigma_yaw_rate",
            "sigma_a_fwd",
            "sigma_a_lat",
            "vel_drag",
        )
        for fname in finite_scalar_fields:
            value = getattr(self, fname)
            if not np.isfinite(value):
                raise ValueError(f"{fname} must be a finite value; got {value!r}.")

        # Duration validation
        if self.duration_s <= 0:
            raise ValueError(
                f"Simulation duration must be positive, got {self.duration_s}s.\n"
                f"Example: duration_s=60.0 (60 seconds)"
            )

        # Sampling rate validation
        if self.fs_imu <= 0:
            raise ValueError(
                f"IMU sampling rate must be positive, got {self.fs_imu} Hz.\n"
                f"Example: fs_imu=200.0 (200 Hz)"
            )

        if self.fs_cam <= 0:
            raise ValueError(
                f"Camera sampling rate must be positive, got {self.fs_cam} Hz.\n"
                f"Example: fs_cam=30.0 (30 Hz)"
            )

        if self.fs_imu <= self.fs_cam:
            warnings.warn(
                f"IMU rate ({self.fs_imu} Hz) should typically be much higher than "
                f"camera rate ({self.fs_cam} Hz) for proper sensor fusion. "
                f"Consider fs_imu >= 100.0 Hz.",
                UserWarning,
                stacklevel=2,
            )

        # Minimum-sample-count gate. ``simulate_rat_imu`` rounds
        # ``duration_s * fs_*`` to integer sample counts; very short
        # positive durations therefore produce zero or one sample per
        # stream and crash mid-simulation (e.g. ``IndexError`` on
        # ``Z_cam_led1[0]`` or ``zero-size reduction`` on ``min(t_imu)``).
        # Require at least 2 samples for both IMU and camera streams so
        # downstream ``np.diff`` / boundary indexing is well-defined.
        T_imu = int(np.round(self.duration_s * self.fs_imu))
        T_cam = int(np.round(self.duration_s * self.fs_cam))
        if T_imu < 2 or T_cam < 2:
            raise ValueError(
                f"duration_s={self.duration_s}s at fs_imu={self.fs_imu} Hz, "
                f"fs_cam={self.fs_cam} Hz produces only T_imu={T_imu}, "
                f"T_cam={T_cam} samples; need at least 2 of each. "
                f"Increase duration_s or sampling rates."
            )

        # Camera-timestamp model validation. cam_latency_s is the
        # *exposure → arrival* latency, so a negative value would mean
        # observations arrive before they're exposed. cam_jitter_s is a
        # timestamp jitter standard deviation and a negative std is
        # meaningless. Validating finiteness alone (above) lets both
        # silently propagate into ``t_cam_obs = t_cam_exp + cam_latency_s``
        # and ``rng.standard_normal(T_cam) * cam_jitter_s``.
        if self.cam_latency_s < 0:
            raise ValueError(
                f"cam_latency_s must be non-negative (it's an exposure → "
                f"arrival latency); got {self.cam_latency_s}."
            )
        if self.cam_jitter_s < 0:
            raise ValueError(
                f"cam_jitter_s must be non-negative (it's a timestamp jitter "
                f"standard deviation); got {self.cam_jitter_s}."
            )

        # Arena validation
        if self.arena_w <= 0 or self.arena_h <= 0:
            raise ValueError(
                f"Arena dimensions must be positive, got width={self.arena_w}m, height={self.arena_h}m.\n"
                f"Example: arena_w=2.0, arena_h=2.0 (2m × 2m arena)"
            )

        # Probability validation
        if not 0 <= self.cam_dropout_prob <= 1:
            raise ValueError(
                f"Dropout probability must be in [0, 1], got {self.cam_dropout_prob}.\n"
                f"Example: cam_dropout_prob=0.15 (15% dropout rate)"
            )

        if not 0 <= self.cam_dropout_correlation <= 1:
            raise ValueError(
                f"Dropout correlation must be in [0, 1], got {self.cam_dropout_correlation}.\n"
                f"0 = independent dropouts, 1 = identical dropouts"
            )

        # LED swap validation
        if self.led_swap_mode not in ["per_frame", "persistent"]:
            raise ValueError(
                f"led_swap_mode must be 'per_frame' or 'persistent', got '{self.led_swap_mode}'.\n"
                f"Example: led_swap_mode='persistent' for event-based swaps"
            )

        if not 0 <= self.led_swap_prob <= 1:
            raise ValueError(
                f"LED swap probability must be in [0, 1], got {self.led_swap_prob}.\n"
                f"Example: led_swap_prob=0.05 (5% swap rate in per_frame mode)"
            )

        if self.led_swap_rate < 0:
            raise ValueError(
                f"led_swap_rate must be non-negative, got {self.led_swap_rate}.\n"
                f"Example: led_swap_rate=0.5 (0.5 swaps per second)"
            )

        if self.led_swap_duration_mean <= 0:
            raise ValueError(
                f"led_swap_duration_mean must be positive, got {self.led_swap_duration_mean}.\n"
                f"Example: led_swap_duration_mean=1.0 (1 second mean duration)"
            )

        if self.led_swap_duration_std < 0:
            raise ValueError(
                f"led_swap_duration_std must be non-negative, got {self.led_swap_duration_std}.\n"
                f"Example: led_swap_duration_std=0.3 (0.3 second std dev)"
            )

        if not 0 <= self.led_wall_reflection_prob <= 1:
            raise ValueError(
                f"LED wall reflection probability must be in [0, 1], got {self.led_wall_reflection_prob}.\n"
                f"Example: led_wall_reflection_prob=0.3 (30% reflection rate near walls)"
            )

        if self.led_wall_reflection_distance < 0:
            raise ValueError(
                f"LED wall reflection distance must be non-negative, got {self.led_wall_reflection_distance}m.\n"
                f"Example: led_wall_reflection_distance=0.2 (20cm threshold)"
            )

        if self.use_confidence:
            if not 0 <= self.confidence_base <= 1:
                raise ValueError(
                    f"Base confidence must be in [0, 1], got {self.confidence_base}.\n"
                    f"Example: confidence_base=0.95"
                )

            if not 0 <= self.confidence_dropout_decay <= 1:
                raise ValueError(
                    f"Confidence dropout decay must be in [0, 1], got {self.confidence_dropout_decay}.\n"
                    f"Example: confidence_dropout_decay=0.3"
                )

        # Physical parameter validation
        if self.speed_clip <= 0:
            raise ValueError(
                f"Speed clip must be positive, got {self.speed_clip} m/s.\n"
                f"Example: speed_clip=1.5 (1.5 m/s max speed)"
            )

        if self.gravity <= 0:
            raise ValueError(
                f"Gravity must be positive, got {self.gravity} m/s².\n"
                f"Standard Earth gravity: gravity=9.80665"
            )

        # Noise parameter validation
        if self.cam_sigma_m < 0:
            raise ValueError(
                f"Camera noise must be non-negative, got {self.cam_sigma_m}m.\n"
                f"Example: cam_sigma_m=0.005 (5mm std noise)"
            )

        if self.gyro_noise_density < 0 or self.accel_noise_density < 0:
            raise ValueError(
                f"IMU noise densities must be non-negative.\n"
                f"Got gyro={self.gyro_noise_density}, accel={self.accel_noise_density}"
            )

        if self.gyro_bias_rw_density < 0 or self.accel_bias_rw_density < 0:
            raise ValueError(
                f"Bias random walk densities must be non-negative.\n"
                f"Got gyro={self.gyro_bias_rw_density}, accel={self.accel_bias_rw_density}"
            )

        # Time constant validation
        if self.tau_yaw_rate <= 0 or self.tau_a_fwd <= 0 or self.tau_a_lat <= 0:
            raise ValueError(
                f"OU time constants must be positive.\n"
                f"Got tau_yaw_rate={self.tau_yaw_rate}, tau_a_fwd={self.tau_a_fwd}, tau_a_lat={self.tau_a_lat}"
            )

        # Drag coefficient validation and backward compatibility
        # Priority: drag_fwd/drag_lat > vel_drag
        if self.drag_fwd is None and self.drag_lat is None:
            # Use legacy vel_drag for both
            self.drag_fwd = self.vel_drag
            self.drag_lat = self.vel_drag
        elif self.drag_fwd is None or self.drag_lat is None:
            raise ValueError(
                "Must specify both drag_fwd and drag_lat, or neither (use vel_drag).\n"
                f"Got drag_fwd={self.drag_fwd}, drag_lat={self.drag_lat}"
            )

        # Validate drag coefficients are non-negative and finite (NaN
        # passes the bare <0 comparison and propagates through dynamics).
        if not np.isfinite(self.drag_fwd) or not np.isfinite(self.drag_lat):
            raise ValueError(
                "drag_fwd and drag_lat must be finite values; got "
                f"drag_fwd={self.drag_fwd!r}, drag_lat={self.drag_lat!r}."
            )
        if self.drag_fwd < 0:
            raise ValueError(
                f"Forward drag coefficient must be non-negative, got {self.drag_fwd} 1/s.\n"
                f"Example: drag_fwd=0.3 (low streamlined drag)"
            )

        if self.drag_lat < 0:
            raise ValueError(
                f"Lateral drag coefficient must be non-negative, got {self.drag_lat} 1/s.\n"
                f"Example: drag_lat=1.2 (high lateral sliding drag)"
            )

        # LED offset shape / finiteness validation. Offsets are documented
        # as body-frame [x, y] arrays; the simulator later indexes ``[0]``
        # / ``[1]`` and folds them into camera observations, so a scalar
        # offset raises a raw IndexError mid-simulation and a NaN offset
        # silently propagates non-finite world-frame LED positions.
        for fname in ("led1_offset_body", "led2_offset_body"):
            value = np.asarray(getattr(self, fname))
            if value.ndim != 1 or value.shape[0] != 2:
                raise ValueError(
                    f"{fname} must be a body-frame [x, y] array of shape (2,); "
                    f"got shape {value.shape}."
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{fname} must contain only finite values; got {value!r}."
                )

        # LED configuration validation
        if not self.use_second_led and self.led_swap_prob > 0:
            warnings.warn(
                f"LED swap probability is {self.led_swap_prob} but use_second_led=False. "
                f"Swaps require two LEDs. Set use_second_led=True or led_swap_prob=0.0",
                UserWarning,
                stacklevel=2,
            )
        if (
            not self.use_second_led
            and self.led_swap_mode == "persistent"
            and self.led_swap_rate > 0
        ):
            warnings.warn(
                f"led_swap_mode='persistent' with led_swap_rate="
                f"{self.led_swap_rate} but use_second_led=False. Persistent "
                f"swaps require two LEDs. Set use_second_led=True or "
                f"led_swap_rate=0.0 (or led_swap_mode='per_frame').",
                UserWarning,
                stacklevel=2,
            )

        # Initial state validation
        if self.m0.shape != (5,):
            raise ValueError(
                f"Initial state m0 must have shape (5,), got {self.m0.shape}.\n"
                f"Expected: [x, y, vx, vy, theta]"
            )

        if self.P0.shape != (5, 5):
            raise ValueError(
                f"Initial covariance P0 must have shape (5, 5), got {self.P0.shape}"
            )

        # Initial-state content validation. The simulator samples the
        # truth trajectory from m0 and P0 — non-finite m0 produces NaN
        # truth, and a non-PSD P0 raises a raw LinAlgError from
        # np.random.multivariate_normal mid-simulation. Catch the actual
        # contract violation here.
        if not np.all(np.isfinite(self.m0)):
            raise ValueError(f"m0 must contain finite values; got {self.m0!r}.")
        if not np.all(np.isfinite(self.P0)):
            raise ValueError(
                f"P0 must contain finite values; got non-finite entries in {self.P0!r}."
            )
        # Symmetric-PSD check. ``np.linalg.cholesky`` would only accept
        # strictly *positive-definite* matrices, contradicting the
        # documented contract and rejecting useful edge cases like
        # ``P0 == 0`` for a deterministic initial state. Use ``eigvalsh``
        # (which works for PSD) and tolerate small negative eigenvalues
        # from floating-point noise.
        if not np.allclose(self.P0, self.P0.T, atol=1e-10):
            raise ValueError(
                "P0 must be symmetric (P0 == P0.T); got an asymmetric matrix."
            )
        eigvals = np.linalg.eigvalsh(self.P0)
        psd_tol = 1e-10
        if eigvals.min() < -psd_tol:
            raise ValueError(
                "P0 must be a positive semi-definite covariance matrix; got "
                f"minimum eigenvalue {eigvals.min():.3e} (tolerance {psd_tol:.0e})."
            )


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------


def simulate_rat_imu(config: RatIMUSimConfig | None = None, seed: int = 0) -> SimOut:
    """Simulate ground truth trajectory, IMU measurements, and camera observations.

    Parameters
    ----------
    config : RatIMUSimConfig, optional
        Simulation configuration (uses defaults if None).
    seed : int, default 0
        Random seed.

    Returns
    -------
    SimOut
        Output with time arrays (t_imu, t_cam_exp, t_cam_obs), ground truth
        channels (X_truth (T_imu,5), yaw_rate_truth, accel_world_truth,
        accel_body_truth), IMU measurements (U_imu (T_imu,3) = [ω_z(rad/s), f_x, f_y]
        with biases), and camera measurements (Z_cam_led1/2 (T_cam,2), masks,
        confidences), plus metadata (config).
    """
    if config is None:
        config = RatIMUSimConfig()

    rng = np.random.default_rng(seed)

    # Type assertions for mypy (config.__post_init__ ensures these are not None)
    assert config.drag_fwd is not None, "drag_fwd should be set by __post_init__"
    assert config.drag_lat is not None, "drag_lat should be set by __post_init__"

    # Time bases
    dt_imu = 1.0 / config.fs_imu
    dt_cam = 1.0 / config.fs_cam
    T_imu = int(np.round(config.duration_s * config.fs_imu))
    T_cam = int(np.round(config.duration_s * config.fs_cam))

    t_imu = np.arange(T_imu) * dt_imu
    t_cam_clean = np.arange(T_cam) * dt_cam

    # Initialize truth state. Prefer Cholesky for the strictly
    # positive-definite case so existing seeded tests stay bit-compatible
    # (Cholesky and an eigendecomposition give different but equally
    # valid square roots, and either choice produces a different sample
    # for a fixed RNG state). Fall back to an eigendecomposition-based
    # square root only for PSD-but-not-PD covariances (including
    # ``P0 == 0`` for a deterministic initial state), where Cholesky
    # would fail.
    try:
        L0 = np.linalg.cholesky(config.P0)
    except np.linalg.LinAlgError:
        eigvals_p0, eigvecs_p0 = np.linalg.eigh(config.P0)
        eigvals_p0 = np.clip(eigvals_p0, 0.0, None)
        L0 = eigvecs_p0 * np.sqrt(eigvals_p0)
    x = config.m0 + L0 @ rng.standard_normal(5)
    x[4] = wrap_angle(x[4])  # wrap θ

    # Initialize biases (type: float | ndarray to support rw_step return type)
    b_gyro: float | np.ndarray = 0.0
    b_accel_x: float | np.ndarray = 0.0
    b_accel_y: float | np.ndarray = 0.0

    # Storage arrays
    X_truth = np.zeros((T_imu, 5))
    U_imu = np.zeros((T_imu, 3))
    bias_gyro = np.zeros(T_imu)
    bias_accel_x = np.zeros(T_imu)
    bias_accel_y = np.zeros(T_imu)

    # Ground truth channels for validation
    yaw_rate_truth = np.zeros(T_imu)
    accel_world_truth = np.zeros((T_imu, 2))
    accel_body_truth = np.zeros((T_imu, 2))
    specific_force_truth = np.zeros((T_imu, 2))

    # IMU noise parameters
    std_gyro = density_to_sample_std(config.gyro_noise_density, dt_imu)
    std_accel = density_to_sample_std(config.accel_noise_density, dt_imu)

    # Compute gravity in tilted IMU frame (constant for entire simulation)
    tilt_roll_rad = np.deg2rad(config.imu_tilt_roll_deg)
    tilt_pitch_rad = np.deg2rad(config.imu_tilt_pitch_deg)
    g_imu_x, g_imu_y = compute_gravity_in_tilted_frame(
        tilt_roll_rad, tilt_pitch_rad, config.gravity
    )

    # OU process states (control inputs in body frame)
    yaw_rate = 0.0
    a_fwd = 0.0  # forward accel (m/s²)
    a_lat = 0.0  # lateral accel (m/s²)

    # Simulate at IMU rate
    for t in range(T_imu):
        px, py, vx, vy, theta = x

        # --- 1) Update OU processes (smooth motion generators) ---
        yaw_rate = ou_step(
            yaw_rate,
            mean=0.0,
            tau=config.tau_yaw_rate,
            sigma=config.sigma_yaw_rate,
            dt=dt_imu,
            rng=rng,
        )
        a_fwd = ou_step(
            a_fwd,
            mean=0.0,
            tau=config.tau_a_fwd,
            sigma=config.sigma_a_fwd,
            dt=dt_imu,
            rng=rng,
        )
        a_lat = ou_step(
            a_lat,
            mean=0.0,
            tau=config.tau_a_lat,
            sigma=config.sigma_a_lat,
            dt=dt_imu,
            rng=rng,
        )

        # --- 2) Rotate body-frame control to world frame ---
        c, s = np.cos(theta), np.sin(theta)
        ax_control_world = c * a_fwd - s * a_lat
        ay_control_world = s * a_fwd + c * a_lat

        # --- 3) Apply anisotropic velocity damping (drag in body frame) ---
        # Transform world velocity to body frame
        vx_body = c * vx + s * vy  # Forward velocity (along heading)
        vy_body = -s * vx + c * vy  # Lateral velocity (perpendicular to heading)

        # Apply drag in body frame (forward vs lateral drag differ)
        ax_drag_body = -config.drag_fwd * vx_body
        ay_drag_body = -config.drag_lat * vy_body

        # Transform drag back to world frame
        ax_drag_world = c * ax_drag_body - s * ay_drag_body
        ay_drag_world = s * ax_drag_body + c * ay_drag_body

        # Total acceleration in world frame
        ax_world = ax_control_world + ax_drag_world
        ay_world = ay_control_world + ay_drag_world

        # --- 4) Integrate kinematics (semi-implicit Euler with second-order position) ---
        vx_new = vx + ax_world * dt_imu
        vy_new = vy + ay_world * dt_imu

        # Smooth speed saturation (differentiable for JAX compatibility)
        # Uses tanh-based saturation: v_sat = v_max * tanh(v / v_max)
        speed = np.hypot(vx_new, vy_new)
        if speed > 1e-6:  # Avoid division by zero
            # Saturation factor: tanh(speed / speed_clip) ≈ 1 for speed << speed_clip
            #                                              ≈ speed_clip/speed for speed >> speed_clip
            sat_factor = np.tanh(speed / config.speed_clip) / (
                speed / config.speed_clip
            )
            vx_new *= sat_factor
            vy_new *= sat_factor

        # Position update with trapezoidal velocity (2nd order accurate)
        # p_new = p + 0.5 * (v_old + v_new) * dt = p + v_old * dt + 0.5 * a * dt²
        px = px + vx * dt_imu + 0.5 * ax_world * dt_imu**2
        py = py + vy * dt_imu + 0.5 * ay_world * dt_imu**2
        theta = wrap_angle(theta + yaw_rate * dt_imu)

        # Update velocity
        vx = vx_new
        vy = vy_new

        # --- 5) Wall reflections (inelastic) ---
        if px < 0.0:
            px = -px
            vx = -0.5 * vx
        elif px > config.arena_w:
            px = 2 * config.arena_w - px
            vx = -0.5 * vx

        if py < 0.0:
            py = -py
            vy = -0.5 * vy
        elif py > config.arena_h:
            py = 2 * config.arena_h - py
            vy = -0.5 * vy

        # Save truth
        x = np.array([px, py, vx, vy, theta])
        X_truth[t] = x

        # --- 6) Bias random walks ---
        b_gyro = rw_step(b_gyro, config.gyro_bias_rw_density, dt_imu, rng)
        b_accel_x = rw_step(b_accel_x, config.accel_bias_rw_density, dt_imu, rng)
        b_accel_y = rw_step(b_accel_y, config.accel_bias_rw_density, dt_imu, rng)

        bias_gyro[t] = b_gyro
        bias_accel_x[t] = b_accel_x
        bias_accel_y[t] = b_accel_y

        # --- 7) IMU measurements (body frame) ---
        # Transform world accelerations to body frame (using current theta)
        ax_body = c * ax_world + s * ay_world
        ay_body = -s * ax_world + c * ay_world

        # Store ground truth inertial acceleration for validation
        yaw_rate_truth[t] = yaw_rate
        accel_world_truth[t] = [ax_world, ay_world]
        accel_body_truth[t] = [ax_body, ay_body]

        # Accelerometer measures SPECIFIC FORCE = a_body - g_body
        # Gravity in tilted IMU frame rotated by current yaw
        g_body_x = c * g_imu_x + s * g_imu_y
        g_body_y = -s * g_imu_x + c * g_imu_y

        specific_force_x = ax_body - g_body_x
        specific_force_y = ay_body - g_body_y

        # Stash the noiseless specific force so diagnostic overlays can
        # compare measured U_imu (noisy specific force) against the
        # quantity it actually represents. ``accel_body_truth`` is the
        # *inertial* acceleration and differs by the rotated gravity
        # term — using it as an overlay misreads the IMU panel as
        # showing a tilt-induced bias.
        specific_force_truth[t] = [specific_force_x, specific_force_y]

        # Add bias and white noise to specific force
        gyro_meas = yaw_rate + b_gyro + std_gyro * rng.standard_normal()
        accel_x_meas = specific_force_x + b_accel_x + std_accel * rng.standard_normal()
        accel_y_meas = specific_force_y + b_accel_y + std_accel * rng.standard_normal()

        U_imu[t] = np.array([gyro_meas, accel_x_meas, accel_y_meas])

    # --- 8) Camera observations ---
    # Apply timestamp jitter and latency.
    #   Exposure time: when light hits sensor (jittered from nominal)
    #   Arrival time:  when data is available (exposure + latency)
    # We then:
    #   1) Clip `t_cam_exp` to the IMU support range so np.interp does not
    #      extrapolate.
    #   2) Compute `t_cam_obs` from the *clipped* exposure so the documented
    #      contract `t_cam_obs[i] - t_cam_exp[i] == cam_latency_s` holds at
    #      the clipping boundaries.
    #   3) Sort by exposure time so downstream code that assumes strictly
    #      increasing camera timestamps holds. The EKF
    #      builds (t_cam[i-1], t_cam[i]] IMU intervals via
    #      compute_imu_index_arrays; a non-monotonic step empties an
    #      interval, while a tied step is rejected by validate_timestamps.
    #      np.searchsorted in the diagnostic-video loader also requires a
    #      sorted t_cam_exp. We sort jittered timestamps (rather than
    #      forbidding cross-frame jitter outright) and bring the matching
    #      `t_cam_obs` along; this preserves jitter realism without
    #      reordering jittered samples relative to anything that depends
    #      on the original frame index later, because all camera-indexed
    #      arrays (LED positions, masks, confidence) are derived from the
    #      sorted `t_cam_exp` below.
    jitter = config.cam_jitter_s * rng.standard_normal(T_cam)
    t_cam_exp = np.clip(t_cam_clean + jitter, t_imu[0], t_imu[-1])
    cam_sort_order = np.argsort(t_cam_exp, kind="stable")
    t_cam_exp = t_cam_exp[cam_sort_order]
    positive_time_steps = [
        np.min(np.diff(t_imu)),
        np.min(np.diff(t_cam_clean)) if T_cam > 1 else np.min(np.diff(t_imu)),
    ]
    min_timestamp_step = min(positive_time_steps) * 1e-6
    t_cam_exp = _make_strictly_increasing_within_bounds(
        t_cam_exp,
        lower=float(t_imu[0]),
        upper=float(t_imu[-1]),
        min_step=float(min_timestamp_step),
    )
    t_cam_obs = t_cam_exp + config.cam_latency_s

    # Interpolate truth at EXPOSURE time (what pixels actually measure)
    px_interp = np.interp(t_cam_exp, t_imu, X_truth[:, 0])
    py_interp = np.interp(t_cam_exp, t_imu, X_truth[:, 1])
    theta_interp = interp_angle(t_cam_exp, t_imu, X_truth[:, 4])

    # Generate LED positions
    def led_position(
        px: np.ndarray, py: np.ndarray, theta: np.ndarray, offset_body: np.ndarray
    ) -> np.ndarray:
        """Transform LED from body frame to world frame."""
        c, s = np.cos(theta), np.sin(theta)
        led_x = px + c * offset_body[0] - s * offset_body[1]
        led_y = py + s * offset_body[0] + c * offset_body[1]
        return np.stack([led_x, led_y], axis=1)

    led1_truth = led_position(
        px_interp, py_interp, theta_interp, config.led1_offset_body
    )

    if config.use_second_led:
        led2_truth = led_position(
            px_interp, py_interp, theta_interp, config.led2_offset_body
        )
    else:
        led2_truth = np.full((T_cam, 2), np.nan)

    # Generate correlated dropouts for LED1 and LED2
    # Use copula approach: generate correlated uniform random variables
    if config.use_second_led:
        # Generate correlated uniform random variables using Gaussian copula
        # rho = correlation coefficient
        rho = config.cam_dropout_correlation

        # Generate bivariate normal with correlation rho
        z1 = rng.standard_normal(T_cam)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(T_cam)

        # Transform to uniform via CDF
        from scipy.stats import norm

        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)

        # Apply threshold
        mask_led1 = u1 > config.cam_dropout_prob
        mask_led2 = u2 > config.cam_dropout_prob
    else:
        # Single LED case: only LED1 mask matters
        mask_led1 = rng.random(T_cam) > config.cam_dropout_prob
        mask_led2 = np.zeros(T_cam, dtype=bool)  # LED2 always dropped

    # Legacy mask_cam for backward compatibility (union of both LEDs)
    mask_cam = mask_led1 | mask_led2

    # Generate confidence scores (optional)
    if config.use_confidence:
        # Base confidence with random variation
        confidence_led1 = config.confidence_base * (0.8 + 0.4 * rng.random(T_cam))
        confidence_led2 = (
            config.confidence_base * (0.8 + 0.4 * rng.random(T_cam))
            if config.use_second_led
            else np.zeros(T_cam)
        )

        # Reduce confidence near dropouts (vectorized)
        # Use convolution to detect neighboring dropouts: [0.5, 1.0, 0.5] kernel
        # If any neighbor is a dropout, the convolution will be non-zero

        # LED1: Zero out dropouts, then apply neighbor decay.
        # Operator precedence trap: ``~mask_led1.astype(int)`` parses as
        # ``~(mask_led1.astype(int))`` and bitwise-negates the int values
        # (True→1→-2, False→0→-1) instead of producing a 0/1 dropout
        # indicator. The convolution then yields strictly-negative
        # values, so ``> 0`` was never satisfied and confidence_dropout_
        # decay was silently a no-op for adjacent visible frames. Force
        # boolean negation first, then cast.
        confidence_led1 = np.where(mask_led1, confidence_led1, 0.0)
        neighbor_dropout_led1 = np.convolve(
            (~mask_led1).astype(int), [0.5, 1.0, 0.5], mode="same"
        )
        # Decay confidence where neighbors are dropouts (but current is valid)
        confidence_led1 *= np.where(
            mask_led1 & (neighbor_dropout_led1 > 0),
            config.confidence_dropout_decay,
            1.0,
        )

        # LED2: Same logic (if enabled)
        if config.use_second_led:
            confidence_led2 = np.where(mask_led2, confidence_led2, 0.0)
            neighbor_dropout_led2 = np.convolve(
                (~mask_led2).astype(int), [0.5, 1.0, 0.5], mode="same"
            )
            confidence_led2 *= np.where(
                mask_led2 & (neighbor_dropout_led2 > 0),
                config.confidence_dropout_decay,
                1.0,
            )

        # Clip confidence to valid range [0, 1] to prevent numerical issues
        np.clip(confidence_led1, 0.0, 1.0, out=confidence_led1)
        np.clip(confidence_led2, 0.0, 1.0, out=confidence_led2)
    else:
        confidence_led1 = np.ones(T_cam)
        confidence_led2 = np.ones(T_cam) if config.use_second_led else np.zeros(T_cam)

    # Add measurement noise (scaled by confidence if enabled)
    if config.use_confidence:
        noise_scale_led1 = confidence_to_noise_scale(
            confidence_led1, config.cam_sigma_m, epsilon=0.01
        )
        if config.use_second_led:
            noise_scale_led2 = confidence_to_noise_scale(
                confidence_led2, config.cam_sigma_m, epsilon=0.01
            )
        else:
            noise_scale_led2 = np.zeros(T_cam)
    else:
        # Uniform noise (broadcast scalar to array for consistent indexing)
        noise_scale_led1 = np.full(T_cam, config.cam_sigma_m)
        noise_scale_led2 = (
            np.full(T_cam, config.cam_sigma_m)
            if config.use_second_led
            else np.zeros(T_cam)
        )

    noise_led1 = noise_scale_led1[:, None] * rng.standard_normal((T_cam, 2))
    Z_cam_led1 = led1_truth + noise_led1

    if config.use_second_led:
        noise_led2 = noise_scale_led2[:, None] * rng.standard_normal((T_cam, 2))
        Z_cam_led2 = led2_truth + noise_led2
    else:
        Z_cam_led2 = np.full((T_cam, 2), np.nan)

    # Apply independent dropouts to each LED
    Z_cam_led1[~mask_led1] = np.nan
    Z_cam_led2[~mask_led2] = np.nan

    # Simulate LED wall reflections (artifacts near arena boundaries)
    # Physical basis: LED light reflects off arena walls (e.g., black plexiglass)
    # creating spurious detections at mirrored positions. Common in SpikeGadgets
    # setups with reflective arena materials. Detection probability increases near
    # walls (within led_wall_reflection_distance).
    led_reflection_applied = np.zeros(T_cam, dtype=bool)

    if config.led_wall_reflection_prob > 0 and config.led_wall_reflection_distance > 0:
        # Compute distance from rat center to nearest wall at each camera frame
        dist_to_left = px_interp
        dist_to_right = config.arena_w - px_interp
        dist_to_bottom = py_interp
        dist_to_top = config.arena_h - py_interp
        dist_to_nearest_wall = np.minimum(
            np.minimum(dist_to_left, dist_to_right),
            np.minimum(dist_to_bottom, dist_to_top),
        )

        # Candidates: near wall and at least one LED visible
        near_wall = dist_to_nearest_wall <= config.led_wall_reflection_distance
        reflection_candidates = near_wall & (mask_led1 | mask_led2)
        candidate_indices = np.where(reflection_candidates)[0]

        if len(candidate_indices) > 0:
            # Per-frame Bernoulli sampling. Using
            # ``round(n_candidates * p)`` produced a deterministic fixed
            # count: small ``p × n`` rounded to 0 disabled the effect
            # entirely on short/sparse runs (e.g. 0.25 × 2 → 0 reflections
            # for every seed), and the exact-count selection eliminated
            # the binomial variance the docstring promises ("per visible
            # frame"). True per-frame probability via ``rng.random < p``.
            draws = rng.random(len(candidate_indices))
            reflection_indices = candidate_indices[
                draws < config.led_wall_reflection_prob
            ]
            if len(reflection_indices) > 0:
                led_reflection_applied[reflection_indices] = True

                # Apply reflection for each selected frame
                for idx in reflection_indices:
                    # Determine which wall is closest (using precomputed distances)
                    dists = [
                        (dist_to_left[idx], "left"),
                        (dist_to_right[idx], "right"),
                        (dist_to_bottom[idx], "bottom"),
                        (dist_to_top[idx], "top"),
                    ]
                    _, closest_wall = min(dists)

                    # Mirror LEDs across the closest wall
                    # Reflection formula: x' = 2*wall_pos - x
                    if closest_wall == "left":
                        # Mirror across x = 0: x → 2*0 - x = -x
                        if mask_led1[idx]:
                            Z_cam_led1[idx, 0] = -Z_cam_led1[idx, 0]
                        if mask_led2[idx]:
                            Z_cam_led2[idx, 0] = -Z_cam_led2[idx, 0]
                    elif closest_wall == "right":
                        # Mirror across x = arena_w: x → 2*arena_w - x
                        if mask_led1[idx]:
                            Z_cam_led1[idx, 0] = 2 * config.arena_w - Z_cam_led1[idx, 0]
                        if mask_led2[idx]:
                            Z_cam_led2[idx, 0] = 2 * config.arena_w - Z_cam_led2[idx, 0]
                    elif closest_wall == "bottom":
                        # Mirror across y = 0: y → 2*0 - y = -y
                        if mask_led1[idx]:
                            Z_cam_led1[idx, 1] = -Z_cam_led1[idx, 1]
                        if mask_led2[idx]:
                            Z_cam_led2[idx, 1] = -Z_cam_led2[idx, 1]
                    elif closest_wall == "top":
                        # Mirror across y = arena_h: y → 2*arena_h - y
                        if mask_led1[idx]:
                            Z_cam_led1[idx, 1] = 2 * config.arena_h - Z_cam_led1[idx, 1]
                        if mask_led2[idx]:
                            Z_cam_led2[idx, 1] = 2 * config.arena_h - Z_cam_led2[idx, 1]

    # Simulate LED swaps (mislabeling front/back during close passes, reflections, etc.)
    # Only swap when both LEDs are visible (otherwise swap doesn't make sense)
    swap_applied = np.zeros(T_cam, dtype=bool)  # Track which frames had swaps

    if config.use_second_led:
        both_visible = mask_led1 & mask_led2

        if config.led_swap_mode == "per_frame":
            # Per-frame Bernoulli swap (matches the documented
            # "each frame independently with probability led_swap_prob").
            # ``round(n_candidates * p)`` produced a deterministic fixed
            # count and stripped the binomial variance — small ``p × n``
            # rounded to 0 silently disabled swaps on short/sparse runs.
            if config.led_swap_prob > 0:
                swap_candidates = np.where(both_visible)[0]
                if len(swap_candidates) > 0:
                    draws = rng.random(len(swap_candidates))
                    swap_indices = swap_candidates[draws < config.led_swap_prob]
                    if len(swap_indices) > 0:
                        swap_applied[swap_indices] = True

        elif config.led_swap_mode == "persistent":
            # Persistent swaps: event-based swaps that last for a duration
            if config.led_swap_rate > 0:
                # Generate swap events using a Poisson process
                # Expected number of swap events in the session
                expected_n_events = config.led_swap_rate * config.duration_s
                n_events = rng.poisson(expected_n_events)

                if n_events > 0:
                    # Generate event start times (uniformly distributed across session)
                    event_start_times = rng.uniform(0, config.duration_s, size=n_events)

                    # Generate event durations (Gaussian with mean and std)
                    event_durations = rng.normal(
                        loc=config.led_swap_duration_mean,
                        scale=config.led_swap_duration_std,
                        size=n_events,
                    )
                    # Clip durations to be positive
                    event_durations = np.maximum(event_durations, dt_cam)

                    # Mark frames as swapped for each event
                    for start_time, duration in zip(
                        event_start_times, event_durations, strict=False
                    ):
                        end_time = start_time + duration

                        # Find camera frames within this swap event
                        # Only swap frames where both LEDs are visible
                        in_event = (t_cam_exp >= start_time) & (t_cam_exp < end_time)
                        swap_frames = np.where(in_event & both_visible)[0]

                        if len(swap_frames) > 0:
                            swap_applied[swap_frames] = True

        # Apply swaps to LED positions and confidences
        if np.any(swap_applied):
            swap_indices = np.where(swap_applied)[0]
            # Swap LED positions and confidences at selected frames
            Z_cam_led1[swap_indices], Z_cam_led2[swap_indices] = (
                Z_cam_led2[swap_indices].copy(),
                Z_cam_led1[swap_indices].copy(),
            )
            if config.use_confidence:
                confidence_led1[swap_indices], confidence_led2[swap_indices] = (
                    confidence_led2[swap_indices].copy(),
                    confidence_led1[swap_indices].copy(),
                )

    return {
        # Time
        "t_imu": t_imu,
        "t_cam_exp": t_cam_exp,  # Exposure time (use for measurement timestamps)
        "t_cam_obs": t_cam_obs,  # Observation arrival time (exposure + latency)
        # Truth
        "X_truth": X_truth,
        "yaw_rate_truth": yaw_rate_truth,
        "accel_world_truth": accel_world_truth,
        "accel_body_truth": accel_body_truth,
        "specific_force_truth": specific_force_truth,
        # IMU
        "U_imu": U_imu,
        "bias_gyro": bias_gyro,
        "bias_accel_x": bias_accel_x,
        "bias_accel_y": bias_accel_y,
        # Camera
        "Z_cam_led1": Z_cam_led1,
        "Z_cam_led2": Z_cam_led2,
        "led1_truth_cam": led1_truth,  # Ground truth LED1 positions (before swaps/noise)
        "led2_truth_cam": led2_truth,  # Ground truth LED2 positions (before swaps/noise)
        "swap_applied": swap_applied,  # Boolean mask: True where LED labels were swapped
        "led_reflection_applied": led_reflection_applied,  # Boolean mask: True where wall reflections were applied
        "confidence_led1": confidence_led1,
        "confidence_led2": confidence_led2,
        "mask_cam": mask_cam,  # Union mask (backward compatibility)
        "mask_led1": mask_led1,  # Individual LED1 mask
        "mask_led2": mask_led2,  # Individual LED2 mask
        # Config
        "config": config,
    }


# -----------------------------------------------------------------------------
# Convenience API
# -----------------------------------------------------------------------------


def make_default_config(**kwargs) -> RatIMUSimConfig:
    """
    Create a RatIMUSimConfig with optional overrides.

    Defaults match SpikeGadgets headstage hardware specifications (104 Hz IMU, realistic noise).

    Parameters
    ----------
    **kwargs
        Fields to override in default config.

    Returns
    -------
    RatIMUSimConfig
        New configuration instance.

    Example:
        >>> config = make_default_config(duration_s=120.0, use_second_led=True)
    """
    # Construct directly with the overrides so RatIMUSimConfig.__post_init__
    # runs against the final field values. Using post-construction setattr
    # would silently bypass every validation block (verified: the previous
    # implementation accepted duration_s=-1.0, fs_imu=0.0, and
    # sigma_yaw_rate=NaN, all of which now raise at construction).
    defaults = RatIMUSimConfig()
    valid_field_names = {f.name for f in dataclass_fields(defaults)}
    unknown = sorted(set(kwargs) - valid_field_names)
    if unknown:
        raise ValueError(f"Unknown config parameter(s): {unknown!r}")
    base = {f.name: getattr(defaults, f.name) for f in dataclass_fields(defaults)}
    base.update(kwargs)
    return RatIMUSimConfig(**base)
