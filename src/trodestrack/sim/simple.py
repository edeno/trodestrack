"""Simple analytic simulations for testing and validation.

Provides three minimal simulation scenarios with closed-form solutions:
1. Stationary: No motion, only sensor noise
2. Constant velocity: Straight line motion
3. Circular: Constant angular velocity motion

These simulations are designed for unit testing, debugging, and verifying
filter correctness with known analytic ground truth.

Usage:
    from trodestrack.sim.simple import simulate_stationary

    sim = simulate_stationary(duration_s=10.0, position=[0.5, 0.5])
    # Returns same dict structure as simulate_rat_imu()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import SimOut, density_to_sample_std


@dataclass
class SimpleSimConfig:
    """Configuration for simple simulations.

    Attributes:
        duration_s: Simulation duration in seconds
        fs_imu: IMU sampling rate (Hz)
        fs_cam: Camera sampling rate (Hz)
        gyro_noise_density: Gyroscope noise density (rad/s/√Hz)
        accel_noise_density: Accelerometer noise density (m/s²/√Hz)
        gyro_bias_std: Initial gyroscope bias std (rad/s)
        accel_bias_std: Initial accelerometer bias std (m/s²)
        cam_noise_std: Camera position noise std (m)
        cam_dropout_prob: Probability of camera dropout per frame
        gravity: Gravitational acceleration (m/s²)
    """

    duration_s: float = 10.0
    fs_imu: float = 104.0  # SpikeGadgets hardware sensor refresh rate
    fs_cam: float = 30.0

    # Sensor noise (conservative values for testing)
    gyro_noise_density: float = 0.0001  # rad/s/√Hz (very low)
    accel_noise_density: float = 0.03  # m/s²/√Hz (very low)
    gyro_bias_std: float = 0.001  # rad/s (small constant bias)
    accel_bias_std: float = 0.01  # m/s² (small constant bias)
    cam_noise_std: float = 0.002  # m (2 mm)
    cam_dropout_prob: float = 0.05  # 5% dropout

    gravity: float = 9.80665  # m/s²

    def __post_init__(self):
        """Validate configuration parameters."""
        # Reject non-finite scalar parameters first. Bare ``<= 0`` / ``< 0``
        # comparisons silently accept NaN (NaN compares False to both),
        # and the simulator then emits non-finite IMU / truth arrays —
        # e.g. gyro_noise_density=NaN produced non-finite U_imu.
        finite_scalar_fields = (
            "duration_s",
            "fs_imu",
            "fs_cam",
            "gyro_noise_density",
            "accel_noise_density",
            "gyro_bias_std",
            "accel_bias_std",
            "cam_noise_std",
            "cam_dropout_prob",
            "gravity",
        )
        for fname in finite_scalar_fields:
            value = getattr(self, fname)
            if not np.isfinite(value):
                raise ValueError(f"{fname} must be a finite value; got {value!r}.")

        # Duration validation
        if self.duration_s <= 0:
            raise ValueError(
                f"Simulation duration must be positive, got {self.duration_s}s.\n"
                f"Example: duration_s=10.0 (10 seconds)"
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

        # Minimum-sample-count gate (mirrors RatIMUSimConfig). The simple
        # simulators compute counts as ``int(duration_s * fs_*)``; very
        # short positive durations therefore produce zero or one sample
        # per stream and crash downstream (e.g. ``prepare_video_data``
        # nearest-neighbor indexing raises IndexError when the camera
        # stream is empty). Require at least 2 IMU and 2 camera samples.
        T_imu = int(self.duration_s * self.fs_imu)
        T_cam = int(self.duration_s * self.fs_cam)
        if T_imu < 2 or T_cam < 2:
            raise ValueError(
                f"duration_s={self.duration_s}s at fs_imu={self.fs_imu} Hz, "
                f"fs_cam={self.fs_cam} Hz produces only T_imu={T_imu}, "
                f"T_cam={T_cam} samples; need at least 2 of each. "
                f"Increase duration_s or sampling rates."
            )

        # Probability validation
        if not 0 <= self.cam_dropout_prob <= 1:
            raise ValueError(
                f"Dropout probability must be in [0, 1], got {self.cam_dropout_prob}.\n"
                f"Example: cam_dropout_prob=0.05 (5% dropout rate)"
            )

        # Physical parameter validation
        if self.gravity <= 0:
            raise ValueError(
                f"Gravity must be positive, got {self.gravity} m/s².\n"
                f"Standard Earth gravity: gravity=9.80665"
            )

        # Noise parameter validation
        if self.cam_noise_std < 0:
            raise ValueError(
                f"Camera noise must be non-negative, got {self.cam_noise_std}m.\n"
                f"Example: cam_noise_std=0.002 (2mm std noise)"
            )

        if self.gyro_noise_density < 0 or self.accel_noise_density < 0:
            raise ValueError(
                f"IMU noise densities must be non-negative.\n"
                f"Got gyro={self.gyro_noise_density}, accel={self.accel_noise_density}"
            )

        if self.gyro_bias_std < 0 or self.accel_bias_std < 0:
            raise ValueError(
                f"IMU bias stds must be non-negative.\n"
                f"Got gyro={self.gyro_bias_std}, accel={self.accel_bias_std}"
            )


def _validate_xy_array(value: np.ndarray, name: str) -> np.ndarray:
    """Coerce ``value`` into a finite ``(2,)`` float array.

    Used by the simple-scenario sims for ``position``, ``initial_position``,
    ``velocity``, and ``center``. Without this guard, a malformed
    ``(1,)`` argument raises a raw IndexError mid-simulation, and a
    NaN argument silently produces non-finite truth/IMU/camera arrays.
    """
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values; got {arr!r}.")
    return arr


def _validate_finite_scalar(value: float, name: str) -> float:
    """Reject non-finite scalar arguments to the simple-sim functions."""

    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite value; got {value!r}.")
    return float(value)


def simulate_stationary(
    config: SimpleSimConfig | None = None,
    position: np.ndarray | None = None,
    heading: float = 0.0,
    seed: int = 0,
) -> SimOut:
    """Simulate stationary rat with no motion.

    Ground truth: constant position, zero velocity, constant heading.
    IMU measures specific force f = a − g in body frame; for a level-mounted
    IMU without motion, f_x ≈ f_y ≈ 0 (gravity is along Z). The camera measures
    fixed position with Gaussian noise and optional dropouts.

    Parameters
    ----------
    config : SimpleSimConfig, optional
        Simulation configuration (defaults used if None).
    position : np.ndarray | None, optional
        Initial position [x, y] (m). Default [0.5, 0.5].
    heading : float, default 0.0
        Heading angle (rad).
    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    SimOut
        Simulation output dict; keys include `t_imu` (N_imu,), `t_cam_exp` (N_cam,),
        `X_truth` (N_imu, 5) [x, y, vx, vy, θ], `U_imu` (N_imu, 3) [ω_z, a_x, a_y],
        `Z_cam_led1` (N_cam, 2), `mask_cam` (N_cam,), biases, and metadata.

    Notes
    -----
    Useful for testing measurement updates, steady-state covariance, and bias
    estimation in the absence of motion.
    """
    if config is None:
        config = SimpleSimConfig()
    if position is None:
        position = np.array([0.5, 0.5])
    position = _validate_xy_array(position, "position")
    heading = _validate_finite_scalar(heading, "heading")

    rng = np.random.default_rng(seed)

    # Time vectors
    dt_imu = 1.0 / config.fs_imu
    dt_cam = 1.0 / config.fs_cam
    T_imu = int(config.duration_s * config.fs_imu)
    T_cam = int(config.duration_s * config.fs_cam)

    t_imu = np.arange(T_imu) * dt_imu
    t_cam_exp = np.arange(T_cam) * dt_cam
    t_cam_obs = t_cam_exp  # No latency for simplicity

    # Ground truth state (constant)
    X_truth = np.zeros((T_imu, 5))
    X_truth[:, 0] = position[0]  # x
    X_truth[:, 1] = position[1]  # y
    X_truth[:, 2] = 0.0  # vx
    X_truth[:, 3] = 0.0  # vy
    X_truth[:, 4] = heading  # θ

    # Generate constant biases (broadcast to arrays for API consistency)
    bias_gyro_scalar = rng.normal(0.0, config.gyro_bias_std)
    bias_accel_x_scalar = rng.normal(0.0, config.accel_bias_std)
    bias_accel_y_scalar = rng.normal(0.0, config.accel_bias_std)

    # Broadcast to arrays for consistency with rat_imu.py
    bias_gyro = np.full(T_imu, bias_gyro_scalar)
    bias_accel_x = np.full(T_imu, bias_accel_x_scalar)
    bias_accel_y = np.full(T_imu, bias_accel_y_scalar)

    # IMU measurements
    # Gyro: no rotation, just noise + bias
    std_gyro = density_to_sample_std(config.gyro_noise_density, dt_imu)
    gyro_noise = rng.normal(0.0, std_gyro, T_imu)
    gyro = bias_gyro + gyro_noise

    # Accelerometer: measures specific force in body frame
    # For level-mounted IMU (no tilt), gravity is along Z-axis
    # X/Y specific force = 0 (no motion, no tilt) + bias + noise
    std_accel = density_to_sample_std(config.accel_noise_density, dt_imu)
    accel_x_noise = rng.normal(0.0, std_accel, T_imu)
    accel_y_noise = rng.normal(0.0, std_accel, T_imu)

    accel_x = bias_accel_x + accel_x_noise
    accel_y = bias_accel_y + accel_y_noise

    U_imu = np.column_stack([gyro, accel_x, accel_y])

    # Camera measurements (with dropout)
    mask_cam = rng.random(T_cam) > config.cam_dropout_prob

    cam_noise_x = rng.normal(0.0, config.cam_noise_std, T_cam)
    cam_noise_y = rng.normal(0.0, config.cam_noise_std, T_cam)

    Z_cam_led1 = np.column_stack(
        [
            position[0] + cam_noise_x,
            position[1] + cam_noise_y,
        ]
    )
    # Match rat_imu's convention: dropped frames carry NaN LED coordinates
    # so diagnostics that infer validity from finite values agree with
    # mask_cam. Filters that honor mask_cam are unaffected.
    Z_cam_led1[~mask_cam] = np.nan

    # Single LED mask (for consistency, LED1 only)
    mask_led1 = mask_cam.copy()
    mask_led2 = np.zeros(T_cam, dtype=bool)  # LED2 always dropped

    # Ground truth LED positions (before noise/swaps/reflections)
    # Simple sims don't apply swaps or reflections
    led1_truth_cam = np.column_stack(
        [position[0] * np.ones(T_cam), position[1] * np.ones(T_cam)]
    )
    led2_truth_cam = np.full((T_cam, 2), np.nan)  # No LED2 in stationary

    # Artifact tracking (simple sims don't apply swaps/reflections)
    swap_applied = np.zeros(T_cam, dtype=bool)
    led_reflection_applied = np.zeros(T_cam, dtype=bool)

    return {
        "t_imu": t_imu,
        "t_cam_exp": t_cam_exp,
        "t_cam_obs": t_cam_obs,
        "X_truth": X_truth,
        "U_imu": U_imu,
        "Z_cam_led1": Z_cam_led1,
        "Z_cam_led2": np.full((T_cam, 2), np.nan),  # Use NaN (dropped) convention
        "mask_cam": mask_cam,
        "mask_led1": mask_led1,  # Same as mask_cam for single LED
        "mask_led2": mask_led2,  # Always False (no LED2)
        "confidence_led1": np.ones(T_cam),
        "confidence_led2": np.zeros(T_cam),
        "led1_truth_cam": led1_truth_cam,
        "led2_truth_cam": led2_truth_cam,
        "swap_applied": swap_applied,
        "led_reflection_applied": led_reflection_applied,
        "bias_gyro": bias_gyro,
        "bias_accel_x": bias_accel_x,
        "bias_accel_y": bias_accel_y,
        "yaw_rate_truth": np.zeros(T_imu),
        "accel_world_truth": np.zeros((T_imu, 2)),
        "accel_body_truth": np.zeros(
            (T_imu, 2)
        ),  # No motion, no tilt → specific force = 0
        # Level IMU + no motion → specific force = inertial accel = 0.
        "specific_force_truth": np.zeros((T_imu, 2)),
        "config": config,
    }


def simulate_constant_velocity(
    config: SimpleSimConfig | None = None,
    initial_position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    seed: int = 0,
) -> SimOut:
    """Simulate constant velocity motion (straight line).

    Ground truth: x(t) = x0 + v t with constant heading aligned to velocity.
    For a level-mounted IMU with constant velocity, specific force in the body
    plane is near zero (gravity along Z).

    Parameters
    ----------
    config : SimpleSimConfig, optional
        Simulation configuration (defaults used if None).
    initial_position : np.ndarray | None, optional
        Starting position [x, y] (m). Default [0.1, 0.1].
    velocity : np.ndarray | None, optional
        Constant velocity [vx, vy] (m/s). Default [0.2, 0.1].
    seed : int, default 0
        Random seed.

    Returns
    -------
    SimOut
        Simulation output dict (same structure as simulate_rat_imu()).

    Notes
    -----
    Useful for testing prediction, position integration, IMU-camera alignment,
    and covariance propagation in low-acceleration regimes.
    """
    if config is None:
        config = SimpleSimConfig()
    if initial_position is None:
        initial_position = np.array([0.1, 0.1])
    if velocity is None:
        velocity = np.array([0.2, 0.1])  # 0.224 m/s = 22.4 cm/s
    initial_position = _validate_xy_array(initial_position, "initial_position")
    velocity = _validate_xy_array(velocity, "velocity")

    rng = np.random.default_rng(seed)

    # Time vectors
    dt_imu = 1.0 / config.fs_imu
    dt_cam = 1.0 / config.fs_cam
    T_imu = int(config.duration_s * config.fs_imu)
    T_cam = int(config.duration_s * config.fs_cam)

    t_imu = np.arange(T_imu) * dt_imu
    t_cam_exp = np.arange(T_cam) * dt_cam
    t_cam_obs = t_cam_exp

    # Ground truth state (linear motion)
    vx, vy = velocity
    heading = np.arctan2(vy, vx)  # Constant heading aligned with velocity

    X_truth = np.zeros((T_imu, 5))
    X_truth[:, 0] = initial_position[0] + vx * t_imu  # x(t) = x0 + vx*t
    X_truth[:, 1] = initial_position[1] + vy * t_imu  # y(t) = y0 + vy*t
    X_truth[:, 2] = vx
    X_truth[:, 3] = vy
    X_truth[:, 4] = heading

    # Generate constant biases (broadcast to arrays for API consistency)
    bias_gyro_scalar = rng.normal(0.0, config.gyro_bias_std)
    bias_accel_x_scalar = rng.normal(0.0, config.accel_bias_std)
    bias_accel_y_scalar = rng.normal(0.0, config.accel_bias_std)

    bias_gyro = np.full(T_imu, bias_gyro_scalar)
    bias_accel_x = np.full(T_imu, bias_accel_x_scalar)
    bias_accel_y = np.full(T_imu, bias_accel_y_scalar)

    # IMU measurements
    # Gyro: no rotation (constant heading), just noise + bias
    std_gyro = density_to_sample_std(config.gyro_noise_density, dt_imu)
    gyro_noise = rng.normal(0.0, std_gyro, T_imu)
    gyro = bias_gyro_scalar + gyro_noise

    # Accelerometer: measures specific force in body frame
    # For level-mounted IMU (no tilt), constant velocity in inertial frame:
    # specific force = 0 (no acceleration, gravity is along Z) + bias + noise
    std_accel = density_to_sample_std(config.accel_noise_density, dt_imu)
    accel_x_noise = rng.normal(0.0, std_accel, T_imu)
    accel_y_noise = rng.normal(0.0, std_accel, T_imu)

    accel_x = bias_accel_x_scalar + accel_x_noise
    accel_y = bias_accel_y_scalar + accel_y_noise

    U_imu = np.column_stack([gyro, accel_x, accel_y])

    # Camera measurements (moving position with dropout)
    mask_cam = rng.random(T_cam) > config.cam_dropout_prob

    cam_noise_x = rng.normal(0.0, config.cam_noise_std, T_cam)
    cam_noise_y = rng.normal(0.0, config.cam_noise_std, T_cam)

    x_cam = initial_position[0] + vx * t_cam_exp
    y_cam = initial_position[1] + vy * t_cam_exp

    Z_cam_led1 = np.column_stack(
        [
            x_cam + cam_noise_x,
            y_cam + cam_noise_y,
        ]
    )
    # Match rat_imu's convention: dropped frames carry NaN LED coordinates
    # so diagnostics inferring validity from finite values agree with mask_cam.
    Z_cam_led1[~mask_cam] = np.nan

    # Single LED mask (for consistency, LED1 only)
    mask_led1 = mask_cam.copy()
    mask_led2 = np.zeros(T_cam, dtype=bool)

    # Ground truth LED positions (before noise/swaps/reflections)
    # Simple sims don't apply swaps or reflections
    led1_truth_cam = np.column_stack([x_cam, y_cam])
    led2_truth_cam = np.full((T_cam, 2), np.nan)  # No LED2 in constant velocity

    # Artifact tracking (simple sims don't apply swaps/reflections)
    swap_applied = np.zeros(T_cam, dtype=bool)
    led_reflection_applied = np.zeros(T_cam, dtype=bool)

    return {
        "t_imu": t_imu,
        "t_cam_exp": t_cam_exp,
        "t_cam_obs": t_cam_obs,
        "X_truth": X_truth,
        "U_imu": U_imu,
        "Z_cam_led1": Z_cam_led1,
        "Z_cam_led2": np.full((T_cam, 2), np.nan),  # Use NaN (dropped) convention
        "mask_cam": mask_cam,
        "mask_led1": mask_led1,
        "mask_led2": mask_led2,
        "confidence_led1": np.ones(T_cam),
        "confidence_led2": np.zeros(T_cam),
        "led1_truth_cam": led1_truth_cam,
        "led2_truth_cam": led2_truth_cam,
        "swap_applied": swap_applied,
        "led_reflection_applied": led_reflection_applied,
        "bias_gyro": bias_gyro,
        "bias_accel_x": bias_accel_x,
        "bias_accel_y": bias_accel_y,
        "yaw_rate_truth": np.zeros(T_imu),
        "accel_world_truth": np.zeros((T_imu, 2)),  # No acceleration in world frame
        "accel_body_truth": np.zeros(
            (T_imu, 2)
        ),  # Constant velocity, no tilt → specific force = 0
        # Level IMU + constant velocity → specific force = inertial accel = 0.
        "specific_force_truth": np.zeros((T_imu, 2)),
        "config": config,
    }


def simulate_circular(
    config: SimpleSimConfig | None = None,
    center: np.ndarray | None = None,
    radius: float = 0.3,
    angular_velocity: float = 1.0,
    seed: int = 0,
) -> SimOut:
    """Simulate circular motion with constant angular velocity.

    Ground truth follows x(t)=x_c+r cos(ωt), y(t)=y_c+r sin(ωt), heading θ(t)=ωt+π/2
    (tangent). IMU measures specific force f=a−g; with level mounting, f is the
    centripetal acceleration rotated to body frame (gravity along Z).

    Parameters
    ----------
    config : SimpleSimConfig, optional
        Simulation configuration.
    center : np.ndarray | None, optional
        Circle center [x, y] (m). Default [0.5, 0.5].
    radius : float, default 0.3
        Circle radius (m).
    angular_velocity : float, default 1.0
        Angular velocity ω (rad/s).
    seed : int, default 0
        Random seed.

    Returns
    -------
    SimOut
        Simulation output dict (same structure as simulate_rat_imu()).

    Notes
    -----
    Useful for testing gyro integration, centripetal acceleration handling,
    heading observability, and NIS/NEES behavior under curved motion.
    """
    if config is None:
        config = SimpleSimConfig()
    if center is None:
        center = np.array([0.5, 0.5])
    center = _validate_xy_array(center, "center")
    radius = _validate_finite_scalar(radius, "radius")
    angular_velocity = _validate_finite_scalar(angular_velocity, "angular_velocity")
    # Heading is computed as ``angle + π/2`` (tangent direction)
    # independent of the sign of ``radius``, while position and
    # velocity scale linearly with ``radius``. A negative radius
    # therefore breaks the documented "heading tangent to motion"
    # invariant — heading and velocity-direction differ by π. Zero
    # radius collapses position to the center and produces zero
    # velocity, which is degenerate for a "circular motion" sim.
    if radius <= 0:
        raise ValueError(
            f"radius must be strictly positive (heading is computed as "
            f"angle + π/2 and is sign-blind to radius); got {radius}."
        )

    rng = np.random.default_rng(seed)

    # Time vectors
    dt_imu = 1.0 / config.fs_imu
    dt_cam = 1.0 / config.fs_cam
    T_imu = int(config.duration_s * config.fs_imu)
    T_cam = int(config.duration_s * config.fs_cam)

    t_imu = np.arange(T_imu) * dt_imu
    t_cam_exp = np.arange(T_cam) * dt_cam
    t_cam_obs = t_cam_exp

    # Ground truth circular motion
    omega = angular_velocity
    angle = omega * t_imu  # Angular position on circle

    # Position (parametric circle)
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)

    # Velocity (tangent to circle)
    vx = -radius * omega * np.sin(angle)
    vy = radius * omega * np.cos(angle)

    # Heading (tangent direction = angle + π/2)
    heading = angle + np.pi / 2
    heading = np.arctan2(np.sin(heading), np.cos(heading))  # Wrap to [-π, π]

    X_truth = np.column_stack([x, y, vx, vy, heading])

    # Generate constant biases
    # Generate constant biases (broadcast to arrays for API consistency)
    bias_gyro_scalar = rng.normal(0.0, config.gyro_bias_std)
    bias_accel_x_scalar = rng.normal(0.0, config.accel_bias_std)
    bias_accel_y_scalar = rng.normal(0.0, config.accel_bias_std)

    # Broadcast to arrays for consistency with rat_imu.py
    bias_gyro = np.full(T_imu, bias_gyro_scalar)
    bias_accel_x = np.full(T_imu, bias_accel_x_scalar)
    bias_accel_y = np.full(T_imu, bias_accel_y_scalar)

    # IMU measurements
    # Gyro: constant angular velocity + noise + bias
    std_gyro = density_to_sample_std(config.gyro_noise_density, dt_imu)
    gyro_noise = rng.normal(0.0, std_gyro, T_imu)
    gyro = omega + bias_gyro + gyro_noise

    # Accelerometer: measures specific force (f = a - g) in body frame
    # For level-mounted IMU: gravity is along world Z, not in X/Y plane
    # World frame centripetal acceleration: a_world = -rω² * [cos(angle), sin(angle)]
    accel_world_x = -radius * omega**2 * np.cos(angle)
    accel_world_y = -radius * omega**2 * np.sin(angle)

    # Specific force in world frame: f_world = a_world (since g is in Z)
    # Rotate to body frame: R(θ).T @ f_world
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    accel_body_x = cos_h * accel_world_x + sin_h * accel_world_y
    accel_body_y = -sin_h * accel_world_x + cos_h * accel_world_y

    std_accel = density_to_sample_std(config.accel_noise_density, dt_imu)
    accel_x_noise = rng.normal(0.0, std_accel, T_imu)
    accel_y_noise = rng.normal(0.0, std_accel, T_imu)

    accel_x = accel_body_x + bias_accel_x + accel_x_noise
    accel_y = accel_body_y + bias_accel_y + accel_y_noise

    U_imu = np.column_stack([gyro, accel_x, accel_y])

    # Camera measurements (moving position with dropout)
    mask_cam = rng.random(T_cam) > config.cam_dropout_prob

    cam_noise_x = rng.normal(0.0, config.cam_noise_std, T_cam)
    cam_noise_y = rng.normal(0.0, config.cam_noise_std, T_cam)

    angle_cam = omega * t_cam_exp
    x_cam = center[0] + radius * np.cos(angle_cam)
    y_cam = center[1] + radius * np.sin(angle_cam)

    # Generate actual LED positions: LED1 = back, LED2 = front
    # Heading is tangent to circle (perpendicular to radius)
    heading_cam = angle_cam + np.pi / 2
    led_distance = 0.04  # 4 cm between LEDs
    dx = 0.5 * led_distance * np.cos(heading_cam)
    dy = 0.5 * led_distance * np.sin(heading_cam)

    # LED1 = back LED (center - offset along heading)
    Z_cam_led1 = np.column_stack(
        [
            x_cam - dx + cam_noise_x,
            y_cam - dy + cam_noise_y,
        ]
    )

    # LED2 = front LED (center + offset along heading)
    cam_noise_x2 = rng.normal(0.0, config.cam_noise_std, T_cam)
    cam_noise_y2 = rng.normal(0.0, config.cam_noise_std, T_cam)
    Z_cam_led2 = np.column_stack(
        [
            x_cam + dx + cam_noise_x2,
            y_cam + dy + cam_noise_y2,
        ]
    )

    # Match rat_imu's convention: dropped frames carry NaN LED coordinates
    # so diagnostics inferring validity from finite values agree with
    # mask_cam. Filters that honor mask_cam are unaffected.
    Z_cam_led1[~mask_cam] = np.nan
    Z_cam_led2[~mask_cam] = np.nan

    # Both LEDs available for heading observability
    mask_led1 = mask_cam.copy()
    mask_led2 = mask_cam.copy()

    # Ground truth LED positions (before noise/swaps/reflections)
    # Simple sims don't apply swaps or reflections
    led1_truth_cam = np.column_stack([x_cam - dx, y_cam - dy])  # Back LED
    led2_truth_cam = np.column_stack([x_cam + dx, y_cam + dy])  # Front LED

    # Artifact tracking (simple sims don't apply swaps/reflections)
    swap_applied = np.zeros(T_cam, dtype=bool)
    led_reflection_applied = np.zeros(T_cam, dtype=bool)

    return {
        "t_imu": t_imu,
        "t_cam_exp": t_cam_exp,
        "t_cam_obs": t_cam_obs,
        "X_truth": X_truth,
        "U_imu": U_imu,
        "Z_cam_led1": Z_cam_led1,
        "Z_cam_led2": Z_cam_led2,
        "mask_cam": mask_cam,
        "mask_led1": mask_led1,
        "mask_led2": mask_led2,
        "confidence_led1": np.ones(T_cam),
        "confidence_led2": np.ones(T_cam),  # Both LEDs have confidence in circular
        "led1_truth_cam": led1_truth_cam,
        "led2_truth_cam": led2_truth_cam,
        "swap_applied": swap_applied,
        "led_reflection_applied": led_reflection_applied,
        "bias_gyro": bias_gyro,
        "bias_accel_x": bias_accel_x,
        "bias_accel_y": bias_accel_y,
        "yaw_rate_truth": np.full(T_imu, omega),
        "accel_world_truth": np.column_stack([accel_world_x, accel_world_y]),
        "accel_body_truth": np.column_stack([accel_body_x, accel_body_y]),
        # Level IMU (no tilt in simple sims) → specific force equals
        # inertial acceleration in body frame.
        "specific_force_truth": np.column_stack([accel_body_x, accel_body_y]),
        "config": config,
    }
