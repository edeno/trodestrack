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
from typing import Optional

import numpy as np

from .utils import SimOut, density_to_sample_std


@dataclass
class SimpleSimConfig:
    """Configuration for simple simulations.

    Attributes:
        duration_s: Simulation duration in seconds
        fs_imu: IMU sampling rate (Hz)
        fs_cam: Camera sampling rate (Hz)

        # Sensor noise parameters
        gyro_noise_density: Gyroscope noise density (rad/s/√Hz)
        accel_noise_density: Accelerometer noise density (m/s²/√Hz)
        gyro_bias_std: Initial gyroscope bias std (rad/s)
        accel_bias_std: Initial accelerometer bias std (m/s²)
        cam_noise_std: Camera position noise std (m)
        cam_dropout_prob: Probability of camera dropout per frame

        # Physical parameters
        gravity: Gravitational acceleration (m/s²)
    """

    duration_s: float = 10.0
    fs_imu: float = 200.0
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


def simulate_stationary(
    config: Optional[SimpleSimConfig] = None,
    position: Optional[np.ndarray] = None,
    heading: float = 0.0,
    seed: int = 0,
) -> SimOut:
    """Simulate stationary rat with no motion.

    Ground truth: constant position, zero velocity, constant heading.
    IMU measures: specific force (f = a - g) in body frame + noise + bias.
        For level-mounted IMU with no motion: f_x = f_y ≈ 0 (gravity is along Z).
    Camera measures: fixed position + noise.

    Args:
        config: Simulation configuration
        position: Initial position [x, y] in meters (default: [0.5, 0.5])
        heading: Heading angle in radians (default: 0.0)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with same structure as simulate_rat_imu():
        - t_imu: IMU timestamps (N_imu,)
        - t_cam_exp: Camera exposure times (N_cam,)
        - t_cam_obs: Camera observation times (N_cam,)
        - X_truth: State [x, y, vx, vy, θ] (N_imu, 5)
        - U_imu: IMU [ω_z, a_x, a_y] (N_imu, 3)
        - Z_cam_led1: Camera observations [x, y] (N_cam, 2)
        - mask_cam: Valid detection mask (N_cam,)
        - bias_gyro: Constant gyro bias (scalar)
        - bias_accel_x: Constant accel X bias (scalar)
        - bias_accel_y: Constant accel Y bias (scalar)
        - config: Simulation config object

    Note:
        This simulation is ideal for testing:
        - Measurement update logic
        - Steady-state covariance
        - Bias estimation with no motion
    """
    if config is None:
        config = SimpleSimConfig()
    if position is None:
        position = np.array([0.5, 0.5])

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

    # Single LED mask (for consistency, LED1 only)
    mask_led1 = mask_cam.copy()
    mask_led2 = np.zeros(T_cam, dtype=bool)  # LED2 always dropped

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
        "bias_gyro": bias_gyro,
        "bias_accel_x": bias_accel_x,
        "bias_accel_y": bias_accel_y,
        "yaw_rate_truth": np.zeros(T_imu),
        "accel_world_truth": np.zeros((T_imu, 2)),
        "accel_body_truth": np.zeros((T_imu, 2)),  # No motion, no tilt → specific force = 0
        "config": config,
    }


def simulate_constant_velocity(
    config: Optional[SimpleSimConfig] = None,
    initial_position: Optional[np.ndarray] = None,
    velocity: Optional[np.ndarray] = None,
    seed: int = 0,
) -> SimOut:
    """Simulate constant velocity motion (straight line).

    Ground truth: linear trajectory, constant velocity, constant heading.
    IMU measures: specific force (f = a - g) in body frame + noise + bias.
        For level-mounted IMU with constant velocity: f_x = f_y ≈ 0 (no acceleration, gravity along Z).
    Camera measures: moving position + noise.

    Args:
        config: Simulation configuration
        initial_position: Starting position [x, y] in meters (default: [0.1, 0.1])
        velocity: Constant velocity [vx, vy] in m/s (default: [0.2, 0.1])
        seed: Random seed for reproducibility

    Returns:
        Dictionary with same structure as simulate_rat_imu()

    Note:
        This simulation is ideal for testing:
        - Prediction step (constant velocity model)
        - Position integration
        - Time alignment between IMU and camera
        - Covariance propagation without process noise
    """
    if config is None:
        config = SimpleSimConfig()
    if initial_position is None:
        initial_position = np.array([0.1, 0.1])
    if velocity is None:
        velocity = np.array([0.2, 0.1])  # 0.224 m/s = 22.4 cm/s

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

    # Single LED mask (for consistency, LED1 only)
    mask_led1 = mask_cam.copy()
    mask_led2 = np.zeros(T_cam, dtype=bool)

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
        "bias_gyro": bias_gyro,
        "bias_accel_x": bias_accel_x,
        "bias_accel_y": bias_accel_y,
        "yaw_rate_truth": np.zeros(T_imu),
        "accel_world_truth": np.zeros((T_imu, 2)),  # No acceleration in world frame
        "accel_body_truth": np.zeros((T_imu, 2)),  # Constant velocity, no tilt → specific force = 0
        "config": config,
    }


def simulate_circular(
    config: Optional[SimpleSimConfig] = None,
    center: Optional[np.ndarray] = None,
    radius: float = 0.3,
    angular_velocity: float = 1.0,
    seed: int = 0,
) -> SimOut:
    """Simulate circular motion with constant angular velocity.

    Ground truth: circular trajectory, tangential velocity, rotating heading.
    IMU measures: specific force (f = a - g) in body frame + noise + bias.
        For level-mounted IMU: f = centripetal acceleration rotated to body frame (gravity is along Z).
    Camera measures: moving position on circle + noise.

    Args:
        config: Simulation configuration
        center: Circle center [x, y] in meters (default: [0.5, 0.5])
        radius: Circle radius in meters (default: 0.3 m)
        angular_velocity: Angular velocity in rad/s (default: 1.0 rad/s = 57.3°/s)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with same structure as simulate_rat_imu()

    Note:
        This simulation is ideal for testing:
        - Gyroscope integration (heading changes)
        - Centripetal acceleration handling
        - Heading measurement from velocity
        - Nonlinear dynamics (circular motion)

        Kinematics:
            x(t) = x_c + r*cos(ωt)
            y(t) = y_c + r*sin(ωt)
            θ(t) = ωt + π/2  (tangent to circle)
            v = rω (tangential speed)
            a_c = v²/r = rω² (centripetal acceleration toward center)
    """
    if config is None:
        config = SimpleSimConfig()
    if center is None:
        center = np.array([0.5, 0.5])

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

    Z_cam_led1 = np.column_stack(
        [
            x_cam + cam_noise_x,
            y_cam + cam_noise_y,
        ]
    )

    # Single LED mask (for consistency, LED1 only)
    mask_led1 = mask_cam.copy()
    mask_led2 = np.zeros(T_cam, dtype=bool)

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
        "bias_gyro": bias_gyro,
        "bias_accel_x": bias_accel_x,
        "bias_accel_y": bias_accel_y,
        "yaw_rate_truth": np.full(T_imu, omega),
        "accel_world_truth": np.column_stack([accel_world_x, accel_world_y]),
        "accel_body_truth": np.column_stack([accel_body_x, accel_body_y]),
        "config": config,
    }
