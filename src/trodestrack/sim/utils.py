"""Shared utilities for simulation modules.

Common functions used across different simulation scenarios (simple, rat_imu, etc.)
to ensure consistency and avoid code duplication.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


# =============================================================================
# Type Definitions
# =============================================================================


class SimOut(TypedDict):
    """Standard output format for all simulation functions.

    This typed dictionary ensures consistent return structure across
    simulate_stationary, simulate_constant_velocity, simulate_circular,
    and simulate_rat_imu.

    Attributes:
        t_imu: IMU timestamps (N_imu,)
        t_cam_exp: Camera exposure timestamps (N_cam,) - use for measurements
        t_cam_obs: Camera observation/arrival timestamps (N_cam,)

        X_truth: Ground truth state [x, y, vx, vy, θ] (N_imu, 5)
        U_imu: IMU measurements [ω_z, f_x, f_y] (N_imu, 3)

        Z_cam_led1: LED1 camera observations (N_cam, 2)
        Z_cam_led2: LED2 camera observations (N_cam, 2)

        mask_cam: Camera validity mask, union of both LEDs (N_cam,)
        mask_led1: LED1 validity mask (N_cam,)
        mask_led2: LED2 validity mask (N_cam,)

        confidence_led1: LED1 confidence scores (N_cam,)
        confidence_led2: LED2 confidence scores (N_cam,)

        bias_gyro: Gyroscope bias time series (N_imu,)
        bias_accel_x: X-axis accelerometer bias time series (N_imu,)
        bias_accel_y: Y-axis accelerometer bias time series (N_imu,)

        yaw_rate_truth: Ground truth yaw rate (N_imu,)
        accel_world_truth: Ground truth inertial accel in world frame (N_imu, 2)
        accel_body_truth: Ground truth inertial accel in body frame (N_imu, 2)

        config: Configuration object used for this simulation
    """

    # Time
    t_imu: np.ndarray
    t_cam_exp: np.ndarray
    t_cam_obs: np.ndarray

    # Truth
    X_truth: np.ndarray
    yaw_rate_truth: np.ndarray
    accel_world_truth: np.ndarray
    accel_body_truth: np.ndarray

    # IMU measurements
    U_imu: np.ndarray
    bias_gyro: np.ndarray
    bias_accel_x: np.ndarray
    bias_accel_y: np.ndarray

    # Camera measurements
    Z_cam_led1: np.ndarray
    Z_cam_led2: np.ndarray
    confidence_led1: np.ndarray
    confidence_led2: np.ndarray
    mask_cam: np.ndarray
    mask_led1: np.ndarray
    mask_led2: np.ndarray

    # Ground truth LED positions (before noise/swaps/reflections)
    led1_truth_cam: np.ndarray
    led2_truth_cam: np.ndarray

    # Artifact tracking
    swap_applied: np.ndarray
    led_reflection_applied: np.ndarray

    # Metadata
    config: Any  # Union of all config types


# =============================================================================
# Angle Utilities
# =============================================================================


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle to (-π, π].

    Args:
        a: Angle(s) in radians

    Returns:
        Wrapped angle(s) in range (-π, π]

    Example:
        >>> wrap_angle(3.5 * np.pi)
        -1.5707963267948966  # ≈ -π/2
    """
    return (a + np.pi) % (2 * np.pi) - np.pi


def interp_angle(t_new: np.ndarray, t_old: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Interpolate wrapped angles using unwrap → interp → rewrap.

    Prevents jumps at ±π discontinuity by unwrapping before interpolation.

    Args:
        t_new: Query timestamps
        t_old: Sample timestamps
        angles: Angle values at t_old (wrapped to [-π, π])

    Returns:
        Interpolated angles at t_new (wrapped to [-π, π])

    Example:
        >>> t_old = np.array([0.0, 1.0, 2.0])
        >>> angles = np.array([3.0, -3.0, 3.0])  # Wraps around ±π
        >>> t_new = np.array([0.5, 1.5])
        >>> interp_angle(t_new, t_old, angles)
        array([3.14159265, -3.14159265])  # Smooth through ±π boundary
    """
    angles_unwrapped = np.unwrap(angles)
    angles_interp = np.interp(t_new, t_old, angles_unwrapped)
    return wrap_angle(angles_interp)  # type: ignore[return-value]


# =============================================================================
# Noise & Bias Utilities
# =============================================================================


def density_to_sample_std(noise_density: float, dt: float) -> float:
    """Convert white noise density (units / √Hz) to discrete-time sample std.

    For white noise with power spectral density S₀ (units² / Hz),
    the discrete-time variance is S₀ / Δt, so std = √(S₀ / Δt).

    Args:
        noise_density: Noise density in units / √Hz
        dt: Sampling period in seconds

    Returns:
        Per-sample standard deviation

    Example:
        >>> density_to_sample_std(0.01, 0.005)  # 0.01 rad/s/√Hz at 200 Hz
        0.14142135623730953  # 0.01 / √(0.005)
    """
    return noise_density / np.sqrt(dt)


def rw_step(
    bias: float | np.ndarray,
    rw_density: float,
    dt: float,
    rng: np.random.Generator,
) -> float | np.ndarray:
    """Random-walk increment for bias with density (units / √s).

    Implements discrete-time random walk:
        bias_{t+1} = bias_t + N(0, rw_density² × dt)

    Args:
        bias: Current bias value (scalar or array)
        rw_density: Random walk density in units / √s
        dt: Time step in seconds
        rng: NumPy random generator

    Returns:
        Updated bias value

    Example:
        >>> rng = np.random.default_rng(42)
        >>> bias = 0.0
        >>> for _ in range(100):
        ...     bias = rw_step(bias, 0.001, 0.01, rng)
        >>> abs(bias) < 0.01  # Typical bias drift after 1 second
        True
    """
    return bias + rw_density * np.sqrt(dt) * rng.standard_normal()


# =============================================================================
# Camera Utilities
# =============================================================================


def confidence_to_noise_scale(
    confidence: np.ndarray, base_std: float, epsilon: float = 0.01
) -> np.ndarray:
    """Map confidence scores to measurement noise scale.

    Uses inverse square root scaling to increase noise for low confidence:
        σ(c) = σ_base / √(ε + c)

    Args:
        confidence: Confidence scores in [0, 1]
        base_std: Base measurement noise std at full confidence
        epsilon: Small constant to prevent division by zero

    Returns:
        Noise scale factors (multiply by standard normal samples)

    Example:
        >>> conf = np.array([1.0, 0.5, 0.1])
        >>> scales = confidence_to_noise_scale(conf, base_std=0.01, epsilon=0.01)
        >>> scales  # Lower confidence → higher noise
        array([0.00990099, 0.01414214, 0.03015113])
    """
    return base_std / np.sqrt(epsilon + confidence)


# =============================================================================
# Ornstein-Uhlenbeck Process
# =============================================================================


def ou_step(
    x: float,
    mean: float,
    tau: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    """Ornstein-Uhlenbeck process step (mean-reverting stochastic process).

    Implements discrete-time OU process:
        dx = (mean - x) / tau × dt + sigma × √dt × N(0, 1)

    Args:
        x: Current value
        mean: Long-term mean (equilibrium value)
        tau: Time constant (relaxation time in seconds)
        sigma: Noise intensity (units / √s)
        dt: Time step in seconds
        rng: NumPy random generator

    Returns:
        Updated value

    Note:
        For tau → ∞, this reduces to Brownian motion.
        For tau → 0, this becomes white noise around mean.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> x = 0.0
        >>> trajectory = [x := ou_step(x, mean=1.0, tau=0.5, sigma=0.1, dt=0.01, rng=rng)
        ...               for _ in range(100)]
        >>> abs(np.mean(trajectory) - 1.0) < 0.5  # Converges toward mean
        True
    """
    return x + (mean - x) * (dt / tau) + sigma * np.sqrt(dt) * rng.standard_normal()
