"""Shared configuration, state containers, and helpers for Kalman filters.

This module provides common dataclasses, utilities, and math helpers used by
both EKF and UKF implementations. All public functions use NumPy-style
docstrings and include array shapes and physical units where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array, lax, tree_util
from jax.scipy.linalg import cho_factor, cho_solve

from trodestrack.models.state_layout import StateLayout, get_heading_index, get_layout


@dataclass(frozen=True)
class FilterCoreConfig:
    """Core filter configuration shared by EKF and UKF.

    Parameters
    ----------
    process_noise_pos : float
        Position random-walk spectral density (m^2/s^3) used in Q.
    process_noise_vel : float
        Velocity random-walk spectral density ((m/s)^2/s).
    process_noise_heading : float
        Heading random-walk spectral density (rad^2/s).
    process_noise_gyro_bias : float
        Gyro bias random-walk spectral density ((rad/s)^2/s).
    process_noise_accel_bias : float
        Accelerometer bias random-walk spectral density ((m/s^2)^2/s).

    measurement_noise_pos : float
        Per-dimension position measurement noise variance (m^2).
    measurement_noise_heading : float
        Heading measurement noise variance (rad^2).

    imu_gyro_noise_density : float
        IMU gyro noise density (rad/s/√Hz) used during blackout adjustments.
    imu_accel_noise_density : float
        IMU accel noise density (m/s^2/√Hz) used during blackout adjustments.

    damping_coeff : float
        Linear velocity damping coefficient (1/s) in dynamics model.
    led_distance : float | None
        Nominal LED spacing (m). If None, spacing is estimated from data.

    use_mahalanobis_gating : bool
        Enable χ²-based outlier rejection on measurement updates.
    mahalanobis_threshold_prob : float
        Probability mass for χ² threshold (e.g., 0.997 ≈ 3σ).

    use_heading_measurement : bool
        Enable heading pseudo-measurement from LED geometry.
    led_distance_tolerance : float
        Relative tolerance for observed LED spacing vs expected (fraction).
    adaptive_heading_noise : bool
        If True, scales heading R by (expected/observed spacing)^2.

    adaptive_q_during_dropout : bool
        If True, increase position/velocity Q during vision dropouts.
    dropout_q_pos_multiplier : float
        Multiplier on position Q during dropout.
    dropout_q_vel_multiplier : float
        Multiplier on velocity Q during dropout.
    dropout_q_bias_multiplier : float
        Multiplier on bias Q during dropout (often < 1 to freeze biases).
    freeze_bias_during_blackout : bool
        If True, set bias Q≈0 during dropout to prevent drift.
    reduce_imu_noise_during_blackout : bool
        If True, scale IMU input noise when vision is absent.
    blackout_imu_noise_scale : float
        Scale applied to IMU noise during blackout when enabled.

    enable_zupt : bool
        Enable zero-velocity pseudo-measurements when nearly stationary.
    zupt_velocity_threshold : float
        Speed threshold (m/s) below which ZUPT applies.
    zupt_measurement_noise : float
        ZUPT measurement noise variance ((m/s)^2).

    state_mode : str
        State layout key, e.g. "2d_full" (8D), "vision_only" (5D), or
        "2d_cam_3d_imu" (10D).
    """

    process_noise_pos: float = 0.02
    process_noise_vel: float = 2.0
    process_noise_heading: float = 0.02
    process_noise_gyro_bias: float = 2e-6
    process_noise_accel_bias: float = 2e-4

    measurement_noise_pos: float = 0.005**2
    measurement_noise_heading: float = 0.05**2

    imu_gyro_noise_density: float = 0.0001
    imu_accel_noise_density: float = 0.005

    damping_coeff: float = 0.5
    led_distance: float | None = 0.04

    use_mahalanobis_gating: bool = False  # Disable by default, enable in production
    mahalanobis_threshold_prob: float = 0.997  # Reject ~0.3% of measurements (3σ)

    use_heading_measurement: bool = False
    led_distance_tolerance: float = 0.3
    adaptive_heading_noise: bool = True

    adaptive_q_during_dropout: bool = True
    dropout_q_pos_multiplier: float = 10.0
    dropout_q_vel_multiplier: float = 10.0
    dropout_q_bias_multiplier: float = 0.1
    freeze_bias_during_blackout: bool = False
    reduce_imu_noise_during_blackout: bool = False
    blackout_imu_noise_scale: float = 0.5

    enable_zupt: bool = False
    zupt_velocity_threshold: float = 0.05
    zupt_measurement_noise: float = 0.01**2

    # State layout mode (controls state dimension and index mapping)
    # Supported for 2D paths: "2d_full" (8D), "vision_only" (5D), "2d_cam_3d_imu" (10D)
    state_mode: str = "2d_full"

    # PyTree support: treat `state_mode` as static auxiliary data.
    _TREE_STATIC_FIELDS: ClassVar[tuple[str, ...]] = ("state_mode",)

    def tree_flatten(self) -> tuple[tuple, dict]:
        """Flatten config for JAX PyTree registration.

        Returns
        -------
        tuple[tuple, dict]
            Children tuple containing dynamic fields and static data dictionary
            containing class reference and static field values.

        Notes
        -----
        This method is part of the JAX PyTree protocol. It separates config
        fields into dynamic (children) and static (auxiliary) data for JIT
        compilation and tracing.
        """
        field_names = list(self.__dataclass_fields__.keys())
        children = []
        static_data = {"cls": self.__class__}

        for name in field_names:
            if name.startswith("_"):
                continue
            value = getattr(self, name)
            if name in self._TREE_STATIC_FIELDS:
                static_data[name] = value
            else:
                children.append(value)

        return tuple(children), static_data

    @classmethod
    def tree_unflatten(cls, static_data: dict, children: tuple) -> FilterCoreConfig:
        """Reconstruct config from PyTree children.

        Parameters
        ----------
        static_data : dict
            Static auxiliary data containing class reference and static field values.
            Must include 'cls' key with the target class.
        children : tuple
            Dynamic field values to reconstruct.

        Returns
        -------
        FilterCoreConfig
            Reconstructed configuration instance with all fields restored.

        Notes
        -----
        This method is part of the JAX PyTree protocol. It reconstructs a config
        instance from the separated dynamic (children) and static (auxiliary) data
        produced by tree_flatten().
        """
        target_cls = static_data.get("cls", cls)
        field_names = list(target_cls.__dataclass_fields__.keys())
        child_iter = iter(children)
        static_fields = getattr(target_cls, "_TREE_STATIC_FIELDS", ())

        kwargs = {}
        for name in field_names:
            if name.startswith("_"):
                continue
            if name in static_fields:
                kwargs[name] = static_data[name]
            else:
                kwargs[name] = next(child_iter)

        return target_cls(**kwargs)


tree_util.register_pytree_node_class(FilterCoreConfig)


class FilterState(NamedTuple):
    """Kalman filter state comprising mean vector and covariance matrix.

    Attributes
    ----------
    mean : jnp.ndarray
        State mean (n,). Units depend on layout; typically
        [x(m), y(m), vx(m/s), vy(m/s), θ(rad), b_gz(rad/s), b_ax(m/s^2), b_ay(m/s^2)].
    cov : jnp.ndarray
        State covariance (n, n).
    """

    mean: jnp.ndarray
    cov: jnp.ndarray


def symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    """Enforce numerical symmetry on a square matrix.

    Parameters
    ----------
    matrix : jnp.ndarray
        Input matrix (n, n).

    Returns
    -------
    jnp.ndarray
        Symmetrized matrix (n, n): 0.5·(A + Aᵀ).
    """

    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def psd_solve(matrix: jnp.ndarray, rhs: jnp.ndarray, diagonal_boost: float = 1e-9) -> jnp.ndarray:
    """Solve A x = b for PSD matrices via Cholesky factorization.

    Parameters
    ----------
    matrix : jnp.ndarray
        Positive semi-definite matrix A (k, k).
    rhs : jnp.ndarray
        Right-hand side b. Shape (k,) or (k, m).
    diagonal_boost : float, optional
        Small value added to diag(A) to improve numerical stability.

    Returns
    -------
    jnp.ndarray
        Solution x with shape matching rhs.
    """

    stabilized = symmetrize(matrix) + diagonal_boost * jnp.eye(matrix.shape[-1])
    chol, lower = cho_factor(stabilized, lower=True)
    return cho_solve((chol, lower), rhs)


def joseph_update(
    cov_prior: jnp.ndarray,
    gain: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
) -> jnp.ndarray:
    """Joseph-form covariance update that preserves PSD and symmetry.

    Parameters
    ----------
    cov_prior : jnp.ndarray
        Prior covariance P⁻ (n, n).
    gain : jnp.ndarray
        Kalman gain K (n, k).
    H : jnp.ndarray
        Measurement Jacobian H (k, n).
    R : jnp.ndarray
        Measurement noise covariance R (k, k).

    Returns
    -------
    jnp.ndarray
        Posterior covariance P⁺ (n, n).

    Notes
    -----
    Uses the numerically stable Joseph form:

    P⁺ = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ
    """

    n = cov_prior.shape[0]
    identity = jnp.eye(n)
    I_minus_KH = identity - gain @ H
    return symmetrize(I_minus_KH @ cov_prior @ I_minus_KH.T + gain @ R @ gain.T)


def wrap_angle(theta: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to (-π, π] to avoid discontinuities.

    Parameters
    ----------
    theta : jnp.ndarray
        Angle(s) in radians, arbitrary shape.

    Returns
    -------
    jnp.ndarray
        Wrapped angle(s), same shape as input.
    """

    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def rotate_body_accel_to_world(
    accel_body: jnp.ndarray, yaw_heading: float | jnp.ndarray
) -> jnp.ndarray:
    """Rotate 3D body-frame acceleration to world frame using yaw angle.

    Parameters
    ----------
    accel_body : jnp.ndarray
        Acceleration in body frame (3,) [ax, ay, az] in m/s².
    yaw_heading : float or jnp.ndarray
        Yaw heading angle (rad). Rotation about vertical (z) axis.

    Returns
    -------
    jnp.ndarray
        Acceleration in world frame (3,) [ax_w, ay_w, az_w] in m/s².

    Notes
    -----
    Applies R_z(θ) rotation matrix to the x-y plane while preserving z:

    [ax_w]   [cos(θ)  -sin(θ)  0] [ax]
    [ay_w] = [sin(θ)   cos(θ)  0] [ay]
    [az_w]   [0        0       1] [az]

    This is the correct transformation for converting IMU measurements from
    the body frame (where the IMU is mounted on the rat) to the world frame
    (the fixed laboratory coordinate system).
    """
    yaw = jnp.asarray(yaw_heading)
    cos_yaw = jnp.cos(yaw)
    sin_yaw = jnp.sin(yaw)

    # Rotation matrix R_z(yaw) for yaw-only rotation (3x3)
    # Only affects x-y plane; z is unchanged
    R_z = jnp.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]])

    return R_z @ accel_body


def gravity_compensate(accel_world: jnp.ndarray, g: float = 9.81) -> jnp.ndarray:
    """Remove gravity from world-frame acceleration.

    Parameters
    ----------
    accel_world : jnp.ndarray
        Acceleration in world frame (3,) [ax, ay, az] in m/s².
    g : float, default 9.81
        Gravitational acceleration magnitude (m/s²).

    Returns
    -------
    jnp.ndarray
        Gravity-compensated acceleration (3,) in m/s².

    Notes
    -----
    Removes the gravitational acceleration vector [0, 0, g] from the
    world-frame measurement. This is necessary because IMUs measure
    specific force (proper acceleration), which includes gravity.

    **IMU Convention:** Assumes IMU reports specific force (proper acceleration).
    Kinematic acceleration is: a = R f_body - g_world

    The IMU at rest reads [0, 0, +g] due to the normal force from the
    surface. Motion creates additional accelerations that add to this
    baseline. Subtracting [0, 0, g] recovers the kinematic acceleration
    (coordinate acceleration) needed for state propagation.
    """
    gravity_vector = jnp.array([0.0, 0.0, g])
    return accel_world - gravity_vector


def chi2_threshold(dof: int, prob: float) -> jnp.ndarray:
    """Closed-form χ² thresholds for common degrees of freedom.

    Parameters
    ----------
    dof : int
        Degrees of freedom, {2, 4} for single-LED vs dual-LED measurements.
    prob : float
        Probability mass for threshold, one of {0.95, 0.975, 0.99, 0.997}.

    Returns
    -------
    jnp.ndarray
        χ² threshold value (scalar).

    Notes
    -----
    Values match ``scipy.stats.chi2.ppf(prob, dof)`` for the listed pairs.
    - dof=2: Mahalanobis distance² for single LED (x, y)
    - dof=4: Mahalanobis distance² for dual LEDs (x1, y1, x2, y2)
    """
    # Thresholds for dof=2 (single LED)
    threshold_2 = lax.select(
        jnp.abs(prob - 0.997) < 0.001,
        11.618,  # 99.7% (3σ) - very tight gate
        lax.select(
            jnp.abs(prob - 0.99) < 0.001,
            9.210,  # 99% - tight gate
            lax.select(
                jnp.abs(prob - 0.975) < 0.001,
                7.378,  # 97.5% (2.5σ) - moderate gate
                5.991,  # 95% (2σ) - loose gate (default fallback)
            ),
        ),
    )

    # Thresholds for dof=4 (dual LEDs)
    threshold_4 = lax.select(
        jnp.abs(prob - 0.997) < 0.001,
        16.014,  # 99.7% (3σ) - very tight gate
        lax.select(
            jnp.abs(prob - 0.99) < 0.001,
            13.277,  # 99% - tight gate
            lax.select(
                jnp.abs(prob - 0.975) < 0.001,
                11.143,  # 97.5% (2.5σ) - moderate gate
                9.488,  # 95% (2σ) - loose gate (default fallback)
            ),
        ),
    )

    return lax.select(dof == 2, threshold_2, threshold_4)


def dynamics_function(
    state: jnp.ndarray,
    imu: jnp.ndarray,
    dt: float,
    damping: float,
    layout: StateLayout,
) -> jnp.ndarray:
    """Constant-acceleration dynamics with linear damping (layout-aware).

    Supports both 2D and 3D IMU inputs:
    - 2D IMU: [ω_z(rad/s), f_x(m/s²), f_y(m/s²)] (3,)
    - 3D IMU: [ω_z(rad/s), f_x(m/s²), f_y(m/s²), f_z(m/s²)] (4,)

    For 3D IMU, applies gravity compensation and 3D rotation.

    Parameters
    ----------
    state : jnp.ndarray
        State vector (n,). Layout-dependent structure.
    imu : jnp.ndarray
        IMU measurements. Either 3-element (2D) or 4-element (3D).
    dt : float
        Time step (s).
    damping : float
        Linear velocity damping coefficient (1/s).
    layout : StateLayout
        State index mapping.

    Returns
    -------
    jnp.ndarray
        Next state (n,).

    Notes
    -----
    For 2D IMU mode:
        vₖ₊₁ = vₖ + (R₂ₓ₂ f − γ vₖ) dt
        pₖ₊₁ = pₖ + vₖ dt + 1/2 (R₂ₓ₂ f − γ vₖ) dt²

    For 3D IMU mode:
        f_world = R₃ₓ₃(θ) @ f_body
        a_kinematic = f_world - [0, 0, g]  (gravity compensation)
        vₖ₊₁ = vₖ + (a_kinematic − γ vₖ) dt
        pₖ₊₁ = pₖ + vₖ dt + 1/2 (a_kinematic − γ vₖ) dt²

    Heading update:
        θₖ₊₁ = θₖ + (ω_z − b_gz) dt

    Unused layout components are propagated as identity.
    """

    # Extract indices (2D heading only)
    h_idx = get_heading_index(layout)
    px_i, py_i = layout.pos_idx[0], layout.pos_idx[1]
    vx_i, vy_i = layout.vel_idx[0], layout.vel_idx[1]

    # Check if layout has 3D velocity (vz)
    has_3d_velocity = len(layout.vel_idx) >= 3

    # Bias indices (may be empty for vision-only)
    b_gz = state[layout.bias_gyro_idx[0]] if len(layout.bias_gyro_idx) >= 1 else 0.0
    b_ax = state[layout.bias_accel_idx[0]] if len(layout.bias_accel_idx) >= 1 else 0.0
    b_ay = state[layout.bias_accel_idx[1]] if len(layout.bias_accel_idx) >= 2 else 0.0
    b_az = state[layout.bias_accel_idx[2]] if len(layout.bias_accel_idx) >= 3 else 0.0

    # Current values
    px, py = state[px_i], state[py_i]
    vx, vy = state[vx_i], state[vy_i]
    theta = state[h_idx]

    # Detect IMU dimension: 3-element (2D) or 4-element (3D)
    imu_is_3d = imu.shape[0] >= 4

    # Update heading
    omega_z = imu[0]
    omega_z_unbiased = omega_z - b_gz
    theta_next = theta + omega_z_unbiased * dt

    if imu_is_3d and has_3d_velocity:
        # 3D IMU mode: [ω_z, fx, fy, fz]
        # Consistency check: 3D velocity requires 3D accel bias
        if len(layout.bias_accel_idx) < 3:
            raise ValueError(
                f"3D IMU mode requires 3D accel bias (b_ax, b_ay, b_az), "
                f"but layout has only {len(layout.bias_accel_idx)} accel bias terms. "
                f"Use LAYOUT_2D_CAM_3D_IMU or ensure bias_accel_idx has length 3."
            )

        # Extract 3D velocity
        vz_i = layout.vel_idx[2]
        vz = state[vz_i]

        # Extract 3D accelerations from IMU
        fx, fy, fz = imu[1], imu[2], imu[3]

        # Remove biases (body frame)
        accel_body = jnp.array([fx - b_ax, fy - b_ay, fz - b_az])

        # Rotate from body frame to world frame using current heading
        accel_world = rotate_body_accel_to_world(accel_body, theta)

        # Compensate for gravity (remove [0, 0, g] from world-frame measurement)
        accel_kinematic = gravity_compensate(accel_world, g=9.81)

        # Update 3D velocity with damping
        vel = jnp.array([vx, vy, vz])
        vel_next = vel + accel_kinematic * dt - damping * vel * dt

        # Update 2D position (only x, y; no z position in LAYOUT_2D_CAM_3D_IMU)
        pos = jnp.array([px, py])
        # Only use horizontal components for position update
        accel_horizontal = accel_kinematic[:2]
        vel_horizontal = vel[:2]
        pos_next = (
            pos
            + vel_horizontal * dt
            + 0.5 * accel_horizontal * dt**2
            - 0.5 * damping * vel_horizontal * dt**2
        )

        # Update state
        next_state = state
        next_state = next_state.at[px_i].set(pos_next[0])
        next_state = next_state.at[py_i].set(pos_next[1])
        next_state = next_state.at[vx_i].set(vel_next[0])
        next_state = next_state.at[vy_i].set(vel_next[1])
        next_state = next_state.at[vz_i].set(vel_next[2])
        next_state = next_state.at[h_idx].set(theta_next)

    else:
        # 2D IMU mode: [ω_z, fx, fy] (backward compatible)
        fx, fy = imu[1], imu[2]
        accel_body = jnp.array([fx - b_ax, fy - b_ay])

        # 2D rotation (yaw only)
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        R = jnp.array([[cos_t, -sin_t], [sin_t, cos_t]])
        accel_world = R @ accel_body

        # Update 2D velocity
        vel = jnp.array([vx, vy])
        vel_next = vel + accel_world * dt - damping * vel * dt

        # Update 2D position
        pos = jnp.array([px, py])
        pos_next = pos + vel * dt + 0.5 * accel_world * dt**2 - 0.5 * damping * vel * dt**2

        # Update state
        next_state = state
        next_state = next_state.at[px_i].set(pos_next[0])
        next_state = next_state.at[py_i].set(pos_next[1])
        next_state = next_state.at[vx_i].set(vel_next[0])
        next_state = next_state.at[vy_i].set(vel_next[1])
        next_state = next_state.at[h_idx].set(theta_next)

    return next_state


def measurement_function(
    state: jnp.ndarray, led_distance: float, layout: StateLayout
) -> jnp.ndarray:
    """Project state into dual-LED measurement space (layout-aware).

    Parameters
    ----------
    state : jnp.ndarray
        State (n,).
    led_distance : float
        LED spacing (m).
    layout : StateLayout
        State index mapping.

    Returns
    -------
    jnp.ndarray
        Measurement vector (4,) ordered as [x1, y1, x2, y2] in meters.
    """

    h_idx = get_heading_index(layout)
    px = state[layout.pos_idx[0]]
    py = state[layout.pos_idx[1]]
    theta = state[h_idx]
    dx = 0.5 * led_distance * jnp.cos(theta)
    dy = 0.5 * led_distance * jnp.sin(theta)
    return jnp.array([px - dx, py - dy, px + dx, py + dy])


def make_led_selector(only_led1: bool | Array, only_led2: bool | Array) -> Array:
    """Create 2×4 selector matrix for single-LED observations.

    Parameters
    ----------
    only_led1 : bool | Array
        Boolean value or boolean scalar array. True if only LED1 is valid.
    only_led2 : bool | Array
        Boolean value or boolean scalar array. True if only LED2 is valid.

    Returns
    -------
    Array
        Selector matrix ``M`` (2, 4) such that ``M @ [x1,y1,x2,y2]`` extracts
        the active LED's 2D subspace.
    """
    # LED1 selector: picks first 2 dimensions
    M_led1 = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    # LED2 selector: picks last 2 dimensions
    M_led2 = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    # Select based on which LED is valid
    return lax.select(only_led1, M_led1, M_led2)


def apply_lifted_inverse(
    S4: jnp.ndarray,
    w4: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> jnp.ndarray:
    """Apply effective inverse in active subspace then lift to 4D.

    Parameters
    ----------
    S4 : jnp.ndarray
        Innovation covariance in full space (4, 4).
    w4 : jnp.ndarray
        Vector to multiply (4,).
    both_leds : bool
        True if both LEDs are valid (4D update).
    only_led1 : bool
        True if only LED1 is valid (2D update).
    only_led2 : bool
        True if only LED2 is valid (2D update).

    Returns
    -------
    jnp.ndarray
        Result x = S_eff⁻¹ @ w in 4D with static shape (4,).

    Notes
    -----
    For single-LED updates, compute in the 2D active subspace and lift:

    x₄ = Mᵀ · (M S₄ Mᵀ)⁻¹ · (M w₄)
    """
    # 4D path: both LEDs valid
    x4_full = psd_solve(S4, w4)

    # 2D path: single LED valid
    M2 = make_led_selector(only_led1, only_led2)  # (2, 4)
    S2 = M2 @ S4 @ M2.T  # (2, 2) - subspace innovation covariance
    w2 = M2 @ w4  # (2,) - project to subspace
    x2 = psd_solve(S2, w2)  # (2,) - solve in subspace
    x4_lifted = M2.T @ x2  # (4,) - lift back to 4D

    # Select based on LED validity (both branches return same shape)
    return lax.select(both_leds, x4_full, x4_lifted)


def initialize_state(
    led1_obs: jnp.ndarray,
    led2_obs: jnp.ndarray,
    observation_mask: jnp.ndarray,
    dt_cam: float | jnp.ndarray,
    led_distance: float = 0.04,
    *,
    layout: StateLayout | None = None,
) -> FilterState:
    """Bootstrap filter state from early LED observations.

    Parameters
    ----------
    led1_obs : jnp.ndarray
        LED1 observations (N, 2) in meters.
    led2_obs : jnp.ndarray
        LED2 observations (N, 2) in meters.
    observation_mask : jnp.ndarray
        Observation validity mask (N,), boolean.
    dt_cam : float or jnp.ndarray
        Camera frame interval (s). JAX scalar allowed for JIT.
    led_distance : float, default 0.04
        LED spacing (m) used to infer heading when both LEDs visible.
    layout : StateLayout, optional
        State mapping; defaults to "2d_full" if not provided.

    Returns
    -------
    FilterState
        Initial mean (n,) and covariance (n, n).

    Notes
    -----
    If all observations are invalid, initializes near the origin with large
    uncertainty, allowing prediction-only filtering to proceed.
    """

    # Find frames with valid observation mask AND finite LED observations
    # (the mask alone isn't sufficient -- LEDs can be NaN even when observation_mask=True)
    led1_finite_mask = jnp.isfinite(led1_obs[:, 0])
    led2_finite_mask = jnp.isfinite(led2_obs[:, 0])
    any_led_finite = led1_finite_mask | led2_finite_mask
    valid_with_data = observation_mask & any_led_finite

    valid_indices = jnp.where(valid_with_data)[0]
    has_valid_obs = len(valid_indices) > 0
    first_valid = valid_indices[0] if has_valid_obs else 0

    # Check LED validity at first valid frame
    led1_valid = jnp.isfinite(led1_obs[first_valid, 0]) if has_valid_obs else False
    led2_valid = jnp.isfinite(led2_obs[first_valid, 0]) if has_valid_obs else False

    # Replace NaN with zero to prevent propagation (only used if marked invalid)
    pos_led1 = jnp.where(
        jnp.isfinite(led1_obs[first_valid]), led1_obs[first_valid], jnp.array([0.0, 0.0])
    )
    pos_led2 = jnp.where(
        jnp.isfinite(led2_obs[first_valid]), led2_obs[first_valid], jnp.array([0.0, 0.0])
    )

    pos_init = jnp.where(
        led1_valid & led2_valid,
        (pos_led1 + pos_led2) / 2.0,
        jnp.where(
            led1_valid,
            pos_led1,
            jnp.where(led2_valid, pos_led2, jnp.array([0.0, 0.0])),
        ),
    )

    def compute_velocity() -> jnp.ndarray:
        idx1 = valid_indices[0]
        idx2 = valid_indices[1]
        dt = (idx2 - idx1) * dt_cam

        led1_1, led2_1 = led1_obs[idx1], led2_obs[idx1]
        led1_2, led2_2 = led1_obs[idx2], led2_obs[idx2]

        led1_1_valid = jnp.isfinite(led1_1[0])
        led2_1_valid = jnp.isfinite(led2_1[0])
        led1_2_valid = jnp.isfinite(led1_2[0])
        led2_2_valid = jnp.isfinite(led2_2[0])

        pos1 = jnp.where(
            led1_1_valid & led2_1_valid,
            (led1_1 + led2_1) / 2.0,
            jnp.where(led1_1_valid, led1_1, led2_1),
        )
        pos2 = jnp.where(
            led1_2_valid & led2_2_valid,
            (led1_2 + led2_2) / 2.0,
            jnp.where(led1_2_valid, led1_2, led2_2),
        )
        return (pos2 - pos1) / dt

    # Only compute velocity if we have at least 2 valid frames
    vel_init = compute_velocity() if len(valid_indices) >= 2 else jnp.zeros(2)

    led_vec = pos_led2 - pos_led1
    heading_from_leds = jnp.arctan2(led_vec[1], led_vec[0])
    heading_init = jnp.where(led1_valid & led2_valid, heading_from_leds, 0.0)

    # Build 8D default mean/cov, then adapt to layout based on desired state_mode
    heading_std = jnp.where(led1_valid & led2_valid, jnp.pi / 4, jnp.pi / 2)

    mean8 = jnp.array(
        [
            pos_init[0],
            pos_init[1],
            vel_init[0],
            vel_init[1],
            heading_init,
            0.0,
            0.0,
            0.0,
        ]
    )
    cov8 = jnp.diag(
        jnp.array(
            [
                0.01**2,
                0.01**2,
                0.1**2,
                0.1**2,
                heading_std**2,
                0.05**2,
                0.1**2,
                0.1**2,
            ]
        )
    )

    # Determine layout (defaults to 2D full if not provided)
    layout = get_layout("2d_full") if layout is None else layout
    n = layout.n
    mean = jnp.zeros(n)
    cov = jnp.eye(n) * 1.0

    # Map 2D pos/vel/heading
    mean = mean.at[layout.pos_idx[0]].set(mean8[0])
    mean = mean.at[layout.pos_idx[1]].set(mean8[1])
    mean = mean.at[layout.vel_idx[0]].set(mean8[2])
    mean = mean.at[layout.vel_idx[1]].set(mean8[3])
    mean = mean.at[get_heading_index(layout)].set(mean8[4])

    cov = cov.at[layout.pos_idx[0], layout.pos_idx[0]].set(cov8[0, 0])
    cov = cov.at[layout.pos_idx[1], layout.pos_idx[1]].set(cov8[1, 1])
    cov = cov.at[layout.vel_idx[0], layout.vel_idx[0]].set(cov8[2, 2])
    cov = cov.at[layout.vel_idx[1], layout.vel_idx[1]].set(cov8[3, 3])
    cov = cov.at[get_heading_index(layout), get_heading_index(layout)].set(cov8[4, 4])

    # Bias variances if present
    if len(layout.bias_gyro_idx) >= 1:
        cov = cov.at[layout.bias_gyro_idx[0], layout.bias_gyro_idx[0]].set(cov8[5, 5])
    if len(layout.bias_accel_idx) >= 1:
        cov = cov.at[layout.bias_accel_idx[0], layout.bias_accel_idx[0]].set(cov8[6, 6])
    if len(layout.bias_accel_idx) >= 2:
        cov = cov.at[layout.bias_accel_idx[1], layout.bias_accel_idx[1]].set(cov8[7, 7])

    return FilterState(mean=mean, cov=cov)


def update_zupt(
    state: FilterState,
    config: FilterCoreConfig,
) -> tuple[FilterState, jnp.ndarray]:
    """Apply zero-velocity pseudo-measurement when nearly stationary.

    Parameters
    ----------
    state : FilterState
        Current state.
    config : FilterCoreConfig
        ZUPT parameters in config.

    Returns
    -------
    tuple[FilterState, jnp.ndarray]
        Updated state and log-likelihood (scalar).

    Notes
    -----
    This function is a compatibility wrapper that uses ZUPTModel internally.
    It maintains the existing API while adopting the new sensor architecture.
    """

    from trodestrack.models.sensors.zupt import ZUPTModel

    mean, cov = state
    n = mean.shape[0]

    # Determine layout from state dimension
    # Find matching layout in registry (same logic as legacy zupt_model)
    # Assumption: unique mapping from state dimension to layout
    # If multiple layouts share same dimension, the first match wins
    from trodestrack.models.state_layout import LAYOUT_REGISTRY

    layout = None
    for mode_layout in LAYOUT_REGISTRY.values():
        if mode_layout.n == n:
            layout = mode_layout
            break

    if layout is None:
        # Fallback: assume 2d_full layout
        from trodestrack.models.state_layout import get_layout

        layout = get_layout("2d_full")

    # Create ZUPT model (fully pure, no mutable state)
    zupt_model = ZUPTModel(
        enable_zupt=config.enable_zupt,
        velocity_threshold=config.zupt_velocity_threshold,
        measurement_noise=config.zupt_measurement_noise,
        layout=layout,
        dtype=mean.dtype,
    )

    # Extract measurement components (all pure functions, JIT-safe)
    meas_pred = zupt_model.predict(mean)
    H = zupt_model.jacobian(mean)
    R = zupt_model.meas_cov_from_pred(meas_pred)  # Pure: derive R from velocity
    innovation = zupt_model.innovation(frame_idx=0, meas_pred=meas_pred)

    # Standard Kalman update
    S = H @ cov @ H.T + R
    K = psd_solve(S, H @ cov).T

    mean_updated = mean + K @ innovation
    cov_updated = joseph_update(cov, K, H, R)

    # Log-likelihood
    log_det = jnp.linalg.slogdet(S)[1]
    innov_quad = innovation @ psd_solve(S, innovation)
    log_likelihood = -0.5 * (2 * jnp.log(2 * jnp.pi) + log_det + innov_quad)

    # Zero out log-likelihood when ZUPT is gated out (derive from same R)
    # R diagonal is 1e6 when disabled (moving or ZUPT off)
    is_active = jnp.trace(R) < 1e5
    log_likelihood = lax.select(
        is_active, log_likelihood, jnp.array(0.0, dtype=log_likelihood.dtype)
    )

    return FilterState(mean=mean_updated, cov=cov_updated), log_likelihood


def confidence_to_R_diagonal(
    confidence: jnp.ndarray | None,
    *,
    base: float,
    size: int,
    clip_min: float = 1e-2,
) -> jnp.ndarray:
    """Map confidence scores to per-dimension measurement noise.

    Parameters
    ----------
    confidence : jnp.ndarray or None
        Confidence per measurement dimension, shape (size,) in [0, 1].
        If None, no scaling is applied.
    base : float
        Base variance per dimension (units^2).
    size : int
        Number of measurement dimensions.
    clip_min : float, default 1e-2
        Lower bound for confidence to avoid division by zero.

    Returns
    -------
    jnp.ndarray
        Diagonal entries of R (size,), where R_i = base / clip(conf_i, clip_min, 1).
    """
    if confidence is None:
        return jnp.full(size, base)
    conf = jnp.clip(confidence, clip_min, 1.0)
    return base / conf


def gaussian_log_likelihood(innovation: jnp.ndarray, covariance: jnp.ndarray) -> jnp.ndarray:
    """Gaussian log-likelihood of an innovation with stability tweaks.

    Parameters
    ----------
    innovation : jnp.ndarray
        Innovation vector v (k,).
    covariance : jnp.ndarray
        Innovation covariance S (k, k).

    Returns
    -------
    jnp.ndarray
        Log-likelihood log p(v | 0, S) (scalar).

    Notes
    -----
    Computes ``-0.5 * (k log(2π) + log det S + vᵀ S⁻¹ v)`` with small diagonal
    jitter to improve conditioning.
    """
    k = innovation.shape[0]

    # Add small jitter to diagonal for numerical stability
    # Scale by mean diagonal value to be adaptive
    jitter = 1e-8 * jnp.trace(covariance) / k
    S_stable = symmetrize(covariance) + jnp.eye(k) * jitter

    # Log determinant using slogdet (more stable than det)
    sign, logdet = jnp.linalg.slogdet(S_stable)

    # Check for numerical issues (sign should be +1 for PSD matrix)
    # If sign <= 0, increase jitter and recompute
    def add_more_jitter():
        jitter_large = 1e-6 * jnp.trace(covariance) / k
        S_jittered = symmetrize(covariance) + jnp.eye(k) * jitter_large
        sign_j, logdet_j = jnp.linalg.slogdet(S_jittered)
        return logdet_j

    # Use original logdet if sign is positive, otherwise use jittered version
    logdet_safe = lax.cond(sign > 0, lambda: logdet, add_more_jitter)

    # Mahalanobis distance: v^T S^{-1} v
    # psd_solve computes S^{-1} @ v, then we dot with v
    S_inv_v = psd_solve(S_stable, innovation)
    mahal = jnp.dot(innovation, S_inv_v)

    # Gaussian log-likelihood
    log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + logdet_safe + mahal)

    return log_prob


def compute_nis_and_loglik(
    innov4: jnp.ndarray,
    S4: jnp.ndarray,
    both_leds: bool,
    only_led1: bool,
    only_led2: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact NIS and log-likelihood in active measurement subspace.

    Parameters
    ----------
    innov4 : jnp.ndarray
        Innovation in 4D measurement space (4,) [x1,y1,x2,y2] (m).
    S4 : jnp.ndarray
        Innovation covariance in 4D (4, 4) (m^2).
    both_leds : bool
        True if both LEDs valid → 4D.
    only_led1 : bool
        True if only LED1 valid → 2D.
    only_led2 : bool
        True if only LED2 valid → 2D.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(nis, log_likelihood)`` scalars.

    Notes
    -----
    Uses Cholesky-based solves. For 2D cases, projects via selector matrix
    before computing statistics, ensuring exact results without diagonal
    approximations.
    """
    from jax.scipy.linalg import cho_solve

    # 4D branch: both LEDs valid
    def compute_4d():
        S4s = symmetrize(S4)
        eps = jnp.asarray(1e-9, dtype=S4s.dtype)
        L4 = jnp.linalg.cholesky(S4s + eps * jnp.eye(4, dtype=S4s.dtype))
        x4 = cho_solve((L4, True), innov4)
        nis = jnp.dot(innov4, x4)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L4)))
        loglik = -0.5 * (logdet + nis + 4 * jnp.log(2 * jnp.pi))
        return nis, loglik

    # 2D branch: single LED valid
    def compute_2d():
        M2 = make_led_selector(only_led1, only_led2)  # (2, 4)
        S2 = M2 @ symmetrize(S4) @ M2.T  # (2, 2)
        innov2 = M2 @ innov4  # (2,)

        eps = jnp.asarray(1e-9, dtype=S2.dtype)
        L2 = jnp.linalg.cholesky(S2 + eps * jnp.eye(2, dtype=S2.dtype))
        x2 = cho_solve((L2, True), innov2)
        nis = jnp.dot(innov2, x2)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L2)))
        loglik = -0.5 * (logdet + nis + 2 * jnp.log(2 * jnp.pi))
        return nis, loglik

    # Select based on LED validity
    return lax.cond(both_leds, compute_4d, compute_2d)


def prepare_heading_measurement(
    z_led1: jnp.ndarray,
    z_led2: jnp.ndarray,
    config: FilterCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare heading pseudo-measurement from LED geometry.

    Parameters
    ----------
    z_led1 : jnp.ndarray
        LED1 observation (2,) [x, y] in meters.
    z_led2 : jnp.ndarray
        LED2 observation (2,) [x, y] in meters.
    config : FilterCoreConfig
        Heading measurement configuration.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(heading_obs, R_heading, use_heading)`` where
        ``heading_obs`` is in radians, ``R_heading`` is variance (rad^2), and
        ``use_heading`` is a boolean JAX scalar.

    Notes
    -----
    Computes θ_obs = arctan2(dy, dx). If both LEDs are valid and the observed
    spacing is within the tolerance of the expected spacing, returns a small
    ``R_heading`` (possibly adapted by spacing ratio). Otherwise, returns a
    large ``R_heading`` (1e6) which effectively gates out the update.
    """
    # Check LED validity
    led1_valid = jnp.isfinite(z_led1).all()
    led2_valid = jnp.isfinite(z_led2).all()
    both_leds = led1_valid & led2_valid

    # Compute heading observation (always compute, gate via R)
    dx = z_led2[0] - z_led1[0]
    dy = z_led2[1] - z_led1[1]
    heading_obs = jnp.arctan2(dy, dx)

    # Check LED spacing validity
    obs_spacing = jnp.sqrt(dx**2 + dy**2)
    obs_spacing_valid = jnp.isfinite(obs_spacing) & (obs_spacing > 1e-6)

    if config.led_distance is not None:
        expected_spacing = jnp.asarray(config.led_distance, dtype=obs_spacing.dtype)
        spacing_ratio = jnp.where(
            obs_spacing_valid,
            obs_spacing / expected_spacing,
            jnp.zeros_like(obs_spacing),
        )
        spacing_valid = obs_spacing_valid & (
            (spacing_ratio > (1 - config.led_distance_tolerance))
            & (spacing_ratio < (1 + config.led_distance_tolerance))
        )
    else:
        expected_spacing = jnp.where(
            obs_spacing_valid,
            obs_spacing,
            jnp.asarray(1.0, dtype=obs_spacing.dtype),
        )
        spacing_valid = obs_spacing_valid

    # Overall validity: both LEDs + spacing OK + feature enabled
    use_heading = config.use_heading_measurement & both_leds & spacing_valid

    # Base heading measurement noise
    R_base = config.measurement_noise_heading

    # Adaptive noise scaling (if enabled and spacing is valid)
    # Clip obs_spacing to avoid division by zero/NaN
    obs_spacing_safe = jnp.where(
        obs_spacing_valid,
        obs_spacing,
        jnp.maximum(expected_spacing, jnp.asarray(1e-3, dtype=obs_spacing.dtype)),
    )
    R_heading_adapted = lax.cond(
        config.adaptive_heading_noise,
        lambda: R_base * (expected_spacing / obs_spacing_safe) ** 2,
        lambda: R_base,
    )

    # Gate via large R (JAX-friendly: no branching)
    # Valid: R ≈ 0.05² → strong update
    # Invalid: R = 1e6 → K ≈ 0 → no update
    R_heading = lax.select(use_heading, R_heading_adapted, 1e6)

    return heading_obs, R_heading, use_heading


# =============================================================================
# LED Spacing Estimation (shared by EKF and UKF)
# =============================================================================


def estimate_led_spacing(
    Z_cam_led1: jnp.ndarray,
    Z_cam_led2: jnp.ndarray,
    mask_cam: jnp.ndarray,
) -> float:
    """Estimate LED spacing from camera observations.

    Uses host-side NumPy to avoid expensive JIT compilation of nanmedian.

    Parameters
    ----------
    Z_cam_led1 : jnp.ndarray
        LED1 positions (N_cam, 2) in meters.
    Z_cam_led2 : jnp.ndarray
        LED2 positions (N_cam, 2) in meters.
    mask_cam : jnp.ndarray
        Camera validity mask (N_cam,), boolean.

    Returns
    -------
    float
        Median LED spacing (m). Falls back to 0.04 m if no valid dual-LED frames.
    """
    # Convert to NumPy for host-side computation (avoid JIT nanmedian)
    import numpy as np

    led1_np = np.asarray(Z_cam_led1)
    led2_np = np.asarray(Z_cam_led2)
    mask_np = np.asarray(mask_cam)

    # Find frames where both LEDs are visible
    led1_valid = np.isfinite(led1_np).all(axis=1)
    led2_valid = np.isfinite(led2_np).all(axis=1)
    both_valid = led1_valid & led2_valid & mask_np

    # Compute distances for valid frames
    distances = np.linalg.norm(led2_np - led1_np, axis=1)

    # Median of valid distances
    valid_distances = np.where(both_valid, distances, np.nan)

    # Use nanmedian, with fallback if all NaN
    median_spacing = np.nanmedian(valid_distances)

    # Fallback to 4 cm if no valid observations
    spacing = 0.04 if not np.isfinite(median_spacing) else float(median_spacing)
    return spacing


# =============================================================================
# IMU Index Computation (previously in filter_utils.py)
# =============================================================================


def compute_imu_index_arrays(
    t_imu: np.ndarray | jnp.ndarray, t_cam: np.ndarray | jnp.ndarray
) -> jnp.ndarray:
    """Build padded index arrays for IMU samples between camera frames.

    Parameters
    ----------
    t_imu : jnp.ndarray
        IMU timestamps (N_imu,) in seconds.
    t_cam : jnp.ndarray
        Camera timestamps (N_cam,) in seconds.

    Returns
    -------
    jnp.ndarray
        Index array (N_cam, max_imu_per_frame) of IMU indices; -1 indicates padding
        (no IMU sample). Returned as a JAX array for device use.

    Notes
    -----
    Host-side precomputation using NumPy avoids dynamic loop unrolling inside JIT.
    For each frame i, finds IMU indices in the half-open interval (t_cam[i-1], t_cam[i]].
    """
    t_imu_np = np.asarray(t_imu)
    t_cam_np = np.asarray(t_cam)

    n_cam = len(t_cam_np)
    all_indices = []

    # First pass: collect all valid index arrays to find max length
    for i in range(n_cam):
        if i == 0:
            # First frame: no IMU propagation
            valid_indices = np.array([], dtype=np.int32)
        else:
            # Find IMU samples in (t_prev, t_current]
            interval_mask = (t_imu_np > t_cam_np[i - 1]) & (t_imu_np <= t_cam_np[i])
            valid_indices = np.nonzero(interval_mask)[0]

        all_indices.append(valid_indices)

    # Compute max length from actual data
    max_imu_per_frame = max(len(idx) for idx in all_indices)

    # Second pass: pad all arrays to max length
    padded_indices = []
    for valid_indices in all_indices:
        indices = np.full(max_imu_per_frame, -1, dtype=np.int32)
        if len(valid_indices) > 0:
            indices[: len(valid_indices)] = valid_indices
        padded_indices.append(indices)

    # Convert to JAX array for device use
    return jnp.array(padded_indices, dtype=jnp.int32)


# =============================================================================
# IMU Noise Propagation Matrices
# =============================================================================


def build_G_matrix(theta: float | Array, dt: float | Array) -> jnp.ndarray:
    """IMU input noise propagation matrix G for standard 8-state model.

    Parameters
    ----------
    theta : float
        Heading angle (rad).
    dt : float
        Time step (s).

    Returns
    -------
    jnp.ndarray
        G matrix (8, 3) mapping IMU noise [ω_z, f_x, f_y] to state.

    Notes
    -----
    State: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]. Input: [ω_z, f_x, f_y].
    Dependencies:
    - θₖ₊₁ = θₖ + (ω_z − b_gz) dt  → ∂θ/∂ω_z = dt
    - vₖ₊₁ = vₖ + R(θ)(f − b_a) dt → ∂v/∂f = R(θ) dt
    - pₖ₊₁ = pₖ + v dt + 0.5 R(θ)(f − b_a) dt² → ∂p/∂f = R(θ) 0.5 dt²
    """
    theta_arr = jnp.asarray(theta)
    dt_arr = jnp.asarray(dt)

    # 2D rotation matrix R(θ)
    c, s = jnp.cos(theta_arr), jnp.sin(theta_arr)
    R_2d = jnp.array([[c, -s], [s, c]])

    # Initialize G matrix: state (8) × input (3)
    # Rows: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    # Cols: [ω_z, f_x, f_y]
    G = jnp.zeros((8, 3))

    # Heading depends on gyro: ∂θ/∂ω_z = dt
    G = G.at[4, 0].set(dt_arr)

    # Velocity depends on accelerometer via rotation: ∂v/∂f = R(θ) * dt
    G = G.at[2:4, 1:3].set(R_2d * dt_arr)

    # Position depends on accelerometer: ∂p/∂f = R(θ) * 0.5 * dt²
    G = G.at[0:2, 1:3].set(R_2d * (0.5 * dt_arr * dt_arr))

    return G


def build_G_matrix_generic(
    n: int,
    theta: float | Array,
    dt: float | Array,
    *,
    pos_idx: tuple[int, int] = (0, 1),
    vel_idx: tuple[int, int] | tuple[int, int, int] = (2, 3),
    theta_idx: int = 4,
    n_accel: int = 2,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Generic IMU input noise mapping G for arbitrary layouts.

    Parameters
    ----------
    n : int
        State dimension.
    theta : float
        Heading angle (rad).
    dt : float
        Time step (s).
    pos_idx : tuple[int, int], default (0, 1)
        Position indices (x, y). Only 2D positions supported.
    vel_idx : tuple[int, int] or tuple[int, int, int], default (2, 3)
        Velocity indices (vx, vy) or (vx, vy, vz). Length must match n_accel.
    theta_idx : int, default 4
        Heading index.
    n_accel : int, default 2
        Number of accelerometer axes (2 for 2D, 3 for 3D).
    dtype : jnp.dtype, default jnp.float32
        Array dtype.

    Returns
    -------
    jnp.ndarray
        G matrix (n, n_accel+1) for [ω_z, f_x, f_y, ...].

    Notes
    -----
    Places ∂θ/∂ω_z = dt at ``theta_idx``, ∂v/∂f = R(θ)·dt at ``vel_idx``,
    and ∂p/∂f = R(θ)·0.5·dt² at ``pos_idx``. Missing/out-of-bounds indices
    are ignored.

    For n_accel=3 (3D accel), the z-axis velocity is affected directly by f_z
    without rotation: ∂vz/∂f_z = dt (since z-axis is vertical).

    Raises
    ------
    ValueError
        If n_accel is not 2 or 3.
        If n_accel=3 but len(vel_idx) != 3 (inconsistent configuration).
        If n_accel=2 but len(vel_idx) not in (2, 3).
    """
    if n_accel not in (2, 3):
        raise ValueError(f"n_accel must be 2 or 3, got {n_accel}")

    # Validate consistency between n_accel and vel_idx
    if n_accel == 3 and len(vel_idx) != 3:
        raise ValueError(f"n_accel=3 requires 3D velocity (len(vel_idx)=3), got {len(vel_idx)}")
    if n_accel == 2 and len(vel_idx) not in (2, 3):
        raise ValueError(f"n_accel=2 requires 2D velocity, got len(vel_idx)={len(vel_idx)}")

    G = jnp.zeros((n, n_accel + 1), dtype=dtype)
    theta_arr = jnp.asarray(theta, dtype=dtype)
    dt_arr = jnp.asarray(dt, dtype=dtype)
    c, s = jnp.cos(theta_arr), jnp.sin(theta_arr)
    R_2d = jnp.array([[c, -s], [s, c]], dtype=dtype)

    # Heading: ∂θ/∂ω_z = dt
    if 0 <= theta_idx < n:
        G = G.at[theta_idx, 0].set(dt_arr)

    # Velocity (x, y components affected by 2D rotated accel)
    if len(vel_idx) >= 2:
        vx_i, vy_i = vel_idx[0], vel_idx[1]
        if 0 <= vx_i < n and 0 <= vy_i < n:
            G = G.at[vx_i : vy_i + 1, 1:3].set(R_2d * dt_arr)

    # Velocity (z component, if present, affected directly by f_z)
    if n_accel == 3 and len(vel_idx) == 3:
        vz_i = vel_idx[2]
        if 0 <= vz_i < n:
            # ∂vz/∂f_z = dt (no rotation, since yaw-only model preserves z)
            # R_z(θ) rotates only in x-y plane; world z ≈ body z
            G = G.at[vz_i, 3].set(dt_arr)

    # Position (only x, y components affected, since we track 2D position)
    px_i, py_i = pos_idx
    if 0 <= px_i < n and 0 <= py_i < n:
        G = G.at[px_i : py_i + 1, 1:3].set(R_2d * (0.5 * dt_arr * dt_arr))

    return G
