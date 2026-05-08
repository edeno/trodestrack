"""Standalone orientation estimation from full 3-axis IMU data.

This module is intentionally independent of the position filters. It estimates
body-to-world orientation from gyroscope propagation plus gated gravity and
optional camera-heading corrections, without using accelerometer samples as
translational acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from numpy.typing import NDArray

from trodestrack.models.quaternion import (
    integrate_body_gyro,
    normalize_quaternion,
    quaternion_from_rotation_vector,
    quaternion_multiply,
    quaternion_to_roll_pitch_yaw,
    quaternion_to_yaw,
    rotate_vector_world_to_body,
)


@dataclass(frozen=True)
class OrientationEstimatorConfig:
    """Configuration for the offline orientation-only estimator.

    Parameters
    ----------
    gravity_m_s2 : float, default 9.80665
        Expected gravity magnitude.
    accel_magnitude_tolerance_m_s2 : float, default 0.5
        Gravity update is allowed when ``abs(||accel|| - gravity)`` is below
        this threshold.
    gyro_norm_threshold_rad_s : float, default 0.2
        Gravity update is allowed when raw gyro norm is below this threshold.
    camera_speed_threshold_m_s : float | None, default None
        Optional camera-speed gate for gravity updates. Requires camera inputs.
    accel_correction_gain : float, default 2.0
        Proportional gravity-direction correction gain in rad/s.
    accel_bias_correction_gain : float, default 0.02
        Integral correction gain for roll/pitch gyro bias.
    camera_yaw_correction_gain : float, default 0.2
        Fraction of LED-heading yaw error applied at each camera-updated IMU
        sample.
    camera_yaw_bias_correction_gain : float, default 0.02
        Integral correction gain for yaw gyro bias from LED-heading error.
    initial_gyro_bias_rad_s : NDArray | None, default None
        Optional initial gyro bias, shape ``(3,)``. If omitted, the estimator
        starts from zero bias to avoid treating real slow rotations as bias.
    initial_quaternion : NDArray | None, default None
        Optional initial body-to-world quaternion in scalar-first
        ``[qw, qx, qy, qz]`` order.
    """

    gravity_m_s2: float = 9.80665
    accel_magnitude_tolerance_m_s2: float = 0.5
    gyro_norm_threshold_rad_s: float = 0.2
    camera_speed_threshold_m_s: float | None = None
    accel_correction_gain: float = 2.0
    accel_bias_correction_gain: float = 0.02
    camera_yaw_correction_gain: float = 0.2
    camera_yaw_bias_correction_gain: float = 0.02
    initial_gyro_bias_rad_s: NDArray[np.floating] | None = None
    initial_quaternion: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Reject configs that would silently propagate NaN or invert physics.

        Mirrors the validation pattern in ``FilterCoreConfig.__post_init__``.
        Without these guards, a NaN ``initial_gyro_bias_rad_s`` would
        produce a NaN final bias array, and a negative ``gravity_m_s2`` /
        ``accel_correction_gain`` would invert the sign of the gravity
        correction step.
        """
        positive_fields = ("gravity_m_s2",)
        for fname in positive_fields:
            value = getattr(self, fname)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{fname} must be a finite strictly-positive value "
                    f"(physical magnitude); got {value!r}."
                )

        # Gains are non-negative: 0 is a valid "disable this correction"
        # value used by tests, but negative or NaN gains would invert the
        # sign of the correction step.
        non_negative_fields = (
            "accel_magnitude_tolerance_m_s2",
            "gyro_norm_threshold_rad_s",
            "accel_correction_gain",
            "accel_bias_correction_gain",
            "camera_yaw_correction_gain",
            "camera_yaw_bias_correction_gain",
        )
        for fname in non_negative_fields:
            value = getattr(self, fname)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{fname} must be a finite non-negative value; got {value!r}."
                )

        if self.camera_speed_threshold_m_s is not None:
            v = self.camera_speed_threshold_m_s
            if not np.isfinite(v) or v < 0:
                raise ValueError(
                    "camera_speed_threshold_m_s must be None or a finite "
                    f"non-negative value; got {v!r}."
                )

        if self.initial_gyro_bias_rad_s is not None:
            arr = np.asarray(self.initial_gyro_bias_rad_s, dtype=float)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                raise ValueError(
                    "initial_gyro_bias_rad_s must be a length-3 finite "
                    f"sequence in rad/s; got {self.initial_gyro_bias_rad_s!r}."
                )

        if self.initial_quaternion is not None:
            q = np.asarray(self.initial_quaternion, dtype=float)
            if q.shape != (4,) or not np.all(np.isfinite(q)):
                raise ValueError(
                    "initial_quaternion must be a length-4 finite "
                    "scalar-first [qw, qx, qy, qz] sequence; got "
                    f"{self.initial_quaternion!r}."
                )
            qnorm = float(np.linalg.norm(q))
            if not (0.5 < qnorm < 2.0):
                raise ValueError(
                    "initial_quaternion must be near unit norm (||q|| in "
                    f"(0.5, 2.0)); got ||q||={qnorm:.3e}."
                )


@dataclass(frozen=True)
class OrientationDiagnostics:
    """Summary diagnostics for an orientation estimate."""

    quaternion_norm_min: float
    quaternion_norm_max: float
    roll_range_rad: tuple[float, float]
    pitch_range_rad: tuple[float, float]
    yaw_range_rad: tuple[float, float]
    gravity_update_fraction: float
    camera_yaw_update_fraction: float
    yaw_camera_rmse_rad: float | None


@dataclass(frozen=True)
class OrientationEstimate:
    """Orientation estimate and update diagnostics."""

    t_imu: NDArray[np.float64]
    quaternions: NDArray[np.float64]
    roll: NDArray[np.float64]
    pitch: NDArray[np.float64]
    yaw: NDArray[np.float64]
    gyro_bias_rad_s: NDArray[np.float64]
    gravity_update_mask: NDArray[np.bool_]
    camera_yaw_update_mask: NDArray[np.bool_]
    diagnostics: OrientationDiagnostics


def format_orientation_diagnostics(result: OrientationEstimate) -> str:
    """Format orientation estimator diagnostics for console output."""

    diag = result.diagnostics
    yaw_rmse = (
        "n/a"
        if diag.yaw_camera_rmse_rad is None
        else f"{diag.yaw_camera_rmse_rad:.4f} rad"
    )
    final_bias = result.gyro_bias_rad_s[-1]
    lines = [
        "Orientation-only IMU estimate",
        "=" * 80,
        "quaternion_norm: "
        f"min={diag.quaternion_norm_min:.8f}, max={diag.quaternion_norm_max:.8f}",
        "roll_range_rad: "
        f"[{diag.roll_range_rad[0]:+.4f}, {diag.roll_range_rad[1]:+.4f}]",
        "pitch_range_rad: "
        f"[{diag.pitch_range_rad[0]:+.4f}, {diag.pitch_range_rad[1]:+.4f}]",
        f"yaw_range_rad: [{diag.yaw_range_rad[0]:+.4f}, {diag.yaw_range_rad[1]:+.4f}]",
        f"gravity_update_fraction: {diag.gravity_update_fraction:.3f}",
        f"camera_yaw_update_fraction: {diag.camera_yaw_update_fraction:.3f}",
        f"yaw_camera_rmse: {yaw_rmse}",
        "final_gyro_bias_rad_s: "
        f"[{final_bias[0]:+.6f}, {final_bias[1]:+.6f}, {final_bias[2]:+.6f}]",
    ]
    return "\n".join(lines)


def estimate_orientation(
    *,
    t_imu: NDArray[np.floating],
    gyro_xyz: NDArray[np.floating],
    accel_xyz: NDArray[np.floating],
    config: OrientationEstimatorConfig | None = None,
    t_cam: NDArray[np.floating] | None = None,
    led1: NDArray[np.floating] | None = None,
    led2: NDArray[np.floating] | None = None,
) -> OrientationEstimate:
    """Estimate orientation from full IMU gyro/accelerometer channels.

    Parameters
    ----------
    t_imu : NDArray
        IMU timestamps in seconds, shape ``(n_time,)``.
    gyro_xyz : NDArray
        Body-frame angular rates in rad/s, shape ``(n_time, 3)``.
    accel_xyz : NDArray
        Body-frame accelerometer samples in m/s², shape ``(n_time, 3)``.
    config : OrientationEstimatorConfig | None, optional
        Estimator configuration.
    t_cam, led1, led2 : optional
        Camera timestamps and two 2D LED positions. When provided, LED heading
        supplies an optional yaw correction and camera speed can gate gravity
        updates if ``camera_speed_threshold_m_s`` is set.

    Returns
    -------
    OrientationEstimate
        Quaternions, Euler angles, estimated gyro bias, and update diagnostics.
    """

    cfg = config or OrientationEstimatorConfig()
    t_arr, gyro_arr, accel_arr = _validate_imu_inputs(t_imu, gyro_xyz, accel_xyz)
    camera = _prepare_camera_inputs(t_arr, t_cam, led1, led2)

    gravity_mask = _gravity_update_mask(t_arr, gyro_arr, accel_arr, cfg, camera.speed)
    gyro_bias = _initial_gyro_bias(cfg)
    q = _initial_quaternion(accel_arr, gravity_mask, cfg)

    camera_heading = (
        np.zeros(t_arr.shape[0], dtype=float)
        if camera.heading is None
        else np.asarray(camera.heading, dtype=float)
    )
    quats, gyro_biases, camera_yaw_update_mask = _orientation_scan_core_jit(
        jnp.asarray(t_arr),
        jnp.asarray(gyro_arr),
        jnp.asarray(accel_arr),
        jnp.asarray(gravity_mask),
        jnp.asarray(camera_heading),
        jnp.asarray(camera.valid_heading),
        jnp.asarray(q),
        jnp.asarray(gyro_bias),
        jnp.asarray(cfg.accel_correction_gain),
        jnp.asarray(cfg.accel_bias_correction_gain),
        jnp.asarray(cfg.camera_yaw_correction_gain),
        jnp.asarray(cfg.camera_yaw_bias_correction_gain),
    )

    quats = np.asarray(quats, dtype=float)
    gyro_biases = np.asarray(gyro_biases, dtype=float)
    camera_yaw_update_mask = np.asarray(camera_yaw_update_mask, dtype=bool)
    roll_jax, pitch_jax, yaw_jax = quaternion_to_roll_pitch_yaw(jnp.asarray(quats))
    roll = np.asarray(roll_jax, dtype=float)
    pitch = np.asarray(pitch_jax, dtype=float)
    yaw = np.asarray(yaw_jax, dtype=float)
    diagnostics = _orientation_diagnostics(
        quats,
        roll,
        pitch,
        yaw,
        gravity_mask,
        camera_yaw_update_mask,
        camera.heading,
        camera.valid_heading,
    )
    return OrientationEstimate(
        t_imu=t_arr,
        quaternions=quats,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        gyro_bias_rad_s=gyro_biases,
        gravity_update_mask=gravity_mask,
        camera_yaw_update_mask=camera_yaw_update_mask,
        diagnostics=diagnostics,
    )


def _canonicalize_quaternion(quat: jnp.ndarray) -> jnp.ndarray:
    q = normalize_quaternion(quat)
    return jnp.where(q[0] >= 0.0, q, -q)


def _wrap_angle_jax(angle: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _orientation_scan_core(
    t_imu: jnp.ndarray,
    gyro_xyz: jnp.ndarray,
    accel_xyz: jnp.ndarray,
    gravity_update_mask: jnp.ndarray,
    camera_heading: jnp.ndarray,
    valid_camera_heading: jnp.ndarray,
    initial_quaternion: jnp.ndarray,
    initial_gyro_bias: jnp.ndarray,
    accel_correction_gain: jnp.ndarray,
    accel_bias_correction_gain: jnp.ndarray,
    camera_yaw_correction_gain: jnp.ndarray,
    camera_yaw_bias_correction_gain: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the orientation estimator recurrence with ``lax.scan``."""

    # Stationary accelerometer specific force points along +z in the world
    # frame (sensor reads +g at rest with level mounting), matching
    # imu_gravity_body=(0, 0, +g) used by the EKF dynamics path.
    gravity_unit_world = jnp.array([0.0, 0.0, 1.0], dtype=t_imu.dtype)
    q0 = _canonicalize_quaternion(initial_quaternion.astype(t_imu.dtype))
    bias0 = initial_gyro_bias.astype(t_imu.dtype)

    def step(carry, inputs):
        q, gyro_bias = carry
        dt, gyro_prev, accel_now, use_gravity, heading_now, use_camera = inputs

        omega = gyro_prev - gyro_bias
        q_pred = _canonicalize_quaternion(integrate_body_gyro(q, omega, dt))

        def apply_gravity(state):
            q_in, bias_in = state
            accel_norm = jnp.linalg.norm(accel_now)
            measured_gravity_body = accel_now / jnp.maximum(
                accel_norm,
                jnp.asarray(1e-12, dtype=accel_now.dtype),
            )
            predicted_gravity_body = rotate_vector_world_to_body(
                q_in, gravity_unit_world
            )
            error_body = jnp.cross(measured_gravity_body, predicted_gravity_body)
            omega_correction = accel_correction_gain * error_body
            bias_out = bias_in - accel_bias_correction_gain * error_body * dt
            q_out = _canonicalize_quaternion(
                integrate_body_gyro(q_in, omega_correction, dt)
            )
            return q_out, bias_out

        q_gravity, bias_gravity = lax.cond(
            use_gravity,
            apply_gravity,
            lambda state: state,
            (q_pred, gyro_bias),
        )

        def apply_camera(state):
            q_in, bias_in = state
            yaw = quaternion_to_yaw(q_in)
            yaw_error = _wrap_angle_jax(heading_now - yaw)
            yaw_delta = camera_yaw_correction_gain * yaw_error
            delta_q = quaternion_from_rotation_vector(
                jnp.array([0.0, 0.0, yaw_delta], dtype=q_in.dtype)
            )
            q_out = _canonicalize_quaternion(quaternion_multiply(delta_q, q_in))
            bias_out = bias_in.at[2].add(
                -camera_yaw_bias_correction_gain * yaw_error * dt
            )
            return q_out, bias_out

        q_next, bias_next = lax.cond(
            use_camera,
            apply_camera,
            lambda state: state,
            (q_gravity, bias_gravity),
        )
        return (q_next, bias_next), (q_next, bias_next, use_camera)

    scan_inputs = (
        jnp.diff(t_imu),
        gyro_xyz[:-1],
        accel_xyz[1:],
        gravity_update_mask[1:],
        camera_heading[1:],
        valid_camera_heading[1:],
    )
    (_, _), (quats_tail, biases_tail, camera_tail) = lax.scan(
        step,
        (q0, bias0),
        scan_inputs,
    )
    quats = jnp.concatenate([q0[None, :], quats_tail], axis=0)
    biases = jnp.concatenate([bias0[None, :], biases_tail], axis=0)
    camera_updates = jnp.concatenate(
        [jnp.array([False], dtype=bool), camera_tail.astype(bool)],
        axis=0,
    )
    return quats, biases, camera_updates


_orientation_scan_core_jit = jax.jit(_orientation_scan_core)


@dataclass(frozen=True)
class _CameraInputs:
    heading: NDArray[np.float64] | None
    valid_heading: NDArray[np.bool_]
    speed: NDArray[np.float64] | None


def _validate_imu_inputs(
    t_imu: NDArray[np.floating],
    gyro_xyz: NDArray[np.floating],
    accel_xyz: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    t_arr = np.asarray(t_imu, dtype=float)
    gyro_arr = np.asarray(gyro_xyz, dtype=float)
    accel_arr = np.asarray(accel_xyz, dtype=float)
    if t_arr.ndim != 1:
        raise ValueError(f"t_imu must be 1D; got shape {t_arr.shape}.")
    if gyro_arr.shape != (t_arr.shape[0], 3):
        raise ValueError(
            f"gyro_xyz must have shape ({t_arr.shape[0]}, 3); got {gyro_arr.shape}."
        )
    if accel_arr.shape != (t_arr.shape[0], 3):
        raise ValueError(
            f"accel_xyz must have shape ({t_arr.shape[0]}, 3); got {accel_arr.shape}."
        )
    if t_arr.shape[0] < 2:
        raise ValueError("At least two IMU samples are required.")
    if not np.isfinite(t_arr).all():
        raise ValueError("t_imu contains non-finite values.")
    if not np.all(np.diff(t_arr) > 0):
        raise ValueError("t_imu must be strictly increasing.")

    # Reject non-finite IMU samples at the public boundary. The scan
    # propagates each gyro_prev through integrate_body_gyro, where a
    # NaN/inf gyro vector produces a non-finite quaternion that
    # normalize_quaternion silently maps back to identity (norm > eps is
    # False for NaN), hiding a corrupted sample as a plausible no-rotation
    # step. The troubleshooting docs already require finite IMU samples,
    # so enforce it here.
    if not np.isfinite(gyro_arr).all():
        n_bad = int(np.sum(~np.isfinite(gyro_arr).all(axis=1)))
        raise ValueError(
            f"gyro_xyz contains non-finite value(s) (NaN/inf) in {n_bad} "
            "row(s); IMU samples must be finite (rad/s)."
        )
    if not np.isfinite(accel_arr).all():
        n_bad = int(np.sum(~np.isfinite(accel_arr).all(axis=1)))
        raise ValueError(
            f"accel_xyz contains non-finite value(s) (NaN/inf) in {n_bad} "
            "row(s); IMU samples must be finite (m/s²)."
        )
    return t_arr, gyro_arr, accel_arr


def _prepare_camera_inputs(
    t_imu: NDArray[np.float64],
    t_cam: NDArray[np.floating] | None,
    led1: NDArray[np.floating] | None,
    led2: NDArray[np.floating] | None,
) -> _CameraInputs:
    if t_cam is None and led1 is None and led2 is None:
        return _CameraInputs(
            heading=None,
            valid_heading=np.zeros(t_imu.shape[0], dtype=bool),
            speed=None,
        )
    if t_cam is None or led1 is None or led2 is None:
        raise ValueError("t_cam, led1, and led2 must be provided together.")

    t_cam_arr = np.asarray(t_cam, dtype=float)
    led1_arr = np.asarray(led1, dtype=float)
    led2_arr = np.asarray(led2, dtype=float)
    if t_cam_arr.ndim != 1:
        raise ValueError(f"t_cam must be 1D; got shape {t_cam_arr.shape}.")
    if led1_arr.shape != (t_cam_arr.shape[0], 2):
        raise ValueError(
            f"led1 must have shape ({t_cam_arr.shape[0]}, 2); got {led1_arr.shape}."
        )
    if led2_arr.shape != (t_cam_arr.shape[0], 2):
        raise ValueError(
            f"led2 must have shape ({t_cam_arr.shape[0]}, 2); got {led2_arr.shape}."
        )
    # Reject non-finite timestamps before the strict-increase check —
    # ``np.diff`` of [..., +inf] is NaN, which fails ``> 0`` and would
    # mask the real problem; an array like [0.0, +inf] also passes the
    # diff-positive check but propagates inf into the np.interp /
    # np.searchsorted paths below.
    if not np.all(np.isfinite(t_cam_arr)):
        n_bad = int(np.sum(~np.isfinite(t_cam_arr)))
        raise ValueError(
            f"t_cam contains {n_bad} non-finite value(s) (NaN/inf); "
            "timestamps must be finite seconds."
        )
    if not np.all(np.diff(t_cam_arr) > 0):
        raise ValueError("t_cam must be strictly increasing.")

    valid_cam = np.isfinite(led1_arr).all(axis=1) & np.isfinite(led2_arr).all(axis=1)
    if np.sum(valid_cam) < 2:
        raise ValueError("At least two finite dual-LED camera samples are required.")

    camera_index = np.arange(t_cam_arr.shape[0], dtype=float)
    led_delta = led2_arr - led1_arr
    heading_cam = np.unwrap(
        np.arctan2(led_delta[valid_cam, 1], led_delta[valid_cam, 0])
    )
    heading_imu = np.full(t_imu.shape[0], np.nan, dtype=float)
    valid_heading = np.zeros(t_imu.shape[0], dtype=bool)
    update_indices = np.searchsorted(t_imu, t_cam_arr[valid_cam], side="left")
    in_bounds = update_indices < t_imu.shape[0]
    heading_imu[update_indices[in_bounds]] = heading_cam[in_bounds]
    valid_heading[update_indices[in_bounds]] = True

    midpoint = 0.5 * (led1_arr + led2_arr)
    midpoint_filled = np.column_stack(
        [
            np.interp(camera_index, camera_index[valid_cam], midpoint[valid_cam, col])
            for col in range(2)
        ]
    )
    edge_order = 2 if t_cam_arr.shape[0] >= 3 else 1
    velocity = np.gradient(midpoint_filled, t_cam_arr, axis=0, edge_order=edge_order)
    speed = np.linalg.norm(velocity, axis=1)
    speed_imu = np.interp(t_imu, t_cam_arr, speed)

    return _CameraInputs(
        heading=heading_imu,
        valid_heading=valid_heading,
        speed=speed_imu,
    )


def _gravity_update_mask(
    t_imu: NDArray[np.float64],
    gyro_xyz: NDArray[np.float64],
    accel_xyz: NDArray[np.float64],
    cfg: OrientationEstimatorConfig,
    camera_speed: NDArray[np.float64] | None,
) -> NDArray[np.bool_]:
    del t_imu
    accel_norm = np.linalg.norm(accel_xyz, axis=1)
    gyro_norm = np.linalg.norm(gyro_xyz, axis=1)
    mask = (
        np.isfinite(gyro_xyz).all(axis=1)
        & np.isfinite(accel_xyz).all(axis=1)
        & (np.abs(accel_norm - cfg.gravity_m_s2) <= cfg.accel_magnitude_tolerance_m_s2)
        & (gyro_norm <= cfg.gyro_norm_threshold_rad_s)
        & (accel_norm > 0.0)
    )
    if cfg.camera_speed_threshold_m_s is not None:
        if camera_speed is None:
            raise ValueError(
                "camera_speed_threshold_m_s requires t_cam, led1, and led2 inputs."
            )
        mask &= np.isfinite(camera_speed) & (
            camera_speed <= cfg.camera_speed_threshold_m_s
        )
    return mask


def _initial_gyro_bias(cfg: OrientationEstimatorConfig) -> NDArray[np.float64]:
    if cfg.initial_gyro_bias_rad_s is not None:
        bias = np.asarray(cfg.initial_gyro_bias_rad_s, dtype=float)
        if bias.shape != (3,):
            raise ValueError(
                f"initial_gyro_bias_rad_s must have shape (3,); got {bias.shape}."
            )
        return bias.copy()
    return np.zeros(3, dtype=float)


def _initial_quaternion(
    accel_xyz: NDArray[np.float64],
    gravity_mask: NDArray[np.bool_],
    cfg: OrientationEstimatorConfig,
) -> NDArray[np.float64]:
    if cfg.initial_quaternion is not None:
        quat = np.asarray(cfg.initial_quaternion, dtype=float)
        if quat.shape != (4,):
            raise ValueError(
                f"initial_quaternion must have shape (4,); got {quat.shape}."
            )
        return _canonicalize_quaternion_np(quat)

    finite = gravity_mask & np.isfinite(accel_xyz).all(axis=1)
    if not np.any(finite):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    measured_gravity_body = np.median(accel_xyz[finite], axis=0)
    measured_gravity_body = measured_gravity_body / np.linalg.norm(
        measured_gravity_body
    )
    return _quaternion_align_vectors(
        source=measured_gravity_body,
        target=np.array([0.0, 0.0, 1.0]),
    )


def _orientation_diagnostics(
    quaternions: NDArray[np.float64],
    roll: NDArray[np.float64],
    pitch: NDArray[np.float64],
    yaw: NDArray[np.float64],
    gravity_update_mask: NDArray[np.bool_],
    camera_yaw_update_mask: NDArray[np.bool_],
    camera_heading: NDArray[np.float64] | None,
    valid_camera_heading: NDArray[np.bool_],
) -> OrientationDiagnostics:
    quat_norm = np.linalg.norm(quaternions, axis=1)
    yaw_rmse = None
    if camera_heading is not None:
        valid = valid_camera_heading & np.isfinite(camera_heading)
        if np.any(valid):
            yaw_error = _wrap_angle(yaw[valid] - camera_heading[valid])
            yaw_rmse = float(np.sqrt(np.mean(yaw_error**2)))
    return OrientationDiagnostics(
        quaternion_norm_min=float(np.min(quat_norm)),
        quaternion_norm_max=float(np.max(quat_norm)),
        roll_range_rad=(float(np.min(roll)), float(np.max(roll))),
        pitch_range_rad=(float(np.min(pitch)), float(np.max(pitch))),
        yaw_range_rad=(float(np.min(yaw)), float(np.max(yaw))),
        gravity_update_fraction=float(np.mean(gravity_update_mask)),
        camera_yaw_update_fraction=float(np.mean(camera_yaw_update_mask)),
        yaw_camera_rmse_rad=yaw_rmse,
    )


def _canonicalize_quaternion_np(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        _canonicalize_quaternion(jnp.asarray(quat, dtype=jnp.float32)),
        dtype=float,
    )


def _quaternion_align_vectors(
    *,
    source: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    source_unit = np.asarray(source, dtype=float)
    source_unit = source_unit / np.linalg.norm(source_unit)
    target_unit = np.asarray(target, dtype=float)
    target_unit = target_unit / np.linalg.norm(target_unit)
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if dot < -1.0 + 1e-10:
        axis = np.cross(source_unit, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(source_unit, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return _canonicalize_quaternion_np(
            np.asarray(quaternion_from_rotation_vector(jnp.asarray(np.pi * axis)))
        )
    axis = np.cross(source_unit, target_unit)
    quat = np.concatenate([[1.0 + dot], axis])
    return _canonicalize_quaternion_np(quat)


def _wrap_angle(angle: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    return np.arctan2(np.sin(angle), np.cos(angle))
