"""Standalone orientation estimation from full 3-axis IMU data.

This module is intentionally independent of the position filters. It estimates
body-to-world orientation from gyroscope propagation plus gated gravity and
optional camera-heading corrections, without using accelerometer samples as
translational acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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

    n_time = t_arr.shape[0]
    quats = np.zeros((n_time, 4), dtype=float)
    gyro_biases = np.zeros((n_time, 3), dtype=float)
    camera_yaw_update_mask = np.zeros(n_time, dtype=bool)

    quats[0] = q
    gyro_biases[0] = gyro_bias
    world_down = np.array([0.0, 0.0, -1.0])

    for idx in range(1, n_time):
        dt = float(t_arr[idx] - t_arr[idx - 1])
        omega = gyro_arr[idx - 1] - gyro_bias
        q = _integrate_body_gyro(q, omega, dt)

        if gravity_mask[idx]:
            measured_down = accel_arr[idx] / np.linalg.norm(accel_arr[idx])
            predicted_down = _rotate_world_to_body(q, world_down)
            error_body = np.cross(measured_down, predicted_down)
            omega_correction = cfg.accel_correction_gain * error_body
            gyro_bias -= cfg.accel_bias_correction_gain * error_body * dt
            q = _integrate_body_gyro(q, omega_correction, dt)

        if camera.heading is not None and camera.valid_heading[idx]:
            yaw = _quaternion_to_yaw(q)
            yaw_error = _wrap_angle(camera.heading[idx] - yaw)
            yaw_delta = cfg.camera_yaw_correction_gain * yaw_error
            q = _quaternion_multiply(
                _quaternion_from_rotation_vector(np.array([0.0, 0.0, yaw_delta])),
                q,
            )
            q = _normalize_quaternion(q)
            gyro_bias[2] -= cfg.camera_yaw_bias_correction_gain * yaw_error * dt
            camera_yaw_update_mask[idx] = True

        quats[idx] = q
        gyro_biases[idx] = gyro_bias

    roll, pitch, yaw = _quaternion_to_roll_pitch_yaw(quats)
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
    if not np.all(np.diff(t_arr) > 0):
        raise ValueError("t_imu must be strictly increasing.")
    if not np.isfinite(t_arr).all():
        raise ValueError("t_imu contains non-finite values.")
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
        return _normalize_quaternion(quat)

    finite = gravity_mask & np.isfinite(accel_xyz).all(axis=1)
    if not np.any(finite):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    measured_down = np.median(accel_xyz[finite], axis=0)
    measured_down = measured_down / np.linalg.norm(measured_down)
    return _quaternion_align_vectors(
        source=measured_down,
        target=np.array([0.0, 0.0, -1.0]),
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


def _integrate_body_gyro(
    quat_body_to_world: NDArray[np.float64],
    omega_body: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    delta_q = _quaternion_from_rotation_vector(np.asarray(omega_body) * dt)
    return _normalize_quaternion(_quaternion_multiply(quat_body_to_world, delta_q))


def _normalize_quaternion(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    quat_arr = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat_arr)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    normalized = quat_arr / norm
    return normalized if normalized[0] >= 0 else -normalized


def _quaternion_multiply(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quaternion_from_rotation_vector(
    rotvec: NDArray[np.float64],
) -> NDArray[np.float64]:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return _normalize_quaternion(np.concatenate([[1.0], 0.5 * rotvec]))
    axis = rotvec / angle
    half_angle = 0.5 * angle
    return _normalize_quaternion(
        np.concatenate([[np.cos(half_angle)], axis * np.sin(half_angle)])
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
        return _quaternion_from_rotation_vector(np.pi * axis)
    axis = np.cross(source_unit, target_unit)
    quat = np.concatenate([[1.0 + dot], axis])
    return _normalize_quaternion(quat)


def _rotate_vector_body_to_world(
    quat_body_to_world: NDArray[np.float64],
    vector_body: NDArray[np.float64],
) -> NDArray[np.float64]:
    q = _normalize_quaternion(quat_body_to_world)
    vector_quat = np.concatenate([[0.0], np.asarray(vector_body, dtype=float)])
    rotated = _quaternion_multiply(
        _quaternion_multiply(q, vector_quat),
        np.array([q[0], -q[1], -q[2], -q[3]]),
    )
    return rotated[1:]


def _rotate_world_to_body(
    quat_body_to_world: NDArray[np.float64],
    vector_world: NDArray[np.float64],
) -> NDArray[np.float64]:
    q = _normalize_quaternion(quat_body_to_world)
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    return _rotate_vector_body_to_world(q_inv, vector_world)


def _quaternion_to_yaw(quat_body_to_world: NDArray[np.float64]) -> float:
    q = _normalize_quaternion(quat_body_to_world)
    qw, qx, qy, qz = q
    return float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )


def _quaternion_to_roll_pitch_yaw(
    quat_body_to_world: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    q = np.asarray(quat_body_to_world, dtype=float)
    qw, qx, qy, qz = np.moveaxis(q, -1, 0)
    roll = np.arctan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return roll, pitch, yaw


def _wrap_angle(angle: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
