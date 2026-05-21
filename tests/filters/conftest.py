"""Shared fixtures for filter-level tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pytest

from trodestrack.models.quaternion import (
    rotate_vector_body_to_world,
    rotate_vector_world_to_body,
)
from trodestrack.sim.simple import SimpleSimConfig, simulate_constant_velocity


@pytest.fixture
def event_channel_sim():
    """Constant-velocity sim used by both EKF and UKF event-channel tests."""
    cfg = SimpleSimConfig(
        duration_s=2.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.0005,
        accel_noise_density=0.02,
        gyro_bias_std=0.01,
        accel_bias_std=0.02,
        cam_noise_std=0.005,
        cam_dropout_prob=0.0,
    )
    return simulate_constant_velocity(cfg, velocity=np.array([0.2, 0.0]))


@pytest.fixture
def event_channel_filter_defaults():
    """Filter-config kwargs shared by EKF and UKF event-channel tests.

    Only filter-agnostic tunings (no ``num_iter``); both ``EKFConfig`` and
    ``UKFConfig`` accept this dict via ``**kwargs``.
    """
    return {
        "state_mode": "2d_full",
        "process_noise_pos": 0.001,
        "process_noise_vel": 0.5,
        "process_noise_heading": 0.5,
        "process_noise_gyro_bias": 7.6e-7,
        "process_noise_accel_bias": 2.4e-9,
        "measurement_noise_pos": 0.005**2,
        "measurement_noise_heading": 0.5**2,
        "led_distance": 0.04,
        "use_heading_measurement": True,
        "damping_coeff": 0.4,
    }


@pytest.fixture
def run_filter_baseline():
    """Return a closure that runs a filter with no event arguments."""

    def _run(filter_fn, filter_config, sim):
        return filter_fn(
            filter_config,
            sim["t_imu"],
            sim["U_imu"],
            sim["t_cam_exp"],
            sim["Z_cam_led1"],
            sim["Z_cam_led2"],
            sim["mask_cam"],
        )

    return _run


@pytest.fixture
def run_filter_with_events():
    """Return a closure that runs a filter with the given event arrays."""

    def _run(filter_fn, filter_config, sim, anchors, covariances, indices_per_frame):
        return filter_fn(
            filter_config,
            sim["t_imu"],
            sim["U_imu"],
            sim["t_cam_exp"],
            sim["Z_cam_led1"],
            sim["Z_cam_led2"],
            sim["mask_cam"],
            event_source_anchors=anchors,
            event_source_covariances=covariances,
            event_indices_per_frame=indices_per_frame,
        )

    return _run


# =============================================================================
# 3D session simulator (shared by 3D EKF / UKF analytic tests)
# =============================================================================
#
# The 2D ``simulate_rat_imu`` helper synthesizes a planar yaw-only rat
# trajectory and only emits 3-channel IMU + 2D LED positions, which doesn't
# exercise the 6-DOF orientation states or 3D camera measurements used by
# the experimental 3D EKF entry point. ``simulate_3d_session`` below builds
# a deterministic 3D ground-truth trajectory, computes the body-frame
# 6-DOF IMU readings by differentiating the trajectory and rotating with
# the truth quaternion, and projects the LEDs to a 3D camera with simple
# additive noise. The helper is intentionally module-level (no leading
# underscore) because Phase 6 imports it as the parity oracle for the
# ``lax.scan`` refactor of ``extended_kalman_filter_3d``.


GRAVITY_W: np.ndarray = np.array([0.0, 0.0, 9.81], dtype=np.float64)


class SimSession(NamedTuple):
    """3D ground-truth trajectory plus 6-DOF IMU and 3D LED observations.

    Shapes
    ------
    t_imu : (N_imu,)
    U_imu : (N_imu, 6)
        [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z] in body frame
        (accel is specific force a_body - g_body, matching the
        ``extended_kalman_filter_3d`` contract).
    t_cam : (N_cam,)
    Z_cam_led1, Z_cam_led2 : (N_cam, 3)
        World-frame LED positions with isotropic Gaussian noise. NaN at
        dropped-out frames.
    mask_cam : (N_cam,)
        Per-frame validity. False during ``dropout_window_s`` if set.
    truth : dict
        ``positions`` (N_imu, 3), ``quaternions`` (N_imu, 4),
        ``gyro_bias`` (3,), ``accel_bias`` (3,).
    """

    t_imu: np.ndarray
    U_imu: np.ndarray
    t_cam: np.ndarray
    Z_cam_led1: np.ndarray
    Z_cam_led2: np.ndarray
    mask_cam: np.ndarray
    truth: dict


MotionMode = Literal[
    "stationary",
    "yaw_only",
    "circular_with_pitch",
    "perfect",
]


def _quaternion_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a body-to-world quaternion from ZYX Euler angles (roll, pitch, yaw)."""

    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return q / np.linalg.norm(q)


@pytest.fixture
def simulate_3d() -> Callable[..., SimSession]:
    """Fixture handle to ``simulate_3d_session`` for the 3D analytic tests.

    Test files use ``simulate_3d(...)`` directly. Phase 6 imports the
    underlying function as ``simulate_3d_session`` for its parity oracle.
    """

    return simulate_3d_session


def simulate_3d_session(
    seed: int,
    duration_s: float = 30.0,
    fs_imu: float = 100.0,
    fs_cam: float = 30.0,
    motion: MotionMode = "stationary",
    init_roll_deg: float = 0.0,
    init_pitch_deg: float = 0.0,
    init_yaw_deg: float = 0.0,
    yaw_rate_dps: float = 0.0,
    pitch_amp_deg: float = 0.0,
    pitch_freq_hz: float = 0.0,
    radius_m: float = 0.0,
    gyro_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    accel_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gyro_noise_std: float = 1e-3,
    accel_noise_std: float = 1e-2,
    cam_noise_std: float = 5e-3,
    dropout_window_s: tuple[float, float] | None = None,
    led_offsets_body: np.ndarray | None = None,
    initial_position: np.ndarray | None = None,
) -> SimSession:
    """Deterministic 3D rat-headstage session for 3D filter analytic tests.

    Generates a ground-truth trajectory (position + orientation), computes
    the body-frame IMU readings that an idealized gyro/accel would record,
    and projects two LEDs into a 3D camera frame with additive isotropic
    Gaussian noise.

    Parameters
    ----------
    seed : int
        Deterministic RNG seed for noise draws.
    duration_s, fs_imu, fs_cam
        Session length and sampling rates.
    motion : MotionMode
        - ``"stationary"``: position fixed; orientation locked to the
          configured ``init_{roll,pitch,yaw}_deg``.
        - ``"yaw_only"``: stationary position; yaw rotates at
          ``yaw_rate_dps`` deg/s about world z.
        - ``"circular_with_pitch"``: circular motion of radius
          ``radius_m`` in the world xy plane at ``yaw_rate_dps``; pitch
          oscillates at ``pitch_freq_hz`` with amplitude
          ``pitch_amp_deg``.
        - ``"perfect"``: same as stationary but with all noise sources
          forced to zero (used by the perfect-input idempotence test).
    init_{roll,pitch,yaw}_deg : float
        Initial orientation in degrees.
    yaw_rate_dps : float
        World-z yaw rate for motion modes that use it.
    pitch_amp_deg, pitch_freq_hz : float
        Pitch oscillation parameters for ``circular_with_pitch``.
    radius_m : float
        Circle radius for ``circular_with_pitch``.
    gyro_bias_xyz, accel_bias_xyz : 3-tuple
        Constant truth biases added to the synthesized IMU output.
    gyro_noise_std, accel_noise_std, cam_noise_std : float
        White-noise standard deviations.
    dropout_window_s : (t_start, t_end) or None
        Camera blackout window (mask_cam becomes False; LEDs become NaN).
    led_offsets_body, initial_position : array-like, optional
        Override LED body-frame offsets (shape ``(2, 3)``) and initial
        world-frame position (shape ``(3,)``).
    """

    rng = np.random.default_rng(seed)
    if motion == "perfect":
        gyro_noise_std = 0.0
        accel_noise_std = 0.0
        cam_noise_std = 0.0

    n_imu = round(duration_s * fs_imu)
    n_cam = round(duration_s * fs_cam)
    dt_imu = 1.0 / fs_imu
    t_imu = np.arange(n_imu, dtype=np.float64) * dt_imu
    t_cam = np.arange(n_cam, dtype=np.float64) / fs_cam
    # Clip camera timestamps to the IMU support so the EKF doesn't
    # extrapolate at the right boundary.
    t_cam = np.clip(t_cam, t_imu[0], t_imu[-1])

    init_pos = (
        np.array([0.0, 0.0, 0.3], dtype=np.float64)
        if initial_position is None
        else np.asarray(initial_position, dtype=np.float64)
    )
    if led_offsets_body is None:
        led_offsets_body = np.array(
            [[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]],
            dtype=np.float64,
        )
    else:
        led_offsets_body = np.asarray(led_offsets_body, dtype=np.float64)
    assert led_offsets_body.shape == (2, 3), "expected two LEDs with 3D offsets"

    # --- Build truth trajectory ---------------------------------------------
    positions = np.zeros((n_imu, 3), dtype=np.float64)
    quats = np.zeros((n_imu, 4), dtype=np.float64)
    omega_world = np.zeros((n_imu, 3), dtype=np.float64)
    accel_world = np.zeros((n_imu, 3), dtype=np.float64)

    yaw_rate_rad = float(np.deg2rad(yaw_rate_dps))
    pitch_amp_rad = float(np.deg2rad(pitch_amp_deg))
    omega_pitch_rad = 2.0 * np.pi * pitch_freq_hz

    if motion in ("stationary", "perfect"):
        roll_t = np.full(n_imu, np.deg2rad(init_roll_deg))
        pitch_t = np.full(n_imu, np.deg2rad(init_pitch_deg))
        yaw_t = np.full(n_imu, np.deg2rad(init_yaw_deg))
        positions[:] = init_pos
    elif motion == "yaw_only":
        roll_t = np.full(n_imu, np.deg2rad(init_roll_deg))
        pitch_t = np.full(n_imu, np.deg2rad(init_pitch_deg))
        yaw_t = np.deg2rad(init_yaw_deg) + yaw_rate_rad * t_imu
        positions[:] = init_pos
    elif motion == "circular_with_pitch":
        roll_t = np.full(n_imu, np.deg2rad(init_roll_deg))
        pitch_t = np.deg2rad(init_pitch_deg) + pitch_amp_rad * np.sin(
            omega_pitch_rad * t_imu
        )
        yaw_t = np.deg2rad(init_yaw_deg) + yaw_rate_rad * t_imu
        cx, cy, cz = init_pos
        positions[:, 0] = cx + radius_m * np.cos(yaw_rate_rad * t_imu)
        positions[:, 1] = cy + radius_m * np.sin(yaw_rate_rad * t_imu)
        positions[:, 2] = cz
    else:
        raise ValueError(f"Unknown motion mode: {motion!r}")

    for i in range(n_imu):
        quats[i] = _quaternion_from_euler_zyx(roll_t[i], pitch_t[i], yaw_t[i])

    # World-frame angular velocity for ZYX Euler:
    #   ω_world = R_yaw R_pitch [ṙoll, 0, 0] + R_yaw [0, ṗitch, 0] + [0, 0, ẏaw]
    # For our trajectories roll is constant and pitch / yaw vary, so use
    # the closed-form world-frame composition. Then rotate to body frame.
    roll_dot = np.gradient(roll_t, dt_imu)
    pitch_dot = np.gradient(pitch_t, dt_imu)
    yaw_dot = np.gradient(yaw_t, dt_imu)

    cy_t = np.cos(yaw_t)
    sy_t = np.sin(yaw_t)
    cp_t = np.cos(pitch_t)
    sp_t = np.sin(pitch_t)
    # See e.g. Diebel (2006). Eq. 405-407 for ZYX body rates ↔ Euler rates.
    omega_world[:, 0] = roll_dot * cy_t * cp_t - pitch_dot * sy_t
    omega_world[:, 1] = roll_dot * sy_t * cp_t + pitch_dot * cy_t
    omega_world[:, 2] = -roll_dot * sp_t + yaw_dot

    # World-frame linear acceleration from finite differences of velocity.
    vel = np.gradient(positions, dt_imu, axis=0)
    accel_world = np.gradient(vel, dt_imu, axis=0)

    # --- Build IMU readings -------------------------------------------------
    quats_jax_ready = quats.astype(np.float64)
    omega_body = np.zeros((n_imu, 3), dtype=np.float64)
    accel_body = np.zeros((n_imu, 3), dtype=np.float64)
    gravity_w = GRAVITY_W
    for i in range(n_imu):
        q = quats_jax_ready[i]
        omega_body[i] = np.asarray(
            rotate_vector_world_to_body(q, omega_world[i].astype(np.float32))
        )
        gravity_body_i = np.asarray(
            rotate_vector_world_to_body(q, gravity_w.astype(np.float32))
        )
        accel_body_kin = np.asarray(
            rotate_vector_world_to_body(q, accel_world[i].astype(np.float32))
        )
        # Accelerometer reports specific force a_body - g_body.
        # Filter convention: gravity vector points +z up so specific force at
        # rest equals -gravity_body. Match the ``extended_kalman_filter_3d``
        # contract (see _extended_kalman_filter_3d_core / filter_common.py
        # accel_body_kinematic = imu[3:6] - accel_bias - expected_gravity_body).
        accel_body[i] = accel_body_kin + gravity_body_i

    gyro_bias = np.asarray(gyro_bias_xyz, dtype=np.float64)
    accel_bias = np.asarray(accel_bias_xyz, dtype=np.float64)

    U_imu = np.zeros((n_imu, 6), dtype=np.float64)
    U_imu[:, 0:3] = omega_body + gyro_bias[None, :]
    U_imu[:, 3:6] = accel_body + accel_bias[None, :]
    if gyro_noise_std > 0:
        U_imu[:, 0:3] += rng.normal(0.0, gyro_noise_std, size=(n_imu, 3))
    if accel_noise_std > 0:
        U_imu[:, 3:6] += rng.normal(0.0, accel_noise_std, size=(n_imu, 3))

    # --- Build 3D LED observations -----------------------------------------
    def _interp_truth(t_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_t = np.empty((t_target.size, 3), dtype=np.float64)
        for d in range(3):
            pos_t[:, d] = np.interp(t_target, t_imu, positions[:, d])
        # Slerp would be more correct, but linear+renorm is good enough
        # for the smooth trajectories used here. Renormalize per-row.
        quat_t = np.empty((t_target.size, 4), dtype=np.float64)
        for d in range(4):
            quat_t[:, d] = np.interp(t_target, t_imu, quats[:, d])
        norms = np.linalg.norm(quat_t, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        quat_t /= norms
        return pos_t, quat_t

    pos_cam, quat_cam = _interp_truth(t_cam)
    leds_world = np.empty((t_cam.size, 2, 3), dtype=np.float64)
    for i in range(t_cam.size):
        q = quat_cam[i]
        leds_world[i] = pos_cam[i][None, :] + np.asarray(
            rotate_vector_body_to_world(q, led_offsets_body.astype(np.float32))
        )

    Z_cam_led1 = leds_world[:, 0, :].copy()
    Z_cam_led2 = leds_world[:, 1, :].copy()
    if cam_noise_std > 0:
        Z_cam_led1 += rng.normal(0.0, cam_noise_std, size=Z_cam_led1.shape)
        Z_cam_led2 += rng.normal(0.0, cam_noise_std, size=Z_cam_led2.shape)

    mask_cam = np.ones(t_cam.size, dtype=bool)
    if dropout_window_s is not None:
        t_lo, t_hi = dropout_window_s
        in_dropout = (t_cam >= t_lo) & (t_cam < t_hi)
        mask_cam[in_dropout] = False
        Z_cam_led1[in_dropout] = np.nan
        Z_cam_led2[in_dropout] = np.nan

    return SimSession(
        t_imu=t_imu,
        U_imu=U_imu,
        t_cam=t_cam,
        Z_cam_led1=Z_cam_led1,
        Z_cam_led2=Z_cam_led2,
        mask_cam=mask_cam,
        truth={
            "positions": positions,
            "quaternions": quats,
            "gyro_bias": gyro_bias,
            "accel_bias": accel_bias,
            "led_offsets_body": led_offsets_body,
        },
    )
