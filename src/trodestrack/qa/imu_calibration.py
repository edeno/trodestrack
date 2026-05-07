"""IMU calibration and signal-alignment diagnostics.

The helpers in this module are intentionally offline diagnostics. They estimate
basic bias/gravity quantities and report how well IMU signals line up with
camera-derived yaw and acceleration proxies, without mutating loader outputs or
changing filter configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trodestrack.qa.metrics import validate_bool_mask_dtype


@dataclass(frozen=True)
class LagFit:
    """Lagged linear fit summary between two scalar signals."""

    lag_s: float
    correlation: float
    slope: float
    intercept: float
    r2: float
    n_samples: int


@dataclass(frozen=True)
class AxisSignDiagnostic:
    """Best signed IMU-axis match for one camera-derived acceleration axis."""

    target_axis: str
    imu_axis: str
    sign: int
    lag_s: float
    correlation: float
    n_samples: int


@dataclass(frozen=True)
class ImuCalibrationReport:
    """Summary of IMU calibration and camera-alignment diagnostics."""

    gyro_bias_z: float
    accel_gravity_body: NDArray[np.float64]
    stationary_fraction: float
    stationary_samples: int
    yaw_rate_fit: LagFit
    accel_axis_diagnostics: tuple[AxisSignDiagnostic, ...]


def finite_dual_led_mask(
    led1: NDArray[np.floating],
    led2: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Return frames where both LEDs have finite 2D coordinates."""

    led1_arr, led2_arr = _validate_led_pair(led1, led2)
    return np.isfinite(led1_arr).all(axis=1) & np.isfinite(led2_arr).all(axis=1)


def compute_led_heading(
    led1: NDArray[np.floating],
    led2: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Compute unwrapped LED-pair heading from two 2D LED tracks.

    Parameters
    ----------
    led1, led2 : NDArray
        LED positions with shape ``(n_time, 2)`` in meters.

    Returns
    -------
    NDArray[np.float64]
        Unwrapped heading in radians, shape ``(n_time,)``.

    Notes
    -----
    Missing dual-LED frames are linearly interpolated after unwrapping valid
    LED-pair headings. Downstream diagnostics should still score fits only on
    originally valid frames.
    """

    led1_arr, led2_arr = _validate_led_pair(led1, led2)
    valid = finite_dual_led_mask(led1_arr, led2_arr)
    if np.sum(valid) < 2:
        raise ValueError("At least two finite dual-LED frames are required.")

    led_vec = led2_arr - led1_arr
    heading_valid = np.unwrap(np.arctan2(led_vec[valid, 1], led_vec[valid, 0]))
    sample_index = np.arange(led1_arr.shape[0], dtype=float)
    return np.interp(sample_index, sample_index[valid], heading_valid)


def finite_difference(
    t: NDArray[np.floating],
    values: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Differentiate samples with respect to possibly nonuniform timestamps."""

    t_arr = np.asarray(t, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if t_arr.ndim != 1:
        raise ValueError(f"t must be 1D; got shape {t_arr.shape}.")
    if values_arr.shape[0] != t_arr.shape[0]:
        raise ValueError(
            f"values first dimension {values_arr.shape[0]} does not match "
            f"len(t)={t_arr.shape[0]}."
        )
    if t_arr.shape[0] < 3:
        raise ValueError("At least three samples are required for finite differences.")
    # Reject non-finite timestamps before the strict-increase check —
    # ``np.diff`` of [..., +inf] is NaN, which fails ``> 0`` and would
    # mask the real problem; an array like [0, 1, +inf] also passes the
    # diff-positive check but produces NaN derivatives downstream.
    if not np.all(np.isfinite(t_arr)):
        n_bad = int(np.sum(~np.isfinite(t_arr)))
        raise ValueError(
            f"t contains {n_bad} non-finite value(s) (NaN/inf); "
            "timestamps must be finite seconds."
        )
    if not np.all(np.diff(t_arr) > 0):
        raise ValueError("t must be strictly increasing.")
    return np.gradient(values_arr, t_arr, axis=0, edge_order=2)


def camera_midpoint(
    led1: NDArray[np.floating],
    led2: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return dual-LED midpoint positions, shape ``(n_time, 2)``.

    Raises
    ------
    ValueError
        If ``led1`` and ``led2`` are not both shape ``(n_time, 2)``.

    Without this guard, numpy broadcasts a malformed ``(2,)`` ``led2``
    across every time step in ``led1`` and produces a finite-but-wrong
    midpoint, which then poisons downstream camera-speed and
    stationary-mask computations.
    """

    led1_arr, led2_arr = _validate_led_pair(led1, led2)
    return 0.5 * (led1_arr + led2_arr)


def smooth_time_series(
    t: NDArray[np.floating],
    values: NDArray[np.floating],
    sigma_s: float,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Interpolate invalid samples and apply Gaussian smoothing.

    Parameters
    ----------
    t : NDArray
        Timestamps, shape ``(n_time,)`` in seconds.
    values : NDArray
        Values with shape ``(n_time,)`` or ``(n_time, n_dim)``.
    sigma_s : float
        Gaussian smoothing standard deviation in seconds. Set to zero to only
        interpolate invalid samples.
    valid_mask : NDArray[np.bool_] | None, optional
        Optional validity mask over samples. Non-finite samples are always
        treated as invalid.

    Returns
    -------
    NDArray[np.float64]
        Smoothed values with the same shape as ``values``.
    """

    # sigma_s flows into `_gaussian_kernel` and then into `int(...)` for
    # the kernel half-width — a NaN/inf would otherwise raise the opaque
    # "cannot convert float NaN to integer" from Python's int(). Reject
    # at the public boundary instead.
    if not np.isfinite(sigma_s) or sigma_s < 0:
        raise ValueError(
            f"sigma_s must be a finite non-negative value in seconds; got {sigma_s!r}."
        )

    t_arr = np.asarray(t, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if t_arr.ndim != 1:
        raise ValueError(f"t must be 1D; got shape {t_arr.shape}.")
    if values_arr.shape[0] != t_arr.shape[0]:
        raise ValueError(
            f"values first dimension {values_arr.shape[0]} does not match "
            f"len(t)={t_arr.shape[0]}."
        )
    if not np.all(np.isfinite(t_arr)):
        n_bad = int(np.sum(~np.isfinite(t_arr)))
        raise ValueError(
            f"t contains {n_bad} non-finite value(s) (NaN/inf); "
            "timestamps must be finite seconds."
        )
    if not np.all(np.diff(t_arr) > 0):
        raise ValueError("t must be strictly increasing.")

    filled = _interpolate_invalid_samples(t_arr, values_arr, valid_mask=valid_mask)
    if sigma_s <= 0:
        return filled

    dt = float(np.median(np.diff(t_arr)))
    sigma_samples = sigma_s / dt
    kernel = _gaussian_kernel(sigma_samples)
    return _convolve_reflect(filled, kernel)


def camera_body_acceleration(
    t_cam: NDArray[np.floating],
    led1: NDArray[np.floating],
    led2: NDArray[np.floating],
    *,
    smoothing_sigma_s: float = 0.1,
) -> NDArray[np.float64]:
    """Estimate camera-derived acceleration in the animal/body xy frame.

    Parameters
    ----------
    t_cam : NDArray
        Camera timestamps, shape ``(n_time,)`` in seconds.
    led1, led2 : NDArray
        LED positions, each with shape ``(n_time, 2)`` in meters.
    smoothing_sigma_s : float, default 0.1
        Gaussian smoothing standard deviation applied before differentiating.

    Returns
    -------
    NDArray[np.float64]
        Body-frame acceleration proxy, shape ``(n_time, 2)`` in m/s².

    Notes
    -----
    This is a noisy proxy, not ground truth. It is meant for signal diagnostics
    and should normally be smoothed or interpreted conservatively.
    """

    valid = finite_dual_led_mask(led1, led2)
    midpoint = smooth_time_series(
        t_cam,
        camera_midpoint(led1, led2),
        smoothing_sigma_s,
        valid_mask=valid,
    )
    heading = smooth_time_series(
        t_cam,
        compute_led_heading(led1, led2),
        smoothing_sigma_s,
        valid_mask=valid,
    )
    velocity_world = finite_difference(t_cam, midpoint)
    acceleration_world = finite_difference(t_cam, velocity_world)

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    return np.column_stack(
        [
            cos_h * acceleration_world[:, 0] + sin_h * acceleration_world[:, 1],
            -sin_h * acceleration_world[:, 0] + cos_h * acceleration_world[:, 1],
        ]
    )


def interpolate_columns(
    t_src: NDArray[np.floating],
    values: NDArray[np.floating],
    t_dst: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Interpolate a 1D or 2D array column-wise onto new timestamps."""

    values_arr = np.asarray(values, dtype=float)
    if values_arr.ndim == 1:
        return np.interp(t_dst, t_src, values_arr)
    if values_arr.ndim == 2:
        return np.column_stack(
            [
                np.interp(t_dst, t_src, values_arr[:, col])
                for col in range(values_arr.shape[1])
            ]
        )
    raise ValueError(f"values must be 1D or 2D; got shape {values_arr.shape}.")


def estimate_stationary_mask(
    t_imu: NDArray[np.floating],
    gyro_z: NDArray[np.floating],
    *,
    t_cam: NDArray[np.floating] | None = None,
    led1: NDArray[np.floating] | None = None,
    led2: NDArray[np.floating] | None = None,
    speed_threshold: float = 0.05,
    gyro_threshold: float = 0.2,
) -> NDArray[np.bool_]:
    """Estimate low-motion IMU samples for bias/gravity summaries.

    Parameters
    ----------
    t_imu : NDArray
        IMU timestamps, shape ``(n_imu,)`` in seconds.
    gyro_z : NDArray
        Yaw gyro signal, shape ``(n_imu,)`` in rad/s.
    t_cam, led1, led2 : optional
        Camera timestamps and LED positions. If provided, camera speed must be
        below ``speed_threshold`` after interpolation to IMU times.
    speed_threshold : float, default 0.05
        Camera midpoint speed threshold in m/s.
    gyro_threshold : float, default 0.2
        Absolute yaw-rate threshold in rad/s.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask over IMU samples.
    """

    t_imu_arr = np.asarray(t_imu, dtype=float)
    gyro_arr = np.asarray(gyro_z, dtype=float)
    if t_imu_arr.ndim != 1:
        raise ValueError(f"t_imu must be 1D; got shape {t_imu_arr.shape}.")
    if gyro_arr.ndim != 1:
        raise ValueError(f"gyro_z must be 1D; got shape {gyro_arr.shape}.")
    if gyro_arr.shape != t_imu_arr.shape:
        # The returned mask is documented as "over IMU samples". Without
        # this guard, gyro-only callers got a mask of shape ``gyro_z.shape``
        # that misaligned with ``(n_imu, 3)`` accelerometer arrays in
        # downstream estimators (estimate_accel_gravity_body, etc.).
        raise ValueError(
            f"gyro_z must have the same length as t_imu; got "
            f"gyro_z {gyro_arr.shape} vs t_imu {t_imu_arr.shape}."
        )
    mask = np.isfinite(gyro_arr) & (np.abs(gyro_arr) < gyro_threshold)

    camera_inputs = (t_cam, led1, led2)
    if any(item is not None for item in camera_inputs):
        if not all(item is not None for item in camera_inputs):
            raise ValueError("t_cam, led1, and led2 must be provided together.")
        midpoint = camera_midpoint(led1, led2)  # type: ignore[arg-type]
        velocity = finite_difference(t_cam, midpoint)  # type: ignore[arg-type]
        speed = np.linalg.norm(velocity, axis=1)
        # ``np.interp`` clamps to ``speed[0]`` / ``speed[-1]`` outside the
        # camera time range; for a session whose camera and IMU clocks
        # don't overlap this turned every out-of-range IMU sample into
        # the camera's first/last speed value, silently counting
        # non-overlapping samples as "stationary" (probe: disjoint
        # camera 0..1 s and IMU 10..11 s gave 101/101 stationary).
        # ``left=NaN`` / ``right=NaN`` propagates through the
        # ``np.isfinite`` filter below so out-of-range samples are
        # explicitly excluded.
        speed_at_imu = np.interp(
            t_imu_arr,
            np.asarray(t_cam, dtype=float),
            speed,
            left=np.nan,
            right=np.nan,
        )
        mask &= np.isfinite(speed_at_imu) & (speed_at_imu < speed_threshold)

    return mask


def estimate_gyro_bias(
    gyro_z: NDArray[np.floating],
    stationary_mask: NDArray[np.bool_],
) -> float:
    """Estimate yaw gyro bias from low-motion samples using a median."""

    gyro_arr = np.asarray(gyro_z, dtype=float)
    # ``np.asarray(..., dtype=bool)`` silently coerces NaN / non-{0,1}
    # integer values to True, so a corrupted ``stationary_mask`` would
    # treat noisy or outlier samples as stationary. Validate the dtype
    # explicitly first.
    mask_arr = validate_bool_mask_dtype(
        np.asarray(stationary_mask), name="stationary_mask"
    )
    # Reject shape mismatch up front. NumPy would otherwise raise a raw
    # broadcasting error from the `mask & isfinite(gyro)` line below.
    if gyro_arr.ndim != 1:
        raise ValueError(f"gyro_z must be 1D; got shape {gyro_arr.shape}.")
    if mask_arr.shape != gyro_arr.shape:
        raise ValueError(
            "stationary_mask must have the same length as gyro_z; got "
            f"stationary_mask {mask_arr.shape} vs gyro_z {gyro_arr.shape}."
        )
    mask = mask_arr & np.isfinite(gyro_arr)
    if not np.any(mask):
        raise ValueError("No finite stationary samples available for gyro bias.")
    return float(np.median(gyro_arr[mask]))


def estimate_accel_gravity_body(
    accel_xyz: NDArray[np.floating],
    stationary_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Estimate stationary accelerometer gravity vector in body/sensor axes.

    This diagnostic reports the raw stationary sensor-frame reading. The EKF
    configuration field ``imu_gravity_body`` keeps a legacy name but is
    interpreted by the tracking dynamics as a world-frame gravity vector.
    """

    accel_arr = np.asarray(accel_xyz, dtype=float)
    if accel_arr.ndim != 2 or accel_arr.shape[1] != 3:
        raise ValueError(
            f"accel_xyz must have shape (n_time, 3); got {accel_arr.shape}."
        )
    mask_arr = validate_bool_mask_dtype(
        np.asarray(stationary_mask), name="stationary_mask"
    )
    # Reject mask-vs-signal length mismatch up front. NumPy would
    # otherwise raise a raw broadcasting error from the AND below.
    if mask_arr.shape != (accel_arr.shape[0],):
        raise ValueError(
            "stationary_mask must have shape (n_time,) matching accel_xyz; "
            f"got stationary_mask {mask_arr.shape} vs accel_xyz {accel_arr.shape}."
        )
    mask = mask_arr & np.isfinite(accel_arr).all(axis=1)
    if not np.any(mask):
        raise ValueError("No finite stationary samples available for accel gravity.")
    return np.median(accel_arr[mask], axis=0)


def lagged_linear_fit(
    t_source: NDArray[np.floating],
    source: NDArray[np.floating],
    t_target: NDArray[np.floating],
    target: NDArray[np.floating],
    candidate_lags_s: NDArray[np.floating],
    *,
    valid_target: NDArray[np.bool_] | None = None,
) -> LagFit:
    """Find the lag that maximizes absolute scalar correlation.

    Positive lag means the source is sampled at ``t_target + lag``.
    """

    t_source_arr = np.asarray(t_source, dtype=float)
    source_arr = np.asarray(source, dtype=float)
    t_target_arr = np.asarray(t_target, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    lags = np.asarray(candidate_lags_s, dtype=float)

    if valid_target is None:
        valid = np.ones_like(target_arr, dtype=bool)
    else:
        # Reject NaN / non-{0,1} integer masks before they coerce
        # silently to True. ``np.asarray(..., dtype=bool)`` on
        # ``[1, 1, 2, 1]`` or ``[1, NaN, 1]`` would treat 2 / NaN as
        # valid samples and bias the fit; mirror the gate used by
        # ``estimate_gyro_bias`` / ``estimate_accel_gravity_body``.
        valid_target_arr = np.asarray(valid_target)
        if valid_target_arr.shape != target_arr.shape:
            raise ValueError(
                f"valid_target must have shape {target_arr.shape}; got "
                f"{valid_target_arr.shape}."
            )
        valid = validate_bool_mask_dtype(valid_target_arr, name="valid_target").copy()
    valid &= np.isfinite(target_arr)

    # Only accept a candidate when the fit produced a finite correlation —
    # ``_linear_fit`` returns a NaN-filled ``LagFit`` when fewer than 3
    # finite samples remain. Initializing ``best_fit`` to that NaN result
    # would silently report a "no-data fit" as success.
    # ``np.interp`` clamps to ``source[0]`` / ``source[-1]`` outside
    # the source time range; combined with non-zero ``lag`` this
    # silently extrapolated the source signal as a constant well
    # past the actual sampled span. ``left=NaN`` / ``right=NaN``
    # propagates through the ``isfinite(source_shifted)`` filter so
    # out-of-overlap samples are excluded from the fit.
    best_fit: LagFit | None = None
    for lag in lags:
        source_shifted = np.interp(
            t_target_arr + lag,
            t_source_arr,
            source_arr,
            left=np.nan,
            right=np.nan,
        )
        finite = valid & np.isfinite(source_shifted)
        fit = _linear_fit(source_shifted[finite], target_arr[finite], lag)
        score = _abs_correlation_score(fit)
        if not np.isfinite(score):
            continue
        if best_fit is None or score > _abs_correlation_score(best_fit):
            best_fit = fit

    if best_fit is None:
        raise ValueError(
            "No candidate lag produced a valid fit. Each tried lag in "
            f"[{float(lags.min()):.3f}, {float(lags.max()):.3f}] s had fewer "
            "than 3 finite (source, target) pairs after applying "
            "``valid_target`` / interpolation. Common causes for this in the "
            "config-driven calibration path: IMU and camera time ranges "
            "don't overlap (check imu.time_offset_s and "
            "camera.time_offset_s); the IMU or camera input is mostly NaN; "
            "or no candidate lag falls within the actual time-range overlap."
        )
    return best_fit


def diagnose_accel_axis_signs(
    t_imu: NDArray[np.floating],
    accel_kinematic_xyz: NDArray[np.floating],
    t_cam: NDArray[np.floating],
    camera_accel_body_xy: NDArray[np.floating],
    candidate_lags_s: NDArray[np.floating],
    *,
    valid_camera: NDArray[np.bool_] | None = None,
) -> tuple[AxisSignDiagnostic, ...]:
    """Find best signed IMU axis for camera body-x and body-y acceleration."""

    t_imu_arr = np.asarray(t_imu, dtype=float)
    accel_arr = np.asarray(accel_kinematic_xyz, dtype=float)
    t_cam_arr = np.asarray(t_cam, dtype=float)
    camera_arr = np.asarray(camera_accel_body_xy, dtype=float)
    if accel_arr.ndim != 2 or accel_arr.shape[1] != 3:
        raise ValueError(
            f"accel_kinematic_xyz must have shape (n_imu, 3); got {accel_arr.shape}."
        )
    if camera_arr.ndim != 2 or camera_arr.shape[1] != 2:
        raise ValueError(
            f"camera_accel_body_xy must have shape (n_cam, 2); got {camera_arr.shape}."
        )

    if valid_camera is None:
        valid = np.isfinite(camera_arr).all(axis=1)
    else:
        # Same dtype contract as the other masked diagnostics in this
        # module — ``np.asarray(..., dtype=bool)`` on ``[1, 1, 2, 1]``
        # / ``[1, NaN, 1]`` would coerce 2 / NaN to True and bias the
        # axis-sign fit.
        valid_camera_arr = np.asarray(valid_camera)
        if valid_camera_arr.shape != (camera_arr.shape[0],):
            raise ValueError(
                "valid_camera must have shape (n_cam,)="
                f"({camera_arr.shape[0]},); got {valid_camera_arr.shape}."
            )
        valid = validate_bool_mask_dtype(
            valid_camera_arr, name="valid_camera"
        ) & np.isfinite(camera_arr).all(axis=1)

    labels = ("x", "y", "z")
    candidates: list[list[AxisSignDiagnostic]] = []
    for target_col, target_axis in enumerate(("body_x", "body_y")):
        target_candidates: list[AxisSignDiagnostic] = []
        target = camera_arr[:, target_col]
        for imu_col, imu_axis in enumerate(labels):
            for sign in (-1, 1):
                fit = lagged_linear_fit(
                    t_imu_arr,
                    sign * accel_arr[:, imu_col],
                    t_cam_arr,
                    target,
                    candidate_lags_s,
                    valid_target=valid,
                )
                diag = AxisSignDiagnostic(
                    target_axis=target_axis,
                    imu_axis=imu_axis,
                    sign=sign,
                    lag_s=fit.lag_s,
                    correlation=fit.correlation,
                    n_samples=fit.n_samples,
                )
                target_candidates.append(diag)
        if not target_candidates:
            raise ValueError(f"No valid axis/sign diagnostic for {target_axis}.")
        candidates.append(target_candidates)

    best_pair: tuple[AxisSignDiagnostic, AxisSignDiagnostic] | None = None
    best_score = -np.inf
    for body_x_diag in candidates[0]:
        for body_y_diag in candidates[1]:
            if body_x_diag.imu_axis == body_y_diag.imu_axis:
                continue
            score = _correlation_score(body_x_diag.correlation) + _correlation_score(
                body_y_diag.correlation
            )
            if score > best_score:
                best_score = score
                best_pair = (body_x_diag, body_y_diag)

    if best_pair is None:
        raise ValueError("No valid one-to-one axis/sign assignment found.")
    return best_pair


def run_imu_calibration_diagnostics(
    *,
    t_imu: NDArray[np.floating],
    gyro_z: NDArray[np.floating],
    accel_xyz: NDArray[np.floating],
    t_cam: NDArray[np.floating],
    led1: NDArray[np.floating],
    led2: NDArray[np.floating],
    candidate_lags_s: NDArray[np.floating] | None = None,
    speed_threshold: float = 0.05,
    gyro_threshold: float = 0.2,
    camera_smoothing_sigma_s: float = 0.1,
) -> ImuCalibrationReport:
    """Run bias, gravity, yaw-lag, and accel-axis diagnostics."""

    if candidate_lags_s is None:
        candidate_lags_s = np.linspace(-0.5, 0.5, 101)

    stationary = estimate_stationary_mask(
        t_imu,
        gyro_z,
        t_cam=t_cam,
        led1=led1,
        led2=led2,
        speed_threshold=speed_threshold,
        gyro_threshold=gyro_threshold,
    )
    gyro_bias = estimate_gyro_bias(gyro_z, stationary)
    gravity_body = estimate_accel_gravity_body(accel_xyz, stationary)

    valid_leds = finite_dual_led_mask(led1, led2)
    heading = smooth_time_series(
        t_cam,
        compute_led_heading(led1, led2),
        camera_smoothing_sigma_s,
        valid_mask=valid_leds,
    )
    heading_rate = finite_difference(t_cam, heading)
    valid_heading = valid_leds & np.isfinite(heading_rate)
    yaw_fit = lagged_linear_fit(
        t_imu,
        np.asarray(gyro_z, dtype=float) - gyro_bias,
        t_cam,
        heading_rate,
        candidate_lags_s,
        valid_target=valid_heading,
    )

    camera_accel = camera_body_acceleration(
        t_cam,
        led1,
        led2,
        smoothing_sigma_s=camera_smoothing_sigma_s,
    )
    accel_kinematic = np.asarray(accel_xyz, dtype=float) - gravity_body
    accel_axes = diagnose_accel_axis_signs(
        t_imu,
        accel_kinematic,
        t_cam,
        camera_accel,
        candidate_lags_s,
        valid_camera=valid_leds & np.isfinite(camera_accel).all(axis=1),
    )

    return ImuCalibrationReport(
        gyro_bias_z=gyro_bias,
        accel_gravity_body=gravity_body,
        stationary_fraction=float(np.mean(stationary)),
        stationary_samples=int(np.sum(stationary)),
        yaw_rate_fit=yaw_fit,
        accel_axis_diagnostics=accel_axes,
    )


def format_imu_calibration_report(report: ImuCalibrationReport) -> str:
    """Format an IMU calibration report for console output."""

    lines = [
        "IMU calibration diagnostics",
        "=" * 80,
        f"gyro_bias_z: {report.gyro_bias_z:.6f} rad/s",
        "accel_gravity_body: "
        f"[{report.accel_gravity_body[0]:.6f}, "
        f"{report.accel_gravity_body[1]:.6f}, "
        f"{report.accel_gravity_body[2]:.6f}] m/s^2",
        f"stationary_samples: {report.stationary_samples}",
        f"stationary_fraction: {report.stationary_fraction:.3f}",
        "",
        "Yaw-rate alignment:",
        f"  lag_s: {report.yaw_rate_fit.lag_s:+.3f}",
        f"  correlation: {report.yaw_rate_fit.correlation:+.3f}",
        f"  slope: {report.yaw_rate_fit.slope:+.3f}",
        f"  intercept: {report.yaw_rate_fit.intercept:+.3f}",
        f"  r2: {report.yaw_rate_fit.r2:.3f}",
        f"  n_samples: {report.yaw_rate_fit.n_samples}",
        "",
        "Accel axis/sign diagnostics:",
    ]
    for diag in report.accel_axis_diagnostics:
        sign_text = "+" if diag.sign > 0 else "-"
        lines.append(
            f"  {diag.target_axis}: {sign_text}imu_{diag.imu_axis}, "
            f"lag_s={diag.lag_s:+.3f}, "
            f"corr={diag.correlation:+.3f}, "
            f"n={diag.n_samples}"
        )
    return "\n".join(lines)


def _linear_fit(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    lag_s: float,
) -> LagFit:
    """Fit ``target ~= intercept + slope * source``."""

    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    finite = np.isfinite(source_arr) & np.isfinite(target_arr)
    source_arr = source_arr[finite]
    target_arr = target_arr[finite]
    if source_arr.size < 3:
        return LagFit(
            lag_s=float(lag_s),
            correlation=np.nan,
            slope=np.nan,
            intercept=np.nan,
            r2=np.nan,
            n_samples=int(source_arr.size),
        )

    source_centered = source_arr - np.mean(source_arr)
    target_centered = target_arr - np.mean(target_arr)
    denom = np.sqrt(np.sum(source_centered**2) * np.sum(target_centered**2))
    correlation = (
        float(np.sum(source_centered * target_centered) / denom)
        if denom > 0
        else np.nan
    )

    design = np.column_stack([np.ones(source_arr.size), source_arr])
    intercept, slope = np.linalg.lstsq(design, target_arr, rcond=None)[0]
    pred = design @ np.array([intercept, slope])
    ss_res = np.sum((target_arr - pred) ** 2)
    ss_tot = np.sum((target_arr - np.mean(target_arr)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return LagFit(
        lag_s=float(lag_s),
        correlation=correlation,
        slope=float(slope),
        intercept=float(intercept),
        r2=r2,
        n_samples=int(source_arr.size),
    )


def _correlation_score(correlation: float) -> float:
    """Return sortable correlation score with NaN treated as worst."""

    return float(correlation) if np.isfinite(correlation) else -np.inf


def _abs_correlation_score(fit: LagFit) -> float:
    """Return sortable absolute-correlation score with NaN treated as worst."""

    return abs(float(fit.correlation)) if np.isfinite(fit.correlation) else -np.inf


def _validate_led_pair(
    led1: NDArray[np.floating],
    led2: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return two 2D LED arrays as float arrays."""

    led1_arr = np.asarray(led1, dtype=float)
    led2_arr = np.asarray(led2, dtype=float)
    if led1_arr.shape != led2_arr.shape or led1_arr.ndim != 2 or led1_arr.shape[1] != 2:
        raise ValueError(
            "led1 and led2 must both have shape (n_time, 2); "
            f"got {led1_arr.shape} and {led2_arr.shape}."
        )
    return led1_arr, led2_arr


def _interpolate_invalid_samples(
    t: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Linearly interpolate invalid samples in a 1D or 2D time series."""

    # Validate the optional ``valid_mask`` once so both the 1D and 2D
    # branches use the same dtype + shape gate. The dtype check rejects
    # NaN / non-{0,1} integer masks; the shape check rejects ``(N, 1)``
    # column-vector masks (raw ``IndexError`` from ``values[mask]``) and
    # length-mismatched masks (raw NumPy broadcast errors).
    if valid_mask is None:
        valid_mask_clean = None
    else:
        mask_arr = np.asarray(valid_mask)
        if mask_arr.shape != (values.shape[0],):
            raise ValueError(
                f"valid_mask must have shape (n_time,)=({values.shape[0]},) "
                f"matching ``values``; got {mask_arr.shape}."
            )
        valid_mask_clean = validate_bool_mask_dtype(mask_arr, name="valid_mask")

    if values.ndim == 1:
        finite = np.isfinite(values)
        valid = finite if valid_mask_clean is None else finite & valid_mask_clean
        if np.sum(valid) < 2:
            raise ValueError(
                "At least two valid samples are required for interpolation."
            )
        return np.interp(t, t[valid], values[valid])

    if values.ndim == 2:
        finite_rows = np.isfinite(values).all(axis=1)
        valid_rows = (
            finite_rows if valid_mask_clean is None else finite_rows & valid_mask_clean
        )
        if np.sum(valid_rows) < 2:
            raise ValueError(
                "At least two valid samples are required for interpolation."
            )
        return np.column_stack(
            [
                np.interp(t, t[valid_rows], values[valid_rows, col])
                for col in range(values.shape[1])
            ]
        )

    raise ValueError(f"values must be 1D or 2D; got shape {values.shape}.")


def _gaussian_kernel(
    sigma_samples: float, truncate: float = 4.0
) -> NDArray[np.float64]:
    """Construct a normalized 1D Gaussian kernel."""

    # Reject non-finite sigma at this private boundary too — public
    # callers (smooth_time_series) already validate, but guarding here
    # turns any future caller's NaN into a clear ValueError instead of
    # the opaque ``int(NaN)`` failure inside ``np.ceil``/``int``.
    if not np.isfinite(sigma_samples) or not np.isfinite(truncate):
        raise ValueError(
            "sigma_samples and truncate must be finite; got "
            f"sigma_samples={sigma_samples!r}, truncate={truncate!r}."
        )
    if sigma_samples <= 0:
        return np.array([1.0])
    radius = max(1, int(np.ceil(truncate * sigma_samples)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    return kernel / np.sum(kernel)


def _convolve_reflect(
    values: NDArray[np.float64],
    kernel: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convolve a 1D or 2D array along time using reflected padding."""

    radius = (kernel.size - 1) // 2
    if radius == 0:
        return values.copy()

    def convolve_one(column: NDArray[np.float64]) -> NDArray[np.float64]:
        padded = np.pad(column, radius, mode="reflect")
        return np.convolve(padded, kernel, mode="valid")

    if values.ndim == 1:
        return convolve_one(values)
    return np.column_stack(
        [convolve_one(values[:, col]) for col in range(values.shape[1])]
    )
