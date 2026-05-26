"""Tests for offline IMU calibration diagnostics."""

import numpy as np
import pytest

from trodestrack.qa.imu_calibration import (
    AxisSignDiagnostic,
    ImuCalibrationReport,
    LagFit,
    camera_body_acceleration,
    camera_midpoint,
    compute_led_heading,
    diagnose_accel_axis_signs,
    estimate_accel_gravity_body,
    estimate_gyro_bias,
    estimate_stationary_mask,
    finite_difference,
    format_imu_calibration_report,
    interpolate_columns,
    lagged_linear_fit,
    run_imu_calibration_diagnostics,
    smooth_time_series,
)


def test_stationary_bias_and_gravity_estimates_recover_known_values() -> None:
    t_imu = np.linspace(0.0, 10.0, 1_001)
    gyro_bias = 0.037
    gravity_body = np.array([0.4, 3.9, -8.4])

    gyro_z = np.full_like(t_imu, gyro_bias)
    gyro_z[700:] = 0.8
    accel_xyz = np.tile(gravity_body, (t_imu.size, 1))
    accel_xyz[700:, 0] += 2.0

    stationary = estimate_stationary_mask(
        t_imu,
        gyro_z,
        gyro_threshold=0.2,
    )

    assert stationary.sum() == 700
    np.testing.assert_allclose(estimate_gyro_bias(gyro_z, stationary), gyro_bias)
    np.testing.assert_allclose(
        estimate_accel_gravity_body(accel_xyz, stationary),
        gravity_body,
    )


def test_estimate_stationary_mask_excludes_non_overlapping_camera_range() -> None:
    """IMU samples outside the camera time range must not count as stationary.

    ``np.interp`` clamps to ``speed[0]`` / ``speed[-1]`` outside the
    camera span, which silently extended the camera speed signal as
    a constant past the actual recording. Probe: disjoint
    camera 0..1 s and IMU 10..11 s used to give 101/101 stationary
    samples — a non-overlapping calibration would have been
    accepted as success. With endpoint NaN, every IMU sample falls
    outside the camera span and the mask is empty.
    """

    t_cam = np.linspace(0.0, 1.0, 11)
    t_imu = np.linspace(10.0, 11.0, 101)
    led1 = np.column_stack([np.zeros_like(t_cam), np.zeros_like(t_cam)])
    led2 = led1 + np.array([0.04, 0.0])
    gyro_z = np.zeros_like(t_imu)

    stationary = estimate_stationary_mask(
        t_imu,
        gyro_z,
        t_cam=t_cam,
        led1=led1,
        led2=led2,
        speed_threshold=1.0,
        gyro_threshold=1.0,
    )

    assert stationary.sum() == 0


def test_calibration_no_stationary_samples_mentions_time_offsets() -> None:
    """Config users need clock-field guidance when calibration has no overlap."""

    t_cam = np.linspace(0.0, 1.0, 20)
    t_imu = np.linspace(10.0, 11.0, 100)
    led1 = np.column_stack([np.zeros_like(t_cam), np.zeros_like(t_cam)])
    led2 = led1 + np.array([0.04, 0.0])
    gyro_z = np.zeros_like(t_imu)
    accel_xyz = np.tile([0.0, 0.0, 9.80665], (t_imu.size, 1))

    with pytest.raises(ValueError) as exc_info:
        run_imu_calibration_diagnostics(
            t_imu=t_imu,
            gyro_z=gyro_z,
            accel_xyz=accel_xyz,
            t_cam=t_cam,
            led1=led1,
            led2=led2,
        )

    message = str(exc_info.value)
    assert "imu.time_offset_s" in message
    assert "camera.time_offset_s" in message


def test_lagged_linear_fit_excludes_samples_outside_source_range() -> None:
    """Target samples shifted outside the source span must not bias the fit.

    With ``np.interp``'s default endpoint clamp, ``t_target + lag``
    values past the source end were silently set to ``source[-1]``,
    so a candidate lag could "fit" with a long tail of constant
    extrapolation. Endpoint NaN now drops those samples and the
    correlation reflects only the genuine overlap.
    """

    rng = np.random.default_rng(0)
    t_source = np.linspace(0.0, 1.0, 101)
    source = rng.standard_normal(t_source.size)
    # Target span extends well past the source span; only the
    # ``[0, 1]`` overlap should contribute to the fit.
    t_target = np.linspace(0.0, 5.0, 501)
    target = np.full_like(t_target, source[-1])
    target[: t_source.size] = source

    fit = lagged_linear_fit(
        t_source,
        source,
        t_target,
        target,
        candidate_lags_s=np.array([0.0]),
    )

    # The clamping bug used to inflate correlation toward 1.0
    # because ``source[-1]`` matched ``target[t_source.size:]`` by
    # construction. With NaN clamping, the fit is restricted to the
    # genuinely overlapping samples.
    assert fit.n_samples == t_source.size
    assert np.isfinite(fit.correlation)


def test_lagged_linear_fit_recovers_lag_scale_and_bias() -> None:
    t_source = np.linspace(0.0, 20.0, 2_001)
    t_target = np.linspace(1.0, 19.0, 901)
    true_lag = 0.12
    source_signal = np.sin(2.0 * np.pi * 0.4 * (t_source - true_lag))
    target_signal = 0.25 + 1.7 * np.sin(2.0 * np.pi * 0.4 * t_target)
    candidate_lags = np.linspace(-0.3, 0.3, 121)

    fit = lagged_linear_fit(
        t_source,
        source_signal,
        t_target,
        target_signal,
        candidate_lags,
    )

    assert abs(fit.lag_s - true_lag) <= 0.01
    assert fit.correlation > 0.99
    np.testing.assert_allclose(fit.slope, 1.7, rtol=1e-2)
    np.testing.assert_allclose(fit.intercept, 0.25, atol=1e-3)


def test_lagged_linear_fit_no_valid_fit_mentions_config_fields() -> None:
    """No-fit diagnostics should point config users at clock and gyro fields."""

    t_source = np.linspace(0.0, 1.0, 20)
    t_target = np.linspace(0.0, 1.0, 20)
    source = np.zeros_like(t_source)
    target = np.sin(t_target)

    with pytest.raises(ValueError) as exc_info:
        lagged_linear_fit(
            t_source,
            source,
            t_target,
            target,
            candidate_lags_s=np.array([0.0]),
        )

    message = str(exc_info.value)
    assert "imu.time_offset_s" in message
    assert "camera.time_offset_s" in message
    assert "imu.axis_map['gyro_z']" in message
    assert "imu.axis_signs['gyro_z']" in message


def test_accel_axis_sign_diagnostics_identify_swaps_and_signs() -> None:
    t_imu = np.linspace(0.0, 10.0, 1_001)
    t_cam = np.linspace(0.5, 9.5, 271)
    body_x = np.sin(2.0 * np.pi * 0.5 * t_imu)
    body_y = np.cos(2.0 * np.pi * 0.3 * t_imu)
    accel_kinematic = np.column_stack(
        [
            0.1 * np.sin(t_imu),
            -body_x,
            body_y,
        ]
    )
    camera_body = np.column_stack(
        [
            np.interp(t_cam, t_imu, body_x),
            np.interp(t_cam, t_imu, body_y),
        ]
    )

    diagnostics = diagnose_accel_axis_signs(
        t_imu,
        accel_kinematic,
        t_cam,
        camera_body,
        candidate_lags_s=np.array([0.0]),
    )

    assert diagnostics[0].target_axis == "body_x"
    assert diagnostics[0].imu_axis == "y"
    assert diagnostics[0].sign == -1
    assert diagnostics[0].correlation > 0.99

    assert diagnostics[1].target_axis == "body_y"
    assert diagnostics[1].imu_axis == "z"
    assert diagnostics[1].sign == 1
    assert diagnostics[1].correlation > 0.99


def test_heading_and_camera_acceleration_handle_missing_led_frames() -> None:
    t_cam = np.linspace(0.0, 1.0, 21)
    led1 = np.column_stack([t_cam, np.zeros_like(t_cam)])
    led2 = led1 + np.array([0.04, 0.0])
    led1[8:11] = np.nan
    led2[9:12] = np.nan

    heading = compute_led_heading(led1, led2)
    accel = camera_body_acceleration(t_cam, led1, led2, smoothing_sigma_s=0.05)

    assert np.isfinite(heading).all()
    assert np.isfinite(accel).all()
    np.testing.assert_allclose(heading, 0.0, atol=1e-7)


def test_accel_axis_sign_diagnostics_use_one_to_one_axis_assignment() -> None:
    t_imu = np.linspace(0.0, 10.0, 1_001)
    t_cam = np.linspace(0.5, 9.5, 271)
    shared = np.sin(2.0 * np.pi * 0.4 * t_imu)
    body_y_true = np.cos(2.0 * np.pi * 0.3 * t_imu)
    accel_kinematic = np.column_stack(
        [
            shared,
            body_y_true,
            0.1 * np.sin(t_imu),
        ]
    )
    camera_body = np.column_stack(
        [
            np.interp(t_cam, t_imu, shared),
            0.95 * np.interp(t_cam, t_imu, shared)
            + 0.25 * np.interp(t_cam, t_imu, body_y_true),
        ]
    )

    diagnostics = diagnose_accel_axis_signs(
        t_imu,
        accel_kinematic,
        t_cam,
        camera_body,
        candidate_lags_s=np.array([0.0]),
    )

    assert diagnostics[0].imu_axis != diagnostics[1].imu_axis


def test_imu_calibration_helpers_reject_corrupted_masks() -> None:
    """``estimate_gyro_bias`` / ``estimate_accel_gravity_body`` must not silently
    coerce NaN / non-{0,1} integer masks to True.

    A bare ``np.asarray(mask).astype(bool)`` previously turned NaN
    samples and integer values like 2 into ``True``, so a corrupted
    ``stationary_mask`` would treat noisy outliers as stationary and
    bias the gravity / gyro-bias estimate. Validate the dtype contract
    explicitly first.
    """
    gyro = np.array([0.001, 0.002, 0.0015, 0.001], dtype=float)
    accel = np.tile(np.array([0.0, 0.0, 9.81]), (4, 1))

    # Clean bool / 0-1 int masks must continue to work.
    estimate_gyro_bias(gyro, np.array([True, True, False, True]))
    estimate_gyro_bias(gyro, np.array([1, 1, 0, 1], dtype=np.int32))
    estimate_accel_gravity_body(accel, np.array([True, True, False, True]))

    # Corrupted masks must raise ValueError instead of silently coercing.
    bad_int = np.array([1, 1, 0, 2], dtype=np.int32)
    with pytest.raises(ValueError, match=r"boolean or 0/1 integer"):
        estimate_gyro_bias(gyro, bad_int)
    with pytest.raises(ValueError, match=r"boolean or 0/1 integer"):
        estimate_accel_gravity_body(accel, bad_int)

    bad_nan = np.array([1.0, 1.0, 0.0, np.nan])
    with pytest.raises(ValueError, match=r"boolean or 0/1 integer"):
        estimate_gyro_bias(gyro, bad_nan)
    with pytest.raises(ValueError, match=r"boolean or 0/1 integer"):
        estimate_accel_gravity_body(accel, bad_nan)


def test_lagged_linear_fit_rejects_corrupted_valid_target() -> None:
    """``lagged_linear_fit`` must enforce the same dtype contract as the
    other masked diagnostics in this module.

    ``np.asarray(valid_target, dtype=bool)`` would silently coerce
    NaN / non-{0,1} integers to True and bias the fit.
    """
    t = np.linspace(0.0, 1.0, 5)
    src = np.zeros(5)
    target = np.zeros(5)
    lags = np.array([0.0])
    bad_int = np.array([1, 1, 2, 1, 1], dtype=np.int32)
    bad_nan = np.array([1.0, 1.0, np.nan, 1.0, 1.0])

    with pytest.raises(ValueError, match=r"valid_target must be boolean or 0/1"):
        lagged_linear_fit(t, src, t, target, lags, valid_target=bad_int)
    with pytest.raises(ValueError, match=r"valid_target must be boolean or 0/1"):
        lagged_linear_fit(t, src, t, target, lags, valid_target=bad_nan)
    # Wrong shape also rejected.
    with pytest.raises(ValueError, match=r"valid_target must have shape"):
        lagged_linear_fit(
            t, src, t, target, lags, valid_target=np.ones((5, 1), dtype=bool)
        )


def test_diagnose_accel_axis_signs_rejects_corrupted_valid_camera() -> None:
    """``diagnose_accel_axis_signs`` must apply the same contract to
    ``valid_camera``.
    """
    t_imu = np.linspace(0.0, 1.0, 5)
    accel = np.zeros((5, 3))
    t_cam = t_imu.copy()
    cam = np.zeros((5, 2))
    lags = np.array([0.0])

    bad_int = np.array([1, 1, 2, 1, 1], dtype=np.int32)
    bad_nan = np.array([1.0, 1.0, np.nan, 1.0, 1.0])

    with pytest.raises(ValueError, match=r"valid_camera must be boolean or 0/1"):
        diagnose_accel_axis_signs(
            t_imu, accel, t_cam, cam, candidate_lags_s=lags, valid_camera=bad_int
        )
    with pytest.raises(ValueError, match=r"valid_camera must be boolean or 0/1"):
        diagnose_accel_axis_signs(
            t_imu, accel, t_cam, cam, candidate_lags_s=lags, valid_camera=bad_nan
        )
    # Wrong shape also rejected.
    with pytest.raises(ValueError, match=r"valid_camera must have shape"):
        diagnose_accel_axis_signs(
            t_imu,
            accel,
            t_cam,
            cam,
            candidate_lags_s=lags,
            valid_camera=np.ones((5, 1), dtype=bool),
        )


def test_smooth_time_series_rejects_wrong_shape_mask() -> None:
    """``smooth_time_series`` must reject ``valid_mask`` shapes that don't
    match ``values`` instead of letting NumPy raise raw IndexError /
    broadcast errors.
    """
    t = np.linspace(0.0, 1.0, 5)
    values = np.zeros(5)

    # (N, 1) column-vector mask: previously raised ``IndexError: too
    # many indices for array``; now must surface the contract error.
    with pytest.raises(ValueError, match=r"valid_mask must have shape"):
        smooth_time_series(
            t, values, sigma_s=0.1, valid_mask=np.ones((5, 1), dtype=bool)
        )

    # Wrong-length mask: previously raised a raw NumPy broadcast error.
    with pytest.raises(ValueError, match=r"valid_mask must have shape"):
        smooth_time_series(t, values, sigma_s=0.1, valid_mask=np.ones(4, dtype=bool))

    # Clean (n_time,) bool / 0-1-int still works.
    smooth_time_series(t, values, sigma_s=0.1, valid_mask=np.ones(5, dtype=bool))
    smooth_time_series(t, values, sigma_s=0.1, valid_mask=np.ones(5, dtype=np.int32))


# Direct tests for previously transitive helpers.
#
# These functions are reachable only via larger integration calls in the
# existing tests, which means a sign error, off-by-one, or shape bug would
# be hidden by downstream filtering. Test them directly.


class TestFiniteDifference:
    def test_derivative_of_linear_function_is_constant_slope(self) -> None:
        t = np.linspace(0.0, 1.0, 101)
        slope = 3.7
        values = slope * t + 1.2
        deriv = finite_difference(t, values)
        np.testing.assert_allclose(deriv, slope, rtol=1e-10, atol=1e-10)

    def test_derivative_of_quadratic_function_matches_analytic(self) -> None:
        t = np.linspace(0.0, 2.0, 201)
        values = t**2  # d/dt = 2t
        deriv = finite_difference(t, values)
        # np.gradient with edge_order=2 is exact on quadratics for uniform spacing.
        np.testing.assert_allclose(deriv, 2.0 * t, rtol=1e-9, atol=1e-9)

    def test_handles_2d_values_column_wise(self) -> None:
        t = np.linspace(0.0, 1.0, 51)
        # Column 0: slope 2.0; column 1: slope -5.0
        values = np.column_stack([2.0 * t, -5.0 * t + 7.0])
        deriv = finite_difference(t, values)
        np.testing.assert_allclose(deriv[:, 0], 2.0, atol=1e-10)
        np.testing.assert_allclose(deriv[:, 1], -5.0, atol=1e-10)

    def test_rejects_nonincreasing_timestamps(self) -> None:
        t = np.array([0.0, 0.1, 0.1, 0.3])
        values = np.zeros_like(t)
        with pytest.raises(ValueError, match="strictly increasing"):
            finite_difference(t, values)

    def test_rejects_nonfinite_timestamps(self) -> None:
        t = np.array([0.0, 0.1, np.inf, 0.3])
        values = np.zeros_like(t)
        with pytest.raises(ValueError, match="non-finite"):
            finite_difference(t, values)


class TestCameraMidpoint:
    def test_midpoint_of_symmetric_leds(self) -> None:
        led1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        led2 = np.array([[2.0, 0.0], [3.0, 3.0]])
        mid = camera_midpoint(led1, led2)
        np.testing.assert_allclose(mid, np.array([[1.0, 0.0], [2.0, 2.0]]))

    def test_rejects_mismatched_shapes(self) -> None:
        led1 = np.zeros((10, 2))
        led2 = np.zeros((10, 3))
        with pytest.raises(ValueError):
            camera_midpoint(led1, led2)

    def test_does_not_broadcast_scalar_led_silently(self) -> None:
        """A 1D led2 would silently broadcast across time without the guard."""
        led1 = np.zeros((5, 2))
        led2 = np.array([0.5, 0.5])  # shape (2,) not (5, 2)
        with pytest.raises(ValueError):
            camera_midpoint(led1, led2)


class TestInterpolateColumns:
    def test_1d_interp_matches_numpy_interp(self) -> None:
        t_src = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 30.0])
        t_dst = np.array([0.5, 1.5])
        result = interpolate_columns(t_src, values, t_dst)
        np.testing.assert_allclose(result, np.array([15.0, 25.0]))

    def test_2d_interp_handles_each_column_independently(self) -> None:
        t_src = np.array([0.0, 1.0, 2.0])
        values = np.column_stack([[10.0, 20.0, 30.0], [100.0, 50.0, 0.0]])
        t_dst = np.array([0.5, 1.5])
        result = interpolate_columns(t_src, values, t_dst)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[:, 0], np.array([15.0, 25.0]))
        np.testing.assert_allclose(result[:, 1], np.array([75.0, 25.0]))

    def test_rejects_3d_input(self) -> None:
        t_src = np.array([0.0, 1.0])
        values = np.zeros((2, 2, 2))
        t_dst = np.array([0.5])
        with pytest.raises(ValueError, match="1D or 2D"):
            interpolate_columns(t_src, values, t_dst)


class TestFormatImuCalibrationReport:
    def _make_report(self) -> ImuCalibrationReport:
        return ImuCalibrationReport(
            gyro_bias_z=0.012345,
            accel_gravity_body=np.array([0.1, 0.2, -9.81]),
            stationary_fraction=0.482,
            stationary_samples=1234,
            yaw_rate_fit=LagFit(
                lag_s=-0.020,
                correlation=0.987,
                slope=1.05,
                intercept=0.001,
                r2=0.974,
                n_samples=600,
            ),
            accel_axis_diagnostics=(
                AxisSignDiagnostic(
                    target_axis="forward",
                    imu_axis="x",
                    sign=+1,
                    lag_s=0.005,
                    correlation=0.91,
                    n_samples=500,
                ),
                AxisSignDiagnostic(
                    target_axis="lateral",
                    imu_axis="y",
                    sign=-1,
                    lag_s=-0.010,
                    correlation=0.85,
                    n_samples=500,
                ),
            ),
        )

    def test_includes_all_top_level_fields(self) -> None:
        text = format_imu_calibration_report(self._make_report())
        assert "gyro_bias_z: 0.012345" in text
        assert "accel_gravity_body: [0.100000, 0.200000, -9.810000]" in text
        assert "stationary_samples: 1234" in text
        assert "stationary_fraction: 0.482" in text

    def test_renders_signed_axis_diagnostics(self) -> None:
        text = format_imu_calibration_report(self._make_report())
        # Positive sign rendered as "+", negative as "-".
        assert "forward: +imu_x" in text
        assert "lateral: -imu_y" in text

    def test_returns_single_string_block(self) -> None:
        text = format_imu_calibration_report(self._make_report())
        assert isinstance(text, str)
        # No accidental list-of-lines return.
        assert "\n" in text
