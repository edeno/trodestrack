"""Tests for config-driven session loading and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trodestrack.config import SessionConfig, load_session_config
from trodestrack.io import (
    PreparedSession,
    load_session,
    run_real_data_safety_check,
    write_session_diagnostics,
)
from trodestrack.io.led_identity import CorrectedLEDIdentity
from trodestrack.io.session import (
    _index_or_time_column,
    _median_led_distance,
    _validate_calibration_for_fusion,
)
from trodestrack.models.ekf import EKFConfig, EKFResult
from trodestrack.qa.imu_calibration import (
    AxisSignDiagnostic,
    ImuCalibrationReport,
    LagFit,
)


def test_load_arthur_style_parquets_removes_sample_hold_and_projects_imu(tmp_path):
    """Arthur/Trodes parquet layout loads into filter-ready arrays."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    imu_path = data_dir / "imu.parquet"
    position_path = data_dir / "position.parquet"
    pd.DataFrame(
        {
            "time": [100.0, 100.005, 100.010, 100.015, 100.020],
            "Headstage_GyroX": [1, 1, 1, 1, 1],
            "Headstage_GyroY": [2, 2, 2, 2, 2],
            "Headstage_GyroZ": [0, 0, 10, 10, 20],
            "Headstage_AccelX": [100, 100, 110, 110, 120],
            "Headstage_AccelY": [200, 200, 210, 210, 220],
            "Headstage_AccelZ": [300, 300, 310, 310, 320],
        }
    ).to_parquet(imu_path)
    pd.DataFrame(
        {
            "time": [100.0, 100.033, 100.066],
            "xloc": [10.0, 11.0, 12.0],
            "yloc": [20.0, 20.0, 20.0],
            "xloc2": [14.0, 15.0, 16.0],
            "yloc2": [20.0, 20.0, 20.0],
        }
    ).to_parquet(position_path)
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
camera:
  meters_per_pixel: 0.01
filter:
  state_mode: 2d_cam_3d_imu
outputs:
  run_safety_checks: false
""".lstrip()
    )

    session = load_session(load_session_config(config_path))

    assert session.t_imu.shape == (3,)
    assert session.U_imu.shape == (3, 4)
    assert session.Z_cam_led1.shape == (3, 2)
    assert session.Z_cam_led2.shape == (3, 2)
    assert session.mask_cam.tolist() == [True, True, True]
    assert session.led_distance == pytest.approx(0.04)
    assert session.diagnostics["loader"]["sample_hold_strategy"] == "gyro_z_change"


def test_orientation_mode_keeps_six_channel_imu(tmp_path):
    """Config-driven 6-DOF orientation mode should receive full IMU samples."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    imu_path = data_dir / "imu.parquet"
    position_path = data_dir / "position.parquet"
    pd.DataFrame(
        {
            "time": [100.0, 100.005, 100.010, 100.015, 100.020],
            "Headstage_GyroX": [1, 1, 2, 2, 3],
            "Headstage_GyroY": [4, 4, 5, 5, 6],
            "Headstage_GyroZ": [0, 0, 10, 10, 20],
            "Headstage_AccelX": [100, 100, 110, 110, 120],
            "Headstage_AccelY": [200, 200, 210, 210, 220],
            "Headstage_AccelZ": [300, 300, 310, 310, 320],
        }
    ).to_parquet(imu_path)
    pd.DataFrame(
        {
            "time": [100.0, 100.033, 100.066],
            "xloc": [10.0, 11.0, 12.0],
            "yloc": [20.0, 20.0, 20.0],
            "xloc2": [14.0, 15.0, 16.0],
            "yloc2": [20.0, 20.0, 20.0],
        }
    ).to_parquet(position_path)
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
camera:
  meters_per_pixel: 0.01
filter:
  state_mode: 2d_cam_6dof_imu_orientation
outputs:
  run_safety_checks: false
""".lstrip()
    )

    session = load_session(load_session_config(config_path))

    assert session.U_imu.shape == (3, 6)
    np.testing.assert_allclose(session.gyro_z_for_led_identity, session.U_imu[:, 2])


def test_prepared_arrays_session_loads_and_writes_diagnostics(tmp_path):
    """Existing prepared text arrays continue to load through the config API."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    np.savetxt(input_dir / "t_imu.txt", [0.0, 0.01, 0.02])
    np.savetxt(input_dir / "U_imu.txt", np.zeros((3, 3)))
    np.savetxt(input_dir / "t_cam.txt", [0.0, 0.033])
    np.savetxt(input_dir / "led1.txt", [[0.0, 0.0], [0.01, 0.0]])
    np.savetxt(input_dir / "led2.txt", [[0.04, 0.0], [0.05, 0.0]])
    np.savetxt(input_dir / "mask.txt", [1, 1], fmt="%d")
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: prepared_arrays
  imu_timestamps: input/t_imu.txt
  imu_measurements: input/U_imu.txt
  camera_timestamps: input/t_cam.txt
  led1_positions: input/led1.txt
  led2_positions: input/led2.txt
  camera_mask: input/mask.txt
outputs:
  output_dir: run
""".lstrip()
    )

    session = load_session(load_session_config(config_path))
    write_session_diagnostics(session, tmp_path / "run")

    assert session.diagnostics["loader"]["format"] == "prepared_arrays"
    assert (tmp_path / "run" / "session_diagnostics.json").exists()


def test_arthur_loader_reports_missing_required_columns(tmp_path):
    """Malformed Arthur parquet inputs fail with actionable column names."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "time": [0.0, 0.01],
            "Headstage_GyroX": [0, 0],
            "Headstage_GyroY": [0, 0],
            "Headstage_GyroZ": [0, 1],
            "Headstage_AccelX": [0, 0],
            "Headstage_AccelY": [0, 0],
            "Headstage_AccelZ": [0, 0],
        }
    ).to_parquet(data_dir / "imu.parquet")
    pd.DataFrame(
        {
            "time": [0.0, 0.033],
            "xloc": [10.0, 11.0],
            "yloc": [20.0, 20.0],
            "xloc2": [14.0, 15.0],
        }
    ).to_parquet(data_dir / "position.parquet")
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
""".lstrip()
    )

    with pytest.raises(ValueError, match="yloc2"):
        load_session(load_session_config(config_path))


def test_vision_only_led_identity_uses_real_gyro_signal(tmp_path, monkeypatch):
    """Gyro-weighted LED identity must not see zeroed vision-only filter inputs."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    imu_path = data_dir / "imu.parquet"
    position_path = data_dir / "position.parquet"
    pd.DataFrame(
        {
            "time": [100.0, 100.005, 100.010, 100.015, 100.020],
            "Headstage_GyroX": [0, 0, 0, 0, 0],
            "Headstage_GyroY": [0, 0, 0, 0, 0],
            "Headstage_GyroZ": [0, 0, 10, 10, 20],
            "Headstage_AccelX": [0, 0, 0, 0, 0],
            "Headstage_AccelY": [0, 0, 0, 0, 0],
            "Headstage_AccelZ": [0, 0, 0, 0, 0],
        }
    ).to_parquet(imu_path)
    pd.DataFrame(
        {
            "time": [100.0, 100.033, 100.066],
            "xloc": [10.0, 11.0, 12.0],
            "yloc": [20.0, 20.0, 20.0],
            "xloc2": [14.0, 15.0, 16.0],
            "yloc2": [20.0, 20.0, 20.0],
        }
    ).to_parquet(position_path)
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
filter:
  state_mode: vision_only
led_identity:
  mode: auto
  gyro_weight: 1.0
outputs:
  run_safety_checks: false
""".lstrip()
    )
    captured: dict[str, np.ndarray | None] = {}

    def capture_resolver(
        t_cam,
        led1,
        led2,
        mask_cam,
        *,
        led_distance,
        config,
        t_imu=None,
        gyro_z=None,
    ):
        captured["gyro_z"] = None if gyro_z is None else np.asarray(gyro_z).copy()
        return CorrectedLEDIdentity(
            led1=np.asarray(led1),
            led2=np.asarray(led2),
            swapped=np.zeros_like(t_cam, dtype=bool),
            diagnostics={"mode": "auto", "n_swapped": 0},
        )

    monkeypatch.setattr("trodestrack.io.session.resolve_led_identity", capture_resolver)

    session = load_session(load_session_config(config_path))

    assert np.all(session.U_imu == 0.0)
    assert captured["gyro_z"] is not None
    np.testing.assert_allclose(
        captured["gyro_z"],
        np.array([0.0, 10.0, 20.0]) * 0.061 * np.pi / 180.0,
    )


def test_prepared_six_channel_led_identity_uses_yaw_gyro(tmp_path, monkeypatch):
    """Prepared 6-channel arrays store yaw gyro in column 2."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    np.savetxt(input_dir / "t_imu.txt", [0.0, 0.01, 0.02])
    np.savetxt(
        input_dir / "U_imu.txt",
        [
            [100.0, 200.0, 1.0, 0.0, 0.0, 9.81],
            [101.0, 201.0, 2.0, 0.0, 0.0, 9.81],
            [102.0, 202.0, 3.0, 0.0, 0.0, 9.81],
        ],
    )
    np.savetxt(input_dir / "t_cam.txt", [0.0, 0.033])
    np.savetxt(input_dir / "led1.txt", [[0.0, 0.0], [0.01, 0.0]])
    np.savetxt(input_dir / "led2.txt", [[0.04, 0.0], [0.05, 0.0]])
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: prepared_arrays
  imu_timestamps: input/t_imu.txt
  imu_measurements: input/U_imu.txt
  camera_timestamps: input/t_cam.txt
  led1_positions: input/led1.txt
  led2_positions: input/led2.txt
filter:
  state_mode: 2d_cam_6dof_imu_orientation
led_identity:
  mode: auto
  gyro_weight: 1.0
""".lstrip()
    )
    captured: dict[str, np.ndarray | None] = {}

    def capture_resolver(
        t_cam,
        led1,
        led2,
        mask_cam,
        *,
        led_distance,
        config,
        t_imu=None,
        gyro_z=None,
    ):
        captured["gyro_z"] = None if gyro_z is None else np.asarray(gyro_z).copy()
        return CorrectedLEDIdentity(
            led1=np.asarray(led1),
            led2=np.asarray(led2),
            swapped=np.zeros_like(t_cam, dtype=bool),
            diagnostics={"mode": "auto", "n_swapped": 0},
        )

    monkeypatch.setattr("trodestrack.io.session.resolve_led_identity", capture_resolver)

    load_session(load_session_config(config_path))

    np.testing.assert_allclose(captured["gyro_z"], [1.0, 2.0, 3.0])


def test_led_identity_correction_swaps_conf_cam_columns(tmp_path, monkeypatch):
    """LED identity correction must also swap per-LED confidence rows.

    ``conf_cam`` is laid out as ``[c1, c1, c2, c2]`` (per-LED
    confidence replicated across x/y). When ``resolve_led_identity``
    swaps LED1/LED2 for a frame, the EKF needs the confidence
    halves to follow or it ends up trusting the wrong physical LED.
    Probe scenario: a known global swap with low LED1 confidence
    (0.1) and high LED2 confidence (0.9) — after correction the
    physically-LED1 row should carry 0.9 in cols 0/1 and 0.1 in
    cols 2/3.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = 5
    t_imu = np.linspace(0.0, 0.3, n)
    pd.DataFrame(
        {
            "time": t_imu,
            "Headstage_GyroX": np.zeros_like(t_imu),
            "Headstage_GyroY": np.zeros_like(t_imu),
            "Headstage_GyroZ": np.zeros_like(t_imu),
            "Headstage_AccelX": np.zeros_like(t_imu),
            "Headstage_AccelY": np.zeros_like(t_imu),
            "Headstage_AccelZ": np.full_like(t_imu, 9.81),
        }
    ).to_parquet(data_dir / "imu.parquet")

    t_cam = np.linspace(0.0, 0.3, n)
    # Observed labels are swapped relative to physical truth; the
    # auto resolver flips every frame after we set
    # ``initial_state="swapped"``.
    pd.DataFrame(
        {
            "time": t_cam,
            "xloc": [4.0, 5.0, 6.0, 7.0, 8.0],
            "yloc": [0.0, 0.0, 0.0, 0.0, 0.0],
            "xloc2": [0.0, 1.0, 2.0, 3.0, 4.0],
            "yloc2": [0.0, 0.0, 0.0, 0.0, 0.0],
            "led1_likelihood": [0.1] * n,
            "led2_likelihood": [0.9] * n,
        }
    ).to_parquet(data_dir / "position.parquet")

    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
  sample_hold_strategy: none
camera:
  meters_per_pixel: 0.01
  confidence_led1_column: led1_likelihood
  confidence_led2_column: led2_likelihood
filter:
  state_mode: vision_only
led_identity:
  mode: auto
  initial_state: swapped
""".lstrip()
    )

    session = load_session(load_session_config(config_path))

    assert session.diagnostics["led_identity_swapped"].all()
    assert session.conf_cam is not None
    # Every row had its LED1/LED2 confidence swapped, so cols 0/1
    # now carry the (originally LED2) high confidence and cols 2/3
    # carry the (originally LED1) low confidence.
    np.testing.assert_allclose(session.conf_cam[:, 0:2], 0.9)
    np.testing.assert_allclose(session.conf_cam[:, 2:4], 0.1)


def test_prepared_arrays_mask_cam_includes_single_led_frames(tmp_path):
    """Prepared-array fallback mask must include single-LED frames.

    When ``camera_mask`` is omitted but ``led2_positions`` is
    provided, the loader's default mask used to be
    ``finite(led1)`` only, which silently dropped LED2-only frames
    even though the EKF supports single-LED updates. The fallback
    must now match the SpikeGadgets path: a frame is usable when
    either LED is finite. (When ``led2_positions`` is absent
    entirely, ``led2`` is all-NaN and the OR collapses back to
    ``finite(led1)`` as before.)
    """

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    np.savetxt(input_dir / "t_imu.txt", [0.0, 0.05, 0.10, 0.15])
    np.savetxt(input_dir / "U_imu.txt", np.zeros((4, 3)))
    np.savetxt(input_dir / "t_cam.txt", [0.0, 0.033, 0.066])
    # Frame 0: both LEDs finite. Frame 1: LED1 only. Frame 2: LED2 only.
    np.savetxt(input_dir / "led1.txt", [[0.0, 0.0], [0.01, 0.0], [np.nan, np.nan]])
    np.savetxt(input_dir / "led2.txt", [[0.04, 0.0], [np.nan, np.nan], [0.05, 0.0]])
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: prepared_arrays
  imu_timestamps: input/t_imu.txt
  imu_measurements: input/U_imu.txt
  camera_timestamps: input/t_cam.txt
  led1_positions: input/led1.txt
  led2_positions: input/led2.txt
filter:
  state_mode: vision_only
""".lstrip()
    )

    session = load_session(load_session_config(config_path))

    np.testing.assert_array_equal(session.mask_cam, [True, True, True])


def test_arthur_loader_includes_single_led_frames_in_mask(tmp_path):
    """``mask_cam`` must include single-LED frames so the EKF can use them.

    The EKF supports LED1-only and LED2-only updates when
    ``mask_cam`` is True (covered by
    ``test_ekf_partial_observations.py``). The earlier loader
    required *both* LEDs finite, silently dropping partial frames
    before the filter even saw them. The fix is OR rather than
    AND on the per-LED finite check.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = 4
    t_imu = np.linspace(0.0, 0.2, n)
    pd.DataFrame(
        {
            "time": t_imu,
            "Headstage_GyroX": np.zeros_like(t_imu),
            "Headstage_GyroY": np.zeros_like(t_imu),
            "Headstage_GyroZ": np.zeros_like(t_imu),
            "Headstage_AccelX": np.zeros_like(t_imu),
            "Headstage_AccelY": np.zeros_like(t_imu),
            "Headstage_AccelZ": np.full_like(t_imu, 9.81),
        }
    ).to_parquet(data_dir / "imu.parquet")
    # Frame 0: both LEDs finite. Frame 1: LED1 only. Frame 2: LED2
    # only. Frame 3: neither (true dropout).
    pd.DataFrame(
        {
            "time": np.linspace(0.0, 0.2, n),
            "xloc": [1.0, 2.0, np.nan, np.nan],
            "yloc": [1.0, 2.0, np.nan, np.nan],
            "xloc2": [3.0, np.nan, 4.0, np.nan],
            "yloc2": [1.0, np.nan, 2.0, np.nan],
        }
    ).to_parquet(data_dir / "position.parquet")
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
  sample_hold_strategy: none
camera:
  meters_per_pixel: 0.01
filter:
  state_mode: vision_only
""".lstrip()
    )

    session = load_session(load_session_config(config_path))

    np.testing.assert_array_equal(session.mask_cam, [True, True, True, False])


def test_arthur_loader_reports_missing_confidence_column(tmp_path):
    """Configured confidence column missing from parquet → friendly ValueError.

    A ``confidence_led1_column`` / ``confidence_led2_column`` set in
    the YAML used to surface as ``Unexpected error: 'likelihood'``
    from a raw KeyError in ``_load_leds`` after the loader had
    already consumed compute. Pulling the column names into the
    up-front ``_require_columns`` call routes them through the
    shared validator (clean ValueError mentioning the source file).
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = 4
    t_imu = np.linspace(0.0, 0.2, n)
    pd.DataFrame(
        {
            "time": t_imu,
            "Headstage_GyroX": np.zeros_like(t_imu),
            "Headstage_GyroY": np.zeros_like(t_imu),
            "Headstage_GyroZ": np.zeros_like(t_imu),
            "Headstage_AccelX": np.zeros_like(t_imu),
            "Headstage_AccelY": np.zeros_like(t_imu),
            "Headstage_AccelZ": np.full_like(t_imu, 9.81),
        }
    ).to_parquet(data_dir / "imu.parquet")
    pd.DataFrame(
        {
            "time": np.linspace(0.0, 0.2, n),
            "xloc": np.zeros(n),
            "yloc": np.zeros(n),
            "xloc2": np.ones(n),
            "yloc2": np.zeros(n),
            # Note: ``led1_likelihood`` is intentionally absent.
        }
    ).to_parquet(data_dir / "position.parquet")
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
imu:
  run_calibration: false
camera:
  meters_per_pixel: 0.01
  confidence_led1_column: led1_likelihood
  confidence_led2_column: led2_likelihood
filter:
  state_mode: vision_only
""".lstrip()
    )

    with pytest.raises(ValueError, match=r"missing required column.*led1_likelihood"):
        load_session(load_session_config(config_path))


def test_safety_check_uses_all_frames_for_fused_vs_vision_deviation(monkeypatch):
    """Single-LED frames must contribute to the deviation gate.

    Probe: a session with one dual-LED frame and many single-LED
    frames used to skip the fused-vs-vision deviation check on
    every single-LED frame, so a fused trajectory that drifted
    only on single-LED frames passed silently. The dual-LED
    midpoint envelope still requires both LEDs (no midpoint
    otherwise), but the deviation gate now uses every frame in
    ``mask_cam`` where fused and vision positions are finite.
    """

    n = 25
    t_cam = np.linspace(0.0, 1.0, n)
    t_imu = np.linspace(0.0, 1.0, 2 * n - 1)
    center = np.column_stack([0.05 * t_cam, np.zeros_like(t_cam)])
    led1 = center - np.array([0.02, 0.0])
    led2 = center + np.array([0.02, 0.0])
    # Frames 1..n-1: only LED1 finite. Frame 0 keeps both for the
    # camera envelope. The OR'd ``mask_cam`` keeps every frame
    # usable, which is why the EKF — and the fixed safety check —
    # still process them.
    led2[1:] = np.nan
    config = SessionConfig.model_validate(
        {
            "inputs": {
                "format": "spikegadgets_trodes",
                "imu_file": "imu.parquet",
                "position_file": "position.parquet",
            },
            "outputs": {
                "safety_max_position_deviation_m": 0.1,
                "safety_p95_position_deviation_m": 0.1,
                "safety_max_speed_mps": 3.0,
                "safety_min_dual_led_frames": 1,
            },
        }
    )
    mask_cam = np.isfinite(led1).all(axis=1) | np.isfinite(led2).all(axis=1)
    session = PreparedSession(
        t_imu=t_imu,
        U_imu=np.zeros((len(t_imu), 3)),
        t_cam=t_cam,
        Z_cam_led1=led1,
        Z_cam_led2=led2,
        mask_cam=mask_cam,
        conf_cam=None,
        led_distance=0.04,
        diagnostics={},
        config=config,
    )

    fused_means = np.zeros((n, 8))
    fused_means[:, :2] = center
    # Drift the fused trajectory by 0.5 m on every single-LED frame
    # (frames 1..n-1). The dual-LED midpoint stays within bounds so
    # the camera-envelope and speed gates pass; only the deviation
    # gate can catch this.
    fused_means[1:, 0] += 0.5
    vision_means = np.zeros((n, 5))
    vision_means[:, :2] = center

    filter_result = EKFResult(
        filtered_means=fused_means,
        filtered_covariances=np.repeat(np.eye(8)[None, :, :], n, axis=0),
        predicted_means=fused_means,
        predicted_covariances=np.repeat(np.eye(8)[None, :, :], n, axis=0),
        marginal_loglik=-1.0,
        estimated_led_distance=0.04,
    )

    def fake_vision_filter(*args, **kwargs):
        return EKFResult(
            filtered_means=vision_means,
            filtered_covariances=np.repeat(np.eye(5)[None, :, :], n, axis=0),
            predicted_means=vision_means,
            predicted_covariances=np.repeat(np.eye(5)[None, :, :], n, axis=0),
            marginal_loglik=-2.0,
            estimated_led_distance=0.04,
        )

    monkeypatch.setattr(
        "trodestrack.io.session.extended_kalman_filter", fake_vision_filter
    )

    report = run_real_data_safety_check(
        session,
        EKFConfig(state_mode="2d_full"),
        filter_result,
    )

    assert not report.passed, (
        "Drift on single-LED frames must trip the deviation gate; "
        f"reported max deviation was {report.max_vision_position_deviation_m}"
    )
    assert report.max_vision_position_deviation_m == pytest.approx(0.5)
    # Diagnostic counts make the dual-LED coverage visible to users.
    assert report.dual_led_frame_count == 1
    assert report.deviation_frame_count == n


def test_safety_check_requires_minimum_dual_led_frames(monkeypatch):
    """Below the configured dual-LED minimum, the camera envelope is unreliable.

    Default ``safety_min_dual_led_frames=20`` rejects sessions whose
    dual-LED coverage is too sparse to produce a meaningful camera
    envelope. The test forces a 5-dual-LED scenario and asserts
    the loader-style ValueError fires before any pass/fail flag.
    """

    n = 25
    t_cam = np.linspace(0.0, 1.0, n)
    t_imu = np.linspace(0.0, 1.0, 2 * n - 1)
    center = np.column_stack([0.05 * t_cam, np.zeros_like(t_cam)])
    led1 = center - np.array([0.02, 0.0])
    led2 = center + np.array([0.02, 0.0])
    # Only 5 dual-LED frames; rest are LED1-only.
    led2[5:] = np.nan
    config = SessionConfig.model_validate(
        {
            "inputs": {
                "format": "spikegadgets_trodes",
                "imu_file": "imu.parquet",
                "position_file": "position.parquet",
            },
            "outputs": {
                # Default is 20; explicit here for the test contract.
                "safety_min_dual_led_frames": 20,
            },
        }
    )
    mask_cam = np.isfinite(led1).all(axis=1) | np.isfinite(led2).all(axis=1)
    session = PreparedSession(
        t_imu=t_imu,
        U_imu=np.zeros((len(t_imu), 3)),
        t_cam=t_cam,
        Z_cam_led1=led1,
        Z_cam_led2=led2,
        mask_cam=mask_cam,
        conf_cam=None,
        led_distance=0.04,
        diagnostics={},
        config=config,
    )
    filter_result = EKFResult(
        filtered_means=np.zeros((n, 8)),
        filtered_covariances=np.repeat(np.eye(8)[None, :, :], n, axis=0),
        predicted_means=np.zeros((n, 8)),
        predicted_covariances=np.repeat(np.eye(8)[None, :, :], n, axis=0),
        marginal_loglik=-1.0,
        estimated_led_distance=0.04,
    )

    with pytest.raises(ValueError, match=r"at least 20 finite dual-LED frame"):
        run_real_data_safety_check(
            session, EKFConfig(state_mode="2d_full"), filter_result
        )


def test_safety_check_rejects_fused_drift_from_vision_baseline(monkeypatch):
    """A broad envelope alone should not hide fused-vs-vision drift."""

    t_cam = np.linspace(0.0, 1.0, 11)
    t_imu = np.linspace(0.0, 1.0, 21)
    center = np.column_stack([0.05 * t_cam, np.zeros_like(t_cam)])
    led1 = center - np.array([0.02, 0.0])
    led2 = center + np.array([0.02, 0.0])
    config = SessionConfig.model_validate(
        {
            "inputs": {
                "format": "spikegadgets_trodes",
                "imu_file": "imu.parquet",
                "position_file": "position.parquet",
            },
            "outputs": {
                "safety_max_position_deviation_m": 0.1,
                "safety_p95_position_deviation_m": 0.1,
                "safety_max_speed_mps": 3.0,
                "safety_min_dual_led_frames": 1,
            },
        }
    )
    session = PreparedSession(
        t_imu=t_imu,
        U_imu=np.zeros((len(t_imu), 3)),
        t_cam=t_cam,
        Z_cam_led1=led1,
        Z_cam_led2=led2,
        mask_cam=np.ones(len(t_cam), dtype=bool),
        conf_cam=None,
        led_distance=0.04,
        diagnostics={},
        config=config,
    )
    fused_means = np.zeros((len(t_cam), 8))
    fused_means[:, :2] = center + np.array([0.2, 0.0])
    vision_means = np.zeros((len(t_cam), 5))
    vision_means[:, :2] = center

    filter_result = EKFResult(
        filtered_means=fused_means,
        filtered_covariances=np.repeat(np.eye(8)[None, :, :], len(t_cam), axis=0),
        predicted_means=fused_means,
        predicted_covariances=np.repeat(np.eye(8)[None, :, :], len(t_cam), axis=0),
        marginal_loglik=-1.0,
        estimated_led_distance=0.04,
    )

    captured_vision_config = {}

    def fake_vision_filter(ekf_config, *args, **kwargs):
        captured_vision_config["config"] = ekf_config
        return EKFResult(
            filtered_means=vision_means,
            filtered_covariances=np.repeat(np.eye(5)[None, :, :], len(t_cam), axis=0),
            predicted_means=vision_means,
            predicted_covariances=np.repeat(np.eye(5)[None, :, :], len(t_cam), axis=0),
            marginal_loglik=-2.0,
            estimated_led_distance=0.04,
        )

    monkeypatch.setattr(
        "trodestrack.io.session.extended_kalman_filter", fake_vision_filter
    )

    report = run_real_data_safety_check(
        session,
        EKFConfig(state_mode="2d_full"),
        filter_result,
    )

    assert not report.passed
    assert captured_vision_config["config"].state_mode == "vision_only"
    assert captured_vision_config["config"].enable_zupt is False
    assert "vision-only baseline" in report.message
    assert report.max_vision_position_deviation_m == pytest.approx(0.2)
    assert report.p95_vision_position_deviation_m == pytest.approx(0.2)


def _calibration_report(
    *,
    gravity_body: tuple[float, float, float] = (0.0, 0.0, 9.80665),
    yaw_correlation: float = 0.9,
    accel_correlation: float = 0.9,
) -> ImuCalibrationReport:
    return ImuCalibrationReport(
        gyro_bias_z=0.0,
        accel_gravity_body=np.asarray(gravity_body, dtype=float),
        stationary_fraction=0.5,
        stationary_samples=100,
        yaw_rate_fit=LagFit(
            lag_s=0.0,
            correlation=yaw_correlation,
            slope=1.0,
            intercept=0.0,
            r2=yaw_correlation**2,
            n_samples=100,
        ),
        accel_axis_diagnostics=(
            AxisSignDiagnostic(
                target_axis="body_x",
                imu_axis="x",
                sign=1,
                lag_s=0.0,
                correlation=accel_correlation,
                n_samples=100,
            ),
            AxisSignDiagnostic(
                target_axis="body_y",
                imu_axis="y",
                sign=1,
                lag_s=0.0,
                correlation=accel_correlation,
                n_samples=100,
            ),
        ),
    )


def _calibration_config(
    *,
    state_mode: str = "2d_full",
    enable_experimental_accel_translation: bool | None = None,
) -> SessionConfig:
    filter_config: dict[str, object] = {"state_mode": state_mode}
    if enable_experimental_accel_translation is not None:
        filter_config["enable_experimental_accel_translation"] = (
            enable_experimental_accel_translation
        )
    return SessionConfig.model_validate(
        {
            "inputs": {
                "format": "spikegadgets_trodes",
                "imu_file": "imu.parquet",
                "position_file": "position.parquet",
            },
            "filter": filter_config,
        }
    )


def test_translation_calibration_rejects_horizontal_gravity() -> None:
    """Tilted stationary gravity should block accel-driven translation."""

    report = _calibration_report(gravity_body=(1.5, 0.0, 9.6))
    config = _calibration_config(state_mode="2d_cam_3d_imu")

    with pytest.raises(ValueError, match="horizontal component"):
        _validate_calibration_for_fusion(report, config)


def test_translation_calibration_rejects_weak_accel_axis_alignment() -> None:
    """Weak camera/IMU acceleration matches should block translation fusion."""

    report = _calibration_report(accel_correlation=0.2)
    config = _calibration_config(state_mode="2d_full")

    with pytest.raises(ValueError, match="weakly matches"):
        _validate_calibration_for_fusion(report, config)


def test_yaw_calibration_rejects_negative_correlation_with_field_guidance() -> None:
    """A flipped gyro_z sign should fail calibration, not pass by absolute value."""

    report = _calibration_report(yaw_correlation=-0.9)
    config = _calibration_config(state_mode="2d_cam_6dof_imu_orientation")

    with pytest.raises(ValueError) as exc_info:
        _validate_calibration_for_fusion(report, config)

    message = str(exc_info.value)
    assert "correlation=-0.900" in message
    assert "imu.axis_map['gyro_z']" in message
    assert "imu.axis_signs['gyro_z']" in message
    assert "imu.time_offset_s" in message
    assert "camera.time_offset_s" in message
    assert "filter.state_mode: vision_only" in message


def test_orientation_without_accel_translation_skips_translation_calibration() -> None:
    """Orientation fusion can pass when only translation-specific checks fail."""

    report = _calibration_report(
        gravity_body=(1.5, 0.0, 9.6),
        accel_correlation=0.2,
    )
    config = _calibration_config(
        state_mode="2d_cam_6dof_imu_orientation",
        enable_experimental_accel_translation=False,
    )

    _validate_calibration_for_fusion(report, config)


def test_orientation_with_accel_translation_uses_translation_calibration() -> None:
    """Enabling experimental translation in orientation mode should be gated."""

    report = _calibration_report(accel_correlation=0.2)
    config = _calibration_config(
        state_mode="2d_cam_6dof_imu_orientation",
        enable_experimental_accel_translation=True,
    )

    with pytest.raises(ValueError, match="weakly matches"):
        _validate_calibration_for_fusion(report, config)


def test_imu_calibration_uses_corrected_led_identity(tmp_path, monkeypatch) -> None:
    """Calibration diagnostics should inspect the same LED identities as filtering."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    imu_path = data_dir / "imu.parquet"
    position_path = data_dir / "position.parquet"
    t_imu = np.linspace(100.0, 100.1, 6)
    pd.DataFrame(
        {
            "time": t_imu,
            "Headstage_GyroX": np.arange(6),
            "Headstage_GyroY": np.arange(10, 16),
            "Headstage_GyroZ": np.arange(20, 26),
            "Headstage_AccelX": np.arange(30, 36),
            "Headstage_AccelY": np.arange(40, 46),
            "Headstage_AccelZ": np.arange(50, 56),
        }
    ).to_parquet(imu_path)
    t_cam = np.linspace(100.0, 100.1, 6)
    center = np.column_stack([0.01 * np.arange(6), np.zeros(6)])
    led1_true = center - np.array([0.02, 0.0])
    led2_true = center + np.array([0.02, 0.0])
    pd.DataFrame(
        {
            "time": t_cam,
            # Store a whole-session global swap in the raw parquet columns.
            "xloc": led2_true[:, 0],
            "yloc": led2_true[:, 1],
            "xloc2": led1_true[:, 0],
            "yloc2": led1_true[:, 1],
        }
    ).to_parquet(position_path)
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
inputs:
  format: spikegadgets_trodes
  imu_file: data/imu.parquet
  position_file: data/position.parquet
camera:
  meters_per_pixel: 1.0
filter:
  state_mode: 2d_cam_3d_imu
led_identity:
  mode: auto
  initial_state: swapped
outputs:
  run_safety_checks: false
""".lstrip()
    )
    captured: dict[str, np.ndarray] = {}

    def fake_calibration(*, led1, led2, **kwargs):
        captured["led1"] = np.asarray(led1).copy()
        captured["led2"] = np.asarray(led2).copy()
        return _calibration_report()

    monkeypatch.setattr(
        "trodestrack.io.session.run_imu_calibration_diagnostics",
        fake_calibration,
    )

    session = load_session(load_session_config(config_path))

    np.testing.assert_allclose(session.Z_cam_led1, led1_true)
    np.testing.assert_allclose(session.Z_cam_led2, led2_true)
    np.testing.assert_allclose(captured["led1"], led1_true)
    np.testing.assert_allclose(captured["led2"], led2_true)
    assert session.diagnostics["imu_calibration_led_identity_applied"] is True


def test_median_led_distance_raises_when_no_dual_led_frames():
    """Auto-detection must fail loudly when no dual-LED frames are valid."""

    led1 = np.ones((5, 2))
    led2 = np.ones((5, 2))
    mask = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError, match="auto-detect LED spacing"):
        _median_led_distance(led1, led2, mask)


def test_index_or_time_column_raises_when_time_missing():
    """Missing 'time' column must raise; present 'time' returns its values."""

    df_missing = pd.DataFrame({"x": [0.0, 1.0], "y": [2.0, 3.0]})
    with pytest.raises(ValueError, match="missing required 'time' column"):
        _index_or_time_column(df_missing, source="test")

    df_with_time = pd.DataFrame({"time": [0.0, 0.01, 0.02], "x": [1.0, 2.0, 3.0]})
    result = _index_or_time_column(df_with_time, source="test")
    np.testing.assert_array_equal(result, np.array([0.0, 0.01, 0.02]))
