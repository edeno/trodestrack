"""Parity gate: pin post-refactor session-loading output.

These tests lock in the exact numerical content of ``PreparedSession``
fields produced by ``load_session`` for both legacy
formats (``spikegadgets_trodes`` and ``prepared_arrays``). A future
refactor that drifts the loader's numerics — pixel→meter scaling, IMU
SI conversion, sample-hold dedup, clock alignment, or state-mode
projection — fails this gate and must update the pinned values
deliberately.

Pinned values are derived from first principles given the YAML config
and the fixture inputs, so the assertions document the expected
formulas rather than capture an opaque snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trodestrack.config import load_session_config
from trodestrack.io import load_session

DEG_TO_RAD = np.pi / 180.0


def test_spikegadgets_trodes_parity(tmp_path) -> None:
    """``spikegadgets_trodes`` loader output is bitwise-stable.

    Locks the four post-refactor numerical surfaces:
    ``t_imu``, ``U_imu``, ``t_cam``, ``Z_cam_led1``, ``Z_cam_led2``.
    The fixture matches the small case used elsewhere in this
    directory so the test setup is comparable to neighboring tests.
    """

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

    # Clock alignment: ``t_start = min(imu_unix[0], cam_unix[0]) = 100.0``;
    # IMU is sample-hold-deduped on gyro_z change to indices [0, 2, 4].
    # Subtract from the literal-100.0 base to reproduce the same
    # float-precision result the loader computes (an analytic
    # ``[0.0, 0.010, 0.020]`` literal differs by ~5e-15 due to IEEE
    # 754 rounding).
    expected_t_imu = np.array([100.0, 100.010, 100.020]) - 100.0
    expected_t_cam = np.array([100.0, 100.033, 100.066]) - 100.0
    np.testing.assert_array_equal(session.t_imu, expected_t_imu)
    np.testing.assert_array_equal(session.t_cam, expected_t_cam)

    # ``state_mode='2d_cam_3d_imu'`` projects 6-channel U_full to
    # ``[gyro_z, accel_x, accel_y, accel_z]``. Constants come from
    # IMUConfig defaults (gyro_scale_dps_per_lsb=0.061,
    # accel_scale_g_per_lsb=0.000061, gravity_mps2=9.80665).
    gyro_scale = 0.061 * DEG_TO_RAD
    accel_scale = 0.000061 * 9.80665
    expected_U_imu = np.array(
        [
            [0 * gyro_scale, 100 * accel_scale, 200 * accel_scale, 300 * accel_scale],
            [
                10 * gyro_scale,
                110 * accel_scale,
                210 * accel_scale,
                310 * accel_scale,
            ],
            [
                20 * gyro_scale,
                120 * accel_scale,
                220 * accel_scale,
                320 * accel_scale,
            ],
        ]
    )
    np.testing.assert_array_equal(session.U_imu, expected_U_imu)

    # Pixel→meter scaling: pixel coords × ``camera.meters_per_pixel``.
    expected_led1 = np.array([[10.0, 20.0], [11.0, 20.0], [12.0, 20.0]]) * 0.01
    expected_led2 = np.array([[14.0, 20.0], [15.0, 20.0], [16.0, 20.0]]) * 0.01
    np.testing.assert_array_equal(session.Z_cam_led1, expected_led1)
    np.testing.assert_array_equal(session.Z_cam_led2, expected_led2)

    np.testing.assert_array_equal(session.mask_cam, [True, True, True])


def test_prepared_arrays_parity(tmp_path) -> None:
    """``prepared_arrays`` loader passes inputs through verbatim.

    The pixel→meter / IMU helpers don't touch this path; this
    test confirms it.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    t_imu_path = data_dir / "t_imu.txt"
    U_imu_path = data_dir / "U_imu.txt"
    t_cam_path = data_dir / "t_cam.txt"
    led1_path = data_dir / "led1.txt"
    led2_path = data_dir / "led2.txt"

    t_imu_values = np.array([0.0, 0.010, 0.020, 0.030])
    U_imu_values = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]
    )
    t_cam_values = np.array([0.0, 0.033, 0.066])
    led1_values = np.array([[0.10, 0.20], [0.11, 0.20], [0.12, 0.20]])
    led2_values = np.array([[0.14, 0.20], [0.15, 0.20], [0.16, 0.20]])

    np.savetxt(t_imu_path, t_imu_values)
    np.savetxt(U_imu_path, U_imu_values)
    np.savetxt(t_cam_path, t_cam_values)
    np.savetxt(led1_path, led1_values)
    np.savetxt(led2_path, led2_values)

    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        f"""
inputs:
  format: prepared_arrays
  imu_timestamps: {t_imu_path}
  imu_measurements: {U_imu_path}
  camera_timestamps: {t_cam_path}
  led1_positions: {led1_path}
  led2_positions: {led2_path}
filter:
  state_mode: 2d_cam_3d_imu
outputs:
  run_safety_checks: false
""".lstrip()
    )

    session = load_session(load_session_config(config_path))

    np.testing.assert_array_equal(session.t_imu, t_imu_values)
    np.testing.assert_array_equal(session.U_imu, U_imu_values)
    np.testing.assert_array_equal(session.t_cam, t_cam_values)
    np.testing.assert_array_equal(session.Z_cam_led1, led1_values)
    np.testing.assert_array_equal(session.Z_cam_led2, led2_values)
