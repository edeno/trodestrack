"""Run IMU calibration diagnostics on the bundled Arthur session.

Usage:
    cd data/
    uv run python diagnose_arthur_imu_calibration.py
"""

from pathlib import Path

import numpy as np
from load_arthur_session import load_arthur_session

from trodestrack.qa.imu_calibration import (
    format_imu_calibration_report,
    run_imu_calibration_diagnostics,
)


def main() -> None:
    """Load Arthur data and print offline IMU calibration diagnostics."""

    script_dir = Path(__file__).parent
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220324_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220324_imu_info.parquet"),
        imu_mode="3d",
        verbose=True,
    )

    # load_arthur_session(mode="3d") exposes full 6-axis IMU:
    # [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z].
    gyro_z = data.U_imu[:, 2]
    accel_xyz = data.U_imu[:, 3:6]
    report = run_imu_calibration_diagnostics(
        t_imu=data.t_imu,
        gyro_z=gyro_z,
        accel_xyz=accel_xyz,
        t_cam=data.t_cam,
        led1=data.Z_cam_led1,
        led2=data.Z_cam_led2,
        candidate_lags_s=np.linspace(-0.5, 0.5, 101),
    )

    print()
    print(format_imu_calibration_report(report))


if __name__ == "__main__":
    main()
