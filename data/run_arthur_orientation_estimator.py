"""Run the standalone orientation estimator on the bundled Arthur session.

Usage:
    uv run python data/run_arthur_orientation_estimator.py
"""

from pathlib import Path

from load_arthur_session import load_arthur_session

from trodestrack.models.orientation import (
    OrientationEstimatorConfig,
    estimate_orientation,
    format_orientation_diagnostics,
)


def main() -> None:
    """Load Arthur data and print orientation-only estimator diagnostics."""

    script_dir = Path(__file__).parent
    data = load_arthur_session(
        position_file=str(script_dir / "arthur20220324_position_info.parquet"),
        imu_file=str(script_dir / "arthur20220324_imu_info.parquet"),
        imu_mode="3d",
        verbose=True,
    )

    result = estimate_orientation(
        t_imu=data.t_imu,
        gyro_xyz=data.U_imu[:, :3],
        accel_xyz=data.U_imu[:, 3:6],
        t_cam=data.t_cam,
        led1=data.Z_cam_led1,
        led2=data.Z_cam_led2,
        config=OrientationEstimatorConfig(
            camera_speed_threshold_m_s=0.05,
        ),
    )

    print()
    print(format_orientation_diagnostics(result))


if __name__ == "__main__":
    main()
