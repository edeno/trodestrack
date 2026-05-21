"""3D quaternion UKF rejection-contract test.

Pins the current behavior: ``unscented_kalman_filter`` rejects every
quaternion-orientation state mode (``3d_quat`` and ``3d_cam_6dof_imu``)
with ``NotImplementedError`` because sigma-point quaternion mean and
covariance handling is not yet implemented (see
``unscented_kalman_filter`` docstring and
``tests/filters/test_imu_shape_validation.py``).

The rejection happens at the very top of the entry point (see ``ukf.py``
around line 1039), before any input tensor is inspected — so this is a
single behavioral contract, not five. When the UKF gains 3D quaternion
support, swap this rejection-contract test for the five analytic tests
that mirror ``test_ekf_3d_analytic.py``. The intended future bounds are:

- ``stationary pitch=10° roll=5°``: pitch/roll within ~2.4° of truth
  (~20% looser than EKF's 2°), quaternion norm 1 ± ~1.2e-6.
- ``yaw-only 30 deg/s, gyro_z bias 0.05``: bias RMSE within ~6e-3 rad/s
  after the bias-observability window.
- ``5 s camera dropout``: drift ≤ ~0.18 m (~20% looser than EKF's 0.15 m).
- ``4D NEES consistency``: mean NEES in ~[1.0, 9.6].
- ``perfect-input idempotence``: posterior matches truth with ``rtol≈1.2e-5``.

The ~20% loosening accounts for sigma-point spread.
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter


def test_ukf_3d_quaternion_modes_raise_not_implemented() -> None:
    """3D-quaternion UKF entry point rejects before consuming any input.

    When this test fails because the UKF gained 3D quaternion support,
    delete it and add the five analytic tests documented in the module
    docstring above (mirroring ``test_ekf_3d_analytic.py``).
    """

    config = UKFConfig(
        state_mode="3d_cam_6dof_imu",
        measurement_noise_pos=5e-3**2,
        process_noise_gyro_bias=1e-7,
        enable_experimental_accel_translation=True,
        enable_zupt=False,
        use_gravity_orientation_update=True,
        use_mahalanobis_gating=False,
    )
    # Zero-filled inputs of the right shape — the entry point raises
    # before any value is consumed, so contents are irrelevant.
    n_imu, n_cam = 50, 15
    with pytest.raises(NotImplementedError, match="quaternion"):
        unscented_kalman_filter(
            config,
            np.linspace(0.0, 0.5, n_imu, dtype=np.float64),
            np.zeros((n_imu, 6), dtype=np.float64),
            np.linspace(0.0, 0.5, n_cam, dtype=np.float64),
            np.zeros((n_cam, 3), dtype=np.float64),
            np.zeros((n_cam, 3), dtype=np.float64),
            np.ones(n_cam, dtype=bool),
        )
