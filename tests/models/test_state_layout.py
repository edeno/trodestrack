"""Tests for state layout definitions and properties."""

import pytest

from trodestrack.models.state_layout import (
    LAYOUT_2D_CAM_3D_IMU,
    LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION,
    LAYOUT_2D_FULL,
    LAYOUT_3D_CAM_6DOF_IMU,
    LAYOUT_3D_EULER,
    LAYOUT_3D_QUAT,
    LAYOUT_REGISTRY,
    LAYOUT_VISION_ONLY,
    get_heading_index,
    get_layout,
)

# =============================================================================
# StateLayout Properties
# =============================================================================


def test_layout_2d_full_properties():
    """Test properties of standard 2D layout."""
    layout = LAYOUT_2D_FULL

    assert layout.n == 8
    assert layout.pos_idx == (0, 1)
    assert layout.vel_idx == (2, 3)
    assert layout.heading_idx == 4
    assert layout.bias_gyro_idx == (5,)
    assert layout.bias_accel_idx == (6, 7)
    assert layout.has_biases is True
    assert layout.spatial_dim == 2
    assert layout.has_heading_2d is True
    assert layout.has_orientation_3d is False


def test_layout_vision_only_properties():
    """Test properties of vision-only layout (no biases)."""
    layout = LAYOUT_VISION_ONLY

    assert layout.n == 5
    assert layout.pos_idx == (0, 1)
    assert layout.vel_idx == (2, 3)
    assert layout.heading_idx == 4
    assert layout.bias_gyro_idx == ()
    assert layout.bias_accel_idx == ()
    assert layout.has_biases is False
    assert layout.spatial_dim == 2
    assert layout.has_heading_2d is True
    assert layout.has_orientation_3d is False


def test_layout_2d_cam_3d_imu_properties():
    """Test properties of 2D camera + 3D IMU layout."""
    layout = LAYOUT_2D_CAM_3D_IMU

    assert layout.n == 10
    assert layout.pos_idx == (0, 1)
    assert layout.vel_idx == (2, 3, 4)  # 3D velocity (vx, vy, vz)
    assert layout.heading_idx == 5
    assert layout.bias_gyro_idx == (6,)
    assert layout.bias_accel_idx == (7, 8, 9)  # 3D accel bias (b_ax, b_ay, b_az)
    assert layout.has_biases is True
    assert layout.spatial_dim == 2  # Position is still 2D
    assert layout.has_heading_2d is True
    assert layout.has_orientation_3d is False


def test_layout_2d_cam_6dof_imu_orientation_properties():
    """Test properties of 2D camera + 6-DOF IMU orientation layout."""
    layout = LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION

    assert layout.n == 14
    assert layout.pos_idx == (0, 1)
    assert layout.vel_idx == (2, 3)
    assert layout.heading_idx == (4, 5, 6, 7)
    assert layout.bias_gyro_idx == (8, 9, 10)
    assert layout.bias_accel_idx == (11, 12, 13)
    assert layout.has_biases is True
    assert layout.spatial_dim == 2
    assert layout.has_heading_2d is False
    assert layout.has_orientation_3d is True
    assert layout.has_quaternion_orientation is True


def test_layout_3d_euler_properties():
    """Test properties of 3D Euler angle layout."""
    layout = LAYOUT_3D_EULER

    assert layout.n == 15
    assert layout.pos_idx == (0, 1, 2)
    assert layout.vel_idx == (3, 4, 5)
    assert layout.heading_idx == (6, 7, 8)  # (roll, pitch, yaw)
    assert layout.bias_gyro_idx == (9, 10, 11)
    assert layout.bias_accel_idx == (12, 13, 14)
    assert layout.has_biases is True
    assert layout.spatial_dim == 3
    assert layout.has_heading_2d is False
    assert layout.has_orientation_3d is True


def test_layout_3d_quat_properties():
    """Test properties of 3D quaternion layout."""
    layout = LAYOUT_3D_QUAT

    assert layout.n == 16
    assert layout.pos_idx == (0, 1, 2)
    assert layout.vel_idx == (3, 4, 5)
    assert layout.heading_idx == (6, 7, 8, 9)  # (qw, qx, qy, qz)
    assert layout.bias_gyro_idx == (10, 11, 12)
    assert layout.bias_accel_idx == (13, 14, 15)
    assert layout.has_biases is True
    assert layout.spatial_dim == 3
    assert layout.has_heading_2d is False
    assert layout.has_orientation_3d is True
    assert layout.has_quaternion_orientation is True


def test_layout_3d_cam_6dof_imu_matches_3d_quaternion_structure():
    """The plan-facing 3D mode uses the 16D quaternion layout."""
    assert LAYOUT_3D_CAM_6DOF_IMU == LAYOUT_3D_QUAT
    assert LAYOUT_3D_CAM_6DOF_IMU is not LAYOUT_3D_QUAT
    assert get_layout("3d_cam_6dof_imu") is LAYOUT_3D_CAM_6DOF_IMU


# =============================================================================
# Index Consistency Tests (Critical for M5)
# =============================================================================


def test_layout_2d_cam_3d_imu_velocity_indices():
    """Verify 3D velocity indices for 2D cam + 3D IMU layout."""
    layout = LAYOUT_2D_CAM_3D_IMU

    # Velocity should be 3D: (vx, vy, vz)
    assert len(layout.vel_idx) == 3
    assert layout.vel_idx == (2, 3, 4)

    # Verify indices are consecutive
    assert layout.vel_idx[1] == layout.vel_idx[0] + 1
    assert layout.vel_idx[2] == layout.vel_idx[1] + 1


def test_layout_2d_cam_3d_imu_accel_bias_indices():
    """Verify 3D accel bias indices for 2D cam + 3D IMU layout."""
    layout = LAYOUT_2D_CAM_3D_IMU

    # Accel bias should be 3D: (b_ax, b_ay, b_az)
    assert len(layout.bias_accel_idx) == 3
    assert layout.bias_accel_idx == (7, 8, 9)

    # Verify indices are consecutive
    assert layout.bias_accel_idx[1] == layout.bias_accel_idx[0] + 1
    assert layout.bias_accel_idx[2] == layout.bias_accel_idx[1] + 1


def test_layout_2d_cam_3d_imu_state_vector_order():
    """Verify state vector ordering: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]."""
    layout = LAYOUT_2D_CAM_3D_IMU

    # Expected state: [x, y, vx, vy, vz, θ, b_gz, b_ax, b_ay, b_az]
    # Verify position indices
    assert layout.pos_idx[0] == 0  # x
    assert layout.pos_idx[1] == 1  # y

    # Verify velocity indices
    assert layout.vel_idx[0] == 2  # vx
    assert layout.vel_idx[1] == 3  # vy
    assert layout.vel_idx[2] == 4  # vz

    # Verify heading index
    assert layout.heading_idx == 5  # θ

    # Verify gyro bias index
    assert layout.bias_gyro_idx[0] == 6  # b_gz

    # Verify accel bias indices
    assert layout.bias_accel_idx[0] == 7  # b_ax
    assert layout.bias_accel_idx[1] == 8  # b_ay
    assert layout.bias_accel_idx[2] == 9  # b_az


def test_all_layouts_have_no_overlapping_indices():
    """Verify that all layouts have no overlapping state indices."""
    layouts_to_test = [
        LAYOUT_2D_FULL,
        LAYOUT_VISION_ONLY,
        LAYOUT_2D_CAM_3D_IMU,
        LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION,
        LAYOUT_3D_EULER,
        LAYOUT_3D_QUAT,
        LAYOUT_3D_CAM_6DOF_IMU,
    ]

    for layout in layouts_to_test:
        # Collect all indices
        all_indices = set()

        # Add position indices
        all_indices.update(layout.pos_idx)

        # Add velocity indices
        all_indices.update(layout.vel_idx)

        # Add heading indices
        if isinstance(layout.heading_idx, int):
            all_indices.add(layout.heading_idx)
        else:
            all_indices.update(layout.heading_idx)

        # Add gyro bias indices
        all_indices.update(layout.bias_gyro_idx)

        # Add accel bias indices
        all_indices.update(layout.bias_accel_idx)

        # Verify total count matches state dimension
        assert len(all_indices) == layout.n, f"Layout {layout} has duplicate indices"

        # Verify all indices are in range [0, n)
        assert min(all_indices) == 0, f"Layout {layout} doesn't start at index 0"
        assert max(all_indices) == layout.n - 1, f"Layout {layout} has gap in indices"


# =============================================================================
# Helper Functions
# =============================================================================


def test_get_heading_index_for_2d_layouts():
    """Test get_heading_index() for 2D heading layouts."""
    assert get_heading_index(LAYOUT_2D_FULL) == 4
    assert get_heading_index(LAYOUT_VISION_ONLY) == 4
    assert get_heading_index(LAYOUT_2D_CAM_3D_IMU) == 5


def test_get_heading_index_raises_for_3d_layouts():
    """Test get_heading_index() raises for 3D orientation layouts."""
    with pytest.raises(NotImplementedError, match="not a single scalar"):
        get_heading_index(LAYOUT_3D_EULER)

    with pytest.raises(NotImplementedError, match="not a single scalar"):
        get_heading_index(LAYOUT_3D_QUAT)

    with pytest.raises(NotImplementedError, match="not a single scalar"):
        get_heading_index(LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION)


def test_get_layout_by_name():
    """Test get_layout() retrieves correct layout by name."""
    assert get_layout("2d_full") is LAYOUT_2D_FULL
    assert get_layout("vision_only") is LAYOUT_VISION_ONLY
    assert get_layout("imu_only") is LAYOUT_2D_FULL  # Same as 2d_full
    assert get_layout("2d_cam_3d_imu") is LAYOUT_2D_CAM_3D_IMU
    assert (
        get_layout("2d_cam_6dof_imu_orientation") is LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION
    )
    assert get_layout("3d_euler") is LAYOUT_3D_EULER
    assert get_layout("3d_quat") is LAYOUT_3D_QUAT
    assert get_layout("3d_cam_6dof_imu") is LAYOUT_3D_CAM_6DOF_IMU


def test_get_layout_raises_for_unknown_name():
    """Test get_layout() raises KeyError for unknown layout name."""
    with pytest.raises(KeyError):
        get_layout("unknown_layout")


def test_layout_registry_completeness():
    """Test that LAYOUT_REGISTRY contains all expected layouts."""
    expected_keys = [
        "2d_full",
        "vision_only",
        "imu_only",
        "2d_cam_3d_imu",
        "2d_cam_6dof_imu_orientation",
        "3d_euler",
        "3d_quat",
        "3d_cam_6dof_imu",
    ]

    for key in expected_keys:
        assert key in LAYOUT_REGISTRY, f"Missing layout: {key}"


# =============================================================================
# Compatibility Tests (Verify process_noise.py assumptions)
# =============================================================================


def test_process_noise_can_infer_n_accel_from_layouts():
    """Verify process_noise.py inference logic: len(bias_accel_idx) → n_accel."""
    # 2D layouts should have 2 accel bias terms
    assert len(LAYOUT_2D_FULL.bias_accel_idx) == 2
    assert len(LAYOUT_VISION_ONLY.bias_accel_idx) == 0  # No biases

    # 2D cam + 3D IMU should have 3 accel bias terms
    assert len(LAYOUT_2D_CAM_3D_IMU.bias_accel_idx) == 3
    assert len(LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION.bias_accel_idx) == 3

    # 3D layouts should have 3 accel bias terms
    assert len(LAYOUT_3D_EULER.bias_accel_idx) == 3
    assert len(LAYOUT_3D_QUAT.bias_accel_idx) == 3


def test_velocity_dimension_matches_accel_bias_dimension():
    """Verify velocity and accel bias dimensions are consistent."""
    # 2D full: 2D velocity, 2D accel bias
    assert len(LAYOUT_2D_FULL.vel_idx) == 2
    assert len(LAYOUT_2D_FULL.bias_accel_idx) == 2

    # 2D cam + 3D IMU: 3D velocity, 3D accel bias
    assert len(LAYOUT_2D_CAM_3D_IMU.vel_idx) == 3
    assert len(LAYOUT_2D_CAM_3D_IMU.bias_accel_idx) == 3

    # 2D cam + 6-DOF IMU orientation: 2D velocity, full 3D accel bias
    assert len(LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION.vel_idx) == 2
    assert len(LAYOUT_2D_CAM_6DOF_IMU_ORIENTATION.bias_accel_idx) == 3

    # 3D Euler: 3D velocity, 3D accel bias
    assert len(LAYOUT_3D_EULER.vel_idx) == 3
    assert len(LAYOUT_3D_EULER.bias_accel_idx) == 3

    # 3D Quat: 3D velocity, 3D accel bias
    assert len(LAYOUT_3D_QUAT.vel_idx) == 3
    assert len(LAYOUT_3D_QUAT.bias_accel_idx) == 3


def test_layout_2d_cam_3d_imu_has_2d_heading():
    """Verify 2D cam + 3D IMU still uses 2D heading (yaw only)."""
    layout = LAYOUT_2D_CAM_3D_IMU

    # Should have 2D heading (scalar), not 3D orientation
    assert layout.has_heading_2d is True
    assert layout.has_orientation_3d is False
    assert isinstance(layout.heading_idx, int)
