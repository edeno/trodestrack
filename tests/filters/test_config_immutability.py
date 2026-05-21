"""Test config immutability during filter execution.

Reproducibility requirement: configs must be immutable. When
auto-inferring parameters like LED spacing, the inferred value
should be returned in the result, not mutate the original config.
"""

import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter
from trodestrack.sim.simple import SimpleSimConfig, simulate_stationary


@pytest.fixture
def sim_data():
    """Generate simple simulation data for testing."""
    config = SimpleSimConfig(duration_s=2.0, fs_imu=200.0, fs_cam=30.0)
    sim_out = simulate_stationary(
        config,
        position=np.array([1.0, 1.0]),
        seed=42,
    )
    return sim_out


class TestEKFConfigImmutability:
    """Test that EKF does not mutate config during execution."""

    def test_config_not_mutated_with_explicit_led_distance(self, sim_data):
        """Config should not be mutated when LED distance is explicitly set."""
        # Original config with explicit LED distance
        config = EKFConfig(led_distance=0.04)
        config_dict_before = config.__dict__.copy()

        # Run filter
        _ = extended_kalman_filter(
            ekf_config=config,
            t_imu=sim_data["t_imu"],
            U_imu=sim_data["U_imu"],
            t_cam=sim_data["t_cam_exp"],
            Z_cam_led1=sim_data["Z_cam_led1"],
            Z_cam_led2=sim_data["Z_cam_led2"],
            mask_cam=sim_data["mask_cam"],
        )

        # Verify config unchanged
        config_dict_after = config.__dict__.copy()
        assert config_dict_before == config_dict_after
        assert config.led_distance == 0.04

    def test_config_not_mutated_with_auto_led_distance(self, sim_data):
        """Config should not be mutated even with auto-detection (led_distance=None)."""
        # Original config with None (auto-detect)
        config = EKFConfig(led_distance=None)
        config_dict_before = config.__dict__.copy()

        # Run filter
        _ = extended_kalman_filter(
            ekf_config=config,
            t_imu=sim_data["t_imu"],
            U_imu=sim_data["U_imu"],
            t_cam=sim_data["t_cam_exp"],
            Z_cam_led1=sim_data["Z_cam_led1"],
            Z_cam_led2=sim_data["Z_cam_led2"],
            mask_cam=sim_data["mask_cam"],
        )

        # Verify config unchanged (led_distance should still be None)
        config_dict_after = config.__dict__.copy()
        assert config_dict_before == config_dict_after
        assert config.led_distance is None

    def test_estimated_led_distance_returned_in_result(self, sim_data):
        """When auto-detecting LED distance, it should be returned in the result."""
        # Config with auto-detect
        config = EKFConfig(led_distance=None)

        # Run filter
        result = extended_kalman_filter(
            ekf_config=config,
            t_imu=sim_data["t_imu"],
            U_imu=sim_data["U_imu"],
            t_cam=sim_data["t_cam_exp"],
            Z_cam_led1=sim_data["Z_cam_led1"],
            Z_cam_led2=sim_data["Z_cam_led2"],
            mask_cam=sim_data["mask_cam"],
        )

        # Result should contain estimated_led_distance
        assert hasattr(result, "estimated_led_distance")
        assert result.estimated_led_distance is not None
        assert isinstance(result.estimated_led_distance, float)
        # Should be close to simulation LED distance (4 cm)
        assert 0.03 < result.estimated_led_distance < 0.05

    def test_estimated_led_distance_none_when_explicit(self, sim_data):
        """When LED distance is explicit, estimated_led_distance should be None."""
        # Config with explicit LED distance
        config = EKFConfig(led_distance=0.04)

        # Run filter
        result = extended_kalman_filter(
            ekf_config=config,
            t_imu=sim_data["t_imu"],
            U_imu=sim_data["U_imu"],
            t_cam=sim_data["t_cam_exp"],
            Z_cam_led1=sim_data["Z_cam_led1"],
            Z_cam_led2=sim_data["Z_cam_led2"],
            mask_cam=sim_data["mask_cam"],
        )

        # estimated_led_distance should be None (not estimated)
        assert hasattr(result, "estimated_led_distance")
        assert result.estimated_led_distance is None


class TestUKFConfigImmutability:
    """Test that UKF does not mutate config during execution."""

    def test_config_not_mutated_with_explicit_led_distance(self, sim_data):
        """Config should not be mutated when LED distance is explicitly set."""
        # Original config with explicit LED distance
        config = UKFConfig(led_distance=0.04)
        config_dict_before = config.__dict__.copy()

        # Run filter
        _ = unscented_kalman_filter(
            ukf_config=config,
            t_imu=sim_data["t_imu"],
            U_imu=sim_data["U_imu"],
            t_cam=sim_data["t_cam_exp"],
            Z_cam_led1=sim_data["Z_cam_led1"],
            Z_cam_led2=sim_data["Z_cam_led2"],
            mask_cam=sim_data["mask_cam"],
        )

        # Verify config unchanged
        config_dict_after = config.__dict__.copy()
        assert config_dict_before == config_dict_after
        assert config.led_distance == 0.04

    def test_config_not_mutated_with_auto_led_distance(self, sim_data):
        """Config should not be mutated even with auto-detection (led_distance=None)."""
        # Original config with None (auto-detect)
        config = UKFConfig(led_distance=None)
        config_dict_before = config.__dict__.copy()

        # Run filter
        _ = unscented_kalman_filter(
            ukf_config=config,
            t_imu=sim_data["t_imu"],
            U_imu=sim_data["U_imu"],
            t_cam=sim_data["t_cam_exp"],
            Z_cam_led1=sim_data["Z_cam_led1"],
            Z_cam_led2=sim_data["Z_cam_led2"],
            mask_cam=sim_data["mask_cam"],
        )

        # Verify config unchanged (led_distance should still be None)
        config_dict_after = config.__dict__.copy()
        assert config_dict_before == config_dict_after
        assert config.led_distance is None

    def test_estimated_led_distance_returned_in_result(self, sim_data):
        """When auto-detecting LED distance, it should be returned in the result."""
        # Config with auto-detect
        config = UKFConfig(led_distance=None)

        # Run filter
        result = unscented_kalman_filter(
            ukf_config=config,
            t_imu=sim_data["t_imu"],
            U_imu=sim_data["U_imu"],
            t_cam=sim_data["t_cam_exp"],
            Z_cam_led1=sim_data["Z_cam_led1"],
            Z_cam_led2=sim_data["Z_cam_led2"],
            mask_cam=sim_data["mask_cam"],
        )

        # Result should contain estimated_led_distance
        assert hasattr(result, "estimated_led_distance")
        assert result.estimated_led_distance is not None
        assert isinstance(result.estimated_led_distance, float)
        # Should be close to simulation LED distance (4 cm)
        assert 0.03 < result.estimated_led_distance < 0.05

    def test_estimated_led_distance_none_when_explicit(self, sim_data):
        """When LED distance is explicit, estimated_led_distance should be None."""
        # Config with explicit LED distance
        config = UKFConfig(led_distance=0.04)

        # Run filter
        result = unscented_kalman_filter(
            ukf_config=config,
            t_imu=sim_data["t_imu"],
            U_imu=sim_data["U_imu"],
            t_cam=sim_data["t_cam_exp"],
            Z_cam_led1=sim_data["Z_cam_led1"],
            Z_cam_led2=sim_data["Z_cam_led2"],
            mask_cam=sim_data["mask_cam"],
        )

        # estimated_led_distance should be None (not estimated)
        assert hasattr(result, "estimated_led_distance")
        assert result.estimated_led_distance is None
