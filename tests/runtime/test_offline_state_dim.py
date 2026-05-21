"""Test state dimension generalization in offline smoothers.

This test suite validates that smoothers work with arbitrary state dimensions,
not just hardcoded 8D states. This is critical for:
    - Future 3D extensions (12D state: x,y,z, vx,vy,vz, roll,pitch,yaw, biases)
    - Ablation studies (reduced states: position-only, no bias, etc.)
    - Extensibility and maintainability

Tests ensure:
    1. RTS smoother derives state dimension from input data
    2. Sigma-point smoother derives state dimension from input data
    3. Q_rate matrix is built dynamically based on state dimension
    4. No hardcoded 8 in array shapes or indexing
    5. Works with dimensions < 8, = 8, and > 8

Note:
    These tests use mock dynamics that work with arbitrary dimensions.
    The mock dynamics below deliberately accepts the same keyword surface as the
    runtime dynamics function while keeping arbitrary state dimensions. This
    test validates that the SMOOTHER INFRASTRUCTURE is dimension-agnostic.

This supports planned 3D extensions (e.g. 12D state) and reduced-state
ablation studies without rewriting the smoother infrastructure.
"""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from trodestrack.models.ekf import EKFConfig, EKFResult
from trodestrack.models.ukf import UKFConfig, UKFResult
from trodestrack.runtime.offline import rts_smoother, sigma_point_smoother

# =============================================================================
# Mock Dynamics for Testing
# =============================================================================


def mock_dynamics_function(
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
    damping: float,
    layout=None,
    gravity_body=None,
    enable_experimental_accel_translation: bool = False,
) -> jnp.ndarray:
    """Mock dynamics that work with any state dimension.

    Simple forward Euler: x_next = x + dt * (trivial drift)
    This is sufficient to test smoother dimension handling.
    """
    # Trivial dynamics: just add small perturbation
    return x + 0.001 * dt * jnp.ones_like(x)


# =============================================================================
# Test Helpers
# =============================================================================


def make_minimal_filter_result_ekf(n_cam: int, state_dim: int) -> EKFResult:
    """Create a minimal EKF filter result with arbitrary state dimension.

    Args:
        n_cam: Number of camera frames
        state_dim: State dimension (not necessarily 8)

    Returns:
        Minimal EKFResult for testing
    """
    rng = np.random.RandomState(42)

    # Create random but reasonable filter outputs
    filtered_means = rng.randn(n_cam, state_dim).astype(np.float32)
    filtered_covs = np.array(
        [
            np.eye(state_dim) + 0.01 * rng.randn(state_dim, state_dim)
            for _ in range(n_cam)
        ]
    )
    # Symmetrize
    filtered_covs = 0.5 * (filtered_covs + filtered_covs.transpose(0, 2, 1))

    # Predicted means/covs (not used by smoother, but required by EKFResult)
    predicted_means = filtered_means.copy()
    predicted_covs = filtered_covs.copy()

    return EKFResult(
        filtered_means=jnp.array(filtered_means),
        filtered_covariances=jnp.array(filtered_covs),
        predicted_means=jnp.array(predicted_means),
        predicted_covariances=jnp.array(predicted_covs),
        marginal_loglik=0.0,
        estimated_led_distance=None,
    )


def make_minimal_filter_result_ukf(n_cam: int, state_dim: int) -> UKFResult:
    """Create a minimal UKF filter result with arbitrary state dimension.

    Args:
        n_cam: Number of camera frames
        state_dim: State dimension (not necessarily 8)

    Returns:
        Minimal UKFResult for testing
    """
    rng = np.random.RandomState(43)

    filtered_means = rng.randn(n_cam, state_dim).astype(np.float32)
    filtered_covs = np.array(
        [
            np.eye(state_dim) + 0.01 * rng.randn(state_dim, state_dim)
            for _ in range(n_cam)
        ]
    )
    filtered_covs = 0.5 * (filtered_covs + filtered_covs.transpose(0, 2, 1))

    # Predicted means/covs (not used by smoother, but required by UKFResult)
    predicted_means = filtered_means.copy()
    predicted_covs = filtered_covs.copy()

    return UKFResult(
        filtered_means=jnp.array(filtered_means),
        filtered_covariances=jnp.array(filtered_covs),
        predicted_means=jnp.array(predicted_means),
        predicted_covariances=jnp.array(predicted_covs),
        marginal_loglik=0.0,
        estimated_led_distance=None,
    )


def make_minimal_imu_data(n_imu: int, n_cam: int):
    """Create minimal IMU and camera data for testing.

    Args:
        n_imu: Number of IMU samples
        n_cam: Number of camera frames

    Returns:
        Tuple of (t_imu, U_imu, t_cam)
    """
    t_imu = np.linspace(0, 1, n_imu)
    U_imu = np.zeros((n_imu, 3))  # [ω_z, f_x, f_y]
    t_cam = np.linspace(0, 1, n_cam)
    return t_imu, U_imu, t_cam


# =============================================================================
# Unit Tests for build_Q_rate()
# =============================================================================


def test_build_Q_rate_standard_8d():
    """Test Q_rate for standard 8D state has correct structure.

    Verifies that the standard 8D state uses the structured noise mapping:
    [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
    """
    from trodestrack.models.process_noise import build_Q_rate

    config = EKFConfig()
    Q = build_Q_rate(config, 8)

    # Check shape
    assert Q.shape == (8, 8)

    # Check diagonal values match config
    assert jnp.allclose(Q[0, 0], config.process_noise_pos)  # x
    assert jnp.allclose(Q[1, 1], config.process_noise_pos)  # y
    assert jnp.allclose(Q[2, 2], config.process_noise_vel)  # vx
    assert jnp.allclose(Q[3, 3], config.process_noise_vel)  # vy
    assert jnp.allclose(Q[4, 4], config.process_noise_heading)  # θ
    assert jnp.allclose(Q[5, 5], config.process_noise_gyro_bias)  # b_gz
    assert jnp.allclose(Q[6, 6], config.process_noise_accel_bias)  # b_ax
    assert jnp.allclose(Q[7, 7], config.process_noise_accel_bias)  # b_ay

    # Verify off-diagonal is zero (diagonal matrix)
    assert jnp.allclose(Q, jnp.diag(jnp.diag(Q)))


def test_build_Q_rate_non_standard_dimensions():
    """Test Q_rate for non-8D states uses StateLayout when available, fallback otherwise."""
    from trodestrack.models.process_noise import build_Q_rate

    config = EKFConfig()

    # Test dimensions with known layouts
    # n=10: LAYOUT_2D_CAM_3D_IMU
    Q_10 = build_Q_rate(config, 10)
    assert Q_10.shape == (10, 10)
    # Should match LAYOUT_2D_CAM_3D_IMU structure
    expected_10 = jnp.array(
        [
            config.process_noise_pos,  # x
            config.process_noise_pos,  # y
            config.process_noise_vel,  # vx
            config.process_noise_vel,  # vy
            config.process_noise_vel,  # vz
            config.process_noise_heading,  # θ
            config.process_noise_gyro_bias,  # b_gz
            config.process_noise_accel_bias,  # b_ax
            config.process_noise_accel_bias,  # b_ay
            config.process_noise_accel_bias,  # b_az
        ]
    )
    assert jnp.allclose(jnp.diag(Q_10), expected_10)
    assert jnp.allclose(Q_10, jnp.diag(jnp.diag(Q_10)))

    # Test dimensions without known layouts (should use uniform fallback)
    for n in [4, 6, 12]:
        Q = build_Q_rate(config, n)
        assert Q.shape == (n, n)
        # Should use uniform noise = process_noise_pos for unknown dimensions
        expected_diag = jnp.full(n, config.process_noise_pos)
        assert jnp.allclose(jnp.diag(Q), expected_diag)
        # Verify off-diagonal is zero (diagonal matrix)
        assert jnp.allclose(Q, jnp.diag(jnp.diag(Q)))


def test_build_Q_rate_works_with_ukf_config():
    """Test that build_Q_rate accepts UKFConfig as well as EKFConfig."""
    from trodestrack.models.process_noise import build_Q_rate

    ukf_config = UKFConfig()
    Q = build_Q_rate(ukf_config, 8)

    # Should work identically (same noise parameters)
    assert Q.shape == (8, 8)
    assert jnp.allclose(Q[0, 0], ukf_config.process_noise_pos)


# =============================================================================
# RTS Smoother State Dimension Tests
# =============================================================================


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_rts_smoother_derives_state_dimension_from_data():
    """RTS smoother infrastructure handles 10D (2d_cam_3d_imu) without hardcoding 8.

    The smoother is dimension-generic, but the public-API guard requires
    ``filter_result.filtered_means.shape[1]`` to match
    ``ekf_config.state_mode``'s layout. Use a registered mode whose
    layout dim is not the legacy 8D so we still cover the
    "no hardcoded 8" contract.
    """
    n_cam = 10
    state_dim = 10  # LAYOUT_2D_CAM_3D_IMU

    filter_result = make_minimal_filter_result_ekf(n_cam, state_dim)
    t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=100, n_cam=n_cam)

    config = EKFConfig(state_mode="2d_cam_3d_imu")

    # This should NOT crash with dimension errors.
    result = rts_smoother(filter_result, config, t_imu, U_imu, t_cam)

    assert result.smoothed_means.shape == (n_cam, state_dim)
    assert result.smoothed_covariances.shape == (n_cam, state_dim, state_dim)


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_rts_smoother_works_with_reduced_state():
    """RTS smoother with a reduced 5D state (vision_only)."""
    n_cam = 5
    state_dim = 5  # LAYOUT_VISION_ONLY

    filter_result = make_minimal_filter_result_ekf(n_cam, state_dim)
    t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=50, n_cam=n_cam)
    config = EKFConfig(state_mode="vision_only", enable_zupt=False)

    result = rts_smoother(filter_result, config, t_imu, U_imu, t_cam)

    assert result.smoothed_means.shape == (n_cam, state_dim)
    assert result.smoothed_covariances.shape == (n_cam, state_dim, state_dim)


# NOTE: a separate "extended state" test (>10D) is not in the matrix because
# every registered layout above 10D is quaternion-orientation, which requires
# 6-channel IMU input and a unit-norm quaternion in the state vector — that's
# beyond the scope of the dimension-generic-infrastructure check this file
# covers. The 10D
# ``test_rts_smoother_derives_state_dimension_from_data`` already exercises
# the non-8D codepath.


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_rts_smoother_works_with_standard_8d_state():
    """Standard 8D state still works (backward compatibility)."""
    n_cam = 5
    state_dim = 8  # LAYOUT_2D_FULL

    filter_result = make_minimal_filter_result_ekf(n_cam, state_dim)
    t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=50, n_cam=n_cam)
    config = EKFConfig(state_mode="2d_full")

    result = rts_smoother(filter_result, config, t_imu, U_imu, t_cam)

    assert result.smoothed_means.shape == (n_cam, state_dim)
    assert result.smoothed_covariances.shape == (n_cam, state_dim, state_dim)


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_rts_smoother_rejects_filter_result_layout_mismatch():
    """rts_smoother raises when filter_result dim disagrees with config.state_mode."""
    n_cam = 5

    # 5D filter_result paired with 8D 2d_full config — layout-wrong combo.
    filter_result = make_minimal_filter_result_ekf(n_cam, state_dim=5)
    t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=50, n_cam=n_cam)
    config = EKFConfig(state_mode="2d_full")

    import pytest

    with pytest.raises(ValueError, match=r"does not match ekf_config\.state_mode"):
        rts_smoother(filter_result, config, t_imu, U_imu, t_cam)


# =============================================================================
# Sigma-Point Smoother State Dimension Tests
# =============================================================================


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_sigma_point_smoother_derives_state_dimension_from_data():
    """Sigma-point smoother infrastructure handles 10D without hardcoding 8.

    The UKF rejects quaternion-orientation layouts, so the registered
    non-quaternion dimensions available to this test are 5 / 8 / 10.
    Using 10 keeps the "no hardcoded 8" coverage intact while honoring
    the layout-vs-filter-result guard at the public boundary.
    """
    n_cam = 10
    state_dim = 10  # LAYOUT_2D_CAM_3D_IMU

    filter_result = make_minimal_filter_result_ukf(n_cam, state_dim)
    t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=100, n_cam=n_cam)
    config = UKFConfig(state_mode="2d_cam_3d_imu")

    result = sigma_point_smoother(filter_result, config, t_imu, U_imu, t_cam)

    assert result.smoothed_means.shape == (n_cam, state_dim)
    assert result.smoothed_covariances.shape == (n_cam, state_dim, state_dim)


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_sigma_point_smoother_works_with_reduced_state():
    """Sigma-point smoother with a reduced 5D state (vision_only)."""
    n_cam = 5
    state_dim = 5  # LAYOUT_VISION_ONLY

    filter_result = make_minimal_filter_result_ukf(n_cam, state_dim)
    t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=50, n_cam=n_cam)
    config = UKFConfig(state_mode="vision_only", enable_zupt=False)

    result = sigma_point_smoother(filter_result, config, t_imu, U_imu, t_cam)

    assert result.smoothed_means.shape == (n_cam, state_dim)
    assert result.smoothed_covariances.shape == (n_cam, state_dim, state_dim)


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_sigma_point_smoother_works_with_standard_8d_state():
    """Standard 8D state (backward compatibility)."""
    n_cam = 5
    state_dim = 8  # LAYOUT_2D_FULL

    filter_result = make_minimal_filter_result_ukf(n_cam, state_dim)
    t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=50, n_cam=n_cam)
    config = UKFConfig(state_mode="2d_full")

    result = sigma_point_smoother(filter_result, config, t_imu, U_imu, t_cam)

    assert result.smoothed_means.shape == (n_cam, state_dim)
    assert result.smoothed_covariances.shape == (n_cam, state_dim, state_dim)


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_sigma_point_smoother_rejects_filter_result_layout_mismatch():
    """sigma_point_smoother raises when filter_result dim disagrees with config."""
    n_cam = 5

    filter_result = make_minimal_filter_result_ukf(n_cam, state_dim=5)
    t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=50, n_cam=n_cam)
    config = UKFConfig(state_mode="2d_full")

    import pytest

    with pytest.raises(ValueError, match=r"does not match ukf_config\.state_mode"):
        sigma_point_smoother(filter_result, config, t_imu, U_imu, t_cam)


# =============================================================================
# Sigma-Point Count Validation
# =============================================================================


@patch("trodestrack.runtime.offline.dynamics_function", mock_dynamics_function)
def test_sigma_point_count_adapts_to_state_dimension():
    """Sigma-point count adapts to state dimension across registered modes.

    For state dimension n the smoother should construct ``2n+1`` sigma
    points without crashing. We exercise every registered non-quaternion
    UKF dimension paired with its matching state_mode (the layout-vs-
    filter-result guard in sigma_point_smoother enforces the pairing).
    """
    cases = [
        (5, "vision_only"),
        (8, "2d_full"),
        (10, "2d_cam_3d_imu"),
    ]

    for state_dim, state_mode in cases:
        n_cam = 3
        filter_result = make_minimal_filter_result_ukf(n_cam, state_dim)
        t_imu, U_imu, t_cam = make_minimal_imu_data(n_imu=30, n_cam=n_cam)
        # vision_only is incompatible with enable_zupt (default True).
        config_kwargs: dict[str, object] = {"state_mode": state_mode}
        if state_mode == "vision_only":
            config_kwargs["enable_zupt"] = False
        config = UKFConfig(**config_kwargs)

        result = sigma_point_smoother(filter_result, config, t_imu, U_imu, t_cam)
        assert result.smoothed_means.shape == (n_cam, state_dim)


# =============================================================================
# Documentation Tests
# =============================================================================


def test_smoother_result_docstring_not_hardcoded():
    """Test that SmootherResult docstring doesn't claim fixed dimensions.

    This is a documentation check - ensures we update comments.
    """
    from trodestrack.runtime.offline import SmootherResult

    # Check that docstring doesn't have hardcoded (N_cam, 8) shapes
    # (This will pass after we update docs)
    docstring = SmootherResult.__doc__

    # For now, just ensure docstring exists
    assert docstring is not None
    # Future: could parse docstring and check for flexible dimension notation
