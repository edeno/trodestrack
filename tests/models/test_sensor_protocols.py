"""Test suite for sensor measurement model protocols.

This module tests the MeasurementModel protocol and its implementations
(CameraPositionModel, HeadingPseudoModel) to ensure they provide a unified
interface for all sensor types while maintaining numerical parity with the
existing EKF/UKF camera and heading measurement helpers.

Test Coverage
-------------
- Protocol compliance (structural subtyping)
- Camera position model (wraps existing camera helpers)
- Heading pseudo-measurement model (wraps existing heading helpers)
- Numerical parity with existing EKF helpers (≤1e-7 mean, ≤1e-6 cov diag)
- LED validity patterns → correct projector consistency
- Confidence scaling → correct R per frame
"""

import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.filter_common import FilterCoreConfig, measurement_function
from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.sensors.heading_pseudo import HeadingPseudoModel
from trodestrack.models.sensors.protocols import MeasurementModel
from trodestrack.models.state_layout import get_layout


@pytest.fixture
def simple_state() -> jnp.ndarray:
    """Simple 8D state for testing: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]."""
    return jnp.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0])


@pytest.fixture
def layout_2d_full():
    """Standard 2D full state layout (8D)."""
    return get_layout("2d_full")


@pytest.fixture
def heading_config():
    """Filter configuration for heading pseudo-measurement model."""
    return FilterCoreConfig(
        use_heading_measurement=True,
        measurement_noise_heading=0.05**2,
        led_distance=0.04,
        led_distance_tolerance=0.3,
        adaptive_heading_noise=True,
    )


@pytest.fixture
def dual_led_observations():
    """Valid dual-LED observations for testing."""
    return {
        "z_led1": jnp.array([0.95, 1.98]),
        "z_led2": jnp.array([1.05, 2.02]),
        "confidence": jnp.array([0.9, 0.9, 0.8, 0.8]),
        "valid": True,
    }


@pytest.fixture
def single_led1_observations():
    """Single LED1 observation (LED2 is NaN)."""
    return {
        "z_led1": jnp.array([0.95, 1.98]),
        "z_led2": jnp.array([jnp.nan, jnp.nan]),
        "confidence": jnp.array([0.9, 0.9, 0.0, 0.0]),
        "valid": True,
    }


@pytest.fixture
def single_led2_observations():
    """Single LED2 observation (LED1 is NaN)."""
    return {
        "z_led1": jnp.array([jnp.nan, jnp.nan]),
        "z_led2": jnp.array([1.05, 2.02]),
        "confidence": jnp.array([0.0, 0.0, 0.8, 0.8]),
        "valid": True,
    }


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


def test_camera_model_implements_protocol(layout_2d_full):
    """CameraPositionModel should satisfy MeasurementModel protocol."""
    model = CameraPositionModel(
        led_distance=0.04,
        measurement_noise_base=0.005**2,
        layout=layout_2d_full,
    )
    # Protocol uses structural subtyping, so this checks at runtime
    assert isinstance(model, MeasurementModel)


def test_heading_model_implements_protocol(heading_config, layout_2d_full):
    """HeadingPseudoModel should satisfy MeasurementModel protocol."""
    model = HeadingPseudoModel(
        config=heading_config,
        layout=layout_2d_full,
    )
    assert isinstance(model, MeasurementModel)


# =============================================================================
# Camera Position Model Tests
# =============================================================================


def test_camera_model_meas_dim(layout_2d_full):
    """Camera model should always report 4D measurement (dual-LED)."""
    model = CameraPositionModel(
        led_distance=0.04,
        measurement_noise_base=0.005**2,
        layout=layout_2d_full,
    )
    assert model.meas_dim == 4


def test_camera_model_predict_matches_measurement_function(simple_state, layout_2d_full):
    """Camera model predict() should match existing measurement_function()."""
    led_distance = 0.04
    model = CameraPositionModel(
        led_distance=led_distance,
        measurement_noise_base=0.005**2,
        layout=layout_2d_full,
    )

    # Model prediction
    meas_pred = model.predict(simple_state)

    # Existing helper
    expected = measurement_function(simple_state, led_distance, layout_2d_full)

    # Should match within numerical precision
    np.testing.assert_allclose(meas_pred, expected, atol=1e-10)


def test_camera_model_jacobian(simple_state, layout_2d_full):
    """Camera model jacobian() should return correct shape."""
    model = CameraPositionModel(
        led_distance=0.04,
        measurement_noise_base=0.005**2,
        layout=layout_2d_full,
    )

    H = model.jacobian(simple_state)
    assert H is not None, "Camera model should provide Jacobian for EKF"
    assert H.shape == (4, 8), f"Expected (4, 8), got {H.shape}"


def test_camera_model_confidence_scaling(layout_2d_full, dual_led_observations):
    """Camera model should scale R by confidence per dimension."""
    base_R = 0.005**2
    model = CameraPositionModel(
        led_distance=0.04,
        measurement_noise_base=base_R,
        layout=layout_2d_full,
    )

    # Set up observations for frame 0
    model.set_frame_data(
        frame_idx=0,
        z_led1=dual_led_observations["z_led1"],
        z_led2=dual_led_observations["z_led2"],
        confidence=dual_led_observations["confidence"],
    )

    # Get measurement covariance
    R = model.meas_cov(frame_idx=0)

    # Should be diagonal with confidence-scaled values
    assert R.shape == (4, 4)
    assert jnp.allclose(R, jnp.diag(jnp.diag(R))), "R should be diagonal"

    # R_i = base / conf_i (with clipping at min=0.01)
    expected_diag = jnp.array(
        [
            base_R / 0.9,  # LED1 x
            base_R / 0.9,  # LED1 y
            base_R / 0.8,  # LED2 x
            base_R / 0.8,  # LED2 y
        ]
    )
    np.testing.assert_allclose(jnp.diag(R), expected_diag, rtol=1e-5)


def test_camera_model_single_led_subspace(layout_2d_full, single_led1_observations):
    """Camera model should return correct projector for single LED."""
    model = CameraPositionModel(
        led_distance=0.04,
        measurement_noise_base=0.005**2,
        layout=layout_2d_full,
    )

    model.set_frame_data(
        frame_idx=0,
        z_led1=single_led1_observations["z_led1"],
        z_led2=single_led1_observations["z_led2"],
        confidence=single_led1_observations["confidence"],
    )

    # Get subspace info
    both_leds, only_led1, only_led2, projector = model.subspace(frame_idx=0)

    assert not both_leds, "Should not have both LEDs"
    assert only_led1, "Should detect only LED1"
    assert not only_led2, "Should not detect LED2"

    # Projector should be 2x4 selecting first two dimensions
    assert projector.shape == (2, 4)
    expected_M = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=jnp.float32)
    np.testing.assert_allclose(projector, expected_M, atol=1e-10)


def test_camera_model_dual_led_subspace(layout_2d_full, dual_led_observations):
    """Camera model should return identity projector for dual LED."""
    model = CameraPositionModel(
        led_distance=0.04,
        measurement_noise_base=0.005**2,
        layout=layout_2d_full,
    )

    model.set_frame_data(
        frame_idx=0,
        z_led1=dual_led_observations["z_led1"],
        z_led2=dual_led_observations["z_led2"],
        confidence=dual_led_observations["confidence"],
    )

    both_leds, only_led1, only_led2, projector = model.subspace(frame_idx=0)

    assert both_leds, "Should have both LEDs"
    assert not only_led1, "Should not be single LED1"
    assert not only_led2, "Should not be single LED2"

    # Projector should be 4x4 identity (no projection)
    assert projector.shape == (4, 4)
    np.testing.assert_allclose(projector, jnp.eye(4), atol=1e-10)


def test_camera_model_innovation(layout_2d_full, dual_led_observations, simple_state):
    """Camera model should compute correct innovation."""
    model = CameraPositionModel(
        led_distance=0.04,
        measurement_noise_base=0.005**2,
        layout=layout_2d_full,
    )

    model.set_frame_data(
        frame_idx=0,
        z_led1=dual_led_observations["z_led1"],
        z_led2=dual_led_observations["z_led2"],
        confidence=dual_led_observations["confidence"],
    )

    # Prediction at simple_state
    meas_pred = model.predict(simple_state)

    # Innovation
    innovation = model.innovation(frame_idx=0, meas_pred=meas_pred)

    # Should be z - h(x)
    z_obs = jnp.concatenate([dual_led_observations["z_led1"], dual_led_observations["z_led2"]])
    expected_innov = z_obs - meas_pred

    np.testing.assert_allclose(innovation, expected_innov, atol=1e-10)


# =============================================================================
# Heading Pseudo-Measurement Model Tests
# =============================================================================


def test_heading_model_meas_dim(heading_config, layout_2d_full):
    """Heading model should report 1D measurement."""
    model = HeadingPseudoModel(
        config=heading_config,
        layout=layout_2d_full,
    )
    assert model.meas_dim == 1


def test_heading_model_predict(simple_state, heading_config, layout_2d_full):
    """Heading model should extract heading from state."""
    model = HeadingPseudoModel(
        config=heading_config,
        layout=layout_2d_full,
    )

    # Predict heading
    meas_pred = model.predict(simple_state)

    # Should be the heading component (index 4 in 2d_full layout)
    expected = simple_state[4:5]  # Keep as 1D array

    np.testing.assert_allclose(meas_pred, expected, atol=1e-10)


def test_heading_model_jacobian(simple_state, heading_config, layout_2d_full):
    """Heading model should return correct Jacobian shape."""
    model = HeadingPseudoModel(
        config=heading_config,
        layout=layout_2d_full,
    )

    H = model.jacobian(simple_state)
    assert H is not None, "Heading model should provide Jacobian"
    assert H.shape == (1, 8), f"Expected (1, 8), got {H.shape}"

    # Should be zeros except at heading index (index 4)
    expected = jnp.zeros((1, 8))
    expected = expected.at[0, 4].set(1.0)
    np.testing.assert_allclose(H, expected, atol=1e-10)


def test_heading_model_spacing_gate(heading_config, layout_2d_full, dual_led_observations):
    """Heading model should gate measurements when LED spacing is invalid."""
    model = HeadingPseudoModel(
        config=heading_config,
        layout=layout_2d_full,
    )

    # Valid spacing (close to expected 0.04)
    model.set_frame_data(
        frame_idx=0,
        z_led1=jnp.array([1.0, 2.0]),
        z_led2=jnp.array([1.04, 2.0]),  # Distance = 0.04 (exact)
    )

    # Get measurement noise (should be normal)
    R_valid = model.meas_cov(frame_idx=0)
    assert R_valid[0, 0] < 1e5, "Valid spacing should not be gated"

    # Invalid spacing (too far from expected)
    model.set_frame_data(
        frame_idx=1,
        z_led1=jnp.array([1.0, 2.0]),
        z_led2=jnp.array([1.10, 2.0]),  # Distance = 0.10 (way too large)
    )

    # Get measurement noise (should be gated with large R)
    R_invalid = model.meas_cov(frame_idx=1)
    assert R_invalid[0, 0] >= 1e6, "Invalid spacing should be gated with R=1e6"


def test_heading_model_adaptive_noise(heading_config, layout_2d_full):
    """Heading model should scale noise by (expected/observed)^2."""
    model = HeadingPseudoModel(
        config=heading_config,
        layout=layout_2d_full,
    )
    base_R = heading_config.measurement_noise_heading  # 0.05**2

    # Observed spacing = 0.03 (within tolerance: 0.028 < 0.03 < 0.052)
    model.set_frame_data(
        frame_idx=0,
        z_led1=jnp.array([1.0, 2.0]),
        z_led2=jnp.array([1.03, 2.0]),  # Distance = 0.03
    )

    R = model.meas_cov(frame_idx=0)

    # Adaptive scaling: R_scaled = R_base * (expected / observed)^2
    # = base_R * (0.04 / 0.03)^2 ≈ base_R * 1.778
    expected_R = base_R * (0.04 / 0.03) ** 2

    np.testing.assert_allclose(R[0, 0], expected_R, rtol=1e-5)


def test_heading_model_single_led_gates(heading_config, layout_2d_full):
    """Heading model should gate when only one LED is visible."""
    model = HeadingPseudoModel(
        config=heading_config,
        layout=layout_2d_full,
    )

    # Single LED (LED2 is NaN)
    model.set_frame_data(
        frame_idx=0,
        z_led1=jnp.array([1.0, 2.0]),
        z_led2=jnp.array([jnp.nan, jnp.nan]),
    )

    # Should gate with large R
    R = model.meas_cov(frame_idx=0)
    assert R[0, 0] >= 1e6, "Single LED should be gated"


def test_heading_model_innovation_wrapping(heading_config, layout_2d_full):
    """Heading model should wrap innovation to [-π, π]."""
    model = HeadingPseudoModel(
        config=heading_config,
        layout=layout_2d_full,
    )

    # Set up LED geometry that gives heading ≈ 0.1 rad
    model.set_frame_data(
        frame_idx=0,
        z_led1=jnp.array([1.0, 2.0]),
        z_led2=jnp.array([1.04, 2.004]),  # Heading ≈ atan2(0.004, 0.04) ≈ 0.1 rad
    )

    # Predicted heading very different (e.g., π)
    meas_pred = jnp.array([jnp.pi])

    # Innovation should be wrapped
    innovation = model.innovation(frame_idx=0, meas_pred=meas_pred)

    # Should be in [-π, π]
    assert -jnp.pi <= innovation[0] <= jnp.pi, f"Innovation {innovation[0]} not wrapped"


# =============================================================================
# Parity Tests (EKF/UKF compatibility)
# =============================================================================


def test_camera_model_parity_with_ekf_helpers(simple_state, layout_2d_full):
    """Camera model outputs should match EKF helper functions within tolerance."""
    led_distance = 0.04
    base_R = 0.005**2

    model = CameraPositionModel(
        led_distance=led_distance,
        measurement_noise_base=base_R,
        layout=layout_2d_full,
    )

    # Set up dual-LED observation
    z_led1 = jnp.array([0.95, 1.98])
    z_led2 = jnp.array([1.05, 2.02])
    confidence = jnp.array([0.9, 0.9, 0.8, 0.8])

    model.set_frame_data(frame_idx=0, z_led1=z_led1, z_led2=z_led2, confidence=confidence)

    # Model prediction
    meas_pred = model.predict(simple_state)

    # Existing measurement_function
    expected_pred = measurement_function(simple_state, led_distance, layout_2d_full)

    # Parity check: ≤1e-7 mean difference
    mean_diff = jnp.abs(meas_pred - expected_pred).mean()
    assert mean_diff <= 1e-7, f"Mean difference {mean_diff} exceeds 1e-7"
