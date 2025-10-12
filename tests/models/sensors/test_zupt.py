"""Unit tests for ZUPTModel (Zero-Velocity Update measurement model).

Tests verify that ZUPTModel correctly implements the MeasurementModel protocol
and provides the expected behavior for zero-velocity pseudo-measurements.
"""

import jax.numpy as jnp
import pytest

from trodestrack.models.sensors.protocols import MeasurementModel
from trodestrack.models.sensors.zupt import ZUPTModel
from trodestrack.models.state_layout import get_layout


class TestZUPTModelProtocol:
    """Test that ZUPTModel implements MeasurementModel protocol correctly."""

    def test_zupt_model_implements_protocol(self):
        """ZUPTModel should implement MeasurementModel protocol."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )
        assert isinstance(model, MeasurementModel)

    def test_meas_dim_is_two(self):
        """ZUPT measures 2D velocity (vx, vy)."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )
        assert model.meas_dim == 2


class TestZUPTModelPrediction:
    """Test ZUPT measurement prediction (velocity extraction)."""

    def test_predict_extracts_velocity(self):
        """predict() should extract [vx, vy] from state."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        # State: [x, y, vx, vy, θ, b_gz, b_ax, b_ay]
        state = jnp.array([0.5, 0.5, 0.15, -0.08, 0.0, 0.0, 0.0, 0.0])
        pred = model.predict(state)

        assert pred.shape == (2,)
        assert jnp.allclose(pred, jnp.array([0.15, -0.08]))

    def test_predict_works_with_different_layouts(self):
        """predict() should work with vision_only layout (indices differ)."""
        layout = get_layout("vision_only")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        # State: [x, y, vx, vy, θ] (5D)
        state = jnp.array([0.3, 0.4, 0.12, 0.09, 1.2])
        pred = model.predict(state)

        assert pred.shape == (2,)
        assert jnp.allclose(pred, jnp.array([0.12, 0.09]))


class TestZUPTModelJacobian:
    """Test ZUPT Jacobian (velocity selector matrix)."""

    def test_jacobian_selects_velocity(self):
        """jacobian() should return selector matrix H (2, n)."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        state = jnp.zeros(8)
        H = model.jacobian(state)

        assert H.shape == (2, 8)
        # Row 0 selects vx (index 2), Row 1 selects vy (index 3)
        expected_H = jnp.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert jnp.allclose(H, expected_H)


class TestZUPTModelMeasurementCovariance:
    """Test ZUPT measurement covariance (R matrix with gating logic)."""

    def test_meas_cov_small_when_stationary(self):
        """meas_cov() should return small R when velocity < threshold."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        # State with low velocity
        state_stationary = jnp.array([0.5, 0.5, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0])
        meas_pred = model.predict(state_stationary)
        R = model.meas_cov_from_pred(meas_pred)

        assert R.shape == (2, 2)
        # Should be diagonal with small noise
        assert jnp.allclose(R, jnp.diag(jnp.array([0.01**2, 0.01**2])), atol=1e-6)

    def test_meas_cov_large_when_moving(self):
        """meas_cov() should return large R when velocity > threshold."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        # State with high velocity
        state_moving = jnp.array([0.5, 0.5, 0.2, 0.15, 0.0, 0.0, 0.0, 0.0])
        meas_pred = model.predict(state_moving)
        R = model.meas_cov_from_pred(meas_pred)

        assert R.shape == (2, 2)
        # Should be diagonal with large noise (gated out)
        assert jnp.allclose(R, jnp.diag(jnp.array([1e6, 1e6])))

    def test_meas_cov_large_when_disabled(self):
        """meas_cov() should return large R when ZUPT is disabled."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=False,  # Disabled
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        # Even with low velocity, should gate out
        state_stationary = jnp.array([0.5, 0.5, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0])
        meas_pred = model.predict(state_stationary)
        R = model.meas_cov_from_pred(meas_pred)

        assert R.shape == (2, 2)
        assert jnp.allclose(R, jnp.diag(jnp.array([1e6, 1e6])))


class TestZUPTModelInnovation:
    """Test ZUPT innovation (negative of predicted velocity)."""

    def test_innovation_is_negative_velocity(self):
        """innovation() should return -[vx, vy] for ZUPT."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        # Stationary state with small velocity
        state = jnp.array([0.5, 0.5, 0.03, -0.02, 0.0, 0.0, 0.0, 0.0])
        meas_pred = model.predict(state)  # [0.03, -0.02]
        innov = model.innovation(frame_idx=0, meas_pred=meas_pred)

        assert innov.shape == (2,)
        # Innovation should be -predicted (since we're measuring zero velocity)
        assert jnp.allclose(innov, jnp.array([-0.03, 0.02]))


class TestZUPTModelSubspace:
    """Test ZUPT subspace (identity projection, no LED logic)."""

    def test_subspace_returns_identity_projection(self):
        """subspace() should return flags indicating no projection needed."""
        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        both_leds, only_led1, only_led2, projector = model.subspace(frame_idx=0)

        # ZUPT doesn't use LED projection
        assert both_leds is True  # "both valid" in the sense of no projection needed
        assert only_led1 is False
        assert only_led2 is False

        # Projector should be 2x2 identity for consistency
        assert projector.shape == (2, 2)
        assert jnp.allclose(projector, jnp.eye(2))


class TestZUPTModelNumericalStability:
    """Test ZUPT numerical stability and edge cases."""

    def test_velocity_threshold_edge_case(self):
        """ZUPT should handle velocity exactly at threshold correctly."""
        layout = get_layout("2d_full")
        threshold = 0.05
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=threshold,
            measurement_noise=0.01**2,
            layout=layout,
        )

        # Velocity magnitude exactly at threshold
        v_mag = threshold
        vx = v_mag / jnp.sqrt(2)
        vy = v_mag / jnp.sqrt(2)
        state = jnp.array([0.5, 0.5, vx, vy, 0.0, 0.0, 0.0, 0.0])
        meas_pred = model.predict(state)
        R = model.meas_cov_from_pred(meas_pred)

        # Should not crash or produce NaN
        assert jnp.all(jnp.isfinite(R))
        # Should be gated out (velocity >= threshold)
        assert jnp.allclose(R, jnp.diag(jnp.array([1e6, 1e6])))

    def test_jax_jit_compatible(self):
        """ZUPT implementation should be JAX JIT compatible."""
        from jax import jit

        layout = get_layout("2d_full")
        model = ZUPTModel(
            enable_zupt=True,
            velocity_threshold=0.05,
            measurement_noise=0.01**2,
            layout=layout,
        )

        state = jnp.array([0.5, 0.5, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0])

        @jit
        def test_fn(state):
            pred = model.predict(state)
            H = model.jacobian(state)
            R = model.meas_cov_from_pred(pred)
            return pred, H, R

        try:
            pred, H, R = test_fn(state)
            assert jnp.all(jnp.isfinite(pred))
            assert jnp.all(jnp.isfinite(H))
            assert jnp.all(jnp.isfinite(R))
        except Exception as e:
            if "ConcretizationError" in str(type(e)):
                pytest.fail(f"ZUPT implementation not JAX JIT compatible: {e}")
            raise
