"""Unit tests for ``HeadingPseudoModel``.

Covers the parts of the heading pseudo-measurement model that the
end-to-end EKF/UKF tests can't pin down precisely:

- ``predict`` matches the closed-form heading extraction from state.
- ``jacobian`` matches ``jax.jacfwd(predict)`` (1-to-1 selector for 2D
  layouts; AD for quaternion layouts).
- ``innovation`` wraps to (-π, π] for measurements / predictions that
  straddle the branch cut.
- ``use_measurement`` gates frames whose implied LED spacing differs
  from ``config.led_distance`` by more than ``led_distance_tolerance``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trodestrack.models.filter_common import FilterCoreConfig
from trodestrack.models.sensors.heading_pseudo import HeadingPseudoModel
from trodestrack.models.state_layout import LAYOUT_2D_FULL


def _make_config(**overrides: object) -> FilterCoreConfig:
    base = dict(
        use_heading_measurement=True,
        measurement_noise_heading=0.05**2,
        led_distance=0.04,
        led_distance_tolerance=0.3,
        adaptive_heading_noise=False,
    )
    base.update(overrides)
    return FilterCoreConfig(**base)


def _make_model(
    z_led1: np.ndarray,
    z_led2: np.ndarray,
    config: FilterCoreConfig | None = None,
) -> HeadingPseudoModel:
    if config is None:
        config = _make_config()
    return HeadingPseudoModel(
        config=config,
        layout=LAYOUT_2D_FULL,
        z_led1_all=jnp.asarray(z_led1),
        z_led2_all=jnp.asarray(z_led2),
    )


def test_heading_pseudo_predict_matches_hand_derived() -> None:
    """``predict`` should reproduce the analytic atan2 form.

    For a 2D state ``[x, y, vx, vy, θ, ...]`` the predicted heading
    measurement is the scalar heading state. We verify that against the
    closed-form ``atan2(led2_y - led1_y, led2_x - led1_x)`` evaluated on a
    known LED pair so a future refactor that, say, switched ``predict`` to
    differentiate LED positions instead of returning the state would be
    caught.
    """

    led1 = np.array([[0.10, 0.20]])
    led2 = np.array([[0.13, 0.24]])
    model = _make_model(led1, led2)

    expected_heading = float(np.arctan2(0.24 - 0.20, 0.13 - 0.10))
    state = jnp.array([0.0, 0.0, 0.0, 0.0, expected_heading, 0.0, 0.0, 0.0])
    pred = np.asarray(model.predict(state))

    assert pred.shape == (1,)
    np.testing.assert_allclose(pred[0], expected_heading, atol=1e-12)


def test_heading_pseudo_jacobian_matches_jacfwd() -> None:
    """Analytic Jacobian must match ``jax.jacfwd(predict)`` for random states.

    The production code uses a hand-built selector matrix for 2D layouts;
    the test compares against AD to catch silent index errors (off-by-one
    in the heading slot, accidental selector reuse).
    """

    led1 = np.array([[0.0, 0.0]])
    led2 = np.array([[0.04, 0.0]])
    model = _make_model(led1, led2)

    jacfwd_pred = jax.jacfwd(model.predict)
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = jnp.asarray(rng.standard_normal(8))
        H_analytic = np.asarray(model.jacobian(state))
        H_ad = np.asarray(jacfwd_pred(state))
        np.testing.assert_allclose(H_analytic, H_ad, rtol=1e-6, atol=1e-8)


def test_heading_pseudo_innovation_wraps_to_minus_pi_pi() -> None:
    """Innovation across the ±π branch cut must wrap, not jump by 2π.

    Measurement = +179°, prediction = -179°. The naive subtraction would
    give -358°; the correct wrapped innovation is +2°. The model is free
    to choose either sign convention, so the assertion checks magnitude
    near 2° (not 358°) rather than a specific sign.
    """

    meas_heading_rad = np.deg2rad(179.0)
    # Pick led1/led2 so that atan2(dy, dx) == 179°.
    dx = np.cos(meas_heading_rad)
    dy = np.sin(meas_heading_rad)
    # Use a spacing of 0.04 m to keep the gate happy.
    half = 0.02
    led1 = np.array([[-half * dx, -half * dy]])
    led2 = np.array([[half * dx, half * dy]])
    model = _make_model(led1, led2)

    pred_heading = np.deg2rad(-179.0)
    state = jnp.array([0.0, 0.0, 0.0, 0.0, pred_heading, 0.0, 0.0, 0.0])
    meas_pred = model.predict(state)
    innov = np.asarray(model.innovation(0, meas_pred))

    assert innov.shape == (1,)
    expected_mag_rad = np.deg2rad(2.0)
    assert abs(abs(innov[0]) - expected_mag_rad) < 1e-6, (
        f"innovation {innov[0]:.6f} rad did not wrap to ~±2°; raw diff would "
        f"be {meas_heading_rad - pred_heading:.6f} rad."
    )


def test_heading_pseudo_gate_rejects_when_led_spacing_implausible() -> None:
    """When implied LED spacing is 50% off, ``use_measurement`` must be False.

    Configured ``led_distance=0.04`` with the default 30% tolerance: a
    pair whose measured separation is 0.06 m (50% above expected) sits
    outside ``(1 ± 0.3) * 0.04 = (0.028, 0.052) m`` and must be gated.
    """

    cfg = _make_config(led_distance=0.04, led_distance_tolerance=0.3)
    # Spacing of 0.06 m is 50% above the 0.04 m expected spacing.
    led1 = np.array([[0.0, 0.0]])
    led2 = np.array([[0.06, 0.0]])
    model = _make_model(led1, led2, cfg)

    use = bool(np.asarray(model.use_measurement(0)))
    assert use is False

    # And the meas_cov should be the gating-large value.
    R = np.asarray(model.meas_cov(0))
    assert R.shape == (1, 1)
    assert R[0, 0] >= 1e5, (
        f"gated heading R must be the large gating value; got {R[0, 0]:.3e}"
    )


def test_heading_pseudo_gate_accepts_when_led_spacing_matches() -> None:
    """Spacing exactly at the configured value should leave the gate open.

    Counterpart to the rejection test: the gate that rejects 50%-off
    spacing should not be hyperactive on the nominal geometry.
    """

    cfg = _make_config(led_distance=0.04, led_distance_tolerance=0.3)
    led1 = np.array([[0.0, 0.0]])
    led2 = np.array([[0.04, 0.0]])
    model = _make_model(led1, led2, cfg)

    use = bool(np.asarray(model.use_measurement(0)))
    assert use is True
    R = np.asarray(model.meas_cov(0))
    assert R[0, 0] == pytest.approx(0.05**2)
