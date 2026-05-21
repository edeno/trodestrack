"""Unit tests for ``CameraPositionModel``.

Covers the dual-LED position model's analytic forms and partial-observation
handling that the end-to-end EKF/UKF tests can't pin down:

- ``predict`` matches the closed-form rotated-offset formula.
- ``jacobian`` matches ``jax.jacfwd(predict)`` for random states.
- ``innovation`` / ``subspace`` correctly mark a single-LED frame: the
  innovation entries for the missing LED collapse to zero residual
  (projection-only approach), and ``subspace`` reports the LED1-only
  flag plus the selector matrix that picks the LED1 rows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.state_layout import LAYOUT_2D_FULL

LED_DISTANCE = 0.04
MEAS_NOISE_BASE = 0.005**2


def _make_model(
    z_led1: np.ndarray,
    z_led2: np.ndarray,
    conf_all: np.ndarray | None = None,
) -> CameraPositionModel:
    return CameraPositionModel(
        led_distance=LED_DISTANCE,
        measurement_noise_base=MEAS_NOISE_BASE,
        layout=LAYOUT_2D_FULL,
        z_led1_all=jnp.asarray(z_led1),
        z_led2_all=jnp.asarray(z_led2),
        conf_all=None if conf_all is None else jnp.asarray(conf_all),
    )


def test_camera_position_predict_matches_hand_derived() -> None:
    """predict() must match the analytic rotated-offset formula.

    For state ``[x, y, vx, vy, θ, ...]``, the dual-LED prediction is
    ``[x - d cosθ, y - d sinθ, x + d cosθ, y + d sinθ]`` with
    ``d = led_distance / 2`` (see filter_common.measurement_function).
    The check uses a non-trivial heading + position so a regression that
    swapped sin/cos or dropped the half-spacing factor would surface.
    """

    led1 = np.zeros((1, 2))
    led2 = np.zeros((1, 2))
    model = _make_model(led1, led2)

    x, y = 0.3, -0.2
    theta = np.deg2rad(30.0)
    state = jnp.array([x, y, 0.0, 0.0, theta, 0.0, 0.0, 0.0])
    pred = np.asarray(model.predict(state))

    d = LED_DISTANCE / 2.0
    expected = np.array(
        [
            x - d * np.cos(theta),
            y - d * np.sin(theta),
            x + d * np.cos(theta),
            y + d * np.sin(theta),
        ]
    )
    np.testing.assert_allclose(pred, expected, atol=1e-12)


def test_camera_position_jacobian_matches_jacfwd() -> None:
    """Analytic Jacobian must agree with ``jax.jacfwd(predict)`` for random states."""

    model = _make_model(np.zeros((1, 2)), np.zeros((1, 2)))
    jacfwd_pred = jax.jacfwd(model.predict)
    rng = np.random.default_rng(1)
    for _ in range(20):
        state = jnp.asarray(rng.standard_normal(8))
        H_analytic = np.asarray(model.jacobian(state))
        H_ad = np.asarray(jacfwd_pred(state))
        np.testing.assert_allclose(H_analytic, H_ad, rtol=1e-6, atol=1e-8)


def test_camera_position_partial_observation_single_led() -> None:
    """Single-LED frame: LED2 masked → zero residual on led2, LED1 selector.

    The production model uses a projection-only approach: ``predict``
    always returns finite (4,) measurements computed from state, and
    ``innovation`` replaces NaN observation components with the
    prediction (yielding zero residual for those slots). The lifted
    update then uses ``subspace`` to extract the 2D LED1 subspace.
    """

    # Predicted LED1 at state (x, y, θ=0) is (x - d, y) — give the
    # observation the exact predicted value so the LED1 residual is zero.
    x, y = 0.30, 0.20
    d = LED_DISTANCE / 2.0
    led1 = np.array([[x - d, y]])
    led2 = np.array([[np.nan, np.nan]])
    model = _make_model(led1, led2)

    state = jnp.array([x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pred = np.asarray(model.predict(state))
    # Prediction itself is finite for both LEDs — partial-observation
    # handling lives in innovation/subspace, not predict.
    assert np.all(np.isfinite(pred))

    innov = np.asarray(model.innovation(0, jnp.asarray(pred)))
    # Valid LED1 slots: observation matches prediction → zero residual.
    np.testing.assert_allclose(innov[:2], np.zeros(2), atol=1e-6)
    # Missing LED2 slots: NaN observation replaced with prediction → zero residual.
    np.testing.assert_allclose(innov[2:], np.zeros(2), atol=1e-6)
    assert np.all(np.isfinite(innov))

    both, only1, only2, selector = model.subspace(0)
    assert bool(np.asarray(both)) is False
    assert bool(np.asarray(only1)) is True
    assert bool(np.asarray(only2)) is False
    # Selector for LED1 is the (2, 4) matrix that picks the first 2 dims.
    expected_selector = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(selector), expected_selector, atol=1e-12)


def test_camera_position_partial_observation_innovation_carries_led1_residual() -> None:
    """Non-trivial LED1 observation must produce the matching LED1 residual.

    Complements the masked-LED2 test: when LED1's observation differs
    from the predicted LED1 position by a known amount, that delta must
    appear in innovation slots ``[0:2]`` while LED2 slots stay at zero.
    Catches a regression where partial-observation handling collapses
    every innovation entry to zero.
    """

    pred_state_xy = np.array([0.30, 0.20])
    delta = np.array([0.01, -0.005])
    led1_obs = pred_state_xy - LED_DISTANCE / 2.0 * np.array([1.0, 0.0]) + delta
    led1 = led1_obs.reshape(1, 2)
    led2 = np.array([[np.nan, np.nan]])
    model = _make_model(led1, led2)

    state = jnp.array([*pred_state_xy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pred = model.predict(state)
    innov = np.asarray(model.innovation(0, pred))

    np.testing.assert_allclose(innov[:2], delta, atol=1e-6)
    np.testing.assert_allclose(innov[2:], np.zeros(2), atol=1e-6)
