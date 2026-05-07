"""Tests for persistent LED identity correction."""

from __future__ import annotations

import numpy as np

from trodestrack.config import LedIdentityConfig
from trodestrack.io.led_identity import resolve_led_identity


def test_auto_led_identity_corrects_persistent_swap():
    """The DP should recover a contiguous swapped segment."""

    n = 30
    t_cam = np.arange(n, dtype=float) / 30.0
    center = np.column_stack([0.01 * np.arange(n), np.zeros(n)])
    led1_true = center - np.array([0.02, 0.0])
    led2_true = center + np.array([0.02, 0.0])
    led1_obs = led1_true.copy()
    led2_obs = led2_true.copy()
    led1_obs[10:20], led2_obs[10:20] = led2_true[10:20], led1_true[10:20]
    mask = np.ones(n, dtype=bool)

    corrected = resolve_led_identity(
        t_cam,
        led1_obs,
        led2_obs,
        mask,
        led_distance=0.04,
        config=LedIdentityConfig(mode="auto", transition_penalty=0.25),
    )

    assert np.array_equal(
        corrected.swapped, np.r_[np.zeros(10), np.ones(10), np.zeros(10)].astype(bool)
    )
    np.testing.assert_allclose(corrected.led1, led1_true)
    np.testing.assert_allclose(corrected.led2, led2_true)


def test_auto_led_identity_carries_dropout_frames_unchanged():
    """Missing dual-LED frames should not crash or invent finite positions."""

    n = 12
    t_cam = np.arange(n, dtype=float) / 30.0
    led1 = np.column_stack([0.01 * np.arange(n), np.zeros(n)])
    led2 = led1 + np.array([0.04, 0.0])
    mask = np.ones(n, dtype=bool)
    mask[5] = False
    led1[5] = np.nan
    led2[5] = np.nan

    corrected = resolve_led_identity(
        t_cam,
        led1,
        led2,
        mask,
        led_distance=0.04,
        config=LedIdentityConfig(mode="auto"),
    )

    assert not corrected.swapped[5]
    assert np.isnan(corrected.led1[5]).all()
    assert np.isnan(corrected.led2[5]).all()


def test_initial_swapped_prior_resolves_global_swap():
    """A whole-session swap needs an explicit initial identity prior."""

    n = 20
    t_cam = np.arange(n, dtype=float) / 30.0
    center = np.column_stack([0.01 * np.arange(n), np.zeros(n)])
    led1_true = center - np.array([0.02, 0.0])
    led2_true = center + np.array([0.02, 0.0])
    mask = np.ones(n, dtype=bool)

    ambiguous = resolve_led_identity(
        t_cam,
        led2_true,
        led1_true,
        mask,
        led_distance=0.04,
        config=LedIdentityConfig(mode="auto"),
    )
    assert not ambiguous.swapped.any()
    assert ambiguous.diagnostics["global_identity_ambiguous"] is True
    np.testing.assert_allclose(ambiguous.led1, led2_true)
    np.testing.assert_allclose(ambiguous.led2, led1_true)

    corrected = resolve_led_identity(
        t_cam,
        led2_true,
        led1_true,
        mask,
        led_distance=0.04,
        config=LedIdentityConfig(mode="auto", initial_state="swapped"),
    )

    assert corrected.swapped.all()
    assert corrected.diagnostics["initial_state"] == "swapped"
    assert corrected.diagnostics["global_identity_ambiguous"] is False
    np.testing.assert_allclose(corrected.led1, led1_true)
    np.testing.assert_allclose(corrected.led2, led2_true)
