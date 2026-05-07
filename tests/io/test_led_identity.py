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


def test_single_led_frames_inside_swapped_interval_are_relabeled():
    """Single-LED frames must inherit the surrounding swap state.

    Probe scenario: a global swap with ``initial_state='swapped'``,
    where frame 2 is single-LED only. The previous implementation
    only swapped frames where ``dual_valid`` was True, so the
    single-LED middle frame stayed in ``corrected.led1`` instead
    of moving to ``corrected.led2``. With state propagated to all
    frames, the single-LED frame is correctly relabeled.
    """

    n = 5
    t_cam = np.arange(n, dtype=float) / 30.0
    center = np.column_stack([0.01 * np.arange(n), np.zeros(n)])
    led1_true = center - np.array([0.02, 0.0])
    led2_true = center + np.array([0.02, 0.0])

    # Whole session is swapped relative to physical truth.
    led1_obs = led2_true.copy()
    led2_obs = led1_true.copy()
    # Frame 2 has only LED1 visible (in swapped labels). Physically
    # this single observation belongs at LED2.
    led2_obs[2] = np.nan
    mask = np.ones(n, dtype=bool)

    corrected = resolve_led_identity(
        t_cam,
        led1_obs,
        led2_obs,
        mask,
        led_distance=0.04,
        config=LedIdentityConfig(mode="auto", initial_state="swapped"),
    )

    # Every frame should be flagged as swapped.
    assert corrected.swapped.all()
    # The single-LED frame's finite observation moves from led1 to led2.
    assert np.isnan(corrected.led1[2]).all()
    np.testing.assert_allclose(corrected.led2[2], led1_obs[2])
    # Diagnostic count includes only dual-LED frames, not single-LED.
    assert corrected.diagnostics["dual_led_frame_count"] == 4


def test_no_dual_led_frames_emits_diagnostic_message():
    """No-dual-LED sessions should record a clear ambiguity diagnostic.

    The earlier early-return only emitted ``{'mode': 'auto',
    'n_swapped': 0}``. Add ``dual_led_frame_count``,
    ``global_identity_ambiguous``, and a message so callers can see
    why correction did nothing.
    """

    n = 4
    t_cam = np.arange(n, dtype=float) / 30.0
    led1 = np.full((n, 2), np.nan)
    led2 = np.full((n, 2), np.nan)
    mask = np.ones(n, dtype=bool)

    corrected = resolve_led_identity(
        t_cam,
        led1,
        led2,
        mask,
        led_distance=0.04,
        config=LedIdentityConfig(mode="auto"),
    )

    diag = corrected.diagnostics
    assert diag["dual_led_frame_count"] == 0
    assert diag["global_identity_ambiguous"] is True
    assert "no dual-LED frames" in str(diag["message"])
    assert not corrected.swapped.any()


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
