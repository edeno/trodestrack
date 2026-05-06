"""Regression tests for diagnostic-video LED labeling conventions."""

from __future__ import annotations

import numpy as np

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.viz.video import _led_label_direction_anomaly

DEFAULT_LED1_OFFSET = np.array([-0.02, 0.0])  # Rear (simulator default)
DEFAULT_LED2_OFFSET = np.array([0.02, 0.0])  # Front


def _flag_count(sim, led1_offset, led2_offset, swap_args=False):
    """Count direction anomalies across a simulation."""
    theta_cam = np.interp(sim["t_cam_exp"], sim["t_imu"], sim["X_truth"][:, 4])
    n = len(sim["Z_cam_led1"])
    if swap_args:
        a, b = sim["Z_cam_led2"], sim["Z_cam_led1"]
    else:
        a, b = sim["Z_cam_led1"], sim["Z_cam_led2"]
    return sum(
        _led_label_direction_anomaly(
            a[i], b[i], float(theta_cam[i]), led1_offset, led2_offset
        )
        for i in range(n)
    ), n


def test_correctly_labeled_led_pair_is_not_flagged_as_swap():
    """Default offsets (LED1 rear, LED2 front) should not flag clean frames.

    Without the convention fix in `_led_label_direction_anomaly`, every
    correctly-labeled frame in a clean simulation flagged as a swap.
    """
    cfg = RatIMUSimConfig(
        duration_s=1.0,
        cam_dropout_prob=0.0,
        cam_sigma_m=0.0,
        led_swap_prob=0.0,
        use_second_led=True,
        led_wall_reflection_prob=0.0,
    )
    sim = simulate_rat_imu(cfg, seed=0)
    flagged, n = _flag_count(sim, DEFAULT_LED1_OFFSET, DEFAULT_LED2_OFFSET)
    assert flagged == 0, f"Clean default-offset sim flagged {flagged}/{n} as swaps."


def test_label_swap_is_detected_default_offsets():
    """Swapping LED1/LED2 inputs should flag every frame under default offsets."""
    cfg = RatIMUSimConfig(
        duration_s=1.0,
        cam_dropout_prob=0.0,
        cam_sigma_m=0.0,
        led_swap_prob=0.0,
        use_second_led=True,
        led_wall_reflection_prob=0.0,
    )
    sim = simulate_rat_imu(cfg, seed=0)
    flagged, n = _flag_count(
        sim, DEFAULT_LED1_OFFSET, DEFAULT_LED2_OFFSET, swap_args=True
    )
    assert flagged == n, f"Swapped labels should flag every frame, got {flagged}/{n}."


def test_custom_led1_front_ordering_is_not_flagged_as_swap():
    """Custom config (LED1 front, LED2 rear) should not flag clean frames.

    Hard-coding the default convention previously false-flagged 30/30 frames
    with this supported configuration.
    """
    led1_offset = np.array([0.025, 0.0])  # Front
    led2_offset = np.array([-0.025, 0.0])  # Rear
    cfg = RatIMUSimConfig(
        duration_s=1.0,
        cam_dropout_prob=0.0,
        cam_sigma_m=0.0,
        led_swap_prob=0.0,
        use_second_led=True,
        led_wall_reflection_prob=0.0,
        led1_offset_body=led1_offset,
        led2_offset_body=led2_offset,
    )
    sim = simulate_rat_imu(cfg, seed=0)
    flagged, n = _flag_count(sim, led1_offset, led2_offset)
    assert flagged == 0, (
        f"Custom-offset clean sim flagged {flagged}/{n} as swaps; "
        "expected 0 when offsets are passed through."
    )

    # And swapping the inputs should still flag every frame.
    flagged_swap, _ = _flag_count(sim, led1_offset, led2_offset, swap_args=True)
    assert flagged_swap == n


def test_synthetic_pose_with_known_heading():
    """Synthetic pose along +x: LED2 ahead is correct under default offsets."""
    led1 = np.array([0.0, 0.0])
    led2 = np.array([0.04, 0.0])
    assert not _led_label_direction_anomaly(
        led1, led2, 0.0, DEFAULT_LED1_OFFSET, DEFAULT_LED2_OFFSET
    )
    assert _led_label_direction_anomaly(
        led2, led1, 0.0, DEFAULT_LED1_OFFSET, DEFAULT_LED2_OFFSET
    )


def test_coincident_offsets_return_false():
    """If LED1/LED2 share offsets the direction check is undefined; return False."""
    same = np.array([0.0, 0.0])
    assert not _led_label_direction_anomaly(
        np.array([0.0, 0.0]), np.array([0.04, 0.0]), 0.0, same, same
    )


def test_create_diagnostic_video_accepts_simple_sim_config(tmp_path):
    """create_diagnostic_video must work with SimpleSimConfig sims.

    SimpleSimConfig has no led1_offset_body / led2_offset_body fields,
    but simulate_circular returns the same SimOut shape with both LEDs
    populated. Hard-coding ``config.led1_offset_body`` previously raised
    AttributeError; the video should now resolve LED offsets via a
    sensible default that matches what simulate_circular actually emits.
    """
    import matplotlib

    matplotlib.use("Agg")
    from trodestrack.sim.simple import SimpleSimConfig, simulate_circular
    from trodestrack.viz.video import create_diagnostic_video

    sim = simulate_circular(
        center=[0.5, 0.5],
        radius=0.3,
        angular_velocity=0.5,
        config=SimpleSimConfig(duration_s=2.0, cam_dropout_prob=0.0),
    )
    assert not hasattr(sim["config"], "led1_offset_body"), (
        "SimpleSimConfig should not expose led1_offset_body — guard relies on getattr"
    )

    from pathlib import Path as _Path

    out_path = tmp_path / "diagnostic.mp4"
    # Low fps + short duration so the test stays fast; we only verify it
    # doesn't raise AttributeError on the missing offset fields.
    result = create_diagnostic_video(sim, str(out_path), fps=5, speedup=10.0)
    # ``return_animation=False`` (default) → just a Path is returned. The
    # mp4 codec may not be available in CI; fall back to .gif sibling.
    result_path = result if isinstance(result, _Path) else result[0]
    assert result_path.exists(), f"video output {result_path} not written"
