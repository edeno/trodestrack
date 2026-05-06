"""Regression tests for diagnostic-video LED labeling conventions."""

from __future__ import annotations

import numpy as np

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.viz.video import _led_label_direction_anomaly, _predict_led_world

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


def test_predict_led_world_translates_and_rotates_offset():
    """LED world prediction = position + R(theta) @ offset_body."""
    pos = np.array([1.0, 2.0])
    # theta=0 → no rotation, just translation by offset.
    np.testing.assert_allclose(
        _predict_led_world(pos, 0.0, np.array([0.025, 0.0])),
        np.array([1.025, 2.0]),
        atol=1e-12,
    )
    # theta=pi/2 → +x body axis maps to +y world axis.
    np.testing.assert_allclose(
        _predict_led_world(pos, np.pi / 2, np.array([0.025, 0.0])),
        np.array([1.0, 2.025]),
        atol=1e-12,
    )
    # Rear LED under default convention should land *behind* the position
    # along heading +x.
    np.testing.assert_allclose(
        _predict_led_world(pos, 0.0, DEFAULT_LED1_OFFSET),
        np.array([1.0 - 0.02, 2.0]),
        atol=1e-12,
    )


def test_residual_prediction_uses_custom_offsets_not_default_convention():
    """Perfect filter on a custom-offset sim must yield ~0 residuals.

    Previously the residual panel computed predicted LED positions as
    ``±0.5 * led_distance`` along heading, hard-coding the default
    LED1=rear / LED2=front convention. With the user's custom config
    (LED1 in front, LED2 in rear), a *perfect* filter produced ~4.5 cm
    residuals — pure visualization-model mismatch, not filter error.
    """
    from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu

    led1_offset = np.array([0.025, 0.0])  # Front
    led2_offset = np.array([-0.025, 0.0])  # Rear
    cfg = RatIMUSimConfig(
        duration_s=2.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,
        cam_sigma_m=0.0,
        led_swap_prob=0.0,
        led_wall_reflection_prob=0.0,
        use_second_led=True,
        led1_offset_body=led1_offset,
        led2_offset_body=led2_offset,
    )
    sim = simulate_rat_imu(cfg, seed=0)

    # Compute residuals using the helper at every frame against the
    # nearest IMU truth sample. With zero camera noise the residual must
    # be small (only IMU↔camera-frame interpolation jitter).
    t_imu = sim["t_imu"]
    t_cam = sim["t_cam_exp"]
    X_truth = sim["X_truth"]

    residuals = []
    for i, t in enumerate(t_cam):
        j = int(min(np.searchsorted(t_imu, t), len(t_imu) - 1))
        position = X_truth[j, :2]
        theta = float(X_truth[j, 4])
        led1_pred = _predict_led_world(position, theta, led1_offset)
        led2_pred = _predict_led_world(position, theta, led2_offset)
        residuals.append(float(np.linalg.norm(sim["Z_cam_led1"][i] - led1_pred)))
        residuals.append(float(np.linalg.norm(sim["Z_cam_led2"][i] - led2_pred)))

    max_residual_cm = float(np.max(residuals)) * 100
    # The previous (buggy) code produced ~4.5 cm residuals here. The
    # interpolation-jitter floor is well under 1 cm at fs_imu=200,
    # fs_cam=30, so a 1 cm gate is comfortable and locks out the bug.
    assert max_residual_cm < 1.0, (
        f"max residual {max_residual_cm:.3f} cm exceeds 1 cm — "
        "looks like the hard-coded ±0.5*led_distance convention is back."
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


def test_video_event_detection_iterates_all_camera_frames(tmp_path, caplog):
    """Event markers must be derived from all camera frames, not video samples.

    Previously the event-detection loop iterated rendered video frames
    (``range(n_frames)``), which at high speedup / low fps samples only
    a fraction of camera frames. A ground-truth swap that fell between
    sampled video frames was silently dropped from the progress-bar
    markers. This test runs the production ``create_diagnostic_video``
    pipeline and inspects its event-count log line — anything weaker
    (e.g. asserting only on ``sim["swap_applied"].sum() == 1``) would
    pass even if the production loop reverted to ``range(n_frames)``.
    """
    import logging

    import matplotlib

    matplotlib.use("Agg")

    from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
    from trodestrack.viz.utils import prepare_video_data
    from trodestrack.viz.video import create_diagnostic_video

    cfg = RatIMUSimConfig(
        duration_s=4.0,
        fs_imu=200.0,
        fs_cam=30.0,
        cam_dropout_prob=0.0,
        use_second_led=True,
        led_swap_prob=0.0,
        led_wall_reflection_prob=0.0,
    )
    sim = simulate_rat_imu(cfg, seed=0)

    # Inject a single ground-truth swap at t≈1.0s. With fps=5 and
    # speedup=10 the rendered video samples sit at t = 0, 2, 4 s, so the
    # buggy video-frame loop would never visit cam_idx=30.
    target_idx = int(np.argmin(np.abs(sim["t_cam_exp"] - 1.0)))
    sim["swap_applied"][target_idx] = True
    assert sim["t_cam_exp"][target_idx] != 0.0

    # Sanity: with these settings the rendered video samples skip the
    # injected event entirely (the original bug condition). This makes
    # the test setup actually exercise the bug — without this gate the
    # detection might "succeed" by accidentally sampling the right
    # camera frame.
    video_data = prepare_video_data(sim, fps=5, speedup=10.0)
    rendered_cam_indices = {int(c) for c in np.asarray(video_data["cam_idx"]).tolist()}
    assert target_idx not in rendered_cam_indices, (
        "test setup precondition: target frame should not be a sampled "
        "video frame so the regression actually exercises the bug."
    )

    # Run the production pipeline and capture the log line that reports
    # the (debounced) event counts. Under the bug this would say
    # "Found 0 LED swaps, ..."; the fix routes the count through the
    # full camera-rate mask scan and yields "Found 1 LED swaps, ...".
    out_path = tmp_path / "diagnostic.mp4"
    with caplog.at_level(logging.INFO, logger="trodestrack.viz.video"):
        create_diagnostic_video(sim, str(out_path), fps=5, speedup=10.0)

    swap_log_lines = [
        rec.getMessage()
        for rec in caplog.records
        if "LED swap" in rec.getMessage() and "debounced" in rec.getMessage()
    ]
    assert swap_log_lines, (
        "expected an event-count log line containing 'LED swap' and "
        f"'debounced', got records: {[r.getMessage() for r in caplog.records]}"
    )
    assert "Found 1 LED swaps" in swap_log_lines[-1], (
        f"event detector missed the injected swap: {swap_log_lines[-1]!r}"
    )

    # Reproduce the new camera-frame walk and confirm it picks up the swap.
    swap_mask = np.asarray(sim["swap_applied"])
    n_swaps = int(swap_mask.sum())
    assert n_swaps == 1
    assert bool(swap_mask[target_idx])
