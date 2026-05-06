"""Regression tests for diagnostic-video LED labeling conventions."""

from __future__ import annotations

import numpy as np

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu
from trodestrack.viz.video import _led_label_direction_anomaly


def test_correctly_labeled_led_pair_is_not_flagged_as_swap():
    """LED2 in front, LED1 in rear should not flag as a directional swap.

    Without the convention fix in `_led_label_direction_anomaly`, every
    correctly-labeled frame in a clean simulation flagged as a swap (the
    helper computed (led1 - led2) and asserted dot > 0 with body +x;
    simulator convention is the opposite).
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

    # Heading at each camera frame, interpolated from IMU truth.
    theta_cam = np.interp(sim["t_cam_exp"], sim["t_imu"], sim["X_truth"][:, 4])

    n_anomalies = sum(
        _led_label_direction_anomaly(
            sim["Z_cam_led1"][i], sim["Z_cam_led2"][i], float(theta_cam[i])
        )
        for i in range(len(sim["Z_cam_led1"]))
    )
    assert n_anomalies == 0, (
        f"Clean simulation flagged {n_anomalies} frames as LED swaps; "
        "expected 0 with simulator convention LED1=rear, LED2=front."
    )


def test_label_swap_is_detected():
    """Swapping LED1/LED2 should make the helper return True."""
    cfg = RatIMUSimConfig(
        duration_s=1.0,
        cam_dropout_prob=0.0,
        cam_sigma_m=0.0,
        led_swap_prob=0.0,
        use_second_led=True,
        led_wall_reflection_prob=0.0,
    )
    sim = simulate_rat_imu(cfg, seed=0)
    theta_cam = np.interp(sim["t_cam_exp"], sim["t_imu"], sim["X_truth"][:, 4])

    # Pass LED1 and LED2 swapped — should flag as a direction anomaly on every
    # frame where the body has a non-zero forward extent.
    n_swap_flags = sum(
        _led_label_direction_anomaly(
            sim["Z_cam_led2"][i], sim["Z_cam_led1"][i], float(theta_cam[i])
        )
        for i in range(len(sim["Z_cam_led1"]))
    )
    assert n_swap_flags == len(sim["Z_cam_led1"]), (
        f"Swapped labels should flag every frame, got {n_swap_flags}/"
        f"{len(sim['Z_cam_led1'])}."
    )


def test_synthetic_pose_with_known_heading():
    """Synthetic LED pair with heading=0 (x-axis) and LED2 ahead in +x is OK."""
    led1 = np.array([0.0, 0.0])  # rear
    led2 = np.array([0.04, 0.0])  # front (along +x)
    assert not _led_label_direction_anomaly(led1, led2, theta=0.0)
    # Swap them → flagged.
    assert _led_label_direction_anomaly(led2, led1, theta=0.0)
