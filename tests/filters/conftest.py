"""Shared fixtures for filter-level tests."""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.sim.simple import SimpleSimConfig, simulate_constant_velocity


@pytest.fixture
def event_channel_sim():
    """Constant-velocity sim used by both EKF and UKF event-channel tests."""
    cfg = SimpleSimConfig(
        duration_s=2.0,
        fs_imu=200.0,
        fs_cam=30.0,
        gyro_noise_density=0.0005,
        accel_noise_density=0.02,
        gyro_bias_std=0.01,
        accel_bias_std=0.02,
        cam_noise_std=0.005,
        cam_dropout_prob=0.0,
    )
    return simulate_constant_velocity(cfg, velocity=np.array([0.2, 0.0]))


@pytest.fixture
def event_channel_filter_defaults():
    """Filter-config kwargs shared by EKF and UKF event-channel tests.

    Only filter-agnostic tunings (no ``num_iter``); both ``EKFConfig`` and
    ``UKFConfig`` accept this dict via ``**kwargs``.
    """
    return {
        "state_mode": "2d_full",
        "process_noise_pos": 0.001,
        "process_noise_vel": 0.5,
        "process_noise_heading": 0.5,
        "process_noise_gyro_bias": 7.6e-7,
        "process_noise_accel_bias": 2.4e-9,
        "measurement_noise_pos": 0.005**2,
        "measurement_noise_heading": 0.5**2,
        "led_distance": 0.04,
        "use_heading_measurement": True,
        "damping_coeff": 0.4,
    }


@pytest.fixture
def run_filter_baseline():
    """Return a closure that runs a filter with no event arguments."""

    def _run(filter_fn, filter_config, sim):
        return filter_fn(
            filter_config,
            sim["t_imu"],
            sim["U_imu"],
            sim["t_cam_exp"],
            sim["Z_cam_led1"],
            sim["Z_cam_led2"],
            sim["mask_cam"],
        )

    return _run


@pytest.fixture
def run_filter_with_events():
    """Return a closure that runs a filter with the given event arrays."""

    def _run(filter_fn, filter_config, sim, anchors, covariances, indices_per_frame):
        return filter_fn(
            filter_config,
            sim["t_imu"],
            sim["U_imu"],
            sim["t_cam_exp"],
            sim["Z_cam_led1"],
            sim["Z_cam_led2"],
            sim["mask_cam"],
            event_source_anchors=anchors,
            event_source_covariances=covariances,
            event_indices_per_frame=indices_per_frame,
        )

    return _run
