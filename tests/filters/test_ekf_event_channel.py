"""Parity tests for the optional TTL-event channel in ``extended_kalman_filter``.

These tests verify that:
- With no event arguments, the filter is bitwise identical to before.
- With event arguments configured but no actual events firing
  (``event_indices_per_frame`` filled with ``-1``), output is identical to
  the no-event-arguments case.
- A single zone-trigger event pulls the filtered position toward the zone
  anchor (smoke-level integration).
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from trodestrack.sim.simple import SimpleSimConfig, simulate_constant_velocity


@pytest.fixture
def sim_data():
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


def _make_ekf_config() -> EKFConfig:
    return EKFConfig(
        state_mode="2d_full",
        process_noise_pos=0.001,
        process_noise_vel=0.5,
        process_noise_heading=0.5,
        process_noise_gyro_bias=7.6e-7,
        process_noise_accel_bias=2.4e-9,
        measurement_noise_pos=0.005**2,
        measurement_noise_heading=0.5**2,
        led_distance=0.04,
        use_heading_measurement=True,
        damping_coeff=0.4,
    )


def _run_baseline(sim, ekf_config):
    return extended_kalman_filter(
        ekf_config,
        sim["t_imu"],
        sim["U_imu"],
        sim["t_cam_exp"],
        sim["Z_cam_led1"],
        sim["Z_cam_led2"],
        sim["mask_cam"],
    )


def _run_with_events(sim, ekf_config, anchors, covariances, indices_per_frame):
    return extended_kalman_filter(
        ekf_config,
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


class TestEventChannelParity:
    def test_no_event_args_unchanged(self, sim_data):
        """Sanity: omitting all event args returns valid output."""
        ekf_config = _make_ekf_config()
        result = _run_baseline(sim_data, ekf_config)
        assert np.all(np.isfinite(np.asarray(result.filtered_means)))

    def test_empty_events_identical_to_no_args(self, sim_data):
        """Event args present but every slot ``-1`` ⇒ no-op equivalence."""
        ekf_config = _make_ekf_config()
        baseline = _run_baseline(sim_data, ekf_config)

        n_cam = sim_data["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        covariances = np.array([[[0.02**2, 0.0], [0.0, 0.02**2]]], dtype=float)
        indices = np.full((n_cam, 4), -1, dtype=np.int32)
        events_run = _run_with_events(
            sim_data, ekf_config, anchors, covariances, indices
        )

        np.testing.assert_allclose(
            np.asarray(baseline.filtered_means),
            np.asarray(events_run.filtered_means),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(baseline.filtered_covariances),
            np.asarray(events_run.filtered_covariances),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            float(baseline.marginal_loglik),
            float(events_run.marginal_loglik),
            atol=1e-4,
        )


class TestEventChannelInfluence:
    def test_zone_trigger_pulls_position(self, sim_data):
        """A zone trigger fires once and pulls posterior toward the anchor."""
        ekf_config = _make_ekf_config()
        baseline = _run_baseline(sim_data, ekf_config)

        n_cam = sim_data["t_cam_exp"].shape[0]
        # Anchor offset from the trajectory by 5 cm.
        baseline_pos_at_mid = np.asarray(baseline.filtered_means)[n_cam // 2, 0:2]
        anchor = baseline_pos_at_mid + np.array([0.05, 0.05])

        anchors = np.array([anchor], dtype=float)
        covariances = np.array([[[0.005**2, 0.0], [0.0, 0.005**2]]], dtype=float)
        # One event at the middle camera frame (compact source index 0).
        indices = np.full((n_cam, 4), -1, dtype=np.int32)
        indices[n_cam // 2, 0] = 0

        events_run = _run_with_events(
            sim_data, ekf_config, anchors, covariances, indices
        )

        new_pos = np.asarray(events_run.filtered_means)[n_cam // 2, 0:2]
        # Posterior position moved meaningfully toward the anchor.
        baseline_to_anchor = np.linalg.norm(baseline_pos_at_mid - anchor)
        new_to_anchor = np.linalg.norm(new_pos - anchor)
        assert new_to_anchor < baseline_to_anchor


class TestEventChannelValidation:
    def test_partial_args_rejected(self, sim_data):
        ekf_config = _make_ekf_config()
        n_cam = sim_data["t_cam_exp"].shape[0]
        with pytest.raises(ValueError, match="must be provided together"):
            extended_kalman_filter(
                ekf_config,
                sim_data["t_imu"],
                sim_data["U_imu"],
                sim_data["t_cam_exp"],
                sim_data["Z_cam_led1"],
                sim_data["Z_cam_led2"],
                sim_data["mask_cam"],
                event_source_anchors=np.zeros((1, 2)),
                event_source_covariances=None,
                event_indices_per_frame=np.full((n_cam, 1), -1, dtype=np.int32),
            )

    def test_index_out_of_range_rejected(self, sim_data):
        ekf_config = _make_ekf_config()
        n_cam = sim_data["t_cam_exp"].shape[0]
        anchors = np.zeros((1, 2))
        covariances = np.broadcast_to(np.eye(2) * 0.01, (1, 2, 2)).copy()
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        indices[0, 0] = 5  # only 1 source configured
        with pytest.raises(ValueError, match="out of"):
            _run_with_events(sim_data, ekf_config, anchors, covariances, indices)
