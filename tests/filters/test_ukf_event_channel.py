"""Parity / influence / validation tests for the UKF TTL event channel.

Mirrors ``test_ekf_event_channel.py`` for the unscented filter so the
two paths share the same boundary-validation contract and the same
empty-channel parity guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.models.ukf import UKFConfig, unscented_kalman_filter


class TestEventChannelParity:
    def test_no_event_args_unchanged(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_baseline,
    ):
        ukf_config = UKFConfig(**event_channel_filter_defaults)
        result = run_filter_baseline(
            unscented_kalman_filter, ukf_config, event_channel_sim
        )
        assert np.all(np.isfinite(np.asarray(result.filtered_means)))

    def test_empty_events_identical_to_no_args(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_baseline,
        run_filter_with_events,
    ):
        """Event args present but every slot ``-1`` ⇒ no-op equivalence.

        Uses ``atol`` rather than ``assert_array_equal`` for the same
        reason as the EKF test: the no-args path runs the JIT'd core
        with ``max_events_per_frame=1`` while this path may run with a
        larger pad width, retracing under different XLA orderings.
        """
        ukf_config = UKFConfig(**event_channel_filter_defaults)
        baseline = run_filter_baseline(
            unscented_kalman_filter, ukf_config, event_channel_sim
        )

        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        covariances = np.array([[[0.02**2, 0.0], [0.0, 0.02**2]]], dtype=float)
        indices = np.full((n_cam, 4), -1, dtype=np.int32)
        events_run = run_filter_with_events(
            unscented_kalman_filter,
            ukf_config,
            event_channel_sim,
            anchors,
            covariances,
            indices,
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
    def test_zone_trigger_pulls_position(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_baseline,
        run_filter_with_events,
    ):
        ukf_config = UKFConfig(**event_channel_filter_defaults)
        baseline = run_filter_baseline(
            unscented_kalman_filter, ukf_config, event_channel_sim
        )

        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        baseline_pos_at_mid = np.asarray(baseline.filtered_means)[n_cam // 2, 0:2]
        anchor = baseline_pos_at_mid + np.array([0.05, 0.05])

        anchors = np.array([anchor], dtype=float)
        covariances = np.array([[[0.005**2, 0.0], [0.0, 0.005**2]]], dtype=float)
        indices = np.full((n_cam, 4), -1, dtype=np.int32)
        indices[n_cam // 2, 0] = 0

        events_run = run_filter_with_events(
            unscented_kalman_filter,
            ukf_config,
            event_channel_sim,
            anchors,
            covariances,
            indices,
        )

        new_pos = np.asarray(events_run.filtered_means)[n_cam // 2, 0:2]
        baseline_to_anchor = np.linalg.norm(baseline_pos_at_mid - anchor)
        new_to_anchor = np.linalg.norm(new_pos - anchor)
        assert new_to_anchor < baseline_to_anchor


class TestEventChannelValidation:
    """The validation surface is shared with the EKF, so these tests
    primarily confirm that the UKF wrapper threads the validator (with
    the correct ``func_name``) and that the same shape/dtype/range
    rules apply at the UKF API boundary."""

    def test_partial_args_rejected(
        self, event_channel_sim, event_channel_filter_defaults
    ):
        ukf_config = UKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        with pytest.raises(ValueError, match="unscented_kalman_filter event channel"):
            unscented_kalman_filter(
                ukf_config,
                event_channel_sim["t_imu"],
                event_channel_sim["U_imu"],
                event_channel_sim["t_cam_exp"],
                event_channel_sim["Z_cam_led1"],
                event_channel_sim["Z_cam_led2"],
                event_channel_sim["mask_cam"],
                event_source_anchors=np.zeros((1, 2)),
                event_source_covariances=None,
                event_indices_per_frame=np.full((n_cam, 1), -1, dtype=np.int32),
            )

    def test_negative_definite_covariance_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        ukf_config = UKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        bad_cov = np.array([[[-0.0001, 0.0], [0.0, -0.0001]]], dtype=float)
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        with pytest.raises(ValueError, match="positive-definite"):
            run_filter_with_events(
                unscented_kalman_filter,
                ukf_config,
                event_channel_sim,
                anchors,
                bad_cov,
                indices,
            )

    def test_index_out_of_range_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        ukf_config = UKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.zeros((1, 2))
        covariances = np.broadcast_to(np.eye(2) * 0.01, (1, 2, 2)).copy()
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        indices[0, 0] = 5
        with pytest.raises(ValueError, match="out of"):
            run_filter_with_events(
                unscented_kalman_filter,
                ukf_config,
                event_channel_sim,
                anchors,
                covariances,
                indices,
            )
