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


class TestEventChannelParity:
    def test_no_event_args_unchanged(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_baseline,
    ):
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        result = run_filter_baseline(
            extended_kalman_filter, ekf_config, event_channel_sim
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

        Uses a small ``atol`` rather than ``assert_array_equal``: the no-args
        path runs the JIT'd core with ``max_events_per_frame=1`` (the default
        no-events fallback), while this path runs it with
        ``max_events_per_frame=4``. ``max_events_per_frame`` is a JIT static
        argument, so the two cases retrace and XLA can choose different but
        equivalent floating-point orderings even though the event-update
        contribution is exactly zero in both paths.
        """
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        baseline = run_filter_baseline(
            extended_kalman_filter, ekf_config, event_channel_sim
        )

        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        covariances = np.array([[[0.02**2, 0.0], [0.0, 0.02**2]]], dtype=float)
        indices = np.full((n_cam, 4), -1, dtype=np.int32)
        events_run = run_filter_with_events(
            extended_kalman_filter,
            ekf_config,
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
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        baseline = run_filter_baseline(
            extended_kalman_filter, ekf_config, event_channel_sim
        )

        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        baseline_pos_at_mid = np.asarray(baseline.filtered_means)[n_cam // 2, 0:2]
        anchor = baseline_pos_at_mid + np.array([0.05, 0.05])

        anchors = np.array([anchor], dtype=float)
        covariances = np.array([[[0.005**2, 0.0], [0.0, 0.005**2]]], dtype=float)
        indices = np.full((n_cam, 4), -1, dtype=np.int32)
        indices[n_cam // 2, 0] = 0

        events_run = run_filter_with_events(
            extended_kalman_filter,
            ekf_config,
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
    def test_partial_args_rejected(
        self, event_channel_sim, event_channel_filter_defaults
    ):
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        with pytest.raises(ValueError, match="must be provided together"):
            extended_kalman_filter(
                ekf_config,
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
        """Direct callers must not be able to pass a non-PSD R into the JIT'd core."""
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        bad_cov = np.array([[[-0.0001, 0.0], [0.0, -0.0001]]], dtype=float)
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        with pytest.raises(ValueError, match="positive-definite"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                anchors,
                bad_cov,
                indices,
            )

    def test_asymmetric_covariance_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        asymmetric = np.array([[[0.01, 0.005], [-0.005, 0.01]]], dtype=float)
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        with pytest.raises(ValueError, match="symmetric"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                anchors,
                asymmetric,
                indices,
            )

    def test_fractional_index_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        """Float indices like 0.9 must not silently truncate to a valid source."""
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        covariances = np.array([[[0.01, 0.0], [0.0, 0.01]]], dtype=float)
        indices = np.full((n_cam, 1), -1, dtype=float)
        indices[0, 0] = 0.9
        with pytest.raises(ValueError, match="non-integer entries"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                anchors,
                covariances,
                indices,
            )

    def test_unsigned_overflow_index_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        """uint64 max wraps to int64 -1 and would silently match the sentinel."""
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        covariances = np.array([[[0.01, 0.0], [0.0, 0.01]]], dtype=float)
        indices = np.full((n_cam, 1), 0, dtype=np.uint64)
        indices[0, 0] = np.iinfo(np.uint64).max
        with pytest.raises(ValueError, match="signed int64 range"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                anchors,
                covariances,
                indices,
            )

    def test_string_anchors_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        bad_anchors = np.array([["0.0", "0.0"]], dtype=object)
        covariances = np.array([[[0.01, 0.0], [0.0, 0.01]]], dtype=float)
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        with pytest.raises(ValueError, match="event_source_anchors must be"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                bad_anchors,
                covariances,
                indices,
            )

    def test_complex_anchors_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        """Complex dtype passes np.number but discards imaginary part on float cast."""
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        complex_anchors = np.array([[0.5 + 1j, 0.5]], dtype=complex)
        covariances = np.array([[[0.01, 0.0], [0.0, 0.01]]], dtype=float)
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        with pytest.raises(ValueError, match="event_source_anchors must be"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                complex_anchors,
                covariances,
                indices,
            )

    def test_complex_covariances_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        complex_cov = np.array([[[0.01 + 1j, 0.0], [0.0, 0.01]]], dtype=complex)
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        with pytest.raises(ValueError, match="event_source_covariances must be"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                anchors,
                complex_cov,
                indices,
            )

    def test_bool_index_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        covariances = np.array([[[0.01, 0.0], [0.0, 0.01]]], dtype=float)
        indices = np.zeros((n_cam, 1), dtype=bool)
        with pytest.raises(ValueError, match="integer or float array"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                anchors,
                covariances,
                indices,
            )

    def test_object_index_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.array([[0.5, 0.5]], dtype=float)
        covariances = np.array([[[0.01, 0.0], [0.0, 0.01]]], dtype=float)
        indices = np.full((n_cam, 1), "0", dtype=object)
        with pytest.raises(ValueError, match="integer or float array"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                anchors,
                covariances,
                indices,
            )

    def test_index_out_of_range_rejected(
        self,
        event_channel_sim,
        event_channel_filter_defaults,
        run_filter_with_events,
    ):
        ekf_config = EKFConfig(**event_channel_filter_defaults)
        n_cam = event_channel_sim["t_cam_exp"].shape[0]
        anchors = np.zeros((1, 2))
        covariances = np.broadcast_to(np.eye(2) * 0.01, (1, 2, 2)).copy()
        indices = np.full((n_cam, 1), -1, dtype=np.int32)
        indices[0, 0] = 5
        with pytest.raises(ValueError, match="out of"):
            run_filter_with_events(
                extended_kalman_filter,
                ekf_config,
                event_channel_sim,
                anchors,
                covariances,
                indices,
            )
