"""Tests for TTL event sensor ingest infrastructure (Milestone 1).

Covers:
- Per-spec ``to_event_source()`` math (BeamSpec / ZoneTriggerSpec / RFIDReaderSpec).
- ``TTLEventsConfig`` schema validation (unique IDs, edges).
- ``load_ttl_events`` parquet reader.
- ``per_frame_event_indices`` compaction + padding + active-edge gating.
- Path resolution of ``ttl_events.events_file`` relative to the session YAML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from trodestrack.config import load_session_config
from trodestrack.config.schemas import (
    BeamSpec,
    RFIDReaderSpec,
    TTLEventsConfig,
    ZoneTriggerSpec,
)
from trodestrack.io.ttl_events import load_ttl_events, per_frame_event_indices

# -----------------------------------------------------------------------------
# Spec.to_event_source() math
# -----------------------------------------------------------------------------


class TestZoneTriggerSpec:
    def test_anchor_and_isotropic_R(self):
        spec = ZoneTriggerSpec(id=7, center=(0.5, -0.25), sigma_m=0.02)
        src = spec.to_event_source()
        assert src.source_id == 7
        np.testing.assert_allclose(src.anchor, np.array([0.5, -0.25]))
        np.testing.assert_allclose(src.R, (0.02**2) * np.eye(2))
        assert src.source_type == "zone"


class TestRFIDReaderSpec:
    def test_anchor_and_isotropic_R_from_radius(self):
        spec = RFIDReaderSpec(id=2, center=(1.0, 2.0), effective_radius_m=0.06)
        src = spec.to_event_source()
        np.testing.assert_allclose(src.anchor, np.array([1.0, 2.0]))
        # σ = r/√2  → σ² = r²/2
        np.testing.assert_allclose(src.R, (0.06**2 / 2.0) * np.eye(2))
        assert src.source_type == "rfid"


class TestBeamSpec:
    def test_anchor_is_midpoint(self):
        spec = BeamSpec(
            id=1, emitter=(0.0, 0.0), receiver=(0.4, 0.0), sigma_perp_m=0.005
        )
        src = spec.to_event_source()
        np.testing.assert_allclose(src.anchor, np.array([0.2, 0.0]))

    def test_long_beam_eigenstructure(self):
        # Beam along x-axis: perpendicular = y, along = x.
        L = 1.0
        sigma_perp = 0.005
        spec = BeamSpec(
            id=1,
            emitter=(-L / 2, 0.0),
            receiver=(L / 2, 0.0),
            sigma_perp_m=sigma_perp,
        )
        src = spec.to_event_source()

        # Eigendecomposition: principal axes = beam tangent (along) and beam normal (perp).
        eigvals, eigvecs = np.linalg.eigh(src.R)
        # Smallest eigenvalue corresponds to perpendicular axis.
        np.testing.assert_allclose(eigvals[0], sigma_perp**2, atol=1e-12)
        # Largest eigenvalue corresponds to along-beam axis: (L/√12)².
        np.testing.assert_allclose(eigvals[1], (L / np.sqrt(12.0)) ** 2, atol=1e-12)

        # Perpendicular eigenvector should align with y axis (±[0, 1]).
        perp_axis = eigvecs[:, 0]
        assert abs(perp_axis[1]) > 0.99
        assert abs(perp_axis[0]) < 1e-9

    def test_short_beam_floor(self):
        # Beam shorter than σ_perp: along-beam variance floored at σ_perp².
        sigma_perp = 0.05
        spec = BeamSpec(
            id=1,
            emitter=(0.0, 0.0),
            receiver=(1e-6, 0.0),  # essentially zero length
            sigma_perp_m=sigma_perp,
        )
        src = spec.to_event_source()
        eigvals = np.linalg.eigvalsh(src.R)
        # Both eigenvalues equal σ_perp² (isotropic).
        np.testing.assert_allclose(eigvals, [sigma_perp**2, sigma_perp**2], atol=1e-9)
        # R well-conditioned.
        cond = eigvals.max() / eigvals.min()
        assert cond < 1.01

    def test_rotated_beam(self):
        # Beam along 45-degree direction.
        L = 0.4
        sigma_perp = 0.005
        spec = BeamSpec(
            id=1,
            emitter=(0.0, 0.0),
            receiver=(L / np.sqrt(2.0), L / np.sqrt(2.0)),
            sigma_perp_m=sigma_perp,
        )
        src = spec.to_event_source()
        # Anchor at midpoint.
        np.testing.assert_allclose(
            src.anchor,
            np.array([L / (2 * np.sqrt(2.0)), L / (2 * np.sqrt(2.0))]),
            atol=1e-12,
        )

        # Perpendicular direction is (-1, 1)/√2.
        _eigvals, eigvecs = np.linalg.eigh(src.R)
        perp_axis = eigvecs[:, 0]
        expected_perp = np.array([-1.0, 1.0]) / np.sqrt(2.0)
        # Either +expected or -expected is acceptable.
        assert abs(abs(np.dot(perp_axis, expected_perp)) - 1.0) < 1e-9

    def test_default_active_edge_is_fall(self):
        spec = BeamSpec(id=1, emitter=(0.0, 0.0), receiver=(0.1, 0.0))
        assert spec.active_edge == "fall"


# -----------------------------------------------------------------------------
# TTLEventsConfig schema validation
# -----------------------------------------------------------------------------


class TestTTLEventsConfigValidation:
    def test_unique_ids_across_lists(self, tmp_path):
        events_file = tmp_path / "events.parquet"
        events_file.write_bytes(b"placeholder")
        with pytest.raises(ValidationError, match="unique"):
            TTLEventsConfig(
                events_file=events_file,
                beams=[BeamSpec(id=1, emitter=(0.0, 0.0), receiver=(0.1, 0.0))],
                zone_triggers=[ZoneTriggerSpec(id=1, center=(0.0, 0.0), sigma_m=0.02)],
            )

    def test_pad_limit_must_be_positive(self, tmp_path):
        events_file = tmp_path / "events.parquet"
        events_file.write_bytes(b"placeholder")
        with pytest.raises(ValidationError):
            TTLEventsConfig(events_file=events_file, max_events_per_frame=0)


# -----------------------------------------------------------------------------
# Path resolution via load_session_config
# -----------------------------------------------------------------------------


def _write_minimal_imu_camera(tmp_path):
    """Write minimal arrays for a prepared_arrays session."""
    t_imu = np.linspace(0.0, 1.0, 100)
    np.savetxt(tmp_path / "t_imu.txt", t_imu)
    np.savetxt(
        tmp_path / "u_imu.txt",
        np.zeros((len(t_imu), 3)),
    )
    t_cam = np.linspace(0.0, 1.0, 30)
    np.savetxt(tmp_path / "t_cam.txt", t_cam)
    np.savetxt(
        tmp_path / "led1.txt",
        np.column_stack([np.linspace(0.0, 1.0, 30), np.zeros(30)]),
    )
    return t_imu, t_cam


def test_load_session_attaches_event_arrays(tmp_path):
    _write_minimal_imu_camera(tmp_path)
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(
        {"time": [0.10, 0.50], "source_id": [1, 1], "edge": ["fall", "fall"]}
    ).to_parquet(events_path)

    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "inputs": {
                    "format": "prepared_arrays",
                    "imu_timestamps": "t_imu.txt",
                    "imu_measurements": "u_imu.txt",
                    "camera_timestamps": "t_cam.txt",
                    "led1_positions": "led1.txt",
                },
                "ttl_events": {
                    "events_file": "events.parquet",
                    "beams": [
                        {
                            "id": 1,
                            "emitter": [0.0, 0.0],
                            "receiver": [0.1, 0.0],
                            "sigma_perp_m": 0.005,
                        }
                    ],
                    "max_events_per_frame": 4,
                },
            }
        )
    )
    from trodestrack.io.session import load_session

    config = load_session_config(config_path)
    session = load_session(config)
    assert len(session.event_sources) == 1
    assert session.event_source_anchors.shape == (1, 2)
    assert session.event_source_covariances.shape == (1, 2, 2)
    assert session.event_indices_per_frame.shape[1] == 4
    assert (session.event_indices_per_frame >= 0).sum() >= 1
    diag = session.diagnostics["ttl_events"]
    assert diag["n_sources"] == 1
    assert diag["max_events_per_frame"] == 4


def test_ttl_events_path_resolved_relative_to_yaml(tmp_path):
    _write_minimal_imu_camera(tmp_path)
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(
        {"time": [0.1, 0.2], "source_id": [1, 1], "edge": ["fall", "fall"]}
    ).to_parquet(events_path)

    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "inputs": {
                    "format": "prepared_arrays",
                    "imu_timestamps": "t_imu.txt",
                    "imu_measurements": "u_imu.txt",
                    "camera_timestamps": "t_cam.txt",
                    "led1_positions": "led1.txt",
                },
                "ttl_events": {
                    "events_file": "events.parquet",
                    "beams": [
                        {
                            "id": 1,
                            "emitter": [0.0, 0.0],
                            "receiver": [0.1, 0.0],
                            "sigma_perp_m": 0.005,
                        }
                    ],
                },
            }
        )
    )
    config = load_session_config(config_path)
    assert config.ttl_events is not None
    assert config.ttl_events.events_file == events_path


# -----------------------------------------------------------------------------
# load_ttl_events parquet loader
# -----------------------------------------------------------------------------


def test_load_ttl_events_basic(tmp_path):
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "time": [0.10, 0.20, 0.30],
            "source_id": [1, 2, 1],
            "edge": ["fall", "rise", "fall"],
        }
    ).to_parquet(events_path)
    t, sid, edge = load_ttl_events(events_path)
    np.testing.assert_allclose(t, [0.10, 0.20, 0.30])
    np.testing.assert_array_equal(sid, [1, 2, 1])
    np.testing.assert_array_equal(edge, [0, 1, 0])  # 0=fall, 1=rise


def test_load_ttl_events_empty(tmp_path):
    events_path = tmp_path / "events.parquet"
    pd.DataFrame({"time": [], "source_id": [], "edge": []}).astype(
        {"time": float, "source_id": int, "edge": str}
    ).to_parquet(events_path)
    t, sid, edge = load_ttl_events(events_path)
    assert t.shape == (0,)
    assert sid.shape == (0,)
    assert edge.shape == (0,)


# -----------------------------------------------------------------------------
# per_frame_event_indices builder
# -----------------------------------------------------------------------------


class TestPerFrameEventIndices:
    def test_single_event_compact_index(self):
        t_evt = np.array([0.025])
        source_id = np.array([7])
        edge = np.array([0])  # fall
        t_cam = np.array([0.0, 0.05, 0.1])
        result = per_frame_event_indices(
            t_evt,
            source_id,
            edge,
            t_cam,
            source_active_edges={7: 0},
            source_id_to_index={7: 3},
            max_events_per_frame=2,
        )
        assert result.shape == (3, 2)
        # Event in (t_cam[0], t_cam[1]] → frame index 1 (0-based: bucket of frame 1).
        # Convention: event at time t goes into the frame whose interval ends at t.
        np.testing.assert_array_equal(result[1], [3, -1])
        # All other frames empty.
        np.testing.assert_array_equal(result[0], [-1, -1])
        np.testing.assert_array_equal(result[2], [-1, -1])

    def test_multiple_events_in_one_frame(self):
        # Two events in the same camera frame.
        t_evt = np.array([0.020, 0.040])
        source_id = np.array([1, 2])
        edge = np.array([0, 1])
        t_cam = np.array([0.0, 0.05])
        result = per_frame_event_indices(
            t_evt,
            source_id,
            edge,
            t_cam,
            source_active_edges={1: 0, 2: 1},
            source_id_to_index={1: 0, 2: 1},
            max_events_per_frame=4,
        )
        assert result.shape == (2, 4)
        # Both events fall into frame 1.
        assert sorted(int(x) for x in result[1] if x >= 0) == [0, 1]

    def test_inactive_edge_filtered(self):
        # source 1 active_edge="fall" (0); rise events should be ignored.
        t_evt = np.array([0.020, 0.040])
        source_id = np.array([1, 1])
        edge = np.array([1, 0])  # rise then fall
        t_cam = np.array([0.0, 0.05])
        result = per_frame_event_indices(
            t_evt,
            source_id,
            edge,
            t_cam,
            source_active_edges={1: 0},
            source_id_to_index={1: 0},
            max_events_per_frame=2,
        )
        # Only the fall event gets recorded.
        assert (result[1] == 0).sum() == 1

    def test_unknown_source_id_raises(self):
        t_evt = np.array([0.025])
        source_id = np.array([999])
        edge = np.array([0])
        t_cam = np.array([0.0, 0.05])
        with pytest.raises(ValueError, match="unknown source"):
            per_frame_event_indices(
                t_evt,
                source_id,
                edge,
                t_cam,
                source_active_edges={1: 0},
                source_id_to_index={1: 0},
                max_events_per_frame=2,
            )

    def test_pad_limit_exceeded(self):
        t_evt = np.array([0.01, 0.02, 0.03])
        source_id = np.array([1, 1, 1])
        edge = np.array([0, 0, 0])
        t_cam = np.array([0.0, 0.05])
        with pytest.raises(ValueError, match="max_events_per_frame"):
            per_frame_event_indices(
                t_evt,
                source_id,
                edge,
                t_cam,
                source_active_edges={1: 0},
                source_id_to_index={1: 0},
                max_events_per_frame=2,
            )

    def test_event_before_first_frame_dropped(self):
        # Events before t_cam[0] have no valid frame interval.
        t_evt = np.array([-0.01, 0.025])
        source_id = np.array([1, 1])
        edge = np.array([0, 0])
        t_cam = np.array([0.0, 0.05])
        result = per_frame_event_indices(
            t_evt,
            source_id,
            edge,
            t_cam,
            source_active_edges={1: 0},
            source_id_to_index={1: 0},
            max_events_per_frame=2,
        )
        # Only the second event ends up in frame 1.
        assert (result == 0).sum() == 1
