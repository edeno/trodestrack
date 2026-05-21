"""Construction-time validation tests for ``EventLocationSource``."""

from __future__ import annotations

import numpy as np
import pytest

from trodestrack.config.schemas import EventLocationSource


def test_event_location_source_accepts_valid_construction():
    """Valid anchor + symmetric PSD covariance + literal type constructs cleanly."""

    source = EventLocationSource(
        source_id=1,
        anchor=np.array([0.5, 1.0]),
        covariance=np.eye(2),
        source_type="beam",
    )

    assert source.anchor.shape == (2,)
    assert source.covariance.shape == (2, 2)
    assert source.source_type == "beam"


def test_event_location_source_rejects_wrong_anchor_shape():
    with pytest.raises(ValueError, match="anchor"):
        EventLocationSource(
            source_id=1,
            anchor=np.zeros(3),
            covariance=np.eye(2),
            source_type="beam",
        )


def test_event_location_source_rejects_non_finite_anchor():
    with pytest.raises(ValueError, match="anchor"):
        EventLocationSource(
            source_id=1,
            anchor=np.array([0.0, np.nan]),
            covariance=np.eye(2),
            source_type="beam",
        )


def test_event_location_source_rejects_wrong_cov_shape():
    with pytest.raises(ValueError, match="covariance"):
        EventLocationSource(
            source_id=1,
            anchor=np.zeros(2),
            covariance=np.eye(3),
            source_type="beam",
        )

    with pytest.raises(ValueError, match="covariance"):
        EventLocationSource(
            source_id=1,
            anchor=np.zeros(2),
            covariance=np.zeros(2),
            source_type="beam",
        )


def test_event_location_source_rejects_non_finite_covariance():
    cov = np.eye(2)
    cov[0, 0] = np.inf
    with pytest.raises(ValueError, match="covariance"):
        EventLocationSource(
            source_id=1,
            anchor=np.zeros(2),
            covariance=cov,
            source_type="beam",
        )


def test_event_location_source_rejects_non_symmetric_cov():
    with pytest.raises(ValueError, match="symmetric"):
        EventLocationSource(
            source_id=1,
            anchor=np.zeros(2),
            covariance=np.array([[1.0, 0.0], [0.5, 1.0]]),
            source_type="beam",
        )


def test_event_location_source_rejects_non_psd_cov():
    with pytest.raises(ValueError, match="PSD"):
        EventLocationSource(
            source_id=1,
            anchor=np.zeros(2),
            covariance=np.array([[1.0, 0.0], [0.0, -1.0]]),
            source_type="beam",
        )


def test_event_location_source_source_type_literal_rejects_unknown():
    with pytest.raises(ValueError, match="source_type"):
        EventLocationSource(
            source_id=1,
            anchor=np.zeros(2),
            covariance=np.eye(2),
            source_type="bogus",  # type: ignore[arg-type]
        )
