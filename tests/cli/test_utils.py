"""Tests for CLI utility functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trodestrack.cli.utils import load_data_file


def test_load_data_file_success(tmp_path: Path) -> None:
    """Test successful data file loading."""
    # Create a test file
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    file_path = tmp_path / "test.txt"
    np.savetxt(file_path, data)

    # Load it
    loaded = load_data_file(file_path, "test data")

    # Should match
    np.testing.assert_array_equal(loaded, data)


def test_load_data_file_with_shape_validation(tmp_path: Path) -> None:
    """Test data file loading with shape validation."""
    # Create a test file
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    file_path = tmp_path / "test.txt"
    np.savetxt(file_path, data)

    # Load with correct shape - should succeed
    loaded = load_data_file(file_path, "test data", expected_shape=(2, 2))
    np.testing.assert_array_equal(loaded, data)


def test_load_data_file_missing_file(tmp_path: Path) -> None:
    """Test error handling for missing file."""
    missing_path = tmp_path / "nonexistent.txt"

    with pytest.raises(SystemExit) as exc_info:
        load_data_file(missing_path, "test data")

    assert exc_info.value.code == 1


def test_load_data_file_wrong_shape(tmp_path: Path) -> None:
    """Test error handling for wrong shape."""
    # Create a test file with shape (2, 2)
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    file_path = tmp_path / "test.txt"
    np.savetxt(file_path, data)

    # Try to load expecting shape (3, 2) - should fail
    with pytest.raises(SystemExit) as exc_info:
        load_data_file(file_path, "test data", expected_shape=(3, 2))

    assert exc_info.value.code == 1


def test_load_data_file_invalid_content(tmp_path: Path) -> None:
    """Test error handling for invalid file content."""
    # Create a file with invalid content
    file_path = tmp_path / "invalid.txt"
    file_path.write_text("not a number matrix")

    with pytest.raises(SystemExit) as exc_info:
        load_data_file(file_path, "test data")

    assert exc_info.value.code == 1
