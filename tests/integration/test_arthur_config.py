"""Slow integration checks for the bundled Arthur real-data config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trodestrack import main


@pytest.mark.slow
def test_bundled_arthur_example_runs_fused_orientation(tmp_path: Path) -> None:
    """The real-data example should run as fused orientation, not vision-only."""

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "examples" / "session_spikegadgets_trodes.yaml"
    required_data = (
        repo_root / "data" / "arthur20220324_imu_info.parquet",
        repo_root / "data" / "arthur20220324_position_info.parquet",
    )
    missing = [path for path in required_data if not path.exists()]
    if missing:
        pytest.skip(
            "Bundled Arthur parquet data is unavailable: "
            + ", ".join(str(path) for path in missing)
        )

    output_dir = tmp_path / "arthur_fused"
    with patch(
        "sys.argv",
        [
            "trodestrack",
            "online",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    ):
        main()

    filtered = np.loadtxt(output_dir / "filtered_means.txt")
    assert filtered.shape == (73450, 14)
    assert np.isfinite(filtered).all()

    with (output_dir / "session_diagnostics.json").open() as f:
        diagnostics = json.load(f)
    safety = diagnostics["safety_check"]
    assert safety["passed"] is True
    assert safety["max_fused_speed_mps"] < 3.0
    assert safety["max_vision_position_deviation_m"] < 0.5
    assert safety["p95_vision_position_deviation_m"] < 0.25
    assert safety["fused_range_m"][0] < safety["camera_range_m"][0] * 3.0 + 0.5
    assert safety["fused_range_m"][1] < safety["camera_range_m"][1] * 3.0 + 0.5

    metadata = (output_dir / "metadata.txt").read_text()
    assert "State mode: 2d_cam_6dof_imu_orientation" in metadata
    assert "Experimental accel translation: False" in metadata
