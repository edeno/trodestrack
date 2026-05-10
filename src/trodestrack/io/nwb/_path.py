"""Path-based NWB session loader (lazy-imports ``pynwb``).

Used by ``inputs.format=nwb`` direct-NWB users. Spyglass / in-memory
callers should use the container-layer entries directly instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trodestrack.config.schemas import IMUConfig, NWBConfig
from trodestrack.io.loaders._shared import PositionPixels
from trodestrack.io.nwb._analog import from_analog_container
from trodestrack.io.nwb._dio import from_behavioral_events
from trodestrack.io.nwb._pose import from_pose_estimation_container
from trodestrack.io.nwb._position import from_position_container
from trodestrack.io.nwb._shared import (
    NWBSessionExtras,
    detect_led_container,
    neurodata_type,
)


def load_nwb_session(
    cfg: NWBConfig,
    *,
    imu_cfg: IMUConfig | None = None,
) -> tuple[PositionPixels, NWBSessionExtras]:
    """Open an NWB file at ``cfg.nwb_file``, pick LED container by
    neurodata type, and delegate to a container-layer entry.

    When ``imu_cfg`` is provided and the file has the standard
    ``processing/analog/analog/analog`` TimeSeries (per
    ``trodes_to_nwb.convert_analog._NWB_ANALOG_DATA_PATH``), the
    extras include the SI-converted ``(t_imu, U_full)`` pair via
    ``from_analog_container``. Without ``imu_cfg`` we don't know
    which channels to select, so the analog container is left
    unread.

    Lazy-imports ``pynwb`` so ``import trodestrack.io.nwb`` does not
    force the ``[nwb]`` extra to be installed.
    """

    try:
        import pynwb  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "inputs.format='nwb' requires the [nwb] extra. "
            "Install with: uv pip install 'trodestrack[nwb]'."
        ) from e

    with pynwb.NWBHDF5IO(str(cfg.nwb_file), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        led_container = detect_led_container(nwbfile, cfg.led_source.container)
        ndt = neurodata_type(led_container)
        if ndt == "Position":
            pixels = from_position_container(led_container, cfg.led_source)
        elif ndt == "PoseEstimation":
            pixels = from_pose_estimation_container(led_container, cfg.led_source)
        else:
            raise ValueError(
                f"Detected LED container has unsupported neurodata_type "
                f"{ndt!r}. Expected 'Position' or 'PoseEstimation'."
            )
        imu_pair: tuple[np.ndarray, np.ndarray] | None = None
        diagnostics: dict[str, Any] = {}
        if imu_cfg is not None:
            analog_ts = _find_analog_timeseries(nwbfile)
            if analog_ts is not None:
                t_imu, U_full = from_analog_container(analog_ts, imu_cfg)
                imu_pair = (t_imu, U_full)
                diagnostics["imu_source"] = "nwb_analog"
                diagnostics["imu_samples"] = int(t_imu.size)
                diagnostics["imu_channel_count"] = int(U_full.shape[1])
            else:
                diagnostics["imu_source"] = None
        dio_events: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if cfg.dio_to_ttl is not None:
            events_container = _find_behavioral_events(nwbfile)
            if events_container is None:
                raise ValueError(
                    f"inputs.nwb.dio_to_ttl is configured but "
                    f"{cfg.nwb_file} has no "
                    "processing['behavior']['behavioral_events'] container."
                )
            dio_events = from_behavioral_events(events_container, cfg.dio_to_ttl)
            diagnostics["dio_event_count"] = int(dio_events[0].size)
            diagnostics["dio_source"] = "nwb_behavioral_events"
        extras = NWBSessionExtras(
            imu=imu_pair, dio_events=dio_events, diagnostics=diagnostics
        )
        return pixels, extras
    # NWBHDF5IO closed here; pixels/extras hold numpy arrays only.


def _find_behavioral_events(nwbfile: Any) -> Any | None:
    """Locate ``processing/behavior/behavioral_events`` if present.

    The Trodes writer creates a ``BehavioralEvents`` container named
    ``"behavioral_events"`` under ``processing["behavior"]`` (per
    ``trodes_to_nwb/convert_dios.py``). Returns the container or
    ``None`` when absent.
    """

    processing = getattr(nwbfile, "processing", None) or {}
    if "behavior" not in processing:
        return None
    behavior = processing["behavior"]
    interfaces = getattr(behavior, "data_interfaces", {})
    if "behavioral_events" not in interfaces:
        return None
    return interfaces["behavioral_events"]


def _find_analog_timeseries(nwbfile: Any) -> Any | None:
    """Locate ``processing/analog/analog/analog`` if present.

    The Trodes writer creates a ``processing["analog"]`` module with a
    ``BehavioralEvents`` container named ``"analog"`` whose
    ``time_series`` dict has a single entry also named ``"analog"``
    (per ``trodes_to_nwb.convert_analog.add_analog_data``). The full
    HDF5 path is ``processing/analog/analog/analog/data``.

    Returns the TimeSeries or ``None`` when the file has no analog
    group.
    """

    processing = getattr(nwbfile, "processing", None) or {}
    if "analog" not in processing:
        return None
    module = processing["analog"]
    interfaces = getattr(module, "data_interfaces", {})
    if "analog" not in interfaces:
        return None
    container = interfaces["analog"]
    time_series = getattr(container, "time_series", None)
    if time_series is None or "analog" not in time_series:
        return None
    return time_series["analog"]
