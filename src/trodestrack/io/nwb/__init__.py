"""Public NWB position / IMU / DIO loader.

Two-layer design (Spyglass integration seam):

- **Container layer**: ``from_position_container`` /
  ``from_pose_estimation_container`` / ``from_analog_container`` /
  ``from_behavioral_events`` accept already-loaded pynwb containers.
  They do **not** import ``pynwb`` at module load — attributes are
  read duck-typed so a Spyglass ``make()`` (which has already
  imported ``pynwb`` to call ``fetch_nwb``) can call them directly
  without trodestrack carrying a hard ``pynwb`` dependency.
- **Path layer**: ``load_nwb_session`` lazy-imports ``pynwb``, opens
  the file with ``NWBHDF5IO``, walks containers, and delegates to the
  container-layer entries. Used by ``inputs.format=nwb`` direct-NWB
  users.

All container entries materialize numpy arrays before returning so
the caller may close its IO handle (or let ``fetch_nwb``'s underlying
file close) without breaking downstream code.

Implementations live in submodules organized by NWB container kind:

- ``_position.py`` — Trodes-style ``Position`` (paired SpatialSeries).
- ``_pose.py`` — ndx-pose ``PoseEstimation``.
- ``_analog.py`` — Trodes-style analog ``TimeSeries`` (IMU).
- ``_dio.py`` — ``BehavioralEvents`` DIO bridge.
- ``_path.py`` — file-opening + container dispatch.
- ``_shared.py`` — cross-cutting helpers and ``NWBSessionExtras``.
"""

from __future__ import annotations

from trodestrack.io.nwb._analog import from_analog_container
from trodestrack.io.nwb._dio import from_behavioral_events
from trodestrack.io.nwb._path import load_nwb_session
from trodestrack.io.nwb._pose import from_pose_estimation_container
from trodestrack.io.nwb._position import from_position_container
from trodestrack.io.nwb._shared import NWBSessionExtras

__all__ = [
    "NWBSessionExtras",
    "from_analog_container",
    "from_behavioral_events",
    "from_pose_estimation_container",
    "from_position_container",
    "load_nwb_session",
]
