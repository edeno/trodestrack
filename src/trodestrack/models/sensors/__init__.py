"""Sensor measurement models for Kalman filter updates.

Two complementary surfaces live here:

- ``MeasurementModel`` protocol implementations for fixed-shape per-frame
  measurements: camera position, heading pseudo-measurements, ZUPT.
- ``EventLocationModel`` + ``update_event_location`` for TTL event sources
  (beam break, zone trigger, RFID reader). The event channel has variable
  per-frame source counts and a stacked Jacobian, so it intentionally does
  not implement ``MeasurementModel``; the EKF and UKF call
  ``update_event_location`` directly for that channel.
"""

from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.sensors.camera_position_3d import Camera3DPositionModel
from trodestrack.models.sensors.event_location import (
    EventLocationModel,
    update_event_location,
)
from trodestrack.models.sensors.heading_pseudo import HeadingPseudoModel
from trodestrack.models.sensors.protocols import MeasurementModel
from trodestrack.models.sensors.zupt import ZUPTModel

__all__ = [
    "Camera3DPositionModel",
    "CameraPositionModel",
    "EventLocationModel",
    "HeadingPseudoModel",
    "MeasurementModel",
    "ZUPTModel",
    "update_event_location",
]
