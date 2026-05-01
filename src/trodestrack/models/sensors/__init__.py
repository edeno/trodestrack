"""Sensor measurement models for Kalman filter updates.

This module provides a unified interface for different measurement types
(camera position, heading pseudo-measurements, ZUPT, future TTL/RFID events)
via the MeasurementModel protocol.
"""

from trodestrack.models.sensors.camera_position import CameraPositionModel
from trodestrack.models.sensors.camera_position_3d import Camera3DPositionModel
from trodestrack.models.sensors.heading_pseudo import HeadingPseudoModel
from trodestrack.models.sensors.protocols import MeasurementModel
from trodestrack.models.sensors.zupt import ZUPTModel

__all__ = [
    "Camera3DPositionModel",
    "CameraPositionModel",
    "HeadingPseudoModel",
    "MeasurementModel",
    "ZUPTModel",
]
