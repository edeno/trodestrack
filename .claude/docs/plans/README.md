# Implementation Plans

This directory contains planning documents for work that is not yet
fully shipped on the `updates` branch.

## Recommended Execution Order

1. `2026-05-07-geom-homography-implementation-plan.md`
   - Adds the camera calibration primitive used by native Trodes and DLC
     loaders.
2. `2026-05-07-native-trodes-dlc-loaders-implementation-plan.md`
   - Depends on the scalar/homography pixel-to-world decision.
3. `2026-05-07-ttl-event-sensors-implementation-plan.md`
   - Independent of camera ingest, but should use the same config and session
     diagnostics conventions.
4. `2026-05-07-online-tracker-streaming-implementation-plan.md`
   - Should happen after real-data config paths are stable so the streaming API
     reuses the final session and filter semantics.
5. `2026-05-07-naming-conventions-cleanup-implementation-plan.md`
   - Best after the larger public surfaces settle, otherwise alias coverage
     churns as new APIs land.

## Status

| Plan | Status | Notes |
| --- | --- | --- |
| `2026-05-07-geom-homography-implementation-plan.md` | Active | Needed before perspective-correct real-data ingest. |
| `2026-05-07-native-trodes-dlc-loaders-implementation-plan.md` | Active | Adds direct Trodes and DLC position loaders. |
| `2026-05-07-ttl-event-sensors-implementation-plan.md` | Active | Unified TTL event-source design covering beam break, zone trigger, and RFID. |
| `2026-05-07-online-tracker-streaming-implementation-plan.md` | Active | Python streaming API; CLI remains batch. |
| `2026-05-07-naming-conventions-cleanup-implementation-plan.md` | Active, lower priority | Public API usability cleanup after functional surfaces settle. |

## Superseded Plans

The following plan files were considered and removed from this
directory; they're documented here for historical reference:

- **`2026-05-07-beam-break-implementation-plan.md`** — superseded by
  `2026-05-07-ttl-event-sensors-implementation-plan.md`, which covers
  beam breaks alongside TTL zone triggers and RFID readers under a
  single `EventLocationModel`.
- **`2026-04-30-tilt-orientation-and-3d-camera-implementation-plan.md`**
  — milestones 1–6 (quaternion utilities, IMU calibration diagnostics,
  orientation estimator, 2D-camera + 6-DOF IMU mode, 3D camera
  measurement model, experimental 3D EKF + RTS smoother) shipped on
  the `updates` branch. Real 3D dataset validation remains an
  open follow-up but is tracked in PR-level test coverage rather
  than as a planning doc.
