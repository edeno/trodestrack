# Phase 0 — Behavior-preserving extraction

[← back to README](README.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

**Inputs to read first:**

- [src/trodestrack/io/session.py](../../../../src/trodestrack/io/session.py) —
  the `_load_leds` body at line 609 and the IMU pipeline at lines
  502, 538, 577.
- [src/trodestrack/config/schemas.py](../../../../src/trodestrack/config/schemas.py) —
  the `CameraConfig` / `IMUConfig` blocks.

**Contracts referenced:**

- [`PositionPixels`](shared-contracts.md#positionpixels--loader-private-intermediate)
- [`pixels_to_meters`](shared-contracts.md#pixels_to_meters--calibration-helper-at-top-level)

## Tasks

Pure refactor; merges before any new format work. Parity-gated against
the existing test suite (output bitwise identical for
`spikegadgets_trodes` and `prepared_arrays`).

- New `src/trodestrack/io/pixel_to_meters.py` exposing
  `pixels_to_meters(...)` with the override → file-side →
  camera-config precedence. Migrate `_load_leds`
  (`session.py:609`) to call it.
- Extract `load_imu_parquet(imu_file, config)` from the inline
  `_remove_sample_hold` / `_convert_imu_to_si` /
  `_project_imu_for_filter` chain (`session.py:502, 538, 577`).
  Migrate `_load_spikegadgets_trodes` to call it.
- `src/trodestrack/io/loaders/__init__.py`, `_shared.py` (just the
  `PositionPixels` dataclass).

## Validation slice

| Test | Asserts |
| --- | --- |
| Existing-tests parity | `spikegadgets_trodes` and `prepared_arrays` filtered means bitwise identical to pre-refactor (parity gate). |
| `pixels_to_meters` precedence | override > file-side > camera-config; all three branches covered by unit tests. |
| `load_imu_parquet` extraction | replaces inline calls without changing values. |

## Fixtures

Existing `tests/` parquet fixtures only — no new fixtures needed.
