# Phase 2 — `trodes_native` loader

[← back to README](README.md) · [design](designs.md#trodes_native) · [shared contracts](shared-contracts.md)

**Inputs to read first:**

- `/Users/edeno/Documents/GitHub/trodes_to_nwb/src/trodes_to_nwb/convert_position.py` —
  the functions to vendor (full list in the design link below):
  `parse_dtype`, `read_trodes_datafile`, `convert_datafile_to_pandas`,
  `get_framerate`, `find_acquisition_timing_pause`,
  `find_large_frame_jumps`, `detect_repeat_timestamps`,
  `detect_trodes_time_repeats_or_frame_jumps`,
  `_get_position_timestamps_ptp`, and the PTP branch of
  `get_position_timestamps`.
- `/Users/edeno/Documents/GitHub/trodes/python/trodes/binary_utils.py` —
  header format reference (`<Start settings>...<End settings>`,
  `Fields` declaration).
- [src/trodestrack/io/session.py](../../../../src/trodestrack/io/session.py) —
  `_load_spikegadgets_trodes` for the matching IMU pattern.

**Contracts referenced:**

- [IMU source resolution](shared-contracts.md#imu-source-resolution) —
  the hard-error vs vision-only fallback rule applies here as written;
  do not weaken it.
- [`PositionPixels`](shared-contracts.md#positionpixels--loader-private-intermediate)

**Design:** [trodes_native](designs.md#trodes_native).

## Tasks

- Vendor the ~300–350 LOC PTP parser into
  `src/trodestrack/io/loaders/_trodes_native.py` with attribution
  header (`# Adapted from trodes_to_nwb at <commit>`).
- `load_trodes_native_position` + `_load_trodes_native` in
  `session.py`.
- IMU-optional path: synthetic zero-IMU stream only when
  `filter.state_mode == vision_only`; **hard error** for any
  IMU-consuming `state_mode` per the IMU source resolution contract.

## Validation slice

| Test | Asserts |
| --- | --- |
| Trodes-native non-PTP rejected | `cameraHWFrameCount` / plain `videoTimeStamps` raises with v1-scope message. |
| Trodes-native PTP gate via column | a HWSync-suffixed file lacking `HWTimestamp` raises with a clearer message than upstream `KeyError`. |
| Trodes-native parity | filtered means identical to `spikegadgets_trodes` parquet of same raw files (tolerance 1e-6). |
| IMU-absent + IMU-consuming state_mode raises | running with the default `state_mode=2d_cam_3d_imu` and no `imu_file` raises with the three-option remediation message. |
| IMU-absent + `vision_only` succeeds | `state_mode=vision_only` with no `imu_file` produces a synthetic zero IMU stream and runs to completion. |

## Fixtures

Small Trodes binaries — one `*.videoPositionTracking` and one
`*.videoTimeStamps.cameraHWSync` (synthesized via the vendored
`parse_dtype` / pack-bytes helpers in a `conftest.py` rather than
checked-in binaries; reuses the same byte layout as upstream). Plus a
deliberately-malformed `*.videoTimeStamps.cameraHWSync` (missing
`HWTimestamp` column) for the PTP-gate test.
