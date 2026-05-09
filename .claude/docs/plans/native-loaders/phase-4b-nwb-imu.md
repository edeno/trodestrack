# Phase 4b — NWB analog IMU

[← back to README](README.md) · [design](designs.md#nwb--extras-imu-dio) · [shared contracts](shared-contracts.md)

**Inputs to read first:**

- `/Users/edeno/Documents/GitHub/trodes_to_nwb/src/trodes_to_nwb/convert_analog.py` —
  lines 89-104 for the analog `TimeSeries` write path; 107-108 for
  channel-id metadata.
- [src/trodestrack/config/schemas.py](../../../../src/trodestrack/config/schemas.py) —
  `IMUConfig` for `axis_map`.

**Contracts referenced:**

- [IMU source resolution](shared-contracts.md#imu-source-resolution) —
  precedence ladder and hard-error policy.
- [NWB container-layer API](shared-contracts.md#nwb-container-layer-api-public-spyglass-integration-seam) —
  eager materialization.

**Design:** [nwb — extras (IMU, DIO)](designs.md#nwb--extras-imu-dio).

## Tasks

Adds `processing["analog"]` IMU read.

- Public container entry `from_analog_container(analog_ts, imu_cfg)`
  in `src/trodestrack/io/nwb/__init__.py`. Eager materialization;
  accepts a `pynwb.base.TimeSeries`.
- `load_nwb_session` populates `NWBSessionExtras.imu` via
  `from_analog_container` when the file has `processing["analog"]`.
- IMU source resolution precedence: `inputs.imu_file` > NWB analog >
  synthetic vision-only fallback.

## Validation slice

| Test | Asserts |
| --- | --- |
| Container API: from_analog_container | takes a pre-loaded `TimeSeries`; produces same `(t_imu, U_imu)` as `load_nwb_session` does for the same file. |
| from_analog_container eager materialization | close IO after the call → returned arrays still readable. |
| NWB IMU read from `processing["analog"]` | channel ids match `IMUConfig.axis_map`; SI conversion identical to parquet path. |
| NWB IMU absent + IMU-consuming state_mode raises | NWB without `processing["analog"]` and default `state_mode` raises with the three-option remediation message; same surface as Trodes-native/DLC. |
| NWB IMU absent + `vision_only` succeeds | falls through to synthetic zero IMU, runs to completion. |
| NWB IMU parquet override | `inputs.imu_file` configured wins over NWB-stored IMU; diagnostics record both. |

## Fixtures

Extend a [Phase 4a](phase-4a-nwb-position.md) NWB fixture to include
`processing["analog"]["analog"]["analog"]` (matching
`_NWB_ANALOG_DATA_PATH = "processing/analog/analog/analog/data"`) with
`Headstage_GyroX/Y/Z` channel ids in the matching metadata location.
A second variant omits the analog group (for the IMU-absent test).
