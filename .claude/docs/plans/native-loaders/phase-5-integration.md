# Phase 5 — Cross-format integration + docs

[← back to README](README.md)

**Inputs to read first:**

- All per-format fixtures from [Phase 2](phase-2-trodes-native.md),
  [Phase 3](phase-3-dlc-keypoints.md), [Phase 4a](phase-4a-nwb-position.md),
  [Phase 4b](phase-4b-nwb-imu.md), [Phase 4c](phase-4c-nwb-dio.md).
- [docs/getting-started/python-api.md](../../../../docs/getting-started/python-api.md).
- [docs/TUNING.md](../../../../docs/TUNING.md).

**Contracts referenced:** none new; this phase exercises everything
prior phases established.

## Tasks

- Cross-format parity test: same session loaded via parquet, native,
  and NWB yields filtered means within 1e-3 m of each other.
- `examples/session_trodes_native.yaml`,
  `examples/session_dlc_keypoints.yaml`, `examples/session_nwb.yaml`.
- New "Loading native formats" section in
  `docs/getting-started/python-api.md`.
- Cross-references in `docs/TUNING.md`.

## Validation slice

| Test | Asserts |
| --- | --- |
| Cross-format parity | three loader paths within tolerance for the same underlying session. |
| Examples runnable | each example YAML parses and loads its (committed-fixture) session. |

## Fixtures

A single committed minimal session converted to all three formats —
Trodes binaries, DLC HDF5 (synthesized from the same ground-truth
pixel trajectory), and an NWB file with both Position and ndx-pose
containers. The shared ground-truth lives in `conftest.py`; each
format adapter writes it out in the matching layout. This is the
authoritative cross-format-parity fixture.
