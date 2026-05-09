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

- Cross-format parity test: same session loaded via the existing
  `spikegadgets_trodes` parquet path AND the three native loaders
  (`trodes_native`, `dlc_keypoints`, `nwb`) yields filtered means
  within 1e-3 m across every pair. The parquet baseline confirms
  the new loaders remain parity-compatible with the established
  workflow.
- Three example YAMLs in `examples/` —
  `session_trodes_native.yaml`, `session_dlc_keypoints.yaml`,
  `session_nwb.yaml`. These are **template configs** following the
  same convention as the existing
  `examples/session_spikegadgets_trodes.yaml`: paths point at
  user-supplied placeholders so the examples don't ship with bundled
  data fixtures. The "examples valid" test parses each via
  `load_session_config` to catch schema drift / YAML breakage at CI
  time without requiring a load against checked-in data.
- New "Loading native formats" section in
  `docs/getting-started/python-api.md`.
- Cross-references in `docs/TUNING.md`.

## Validation slice

| Test | Asserts |
| --- | --- |
| Cross-format parity | parquet, trodes_native, dlc_keypoints, and nwb loader paths produce filtered means within 1e-3 m across every pair for the same in-memory ground-truth. |
| Examples valid | each example YAML parses cleanly via `load_session_config` (schema-valid + paths resolve). Loading against placeholder paths is intentionally deferred — the parity test above already exercises end-to-end load + EKF for every format. |

## Fixtures

The cross-format-parity test synthesizes a single ground-truth pixel
trajectory in-process and writes it into all four format layouts
(parquet pair, Trodes binaries, DLC HDF5, NWB Position) at test
time. No checked-in binary fixtures — every byte the test reads is
generated from numpy arrays via `struct.pack` / `pandas.to_hdf` /
`pynwb`. This matches the no-checked-in-binaries policy the earlier
phases (2 / 3 / 4a / 4b / 4c) established.
