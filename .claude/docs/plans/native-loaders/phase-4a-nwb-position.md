# Phase 4a — NWB position

[← back to README](README.md) · [design](designs.md#nwb--position) · [shared contracts](shared-contracts.md)

**Inputs to read first:**

- `/Users/edeno/Documents/GitHub/trodes_to_nwb/src/trodes_to_nwb/convert_position.py` —
  NWB write side, lines 1067-1079 for the SpatialSeries naming
  convention.
- `/Users/edeno/Documents/GitHub/ndx-pose/spec/ndx-pose.extensions.yaml` —
  `PoseEstimation`, `PoseEstimationSeries`, `Skeleton`, `Skeletons`
  schema.
- `/Users/edeno/Documents/GitHub/ndx-pose/src/pynwb/ndx_pose/io/pose.py:14-25, 33-44, 53-58` —
  custom IO mapping (`definition`/`version` rename, v0.1.x
  inline-`nodes` back-compat).

**Contracts referenced:**

- [NWB container-layer API (public; Spyglass integration seam)](shared-contracts.md#nwb-container-layer-api-public-spyglass-integration-seam) —
  must be honored exactly. Annotations on container parameters must
  not trigger `pynwb` import at module load.
- [`PositionPixels`](shared-contracts.md#positionpixels--loader-private-intermediate)

**Design:** [nwb — position](designs.md#nwb--position).

## Tasks

NWB position only. No NWB IMU, no DIO bridge.

- `src/trodestrack/io/nwb/__init__.py` exposing the public container
  API: `from_position_container`, `from_pose_estimation_container`,
  `load_nwb_session`. Path layer (`load_nwb_session`) is a thin
  wrapper that opens the NWB file, picks containers by neurodata type,
  and delegates to the container-layer entries.
- Lazy `pynwb` import inside `load_nwb_session`. Container entries do
  **not** import `pynwb` themselves — they accept already-loaded
  containers, so a Spyglass `make()` (which has already imported
  `pynwb` to call `fetch_nwb`) can call them directly.
- Trodes-position branch, ndx-pose branch with v0.1/v0.2 Skeleton
  fallback, conversion=1.0 sentinel handling — all in the container
  entry points.
- Eager numpy materialization in both container entries; no lazy
  `h5py.Dataset` references in returned `PositionPixels`.
- IMU still loaded via parquet or synthetic-vision-only fallback (NWB
  IMU lands in [Phase 4b](phase-4b-nwb-imu.md)).

## Validation slice

| Test | Asserts |
| --- | --- |
| NWB `[nwb]` extra missing | `inputs.format=nwb` without `pynwb` raises `ImportError`. |
| `import trodestrack.io.nwb` does not import `pynwb` | with `pynwb` removed from `sys.modules`, importing the public module succeeds; only `load_nwb_session()` triggers the lazy import. Confirms the `TYPE_CHECKING`-guarded annotations don't leak. |
| Container API: from_position_container | takes a pre-loaded `Position` object; produces same `PositionPixels` as `load_nwb_session` does for the same file. |
| Container API: from_pose_estimation_container | takes a pre-loaded `PoseEstimation` object; produces same `PositionPixels` as `load_nwb_session` does for the same file. |
| Container API eager materialization | open NWB → call container entry → close `NWBHDF5IO` → returned arrays still readable (no `h5py.Dataset` references retained). |
| NWB auto-detect by neurodata type | non-default container names still resolved by `load_nwb_session`. |
| NWB ndx-pose v0.1.x compatibility | inline `nodes` resolves; v0.2.x `Skeleton` link resolves; both produce same downstream output. |
| NWB conversion=1.0 sentinel | `unit="pixels"` + `conversion=1.0` requires `meters_per_pixel_override`; `unit="meters"` accepts as calibrated. |
| NWB attribute-name policy | loader reads `definition` and `version` regardless of whether `ndx-pose` is installed. |
| NWB conversion override | `meters_per_pixel_override` wins over file-stored conversion. |

## Fixtures

Synthesize NWB files in `conftest.py` via `pynwb` itself:

- One with a Trodes-style `Position` (two `SpatialSeries`,
  `unit="pixels"`, `conversion=1.0`).
- One with a `unit="meters"` variant.
- One with an `ndx-pose` v0.2.x `PoseEstimation` + `Skeletons` chain.
- One with the v0.1.x inline-`nodes` shape. The v0.1.x fixture is
  built by writing the on-disk attributes directly via h5py (skipping
  the `ndx-pose` typed-class path) so the test does not require
  pinning an old `ndx-pose` version.
