# Overview — Scope, dependencies, integration, risks

[← back to README](README.md)

## Current trodestrack integration points

Anchored in the live codebase so the implementer knows what to extend
vs what to leave alone:

- `src/trodestrack/io/session.py:75` — `load_session(...)` dispatches
  on `config.inputs.format`. Adds three new `elif` branches.
- `src/trodestrack/io/session.py:243` — `_load_prepared_arrays` (txt
  files; test/debug only). Untouched.
- `src/trodestrack/io/session.py:288` — `_load_spikegadgets_trodes`
  (parquet workflow). **Lightly refactored in Phase 0** to call the
  shared helpers extracted there; behavior must be unchanged
  (parity-gated).
- `src/trodestrack/io/session.py:502, 538, 577` —
  `_remove_sample_hold`, `_convert_imu_to_si`,
  `_project_imu_for_filter`. Today inline-only; Phase 0 extracts them
  into a public `load_imu_parquet(...)` helper that the new loaders
  also use.
- `src/trodestrack/io/session.py:609` — `_load_leds`. Today applies
  `meters_per_pixel` inline. Phase 0 migrates it to call the new
  shared `pixels_to_meters` helper, so old and new loaders share one
  calibration code path.
- `src/trodestrack/config/schemas.py:14, 19` — `InputsConfig`,
  `format: Literal[...]`. Phase 1 adds three new literal values and
  three matching per-format Pydantic blocks.
- `src/trodestrack/config/schemas.py:166, 176` — `CameraConfig` with
  `meters_per_pixel: float = Field(default=0.0022, gt=0.0)`. The
  non-null default matters for the calibration precedence rule below.
- `src/trodestrack/config/schemas.py` — `TTLEventsConfig`. Phase 4c
  (not Phase 1) is the only phase that touches it: relaxes
  `events_file` to optional, adds the conditional-required validator,
  and adds the `SessionConfig`-level cross-validator. Earlier phases
  leave it untouched so a malformed session can't reach the loader
  before the DIO bridge exists to consume it.

## Scope and Dependency Policy

### Goals

- Three new `inputs.format` values:
  - `trodes_native`: `*.videoPositionTracking` +
    `*.videoTimeStamps.cameraHWSync` (PTP-synced).
  - `dlc_keypoints`: `<video_stem>DLC_*.h5` + `_meta.pickle`, with
    bodypart-name selection and likelihood gating.
  - `nwb`: NWB file with auto-detected position container (Trodes
    `Position` or ndx-pose `PoseEstimation`); optional NWB IMU read
    from `processing["analog"]`; optional DIO → TTL events bridge from
    `behavioral_events`.
- Every loader produces the same `PreparedSession` contract; the EKF /
  smoother / safety-check paths see no format-specific code.
- IMU loading is read from the user's actual data wherever possible
  (NWB `processing["analog"]`); raw-Trodes IMU and the parquet path
  are options, not requirements.

### Non-Goals

- Hardware acquisition / live ingest.
- Standalone conversion CLI (`trodestrack convert-trodes`); loaders
  are part of the session pipeline.
- DLC re-tracking, frame extraction, model training.
- Non-PTP Trodes timestamp variants
  (`*.videoTimeStamps.cameraHWFrameCount`, plain `*.videoTimeStamps`).
  Adding them needs sample-rate / clock-stitching logic out of scope
  for v1.
- Multi-animal projects. ndx-pose stores one animal per file. DLC
  multi-animal mode is rejected up front (clear single-animal-contract
  error).
- Importing `trodes_to_nwb`, `deeplabcut`, or `ndx-pose` as runtime
  deps. Vendor the small Trodes parser; read DLC HDF5 with plain
  pandas; read NWB extensions via `pynwb`'s dynamic-spec loading.
- Replacing `prepared_arrays` or `spikegadgets_trodes`. Both stay.
- Reading IMU from raw Trodes `.rec` files. Requires the `SpikeGadgets
  RawIO` Neo subclass and would pull `neo` into the dependency graph.
  Deferred.

### Dependency policy

- **Base install adds zero new runtime deps.** `numpy`, `pandas`,
  `scipy` (already in base) cover the vendored Trodes parser.
- **`[dlc]` extra** adds `tables` (PyTables — needed because DLC writes
  HDF5 with `format="table"`, which `pandas.read_hdf` drives via
  PyTables). Selecting `inputs.format=dlc_keypoints` without the extra
  raises a clear `ImportError` naming the install command.
- **`[nwb]` extra** adds `pynwb>=2.5`. Selecting
  `inputs.format=nwb` without the extra raises a clear `ImportError`
  naming the install command. Reading ndx-pose containers does **not**
  require `pip install ndx-pose` — `pynwb` reads the embedded
  namespace from the NWB file's `/specifications` group dynamically.

## Metrics

- **Adoption ergonomics**: a user with a Trodes session + DLC analysis
  goes from "manually convert two files to parquet" to "point YAML at
  the raw files" in < 5 minutes. Measured by walk-through length in
  `docs/getting-started/python-api.md` (target ≤ 30 lines).
- **Filter-output parity** across formats: ≤ 1e-3 m on filtered
  position means after camera-clock alignment.
- **Dependency footprint**: base install gains zero new runtime deps;
  `[dlc]` adds `tables`; `[nwb]` adds `pynwb>=2.5`.
- **Loader runtime**: each loader processes a 30-min session in ≤ 10 s
  on the existing benchmark hardware. NWB loader may be slower due to
  HDF5 chunking; ≤ 15 s acceptable.

## Risks and Mitigations

Design-relevant risks the implementer must respect (operational risks
like upstream drift / docs ambiguity removed — those belong in PR
review, not the agent prompt):

| Risk | Mitigation |
| --- | --- |
| DLC multi-animal silently slices individual 0 | Reject up front with a clear error pointing at the single-animal contract. |
| ndx-pose schema evolves (already at v0.2.0) | Use dynamic-spec loading (no typed-class dep), read by dataset path. Parity test against fixtures from each schema version supported. |
| NWB file's `conversion` is wrong (post-hoc re-calibration) | `meters_per_pixel_override` lets the YAML take precedence. Diagnostics record both file-stored and override values. |
| DIO bridge maps the wrong channel | Schema-time: `SessionConfig` validator checks `name_to_source_id ⊆ ttl_events.{...}.id`. Loader-time: NWB loader confirms each name resolves to a `TimeSeries` in `behavioral_events`. |
| NWB IMU channel-id mismatch | Loader compares `IMUConfig.axis_map` values to channel ids found in the file; missing channels raise with the available names listed. |
| Five-format `inputs.format` literal grows further | Refactor to `Annotated[Union[...], Discriminator("format")]`. Out of scope here; flag as the trigger condition. |

## Rollout Strategy

- Ship behind the new `inputs.format` literals; absent format choice
  leaves existing two formats unchanged.
- Each new format ships with a committed-fixture end-to-end test plus
  the cross-format parity test.
- Pre-merge: run the full suite plus the cross-format parity test on a
  single shared session converted three ways; confirm filtered means
  within tolerance.

## Open Questions

1. **Should v1 read IMU from raw Trodes `.rec` files?** No — deferred.
   Parsing `.rec` requires the `SpikeGadgets RawIO` Neo subclass, which
   would drag `neo` into the dependency graph.
2. **Should DLC `_filtered.h5` ever be the default?** No. `cfg.h5_file`
   may point at either; no special-casing.
3. **Should the loader support DLC's `.csv` output?** No. CSV is the
   `--save-as-csv` opt-in; HDF5 is universal. CSV-only users re-export.
4. **What about SLEAP's NWB output (also via ndx-pose)?** Already
   covered. `source_software` distinguishes for diagnostics.
5. **NeuroConv conversion path?** NeuroConv writes the same NWB
   containers; covered by the same loader.

## Estimated Effort

LOC sanity check for diff sizing (no time estimate — irrelevant for
agent invocation):

- ~300–350 LOC vendored Trodes PTP parser; ~50 LOC loader wrapping it.
- ~200 LOC DLC loader (saver compatibility, three timestamp sources,
  multi-animal rejection).
- ~300 LOC NWB loader (auto-detect, ndx-pose v0.1/v0.2, conversion=1.0
  fallback, NWB IMU read, DIO bridge).
- ~100 LOC shared scaffolding (`PositionPixels`, `pixels_to_meters`,
  `load_imu_parquet` extraction, lazy-import error helpers).
- **~900–950 LOC source.** ~500 LOC tests. ~150 lines docs.
- Two new optional dependencies: `pynwb` under `[nwb]`, `tables` under
  `[dlc]`. No base-install changes.
