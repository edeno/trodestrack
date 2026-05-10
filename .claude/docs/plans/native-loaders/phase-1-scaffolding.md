# Phase 1 — Schemas and loader scaffolding

[← back to README](README.md) · [overview](overview.md) · [designs](designs.md)

**Inputs to read first:**

- [src/trodestrack/config/schemas.py](../../../../src/trodestrack/config/schemas.py) —
  `InputsConfig` at line 14 and `TTLEventsConfig` at line 424.
- [src/trodestrack/io/session.py](../../../../src/trodestrack/io/session.py) —
  the format dispatch around line 75 + the per-format `_load_*`
  helpers.

**Contracts referenced:** none — schema-only PR.

**Designs referenced:** schema blocks under
[`trodes_native`](designs.md#trodes_native),
[`dlc_keypoints`](designs.md#dlc_keypoints),
[`nwb` — position](designs.md#nwb--position),
[`nwb` — extras](designs.md#nwb--extras-imu-dio).

## Tasks

Pydantic surface for all three new formats and **dependency-only loader
stubs**. No functional ingest yet.

- `format: Literal[..., "trodes_native", "dlc_keypoints", "nwb"]`.
- `TrodesNativeConfig`, `DLCKeypointsConfig`, `NWBConfig`,
  `NWBLEDSourceConfig`, `NWBDIOToTTLConfig`. Path resolution in
  `_resolve_paths`. Format-specific branches in
  `_validate_required_paths`.
- `[dlc]` and `[nwb]` extras in `pyproject.toml`.
- **Stub `_load_*` branches in `session.py`.** Each new format gets a
  dispatch branch that lazy-imports its required extra and raises
  `ImportError` when the extra is missing; otherwise raises
  `NotImplementedError("implemented in Phase N")`. This is the only
  way the "missing extra" test surface is realizable without the full
  loader — and it gives users a fast feedback loop ("did I install the
  right extra?") even before the loader logic lands.

**Deliberately not in Phase 1** (lands in [Phase 4c](phase-4c-nwb-dio.md)
when the DIO bridge is wired up):

- `TTLEventsConfig.events_file` relaxation to optional.
- The conditional-required validator on `events_file`.
- `SessionConfig`-level cross-validator for
  `name_to_source_id ⊆ ttl_events.{...}.id`.
- `SessionConfig`-level validator forcing `ttl_events` non-None when
  `inputs.nwb.dio_to_ttl` is set.

These changes can't ship in isolation — they would let a user
configure an NWB+DIO session that the loader can't actually run.

## Validation slice

| Test | Asserts |
| --- | --- |
| Schema: missing per-format block | `inputs.format=...` without the matching block raises `ValidationError`. |
| Path resolution | relative paths in each format block resolve relative to the YAML. |
| Existing TTL behavior unchanged | `TTLEventsConfig.events_file` still required (Phase 4c relaxes it). |
| Extra missing | `dlc_keypoints` without `[dlc]` and `nwb` without `[nwb]` raise `ImportError` naming the install command (via the stub branches above). |
| Stub raises `NotImplementedError` with extras present | informative message naming the implementing phase. |

## Fixtures

None — schema-only tests.
