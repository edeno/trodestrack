# Phase 4c — NWB DIO → TTL bridge

[← back to README](README.md) · [design](designs.md#nwb--extras-imu-dio) · [shared contracts](shared-contracts.md)

**Inputs to read first:**

- [src/trodestrack/config/schemas.py](../../../../src/trodestrack/config/schemas.py) —
  `TTLEventsConfig` at line 424 and `SessionConfig.ttl_events` at line
  470 (note that `ttl_events` is currently `| None`).
- [src/trodestrack/io/session.py](../../../../src/trodestrack/io/session.py) —
  `_attach_ttl_events` near line 538, which becomes the dual-source
  dispatch.
- `/Users/edeno/Documents/GitHub/trodes_to_nwb/src/trodes_to_nwb/spike_gadgets_raw_io.py:953` —
  DIO `int8` 0/1 dtype.

**Contracts referenced:**

- [NWB DIO → TTL events bridge (schema contract)](shared-contracts.md#nwb-dio--ttl-events-bridge-schema-contract) —
  the ttl-events geometry block
  (`beams`/`zone_triggers`/`rfid_readers`) is required for any DIO
  session; schema validators below must enforce that.
- [NWB container-layer API](shared-contracts.md#nwb-container-layer-api-public-spyglass-integration-seam)

**Design:** [nwb — extras (IMU, DIO)](designs.md#nwb--extras-imu-dio).

## Tasks

Generalizes `_attach_ttl_events` to accept either parquet or NWB DIO
table, **and** lands the schema relaxation that lets a user configure
an NWB+DIO session without a parquet `events_file` at all. These ship
together because the schema and the loader must move in lockstep —
relaxing the schema before the loader exists would let users write a
configuration the system can't run.

### Schema-side changes (deferred from [Phase 1](phase-1-scaffolding.md))

- `TTLEventsConfig.events_file: Path | None = None` (was required).
- `SessionConfig`-level conditional-required validator on `events_file`
  (required unless `inputs.format == "nwb"` and
  `inputs.nwb.dio_to_ttl` is set).
- `SessionConfig`-level validator: if `inputs.nwb.dio_to_ttl` is set,
  `ttl_events` itself must be present (not `None`) — the geometry
  block is required because the EKF/UKF event channel needs source
  positions, covariances, and `id`s. Today `SessionConfig.ttl_events`
  is optional ([schemas.py:470](../../../../src/trodestrack/config/schemas.py#L470));
  this validator closes the loophole.
- `SessionConfig`-level cross-validator: every value in
  `inputs.nwb.dio_to_ttl.name_to_source_id` must be an `id` in
  `ttl_events.{beams,zone_triggers,rfid_readers}`.

### Loader-side changes

- Public container entry `from_behavioral_events(events, dio_cfg)` in
  `src/trodestrack/io/nwb/__init__.py`. Trodestrack-canonical input
  shapes: a `BehavioralEvents` container (path-loader source) **or**
  a `dict[str, TimeSeries]` keyed by event name. Spyglass's
  `(DIOEvents & key).fetch_nwb()` returns a `list[dict]` with
  `dio_event_name` / `dio` columns; Spyglass's `make()` assembles the
  dict shape (`{row["dio_event_name"]: row["dio"] for row in ...}`)
  before calling. Eager materialization; performs edge detection on
  `int8` 0/1 streams and drops the first sample (initial level).
- `load_nwb_session` populates `NWBSessionExtras.dio_events` via
  `from_behavioral_events` when `cfg.dio_to_ttl` is set.
- `_attach_ttl_events` accepts `events_file=None` when the table is
  pre-built; loader-time check confirms each `name_to_source_id` key
  resolves to a `TimeSeries` under
  `processing["behavior"]["behavioral_events"]` (path loader) or in
  the dict keys (container API).

## Validation slice

| Test | Asserts |
| --- | --- |
| `events_file` optional iff NWB DIO bridge configured | exactly the four states (NWB+DIO+file, NWB+DIO+no-file, NWB-only+file, NWB-only+no-file) pass/fail correctly. |
| `ttl_events` required when DIO bridge set | `inputs.nwb.dio_to_ttl` set with `ttl_events: null` raises a schema error naming the geometry block requirement. |
| `name_to_source_id` schema cross-check | values referencing unknown `ttl_events` ids raise at schema time. |
| Container API: from_behavioral_events accepts dict shape | `dict[str, TimeSeries]` (the trodestrack-canonical input that Spyglass's `make()` assembles from `fetch_nwb()` rows) and `BehavioralEvents` container produce identical event tables for the same data. |
| from_behavioral_events eager materialization | close IO after the call → returned event table still readable. |
| NWB DIO bridge populates events | `behavioral_events` populates `session.event_*` arrays per `name_to_source_id`. |
| NWB no DIO config | NWB without `dio_to_ttl` does not populate event arrays. |
| NWB DIO loader-time validation | `name_to_source_id` referring to a nonexistent `TimeSeries` raises with the available names listed. |
| NWB DIO + parquet conflict | both configured → parquet wins, NWB DIO ignored, both recorded in diagnostics. |

## Fixtures

Extend a [Phase 4a](phase-4a-nwb-position.md) NWB fixture to add
`processing["behavior"]["behavioral_events"]` with two int8 0/1
TimeSeries (e.g. `beam_1`, `zone_a`); pair with a `SessionConfig`
fixture providing matching `ttl_events.beams` / `zone_triggers`
geometry plus `inputs.nwb.dio_to_ttl.name_to_source_id`. A second NWB
variant omits `behavioral_events` for the loader-time validation test.
