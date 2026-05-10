# Native Loaders (Trodes / DLC / NWB) Implementation Plan

**Status:** Not started.

Adds three new `inputs.format` choices that ingest the raw files real
users have (Trodes `.videoPositionTracking`, DLC `.h5`, NWB) instead of
requiring an upstream conversion to the project's internal-convenience
parquet format.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each
   phase file is self-contained: it lists the upstream files to read,
   the contracts/designs it depends on (with relative links), the
   tasks, the validation matrix, and the fixture-provisioning notes.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-format design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).
5. **Need upstream-repo line refs / on-disk format details?** [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — Status, scope, dependency policy,
  integration points, risks, metrics, rollout, open questions, effort.
- [shared-contracts.md](shared-contracts.md) — `PositionPixels`,
  `pixels_to_meters`, IMU source resolution, NWB DIO schema contract,
  NWB container-layer API.
- [designs.md](designs.md) — Per-format designs for `trodes_native`,
  `dlc_keypoints`, `nwb`-position, `nwb`-extras.
- Phases (each ships as a separable PR):
  - [phase-0-extraction.md](phase-0-extraction.md) — Behavior-preserving extraction.
  - [phase-1-scaffolding.md](phase-1-scaffolding.md) — Schemas + dependency-only loader stubs.
  - [phase-2-trodes-native.md](phase-2-trodes-native.md) — `trodes_native` loader.
  - [phase-3-dlc-keypoints.md](phase-3-dlc-keypoints.md) — `dlc_keypoints` loader.
  - [phase-4a-nwb-position.md](phase-4a-nwb-position.md) — NWB position (path + container API).
  - [phase-4b-nwb-imu.md](phase-4b-nwb-imu.md) — NWB analog IMU.
  - [phase-4c-nwb-dio.md](phase-4c-nwb-dio.md) — NWB DIO → TTL bridge + schema relaxation.
  - [phase-5-integration.md](phase-5-integration.md) — Cross-format integration + docs.
- [appendix.md](appendix.md) — Reference repos (local clones) +
  on-disk binary / NWB encoding details.
