# Native Loaders Implementation Plan

**This plan has been split into per-phase files for easier
agent-by-phase invocation.**

→ See [native-loaders/](native-loaders/README.md) for the index and
all content.

The split organizes the plan as:

- `native-loaders/README.md` — index
- `native-loaders/overview.md` — scope, dependency policy,
  integration points, risks, metrics, rollout, open questions, effort
- `native-loaders/shared-contracts.md` — `PositionPixels`,
  `pixels_to_meters`, IMU source resolution, NWB DIO schema contract,
  NWB container-layer API
- `native-loaders/designs.md` — per-format designs (`trodes_native`,
  `dlc_keypoints`, `nwb`-position, `nwb`-extras)
- `native-loaders/phase-0-extraction.md` …
  `native-loaders/phase-5-integration.md` — one file per phase, each
  self-contained for agent invocation
- `native-loaders/appendix.md` — reference repos and on-disk format
  details
