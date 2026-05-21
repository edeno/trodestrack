# Post-0.2.0 Review Fixes — Implementation Plan

**Status:** Not started.

Fix the findings from the comprehensive multi-agent review of trodestrack at `master` @ `d853bbc`. Work ships as six independently-deliverable PRs covering critical silent-failure fixes, test coverage gaps, scaffolding cleanup, CLI/UX polish, type hardening, and filter-core refactors. No backwards-compatibility shims — public APIs are changed in place per the project's "just change the code" default.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — Cross-phase context: integration points, scope, risks, rollout, metrics.
- [shared-contracts.md](shared-contracts.md) — Conventions that appear in ≥2 phases: `state_mode` Literal alias, deprecation policy (none), test layout.
- Phases (each ships as a separable PR):
  - [phase-1-critical-fixes.md](phase-1-critical-fixes.md) — Critical silent-failure & version fixes (C1, C2, C3, I1, I3, I4).
  - [phase-2-test-coverage.md](phase-2-test-coverage.md) — Tests for `report` CLI, 3D EKF/UKF analytic, gimbal lock, real-vision safety check (C4, C5).
  - [phase-3-scaffolding-cleanup.md](phase-3-scaffolding-cleanup.md) — Rename PRD references, fix stale shape annotations, sweep trivial comments.
  - [phase-4-ux.md](phase-4-ux.md) — Rename `online`→`filter`, progress reporting, `report` bridge, QA-report polish, README cleanup.
  - [phase-5-type-hardening.md](phase-5-type-hardening.md) — `state_mode` Literal, `EventLocationSource` validation, `EventChannel` grouping, `FilterState.create()`.
  - [phase-6-filter-polish.md](phase-6-filter-polish.md) — 3D IEKF `lax.scan` parity, layout-aware ZUPT, dead-code removal, analytic 3D Jacobian.
