# Incremental Refactoring Plan for Extensible Filter Architecture

This plan provides a **step-by-step refactor strategy** aligned with CLAUDE.md best practices. Each step is small, reviewable, and fully test-backed, improving maintainability and extensibility.

---

## Phase 0 — Stabilize the Foundation

### PR0: Packaging & CI Guardrails

- Add `trodestrack/__main__.py` with `main()`; wire `[project.scripts].trodestrack`.
- Unify dependencies (`uv`).
- Pin compatible JAX/NumPy versions.
- Add `pytest.ini` with skip marks (`slow`, `benchmark`).

**DoD:** CLI runs locally; CI green.

---

## Phase 1 — Shared Primitives (DRY Foundation)

### PR1: Centralize Gating, Confidence, and Angle Wrap

- Implement in `filter_common.py`:
  - `confidence_to_R(...)`
  - `mahalanobis_gate(...)`
  - `wrap_angle(...)`
- Replace duplicates in EKF/UKF.

**DoD:** EKF/UKF tests pass; functions typed with NumPy docstrings.

---

## Phase 2 — Measurement Model Interface

### PR2: Introduce `MeasurementModel` Protocol

- Add `sensors/base.py` defining a typed `MeasurementModel` interface.
- Move existing camera logic into `sensors/camera_position.py`.

**DoD:** Camera code modularized; filters unchanged.

---

## Phase 3 — Sensor Registry and Fusion Tick

### PR3: Sensor Registry + Tick Builder

- Add `SensorRegistry` and `Tick` dataclasses.
- Centralize data alignment (IMU, camera, etc.).

**DoD:** Existing runtime compatible.

---

## Phase 4 — Unified EKF Update

### PR4: `update_generic()` with Joseph Form

- Move EKF update math into one projected Joseph-form function.
- Factor once (`S`, `L`, `nis`, `K`), reuse results.

**DoD:** Numerical parity; PSD preserved.

---

## Phase 5 — UKF Parity

### PR5: UKF Uses Same Update Primitive

- Replace redundant math; add `projected_joseph()`.
- Fix-size projections for static JIT shape.

**DoD:** EKF/UKF parity; PSD-safe UKF.

---

## Phase 6 — Performance Optimizations

### PR6: JIT Scan + Donation

- Fuse predict+update in a single `jit(lax.scan)`.
- Donate large buffers (`x`, `P`).
- Replace Python control with `lax.cond`.

**DoD:** ~30% faster steady-state; fewer recompiles.

---

## Phase 7 — ZUPT as Sensor

### PR7: Port ZUPT to `MeasurementModel`

- Implement `sensors/zupt.py`.
- IMU-only mode supported via config.

**DoD:** Identical ZUPT behavior; IMU-only supported.

---

## Phase 8 — TTL/RFID Integration

### PR8: Add Sparse Sensors

- Add `sensors/ttl.py` and `sensors/rfid.py`.
- Sparse event-based models.

**DoD:** Examples + basic tests; disabled by default.

---

## Phase 9 — Multi-Camera Support

### PR9: Multiple Camera Instances

- Support multiple `CameraPositionModel`s.
- Fuse sequential updates in each tick.

**DoD:** Multi-camera accuracy ≥ single camera.

---

## Phase 10 — Smoother & IEKS Improvements

### PR10: RTS / IEKS Optimizations

- Replace inverses with triangular solves.
- Add `jax.checkpoint()` and chunked smoothing.

**DoD:** Memory 40% lower; equivalent accuracy.

---

## Phase 11 — Layout v2 and 3D Hooks

### PR11: Extend `StateLayout` for 3D

- Add explicit indices and guards for unsupported modes.

**DoD:** 3D-ready API; no silent misuse.

---

## Phase 12 — CLI & Documentation

### PR12: CLI Commands + README Refresh

- Add `trodestrack filter|simulate|report` subcommands.
- Update install matrix, configs, and examples.

**DoD:** End-to-end pipeline runnable via CLI.

---

## Rollout Order

1. PR0 → PR1 → PR2 (no behavior changes)
2. PR3 (structural only)
3. PR4 → PR5 (filter refactors)
4. PR6 (perf JIT/scan)
5. PR7–PR9 (sensors)
6. PR10–PR12 (smoother, layout, CLI)

---

## CLAUDE.md Alignment Checklist

✅ Typed public APIs
✅ ≤20-line functions
✅ NumPy docstrings with shapes/units
✅ Deterministic PR scope
✅ Unit tests per module
✅ No mutable defaults
✅ `pathlib.Path`, `logging`, `warnings`
✅ CI-safe, lint/format on commit

---
