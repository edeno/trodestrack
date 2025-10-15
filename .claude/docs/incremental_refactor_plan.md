# Trodestrack Incremental Refactor Plan

This plan defines focused, incremental changes to modernize the filter architecture of Trodestrack.
It follows **CLAUDE.md** conventions: small PRs, typed APIs, shape-stable JAX code, and strong test parity.
Priority: **2D pose + 3D IMU** readiness. TTL/RFID sensors deferred to a later milestone.

---

## Guiding Principles

- **Zero behavior drift** until explicitly allowed (numerical parity enforced).
- **Static shapes** for JAX (no ragged measurement vectors).
- **Explicit, typed interfaces** using `Protocol` and dataclasses.
- **One concern per PR**, each ≤ 400 LOC.
- **Names that read like prose** (`state_mean`, `state_cov`, `meas_pred`, `jacobian_H`, `dt_seconds`).

---

## PR0 — Housekeeping & Guardrails

**Goal:** ensure stable diffs and reproducible results.

**Changes**

- `pyproject.toml`: add black, isort, ruff, mypy strict mode for `src/trodestrack/models`.
- `src/trodestrack/__init__.py`: expose public API via `__all__`.
- Add `scripts/check_parity.sh` to run regression suites.

**Acceptance**

- All tests green.
- No mypy or ruff violations.

---

## PR1 — MeasurementModel Protocol (Camera + Heading)

**Goal:** introduce a minimal, typed interface for all sensor measurements.
No functional change yet — wraps existing helpers.

**New files**

```
src/trodestrack/models/sensors/protocols.py
src/trodestrack/models/sensors/camera_position.py
src/trodestrack/models/sensors/heading_pseudo.py
```

**Protocol**

```python
@runtime_checkable
class MeasurementModel(Protocol):
    @property
    def meas_dim(self) -> int: ...
    def predict(self, state_mean: jnp.ndarray) -> jnp.ndarray: ...
    def jacobian(self, state_mean: jnp.ndarray) -> jnp.ndarray | None: ...
    def meas_cov(self, frame_idx: int) -> jnp.ndarray: ...
    def innovation(self, frame_idx: int, meas_pred: jnp.ndarray) -> jnp.ndarray: ...
    def subspace(self, frame_idx: int) -> tuple[bool, bool, bool, jnp.ndarray]: ...
```

**Acceptance**

- Parity with existing EKF/UKF outputs (≤1e‑7 mean, ≤1e‑6 cov diag).
- Unit tests validate LED validity → projector consistency.

---

## PR2 — Generic Projected Update Primitives

**Goal:** unify duplicated EKF/UKF lifted update logic.

**New module**

```
src/trodestrack/models/filter_update.py
```

**Functions**

```python
def ekf_projected_update(...):  # Joseph form, 4D→2D projection
def ukf_projected_update(...):  # reconstruct covariance from sigma points
```

**Acceptance**

- Full numerical parity on NIS/loglikelihood and state updates across all LED modes.
- Benchmarks regress <5%.

---

## PR3 — Integrate MeasurementModel and Generic Updates

**Goal:** wire EKF/UKF to use the new interface and updates.

**Changes**

- `ekf.py` / `ukf.py`: replace inlined camera/heading logic with model calls.
- Pass `layout` explicitly.
- Keep public signatures identical.

**Acceptance**

- All regression and filter tests pass within tolerances.
- Throughput unchanged (±5%).

---

## PR4 — ZUPT as a First-Class Sensor

**Goal:** unify ZUPT handling with the new measurement system.

**Changes**

- `src/trodestrack/models/sensors/zupt.py` implements `MeasurementModel`.
- Runtime builds `active_models` per frame: `[camera, heading?, zupt?]`.

**Acceptance**

- Same state trajectories when stationary windows occur.
- Off → identical to current behavior.

---

## PR5 — 2D Pose + 3D IMU (Gravity-Aware Dynamics)

**Goal:** extend process model to handle 3D IMU while keeping 2D position estimate.

**Changes**

- **filter_common.py**: add

  ```python
  def rotate_body_accel_to_world(accel_body, yaw_heading) -> jnp.ndarray
  def gravity_compensate(accel_world, g=9.81) -> jnp.ndarray
  ```

- **process_noise.py**: update `assemble_Q()` to consume all 3 accel axes.
- **state_layout.py**: clarify indices for velocity, accel bias (3D).
- Update `dynamics_function()` to use new helpers.

**Acceptance**

- `tests/sim/test_rat_imu_gravity.py` passes with improved drift and gravity magnitude ≈9.81 m/s².
- No API or layout breakage.

---

## PR6 — Performance Tighten (JIT + Donation)

**Goal:** optimize scan performance and memory.

**Changes**

- Wrap main run functions in `jax.jit(static_argnames=("layout",))`.
- Use `donate_argnums` for large arrays.
- Remove Python `if` branches inside JAX scans; use projection masking.

**Acceptance**

- ≥20% speedup or ≤ same time with lower peak memory.
- No new recompilations.

---

## PR7 — CLI & Docs

**Goal:** expose new behavior and document it.

**Changes**

- CLI flags for enabling heading/ZUPT sensors.
- Update `README.md`, `TUNING.md`, `TROUBLESHOOTING.md`.

**Acceptance**

- CLI tests green.
- Docs accurately reflect 2D+3D IMU readiness.

---

## Deferred: TTL/RFID Event Sensors

These will reuse the `MeasurementModel` protocol and the same update primitives, but are **not part of this milestone**.

---

## Variable Naming Guidelines

| Concept | Preferred name |
|----------|----------------|
| State mean | `state_mean` |
| State covariance | `state_cov` |
| Dynamics step | `dynamics_fn(state_mean, control, dt_seconds, layout)` |
| Measurement prediction | `meas_pred` |
| Innovation | `innovation` |
| Measurement covariance | `meas_cov` |
| Jacobian | `jacobian_H` |
| LED projection | `projector_M2` |
| Frame index | `frame_idx` |
| Time step | `dt_seconds` |

---

## Test Plan Summary

| PR | Key Tests | Criteria |
|----|------------|----------|
| PR1 | LED validity / projector | parity ≤1e‑7 mean |
| PR2 | EKF/UKF update parity | same NIS/loglik |
| PR3 | Integration | full regression parity |
| PR4 | ZUPT toggle | matches old behavior |
| PR5 | 3D IMU gravity | reduced drift, correct g |
| PR6 | Performance | ≥20% faster |
| PR7 | CLI/docs | correct sensor toggles |

---

## Risk & Mitigation

- **JIT recompiles:** keep measurement shapes static.
- **Scope creep:** 3D orientation deferred; focus on 2D pose + 3D IMU.
- **Regression:** parity tests guard every PR.

---

## Implementation Order

1. PR0 → PR1 → PR2 → PR3 (core refactor group)
2. PR4 (ZUPT integration)
3. PR5 (2D pose + 3D IMU — highest priority)
4. PR6 (performance)
5. PR7 (CLI/docs)
6. TTL/RFID deferred
