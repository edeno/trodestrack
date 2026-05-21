# Phase 6 — Filter core polish

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Low-priority refactors with no user-visible behavior change. Ships as `0.4.1` (patch). All changes preserve numerical output to `rtol=1e-7`; use the [Baseline-then-refactor pattern](shared-contracts.md#baseline-then-refactor-pattern).

**Inputs to read first:**

- [src/trodestrack/models/ekf.py](../../../../src/trodestrack/models/ekf.py) lines 1455-1490 — 3D IEKF Python `for` loop. Refactor target.
- [src/trodestrack/models/ekf.py](../../../../src/trodestrack/models/ekf.py) lines 737-739 — 2D IEKF `lax.scan` pattern. Reference implementation to mirror.
- [src/trodestrack/models/filter_common.py](../../../../src/trodestrack/models/filter_common.py) lines 2020-2043 — `update_zupt` dimension-only layout lookup.
- [src/trodestrack/models/filter_common.py](../../../../src/trodestrack/models/filter_common.py) lines 2315-2342 — `gaussian_log_likelihood` dead jitter fallback.
- [src/trodestrack/models/filter_common.py](../../../../src/trodestrack/models/filter_common.py) lines 1834-1889 — `initialize_state` Python branching on JAX scalars.
- [src/trodestrack/models/filter_common.py](../../../../src/trodestrack/models/filter_common.py) lines 1609-1628 — `build_quaternion_transition_jacobian` optional `u_imu`.
- [src/trodestrack/models/sensors/camera_position_3d.py](../../../../src/trodestrack/models/sensors/camera_position_3d.py) lines 241-243 — `geometric_jacobian` uses `jacfwd`; analytic form mirrors 2D model.
- [src/trodestrack/models/sensors/camera_position.py](../../../../src/trodestrack/models/sensors/camera_position.py) lines 244-299 — analytic 2D Jacobian; reference for the 3D refactor.
- [src/trodestrack/models/process_noise.py](../../../../src/trodestrack/models/process_noise.py) lines 236-264, 353-366 — per-index `.at[].set()` loop and the `freeze_bias_during_blackout` vectorized pattern.
- [src/trodestrack/models/ukf.py](../../../../src/trodestrack/models/ukf.py) lines 374-389 — `_outer_product_batch` using `vmap(lambda a, b: jnp.atleast_2d(a).T @ jnp.atleast_2d(b))`.
- [src/trodestrack/runtime/offline.py](../../../../src/trodestrack/runtime/offline.py) lines 680-682, 725-726 — clean `vmap(jnp.outer, in_axes=(0, 0))` reference.
- [tests/filters/test_ekf_3d_analytic.py](../../../../tests/filters/test_ekf_3d_analytic.py) — created in Phase 2; this is the parity oracle for the 3D IEKF refactor. PHASE 2 MUST SHIP BEFORE THIS PHASE.
- The shared fixture is `simulate_3d_session` (no leading underscore) in `tests/filters/conftest.py`. Phase 2 owns its creation; Phase 6 only imports it. If Phase 2 named the fixture differently, align here.

**Contracts referenced:**

- [Baseline-then-refactor pattern](shared-contracts.md#baseline-then-refactor-pattern) — capture EKF3DResult and EKFResult arrays before each refactor; assert `rtol=1e-7` after.

## Tasks

### Task 1 — Capture pre-refactor baselines

Before any code changes, run the Phase 2 analytic tests and pickle the `EKF3DResult` / `EKFResult` outputs:

```python
import pickle, numpy as np, jax.numpy as jnp
from tests.filters.conftest import simulate_3d_session  # added in Phase 2
from trodestrack.models.ekf import extended_kalman_filter_3d

session = simulate_3d_session(seed=42, duration_s=30)
result = extended_kalman_filter_3d(**session.filter_kwargs())
baseline = {
    "means": np.asarray(result.means),
    "covariances": np.asarray(result.covariances),
    "marginal_loglik": float(np.asarray(result.marginal_loglik)),
}
with open(f"/tmp/phase6-baseline-{git_hash}.pkl", "wb") as f:
    pickle.dump(baseline, f)
```

Also capture timing: `time uv run pytest tests/benchmark/test_ekf_3d_core_jit.py -v` and record JIT compile + execute time. Phase 6 should reduce JIT compile time (no Python-loop unrolling) and preserve execute time.

### Task 2 — Convert 3D IEKF Python `for` loop to `lax.scan`

In [src/trodestrack/models/ekf.py:1464-1488](../../../../src/trodestrack/models/ekf.py#L1464-L1488), the current implementation unrolls `config.num_iter` iterations as a Python loop, which JIT-compiles linearly with `num_iter`. Mirror the 2D pattern at [src/trodestrack/models/ekf.py:737-739](../../../../src/trodestrack/models/ekf.py#L737-L739):

```python
def do_update(_: None) -> tuple[EKFState, jnp.ndarray]:
    def iter_body(carry: tuple[EKFState, jnp.ndarray, jnp.ndarray], _i: int):
        state_iter, _innov, _S = carry
        new_state, innov, S = _camera_3d_linear_update(
            state, state_iter.mean, camera_model, frame_idx, layout,
        )
        return (new_state, innov, S), None

    init = (
        state,
        jnp.zeros(camera_model.meas_dim, dtype=state.mean.dtype),
        jnp.eye(camera_model.meas_dim, dtype=state.mean.dtype),
    )
    (state_iter, innovation, S), _ = lax.scan(
        iter_body, init, jnp.arange(config.num_iter)
    )

    log_lik = gaussian_log_likelihood_masked(innovation, S, active_mask)
    if config.use_mahalanobis_gating:
        nis = _mahalanobis_distance_masked(innovation, S, active_mask)
        accept_update = nis < chi2_thresholds[dof]
    else:
        accept_update = jnp.asarray(True)
    return lax.cond(
        accept_update,
        lambda _: (state_iter, log_lik),
        lambda _: (state, jnp.asarray(0.0, dtype=state.mean.dtype)),
        operand=None,
    )
```

Run the Phase 2 analytic tests. Assert outputs match the pickle baseline within `rtol=1e-7` (small floating-point reorderings are acceptable; semantic equivalence is required).

### Task 3 — Make `update_zupt` accept explicit `layout`

In [src/trodestrack/models/filter_common.py:2020-2043](../../../../src/trodestrack/models/filter_common.py#L2020-L2043), change the signature to require `layout`:

```python
def update_zupt(
    state: FilterState,
    *,
    config: FilterCoreConfig,
    active: bool | jnp.ndarray,
    layout: StateLayout,   # NEW: required
) -> tuple[FilterState, jnp.ndarray]:
    """..."""
    mean, cov = state
    # Dimension lookup is gone; layout is the source of truth.
    if mean.shape[0] != layout.n:
        raise ValueError(
            f"update_zupt: state has dim {mean.shape[0]} but layout.n={layout.n}."
        )
    # ... (rest of the function uses `layout` directly)
```

Update every call site. Grep `grep -rn 'update_zupt' src/`. Each caller already has a `layout` in scope (via `FilterCoreConfig.state_mode` → `get_layout(state_mode)`); pass it explicitly. The 2D EKF call site is around `src/trodestrack/models/ekf.py` (grep `update_zupt`); same for UKF.

Delete the dimension-only registry walk at lines 2031-2043 — it's replaced by the explicit parameter.

### Task 4 — Delete dead jitter fallback in `gaussian_log_likelihood`

In [src/trodestrack/models/filter_common.py:2315-2342](../../../../src/trodestrack/models/filter_common.py#L2315-L2342) (re-grep `gaussian_log_likelihood` to confirm line range), remove the `add_more_jitter` fallback path. The adaptive leading jitter (`1e-8 * trace/k`) is sufficient; if `sign <= 0` after that, the right action is to surface the LinAlgError. Per the project's "no defensive handling for impossible cases" rule.

Add a one-line comment explaining the deletion: `# If sign <= 0 after symmetrize + adaptive jitter, the matrix is genuinely indefinite — let slogdet's sign field propagate so callers see the issue.`

Run the parity test from Task 1 — outputs should be bit-exact (this branch was never taken on healthy inputs).

### Task 5 — Replace `jacfwd` with analytic Jacobian in `Camera3DPositionModel`

In [src/trodestrack/models/sensors/camera_position_3d.py:241-243](../../../../src/trodestrack/models/sensors/camera_position_3d.py#L241-L243), replace the autodiff Jacobian with the analytic form. The 3D camera observation is `h(x) = [position + R(q) @ offset_led1; position + R(q) @ offset_led2]`. The Jacobian has two blocks:

- `∂h/∂position = stack[I₃; I₃]` (each LED contributes an identity wrt position).
- `∂h/∂quaternion = stack[∂R(q)/∂q @ offset_led1; ∂R(q)/∂q @ offset_led2]`.

The 2D model at [src/trodestrack/models/sensors/camera_position.py:244-299](../../../../src/trodestrack/models/sensors/camera_position.py#L244-L299) provides the analytic 2D template. For 3D, use the quaternion-to-rotation derivative `∂R(q)/∂q` (a `(3, 3, 4)` tensor) — derive it once on paper, write it out:

```python
def _drotation_dquaternion(q: jnp.ndarray) -> jnp.ndarray:
    """Derivative of the quaternion-to-rotation-matrix wrt q (scalar-first).

    Returns
    -------
    jnp.ndarray, shape (3, 3, 4)
        dR/dq[i, j, k] = ∂R[i, j] / ∂q[k].
    """
    qw, qx, qy, qz = q
    # ... derived from R(q) = ... ; verify against jacfwd in test.
```

Add a regression test in `tests/models/sensors/test_camera_position_3d.py` (created in Phase 2):

- `test_geometric_jacobian_matches_jacfwd` — sample 50 random (state, quaternion) pairs; assert `model.geometric_jacobian(state)` agrees with `jax.jacfwd(model.predict)(state)` to `rtol=1e-6`.

Ship only if the test passes for all 50 samples. If any sample diverges (likely due to non-unit-norm quaternion edge cases), document and revert to `jacfwd` for this phase.

### Task 6 — Vectorize per-index multipliers in `assemble_Q`

In [src/trodestrack/models/process_noise.py:236-264](../../../../src/trodestrack/models/process_noise.py#L236-L264), the current implementation iterates `for idx in layout.pos_idx: Q = Q.at[idx, idx].mul(pos_mult)` and similar. For 16D layouts this is 12 separate `.at[].set()` calls. Collapse to a single vectorized scaling:

```python
# Build a per-component multiplier vector
scaling = jnp.ones(layout.n, dtype=Q.dtype)
scaling = scaling.at[jnp.asarray(layout.pos_idx)].mul(pos_mult)
scaling = scaling.at[jnp.asarray(layout.vel_idx)].mul(vel_mult)
# ... etc for bias_gyro, bias_accel ...

# Apply as outer-product scaling (preserves the rate * dt^k structure)
Q_scaled = Q * scaling[:, None] * scaling[None, :]
```

(This mirrors the `freeze_bias_during_blackout` pattern at lines 353-366.) Verify outputs match the pickle baseline within `rtol=1e-7`.

### Task 7 — Simplify `_outer_product_batch` in `ukf.py`

In [src/trodestrack/models/ukf.py:374-389](../../../../src/trodestrack/models/ukf.py#L374-L389), the current implementation is:

```python
def _outer_product_batch(a, b):
    return vmap(lambda a_, b_: jnp.atleast_2d(a_).T @ jnp.atleast_2d(b_))(a, b)
```

Replace with the cleaner formulation already used elsewhere (`offline.py:680-682`):

```python
def _outer_product_batch(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Per-sample outer product: (N, k) × (N, k) → (N, k, k)."""
    return vmap(jnp.outer, in_axes=(0, 0))(a, b)
```

The 1-D-only call sites in `ukf.py` (UKF covariance reconstruction) produce identical outputs. Verify parity against pickle baseline.

### Task 8 — Require `u_imu` in `build_quaternion_transition_jacobian`

In [src/trodestrack/models/filter_common.py:1609-1628](../../../../src/trodestrack/models/filter_common.py#L1609-L1628), the function currently builds `quat_self_block = projector` and only overwrites when `u_imu is not None`. The docstring warns the no-`u_imu` path makes "covariance / gating / smoothing inconsistent with the mean for non-zero gyro samples". Every actual caller passes `u_imu`.

Drop the optional `u_imu` parameter; make it required:

```python
def build_quaternion_transition_jacobian(
    q: jnp.ndarray,
    dt: float | jnp.ndarray,
    u_imu: jnp.ndarray,    # was: jnp.ndarray | None = None
    *,
    bias_gyro_idx: tuple[int, ...] | None,
    state_dim: int,
) -> jnp.ndarray:
    """..."""
```

Update the docstring to remove the warning about the no-`u_imu` path. Verify callers compile (existing tests cover this).

### Task 9 — Document `initialize_state` as host-only

In [src/trodestrack/models/filter_common.py:1834-1889](../../../../src/trodestrack/models/filter_common.py#L1834-L1889), `initialize_state` uses Python `if`/`len` on what may be JAX scalars. It works today because all call sites are host-side. Don't refactor (Task 9 is documentation only): add an explicit warning to the docstring:

```python
def initialize_state(...):
    """Initialize filter state from first valid observation.

    Notes
    -----
    This function is host-only and must not be called inside
    ``jax.jit``. It uses Python-level branching on observation validity
    that traces non-statically. Call from host code that resolves
    initial state once before entering the JIT-compiled scan.
    """
```

### Task 10 — Remove redundant `replace()` re-validation in `extended_kalman_filter` / `unscented_kalman_filter`

In [src/trodestrack/models/ekf.py](../../../../src/trodestrack/models/ekf.py) (grep `replace(.*EKFConfig` or `config_for_filter = replace`) and [src/trodestrack/models/ukf.py](../../../../src/trodestrack/models/ukf.py) (around `ukf.py:1109`):

When `led_distance is None`, the current code uses `dataclasses.replace(ekf_config, led_distance=auto)` which re-runs `FilterCoreConfig.__post_init__` validation. The original config already passed validation; re-running it is wasted work.

Replace with a `_with_led_distance` helper that bypasses `__post_init__`:

```python
def _with_led_distance(config: EKFConfig, led_distance: float) -> EKFConfig:
    """Shallow-clone config with a new led_distance, skipping re-validation."""
    new_cfg = object.__new__(type(config))
    object.__setattr__(new_cfg, "__dict__", dict(config.__dict__))
    object.__setattr__(new_cfg, "led_distance", led_distance)
    return new_cfg
```

(For dataclasses without `__dict__` because of `__slots__`, use `dataclasses.fields` instead. EKFConfig is not slotted today; verify before assuming.)

Caveat: this skips PyTree round-trip safety. If `EKFConfig` is constructed inside a JAX trace, the `__post_init__` validation is needed for correctness. Verify by reading the existing call sites — the `_with_led_distance` is only called once host-side per filter run, never inside a scan.

### Task 11 — Post-refactor baseline comparison

Re-run the Phase 2 analytic tests; load each pickle baseline; assert `rtol=1e-7` for every output array:

```python
for key in ("means", "covariances"):
    np.testing.assert_allclose(new[key], baseline[key], rtol=1e-7, err_msg=key)
np.testing.assert_allclose(new["marginal_loglik"], baseline["marginal_loglik"], rtol=1e-9)
```

Also re-run the timing comparison from Task 1. Expected outcome: JIT compile time decreased by ~`(num_iter - 1) × per-iter-compile` seconds (the unrolled Python loop is gone); execute time unchanged within 5%.

### Task 12 — CHANGELOG entry

Add under `## [0.4.1] — unreleased`:

```
### Changed (internal)
- 3D IEKF (`extended_kalman_filter_3d`) now uses `lax.scan` for IEKF iterations, matching the 2D pattern. Reduces JIT compile time linearly with `num_iter`; numerical output preserved to rtol=1e-7.
- `update_zupt` now requires an explicit `layout` parameter, removing a dimension-based registry lookup that would silently miswire any future layout sharing a dimension with an existing one.
- `build_quaternion_transition_jacobian` now requires `u_imu` (previously optional with a warning); every real call site passed it.
- `Camera3DPositionModel.geometric_jacobian` uses an analytic Jacobian instead of `jacfwd`. Faster JIT compile; bit-exact within rtol=1e-6.
- `process_noise.assemble_Q` vectorizes per-component multipliers via outer-product scaling instead of per-index `.at[].set()` loops.
- `_outer_product_batch` in ukf.py simplified to `vmap(jnp.outer)`.
- Dead `add_more_jitter` fallback removed from `gaussian_log_likelihood`.
- `extended_kalman_filter` and `unscented_kalman_filter` no longer re-run full `FilterCoreConfig.__post_init__` validation when overriding `led_distance` at runtime.

### Documented
- `initialize_state` docstring now states it is host-only; do not call inside `jax.jit`.
```

## Deliberately not in this phase

- **Layout-aware `EKFResult` / `EKFR3DResult` types** (carrying the `StateLayout` that produced them) — type-design suggestion, deferred to a follow-up phase. The dimension checks at the smoother boundary (`offline.py:482-491`, `:860-869`) remain in place.
- **Variable-`dim` likelihood normalization fix in `gaussian_log_likelihood_masked`** — needs design work; skipped here. Document as an open issue.
- **MeasurementModel protocol refactor** (UKF/EKF split) — type-design suggestion; skipped.
- **EventLocationModel `predict(*, source_indices=...)` signature unification** — protocol-divergence concern; intentionally left as-is per the docstring rationale.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_ekf_3d_lax_scan_parity_with_baseline` | Phase 6 outputs match Phase 6 pre-baseline within `rtol=1e-7` for means, covariances, marginal_loglik. |
| `test_ekf_3d_jit_compile_time_decreased` | `pytest tests/benchmark/test_ekf_3d_core_jit.py` — measured compile time ≤ baseline (the Python-loop unrolling cost is gone). |
| `test_update_zupt_requires_explicit_layout` | `update_zupt(state, config=cfg, active=True)` (without layout) raises `TypeError` or `ValueError`. |
| `test_update_zupt_rejects_layout_dim_mismatch` | `update_zupt(state(n=10), ..., layout=layout(n=8))` raises ValueError. |
| `test_gaussian_log_likelihood_no_more_jitter_branch` | Coverage check: the deleted branch has no live code; explicit test that passes a near-singular matrix and asserts the function still returns a finite value or propagates the LinAlgError. |
| `test_camera_3d_geometric_jacobian_matches_jacfwd` | 50 random (state, q) samples — analytic Jacobian agrees with `jacfwd` to `rtol=1e-6`. |
| `test_assemble_q_vectorized_matches_loop` | Build Q via the new vectorized path; assert bit-exact equality with a reference loop implementation kept in the test file. |
| `test_outer_product_batch_vmap_outer_parity` | New `vmap(jnp.outer)` produces same result as the old `vmap(atleast_2d.T @ atleast_2d)` for 100 random (N, k) inputs. |
| `test_build_quaternion_transition_jacobian_no_optional_u_imu` | Calling without `u_imu` is now a `TypeError`. |
| `test_with_led_distance_skips_revalidation` | Construct `EKFConfig` once; call `_with_led_distance` 1000 times; assert wall-clock is dominated by the original construction (re-validation cost is gone). |
| `test_full_test_suite_passes` | `uv run pytest -m "not slow"` exits 0. |
| `test_full_slow_test_suite_passes` | `uv run pytest -m slow` exits 0. |

## Fixtures

- Reuse `simulate_3d_session` from `tests/filters/conftest.py` (added in Phase 2).
- Add a `tests/regression/test_3d_iekf_lax_scan_parity.py` test file that captures-and-compares against the pickle baseline. The pickle itself is not checked in; the test runs the baseline computation once at the top of the test, then again after each refactor and compares. This makes the test self-contained. The file name describes the behavior under test, not the plan milestone — do NOT name it `test_phase6_*`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented.
- Phase 2 has shipped; the 3D analytic tests exist (Task 2's parity oracle).
- Baseline-comparison tests pass with `rtol=1e-7`.
- JIT compile time on `tests/benchmark/test_ekf_3d_core_jit.py` decreased or stayed equal (recorded in PR description).
- `grep -rn 'update_zupt' src/` shows every call site now passes `layout=` explicitly.
- The dead jitter fallback is gone (no `add_more_jitter` branch).
- The analytic 3D Jacobian agrees with `jacfwd` on 50 random samples — if not, the refactor reverted to `jacfwd` and that's documented in the PR.
- `_with_led_distance` helper does not skip validation when the config is constructed inside a JAX trace (verify by reading call sites).
- CHANGELOG entry is in `## [0.4.1]` (patch — internal-only refactor).
- Docstrings, test names, and module names don't reference this plan.
