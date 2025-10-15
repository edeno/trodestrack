# JAX optimization principles

## 1) Shapes & compilation

* **Static shapes win.** Avoid data-dependent tensor shapes inside `jit`/`scan`. Use fixed-size arrays + **projection** or **masking** instead of ragged tensors.
* **Freeze config as static.** `jax.jit(static_argnames=("layout", "config_flags", ...))` so these don’t trigger recompiles.
* **Avoid Python polymorphism.** No subclass dispatch or Python conditionals on array values in jitted code; prefer `lax.cond` / `lax.switch`.
* **Prefer `jnp.asarray` at the boundary.** Convert inputs once; don’t bounce between NumPy and JAX.
* **Keep dtypes stable.** Don’t mix float32/float64 in hot paths; opt in to 64-bit via `jax_enable_x64` only if needed.

## 2) Loops & control flow

* **Use `lax.scan` for time loops.** Single compilation, linear memory footprint, good for filters/smoothers.
* **Vectorize with `vmap`** instead of Python `for` where possible (e.g., batch updates, sigma-point transforms).
* **Use `lax.cond`/`lax.select` for branches** on array predicates (e.g., single-LED vs dual-LED).
* **Make loop bodies pure.** No mutation of Python objects or closures during the scan.

## 3) Data movement & device residency

* **Keep everything on device.** Preload arrays to JAX (e.g., IMU indices, camera tracks). Index them with JAX integers in the scan.
* **Minimize host↔device transfers.** No `.block_until_ready()` or `.item()` in hot paths; log outside the `jit` or via `jax.debug.print` sparingly.
* **Avoid Python containers in the trace.** Use JAX arrays/pytrees, not dicts/lists, for per-frame data.

## 4) Memory, allocation, and throughput

* **Donate large buffers.** `jax.jit(donate_argnums=(0, ...))` for big arrays updated in place.
* **Re-use workspaces.** Prefer functional updates on preallocated arrays; avoid building new pytrees every step.
* **Control materialization.** Beware of `jnp.where` with large branches—both sides materialize; push heavy compute behind `lax.cond` when only one side is needed.
* **Chunk long scans** (e.g., by seconds or frames) if you hit memory limits; stitch outputs.

## 5) Numerics & stability

* **Symmetrize & PSD-solve.** Use `symmetrize(P)` and PSD-aware solves (`cholesky`, `triangular_solve`) instead of naive `inv`.
* **Joseph form for EKF**, and careful covariance reconstruction for UKF.
* **Regularize carefully.** Tiny jitter before Cholesky; don’t silently over-inflate.
* **Angle wrap at the edges.** Enforce invariants (e.g., heading ∈ [−π, π]) right after predict/update.

## 6) Randomness & determinism

* **Stateless RNG.** Thread PRNGKeys explicitly; split keys once per step; don’t hide state in globals.
* **Deterministic tests.** Fix seeds; assert numerical tolerances (means, cov diag) not bit-exactness.

## 7) Autodiff hygiene

* **Prefer analytic Jacobians** in hot paths (cheap & stable) and use autodiff where it’s complex/offline.
* **Stop gradients where intended.** Use `lax.stop_gradient` when residuals shouldn’t backprop (often in filters).
* **Custom VJP/JVP** only if profiling shows it’s a win; keep the reference implementation for tests.

## 8) API surfaces (to avoid recompiles)

* **Typed, minimal protocols.** Structural typing (`Protocol`) over inheritance; no runtime MRO during jit.
* **Static measurement dims.** Camera exports fixed 4D; use a `(2×4)` selector for single-LED projection (shapes never change).
* **Pass layout explicitly.** Don’t call global getters inside jit’d code; treat layout as a static arg.

## 9) Profiling & debugging

* **See what compiled.** `jax.log_compiles(True)` (temporarily) and watch for unexpected recompiles.
* **Trace the HLO/JAXPR.** `jax.make_jaxpr(fn)(args...)` for control flow and staging issues.
* **Profile hot paths.** `jax.profiler.trace(...)/xprof` to find allocation or dispatch bottlenecks.
* **Use `debug.print` sparingly** inside jit for scalar breadcrumbs; remove after validation.

## 10) Project-specific patterns (trodestrack)

* **Projection over padding.** For camera, do full 4D math; project to 2D with a fixed selector when only one LED is valid—no ragged vectors.
* **Precompute indices.** IMU→camera index arrays are fixed-width JAX arrays padded with `-1`; guard with `>=0` in `lax.cond`.
* **One update primitive per filter.** Share EKF/UKF projected update code; keep shapes and gating consistent.
* **Use predicted heading for Q.** Build `G(θ⁺)`/`Q(θ⁺)` after predict for tighter alignment.
* **Keep sensors pure.** Measurement models compute `meas_pred`, `jacobian_H`, `innovation`, and `meas_cov` without mutating internal state.

---

### Tiny exemplar snippets

#### Static-arg JIT

```python
@jax.jit(static_argnames=("layout",))
def run_filter(state0, data, *, layout, config):
    return lax.scan(lambda st, u: step(st, u, layout, config), state0, data)
```

#### Branchless camera update

```python
both, only1, only2, M2 = camera.subspace(i)     # M2: (2,4) selector, static shape
state, nis, ll = ekf_projected_update(
    state, innovation4, H4, R4, both, only1, only2, M2
)
```

#### JAX-friendly indexing

```python
def propagate(state, imu_idx):
    return lax.cond(
        imu_idx >= 0,
        lambda s: predict(s, imu[imu_idx], dt(imu_idx)),
        lambda s: s,
        state,
    )
```

If you keep these principles as guardrails, you’ll get stable compile counts, predictable performance, and numerically robust filters—exactly the CLAUDE.md vibe.
