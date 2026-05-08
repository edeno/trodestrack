# OnlineTracker Streaming Class Implementation Plan

## Status

Not started. The current `trodestrack online` CLI is a forward-only
batch command (loads complete IMU/camera arrays from disk,
calls `extended_kalman_filter` once over the full session). PRD §9
specifies a `runtime.online.OnlineTracker(cfg)` Python class for
true per-frame streaming ingest; that surface does not exist yet.

## Goals

- Provide a Python class that accepts IMU samples and camera frames
  one at a time, returns the current filter state immediately, and
  meets PRD §4.4's `≤33 ms` per-frame latency budget on CPU under
  realistic conditions.
- Keep the underlying numerics identical to the existing batch
  EKF — same per-step `predict`/`update` math, same configuration,
  same outputs at the camera-frame cadence.
- Expose a clean ingest API with explicit timestamp arguments so
  callers can drive the tracker from any source (live SpikeGadgets
  ingest, ROS bridge, replay harness).

## Non-Goals

- Real-time guarantees beyond best-effort wall-clock (this is
  Python; users needing hard deadlines should drive the underlying
  `predict_step / update_step` directly).
- Multi-session or multi-tracker management (each `OnlineTracker`
  serves one session).
- IMU pre-integration policies that differ from the batch path
  (the integrator is the same; only the *ingest pattern* changes).
- Streaming smoothing (RTS by definition needs the full session).
- A streaming CLI. PRD lists `trodestrack online --config` as a
  batch command and that's what it stays; the streaming surface is
  Python-only.

## Background

The batch `extended_kalman_filter` is implemented as a
`jax.lax.scan` over camera frames. Inside the scan body, IMU
samples in the camera interval are folded into the prediction via
a nested `lax.scan`. The whole thing is JIT-compiled.

Streaming breaks the outer scan. Each public method call
processes one frame's worth of work. To stay fast we need:

1. **JIT a per-frame step**, not the whole session. The current 2D
   batch path is implemented inside `_extended_kalman_filter_impl`
   and the 3D path inside `_extended_kalman_filter_3d_core`; their
   scan bodies need to be extracted into reusable per-camera-step
   functions with the same interval and masking semantics.
2. **Cache the JIT compilation across calls**. Tracer signatures
   (state shape, IMU sample shape, layout) are constant for the
   lifetime of the tracker, so the first call pays the warmup
   cost and every subsequent call hits the cache.
3. **Avoid Python-level allocations in the hot path**. Use
   pre-allocated JAX arrays where possible; rely on JAX's
   donate-argnums to recycle state buffers.

The PRD's `≤33 ms` budget is comfortable for CPU JAX once warm:
the batch benchmark's amortized mean is ~0.41 ms/frame on M-series
CPU. Streaming overhead from Python dispatch + buffer marshalling
typically adds ~1–2 ms per call. We have plenty of headroom.

## Design Principles

- **One-step JIT, not whole-session JIT.** Compile the per-frame
  scan body once; call it imperatively from Python.
- **Same numerics as batch.** Streaming vs batch must call the same
  extracted per-frame math. **Target: bitwise-identical parity** for
  both replay-from-arrays and true per-call streaming. The streaming
  wrapper has full control over IMU window padding and the JAX
  function signature, so mirroring the batch padding scheme exactly
  is achievable. Only relax to `assert_allclose(rtol=1e-12)` if
  implementation evidence shows that host/device boundaries or
  reduction ordering produce a real, unavoidable divergence —
  pre-relaxing the contract weakens the regression guard.
- **Explicit timestamps.** Every ingest method takes a timestamp
  in seconds; no "current time" magic.
- **Fail fast on out-of-order timestamps.** Strictly increasing
  IMU and camera timestamps are part of the contract; reject NaN
  / inf / non-increasing values at the public boundary.
- **Donate state buffers** so JAX can recycle covariance arrays
  in place rather than re-allocating per call.

## Architecture

### New module — `src/trodestrack/runtime/online.py`

```python
@dataclass
class OnlineTrackerOutput:
    t: float
    mean: np.ndarray            # (n_state,)
    cov: np.ndarray             # (n_state, n_state)
    log_likelihood_increment: float
    predicted_only: bool        # True if no camera frame was consumed

class OnlineTracker:
    def __init__(
        self,
        ekf_config: EKFConfig,
        *,
        initial_state: EKFState | None = None,
        led_distance: float | None = None,
    ): ...

    def push_imu(self, t: float, imu: np.ndarray) -> None:
        """Buffer a single IMU sample; no state update yet."""

    def push_camera(
        self,
        t: float,
        Z_led1: np.ndarray,         # (2,) or (3,)
        Z_led2: np.ndarray | None,
        valid: bool,
        conf: np.ndarray | None = None,
    ) -> OnlineTrackerOutput:
        """Drain buffered IMU samples up to ``t``, run camera
        update, return the new state."""

    def state(self) -> OnlineTrackerOutput:
        """Snapshot of the current state without consuming any
        new measurement."""

    def reset(self, *, initial_state: EKFState | None = None) -> None: ...
```

Internally:

- `self._step_jit` is `jax.jit(per_camera_step, static_argnames=...)`
  with `donate_argnums` on the state.
- `self._imu_buffer` is a fixed-size ring buffer of pending IMU
  samples; sized to `max(N_imu_samples_per_camera_frame * 4, 64)`.
- `push_imu` appends; `push_camera` drains the buffer up to
  the camera timestamp, calls `_step_jit`, and returns the new
  state.

### Optional convenience — `OnlineTrackerSession`

Higher-level wrapper that drives the tracker from arrays for
testing and replay. Bridges the batch / streaming paths so the
parity test can call it with batch inputs and compare.

```python
def replay_streaming(
    ekf_config: EKFConfig,
    t_imu: np.ndarray,
    U_imu: np.ndarray,
    t_cam: np.ndarray,
    Z_cam_led1: np.ndarray,
    Z_cam_led2: np.ndarray,
    mask_cam: np.ndarray,
) -> EKFResult: ...
```

This is the contract surface for the parity test: feed the same
inputs as batch, get back an `EKFResult`.

### No public behavior changes to `extended_kalman_filter`

The batch public API and outputs stay unchanged. Milestone 1 refactors
the implementation to expose a shared per-frame core, and the existing
batch wrapper continues to drive it through `lax.scan`.

## Milestones

### Milestone 1 — Per-frame core extraction

- Refactor `extended_kalman_filter`'s scan body into a standalone
  `_per_camera_step` function with a clean signature
  `(state, ekf_config, layout, imu_window, t_imu_window, frame_obs) -> (state, scan_output)`.
- Preserve the existing `compute_imu_index_arrays` interval boundaries,
  padded-window shape, `dt` handling, ZUPT gates, and `has_seen_vision`
  carry semantics exactly.
- Verify batch path still produces identical output (regression
  test: full sweep stays green).
- Document the function's contract in the module docstring.

**Exit criteria:** zero numerical change in batch output;
`_per_camera_step` documented and reusable.

### Milestone 2 — `OnlineTracker` skeleton + parity

- `runtime/online.py` with `OnlineTracker` class.
- `replay_streaming` convenience that drives the tracker from
  batch arrays.
- **Parity test**: target bitwise-identical output between
  `extended_kalman_filter` and `replay_streaming` for both
  replay-from-arrays and the public ring-buffer streaming path. Use
  `assert_array_equal`. Drop to `assert_allclose(rtol=1e-12)` only
  if implementation evidence forces the relaxation; document the
  cause in the test if so.
- Strict-increasing timestamp + finite IMU validation at
  `push_imu` / `push_camera` boundaries.

**Exit criteria:** `tests/runtime/test_online_tracker.py` green;
parity holds across all default state modes
(`vision_only`, `2d_full`, `2d_cam_3d_imu`).

### Milestone 3 — Hot-path latency

- JIT the per-frame step with explicit static argnames and
  donate buffers.
- Warmup helper `OnlineTracker.warmup()` that runs one synthetic
  step at construction time so the first real call is hot.
- Benchmark: latency histogram on M-series Mac CPU; assert p95 ≤
  10 ms after warmup. Add to `tests/benchmark/` (slow + benchmark
  marked).

**Exit criteria:** benchmark documents headroom under PRD's 33 ms
target; no allocator churn in steady state (verified via JAX's
`block_until_ready` + `time.perf_counter` measurements per call).

### Milestone 4 — Quaternion-orientation streaming

- Extend the streaming surface to the 14D
  (`2d_cam_6dof_imu_orientation`) and 16D (`3d_cam_6dof_imu`)
  layouts. The `_per_camera_step_3d` body for the 3D EKF gets the
  same extraction treatment.
- Parity tests for the orientation layouts under streaming.

**Exit criteria:** streaming covers every state mode the batch
path covers; UKF stays excluded (UKF rejects quaternion layouts;
streaming UKF is a non-goal).

### Milestone 5 — Documentation and replay harness

- `docs/getting-started/python-api.md` worked example: ingest
  loop driving `OnlineTracker` from a `simulate_rat_imu` session.
- `examples/09_online_streaming.py` demonstrating the API.
- README "Online filtering" section now distinguishes the
  *batch CLI* from the *streaming Python API* explicitly.

**Exit criteria:** docs pass strict mkdocs build; the example
script runs end-to-end and produces a result that matches
`extended_kalman_filter` to numerical tolerance.

## Validation Matrix

| Test | Layer | Asserts |
|---|---|---|
| Parity: replay_streaming vs batch | runtime | bitwise identical state and log-likelihood when using the same padded windows as batch |
| Out-of-order IMU timestamp | API | `ValueError` at `push_imu` |
| Out-of-order camera timestamp | API | `ValueError` at `push_camera` |
| Empty IMU buffer at camera time | API | allowed for `vision_only`; otherwise clear `ValueError` or documented no-IMU prediction behavior |
| Latency p95 ≤ 10 ms after warmup | benchmark | wall-clock per `push_camera` call |
| Quaternion-layout streaming parity | runtime | identical to batch for 14D / 16D |
| Reset preserves config | API | post-reset filter state matches a fresh tracker with same config |

## Metrics

- **Per-frame latency** (camera-frame ingest, post-warmup):
  - p50: ≤ 2 ms (batch reference: ~0.41 ms amortized mean)
  - p95: ≤ 10 ms
  - max: ≤ 33 ms (PRD floor)
- **Numerical parity**: target exact equality (`assert_array_equal`)
  for both replay-from-arrays and per-call streaming. Fall back to
  `assert_allclose(rtol=1e-12)` only if implementation evidence
  forces it; document the cause in the test.
- **Memory**: O(1) per-call after warmup; no allocations in the
  steady state.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| JIT recompiles on every call due to shape drift | Static argnames pinned; assertion that IMU buffer length is constant after warmup. |
| Python dispatch overhead exceeds 33 ms on slow hardware | Document the realistic headroom; offer a "raw-step" API (`OnlineTracker.step_raw(...)`) that skips Python validation for users who pre-validate. |
| Users feed unsorted IMU samples | Strict-increasing-timestamps gate at `push_imu`; clear error message. |
| Tracker state divergence between `replay_streaming` and batch under floating-point reordering | The per-frame core extraction is mechanical (same code path); parity test catches any regression on contributor PRs. |
| Donate-argnums semantics surprises users (state buffer reused) | The public API returns NumPy copies via `np.asarray(...)`; donation is invisible to the user. |

## Rollout Strategy

- Ship behind a new `runtime.online.OnlineTracker` import; no
  existing code changes path.
- Parity test gates the wiring; if it fails the streaming
  surface doesn't ship.
- Mention in README's "Online filtering" section that the API is
  Python-only and the CLI stays batch.
- After one user has driven it from a real Trodes ingest loop,
  promote to "stable" in docs.

## Documentation Updates

- New section in `docs/getting-started/python-api.md`: "Streaming
  ingest with `OnlineTracker`" with a 30-line worked example
  driving the tracker from `simulate_rat_imu`.
- README "Online filtering" section now distinguishes the *batch
  CLI* from the *Python streaming API*.
- New example script `examples/09_online_streaming.py`.
- Update `docs/TROUBLESHOOTING.md` with a "streaming latency too
  high" section pointing at warmup and `block_until_ready`.

## Open Questions

1. Should `push_camera` accept partial-LED frames (one LED finite,
   the other NaN) the same way the batch path does? Default yes —
   the camera-position model already handles partial frames.
2. Is there value in a `push_imu_batch(t_arr, imu_arr)` for
   replay scenarios? Probably yes; small addition.
3. Should we expose `predict_only(t)` for callers who want a
   prediction at an arbitrary time without a camera frame? PRD
   doesn't require it but it'd be cheap.
4. What's the right behavior when `push_camera(t)` is called with
   no IMU samples buffered? Default is state-mode dependent:
   allow it for `vision_only` because no IMU integration is needed;
   for IMU-fused modes, raise a clear `ValueError` unless the
   implementation explicitly documents and tests a no-IMU prediction
   path. Do not silently reuse the last IMU sample.

## Estimated Effort

- ~250 LOC source + ~300 LOC tests + benchmark + docs.
- 1–2 weeks focused work for one engineer familiar with the EKF
  internals. Most of the time is parity-test debugging at the
  per-camera-frame boundary.
- No new dependencies.
