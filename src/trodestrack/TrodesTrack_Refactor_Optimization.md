
# TrodesTrack Refactor & Optimization Plan

## TL;DR (land these first)

1. **Single scan, single JIT**: Wrap the whole timestep loop in `lax.scan`; precompute device-side indices and measurement models.
2. **Unify Q-building** across filters & smoothers (`assemble_Q(...)`), including `G @ Q_u @ Gᵀ`, blackout scaling, and optional bias freeze.
3. **Joseph form everywhere** for all measurement paths (pos 2D/4D, heading 1D, ZUPT; EKF & UKF).
4. **Branchless robustness**: min-NIS LED swap chooser, gating, and ZUPT as measurement—no Python branching.
6. **Device-friendly precompute**: Build `H/R/innov/selectors/confidences` ahead of the scan; keep shapes static.
7. **Dim-agnostic helpers** for selectors and Q, removing “magic 8s”.

---

## Project Structure (proposed)

```
trodestrack/
  models/
    filter_common.py        # shared math (linalg, joseph, psd_solve, angle_wrap)
    measurement.py          # MeasurementModel, ekf_update, ukf_update, gaussian_loglik, gating
    process_noise.py        # Q_rate, input_noise_cov, G(theta,dt), assemble_Q(...)
    association.py          # LED swap resolution (min-NIS), constants (SWAP4)
    zupt.py                 # ZUPT model + branchless apply
    ekf.py                  # dynamics, predict, calls shared update engine
    ukf.py                  # sigma-point plumbing, calls shared update engine
  runtime/
    offline.py              # smoothers (use assemble_Q, dim-agnostic)
```

---

## Key APIs (drop-in ready)

### 1) Measurement model & EKF/UKF updates

```python
# measurement.py
from dataclasses import dataclass

@dataclass(frozen=True)
class MeasurementModel:
    H: jnp.ndarray      # (m, n)
    R: jnp.ndarray      # (m, m)
    innov: jnp.ndarray  # (m,)

@dataclass(frozen=True)
class UpdateStats:
    K: jnp.ndarray
    nis: jnp.ndarray
    loglik: jnp.ndarray
    accepted: jnp.ndarray  # after gating

def joseph_update(P, K, H, R):
    I = jnp.eye(P.shape[0], dtype=P.dtype)
    A = I - K @ H
    return (A @ P @ A.T) + (K @ R @ K.T)

def mahal2(innov, S):
    # Reuse psd_solve from filter_common
    x = psd_solve(S, innov)
    return innov @ x

def chi2_threshold(dof: int, prob: float) -> float:
    CHI2 = {(2, .95): 5.991, (2, .99): 9.210, (2, .997): 11.618,
            (4, .95): 9.488, (4, .99): 13.277, (4, .997): 16.014}
    return CHI2[(dof, prob)]

def gaussian_loglik(innov, S, eps=1e-9):
    S = 0.5*(S + S.T) + eps * jnp.eye(S.shape[0], S.dtype)
    sign, logdet = jnp.linalg.slogdet(S)
    # If sign<0, bump eps (branchless)
    bump = (sign <= 0).astype(S.dtype) * (1e-6)
    S = 0.5*(S + S.T) + bump * jnp.eye(S.shape[0], S.dtype)
    mah = mahal2(innov, S)
    sign, logdet = jnp.linalg.slogdet(S)
    return -0.5 * (mah + logdet + S.shape[0]*jnp.log(2*jnp.pi))

def ekf_update(mean, cov, model: MeasurementModel, *, gate_prob=0.997, use_joseph=True):
    H, R, y = model.H, model.R, model.innov
    S = H @ cov @ H.T + R
    K = cov @ H.T @ jnp.linalg.inv(S)  # or psd_solve with matmul
    dof = y.shape[0]
    nis = mahal2(y, S)
    thr = chi2_threshold(dof, gate_prob)
    accept = nis <= thr

    # Branchless gating: if reject, set K=0 and R=0 contribution
    K_eff = jnp.where(accept, K, jnp.zeros_like(K))
    cov_upd = joseph_update(cov, K_eff, H, R) if use_joseph else (cov - K_eff @ S @ K_eff.T)
    mean_upd = mean + K_eff @ y
    loglik = gaussian_loglik(y, S)
    return mean_upd, cov_upd, UpdateStats(K=K_eff, nis=nis, loglik=loglik, accepted=accept)
```

### 2) Process noise (shared by forward filters and smoothers)

```python
# process_noise.py
def build_Q_rate(cfg, n: int):
    if n == 8:
        return jnp.diag(jnp.array([
            cfg.process_noise_pos, cfg.process_noise_pos,
            cfg.process_noise_vel, cfg.process_noise_vel,
            cfg.process_noise_heading,
            cfg.process_noise_gyro_bias,
            cfg.process_noise_accel_bias, cfg.process_noise_accel_bias
        ], dtype=jnp.float32))
    return jnp.eye(n, dtype=jnp.float32) * cfg.process_noise_pos

def build_input_noise_cov(cfg, dt: float):
    # gyro_density [rad/s/√Hz], accel_density [m/s²/√Hz] → per-sample std
    sg = (cfg.gyro_noise_density * jnp.sqrt(dt))**2
    sa = (cfg.accel_noise_density * jnp.sqrt(dt))**2
    # Map to (gz, ax, ay) ordering
    return jnp.diag(jnp.array([sg, sa, sa], dtype=jnp.float32))

def build_G(theta: float, dt: float, n: int):
    # Map IMU noise into state (velocity/heading/biases)
    # Minimal 8D example; extend as needed for 3D
    c, s = jnp.cos(theta), jnp.sin(theta)
    G = jnp.zeros((n, 3), dtype=jnp.float32)
    # ax, ay affect vx, vy via rotation
    G = G.at[2, 1].set(dt * c).at[2, 2].set(-dt * s)
    G = G.at[3, 1].set(dt * s).at[3, 2].set(dt * c)
    # gz affects heading
    G = G.at[4, 0].set(dt)
    return G

def assemble_Q(cfg, theta, dt, n, *, in_blackout, freeze_bias):
    Q = build_Q_rate(cfg, n) * dt
    Qu = build_input_noise_cov(cfg, dt)
    G  = build_G(theta, dt, n)
    Q  = 0.5*(Q + Q.T) + G @ Qu @ G.T
    # Blackout scaling / bias freeze
    if in_blackout:
        Q = Q * cfg.blackout_q_scale
        if freeze_bias:
            # zero out bias rows/cols (indices depend on state layout)
            bias_idx = jnp.array([5, 6, 7])
            Q = Q.at[bias_idx, :].set(0.0).at[:, bias_idx].set(0.0)
    return 0.5*(Q + Q.T)
```

### 3) LED swap resolution (min-NIS, branchless)

```python
# association.py
SWAP4 = jnp.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=jnp.float32)

def choose_led_assignment(model4: MeasurementModel, cov):
    H, R, y = model4.H, model4.R, model4.innov
    S = H @ cov @ H.T + R
    nis_a = mahal2(y, S)
    # swapped hypothesis
    Hs = SWAP4 @ H
    Rs = SWAP4 @ R @ SWAP4.T
    ys = SWAP4 @ y
    Ss = Hs @ cov @ Hs.T + Rs
    nis_b = mahal2(ys, Ss)
    take_b = nis_b < (nis_a - 1e-6)
    Hf = jnp.where(take_b, Hs, H)
    Rf = jnp.where(take_b, Rs, R)
    yf = jnp.where(take_b, ys, y)
    return MeasurementModel(Hf, Rf, yf), take_b
```

### 4) ZUPT as a measurement (branchless)

```python
# zupt.py
def H_vel(n: int):
    # select vx, vy rows
    H = jnp.zeros((2, n), dtype=jnp.float32)
    H = H.at[0, 2].set(1.0).at[1, 3].set(1.0)
    return H

def zupt_model(cfg, state_mean, n: int):
    H = H_vel(n)
    y = - (H @ state_mean)   # target zero velocity
    R = jnp.eye(2, dtype=jnp.float32) * cfg.zupt_noise
    return MeasurementModel(H=H, R=R, innov=y)
```

---

## Scan-based runner (skeleton)

```python
@jax.jit
def run_filter(init_state, imu, cam_models, idx_start, idx_end, cfg):
    def step(state, i):
        # integrate IMU between frames
        def f(s, k):
            return predict_step(s, imu[k], cfg), None
        s, _ = lax.scan(f, state, jnp.arange(idx_start[i], idx_end[i]))

        # position update (2D/4D) with optional LED swap + gating
        meas = cam_models[i]              # prebuilt MeasurementModel
        if meas.H.shape[0] == 4:
            meas, _ = choose_led_assignment(meas, s.cov)
        s.mean, s.cov, _ = ekf_update(s.mean, s.cov, meas, gate_prob=cfg.mahal_prob)

        # heading pseudo-measurement (if valid)
        h_meas = heading_models[i]        # or encode validity inside R
        s.mean, s.cov, _ = ekf_update(s.mean, s.cov, h_meas, gate_prob=cfg.mahal_prob)

        # ZUPT
        z_meas = zupt_model(cfg, s.mean, s.cov.shape[0])
        s.mean, s.cov, _ = ekf_update(s.mean, s.cov, z_meas, gate_prob=1.0)

        return s, s
    final, traj = lax.scan(step, init_state, jnp.arange(cam_models.shape[0]))
    return final, traj
```

---

## Acceptance Checklist

- [ ] Full filter wrapped in a single `lax.scan` (no Python loops inside the JIT).
- [ ] `assemble_Q(...)` used in **both** forward filters and **both** smoothers.
- [ ] Joseph-form covariance update in **all** measurement paths.
- [ ] Branchless LED swap resolution & gating.
- [ ] Confidence→R scaling centralized and shared with sims.
- [ ] Dim-agnostic state helpers (selectors, Q, G) with 8D fast path.
- [ ] UKF presets documented (conservative vs aggressive).

## CI / Testing

- Default CI: `pytest -m "not slow"`
- Slow suite nightly: dropout PRD (§4.2), Monte Carlo NEES calibration.
- Add regression tests for swap chooser, exposure vs arrival, only-IMU/only-cam modes.

---

## Notes

- Keep `float32` unless a specific metric fails; centralize `EPS` (1e-9) for innovation jitter.
- Re-export `dynamics_function` from `runtime.offline` for patchable tests.
- Emit a `RunSummary` with gate counts, swap flags, ZUPT activations, blackout windows, and estimated LED spacing.
