# PLANNING.md

## 1. Architecture

The package will be structured as a modular, layered system:

```
trodestrack/
  config/          # Configuration schemas, defaults (pydantic)
  io/              # Loaders: trodes, dlc, spikegadgets formats
  geom/            # Homography, arena bounds
  imu/             # Conversions, preprocessing, pre-integration
  models/          # State representation, EKF/UKF, dynamics, measurements
  sim/             # Synthetic data generator (IMU + video) for tests & QA
  runtime/         # Online filter API, offline smoother
  qa/              # Metrics, diagnostics, NEES, plots, tuning
  cli/             # CLI tools: smooth | online | report | calib-homography
  examples/        # Example notebooks, demos
  tests/           # Unit, property, scenario tests
```

**Design principles**

- Separation of concerns (I/O, math, runtime, QA).
- Functional-first design: short, composable functions.
- JAX-first implementation for differentiability and speed.
- Strict typing and configuration validation with Pydantic.
- CLI and Python APIs expose the same core runtime.

---

## 2. Technology Stack

**Language & Frameworks**

- Python ≥ 3.10
- JAX ≥ 0.4.x for numerical kernels and filtering/smoothing (`jax.lax.scan`)

**Core Dependencies**

- `jax`, `numpy`, `scipy`, `chex`, `optax` (math, testing, optimization)
- `pydantic`, `pyyaml` (configuration management)
- `pandas` (optional I/O convenience)
- `matplotlib`, `tqdm` (QA, visualization)

**Code Quality & Testing**

- `pytest` (unit, property, scenario tests)
- `hypothesis` (property-based testing)
- `pytest-benchmark` (performance testing)
- `black`, `mypy`, `ruff` (style, static typing, linting)

**CI/CD**

- GitHub Actions (unit, property, style, type checks, benchmarks)

**Documentation**

- `mkdocs` or `sphinx` for API reference
- Jupyter notebooks in `examples/`

---

## 3. Development Processes & Workflow

**General workflow**

1. Start every new session by reviewing:
   - **PLANNING.md** (this file)
   - **TASKS.md** (open tasks)
   - **SCRATCHPAD.md** (active notes, experiments)
2. Follow **test-driven development (TDD)**:
   - Write failing tests first (unit, property, scenario).
   - Run tests and confirm failure.
   - Implement solution until tests pass.
   - Never modify tests just to make code pass.
3. Commit frequently with small, meaningful changes.
   - Example: `git commit -m "Add imu unit conversion + tests"`
4. Always include:
   - A smoke test for `main()`.
   - Benchmarks for runtime-critical paths.
5. Use **pytest fixtures** to share resources between tests.
6. Keep functions short, modular, and well-documented.

**Collaboration workflow**

- Branch-per-feature workflow (`feature/imu-preintegration`).
- Pull requests require green CI and code review.
- Keep `main` stable and always passing tests.

---

## 4. Required Tools

**Development environment**

- Python ≥ 3.10
- uv for environment management
- JupyterLab for exploration & demos

**Core tools**

- Git + GitHub
- Black (auto-formatting)
- Ruff (linting)
- Pytest (testing)
- Mkdocs/Sphinx (documentation)

**Optional productivity tools**

- Tmux or VSCode devcontainers
- Pre-commit hooks (black, ruff, mypy, pytest --quick)
