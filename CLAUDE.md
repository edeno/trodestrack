# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Package Management & Environment:**

- `uv sync` - Install dependencies and set up virtual environment
- `uv add <package>` - Add a new dependency to the project
- `uv add --dev <package>` - Add a development dependency
- `uv remove <package>` - Remove a dependency

**Running the Package:**

- `uv run trodestrack` - Run the package entry point (requires implementing `main()` function in `src/trodestrack/__init__.py`)
- `uv run python -m trodestrack` - Alternative way to run the package

**Build & Distribution:**

- `uv build` - Build the package for distribution (creates wheel and sdist)

## Project Architecture

**Package Structure:**

- Uses src-layout: `src/trodestrack/__init__.py`
- Entry point configured as `trodestrack = "trodestrack:main"` in pyproject.toml
- Build system uses `hatchling` backend (see pyproject.toml `[build-system]`)

**Python Version:**

- Requires Python >=3.11
- Project configured for Python 3.13 (see .python-version)

**Package Manager:**

- Uses `uv` for dependency management and build system
- Virtual environment automatically managed by `uv`
- Dependencies defined in pyproject.toml

## Development Notes

**Entry Point:**
The package is configured with an entry point `trodestrack:main`, but the `main()` function needs to be implemented in `src/trodestrack/__init__.py`.

**Testing:**

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/trodestrack --cov-report=html

# Run specific test file
uv run pytest tests/sim/test_simple.py -v
```

**Code Quality:**

The project uses pre-commit hooks to enforce code quality standards:

```bash
# Install pre-commit hooks (one-time setup)
uv run pre-commit install

# Run hooks manually on all files
uv run pre-commit run --all-files

# Run hooks on staged files (happens automatically on commit)
git commit -m "your message"
```

**Manual Code Quality Commands:**

```bash
# Type checking
uv run mypy src/trodestrack --ignore-missing-imports

# Linting (with auto-fix)
uv run ruff check src/ tests/ --fix

# Formatting
uv run ruff format src/ tests/

# Format check without modifying files
uv run ruff format --check src/ tests/
```

**Pre-commit Hooks:**

The following hooks run automatically on commit:

- `ruff` - Linting and import sorting (with auto-fix)
- `ruff-format` - Code formatting
- `trailing-whitespace` - Remove trailing whitespace
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax
- `check-added-large-files` - Prevent committing large files
- `check-merge-conflict` - Detect merge conflict markers
- `debug-statements` - Detect debug statements

**Note:** `mypy` is available as a manual check but not enforced in pre-commit hooks to avoid blocking commits on minor type issues in examples and scripts.
