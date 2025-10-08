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
- Build system uses `uv_build` backend (modern Python packaging)

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

**Adding Testing:**
When adding tests, consider using pytest:

```bash
uv add --dev pytest
# Create tests/ directory or test files following pytest conventions
uv run pytest
```

**Adding Code Quality Tools:**
For linting and formatting, consider:

```bash
uv add --dev ruff  # For linting and formatting
uv add --dev mypy  # For type checking
```
