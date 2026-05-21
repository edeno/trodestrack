"""Tests that pin the package version string to pyproject.toml.

A drift between ``trodestrack.__version__`` and the ``[project].version``
field in ``pyproject.toml`` produces misleading ``trodestrack --version``
output and bad bug reports. This regression test fails fast when the two
diverge.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import trodestrack


def test_version_string_matches_pyproject() -> None:
    """``trodestrack.__version__`` must equal pyproject's ``[project].version``."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        pyproject = tomllib.load(fh)

    expected = pyproject["project"]["version"]
    assert trodestrack.__version__ == expected, (
        f"trodestrack.__version__ ({trodestrack.__version__!r}) does not match "
        f"pyproject.toml [project].version ({expected!r}). Update "
        "src/trodestrack/__init__.py to match."
    )
