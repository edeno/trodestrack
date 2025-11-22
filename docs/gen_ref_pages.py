"""Generate the code reference pages and navigation.

This script is run by mkdocs-gen-files to automatically generate
API reference documentation from the source code.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent.parent / "src"

# Modules to document (in order for navigation)
MODULE_ORDER = [
    "trodestrack",
    "trodestrack.models",
    "trodestrack.models.ekf",
    "trodestrack.models.ukf",
    "trodestrack.models.state_layout",
    "trodestrack.models.filter_common",
    "trodestrack.models.filter_update",
    "trodestrack.models.process_noise",
    "trodestrack.models.sensors",
    "trodestrack.sim",
    "trodestrack.sim.simple",
    "trodestrack.sim.rat_imu",
    "trodestrack.sim.utils",
    "trodestrack.runtime",
    "trodestrack.runtime.offline",
    "trodestrack.qa",
    "trodestrack.qa.metrics",
    "trodestrack.qa.plots",
    "trodestrack.qa.report",
    "trodestrack.viz",
    "trodestrack.viz.video",
    "trodestrack.viz.components",
    "trodestrack.viz.utils",
    "trodestrack.viz.styles",
    "trodestrack.cli",
]

for path in sorted(src.rglob("*.py")):
    # Skip private modules and __pycache__
    if any(part.startswith("_") and part != "__init__.py" for part in path.parts):
        continue
    if "__pycache__" in str(path):
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Handle __init__.py files
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    # Skip if no parts (shouldn't happen)
    if not parts:
        continue

    # Build module identifier
    module_ident = ".".join(parts)

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Generate the documentation page
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# {parts[-1]}\n\n")
        fd.write(f"::: {module_ident}\n")

    # Set edit path for GitHub
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src.parent))

# Write the navigation file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
