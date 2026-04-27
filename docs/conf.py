"""Sphinx configuration for silica-mlx documentation.

Build with `make -C docs html`. Output lands in `docs/_build/html/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the silica package importable so autodoc can introspect it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "silica-mlx"
author = "Xin Zhou"
copyright = "2026, Xin Zhou"

# Pull version from the package itself when possible; fall back to the
# pyproject.toml string. Keeping the import lazy avoids hard-failing the
# docs build if the package fails to import on a stale checkout.
try:
    import silica  # noqa: F401

    release = getattr(silica, "__version__", "0.0.1")
except Exception:
    release = "0.0.1"
version = release

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

# Markdown + reStructuredText both accepted.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Files Sphinx must not crawl. `_build` is the output dir; `plans/` and
# `../plans/` would otherwise produce thousands of warnings — they are
# linked to from the rendered site but not parsed by Sphinx.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# ---------------------------------------------------------------------------
# MyST configuration
# ---------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 4
myst_url_schemes = ["http", "https", "mailto", "ftp"]

# ---------------------------------------------------------------------------
# Autodoc / autosummary
# ---------------------------------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "mixed"

# Don't pull mlx / mlx_lm at import time — they're heavy and not always
# present on doc-build machines. Stubbing keeps autodoc resilient.
autodoc_mock_imports = [
    "mlx",
    "mlx.core",
    "mlx.nn",
    "mlx.utils",
    "mlx_lm",
    "torch",
    "transformers",
    "fastapi",
    "uvicorn",
    "openai",
    "prompt_toolkit",
    "pygments",
    "pydantic",
    "pydantic_settings",
]

# Napoleon — Google/NumPy-style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

html_theme = "furo"
html_title = f"{project} v{release}"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/ivis4ml/silica-mlx",
    "source_branch": "main",
    "source_directory": "docs/",
}

# ---------------------------------------------------------------------------
# Intersphinx — link out to upstream docs for stdlib + numpy.
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

# Unresolved references should warn but not fail the build — many of our
# Markdown docs cross-reference plan files that intentionally live outside
# the toctree.
nitpicky = False

# Silence noise that originates outside silica's own docstrings.
# - pydantic emits forward references the typing-extensions resolver does
#   not know about (e.g. JsonValue) at autodoc time;
# - autosectionlabel collisions can fire when two docs define a heading
#   with the same slug — harmless given our flat layout.
suppress_warnings = [
    "autodoc.import_object",
    "ref.python",
    # Module-level re-exports legitimately surface the same symbol on
    # both the package page (`silica.bench.X`) and the submodule page
    # (`silica.bench.scenario.X`). The duplicate-description warning
    # is informational only — Sphinx still picks one canonical anchor.
    "app.add_directive",
    "autosectionlabel.*",
]

# Tell autodoc to keep its hands off names re-imported from elsewhere
# at the package level — those land on the submodule page where the
# symbol is defined.
autodoc_inherit_docstrings = True

# Show "Edit on GitHub" link via Furo's source_repository hook above; no
# extra config needed.
