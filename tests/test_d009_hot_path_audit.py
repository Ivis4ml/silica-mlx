"""P5.9 step 2(e) — D-009 hot-path audit lock-in.

D-021 step 2(e) (v1.7.14): regression-locked check verifying no
``torch.Tensor`` / ``numpy.ndarray`` reaches ``silica.engine`` /
``silica.scheduler`` / ``silica.mlx`` / ``silica.kvcache`` /
``silica.models`` / ``silica.vq`` hot path. Pre-step-2(e) this
constraint was enforced only at PR review time; under P-6's
expanding code volume (Track A engine fusion, Track C speculative
variants, Track E streaming) the rule needs an automated gate.

Approach: AST-based import audit on every ``.py`` file under the six
hot-path package roots. The audit rejects any ``import torch`` /
``import numpy`` (and their ``from`` / ``as`` variants) at the
syntactic level — comments and docstrings that mention the names
do not trigger because they are not AST nodes.

Allowlist: a single entry, ``silica/vq/_calibration.py``. PLAN.md
D-009 explicitly permits "numpy as scalar/config/list auxiliary"
in calibration code; the codec ``__init__`` uploads pre-computed
numpy centroids / boundaries to ``mx.array`` once at construction
and never touches numpy again on the per-token forward path. The
file is named ``_calibration`` (underscore prefix → private /
build-time) precisely to mark it as the one D-009-permitted
seam between numpy reference values and the MLX-native runtime.

Adding a new allowlist entry is a deliberate plan-level decision
(the kind that would normally need a new Decisions Log entry); the
test message points future authors at this convention so it is not
relaxed casually.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# The six hot-path packages PLAN.md D-009 names. The audit walks
# every .py file under these roots recursively.
HOT_PATH_PACKAGES: tuple[str, ...] = (
    "engine",
    "scheduler",
    "mlx",
    "kvcache",
    "models",
    "vq",
)

# Disallowed import roots. ``torch.Tensor`` / ``numpy.ndarray`` reach
# the hot path through ``import torch`` / ``import numpy`` (or
# from-imports of either). Catching the import statement is
# sufficient to lock D-009; the only way to use either library
# without an import is via stringly-loaded names which we do not
# do anywhere in silica.
DISALLOWED_ROOTS: frozenset[str] = frozenset({"torch", "numpy"})

# Build-time-only exceptions. Each entry is a path relative to the
# repository root. Adding a new entry should be a deliberate
# plan-level decision; if you find yourself adding an entry, ask
# whether the actual fix is to push the numpy/torch usage out of
# silica's hot path entirely.
#
# Current entries:
#  - silica/vq/_calibration.py: D-009 footnote explicitly permits
#    numpy at codec __init__ time for centroid / boundary upload.
#    The runtime encode / decode bodies are MLX-native.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "silica/vq/_calibration.py",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _silica_root() -> Path:
    return _repo_root() / "silica"


def _hot_path_files() -> list[Path]:
    """All .py files under the six hot-path package roots."""
    files: list[Path] = []
    root = _silica_root()
    for pkg in HOT_PATH_PACKAGES:
        pkg_path = root / pkg
        if not pkg_path.is_dir():
            raise RuntimeError(
                f"hot-path package {pkg!r} not found at {pkg_path}; "
                f"the HOT_PATH_PACKAGES list in this test must match "
                f"the silica/ layout"
            )
        for p in pkg_path.rglob("*.py"):
            files.append(p)
    return files


def _rel_posix(path: Path) -> str:
    """Repo-relative POSIX path, used as the allowlist key + violation
    report. Paths outside the repo (e.g. tmp_path fixtures used by the
    negative-control test) round-trip as their absolute POSIX path —
    those will never match any ALLOWLIST entry, so the audit body
    runs fully on them, which is exactly what the negative control
    needs.
    """
    try:
        return path.relative_to(_repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _violations_for_file(path: Path) -> list[tuple[int, str]]:
    """Return ``[(line, statement_text), ...]`` for any disallowed
    import in ``path``. Empty list means the file is clean.
    """
    rel = _rel_posix(path)
    if rel in ALLOWLIST:
        return []
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover — would fail mypy/ruff first
        raise RuntimeError(
            f"D-009 audit: failed to parse {rel}: {exc}"
        ) from exc
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in DISALLOWED_ROOTS:
                    if alias.asname:
                        stmt = f"import {alias.name} as {alias.asname}"
                    else:
                        stmt = f"import {alias.name}"
                    violations.append((node.lineno, stmt))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            if root in DISALLOWED_ROOTS:
                names = ", ".join(
                    (a.name + (f" as {a.asname}" if a.asname else ""))
                    for a in node.names
                )
                violations.append(
                    (node.lineno, f"from {node.module} import {names}")
                )
    return violations


def test_hot_path_audit_finds_files() -> None:
    """Sanity: the hot-path enumeration actually picks up a non-trivial
    set of files. Without this guard a regression that empties one of
    the package directories (or a mis-typed package name) would let
    the main test pass vacuously.
    """
    files = _hot_path_files()
    # Empirical floor at v1.7.14: 34 files across the six hot-path
    # packages (engine 1 + scheduler 4 + mlx 2 + kvcache 7 + models 11
    # + vq 9). The threshold sits at 30 to give natural headroom for
    # P-6 additions while still failing loudly if a package
    # disappears or the enumeration breaks.
    assert len(files) >= 30, (
        f"D-009 audit picked up {len(files)} hot-path files; "
        f"silica should have at least 30 .py files across the six "
        f"hot-path packages (34 at v1.7.14 landing). Did one of "
        f"the package roots disappear or did the rglob walk break?"
    )


def test_d009_no_torch_numpy_in_hot_path() -> None:
    """D-009 lock-in: no ``torch`` / ``numpy`` imports anywhere under
    the six hot-path packages, except the ALLOWLIST entries.

    Failure message names every offending ``file:line`` plus the
    exact import statement, so a CI or pre-merge run surfaces the
    fix site directly. Adding a new allowlist entry should be a
    deliberate plan-level decision (see ALLOWLIST docstring above);
    if you find yourself wanting to add one, the right question is
    whether the actual fix is to push the numpy / torch usage out
    of silica's hot path entirely (e.g. into a build-time helper
    module like ``silica/vq/_calibration.py``).
    """
    all_violations: list[str] = []
    for f in _hot_path_files():
        for line, stmt in _violations_for_file(f):
            all_violations.append(f"  {_rel_posix(f)}:{line}  {stmt}")

    if all_violations:
        msg = (
            "D-009 hot-path audit found disallowed imports of torch / "
            "numpy in silica's hot-path packages "
            f"({', '.join(HOT_PATH_PACKAGES)}):\n"
            + "\n".join(all_violations)
            + "\n\nD-009 reminder: silica's runtime hot path is "
            "MLX-native — every tensor must be ``mx.array``, every "
            "op must go through MLX. ``torch.Tensor`` / "
            "``numpy.ndarray`` are not allowed in the inference hot "
            "path.\n\nIf this is a build-time-only seam (codec "
            "calibration upload, etc.), the canonical pattern is "
            "to put it in a dedicated ``_calibration`` / build-time "
            "module and add an explicit ALLOWLIST entry in "
            "tests/test_d009_hot_path_audit.py with a comment "
            "naming the D-009 footnote that authorises it.\n\nIf "
            "this is a real hot-path leak, fix the import — that "
            "is the whole point of D-009."
        )
        pytest.fail(msg)


def test_allowlist_entries_actually_exist() -> None:
    """Regression guard: every ALLOWLIST entry must point at a real
    file. A stale entry (e.g. file moved or removed) would silently
    let new violations slip in if a future contributor adds numpy
    back at the same path.
    """
    repo = _repo_root()
    for entry in ALLOWLIST:
        path = repo / entry
        assert path.is_file(), (
            f"ALLOWLIST entry {entry!r} does not point at a real "
            f"file (resolved to {path}); remove the stale entry "
            f"from tests/test_d009_hot_path_audit.py or restore the "
            f"file."
        )


def test_allowlist_only_covers_calibration_seam() -> None:
    """Pin the current allowlist composition. A new ALLOWLIST entry
    should be a deliberate plan-level decision (preferably with a
    Decisions Log entry naming the D-009 footnote it relies on);
    this test forces that decision to be visible — adding an entry
    requires updating this expected-set assertion in the same PR.
    """
    expected = frozenset({"silica/vq/_calibration.py"})
    assert ALLOWLIST == expected, (
        f"ALLOWLIST changed from {sorted(expected)} to "
        f"{sorted(ALLOWLIST)}. Adding a hot-path numpy / torch "
        f"exception is a D-009 scope decision — please make sure "
        f"the change carries a PLAN.md Decisions Log entry (or a "
        f"D-009 footnote update) naming the build-time seam this "
        f"new entry authorises, then update this test to match."
    )


def test_audit_would_fire_on_synthetic_violation(tmp_path: Path) -> None:
    """Negative-control: synthesise a file that imports numpy and
    confirm ``_violations_for_file`` flags it. Pinning this prevents
    a refactor from accidentally turning the audit into a no-op.
    """
    bad = tmp_path / "fake_hot_path.py"
    bad.write_text(
        "import torch\n"
        "import numpy as np\n"
        "from numpy import array\n"
        "from torch.nn import functional as F\n"
        "x = 1\n"
    )
    # ``_violations_for_file`` keys allowlist on repo-relative path,
    # so a tmp_path file is implicitly outside the allowlist.
    violations = _violations_for_file(bad)
    stmts = [v[1] for v in violations]
    assert "import torch" in stmts
    assert "import numpy as np" in stmts
    assert "from numpy import array" in stmts
    assert "from torch.nn import functional as F" in stmts
    assert len(violations) == 4
