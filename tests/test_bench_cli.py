"""CLI smoke tests for ``scripts/bench.py`` (P-4.1).

Two thin tests guard the entry points a human is most likely to
hit first:

  * ``--list`` — must not require any environment setup beyond a
    normal checkout; should print each known scenario id on its
    own line and exit 0. Regression-tests the repo-root import
    shim (direct ``python scripts/bench.py`` without PYTHONPATH).
  * unknown ``--scenario`` — must surface the invalid id as a
    stderr error line and exit 2 (argparse-usage class), not a
    traceback. Regression-tests the ``KeyError`` handling on
    ``get_scenario``.

Deliberately does NOT exercise ``--scenario qwen3-0.6b-smoke`` —
that path requires the Qwen3-0.6B cache and the real engine, which
is what ``tests/test_bench_runner.py`` covers via fakes and what
``scripts/bench.py --scenario qwen3-0.6b-smoke`` verifies ad-hoc.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "bench.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke ``scripts/bench.py`` with the current interpreter.

    Runs from a neutral cwd (``/``) so the test actually exercises
    the repo-root path-insertion shim instead of accidentally
    picking up ``silica`` from ``cwd == REPO_ROOT``.
    """
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd="/",
        check=False,
    )


def test_cli_list_does_not_require_pythonpath() -> None:
    result = _run_cli("--list")
    assert result.returncode == 0, (
        f"--list exited {result.returncode}: stderr={result.stderr!r}"
    )
    # qwen3-0.6b-smoke is the only P-4.1 scenario; the line format
    # is "<id>\t<repo>\t<oracle>" so we check for the id prefix.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert any(ln.startswith("qwen3-0.6b-smoke\t") for ln in lines), (
        f"expected qwen3-0.6b-smoke in --list output, got: {result.stdout!r}"
    )


def test_cli_unknown_scenario_returns_2_with_stderr() -> None:
    result = _run_cli("--scenario", "does-not-exist")
    assert result.returncode == 2, (
        f"unknown scenario should exit 2, got {result.returncode}; "
        f"stderr={result.stderr!r}"
    )
    # Error goes to stderr, not stdout. Check for the "unknown
    # scenario id" phrase from the KeyError message in scenarios.py.
    assert "unknown scenario id" in result.stderr
    assert "'does-not-exist'" in result.stderr
    # Stable summary-table header must NOT have printed — bad input
    # should short-circuit before touching the runner.
    assert "| id | status |" not in result.stdout
