"""silica.bench.vqbench_baseline — subprocess wrapper around vqbench's
reproduce_* scripts to collect a PPL reference column.

PLAN §P-4 names this deliverable: "runs the ready-made vqbench
scripts (``reproduce_qwen35_4b_headline.py`` etc.) in a separate
subprocess to collect PPL as a reference column. D-009 explicitly
allows this 'separate-process comparison' path; it serves the P-5
numeric cross-check acceptance."

The subprocess boundary is deliberate: vqbench ships with
``torch`` + ``transformers`` + ``datasets`` extras (see
``vqbench/pyproject.toml``), which the silica-mlx venv does
**not** depend on — MLX-native is a hard project constraint
(D-009). A dedicated vqbench venv is the expected setup, and its
Python executable is passed via ``python_executable``.

What this module does:

  * Invokes a vqbench reproduce-script in a subprocess with a
    caller-controlled Python executable + script args + timeout.
  * Parses the "Headline table row:" line from stdout (the stable
    format the reproduce scripts emit right before exit) to
    extract ``model``, ``method``, ``bits``, ``ppl_fp16``,
    ``ppl_quant``, ``delta_ppl``, ``delta_pct``.
  * Returns a flat :class:`VqbenchBaselineResult` dataclass so the
    CLI can serialise it to JSONL with a single ``asdict`` call.

What this module deliberately does NOT do:

  * Run vqbench in-process — D-009 forbids importing vqbench from
    silica hot paths. The subprocess boundary is the fix.
  * Orchestrate across multiple reproduce-script invocations —
    the caller drives the sweep; this module produces one row per
    call. Multiple rows → multiple calls, same JSONL file.
  * Integrate with :class:`BenchRunner` — vqbench results have a
    different shape from bench ``ScenarioResult`` rows (PPL vs
    token streams). A unified report view is a later-phase
    concern; for P-4.4 the separate schema keeps the seam tight.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Headline regex: matches the final line
# ``model=... method=... K=... fp16=X.XXXX quant=X.XXXX ΔPPL=+X.XXXX (+X.XX%)``
# emitted by reproduce_qwen35_4b_headline.py and siblings.
#
# The Δ character in ΔPPL is Unicode \u0394; the regex accepts it
# optionally so variants that spell out "delta" (in case future
# scripts do so for stdout-safety) still match.
_HEADLINE_REGEX = re.compile(
    r"model=(\S+)\s+"
    r"method=(\S+)\s+"
    r"K=(\d+)\s+"
    r"fp16=([\d.]+)\s+"
    r"quant=([\d.]+)\s+"
    r"(?:\u0394|delta)?PPL=([+\-]?[\d.]+)\s+"
    r"\(([+\-]?[\d.]+)%\)"
)


@dataclass
class VqbenchBaselineResult:
    """One vqbench reproduce-script invocation outcome.

    ``status`` is one of ``"ok"`` / ``"skipped"`` / ``"failed"``.
    ``skipped`` is reserved for environmentally-blocked runs
    (script missing, python executable missing); ``failed`` covers
    subprocess failures (non-zero exit, timeout, parse failure).

    Parsed numeric fields are ``None`` when parsing did not reach
    them (either the subprocess failed, timed out, or the headline
    row was absent). ``stdout_tail`` is always populated when a
    subprocess actually ran — it carries the last ~30 lines for
    debugging "why did the parse fail" without capturing multi-MB
    logs in JSONL.
    """

    status: str
    reason: str | None = None
    script: str | None = None
    python_executable: str | None = None
    script_args: tuple[str, ...] = ()
    model: str | None = None
    method: str | None = None
    bits: int | None = None
    ppl_fp16: float | None = None
    ppl_quant: float | None = None
    delta_ppl: float | None = None
    delta_pct: float | None = None
    wall_s: float | None = None
    returncode: int | None = None
    stdout_tail: str | None = None


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def parse_headline_row(stdout: str) -> dict[str, Any] | None:
    """Extract the PPL headline fields from a reproduce-script stdout.

    Returns a dict with
    ``{"model", "method", "bits", "ppl_fp16", "ppl_quant",
    "delta_ppl", "delta_pct"}`` when the headline row is found; ``None``
    otherwise. Callers treat ``None`` as a parse failure (the
    script ran but its output shape drifted or completed early).
    """
    for line in stdout.splitlines():
        match = _HEADLINE_REGEX.search(line)
        if match is None:
            continue
        return {
            "model": match.group(1),
            "method": match.group(2),
            "bits": int(match.group(3)),
            "ppl_fp16": float(match.group(4)),
            "ppl_quant": float(match.group(5)),
            "delta_ppl": float(match.group(6)),
            "delta_pct": float(match.group(7)),
        }
    return None


def run_vqbench_baseline(
    script: Path | str,
    script_args: Sequence[str] | None = None,
    *,
    cwd: Path | str | None = None,
    python_executable: str | None = None,
    timeout_s: float = 600.0,
    stdout_tail_lines: int = 30,
    subprocess_runner: SubprocessRunner | None = None,
) -> VqbenchBaselineResult:
    """Run ``script`` in a subprocess; parse its PPL headline.

    Args:
        script: Path to the vqbench reproduce-script (absolute or
            relative to ``cwd``). ``skipped`` if missing.
        script_args: argv suffix to forward to the script
            (``--bits 4``, ``--model <repo>``, etc.).
        cwd: Working directory for the subprocess. Defaults to
            the parent of ``script``.
        python_executable: Interpreter to invoke. Defaults to
            ``sys.executable`` — but silica's venv likely does not
            have vqbench's torch/transformers deps, so real runs
            will point this at a dedicated vqbench venv's Python.
        timeout_s: Subprocess timeout. Defaults to 600 s (the
            reference reproduce_qwen35_4b_headline.py takes ~60 s
            on M5 Pro).
        stdout_tail_lines: Lines of stdout to keep in the result
            for debugging.
        subprocess_runner: Injectable for tests —
            ``subprocess.run``-shaped callable. Default invokes
            :func:`subprocess.run` directly.

    Returns:
        A :class:`VqbenchBaselineResult` with ``status`` set to
        one of ``ok`` / ``skipped`` / ``failed`` plus the parsed
        PPL fields when available.
    """
    # Relative ``script`` semantics: if the caller supplies
    # ``cwd``, resolve the relative path against it (matches the
    # docstring's "relative to cwd" claim); otherwise fall back
    # to the process cwd. Programmatic callers that pass
    # ``cwd=/path/to/vqbench, script="scripts/reproduce_..."``
    # used to land at the wrong absolute path.
    script_path = Path(script)
    if not script_path.is_absolute():
        if cwd is not None:
            script_path = (Path(cwd) / script_path).resolve()
        else:
            script_path = script_path.resolve()
    args_tuple = tuple(script_args or ())
    py = python_executable or sys.executable

    if not script_path.exists():
        return VqbenchBaselineResult(
            status="skipped",
            reason=f"script_not_found:{script_path}",
            script=str(script_path),
            python_executable=py,
            script_args=args_tuple,
        )

    run: SubprocessRunner = subprocess_runner or subprocess.run
    cmd: list[str] = [py, str(script_path), *args_tuple]
    effective_cwd = str(cwd) if cwd is not None else str(script_path.parent)

    t_start = time.perf_counter()
    try:
        proc = run(
            cmd,
            capture_output=True,
            text=True,
            cwd=effective_cwd,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t_start
        return VqbenchBaselineResult(
            status="failed",
            reason=f"subprocess_timeout:{timeout_s}s",
            script=str(script_path),
            python_executable=py,
            script_args=args_tuple,
            wall_s=wall,
        )
    except FileNotFoundError as exc:
        wall = time.perf_counter() - t_start
        return VqbenchBaselineResult(
            status="failed",
            reason=f"python_executable_missing:{exc}",
            script=str(script_path),
            python_executable=py,
            script_args=args_tuple,
            wall_s=wall,
        )

    wall = time.perf_counter() - t_start
    stdout = proc.stdout or ""
    tail_lines = stdout.splitlines()[-stdout_tail_lines:]
    stdout_tail = "\n".join(tail_lines)

    if proc.returncode != 0:
        return VqbenchBaselineResult(
            status="failed",
            reason=f"subprocess_exit:{proc.returncode}",
            script=str(script_path),
            python_executable=py,
            script_args=args_tuple,
            returncode=proc.returncode,
            wall_s=wall,
            stdout_tail=stdout_tail,
        )

    parsed = parse_headline_row(stdout)
    if parsed is None:
        return VqbenchBaselineResult(
            status="failed",
            reason="headline_row_not_found",
            script=str(script_path),
            python_executable=py,
            script_args=args_tuple,
            returncode=proc.returncode,
            wall_s=wall,
            stdout_tail=stdout_tail,
        )

    return VqbenchBaselineResult(
        status="ok",
        script=str(script_path),
        python_executable=py,
        script_args=args_tuple,
        returncode=proc.returncode,
        wall_s=wall,
        stdout_tail=stdout_tail,
        model=parsed["model"],
        method=parsed["method"],
        bits=parsed["bits"],
        ppl_fp16=parsed["ppl_fp16"],
        ppl_quant=parsed["ppl_quant"],
        delta_ppl=parsed["delta_ppl"],
        delta_pct=parsed["delta_pct"],
    )


def default_reproduce_script_path() -> Path:
    """Return the checked-in reproduce script path (relative to
    the silica-mlx checkout root).

    Kept as a module function (not a constant) so the lookup
    happens only when callers actually ask — importing this
    module stays cheap for unit tests that inject fake
    subprocess runners and never touch the filesystem.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "vqbench" / "scripts" / "reproduce_qwen35_4b_headline.py"


__all__ = [
    "SubprocessRunner",
    "VqbenchBaselineResult",
    "default_reproduce_script_path",
    "parse_headline_row",
    "run_vqbench_baseline",
]
