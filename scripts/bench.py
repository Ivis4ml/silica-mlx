"""scripts/bench.py — P-4.1 CLI entry for the unified bench harness.

Invocation:

    # List registered scenarios
    python scripts/bench.py --list

    # Run one scenario, print a summary table, no JSONL
    python scripts/bench.py --scenario qwen3-0.6b-smoke

    # Run every built-in scenario, append JSONL to the given path
    python scripts/bench.py --all --out bench-results.jsonl

The CLI is intentionally thin: it resolves scenario ids against
``silica.bench.BUILTIN_SCENARIOS`` and hands the Scenario objects to
``BenchRunner``. Anyone adding a new scenario in P-4.2 only touches
``silica/bench/scenarios.py`` and the row shows up in ``--list``
automatically.

Exit codes:
  * 0 — every non-skipped scenario passed its oracle.
  * 1 — at least one scenario failed.
  * 2 — argument parsing rejected the invocation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Ensure the repo root is importable when run as `python scripts/bench.py`
# from any cwd — mirrors the pattern used by every other scripts/*.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from silica.bench import (  # noqa: E402  (path hack above requires late import)
    BUILTIN_SCENARIOS,
    BenchRunner,
    ScenarioResult,
    get_scenario,
    list_scenario_ids,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="silica-bench",
        description=(
            "Run silica-mlx bench scenarios end-to-end (load, "
            "generate, oracle, metrics, JSONL)."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list known scenario ids and exit",
    )
    p.add_argument(
        "--scenario",
        action="append",
        default=[],
        metavar="ID",
        help="scenario id to run (repeatable); see --list for options",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="run every built-in scenario",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="append one JSONL row per result to PATH",
    )
    return p


def _print_list() -> None:
    for sid in list_scenario_ids():
        scenario = BUILTIN_SCENARIOS[sid]
        print(f"{sid}\t{scenario.repo}\t{scenario.oracle.value}")


def _print_table(results: Sequence[ScenarioResult]) -> None:
    header = (
        "| id | status | reason | ttft_ms | decode_tok_s | "
        "resident_mb | peak_mb | wall_s | tokens |"
    )
    print(header)
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(
            "| {id} | {status} | {reason} | {ttft} | {dec} | "
            "{res} | {peak} | {wall} | {tok} |".format(
                id=r.scenario_id,
                status=r.status,
                reason=r.reason or "",
                ttft=_fmt(r.ttft_ms, ".1f"),
                dec=_fmt(r.decode_tok_s, ".1f"),
                res=_fmt(r.resident_mb, ".1f"),
                peak=_fmt(r.peak_memory_mb, ".1f"),
                wall=_fmt(r.wall_s, ".3f"),
                tok=r.total_tokens if r.total_tokens is not None else "",
            )
        )


def _fmt(value: float | None, spec: str) -> str:
    if value is None:
        return ""
    return format(value, spec)


def _resolve_selection(
    args: argparse.Namespace,
) -> list[str]:
    """Expand CLI selection flags into a concrete scenario-id list."""
    if args.all:
        return list_scenario_ids()
    return list(args.scenario)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        _print_list()
        return 0

    selected = _resolve_selection(args)
    if not selected:
        print(
            "error: pass --scenario ID (repeatable), --all, or --list",
            file=sys.stderr,
        )
        return 2

    try:
        scenarios = [get_scenario(sid) for sid in selected]
    except KeyError as exc:
        # get_scenario already builds a message listing known ids;
        # surface it on stderr instead of a raw traceback, return 2
        # (the argparse-usage exit code) so callers can distinguish
        # bad-input from runtime failure.
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2

    runner = BenchRunner(out_path=args.out)
    results = runner.run(scenarios)
    _print_table(results)

    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
