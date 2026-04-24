"""scripts/bench.py — P-4.1 CLI entry for the unified bench harness.

Invocation:

    # List registered scenarios
    python scripts/bench.py --list

    # Run one scenario, print a summary table, no JSONL
    python scripts/bench.py --scenario qwen3-0.6b-smoke

    # Run every built-in scenario, append JSONL to the given path
    python scripts/bench.py --all --out bench-results.jsonl

    # Also produce a human-readable Markdown report (paste-able into a PR)
    python scripts/bench.py --all \
        --out bench-results.jsonl \
        --report-md bench-results.md

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
    Scenario,
    ScenarioResult,
    get_scenario,
    list_scenario_ids,
    render_markdown_report,
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
    p.add_argument(
        "--report-md",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "write a human-readable Markdown report (GFM table + per-"
            "scenario detail block) to PATH; combine with --out for "
            "both machine-readable JSONL and paste-ready Markdown"
        ),
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0",
        metavar="INT[,INT...]",
        help=(
            "comma-separated RNG seeds to fan every scenario over "
            "(default: 0). Each (scenario, seed) becomes one JSONL "
            "row with metadata.seed=<seed>; aggregation across seeds "
            "is a report-layer concern. Duplicates are dropped with "
            "a stderr warning; each value must satisfy "
            "0 <= seed < 2**32"
        ),
    )
    kv_codec_group = p.add_mutually_exclusive_group()
    kv_codec_group.add_argument(
        "--kv-codec",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "override every scenario's workload.kv_codec with the "
            "given registry id (e.g. 'fp16', 'block_tq_b64_b4'). "
            "Scenarios that pin a specific codec (PPL / STORAGE / "
            "prefix-hit-decode rows) get their baked value replaced; "
            "scenarios without prefix_cache=True will fail the "
            "Workload guard and surface as failed rows. See "
            "`silica.bench.codec_registry.CODEC_REGISTRY` for the "
            "full list of ids"
        ),
    )
    kv_codec_group.add_argument(
        "--all-kv-codecs",
        action="store_true",
        help=(
            "fan every scenario over every codec in "
            "`CODEC_REGISTRY` in sorted order. Equivalent to "
            "iterating --kv-codec over the full registry in one "
            "invocation. Mutually exclusive with --kv-codec "
            "(argparse rejects the combination at parse time). "
            "Each (scenario, codec, seed) becomes one JSONL row; "
            "the report groups rows by (scenario_id, codec_id) so "
            "per-codec mean ± std across seeds is the natural "
            "aggregation"
        ),
    )
    p.add_argument(
        "--vqbench-xcheck",
        action="store_true",
        help=(
            "for every PPL scenario that declares vqbench_xcheck "
            "metadata, invoke the configured vqbench reproduce "
            "script in a subprocess after silica's own oracle "
            "succeeds. Results are flattened into metadata with "
            "``vqbench_`` prefix (ppl_fp16, ppl_quant, delta_ppl, "
            "wall_s, stdout_tail, ...). Step 1 observability "
            "only; divergence gate (|silica - vqbench| / silica "
            "< 0.01) is C.6 step 2"
        ),
    )
    p.add_argument(
        "--vqbench-python",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Python interpreter for the vqbench subprocess. "
            "Defaults to sys.executable, but silica-mlx's venv "
            "does not carry vqbench's torch / transformers deps; "
            "typical usage points this at a dedicated vqbench "
            "venv's python (e.g. "
            "/path/to/vqbench-venv/bin/python). Only meaningful "
            "with --vqbench-xcheck"
        ),
    )
    p.add_argument(
        "--vqbench-epsilon",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help=(
            "divergence gate threshold on |silica.delta_ppl - "
            "vqbench.delta_ppl|. Default 0.01 matches opening "
            "doc P-5 acceptance §7(b). The companion pct "
            "threshold (0.1 percentage points on delta_ppl_pct) "
            "is pinned in the runner per §7(b) and not exposed "
            "here. Only meaningful with --vqbench-xcheck"
        ),
    )
    return p


_SEED_MAX = 1 << 32


def _parse_seeds(raw: str) -> list[int]:
    """Parse ``--seeds VALUE`` into an ordered, deduplicated int list.

    - Splits on ``,``; empty tokens from trailing / duplicate commas are
      an error (surfaces a typo cleanly rather than silently dropping).
    - Each token must parse as ``int`` and land in ``[0, 2**32)`` —
      NumPy's ``seed()`` accepts negative values on some platforms and
      rejects them on others, so we draw the lower bound early.
    - Order-preserving dedup: ``"42,42,43"`` becomes ``[42, 43]``; a
      single warning goes to stderr naming the dropped duplicates.
    - Raises ``ValueError`` on malformed input; callers translate to
      exit code 2 (argparse-usage class).
    """
    tokens = [tok.strip() for tok in raw.split(",")]
    if any(t == "" for t in tokens):
        raise ValueError(
            f"--seeds value {raw!r} contains an empty token "
            f"(leading, trailing, or consecutive ','); use "
            f"'--seeds 42' or '--seeds 42,43', not '--seeds 42,'"
        )
    seeds: list[int] = []
    for tok in tokens:
        try:
            value = int(tok)
        except ValueError:
            raise ValueError(
                f"--seeds token {tok!r} is not an integer"
            ) from None
        if not (0 <= value < _SEED_MAX):
            raise ValueError(
                f"--seeds value {value} is out of range; must "
                f"satisfy 0 <= seed < 2**32 ({_SEED_MAX})"
            )
        seeds.append(value)
    seen: set[int] = set()
    unique: list[int] = []
    dropped: list[int] = []
    for value in seeds:
        if value in seen:
            dropped.append(value)
        else:
            seen.add(value)
            unique.append(value)
    if dropped:
        print(
            f"warning: --seeds dropped duplicate seed(s) "
            f"{sorted(set(dropped))}; running unique set "
            f"{unique}",
            file=sys.stderr,
        )
    return unique


def _print_list() -> None:
    for sid in list_scenario_ids():
        scenario = BUILTIN_SCENARIOS[sid]
        print(f"{sid}\t{scenario.repo}\t{scenario.oracle.value}")


def _print_table(results: Sequence[ScenarioResult]) -> None:
    # Columns pin the (id, codec, seed) triple as the execution
    # identity post-C.5 step 2. Without the ``codec`` cell,
    # ``--all-kv-codecs`` would render K seed-adjacent rows per
    # scenario that only differed in a runner-injected metadata
    # value and were otherwise visually indistinguishable; the
    # terminal stream would force a JSONL cross-reference to
    # disambiguate. ``codec`` goes immediately after ``id`` (and
    # before ``seed``) because codec binds tighter to scenario
    # identity — "this scenario under block_tq_b64_b4" reads
    # naturally, while interleaving codec between seeds would
    # split a natural mental grouping.
    header = (
        "| id | codec | seed | status | reason | ttft_ms | decode_tok_s | "
        "resident_mb | peak_mb | wall_s | tokens |"
    )
    print(header)
    print(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for r in results:
        metadata = r.metadata or {}
        seed = metadata.get("seed", "")
        # ``codec_id`` may be ``None`` (scenarios without
        # workload.kv_codec, no override applied); render the empty
        # string to keep the cell visually consistent with missing
        # numeric values.
        codec_id = metadata.get("codec_id")
        codec_cell = codec_id if codec_id is not None else ""
        print(
            "| {id} | {codec} | {seed} | {status} | {reason} | "
            "{ttft} | {dec} | {res} | {peak} | {wall} | {tok} |".format(
                id=r.scenario_id,
                codec=codec_cell,
                seed=seed,
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


def _dedupe_scenario_ids(selected: list[str]) -> list[str]:
    """Collapse repeated ``--scenario foo`` entries into a single run.

    Parallel to :func:`_parse_seeds`: ``--scenario`` is marked
    repeatable in ``build_parser``, so a pasted shell variable or a
    command-line typo can legitimately produce duplicates. Running
    ``foo`` twice from the same invocation is almost certainly not
    what the user meant — the aggregated report would already
    collapse the duplicate into one table row, and executing it
    twice just burns cycles. Dedupe with a stderr warning so the
    user notices the drop, matching ``_parse_seeds``'s shape
    (never a hard error for a likely typo).
    """
    seen: set[str] = set()
    unique: list[str] = []
    dropped: list[str] = []
    for sid in selected:
        if sid in seen:
            dropped.append(sid)
        else:
            seen.add(sid)
            unique.append(sid)
    if dropped:
        print(
            f"warning: --scenario dropped duplicate id(s) "
            f"{sorted(set(dropped))}; running unique set "
            f"{unique}",
            file=sys.stderr,
        )
    return unique


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

    selected = _dedupe_scenario_ids(selected)

    try:
        scenarios = [get_scenario(sid) for sid in selected]
    except KeyError as exc:
        # get_scenario already builds a message listing known ids;
        # surface it on stderr instead of a raw traceback, return 2
        # (the argparse-usage exit code) so callers can distinguish
        # bad-input from runtime failure.
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2

    try:
        seeds = _parse_seeds(args.seeds)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # --kv-codec / --all-kv-codecs validation + expansion. The two
    # flags are declared mutually exclusive at argparse level, so at
    # most one is set; main() only decides how to build
    # codec_overrides from the winner.
    #
    # Default (neither flag) → (None,): runner uses each scenario's
    # baked workload.kv_codec verbatim, no override applied.
    # --kv-codec X → (X,): single override, validated against
    # CODEC_REGISTRY at CLI time for a CLI-friendly error shape.
    # --all-kv-codecs → list_codec_ids(): iterate every registered
    # codec in alphabetical order. Deliberately excludes the None
    # "baked arm" — --all-kv-codecs represents "run every codec",
    # not "run baked + every codec"; users who want the baked arm
    # plus an override use separate invocations.
    codec_overrides: list[str | None] = [None]
    if args.kv_codec is not None:
        try:
            from silica.bench.codec_registry import get_codec_spec

            get_codec_spec(args.kv_codec)
        except KeyError as exc:
            print(f"error: {exc.args[0]}", file=sys.stderr)
            return 2
        codec_overrides = [args.kv_codec]
    elif args.all_kv_codecs:
        from silica.bench.codec_registry import list_codec_ids

        codec_overrides = list(list_codec_ids())

    # P-5-C.6 step 1: --vqbench-xcheck warn-once. When the flag is
    # set AND the user did not supply --vqbench-python AND at least
    # one selected scenario declares vqbench_xcheck, emit a single
    # stderr warning: sys.executable is almost certainly NOT a
    # vqbench venv (silica-mlx's venv doesn't carry torch /
    # transformers), so the subprocess will fail the reproduce
    # script's imports. Loud-fail behavior stays downstream; this
    # warning is UX nudge, not blocker.
    if (
        args.vqbench_xcheck
        and args.vqbench_python is None
        and any(s.vqbench_xcheck is not None for s in scenarios)
    ):
        print(
            f"warning: --vqbench-xcheck active but --vqbench-python "
            f"not set; falling back to {sys.executable} which "
            f"likely does not carry vqbench's torch / transformers "
            f"deps. Pass --vqbench-python /path/to/vqbench-venv/bin/"
            f"python to silence",
            file=sys.stderr,
        )

    runner = BenchRunner(
        out_path=args.out,
        seeds=seeds,
        codec_overrides=codec_overrides,
        vqbench_xcheck_enabled=args.vqbench_xcheck,
        vqbench_python=args.vqbench_python,
        vqbench_epsilon=args.vqbench_epsilon,
    )
    results = runner.run(scenarios)
    _print_table(results)

    if args.report_md is not None:
        # C.4 step 2: renderer now looks scenarios up by id and
        # aggregates results per scenario, so the CLI no longer
        # needs to expand scenarios to match results length.
        _write_markdown_report(scenarios, results, args.report_md)

    return 1 if any(r.status == "failed" for r in results) else 0


def _write_markdown_report(
    scenarios: Sequence[Scenario],
    results: Sequence[ScenarioResult],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_report(scenarios, results), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
