"""silica.bench.report — render bench results as a Markdown document.

PLAN §P-4 lists "jsonl + markdown report" as a single deliverable;
JSONL ships with :class:`BenchRunner`, and this module supplies the
human-readable half. The output is designed to be pasted directly
into a PR description or a docs/ note — GitHub-flavoured Markdown,
wide but fixed-column GFM table, no ANSI escapes, no emoji.

P-5-C.4 step 2 shape:

Under multi-seed fan-out (P-5-C.4 step 1 added ``--seeds 42,43,44``
to the CLI) the runner produces one :class:`ScenarioResult` per
``(scenario, seed)`` execution, so a ``--scenario A --scenario B
--seeds 42,43`` invocation yields four results. The renderer splits
that into two audiences:

  * **Aggregated table** — one row per *scenario* (not per
    execution). Numeric columns show ``mean ± std`` across the
    scenario's seeds; deterministic quantities (std == 0) collapse
    to a single value. Status counts (ok / skipped / failed) sum
    per-execution so ``ok + skipped + failed`` equals ``runs`` on
    that row.
  * **Per-execution detail** — one subsection per raw result,
    heading suffixed ``(seed=<N>)``. Preserves everything a reader
    needed from the old one-row-per-result shape (repo, oracle,
    gate, metadata blob) without forcing that granularity into the
    table.

Input shape:

  * ``scenarios`` — the unique scenario list the CLI resolved from
    ``--scenario`` / ``--all``. Each scenario appears once
    regardless of how many seeds ran against it. The renderer looks
    up scenario metadata (repo, oracle, description) by id, so
    caller order is preserved in the table.
  * ``results`` — every ``ScenarioResult`` the runner produced, in
    scenario-major order (``A-seed0, A-seed1, ..., B-seed0, ...``).
    One row becomes one detail subsection.

Input invariants (caller bugs, not user errors):

  * Every result's ``scenario_id`` must appear in ``scenarios``.
  * Every scenario in ``scenarios`` must appear in at least one
    result. A scenario with zero results indicates the runner
    dropped it silently — surface that loudly rather than rendering
    an empty row.

Deliberately not included in v0.1:

  * Per-column sorting / filtering — the report is a snapshot; if
    a reader wants to slice, the JSONL is next to the ``.md``.
  * Git SHA / uname / env-var stamping — bench reproducibility
    metadata is a P-4.4 concern (vqbench_baseline prerequisite),
    out of scope for the v0.1 report-file requirement.
  * Per-oracle custom metadata rendering — the catch-all
    "Metadata" block dumps the dict; structure-aware rendering
    (e.g. collapsed row tables for BGT1 parity) can land when the
    unstructured dump proves insufficient in practice.

The renderer is a pure function: no file I/O, no clock reads
unless the caller leaves ``generated_at=None``. This keeps unit
tests deterministic and makes the CLI shim's file-write path
a one-liner.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from silica.bench.scenario import Scenario, ScenarioResult

# Numeric ScenarioResult fields that flow into the aggregated
# table. Keys are ScenarioResult attribute names; values are the
# column name shown in the Markdown header. Ordered so the table
# layout tracks the ScenarioResult schema — changing this order
# requires updating the header and row-format functions together.
_NUMERIC_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("ttft_ms", "ttft_ms", ".1f"),
    ("decode_tok_s", "decode_tok_s", ".1f"),
    ("resident_mb", "resident_mb", ".1f"),
    ("peak_memory_mb", "peak_mb", ".1f"),
    ("wall_s", "wall_s", ".3f"),
)

# total_tokens is an int but still gets aggregated because multi-
# seed runs may produce different token counts when an oracle
# consumes RNG. Formatted with ``.0f`` so deterministic rows
# render as ``4`` rather than ``4.0``.
_TOKENS_COLUMN = ("total_tokens", "tokens", ".0f")


@dataclass
class _ScenarioAggregate:
    """Per-scenario aggregation buffer.

    Built in a single pass over ``results``; consumed once by the
    table renderer. ``runs`` is the number of ``ScenarioResult``
    rows seen (including skipped / failed); ``ok``/``skipped``/
    ``failed`` are per-execution status counts that must sum to
    ``runs``. Numeric lists exclude ``None`` values so a mix of
    skipped + ok rows aggregates cleanly — skipped rows contribute
    to status counts but not to mean / std.
    """

    runs: int = 0
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    numeric: dict[str, list[float]] = field(default_factory=dict)


def _aggregate_by_scenario(
    results: Sequence[ScenarioResult],
) -> dict[str, _ScenarioAggregate]:
    """Bucket ``results`` by ``scenario_id`` for table rendering.

    Preserves first-seen order of scenario ids via ``dict`` insertion
    order — callers that iterate the returned mapping in insertion
    order walk scenarios in the same sequence the results came in,
    which under scenario-major runner output equals the CLI selection
    order.
    """
    buckets: dict[str, _ScenarioAggregate] = {}
    for r in results:
        bucket = buckets.get(r.scenario_id)
        if bucket is None:
            bucket = _ScenarioAggregate()
            buckets[r.scenario_id] = bucket
        bucket.runs += 1
        if r.status == "ok":
            bucket.ok += 1
        elif r.status == "skipped":
            bucket.skipped += 1
        elif r.status == "failed":
            bucket.failed += 1

        for attr, _col, _spec in (*_NUMERIC_COLUMNS, _TOKENS_COLUMN):
            value = getattr(r, attr, None)
            if value is None:
                continue
            bucket.numeric.setdefault(attr, []).append(float(value))
    return buckets


def _format_mean_std(values: Sequence[float], spec: str) -> str:
    """Render a numeric column cell.

    - Empty list → ``""`` (all runs had the field as ``None``;
      cell stays blank, matching pre-C.4 skipped-row behaviour).
    - Single value → format that value alone. Covers N=1 and
      the ``seeds=(0,)`` default path so the cell does not
      gain a spurious ``± 0.000`` suffix.
    - std == 0 (deterministic across seeds) → format the mean
      alone. Common case for ``total_tokens`` when the oracle
      does not consume RNG.
    - Otherwise → ``"<mean> ± <std>"`` using sample std
      (``statistics.stdev`` divides by N-1, the uncertainty-of-
      estimate convention a reader of mean ± std expects).
    """
    if not values:
        return ""
    if len(values) == 1:
        return format(values[0], spec)
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return format(mean, spec)
    return f"{format(mean, spec)} ± {format(std, spec)}"


def render_markdown_report(
    scenarios: Sequence[Scenario],
    results: Sequence[ScenarioResult],
    *,
    generated_at: str | None = None,
) -> str:
    """Return a Markdown report covering ``results``.

    ``scenarios`` is the unique scenario list; ``results`` is the
    full per-execution list (one row per ``(scenario, seed)``
    under multi-seed fan-out). ``generated_at`` defaults to
    ``datetime.now().isoformat(...)`` when ``None``; tests inject a
    fixed string for determinism.

    Raises ``ValueError`` if:

      * ``scenarios`` contains duplicate ids — the aggregated table
        iterates ``scenarios`` in order, so a duplicate would
        produce two identical rows pointing at the same aggregate
        bucket, and the summary's ``Scenarios: total=`` would be
        inflated by the duplicate selection. Callers (including
        the CLI) must dedupe upstream; this renderer check is
        defense-in-depth for direct programmatic callers
      * a result references a scenario id not in ``scenarios``
        (unknown id — caller passed inconsistent lists)
      * a scenario in ``scenarios`` has zero results (runner
        dropped it silently — suppressing this would hide a bug)
    """
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for s in scenarios:
        if s.id in seen_ids:
            duplicate_ids.append(s.id)
        else:
            seen_ids.add(s.id)
    if duplicate_ids:
        raise ValueError(
            f"render_markdown_report: scenarios list contains "
            f"duplicate id(s) {sorted(set(duplicate_ids))!r}; the "
            f"aggregated table is keyed by scenario_id, so "
            f"duplicates would render as identical rows pointing "
            f"at the same bucket. CLI callers should dedupe "
            f"upstream (see scripts/bench.py:_dedupe_scenario_ids)"
        )
    scenario_by_id = {s.id: s for s in scenarios}
    unknown_ids = [
        r.scenario_id for r in results if r.scenario_id not in scenario_by_id
    ]
    if unknown_ids:
        raise ValueError(
            f"render_markdown_report: results reference unknown "
            f"scenario id(s) {sorted(set(unknown_ids))!r}; every "
            f"result.scenario_id must appear in the scenarios list"
        )
    aggregates = _aggregate_by_scenario(results)
    missing = [s.id for s in scenarios if s.id not in aggregates]
    if missing:
        raise ValueError(
            f"render_markdown_report: scenarios {missing!r} have "
            f"zero results — the runner dropped them silently or "
            f"the caller passed an expanded scenarios list that "
            f"does not appear in results"
        )
    timestamp = (
        generated_at
        if generated_at is not None
        else datetime.now().isoformat(timespec="seconds")
    )

    lines: list[str] = []
    lines.append("# silica-mlx bench report")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    lines.append(_render_summary_line(scenarios, results))
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.extend(_render_aggregated_table(scenarios, aggregates))
    lines.append("")
    lines.append("## Scenario details")
    lines.append("")
    for result in results:
        scenario = scenario_by_id[result.scenario_id]
        lines.extend(_render_scenario_detail(scenario, result))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_summary_line(
    scenarios: Sequence[Scenario], results: Sequence[ScenarioResult]
) -> str:
    # Status counts sum per-execution, not per-scenario: under
    # fan-out a single scenario may have mixed-status seeds, and
    # collapsing that to one status bucket per scenario would
    # require inventing a "mixed" bucket or picking strictest-
    # wins. Summing executions keeps the arithmetic clean —
    # ``ok + skipped + failed == runs`` — and the aggregated
    # table shows per-scenario breakdowns so the information
    # is not lost.
    total_runs = len(results)
    total_scenarios = len(scenarios)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return (
        f"Scenarios: total={total_scenarios} "
        f"Runs: total={total_runs} "
        f"ok={counts['ok']} "
        f"skipped={counts['skipped']} "
        f"failed={counts['failed']}"
    )


def _render_aggregated_table(
    scenarios: Sequence[Scenario],
    aggregates: dict[str, _ScenarioAggregate],
) -> list[str]:
    # Column set: id + per-run status breakdown + numeric columns
    # aggregated with mean ± std. Reason column is deliberately not
    # included — under fan-out different seeds can fail with
    # different reasons, and either "varied" or a pick-one policy
    # would be a lossy aggregation. Reasons live in the per-
    # execution detail section instead.
    header_cells = [
        "id",
        "runs",
        "ok",
        "skipped",
        "failed",
        *(col for _attr, col, _spec in _NUMERIC_COLUMNS),
        _TOKENS_COLUMN[1],
    ]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "| " + " | ".join("---" for _ in header_cells) + " |"
    rows: list[str] = [header, sep]
    for scenario in scenarios:
        agg = aggregates[scenario.id]
        cells = [
            _md_cell(scenario.id),
            str(agg.runs),
            str(agg.ok),
            str(agg.skipped),
            str(agg.failed),
        ]
        for attr, _col, spec in _NUMERIC_COLUMNS:
            cells.append(
                _format_mean_std(agg.numeric.get(attr, []), spec)
            )
        cells.append(
            _format_mean_std(
                agg.numeric.get(_TOKENS_COLUMN[0], []), _TOKENS_COLUMN[2]
            )
        )
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _render_scenario_detail(
    scenario: Scenario, result: ScenarioResult
) -> list[str]:
    # Detail heading suffixed with (seed=<N>) so under multi-seed
    # fan-out each execution has a unique section. Missing seed
    # (metadata lacks the key) falls back to the plain id — keeps
    # the renderer robust against pre-C.4 callers / manual test
    # fixtures that construct ScenarioResult without going through
    # BenchRunner.
    seed = result.metadata.get("seed") if result.metadata else None
    heading_suffix = f" (seed={seed})" if seed is not None else ""
    lines: list[str] = [
        f"### `{scenario.id}`{heading_suffix}",
        "",
    ]
    lines.append(f"- repo: `{scenario.repo}`")
    lines.append(f"- oracle: `{scenario.oracle.value}`")
    gate = scenario.gate_env_var or "(cache-only)"
    lines.append(f"- gate: `{gate}`")
    lines.append(
        f"- workload: `max_batch_size={scenario.workload.max_batch_size}`, "
        f"`max_tokens={scenario.workload.max_tokens}`, "
        f"`prompts={len(scenario.workload.prompts)}`"
    )
    lines.append(f"- status: **{result.status}**")
    if result.reason:
        lines.append(f"- reason: `{_md_cell(result.reason)}`")
    if scenario.description:
        lines.append("")
        lines.append(scenario.description)
    if result.metadata:
        lines.append("")
        lines.append("Metadata:")
        lines.append("")
        lines.append("```")
        lines.extend(_render_metadata_block(result.metadata))
        lines.append("```")
    return lines


def _render_metadata_block(metadata: dict[str, Any]) -> list[str]:
    import json

    out = json.dumps(metadata, indent=2, sort_keys=True, default=str)
    return out.splitlines()


def _md_cell(value: str) -> str:
    """Escape characters that would break a GFM table row."""
    return value.replace("|", "\\|").replace("\n", " ")


__all__ = ["render_markdown_report"]
