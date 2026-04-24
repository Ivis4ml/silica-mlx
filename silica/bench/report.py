"""silica.bench.report — render bench results as a Markdown document.

PLAN §P-4 lists "jsonl + markdown report" as a single deliverable;
JSONL ships with :class:`BenchRunner`, and this module supplies the
human-readable half. The output is designed to be pasted directly
into a PR description or a docs/ note — GitHub-flavoured Markdown,
wide but fixed-column GFM table, no ANSI escapes, no emoji.

Input shape:

  * ``scenarios`` — the input list passed to
    :meth:`BenchRunner.run`, same order as the results list. We
    consume the scenario metadata (repo, oracle, description) to
    make the report self-describing; without it the reader would
    need the CLI config to interpret row ids.
  * ``results`` — the list returned by the runner, same order.
    One row becomes one GFM table row plus one detail subsection.

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

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from silica.bench.scenario import Scenario, ScenarioResult


def render_markdown_report(
    scenarios: Sequence[Scenario],
    results: Sequence[ScenarioResult],
    *,
    generated_at: str | None = None,
) -> str:
    """Return a Markdown report covering ``results``.

    ``scenarios`` and ``results`` must be same-length and aligned
    by index (this is how :meth:`BenchRunner.run` returns them).
    ``generated_at`` defaults to ``datetime.now().isoformat(...)``
    when ``None``; tests inject a fixed string for determinism.
    """
    if len(scenarios) != len(results):
        raise ValueError(
            f"scenarios ({len(scenarios)}) and results ({len(results)}) "
            f"must have the same length"
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
    lines.append(_render_summary_line(results))
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.extend(_render_results_table(results))
    lines.append("")
    lines.append("## Scenario details")
    lines.append("")
    for scenario, result in zip(scenarios, results, strict=True):
        lines.extend(_render_scenario_detail(scenario, result))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_summary_line(results: Sequence[ScenarioResult]) -> str:
    # Label is "Runs" rather than "Scenarios" because after
    # P-5-C.4 step 1 each ScenarioResult is one ``(scenario, seed)``
    # execution, so ``len(results)`` is the number of executions —
    # not the number of distinct scenarios selected on the CLI.
    # Step 2 adds cross-seed aggregation; until then, the summary
    # counts executions, not scenarios.
    total = len(results)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return (
        f"Runs: total={total} "
        f"ok={counts['ok']} "
        f"skipped={counts['skipped']} "
        f"failed={counts['failed']}"
    )


def _render_results_table(results: Sequence[ScenarioResult]) -> list[str]:
    header = (
        "| id | status | reason | ttft_ms | decode_tok_s | "
        "resident_mb | peak_mb | wall_s | tokens |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    rows: list[str] = [header, sep]
    for r in results:
        rows.append(
            "| {id} | {status} | {reason} | {ttft} | {dec} | "
            "{res} | {peak} | {wall} | {tok} |".format(
                id=_md_cell(r.scenario_id),
                status=r.status,
                reason=_md_cell(r.reason or ""),
                ttft=_fmt(r.ttft_ms, ".1f"),
                dec=_fmt(r.decode_tok_s, ".1f"),
                res=_fmt(r.resident_mb, ".1f"),
                peak=_fmt(r.peak_memory_mb, ".1f"),
                wall=_fmt(r.wall_s, ".3f"),
                tok=r.total_tokens if r.total_tokens is not None else "",
            )
        )
    return rows


def _render_scenario_detail(
    scenario: Scenario, result: ScenarioResult
) -> list[str]:
    lines: list[str] = [f"### `{scenario.id}`", ""]
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


def _fmt(value: float | None, spec: str) -> str:
    if value is None:
        return ""
    return format(value, spec)


__all__ = ["render_markdown_report"]
