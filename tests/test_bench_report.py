"""Unit tests for ``silica.bench.report.render_markdown_report``.

The report is a pure rendering function — no I/O, no clock unless
``generated_at`` is omitted — so every test constructs a small
scenarios/results pair and inspects the returned string.

P-5-C.4 step 2 changed the renderer's shape:

  * Table aggregates per-scenario (one row regardless of seed count).
  * Numeric columns show ``mean ± std`` across a scenario's seeds;
    single-value / deterministic rows collapse to just the value.
  * Reason column moved to the per-execution detail section.
  * ``(scenario, seed)`` detail headings include the seed suffix.
  * Length-equality invariant replaced with id-based invariants.

Structural assertions only: header present, table rows count
correctly, counts line reflects statuses, aggregation math matches.
Full rendering fidelity is not pinned character-for-character
because the format will keep evolving (aggregation policies,
vqbench PPL reference column, oracle-specific metadata tables).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from silica.bench import (
    OracleKind,
    Scenario,
    ScenarioResult,
    Workload,
    render_markdown_report,
)


def _count_unescaped_pipes(line: str) -> int:
    """Count GFM column-delimiter pipes, ignoring backslash-escaped ones."""
    return len(re.findall(r"(?<!\\)\|", line))


def _scenario(
    scenario_id: str,
    repo: str = "demo/owner",
    oracle: OracleKind = OracleKind.SMOKE,
    gate_env_var: str | None = None,
    description: str = "",
) -> Scenario:
    return Scenario(
        id=scenario_id,
        repo=repo,
        workload=Workload(
            name="w", prompts=("p",), max_tokens=4, max_batch_size=1
        ),
        oracle=oracle,
        gate_env_var=gate_env_var,
        description=description,
    )


def _result(
    scenario_id: str,
    *,
    status: str = "ok",
    reason: str | None = None,
    metadata: dict | None = None,
    seed: int | None = 0,
    **metrics: Any,
) -> ScenarioResult:
    """Build a ``ScenarioResult`` with ``metadata["seed"]`` already
    set so the detail heading suffix ``(seed=<N>)`` appears naturally.

    Pre-C.4 test vectors that omitted seed now implicitly get
    ``seed=0`` which matches the runner's default behaviour. Tests
    exercising seed=None behaviour pass ``seed=None`` explicitly;
    the merge below keeps any caller-provided metadata dict intact
    while injecting the seed under the expected key.
    """
    md = dict(metadata or {})
    if seed is not None and "seed" not in md:
        md["seed"] = seed
    return ScenarioResult(
        scenario_id=scenario_id,
        status=status,
        reason=reason,
        metadata=md,
        **metrics,
    )


# ----- Header / timestamp --------------------------------------------


def test_report_has_header_and_fixed_timestamp() -> None:
    scenarios = [_scenario("one")]
    results = [_result("one", ttft_ms=10.0, total_tokens=4, wall_s=0.5)]
    out = render_markdown_report(
        scenarios, results, generated_at="2026-04-20T12:34:56"
    )
    assert out.splitlines()[0] == "# silica-mlx bench report"
    assert "Generated: 2026-04-20T12:34:56" in out


# ----- Summary line --------------------------------------------------


def test_report_summary_counts_statuses() -> None:
    """Status counts sum per-execution (post-C.4 step 2 semantic).

    With 4 scenarios × 1 seed each, ``Runs: total=4`` equals
    ``Scenarios: total=4`` — the fan-out is degenerate at N=1.
    Multi-seed tests below exercise the divergence explicitly.
    """
    scenarios = [_scenario(sid) for sid in ("a", "b", "c", "d")]
    results = [
        _result("a", status="ok"),
        _result("b", status="ok"),
        _result("c", status="skipped", reason="cache_missing:/x"),
        _result("d", status="failed", reason="RuntimeError: boom"),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "Scenarios: total=4" in out
    assert "Runs: total=4" in out
    assert "ok=2" in out
    assert "skipped=1" in out
    assert "failed=1" in out


def test_report_summary_counts_scenarios_vs_runs_separately() -> None:
    """Two scenarios × three seeds = 2 Scenarios, 6 Runs. The
    summary line must surface both counts; collapsing them would
    hide fan-out from a PR reviewer."""
    scenarios = [_scenario("a"), _scenario("b")]
    results = [
        _result("a", seed=0),
        _result("a", seed=1),
        _result("a", seed=2),
        _result("b", seed=0),
        _result("b", seed=1),
        _result("b", seed=2),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "Scenarios: total=2" in out
    assert "Runs: total=6" in out
    assert "ok=6" in out


def test_report_summary_status_counts_sum_per_execution() -> None:
    """Within a single scenario, mixed-status seeds surface as
    per-execution counts that sum to the run count. A naive
    per-scenario rollup ("1 failed scenario") would lose the
    information that 2 out of 3 seeds passed."""
    scenarios = [_scenario("mixed")]
    results = [
        _result("mixed", seed=0, status="ok"),
        _result("mixed", seed=1, status="ok"),
        _result("mixed", seed=2, status="failed", reason="RuntimeError"),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "Runs: total=3" in out
    assert "ok=2" in out
    assert "failed=1" in out


# ----- Aggregated table ---------------------------------------------


def test_report_table_has_row_per_scenario() -> None:
    """Post-C.4 step 2: table aggregates per-scenario. Two scenarios
    → two data rows regardless of how many seeds ran."""
    scenarios = [_scenario("one"), _scenario("two")]
    results = [
        _result("one", seed=0, ttft_ms=11.1, decode_tok_s=222.2, total_tokens=4),
        _result("one", seed=1, ttft_ms=12.3, decode_tok_s=220.0, total_tokens=4),
        _result(
            "two",
            seed=0,
            status="skipped",
            reason="env_var_not_set:DEMO_GATE",
        ),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    table_start = out.index("## Results")
    detail_start = out.index("## Scenario details")
    table_block = out[table_start:detail_start]
    lines = [ln for ln in table_block.splitlines() if ln.startswith("|")]
    # header + separator + 2 data rows = 4
    assert len(lines) == 4
    assert "| one |" in table_block
    assert "| two |" in table_block


def test_report_table_row_preserves_scenarios_input_order() -> None:
    """Row order follows the ``scenarios`` argument order, not
    whatever order scenarios first appeared in the results list.
    The CLI passes scenarios in user-selection order (repeatable
    ``--scenario`` flags), which the report must preserve."""
    scenarios = [_scenario("second"), _scenario("first")]
    results = [
        _result("first", seed=0),  # appears first in results
        _result("second", seed=0),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    table_start = out.index("## Results")
    detail_start = out.index("## Scenario details")
    table_block = out[table_start:detail_start]
    # The data rows start with the scenario id in the first cell;
    # "second" must appear before "first" because that is the
    # scenarios-argument order.
    second_idx = table_block.index("| second |")
    first_idx = table_block.index("| first |")
    assert second_idx < first_idx


def test_report_table_header_has_aggregate_columns() -> None:
    """Post-C.5 step 2 column set: id + codec + runs +
    ok/skipped/failed + numeric metrics + tokens. Reason column
    is gone (moved to detail)."""
    scenarios = [_scenario("one")]
    results = [_result("one")]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "| id | codec | runs | ok | skipped | failed |" in out
    # reason column must NOT be in the aggregated table
    assert (
        "| id | codec | runs | ok | skipped | failed | reason |"
        not in out
    )


def test_report_table_runs_column_reports_count() -> None:
    """The ``runs`` column on each row equals how many ScenarioResults
    had that (scenario, codec) key. A drift here would mean the
    aggregation lost or double-counted executions."""
    scenarios = [_scenario("x")]
    results = [_result("x", seed=i) for i in range(5)]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # Data row looks like "| x |  | 5 | 5 | 0 | 0 | ..." — codec
    # cell is empty because _result does not set metadata.codec_id.
    data_row = next(
        ln for ln in out.splitlines() if ln.startswith("| x |")
    )
    cells = [c.strip() for c in data_row.strip("| ").split("|")]
    assert cells[0] == "x"
    assert cells[1] == ""  # codec (None → empty cell)
    assert cells[2] == "5"  # runs
    assert cells[3] == "5"  # ok
    assert cells[4] == "0"  # skipped
    assert cells[5] == "0"  # failed


# ----- mean ± std rendering ------------------------------------------


def test_report_single_seed_renders_without_std_suffix() -> None:
    """Default ``seeds=(0,)`` path: N=1 per scenario. Numeric cells
    must show just the value, not ``value ± 0``."""
    scenarios = [_scenario("m")]
    results = [
        _result("m", ttft_ms=12.345, wall_s=0.6123, total_tokens=8)
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "| 12.3 |" in out  # ttft_ms
    assert "| 0.612 |" in out  # wall_s
    assert "| 8 |" in out  # tokens with .0f
    assert " ± " not in out


def test_report_multi_seed_renders_mean_std() -> None:
    """N=2 with distinct values: cell becomes ``mean ± std``
    where ``std`` is sample std (``stdev`` divides by N-1)."""
    scenarios = [_scenario("m")]
    results = [
        _result("m", seed=0, ttft_ms=10.0, wall_s=0.500),
        _result("m", seed=1, ttft_ms=14.0, wall_s=0.700),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # ttft: mean=12.0, stdev of [10, 14] = sqrt(8) ≈ 2.828 → "12.0 ± 2.8"
    # with .1f precision.
    assert "12.0 ± 2.8" in out
    # wall_s: mean=0.600, stdev = sqrt(0.02) ≈ 0.1414 → "0.600 ± 0.141"
    # with .3f precision.
    assert "0.600 ± 0.141" in out


def test_report_multi_seed_std_zero_collapses_to_mean() -> None:
    """Deterministic quantities (tokens, scheduler-driven timings
    on CPU-bound fakes) produce identical values across seeds.
    std == 0 must render as a single value, not ``4.0 ± 0.0``."""
    scenarios = [_scenario("det")]
    results = [
        _result("det", seed=0, total_tokens=4, ttft_ms=10.0),
        _result("det", seed=1, total_tokens=4, ttft_ms=10.0),
        _result("det", seed=2, total_tokens=4, ttft_ms=10.0),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # tokens deterministic: ".0f" → "4" (not "4.0")
    assert "| 4 |" in out
    # ttft deterministic: ".1f" → "10.0"
    assert "| 10.0 |" in out
    # No ± anywhere on the deterministic row
    data_row = next(
        ln for ln in out.splitlines() if ln.startswith("| det |")
    )
    assert " ± " not in data_row


def test_report_tokens_column_uses_integer_formatting() -> None:
    """total_tokens is conceptually integer; ``.0f`` prevents the
    mean from rendering as ``4.0`` in the common deterministic case
    and keeps mixed-seed rendering sensible (``5 ± 1`` not ``5.0 ± 1.0``)."""
    scenarios = [_scenario("t")]
    results = [
        _result("t", seed=0, total_tokens=4),
        _result("t", seed=1, total_tokens=6),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # mean = 5, stdev of [4, 6] = sqrt(2) ≈ 1.414 → rounded to "5 ± 1"
    # with .0f precision.
    data_row = next(
        ln for ln in out.splitlines() if ln.startswith("| t |")
    )
    assert "5 ± 1" in data_row


def test_report_mean_std_ignores_none_values() -> None:
    """A mix of skipped (all-None) and ok (populated) rows
    aggregates cleanly: numeric lists contain only non-None
    entries, so mean / std reflect the actual measurements without
    a skipped row counting as zero."""
    scenarios = [_scenario("mix")]
    results = [
        _result("mix", seed=0, status="skipped", reason="cache_missing:/x"),
        _result("mix", seed=1, ttft_ms=10.0),
        _result("mix", seed=2, ttft_ms=14.0),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # ttft: only the two populated values contribute; mean=12,
    # stdev of [10, 14] ≈ 2.828 → "12.0 ± 2.8" with .1f.
    assert "12.0 ± 2.8" in out


def test_report_renders_all_none_numeric_columns_as_empty_cells() -> None:
    """If every seed was skipped the numeric columns stay blank,
    matching pre-C.4 behaviour. Guards against accidental column
    loss on all-None scenarios (pipe count must match the header)."""
    scenarios = [_scenario("skip")]
    results = [
        _result("skip", seed=0, status="skipped", reason="cache_missing:/x"),
        _result("skip", seed=1, status="skipped", reason="cache_missing:/x"),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    data_row = next(
        ln for ln in out.splitlines() if ln.startswith("| skip |")
    )
    header_row = next(
        ln for ln in out.splitlines() if ln.startswith("| id |")
    )
    # Same pipe count as header, even when numeric cells are blank
    assert _count_unescaped_pipes(data_row) == _count_unescaped_pipes(
        header_row
    )
    # At least one ``|  |`` (an empty cell between pipes) must exist
    assert "|  |" in data_row


# ----- Detail section -----------------------------------------------


def test_report_scenario_detail_block_lists_repo_oracle_and_gate() -> None:
    """Detail block carries everything it did pre-C.4 — the only
    shape change is the ``(seed=<N>)`` heading suffix."""
    scenarios = [
        _scenario(
            "one",
            repo="foo/bar",
            oracle=OracleKind.B1_PARITY_VS_SINGLE,
            gate_env_var="DEMO_GATE",
            description="Parity row for the demo model.",
        )
    ]
    results = [
        _result(
            "one",
            status="ok",
            metadata={
                "reference_len": 4,
                "batch_len": 4,
                "first_mismatch_index": -1,
            },
        )
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "### `one`" in out
    assert "`foo/bar`" in out
    assert "`b1_parity_vs_single`" in out
    assert "`DEMO_GATE`" in out
    assert "Parity row for the demo model." in out
    assert "Metadata:" in out
    # metadata rendered as a JSON block inside a fenced code block
    assert '"reference_len": 4' in out
    assert '"first_mismatch_index": -1' in out


def test_report_detail_heading_includes_seed_suffix() -> None:
    """Multi-seed scenarios get one detail section per execution,
    each heading suffixed ``(seed=<N>)`` so a reader can locate
    the seed that failed without counting position."""
    scenarios = [_scenario("multi")]
    results = [
        _result("multi", seed=42),
        _result("multi", seed=43),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "### `multi` (seed=42)" in out
    assert "### `multi` (seed=43)" in out


def test_report_detail_heading_includes_codec_under_multi_codec_fanout() -> None:
    """Post-C.5 step 2: when a scenario spans multiple codec arms,
    the detail heading must disambiguate by codec too. A heading
    that only said ``(seed=42)`` would duplicate across
    ``--all-kv-codecs --seeds 42,43`` — once per codec at the
    same seed — and anchor navigation would collapse onto the
    first match. Appending ``codec=<id>`` restores uniqueness."""
    scenarios = [_scenario("multi-arm")]
    results = [
        _result(
            "multi-arm",
            seed=42,
            metadata={"codec_id": "fp16"},
        ),
        _result(
            "multi-arm",
            seed=42,
            metadata={"codec_id": "block_tq_b64_b4"},
        ),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # Both (seed=42) rows must produce distinct headings.
    assert "### `multi-arm` (codec=fp16, seed=42)" in out
    assert "### `multi-arm` (codec=block_tq_b64_b4, seed=42)" in out
    # The plain ``(seed=42)`` form (without codec prefix) must NOT
    # leak through — that would indicate the heading still mostly
    # keyed on seed alone.
    heading_lines = [
        ln for ln in out.splitlines() if ln.startswith("### `multi-arm`")
    ]
    for ln in heading_lines:
        assert "codec=" in ln, (
            f"heading missing codec disambiguator: {ln!r}"
        )


def test_report_detail_heading_keeps_seed_only_when_codec_id_is_none() -> None:
    """Pure SMOKE / parity scenarios have ``codec_id=None`` in
    metadata and must keep the simpler ``(seed=<N>)`` heading —
    adding ``codec=None`` would be visual noise for the common
    non-codec case."""
    scenarios = [_scenario("no-codec-arm")]
    results = [
        _result(
            "no-codec-arm",
            seed=42,
            metadata={"codec_id": None},
        )
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "### `no-codec-arm` (seed=42)" in out
    # No codec= fragment creeps in under None
    assert "codec=" not in out


def test_report_detail_heading_omits_suffix_when_seed_metadata_missing() -> None:
    """Robustness guard: a ScenarioResult built without going
    through BenchRunner (e.g. in a bench-adjacent test fixture
    that constructs results manually) will lack ``metadata['seed']``.
    The detail heading must fall back to the plain id rather than
    rendering ``(seed=None)``."""
    scenarios = [_scenario("no-seed")]
    results = [_result("no-seed", seed=None)]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "### `no-seed`" in out
    assert "(seed=None)" not in out
    assert "(seed=" not in out  # defensive — no suffix at all


def test_report_detail_preserves_raw_result_order() -> None:
    """Detail sections follow the ``results`` argument order (which
    under scenario-major fan-out places all of scenario A's seeds
    before scenario B's). A reader skimming the file sees related
    rows contiguously."""
    scenarios = [_scenario("a"), _scenario("b")]
    results = [
        _result("a", seed=0),
        _result("a", seed=1),
        _result("b", seed=0),
        _result("b", seed=1),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # Extract the sequence of scenario-detail headings
    heading_ids = [
        ln
        for ln in out.splitlines()
        if ln.startswith("### `")
    ]
    assert heading_ids == [
        "### `a` (seed=0)",
        "### `a` (seed=1)",
        "### `b` (seed=0)",
        "### `b` (seed=1)",
    ]


def test_report_cache_only_gate_renders_placeholder() -> None:
    scenarios = [_scenario("plain", gate_env_var=None)]
    results = [_result("plain")]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "gate: `(cache-only)`" in out


def test_report_escapes_pipe_in_detail_reason() -> None:
    """A failure reason containing a literal pipe must not break
    the GFM detail section's reason line (the renderer should
    backslash-escape it). Post-C.4 step 2 the reason column moved
    from the aggregated table to the per-execution detail; this
    test follows it there."""
    scenarios = [_scenario("x")]
    results = [
        _result(
            "x",
            status="failed",
            reason="ValueError: weird | message with pipe",
        )
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # Detail reason line has form: "- reason: `...`" with backslash
    # escaping on the inner pipe.
    reason_lines = [
        ln for ln in out.splitlines() if ln.startswith("- reason:")
    ]
    assert reason_lines, "expected a reason line in the detail block"
    assert "\\|" in reason_lines[0], (
        f"reason line did not escape the pipe: {reason_lines[0]!r}"
    )


# ----- Id-based invariants (replaces length-equality) ---------------


def test_report_rejects_result_with_unknown_scenario_id() -> None:
    """A result whose ``scenario_id`` is not in the ``scenarios``
    list is a caller-bug: either the runner produced a spurious row
    or the CLI passed an out-of-sync scenarios list. Fail loudly
    rather than silently dropping the result from the table."""
    scenarios = [_scenario("known")]
    results = [_result("unknown")]
    with pytest.raises(ValueError, match="unknown"):
        render_markdown_report(scenarios, results)


def test_report_rejects_scenario_with_zero_results() -> None:
    """A scenario listed but never executed would render as an
    empty / NaN row. That masks a runner bug (dropped the row
    silently) and is just as bad as an unknown-id result — fail
    fast on both sides of the id contract."""
    scenarios = [_scenario("ran"), _scenario("never_ran")]
    results = [_result("ran")]
    with pytest.raises(ValueError, match="zero results"):
        render_markdown_report(scenarios, results)


def test_report_multi_codec_scenario_produces_one_row_per_codec() -> None:
    """Post-C.5 step 2 positive: a scenario with rows across
    multiple codec_ids aggregates into K arm rows, one per codec.
    The step 1 rejection guard is gone because the aggregation
    key is now ``(scenario_id, codec_id)``, not just ``scenario_id``."""
    scenarios = [_scenario("multi")]
    results = [
        _result(
            "multi", seed=0, metadata={"codec_id": "fp16"}, ttft_ms=10.0
        ),
        _result(
            "multi",
            seed=0,
            metadata={"codec_id": "block_tq_b64_b4"},
            ttft_ms=20.0,
        ),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # Two data rows, each carrying a distinct codec cell.
    data_rows = [
        ln for ln in out.splitlines() if ln.startswith("| multi |")
    ]
    assert len(data_rows) == 2
    # Same scenario_id appears twice; the second cell disambiguates.
    codecs_seen = [ln.split("|")[2].strip() for ln in data_rows]
    assert codecs_seen == ["block_tq_b64_b4", "fp16"]  # alpha sort
    # Each arm's ttft cell reflects only its own codec's row.
    assert "| 10.0 |" in out  # fp16 single-seed value
    assert "| 20.0 |" in out  # block_tq single-seed value


def test_report_none_and_concrete_codec_produce_separate_rows() -> None:
    """C.5 step 1 rejected ``None`` + concrete id in the same
    bucket; step 2 accepts them as two distinct arms (the baked /
    no-override arm vs the explicit codec arm). ``None`` sorts
    before concrete ids per ``_codec_sort_key``."""
    scenarios = [_scenario("mixed-none")]
    results = [
        _result(
            "mixed-none", seed=0, metadata={"codec_id": None}, ttft_ms=5.0
        ),
        _result(
            "mixed-none",
            seed=0,
            metadata={"codec_id": "fp16"},
            ttft_ms=15.0,
        ),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    data_rows = [
        ln for ln in out.splitlines() if ln.startswith("| mixed-none |")
    ]
    assert len(data_rows) == 2
    codec_cells = [ln.split("|")[2].strip() for ln in data_rows]
    # None first, concrete id second (sort key: (0, "") < (1, "fp16"))
    assert codec_cells == ["", "fp16"]


def test_report_uniform_none_codec_aggregates_to_single_arm() -> None:
    """The SMOKE / parity row baseline case: every row carries
    ``codec_id=None`` (no override, scenario has no baked kv_codec).
    Aggregates to one ``codec=None`` arm, not N separate rows."""
    scenarios = [_scenario("plain")]
    results = [
        _result("plain", seed=0, metadata={"codec_id": None}),
        _result("plain", seed=1, metadata={"codec_id": None}),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    data_rows = [
        ln for ln in out.splitlines() if ln.startswith("| plain |")
    ]
    assert len(data_rows) == 1
    # codec cell renders as empty string for None
    assert data_rows[0].split("|")[2].strip() == ""


def test_report_multi_codec_multi_seed_aggregates_per_arm() -> None:
    """Seed fan-out within each ``(scenario, codec)`` arm collapses
    to ``mean ± std`` per arm. Cross-codec averaging would
    conflate arms; this test pins that each arm aggregates
    independently."""
    scenarios = [_scenario("arm-agg")]
    results = [
        _result(
            "arm-agg", seed=0, metadata={"codec_id": "fp16"}, ttft_ms=10.0
        ),
        _result(
            "arm-agg", seed=1, metadata={"codec_id": "fp16"}, ttft_ms=14.0
        ),
        _result(
            "arm-agg",
            seed=0,
            metadata={"codec_id": "block_tq_b64_b4"},
            ttft_ms=20.0,
        ),
        _result(
            "arm-agg",
            seed=1,
            metadata={"codec_id": "block_tq_b64_b4"},
            ttft_ms=28.0,
        ),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # fp16 arm: mean=12, stdev of [10, 14] = sqrt(8) ≈ 2.828
    # → "12.0 ± 2.8" with .1f
    assert "12.0 ± 2.8" in out
    # block_tq arm: mean=24, stdev of [20, 28] = sqrt(32) ≈ 5.657
    # → "24.0 ± 5.7" with .1f
    assert "24.0 ± 5.7" in out


def test_report_codec_rows_sorted_none_first_then_alpha() -> None:
    """Within a scenario's block of arm rows, ordering is:
    ``None`` (if present) first, then concrete codec ids in
    alphabetical order. Stable ordering matters for PR review
    diffs — two runs with the same codec set must produce the
    same row order."""
    scenarios = [_scenario("ordering")]
    results = [
        _result(
            "ordering",
            seed=0,
            metadata={"codec_id": "fp16"},
            ttft_ms=1.0,
        ),
        _result(
            "ordering",
            seed=0,
            metadata={"codec_id": "block_tq_b64_b4"},
            ttft_ms=2.0,
        ),
        _result(
            "ordering",
            seed=0,
            metadata={"codec_id": None},
            ttft_ms=3.0,
        ),
        _result(
            "ordering",
            seed=0,
            metadata={"codec_id": "ext_rabitq_b4"},
            ttft_ms=4.0,
        ),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    data_rows = [
        ln for ln in out.splitlines() if ln.startswith("| ordering |")
    ]
    codec_cells = [ln.split("|")[2].strip() for ln in data_rows]
    assert codec_cells == [
        "",  # None first
        "block_tq_b64_b4",
        "ext_rabitq_b4",
        "fp16",
    ]


def test_report_rejects_duplicate_scenario_ids() -> None:
    """The aggregated table is keyed by scenario_id; if scenarios
    contains the same id twice, the iteration order in
    ``_render_aggregated_table`` would render two identical rows
    pointing at the same aggregate bucket, and the summary line's
    ``Scenarios: total=`` would double-count. Callers must dedupe
    upstream — the CLI's ``_dedupe_scenario_ids`` helper handles
    the common repeatable-flag case; this check is defense-in-depth
    for direct programmatic callers (tests, notebooks)."""
    scenarios = [_scenario("foo"), _scenario("foo")]
    results = [_result("foo", seed=0), _result("foo", seed=1)]
    with pytest.raises(ValueError, match="duplicate id"):
        render_markdown_report(scenarios, results)


def test_report_accepts_fan_out_results_with_matching_ids() -> None:
    """The replacement invariant for the pre-C.4 length-equality
    check: differing lengths are fine as long as every result id is
    in scenarios and every scenario is represented."""
    scenarios = [_scenario("a"), _scenario("b")]
    results = [
        _result("a", seed=0),
        _result("a", seed=1),
        _result("b", seed=0),
    ]
    # No exception raised
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "| a |" in out
    assert "| b |" in out


# ----- Empty + metadata JSON ----------------------------------------


def test_report_handles_empty_inputs() -> None:
    out = render_markdown_report([], [], generated_at="T")
    assert out.startswith("# silica-mlx bench report")
    assert "Scenarios: total=0" in out
    assert "Runs: total=0" in out
    assert "ok=0 skipped=0 failed=0" in out
    # Table still renders header + separator even with zero rows so
    # downstream renderers do not need to special-case empty runs.
    assert "| id | codec | runs | ok | skipped | failed |" in out


def test_report_metadata_block_is_valid_json() -> None:
    """Regression guard: metadata is dumped into a fenced code block
    as JSON, so a reader can paste it into a JSON parser without
    fixing up Python repr artefacts (True, None, single quotes)."""
    scenarios = [_scenario("x")]
    results = [
        _result(
            "x",
            metadata={
                "first_failure": None,
                "rows": [{"row": 0, "first_mismatch_index": -1}],
                "flag": True,
            },
        )
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # Extract the text between the first pair of ``` fences
    fence = out.index("```")
    end_fence = out.index("```", fence + 3)
    body = out[fence + 3 : end_fence].strip()
    parsed = json.loads(body)
    assert parsed["flag"] is True
    assert parsed["first_failure"] is None
    assert parsed["rows"][0]["first_mismatch_index"] == -1
    # seed=0 was injected by _result; it is part of the metadata
    # block alongside the caller-provided keys.
    assert parsed["seed"] == 0
