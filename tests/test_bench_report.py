"""Unit tests for ``silica.bench.report.render_markdown_report``.

The report is a pure rendering function — no I/O, no clock unless
``generated_at`` is omitted — so every test constructs a small
scenarios/results pair and inspects the returned string. Structural
assertions only (header present, table rows count correctly,
counts line reflects statuses); full rendering fidelity is not
pinned character-for-character because the format will evolve as
later phases land (TEACHER_FORCED_ARGMAX metadata, vqbench PPL
reference column) and character pins would be pure friction.
"""

from __future__ import annotations

import json
import re

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
    **metrics: float,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id,
        status=status,
        reason=reason,
        metadata=metadata or {},
        **metrics,
    )


def test_report_has_header_and_fixed_timestamp() -> None:
    scenarios = [_scenario("one")]
    results = [_result("one", ttft_ms=10.0, total_tokens=4, wall_s=0.5)]
    out = render_markdown_report(
        scenarios, results, generated_at="2026-04-20T12:34:56"
    )
    assert out.splitlines()[0] == "# silica-mlx bench report"
    assert "Generated: 2026-04-20T12:34:56" in out


def test_report_summary_counts_statuses() -> None:
    scenarios = [_scenario(sid) for sid in ("a", "b", "c", "d")]
    results = [
        _result("a", status="ok"),
        _result("b", status="ok"),
        _result("c", status="skipped", reason="cache_missing:/x"),
        _result("d", status="failed", reason="RuntimeError: boom"),
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "total=4" in out
    assert "ok=2" in out
    assert "skipped=1" in out
    assert "failed=1" in out


def test_report_table_has_row_per_result() -> None:
    scenarios = [_scenario("one"), _scenario("two")]
    results = [
        _result("one", ttft_ms=11.1, decode_tok_s=222.2, total_tokens=4),
        _result(
            "two", status="skipped", reason="env_var_not_set:DEMO_GATE"
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
    assert "env_var_not_set:DEMO_GATE" in table_block


def test_report_renders_numeric_columns_with_fixed_precision() -> None:
    scenarios = [_scenario("m")]
    results = [
        _result(
            "m",
            ttft_ms=12.345,
            decode_tok_s=111.987,
            resident_mb=29.36,
            peak_memory_mb=1234.5,
            wall_s=0.6123,
            total_tokens=8,
        )
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # ttft .1f, decode .1f, resident .1f, peak .1f, wall .3f
    assert "| 12.3 |" in out  # ttft
    assert "| 112.0 |" in out  # decode_tok_s rounded
    assert "| 29.4 |" in out  # resident_mb
    assert "| 1234.5 |" in out  # peak_mb
    assert "| 0.612 |" in out  # wall_s


def test_report_renders_none_metrics_as_empty_cells() -> None:
    scenarios = [_scenario("m")]
    results = [_result("m", status="skipped", reason="cache_missing:/x")]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # skipped rows have all metrics None — cells should be blank (|  |)
    assert "|  |" in out
    # total_tokens is None -> blank
    lines = [ln for ln in out.splitlines() if ln.startswith("| m |")]
    assert lines, "expected data row for scenario 'm'"
    row = lines[0]
    # count pipes; data row must have the same column count as the
    # header regardless of Nones (guards against accidental column
    # loss on all-None scenarios)
    header_line = next(
        ln for ln in out.splitlines() if ln.startswith("| id |")
    )
    assert row.count("|") == header_line.count("|")


def test_report_scenario_detail_block_lists_repo_oracle_and_gate() -> None:
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
            metadata={"reference_len": 4, "batch_len": 4, "first_mismatch_index": -1},
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


def test_report_cache_only_gate_renders_placeholder() -> None:
    scenarios = [_scenario("plain", gate_env_var=None)]
    results = [_result("plain")]
    out = render_markdown_report(scenarios, results, generated_at="T")
    assert "gate: `(cache-only)`" in out


def test_report_escapes_pipe_in_reasons() -> None:
    """A failure reason containing a literal pipe must not break the
    GFM table (the renderer should backslash-escape it)."""
    scenarios = [_scenario("x")]
    results = [
        _result(
            "x",
            status="failed",
            reason="ValueError: weird | message with pipe",
        )
    ]
    out = render_markdown_report(scenarios, results, generated_at="T")
    # Escaped pipe appears in the reason cell; unescaped (column-
    # delimiter) pipe count must still match the header row, otherwise
    # the GFM table shape collapses on the downstream Markdown render.
    data_row = next(
        ln for ln in out.splitlines() if ln.startswith("| x |")
    )
    header_row = next(
        ln for ln in out.splitlines() if ln.startswith("| id |")
    )
    assert _count_unescaped_pipes(data_row) == _count_unescaped_pipes(
        header_row
    )
    assert "\\|" in data_row


def test_report_rejects_misaligned_input_lengths() -> None:
    scenarios = [_scenario("a"), _scenario("b")]
    results = [_result("a")]
    with pytest.raises(ValueError, match="same length"):
        render_markdown_report(scenarios, results)


def test_report_handles_empty_inputs() -> None:
    out = render_markdown_report([], [], generated_at="T")
    assert out.startswith("# silica-mlx bench report")
    assert "total=0 ok=0 skipped=0 failed=0" in out
    # Table still renders header + separator even with zero rows so
    # downstream renderers do not need to special-case empty runs.
    assert "| id | status |" in out


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
