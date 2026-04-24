"""CLI smoke tests for ``scripts/bench.py``.

Thin tests guarding the entry points a human is most likely to hit
first:

  * ``--list`` — must not require any environment setup beyond a
    normal checkout; should print each known scenario id on its
    own line and exit 0. Regression-tests the repo-root import
    shim (direct ``python scripts/bench.py`` without PYTHONPATH).
  * unknown ``--scenario`` — must surface the invalid id as a
    stderr error line and exit 2 (argparse-usage class), not a
    traceback. Regression-tests the ``KeyError`` handling on
    ``get_scenario``.
  * ``--report-md`` — writes a Markdown report file alongside the
    optional JSONL. Tested against a scenario that is guaranteed
    to skip on any dev box (missing-cache repo) so the CLI does
    not need real weights to exercise the report code path.
  * ``--seeds`` (P-5-C.4 step 1) — parsing / validation unit tests
    for ``_parse_seeds`` plus a subprocess integration test that
    confirms ``--seeds 42,43`` produces two JSONL rows with the
    two seeds recorded in ``metadata.seed``.

Deliberately does NOT exercise ``--scenario qwen3-0.6b-smoke`` —
that path requires the Qwen3-0.6B cache and the real engine, which
is what ``tests/test_bench_runner.py`` covers via fakes and what
``scripts/bench.py --scenario qwen3-0.6b-smoke`` verifies ad-hoc.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "bench.py"


def _load_bench_cli_module() -> ModuleType:
    """Load ``scripts/bench.py`` as an importable module.

    ``scripts/bench.py`` is a standalone script, not a Python
    package member, so direct ``import bench`` does not work.
    ``importlib`` loads it by path so we can unit-test the
    ``_parse_seeds`` helper without the subprocess overhead of a
    full CLI invocation.
    """
    spec = importlib.util.spec_from_file_location(
        "silica_mlx_scripts_bench", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_cli_report_md_writes_file(tmp_path: Path) -> None:
    """``--report-md PATH`` writes a GFM Markdown report whose header
    and summary line match the Python-side renderer. Uses the
    built-in qwen3-0.6b-smoke scenario and accepts either status
    (ok if cache present, skipped if not) so the test does not
    require the 0.6B cache to be populated."""
    out_path = tmp_path / "report.md"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke", "--report-md", str(out_path)
    )
    assert result.returncode in (0, 1), (
        f"bench CLI should exit 0 or 1; got {result.returncode}; "
        f"stderr={result.stderr!r}"
    )
    assert out_path.exists(), (
        f"--report-md should create {out_path}; stderr={result.stderr!r}"
    )
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("# silica-mlx bench report")
    assert "Generated:" in text
    assert "Runs: total=1" in text
    # Post-C.5 step 2 Markdown header is the aggregated shape with
    # codec column; the terminal-only ``| id | codec | seed | status |``
    # header belongs to scripts/bench.py:_print_table, not the
    # report renderer.
    assert "| id | codec | runs | ok | skipped | failed |" in text
    assert "| qwen3-0.6b-smoke |" in text
    assert "### `qwen3-0.6b-smoke`" in text


def test_cli_report_md_and_out_coexist(tmp_path: Path) -> None:
    """``--report-md`` and ``--out`` target different files and must
    both populate on the same run."""
    pytest.importorskip("json")
    jsonl = tmp_path / "results.jsonl"
    md = tmp_path / "report.md"
    result = _run_cli(
        "--scenario",
        "qwen3-0.6b-smoke",
        "--out",
        str(jsonl),
        "--report-md",
        str(md),
    )
    assert result.returncode in (0, 1)
    assert jsonl.exists()
    assert md.exists()
    # JSONL is line-delimited JSON, one row per scenario
    assert len([ln for ln in jsonl.read_text().splitlines() if ln]) == 1
    # Markdown has the detail block keyed on the scenario id
    assert "### `qwen3-0.6b-smoke`" in md.read_text()


# ---------- P-5-C.4 step 1 — --seeds parsing --------------------------


class TestParseSeedsHappyPath:
    def test_single_seed(self) -> None:
        bench = _load_bench_cli_module()
        assert bench._parse_seeds("0") == [0]

    def test_multiple_distinct_seeds_preserve_order(self) -> None:
        bench = _load_bench_cli_module()
        assert bench._parse_seeds("42,43,44") == [42, 43, 44]

    def test_whitespace_around_tokens_allowed(self) -> None:
        """CLI paste-ability: ``--seeds '42, 43, 44'`` is a natural
        shape that pastes fine from a shell variable."""
        bench = _load_bench_cli_module()
        assert bench._parse_seeds(" 42, 43 , 44 ") == [42, 43, 44]

    def test_zero_is_valid_seed(self) -> None:
        """Lower bound: 0 is a legal NumPy seed."""
        bench = _load_bench_cli_module()
        assert bench._parse_seeds("0,1") == [0, 1]

    def test_upper_bound_seed_accepted(self) -> None:
        """2**32 - 1 is the last legal NumPy seed."""
        bench = _load_bench_cli_module()
        assert bench._parse_seeds(str((1 << 32) - 1)) == [(1 << 32) - 1]


class TestParseSeedsDedup:
    def test_duplicate_seeds_dropped_with_stderr_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--seeds 42,42,43`` becomes ``[42, 43]``; warning goes to
        stderr so the user notices the drift but the run still
        proceeds."""
        bench = _load_bench_cli_module()
        result = bench._parse_seeds("42,42,43")
        assert result == [42, 43]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "warning" in captured.err
        assert "42" in captured.err

    def test_multiple_duplicates_reported_once(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bench = _load_bench_cli_module()
        result = bench._parse_seeds("1,2,1,3,2,1")
        assert result == [1, 2, 3]
        captured = capsys.readouterr()
        assert captured.err.count("warning:") == 1

    def test_no_warning_when_input_already_unique(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bench = _load_bench_cli_module()
        bench._parse_seeds("10,11,12")
        captured = capsys.readouterr()
        assert captured.err == ""


class TestParseSeedsRejectsInvalid:
    def test_rejects_non_integer_token(self) -> None:
        bench = _load_bench_cli_module()
        with pytest.raises(ValueError, match="not an integer"):
            bench._parse_seeds("42,abc,44")

    def test_rejects_negative_seed(self) -> None:
        bench = _load_bench_cli_module()
        with pytest.raises(ValueError, match="out of range"):
            bench._parse_seeds("42,-1")

    def test_rejects_seed_at_or_above_2_to_32(self) -> None:
        """NumPy accepts ``[0, 2**32)`` so ``2**32`` itself is out
        of range."""
        bench = _load_bench_cli_module()
        with pytest.raises(ValueError, match="out of range"):
            bench._parse_seeds(str(1 << 32))

    def test_rejects_empty_token_trailing_comma(self) -> None:
        bench = _load_bench_cli_module()
        with pytest.raises(ValueError, match="empty token"):
            bench._parse_seeds("42,")

    def test_rejects_empty_token_consecutive_commas(self) -> None:
        bench = _load_bench_cli_module()
        with pytest.raises(ValueError, match="empty token"):
            bench._parse_seeds("42,,43")

    def test_rejects_entirely_empty_string(self) -> None:
        bench = _load_bench_cli_module()
        with pytest.raises(ValueError, match="empty token"):
            bench._parse_seeds("")


# ---------- P-5-C.4 step 2 — --scenario duplicate handling -----------


class TestDedupeScenarioIds:
    """Direct unit tests on ``_dedupe_scenario_ids`` — parallel to the
    ``_parse_seeds`` coverage below. ``--scenario`` is marked
    repeatable in ``build_parser``, so a pasted shell variable or a
    typo can produce duplicates; the helper collapses them with a
    stderr warning rather than running the same scenario twice."""

    def test_no_duplicates_is_a_noop(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bench = _load_bench_cli_module()
        assert bench._dedupe_scenario_ids(["a", "b", "c"]) == ["a", "b", "c"]
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_duplicate_ids_dropped_with_stderr_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bench = _load_bench_cli_module()
        result = bench._dedupe_scenario_ids(["foo", "foo", "bar"])
        assert result == ["foo", "bar"]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "warning" in captured.err
        assert "foo" in captured.err

    def test_dedup_preserves_first_occurrence_order(self) -> None:
        """Order-preserving dedup: output order matches first-seen
        order, not sorted order. Matches ``_parse_seeds`` semantic."""
        bench = _load_bench_cli_module()
        assert bench._dedupe_scenario_ids(
            ["c", "a", "b", "a", "c"]
        ) == ["c", "a", "b"]


def test_cli_duplicate_scenario_dedupes_with_warning(
    tmp_path: Path,
) -> None:
    """``--scenario qwen3-0.6b-smoke --scenario qwen3-0.6b-smoke``
    dedupes to a single run with a stderr warning. Without this the
    aggregated report would show two identical rows pointing at the
    same bucket (if the renderer did not also reject it as it now
    does — the CLI dedup is the friendly upstream layer)."""
    out_path = tmp_path / "dedup.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--scenario", "qwen3-0.6b-smoke",
        "--out", str(out_path),
    )
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    )
    assert "warning" in result.stderr.lower()
    # Single (scenario, seed) with default seeds=(0,) → one row
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "qwen3-0.6b-smoke"


# ---------- P-5-C.4 step 1 — --seeds integration via subprocess -------


def test_cli_seeds_invalid_exits_2(tmp_path: Path) -> None:
    """Invalid ``--seeds`` value returns 2 (argparse-usage class),
    not a traceback. ``_parse_seeds`` raises ``ValueError`` inside
    ``main()``; the CLI translates it to exit 2 just like unknown
    scenario ids."""
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke", "--seeds", "not-a-seed"
    )
    assert result.returncode == 2, (
        f"expected exit 2 on bad --seeds, got {result.returncode}; "
        f"stderr={result.stderr!r}"
    )
    assert "error:" in result.stderr


def test_cli_seeds_produces_one_jsonl_row_per_seed(
    tmp_path: Path,
) -> None:
    """``--seeds 42,43`` with one scenario that is cache-missing on
    the dev box → 2 skipped JSONL rows, each carrying
    ``metadata.seed``. We use the deliberately-uncached
    ``qwen3-0.6b-smoke`` row so the test does not require weights.

    The subprocess path covers:
    - argparse flag registration (``--seeds`` exists)
    - parsing delegation (``_parse_seeds`` is invoked on the value)
    - runner fan-out (BenchRunner.run produces N*M rows)
    - JSONL emission (every row carries metadata.seed)
    """
    out_path = tmp_path / "seeds.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--seeds", "42,43",
        "--out", str(out_path),
    )
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    )
    assert out_path.exists()
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 2, f"expected 2 rows (2 seeds × 1 scenario), got {len(rows)}"
    seeds_seen = [row["metadata"]["seed"] for row in rows]
    assert seeds_seen == [42, 43]
    # Scenario-major ordering: same scenario_id on both rows.
    assert rows[0]["scenario_id"] == rows[1]["scenario_id"]


def test_cli_seeds_duplicate_warns_and_dedupes(tmp_path: Path) -> None:
    """``--seeds 42,42,43`` dedupes to 2 rows with a stderr warning."""
    out_path = tmp_path / "dedupe.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--seeds", "42,42,43",
        "--out", str(out_path),
    )
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    )
    assert "warning" in result.stderr.lower()
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 2
    assert [row["metadata"]["seed"] for row in rows] == [42, 43]


# ---------- P-5-C.5 step 1 — --kv-codec CLI surface ------------------


def test_cli_kv_codec_default_none_preserves_baked_codec(
    tmp_path: Path,
) -> None:
    """No ``--kv-codec`` flag → codec override is ``None`` → the
    scenario's baked codec flows through verbatim. For
    ``qwen3-0.6b-smoke`` (no kv_codec), metadata.codec_id is
    null in JSON — the key still exists so consumers can rely on
    a stable schema."""
    out_path = tmp_path / "default_codec.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--out", str(out_path),
    )
    assert result.returncode in (0, 1)
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert "codec_id" in rows[0]["metadata"]
    assert rows[0]["metadata"]["codec_id"] is None


def test_cli_kv_codec_override_records_override_id(
    tmp_path: Path,
) -> None:
    """``--kv-codec fp16`` applied to any scenario records the
    override in ``metadata.codec_id``. We target
    ``qwen3-0.6b-compression-block-tq-b64-b4`` (a STORAGE row
    baked with ``block_tq_b64_b4``) so the override visibly
    replaces the baked value.

    Accepts any terminal status because the scenario may either
    run (cache present) or skip (cache missing); the codec_id
    injection happens before both branches, so the test is
    independent of HF cache state."""
    out_path = tmp_path / "override_codec.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-compression-block-tq-b64-b4",
        "--kv-codec", "fp16",
        "--out", str(out_path),
    )
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    )
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["metadata"]["codec_id"] == "fp16"


def test_cli_kv_codec_unknown_id_exits_2(tmp_path: Path) -> None:
    """Unknown codec id exits 2 with a CLI-friendly error message
    that lists available ids. Mirrors the unknown-scenario-id
    exit contract so callers can distinguish bad input from
    runtime failure."""
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--kv-codec", "not-a-real-codec",
    )
    assert result.returncode == 2, (
        f"expected exit 2 for unknown codec id, got "
        f"{result.returncode}; stderr={result.stderr!r}"
    )
    assert "error:" in result.stderr
    assert "not-a-real-codec" in result.stderr
    # The registry error helper lists available ids so a user with
    # a typo sees the closest match.
    assert "available:" in result.stderr
    assert "fp16" in result.stderr
    assert "block_tq_b64_b4" in result.stderr


def test_cli_kv_codec_and_all_kv_codecs_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    """argparse's ``mutually_exclusive_group`` rejects
    ``--kv-codec X --all-kv-codecs`` at parse time with exit 2.
    CLI-friendly: the user gets a specific 'not allowed with'
    message rather than a runtime error after loading weights."""
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--kv-codec", "fp16",
        "--all-kv-codecs",
    )
    assert result.returncode == 2, (
        f"expected exit 2 for mutex violation, got "
        f"{result.returncode}; stderr={result.stderr!r}"
    )
    # argparse's exact phrasing for mutex violations
    assert "not allowed with" in result.stderr


def test_cli_all_kv_codecs_fans_over_every_registered_codec(
    tmp_path: Path,
) -> None:
    """``--all-kv-codecs`` on a SMOKE scenario (prefix_cache=False)
    produces K failed rows — one per codec in
    ``CODEC_REGISTRY`` — each with the attempted codec_id in
    metadata and a ``codec_override_invalid`` reason.

    Why this particular scenario: SMOKE is deterministically
    incompatible with any codec (no prefix cache), so all K
    overrides fail at ``Workload.__post_init__``. That gives a
    stable test assertion: len(rows) == len(registry). A
    compatible scenario (e.g. compression-*) would depend on
    weights being cached and turn into an env-gated test."""
    from silica.bench.codec_registry import list_codec_ids

    out_path = tmp_path / "all_codecs.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--all-kv-codecs",
        "--out", str(out_path),
    )
    # Every row failed → exit 1
    assert result.returncode == 1
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_codecs = list_codec_ids()
    assert len(rows) == len(expected_codecs), (
        f"expected one row per codec, got {len(rows)} vs "
        f"{len(expected_codecs)}"
    )
    observed_codecs = [row["metadata"]["codec_id"] for row in rows]
    # Scenario-major / codec middle / seed inner; within the
    # single scenario the codec order matches the runner's
    # codec_overrides, which the CLI built from list_codec_ids()
    # (already sorted). So observed order == expected_codecs.
    assert observed_codecs == list(expected_codecs)
    # Every row should be a codec_override_invalid failure.
    assert all(row["status"] == "failed" for row in rows)
    assert all(
        "codec_override_invalid" in row["reason"] for row in rows
    )


def test_cli_all_kv_codecs_combined_with_seeds_is_3d_fan_out(
    tmp_path: Path,
) -> None:
    """Headline step 2 invariant: scenario × codec × seed fan-out
    produces ``1 * K * M`` JSONL rows. End-to-end coverage that
    the unit tests on ``_aggregate_by_scenario_codec`` + the
    subprocess CLI paths actually compose. A regression here
    would mean either BenchRunner.run drops an axis or the CLI
    fails to stitch seeds + codecs into codec_overrides × seeds
    correctly."""
    from silica.bench.codec_registry import list_codec_ids

    out_path = tmp_path / "all_3d.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--all-kv-codecs",
        "--seeds", "42,43",
        "--out", str(out_path),
    )
    assert result.returncode in (0, 1)
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    k = len(list_codec_ids())
    assert len(rows) == k * 2, (
        f"expected {k * 2} rows (K codecs * 2 seeds), got {len(rows)}"
    )
    # Every (codec, seed) pair must appear exactly once.
    observed_pairs = {
        (row["metadata"]["codec_id"], row["metadata"]["seed"])
        for row in rows
    }
    expected_pairs = {
        (codec, seed) for codec in list_codec_ids() for seed in (42, 43)
    }
    assert observed_pairs == expected_pairs


def test_cli_all_kv_codecs_report_renders_per_codec_rows(
    tmp_path: Path,
) -> None:
    """``--all-kv-codecs --report-md`` produces one aggregated row
    per ``(scenario, codec)`` arm. For a SMOKE scenario that
    fails every codec override, we get K failed arms, each with
    runs=1 failed=1 in the table."""
    from silica.bench.codec_registry import list_codec_ids

    md_path = tmp_path / "all_codecs.md"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--all-kv-codecs",
        "--report-md", str(md_path),
    )
    assert result.returncode in (0, 1)
    md = md_path.read_text(encoding="utf-8")

    # Summary line reflects K runs (one per codec).
    k = len(list_codec_ids())
    assert f"Runs: total={k}" in md
    assert "Scenarios: total=1" in md

    # One data row per codec in the aggregated table. Count rows
    # whose first non-separator cell starts with the scenario id.
    table_start = md.index("## Results")
    detail_start = md.index("## Scenario details")
    table_block = md[table_start:detail_start]
    data_rows = [
        ln
        for ln in table_block.splitlines()
        if ln.startswith("| qwen3-0.6b-smoke |")
    ]
    assert len(data_rows) == k, (
        f"expected {k} per-codec arm rows, got {len(data_rows)}"
    )


def test_cli_kv_codec_on_non_prefix_cache_scenario_produces_failed_row(
    tmp_path: Path,
) -> None:
    """``--kv-codec X --scenario qwen3-0.6b-smoke`` applied to a
    non-prefix-cache scenario surfaces as a failed row (not exit 2
    or crash) — the Workload guard fires at dataclass
    construction time inside ``_run_one``, which the exception
    boundary collapses to a ``codec_override_invalid`` reason."""
    out_path = tmp_path / "incompatible.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--kv-codec", "block_tq_b64_b4",
        "--out", str(out_path),
    )
    # Failed row → exit 1 (some scenario failed) is expected.
    assert result.returncode == 1, (
        f"expected exit 1 for failed row, got {result.returncode}; "
        f"stderr={result.stderr!r}"
    )
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "codec_override_invalid" in rows[0]["reason"]
    assert rows[0]["metadata"]["codec_id"] == "block_tq_b64_b4"


def test_cli_seeds_default_produces_one_row(tmp_path: Path) -> None:
    """Omitting ``--seeds`` keeps the pre-C.4 single-row shape: one
    JSONL row per scenario, ``metadata.seed == 0``. This is the
    backward-compat guard the user's Q1 lean hinges on."""
    out_path = tmp_path / "default.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--out", str(out_path),
    )
    assert result.returncode in (0, 1)
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["metadata"]["seed"] == 0


def test_cli_terminal_table_includes_seed_and_codec_columns(
    tmp_path: Path,
) -> None:
    """Under ``--seeds 42,43`` (and post-C.5 step 2 also
    ``--all-kv-codecs``) the terminal table prints one row per
    ``(scenario, codec, seed)`` execution, and each row must be
    disambiguated by ``codec`` and ``seed`` cells — otherwise a
    multi-arm run renders as visually-identical rows and the
    operator cannot tell which arm failed from stdout alone.

    Expected header order is ``| id | codec | seed | status | ...``,
    with codec bound tighter to scenario identity than seed
    (matches the report's row grouping by (scenario_id, codec_id)).
    """
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--seeds", "42,43",
    )
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    )
    assert "| id | codec | seed | status |" in result.stdout, (
        f"expected codec+seed columns in header; got:\n{result.stdout}"
    )
    # Both seed values must appear in their own cell. The full
    # ``| 42 |`` / ``| 43 |`` pattern (with surrounding spaces and
    # pipes) is more specific than a bare integer match — it binds
    # the assertion to a table cell, not an incidental substring.
    assert "| 42 |" in result.stdout
    assert "| 43 |" in result.stdout


def test_cli_report_md_detail_order_is_scenario_major(
    tmp_path: Path,
) -> None:
    """Regression guard on the ``scenarios_expanded`` list
    comprehension in ``main()``. The expression

        [s for s in scenarios for _ in seeds]

    produces scenario-major order ``[A, A, B, B]`` that aligns with
    ``BenchRunner.run``'s scenario-major result order. Flipping to
    ``[s for _ in seeds for s in scenarios]`` yields seed-major
    ``[A, B, A, B]`` — same length, zip-happy, every detail section
    silently mis-describes the wrong result. This test feeds two
    different scenario ids + two seeds and asserts the rendered
    Markdown detail sections appear in scenario-major order.

    Uses two ``Qwen/Qwen3-0.6B`` scenarios so cache state is
    uniform (both skip on a cacheless box, both run if cached);
    the order check is independent of pass/fail outcome. No
    real model load is required — the JSONL metadata carries seed
    on any status including ``skipped``.
    """
    out_path = tmp_path / "order.md"
    jsonl_path = tmp_path / "order.jsonl"
    result = _run_cli(
        "--scenario", "qwen3-0.6b-smoke",
        "--scenario", "qwen3-0.6b-long-in-short-out",
        "--seeds", "42,43",
        "--out", str(jsonl_path),
        "--report-md", str(out_path),
    )
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    )

    # JSONL: 2 scenarios × 2 seeds = 4 rows in scenario-major order.
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 4
    observed_jsonl = [
        (row["scenario_id"], row["metadata"]["seed"]) for row in rows
    ]
    assert observed_jsonl == [
        ("qwen3-0.6b-smoke", 42),
        ("qwen3-0.6b-smoke", 43),
        ("qwen3-0.6b-long-in-short-out", 42),
        ("qwen3-0.6b-long-in-short-out", 43),
    ]

    # Markdown: the ``### `<scenario_id>` (seed=<N>)`` detail headings
    # must appear in scenario-major order. Post-C.4 step 2 the
    # renderer looks scenarios up by id and expands detail from the
    # raw results list, so the heading sequence is the direct
    # signal of result ordering.
    md_text = out_path.read_text(encoding="utf-8")
    heading_ids: list[str] = [
        line.strip()
        for line in md_text.splitlines()
        if line.startswith("### `")
    ]
    assert heading_ids == [
        "### `qwen3-0.6b-smoke` (seed=42)",
        "### `qwen3-0.6b-smoke` (seed=43)",
        "### `qwen3-0.6b-long-in-short-out` (seed=42)",
        "### `qwen3-0.6b-long-in-short-out` (seed=43)",
    ]
