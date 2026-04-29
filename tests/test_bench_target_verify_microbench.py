"""Tests for ``silica.bench.microbench.target_verify`` (P-6.0.5 sub-unit 7).

Pure-function coverage for the microbench's helpers and JSONL row
schema. The end-to-end ``run()`` entry point loads
``mlx-community/Qwen3.5-27B-4bit`` (~16 GB) and is exercised by
the manual on-device pass during the P-6.0.5 acceptance run, not
by the unit suite.

What this file covers:

  * ``make_candidate_token_ids`` — exact token-count contract
    (length matches ``verify_k`` by construction, no tokenizer
    drift) plus boundary checks.
  * ``percentile`` — p50 / p95 + edge cases (n=1, monotone vs
    sorted samples).
  * ``measurement_to_jsonl_row`` — every documented field is
    populated and typed; ``candidate_token_count`` echoes the
    measurement's exact token count.
  * ``render_markdown_report`` — header presence, table row
    cardinality, derived ``marginal`` column reads correctly.
  * ``_parse_verify_ks`` — CLI arg parsing accepts valid lists
    and rejects empty / out-of-range / non-integer inputs.
  * Analytic estimators — ``estimate_weight_bytes_read`` and
    ``estimate_kv_bytes_read`` produce sane numbers from a stub
    adapter and degrade safely when ``num_parameters`` is absent.
  * ``run`` validation — rejects ``verify_ks`` that omit k=1
    (the marginal baseline) before loading the model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from silica.bench.microbench.target_verify import (
    MAX_VERIFY_K,
    _MeasureResult,
    _parse_verify_ks,
    _RunConfig,
    estimate_kv_bytes_read,
    estimate_weight_bytes_read,
    make_candidate_token_ids,
    measurement_to_jsonl_row,
    percentile,
    render_markdown_report,
    run,
    write_artefacts,
)

# ---------------------------------------------------------------------------
# make_candidate_token_ids
# ---------------------------------------------------------------------------


def test_make_candidate_token_ids_returns_exact_length() -> None:
    """The candidate slice must have ``len == verify_k`` exactly so
    the timed forward processes ``verify_k`` tokens without tokenizer
    drift contaminating the cost curve."""
    for k in (1, 2, 4, 8, 16):
        ids = make_candidate_token_ids(k)
        assert len(ids) == k


def test_make_candidate_token_ids_uses_zero_id() -> None:
    """Token id 0 is benign across the Qwen3 / Gemma4 family
    tokenizers; pin the choice so a future edit cannot silently
    pick a content-dependent id."""
    assert make_candidate_token_ids(4) == [0, 0, 0, 0]


def test_make_candidate_token_ids_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match=r"verify_k must be in"):
        make_candidate_token_ids(0)
    with pytest.raises(ValueError, match=r"verify_k must be in"):
        make_candidate_token_ids(-1)


def test_make_candidate_token_ids_rejects_above_max() -> None:
    with pytest.raises(ValueError, match=r"verify_k must be in"):
        make_candidate_token_ids(MAX_VERIFY_K + 1)


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_p50_returns_median() -> None:
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(samples, 50.0) == 30.0


def test_percentile_p95_with_ten_samples_interpolates() -> None:
    """N=10 sorted samples [1..10]; p95 rank = 0.95 × 9 = 8.55, so
    the value sits between the 9th (= 9.0) and 10th (= 10.0)
    samples at frac=0.55 → 9.55."""
    samples = [float(i + 1) for i in range(10)]
    assert percentile(samples, 95.0) == pytest.approx(9.55)


def test_percentile_with_single_sample_returns_that_sample() -> None:
    assert percentile([42.0], 50.0) == 42.0
    assert percentile([42.0], 95.0) == 42.0


def test_percentile_handles_unsorted_input() -> None:
    samples = [5.0, 3.0, 9.0, 1.0, 7.0]  # sorted: 1, 3, 5, 7, 9
    assert percentile(samples, 50.0) == 5.0


def test_percentile_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        percentile([], 50.0)


def test_percentile_rejects_q_out_of_band() -> None:
    with pytest.raises(ValueError, match=r"q must be in \[0, 100\]"):
        percentile([1.0, 2.0], -1.0)
    with pytest.raises(ValueError, match=r"q must be in \[0, 100\]"):
        percentile([1.0, 2.0], 101.0)


# ---------------------------------------------------------------------------
# measurement_to_jsonl_row
# ---------------------------------------------------------------------------


def test_measurement_to_jsonl_row_carries_documented_fields() -> None:
    measurement = _MeasureResult(
        verify_k=4,
        samples_ms=[100.0, 110.0, 120.0, 130.0, 140.0],
        peak_memory_mb=15336.5,
        prefix_token_count=113,
        candidate_token_count=4,
    )
    row = measurement_to_jsonl_row(
        measurement=measurement,
        baseline_p50_ms=95.0,
        weight_bytes_read=13_500_000_000,
        seed=42,
        repo="mlx-community/Qwen3.5-27B-4bit",
    )
    expected_keys = {
        "repo",
        "verify_k",
        "candidate_token_count",
        "forward_ms_p50",
        "forward_ms_p95",
        "peak_memory_mb",
        "verify_k_marginal_ms",
        "kv_bytes_read_estimate",
        "weight_bytes_read_estimate",
        "prefix_token_count",
        "seed",
        "n_warmup_reps_discarded",
        "n_timed_reps",
    }
    assert expected_keys.issubset(row)
    assert row["verify_k"] == 4
    assert row["candidate_token_count"] == 4
    assert row["prefix_token_count"] == 113
    assert row["forward_ms_p50"] == 120.0  # median of [100..140 step 10]
    assert row["verify_k_marginal_ms"] == 25.0  # 120 - 95
    assert row["weight_bytes_read_estimate"] == 13_500_000_000
    assert row["seed"] == 42
    assert row["repo"] == "mlx-community/Qwen3.5-27B-4bit"
    assert row["n_timed_reps"] == 5


def test_measurement_to_jsonl_row_marginal_zero_at_baseline() -> None:
    """For ``verify_k=1`` the row's marginal must equal 0 because
    the baseline IS k=1's p50. Pin this so a future helper-function
    edit cannot silently introduce a sign error."""
    measurement = _MeasureResult(
        verify_k=1,
        samples_ms=[100.0, 100.0, 100.0],
        peak_memory_mb=15000.0,
        prefix_token_count=114,
        candidate_token_count=1,
    )
    row = measurement_to_jsonl_row(
        measurement=measurement,
        baseline_p50_ms=100.0,
        weight_bytes_read=10_000,
        seed=0,
        repo="x/y",
    )
    assert row["verify_k_marginal_ms"] == 0.0
    assert row["candidate_token_count"] == 1


def test_measurement_to_jsonl_row_records_candidate_token_count_separately() -> None:
    """``candidate_token_count`` is stored as a distinct column from
    ``verify_k`` even though they are equal by construction. The
    field documents the exact size of the timed forward (no
    tokenizer drift) so a future change that drifts them apart
    surfaces here rather than silently in the analysis."""
    measurement = _MeasureResult(
        verify_k=8,
        samples_ms=[200.0],
        peak_memory_mb=15000.0,
        prefix_token_count=113,
        candidate_token_count=8,
    )
    row = measurement_to_jsonl_row(
        measurement=measurement,
        baseline_p50_ms=180.0,
        weight_bytes_read=0,
        seed=0,
        repo="x/y",
    )
    assert row["verify_k"] == row["candidate_token_count"] == 8


# ---------------------------------------------------------------------------
# render_markdown_report
# ---------------------------------------------------------------------------


def test_render_markdown_report_has_header_and_one_row_per_k() -> None:
    rows = [
        {
            "repo": "x/y",
            "verify_k": k,
            "candidate_token_count": k,
            "forward_ms_p50": 100.0 + k,
            "forward_ms_p95": 110.0 + k,
            "peak_memory_mb": 15336.5,
            "verify_k_marginal_ms": float(k - 1),
            "kv_bytes_read_estimate": 1_000_000 * k,
            "weight_bytes_read_estimate": 13_500_000_000,
            "prefix_token_count": 113,
            "seed": 0,
            "n_warmup_reps_discarded": 2,
            "n_timed_reps": 10,
        }
        for k in (1, 2, 4, 8)
    ]
    md = render_markdown_report(rows, repo="x/y")
    assert "# Target-Verify Microbench Report" in md
    assert "**Repo**: `x/y`" in md
    table_rows = [
        line
        for line in md.splitlines()
        if line.startswith("| 1 ")
        or line.startswith("| 2 ")
        or line.startswith("| 4 ")
        or line.startswith("| 8 ")
    ]
    assert len(table_rows) == 4
    # Marginal column carries signed value.
    assert "+0.00" in md  # k=1 row marginal == 0
    assert "+7.00" in md  # k=8 row marginal == 7


# ---------------------------------------------------------------------------
# _parse_verify_ks
# ---------------------------------------------------------------------------


def test_parse_verify_ks_accepts_canonical_set() -> None:
    assert _parse_verify_ks("1,2,4,8") == (1, 2, 4, 8)


def test_parse_verify_ks_strips_whitespace() -> None:
    assert _parse_verify_ks(" 1, 2 , 4 ,8 ") == (1, 2, 4, 8)


def test_parse_verify_ks_rejects_empty_string() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=r"may not be empty"):
        _parse_verify_ks("")


def test_parse_verify_ks_rejects_non_integer() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=r"is not an integer"):
        _parse_verify_ks("1,foo,2")


def test_parse_verify_ks_rejects_out_of_range() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=r"outside"):
        _parse_verify_ks("0,1,2")
    with pytest.raises(argparse.ArgumentTypeError, match=r"outside"):
        _parse_verify_ks(f"1,2,{MAX_VERIFY_K + 1}")


def test_parse_verify_ks_rejects_duplicates() -> None:
    """``run`` keys measurements by verify_k in a dict; duplicates
    would silently overwrite each other and then emit duplicate
    JSONL rows with shared underlying measurements. Reject at parse
    time so the contract is loud rather than silently corrupting
    the artefact."""
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"appears more than once"
    ):
        _parse_verify_ks("1,1,2")
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"appears more than once"
    ):
        _parse_verify_ks("1,2,4,8,4")


# ---------------------------------------------------------------------------
# Analytic estimators (stub adapter, no real-model load)
# ---------------------------------------------------------------------------


@dataclass
class _StubKVLayout:
    num_layers: int
    n_kv_heads: int
    head_dim: int
    dtype: Any  # mx.Dtype-like; the helper reads ``.size``


@dataclass
class _StubDtype:
    size: int  # bytes per element


@dataclass
class _StubConfig:
    num_layers: int = 64
    hidden_size: int = 5120
    num_parameters: int = 27_000_000_000


class _StubAdapter:
    def __init__(
        self,
        *,
        num_parameters: int | None = 27_000_000_000,
        num_layers: int = 64,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        dtype_bytes: int = 2,
    ) -> None:
        if num_parameters is None:
            cfg = _StubConfig()
            del cfg.num_parameters  # type: ignore[misc]
            self.config = _StubConfig(num_layers=num_layers)
        else:
            self.config = _StubConfig(
                num_layers=num_layers,
                num_parameters=num_parameters,
            )
        self._layout = _StubKVLayout(
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=_StubDtype(size=dtype_bytes),
        )

    def kv_layout(self) -> _StubKVLayout:
        return self._layout


def test_estimate_weight_bytes_read_uses_num_parameters_at_4bit() -> None:
    """27e9 params × 0.5 B/param = 13.5e9 B — matches PLAN
    `bandwidth ceiling` anchor (`plans/P6_OPENING.md` §1.2)."""
    adapter = _StubAdapter(num_parameters=27_000_000_000)
    assert estimate_weight_bytes_read(adapter) == 13_500_000_000


def test_estimate_weight_bytes_read_falls_back_when_num_parameters_missing() -> None:
    """Missing ``num_parameters`` should not crash — the fallback
    structural estimate is coarse but well-defined."""

    class _ConfigSansParams:
        num_layers = 64
        hidden_size = 5120
        # num_parameters intentionally absent.

    class _AdapterSansParams:
        config = _ConfigSansParams()

    bytes_read = estimate_weight_bytes_read(_AdapterSansParams())
    assert bytes_read > 0


def test_estimate_kv_bytes_read_matches_formula() -> None:
    """Formula: ``2 × seqlen × num_layers × n_kv_heads × head_dim ×
    dtype_bytes``. With Qwen3.5-27B (64 × 8 × 128 fp16) at 128
    tokens, expect 128 × 64 × 8 × 128 × 2 × 2 = 33_554_432 B."""
    adapter = _StubAdapter(
        num_layers=64, n_kv_heads=8, head_dim=128, dtype_bytes=2
    )
    assert estimate_kv_bytes_read(adapter, 128) == (
        2 * 128 * 64 * 8 * 128 * 2
    )


def test_estimate_kv_bytes_read_zero_at_zero_seqlen() -> None:
    """A zero seqlen anchors the bytes-per-step formula at 0 (KV
    is empty before the first prefill)."""
    adapter = _StubAdapter()
    assert estimate_kv_bytes_read(adapter, 0) == 0


def test_estimate_kv_bytes_read_dtype_bytes_default_two_when_size_absent() -> None:
    """If the dtype object lacks a ``.size`` attribute, fall back
    to 2 (fp16)."""

    class _DtypeNoSize:
        pass

    class _LayoutNoDtypeSize:
        num_layers = 4
        n_kv_heads = 2
        head_dim = 64
        dtype = _DtypeNoSize()

    class _AdapterNoDtypeSize:
        def kv_layout(self) -> _LayoutNoDtypeSize:
            return _LayoutNoDtypeSize()

    # 2 × 10 × 4 × 2 × 64 × 2 (fallback) = 20480
    assert estimate_kv_bytes_read(_AdapterNoDtypeSize(), 10) == 20480


# ---------------------------------------------------------------------------
# run() — validation only (real-model path needs the cache)
# ---------------------------------------------------------------------------


def test_run_rejects_verify_ks_without_one() -> None:
    """``run`` must refuse to start if ``verify_ks`` lacks 1 — the
    marginal column is defined relative to k=1, so without it the
    baseline is undefined. Validation must fire BEFORE the model
    loads (otherwise a typo in --verify-ks costs a multi-GB
    download)."""
    config = _RunConfig(
        repo="anything/anything",
        verify_ks=(2, 4, 8),
        warmup_reps=2,
        timed_reps=10,
        out_jsonl=Path("/tmp/should_not_be_written.jsonl"),
        out_md=Path("/tmp/should_not_be_written.md"),
    )
    with pytest.raises(ValueError, match=r"must include 1"):
        run(config)


# ---------------------------------------------------------------------------
# write_artefacts — JSONL + Markdown sibling write
# ---------------------------------------------------------------------------


def _row(verify_k: int) -> dict[str, Any]:
    return {
        "repo": "x/y",
        "verify_k": verify_k,
        "candidate_token_count": verify_k,
        "forward_ms_p50": 100.0 + verify_k,
        "forward_ms_p95": 110.0 + verify_k,
        "peak_memory_mb": 15336.5,
        "verify_k_marginal_ms": float(verify_k - 1),
        "kv_bytes_read_estimate": 1_000_000 * verify_k,
        "weight_bytes_read_estimate": 13_500_000_000,
        "prefix_token_count": 113,
        "seed": 0,
        "n_warmup_reps_discarded": 2,
        "n_timed_reps": 10,
    }


def test_write_artefacts_creates_parent_dirs(tmp_path: Path) -> None:
    """Both ``out_jsonl.parent`` and ``out_md.parent`` are created
    up front. Pin this so a future refactor cannot regress to
    "JSONL written, Markdown sibling fails because its directory
    does not exist" — that would leave a half-completed artefact
    on disk and silently break the P-6.0.5 baseline pattern."""
    out_jsonl = tmp_path / "nested_jsonl_dir" / "x.jsonl"
    out_md = tmp_path / "completely_separate_md_dir" / "x.md"
    rows = [_row(k) for k in (1, 2, 4, 8)]

    write_artefacts(rows, out_jsonl=out_jsonl, out_md=out_md, repo="x/y")

    assert out_jsonl.parent.is_dir()
    assert out_md.parent.is_dir()
    assert out_jsonl.is_file()
    assert out_md.is_file()


def test_write_artefacts_emits_one_jsonl_row_per_input(tmp_path: Path) -> None:
    """Each row in the input list yields exactly one JSON line."""
    import json as _json

    out_jsonl = tmp_path / "x.jsonl"
    out_md = tmp_path / "x.md"
    rows = [_row(k) for k in (1, 2, 4, 8)]

    write_artefacts(rows, out_jsonl=out_jsonl, out_md=out_md, repo="x/y")

    lines = out_jsonl.read_text().splitlines()
    assert len(lines) == 4
    parsed = [_json.loads(line) for line in lines]
    assert [r["verify_k"] for r in parsed] == [1, 2, 4, 8]


def test_write_artefacts_idempotent_on_existing_dirs(tmp_path: Path) -> None:
    """If the parent dirs already exist, ``write_artefacts`` must
    not raise — ``mkdir(exist_ok=True)`` is the contract."""
    out_jsonl = tmp_path / "x.jsonl"
    out_md = tmp_path / "x.md"
    write_artefacts([_row(1)], out_jsonl=out_jsonl, out_md=out_md, repo="x/y")
    # Second call into the same paths must succeed (overwrites files).
    write_artefacts(
        [_row(1), _row(2)],
        out_jsonl=out_jsonl,
        out_md=out_md,
        repo="x/y",
    )
    assert len(out_jsonl.read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# Mocked positive-path coverage of the real-model entry points
#
# These tests patch the heavy MLX-dependent imports so the load-bearing
# semantics that make the P-6.0.5 sub-unit 7 measurement valid can be
# pinned without a 16 GB model load:
#
#   * ``_measure_one_k_real`` per-rep call sequence: fresh
#     ``make_prompt_cache`` per rep, prefix forward (untimed) before
#     candidate forward (timed), ``perf_counter`` brackets only the
#     candidate forward.
#   * ``run`` orchestration with reordered ``verify_ks``: input order
#     preserved in output rows; baseline lookup uses ``verify_k == 1``,
#     not ``measurements[0]``.
# ---------------------------------------------------------------------------


class _FakeArray:
    """Minimal ``mx.array`` stand-in: only the ``.size`` attribute the
    fake forward reads is provided."""

    def __init__(self, size: int) -> None:
        self.size = size


class _FakeMx:
    """Replace the ``mx`` binding inside ``target_verify`` so no real
    MLX calls happen during the per-rep sequence test. The fake
    captures every relevant call into the test's event log."""

    int32 = "int32"
    _events: list[Any] = []

    @classmethod
    def array(cls, data: Any, dtype: Any = None) -> _FakeArray:
        size = len(data) if hasattr(data, "__len__") else 0
        return _FakeArray(size=size)

    @classmethod
    def eval(cls, *args: Any) -> None:
        cls._events.append(("eval",))

    @classmethod
    def reset_peak_memory(cls) -> None:
        cls._events.append(("reset_peak",))

    @classmethod
    def get_peak_memory(cls) -> int:
        cls._events.append(("get_peak",))
        return 1_000_000  # 1 MB → 1.0 in MB after / 1e6


def test_measure_one_k_real_per_rep_call_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the per-rep contract that the negative validation test
    cannot: each rep gets a fresh ``make_prompt_cache``, prefix
    forward fires (untimed) before the candidate forward, and the
    two ``perf_counter`` calls bracket **only** the candidate
    forward — not the prefix prefill.
    """
    from silica.bench.microbench import target_verify

    events: list[Any] = []
    _FakeMx._events = events

    cache_counter = [0]

    class _FakeMlxCache:
        @staticmethod
        def make_prompt_cache(model: Any) -> str:
            cache_counter[0] += 1
            cid = f"cache_{cache_counter[0]}"
            events.append(("make_cache", cid))
            return cid

    def fake_forward(model: Any, tokens: Any, cache_list: Any) -> Any:
        events.append(("forward", cache_list, tokens.size))
        return object()

    perf_counter_calls = [0]

    def fake_perf_counter() -> float:
        perf_counter_calls[0] += 1
        events.append(("perf_counter", perf_counter_calls[0]))
        return perf_counter_calls[0] * 0.001

    class _FakeTimeModule:
        perf_counter = staticmethod(fake_perf_counter)

    monkeypatch.setattr(target_verify, "mx", _FakeMx)
    monkeypatch.setattr(target_verify, "mlx_cache", _FakeMlxCache)
    monkeypatch.setattr(target_verify, "_silica_forward", fake_forward)
    monkeypatch.setattr(target_verify, "time", _FakeTimeModule)

    result = target_verify._measure_one_k_real(
        model="FAKE_MODEL",
        prefix_token_ids=[1, 2, 3],  # 3-token prefix
        verify_k=4,
        warmup_reps=1,
        timed_reps=2,
    )

    # 3 reps × 9 events per rep (cache, prefix-fwd, eval, reset_peak,
    # perf_counter, candidate-fwd, eval, perf_counter, get_peak).
    assert len(events) == 3 * 9, f"unexpected event count: {len(events)}"

    expected_per_rep_types = [
        "make_cache",
        "forward",  # prefix
        "eval",
        "reset_peak",
        "perf_counter",  # t_start
        "forward",  # candidate
        "eval",
        "perf_counter",  # t_end
        "get_peak",
    ]
    for rep_idx in range(3):
        rep_events = events[rep_idx * 9 : (rep_idx + 1) * 9]
        assert [e[0] for e in rep_events] == expected_per_rep_types, (
            f"rep {rep_idx} sequence mismatch: {[e[0] for e in rep_events]}"
        )

        # Cache is unique per rep (fresh state each iteration).
        cache_id = rep_events[0][1]
        # Both forwards in this rep use the same cache_list.
        prefix_fwd, candidate_fwd = rep_events[1], rep_events[5]
        assert prefix_fwd[1] == cache_id
        assert candidate_fwd[1] == cache_id
        # Prefix forward sees the prefix length; candidate forward
        # sees verify_k.
        assert prefix_fwd[2] == 3
        assert candidate_fwd[2] == 4

    # Across reps, every cache id is unique — no leak between iterations.
    cache_ids = [
        e[1] for e in events if e[0] == "make_cache"
    ]
    assert len(set(cache_ids)) == 3

    # Result has 2 timed samples (warmup discarded).
    assert len(result.samples_ms) == 2
    assert result.verify_k == 4
    assert result.candidate_token_count == 4
    assert result.prefix_token_count == 3


def test_run_baseline_lookup_uses_verify_k_one_not_first_in_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin: ``run`` looks up the marginal baseline by
    ``verify_k == 1``, not ``measurements[0]``. Protects against
    a future regression where reordering ``--verify-ks`` silently
    corrupts the marginal column.

    Construction: ``verify_ks=(2, 1)`` puts k=1 LAST. The fake
    measurement returns ``samples_ms = [k * 1.0] * timed_reps``,
    so ``p50[k=1] = 1.0`` and ``p50[k=2] = 2.0``. Correct
    implementation: ``marginal[k=2] = 2.0 - 1.0 = 1.0`` and
    ``marginal[k=1] = 0.0``. A buggy implementation that uses
    ``measurements[0]`` (= k=2) as baseline would produce
    ``marginal[k=2] = 0.0`` and ``marginal[k=1] = -1.0``.
    """
    from silica.bench.microbench import target_verify

    @dataclass
    class _StubKVLayout:
        num_layers: int = 4
        n_kv_heads: int = 2
        head_dim: int = 16
        dtype: Any = None

    class _StubDtype:
        size = 2

    class _StubConfig:
        num_parameters = 1_000_000
        num_layers = 4
        hidden_size = 32

    class _StubTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(50))

    class _StubAdapter:
        _model = "FAKE_MODEL"
        config = _StubConfig()

        def tokenizer(self) -> _StubTokenizer:
            return _StubTokenizer()

        def kv_layout(self) -> _StubKVLayout:
            return _StubKVLayout(dtype=_StubDtype())

    monkeypatch.setattr(
        target_verify,
        "adapter_for_repo",
        lambda repo: (_StubAdapter(), None),
    )

    def fake_measure(
        *,
        model: Any,
        prefix_token_ids: list[int],
        verify_k: int,
        warmup_reps: int,
        timed_reps: int,
    ) -> target_verify._MeasureResult:
        return target_verify._MeasureResult(
            verify_k=verify_k,
            samples_ms=[verify_k * 1.0] * timed_reps,
            peak_memory_mb=15000.0,
            prefix_token_count=len(prefix_token_ids),
            candidate_token_count=verify_k,
        )

    monkeypatch.setattr(target_verify, "_measure_one_k_real", fake_measure)

    config = target_verify._RunConfig(
        repo="x/y",
        verify_ks=(2, 1),  # k=1 LAST
        warmup_reps=1,
        timed_reps=3,
        out_jsonl=tmp_path / "out.jsonl",
        out_md=tmp_path / "out.md",
    )
    rows = target_verify.run(config)

    # Output row order matches input order (k=2 first), not sorted.
    assert [r["verify_k"] for r in rows] == [2, 1]

    row_k1 = next(r for r in rows if r["verify_k"] == 1)
    row_k2 = next(r for r in rows if r["verify_k"] == 2)
    assert row_k1["forward_ms_p50"] == 1.0
    assert row_k2["forward_ms_p50"] == 2.0

    # The contract: baseline is k=1's p50, not measurements[0]'s p50.
    assert row_k1["verify_k_marginal_ms"] == 0.0
    assert row_k2["verify_k_marginal_ms"] == 1.0  # NOT 0.0

    # candidate_token_count tracks verify_k exactly, regardless of order.
    assert row_k1["candidate_token_count"] == 1
    assert row_k2["candidate_token_count"] == 2

    # Artefacts written to tmp_path (mkdir already exercised in
    # write_artefacts tests; here just confirm the orchestration
    # reaches that step).
    assert (tmp_path / "out.jsonl").is_file()
    assert (tmp_path / "out.md").is_file()
