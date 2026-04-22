"""Tests for silica.bench.oracles.decode_tok_s_with_prefix_hit_oracle.

P-5-A.3b scope: the oracle pure function, exercised with canned
``(tokens, token_ts_ms) + context`` inputs so no Engine / adapter /
model load is needed. End-to-end runner integration through
``_run_prefix_hit_decode`` is covered separately by the gated
acceptance test that P-5-A.3c adds against a real Qwen3.5-0.8B load.

The oracle's job is:
1. Validate row shape (non-empty, valid token ids).
2. Reject rows where row 1 has < 2 tokens (inter-token interval
   formula needs ≥ 2 samples).
3. Reject scenarios where prefix_cache_hits == 0 (row 1 ran the
   miss path — the measurement claim is wrong).
4. Report ``row1_decode_tok_s = (N - 1) / (t_last - t_first)`` in
   metadata + ``row1_first_token_ms`` separately.

The 0.85× BlockTQ-vs-fp16 ratio gate is NOT in this oracle — that
is a comparison between two scenario results, landed in the A.3c
acceptance test. This oracle just reports an honest row-1 decode
tok/s.
"""

from __future__ import annotations

from typing import Any

from silica.bench.oracles import decode_tok_s_with_prefix_hit_oracle
from silica.bench.scenario import OracleKind, Scenario, Workload


def _stub_scenario(
    *,
    prompts: tuple[str, ...] = ("hello", "hello"),
    max_batch_size: int = 1,
    prefix_cache: bool = True,
    kv_codec: str | None = "fp16",
) -> Scenario:
    return Scenario(
        id="stub-prefix-hit-decode",
        repo="stub/stub",
        workload=Workload(
            name="stub",
            prompts=prompts,
            max_tokens=8,
            max_batch_size=max_batch_size,
            prefix_cache=prefix_cache,
            kv_codec=kv_codec,
        ),
        oracle=OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT,
    )


def _happy_collected(
    *,
    row1_ts_ms: list[float] | None = None,
    row0_ts_ms: list[float] | None = None,
    vocab_size: int = 1024,
) -> tuple[tuple[dict[int, list[int]], dict[int, list[float]]], dict[str, Any]]:
    """Hand-built collector output: 8 valid tokens per row, known
    timestamps. Default row-1 timestamps give 100 ms first-token + 7
    × 20 ms steady-state intervals → 7 tokens over 140 ms →
    50 tok/s exactly."""
    if row1_ts_ms is None:
        row1_ts_ms = [100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0]
    if row0_ts_ms is None:
        row0_ts_ms = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    n0 = len(row0_ts_ms)
    n1 = len(row1_ts_ms)
    tokens = {
        0: list(range(1, n0 + 1)),
        1: list(range(1, n1 + 1)),
    }
    ts = {0: row0_ts_ms, 1: row1_ts_ms}
    context = {"vocab_size": vocab_size, "prefix_cache_hits": 2}
    return (tokens, ts), context


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_reports_row1_decode_tok_s() -> None:
    collected, context = _happy_collected()
    scn = _stub_scenario()
    ok, reason, md = decode_tok_s_with_prefix_hit_oracle(scn, collected, context)
    assert ok is True
    assert reason is None
    # (8-1) / ((240 - 100)/1000) = 7 / 0.14 = 50 tok/s exactly.
    assert abs(md["row1_decode_tok_s"] - 50.0) < 1e-6
    assert md["row1_first_token_ms"] == 100.0
    assert md["row1_tokens"] == 8
    assert md["row0_tokens"] == 8
    assert md["prefix_cache_hits"] == 2


def test_row0_decode_tok_s_populated_when_available() -> None:
    collected, context = _happy_collected()
    scn = _stub_scenario()
    _, _, md = decode_tok_s_with_prefix_hit_oracle(scn, collected, context)
    # Row 0 ts: 10..80 → 7 tokens over 70 ms → 100 tok/s.
    assert md["row0_decode_tok_s"] is not None
    assert abs(md["row0_decode_tok_s"] - 100.0) < 1e-6
    assert md["row0_first_token_ms"] == 10.0


def test_excludes_first_token_latency_from_decode_tok_s() -> None:
    """Verifies the formula is ``(N-1)/(t_last - t_first)``, not
    ``N/t_last`` — the latter would conflate seeded-admission cost
    with steady-state decode throughput."""
    # Artificial: huge first-token latency (1000 ms), fast steady-
    # state (1 ms intervals × 7 → 7 ms).
    row1 = [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0]
    collected, context = _happy_collected(row1_ts_ms=row1)
    ok, _, md = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), collected, context
    )
    assert ok is True
    # (8-1) / (1007 - 1000)/1000 = 7 / 0.007 = 1000 tok/s.
    assert abs(md["row1_decode_tok_s"] - 1000.0) < 1e-3
    # First-token latency reported separately and is still large.
    assert md["row1_first_token_ms"] == 1000.0


# ---------------------------------------------------------------------------
# Prefix-cache hit gate
# ---------------------------------------------------------------------------


def test_rejects_zero_prefix_cache_hits() -> None:
    """If row 1 ran the miss path, the measurement is not what the
    scenario claims. Oracle must reject rather than silently
    reporting a miss-path decode tok/s as the prefix-hit figure."""
    collected, context = _happy_collected()
    context["prefix_cache_hits"] = 0
    ok, reason, md = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), collected, context
    )
    assert ok is False
    assert reason == "decode_tok_s_prefix_cache_never_hit"
    assert md["prefix_cache_hits"] == 0


def test_accepts_any_positive_prefix_cache_hits() -> None:
    """Non-zero hits is sufficient — the number grows with the
    number of hit blocks, which depends on prompt length and
    block_size; the oracle does not pin a specific value."""
    collected, context = _happy_collected()
    context["prefix_cache_hits"] = 1
    ok, _, _ = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), collected, context
    )
    assert ok is True


# ---------------------------------------------------------------------------
# Row-shape validation
# ---------------------------------------------------------------------------


def test_rejects_row_1_empty() -> None:
    (tokens, ts), context = _happy_collected()
    tokens[1] = []
    ts[1] = []
    ok, reason, _ = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), (tokens, ts), context
    )
    assert ok is False
    assert reason == "decode_tok_s_row_1_no_tokens_emitted"


def test_rejects_row_1_only_one_token() -> None:
    """Inter-token interval formula needs at least two samples;
    a 1-token row would divide by zero and silently report ∞.
    Reject explicitly."""
    (tokens, ts), context = _happy_collected()
    tokens[1] = [42]
    ts[1] = [100.0]
    ok, reason, md = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), (tokens, ts), context
    )
    assert ok is False
    assert reason == "decode_tok_s_row_1_needs_at_least_2_tokens_for_interval"
    assert md["row1_tokens"] == 1


def test_rejects_row_0_empty() -> None:
    (tokens, ts), context = _happy_collected()
    tokens[0] = []
    ts[0] = []
    ok, reason, _ = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), (tokens, ts), context
    )
    assert ok is False
    assert reason == "decode_tok_s_row_0_no_tokens_emitted"


def test_rejects_out_of_vocab_token() -> None:
    (tokens, ts), context = _happy_collected(vocab_size=1024)
    tokens[1][3] = 999_999  # outside vocab
    ok, reason, _ = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), (tokens, ts), context
    )
    assert ok is False
    assert "out_of_vocab" in (reason or "")


def test_rejects_non_int_token() -> None:
    (tokens, ts), context = _happy_collected()
    # The runner collector guarantees int token ids, but an oracle
    # contract test still checks the type guard — bypass mypy with
    # an any-typed list assignment so the type-checker doesn't reject
    # the fault injection.
    bad_tokens: list[Any] = list(tokens[1])
    bad_tokens[2] = "not an int"
    tokens[1] = bad_tokens
    ok, reason, _ = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), (tokens, ts), context
    )
    assert ok is False
    assert "not_int" in (reason or "")


# ---------------------------------------------------------------------------
# Context validation
# ---------------------------------------------------------------------------


def test_rejects_missing_context() -> None:
    collected, _ = _happy_collected()
    ok, reason, _ = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), collected, None
    )
    assert ok is False
    assert reason == "decode_tok_s_missing_context"


def test_rejects_context_missing_required_keys() -> None:
    collected, _ = _happy_collected()
    ok, reason, md = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), collected, {"vocab_size": 1024}  # no prefix_cache_hits
    )
    assert ok is False
    assert reason == "decode_tok_s_context_missing_required_keys"
    assert "vocab_size" in md["keys_present"]


def test_rejects_collected_shape_mismatch() -> None:
    """Collected must be a 2-tuple of dicts. Any other shape → fail."""
    scn = _stub_scenario()
    context = {"vocab_size": 1024, "prefix_cache_hits": 2}
    ok, reason, _ = decode_tok_s_with_prefix_hit_oracle(
        scn, "not a tuple", context
    )
    assert ok is False
    assert reason == "decode_tok_s_collected_shape_mismatch"


# ---------------------------------------------------------------------------
# Edge case: identical timestamps (nonpositive interval)
# ---------------------------------------------------------------------------


def test_rejects_row_1_zero_interval() -> None:
    """If the clock never advances across row 1's tokens (broken
    measurement), interval is 0 → division would go to ∞. Reject."""
    (tokens, ts), context = _happy_collected()
    ts[1] = [50.0] * 8  # all identical
    ok, reason, _ = decode_tok_s_with_prefix_hit_oracle(
        _stub_scenario(), (tokens, ts), context
    )
    assert ok is False
    assert reason == "decode_tok_s_row_1_nonpositive_interval"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_oracle_registered_in_ORACLES() -> None:
    """Every OracleKind must map to a function in the ORACLES dict;
    the runner's dispatch relies on this."""
    from silica.bench.oracles import ORACLES

    assert OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT in ORACLES
    assert ORACLES[OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT] is (
        decode_tok_s_with_prefix_hit_oracle
    )
