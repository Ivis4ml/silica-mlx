"""Unit tests for :func:`silica.bench.oracles.warm_decode_oracle`.

Pure-function validation on the ``(collected, context)`` pair the
runner's ``_run_warm_decode`` produces. Does not touch the runner,
adapter, or engine — synthetic per-row timestamp arrays drive the
two-stage warm-up rule + steady-state metric directly.

Pinned behaviours:
1. Successful B=1 measurement with stable steady state.
2. Successful B>1 aggregate across rows.
3. Two-stage warm-up rule: discard at least ``warmup_min_steps`` AND
   advance until rolling-window std/mean falls below threshold —
   later wins.
4. Too-few-tokens rejection when a row cannot fit warm-up + window
   + measurement_steps_min in the emitted token count.
5. Warmup-never-stabilizes rejection when the timeline stays noisy.
6. Out-of-vocab token rejection.
7. Missing-context-key rejection.
8. Collected-shape-mismatch rejection.
9. Aggregate metric correctly weights total measurement decodes by
   the union of per-row warm-window walls.
"""

from __future__ import annotations

from typing import Any

from silica.bench.oracles import ORACLES, warm_decode_oracle
from silica.bench.scenario import OracleKind, Scenario, Workload


def _make_scenario() -> Scenario:
    """Minimal scenario stub — the oracle does not read scenario fields
    other than for diagnostic logging it does not currently emit.
    """
    return Scenario(
        id="test-warm-decode",
        repo="dummy/repo",
        workload=Workload(
            name="x", prompts=("p",), max_tokens=288
        ),
        oracle=OracleKind.WARM_DECODE,
    )


_DEFAULT_CONTEXT: dict[str, Any] = {
    "vocab_size": 200_000,
    "warmup_min_steps": 32,
    "warmup_rolling_window": 16,
    "warmup_rel_std_threshold": 0.05,
    "measurement_steps_min": 64,
}


def _build_timeline(
    cold_ttft_ms: float,
    warmup_intervals: list[float],
    steady_intervals: list[float],
) -> list[float]:
    """Compose a per-row timeline starting at cold_ttft_ms then
    accumulating the given inter-token intervals.
    """
    ts = [cold_ttft_ms]
    for d in warmup_intervals:
        ts.append(ts[-1] + d)
    for d in steady_intervals:
        ts.append(ts[-1] + d)
    return ts


def _stable(n: int, ms: float = 6.5, jitter: float = 0.05) -> list[float]:
    """``n`` intervals of mean ``ms`` with low jitter so rel_std stays
    well below the 0.05 threshold the default context uses.
    """
    return [ms - jitter + (2 * jitter * (i % 2)) for i in range(n)]


def _noisy(n: int, lo: float = 8.0, hi: float = 15.0) -> list[float]:
    """``n`` intervals alternating between ``lo`` and ``hi`` so
    rel_std stays well above the 0.05 threshold (the warm-up rule
    keeps advancing).
    """
    return [lo if i % 2 == 0 else hi for i in range(n)]


def test_b1_steady_state_measurement() -> None:
    """Stable 6.5 ms steady-state after a 40-step noisy warm-up gives
    the expected ~154 tok/s warm aggregate (1000 / 6.5)."""
    ts = _build_timeline(
        cold_ttft_ms=2400.0,
        warmup_intervals=_noisy(40),
        steady_intervals=_stable(200),
    )
    tokens = [42] * len(ts)
    collected = ({0: tokens}, {0: ts})

    ok, reason, meta = warm_decode_oracle(
        _make_scenario(), collected, _DEFAULT_CONTEXT
    )
    assert ok, f"reason={reason}, meta={meta}"
    expected = 1000.0 / 6.5
    err = abs(meta["decode_tok_s_warm_aggregate"] - expected) / expected
    assert err < 0.02, (
        f"warm aggregate too far from expected: got "
        f"{meta['decode_tok_s_warm_aggregate']:.3f}, expected "
        f"{expected:.3f} (err={err:.3%})"
    )
    row = meta["rows"][0]
    # Warm-up boundary should land at or after warmup_min_steps=32 and
    # near the noisy-stable transition (40).
    assert row["warmup_steps_used"] >= 32
    assert row["warmup_steps_used"] <= 50
    assert row["measurement_steps"] >= 64
    assert row["cold_ttft_ms"] == 2400.0


def test_bgt1_aggregate() -> None:
    """B>1 aggregate weights all rows' measurement decodes by the
    union wall (max(t_last) - min(t_first_meas))."""
    # Two rows with identical 6.5 ms steady-state shapes, started
    # back-to-back. Aggregate decode_tok_s should be roughly 2 ×
    # per-row, since both rows emit decodes during the same window.
    ts0 = _build_timeline(
        cold_ttft_ms=100.0,
        warmup_intervals=_noisy(40),
        steady_intervals=_stable(200, ms=6.5),
    )
    # Row 1 starts slightly later (cold TTFT 110 ms) but has the
    # same shape, so per-row tok/s match.
    ts1 = _build_timeline(
        cold_ttft_ms=110.0,
        warmup_intervals=_noisy(40),
        steady_intervals=_stable(200, ms=6.5),
    )
    tokens = {0: [1] * len(ts0), 1: [2] * len(ts1)}
    token_ts_ms = {0: ts0, 1: ts1}
    ok, reason, meta = warm_decode_oracle(
        _make_scenario(), (tokens, token_ts_ms), _DEFAULT_CONTEXT
    )
    assert ok, f"reason={reason}, meta={meta}"
    per_row = meta["decode_tok_s_warm_per_row_mean"]
    # Per-row mean rate ≈ 1000 / 6.5 ≈ 154 tok/s.
    assert 140.0 < per_row < 170.0, per_row
    # Aggregate is bounded above by 2 × per_row (two rows in parallel)
    # and bounded below by per_row (one row alone). On real hardware
    # batched B=2 typically lands at 1.7-1.95×.
    aggregate = meta["decode_tok_s_warm_aggregate"]
    assert per_row <= aggregate <= 2.0 * per_row + 1.0, (
        f"aggregate {aggregate} not in [{per_row}, 2× {per_row}]"
    )
    assert len(meta["rows"]) == 2


def test_warmup_min_steps_floor_enforced() -> None:
    """Even if the rolling-window std drops below threshold before
    step 32, the boundary stays at warmup_min_steps=32. This pins the
    "later wins" rule from the docstring.
    """
    # Whole timeline is rock-stable (0.05% rel_std) — rolling rule
    # would fire on the first window if the floor were 0.
    ts = _build_timeline(
        cold_ttft_ms=10.0,
        warmup_intervals=[],
        steady_intervals=_stable(200, ms=6.5, jitter=0.001),
    )
    tokens = [3] * len(ts)
    ok, reason, meta = warm_decode_oracle(
        _make_scenario(),
        ({0: tokens}, {0: ts}),
        _DEFAULT_CONTEXT,
    )
    assert ok, f"reason={reason}, meta={meta}"
    row = meta["rows"][0]
    # Floor = warmup_min_steps = 32; with stable timeline the rolling
    # rule passes immediately at step 32, so the boundary is exactly 32.
    assert row["warmup_steps_used"] == 32.0


def test_rolling_window_extends_warmup_past_floor() -> None:
    """When the timeline stays noisy past step 32, the rolling-window
    rule extends the warm-up boundary past the floor."""
    # 80 noisy steps before stabilizing → boundary should land near 80.
    ts = _build_timeline(
        cold_ttft_ms=10.0,
        warmup_intervals=_noisy(80),
        steady_intervals=_stable(160),
    )
    tokens = [4] * len(ts)
    ok, _, meta = warm_decode_oracle(
        _make_scenario(),
        ({0: tokens}, {0: ts}),
        _DEFAULT_CONTEXT,
    )
    assert ok
    row = meta["rows"][0]
    assert row["warmup_steps_used"] >= 70, row["warmup_steps_used"]
    assert row["warmup_steps_used"] <= 90, row["warmup_steps_used"]


def test_too_few_tokens_rejected() -> None:
    """Row that emits fewer tokens than warmup + window + min + 1
    fails with a structured reason, not a silent infinity."""
    # 32 + 16 + 64 + 1 = 113 required; emit 80 → too few.
    ts = _build_timeline(
        cold_ttft_ms=10.0,
        warmup_intervals=[],
        steady_intervals=_stable(79, ms=6.5),
    )
    tokens = [5] * len(ts)
    ok, reason, _ = warm_decode_oracle(
        _make_scenario(),
        ({0: tokens}, {0: ts}),
        _DEFAULT_CONTEXT,
    )
    assert not ok
    assert reason is not None
    assert "too_few_tokens" in reason


def test_warmup_never_stabilizes_rejected() -> None:
    """Persistently noisy timeline never satisfies the rolling-window
    rule; oracle fails with a stabilization reason rather than picking
    an arbitrary boundary."""
    # Entire 200-step timeline is _noisy(); rel_std stays around 0.3.
    ts = _build_timeline(
        cold_ttft_ms=10.0,
        warmup_intervals=[],
        steady_intervals=_noisy(200),
    )
    tokens = [6] * len(ts)
    ok, reason, _ = warm_decode_oracle(
        _make_scenario(),
        ({0: tokens}, {0: ts}),
        _DEFAULT_CONTEXT,
    )
    assert not ok
    assert reason is not None
    assert "warmup_did_not_stabilize" in reason


def test_token_out_of_vocab_rejected() -> None:
    """An out-of-vocab token id surfaces a structured reason; warm-
    window math is not even attempted."""
    ts = _build_timeline(
        cold_ttft_ms=10.0,
        warmup_intervals=[],
        steady_intervals=_stable(200),
    )
    # vocab_size=200_000 in default context; 999_999 is out.
    tokens = [7] * (len(ts) - 1) + [999_999]
    ok, reason, _ = warm_decode_oracle(
        _make_scenario(),
        ({0: tokens}, {0: ts}),
        _DEFAULT_CONTEXT,
    )
    assert not ok
    assert reason is not None
    assert "out_of_vocab" in reason


def test_missing_context_key_rejected() -> None:
    """Removing a required key from context surfaces it by name."""
    incomplete = {k: v for k, v in _DEFAULT_CONTEXT.items() if k != "warmup_min_steps"}
    ok, reason, meta = warm_decode_oracle(
        _make_scenario(),
        ({0: [1] * 200}, {0: [float(i) for i in range(200)]}),
        incomplete,
    )
    assert not ok
    assert reason is not None
    assert "warmup_min_steps" in reason
    assert "keys_present" in meta


def test_collected_shape_mismatch_rejected() -> None:
    """Wrong-shape collected (e.g. a list instead of the
    (tokens, ts) tuple) fails fast."""
    ok, reason, _ = warm_decode_oracle(
        _make_scenario(), [1, 2, 3], _DEFAULT_CONTEXT
    )
    assert not ok
    assert reason == "warm_decode_collected_shape_mismatch"


def test_oracle_registry_dispatch() -> None:
    """``ORACLES[OracleKind.WARM_DECODE]`` resolves to the function
    this test file imports — the runner's dispatch goes through this
    table, so a registry mismatch would silently break the gate."""
    assert ORACLES[OracleKind.WARM_DECODE] is warm_decode_oracle


def test_bgt1_aggregate_uses_intersection_not_union() -> None:
    """When per-row warm-up boundaries land at different step indices
    (heterogeneous-warmup case), the aggregate must use the
    intersection of per-row warm windows. A union-window denominator
    would systematically inflate the aggregate by counting only
    post-warmup decodes in the numerator while including warm-up
    wall in the denominator's time span.

    Construction: row 0 has 30 noisy + 200 stable; row 1 has 80 noisy
    + 200 stable. Both rows hit the same 6.5 ms steady-state rate, so
    the honest aggregate is 2 × per-row-mean — no more, no less.
    A union-window implementation would produce ~1.5× per-row-mean
    (denominator inflated by ~0.3 s of row-1 warm-up).
    """
    ts0 = _build_timeline(
        cold_ttft_ms=10.0,
        warmup_intervals=_noisy(30),
        steady_intervals=_stable(200, ms=6.5),
    )
    ts1 = _build_timeline(
        cold_ttft_ms=10.0,
        warmup_intervals=_noisy(80),
        steady_intervals=_stable(200, ms=6.5),
    )
    tokens = {0: [9] * len(ts0), 1: [10] * len(ts1)}
    token_ts_ms = {0: ts0, 1: ts1}
    ok, reason, meta = warm_decode_oracle(
        _make_scenario(), (tokens, token_ts_ms), _DEFAULT_CONTEXT
    )
    assert ok, f"reason={reason}"
    per_row = meta["decode_tok_s_warm_per_row_mean"]
    aggregate = meta["decode_tok_s_warm_aggregate"]
    # Honest aggregate = 2 × per-row (both rows producing at same
    # rate during the overlap); allow ±10% tolerance for endpoint
    # arithmetic in the intersection-window count.
    expected = 2.0 * per_row
    err = abs(aggregate - expected) / expected
    assert err < 0.10, (
        f"aggregate {aggregate:.2f} not close to 2× per-row "
        f"{per_row:.2f} (err={err:.3%}); union-window math would "
        f"have produced ~1.5× per-row instead"
    )
    # Diagnostic fields must surface so a reviewer can verify the
    # math from the JSONL row directly.
    assert "aggregate_overlap_window_ms" in meta
    assert "aggregate_overlap_decodes" in meta
    assert meta["aggregate_overlap_window_ms"] > 0


def test_no_overlap_rejected() -> None:
    """Row 0 finishes its measurement window before row 1 enters
    measurement: no overlap. Oracle fails with a structured reason
    rather than reporting a meaningless or infinite aggregate."""
    # Row 0: short timeline ending at ~1 second.
    ts0 = _build_timeline(
        cold_ttft_ms=0.0,
        warmup_intervals=_noisy(40),
        steady_intervals=_stable(80, ms=6.5),
    )
    # Row 1: starts after row 0 ends; ts1[0] = 5000 ms.
    ts1 = _build_timeline(
        cold_ttft_ms=5000.0,
        warmup_intervals=_noisy(40),
        steady_intervals=_stable(80, ms=6.5),
    )
    tokens = {0: [11] * len(ts0), 1: [12] * len(ts1)}
    token_ts_ms = {0: ts0, 1: ts1}
    ok, reason, meta = warm_decode_oracle(
        _make_scenario(), (tokens, token_ts_ms), _DEFAULT_CONTEXT
    )
    assert not ok
    assert reason == "warm_decode_warm_windows_do_not_overlap"
    # Diagnostic payload helps the user understand which rows
    # straddled the gap.
    assert "t_overlap_start_ms" in meta
    assert "t_overlap_end_ms" in meta
    assert meta["t_overlap_start_ms"] > meta["t_overlap_end_ms"]


def test_decode_tok_s_excludes_cold_ttft() -> None:
    """The warm decode_tok_s computed by the oracle must reflect only
    steady-state inter-token intervals, not the cold TTFT (which can
    be ~50× larger and would otherwise contaminate the rate)."""
    # 2400 ms cold TTFT (kernel compile) + stable 6.5 ms decodes.
    # If the cold TTFT leaked into the rate, warm_aggregate would
    # land near 200 / (2400 + 200×6.5) × 1000 ≈ 53 tok/s — 1/3 of
    # the steady-state rate. A clean implementation lands at 154.
    ts = _build_timeline(
        cold_ttft_ms=2400.0,
        warmup_intervals=_noisy(40),
        steady_intervals=_stable(200, ms=6.5),
    )
    tokens = [8] * len(ts)
    ok, _, meta = warm_decode_oracle(
        _make_scenario(),
        ({0: tokens}, {0: ts}),
        _DEFAULT_CONTEXT,
    )
    assert ok
    assert meta["decode_tok_s_warm_aggregate"] > 100.0
    assert meta["decode_tok_s_warm_aggregate"] < 200.0
