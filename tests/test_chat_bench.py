"""Tests for ``silica.bench.chat_bench``.

CHAT-CLI-HARDENING-7. Pure-Python coverage with injected fake
adapter / engine / cache / session factories. Verifies the
harness's wire-up (right number of turns, prompts forwarded,
metrics captured), the Q-012 verdict logic (passed / failed /
n/a paths), and the renderers (text + JSON well-formed).

Real-model end-to-end smoke is HARDENING-9's job; this file
locks the harness shape so the smoke run is failure-mode
diagnosable rather than "either it works or it doesn't".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from silica.bench.chat_bench import (
    ChatBenchReport,
    TurnRecord,
    default_user_prompts,
    render_json_report,
    render_text_report,
    run_chat_bench,
)
from silica.chat.session import TurnMetrics

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSession:
    """Records each ``chat()`` call, replaying a pre-loaded
    sequence of :class:`TurnMetrics`. Driven by the harness; the
    real ``ChatSession`` would do the work end-to-end."""

    def __init__(
        self,
        *,
        adapter: Any,
        engine: Any,
        system_prompt: str | None,
        prefix_cache: Any,
        replay: list[TurnMetrics],
    ) -> None:
        self.adapter = adapter
        self.engine = engine
        self.system_prompt = system_prompt
        self.prefix_cache = prefix_cache
        self._replay = list(replay)
        self.prompts_seen: list[str] = []
        self.params_seen: list[Any] = []

    def chat(
        self,
        user_text: str,
        *,
        sampling_params: Any = None,
        stream_to: Any = None,
    ) -> TurnMetrics:
        del stream_to
        if not self._replay:
            raise RuntimeError(
                "_FakeSession ran out of pre-loaded TurnMetrics"
            )
        self.prompts_seen.append(user_text)
        self.params_seen.append(sampling_params)
        return self._replay.pop(0)


def _session_factory(
    replay: list[TurnMetrics],
) -> Callable[..., _FakeSession]:
    """Build a ``session_cls``-shaped factory bound to a replay."""

    def _make(
        adapter: Any,
        engine: Any,
        *,
        system_prompt: str | None = None,
        prefix_cache: Any = None,
        **_: Any,
    ) -> _FakeSession:
        return _FakeSession(
            adapter=adapter,
            engine=engine,
            system_prompt=system_prompt,
            prefix_cache=prefix_cache,
            replay=replay,
        )

    return _make


def _three_turn_metrics(
    *,
    miss_indices: tuple[int, ...] = (0,),
    hit_blocks_for_hit: int = 12,
    hit_tokens_for_hit: int = 48,
) -> list[TurnMetrics]:
    """Three-turn metrics replay. ``miss_indices`` selects which
    turns report ``prefix_hit_blocks=0`` (default: only turn 0
    misses, turns 1 and 2 hit)."""
    out: list[TurnMetrics] = []
    for i in range(3):
        is_miss = i in miss_indices
        out.append(
            TurnMetrics(
                reply=f"reply-{i}",
                prompt_tokens=100 + 20 * i,
                output_tokens=20,
                finish_reason="done",
                ttft_ms=120.0 - 30.0 * i,
                decode_tok_s=40.0 + i,
                peak_memory_mb=1500.0 + 10.0 * i,
                prefix_hit_blocks=0 if is_miss else hit_blocks_for_hit,
                prefix_hit_tokens=0 if is_miss else hit_tokens_for_hit,
                prefix_store_resident_bytes=10_000 * (i + 1),
                prefix_store_logical_bytes=10_000 * (i + 1),
            )
        )
    return out


def _drive_run_chat_bench(
    *,
    n_turns: int = 3,
    user_prompts: list[str] | None = None,
    metrics: list[TurnMetrics] | None = None,
    codec_id: str | None = None,
) -> tuple[ChatBenchReport, _FakeSession]:
    """Drive ``run_chat_bench`` with reusable fakes."""
    replay = (
        list(metrics)
        if metrics is not None
        else _three_turn_metrics()
    )
    session_holder: list[_FakeSession] = []

    def _make_session(
        adapter: Any,
        engine: Any,
        *,
        system_prompt: str | None = None,
        prefix_cache: Any = None,
        **_: Any,
    ) -> _FakeSession:
        s = _FakeSession(
            adapter=adapter,
            engine=engine,
            system_prompt=system_prompt,
            prefix_cache=prefix_cache,
            replay=replay,
        )
        session_holder.append(s)
        return s

    def _make_engine(_adapter: Any, _kv: Any) -> str:
        return "fake-engine"

    def _make_cache(_adapter: Any, codec_id_: str | None) -> str:
        return f"fake-cache:{codec_id_}"

    def _get_adapter(_repo: str) -> tuple[str, str]:
        return ("fake-adapter", "fake-kv")

    report = run_chat_bench(
        model="Qwen/Qwen3-0.6B",
        codec_id=codec_id,
        n_turns=n_turns,
        user_prompts=user_prompts,
        get_adapter=_get_adapter,
        engine_cls=_make_engine,
        cache_builder=_make_cache,
        session_cls=_make_session,
    )
    assert session_holder, "_FakeSession was never constructed"
    return report, session_holder[0]


# ---------------------------------------------------------------------------
# default_user_prompts
# ---------------------------------------------------------------------------


def test_default_user_prompts_returns_three_for_n3() -> None:
    """Default prompt set covers the canonical three-turn
    deep-learning thread."""
    prompts = default_user_prompts(3)
    assert len(prompts) == 3
    assert all(isinstance(p, str) and p.strip() for p in prompts)


def test_default_user_prompts_truncates_for_smaller_n() -> None:
    prompts = default_user_prompts(2)
    assert len(prompts) == 2


def test_default_user_prompts_pads_for_larger_n() -> None:
    """Beyond three turns the harness still produces ``n``
    prompts so the user can run longer sessions without
    refusing."""
    prompts = default_user_prompts(5)
    assert len(prompts) == 5
    assert prompts[3].startswith("Continue")
    assert prompts[4].startswith("Continue")


def test_default_user_prompts_rejects_zero() -> None:
    with pytest.raises(ValueError, match="n_turns"):
        default_user_prompts(0)


# ---------------------------------------------------------------------------
# run_chat_bench — wire-up
# ---------------------------------------------------------------------------


def test_run_chat_bench_drives_three_turns_by_default() -> None:
    """Default ``n_turns=3`` invokes the session three times with
    the canonical default prompts."""
    report, session = _drive_run_chat_bench()
    assert report.n_turns == 3
    assert len(report.turns) == 3
    assert len(session.prompts_seen) == 3
    assert session.prompts_seen == default_user_prompts(3)


def test_run_chat_bench_forwards_custom_prompts() -> None:
    """Caller-supplied prompts are forwarded to the session
    in order."""
    custom = ["alpha", "beta", "gamma"]
    report, session = _drive_run_chat_bench(user_prompts=custom)
    assert session.prompts_seen == custom
    assert [r.user_prompt for r in report.turns] == custom


def test_run_chat_bench_rejects_prompt_length_mismatch() -> None:
    """Caller is responsible for matching ``user_prompts`` length
    to ``n_turns``."""
    with pytest.raises(ValueError, match="user_prompts length"):
        run_chat_bench(
            model="dummy",
            n_turns=3,
            user_prompts=["one", "two"],
            get_adapter=lambda _r: ("a", "k"),
            engine_cls=lambda *_a, **_k: "e",
            cache_builder=lambda _a, _c: "c",
            session_cls=_session_factory(_three_turn_metrics()),
        )


def test_run_chat_bench_rejects_zero_n_turns() -> None:
    """A zero-turn run is nonsensical (the harness has nothing to
    record); reject before any work happens regardless of whether
    the caller passed an empty ``user_prompts`` list explicitly.
    Matches the CLI wrapper's lower bound."""
    with pytest.raises(ValueError, match="n_turns"):
        run_chat_bench(
            model="dummy",
            n_turns=0,
            user_prompts=[],
            get_adapter=lambda _r: ("a", "k"),
            engine_cls=lambda *_a, **_k: "e",
            cache_builder=lambda _a, _c: "c",
            session_cls=_session_factory([]),
        )


def test_run_chat_bench_propagates_codec_id_to_cache_builder() -> None:
    """The harness threads ``codec_id`` through to the cache
    factory so block_tq runs use the right codec."""
    seen_codec: list[str | None] = []

    def _make_cache(_adapter: Any, codec_id: str | None) -> str:
        seen_codec.append(codec_id)
        return "cache"

    run_chat_bench(
        model="Qwen/Qwen3-0.6B",
        codec_id="block_tq_b64_b4",
        n_turns=1,
        user_prompts=["one"],
        get_adapter=lambda _r: ("a", "k"),
        engine_cls=lambda *_a, **_k: "e",
        cache_builder=_make_cache,
        session_cls=_session_factory(_three_turn_metrics()[:1]),
    )
    assert seen_codec == ["block_tq_b64_b4"]


def test_run_chat_bench_records_per_turn_metrics() -> None:
    """Each ``TurnMetrics`` field of interest reaches the
    matching ``TurnRecord`` field unchanged."""
    metrics = _three_turn_metrics()
    report, _ = _drive_run_chat_bench(metrics=metrics)
    for rec, m in zip(report.turns, metrics, strict=True):
        assert rec.prompt_tokens == m.prompt_tokens
        assert rec.output_tokens == m.output_tokens
        assert rec.ttft_ms == m.ttft_ms
        assert rec.decode_tok_s == m.decode_tok_s
        assert rec.peak_memory_mb == m.peak_memory_mb
        assert rec.prefix_hit_blocks == m.prefix_hit_blocks
        assert rec.prefix_hit_tokens == m.prefix_hit_tokens
        assert (
            rec.prefix_store_resident_bytes
            == m.prefix_store_resident_bytes
        )
        assert (
            rec.prefix_store_logical_bytes
            == m.prefix_store_logical_bytes
        )


def test_run_chat_bench_passes_sampling_params_to_session() -> None:
    """``max_tokens`` / ``temperature`` / ``seed`` propagate to
    the per-turn ``chat()`` call."""
    _, session = _drive_run_chat_bench()
    assert len(session.params_seen) == 3
    for p in session.params_seen:
        assert p.max_tokens == 64
        assert p.temperature == 0.0


# ---------------------------------------------------------------------------
# Q-012 verdict
# ---------------------------------------------------------------------------


def test_q012_verdict_passed_when_all_subsequent_turns_hit() -> None:
    """Default fake metrics: turn 0 misses, turns 1+ hit. The
    headline Q-012 reuse claim is satisfied."""
    report, _ = _drive_run_chat_bench()
    assert report.q012_verdict == "passed"
    assert "turns [2, 3]" in report.q012_reason


def test_q012_verdict_failed_when_no_subsequent_turn_hits() -> None:
    """Every turn missing the cache → unconditional failure."""
    metrics = _three_turn_metrics(miss_indices=(0, 1, 2))
    report, _ = _drive_run_chat_bench(metrics=metrics)
    assert report.q012_verdict == "failed"
    assert "no subsequent turn" in report.q012_reason


def test_q012_verdict_failed_when_some_subsequent_turn_misses() -> None:
    """Turn 1 hits but turn 2 misses → still a failure (the
    reuse claim is "every subsequent turn", not "any")."""
    metrics = _three_turn_metrics(miss_indices=(0, 2))
    report, _ = _drive_run_chat_bench(metrics=metrics)
    assert report.q012_verdict == "failed"
    assert "turns [3]" in report.q012_reason


def test_q012_verdict_failed_when_hit_count_is_none() -> None:
    """A backend that does not surface ``prefix_hit_blocks``
    (returns ``None``) is treated as a miss for verdict
    purposes — the harness cannot prove reuse."""
    metrics = _three_turn_metrics()
    metrics[1] = TurnMetrics(
        reply=metrics[1].reply,
        prompt_tokens=metrics[1].prompt_tokens,
        output_tokens=metrics[1].output_tokens,
        finish_reason="done",
        prefix_hit_blocks=None,
    )
    report, _ = _drive_run_chat_bench(metrics=metrics)
    assert report.q012_verdict == "failed"


def test_q012_verdict_n_a_for_one_turn_run() -> None:
    """One turn cannot exercise cross-call reuse — verdict is
    ``n/a``, not a failure."""
    metrics = _three_turn_metrics()[:1]
    report, _ = _drive_run_chat_bench(
        n_turns=1, user_prompts=["only-turn"], metrics=metrics
    )
    assert report.q012_verdict == "n/a"
    assert "at least 2 turns" in report.q012_reason


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_render_text_report_includes_header_and_per_turn_rows() -> None:
    report, _ = _drive_run_chat_bench()
    text = render_text_report(report)
    assert "Qwen/Qwen3-0.6B" in text
    assert "fp16" in text  # codec=None renders as fp16
    # One row per turn.
    assert text.count("\n") >= 5
    # Verdict line.
    assert "Q-012 cross-call reuse: PASSED" in text


def test_render_text_report_shows_em_dash_for_none_fields() -> None:
    """Per :class:`TurnRecord` semantics, ``None`` fields render
    as em-dash so the operator can distinguish "not measured"
    from "zero"."""
    report = ChatBenchReport(
        model="m",
        codec_id=None,
        n_turns=1,
        turns=[
            TurnRecord(
                turn_index=1,
                user_prompt="x",
                prompt_tokens=10,
                output_tokens=2,
                ttft_ms=None,
                decode_tok_s=None,
                prefix_hit_blocks=None,
                prefix_store_resident_bytes=None,
            )
        ],
        q012_verdict="n/a",
        q012_reason="single turn",
    )
    text = render_text_report(report)
    assert "—" in text  # em-dash placeholder


def test_render_text_report_codec_field_uses_codec_id_when_present() -> None:
    metrics = _three_turn_metrics()
    report, _ = _drive_run_chat_bench(
        metrics=metrics, codec_id="block_tq_b64_b4"
    )
    text = render_text_report(report)
    assert "codec=block_tq_b64_b4" in text


def test_render_json_report_round_trips_via_json_loads() -> None:
    """JSON renderer must produce parseable output with the
    expected top-level keys."""
    report, _ = _drive_run_chat_bench()
    payload_text = render_json_report(report)
    payload = json.loads(payload_text)
    assert payload["model"] == "Qwen/Qwen3-0.6B"
    assert payload["n_turns"] == 3
    assert payload["q012_verdict"] == "passed"
    assert isinstance(payload["turns"], list)
    assert len(payload["turns"]) == 3
    # Each turn entry preserves field names.
    first = payload["turns"][0]
    for field in (
        "turn_index",
        "user_prompt",
        "prompt_tokens",
        "output_tokens",
        "ttft_ms",
        "decode_tok_s",
        "prefix_hit_blocks",
    ):
        assert field in first
