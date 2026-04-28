"""silica.bench.chat_bench — non-interactive chat-REPL metric harness.

CHAT-CLI-HARDENING-7. The chat REPL's headline correctness story
since v1.7.15 is Q-012 cross-call prefix reuse: the second turn of
a multi-turn conversation should hit the radix cache on the
shared prefix (system + prior user + prior assistant) and skip
the corresponding prefill compute. Pre-HARDENING-7 the only
end-to-end verification of that claim lived in the unit-test
fake-cohort (``test_chat_session.py``) plus the manual REPL
smoke. This module ships the missing third leg: a non-interactive
harness that drives ``ChatSession`` against a real model + real
``RadixPrefixCache`` for ``n_turns`` turns, records per-turn
metrics, and emits a structured verdict.

The harness is library-shaped on purpose. ``run_chat_bench``
takes injected adapter / engine / session / cache factories so
the test surface can drive every code path with fakes; the
matching ``scripts/chat_bench.py`` wrapper supplies the real
factories. Mirrors ``silica.bench.runner`` / ``scripts/bench.py``.

Q-012 verdict rule. Subsequent turns (turn 2 onwards) must show
``prefix_hit_blocks > 0``; one-turn runs receive an ``n/a``
verdict because there is nothing to reuse against. A failed
verdict is the harness's way of saying "the chat REPL has
regressed — investigate whether the prefix cache is being built,
inserted into, or queried correctly". Locks the v1.7.15 P5.9
step 2(b) resolution against future drift.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from silica.chat.session import ChatSession
from silica.core.sampling import SamplingParams


@dataclass(frozen=True)
class TurnRecord:
    """One turn's per-turn measurement, captured from
    :class:`silica.chat.session.TurnMetrics` plus the index.

    Fields are nullable where the upstream ``TurnMetrics`` field is —
    the harness preserves "not measured" rather than coercing to
    zero so a downstream report can render ``—`` honestly.
    """

    turn_index: int
    user_prompt: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float | None = None
    decode_tok_s: float | None = None
    peak_memory_mb: float | None = None
    prefix_hit_blocks: int | None = None
    prefix_hit_tokens: int | None = None
    prefix_store_resident_bytes: int | None = None
    prefix_store_logical_bytes: int | None = None


@dataclass(frozen=True)
class ChatBenchReport:
    """Structured output of one ``run_chat_bench`` invocation."""

    model: str
    codec_id: str | None
    n_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    q012_verdict: str = "n/a"
    q012_reason: str = ""


_DEFAULT_USER_PROMPTS: tuple[str, ...] = (
    "Briefly explain why GPUs are commonly used for "
    "deep-learning training.",
    "What about Apple Silicon — how does the Metal compute "
    "pipeline differ from CUDA at a high level?",
    "And how does unified memory in M-series chips affect "
    "KV-cache management strategies for LLM inference?",
)


def default_user_prompts(n_turns: int) -> list[str]:
    """Build a default ``n_turns``-long sequence of user prompts.

    The first three are pre-written follow-up questions sharing a
    deep-learning-on-Apple-Silicon thread so the chat-template
    rendering naturally accumulates a long shared prefix across
    turns. Beyond three, additional turns synthesise a generic
    follow-up so the harness still runs at higher ``n_turns``
    without refusing.
    """
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")
    out: list[str] = list(_DEFAULT_USER_PROMPTS[:n_turns])
    while len(out) < n_turns:
        idx = len(out) + 1
        out.append(
            f"Continue with one more brief follow-up point "
            f"(#{idx})."
        )
    return out


def _evaluate_q012(records: list[TurnRecord]) -> tuple[str, str]:
    """Verdict + reason string from the per-turn record list."""
    if not records:
        return ("n/a", "no turns were recorded")
    if len(records) < 2:
        return (
            "n/a",
            "need at least 2 turns to verify cross-call reuse",
        )
    misses: list[int] = []
    hits: list[int] = []
    for rec in records[1:]:
        if rec.prefix_hit_blocks is None or rec.prefix_hit_blocks <= 0:
            misses.append(rec.turn_index)
        else:
            hits.append(rec.turn_index)
    if not misses:
        return (
            "passed",
            f"turns {hits} all hit the prefix cache "
            f"(prefix_hit_blocks > 0)",
        )
    if not hits:
        return (
            "failed",
            f"no subsequent turn showed any prefix-cache hit "
            f"(turns {misses})",
        )
    return (
        "failed",
        f"turns {misses} did not hit the prefix cache; "
        f"turns {hits} did",
    )


def run_chat_bench(
    *,
    model: str,
    codec_id: str | None = None,
    n_turns: int = 3,
    user_prompts: list[str] | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    seed: int | None = 42,
    get_adapter: Callable[[str], tuple[Any, Any]] | None = None,
    engine_cls: Any = None,
    cache_builder: Callable[[Any, str | None], Any] | None = None,
    session_cls: Any = None,
) -> ChatBenchReport:
    """Drive ``n_turns`` turns through ``ChatSession`` against
    ``model`` and return a :class:`ChatBenchReport`.

    Injection seams. ``get_adapter`` / ``engine_cls`` /
    ``cache_builder`` / ``session_cls`` default to the real
    factories used by the chat REPL itself; tests override them
    with fakes. The defaults are imported lazily so this module
    stays importable without MLX / engine / model dependencies.

    The ``cache_builder`` signature is
    ``(adapter, codec_id) -> RadixPrefixCache``-shaped — the same
    shape the chat REPL's ``_build_prefix_cache`` produces.

    Sampling. Defaults to deterministic greedy decoding
    (``temperature=0.0``, ``seed=42``, ``max_tokens=64``) so
    repeat runs against the same model produce byte-identical
    token streams. Higher ``max_tokens`` increases turn-2's
    cache-residency span (more tokens to reuse on turn 3) but
    also runtime; 64 is the smallest figure that exercises the
    decode path meaningfully on Qwen3-family models.
    """
    # Match the CLI wrapper's lower bound — a zero-turn run is
    # nonsensical (the harness has nothing to record) and would
    # silently emit a vacuously-passing ``n/a`` report. Reject
    # before any work happens regardless of whether the caller
    # passed an explicit empty ``user_prompts`` list.
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")

    if get_adapter is None:
        from silica.models.factory import adapter_for_repo

        get_adapter = adapter_for_repo
    if engine_cls is None:
        from silica.engine import Engine

        engine_cls = Engine
    if session_cls is None:
        session_cls = ChatSession
    if cache_builder is None:
        cache_builder = _default_cache_builder

    if user_prompts is None:
        user_prompts = default_user_prompts(n_turns)
    if len(user_prompts) != n_turns:
        raise ValueError(
            f"user_prompts length {len(user_prompts)} != n_turns {n_turns}"
        )

    adapter, kv = get_adapter(model)
    engine = engine_cls(adapter, kv)
    cache = cache_builder(adapter, codec_id)
    session = session_cls(
        adapter,
        engine,
        system_prompt=system_prompt,
        prefix_cache=cache,
    )

    sp_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        sp_kwargs["seed"] = seed
    params = SamplingParams(**sp_kwargs)

    records: list[TurnRecord] = []
    for i, prompt in enumerate(user_prompts):
        metrics = session.chat(prompt, sampling_params=params)
        records.append(
            TurnRecord(
                turn_index=i + 1,
                user_prompt=prompt,
                prompt_tokens=metrics.prompt_tokens,
                output_tokens=metrics.output_tokens,
                ttft_ms=metrics.ttft_ms,
                decode_tok_s=metrics.decode_tok_s,
                peak_memory_mb=metrics.peak_memory_mb,
                prefix_hit_blocks=metrics.prefix_hit_blocks,
                prefix_hit_tokens=metrics.prefix_hit_tokens,
                prefix_store_resident_bytes=metrics.prefix_store_resident_bytes,
                prefix_store_logical_bytes=metrics.prefix_store_logical_bytes,
            )
        )

    verdict, reason = _evaluate_q012(records)
    return ChatBenchReport(
        model=model,
        codec_id=codec_id,
        n_turns=n_turns,
        turns=records,
        q012_verdict=verdict,
        q012_reason=reason,
    )


def _default_cache_builder(
    adapter: Any, codec_id: str | None
) -> Any:
    """Real-factory cache builder. Block size 4 mirrors the chat
    REPL's ``_PREFIX_CACHE_BLOCK_SIZE`` so the harness exercises
    the same admit/evict shape as a live conversation.

    KEEP IN SYNC with ``silica/chat/cli/app.py:_PREFIX_CACHE_BLOCK_SIZE``
    and ``silica/chat/cli/app.py:_build_prefix_cache``. The two
    helpers duplicate codec / store / cache construction by
    design — the chat REPL's helper has additional injection
    seams (``store_cls`` / ``cache_cls``) for its own
    follow-up tests; this harness uses a tighter signature so
    the test fakes can drop in without re-implementing the
    REPL's injection contract. If ``_PREFIX_CACHE_BLOCK_SIZE``
    changes in the REPL, update the literal below to match.
    """
    from silica.bench.codec_registry import get_codec_spec
    from silica.kvcache.prefix import RadixPrefixCache
    from silica.kvcache.store import SyntheticPrefixBlockStore

    block_size = 4
    layout = adapter.kv_layout()
    codec: Any = None
    if codec_id is not None:
        spec = get_codec_spec(codec_id)
        codec = spec.factory(
            block_size=block_size,
            n_kv_heads=layout.n_kv_heads,
            head_dim=layout.head_dim,
            dtype=layout.dtype,
            seed=42,
        )
    store = SyntheticPrefixBlockStore(
        block_size=block_size, codec=codec
    )
    return RadixPrefixCache(block_size=block_size, store=store)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_text_report(report: ChatBenchReport) -> str:
    """Human-readable single-screen summary.

    Columns: turn, ttft, tok/s, prompt-tokens, output-tokens,
    prefix-hit blocks/tokens, prefix-store residency. Trailing
    line names the Q-012 verdict and reason. Numeric fields with
    a ``None`` source render as ``—`` so the verdict rule
    (``prefix_hit_blocks > 0``) reads honestly even on backends
    that do not surface the metric.
    """
    em = "—"
    lines: list[str] = []
    lines.append(
        f"silica chat bench — model={report.model} "
        f"codec={report.codec_id or 'fp16'} n_turns={report.n_turns}"
    )
    lines.append("")
    header = (
        f"{'turn':>4}  {'ttft':>8}  {'tok/s':>7}  "
        f"{'prompt':>7}  {'out':>5}  {'prefix_hit':>14}  "
        f"{'kv_resident':>11}"
    )
    lines.append(header)
    for rec in report.turns:
        ttft = f"{rec.ttft_ms:.0f}ms" if rec.ttft_ms is not None else em
        toks = (
            f"{rec.decode_tok_s:.1f}"
            if rec.decode_tok_s is not None
            else em
        )
        if rec.prefix_hit_blocks is None:
            hit = em
        else:
            tok_part = (
                rec.prefix_hit_tokens
                if rec.prefix_hit_tokens is not None
                else 0
            )
            hit = f"{rec.prefix_hit_blocks}b/{tok_part}t"
        if rec.prefix_store_resident_bytes is None:
            kv = em
        else:
            kv = f"{rec.prefix_store_resident_bytes / 1e6:.1f}MB"
        lines.append(
            f"{rec.turn_index:>4}  {ttft:>8}  {toks:>7}  "
            f"{rec.prompt_tokens:>7}  {rec.output_tokens:>5}  "
            f"{hit:>14}  {kv:>11}"
        )
    lines.append("")
    lines.append(f"Q-012 cross-call reuse: {report.q012_verdict.upper()}")
    if report.q012_reason:
        lines.append(f"  {report.q012_reason}")
    return "\n".join(lines)


def render_json_report(report: ChatBenchReport) -> str:
    """Machine-readable JSON dump of the same data, formatted
    with ``indent=2`` for human inspection. ``asdict`` walks the
    frozen dataclass tree; nested ``TurnRecord`` entries serialise
    by field name."""
    payload = {
        "model": report.model,
        "codec_id": report.codec_id,
        "n_turns": report.n_turns,
        "q012_verdict": report.q012_verdict,
        "q012_reason": report.q012_reason,
        "turns": [asdict(r) for r in report.turns],
    }
    return json.dumps(payload, indent=2)


__all__ = [
    "ChatBenchReport",
    "TurnRecord",
    "default_user_prompts",
    "render_json_report",
    "render_text_report",
    "run_chat_bench",
]
