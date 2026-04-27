"""Unit tests for ``silica.chat.session.ChatSession``.

Covers the behaviour a user observes across turns:

  * Message history grows with each turn, then shrinks to just
    the system prompt on ``reset()``.
  * ``apply_chat_template`` is preferred; manual
    ``<|im_start|>`` fallback kicks in only when the tokenizer
    does not expose one (or the exposed template raises).
  * EOS / max_tokens / empty finish reasons land on the right
    taxonomy value.
  * Streaming callback receives incremental deltas (never the
    full accumulated text twice).
  * Per-turn metrics carry engine snapshot + peak memory +
    wall-clock through from injected hooks.

No real weights: every test builds a fake adapter + fake engine
with fully in-memory state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from silica.chat.session import ChatSession, TurnMetrics
from silica.core.sampling import SamplingParams

# ---------- fakes ----------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer exposing the handful of methods
    ChatSession touches. Token ids are per-character mod vocab_size
    so tests can control prompt_token counts by string length."""

    def __init__(
        self,
        *,
        vocab_size: int = 200,
        eos_token_ids: set[int] | None = None,
        apply_template: Callable[..., Any] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.eos_token_ids: set[int] = (
            set(eos_token_ids) if eos_token_ids else set()
        )
        self._apply_template = apply_template

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % max(1, self.vocab_size)) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(i % 0x10FFFF) for i in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> Any:
        if self._apply_template is None:
            raise AttributeError("apply_chat_template disabled in this fake")
        return self._apply_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


@dataclass
class _FakeConfig:
    vocab_size: int = 200


class _FakeAdapter:
    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.config = _FakeConfig(vocab_size=tokenizer.vocab_size)
        self._tokenizer = tokenizer

    def tokenizer(self) -> _FakeTokenizer:
        return self._tokenizer


@dataclass
class _FakeSnapshot:
    ttft_ms: float | None = 12.3
    prefill_tok_s: float | None = 1500.0
    decode_tok_s: float | None = 120.0
    resident_mb: float | None = 28.5
    logical_kv_bytes: int | None = 1_000_000


@dataclass
class _FakeMetrics:
    snap: _FakeSnapshot = field(default_factory=_FakeSnapshot)

    def snapshot(self) -> _FakeSnapshot:
        return self.snap


@dataclass
class _FakeBudget:
    resident_bytes: int = 32_000_000
    logical_bytes: int = 48_000_000


class _FakeKVManager:
    def budget(self) -> _FakeBudget:
        return _FakeBudget()


class _FakeEngine:
    """Yields ``tokens`` once; ``prompts_seen`` records every
    prompt the session handed to ``generate``.

    Mirrors the real :class:`silica.engine.Engine` EOS convention:
    if an emitted token is in ``params.stop_token_ids``, the
    generator yields it (so the caller sees the stop token) and
    then terminates — matches vLLM / mlx-lm semantics. Tests that
    pass ``eos_token_ids`` observable from the session side rely
    on this alignment to exercise the stop-finish branch."""

    def __init__(self, tokens: list[int]) -> None:
        self._tokens = list(tokens)
        self.prompts_seen: list[str] = []
        self.params_seen: list[SamplingParams] = []
        self.metrics = _FakeMetrics()
        self.kv_manager = _FakeKVManager()

    def generate(
        self, prompt: str, params: SamplingParams | None = None
    ) -> Iterator[int]:
        effective = params if params is not None else SamplingParams()
        self.prompts_seen.append(prompt)
        self.params_seen.append(effective)
        stop_ids = set(effective.stop_token_ids or ())
        n = 0
        for tok in self._tokens:
            if n >= effective.max_tokens:
                break
            yield tok
            n += 1
            if tok in stop_ids:
                break


def _build_session(
    *,
    system_prompt: str | None = None,
    eos_token_ids: set[int] | None = None,
    apply_template: Callable[..., Any] | None = None,
    engine_tokens: list[int] | None = None,
    peak_memory_mb: float | None = 128.0,
) -> tuple[ChatSession, _FakeEngine, _FakeTokenizer]:
    tok = _FakeTokenizer(
        eos_token_ids=eos_token_ids if eos_token_ids is not None else {99},
        apply_template=apply_template,
    )
    adapter = _FakeAdapter(tok)
    engine = _FakeEngine(
        engine_tokens if engine_tokens is not None else [65, 66, 67]
    )

    peak_calls: dict[str, int] = {"reset": 0}

    def fake_reset() -> None:
        peak_calls["reset"] += 1

    def fake_read() -> float | None:
        return peak_memory_mb

    session = ChatSession(
        adapter,
        engine,
        system_prompt=system_prompt,
        reset_peak_memory=fake_reset,
        read_peak_memory_mb=fake_read,
    )
    return session, engine, tok


# ---------- history + reset -----------------------------------------


def test_session_accumulates_messages_across_turns() -> None:
    session, engine, _ = _build_session(
        system_prompt="sys", engine_tokens=[72, 73, 74]
    )
    session.chat("hello")
    session.chat("again")
    roles = [m["role"] for m in session.messages]
    assert roles == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    # Engine saw two prompts — one per chat call.
    assert len(engine.prompts_seen) == 2


def test_session_reset_retains_only_system_prompt() -> None:
    session, _, _ = _build_session(system_prompt="sys")
    session.chat("hello")
    session.chat("again")
    session.reset()
    assert [m["role"] for m in session.messages] == ["system"]
    assert session.messages[0]["content"] == "sys"


def test_session_with_no_system_prompt_resets_to_empty() -> None:
    session, _, _ = _build_session(system_prompt=None)
    session.chat("hello")
    session.reset()
    assert session.messages == []


# ---------- chat-template preference --------------------------------


def test_apply_chat_template_path_feeds_prompt_to_engine() -> None:
    seen_messages: list[list[dict[str, str]]] = []

    def fake_template(
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        seen_messages.append(list(messages))
        # Return ids decodable as a specific marker so we can
        # assert the session really used this template path.
        return list(b"TEMPLATE:") + [(ord(c) % 128) for c in messages[-1]["content"]]

    session, engine, _ = _build_session(apply_template=fake_template)
    session.chat("world")
    assert len(seen_messages) == 1
    assert seen_messages[0][-1]["content"] == "world"
    assert engine.prompts_seen[0].startswith("TEMPLATE:")


def test_fallback_template_kicks_in_when_apply_template_absent() -> None:
    session, engine, _ = _build_session(apply_template=None)
    session.chat("hi")
    prompt = engine.prompts_seen[0]
    assert "<|im_start|>user" in prompt
    assert "hi" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_fallback_template_kicks_in_when_apply_template_raises() -> None:
    def broken_template(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no template configured")

    session, engine, _ = _build_session(apply_template=broken_template)
    session.chat("hi")
    prompt = engine.prompts_seen[0]
    assert "<|im_start|>user" in prompt


# ---------- finish reasons -------------------------------------------


def test_finish_reason_stop_token_when_last_token_is_eos() -> None:
    # Engine yields two normal tokens and then an EOS id (42).
    session, engine, _ = _build_session(
        eos_token_ids={42}, engine_tokens=[65, 66, 42]
    )
    m = session.chat("hi")
    assert m.finish_reason == "stop_token"
    assert m.output_tokens == 3


def test_finish_reason_max_tokens_when_output_hits_cap() -> None:
    session, engine, _ = _build_session(
        eos_token_ids={999}, engine_tokens=[65, 66, 67]
    )
    m = session.chat(
        "hi",
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=3
        ),
    )
    assert m.finish_reason == "max_tokens"
    assert m.output_tokens == 3


def test_finish_reason_empty_when_engine_yields_nothing() -> None:
    session, engine, _ = _build_session(engine_tokens=[])
    m = session.chat("hi")
    assert m.finish_reason == "empty"
    assert m.output_tokens == 0
    assert m.reply == ""


# ---------- streaming deltas ----------------------------------------


def test_streaming_receives_incremental_deltas_not_cumulative() -> None:
    captured: list[str] = []
    session, engine, _ = _build_session(engine_tokens=[65, 66, 67])
    session.chat("hi", stream_to=captured.append)
    # Each delta should be a single new character once decoded.
    assert "".join(captured) == "ABC"
    # And no delta should be longer than the cumulative suffix.
    # (ruff: use sum of lengths)
    assert sum(len(d) for d in captured) == len("ABC")


def test_streaming_skipped_when_stream_to_is_none() -> None:
    session, _, _ = _build_session(engine_tokens=[65, 66, 67])
    m = session.chat("hi", stream_to=None)
    assert m.reply == "ABC"


# ---------- metrics --------------------------------------------------


def test_metrics_include_snapshot_and_peak() -> None:
    session, engine, _ = _build_session(
        engine_tokens=[65, 66, 67], peak_memory_mb=256.5
    )
    m = session.chat("hi")
    assert m.ttft_ms == 12.3
    assert m.prefill_tok_s == 1500.0
    assert m.decode_tok_s == 120.0
    assert m.resident_mb == 28.5
    assert m.peak_memory_mb == 256.5
    assert m.logical_kv_bytes == 1_000_000
    assert m.wall_s is not None
    assert m.wall_s >= 0.0
    assert m.output_tokens == 3
    assert m.prompt_tokens > 0


def test_metrics_dataclass_defaults_populated() -> None:
    """TurnMetrics has typed fields; accidental renames surface
    here rather than as a downstream attribute error."""
    m = TurnMetrics(
        reply="x",
        prompt_tokens=1,
        output_tokens=1,
        finish_reason="done",
    )
    assert m.reply == "x"
    assert m.ttft_ms is None
    assert m.peak_memory_mb is None


# ---------- sampling-params plumbing --------------------------------


def test_default_sampling_params_attach_eos_stop_ids() -> None:
    session, engine, _ = _build_session(eos_token_ids={10, 20})
    session.chat("hi")
    params = engine.params_seen[0]
    assert set(params.stop_token_ids) == {10, 20}


def test_explicit_params_without_stop_ids_get_eos_injected() -> None:
    session, engine, _ = _build_session(eos_token_ids={10, 20})
    session.chat(
        "hi",
        sampling_params=SamplingParams(
            temperature=0.5, max_tokens=16
        ),
    )
    params = engine.params_seen[0]
    assert params.temperature == 0.5
    assert set(params.stop_token_ids) == {10, 20}


def test_explicit_params_with_stop_ids_preserved_verbatim() -> None:
    session, engine, _ = _build_session(eos_token_ids={10, 20})
    session.chat(
        "hi",
        sampling_params=SamplingParams(
            temperature=0.5,
            max_tokens=16,
            stop_token_ids=(42,),
        ),
    )
    params = engine.params_seen[0]
    assert params.stop_token_ids == (42,)


# ---------- regression: reset + chat ---------------------------------


def test_reset_between_turns_restarts_history() -> None:
    session, engine, _ = _build_session(
        system_prompt="sys", engine_tokens=[65]
    )
    session.chat("first")
    session.reset()
    session.chat("second")
    # Second turn must not see the first turn's user / assistant
    # in its rendered prompt. Manual fallback template path
    # concatenates messages in order, so "first" should not
    # appear in the second prompt.
    assert "first" not in engine.prompts_seen[1]
    assert "second" in engine.prompts_seen[1]


def test_empty_user_text_still_records_message() -> None:
    """The session does not filter empty user_text — caller (CLI)
    is expected to skip empties. This pins the behaviour so a
    future "helpful" filter does not sneak in."""
    session, _, _ = _build_session(engine_tokens=[65])
    session.chat("")
    assert session.messages[-2] == {"role": "user", "content": ""}


def test_peak_reset_called_once_per_turn() -> None:
    """Peak-memory reset must fire exactly once per turn so the
    per-turn peak is not polluted by prior-turn allocations."""
    count: dict[str, int] = {"reset": 0}

    def fake_reset() -> None:
        count["reset"] += 1

    tok = _FakeTokenizer(eos_token_ids={99})
    adapter = _FakeAdapter(tok)
    engine = _FakeEngine([65, 66, 67])
    session = ChatSession(
        adapter,
        engine,
        reset_peak_memory=fake_reset,
        read_peak_memory_mb=lambda: 128.0,
    )
    session.chat("one")
    session.chat("two")
    assert count["reset"] == 2


def test_eos_token_ids_exposed_via_property() -> None:
    session, _, _ = _build_session(eos_token_ids={7, 8, 9})
    assert set(session.eos_token_ids) == {7, 8, 9}


def test_messages_property_returns_copy_not_reference() -> None:
    session, _, _ = _build_session(system_prompt="sys")
    observed = session.messages
    observed.append({"role": "user", "content": "sneaky"})
    # Session's internal history must not be mutated.
    assert [m["role"] for m in session.messages] == ["system"]


@pytest.mark.parametrize(
    "eos_ids,expected",
    [
        ({99}, "done"),
        ({65}, "stop_token"),  # first token is EOS → early stop
    ],
)
def test_finish_reason_parametrized(
    eos_ids: set[int], expected: str
) -> None:
    session, _, _ = _build_session(
        eos_token_ids=eos_ids, engine_tokens=[65, 66, 67]
    )
    m = session.chat(
        "hi",
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=100
        ),
    )
    assert m.finish_reason == expected


# ---------- C-4 prefix-cache integration ---------------------------


@dataclass
class _FakePrefixHit:
    block_ids: tuple[int, ...] = ()
    num_hit_tokens: int = 0


class _FakePrefixCache:
    """Minimal prefix-cache fake used by the C-4 tests.

    Exposes ``peek(tokens) -> _FakePrefixHit`` returning whatever
    the test pre-loaded via :meth:`set_hit`, plus ``block_size``
    so the session's Protocol-conformance read still resolves.
    Records every ``peek`` call so tests can assert the session
    consults the cache exactly once per turn (before the engine
    runs).
    """

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = block_size
        self._hit = _FakePrefixHit()
        self.peek_calls: list[list[int]] = []

    def set_hit(
        self, *, block_ids: tuple[int, ...], num_hit_tokens: int
    ) -> None:
        self._hit = _FakePrefixHit(
            block_ids=block_ids, num_hit_tokens=num_hit_tokens
        )

    def peek(self, tokens: Any) -> _FakePrefixHit:
        self.peek_calls.append(list(tokens))
        return self._hit


class _FakeBatchedEngine:
    """Engine fake that supports both the single-request
    ``generate`` path and the batched ``generate_batch`` path.

    Yields a sequence of :class:`silica.core.events.BatchEvent` for
    one row when ``generate_batch`` is called; mirrors the real
    engine's terminal-event convention (one ``done`` after the
    last token). Records the prefix_cache it received so tests
    can assert routing.
    """

    def __init__(
        self,
        tokens: list[int],
        *,
        finish_reason: str = "done",
    ) -> None:
        self._tokens = list(tokens)
        self._finish_reason = finish_reason
        self.prompts_seen: list[str] = []
        self.batched_prompts_seen: list[Any] = []
        self.batched_prefix_cache_seen: list[Any] = []
        self.metrics = _FakeMetrics()
        self.kv_manager = _FakeKVManager()

    def generate(
        self, prompt: str, params: SamplingParams | None = None
    ) -> Iterator[int]:
        # Provide the single-request path too so tests can flip
        # between paths on the same fake.
        effective = params if params is not None else SamplingParams()
        self.prompts_seen.append(prompt)
        stop_ids = set(effective.stop_token_ids or ())
        n = 0
        for tok in self._tokens:
            if n >= effective.max_tokens:
                break
            yield tok
            n += 1
            if tok in stop_ids:
                break

    def generate_batch(
        self,
        prompts: Any,
        params: SamplingParams | list[SamplingParams] | None = None,
        *,
        max_batch_size: int | None = None,
        prefix_cache: Any = None,
        length_spread_threshold: float = 2.0,
    ) -> Iterator[Any]:
        from silica.core.events import BatchEvent

        del max_batch_size, length_spread_threshold
        self.batched_prompts_seen.append(list(prompts))
        self.batched_prefix_cache_seen.append(prefix_cache)
        # B=1 — one row, req_index=0.
        for tok in self._tokens:
            yield BatchEvent.token(req_index=0, token_id=tok)
        yield BatchEvent.done(req_index=0, reason=self._finish_reason)


def _build_session_with_cache(
    *,
    engine_tokens: list[int] | None = None,
    cache: _FakePrefixCache | None = None,
) -> tuple[ChatSession, _FakeBatchedEngine, _FakePrefixCache]:
    tok = _FakeTokenizer(eos_token_ids={99})
    adapter = _FakeAdapter(tok)
    engine = _FakeBatchedEngine(
        engine_tokens if engine_tokens is not None else [65, 66, 67]
    )
    pc = cache if cache is not None else _FakePrefixCache()
    session = ChatSession(
        adapter,
        engine,
        prefix_cache=pc,
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 256.0,
    )
    return session, engine, pc


def test_prefix_cache_session_routes_through_generate_batch() -> None:
    """When constructed with a prefix cache, ``chat()`` must use
    ``engine.generate_batch`` and pass the cache through."""
    session, engine, pc = _build_session_with_cache()
    session.chat("hello")
    # Single-request generate path should not have been touched.
    assert engine.prompts_seen == []
    # Batched path called exactly once with our cache.
    assert len(engine.batched_prompts_seen) == 1
    assert len(engine.batched_prefix_cache_seen) == 1
    assert engine.batched_prefix_cache_seen[0] is pc


def test_prefix_cache_session_peeks_before_turn() -> None:
    """``peek`` is called exactly once per turn, before the engine
    runs — surfacing the hit count on ``TurnMetrics``."""
    session, _, pc = _build_session_with_cache()
    pc.set_hit(block_ids=(7, 8, 9), num_hit_tokens=48)
    metrics = session.chat("hello world")
    assert len(pc.peek_calls) == 1
    assert metrics.prefix_hit_blocks == 3
    assert metrics.prefix_hit_tokens == 48


def test_prefix_cache_session_zero_hit_recorded() -> None:
    """When the cache is present but the prompt is a miss, hit
    fields should be 0 (not None) so the toolbar distinguishes
    "cache present, no match" from "no cache configured"."""
    session, _, pc = _build_session_with_cache()
    metrics = session.chat("first turn")
    assert metrics.prefix_hit_blocks == 0
    assert metrics.prefix_hit_tokens == 0


def test_no_prefix_cache_leaves_hit_fields_none() -> None:
    """The single-request path leaves prefix-hit fields at None
    so the toolbar renders ``prefix_hit=—`` instead of ``0/0``."""
    session, _, _ = _build_session(engine_tokens=[65, 66, 67])
    metrics = session.chat("hi")
    assert metrics.prefix_hit_blocks is None
    assert metrics.prefix_hit_tokens is None


def test_prefix_cache_session_finish_reason_from_event() -> None:
    """The batched path must use the terminal BatchEvent's
    ``finish_reason`` rather than reclassifying from tokens."""
    tok = _FakeTokenizer(eos_token_ids={99})
    adapter = _FakeAdapter(tok)
    engine = _FakeBatchedEngine([65, 66], finish_reason="max_tokens")
    pc = _FakePrefixCache()
    session = ChatSession(
        adapter,
        engine,
        prefix_cache=pc,
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 256.0,
    )
    metrics = session.chat("hi")
    assert metrics.finish_reason == "max_tokens"


def test_set_prefix_cache_swaps_active_cache() -> None:
    """``set_prefix_cache`` replaces the active instance — the
    chat-CLI shell calls this on /reset to invalidate the
    previous conversation's cached blocks."""
    session, _, pc1 = _build_session_with_cache()
    assert session.prefix_cache is pc1
    pc2 = _FakePrefixCache()
    session.set_prefix_cache(pc2)
    assert session.prefix_cache is pc2
    # Subsequent peek lands on the new cache, not the old one.
    session.chat("after swap")
    assert len(pc1.peek_calls) == 0
    assert len(pc2.peek_calls) == 1


def test_set_prefix_cache_to_none_disables_routing() -> None:
    """Passing None to ``set_prefix_cache`` reverts to the
    single-request path."""
    session, engine, _ = _build_session_with_cache()
    session.chat("with cache")
    assert len(engine.batched_prompts_seen) == 1
    session.set_prefix_cache(None)
    session.chat("without cache")
    # batched count unchanged; single-request count incremented.
    assert len(engine.batched_prompts_seen) == 1
    assert engine.prompts_seen == [
        # The second turn went through the single-request path.
        engine.prompts_seen[0]
    ]


def test_prefix_cache_session_reset_does_not_clear_cache() -> None:
    """``reset()`` clears messages only — the cache instance is
    held until the caller swaps it via ``set_prefix_cache``.
    Documents the contract that the chat-CLI shell is responsible
    for invalidating cache on /reset (avoids a leak of prior-
    conversation tokens but lets non-REPL callers keep cache
    across reset boundaries if they choose)."""
    session, _, pc = _build_session_with_cache()
    session.chat("turn one")
    session.reset()
    # Cache reference unchanged.
    assert session.prefix_cache is pc
