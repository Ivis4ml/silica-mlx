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

from silica.chat.session import (
    ChatSession,
    TurnMetrics,
    _strip_thinking_block,
)
from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import PrefixCacheStats

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
        **kwargs: Any,
    ) -> Any:
        if self._apply_template is None:
            raise AttributeError("apply_chat_template disabled in this fake")
        # Record the kwargs so HARDENING-2 tests can assert which
        # template parameters ChatSession forwarded (notably
        # ``enable_thinking``).
        self.last_apply_template_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **kwargs,
        }
        return self._apply_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
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
    thinking_mode: bool | None = None,
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
        thinking_mode=thinking_mode,
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


# ---------- set_system_prompt (CHAT-CLI-HARDENING-1, F1) -------------


def test_set_system_prompt_replaces_existing_system_message() -> None:
    """The original system prompt is supplied at construction. After
    ``set_system_prompt('new ...')`` the live session's leading
    system-role message carries the new content. Pre-HARDENING-1 the
    /system command wrote ``state.config["system_prompt"]`` only and
    never touched the live session, so user-facing chat continued
    against the original construction-time prompt."""
    session, _, _ = _build_session(system_prompt="sys-original")
    session.set_system_prompt("sys-replacement")
    msgs = session.messages
    assert msgs[0] == {"role": "system", "content": "sys-replacement"}
    # Only one system message; the replacement does not append.
    sys_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(sys_msgs) == 1


def test_set_system_prompt_inserts_when_session_had_none() -> None:
    """A session built with ``system_prompt=None`` accepts a later
    ``set_system_prompt('...')`` and has the new message prepended
    at index 0."""
    session, _, _ = _build_session(system_prompt=None)
    session.chat("hello")
    # Pre-set: messages = [user, assistant].
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    session.set_system_prompt("system-injected")
    # Post-set: system is at index 0; the rest of history is preserved.
    roles = [m["role"] for m in session.messages]
    assert roles == ["system", "user", "assistant"]
    assert session.messages[0]["content"] == "system-injected"


def test_set_system_prompt_empty_string_clears() -> None:
    session, _, _ = _build_session(system_prompt="sys-original")
    session.set_system_prompt("")
    assert all(m["role"] != "system" for m in session.messages)


def test_set_system_prompt_none_clears() -> None:
    session, _, _ = _build_session(system_prompt="sys-original")
    session.set_system_prompt(None)
    assert all(m["role"] != "system" for m in session.messages)


def test_set_system_prompt_preserves_non_system_history() -> None:
    """Setting / clearing the system prompt must not touch the
    user / assistant turns. This pins the contract that /system
    only edits role=system entries — the conversation log is
    not collateral damage."""
    session, _, _ = _build_session(system_prompt="sys-original")
    session.chat("hello")
    session.chat("again")
    user_assist_before = [
        m for m in session.messages if m["role"] != "system"
    ]
    session.set_system_prompt("sys-replacement")
    user_assist_after = [
        m for m in session.messages if m["role"] != "system"
    ]
    assert user_assist_before == user_assist_after
    session.set_system_prompt(None)  # clear
    user_assist_cleared = [
        m for m in session.messages if m["role"] != "system"
    ]
    assert user_assist_cleared == user_assist_before


def test_set_system_prompt_takes_effect_in_next_chat_render() -> None:
    """The next ``chat()`` call must render the new system prompt
    into the prompt the engine sees. This is the user-facing
    correctness contract HARDENING-1 fixes — the /system command
    promises 'for the rest of the session', not just 'in saved
    state'."""
    session, engine, _ = _build_session(system_prompt="sys-original")
    session.set_system_prompt("sys-replacement")
    session.chat("hi")
    # The fake template includes message contents verbatim; the
    # engine sees the new system prompt in the rendered text.
    assert engine.prompts_seen, "engine never saw a prompt"
    last_prompt = engine.prompts_seen[-1]
    assert "sys-replacement" in last_prompt
    assert "sys-original" not in last_prompt


# ---------- thinking_mode threading (CHAT-CLI-HARDENING-2, F2) ------


def _identity_template_via_text(
    messages: list[dict[str, str]],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = False,
    **kwargs: Any,
) -> Any:
    """Test apply_chat_template that renders messages by joining
    role / content with a separator the test can pattern-match.
    Returns token IDs when ``tokenize=True``; the kwargs are
    accepted but ignored — the recording lives on the
    ``_FakeTokenizer.last_apply_template_kwargs`` attribute."""
    del kwargs  # consumed via the fake's recording, not by this template
    rendered = "\n".join(f"<{m['role']}>{m['content']}" for m in messages)
    if add_generation_prompt:
        rendered += "\n<assistant>"
    if tokenize:
        return [(ord(c) % 200) for c in rendered]
    return rendered


def test_thinking_mode_True_threads_enable_thinking_kwarg() -> None:
    """Pre-HARDENING-2 the chat template was called without
    ``enable_thinking``, so toggling /config thinking_mode=on/off
    only flipped the parser-side fold and the model could still
    emit reasoning regardless. After HARDENING-2 the kwarg is
    forwarded so the template builds the right prompt for the
    requested mode."""
    session, _, tok = _build_session(
        apply_template=_identity_template_via_text,
        thinking_mode=True,
    )
    session.chat("hello")
    assert tok.last_apply_template_kwargs is not None
    assert tok.last_apply_template_kwargs.get("enable_thinking") is True


def test_thinking_mode_False_threads_enable_thinking_kwarg() -> None:
    session, _, tok = _build_session(
        apply_template=_identity_template_via_text,
        thinking_mode=False,
    )
    session.chat("hello")
    assert tok.last_apply_template_kwargs is not None
    assert tok.last_apply_template_kwargs.get("enable_thinking") is False


def test_thinking_mode_None_omits_enable_thinking_kwarg() -> None:
    """When the caller leaves ``thinking_mode`` unset (the v1.7.x
    default), ChatSession does not pass ``enable_thinking`` at
    all, preserving backward-compat for tokenizers / templates
    that do not understand the kwarg or whose default mode is
    correct already."""
    session, _, tok = _build_session(
        apply_template=_identity_template_via_text,
        thinking_mode=None,
    )
    session.chat("hello")
    assert tok.last_apply_template_kwargs is not None
    assert "enable_thinking" not in tok.last_apply_template_kwargs


def test_set_thinking_mode_takes_effect_on_next_chat() -> None:
    """Mid-session ``set_thinking_mode(False)`` flips the kwarg
    from True (or None) to False on the next ``chat()`` call —
    this is the contract /config thinking_mode=off needs to be
    honest about the live session, not just save / load state."""
    session, _, tok = _build_session(
        apply_template=_identity_template_via_text,
        thinking_mode=True,
    )
    session.chat("first")
    assert tok.last_apply_template_kwargs.get("enable_thinking") is True
    session.set_thinking_mode(False)
    session.chat("second")
    assert tok.last_apply_template_kwargs.get("enable_thinking") is False


def test_set_thinking_mode_None_drops_kwarg_on_next_chat() -> None:
    session, _, tok = _build_session(
        apply_template=_identity_template_via_text,
        thinking_mode=True,
    )
    session.chat("first")
    assert "enable_thinking" in tok.last_apply_template_kwargs
    session.set_thinking_mode(None)
    session.chat("second")
    assert "enable_thinking" not in tok.last_apply_template_kwargs


# ---------------------------------------------------------------------------
# pop_last_exchange (CHAT-CLI-HARDENING-4 / F4)
# ---------------------------------------------------------------------------


def test_pop_last_exchange_returns_user_content_and_drops_pair() -> None:
    """After one full turn, ``pop_last_exchange`` removes the
    ``(user, assistant)`` pair and returns the popped user text."""
    session, _, _ = _build_session(engine_tokens=[65, 66, 67])
    session.chat("hello world")
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-2]["role"] == "user"

    popped = session.pop_last_exchange()
    assert popped == "hello world"
    # Only the system prompt remains (or empty if none).
    assert all(m["role"] == "system" for m in session.messages)


def test_pop_last_exchange_returns_none_on_fresh_session() -> None:
    """A fresh session has no ``(user, assistant)`` pair; the
    method returns ``None`` and does not touch the message log."""
    session, _, _ = _build_session()
    before = list(session.messages)
    assert session.pop_last_exchange() is None
    assert session.messages == before


def test_pop_last_exchange_returns_none_when_only_system_present() -> None:
    """System-only history (no user / assistant turns yet) reports
    nothing to regenerate."""
    session, _, _ = _build_session(system_prompt="be terse")
    assert session.pop_last_exchange() is None
    assert session.messages == [
        {"role": "system", "content": "be terse"}
    ]


def test_pop_last_exchange_returns_none_when_last_is_user_only() -> None:
    """A trailing user-only message (e.g. mid-generation abort
    leaving the appended user message but no assistant reply) is
    NOT a valid regenerate target — strict shape returns None."""
    session, _, _ = _build_session()
    # Manually shape the history to model an aborted turn: user
    # message appended but no assistant reply.
    session.replace_messages(
        [
            {"role": "user", "content": "hi"},
        ]
    )
    assert session.pop_last_exchange() is None
    assert session.messages == [
        {"role": "user", "content": "hi"}
    ]


def test_pop_last_exchange_preserves_system_prompt() -> None:
    """The system prompt sits at index 0 and is independent of
    user / assistant turns; pop_last_exchange must not touch it."""
    session, _, _ = _build_session(
        system_prompt="be precise", engine_tokens=[65, 66, 67]
    )
    session.chat("hello")
    popped = session.pop_last_exchange()
    assert popped == "hello"
    assert session.messages == [
        {"role": "system", "content": "be precise"}
    ]


def test_pop_last_exchange_preserves_earlier_turns() -> None:
    """Multi-turn history: only the most recent
    ``(user, assistant)`` pair is dropped; earlier turns survive."""
    session, _, _ = _build_session(engine_tokens=[65, 66, 67])
    session.chat("first")
    session.chat("second")
    popped = session.pop_last_exchange()
    assert popped == "second"
    # First turn's pair survives.
    roles = [m["role"] for m in session.messages]
    contents = [m["content"] for m in session.messages]
    assert roles[-2:] == ["user", "assistant"]
    assert contents[-2] == "first"


def test_pop_last_exchange_then_chat_round_trips() -> None:
    """End-to-end: pop, then re-issue chat() with the returned
    prompt — the message log returns to a one-turn shape and the
    next assistant message is freshly generated."""
    session, engine, _ = _build_session(engine_tokens=[65, 66, 67])
    session.chat("regen me")
    n_calls_before = len(engine.prompts_seen)
    popped = session.pop_last_exchange()
    assert popped == "regen me"

    session.chat(popped)
    # One additional engine call.
    assert len(engine.prompts_seen) == n_calls_before + 1
    # History shape after regenerate is identical to the original
    # one-turn shape.
    roles = [m["role"] for m in session.messages]
    assert roles[-2:] == ["user", "assistant"]
    assert session.messages[-2]["content"] == "regen me"


def test_pop_last_exchange_does_not_clear_prefix_cache() -> None:
    """The prefix cache survives the pop — same user text re-
    tokenises to identical prompt ids, and Q-012 cross-call reuse
    means the next chat() turn hits the prior prefill exactly."""
    session, _, pc = _build_session_with_cache()
    pc_before = session.prefix_cache
    session.chat("hello")
    session.pop_last_exchange()
    # Cache instance unchanged — caller is responsible if they
    # want to invalidate.
    assert session.prefix_cache is pc_before


def test_messages_snapshot_round_trips_via_replace_messages() -> None:
    """Load-bearing contract for ``/regenerate`` rollback (F4): the
    chat-CLI app snapshots ``chat_session.messages`` before
    ``pop_last_exchange`` and restores via ``replace_messages`` if
    the regenerated turn aborts. The snapshot must round-trip
    byte-equivalently — mutating the live session after capturing
    it must NOT alter the snapshot, and restoring must yield the
    exact pre-pop state."""
    session, _, _ = _build_session(
        system_prompt="be terse", engine_tokens=[65, 66, 67]
    )
    session.chat("hello")
    snapshot = session.messages
    assert [m["role"] for m in snapshot] == [
        "system",
        "user",
        "assistant",
    ]

    session.pop_last_exchange()
    # Snapshot must be unaffected by the pop.
    assert [m["role"] for m in snapshot] == [
        "system",
        "user",
        "assistant",
    ]

    # Simulate the app's abort branch appending a user message
    # (chat() does this before the engine call) then rolling back.
    session.replace_messages(
        [*session.messages, {"role": "user", "content": "hello"}]
    )
    assert [m["role"] for m in session.messages] == [
        "system",
        "user",
    ]
    # Rollback restores the pre-pop state byte-equivalently.
    session.replace_messages(snapshot)
    assert session.messages == snapshot


def test_pop_last_exchange_then_chat_re_peeks_identical_prompt_ids() -> None:
    """Load-bearing claim for Q-012 cross-call reuse: after
    ``pop_last_exchange()``, a fresh ``chat()`` with the popped
    user text must produce identical prompt ids to the original
    turn — so the cache's ``peek`` call on the regenerated turn is
    a byte-for-byte replay of the first turn's peek (and would
    therefore hit every block of the prior prefill if those blocks
    were inserted)."""
    session, _, pc = _build_session_with_cache(
        engine_tokens=[65, 66, 67]
    )
    session.chat("regen me")
    first_peek_ids = list(pc.peek_calls[-1])

    popped = session.pop_last_exchange()
    assert popped == "regen me"

    session.chat(popped)
    second_peek_ids = list(pc.peek_calls[-1])

    assert first_peek_ids == second_peek_ids


def test_thinking_mode_default_is_None() -> None:
    """ChatSession's default thinking_mode is None (no propagation),
    not True/False. Explicit construction-time / shell-side wiring
    is required to opt into either explicit mode. This default
    keeps existing call sites that did not know about the
    parameter byte-equivalent to pre-HARDENING-2 behaviour."""
    # Construct without the helper's thinking_mode= so we exercise
    # ChatSession.__init__ default directly.
    tok = _FakeTokenizer(
        apply_template=_identity_template_via_text,
        eos_token_ids={99},
    )
    adapter = _FakeAdapter(tok)
    engine = _FakeEngine([65, 66, 67])
    session = ChatSession(adapter, engine)  # no thinking_mode= kwarg
    session.chat("hello")
    assert "enable_thinking" not in tok.last_apply_template_kwargs


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


# ---------- replace_messages (C-7 /load) -----------------------------


def test_replace_messages_swaps_history_wholesale() -> None:
    """``replace_messages`` is the chat-CLI ``/load`` hook: the
    restored conversation replaces every existing message —
    including the system prompt — so the loaded file is the
    single source of truth for what the next turn sees."""
    session, _, _ = _build_session(
        system_prompt="original system", engine_tokens=[65]
    )
    session.chat("first user")
    assert len(session.messages) == 3  # system + user + assistant
    new_history = [
        {"role": "system", "content": "loaded system"},
        {"role": "user", "content": "loaded user"},
        {"role": "assistant", "content": "loaded reply"},
    ]
    session.replace_messages(new_history)
    assert session.messages == new_history


def test_replace_messages_returns_independent_copy() -> None:
    """Mutating the input list after the call must not affect the
    session's stored history (and vice versa)."""
    session, _, _ = _build_session(engine_tokens=[65])
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    session.replace_messages(history)
    history.append({"role": "user", "content": "MUTATED"})
    assert {"role": "user", "content": "MUTATED"} not in session.messages


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


# ---------- TurnMetrics timing fallbacks ---------------------------


def test_turn_metrics_compute_ttft_when_engine_snapshot_missing() -> None:
    """The single-request engine path populates ``ttft_ms`` via
    ``engine.metrics.set_metric``, but the prefix-cache /
    batched path (Engine.generate_batch → ContinuousBatcher) does
    not. ChatSession must compute TTFT from its own wall clock so
    ``TurnMetrics.ttft_ms`` is populated regardless of which path
    drove the turn."""
    # _FakeMetrics's snapshot reports ttft_ms=12.3 by default, so
    # the engine-supplied value wins. Override to None to force
    # the fallback path.
    session, engine, _ = _build_session(
        engine_tokens=[65, 66, 67]
    )
    engine.metrics.snap = _FakeSnapshot(  # type: ignore[attr-defined]
        ttft_ms=None,
        prefill_tok_s=None,
        decode_tok_s=None,
        resident_mb=None,
        logical_kv_bytes=None,
    )
    metrics = session.chat("hi")
    # The fake engine yields all 3 tokens immediately; t_first ≈
    # t_start so ttft_ms is small but non-None.
    assert metrics.ttft_ms is not None
    assert metrics.ttft_ms >= 0.0


def test_turn_metrics_prefer_engine_snapshot_when_present() -> None:
    """When the engine populates ttft_ms / decode_tok_s, those
    values flow through unchanged — ChatSession's wall-clock
    fallback is the alternative source, not a replacement."""
    session, engine, _ = _build_session(engine_tokens=[65, 66, 67])
    # Default _FakeSnapshot already has ttft_ms=12.3, decode_tok_s=120.0.
    metrics = session.chat("hi")
    assert metrics.ttft_ms == pytest.approx(12.3)
    assert metrics.decode_tok_s == pytest.approx(120.0)


# ---------- C-8 EOS-token suppression ------------------------------


def test_reply_text_does_not_include_trailing_eos_token() -> None:
    """vLLM / mlx-lm convention yields the EOS token before
    terminating. Including it in the stored assistant message
    leaks ``<|im_end|>`` (or whatever the EOS decodes to) into
    every reply. ChatSession must strip the trailing EOS before
    decoding for the message log."""
    session, engine, tok = _build_session(
        eos_token_ids={99}, engine_tokens=[65, 66, 99]
    )
    metrics = session.chat("hi")
    # 65/66/99 → "AB" + decoded EOS char. Stored reply must be
    # just "AB" — no trailing EOS character.
    assert metrics.reply == tok.decode([65, 66])
    assert metrics.reply == "AB"
    # Message history mirror of the same constraint.
    assert session.messages[-1]["content"] == "AB"


def test_streaming_callback_does_not_emit_eos_token_text() -> None:
    """Streamed deltas must omit the EOS token's decoded bytes so
    the chat REPL does not show ``<|im_end|>`` at the end of
    every assistant turn."""
    session, _, _ = _build_session(
        eos_token_ids={99}, engine_tokens=[65, 66, 99]
    )
    streamed: list[str] = []
    session.chat("hi", stream_to=streamed.append)
    joined = "".join(streamed)
    assert joined == "AB"  # no EOS character at the end


def test_eos_only_reply_yields_empty_string() -> None:
    """If the model emits the EOS token immediately (no real
    reply tokens), the stored reply must be the empty string —
    not the EOS character."""
    session, _, _ = _build_session(
        eos_token_ids={99}, engine_tokens=[99]
    )
    metrics = session.chat("hi")
    assert metrics.reply == ""


class _PartialUtf8Tokenizer:
    """Tokenizer fake whose ``decode`` produces a transient
    ``\\ufffd`` at one specific token-count boundary.

    Models the BPE-emoji-split case: the cumulative byte sequence
    after token N has invalid trailing UTF-8 (one byte of a
    multi-byte char), so the tokenizer emits ``\\ufffd``; token
    N+1 contributes the missing bytes and the replacement char
    is replaced by the real character. The streaming code must
    not leak the ``\\ufffd`` to the consumer in the meantime.
    """

    eos_token_ids: set[int] = {99}
    vocab_size: int = 200

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        n = len(token_ids)
        if n == 0:
            return ""
        if n == 1:
            return "Hello"
        if n == 2:
            return "Hello �"  # partial emoji
        if n == 3:
            return "Hello 😀"  # closing bytes arrived
        return "Hello 😀 done"


def test_streaming_holds_back_trailing_replacement_char() -> None:
    """The tokenizer's mid-stream ``\\ufffd`` (commonly emoji bytes
    split across BPE tokens) must NOT be streamed verbatim — it
    becomes a permanent ``?`` glyph in the conversation log even
    after the next token completes the multi-byte sequence."""
    tok = _PartialUtf8Tokenizer()
    # Hand-build the session — _build_session uses _FakeTokenizer
    # with deterministic mod-based decode that does not exercise
    # the boundary case.
    adapter = _FakeAdapter(tok)
    engine = _FakeEngine([1, 2, 3, 4])
    session = ChatSession(
        adapter,
        engine,
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 0.0,
    )
    streamed: list[str] = []
    metrics = session.chat(
        "hi",
        sampling_params=SamplingParams(max_tokens=4),
        stream_to=streamed.append,
    )
    full_streamed = "".join(streamed)
    # The streamed output must NEVER contain a ``�``.
    assert "�" not in full_streamed
    # The decoded final replied (after token 3) has the real
    # emoji; the token-4 step extends with " done"; together the
    # streamed output reaches the full final.
    assert full_streamed == "Hello 😀 done"
    # The stored reply also strips any trailing replacement char
    # (defensive — full final has none here).
    assert "�" not in metrics.reply
    assert metrics.reply == "Hello 😀 done"


def test_streaming_drops_truly_invalid_trailing_replacement_char() -> None:
    """If the model emits a partial multi-byte sequence and then
    EOS arrives without completing it, the trailing ``\\ufffd``
    is permanent. The streaming output drops it (better to lose
    one glyph than to render a permanent ``?``); the stored
    reply matches by also stripping."""

    class _CutOffTokenizer:
        eos_token_ids: set[int] = {99}
        vocab_size: int = 200

        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

        def decode(self, token_ids: list[int]) -> str:
            n = len(token_ids)
            if n == 0:
                return ""
            if n == 1:
                return "Hello"
            # Token 2 is EOS; the decoded prefix-without-EOS
            # leaves "Hello�" — partial emoji, never
            # completed.
            if n == 2:
                return "Hello�"
            return "Hello�"  # never gets here

    tok = _CutOffTokenizer()
    adapter = _FakeAdapter(tok)
    engine = _FakeEngine([1, 99])
    session = ChatSession(
        adapter,
        engine,
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 0.0,
    )
    streamed: list[str] = []
    metrics = session.chat(
        "hi",
        sampling_params=SamplingParams(max_tokens=10),
        stream_to=streamed.append,
    )
    assert "�" not in "".join(streamed)
    assert "�" not in metrics.reply
    assert metrics.reply == "Hello"


def test_non_eos_finish_keeps_full_decoded_text() -> None:
    """When generation ends by max_tokens (no EOS yielded), the
    reply text retains every emitted token. Strip-trailing-EOS is
    conditional on the last token actually being in the EOS set."""
    session, _, tok = _build_session(
        eos_token_ids={99}, engine_tokens=[65, 66, 67]
    )
    metrics = session.chat(
        "hi", sampling_params=SamplingParams(max_tokens=3)
    )
    assert metrics.reply == tok.decode([65, 66, 67])
    assert metrics.reply == "ABC"


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


class _FakePrefixCacheWithStats:
    """Minimal prefix cache exposing the public ``stats()`` API
    (HARDENING-3 / F6). Returns a fixed ``PrefixCacheStats`` so the
    test can assert that ChatSession reads the public surface and
    surfaces the figures on TurnMetrics, without poking the store's
    private attributes."""

    def __init__(
        self,
        *,
        resident_bytes: int = 1_234_567,
        logical_bytes: int | None = None,
    ) -> None:
        self.block_size = 4
        self._hit = _FakePrefixHit()
        self._stats = PrefixCacheStats(
            block_size=4,
            hits=0,
            num_blocks=1,
            resident_bytes=resident_bytes,
            logical_bytes=(
                resident_bytes
                if logical_bytes is None
                else logical_bytes
            ),
            has_codec=False,
        )

    def peek(self, tokens: Any) -> _FakePrefixHit:
        return self._hit

    def stats(self) -> PrefixCacheStats:
        return self._stats


def test_turn_metrics_carry_prefix_store_resident_bytes() -> None:
    """When the prefix cache exposes ``stats()``, ChatSession must
    read the public snapshot and surface ``resident_bytes`` /
    ``logical_bytes`` on TurnMetrics so the chat-CLI toolbar can
    show prefix-store occupancy (the ``kv=`` field) instead of the
    always-zero active-KV figure."""
    tok = _FakeTokenizer(eos_token_ids={99})
    adapter = _FakeAdapter(tok)
    engine = _FakeBatchedEngine([65, 66, 67])
    pc = _FakePrefixCacheWithStats()
    session = ChatSession(
        adapter,
        engine,
        prefix_cache=pc,
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 256.0,
    )
    metrics = session.chat("hi")
    assert metrics.prefix_store_resident_bytes == 1_234_567
    # No codec on the fake → logical == resident.
    assert metrics.prefix_store_logical_bytes == 1_234_567


def test_turn_metrics_prefix_store_none_when_cache_lacks_stats() -> None:
    """Backends without ``stats()`` (older shapes, or paged-only
    caches that have not yet adopted the public API) must not break
    the path — ChatSession's capability check returns ``None`` for
    both fields."""

    class _NoStats:
        def __init__(self) -> None:
            self.block_size = 4

        def peek(self, tokens: Any) -> _FakePrefixHit:
            return _FakePrefixHit()

    tok = _FakeTokenizer(eos_token_ids={99})
    adapter = _FakeAdapter(tok)
    engine = _FakeBatchedEngine([65, 66])
    pc = _NoStats()
    session = ChatSession(
        adapter,
        engine,
        prefix_cache=pc,  # type: ignore[arg-type]
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 0.0,
    )
    metrics = session.chat("hi")
    assert metrics.prefix_store_resident_bytes is None
    assert metrics.prefix_store_logical_bytes is None


def test_turn_metrics_prefix_store_none_when_stats_returns_none_fields() -> (
    None
):
    """``stats()`` may legitimately return ``resident_bytes=None`` /
    ``logical_bytes=None`` on backends that do not track residency
    (PagedPrefixBlockStore). ChatSession must propagate ``None``
    rather than coercing to zero."""

    class _StatsNoneFields:
        def __init__(self) -> None:
            self.block_size = 4

        def peek(self, tokens: Any) -> _FakePrefixHit:
            return _FakePrefixHit()

        def stats(self) -> PrefixCacheStats:
            return PrefixCacheStats(
                block_size=4,
                hits=0,
                num_blocks=None,
                resident_bytes=None,
                logical_bytes=None,
                has_codec=False,
            )

    tok = _FakeTokenizer(eos_token_ids={99})
    adapter = _FakeAdapter(tok)
    engine = _FakeBatchedEngine([65, 66])
    pc = _StatsNoneFields()
    session = ChatSession(
        adapter,
        engine,
        prefix_cache=pc,  # type: ignore[arg-type]
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 0.0,
    )
    metrics = session.chat("hi")
    assert metrics.prefix_store_resident_bytes is None
    assert metrics.prefix_store_logical_bytes is None


def test_no_prefix_cache_leaves_prefix_store_fields_none() -> None:
    """Single-request path (no cache wired) leaves prefix-store
    fields at None so the toolbar's fallback to engine-budget
    figures kicks in."""
    session, _, _ = _build_session(engine_tokens=[65, 66, 67])
    metrics = session.chat("hi")
    assert metrics.prefix_store_resident_bytes is None
    assert metrics.prefix_store_logical_bytes is None


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


class _DrainProbeEngine:
    """Engine fake that records whether the consumer drained the
    full ``generate_batch`` stream.

    The real ``ContinuousBatcher`` runs prefix-cache insertion in a
    DEFERRED reclaim step that fires on the iteration AFTER the
    terminal ``done`` event. If ChatSession breaks out of the
    event loop on ``done``, the generator is closed before the
    reclaim step runs and ``_extract_and_insert_prefix`` never
    fires — silica/scheduler/batcher.py §reclaim_terminated. The
    cache stays empty and ``prefix_hit=N/M`` permanently reads
    0/N regardless of conversation length.

    This fake yields a ``post_done_marker`` event AFTER the
    terminal ``done``, simulating the deferred-reclaim step's
    final yield. If ChatSession drains correctly,
    ``self.fully_drained`` becomes True; if it breaks early it
    stays False — a one-bit signal for this regression.
    """

    def __init__(self, tokens: list[int]) -> None:
        self._tokens = list(tokens)
        self.metrics = _FakeMetrics()
        self.kv_manager = _FakeKVManager()
        self.fully_drained = False

    def generate(
        self, prompt: str, params: SamplingParams | None = None
    ) -> Iterator[int]:
        # Single-request path not exercised by the regression
        # test; keep the method present for Protocol conformance.
        del prompt, params
        yield from self._tokens

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

        del prompts, params, max_batch_size, prefix_cache
        del length_spread_threshold
        for tok in self._tokens:
            yield BatchEvent.token(req_index=0, token_id=tok)
        yield BatchEvent.done(req_index=0, reason="stop_token")
        # Simulated post-terminal yield representing the batcher's
        # next-step reclaim phase. If the consumer broke on
        # ``done``, this never gets reached.
        self.fully_drained = True


def test_prefix_cache_session_drains_after_terminal_event() -> None:
    """Regression: ChatSession must not break on ``done`` — the
    real batcher's prefix-cache insertion runs in a deferred
    reclaim step that fires after the terminal event. Breaking
    early aborts the generator before reclaim, leaving the cache
    empty (the bug observed during C-4 manual smoke; fixed here).

    This test does not exercise the full batcher; it pins the
    consumer-side contract via :class:`_DrainProbeEngine` whose
    ``fully_drained`` flag flips True only when the entire
    generator is iterated to completion.
    """
    tok = _FakeTokenizer(eos_token_ids={99})
    adapter = _FakeAdapter(tok)
    engine = _DrainProbeEngine([65, 66, 67])
    pc = _FakePrefixCache()
    session = ChatSession(
        adapter,
        engine,
        prefix_cache=pc,
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 256.0,
    )
    metrics = session.chat("hi")
    assert engine.fully_drained is True, (
        "ChatSession broke out of the batched event stream before "
        "the generator completed; the batcher's deferred "
        "_extract_and_insert_prefix would never fire and the "
        "prefix cache would stay empty across turns."
    )
    # Sanity: tokens were collected and the finish reason from
    # the terminal event still reached TurnMetrics.
    assert metrics.output_tokens == 3
    assert metrics.finish_reason == "stop_token"


# ---------------------------------------------------------------------------
# CHAT-CLI-RESPONSE-POLICY RP-1 — _strip_thinking_block helper unit tests
# ---------------------------------------------------------------------------


def test_strip_thinking_block_empty_text_unchanged() -> None:
    assert _strip_thinking_block("", implicit_leading=False) == ""
    assert _strip_thinking_block("", implicit_leading=True) == ""


def test_strip_thinking_block_no_tags_returns_unchanged() -> None:
    """``implicit_leading=False`` and no ``<think>`` tags → text
    is returned verbatim."""
    text = "Hello, world!"
    assert (
        _strip_thinking_block(text, implicit_leading=False) == text
    )


def test_strip_thinking_block_implicit_leading_drops_up_to_close_tag() -> None:
    """Implicit-leading: reply starts inside a ``<think>`` block;
    everything up to and including the first ``</think>`` (plus a
    single trailing newline) is dropped."""
    text = "reasoning here\n</think>\nvisible answer"
    out = _strip_thinking_block(text, implicit_leading=True)
    assert out == "visible answer"


def test_strip_thinking_block_implicit_leading_no_close_tag_unchanged() -> None:
    """Implicit-leading, but the model never closed ``</think>``
    (e.g. truncated mid-think). The helper preserves the text
    rather than guessing where reasoning ended."""
    text = "still reasoning, no close tag yet"
    out = _strip_thinking_block(text, implicit_leading=True)
    assert out == text


def test_strip_thinking_block_implicit_leading_no_trailing_newline() -> None:
    """If ``</think>`` is not followed by a newline, the helper
    does not eat any character of the visible text that follows."""
    text = "thoughts</think>visible"
    out = _strip_thinking_block(text, implicit_leading=True)
    assert out == "visible"


def test_strip_thinking_block_explicit_pair_removed() -> None:
    """Explicit ``<think>...</think>`` span gets removed even
    when ``implicit_leading=False``."""
    text = "before<think>secret</think>after"
    out = _strip_thinking_block(text, implicit_leading=False)
    assert out == "beforeafter"


def test_strip_thinking_block_explicit_pair_with_trailing_newline() -> None:
    """Trailing newline after the closing tag is consumed by the
    explicit-strip path too, mirroring the implicit-leading
    behaviour."""
    text = "intro\n<think>x</think>\nbody"
    out = _strip_thinking_block(text, implicit_leading=False)
    assert out == "intro\nbody"


def test_strip_thinking_block_unclosed_explicit_drops_remainder() -> None:
    """An ``<think>`` with no matching close tag is conservative:
    the helper drops everything from the open tag onward."""
    text = "before<think>truncated"
    out = _strip_thinking_block(text, implicit_leading=False)
    assert out == "before"


def test_strip_thinking_block_multiple_explicit_pairs() -> None:
    """All ``<think>...</think>`` spans are removed, in order."""
    text = "a<think>1</think>b<think>2</think>c"
    out = _strip_thinking_block(text, implicit_leading=False)
    assert out == "abc"


def test_strip_thinking_block_implicit_plus_trailing_explicit() -> None:
    """Composition: implicit-leading first, then explicit pairs in
    the remainder."""
    text = "lead</think>\npart1<think>more</think>part2"
    out = _strip_thinking_block(text, implicit_leading=True)
    assert out == "part1part2"


# ---------------------------------------------------------------------------
# CHAT-CLI-RESPONSE-POLICY RP-1 — ChatSession integration
# ---------------------------------------------------------------------------


def _engine_yielding_text(
    text: str, *, eos_id: int | None = None
) -> _FakeEngine:
    """Build an engine fake that decodes byte-for-byte to ``text``
    via ``_FakeTokenizer.decode`` (each char → ``ord(char)``).
    Optionally append an EOS token id so the generated turn
    finishes with ``finish_reason='stop_token'``."""
    tokens = [ord(c) for c in text]
    if eos_id is not None:
        tokens.append(eos_id)
    return _FakeEngine(tokens)


def _make_session_for_history_test(
    *,
    reply_text: str,
    eos_id: int | None,
    thinking_history: str = "strip",
    implicit_thinking_supported: bool = False,
    thinking_mode: bool | None = None,
) -> tuple[ChatSession, _FakeEngine]:
    """Wire a session whose engine decodes byte-for-byte to
    ``reply_text``."""
    eos_set: set[int] = set()
    if eos_id is not None:
        eos_set.add(eos_id)
    tok = _FakeTokenizer(eos_token_ids=eos_set)
    adapter = _FakeAdapter(tok)
    engine = _engine_yielding_text(reply_text, eos_id=eos_id)
    session = ChatSession(
        adapter,  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        thinking_mode=thinking_mode,
        thinking_history=thinking_history,
        implicit_thinking_supported=implicit_thinking_supported,
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 0.0,
    )
    return session, engine


def test_chat_session_strip_natural_completion_drops_implicit_think() -> None:
    """Natural completion (EOS) under ``thinking_history=strip``
    with implicit-leading support → the assistant message is
    stripped and ``raw_reply`` carries the full decoded text."""
    eos = 250
    session, _ = _make_session_for_history_test(
        reply_text="reasoning\n</think>\nvisible answer",
        eos_id=eos,
        thinking_history="strip",
        implicit_thinking_supported=True,
        thinking_mode=True,
    )
    metrics = session.chat("hi")
    assert metrics.finish_reason == "stop_token"
    assert metrics.reply == "visible answer"
    assert metrics.raw_reply == "reasoning\n</think>\nvisible answer"
    # Stored message matches metrics.reply (post-strip).
    assert session.messages[-1]["content"] == "visible answer"


def test_chat_session_strip_natural_completion_with_explicit_pair() -> None:
    """Strip mode with ``implicit_thinking_supported=False`` still
    handles explicit ``<think>...</think>`` pairs."""
    eos = 250
    session, _ = _make_session_for_history_test(
        reply_text="prelude<think>hidden</think>postlude",
        eos_id=eos,
        thinking_history="strip",
        implicit_thinking_supported=False,
        thinking_mode=False,
    )
    metrics = session.chat("hi")
    assert metrics.reply == "preludepostlude"
    assert metrics.raw_reply == "prelude<think>hidden</think>postlude"


def test_chat_session_keep_mode_stores_raw() -> None:
    """``thinking_history=keep`` writes the raw decoded reply
    verbatim to history regardless of finish_reason or
    implicit-leading support."""
    eos = 250
    raw = "reasoning\n</think>\nvisible answer"
    session, _ = _make_session_for_history_test(
        reply_text=raw,
        eos_id=eos,
        thinking_history="keep",
        implicit_thinking_supported=True,
        thinking_mode=True,
    )
    metrics = session.chat("hi")
    assert metrics.reply == raw
    assert metrics.raw_reply == raw
    assert session.messages[-1]["content"] == raw


def test_chat_session_strip_max_tokens_defers_finalize() -> None:
    """Truncation under strip mode keeps the raw reply on the
    assistant message (RP-2 ``/continue`` needs the prefix) and
    sets the deferred-finalize flag."""
    # No EOS in the engine output, so finish_reason is max_tokens
    # once params.max_tokens is reached.
    raw = "reasoning"  # no </think>; mid-think truncation
    session, engine = _make_session_for_history_test(
        reply_text=raw,
        eos_id=None,
        thinking_history="strip",
        implicit_thinking_supported=True,
        thinking_mode=True,
    )
    # max_tokens=len(raw) so engine produces exactly that many.
    metrics = session.chat("hi", sampling_params=SamplingParams(max_tokens=len(raw)))
    assert metrics.finish_reason == "max_tokens"
    # Assistant message is RAW (deferred strip).
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["content"] == raw
    assert metrics.reply == raw
    assert metrics.raw_reply == raw
    # The deferred-finalize flag is what drives RP-1's strip-on-
    # next-user-turn path. Test the observable consequence by
    # running a second turn below.


def test_chat_session_strip_max_tokens_finalises_on_next_user_turn() -> None:
    """After a max_tokens turn under strip mode, the next
    ``chat()`` call finalises the previous assistant message
    before the new user message lands."""
    eos = 250
    # Turn 1: truncated mid-think (no </think> → strip leaves raw
    # unchanged because the helper has no anchor; we use an
    # explicit pair so finalisation has something to remove).
    session, _ = _make_session_for_history_test(
        reply_text="<think>secret</think>oops",
        eos_id=None,  # max_tokens path
        thinking_history="strip",
        implicit_thinking_supported=False,
        thinking_mode=False,
    )
    raw_len = len("<think>secret</think>oops")
    session.chat(
        "first", sampling_params=SamplingParams(max_tokens=raw_len)
    )
    # Pre-finalise: raw on assistant message.
    assert (
        session.messages[-1]["content"]
        == "<think>secret</think>oops"
    )

    # Swap engine for turn 2 so the second chat() has fresh
    # tokens. The session's existing _engine is exhausted.
    eos_set = {eos}
    tok = session._tokenizer  # reuse the same fake
    tok.eos_token_ids = eos_set  # type: ignore[attr-defined]
    session._engine = _engine_yielding_text(  # type: ignore[assignment]
        "done", eos_id=eos
    )

    session.chat("second")

    # After turn 2 starts, the previous truncated assistant
    # message is finalised: raw stripped → "oops".
    msgs = session.messages
    # roles: [user1, asst1-stripped, user2, asst2]
    assert [m["role"] for m in msgs] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert msgs[1]["content"] == "oops"


def test_chat_session_set_thinking_history_mutator_switches_modes() -> None:
    """``set_thinking_history`` lets the chat-CLI flip the policy
    mid-session without rebuilding ChatSession."""
    eos = 250
    session, _ = _make_session_for_history_test(
        reply_text="<think>secret</think>visible",
        eos_id=eos,
        thinking_history="strip",
    )
    m1 = session.chat("first")
    assert m1.reply == "visible"

    session.set_thinking_history("keep")
    session._engine = _engine_yielding_text(  # type: ignore[assignment]
        "<think>second-secret</think>visible-2", eos_id=eos
    )
    m2 = session.chat("second")
    # Under keep mode, raw is preserved.
    assert m2.reply == "<think>second-secret</think>visible-2"


def test_chat_session_set_thinking_history_invalid_raises() -> None:
    session, _ = _make_session_for_history_test(
        reply_text="x", eos_id=None
    )
    with pytest.raises(ValueError, match="thinking_history"):
        session.set_thinking_history("delete")  # type: ignore[arg-type]


def test_chat_session_default_thinking_history_is_strip() -> None:
    """A ChatSession constructed without an explicit
    ``thinking_history`` kwarg defaults to strip mode."""
    eos = 250
    tok = _FakeTokenizer(eos_token_ids={eos})
    adapter = _FakeAdapter(tok)
    engine = _engine_yielding_text(
        "before<think>x</think>after", eos_id=eos
    )
    session = ChatSession(
        adapter,  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        reset_peak_memory=lambda: None,
        read_peak_memory_mb=lambda: 0.0,
    )
    metrics = session.chat("hi")
    # Default strip + no implicit-leading flag → explicit pairs
    # are removed, implicit-leading is not assumed.
    assert metrics.reply == "beforeafter"


def test_chat_session_implicit_leading_only_when_thinking_mode_not_false() -> None:
    """When ``thinking_mode=False`` the chat template did NOT
    prepend ``<think>\\n``, so implicit-leading strip must NOT
    apply even if the model is from a Qwen3-shaped family."""
    eos = 250
    raw = "fake-leading text\n</think>\nrest"
    session, _ = _make_session_for_history_test(
        reply_text=raw,
        eos_id=eos,
        thinking_history="strip",
        implicit_thinking_supported=True,
        thinking_mode=False,  # <-- key
    )
    metrics = session.chat("hi")
    # Because thinking_mode is False, implicit-leading strip
    # is skipped. Explicit pairs are still stripped — but raw
    # has none, so output equals input.
    assert metrics.reply == raw


# ---------------------------------------------------------------------------
# RP-1 repair (post-4f98648): deferred-finalise correctness
# ---------------------------------------------------------------------------


def test_pending_finalise_uses_truncation_time_implicit_snapshot() -> None:
    """If thinking_mode flips between the truncated turn and the
    next user message, the deferred finalise must use the snapshot
    captured at truncation — not the current live mode. Otherwise
    a turn generated with thinking-on (implicit-leading text) would
    NOT get its leading reasoning stripped after the user flips
    thinking_mode off."""
    eos = 250
    raw = "leading reasoning\n</think>\nvisible body"
    session, _ = _make_session_for_history_test(
        reply_text=raw,
        eos_id=None,  # forces max_tokens path
        thinking_history="strip",
        implicit_thinking_supported=True,
        thinking_mode=True,  # turn 1: thinking on
    )
    # Turn 1: truncated mid-completion; deferred state captured
    # while thinking_mode=True.
    session.chat(
        "first",
        sampling_params=SamplingParams(max_tokens=len(raw)),
    )
    # User flips thinking_mode off between turns. WITHOUT the
    # snapshot fix, the deferred finalise would now consult the
    # live (False) mode and skip implicit-leading strip — leaving
    # "leading reasoning" embedded in history.
    session.set_thinking_mode(False)
    # Swap engine for turn 2.
    session._engine = _engine_yielding_text(  # type: ignore[assignment]
        "ack", eos_id=eos
    )
    tok = session._tokenizer
    tok.eos_token_ids = {eos}  # type: ignore[attr-defined]
    session.chat("second")
    # The first assistant message must show implicit-leading
    # strip applied — "visible body" is what survives.
    assert session.messages[1]["content"] == "visible body"


def test_pending_finalise_respects_keep_when_user_flips_history() -> None:
    """If the user flips thinking_history=strip→keep between the
    truncated turn and the next user message, the deferred path
    must NOT strip — the user explicitly asked to preserve raw."""
    eos = 250
    raw = "<think>secret</think>oops"
    session, _ = _make_session_for_history_test(
        reply_text=raw,
        eos_id=None,
        thinking_history="strip",
        implicit_thinking_supported=False,
        thinking_mode=False,
    )
    session.chat(
        "first",
        sampling_params=SamplingParams(max_tokens=len(raw)),
    )
    # Pre-finalise: raw on assistant.
    assert session.messages[-1]["content"] == raw
    # User flips history policy to keep.
    session.set_thinking_history("keep")
    session._engine = _engine_yielding_text(  # type: ignore[assignment]
        "ack", eos_id=eos
    )
    tok = session._tokenizer
    tok.eos_token_ids = {eos}  # type: ignore[attr-defined]
    session.chat("second")
    # Under live keep policy, the deferred finalise must not
    # strip — raw is preserved on the prior assistant message.
    assert session.messages[1]["content"] == raw


def test_reset_clears_pending_finalise_flag() -> None:
    """``reset()`` drops the truncated assistant message; the
    pending-finalise flag must clear too so subsequent turns do
    not try to strip a now-missing (or replaced) message."""
    session, _ = _make_session_for_history_test(
        reply_text="<think>x</think>oops",
        eos_id=None,
        thinking_history="strip",
        implicit_thinking_supported=False,
    )
    raw_len = len("<think>x</think>oops")
    session.chat(
        "first", sampling_params=SamplingParams(max_tokens=raw_len)
    )
    assert session._pending_finalize is True
    session.reset()
    assert session._pending_finalize is False
    assert session._pending_finalize_implicit_leading is None


def test_replace_messages_clears_pending_finalise_flag() -> None:
    """``/load`` route — wholesale message replacement clears the
    deferred-finalise state because the prior assistant target is
    gone."""
    session, _ = _make_session_for_history_test(
        reply_text="<think>x</think>oops",
        eos_id=None,
        thinking_history="strip",
        implicit_thinking_supported=False,
    )
    raw_len = len("<think>x</think>oops")
    session.chat(
        "first", sampling_params=SamplingParams(max_tokens=raw_len)
    )
    assert session._pending_finalize is True
    session.replace_messages(
        [
            {"role": "system", "content": "fresh"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    )
    assert session._pending_finalize is False
    assert session._pending_finalize_implicit_leading is None


def test_pop_last_exchange_clears_pending_finalise_flag() -> None:
    """``/regenerate`` route — popping the last (user, assistant)
    pair removes the deferred-finalise target; the flag must
    clear so the regenerated turn does not strip a stale buffer."""
    session, _ = _make_session_for_history_test(
        reply_text="<think>x</think>oops",
        eos_id=None,
        thinking_history="strip",
        implicit_thinking_supported=False,
    )
    raw_len = len("<think>x</think>oops")
    session.chat(
        "first", sampling_params=SamplingParams(max_tokens=raw_len)
    )
    assert session._pending_finalize is True
    popped = session.pop_last_exchange()
    assert popped == "first"
    assert session._pending_finalize is False
    assert session._pending_finalize_implicit_leading is None


def test_pending_finalise_snapshot_cleared_on_natural_completion() -> None:
    """A turn that completes naturally (EOS) under strip mode
    must clear the snapshot too — only truncated turns set it."""
    eos = 250
    session, _ = _make_session_for_history_test(
        reply_text="<think>x</think>visible",
        eos_id=eos,
        thinking_history="strip",
        implicit_thinking_supported=False,
    )
    session.chat("hi")
    assert session._pending_finalize is False
    assert session._pending_finalize_implicit_leading is None
