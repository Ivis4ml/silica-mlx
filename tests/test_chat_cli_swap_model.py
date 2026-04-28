"""Tests for ``silica.chat.cli.app._swap_model`` keep-history wiring.

CHAT-CLI-HARDENING-5 (F5). The pre-HARDENING-5 ``_swap_model`` reset
the conversation log unconditionally; ``/model new-repo`` silently
dropped the chat. These tests inject fakes for the heavy
dependencies (model loader, engine, session, cache) and verify that
``keep_history`` correctly propagates: True preserves the message
log via ``replace_messages``, False matches the legacy reset
behaviour. The session-construction error path is covered too — a
``get_adapter`` raise must leave the prior triple intact and not
mutate state.
"""

from __future__ import annotations

import argparse
from typing import Any

from silica.chat.cli.app import _swap_model
from silica.chat.cli.palette import Palette
from silica.chat.cli.state import ChatCliState


class _FakeSession:
    """Minimal ChatSession surrogate exposing the read/write surface
    ``_swap_model`` needs: ``messages`` property and
    ``replace_messages``."""

    def __init__(
        self,
        *,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        self._messages: list[dict[str, str]] = (
            list(messages) if messages else []
        )

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def replace_messages(
        self, messages: list[dict[str, str]]
    ) -> None:
        self._messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]


def _make_session(
    _adapter: Any,
    _engine: Any,
    *,
    system_prompt: str | None = None,
    **_: Any,
) -> _FakeSession:
    """``session_cls`` factory: matches ``ChatSession.__init__``
    positional shape (adapter, engine) and seeds the message log
    with the system prompt the way the real ChatSession would."""
    if system_prompt:
        return _FakeSession(
            messages=[{"role": "system", "content": system_prompt}]
        )
    return _FakeSession()


def _make_args(*, kv_codec: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(kv_codec=kv_codec)


def _seeded_state(*, model_name: str = "old-model") -> ChatCliState:
    state = ChatCliState(model_name=model_name)
    state.turn = 5
    state.last_turn_thinking = "prior reasoning"
    state.total_decode_tokens = 1000
    state.total_decode_seconds = 10.0
    state.tok_per_sec = 100.0
    state.total_prefix_hit_tokens = 200
    state.kv_resident_mb = 300.0
    state.last_ttft_ms = 50.0
    return state


def _swap(
    *,
    keep_history: bool,
    prior_messages: list[dict[str, str]] | None = None,
    state: ChatCliState | None = None,
    raise_on_load: Exception | None = None,
) -> tuple[Any, Any, Any] | None:
    """Drive ``_swap_model`` with reusable fakes."""

    def get_adapter(_repo: str) -> tuple[Any, Any]:
        if raise_on_load is not None:
            raise raise_on_load
        return ("new-adapter", "new-kv")

    def engine_cls(_adapter: Any, _kv: Any) -> str:
        return "new-engine"

    def cache_builder(_adapter: Any) -> str:
        return "new-cache"

    return _swap_model(
        "Qwen/Qwen3-4B",
        args=_make_args(),
        state=state if state is not None else _seeded_state(),
        keep_history=keep_history,
        prior_session=_FakeSession(messages=prior_messages),
        get_adapter=get_adapter,
        engine_cls=engine_cls,
        session_cls=_make_session,
        cache_builder=cache_builder,
        palette=Palette.plain(),
    )


# ---------------------------------------------------------------------------
# default behaviour: keep_history=False (legacy reset)
# ---------------------------------------------------------------------------


def test_swap_drops_history_by_default() -> None:
    """``keep_history=False`` (the default) wipes the conversation
    log on the new session — only the system prompt remains, exactly
    what ``ChatSession(system_prompt=...)`` puts there at
    construction."""
    state = _seeded_state()
    state.config["system_prompt"] = "be terse"
    prior = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    outcome = _swap(
        keep_history=False, prior_messages=prior, state=state
    )
    assert outcome is not None
    _adapter, _engine, new_session = outcome
    # New session was built with the system prompt only; no
    # replace_messages call propagated user / assistant content.
    assert [m["role"] for m in new_session.messages] == ["system"]
    assert new_session.messages[0]["content"] == "be terse"


def test_swap_drops_history_resets_conversation_state() -> None:
    """``keep_history=False`` zeroes turn-counter / cumulative
    metrics so the showcase narrative reflects the fresh session."""
    state = _seeded_state()
    outcome = _swap(keep_history=False, prior_messages=[], state=state)
    assert outcome is not None
    assert state.turn == 0
    assert state.last_turn_thinking == ""
    assert state.total_decode_tokens == 0
    assert state.total_decode_seconds == 0.0
    assert state.total_prefix_hit_tokens == 0
    assert state.tok_per_sec is None


def test_swap_drops_history_updates_model_name() -> None:
    """The basename of the new repo replaces ``state.model_name``."""
    state = _seeded_state()
    outcome = _swap(keep_history=False, state=state)
    assert outcome is not None
    assert state.model_name == "Qwen3-4B"


# ---------------------------------------------------------------------------
# keep_history=True
# ---------------------------------------------------------------------------


def test_swap_keep_history_propagates_messages_to_new_session() -> None:
    """``keep_history=True`` calls ``replace_messages`` on the new
    session with the prior session's text-level log; user /
    assistant content survives the swap byte-equivalently."""
    state = _seeded_state()
    state.config["system_prompt"] = "be precise"
    prior = [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "what is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    outcome = _swap(
        keep_history=True, prior_messages=prior, state=state
    )
    assert outcome is not None
    _adapter, _engine, new_session = outcome
    assert new_session.messages == prior


def test_swap_keep_history_preserves_conversation_state() -> None:
    """Turn counter and cumulative metrics survive when history
    does — the showcase narrative is conversation-level, not
    model-level."""
    state = _seeded_state()
    outcome = _swap(
        keep_history=True,
        prior_messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        state=state,
    )
    assert outcome is not None
    assert state.turn == 5
    assert state.last_turn_thinking == "prior reasoning"
    assert state.total_decode_tokens == 1000
    assert state.total_decode_seconds == 10.0


def test_swap_keep_history_resets_model_derived_metrics() -> None:
    """Model-derived metrics (KV residency, last TTFT, live decode
    tok/s) reset regardless of keep_history — they refer to the
    swapped-in model and would mislead if carried forward.
    ``tok_per_sec`` in particular is the in-flight turn's decode
    speed, not a conversation-level rolling figure, so the
    toolbar must not continue to show the prior model's number."""
    state = _seeded_state()
    outcome = _swap(
        keep_history=True, prior_messages=[], state=state
    )
    assert outcome is not None
    assert state.kv_resident_mb is None
    assert state.last_ttft_ms is None
    assert state.prefix_hit_blocks is None
    assert state.tok_per_sec is None


def test_swap_keep_history_with_empty_prior_is_no_op_on_messages() -> None:
    """A fresh prior session (no user / assistant turns) yields a
    new session whose only message is the system prompt the
    ChatSession constructor inserted — replace_messages overwrites
    that with the empty prior, leaving zero messages."""
    state = _seeded_state()
    state.config.pop("system_prompt", None)
    outcome = _swap(
        keep_history=True, prior_messages=[], state=state
    )
    assert outcome is not None
    _adapter, _engine, new_session = outcome
    assert new_session.messages == []


# ---------------------------------------------------------------------------
# error path: get_adapter raises
# ---------------------------------------------------------------------------


def test_swap_load_failure_returns_none_and_does_not_mutate_state() -> None:
    """A ``get_adapter`` raise (e.g. unknown repo, network error)
    leaves the prior session triple intact: ``_swap_model`` returns
    ``None`` and ``state`` is untouched. The shell's caller checks
    the ``None`` return and keeps using the old model."""
    state = _seeded_state()
    pre_swap_turn = state.turn
    pre_swap_thinking = state.last_turn_thinking
    pre_swap_model = state.model_name
    outcome = _swap(
        keep_history=True,
        prior_messages=[{"role": "user", "content": "hi"}],
        state=state,
        raise_on_load=RuntimeError("HF gated repo"),
    )
    assert outcome is None
    assert state.turn == pre_swap_turn
    assert state.last_turn_thinking == pre_swap_thinking
    assert state.model_name == pre_swap_model
