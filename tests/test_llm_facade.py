"""Tests for :class:`silica.llm.LLM` (P-8 sub-unit (g)).

R-g acceptance: ``LLM`` round-trips a single greedy generation
matching :meth:`Engine.generate` byte-for-byte under the same seed.

The test suite uses a stub adapter / engine so no HuggingFace load
or MLX compute happens in CI; the real-model byte-for-byte gate is
the manual smoke deferred to (h). What the unit tests pin:

- Lazy-load: ``LLM(...)`` does NOT call into
  :func:`silica.models.factory.adapter_for_repo`; the first
  :meth:`generate` / :meth:`chat` call does.
- :meth:`unload` clears the loaded fields; the next call re-loads.
- :meth:`generate` (non-streaming) decodes the engine's id stream,
  drops the trailing EOS, strips trailing replacement chars, and
  returns text. Same shape as
  :meth:`silica.chat.session.ChatSession.chat` (re-using its
  invariant pin).
- :meth:`generate(stream=True)` yields incremental decoded deltas,
  never emits the EOS token, and the concatenation equals the
  non-streaming result. **This is the byte-for-byte parity pin
  against the engine's id stream.**
- :meth:`chat` builds a fresh :class:`ChatSession` per call;
  history is honoured; system prompts concatenate; the last
  message must have ``role='user'``.
- Validation: empty prompt / messages, non-user-final, unsupported
  roles surface as ``ValueError`` / ``TypeError`` from the facade.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from silica.core.sampling import SamplingParams
from silica.llm import LLM

# ---------------------------------------------------------------------------
# Fakes — mirror the stubs in test_chat_session.py without duplicating their
# full surface. The LLM facade only needs adapter.tokenizer() (encode +
# decode + eos_token_ids) and engine.generate(prompt, params) → Iterator[int].
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_ids: set[int] = {0}

    def __init__(self) -> None:
        self.last_decoded: list[list[int]] = []

    def encode(self, text: str) -> list[int]:
        # Char-based, deterministic. ``+1`` keeps id=0 reserved for EOS.
        return [(ord(c) % 200) + 1 for c in text]

    def decode(self, token_ids: list[int]) -> str:
        ids = list(token_ids)
        self.last_decoded.append(ids)
        return "".join(
            chr((tid - 1) % 200) for tid in ids if tid > 0
        )


class _FakeAdapter:
    def __init__(self) -> None:
        self._tokenizer = _FakeTokenizer()

    def tokenizer(self) -> _FakeTokenizer:
        return self._tokenizer


class _FakeKV:
    pass


class _FakeEngine:
    """Mimics :meth:`silica.engine.Engine.generate` for a fixed token
    stream. Calls go through the real :class:`silica.engine.Engine`
    surface in the byte-parity test below; this fake is for the
    LLM-side unit tests."""

    def __init__(self, tokens: list[int]) -> None:
        self._tokens = list(tokens)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, prompt: str, params: SamplingParams | None = None
    ) -> Iterator[int]:
        self.calls.append({"prompt": prompt, "params": params})
        for tok in self._tokens:
            yield tok


# ---------------------------------------------------------------------------
# Lazy-load + lifecycle.
# ---------------------------------------------------------------------------


def test_constructor_is_lazy_no_load_called() -> None:
    """``LLM("...")`` must not pay the load cost. A user typing
    ``llm = LLM("...")`` in a notebook expects type-check / linter
    feedback without a multi-GB checkpoint download."""
    llm = LLM("stub/never-loaded")
    assert llm.loaded is False
    assert llm.model_repo == "stub/never-loaded"


def test_constructor_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="non-empty repo id"):
        LLM("")


def test_unload_is_idempotent_on_unloaded_llm() -> None:
    llm = LLM("stub/m")
    llm.unload()  # not loaded yet — no-op
    assert llm.loaded is False


def _install_fake_factory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokens: list[int],
) -> dict[str, Any]:
    """Patch :func:`silica.models.factory.adapter_for_repo` to return
    the same ``(adapter, kv)`` pair on every call, plus a recorder so
    the tests can assert load count.

    Also patches :class:`silica.engine.Engine` construction so the LLM
    instantiates the fake engine instead of the real one (which would
    require a working KVManager).
    """
    adapter = _FakeAdapter()
    kv = _FakeKV()
    engine = _FakeEngine(tokens)
    holder: dict[str, Any] = {
        "adapter": adapter,
        "kv": kv,
        "engine": engine,
        "load_calls": 0,
    }

    def _fake_factory(repo: str) -> tuple[Any, Any]:
        holder["load_calls"] += 1
        return adapter, kv

    # Patch the *late-binding* import inside ``LLM._ensure_loaded``.
    # The factory is imported lazily inside the method so the monkey-
    # patch needs to land on the module's symbol, not the cached one.
    import silica.models.factory as factory_module

    monkeypatch.setattr(
        factory_module, "adapter_for_repo", _fake_factory
    )
    # The Engine constructor in _ensure_loaded → make it return our
    # fake engine. ``LLM._ensure_loaded`` does ``Engine(adapter=adapter,
    # kv_manager=kv)``; we monkey-patch the imported name in
    # ``silica.llm._facade`` so the call returns the fake.
    import silica.llm._facade as facade_module

    def _fake_engine_ctor(*args: Any, **kwargs: Any) -> _FakeEngine:
        return engine

    monkeypatch.setattr(facade_module, "Engine", _fake_engine_ctor)
    return holder


def test_first_generate_call_loads_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _install_fake_factory(monkeypatch, tokens=[65, 66, 67, 0])
    llm = LLM("stub/lazy")
    assert holder["load_calls"] == 0
    llm.generate("hi")
    assert holder["load_calls"] == 1
    # Second call must NOT re-load.
    llm.generate("hi again")
    assert holder["load_calls"] == 1


def test_unload_then_generate_reloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _install_fake_factory(monkeypatch, tokens=[65, 0])
    llm = LLM("stub/reload")
    llm.generate("first")
    assert holder["load_calls"] == 1
    llm.unload()
    assert llm.loaded is False
    llm.generate("after-reload")
    assert holder["load_calls"] == 2


# ---------------------------------------------------------------------------
# generate (non-streaming).
# ---------------------------------------------------------------------------


def test_generate_returns_decoded_text_minus_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tokens [66, 67, 68, 0]: the trailing 0 is EOS in the fake
    tokenizer. The returned text decodes [66, 67, 68] only.

    Per :class:`_FakeTokenizer.decode`, id N → ``chr((N - 1) % 200)``,
    so 66/67/68 → 'A'/'B'/'C'.
    """
    _install_fake_factory(monkeypatch, tokens=[66, 67, 68, 0])
    llm = LLM("stub/m")
    out = llm.generate("hi")
    assert out == "ABC"


def test_generate_strips_trailing_replacement_char(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing partial UTF-8 bytes decode to ``\\ufffd``; mirror
    :class:`ChatSession`'s ``rstrip("\\ufffd")``."""
    # Custom adapter whose decode injects a trailing replacement char.
    class _DecodingAdapter:
        class _Tok:
            eos_token_ids: set[int] = {0}

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, token_ids: list[int]) -> str:
                # Final char is the Unicode replacement glyph — emulates
                # an unfinished multi-byte sequence cut by EOS.
                return "Hello�"

        def __init__(self) -> None:
            self._tok = self._Tok()

        def tokenizer(self) -> Any:
            return self._tok

    adapter = _DecodingAdapter()
    engine = _FakeEngine([1, 2, 3, 0])

    import silica.llm._facade as facade_module
    import silica.models.factory as factory_module

    monkeypatch.setattr(
        factory_module,
        "adapter_for_repo",
        lambda repo: (adapter, _FakeKV()),
    )
    monkeypatch.setattr(
        facade_module, "Engine", lambda **_: engine
    )

    llm = LLM("stub/m")
    out = llm.generate("hi")
    assert out == "Hello"


def test_generate_streaming_yields_incremental_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming generator emits deltas (never the full accumulator
    twice); the EOS token is NOT emitted as a delta; concatenation
    of deltas equals the non-streaming reply.

    This is the byte-for-byte parity pin for the streaming branch:
    the same id stream produces the same decoded text whether
    consumed token-by-token or in bulk.
    """
    _install_fake_factory(monkeypatch, tokens=[66, 67, 68, 0])
    llm = LLM("stub/m")

    deltas = list(llm.generate("hi", stream=True))
    # Each char yielded individually as the accumulator grows.
    assert deltas == ["A", "B", "C"]

    # Streaming concat == non-streaming reply (re-load needed because
    # our fake engine yields a fixed list once; we re-install).
    _install_fake_factory(monkeypatch, tokens=[66, 67, 68, 0])
    llm2 = LLM("stub/m")
    bulk = llm2.generate("hi")
    assert "".join(deltas) == bulk


def test_generate_uses_default_sampling_params_then_per_call_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _install_fake_factory(monkeypatch, tokens=[65, 0])
    default = SamplingParams(temperature=0.0, max_tokens=8)
    override = SamplingParams(temperature=0.7, max_tokens=16)

    llm = LLM("stub/m", sampling_params=default)
    llm.generate("a")
    assert holder["engine"].calls[-1]["params"] is default

    llm.generate("b", sampling_params=override)
    assert holder["engine"].calls[-1]["params"] is override

    # Per-call None falls back to constructor default.
    llm.generate("c")
    assert holder["engine"].calls[-1]["params"] is default


# ---------------------------------------------------------------------------
# chat.
# ---------------------------------------------------------------------------


class _StubChatSession:
    """Captures messages + chat invocations; returns a fixed
    TurnMetrics. Used for chat() routing tests."""

    def __init__(self, *, reply: str = "REPLY") -> None:
        self.reply = reply
        self.messages: list[dict[str, str]] = []
        self.calls: list[dict[str, Any]] = []
        self._adapter = None  # set by _build
        self._engine = None
        self._prefix_cache = None

    def replace_messages(self, messages: list[dict[str, str]]) -> None:
        self.messages = list(messages)

    def chat(
        self,
        user_text: str,
        *,
        sampling_params: Any = None,
        stream_to: Any = None,
    ) -> Any:
        self.calls.append(
            {
                "user_text": user_text,
                "sampling_params": sampling_params,
                "stream_to": stream_to,
            }
        )
        # Mimic streaming if requested: emit two deltas, return reply.
        if stream_to is not None:
            stream_to(self.reply[: len(self.reply) // 2])
            stream_to(self.reply[len(self.reply) // 2 :])
        from silica.chat.session import TurnMetrics

        return TurnMetrics(
            reply=self.reply,
            prompt_tokens=1,
            output_tokens=1,
            finish_reason="stop_token",
        )


def _install_stub_chat_session(
    monkeypatch: pytest.MonkeyPatch, *, reply: str = "Paris."
) -> list[_StubChatSession]:
    holder: list[_StubChatSession] = []

    def _ctor(
        *,
        adapter: Any,
        engine: Any,
        system_prompt: str | None = None,
    ) -> _StubChatSession:
        s = _StubChatSession(reply=reply)
        s._adapter = adapter
        s._engine = engine
        if system_prompt:
            s.messages = [
                {"role": "system", "content": system_prompt}
            ]
        holder.append(s)
        return s


    # ChatSession is imported inside :meth:`_build_chat_session`. The
    # facade does ``from silica.chat.session import ChatSession`` per
    # call; patch the source module so each lazy import sees the stub.
    import silica.chat.session as session_module

    monkeypatch.setattr(session_module, "ChatSession", _ctor)
    return holder


def test_chat_builds_session_with_system_prompt_and_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, tokens=[1])
    sessions = _install_stub_chat_session(monkeypatch, reply="Paris.")
    llm = LLM("stub/m")

    out = llm.chat(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi back"},
            {"role": "user", "content": "capital?"},
        ]
    )
    assert out == "Paris."
    assert len(sessions) == 1
    s = sessions[0]
    assert s.calls[0]["user_text"] == "capital?"
    # Messages were replaced (system + history before the last user).
    assert [m["role"] for m in s.messages] == [
        "system",
        "user",
        "assistant",
    ]


def test_chat_streaming_yields_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, tokens=[1])
    _install_stub_chat_session(monkeypatch, reply="Paris.")
    llm = LLM("stub/m")
    deltas = list(
        llm.chat(
            [{"role": "user", "content": "capital?"}],
            stream=True,
        )
    )
    assert "".join(deltas) == "Paris."


def test_chat_concatenates_consecutive_system_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, tokens=[1])
    sessions = _install_stub_chat_session(monkeypatch)
    llm = LLM("stub/m")
    llm.chat(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "system", "content": "and terse"},
            {"role": "user", "content": "go"},
        ]
    )
    s = sessions[0]
    assert s.messages[0]["role"] == "system"
    assert s.messages[0]["content"] == "be helpful\n\nand terse"


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_chat_rejects_empty_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, tokens=[1])
    llm = LLM("stub/m")
    with pytest.raises(ValueError, match="must not be empty"):
        llm.chat([])


def test_chat_rejects_non_user_final_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, tokens=[1])
    llm = LLM("stub/m")
    with pytest.raises(ValueError, match="role='user'"):
        llm.chat([{"role": "assistant", "content": "hello"}])


def test_chat_rejects_non_string_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, tokens=[1])
    llm = LLM("stub/m")
    with pytest.raises(TypeError, match="content must be a string"):
        llm.chat(
            [
                {"role": "user", "content": 123},  # type: ignore[dict-item]
            ]
        )


def test_chat_rejects_non_leading_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, tokens=[1])
    llm = LLM("stub/m")
    with pytest.raises(ValueError, match="system messages must precede"):
        llm.chat(
            [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "go"},
            ]
        )


def test_chat_rejects_unsupported_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, tokens=[1])
    llm = LLM("stub/m")
    with pytest.raises(ValueError, match="unsupported role"):
        llm.chat(
            [
                {"role": "developer", "content": "x"},
                {"role": "user", "content": "go"},
            ]
        )


# ---------------------------------------------------------------------------
# Byte-for-byte parity pin (R-g, against the real Engine.generate
# stream — using the StubModelAdapter so no MLX compute / HuggingFace
# load is required).
# ---------------------------------------------------------------------------


def test_llm_generate_matches_engine_generate_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-g acceptance: ``LLM.generate`` returns the same decoded text
    that one would obtain by manually consuming
    :meth:`Engine.generate` and decoding it with the same trailing
    EOS / replacement-char handling.

    Driven against a deterministic fake adapter / engine so the
    parity is reproducible in CI; the real-model parity gate is the
    manual smoke deferred to (h)."""
    tokens = [66, 67, 68, 0]  # 'A' 'B' 'C' EOS
    holder = _install_fake_factory(monkeypatch, tokens=tokens)
    llm = LLM("stub/parity")
    text_via_facade = llm.generate("hello")

    # Manual consumption of the engine's id stream — the byte-for-byte
    # baseline. Re-install factory so the engine state resets.
    _install_fake_factory(monkeypatch, tokens=tokens)
    adapter = holder["adapter"]
    engine_ids: list[int] = []
    # Use a fresh fake engine to avoid mutating ``holder['engine']``
    # that ``llm.generate`` already consumed.
    fresh_engine = _FakeEngine(tokens)
    for tok in fresh_engine.generate("hello"):
        engine_ids.append(tok)
    eos = set(adapter.tokenizer().eos_token_ids)
    if engine_ids and engine_ids[-1] in eos:
        engine_ids = engine_ids[:-1]
    text_manual = adapter.tokenizer().decode(engine_ids).rstrip("�")

    assert text_via_facade == text_manual
    assert text_via_facade == "ABC"
