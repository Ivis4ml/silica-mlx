"""Tests for SessionManager wiring into POST /v1/chat/completions
(P-8 sub-unit (f)).

Covers the route-level contract:

- ``X-Silica-Session-ID`` HTTP header threads through to
  :meth:`SessionManager.get_or_create`.
- ``extension.session_id`` body field threads through when the
  header is absent.
- Header takes precedence when both are present.
- Empty selector is treated as absent (a fresh per-call session,
  not a persistent session keyed by ``""``).
- Multiple requests with the same selector reuse the same
  :class:`ChatSession` instance, which is the load-bearing
  pre-condition for cross-request prefix reuse.
- Concurrent requests sharing a session_id queue on
  ``runtime.engine_lock``: the second request's
  :meth:`replace_messages` only fires after the first turn's
  ``ChatSession.chat`` returns. Sub-unit (f) concurrency contract.

Plus the R-f acceptance demo: 3 turns through TestClient with the
shared system prompt and same session_id; assert the persistent
prefix cache sees a peek-hit on turn 2 and turn 3.

The R-f acceptance is exercised against a near-real engine
(:class:`_PrefixInsertingFakeEngine`) that mimics
:class:`ContinuousBatcher.reclaim_terminated` — it inserts the
prompt's block-aligned tokens into the provided
:class:`RadixPrefixCache` before yielding terminal events. K/V
tensors are not constructed (the prefix-cache test surface only
exercises tree shape and refcounts, not actual decode), so we use
:meth:`RadixPrefixCache.insert` instead of ``insert_detached``.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="P-8 [serve] extra not installed")
pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")

from fastapi.testclient import TestClient  # noqa: E402

from silica.chat.session import TurnMetrics  # noqa: E402
from silica.core.events import BatchEvent  # noqa: E402
from silica.core.profiler import MetricsRegistry  # noqa: E402
from silica.core.sampling import SamplingParams  # noqa: E402
from silica.kvcache.manager import NullKVManager  # noqa: E402
from silica.kvcache.prefix import RadixPrefixCache  # noqa: E402
from silica.models.adapter import (  # noqa: E402
    AttentionKind,
    AttentionPattern,
    StubModelAdapter,
)
from silica.models.capabilities import (  # noqa: E402
    capabilities_from_attention_pattern,
)
from silica.server import openai_api  # noqa: E402
from silica.server.routes import chat_completions as cc  # noqa: E402
from silica.server.runtime import Runtime  # noqa: E402

# ---------------------------------------------------------------------------
# Stub session that records its construction and chat invocations.
# Used for the routing tests where R-f's prefix-cache mechanics are
# not under test.
# ---------------------------------------------------------------------------


class _RecordingStubSession:
    """Stub :class:`ChatSession` that records each ``chat`` call.

    Used by the routing tests to verify the route's session-resolution
    branches without exercising prefix-cache mechanics. The R-f
    acceptance demo uses a real ChatSession + real RadixPrefixCache
    (see :func:`_install_prefix_inserting_engine`).
    """

    def __init__(
        self,
        *,
        session_id: str | None,
        reply: str = "ok",
        prompt_tokens: int = 5,
        output_tokens: int = 1,
        prefix_hit_tokens: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.reply = reply
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.prefix_hit_tokens = prefix_hit_tokens
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        user_text: str,
        *,
        sampling_params: Any = None,
        stream_to: Any = None,
    ) -> TurnMetrics:
        self.calls.append(
            {
                "user_text": user_text,
                "sampling_params": sampling_params,
                "stream_to": stream_to,
            }
        )
        # Streaming branch path: emit a single delta so the SSE
        # generator has at least one token chunk to forward before
        # the terminal :class:`TurnMetrics` lands.
        if stream_to is not None:
            stream_to(self.reply)
        return TurnMetrics(
            reply=self.reply,
            prompt_tokens=self.prompt_tokens,
            output_tokens=self.output_tokens,
            finish_reason="stop_token",
            prefix_hit_tokens=self.prefix_hit_tokens,
        )


def _build_runtime() -> Runtime:
    return Runtime(
        StubModelAdapter(),
        NullKVManager(),
        model_repo="stub/model",
        metrics=MetricsRegistry(),
    )


@pytest.fixture(autouse=True)
def _isolate_module_state() -> Iterator[None]:
    openai_api._config = None
    yield
    openai_api._config = None


def _configure_runtime(runtime: Runtime | None = None) -> Runtime:
    runtime = runtime if runtime is not None else _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    return runtime


def _install_recording_manager(
    monkeypatch: pytest.MonkeyPatch, runtime: Runtime
) -> dict[str, Any]:
    """Override ``runtime.session_manager.get_or_create`` to record
    invocations and return a :class:`_RecordingStubSession` per call.

    Returns a holder dict the test populates and inspects after the
    request runs.
    """
    sessions: dict[str, _RecordingStubSession] = {}
    holder: dict[str, Any] = {
        "calls": [],
        "sessions": sessions,
    }

    def _get_or_create(
        session_id: str,
        *,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> _RecordingStubSession:
        holder["calls"].append(
            {
                "session_id": session_id,
                "system_prompt": system_prompt,
                "history": list(history or []),
            }
        )
        # Cache by session_id so identity-based reuse assertions work.
        if session_id not in sessions:
            sessions[session_id] = _RecordingStubSession(
                session_id=session_id
            )
        return sessions[session_id]

    monkeypatch.setattr(
        runtime.session_manager,
        "get_or_create",
        _get_or_create,
    )
    return holder


def _install_default_factory_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """For the no-session_id branch: monkeypatch ``_session_factory``
    to record fresh-session construction. Mirrors the (c)/(d) test
    pattern; the holder lets the test assert the route did NOT touch
    SessionManager when the selector is absent."""
    holder: dict[str, Any] = {"sessions": []}

    def _factory(
        runtime: Runtime,
        *,
        system_prompt: str | None,
        history: list[dict[str, str]],
    ) -> _RecordingStubSession:
        session = _RecordingStubSession(session_id=None)
        holder["sessions"].append(session)
        return session

    monkeypatch.setattr(cc, "_session_factory", _factory)
    return holder


# ---------------------------------------------------------------------------
# Header / extension routing tests.
# ---------------------------------------------------------------------------


def test_session_id_header_routes_through_session_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _configure_runtime()
    holder = _install_recording_manager(monkeypatch, runtime)
    factory_holder = _install_default_factory_recorder(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [
                    {"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "hi"},
                ],
            },
            headers={"X-Silica-Session-ID": "alpha-123"},
        )

    assert response.status_code == 200
    # Manager called exactly once with the header-derived session_id.
    assert len(holder["calls"]) == 1
    assert holder["calls"][0]["session_id"] == "alpha-123"
    assert holder["calls"][0]["system_prompt"] == "be helpful"
    # Fresh-session factory was NOT consulted on the session_id branch.
    assert factory_holder["sessions"] == []


def test_extension_session_id_routes_through_session_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the header is absent, ``extension.session_id`` is the
    documented fallback for SDKs that cannot set custom headers."""
    runtime = _configure_runtime()
    holder = _install_recording_manager(monkeypatch, runtime)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
                "extension": {"session_id": "body-session-1"},
            },
        )

    assert response.status_code == 200
    assert len(holder["calls"]) == 1
    assert holder["calls"][0]["session_id"] == "body-session-1"


def test_header_takes_precedence_over_extension_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENING §4.3 names the header as the canonical selector. When
    both are set the header wins so a misconfigured client can be
    overridden at the proxy layer without touching the request body."""
    runtime = _configure_runtime()
    holder = _install_recording_manager(monkeypatch, runtime)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
                "extension": {"session_id": "from-body"},
            },
            headers={"X-Silica-Session-ID": "from-header"},
        )

    assert response.status_code == 200
    assert holder["calls"][0]["session_id"] == "from-header"


def test_empty_session_id_header_falls_back_to_extension_or_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty-string header value (a client clearing the field)
    must NOT be billed as a fresh persistent session. The body
    extension is the next selector in line; absent both, the route
    falls through to the fresh-session factory."""
    runtime = _configure_runtime()
    holder = _install_recording_manager(monkeypatch, runtime)
    factory_holder = _install_default_factory_recorder(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"X-Silica-Session-ID": ""},
        )

    assert response.status_code == 200
    # Empty header → no manager call, fresh factory consulted instead.
    assert holder["calls"] == []
    assert len(factory_holder["sessions"]) == 1


def test_no_selector_uses_fresh_session_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The (c)/(d) shape: no selector → fresh ChatSession per request.
    This is the default path for OpenAI clients that do not opt in to
    persistent sessions."""
    runtime = _configure_runtime()
    holder = _install_recording_manager(monkeypatch, runtime)
    factory_holder = _install_default_factory_recorder(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert holder["calls"] == []
    assert len(factory_holder["sessions"]) == 1


def test_extension_session_id_alone_does_not_trigger_501(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-unit (f) honours ``extension.session_id``; the (c) blanket
    501 on non-empty extension is relaxed for this field. Other
    fields (``thinking_mode`` / ``continue_truncated``) remain
    unimplemented and still 501."""
    runtime = _configure_runtime()
    _install_recording_manager(monkeypatch, runtime)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
                "extension": {"session_id": "sess-x"},
            },
        )

    assert response.status_code == 200


def test_extension_session_id_plus_thinking_mode_returns_501(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``thinking_mode`` is still unimplemented after (f); the route
    must reject any extension that names a non-honoured field even
    when ``session_id`` is also present."""
    _configure_runtime()

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
                "extension": {
                    "session_id": "sess-x",
                    "thinking_mode": "on",
                },
            },
        )

    assert response.status_code == 501
    assert "thinking_mode" in response.json()["error"]["message"]
    # session_id is honoured, so it must NOT appear in the rejection
    # list — guards against a regression where the route lumps it
    # back in with the unsupported set.
    assert "session_id" not in response.json()["error"]["message"]


def test_session_id_returns_501_for_sliding_window_adapter() -> None:
    """SLIDING-bearing adapters (Gemma4 31B today) cannot host
    persistent sessions in v0.1: SessionManager always builds a
    :class:`RadixPrefixCache`, but :class:`ContinuousBatcher`
    rejects that cache + SLIDING combination at construction time
    (silica/scheduler/batcher.py:292). The route must surface a
    501 with an actionable message instead of letting the request
    fall through to a 500 / broken SSE socket.
    """
    sliding_adapter = StubModelAdapter()
    # Override the stub's all-GLOBAL pattern with a SLIDING+GLOBAL
    # mix so capabilities() reports SLIDING in attention_kinds.
    # Mirrors the Gemma4 31B layout shape (alternating sliding /
    # full layers) that motivates the guard.
    sliding_adapter._pattern = AttentionPattern(
        per_layer=(AttentionKind.SLIDING, AttentionKind.GLOBAL)
    )

    runtime = Runtime(
        sliding_adapter,
        NullKVManager(),
        model_repo="stub/sliding-model",
        metrics=MetricsRegistry(),
    )
    _configure_runtime(runtime)

    with TestClient(openai_api.app) as client:
        # No session_id: works (fresh per-call ChatSession path).
        ok = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/sliding-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert ok.status_code == 200, ok.text

        # session_id via header: 501 with the SLIDING / sliding
        # explanation in the detail.
        rejected = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/sliding-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"X-Silica-Session-ID": "any-sid"},
        )
        assert rejected.status_code == 501
        detail = rejected.json()["error"]["message"]
        assert "SLIDING" in detail or "sliding" in detail
        assert "session_id" in detail

        # Same path via body extension: 501 too.
        rejected_body = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/sliding-model",
                "messages": [{"role": "user", "content": "hi"}],
                "extension": {"session_id": "via-body"},
            },
        )
        assert rejected_body.status_code == 501

    # SessionManager itself reports the capability via the public
    # property — admin endpoints / future CLI flags can consult it
    # without inspecting adapter internals.
    assert runtime.session_manager.supports_prefix_reuse is False
    # Sanity: the synthetic capability path equals what
    # capabilities_from_attention_pattern would return — guards
    # against an over-clever override slipping past the SessionManager
    # check.
    assert (
        AttentionKind.SLIDING
        in capabilities_from_attention_pattern(
            sliding_adapter._pattern
        ).attention_kinds
    )


def test_concurrent_same_session_id_serialises_through_engine_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The (f) concurrency contract: two requests sharing one
    ``session_id`` queue on ``runtime.engine_lock``; the second
    request's :meth:`SessionManager.get_or_create` and
    :meth:`ChatSession.replace_messages` must NOT fire while the
    first turn's ``ChatSession.chat`` is still running. Without
    the lock-held resolve, the second request's
    ``replace_messages`` would clobber the in-flight conversation
    history mid-decode — the OPENING-cited race that motivates
    sub-unit (f).

    Both requests below carry the **same** ``X-Silica-Session-ID``
    so :meth:`SessionManager.get_or_create` returns the same stub
    session instance both times. The first ``chat`` call (request
    1's turn) blocks on a :class:`threading.Event`; the second
    call (request 2's turn) flows through. We observe the
    recorded operation order while the first is parked: the
    second request's ``get_or_create`` must NOT have fired until
    the first's ``chat_end`` lands.
    """
    runtime = _configure_runtime()

    record: list[str] = []
    record_lock = threading.Lock()
    barrier = threading.Event()

    def _push(event: str) -> None:
        with record_lock:
            record.append(event)

    class _BlockingStubSession:
        """One session instance shared across both requests
        (because they share a ``session_id``). The first ``chat``
        invocation blocks on the barrier; subsequent invocations
        flow through. Each ``chat`` is tagged by its
        ``user_text`` argument so the test can distinguish the
        first turn's events from the second's in the recorded
        order.
        """

        def __init__(self) -> None:
            self.call_count = 0

        def chat(
            self,
            user_text: str,
            *,
            sampling_params: Any = None,
            stream_to: Any = None,
        ) -> TurnMetrics:
            _push(f"chat_start:{user_text}")
            self.call_count += 1
            if self.call_count == 1:
                # Hold engine_lock until the test releases the
                # barrier. The second request is queued waiting
                # for the lock during this window.
                assert barrier.wait(timeout=5.0), (
                    "barrier never released — test deadlocked"
                )
            _push(f"chat_end:{user_text}")
            return TurnMetrics(
                reply="ok",
                prompt_tokens=1,
                output_tokens=1,
                finish_reason="stop_token",
            )

    shared_session = _BlockingStubSession()

    def _get_or_create(
        session_id: str,
        *,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> _BlockingStubSession:
        # Tag with the request's system_prompt — each request
        # sends a unique marker as its system message so the
        # recorded ordering distinguishes turn 1's resolve from
        # turn 2's despite both sharing the session_id.
        marker = system_prompt if system_prompt else "(none)"
        _push(f"get_or_create:{marker}")
        return shared_session

    monkeypatch.setattr(
        runtime.session_manager,
        "get_or_create",
        _get_or_create,
    )

    sid = "shared-sid"

    # Single TestClient lifecycle for both requests so the
    # SessionManager / lifespan-built runtime is shared.
    with TestClient(openai_api.app) as client:

        def _post(marker: str) -> int:
            return client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub/model",
                    "messages": [
                        {"role": "system", "content": marker},
                        {"role": "user", "content": marker},
                    ],
                },
                headers={"X-Silica-Session-ID": sid},
            ).status_code

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        ) as ex:
            f1 = ex.submit(_post, "first-turn")
            # Spin until the first request has acquired the lock
            # and entered ``chat``. Bounded retry instead of a
            # fixed sleep so the test stays robust under loaded
            # CI.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with record_lock:
                    if "chat_start:first-turn" in record:
                        break
                time.sleep(0.01)
            with record_lock:
                assert "chat_start:first-turn" in record, (
                    "first request never entered chat"
                )
                assert "get_or_create:second-turn" not in record

            # Fire the second request while the first holds the
            # lock. Same session_id → same shared session → same
            # ``replace_messages`` target. If the route resolved
            # the session OUTSIDE the lock, the second's
            # get_or_create would fire here; we assert it does
            # NOT.
            f2 = ex.submit(_post, "second-turn")
            # Give the second request enough scheduler slices to
            # have begun resolving the session.
            time.sleep(0.2)
            with record_lock:
                assert "get_or_create:second-turn" not in record, (
                    "second request's get_or_create fired while "
                    "the first was still in chat — lock-held "
                    f"resolve broken. record={record!r}"
                )

            # Release the first request; both should complete.
            barrier.set()
            assert f1.result(timeout=5.0) == 200
            assert f2.result(timeout=5.0) == 200

    # Final ordering: first's full lifecycle (resolve → chat →
    # end) lands before the second begins. Both turns operated on
    # the same shared ChatSession, so a session-shared
    # replace_messages clobber is the failure this test detects.
    expected = [
        "get_or_create:first-turn",
        "chat_start:first-turn",
        "chat_end:first-turn",
        "get_or_create:second-turn",
        "chat_start:second-turn",
        "chat_end:second-turn",
    ]
    assert record == expected, f"got {record!r}"


def test_session_id_routes_through_session_manager_under_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke pin for the streaming branch.

    ``_resolve_session`` is unified across non-streaming and streaming,
    so the body-level routing assertions land via the non-streaming
    tests above. This test guards against a regression in the
    streaming-specific call site (the
    ``session_holder["session"] = _resolve_session(...)`` line that
    runs inside the SSE generator's ``async with engine_lock:``
    block) by driving an SSE request and asserting the manager saw
    the header-derived session_id.
    """
    runtime = _configure_runtime()
    holder = _install_recording_manager(monkeypatch, runtime)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"X-Silica-Session-ID": "stream-sid"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"data: [DONE]" in response.content
    assert len(holder["calls"]) == 1
    assert holder["calls"][0]["session_id"] == "stream-sid"


def test_same_session_id_reuses_chat_session_across_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load-bearing for cross-request prefix reuse: two requests
    naming the same session_id must hit the same persistent
    ChatSession instance."""
    runtime = _configure_runtime()
    holder = _install_recording_manager(monkeypatch, runtime)

    with TestClient(openai_api.app) as client:
        for _ in range(3):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub/model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"X-Silica-Session-ID": "shared-sid"},
            )

    # Manager got 3 get_or_create calls, all with the same id.
    assert [c["session_id"] for c in holder["calls"]] == [
        "shared-sid",
        "shared-sid",
        "shared-sid",
    ]
    # The cached session was reused — chat was invoked on the same
    # _RecordingStubSession three times.
    session = holder["sessions"]["shared-sid"]
    assert len(session.calls) == 3


# ---------------------------------------------------------------------------
# R-f acceptance: 3 turns shared prefix; turn 2 + turn 3 see prefix
# hits in the persistent prefix cache.
# ---------------------------------------------------------------------------


class _DeterministicTokenizer:
    """Char-based tokenizer with a Qwen-shaped chat template.

    Mirrors :class:`tests.test_chat_session._FakeTokenizer` —
    deterministic per-character ids so two prompts with a shared
    text prefix produce id sequences that share the same prefix.
    The template uses the manual ``<|im_start|>{role}\\n`` block
    list so we don't need a real Jinja template loader.
    """

    eos_token_ids: set[int] = {0}
    vocab_size: int = 1024

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % self.vocab_size) + 1 for c in text]

    def decode(self, token_ids: Any) -> str:
        """Inverse of :meth:`encode` for ASCII input.

        :class:`ChatSession._render_prompt` builds ``prompt_text`` by
        decoding the template's ``prompt_ids``, then hands the text
        to ``engine.generate_batch``. The real engine re-tokenises
        the text and the ids must match the originals — otherwise
        the prefix-cache peek runs against different ids than were
        inserted, missing every block.
        """
        out: list[str] = []
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int <= 0:
                continue
            out.append(chr((tid_int - 1) % self.vocab_size))
        return "".join(out)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> Any:
        parts: list[str] = []
        for m in messages:
            parts.append(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            )
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        return self.encode(text) if tokenize else text


class _DeterministicAdapter:
    """Adapter wrapping :class:`_DeterministicTokenizer`.

    The R-f test drives the real :class:`ChatSession` constructed by
    the SessionManager, so the adapter only needs to expose
    ``tokenizer()``. Other adapter surfaces are not consulted on
    this path.
    """

    def __init__(self) -> None:
        self._tokenizer = _DeterministicTokenizer()

    def tokenizer(self) -> _DeterministicTokenizer:
        return self._tokenizer


class _PrefixInsertingFakeEngine:
    """Fake batched engine that mimics ContinuousBatcher.reclaim_terminated.

    On each ``generate_batch`` call:

    1. Records the prompt text + its tokenizer-encoded ids.
    2. Records the prefix-cache peek hit count BEFORE inserting —
       this is the same value :class:`ChatSession` records as
       :attr:`TurnMetrics.prefix_hit_tokens` for the turn.
    3. Inserts the prompt's block-aligned ids into the prefix cache
       so subsequent turns sharing the prefix see hits. Uses
       :meth:`RadixPrefixCache.insert` rather than
       ``insert_detached`` since this fake does not produce K/V
       tensors — the radix tree shape + source refs are what the
       R-f acceptance is asserting.
    4. Yields a single done event so :class:`ChatSession.chat`
       returns a well-formed :class:`TurnMetrics` (reply text is
       empty because the tokenizer's ``decode`` returns ``""`` for
       any id sequence).
    """

    def __init__(self) -> None:
        self._tokenizer = _DeterministicTokenizer()
        self.metrics = _FakeMetrics()
        self.kv_manager = _FakeKVManager()
        self.peek_hit_tokens_at_entry: list[int] = []
        self.prompt_ids_seen: list[list[int]] = []

    def generate(
        self, prompt: str, params: SamplingParams | None = None
    ) -> Iterator[int]:
        # The prefix-cache path goes through generate_batch — ChatSession
        # only routes through ``generate`` when no prefix cache is
        # present, which is not the (f) shape. Keep the surface
        # implemented so the Engine Protocol is satisfied.
        return iter(())

    def generate_batch(
        self,
        prompts: Any,
        params: SamplingParams | list[SamplingParams] | None = None,
        *,
        max_batch_size: int | None = None,
        prefix_cache: Any = None,
        length_spread_threshold: float = 2.0,
    ) -> Iterator[BatchEvent]:
        del max_batch_size, length_spread_threshold, params
        prompt_text = prompts[0]
        prompt_ids = self._tokenizer.encode(prompt_text)
        self.prompt_ids_seen.append(list(prompt_ids))

        if prefix_cache is not None:
            hit = prefix_cache.peek(prompt_ids)
            self.peek_hit_tokens_at_entry.append(hit.num_hit_tokens)
            block_size = prefix_cache.block_size
            n_blocks = len(prompt_ids) // block_size
            if n_blocks > 0:
                store = prefix_cache.store
                allocated = [
                    store.allocate_id() for _ in range(n_blocks)
                ]
                prefix_cache.insert(prompt_ids, allocated)
        else:
            self.peek_hit_tokens_at_entry.append(0)

        yield BatchEvent.done(req_index=0, reason="done")


class _FakeSnapshot:
    ttft_ms: float | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    resident_mb: float | None = None
    logical_kv_bytes: int | None = None


class _FakeMetrics:
    def snapshot(self) -> _FakeSnapshot:
        return _FakeSnapshot()


class _FakeBudget:
    resident_bytes: int = 0
    logical_bytes: int = 0


class _FakeKVManager:
    def budget(self) -> _FakeBudget:
        return _FakeBudget()


def _install_prefix_inserting_engine(
    runtime: Runtime,
) -> _PrefixInsertingFakeEngine:
    """Replace the SessionManager's adapter + engine with the
    near-real R-f harness.

    Reaches into ``runtime.session_manager`` directly (private
    attributes) so future-built ChatSessions consult the
    deterministic tokenizer + prefix-inserting engine. This is the
    one supported way to drive R-f without a real model load.
    """
    fake_engine = _PrefixInsertingFakeEngine()
    runtime.session_manager._adapter = _DeterministicAdapter()  # type: ignore[attr-defined]
    runtime.session_manager._engine = fake_engine  # type: ignore[attr-defined]
    return fake_engine


def test_three_turn_shared_prefix_demo_logs_prefix_hits_after_turn_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-f acceptance (M-9 row #2): 3 requests sharing a system prompt
    in one ``X-Silica-Session-ID``; turn 2 + turn 3 must see
    ``prefix_hit_tokens > 0`` against the persistent prefix cache.

    The fake engine records the peek result at each ``generate_batch``
    entry — same value :class:`ChatSession` would surface on
    :attr:`TurnMetrics.prefix_hit_tokens`. We assert turn 1 sees no
    hits (cache is empty); turn 2 + turn 3 see hits because their
    prompts share the system + earlier-history prefix with what was
    inserted in prior turns.
    """
    runtime = _configure_runtime()
    fake_engine = _install_prefix_inserting_engine(runtime)
    sid = "rf-acceptance"

    base_messages = [
        {"role": "system", "content": "be terse and helpful"},
    ]

    with TestClient(openai_api.app) as client:
        # Turn 1: just the system prompt + first user.
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": base_messages
                + [{"role": "user", "content": "what is the capital?"}],
            },
            headers={"X-Silica-Session-ID": sid},
        )
        assert r1.status_code == 200, r1.text

        # Turn 2: full conversation with assistant1 from turn 1.
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": base_messages
                + [
                    {"role": "user", "content": "what is the capital?"},
                    {"role": "assistant", "content": "Paris."},
                    {"role": "user", "content": "and the population?"},
                ],
            },
            headers={"X-Silica-Session-ID": sid},
        )
        assert r2.status_code == 200, r2.text

        # Turn 3: extend further.
        r3 = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": base_messages
                + [
                    {"role": "user", "content": "what is the capital?"},
                    {"role": "assistant", "content": "Paris."},
                    {"role": "user", "content": "and the population?"},
                    {"role": "assistant", "content": "About 2.1M."},
                    {"role": "user", "content": "and the language?"},
                ],
            },
            headers={"X-Silica-Session-ID": sid},
        )
        assert r3.status_code == 200, r3.text

        # Snapshot the persistent session state BEFORE the TestClient
        # context manager exits — the lifespan shutdown will call
        # ``runtime.close``, which invokes
        # :meth:`SessionManager.close` and drops every entry. Reading
        # the cache after that point would always return ``None``.
        persisted = runtime.session_manager.get(sid)
        assert persisted is not None
        cache = persisted.prefix_cache
        # SessionManager always constructs a concrete
        # RadixPrefixCache; the ChatSession-level Protocol type is
        # ``_PrefixCacheLike | None`` (a narrow read surface) which
        # does not advertise ``node_count``. ``isinstance`` narrows
        # for mypy and pins the construction-site contract.
        assert isinstance(cache, RadixPrefixCache)
        # ``stats().num_blocks`` counts *detached* blocks; this fake
        # uses :meth:`RadixPrefixCache.insert` (no detached K/V
        # tensors needed for the prefix-reuse contract), so we read
        # the radix-tree node count via the public debug surface.
        node_count_during = cache.node_count()

    # Three engine invocations recorded.
    assert len(fake_engine.peek_hit_tokens_at_entry) == 3
    # Turn 1: cache empty, no peek hit.
    assert fake_engine.peek_hit_tokens_at_entry[0] == 0
    # Turn 2: cache has turn-1 blocks; turn 2's prompt shares the
    # system + first-user-turn prefix → peek lands hits.
    assert fake_engine.peek_hit_tokens_at_entry[1] > 0
    # Turn 3: more shared prefix, more hits (or at least equal to
    # turn 2's count — strict monotonicity isn't guaranteed by the
    # cache, but turn 2's hits are a lower bound).
    assert (
        fake_engine.peek_hit_tokens_at_entry[2]
        >= fake_engine.peek_hit_tokens_at_entry[1]
    )
    # Cache held block-aligned tree nodes from all three turns at
    # request time. The exact count is template- and prompt-length
    # dependent; assert the lower bound (turn 1 inserted at least
    # one node, turn 2/3 added their own divergent suffix).
    assert node_count_during > 0
