"""Tests for the streaming branch of POST /v1/chat/completions
(P-8 sub-unit (d)).

Covers the SSE wire format, the ``[DONE]`` sentinel, the
``stream_options.include_usage`` honouring, the threading-event
disconnect path, and a worker-error close. Mirrors the (c) test
file's stub-session pattern: a stub session whose ``chat`` method
walks the supplied ``stream_to`` callback over a deterministic list
of text deltas, then returns a fixed :class:`TurnMetrics`. No MLX
compute, no HuggingFace load.

Test surface:

- 200 SSE round trip: ``Content-Type: text/event-stream``; first
  chunk carries ``role='assistant'``; subsequent chunks each carry
  one ``delta.content`` string from the stub; terminal chunk has
  empty delta and ``finish_reason``; ``[DONE]`` sentinel is the
  last frame.
- ``stream_options.include_usage`` true: a usage-only chunk
  (``choices=[]``, populated ``usage``) appears between the finish
  chunk and ``[DONE]``.
- ``stream_options.include_usage`` unset / false: no usage chunk.
- ``stream_options`` envelope is permissive (``extra='allow'``):
  unknown fields like ``includ_usage`` parse silently and the
  request still streams a ``200`` SSE without honouring usage —
  rationale lives on :class:`StreamOptions` (OpenAI-owned surface,
  not silica's strict envelope).
- Disconnect path: when ``abort_event`` fires mid-stream, the
  callback raises :class:`_ClientDisconnected`, the worker exits,
  the queue receives the :data:`_DISCONNECTED` sentinel, and the
  generator closes without ``[DONE]``.
- :func:`_abortable_put` white-box: aborts immediately when the
  event is preset; aborts mid-wait once the event flips during a
  full-queue stall.
- Worker-error path: an unexpected exception inside ``session.chat``
  closes the SSE stream cleanly without ``[DONE]``.
- ``stream=True`` is no longer rejected by the route's 501 path —
  it dispatches into the streaming branch (regression pin against
  re-introducing the (c)-era 501).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="P-8 [serve] extra not installed")
pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")

from fastapi.testclient import TestClient  # noqa: E402

from silica.chat.session import TurnMetrics  # noqa: E402
from silica.core.profiler import MetricsRegistry  # noqa: E402
from silica.kvcache.manager import NullKVManager  # noqa: E402
from silica.models.adapter import StubModelAdapter  # noqa: E402
from silica.server import openai_api  # noqa: E402
from silica.server.routes import chat_completions as cc  # noqa: E402
from silica.server.runtime import Runtime  # noqa: E402

# ---------------------------------------------------------------------------
# Stub session that walks a deterministic delta list through stream_to.
# ---------------------------------------------------------------------------


class _StreamingStubSession:
    """Synchronously calls ``stream_to`` once per delta then returns
    a fixed :class:`TurnMetrics`. Optional hooks exercise the
    abort / error paths."""

    def __init__(
        self,
        *,
        deltas: list[str],
        prompt_tokens: int = 5,
        output_tokens: int | None = None,
        finish_reason: str = "stop_token",
        on_each_delta: Callable[[int, str], None] | None = None,
        raise_on_delta: int | None = None,
    ) -> None:
        self.deltas = deltas
        self.prompt_tokens = prompt_tokens
        self.output_tokens = (
            output_tokens if output_tokens is not None else len(deltas)
        )
        self.finish_reason = finish_reason
        self.on_each_delta = on_each_delta
        self.raise_on_delta = raise_on_delta
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        user_text: str,
        *,
        sampling_params: Any = None,
        stream_to: Callable[[str], None] | None = None,
    ) -> TurnMetrics:
        self.calls.append(
            {"user_text": user_text, "sampling_params": sampling_params}
        )
        for i, delta in enumerate(self.deltas):
            if self.raise_on_delta is not None and i == self.raise_on_delta:
                raise RuntimeError("worker boom (test)")
            if self.on_each_delta is not None:
                self.on_each_delta(i, delta)
            if stream_to is not None:
                stream_to(delta)  # may raise _ClientDisconnected
        # Total output_tokens reflects what the model would have
        # emitted; the route uses this to populate Usage.
        reply = "".join(self.deltas)
        return TurnMetrics(
            reply=reply,
            prompt_tokens=self.prompt_tokens,
            output_tokens=self.output_tokens,
            finish_reason=self.finish_reason,
        )


def _build_stub_runtime() -> Runtime:
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


def _configure_runtime() -> Runtime:
    runtime = _build_stub_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    return runtime


def _install_streaming_stub(
    monkeypatch: pytest.MonkeyPatch,
    deltas: list[str],
    **stub_kwargs: Any,
) -> dict[str, Any]:
    holder: dict[str, Any] = {"session": None}

    def _factory(
        runtime: Runtime,
        *,
        system_prompt: str | None,
        history: list[dict[str, str]],
    ) -> _StreamingStubSession:
        session = _StreamingStubSession(deltas=deltas, **stub_kwargs)
        holder["session"] = session
        return session

    monkeypatch.setattr(cc, "_session_factory", _factory)
    return holder


def _parse_sse_frames(raw: bytes) -> list[Any]:
    """Split a raw SSE response body into frames and JSON-decode each
    ``data:`` payload. The terminal ``[DONE]`` sentinel is yielded as
    the literal string ``"[DONE]"``."""
    frames: list[Any] = []
    for raw_frame in raw.split(b"\n\n"):
        frame = raw_frame.strip()
        if not frame:
            continue
        assert frame.startswith(b"data: "), frame
        payload = frame[len(b"data: "):]
        if payload == b"[DONE]":
            frames.append("[DONE]")
        else:
            frames.append(json.loads(payload))
    return frames


# ---------------------------------------------------------------------------
# 200 SSE round trip + wire-format pins.
# ---------------------------------------------------------------------------


def test_streaming_sse_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_streaming_stub(
        monkeypatch,
        deltas=["Hello", " ", "world", "!"],
        prompt_tokens=7,
        output_tokens=4,
        finish_reason="stop_token",
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers.get("cache-control") == "no-cache"

    frames = _parse_sse_frames(response.content)
    # Initial role chunk + 4 delta chunks + finish chunk + [DONE]
    assert len(frames) == 7
    assert frames[-1] == "[DONE]"

    role = frames[0]
    assert role["object"] == "chat.completion.chunk"
    assert role["model"] == "stub/model"
    assert role["choices"][0]["delta"] == {"role": "assistant"}
    # ``exclude_none=True`` on the chunk serializer drops nullable
    # fields from intermediate frames; ``.get`` accepts either
    # explicit-null or omitted, both of which are valid OpenAI
    # SSE wire forms in practice.
    assert role["choices"][0].get("finish_reason") is None

    # All chunks share the same id and created stamp.
    assert {f["id"] for f in frames[:-1]} == {role["id"]}
    assert {f["created"] for f in frames[:-1]} == {role["created"]}

    # Token chunks: empty role / finish_reason, content carries the
    # delta string verbatim.
    deltas_seen = [f["choices"][0]["delta"]["content"] for f in frames[1:5]]
    assert deltas_seen == ["Hello", " ", "world", "!"]
    for f in frames[1:5]:
        assert f["choices"][0].get("finish_reason") is None
        assert "role" not in f["choices"][0]["delta"]
        assert "usage" not in f  # token chunks do NOT carry usage

    # Finish chunk: empty delta, finish_reason='stop'.
    finish = frames[5]
    assert finish["choices"][0]["delta"] == {}
    assert finish["choices"][0]["finish_reason"] == "stop"
    # No usage on the finish chunk when include_usage was not set.
    assert "usage" not in finish


# ---------------------------------------------------------------------------
# stream_options.include_usage honour.
# ---------------------------------------------------------------------------


def test_streaming_include_usage_emits_usage_chunk_before_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_streaming_stub(
        monkeypatch,
        deltas=["a", "b"],
        prompt_tokens=11,
        output_tokens=2,
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    # role + 2 delta + finish + usage + [DONE]
    assert len(frames) == 6
    assert frames[-1] == "[DONE]"

    usage_chunk = frames[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "total_tokens": 13,
    }
    # Usage chunk reuses the same id / created / model as token chunks.
    assert usage_chunk["id"] == frames[0]["id"]
    assert usage_chunk["created"] == frames[0]["created"]
    assert usage_chunk["model"] == frames[0]["model"]


def test_streaming_include_usage_false_suppresses_usage_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_streaming_stub(monkeypatch, deltas=["a"])

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
                "stream_options": {"include_usage": False},
            },
        )

    frames = _parse_sse_frames(response.content)
    # role + delta + finish + [DONE] — no usage chunk.
    assert len(frames) == 4
    assert frames[-1] == "[DONE]"
    for frame in frames[:-1]:
        assert "usage" not in frame


def test_stream_options_unknown_fields_pass_through_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stream_options`` is OpenAI-owned, parsed under
    ``extra='allow'`` (silica/server/schemas.py StreamOptions docstring).
    A typo such as ``includ_usage`` must NOT 422 — future SDK versions
    may add fields, and rejecting them would break clients on the
    next OpenAI API revision. Unknown fields parse silently; the route
    only honours :attr:`StreamOptions.include_usage`. Since the typo
    leaves ``include_usage`` at its default ``None``, no usage chunk
    appears between the finish chunk and ``[DONE]``."""
    _configure_runtime()
    _install_streaming_stub(monkeypatch, deltas=["a"])

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
                "stream_options": {"includ_usage": True},
            },
        )

    assert response.status_code == 200, response.text
    frames = _parse_sse_frames(response.content)
    # role + 1 delta + finish + [DONE] — the typo did NOT activate
    # include_usage, so no usage chunk appears.
    assert len(frames) == 4
    assert frames[-1] == "[DONE]"
    for frame in frames[:-1]:
        assert "usage" not in frame


# ---------------------------------------------------------------------------
# Disconnect path: callback aborts at the next delta boundary.
# ---------------------------------------------------------------------------


def test_streaming_disconnected_branch_emits_no_done_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit-test the worker's ``_ClientDisconnected`` handling.

    The full FastAPI client-disconnect flow is awkward to drive from
    a sync :class:`TestClient` (the :class:`StreamingResponse`
    generator is fully drained before the response object returns to
    the caller). Instead this test exercises the worker's
    ``_ClientDisconnected`` handling at the unit level: a stub
    session that walks two deltas then raises
    :class:`_ClientDisconnected` mimics what the route's
    :func:`_on_delta` callback would do when the generator's cleanup
    sets ``abort_event``. The route's worker must catch the
    exception, push :data:`_DISCONNECTED` onto the queue, and the
    generator must close the SSE stream *without* emitting
    ``[DONE]`` (a dead consumer cannot make use of it).

    The straight-line callback wiring (
    ``if abort_event.is_set(): raise _ClientDisconnected()``) is
    simple enough that the worker-side handling is the interesting
    bit; this test pins it.
    """
    _configure_runtime()

    deltas_seen: list[str] = []

    from silica.server.routes.chat_completions import (
        _ClientDisconnected,
    )

    class _AbortingStubSession:
        def chat(
            self,
            user_text: str,
            *,
            sampling_params: Any = None,
            stream_to: Callable[[str], None] | None = None,
        ) -> TurnMetrics:
            deltas = ["one", "two", "three", "four"]
            for i, d in enumerate(deltas):
                if i == 2:
                    raise _ClientDisconnected()
                if stream_to is not None:
                    stream_to(d)
                deltas_seen.append(d)
            return TurnMetrics(
                reply="".join(deltas),
                prompt_tokens=4,
                output_tokens=len(deltas),
                finish_reason="stop_token",
            )

    monkeypatch.setattr(
        cc,
        "_session_factory",
        lambda runtime, *, system_prompt, history: _AbortingStubSession(),
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    # role + two deltas + (NO finish chunk, NO [DONE]) — the
    # disconnected branch closes the stream cleanly.
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    contents = [
        f["choices"][0]["delta"]["content"] for f in frames[1:]
    ]
    assert contents == ["one", "two"]
    assert "[DONE]" not in frames

    # Sanity: the worker walked those two deltas before aborting.
    assert deltas_seen == ["one", "two"]


def test_on_delta_raises_client_disconnected_when_event_is_set() -> None:
    """Direct white-box pin on the abort wiring contract.

    Re-implements the route's :func:`_on_delta` semantics in
    isolation: a :class:`threading.Event` flipped on, then a callback
    invocation must raise :class:`_ClientDisconnected` *before*
    attempting the thread-to-loop transfer. This pins the
    contract that the route's per-request callback obeys.
    """
    from silica.server.routes.chat_completions import (
        _ClientDisconnected,
    )

    abort_event = threading.Event()

    def _on_delta(delta_text: str) -> None:
        if abort_event.is_set():
            raise _ClientDisconnected()
        # In production this would call run_coroutine_threadsafe;
        # the test only exercises the abort branch.
        raise AssertionError(
            "abort branch should have fired before this point"
        )

    abort_event.set()
    with pytest.raises(_ClientDisconnected):
        _on_delta("any")


# ---------------------------------------------------------------------------
# _abortable_put white-box: aborts immediately when event preset; aborts
# mid-wait once the event flips during a full-queue stall.
# ---------------------------------------------------------------------------


def test_abortable_put_raises_immediately_when_event_preset() -> None:
    """If ``abort_event`` is already set on entry, the helper must
    raise :class:`_ClientDisconnected` *before* scheduling the
    coroutine onto the event loop. This guarantees the (a1)
    lock+thread contract holds even when the consumer has already
    disconnected by the time the worker invokes the callback."""
    import asyncio

    from silica.server.routes.chat_completions import (
        _abortable_put,
        _ClientDisconnected,
    )

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        abort_event = threading.Event()
        abort_event.set()

        # Run the helper from a worker thread (mimics the production
        # call site: ``ChatSession.chat`` invokes ``stream_to`` from
        # an :func:`asyncio.to_thread` thread).
        def _call() -> Exception | None:
            try:
                _abortable_put(queue, "x", loop, abort_event)
            except _ClientDisconnected as exc:
                return exc
            return None

        result = await asyncio.to_thread(_call)
        assert isinstance(result, _ClientDisconnected)
        # Queue stays empty — the helper bailed before scheduling
        # ``queue.put``.
        assert queue.empty()

    asyncio.run(_run())


def test_abortable_put_succeeds_when_consumer_drains_before_abort() -> None:
    """No-drop pin for the success path.

    When the queue is full and the consumer drains before any abort
    signal, :func:`_abortable_put` must complete the put rather than
    drop on timeout or block forever. This pins the contract that
    the worker's terminal metrics push relies on: a slow consumer
    (full queue) must still receive ``TurnMetrics`` once it catches
    up, otherwise the generator hangs on its next ``queue.get()``
    waiting for a finish/[DONE] sequence that will never arrive.
    """
    import asyncio

    from silica.server.routes.chat_completions import _abortable_put

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        await queue.put("filler")

        abort_event = threading.Event()
        result_box: dict[str, Any] = {}

        def _call() -> None:
            try:
                _abortable_put(
                    queue,
                    "metrics",
                    loop,
                    abort_event,
                    poll_interval=0.05,
                )
            except Exception as exc:  # noqa: BLE001
                result_box["exc"] = exc
            else:
                result_box["exc"] = None

        worker_task = asyncio.create_task(asyncio.to_thread(_call))

        # Let the worker enter the polling loop while the queue is
        # full.
        await asyncio.sleep(0.15)
        assert not worker_task.done()

        # Consumer drains the filler. Worker's pending put should
        # then succeed in roughly one ``poll_interval`` tick.
        assert await queue.get() == "filler"
        await asyncio.wait_for(worker_task, timeout=1.0)

        assert result_box["exc"] is None
        # The metrics item landed in the queue.
        assert await queue.get() == "metrics"

    asyncio.run(_run())


def test_abortable_put_raises_mid_wait_when_event_flips_during_full_queue(
) -> None:
    """When the queue is full and the event flips while
    :func:`_abortable_put` is polling its in-flight future, the helper
    must raise :class:`_ClientDisconnected` within roughly one
    ``poll_interval`` rather than block forever. This is the G-1
    blocker that the helper exists to fix.
    """
    import asyncio

    from silica.server.routes.chat_completions import (
        _abortable_put,
        _ClientDisconnected,
    )

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        # Pre-fill the queue so the next put blocks indefinitely under
        # the plain ``run_coroutine_threadsafe`` pattern.
        await queue.put("filler")

        abort_event = threading.Event()
        result_box: dict[str, Any] = {}

        def _call() -> None:
            try:
                _abortable_put(
                    queue,
                    "x",
                    loop,
                    abort_event,
                    poll_interval=0.05,
                )
            except _ClientDisconnected as exc:
                result_box["exc"] = exc
            except Exception as exc:  # noqa: BLE001
                result_box["exc"] = exc
            else:
                result_box["exc"] = None

        worker_task = asyncio.create_task(asyncio.to_thread(_call))

        # Give the worker a couple of poll intervals to enter the
        # blocking ``future.result(timeout=...)`` loop, then abort.
        await asyncio.sleep(0.15)
        abort_event.set()

        # Worker must unwind well within the poll interval after the
        # event flips. 1.0s is generous; without the helper it would
        # hang until queue room appears (forever in this test).
        await asyncio.wait_for(worker_task, timeout=1.0)

        assert isinstance(result_box["exc"], _ClientDisconnected)
        # Queue still holds the filler (no successful put).
        assert queue.qsize() == 1
        assert await queue.get() == "filler"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Deep stream: backpressure round trip, no metrics drop on terminal push.
# ---------------------------------------------------------------------------


def test_streaming_deep_stream_completes_without_metrics_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin: a stream longer than :data:`_STREAM_QUEUE_MAXSIZE` (64)
    must still deliver every delta + the terminal finish/[DONE]
    sequence. This is the regression pin against the earlier
    ``_push_terminal`` design that timed out and dropped the
    metrics item, leaving the generator hung on its next
    ``queue.get()``.
    """
    deltas = [f"d{i}" for i in range(100)]
    _configure_runtime()
    _install_streaming_stub(monkeypatch, deltas=deltas, prompt_tokens=3)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    # role + 100 deltas + finish + [DONE] = 103 frames
    assert len(frames) == 103
    assert frames[-1] == "[DONE]"
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"

    contents = [
        f["choices"][0]["delta"]["content"] for f in frames[1:-2]
    ]
    assert contents == deltas

    finish = frames[-2]
    assert finish["choices"][0]["delta"] == {}
    assert finish["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Worker error: SSE stream closes cleanly, no [DONE].
# ---------------------------------------------------------------------------


def test_streaming_worker_error_closes_stream_without_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_streaming_stub(
        monkeypatch,
        deltas=["a", "b", "c"],
        raise_on_delta=2,  # raise inside chat() before delta index 2
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    # role + two deltas, then the worker raised; no finish, no usage,
    # no [DONE].
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    contents = [
        f["choices"][0]["delta"]["content"] for f in frames[1:]
    ]
    assert contents == ["a", "b"]
    assert "[DONE]" not in frames


# ---------------------------------------------------------------------------
# stream=True is dispatched (regression pin: (c) used to 501 it).
# ---------------------------------------------------------------------------


def test_stream_true_no_longer_returns_501(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sub-unit (c) returned 501 on stream=True; (d) lifts that gate.
    A regression that re-introduced the 501 would surface here."""
    _configure_runtime()
    _install_streaming_stub(monkeypatch, deltas=["x"])

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")


# ---------------------------------------------------------------------------
# Finish-reason map propagates into the streaming finish chunk.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "silica_reason, expected_oai",
    [
        ("stop_token", "stop"),
        ("max_tokens", "length"),
        ("empty", "stop"),
    ],
)
def test_streaming_finish_reason_map(
    monkeypatch: pytest.MonkeyPatch,
    silica_reason: str,
    expected_oai: str,
) -> None:
    _configure_runtime()
    _install_streaming_stub(
        monkeypatch, deltas=["x"], finish_reason=silica_reason
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
            },
        )

    frames = _parse_sse_frames(response.content)
    finish_chunk = frames[-2]  # last is [DONE]
    assert finish_chunk["choices"][0]["finish_reason"] == expected_oai


