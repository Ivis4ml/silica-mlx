"""silica.server.routes.chat_completions — POST /v1/chat/completions
non-streaming + streaming branches (P-8 sub-units (c) and (d)).

Non-streaming (``stream=False``) is driven by
:meth:`silica.chat.session.ChatSession.chat` with ``stream_to=None``;
streaming (``stream=True``) drives the same call with a
:func:`_on_delta` callback that bridges MLX-thread token deltas onto
an asyncio queue, which the SSE :class:`StreamingResponse` generator
drains and re-frames as OpenAI ``chat.completion.chunk`` events.

One conversation per HTTP request; v0.1 has no cross-request session
reuse (sub-unit (f) lands ``SessionManager``).

Pipeline
========

1. **Model-id match** — return 404 if ``body.model`` differs from
   ``runtime.model_repo`` (the single-process server serves exactly
   one model; echoing a wrong id would mislead OpenAI clients).
2. :func:`_validate_unsupported` — return 501 for fields that
   change output semantics or are not yet wired
   (string-sequence ``stop``, non-empty ``extension`` envelope,
   ``n>1``, ``tools``, ``tool_choice``, ``logprobs``,
   ``top_logprobs``, ``logit_bias``, ``presence_penalty != 0``,
   ``frequency_penalty != 0``, ``response_format.type != 'text'``,
   multimodal content, non-{system, user, assistant} roles).
   ``stream=True`` was rejected here in (c) but is honoured by the
   streaming branch added in (d).
3. :func:`_validate_sampling_bounds` — return 400 for sampling
   parameters outside their accepted range (temperature, top_p,
   max_tokens, max_completion_tokens). Without this the pydantic
   ``Field(ge / gt)`` constraints on
   :class:`silica.core.sampling.SamplingParams` would raise inside
   the handler and surface as a 500.
4. :func:`_resolve_max_tokens` — pick precedence between
   ``max_tokens`` and ``max_completion_tokens``; 400 if both differ.
5. :func:`_extract_messages` — split into system_prompt / history /
   last-user-text; 400 on empty messages, non-user-last, or a
   non-leading system message.
6. :func:`_build_sampling_params` — assemble
   :class:`silica.core.sampling.SamplingParams`. The ``stop`` field
   is rejected upstream so this always passes ``stop=()``.
7. :func:`_session_factory` — module-level factory hook (tests
   monkeypatch this to return a stub :class:`ChatSession` whose
   ``chat`` method returns a deterministic :class:`TurnMetrics`).
8. ``async with runtime.engine_lock: await asyncio.to_thread(
   session.chat, ...)`` — runs MLX compute off the event loop per
   the Runtime docstring's lock+thread contract.
9. :func:`_build_response` — assembles
   :class:`ChatCompletionResponse` with strict
   :class:`AssistantMessage` and :class:`Usage` from
   :class:`TurnMetrics`.

Finish-reason mapping (silica → OpenAI): ``stop_token`` → ``stop``,
``max_tokens`` → ``length``, ``empty`` → ``stop`` (model produced
nothing; treat as a stop).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from silica.chat.session import ChatSession, TurnMetrics
from silica.core.logger import get_logger
from silica.core.sampling import SamplingParams
from silica.server.runtime import Runtime
from silica.server.schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)

log = get_logger(__name__)

router = APIRouter()


# Streaming-specific bridge constants. The bounded queue depth provides
# G-3 backpressure (OPENING §6.1.2): when a slow consumer fails to
# drain, ``run_coroutine_threadsafe(queue.put(item), loop).result()``
# blocks the worker thread on a full queue, which is what we want —
# slow client → slow MLX decode → no token drop. Hard-coded in v0.1
# per the (d) review; tunable via env / CLI flag is deferred to (h).
_STREAM_QUEUE_MAXSIZE = 64


# Sentinel placed on the queue when the worker noticed a client
# disconnect (via :class:`_ClientDisconnected`). The generator drains
# this and exits cleanly without emitting ``[DONE]`` (the client is
# already gone; SSE framing on a disconnected socket is moot).
_DISCONNECTED = object()


class _ClientDisconnected(Exception):
    """Raised inside the worker-thread :func:`_on_delta` callback when
    the SSE generator's cleanup path has set the abort event after a
    client disconnect. Caught by the worker loop so ``ChatSession.chat``
    unwinds cleanly at the next streamed-delta boundary instead of
    running to completion against a dead consumer.

    Caveat per OPENING G-3: an in-flight MLX decode step can NOT be
    interrupted; abort takes effect at the next ``stream_to`` invocation.
    Acceptable v0.1 trade-off (the alternative requires a per-step
    cancellation hook on ``Engine.generate``, deferred to (h) hardening).
    """


_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})

# Map silica's TurnMetrics.finish_reason to OpenAI's choices[].finish_reason.
_FINISH_REASON_MAP: dict[str, str] = {
    "stop_token": "stop",
    "max_tokens": "length",
    "empty": "stop",
}


# ---------------------------------------------------------------------------
# Module-level seams (monkeypatched in tests).
# ---------------------------------------------------------------------------


SessionFactory = Callable[..., Any]
"""Signature: ``(runtime, *, system_prompt, history) -> ChatSession-like``.
Tests inject a stub returning an object with a synchronous ``chat``
method that yields a deterministic :class:`TurnMetrics`.
"""


def _default_session_factory(
    runtime: Runtime,
    *,
    system_prompt: str | None,
    history: list[dict[str, str]],
) -> ChatSession:
    """Production session factory.

    Builds a fresh :class:`ChatSession` over ``runtime.adapter`` /
    ``runtime.engine`` and seeds the message history. v0.1 has no
    session reuse — every request gets a fresh ChatSession; (f)
    replaces this with a :class:`SessionManager` lookup.
    """
    session = ChatSession(
        adapter=runtime.adapter,
        # ChatSession's ``_EngineLike`` Protocol declares ``kv_manager``
        # as a mutable variable, but ``Engine.kv_manager`` is a
        # read-only property — Pydantic / mypy flags the variance even
        # though the runtime contract is satisfied. The chat-CLI's app
        # layer uses the same ignore (silica/chat/cli/app.py:475).
        engine=runtime.engine,  # type: ignore[arg-type]
        system_prompt=system_prompt,
    )
    if history:
        # ChatSession.__init__ already prepended the system_prompt to
        # ``_messages``; replace_messages overwrites the whole list, so
        # we rebuild the full conversation including the system slot.
        full: list[dict[str, str]] = []
        if system_prompt:
            full.append({"role": "system", "content": system_prompt})
        full.extend(history)
        session.replace_messages(full)
    return session


_session_factory: SessionFactory = _default_session_factory


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _get_runtime(request: Request) -> Runtime:
    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.closed:
        raise HTTPException(status_code=503, detail="engine not ready")
    return runtime


def _validate_unsupported(req: ChatCompletionRequest) -> None:
    """Return 400 or 501 for fields that change output semantics or
    are not wired in v0.1. Schema layer parses everything under
    ``extra='allow'``; this is the route-level rejection per (b)
    review's "parse != supported" rule.
    """
    # silica.engine.Engine v0.1 only honours stop_token_ids (set from
    # the tokenizer's EOS set); string-sequence ``stop`` is a P-2
    # concern per silica/engine/__init__.py:17. Accepting the field
    # silently would generate past the requested terminator, which is
    # worse than a clear 501.
    if req.stop is not None:
        raise HTTPException(
            status_code=501,
            detail=(
                "string-sequence 'stop' not supported in v0.1; "
                "generation terminates only on tokenizer EOS"
            ),
        )

    # silica's Extension envelope is parsed strictly by (b), but the
    # honouring sites land later: session_id in (f), thinking_mode and
    # continue_truncated also in (f). Accepting non-empty extension in
    # (c) would silently fall back to a fresh ChatSession with default
    # thinking, which contradicts what the field claims to do.
    if req.extension is not None:
        ext_fields = req.extension.model_dump(exclude_none=True)
        if ext_fields:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"extension fields {sorted(ext_fields.keys())} are "
                    "not honoured in v0.1 (session_id, thinking_mode, "
                    "continue_truncated land in P-8 sub-unit (f))"
                ),
            )

    if req.response_format is not None and req.response_format.type != "text":
        raise HTTPException(
            status_code=501,
            detail=(
                f"response_format.type={req.response_format.type!r} not "
                "supported in v0.1; only 'text' is honoured"
            ),
        )

    extra: dict[str, Any] = req.model_extra or {}
    if extra.get("n") not in (None, 1):
        raise HTTPException(
            status_code=501,
            detail=(
                f"n={extra['n']} not supported in v0.1; only one "
                "completion per request is produced"
            ),
        )
    if extra.get("tools") is not None:
        raise HTTPException(
            status_code=501, detail="tool calling not supported in v0.1"
        )
    if extra.get("tool_choice") is not None:
        raise HTTPException(
            status_code=501, detail="tool_choice not supported in v0.1"
        )
    if extra.get("logprobs"):
        raise HTTPException(
            status_code=501, detail="logprobs not supported in v0.1"
        )
    if extra.get("top_logprobs") is not None:
        raise HTTPException(
            status_code=501, detail="top_logprobs not supported in v0.1"
        )
    if extra.get("logit_bias") is not None:
        raise HTTPException(
            status_code=501, detail="logit_bias not supported in v0.1"
        )
    if extra.get("presence_penalty") not in (None, 0, 0.0):
        raise HTTPException(
            status_code=501,
            detail="presence_penalty not supported in v0.1",
        )
    if extra.get("frequency_penalty") not in (None, 0, 0.0):
        raise HTTPException(
            status_code=501,
            detail="frequency_penalty not supported in v0.1",
        )


def _validate_sampling_bounds(req: ChatCompletionRequest) -> None:
    """Return 400 for sampling parameters outside their accepted range.

    Schema (b) keeps requests as ``extra='allow'`` pure-shape with no
    bounds, so an out-of-range temperature / top_p / max_tokens would
    otherwise reach :class:`silica.core.sampling.SamplingParams`
    (which has ``Field(ge / gt / le)`` constraints) and surface as a
    Pydantic 500 from the route. That UX is worse than an explicit
    400; we mirror OpenAI's bounds checking here. Bound choices:

    - ``temperature`` ∈ [0, 2] (OpenAI's documented range).
    - ``top_p`` ∈ (0, 1] (OpenAI's documented range; 0 is degenerate).
    - ``max_tokens`` / ``max_completion_tokens`` strictly positive.
    """
    if req.temperature is not None and not (0.0 <= req.temperature <= 2.0):
        raise HTTPException(
            status_code=400,
            detail=(
                f"temperature={req.temperature} out of range [0, 2]"
            ),
        )
    if req.top_p is not None and not (0.0 < req.top_p <= 1.0):
        raise HTTPException(
            status_code=400,
            detail=f"top_p={req.top_p} out of range (0, 1]",
        )
    if req.max_tokens is not None and req.max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens={req.max_tokens} must be a positive integer",
        )
    if (
        req.max_completion_tokens is not None
        and req.max_completion_tokens <= 0
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_completion_tokens={req.max_completion_tokens} "
                "must be a positive integer"
            ),
        )


def _resolve_max_tokens(req: ChatCompletionRequest) -> int | None:
    """Pick the cap. Returns ``None`` to defer to the silica default
    when neither field is set.

    OpenAI deprecated ``max_tokens`` for chat completions in favour of
    ``max_completion_tokens``; we accept either. If both are set with
    different values, return 400 — the schema parsed both per (b)
    rule "no precedence in schema, route decides", and disagreeing
    values are a client bug.
    """
    if req.max_tokens is not None and req.max_completion_tokens is not None:
        if req.max_tokens != req.max_completion_tokens:
            raise HTTPException(
                status_code=400,
                detail=(
                    "max_tokens and max_completion_tokens are both set "
                    "with different values; pick one"
                ),
            )
        return req.max_completion_tokens
    # Explicit None checks — falsy ``or`` would mishandle 0 (which is
    # already blocked by ``_validate_sampling_bounds``, but the
    # explicit form keeps the intent obvious and survives if the
    # bounds check is ever relaxed).
    if req.max_completion_tokens is not None:
        return req.max_completion_tokens
    return req.max_tokens


def _extract_messages(
    req: ChatCompletionRequest,
) -> tuple[str | None, list[dict[str, str]], str]:
    """Split the OpenAI ``messages`` list into:

    - ``system_prompt``: concatenation of all leading system messages
      with ``\\n\\n`` separators, or ``None`` if no system messages.
    - ``history``: user / assistant turns *before* the final user
      message, in original order.
    - ``user_text``: the content of the final message, which must
      have role ``user``.

    Returns 400 on empty messages, on a non-user final message, or on
    an unknown role; returns 501 on non-text content (multimodal
    parts parse at schema level but routes do not consume them).
    """
    if not req.messages:
        raise HTTPException(
            status_code=400, detail="messages must not be empty"
        )

    last = req.messages[-1]
    if last.role != "user":
        raise HTTPException(
            status_code=400,
            detail=(
                f"the last message must have role='user', got "
                f"role={last.role!r}"
            ),
        )
    if not isinstance(last.content, str):
        raise HTTPException(
            status_code=501,
            detail=(
                "non-text content on the final user message "
                "(multimodal) not supported in v0.1"
            ),
        )
    user_text = last.content

    system_parts: list[str] = []
    history: list[dict[str, str]] = []
    seen_non_system = False

    for msg in req.messages[:-1]:
        if msg.role not in _ALLOWED_ROLES:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"role={msg.role!r} not supported in v0.1 "
                    "(only system / user / assistant)"
                ),
            )
        if not isinstance(msg.content, str):
            raise HTTPException(
                status_code=501,
                detail=(
                    f"non-text content on a {msg.role!r} message "
                    "(multimodal) not supported in v0.1"
                ),
            )
        if msg.role == "system":
            # Only leading system messages are accepted — once we have
            # seen any user / assistant turn, a later system message
            # would silently be hoisted to the front by the chat
            # template, reordering the conversation. That is a
            # semantic change, not a wording fix; reject it.
            if seen_non_system:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "system messages must precede all user / "
                        "assistant messages; out-of-order system "
                        "messages would be silently hoisted by the "
                        "chat template"
                    ),
                )
            system_parts.append(msg.content)
        else:
            seen_non_system = True
            history.append({"role": msg.role, "content": msg.content})

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, history, user_text


def _build_sampling_params(
    req: ChatCompletionRequest,
    *,
    max_tokens: int | None,
    eos_ids: tuple[int, ...],
) -> SamplingParams:
    """Map OpenAI request fields onto silica's sampling-params model.

    - ``temperature``: pass-through; defaults to silica's 1.0 if unset.
    - ``top_p``: pass-through.
    - ``top_k``: not in OpenAI Chat Completions; left at silica's
      default (None / disabled).
    - ``max_tokens``: from :func:`_resolve_max_tokens`; defaults to
      silica's 256 if neither OpenAI cap was set.
    - ``stop``: not honoured in v0.1 (rejected upstream by
      :func:`_validate_unsupported`); always passed as the empty
      tuple here.
    - ``stop_token_ids``: tokenizer EOS set so generation halts on
      the model's natural EOS token.
    - ``seed``: pass-through.
    """
    return SamplingParams(
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p,
        top_k=None,
        max_tokens=max_tokens if max_tokens is not None else 256,
        stop=(),
        stop_token_ids=eos_ids,
        seed=req.seed,
    )


def _build_response(
    metrics: TurnMetrics, *, model: str
) -> ChatCompletionResponse:
    """Assemble the strict :class:`ChatCompletionResponse` from a
    :class:`TurnMetrics`. The response uses :class:`AssistantMessage`
    so a future bug emitting role='user' or extra fields fails loud
    via Pydantic validation rather than landing on the wire.
    """
    finish = _FINISH_REASON_MAP.get(metrics.finish_reason, "stop")
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(content=metrics.reply),
                finish_reason=finish,  # type: ignore[arg-type]
            )
        ],
        usage=Usage(
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.output_tokens,
            total_tokens=metrics.prompt_tokens + metrics.output_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Streaming bridge primitives (sub-unit (d)).
# ---------------------------------------------------------------------------


def _abortable_put(
    queue: asyncio.Queue[Any],
    item: Any,
    loop: asyncio.AbstractEventLoop,
    abort_event: threading.Event,
    *,
    poll_interval: float = 0.1,
) -> None:
    """Thread-to-loop transfer that can abort while the queue is full.

    The plain pattern
    ``run_coroutine_threadsafe(queue.put(item), loop).result()`` blocks
    the worker thread indefinitely if the queue is full and the
    consumer has gone away — the generator's ``finally`` block would
    then time out, swallow the exception, and release
    ``runtime.engine_lock`` while the worker is still alive. That
    breaks G-1 (single active decode under engine_lock).

    This helper instead polls the future on ``poll_interval`` ticks
    and re-checks ``abort_event`` between polls. When the event fires
    mid-wait it cancels the in-flight ``queue.put`` coroutine and
    raises :class:`_ClientDisconnected`, which the worker catches to
    unwind ``ChatSession.chat`` cleanly.
    """
    if abort_event.is_set():
        raise _ClientDisconnected()
    future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
    while True:
        try:
            future.result(timeout=poll_interval)
            return
        except concurrent.futures.TimeoutError:
            if abort_event.is_set():
                future.cancel()
                raise _ClientDisconnected() from None


# ---------------------------------------------------------------------------
# SSE chunk formatting (sub-unit (d)).
# ---------------------------------------------------------------------------


def _format_sse(chunk: ChatCompletionChunk) -> bytes:
    """Serialize a chunk to the OpenAI SSE wire frame
    ``data: {...}\\n\\n``. Pydantic's ``model_dump_json`` with
    ``exclude_none=True`` drops the ``usage=None`` field on token
    chunks (only the terminal usage chunk emits it)."""
    payload = chunk.model_dump_json(exclude_none=True)
    return f"data: {payload}\n\n".encode()


def _make_role_chunk(
    *, request_id: str, model: str, created: int
) -> ChatCompletionChunk:
    """First SSE chunk: declares ``role='assistant'`` and an empty
    content delta. The OpenAI SDK uses this to assemble the
    streaming reply object before any tokens arrive."""
    return ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )


def _make_delta_chunk(
    *, request_id: str, model: str, created: int, content: str
) -> ChatCompletionChunk:
    """Token chunk: empty role, ``delta.content`` carries the
    text-delta string verbatim from
    :func:`ChatSession._on_token`'s already-UTF-8-corrected output."""
    return ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(content=content),
            )
        ],
    )


def _make_finish_chunk(
    *,
    request_id: str,
    model: str,
    created: int,
    finish_reason: str,
) -> ChatCompletionChunk:
    """Terminal token-side chunk: empty delta, populated
    ``finish_reason``. Followed by an optional usage chunk (when
    ``stream_options.include_usage=True``) and then ``data: [DONE]``."""
    return ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason=finish_reason,  # type: ignore[arg-type]
            )
        ],
    )


def _make_usage_chunk(
    *,
    request_id: str,
    model: str,
    created: int,
    metrics: TurnMetrics,
) -> ChatCompletionChunk:
    """Usage-only chunk emitted before ``[DONE]`` when
    ``stream_options.include_usage=True``. Reuses the same
    ``id`` / ``created`` / ``model`` as the token chunks per
    OpenAI's published wire shape; ``choices`` is empty and
    ``usage`` carries the totals."""
    return ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[],
        usage=Usage(
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.output_tokens,
            total_tokens=metrics.prompt_tokens + metrics.output_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Streaming branch.
# ---------------------------------------------------------------------------


def _stream_chat_completion(
    body: ChatCompletionRequest,
    runtime: Runtime,
    session: Any,
    user_text: str,
    params: SamplingParams,
) -> StreamingResponse:
    """Build the SSE :class:`StreamingResponse` for ``stream=True``.

    Architecture:

    1. A bounded :class:`asyncio.Queue` (size :data:`_STREAM_QUEUE_MAXSIZE`)
       carries items from the worker thread back to the SSE generator
       coroutine. Items are tagged by type — ``str`` is a text delta,
       :class:`TurnMetrics` is success completion, :data:`_DISCONNECTED`
       signals a client disconnect noticed by the worker, and any
       :class:`BaseException` is a worker error.
    2. A :class:`threading.Event` carries the abort signal from the
       generator's cleanup path back to the worker. The
       :func:`_on_delta` callback routes through :func:`_abortable_put`,
       which polls the in-flight ``run_coroutine_threadsafe`` future
       and re-checks the event between polls so a full queue + late
       disconnect raises :class:`_ClientDisconnected` instead of
       blocking the worker. ``ChatSession.chat`` then unwinds at the
       next streamed-delta boundary.
    3. The generator holds ``runtime.engine_lock`` for the **whole
       lifetime of the worker** (per OPENING G-1: single active
       decode). The ``finally`` block keeps draining the queue until
       ``worker.done()`` so the worker's in-flight ``queue.put`` can
       always complete, then surfaces any worker exception via
       :meth:`asyncio.Task.exception` synchronously (no
       ``await worker`` — that would tight-loop on a pending outer
       ``CancelledError``). This ensures G-1 is preserved even when
       the client disconnects mid-stream with the bounded queue
       full.
    4. Terminal items use no-drop transfers: the success-path
       metrics push and the failure-path exception push both go
       through :func:`_abortable_put` so a slow consumer never
       loses the finish/[DONE] sequence. Only the
       :data:`_DISCONNECTED` sentinel (where the consumer is
       already gone by definition) uses a bounded best-effort put.
    5. Every thread → loop transfer goes through
       :func:`asyncio.run_coroutine_threadsafe` per the (a1)
       lock+thread contract — never :meth:`asyncio.Queue.put_nowait`
       from the worker thread, which is not thread-safe and would
       drop tokens on a full queue.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    include_usage = bool(
        body.stream_options is not None
        and body.stream_options.include_usage
    )

    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
    abort_event = threading.Event()
    loop = asyncio.get_running_loop()

    def _on_delta(delta_text: str) -> None:
        # Per-token thread-to-loop transfer. Goes through
        # :func:`_abortable_put` so a full queue + late client
        # disconnect raises :class:`_ClientDisconnected` instead of
        # blocking the worker forever.
        _abortable_put(queue, delta_text, loop, abort_event)

    def _run_worker() -> None:
        try:
            metrics = session.chat(
                user_text,
                sampling_params=params,
                stream_to=_on_delta,
            )
        except _ClientDisconnected:
            # Consumer already gone — best-effort sentinel push. The
            # cleanup loop is draining and may not even need the
            # sentinel (it exits on ``worker.done()``), but emitting
            # it lets the generator's main loop close cleanly if it
            # is still active in the disconnect race window. Bounded
            # at 10 s because the consumer is by definition gone;
            # losing the sentinel is harmless.
            try:
                asyncio.run_coroutine_threadsafe(
                    queue.put(_DISCONNECTED), loop
                ).result(timeout=10.0)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "chat.completions stream disconnect sentinel "
                    "dropped (reason=%r)",
                    exc,
                )
            return
        except Exception as exc:
            # Always log at the worker site so a downstream queue.put
            # failure cannot lose the diagnostic.
            log.exception(
                "chat.completions stream worker session.chat failed"
            )
            # Surface the exception to the generator. Use abortable
            # put — must not drop on timeout (the generator still
            # needs the item to log + close) but a late client
            # disconnect should still unwind us through
            # :class:`_ClientDisconnected`.
            try:
                _abortable_put(queue, exc, loop, abort_event)
            except _ClientDisconnected:
                # Cleanup beat the exception push. Already logged
                # above; nothing more to do.
                pass
            return

        # Success path: the generator is blocked on this terminal
        # metrics item to emit the finish chunk + optional usage +
        # ``[DONE]``. MUST NOT drop on timeout — that would hang the
        # generator forever on the next ``queue.get()``. Use
        # ``_abortable_put`` so a slow consumer eventually unblocks
        # (the cleanup loop also drains, so a disconnect race
        # resolves through ``abort_event``).
        try:
            _abortable_put(queue, metrics, loop, abort_event)
        except _ClientDisconnected:
            # Client disconnected during the success-push window.
            # The generator already returned; nothing to push to.
            pass

    async def _generator() -> AsyncIterator[bytes]:
        async with runtime.engine_lock:
            worker = asyncio.create_task(asyncio.to_thread(_run_worker))
            try:
                yield _format_sse(
                    _make_role_chunk(
                        request_id=request_id,
                        model=body.model,
                        created=created,
                    )
                )

                while True:
                    item = await queue.get()
                    if isinstance(item, str):
                        yield _format_sse(
                            _make_delta_chunk(
                                request_id=request_id,
                                model=body.model,
                                created=created,
                                content=item,
                            )
                        )
                    elif isinstance(item, TurnMetrics):
                        finish = _FINISH_REASON_MAP.get(
                            item.finish_reason, "stop"
                        )
                        yield _format_sse(
                            _make_finish_chunk(
                                request_id=request_id,
                                model=body.model,
                                created=created,
                                finish_reason=finish,
                            )
                        )
                        if include_usage:
                            yield _format_sse(
                                _make_usage_chunk(
                                    request_id=request_id,
                                    model=body.model,
                                    created=created,
                                    metrics=item,
                                )
                            )
                        yield b"data: [DONE]\n\n"
                        return
                    elif item is _DISCONNECTED:
                        # Client gone before completion. No [DONE] —
                        # the consumer is no longer reading.
                        return
                    elif isinstance(item, BaseException):
                        log.error(
                            "chat.completions stream worker failed: %r",
                            item,
                        )
                        # SSE stream is mid-flight; closing the socket
                        # without [DONE] is the canonical "error
                        # mid-stream" signal in OpenAI-SSE land.
                        return
            finally:
                # G-1: hold ``runtime.engine_lock`` until the worker
                # truly exits. The worker may be stalled inside
                # ``_abortable_put`` waiting for queue room; we drain
                # the queue continuously so the next ``put`` (or the
                # in-flight one) can complete and the worker observes
                # ``abort_event`` on its next ``stream_to`` boundary.
                #
                # Cancellation handling: if Starlette propagates a
                # second cancellation (or a ``GeneratorExit`` chain
                # surfaces ``CancelledError`` at our awaits) during
                # cleanup, we absorb it and keep draining. Releasing
                # the lock before the worker exits would re-introduce
                # the G-1 violation that this drain-during-cleanup
                # design is here to prevent.
                abort_event.set()
                while not worker.done():
                    try:
                        await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        # Queue empty; loop and re-check ``worker.done()``.
                        # The worker either finished between checks or
                        # is still running through MLX compute; either
                        # way another short wait is the right move.
                        pass
                    except asyncio.CancelledError:
                        # Outer cancellation during cleanup — keep
                        # draining. The worker is still alive and we
                        # must hold engine_lock until it exits per
                        # OPENING G-1.
                        pass
                # Worker is done. Surface its outcome via sync
                # accessors instead of ``await worker``: an await on
                # an already-done task can still raise
                # ``CancelledError`` if our coroutine has a pending
                # cancellation, which would tight-loop a retry; and
                # we never want outer cancellation to reach the
                # worker task itself (sync access does neither).
                if not worker.cancelled():
                    exc = worker.exception()
                    if exc is not None:
                        log.error(
                            "chat.completions stream worker exit "
                            "error: %r",
                            exc,
                        )

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering (nginx, etc.) so chunks arrive
            # without a buffer flush delay. OpenAI's docs use this
            # header for the same reason.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Route.
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest, request: Request
) -> Any:
    """POST /v1/chat/completions dispatcher.

    Returns either a :class:`ChatCompletionResponse` (when
    ``body.stream`` is falsy) or a :class:`StreamingResponse` of SSE
    chunks (when ``body.stream`` is truthy). Pre-dispatch validation
    is identical for both branches.
    """
    runtime = _get_runtime(request)

    # Single-process server (OPENING §6.1.2 G-1) — silica.serve loads
    # exactly one model. A request for a different model is a 404 per
    # OpenAI's model_not_found semantics; without this check the
    # response would echo the wrong model id, which an OpenAI client
    # would treat as the answer coming from the requested model.
    if body.model != runtime.model_repo:
        raise HTTPException(
            status_code=404,
            detail=(
                f"model={body.model!r} is not loaded; this server "
                f"serves {runtime.model_repo!r}"
            ),
        )

    _validate_unsupported(body)
    _validate_sampling_bounds(body)
    max_tokens = _resolve_max_tokens(body)
    system_prompt, history, user_text = _extract_messages(body)

    tokenizer = runtime.adapter.tokenizer()
    eos_ids = tuple(
        sorted(getattr(tokenizer, "eos_token_ids", set()) or ())
    )
    params = _build_sampling_params(
        body, max_tokens=max_tokens, eos_ids=eos_ids
    )

    session = _session_factory(
        runtime, system_prompt=system_prompt, history=history
    )

    log.info(
        "chat.completions request model=%s stream=%s history_turns=%d "
        "user_text_len=%d max_tokens=%s",
        body.model,
        bool(body.stream),
        len(history),
        len(user_text),
        params.max_tokens,
    )

    if body.stream:
        return _stream_chat_completion(
            body, runtime, session, user_text, params
        )

    async with runtime.engine_lock:
        metrics: TurnMetrics = await asyncio.to_thread(
            session.chat,
            user_text,
            sampling_params=params,
        )

    log.info(
        "chat.completions reply model=%s finish_reason=%s "
        "prompt_tokens=%d output_tokens=%d",
        body.model,
        metrics.finish_reason,
        metrics.prompt_tokens,
        metrics.output_tokens,
    )

    return _build_response(metrics, model=body.model)
