"""silica.server.routes.chat_completions — POST /v1/chat/completions
non-streaming branch (P-8 sub-unit (c)).

Synchronous chat-turn handler driven by
:meth:`silica.chat.session.ChatSession.chat` with ``stream_to=None``.
One conversation per HTTP request; v0.1 has no cross-request session
reuse (sub-unit (f) lands ``SessionManager``).

Pipeline
========

1. **Model-id match** — return 404 if ``body.model`` differs from
   ``runtime.model_repo`` (the single-process server serves exactly
   one model; echoing a wrong id would mislead OpenAI clients).
2. :func:`_validate_unsupported` — return 501 for fields that
   change output semantics or are not yet wired
   (``stream=True``, string-sequence ``stop``, non-empty
   ``extension`` envelope, ``n>1``, ``tools``, ``tool_choice``,
   ``logprobs``, ``top_logprobs``, ``logit_bias``,
   ``presence_penalty != 0``, ``frequency_penalty != 0``,
   ``response_format.type != 'text'``, multimodal content,
   non-{system, user, assistant} roles).
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
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from silica.chat.session import ChatSession, TurnMetrics
from silica.core.logger import get_logger
from silica.core.sampling import SamplingParams
from silica.server.runtime import Runtime
from silica.server.schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)

log = get_logger(__name__)

router = APIRouter()


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
    if req.stream:
        # /v1/chat/completions accepts both stream=True and stream=False;
        # (c) is the non-streaming branch. (d) lands the SSE branch and
        # will replace this 501.
        raise HTTPException(
            status_code=501,
            detail="streaming not yet implemented (P-8 sub-unit (d))",
        )

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
# Route.
# ---------------------------------------------------------------------------


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
async def chat_completions(
    body: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse:
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
        "chat.completions request model=%s history_turns=%d "
        "user_text_len=%d max_tokens=%s",
        body.model,
        len(history),
        len(user_text),
        params.max_tokens,
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
