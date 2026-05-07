"""silica.server.routes.completions — POST /v1/completions
non-streaming branch (P-8 sub-unit (e)).

Legacy text-completion endpoint. Drives :meth:`silica.engine.Engine.generate`
directly — no chat template, no system prompt, no message history. The
prompt string is tokenised, fed through prefill + decode, and the
detokenised reply lands as a single :class:`CompletionChoice`.

This is intentionally lighter than ``/v1/chat/completions``: there is
no streaming branch in v0.1 (``stream=True`` returns 501; lifting
that gate is deferred to a follow-on slice that will mirror the
``_abortable_put`` bridge from sub-unit (d)). Single-customer
serialisation under ``runtime.engine_lock`` matches the OPENING G-1
contract.

Pipeline
========

1. **Model-id match** — return 404 if ``body.model`` differs from
   ``runtime.model_repo``.
2. :func:`_validate_unsupported` — return 501 for fields that change
   output semantics or are not yet wired (``stream=True``, non-str
   prompt, string-sequence ``stop``, non-empty ``extension``,
   ``n>1``, ``logprobs``, ``presence_penalty != 0``,
   ``frequency_penalty != 0``, ``suffix``, ``echo``, ``best_of``).
3. :func:`_validate_sampling_bounds` — return 400 for sampling
   parameters outside their accepted range (temperature, top_p,
   max_tokens). Without this the pydantic ``Field(ge / gt)``
   constraints on :class:`silica.core.sampling.SamplingParams` would
   raise inside the handler and surface as a 500.
4. :func:`_build_sampling_params` — assemble
   :class:`silica.core.sampling.SamplingParams`. Tokenizer EOS ids
   are wired into ``stop_token_ids`` so generation halts naturally.
5. ``async with runtime.engine_lock: await asyncio.to_thread(
   _drive_generate, ...)`` — runs MLX compute off the event loop
   per the Runtime docstring's lock+thread contract.
6. :func:`_build_response` — assembles
   :class:`CompletionResponse` with strict
   :class:`CompletionChoice` and :class:`Usage`.

Finish-reason logic mirrors :meth:`ChatSession._classify_finish`:
last yielded token in tokenizer EOS set → ``stop`` (the model
emitted a natural terminator, even when the cap was reached on
the same step); ``len(output_ids) >= max_tokens`` without EOS →
``length``; empty / shorter output → ``stop``. The trailing EOS
token is also stripped before decoding the response text so
``choices[0].text`` never leaks the literal ``<|im_end|>``-style
marker — same invariant the chat-CLI enforces at
:meth:`ChatSession.chat`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from silica.core.logger import get_logger
from silica.core.sampling import SamplingParams
from silica.server.runtime import Runtime
from silica.server.schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)

log = get_logger(__name__)

router = APIRouter()


def _get_runtime(request: Request) -> Runtime:
    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.closed:
        raise HTTPException(status_code=503, detail="engine not ready")
    return runtime


def _validate_unsupported(req: CompletionRequest) -> None:
    """Return 501 for fields that change output semantics or are not
    wired in v0.1. Schema layer parses everything under
    ``extra='allow'``; this is the route-level rejection per the (b)
    review's "parse != supported" rule.
    """
    if req.stream:
        # Streaming for /v1/completions is deferred — re-using the
        # chat-completions SSE bridge here is a follow-on slice
        # (``_abortable_put`` is module-private to chat_completions
        # in v0.1; lifting it requires either an extraction or a
        # per-route copy).
        raise HTTPException(
            status_code=501,
            detail=(
                "stream=True on /v1/completions is not yet wired in "
                "v0.1; non-streaming is the only supported branch"
            ),
        )

    if not isinstance(req.prompt, str):
        raise HTTPException(
            status_code=501,
            detail=(
                "non-string prompt (list[str] / list[int] / "
                "list[list[int]]) is not supported in v0.1; pass a "
                "single string prompt"
            ),
        )

    if req.stop is not None:
        raise HTTPException(
            status_code=501,
            detail=(
                "string-sequence 'stop' not supported in v0.1; "
                "generation terminates only on tokenizer EOS"
            ),
        )

    if req.extension is not None:
        ext_fields = req.extension.model_dump(exclude_none=True)
        if ext_fields:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"extension fields {sorted(ext_fields.keys())} "
                    "are not honoured in v0.1"
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
    if extra.get("logprobs") is not None:
        raise HTTPException(
            status_code=501, detail="logprobs not supported in v0.1"
        )
    if extra.get("suffix") is not None:
        raise HTTPException(
            status_code=501, detail="suffix not supported in v0.1"
        )
    if extra.get("echo"):
        raise HTTPException(
            status_code=501, detail="echo not supported in v0.1"
        )
    if extra.get("best_of") not in (None, 1):
        raise HTTPException(
            status_code=501, detail="best_of not supported in v0.1"
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
    if extra.get("logit_bias") is not None:
        raise HTTPException(
            status_code=501, detail="logit_bias not supported in v0.1"
        )


def _validate_sampling_bounds(req: CompletionRequest) -> None:
    """Return 400 for sampling parameters outside their accepted
    range (mirror of the chat-completions implementation)."""
    if req.temperature is not None and not (0.0 <= req.temperature <= 2.0):
        raise HTTPException(
            status_code=400,
            detail=f"temperature={req.temperature} out of range [0, 2]",
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


def _build_sampling_params(
    req: CompletionRequest,
    *,
    eos_ids: tuple[int, ...],
) -> SamplingParams:
    """Map OpenAI request fields onto silica's sampling-params model."""
    return SamplingParams(
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p,
        top_k=None,
        max_tokens=req.max_tokens if req.max_tokens is not None else 256,
        stop=(),
        stop_token_ids=eos_ids,
        seed=req.seed,
    )


def _drive_generate(
    runtime: Runtime, prompt: str, params: SamplingParams
) -> tuple[list[int], int]:
    """Run :meth:`Engine.generate` to exhaustion and return the
    raw output token ids alongside the prompt-token count.

    Returns ``(output_ids, prompt_tokens)``. The ids include any
    trailing EOS that ``Engine.generate`` yielded as its terminator
    (per the engine contract: stop tokens are emitted before
    termination). The route strips that EOS before decoding for the
    response text, mirroring
    :meth:`silica.chat.session.ChatSession.chat`'s pattern. Tests
    monkeypatch this function at the module level to inject
    deterministic id sequences without driving a real model.

    Re-tokenises the prompt locally for the prompt-token count
    rather than reaching into engine state — the engine's tokenizer
    is the same object as ``runtime.adapter.tokenizer()`` so the
    count agrees, and re-tokenising costs nothing relative to the
    generation forward.
    """
    tokenizer = runtime.adapter.tokenizer()
    prompt_token_count = len(tokenizer.encode(prompt))
    output_ids = list(runtime.engine.generate(prompt, params))
    return output_ids, prompt_token_count


def _classify_finish_reason(
    output_ids: list[int],
    *,
    eos_ids: frozenset[int],
    max_tokens: int,
) -> str:
    """Map an engine output sequence to OpenAI's ``finish_reason``.

    Mirrors :meth:`ChatSession._classify_finish` then collapses the
    silica vocabulary into OpenAI's two-value choice for completions:

    - last yielded token is EOS → ``"stop"`` (the model emitted a
      natural terminator, even if the cap was simultaneously reached);
    - sequence reached ``max_tokens`` without EOS → ``"length"``;
    - any other shape (empty output, plain stop) → ``"stop"``.

    The EOS-first ordering is load-bearing: when the model emits EOS
    as its ``max_tokens``-th yielded token, the OpenAI semantics
    require ``"stop"`` (natural terminator), not ``"length"`` (cap
    hit). The reverse ordering would mis-classify and the route
    would also leak the EOS marker into ``choices[0].text``.
    """
    if not output_ids:
        return "stop"
    if output_ids[-1] in eos_ids:
        return "stop"
    if len(output_ids) >= max_tokens:
        return "length"
    return "stop"


def _strip_trailing_eos(
    output_ids: list[int], *, eos_ids: frozenset[int]
) -> list[int]:
    """Drop a trailing EOS token if present.

    ``Engine.generate`` yields the stop token before terminating
    (see :mod:`silica.engine.__init__`), and :meth:`Tokenizer.decode`
    on Qwen / mlx-lm tokenizers materialises the literal
    ``<|im_end|>``-style marker into the response text. Strip it
    here so ``choices[0].text`` does not leak the marker — same
    invariant the chat-CLI enforces at
    ``silica.chat.session.ChatSession.chat``.
    """
    if output_ids and output_ids[-1] in eos_ids:
        return output_ids[:-1]
    return output_ids


def _build_response(
    *,
    text: str,
    prompt_tokens: int,
    output_tokens: int,
    finish_reason: str,
    model: str,
) -> CompletionResponse:
    """Assemble the strict :class:`CompletionResponse`."""
    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            CompletionChoice(
                text=text,
                index=0,
                finish_reason=finish_reason,  # type: ignore[arg-type]
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
        ),
    )


@router.post("/v1/completions")
async def completions(
    body: CompletionRequest, request: Request
) -> CompletionResponse:
    """POST /v1/completions non-streaming dispatcher."""
    runtime = _get_runtime(request)

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
    # body.prompt is guaranteed to be str at this point — the
    # _validate_unsupported step above 501s every other shape.
    assert isinstance(body.prompt, str)
    prompt: str = body.prompt

    tokenizer = runtime.adapter.tokenizer()
    eos_ids_tuple = tuple(
        sorted(getattr(tokenizer, "eos_token_ids", set()) or ())
    )
    eos_ids = frozenset(eos_ids_tuple)
    params = _build_sampling_params(body, eos_ids=eos_ids_tuple)

    log.info(
        "completions request model=%s prompt_len=%d max_tokens=%s",
        body.model,
        len(prompt),
        params.max_tokens,
    )

    async with runtime.engine_lock:
        output_ids, prompt_tokens = await asyncio.to_thread(
            _drive_generate, runtime, prompt, params
        )

    finish_reason = _classify_finish_reason(
        output_ids, eos_ids=eos_ids, max_tokens=params.max_tokens
    )
    reply_ids = _strip_trailing_eos(output_ids, eos_ids=eos_ids)
    # ``rstrip("�")`` mirrors ChatSession.chat — a trailing replacement
    # char indicates an incomplete multi-byte UTF-8 sequence cut off
    # by EOS / max_tokens. Match the chat-CLI invariant so completion
    # text does not surface a half-decoded byte.
    text = tokenizer.decode(reply_ids).rstrip("�")

    log.info(
        "completions reply model=%s prompt_tokens=%d output_tokens=%d "
        "finish_reason=%s",
        body.model,
        prompt_tokens,
        len(output_ids),
        finish_reason,
    )

    return _build_response(
        text=text,
        prompt_tokens=prompt_tokens,
        output_tokens=len(output_ids),
        finish_reason=finish_reason,
        model=body.model,
    )
