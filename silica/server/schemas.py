"""silica.server.schemas — OpenAI-compatible request / response Pydantic
v2 models for the silica HTTP server (P-8 sub-unit (b)).

Wire shapes covered:

- ``/v1/chat/completions`` — :class:`ChatCompletionRequest`,
  :class:`ChatCompletionResponse`, :class:`ChatCompletionChunk`.
- ``/v1/completions`` — :class:`CompletionRequest`,
  :class:`CompletionResponse`.
- ``/v1/models`` — :class:`ModelInfo`, :class:`ModelsListResponse`.

Design rules (per ``plans/P8_OPENING.md`` review of (b)):

- **Requests use** ``extra="allow"``. The OpenAI SDK evolves; silica
  must not 400 on every new field. Unknown fields parse through;
  routes decide whether to ignore or reject (semantic-changing
  fields like ``n > 1`` / ``tools`` / ``logprobs`` get 400 / 501 in
  (c) / (d), purely cosmetic SDK additions get ignored).
- **Responses + the silica :class:`Extension` envelope use**
  ``extra="forbid"``. Responses are silica's wire shape; we control
  what we emit. The Extension envelope is silica's own surface — a
  typo in ``extra_body.extension.session_id`` should fail loud, not
  silently no-op.
- **Schema layer is pure shape; business semantics live in routes.**
  Fields like ``n > 1``, ``tools`` / ``tool_choice``, ``logprobs``,
  ``logit_bias``, ``presence_penalty``, ``frequency_penalty`` parse
  through (under ``extra="allow"`` on requests, plus ``logprobs`` /
  ``response_format`` declared but not honoured below), but routes
  (c) / (d) / (f) are responsible for returning 400 / 501 when an
  unsupported field changes output semantics. The (b) acceptance is
  *parse*, not *support*.
- **Both ``max_tokens`` and ``max_completion_tokens`` parse.** Routes
  decide precedence (and may 400 when both are set with conflicting
  values) — the schema does not pick a winner.

The :class:`Extension` envelope rides under a top-level ``extension``
field in the JSON body; the OpenAI Python client sends silica
extensions via ``extra_body={"extension": {...}}`` which the SDK
flattens into the body. The canonical session selector is the
``X-Silica-Session-ID`` HTTP header (or
``extra_body.extension.session_id``) — see
``plans/P8_OPENING.md`` §4.3 + §6.1.2; the OpenAI ``user`` field is
**not** consulted as session id.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------


class Extension(BaseModel):
    """silica-specific request extensions.

    Lives at top level in the JSON body under the ``extension`` field.
    The openai Python client sends this via
    ``extra_body={"extension": {"session_id": "...", ...}}``. The
    envelope is strict (``extra="forbid"``): a typo in a known field
    raises 422 rather than silently no-op.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(
        default=None,
        description=(
            "Conversation session selector. Preferred form is the "
            "``X-Silica-Session-ID`` HTTP header; this body-level "
            "shadow is provided so SDKs that cannot set headers can "
            "still address a persistent ChatSession."
        ),
    )
    thinking_mode: Literal["auto", "on", "off"] | None = Field(
        default=None,
        description=(
            "Override Qwen3 / Qwen3.5 reasoning-mode behaviour for "
            "this request. ``auto`` keeps the chat-template default "
            "(reasoning on for those families); ``on`` forces "
            "reasoning; ``off`` disables it."
        ),
    )
    continue_truncated: bool | None = Field(
        default=None,
        description=(
            "If ``true`` and the addressed session's last turn ended "
            "with finish_reason=length, continue from the truncation "
            "point via :meth:`ChatSession.continue_last`. Default "
            "``false``."
        ),
    )


class Usage(BaseModel):
    """Token-count usage report attached to non-streaming responses
    and the terminal SSE chunk's ``usage`` field (when emitted).
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StreamOptions(BaseModel):
    """``stream_options`` request field — applies only when
    ``stream=True``.

    Uses ``extra="allow"`` because ``stream_options`` is an
    OpenAI-owned surface (not a silica-owned envelope like
    :class:`Extension`); future SDK additions must not 422 the
    endpoint. v0.1 honours :attr:`include_usage`; unknown fields
    parse through here and are ignored by the route, with the
    parse-but-not-support trade-off matching the request-level
    ``extra="allow"`` policy at the top of this module. (h) hardening
    can promote specific known-but-unsupported fields to route-level
    501.
    """

    model_config = ConfigDict(extra="allow")

    include_usage: bool | None = None


class ResponseFormat(BaseModel):
    """``response_format`` request field.

    v0.1 parses this but does not honour it — routes (c) / (d) return
    501 if a non-``text`` ``type`` is requested. Kept here so the
    interface slot is reserved for future structured-output work
    (sub-unit (h)).
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


# Finish reasons silica emits. v0.1 does not produce ``tool_calls`` or
# ``content_filter``; the latter is reserved for later moderation work.
FinishReason = Literal["stop", "length", "content_filter"]


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """One message in an inbound chat conversation (request side only).

    ``content`` is permissive — OpenAI's wire shape allows a string
    or a list of content parts (text, images, audio, …). v0.1 only
    consumes text content; non-text parts parse here, and the route
    returns 501 / 400 for non-text content (no silent drop). The
    response side uses :class:`AssistantMessage`, which is strict
    (``extra="forbid"``) so silica only emits well-formed assistant
    replies.
    """

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[Any] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[Any] | None = None


class AssistantMessage(BaseModel):
    """One message silica emits in a non-streaming chat-completion
    response.

    Strict by construction (``extra="forbid"``, ``role`` pinned to
    ``"assistant"``, ``content`` either ``str`` or ``None``) so a
    route bug that constructs a malformed response — wrong role,
    nested tool-call payload, stray field — surfaces as a 500 from
    Pydantic rather than as a quietly out-of-spec OpenAI response.
    Tool-calls / multimodal output are reserved for post-v0.1.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` request body.

    ``extra="allow"`` so unknown OpenAI-SDK fields parse through;
    routes own the 400 / 501 rejection for fields that change output
    semantics (``n != 1``, ``tools``, ``tool_choice``, ``logprobs``,
    ``logit_bias``).
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str
    messages: list[ChatMessage]

    # Sampling (passed to silica.core.sampling.SamplingParams in (c)).
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None  # OpenAI abuse-monitor field — NOT a session id.

    # Streaming.
    stream: bool | None = None
    stream_options: StreamOptions | None = None

    # Reserved interface slot — v0.1 routes 501 unless type == "text".
    response_format: ResponseFormat | None = None

    # silica-specific extensions.
    extension: Extension | None = None


class ChatCompletionChoice(BaseModel):
    """Non-streaming response choice. ``message`` is the strict
    :class:`AssistantMessage` rather than the request-side
    :class:`ChatMessage`, so silica only emits well-formed assistant
    replies on the wire."""

    model_config = ConfigDict(extra="forbid")

    index: int
    message: AssistantMessage
    finish_reason: FinishReason | None = None
    logprobs: None = None  # not produced; reserved for future


class ChatCompletionResponse(BaseModel):
    """``POST /v1/chat/completions`` non-streaming response body."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunkDelta(BaseModel):
    """Per-token delta inside a streaming chunk."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """Streaming-chunk choice."""

    model_config = ConfigDict(extra="forbid")

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: FinishReason | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    """One chunk of a streaming chat-completion response.

    Wire form is the SSE ``data:`` payload — the route serialises
    this model to JSON and frames it with ``data: {...}\\n\\n``.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # populated only on the terminal chunk


# ---------------------------------------------------------------------------
# Completions (legacy /v1/completions)
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """``POST /v1/completions`` request body (legacy, no chat template).

    v0.1 supports a single ``str`` prompt; ``list[str]``,
    ``list[int]``, and ``list[list[int]]`` parse here under the
    structural union but the route returns 501 if anything other
    than ``str`` is provided (P-8 v0.1 ties completions to a single
    tokenised prompt; batch / token-id inputs are post-announce).
    Mirrors the chat-completions "parsed but not supported → 501"
    rule from sub-unit (c).
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None

    extension: Extension | None = None


class CompletionChoice(BaseModel):
    """Non-streaming completion choice."""

    model_config = ConfigDict(extra="forbid")

    text: str
    index: int
    finish_reason: FinishReason | None = None
    logprobs: None = None


class CompletionResponse(BaseModel):
    """``POST /v1/completions`` non-streaming response body."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """Single ``/v1/models`` entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "silica"


class ModelsListResponse(BaseModel):
    """``GET /v1/models`` response body."""

    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[ModelInfo]


# Public re-export list. Routes / tests import via these names so
# refactors stay localised.
__all__ = [
    "AssistantMessage",
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "CompletionChoice",
    "CompletionRequest",
    "CompletionResponse",
    "Extension",
    "FinishReason",
    "ModelInfo",
    "ModelsListResponse",
    "ResponseFormat",
    "StreamOptions",
    "Usage",
]
