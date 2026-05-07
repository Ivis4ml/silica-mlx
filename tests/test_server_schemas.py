"""Tests for :mod:`silica.server.schemas` (P-8 sub-unit (b)).

Covers the (b) acceptance row from ``plans/P8_OPENING.md`` §5:

- Five canonical OpenAI request fixtures (greedy / temperature /
  top-p / stream / non-stream) parse without validation errors.
- ``extra_body.extension.session_id`` parses into the silica
  :class:`Extension` envelope.
- The Extension envelope rejects unknown fields (``extra="forbid"``).
- Response models round-trip cleanly with ``extra="forbid"``.
- Unsupported-field policy: schemas under ``extra="allow"`` parse
  ``n > 1``, ``tools``, ``logprobs``, ``logit_bias`` etc. without
  error; routes (c) / (d) own the 400 / 501 rejection. The schema
  layer is *parse*, not *support*.
- ``max_tokens`` and ``max_completion_tokens`` both parse with no
  precedence decision; route (c) chooses.
- An openai-SDK round-trip — the openai Python client's
  ``client.chat.completions.create(...)`` serialises into a JSON
  body that :class:`ChatCompletionRequest` accepts.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from silica.server.schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Extension,
    ModelInfo,
    ModelsListResponse,
    Usage,
)

# Pure-pydantic schema tests run without the [serve] extra. The openai
# SDK round-trip tests below are guarded with a skipif so a dev env
# without [serve] still exercises the schema contract.
try:
    import httpx
    from openai import OpenAI

    _HAS_OPENAI_SDK = True
except ImportError:
    _HAS_OPENAI_SDK = False

# ---------------------------------------------------------------------------
# Five canonical OpenAI request fixtures (the (b) acceptance row).
# ---------------------------------------------------------------------------


_GREEDY: dict[str, Any] = {
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
    "temperature": 0,
    "max_tokens": 32,
}

_TEMPERATURE: dict[str, Any] = {
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "temperature": 0.7,
    "max_tokens": 64,
}

_TOP_P: dict[str, Any] = {
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Recommend a book."}],
    "top_p": 0.9,
    "temperature": 1.0,
    "max_tokens": 128,
}

_STREAM: dict[str, Any] = {
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": True,
    "max_tokens": 16,
    "stream_options": {"include_usage": True},
}

_NON_STREAM: dict[str, Any] = {
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": False,
    "max_tokens": 16,
}


@pytest.mark.parametrize(
    "fixture_name, payload",
    [
        ("greedy", _GREEDY),
        ("temperature", _TEMPERATURE),
        ("top_p", _TOP_P),
        ("stream", _STREAM),
        ("non_stream", _NON_STREAM),
    ],
)
def test_canonical_chat_request_fixtures_parse(
    fixture_name: str, payload: dict[str, Any]
) -> None:
    """The five canonical fixtures called out in the (b) acceptance row
    must parse without raising. Round-trip back to dict to check the
    declared fields survive.
    """
    request = ChatCompletionRequest(**payload)
    dumped = request.model_dump(exclude_none=True)

    assert request.model == payload["model"]
    assert len(request.messages) == len(payload["messages"])
    # Streaming flag must round-trip when explicitly set.
    if "stream" in payload:
        assert dumped.get("stream") == payload["stream"]


# ---------------------------------------------------------------------------
# Extension envelope (silica's strict surface).
# ---------------------------------------------------------------------------


def test_extension_parses_known_fields() -> None:
    ext = Extension(
        session_id="sess-abc",
        thinking_mode="off",
        continue_truncated=True,
    )

    assert ext.session_id == "sess-abc"
    assert ext.thinking_mode == "off"
    assert ext.continue_truncated is True


def test_extension_defaults_are_none() -> None:
    ext = Extension()
    assert ext.session_id is None
    assert ext.thinking_mode is None
    assert ext.continue_truncated is None


def test_extension_rejects_unknown_field() -> None:
    """A typo like ``sesion_id`` must fail loud rather than silently
    no-op. The Extension envelope is silica's own surface — strict.
    """
    with pytest.raises(ValidationError) as excinfo:
        Extension(sesion_id="sess-abc")  # type: ignore[call-arg]

    assert "sesion_id" in str(excinfo.value)


def test_extension_rides_under_top_level_field_in_request() -> None:
    """The openai client's ``extra_body={"extension": {...}}`` flattens
    into a top-level ``extension`` key in the JSON body; our request
    schema reads it from there.
    """
    payload = dict(_GREEDY)
    payload["extension"] = {"session_id": "sess-42", "thinking_mode": "auto"}

    request = ChatCompletionRequest(**payload)

    assert request.extension is not None
    assert request.extension.session_id == "sess-42"
    assert request.extension.thinking_mode == "auto"


# ---------------------------------------------------------------------------
# Permissive request fields (parse-but-not-support contract).
# ---------------------------------------------------------------------------


def test_unsupported_request_fields_parse_under_allow() -> None:
    """Schema layer is pure shape: ``n > 1``, ``tools``,
    ``tool_choice``, ``logprobs``, ``logit_bias``, ``presence_penalty``,
    ``frequency_penalty`` parse through under ``extra="allow"``.
    Routes (c) / (d) own the 400 / 501 rejection — pinned in their
    own test files later.
    """
    payload = dict(_GREEDY)
    payload.update(
        n=3,
        tools=[{"type": "function", "function": {"name": "noop"}}],
        tool_choice="auto",
        logprobs=True,
        top_logprobs=5,
        logit_bias={"123": 5},
        presence_penalty=0.5,
        frequency_penalty=-0.5,
    )

    # Must NOT raise. The route layer is what rejects these.
    request = ChatCompletionRequest(**payload)

    # Declared fields survive; the unsupported extras land in
    # model_extra (pydantic 2's bucket for ``extra="allow"`` fields).
    assert request.model_extra is not None
    assert request.model_extra.get("n") == 3
    assert request.model_extra.get("tools") == payload["tools"]
    assert request.model_extra.get("logprobs") is True


def test_max_tokens_and_max_completion_tokens_both_parse() -> None:
    """Schema parses both fields with no precedence decision. Route
    (c) decides what wins (and may 400 if they conflict).
    """
    payload = dict(_GREEDY)
    payload["max_tokens"] = 32
    payload["max_completion_tokens"] = 64

    request = ChatCompletionRequest(**payload)

    assert request.max_tokens == 32
    assert request.max_completion_tokens == 64


# ---------------------------------------------------------------------------
# ChatMessage content shapes.
# ---------------------------------------------------------------------------


def test_chat_message_accepts_string_content() -> None:
    msg = ChatMessage(role="user", content="hello")
    assert msg.content == "hello"


def test_chat_message_accepts_list_content_for_forward_compat() -> None:
    """OpenAI's wire shape allows multimodal content parts. v0.1
    routes only consume text, but the schema must parse the list
    form so a multimodal request gets a 501 from the route, not a
    422 from pydantic.
    """
    msg = ChatMessage(
        role="user",
        content=[{"type": "text", "text": "describe this"}],
    )
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "text"


# ---------------------------------------------------------------------------
# Response / chunk models (strict ``extra="forbid"``).
# ---------------------------------------------------------------------------


def test_chat_completion_response_round_trip() -> None:
    response = ChatCompletionResponse(
        id="chatcmpl-test",
        created=1700000000,
        model="Qwen/Qwen3.5-0.8B",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(content="ok"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
    )

    dumped = response.model_dump()
    assert dumped["object"] == "chat.completion"
    assert dumped["choices"][0]["message"]["role"] == "assistant"
    assert dumped["choices"][0]["message"]["content"] == "ok"
    assert dumped["usage"]["total_tokens"] == 5


def test_chat_completion_response_rejects_unknown_field() -> None:
    """Responses are silica's wire shape — strict."""
    with pytest.raises(ValidationError):
        ChatCompletionResponse(
            id="x",
            created=0,
            model="m",
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            mystery_field="boom",  # type: ignore[call-arg]
        )


def test_assistant_message_rejects_non_assistant_role() -> None:
    """AssistantMessage pins role='assistant'. A route bug that emits
    role='user' on a response must surface as a Pydantic validation
    error, not as a quietly out-of-spec wire shape on the
    OpenAI client.
    """
    with pytest.raises(ValidationError) as excinfo:
        AssistantMessage(role="user", content="ok")  # type: ignore[arg-type]

    assert "role" in str(excinfo.value)


def test_assistant_message_rejects_unknown_field() -> None:
    """AssistantMessage forbids extras. Stray fields like
    ``tool_calls`` from a future routing change must not slip onto
    the wire silently.
    """
    with pytest.raises(ValidationError) as excinfo:
        AssistantMessage(
            content="ok",
            tool_calls=[{"id": "x"}],  # type: ignore[call-arg]
        )

    assert "tool_calls" in str(excinfo.value)


def test_chat_completion_chunk_round_trip() -> None:
    chunk = ChatCompletionChunk(
        id="chatcmpl-test",
        created=1700000000,
        model="Qwen/Qwen3.5-0.8B",
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant", content="hi"),
            )
        ],
    )

    dumped = chunk.model_dump(exclude_none=True)
    assert dumped["object"] == "chat.completion.chunk"
    assert dumped["choices"][0]["delta"]["content"] == "hi"
    assert "usage" not in dumped  # only the terminal chunk carries it


def test_chat_completion_chunk_terminal_emits_usage() -> None:
    chunk = ChatCompletionChunk(
        id="x",
        created=0,
        model="m",
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )

    dumped = chunk.model_dump(exclude_none=True)
    assert dumped["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }


# ---------------------------------------------------------------------------
# Completions (legacy) + models listing.
# ---------------------------------------------------------------------------


def test_completion_request_parses_string_prompt() -> None:
    request = CompletionRequest(
        model="Qwen/Qwen3.5-0.8B",
        prompt="The capital of France is",
        max_tokens=8,
    )
    assert request.prompt == "The capital of France is"


def test_completion_request_parses_list_prompt_for_forward_compat() -> None:
    """List prompts parse at the schema layer; v0.1 route returns 400
    if anything other than a single string is supplied (per
    docstring on :class:`CompletionRequest`)."""
    request = CompletionRequest(
        model="m",
        prompt=["a", "b"],
        max_tokens=4,
    )
    assert request.prompt == ["a", "b"]


def test_completion_response_round_trip() -> None:
    response = CompletionResponse(
        id="cmpl-test",
        created=1700000000,
        model="Qwen/Qwen3.5-0.8B",
        choices=[CompletionChoice(text=" Paris.", index=0, finish_reason="stop")],
        usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )

    dumped = response.model_dump()
    assert dumped["object"] == "text_completion"
    assert dumped["choices"][0]["text"] == " Paris."


def test_models_list_response_round_trip() -> None:
    listing = ModelsListResponse(
        data=[ModelInfo(id="Qwen/Qwen3.5-0.8B", created=1700000000)]
    )

    dumped = listing.model_dump()
    assert dumped["object"] == "list"
    assert dumped["data"][0]["owned_by"] == "silica"


# ---------------------------------------------------------------------------
# OpenAI-SDK serialisation round-trip (the load-bearing acceptance).
# ---------------------------------------------------------------------------


def _capture_openai_request_body(**create_kwargs: Any) -> dict[str, Any]:
    """Drive ``openai.OpenAI(...).chat.completions.create(...)`` against
    a captured :class:`httpx.MockTransport` and return the raw JSON
    body the SDK would send on the wire.

    Returns a stub :class:`ChatCompletionResponse` so the SDK does
    not raise on the response side; the test only inspects what was
    sent.
    """
    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": "stub",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    transport = httpx.MockTransport(_handler)
    client = OpenAI(
        api_key="dummy",
        base_url="http://localhost:9999/v1",
        http_client=httpx.Client(transport=transport),
    )
    client.chat.completions.create(**create_kwargs)
    body: dict[str, Any] = captured["body"]
    return body


@pytest.mark.skipif(
    not _HAS_OPENAI_SDK,
    reason="P-8 [serve] extra not installed (openai / httpx missing)",
)
def test_openai_sdk_serialises_into_chat_completion_request() -> None:
    """The openai Python client's ``chat.completions.create`` call
    produces a wire body that :class:`ChatCompletionRequest` accepts
    without validation errors. This is the load-bearing M-9.1
    plumbing: if our schema cannot parse what the SDK emits, sub-unit
    (c) cannot serve the SDK.
    """
    body = _capture_openai_request_body(
        model="Qwen/Qwen3.5-0.8B",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi."},
        ],
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        stream=False,
    )

    # Sanity on the captured body.
    assert body["model"] == "Qwen/Qwen3.5-0.8B"
    assert body["messages"][0]["role"] == "system"

    # The (b) acceptance: schema accepts what the SDK emits.
    request = ChatCompletionRequest(**body)
    assert request.model == "Qwen/Qwen3.5-0.8B"
    assert len(request.messages) == 2


@pytest.mark.skipif(
    not _HAS_OPENAI_SDK,
    reason="P-8 [serve] extra not installed (openai / httpx missing)",
)
def test_openai_sdk_extra_body_extension_round_trips() -> None:
    """``extra_body={"extension": {...}}`` flattens into a top-level
    ``extension`` field on the wire, which our :class:`Extension`
    envelope parses strictly.
    """
    body = _capture_openai_request_body(
        model="Qwen/Qwen3.5-0.8B",
        messages=[{"role": "user", "content": "Hi."}],
        extra_body={
            "extension": {
                "session_id": "sess-openai-sdk",
                "thinking_mode": "off",
            }
        },
    )

    assert body["extension"]["session_id"] == "sess-openai-sdk"

    request = ChatCompletionRequest(**body)
    assert request.extension is not None
    assert request.extension.session_id == "sess-openai-sdk"
    assert request.extension.thinking_mode == "off"


def _capture_openai_completions_request_body(
    **create_kwargs: Any,
) -> dict[str, Any]:
    """Drive ``openai.OpenAI(...).completions.create(...)`` against a
    captured :class:`httpx.MockTransport` and return the raw JSON body
    the SDK would send on the wire. Returns a stub
    :class:`CompletionResponse` so the SDK does not raise on the
    response side; the test only inspects what was sent.
    """
    captured: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-stub",
                "object": "text_completion",
                "created": 0,
                "model": "stub",
                "choices": [
                    {
                        "text": " stub",
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    transport = httpx.MockTransport(_handler)
    client = OpenAI(
        api_key="dummy",
        base_url="http://localhost:9999/v1",
        http_client=httpx.Client(transport=transport),
    )
    client.completions.create(**create_kwargs)
    body: dict[str, Any] = captured["body"]
    return body


@pytest.mark.skipif(
    not _HAS_OPENAI_SDK,
    reason="P-8 [serve] extra not installed (openai / httpx missing)",
)
def test_openai_sdk_serialises_into_completion_request() -> None:
    """``client.completions.create(...)`` produces a wire body that
    :class:`CompletionRequest` accepts. R-e gate parity for the
    legacy text-completions endpoint."""
    body = _capture_openai_completions_request_body(
        model="Qwen/Qwen3.5-0.8B",
        prompt="The capital of France is",
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        stream=False,
    )

    assert body["model"] == "Qwen/Qwen3.5-0.8B"
    assert body["prompt"] == "The capital of France is"

    request = CompletionRequest(**body)
    assert request.model == "Qwen/Qwen3.5-0.8B"
    assert request.prompt == "The capital of France is"
    assert request.max_tokens == 32


@pytest.mark.skipif(
    not _HAS_OPENAI_SDK,
    reason="P-8 [serve] extra not installed (openai / httpx missing)",
)
def test_openai_sdk_parses_completion_response() -> None:
    """The SDK can deserialise a :class:`CompletionResponse`-shaped
    payload into its ``Completion`` object. R-e response-side gate.
    """
    captured: dict[str, Any] = {}

    response_body = CompletionResponse(
        id="cmpl-test",
        created=1_700_000_000,
        model="Qwen/Qwen3.5-0.8B",
        choices=[
            CompletionChoice(text=" Paris.", index=0, finish_reason="stop")
        ],
        usage=Usage(prompt_tokens=8, completion_tokens=2, total_tokens=10),
    ).model_dump()

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json=response_body)

    transport = httpx.MockTransport(_handler)
    client = OpenAI(
        api_key="dummy",
        base_url="http://localhost:9999/v1",
        http_client=httpx.Client(transport=transport),
    )

    sdk_response = client.completions.create(
        model="Qwen/Qwen3.5-0.8B",
        prompt="The capital of France is",
        max_tokens=8,
    )

    assert captured["path"] == "/v1/completions"
    assert sdk_response.id == "cmpl-test"
    assert sdk_response.choices[0].text == " Paris."
    assert sdk_response.choices[0].finish_reason == "stop"
    assert sdk_response.usage is not None
    assert sdk_response.usage.total_tokens == 10


@pytest.mark.skipif(
    not _HAS_OPENAI_SDK,
    reason="P-8 [serve] extra not installed (openai / httpx missing)",
)
def test_openai_sdk_parses_models_list_response() -> None:
    """The SDK can deserialise a :class:`ModelsListResponse`-shaped
    payload into its model list. R-e response-side gate for the
    metadata endpoint.
    """
    captured: dict[str, Any] = {}

    response_body = ModelsListResponse(
        data=[
            ModelInfo(id="Qwen/Qwen3.5-0.8B", created=1_700_000_000),
        ]
    ).model_dump()

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["method"] = request.method
        return httpx.Response(200, json=response_body)

    transport = httpx.MockTransport(_handler)
    client = OpenAI(
        api_key="dummy",
        base_url="http://localhost:9999/v1",
        http_client=httpx.Client(transport=transport),
    )

    page = client.models.list()
    models = list(page)

    assert captured["method"] == "GET"
    assert captured["path"] == "/v1/models"
    assert len(models) == 1
    assert models[0].id == "Qwen/Qwen3.5-0.8B"
    assert models[0].owned_by == "silica"
    assert models[0].object == "model"
    assert models[0].created == 1_700_000_000
