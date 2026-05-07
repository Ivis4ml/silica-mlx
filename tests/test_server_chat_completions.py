"""Tests for :mod:`silica.server.routes.chat_completions` (P-8 sub-unit (c)).

Lands the (c) **unit** coverage: an HTTP POST to
``/v1/chat/completions`` with ``stream=False`` round-trips through
the route into :meth:`ChatSession.chat` and back to a well-formed
:class:`ChatCompletionResponse`. The OPENING (c) acceptance row R-c
(``openai.OpenAI`` client against real Qwen/Qwen3.5-0.8B with
``usage.prompt_tokens`` parity) is the manual real-model smoke and
is **deferred** to (h) — these tests use a stub session factory
returning a deterministic :class:`TurnMetrics` so no MLX compute or
HuggingFace load happens in CI.

Test surface:

- 200 round trip: route correctly maps OpenAI request → session
  factory → ChatSession.chat → ChatCompletionResponse with
  ``usage.prompt_tokens`` matching :class:`TurnMetrics.prompt_tokens``
  exactly (per-field unit pin; full-system R-c parity is the
  deferred manual smoke).
- ``messages`` extraction: leading system messages are concatenated;
  prior user / assistant turns become history; the last user
  message becomes ``user_text``.
- Sampling-params mapping: ``temperature`` / ``top_p`` / ``seed`` /
  ``max_tokens`` propagate; ``max_completion_tokens`` is honoured
  when set; default is silica's 256 when neither cap is provided;
  ``stop`` is rejected upstream so the route always emits
  ``stop=()``.
- 400 reject matrix: empty messages, last-not-user,
  ``max_tokens`` vs ``max_completion_tokens`` conflict, non-leading
  system message, sampling parameters out of range
  (``temperature``, ``top_p``, ``max_tokens``,
  ``max_completion_tokens``).
- 501 reject matrix: ``stream=True``, string-sequence ``stop``,
  non-empty ``extension`` envelope, ``n>1``, ``tools``,
  ``tool_choice``, ``logprobs``, ``top_logprobs``, ``logit_bias``,
  ``presence_penalty``, ``frequency_penalty``,
  ``response_format.type='json_object'``, multimodal content,
  ``role='developer'``.
- 404 on ``body.model`` ≠ ``runtime.model_repo``.
- 503 when runtime is closed mid-request.
- Finish-reason map: ``stop_token`` → ``stop``, ``max_tokens`` →
  ``length``, ``empty`` → ``stop``.
"""

from __future__ import annotations

from collections.abc import Iterator
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
# Stub session that captures the route's call and returns a fixed
# TurnMetrics. Used as the body of the monkeypatched session factory.
# ---------------------------------------------------------------------------


class _CapturingStubSession:
    """Records the route's invocation and returns a deterministic
    :class:`TurnMetrics`. The route only calls ``chat`` so that's the
    only method we implement; future routes (continue_last in (f))
    will need additions when they are added.
    """

    def __init__(
        self,
        *,
        reply: str = "Hello there!",
        prompt_tokens: int = 11,
        output_tokens: int = 3,
        finish_reason: str = "stop_token",
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> None:
        self.reply = reply
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason
        self.system_prompt = system_prompt
        self.history = list(history or [])
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
        return TurnMetrics(
            reply=self.reply,
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
def _isolate_module_state(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Wipe :data:`openai_api._config` between tests and restore the
    default session factory after each test so monkeypatched stubs
    do not leak."""
    openai_api._config = None
    yield
    openai_api._config = None


def _configure_runtime() -> Runtime:
    runtime = _build_stub_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    return runtime


def _install_stub_session(
    monkeypatch: pytest.MonkeyPatch,
    **stub_kwargs: Any,
) -> dict[str, Any]:
    """Replace ``cc._session_factory`` with a hook that returns a
    :class:`_CapturingStubSession`. Returns a dict ``{"session": ...}``
    that the test populates and inspects after the request runs.
    """
    holder: dict[str, Any] = {"session": None}

    def _factory(
        runtime: Runtime,
        *,
        system_prompt: str | None,
        history: list[dict[str, str]],
    ) -> _CapturingStubSession:
        session = _CapturingStubSession(
            system_prompt=system_prompt,
            history=history,
            **stub_kwargs,
        )
        holder["session"] = session
        return session

    monkeypatch.setattr(cc, "_session_factory", _factory)
    return holder


# ---------------------------------------------------------------------------
# 200 round trip + (c) acceptance pin (usage.prompt_tokens parity).
# ---------------------------------------------------------------------------


def test_chat_completions_round_trip_with_stub_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    holder = _install_stub_session(
        monkeypatch,
        reply="Paris.",
        prompt_tokens=12,
        output_tokens=2,
        finish_reason="stop_token",
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Capital of France?"},
                ],
                "max_tokens": 16,
                "temperature": 0.7,
            },
        )

    assert response.status_code == 200
    data = response.json()

    # Response shape pins.
    assert data["object"] == "chat.completion"
    assert data["model"] == "stub/model"
    assert data["id"].startswith("chatcmpl-")
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"] == {"role": "assistant", "content": "Paris."}
    assert choice["finish_reason"] == "stop"  # stop_token -> stop

    # (c) acceptance: usage.prompt_tokens matches TurnMetrics.prompt_tokens.
    assert data["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 2,
        "total_tokens": 14,
    }

    # Session received the right user_text + system + history split.
    session = holder["session"]
    assert session is not None
    assert session.system_prompt == "You are helpful."
    assert session.history == []
    assert session.calls[0]["user_text"] == "Capital of France?"


# ---------------------------------------------------------------------------
# Message extraction.
# ---------------------------------------------------------------------------


def test_chat_completions_concatenates_multiple_system_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    holder = _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [
                    {"role": "system", "content": "rule one"},
                    {"role": "system", "content": "rule two"},
                    {"role": "user", "content": "go"},
                ],
            },
        )

    session = holder["session"]
    assert session.system_prompt == "rule one\n\nrule two"


def test_chat_completions_extracts_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    holder = _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [
                    {"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "second"},
                ],
            },
        )

    session = holder["session"]
    assert session.system_prompt == "be helpful"
    assert session.history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
    ]
    assert session.calls[0]["user_text"] == "second"


def test_chat_completions_no_system_returns_none_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    holder = _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert holder["session"].system_prompt is None


# ---------------------------------------------------------------------------
# Sampling-params mapping.
# ---------------------------------------------------------------------------


def test_chat_completions_propagates_sampling_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    holder = _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "temperature": 0.5,
                "top_p": 0.95,
                "seed": 42,
                "max_tokens": 128,
            },
        )

    params = holder["session"].calls[0]["sampling_params"]
    assert params.temperature == 0.5
    assert params.top_p == 0.95
    assert params.seed == 42
    # 'stop' string-sequences are rejected upstream by
    # _validate_unsupported; the route always emits the empty tuple
    # here. Tokenizer EOS still flows through stop_token_ids.
    assert params.stop == ()
    assert params.max_tokens == 128


def test_chat_completions_max_completion_tokens_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When only ``max_completion_tokens`` is set, it is honoured."""
    _configure_runtime()
    holder = _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "max_completion_tokens": 64,
            },
        )

    params = holder["session"].calls[0]["sampling_params"]
    assert params.max_tokens == 64


def test_chat_completions_default_max_tokens_when_neither_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    holder = _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
            },
        )

    params = holder["session"].calls[0]["sampling_params"]
    assert params.max_tokens == 256  # silica default


@pytest.mark.parametrize(
    "stop_value",
    ["END", ["END", "STOP"]],
    ids=["stop_str", "stop_list"],
)
def test_chat_completions_501_on_string_sequence_stop(
    monkeypatch: pytest.MonkeyPatch,
    stop_value: Any,
) -> None:
    """``Engine.generate`` v0.1 only honours ``stop_token_ids``;
    accepting a string-sequence ``stop`` would be a silent contract
    violation (generation continues past the requested terminator).
    Reject it explicitly until P-2 wires text-mode stop trimming.
    """
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "stop": stop_value,
            },
        )

    assert response.status_code == 501
    assert "string-sequence 'stop'" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Sampling-param bounds (route-level 400 to mirror OpenAI semantics).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, bad_value, expected_substring",
    [
        ("temperature", -0.1, "temperature=-0.1"),
        ("temperature", 2.5, "temperature=2.5"),
        ("top_p", 0.0, "top_p=0.0"),
        ("top_p", 1.5, "top_p=1.5"),
        ("max_tokens", 0, "max_tokens=0"),
        ("max_tokens", -1, "max_tokens=-1"),
        ("max_completion_tokens", 0, "max_completion_tokens=0"),
        ("max_completion_tokens", -5, "max_completion_tokens=-5"),
    ],
)
def test_chat_completions_400_on_out_of_range_sampling_param(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: Any,
    expected_substring: str,
) -> None:
    """Out-of-range sampling parameters return a clean 400 from the
    route, not a 500 from the SamplingParams pydantic ``Field`` check
    inside the handler."""
    _configure_runtime()
    _install_stub_session(monkeypatch)

    payload: dict[str, Any] = {
        "model": "stub/model",
        "messages": [{"role": "user", "content": "go"}],
        field: bad_value,
    }

    with TestClient(openai_api.app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400, response.json()
    assert expected_substring in response.json()["detail"]


# ---------------------------------------------------------------------------
# Extension envelope rejection (parsed strictly by (b); 501 in (c)
# until (f) wires the honouring sites).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extension_payload, expected_substring",
    [
        ({"thinking_mode": "off"}, "thinking_mode"),
        ({"continue_truncated": True}, "continue_truncated"),
        (
            {"session_id": "s", "thinking_mode": "auto"},
            "thinking_mode",
        ),
    ],
)
def test_chat_completions_501_on_non_empty_extension(
    monkeypatch: pytest.MonkeyPatch,
    extension_payload: dict[str, Any],
    expected_substring: str,
) -> None:
    """Accepting non-empty extension would silently fall back to default
    behaviour, contradicting what the field claims to do. After (f)
    ``session_id`` is honoured; ``thinking_mode`` and
    ``continue_truncated`` remain unimplemented and return 501.
    """
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "extension": extension_payload,
            },
        )

    assert response.status_code == 501
    assert expected_substring in response.json()["detail"]


def test_chat_completions_accepts_empty_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``extension`` envelope with no fields set is a no-op and
    must NOT trigger the 501 path."""
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "extension": {},
            },
        )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Conversation ordering: out-of-order system messages.
# ---------------------------------------------------------------------------


def test_chat_completions_400_on_non_leading_system_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``system`` message after a user / assistant turn would be
    silently hoisted to the front by the chat template; that is a
    semantic change (the conversation order shifts) and must be
    surfaced as a 400."""
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "system", "content": "rule"},
                    {"role": "user", "content": "second"},
                ],
            },
        )

    assert response.status_code == 400
    assert "system messages must precede" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Model-id matching.
# ---------------------------------------------------------------------------


def test_chat_completions_404_on_model_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-process server serves exactly one model; a request for
    a different model returns 404 rather than echoing the wrong
    model id."""
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "go"}],
            },
        )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "gpt-4o" in detail
    assert "stub/model" in detail


# ---------------------------------------------------------------------------
# Finish-reason mapping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "silica_reason, expected_oai",
    [
        ("stop_token", "stop"),
        ("max_tokens", "length"),
        ("empty", "stop"),
    ],
)
def test_chat_completions_finish_reason_map(
    monkeypatch: pytest.MonkeyPatch,
    silica_reason: str,
    expected_oai: str,
) -> None:
    _configure_runtime()
    _install_stub_session(monkeypatch, finish_reason=silica_reason)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
            },
        )

    assert response.json()["choices"][0]["finish_reason"] == expected_oai


# ---------------------------------------------------------------------------
# 400 / 501 reject matrix.
# ---------------------------------------------------------------------------


def test_chat_completions_rejects_empty_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "stub/model", "messages": []},
        )

    assert response.status_code == 400
    assert "must not be empty" in response.json()["detail"]


def test_chat_completions_rejects_non_user_last_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        )

    assert response.status_code == 400
    assert "role='user'" in response.json()["detail"]


def test_chat_completions_rejects_max_tokens_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 32,
                "max_completion_tokens": 64,
            },
        )

    assert response.status_code == 400
    assert "max_tokens" in response.json()["detail"]


def test_chat_completions_accepts_matching_max_tokens_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal values for both fields are not a conflict."""
    _configure_runtime()
    holder = _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 64,
                "max_completion_tokens": 64,
            },
        )

    assert response.status_code == 200
    assert holder["session"].calls[0]["sampling_params"].max_tokens == 64


@pytest.mark.parametrize(
    "extra_field, expected_substring",
    [
        # ``stream=True`` is no longer 501 — sub-unit (d) lifted that
        # gate; the SSE path is exercised in
        # tests/test_server_chat_completions_streaming.py.
        ({"n": 2}, "n=2 not supported"),
        ({"tools": [{"type": "function"}]}, "tool calling"),
        ({"tool_choice": "auto"}, "tool_choice"),
        ({"logprobs": True}, "logprobs"),
        ({"top_logprobs": 5}, "top_logprobs"),
        ({"logit_bias": {"1": 5}}, "logit_bias"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"frequency_penalty": -0.5}, "frequency_penalty"),
        (
            {"response_format": {"type": "json_object"}},
            "response_format",
        ),
    ],
)
def test_chat_completions_501_unsupported_fields(
    monkeypatch: pytest.MonkeyPatch,
    extra_field: dict[str, Any],
    expected_substring: str,
) -> None:
    _configure_runtime()
    _install_stub_session(monkeypatch)

    payload: dict[str, Any] = {
        "model": "stub/model",
        "messages": [{"role": "user", "content": "go"}],
    }
    payload.update(extra_field)

    with TestClient(openai_api.app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 501
    assert expected_substring in response.json()["detail"]


def test_chat_completions_501_multimodal_user_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi"}],
                    }
                ],
            },
        )

    assert response.status_code == 501
    assert "multimodal" in response.json()["detail"]


def test_chat_completions_501_developer_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [
                    {"role": "developer", "content": "rule"},
                    {"role": "user", "content": "go"},
                ],
            },
        )

    assert response.status_code == 501
    assert "role='developer'" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Runtime readiness.
# ---------------------------------------------------------------------------


def test_chat_completions_503_when_runtime_closed_mid_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _configure_runtime()
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        # Sanity: ok before close.
        assert (
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub/model",
                    "messages": [{"role": "user", "content": "go"}],
                },
            ).status_code
            == 200
        )
        runtime.close()
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "go"}],
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "engine not ready"}
