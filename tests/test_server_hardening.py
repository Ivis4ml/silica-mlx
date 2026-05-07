"""Tests for P-8 sub-unit (h) hardening: auth + rate-limit + error envelope.

Pins:

- Bearer-token auth: SILICA_API_KEY-equivalent ServerConfig.api_key
  rejects missing / malformed / wrong tokens with the OpenAI 401
  envelope; ``/healthz`` is exempt.
- Rate limit: per-key token-bucket; configured RPM cap enforces 429
  when exhausted, ``Retry-After`` header carries the refill delay.
  Disabled when rpm is None / 0.
- Error envelope: every HTTPException raised by the route layer
  surfaces as ``{"error": {"message": ..., "type": ...}}`` matching
  OpenAI's wire shape; pydantic validation errors land in 422 with
  the same envelope.
- Structured-output slot: ``response_format.type='json_schema'`` is
  rejected as 501 with the OpenAI envelope; the request's schema
  name is logged for future grammar-engine analysis.
"""

from __future__ import annotations

import time
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
from silica.server.ratelimit import _reset_limiters  # noqa: E402
from silica.server.routes import chat_completions as cc  # noqa: E402
from silica.server.runtime import Runtime  # noqa: E402


class _StubSession:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def chat(
        self,
        user_text: str,
        *,
        sampling_params: Any = None,
        stream_to: Any = None,
    ) -> TurnMetrics:
        self.calls.append(user_text)
        return TurnMetrics(
            reply="ok",
            prompt_tokens=1,
            output_tokens=1,
            finish_reason="stop_token",
        )


def _build_runtime() -> Runtime:
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
    """Reset module-level config + rate-limit buckets between tests."""
    openai_api._config = None
    _reset_limiters()
    yield
    openai_api._config = None
    _reset_limiters()


def _install_stub_session(monkeypatch: pytest.MonkeyPatch) -> _StubSession:
    session = _StubSession()
    monkeypatch.setattr(
        cc,
        "_session_factory",
        lambda runtime, *, system_prompt, history: session,
    )
    return session


def _basic_chat_payload() -> dict[str, Any]:
    return {
        "model": "stub/model",
        "messages": [{"role": "user", "content": "hi"}],
    }


# ---------------------------------------------------------------------------
# Auth.
# ---------------------------------------------------------------------------


def _configure_with_auth(api_key: str) -> Runtime:
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(
            runtime_factory=lambda: runtime,
            api_key=api_key,
        )
    )
    return runtime


def test_auth_disabled_by_default_no_credentials_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1 default is no auth — the (a)–(g) tests assume this. We
    re-pin it here so a future config-default change can't silently
    break the dev path."""
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions", json=_basic_chat_payload()
        )
    assert response.status_code == 200


def test_auth_401_when_authorization_header_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_with_auth("secret-1")
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions", json=_basic_chat_payload()
        )
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_api_key"
    assert "missing" in body["error"]["message"].lower()


def test_auth_401_when_authorization_header_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Bearer schemes (Basic / Digest / raw token) are rejected
    so a future per-scheme branch cannot silently fall back to a
    less-strict path."""
    _configure_with_auth("secret-1")
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
    assert response.status_code == 401
    assert "Bearer" in response.json()["error"]["message"]


def test_auth_401_when_token_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_with_auth("secret-1")
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert response.status_code == 401


def test_auth_200_when_token_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_with_auth("secret-1")
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer secret-1"},
        )
    assert response.status_code == 200


def test_auth_healthz_exempt_from_authentication() -> None:
    """Liveness probes must not need credentials."""
    _configure_with_auth("secret-1")
    with TestClient(openai_api.app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate limit.
# ---------------------------------------------------------------------------


def _configure_with_rpm(rpm: int) -> Runtime:
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(
            runtime_factory=lambda: runtime,
            rate_limit_rpm=rpm,
        )
    )
    return runtime


def test_rate_limit_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rpm=None pass-through is the v0.1 default."""
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        for _ in range(5):
            response = client.post(
                "/v1/chat/completions", json=_basic_chat_payload()
            )
            assert response.status_code == 200


def test_rate_limit_429_after_exhausting_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bucket capacity == rpm. RPM=2 → 2 successful requests, then
    429 until the next refill."""
    _configure_with_rpm(2)
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        ok1 = client.post("/v1/chat/completions", json=_basic_chat_payload())
        ok2 = client.post("/v1/chat/completions", json=_basic_chat_payload())
        rejected = client.post(
            "/v1/chat/completions", json=_basic_chat_payload()
        )

    assert ok1.status_code == 200
    assert ok2.status_code == 200
    assert rejected.status_code == 429
    body = rejected.json()
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["code"] == "rate_limit_exceeded"
    # Retry-After is the refill delay in seconds.
    assert "Retry-After" in rejected.headers
    assert int(rejected.headers["Retry-After"]) >= 1


def test_rate_limit_unauthenticated_traffic_shares_per_ip_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When auth is disabled, the rate limiter MUST key on the
    client IP — not on the ``Authorization`` header. An attacker
    could otherwise rotate header values to bypass the cap.

    This is the (h) follow-up fix: with auth disabled, the
    Authorization header is untrusted input and must not influence
    bucket selection. The same IP exhausting its bucket triggers
    429 regardless of header rotation.
    """
    _configure_with_rpm(1)
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        first = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer rotating-1"},
        )
        # Different header value, same client IP. With the bug the
        # second request would get a fresh bucket; the fix makes it
        # share the per-IP bucket and trip 429.
        second = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer rotating-2"},
        )

    assert first.status_code == 200
    assert second.status_code == 429


def test_rate_limit_blocks_token_rotation_attack_under_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The (h) follow-up motivating case: with auth ENABLED, an
    attacker rotating wrong bearer tokens (Bearer wrong-1, Bearer
    wrong-2, ...) must eventually trip the per-IP rate limit and
    get 429, not unlimited 401s.

    Without the fix, the rate limiter keyed each wrong token to a
    fresh bucket and never charged the attacker's IP — so
    rate-limit-before-auth ordering reduced to per-token-spam over
    401s.
    """
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(
            runtime_factory=lambda: runtime,
            api_key="real-secret",
            rate_limit_rpm=2,
        )
    )
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        # Two wrong-token requests fit in the bucket — both 401.
        first = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer wrong-1"},
        )
        second = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer wrong-2"},
        )
        # Third rotates a fresh wrong token but the per-IP bucket
        # is empty → 429, not 401.
        third = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer wrong-3"},
        )

    assert first.status_code == 401
    assert second.status_code == 401
    assert third.status_code == 429


def test_rate_limit_valid_token_isolated_from_bad_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validated callers share the auth-keyed bucket; bad traffic
    on the same IP does not starve them. We exhaust the per-IP
    bucket with wrong tokens, then a request with the correct
    token still succeeds because it lands on a separate bucket."""
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(
            runtime_factory=lambda: runtime,
            api_key="real-secret",
            rate_limit_rpm=1,
        )
    )
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        # Burn the per-IP bucket with one wrong-token 401.
        bad = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer wrong"},
        )
        # Validated request: lands on the auth-keyed bucket and
        # succeeds even though the IP bucket is now empty.
        good = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer real-secret"},
        )

    assert bad.status_code == 401
    assert good.status_code == 200


def test_rate_limit_healthz_exempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Liveness probes must not consume bucket budget."""
    _configure_with_rpm(1)
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        # Hit /healthz many times — never decrements the chat bucket.
        for _ in range(10):
            response = client.get("/healthz")
            assert response.status_code == 200
        # Chat bucket still has its first token.
        response = client.post(
            "/v1/chat/completions", json=_basic_chat_payload()
        )
        assert response.status_code == 200


def test_rate_limit_runs_before_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attacker spamming wrong API keys must hit the per-IP
    bucket, not get cheap 401s. We configure auth + rate-limit
    together; with rpm=1, the second wrong-key request returns 429,
    not 401."""
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(
            runtime_factory=lambda: runtime,
            api_key="secret-1",
            rate_limit_rpm=1,
        )
    )
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        # First wrong-key request: rate limit allows, auth rejects.
        first = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer wrong"},
        )
        # Second wrong-key request: rate limit hits FIRST → 429,
        # auth never runs. Without rate-limit-outer-of-auth, this
        # would be a 401 and the attacker would get unbounded
        # validation cost.
        second = client.post(
            "/v1/chat/completions",
            json=_basic_chat_payload(),
            headers={"Authorization": "Bearer wrong"},
        )

    assert first.status_code == 401
    assert second.status_code == 429


def test_rate_limit_refills_over_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the refill window passes, the next request succeeds.

    We use rpm=120 (one token every 0.5s) and sleep slightly over a
    refill window so the cap-1 bucket goes from empty to one token
    available. Time-sensitive test; if CI is wildly slow the sleep
    can be adjusted, but 0.6s is comfortable for a 0.5s refill.
    """
    _configure_with_rpm(120)
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        # Drain the bucket: capacity=120; we hit it 120 times.
        for _ in range(120):
            response = client.post(
                "/v1/chat/completions", json=_basic_chat_payload()
            )
            assert response.status_code == 200
        # Next request: 429 (bucket empty).
        rejected = client.post(
            "/v1/chat/completions", json=_basic_chat_payload()
        )
        assert rejected.status_code == 429
        # Sleep to let one token refill; 60/120 = 0.5s minimum.
        time.sleep(0.6)
        accepted = client.post(
            "/v1/chat/completions", json=_basic_chat_payload()
        )
        assert accepted.status_code == 200


# ---------------------------------------------------------------------------
# Error envelope.
# ---------------------------------------------------------------------------


def test_error_envelope_404_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong/model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 404
    body = response.json()
    assert body == {
        "error": {
            "message": (
                "model='wrong/model' is not loaded; this server "
                "serves 'stub/model'"
            ),
            "type": "invalid_request_error",
        }
    }


def test_error_envelope_501_string_sequence_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["END"],
            },
        )
    assert response.status_code == 501
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "string-sequence 'stop'" in body["error"]["message"]


def test_error_envelope_400_temperature_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    _install_stub_session(monkeypatch)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub/model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 3.0,  # > 2 is OOR
            },
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"


def test_error_envelope_422_pydantic_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request that fails schema parse (missing required field)
    surfaces as 422 with the OpenAI envelope, not FastAPI's default
    structured ``detail`` list."""
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )

    with TestClient(openai_api.app) as client:
        # Missing ``model`` and ``messages`` → 422.
        response = client.post(
            "/v1/chat/completions",
            json={},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# Structured-output slot (response_format).
# ---------------------------------------------------------------------------


def test_response_format_json_schema_returns_501_with_envelope(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The request schema accepts ``response_format``, but v0.1
    rejects non-text types as 501. The slot is wired so the future
    grammar engine has a place to land; we log the request payload
    so the implementer can analyse what schemas downstream callers
    actually need."""
    runtime = _build_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    _install_stub_session(monkeypatch)

    import logging

    with caplog.at_level(logging.INFO, logger="silica.server.routes.chat_completions"):
        with TestClient(openai_api.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "stub/model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Address",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"}
                                },
                            },
                        },
                    },
                },
            )

    assert response.status_code == 501
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "json_schema" in body["error"]["message"]
    # Schema name landed in the log so a future implementer can
    # discover what callers requested.
    log_records = [
        r.getMessage() for r in caplog.records
        if "structured_output.requested" in r.getMessage()
    ]
    assert any("Address" in m for m in log_records), log_records
