"""Tests for GET /v1/models (P-8 sub-unit (e)).

Covers the single-entry response shape, ``id`` matching the loaded
``runtime.model_repo``, the ``created`` field reflecting the runtime
construction timestamp, and the 503 path when the runtime has not
been built or has been closed. Mirrors the (a)/(c) test files'
:class:`StubModelAdapter` + :class:`NullKVManager` runtime pattern;
no MLX compute, no HuggingFace load.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi", reason="P-8 [serve] extra not installed")
pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")

from fastapi.testclient import TestClient  # noqa: E402

from silica.core.profiler import MetricsRegistry  # noqa: E402
from silica.kvcache.manager import NullKVManager  # noqa: E402
from silica.models.adapter import StubModelAdapter  # noqa: E402
from silica.server import openai_api  # noqa: E402
from silica.server.runtime import Runtime  # noqa: E402


def _build_stub_runtime(
    *, model_repo: str = "stub/model", created_at: int = 1_700_000_000
) -> Runtime:
    return Runtime(
        StubModelAdapter(),
        NullKVManager(),
        model_repo=model_repo,
        metrics=MetricsRegistry(),
        created_at=created_at,
    )


@pytest.fixture(autouse=True)
def _isolate_module_state() -> Iterator[None]:
    openai_api._config = None
    yield
    openai_api._config = None


def _configure(runtime: Runtime) -> None:
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )


def test_list_models_returns_single_entry_with_loaded_model_id() -> None:
    """The single-model server lists exactly one entry whose ``id``
    matches ``runtime.model_repo``. An OpenAI client can use this id
    as the ``model`` field on a subsequent /v1/chat/completions
    request without translation."""
    runtime = _build_stub_runtime(
        model_repo="Qwen/Qwen3.5-0.8B", created_at=1_715_000_000
    )
    _configure(runtime)

    with TestClient(openai_api.app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1

    entry = body["data"][0]
    assert entry["id"] == "Qwen/Qwen3.5-0.8B"
    assert entry["object"] == "model"
    assert entry["created"] == 1_715_000_000
    assert entry["owned_by"] == "silica"


def test_list_models_response_shape_is_strict() -> None:
    """``ModelsListResponse`` and ``ModelInfo`` are silica-owned with
    ``extra='forbid'``; the wire body must contain exactly the
    documented fields. A regression that added a stray field would
    surface here."""
    runtime = _build_stub_runtime()
    _configure(runtime)

    with TestClient(openai_api.app) as client:
        response = client.get("/v1/models")

    body = response.json()
    assert set(body.keys()) == {"object", "data"}
    assert set(body["data"][0].keys()) == {
        "id",
        "object",
        "created",
        "owned_by",
    }


def test_list_models_503_when_runtime_closed() -> None:
    """After ``runtime.close()`` the readiness gate returns 503,
    matching the /healthz contract from sub-unit (a2) and the
    chat-completions readiness behaviour."""
    runtime = _build_stub_runtime()
    _configure(runtime)

    with TestClient(openai_api.app) as client:
        # Force-close the runtime mid-lifespan to drive the 503 path.
        runtime.close()
        response = client.get("/v1/models")

    assert response.status_code == 503
    assert response.json() == {"error": {"message": "engine not ready", "type": "server_error"}}


def test_list_models_503_when_runtime_unset() -> None:
    """The other half of the readiness contract: when the lifespan
    has not registered a runtime on ``app.state``, the route returns
    the same 503 envelope. Driven by clearing
    ``app.state.runtime`` after TestClient enters its context — the
    fixture's ``_isolate_module_state`` pre-clear targets
    ``openai_api._config`` only, not the live state attribute."""
    runtime = _build_stub_runtime()
    _configure(runtime)

    with TestClient(openai_api.app) as client:
        openai_api.app.state.runtime = None
        response = client.get("/v1/models")

    assert response.status_code == 503
    assert response.json() == {"error": {"message": "engine not ready", "type": "server_error"}}
