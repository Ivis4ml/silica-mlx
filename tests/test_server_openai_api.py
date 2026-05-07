"""Tests for :mod:`silica.server.openai_api` (P-8 sub-unit (a2)).

Pins the FastAPI app + lifespan + ``/healthz`` contract introduced at
sub-unit (a2). Uses the :attr:`ServerConfig.runtime_factory` DI seam to
inject a stub :class:`Runtime` so no HuggingFace load is triggered.

Test surface:

- :class:`ServerConfig` rejects 0 or 2 seams (exactly one of
  ``model_repo`` / ``runtime_factory`` is required).
- The lifespan builds a runtime via the factory seam, assigns
  ``app.state.runtime``, and closes the runtime on shutdown.
- ``/healthz`` returns 200 ``{"status": "ok"}`` while the lifespan is
  active.
- Without a registered config, the lifespan refuses to start (no
  silent fall-back to a half-baked runtime).
- ``/healthz`` flips to 503 mid-lifespan if the runtime is closed.
"""

from __future__ import annotations

import pytest

# The (a2) tests require the [serve] optional-dependencies extra. If
# fastapi / httpx are missing, skip the whole module cleanly rather
# than collecting an import error.
pytest.importorskip("fastapi", reason="P-8 [serve] extra not installed")
pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")

from collections.abc import Iterator  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from silica.core.profiler import MetricsRegistry  # noqa: E402
from silica.kvcache.manager import NullKVManager  # noqa: E402
from silica.models.adapter import StubModelAdapter  # noqa: E402
from silica.server import openai_api  # noqa: E402
from silica.server.runtime import Runtime  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_module_config() -> Iterator[None]:
    """Each test starts and ends with no registered configuration.

    The module-level ``_config`` slot is mutated by ``configure``;
    leaking state across tests would let the order of execution
    determine the lifespan path. The fixture wipes both before and
    after so failures do not poison subsequent tests.
    """
    openai_api._config = None
    yield
    openai_api._config = None


def _build_stub_runtime() -> Runtime:
    return Runtime(
        StubModelAdapter(),
        NullKVManager(),
        model_repo="stub/model",
        metrics=MetricsRegistry(),
    )


def test_server_config_requires_exactly_one_seam() -> None:
    with pytest.raises(ValueError, match="exactly one of model_repo"):
        openai_api.ServerConfig()
    with pytest.raises(ValueError, match="exactly one of model_repo"):
        openai_api.ServerConfig(
            model_repo="foo",
            runtime_factory=_build_stub_runtime,
        )


def test_lifespan_starts_runtime_via_factory_and_healthz_returns_ok() -> None:
    runtime = _build_stub_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )

    with TestClient(openai_api.app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert openai_api.app.state.runtime is runtime
        assert runtime.closed is False

    # Lifespan shutdown ran on context exit.
    assert runtime.closed is True


def test_lifespan_without_configure_raises_at_startup() -> None:
    with pytest.raises(RuntimeError, match="ServerConfig was registered"):
        with TestClient(openai_api.app):
            pass


def test_healthz_returns_503_when_runtime_closed_mid_lifespan() -> None:
    runtime = _build_stub_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )

    with TestClient(openai_api.app) as client:
        assert client.get("/healthz").status_code == 200
        runtime.close()
        response = client.get("/healthz")
        assert response.status_code == 503
        assert response.json() == {"detail": "engine not ready"}
