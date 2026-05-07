"""silica.server.openai_api — FastAPI app + lifespan + /healthz (P-8 a2).

Module-level :data:`app` is the bare FastAPI instance with the
``/healthz`` route and a lifespan that builds + tears down a
:class:`silica.server.runtime.Runtime`. The lifespan reads a
module-level :data:`_config` set via :func:`configure`. Two intended
call sites:

- **Production** (wired in sub-unit (a3)): :mod:`silica.server.cli`
  parses ``silica serve --model ...`` args, calls ``configure(
  ServerConfig(model_repo="..."))``, then runs uvicorn against
  ``silica.server.openai_api:app``.
- **Tests**: a fixture calls ``configure(ServerConfig(
  runtime_factory=lambda: stub_runtime))`` before driving
  :class:`fastapi.testclient.TestClient`, so no HuggingFace load
  happens in CI.

Design constraints from ``plans/P8_OPENING.md`` §6.1.2 and the (a)
acceptance row:

- Module import must NOT load any model. :func:`configure` also does
  not load — model construction happens on the lifespan startup hook,
  on the worker that drives the app.
- ``/healthz`` is **strict**: returns 200 only AFTER the lifespan
  startup hook has successfully built the runtime. Returns 503 while
  the lifespan has not yet run, after a shutdown, or if the runtime
  has been closed.
- Concurrent HTTP request handlers (sub-units (c)/(d)/(f)) use
  :attr:`Runtime.engine_lock` + :func:`asyncio.to_thread` per the
  Runtime docstring. ``/healthz`` itself never touches MLX so the
  lock+thread contract does not bite here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response

from silica.core.logger import get_logger
from silica.server.runtime import Runtime

log = get_logger(__name__)


@dataclass
class ServerConfig:
    """Configuration consumed by the FastAPI lifespan.

    Exactly one of :attr:`model_repo` / :attr:`runtime_factory` must
    be set:

    - :attr:`model_repo`: production path. The lifespan invokes
      :meth:`Runtime.from_repo` which loads weights via
      :func:`silica.models.factory.adapter_for_repo`.
    - :attr:`runtime_factory`: a zero-argument callable returning a
      pre-built :class:`Runtime`. Tests use this seam to inject a
      stub runtime (for example over :class:`StubModelAdapter` +
      :class:`NullKVManager`) so the smoke path stays off the
      HuggingFace network.

    Hardening knobs (sub-unit (h)):

    - :attr:`api_key`: bearer token enforced on ``/v1/...`` routes.
      ``None`` (default) disables auth — every request is accepted.
      Production deployments set ``SILICA_API_KEY`` env var or pass
      ``--api-key`` to ``silica serve``.
    - :attr:`rate_limit_rpm`: per-key requests-per-minute cap.
      ``None`` (default) or ``<= 0`` disables rate limiting.
    """

    model_repo: str | None = None
    runtime_factory: Callable[[], Runtime] | None = None
    api_key: str | None = None
    rate_limit_rpm: int | None = None

    def __post_init__(self) -> None:
        if (self.model_repo is None) == (self.runtime_factory is None):
            raise ValueError(
                "ServerConfig requires exactly one of model_repo "
                "or runtime_factory to be set"
            )


_config: ServerConfig | None = None


def configure(config: ServerConfig) -> None:
    """Register the server configuration consumed by the lifespan.

    Must be called before the app boots — before
    :class:`fastapi.testclient.TestClient` enters its context, or
    before ``uvicorn.run`` is invoked. Overwrites any previously
    registered config; the lifespan reads :data:`_config` once at
    startup.
    """
    global _config
    _config = config


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _config is None:
        raise RuntimeError(
            "silica.server.openai_api: lifespan started but no "
            "ServerConfig was registered. Call configure(...) before "
            "the app boots; the CLI does this in (a3) and tests must "
            "do it before TestClient enters its context."
        )

    log.info("server.startup begin")
    if _config.runtime_factory is not None:
        runtime = _config.runtime_factory()
    else:
        assert _config.model_repo is not None  # invariant per __post_init__
        runtime = Runtime.from_repo(_config.model_repo)
    app.state.runtime = runtime
    log.info("server.startup ready model=%s", runtime.model_repo)

    try:
        yield
    finally:
        log.info("server.shutdown begin model=%s", runtime.model_repo)
        runtime.close()
        app.state.runtime = None
        log.info("server.shutdown done")


app = FastAPI(
    title="silica-mlx OpenAI-compatible server",
    description=(
        "P-8 v0.1 — local single-user OpenAI HTTP server "
        "(plans/P8_OPENING.md §6.1.2)."
    ),
    lifespan=_lifespan,
)

# OpenAI-shaped error envelope (sub-unit (h)). Installed before the
# routers so any HTTPException raised by the route layer (404s, 501s,
# 503s) and any pydantic validation error during request parsing
# surfaces to the client as ``{"error": {"message": ..., "type":
# ...}}`` instead of FastAPI's default ``{"detail": ...}``.
from silica.server.errors import install_exception_handlers  # noqa: E402

install_exception_handlers(app)

# Route registration: each routes/* module exports an APIRouter that
# gets mounted here. Module imports are cheap (no model load); the
# heavy work happens inside route handlers under the lifespan-built
# runtime.
from silica.server.routes import chat_completions as _chat_completions  # noqa: E402
from silica.server.routes import completions as _completions  # noqa: E402
from silica.server.routes import models as _models  # noqa: E402

app.include_router(_chat_completions.router)
app.include_router(_completions.router)
app.include_router(_models.router)

# Hardening middleware (sub-unit (h)). One inline middleware
# consults :data:`_config` at every dispatch so the test seam
# (``configure`` re-binding between TestClient runs) and a future
# hot-reload both work without re-installing middleware. Order:
# **rate-limit (outer) → auth (inner) → app**. Rate-limit runs
# first so a flood of unauthenticated requests decrements the
# offending IP's bucket — otherwise an attacker could spam
# ``/v1/...`` past the bucket cap and pay only the 401 round trip.
#
# Both gates short-circuit when their config knob is unset
# (``api_key=None`` / ``rate_limit_rpm`` is None or <= 0) so the
# dev-default (no auth, no rate limit) preserves the (a)–(g) test
# surface byte-for-byte.
from silica.server.auth import (  # noqa: E402
    auth_response_for_state,
    preview_bearer_auth,
)
from silica.server.ratelimit import (  # noqa: E402
    consume_rate_limit_token,
)


@app.middleware("http")
async def _hardening_chain(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    cfg = _config
    api_key = cfg.api_key if cfg is not None else None
    rpm = cfg.rate_limit_rpm if cfg is not None else None

    # Pre-check auth so the rate-limit gate can route invalid /
    # missing-token requests to the per-IP bucket (per (h)
    # follow-up: token rotation would otherwise dodge the cap).
    # The rate-limit response is emitted before the auth response
    # so an attacker spamming wrong tokens trips the per-IP 429
    # rather than walking 401s indefinitely.
    auth_state = preview_bearer_auth(request, api_key=api_key)

    rl_response = consume_rate_limit_token(
        request, rpm=rpm, auth_state=auth_state
    )
    if rl_response is not None:
        return rl_response

    auth_response = auth_response_for_state(
        auth_state, request_path=request.url.path
    )
    if auth_response is not None:
        return auth_response

    return await call_next(request)


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    """Strict liveness probe.

    Returns 200 ``{"status": "ok"}`` once the lifespan startup hook
    has successfully built the runtime. Returns 503 if the runtime
    has not been registered (lifespan not yet run) or if it has been
    closed (shutdown in progress or completed).
    """
    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.closed:
        raise HTTPException(status_code=503, detail="engine not ready")
    return {"status": "ok"}
