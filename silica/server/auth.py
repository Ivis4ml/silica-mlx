"""silica.server.auth — bearer-token authentication (P-8 sub-unit (h)).

Single shared bearer token, sourced from ``SILICA_API_KEY`` env var
or the ``--api-key`` CLI flag. When configured, every request to the
``/v1/...`` routes must carry a matching ``Authorization: Bearer <key>``
header. ``/healthz`` is always public — load balancers and process
supervisors need to probe liveness without auth.

When no API key is configured, auth is disabled and every request is
accepted. This is the development default and matches the v0.1
single-user-on-localhost framing in
``plans/P8_OPENING.md`` §6.1.2 G-1.

The auth check runs as a request-level helper (called from the
inline middleware in :mod:`silica.server.openai_api`) so it applies
uniformly across every route registered on the app — the routes
themselves do not need ``Depends(verify_auth)`` decorations. Failed
auth returns the OpenAI-shaped error envelope (``{"error": {...}}``)
via :mod:`silica.server.errors`.

Multi-tenant / per-user keys are post-announce: a single shared
secret is sufficient for the local-developer / single-user-VPS framing
and avoids pulling in a credential database. (h) hardening only locks
the door; rotation, scoping, and audit logging are deferred.
"""

from __future__ import annotations

import hmac

from starlette.requests import Request
from starlette.responses import JSONResponse

from silica.core.logger import get_logger
from silica.server.errors import openai_error_payload

log = get_logger(__name__)

_BEARER_PREFIX = "Bearer "

_AUTH_EXEMPT_PATHS: frozenset[str] = frozenset({"/healthz"})
"""Routes that bypass auth entirely.

Liveness / readiness probes (LB health checks, k8s readiness)
must not need a credential — otherwise an unauthenticated probe
loop would log a 401 storm. ``/healthz`` is the only such route
in v0.1; future ``/metrics`` / ``/version`` would join this set.
"""


def check_bearer_auth(
    request: Request, *, api_key: str | None
) -> JSONResponse | None:
    """Validate the request's ``Authorization`` header.

    Returns ``None`` to allow the request to proceed; returns a
    :class:`JSONResponse` carrying the OpenAI 401 envelope when the
    request fails the check.

    Behaviour:

    - ``api_key is None`` — auth disabled, always allow.
    - Path is in :data:`_AUTH_EXEMPT_PATHS` — always allow (e.g.
      ``/healthz``).
    - Missing / malformed ``Authorization`` header — 401.
    - Wrong bearer token — 401.

    The check uses :func:`hmac.compare_digest` so a timing side
    channel cannot leak the secret. Logging avoids printing the
    supplied or configured value — only ``status=missing|invalid``
    is recorded.
    """
    if api_key is None:
        return None
    if request.url.path in _AUTH_EXEMPT_PATHS:
        return None

    header = request.headers.get("authorization")
    if header is None:
        log.info("auth.deny path=%s reason=missing", request.url.path)
        return _auth_error_response(
            "missing Authorization header",
            status_code=401,
        )
    if not header.startswith(_BEARER_PREFIX):
        log.info(
            "auth.deny path=%s reason=malformed", request.url.path
        )
        return _auth_error_response(
            "Authorization header must use the Bearer scheme",
            status_code=401,
        )
    provided = header[len(_BEARER_PREFIX):].strip()
    # Constant-time comparison: a regular ``==`` would short-circuit
    # on the first mismatching byte, leaking the prefix to a remote
    # attacker via response-time variance.
    if not hmac.compare_digest(provided, api_key):
        log.info("auth.deny path=%s reason=invalid", request.url.path)
        return _auth_error_response(
            "invalid API key",
            status_code=401,
        )
    return None


def _auth_error_response(message: str, *, status_code: int) -> JSONResponse:
    """Format an OpenAI-shaped 401 envelope.

    Lives upstream of the FastAPI exception handlers registered in
    :mod:`silica.server.errors`, so we shape the envelope inline
    here using :func:`openai_error_payload` to keep the wire format
    consistent with the rest of the route surface.
    """
    return JSONResponse(
        status_code=status_code,
        content=openai_error_payload(
            message=message,
            error_type="invalid_request_error",
            code="invalid_api_key",
        ),
    )


__all__ = [
    "check_bearer_auth",
]
