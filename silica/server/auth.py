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

The flow is a two-step pattern owned by the inline middleware in
:mod:`silica.server.openai_api`:

1. :func:`preview_bearer_auth` returns an :class:`AuthState`
   indicating whether the request would pass auth, **without**
   producing a response yet. The rate-limit gate consults this
   so it can route unauthenticated requests to the per-IP bucket
   (sub-unit (h) follow-up: the prior implementation keyed by the
   raw ``Authorization`` header value, which let an attacker spam
   wrong tokens with rotating values to dodge the per-IP cap).
2. :func:`auth_response_for_state` produces the 401 envelope for
   non-:attr:`AuthState.VALID` / non-:attr:`AuthState.DISABLED`
   states. The middleware emits this only after the rate-limit
   gate has had a chance to charge the bucket.

Failed auth returns the OpenAI-shaped error envelope
(``{"error": {...}}``) via :mod:`silica.server.errors`.

Multi-tenant / per-user keys are post-announce: a single shared
secret is sufficient for the local-developer / single-user-VPS framing
and avoids pulling in a credential database. (h) hardening only locks
the door; rotation, scoping, and audit logging are deferred.
"""

from __future__ import annotations

import hmac
from enum import Enum

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


class AuthState(str, Enum):
    """Result of the :func:`preview_bearer_auth` pre-check.

    Values:

    - :attr:`DISABLED` — no API key configured (or path is
      auth-exempt). Allow the request without a header check.
    - :attr:`VALID` — request carries a correct bearer token.
    - :attr:`MISSING` — auth is configured but no
      ``Authorization`` header was supplied.
    - :attr:`MALFORMED` — header present but not ``Bearer ...``.
    - :attr:`INVALID` — bearer token does not match the
      configured key.

    The rate-limit gate uses :attr:`VALID` vs everything else to
    pick the bucket key (validated callers share the auth bucket;
    everyone else shares the per-IP bucket).
    """

    DISABLED = "disabled"
    VALID = "valid"
    MISSING = "missing"
    MALFORMED = "malformed"
    INVALID = "invalid"


def preview_bearer_auth(
    request: Request, *, api_key: str | None
) -> AuthState:
    """Pre-check the ``Authorization`` header without producing a response.

    The check uses :func:`hmac.compare_digest` so a timing side
    channel cannot leak the secret. Logging is intentionally
    deferred to :func:`auth_response_for_state` so the middleware
    only logs deny events that actually short-circuit the request
    (a rate-limit-triggered 429 is the last word on the request
    even when auth would also have rejected it; logging both would
    double-count the failure).
    """
    if api_key is None:
        return AuthState.DISABLED
    if request.url.path in _AUTH_EXEMPT_PATHS:
        return AuthState.DISABLED

    header = request.headers.get("authorization")
    if header is None:
        return AuthState.MISSING
    if not header.startswith(_BEARER_PREFIX):
        return AuthState.MALFORMED
    provided = header[len(_BEARER_PREFIX):].strip()
    # Constant-time comparison: a regular ``==`` would short-circuit
    # on the first mismatching byte, leaking the prefix to a remote
    # attacker via response-time variance.
    if not hmac.compare_digest(provided, api_key):
        return AuthState.INVALID
    return AuthState.VALID


def auth_response_for_state(
    state: AuthState, *, request_path: str
) -> JSONResponse | None:
    """Convert an :class:`AuthState` into a 401 :class:`JSONResponse`,
    or ``None`` for states that allow the request through.

    Logs the deny reason at INFO so operators can see ``auth.deny
    reason=missing|malformed|invalid`` in production logs. Body of
    the supplied / configured key is never logged.
    """
    if state in (AuthState.DISABLED, AuthState.VALID):
        return None
    log.info("auth.deny path=%s reason=%s", request_path, state.value)
    if state == AuthState.MISSING:
        return _auth_error_response(
            "missing Authorization header", status_code=401
        )
    if state == AuthState.MALFORMED:
        return _auth_error_response(
            "Authorization header must use the Bearer scheme",
            status_code=401,
        )
    # AuthState.INVALID
    return _auth_error_response("invalid API key", status_code=401)


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
    "AuthState",
    "preview_bearer_auth",
    "auth_response_for_state",
]
