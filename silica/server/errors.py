"""silica.server.errors — OpenAI-shaped error envelope (P-8 sub-unit (h)).

OpenAI's wire format is ``{"error": {"message": "...", "type": "...",
"code": "..."}}``. FastAPI's default :class:`HTTPException` handler
emits ``{"detail": "..."}`` which the openai Python SDK parses as a
generic error, losing the typed taxonomy. This module wires a global
exception handler that wraps every :class:`HTTPException` (and the
Pydantic :class:`RequestValidationError`) into the OpenAI shape.

The taxonomy (``error.type``) maps from HTTP status to the OpenAI
strings the SDK switches on:

- 400 → ``invalid_request_error``
- 401 → ``invalid_request_error`` (auth lives in ``auth.py``)
- 404 → ``invalid_request_error`` (model / endpoint not found)
- 422 → ``invalid_request_error`` (request schema validation)
- 429 → ``rate_limit_error``
- 500 → ``server_error``
- 501 → ``invalid_request_error`` (unsupported feature)
- 503 → ``server_error`` (engine not ready / shutting down)

Other status codes default to ``api_error`` (the OpenAI SDK's
fallback type).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from silica.core.logger import get_logger

log = get_logger(__name__)


# Mapping from HTTP status code → OpenAI ``error.type`` string.
# Status codes outside the table fall back to ``api_error``.
_STATUS_TO_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "invalid_request_error",
    404: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "server_error",
    501: "invalid_request_error",
    503: "server_error",
}


def openai_error_payload(
    *,
    message: str,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    """Build the JSON body for an OpenAI-shaped error.

    The ``param`` field is reserved for v0.2 — the OpenAI SDK uses it
    to localise the offending request field on validation errors. v0.1
    leaves it ``None`` so the wire shape stays minimal but the function
    signature is forward-compatible.
    """
    err: dict[str, Any] = {
        "message": message,
        "type": error_type,
    }
    if code is not None:
        err["code"] = code
    if param is not None:
        err["param"] = param
    return {"error": err}


def _error_type_for_status(status: int) -> str:
    return _STATUS_TO_TYPE.get(status, "api_error")


async def http_exception_handler(
    request: Any, exc: Exception
) -> JSONResponse:
    """Wrap :class:`HTTPException` into the OpenAI envelope.

    Starlette type-erases the handler signature to
    ``Exception``; the runtime ``isinstance`` check narrows for
    mypy and surfaces a programming error if the handler is ever
    re-registered against a different exception class. The route
    layer raises with a string ``detail``; we reuse it as the
    user-facing ``message``. The status code drives the
    ``error.type`` mapping. Headers from the original exception are
    preserved (e.g. ``Retry-After`` on 429s).
    """
    assert isinstance(exc, StarletteHTTPException)
    detail = exc.detail
    # Detail can be a non-string (a dict in odd cases). Coerce to a
    # printable form so the response stays well-typed; reaching this
    # branch is a route-side bug worth surfacing as a stable string.
    message = (
        detail
        if isinstance(detail, str)
        else jsonable_encoder(detail)
    )
    payload = openai_error_payload(
        message=message if isinstance(message, str) else str(message),
        error_type=_error_type_for_status(exc.status_code),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload,
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(
    request: Any, exc: Exception
) -> JSONResponse:
    """Wrap pydantic / FastAPI validation errors into the OpenAI envelope.

    The default FastAPI handler returns ``{"detail": [{"loc": [...],
    "msg": "...", "type": "..."}, ...]}`` — useful for debugging but
    incompatible with the openai Python SDK's parser. We collapse the
    list into a single human-readable message; the full structured
    list remains available via the server log.
    """
    assert isinstance(exc, RequestValidationError)
    errors = exc.errors()
    log.info("request.validation.fail errors=%r", errors)
    if errors:
        first = errors[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "<root>"
        msg = first.get("msg") or "validation error"
        message = f"{loc}: {msg}"
    else:
        message = "request validation failed"
    payload = openai_error_payload(
        message=message,
        error_type="invalid_request_error",
        code="invalid_request",
    )
    return JSONResponse(status_code=422, content=payload)


def install_exception_handlers(app: FastAPI) -> None:
    """Register the OpenAI-shaped handlers on a FastAPI app.

    Called from the app construction site in
    :mod:`silica.server.openai_api` after the routers are mounted.
    Idempotent: re-registering the same handler replaces the prior
    binding rather than chaining handlers.
    """
    app.add_exception_handler(
        StarletteHTTPException, http_exception_handler
    )
    app.add_exception_handler(
        RequestValidationError, validation_exception_handler
    )


__all__ = [
    "install_exception_handlers",
    "openai_error_payload",
]
