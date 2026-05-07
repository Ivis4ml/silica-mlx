"""silica.server.routes.models — GET /v1/models (P-8 sub-unit (e)).

Single-model server: the response always lists exactly one entry,
the runtime's loaded model. The ``id`` echoes ``runtime.model_repo``
verbatim so an OpenAI client can use it as the ``model`` field on a
subsequent ``/v1/chat/completions`` request without translation.

The endpoint is intentionally lightweight — no MLX compute, no
``runtime.engine_lock`` acquisition. The ``Runtime`` readiness gate
(503 when the lifespan has not built it or after shutdown) is the
only liveness check.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from silica.server.runtime import Runtime
from silica.server.schemas import ModelInfo, ModelsListResponse

router = APIRouter()


def _get_runtime(request: Request) -> Runtime:
    runtime: Runtime | None = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.closed:
        raise HTTPException(status_code=503, detail="engine not ready")
    return runtime


@router.get("/v1/models")
async def list_models(request: Request) -> ModelsListResponse:
    """Return the single-entry model list for this server."""
    runtime = _get_runtime(request)
    return ModelsListResponse(
        data=[
            ModelInfo(
                id=runtime.model_repo,
                created=runtime.created_at,
            )
        ]
    )
