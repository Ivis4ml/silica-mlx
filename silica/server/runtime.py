"""silica.server.runtime — Engine-owning wrapper for the HTTP server (P-8).

The :class:`Runtime` instance is built in the FastAPI lifespan startup
hook (see :mod:`silica.server.openai_api`) and held under
``app.state.runtime`` for the duration of the process. Per
``plans/P8_OPENING.md`` §6.1.2 G-1 (Option A endpoint routing), the
runtime serialises concurrent HTTP requests on a single
:class:`silica.engine.Engine` via :attr:`Runtime.engine_lock` — *one
active decode turn at a time*. This keeps v0.1 routing identical to the
chat-CLI's single-customer path; multi-customer scheduler routing
(Options B/C in OPENING §6.1.1) is a post-announce follow-on.

Construction paths
------------------

- :meth:`Runtime.from_repo` — production path; loads a model via
  :func:`silica.models.factory.adapter_for_repo` and constructs the
  :class:`silica.engine.Engine` against it. Called from the lifespan
  startup hook.
- ``Runtime(adapter, kv_manager, model_repo=...)`` — direct constructor
  for tests; pass any I-1 / I-2 stubs (see ``tests/test_engine.py`` for
  the in-tree pattern). Tests that do not exercise model loading should
  use this path to keep CI off the HuggingFace network.

The :meth:`close` seam currently has no work to do — MLX arrays are
released by the GC and the adapter does not own external resources. It
is kept as a contract so future variants (P-6 weight streaming,
KV-codec store handles, background prefetch workers) can release
without changing the lifespan shape.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from silica.core.logger import get_logger
from silica.core.profiler import MetricsRegistry
from silica.engine import Engine
from silica.server.session import (
    DEFAULT_MAX_SESSIONS,
    DEFAULT_PREFIX_CACHE_BLOCK_SIZE,
    DEFAULT_SESSION_TTL_S,
    SessionManager,
)

if TYPE_CHECKING:
    from silica.kvcache.manager import KVManager
    from silica.models.adapter import ModelAdapter

log = get_logger(__name__)


class Runtime:
    """Owns one :class:`Engine` for the duration of a server lifespan.

    Per OPENING §6.1.2 G-1, concurrent HTTP request handlers must
    acquire :attr:`engine_lock` before invoking any Engine method.
    The lock is an :class:`asyncio.Lock` so coroutines can await it
    cooperatively without spinning a real OS thread on contention.

    .. important::

        Inside the critical section, MLX compute MUST run off the
        event loop — wrap it in :func:`asyncio.to_thread` (or a
        custom executor) — otherwise the entire FastAPI app stalls
        during prefill / decode and SSE streaming for other requests
        is starved.

        A subtlety: :meth:`Engine.generate` is a generator function,
        so calling it returns an iterator immediately *before any
        compute runs*. Passing the bound method to
        :func:`asyncio.to_thread` by itself only returns the iterator
        from the worker thread; the actual decode loop still executes
        wherever the iterator is consumed. Either consume the
        iterator inside the thread (non-streaming path) or drive the
        engine through a callback-based facade
        (:meth:`silica.chat.session.ChatSession.chat` with
        ``stream_to=...``) that runs the generator internally and
        fires per-token callbacks.

    Non-streaming pattern (used by sub-unit (e) ``/v1/completions``
    and sub-unit (g) ``silica.llm.LLM.generate``)::

        async with runtime.engine_lock:
            tokens = await asyncio.to_thread(
                lambda: list(runtime.engine.generate(prompt, params)),
            )

    Streaming / chat-session pattern (used by sub-units (c) / (d)
    and sub-unit (f))::

        async with runtime.engine_lock:
            metrics = await asyncio.to_thread(
                session.chat,
                user_text,
                sampling_params=params,
                stream_to=callback,
            )

    In the streaming pattern, ``callback`` is a thread-to-loop
    bridge. :class:`asyncio.Queue` is **not** thread-safe — a worker
    thread must not call :meth:`asyncio.Queue.put_nowait` directly,
    and ``put_nowait`` would in any case drop tokens (or raise
    ``QueueFull``) on a full queue, breaking the OPENING §6.1.2 G-3
    no-drop / backpressure invariant. Two correct shapes:

    - **Non-blocking notification** (no backpressure, fine for
      lightweight signals such as "request was aborted"):
      :meth:`asyncio.AbstractEventLoop.call_soon_threadsafe` to
      schedule a callback on the loop.
    - **G-3 no-drop streaming** (the canonical SSE / chat path):
      run a coroutine on the loop from the worker thread via
      :func:`asyncio.run_coroutine_threadsafe` and *block* the
      thread on its :meth:`concurrent.futures.Future.result` —
      e.g. ``run_coroutine_threadsafe(queue.put(delta),
      loop).result()``. ``queue.put`` awaits when a bounded
      :class:`asyncio.Queue` is full, which blocks the decode
      worker thread on the loop side until the SSE consumer drains
      a slot, propagating slow-client backpressure into MLX
      compute. An equivalent dedicated thread-safe queue bridge is
      acceptable.

    Sub-unit (d) lands the concrete bridge; this docstring fixes
    the contract so (c)/(f) wire it the same way.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        kv_manager: KVManager,
        *,
        model_repo: str,
        metrics: MetricsRegistry | None = None,
        created_at: int | None = None,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        session_ttl_s: float = DEFAULT_SESSION_TTL_S,
        prefix_cache_block_size: int = DEFAULT_PREFIX_CACHE_BLOCK_SIZE,
    ) -> None:
        self._adapter = adapter
        self._kv_manager = kv_manager
        self._model_repo = model_repo
        self._engine = Engine(
            adapter=adapter,
            kv_manager=kv_manager,
            metrics=metrics,
        )
        # Stamped at construction so /v1/models reports a stable
        # creation timestamp for the lifetime of this runtime.
        # Tests inject a fixed value via the keyword.
        self._created_at = (
            created_at if created_at is not None else int(time.time())
        )
        self.engine_lock = asyncio.Lock()
        self._closed = False
        # SessionManager carries the X-Silica-Session-ID → ChatSession
        # map for cross-request prefix reuse (P-8 sub-unit (f)). One
        # manager per Runtime; tunables come from constructor kwargs
        # so tests can size the LRU + TTL down for fast eviction
        # coverage. (h) hardening exposes these as ``silica serve``
        # CLI flags.
        self._session_manager = SessionManager(
            adapter=adapter,
            engine=self._engine,
            max_sessions=max_sessions,
            session_ttl_s=session_ttl_s,
            prefix_cache_block_size=prefix_cache_block_size,
        )

    @classmethod
    def from_repo(cls, model_repo: str) -> Runtime:
        """Build a runtime by loading the model via the adapter factory.

        Production path called from the FastAPI lifespan startup hook.
        The factory import is deferred so the module-import cost stays
        cheap for callers (tests, type checkers) that never load a
        real model.
        """
        from silica.models.factory import adapter_for_repo

        log.info("runtime.load model=%s", model_repo)
        adapter, kv = adapter_for_repo(model_repo)
        return cls(adapter, kv, model_repo=model_repo)

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def adapter(self) -> ModelAdapter:
        return self._adapter

    @property
    def kv_manager(self) -> KVManager:
        return self._kv_manager

    @property
    def model_repo(self) -> str:
        return self._model_repo

    @property
    def created_at(self) -> int:
        """Unix timestamp captured at construction.

        Surfaced by ``GET /v1/models`` as the model entry's ``created``
        field per OpenAI's wire shape. Stable for the lifetime of this
        Runtime; a new Runtime (lifespan restart, ``--reload`` etc.)
        gets a fresh stamp.
        """
        return self._created_at

    @property
    def session_manager(self) -> SessionManager:
        """SessionManager owning the X-Silica-Session-ID → ChatSession map.

        Used by :mod:`silica.server.routes.chat_completions` when a
        request carries a session selector header / extension field.
        Built in :meth:`__init__`; persists for the Runtime lifetime
        and is cleared in :meth:`close`.
        """
        return self._session_manager

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release runtime-owned resources.

        Called from the FastAPI lifespan shutdown hook. Idempotent —
        a second call is silently ignored. Drops every persisted
        ChatSession via :meth:`SessionManager.close` so the
        per-session prefix caches and their underlying stores become
        GC-eligible immediately. Future variants land their cleanup
        here (codec store handles, prefetch worker drain,
        weight-streaming page table free).
        """
        if self._closed:
            return
        self._closed = True
        self._session_manager.close()
        log.info("runtime.close model=%s", self._model_repo)
