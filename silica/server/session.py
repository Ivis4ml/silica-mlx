"""silica.server.session — SessionManager (P-8 sub-unit (f)).

Holds the ``session_id → ChatSession`` map that backs cross-request
prefix reuse. Per ``plans/P8_OPENING.md`` §6.1.2 G-2, each persisted
:class:`silica.chat.session.ChatSession` carries its own
:class:`silica.kvcache.prefix.RadixPrefixCache` so reuse is isolated
to one conversation; cross-session shared-system-prompt reuse is a
post-announce follow-on.

Concurrency contract (v0.1, G-1 single-active-decode)
=====================================================

The route resolves the session **inside** ``runtime.engine_lock`` —
:meth:`get_or_create` is invoked from the lock-held branch in
:mod:`silica.server.routes.chat_completions`. That gives two
properties for free:

1. **No torn writes on shared session_ids.** Two concurrent requests
   that name the same session both wait for the lock; the second
   only calls :meth:`get_or_create` after the first request's
   ``ChatSession.chat`` has returned. Without this the second
   request's ``replace_messages`` call could clobber the first
   request's history before the first turn ran.
2. **No eviction-during-use.** ``_evict_idle`` and ``_evict_lru``
   run from inside :meth:`get_or_create` (no other call site under
   v0.1), so eviction is gated on the same lock as the request
   that created the entry. A future maintenance hook (periodic
   eviction task, ``/v1/sessions`` admin endpoint) would need an
   explicit in-flight refcount to preserve this invariant; that is
   deferred to (h) hardening.

Lifecycle defaults
==================

- ``max_sessions = 64`` — LRU cap. The OpenAI ``user`` field
  cannot collide with this map (we ignore ``user`` per OPENING
  §4.3), so the only way to grow the map is the ``X-Silica-
  Session-ID`` header / ``extension.session_id`` body field.
- ``session_ttl_s = 1800`` — 30 minutes idle. Tracks
  ``time.monotonic`` so wall-clock skews / DST transitions cannot
  evict an active session.
- ``prefix_cache_block_size = 4`` — mirrors
  :mod:`silica.chat.cli.app` (``_PREFIX_CACHE_BLOCK_SIZE``). The
  chat-CLI bench showed 4 is the right balance between hit
  granularity and radix-tree node count for chat workloads.

These defaults are tunable via constructor kwargs; (h) hardening
exposes them as ``silica serve`` CLI flags.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from silica.chat.session import ChatSession
from silica.core.logger import get_logger
from silica.kvcache.prefix import PrefixCacheStats, RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.adapter import AttentionKind

log = get_logger(__name__)


DEFAULT_PREFIX_CACHE_BLOCK_SIZE = 4
"""Mirrors :mod:`silica.chat.cli.app` ``_PREFIX_CACHE_BLOCK_SIZE``.
One source of truth for v0.1; promote to a config knob in (h)."""

DEFAULT_MAX_SESSIONS = 64
"""LRU cap from OPENING §4.2."""

DEFAULT_SESSION_TTL_S = 30 * 60.0
"""Idle TTL from OPENING §4.2."""


@dataclass
class _SessionEntry:
    """Per-session-id record. Internal — tests reach in via
    :meth:`SessionManager.get` for assertions but should not
    construct these directly."""

    session_id: str
    chat_session: ChatSession
    prefix_cache: RadixPrefixCache
    last_access_s: float


class SessionManager:
    """Owns the session_id → ChatSession map for the server lifetime.

    See module docstring for the concurrency contract. The class is
    not thread-safe in isolation; relies on the route holding
    ``runtime.engine_lock`` while calling :meth:`get_or_create`.

    The ``adapter`` and ``engine`` arguments are typed as ``Any``
    because the manager just plumbs them into :class:`ChatSession`,
    which has its own Protocol-level constraints; importing
    ``ModelAdapter`` / ``Engine`` here would create an unnecessary
    runtime dependency on those modules at server-import time.
    """

    def __init__(
        self,
        *,
        adapter: Any,
        engine: Any,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        session_ttl_s: float = DEFAULT_SESSION_TTL_S,
        prefix_cache_block_size: int = DEFAULT_PREFIX_CACHE_BLOCK_SIZE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions <= 0:
            raise ValueError(
                f"max_sessions must be > 0, got {max_sessions}"
            )
        if session_ttl_s <= 0:
            raise ValueError(
                f"session_ttl_s must be > 0, got {session_ttl_s}"
            )
        if prefix_cache_block_size <= 0:
            raise ValueError(
                f"prefix_cache_block_size must be > 0, got "
                f"{prefix_cache_block_size}"
            )
        self._adapter = adapter
        self._engine = engine
        self._max_sessions = max_sessions
        self._ttl_s = session_ttl_s
        self._block_size = prefix_cache_block_size
        self._clock = clock
        # Insertion order = recency. Python dicts preserve insertion
        # order, so eviction iterates ``self._sessions`` from oldest
        # access; touching an entry pops + reinserts to push it to
        # the tail.
        self._sessions: dict[str, _SessionEntry] = {}
        # Adapters whose attention pattern includes
        # :attr:`AttentionKind.SLIDING` cannot be driven through
        # :meth:`Engine.generate_batch` with a non-None
        # ``RadixPrefixCache`` — :class:`silica.scheduler.batcher.ContinuousBatcher`
        # rejects that combination at line 292 because the
        # :class:`BatchRotatingKVCache` window-truncation / offset /
        # rotated-state semantics under the seeded admission path
        # have not been validated (P-3-D3 follow-up). Persistent
        # sessions in v0.1 always carry a :class:`RadixPrefixCache`,
        # so a SLIDING-bearing adapter cannot host them and the
        # route must surface a 501 instead of attempting to drive
        # ``generate_batch`` with an incompatible cache.
        self._supports_prefix_reuse = (
            AttentionKind.SLIDING
            not in adapter.capabilities().attention_kinds
        )

    # --- primary surface ------------------------------------------------

    @property
    def supports_prefix_reuse(self) -> bool:
        """Whether the bound adapter is compatible with persistent
        sessions backed by :class:`RadixPrefixCache`.

        ``False`` for adapters whose attention pattern includes
        :attr:`AttentionKind.SLIDING` (Gemma4 31B today). The route
        consults this before invoking :meth:`get_or_create` and
        returns 501 when a request names a ``session_id`` against a
        non-supporting adapter — silently degrading would let the
        client believe their conversation is being persisted while
        :class:`ContinuousBatcher` raises ``NotImplementedError``
        deeper in the stack.

        Lifting this restriction is post-P-8: the
        sliding-window-aware prefix-cache seed path is tracked as
        a P-3-D3 local follow-up.
        """
        return self._supports_prefix_reuse

    def get_or_create(
        self,
        session_id: str,
        *,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ChatSession:
        """Return the persisted ChatSession for ``session_id``.

        On a cache miss, builds a fresh :class:`ChatSession` carrying
        a freshly-constructed :class:`RadixPrefixCache`. On a hit,
        reuses the existing session and prefix cache so the next
        ``ChatSession.chat`` call's ``peek`` lands hits accumulated
        from prior turns.

        Raises :class:`RuntimeError` if the bound adapter does not
        support persistent prefix reuse (see
        :attr:`supports_prefix_reuse`); the route validates this
        upstream and returns 501 instead, so reaching this branch
        from production code is a programming error.

        ``system_prompt`` + ``history`` overwrite the session's
        message log via :meth:`ChatSession.replace_messages`. This
        is **option α** from OPENING §4.2 design notes: the OpenAI
        client sends the full conversation each request and is
        authoritative for message state; the prefix cache survives
        the message replace because
        :meth:`ChatSession.replace_messages` only resets
        ``_messages`` and pending-finalise flags, never
        ``_prefix_cache``.

        Eviction sweeps run inline before allocation:

        - :meth:`_evict_idle` drops entries whose ``last_access_s``
          is older than ``session_ttl_s``.
        - :meth:`_evict_lru` drops the oldest entries until the
          map size respects ``max_sessions`` after insertion.
        """
        if not self._supports_prefix_reuse:
            raise RuntimeError(
                "SessionManager.get_or_create called for an adapter "
                "whose attention_kinds include SLIDING. Persistent "
                "sessions in v0.1 require a RadixPrefixCache, which "
                "ContinuousBatcher does not support against "
                "sliding-window adapters; the route must check "
                "supports_prefix_reuse and return 501 before "
                "reaching this method."
            )
        now = self._clock()
        self._evict_idle_at(now)

        entry = self._sessions.pop(session_id, None)
        if entry is None:
            entry = self._build_entry(session_id, now=now)
            log.info("session.create id=%s", session_id)
        else:
            entry.last_access_s = now

        self._sessions[session_id] = entry
        self._evict_lru_to_cap()

        full: list[dict[str, str]] = []
        if system_prompt is not None:
            full.append({"role": "system", "content": system_prompt})
        if history:
            full.extend(history)
        entry.chat_session.replace_messages(full)
        return entry.chat_session

    def get(self, session_id: str) -> ChatSession | None:
        """Return the persisted ChatSession without creating one.

        Used by tests / introspection to inspect a session's
        prefix cache after a request. Does NOT update LRU recency
        or check TTL — read-only peek.
        """
        entry = self._sessions.get(session_id)
        return entry.chat_session if entry else None

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    # --- maintenance ----------------------------------------------------

    def evict_idle(self) -> int:
        """Drop entries idle longer than ``session_ttl_s``.

        Returns the number of entries dropped. Public surface for
        future maintenance hooks ((h) hardening / admin endpoints).
        """
        return self._evict_idle_at(self._clock())

    def evict_lru(self, target_count: int | None = None) -> int:
        """Drop oldest entries until the map size is at or below the
        target. ``target_count=None`` (default) drops to
        ``max_sessions``; an explicit non-negative value overrides
        for one call.

        Returns the number of entries dropped.

        Raises :class:`ValueError` for negative ``target_count`` —
        the existing :meth:`_evict_lru_to_cap` over-computes
        ``excess`` against a negative cap and would advertise a
        nonsensical size, so reject the input rather than silently
        misbehaving in admin / maintenance hooks.
        """
        if target_count is None:
            return self._evict_lru_to_cap()
        if target_count < 0:
            raise ValueError(
                f"target_count must be >= 0, got {target_count}"
            )
        original = self._max_sessions
        self._max_sessions = target_count
        try:
            return self._evict_lru_to_cap()
        finally:
            self._max_sessions = original

    def stats(self) -> Mapping[str, PrefixCacheStats]:
        """Snapshot of per-session prefix-cache stats.

        Returns a fresh dict keyed by session_id; each value is the
        result of :meth:`RadixPrefixCache.stats`. Used by R-f
        acceptance and by future ``/v1/sessions`` admin endpoints
        ((h)). Lookups walk the underlying store, so prefer to call
        infrequently — this is not a hot-path metric source.
        """
        return {
            sid: entry.prefix_cache.stats()
            for sid, entry in self._sessions.items()
        }

    def close(self) -> None:
        """Drop every session.

        Called from :meth:`silica.server.runtime.Runtime.close` on
        FastAPI lifespan shutdown. Idempotent.
        """
        if not self._sessions:
            return
        log.info("session.manager.close count=%d", len(self._sessions))
        self._sessions.clear()

    # --- internals ------------------------------------------------------

    def _build_entry(
        self, session_id: str, *, now: float
    ) -> _SessionEntry:
        store = SyntheticPrefixBlockStore(block_size=self._block_size)
        prefix_cache = RadixPrefixCache(
            block_size=self._block_size, store=store
        )
        chat_session = ChatSession(
            adapter=self._adapter,
            engine=self._engine,
            # The Protocol member ``_PrefixCacheLike.block_size`` is
            # declared as a settable variable for the chat-CLI's
            # fake caches; the concrete ``RadixPrefixCache`` exposes
            # it as a read-only ``@property``. Variance flags this
            # at strict-mypy time even though the runtime contract
            # is satisfied. Same shape as the chat-CLI's
            # ``silica/chat/cli/app.py`` constructor site.
            prefix_cache=prefix_cache,  # type: ignore[arg-type]
        )
        return _SessionEntry(
            session_id=session_id,
            chat_session=chat_session,
            prefix_cache=prefix_cache,
            last_access_s=now,
        )

    def _evict_idle_at(self, now: float) -> int:
        cutoff = now - self._ttl_s
        ids_to_drop = [
            sid
            for sid, entry in self._sessions.items()
            if entry.last_access_s < cutoff
        ]
        for sid in ids_to_drop:
            log.info("session.evict.idle id=%s", sid)
            del self._sessions[sid]
        return len(ids_to_drop)

    def _evict_lru_to_cap(self) -> int:
        excess = len(self._sessions) - self._max_sessions
        if excess <= 0:
            return 0
        sids = list(self._sessions.keys())
        for sid in sids[:excess]:
            log.info("session.evict.lru id=%s", sid)
            del self._sessions[sid]
        return excess
