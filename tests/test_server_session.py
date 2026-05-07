"""Unit tests for :class:`silica.server.session.SessionManager`
(P-8 sub-unit (f)).

Pins the lifecycle contract:

- ``get_or_create`` returns the same :class:`ChatSession` instance for
  the same ``session_id`` across calls (the load-bearing acceptance
  for cross-request prefix reuse).
- A fresh ``session_id`` produces a fresh ``ChatSession`` whose
  prefix cache is a fresh :class:`RadixPrefixCache`.
- ``replace_messages`` runs on every call so the OpenAI client's
  full-history send pattern is authoritative; the prefix cache
  instance carried by the session is preserved across the replace.
- LRU eviction drops oldest entries when ``max_sessions`` is
  exceeded.
- TTL eviction drops idle entries on the next ``get_or_create``
  (an out-of-band ``evict_idle`` reaches the same code path).
- ``stats`` returns a per-session :class:`PrefixCacheStats` snapshot.
- ``close`` clears the map and is idempotent.

No real model is loaded; tests pass minimal adapter / engine fakes
that never exercise generate / generate_batch.
"""

from __future__ import annotations

from typing import Any

import pytest

from silica.kvcache.prefix import PrefixCacheStats
from silica.models.adapter import AttentionKind, AttentionPattern
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)
from silica.server.session import (
    DEFAULT_MAX_SESSIONS,
    DEFAULT_PREFIX_CACHE_BLOCK_SIZE,
    DEFAULT_SESSION_TTL_S,
    SessionManager,
)

# ---------------------------------------------------------------------------
# Minimal fakes: SessionManager plumbs adapter / engine through to
# ChatSession.__init__, which reads ``adapter.tokenizer()`` once, and
# reads ``adapter.capabilities()`` once at SessionManager construction
# (the SLIDING-attention guard, sub-unit (f) Finding 1). None of the
# tests below drive ``ChatSession.chat`` so we never need a real
# engine; the integration test for that lives in test_server_chat_completions.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_ids: set[int] = set()


class _FakeAdapter:
    """Adapter stub that defaults to all-GLOBAL attention so
    ``supports_prefix_reuse`` is ``True``.

    Tests that need to pin the SLIDING rejection path override
    ``capabilities_override`` on a per-instance basis to surface
    :attr:`AttentionKind.SLIDING` in the capability set.
    """

    def __init__(
        self,
        *,
        capabilities_override: ModelCapabilities | None = None,
    ) -> None:
        self._capabilities = (
            capabilities_override
            if capabilities_override is not None
            else capabilities_from_attention_pattern(
                AttentionPattern(per_layer=(AttentionKind.GLOBAL,))
            )
        )

    def tokenizer(self) -> _FakeTokenizer:
        return _FakeTokenizer()

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities


class _FakeEngine:
    """Minimal stand-in for :class:`silica.engine.Engine`.

    ChatSession's ``__init__`` only reads ``self._engine`` for storage,
    so this fake does not need to implement ``generate`` /
    ``generate_batch`` for the SessionManager tests below. Tests that
    drive a full chat turn use a richer fake engine in
    test_server_chat_completions.
    """


def _make_manager(
    *,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    session_ttl_s: float = DEFAULT_SESSION_TTL_S,
    block_size: int = DEFAULT_PREFIX_CACHE_BLOCK_SIZE,
    clock_fn: Any = None,
    adapter: _FakeAdapter | None = None,
) -> SessionManager:
    return SessionManager(
        adapter=adapter if adapter is not None else _FakeAdapter(),
        engine=_FakeEngine(),
        max_sessions=max_sessions,
        session_ttl_s=session_ttl_s,
        prefix_cache_block_size=block_size,
        clock=clock_fn if clock_fn is not None else _ClockStub(),
    )


class _ClockStub:
    """Monotonic clock with manual advance.

    SessionManager defaults to :func:`time.monotonic`; tests need a
    deterministic surface so TTL eviction is reproducible without
    sleeping.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# get_or_create stable-instance contract (load-bearing for prefix reuse).
# ---------------------------------------------------------------------------


def test_get_or_create_returns_same_chat_session_for_same_id() -> None:
    """The acceptance promise: a session_id revisited returns the
    *same* ChatSession instance. Prefix-cache reuse is impossible
    without this — a fresh ChatSession per call would carry a fresh
    RadixPrefixCache and the second turn's peek would always be 0.
    """
    manager = _make_manager()
    s1 = manager.get_or_create("alpha", system_prompt="hi", history=[])
    s2 = manager.get_or_create("alpha", system_prompt="hi", history=[])
    assert s1 is s2
    # The persistent ChatSession's prefix_cache also stays the same
    # across calls — the load-bearing identity.
    assert s1.prefix_cache is s2.prefix_cache


def test_get_or_create_returns_distinct_sessions_for_different_ids() -> None:
    """Different session_ids must produce isolated ChatSessions —
    each with its own RadixPrefixCache so the G-2 isolation
    invariant (per-session cache) holds."""
    manager = _make_manager()
    s1 = manager.get_or_create("alpha", system_prompt=None, history=[])
    s2 = manager.get_or_create("beta", system_prompt=None, history=[])
    assert s1 is not s2
    assert s1.prefix_cache is not s2.prefix_cache


def test_get_or_create_replaces_messages_on_each_call() -> None:
    """Option α from OPENING §4.2: the OpenAI client sends the full
    conversation each request, so the route's history is authoritative
    over the session's prior state. ``replace_messages`` on every call
    is the wire-level enforcement."""
    manager = _make_manager()
    s = manager.get_or_create(
        "alpha",
        system_prompt="be terse",
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    assert [m["role"] for m in s.messages] == [
        "system",
        "user",
        "assistant",
    ]
    # Same session_id, fresh history (no system prompt this time).
    s_same = manager.get_or_create(
        "alpha",
        system_prompt=None,
        history=[{"role": "user", "content": "again"}],
    )
    assert s_same is s
    assert [m["role"] for m in s_same.messages] == ["user"]


def test_replace_messages_does_not_drop_prefix_cache() -> None:
    """The prefix cache instance must survive the ``replace_messages``
    call inside :meth:`get_or_create` — otherwise repeated
    same-session calls would silently reset cached blocks every
    turn, defeating the whole sub-unit."""
    manager = _make_manager()
    s = manager.get_or_create("alpha", system_prompt="x", history=[])
    pc_before = s.prefix_cache
    manager.get_or_create(
        "alpha",
        system_prompt="y",
        history=[{"role": "user", "content": "go"}],
    )
    assert s.prefix_cache is pc_before


def test_get_returns_none_for_unknown_id() -> None:
    """Read-only :meth:`get` does not auto-create; tests / introspection
    paths use it without growing the map."""
    manager = _make_manager()
    assert manager.get("missing") is None


def test_contains_and_len_track_active_sessions() -> None:
    manager = _make_manager()
    assert len(manager) == 0
    assert "alpha" not in manager
    manager.get_or_create("alpha", system_prompt=None, history=[])
    assert len(manager) == 1
    assert "alpha" in manager


# ---------------------------------------------------------------------------
# LRU eviction.
# ---------------------------------------------------------------------------


def test_lru_evicts_oldest_when_cap_exceeded() -> None:
    """Insertion-order in the dict tracks recency; the oldest
    insertion is dropped when the cap is exceeded by the n-th
    create call."""
    manager = _make_manager(max_sessions=2)
    s_a = manager.get_or_create("a", system_prompt=None, history=[])
    manager.get_or_create("b", system_prompt=None, history=[])
    manager.get_or_create("c", system_prompt=None, history=[])
    assert "a" not in manager
    assert "b" in manager
    assert "c" in manager
    # The session for ``a`` is gone — a new get_or_create rebuilds a
    # fresh one with a fresh prefix cache.
    s_a_new = manager.get_or_create("a", system_prompt=None, history=[])
    assert s_a_new is not s_a


def test_get_or_create_refreshes_lru_recency() -> None:
    """Re-accessing a session moves it to the most-recent slot so
    a subsequent eviction hits a different (older) entry."""
    manager = _make_manager(max_sessions=2)
    manager.get_or_create("a", system_prompt=None, history=[])
    manager.get_or_create("b", system_prompt=None, history=[])
    # Touch ``a``: now ``b`` is the oldest.
    manager.get_or_create("a", system_prompt=None, history=[])
    manager.get_or_create("c", system_prompt=None, history=[])
    assert "a" in manager
    assert "b" not in manager
    assert "c" in manager


def test_evict_lru_with_explicit_target() -> None:
    """The public :meth:`evict_lru` takes an optional cap override
    for one call so admin endpoints / tests can shrink the map
    without permanently changing ``max_sessions``."""
    manager = _make_manager(max_sessions=10)
    for i in range(5):
        manager.get_or_create(
            f"sess-{i}", system_prompt=None, history=[]
        )
    assert len(manager) == 5
    dropped = manager.evict_lru(target_count=2)
    assert dropped == 3
    assert len(manager) == 2
    # Cap restored — adding a sixth session does not trigger eviction.
    manager.get_or_create("after", system_prompt=None, history=[])
    assert len(manager) == 3


# ---------------------------------------------------------------------------
# TTL eviction.
# ---------------------------------------------------------------------------


def test_ttl_evicts_idle_entries_on_next_get_or_create() -> None:
    """Idle eviction runs inline on the next :meth:`get_or_create`
    call — the busy path does the cleanup, so a long-idle process
    does not need a background sweeper."""
    clock = _ClockStub()
    manager = _make_manager(session_ttl_s=10.0, clock_fn=clock)
    manager.get_or_create("alpha", system_prompt=None, history=[])
    clock.advance(11.0)
    # Touch a different session_id — the inline sweep evicts ``alpha``.
    manager.get_or_create("beta", system_prompt=None, history=[])
    assert "alpha" not in manager
    assert "beta" in manager


def test_ttl_does_not_evict_recently_accessed_entries() -> None:
    """Below TTL the entry stays. Touching it inside the window
    pushes the access timestamp forward, preventing eviction."""
    clock = _ClockStub()
    manager = _make_manager(session_ttl_s=10.0, clock_fn=clock)
    manager.get_or_create("alpha", system_prompt=None, history=[])
    clock.advance(5.0)
    manager.get_or_create("alpha", system_prompt=None, history=[])
    clock.advance(8.0)  # 13s since first access; 8s since the touch
    manager.get_or_create("beta", system_prompt=None, history=[])
    assert "alpha" in manager


def test_evict_idle_public_returns_dropped_count() -> None:
    """Out-of-band sweep: useful for an admin / maintenance hook."""
    clock = _ClockStub()
    manager = _make_manager(session_ttl_s=10.0, clock_fn=clock)
    manager.get_or_create("alpha", system_prompt=None, history=[])
    manager.get_or_create("beta", system_prompt=None, history=[])
    clock.advance(20.0)
    dropped = manager.evict_idle()
    assert dropped == 2
    assert len(manager) == 0


# ---------------------------------------------------------------------------
# stats.
# ---------------------------------------------------------------------------


def test_stats_returns_per_session_prefix_cache_stats() -> None:
    """The R-f acceptance reads the manager's stats to verify
    cross-request prefix reuse. The shape pin guards against a
    refactor accidentally swapping the stats source.
    """
    manager = _make_manager()
    manager.get_or_create("alpha", system_prompt=None, history=[])
    manager.get_or_create("beta", system_prompt=None, history=[])
    stats = manager.stats()
    assert set(stats.keys()) == {"alpha", "beta"}
    for entry in stats.values():
        assert isinstance(entry, PrefixCacheStats)
        assert entry.block_size == DEFAULT_PREFIX_CACHE_BLOCK_SIZE
        # Fresh cache — no hits yet, no blocks resident.
        assert entry.hits == 0


# ---------------------------------------------------------------------------
# close.
# ---------------------------------------------------------------------------


def test_close_clears_all_sessions() -> None:
    manager = _make_manager()
    manager.get_or_create("alpha", system_prompt=None, history=[])
    manager.get_or_create("beta", system_prompt=None, history=[])
    manager.close()
    assert len(manager) == 0
    # Idempotent: a second close is a no-op.
    manager.close()
    assert len(manager) == 0


def test_close_is_idempotent_on_empty_manager() -> None:
    manager = _make_manager()
    manager.close()
    assert len(manager) == 0


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message_substring",
    [
        ({"max_sessions": 0}, "max_sessions"),
        ({"max_sessions": -1}, "max_sessions"),
        ({"session_ttl_s": 0.0}, "session_ttl_s"),
        ({"session_ttl_s": -1.0}, "session_ttl_s"),
        ({"prefix_cache_block_size": 0}, "prefix_cache_block_size"),
    ],
)
def test_constructor_rejects_invalid_kwargs(
    kwargs: dict[str, Any], message_substring: str
) -> None:
    """Cheap input validation: non-positive sizes / TTLs would
    silently break eviction or radix-tree construction. Surfaces
    misconfiguration at server startup, not on the first request."""
    with pytest.raises(ValueError, match=message_substring):
        SessionManager(
            adapter=_FakeAdapter(), engine=_FakeEngine(), **kwargs
        )


# ---------------------------------------------------------------------------
# Capability guard: SLIDING-bearing adapters cannot host persistent
# sessions in v0.1 (sub-unit (f) Finding 1).
# ---------------------------------------------------------------------------


def test_supports_prefix_reuse_true_for_global_only_adapter() -> None:
    """The Qwen3 / Qwen3.5 dense path (all-GLOBAL attention) is
    compatible with the radix prefix cache, so the route can hand
    ``session_id`` requests to the manager without further checks."""
    manager = _make_manager()
    assert manager.supports_prefix_reuse is True


def test_supports_prefix_reuse_false_for_sliding_adapter() -> None:
    """Gemma4 31B today: SLIDING in attention_kinds means
    :class:`ContinuousBatcher` rejects ``RadixPrefixCache`` in
    construction (silica/scheduler/batcher.py:292), so persistent
    sessions are unsupported. The manager flags this so the route
    can return 501 instead of letting the request fall through to a
    500."""
    sliding_adapter = _FakeAdapter(
        capabilities_override=capabilities_from_attention_pattern(
            AttentionPattern(
                per_layer=(
                    AttentionKind.SLIDING,
                    AttentionKind.GLOBAL,
                )
            )
        )
    )
    manager = _make_manager(adapter=sliding_adapter)
    assert manager.supports_prefix_reuse is False


def test_get_or_create_raises_for_unsupported_adapter() -> None:
    """If the route fails to gate on
    :attr:`supports_prefix_reuse` the public API surfaces the
    misuse instead of silently constructing a doomed session
    whose first ``chat`` would crash inside the batcher.
    """
    sliding_adapter = _FakeAdapter(
        capabilities_override=capabilities_from_attention_pattern(
            AttentionPattern(per_layer=(AttentionKind.SLIDING,))
        )
    )
    manager = _make_manager(adapter=sliding_adapter)
    with pytest.raises(RuntimeError, match="SLIDING"):
        manager.get_or_create("any", system_prompt=None, history=[])


# ---------------------------------------------------------------------------
# evict_lru input validation (sub-unit (f) Finding 4).
# ---------------------------------------------------------------------------


def test_evict_lru_rejects_negative_target_count() -> None:
    """``evict_lru(target_count=-N)`` would over-compute ``excess``
    (``len(self) - (-N)``) and drop more entries than exist —
    actively misleading instead of safe. Reject up front."""
    manager = _make_manager()
    manager.get_or_create("a", system_prompt=None, history=[])
    with pytest.raises(ValueError, match="target_count"):
        manager.evict_lru(target_count=-1)
    # State is unchanged after the rejection.
    assert "a" in manager


def test_evict_lru_target_zero_drops_everything() -> None:
    """The boundary value: ``target_count=0`` is a meaningful
    explicit drop-all, and the validator must let it through."""
    manager = _make_manager()
    manager.get_or_create("a", system_prompt=None, history=[])
    manager.get_or_create("b", system_prompt=None, history=[])
    dropped = manager.evict_lru(target_count=0)
    assert dropped == 2
    assert len(manager) == 0
