"""silica.speculative.draft_target — D-021 step 5 sub-units (a) + (c) slice 2a.

C.1 baseline draft engine: a small autoregressive draft model + an
internal ``KVManager`` instance, wrapped behind the I-5 ``DraftEngine``
Protocol so the engine main loop swaps it in for ``NoopDraftEngine``
without changing its calling shape.

Convention pinned in ``plans/P6_SPEC_FOUNDATION_OPENING.md`` §4.5:

  - ``k`` (the Protocol parameter) = γ = number of drafts proposed per
    cycle, equivalently ``verify_k - 1`` where ``verify_k`` is the target
    verify forward input length. The draft itself does not see ``verify_k``;
    it only knows γ — the count its caller asks for.
  - On every cycle the draft's KV must stay in sync with the target's
    *committed* token sequence — exactly ``ctx.request.token_ids +
    ctx.output_token_ids`` at the moment ``propose`` is entered.
  - ``commit(ctx, accepted_len)`` rolls the per-``req_id`` draft KV back
    by ``γ - accepted_len`` positions; the bonus token sampled by the
    engine is consumed lazily at the next ``propose`` call's catch-up
    phase (single ``decode_step`` over the new tail of
    ``output_token_ids``).
  - Recurrent-state rollback: on hybrid drafts (Qwen3.5-0.8B), KV-only
    rollback via ``KVManager.rollback`` is best-effort — non-trimmable
    recurrent caches no-op silently. Sub-unit (a) handles plain-KV drafts
    correctly (Qwen3-0.6B). Full hybrid correctness joins at sub-unit (e)
    when ``ModelAdapter.rollback_recurrent_state`` lands; the parity test
    in sub-unit (f) exercises both paths together.

Slice 2a — D-021 step 5 sub-unit (c) prerequisite for the multi-row
batcher (slice 2b). Every per-request bookkeeping field is keyed by
``req_id`` derived from ``ctx.request.request_id`` at propose / commit /
reset time. ``DraftTargetEngine`` now serves N concurrent target
requests through one drafter instance — the architectural enabler for
slice 2b's per-step batched verify forward where γ propose calls
happen in lockstep over multiple rows.

The legacy single-request ``SimpleKVCache`` is incompatible with
concurrent reservations (silica/kvcache/simple.py:62-66 — it raises on
a double-claim). Slice 2a introduces ``_MultiKVCache``, a private
multi-request wrapper that satisfies the same I-2 surface adapters
consume (``cache_list(req_id)`` / ``reserve_for_prefill`` /
``rollback`` / ``free``) but holds an mlx-lm cache list per ``req_id``.
``from_repo`` swaps the adapter's ``_kv_manager`` field with the
multi-wrapper at construction so every adapter forward call dispatches
through the per-``req_id`` cache list. Test fixtures inject any
multi-aware fake directly via ``__init__``.

Key contract: ``DraftTargetEngine`` does **not** see the target's KV.
Verification + ``accepted_len`` is the engine main loop's job; the
draft only knows what it proposed last and how much survived, keyed
per ``req_id``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
from mlx_lm.models import cache as mlx_cache

from silica.core.request import RequestState
from silica.kvcache.manager import (
    BlockList,
    KVHandle,
    MemoryBudget,
    PrefixHit,
)
from silica.kvcache.simple import SimpleKVCache
from silica.models.adapter import ModelAdapter
from silica.speculative.engine import DraftTokens

_MLX_KVCACHE_GROWTH_CHUNK = 256


class _MultiKVCache:
    """Multi-request KVManager for DraftTargetEngine.

    Wraps a single mlx-lm-shaped model's cache factory so each ``req_id``
    gets its own auto-grown cache list. Implements the I-2 KVManager
    surface that ``ModelAdapter`` consumes — ``cache_list(req_id)``,
    ``reserve_for_prefill``, ``rollback``, ``free`` — keying every
    operation on ``req_id`` and never raising the single-owner conflict
    ``SimpleKVCache`` enforces. The other I-2 methods (``commit``,
    ``append_slot``, ``get_computed_blocks``, ``available_blocks``,
    ``budget``) are stubs sized for the drafter's needs; this class is
    private to the speculative module and is NOT a general-purpose
    paged manager.

    Why not run multiple ``SimpleKVCache`` instances inside the drafter:
    each ``SimpleKVCache`` carries a single owner-flag and rejects a
    second claim by a different ``req_id``. Holding N instances would
    duplicate the per-instance overhead and complicate the dispatch
    surface; one ``_MultiKVCache`` holding N independent cache lists is
    structurally equivalent and exposes a uniform ``cache_list(req_id)``
    that the adapter can consume without any per-request branching.
    """

    block_size: int = _MLX_KVCACHE_GROWTH_CHUNK

    def __init__(self, model: Any) -> None:
        self._model = model
        self._caches: dict[str, list[Any]] = {}

    # --- adapter consumption surface ---

    def cache_list(self, req_id: str) -> list[Any]:
        """Return the mlx-lm per-layer cache list for ``req_id``.

        Adapter ``prefill`` / ``decode_step`` / ``decode_step_multi``
        invoke this to drive ``model(tokens, cache=cache_list)``. Raises
        ``ValueError`` when ``req_id`` has not been reserved — surfaces
        a misuse loud rather than silently growing a fresh slot.
        """
        if req_id not in self._caches:
            raise ValueError(
                f"_MultiKVCache: req_id {req_id!r} has no reservation; "
                f"call reserve_for_prefill before adapter forward calls"
            )
        return self._caches[req_id]

    # --- I-2 KVManager Protocol surface ---

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        del token_ids
        if req_id in self._caches:
            raise ValueError(
                f"_MultiKVCache: req_id {req_id!r} already reserved; "
                f"call free first to release"
            )
        self._caches[req_id] = mlx_cache.make_prompt_cache(self._model)
        return BlockList()

    def append_slot(self, req_id: str, n: int) -> BlockList:
        del n
        if req_id not in self._caches:
            raise ValueError(
                f"_MultiKVCache: req_id {req_id!r} not reserved"
            )
        return BlockList()

    def commit(self, req_id: str, n_accepted: int) -> None:
        del n_accepted
        if req_id not in self._caches:
            raise ValueError(
                f"_MultiKVCache: req_id {req_id!r} not reserved"
            )

    def rollback(self, req_id: str, n_reject: int) -> None:
        if req_id not in self._caches:
            raise ValueError(
                f"_MultiKVCache: req_id {req_id!r} not reserved"
            )
        if n_reject <= 0:
            return
        cache = self._caches[req_id]
        if mlx_cache.can_trim_prompt_cache(cache):
            mlx_cache.trim_prompt_cache(cache, n_reject)

    def free(self, req_id: str) -> None:
        # Idempotent: free of a non-reserved req_id is a no-op so the
        # drafter's reset(req_id=None) sweep does not need to track
        # which req_ids actually have live cache lists.
        self._caches.pop(req_id, None)

    def get_computed_blocks(
        self, token_ids: Sequence[int]
    ) -> PrefixHit:
        del token_ids
        return PrefixHit()

    def available_blocks(self) -> int:
        return 0

    def budget(self) -> MemoryBudget:
        total = 0
        for cache_list in self._caches.values():
            total += sum(
                int(c.nbytes) for c in cache_list if hasattr(c, "nbytes")
            )
        return MemoryBudget(
            logical_bytes=total,
            resident_bytes=total,
            headroom_bytes=0,
        )


class DraftTargetEngine:
    """I-5 ``DraftEngine`` backed by a real autoregressive draft model.

    Owns its own ``ModelAdapter`` and a multi-request KV manager
    (``_MultiKVCache`` in production via :meth:`from_repo`; tests
    inject any I-2-shaped fake that supports concurrent ``req_id``
    reservations). The target's KV is invisible to this class — the
    engine main loop is the only thing that bridges target-side
    verification to ``commit``.

    Slice 2a (D-021 step 5 sub-unit (c)): every per-request bookkeeping
    field is keyed by ``req_id`` derived from
    ``ctx.request.request_id`` at propose / commit / reset time. One
    drafter instance can serve N concurrent target requests; A→B→A
    switching preserves each ``req_id``'s draft state across the
    interleaved cycles.

    Construct via ``__init__`` for tests (inject a fake adapter +
    multi-aware fake KV) or via :meth:`from_repo` for production wiring
    (loads via ``silica.models.factory.adapter_for_repo`` and replaces
    the adapter's single-owner ``SimpleKVCache`` with a
    ``_MultiKVCache``).
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        kv: SimpleKVCache,
    ) -> None:
        # ``kv`` is annotated ``SimpleKVCache`` for backward compatibility
        # with existing call sites; at runtime any I-2 manager that
        # supports concurrent ``req_id`` reservations works
        # (``_MultiKVCache`` in production; a multi-aware fake in tests).
        # The single-owner SimpleKVCache works only when at most one
        # ``req_id`` is ever passed through propose / commit / reset
        # for the lifetime of the drafter — kept as a permitted
        # legacy single-request shape (existing
        # ``tests/test_draft_target_engine.py`` fixtures use this).
        self._adapter = adapter
        self._kv = kv
        # Per-``req_id`` bookkeeping. Populated lazily on first propose
        # for a given ``req_id``; cleaned up by ``commit`` (the
        # per-cycle ``last_propose_count`` reset) and ``reset(req_id)``
        # (full per-request teardown).
        self._reserved: set[str] = set()
        self._draft_kv_pos: dict[str, int] = {}
        self._last_propose_count: dict[str, int] = {}
        self._cached_last_logits: dict[str, mx.array] = {}

    @classmethod
    def from_repo(cls, repo: str) -> DraftTargetEngine:
        """Load ``repo`` via the silica factory and wrap as a draft engine.

        Replaces the adapter's single-owner ``SimpleKVCache`` (built by
        the factory) with a ``_MultiKVCache`` so production drafter use
        supports concurrent ``req_id``s out of the box. The replacement
        is a private-field swap on the adapter; this is acceptable
        because the adapter loaded here is dedicated to the drafter's
        lifecycle and shared by no other consumer.
        """
        from silica.models.factory import adapter_for_repo

        adapter, _legacy_kv = adapter_for_repo(repo)
        # Pull the model the factory already loaded. ``_model`` is a
        # private field on every concrete adapter (qwen3.py / qwen3_5.py
        # / gemma4.py / qwen3_5_moe.py / gemma4_moe.py); the drafter
        # owns the adapter's lifetime so reaching into the private
        # field at construction time is bounded in scope.
        model: Any = getattr(adapter, "_model")
        multi_kv = _MultiKVCache(model)
        # Swap the adapter's ``_kv_manager`` so adapter forward calls
        # dispatch through the per-``req_id`` cache list.
        adapter._kv_manager = multi_kv  # type: ignore[attr-defined]
        # Cast through ``Any`` because ``DraftTargetEngine.__init__`` is
        # typed against ``SimpleKVCache`` for backward compat; the
        # multi-wrapper is structurally compatible at runtime.
        return cls(adapter, multi_kv)  # type: ignore[arg-type]

    # --- I-5 DraftEngine Protocol surface ---

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        """Run ``k = γ`` greedy autoregressive forwards on the draft.

        Slice 2a: ``ctx.request.request_id`` is the per-request key.
        Bookkeeping for the request is initialised lazily on cycle 0
        (when the request first appears) and survives across A→B→A
        interleavings without context-switch overhead.

        Returns the γ drafted ids plus per-token logprobs. The
        request's draft KV is advanced by (catch-up tokens since last
        cycle for this req_id) + γ. ``k <= 0`` is legal and returns
        an empty draft (no forward issued).
        """
        gamma = int(k)
        req_id = ctx.request.request_id
        if gamma <= 0:
            self._last_propose_count[req_id] = 0
            return DraftTokens(token_ids=())

        target_committed: list[int] = list(ctx.request.token_ids) + list(
            ctx.output_token_ids
        )
        draft_kv_pos = self._draft_kv_pos.get(req_id, 0)
        n_to_consume = len(target_committed) - draft_kv_pos
        if n_to_consume < 0:
            raise RuntimeError(
                f"draft {req_id!r}: target committed "
                f"{len(target_committed)} tokens but draft KV is at "
                f"{draft_kv_pos}; draft has run ahead of target — "
                f"bookkeeping bug"
            )

        handle = KVHandle(req_id=req_id)
        logits: mx.array | None
        if draft_kv_pos == 0:
            # Cycle 0 for this req_id: prefill the prompt + the engine's
            # first sampled token (already in output_token_ids by the
            # time propose is entered).
            if not target_committed:
                raise RuntimeError(
                    f"draft {req_id!r}: cycle-0 propose entered with "
                    f"empty target_committed (no prompt and no anchor)"
                )
            self._kv.reserve_for_prefill(req_id, target_committed)
            self._reserved.add(req_id)
            tokens_arr = mx.array(target_committed, dtype=mx.int32)
            logits, _ = self._adapter.prefill(tokens_arr, handle)
            draft_kv_pos = len(target_committed)
            self._cached_last_logits.pop(req_id, None)
        elif n_to_consume > 0:
            # Catch-up: bonus token(s) committed by the engine since
            # this req_id's last propose. Feed them through decode_step
            # one at a time so the tail logits we use to draft come
            # from the correct state.
            tail = target_committed[draft_kv_pos:]
            logits = None
            for tok in tail:
                logits, _ = self._adapter.decode_step(
                    mx.array([tok], dtype=mx.int32), handle
                )
            draft_kv_pos = len(target_committed)
            self._cached_last_logits.pop(req_id, None)
            assert logits is not None  # tail was non-empty
        else:
            # n_to_consume == 0: no new committed tokens since this
            # req_id's last propose. Use cached tail logits if
            # available; otherwise refuse (likely two propose calls
            # without an intervening commit).
            cached = self._cached_last_logits.get(req_id)
            if cached is None:
                raise RuntimeError(
                    f"draft {req_id!r}: propose entered with no new "
                    f"committed tokens and no cached tail logits — "
                    f"likely propose called twice without commit "
                    f"in between"
                )
            logits = cached

        # Greedy autoregressive loop: γ forwards.
        drafts: list[int] = []
        logprobs: list[float] = []
        for _ in range(gamma):
            tok_id = int(mx.argmax(logits).item())
            # log_softmax in mlx-native ops (D-009 hot-path constraint):
            # no numpy / torch round-trips.
            lp = float(
                (logits[tok_id] - mx.logsumexp(logits, axis=-1)).item()
            )
            drafts.append(tok_id)
            logprobs.append(lp)
            logits, _ = self._adapter.decode_step(
                mx.array([tok_id], dtype=mx.int32), handle
            )
            draft_kv_pos += 1

        self._draft_kv_pos[req_id] = draft_kv_pos
        self._last_propose_count[req_id] = gamma
        # Snapshot the post-final-forward logits so a full-accept
        # next-cycle (where the bonus token IS the γ-th draft's
        # prediction, already consumed into KV) can reuse them.
        # Partial-accept paths clear this in commit.
        self._cached_last_logits[req_id] = logits
        return DraftTokens(
            token_ids=tuple(drafts),
            draft_logprobs=tuple(logprobs),
        )

    def commit(self, ctx: RequestState, accepted_len: int) -> None:
        """Roll the request's draft KV back by ``γ - accepted_len``.

        Slice 2a: ``ctx.request.request_id`` keys the bookkeeping;
        commit only mutates the named request's draft state and never
        touches another request's KV.

        ``accepted_len == γ`` is a no-op (draft and target stay in
        sync via the cached tail logits). ``accepted_len < γ`` trims
        the request's draft KV back to the accepted prefix; the bonus
        token from the rejected position is consumed at the next
        ``propose`` catch-up phase for this same req_id.
        """
        req_id = ctx.request.request_id
        gamma = self._last_propose_count.get(req_id, 0)
        if gamma == 0:
            # No drafts to commit for this req_id. Tolerate per
            # Protocol — commit must accept any accepted_len without
            # raising.
            return
        if accepted_len < 0 or accepted_len > gamma:
            raise ValueError(
                f"draft {req_id!r}: accepted_len={accepted_len} not "
                f"in [0, {gamma}] (last propose returned {gamma} drafts)"
            )
        n_reject = gamma - accepted_len
        if n_reject > 0:
            # KV rollback. Best-effort: trimmable layers shrink,
            # recurrent (non-trimmable) layers no-op silently. Plain-KV
            # drafts are fully correct here; hybrid drafts will need
            # ``adapter.rollback_recurrent_state`` from sub-unit (e)
            # for full correctness on partial accept.
            self._kv.rollback(req_id, n_reject)
            self._draft_kv_pos[req_id] = (
                self._draft_kv_pos.get(req_id, 0) - n_reject
            )
            # Tail logits no longer reflect the post-rollback state.
            # Force the next propose's catch-up branch for this req_id.
            self._cached_last_logits.pop(req_id, None)
        # last_propose_count must reset so a commit-then-commit (or
        # commit-then-empty-propose) sequence does not double-count.
        self._last_propose_count[req_id] = 0

    # --- lifecycle ---

    def reset(self, req_id: str | None = None) -> None:
        """Release draft state for ``req_id``, or for every active
        ``req_id`` when ``req_id is None`` (teardown sweep).

        Slice 2a: per-request reset releases just the named request's
        KV slot and clears its bookkeeping; other active req_ids are
        unaffected. The ``req_id is None`` overload sweeps every
        currently-reserved req_id — the existing call shape Engine /
        spec-active batcher already use (``getattr(drafter, "reset",
        None)()``) reaches this branch and works as before.

        Crucially, this **trims the underlying KV cache list** (via a
        full-depth ``rollback`` issued before ``free``) rather than
        only releasing ownership. Without this trim, a subsequent
        request's cycle-0 ``prefill`` would land on top of the
        previous request's tokens in the underlying mlx-lm cache,
        producing wrong logits. ``rollback`` must run before ``free``
        because the I-2 owner check rejects rollback once ownership
        is released.

        Caveat for hybrid drafts (Qwen3.5-0.8B family): underlying
        ``trim_prompt_cache`` silently no-ops on non-trimmable
        recurrent caches. Plain-KV drafts (Qwen3-0.6B) are fully
        clean after ``reset``; full hybrid cleanup joins at
        sub-unit (e) when ``ModelAdapter.rollback_recurrent_state``
        lands.
        """
        if req_id is None:
            ids = list(self._reserved)
        else:
            ids = [req_id]
        for rid in ids:
            if rid in self._reserved:
                pos = self._draft_kv_pos.get(rid, 0)
                if pos > 0:
                    self._kv.rollback(rid, pos)
                self._kv.free(rid)
                self._reserved.discard(rid)
            self._draft_kv_pos.pop(rid, None)
            self._last_propose_count.pop(rid, None)
            self._cached_last_logits.pop(rid, None)

    # --- introspection (test-facing; not part of I-5) ---

    def draft_kv_pos_for(self, req_id: str) -> int:
        """Number of tokens currently in ``req_id``'s draft KV — for tests."""
        return self._draft_kv_pos.get(req_id, 0)

    def last_propose_count_for(self, req_id: str) -> int:
        """γ from the most recent propose for ``req_id`` — for tests."""
        return self._last_propose_count.get(req_id, 0)

    def active_req_ids(self) -> frozenset[str]:
        """The set of currently-reserved req_ids — for tests."""
        return frozenset(self._reserved)


__all__ = ["DraftTargetEngine"]
