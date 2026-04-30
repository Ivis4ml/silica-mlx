"""silica.speculative.draft_target — D-021 step 5 sub-unit (a).

C.1 baseline draft engine: a small autoregressive draft model + its own
``KVManager`` instance, wrapped behind the I-5 ``DraftEngine`` Protocol so
the engine main loop swaps it in for ``NoopDraftEngine`` without changing
its calling shape.

Convention pinned in ``plans/P6_SPEC_FOUNDATION_OPENING.md`` §4.5:

  - ``k`` (the Protocol parameter) = γ = number of drafts proposed per
    cycle, equivalently ``verify_k - 1`` where ``verify_k`` is the target
    verify forward input length. The draft itself does not see ``verify_k``;
    it only knows γ — the count its caller asks for.
  - On every cycle the draft's KV must stay in sync with the target's
    *committed* token sequence — exactly ``ctx.request.token_ids +
    ctx.output_token_ids`` at the moment ``propose`` is entered.
  - ``commit(accepted_len)`` rolls the draft's own KV back by
    ``γ - accepted_len`` positions; the bonus token sampled by the engine
    is consumed lazily at the next ``propose`` call's catch-up phase
    (single ``decode_step`` over the new tail of ``output_token_ids``).
  - Recurrent-state rollback: on hybrid drafts (Qwen3.5-0.8B), KV-only
    rollback via ``KVManager.rollback`` is best-effort — non-trimmable
    recurrent caches no-op silently. Sub-unit (a) handles plain-KV drafts
    correctly (Qwen3-0.6B). Full hybrid correctness joins at sub-unit (e)
    when ``ModelAdapter.rollback_recurrent_state`` lands; the parity test
    in sub-unit (f) exercises both paths together.

Key contract: ``DraftTargetEngine`` does **not** see the target's KV.
Verification + ``accepted_len`` is the engine main loop's job; the draft
only knows what it proposed last and how much survived.
"""

from __future__ import annotations

import mlx.core as mx

from silica.core.request import RequestState
from silica.kvcache.manager import KVHandle
from silica.kvcache.simple import SimpleKVCache
from silica.models.adapter import ModelAdapter
from silica.speculative.engine import DraftTokens

_DEFAULT_DRAFT_REQ_ID = "draft"


class DraftTargetEngine:
    """I-5 ``DraftEngine`` backed by a real autoregressive draft model.

    Owns its own (``ModelAdapter``, ``SimpleKVCache``) pair. The target's KV
    is invisible to this class — the engine main loop is the only thing that
    bridges target-side verification to ``commit``.

    Construct via ``__init__`` for tests (inject a fake adapter) or via
    :meth:`from_repo` for production wiring (loads via
    ``silica.models.factory.adapter_for_repo``).
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        kv: SimpleKVCache,
        *,
        req_id: str = _DEFAULT_DRAFT_REQ_ID,
    ) -> None:
        self._adapter = adapter
        self._kv = kv
        self._req_id = req_id
        self._handle = KVHandle(req_id=req_id)
        # How many tokens of the running request the draft's KV has consumed.
        # Always tracks the prefix of (prompt + output_token_ids) the draft
        # has run forward through. After commit's rollback this drops by
        # ``n_reject``; the catch-up branch in propose brings it back up to
        # ``len(target_committed)`` before drafting starts.
        self._draft_kv_pos = 0
        # γ from the most recent propose — used by commit to compute n_reject.
        # 0 when no propose has run since the last commit.
        self._last_propose_count = 0
        # True once ``reserve_for_prefill`` has run on this engine. Idempotent
        # over the engine's lifetime — SimpleKVCache rejects a second claim by
        # a different req_id and we keep the same one.
        self._reserved = False
        # Cached "last logits" from the tail of the most recent propose's
        # autoregressive loop. Populated only on full-accept commits where the
        # next propose enters with ``n_to_consume == 0`` and would otherwise
        # need an extra forward to recover them. (Partial-accept commits roll
        # the draft KV back to ``draft_kv_pos = ... - n_reject``, after which
        # the catch-up branch in propose re-feeds the bonus and produces fresh
        # logits — those cached logits are wrong for the new state and are
        # cleared on every commit that rolls back.)
        self._cached_last_logits: mx.array | None = None

    @classmethod
    def from_repo(
        cls, repo: str, *, req_id: str = _DEFAULT_DRAFT_REQ_ID
    ) -> DraftTargetEngine:
        """Load ``repo`` via the silica factory and wrap as a draft engine."""
        # Local import keeps the module importable without mlx_lm at hand for
        # tests that only inject a fake adapter via __init__.
        from silica.models.factory import adapter_for_repo

        adapter, kv = adapter_for_repo(repo)
        return cls(adapter, kv, req_id=req_id)

    # --- I-5 DraftEngine Protocol surface ---

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        """Run ``k = γ`` greedy autoregressive forwards on the draft.

        Returns the γ drafted ids plus per-token logprobs. The draft's KV
        is advanced by (catch-up tokens since last cycle) + γ. ``k <= 0``
        is legal and returns an empty draft (no forward issued).
        """
        gamma = int(k)
        if gamma <= 0:
            self._last_propose_count = 0
            return DraftTokens(token_ids=())

        target_committed: list[int] = list(ctx.request.token_ids) + list(
            ctx.output_token_ids
        )
        n_to_consume = len(target_committed) - self._draft_kv_pos
        if n_to_consume < 0:
            raise RuntimeError(
                f"draft {self._req_id!r}: target committed "
                f"{len(target_committed)} tokens but draft KV is at "
                f"{self._draft_kv_pos}; draft has run ahead of target — "
                f"bookkeeping bug"
            )

        logits: mx.array | None
        if self._draft_kv_pos == 0:
            # Cycle 0: prefill the prompt + the engine's first sampled token
            # (already in output_token_ids by the time propose is entered).
            if not target_committed:
                raise RuntimeError(
                    f"draft {self._req_id!r}: cycle-0 propose entered with "
                    f"empty target_committed (no prompt and no anchor)"
                )
            self._kv.reserve_for_prefill(self._req_id, target_committed)
            self._reserved = True
            tokens_arr = mx.array(target_committed, dtype=mx.int32)
            logits, _ = self._adapter.prefill(tokens_arr, self._handle)
            self._draft_kv_pos = len(target_committed)
            self._cached_last_logits = None
        elif n_to_consume > 0:
            # Catch-up: bonus token(s) committed by the engine since the last
            # propose. Feed them through decode_step one at a time so the
            # tail logits we use to draft come from the correct state.
            tail = target_committed[self._draft_kv_pos:]
            logits = None
            for tok in tail:
                logits, _ = self._adapter.decode_step(
                    mx.array([tok], dtype=mx.int32), self._handle
                )
            self._draft_kv_pos = len(target_committed)
            self._cached_last_logits = None
            assert logits is not None  # tail was non-empty (n_to_consume > 0)
        else:
            # n_to_consume == 0: no new committed tokens since last propose.
            # This happens when the previous cycle was full-accept AND the
            # engine somehow did not yield a bonus (degenerate); or when
            # propose is called twice in a row without an intervening commit.
            # Use cached tail logits if available; otherwise refuse.
            if self._cached_last_logits is None:
                raise RuntimeError(
                    f"draft {self._req_id!r}: propose entered with no new "
                    f"committed tokens and no cached tail logits — likely "
                    f"propose called twice without commit in between"
                )
            logits = self._cached_last_logits

        # Greedy autoregressive loop: γ forwards.
        drafts: list[int] = []
        logprobs: list[float] = []
        for _ in range(gamma):
            tok_id = int(mx.argmax(logits).item())
            # log_softmax keeps the math in mlx-native ops (D-009 hot-path
            # constraint) — no numpy or torch round-trips.
            lp = float(
                (logits[tok_id] - mx.logsumexp(logits, axis=-1)).item()
            )
            drafts.append(tok_id)
            logprobs.append(lp)
            # Last iteration's forward is wasted under our convention — the
            # γ-th draft's K/V slot is needed for the next propose's catch-up
            # to land at the right position, so the forward must run. Its
            # logits become the "if no rollback happens" cache.
            logits, _ = self._adapter.decode_step(
                mx.array([tok_id], dtype=mx.int32), self._handle
            )
            self._draft_kv_pos += 1

        self._last_propose_count = gamma
        # Snapshot the post-final-forward logits so a full-accept next-cycle
        # (where the bonus token IS the γ-th draft's prediction, already
        # consumed into KV) can reuse them. Partial-accept paths clear this
        # in commit before any subsequent propose can read it.
        self._cached_last_logits = logits
        return DraftTokens(
            token_ids=tuple(drafts),
            draft_logprobs=tuple(logprobs),
        )

    def commit(self, ctx: RequestState, accepted_len: int) -> None:
        """Roll the draft's own KV back by ``γ - accepted_len`` positions.

        ``accepted_len == γ`` is a no-op (draft and target stay in sync via
        the cached tail logits). ``accepted_len < γ`` trims the draft KV
        back to the accepted prefix; the bonus token from the rejected
        position will be consumed at the next ``propose`` catch-up phase.
        """
        # ctx is part of the I-5 Protocol signature so the call shape stays
        # uniform with NoopDraftEngine; this implementation keys all draft
        # state on req_id internally and does not consult ctx here.
        del ctx
        gamma = self._last_propose_count
        if gamma == 0:
            # No drafts to commit. propose returned empty (or wasn't called
            # since the last commit). Tolerate per Protocol — commit must
            # accept any accepted_len without raising.
            return
        if accepted_len < 0 or accepted_len > gamma:
            raise ValueError(
                f"draft {self._req_id!r}: accepted_len={accepted_len} not in "
                f"[0, {gamma}] (last propose returned {gamma} drafts)"
            )
        n_reject = gamma - accepted_len
        if n_reject > 0:
            # KV rollback. Best-effort under SimpleKVCache: trimmable layers
            # shrink, recurrent (non-trimmable) layers no-op silently. Plain-
            # KV drafts are fully correct here; hybrid drafts will need
            # ``adapter.rollback_recurrent_state`` from sub-unit (e) for full
            # correctness on partial accept (sub-unit (f) parity test gates
            # the joined behaviour).
            self._kv.rollback(self._req_id, n_reject)
            self._draft_kv_pos -= n_reject
            # Tail logits no longer reflect the post-rollback state. Force
            # the next propose's catch-up branch by clearing them.
            self._cached_last_logits = None
        # last_propose_count must reset so a commit-then-commit (or
        # commit-then-empty-propose) sequence does not double-count.
        self._last_propose_count = 0

    # --- lifecycle ---

    def reset(self) -> None:
        """Release the draft's KV slot and zero the bookkeeping.

        Called by the engine on request termination. Subsequent ``propose``
        on a new request re-enters the cycle-0 branch and re-reserves the
        slot under the same ``req_id``.

        Crucially, this **trims the underlying KV cache list** (via a
        full-depth ``rollback`` issued before ``free``) rather than only
        releasing ownership. ``KVManager.free`` only flips the owner flag;
        it does not shrink the per-layer KVCache offsets. Without this
        trim, a subsequent request's cycle-0 ``prefill`` would land on top
        of the previous request's tokens in the underlying mlx-lm cache,
        producing wrong logits. ``rollback`` must run before ``free``
        because the I-2 owner check rejects rollback once ownership is
        released.

        Caveat for hybrid drafts (Qwen3.5-0.8B family): SimpleKVCache's
        rollback delegates to ``trim_prompt_cache``, which silently no-ops
        on non-trimmable recurrent caches. Plain-KV drafts (Qwen3-0.6B)
        are fully clean after ``reset``; full hybrid cleanup joins at
        sub-unit (e) when ``ModelAdapter.rollback_recurrent_state`` lands.
        """
        if self._reserved:
            if self._draft_kv_pos > 0:
                self._kv.rollback(self._req_id, self._draft_kv_pos)
            self._kv.free(self._req_id)
            self._reserved = False
        self._draft_kv_pos = 0
        self._last_propose_count = 0
        self._cached_last_logits = None

    # --- introspection (test-facing; not part of I-5) ---

    @property
    def draft_kv_pos(self) -> int:
        """Number of tokens currently in the draft's KV — for tests."""
        return self._draft_kv_pos

    @property
    def last_propose_count(self) -> int:
        """γ from the most recent propose — for tests."""
        return self._last_propose_count


__all__ = ["DraftTargetEngine"]
