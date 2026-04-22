"""silica.scheduler.budget — MemoryBudgeter (P-2 Unit #15 / #16d).

Admission + preemption **policy** layer. Returns decision objects rather
than mutating state directly — ``ContinuousBatcher`` (Unit #16) is the
single place that applies decisions. Separating decide / apply keeps
this unit testable without wiring up real caches.

**16d-1 refactor:** the budgeter originally took ``kv: PagedKVCache`` for
its ``block_size``. Under Option B (docs/P2_OPENING.md v2.1), the
batcher uses ``BatchKVCache`` not ``PagedKVCache``, and the budgeter
never actually queried kv residency at decision time — it only needed
``block_size``. We now take ``block_size: int`` directly, keep
``prefix_cache`` (still required for ``_count_evictable_prefix_blocks``),
and offer a ``for_adapter`` convenience factory so callers do not
recompute ``bytes_per_token``.

Two accountings (docs/P2_OPENING.md v2.1 §MemoryBudgeter):

  - **resident_bytes** — D-012 canonical; a non-normative reference for
    what the kv backend actually holds. Not read at decision time.
  - **reserved_bytes** — admission-time upper bound. For each admitted
    (non-terminated) request, the budgeter records
    ``(n_prompt + max_tokens) * bytes_per_token``. Conflating this with
    resident makes admission systematically over-admit; keeping the split
    is the main reason this class exists.

The ``weights`` and ``bytes_per_token`` inputs are static construction
inputs (caller knows ``ModelConfig`` + dtype). The budgeter does not
read adapter state at decision time.

Decision taxonomy:

  - ``AdmitDecision``                — accept the request as-is.
  - ``AdmitAfterEvictDecision(n)``   — accept if caller first evicts
                                       ``n`` unpinned prefix-cache blocks.
  - ``AdmitAfterPreemptDecision(req)`` — accept if caller first preempts
                                       ``req`` (a currently decoding one).
  - ``RejectDecision(reason)``       — cap cannot be satisfied even
                                       after eviction + preemption; the
                                       ContinuousBatcher then ABORTs the
                                       request with the carried reason.

Policy steps (§MemoryBudgeter Policy):

  1. Fits as-is?                                           → Admit.
  2. Fits after LRU eviction of unpinned prefix blocks?    → AdmitAfterEvict.
  3. Fits after preempting FIFO-newest DECODE request?     → AdmitAfterPreempt.
  4. None of the above.                                    → Reject.

Preempt-then-evict and preempt-twice are **not** considered in P-2 — one
admission decision may release at most one mechanism's worth of space.
That is a deliberate simplification: cascaded preemption would couple the
admission decision to the order of subsequent events, which is the class
of subtle scheduler bug P-2 is specifically trying to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from silica.kvcache.prefix import RadixPrefixCache

if TYPE_CHECKING:
    from silica.models.adapter import ModelAdapter


@dataclass(frozen=True)
class AdmitDecision:
    """Admit the incoming request; no preparatory action required."""

    reserved_delta: int
    """Bytes the caller must add to the reserved_bytes tally on apply."""


@dataclass(frozen=True)
class AdmitAfterEvictDecision:
    """Admit after evicting `n_blocks` unpinned prefix-cache leaves."""

    n_blocks: int
    reserved_delta: int


@dataclass(frozen=True)
class AdmitAfterPreemptDecision:
    """Admit after preempting a running DECODE request."""

    preempt_req_id: str
    reserved_delta: int


@dataclass(frozen=True)
class RejectDecision:
    """Cap cannot be satisfied — caller ABORTs with this reason."""

    reason: str = "budget-exhausted"


AdmissionDecision = (
    AdmitDecision
    | AdmitAfterEvictDecision
    | AdmitAfterPreemptDecision
    | RejectDecision
)


class MemoryBudgeter:
    """Budget-aware admission + preemption policy (P-2).

    Lifecycle:
      - ``admit(req_id, n_prompt, max_tokens)`` → returns AdmissionDecision.
        On any ``Admit*`` variant the caller applies the action and then
        calls ``apply_admit(req_id, reserved_delta)`` to record the
        reservation. On ``Reject`` the caller ABORTs.
      - When a request enters terminal state (DONE/ABORTED) or PREEMPTED,
        the caller calls ``release(req_id)`` to free the reservation.

    Construction inputs are static for the budgeter's lifetime:

        weights_bytes: bytes held by static model weights + activations.
        bytes_per_token: (2 * num_layers * n_kv_heads * head_dim * dtype_bytes)
                          — per-token KV bytes (one K + one V per layer).
        block_size: token count per radix block. Together with
                    bytes_per_token, gives the per-block byte cost used
                    by the evict-shortfall arithmetic.
        cap_bytes: target resident cap (default 80% of unified memory,
                   tunable via SILICA_RESIDENT_CAP_MB; resolution lives
                   in the ContinuousBatcher, not here).

    For typical callsites, see ``MemoryBudgeter.for_adapter`` which
    derives ``bytes_per_token`` from an adapter's KV layout.
    """

    def __init__(
        self,
        *,
        prefix_cache: RadixPrefixCache,
        weights_bytes: int,
        bytes_per_token: int,
        block_size: int,
        cap_bytes: int,
        account_prefix_residency: bool = True,
    ) -> None:
        """Construct a ``MemoryBudgeter``.

        ``account_prefix_residency`` (P-5-A.2, opening §4.7):

        - ``True`` (default) — new P-5 behaviour. ``headroom_bytes()``
          subtracts ``prefix_cache.store.resident_bytes()`` when the
          bound store exposes that method (synthetic stores do; paged
          stores do not — absence is treated as zero prefix residency
          via a ``hasattr`` capability check). Under ``IdentityCodec``
          this is honest fp16 bytes (§4.7 mode B); under compressed
          codecs it is honest compressed bytes (§4.7 mode C).
          Evict-shortfall math consults
          ``store.resident_bytes_per_block()`` for the same reason.
        - ``False`` — P-4.5 byte-for-byte regression path (§4.7 mode
          A). ``headroom_bytes() = cap - weights - reserved``; the
          evict branch uses the fp16 ``bytes_per_token × block_size``
          formula. Set by the P-4.5 close regression sweep and by any
          caller pinning pre-P-5 behaviour.

        Active-KV reservation (``worst_case_bytes``) remains at fp16
        worst-case under both modes — D-003 keeps the active-KV upper
        bound at fp16 regardless of codec.
        """
        if weights_bytes < 0:
            raise ValueError(
                f"weights_bytes must be >= 0, got {weights_bytes}"
            )
        if bytes_per_token <= 0:
            raise ValueError(
                f"bytes_per_token must be > 0, got {bytes_per_token}"
            )
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if cap_bytes <= 0:
            raise ValueError(f"cap_bytes must be > 0, got {cap_bytes}")
        self._pc = prefix_cache
        self._weights_bytes = weights_bytes
        self._bytes_per_token = bytes_per_token
        self._block_size = block_size
        self._cap_bytes = cap_bytes
        self._account_prefix_residency = account_prefix_residency
        # FIFO-order list of req_ids that have been admitted and not
        # released; appended on apply_admit, removed on release. Preempt
        # policy uses the tail (newest DECODE).
        self._admitted: list[str] = []
        self._reserved_per_req: dict[str, int] = {}

    @classmethod
    def for_adapter(
        cls,
        adapter: "ModelAdapter",
        *,
        prefix_cache: RadixPrefixCache,
        weights_bytes: int,
        cap_bytes: int,
        account_prefix_residency: bool = True,
    ) -> "MemoryBudgeter":
        """Construct a ``MemoryBudgeter`` for a concrete ``ModelAdapter``.

        Derives ``bytes_per_token`` from the adapter's
        ``kv_layout()``. Two paths:

        - ``layout.bytes_per_token_total`` set (P-3-D4):
          used verbatim. Adapters with heterogeneous per-layer KV
          shapes (e.g. Gemma4's sliding 16×256 + full 4×512 mix)
          populate this field from a per-kind sum so the naive
          ``num_layers × n_kv_heads × head_dim`` formula does not
          systematically over-count.
        - ``layout.bytes_per_token_total`` left at ``None``: one K
          plus one V per layer, ``n_kv_heads`` heads, ``head_dim``
          each, at the layout's dtype — the pre-D4 formula,
          correct for homogeneous-shape models (plain Qwen3,
          Qwen3.5 dense).

        ``block_size`` comes from ``prefix_cache.block_size`` — the
        radix cache is the authority on block granularity and keeping
        it single-sourced avoids drift.

        Intended for callers that would otherwise recompute this
        arithmetic at every test / harness site. The batcher still
        takes an explicit ``budgeter`` kwarg — this factory does not
        remove the ownership boundary (16d non-goal).
        """
        layout = adapter.kv_layout()
        if layout.bytes_per_token_total is not None:
            bytes_per_token = layout.bytes_per_token_total
        else:
            bytes_per_token = (
                2  # one K, one V
                * layout.num_layers
                * layout.n_kv_heads
                * layout.head_dim
                * layout.dtype.size
            )
        return cls(
            prefix_cache=prefix_cache,
            weights_bytes=weights_bytes,
            bytes_per_token=bytes_per_token,
            block_size=prefix_cache.block_size,
            cap_bytes=cap_bytes,
            account_prefix_residency=account_prefix_residency,
        )

    # --- inspection ---

    @property
    def cap_bytes(self) -> int:
        return self._cap_bytes

    @property
    def weights_bytes(self) -> int:
        return self._weights_bytes

    @property
    def bytes_per_token(self) -> int:
        return self._bytes_per_token

    def reserved_bytes(self) -> int:
        return sum(self._reserved_per_req.values())

    def headroom_bytes(self) -> int:
        """Bytes available for a new admission (can be negative).

        Under ``account_prefix_residency=True`` (default), subtracts
        prefix-cache residency in addition to weights + reserved:
        ``headroom = cap - weights - reserved - prefix_resident``.
        Under ``account_prefix_residency=False``, falls back to the
        P-4.5 byte-for-byte formula ``cap - weights - reserved`` (§4.7
        mode A regression path).

        Prefix residency is read via a structural capability check
        (``hasattr(store, 'resident_bytes')``) — synthetic stores
        expose it, paged stores do not (paged-backend residency is
        not modelled in v0.1 per D-003 / Q-009 / R-7); paged path
        treats absence as zero residency.
        """
        return (
            self._cap_bytes
            - self._weights_bytes
            - self.reserved_bytes()
            - self._prefix_resident_bytes()
        )

    def _prefix_resident_bytes(self) -> int:
        """Prefix-cache residency consumed by the bound store.

        Zero under ``account_prefix_residency=False`` OR when the
        bound store does not expose ``resident_bytes()`` (paged
        backend). Otherwise returns the store's honest per-side
        payload-byte sum (fp16 under IdentityCodec, compressed under
        BlockTQ / RaBitQ).
        """
        if not self._account_prefix_residency:
            return 0
        store = self._pc.store if self._pc is not None else None
        if store is None:
            return 0
        fn = getattr(store, "resident_bytes", None)
        if not callable(fn):
            return 0
        return int(fn())

    def worst_case_bytes(self, n_prompt: int, max_tokens: int) -> int:
        """Upper-bound KV bytes for a request with this shape."""
        if n_prompt < 0 or max_tokens < 0:
            raise ValueError(
                f"n_prompt and max_tokens must be >= 0, "
                f"got {n_prompt}, {max_tokens}"
            )
        return (n_prompt + max_tokens) * self._bytes_per_token

    # --- decision ---

    def admit(
        self,
        req_id: str,
        n_prompt: int,
        max_tokens: int,
    ) -> AdmissionDecision:
        """Compute an admission decision for one incoming request.

        Pure — does not mutate budgeter state. Caller applies the
        decision then calls ``apply_admit``.
        """
        if req_id in self._reserved_per_req:
            raise ValueError(f"{req_id!r} already admitted")

        new_worst = self.worst_case_bytes(n_prompt, max_tokens)
        fit_headroom = self.headroom_bytes()

        # (1) Fits as-is.
        if new_worst <= fit_headroom:
            return AdmitDecision(reserved_delta=new_worst)

        # Shortfall is how many bytes we need to free to admit.
        shortfall = new_worst - fit_headroom

        # (2) Try evicting unpinned prefix-cache source blocks.
        evictable_blocks = self._count_evictable_prefix_blocks()
        block_bytes = self._evict_bytes_per_block()
        if evictable_blocks > 0 and block_bytes > 0:
            blocks_needed = (shortfall + block_bytes - 1) // block_bytes
            if blocks_needed <= evictable_blocks:
                return AdmitAfterEvictDecision(
                    n_blocks=blocks_needed,
                    reserved_delta=new_worst,
                )

        # (3) Try preempting the FIFO-newest DECODE request.
        victim = self._pick_preempt_victim()
        if victim is not None:
            freed = self._reserved_per_req[victim]
            # After preempt, headroom grows by freed bytes and (optionally)
            # by whatever evictable prefix blocks still remain. For the
            # P-2 "at most one mechanism" rule we consider preempt alone.
            if new_worst <= fit_headroom + freed:
                return AdmitAfterPreemptDecision(
                    preempt_req_id=victim,
                    reserved_delta=new_worst,
                )

        # (4) Nothing can be done.
        return RejectDecision()

    # --- apply / release ---

    def apply_admit(self, req_id: str, reserved_delta: int) -> None:
        """Record an admission after the caller has applied the decision.

        Call after ``admit`` returned any ``Admit*`` variant, once the
        caller has executed the prerequisite (evict / preempt) and the
        request has been reserved in the kv manager. Idempotent if called
        twice with the same req_id — it raises.
        """
        if req_id in self._reserved_per_req:
            raise ValueError(
                f"{req_id!r} already has an active reservation"
            )
        if reserved_delta < 0:
            raise ValueError(
                f"reserved_delta must be >= 0, got {reserved_delta}"
            )
        self._reserved_per_req[req_id] = reserved_delta
        self._admitted.append(req_id)

    def release(self, req_id: str) -> None:
        """Release ``req_id``'s reservation (on DONE / ABORTED / PREEMPTED).

        Idempotent on unknown req_id (parallels ``PagedKVCache.free``).
        """
        if req_id not in self._reserved_per_req:
            return
        del self._reserved_per_req[req_id]
        self._admitted.remove(req_id)

    def active_requests(self) -> list[str]:
        """Admitted, non-released req_ids in admission order (FIFO)."""
        return list(self._admitted)

    # --- internals ---

    def _kv_bytes_per_block(self) -> int:
        """fp16-baseline bytes per radix block.

        ``bytes_per_token * block_size`` — matches ``PagedKVCache``'s
        internal accounting. Used as the fp16-worst-case fallback in
        :meth:`_evict_bytes_per_block` when no store-level per-block
        residency is exposed.
        """
        return self._bytes_per_token * self._block_size

    def _evict_bytes_per_block(self) -> int:
        """Actual bytes freed by evicting one prefix-cache block.

        Under ``account_prefix_residency=True`` (default), consults
        the bound store's ``resident_bytes_per_block()`` — honest
        per-block residency reflecting codec compression. Falls back
        to the fp16 ``bytes_per_token × block_size`` baseline when
        the flag is off, when no store is bound, when the store does
        not expose the method (paged backend), or when the method
        returns ``None`` (synthetic pass-through path — no codec
        installed, per-block raw size not symbolically tracked).

        The evict-shortfall math in :meth:`admit` divides
        ``shortfall_bytes`` by this value to get ``blocks_needed``;
        using honest per-block bytes under a compressed codec
        reflects real bytes-freed rather than the fp16 estimate. This
        is §4.7 bullet 3 of the opening doc. Reservation math
        (``worst_case_bytes``) still uses the fp16 formula — D-003
        keeps the active-KV upper bound at fp16 regardless of codec.
        """
        if not self._account_prefix_residency:
            return self._kv_bytes_per_block()
        store = self._pc.store if self._pc is not None else None
        if store is None:
            return self._kv_bytes_per_block()
        fn = getattr(store, "resident_bytes_per_block", None)
        if not callable(fn):
            return self._kv_bytes_per_block()
        per_block = fn()
        if per_block is None:
            return self._kv_bytes_per_block()
        return int(per_block)

    def _count_evictable_prefix_blocks(self) -> int:
        """Best-effort count of leaf blocks with zero live hits.

        Walks the prefix cache's internal nodes — uses the private
        debug helpers exposed by ``RadixPrefixCache``. A slight
        encapsulation break, but confined to this one method: if the
        prefix cache ever grows a public `evictable_block_count()` the
        change is a one-line swap here.
        """
        count = 0
        for node in self._pc._walk_non_root():  # noqa: SLF001
            if node.children:
                continue
            if self._pc.live_hits(node.block_id) > 0:
                continue
            count += 1
        return count

    def _pick_preempt_victim(self) -> str | None:
        """Return the FIFO-newest active request, or None if none."""
        if not self._admitted:
            return None
        return self._admitted[-1]
