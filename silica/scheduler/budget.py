"""silica.scheduler.budget — MemoryBudgeter (P-2 Unit #15).

Admission + preemption **policy** layer. Returns decision objects rather
than mutating state directly — ``ContinuousBatcher`` (Unit #16) is the
single place that applies decisions against ``PagedKVCache`` and
``RadixPrefixCache``. Separating decide / apply keeps this unit testable
without wiring up real caches.

Two accountings (docs/P2_OPENING.md v2.1 §MemoryBudgeter):

  - **resident_bytes** — D-012 canonical; read from ``PagedKVCache.budget``
    each query. Reflects reality.
  - **reserved_bytes** — admission-time upper bound. For each admitted
    (non-terminated) request, the budgeter records
    ``(n_prompt + max_tokens) * bytes_per_token``. Conflating this with
    resident makes admission systematically over-admit; keeping the split
    is the main reason this class exists.

The ``weights`` and ``bytes_per_token`` inputs are provided by the
ContinuousBatcher at construction time (it knows ``ModelConfig`` + quant
profile). The budgeter does not read adapter state.

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

from silica.kvcache.paged import PagedKVCache
from silica.kvcache.prefix import RadixPrefixCache


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
        cap_bytes: target resident cap (default 80% of unified memory,
                   tunable via SILICA_RESIDENT_CAP_MB; resolution lives
                   in the ContinuousBatcher, not here).
    """

    def __init__(
        self,
        *,
        kv: PagedKVCache,
        prefix_cache: RadixPrefixCache,
        weights_bytes: int,
        bytes_per_token: int,
        cap_bytes: int,
    ) -> None:
        if weights_bytes < 0:
            raise ValueError(
                f"weights_bytes must be >= 0, got {weights_bytes}"
            )
        if bytes_per_token <= 0:
            raise ValueError(
                f"bytes_per_token must be > 0, got {bytes_per_token}"
            )
        if cap_bytes <= 0:
            raise ValueError(f"cap_bytes must be > 0, got {cap_bytes}")
        self._kv = kv
        self._pc = prefix_cache
        self._weights_bytes = weights_bytes
        self._bytes_per_token = bytes_per_token
        self._cap_bytes = cap_bytes
        # FIFO-order list of req_ids that have been admitted and not
        # released; appended on apply_admit, removed on release. Preempt
        # policy uses the tail (newest DECODE).
        self._admitted: list[str] = []
        self._reserved_per_req: dict[str, int] = {}

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
        """Cap minus weights minus current reservation (can be negative)."""
        return (
            self._cap_bytes
            - self._weights_bytes
            - self.reserved_bytes()
        )

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
        block_bytes = self._kv_bytes_per_block()
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
        """Infer bytes-per-block from the kv manager's accounting.

        We compute via a hypothetical reserve: since PagedKVCache's
        bytes_per_block is private, we use the ratio
        ``block_bytes = bytes_per_token * block_size``. That matches how
        ``PagedKVCache`` calculates it internally.
        """
        return self._bytes_per_token * self._kv.block_size

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
