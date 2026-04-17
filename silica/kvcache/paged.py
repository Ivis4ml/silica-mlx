"""silica.kvcache.paged — I-2 KVManager for P-2 multi-request (Unit #13).

Paged bookkeeping container for P-2 ``ContinuousBatcher``. Tracks, per batch
session:

  - **Layer A (physical slot table).** Each request occupies one row in the
    batched K / V tensor for its entire lifetime. Row assignment is
    ``slot_table: req_id → int`` with lifecycle tracked in ``row_state``.
    Row *fixed for lifetime*, no compaction (P-2 design, see
    ``docs/P2_OPENING.md`` Layer A invariant).
  - **Layer B (logical page table).** Over the rows we lay a logical block
    map: ``page_table: req_id → list[block_id]`` plus a shared
    ``refcount`` and ``free_blocks`` list. Block ids are abstract
    handles — in P-2 they name the block for prefix-cache source-retention
    purposes, not physical memory (no paged-attention kernel, per Q-009).

What this unit does *not* own:

  - Physical ``BatchKVCache`` construction — that requires per-row
    ``left_padding`` which only the batcher knows. Deferred to Unit #16
    (ContinuousBatcher).
  - ``get_computed_blocks`` returns an empty ``PrefixHit`` until Unit #14
    (RadixPrefixCache) plugs in.
  - Speculative commit / rollback at the physical layer. The Protocol
    methods validate the request exists but do not touch K / V; the
    batcher handles physical trim via mlx-lm's primitives at P-7.

Refcount semantics mirror the opening doc: a block's refcount counts
**retention sources** (owning request + any prefix-cache references). The
block returns to ``free_blocks`` only when the count reaches zero — this
is the load-bearing invariant for option B prefix reuse (copy-on-admit),
since a prefix-cache incref before the owning request's ``free`` call
keeps the source alive for later copy.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence

from silica.kvcache.manager import BlockList, MemoryBudget, PrefixHit


class RowState(str, enum.Enum):
    """Lifecycle of one batch row in the physical cache.

    FREE → RESERVED (on ``reserve_for_prefill``) → ACTIVE (on ``mark_active``
    after prefill completes) → FREE (on ``free``). Rows do not return to
    RESERVED; a new admission claims a different FREE row.
    """

    FREE = "free"
    RESERVED = "reserved"
    ACTIVE = "active"


class PagedKVCache:
    """P-2 I-2 KVManager — paged bookkeeping over a batched K / V store.

    All operations are O(1) or O(block_count_for_request); no cross-request
    coupling. Designed so ``MemoryBudgeter`` (Unit #15) can query
    ``available_blocks`` / ``budget`` and ``ContinuousBatcher`` (Unit #16)
    can drive the state transitions.
    """

    block_size: int

    def __init__(
        self,
        *,
        num_layers: int,
        max_batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int = 16,
        dtype_bytes: int = 2,
    ) -> None:
        """Construct the bookkeeping state.

        Args:
            num_layers: Total transformer layers (all contribute to bytes
                per block — vLLM convention where a block id names the
                same logical block across every layer).
            max_batch_size: Fixed batch axis size of the physical cache;
                a request claims one of these rows for its lifetime.
            n_kv_heads: Per-layer number of KV heads (after any
                grouped-query folding).
            head_dim: Per-head dimension of K / V.
            num_blocks: Total size of the logical block pool.
            block_size: Tokens per block; PLAN default 16.
            dtype_bytes: Bytes per K / V element (2 for fp16, 1 for int8).
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be > 0, got {max_batch_size}"
            )
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")

        self.block_size = block_size
        self._num_layers = num_layers
        self._max_batch_size = max_batch_size
        self._num_blocks = num_blocks
        # 2 = K + V; applies across all layers per block id (see class docstring).
        self._bytes_per_block = (
            2 * num_layers * n_kv_heads * head_dim * block_size * dtype_bytes
        )

        # Layer A — physical slot bookkeeping.
        self._slot_table: dict[str, int] = {}
        self._row_state: list[RowState] = [
            RowState.FREE for _ in range(max_batch_size)
        ]

        # Layer B — logical page bookkeeping.
        self._page_table: dict[str, list[int]] = {}
        self._refcount: dict[int, int] = {}
        self._free_blocks: list[int] = list(range(num_blocks))

        # Per-request token count (drives append_slot block math).
        self._num_tokens: dict[str, int] = {}

    # --- extension surface used by ContinuousBatcher / RadixPrefixCache ---

    def slot_of(self, req_id: str) -> int:
        """Batch row index for ``req_id``.

        Raises ``KeyError`` if the request has not been reserved.
        """
        if req_id not in self._slot_table:
            raise KeyError(f"no slot for req_id {req_id!r}")
        return self._slot_table[req_id]

    def row_states(self) -> list[RowState]:
        """Snapshot of the physical row lifecycle array (defensive copy)."""
        return list(self._row_state)

    def mark_active(self, req_id: str) -> None:
        """Transition the request's row from RESERVED to ACTIVE.

        Called by the batcher once prefill finishes and decode begins.
        The RESERVED → ACTIVE split lets the batcher distinguish
        "reserved but not yet written" from "participating in decode".
        """
        self._require_reserved(req_id)
        self._row_state[self._slot_table[req_id]] = RowState.ACTIVE

    def incref(self, block_id: int) -> None:
        """Add a retention reference to ``block_id``.

        Extension for ``RadixPrefixCache``: the prefix cache incref's a
        block before the owning request's ``free`` so the block survives
        as a copy source (option B prefix reuse).
        """
        if block_id not in self._refcount:
            raise KeyError(
                f"block {block_id} is not currently held by any request; "
                f"cannot incref from the free pool"
            )
        self._refcount[block_id] += 1

    def decref(self, block_id: int) -> None:
        """Drop a retention reference to ``block_id``.

        If the refcount drops to zero, the block returns to
        ``free_blocks``. Extension for ``RadixPrefixCache`` paired with
        ``incref``.
        """
        if block_id not in self._refcount:
            raise KeyError(f"block {block_id} has no outstanding refs")
        self._refcount[block_id] -= 1
        if self._refcount[block_id] <= 0:
            del self._refcount[block_id]
            self._free_blocks.append(block_id)

    def num_tokens(self, req_id: str) -> int:
        """Total tokens (prompt + decoded) held for ``req_id``."""
        if req_id not in self._num_tokens:
            raise KeyError(f"no token count for req_id {req_id!r}")
        return self._num_tokens[req_id]

    # --- I-2 KVManager Protocol surface ---

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        """Allocate a slot + logical blocks for ``req_id``'s prefill.

        The request enters RowState.RESERVED. Each allocated block's
        refcount starts at 1 (owned by this request). Raises
        ``ValueError`` on duplicate ``req_id``; raises ``RuntimeError``
        if no free slot or insufficient free blocks.
        """
        if req_id in self._slot_table:
            raise ValueError(f"{req_id!r} already reserved")

        n_blocks = self._blocks_needed(len(token_ids))
        if len(self._free_blocks) < n_blocks:
            raise RuntimeError(
                f"{req_id!r}: not enough free blocks "
                f"(need {n_blocks}, have {len(self._free_blocks)})"
            )
        slot = self._allocate_slot()
        if slot is None:
            raise RuntimeError(
                f"{req_id!r}: no free slots (max_batch_size="
                f"{self._max_batch_size}, all in use)"
            )

        # Both resources available; commit the allocation.
        self._slot_table[req_id] = slot
        self._row_state[slot] = RowState.RESERVED
        blocks = self._take_free_blocks(n_blocks)
        self._page_table[req_id] = blocks
        for b in blocks:
            self._refcount[b] = 1
        self._num_tokens[req_id] = len(token_ids)
        return BlockList(block_ids=tuple(blocks))

    def append_slot(self, req_id: str, n: int) -> BlockList:
        """Extend ``req_id``'s page table to cover ``n`` additional tokens.

        Returns the BlockList of *newly* allocated block ids (empty if
        the current last block still has room for ``n`` more tokens).
        Raises ``KeyError`` if the request is not reserved.
        """
        self._require_reserved(req_id)
        new_total = self._num_tokens[req_id] + n
        current_blocks = len(self._page_table[req_id])
        needed_blocks = self._blocks_needed(new_total)
        extra = needed_blocks - current_blocks
        newly_allocated: list[int] = []
        if extra > 0:
            if len(self._free_blocks) < extra:
                raise RuntimeError(
                    f"{req_id!r}: append_slot cannot allocate "
                    f"{extra} more blocks (have {len(self._free_blocks)})"
                )
            newly_allocated = self._take_free_blocks(extra)
            self._page_table[req_id].extend(newly_allocated)
            for b in newly_allocated:
                self._refcount[b] = 1
        self._num_tokens[req_id] = new_total
        return BlockList(block_ids=tuple(newly_allocated))

    def commit(self, req_id: str, n_accepted: int) -> None:
        """Speculative accept (P-7 forward-compat; bookkeeping no-op in P-2)."""
        self._require_reserved(req_id)

    def rollback(self, req_id: str, n_reject: int) -> None:
        """Speculative reject (P-7 forward-compat; bookkeeping no-op in P-2)."""
        self._require_reserved(req_id)

    def free(self, req_id: str) -> None:
        """Release ``req_id``'s slot and decrement each of its block refs.

        Idempotent — freeing an unknown req_id is a no-op (mirrors
        NullKVManager convention; lets the batcher call ``free`` in a
        finally block without guarding). Blocks whose total refcount
        drops to zero return to the free pool; blocks pinned by the
        prefix cache survive.
        """
        if req_id not in self._slot_table:
            return
        slot = self._slot_table[req_id]
        self._row_state[slot] = RowState.FREE
        for b in self._page_table[req_id]:
            self.decref(b)
        del self._slot_table[req_id]
        del self._page_table[req_id]
        del self._num_tokens[req_id]

    def get_computed_blocks(
        self, token_ids: Sequence[int]
    ) -> PrefixHit:
        """Prefix-cache lookup. Unit #13 returns empty; Unit #14 injects
        ``RadixPrefixCache`` at a later attachment point."""
        return PrefixHit()

    def available_blocks(self) -> int:
        return len(self._free_blocks)

    def budget(self) -> MemoryBudget:
        """Residency approximated from block allocation × per-block bytes.

        This counts *claimed* blocks, not *used* bytes within them — a
        block with one token written still reports a full block of
        residency. That matches the scheduler's worst-case view (the
        same block cannot be used for anyone else). ``logical_bytes``
        and ``resident_bytes`` are equal in P-2; ``headroom_bytes`` is 0
        because P-2 doesn't track a target cap at this layer (that lives
        in ``MemoryBudgeter``).
        """
        n_claimed = self._num_blocks - len(self._free_blocks)
        resident = n_claimed * self._bytes_per_block
        return MemoryBudget(
            logical_bytes=resident,
            resident_bytes=resident,
            headroom_bytes=0,
        )

    # --- internals ---

    def _blocks_needed(self, n_tokens: int) -> int:
        if n_tokens <= 0:
            return 0
        return (n_tokens + self.block_size - 1) // self.block_size

    def _allocate_slot(self) -> int | None:
        for i, state in enumerate(self._row_state):
            if state == RowState.FREE:
                return i
        return None

    def _take_free_blocks(self, n: int) -> list[int]:
        taken = self._free_blocks[:n]
        self._free_blocks = self._free_blocks[n:]
        return taken

    def _require_reserved(self, req_id: str) -> None:
        if req_id not in self._slot_table:
            raise KeyError(f"{req_id!r} not reserved")
