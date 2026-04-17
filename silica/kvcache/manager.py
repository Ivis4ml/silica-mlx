"""silica.kvcache.manager — I-2 KVManager Protocol + auxiliary types + NullKVManager.

I-2 (PLAN.md §6) owns paged / block KV, the prefix cache, the memory budget,
and the incremental mutation primitives required by continuous batching and
speculative decoding (commit / rollback).

Auxiliary types:
  - KVHandle: opaque binding of req_id so ModelAdapter can reach KV without
    holding block pointers directly (I-1 key constraint #2).
  - BlockList: ordered block ids belonging to one request.
  - PrefixHit: return of get_computed_blocks — reusable blocks + hit length.
  - MemoryBudget: Principle 8 snapshot; both logical (fp16-equivalent) and
    resident (D-012-canonical) views.

P-0 `NullKVManager` stub: zero-sized, no-op. Real allocation lands in P-1's
SimpleKVCache (single request) and P-2's PagedKVCache (paged + prefix cache).
The NullKVManager exists only to satisfy the I-2 shape for test_interfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class KVHandle:
    """Binding of a request id to its KV manager.

    The handle is issued by `KVManager.reserve_for_prefill` (via the engine
    / scheduler plumbing) and carried through `ModelAdapter.prefill` /
    `decode_step` calls so the adapter can read / write KV via the manager
    without ever holding block pointers directly (I-1 key constraint #2).

    Frozen so the adapter cannot re-point a handle to a different request.
    """

    req_id: str


@dataclass
class BlockList:
    """Ordered list of KV block ids for one request."""

    block_ids: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.block_ids)


@dataclass
class PrefixHit:
    """Result of `KVManager.get_computed_blocks` (prefix cache lookup).

    `block_ids` are the already-computed blocks the scheduler can reuse and
    pin on admission; `num_hit_tokens` is how many of the query tokens are
    covered. `num_hit_tokens == 0` and empty `block_ids` == cache miss.
    """

    block_ids: tuple[int, ...] = ()
    num_hit_tokens: int = 0


@dataclass
class MemoryBudget:
    """Memory-budget snapshot consumed by the scheduler's admission logic.

    `logical_bytes`: fp16-baseline equivalent (what fp16 KV would cost).
    `resident_bytes`: actual physical bytes held (D-012 canonical).
    `headroom_bytes`: `target_resident_bytes - resident_bytes`; negative
    values mean over-budget.
    """

    logical_bytes: int = 0
    resident_bytes: int = 0
    headroom_bytes: int = 0


@runtime_checkable
class KVManager(Protocol):
    """Block-allocated KV cache + prefix cache + memory budget (I-2)."""

    block_size: int

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList: ...

    def append_slot(self, req_id: str, n: int) -> BlockList: ...

    def commit(self, req_id: str, n_accepted: int) -> None: ...

    def rollback(self, req_id: str, n_reject: int) -> None: ...

    def free(self, req_id: str) -> None: ...

    def get_computed_blocks(self, token_ids: Sequence[int]) -> PrefixHit: ...

    def available_blocks(self) -> int: ...

    def budget(self) -> MemoryBudget: ...


class NullKVManager:
    """Empty KV manager satisfying I-2 — all allocation operations return
    empty block lists, `budget()` reports zeros, `available_blocks()` is 0.

    Used only for P-0 interface conformance. P-1 replaces with
    `SimpleKVCache` (single-request); P-2 replaces with `PagedKVCache`
    (paged + prefix cache + memory budgeter).
    """

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = block_size

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        return BlockList()

    def append_slot(self, req_id: str, n: int) -> BlockList:
        return BlockList()

    def commit(self, req_id: str, n_accepted: int) -> None:
        return None

    def rollback(self, req_id: str, n_reject: int) -> None:
        return None

    def free(self, req_id: str) -> None:
        return None

    def get_computed_blocks(self, token_ids: Sequence[int]) -> PrefixHit:
        return PrefixHit()

    def available_blocks(self) -> int:
        return 0

    def budget(self) -> MemoryBudget:
        return MemoryBudget()
