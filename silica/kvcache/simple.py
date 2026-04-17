"""silica.kvcache.simple — I-2 KVManager for P-1 single-request decode (D-010).

Minimal implementation of the I-2 ``KVManager`` Protocol that adapts mlx-lm's
heterogeneous per-layer cache list for the P-1 single-request text-only decode
loop (PLAN.md §7 P-1, D-010, D-014).

Design constraints:
  - **Single request only.** ``req_id`` is carried to satisfy the Protocol; any
    attempt to claim a second ``req_id`` while one is held raises ValueError.
    This turns the "single request" assumption from documentation into a
    loud-failure (D-011) invariant.
  - **No paging, no prefix cache.** ``reserve_for_prefill`` and ``append_slot``
    return an empty ``BlockList`` — mlx-lm's ``KVCache`` auto-grows in 256-token
    chunks, and P-1's decode loop does not need block-level accounting.
  - **Heterogeneous layer dispatch preserved.** For Qwen3.5 the per-layer list
    is 18 ``ArraysCache`` (DeltaNet) + 6 ``KVCache`` (full attention) matching
    the ``HYBRID_DELTANET`` attention pattern (D-015); SimpleKVCache stores the
    list as produced by the model's ``make_cache()`` and hands it through
    untouched.
  - **Rollback delegates to mlx-lm.** ``rollback(n)`` calls
    ``mlx_lm.models.cache.trim_prompt_cache``; it is best-effort — entries that
    are not ``is_trimmable()`` (e.g. some recurrent caches) silently no-op,
    which is the correct behaviour for P-1 (speculative decode is a P-7 item).
  - **Resident bytes aggregate nbytes.** Per the Gate A finding (D-012), mlx-lm
    already exposes ``nbytes`` as the canonical measurement; no reconciliation.

P-2 replaces this class with ``PagedKVCache``; the extension method
``cache_list()`` will be dropped because P-2's adapter works from
``BlockList`` + the page table, not from an mlx-lm-shaped per-layer list.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mlx_lm.models import cache as mlx_cache

from silica.kvcache.manager import BlockList, MemoryBudget, PrefixHit

# Matches ``mlx_lm.models.cache.KVCache.step`` — the chunk size mlx-lm's
# auto-growing KVCache allocates at a time. Not a block size in the paged
# sense; kept here only to give ``block_size`` a concrete value that reflects
# the real allocation granularity.
_MLX_KVCACHE_GROWTH_CHUNK = 256


class SimpleKVCache:
    """P-1 single-request I-2 KVManager over mlx-lm's per-layer cache list."""

    block_size: int

    def __init__(self, cache_list: list[Any]) -> None:
        self._cache_list = cache_list
        self._owner: str | None = None
        self.block_size = _MLX_KVCACHE_GROWTH_CHUNK

    @classmethod
    def from_model(cls, model: Any) -> SimpleKVCache:
        """Build a SimpleKVCache from mlx-lm's per-model cache factory.

        Delegates to ``mlx_lm.models.cache.make_prompt_cache(model)``, which
        calls ``model.make_cache()`` if the model defines one (Qwen3.5 returns
        the 18+6 hybrid list) and otherwise returns ``[KVCache()] * num_layers``.
        """
        return cls(mlx_cache.make_prompt_cache(model))

    def cache_list(self, req_id: str) -> list[Any]:
        """Return the mlx-lm per-layer cache list for the owning request.

        Extension beyond the I-2 Protocol: the P-1 adapter calls this through
        the concrete ``SimpleKVCache`` type to hand the list to
        ``model(tokens, cache=...)``. P-2's paged cache has no equivalent.
        """
        self._require_owner(req_id)
        return self._cache_list

    # --- I-2 KVManager Protocol surface ---

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        self._claim(req_id)
        return BlockList()

    def append_slot(self, req_id: str, n: int) -> BlockList:
        self._require_owner(req_id)
        return BlockList()

    def commit(self, req_id: str, n_accepted: int) -> None:
        self._require_owner(req_id)

    def rollback(self, req_id: str, n_reject: int) -> None:
        self._require_owner(req_id)
        if n_reject <= 0:
            return
        if mlx_cache.can_trim_prompt_cache(self._cache_list):
            mlx_cache.trim_prompt_cache(self._cache_list, n_reject)

    def free(self, req_id: str) -> None:
        self._require_owner(req_id)
        self._owner = None

    def get_computed_blocks(self, token_ids: Sequence[int]) -> PrefixHit:
        return PrefixHit()

    def available_blocks(self) -> int:
        # SimpleKVCache is non-paged; mlx-lm's KVCache auto-grows to whatever
        # fits in unified memory. Block accounting is not meaningful here.
        # Report zero to distinguish from a paged manager that has idle
        # capacity to allocate.
        return 0

    def budget(self) -> MemoryBudget:
        resident = sum(
            int(c.nbytes) for c in self._cache_list if hasattr(c, "nbytes")
        )
        return MemoryBudget(
            logical_bytes=resident,
            resident_bytes=resident,
            headroom_bytes=0,
        )

    # --- single-request ownership discipline ---

    def _claim(self, req_id: str) -> None:
        if self._owner is not None and self._owner != req_id:
            raise ValueError(
                f"SimpleKVCache is single-request; already owned by "
                f"{self._owner!r}, cannot accept {req_id!r}"
            )
        self._owner = req_id

    def _require_owner(self, req_id: str) -> None:
        if self._owner != req_id:
            raise ValueError(
                f"req_id mismatch: SimpleKVCache owned by {self._owner!r}, "
                f"got {req_id!r}"
            )
