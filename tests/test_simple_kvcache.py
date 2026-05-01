"""Tests for silica.kvcache.simple — P-1 single-request I-2 KVManager (D-010).

Covers:
  - I-2 Protocol shape (runtime_checkable isinstance).
  - Factory `SimpleKVCache.from_model` drives mlx-lm's make_prompt_cache.
  - Single-request ownership discipline (second req_id raises, free releases).
  - Heterogeneous per-layer list is preserved (ArraysCache + KVCache mix).
  - budget.resident_bytes aggregates nbytes (D-012 alignment).
  - rollback walks per-layer; trimmable entries trim, non-trimmable pass through.
  - cache_list extension method requires owner match.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from silica.kvcache.manager import BlockList, KVManager, MemoryBudget, PrefixHit
from silica.kvcache.simple import SimpleKVCache


def _fresh_kv_list() -> list[Any]:
    """Build a heterogeneous per-layer list like Qwen3.5 (linear + full)."""
    return [KVCache(), KVCache(), ArraysCache(size=2)]


def _feed_kvcache(entry: KVCache, n_tokens: int) -> None:
    """Drive KVCache.update_and_fetch so it has real resident bytes."""
    k = mx.zeros((1, 1, n_tokens, 4), dtype=mx.float16)
    v = mx.zeros((1, 1, n_tokens, 4), dtype=mx.float16)
    entry.update_and_fetch(k, v)


# --- I-2 Protocol shape ---


def test_satisfies_kv_manager_protocol() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    assert isinstance(kv, KVManager)


def test_exposes_block_size_attribute() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    assert kv.block_size == 256  # matches mlx_lm.models.cache.KVCache.step


# --- factory ---


def test_from_model_uses_mlx_lm_factory() -> None:
    """A toy model with `make_cache` drives the factory path."""

    class _ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [nn.Linear(4, 4), nn.Linear(4, 4)]

        def make_cache(self) -> list[Any]:
            return [KVCache(), KVCache()]

    kv = SimpleKVCache.from_model(_ToyModel())
    kv.reserve_for_prefill("req-a", [])
    per_layer = kv.cache_list("req-a")
    assert len(per_layer) == 2
    assert all(isinstance(c, KVCache) for c in per_layer)


# --- ownership discipline (single request) ---


def test_reserve_claims_request() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    kv.reserve_for_prefill("req-a", [1, 2, 3])
    # Second claim with the same id is idempotent.
    kv.reserve_for_prefill("req-a", [4, 5])


def test_second_request_id_raises() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    kv.reserve_for_prefill("req-a", [1, 2, 3])
    with pytest.raises(ValueError, match="single-request"):
        kv.reserve_for_prefill("req-b", [4])


def test_free_releases_ownership() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    kv.reserve_for_prefill("req-a", [1])
    kv.free("req-a")
    # A new request can now claim.
    kv.reserve_for_prefill("req-b", [2])


def test_append_slot_requires_owner() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    with pytest.raises(ValueError, match="req_id mismatch"):
        kv.append_slot("not-claimed", 1)


def test_cache_list_requires_owner() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    with pytest.raises(ValueError, match="req_id mismatch"):
        kv.cache_list("not-claimed")


def test_commit_requires_owner() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    with pytest.raises(ValueError, match="req_id mismatch"):
        kv.commit("not-claimed", 1)


# --- block-list semantics (single-request, non-paged) ---


def test_reserve_for_prefill_returns_empty_block_list() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    out = kv.reserve_for_prefill("req-a", [1, 2, 3])
    assert isinstance(out, BlockList)
    assert len(out) == 0


def test_append_slot_returns_empty_block_list() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    kv.reserve_for_prefill("req-a", [1])
    out = kv.append_slot("req-a", 1)
    assert isinstance(out, BlockList)
    assert len(out) == 0


def test_get_computed_blocks_is_miss() -> None:
    kv = SimpleKVCache(_fresh_kv_list())
    hit = kv.get_computed_blocks([1, 2, 3])
    assert isinstance(hit, PrefixHit)
    assert hit.num_hit_tokens == 0


def test_available_blocks_is_zero() -> None:
    assert SimpleKVCache(_fresh_kv_list()).available_blocks() == 0


# --- heterogeneous per-layer list preserved ---


def test_cache_list_preserves_heterogeneous_layout() -> None:
    original = _fresh_kv_list()
    kv = SimpleKVCache(original)
    kv.reserve_for_prefill("req-a", [1])
    per_layer = kv.cache_list("req-a")
    assert per_layer is original
    assert [type(c).__name__ for c in per_layer] == [
        "KVCache",
        "KVCache",
        "ArraysCache",
    ]


# --- budget (D-012 resident_bytes from nbytes) ---


def test_budget_on_empty_cache_is_zero() -> None:
    kv = SimpleKVCache([KVCache(), KVCache()])
    b = kv.budget()
    assert isinstance(b, MemoryBudget)
    assert b.resident_bytes == 0


def test_budget_aggregates_nbytes_after_fill() -> None:
    entries = [KVCache(), KVCache()]
    _feed_kvcache(entries[0], n_tokens=4)
    _feed_kvcache(entries[1], n_tokens=4)
    kv = SimpleKVCache(entries)
    b = kv.budget()
    expected = sum(int(c.nbytes) for c in entries)
    assert b.resident_bytes == expected
    assert b.logical_bytes == expected


# --- rollback trims per layer ---


def test_rollback_trims_trimmable_entries() -> None:
    entries = [KVCache(), KVCache()]
    _feed_kvcache(entries[0], n_tokens=8)
    _feed_kvcache(entries[1], n_tokens=8)
    kv = SimpleKVCache(entries)
    kv.reserve_for_prefill("req-a", [1])
    assert entries[0].offset == 8
    kv.rollback("req-a", 3)
    assert entries[0].offset == 5
    assert entries[1].offset == 5


def test_rollback_with_zero_is_noop() -> None:
    entries = [KVCache()]
    _feed_kvcache(entries[0], n_tokens=4)
    kv = SimpleKVCache(entries)
    kv.reserve_for_prefill("req-a", [1])
    kv.rollback("req-a", 0)
    assert entries[0].offset == 4


def test_rollback_trims_only_trimmable_in_hybrid_list() -> None:
    """D-021 step 5 sub-unit (e) slice 1: per-layer trim.

    Hybrid lists mix trimmable (``KVCache``) and non-trimmable
    (``ArraysCache``) entries. The earlier all-or-nothing
    ``can_trim_prompt_cache`` guard skipped every trim once any
    entry was non-trimmable, blocking attention KV rollback for the
    speculative engine path on Qwen3.5. Per-layer trim now trims the
    trimmable entries individually and passes the recurrent slots
    through; recurrent rollback is the adapter's job (D-021 step 5
    sub-unit (e) slice 2).
    """
    entries: list[Any] = [KVCache(), KVCache(), ArraysCache(size=2)]
    _feed_kvcache(entries[0], n_tokens=4)
    _feed_kvcache(entries[1], n_tokens=4)
    kv = SimpleKVCache(entries)
    kv.reserve_for_prefill("req-a", [1])

    kv.rollback("req-a", 2)

    assert entries[0].offset == 2  # KVCache trimmed
    assert entries[1].offset == 2  # KVCache trimmed
    # ArraysCache untouched — recurrent rollback is the adapter's job.
    assert isinstance(entries[2], ArraysCache)
    assert entries[2].cache == [None, None]


def test_rollback_on_all_recurrent_list_is_noop() -> None:
    """No KVCache in the list (degenerate model with only recurrent
    layers): per-layer trim finds nothing trimmable and passes through
    cleanly. Mirrors the previous no-op semantic on a list where every
    entry happens to be non-trimmable."""
    entries: list[Any] = [ArraysCache(size=2), ArraysCache(size=2)]
    kv = SimpleKVCache(entries)
    kv.reserve_for_prefill("req-a", [1])
    # Should not raise; ArraysCache has no `trim` method but
    # `is_trimmable()` returns False so the loop skips it.
    kv.rollback("req-a", 3)
    assert entries[0].cache == [None, None]
    assert entries[1].cache == [None, None]
