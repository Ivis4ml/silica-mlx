"""Tests for RadixPrefixCache over SyntheticPrefixBlockStore (Unit 16c.2 step 3).

These tests exercise the 16c.2-new admission path: ``insert_detached``
(store allocates ids + owns detached K/V) and ``peek`` (side-effect-free
lookup). The Paged-backend path is covered in ``tests/test_prefix_cache.py``;
this file stays focused on the Synthetic-backend new surface so failures
point at the right layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import pytest

from silica.kvcache.manager import PrefixHit
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore

BLOCK_SIZE = 4
N_LAYERS = 2
N_KV_HEADS = 2
HEAD_DIM = 8


def _store() -> SyntheticPrefixBlockStore:
    return SyntheticPrefixBlockStore(block_size=BLOCK_SIZE)


def _pc(store: SyntheticPrefixBlockStore | None = None) -> RadixPrefixCache:
    return RadixPrefixCache(block_size=BLOCK_SIZE, store=store or _store())


class _FailingRegisterStore(SyntheticPrefixBlockStore):
    def register_detached(
        self,
        block_id: int,
        per_layer_kv: Sequence[tuple[mx.array, mx.array]],
    ) -> None:
        raise RuntimeError("synthetic register boom")


def _per_layer_kv(seed: float = 0.0) -> list[tuple[mx.array, mx.array]]:
    shape = (1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    return [
        (
            mx.full(shape, seed + layer, dtype=mx.float16),
            mx.full(shape, seed + layer + 0.5, dtype=mx.float16),
        )
        for layer in range(N_LAYERS)
    ]


def _detached_blocks(
    n_blocks: int, seed_base: float = 0.0
) -> list[list[tuple[mx.array, mx.array]]]:
    """``detached_blocks[b][layer] = (K, V)`` shaped (1, H, block_size, D)."""
    return [_per_layer_kv(seed=seed_base + b * 100.0) for b in range(n_blocks)]


def _arrays_equal(a: mx.array, b: mx.array) -> bool:
    if tuple(a.shape) != tuple(b.shape):
        return False
    eq = a == b
    if isinstance(eq, bool):
        return eq
    return bool(mx.all(eq).item())


# --- peek is side-effect-free ---


def test_peek_matches_lookup_shape_without_any_side_effects() -> None:
    store = _store()
    pc = _pc(store)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks
    ids = pc.insert_detached(tokens, _detached_blocks(2))
    assert len(ids) == 2

    # Snapshot every piece of state peek must leave alone.
    before_hits = pc.hits
    before_hit_refs = {b: store.hit_refs(b) for b in ids}
    before_source_refs = {b: store.source_refs(b) for b in ids}
    before_node_count = pc.node_count()
    # Exercise peek multiple times — side-effect-freeness must be idempotent.
    for _ in range(3):
        result = pc.peek(tokens)
        assert result.num_hit_tokens == 8
        assert result.block_ids == ids

    assert pc.hits == before_hits
    for b in ids:
        assert store.hit_refs(b) == before_hit_refs[b]
        assert store.source_refs(b) == before_source_refs[b]
    assert pc.node_count() == before_node_count


def test_peek_on_miss_returns_empty_and_no_side_effects() -> None:
    store = _store()
    pc = _pc(store)
    pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))
    result = pc.peek([99, 99, 99, 99])
    assert result == PrefixHit()
    assert pc.hits == 0


def test_peek_vs_lookup_divergence_on_hit() -> None:
    """Same tokens: peek leaves pc.hits / hit_refs at 0; lookup bumps both."""
    store = _store()
    pc = _pc(store)
    tokens = [1, 2, 3, 4]
    (bid,) = pc.insert_detached(tokens, _detached_blocks(1))

    pc.peek(tokens)
    assert pc.hits == 0
    assert store.hit_refs(bid) == 0

    hit = pc.lookup(tokens)
    assert pc.hits == 1
    assert store.hit_refs(bid) == 1
    pc.release(list(hit.block_ids))


# --- insert_detached ---


def test_insert_detached_allocates_fresh_ids_and_registers_kv() -> None:
    store = _store()
    pc = _pc(store)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    blocks = _detached_blocks(2, seed_base=10.0)
    ids = pc.insert_detached(tokens, blocks)

    assert len(ids) == 2
    assert ids[0] != ids[1]
    for bid in ids:
        assert store.source_refs(bid) == 1
        assert store.has_detached(bid)
    # Detached K/V content round-trip through fetch.
    for expected_pair_list, bid in zip(blocks, ids, strict=True):
        got = store.fetch_detached(bid)
        for (k_in, v_in), (k_out, v_out) in zip(
            expected_pair_list, got, strict=True
        ):
            mx.eval(k_in, v_in, k_out, v_out)
            assert _arrays_equal(k_in, k_out)
            assert _arrays_equal(v_in, v_out)


def test_insert_detached_drops_partial_trailing_block() -> None:
    store = _store()
    pc = _pc(store)
    # 5 tokens = 1 aligned block + 1 partial tail.
    tokens = [1, 2, 3, 4, 5]
    # Caller provides detached K/V for 2 blocks but only 1 is feasible.
    blocks = _detached_blocks(2)
    ids = pc.insert_detached(tokens, blocks)
    assert len(ids) == 1
    assert pc.node_count() == 1


def test_insert_detached_requires_kv_for_each_aligned_block() -> None:
    store = _store()
    pc = _pc(store)
    # 8 tokens = 2 aligned blocks, so one detached block would make the
    # prefix silently shorter. That is a caller bug in the reclaim path.
    with pytest.raises(ValueError, match="cover every aligned token block"):
        pc.insert_detached([1, 2, 3, 4, 5, 6, 7, 8], _detached_blocks(1))


def test_insert_detached_failure_rolls_back_source_and_tree() -> None:
    store = _FailingRegisterStore(block_size=BLOCK_SIZE)
    pc = _pc(store)

    with pytest.raises(RuntimeError, match="synthetic register boom"):
        pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))

    assert pc.node_count() == 0
    assert store.source_refs(0) == 0
    assert not store.has_detached(0)


def test_insert_detached_duplicate_prefix_reuses_existing_node() -> None:
    """Second insert of the same prefix tokens must NOT register new detached K/V."""
    store = _store()
    pc = _pc(store)
    tokens = [1, 2, 3, 4]

    first_ids = pc.insert_detached(tokens, _detached_blocks(1, seed_base=1.0))
    second_ids = pc.insert_detached(tokens, _detached_blocks(1, seed_base=2.0))
    assert len(first_ids) == 1
    assert second_ids == ()  # no new nodes inserted
    # Only one block registered in the store.
    assert pc.node_count() == 1
    assert store.source_refs(first_ids[0]) == 1  # not 2 — the idempotent path
    # The original detached K/V survived; the second call's blocks are GC-able.
    hit = pc.lookup(tokens)
    got = store.fetch_detached(first_ids[0])
    # First-layer K's seed should be 1.0 (first insert), not 2.0.
    k0, _ = got[0]
    mx.eval(k0)
    # Every element equals seed=1.0+layer=0 → 1.0. Fast single-element check.
    assert float(k0.flatten()[0].item()) == 1.0
    pc.release(list(hit.block_ids))


def test_insert_detached_extends_existing_branch_when_first_chunk_shared() -> None:
    store = _store()
    pc = _pc(store)
    # First insert: 1 block.
    short_ids = pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))
    # Second insert: shares first block, adds a second.
    long_ids = pc.insert_detached(
        [1, 2, 3, 4, 5, 6, 7, 8], _detached_blocks(2, seed_base=50.0)
    )
    assert len(short_ids) == 1
    # Only the SECOND block is newly inserted.
    assert len(long_ids) == 1
    assert long_ids[0] != short_ids[0]
    assert pc.node_count() == 2


# --- eviction with detached storage ---


def test_evict_releases_detached_kv_atomically() -> None:
    store = _store()
    pc = _pc(store)
    (bid,) = pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))
    assert store.has_detached(bid)
    assert store.source_refs(bid) == 1

    freed = pc.evict_until(1)
    assert freed == 1
    assert not store.has_detached(bid)
    assert store.source_refs(bid) == 0
    assert pc.node_count() == 0


def test_evict_skips_block_with_live_hit() -> None:
    store = _store()
    pc = _pc(store)
    (bid,) = pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))
    hit = pc.lookup([1, 2, 3, 4])
    assert store.hit_refs(bid) == 1

    freed = pc.evict_until(5)
    assert freed == 0
    assert pc.node_count() == 1
    # Release — now evictable.
    pc.release(list(hit.block_ids))
    assert store.hit_refs(bid) == 0
    assert pc.evict_until(1) == 1


# --- release ---


def test_release_over_synthetic_store_preserves_detached() -> None:
    store = _store()
    pc = _pc(store)
    (bid,) = pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))
    hit = pc.lookup([1, 2, 3, 4])
    pc.release(list(hit.block_ids))
    assert store.hit_refs(bid) == 0
    # Detached survives release — future lookups can still hit.
    assert store.has_detached(bid)
    assert store.source_refs(bid) == 1


def test_release_mismatched_raises() -> None:
    store = _store()
    pc = _pc(store)
    with pytest.raises(KeyError, match="no outstanding hit ref"):
        pc.release([42])
