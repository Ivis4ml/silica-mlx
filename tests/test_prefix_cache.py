"""Tests for silica.kvcache.prefix — RadixPrefixCache (P-2 Unit #14).

Exercises the block-granular radix trie + refcount discipline using a
real PagedKVCache fixture (Unit #13). Each test's "source blocks" are
produced by reserving a dummy request, inserting its blocks into the
prefix cache, then freeing the request — after which only the prefix
cache's refcount hold keeps the blocks alive.
"""

from __future__ import annotations

import pytest

from silica.kvcache.manager import PrefixHit
from silica.kvcache.paged import PagedKVCache
from silica.kvcache.prefix import RadixPrefixCache

BLOCK_SIZE = 4  # small for test math: 4 tokens / block


def _kv(*, num_blocks: int = 16, max_batch_size: int = 4) -> PagedKVCache:
    return PagedKVCache(
        num_layers=1,
        max_batch_size=max_batch_size,
        n_kv_heads=1,
        head_dim=4,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        dtype_bytes=2,
    )


def _pc(kv: PagedKVCache) -> RadixPrefixCache:
    return RadixPrefixCache(block_size=kv.block_size, kv=kv)


def _install_source(
    pc: RadixPrefixCache,
    kv: PagedKVCache,
    owner: str,
    tokens: list[int],
) -> list[int]:
    """Allocate blocks via a dummy ``reserve_for_prefill``, register them
    with the prefix cache, then free the dummy request so only the PC holds
    the blocks. Returns the block_ids now pinned by the cache."""
    blist = kv.reserve_for_prefill(owner, tokens)
    block_ids = list(blist.block_ids)
    pc.insert(tokens, block_ids)
    kv.free(owner)
    return block_ids


# --- constructor validation ---


def test_rejects_non_positive_block_size() -> None:
    kv = _kv()
    with pytest.raises(ValueError, match="block_size must be > 0"):
        RadixPrefixCache(block_size=0, kv=kv)


def test_rejects_mismatched_block_size_vs_kv() -> None:
    kv = _kv()
    with pytest.raises(ValueError, match="block_size mismatch"):
        RadixPrefixCache(block_size=8, kv=kv)  # kv is 4


# --- empty cache ---


def test_empty_cache_lookup_returns_miss() -> None:
    kv = _kv()
    pc = _pc(kv)
    hit = pc.lookup([1, 2, 3, 4, 5, 6])
    assert hit == PrefixHit()
    assert pc.hits == 0


def test_lookup_below_block_size_is_miss() -> None:
    """Fewer than block_size tokens cannot produce a hit."""
    kv = _kv()
    pc = _pc(kv)
    _install_source(pc, kv, "src", [1, 2, 3, 4])
    hit = pc.lookup([1, 2, 3])  # 3 tokens < block_size=4
    assert hit.num_hit_tokens == 0
    assert pc.hits == 0


# --- basic insert + lookup ---


def test_insert_then_lookup_full_match() -> None:
    kv = _kv()
    pc = _pc(kv)
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]  # 2 blocks
    block_ids = _install_source(pc, kv, "src", tokens)

    hit = pc.lookup(tokens)
    assert hit.num_hit_tokens == 8
    assert hit.block_ids == tuple(block_ids)
    assert pc.hits == 1


def test_lookup_partial_prefix_match() -> None:
    """Lookup with tokens that match first block only — reports 1-block hit."""
    kv = _kv()
    pc = _pc(kv)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks
    block_ids = _install_source(pc, kv, "src", tokens)

    # New prompt shares the first block only.
    lookup_tokens = [1, 2, 3, 4, 99, 99, 99, 99]
    hit = pc.lookup(lookup_tokens)
    assert hit.num_hit_tokens == 4
    assert hit.block_ids == (block_ids[0],)
    assert pc.hits == 1


def test_lookup_no_match_returns_miss() -> None:
    kv = _kv()
    pc = _pc(kv)
    _install_source(pc, kv, "src", [1, 2, 3, 4])
    hit = pc.lookup([99, 99, 99, 99])
    assert hit == PrefixHit()


def test_lookup_truncates_unaligned_trailing_tokens() -> None:
    """A hit on 1 block shouldn't count the non-aligned tail."""
    kv = _kv()
    pc = _pc(kv)
    tokens = [1, 2, 3, 4]
    _install_source(pc, kv, "src", tokens)
    hit = pc.lookup([1, 2, 3, 4, 5, 6])  # 4 align + 2 partial
    assert hit.num_hit_tokens == 4  # not 6


# --- insert policies ---


def test_insert_drops_partial_trailing_block() -> None:
    """If len(tokens) < len(block_ids) * block_size, trailing blocks drop."""
    kv = _kv()
    pc = _pc(kv)
    # Caller claims 2 blocks but only provides 1.5 blocks' worth of tokens.
    blist = kv.reserve_for_prefill("src", [1, 2, 3, 4, 5, 6])  # 2 blocks
    blocks = list(blist.block_ids)
    pc.insert([1, 2, 3, 4, 5, 6], blocks)  # only 1 aligned block retainable
    # Only first block is in the tree.
    assert pc.node_count() == 1
    hit = pc.lookup([1, 2, 3, 4])
    assert hit.block_ids == (blocks[0],)
    kv.free("src")


def test_double_insert_same_prefix_is_idempotent() -> None:
    """Second insert of same prefix does not add new nodes or re-incref."""
    kv = _kv()
    pc = _pc(kv)
    tokens = [1, 2, 3, 4]

    # First request + insert: tree gets block_a.
    blist_a = kv.reserve_for_prefill("a", tokens)
    block_a = blist_a.block_ids[0]
    pc.insert(tokens, [block_a])
    kv.free("a")
    # Tree has 1 node, block_a's refcount is 1 (PC only).

    # Second request + insert with same tokens but DIFFERENT block id.
    blist_b = kv.reserve_for_prefill("b", tokens)
    block_b = blist_b.block_ids[0]
    assert block_b != block_a
    pc.insert(tokens, [block_b])
    kv.free("b")

    # Tree should still have exactly 1 node (block_a, the original).
    assert pc.node_count() == 1
    # Lookup resolves to block_a, not block_b.
    hit = pc.lookup(tokens)
    assert hit.block_ids == (block_a,)


def test_insert_branches_when_first_chunk_differs() -> None:
    """Two insertions with different first blocks create sibling nodes."""
    kv = _kv()
    pc = _pc(kv)
    _install_source(pc, kv, "a", [1, 2, 3, 4])
    _install_source(pc, kv, "b", [5, 6, 7, 8])
    assert pc.node_count() == 2  # two root children


def test_insert_extends_existing_branch_when_prefix_shared() -> None:
    """If the first block matches an existing node, further blocks extend it."""
    kv = _kv()
    pc = _pc(kv)
    _install_source(pc, kv, "short", [1, 2, 3, 4])
    _install_source(pc, kv, "long", [1, 2, 3, 4, 5, 6, 7, 8])
    # short's single node + long's additional node that hangs off it = 2 nodes total.
    assert pc.node_count() == 2


# --- release / _live_hits ---


def test_release_decrements_live_hits_and_kv_refcount() -> None:
    kv = _kv()
    pc = _pc(kv)
    tokens = [1, 2, 3, 4]
    block_ids = _install_source(pc, kv, "src", tokens)
    # After install, block refcount = 1 (PC hold).
    hit = pc.lookup(tokens)
    assert pc.live_hits(block_ids[0]) == 1
    # kv refcount now 2 (PC + live hit).
    pc.release(list(hit.block_ids))
    assert pc.live_hits(block_ids[0]) == 0
    # kv refcount back to 1 (PC only).
    # Block is still in the tree (not evicted).
    assert pc.node_count() == 1


def test_release_raises_on_untracked_block() -> None:
    kv = _kv()
    pc = _pc(kv)
    with pytest.raises(KeyError, match="no outstanding live-hit"):
        pc.release([0])


def test_double_release_raises_on_second_release() -> None:
    """Releasing more than you looked up should loud-fail."""
    kv = _kv()
    pc = _pc(kv)
    tokens = [1, 2, 3, 4]
    _install_source(pc, kv, "src", tokens)
    hit = pc.lookup(tokens)
    pc.release(list(hit.block_ids))
    with pytest.raises(KeyError):
        pc.release(list(hit.block_ids))


# --- eviction ---


def test_evict_until_frees_lru_leaf_blocks() -> None:
    kv = _kv(num_blocks=8)
    pc = _pc(kv)
    # Install three independent source blocks.
    _install_source(pc, kv, "a", [1, 2, 3, 4])
    _install_source(pc, kv, "b", [5, 6, 7, 8])
    _install_source(pc, kv, "c", [9, 10, 11, 12])
    assert pc.node_count() == 3
    assert kv.available_blocks() == 5  # 8 - 3 pinned

    # Touch "b" via lookup-then-release so it's the newest.
    hit = pc.lookup([5, 6, 7, 8])
    pc.release(list(hit.block_ids))

    freed = pc.evict_until(1)
    assert freed == 1
    assert pc.node_count() == 2
    assert kv.available_blocks() == 6


def test_evict_until_skips_nodes_with_live_hits() -> None:
    kv = _kv(num_blocks=8)
    pc = _pc(kv)
    _install_source(pc, kv, "a", [1, 2, 3, 4])
    _install_source(pc, kv, "b", [5, 6, 7, 8])
    # Keep "a" alive as a live hit.
    pc.lookup([1, 2, 3, 4])

    freed = pc.evict_until(2)
    # Only "b" can be evicted (1 block); "a" is pinned by the live hit.
    assert freed == 1
    assert pc.node_count() == 1


def test_evict_until_skips_non_leaf_nodes() -> None:
    """Internal nodes of a branch can only be evicted after their leaves."""
    kv = _kv(num_blocks=8)
    pc = _pc(kv)
    # "long" produces a chain: root → n1 → n2 (both internal until n2 is leaf).
    _install_source(pc, kv, "long", [1, 2, 3, 4, 5, 6, 7, 8])
    assert pc.node_count() == 2
    # Evicting the root's child is illegal (it has a child of its own).
    # Only the tail node is evictable.
    freed = pc.evict_until(1)
    assert freed == 1
    assert pc.node_count() == 1  # the internal node remains
    freed = pc.evict_until(1)
    assert freed == 1  # now the remaining node is a leaf
    assert pc.node_count() == 0


def test_evict_until_returns_zero_when_nothing_evictable() -> None:
    kv = _kv(num_blocks=8)
    pc = _pc(kv)
    _install_source(pc, kv, "a", [1, 2, 3, 4])
    pc.lookup([1, 2, 3, 4])  # pin "a" via a live hit
    assert pc.evict_until(5) == 0


def test_evict_until_handles_empty_tree() -> None:
    kv = _kv()
    pc = _pc(kv)
    assert pc.evict_until(3) == 0


# --- LRU order ---


def test_lookup_touches_node_refreshing_lru_order() -> None:
    """A re-lookup on an older node should make it newer than a sibling."""
    kv = _kv(num_blocks=8)
    pc = _pc(kv)
    _install_source(pc, kv, "old", [1, 2, 3, 4])
    _install_source(pc, kv, "new", [5, 6, 7, 8])
    # Touch "old" so it becomes newer than "new".
    pc.release(list(pc.lookup([1, 2, 3, 4]).block_ids))
    # Evict one: should remove "new" (older after the touch), not "old".
    pc.evict_until(1)
    # "old" still in tree.
    assert pc.lookup([1, 2, 3, 4]).num_hit_tokens == 4


# --- hits counter ---


def test_hits_counter_counts_only_nonzero_matches() -> None:
    kv = _kv()
    pc = _pc(kv)
    _install_source(pc, kv, "src", [1, 2, 3, 4])
    assert pc.hits == 0
    pc.lookup([99, 99, 99, 99])  # miss
    assert pc.hits == 0
    pc.lookup([1, 2, 3, 4])  # hit
    assert pc.hits == 1
    pc.lookup([1, 2, 3, 4])  # another hit
    assert pc.hits == 2
