"""P-3-C5.3.0 — RadixPrefixCache recurrent-snapshot surface unit tests.

Covers the C5.3.0 acceptance gates per ``plans/P3_C5_3_DESIGN.md`` §4.1:

- ``_Node`` carries a ``recurrent_snapshot`` slot (default ``None``,
  assignable).
- ``peek_with_node`` returns the deepest matched node alongside the
  ``PrefixHit`` and is **side-effect-free** (mirrors ``peek`` semantics
  bit-for-bit).
- ``lookup_with_node`` returns the deepest matched node alongside the
  ``PrefixHit`` and applies retain / touch / hits semantics identical
  to the existing ``lookup``.
- ``insert_detached`` accepts an optional ``recurrent_snapshots`` list:
  attaches per-block on new nodes; duplicate-prefix branch backfills
  ``None`` slots with caller-provided snapshots (§3.5.1) and keeps
  existing snapshots when both are non-None.
- Evicting a node releases its snapshot reference (verified via
  ``weakref``).

No batcher integration in C5.3.0 — those wirings land at C5.3.3.
"""

from __future__ import annotations

import gc
import weakref

import mlx.core as mx
import pytest

from silica.kvcache.prefix import RadixPrefixCache, _Node
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.recurrent import RecurrentSnapshot

BLOCK_SIZE = 4
N_LAYERS = 2
N_KV_HEADS = 2
HEAD_DIM = 8


def _store() -> SyntheticPrefixBlockStore:
    return SyntheticPrefixBlockStore(block_size=BLOCK_SIZE)


def _pc(store: SyntheticPrefixBlockStore | None = None) -> RadixPrefixCache:
    return RadixPrefixCache(block_size=BLOCK_SIZE, store=store or _store())


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
    return [
        _per_layer_kv(seed=seed_base + b * 100.0) for b in range(n_blocks)
    ]


def _snap() -> RecurrentSnapshot:
    """Synthetic snapshot, distinguishable by ``id()``.

    ``RecurrentSnapshot`` carries no boundary metadata; the kvcache
    layer never inspects its contents (only stores / forwards), so
    empty ``entries`` is sufficient for these tests.
    """
    return RecurrentSnapshot(entries=(), nbytes=0)


# --- _Node slot ---


class TestNodeSnapshotSlot:
    def test_default_is_none(self) -> None:
        node = _Node(parent=None, tokens=(), block_id=0)
        assert node.recurrent_snapshot is None

    def test_assignable_via_init(self) -> None:
        snap = _snap()
        node = _Node(
            parent=None, tokens=(), block_id=0, recurrent_snapshot=snap
        )
        assert node.recurrent_snapshot is snap

    def test_assignable_post_init(self) -> None:
        node = _Node(parent=None, tokens=(), block_id=0)
        snap = _snap()
        node.recurrent_snapshot = snap
        assert node.recurrent_snapshot is snap

    def test_slot_is_in_slots_tuple(self) -> None:
        # Catches accidental migration to __dict__-backed storage.
        assert "recurrent_snapshot" in _Node.__slots__


# --- peek_with_node side-effect-freeness ---


class TestPeekWithNode:
    def test_returns_deepest_matched_node(self) -> None:
        pc = _pc()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        snap_b0, snap_b1 = _snap(), _snap()
        ids = pc.insert_detached(
            tokens, _detached_blocks(2), recurrent_snapshots=[snap_b0, snap_b1]
        )
        assert len(ids) == 2

        hit, deepest = pc.peek_with_node(tokens)
        assert hit.num_hit_tokens == 8
        assert hit.block_ids == ids
        assert deepest is not None
        assert deepest.block_id == ids[1]
        assert deepest.recurrent_snapshot is snap_b1
        # And the parent is the first-block node with snap_b0.
        assert deepest.parent is not None
        assert deepest.parent.block_id == ids[0]
        assert deepest.parent.recurrent_snapshot is snap_b0

    def test_no_match_returns_none_node(self) -> None:
        pc = _pc()
        pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))
        hit, deepest = pc.peek_with_node([9, 9, 9, 9])
        assert hit.num_hit_tokens == 0
        assert hit.block_ids == ()
        assert deepest is None

    def test_no_side_effects_match_peek(self) -> None:
        # Build two identical caches; peek on one and peek_with_node on
        # the other; observable state must match bit-for-bit.
        store_a = _store()
        store_b = _store()
        pc_a = _pc(store_a)
        pc_b = _pc(store_b)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ids_a = pc_a.insert_detached(tokens, _detached_blocks(2))
        ids_b = pc_b.insert_detached(tokens, _detached_blocks(2))
        # Synthetic store allocates fresh ids per cache; ids_a and
        # ids_b are independent — compare side effects, not ids.

        before_hits_a = pc_a.hits
        before_hits_b = pc_b.hits
        before_node_count_a = pc_a.node_count()
        before_node_count_b = pc_b.node_count()
        before_hit_refs_a = {b: store_a.hit_refs(b) for b in ids_a}
        before_hit_refs_b = {b: store_b.hit_refs(b) for b in ids_b}
        before_source_refs_a = {b: store_a.source_refs(b) for b in ids_a}
        before_source_refs_b = {b: store_b.source_refs(b) for b in ids_b}

        # Repeat to catch any per-call side-effect.
        for _ in range(3):
            pc_a.peek(tokens)
            pc_b.peek_with_node(tokens)

        assert pc_a.hits == before_hits_a
        assert pc_b.hits == before_hits_b
        assert pc_a.node_count() == before_node_count_a
        assert pc_b.node_count() == before_node_count_b
        for b in ids_a:
            assert store_a.hit_refs(b) == before_hit_refs_a[b]
            assert store_a.source_refs(b) == before_source_refs_a[b]
        for b in ids_b:
            assert store_b.hit_refs(b) == before_hit_refs_b[b]
            assert store_b.source_refs(b) == before_source_refs_b[b]

    def test_does_not_advance_lru_tick(self) -> None:
        pc = _pc()
        tokens = [1, 2, 3, 4]
        pc.insert_detached(tokens, _detached_blocks(1))
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        tick_before = deepest.access_tick
        for _ in range(3):
            pc.peek_with_node(tokens)
        assert deepest.access_tick == tick_before


# --- lookup_with_node parity with lookup ---


class TestLookupWithNode:
    def test_returns_deepest_matched_node(self) -> None:
        pc = _pc()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        snap_b0, snap_b1 = _snap(), _snap()
        ids = pc.insert_detached(
            tokens,
            _detached_blocks(2),
            recurrent_snapshots=[snap_b0, snap_b1],
        )

        hit, deepest = pc.lookup_with_node(tokens)
        try:
            assert hit.num_hit_tokens == 8
            assert hit.block_ids == ids
            assert deepest is not None
            assert deepest.block_id == ids[1]
            assert deepest.recurrent_snapshot is snap_b1
        finally:
            pc.release(hit.block_ids)

    def test_no_match_returns_none_node(self) -> None:
        pc = _pc()
        pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))
        hit, deepest = pc.lookup_with_node([9, 9, 9, 9])
        assert hit.num_hit_tokens == 0
        assert hit.block_ids == ()
        assert deepest is None
        # No retain happened — release would raise. Skip release.

    def test_side_effects_match_lookup(self) -> None:
        # Mirror caches; lookup vs lookup_with_node; compare every
        # piece of observable state.
        store_a = _store()
        store_b = _store()
        pc_a = _pc(store_a)
        pc_b = _pc(store_b)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        ids_a = pc_a.insert_detached(tokens, _detached_blocks(2))
        ids_b = pc_b.insert_detached(tokens, _detached_blocks(2))

        hit_a = pc_a.lookup(tokens)
        hit_b, deepest_b = pc_b.lookup_with_node(tokens)
        try:
            # Counters identical.
            assert pc_a.hits == pc_b.hits == 1
            assert hit_a.num_hit_tokens == hit_b.num_hit_tokens == 8
            # hit_refs incremented by exactly one per hit block.
            for b in ids_a:
                assert store_a.hit_refs(b) == 1
            for b in ids_b:
                assert store_b.hit_refs(b) == 1
            # source_refs unchanged (still 1 from insert).
            for b in ids_a:
                assert store_a.source_refs(b) == 1
            for b in ids_b:
                assert store_b.source_refs(b) == 1
            # The new method also returns a non-None deepest node.
            assert deepest_b is not None
            assert deepest_b.block_id == ids_b[-1]
        finally:
            pc_a.release(hit_a.block_ids)
            pc_b.release(hit_b.block_ids)

    def test_increments_hits_on_match(self) -> None:
        pc = _pc()
        tokens = [1, 2, 3, 4]
        pc.insert_detached(tokens, _detached_blocks(1))
        before = pc.hits
        hit, _ = pc.lookup_with_node(tokens)
        try:
            assert pc.hits == before + 1
        finally:
            pc.release(hit.block_ids)

    def test_does_not_increment_hits_on_zero_match(self) -> None:
        pc = _pc()
        pc.insert_detached([1, 2, 3, 4], _detached_blocks(1))
        before = pc.hits
        hit, _ = pc.lookup_with_node([9, 9, 9, 9])
        assert pc.hits == before
        # No retain; nothing to release.
        assert hit.block_ids == ()


# --- insert_detached snapshot extension ---


class TestInsertDetachedSnapshots:
    def test_default_none_preserves_pre_c5_3_behaviour(self) -> None:
        # Without recurrent_snapshots, every newly-created node has
        # recurrent_snapshot=None — bit-for-bit pre-C5.3 behaviour.
        pc = _pc()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        pc.insert_detached(tokens, _detached_blocks(2))
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        assert deepest.recurrent_snapshot is None
        assert deepest.parent is not None
        assert deepest.parent.recurrent_snapshot is None

    def test_attach_per_block_on_new_nodes(self) -> None:
        pc = _pc()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        snap_b0, snap_b1, snap_b2 = _snap(), _snap(), _snap()
        pc.insert_detached(
            tokens,
            _detached_blocks(3),
            recurrent_snapshots=[snap_b0, snap_b1, snap_b2],
        )
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        assert deepest.recurrent_snapshot is snap_b2
        assert deepest.parent is not None
        assert deepest.parent.recurrent_snapshot is snap_b1
        assert deepest.parent.parent is not None
        assert deepest.parent.parent.recurrent_snapshot is snap_b0

    def test_attach_with_some_none_entries(self) -> None:
        # recurrent_snapshots[i] may itself be None — attaches None.
        pc = _pc()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        snap_b0 = _snap()
        pc.insert_detached(
            tokens,
            _detached_blocks(2),
            recurrent_snapshots=[snap_b0, None],
        )
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        assert deepest.recurrent_snapshot is None
        assert deepest.parent is not None
        assert deepest.parent.recurrent_snapshot is snap_b0

    def test_length_mismatch_raises(self) -> None:
        pc = _pc()
        with pytest.raises(ValueError, match="recurrent_snapshots"):
            pc.insert_detached(
                [1, 2, 3, 4, 5, 6, 7, 8],
                _detached_blocks(2),
                recurrent_snapshots=[_snap()],  # only 1, need 2
            )

    def test_duplicate_insert_backfills_none(self) -> None:
        # First insert leaves nodes with snapshot=None; second
        # duplicate insert with non-None snapshots backfills them.
        pc = _pc()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        pc.insert_detached(
            tokens,
            _detached_blocks(2),
            recurrent_snapshots=[None, None],
        )
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        assert deepest.recurrent_snapshot is None
        assert deepest.parent is not None
        assert deepest.parent.recurrent_snapshot is None

        snap_b0, snap_b1 = _snap(), _snap()
        new_ids = pc.insert_detached(
            tokens,
            _detached_blocks(2),
            recurrent_snapshots=[snap_b0, snap_b1],
        )
        # Duplicate insert returns empty new_ids tuple.
        assert new_ids == ()
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        assert deepest.recurrent_snapshot is snap_b1
        assert deepest.parent is not None
        assert deepest.parent.recurrent_snapshot is snap_b0

    def test_duplicate_insert_keeps_existing_when_both_non_none(self) -> None:
        # First insert attaches snap_a; second duplicate insert with
        # snap_b must keep snap_a.
        pc = _pc()
        tokens = [1, 2, 3, 4]
        snap_a = _snap()
        snap_b = _snap()
        pc.insert_detached(
            tokens, _detached_blocks(1), recurrent_snapshots=[snap_a]
        )
        pc.insert_detached(
            tokens, _detached_blocks(1), recurrent_snapshots=[snap_b]
        )
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        assert deepest.recurrent_snapshot is snap_a
        assert deepest.recurrent_snapshot is not snap_b

    def test_duplicate_insert_with_none_does_not_overwrite_existing(
        self,
    ) -> None:
        # Existing snapshot must survive a re-insertion that omits it.
        pc = _pc()
        tokens = [1, 2, 3, 4]
        snap_a = _snap()
        pc.insert_detached(
            tokens, _detached_blocks(1), recurrent_snapshots=[snap_a]
        )
        pc.insert_detached(
            tokens, _detached_blocks(1), recurrent_snapshots=[None]
        )
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        assert deepest.recurrent_snapshot is snap_a

    def test_partial_backfill_only_touches_none_nodes(self) -> None:
        # Mixed prefix: first node has snapshot, second is None.
        # Re-insert with snapshots for both. Only the None one is
        # backfilled; the existing-non-None stays.
        pc = _pc()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        snap_a_orig = _snap()
        pc.insert_detached(
            tokens,
            _detached_blocks(2),
            recurrent_snapshots=[snap_a_orig, None],
        )

        snap_a_new, snap_b_new = _snap(), _snap()
        pc.insert_detached(
            tokens,
            _detached_blocks(2),
            recurrent_snapshots=[snap_a_new, snap_b_new],
        )
        _, deepest = pc.peek_with_node(tokens)
        assert deepest is not None
        # Block 0: kept the original.
        assert deepest.parent is not None
        assert deepest.parent.recurrent_snapshot is snap_a_orig
        # Block 1: backfilled.
        assert deepest.recurrent_snapshot is snap_b_new


# --- eviction releases the snapshot ---


class TestEvictionDropsSnapshot:
    def test_evicted_node_releases_snapshot_reference(self) -> None:
        pc = _pc()
        tokens = [1, 2, 3, 4]
        snap = _snap()
        weak = weakref.ref(snap)
        pc.insert_detached(
            tokens, _detached_blocks(1), recurrent_snapshots=[snap]
        )
        # Drop the test's strong reference; only the node's slot
        # holds the snapshot now.
        del snap
        gc.collect()
        assert weak() is not None  # node still alive — still reachable.

        freed = pc.evict_until(1)
        assert freed == 1
        assert pc.node_count() == 0
        gc.collect()
        assert weak() is None, (
            "snapshot should be GC'd once the only reference (node slot) "
            "drops via _evict_node"
        )

    def test_node_count_drops_to_zero_after_full_eviction(self) -> None:
        pc = _pc()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        pc.insert_detached(
            tokens,
            _detached_blocks(2),
            recurrent_snapshots=[_snap(), _snap()],
        )
        assert pc.node_count() == 2
        # Two evictions: leaf goes first, then the now-leaf interior
        # node.
        freed = pc.evict_until(2)
        assert freed == 2
        assert pc.node_count() == 0
