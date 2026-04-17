"""Unit tests for silica.kvcache.store (Unit 16c.2 step 2).

Scope: SyntheticPrefixBlockStore — the PrefixBlockStore implementation
that 16c.2's ContinuousBatcher will install. RadixPrefixCache-level
tests (peek side-effect-freeness, evict-on-dead-leaf, etc.) land in
step 3 when RadixPrefixCache itself is refactored to the Protocol.

Covers the three lifetime invariants from docs/P2_UNIT_16C_2_PREP.md §2:
  L-1 source: retain/release pair, allocate_id monotonic, reuse-free ids.
  L-2 hit ⊆ source: retain_hit requires live source; release_hit loud-fail.
  L-3 detached ⊆ source: register requires live source; release_source
      loud-fails on transition to zero if detached still registered
      OR hits > 0.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.kvcache.store import PrefixBlockStore, SyntheticPrefixBlockStore


def _fake_per_layer_kv(
    n_layers: int = 2,
    *,
    n_kv_heads: int = 2,
    block_size: int = 4,
    head_dim: int = 8,
    seed: float = 0.0,
) -> list[tuple[mx.array, mx.array]]:
    """Tiny per-layer (K, V) slices for register/fetch tests.

    Values carry the seed so fetched arrays can be identity-checked.
    """
    shape = (1, n_kv_heads, block_size, head_dim)
    return [
        (
            mx.full(shape, seed + layer, dtype=mx.float16),
            mx.full(shape, seed + layer + 0.5, dtype=mx.float16),
        )
        for layer in range(n_layers)
    ]


def _arrays_equal(a: mx.array, b: mx.array) -> bool:
    if tuple(a.shape) != tuple(b.shape):
        return False
    eq = a == b
    if isinstance(eq, bool):
        return eq
    return bool(mx.all(eq).item())


# --- construction / Protocol conformance ---


def test_construction_rejects_nonpositive_block_size() -> None:
    with pytest.raises(ValueError, match="block_size"):
        SyntheticPrefixBlockStore(block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        SyntheticPrefixBlockStore(block_size=-1)


def test_is_prefix_block_store_protocol() -> None:
    store = SyntheticPrefixBlockStore(block_size=16)
    assert isinstance(store, PrefixBlockStore)


# --- allocate_id ---


def test_allocate_id_is_monotonic_and_never_reuses() -> None:
    store = SyntheticPrefixBlockStore(block_size=16)
    ids = [store.allocate_id() for _ in range(5)]
    assert ids == [0, 1, 2, 3, 4]
    # Retain-release the first block; id 0 must NOT come back.
    store.retain_source(ids[0])
    store.release_source(ids[0])
    assert store.allocate_id() == 5


# --- L-1 source lifetime ---


def test_source_retain_release_pair() -> None:
    store = SyntheticPrefixBlockStore(block_size=16)
    b = store.allocate_id()
    store.retain_source(b)
    assert store.source_refs(b) == 1
    store.release_source(b)
    assert store.source_refs(b) == 0
    assert b not in store.live_block_ids()


def test_source_retain_is_a_counter() -> None:
    store = SyntheticPrefixBlockStore(block_size=16)
    b = store.allocate_id()
    store.retain_source(b)
    store.retain_source(b)
    assert store.source_refs(b) == 2
    store.release_source(b)
    assert store.source_refs(b) == 1
    store.release_source(b)
    assert b not in store.live_block_ids()


def test_release_source_on_unknown_fails() -> None:
    store = SyntheticPrefixBlockStore(block_size=16)
    with pytest.raises(KeyError, match="no outstanding source ref"):
        store.release_source(42)


# --- L-2 hit ⊆ source ---


def test_retain_hit_requires_live_source() -> None:
    store = SyntheticPrefixBlockStore(block_size=16)
    # Never retained as source.
    with pytest.raises(KeyError, match="no source ref"):
        store.retain_hit(0)


def test_hit_lifecycle_independent_of_source_count() -> None:
    store = SyntheticPrefixBlockStore(block_size=16)
    b = store.allocate_id()
    store.retain_source(b)
    store.retain_hit(b)
    store.retain_hit(b)
    assert store.hit_refs(b) == 2
    store.release_hit(b)
    assert store.hit_refs(b) == 1
    # Source unchanged throughout.
    assert store.source_refs(b) == 1
    store.release_hit(b)
    assert store.hit_refs(b) == 0


def test_release_hit_without_retain_fails() -> None:
    store = SyntheticPrefixBlockStore(block_size=16)
    b = store.allocate_id()
    store.retain_source(b)
    with pytest.raises(KeyError, match="no outstanding hit ref"):
        store.release_hit(b)


def test_release_source_fails_while_hit_refs_positive() -> None:
    """Eviction must not strand a live hit (L-2 ⊆ L-1)."""
    store = SyntheticPrefixBlockStore(block_size=16)
    b = store.allocate_id()
    store.retain_source(b)
    store.retain_hit(b)
    with pytest.raises(RuntimeError, match="hit_refs=1"):
        store.release_source(b)


# --- L-3 detached ⊆ source ---


def test_register_detached_requires_live_source() -> None:
    store = SyntheticPrefixBlockStore(block_size=4)
    with pytest.raises(KeyError, match="requires a live source ref"):
        store.register_detached(0, _fake_per_layer_kv(block_size=4))


def test_register_detached_duplicate_fails() -> None:
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)
    store.register_detached(b, _fake_per_layer_kv(block_size=4))
    with pytest.raises(ValueError, match="already registered"):
        store.register_detached(b, _fake_per_layer_kv(block_size=4))


def test_fetch_detached_returns_registered_slices() -> None:
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)
    kv = _fake_per_layer_kv(n_layers=3, block_size=4, seed=7.0)
    store.register_detached(b, kv)
    got = store.fetch_detached(b)
    assert len(got) == 3
    for (k_in, v_in), (k_out, v_out) in zip(kv, got, strict=True):
        mx.eval(k_in, v_in, k_out, v_out)
        assert _arrays_equal(k_in, k_out)
        assert _arrays_equal(v_in, v_out)


def test_fetch_detached_missing_fails() -> None:
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)
    with pytest.raises(KeyError, match="no detached K/V registered"):
        store.fetch_detached(b)


def test_release_detached_drops_the_entry() -> None:
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)
    store.register_detached(b, _fake_per_layer_kv(block_size=4))
    assert store.has_detached(b)
    store.release_detached(b)
    assert not store.has_detached(b)


def test_release_detached_missing_fails() -> None:
    store = SyntheticPrefixBlockStore(block_size=4)
    with pytest.raises(KeyError, match="no detached K/V to release"):
        store.release_detached(99)


def test_release_source_fails_while_detached_registered() -> None:
    """Eviction order must be release_detached before release_source."""
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)
    store.register_detached(b, _fake_per_layer_kv(block_size=4))
    with pytest.raises(RuntimeError, match="detached K/V is still registered"):
        store.release_source(b)


def test_release_detached_does_not_touch_source_or_hits() -> None:
    """Detached lifetime is independent below the source layer — releasing
    detached K/V does not affect source or hit refcounts. Only the
    reverse direction is constrained (release_source blocks on detached).
    """
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)
    store.retain_hit(b)
    store.register_detached(b, _fake_per_layer_kv(block_size=4))
    store.release_detached(b)
    assert store.source_refs(b) == 1
    assert store.hit_refs(b) == 1


# --- Combined workflow — insert-lookup-release-evict lifecycle ---


def test_insert_lookup_release_evict_sequence() -> None:
    """End-to-end: one block goes through the full 16c.2 lifecycle."""
    store = SyntheticPrefixBlockStore(block_size=4)
    # 1. Insert: allocate, retain_source, register_detached.
    b = store.allocate_id()
    store.retain_source(b)
    store.register_detached(b, _fake_per_layer_kv(block_size=4))
    assert store.source_refs(b) == 1
    assert store.has_detached(b)
    # 2. Lookup: retain_hit.
    store.retain_hit(b)
    assert store.hit_refs(b) == 1
    # 3. Batcher copies K/V into the row, then release.
    _ = store.fetch_detached(b)  # would be copied into BatchKVCache
    store.release_hit(b)
    assert store.hit_refs(b) == 0
    # Detached K/V survives release — future lookups can still hit.
    assert store.has_detached(b)
    # 4. Evict: release_detached then release_source.
    store.release_detached(b)
    store.release_source(b)
    assert b not in store.live_block_ids()
    assert not store.has_detached(b)


def test_duplicate_source_paths_do_not_leak_detached() -> None:
    """Two insert paths may retain_source on the same block id (e.g. if the
    store were wired to allow id sharing); detached K/V belongs to ONE of
    them — registration must not silently overwrite. 16c.2's
    insert_detached uses allocate_id per new node so this path isn't
    hit in practice, but the store's contract should still refuse it.
    """
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)  # path A
    store.register_detached(b, _fake_per_layer_kv(block_size=4, seed=1.0))
    store.retain_source(b)  # path B — source-only, no re-register

    # Path B attempting to re-register for the same id: fail.
    with pytest.raises(ValueError, match="already registered"):
        store.register_detached(b, _fake_per_layer_kv(block_size=4, seed=2.0))

    # Drop path B's source — detached survives because source still > 0.
    store.release_source(b)
    assert store.source_refs(b) == 1
    assert store.has_detached(b)
