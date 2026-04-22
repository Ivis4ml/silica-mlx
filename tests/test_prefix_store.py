"""Unit tests for silica.kvcache.store (Unit 16c.2 step 2 + P-5-A.0.4).

Scope: SyntheticPrefixBlockStore — the PrefixBlockStore implementation
that 16c.2's ContinuousBatcher installs. RadixPrefixCache-level tests
(peek side-effect-freeness, evict-on-dead-leaf, etc.) live elsewhere.

Covers the three lifetime invariants from docs/P2_UNIT_16C_2_PREP.md §2:
  L-1 source: retain/release pair, allocate_id monotonic, reuse-free ids.
  L-2 hit ⊆ source: retain_hit requires live source; release_hit loud-fail.
  L-3 detached ⊆ source: register requires live source; release_source
      loud-fails on transition to zero if detached still registered
      OR hits > 0.

P-5-A.0.4 additions (below ``--- P-5 side-level codec dispatch ---``):
  - constructor rejection of ``codec=X, k_codec=Y`` combination.
  - constructor rejection of mixed-None split (one side None, one non-None).
  - constructor rejection of ``codec.block_size`` mismatch (shorthand + split).
  - pass-through identity-by-reference (no codec → ``fetch_detached`` tensors
    are the original ``mx.array`` objects).
  - shorthand dispatch via a single shared counting codec; total side call
    count = ``2 × num_layers`` per ``register_detached`` call.
  - split-mode K/V dispatch independence: two distinct counting codec
    instances observe exactly their own side's tensors by ``is``-identity.
    Regression-locks the Q-008 resolution.
  - ``resident_bytes`` arithmetic under pass-through, shorthand, and split.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.kvcache.codec import (
    CodedPayload,
    IdentityCodec,
    RawFp16Payload,
)
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


# ============================================================================
# P-5 side-level codec dispatch (P-5-A.0.4)
# ============================================================================


class _TaggedCountingCodec:
    """Counting IdentityCodec wrapper tagged with a side label.

    Used in split-mode tests to assert K/V dispatch independence: each
    instance records the ``id()`` of every tensor that flows through its
    ``encode_tensor`` and ``decode_tensor`` so the test can check that K
    tensors land only in the K codec and V tensors only in the V codec.
    """

    def __init__(
        self,
        *,
        tag: str,
        block_size: int = 4,
        n_kv_heads: int = 2,
        head_dim: int = 8,
    ) -> None:
        self.tag = tag
        self._inner = IdentityCodec(
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        self.block_size = self._inner.block_size
        self.dtype = self._inner.dtype
        self.encode_calls = 0
        self.decode_calls = 0
        self.seen_encode_ids: list[int] = []
        self.seen_decode_ids: list[int] = []

    def encode_tensor(self, x: mx.array) -> RawFp16Payload:
        self.encode_calls += 1
        self.seen_encode_ids.append(id(x))
        return self._inner.encode_tensor(x)

    def decode_tensor(self, payload: RawFp16Payload) -> mx.array:
        self.decode_calls += 1
        self.seen_decode_ids.append(id(payload.t))
        return self._inner.decode_tensor(payload)

    def logical_bytes(self, num_tokens: int) -> int:
        return self._inner.logical_bytes(num_tokens)

    def resident_bytes(self, num_blocks: int) -> int:
        return self._inner.resident_bytes(num_blocks)


# --- constructor rejection rules ---


def test_rejects_codec_and_k_codec_combo() -> None:
    """`codec=` shorthand cannot combine with an explicit side kwarg."""
    ic = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="shorthand cannot be combined"):
        SyntheticPrefixBlockStore(block_size=4, codec=ic, k_codec=ic)


def test_rejects_codec_and_v_codec_combo() -> None:
    ic = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="shorthand cannot be combined"):
        SyntheticPrefixBlockStore(block_size=4, codec=ic, v_codec=ic)


def test_rejects_mixed_none_split_k_only() -> None:
    """Split form is both-or-neither — passing only k_codec raises."""
    ic = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="both provided or both None"):
        SyntheticPrefixBlockStore(block_size=4, k_codec=ic)


def test_rejects_mixed_none_split_v_only() -> None:
    ic = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="both provided or both None"):
        SyntheticPrefixBlockStore(block_size=4, v_codec=ic)


def test_rejects_shorthand_block_size_mismatch() -> None:
    """The block_size precondition applies to the shorthand codec."""
    ic = IdentityCodec(block_size=8, n_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="k_codec.block_size"):
        SyntheticPrefixBlockStore(block_size=4, codec=ic)


def test_rejects_split_k_block_size_mismatch() -> None:
    k_ic = IdentityCodec(block_size=8, n_kv_heads=2, head_dim=8)
    v_ic = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="k_codec.block_size"):
        SyntheticPrefixBlockStore(block_size=4, k_codec=k_ic, v_codec=v_ic)


def test_rejects_split_v_block_size_mismatch() -> None:
    k_ic = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    v_ic = IdentityCodec(block_size=8, n_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="v_codec.block_size"):
        SyntheticPrefixBlockStore(block_size=4, k_codec=k_ic, v_codec=v_ic)


# --- pass-through path ---


def test_pass_through_holds_raw_tensors_by_reference() -> None:
    """All three codec kwargs ``None`` → fetch returns the original
    ``mx.array`` objects with ``is`` identity preserved."""
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)
    kv = _fake_per_layer_kv(n_layers=2, block_size=4)
    store.register_detached(b, kv)
    got = store.fetch_detached(b)
    for (k_in, v_in), (k_out, v_out) in zip(kv, got, strict=True):
        assert k_out is k_in
        assert v_out is v_in


def test_pass_through_resident_bytes_sums_raw_nbytes() -> None:
    store = SyntheticPrefixBlockStore(block_size=4)
    b = store.allocate_id()
    store.retain_source(b)
    kv = _fake_per_layer_kv(n_layers=3, block_size=4)
    expected = sum(int(k.nbytes) + int(v.nbytes) for (k, v) in kv)
    store.register_detached(b, kv)
    assert store.resident_bytes() == expected


# --- shorthand path (codec=IdentityCodec) ---


def test_shorthand_wraps_tensors_in_raw_fp16_payload() -> None:
    ic = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    store = SyntheticPrefixBlockStore(block_size=4, codec=ic)
    b = store.allocate_id()
    store.retain_source(b)
    kv = _fake_per_layer_kv(n_layers=2, block_size=4)
    store.register_detached(b, kv)
    # Internal layout: each layer holds two RawFp16Payload instances.
    for layer in store._detached[b]:
        assert isinstance(layer.k, CodedPayload)
        assert isinstance(layer.v, CodedPayload)


def test_shorthand_fetch_returns_identity_by_reference() -> None:
    """Under ``IdentityCodec`` shorthand, ``fetch_detached`` still returns
    the original tensor refs (no defensive copy)."""
    ic = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    store = SyntheticPrefixBlockStore(block_size=4, codec=ic)
    b = store.allocate_id()
    store.retain_source(b)
    kv = _fake_per_layer_kv(n_layers=2, block_size=4)
    store.register_detached(b, kv)
    got = store.fetch_detached(b)
    for (k_in, v_in), (k_out, v_out) in zip(kv, got, strict=True):
        assert k_out is k_in
        assert v_out is v_in


def test_shorthand_counts_both_sides_through_one_codec() -> None:
    """Shorthand sets ``k_codec = v_codec = same instance``; a single
    counter tallies 2 × num_layers encode calls per ``register_detached``.
    """
    counter = _TaggedCountingCodec(tag="both", block_size=4)
    store = SyntheticPrefixBlockStore(block_size=4, codec=counter)
    b = store.allocate_id()
    store.retain_source(b)
    n_layers = 3
    store.register_detached(b, _fake_per_layer_kv(n_layers=n_layers, block_size=4))
    # K and V each fire once per layer through the one shared codec.
    assert counter.encode_calls == 2 * n_layers
    assert counter.decode_calls == 0
    # Fetch fires 2 × n_layers decode calls.
    _ = store.fetch_detached(b)
    assert counter.decode_calls == 2 * n_layers


# --- split-mode K/V dispatch independence (Q-008 regression lock) ---


def test_split_mode_k_codec_sees_only_k_tensors() -> None:
    """Split ``k_codec`` / ``v_codec`` with two distinct counting instances:
    K tensors by ``is`` identity flow only through the K codec."""
    k_codec = _TaggedCountingCodec(tag="k", block_size=4)
    v_codec = _TaggedCountingCodec(tag="v", block_size=4)
    store = SyntheticPrefixBlockStore(
        block_size=4, k_codec=k_codec, v_codec=v_codec
    )
    b = store.allocate_id()
    store.retain_source(b)
    kv = _fake_per_layer_kv(n_layers=3, block_size=4)
    store.register_detached(b, kv)

    expected_k_ids = [id(k) for (k, _v) in kv]
    expected_v_ids = [id(v) for (_k, v) in kv]

    assert k_codec.seen_encode_ids == expected_k_ids
    assert v_codec.seen_encode_ids == expected_v_ids
    assert k_codec.encode_calls == 3
    assert v_codec.encode_calls == 3
    # Cross-contamination check: no V tensor id observed in K codec and vice versa.
    assert set(k_codec.seen_encode_ids).isdisjoint(set(expected_v_ids))
    assert set(v_codec.seen_encode_ids).isdisjoint(set(expected_k_ids))


def test_split_mode_decode_dispatch_independence() -> None:
    """On fetch, each side's ``decode_tensor`` fires exactly ``num_layers``
    times and never receives the other side's payload."""
    k_codec = _TaggedCountingCodec(tag="k", block_size=4)
    v_codec = _TaggedCountingCodec(tag="v", block_size=4)
    store = SyntheticPrefixBlockStore(
        block_size=4, k_codec=k_codec, v_codec=v_codec
    )
    b = store.allocate_id()
    store.retain_source(b)
    kv = _fake_per_layer_kv(n_layers=2, block_size=4)
    store.register_detached(b, kv)

    k_codec.encode_calls = 0  # reset to isolate decode counts
    v_codec.encode_calls = 0

    _ = store.fetch_detached(b)
    assert k_codec.decode_calls == 2
    assert v_codec.decode_calls == 2
    # Decoded payload ids should round-trip back to the original K / V
    # tensor ids (identity codec keeps the reference).
    assert k_codec.seen_decode_ids == [id(k) for (k, _v) in kv]
    assert v_codec.seen_decode_ids == [id(v) for (_k, v) in kv]


def test_split_mode_resident_bytes_sums_both_sides() -> None:
    """Under split identity codecs, ``store.resident_bytes()`` equals the
    sum of per-side payload ``resident_bytes`` across all layers. Each
    side's payload is an honest ``RawFp16Payload`` → ``nbytes`` of the
    stored tensor."""
    k_codec = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    v_codec = IdentityCodec(block_size=4, n_kv_heads=2, head_dim=8)
    store = SyntheticPrefixBlockStore(
        block_size=4, k_codec=k_codec, v_codec=v_codec
    )
    b = store.allocate_id()
    store.retain_source(b)
    kv = _fake_per_layer_kv(n_layers=3, block_size=4)
    expected = sum(int(k.nbytes) + int(v.nbytes) for (k, v) in kv)
    store.register_detached(b, kv)
    assert store.resident_bytes() == expected


def test_resident_bytes_matches_shorthand_codec_arithmetic() -> None:
    """Under the ``codec=IdentityCodec(...)`` shorthand with N detached
    blocks and L layers, ``store.resident_bytes()`` equals
    ``N × (k_codec.resident_bytes(L) + v_codec.resident_bytes(L))`` —
    except here both sides share one codec instance, so the arithmetic is
    ``N × 2 × codec.resident_bytes(L / L)``... actually we test directly
    against the per-side fp16 byte count per block × L × 2 sides × N blocks.
    """
    n_layers = 3
    n_blocks = 2
    n_kv_heads = 2
    head_dim = 8
    block_size = 4
    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    store = SyntheticPrefixBlockStore(block_size=block_size, codec=ic)

    for i in range(n_blocks):
        b = store.allocate_id()
        store.retain_source(b)
        store.register_detached(
            b,
            _fake_per_layer_kv(
                n_layers=n_layers,
                n_kv_heads=n_kv_heads,
                block_size=block_size,
                head_dim=head_dim,
                seed=float(i),
            ),
        )

    per_side_bytes_one_block = block_size * n_kv_heads * head_dim * mx.float16.size
    expected = n_blocks * n_layers * 2 * per_side_bytes_one_block
    assert store.resident_bytes() == expected
