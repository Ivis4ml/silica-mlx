"""P-3-C5.3.3-het.2 — hybrid heterogeneous admit-side seed assembly.

Companion to ``tests/test_batcher_extract_hybrid_heterogeneity.py``
(het.1, extract-side). Het.1 fixed the producer side: extract reads
offset / left_padding from the first attention layer and stores
``detached_blocks[b][i]`` indexed by attention-layer position, not by
absolute transformer layer index.

Het.2 fixes the consumer side. The pre-het.2 ``_admit_single_hit_row``
called ``build_seeded_batch_kv(detached, num_layers=adapter.config
.num_layers)``, which (a) trips the helper's
``len(detached[0]) == num_layers`` validator on hybrid (det
ached has only ``num_attention_layers`` entries) and (b) would yield
an all-``BatchKVCache`` list incompatible with the adapter's
heterogeneous shape — ``restore_recurrent_state`` cannot splice
DeltaNet state into a ``BatchKVCache`` slot.

The het.2 fix:

  1. Build an empty heterogeneous row_cache via
     ``_make_batch_cache(self._adapter, [0])``.
  2. Derive attention positions via ``_token_kv_layer_indices(empty
     _row_cache)`` (het.2 added the optional ``cache_list`` arg).
  3. ``build_seeded_batch_kv(detached, num_layers=len(attn_indices))``
     produces an attention-only list of seeded ``BatchKVCache``.
  4. Interleave seeded into the heterogeneous empty cache: for each
     attention position ``i``, ``row_cache[attn_indices[i]] = seeded[i]``.
  5. ``restore_recurrent_state(row_cache, 0, snapshot)`` populates
     DeltaNet ``ArraysCache`` slots at row 0 (already hybrid-ready
     via ``_splice_row`` case 4 — snapshot data + None live slot).
  6. Suffix prefill flows transparently: the slice helper /
     ``forward_batched`` accept the heterogeneous row_cache; the
     model's per-layer dispatch handles each cache type.

Acceptance gates pinned here:

- ``row_cache`` exposed as ``self._batch_cache`` after admission has
  the heterogeneous per-position types (no homogenization to
  ``BatchKVCache``).
- Inverse mapping correctness: the seeded marker for attention-
  position ``p`` must land at transformer-layer ``attn_indices[p]``
  in the post-admission cache. Mirrors het.1's forward-mapping
  acceptance test; together they pin the off-by-one regression
  class on both sides of the prefix-cache boundary.
- ``restore_recurrent_state`` is called with a cache_list that is
  heterogeneous AT THE TIME OF THE CALL — directly proves the seed
  assembly didn't accidentally collapse to all-``BatchKVCache``
  before reaching restore.
- Pure-attention regression: all-GLOBAL pattern still admits with a
  homogeneous all-``BatchKVCache`` row_cache (every transformer
  layer is an attention layer; the heterogeneous assembly degenerates
  to the original behaviour).

The C5.3.3b real-model byte-exact gate test (currently held untracked)
becomes runnable after this commit lands.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
)
from silica.models.recurrent import RecurrentSnapshot
from silica.scheduler.batcher import ContinuousBatcher
from tests.test_batcher_extract_hybrid_heterogeneity import (
    BLOCK_SIZE,
    HEAD_DIM,
    N_KV,
    _ScriptedHybridAdapter,
    _SpyLog,
)

# --- helpers ---


def _params(max_tokens: int = 1) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _prefix_cache(block_size: int = BLOCK_SIZE) -> RadixPrefixCache:
    return RadixPrefixCache(
        block_size=block_size,
        store=SyntheticPrefixBlockStore(block_size=block_size),
    )


def _make_batcher(
    pattern: AttentionPattern,
    log: _SpyLog,
    pc: RadixPrefixCache,
) -> ContinuousBatcher:
    # Suffix is one slice forward for K=1 cached + 8-token prompt; the
    # spy's logits target is unused (max_tokens=1 emits one decode but
    # admission only runs the prefill chunk before sampling).
    adapter = _ScriptedHybridAdapter(pattern, log, script=(0,))
    return ContinuousBatcher(
        adapter,
        prefix_cache=pc,
    )


def _prep_cohort(batcher: ContinuousBatcher) -> None:
    """Empty step flips ``_cohort_prepared=True`` so the next
    ``add_request`` lands in the waiting queue and exercises the
    Phase-B classifier + ``_admit_single_hit_row`` path."""
    assert batcher.step() == []


def _per_attn_position_kv(
    markers: list[float],
) -> list[tuple[mx.array, mx.array]]:
    """One ``(K, V)`` tuple per attention position. ``markers[pos]``
    fills K, ``markers[pos] + 0.5`` fills V. Length matches
    ``num_attention_layers`` in the test pattern.
    """
    shape = (1, N_KV, BLOCK_SIZE, HEAD_DIM)
    return [
        (
            mx.full(shape, m, dtype=mx.float16),
            mx.full(shape, m + 0.5, dtype=mx.float16),
        )
        for m in markers
    ]


def _placeholder_snapshot() -> RecurrentSnapshot:
    """Non-None snapshot for Phase-B routing. The spy's
    ``restore_recurrent_state`` is no-op, so the entries' contents
    don't drive any side effects under test — only the
    ``snapshot is not None`` predicate inside the classifier matters
    here, plus the heterogeneity assertion on the cache_list passed
    in to restore.
    """
    return RecurrentSnapshot(entries=(), nbytes=0)


def _seed_prefix_with_markers(
    pc: RadixPrefixCache,
    prompt: list[int],
    blocks_markers: list[list[float]],
) -> None:
    """Seed ``pc`` with ``len(blocks_markers)`` blocks. Each block
    carries one ``(K, V)`` per attention position with the given
    markers. Every block's recurrent_snapshot is set non-None so
    Phase-B routes to hit regardless of which depth is the deepest
    USABLE node."""
    n_blocks = len(blocks_markers)
    detached = [_per_attn_position_kv(m) for m in blocks_markers]
    snapshots = [_placeholder_snapshot() for _ in range(n_blocks)]
    pc.insert_detached(
        prompt[: n_blocks * BLOCK_SIZE],
        detached,
        recurrent_snapshots=snapshots,
    )


def _read_marker(arr: mx.array, axis2_idx: int) -> float:
    """Read marker at K[0, 0, axis2_idx, 0] (the spy fills uniformly
    so any cell along the marked dim returns the same value)."""
    return float(arr[0, 0, axis2_idx, 0])


# --- assembly: heterogeneous per-position cache types ---


class TestHybridAdmitAssemblesHeterogeneousRowCache:
    def test_row_cache_per_position_types_match_adapter_shape(
        self,
    ) -> None:
        # Pattern alternates GLOBAL and DELTANET so attention positions
        # land at non-contiguous transformer indices [0, 2]. The
        # admission path must:
        #   - place seeded BatchKVCache at indices 0, 2 (attention)
        #   - leave indices 1, 3 as the empty ArraysCache from
        #     adapter.make_batch_cache (DeltaNet)
        # After admission, batcher._batch_cache (which Insertion C
        # stitched from the row_cache) carries that exact layout.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))  # 8 tokens
        # K=1 block in tree; suffix = 4 tokens = 1 block.
        # 2 attention positions → 2 markers per block.
        _seed_prefix_with_markers(pc, prompt, [[100.0, 300.0]])
        batcher = _make_batcher(pattern, log, pc)

        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert batcher.prefix_hits == 1
        assert batcher._batch_cache is not None
        layers = batcher._batch_cache
        assert len(layers) == 4
        assert isinstance(layers[0], BatchKVCache), (
            f"layer 0 should be BatchKVCache (GLOBAL); got "
            f"{type(layers[0]).__name__}"
        )
        assert isinstance(layers[1], ArraysCache), (
            f"layer 1 should be ArraysCache (DELTANET); got "
            f"{type(layers[1]).__name__}"
        )
        assert isinstance(layers[2], BatchKVCache), (
            f"layer 2 should be BatchKVCache (GLOBAL); got "
            f"{type(layers[2]).__name__}"
        )
        assert isinstance(layers[3], ArraysCache), (
            f"layer 3 should be ArraysCache (DELTANET); got "
            f"{type(layers[3]).__name__}"
        )


# --- inverse mapping: seeded markers land at the right transformer ---


class TestInverseMappingAttnPositionToTransformerLayer:
    def test_seeded_attn_markers_land_at_correct_transformer_indices(
        self,
    ) -> None:
        # Pattern [GLOBAL, DELTANET, GLOBAL, DELTANET]. The seeded
        # BatchKVCache for attention-position 0 must land at
        # transformer layer 0; position 1 at layer 2. After the suffix
        # forward, the spy model writes per-layer markers into the
        # BatchKVCache at indices 0 and 2 (1.0 and 3.0 respectively).
        # Both reads pin the inverse of het.1's forward mapping.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        _seed_prefix_with_markers(pc, prompt, [[100.0, 300.0]])
        batcher = _make_batcher(pattern, log, pc)

        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert batcher.prefix_hits == 1
        assert batcher._batch_cache is not None
        cache_l0 = batcher._batch_cache[0]
        cache_l2 = batcher._batch_cache[2]
        assert isinstance(cache_l0, BatchKVCache)
        assert isinstance(cache_l2, BatchKVCache)
        assert cache_l0.keys is not None
        assert cache_l2.keys is not None
        # Content length is K * BLOCK_SIZE seeded + 1 * BLOCK_SIZE suffix = 8.
        # ``cache.keys.shape[2]`` is the allocated buffer (rounded up
        # to ``BatchKVCache.step``); ``cache.offset[0]`` is the
        # authoritative content length.
        assert int(cache_l0.offset[0].item()) == 2 * BLOCK_SIZE
        assert int(cache_l2.offset[0].item()) == 2 * BLOCK_SIZE
        # Seeded portion (first BLOCK_SIZE tokens):
        # transformer layer 0 carries attn-pos 0's marker (100.0).
        seed_l0 = _read_marker(cache_l0.keys, 0)
        seed_l2 = _read_marker(cache_l2.keys, 0)
        assert seed_l0 == 100.0, (
            f"transformer layer 0 seed marker expected 100.0 (attn-pos 0); "
            f"got {seed_l0} — inverse mapping broke"
        )
        assert seed_l2 == 300.0, (
            f"transformer layer 2 seed marker expected 300.0 (attn-pos 1); "
            f"got {seed_l2} — inverse mapping broke"
        )
        # Suffix portion (next BLOCK_SIZE tokens): the spy model writes
        # marker = layer_idx + 1, so transformer layer 0 carries 1.0
        # and layer 2 carries 3.0. Confirms the suffix forward routed
        # through the heterogeneous row_cache and the spy hit the
        # correct layer dispatch.
        suffix_l0 = _read_marker(cache_l0.keys, BLOCK_SIZE)
        suffix_l2 = _read_marker(cache_l2.keys, BLOCK_SIZE)
        assert suffix_l0 == 1.0, (
            f"transformer layer 0 suffix marker expected 1.0; got {suffix_l0}"
        )
        assert suffix_l2 == 3.0, (
            f"transformer layer 2 suffix marker expected 3.0; got {suffix_l2}"
        )


# --- restore is called with the heterogeneous cache_list ---


class TestRestoreCalledWithHeterogeneousCacheList:
    def test_restore_call_layer_types_match_adapter_shape(self) -> None:
        # The single most direct guard against the seed-assembly
        # collapsing to homogeneous BatchKVCache: record the
        # cache_list's layer types AT THE TIME restore was invoked
        # (recorded by the het.1 spy's restore_recurrent_state). If
        # the assembly went wrong, all four entries would read
        # "BatchKVCache" — the assertion below specifically forbids
        # that.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        _seed_prefix_with_markers(pc, prompt, [[100.0, 300.0]])
        batcher = _make_batcher(pattern, log, pc)

        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert len(log.restore_calls) == 1, (
            f"expected exactly one restore call from Insertion B; "
            f"got {len(log.restore_calls)}"
        )
        call = log.restore_calls[0]
        assert call["row_idx"] == 0
        assert call["layer_types"] == [
            "BatchKVCache",
            "ArraysCache",
            "BatchKVCache",
            "ArraysCache",
        ], (
            f"restore was called with the wrong cache shape; "
            f"layer_types at call time: {call['layer_types']}"
        )

    def test_restore_layer_types_for_first_layer_global_pattern(
        self,
    ) -> None:
        # A different pattern (DELTANET first, then alternating) — same
        # invariant: restore sees the heterogeneous shape, never an
        # all-BatchKVCache list.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
            )
        )
        log = _SpyLog()
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        # Attention positions are [1, 3] in this pattern → 2 markers.
        _seed_prefix_with_markers(pc, prompt, [[200.0, 400.0]])
        batcher = _make_batcher(pattern, log, pc)

        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert len(log.restore_calls) == 1
        assert log.restore_calls[0]["layer_types"] == [
            "ArraysCache",
            "BatchKVCache",
            "ArraysCache",
            "BatchKVCache",
        ]
        # Inverse mapping check on this pattern too: attn-pos 0 → layer 1,
        # attn-pos 1 → layer 3. Confirms the assembly logic is
        # pattern-agnostic.
        assert batcher._batch_cache is not None
        cache_l1 = batcher._batch_cache[1]
        cache_l3 = batcher._batch_cache[3]
        assert isinstance(cache_l1, BatchKVCache)
        assert isinstance(cache_l3, BatchKVCache)
        assert cache_l1.keys is not None
        assert cache_l3.keys is not None
        assert _read_marker(cache_l1.keys, 0) == 200.0
        assert _read_marker(cache_l3.keys, 0) == 400.0


# --- pure-attention regression ---


class TestPureAttentionAdmitUnchanged:
    def test_all_attention_pattern_admit_homogeneous_row_cache(
        self,
    ) -> None:
        # All-GLOBAL pattern: every transformer layer is an attention
        # layer. The heterogeneous assembly degenerates to today's
        # behaviour — empty_row_cache is all-BatchKVCache,
        # attn_layer_indices == range(num_layers), seeded entries
        # replace every slot 1:1. row_cache is homogeneous, restore
        # sees a homogeneous cache_list.
        pattern = AttentionPattern(
            per_layer=tuple([AttentionKind.GLOBAL] * 3)
        )
        log = _SpyLog()
        pc = _prefix_cache()
        prompt = list(range(1, 2 * BLOCK_SIZE + 1))
        # 3 attention positions → 3 markers per block.
        _seed_prefix_with_markers(pc, prompt, [[10.0, 20.0, 30.0]])
        batcher = _make_batcher(pattern, log, pc)

        _prep_cohort(batcher)
        batcher.add_request(0, prompt, _params(max_tokens=1))
        batcher.step()

        assert batcher.prefix_hits == 1
        assert batcher._batch_cache is not None
        layers = batcher._batch_cache
        assert len(layers) == 3
        for layer_idx, layer in enumerate(layers):
            assert isinstance(layer, BatchKVCache), (
                f"all-GLOBAL pattern: layer {layer_idx} must be "
                f"BatchKVCache; got {type(layer).__name__}"
            )

        assert len(log.restore_calls) == 1
        assert log.restore_calls[0]["layer_types"] == [
            "BatchKVCache",
            "BatchKVCache",
            "BatchKVCache",
        ]
        # Forward-mapping holds trivially for pure-attention: pos 0 →
        # layer 0, pos 1 → layer 1, pos 2 → layer 2. Verify each
        # seeded marker landed at its same-index transformer layer.
        for layer_idx, expected in enumerate([10.0, 20.0, 30.0]):
            cache = layers[layer_idx]
            assert isinstance(cache, BatchKVCache)
            assert cache.keys is not None
            seed = _read_marker(cache.keys, 0)
            assert seed == expected, (
                f"pure-attention layer {layer_idx} seed marker expected "
                f"{expected}; got {seed}"
            )


# --- _token_kv_layer_indices with explicit cache_list ---


class TestTokenKVLayerIndicesWithCacheList:
    def test_explicit_cache_list_introspection(self) -> None:
        # Direct unit test of the het.2 ``cache_list`` arg: passing an
        # explicit list bypasses the ``self._batch_cache`` requirement,
        # which is what the admit-path uses to introspect a freshly-
        # built empty row_cache before any forward has populated
        # ``self._batch_cache``.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        pc = _prefix_cache()
        batcher = _make_batcher(pattern, log, pc)
        # No step() yet → batcher._batch_cache is None. The default
        # arg path would assert; the explicit cache_list path works.
        # Cast through the spy class — ``_make_batch_cache`` /
        # ``ModelAdapter`` Protocol intentionally omits
        # ``make_batch_cache`` (D-016 keeps the Protocol lean and
        # callers go through the ``_make_batch_cache`` helper). Tests
        # that need the empty cache directly reach the spy adapter
        # via its concrete type.
        adapter = batcher._adapter
        assert isinstance(adapter, _ScriptedHybridAdapter)
        empty = adapter.make_batch_cache([0])
        assert batcher._token_kv_layer_indices(empty) == [0, 2]

    def test_explicit_pure_recurrent_loud_fails(self) -> None:
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.HYBRID_DELTANET,
            )
        )
        log = _SpyLog()
        pc = _prefix_cache()
        batcher = _make_batcher(pattern, log, pc)
        # Cast through the spy class — ``_make_batch_cache`` /
        # ``ModelAdapter`` Protocol intentionally omits
        # ``make_batch_cache`` (D-016 keeps the Protocol lean and
        # callers go through the ``_make_batch_cache`` helper). Tests
        # that need the empty cache directly reach the spy adapter
        # via its concrete type.
        adapter = batcher._adapter
        assert isinstance(adapter, _ScriptedHybridAdapter)
        empty = adapter.make_batch_cache([0])
        with pytest.raises(
            RuntimeError, match="no token-K/V layers found"
        ):
            batcher._token_kv_layer_indices(empty)

    def test_default_arg_still_uses_self_batch_cache(self) -> None:
        # Regression: the default-arg path (no cache_list) must still
        # introspect ``self._batch_cache`` — het.1 callers pass nothing.
        pattern = AttentionPattern(
            per_layer=(
                AttentionKind.GLOBAL,
                AttentionKind.HYBRID_DELTANET,
                AttentionKind.GLOBAL,
            )
        )
        log = _SpyLog()
        pc = _prefix_cache()
        batcher = _make_batcher(pattern, log, pc)
        # Drive a step to populate _batch_cache.
        batcher.add_request(0, [1, 2, 3, 4], _params(max_tokens=1))
        batcher.step()
        assert batcher._token_kv_layer_indices() == [0, 2]
