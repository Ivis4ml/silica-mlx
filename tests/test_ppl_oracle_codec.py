"""P-5-C.2 unit tests for
:func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll_with_codec`.

Four invariants pinned here:

1. **Identity-codec equivalence.** Under
   :class:`silica.kvcache.codec.IdentityCodec` the encode/decode round
   trip is lossless, so the codec-backed C.2 oracle must agree with
   the shared-cache C.1 fp16 baseline to fp tolerance on the same
   token stream. If the two drift it means the seed-cache-per-chunk
   path produces different K/V than the shared-cache-across-chunks
   path — the C.2 arm would be silently biased before ever meeting a
   lossy codec.

2. **Codec hot path actually fires.** A counting wrapper around
   :class:`IdentityCodec` reveals that every chunk ≥ 1 runs
   ``decode_tensor`` on the prefix blocks it consumes and
   ``encode_tensor`` on the newly-computed blocks it inserts. A PPL
   run that quietly bypasses the codec — e.g. because the oracle
   allocates a fresh ``BatchKVCache`` and never re-routes through
   ``prefix_cache`` — would report fp16 numbers under a quantized
   codec configuration, the exact failure mode the counting harness
   is there to catch.

3. **Alignment contract.** ``chunk_size % block_size == 0`` is
   required; the oracle raises otherwise. Misaligned chunk sizes
   leave a sub-block tail outside the prefix cache's block-granular
   coverage and would drop K/V between chunks.

4. **Shape / batch guards mirror C.1.** Same 1-D rejection, same
   ``B != 1`` rejection, same non-positive ``chunk_size`` rejection
   as :func:`teacher_forced_chunked_nll`.

A tiny BatchKVCache-compatible fake adapter + fake model live in
this file (not shared with :mod:`tests.test_ppl_oracle` — the C.1
``_FakeCache`` is a plain Python offset-counter; the C.2 path routes
through ``build_seeded_batch_kv`` which produces a real
``mlx_lm.models.cache.BatchKVCache``, so the fake model has to speak
that interface instead).
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import BatchKVCache

from silica.bench.ppl_oracle import (
    teacher_forced_chunked_nll,
    teacher_forced_chunked_nll_with_codec,
)
from silica.kvcache.codec import IdentityCodec, RawFp16Payload, VectorCodec
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore

# =============================================================================
# Tiny BatchKVCache-compatible fake model + adapter
# =============================================================================


_VOCAB = 8
_N_KV_HEADS = 2
_HEAD_DIM = 8
_BLOCK_SIZE = 4  # small enough that tiny sequences exercise 2-3 blocks
_N_LAYERS = 1


class _BatchKVCacheCompatModel:
    """Fake model that plays nicely with ``BatchKVCache``.

    - Reads absolute sequence position via the cache's own
      ``offset`` tensor, so the logits value at each emitted
      position depends on the **pre-forward** cache state.
    - Drives ``update_and_fetch`` with deterministic K/V so the cache
      accumulates state the way a real mlx-lm attention module would
      (the C.2 oracle slices that state back out after the forward).

    Logits formula mirrors the C.1 test's ``_FakeModel``:
    ``cos(pi * (pos + 1) / 32 * (v + 1) / V) * 3`` — deterministic
    function of absolute position; any seeded-cache vs shared-cache
    drift would change the per-token CE.
    """

    N_LAYERS = _N_LAYERS
    VOCAB = _VOCAB
    N_KV_HEADS = _N_KV_HEADS
    HEAD_DIM = _HEAD_DIM

    def __call__(
        self, tokens: mx.array, cache: list[BatchKVCache]
    ) -> mx.array:
        B, T = tokens.shape
        cache_obj = cache[0]

        # BatchKVCache.offset is an (B,) mx.array populated either by
        # build_seeded_batch_kv or by prior update_and_fetch calls.
        pre_offset = int(cache_obj.offset[0].item())

        # Deterministic K/V that depends on absolute position — exact
        # value does not matter beyond "the function is a function of
        # position so a different seed would show up in downstream CE".
        positions = (mx.arange(T, dtype=mx.float16) + float(pre_offset))
        k = mx.broadcast_to(
            positions[None, None, :, None],
            (B, self.N_KV_HEADS, T, self.HEAD_DIM),
        ).astype(mx.float16)
        v = (k + mx.array(0.125, dtype=mx.float16)).astype(mx.float16)
        cache_obj.update_and_fetch(k, v)

        # Logits: position-dependent, matches C.1 fake model formula.
        positions_fp32 = (
            mx.arange(T, dtype=mx.float32) + float(pre_offset)
        )
        v_axis = mx.arange(self.VOCAB, dtype=mx.float32)
        pos_scaled = (positions_fp32 + 1.0) / 32.0
        v_scaled = (v_axis + 1.0) / float(self.VOCAB)
        grid = pos_scaled[:, None] * v_scaled[None, :]
        per_pos_logits = mx.cos(grid * math.pi) * 3.0
        logits = mx.broadcast_to(
            per_pos_logits[None, :, :], (B, T, self.VOCAB)
        )
        return logits


class _BatchKVCacheCompatAdapter:
    """Adapter exposing ``_model`` + ``make_batch_cache([0])`` returning
    real ``BatchKVCache`` instances (one per layer)."""

    def __init__(self) -> None:
        self._model = _BatchKVCacheCompatModel()

    def make_batch_cache(
        self, left_padding: list[int]
    ) -> list[BatchKVCache]:
        assert left_padding == [0], (
            f"fake adapter supports left_padding=[0] only; got {left_padding}"
        )
        return [
            BatchKVCache(left_padding=list(left_padding))
            for _ in range(_N_LAYERS)
        ]


# =============================================================================
# Counting codec wrapper — observes encode_tensor / decode_tensor calls
# without altering the payload (delegates to an inner IdentityCodec).
# =============================================================================


class _CountingVectorCodec:
    """Wraps a :class:`VectorCodec` and counts encode/decode calls.

    Proxies every ``VectorCodec`` method to the inner codec; the only
    observable additions are ``encode_calls`` / ``decode_calls``
    counters. Used by the codec-call-count test to prove the oracle
    actually routes every chunk through the codec hot path.
    """

    def __init__(self, inner: VectorCodec[RawFp16Payload]) -> None:
        self._inner = inner
        self.block_size: int = inner.block_size
        self.dtype: mx.Dtype = inner.dtype
        self.encode_calls = 0
        self.decode_calls = 0

    def encode_tensor(self, x: mx.array) -> RawFp16Payload:
        self.encode_calls += 1
        return self._inner.encode_tensor(x)

    def decode_tensor(self, payload: RawFp16Payload) -> mx.array:
        self.decode_calls += 1
        return self._inner.decode_tensor(payload)

    def logical_bytes(self, num_tokens: int) -> int:
        return self._inner.logical_bytes(num_tokens)

    def resident_bytes(self, num_blocks: int) -> int:
        return self._inner.resident_bytes(num_blocks)


# =============================================================================
# Fixtures
# =============================================================================


def _make_tokens(seq_len: int, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, _VOCAB, size=(1, seq_len), dtype=np.int32)
    return mx.array(ids)


def _make_prefix_cache(
    *,
    k_codec: Any = None,
    v_codec: Any = None,
) -> RadixPrefixCache:
    """Build an empty ``RadixPrefixCache`` over a synthetic store.

    ``k_codec`` / ``v_codec`` default to fresh ``IdentityCodec``s that
    match this test's fake-adapter shape; override with
    ``_CountingVectorCodec`` wrappers to inspect call counts.
    """
    effective_k = k_codec if k_codec is not None else IdentityCodec(
        block_size=_BLOCK_SIZE,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
    )
    effective_v = v_codec if v_codec is not None else IdentityCodec(
        block_size=_BLOCK_SIZE,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
    )
    store = SyntheticPrefixBlockStore(
        block_size=_BLOCK_SIZE,
        k_codec=effective_k,
        v_codec=effective_v,
    )
    return RadixPrefixCache(block_size=_BLOCK_SIZE, store=store)


# =============================================================================
# Tests
# =============================================================================


class TestIdentityCodecEquivalence:
    """Under identity codec the C.2 codec-backed oracle must agree with
    the C.1 fp16 baseline to fp tolerance. If the two drift it means
    seed-cache-per-chunk produces different K/V than shared-cache-
    across-chunks, and the C.2 arm would be silently biased before
    ever meeting a lossy codec."""

    @pytest.mark.parametrize("chunk_size", [_BLOCK_SIZE, _BLOCK_SIZE * 2, _BLOCK_SIZE * 4])
    def test_matches_fp16_baseline(self, chunk_size: int) -> None:
        adapter = _BatchKVCacheCompatAdapter()
        tokens = _make_tokens(seq_len=chunk_size * 3 + 2, seed=123)

        # Fresh adapter per call — the C.1 oracle mutates its cache.
        nll_c1, n_c1 = teacher_forced_chunked_nll(
            _BatchKVCacheCompatAdapter(), tokens, chunk_size=chunk_size
        )
        prefix_cache = _make_prefix_cache()
        nll_c2, n_c2 = teacher_forced_chunked_nll_with_codec(
            adapter, prefix_cache, tokens, chunk_size=chunk_size
        )

        assert n_c1 == n_c2, (
            f"scored-token count mismatch: C.1 {n_c1} vs C.2 {n_c2}"
        )
        assert nll_c2 == pytest.approx(nll_c1, rel=1e-5, abs=1e-5), (
            f"identity-codec C.2 NLL {nll_c2:.6f} does not match "
            f"fp16 baseline C.1 NLL {nll_c1:.6f}"
        )


class TestCodecCallCounts:
    """``encode_tensor`` / ``decode_tensor`` must fire on the codec
    hot path. A quantized PPL row that silently equals fp16 because
    no encode/decode ever ran is the specific failure this test
    catches."""

    def test_encode_fires_on_chunk_zero(self) -> None:
        """Cold chunk 0 inserts its aligned blocks — ``encode_tensor``
        fires for every (block, layer, side) cell."""
        counting_k = _CountingVectorCodec(
            IdentityCodec(
                block_size=_BLOCK_SIZE,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
            )
        )
        counting_v = _CountingVectorCodec(
            IdentityCodec(
                block_size=_BLOCK_SIZE,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
            )
        )
        prefix_cache = _make_prefix_cache(
            k_codec=counting_k, v_codec=counting_v
        )
        adapter = _BatchKVCacheCompatAdapter()
        # chunk_size == block_size, seq_len == 2 blocks → chunk 0 only
        # fills the first block; no chunk 1 needed, so no decode yet.
        tokens = _make_tokens(_BLOCK_SIZE)

        teacher_forced_chunked_nll_with_codec(
            adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE
        )

        # 1 aligned block × 1 layer × 1 side = 1 encode call per side.
        assert counting_k.encode_calls == 1
        assert counting_v.encode_calls == 1
        assert counting_k.decode_calls == 0
        assert counting_v.decode_calls == 0

    def test_decode_fires_on_chunks_one_and_later(self) -> None:
        """Chunk 1+ seeds its cache from prefix blocks —
        ``decode_tensor`` fires for every (block, layer, side) cell
        on the hit path."""
        counting_k = _CountingVectorCodec(
            IdentityCodec(
                block_size=_BLOCK_SIZE,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
            )
        )
        counting_v = _CountingVectorCodec(
            IdentityCodec(
                block_size=_BLOCK_SIZE,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
            )
        )
        prefix_cache = _make_prefix_cache(
            k_codec=counting_k, v_codec=counting_v
        )
        adapter = _BatchKVCacheCompatAdapter()
        # chunk_size = block_size, seq_len = 3 blocks → 3 chunks.
        # Chunk 0: cold, encode 1 new block (= 1 per side).
        # Chunk 1: decode 1-block prefix, insert 2 blocks (1 touched,
        # 1 new encode) → decode + 1 new encode per side.
        # Chunk 2: decode 2-block prefix, insert 3 blocks (2 touched,
        # 1 new encode) → 2 new decodes + 1 new encode per side.
        seq_len = _BLOCK_SIZE * 3
        tokens = _make_tokens(seq_len)

        teacher_forced_chunked_nll_with_codec(
            adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE
        )

        # Encodes: 1 (chunk 0) + 1 (chunk 1 new) + 1 (chunk 2 new) = 3.
        assert counting_k.encode_calls == 3
        assert counting_v.encode_calls == 3
        # Decodes: chunk 1 fetches 1 block + chunk 2 fetches 2 blocks = 3.
        assert counting_k.decode_calls == 3
        assert counting_v.decode_calls == 3

    def test_both_sides_fire_symmetrically_under_symmetric_codec(
        self,
    ) -> None:
        """K and V counts track identically under a symmetric codec —
        any drift would indicate the oracle's extract/insert logic
        drops one side."""
        counting_k = _CountingVectorCodec(
            IdentityCodec(
                block_size=_BLOCK_SIZE,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
            )
        )
        counting_v = _CountingVectorCodec(
            IdentityCodec(
                block_size=_BLOCK_SIZE,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
            )
        )
        prefix_cache = _make_prefix_cache(
            k_codec=counting_k, v_codec=counting_v
        )
        tokens = _make_tokens(_BLOCK_SIZE * 4)

        teacher_forced_chunked_nll_with_codec(
            _BatchKVCacheCompatAdapter(),
            prefix_cache,
            tokens,
            chunk_size=_BLOCK_SIZE * 2,
        )

        assert counting_k.encode_calls == counting_v.encode_calls
        assert counting_k.decode_calls == counting_v.decode_calls
        assert counting_k.encode_calls > 0
        assert counting_k.decode_calls > 0


class TestInputValidation:
    def test_rejects_1d_token_ids(self) -> None:
        adapter = _BatchKVCacheCompatAdapter()
        prefix_cache = _make_prefix_cache()
        tokens = mx.array([1, 2, 3], dtype=mx.int32)
        with pytest.raises(ValueError, match="must be 2-D"):
            teacher_forced_chunked_nll_with_codec(
                adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE
            )

    def test_rejects_batch_greater_than_one(self) -> None:
        adapter = _BatchKVCacheCompatAdapter()
        prefix_cache = _make_prefix_cache()
        tokens = mx.zeros((2, _BLOCK_SIZE), dtype=mx.int32)
        with pytest.raises(ValueError, match="B=1"):
            teacher_forced_chunked_nll_with_codec(
                adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE
            )

    def test_rejects_zero_chunk_size(self) -> None:
        adapter = _BatchKVCacheCompatAdapter()
        prefix_cache = _make_prefix_cache()
        tokens = _make_tokens(_BLOCK_SIZE)
        with pytest.raises(ValueError, match="chunk_size"):
            teacher_forced_chunked_nll_with_codec(
                adapter, prefix_cache, tokens, chunk_size=0
            )

    def test_rejects_chunk_size_not_multiple_of_block_size(self) -> None:
        adapter = _BatchKVCacheCompatAdapter()
        prefix_cache = _make_prefix_cache()
        tokens = _make_tokens(_BLOCK_SIZE * 2)
        # _BLOCK_SIZE = 4; 5 is not a multiple.
        with pytest.raises(ValueError, match="must be a positive multiple"):
            teacher_forced_chunked_nll_with_codec(
                adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE + 1
            )

    def test_chunk_size_equals_block_size_is_allowed(self) -> None:
        """``chunk_size == block_size`` is the tightest alignment; one
        block per chunk."""
        adapter = _BatchKVCacheCompatAdapter()
        prefix_cache = _make_prefix_cache()
        tokens = _make_tokens(_BLOCK_SIZE * 2)
        nll, n = teacher_forced_chunked_nll_with_codec(
            adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE
        )
        assert n == _BLOCK_SIZE * 2 - 1
        assert math.isfinite(nll)


class TestEdgeCases:
    def test_empty_sequence_returns_zero(self) -> None:
        adapter = _BatchKVCacheCompatAdapter()
        prefix_cache = _make_prefix_cache()
        tokens = mx.zeros((1, 0), dtype=mx.int32)
        nll, n = teacher_forced_chunked_nll_with_codec(
            adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE
        )
        assert nll == 0.0
        assert n == 0

    def test_single_token_sequence_scores_zero(self) -> None:
        adapter = _BatchKVCacheCompatAdapter()
        prefix_cache = _make_prefix_cache()
        tokens = _make_tokens(1)
        nll, n = teacher_forced_chunked_nll_with_codec(
            adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE
        )
        # seq_len=1 has no prior-context tokens to score.
        assert n == 0
        assert nll == 0.0

    def test_sequence_shorter_than_block_size_scores_without_codec(
        self,
    ) -> None:
        """If the whole sequence is shorter than one block, no aligned
        blocks exist to extract; the cold-path forward still runs and
        the oracle scores ``seq_len - 1`` tokens."""
        counting_k = _CountingVectorCodec(
            IdentityCodec(
                block_size=_BLOCK_SIZE,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
            )
        )
        counting_v = _CountingVectorCodec(
            IdentityCodec(
                block_size=_BLOCK_SIZE,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
            )
        )
        prefix_cache = _make_prefix_cache(
            k_codec=counting_k, v_codec=counting_v
        )
        adapter = _BatchKVCacheCompatAdapter()
        # block_size=4, chunk_size=4, seq_len=3 → one chunk, sub-block
        # tail, no extract.
        tokens = _make_tokens(_BLOCK_SIZE - 1)
        nll, n = teacher_forced_chunked_nll_with_codec(
            adapter, prefix_cache, tokens, chunk_size=_BLOCK_SIZE
        )
        assert n == _BLOCK_SIZE - 2
        assert math.isfinite(nll)
        # No aligned block covered → no encode / decode fired.
        assert counting_k.encode_calls == 0
        assert counting_v.encode_calls == 0
        assert counting_k.decode_calls == 0
        assert counting_v.decode_calls == 0
