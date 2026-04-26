"""P-3-C5-step2-W — bench oracle codec PPL on hybrid (Qwen3.5).

Pre-W ``teacher_forced_chunked_nll_with_codec`` was hardcoded to
the pure-attention shape: built ``cache_list`` via
``build_seeded_batch_kv(detached, num_layers=total)`` (all
``BatchKVCache``) and extracted from every layer in
``_extract_aligned_blocks_from_seeded_cache``. On a hybrid adapter
(Qwen3.5: DeltaNet ``ArraysCache`` + full-attention
``BatchKVCache`` interleaved) the extract path tripped on
``ArraysCache.keys`` (no such attribute), and even bypassing that
the all-``BatchKVCache`` cache shape would mismatch the model's
expected heterogeneous layout.

W's bench-layer fix (``silica/bench/ppl_oracle.py``):

- ``_extract_aligned_blocks_from_seeded_cache`` introspects each
  layer's type and only extracts from ``BatchKVCache`` layers.
- ``teacher_forced_chunked_nll_with_codec``:
  - Tracks ``attn_layer_indices`` from the cold cache.
  - At hit chunks, builds an empty heterogeneous cache via
    ``adapter.make_batch_cache([0])`` and interleaves seeded
    ``BatchKVCache`` (one per attention layer) into the
    attention positions; DeltaNet positions stay empty
    ``ArraysCache``.
  - For ``RecurrentStateAdapter`` adapters, captures the
    post-forward recurrent snapshot and restores it onto the
    next chunk's freshly-built cache (advisor sharpening: the
    snapshot/restore option mirrors C5.3's production capture
    contract — the same per-block snapshot that C5.3.3b validates
    byte-exact in the scheduler).

Real-model smoke (HF-cache-skip-gated) drives Qwen3.5-0.8B
through both paths on a wikitext-2 prefix and asserts:

- fp16 PPL via ``teacher_forced_chunked_nll`` yields a finite
  number.
- Codec PPL via ``teacher_forced_chunked_nll_with_codec`` yields a
  finite number.
- ``BlockTurboQuantMSE`` 4-bit ΔPPL stays within a generous gate
  (< 1.0 absolute on a 128-token prefix — production codec is
  near-lossless; the gate catches gross regressions like a
  recurrent-state reset that would inflate PPL by 2-3x).
- ``IdentityCodec`` ΔPPL is near zero (lossless round trip; the
  remaining noise is fp accumulation across the codec's
  encode→decode pipeline).
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from silica.bench.codec_registry import get_codec_spec
from silica.bench.ppl_oracle import (
    perplexity_from_nll,
    teacher_forced_chunked_nll,
    teacher_forced_chunked_nll_with_codec,
)
from silica.kvcache.codec import IdentityCodec
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore

REPO = "Qwen/Qwen3.5-0.8B"
BLOCK_SIZE = 16
CHUNK_SIZE = 64
SEQ_LEN = 128

_HF_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3.5-0.8B"
)
_WIKITEXT = Path.home() / ".cache" / "silica" / "wikitext2-test.txt"
_SKIP_REASON = (
    f"Qwen3.5-0.8B not cached at {_HF_CACHE}, or wikitext-2 not "
    f"cached at {_WIKITEXT}; populate with scripts/probe_qwen3_5_load.py "
    f"and scripts/prepare_wikitext2_cache.py."
)
_SKIP = (
    not _HF_CACHE.exists()
    or not _WIKITEXT.exists()
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)


def _load_tokens(adapter: object) -> mx.array:
    with open(_WIKITEXT) as f:
        text = f.read()
    tokenizer = adapter.tokenizer()  # type: ignore[attr-defined]
    return mx.array(
        [tokenizer.encode(text)[:SEQ_LEN]], dtype=mx.int32
    )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_codec_ppl_runs_on_qwen3_5_0_8b_hybrid_with_block_tq() -> None:
    """W gate: codec PPL drives Qwen3.5-0.8B end-to-end without
    crashing on the hybrid cache shape. Pre-W this raised
    ``AttributeError: 'ArraysCache' object has no attribute 'keys'``
    inside ``_extract_aligned_blocks_from_seeded_cache``.
    """
    from silica.models.factory import adapter_for_repo

    adapter, _ = adapter_for_repo(REPO)
    layout = adapter.kv_layout()
    spec = get_codec_spec("block_tq_b64_b4")
    codec = spec.factory(
        block_size=BLOCK_SIZE,
        n_kv_heads=layout.n_kv_heads,
        head_dim=layout.head_dim,
        dtype=layout.dtype,
        seed=0,
    )
    pc = RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(
            block_size=BLOCK_SIZE, codec=codec
        ),
    )
    token_ids = _load_tokens(adapter)

    nll, n_tokens = teacher_forced_chunked_nll_with_codec(
        adapter, pc, token_ids, chunk_size=CHUNK_SIZE
    )
    ppl = perplexity_from_nll(nll, n_tokens)
    assert n_tokens == SEQ_LEN - 1
    # PPL should be finite and positive; wikitext on a 128-token
    # prefix lands in the 5-50 range across model sizes.
    assert 1.0 < ppl < 1e6, f"ppl out of plausible range: {ppl}"


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_block_tq_b4_delta_ppl_is_bounded_on_qwen3_5_0_8b_hybrid() -> None:
    """ΔPPL gate: ``block_tq_b64_b4`` is near-lossless under the
    production C5.3 capture/restore contract. Bound the residual
    against fp16 to catch regressions that would inflate PPL —
    notably a recurrent-state reset between chunks (W's
    snapshot/restore is the load-bearing piece for hybrid).
    """
    from silica.models.factory import adapter_for_repo

    adapter, _ = adapter_for_repo(REPO)
    layout = adapter.kv_layout()
    token_ids = _load_tokens(adapter)

    fp16_nll, fp16_n = teacher_forced_chunked_nll(
        adapter, token_ids, chunk_size=CHUNK_SIZE
    )
    fp16_ppl = perplexity_from_nll(fp16_nll, fp16_n)

    spec = get_codec_spec("block_tq_b64_b4")
    codec = spec.factory(
        block_size=BLOCK_SIZE,
        n_kv_heads=layout.n_kv_heads,
        head_dim=layout.head_dim,
        dtype=layout.dtype,
        seed=0,
    )
    pc = RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(
            block_size=BLOCK_SIZE, codec=codec
        ),
    )
    codec_nll, codec_n = teacher_forced_chunked_nll_with_codec(
        adapter, pc, token_ids, chunk_size=CHUNK_SIZE
    )
    codec_ppl = perplexity_from_nll(codec_nll, codec_n)

    delta = codec_ppl - fp16_ppl
    # Generous absolute gate (< 1.0 PPL). A recurrent-state reset
    # between chunks would push delta to several PPL even on this
    # short prefix; the gate fires loudly on that regression.
    assert abs(delta) < 1.0, (
        f"ΔPPL out of bounds: fp16={fp16_ppl:.4f} codec="
        f"{codec_ppl:.4f} delta={delta:+.4f}"
    )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_identity_codec_ppl_matches_fp16_baseline_on_qwen3_5_0_8b_hybrid(
) -> None:
    """Regression equivalence: under ``IdentityCodec`` the codec
    PPL must match the fp16 baseline to fp tolerance — the codec
    is a no-op round trip, and W's recurrent snapshot/restore at
    chunk boundaries should reproduce the fp16 trajectory exactly
    (modulo fp accumulation in the codec's pass-through).
    """
    from silica.models.factory import adapter_for_repo

    adapter, _ = adapter_for_repo(REPO)
    layout = adapter.kv_layout()
    token_ids = _load_tokens(adapter)

    fp16_nll, fp16_n = teacher_forced_chunked_nll(
        adapter, token_ids, chunk_size=CHUNK_SIZE
    )
    fp16_ppl = perplexity_from_nll(fp16_nll, fp16_n)

    codec = IdentityCodec(
        block_size=BLOCK_SIZE,
        n_kv_heads=layout.n_kv_heads,
        head_dim=layout.head_dim,
        dtype=layout.dtype,
    )
    pc = RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(
            block_size=BLOCK_SIZE, codec=codec
        ),
    )
    codec_nll, codec_n = teacher_forced_chunked_nll_with_codec(
        adapter, pc, token_ids, chunk_size=CHUNK_SIZE
    )
    codec_ppl = perplexity_from_nll(codec_nll, codec_n)

    # Tighter tolerance on identity round trip — the codec doesn't
    # alter values, so any drift comes from the W path's cache
    # rebuild (heterogeneous cache assembly + recurrent snapshot
    # restore). On a 128-token prefix the residual should be < 0.1
    # PPL; pre-W the recurrent reset would have produced several
    # PPL of drift even under identity.
    delta = codec_ppl - fp16_ppl
    assert abs(delta) < 0.1, (
        f"identity-codec PPL drifts from fp16 baseline: "
        f"fp16={fp16_ppl:.6f} codec={codec_ppl:.6f} delta={delta:+.6f}"
    )
