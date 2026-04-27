"""P-5-D.2 probe 3: full silica prefix-cache round-trip Frobenius.

Checks if insert_detached -> fetch_detached -> build_seeded_batch_kv
introduces any noise beyond the codec's encode/decode. If the full
round trip gives the same Frobenius as direct encode/decode, silica's
cache machinery is numerically neutral. If it's worse, there is
extra drift somewhere in the pipeline."""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from silica.bench.wikitext import load_wikitext_text, tokenize_for_ppl
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.mlx.runner import forward_batched_full
from silica.models.factory import adapter_for_repo
from silica.scheduler.seed_kv import build_seeded_batch_kv
from silica.vq import BlockTurboQuantMSE


def frobenius(ref: mx.array, approx: mx.array) -> float:
    diff = (approx.astype(mx.float32) - ref.astype(mx.float32))
    num = float(mx.sqrt(mx.sum(diff * diff)).item())
    den = float(
        mx.sqrt(
            mx.sum(ref.astype(mx.float32) * ref.astype(mx.float32))
        ).item()
    )
    return num / max(den, 1e-30)


def main() -> int:
    adapter, _ = adapter_for_repo("Qwen/Qwen3-0.6B")
    model = adapter._model
    layout = adapter.kv_layout()
    head_dim = layout.head_dim
    n_kv_heads = layout.n_kv_heads
    num_layers = layout.num_layers
    block_size = 16

    text = load_wikitext_text("/Users/xinyu/.cache/silica/wikitext2-test.txt")
    tokens = tokenize_for_ppl(adapter.tokenizer(), text, max_tokens=256, min_tokens=16)
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    seq_len = tokens.shape[-1]
    print(f"prompt seq_len={seq_len}")

    # Forward once with fresh fp16 cache to capture reference K at all
    # layers, all 16 aligned blocks.
    ref_cache_list = adapter.make_batch_cache([0])
    _logits = forward_batched_full(model, tokens, ref_cache_list)
    mx.eval(_logits)
    reference_k_per_layer: list[mx.array] = []  # each (1, n_kv_heads, 256, head_dim)
    reference_v_per_layer: list[mx.array] = []
    for layer_idx in range(num_layers):
        k, v, *_ = ref_cache_list[layer_idx].state
        reference_k_per_layer.append(mx.array(k))
        reference_v_per_layer.append(mx.array(v))

    # === Path 1: direct codec encode/decode on layer-0 block-0 K ===
    codec = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=64,
        num_bits=4,
        seed=42,
    )
    layer = 0
    block = 0
    ref_k_block = reference_k_per_layer[layer][:, :, block * block_size : (block + 1) * block_size, :].astype(mx.float16)
    direct_decoded = codec.decode_tensor(codec.encode_tensor(ref_k_block))
    mx.eval(direct_decoded)
    frob_direct = frobenius(ref_k_block, direct_decoded)
    print(f"\n[direct codec] layer 0 block 0 K Frobenius: {frob_direct:.6f}")

    # === Path 2: full silica round trip via RadixPrefixCache ===
    # Build prefix cache w/ block_tq_b64_b4 via factory
    from silica.bench.codec_registry import get_codec_spec
    spec = get_codec_spec("block_tq_b64_b4")
    codec_k = spec.factory(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=mx.bfloat16,  # Qwen3-0.6B runtime dtype
        seed=42,
    )
    codec_v = spec.factory(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=mx.bfloat16,
        seed=42,
    )
    store = SyntheticPrefixBlockStore(
        block_size=block_size,
        k_codec=codec_k,
        v_codec=codec_v,
    )
    prefix_cache = RadixPrefixCache(block_size=block_size, store=store)

    # Pack reference K/V as [block_idx][layer_idx] -> (K, V) per-block
    num_blocks = seq_len // block_size
    detached_blocks: list[list[tuple[mx.array, mx.array]]] = []
    for b in range(num_blocks):
        per_layer: list[tuple[mx.array, mx.array]] = []
        for layer_idx in range(num_layers):
            k_slice = reference_k_per_layer[layer_idx][:, :, b * block_size : (b + 1) * block_size, :]
            v_slice = reference_v_per_layer[layer_idx][:, :, b * block_size : (b + 1) * block_size, :]
            per_layer.append((mx.array(k_slice), mx.array(v_slice)))
        detached_blocks.append(per_layer)

    token_list = tokens[0].tolist()
    prefix_cache.insert_detached(token_list[: num_blocks * block_size], detached_blocks)

    # Now fetch back
    hit = prefix_cache.lookup(token_list[: num_blocks * block_size])
    assert hit.num_hit_tokens == num_blocks * block_size
    try:
        fetched_blocks = prefix_cache.fetch_detached_blocks(hit.block_ids)
        # fetched_blocks[block_idx][layer_idx] -> (K_decoded, V_decoded)
        # Compare layer 0 block 0
        fetched_k = fetched_blocks[0][0][0]  # block 0, layer 0, K
        mx.eval(fetched_k)
        # direct reference for this slice:
        ref_k_block_bf16 = reference_k_per_layer[0][:, :, :block_size, :].astype(mx.bfloat16)
        frob_cache = frobenius(ref_k_block_bf16, fetched_k)
        print(f"[cache round-trip] layer 0 block 0 K Frobenius: {frob_cache:.6f}")
        # Compare direct codec encode/decode at bf16 to rule out dtype
        codec_bf16 = BlockTurboQuantMSE(
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            vq_block_size=64,
            num_bits=4,
            seed=42,
            dtype=mx.bfloat16,
        )
        direct_bf16 = codec_bf16.decode_tensor(codec_bf16.encode_tensor(ref_k_block_bf16))
        mx.eval(direct_bf16)
        frob_direct_bf16 = frobenius(ref_k_block_bf16, direct_bf16)
        print(f"[direct codec, bf16] layer 0 block 0 K Frobenius: {frob_direct_bf16:.6f}")
        # Compare fetched vs direct — should be IDENTICAL if no drift
        d = (fetched_k.astype(mx.float32) - direct_bf16.astype(mx.float32))
        max_absdiff = float(mx.max(mx.abs(d)).item())
        print(f"max |fetched - direct_bf16| = {max_absdiff:.6e}")
    finally:
        prefix_cache.release(hit.block_ids)

    # === Build seeded cache and measure offset/state sanity ===
    seeded_caches = build_seeded_batch_kv(fetched_blocks, num_layers=num_layers)
    k_seeded_layer0 = seeded_caches[0].state[0]  # (1, n_kv_heads, 256, head_dim)
    ref_k_full = reference_k_per_layer[0].astype(mx.bfloat16)
    frob_seeded = frobenius(ref_k_full, k_seeded_layer0)
    print(f"[seeded cache] layer 0 full K Frobenius: {frob_seeded:.6f}")
    print(f"seeded cache.offset = {seeded_caches[0].offset}")
    print(f"seeded cache._idx   = {seeded_caches[0]._idx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
