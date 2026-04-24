"""P-5-D.2 probe 2: shared-rotation (silica) vs per-head-rotation (vqbench)
Frobenius on real Qwen3-0.6B post-RoPE K."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load  # noqa: F401

from silica.bench.wikitext import load_wikitext_text, tokenize_for_ppl
from silica.models.factory import adapter_for_repo
from silica.mlx.runner import forward_batched_full
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
    block_size = 16
    vq_block_size = 64

    text = load_wikitext_text("/Users/xinyu/.cache/silica/wikitext2-test.txt")
    tokens = tokenize_for_ppl(adapter.tokenizer(), text, max_tokens=256, min_tokens=16)
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)

    cache_list = adapter.make_batch_cache([0])
    _logits = forward_batched_full(model, tokens, cache_list)
    mx.eval(_logits)

    # Grab layer-10 post-RoPE K, one aligned block_size slice
    k = cache_list[10].state[0][:, :, :block_size, :].astype(mx.float16)
    mx.eval(k)
    print(f"real K block: shape={k.shape} dtype={k.dtype}")

    # === Silica default: one shared Haar rotation across all heads ===
    codec_shared = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=vq_block_size,
        num_bits=4,
        seed=42,
    )
    shared_out = codec_shared.decode_tensor(codec_shared.encode_tensor(k))
    mx.eval(shared_out)
    frob_shared_total = frobenius(k, shared_out)
    print(f"\n[shared rotation] total Frobenius: {frob_shared_total:.6f}")
    for h in range(n_kv_heads):
        k_h = k[:, h : h + 1, :, :]
        out_h = shared_out[:, h : h + 1, :, :]
        print(f"  head {h}: Frobenius = {frobenius(k_h, out_h):.6f}")

    # === vqbench-style: per-head independent Haar rotation ===
    # Run each head through its own codec instance with seed=h (matches
    # quant_dequant_tensor's `q = quantizer_factory(D, seed=h)`).
    print(f"\n[per-head rotation, seed=h] per-head Frobenius:")
    per_head_outs: list[mx.array] = []
    for h in range(n_kv_heads):
        codec_h = BlockTurboQuantMSE(
            block_size=block_size,
            n_kv_heads=1,
            head_dim=head_dim,
            vq_block_size=vq_block_size,
            num_bits=4,
            seed=h,
        )
        k_h = k[:, h : h + 1, :, :]
        out_h = codec_h.decode_tensor(codec_h.encode_tensor(k_h))
        mx.eval(out_h)
        per_head_outs.append(out_h)
        print(f"  head {h}: Frobenius = {frobenius(k_h, out_h):.6f}")
    per_head_full = mx.concatenate(per_head_outs, axis=1)
    mx.eval(per_head_full)
    frob_per_head_total = frobenius(k, per_head_full)
    print(f"\n[per-head rotation] total Frobenius: {frob_per_head_total:.6f}")
    print(f"  shared / per-head ratio: {frob_shared_total / max(frob_per_head_total, 1e-30):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
