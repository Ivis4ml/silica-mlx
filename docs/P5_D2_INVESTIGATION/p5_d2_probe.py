"""P-5-D.2 probe: measure silica BlockTQ Frobenius on pre-RoPE vs post-RoPE
Qwen3-0.6B K activations.

Forwards a fixed WikiText prefix through Qwen3-0.6B, intercepts K at
both k_proj output (pre-RoPE) and cache.update_and_fetch input
(post-RoPE), runs silica BlockTurboQuantMSE(B=64, b=4) round-trip on
each, reports per-block relative Frobenius. Also reports the Haar
rotation's own effect on pre-RoPE vs post-RoPE as a sanity check
(identity codec => bit-identical).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load

from silica.bench.wikitext import load_wikitext_text, tokenize_for_ppl
from silica.models.factory import adapter_for_repo
from silica.vq import BlockTurboQuantMSE


def relative_frobenius(ref: mx.array, approx: mx.array) -> float:
    """||approx - ref||_F / ||ref||_F over the full tensor."""
    diff = (approx.astype(mx.float32) - ref.astype(mx.float32))
    num = float(mx.sqrt(mx.sum(diff * diff)).item())
    den = float(
        mx.sqrt(
            mx.sum(ref.astype(mx.float32) * ref.astype(mx.float32))
        ).item()
    )
    return num / max(den, 1e-30)


def main() -> int:
    adapter, _kv = adapter_for_repo("Qwen/Qwen3-0.6B")
    model = adapter._model
    layout = adapter.kv_layout()
    head_dim = layout.head_dim
    n_kv_heads = layout.n_kv_heads
    num_layers = layout.num_layers
    print(f"layout: layers={num_layers} n_kv_heads={n_kv_heads} head_dim={head_dim} dtype={layout.dtype}")

    wikitext_path = "/Users/xinyu/.cache/silica/wikitext2-test.txt"
    text = load_wikitext_text(wikitext_path)
    tokenizer = adapter.tokenizer()
    tokens = tokenize_for_ppl(tokenizer, text, max_tokens=256, min_tokens=16)
    # Reshape to (1, seq_len)
    seq_len = int(tokens.shape[-1]) if tokens.ndim else 256
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    print(f"prompt tokens: shape={tokens.shape}")

    # Hook: monkey-wrap k_proj on layer 10 (mid-stack) to capture
    # pre-RoPE K. Also capture post-RoPE K from cache.update after
    # the forward pass.
    TARGET_LAYER = 10
    pre_rope_capture: dict[str, mx.array] = {}

    attn = model.layers[TARGET_LAYER].self_attn
    orig_k_proj = attn.k_proj

    class _CapturingKProj:
        def __call__(self, x: mx.array) -> mx.array:
            out = orig_k_proj(x)
            B, L, _ = out.shape
            reshaped = out.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
            pre_rope_capture["k"] = mx.stop_gradient(reshaped)
            return out

    attn.k_proj = _CapturingKProj()  # type: ignore[assignment]

    # Allocate cache and run one forward
    cache_list = adapter.make_batch_cache([0])
    from silica.mlx.runner import forward_batched_full
    logits = forward_batched_full(model, tokens, cache_list)
    mx.eval(logits)

    # Restore k_proj
    attn.k_proj = orig_k_proj  # type: ignore[assignment]

    # Extract post-RoPE K from cache at TARGET_LAYER
    layer_cache = cache_list[TARGET_LAYER]
    # mlx-lm BatchKVCache.state returns (keys, values); both shape
    # (B, n_kv_heads, seq_len, head_dim)
    post_rope_k = layer_cache.state[0]
    mx.eval(post_rope_k)
    pre_rope_k = pre_rope_capture["k"]
    mx.eval(pre_rope_k)

    print(f"pre-RoPE K:  shape={pre_rope_k.shape} dtype={pre_rope_k.dtype}")
    print(f"post-RoPE K: shape={post_rope_k.shape} dtype={post_rope_k.dtype}")

    # Slice to one aligned block (block_size=16, kvcache block)
    # BlockTurboQuantMSE expects (1, n_kv_heads, block_size, head_dim).
    block_size = 16
    # Use the first aligned block.
    pre_block = pre_rope_k[:, :, :block_size, :].astype(mx.float16)
    post_block = post_rope_k[:, :, :block_size, :].astype(mx.float16)

    # Run BlockTQ B=64 b=4 round-trip on each.
    codec = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=64,
        num_bits=4,
        seed=42,
    )
    pre_decoded = codec.decode_tensor(codec.encode_tensor(pre_block))
    post_decoded = codec.decode_tensor(codec.encode_tensor(post_block))
    mx.eval(pre_decoded)
    mx.eval(post_decoded)

    frob_pre = relative_frobenius(pre_block, pre_decoded)
    frob_post = relative_frobenius(post_block, post_decoded)
    print(f"\nBlockTurboQuantMSE(B=64, b=4, seed=42) relative Frobenius:")
    print(f"  pre-RoPE K  : {frob_pre:.6f}")
    print(f"  post-RoPE K : {frob_post:.6f}")
    print(f"  ratio       : {frob_post / max(frob_pre, 1e-30):.2f}x")

    # Also run the NumPy reference on the same input — if silica's MLX
    # BlockTQ gives a very different Frobenius from the NumPy reference,
    # silica has a port bug; if they agree, the ~0.09 Frobenius is
    # intrinsic to BlockTQ on real Qwen K (not a silica defect).
    import math
    from silica.vq._calibration import haar_rotation, lloyd_max_codebook
    rotation_np = haar_rotation(head_dim, 42)
    vq_block_size = 64
    sigma = 1.0 / math.sqrt(vq_block_size)
    centroids_np, boundaries_np = lloyd_max_codebook(4, sigma)

    import sys
    sys.path.insert(0, 'tests')
    from test_block_tq_vqbench_xcheck import _numpy_block_tq_round_trip

    def np_frobenius(block_mx: mx.array) -> float:
        # block shape (1, n_kv_heads, block_size, head_dim) -> (n_vectors, head_dim)
        arr = np.array(block_mx.astype(mx.float32)).reshape(-1, head_dim).astype(np.float64)
        recon = _numpy_block_tq_round_trip(
            arr,
            rotation=rotation_np,
            centroids=centroids_np,
            boundaries=boundaries_np,
            vq_block_size=vq_block_size,
            norm_correction=True,
        )
        num = float(np.linalg.norm(recon - arr))
        den = float(np.linalg.norm(arr))
        return num / max(den, 1e-30)

    frob_pre_np = np_frobenius(pre_block)
    frob_post_np = np_frobenius(post_block)
    print(f"\nNumPy reference BlockTQ relative Frobenius (same input):")
    print(f"  pre-RoPE K  : {frob_pre_np:.6f}")
    print(f"  post-RoPE K : {frob_post_np:.6f}")
    print(f"  silica MLX vs NumPy ratio (pre) : {frob_pre / max(frob_pre_np, 1e-30):.3f}")
    print(f"  silica MLX vs NumPy ratio (post): {frob_post / max(frob_post_np, 1e-30):.3f}")

    # Also compute norms of the inputs (Gaussian input assumption check)
    pre_np = np.array(pre_block.astype(mx.float32))
    post_np = np.array(post_block.astype(mx.float32))
    pre_std_per_coord = float(pre_np.std())
    post_std_per_coord = float(post_np.std())
    pre_norm_per_vec = float(np.linalg.norm(pre_np.reshape(-1, head_dim), axis=-1).mean())
    post_norm_per_vec = float(np.linalg.norm(post_np.reshape(-1, head_dim), axis=-1).mean())
    print(f"\nInput statistics (head_dim={head_dim}):")
    print(f"  pre-RoPE  coord std : {pre_std_per_coord:.5f}  vec-norm mean : {pre_norm_per_vec:.5f}")
    print(f"  post-RoPE coord std : {post_std_per_coord:.5f}  vec-norm mean : {post_norm_per_vec:.5f}")
    # N(0, 1/d) Gaussian assumption: coord std should be ~1/sqrt(d) ~= 0.0884
    # for head_dim=128. Big departure from that breaks Lloyd-Max codebook fit.
    print(f"  reference std for N(0, 1/d): {(1.0 / head_dim) ** 0.5:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
