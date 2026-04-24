"""P-5-D.2 probe 4: vqbench NumPy BlockTurboQuantMSE Frobenius on real K.

If vqbench's NumPy codec gives lower Frobenius than silica's MLX codec
on the same real K input, silica's port has a bug. If they agree,
the 9% Frobenius is algorithmic and not a silica defect.

Run under miniconda Python (vqbench deps + torch + transformers).
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/Users/xinyu/Desktop/silica-mlx/vqbench')
sys.path.insert(0, '/Users/xinyu/Desktop/silica-mlx')

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE as VqbenchBlockTQ


def frobenius(ref: np.ndarray, approx: np.ndarray) -> float:
    diff = approx.astype(np.float64) - ref.astype(np.float64)
    num = float(np.linalg.norm(diff))
    den = float(np.linalg.norm(ref.astype(np.float64)))
    return num / max(den, 1e-30)


def main() -> int:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    with open("/Users/xinyu/.cache/silica/wikitext2-test.txt", "r") as f:
        text = f.read()
    ids = tok(text, return_tensors="pt")["input_ids"][:, :256]
    print(f"tokens: shape={tuple(ids.shape)}")

    # Capture layer-10 post-RoPE K via a forward hook on attention module
    captured: dict[str, torch.Tensor] = {}

    def hook(module, args, output):
        # For Qwen3 attention the forward returns (attn_output,). We
        # want post-RoPE K which lives in past_key_value or in
        # intermediate state. Instead wrap k_proj directly and then
        # manually apply RoPE to match mlx-lm's post-RoPE K.
        pass

    # Simpler: just capture k_proj output (pre-RoPE), apply RoPE via
    # the model's own rotary_emb, and compare that with silica.
    attn = model.model.layers[10].self_attn
    orig_k_proj_forward = attn.k_proj.forward

    def capture_k_proj(x, *a, **kw):
        out = orig_k_proj_forward(x, *a, **kw)
        captured["pre_rope_k"] = out.detach().clone()
        return out

    attn.k_proj.forward = capture_k_proj  # type: ignore[assignment]

    with torch.no_grad():
        _ = model(ids, use_cache=True)

    attn.k_proj.forward = orig_k_proj_forward  # type: ignore[assignment]

    pre_rope_k = captured["pre_rope_k"]  # (1, seq, n_kv_heads * head_dim)
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    print(f"pre_rope_k shape={tuple(pre_rope_k.shape)} dtype={pre_rope_k.dtype}")
    B, S, _ = pre_rope_k.shape
    # Reshape to (B, n_kv_heads, S, head_dim) matching silica's tensor layout
    k_4d = pre_rope_k.reshape(B, S, n_kv_heads, head_dim).permute(0, 2, 1, 3)
    print(f"4D shape: {tuple(k_4d.shape)}")

    # Take the first aligned block (block_size=16)
    block_size = 16
    vq_block_size = 64
    num_bits = 4
    one_block = k_4d[:, :, :block_size, :]  # (1, n_kv_heads, 16, head_dim)
    print(f"one_block shape: {tuple(one_block.shape)}")

    # Flatten to (n_vectors, head_dim) for vqbench API
    n_vectors = B * n_kv_heads * block_size
    block_np = one_block.float().cpu().numpy().reshape(-1, head_dim)
    print(f"block_np shape: {block_np.shape}")

    # vqbench BlockTQ at the same (head_dim, block_size, num_bits, seed=42)
    vq_codec = VqbenchBlockTQ(
        d=head_dim,
        num_bits=num_bits,
        block_size=vq_block_size,
        seed=42,
        norm_correction=True,
    )
    qvs = vq_codec.quantize_batch(block_np)
    decoded = vq_codec.dequantize_batch(qvs)
    frob = frobenius(block_np, decoded)
    print(f"\n[vqbench NumPy BlockTurboQuantMSE, B={vq_block_size}, b={num_bits}, seed=42]")
    print(f"  Frobenius on real Qwen3-0.6B pre-RoPE K block: {frob:.6f}")

    # Also compare per head
    for h in range(n_kv_heads):
        head_slice = one_block[0, h].float().cpu().numpy()  # (block_size, head_dim)
        qvs_h = vq_codec.quantize_batch(head_slice)
        dec_h = vq_codec.dequantize_batch(qvs_h)
        fh = frobenius(head_slice, dec_h)
        print(f"  head {h}: {fh:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
