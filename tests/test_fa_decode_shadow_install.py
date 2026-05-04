"""Shadow-install smoke test: SILICA_USE_FA_DECODE_V10=1 produces the same
output as the un-fused mx.fast.scaled_dot_product_attention + sigmoid+multiply
path, on a synthetic Qwen3NextAttention forward.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import pytest


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


@pytest.fixture
def fresh_install():
    """Restore shadow install state before/after each test."""
    from silica.kernels import shadow_install
    shadow_install.restore()
    saved_env = {
        k: os.environ.get(k)
        for k in [
            "SILICA_USE_FA_DECODE_V10",
            "SILICA_USE_FUSED_GATED_OUTPUT",
            "SILICA_USE_FUSED_SILU_MUL",
        ]
    }
    yield
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    shadow_install.restore()


def test_fa_decode_v10_shadow_matches_baseline(fresh_install) -> None:
    pytest.importorskip("mlx_lm.models.qwen3_next")
    from mlx_lm.models import qwen3_next as qn
    from silica.kernels import shadow_install

    # Construct a minimal Qwen3.5 attention with production shape:
    # H_q=24, H_kv=4, D=256, num_q heads with gate split → q_proj output
    # is (2*H_q*D) wide.
    cfg = qn.ModelArgs(
        model_type="qwen3_next",
        hidden_size=24 * 256,                  # H_q * D
        num_hidden_layers=1,
        intermediate_size=512,
        num_attention_heads=24,
        linear_num_value_heads=0,
        linear_num_key_heads=0,
        linear_key_head_dim=0,
        linear_value_head_dim=0,
        linear_conv_kernel_dim=0,
        num_experts=1,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=512,
        mlp_only_layers=[],
        moe_intermediate_size=512,
        rms_norm_eps=1e-6,
        vocab_size=4,
        num_key_value_heads=4,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        max_position_embeddings=4096,
        head_dim=256,
        full_attention_interval=1,
    )

    mx.random.seed(0)
    attn_baseline = qn.Qwen3NextAttention(cfg)
    # Cast layer params to fp16 to match decode dtype.
    def _to_fp16(model):
        for name, param in model.named_parameters():
            if hasattr(param, "dtype") and param.dtype == mx.float32:
                model.update({name: param.astype(mx.float16)})
    # mlx-lm modules are nn.Module; their submodules also have parameters.
    # Convert all model weights to float16 in-place.
    attn_baseline.set_dtype(mx.float16)
    mx.eval(attn_baseline.parameters())

    from mlx_lm.models.cache import KVCache
    cache_baseline = KVCache()

    # Prefill 8 tokens, then run T_q=1 decode step (this is where v10 fires).
    B = 1
    prompt = (mx.random.normal((B, 8, cfg.hidden_size)) * 0.1).astype(mx.float16)
    x = (mx.random.normal((B, 1, cfg.hidden_size)) * 0.1).astype(mx.float16)
    _ = attn_baseline(prompt, cache=cache_baseline)
    out_baseline = attn_baseline(x, cache=cache_baseline)
    mx.eval(out_baseline)

    # Install v10 shadow.
    os.environ["SILICA_USE_FA_DECODE_V10"] = "1"
    installed = shadow_install.install(model=None)
    assert installed["fa_decode_v10"] is True, f"v10 did not install: {installed}"

    cache_v10 = KVCache()
    _ = attn_baseline(prompt, cache=cache_v10)
    out_v10 = attn_baseline(x, cache=cache_v10)
    mx.eval(out_v10)

    err = _maxabs(out_v10, out_baseline)
    # Tolerance: FA-decode v10 introduces minor numerical drift vs the
    # un-fused sigmoid+multiply path because the gate is applied inside the
    # FA epilogue (one extra fp16 round-trip vs the un-fused
    # output * sigmoid(gate). Both paths are fp16-ULP-correct, but they are
    # not bit-identical. Allow a generous bound.
    assert err < 1e-2, f"v10 shadow path output drift {err} too large vs baseline"
