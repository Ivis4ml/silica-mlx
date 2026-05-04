"""Correctness tests for silica.kernels.flash_attention_decode v6/v7/v8.

Cycle 11 (2026-05-04) FA-decode port. Verifies the FlashDecoding kernels
match mx.fast.scaled_dot_product_attention within fp16 ULP across the
production decode shape sweep, with and without the Qwen3.5 sigmoid-gate
fusion. v7 adds vectorized half4 K/V loads; v8 also vectorizes the inner
score and V-multiply.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.kernels.flash_attention_decode_v6 import flash_attention_decode_v6
from silica.kernels.flash_attention_decode_v7 import flash_attention_decode_v7
from silica.kernels.flash_attention_decode_v8 import flash_attention_decode_v8
from silica.kernels.flash_attention_decode_v10 import flash_attention_decode_v10


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _ref(q, k, v, *, gate=None, scale=None):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    if gate is not None:
        out = out * mx.sigmoid(gate)
    return out


_SHAPES = [
    (1, 24, 4, 32, 256),
    (4, 24, 4, 128, 256),
    (16, 24, 4, 256, 256),
    (48, 24, 4, 128, 256),
    (48, 24, 4, 512, 256),
]


@pytest.mark.parametrize(("B", "H_q", "H_kv", "T_kv", "D"), _SHAPES)
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "kernel_fn",
    [
        flash_attention_decode_v6,
        flash_attention_decode_v7,
        flash_attention_decode_v8,
        flash_attention_decode_v10,
    ],
    ids=["v6", "v7", "v8", "v10"],
)
def test_flash_attention_decode_matches_mlx_sdpa(
    B: int, H_q: int, H_kv: int, T_kv: int, D: int, gated: bool, kernel_fn
) -> None:
    mx.random.seed(0)
    q = (mx.random.normal((B, H_q, 1, D)) * 0.1).astype(mx.float16)
    k = (mx.random.normal((B, H_kv, T_kv, D)) * 0.1).astype(mx.float16)
    v = (mx.random.normal((B, H_kv, T_kv, D)) * 0.1).astype(mx.float16)
    gate = (mx.random.normal((B, H_q, 1, D)) * 0.5).astype(mx.float16) if gated else None
    mx.eval(q, k, v)
    if gate is not None:
        mx.eval(gate)

    out_ref = _ref(q, k, v, gate=gate)
    out = kernel_fn(q, k, v, gate=gate)
    mx.eval(out_ref, out)
    err = _maxabs(out, out_ref)
    assert err < 5e-4, f"{kernel_fn.__name__} max-abs {err} > tolerance, gated={gated}"


def test_flash_attention_decode_v6_rejects_T_q_gt_1() -> None:
    q = mx.zeros((1, 24, 2, 256), dtype=mx.float16)
    k = mx.zeros((1, 4, 32, 256), dtype=mx.float16)
    v = mx.zeros((1, 4, 32, 256), dtype=mx.float16)
    with pytest.raises(NotImplementedError, match="T_q=1"):
        flash_attention_decode_v6(q, k, v)


def test_flash_attention_decode_v6_rejects_wrong_head_dim() -> None:
    q = mx.zeros((1, 24, 1, 128), dtype=mx.float16)
    k = mx.zeros((1, 4, 32, 128), dtype=mx.float16)
    v = mx.zeros((1, 4, 32, 128), dtype=mx.float16)
    with pytest.raises(NotImplementedError, match="head_dim"):
        flash_attention_decode_v6(q, k, v)


def test_flash_attention_decode_v6_rejects_wrong_gqa_ratio() -> None:
    q = mx.zeros((1, 16, 1, 256), dtype=mx.float16)
    k = mx.zeros((1, 4, 32, 256), dtype=mx.float16)
    v = mx.zeros((1, 4, 32, 256), dtype=mx.float16)
    with pytest.raises(NotImplementedError, match="Q_PER_KV"):
        flash_attention_decode_v6(q, k, v)
