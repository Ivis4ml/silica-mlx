"""Fused output-gate Metal kernel for Qwen3.5 Gated Attention.

Mechanism
---------
Qwen3.5's full-attention layers use Gated Attention (`attn_output_gate=true`
in the upstream config): the attention output is multiplied by a learned
sigmoid gate before the o-projection. mlx-lm's reference implementation
in ``qwen3_next.py:263-274`` does this as TWO separate MLX ops on the
SDPA output tensor:

    output = mx.fast.scaled_dot_product_attention(q, k, v, scale=...)
    output = output * mx.sigmoid(gate)

The 2026-Q2 kernel ecosystem survey (2026-05-03) confirmed: no public
Apple-Silicon Metal codebase implements the gate inside the attention
epilogue. mlx-lm uses two unfused ops; mfa, dflash-mlx, ddtree-mlx,
vllm-metal all use vanilla SDPA without any gate. The post-SDPA gate is
therefore a measurement-anchored kernel-fusion gap.

This module fuses ``out * sigmoid(gate)`` into a single Metal kernel,
eliminating one kernel-launch overhead and one round-trip of the
``output`` tensor through HBM per attention layer per decode step.

For Qwen3.5-27B at B=4, T=1, H=24, head_dim=256, fp16: each output
tensor is 4 × 24 × 1 × 256 × 2 = 49.2 KB. Across 16 attention layers
per step that is 0.79 MB of HBM round-trip eliminated, plus 16 kernel
launches per step (≈10-20 μs/launch on M5 Pro).

Quantitative claim: this single-op fusion is small (≤1% step-time
reduction projected). It serves as the FOUNDATION for the larger fused
gated-SDPA kernel (FA-style Q@K^T → softmax → @V → output_gate → o_proj
in one dispatch), which is the load-bearing kernel for the (1b) ≥60
tok/s milestone. The simple fusion here demonstrates the harness
(correctness gate, perf microbench, shadow-mode integration) the larger
kernel will reuse.

Numerical equivalence
---------------------
The kernel computes ``y = x * sigmoid(g)`` element-wise. The MLX
reference is ``y_ref = x * mx.sigmoid(g)``. Both compute the same
mathematical function; the numerical difference is bounded by the
order of operations. Specifically the kernel uses
``sigmoid(g) = 1 / (1 + exp(-g))`` directly, while ``mx.sigmoid`` may
use a numerically tighter form. Expected fp16 max-abs error: < 1e-3
across normal-range inputs; max-rel error < 1e-3 for |x|, |g| in
[1e-4, 1e4].
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= numel) return;

    T x_val = x[elem];
    T g_val = g[elem];

    // sigmoid(g) = 1 / (1 + exp(-g))
    // For numerical stability under fp16, clamp g to the range where
    // sigmoid saturates: |g| > 16 → sigmoid(g) effectively 0 or 1.
    T abs_g = metal::abs(g_val);
    T s;
    if (abs_g > T(16.0)) {
        s = (g_val > T(0.0)) ? T(1.0) : T(0.0);
    } else {
        s = T(1.0) / (T(1.0) + metal::exp(-g_val));
    }
    out[elem] = x_val * s;
"""


def _build_kernel() -> object:
    """Build the metal_kernel; cached at module level."""
    return mx.fast.metal_kernel(
        name="silica_fused_gated_output",
        input_names=["x", "g", "numel"],
        output_names=["out"],
        source=_KERNEL_SOURCE,
    )


_KERNEL: object | None = None


def fused_gated_output(x: mx.array, g: mx.array) -> mx.array:
    """Compute ``x * sigmoid(g)`` in a single fused Metal dispatch.

    Args:
        x: attention output of any shape; typically (B, H, T, D) for
           Qwen3.5 full-attention layers. Must be row-contiguous (the
           default for mlx arrays returned by mx.fast operations).
        g: gate tensor of identical shape and dtype to ``x``.

    Returns:
        ``x * sigmoid(g)``, same shape and dtype as ``x``.

    Numerical reference: ``y_ref = x * mx.sigmoid(g)``. Expected
    fp16 max-abs error < 1e-3 in normal range.
    """
    if x.shape != g.shape:
        raise ValueError(
            f"fused_gated_output: shape mismatch x={x.shape} vs g={g.shape}"
        )
    if x.dtype != g.dtype:
        raise ValueError(
            f"fused_gated_output: dtype mismatch x={x.dtype} vs g={g.dtype}"
        )
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError(
            f"fused_gated_output: unsupported dtype {x.dtype}; "
            f"expected float16/bfloat16/float32"
        )

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    numel = x.size
    # Choose threadgroup: typical Apple Silicon GPUs prefer 256 threads
    # per group for memory-bound elementwise; round grid up to a multiple.
    tg = 256
    grid = ((numel + tg - 1) // tg) * tg
    # Pick the metal scalar type matching the input dtype.
    if x.dtype == mx.float32:
        tdtype = mx.float32
    elif x.dtype == mx.float16:
        tdtype = mx.float16
    else:
        tdtype = mx.bfloat16

    out = _KERNEL(  # type: ignore[operator]
        inputs=[x, g, mx.array(numel, dtype=mx.uint32)],
        template=[("T", tdtype)],
        grid=(grid, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out[0]


def reference_gated_output(x: mx.array, g: mx.array) -> mx.array:
    """The MLX-native two-op reference ``x * mx.sigmoid(g)``.

    Used by correctness microbenches and as the baseline for performance
    comparisons. Mirrors the exact pattern from mlx-lm
    ``qwen3_next.py:263-274``.
    """
    return x * mx.sigmoid(g)


__all__ = ["fused_gated_output", "reference_gated_output"]
