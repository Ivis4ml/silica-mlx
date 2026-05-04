"""Fused SwiGLU activation kernel — `silu(gate) * up` in one Metal dispatch.

Replaces mlx-lm's ``_precise_swiglu`` (qwen3_next.py:58-62), which is itself
``@mx.compile``-fused into one kernel but goes through fp16 → fp32 → fp16
casts that materialise a temporary fp32 tensor. This kernel does silu in
fp32 inside registers then writes fp16, avoiding the fp32 HBM round-trip.

Mechanism:
    Stock:  gate_f32 = silu(gate.astype(f32))
            x_f32 = up.astype(f32)
            out = (gate_f32 * x_f32).astype(out_dtype)
            # mx.compile fuses these into one kernel that internally
            # materialises gate_f32 and x_f32 (twice the working set).

    Fused:  per-element kernel that reads gate (fp16), up (fp16), computes
            silu(gate) * up in fp32 register precision, writes fp16.
            No fp32 HBM materialisation.

For Qwen3.5-27B-4bit at B=4, T=1, intermediate_size=17408, fp16:
    Working set saved = 2 × (4 × 1 × 17408 × 2 bytes) = 280 KB per MLP layer
    Across 64 layers per step = 17.5 MB
    At 307 GB/s = ~57 μs / step potential save (modulo mx.compile already
    fusing some of this).

Numerical equivalence:
    silu(g) = g / (1 + exp(-g))
    output = silu(g) * x
    Computed in fp32 register precision; cast to dtype on store.
    Expected fp16 max-abs error vs ``_precise_swiglu``: < 1e-3 (both use
    fp32 silu internally; only the cast point differs).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_KERNEL_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= numel) return;

    // Load fp16/bf16 inputs, promote to fp32 for silu numerical stability.
    float g_val = float(gate[elem]);
    float u_val = float(up[elem]);

    // silu(g) = g / (1 + exp(-g))
    // Saturate exp argument to avoid fp32 overflow (silu(g) -> g for large g, 0 for small g).
    float silu_g;
    if (g_val > 16.0f) {
        silu_g = g_val;
    } else if (g_val < -16.0f) {
        silu_g = 0.0f;
    } else {
        silu_g = g_val / (1.0f + metal::exp(-g_val));
    }

    // Multiply in fp32 register precision; cast to output dtype on store.
    float result_f32 = silu_g * u_val;
    out[elem] = T(result_f32);
"""


_KERNEL: object | None = None


def _build_kernel() -> object:
    return mx.fast.metal_kernel(
        name="silica_fused_silu_mul",
        input_names=["gate", "up", "numel"],
        output_names=["out"],
        source=_KERNEL_SOURCE,
    )


def fused_silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    """Compute ``silu(gate) * up`` in a single fused Metal dispatch.

    Args:
        gate: SwiGLU gate output, any shape; fp16/bf16/fp32.
        up: SwiGLU up output, same shape and dtype as gate.

    Returns:
        ``silu(gate) * up`` of the same shape and dtype as gate.
    """
    if gate.shape != up.shape:
        raise ValueError(
            f"fused_silu_mul: shape mismatch gate={gate.shape} vs up={up.shape}"
        )
    if gate.dtype != up.dtype:
        raise ValueError(
            f"fused_silu_mul: dtype mismatch gate={gate.dtype} vs up={up.dtype}"
        )
    if gate.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError(
            f"fused_silu_mul: unsupported dtype {gate.dtype}"
        )

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    numel = gate.size
    tg = 256
    grid = ((numel + tg - 1) // tg) * tg

    if gate.dtype == mx.float32:
        tdtype = mx.float32
    elif gate.dtype == mx.float16:
        tdtype = mx.float16
    else:
        tdtype = mx.bfloat16

    out = _KERNEL(  # type: ignore[operator]
        inputs=[gate, up, mx.array(numel, dtype=mx.uint32)],
        template=[("T", tdtype)],
        grid=(grid, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )
    return out[0]


def reference_silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    """MLX reference: ``silu(gate) * up`` via nn.silu + multiply (no fp32 promotion)."""
    return nn.silu(gate) * up


def reference_silu_mul_precise(gate: mx.array, up: mx.array) -> mx.array:
    """MLX reference matching mlx-lm `_precise_swiglu` semantics (fp32 internally)."""
    g_f32 = nn.silu(gate.astype(mx.float32))
    x_f32 = up.astype(mx.float32)
    return (g_f32 * x_f32).astype(gate.dtype)


__all__ = ["fused_silu_mul", "reference_silu_mul", "reference_silu_mul_precise"]
