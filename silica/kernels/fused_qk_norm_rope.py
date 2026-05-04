"""Fused Q-norm + K-norm kernel for Qwen3.5 Gated Attention.

Mechanism
---------
Qwen3.5 full-attention layers (qwen3_next.py:137-138) apply RMSNorm to
queries and keys separately:
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(...)).transpose(0, 2, 1, 3)

Each ``q_norm`` / ``k_norm`` call is one ``mx.fast.rms_norm`` Metal
launch. For 16 attention layers per step, that's 32 RMSNorm launches
plus the 2 transposes that follow. Each launch carries ~10-20 μs of
overhead on M5 Pro.

This kernel fuses Q-norm and K-norm into one Metal dispatch when the two
share the same ``head_dim`` and ``epsilon``. It does not yet fuse the
RoPE step that follows (that requires offset-aware kernel support),
but eliminates one launch per layer.

Production shapes (Qwen3.5-27B at B=4, decode, full-attn layers):
    q: (B=4, T=1, num_heads=24, head_dim=256), 24 KB at fp16
    k: (B=4, T=1, num_kv_heads=4, head_dim=256), 4 KB at fp16
    weight: (head_dim=256,) per norm, 0.5 KB at fp16

Numerical equivalence:
    rms_norm(x, w, eps) = x * w / sqrt(mean(x^2) + eps)
    Computed in fp32 register precision; cast back to dtype on store.
    Expected fp16 max-abs error vs ``mx.fast.rms_norm``: < 1e-3.

Implementation notes:
    The kernel processes one row at a time (one [head_dim] vector) per
    threadgroup. Two tensors are passed; each threadgroup chooses whether
    to read q or k based on its grid position. Output buffer layout
    matches the inputs.
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_SOURCE = """
    // Each threadgroup processes one head_dim-sized vector from either
    // q_in (first n_q_rows rows) or k_in (next n_k_rows rows).
    // grid.x = head_dim, grid.y = total_rows = n_q_rows + n_k_rows
    uint head_dim = uint(D);
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    uint n_q_rows = n_q[0];
    uint n_k_rows = n_k[0];
    float eps_val = eps_in[0];

    if (col >= head_dim || row >= (n_q_rows + n_k_rows)) return;

    // Threadgroup-shared scratch for the reduction.
    threadgroup float sum_sq[256];

    bool is_q = (row < n_q_rows);
    uint local_row = is_q ? row : (row - n_q_rows);
    uint base = local_row * head_dim;

    T x_val;
    if (is_q) {
        x_val = q_in[base + col];
    } else {
        x_val = k_in[base + col];
    }

    float xf = float(x_val);
    sum_sq[col] = xf * xf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction (head_dim is power of 2 typically; 256 in Qwen3.5)
    for (uint stride = head_dim / 2; stride > 0; stride /= 2) {
        if (col < stride && (col + stride) < head_dim) {
            sum_sq[col] += sum_sq[col + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean_sq = sum_sq[0] / float(head_dim);
    float inv_rms = metal::rsqrt(mean_sq + eps_val);

    float weight_val;
    if (is_q) {
        weight_val = float(q_weight[col]);
    } else {
        weight_val = float(k_weight[col]);
    }

    float out_val = xf * inv_rms * weight_val;
    if (is_q) {
        q_out[base + col] = T(out_val);
    } else {
        k_out[base + col] = T(out_val);
    }
"""


_KERNEL: object | None = None


def _build_kernel() -> object:
    return mx.fast.metal_kernel(
        name="silica_fused_qk_norm",
        input_names=["q_in", "k_in", "q_weight", "k_weight", "n_q", "n_k", "eps_in"],
        output_names=["q_out", "k_out"],
        source=_KERNEL_SOURCE,
    )


def fused_qk_norm(
    q: mx.array, k: mx.array, q_weight: mx.array, k_weight: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """Compute RMSNorm(q, q_weight, eps) and RMSNorm(k, k_weight, eps) in one dispatch.

    Args:
        q: queries before norm, shape (..., head_dim).
        k: keys before norm, shape (..., head_dim).
        q_weight, k_weight: per-channel scale of shape (head_dim,).
        eps: RMSNorm epsilon.

    Returns:
        (q_normed, k_normed) of the same shapes/dtypes as inputs.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"fused_qk_norm: head_dim mismatch q={q.shape} vs k={k.shape}"
        )
    if q.dtype != k.dtype:
        raise ValueError(
            f"fused_qk_norm: dtype mismatch q={q.dtype} vs k={k.dtype}"
        )
    head_dim = q.shape[-1]
    n_q = q.size // head_dim
    n_k = k.size // head_dim

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    if q.dtype == mx.float32:
        tdtype = mx.float32
    elif q.dtype == mx.float16:
        tdtype = mx.float16
    else:
        tdtype = mx.bfloat16

    # Reshape to 2D for the kernel; reshape back at the end.
    q_2d = q.reshape(n_q, head_dim)
    k_2d = k.reshape(n_k, head_dim)

    outs = _KERNEL(  # type: ignore[operator]
        inputs=[
            q_2d, k_2d, q_weight, k_weight,
            mx.array([n_q], dtype=mx.uint32),
            mx.array([n_k], dtype=mx.uint32),
            mx.array([eps], dtype=mx.float32),
        ],
        template=[("T", tdtype), ("D", head_dim)],
        grid=(head_dim, n_q + n_k, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(n_q, head_dim), (n_k, head_dim)],
        output_dtypes=[q.dtype, k.dtype],
    )
    q_normed = outs[0].reshape(q.shape)
    k_normed = outs[1].reshape(k.shape)
    return q_normed, k_normed


def reference_qk_norm(
    q: mx.array, k: mx.array, q_weight: mx.array, k_weight: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """MLX reference: two separate ``mx.fast.rms_norm`` calls."""
    q_normed = mx.fast.rms_norm(q, q_weight, eps)
    k_normed = mx.fast.rms_norm(k, k_weight, eps)
    return q_normed, k_normed


__all__ = ["fused_qk_norm", "reference_qk_norm"]
