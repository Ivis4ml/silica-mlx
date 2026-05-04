"""Custom 4-bit affine quantised matmul kernel for B=4 decode shape.

Per the 2026-Q2 survey + cycle-2 microbenches (`plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_2.md`),
the load-bearing bottleneck on Qwen3.5-27B-4bit dense decode is per-layer
quantised matmul weight bandwidth. mlx core ships ``mx.quantized_matmul``
as a hand-tuned Metal kernel; this module attempts a simpler MX kernel
to (a) document the technique inside Silica, (b) provide a baseline that
later SIMD-group MMA tuning can beat, (c) measure the gap to mlx's
internal kernel.

Layout (matches ``mx.quantize(w, group_size=64, bits=4, mode='affine')``):
    - w_q: (N, K/8) uint32, each uint32 packs 8 nibbles (4-bit each), LSB first.
    - w_s: (N, K/group_size) fp16 scales.
    - w_b: (N, K/group_size) fp16 biases.
    - Dequantised: ``w[n, k] = nibble[n, k] * w_s[n, k/group_size] + w_b[n, k/group_size]``.

Operation:
    ``y[b, n] = sum_k(x[b, k] * w[n, k])`` for ``b ∈ [0, B), n ∈ [0, N)``.
    This is matmul with `transpose=True` semantics — the same that mlx-lm
    Linear modules use.

Production shapes (Qwen3.5-27B at B=4, decode, T=1):
    gate_proj / up_proj: (B=4, K=5120, N=17408) — 0.18 GB weight read per call
    down_proj:           (B=4, K=17408, N=5120) — 0.61 GB weight read per call
    q_proj (doubled):    (B=4, K=5120, N=12288) — 0.13 GB weight read per call
    k/v_proj:            (B=4, K=5120, N=1024)  — 0.01 GB weight read per call
    o_proj:              (B=4, K=6144, N=5120)  — 0.06 GB weight read per call

Per-layer total ~1 GB of weight reads × 64 layers = 64 GB; at 307 GB/s this
is the 200 ms physical floor. Real wall is 95 ms / step → 47% of bw used.
A faster custom QMM kernel narrows this gap.
"""

from __future__ import annotations

import mlx.core as mx

# Variant 1 — naive 1-thread-per-output kernel.
# Each thread computes ONE output element (b, n) by walking the full K
# dimension. No SIMD-group cooperation, no simdgroup_matrix MMA. Slower
# than mlx's internal mx.quantized_matmul (which uses simdgroup_matrix
# primitives) by ~1.4× at production shape, but correct and simple. Two
# alternative variants (SIMD-cooperative with simd_sum reduction; tiled
# threadgroup with simdgroup_matrix MMA) were tested in 2026-05-03
# cycle-3 — the SIMD-cooperative variant was 4× slower due to
# threadgroup-launch overhead from 32× more threadgroups; the
# simdgroup_matrix variant requires multi-day engineering to match
# mlx core. Documented in plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_3.md.
_KERNEL_SOURCE = """
    uint n = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    uint K_val = uint(K);
    uint N_val = uint(N);
    uint G_val = uint(GROUP);
    if (n >= N_val || b >= uint(B)) return;

    uint groups_per_row = K_val / G_val;
    uint uint32s_per_row = K_val / 8;
    uint uint32s_per_group = G_val / 8;

    float acc = 0.0f;

    for (uint g = 0; g < groups_per_row; g++) {
        float scale = float(w_s[n * groups_per_row + g]);
        float bias = float(w_b[n * groups_per_row + g]);

        for (uint u = 0; u < uint32s_per_group; u++) {
            uint k_base = g * G_val + u * 8;
            uint packed = w_q[n * uint32s_per_row + g * uint32s_per_group + u];

            for (uint i = 0; i < 8; i++) {
                uint nibble = (packed >> (i * 4)) & 0xFu;
                float w_val = float(nibble) * scale + bias;
                float x_val = float(x[b * K_val + k_base + i]);
                acc += x_val * w_val;
            }
        }
    }

    y[b * N_val + n] = T(acc);
"""


_KERNEL: object | None = None


def _build_kernel() -> object:
    return mx.fast.metal_kernel(
        name="silica_fused_qmm_decode",
        input_names=["x", "w_q", "w_s", "w_b"],
        output_names=["y"],
        source=_KERNEL_SOURCE,
    )


def fused_qmm_decode(
    x: mx.array,
    w_q: mx.array,
    w_s: mx.array,
    w_b: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Compute ``y = x @ dequant(w_q, w_s, w_b).T`` via a custom Metal kernel.

    Args:
        x: input (B, K) fp16.
        w_q: packed weights (N, K/8) uint32, 4-bit affine layout.
        w_s: scales (N, K/group_size) fp16.
        w_b: biases (N, K/group_size) fp16.
        group_size: must be 64.
        bits: must be 4.

    Returns:
        y: output (B, N) fp16.
    """
    if bits != 4 or group_size != 64:
        raise NotImplementedError(
            f"fused_qmm_decode currently only supports bits=4, group_size=64; "
            f"got bits={bits}, group_size={group_size}"
        )
    if x.ndim == 3:
        # (B, T, K) with T=1: squeeze T to (B, K) for the kernel.
        if x.shape[1] != 1:
            raise NotImplementedError(
                f"fused_qmm_decode currently supports T=1 decode only; got T={x.shape[1]}"
            )
        x_2d = x.reshape(x.shape[0], x.shape[2])
    elif x.ndim == 2:
        x_2d = x
    else:
        raise ValueError(f"x must be 2D or 3D; got {x.shape}")

    B, K = x_2d.shape
    N = w_q.shape[0]

    if w_q.shape[1] != K // 8:
        raise ValueError(f"w_q shape mismatch: expected (N, K/8) = ({N}, {K // 8}); got {w_q.shape}")

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    if x_2d.dtype == mx.float16:
        tdtype = mx.float16
    elif x_2d.dtype == mx.bfloat16:
        tdtype = mx.bfloat16
    elif x_2d.dtype == mx.float32:
        tdtype = mx.float32
    else:
        raise ValueError(f"unsupported dtype {x_2d.dtype}")

    # Naive 1-thread-per-output kernel. Threadgroup=32, grid=(N, B).
    tg = 32
    grid_x_aligned = ((N + tg - 1) // tg) * tg

    out = _KERNEL(  # type: ignore[operator]
        inputs=[x_2d, w_q, w_s, w_b],
        template=[("T", tdtype), ("B", B), ("K", K), ("N", N), ("GROUP", group_size)],
        grid=(grid_x_aligned, B, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, N)],
        output_dtypes=[x_2d.dtype],
    )
    y = out[0]
    if x.ndim == 3:
        y = y.reshape(x.shape[0], 1, N)
    return y


def reference_qmm_decode(
    x: mx.array,
    w_q: mx.array,
    w_s: mx.array,
    w_b: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """MLX reference using ``mx.quantized_matmul`` (the kernel under attack)."""
    return mx.quantized_matmul(
        x, w_q, w_s, w_b,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


__all__ = ["fused_qmm_decode", "reference_qmm_decode"]
