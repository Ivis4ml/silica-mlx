"""4-bit QMM kernel v3 — simdgroup_matrix MMA + hoisted scale/bias loads.

Cycle 7 first tuning iteration on cycle-6 simdgroup_matrix kernel.

Optimization: scale/bias is per-group (group_size=64), but cycle-6 v2 read it
on every K_TILE iteration (8 K values per tile, 8 tiles per group = 8x
redundant reads). v3 hoists the load: each group spans 8 K_TILEs; load
scale/bias once per group, reuse across 8 K_TILE iterations.

Expected savings: 7/8 of scale/bias reads. For B=4 K=5120 N=17408:
- Total scale reads per simdgroup before: 32 threads × 2 reads × 640 K_TILEs = 40,960
- Total scale reads per simdgroup after:  32 threads × 2 reads × 80 groups   = 5,120
- Reduction: 8x for scale/bias loads
- Each load is a fp16 = 2 bytes; saved bytes per simdgroup ≈ 70 KB
- Across 2176 simdgroups: ~150 MB of scale/bias reads avoided
- At 307 GB/s, that's ~0.5 ms of bandwidth — substantial fraction of 0.98 ms!
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_SOURCE = """
    #include <metal_simdgroup_matrix>
    using namespace metal;

    constexpr uint K_TILE = 8;
    constexpr uint N_TILE = 8;
    constexpr uint G = 64;
    constexpr uint TILES_PER_GROUP = G / K_TILE;  // = 8

    uint K_val = uint(K);
    uint N_val = uint(N);

    uint n_tile = thread_position_in_grid.x / 32u;
    uint lid = thread_position_in_threadgroup.x;
    if (n_tile * N_TILE >= N_val) return;

    threadgroup half b_tile[N_TILE][K_TILE];

    simdgroup_matrix<float, 8, 8> C(0.0f);

    uint groups_per_row = K_val / G;
    uint uint32s_per_row = K_val / 8u;

    // Per-thread cache: scale/bias for each thread's "row" within the N tile.
    // Each thread handles 2 (row, col) pairs in the dequant loop. The 2 rows
    // are lid/8 and (lid+32)/8 = lid/8 + 4. Cache scale/bias for these.
    uint row0 = lid / K_TILE;       // 0..3 (since lid 0..31 → row 0..3)
    uint row1 = (lid + 32u) / K_TILE; // 4..7
    uint global_n0 = n_tile * N_TILE + row0;
    uint global_n1 = n_tile * N_TILE + row1;

    // Iterate over K in GROUPS (G=64 K-values = 8 K_TILEs).
    for (uint group_base = 0; group_base < K_val; group_base += G) {
        uint group_idx = group_base / G;

        // Load scale/bias ONCE per group per row (instead of per K_TILE).
        half scale0 = (global_n0 < N_val) ? w_s[global_n0 * groups_per_row + group_idx] : 0.0h;
        half bias0  = (global_n0 < N_val) ? w_b[global_n0 * groups_per_row + group_idx] : 0.0h;
        half scale1 = (global_n1 < N_val) ? w_s[global_n1 * groups_per_row + group_idx] : 0.0h;
        half bias1  = (global_n1 < N_val) ? w_b[global_n1 * groups_per_row + group_idx] : 0.0h;

        // Within the group, iterate over the 8 K_TILEs.
        for (uint t = 0; t < TILES_PER_GROUP; t++) {
            uint k_base = group_base + t * K_TILE;

            // Dequantize 8x8 = 64 weights. 32 threads, 2 each.
            uint col0 = lid % K_TILE;
            uint col1 = (lid + 32u) % K_TILE;
            uint global_k0 = k_base + col0;
            uint global_k1 = k_base + col1;

            half w_val_0 = 0.0h;
            half w_val_1 = 0.0h;
            if (global_n0 < N_val) {
                uint uint32_idx = global_k0 / 8u;
                uint nibble_pos = global_k0 % 8u;
                uint packed = w_q[global_n0 * uint32s_per_row + uint32_idx];
                uint nibble = (packed >> (nibble_pos * 4u)) & 0xFu;
                w_val_0 = half(nibble) * scale0 + bias0;
            }
            if (global_n1 < N_val) {
                uint uint32_idx = global_k1 / 8u;
                uint nibble_pos = global_k1 % 8u;
                uint packed = w_q[global_n1 * uint32s_per_row + uint32_idx];
                uint nibble = (packed >> (nibble_pos * 4u)) & 0xFu;
                w_val_1 = half(nibble) * scale1 + bias1;
            }
            b_tile[row0][col0] = w_val_0;
            b_tile[row1][col1] = w_val_1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load A 8x8 directly from x (pre-padded to 8 rows by caller).
            simdgroup_matrix<half, 8, 8> A_tile;
            simdgroup_load(A_tile, &x[k_base], K_val);

            simdgroup_matrix<half, 8, 8> B_tile;
            simdgroup_load(B_tile, &b_tile[0][0], K_TILE, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(C, A_tile, B_tile, C);
        }
    }

    // Store C to threadgroup buffer, convert to half, write to y.
    threadgroup float c_buf[8][8];
    simdgroup_store(C, &c_buf[0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint local = 0; local < 2u; local++) {
        uint linear = lid + local * 32u;
        uint row = linear / N_TILE;
        uint col = linear % N_TILE;
        uint global_n = n_tile * N_TILE + col;
        if (global_n < N_val) {
            y[row * N_val + global_n] = T(c_buf[row][col]);
        }
    }
"""


_KERNEL: object | None = None


def _build_kernel() -> object:
    return mx.fast.metal_kernel(
        name="silica_fused_qmm_simdgroup_v3",
        input_names=["x", "w_q", "w_s", "w_b"],
        output_names=["y"],
        source=_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_qmm_simdgroup_v3(
    x: mx.array,
    w_q: mx.array,
    w_s: mx.array,
    w_b: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """v3 with hoisted scale/bias loads (per group, not per K_TILE)."""
    if bits != 4 or group_size != 64:
        raise NotImplementedError("only bits=4 group_size=64")
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise NotImplementedError("only T=1")
        x_2d = x.reshape(x.shape[0], x.shape[2])
    elif x.ndim == 2:
        x_2d = x
    else:
        raise ValueError(f"x must be 2D or 3D; got {x.shape}")

    B, K = x_2d.shape
    N = w_q.shape[0]
    if K % group_size != 0 or N % 8 != 0:
        raise ValueError(f"K%{group_size}=0 and N%8=0 required; K={K} N={N}")
    if B > 8:
        raise NotImplementedError(f"B>8 not supported; got B={B}")

    if B < 8:
        pad = mx.zeros((8 - B, K), dtype=x_2d.dtype)
        x_padded = mx.concatenate([x_2d, pad], axis=0)
    else:
        x_padded = x_2d

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    tdtype = mx.float16 if x_2d.dtype == mx.float16 else mx.bfloat16
    n_tiles = N // 8
    grid_x = n_tiles * 32
    out = _KERNEL(  # type: ignore[operator]
        inputs=[x_padded, w_q, w_s, w_b],
        template=[("T", tdtype), ("K", K), ("N", N)],
        grid=(grid_x, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(8, N)],
        output_dtypes=[x_2d.dtype],
    )
    y_padded = out[0]
    y = y_padded[:B, :]
    if x.ndim == 3:
        y = y.reshape(x.shape[0], 1, N)
    return y


__all__ = ["fused_qmm_simdgroup_v3"]
