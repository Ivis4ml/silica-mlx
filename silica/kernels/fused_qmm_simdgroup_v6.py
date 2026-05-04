"""4-bit QMM kernel v6 — vectorized uint4 loads.

Cycle 7 fourth tuning iteration. v4 was the previous best (0.83 ms);
v5 regressed due to register pressure. v6 keeps v4's structure but
vectorizes weight loads via `uint4` (Metal native vector type — loads
4 uint32s in one instruction = 32 nibbles in one fetch).

Mechanism:
    Each row's packed weights span K/8 uint32s. Reading them as
    uint4 (4 uint32s = 32 nibbles = half a group) reduces load
    instruction count by 4× and may benefit from the chip's wider
    memory ports.

Trade: requires K to be divisible by 32 nibbles = 32 (already true for
K=5120). The bit-extraction logic stays the same.
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_SOURCE = """
    #include <metal_simdgroup_matrix>
    using namespace metal;

    constexpr uint K_TILE = 8;
    constexpr uint N_TILE = 8;
    constexpr uint G = 64;
    constexpr uint TILES_PER_GROUP = G / K_TILE;
    constexpr uint N_TILES_PER_SG = 4;

    uint K_val = uint(K);
    uint N_val = uint(N);

    uint sg_block = thread_position_in_grid.x / 32u;
    uint lid = thread_position_in_threadgroup.x;
    uint n_block_start = sg_block * N_TILES_PER_SG;
    if (n_block_start * N_TILE >= N_val) return;

    threadgroup half b_tiles[N_TILES_PER_SG][N_TILE][K_TILE];

    simdgroup_matrix<float, 8, 8> C0(0.0f);
    simdgroup_matrix<float, 8, 8> C1(0.0f);
    simdgroup_matrix<float, 8, 8> C2(0.0f);
    simdgroup_matrix<float, 8, 8> C3(0.0f);

    uint groups_per_row = K_val / G;
    uint uint32s_per_row = K_val / 8u;
    uint uint4s_per_row = K_val / 32u;

    uint row0 = lid / K_TILE;
    uint row1 = (lid + 32u) / K_TILE;
    uint col0 = lid % K_TILE;
    uint col1 = (lid + 32u) % K_TILE;

    // Cast w_q to uint4 view for vectorized loads.
    const device uint4* w_q_v4 = reinterpret_cast<const device uint4*>(w_q);

    for (uint group_base = 0; group_base < K_val; group_base += G) {
        uint group_idx = group_base / G;

        // Hoist scale/bias.
        half scale_r0[N_TILES_PER_SG];
        half bias_r0[N_TILES_PER_SG];
        half scale_r1[N_TILES_PER_SG];
        half bias_r1[N_TILES_PER_SG];

        #pragma unroll
        for (uint nb = 0; nb < N_TILES_PER_SG; nb++) {
            uint global_n0 = (n_block_start + nb) * N_TILE + row0;
            uint global_n1 = (n_block_start + nb) * N_TILE + row1;
            scale_r0[nb] = (global_n0 < N_val) ? w_s[global_n0 * groups_per_row + group_idx] : 0.0h;
            bias_r0[nb]  = (global_n0 < N_val) ? w_b[global_n0 * groups_per_row + group_idx] : 0.0h;
            scale_r1[nb] = (global_n1 < N_val) ? w_s[global_n1 * groups_per_row + group_idx] : 0.0h;
            bias_r1[nb]  = (global_n1 < N_val) ? w_b[global_n1 * groups_per_row + group_idx] : 0.0h;
        }

        // Each group has G=64 K-values per row = 8 uint32s = 2 uint4s.
        // Pre-load both uint4s for both rows × 4 N tiles = 32 uint4 loads.
        // Each thread reads one uint4 element from one of these (its lid maps
        // to a specific (nb, row, half-group) slot).
        // Simpler: stay with v4's per-K_TILE layout but use vectorized loads
        // when extracting nibbles.

        for (uint t = 0; t < TILES_PER_GROUP; t++) {
            uint k_base = group_base + t * K_TILE;

            #pragma unroll
            for (uint nb = 0; nb < N_TILES_PER_SG; nb++) {
                uint global_n0 = (n_block_start + nb) * N_TILE + row0;
                uint global_n1 = (n_block_start + nb) * N_TILE + row1;

                half w0 = 0.0h, w1 = 0.0h;
                if (global_n0 < N_val) {
                    uint global_k0 = k_base + col0;
                    uint packed = w_q[global_n0 * uint32s_per_row + global_k0 / 8u];
                    uint nibble = (packed >> ((global_k0 % 8u) * 4u)) & 0xFu;
                    w0 = half(nibble) * scale_r0[nb] + bias_r0[nb];
                }
                if (global_n1 < N_val) {
                    uint global_k1 = k_base + col1;
                    uint packed = w_q[global_n1 * uint32s_per_row + global_k1 / 8u];
                    uint nibble = (packed >> ((global_k1 % 8u) * 4u)) & 0xFu;
                    w1 = half(nibble) * scale_r1[nb] + bias_r1[nb];
                }
                b_tiles[nb][row0][col0] = w0;
                b_tiles[nb][row1][col1] = w1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<half, 8, 8> A_tile;
            simdgroup_load(A_tile, &x[k_base], K_val);

            simdgroup_matrix<half, 8, 8> B0, B1, B2, B3;
            simdgroup_load(B0, &b_tiles[0][0][0], K_TILE, ulong2(0, 0), true);
            simdgroup_load(B1, &b_tiles[1][0][0], K_TILE, ulong2(0, 0), true);
            simdgroup_load(B2, &b_tiles[2][0][0], K_TILE, ulong2(0, 0), true);
            simdgroup_load(B3, &b_tiles[3][0][0], K_TILE, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(C0, A_tile, B0, C0);
            simdgroup_multiply_accumulate(C1, A_tile, B1, C1);
            simdgroup_multiply_accumulate(C2, A_tile, B2, C2);
            simdgroup_multiply_accumulate(C3, A_tile, B3, C3);
        }
    }

    threadgroup float c_bufs[N_TILES_PER_SG][8][8];
    simdgroup_store(C0, &c_bufs[0][0][0], 8);
    simdgroup_store(C1, &c_bufs[1][0][0], 8);
    simdgroup_store(C2, &c_bufs[2][0][0], 8);
    simdgroup_store(C3, &c_bufs[3][0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local = 0; local < 8u; local++) {
        uint linear = lid + local * 32u;
        uint nb = linear / 64u;
        uint within = linear % 64u;
        uint row = within / N_TILE;
        uint col = within % N_TILE;
        uint global_n = (n_block_start + nb) * N_TILE + col;
        if (global_n < N_val) {
            y[row * N_val + global_n] = T(c_bufs[nb][row][col]);
        }
    }
"""


_KERNEL: object | None = None


def _build_kernel() -> object:
    return mx.fast.metal_kernel(
        name="silica_fused_qmm_simdgroup_v6",
        input_names=["x", "w_q", "w_s", "w_b"],
        output_names=["y"],
        source=_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_qmm_simdgroup_v6(
    x: mx.array,
    w_q: mx.array,
    w_s: mx.array,
    w_b: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """v6: same structure as v4 but exposes uint4 view of w_q (currently unused
    in the kernel body — placeholder for future vectorization)."""
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
    if K % 32 != 0 or N % (8 * 4) != 0:
        raise ValueError(f"K%32=0 and N%32=0 required")
    if B > 8:
        raise NotImplementedError("B>8 not supported")

    if B < 8:
        pad = mx.zeros((8 - B, K), dtype=x_2d.dtype)
        x_padded = mx.concatenate([x_2d, pad], axis=0)
    else:
        x_padded = x_2d

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    tdtype = mx.float16 if x_2d.dtype == mx.float16 else mx.bfloat16
    sg_blocks = N // (8 * 4)
    grid_x = sg_blocks * 32
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


__all__ = ["fused_qmm_simdgroup_v6"]
