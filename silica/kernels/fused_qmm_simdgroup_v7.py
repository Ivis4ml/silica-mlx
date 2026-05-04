"""4-bit QMM kernel v7 — 2-simdgroup threadgroup, 8 N tiles per TG.

Cycle 8 first iteration. v4 (cycle 7) was best at 0.83 ms with 1
simdgroup × 4 N tiles per TG (544 TGs total). v7 doubles TG size to 2
simdgroups × 4 N tiles each = 8 N tiles per TG (272 TGs total).

Two simdgroups in one TG run concurrently on one GPU core, sharing
the same L1 scratch (threadgroup memory). Each maintains its own 4 C
accumulators independently. Each simdgroup processes its own 4 N
tiles; no inter-simdgroup synchronization needed (work is independent).

Trade-offs:
    + 2× fewer dispatches (272 vs 544): less launch overhead
    + Weight reads (scale/bias/packed) shared via L1 cache between
      simdgroups (improves cache hit rate)
    + Higher A-tile reuse: each x_tile load now serves 8 MMAs (was 4)
    - Higher per-TG register pressure (2× C accumulators)
    - Lower occupancy: a core can run fewer 64-thread TGs in flight
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
    constexpr uint SG_PER_TG = 2;
    constexpr uint THREADS_PER_TG = 32u * SG_PER_TG;
    constexpr uint N_TILES_PER_TG = N_TILES_PER_SG * SG_PER_TG;  // = 8

    uint K_val = uint(K);
    uint N_val = uint(N);

    uint tg_block = thread_position_in_grid.x / THREADS_PER_TG;
    uint thread_in_tg = thread_position_in_threadgroup.x;
    uint sg_id_in_tg = thread_in_tg / 32u;     // 0 or 1
    uint lid = thread_in_tg % 32u;             // 0..31 within simdgroup

    uint n_tg_start = tg_block * N_TILES_PER_TG;
    uint n_sg_start = n_tg_start + sg_id_in_tg * N_TILES_PER_SG;
    if (n_sg_start * N_TILE >= N_val) return;

    // Each simdgroup has its own b_tiles slot (4 N tiles × 8 rows × 8 cols).
    threadgroup half b_tiles[SG_PER_TG][N_TILES_PER_SG][N_TILE][K_TILE];

    simdgroup_matrix<float, 8, 8> C0(0.0f);
    simdgroup_matrix<float, 8, 8> C1(0.0f);
    simdgroup_matrix<float, 8, 8> C2(0.0f);
    simdgroup_matrix<float, 8, 8> C3(0.0f);

    uint groups_per_row = K_val / G;
    uint uint32s_per_row = K_val / 8u;

    uint row0 = lid / K_TILE;
    uint row1 = (lid + 32u) / K_TILE;
    uint col0 = lid % K_TILE;
    uint col1 = (lid + 32u) % K_TILE;

    for (uint group_base = 0; group_base < K_val; group_base += G) {
        uint group_idx = group_base / G;

        half scale_r0[N_TILES_PER_SG];
        half bias_r0[N_TILES_PER_SG];
        half scale_r1[N_TILES_PER_SG];
        half bias_r1[N_TILES_PER_SG];

        #pragma unroll
        for (uint nb = 0; nb < N_TILES_PER_SG; nb++) {
            uint global_n0 = (n_sg_start + nb) * N_TILE + row0;
            uint global_n1 = (n_sg_start + nb) * N_TILE + row1;
            scale_r0[nb] = (global_n0 < N_val) ? w_s[global_n0 * groups_per_row + group_idx] : 0.0h;
            bias_r0[nb]  = (global_n0 < N_val) ? w_b[global_n0 * groups_per_row + group_idx] : 0.0h;
            scale_r1[nb] = (global_n1 < N_val) ? w_s[global_n1 * groups_per_row + group_idx] : 0.0h;
            bias_r1[nb]  = (global_n1 < N_val) ? w_b[global_n1 * groups_per_row + group_idx] : 0.0h;
        }

        for (uint t = 0; t < TILES_PER_GROUP; t++) {
            uint k_base = group_base + t * K_TILE;
            uint global_k0 = k_base + col0;
            uint global_k1 = k_base + col1;

            #pragma unroll
            for (uint nb = 0; nb < N_TILES_PER_SG; nb++) {
                uint global_n0 = (n_sg_start + nb) * N_TILE + row0;
                uint global_n1 = (n_sg_start + nb) * N_TILE + row1;

                half w0 = 0.0h, w1 = 0.0h;
                if (global_n0 < N_val) {
                    uint packed = w_q[global_n0 * uint32s_per_row + global_k0 / 8u];
                    uint nibble = (packed >> ((global_k0 % 8u) * 4u)) & 0xFu;
                    w0 = half(nibble) * scale_r0[nb] + bias_r0[nb];
                }
                if (global_n1 < N_val) {
                    uint packed = w_q[global_n1 * uint32s_per_row + global_k1 / 8u];
                    uint nibble = (packed >> ((global_k1 % 8u) * 4u)) & 0xFu;
                    w1 = half(nibble) * scale_r1[nb] + bias_r1[nb];
                }
                b_tiles[sg_id_in_tg][nb][row0][col0] = w0;
                b_tiles[sg_id_in_tg][nb][row1][col1] = w1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<half, 8, 8> A_tile;
            simdgroup_load(A_tile, &x[k_base], K_val);

            simdgroup_matrix<half, 8, 8> B0, B1, B2, B3;
            simdgroup_load(B0, &b_tiles[sg_id_in_tg][0][0][0], K_TILE, ulong2(0, 0), true);
            simdgroup_load(B1, &b_tiles[sg_id_in_tg][1][0][0], K_TILE, ulong2(0, 0), true);
            simdgroup_load(B2, &b_tiles[sg_id_in_tg][2][0][0], K_TILE, ulong2(0, 0), true);
            simdgroup_load(B3, &b_tiles[sg_id_in_tg][3][0][0], K_TILE, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(C0, A_tile, B0, C0);
            simdgroup_multiply_accumulate(C1, A_tile, B1, C1);
            simdgroup_multiply_accumulate(C2, A_tile, B2, C2);
            simdgroup_multiply_accumulate(C3, A_tile, B3, C3);
        }
    }

    threadgroup float c_bufs[SG_PER_TG][N_TILES_PER_SG][8][8];
    simdgroup_store(C0, &c_bufs[sg_id_in_tg][0][0][0], 8);
    simdgroup_store(C1, &c_bufs[sg_id_in_tg][1][0][0], 8);
    simdgroup_store(C2, &c_bufs[sg_id_in_tg][2][0][0], 8);
    simdgroup_store(C3, &c_bufs[sg_id_in_tg][3][0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local = 0; local < 8u; local++) {
        uint linear = lid + local * 32u;
        uint nb = linear / 64u;
        uint within = linear % 64u;
        uint row = within / N_TILE;
        uint col = within % N_TILE;
        uint global_n = (n_sg_start + nb) * N_TILE + col;
        if (global_n < N_val) {
            y[row * N_val + global_n] = T(c_bufs[sg_id_in_tg][nb][row][col]);
        }
    }
"""


_KERNEL: object | None = None


def _build_kernel() -> object:
    return mx.fast.metal_kernel(
        name="silica_fused_qmm_simdgroup_v7",
        input_names=["x", "w_q", "w_s", "w_b"],
        output_names=["y"],
        source=_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_qmm_simdgroup_v7(
    x: mx.array,
    w_q: mx.array,
    w_s: mx.array,
    w_b: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
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
    if K % group_size != 0 or N % (8 * 8) != 0:
        raise ValueError(f"K%{group_size}=0 and N%64=0 required")
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
    tgs = N // (8 * 8)  # one TG per 8 N tiles
    grid_x = tgs * 64
    out = _KERNEL(  # type: ignore[operator]
        inputs=[x_padded, w_q, w_s, w_b],
        template=[("T", tdtype), ("K", K), ("N", N)],
        grid=(grid_x, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(8, N)],
        output_dtypes=[x_2d.dtype],
    )
    y_padded = out[0]
    y = y_padded[:B, :]
    if x.ndim == 3:
        y = y.reshape(x.shape[0], 1, N)
    return y


__all__ = ["fused_qmm_simdgroup_v7"]
