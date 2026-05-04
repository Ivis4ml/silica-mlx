"""4-bit QMM kernel using ``simdgroup_matrix`` MMA primitives.

Cycle 6 attempt at the multi-day kernel direction the cycle-3 honest
negative pointed at. mlx's internal ``mx.quantized_matmul`` uses
Apple Silicon's ``simdgroup_matrix<T, 8, 8>`` MMA tensor-core-equivalent
primitives; cycle-3 naive 1-thread-per-output kernels could not match
this performance (1.44× slower). This kernel attempts the same approach.

Algorithm:
    For y = x @ dequant(W).T at production shape (B=4, K=5120, N=17408):
    - Pad B from 4 to 8 (M_TILE=8 fixed by Apple Silicon simdgroup_matrix)
    - Tile output into (M=8, N=8) blocks; each simdgroup handles one tile
    - Iterate over K in 8-element chunks; for each:
        * Cooperatively dequantize 8x8 B tile (8 N-cols × 8 K-positions)
          into threadgroup memory using per-group scale/bias
        * simdgroup_load A 8x8 tile from x[B-pad, k:k+8]
        * simdgroup_load B 8x8 tile from threadgroup memory
        * simdgroup_multiply_accumulate(C, A, B, C)
    - simdgroup_store C tile to y; cooperatively write valid (B<8) rows

Padding waste: at B=4, 50% of MMA FLOPs are on padding rows — wasted
work but small (M5 Pro tensor-core throughput is high). The B=4 shape
is chosen by the production warm-decode-b4 scenario; spec-verify k=4
also at B=1×4 = 4 effective rows.

Limitation: this is a single-session attempt at multi-day work. Even
if correctness passes, performance parity with mlx's internal QMM is
unlikely without further tuning (loop unrolling, vectorised loads,
optimal threadgroup sizing). Documented as foundation for future work.
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_SOURCE = """
    #include <metal_simdgroup_matrix>
    using namespace metal;

    // Tile dims: M=8 (B padded by caller), K=8 (per MMA step), N=8 per output block
    constexpr uint K_TILE = 8;
    constexpr uint N_TILE = 8;
    constexpr uint G = 64;  // group_size

    uint K_val = uint(K);
    uint N_val = uint(N);
    uint B_padded = uint(B_PAD);  // caller pads B to 8

    uint n_tile = thread_position_in_grid.x / 32u;
    uint lid = thread_position_in_threadgroup.x;
    if (n_tile * N_TILE >= N_val) return;

    // Single threadgroup memory buffer for B tile (A loaded directly via simdgroup_load
    // since x is pre-padded by caller to (8, K) — no OOB risk).
    threadgroup half b_tile[N_TILE][K_TILE];

    simdgroup_matrix<float, 8, 8> C(0.0f);

    uint groups_per_row = K_val / G;
    uint uint32s_per_row = K_val / 8u;

    for (uint k_base = 0; k_base < K_val; k_base += K_TILE) {
        // Cooperatively dequantize 8x8 = 64 weights across 32 threads (2 each).
        for (uint local = 0; local < 2u; local++) {
            uint linear = lid + local * 32u;  // 0..63
            uint row = linear / K_TILE;        // 0..7 N offset
            uint col = linear % K_TILE;        // 0..7 K offset within tile

            uint global_n = n_tile * N_TILE + row;
            uint global_k = k_base + col;

            half w_val = 0.0h;
            if (global_n < N_val) {
                uint group_idx = global_k / G;
                uint uint32_idx = global_k / 8u;
                uint nibble_pos = global_k % 8u;
                uint packed = w_q[global_n * uint32s_per_row + uint32_idx];
                uint nibble = (packed >> (nibble_pos * 4u)) & 0xFu;
                half scale = w_s[global_n * groups_per_row + group_idx];
                half bias = w_b[global_n * groups_per_row + group_idx];
                w_val = half(nibble) * scale + bias;
            }
            b_tile[row][col] = w_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A 8x8 directly from x (pre-padded to 8 rows by caller).
        // x storage: (B_pad=8, K). For tile at k_base, read x[0..7, k_base..k_base+7].
        // Stride = K_val.
        simdgroup_matrix<half, 8, 8> A_tile;
        simdgroup_load(A_tile, &x[k_base], K_val);

        // Load B from threadgroup memory, transpose to get K x N layout.
        simdgroup_matrix<half, 8, 8> B_tile;
        simdgroup_load(B_tile, &b_tile[0][0], K_TILE, ulong2(0, 0), true);

        simdgroup_multiply_accumulate(C, A_tile, B_tile, C);
    }

    // Store C (fp32) to threadgroup, convert to half, write to y (fp16).
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
        name="silica_fused_qmm_simdgroup_v2",
        input_names=["x", "w_q", "w_s", "w_b"],
        output_names=["y"],
        source=_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_qmm_simdgroup(
    x: mx.array,
    w_q: mx.array,
    w_s: mx.array,
    w_b: mx.array,
    *,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """``y = x @ dequant(w).T`` via simdgroup_matrix MMA, M=8 padded.

    Caller-side: x is padded with zeros to 8 rows; output y is 8 rows; the
    real B rows are sliced from the result (rows 0..B-1).
    """
    if bits != 4 or group_size != 64:
        raise NotImplementedError(
            f"only bits=4 group_size=64 supported; got bits={bits} g={group_size}"
        )
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise NotImplementedError("only T=1 supported")
        x_2d = x.reshape(x.shape[0], x.shape[2])
    elif x.ndim == 2:
        x_2d = x
    else:
        raise ValueError(f"x must be 2D or 3D; got {x.shape}")

    B, K = x_2d.shape
    N = w_q.shape[0]
    if K % 8 != 0 or K % group_size != 0:
        raise ValueError(f"K={K} divisible by 8 and {group_size}")
    if N % 8 != 0:
        raise ValueError(f"N={N} divisible by 8 (simdgroup_matrix tile)")
    if B > 8:
        raise NotImplementedError(f"B>8 not supported; got B={B}")

    # Pad B to 8 with zeros if needed.
    if B < 8:
        pad = mx.zeros((8 - B, K), dtype=x_2d.dtype)
        x_padded = mx.concatenate([x_2d, pad], axis=0)
    else:
        x_padded = x_2d

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()

    if x_2d.dtype == mx.float16:
        tdtype = mx.float16
    elif x_2d.dtype == mx.bfloat16:
        tdtype = mx.bfloat16
    else:
        raise ValueError(f"only fp16/bf16 supported; got {x_2d.dtype}")

    n_tiles = N // 8
    grid_x = n_tiles * 32
    out = _KERNEL(  # type: ignore[operator]
        inputs=[x_padded, w_q, w_s, w_b],
        template=[("T", tdtype), ("B_PAD", 8), ("K", K), ("N", N)],
        grid=(grid_x, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(8, N)],
        output_dtypes=[x_2d.dtype],
    )
    y_padded = out[0]
    y = y_padded[:B, :]  # slice to real B rows
    if x.ndim == 3:
        y = y.reshape(x.shape[0], 1, N)
    return y


__all__ = ["fused_qmm_simdgroup"]
