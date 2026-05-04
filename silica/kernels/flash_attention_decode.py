"""FlashAttention-decode (FlashDecoding) kernel for MLX, with Qwen3.5 output gate.

Cycle 11. Port of Tri Dao's FlashAttention-2 algorithm (./flash-attention) to
native MLX via ``mx.fast.metal_kernel``. For T_q=1 (decode) the algorithm
simplifies to FlashDecoding: streaming online-softmax over K-tiles, no Q
tiling needed.

Algorithm (per output (b, h_q)):
    Initialize: m = -inf, l = 0, o = zeros(D)
    For each K-tile (block of TK rows):
        s = Q @ K_tile.T / sqrt(D)             # (TK,)
        m_new = max(m, max(s))
        alpha = exp(m - m_new)
        p = exp(s - m_new)                     # (TK,)
        o = alpha * o + p @ V_tile             # (D,)
        l = alpha * l + sum(p)
        m = m_new
    O = (1 / l) * o                            # final attention output

Qwen3.5 gated-attention extension (the load-bearing Silica delta vs public FA-2):
    O_final = sigmoid(gate) * O
    The 2026-Q2 survey confirmed no public Apple-Silicon kernel implements
    this fused output gate inside the FA epilogue — mlx-lm does it as
    two un-fused ops on the SDPA output (qwen3_next.py:158).

Shape contract (Qwen3.5-27B-4bit production decode):
    Q:    (B, H_q, T_q=1, D=256)
    K:    (B, H_kv, T_kv, D=256)            # T_kv = current context length
    V:    (B, H_kv, T_kv, D_v=256)
    gate: (B, H_q, T_q=1, D=256)             # post-q_proj split
    Out:  (B, H_q, T_q=1, D=256)

GQA: H_q / H_kv = 6 (24 Q heads / 4 KV heads per Qwen3.5-27B-4bit config).

Threadgroup layout:
    - One simdgroup per (b, h_q) pair → B × H_q simdgroups total
    - 32 threads cooperate on the K dimension
    - Each thread holds D/32 = 8 elements of Q in registers (256/32)
    - K/V tiles read via cooperative threadgroup loads
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_SOURCE = """
    constexpr uint TK = 32u;          // K-tile size
    constexpr uint D = 256u;           // head dim (Qwen3.5-27B)
    constexpr uint TPS = 32u;          // threads per simdgroup
    constexpr uint D_PER_T = D / TPS;  // = 8 elements of Q per thread

    uint H_q = uint(HQ);
    uint H_kv = uint(HKV);
    uint q_per_kv = H_q / H_kv;  // GQA ratio
    uint T_kv = uint(TKV);
    uint B_val = uint(B);

    uint sg_id = thread_position_in_grid.x / TPS;
    uint lid = thread_position_in_threadgroup.x;
    uint b = sg_id / H_q;
    uint h_q = sg_id % H_q;
    uint h_kv = h_q / q_per_kv;
    if (b >= B_val) return;

    // Load Q[b, h_q, 0, :] into thread registers.
    // Q stored row-major (B, H_q, T_q=1, D); base offset = (b * H_q + h_q) * D.
    half q_local[D_PER_T];
    uint q_base = (b * H_q + h_q) * D;
    for (uint i = 0; i < D_PER_T; i++) {
        uint d_idx = lid * D_PER_T + i;
        q_local[i] = q[q_base + d_idx];
    }

    // Online softmax state, fp32 for numerical stability.
    float m_cur = -INFINITY;
    float l_cur = 0.0f;
    float o_local[D_PER_T];
    for (uint i = 0; i < D_PER_T; i++) o_local[i] = 0.0f;

    float scale = scale_buf[0];

    // Threadgroup memory for K and V tiles.
    threadgroup half k_tile[TK][D];
    threadgroup half v_tile[TK][D];

    // Iterate K-tiles
    uint kv_base = (b * H_kv + h_kv) * T_kv * D;
    uint k_pos = 0;
    while (k_pos < T_kv) {
        uint tk_actual = min(TK, T_kv - k_pos);

        // Cooperatively load TK × D K and V values into threadgroup memory.
        // 32 threads × D_PER_T = 256 = D elements per K row → each thread
        // loads D_PER_T = 8 elements per (k row). Across TK=32 K rows,
        // each thread handles 1 K row × 8 elements? No: 32 threads × TK=32 = 1024
        // (row, col) cells, threads each load 1024/32 = 32 cells = 1 row × D_PER_T cols × 4 rounds.
        for (uint local_k = 0; local_k < TK; local_k += 1) {
            // For each k_row in this iter, threads cooperate to load D=256 values.
            uint global_k = k_pos + local_k;
            if (global_k >= T_kv) {
                // Pad with zero for safety (will be masked by m_cur below)
                for (uint i = 0; i < D_PER_T; i++) {
                    k_tile[local_k][lid * D_PER_T + i] = 0.0h;
                    v_tile[local_k][lid * D_PER_T + i] = 0.0h;
                }
                continue;
            }
            for (uint i = 0; i < D_PER_T; i++) {
                uint d_idx = lid * D_PER_T + i;
                k_tile[local_k][d_idx] = k[kv_base + global_k * D + d_idx];
                v_tile[local_k][d_idx] = v[kv_base + global_k * D + d_idx];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores s[k] = Q · K[k] / sqrt(D) for each k in this tile.
        // Use simd_sum to combine partial dot products from each thread.
        float scores[TK];
        for (uint k_idx = 0; k_idx < TK; k_idx++) {
            uint global_k = k_pos + k_idx;
            float partial = 0.0f;
            if (global_k < T_kv) {
                for (uint i = 0; i < D_PER_T; i++) {
                    float q_val = float(q_local[i]);
                    float k_val = float(k_tile[k_idx][lid * D_PER_T + i]);
                    partial += q_val * k_val;
                }
            }
            // simd_sum reduces across the 32 threads, all get the same total.
            scores[k_idx] = simd_sum(partial) * scale;
            if (global_k >= T_kv) scores[k_idx] = -INFINITY;
        }

        // Online softmax update.
        float m_new = m_cur;
        for (uint k_idx = 0; k_idx < TK; k_idx++) {
            m_new = max(m_new, scores[k_idx]);
        }

        float alpha = (m_cur == -INFINITY) ? 0.0f : metal::exp(m_cur - m_new);
        float l_new = l_cur * alpha;

        // p[k] = exp(s[k] - m_new); o += p · V[k]
        for (uint i = 0; i < D_PER_T; i++) {
            o_local[i] *= alpha;
        }
        for (uint k_idx = 0; k_idx < TK; k_idx++) {
            float p_k = metal::exp(scores[k_idx] - m_new);
            l_new += p_k;
            for (uint i = 0; i < D_PER_T; i++) {
                float v_val = float(v_tile[k_idx][lid * D_PER_T + i]);
                o_local[i] += p_k * v_val;
            }
        }

        m_cur = m_new;
        l_cur = l_new;
        k_pos += TK;
    }

    // Finalize: O = (1/l) * o; apply gate if requested.
    float inv_l = 1.0f / l_cur;
    uint o_base = (b * H_q + h_q) * D;
    for (uint i = 0; i < D_PER_T; i++) {
        uint d_idx = lid * D_PER_T + i;
        float out_val = o_local[i] * inv_l;
        if (HAS_GATE) {
            float g_val = float(gate[q_base + d_idx]);
            // sigmoid with saturation guard for fp16 stability
            float s_val;
            if (g_val > 16.0f) s_val = 1.0f;
            else if (g_val < -16.0f) s_val = 0.0f;
            else s_val = 1.0f / (1.0f + metal::exp(-g_val));
            out_val *= s_val;
        }
        out[o_base + d_idx] = T(out_val);
    }
"""


_KERNEL_PLAIN: object | None = None
_KERNEL_GATED: object | None = None


def _build_kernel(has_gate: bool) -> object:
    """Build the kernel; cached separately for plain / gated variants.

    ``gate`` is always declared as a kernel parameter so the source compiles
    cleanly. When ``HAS_GATE=0`` the gate-access branch is dead-code-eliminated
    by the Metal compiler at compile time. Plain calls pass ``q`` as a
    placeholder for the ``gate`` slot.
    """
    return mx.fast.metal_kernel(
        name=f"silica_flash_attention_decode_{'gated' if has_gate else 'plain'}",
        input_names=["q", "k", "v", "scale_buf", "gate"],
        output_names=["out"],
        source=_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def flash_attention_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    gate: mx.array | None = None,
) -> mx.array:
    """FlashDecoding for Qwen3.5-27B shape, optionally fused with output gate.

    Args:
        q: (B, H_q, T_q=1, D=256) queries.
        k: (B, H_kv, T_kv, D=256) keys (current ctx).
        v: (B, H_kv, T_kv, D=256) values.
        scale: 1/sqrt(D) by default.
        gate: optional (B, H_q, T_q=1, D=256). When given, fused
              ``out = sigmoid(gate) * SDPA(...)`` per Qwen3.5 gated attention.

    Returns:
        out: (B, H_q, T_q=1, D=256). dtype matches q.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be 4D; got {q.shape}, {k.shape}, {v.shape}")
    B, H_q, T_q, D = q.shape
    Bk, H_kv, T_kv, Dk = k.shape
    if T_q != 1:
        raise NotImplementedError(f"only T_q=1 (decode) supported; got {T_q}")
    if D != 256 or Dk != 256:
        raise NotImplementedError(f"only head_dim=256 supported; got q D={D} k D={Dk}")
    if B != Bk:
        raise ValueError(f"B mismatch: q B={B}, k B={Bk}")
    if H_q % H_kv != 0:
        raise ValueError(f"H_q ({H_q}) must be divisible by H_kv ({H_kv})")
    if scale is None:
        scale = D ** -0.5

    has_gate = gate is not None
    if has_gate:
        if gate.shape != q.shape:
            raise ValueError(f"gate shape {gate.shape} must match q shape {q.shape}")

    global _KERNEL_PLAIN, _KERNEL_GATED
    kernel_holder = _KERNEL_GATED if has_gate else _KERNEL_PLAIN
    if kernel_holder is None:
        kernel_holder = _build_kernel(has_gate)
        if has_gate:
            _KERNEL_GATED = kernel_holder
        else:
            _KERNEL_PLAIN = kernel_holder

    if q.dtype == mx.float16:
        tdtype = mx.float16
    elif q.dtype == mx.bfloat16:
        tdtype = mx.bfloat16
    else:
        raise ValueError(f"only fp16/bf16; got {q.dtype}")

    n_simdgroups = B * H_q
    grid_x = n_simdgroups * 32
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    inputs = [q, k, v, scale_arr, gate if has_gate else q]

    out = kernel_holder(  # type: ignore[operator]
        inputs=inputs,
        template=[
            ("T", tdtype),
            ("HQ", H_q),
            ("HKV", H_kv),
            ("TKV", T_kv),
            ("B", B),
            ("HAS_GATE", 1 if has_gate else 0),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H_q, 1, D)],
        output_dtypes=[q.dtype],
    )
    return out[0]


def reference_flash_attention_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    gate: mx.array | None = None,
) -> mx.array:
    """MLX reference using mx.fast.scaled_dot_product_attention + optional gate."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    if gate is not None:
        out = out * mx.sigmoid(gate)
    return out


__all__ = ["flash_attention_decode", "reference_flash_attention_decode"]
