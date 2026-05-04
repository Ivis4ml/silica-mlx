"""FlashAttention-decode v2: GQA-aware K/V tile sharing across q_per_kv heads.

Cycle 11 v2. The v1 kernel had each simdgroup load its own K/V tile from HBM,
so within a GQA group of q_per_kv=6 q heads (sharing 1 kv head) we paid 6×
the bandwidth. v2 puts q_per_kv simdgroups inside one threadgroup and loads
K/V once per (b, h_kv) per K-tile.

Layout (Qwen3.5 GQA: H_q=24, H_kv=4, q_per_kv=6):
    - 1 threadgroup per (b, h_kv) → B * H_kv = 48 * 4 = 192 TGs (was 1152)
    - 6 simdgroups per TG, one per h_q in the GQA group → 192 threads
    - Threadgroup mem: shared k_tile[TK][D] then v_tile[TK][D] (loaded sequentially
      to fit in 32 KB)
    - Each simdgroup keeps its own (q_local, m, l, o) registers

Algorithm per (b, h_kv):
    For each K-tile:
        Cooperatively load K tile (192 threads, ~43 cells each).
        For each simdgroup in TG (h_q = h_kv*q_per_kv + sg_idx):
            Compute scores s[k] = q · K[k] / sqrt(D) using simd_sum.
            Update online-softmax (m, l, alpha).
        Cooperatively load V tile (overwriting K tile region).
        For each simdgroup:
            o = alpha * o + softmax(s) · V

Final per-simdgroup: O = (1/l) * o, with optional sigmoid-gate fusion.
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_SOURCE = """
    constexpr uint TK = 32u;
    constexpr uint D = 256u;
    constexpr uint TPS = 32u;
    constexpr uint D_PER_T = D / TPS;        // = 8
    constexpr uint Q_PER_KV = 6u;            // Qwen3.5: H_q=24, H_kv=4
    constexpr uint THREADS_PER_TG = TPS * Q_PER_KV;  // 192

    uint H_q = uint(HQ);
    uint H_kv = uint(HKV);
    uint T_kv = uint(TKV);
    uint B_val = uint(B);

    // Thread / simdgroup indexing within the threadgroup
    uint tid_in_tg = thread_position_in_threadgroup.x;
    uint lid = tid_in_tg % TPS;
    uint sg_in_tg = tid_in_tg / TPS;          // 0..Q_PER_KV-1
    uint tg_idx = threadgroup_position_in_grid.x;
    uint b = tg_idx / H_kv;
    uint h_kv = tg_idx % H_kv;
    uint h_q = h_kv * Q_PER_KV + sg_in_tg;
    if (b >= B_val) return;

    // Load Q[b, h_q, 0, :] into per-simdgroup thread registers.
    half q_local[D_PER_T];
    uint q_base = (b * H_q + h_q) * D;
    for (uint i = 0; i < D_PER_T; i++) {
        uint d_idx = lid * D_PER_T + i;
        q_local[i] = q[q_base + d_idx];
    }

    // Online softmax state (per simdgroup, in registers)
    float m_cur = -INFINITY;
    float l_cur = 0.0f;
    float o_local[D_PER_T];
    for (uint i = 0; i < D_PER_T; i++) o_local[i] = 0.0f;

    float scale = scale_buf[0];

    // Threadgroup-shared K/V tile (single buffer, K then V)
    threadgroup half kv_tile[TK][D];

    uint kv_base = (b * H_kv + h_kv) * T_kv * D;
    uint k_pos = 0;
    while (k_pos < T_kv) {
        uint tk_actual = min(TK, T_kv - k_pos);

        // ============ Stage 1: cooperative K load ============
        // 192 threads cooperate to load TK=32 × D=256 = 8192 cells.
        // Each thread loads ceil(8192/192) ≈ 43 cells, strided by THREADS_PER_TG.
        for (uint cell_idx = tid_in_tg; cell_idx < TK * D; cell_idx += THREADS_PER_TG) {
            uint row = cell_idx / D;
            uint col = cell_idx % D;
            half val;
            if (row < tk_actual) {
                val = k[kv_base + (k_pos + row) * D + col];
            } else {
                val = 0.0h;
            }
            kv_tile[row][col] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============ Stage 2: compute scores per simdgroup ============
        float scores[TK];
        for (uint k_idx = 0; k_idx < TK; k_idx++) {
            float partial = 0.0f;
            if (k_idx < tk_actual) {
                for (uint i = 0; i < D_PER_T; i++) {
                    partial += float(q_local[i]) * float(kv_tile[k_idx][lid * D_PER_T + i]);
                }
            }
            scores[k_idx] = simd_sum(partial) * scale;
            if (k_idx >= tk_actual) scores[k_idx] = -INFINITY;
        }

        // ============ Stage 3: online-softmax update ============
        float m_new = m_cur;
        for (uint k_idx = 0; k_idx < TK; k_idx++) {
            m_new = max(m_new, scores[k_idx]);
        }
        float alpha = (m_cur == -INFINITY) ? 0.0f : metal::exp(m_cur - m_new);
        float l_new = l_cur * alpha;
        for (uint i = 0; i < D_PER_T; i++) {
            o_local[i] *= alpha;
        }

        // Compute p[k] = exp(s[k] - m_new) AND sum-of-p, store p to shared mem
        // for the V multiply stage. We reuse the kv_tile buffer's first row as
        // p storage (flat p[k] for k=0..TK-1) since K is no longer needed.
        // Actually we need p per simdgroup. Keep p in registers + scalar l accumulation.
        float p_local[TK];
        for (uint k_idx = 0; k_idx < TK; k_idx++) {
            if (k_idx < tk_actual) {
                float p_k = metal::exp(scores[k_idx] - m_new);
                p_local[k_idx] = p_k;
                l_new += p_k;
            } else {
                p_local[k_idx] = 0.0f;
            }
        }

        // ============ Stage 4: barrier, then load V into the same tile slot ============
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint cell_idx = tid_in_tg; cell_idx < TK * D; cell_idx += THREADS_PER_TG) {
            uint row = cell_idx / D;
            uint col = cell_idx % D;
            half val;
            if (row < tk_actual) {
                val = v[kv_base + (k_pos + row) * D + col];
            } else {
                val = 0.0h;
            }
            kv_tile[row][col] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============ Stage 5: o += p · V ============
        for (uint k_idx = 0; k_idx < TK; k_idx++) {
            float p_k = p_local[k_idx];
            if (p_k != 0.0f) {
                for (uint i = 0; i < D_PER_T; i++) {
                    float v_val = float(kv_tile[k_idx][lid * D_PER_T + i]);
                    o_local[i] += p_k * v_val;
                }
            }
        }

        m_cur = m_new;
        l_cur = l_new;
        k_pos += TK;
    }

    // ============ Finalize: O = (1/l) * o; apply gate if requested ============
    float inv_l = 1.0f / l_cur;
    uint o_base = (b * H_q + h_q) * D;
    for (uint i = 0; i < D_PER_T; i++) {
        uint d_idx = lid * D_PER_T + i;
        float out_val = o_local[i] * inv_l;
        if (HAS_GATE) {
            float g_val = float(gate[q_base + d_idx]);
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
    return mx.fast.metal_kernel(
        name=f"silica_flash_attention_decode_v2_{'gated' if has_gate else 'plain'}",
        input_names=["q", "k", "v", "scale_buf", "gate"],
        output_names=["out"],
        source=_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def flash_attention_decode_v2(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    gate: mx.array | None = None,
) -> mx.array:
    """v2 with GQA-aware K/V tile sharing. Requires q_per_kv=6 (Qwen3.5 shape)."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be 4D; got {q.shape}, {k.shape}, {v.shape}")
    B, H_q, T_q, D = q.shape
    Bk, H_kv, T_kv, Dk = k.shape
    if T_q != 1:
        raise NotImplementedError(f"only T_q=1 (decode) supported; got {T_q}")
    if D != 256 or Dk != 256:
        raise NotImplementedError(f"only head_dim=256 supported; got q D={D} k D={Dk}")
    if H_q // H_kv != 6:
        raise NotImplementedError(
            f"v2 hardcodes Q_PER_KV=6 (Qwen3.5 GQA); got H_q={H_q} H_kv={H_kv}"
        )
    if scale is None:
        scale = D ** -0.5

    has_gate = gate is not None
    if has_gate and gate.shape != q.shape:
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

    n_tg = B * H_kv
    threads_per_tg = 32 * 6  # = 192
    grid_x = n_tg * threads_per_tg
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
        threadgroup=(threads_per_tg, 1, 1),
        output_shapes=[(B, H_q, 1, D)],
        output_dtypes=[q.dtype],
    )
    return out[0]


__all__ = ["flash_attention_decode_v2"]
