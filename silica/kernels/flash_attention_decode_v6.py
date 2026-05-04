"""FA-decode v6: K-split (v4) + streaming softmax (v5) + tighter inner loop.

v4's K-split helped at long T_kv but still trailed mlx by 1.5×. v6:
  - keeps v4's K-split (parallel TGs over K chunks)
  - drops scores[TK] register array (streaming softmax from v5)
  - tightens the per-K-row inner loop: q dot, simd_sum, online-softmax,
    V accumulate all in one pass

The forward kernel is now leaner. Merge kernel is unchanged from v4.
"""

from __future__ import annotations

import mlx.core as mx

SPLIT_K = 128


_FORWARD_SOURCE = """
    constexpr uint TK = 32u;
    constexpr uint D = 256u;
    constexpr uint TPS = 32u;
    constexpr uint D_PER_T = D / TPS;
    constexpr uint Q_PER_KV = 6u;
    constexpr uint THREADS_PER_TG = TPS * Q_PER_KV;

    uint H_q = uint(HQ);
    uint H_kv = uint(HKV);
    uint T_kv = uint(TKV);
    uint B_val = uint(B);
    uint N_splits = uint(NSPLITS);
    uint split_k = uint(SPLITK);

    uint tid_in_tg = thread_position_in_threadgroup.x;
    uint lid = tid_in_tg % TPS;
    uint sg_in_tg = tid_in_tg / TPS;
    uint tg_idx = threadgroup_position_in_grid.x;
    uint split_idx = tg_idx % N_splits;
    uint bhk = tg_idx / N_splits;
    uint b = bhk / H_kv;
    uint h_kv = bhk % H_kv;
    uint h_q = h_kv * Q_PER_KV + sg_in_tg;
    if (b >= B_val) return;

    uint k_start = split_idx * split_k;
    uint k_end = min(k_start + split_k, T_kv);
    if (k_start >= T_kv) {
        if (lid == 0) {
            uint pml_base = (b * H_q + h_q) * N_splits + split_idx;
            partial_m[pml_base] = -INFINITY;
            partial_l[pml_base] = 0.0f;
        }
        uint po_base = ((b * H_q + h_q) * N_splits + split_idx) * D;
        for (uint i = 0; i < D_PER_T; i++) {
            partial_o[po_base + lid * D_PER_T + i] = 0.0f;
        }
        return;
    }

    half q_local[D_PER_T];
    uint q_base = (b * H_q + h_q) * D;
    for (uint i = 0; i < D_PER_T; i++) {
        q_local[i] = q[q_base + lid * D_PER_T + i];
    }

    float m_cur = -INFINITY;
    float l_cur = 0.0f;
    float o_local[D_PER_T];
    for (uint i = 0; i < D_PER_T; i++) o_local[i] = 0.0f;

    float scale = scale_buf[0];

    threadgroup half k_tile[TK][D];
    threadgroup half v_tile[TK][D];

    uint kv_base = (b * H_kv + h_kv) * T_kv * D;
    uint k_pos = k_start;
    while (k_pos < k_end) {
        uint tk_actual = min(TK, k_end - k_pos);

        for (uint cell_idx = tid_in_tg; cell_idx < TK * D; cell_idx += THREADS_PER_TG) {
            uint row = cell_idx / D;
            uint col = cell_idx % D;
            if (row < tk_actual) {
                k_tile[row][col] = k[kv_base + (k_pos + row) * D + col];
                v_tile[row][col] = v[kv_base + (k_pos + row) * D + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Streaming softmax over each K row in this tile.
        for (uint k_idx = 0; k_idx < tk_actual; k_idx++) {
            float partial = 0.0f;
            for (uint i = 0; i < D_PER_T; i++) {
                partial += float(q_local[i]) * float(k_tile[k_idx][lid * D_PER_T + i]);
            }
            float s_k = simd_sum(partial) * scale;

            float m_new = max(m_cur, s_k);
            float alpha = (m_cur == -INFINITY) ? 0.0f : metal::exp(m_cur - m_new);
            float p_k = metal::exp(s_k - m_new);

            float l_new = l_cur * alpha + p_k;
            for (uint i = 0; i < D_PER_T; i++) {
                float v_val = float(v_tile[k_idx][lid * D_PER_T + i]);
                o_local[i] = alpha * o_local[i] + p_k * v_val;
            }
            m_cur = m_new;
            l_cur = l_new;
        }

        k_pos += TK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint pml_base = (b * H_q + h_q) * N_splits + split_idx;
    if (lid == 0) {
        partial_m[pml_base] = m_cur;
        partial_l[pml_base] = l_cur;
    }
    uint po_base = pml_base * D;
    for (uint i = 0; i < D_PER_T; i++) {
        partial_o[po_base + lid * D_PER_T + i] = o_local[i];
    }
"""


_MERGE_SOURCE = """
    constexpr uint D = 256u;
    constexpr uint TPS = 32u;
    constexpr uint D_PER_T = D / TPS;

    uint H_q = uint(HQ);
    uint B_val = uint(B);
    uint N_splits = uint(NSPLITS);

    uint tid = thread_position_in_grid.x;
    uint sg_id = tid / TPS;
    uint lid = tid % TPS;
    uint b = sg_id / H_q;
    uint h_q = sg_id % H_q;
    if (b >= B_val) return;

    float m_total = -INFINITY;
    uint pml_base_bh = (b * H_q + h_q) * N_splits;
    for (uint s = 0; s < N_splits; s++) {
        m_total = max(m_total, partial_m[pml_base_bh + s]);
    }

    float l_total = 0.0f;
    float o_acc[D_PER_T];
    for (uint i = 0; i < D_PER_T; i++) o_acc[i] = 0.0f;

    for (uint s = 0; s < N_splits; s++) {
        float m_s = partial_m[pml_base_bh + s];
        float l_s = partial_l[pml_base_bh + s];
        if (l_s == 0.0f || m_s == -INFINITY) continue;
        float alpha = metal::exp(m_s - m_total);
        l_total += alpha * l_s;
        uint po_base = (pml_base_bh + s) * D;
        for (uint i = 0; i < D_PER_T; i++) {
            o_acc[i] += alpha * partial_o[po_base + lid * D_PER_T + i];
        }
    }

    float inv_l = 1.0f / l_total;
    uint q_base = (b * H_q + h_q) * D;
    uint o_base = q_base;
    for (uint i = 0; i < D_PER_T; i++) {
        uint d_idx = lid * D_PER_T + i;
        float out_val = o_acc[i] * inv_l;
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


_FWD_KERNEL: object | None = None
_MERGE_KERNEL_PLAIN: object | None = None
_MERGE_KERNEL_GATED: object | None = None


def _build_fwd() -> object:
    return mx.fast.metal_kernel(
        name="silica_flash_attention_decode_v6_fwd",
        input_names=["q", "k", "v", "scale_buf"],
        output_names=["partial_o", "partial_m", "partial_l"],
        source=_FORWARD_SOURCE,
        ensure_row_contiguous=True,
    )


def _build_merge(has_gate: bool) -> object:
    return mx.fast.metal_kernel(
        name=f"silica_flash_attention_decode_v6_merge_{'gated' if has_gate else 'plain'}",
        input_names=["partial_o", "partial_m", "partial_l", "gate"],
        output_names=["out"],
        source=_MERGE_SOURCE,
        ensure_row_contiguous=True,
    )


def flash_attention_decode_v6(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    gate: mx.array | None = None,
) -> mx.array:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be 4D")
    B, H_q, T_q, D = q.shape
    Bk, H_kv, T_kv, Dk = k.shape
    if T_q != 1:
        raise NotImplementedError(f"only T_q=1 (decode); got {T_q}")
    if D != 256:
        raise NotImplementedError(f"only head_dim=256")
    if H_q // H_kv != 6:
        raise NotImplementedError(f"v6 hardcodes Q_PER_KV=6")
    if scale is None:
        scale = D ** -0.5

    has_gate = gate is not None
    if has_gate and gate.shape != q.shape:
        raise ValueError("gate shape mismatch")

    n_splits = (T_kv + SPLIT_K - 1) // SPLIT_K

    global _FWD_KERNEL, _MERGE_KERNEL_PLAIN, _MERGE_KERNEL_GATED
    if _FWD_KERNEL is None:
        _FWD_KERNEL = _build_fwd()
    merge_holder = _MERGE_KERNEL_GATED if has_gate else _MERGE_KERNEL_PLAIN
    if merge_holder is None:
        merge_holder = _build_merge(has_gate)
        if has_gate:
            _MERGE_KERNEL_GATED = merge_holder
        else:
            _MERGE_KERNEL_PLAIN = merge_holder

    if q.dtype == mx.float16:
        tdtype = mx.float16
    elif q.dtype == mx.bfloat16:
        tdtype = mx.bfloat16
    else:
        raise ValueError(f"only fp16/bf16; got {q.dtype}")

    threads_per_tg = 32 * 6
    n_fwd_tg = B * H_kv * n_splits
    fwd_grid_x = n_fwd_tg * threads_per_tg
    scale_arr = mx.array([float(scale)], dtype=mx.float32)

    p_o, p_m, p_l = _FWD_KERNEL(  # type: ignore[operator]
        inputs=[q, k, v, scale_arr],
        template=[
            ("T", tdtype),
            ("HQ", H_q),
            ("HKV", H_kv),
            ("TKV", T_kv),
            ("B", B),
            ("NSPLITS", n_splits),
            ("SPLITK", SPLIT_K),
        ],
        grid=(fwd_grid_x, 1, 1),
        threadgroup=(threads_per_tg, 1, 1),
        output_shapes=[
            (B, H_q, n_splits, D),
            (B, H_q, n_splits),
            (B, H_q, n_splits),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    n_merge_sg = B * H_q
    merge_grid_x = n_merge_sg * 32
    out_arr = merge_holder(  # type: ignore[operator]
        inputs=[p_o, p_m, p_l, gate if has_gate else q],
        template=[
            ("T", tdtype),
            ("HQ", H_q),
            ("B", B),
            ("NSPLITS", n_splits),
            ("HAS_GATE", 1 if has_gate else 0),
        ],
        grid=(merge_grid_x, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H_q, 1, D)],
        output_dtypes=[q.dtype],
    )
    return out_arr[0]


__all__ = ["flash_attention_decode_v6"]
