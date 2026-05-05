"""FA-decode v8: v7 + vectorized inner score and V-multiply via half4.

v7 vectorized only the K/V load. v8 also reads from threadgroup memory in
half4 chunks for both the Q·K dot product and the p·V accumulate. Because
lid * D_PER_T = 8 * lid is 8-aligned (= 2 half4 boundary), TG-memory reads
can be vectorized cleanly.
"""

from __future__ import annotations

import mlx.core as mx

SPLIT_K = 128


_FORWARD_SOURCE = """
    constexpr uint TK = 32u;
    constexpr uint D = 256u;
    constexpr uint TPS = 32u;
    constexpr uint D_PER_T = D / TPS;        // = 8 scalar halfs per thread per row
    constexpr uint D_PER_T_VEC = D_PER_T / 4; // = 2 half4 per thread per row
    constexpr uint D_VEC = D / 4;             // = 64 half4 along D
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

    // Load Q as half4 vectors per thread (D_PER_T_VEC = 2 vectors per thread).
    half4 q_local_vec[D_PER_T_VEC];
    uint q_base = (b * H_q + h_q) * D;
    device half4 const* q_vec_ptr = reinterpret_cast<device half4 const*>(q);
    uint q_base_vec = q_base / 4;
    for (uint vi = 0; vi < D_PER_T_VEC; vi++) {
        q_local_vec[vi] = q_vec_ptr[q_base_vec + lid * D_PER_T_VEC + vi];
    }

    float m_cur = -INFINITY;
    float l_cur = 0.0f;
    float4 o_local_vec[D_PER_T_VEC];
    for (uint vi = 0; vi < D_PER_T_VEC; vi++) o_local_vec[vi] = float4(0.0f);

    float scale = scale_buf[0];

    threadgroup half k_tile[TK][D];
    threadgroup half v_tile[TK][D];

    device half4 const* k_vec = reinterpret_cast<device half4 const*>(k);
    device half4 const* v_vec = reinterpret_cast<device half4 const*>(v);

    uint kv_base = (b * H_kv + h_kv) * T_kv * D;
    uint kv_base_vec = kv_base / 4;
    uint k_pos = k_start;
    while (k_pos < k_end) {
        uint tk_actual = min(TK, k_end - k_pos);

        // Vectorized cooperative load of K, V tiles.
        for (uint vec_idx = tid_in_tg; vec_idx < TK * D_VEC; vec_idx += THREADS_PER_TG) {
            uint row = vec_idx / D_VEC;
            uint vcol = vec_idx % D_VEC;
            if (row < tk_actual) {
                uint global_row = k_pos + row;
                uint global_idx = kv_base_vec + global_row * D_VEC + vcol;
                half4 kv_k = k_vec[global_idx];
                half4 kv_v = v_vec[global_idx];
                uint col4 = vcol * 4;
                k_tile[row][col4 + 0] = kv_k.x;
                k_tile[row][col4 + 1] = kv_k.y;
                k_tile[row][col4 + 2] = kv_k.z;
                k_tile[row][col4 + 3] = kv_k.w;
                v_tile[row][col4 + 0] = kv_v.x;
                v_tile[row][col4 + 1] = kv_v.y;
                v_tile[row][col4 + 2] = kv_v.z;
                v_tile[row][col4 + 3] = kv_v.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Vectorized streaming softmax over each K row.
        threadgroup half4* k_tile_vec = reinterpret_cast<threadgroup half4*>(&k_tile[0][0]);
        threadgroup half4* v_tile_vec = reinterpret_cast<threadgroup half4*>(&v_tile[0][0]);

        for (uint k_idx = 0; k_idx < tk_actual; k_idx++) {
            // Q · K[k_idx] via half4 dot products.
            float partial = 0.0f;
            for (uint vi = 0; vi < D_PER_T_VEC; vi++) {
                half4 k_vec4 = k_tile_vec[k_idx * D_VEC + lid * D_PER_T_VEC + vi];
                half4 q_vec4 = q_local_vec[vi];
                // dot(half4, half4) -> half; cast to float for accumulation.
                partial += float(metal::dot(q_vec4, k_vec4));
            }
            float s_k = simd_sum(partial) * scale;

            float m_new = max(m_cur, s_k);
            float alpha = (m_cur == -INFINITY) ? 0.0f : metal::exp(m_cur - m_new);
            float p_k = metal::exp(s_k - m_new);

            float l_new = l_cur * alpha + p_k;

            // Vectorized o += alpha * o + p_k * V[k_idx]
            for (uint vi = 0; vi < D_PER_T_VEC; vi++) {
                half4 v_vec4 = v_tile_vec[k_idx * D_VEC + lid * D_PER_T_VEC + vi];
                float4 v_f = float4(v_vec4);
                o_local_vec[vi] = alpha * o_local_vec[vi] + p_k * v_f;
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
    device float4* po_vec = reinterpret_cast<device float4*>(partial_o);
    uint po_base_vec = po_base / 4;
    for (uint vi = 0; vi < D_PER_T_VEC; vi++) {
        po_vec[po_base_vec + lid * D_PER_T_VEC + vi] = o_local_vec[vi];
    }
"""


_FORWARD_SOURCE_BF16 = (
    _FORWARD_SOURCE
    .replace("threadgroup half k_tile", "threadgroup bfloat16_t k_tile")
    .replace("threadgroup half v_tile", "threadgroup bfloat16_t v_tile")
    .replace("half4", "bfloat4")
    .replace(
        "partial += float(metal::dot(q_vec4, k_vec4));",
        "partial += metal::dot(float4(q_vec4), float4(k_vec4));",
    )
)


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


_FWD_KERNEL_FP16: object | None = None
_FWD_KERNEL_BF16: object | None = None
_MERGE_KERNEL_PLAIN: object | None = None
_MERGE_KERNEL_GATED: object | None = None


def _build_fwd(*, bf16: bool) -> object:
    return mx.fast.metal_kernel(
        name=f"silica_flash_attention_decode_v8_fwd_{'bf16' if bf16 else 'fp16'}",
        input_names=["q", "k", "v", "scale_buf"],
        output_names=["partial_o", "partial_m", "partial_l"],
        source=_FORWARD_SOURCE_BF16 if bf16 else _FORWARD_SOURCE,
        ensure_row_contiguous=True,
    )


def _build_merge(has_gate: bool) -> object:
    return mx.fast.metal_kernel(
        name=f"silica_flash_attention_decode_v8_merge_{'gated' if has_gate else 'plain'}",
        input_names=["partial_o", "partial_m", "partial_l", "gate"],
        output_names=["out"],
        source=_MERGE_SOURCE,
        ensure_row_contiguous=True,
    )


def flash_attention_decode_v8(
    q: mx.array, k: mx.array, v: mx.array, *,
    scale: float | None = None, gate: mx.array | None = None,
) -> mx.array:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be 4D")
    B, H_q, T_q, D = q.shape
    Bk, H_kv, T_kv, Dk = k.shape
    if T_q != 1:
        raise NotImplementedError(f"only T_q=1; got {T_q}")
    if D != 256:
        raise NotImplementedError("only head_dim=256")
    if H_q // H_kv != 6:
        raise NotImplementedError("v8 hardcodes Q_PER_KV=6")
    if scale is None:
        scale = D ** -0.5

    has_gate = gate is not None
    if gate is not None and gate.shape != q.shape:
        raise ValueError("gate shape mismatch")

    n_splits = (T_kv + SPLIT_K - 1) // SPLIT_K

    global _FWD_KERNEL_FP16, _FWD_KERNEL_BF16, _MERGE_KERNEL_PLAIN, _MERGE_KERNEL_GATED
    merge_holder = _MERGE_KERNEL_GATED if has_gate else _MERGE_KERNEL_PLAIN
    if merge_holder is None:
        merge_holder = _build_merge(has_gate)
        if has_gate:
            _MERGE_KERNEL_GATED = merge_holder
        else:
            _MERGE_KERNEL_PLAIN = merge_holder

    if q.dtype == mx.float16:
        tdtype = mx.float16
        fwd_holder = _FWD_KERNEL_FP16
        if fwd_holder is None:
            fwd_holder = _build_fwd(bf16=False)
            _FWD_KERNEL_FP16 = fwd_holder
    elif q.dtype == mx.bfloat16:
        tdtype = mx.bfloat16
        fwd_holder = _FWD_KERNEL_BF16
        if fwd_holder is None:
            fwd_holder = _build_fwd(bf16=True)
            _FWD_KERNEL_BF16 = fwd_holder
    else:
        raise ValueError(f"only fp16/bf16; got {q.dtype}")

    threads_per_tg = 32 * 6
    n_fwd_tg = B * H_kv * n_splits
    fwd_grid_x = n_fwd_tg * threads_per_tg
    scale_arr = mx.array([float(scale)], dtype=mx.float32)

    p_o, p_m, p_l = fwd_holder(  # type: ignore[operator]
        inputs=[q, k, v, scale_arr],
        template=[
            ("T", tdtype), ("HQ", H_q), ("HKV", H_kv), ("TKV", T_kv),
            ("B", B), ("NSPLITS", n_splits), ("SPLITK", SPLIT_K),
        ],
        grid=(fwd_grid_x, 1, 1),
        threadgroup=(threads_per_tg, 1, 1),
        output_shapes=[(B, H_q, n_splits, D), (B, H_q, n_splits), (B, H_q, n_splits)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    n_merge_sg = B * H_q
    merge_grid_x = n_merge_sg * 32
    out_arr = merge_holder(  # type: ignore[operator]
        inputs=[p_o, p_m, p_l, gate if has_gate else q],
        template=[("T", tdtype), ("HQ", H_q), ("B", B), ("NSPLITS", n_splits), ("HAS_GATE", 1 if has_gate else 0)],
        grid=(merge_grid_x, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H_q, 1, D)],
        output_dtypes=[q.dtype],
    )
    return out_arr[0]  # type: ignore[no-any-return]


__all__ = ["flash_attention_decode_v8"]
