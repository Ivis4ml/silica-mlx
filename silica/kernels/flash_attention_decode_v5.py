"""FA-decode v5: streaming online-softmax (no scores[] register array).

v3/v4 stored scores[TK=32] in per-thread registers, then made a second pass
to update softmax + accumulate V. v5 fuses the score, softmax update, and V
accumulate into a single per-K-row pass — no scores[] array.

Per-thread register footprint: q_local[8] half + m, l float + o_local[8]
float ≈ 56 B (vs ~176 B in v3/v4). Smaller registers may let the compiler
keep more state hot and reduce spills.

Combined with v3's GQA-aware K/V tile sharing (1 TG per (b, h_kv), 6
simdgroups inside).
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_SOURCE = """
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

    uint tid_in_tg = thread_position_in_threadgroup.x;
    uint lid = tid_in_tg % TPS;
    uint sg_in_tg = tid_in_tg / TPS;
    uint tg_idx = threadgroup_position_in_grid.x;
    uint b = tg_idx / H_kv;
    uint h_kv = tg_idx % H_kv;
    uint h_q = h_kv * Q_PER_KV + sg_in_tg;
    if (b >= B_val) return;

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
    uint k_pos = 0;
    while (k_pos < T_kv) {
        uint tk_actual = min(TK, T_kv - k_pos);

        for (uint cell_idx = tid_in_tg; cell_idx < TK * D; cell_idx += THREADS_PER_TG) {
            uint row = cell_idx / D;
            uint col = cell_idx % D;
            if (row < tk_actual) {
                k_tile[row][col] = k[kv_base + (k_pos + row) * D + col];
                v_tile[row][col] = v[kv_base + (k_pos + row) * D + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Streaming softmax: process one K row at a time, no scores[] array.
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
        name=f"silica_flash_attention_decode_v5_{'gated' if has_gate else 'plain'}",
        input_names=["q", "k", "v", "scale_buf", "gate"],
        output_names=["out"],
        source=_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def flash_attention_decode_v5(
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
        raise NotImplementedError(f"v5 hardcodes Q_PER_KV=6")
    if scale is None:
        scale = D ** -0.5

    has_gate = gate is not None
    if has_gate and gate.shape != q.shape:
        raise ValueError("gate shape mismatch")

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
    threads_per_tg = 32 * 6
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


__all__ = ["flash_attention_decode_v5"]
