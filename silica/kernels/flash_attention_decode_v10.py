"""FA-decode v10: v8 with single-pass fast path for short T_kv.

When T_kv ≤ SPLIT_K (no K-split needed), v8 still ran the two-kernel
forward + merge sequence, paying the merge launch + global memory
roundtrip overhead. v10 adds a single-pass kernel that writes directly
to the output (fp16) and applies the optional gate in the epilogue.

For T_kv > SPLIT_K, v10 falls back to v8's two-kernel split path.
"""

from __future__ import annotations

import mlx.core as mx
from silica.kernels.flash_attention_decode_v8 import flash_attention_decode_v8

SPLIT_K = 128


_SINGLE_PASS_SOURCE = """
    constexpr uint TK = 32u;
    constexpr uint D = 256u;
    constexpr uint TPS = 32u;
    constexpr uint D_PER_T = D / TPS;
    constexpr uint D_PER_T_VEC = D_PER_T / 4;
    constexpr uint D_VEC = D / 4;
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
    uint k_pos = 0;
    while (k_pos < T_kv) {
        uint tk_actual = min(TK, T_kv - k_pos);

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

        threadgroup half4* k_tile_vec = reinterpret_cast<threadgroup half4*>(&k_tile[0][0]);
        threadgroup half4* v_tile_vec = reinterpret_cast<threadgroup half4*>(&v_tile[0][0]);

        for (uint k_idx = 0; k_idx < tk_actual; k_idx++) {
            float partial = 0.0f;
            for (uint vi = 0; vi < D_PER_T_VEC; vi++) {
                half4 k_vec4 = k_tile_vec[k_idx * D_VEC + lid * D_PER_T_VEC + vi];
                partial += float(metal::dot(q_local_vec[vi], k_vec4));
            }
            float s_k = simd_sum(partial) * scale;

            float m_new = max(m_cur, s_k);
            float alpha = (m_cur == -INFINITY) ? 0.0f : metal::exp(m_cur - m_new);
            float p_k = metal::exp(s_k - m_new);

            float l_new = l_cur * alpha + p_k;
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

    // Final: O = (1/l) * o; apply gate; write to fp16 output directly.
    float inv_l = 1.0f / l_cur;
    uint o_base = (b * H_q + h_q) * D;
    for (uint vi = 0; vi < D_PER_T_VEC; vi++) {
        float4 out_vec = o_local_vec[vi] * inv_l;
        if (HAS_GATE) {
            uint d_idx_base = lid * D_PER_T + vi * 4;
            for (uint c = 0; c < 4; c++) {
                float g_val = float(gate[q_base + d_idx_base + c]);
                float s_val;
                if (g_val > 16.0f) s_val = 1.0f;
                else if (g_val < -16.0f) s_val = 0.0f;
                else s_val = 1.0f / (1.0f + metal::exp(-g_val));
                out_vec[c] *= s_val;
            }
        }
        // Write as half4 store.
        device half4* out_vec_ptr = reinterpret_cast<device half4*>(out);
        uint out_base_vec = o_base / 4;
        out_vec_ptr[out_base_vec + lid * D_PER_T_VEC + vi] = half4(out_vec);
    }
"""


_SINGLE_PASS_SOURCE_BF16 = (
    _SINGLE_PASS_SOURCE
    .replace("threadgroup half k_tile", "threadgroup bfloat16_t k_tile")
    .replace("threadgroup half v_tile", "threadgroup bfloat16_t v_tile")
    .replace("half4", "bfloat4")
    .replace(
        "partial += float(metal::dot(q_local_vec[vi], k_vec4));",
        "partial += metal::dot(float4(q_local_vec[vi]), float4(k_vec4));",
    )
)


_SINGLE_PASS_KERNEL_PLAIN_FP16: object | None = None
_SINGLE_PASS_KERNEL_GATED_FP16: object | None = None
_SINGLE_PASS_KERNEL_PLAIN_BF16: object | None = None
_SINGLE_PASS_KERNEL_GATED_BF16: object | None = None


def _build_single_pass(has_gate: bool, *, bf16: bool) -> object:
    return mx.fast.metal_kernel(
        name=(
            "silica_flash_attention_decode_v10_single_"
            f"{'gated' if has_gate else 'plain'}_{'bf16' if bf16 else 'fp16'}"
        ),
        input_names=["q", "k", "v", "scale_buf", "gate"],
        output_names=["out"],
        source=_SINGLE_PASS_SOURCE_BF16 if bf16 else _SINGLE_PASS_SOURCE,
        ensure_row_contiguous=True,
    )


def flash_attention_decode_v10(
    q: mx.array, k: mx.array, v: mx.array, *,
    scale: float | None = None, gate: mx.array | None = None,
) -> mx.array:
    """v10: single-pass fast-path for T_kv ≤ SPLIT_K, else v8 fallback."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be 4D")
    B, H_q, T_q, D = q.shape
    Bk, H_kv, T_kv, Dk = k.shape
    if T_q != 1:
        raise NotImplementedError(f"only T_q=1; got {T_q}")
    if D != 256:
        raise NotImplementedError("only head_dim=256")
    if H_q // H_kv != 6:
        raise NotImplementedError("v10 hardcodes Q_PER_KV=6")
    if scale is None:
        scale = D ** -0.5

    # Fall back to v8 (two-kernel K-split) for long contexts.
    if T_kv > SPLIT_K:
        return flash_attention_decode_v8(q, k, v, scale=scale, gate=gate)

    has_gate = gate is not None
    if has_gate and gate.shape != q.shape:
        raise ValueError("gate shape mismatch")

    if q.dtype == mx.float16:
        tdtype = mx.float16
        bf16 = False
    elif q.dtype == mx.bfloat16:
        tdtype = mx.bfloat16
        bf16 = True
    else:
        raise ValueError(f"only fp16/bf16; got {q.dtype}")

    global _SINGLE_PASS_KERNEL_PLAIN_FP16, _SINGLE_PASS_KERNEL_GATED_FP16
    global _SINGLE_PASS_KERNEL_PLAIN_BF16, _SINGLE_PASS_KERNEL_GATED_BF16
    if bf16:
        kernel_holder = (
            _SINGLE_PASS_KERNEL_GATED_BF16 if has_gate else _SINGLE_PASS_KERNEL_PLAIN_BF16
        )
        if kernel_holder is None:
            kernel_holder = _build_single_pass(has_gate, bf16=True)
            if has_gate:
                _SINGLE_PASS_KERNEL_GATED_BF16 = kernel_holder
            else:
                _SINGLE_PASS_KERNEL_PLAIN_BF16 = kernel_holder
    else:
        kernel_holder = (
            _SINGLE_PASS_KERNEL_GATED_FP16 if has_gate else _SINGLE_PASS_KERNEL_PLAIN_FP16
        )
        if kernel_holder is None:
            kernel_holder = _build_single_pass(has_gate, bf16=False)
            if has_gate:
                _SINGLE_PASS_KERNEL_GATED_FP16 = kernel_holder
            else:
                _SINGLE_PASS_KERNEL_PLAIN_FP16 = kernel_holder

    threads_per_tg = 32 * 6
    n_tg = B * H_kv
    grid_x = n_tg * threads_per_tg
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    inputs = [q, k, v, scale_arr, gate if has_gate else q]

    out = kernel_holder(  # type: ignore[operator]
        inputs=inputs,
        template=[
            ("T", tdtype), ("HQ", H_q), ("HKV", H_kv), ("TKV", T_kv),
            ("B", B), ("HAS_GATE", 1 if has_gate else 0),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(threads_per_tg, 1, 1),
        output_shapes=[(B, H_q, 1, D)],
        output_dtypes=[q.dtype],
    )
    return out[0]


__all__ = ["flash_attention_decode_v10"]
