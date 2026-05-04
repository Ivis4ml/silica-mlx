"""Cycle 31 — vectorized silica DeltaNet recurrent-step kernel.

mlx_lm.models.gated_delta._make_gated_delta_kernel uses scalar half/bf16
reads in the inner state-update + score-compute loops. At Qwen3.5-27B-4bit
B=64 production decode, DeltaNet is 87.9% of step time (cycle 30
decomposition). This kernel applies the v6→v7 half4 trick from
cycle 11 FA-decode to the recurrent path:

- ``bfloat4`` (or ``half4``) vectorized loads of q, k, v
- ``float4`` vectorized state R/W
- Same simdgroup reduction structure as the mlx kernel

Target: ≥1.5× speedup over mlx's gated_delta kernel at production
shape (B=52/64, T=1, Hk=16, Hv=48, Dk=Dv=128).

Algorithm (mirrors mlx exactly; only memory access pattern differs):

For each (b, hv, dv) triple → one simdgroup of 32 threads:
  Load state[n_per_t=4] (= Dk/32) from HBM into per-thread registers.
  For each time step t:
    g_decay = g[b, t, hv]   (or g[b, t, hv, s_idx] in vectorized variant)
    For i in 0..n_per_t-1:
      state[i] *= g_decay
      kv_mem += state[i] * k_[s_idx]
    kv_mem = simd_sum(kv_mem)        # collapse across 32 threads
    delta = (v_[dv] - kv_mem) * beta_[hv]
    For i in 0..n_per_t-1:
      state[i] += k_[s_idx] * delta
      out += state[i] * q_[s_idx]
    out = simd_sum(out)
    if thread_idx_in_simdgroup == 0: y[dv] = out
  Store state[n_per_t] back to HBM.

Vectorization: each thread's n_per_t=4 reads of (q, k, state) become a
single 4-element vector load. Reads from HBM coalesce better.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def _compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def _make_kernel(*, has_mask: bool = False, vectorized_g: bool = False, bf16: bool = True) -> object:
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    if vectorized_g:
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    # Vector type alias depends on dtype: bfloat4 for bf16 path, half4 for fp16.
    # InT is always 4-element vector aligned because n_per_t = Dk/32 = 4 for Dk=128.
    in_vec_t = "bfloat4" if bf16 else "half4"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;       // = 4 for Dk=128

        // q, k: [B, T, Hk, Dk]. v, y: [B, T, Hv, Dv]. State: [B, Hv, Dv, Dk].
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        // Vectorized state load: read n_per_t=4 floats as a single float4
        // (state is fp32 internally; storage type StT may be bf16/fp16/fp32).
        // For mixed-precision storage we still load element-wise then promote
        // to float4, since StT may be a 2-byte type that doesn't have a
        // 4-element vector counterpart everywhere on Apple Silicon.
        float4 state_v;
        {{
            auto s_base = n_per_t * dk_idx;
            state_v.x = static_cast<float>(i_state[s_base + 0]);
            state_v.y = static_cast<float>(i_state[s_base + 1]);
            state_v.z = static_cast<float>(i_state[s_base + 2]);
            state_v.w = static_cast<float>(i_state[s_base + 3]);
        }}

        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
            if ({mask_source}) {{
                // Vectorized k load (4 contiguous elements per thread)
                auto s_base = n_per_t * dk_idx;
                {in_vec_t} k_v;
                k_v.x = k_[s_base + 0];
                k_v.y = k_[s_base + 1];
                k_v.z = k_[s_base + 2];
                k_v.w = k_[s_base + 3];

                // Decay state by g; accumulate kv_mem partial.
                float kv_mem = 0.0f;
                {{
                    float g_v0 = float({g_access.replace("[s_idx]", "[s_base + 0]") if vectorized_g else g_access});
                    float g_v1 = float({g_access.replace("[s_idx]", "[s_base + 1]") if vectorized_g else g_access});
                    float g_v2 = float({g_access.replace("[s_idx]", "[s_base + 2]") if vectorized_g else g_access});
                    float g_v3 = float({g_access.replace("[s_idx]", "[s_base + 3]") if vectorized_g else g_access});
                    state_v.x *= g_v0; kv_mem += state_v.x * float(k_v.x);
                    state_v.y *= g_v1; kv_mem += state_v.y * float(k_v.y);
                    state_v.z *= g_v2; kv_mem += state_v.z * float(k_v.z);
                    state_v.w *= g_v3; kv_mem += state_v.w * float(k_v.w);
                }}
                kv_mem = simd_sum(kv_mem);

                float delta = (float(v_[dv_idx]) - kv_mem) * float(beta_[hv_idx]);

                // Vectorized q load
                {in_vec_t} q_v;
                q_v.x = q_[s_base + 0];
                q_v.y = q_[s_base + 1];
                q_v.z = q_[s_base + 2];
                q_v.w = q_[s_base + 3];

                // state += k * delta; out += state * q
                float out = 0.0f;
                state_v.x += float(k_v.x) * delta; out += state_v.x * float(q_v.x);
                state_v.y += float(k_v.y) * delta; out += state_v.y * float(q_v.y);
                state_v.z += float(k_v.z) * delta; out += state_v.z * float(q_v.z);
                state_v.w += float(k_v.w) * delta; out += state_v.w * float(q_v.w);
                out = simd_sum(out);

                if (thread_index_in_simdgroup == 0) {{
                    y[dv_idx] = static_cast<InT>(out);
                }}
            }} else {{
                y[dv_idx] = static_cast<InT>(0);
            }}
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            {g_advance}
            beta_ += Hv;
        }}

        // Vectorized state store
        {{
            auto s_base = n_per_t * dk_idx;
            o_state[s_base + 0] = static_cast<StT>(state_v.x);
            o_state[s_base + 1] = static_cast<StT>(state_v.y);
            o_state[s_base + 2] = static_cast<StT>(state_v.z);
            o_state[s_base + 3] = static_cast<StT>(state_v.w);
        }}
    """

    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized_g:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"
    suffix += "_bf16" if bf16 else "_fp16"

    return mx.fast.metal_kernel(
        name=f"silica_gated_delta_v2_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_KERNEL_BF16: object | None = None
_KERNEL_FP16: object | None = None
_KERNEL_VEC_BF16: object | None = None
_KERNEL_VEC_FP16: object | None = None


def _get_kernel(*, vectorized_g: bool, bf16: bool, has_mask: bool = False) -> object:
    global _KERNEL_BF16, _KERNEL_FP16, _KERNEL_VEC_BF16, _KERNEL_VEC_FP16
    # Build on first use; ignore mask for now (no current shadow path uses it).
    if has_mask:
        # Build per-call when mask path is needed
        return _make_kernel(has_mask=True, vectorized_g=vectorized_g, bf16=bf16)
    if vectorized_g and bf16:
        if _KERNEL_VEC_BF16 is None:
            _KERNEL_VEC_BF16 = _make_kernel(vectorized_g=True, bf16=True)
        return _KERNEL_VEC_BF16
    if vectorized_g and not bf16:
        if _KERNEL_VEC_FP16 is None:
            _KERNEL_VEC_FP16 = _make_kernel(vectorized_g=True, bf16=False)
        return _KERNEL_VEC_FP16
    if not vectorized_g and bf16:
        if _KERNEL_BF16 is None:
            _KERNEL_BF16 = _make_kernel(vectorized_g=False, bf16=True)
        return _KERNEL_BF16
    if _KERNEL_FP16 is None:
        _KERNEL_FP16 = _make_kernel(vectorized_g=False, bf16=False)
    return _KERNEL_FP16


def gated_delta_kernel_v2(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0:
        raise NotImplementedError("v2 requires Dk % 32 == 0 (= 4-element vectorization aligned)")
    if Dk // 32 != 4:
        raise NotImplementedError(f"v2 hardcodes n_per_t=4 (Dk=128); got Dk={Dk}")

    bf16 = q.dtype == mx.bfloat16
    vectorized_g = g.ndim == 4

    kernel = _get_kernel(vectorized_g=vectorized_g, bf16=bf16, has_mask=mask is not None)

    inputs = [q, k, v, g, beta, state, T]
    if mask is not None:
        inputs.append(mask)

    return kernel(  # type: ignore[operator]
        inputs=inputs,
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


__all__ = ["gated_delta_kernel_v2"]
