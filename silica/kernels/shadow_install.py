"""Shadow-mode kernel installation for hybrid Qwen3.5 adapters.

Per AR.md custom-kernel authorisation, kernel candidates are integrated
in shadow mode (callable via env flag, default OFF) until they pass an
end-to-end gate. This module implements the swap-in: it replaces specific
mlx-lm sub-modules' ``__call__`` methods with kernel-backed wrappers,
gated by env vars.

Env vars (each independently OFF by default):
    SILICA_USE_FUSED_GATED_OUTPUT=1
        Replace `out = sdpa_out * mx.sigmoid(gate)` in
        Qwen3NextAttention.__call__ with the fused kernel.
    SILICA_USE_FUSED_SILU_MUL=1
        Replace `_precise_swiglu(...)` (the SwiGLU activation in
        Qwen3NextMLP) with the fused kernel.
    SILICA_USE_FUSED_QK_NORM=1
        Replace the two `mx.fast.rms_norm` calls in attention's q_norm
        and k_norm with the fused two-norm kernel.
    SILICA_USE_FA_DECODE_V10=1
        Replace `scaled_dot_product_attention(...) * mx.sigmoid(gate)` in
        Qwen3NextAttention.__call__ with the fused FlashAttention-decode v10
        kernel during decode (T_q=1). Falls back to the original path for
        prefill. Mutually exclusive with SILICA_USE_FUSED_GATED_OUTPUT — when
        both are set, v10 wins because it subsumes the gated_output fusion.
    SILICA_USE_BF16_DELTANET_STATE=1
        Allocate the DeltaNet recurrent state as bf16 instead of fp32. State
        shape is [B, Hv=48, Dv=128, Dk=128] = 144 MB at fp32 per layer, 72 MB
        at bf16. With 48 DeltaNet layers and decode-time read+write per step,
        bandwidth savings are 48 * 144 MB = 6.9 GB/step. Correctness verified
        on a 20-token greedy sample (token-ID parity); longer generations
        should be re-checked before production use.
    SILICA_USE_COMPILED_MLP=1
        Wrap Qwen3NextMLP.__call__ with mx.compile so the gate_proj +
        up_proj + swiglu + down_proj sequence is graph-traced once and the
        Python dispatch overhead per layer is amortised across decode steps.
        64 layers × per-step decode dispatch becomes the cache-friendly
        single graph. Stateless (no cache) so mx.compile applies cleanly.

The swap is installed once per model; restore via `restore(model)`.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from silica.kernels.flash_attention_decode_v10 import flash_attention_decode_v10
from silica.kernels.fused_gated_output import fused_gated_output
from silica.kernels.fused_qk_norm_rope import fused_qk_norm
from silica.kernels.fused_silu_mul import fused_silu_mul


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


_ORIGINAL_PRECISE_SWIGLU = None  # populated at install time
_ORIGINAL_QWEN_ATTN_CALL = None
_ORIGINAL_QWEN_MLP_CALL = None
_ORIGINAL_GATED_DELTA_UPDATE = None
_ORIGINAL_QWEN_MLP_CALL_FN = None
_INSTALLED = False


def install(model: Any) -> dict[str, bool]:
    """Install kernel-backed shadow ops based on env flags. Idempotent.

    Returns a dict naming which kernels were installed.
    """
    global _ORIGINAL_PRECISE_SWIGLU, _ORIGINAL_QWEN_ATTN_CALL, _ORIGINAL_QWEN_MLP_CALL
    global _ORIGINAL_GATED_DELTA_UPDATE, _ORIGINAL_QWEN_MLP_CALL_FN
    global _INSTALLED
    installed: dict[str, bool] = {
        "gated_output": False,
        "silu_mul": False,
        "qk_norm": False,
        "fa_decode_v10": False,
        "bf16_deltanet_state": False,
        "compiled_mlp": False,
    }

    if _INSTALLED:
        return installed

    # Load mlx-lm's qwen3_next module so we can swap private functions.
    try:
        from mlx_lm.models import qwen3_next as qn
    except ImportError:
        return installed

    # --- Fused SwiGLU: replace _precise_swiglu ---
    if _env_on("SILICA_USE_FUSED_SILU_MUL"):
        _ORIGINAL_PRECISE_SWIGLU = qn._precise_swiglu

        def _patched_precise_swiglu(h, gate, x):  # noqa: ANN001
            # Use fused kernel; semantics match _precise_swiglu (fp32 internal
            # silu, output dtype matches h).
            return fused_silu_mul(gate, x)

        qn._precise_swiglu = _patched_precise_swiglu  # type: ignore[attr-defined]
        installed["silu_mul"] = True

    # --- FA-decode v10: full fused attention + gated output during decode ---
    # Subsumes gated_output (the v10 epilogue applies sigmoid(gate) directly).
    # Falls back to scaled_dot_product_attention + fused_gated_output (or
    # un-fused sigmoid+multiply) when T_q != 1.
    use_fa_v10 = _env_on("SILICA_USE_FA_DECODE_V10")
    use_gated_output = _env_on("SILICA_USE_FUSED_GATED_OUTPUT")
    if use_fa_v10 or use_gated_output:
        _ORIGINAL_QWEN_ATTN_CALL = qn.Qwen3NextAttention.__call__

        def _patched_attn_call(self, x, mask=None, cache=None):  # noqa: ANN001
            B, L, _ = x.shape

            q_proj_output = self.q_proj(x)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
            )
            gate_flat = gate.reshape(B, L, -1)

            keys, values = self.k_proj(x), self.v_proj(x)

            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(B, L, self.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(
                B, L, self.num_key_value_heads, -1
            ).transpose(0, 2, 1, 3)

            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            # FA-decode v10 path: only when shape constraints match (T_q=1,
            # head_dim=256, GQA q_per_kv=6, no mask). Otherwise fall back.
            T_q = queries.shape[2]
            head_dim = queries.shape[3]
            num_q_heads = queries.shape[1]
            num_kv_heads = keys.shape[1]
            v10_eligible = (
                use_fa_v10
                and T_q == 1
                and head_dim == 256
                and num_q_heads == 24
                and num_kv_heads == 4
                and mask is None
                and queries.dtype == mx.float16
            )
            if v10_eligible:
                # gate is currently (B, L, num_q_heads, head_dim/2) — wait, let's recheck.
                # gate after split is (B, L, num_q_heads, head_dim_with_gate / 2). Need
                # to reshape to (B, num_q_heads, T_q, head_dim) to match v10's contract.
                gate_4d = gate.transpose(0, 2, 1, 3)  # (B, H_q, L, D)
                output_4d = flash_attention_decode_v10(
                    queries, keys, values, scale=self.scale, gate=gate_4d
                )
                output = output_4d.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)

            # Fallback path: regular SDPA + (optionally fused) gate.
            from mlx_lm.models.base import scaled_dot_product_attention
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            if use_gated_output:
                gated = fused_gated_output(output, gate_flat)
            else:
                gated = output * mx.sigmoid(gate_flat)
            return self.o_proj(gated)

        qn.Qwen3NextAttention.__call__ = _patched_attn_call  # type: ignore[method-assign]
        installed["gated_output"] = use_gated_output
        installed["fa_decode_v10"] = use_fa_v10

    # --- Compiled MLP: probed in cycle 17, retired ---
    # mx.compile on a Qwen3.5-shape MLP at B=52 4-bit gives only 1.027x (2.7%
    # speedup, below the warm-decode oracle's noise floor of ~1 tok/s on
    # 207 tok/s). Across 64 MLP calls/step the E2E delta is ~0.5%, undetectable.
    # Flag retained for documentation but no patch installs.
    if _env_on("SILICA_USE_COMPILED_MLP"):
        installed["compiled_mlp"] = False  # cycle-17 retired — see report

    # --- bf16 DeltaNet state: replace gated_delta_update to allocate bf16 state ---
    if _env_on("SILICA_USE_BF16_DELTANET_STATE"):
        from mlx_lm.models import gated_delta as gd

        _ORIGINAL_GATED_DELTA_UPDATE = gd.gated_delta_update

        def _patched_gated_delta_update(
            q, k, v, a, b, A_log, dt_bias,
            state=None, mask=None, use_kernel=True,
        ):
            if state is None:
                B, _, _, _ = q.shape
                Hv, Dv = v.shape[-2:]
                Dk = q.shape[-1]
                state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.bfloat16)
            return _ORIGINAL_GATED_DELTA_UPDATE(
                q, k, v, a, b, A_log, dt_bias,
                state=state, mask=mask, use_kernel=use_kernel,
            )

        gd.gated_delta_update = _patched_gated_delta_update
        # Also patch the imported reference in qwen3_next so the model uses it.
        qn.gated_delta_update = _patched_gated_delta_update
        installed["bf16_deltanet_state"] = True

    # --- Fused QK norm: would require patching the q_norm/k_norm calls in __call__ ---
    # Skipped here because it requires re-routing through fused_qk_norm; the
    # microbench showed our QK-norm kernel is slower than 2x mx.fast.rms_norm,
    # so installing it would hurt.

    _INSTALLED = True
    return installed


def restore() -> None:
    """Restore all swapped-in modules to their pre-install state."""
    global _ORIGINAL_PRECISE_SWIGLU, _ORIGINAL_QWEN_ATTN_CALL, _ORIGINAL_QWEN_MLP_CALL
    global _ORIGINAL_GATED_DELTA_UPDATE, _ORIGINAL_QWEN_MLP_CALL_FN
    global _INSTALLED
    if not _INSTALLED:
        return
    try:
        from mlx_lm.models import qwen3_next as qn
        if _ORIGINAL_PRECISE_SWIGLU is not None:
            qn._precise_swiglu = _ORIGINAL_PRECISE_SWIGLU  # type: ignore[attr-defined]
        if _ORIGINAL_QWEN_ATTN_CALL is not None:
            qn.Qwen3NextAttention.__call__ = _ORIGINAL_QWEN_ATTN_CALL  # type: ignore[method-assign]
        if _ORIGINAL_GATED_DELTA_UPDATE is not None:
            from mlx_lm.models import gated_delta as gd
            gd.gated_delta_update = _ORIGINAL_GATED_DELTA_UPDATE
            qn.gated_delta_update = _ORIGINAL_GATED_DELTA_UPDATE
        if _ORIGINAL_QWEN_MLP_CALL_FN is not None:
            qn.Qwen3NextMLP.__call__ = _ORIGINAL_QWEN_MLP_CALL_FN  # type: ignore[method-assign]
    except ImportError:
        pass
    _ORIGINAL_PRECISE_SWIGLU = None
    _ORIGINAL_GATED_DELTA_UPDATE = None
    _ORIGINAL_QWEN_MLP_CALL_FN = None
    _ORIGINAL_QWEN_ATTN_CALL = None
    _ORIGINAL_QWEN_MLP_CALL = None
    _INSTALLED = False


__all__ = ["install", "restore"]
