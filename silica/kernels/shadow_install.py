"""Shadow-mode kernel installer for hybrid Qwen3.5 adapters.

Installs the two production-grade levers from the P-6 autoresearch loop
behind opt-in environment flags, default OFF. Default install is a
no-op; flags must be explicitly set to opt in.

Env vars (each independently OFF by default):

    SILICA_USE_FA_DECODE_V10=1
        Replace ``Qwen3NextAttention.__call__`` with a wrapper that
        routes single-token decode (``T_q=1``, ``head_dim=256``,
        ``GQA q_per_kv=6``, no mask, fp16/bf16 queries) through the
        fused FlashAttention-decode v10 kernel. Other shapes (prefill,
        masks, dtype mismatches) fall back to the vanilla
        ``scaled_dot_product_attention`` + ``mx.sigmoid(gate)`` path.
        v10 dispatches internally to v8 for ``T_kv > 128``.

    SILICA_USE_BF16_DELTANET_STATE=1
        Allocate the DeltaNet recurrent state as ``bf16`` instead of
        ``fp32``. State shape ``[B, Hv=48, Dv=128, Dk=128]`` is 144 MB
        at fp32 per layer, 72 MB at bf16; over 48 DeltaNet layers the
        peak-memory save is ~3.5 GB, which is the load-bearing lever
        for the cycle-13 (1b) clear at B=52 within the 36 GB envelope.

Per the v1.7.23 P-6 strategic re-anchor, only these two flags are
exposed; retired exploration flags (fused gated output, fused SwiGLU,
fused QK norm, mx.compile MLP) are not present here. The fallback path
inside the v10 wrapper uses vanilla ``mx.sigmoid``, with no
``fused_*`` references.

The swap is idempotent: re-calling ``install(model)`` after the first
call returns the same dict without re-patching. ``restore()`` reverts
the patches and clears the installed flag so a future ``install`` call
will re-apply them.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from silica.kernels.flash_attention_decode_v10 import flash_attention_decode_v10


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


_ORIGINAL_QWEN_ATTN_CALL: Any = None
_ORIGINAL_GATED_DELTA_UPDATE: Any = None
_INSTALLED = False


def install(model: Any) -> dict[str, bool]:
    """Install kernel-backed shadow ops based on env flags. Idempotent.

    Returns a dict naming which flags were active on this install. After
    the first install the dict reflects the original install's flag set;
    repeat calls do not re-read the environment.
    """
    global _ORIGINAL_QWEN_ATTN_CALL, _ORIGINAL_GATED_DELTA_UPDATE, _INSTALLED
    installed: dict[str, bool] = {
        "fa_decode_v10": False,
        "bf16_deltanet_state": False,
    }

    if _INSTALLED:
        return installed

    try:
        from mlx_lm.models import qwen3_next as qn
    except ImportError:
        return installed

    if _env_on("SILICA_USE_FA_DECODE_V10"):
        _ORIGINAL_QWEN_ATTN_CALL = qn.Qwen3NextAttention.__call__

        def _patched_attn_call(self, x, mask=None, cache=None):  # type: ignore[no-untyped-def]  # noqa: ANN001
            B, L, _ = x.shape

            q_proj_output = self.q_proj(x)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, self.num_attention_heads, -1),
                2,
                axis=-1,
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

            T_q = queries.shape[2]
            head_dim = queries.shape[3]
            num_q_heads = queries.shape[1]
            num_kv_heads = keys.shape[1]
            v10_eligible = (
                T_q == 1
                and head_dim == 256
                and num_q_heads == 24
                and num_kv_heads == 4
                and mask is None
                and queries.dtype in (mx.float16, mx.bfloat16)
            )
            if v10_eligible:
                gate_4d = gate.transpose(0, 2, 1, 3)
                output_4d = flash_attention_decode_v10(
                    queries, keys, values, scale=self.scale, gate=gate_4d
                )
                output = output_4d.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)

            from mlx_lm.models.base import scaled_dot_product_attention

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            gated = output * mx.sigmoid(gate_flat)
            return self.o_proj(gated)

        qn.Qwen3NextAttention.__call__ = _patched_attn_call  # type: ignore[method-assign]
        installed["fa_decode_v10"] = True

    if _env_on("SILICA_USE_BF16_DELTANET_STATE"):
        from mlx_lm.models import gated_delta as gd

        _ORIGINAL_GATED_DELTA_UPDATE = gd.gated_delta_update

        def _patched_gated_delta_update(  # type: ignore[no-untyped-def]
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state=None,
            mask=None,
            use_kernel=True,
        ):
            if state is None:
                B_dim, _, _, _ = q.shape
                Hv, Dv = v.shape[-2:]
                Dk = q.shape[-1]
                state = mx.zeros((B_dim, Hv, Dv, Dk), dtype=mx.bfloat16)
            return _ORIGINAL_GATED_DELTA_UPDATE(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state=state,
                mask=mask,
                use_kernel=use_kernel,
            )

        gd.gated_delta_update = _patched_gated_delta_update
        # Patch the imported reference inside qwen3_next as well so the
        # model's attribute lookup reaches the bf16 path.
        qn.gated_delta_update = _patched_gated_delta_update
        installed["bf16_deltanet_state"] = True

    _INSTALLED = True
    return installed


def restore() -> None:
    """Restore swapped-in modules to their pre-install state."""
    global _ORIGINAL_QWEN_ATTN_CALL, _ORIGINAL_GATED_DELTA_UPDATE, _INSTALLED
    if not _INSTALLED:
        return
    try:
        from mlx_lm.models import qwen3_next as qn

        if _ORIGINAL_QWEN_ATTN_CALL is not None:
            qn.Qwen3NextAttention.__call__ = _ORIGINAL_QWEN_ATTN_CALL  # type: ignore[method-assign]
        if _ORIGINAL_GATED_DELTA_UPDATE is not None:
            from mlx_lm.models import gated_delta as gd

            gd.gated_delta_update = _ORIGINAL_GATED_DELTA_UPDATE
            qn.gated_delta_update = _ORIGINAL_GATED_DELTA_UPDATE
    except ImportError:
        pass
    _ORIGINAL_QWEN_ATTN_CALL = None
    _ORIGINAL_GATED_DELTA_UPDATE = None
    _INSTALLED = False


__all__ = ["install", "restore"]
