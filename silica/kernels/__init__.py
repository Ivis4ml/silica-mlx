"""Custom MLX-native Metal kernels for the Silica hot path.

Per ``AR.md`` Custom kernel authorization (relaxed 2026-05-02 / mandate
2026-05-03), this package hosts hand-rolled Metal kernels via
``mx.fast.metal_kernel``. Each kernel ships with:

- A correctness microbench (max-abs / max-rel error vs an MLX reference
  on at least three input shapes spanning the production decode profile).
- A performance microbench (p50 / p95 latency at production shapes).
- A shadow-mode integration path (kernel callable via env flag, default
  OFF) so the kernel can be A/B-tested without modifying production
  paths.

Hot-path replacement requires explicit user approval.
"""

from silica.kernels.flash_attention_decode import flash_attention_decode
from silica.kernels.flash_attention_decode_v6 import flash_attention_decode_v6
from silica.kernels.flash_attention_decode_v8 import flash_attention_decode_v8
from silica.kernels.flash_attention_decode_v10 import flash_attention_decode_v10
from silica.kernels.fused_gated_output import fused_gated_output
from silica.kernels.fused_qk_norm_rope import fused_qk_norm
from silica.kernels.fused_silu_mul import fused_silu_mul

__all__ = [
    "flash_attention_decode",
    "flash_attention_decode_v6",
    "flash_attention_decode_v8",
    "flash_attention_decode_v10",
    "fused_gated_output",
    "fused_qk_norm",
    "fused_silu_mul",
]
