"""Custom MLX-native Metal kernels for the Silica hot path.

Per P6_AUTORESEARCH.md Custom kernel authorization (relaxed 2026-05-02 / mandate
2026-05-03), this package hosts hand-rolled Metal kernels via
``mx.fast.metal_kernel``. The public surface is intentionally narrow
after the v1.7.23 P-6 strategic re-anchor: the running-best dense-27B
decode (204 tok/s @ B=52 within 36 GB envelope; 232 tok/s @ B=64 at the
48 GB hardware ceiling) is attributed to the C10 axis-shift x C12 bf16
DeltaNet recurrent state composition, not to a custom attention kernel
(see ``plans/P6_AUTORESEARCH_NOTES.md`` and the v1.7.23 changelog in
``plans/PLAN.md``).

Public surface:

- ``flash_attention_decode_v10`` — fused gated FlashAttention decode
  kernel. Microbench-winning (1.28-2.14x over
  ``mx.fast.scaled_dot_product_attention`` on bf16 at the production
  shape), but cycle-27 honest reattribution measured the E2E
  contribution at +0.5 tok/s @ B=52 / -1.7 tok/s @ B=64, both within
  noise. Retained as a documentation-grade probe for future stack
  re-evaluation. Internally dispatches to ``flash_attention_decode_v8``
  for ``T_kv > 128`` (long-context K-split path); v8 is therefore part
  of the package as v10's runtime dependency, not as a public
  shadow-install flag.
- ``shadow_install`` — env-flag-gated installer for v10 + bf16 DeltaNet
  recurrent state. Default OFF; safe to import without side effects.

Hot-path replacement requires explicit user approval.
"""

from silica.kernels import shadow_install
from silica.kernels.flash_attention_decode_v10 import flash_attention_decode_v10

__all__ = ["flash_attention_decode_v10", "shadow_install"]
