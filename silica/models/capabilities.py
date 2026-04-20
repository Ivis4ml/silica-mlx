"""silica.models.capabilities — typed summary of adapter capabilities (D-016).

``ModelCapabilities`` is a read-only summary that each ``I-1 ModelAdapter``
returns alongside the existing ``attention_pattern()``. Consumers — the
scheduler's capability gate today, an MoE-aware budgeter and a P-4 bench
harness tomorrow — read this typed object instead of re-interpreting
``AttentionPattern`` on the fly.

``AttentionPattern`` remains the **authoritative** per-layer routing source
(D-015). ``ModelCapabilities`` is a strictly coarser summary: which
``AttentionKind`` values appear across all layers, whether recurrent state
is involved, whether MoE routing is present. Adapters are not expected to
carry two independent truths — the canonical construction path is
``capabilities_from_attention_pattern``; adapters that also have MoE
routing override ``has_moe=True`` at the call site.

Scope (v0.1 / P-3 opening):
  - ``attention_kinds``       set of AttentionKind values seen in the model.
  - ``has_recurrent_state``   True iff any RECURRENT / HYBRID_DELTANET layer.
  - ``has_moe``               True iff adapter routes tokens through MoE.

Additional capability fields (e.g. ``supports_prefix_cache``,
``kv_codec_compatible``, ``activated_params_per_token``) are deliberately
**not** added speculatively — they land when concrete consumers in P-5 /
P-6 / P-7 actually require them.
"""

from __future__ import annotations

from dataclasses import dataclass

from silica.models.adapter import AttentionKind, AttentionPattern

_RECURRENT_KINDS: frozenset[AttentionKind] = frozenset(
    {AttentionKind.RECURRENT, AttentionKind.HYBRID_DELTANET}
)


@dataclass(frozen=True)
class ModelCapabilities:
    """Typed summary of what an adapter needs from the scheduler (D-016).

    Frozen and hashable: callers can cache derived decisions keyed on the
    capabilities tuple if that ever becomes useful.
    """

    attention_kinds: frozenset[AttentionKind]
    has_recurrent_state: bool
    has_moe: bool


def capabilities_from_attention_pattern(
    pattern: AttentionPattern,
    *,
    has_moe: bool = False,
) -> ModelCapabilities:
    """Derive ``ModelCapabilities`` from an ``AttentionPattern``.

    Adapters that declare no MoE routing (the dense Qwen3 / Qwen3.5 cases)
    pass ``has_moe=False`` (the default). MoE adapters set it at the call
    site; ``AttentionPattern`` has no MoE signal, hence the separate
    keyword.

    This is the intended construction path. Protocol default
    implementations do not reach concrete adapters because I-1 is
    structurally typed — each adapter calls this helper explicitly from
    its own ``capabilities()``.
    """
    kinds = frozenset(pattern.per_layer)
    has_recurrent = bool(kinds & _RECURRENT_KINDS)
    return ModelCapabilities(
        attention_kinds=kinds,
        has_recurrent_state=has_recurrent,
        has_moe=has_moe,
    )


__all__ = ["ModelCapabilities", "capabilities_from_attention_pattern"]
