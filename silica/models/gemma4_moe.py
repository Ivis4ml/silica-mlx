"""silica.models.gemma4_moe — I-1 adapter for Gemma4-MoE (P-3-E1.2).

**Scope: Gemma4-MoE sparse variants (e.g. Gemma4-26B-A4B).** The
adapter wraps mlx-lm's already-constructed Gemma4-MoE graph. Unlike
Qwen3.5-MoE (which lives in its own ``qwen3_5_moe.py`` module),
Gemma4's MoE variant shares the ``gemma4_text.py`` source file with
the dense Gemma4-31B path, gated by ``enable_moe_block``. Likewise
both variants report ``model_type="gemma4"`` at the outer wrapper,
so factory dispatch on ``model_type`` alone is ambiguous — the
factory's ``_build_gemma4`` branch reads ``enable_moe_block`` from
``args.text_config`` to route dense vs MoE.

Key structural properties (read in E0, verified on the 26B-A4B-4bit
checkpoint probe 2026-04-20):

- Every decoder layer constructs ``self.mlp = MLP(config,
  layer_idx)`` unconditionally. When ``enable_moe_block=True`` the
  forward runs BOTH the dense MLP branch AND the top-k experts
  branch (via ``Router`` + ``Experts``), then sums ``h = h1 + h2``
  ungated. Three extra layernorms are added in MoE mode
  (``post_feedforward_layernorm_1``, ``pre_feedforward_layernorm_2``,
  ``post_feedforward_layernorm_2``).
- The MoE FFN path goes through ``layer.experts.switch_glu``
  (``Experts`` wraps ``SwitchGLU`` with ``GeGLU`` activation).
  Contrast with Qwen3.5-MoE's ``layer.mlp.switch_mlp`` path.
- Layer attention kinds follow the same sliding+full hybrid as
  Gemma4-31B dense (D-track). On 26B-A4B: 25 sliding + 5 full over
  30 layers, 5:1 ratio.
- No shared-expert branch (that is Qwen3-Next-specific).

This adapter does **not** reimplement any MoE math. All routing,
``SwitchGLU``, always-on dense MLP summation, layernorms, and 4-bit
``QuantizedSwitchLinear`` stay inside mlx-lm. Silica contributes:

  - Capability declaration (``has_moe=True``;
    ``has_recurrent_state=False``; attention kinds inherit the
    ``{SLIDING, GLOBAL}`` set from the dense parent).
  - MoE-aware variant guard that REQUIRES ``enable_moe_block=True``
    and ``num_experts > 0``, then validates ``top_k_experts``, and
    keeps the dense parent's ``hidden_size_per_layer_input`` and
    ``num_kv_shared_layers`` rejections (orthogonal to MoE).
  - ``config.extra`` MoE metadata, including the distinguishing
    ``has_always_on_dense_mlp=True`` fact and the Gemma4-specific
    ``moe_expert_path="layer.experts.switch_glu"`` pointer.
  - Option (c) dispatch-observation seam via
    ``install_dispatch_proxy`` — walks ``layer.experts.switch_glu``
    (not the dense ``layer.mlp``) and wraps each with the shared
    ``_DispatchProxy`` imported from ``qwen3_5_moe``.

Batched execution opens at E4 (smoke-only, parity deferred): the
``ContinuousBatcher._enforce_capability_gate`` ``has_moe=True``
rejection is lifted. mlx-lm's ``SwitchGLU`` + ``gather_mm`` path
is B-agnostic per the P3_MOE_SURVEY §5 E4 audit — a batched
forward dispatches per-row top-k experts without further
scheduler work. Real-model B=2 coverage:
``tests/test_p3_gemma4_moe_batched_smoke.py``. The dense
``Gemma4Adapter`` guard is intentionally unchanged — constructing
the dense adapter directly on a MoE checkpoint must still loud-fail
(factory routing is the normal path; the dense guard is the
defence-in-depth).
"""

from __future__ import annotations

from typing import Any

from silica.kvcache.simple import SimpleKVCache
from silica.models.adapter import ModelConfig
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)
from silica.models.gemma4 import Gemma4Adapter
from silica.models.qwen3_5_moe import DispatchObserver, _DispatchProxy


class Gemma4MoeAdapter(Gemma4Adapter):
    """I-1 ModelAdapter for Gemma4-MoE (26B-A4B class).

    Inherits attention pattern, KV layout (including D4's per-kind
    ``bytes_per_token_total``), ``make_batch_cache`` (sliding ->
    ``BatchRotatingKVCache``, global -> ``BatchKVCache``),
    ``prefill`` / ``decode_step`` from ``Gemma4Adapter`` unchanged.
    Only the capability declaration, the MoE-aware variant guard
    (polymorphic override of the dense guard, reached from the
    parent's ``__init__`` via ``type(self)`` lookup), the config.extra
    augmentation, and ``install_dispatch_proxy`` are new.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        kv_manager: SimpleKVCache,
    ) -> None:
        # Gemma4Adapter.__init__ calls self._validate_supported_variant
        # first. Because Python resolves method lookup via type(self),
        # the polymorphic override below runs instead of the dense
        # parent's guard — even for @staticmethod. That is what lets
        # this subclass accept MoE checkpoints without touching the
        # parent's guard code.
        super().__init__(model, tokenizer, kv_manager=kv_manager)
        self.config = self._augment_config_with_moe_metadata(
            self.config, model
        )

    def capabilities(self) -> ModelCapabilities:
        # Override Gemma4Adapter.capabilities to flip has_moe=True.
        # attention_kinds and has_recurrent_state come from the
        # inherited {SLIDING, GLOBAL} pattern — MoE does not touch
        # attention. has_recurrent_state is False (same as dense).
        return capabilities_from_attention_pattern(
            self._attention_pattern, has_moe=True
        )

    def install_dispatch_proxy(self, observer: DispatchObserver) -> int:
        """Wrap each MoE layer's ``experts.switch_glu`` with a
        dispatch-observing proxy (option (c) seam, E-open-1 resolution).

        Gemma4-MoE's expert dispatch lives at
        ``layer.experts.switch_glu`` — NOT ``layer.mlp.switch_mlp``
        (which is Qwen3.5-MoE's path). The dense ``layer.mlp`` branch
        stays untouched because it is not the MoE dispatch point;
        it is the always-on dense MLP that Gemma4-MoE sums with the
        experts output.

        Contract matches the Qwen3.5-MoE sibling:
          - Observer is called on every sparse-MLP forward with
            ``(layer_idx, indices)`` before delegation.
          - Layers whose ``experts`` has no ``switch_glu`` (non-MoE,
            shared-expert-only) are skipped silently. On Gemma4-MoE
            every decoder layer has ``experts`` because
            ``enable_moe_block`` is a single bool on the config (not
            per-layer); so the skip path only fires on synthetic test
            fixtures.
          - Idempotent: re-installing over an already-wrapped layer
            is a no-op.
          - NOT invoked by ``build``: default single-request path
            through the dense ``ResidentWeightProvider`` (whose
            ``get_expert`` raises by design) continues to work.

        Returns the number of layers wrapped.
        """
        wrapped = 0
        for layer_idx, layer in enumerate(self._model.layers):
            experts = getattr(layer, "experts", None)
            if experts is None:
                continue
            switch_glu = getattr(experts, "switch_glu", None)
            if switch_glu is None:
                continue
            if isinstance(switch_glu, _DispatchProxy):
                continue  # idempotent
            experts.switch_glu = _DispatchProxy(
                switch_glu, layer_idx=layer_idx, observer=observer
            )
            wrapped += 1
        return wrapped

    # --- MoE variant guard (polymorphic override) ----------------------------

    @staticmethod
    def _validate_supported_variant(model: Any) -> None:
        """Reject Gemma4-MoE variants E1.2 does not cover.

        Polymorphic override of ``Gemma4Adapter._validate_supported_variant``:
        the dense parent's guard rejects MoE; this override REQUIRES
        MoE. Branches:

        - ``enable_moe_block`` must be True AND ``num_experts > 0``.
          A dense Gemma4 checkpoint routed here by mistake would be
          rejected, directing the caller to ``Gemma4Adapter`` (dense)
          via the ``enable_moe_block=False`` factory branch.
        - ``top_k_experts > 0`` AND ``top_k_experts <= num_experts``.
        - ``hidden_size_per_layer_input > 0``: same rejection as the
          dense parent — 2B / 4B variants need a per_layer_inputs
          signature this adapter does not thread through.
        - ``num_kv_shared_layers > 0``: same rejection as the dense
          parent — shared-KV forwards need a different cache-list
          length / routing that this adapter does not implement.
        """
        tc = Gemma4Adapter._text_config_dict(model)

        enable_moe = bool(tc.get("enable_moe_block", False))
        num_experts = int(tc.get("num_experts", 0) or 0)
        if not enable_moe or num_experts <= 0:
            raise NotImplementedError(
                "Gemma4MoeAdapter requires a MoE checkpoint "
                "(enable_moe_block=True and num_experts > 0). Got "
                f"enable_moe_block={enable_moe}, num_experts={num_experts}. "
                "Dense Gemma4 checkpoints go through silica.models.gemma4."
                "Gemma4Adapter; the factory's _build_gemma4 branches on "
                "enable_moe_block to route between the two."
            )

        top_k = int(tc.get("top_k_experts", 0) or 0)
        if top_k <= 0:
            raise NotImplementedError(
                "Gemma4MoeAdapter requires top_k_experts > 0; got "
                f"top_k_experts={top_k}."
            )
        if top_k > num_experts:
            raise NotImplementedError(
                "Gemma4MoeAdapter requires top_k_experts <= num_experts; "
                f"got top_k_experts={top_k}, num_experts={num_experts}."
            )

        per_layer_input = int(tc.get("hidden_size_per_layer_input", 0) or 0)
        if per_layer_input > 0:
            raise NotImplementedError(
                "Gemma4MoeAdapter does not thread per_layer_inputs through "
                "prefill / decode_step; the 2B / 4B variants with "
                "hidden_size_per_layer_input>0 need an extended call "
                f"signature. Got hidden_size_per_layer_input={per_layer_input}."
            )

        num_kv_shared = int(tc.get("num_kv_shared_layers", 0) or 0)
        if num_kv_shared > 0:
            raise NotImplementedError(
                "Gemma4MoeAdapter assumes len(make_cache()) == "
                "num_hidden_layers; shared-KV variants produce a shorter "
                "cache list and route via Gemma4TextModel.previous_kvs. "
                f"Got num_kv_shared_layers={num_kv_shared}."
            )

    # --- config extension ----------------------------------------------------

    @staticmethod
    def _augment_config_with_moe_metadata(
        base_config: ModelConfig, model: Any
    ) -> ModelConfig:
        """Attach MoE-specific metadata to ``ModelConfig.extra``.

        Surfaces the structural facts that E0 identified as relevant
        for downstream consumers (bench, dispatch seam, future
        HF-reference comparisons) without asking them to re-parse
        ``args.text_config``:

        - ``num_experts``, ``top_k_experts``, ``moe_intermediate_size``
          — plain routing metadata.
        - ``has_always_on_dense_mlp=True`` — documents Gemma4-MoE's
          key structural quirk vs Qwen3.5-MoE (where the sparse MoE
          block REPLACES the dense MLP).
        - ``moe_expert_path="layer.experts.switch_glu"`` — makes the
          proxy walk path machine-readable. If future tooling needs
          to reach the sparse block, it does not have to re-derive
          this from source.
        - ``is_moe_adapter=True`` — same marker key as Qwen3.5-MoE.
        """
        tc = Gemma4MoeAdapter._text_config_dict(model)
        extra = dict(base_config.extra)

        extra["num_experts"] = int(tc.get("num_experts", 0) or 0)
        extra["top_k_experts"] = int(tc.get("top_k_experts", 0) or 0)
        extra["moe_intermediate_size"] = int(
            tc.get("moe_intermediate_size", 0) or 0
        )
        extra["has_always_on_dense_mlp"] = True
        extra["moe_expert_path"] = "layer.experts.switch_glu"
        extra["is_moe_adapter"] = True

        return ModelConfig(
            model_name=base_config.model_name,
            num_layers=base_config.num_layers,
            hidden_size=base_config.hidden_size,
            vocab_size=base_config.vocab_size,
            extra=extra,
        )


__all__ = ["Gemma4MoeAdapter"]
