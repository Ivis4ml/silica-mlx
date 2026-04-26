"""silica.models.qwen3_5_moe — I-1 adapter for the Qwen3.5-MoE family (P-3-E1.1).

**Scope: Qwen3.5-MoE dense-router sparse-MoE variants (e.g. Qwen3.5-35B-A3B).**
The adapter wraps mlx-lm's already-constructed Qwen3.5-MoE graph:
``mlx_lm.models.qwen3_5_moe.Model`` is a 53-line wrapper around
``qwen3_5.Model`` whose ``DecoderLayer.__init__`` branches
``self.mlp = SparseMoeBlock(args)`` when ``num_experts > 0``.
``SparseMoeBlock`` is imported from ``qwen3_next.py`` as
``Qwen3NextSparseMoeBlock`` — SwitchGLU + softmax top-k routing,
**plus** a sigmoid-gated shared-expert branch (
``shared_expert`` + ``shared_expert_gate``).

This adapter does **not** reimplement any MoE math. All routing,
``SwitchGLU``, shared-expert handling, and quantization stay inside
mlx-lm. Silica contributes:

  - Capability declaration (``has_moe=True``; attention kinds
    inherit the dense Qwen3.5 3:1 ``[D, D, D, G]`` hybrid pattern
    via ``Qwen3_5Adapter._build_attention_pattern``).
  - Factory dispatch via ``model_type="qwen3_5_moe"``.
  - Variant guards rejecting checkpoints the adapter does not
    cover (``num_experts <= 0``, ``num_experts_per_tok <= 0``,
    ``num_experts_per_tok > num_experts``, or non-empty
    ``mlp_only_layers`` — see E-open-2 in
    ``docs/P3_MOE_SURVEY.md``; the 35B-A3B probe found this list
    empty, but guarding makes the adapter refuse checkpoints
    that would trip the mlx-lm qwen3_5 DecoderLayer's silent
    MoE-onto-dense-MLP behaviour).
  - Option (c) dispatch-observation seam (
    ``install_dispatch_proxy``): wraps each MoE layer's
    ``layer.mlp.switch_mlp`` with a thin forwarding proxy that
    reports ``(layer_idx, indices)`` to a caller-supplied
    observer before delegating to the original ``SwitchGLU``.
    **Not installed by default**; the dense
    ``ResidentWeightProvider.get_expert`` raises
    ``NotImplementedError`` by design (D-011), so wiring the
    proxy unconditionally would break smoke / single-request
    paths. E2 uses this seam with a mock provider to pin the
    D-011 "per-expert at dispatch, fused at fetch" resolution
    (E-open-1 resolution, 2026-04-20).

Batched execution opens at E4 (smoke-only, parity deferred): the
``ContinuousBatcher._enforce_capability_gate`` ``has_moe=True``
rejection is lifted. mlx-lm's ``SwitchGLU`` + ``gather_mm`` path
is B-agnostic per the P3_MOE_SURVEY §5 E4 audit — a batched
forward dispatches per-row top-k experts without further
scheduler work. Real-model B=2 coverage:
``tests/test_p3_qwen3_5_moe_batched_smoke.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from silica.kvcache.simple import SimpleKVCache
from silica.models.adapter import (
    ModelConfig,
    Module,
)
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)
from silica.models.qwen3_5 import Qwen3_5Adapter
from silica.weights.provider import WeightProvider

DispatchObserver = Callable[[int, Any], None]


class _DispatchProxy:
    """Minimal forwarding proxy around a ``SwitchGLU`` instance.

    Purpose: record ``(layer_idx, indices)`` on every MoE FFN dispatch
    so a caller-supplied observer can assert which experts were
    activated, while still delegating the real sparse matmul to the
    original ``SwitchGLU`` (preserves ``gather_mm`` fast path, incl.
    quantized ``QuantizedSwitchLinear``).

    Attribute access (``__getattr__``) falls through to the inner
    object so the rest of the ``SwitchGLU`` interface — ``weights``,
    ``to_quantized``, etc. — remains reachable.
    """

    def __init__(
        self,
        inner: Any,
        *,
        layer_idx: int,
        observer: DispatchObserver,
    ) -> None:
        self._inner = inner
        self._layer_idx = layer_idx
        self._observer = observer

    def __call__(self, x: Any, indices: Any) -> Any:
        self._observer(self._layer_idx, indices)
        return self._inner(x, indices)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class Qwen3_5MoeAdapter(Qwen3_5Adapter):
    """I-1 ModelAdapter for Qwen3.5-MoE (35B-A3B class).

    Inherits every I-1 method from ``Qwen3_5Adapter`` unchanged —
    the MoE variant uses the same hybrid DeltaNet + GQA attention,
    the same KV layout, the same ``make_batch_cache`` factory, and
    the same single-request ``prefill`` / ``decode_step``. Only the
    capability declaration, the MoE variant guard, and the
    ``install_dispatch_proxy`` seam are new.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        kv_manager: SimpleKVCache,
    ) -> None:
        self._validate_supported_moe_variant(model)
        super().__init__(model, tokenizer, kv_manager=kv_manager)
        self.config = self._augment_config_with_moe_metadata(
            self.config, model
        )

    def capabilities(self) -> ModelCapabilities:
        # Override Qwen3_5Adapter.capabilities to flip has_moe=True.
        # attention_pattern and has_recurrent_state are inherited
        # verbatim — Qwen3.5-MoE has the same 3:1 [D, D, D, G]
        # hybrid pattern as dense Qwen3.5, so the existing derivation
        # is already correct.
        return capabilities_from_attention_pattern(
            self._attention_pattern, has_moe=True
        )

    def build(self, weight_provider: WeightProvider) -> Module:
        # Same as dense: borrow mlx-lm's already-loaded model. The
        # dispatch proxy is installed via install_dispatch_proxy on
        # explicit opt-in; build() stays untouched so the default
        # path (ResidentWeightProvider, whose get_expert raises) is
        # never invoked by accident on a plain Engine.generate call.
        return super().build(weight_provider)

    def install_dispatch_proxy(self, observer: DispatchObserver) -> int:
        """Wrap each MoE layer's ``switch_mlp`` with a dispatch-observing
        proxy (option (c) seam, E-open-1 resolution).

        The observer is called on every sparse-MLP forward with
        ``(layer_idx, indices)``; the original ``SwitchGLU`` then
        produces the actual output. Returns the number of layers
        wrapped. Idempotent: re-installing over an already-wrapped
        layer is a no-op.

        Contract:
          - Only the ``switch_mlp`` inside each ``SparseMoeBlock`` is
            wrapped. Shared-expert weights, router gate, etc. are
            untouched — they are part of the fast path.
          - Layers whose ``mlp`` has no ``switch_mlp`` attribute
            (dense MLP, shared-expert-only, or non-MoE) are skipped
            silently.
          - The wrapper does **not** call
            ``WeightProvider.get_expert`` by itself — the observer
            is the integration point for that, wired by E2 with a
            mock provider.

        Intentionally NOT invoked by ``build`` so the adapter's
        default single-request path works with the dense
        ``ResidentWeightProvider``; the dense provider's
        ``get_expert`` raises ``NotImplementedError`` by D-011, and
        an unconditional proxy install would propagate that failure
        into smoke tests.
        """
        wrapped = 0
        for layer_idx, layer in enumerate(self._model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            switch_mlp = getattr(mlp, "switch_mlp", None)
            if switch_mlp is None:
                continue
            if isinstance(switch_mlp, _DispatchProxy):
                continue  # idempotent
            mlp.switch_mlp = _DispatchProxy(
                switch_mlp, layer_idx=layer_idx, observer=observer
            )
            wrapped += 1
        return wrapped

    # --- MoE variant guard ----------------------------------------------------

    @staticmethod
    def _validate_supported_moe_variant(model: Any) -> None:
        """Reject Qwen3.5-MoE variants E1.1 does not cover.

        Loud-fail branches (each raises ``NotImplementedError`` at
        construction):

        - ``num_experts <= 0`` — the checkpoint is not MoE; use the
          dense ``Qwen3_5Adapter`` via ``model_type="qwen3_5"``.
        - ``num_experts_per_tok <= 0`` — malformed config.
        - ``num_experts_per_tok > num_experts`` — impossible top-k.
        - ``mlp_only_layers`` non-empty —
          ``qwen3_5.DecoderLayer.__init__`` does NOT consult this
          field (only ``qwen3_next.Qwen3NextDecoderLayer`` does), so
          a non-empty list would mean mlx-lm silently wires MoE onto
          layers the checkpoint author intended to be dense MLP.
          E-open-2 in ``docs/P3_MOE_SURVEY.md`` tracks this; on the
          probed 35B-A3B-4bit checkpoint the list is empty.
        """
        tc = Qwen3_5MoeAdapter._text_config_dict(model)

        num_experts = int(tc.get("num_experts", 0) or 0)
        if num_experts <= 0:
            raise NotImplementedError(
                "Qwen3_5MoeAdapter requires a MoE checkpoint with "
                "num_experts > 0; got num_experts="
                f"{num_experts}. The dense Qwen3.5 family lives in "
                "silica.models.qwen3_5.Qwen3_5Adapter "
                "(model_type='qwen3_5')."
            )
        top_k = int(tc.get("num_experts_per_tok", 0) or 0)
        if top_k <= 0:
            raise NotImplementedError(
                "Qwen3_5MoeAdapter requires num_experts_per_tok > 0; "
                f"got num_experts_per_tok={top_k}."
            )
        if top_k > num_experts:
            raise NotImplementedError(
                "Qwen3_5MoeAdapter requires num_experts_per_tok <= "
                f"num_experts; got num_experts_per_tok={top_k}, "
                f"num_experts={num_experts}."
            )

        mlp_only_layers = list(tc.get("mlp_only_layers") or [])
        if mlp_only_layers:
            raise NotImplementedError(
                "Qwen3_5MoeAdapter: checkpoint declares non-empty "
                f"mlp_only_layers={mlp_only_layers}, but mlx-lm's "
                "qwen3_5.DecoderLayer does not consult this field "
                "(only qwen3_next.Qwen3NextDecoderLayer does). A "
                "non-empty list would mean mlx-lm silently wires "
                "SparseMoeBlock onto layers the config intends to "
                "be dense MLP — refusing rather than accepting a "
                "silently-wrong wiring. See E-open-2 in "
                "docs/P3_MOE_SURVEY.md; the probed 35B-A3B-4bit "
                "checkpoint has this list empty."
            )

    # --- config extension -----------------------------------------------------

    @staticmethod
    def _augment_config_with_moe_metadata(
        base_config: ModelConfig, model: Any
    ) -> ModelConfig:
        """Attach MoE-specific metadata to ``ModelConfig.extra``.

        Written as a read into ``args.text_config`` rather than a
        parallel parse of the checkpoint — the dense
        ``Qwen3_5Adapter._build_config`` has already populated the
        base config and deposited ``text_config_keys`` in ``extra``;
        we only add the MoE-specific overlay and the runtime
        resolutions for fields the probe would otherwise misread
        (``norm_topk_prob``, ``attn_output_gate``).
        """
        tc = Qwen3_5MoeAdapter._text_config_dict(model)
        extra = dict(base_config.extra)

        extra["num_experts"] = int(tc.get("num_experts", 0) or 0)
        extra["num_experts_per_tok"] = int(
            tc.get("num_experts_per_tok", 0) or 0
        )
        extra["moe_intermediate_size"] = int(
            tc.get("moe_intermediate_size", 0) or 0
        )
        extra["shared_expert_intermediate_size"] = int(
            tc.get("shared_expert_intermediate_size", 0) or 0
        )

        # norm_topk_prob: the key is commonly absent in MoE configs;
        # the runtime value comes from qwen3_5.TextModelArgs dataclass
        # default (True). qwen3_next.ModelArgs has the opposite
        # default (False), so checkpoints that go through a different
        # wrapper could resolve differently — surface the runtime
        # value explicitly so downstream consumers do not re-read the
        # raw config key and miss the default.
        if "norm_topk_prob" in tc:
            extra["norm_topk_prob_runtime"] = bool(tc["norm_topk_prob"])
        else:
            # Qwen3.5 family dataclass default.
            extra["norm_topk_prob_runtime"] = True

        # attn_output_gate: the 35B-A3B checkpoint sets this to True,
        # but a repo-wide grep of mlx_lm.models finds no consumer —
        # mlx-lm silently drops the flag. Record both so a future
        # adapter / bench tool can compare HF-vs-mlx-lm behaviour
        # without having to re-derive this from source.
        extra["attn_output_gate_config"] = bool(
            tc.get("attn_output_gate", False)
        )
        extra["attn_output_gate_mlx_lm_honors"] = False

        extra["mlp_only_layers"] = list(tc.get("mlp_only_layers") or [])
        extra["is_moe_adapter"] = True

        return ModelConfig(
            model_name=base_config.model_name,
            num_layers=base_config.num_layers,
            hidden_size=base_config.hidden_size,
            vocab_size=base_config.vocab_size,
            extra=extra,
        )

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _text_config_dict(model: Any) -> dict[str, Any]:
        """Read ``model.args.text_config`` as a dict.

        Qwen3.5-MoE stores structural fields under
        ``args.text_config`` as a plain dict (identical to the dense
        Qwen3.5 layout). Returns an empty dict when the attribute
        chain is missing so downstream ``tc.get(...)`` calls stay
        non-crashing during unit testing with fake models.
        """
        args = getattr(model, "args", None)
        if args is None:
            return {}
        tc = getattr(args, "text_config", None)
        if isinstance(tc, dict):
            return tc
        return {}


__all__ = ["Qwen3_5MoeAdapter", "DispatchObserver"]
