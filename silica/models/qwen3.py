"""silica.models.qwen3 — Qwen3.5 I-1 ModelAdapter (P-1 D-004 + D-010 + D-014).

P-1 adapter for the Qwen3.5 family (dev-loop model: Qwen3.5-0.8B; validated at
Gate B, see ``docs/P1_DAY1_GATE_B.md``). Borrows mlx-lm's ``load()`` for model
structure + tokenizer + weight loading (D-004), but routes KV cache through
Silica's ``SimpleKVCache`` (D-010 clean injection, confirmed at Gate A).

Per-layer attention dispatch (D-015): ``model.layers[i].is_linear`` toggles
between Gated DeltaNet (``HYBRID_DELTANET``) and full attention (``GLOBAL``).
Qwen3.5-0.8B is 18 linear + 6 full; other sizes share the same is_linear flag
and should work unchanged.

P-1 scope (D-014):
  - Text-only. Multimodal heads auto-filtered by mlx-lm's ``sanitize()``.
  - MTP disabled. ``sanitize()`` drops MTP weights and applies a +1.0 RMSNorm
    shift — non-obvious weight-correctness logic that makes D-004's
    loader-borrowing load-correctness-critical, not ergonomic.
  - DeltaNet recurrent state lives inside mlx-lm's ``ArraysCache`` entries,
    which are held by ``SimpleKVCache``'s per-layer list. The adapter does
    not surface recurrent state through ``StateDelta`` at P-1; accounting is
    a P-3 ``MemoryBudgeter`` concern.
  - Tokenizer is mlx-lm's ``TokenizerWrapper`` over HF ``AutoTokenizer`` —
    structurally satisfies Silica's ``Tokenizer`` protocol (Gate B (c)).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.utils import load as _mlx_lm_load

from silica.kvcache.manager import KVHandle
from silica.kvcache.simple import SimpleKVCache
from silica.mlx.runner import forward
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    Module,
    StateDelta,
    Tokenizer,
)
from silica.weights.provider import WeightProvider


class Qwen3Adapter:
    """I-1 ModelAdapter for Qwen3.5 (Gated DeltaNet + Gated Attention hybrid).

    Usage:

        model, tokenizer = mlx_lm.load("Qwen/Qwen3.5-0.8B")
        kv = SimpleKVCache.from_model(model)
        adapter = Qwen3Adapter(model, tokenizer, kv_manager=kv)

    Or the one-shot factory:

        adapter, kv = Qwen3Adapter.from_hf_repo("Qwen/Qwen3.5-0.8B")
    """

    config: ModelConfig

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        kv_manager: SimpleKVCache,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._kv_manager = kv_manager
        self.config = self._build_config(model, tokenizer)
        self._kv_layout = self._build_kv_layout(model)
        self._attention_pattern = self._build_attention_pattern(model)

    @classmethod
    def from_hf_repo(cls, repo: str) -> tuple[Qwen3Adapter, SimpleKVCache]:
        """Load ``repo`` via mlx-lm, build ``SimpleKVCache`` + adapter.

        Returns ``(adapter, kv)`` so the Engine can drive both. The KVManager
        is built here so ``SimpleKVCache.from_model`` is called exactly once
        on the post-load, post-sanitize model.
        """
        # mlx-lm load() returns Union[2tuple, 3tuple] without @overload;
        # with return_config omitted we get the 2-tuple variant at runtime.
        model, tokenizer = _mlx_lm_load(repo)  # type: ignore[misc]
        kv = SimpleKVCache.from_model(model)
        return cls(model, tokenizer, kv_manager=kv), kv

    # --- I-1 ModelAdapter Protocol surface ---

    def build(self, weight_provider: WeightProvider) -> Module:
        """Return the already-loaded mlx-lm model.

        D-004 / D-010 note: P-1 borrows mlx-lm's loader (including its
        Qwen3.5-specific ``sanitize()`` — which filters vision + MTP weights
        and applies the +1.0 RMSNorm shift). ``weight_provider`` is accepted
        for Protocol conformance but not used; mlx-lm has already consumed
        the safetensors by the time the adapter is constructed. P-3 revisits
        this when ``WeightProvider`` needs to own the bytes for MoE + VQ +
        NVMe residency.
        """
        del weight_provider
        return self._model

    def kv_layout(self) -> KVLayout:
        return self._kv_layout

    def attention_pattern(self) -> AttentionPattern:
        return self._attention_pattern

    def tokenizer(self) -> Tokenizer:
        # mlx-lm's TokenizerWrapper is structurally a Tokenizer; mlx-lm stubs
        # expose it as Any, so mypy cannot verify the structural cast.
        return self._tokenizer  # type: ignore[no-any-return]

    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        cache_list = self._kv_manager.cache_list(kv_handle.req_id)
        logits = forward(self._model, tokens, cache_list)
        return logits, StateDelta()

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        cache_list = self._kv_manager.cache_list(kv_handle.req_id)
        logits = forward(self._model, token, cache_list)
        return logits, StateDelta()

    # --- Silica config builders ---

    @staticmethod
    def _build_config(model: Any, tokenizer: Any) -> ModelConfig:
        text_config = Qwen3Adapter._text_config_dict(model)
        return ModelConfig(
            model_name=str(getattr(model, "model_type", "qwen3_5")),
            num_layers=len(model.layers),
            hidden_size=int(text_config.get("hidden_size", 0) or 0),
            vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 0),
            extra={"text_config_keys": sorted(text_config.keys())},
        )

    @staticmethod
    def _build_kv_layout(model: Any) -> KVLayout:
        """Extract KV shape from the first full-attention (non-linear) layer.

        For a pure-recurrent stack (hypothetical future Qwen variant with no
        full-attention layers) this returns zeros; paged KV is then trivial.
        """
        for layer in model.layers:
            if not getattr(layer, "is_linear", False):
                sa = getattr(layer, "self_attn", None)
                if sa is not None:
                    return KVLayout(
                        num_layers=len(model.layers),
                        n_kv_heads=int(getattr(sa, "num_key_value_heads", 0) or 0),
                        head_dim=int(getattr(sa, "head_dim", 0) or 0),
                        dtype=mx.float16,
                    )
        return KVLayout(
            num_layers=len(model.layers),
            n_kv_heads=0,
            head_dim=0,
            dtype=mx.float16,
        )

    @staticmethod
    def _build_attention_pattern(model: Any) -> AttentionPattern:
        """Per-layer AttentionKind for Qwen3.5 (D-015).

        ``layer.is_linear`` is the mlx-lm-native toggle for DeltaNet vs full
        attention layers. Full attention in Qwen3.5 is causal (no sliding
        window at config level — verified against ``mlx_lm/models/qwen3_5.py``
        ``create_attention_mask`` call), so GLOBAL is the correct tag.
        """
        kinds = tuple(
            AttentionKind.HYBRID_DELTANET
            if getattr(layer, "is_linear", False)
            else AttentionKind.GLOBAL
            for layer in model.layers
        )
        return AttentionPattern(per_layer=kinds)

    @staticmethod
    def _text_config_dict(model: Any) -> dict[str, Any]:
        """Normalise ``model.args.text_config`` to a dict.

        mlx-lm stores Qwen3.5's text-config as a raw dict on the ``ModelArgs``
        dataclass (``from_dict`` preserves the dict). Return an empty dict if
        the model is a fake / malformed fixture; callers default gracefully.
        """
        args = getattr(model, "args", None)
        if args is None:
            return {}
        raw = getattr(args, "text_config", None)
        if isinstance(raw, dict):
            return raw
        return {}
