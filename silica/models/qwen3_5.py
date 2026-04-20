"""silica.models.qwen3_5 — I-1 adapter for the Qwen3.5 hybrid family.

Scope: Qwen3.5 models — 0.8B / 27B / 35B-A3B / ... — all of which share
an interleaved Gated DeltaNet + Gated Attention stack, MTP head, and a
multimodal lm_head built on top. mlx-lm ships them under
``mlx_lm.models.qwen3_5``; this adapter wraps that module for Silica's
I-1 Protocol.

For plain Qwen3 (0.6B / 4B / 7B etc., pure KV attention) use
``silica.models.qwen3.Qwen3Adapter`` instead. One adapter per model
generation keeps each family's quirks explicit; adding a new family
(Qwen4, DeepSeek, Kimi, GLM, MiniMax, …) is a new file, not a growing
conditional chain.

Qwen3.5-specific traits this adapter handles:

- `model.model_type == "qwen3_5"`.
- Per-layer `layer.is_linear` distinguishes DeltaNet from full attention;
  Silica maps it to ``AttentionKind.HYBRID_DELTANET`` vs ``GLOBAL``
  (D-015). The dispatch lines up 1:1 with mlx-lm's cache factory
  (``ArraysCache`` for linear, ``KVCache`` for full attention).
- `self_attn.num_key_value_heads` + `self_attn.head_dim` on the
  full-attention blocks (NOT `n_kv_heads` — that name is Qwen3-plain's).
- `model.args.text_config` is a **nested dict** carrying hidden_size /
  num_hidden_layers (flat Qwen3 exposes these on ``args`` directly).
- mlx-lm's ``Model.sanitize`` drops ``vision_tower`` / ``model.visual``
  at load; ``TextModel.sanitize`` drops ``mtp.*`` and shifts any
  ``RMSNorm.weight`` by +1.0 when MTP weights were present in the
  checkpoint (non-obvious load-correctness detail that D-004 explicitly
  relies on — one of the reasons Silica borrows mlx-lm's loader).
- P-1 runs single-request through ``SimpleKVCache`` (verified at M-2).
- P-2 batched path for hybrid DeltaNet is deferred to P-3 via an
  adapter-owned ``BatchRecurrentStateStore`` (see ``docs/P2_OPENING.md``
  scope statement). The P-2 ``ContinuousBatcher`` will refuse to accept
  this adapter at construction time.
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
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)
from silica.weights.provider import WeightProvider


class Qwen3_5Adapter:
    """I-1 ModelAdapter for the Qwen3.5 family (hybrid DeltaNet + full attn)."""

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
    def from_hf_repo(cls, repo: str) -> tuple[Qwen3_5Adapter, SimpleKVCache]:
        """Load ``repo`` via mlx-lm, build ``SimpleKVCache`` + adapter."""
        # mlx-lm's load() returns Union[2tuple, 3tuple]; the 2-tuple variant
        # applies with return_config omitted.
        model, tokenizer = _mlx_lm_load(repo)  # type: ignore[misc]
        kv = SimpleKVCache.from_model(model)
        return cls(model, tokenizer, kv_manager=kv), kv

    def build(self, weight_provider: WeightProvider) -> Module:
        """Return the already-loaded mlx-lm model (D-004: borrow the loader)."""
        del weight_provider
        return self._model

    def kv_layout(self) -> KVLayout:
        return self._kv_layout

    def attention_pattern(self) -> AttentionPattern:
        return self._attention_pattern

    def capabilities(self) -> ModelCapabilities:
        # Qwen3.5 is hybrid DeltaNet + GQA; has_recurrent_state is True
        # via ``HYBRID_DELTANET`` layers in the pattern. No MoE in the
        # dense variants; the MoE-A3B variant lives in a separate adapter
        # that will override ``has_moe=True`` when it lands in P-3.
        return capabilities_from_attention_pattern(self._attention_pattern)

    def tokenizer(self) -> Tokenizer:
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

    # --- family-specific metadata extraction ---

    @staticmethod
    def _build_config(model: Any, tokenizer: Any) -> ModelConfig:
        text_config = Qwen3_5Adapter._text_config_dict(model)
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

        Returns zeros for a pure-recurrent stack — P-2 paged KV of size 0
        is a no-op, which is the correct semantics for that case.
        """
        for layer in model.layers:
            if getattr(layer, "is_linear", False):
                continue
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
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
        """``layer.is_linear`` → ``HYBRID_DELTANET``; else ``GLOBAL`` (D-015)."""
        kinds = tuple(
            AttentionKind.HYBRID_DELTANET
            if getattr(layer, "is_linear", False)
            else AttentionKind.GLOBAL
            for layer in model.layers
        )
        return AttentionPattern(per_layer=kinds)

    @staticmethod
    def _text_config_dict(model: Any) -> dict[str, Any]:
        """Return ``model.args.text_config`` as a dict, or ``{}`` if absent."""
        args = getattr(model, "args", None)
        if args is None:
            return {}
        raw = getattr(args, "text_config", None)
        if isinstance(raw, dict):
            return raw
        return {}
