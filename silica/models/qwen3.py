"""silica.models.qwen3 — I-1 adapter for the plain Qwen3 family (pure KV).

Scope: Qwen3 models with pure GQA causal attention — 0.6B / 4B / 7B /
14B / 32B ... — that mlx-lm ships under ``mlx_lm.models.qwen3``. These
are the P-2 dev-loop targets because all layers share one cache type
(``KVCache``), which is what the batched ``ContinuousBatcher`` is built
for.

For hybrid Qwen3.5 (DeltaNet + GQA + MTP + multimodal) use
``silica.models.qwen3_5.Qwen3_5Adapter``; it lives in a separate module
so plain-family adapter logic never drifts to accommodate hybrid
concerns (and vice versa). Adding a new family (Qwen4, DeepSeek, Kimi,
GLM, MiniMax, …) is a new file per family — not a growing conditional
in this one.

Plain-Qwen3 traits this adapter handles:

- `model.model_type == "qwen3"`.
- All layers are GQA KV attention → every entry in ``AttentionPattern``
  is ``AttentionKind.GLOBAL``.
- `self_attn.n_kv_heads` (NOT ``num_key_value_heads`` — that name is
  Qwen3.5's).
- `head_dim` can be stored as `self_attn.head_dim`, `model.args.head_dim`,
  or absent; fall back to ``hidden_size // num_attention_heads`` in the
  absent case. Qwen3-0.6B specifically uses
  ``args.head_dim = 128`` (larger than the naive 64 = 1024 / 16);
  reading ``args.head_dim`` primary is load-bearing for P-2
  ``MemoryBudgeter.bytes_per_token`` accuracy.
- `model.args.hidden_size` is on ``args`` directly (flat), not under
  ``args.text_config``.
- No MTP, no multimodal, no RMSNorm +1.0 sanitize (those are Qwen3.5).
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


class Qwen3Adapter:
    """I-1 ModelAdapter for the plain Qwen3 family (pure KV attention)."""

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
        """Load ``repo`` via mlx-lm, build ``SimpleKVCache`` + adapter."""
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
        # Plain Qwen3 is pure GQA attention, no recurrent state, no MoE.
        # The helper reduces self._attention_pattern to the typed summary.
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
        """Read hidden_size from flat ``model.args`` (plain-Qwen3 layout)."""
        args = getattr(model, "args", None)
        hidden_size = int(getattr(args, "hidden_size", 0) or 0) if args else 0
        return ModelConfig(
            model_name=str(getattr(model, "model_type", "qwen3")),
            num_layers=len(model.layers),
            hidden_size=hidden_size,
            vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 0),
            extra={},
        )

    @staticmethod
    def _build_kv_layout(model: Any) -> KVLayout:
        """KV shape from plain-Qwen3 attributes (``n_kv_heads``, ``args.head_dim``)."""
        for layer in model.layers:
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
            n_kv = int(getattr(sa, "n_kv_heads", 0) or 0)
            head_dim = int(getattr(sa, "head_dim", 0) or 0)
            if head_dim == 0:
                head_dim = Qwen3Adapter._head_dim_from_args(model)
            return KVLayout(
                num_layers=len(model.layers),
                n_kv_heads=n_kv,
                head_dim=head_dim,
                dtype=mx.float16,
            )
        return KVLayout(
            num_layers=len(model.layers),
            n_kv_heads=0,
            head_dim=0,
            dtype=mx.float16,
        )

    @staticmethod
    def _head_dim_from_args(model: Any) -> int:
        """``args.head_dim`` primary; fall back to ``hidden / num_heads``."""
        args = getattr(model, "args", None)
        if args is None:
            return 0
        h = int(getattr(args, "head_dim", 0) or 0)
        if h > 0:
            return h
        hidden = int(getattr(args, "hidden_size", 0) or 0)
        heads = int(getattr(args, "num_attention_heads", 1) or 1)
        return hidden // max(heads, 1)

    @staticmethod
    def _build_attention_pattern(model: Any) -> AttentionPattern:
        """Plain Qwen3 is pure KV — every layer is ``GLOBAL``."""
        return AttentionPattern(
            per_layer=tuple(AttentionKind.GLOBAL for _ in model.layers)
        )
