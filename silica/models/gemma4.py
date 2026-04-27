"""silica.models.gemma4 — I-1 adapter for the Gemma4 dense 31B text path (P-3-D1).

**Scope: Gemma4-31B dense text path ONLY** (D-open-2 resolution). Other
Gemma4 variants intentionally raise at construction time so a silent
misconfiguration cannot slip into production:

- 1B / 2B / 4B variants with ``hidden_size_per_layer_input > 0`` — the
  forward expects a ``per_layer_inputs`` argument threaded from the
  adapter, which ``Qwen3Adapter``-style ``forward(model, tokens,
  cache_list)`` does not supply.
- 26B-A4B MoE variants with ``enable_moe_block=True`` — expert routing
  and the ``SwitchGLU`` path belong to P-3-E, not here.
- Shared-KV variants with ``num_kv_shared_layers > 0`` — the cache
  list is shorter than ``num_hidden_layers`` and the forward reroutes
  per ``Gemma4TextModel.previous_kvs``. Spot-checked at construction;
  D-open-3 defers a full-family solution to a later unit.

Gemma4-31B has two coexisting KV shapes — sliding layers use
``(n_kv_heads=16, head_dim=256)``, full layers use
``(n_kv_heads=4, head_dim=512)`` with ``attention_k_eq_v=True`` (see
``plans/P3_GEMMA4_SURVEY.md`` §3.4). ``KVLayout``'s four primary fields
stay as a single-shape summary; per D-open-1 option (a) we populate
them from the sliding (majority) layer and record the full per-kind
detail in ``ModelConfig.extra`` under explicit keys plus a
``kv_layout_caveat`` string that makes the summary nature readable
from any consumer. P-3-D4 additionally populates
``KVLayout.bytes_per_token_total`` with an explicit per-kind sum so
``MemoryBudgeter.for_adapter`` no longer derives a wrong total from
those summary fields. The D4 scalar still assumes unbounded growth
on sliding layers; a full sliding-window-aware budget model is
future work, tracked separately from D4.

``make_batch_cache`` (P-3-D2) returns a hybrid per-layer list of
``BatchRotatingKVCache`` (sliding) and ``BatchKVCache`` (full) from
``mlx_lm.models.cache``. As of P-3-D3 the capability gate accepts
``SLIDING`` under the miss-only admission path (constructor rejects
``prefix_cache != None`` when SLIDING is present), so batched
execution through ``Engine.generate_batch(..., prefix_cache=None)``
is the supported batched path today.

Adapter-local D-015 lifecycle helpers (``commit_state`` /
``rollback_state`` / ``state_from_prefix`` / ``free_state``) are NOT
defined here — Gemma4 is pure KV-attention, ``has_recurrent_state=False``.
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
from silica.models.pre_norm_capture import (
    _PreNormCaptureBufferHolder,
    apply_k_norm_then_rope_to_block,
    install_pre_norm_capture_proxies,
)
from silica.weights.provider import WeightProvider

_KV_LAYOUT_CAVEAT = (
    "Gemma4 has two coexisting KV shapes (sliding_attention + "
    "full_attention). KVLayout here is a summary populated from the "
    "sliding-layer fields; per-kind details are in ModelConfig.extra. "
    "See plans/P3_GEMMA4_SURVEY.md §5.1."
)

_DTYPE_BY_NAME: dict[str, mx.Dtype] = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}


class Gemma4Adapter:
    """I-1 ModelAdapter for Gemma4-31B dense (sliding + full attention)."""

    config: ModelConfig

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        kv_manager: SimpleKVCache,
    ) -> None:
        self._validate_supported_variant(model)
        self._model = model
        self._tokenizer = tokenizer
        self._kv_manager = kv_manager
        # Order matters: _build_attention_pattern runs the strict
        # layer_types guards (missing / empty / length mismatch / unknown
        # value), so building it first means _build_kv_layout is only
        # ever called on a config that has already been validated — and
        # can consume the pattern to compute a per-kind
        # bytes_per_token_total.
        self._attention_pattern = self._build_attention_pattern(model)
        self.config = self._build_config(model, tokenizer)
        self._kv_layout = self._build_kv_layout(model, self._attention_pattern)
        # P-5-F F.1: install pre-norm K capture proxies on every
        # attention layer. Gemma4-31B has both sliding and global
        # attention layers; both use the same ``Attention`` class
        # with ``k_proj`` / ``k_norm`` / ``rope`` per
        # ``mlx_lm/models/gemma4_text.py``, so all layers participate
        # in K_pre capture. F.2 scheduler integration (next sub-unit)
        # must handle the sliding-layer reconstruction case
        # specifically: ``cache_list[src_idx] = seeded_attn[pos]``
        # would replace a ``BatchRotatingKVCache`` (sliding) with a
        # fresh ``BatchKVCache``, breaking the rotating-window
        # invariants. F.2's seeded-cache assembly needs either a
        # Gemma4-specific filter that excludes sliding layers from
        # ``attn_layer_indices`` for cache reconstruction (capture
        # still active on all layers — only the seeded admit path is
        # affected) or a sliding-aware seeded-cache build that
        # re-constructs into a ``BatchRotatingKVCache``.
        self._attn_layer_indices: list[int] = list(range(len(model.layers)))
        self._capture_holder = _PreNormCaptureBufferHolder()
        install_pre_norm_capture_proxies(
            model,
            attn_layer_indices=self._attn_layer_indices,
            holder=self._capture_holder,
        )

    @classmethod
    def from_hf_repo(cls, repo: str) -> tuple[Gemma4Adapter, SimpleKVCache]:
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
        # Gemma4-31B is sliding + full-attention hybrid (no DeltaNet, no
        # MoE). has_recurrent_state is False. The capability gate
        # accepts SLIDING after P-3-D3; the constructor rejects a
        # non-None prefix_cache for SLIDING-bearing adapters, so
        # batched execution is available via the miss-only path.
        return capabilities_from_attention_pattern(self._attention_pattern)

    def make_batch_cache(self, left_padding: list[int]) -> list[Any]:
        """Build a hybrid per-layer batched cache list (P-3-D2).

        Sliding layers get ``BatchRotatingKVCache(max_size=sliding_window,
        left_padding=...)``; full-attention layers get ``BatchKVCache(
        left_padding=...)``. Both primitives already ship in mlx-lm
        (``mlx_lm.models.cache``); see ``plans/P3_BATCH_ROTATING_KV_SURVEY.md``.

        The per-layer ordering follows ``self._attention_pattern.per_layer``,
        which was built from ``config.text_config['layer_types']`` under
        the strict guards in ``_build_attention_pattern``. As of
        P-3-D3 the capability gate accepts ``SLIDING`` under the
        miss-only admission path.

        A ``sliding_window <= 0`` when the pattern contains ``SLIDING``
        layers is a config inconsistency and raises loudly rather than
        passing a sentinel to ``BatchRotatingKVCache``.
        """
        from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

        per_layer = self._attention_pattern.per_layer
        sliding_window = int(self.config.extra.get("sliding_window", 0) or 0)
        if AttentionKind.SLIDING in per_layer and sliding_window <= 0:
            raise NotImplementedError(
                "Gemma4Adapter.make_batch_cache: attention pattern "
                "contains SLIDING layers but text_config['sliding_window']"
                f"={sliding_window}. BatchRotatingKVCache requires a "
                "positive max_size; check the loaded repo's config."
            )

        caches: list[Any] = []
        for kind in per_layer:
            if kind is AttentionKind.SLIDING:
                caches.append(
                    BatchRotatingKVCache(
                        max_size=sliding_window, left_padding=left_padding
                    )
                )
            elif kind is AttentionKind.GLOBAL:
                caches.append(BatchKVCache(left_padding=left_padding))
            else:
                raise NotImplementedError(
                    "Gemma4Adapter.make_batch_cache: unexpected "
                    f"AttentionKind.{kind.name} in per-layer pattern. "
                    "Supported kinds today: SLIDING, GLOBAL."
                )
        return caches

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

    # --- P-5-F F.1: PreNormCaptureAdapter implementation ---

    def install_pre_norm_capture(
        self, buffer: dict[int, mx.array] | None
    ) -> None:
        """Arm or disarm the K_pre capture buffer for the next forward.

        Disarmed by default; sliding and global layers share the same
        proxy treatment.
        """
        self._capture_holder.buffer = buffer

    def apply_k_norm_then_rope(
        self,
        attn_layer_pos: int,
        k_pre_block: mx.array,
        *,
        offset: int,
    ) -> mx.array:
        """Reconstruct post-RoPE K on the layer at
        ``self._attn_layer_indices[attn_layer_pos]``.

        Gemma4 has all layers as attention (sliding or global), so
        ``attn_layer_pos == layer_idx`` here. Both kinds share the
        same per-layer ``self_attn.k_norm`` / ``self_attn.rope`` API.
        """
        layer_idx = self._attn_layer_indices[attn_layer_pos]
        attn = self._model.layers[layer_idx].self_attn
        return apply_k_norm_then_rope_to_block(
            k_pre_block,
            k_norm=attn.k_norm,
            rope_instance=attn.rope,
            offset=offset,
        )

    # --- variant guard ---

    @staticmethod
    def _validate_supported_variant(model: Any) -> None:
        """Reject Gemma4 variants P-3-D1 does not cover.

        Raised at construction time so a factory-level ``adapter_for_repo``
        cannot hand a misconfigured adapter to the engine. Each branch
        cites the concrete code path that is missing and the future
        sub-unit that is expected to land it.
        """
        tc = Gemma4Adapter._text_config_dict(model)

        enable_moe = bool(tc.get("enable_moe_block", False))
        num_experts = tc.get("num_experts")
        if enable_moe or (num_experts is not None and int(num_experts) > 0):
            raise NotImplementedError(
                "Gemma4Adapter (P-3-D1) supports the dense 31B text path "
                "only; MoE variants (e.g. gemma-4-26B-A4B) need expert "
                "routing plumbing scheduled for P-3-E. Got "
                f"enable_moe_block={enable_moe}, num_experts={num_experts}."
            )

        per_layer_input = int(tc.get("hidden_size_per_layer_input", 0) or 0)
        if per_layer_input > 0:
            raise NotImplementedError(
                "Gemma4Adapter (P-3-D1) does not thread per_layer_inputs "
                "through prefill / decode_step; the 2B / 4B variants with "
                "hidden_size_per_layer_input>0 need an extended call "
                f"signature. Got hidden_size_per_layer_input={per_layer_input}."
            )

        num_kv_shared = int(tc.get("num_kv_shared_layers", 0) or 0)
        if num_kv_shared > 0:
            raise NotImplementedError(
                "Gemma4Adapter (P-3-D1) assumes len(make_cache()) == "
                "num_hidden_layers; shared-KV variants produce a shorter "
                "cache list and route via Gemma4TextModel.previous_kvs. "
                f"Got num_kv_shared_layers={num_kv_shared}."
            )

    # --- family-specific metadata extraction ---

    @staticmethod
    def _text_config_dict(model: Any) -> dict[str, Any]:
        """Read ``model.args.text_config`` as a dict.

        Gemma4 stores structural fields under ``args.text_config`` as a
        plain dict (not an object), matching the Qwen3.5 layout. Return
        an empty dict when the attribute chain is missing so downstream
        ``tc.get(...)`` calls stay non-crashing during unit-testing.
        """
        args = getattr(model, "args", None)
        if args is None:
            return {}
        tc = getattr(args, "text_config", None)
        if isinstance(tc, dict):
            return tc
        return {}

    @staticmethod
    def _build_config(model: Any, tokenizer: Any) -> ModelConfig:
        """Populate ``ModelConfig`` with Gemma4 text fields + per-kind extras.

        ``extra`` carries the information ``KVLayout`` cannot express in
        its single-shape form (per D-open-1 option (a)): the sliding
        and global KV shapes, the sliding window, the k_eq_v flag, and
        an explicit caveat string for any consumer inspecting the
        config.
        """
        tc = Gemma4Adapter._text_config_dict(model)
        extra: dict[str, Any] = {
            "kv_layout_summary": "sliding_attention",
            "kv_layout_caveat": _KV_LAYOUT_CAVEAT,
            "sliding_kv_heads": int(tc.get("num_key_value_heads", 0) or 0),
            "sliding_head_dim": int(tc.get("head_dim", 0) or 0),
            "global_kv_heads": int(tc.get("num_global_key_value_heads", 0) or 0),
            "global_head_dim": int(tc.get("global_head_dim", 0) or 0),
            "sliding_window": int(tc.get("sliding_window", 0) or 0),
            "sliding_window_pattern": int(
                tc.get("sliding_window_pattern", 0) or 0
            ),
            "attention_k_eq_v": bool(tc.get("attention_k_eq_v", False)),
            "text_config_keys": sorted(tc.keys()),
        }
        return ModelConfig(
            model_name=str(getattr(model, "model_type", "gemma4")),
            num_layers=len(model.layers),
            hidden_size=int(tc.get("hidden_size", 0) or 0),
            vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 0),
            extra=extra,
        )

    @staticmethod
    def _build_kv_layout(
        model: Any, pattern: AttentionPattern
    ) -> KVLayout:
        """Sliding-layer summary + explicit per-kind byte budget (D-open-1 / D4).

        The four primary ``KVLayout`` fields (``num_layers``,
        ``n_kv_heads``, ``head_dim``, ``dtype``) stay populated from the
        sliding-layer fields because sliding is the majority (50 of 60
        on Gemma4-31B); homogeneous-shape consumers that read these
        directly see a reasonable single-shape summary with a caveat
        recorded in ``ModelConfig.extra``.

        ``bytes_per_token_total`` is populated explicitly (P-3-D4) by
        summing K+V contributions per attention kind using the per-kind
        shape fields from ``text_config``:

        - SLIDING: ``num_key_value_heads * head_dim``
        - GLOBAL (full attention): ``num_global_key_value_heads *
          global_head_dim``

        ``attention_k_eq_v=True`` shares the ``v_proj`` *weight* matrix
        on full-attention layers but keeps K and V as separate cache
        tensors at runtime (``gemma4_text.py`` stores both via
        ``cache.update_and_fetch(keys, values)`` after an independent
        ``v_norm`` on V). The factor 2 therefore applies uniformly.

        Caveat: this is an unbounded-window assumption — sliding
        layers are actually capped at ``sliding_window`` tokens of KV,
        so very long sequences over-estimate sliding contributions and
        under-estimate the fixed per-request sliding footprint. See
        ``plans/P3_BATCH_ROTATING_KV_SURVEY.md`` §4.4. The D4 single
        scalar is still strictly better than the pre-D4 "sliding
        shape × all 60 layers" over-count by ~9% on Gemma4-31B.
        """
        tc = Gemma4Adapter._text_config_dict(model)
        n_kv_heads = int(tc.get("num_key_value_heads", 0) or 0)
        head_dim = int(tc.get("head_dim", 0) or 0)
        dtype_name = str(tc.get("dtype", "bfloat16"))
        dtype = _DTYPE_BY_NAME.get(dtype_name, mx.bfloat16)
        dtype_bytes = dtype.size

        sliding_kv_heads = int(tc.get("num_key_value_heads", 0) or 0)
        sliding_head_dim = int(tc.get("head_dim", 0) or 0)
        global_kv_heads = int(tc.get("num_global_key_value_heads", 0) or 0)
        global_head_dim = int(tc.get("global_head_dim", 0) or 0)

        n_sliding = sum(
            1 for k in pattern.per_layer if k is AttentionKind.SLIDING
        )
        n_full = sum(
            1 for k in pattern.per_layer if k is AttentionKind.GLOBAL
        )
        bytes_per_token_total = (
            n_sliding * 2 * sliding_kv_heads * sliding_head_dim * dtype_bytes
            + n_full * 2 * global_kv_heads * global_head_dim * dtype_bytes
        )

        return KVLayout(
            num_layers=len(model.layers),
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            bytes_per_token_total=bytes_per_token_total,
        )

    @staticmethod
    def _build_attention_pattern(model: Any) -> AttentionPattern:
        """Strict mapping from ``config.layer_types`` to ``AttentionKind``.

        Three loud-fail guards:

        - ``layer_types`` must be present and non-empty. An empty or
          missing value would produce ``AttentionPattern(per_layer=())``
          whose ``attention_kinds == frozenset()`` is a subset of
          every supported set, so the capability gate would silently
          admit the adapter.
        - ``len(layer_types)`` must equal ``len(model.layers)``.
          Otherwise the scheduler walks per-layer caches with
          inconsistent indexing.
        - Unknown per-layer values raise so a new Gemma4 attention
          variant surfaces at adapter construction rather than
          decaying into a mis-kinded pattern.
        """
        tc = Gemma4Adapter._text_config_dict(model)
        layer_types = tc.get("layer_types")
        num_model_layers = len(model.layers)

        if not layer_types:
            raise NotImplementedError(
                "Gemma4Adapter (P-3-D1): config.text_config['layer_types'] "
                "is missing or empty. The adapter relies on explicit "
                "per-layer attention kinds rather than deriving them from "
                "sliding_window_pattern — an empty list would produce an "
                "empty AttentionPattern that silently passes the "
                "capability gate."
            )
        if len(layer_types) != num_model_layers:
            raise NotImplementedError(
                "Gemma4Adapter (P-3-D1): len(config.text_config["
                "'layer_types'])"
                f"={len(layer_types)} does not match len(model.layers)="
                f"{num_model_layers}. The forward walks cache entries "
                "per layer; a length mismatch desynchronises routing."
            )

        kinds: list[AttentionKind] = []
        for i, lt in enumerate(layer_types):
            if lt == "sliding_attention":
                kinds.append(AttentionKind.SLIDING)
            elif lt == "full_attention":
                kinds.append(AttentionKind.GLOBAL)
            else:
                raise NotImplementedError(
                    f"Gemma4Adapter (P-3-D1): unknown layer_types[{i}]="
                    f"{lt!r}. Supported values: "
                    "'sliding_attention', 'full_attention'."
                )
        return AttentionPattern(per_layer=tuple(kinds))
