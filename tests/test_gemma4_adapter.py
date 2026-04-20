"""Tests for silica.models.gemma4 — I-1 Gemma4Adapter (P-3-D1, 31B dense).

Uses a fake mlx-lm-gemma4-shaped model so the adapter's Silica-side
translation (ModelConfig / KVLayout / AttentionPattern / prefill /
decode / _validate_supported_variant / make_batch_cache hybrid
sliding+global wiring) is exercised without requiring the 18.4 GB
Gemma4-31B-4bit checkpoint.
End-to-end real-model smoke is P-3-D1.1 (``tests/test_p3_gemma4_*``)
and is separate from this unit file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)

from silica.kvcache.manager import KVHandle
from silica.kvcache.simple import SimpleKVCache
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelAdapter,
    ModelConfig,
    StateDelta,
)
from silica.models.capabilities import ModelCapabilities
from silica.models.factory import supported_model_types
from silica.models.gemma4 import Gemma4Adapter
from silica.weights.resident import ResidentWeightProvider

# --- Gemma4-31B-shaped fakes ---


def _default_31b_text_config() -> dict[str, Any]:
    """Gemma4-31B-4bit text_config values as observed 2026-04-20.

    Captures the concrete dense-31B shape the adapter targets. Tests
    that need to exercise variant-guard branches override one field
    at a time — see _bad_variant_config().
    """
    # 5:1 sliding:full pattern repeating — 50 sliding + 10 full over 60 layers.
    layer_types = ([
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
    ] * 10)
    return {
        "model_type": "gemma4_text",
        "num_hidden_layers": 60,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "sliding_window_pattern": 5,
        "attention_k_eq_v": True,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
        "max_position_embeddings": 262144,
        "enable_moe_block": False,
        "num_experts": None,
        "top_k_experts": None,
        "use_double_wide_mlp": False,
        "use_bidirectional_attention": "vision",
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": True,
        "dtype": "bfloat16",
        "layer_types": layer_types,
    }


@dataclass
class _FakeGemma4Args:
    model_type: str = "gemma4"
    text_config: dict[str, Any] = field(
        default_factory=_default_31b_text_config
    )
    vocab_size: int = 262144


@dataclass
class _FakeGemma4Layer:
    """Stand-in for gemma4_text.DecoderLayer (metadata only).

    The adapter inspects ``model.layers`` only for its length; layer
    internals are reached through ``args.text_config["layer_types"]``.
    Keeping this stub attribute-free avoids leaking real
    ``gemma4_text`` internals into unit tests.
    """


class _FakeGemma4Model:
    """Minimal stand-in for mlx_lm.models.gemma4.Model.

    Exposes ``model_type``, ``args``, ``layers``, and a callable
    forward that is cache-type-agnostic (does not touch the cache
    entries) — sufficient for the adapter's prefill / decode_step to
    return sensibly-shaped logits. Load-bearing fields come from
    ``args.text_config``.
    """

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vocab_size: int = 262144,
    ) -> None:
        self.model_type = "gemma4"
        self.args = _FakeGemma4Args(
            text_config=text_config or _default_31b_text_config(),
            vocab_size=vocab_size,
        )
        self._vocab_size = vocab_size
        n_layers = int(self.args.text_config.get("num_hidden_layers", 0))
        self.layers = [_FakeGemma4Layer() for _ in range(n_layers)]

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        return mx.zeros((B, T, self._vocab_size), dtype=mx.float16)


class _FakeTokenizer:
    vocab_size = 262144

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: Any) -> str:
        return "stub"


def _make_adapter_and_kv(
    text_config: dict[str, Any] | None = None,
) -> tuple[Gemma4Adapter, SimpleKVCache, _FakeGemma4Model]:
    """Build a fake 31B-shaped adapter + kv.

    The kv list mirrors Gemma4's real ``make_cache``: ``KVCache`` for
    full-attention layers, ``RotatingKVCache`` for sliding layers.
    The adapter does not inspect cache element types, but matching the
    real shape keeps any future regression test aware of the contract.
    """
    model = _FakeGemma4Model(text_config=text_config)
    tokenizer = _FakeTokenizer()
    tc = model.args.text_config
    cache_list: list[Any] = []
    for lt in tc.get("layer_types", []):
        if lt == "full_attention":
            cache_list.append(KVCache())
        else:
            cache_list.append(
                RotatingKVCache(max_size=tc.get("sliding_window", 1024), keep=0)
            )
    kv = SimpleKVCache(cache_list)
    adapter = Gemma4Adapter(model, tokenizer, kv_manager=kv)
    return adapter, kv, model


# --- I-1 Protocol shape ---


def test_adapter_satisfies_model_adapter_protocol() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter, ModelAdapter)


def test_adapter_exposes_config_attribute() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter.config, ModelConfig)
    assert adapter.config.model_name == "gemma4"
    assert adapter.config.num_layers == 60
    assert adapter.config.hidden_size == 5376
    assert adapter.config.vocab_size == 262144


# --- kv_layout: D-open-1 option (a) sliding summary + caveat in extra ---


def test_kv_layout_is_sliding_summary() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    layout = adapter.kv_layout()
    assert isinstance(layout, KVLayout)
    assert layout.num_layers == 60
    # Sliding-layer shape populates the layout (majority: 50 of 60).
    assert layout.n_kv_heads == 16
    assert layout.head_dim == 256
    assert layout.dtype == mx.bfloat16


def test_kv_layout_bytes_per_token_total_is_explicit_per_kind_sum() -> None:
    """P-3-D4: ``KVLayout.bytes_per_token_total`` is populated on
    Gemma4 with an explicit per-kind K+V contribution so
    ``MemoryBudgeter.for_adapter`` does not derive a wrong total from
    the sliding-shape summary fields.

    Formula: ``n_sliding * 2 * sliding_kv_heads * sliding_head_dim *
    dtype.size + n_full * 2 * global_kv_heads * global_head_dim *
    dtype.size`` under the unbounded-window assumption (sliding-window
    truncation is a future budget-model refinement, not D4).

    For Gemma4-31B defaults (50 sliding @ 16×256, 10 full @ 4×512,
    bfloat16=2 bytes) that is
    ``50*2*16*256*2 + 10*2*4*512*2 = 819_200 + 81_920 = 901_120``."""
    adapter, _, _ = _make_adapter_and_kv()
    layout = adapter.kv_layout()
    assert layout.bytes_per_token_total == (
        50 * 2 * 16 * 256 * 2 + 10 * 2 * 4 * 512 * 2
    )
    assert layout.bytes_per_token_total == 901_120


def test_kv_layout_bytes_per_token_total_corrects_pre_d4_overcount() -> None:
    """Pre-D4 ``MemoryBudgeter.for_adapter`` would multiply the
    sliding-shape summary over all 60 layers: ``2*60*16*256*2 =
    983_040``. D4's per-kind total is strictly smaller (~9% less);
    this test pins the improvement direction so a regression that
    restores the naive formula fails loudly."""
    adapter, _, _ = _make_adapter_and_kv()
    layout = adapter.kv_layout()
    naive = 2 * layout.num_layers * layout.n_kv_heads * layout.head_dim * layout.dtype.size
    assert layout.bytes_per_token_total is not None
    assert layout.bytes_per_token_total < naive
    assert naive - layout.bytes_per_token_total == 983_040 - 901_120  # 81_920


def test_kv_layout_bytes_per_token_total_follows_layer_type_mix() -> None:
    """Flipping a layer from sliding to full must change
    ``bytes_per_token_total`` by the per-layer delta. Guards against
    a regression where the counts are derived from total layer count
    rather than ``attention_pattern.per_layer``."""
    tc = _default_31b_text_config()
    # Convert one sliding layer into a full_attention layer: now 49
    # sliding + 11 full.
    types = list(tc["layer_types"])
    first_sliding = types.index("sliding_attention")
    types[first_sliding] = "full_attention"
    tc["layer_types"] = types

    adapter, _, _ = _make_adapter_and_kv(text_config=tc)
    layout = adapter.kv_layout()
    assert layout.bytes_per_token_total == (
        49 * 2 * 16 * 256 * 2 + 11 * 2 * 4 * 512 * 2
    )


def test_config_extra_records_per_kind_detail_and_caveat() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    extra = adapter.config.extra
    assert extra["kv_layout_summary"] == "sliding_attention"
    assert "summary" in extra["kv_layout_caveat"]
    assert extra["sliding_kv_heads"] == 16
    assert extra["sliding_head_dim"] == 256
    assert extra["global_kv_heads"] == 4
    assert extra["global_head_dim"] == 512
    assert extra["sliding_window"] == 1024
    assert extra["sliding_window_pattern"] == 5
    assert extra["attention_k_eq_v"] is True
    assert "layer_types" in extra["text_config_keys"]


# --- attention_pattern: strict layer_types → AttentionKind ---


def test_attention_pattern_maps_layer_types_to_sliding_and_global() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    pattern = adapter.attention_pattern()
    assert isinstance(pattern, AttentionPattern)
    assert len(pattern.per_layer) == 60
    sliding_count = sum(
        1 for k in pattern.per_layer if k == AttentionKind.SLIDING
    )
    global_count = sum(
        1 for k in pattern.per_layer if k == AttentionKind.GLOBAL
    )
    assert sliding_count == 50
    assert global_count == 10
    # First 6 layers match the 5:1 repeating pattern exactly.
    assert pattern.per_layer[:6] == (
        AttentionKind.SLIDING,
        AttentionKind.SLIDING,
        AttentionKind.SLIDING,
        AttentionKind.SLIDING,
        AttentionKind.SLIDING,
        AttentionKind.GLOBAL,
    )


def test_attention_pattern_raises_on_unknown_layer_type() -> None:
    """Unknown values surface immediately at construction rather than
    decaying into a mis-kinded pattern the capability gate may accept
    or reject inconsistently."""
    tc = _default_31b_text_config()
    tc["layer_types"] = tc["layer_types"][:]
    tc["layer_types"][3] = "mystery_attention"
    with pytest.raises(NotImplementedError, match="mystery_attention"):
        _make_adapter_and_kv(text_config=tc)


def test_attention_pattern_raises_on_empty_layer_types() -> None:
    """An empty layer_types list would produce an empty
    ``AttentionPattern`` whose ``attention_kinds`` is a subset of
    every supported set — the capability gate would silently admit
    the adapter. Construction must loud-fail instead."""
    tc = _default_31b_text_config()
    tc["layer_types"] = []
    with pytest.raises(NotImplementedError, match="layer_types"):
        _make_adapter_and_kv(text_config=tc)


def test_attention_pattern_raises_on_missing_layer_types() -> None:
    """Same failure mode as the empty-list case: a missing key leaves
    ``tc.get(...)`` returning ``None``, which the guard must treat as
    equally unsafe."""
    tc = _default_31b_text_config()
    del tc["layer_types"]
    with pytest.raises(NotImplementedError, match="layer_types"):
        _make_adapter_and_kv(text_config=tc)


def test_attention_pattern_raises_on_length_mismatch_vs_model_layers() -> None:
    """Cache / layer indexing walks zip(layers, cache) pairwise; a
    layer_types list shorter or longer than ``len(model.layers)``
    would desynchronise routing. Loud-fail at construction."""
    tc = _default_31b_text_config()
    tc["layer_types"] = tc["layer_types"][:30]  # half length
    with pytest.raises(NotImplementedError, match="does not match"):
        _make_adapter_and_kv(text_config=tc)


# --- capabilities (D-016) ---


def test_capabilities_declare_sliding_and_global_no_recurrent_no_moe() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert caps.attention_kinds == frozenset(
        {AttentionKind.SLIDING, AttentionKind.GLOBAL}
    )
    assert caps.has_recurrent_state is False
    assert caps.has_moe is False


# --- variant guard: loud-fail on unsupported 31B-only scope ---


def test_validate_rejects_moe_variant() -> None:
    tc = _default_31b_text_config()
    tc["enable_moe_block"] = True
    tc["num_experts"] = 8
    with pytest.raises(NotImplementedError, match="P-3-E"):
        _make_adapter_and_kv(text_config=tc)


def test_validate_rejects_num_experts_without_enable_flag() -> None:
    """``num_experts>0`` alone is enough to flag the MoE path — guards
    against checkpoints that leave ``enable_moe_block`` implicit."""
    tc = _default_31b_text_config()
    tc["enable_moe_block"] = False
    tc["num_experts"] = 128
    with pytest.raises(NotImplementedError, match="MoE variants"):
        _make_adapter_and_kv(text_config=tc)


def test_validate_rejects_per_layer_input_variant() -> None:
    tc = _default_31b_text_config()
    tc["hidden_size_per_layer_input"] = 256
    with pytest.raises(NotImplementedError, match="per_layer_inputs"):
        _make_adapter_and_kv(text_config=tc)


def test_validate_rejects_shared_kv_variant() -> None:
    tc = _default_31b_text_config()
    tc["num_kv_shared_layers"] = 20
    with pytest.raises(NotImplementedError, match="shared-KV"):
        _make_adapter_and_kv(text_config=tc)


# --- make_batch_cache: hybrid BatchRotatingKVCache + BatchKVCache list (D2) ---


def test_make_batch_cache_returns_per_layer_hybrid_list() -> None:
    """Per-layer cache list must mirror ``attention_pattern.per_layer``:
    ``BatchRotatingKVCache`` for SLIDING, ``BatchKVCache`` for GLOBAL.
    Walking ``attention_pattern`` (instead of a parallel re-derivation
    from ``layer_types``) ensures the batched factory and the
    capability gate stay in lockstep."""
    adapter, _, _ = _make_adapter_and_kv()
    caches = adapter.make_batch_cache(left_padding=[0, 1])
    assert len(caches) == 60
    pattern = adapter.attention_pattern().per_layer
    for i, (kind, cache) in enumerate(zip(pattern, caches, strict=True)):
        if kind == AttentionKind.SLIDING:
            assert isinstance(cache, BatchRotatingKVCache), (
                f"layer {i}: expected BatchRotatingKVCache, got {type(cache)}"
            )
        else:
            assert isinstance(cache, BatchKVCache), (
                f"layer {i}: expected BatchKVCache, got {type(cache)}"
            )


def test_make_batch_cache_sliding_max_size_matches_text_config() -> None:
    """``BatchRotatingKVCache.max_size`` must equal
    ``text_config['sliding_window']`` — mismatched window sizes
    silently corrupt attention masks rather than raising."""
    adapter, _, _ = _make_adapter_and_kv()
    caches = adapter.make_batch_cache(left_padding=[0, 1])
    pattern = adapter.attention_pattern().per_layer
    expected = adapter.config.extra["sliding_window"]
    for kind, cache in zip(pattern, caches, strict=True):
        if kind == AttentionKind.SLIDING:
            assert cache.max_size == expected


def test_make_batch_cache_threads_left_padding_through() -> None:
    """Both cache types must receive the caller's ``left_padding`` so
    masks account for per-row prefix padding. mlx-lm stores it as an
    ``mx.array``; we compare the ``.tolist()`` view for stable equality."""
    left_padding = [2, 0, 5]
    adapter, _, _ = _make_adapter_and_kv()
    caches = adapter.make_batch_cache(left_padding=left_padding)
    # Sample one of each kind.
    sliding_caches = [
        c for c in caches if isinstance(c, BatchRotatingKVCache)
    ]
    global_caches = [c for c in caches if isinstance(c, BatchKVCache)]
    assert sliding_caches and global_caches
    assert sliding_caches[0].left_padding.tolist() == left_padding
    assert global_caches[0].left_padding.tolist() == left_padding


def test_make_batch_cache_raises_when_sliding_window_is_zero() -> None:
    """A SLIDING-bearing pattern with ``sliding_window == 0`` is a
    config inconsistency. Passing ``max_size=0`` to
    ``BatchRotatingKVCache`` would crash opaquely deep in
    ``update_and_fetch``; surface the problem at factory time."""
    tc = _default_31b_text_config()
    tc["sliding_window"] = 0
    adapter, _, _ = _make_adapter_and_kv(text_config=tc)
    with pytest.raises(NotImplementedError, match="sliding_window"):
        adapter.make_batch_cache(left_padding=[0, 1])


# --- tokenizer / build / prefill / decode_step ---


def test_tokenizer_passes_through() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    tok = adapter.tokenizer()
    assert tok.vocab_size == 262144
    assert tok.encode("x") == [1, 2, 3]


def test_build_returns_injected_model_ignores_weight_provider() -> None:
    adapter, _, model = _make_adapter_and_kv()
    built = adapter.build(ResidentWeightProvider())
    assert built is model


def test_prefill_returns_logits_and_empty_state_delta() -> None:
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1, 2, 3])
    tokens = mx.array([1, 2, 3], dtype=mx.int32)
    logits, delta = adapter.prefill(tokens, KVHandle(req_id="req-a"))
    # ``silica.mlx.runner.forward`` slices to the last-position logit
    # of the last row, returning shape ``(vocab_size,)``.
    assert logits.shape == (adapter.config.vocab_size,)
    assert isinstance(delta, StateDelta)
    # Gemma4 is pure KV attention — recurrent_bytes stays at 0 default.
    assert delta.recurrent_bytes() == 0


def test_decode_step_returns_logits_and_empty_state_delta() -> None:
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1])
    token = mx.array([7], dtype=mx.int32)
    logits, delta = adapter.decode_step(token, KVHandle(req_id="req-a"))
    assert logits.shape == (adapter.config.vocab_size,)
    assert isinstance(delta, StateDelta)


# --- factory dispatch (P-3-D1 registers only 'gemma4') ---


def test_factory_registers_gemma4_only() -> None:
    """D1 registers the outer multimodal-shell model_type only. The
    inner ``"gemma4_text"`` checkpoint stores args differently
    (``ModelArgs`` dataclass rather than nested dict), so registering
    it without also widening ``_text_config_dict`` would produce an
    adapter whose variant guard and layer_types reader see an empty
    config. Add the alias when a bare text-model is a real target."""
    supported = supported_model_types()
    assert "gemma4" in supported
    assert "gemma4_text" not in supported


def test_factory_still_supports_qwen3_families() -> None:
    """Regression guard for existing families after registering Gemma4."""
    supported = supported_model_types()
    assert "qwen3" in supported
    assert "qwen3_5" in supported
