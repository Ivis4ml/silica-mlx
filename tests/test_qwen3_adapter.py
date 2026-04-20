"""Tests for silica.models.qwen3 — I-1 Qwen3Adapter (plain KV).

Uses a fake mlx-lm-qwen3-shaped model. Real-model end-to-end load +
P-1 Engine.generate baseline is in ``scripts/probe_p2_preload.py``.
Hybrid Qwen3.5 tests live in ``tests/test_qwen3_5_adapter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

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
from silica.models.qwen3 import Qwen3Adapter
from silica.weights.resident import ResidentWeightProvider

# --- plain-Qwen3-shaped fakes ---


@dataclass
class _PlainSelfAttn:
    """Mirrors plain Qwen3 Attention: ``n_kv_heads`` (NOT ``num_key_value_heads``)."""

    n_kv_heads: int = 8


@dataclass
class _PlainLayer:
    self_attn: _PlainSelfAttn | None = None


@dataclass
class _PlainArgs:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    head_dim: int = 128  # Qwen3-0.6B uses head_dim=128, not hidden/heads=64


class _PlainQwenModel:
    """Minimal stand-in for mlx_lm.models.qwen3.Model (plain Qwen3)."""

    VOCAB = 32

    def __init__(self, n_layers: int = 4) -> None:
        self.model_type = "qwen3"
        self.args = _PlainArgs()
        self.layers = [
            _PlainLayer(self_attn=_PlainSelfAttn()) for _ in range(n_layers)
        ]

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        if cache is not None:
            for i in range(len(self.layers)):
                if cache[i] is not None:
                    n_kv = self.args.__dict__  # for mypy — use args fields
                    k = mx.zeros(
                        (B, self.args.hidden_size // self.args.head_dim,
                         T, self.args.head_dim),
                        dtype=mx.float16,
                    )
                    v = mx.zeros(k.shape, dtype=mx.float16)
                    del n_kv
                    cache[i].update_and_fetch(k, v)
        return mx.zeros((B, T, self.VOCAB), dtype=mx.float16)


class _FakeTokenizer:
    vocab_size = 32

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: Any) -> str:
        return "stub"


def _make_adapter_and_kv(
    n_layers: int = 4,
) -> tuple[Qwen3Adapter, SimpleKVCache, _PlainQwenModel]:
    model = _PlainQwenModel(n_layers=n_layers)
    tokenizer = _FakeTokenizer()
    kv = SimpleKVCache([KVCache() for _ in range(n_layers)])
    adapter = Qwen3Adapter(model, tokenizer, kv_manager=kv)
    return adapter, kv, model


# --- I-1 Protocol shape ---


def test_adapter_satisfies_model_adapter_protocol() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter, ModelAdapter)


def test_adapter_exposes_config_attribute_from_flat_args() -> None:
    """Plain Qwen3 has ``hidden_size`` on ``model.args`` (not in text_config)."""
    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter.config, ModelConfig)
    assert adapter.config.model_name == "qwen3"
    assert adapter.config.num_layers == 4
    assert adapter.config.hidden_size == 1024
    assert adapter.config.vocab_size == 32


# --- kv_layout (plain-Qwen3 naming) ---


def test_kv_layout_reads_n_kv_heads_attribute() -> None:
    """Qwen3-0.6B uses ``self_attn.n_kv_heads``, not ``num_key_value_heads``."""
    adapter, _, _ = _make_adapter_and_kv()
    layout = adapter.kv_layout()
    assert isinstance(layout, KVLayout)
    assert layout.num_layers == 4
    assert layout.n_kv_heads == 8
    assert layout.head_dim == 128  # from args.head_dim, NOT hidden/heads=64
    assert layout.dtype == mx.float16


def test_kv_layout_falls_back_to_hidden_div_heads_when_head_dim_absent() -> None:
    """When args.head_dim is absent, derive from hidden_size / num_attention_heads."""

    @dataclass
    class _ArgsNoHeadDim:
        hidden_size: int = 1024
        num_attention_heads: int = 16

    class _Model:
        def __init__(self) -> None:
            self.model_type = "qwen3"
            self.args = _ArgsNoHeadDim()
            self.layers = [_PlainLayer(self_attn=_PlainSelfAttn()) for _ in range(4)]

    kv = SimpleKVCache([KVCache() for _ in range(4)])
    adapter = Qwen3Adapter(_Model(), _FakeTokenizer(), kv_manager=kv)
    assert adapter.kv_layout().head_dim == 64  # 1024 // 16


def test_kv_layout_returns_zeros_for_empty_layer_stack() -> None:
    class _EmptyModel:
        def __init__(self) -> None:
            self.model_type = "qwen3"
            self.args = _PlainArgs()
            self.layers: list[Any] = []

    kv = SimpleKVCache([])
    adapter = Qwen3Adapter(_EmptyModel(), _FakeTokenizer(), kv_manager=kv)
    layout = adapter.kv_layout()
    assert layout.num_layers == 0
    assert layout.n_kv_heads == 0
    assert layout.head_dim == 0


# --- attention_pattern: plain Qwen3 is all GLOBAL, no hybrid ---


def test_attention_pattern_all_global() -> None:
    adapter, _, _ = _make_adapter_and_kv(n_layers=6)
    pattern = adapter.attention_pattern()
    assert isinstance(pattern, AttentionPattern)
    assert pattern.per_layer == tuple(AttentionKind.GLOBAL for _ in range(6))


def test_attention_pattern_contains_no_hybrid_deltanet() -> None:
    adapter, _, _ = _make_adapter_and_kv(n_layers=28)
    kinds = set(adapter.attention_pattern().per_layer)
    assert AttentionKind.HYBRID_DELTANET not in kinds
    assert AttentionKind.RECURRENT not in kinds


# --- capabilities (D-016) ---


def test_capabilities_is_pure_global_no_recurrent_no_moe() -> None:
    adapter, _, _ = _make_adapter_and_kv(n_layers=6)
    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert caps.attention_kinds == frozenset({AttentionKind.GLOBAL})
    assert caps.has_recurrent_state is False
    assert caps.has_moe is False


# --- make_batch_cache (P-3-C3a) ---


def test_make_batch_cache_returns_all_batch_kv_cache() -> None:
    """Plain Qwen3 is pure GQA — every layer is a BatchKVCache with the
    shared left_padding. Length matches ``config.num_layers``."""
    from mlx_lm.models.cache import BatchKVCache

    adapter, _, _ = _make_adapter_and_kv(n_layers=6)
    caches = adapter.make_batch_cache(left_padding=[0, 1, 2])
    assert len(caches) == adapter.config.num_layers == 6
    assert all(isinstance(c, BatchKVCache) for c in caches)


# --- tokenizer / build ---


def test_tokenizer_passes_through() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    tok = adapter.tokenizer()
    assert tok.vocab_size == 32
    assert tok.encode("x") == [1, 2, 3]


def test_build_returns_injected_model_ignores_weight_provider() -> None:
    adapter, _, model = _make_adapter_and_kv()
    built = adapter.build(ResidentWeightProvider())
    assert built is model


# --- prefill / decode_step end-to-end (with SimpleKVCache wiring) ---


def test_prefill_returns_logits_and_state_delta() -> None:
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1, 2, 3])
    handle = KVHandle(req_id="req-a")
    tokens = mx.array([1, 2, 3], dtype=mx.int32)
    logits, delta = adapter.prefill(tokens, handle)
    assert logits.shape == (_PlainQwenModel.VOCAB,)
    assert isinstance(delta, StateDelta)
    assert delta.recurrent_bytes() == 0


def test_decode_step_returns_logits_and_state_delta() -> None:
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1])
    handle = KVHandle(req_id="req-a")
    token = mx.array([7], dtype=mx.int32)
    logits, delta = adapter.decode_step(token, handle)
    assert logits.shape == (_PlainQwenModel.VOCAB,)
    assert isinstance(delta, StateDelta)


def test_prefill_then_decode_accumulates_cache() -> None:
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1, 2, 3])
    handle = KVHandle(req_id="req-a")
    adapter.prefill(mx.array([1, 2, 3], dtype=mx.int32), handle)
    adapter.decode_step(mx.array([4], dtype=mx.int32), handle)
    cache_list = kv.cache_list("req-a")
    # All layers use full-attention KVCache; every entry's offset == 4.
    for c in cache_list:
        assert c.offset == 4


def test_prefill_requires_kv_handle_owner() -> None:
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1])
    handle = KVHandle(req_id="req-b")
    with pytest.raises(ValueError, match="req_id mismatch"):
        adapter.prefill(mx.array([1], dtype=mx.int32), handle)
