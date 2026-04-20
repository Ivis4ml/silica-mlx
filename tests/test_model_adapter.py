"""Tests for silica.models.adapter — I-1 contract + D-015 auxiliary types.

Covers:
  - I-1 Protocol shape (@runtime_checkable sanity + required attrs).
  - D-015 AttentionKind enum carries the hybrid_deltanet values.
  - D-015 StateDelta is frozen, read-only, exposes only recurrent_bytes().
  - StubModelAdapter returns correctly-shaped logits and empty StateDelta.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.kvcache.manager import KVHandle
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelAdapter,
    ModelConfig,
    StateDelta,
    StubModelAdapter,
    Tokenizer,
)
from silica.weights.resident import ResidentWeightProvider


@pytest.fixture
def adapter() -> StubModelAdapter:
    return StubModelAdapter(num_layers=4, vocab_size=16)


# --- I-1 Protocol shape ---


def test_stub_satisfies_model_adapter_protocol(adapter: StubModelAdapter) -> None:
    assert isinstance(adapter, ModelAdapter)


def test_stub_exposes_config_attr(adapter: StubModelAdapter) -> None:
    assert isinstance(adapter.config, ModelConfig)
    assert adapter.config.num_layers == 4
    assert adapter.config.vocab_size == 16


# --- kv_layout ---


def test_kv_layout_matches_config(adapter: StubModelAdapter) -> None:
    layout = adapter.kv_layout()
    assert isinstance(layout, KVLayout)
    assert layout.num_layers == 4
    assert layout.dtype == mx.float16


# --- attention pattern (D-015 enum + per-layer coverage) ---


def test_attention_pattern_has_entry_per_layer(adapter: StubModelAdapter) -> None:
    pattern = adapter.attention_pattern()
    assert isinstance(pattern, AttentionPattern)
    assert len(pattern.per_layer) == adapter.config.num_layers


def test_attention_kind_enum_carries_d015_values() -> None:
    """D-015 requires both recurrent (pure linear) and hybrid_deltanet (Qwen3.5 stack)."""
    values = {k.value for k in AttentionKind}
    required = {"global", "sliding", "hybrid", "recurrent", "hybrid_deltanet"}
    assert required.issubset(values)


# --- tokenizer ---


def test_stub_tokenizer_conforms(adapter: StubModelAdapter) -> None:
    tok = adapter.tokenizer()
    assert isinstance(tok, Tokenizer)
    assert tok.vocab_size == 16


# --- prefill / decode return shapes ---


def test_prefill_returns_logits_and_state_delta(adapter: StubModelAdapter) -> None:
    tokens = mx.array([1, 2, 3], dtype=mx.int32)
    handle = KVHandle(req_id="req-a")
    logits, delta = adapter.prefill(tokens, handle)
    assert logits.shape == (adapter.config.vocab_size,)
    assert isinstance(delta, StateDelta)


def test_decode_step_returns_logits_and_state_delta(adapter: StubModelAdapter) -> None:
    token = mx.array([7], dtype=mx.int32)
    handle = KVHandle(req_id="req-a")
    logits, delta = adapter.decode_step(token, handle)
    assert logits.shape == (adapter.config.vocab_size,)
    assert isinstance(delta, StateDelta)


# --- D-015 StateDelta contract: read-only, recurrent_bytes is the only public method ---


def test_state_delta_recurrent_bytes_default_zero() -> None:
    assert StateDelta().recurrent_bytes() == 0


def test_state_delta_carries_recurrent_bytes() -> None:
    sd = StateDelta(_recurrent_bytes=4096)
    assert sd.recurrent_bytes() == 4096


def test_state_delta_is_frozen() -> None:
    sd = StateDelta(_recurrent_bytes=1024)
    with pytest.raises(Exception):
        sd._recurrent_bytes = 2048  # type: ignore[misc]


# --- build ---


def test_build_accepts_a_weight_provider(adapter: StubModelAdapter) -> None:
    # Stub tolerates any WeightProvider, returns a sentinel module.
    module = adapter.build(ResidentWeightProvider())
    assert module is not None


# --- make_batch_cache (P-3-C3a) ---


def test_stub_make_batch_cache_returns_all_batch_kv_cache(
    adapter: StubModelAdapter,
) -> None:
    """Stub has no real forward; ``make_batch_cache`` still matches the
    scheduler's shape expectation — one ``BatchKVCache`` per layer with
    the shared ``left_padding``."""
    from mlx_lm.models.cache import BatchKVCache

    caches = adapter.make_batch_cache(left_padding=[0, 1, 2])
    assert len(caches) == adapter.config.num_layers
    assert all(isinstance(c, BatchKVCache) for c in caches)
