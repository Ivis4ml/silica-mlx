"""Tests for silica.models.qwen3_5 — I-1 Qwen3_5Adapter (hybrid, D-015).

Uses a fake mlx-lm-qwen3_5-shaped model so the adapter's Silica-side
translation (ModelConfig / KVLayout / AttentionPattern / prefill /
decode) is exercised without requiring the 1.77 GB Qwen3.5-0.8B
checkpoint. End-to-end load + greedy parity with mlx-lm is the P-1
acceptance test (scripts/acceptance_p1_mlx_lm_parity.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from silica.models.qwen3_5 import Qwen3_5Adapter
from silica.weights.resident import ResidentWeightProvider

# --- fakes that mirror the shape of mlx-lm's qwen3_5.Model ---


@dataclass
class _FakeSelfAttn:
    num_key_value_heads: int = 2
    head_dim: int = 8


@dataclass
class _FakeLayer:
    is_linear: bool = False
    self_attn: _FakeSelfAttn | None = None


@dataclass
class _FakeArgs:
    text_config: dict[str, Any] = field(
        default_factory=lambda: {"hidden_size": 1024, "num_hidden_layers": 4}
    )


class _FakeQwen35Model:
    """Minimal stand-in for mlx_lm.models.qwen3_5.Model.

    Has the attributes the adapter inspects (``model_type``, ``args``,
    ``layers``) and a callable forward that updates the per-layer cache
    entries for non-linear layers so ``runner.forward`` exercise is
    realistic.
    """

    def __init__(
        self,
        n_linear: int = 2,
        n_full: int = 2,
        vocab_size: int = 32,
        n_kv_heads: int = 2,
        head_dim: int = 8,
    ) -> None:
        self.model_type = "qwen3_5"
        self.args = _FakeArgs()
        self._vocab_size = vocab_size
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self.layers = [
            _FakeLayer(is_linear=True) for _ in range(n_linear)
        ] + [
            _FakeLayer(
                is_linear=False,
                self_attn=_FakeSelfAttn(
                    num_key_value_heads=n_kv_heads, head_dim=head_dim
                ),
            )
            for _ in range(n_full)
        ]

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        if cache is not None:
            for i, layer in enumerate(self.layers):
                if not layer.is_linear and cache[i] is not None:
                    k = mx.zeros(
                        (B, self._n_kv_heads, T, self._head_dim),
                        dtype=mx.float16,
                    )
                    v = mx.zeros(
                        (B, self._n_kv_heads, T, self._head_dim),
                        dtype=mx.float16,
                    )
                    cache[i].update_and_fetch(k, v)
        return mx.zeros((B, T, self._vocab_size), dtype=mx.float16)


class _FakeTokenizer:
    vocab_size = 32

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: Any) -> str:
        return "stub"


def _make_adapter_and_kv() -> tuple[Qwen3_5Adapter, SimpleKVCache, _FakeQwen35Model]:
    model = _FakeQwen35Model(n_linear=2, n_full=2)
    tokenizer = _FakeTokenizer()
    # Hand-crafted cache list: 4 entries, all KVCache for simplicity.
    # (ArraysCache would also work; the adapter does not inspect types.)
    kv = SimpleKVCache([KVCache(), KVCache(), KVCache(), KVCache()])
    adapter = Qwen3_5Adapter(model, tokenizer, kv_manager=kv)
    return adapter, kv, model


# --- I-1 Protocol shape ---


def test_adapter_satisfies_model_adapter_protocol() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter, ModelAdapter)


def test_adapter_exposes_config_attribute() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter.config, ModelConfig)
    assert adapter.config.model_name == "qwen3_5"
    assert adapter.config.num_layers == 4
    assert adapter.config.hidden_size == 1024
    assert adapter.config.vocab_size == 32


# --- kv_layout (reads nested text_config / num_key_value_heads / head_dim) ---


def test_kv_layout_reads_from_first_full_attention_layer() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    layout = adapter.kv_layout()
    assert isinstance(layout, KVLayout)
    assert layout.num_layers == 4
    assert layout.n_kv_heads == 2
    assert layout.head_dim == 8
    assert layout.dtype == mx.float16


def test_kv_layout_all_linear_returns_zeros() -> None:
    """Pure-DeltaNet hypothetical model: no full-attention, KVLayout zeroes."""
    model = _FakeQwen35Model(n_linear=3, n_full=0)
    kv = SimpleKVCache([KVCache(), KVCache(), KVCache()])
    adapter = Qwen3_5Adapter(model, _FakeTokenizer(), kv_manager=kv)
    layout = adapter.kv_layout()
    assert layout.num_layers == 3
    assert layout.n_kv_heads == 0
    assert layout.head_dim == 0


# --- attention_pattern (D-015) ---


def test_attention_pattern_maps_is_linear_to_hybrid_deltanet() -> None:
    """Linear layers → HYBRID_DELTANET; full → GLOBAL."""
    adapter, _, _ = _make_adapter_and_kv()
    pattern = adapter.attention_pattern()
    assert isinstance(pattern, AttentionPattern)
    # 2 linear + 2 full, order preserved
    assert pattern.per_layer == (
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.GLOBAL,
        AttentionKind.GLOBAL,
    )


def test_attention_pattern_length_matches_num_layers() -> None:
    adapter, _, _ = _make_adapter_and_kv()
    pattern = adapter.attention_pattern()
    assert len(pattern.per_layer) == adapter.config.num_layers


# --- capabilities (D-016) ---


def test_capabilities_carry_hybrid_deltanet_and_set_recurrent_state() -> None:
    """Hybrid Qwen3.5 fake has 2 linear + 2 full layers; capabilities
    must reflect HYBRID_DELTANET presence and has_recurrent_state=True."""
    adapter, _, _ = _make_adapter_and_kv()
    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert AttentionKind.HYBRID_DELTANET in caps.attention_kinds
    assert AttentionKind.GLOBAL in caps.attention_kinds
    assert caps.has_recurrent_state is True
    # MoE A3B variant will live in a separate adapter; the dense
    # Qwen3.5 adapter declares has_moe=False.
    assert caps.has_moe is False


# --- D-015 adapter-local state helpers (P-3-C1) ---


def test_commit_state_is_a_noop_returning_none() -> None:
    """Under mlx-lm's in-place forward, commit_state has nothing to do."""
    adapter, _, _ = _make_adapter_and_kv()
    assert adapter.commit_state("req-0", 0) is None
    assert adapter.commit_state("req-0", 7) is None


def test_rollback_state_raises_not_implemented_naming_p7() -> None:
    """Rollback requires a pre-draft snapshot that P-3 does not take;
    raising is intentional so a caller that wires in a draft source
    without the snapshot pathway fails loudly rather than silently
    corrupting decoding."""
    adapter, _, _ = _make_adapter_and_kv()
    with pytest.raises(NotImplementedError, match="P-7"):
        adapter.rollback_state("req-0", 3)


def test_state_from_prefix_returns_none_for_any_input() -> None:
    """D-015 v0.1 rule: reuse recurrent state only when the full KV
    prefix is reused; the adapter has no way to distinguish that today,
    so it conservatively returns None for every prefix request."""
    adapter, _, _ = _make_adapter_and_kv()
    assert adapter.state_from_prefix("req-0", []) is None
    assert adapter.state_from_prefix("req-0", [1, 2, 3]) is None
    assert adapter.state_from_prefix("req-0", list(range(1024))) is None


def test_free_state_is_a_noop_returning_none() -> None:
    """The SimpleKVCache already discards the per-request ArraysCache
    slots via KVManager.free; the adapter has no second tenant to
    release in the single-request path."""
    adapter, _, _ = _make_adapter_and_kv()
    assert adapter.free_state("req-0") is None


def test_helpers_signatures_match_d015_pairing() -> None:
    """D-015 pairs commit_state / rollback_state with KVManager.commit
    / KVManager.rollback, and free_state with KVManager.free. Verify
    the helpers accept the same ``(req_id, n)`` / ``(req_id,)``
    signatures so the engine can call them symmetrically without
    type-juggling."""
    import inspect

    adapter, _, _ = _make_adapter_and_kv()
    commit_params = list(inspect.signature(adapter.commit_state).parameters)
    rollback_params = list(inspect.signature(adapter.rollback_state).parameters)
    state_prefix_params = list(
        inspect.signature(adapter.state_from_prefix).parameters
    )
    free_params = list(inspect.signature(adapter.free_state).parameters)
    assert commit_params == ["req_id", "n_accepted"]
    assert rollback_params == ["req_id", "n_reject"]
    assert state_prefix_params == ["req_id", "token_ids"]
    assert free_params == ["req_id"]


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
    assert logits.shape == (adapter.config.vocab_size,)
    assert isinstance(delta, StateDelta)
    assert delta.recurrent_bytes() == 0


def test_decode_step_returns_logits_and_state_delta() -> None:
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1])
    handle = KVHandle(req_id="req-a")
    token = mx.array([7], dtype=mx.int32)
    logits, delta = adapter.decode_step(token, handle)
    assert logits.shape == (adapter.config.vocab_size,)
    assert isinstance(delta, StateDelta)


def test_prefill_then_decode_accumulates_cache() -> None:
    """After prefill (T=3) + decode (T=1), full-attention cache offsets == 4."""
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1, 2, 3])
    handle = KVHandle(req_id="req-a")
    adapter.prefill(mx.array([1, 2, 3], dtype=mx.int32), handle)
    adapter.decode_step(mx.array([4], dtype=mx.int32), handle)
    cache_list = kv.cache_list("req-a")
    # Last two entries are the full-attention KVCaches.
    assert cache_list[2].offset == 4
    assert cache_list[3].offset == 4


def test_prefill_requires_kv_handle_owner() -> None:
    """If handle's req_id doesn't match SimpleKVCache owner, access fails."""
    adapter, kv, _ = _make_adapter_and_kv()
    kv.reserve_for_prefill("req-a", [1])
    handle = KVHandle(req_id="req-b")
    with pytest.raises(ValueError, match="req_id mismatch"):
        adapter.prefill(mx.array([1], dtype=mx.int32), handle)
