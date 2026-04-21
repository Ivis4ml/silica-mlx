"""Tests for silica.models.gemma4_moe — I-1 Gemma4MoeAdapter (P-3-E1.2).

Uses a fake Gemma4-MoE-shaped model so the adapter's Silica-side
contract (capabilities override, MoE-aware variant guard,
factory-branch dispatch, per-kind KV budget inheritance, proxy
walk over layer.experts.switch_glu) is exercised without loading
the ~16 GB Gemma4-26B-A4B-4bit checkpoint. Real-model forward
smoke is P-3-E3 territory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache

from silica.kvcache.manager import KVHandle
from silica.kvcache.simple import SimpleKVCache
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    ModelAdapter,
    ModelConfig,
    StateDelta,
)
from silica.models.capabilities import ModelCapabilities
from silica.models.factory import adapter_from_loaded_model
from silica.models.gemma4 import Gemma4Adapter
from silica.models.gemma4_moe import Gemma4MoeAdapter
from silica.models.qwen3_5_moe import _DispatchProxy

# --- Gemma4-MoE-shaped fakes (matches 26B-A4B probe, 2026-04-20) -------------


def _default_26b_a4b_text_config() -> dict[str, Any]:
    """Gemma4-26B-A4B-4bit text_config values as observed 2026-04-20.

    30 layers with a 5:1 sliding:full repeating pattern gives 25
    sliding + 5 full — matches the real checkpoint exactly so the
    KV-budget test can pin 225,280 bytes/token without hedging.
    """
    layer_types = ([
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
    ] * 5)
    return {
        "model_type": "gemma4_text",
        "num_hidden_layers": 30,
        "hidden_size": 2816,
        "intermediate_size": 11264,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "sliding_window_pattern": None,
        "attention_k_eq_v": True,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "vocab_size": 262144,
        "max_position_embeddings": 262144,
        # MoE-specific — real 26B-A4B values.
        "enable_moe_block": True,
        "num_experts": 128,
        "top_k_experts": 8,
        "moe_intermediate_size": 704,
        "dtype": "bfloat16",
        "layer_types": layer_types,
    }


def _dense_gemma4_text_config() -> dict[str, Any]:
    """Minimal dense Gemma4 config for the factory-branch regression
    test — enable_moe_block=False + num_experts=None should route
    through Gemma4Adapter, not Gemma4MoeAdapter."""
    return {
        "model_type": "gemma4_text",
        "num_hidden_layers": 6,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "num_global_key_value_heads": 2,
        "head_dim": 64,
        "global_head_dim": 128,
        "sliding_window": 256,
        "attention_k_eq_v": False,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "vocab_size": 8,
        "dtype": "bfloat16",
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
        ],
        "enable_moe_block": False,
        "num_experts": None,
    }


@dataclass
class _FakeArgs:
    model_type: str = "gemma4"
    text_config: dict[str, Any] = field(
        default_factory=_default_26b_a4b_text_config
    )
    vocab_size: int = 262144


class _FakeSwitchGLU:
    """Stand-in for SwitchGLU with GeGLU activation.

    Records (x, indices) calls; returns a sentinel so the proxy
    delegation test can assert pass-through.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.sentinel = object()

    def __call__(self, x: Any, indices: Any) -> Any:
        self.calls.append((x, indices))
        return self.sentinel


class _FakeExperts:
    """Stand-in for gemma4_text.Experts (wraps SwitchGLU)."""

    def __init__(self) -> None:
        self.switch_glu = _FakeSwitchGLU()


class _FakeRouter:
    """Stand-in for gemma4_text.Router (not exercised directly, but
    present so tests match the real layer structure)."""


class _FakeDenseMLP:
    """Stand-in for the always-on dense MLP branch."""


class _FakeGemma4MoELayer:
    """Gemma4-MoE decoder layer: carries BOTH a dense ``mlp`` (always-
    on additive branch) AND ``router`` + ``experts.switch_glu``."""

    def __init__(self) -> None:
        self.mlp = _FakeDenseMLP()
        self.router = _FakeRouter()
        self.experts = _FakeExperts()


class _FakeGemma4DenseLayer:
    """Dense Gemma4 decoder layer (no router, no experts). Used by
    the factory-branch regression test to prove dispatch goes to
    Gemma4Adapter, not Gemma4MoeAdapter, when enable_moe_block=False."""

    def __init__(self) -> None:
        self.mlp = _FakeDenseMLP()


class _FakeGemma4MoEModel:
    def __init__(
        self, text_config: dict[str, Any] | None = None
    ) -> None:
        self.model_type = "gemma4"
        tc = text_config or _default_26b_a4b_text_config()
        self.args = _FakeArgs(text_config=tc)
        self._vocab_size = int(tc.get("vocab_size", 262144))
        n_layers = int(tc.get("num_hidden_layers", 30))
        is_moe = bool(tc.get("enable_moe_block", False))
        layer_cls = _FakeGemma4MoELayer if is_moe else _FakeGemma4DenseLayer
        self.layers = [layer_cls() for _ in range(n_layers)]

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


def _make_adapter(
    text_config: dict[str, Any] | None = None,
) -> tuple[Gemma4MoeAdapter, SimpleKVCache, _FakeGemma4MoEModel]:
    model = _FakeGemma4MoEModel(text_config=text_config)
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
    adapter = Gemma4MoeAdapter(model, tokenizer, kv_manager=kv)
    return adapter, kv, model


# --- I-1 Protocol conformance -------------------------------------------------


def test_adapter_satisfies_model_adapter_protocol() -> None:
    adapter, _, _ = _make_adapter()
    assert isinstance(adapter, ModelAdapter)


def test_adapter_exposes_config_attribute() -> None:
    adapter, _, _ = _make_adapter()
    assert isinstance(adapter.config, ModelConfig)
    assert adapter.config.num_layers == 30
    assert adapter.config.hidden_size == 2816


# --- capabilities: has_moe=True override, has_recurrent_state=False ----------


def test_capabilities_declare_has_moe_true() -> None:
    adapter, _, _ = _make_adapter()
    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert caps.has_moe is True
    # Gemma4 is pure KV attention — no recurrent state.
    assert caps.has_recurrent_state is False


def test_capabilities_inherit_sliding_and_global_attention_kinds() -> None:
    """Attention kinds come from the inherited sliding+full hybrid
    pattern; MoE does not change attention routing."""
    adapter, _, _ = _make_adapter()
    caps = adapter.capabilities()
    assert caps.attention_kinds == frozenset(
        {AttentionKind.SLIDING, AttentionKind.GLOBAL}
    )


def test_attention_pattern_inherits_25_sliding_plus_5_full() -> None:
    """5:1 repeating pattern over 30 layers → 25 sliding + 5 full.
    Matches the real Gemma4-26B-A4B-4bit checkpoint."""
    adapter, _, _ = _make_adapter()
    pattern = adapter.attention_pattern()
    assert isinstance(pattern, AttentionPattern)
    assert len(pattern.per_layer) == 30
    sliding_count = sum(
        1 for k in pattern.per_layer if k == AttentionKind.SLIDING
    )
    global_count = sum(
        1 for k in pattern.per_layer if k == AttentionKind.GLOBAL
    )
    assert sliding_count == 25
    assert global_count == 5


# --- KV layout inherits D4 per-kind formula ----------------------------------


def test_kv_layout_bytes_per_token_total_matches_26b_a4b_real_shape() -> None:
    """D4's per-kind KV budget generalises to Gemma4-26B-A4B
    unchanged:
      25 × 2 × 8 × 256 × 2 (sliding)  + 5 × 2 × 2 × 512 × 2 (full)
      = 204,800 + 20,480 = 225,280 bytes/token.
    Pins the E0 survey §3.2 number."""
    adapter, _, _ = _make_adapter()
    layout = adapter.kv_layout()
    assert layout.bytes_per_token_total == (
        25 * 2 * 8 * 256 * 2 + 5 * 2 * 2 * 512 * 2
    )
    assert layout.bytes_per_token_total == 225_280


def test_kv_layout_num_layers_matches_model() -> None:
    adapter, _, _ = _make_adapter()
    layout = adapter.kv_layout()
    assert layout.num_layers == 30


# --- MoE-specific config.extra metadata --------------------------------------


def test_config_extra_records_moe_metadata() -> None:
    """MoE-specific fields land on config.extra so downstream
    consumers (bench, future MoE scheduler, dispatch seam) can read
    them without re-parsing text_config."""
    adapter, _, _ = _make_adapter()
    extra = adapter.config.extra
    assert extra["num_experts"] == 128
    assert extra["top_k_experts"] == 8
    assert extra["moe_intermediate_size"] == 704
    assert extra["is_moe_adapter"] is True


def test_config_extra_records_always_on_dense_mlp_property() -> None:
    """Distinguishing Gemma4-MoE structural fact: the dense MLP
    branch is always-on, summed ungated with the experts branch.
    Unlike Qwen3.5-MoE where SparseMoeBlock REPLACES the dense MLP.
    E1.2 records this on config.extra so downstream consumers do
    not re-derive it from mlx-lm source."""
    adapter, _, _ = _make_adapter()
    assert adapter.config.extra["has_always_on_dense_mlp"] is True


def test_config_extra_records_moe_expert_path() -> None:
    """Downstream tooling reaching the sparse block uses the dotted
    path recorded in config.extra rather than re-deriving it from
    source. Gemma4 path differs from Qwen3.5's."""
    adapter, _, _ = _make_adapter()
    assert (
        adapter.config.extra["moe_expert_path"]
        == "layer.experts.switch_glu"
    )


# --- MoE-aware variant guard (polymorphic override) --------------------------


def test_variant_guard_rejects_enable_moe_block_false() -> None:
    """This adapter is for MoE checkpoints only; enable_moe_block=False
    routes to Gemma4Adapter via the factory, not here."""
    tc = _default_26b_a4b_text_config()
    tc["enable_moe_block"] = False
    with pytest.raises(NotImplementedError, match="enable_moe_block"):
        _make_adapter(text_config=tc)


def test_variant_guard_rejects_num_experts_zero() -> None:
    tc = _default_26b_a4b_text_config()
    tc["num_experts"] = 0
    with pytest.raises(NotImplementedError, match="num_experts"):
        _make_adapter(text_config=tc)


def test_variant_guard_rejects_top_k_zero() -> None:
    tc = _default_26b_a4b_text_config()
    tc["top_k_experts"] = 0
    with pytest.raises(NotImplementedError, match="top_k_experts"):
        _make_adapter(text_config=tc)


def test_variant_guard_rejects_top_k_greater_than_experts() -> None:
    tc = _default_26b_a4b_text_config()
    tc["num_experts"] = 4
    tc["top_k_experts"] = 8
    with pytest.raises(NotImplementedError, match="<= num_experts"):
        _make_adapter(text_config=tc)


def test_variant_guard_rejects_per_layer_input_variant() -> None:
    """Same rejection as dense Gemma4Adapter — per-layer-input
    plumbing orthogonal to MoE."""
    tc = _default_26b_a4b_text_config()
    tc["hidden_size_per_layer_input"] = 256
    with pytest.raises(NotImplementedError, match="per_layer_inputs"):
        _make_adapter(text_config=tc)


def test_variant_guard_rejects_shared_kv_variant() -> None:
    """Same rejection as dense Gemma4Adapter — shared-KV routing
    orthogonal to MoE."""
    tc = _default_26b_a4b_text_config()
    tc["num_kv_shared_layers"] = 10
    with pytest.raises(NotImplementedError, match="shared-KV"):
        _make_adapter(text_config=tc)


# --- Dense Gemma4Adapter still rejects MoE (defence in depth) ----------------


def test_dense_adapter_still_rejects_moe_directly() -> None:
    """Constructing Gemma4Adapter directly on a MoE checkpoint must
    still loud-fail. Factory routing via _build_gemma4 is the
    normal path; the dense adapter guard is defence-in-depth that
    protects against manual misuse (e.g. an Engine harness that
    bypasses the factory). P-3-E1.2 does NOT relax the dense
    guard."""
    model = _FakeGemma4MoEModel()
    tokenizer = _FakeTokenizer()
    tc = model.args.text_config
    kv = SimpleKVCache([KVCache() for _ in range(tc["num_hidden_layers"])])
    with pytest.raises(NotImplementedError, match="MoE variants"):
        Gemma4Adapter(model, tokenizer, kv_manager=kv)


# --- make_batch_cache inheritance --------------------------------------------


def test_make_batch_cache_inherits_sliding_plus_full_mixing() -> None:
    """Inherited from Gemma4Adapter.make_batch_cache (D2):
    SLIDING → BatchRotatingKVCache, GLOBAL → BatchKVCache. MoE
    routing does not touch KV cache typing."""
    adapter, _, _ = _make_adapter()
    caches = adapter.make_batch_cache(left_padding=[0, 1])
    assert len(caches) == 30
    pattern = adapter.attention_pattern().per_layer
    for idx, (kind, cache) in enumerate(zip(pattern, caches, strict=True)):
        if kind == AttentionKind.SLIDING:
            assert isinstance(cache, BatchRotatingKVCache), (
                f"layer {idx}: expected BatchRotatingKVCache, "
                f"got {type(cache).__name__}"
            )
        else:
            assert isinstance(cache, BatchKVCache), (
                f"layer {idx}: expected BatchKVCache, "
                f"got {type(cache).__name__}"
            )


# --- install_dispatch_proxy walks layer.experts.switch_glu -------------------


def test_install_dispatch_proxy_wraps_every_moe_layer_at_experts_path() -> None:
    """Gemma4-MoE's proxy walk targets ``layer.experts.switch_glu``
    (not ``layer.mlp.switch_mlp`` — that is Qwen3.5-MoE's path).
    Every decoder layer of a MoE checkpoint carries ``experts``
    because ``enable_moe_block`` is a single bool on the config
    (not per-layer), so all 30 layers must be wrapped."""
    adapter, _, model = _make_adapter()
    observer = Mock()
    wrapped = adapter.install_dispatch_proxy(observer)
    assert wrapped == len(model.layers)
    for idx, layer in enumerate(model.layers):
        assert isinstance(
            layer.experts.switch_glu, _DispatchProxy
        ), f"layer {idx}: switch_glu not wrapped"


def test_install_dispatch_proxy_leaves_dense_mlp_untouched() -> None:
    """The always-on dense MLP branch (``layer.mlp``) is not the
    MoE dispatch point; it must remain a plain (non-proxy) object.
    Regression guard against accidentally wrapping the dense
    branch if the proxy walk is ever generalised."""
    adapter, _, model = _make_adapter()
    observer = Mock()
    adapter.install_dispatch_proxy(observer)
    for idx, layer in enumerate(model.layers):
        assert not isinstance(
            layer.mlp, _DispatchProxy
        ), f"layer {idx}: dense mlp unexpectedly wrapped"


def test_install_dispatch_proxy_is_idempotent() -> None:
    adapter, _, _ = _make_adapter()
    observer = Mock()
    first = adapter.install_dispatch_proxy(observer)
    second = adapter.install_dispatch_proxy(observer)
    assert first == 30
    assert second == 0


def test_proxy_forward_delegates_to_inner_switch_glu() -> None:
    """Wrapped layer's proxy calls the observer and returns what
    the inner SwitchGLU returned. Uses a _DispatchProxy directly
    to keep the test decoupled from mlx-lm's real forward."""
    inner = _FakeSwitchGLU()
    observer_calls: list[tuple[int, Any]] = []

    def observer(layer_idx: int, indices: Any) -> None:
        observer_calls.append((layer_idx, indices))

    proxy = _DispatchProxy(inner, layer_idx=13, observer=observer)
    out = proxy("tok_hidden", "topk_indices")
    assert observer_calls == [(13, "topk_indices")]
    assert inner.calls == [("tok_hidden", "topk_indices")]
    assert out is inner.sentinel


def test_build_does_not_install_proxy_by_default() -> None:
    """Same invariant as Qwen3.5-MoE: proxy is opt-in via
    install_dispatch_proxy; build() does not install it."""
    from silica.weights.resident import ResidentWeightProvider

    adapter, _, model = _make_adapter()
    adapter.build(ResidentWeightProvider())
    for layer in model.layers:
        assert not isinstance(
            layer.experts.switch_glu, _DispatchProxy
        )


# --- prefill / decode_step inheritance ---------------------------------------


def test_prefill_and_decode_inherit_from_dense() -> None:
    """prefill / decode_step come from Gemma4Adapter unchanged —
    MoE routing is internal to mlx-lm's model forward."""
    adapter, kv, _ = _make_adapter()
    kv.reserve_for_prefill("req-a", [1, 2, 3])
    tokens = mx.array([1, 2, 3], dtype=mx.int32)
    logits, delta = adapter.prefill(tokens, KVHandle(req_id="req-a"))
    assert logits.shape == (adapter.config.vocab_size,)
    assert isinstance(delta, StateDelta)

    token = mx.array([7], dtype=mx.int32)
    decode_logits, decode_delta = adapter.decode_step(
        token, KVHandle(req_id="req-a")
    )
    assert decode_logits.shape == (adapter.config.vocab_size,)
    assert isinstance(decode_delta, StateDelta)


# --- Factory branch: model_type="gemma4" routes by enable_moe_block ----------


def test_factory_routes_gemma4_with_moe_flag_to_moe_adapter() -> None:
    """Factory local branch inside _build_gemma4: the Gemma4-MoE
    checkpoint reports model_type="gemma4" (same as Gemma4-dense);
    routing is disambiguated by text_config.enable_moe_block=True
    → Gemma4MoeAdapter, else Gemma4Adapter."""
    model = _FakeGemma4MoEModel()  # enable_moe_block=True by default
    tokenizer = _FakeTokenizer()
    adapter, kv = adapter_from_loaded_model(model, tokenizer)
    assert isinstance(adapter, Gemma4MoeAdapter)
    assert isinstance(kv, SimpleKVCache)


def test_factory_routes_dense_gemma4_to_dense_adapter() -> None:
    """Regression guard: a dense Gemma4 checkpoint
    (enable_moe_block=False, num_experts absent) must still route
    to Gemma4Adapter, not Gemma4MoeAdapter — otherwise E1.2's
    factory branch would break dense Gemma4-31B dispatch."""
    model = _FakeGemma4MoEModel(text_config=_dense_gemma4_text_config())
    tokenizer = _FakeTokenizer()
    adapter, _ = adapter_from_loaded_model(model, tokenizer)
    assert isinstance(adapter, Gemma4Adapter)
    assert not isinstance(adapter, Gemma4MoeAdapter)
