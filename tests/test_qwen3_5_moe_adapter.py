"""Tests for silica.models.qwen3_5_moe — I-1 Qwen3_5MoeAdapter (P-3-E1.1).

Uses a fake Qwen3.5-MoE-shaped model so the adapter's Silica-side
contract (capabilities override, MoE variant guards, config.extra
MoE metadata, install_dispatch_proxy seam) is exercised without
loading the ~20 GB Qwen3.5-35B-A3B-4bit checkpoint. Real-model
forward smoke is P-3-E3 territory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

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
from silica.models.factory import adapter_from_loaded_model, supported_model_types
from silica.models.qwen3_5_moe import Qwen3_5MoeAdapter, _DispatchProxy

# --- Qwen3.5-MoE-shaped fakes -------------------------------------------------


def _default_35b_a3b_text_config() -> dict[str, Any]:
    """Qwen3.5-35B-A3B-4bit text_config values as observed 2026-04-20.

    Captures the concrete 35B-A3B shape the adapter targets. Tests
    that need to exercise variant-guard branches override one field
    at a time.
    """
    return {
        "model_type": "qwen3_5_moe",
        "num_hidden_layers": 8,  # small fake; real checkpoint has 40
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248044,
        # MoE-specific (real 35B-A3B values).
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        # norm_topk_prob intentionally absent — the real checkpoint
        # omits it, and the runtime value comes from
        # qwen3_5.TextModelArgs dataclass default (True). The adapter
        # must resolve the runtime value without the key being
        # present.
        # mlp_only_layers empty — matches the probed checkpoint.
        "mlp_only_layers": [],
        # attn_output_gate — present in config, silently ignored by
        # mlx-lm. The adapter records both the config value and the
        # "mlx-lm honours it" resolution.
        "attn_output_gate": True,
    }


@dataclass
class _FakeQwen35MoEArgs:
    model_type: str = "qwen3_5_moe"
    text_config: dict[str, Any] = field(
        default_factory=_default_35b_a3b_text_config
    )


@dataclass
class _FakeSelfAttn:
    num_key_value_heads: int = 2
    head_dim: int = 256


class _FakeMoELayer:
    """Stand-in for a full-attention (non-linear) decoder layer.

    Mirrors mlx-lm's ``qwen3_5.DecoderLayer`` shape when
    ``num_experts > 0``: both the ``self_attn`` module AND an
    ``mlp = SparseMoeBlock(args)`` are present. The dense
    ``Qwen3_5Adapter._build_kv_layout`` reads
    ``self_attn.num_key_value_heads`` / ``self_attn.head_dim`` from
    the first non-linear layer.
    """

    is_linear = False

    def __init__(self) -> None:
        self.self_attn = _FakeSelfAttn()
        self.mlp = _FakeSparseMoeBlock()


class _FakeLinearLayer:
    """DeltaNet (linear-attention) decoder layer stand-in.

    Important: mlx-lm's ``qwen3_5.DecoderLayer`` puts
    ``self.mlp = SparseMoeBlock(args)`` on EVERY layer when
    ``num_experts > 0``, regardless of ``is_linear`` — the
    ``is_linear`` flag only selects the attention branch
    (``GatedDeltaNet`` vs ``Attention``). So a MoE-shaped fake linear
    layer must ALSO carry ``mlp.switch_mlp``, otherwise the dispatch
    proxy would never see it and a regression that erroneously skips
    DeltaNet MoE layers would silently pass.
    """

    is_linear = True

    def __init__(self) -> None:
        self.mlp = _FakeSparseMoeBlock()


class _FakeNoMoELayer:
    """Layer with no ``switch_mlp`` — used to pin skip behaviour.

    The ``install_dispatch_proxy`` contract says layers whose ``mlp``
    has no ``switch_mlp`` attribute (dense MLP, shared-expert-only,
    non-MoE) are skipped silently. This fake provides exactly that
    shape without being confused with either the DeltaNet or the
    full-attention MoE fakes.
    """

    is_linear = False

    class _DenseMLP:
        """Stand-in for a dense mlx-lm MLP (no ``switch_mlp``)."""

    def __init__(self) -> None:
        self.self_attn = _FakeSelfAttn()
        self.mlp = _FakeNoMoELayer._DenseMLP()


class _FakeSparseMoeBlock:
    """Stand-in for Qwen3NextSparseMoeBlock.

    Only exposes the attributes the dispatch proxy needs (``switch_mlp``).
    The ``switch_mlp`` itself is callable and returns a constant
    sentinel so tests can verify the proxy delegates.
    """

    def __init__(self) -> None:
        self.switch_mlp = _FakeSwitchGLU()


class _FakeSwitchGLU:
    """Stand-in for SwitchGLU: callable, records last call.

    ``weights`` is declared up-front as ``Any`` so the proxy
    ``__getattr__`` fall-through test can assign to it without
    triggering a mypy "attribute is not defined" complaint if the
    project later extends mypy coverage to ``tests/``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.sentinel = object()
        self.weights: Any = None

    def __call__(self, x: Any, indices: Any) -> Any:
        self.calls.append((x, indices))
        return self.sentinel


class _FakeQwen35MoEModel:
    """Minimal stand-in for mlx_lm.models.qwen3_5_moe.Model."""

    def __init__(
        self, text_config: dict[str, Any] | None = None
    ) -> None:
        self.model_type = "qwen3_5_moe"
        self.args = _FakeQwen35MoEArgs(
            text_config=text_config or _default_35b_a3b_text_config()
        )
        self._vocab_size = int(
            self.args.text_config.get("vocab_size", 248044)
        )
        n = int(self.args.text_config.get("num_hidden_layers", 8))
        interval = int(self.args.text_config.get("full_attention_interval", 4))
        # Mirror qwen3_5.DecoderLayer: is_linear when (idx+1) % interval != 0.
        self.layers = [
            _FakeLinearLayer() if (i + 1) % interval != 0 else _FakeMoELayer()
            for i in range(n)
        ]

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        return mx.zeros((B, T, self._vocab_size), dtype=mx.float16)

    def make_cache(self) -> list[Any]:
        return [KVCache() for _ in self.layers]


class _FakeTokenizer:
    vocab_size = 248044

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: Any) -> str:
        return "stub"


def _make_adapter(
    text_config: dict[str, Any] | None = None,
) -> tuple[Qwen3_5MoeAdapter, SimpleKVCache, _FakeQwen35MoEModel]:
    model = _FakeQwen35MoEModel(text_config=text_config)
    tokenizer = _FakeTokenizer()
    kv = SimpleKVCache(model.make_cache())
    adapter = Qwen3_5MoeAdapter(model, tokenizer, kv_manager=kv)
    return adapter, kv, model


# --- I-1 Protocol conformance -------------------------------------------------


def test_adapter_satisfies_model_adapter_protocol() -> None:
    adapter, _, _ = _make_adapter()
    assert isinstance(adapter, ModelAdapter)


def test_adapter_exposes_config_attribute() -> None:
    adapter, _, _ = _make_adapter()
    assert isinstance(adapter.config, ModelConfig)
    assert adapter.config.num_layers == 8


# --- capabilities: has_moe=True override --------------------------------------


def test_capabilities_declare_has_moe_true() -> None:
    """The load-bearing override: Qwen3.5-MoE adapter must report
    has_moe=True so the ContinuousBatcher capability gate routes it
    to the MoE branch (rejected until P-3-E4)."""
    adapter, _, _ = _make_adapter()
    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert caps.has_moe is True


def test_capabilities_inherit_attention_kinds_from_qwen3_5() -> None:
    """attention_kinds / has_recurrent_state come from the inherited
    Qwen3.5 hybrid pattern — MoE does not touch attention."""
    adapter, _, _ = _make_adapter()
    caps = adapter.capabilities()
    assert caps.attention_kinds == frozenset(
        {AttentionKind.GLOBAL, AttentionKind.HYBRID_DELTANET}
    )
    assert caps.has_recurrent_state is True


def test_attention_pattern_inherits_qwen3_5_3_to_1_hybrid() -> None:
    """8 fake layers with full_attention_interval=4 yield
    [D, D, D, G, D, D, D, G] — the same 3:1 pattern as dense
    Qwen3.5."""
    adapter, _, _ = _make_adapter()
    pattern = adapter.attention_pattern()
    assert isinstance(pattern, AttentionPattern)
    assert pattern.per_layer == (
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.GLOBAL,
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.GLOBAL,
    )


# --- MoE metadata on config.extra ---------------------------------------------


def test_config_extra_records_moe_metadata() -> None:
    """Every MoE-specific structural field lands on config.extra so
    downstream consumers (memory budget, telemetry, future MoE
    scheduler) can read them without re-parsing text_config."""
    adapter, _, _ = _make_adapter()
    extra = adapter.config.extra
    assert extra["num_experts"] == 256
    assert extra["num_experts_per_tok"] == 8
    assert extra["moe_intermediate_size"] == 512
    assert extra["shared_expert_intermediate_size"] == 512
    assert extra["mlp_only_layers"] == []
    assert extra["is_moe_adapter"] is True


def test_config_extra_records_norm_topk_prob_runtime_default_true() -> None:
    """The default 35B-A3B config omits norm_topk_prob; the runtime
    value comes from qwen3_5.TextModelArgs dataclass default (True).
    The adapter resolves this and records the runtime value so
    downstream consumers do not re-derive the default."""
    adapter, _, _ = _make_adapter()
    extra = adapter.config.extra
    assert extra["norm_topk_prob_runtime"] is True


def test_config_extra_records_norm_topk_prob_runtime_explicit_false() -> None:
    """When the config explicitly sets norm_topk_prob, the recorded
    runtime value follows the config (regardless of dataclass
    default)."""
    tc = _default_35b_a3b_text_config()
    tc["norm_topk_prob"] = False
    adapter, _, _ = _make_adapter(text_config=tc)
    assert adapter.config.extra["norm_topk_prob_runtime"] is False


def test_config_extra_records_attn_output_gate_divergence() -> None:
    """attn_output_gate is set in the 35B-A3B config but mlx-lm does
    not consume it (repo-wide grep of mlx_lm/models returns zero
    references as of 2026-04-20). The adapter records both the
    config value and the resolution so a future HF-vs-mlx-lm
    comparison does not have to re-derive this from source."""
    adapter, _, _ = _make_adapter()
    extra = adapter.config.extra
    assert extra["attn_output_gate_config"] is True
    assert extra["attn_output_gate_mlx_lm_honors"] is False


# --- MoE variant guard --------------------------------------------------------


def test_variant_guard_rejects_num_experts_zero() -> None:
    """num_experts == 0 means a non-MoE checkpoint; use the dense
    Qwen3_5Adapter instead."""
    tc = _default_35b_a3b_text_config()
    tc["num_experts"] = 0
    with pytest.raises(NotImplementedError, match="num_experts"):
        _make_adapter(text_config=tc)


def test_variant_guard_rejects_top_k_zero() -> None:
    tc = _default_35b_a3b_text_config()
    tc["num_experts_per_tok"] = 0
    with pytest.raises(NotImplementedError, match="num_experts_per_tok"):
        _make_adapter(text_config=tc)


def test_variant_guard_rejects_top_k_greater_than_experts() -> None:
    tc = _default_35b_a3b_text_config()
    tc["num_experts"] = 4
    tc["num_experts_per_tok"] = 8
    with pytest.raises(NotImplementedError, match="<= num_experts"):
        _make_adapter(text_config=tc)


def test_variant_guard_rejects_non_empty_mlp_only_layers() -> None:
    """Pins E-open-2: qwen3_5.DecoderLayer does not consult
    mlp_only_layers, so a non-empty list would mean mlx-lm silently
    wires MoE onto layers the config marks as dense MLP. The adapter
    refuses rather than accept a silently-wrong wiring."""
    tc = _default_35b_a3b_text_config()
    tc["mlp_only_layers"] = [0, 3, 7]
    with pytest.raises(NotImplementedError, match="mlp_only_layers"):
        _make_adapter(text_config=tc)


# --- install_dispatch_proxy (option (c) seam) --------------------------------


def test_install_dispatch_proxy_wraps_every_moe_layer() -> None:
    """In real Qwen3.5-MoE (``qwen3_5.py:223``),
    ``self.mlp = SparseMoeBlock(args)`` runs on EVERY decoder layer
    when ``num_experts > 0`` — including ``is_linear=True`` DeltaNet
    layers. The ``is_linear`` flag gates the attention branch only,
    not the MLP branch. So a correct ``install_dispatch_proxy``
    implementation must wrap the ``switch_mlp`` on every layer of a
    MoE checkpoint, and this test enforces that: the fake model
    gives all 8 layers (4 DeltaNet + 4 full-attention) a
    SparseMoeBlock-shaped mlp.

    Regression guard: an earlier version of this test used a fake
    model where only the full-attention layers carried a
    ``switch_mlp``, which would silently pass even if the adapter
    incorrectly skipped DeltaNet MoE layers."""
    adapter, _, model = _make_adapter()
    observer = Mock()
    wrapped = adapter.install_dispatch_proxy(observer)

    assert wrapped == len(model.layers)
    for idx, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        assert mlp is not None, f"layer {idx}: mlp missing on MoE fake"
        switch_mlp = getattr(mlp, "switch_mlp", None)
        assert isinstance(switch_mlp, _DispatchProxy), (
            f"layer {idx}: expected _DispatchProxy, got {type(switch_mlp).__name__}"
        )


def test_install_dispatch_proxy_skips_layers_without_switch_mlp() -> None:
    """Layers whose ``mlp`` has no ``switch_mlp`` (dense MLP,
    shared-expert-only, non-MoE) are skipped silently. Constructs a
    mixed model where one layer has a dense ``mlp`` (no
    ``switch_mlp``) and verifies ``install_dispatch_proxy`` wraps the
    others and leaves the dense one alone.

    Note: a real Qwen3.5-MoE checkpoint has SparseMoeBlock on every
    layer (``mlp_only_layers`` is empty in practice, per
    E-open-2 resolution); the dense-mlp fake here is synthetic, used
    to pin the skip code path. If ``mlp_only_layers`` ever becomes
    non-empty, the adapter's variant guard rejects the checkpoint
    before ``install_dispatch_proxy`` is called."""
    # Build a 4-layer fake where layer 2 has a dense mlp (no switch).
    model = _FakeQwen35MoEModel()
    model.layers = [
        _FakeMoELayer(),
        _FakeMoELayer(),
        _FakeNoMoELayer(),
        _FakeMoELayer(),
    ]
    tokenizer = _FakeTokenizer()
    kv = SimpleKVCache([KVCache() for _ in model.layers])
    adapter = Qwen3_5MoeAdapter(model, tokenizer, kv_manager=kv)

    observer = Mock()
    wrapped = adapter.install_dispatch_proxy(observer)

    assert wrapped == 3  # every layer except index 2
    assert isinstance(model.layers[0].mlp.switch_mlp, _DispatchProxy)
    assert isinstance(model.layers[1].mlp.switch_mlp, _DispatchProxy)
    assert isinstance(model.layers[3].mlp.switch_mlp, _DispatchProxy)
    # The dense-mlp layer's mlp has no switch_mlp at all — nothing to assert
    # about it except that the proxy count skipped it.
    assert not hasattr(model.layers[2].mlp, "switch_mlp")


def test_install_dispatch_proxy_is_idempotent() -> None:
    """Calling install twice must not double-wrap the same layer."""
    adapter, _, _ = _make_adapter()
    observer = Mock()
    first = adapter.install_dispatch_proxy(observer)
    second = adapter.install_dispatch_proxy(observer)
    assert first > 0
    assert second == 0


def test_proxy_forward_calls_observer_then_delegates() -> None:
    """The proxy must invoke the observer with
    (layer_idx, indices) on each forward and return the inner
    SwitchGLU's output unchanged. Tested directly on a _DispatchProxy
    to avoid touching the real mlx-lm MoE graph."""
    inner = _FakeSwitchGLU()
    observer_calls: list[tuple[int, Any]] = []

    def observer(layer_idx: int, indices: Any) -> None:
        observer_calls.append((layer_idx, indices))

    proxy = _DispatchProxy(inner, layer_idx=17, observer=observer)
    x = "tok_hidden"
    idx = "topk_indices"
    out = proxy(x, idx)
    # Observer called exactly once with the layer index and indices arg.
    assert observer_calls == [(17, idx)]
    # Inner was called through with the same (x, indices).
    assert inner.calls == [(x, idx)]
    # Return value is the inner's output unchanged.
    assert out is inner.sentinel


def test_proxy_getattr_forwards_to_inner() -> None:
    """The proxy preserves the SwitchGLU interface for
    attribute access (weights, to_quantized, etc.) via __getattr__."""
    inner = _FakeSwitchGLU()
    inner.weights = "stacked_expert_tensor"  # synthetic attribute
    proxy = _DispatchProxy(inner, layer_idx=0, observer=lambda _l, _i: None)
    # Attribute fall-through.
    assert proxy.weights == "stacked_expert_tensor"


def test_build_does_not_install_proxy_by_default() -> None:
    """Calling build(provider) must NOT install the proxy. The
    default single-request path uses ResidentWeightProvider whose
    get_expert raises; auto-installing would propagate that failure
    into smoke tests. install_dispatch_proxy is an explicit opt-in
    used by E2."""
    from silica.weights.resident import ResidentWeightProvider

    adapter, _, model = _make_adapter()
    adapter.build(ResidentWeightProvider())
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        switch_mlp = getattr(mlp, "switch_mlp", None) if mlp else None
        assert not isinstance(switch_mlp, _DispatchProxy)


# --- kv_layout / make_batch_cache inheritance --------------------------------


def test_kv_layout_inherits_from_dense_qwen3_5_formula() -> None:
    """MoE routing does not change KV footprint. Qwen3.5-MoE is
    homogeneous GQA (2 KV heads x 256 head_dim x num_layers), so
    bytes_per_token_total stays None and MemoryBudgeter.for_adapter
    falls back to the dense formula (P-3-D4)."""
    adapter, _, _ = _make_adapter()
    layout = adapter.kv_layout()
    assert layout.num_layers == 8
    assert layout.n_kv_heads == 2
    assert layout.head_dim == 256
    # Inherited from dense: bytes_per_token_total is None (fallback
    # formula applies). MoE does not change KV shape, so no
    # per-kind override is needed here.
    assert layout.bytes_per_token_total is None


def test_make_batch_cache_inherits_hybrid_deltanet_mixing() -> None:
    """The dense Qwen3_5Adapter.make_batch_cache produces
    [ArraysCache / BatchKVCache] per layer based on layer.is_linear.
    MoE routing does not change this — the MoE adapter inherits the
    factory unchanged and every DeltaNet layer gets ArraysCache,
    every global-attention layer gets BatchKVCache."""
    adapter, _, model = _make_adapter()
    caches = adapter.make_batch_cache(left_padding=[0, 1])
    assert len(caches) == len(model.layers)
    for idx, (layer, cache) in enumerate(zip(model.layers, caches, strict=True)):
        if getattr(layer, "is_linear", False):
            assert isinstance(cache, ArraysCache), f"layer {idx}"
        else:
            assert isinstance(cache, BatchKVCache), f"layer {idx}"


# --- prefill / decode_step inheritance ---------------------------------------


def test_prefill_and_decode_inherit_from_dense() -> None:
    """prefill / decode_step come from Qwen3_5Adapter — MoE routing
    is internal to mlx-lm's model forward. Silica just threads
    tokens through unchanged. Both entry points covered in one test
    because the MoE adapter's expected behaviour is "Qwen3_5Adapter
    verbatim on these methods"; if either diverges, this test
    fails and the commit message's "inherit" claim is falsified."""
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


# --- factory dispatch --------------------------------------------------------


def test_factory_dispatches_qwen3_5_moe_to_moe_adapter() -> None:
    """model_type='qwen3_5_moe' must route through the factory to
    Qwen3_5MoeAdapter (not the dense Qwen3_5Adapter)."""
    model = _FakeQwen35MoEModel()
    tokenizer = _FakeTokenizer()
    adapter, kv = adapter_from_loaded_model(model, tokenizer)
    assert isinstance(adapter, Qwen3_5MoeAdapter)
    assert isinstance(kv, SimpleKVCache)


def test_factory_supported_list_includes_qwen3_5_moe() -> None:
    supported = supported_model_types()
    assert "qwen3_5_moe" in supported
    # Existing dense keys still registered — regression guard for E1.1.
    assert "qwen3_5" in supported
    assert "qwen3" in supported
    assert "gemma4" in supported
