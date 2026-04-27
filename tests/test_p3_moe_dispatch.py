"""P-3-E2: option (c) dispatch-observation contract for MoE adapters.

Pins the E-open-1 resolution (recorded 2026-04-20 in
``plans/P3_MOE_SURVEY.md`` §4.1 / §6) as executable tests: when a
MoE adapter's ``install_dispatch_proxy(observer)`` seam is wired,
the observer sees ``(layer_idx, indices)`` for every sparse-MLP
forward, and mapping those indices to ``WeightProvider.get_expert``
calls preserves coverage of every activated expert. The underlying
``SwitchGLU`` + ``gather_mm`` fetch stays fused — ``get_layer`` is
never called on the dispatch path.

Deliberately narrow:
  - Targets ``Qwen3_5MoeAdapter`` only. E1.2 adds the same shape
    for ``Gemma4MoeAdapter``; the observer / mock-provider helpers
    in this file are designed to be family-agnostic so the
    Gemma4 tests can import them directly when they land.
  - No real MoE math, no real weight fetch, no real-model
    forward. Uses a fake SwitchGLU that records its arguments.
  - Semantic pinned: **per layer, observed unique expert ids
    cover all ids present in the dispatch ``indices``**. The test
    does not count occurrences — P-6 residency needs the resident
    set of experts, not a per-token hit counter.

Out of scope (will land later):
  - E3 real-model smoke (``Engine.generate`` on the cached 35B-A3B
    checkpoint, dual-gated).
  - E1.2 Gemma4-MoE adapter.
  - E4 batched MoE smoke + parity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from silica.kvcache.simple import SimpleKVCache
from silica.models.qwen3_5_moe import Qwen3_5MoeAdapter, _DispatchProxy
from silica.weights.provider import ExpertWeights, LayerWeights
from silica.weights.resident import ResidentWeightProvider

# --- Minimal Qwen3.5-MoE-shaped fakes (self-contained) -----------------------
#
# Duplicated from ``tests/test_qwen3_5_moe_adapter.py`` on purpose: this file
# pins a different contract (E2 dispatch observation), so importing
# underscored private helpers across test modules was intentionally avoided.
# The fakes here are the minimum surface needed to construct a
# Qwen3_5MoeAdapter that exposes a wrappable ``switch_mlp`` on every layer.


def _default_text_config() -> dict[str, Any]:
    return {
        "model_type": "qwen3_5_moe",
        "num_hidden_layers": 4,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248044,
        "num_experts": 16,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "mlp_only_layers": [],
    }


@dataclass
class _FakeArgs:
    model_type: str = "qwen3_5_moe"
    text_config: dict[str, Any] = field(default_factory=_default_text_config)


@dataclass
class _FakeSelfAttn:
    num_key_value_heads: int = 2
    head_dim: int = 256


class _FakeSwitchGLU:
    """Stand-in for SwitchGLU: callable, records (x, indices) calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.sentinel = object()

    def __call__(self, x: Any, indices: Any) -> Any:
        self.calls.append((x, indices))
        return self.sentinel


class _FakeSparseMoeBlock:
    def __init__(self) -> None:
        self.switch_mlp = _FakeSwitchGLU()


class _FakeMoELayer:
    """Full-attention MoE layer: has ``self_attn`` and MoE ``mlp``."""

    is_linear = False

    def __init__(self) -> None:
        self.self_attn = _FakeSelfAttn()
        self.mlp = _FakeSparseMoeBlock()


class _FakeLinearMoELayer:
    """DeltaNet-linear MoE layer: ``is_linear=True`` BUT mlp is still MoE.

    Mirrors mlx-lm's ``qwen3_5.DecoderLayer``: the MLP branch is
    ``SparseMoeBlock(args)`` on every layer when ``num_experts > 0``;
    ``is_linear`` only selects the attention branch.
    """

    is_linear = True

    def __init__(self) -> None:
        self.mlp = _FakeSparseMoeBlock()


class _FakeModel:
    def __init__(self, num_layers: int = 4) -> None:
        self.model_type = "qwen3_5_moe"
        tc = _default_text_config()
        tc["num_hidden_layers"] = num_layers
        self.args = _FakeArgs(text_config=tc)
        self._vocab_size = int(tc["vocab_size"])
        interval = int(tc["full_attention_interval"])
        self.layers = [
            _FakeLinearMoELayer()
            if (i + 1) % interval != 0
            else _FakeMoELayer()
            for i in range(num_layers)
        ]

    def make_cache(self) -> list[Any]:
        return [KVCache() for _ in self.layers]


class _FakeTokenizer:
    vocab_size = 248044

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: Any) -> str:
        return "stub"


def _make_adapter(
    num_layers: int = 4,
) -> tuple[Qwen3_5MoeAdapter, _FakeModel]:
    model = _FakeModel(num_layers=num_layers)
    tokenizer = _FakeTokenizer()
    kv = SimpleKVCache(model.make_cache())
    adapter = Qwen3_5MoeAdapter(model, tokenizer, kv_manager=kv)
    return adapter, model


# --- Observer helper ---------------------------------------------------------


def _unique_expert_ids(indices: Any) -> set[int]:
    """Flatten ``indices`` (mx.array or nested Python list) and return
    the set of Python int expert ids present.

    Option (c) semantic: per layer, observed unique expert ids cover
    all ids present in ``indices``. The observer normalises shape via
    this helper so tests can feed ``mx.array([[[3, 7, 3, 9]]])`` or a
    raw list and get the same deduplicated set.
    """
    data = indices.tolist() if hasattr(indices, "tolist") else indices

    seen: set[int] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)
        else:
            seen.add(int(obj))

    _walk(data)
    return seen


# --- Mock WeightProvider ------------------------------------------------------


class _ObservingMockProvider:
    """WeightProvider-shaped mock that records ``get_expert`` calls.

    Implements every I-4 / D-011 method on the Protocol, but the
    dispatch-observation tests only exercise ``get_expert``.
    ``get_layer`` is recordable so tests can assert it is NOT touched
    on the MoE dispatch path; ``prefetch`` / ``release`` variants are
    no-ops.

    Not used as a runnable model — returns empty ``LayerWeights`` /
    ``ExpertWeights`` objects. The tests assert on the recorded call
    log, not on the fetched tensors.
    """

    def __init__(self) -> None:
        self.get_expert_calls: list[tuple[int, int]] = []
        self.get_layer_calls: list[int] = []

    def get_layer(self, layer_idx: int) -> LayerWeights:
        self.get_layer_calls.append(int(layer_idx))
        return LayerWeights()

    def prefetch(self, layer_indices: Sequence[int]) -> None:
        return None

    def release(self, layer_idx: int) -> None:
        return None

    def resident_bytes(self) -> int:
        return 0

    def get_expert(self, layer_idx: int, expert_id: int) -> ExpertWeights:
        self.get_expert_calls.append((int(layer_idx), int(expert_id)))
        return ExpertWeights()

    def prefetch_experts(
        self, layer_idx: int, expert_ids: Sequence[int]
    ) -> None:
        return None

    def release_expert(self, layer_idx: int, expert_id: int) -> None:
        return None


def _make_get_expert_observer(
    provider: _ObservingMockProvider,
):
    """Build the E2 observer: maps (layer_idx, indices) to
    ``provider.get_expert`` calls on each unique expert id.

    Deliberately a test-side helper (not a method on the adapter) —
    the adapter's ``install_dispatch_proxy`` contract is
    generic-callable observer; interpreting indices as
    WeightProvider expert ids is E2's concern, not the adapter's.
    """

    def observer(layer_idx: int, indices: Any) -> None:
        for expert_id in sorted(_unique_expert_ids(indices)):
            provider.get_expert(layer_idx, expert_id)

    return observer


# --- Tests -------------------------------------------------------------------


def test_dispatch_proxy_observer_calls_get_expert_for_unique_ids() -> None:
    """Manual switch_mlp call with a hand-crafted indices array
    produces one ``get_expert`` call per unique expert id on that
    layer — NOT one per occurrence. Pins the "unique ids cover"
    semantic from plans/P3_MOE_SURVEY.md §4.1."""
    adapter, model = _make_adapter(num_layers=4)
    provider = _ObservingMockProvider()
    wrapped = adapter.install_dispatch_proxy(_make_get_expert_observer(provider))
    assert wrapped == len(model.layers)

    # Pick layer index 2 and fire the proxy directly with repeating
    # expert ids. Expected unique set {3, 7, 9}.
    target_layer = 2
    indices = mx.array([[[3, 7, 3, 9, 7, 3]]], dtype=mx.int32)
    switch_mlp = model.layers[target_layer].mlp.switch_mlp
    assert isinstance(switch_mlp, _DispatchProxy)
    switch_mlp("x", indices)

    # Exactly the unique ids, once each, ordered.
    assert provider.get_expert_calls == [
        (target_layer, 3),
        (target_layer, 7),
        (target_layer, 9),
    ]


def test_dispatch_observer_covers_every_moe_layer() -> None:
    """Every layer of a Qwen3.5-MoE checkpoint carries a
    ``SparseMoeBlock`` (mlx-lm's ``qwen3_5.DecoderLayer`` constructs
    ``self.mlp = SparseMoeBlock(args)`` on EVERY layer when
    ``num_experts > 0``, regardless of ``is_linear``). The observer
    must therefore see every layer index at least once across a
    full forward. Simulated here by firing each layer's wrapped
    switch_mlp with a different indices array."""
    adapter, model = _make_adapter(num_layers=4)
    provider = _ObservingMockProvider()
    adapter.install_dispatch_proxy(_make_get_expert_observer(provider))

    # Fire each layer with an indices slice that includes a layer-
    # dependent expert id so the assertion can verify both coverage
    # AND correct layer_idx propagation.
    for layer_idx, layer in enumerate(model.layers):
        switch_mlp = layer.mlp.switch_mlp
        indices = mx.array([[[layer_idx]]], dtype=mx.int32)
        switch_mlp("x", indices)

    covered_layer_ids = {call[0] for call in provider.get_expert_calls}
    assert covered_layer_ids == set(range(len(model.layers)))

    # Per-layer expert ids were the layer index itself.
    for layer_idx in range(len(model.layers)):
        assert (layer_idx, layer_idx) in provider.get_expert_calls


def test_dispatch_observer_does_not_require_get_layer() -> None:
    """Option (c) specifies that the observer path reads per-expert
    ids only — it must NOT fall back to ``get_layer`` even when
    the provider implements it. Regression guard against a future
    observer implementation that silently wakes up the whole-layer
    fetch path (which would mask D-011 violations)."""
    adapter, model = _make_adapter(num_layers=4)
    provider = _ObservingMockProvider()
    adapter.install_dispatch_proxy(_make_get_expert_observer(provider))

    # Fire every layer — observer should only touch get_expert.
    for layer in model.layers:
        layer.mlp.switch_mlp("x", mx.array([[[0, 1]]], dtype=mx.int32))

    assert provider.get_expert_calls, "observer never called get_expert"
    assert provider.get_layer_calls == [], (
        "observer unexpectedly invoked get_layer — option (c) forbids "
        "this on the dispatch path"
    )


def test_default_build_does_not_trigger_get_expert_on_observing_provider() -> (
    None
):
    """E1.1 boundary re-pinned in the E2 mock-provider context:
    ``adapter.build(provider)`` does NOT install the dispatch proxy.
    Even when paired with an ``_ObservingMockProvider`` whose
    ``get_expert`` would record calls, a bare ``build`` followed by
    direct switch_mlp invocations must leave the call log empty
    — the proxy is opt-in, not automatic. Prevents E1.1's explicit-
    opt-in invariant from regressing silently when E2 adds the
    mock-provider surface.

    Also verifies the converse: after explicit
    ``install_dispatch_proxy`` the first switch_mlp call DOES record
    a get_expert call, so the test did not pass because the machinery
    is broken on both sides."""
    adapter, model = _make_adapter(num_layers=4)
    provider = _ObservingMockProvider()

    # build() must not install the proxy.
    adapter.build(ResidentWeightProvider())

    # Fire every layer directly — no proxy is in place.
    for layer in model.layers:
        assert not isinstance(layer.mlp.switch_mlp, _DispatchProxy)
        layer.mlp.switch_mlp("x", mx.array([[[0]]], dtype=mx.int32))

    assert provider.get_expert_calls == []
    assert provider.get_layer_calls == []

    # Now install explicitly and fire one layer — demonstrates the
    # test's positive path is actually reachable.
    adapter.install_dispatch_proxy(_make_get_expert_observer(provider))
    model.layers[0].mlp.switch_mlp("x", mx.array([[[5]]], dtype=mx.int32))
    assert provider.get_expert_calls == [(0, 5)]


# --- Unique-id helper: property pins -----------------------------------------


@pytest.mark.parametrize(
    "indices,expected",
    [
        (mx.array([[[3, 7, 3, 9]]], dtype=mx.int32), {3, 7, 9}),
        (mx.array([[[0, 0, 0]]], dtype=mx.int32), {0}),
        ([[1, 2, 3]], {1, 2, 3}),
        ([1, [2, [3]]], {1, 2, 3}),
        (mx.array([[0, 1], [2, 3]], dtype=mx.int32), {0, 1, 2, 3}),
    ],
)
def test_unique_expert_ids_handles_mx_array_and_nested_lists(
    indices: Any, expected: set[int]
) -> None:
    """The observer helper needs to accept both ``mx.array`` (the
    real-model path) and nested Python lists (convenient for tests)
    and normalise to a set of Python ints. Parametrised so a
    regression in the normalisation (shape assumptions, int
    conversion) shows a specific failing case."""
    assert _unique_expert_ids(indices) == expected
