"""Tests for silica.weights.provider and silica.weights.resident.

Covers:
  - I-4 Protocol shape (@runtime_checkable sanity + method presence).
  - ResidentWeightProvider happy-path semantics (get_layer / resident_bytes).
  - D-011 "loud failure" contract for dense MoE methods.
  - D-012 resident_bytes idempotency.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.weights.provider import (
    ExpertWeights,
    LayerWeights,
    WeightProvider,
)
from silica.weights.resident import ResidentWeightProvider


def _mk_layer(name: str = "weight", shape: tuple[int, ...] = (4, 4)) -> LayerWeights:
    tensor = mx.zeros(shape, dtype=mx.float16)
    return LayerWeights(tensors={name: tensor}, resident_bytes=tensor.nbytes)


@pytest.fixture
def two_layers() -> dict[int, LayerWeights]:
    return {0: _mk_layer(), 1: _mk_layer()}


@pytest.fixture
def provider(two_layers: dict[int, LayerWeights]) -> ResidentWeightProvider:
    return ResidentWeightProvider(two_layers)


# --- I-4 Protocol shape ---


def test_resident_provider_satisfies_weight_provider(
    provider: ResidentWeightProvider,
) -> None:
    assert isinstance(provider, WeightProvider)


# --- dense happy path ---


def test_get_layer_returns_stored_weights(
    provider: ResidentWeightProvider, two_layers: dict[int, LayerWeights]
) -> None:
    assert provider.get_layer(0) is two_layers[0]
    assert provider.get_layer(1) is two_layers[1]


def test_get_layer_missing_raises_key_error(
    provider: ResidentWeightProvider,
) -> None:
    with pytest.raises(KeyError):
        provider.get_layer(999)


def test_prefetch_and_release_are_noop(provider: ResidentWeightProvider) -> None:
    # Must accept any layer index sequence without raising or mutating state.
    before = provider.resident_bytes()
    provider.prefetch([0, 1, 42])
    provider.release(0)
    provider.release(1)
    assert provider.resident_bytes() == before


def test_resident_bytes_sums_held_layers(
    provider: ResidentWeightProvider, two_layers: dict[int, LayerWeights]
) -> None:
    expected = sum(layer.resident_bytes for layer in two_layers.values())
    assert provider.resident_bytes() == expected


def test_empty_provider_has_zero_resident_bytes() -> None:
    provider = ResidentWeightProvider()
    assert provider.resident_bytes() == 0


# --- D-011 loud failure on MoE methods ---


def test_get_expert_raises_not_implemented(provider: ResidentWeightProvider) -> None:
    with pytest.raises(NotImplementedError, match="dense provider has no per-expert path"):
        provider.get_expert(0, 0)


def test_prefetch_experts_raises_not_implemented(
    provider: ResidentWeightProvider,
) -> None:
    with pytest.raises(NotImplementedError, match="dense provider has no per-expert path"):
        provider.prefetch_experts(0, [0, 1])


def test_release_expert_raises_not_implemented(
    provider: ResidentWeightProvider,
) -> None:
    with pytest.raises(NotImplementedError, match="dense provider has no per-expert path"):
        provider.release_expert(0, 0)


# --- D-012 idempotency ---


def test_resident_bytes_is_idempotent(provider: ResidentWeightProvider) -> None:
    first = provider.resident_bytes()
    for _ in range(5):
        assert provider.resident_bytes() == first


# --- dataclass basics ---


def test_layer_weights_defaults() -> None:
    lw = LayerWeights()
    assert lw.tensors == {}
    assert lw.resident_bytes == 0


def test_expert_weights_defaults() -> None:
    ew = ExpertWeights()
    assert ew.tensors == {}
    assert ew.resident_bytes == 0
