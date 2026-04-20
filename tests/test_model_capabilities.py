"""Tests for silica.models.capabilities — D-016 ModelCapabilities + helper.

Scope:
  - ``capabilities_from_attention_pattern`` derives ``attention_kinds`` /
    ``has_recurrent_state`` correctly for every ``AttentionKind``.
  - ``has_moe`` is a pass-through keyword — ``AttentionPattern`` has no
    MoE signal, so adapters must set it explicitly.
  - ``ModelCapabilities`` is frozen and hashable.
  - ``StubModelAdapter.capabilities()`` returns the typed summary derived
    from its (all-GLOBAL) pattern.
  - ``StubModelAdapter`` still satisfies the ``ModelAdapter`` Protocol
    after the D-016 extension.
"""

from __future__ import annotations

import pytest

from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    ModelAdapter,
    StubModelAdapter,
)
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)

# --- capabilities_from_attention_pattern ---


def test_helper_pure_global_pattern() -> None:
    pattern = AttentionPattern(per_layer=(AttentionKind.GLOBAL,) * 4)
    caps = capabilities_from_attention_pattern(pattern)
    assert caps.attention_kinds == frozenset({AttentionKind.GLOBAL})
    assert caps.has_recurrent_state is False
    assert caps.has_moe is False


def test_helper_hybrid_deltanet_sets_recurrent_state() -> None:
    pattern = AttentionPattern(
        per_layer=(
            AttentionKind.GLOBAL,
            AttentionKind.HYBRID_DELTANET,
            AttentionKind.GLOBAL,
        )
    )
    caps = capabilities_from_attention_pattern(pattern)
    assert AttentionKind.HYBRID_DELTANET in caps.attention_kinds
    assert AttentionKind.GLOBAL in caps.attention_kinds
    assert caps.has_recurrent_state is True


def test_helper_pure_recurrent_sets_recurrent_state() -> None:
    pattern = AttentionPattern(per_layer=(AttentionKind.RECURRENT,) * 3)
    caps = capabilities_from_attention_pattern(pattern)
    assert caps.attention_kinds == frozenset({AttentionKind.RECURRENT})
    assert caps.has_recurrent_state is True


def test_helper_sliding_alone_does_not_set_recurrent_state() -> None:
    """SLIDING is a KV-attention variant, not recurrent (D-015)."""
    pattern = AttentionPattern(per_layer=(AttentionKind.SLIDING,) * 2)
    caps = capabilities_from_attention_pattern(pattern)
    assert caps.has_recurrent_state is False


def test_helper_hybrid_alone_does_not_set_recurrent_state() -> None:
    """HYBRID (sliding/global mix) is KV-attention, not recurrent."""
    pattern = AttentionPattern(per_layer=(AttentionKind.HYBRID,))
    caps = capabilities_from_attention_pattern(pattern)
    assert caps.has_recurrent_state is False


def test_helper_has_moe_override_is_passthrough() -> None:
    pattern = AttentionPattern(per_layer=(AttentionKind.GLOBAL,))
    caps = capabilities_from_attention_pattern(pattern, has_moe=True)
    assert caps.has_moe is True
    assert caps.attention_kinds == frozenset({AttentionKind.GLOBAL})
    assert caps.has_recurrent_state is False


def test_helper_empty_pattern_yields_empty_kinds() -> None:
    caps = capabilities_from_attention_pattern(AttentionPattern(per_layer=()))
    assert caps.attention_kinds == frozenset()
    assert caps.has_recurrent_state is False
    assert caps.has_moe is False


# --- ModelCapabilities shape ---


def test_model_capabilities_is_frozen() -> None:
    caps = ModelCapabilities(
        attention_kinds=frozenset({AttentionKind.GLOBAL}),
        has_recurrent_state=False,
        has_moe=False,
    )
    with pytest.raises(Exception):
        caps.has_moe = True  # type: ignore[misc]


def test_model_capabilities_is_hashable() -> None:
    caps = ModelCapabilities(
        attention_kinds=frozenset({AttentionKind.GLOBAL}),
        has_recurrent_state=False,
        has_moe=False,
    )
    # Hashability is documented in the module docstring — dataclass(frozen=True)
    # with a frozenset field is hashable.
    assert hash(caps) == hash(caps)


# --- StubModelAdapter integration ---


def test_stub_adapter_capabilities_are_pure_global() -> None:
    adapter = StubModelAdapter(num_layers=3)
    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert caps.attention_kinds == frozenset({AttentionKind.GLOBAL})
    assert caps.has_recurrent_state is False
    assert caps.has_moe is False


def test_stub_adapter_still_satisfies_model_adapter_protocol() -> None:
    """D-016 extended I-1 with ``capabilities()``; the Stub must carry it."""
    adapter = StubModelAdapter(num_layers=2)
    assert isinstance(adapter, ModelAdapter)
    assert hasattr(adapter, "capabilities")
    assert callable(adapter.capabilities)
