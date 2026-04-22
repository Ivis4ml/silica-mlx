"""P-0 capstone — all five core interfaces (I-1..I-5) are Protocol-shaped and
every stub instantiates and conforms.

This is the unified "freeze-readiness" check mandated by P-0 Acceptance:
after P-0 exit the five signatures are frozen (PLAN.md §6 header). Any
change to a Protocol's public surface — adding / removing a method or
required attribute — will break one of the assertions here and force the
change to go through the Decisions Log (D-NNN).

Per-Protocol behavior tests live in the companion files (`test_kvcodec.py`,
`test_kv_manager.py`, `test_weight_provider.py`, `test_draft_engine.py`,
`test_model_adapter.py`). This file intentionally tests only the public
contract shape.
"""

from __future__ import annotations

from typing import Any

import pytest

from silica.kvcache.codec import IdentityCodec, VectorCodec
from silica.kvcache.manager import KVManager, NullKVManager
from silica.models.adapter import ModelAdapter, StubModelAdapter
from silica.speculative.engine import DraftEngine, NoopDraftEngine
from silica.weights.provider import WeightProvider
from silica.weights.resident import ResidentWeightProvider

# Each entry: (Protocol, stub factory, required public attributes).
# Attributes are checked separately because @runtime_checkable verifies
# methods only.
INTERFACE_TABLE: list[tuple[str, type, Any, tuple[str, ...]]] = [
    (
        "I-1 ModelAdapter",
        ModelAdapter,
        lambda: StubModelAdapter(),
        ("config",),
    ),
    (
        "I-2 KVManager",
        KVManager,
        lambda: NullKVManager(),
        ("block_size",),
    ),
    (
        "I-3 VectorCodec",
        VectorCodec,
        lambda: IdentityCodec(
            block_size=16, n_kv_heads=2, head_dim=8
        ),
        ("block_size", "dtype"),
    ),
    (
        "I-4 WeightProvider",
        WeightProvider,
        lambda: ResidentWeightProvider(),
        (),
    ),
    (
        "I-5 DraftEngine",
        DraftEngine,
        lambda: NoopDraftEngine(),
        (),
    ),
]


@pytest.mark.parametrize(
    "label,protocol,stub_factory,attrs",
    INTERFACE_TABLE,
    ids=[row[0] for row in INTERFACE_TABLE],
)
def test_stub_satisfies_protocol(
    label: str, protocol: type, stub_factory: Any, attrs: tuple[str, ...]
) -> None:
    """Stub is a runtime-checkable instance of its Protocol (method surface)."""
    stub = stub_factory()
    assert isinstance(stub, protocol), f"{label}: stub does not satisfy Protocol"


@pytest.mark.parametrize(
    "label,protocol,stub_factory,attrs",
    INTERFACE_TABLE,
    ids=[row[0] for row in INTERFACE_TABLE],
)
def test_stub_has_required_attributes(
    label: str, protocol: type, stub_factory: Any, attrs: tuple[str, ...]
) -> None:
    """Required class attributes (not covered by @runtime_checkable) are present."""
    stub = stub_factory()
    for attr in attrs:
        assert hasattr(stub, attr), f"{label}: stub missing attribute {attr!r}"


@pytest.mark.parametrize(
    "label,protocol,stub_factory,attrs",
    INTERFACE_TABLE,
    ids=[row[0] for row in INTERFACE_TABLE],
)
def test_protocol_is_runtime_checkable(
    label: str, protocol: type, stub_factory: Any, attrs: tuple[str, ...]
) -> None:
    """All five P-0 Protocols are @runtime_checkable so test harnesses can
    use isinstance to verify the method surface."""
    # Protocols carry a private marker once runtime_checkable decoration is applied.
    # The dunder name is _is_runtime_protocol in typing.Protocol.
    assert getattr(protocol, "_is_runtime_protocol", False), (
        f"{label}: Protocol must be @runtime_checkable"
    )


def test_exactly_five_interfaces_are_under_freeze() -> None:
    """PLAN.md §6 freezes exactly five core interfaces (I-1..I-5); D-013
    keeps Sampler as a concrete class, not a sixth Protocol."""
    assert len(INTERFACE_TABLE) == 5
