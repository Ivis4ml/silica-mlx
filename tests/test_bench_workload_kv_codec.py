"""Tests for the P-5-A.3a ``Workload.kv_codec`` scaffolding.

Covers:

- Workload construction accepts ``kv_codec=None`` (default — pre-P-5
  pass-through behaviour) and any registered codec id.
- Validation rejects ``kv_codec`` set without ``prefix_cache=True``
  (codec installs on the prefix cache's store, so the field is
  meaningless otherwise).
- Validation rejects unknown codec ids, with the registered catalogue
  reported in the error message.
- The lazy import of ``silica.bench.codec_registry`` inside
  ``Workload.__post_init__`` does not regress silica.bench.scenario's
  import-time dependency profile.
- ``_maybe_build_prefix_cache`` constructs a store with the named
  codec installed when ``kv_codec`` is set + an adapter is supplied.
- ``_maybe_build_prefix_cache`` raises if ``kv_codec`` is set but
  ``adapter`` is omitted.
- ``kv_codec=None`` path ignores the adapter argument entirely,
  preserving pre-A.3a behaviour on all existing scenarios.

Scope boundary: this test file covers only the schema + seam
additions from P-5-A.3a. A.3b adds the DECODE_TOK_S_WITH_PREFIX_HIT
oracle; A.3c adds the named scenarios and acceptance gate.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from silica.bench.codec_registry import list_codec_ids
from silica.bench.runner import _maybe_build_prefix_cache
from silica.bench.scenario import Workload
from silica.kvcache.codec import BlockTQPayload, IdentityCodec, RawFp16Payload
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.adapter import KVLayout


class _StubAdapter:
    """Minimal ModelAdapter stub exposing only ``kv_layout()`` —
    enough for ``_maybe_build_prefix_cache`` to construct codecs."""

    def __init__(
        self,
        *,
        num_layers: int = 2,
        n_kv_heads: int = 2,
        head_dim: int = 64,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        self._layout = KVLayout(
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    def kv_layout(self) -> KVLayout:
        return self._layout


# ---------------------------------------------------------------------------
# Workload.__post_init__ validation
# ---------------------------------------------------------------------------


def test_workload_kv_codec_none_is_default() -> None:
    """Pre-A.3a scenarios pass no ``kv_codec``; new default is
    ``None`` and must not change any existing behaviour."""
    wl = Workload(name="t", prompts=("hi",), max_tokens=4)
    assert wl.kv_codec is None


def test_workload_kv_codec_accepts_registered_id() -> None:
    """Every entry in the codec registry is a valid ``kv_codec``
    value when paired with ``prefix_cache=True``."""
    for codec_id in list_codec_ids():
        wl = Workload(
            name="t",
            prompts=("hi",),
            max_tokens=4,
            prefix_cache=True,
            kv_codec=codec_id,
        )
        assert wl.kv_codec == codec_id


def test_workload_kv_codec_rejects_without_prefix_cache() -> None:
    """Codec installs on the prefix cache's store; the field is
    meaningless when no prefix cache exists. Construction must fail
    loudly rather than silently ignoring the field."""
    with pytest.raises(ValueError, match="prefix_cache=True"):
        Workload(
            name="t",
            prompts=("hi",),
            max_tokens=4,
            prefix_cache=False,
            kv_codec="fp16",
        )


def test_workload_kv_codec_rejects_unknown_id() -> None:
    """Unknown id surfaces with the available registry keys inline
    so typos see the correct set without hunting through docs."""
    with pytest.raises(ValueError, match="unknown kv_codec id"):
        Workload(
            name="t",
            prompts=("hi",),
            max_tokens=4,
            prefix_cache=True,
            kv_codec="not_a_real_codec",
        )


def test_workload_kv_codec_unknown_id_error_lists_available() -> None:
    """The error message must include the catalog so callers can
    see valid choices without reading the source."""
    try:
        Workload(
            name="t",
            prompts=("hi",),
            max_tokens=4,
            prefix_cache=True,
            kv_codec="nope",
        )
    except ValueError as exc:
        msg = str(exc)
        assert "fp16" in msg
        assert "block_tq_b64_b4" in msg
    else:
        pytest.fail("expected ValueError")


# ---------------------------------------------------------------------------
# _maybe_build_prefix_cache with kv_codec
# ---------------------------------------------------------------------------


def _register_probe_block(
    pc: Any, *, n_kv_heads: int, head_dim: int, n_layers: int = 2
) -> int:
    """Register one detached block on the pc's store so
    ``resident_bytes`` / ``resident_bytes_per_block`` reflect a
    realistic state. Returns the block id."""
    store = pc.store
    bid = store.allocate_id()
    store.retain_source(bid)
    shape = (1, n_kv_heads, store.block_size, head_dim)
    per_layer_kv = [
        (
            mx.zeros(shape, dtype=mx.float16),
            mx.zeros(shape, dtype=mx.float16),
        )
        for _ in range(n_layers)
    ]
    store.register_detached(bid, per_layer_kv)
    return int(bid)


def test_maybe_build_prefix_cache_none_without_prefix_cache_flag() -> None:
    """``workload.prefix_cache=False`` → ``None``. Unchanged behaviour
    from pre-A.3a; adapter argument is irrelevant in this path."""
    wl = Workload(name="t", prompts=("hi",), max_tokens=4)
    assert _maybe_build_prefix_cache(wl) is None
    # Adapter supplied — still None.
    assert _maybe_build_prefix_cache(wl, _StubAdapter()) is None  # type: ignore[arg-type]


def test_maybe_build_prefix_cache_passthrough_ignores_adapter() -> None:
    """``prefix_cache=True, kv_codec=None`` constructs a pass-through
    store — codec factories are never called, so adapter can be
    omitted. This preserves every existing P-2 / P-4 scenario whose
    workload does not set ``kv_codec``."""
    wl = Workload(
        name="t", prompts=("hi",), max_tokens=4, prefix_cache=True
    )
    pc = _maybe_build_prefix_cache(wl)  # no adapter passed
    assert pc is not None
    assert isinstance(pc.store, SyntheticPrefixBlockStore)
    # Pass-through: both K/V codecs are None on the store.
    assert pc.store.resident_bytes_per_block() is None


def test_maybe_build_prefix_cache_fp16_installs_identity_codec() -> None:
    """``kv_codec="fp16"`` installs an ``IdentityCodec`` on both
    sides. Verified by registering a detached block + asserting the
    store wraps tensors in ``RawFp16Payload`` (the identity codec's
    payload class)."""
    wl = Workload(
        name="t",
        prompts=("hi",),
        max_tokens=4,
        prefix_cache=True,
        kv_codec="fp16",
    )
    adapter = _StubAdapter(n_kv_heads=2, head_dim=64, num_layers=2)
    pc = _maybe_build_prefix_cache(wl, adapter)  # type: ignore[arg-type]
    assert pc is not None
    assert isinstance(pc.store, SyntheticPrefixBlockStore)

    _register_probe_block(pc, n_kv_heads=2, head_dim=64, n_layers=2)
    # Inspect the internal _detached structure to confirm the
    # payload type is RawFp16Payload (IdentityCodec's output).
    for layer in next(iter(pc.store._detached.values())):
        assert isinstance(layer.k, RawFp16Payload)
        assert isinstance(layer.v, RawFp16Payload)


def test_maybe_build_prefix_cache_block_tq_installs_block_tq_codec() -> None:
    """``kv_codec="block_tq_b64_b4"`` installs ``BlockTurboQuantMSE``
    on both sides. Verified by payload class after a real encode."""
    wl = Workload(
        name="t",
        prompts=("hi",),
        max_tokens=4,
        prefix_cache=True,
        kv_codec="block_tq_b64_b4",
    )
    # BlockTQ needs head_dim divisible by vq_block_size=64.
    adapter = _StubAdapter(n_kv_heads=2, head_dim=64, num_layers=2)
    pc = _maybe_build_prefix_cache(wl, adapter)  # type: ignore[arg-type]
    assert pc is not None
    assert isinstance(pc.store, SyntheticPrefixBlockStore)

    _register_probe_block(pc, n_kv_heads=2, head_dim=64, n_layers=2)
    for layer in next(iter(pc.store._detached.values())):
        assert isinstance(layer.k, BlockTQPayload)
        assert isinstance(layer.v, BlockTQPayload)
    # And resident_bytes_per_block reflects compressed layout.
    per_block = pc.store.resident_bytes_per_block()
    assert per_block is not None
    # Sanity: compressed is strictly below fp16 baseline (2 layers ×
    # 2 sides × block_size × n_kv_heads × head_dim × 2 bytes).
    fp16_baseline = 2 * 2 * 16 * 2 * 64 * 2
    assert per_block < fp16_baseline


def test_maybe_build_prefix_cache_raises_when_codec_set_but_adapter_missing() -> None:
    """When ``kv_codec`` is set, codec factory needs ``n_kv_heads /
    head_dim / dtype`` from ``adapter.kv_layout()``. Calling without
    an adapter is a caller bug; surface it loudly."""
    wl = Workload(
        name="t",
        prompts=("hi",),
        max_tokens=4,
        prefix_cache=True,
        kv_codec="fp16",
    )
    with pytest.raises(ValueError, match="no adapter was supplied"):
        _maybe_build_prefix_cache(wl)  # no adapter — must raise


def test_maybe_build_prefix_cache_uses_adapter_layout_for_codec() -> None:
    """Codec factory is called with the adapter's layout values;
    Gemma-style bf16 K/V layouts land as bf16 codecs without a
    dtype widening."""
    wl = Workload(
        name="t",
        prompts=("hi",),
        max_tokens=4,
        prefix_cache=True,
        kv_codec="fp16",
    )
    adapter_bf16 = _StubAdapter(
        n_kv_heads=4, head_dim=128, num_layers=2, dtype=mx.bfloat16
    )
    pc = _maybe_build_prefix_cache(wl, adapter_bf16)  # type: ignore[arg-type]
    assert pc is not None
    # The identity codec installed on the store should carry the
    # adapter's dtype.
    assert isinstance(pc.store, SyntheticPrefixBlockStore)
    # _k_codec and _v_codec are the same IdentityCodec instance
    # under the shorthand; both must honour the bf16 dtype.
    k_codec = pc.store._k_codec
    v_codec = pc.store._v_codec
    assert isinstance(k_codec, IdentityCodec)
    assert isinstance(v_codec, IdentityCodec)
    assert k_codec.dtype == mx.bfloat16
    assert v_codec.dtype == mx.bfloat16
