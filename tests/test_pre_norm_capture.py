"""Tests for silica.models.pre_norm_capture — P-5-F F.1 adapter Protocol.

Covers:

- Proxy capture-buffer fast path: ``buffer is None`` → no write.
- Proxy capture write semantics: ``buffer is not None`` → ``buffer[layer_pos]``
  receives ``k_proj(x)`` and ``__call__`` returns the same value.
- ``__getattr__`` forwarding to the original projection.
- ``apply_k_norm_then_rope_to_block`` reconstruction matches
  mlx-lm's attention forward order under IdentityCodec round-trip.
- Adapter Protocol surface: ``Qwen3Adapter`` / ``Qwen3_5Adapter`` /
  ``Gemma4Adapter`` instances satisfy
  ``isinstance(adapter, PreNormCaptureAdapter)``.
- Per-family ``install_pre_norm_capture`` armor-disarm cycle drives
  the holder; the proxy short-circuits when disarmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from silica.models.pre_norm_capture import (
    PreNormCaptureAdapter,
    _PreNormCaptureBufferHolder,
    _PreNormCaptureProxy,
    apply_k_norm_then_rope_to_block,
    install_pre_norm_capture_proxies,
)

# --- minimal stand-ins for mlx-lm attention pieces -----------------------------


class _IdentityProj:
    """Stand-in for ``nn.Linear`` that returns input unchanged.

    The proxy is a structural wrapper — its only job is to call the
    inner projection and side-effect to a buffer. Using identity here
    keeps the test focused on the proxy's behaviour without dragging
    in an actual matmul or weight tensor.
    """

    def __init__(self, weight: mx.array | None = None) -> None:
        self.weight = (
            weight if weight is not None else mx.zeros((4, 4), dtype=mx.float16)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return x


class _ScalingProj:
    """A projection that produces a recognisable, non-trivial output.

    Useful to verify the buffer received the projection's output (not
    the projection's input or some intermediate). Multiplying by a
    constant yields a tensor distinct from the input even when the
    identity projection would happen to round-trip to identical bits.
    """

    def __init__(self, scale: float = 2.5) -> None:
        self.scale = scale
        self.weight = mx.full((4,), scale, dtype=mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


@dataclass
class _FakeAttn:
    k_proj: Any
    k_norm: Any
    rope: Any


@dataclass
class _FakeLayer:
    self_attn: Any


@dataclass
class _FakeModel:
    layers: list[_FakeLayer]


def _make_fake_model(num_layers: int = 3) -> _FakeModel:
    """Two-layer fake model with simple projections / k_norm / rope."""
    return _FakeModel(
        layers=[
            _FakeLayer(
                self_attn=_FakeAttn(
                    k_proj=_ScalingProj(scale=1.5 + i * 0.5),
                    k_norm=nn.RMSNorm(4, eps=1e-6),
                    rope=nn.RoPE(dims=4, base=10000.0),
                )
            )
            for i in range(num_layers)
        ]
    )


# --- proxy-level tests ---------------------------------------------------------


def test_proxy_disarmed_returns_unchanged_no_write() -> None:
    """``buffer is None`` → proxy returns ``k_proj(x)`` and writes nothing."""
    holder = _PreNormCaptureBufferHolder()
    holder.buffer = None
    orig = _ScalingProj(scale=2.0)
    proxy = _PreNormCaptureProxy(orig, holder=holder, layer_pos=0)
    x = mx.arange(8, dtype=mx.float16).reshape(1, 4, 2)

    out = proxy(x)
    expected = orig(x)
    assert mx.allclose(out, expected).item()
    # Buffer is still None — proxy must not have allocated or written.
    assert holder.buffer is None


def test_proxy_armed_writes_and_returns_unchanged() -> None:
    """``buffer is not None`` → ``buffer[layer_pos] = k_proj(x)`` and return same."""
    holder = _PreNormCaptureBufferHolder()
    captured: dict[int, mx.array] = {}
    holder.buffer = captured
    orig = _ScalingProj(scale=3.0)
    proxy = _PreNormCaptureProxy(orig, holder=holder, layer_pos=2)
    x = mx.arange(12, dtype=mx.float16).reshape(1, 6, 2)

    out = proxy(x)
    expected = orig(x)
    # Forward result is unchanged.
    assert mx.allclose(out, expected).item()
    # Buffer received the projection output at the proxy's layer_pos.
    assert 2 in captured
    assert mx.allclose(captured[2], expected).item()


def test_proxy_layer_pos_routing() -> None:
    """Multiple proxies each write to their own layer_pos slot."""
    holder = _PreNormCaptureBufferHolder()
    captured: dict[int, mx.array] = {}
    holder.buffer = captured

    proxies = [
        _PreNormCaptureProxy(
            _ScalingProj(scale=1.0 + i), holder=holder, layer_pos=i
        )
        for i in range(3)
    ]
    x = mx.arange(8, dtype=mx.float16).reshape(1, 4, 2)
    for p in proxies:
        p(x)

    assert set(captured.keys()) == {0, 1, 2}
    # Each entry should carry the corresponding projection's output.
    for i, p in enumerate(proxies):
        assert mx.allclose(captured[i], _ScalingProj(scale=1.0 + i)(x)).item()


def test_proxy_getattr_delegates_to_original() -> None:
    """Introspection of ``proxy.weight`` etc. forwards to the original."""
    holder = _PreNormCaptureBufferHolder()
    weight = mx.full((4, 4), 7.0, dtype=mx.float16)
    orig = _IdentityProj(weight=weight)
    proxy = _PreNormCaptureProxy(orig, holder=holder, layer_pos=0)

    # The proxy itself does not declare ``.weight``; ``__getattr__``
    # falls through to ``orig.weight``.
    assert mx.allclose(proxy.weight, weight).item()


def test_proxy_buffer_swap_arms_and_disarms_in_place() -> None:
    """Swapping ``holder.buffer`` mid-life takes effect on the next call."""
    holder = _PreNormCaptureBufferHolder()
    orig = _ScalingProj(scale=1.0)
    proxy = _PreNormCaptureProxy(orig, holder=holder, layer_pos=0)
    x = mx.arange(4, dtype=mx.float16).reshape(1, 2, 2)

    # Call 1: disarmed.
    proxy(x)
    assert holder.buffer is None

    # Arm: subsequent call writes.
    captured: dict[int, mx.array] = {}
    holder.buffer = captured
    proxy(x)
    assert 0 in captured

    # Disarm again: subsequent call does not extend the previous dict.
    holder.buffer = None
    captured.clear()
    proxy(x)
    assert captured == {}


# --- math helper test ----------------------------------------------------------


def test_apply_k_norm_then_rope_matches_inline_path() -> None:
    """Helper output equals the inline ``k_norm(...).transpose(...).rope(...)`` path.

    Mirrors mlx-lm's attention forward order — applying our helper
    against synthetic inputs must produce the same tensor as running
    the same operations directly.
    """
    B, n_kv_heads, L, head_dim = 1, 2, 4, 8
    k_pre_block = mx.random.normal(
        (B, n_kv_heads, L, head_dim), dtype=mx.float32
    )
    k_norm = nn.RMSNorm(head_dim, eps=1e-6)
    rope = nn.RoPE(dims=head_dim, base=10000.0)

    out = apply_k_norm_then_rope_to_block(
        k_pre_block, k_norm=k_norm, rope_instance=rope, offset=0
    )

    # Direct path: per-token k_norm, then per-head transpose, then RoPE.
    direct = rope(
        k_norm(k_pre_block.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3),
        offset=0,
    )
    assert mx.allclose(out, direct).item()


def test_apply_k_norm_then_rope_offset_threads_through() -> None:
    """Different offsets produce different RoPE phases (sanity check)."""
    B, n_kv_heads, L, head_dim = 1, 1, 2, 4
    k_pre_block = mx.random.normal(
        (B, n_kv_heads, L, head_dim), dtype=mx.float32
    )
    k_norm = nn.RMSNorm(head_dim, eps=1e-6)
    rope = nn.RoPE(dims=head_dim, base=10000.0)

    out_offset0 = apply_k_norm_then_rope_to_block(
        k_pre_block, k_norm=k_norm, rope_instance=rope, offset=0
    )
    out_offset16 = apply_k_norm_then_rope_to_block(
        k_pre_block, k_norm=k_norm, rope_instance=rope, offset=16
    )

    # RoPE at offset 16 differs from offset 0 (unless input is zero).
    assert not mx.allclose(out_offset0, out_offset16).item()


# --- install_pre_norm_capture_proxies test ------------------------------------


def test_install_proxies_skips_layers_missing_k_proj() -> None:
    """Layer with no ``self_attn.k_proj`` is silently skipped."""
    holder = _PreNormCaptureBufferHolder()

    @dataclass
    class _BareLayer:
        self_attn: Any = None

    @dataclass
    class _BareModel:
        layers: list[Any]

    model = _BareModel(layers=[_BareLayer(), _BareLayer()])
    install_pre_norm_capture_proxies(
        model, attn_layer_indices=[0, 1], holder=holder
    )
    # Both layers' self_attn was None → install is a no-op; nothing to
    # assert beyond the absence of an exception.


def test_install_proxies_wraps_present_k_proj() -> None:
    """Layers carrying real ``k_proj`` get their projection wrapped."""
    holder = _PreNormCaptureBufferHolder()
    model = _make_fake_model(num_layers=3)
    original_projections = [layer.self_attn.k_proj for layer in model.layers]

    install_pre_norm_capture_proxies(
        model, attn_layer_indices=[0, 1, 2], holder=holder
    )

    for i, layer in enumerate(model.layers):
        assert isinstance(layer.self_attn.k_proj, _PreNormCaptureProxy)
        # Proxy duck-type forwards: orig kept reachable via __getattr__.
        proxy = layer.self_attn.k_proj
        # __getattr__ falls through to the original ScalingProj's weight.
        assert mx.allclose(proxy.weight, original_projections[i].weight).item()


def test_install_proxies_with_arm_drives_capture() -> None:
    """Arming the holder routes calls into the supplied buffer."""
    holder = _PreNormCaptureBufferHolder()
    model = _make_fake_model(num_layers=2)
    install_pre_norm_capture_proxies(
        model, attn_layer_indices=[0, 1], holder=holder
    )

    captured: dict[int, mx.array] = {}
    holder.buffer = captured
    x = mx.arange(8, dtype=mx.float16).reshape(1, 4, 2)
    for layer in model.layers:
        layer.self_attn.k_proj(x)

    assert set(captured.keys()) == {0, 1}


# --- adapter Protocol conformance ---------------------------------------------


def test_qwen3_adapter_satisfies_pre_norm_capture_protocol() -> None:
    """``Qwen3Adapter`` is structurally a ``PreNormCaptureAdapter``."""
    from tests.test_qwen3_adapter import _make_adapter_and_kv

    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter, PreNormCaptureAdapter)


def test_qwen3_5_adapter_satisfies_pre_norm_capture_protocol() -> None:
    """``Qwen3_5Adapter`` is structurally a ``PreNormCaptureAdapter``."""
    pytest.importorskip(
        "mlx_lm.models.qwen3_5",
        reason="qwen3_5 mlx-lm module required for adapter construction",
    )
    from tests.test_qwen3_5_adapter import _make_adapter_and_kv

    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter, PreNormCaptureAdapter)


def test_gemma4_adapter_satisfies_pre_norm_capture_protocol() -> None:
    """``Gemma4Adapter`` is structurally a ``PreNormCaptureAdapter``."""
    pytest.importorskip(
        "mlx_lm.models.gemma4_text",
        reason="gemma4_text mlx-lm module required for adapter construction",
    )
    from tests.test_gemma4_adapter import _make_adapter_and_kv

    adapter, _, _ = _make_adapter_and_kv()
    assert isinstance(adapter, PreNormCaptureAdapter)


def test_qwen3_adapter_install_pre_norm_capture_drives_holder() -> None:
    """Calling ``adapter.install_pre_norm_capture`` flips the holder.

    The proxy installation in ``Qwen3Adapter.__init__`` skips fake
    layers (``_PlainSelfAttn`` has no ``k_proj``), but the adapter's
    holder is still constructed and the install/disarm method still
    flips the holder's buffer attribute. This is the minimum
    Protocol-method coverage that doesn't require an mlx-lm load.
    """
    from tests.test_qwen3_adapter import _make_adapter_and_kv

    adapter, _, _ = _make_adapter_and_kv()

    # Default: disarmed.
    assert adapter._capture_holder.buffer is None

    captured: dict[int, mx.array] = {}
    adapter.install_pre_norm_capture(captured)
    assert adapter._capture_holder.buffer is captured

    adapter.install_pre_norm_capture(None)
    assert adapter._capture_holder.buffer is None
