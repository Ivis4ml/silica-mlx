"""P-5-F F.1 — adapter-owned pre-norm K capture and reconstruction.

Shared scaffolding for the (3b) production prefix-cache:

- ``_PreNormCaptureProxy`` wraps each attention layer's ``k_proj`` to
  capture ``k_proj(x)`` (pre-k_norm K) without modifying the in-flight
  forward. The proxy returns ``k_proj(x)`` unchanged on every call;
  when the active capture buffer is non-None, it side-effects a write
  to ``buffer[attn_layer_pos]``. When the buffer is None the proxy
  short-circuits and skips the write entirely (decode-step overhead
  must stay at zero per ``docs/P5_F_OPENING.md`` §6.6).
- ``_PreNormCaptureBufferHolder`` is a mutable, single-attribute holder
  every proxy reads on every call. Swapping the active buffer is a
  single attribute write on the holder; proxies are not re-installed.
- ``install_pre_norm_capture_proxies`` installs proxies in place on
  ``model.layers[layer_idx].self_attn.k_proj`` for every attention
  layer index passed in. DeltaNet / linear layers (Qwen3.5) and any
  family-specific non-attention layers are filtered out by the caller.
- ``apply_k_norm_then_rope_to_block`` is the math helper invoked on
  hit-path admit: takes a per-head pre-norm K block and reconstructs
  the post-RoPE K mlx-lm's attention forward expects in the seeded
  cache.
- ``PreNormCaptureAdapter`` is the runtime-checkable Protocol mixin
  every adapter that ships P-5-F implements (Qwen3 / Qwen3.5 / Gemma4
  + their MoE variants). Mirrors the
  ``RecurrentStateAdapter`` mixin pattern in
  ``silica/models/recurrent.py``.

Why a shared module: the F.0b' bench oracle in
``silica/bench/ppl_oracle.py`` had a local copy of this scaffolding
during F.0b' verification. F.1 lifts it onto the adapter Protocol so
the production hot path (``ContinuousBatcher`` / ``RadixPrefixCache``)
and the bench oracle exercise the same code (per
``docs/P5_F_OPENING.md`` §6.6).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import mlx.core as mx


class _PreNormCaptureBufferHolder:
    """Mutable holder for the currently-active K_pre capture buffer.

    Every ``_PreNormCaptureProxy`` installed on the model holds a
    reference to one shared holder. ``install_pre_norm_capture(buffer)``
    on the adapter sets ``holder.buffer = buffer`` (or ``None`` to
    disable). Swapping the buffer is a single attribute write — proxies
    are not re-installed and decode forwards are uncharged when the
    holder's buffer is None.
    """

    __slots__ = ("buffer",)

    def __init__(self) -> None:
        self.buffer: dict[int, mx.array] | None = None


class _PreNormCaptureProxy:
    """Wraps ``attn.k_proj`` to capture K_pre without altering the forward.

    On every ``__call__(x)`` the proxy:

    1. Computes ``out = self._orig(x)`` — the original projection.
    2. Reads ``self._holder.buffer``.
       - When ``None``, returns ``out`` immediately. The fast path
         keeps decode steps and capture-disabled prefills at zero
         overhead beyond one attribute load.
       - When non-None, writes ``buffer[self._layer_pos] = out`` and
         returns ``out``.

    The output is always ``self._orig(x)`` — the in-flight forward sees
    no change, the proxy is purely a side-effect for the prefix-cache
    store path.

    ``__getattr__`` forwards unknown attribute reads to the original
    projection, so quantized-Linear introspection (``weight``,
    ``scales``, ``biases``, etc.) and any ``isinstance`` checks against
    a ``Linear`` shape continue to work after wrapping.
    """

    __slots__ = ("_orig", "_holder", "_layer_pos")

    def __init__(
        self,
        orig: Any,
        *,
        holder: _PreNormCaptureBufferHolder,
        layer_pos: int,
    ) -> None:
        self._orig = orig
        self._holder = holder
        self._layer_pos = layer_pos

    def __call__(self, x: mx.array) -> mx.array:
        out: mx.array = self._orig(x)
        buffer = self._holder.buffer
        if buffer is None:
            return out
        buffer[self._layer_pos] = out
        return out

    def __getattr__(self, name: str) -> Any:
        # Note: __getattr__ is only invoked when normal attribute lookup
        # fails, so the __slots__ attributes (_orig / _holder /
        # _layer_pos) never reach here. This is the duck-typing seam
        # that keeps the proxy usable wherever the original Linear is
        # introspected.
        return getattr(self._orig, name)


def install_pre_norm_capture_proxies(
    model: Any,
    *,
    attn_layer_indices: list[int],
    holder: _PreNormCaptureBufferHolder,
) -> None:
    """Install ``_PreNormCaptureProxy`` on every attention layer's k_proj.

    ``attn_layer_indices`` is the dense list of absolute layer indices
    that carry full / sliding KV attention. Hybrid families
    (Qwen3.5: ``layer.is_linear == True`` for DeltaNet) filter out
    non-attention layers before passing in. The proxy's ``layer_pos``
    is the position into ``attn_layer_indices`` (0-based, dense), which
    is the same indexing the scheduler uses when assembling per-layer
    payloads for the prefix store.

    The holder is shared across all installed proxies. Capture is
    disabled by default (``holder.buffer is None``); the adapter's
    ``install_pre_norm_capture(buffer)`` call enables it for the next
    forward.

    Layers whose ``self_attn`` lacks a ``k_proj`` attribute are
    skipped silently. Real mlx-lm models always have ``k_proj`` on
    attention layers (Qwen3 / Qwen3.5 / Gemma4 all build it via
    ``nn.Linear`` in their ``Attention.__init__``), so this branch
    only fires under minimal test fakes that stub the attention
    module without a projection. F.1 capture functionality is then
    inactive on those layers, but adapter construction still
    succeeds — Protocol-conformance tests can use bare fakes
    without rebuilding the entire mlx-lm attention stack.
    """
    for layer_pos, layer_idx in enumerate(attn_layer_indices):
        layer = model.layers[layer_idx]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        orig_k = getattr(attn, "k_proj", None)
        if orig_k is None:
            continue
        attn.k_proj = _PreNormCaptureProxy(
            orig_k, holder=holder, layer_pos=layer_pos
        )


def apply_k_norm_then_rope_to_block(
    k_pre_block: mx.array,
    *,
    k_norm: Any,
    rope_instance: Any,
    offset: int,
) -> mx.array:
    """Reconstruct post-RoPE K from a pre-norm K block.

    Input ``k_pre_block`` shape: ``(B, n_kv_heads, block_size, head_dim)``
    (per-head transposed, the same layout the live cache uses).
    Output: same shape, post-k_norm and post-RoPE.

    The reconstruction mirrors mlx-lm's attention forward exactly. Both
    Qwen3 / Qwen3.5 (via ``Qwen3NextAttention``) and Gemma4
    (``gemma4_text.Attention``) apply ``k_norm`` on the
    ``(B, L, n_kv_heads, head_dim)`` per-token layout, then transpose to
    per-head and apply RoPE. We accept per-head input (the codec works
    in per-head shape), transpose back to per-token for k_norm, then
    transpose forward and RoPE.

    ``offset`` is the absolute token-position of the block's first
    token in the request's prefix.
    """
    # k_norm expects (B, L, n_kv_heads, head_dim)
    k_pre_seq = k_pre_block.transpose(0, 2, 1, 3)
    k_post_norm_seq = k_norm(k_pre_seq)
    # Back to (B, n_kv_heads, L, head_dim) for RoPE
    k_post_norm_per_head = k_post_norm_seq.transpose(0, 2, 1, 3)
    k_post_rope: mx.array = rope_instance(
        k_post_norm_per_head, offset=offset
    )
    return k_post_rope


@runtime_checkable
class PreNormCaptureAdapter(Protocol):
    """Mixin implemented by adapters that ship P-5-F (3b) prefix-cache.

    Every Silica adapter family (Qwen3 / Qwen3.5 / Gemma4 + their MoE
    variants) implements this. Stub adapters and any future families
    that opt out of the pre-norm prefix-cache path simply do not
    implement these methods; ``isinstance(adapter, PreNormCaptureAdapter)``
    returns ``False`` and callers fall back to the legacy post-RoPE
    store semantics.

    Mirrors the ``RecurrentStateAdapter`` pattern (see
    ``silica/models/recurrent.py``): a separate sub-Protocol rather
    than folded into ``ModelAdapter``, so adapters that lack the
    capability stay fully Protocol-conformant against ``ModelAdapter``.
    """

    def install_pre_norm_capture(
        self, buffer: dict[int, mx.array] | None
    ) -> None:
        """Arm or disarm the K_pre capture buffer on every attention
        layer's projection proxy.

        Setting ``buffer=None`` returns the proxies to no-op (decode
        path uses this; the in-flight forward is bit-identical to the
        unwrapped reference). Setting a non-None ``dict[int, mx.array]``
        arms the next forward — each attention layer's ``k_proj`` will
        write its output to ``buffer[attn_layer_pos]`` on every call.

        The proxy is installed permanently at adapter construction
        time; this method only swaps the active buffer pointer.
        """
        ...

    def apply_k_norm_then_rope(
        self,
        attn_layer_pos: int,
        k_pre_block: mx.array,
        *,
        offset: int,
    ) -> mx.array:
        """Reconstruct post-RoPE K from a pre-norm K block.

        ``attn_layer_pos`` is the index into the dense list of
        attention-layer positions the proxies were installed at — the
        same indexing ``buffer[attn_layer_pos]`` uses on the capture
        side. The adapter looks up the corresponding layer's
        ``self_attn.k_norm`` and ``self_attn.rope`` and applies them in
        mlx-lm's order.

        ``offset`` is the absolute token-position of the block's first
        token in the request's prefix.

        Input shape: ``(B, n_kv_heads, block_size, head_dim)``. Output:
        same shape, post-k_norm and post-RoPE.
        """
        ...


__all__ = [
    "PreNormCaptureAdapter",
    "_PreNormCaptureBufferHolder",
    "_PreNormCaptureProxy",
    "apply_k_norm_then_rope_to_block",
    "install_pre_norm_capture_proxies",
]
