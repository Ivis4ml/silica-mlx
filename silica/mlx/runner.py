"""silica.mlx.runner — low-level MLX forward wrapper for P-1 (D-010).

Thin bridge between Silica's I-1 ``ModelAdapter`` and mlx-lm's duck-typed
``model(tokens, cache=...)`` forward call. Centralises the small amount of
marshalling that sits between the engine's token stream and mlx-lm's module:

  - 1-D ``(T,)`` input normalised to 2-D ``(B=1, T)`` for mlx-lm.
  - Silica-owned cache list passed through unchanged (D-010 clean path —
    mlx-lm's ``update_and_fetch`` writes straight into Silica storage).
  - Last-position logits ``(V,)`` extracted — prefill cares only about the
    final token's distribution, and decode inputs a single token so the last
    position coincides with the only position.

Deliberately small. Does not own:
  - Sampling (Silica's P-0 ``Sampler`` is called by ``Engine`` on the logits
    returned from this module).
  - Prompt chunking (P-1 feeds the whole prompt at once; chunked-prefill is a
    P-2 concern gated on Q-010).
  - Cache lifecycle (``KVManager`` owns ``reserve`` / ``free`` / ``trim``).

Even at this size the module earns its keep: it is the single place in
Silica where the shape of mlx-lm's forward signature is referenced, so any
future upstream change is fixed here rather than across every adapter.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def forward(
    model: Any,
    tokens: mx.array,
    cache_list: list[Any],
) -> mx.array:
    """Run one mlx-lm forward pass with Silica's cache injected.

    Args:
        model: an mlx-lm-compatible module whose ``__call__`` accepts
            ``(input_tokens_2d, cache=list)`` and returns logits of shape
            ``(B, T, V)``.
        tokens: 1-D token ids of shape ``(T,)`` — a prompt for prefill or a
            single token (``T == 1``) for decode.
        cache_list: Silica-owned per-layer cache list (from
            ``SimpleKVCache.cache_list()``); mlx-lm mutates the entries
            in-place via ``update_and_fetch``.

    Returns:
        Logits at the last position, shape ``(V,)``.
    """
    if tokens.ndim != 1:
        raise ValueError(
            f"expected 1-D tokens (T,), got shape {tuple(tokens.shape)}"
        )
    if tokens.size == 0:
        raise ValueError("tokens must be non-empty")

    batched = tokens[None]  # (1, T)
    logits = model(batched, cache=cache_list)  # (1, T, V)
    # mx.array subscript returns Any per mlx stubs; slice is always mx.array.
    return logits[0, -1, :]  # type: ignore[no-any-return]
