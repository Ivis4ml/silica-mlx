"""silica.mlx.runner — low-level MLX forward wrappers.

Thin bridge between Silica's I-1 ``ModelAdapter`` and mlx-lm's duck-typed
``model(tokens, cache=...)`` forward call. Two entry points:

- ``forward_batched`` takes 2-D ``(B, T)`` tokens and returns per-row
  last-position logits ``(B, V)``. This is the P-2 ``ContinuousBatcher``
  path (Unit 16a onwards) and the canonical shape mlx-lm expects.
- ``forward`` takes 1-D ``(T,)`` tokens and returns ``(V,)`` — a thin
  adapter over ``forward_batched`` kept so P-1's ``Engine.generate``
  path is unchanged.

Deliberately small. Does not own:
  - Sampling (Silica's P-0 ``Sampler`` is called by ``Engine`` or
    ``ContinuousBatcher`` on the logits returned from this module).
  - Prompt chunking (P-1 / P-2 feed the whole prompt at once; chunked-
    prefill is gated on Q-010).
  - Cache lifecycle (``KVManager`` / ``ContinuousBatcher`` own
    ``reserve`` / ``free`` / ``filter`` / ``extend``).

Even at this size the module earns its keep: it is the single place in
Silica where the shape of mlx-lm's forward signature is referenced, so
any future upstream change is fixed here rather than across every
adapter or scheduler phase.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def forward_batched(
    model: Any,
    tokens: mx.array,
    cache_list: list[Any],
) -> mx.array:
    """Run one mlx-lm forward pass on a batched token tensor.

    Args:
        model: an mlx-lm-compatible module whose ``__call__`` accepts
            ``(input_tokens_2d, cache=list)`` and returns logits of
            shape ``(B, T, V)``.
        tokens: 2-D token ids ``(B, T)``. Rows may represent freshly
            admitted requests (``T = prompt_len``, left-padded where
            lengths differ) or decoding rows (``T = 1``). Mixing row
            lengths is not supported — the batcher runs prefill and
            decode as separate batched calls.
        cache_list: Silica-owned per-layer batched cache list. mlx-lm
            mutates the entries in-place via ``update_and_fetch``.

    Returns:
        Per-row last-position logits, shape ``(B, V)``.
    """
    if tokens.ndim != 2:
        raise ValueError(
            f"expected 2-D tokens (B, T), got shape {tuple(tokens.shape)}"
        )
    B, T = tokens.shape
    if B == 0 or T == 0:
        raise ValueError(
            f"tokens must have non-zero B and T; got shape (B={B}, T={T})"
        )

    logits = model(tokens, cache=cache_list)  # (B, T, V)
    # mx.array subscript returns Any per mlx stubs; slice is always mx.array.
    return logits[:, -1, :]  # type: ignore[no-any-return]


def forward(
    model: Any,
    tokens: mx.array,
    cache_list: list[Any],
) -> mx.array:
    """Single-request forward — wraps ``forward_batched`` at B=1.

    P-1 callers (``Engine.generate``) pass 1-D ``(T,)`` tokens and
    expect ``(V,)`` logits. Implemented in terms of ``forward_batched``
    so any future change to the batched path (e.g. dtype promotion,
    additional model-call kwargs) automatically propagates.
    """
    if tokens.ndim != 1:
        raise ValueError(
            f"expected 1-D tokens (T,), got shape {tuple(tokens.shape)}"
        )
    if tokens.size == 0:
        raise ValueError("tokens must be non-empty")

    batched = forward_batched(model, tokens[None], cache_list)  # (1, V)
    return batched[0]  # type: ignore[no-any-return]
