"""Tests for silica.mlx.runner — mlx-lm forward wrapper (P-1 D-010)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from silica.mlx.runner import forward


class _TracedKVCache(KVCache):
    """KVCache that counts update_and_fetch calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self.calls += 1
        return super().update_and_fetch(keys, values)  # type: ignore[no-any-return]


class _PositionAwareModel:
    """Model returning logits that encode (token_id, position) so tests can
    verify last-position slicing independently of cache state."""

    vocab_size = 6
    n_kv_heads = 1
    head_dim = 4

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        k = mx.zeros((B, self.n_kv_heads, T, self.head_dim), dtype=mx.float16)
        v = mx.zeros((B, self.n_kv_heads, T, self.head_dim), dtype=mx.float16)
        if cache is not None and cache[0] is not None:
            cache[0].update_and_fetch(k, v)
        # Logits at (b, t, v) = t (so last-position logits equal [T-1]*V).
        logits = mx.broadcast_to(
            mx.arange(T, dtype=mx.float16).reshape(1, T, 1),
            (B, T, self.vocab_size),
        )
        return logits


# --- happy path ---


def test_forward_prefill_returns_last_position_logits() -> None:
    model = _PositionAwareModel()
    cache = [_TracedKVCache()]
    tokens = mx.array([1, 2, 3], dtype=mx.int32)
    logits = forward(model, tokens, cache)
    assert logits.shape == (model.vocab_size,)
    # Position-aware model: last-position logits == (T-1) = 2.
    assert float(logits[0].item()) == 2.0


def test_forward_decode_returns_vocab_logits() -> None:
    model = _PositionAwareModel()
    cache = [_TracedKVCache()]
    tokens = mx.array([7], dtype=mx.int32)
    logits = forward(model, tokens, cache)
    assert logits.shape == (model.vocab_size,)


def test_forward_passes_cache_to_model() -> None:
    model = _PositionAwareModel()
    traced = _TracedKVCache()
    tokens = mx.array([1, 2], dtype=mx.int32)
    forward(model, tokens, [traced])
    assert traced.calls == 1
    assert traced.offset == 2


def test_forward_is_sequential_on_cache() -> None:
    """Repeated calls grow the cache (mlx-lm's KVCache accumulates)."""
    model = _PositionAwareModel()
    traced = _TracedKVCache()
    forward(model, mx.array([1, 2, 3], dtype=mx.int32), [traced])
    forward(model, mx.array([4], dtype=mx.int32), [traced])
    assert traced.calls == 2
    assert traced.offset == 4


# --- input validation ---


def test_forward_rejects_2d_input() -> None:
    model = _PositionAwareModel()
    cache = [_TracedKVCache()]
    bad = mx.zeros((1, 3), dtype=mx.int32)
    with pytest.raises(ValueError, match="expected 1-D tokens"):
        forward(model, bad, cache)


def test_forward_rejects_empty_input() -> None:
    model = _PositionAwareModel()
    cache = [_TracedKVCache()]
    empty = mx.array([], dtype=mx.int32)
    with pytest.raises(ValueError, match="non-empty"):
        forward(model, empty, cache)
