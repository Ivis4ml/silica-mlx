"""Tests for silica.mlx.runner.forward_batched — P-2 batched forward."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache, KVCache

from silica.mlx.runner import forward, forward_batched, forward_batched_full


class _TracedBatchKVCache(BatchKVCache):
    """BatchKVCache that counts update_and_fetch calls for assertion."""

    def __init__(self, left_padding: list[int]) -> None:
        super().__init__(left_padding)
        self.calls = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self.calls += 1
        return super().update_and_fetch(keys, values)  # type: ignore[no-any-return]


class _BatchedModel:
    """Duck-typed model: writes zeros through the cache, returns (B,T,V) logits."""

    VOCAB = 7
    N_KV = 1
    HEAD_DIM = 4

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        if cache is not None and cache[0] is not None:
            k = mx.zeros((B, self.N_KV, T, self.HEAD_DIM), dtype=mx.float16)
            v = mx.zeros((B, self.N_KV, T, self.HEAD_DIM), dtype=mx.float16)
            cache[0].update_and_fetch(k, v)
        return mx.zeros((B, T, self.VOCAB), dtype=mx.float16)


# --- forward_batched happy path ---


def test_forward_batched_returns_last_position_per_row() -> None:
    model = _BatchedModel()
    cache = [_TracedBatchKVCache([0, 0])]
    tokens = mx.zeros((2, 3), dtype=mx.int32)
    logits = forward_batched(model, tokens, cache)
    assert tuple(logits.shape) == (2, model.VOCAB)


def test_forward_batched_drives_cache() -> None:
    model = _BatchedModel()
    traced = _TracedBatchKVCache([0, 0, 0])
    tokens = mx.zeros((3, 4), dtype=mx.int32)
    forward_batched(model, tokens, [traced])
    assert traced.calls == 1
    # _idx advances by T across all rows in lockstep (Gate-0 invariant).
    assert traced._idx == 4


def test_forward_batched_sequential_calls_accumulate_idx() -> None:
    model = _BatchedModel()
    traced = _TracedBatchKVCache([0, 0])
    forward_batched(model, mx.zeros((2, 5), dtype=mx.int32), [traced])
    forward_batched(model, mx.zeros((2, 1), dtype=mx.int32), [traced])
    assert traced._idx == 6
    assert traced.calls == 2


# --- forward_batched validation ---


def test_forward_batched_rejects_1d_tokens() -> None:
    model = _BatchedModel()
    bad = mx.array([1, 2, 3], dtype=mx.int32)
    with pytest.raises(ValueError, match="expected 2-D tokens"):
        forward_batched(model, bad, [_TracedBatchKVCache([0])])


def test_forward_batched_rejects_3d_tokens() -> None:
    model = _BatchedModel()
    bad = mx.zeros((1, 2, 3), dtype=mx.int32)
    with pytest.raises(ValueError, match="expected 2-D tokens"):
        forward_batched(model, bad, [_TracedBatchKVCache([0])])


def test_forward_batched_rejects_empty_B_or_T() -> None:
    model = _BatchedModel()
    with pytest.raises(ValueError, match="non-zero"):
        forward_batched(
            model,
            mx.zeros((0, 3), dtype=mx.int32),
            [_TracedBatchKVCache([])],
        )
    with pytest.raises(ValueError, match="non-zero"):
        forward_batched(
            model,
            mx.zeros((2, 0), dtype=mx.int32),
            [_TracedBatchKVCache([0, 0])],
        )


# --- forward (1-D wrapper) still works after refactor ---


class _UnbatchedKV(KVCache):
    """Plain single-request KVCache counting update_and_fetch calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self.calls += 1
        return super().update_and_fetch(keys, values)  # type: ignore[no-any-return]


class _SingleRequestModel:
    """Uses plain KVCache (not BatchKVCache) so forward wrapper stays on P-1 path."""

    VOCAB = 5

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        if cache is not None and cache[0] is not None:
            k = mx.zeros((B, 1, T, 4), dtype=mx.float16)
            v = mx.zeros((B, 1, T, 4), dtype=mx.float16)
            cache[0].update_and_fetch(k, v)
        return mx.zeros((B, T, self.VOCAB), dtype=mx.float16)


def test_forward_single_request_returns_1d() -> None:
    model = _SingleRequestModel()
    cache = [_UnbatchedKV()]
    tokens = mx.array([1, 2, 3], dtype=mx.int32)
    logits = forward(model, tokens, cache)
    assert tuple(logits.shape) == (model.VOCAB,)


def test_forward_rejects_2d_input() -> None:
    model = _SingleRequestModel()
    with pytest.raises(ValueError, match="expected 1-D tokens"):
        forward(model, mx.zeros((1, 3), dtype=mx.int32), [_UnbatchedKV()])


def test_forward_rejects_empty_input() -> None:
    model = _SingleRequestModel()
    with pytest.raises(ValueError, match="non-empty"):
        forward(model, mx.array([], dtype=mx.int32), [_UnbatchedKV()])


# ---------------------------------------------------------------------------
# forward_batched_full — P-5-C.1 full-position logits entry point
# ---------------------------------------------------------------------------


class _PositionDependentModel:
    """Duck-typed model whose logits encode absolute position + token id
    so ``forward_batched`` (last-position only) and
    ``forward_batched_full`` (all positions) can be distinguished."""

    VOCAB = 5
    N_KV = 1
    HEAD_DIM = 4

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        if cache is not None and cache[0] is not None:
            k = mx.zeros((B, self.N_KV, T, self.HEAD_DIM), dtype=mx.float16)
            v = mx.zeros((B, self.N_KV, T, self.HEAD_DIM), dtype=mx.float16)
            cache[0].update_and_fetch(k, v)
        # Logits[b, t, v] = b * 100 + t * 10 + v — distinct across every
        # position so a wrong slice would change the returned values.
        b_axis = mx.arange(B, dtype=mx.float32).reshape(B, 1, 1) * 100.0
        t_axis = mx.arange(T, dtype=mx.float32).reshape(1, T, 1) * 10.0
        v_axis = mx.arange(self.VOCAB, dtype=mx.float32).reshape(1, 1, self.VOCAB)
        return b_axis + t_axis + v_axis


def test_forward_batched_full_returns_btv_shape() -> None:
    model = _PositionDependentModel()
    cache = [_TracedBatchKVCache([0, 0])]
    tokens = mx.zeros((2, 3), dtype=mx.int32)
    logits = forward_batched_full(model, tokens, cache)
    assert tuple(logits.shape) == (2, 3, model.VOCAB)


def test_forward_batched_is_last_position_of_full() -> None:
    """``forward_batched`` must equal ``forward_batched_full[:, -1, :]``
    bit-identically. Guarantees the wrapper refactor does not drift the
    two callers apart — sampling (last-position) and PPL (all-
    position) share the same model call."""
    # Two separate cache instances since forward_batched_full mutates
    # the first one via update_and_fetch.
    model = _PositionDependentModel()
    cache_for_full = [_TracedBatchKVCache([0, 0])]
    cache_for_last = [_TracedBatchKVCache([0, 0])]
    tokens = mx.zeros((2, 4), dtype=mx.int32)

    full = forward_batched_full(model, tokens, cache_for_full)
    last = forward_batched(model, tokens, cache_for_last)

    assert tuple(full.shape) == (2, 4, model.VOCAB)
    assert tuple(last.shape) == (2, model.VOCAB)
    assert bool(mx.array_equal(full[:, -1, :], last).item())


def test_forward_batched_full_rejects_1d_tokens() -> None:
    model = _PositionDependentModel()
    with pytest.raises(ValueError, match="expected 2-D tokens"):
        forward_batched_full(
            model, mx.zeros((3,), dtype=mx.int32), [_TracedBatchKVCache([0])]
        )


def test_forward_batched_full_rejects_empty_batch() -> None:
    model = _PositionDependentModel()
    with pytest.raises(ValueError, match="non-zero B and T"):
        forward_batched_full(
            model, mx.zeros((0, 3), dtype=mx.int32), []
        )


def test_forward_batched_full_rejects_empty_seq() -> None:
    model = _PositionDependentModel()
    with pytest.raises(ValueError, match="non-zero B and T"):
        forward_batched_full(
            model, mx.zeros((1, 0), dtype=mx.int32), [_TracedBatchKVCache([0])]
        )


def test_forward_batched_full_drives_cache() -> None:
    """update_and_fetch is called exactly once per forward pass, same
    as the non-full variant."""
    model = _PositionDependentModel()
    traced = _TracedBatchKVCache([0, 0])
    forward_batched_full(model, mx.zeros((2, 5), dtype=mx.int32), [traced])
    assert traced.calls == 1
    assert traced._idx == 5
