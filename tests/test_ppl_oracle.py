"""P-5-C.1 unit tests for silica.bench.ppl_oracle.

Two invariants to lock:

1. **Chunk invariance**: running streaming PPL with chunk_size=16 /
   32 / 64 / full-sequence on the *same* token stream must produce
   the *same* NLL sum + token count (to fp rounding). If PPL drifts
   with chunk size, the cache-across-chunks plumbing is broken —
   matches vqbench's streaming_ppl module-docstring invariant.
2. **Known PPL reference**: for a deterministic fake model with
   analytically computable cross-entropy, the oracle's output
   matches the closed-form NLL sum.

A tiny fake adapter + fake model lives in this file (not a shared
fixture) so the test is self-contained. The fake model emits logits
that depend on the *absolute* position in the sequence (via a
``_FakeCache.offset`` counter that is incremented per forward call)
— if the oracle fails to share the cache across chunks, the
position counter stays at zero across chunks and the test catches
the drift.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from silica.bench.ppl_oracle import (
    perplexity_from_nll,
    teacher_forced_chunked_nll,
)

_VOCAB = 8


class _FakeCache:
    """Position-tracking stand-in for an mlx-lm ``BatchKVCache``.

    The oracle never touches the cache itself — it only passes the
    list through to the model's forward call. The fake model uses
    ``cache[0].offset`` to know the *absolute* sequence position of
    each token in the current chunk, so the test can assert chunk
    invariance: with chunks 16 / 32 / 64, the model sees the same
    absolute positions regardless of chunking, and therefore emits
    the same logits at each scored token.
    """

    def __init__(self) -> None:
        self.offset = 0

    def advance(self, n: int) -> None:
        self.offset += n


class _FakeModel:
    """Deterministic model: ``logits[b, t, v] = f(absolute_position, v)``.

    Defined so chunk invariance has something to actually grip onto.
    If the oracle correctly shares the cache across chunks, the
    ``absolute_position`` of each token is the same whether we run
    the whole sequence in one chunk or in 16 small ones, so the
    logits at every scored position are identical and the NLL sum
    is identical.

    Logits formula: ``logits[b, t, v] = cos(pi * (pos + 1) / seq_len
    * (v + 1) / V) * 3.0`` — moderate range so log-softmax has
    meaningful probability mass across the vocabulary, not a
    degenerate one-hot. The exact formula does not matter beyond
    "different at every (pos, v) pair"; the chunk-invariance test
    only needs logits to be a deterministic function of absolute
    position.
    """

    def __init__(self, vocab_size: int = _VOCAB) -> None:
        self.V = vocab_size

    def __call__(
        self, tokens: mx.array, cache: list[Any]
    ) -> mx.array:
        B, T = tokens.shape
        cache_obj = cache[0]
        offset = cache_obj.offset

        positions = mx.arange(T, dtype=mx.float32) + float(offset)  # (T,)
        v_axis = mx.arange(self.V, dtype=mx.float32)  # (V,)

        # Broadcast to (T, V): cos(pi * (pos + 1) / 32 * (v + 1) / V) * 3
        pos_scaled = (positions + 1.0) / 32.0  # (T,)
        v_scaled = (v_axis + 1.0) / float(self.V)  # (V,)
        grid = pos_scaled[:, None] * v_scaled[None, :]  # (T, V)
        per_pos_logits = mx.cos(grid * math.pi) * 3.0  # (T, V)

        # Expand to (B, T, V).
        logits = mx.broadcast_to(
            per_pos_logits[None, :, :], (B, T, self.V)
        )

        cache_obj.advance(T)
        return logits


class _FakePPLAdapter:
    """Minimal adapter shape for the PPL oracle: ``_model`` attribute +
    ``make_batch_cache([0])`` method. Nothing else is consumed."""

    def __init__(self, vocab_size: int = _VOCAB) -> None:
        self._model = _FakeModel(vocab_size=vocab_size)

    def make_batch_cache(self, left_padding: list[int]) -> list[Any]:
        # PPL oracle calls with [0]; single-layer cache is enough since
        # the fake model only inspects cache[0].offset.
        assert left_padding == [0], (
            f"fake adapter supports [0] left_padding only; got {left_padding}"
        )
        return [_FakeCache()]


# =============================================================================
# Reference NLL — computed independently from the fake model's logits
# =============================================================================


def _reference_nll_sum(token_ids: mx.array, vocab_size: int = _VOCAB) -> tuple[float, int]:
    """Compute total NLL by running the fake model on the full sequence
    in one shot and scoring every token past the first — no chunking,
    serves as the ground truth the chunked oracle must agree with."""
    assert tuple(token_ids.shape)[0] == 1
    seq_len = int(token_ids.shape[1])
    if seq_len < 2:
        return 0.0, 0

    model = _FakeModel(vocab_size=vocab_size)
    cache = [_FakeCache()]
    logits = model(token_ids, cache)  # (1, seq_len, V)

    # Predict tokens[:, 1:] from logits[:, :-1, :] — standard shift-by-1.
    shifted_logits = logits[:, :-1, :].astype(mx.float32)  # (1, seq_len - 1, V)
    targets = token_ids[:, 1:]  # (1, seq_len - 1)
    log_probs = shifted_logits - mx.logsumexp(
        shifted_logits, axis=-1, keepdims=True
    )
    gathered = mx.take_along_axis(
        log_probs, targets.astype(mx.int32)[..., None], axis=-1
    )
    nll_sum = float(-mx.sum(gathered).item())
    return nll_sum, seq_len - 1


# =============================================================================
# Tests
# =============================================================================


def _make_tokens(seq_len: int, seed: int = 0, vocab_size: int = _VOCAB) -> mx.array:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab_size, size=(1, seq_len), dtype=np.int32)
    return mx.array(ids)


class TestChunkInvariance:
    """Running the same sequence at different chunk sizes must produce
    the same cumulative NLL and the same token count. If the shared-
    cache-across-chunks logic breaks, the fake model emits different
    logits at chunk boundaries and this test fires."""

    @pytest.mark.parametrize("chunk_size", [8, 16, 32, 64, 128])
    def test_nll_equals_full_sequence_reference(self, chunk_size: int) -> None:
        adapter = _FakePPLAdapter()
        seq_len = 128
        tokens = _make_tokens(seq_len, seed=123)

        ref_nll, ref_n = _reference_nll_sum(tokens)
        chunked_nll, chunked_n = teacher_forced_chunked_nll(
            adapter, tokens, chunk_size=chunk_size
        )

        assert chunked_n == ref_n, (
            f"chunk_size={chunk_size}: scored {chunked_n} tokens, "
            f"reference {ref_n}"
        )
        # fp32 tolerance; chunk-boundary token adds one extra
        # log_softmax call vs reference, but the math is identical.
        assert chunked_nll == pytest.approx(ref_nll, rel=1e-5, abs=1e-5), (
            f"chunk_size={chunk_size}: nll {chunked_nll:.6f}, "
            f"reference {ref_nll:.6f}"
        )

    def test_invariance_pairwise(self) -> None:
        """Direct chunk-vs-chunk: NLL at chunk=8 equals NLL at chunk=64."""
        adapter_a = _FakePPLAdapter()
        adapter_b = _FakePPLAdapter()
        tokens = _make_tokens(64, seed=7)

        nll_a, n_a = teacher_forced_chunked_nll(
            adapter_a, tokens, chunk_size=8
        )
        nll_b, n_b = teacher_forced_chunked_nll(
            adapter_b, tokens, chunk_size=64
        )
        assert n_a == n_b
        assert nll_a == pytest.approx(nll_b, rel=1e-5, abs=1e-5)


class TestReturnTypes:
    def test_returns_tuple_of_float_int(self) -> None:
        adapter = _FakePPLAdapter()
        tokens = _make_tokens(16)
        nll, n = teacher_forced_chunked_nll(adapter, tokens, chunk_size=8)
        assert isinstance(nll, float)
        assert isinstance(n, int)

    def test_scored_token_count_is_seq_len_minus_one(self) -> None:
        adapter = _FakePPLAdapter()
        for seq_len in (1, 2, 16, 127):
            tokens = _make_tokens(seq_len)
            _, n = teacher_forced_chunked_nll(adapter, tokens, chunk_size=32)
            assert n == max(0, seq_len - 1), f"seq_len={seq_len}: n={n}"


class TestEdgeCases:
    def test_single_token_sequence_is_zero_nll(self) -> None:
        """seq_len=1 has no prior-context tokens to score → (0.0, 0)."""
        adapter = _FakePPLAdapter()
        tokens = _make_tokens(1)
        nll, n = teacher_forced_chunked_nll(adapter, tokens, chunk_size=8)
        assert nll == 0.0
        assert n == 0

    def test_empty_sequence_returns_zero(self) -> None:
        adapter = _FakePPLAdapter()
        tokens = mx.zeros((1, 0), dtype=mx.int32)
        nll, n = teacher_forced_chunked_nll(adapter, tokens, chunk_size=8)
        assert nll == 0.0
        assert n == 0

    def test_chunk_size_larger_than_seq_runs_in_one_forward(self) -> None:
        adapter = _FakePPLAdapter()
        tokens = _make_tokens(16)
        nll, n = teacher_forced_chunked_nll(
            adapter, tokens, chunk_size=1024
        )
        # One chunk = 16 tokens = 15 scored positions, no chunk-boundary.
        assert n == 15
        # Must match the reference (single-forward) computation.
        ref_nll, _ = _reference_nll_sum(tokens)
        assert nll == pytest.approx(ref_nll, rel=1e-5, abs=1e-5)


class TestInputValidation:
    def test_rejects_1d_token_ids(self) -> None:
        adapter = _FakePPLAdapter()
        tokens = mx.array([1, 2, 3], dtype=mx.int32)
        with pytest.raises(ValueError, match="must be 2-D"):
            teacher_forced_chunked_nll(adapter, tokens, chunk_size=8)

    def test_rejects_batch_greater_than_one(self) -> None:
        adapter = _FakePPLAdapter()
        tokens = mx.zeros((2, 8), dtype=mx.int32)
        with pytest.raises(ValueError, match="B=1"):
            teacher_forced_chunked_nll(adapter, tokens, chunk_size=8)

    def test_rejects_zero_chunk_size(self) -> None:
        adapter = _FakePPLAdapter()
        tokens = _make_tokens(8)
        with pytest.raises(ValueError, match="chunk_size"):
            teacher_forced_chunked_nll(adapter, tokens, chunk_size=0)

    def test_rejects_negative_chunk_size(self) -> None:
        adapter = _FakePPLAdapter()
        tokens = _make_tokens(8)
        with pytest.raises(ValueError, match="chunk_size"):
            teacher_forced_chunked_nll(adapter, tokens, chunk_size=-1)


class TestPerplexityFromNll:
    def test_ppl_exp_mean_nll(self) -> None:
        # NLL sum 10.0 over 5 tokens → mean = 2.0 → ppl = exp(2.0)
        assert perplexity_from_nll(10.0, 5) == pytest.approx(math.exp(2.0))

    def test_zero_tokens_returns_inf(self) -> None:
        assert perplexity_from_nll(0.0, 0) == float("inf")

    def test_negative_tokens_raises(self) -> None:
        # A negative scored-token count is an accounting bug in the
        # caller; staying silent (old behaviour: return inf) would
        # hide drift in the accumulator. Fail loudly.
        with pytest.raises(ValueError, match="n_tokens must be >= 0"):
            perplexity_from_nll(0.0, -1)

    def test_round_trip_with_oracle(self) -> None:
        """end-to-end: oracle output → perplexity_from_nll is finite
        and positive for the fake model."""
        adapter = _FakePPLAdapter()
        tokens = _make_tokens(64)
        nll, n = teacher_forced_chunked_nll(adapter, tokens, chunk_size=16)
        ppl = perplexity_from_nll(nll, n)
        assert math.isfinite(ppl)
        assert ppl > 0.0
