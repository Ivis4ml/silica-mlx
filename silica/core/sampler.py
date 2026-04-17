"""Sampler + LogitProcessor: logits -> token, per D-013 (PLAN.md).

The Sampler applies processors in the fixed order
`temperature -> repetition penalty -> top-k -> top-p -> sample` (matches mlx-lm).
Greedy fast path when `temperature == 0` bypasses the entire chain and returns
argmax directly.

`LogitProcessor` is a lightweight local `typing.Protocol` for type-hinting
within this module; it is NOT one of the five frozen core interfaces
(I-1..I-5) — see D-013.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import mlx.core as mx

from silica.core.sampling import SamplingParams


class LogitProcessor(Protocol):
    """Signature for a single logit-transform stage in the Sampler chain.

    Called once per decode step with the current logits, the already-generated
    token ids, and the request's `SamplingParams`. Returns transformed logits.
    Processors must be pure: no mutation of the input `logits` array.

    Users implementing a custom processor as a class should conform to this
    Protocol; internally the built-in stateless processors are plain functions
    typed via `_ProcessorFn` (mypy does not coerce bare functions to a
    `__call__`-Protocol).
    """

    def __call__(
        self,
        logits: mx.array,
        token_history: mx.array,
        params: SamplingParams,
    ) -> mx.array: ...


_ProcessorFn = Callable[[mx.array, mx.array, SamplingParams], mx.array]


def apply_temperature(
    logits: mx.array,
    _token_history: mx.array,
    params: SamplingParams,
) -> mx.array:
    if params.temperature == 1.0:
        return logits
    return logits / params.temperature


def apply_repetition_penalty(
    logits: mx.array,
    token_history: mx.array,
    params: SamplingParams,
) -> mx.array:
    if params.repetition_penalty == 1.0 or token_history.size == 0:
        return logits
    penalty = params.repetition_penalty
    gathered = logits[token_history]
    penalized = mx.where(gathered > 0, gathered / penalty, gathered * penalty)
    # put_along_axis is non-mutating; duplicate indices write the same value.
    return mx.put_along_axis(logits, token_history, penalized, axis=0)


def apply_top_k(
    logits: mx.array,
    _token_history: mx.array,
    params: SamplingParams,
) -> mx.array:
    if params.top_k is None:
        return logits
    k = params.top_k
    if k >= logits.shape[-1]:
        return logits
    # Ascending sort: k-th largest is at position -k.
    threshold = mx.sort(logits)[-k]
    neg_inf = mx.array(-float("inf"), dtype=logits.dtype)
    return mx.where(logits < threshold, neg_inf, logits)


def apply_top_p(
    logits: mx.array,
    _token_history: mx.array,
    params: SamplingParams,
) -> mx.array:
    if params.top_p is None:
        return logits
    p = params.top_p
    # Inverse-permutation pattern avoids an explicit scatter: work in
    # descending-by-logit sorted space, then map the keep mask back to
    # original positions via the inverse permutation of argsort.
    sorted_idx = mx.argsort(-logits)
    inv_idx = mx.argsort(sorted_idx)
    sorted_logits = logits[sorted_idx]
    sorted_probs = mx.softmax(sorted_logits)
    # cum_before[i] = sum(sorted_probs[:i]); first rank always kept since
    # cum_before[0] == 0 < p for any p > 0.
    cum_before = mx.cumsum(sorted_probs) - sorted_probs
    keep_sorted = cum_before < p
    keep = keep_sorted[inv_idx]
    neg_inf = mx.array(-float("inf"), dtype=logits.dtype)
    return mx.where(keep, logits, neg_inf)


class Sampler:
    """Applies the D-013 processor chain and samples a token.

    The chain is a class-level tuple; v0.1 does not let users reorder or extend
    it at runtime (ordering itself is part of D-013 — changing it requires a
    new decisions-log entry). Subclassing and overriding `PROCESSORS` is the
    escape hatch for v0.2 experiments such as grammar-constrained decoding
    (Q-011).
    """

    PROCESSORS: tuple[_ProcessorFn, ...] = (
        apply_temperature,
        apply_repetition_penalty,
        apply_top_k,
        apply_top_p,
    )

    def sample(
        self,
        logits: mx.array,
        token_history: Sequence[int] | mx.array,
        params: SamplingParams,
        key: mx.array | None = None,
    ) -> mx.array:
        """Transform logits through the chain and sample a token id.

        Returns a 0-d `mx.array` holding the sampled token id. v0.1 assumes
        1-D logits of shape `(vocab,)`; batched logits are a P-2 concern.
        """
        if params.is_greedy:
            return mx.argmax(logits, axis=-1)

        if not isinstance(token_history, mx.array):
            token_history = mx.array(list(token_history), dtype=mx.int32)

        for processor in self.PROCESSORS:
            logits = processor(logits, token_history, params)

        if key is not None:
            return mx.random.categorical(logits, key=key)
        return mx.random.categorical(logits)
