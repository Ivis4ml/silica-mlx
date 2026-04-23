"""silica.bench.ppl_oracle — MLX-native teacher-forced streaming PPL.

P-5-C.1 core: evaluate perplexity of a token sequence under the model
currently bound to a :class:`ModelAdapter`, with chunked streaming so
arbitrarily long contexts do not need to fit in one forward pass.

Algorithm mirrors vqbench's ``vqbench/validation/streaming_ppl.py``
``evaluate_streaming_ppl`` verbatim at the math level (manual NLL,
chunk-boundary scoring, shared cache across chunks):

1. Allocate one cache list at sequence start via
   ``adapter.make_batch_cache([0])``. Do **not** reset between chunks —
   mlx-lm mutates the cache in place so prior chunks' K/V remain
   available when the next chunk runs. This is what makes the PPL
   result chunk-size invariant.
2. For each ``chunk`` of ``chunk_size`` tokens, forward through the
   model using the (growing) cache list and grab all-position logits
   ``(1, chunk_len, V)``. Use :func:`forward_batched_full` — the
   last-position-only ``forward_batched`` is insufficient.
3. **Chunk boundary scoring**: starting from the second chunk, the
   first token of the current chunk is predicted from the last
   position of the previous chunk's logits. Add
   ``CE(prev_last_logit, chunk_tokens[:, 0])`` to the running nll.
4. **Within-chunk scoring**: for every position ``i > 0`` within a
   chunk, the model predicts ``chunk_tokens[i]`` from
   ``logits[:, i - 1, :]``. The standard shift-by-1 computation
   ``CE(logits[:, :-1, :], chunk_tokens[:, 1:])`` captures this.
5. Total scored tokens = ``seq_len - 1`` (the very first token has no
   prior context; every other token is scored exactly once).

Chunk invariance is a correctness invariant: PPL(chunk=128) must
equal PPL(chunk=256) must equal PPL(chunk=512) to within fp rounding.
Test pinned in :mod:`tests.test_ppl_oracle`.

Scope boundary:

- This module takes ``token_ids`` as input; it does not tokenize raw
  text. C.2 will add a tokenizer wrapper for the WikiText-2 bench
  row. Keeping the oracle helper text-free makes unit testing
  cheaper (no network / dataset dependency) and lets the same helper
  power future PPL evaluations against arbitrary token streams.
- The compressed KV-cache arm (compressed-codec streaming PPL) is
  also deferred — v0.1 ships PPL against the *bound* adapter's cache
  (fp16 `BatchKVCache` or compressed `SyntheticPrefixBlockStore`-
  derived caches once wiring exists). For P-5-C.1 the oracle is used
  by the fp16 baseline row; later scenarios swap the adapter's cache
  to the compressed path.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from silica.mlx.runner import forward_batched_full


def teacher_forced_chunked_nll(
    adapter: Any,
    token_ids: mx.array,
    *,
    chunk_size: int = 256,
) -> tuple[float, int]:
    """Compute cumulative teacher-forced NLL over a token sequence.

    Drives ``adapter._model`` with a fresh batched cache (from
    ``adapter.make_batch_cache([0])``) that is reused across chunks
    for the whole sequence. Scores every token except the very first:
    each later token is predicted from the logits at the prior
    position (from this chunk when available, or the tail of the
    previous chunk at the chunk-boundary case).

    The ``_model`` attribute access mirrors
    :mod:`silica.scheduler.batcher` — every concrete adapter
    (Qwen3, Gemma4, ...) stores the built mlx-lm module as
    ``self._model``. Formalizing this as a Protocol method is a v0.2
    extension out of P-5 scope; the current attribute-access
    convention is load-bearing for the scheduler and now for PPL.

    Args:
        adapter: a built :class:`silica.models.adapter.ModelAdapter`.
            Must expose ``_model`` (the mlx-lm module) and
            ``make_batch_cache([0])`` (per-layer batched cache list at
            batch=1).
        token_ids: ``(1, seq_len)`` ``mx.array`` of token ids. Batch
            size must be 1 — streaming PPL is a per-sequence quantity.
        chunk_size: tokens per forward pass. Must be >= 1. Larger
            chunks pay fewer Python-level forward calls but risk
            peak-memory spikes on long contexts; vqbench defaults to
            512, silica defaults to 256 for fp16 K/V baseline
            measurements on the M5 Pro target.

    Returns:
        ``(nll_sum, n_tokens_scored)`` — cumulative negative
        log-likelihood (sum, not mean) and the count of tokens
        scored (``seq_len - 1``). Caller chooses whether to feed this
        into :func:`perplexity_from_nll` or accumulate across
        sequences first.

    Raises:
        ValueError: if ``token_ids`` has the wrong shape / batch, or
            ``chunk_size < 1``.
    """
    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must be 2-D (1, seq_len); got shape "
            f"{tuple(token_ids.shape)}"
        )
    B, seq_len = token_ids.shape
    if B != 1:
        raise ValueError(
            f"teacher_forced_chunked_nll supports B=1 only (streaming "
            f"PPL is per-sequence); got B={B}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    if seq_len == 0:
        return 0.0, 0

    model = adapter._model
    cache_list = adapter.make_batch_cache([0])

    total_nll = 0.0
    total_tokens = 0
    prev_last_logit: mx.array | None = None  # (1, V) from previous chunk's last position

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_tokens = token_ids[:, start:end]  # (1, chunk_len)
        chunk_len = end - start

        logits = forward_batched_full(
            model, chunk_tokens, cache_list
        )  # (1, chunk_len, V)

        # Chunk-boundary score: predict chunk_tokens[:, 0] from the
        # previous chunk's last-position logits. First chunk has no
        # prior context, so skip.
        if prev_last_logit is not None:
            target = chunk_tokens[:, 0]  # (1,)
            total_nll += float(_cross_entropy_sum(prev_last_logit, target).item())
            total_tokens += 1

        # Within-chunk score: predict chunk_tokens[:, i] from
        # logits[:, i - 1, :] for every i in [1, chunk_len). Shape
        # reduces to logits[:, :-1, :] vs chunk_tokens[:, 1:].
        if chunk_len > 1:
            within_logits = logits[:, :-1, :]  # (1, chunk_len - 1, V)
            within_targets = chunk_tokens[:, 1:]  # (1, chunk_len - 1)
            total_nll += float(
                _cross_entropy_sum(
                    within_logits.reshape(-1, within_logits.shape[-1]),
                    within_targets.reshape(-1),
                ).item()
            )
            total_tokens += chunk_len - 1

        # Save the last logit of this chunk for the next chunk's
        # boundary token. Slice is (1, V) cast to fp32 and forced to
        # evaluate via ``mx.contiguous`` + ``mx.eval`` now, rather
        # than kept as a lazy view on ``logits``: the next chunk's
        # forward mutates the shared cache in place, and we do not
        # want the boundary logit's deferred graph node to pick up
        # post-mutation state. Mirrors the ``mx.contiguous`` +
        # ``mx.eval`` detach pattern used by
        # :mod:`silica.scheduler.batcher` before a cache source is
        # filtered.
        prev_last_logit = mx.contiguous(
            logits[:, -1, :].astype(mx.float32)
        )
        mx.eval(prev_last_logit)

    return total_nll, total_tokens


def perplexity_from_nll(nll_sum: float, n_tokens: int) -> float:
    """Convert cumulative NLL sum + token count to perplexity.

    PPL = exp(mean NLL) = exp(nll_sum / n_tokens).

    Args:
        nll_sum: cumulative negative log-likelihood (natural log).
        n_tokens: number of tokens the NLL was summed over. Must be
            ``>= 0``; a zero-token run is a measurement degenerate
            case (returns ``inf``), a negative count is a caller-
            side accounting bug (raises ``ValueError``).

    Returns:
        Perplexity. ``n_tokens == 0`` returns ``float('inf')`` rather
        than raising — keeps call sites that concatenate multi-
        sequence results clean.

    Raises:
        ValueError: if ``n_tokens < 0``. A negative scored-token
            count cannot arise from a correct oracle run; silently
            collapsing it to ``inf`` would hide drift in the
            accumulator, so we fail loudly instead.
    """
    if n_tokens < 0:
        raise ValueError(
            f"n_tokens must be >= 0; got {n_tokens}. Negative scored-"
            f"token count indicates a caller-side accounting bug "
            f"(e.g. double-subtraction of the first-token skip)."
        )
    if n_tokens == 0:
        return float("inf")
    return math.exp(nll_sum / n_tokens)


# =============================================================================
# Cross-entropy helper
# =============================================================================


def _cross_entropy_sum(logits: mx.array, targets: mx.array) -> mx.array:
    """Sum of token-wise cross-entropy: ``sum_i -log p_i[targets_i]``.

    Computes cross-entropy manually in fp32 via ``logits - logsumexp``
    + gather. Keeping it as ``mx.*`` primitives rather than reaching
    into ``mlx.nn`` keeps the oracle's hot path on the lightweight
    ``mlx.core`` module only.

    Args:
        logits: ``(N, V)`` float32 / float16 logits.
        targets: ``(N,)`` integer token ids.

    Returns:
        Scalar ``mx.array`` with the sum of token-wise NLL values.
    """
    logits_fp32 = logits.astype(mx.float32)
    # log_softmax(x) = x - logsumexp(x) — numerically stable per the
    # logsumexp subtraction trick.
    log_probs = logits_fp32 - mx.logsumexp(logits_fp32, axis=-1, keepdims=True)
    targets_i32 = targets.astype(mx.int32)
    # Gather log-prob at each target; take_along_axis expects index
    # shape matching the reduced-over axis position.
    gathered = mx.take_along_axis(
        log_probs, targets_i32[..., None], axis=-1
    )  # (N, 1)
    return -mx.sum(gathered)
