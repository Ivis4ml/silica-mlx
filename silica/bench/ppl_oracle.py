"""silica.bench.ppl_oracle — MLX-native teacher-forced streaming PPL.

Two entry points, one per P-5-C sub-unit:

- :func:`teacher_forced_chunked_nll` — P-5-C.1 landed. Cache-agnostic
  fp16 baseline oracle. One ``BatchKVCache`` allocated at sequence
  start via ``adapter.make_batch_cache([0])`` and *shared across
  chunks* (mlx-lm mutates in place); chunk-boundary scored via the
  previous chunk's last logit; within-chunk scored via shift-by-1.
  The codec is **never** exercised on this path — whatever cache the
  adapter hands out is used directly. Mirrors vqbench's
  ``validation/streaming_ppl.py`` math at the fp16 baseline.

- :func:`teacher_forced_chunked_nll_with_codec` — P-5-C.2 landing.
  Codec-backed arm. Each chunk ≥ 1 seeds a **fresh** per-chunk
  ``BatchKVCache`` from the prior prefix blocks (``prefix_cache.lookup``
  → ``fetch_detached_blocks`` → :func:`build_seeded_batch_kv`) — the
  decode-tensor hot path fires here. After each forward, the newly-
  grown aligned-prefix blocks are extracted from the post-forward
  cache and re-registered via ``prefix_cache.insert_detached`` so the
  next chunk can consume them through the codec — the encode-tensor
  hot path fires here. A payload under a non-identity codec (BlockTQ,
  ExtRaBitQ) loses information on the store round trip, so ΔPPL
  against the fp16 baseline becomes an observable. Under
  ``IdentityCodec`` the two entry points coincide (lossless round
  trip — regression-locked in the C.2 test suite).

Chunk-invariance invariant (C.1): ``PPL(chunk=128) == PPL(chunk=256)
== PPL(chunk=512)`` to within fp rounding. Tested in
:mod:`tests.test_ppl_oracle` (C.1) and :mod:`tests.test_ppl_oracle_codec`
(C.2 under identity codec — the codec-backed path must match the
shared-cache fp16 baseline to fp tolerance).

Scope boundary:

- Both entry points take ``token_ids`` as input; they do not tokenize
  raw text. C.2's WikiText-2 loader lands as a separate helper so the
  oracle itself stays text-free and unit-testable without a tokenizer
  or dataset dependency.
- Neither entry point routes through ``Engine.generate_batch`` — that
  is a sampling API that does not return positional logits. Both go
  straight through ``adapter._model`` (scheduler convention) using
  :func:`silica.mlx.runner.forward_batched_full` for all-position
  logits.
"""

from __future__ import annotations

import math
from typing import Any, cast

import mlx.core as mx

from silica.mlx.runner import forward_batched_full
from silica.scheduler.seed_kv import build_seeded_batch_kv


def _mx_1d_to_int_list(arr: mx.array) -> list[int]:
    """Convert a 1-D ``mx.array`` of integer token ids to ``list[int]``.

    Wraps ``.tolist()`` + ``cast`` to satisfy mypy's strict view of the
    union-returning stub; at runtime a 1-D integer array always gives
    a ``list[int]``.
    """
    raw = cast(list[int], arr.tolist())
    return [int(x) for x in raw]


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


def teacher_forced_chunked_nll_with_codec(
    adapter: Any,
    prefix_cache: Any,
    token_ids: mx.array,
    *,
    chunk_size: int = 256,
) -> tuple[float, int]:
    """Cumulative teacher-forced NLL routed through a ``RadixPrefixCache``.

    The P-5-C.2 codec-backed oracle arm. Unlike
    :func:`teacher_forced_chunked_nll` (which shares the adapter's
    ``BatchKVCache`` across chunks verbatim, never touching the
    codec), this entry point rebuilds the cache each chunk from the
    prior aligned-prefix blocks owned by ``prefix_cache`` — the store's
    ``encode_tensor`` / ``decode_tensor`` hot path fires on every
    insertion and retrieval.

    Per-chunk flow:

    1. **Chunk 0 (cold).** Allocate a fresh
       ``adapter.make_batch_cache([0])``; run the forward. The cache
       has no pre-forward K/V to decode.
    2. **Chunk i ≥ 1 (hit).** Look up the aligned prefix
       ``tokens[:i * chunk_size]`` in ``prefix_cache``;
       ``fetch_detached_blocks`` decodes every hit block through the
       store's per-side codecs (``decode_tensor`` fires here), then
       :func:`build_seeded_batch_kv` wires the decoded K/V into a
       fresh ``BatchKVCache(B=1)``. The new chunk is forwarded through
       that seeded cache.
    3. **After each forward.** Extract every aligned block from the
       post-forward cache up to the aligned high-water mark; call
       ``prefix_cache.insert_detached`` which registers the newly
       computed blocks through the store (``encode_tensor`` fires
       here) and touches the already-present duplicate-prefix blocks
       (no re-encode).

    Scoring mirrors :func:`teacher_forced_chunked_nll` verbatim:
    chunk-boundary token scored against the previous chunk's last
    logit (materialized via ``mx.contiguous`` + ``mx.eval`` before the
    next seed rebuild discards the current cache), within-chunk
    scored via the shift-by-1 pattern.

    Under :class:`silica.kvcache.codec.IdentityCodec` the codec is a
    lossless round trip and this function's returned ``(nll_sum,
    n_tokens)`` matches :func:`teacher_forced_chunked_nll` on the same
    token stream to fp rounding (regression-locked in the C.2 test
    suite). Under a lossy codec (BlockTQ, ExtRaBitQ), prior chunks'
    K/V is distorted on the encode/decode round trip; the resulting
    NLL delta against the fp16 baseline is the ΔPPL the bench row
    reports.

    Args:
        adapter: same contract as :func:`teacher_forced_chunked_nll`
            — must expose ``_model`` (the mlx-lm module) and
            ``make_batch_cache([0])`` (fresh per-layer ``BatchKVCache``
            list at batch=1, ``left_padding=[0]``).
        prefix_cache: a :class:`silica.kvcache.prefix.RadixPrefixCache`
            whose backing store is a
            :class:`silica.kvcache.store.SyntheticPrefixBlockStore`
            with the desired ``k_codec`` / ``v_codec`` installed. The
            cache is expected to be empty at entry (the oracle drives
            insertions itself) but is not required to be; pre-populated
            caches simply become a cold-start amortization for long
            sequences.
        token_ids: ``(1, seq_len)`` ``mx.array`` of token ids. B=1 only.
        chunk_size: tokens per forward. Must be a positive multiple of
            ``prefix_cache.block_size`` — the oracle relies on each
            chunk's boundary being aligned with block granularity so
            that ``prefix_cache.lookup`` returns a full prefix hit for
            every token seen so far. Misalignment would leave a tail
            of tokens outside the prefix cache's coverage and break
            the seeded-cache-per-chunk assumption. vqbench's
            ``streaming_ppl`` uses ``chunk_size=256`` against its own
            ``block_size=16`` KV-cache block; silica mirrors the
            divisibility contract.

    Returns:
        ``(nll_sum, n_tokens_scored)`` — same shape as the C.1 oracle.
        Feed into :func:`perplexity_from_nll` for PPL.

    Raises:
        ValueError: shape / B / chunk_size validation; also raised
            when ``chunk_size % prefix_cache.block_size != 0``.
    """
    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must be 2-D (1, seq_len); got shape "
            f"{tuple(token_ids.shape)}"
        )
    B, seq_len = token_ids.shape
    if B != 1:
        raise ValueError(
            f"teacher_forced_chunked_nll_with_codec supports B=1 only "
            f"(streaming PPL is per-sequence); got B={B}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    block_size = prefix_cache.block_size
    if chunk_size % block_size != 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be a positive multiple of "
            f"prefix_cache.block_size ({block_size}); misaligned chunks "
            f"leave a tail outside the prefix cache's block-granular "
            f"coverage and break the seeded-cache-per-chunk contract."
        )
    if seq_len == 0:
        return 0.0, 0

    model = adapter._model

    total_nll = 0.0
    total_tokens = 0
    prev_last_logit: mx.array | None = None
    num_layers: int | None = None

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_tokens = token_ids[:, start:end]  # (1, chunk_len)
        chunk_len = end - start

        if start == 0:
            # Cold path — no prior prefix, allocate a fresh cache.
            cache_list = adapter.make_batch_cache([0])
            num_layers = len(cache_list)
        else:
            # Hit path — look up aligned prefix and seed a fresh cache
            # from the store-owned decoded K/V. ``lookup`` increments
            # hit refs; the ``release`` call after ``fetch_detached_blocks``
            # matches the scheduler's admission-hit pattern.
            assert num_layers is not None
            prefix_token_list = _mx_1d_to_int_list(token_ids[0, :start])
            hit = prefix_cache.lookup(prefix_token_list)
            # ``lookup`` retained hit refs on every returned block; the
            # matched scheduler pattern releases them after the seeded
            # cache is materialized. Wrap in ``try/finally`` so a
            # partial-hit RuntimeError or an exception from
            # ``fetch_detached_blocks`` / ``build_seeded_batch_kv``
            # does not leak live-hit refs into the store — a leaked
            # ref would poison later eviction / reuse if the caller
            # shares this ``prefix_cache`` across sequences.
            try:
                if hit.num_hit_tokens != start:
                    raise RuntimeError(
                        f"prefix_cache.lookup returned {hit.num_hit_tokens} "
                        f"hit tokens; expected full prefix hit of {start} "
                        f"tokens (chunk_size % block_size contract broken, "
                        f"or prefix cache lost blocks between chunks)."
                    )
                detached_blocks = prefix_cache.fetch_detached_blocks(
                    hit.block_ids
                )
                cache_list = build_seeded_batch_kv(
                    detached_blocks, num_layers=num_layers
                )
            finally:
                prefix_cache.release(hit.block_ids)

        logits = forward_batched_full(
            model, chunk_tokens, cache_list
        )  # (1, chunk_len, V)

        # Chunk-boundary score — predict chunk_tokens[:, 0] from the
        # previous chunk's last-position logit. Same pattern as the
        # C.1 oracle.
        if prev_last_logit is not None:
            target = chunk_tokens[:, 0]
            total_nll += float(
                _cross_entropy_sum(prev_last_logit, target).item()
            )
            total_tokens += 1

        # Within-chunk score — shift-by-1 over chunk_tokens.
        if chunk_len > 1:
            within_logits = logits[:, :-1, :]
            within_targets = chunk_tokens[:, 1:]
            total_nll += float(
                _cross_entropy_sum(
                    within_logits.reshape(-1, within_logits.shape[-1]),
                    within_targets.reshape(-1),
                ).item()
            )
            total_tokens += chunk_len - 1

        # Materialize the boundary logit now — the next iteration
        # throws away ``cache_list`` and the lazy slice would lose
        # its source. Same detach pattern as the C.1 oracle.
        prev_last_logit = mx.contiguous(
            logits[:, -1, :].astype(mx.float32)
        )
        mx.eval(prev_last_logit)

        # Extract every aligned block covered by the current cache
        # ([0, aligned_end)) and hand the prefix over to
        # ``prefix_cache.insert_detached``. Duplicate-prefix branches
        # (blocks already registered from prior chunks) are touched
        # without re-encoding; only the newly-computed tail blocks
        # fire ``encode_tensor`` on the store's codecs.
        aligned_end = (end // block_size) * block_size
        if aligned_end > 0:
            detached_to_insert = _extract_aligned_blocks_from_seeded_cache(
                cache_list, block_size, aligned_end
            )
            prefix_tokens_to_insert = _mx_1d_to_int_list(
                token_ids[0, :aligned_end]
            )
            prefix_cache.insert_detached(
                prefix_tokens_to_insert, detached_to_insert
            )

    return total_nll, total_tokens


def _extract_aligned_blocks_from_seeded_cache(
    cache_list: list[Any],
    block_size: int,
    num_aligned_tokens: int,
) -> list[list[tuple[mx.array, mx.array]]]:
    """Slice per-block K/V out of a B=1 ``BatchKVCache`` list.

    Returns a list indexed ``[block_idx][layer_idx] -> (K, V)`` that
    :meth:`silica.kvcache.prefix.RadixPrefixCache.insert_detached`
    consumes. Each ``(K, V)`` has shape
    ``(1, n_kv_heads, block_size, head_dim)`` — the shape contract
    ``build_seeded_batch_kv`` and ``register_detached`` share.

    Precondition: every cache in ``cache_list`` has
    ``left_padding[0] == 0`` (B=1 fresh admission). The seeded caches
    the oracle builds satisfy this by construction (:func:`build_seeded_batch_kv`
    sets ``left_padding=[0]``); the cold-path cache from
    ``adapter.make_batch_cache([0])`` does too.

    Slices are passed through ``mx.contiguous`` **and**
    ``mx.eval`` before return so the extracted arrays carry their own
    materialized backing, not lazy views into the cache's internal
    tensors. The oracle's outer loop discards ``cache_list`` on the
    next iteration (chunks rebuild a fresh seeded cache) — a lazy
    slice held by the store would source from a dead cache on
    subsequent decode. Mirrors the eager-materialization step
    :meth:`silica.scheduler.batcher.ContinuousBatcher._extract_and_insert_prefix`
    performs before every ``insert_detached`` call.

    Raises:
        ValueError: if ``num_aligned_tokens`` is not a non-negative
            multiple of ``block_size``, or any layer's cache is
            missing its K/V (forward never ran).
    """
    if num_aligned_tokens < 0:
        raise ValueError(
            f"num_aligned_tokens must be >= 0; got {num_aligned_tokens}"
        )
    if num_aligned_tokens % block_size != 0:
        raise ValueError(
            f"num_aligned_tokens ({num_aligned_tokens}) must be a "
            f"multiple of block_size ({block_size})"
        )
    num_blocks = num_aligned_tokens // block_size
    num_layers = len(cache_list)

    detached: list[list[tuple[mx.array, mx.array]]] = []
    for b_idx in range(num_blocks):
        start = b_idx * block_size
        end = start + block_size
        per_layer: list[tuple[mx.array, mx.array]] = []
        for layer_idx in range(num_layers):
            cache_obj = cache_list[layer_idx]
            keys = cache_obj.keys
            values = cache_obj.values
            if keys is None or values is None:
                raise ValueError(
                    f"layer {layer_idx} has no cache state "
                    f"(forward never ran or cache was filtered)."
                )
            k = mx.contiguous(keys[:, :, start:end, :])
            v = mx.contiguous(values[:, :, start:end, :])
            per_layer.append((k, v))
        detached.append(per_layer)

    # Eager-materialize every extracted slice before the caller
    # registers it. IdentityCodec passes the ``mx.array`` reference
    # straight through into the store's ``_DetachedLayer``, and
    # quantizing codecs may produce payload fields that are still
    # lazy views over these slices. Either way the store must not
    # hold a view into the soon-to-be-discarded ``cache_list``.
    mx.eval(
        *[
            arr
            for per_layer in detached
            for (k, v) in per_layer
            for arr in (k, v)
        ]
    )
    return detached


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
