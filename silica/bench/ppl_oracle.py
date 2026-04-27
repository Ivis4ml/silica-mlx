"""silica.bench.ppl_oracle — MLX-native teacher-forced streaming PPL.

Three entry points, one per P-5-C / P-5-D oracle sub-unit:

- :func:`teacher_forced_chunked_nll` — P-5-C.1 landed. Cache-agnostic
  fp16 baseline oracle. One ``BatchKVCache`` allocated at sequence
  start via ``adapter.make_batch_cache([0])`` and *shared across
  chunks* (mlx-lm mutates in place); chunk-boundary scored via the
  previous chunk's last logit; within-chunk scored via shift-by-1.
  The codec is **never** exercised on this path — whatever cache the
  adapter hands out is used directly. Mirrors vqbench's
  ``validation/streaming_ppl.py`` math at the fp16 baseline.

- :func:`teacher_forced_chunked_nll_with_codec` — P-5-C.2 landing.
  Codec-backed arm (``codec_quality_path="prefix_store_post_rope"``).
  Each chunk ≥ 1 seeds a **fresh** per-chunk ``BatchKVCache`` from
  the prior prefix blocks (``prefix_cache.lookup`` →
  ``fetch_detached_blocks`` → :func:`build_seeded_batch_kv`) — the
  decode-tensor hot path fires here. After each forward, the newly-
  grown aligned-prefix blocks are extracted from the post-forward
  cache and re-registered via ``prefix_cache.insert_detached`` so the
  next chunk can consume them through the codec — the encode-tensor
  hot path fires here. Noise is injected *after* RoPE, on the
  post-rotation K that the cache actually stores. A payload under a
  non-identity codec (BlockTQ, ExtRaBitQ) loses information on the
  store round trip, so ΔPPL against the fp16 baseline becomes an
  observable of the production store path. Under ``IdentityCodec``
  this entry point matches the C.1 baseline to fp tolerance
  (regression-locked in the C.2 test suite).

- :func:`teacher_forced_chunked_nll_vqbench_aligned` — P-5-D.2a
  landing (``codec_quality_path="vqbench_aligned"``). Mirrors
  vqbench's ``methods.common.monkey_patch._QuantizedProj`` semantic:
  wraps the attention's ``k_proj`` (and optionally ``v_proj``) in a
  quantize→dequantize pass so *all* K/V (current + past) flow through
  the codec in pre-RoPE space on every forward. Shape unchanged —
  the wrapped projection still returns the ``(B, L, nkv*hd)`` layout
  the downstream RoPE + cache expect. This entry point exists because
  the P-5-D.2 investigation (see ``plans/P5_D2_INVESTIGATION``) found
  that post-RoPE noise injection gives a ΔPPL roughly 20× larger than
  the pre-RoPE vqbench baseline at the same Frobenius, even though
  the raw K Frobenius is bit-identical between the two spaces. The
  two paths therefore answer different questions:

  * ``prefix_store_post_rope`` — "how much quality does the production
    prefix-cache store cost, in the post-RoPE space the cache lives
    in?"
  * ``vqbench_aligned`` — "what is vqbench's pre-RoPE-codec quality
    claim when silica runs its own BlockTQ/ExtRaBitQ port on the same
    real Qwen activations?"

  Both numbers are valid and live in metadata under explicit
  ``codec_quality_path`` labels; the C.6 ΔPPL vs vqbench cross-check
  binds against ``vqbench_aligned`` because vqbench itself wraps
  pre-RoPE projections, i.e. the two sides now inject noise in the
  same space. Whether the residual row-level gap closes the old
  per-row ``_compute_gap_fields`` epsilon gate (``0.01`` abs /
  ``0.1%`` rel) or needs a mean-aggregated / noise-window rule is
  the subject of P-5-D.3, not D.2a. The production store path is
  not silenced — its ΔPPL is preserved as a separate observable.

Chunk-invariance invariant (C.1): ``PPL(chunk=128) == PPL(chunk=256)
== PPL(chunk=512)`` to within fp rounding. Tested in
:mod:`tests.test_ppl_oracle` (C.1) and :mod:`tests.test_ppl_oracle_codec`
(C.2 under identity codec — the codec-backed path must match the
shared-cache fp16 baseline to fp tolerance).

Scope boundary:

- All three entry points take ``token_ids`` as input; they do not
  tokenize raw text. C.2's WikiText-2 loader lands as a separate
  helper so the oracle itself stays text-free and unit-testable
  without a tokenizer or dataset dependency.
- None of the entry points route through ``Engine.generate_batch`` —
  that is a sampling API that does not return positional logits. All
  three go straight through ``adapter._model`` (scheduler convention)
  using :func:`silica.mlx.runner.forward_batched_full` for
  all-position logits.
- The vqbench-aligned path monkey-patches attention modules on
  ``adapter._model`` for the duration of a single oracle call. The
  wrapper is installed at function entry and restored in a
  ``try/finally`` so an exception during the forward does not leave
  the adapter in a partially-wrapped state. This patching is scoped
  to the benchmark oracle — nothing on the serving hot path depends
  on the wrapped layout, and no production code should call this
  entry point.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

from silica.kvcache.codec import VectorCodec
from silica.mlx.runner import forward_batched_full
from silica.models.recurrent import (
    RecurrentSnapshot,
    RecurrentStateAdapter,
)
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
    attn_layer_indices: list[int] | None = None

    # P-3-C5-step2-W: hybrid (recurrent) adapters carry per-layer
    # recurrent state in DeltaNet ``ArraysCache`` slots that the
    # K/V codec doesn't touch. Without explicit handling, each hit
    # chunk's freshly-built cache would lose the prior chunks'
    # recurrent trajectory, inflating PPL artificially. Mirror C5.3's
    # capture/restore at chunk boundaries (advisor sharpening:
    # snapshot/restore is the highest-fidelity option for the
    # "compare to vqbench" claim — same per-block snapshot
    # contract C5.3.3b validates byte-exact in production).
    is_hybrid_recurrent = isinstance(adapter, RecurrentStateAdapter)
    prev_recurrent_snapshot: RecurrentSnapshot | None = None

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_tokens = token_ids[:, start:end]  # (1, chunk_len)
        chunk_len = end - start

        if start == 0:
            # Cold path — no prior prefix, allocate a fresh cache.
            # The cache may be heterogeneous on hybrid adapters
            # (interleaved ``ArraysCache`` + ``BatchKVCache``).
            cache_list = adapter.make_batch_cache([0])
            attn_layer_indices = [
                i
                for i, c in enumerate(cache_list)
                if isinstance(c, BatchKVCache)
            ]
        else:
            # Hit path — look up aligned prefix and seed a fresh cache
            # from the store-owned decoded K/V. ``lookup`` increments
            # hit refs; the ``release`` call after ``fetch_detached_blocks``
            # matches the scheduler's admission-hit pattern.
            assert attn_layer_indices is not None
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
                # P-3-C5-step2-W: per-block detached_blocks carry one
                # ``(K, V)`` per attention-layer position (the bench
                # extract below stores it that way). For hybrid
                # adapters, build an empty heterogeneous cache via
                # ``adapter.make_batch_cache`` and interleave the
                # seeded ``BatchKVCache`` into the attention positions;
                # DeltaNet positions remain empty ``ArraysCache`` (the
                # recurrent state is restored separately below). For
                # pure-attention adapters, ``attn_layer_indices`` is
                # ``range(num_layers)`` and the assembly degenerates
                # to today's all-``BatchKVCache`` shape.
                seeded_attn = build_seeded_batch_kv(
                    detached_blocks, num_layers=len(attn_layer_indices)
                )
                cache_list = adapter.make_batch_cache([0])
                for pos, src_idx in enumerate(attn_layer_indices):
                    cache_list[src_idx] = seeded_attn[pos]
            finally:
                prefix_cache.release(hit.block_ids)

            # P-3-C5-step2-W: restore the prior chunk's recurrent state
            # so DeltaNet layers continue from the correct trajectory
            # rather than re-initialising lazily on this chunk's first
            # forward. Mirrors ``_admit_single_hit_row``'s Insertion B
            # (P-3-C5.3.3a) — the cache_list at this point has empty
            # ``ArraysCache`` slots that ``_splice_row`` case 4 fills
            # in directly from the snapshot.
            if is_hybrid_recurrent and prev_recurrent_snapshot is not None:
                assert isinstance(adapter, RecurrentStateAdapter)
                adapter.restore_recurrent_state(
                    cache_list, 0, prev_recurrent_snapshot
                )

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

        # P-3-C5-step2-W: capture the post-forward recurrent state so
        # the next chunk's hit path can restore it onto the freshly-
        # built cache. The capture covers every DeltaNet layer; pure-
        # attention adapters skip this (no RecurrentStateAdapter
        # mixin → predicate False).
        if is_hybrid_recurrent:
            assert isinstance(adapter, RecurrentStateAdapter)
            prev_recurrent_snapshot = adapter.snapshot_recurrent_state(
                cache_list, row_idx=0
            )

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
    """Slice per-block K/V out of a B=1 cache list.

    Returns a list indexed ``[block_idx][attn_pos] -> (K, V)`` that
    :meth:`silica.kvcache.prefix.RadixPrefixCache.insert_detached`
    consumes. Each ``(K, V)`` has shape
    ``(1, n_kv_heads, block_size, head_dim)`` — the shape contract
    ``build_seeded_batch_kv`` and ``register_detached`` share.

    P-3-C5-step2-W: under hybrid adapters the cache_list is
    heterogeneous (interleaved ``ArraysCache`` for DeltaNet layers
    + ``BatchKVCache`` for full-attention layers). DeltaNet layers
    carry recurrent state, not token K/V — there is no per-block
    slicing to do for them. The helper introspects each layer's
    type and only extracts from ``BatchKVCache`` layers; the
    output's per-block list length is the number of attention
    layers, not the total transformer layer count. Pure-attention
    adapters degenerate to today's behaviour (every layer is a
    ``BatchKVCache`` so ``attn_pos == layer_idx``).

    Precondition: every ``BatchKVCache`` in ``cache_list`` has
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
            multiple of ``block_size``, or any attention layer's
            cache is missing its K/V (forward never ran).
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
    attn_layer_indices = [
        i
        for i, c in enumerate(cache_list)
        if isinstance(c, BatchKVCache)
    ]

    detached: list[list[tuple[mx.array, mx.array]]] = []
    for b_idx in range(num_blocks):
        start = b_idx * block_size
        end = start + block_size
        per_layer: list[tuple[mx.array, mx.array]] = []
        for layer_idx in attn_layer_indices:
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


# =============================================================================
# P-5-F F.0b — pre-RoPE store oracle (Option A minimum prototype).
#
# Mirrors ``teacher_forced_chunked_nll_with_codec`` (the post-RoPE
# production path) but inserts an inverse-RoPE round-trip at the
# encode seam and a forward-RoPE re-application at the decode seam
# so the codec sees pre-RoPE K. RoPE is orthogonal, so injecting
# noise in pre-RoPE space is mathematically equivalent to vqbench's
# ``_QuantizedProj`` injection (per ``plans/P5_F_OPENING.md`` §3.1).
# Persistence and block-grained encoding match the production
# store, unlike the D.2a oracle which re-encodes per chunk.
#
# This is a flag-gated minimum prototype, NOT the final F.1+
# adapter Protocol. Goal: measure ΔPPL on the F.0 (b) gate before
# committing to the full Option A architecture (F.1: adapter
# `apply_rope_inverse_to_k`; F.2: store-side flag).
#
# V is not rotated in any silica-supported family
# (P5_F_OPENING.md §4) so V passes through unchanged on this
# path; only K participates in the RoPE round-trip.
# =============================================================================


def _apply_rope_to_k_block(
    k_block: mx.array,
    rope_instance: Any,
    *,
    base_offset: int,
    inverse: bool,
) -> mx.array:
    """Apply forward or inverse RoPE to one (1, n_kv_heads, block_size, head_dim) K block.

    Inverse path: ``mx.fast.rope(..., freqs=-freqs)`` (or ``scale=-1.0``
    on the freqs path), verified to match identity round-trip at
    Frobenius ratio 9.6e-8 fp32 / 1.36e-4 fp16 in the F.0b numerical
    readiness probe (``plans/P5_F_OPENING.md`` §5.1).

    Supports two RoPE classes that silica's adapters actually
    instantiate (F.1 inventory in ``plans/P5_F_OPENING.md`` §4):

    - ``nn.RoPE`` — exposes ``dims`` / ``base`` / ``scale`` /
      ``traditional``; freqs is reconstructed as
      ``base ** (mx.arange(0, dims, 2) / dims)``.
    - ``ProportionalRoPE`` — exposes ``_freqs`` (with infinite
      entries for the non-rotated dimensions); ``mx.fast.rope``
      handles the partial-rotation case via the ``dims`` arg
      naming the rotated prefix.

    Other RoPE classes (``Llama3RoPE``, ``YarnRoPE``, ``SuScaledRoPE``)
    raise ``NotImplementedError`` here — their forward paths apply
    additional ``mscale`` factors before fast.rope, which the
    inverse must also undo. None are in use on silica's targets,
    so handling them is forward-compat work for F.1 not F.0b.
    """
    cls_name = type(rope_instance).__name__
    if cls_name == "RoPE":
        # mlx.nn.RoPE — reconstruct freqs from base.
        dims = rope_instance.dims
        base = rope_instance.base
        traditional = rope_instance.traditional
        scale = rope_instance.scale
        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    elif cls_name == "ProportionalRoPE":
        dims = rope_instance.dims
        traditional = rope_instance.traditional
        scale = 1.0
        freqs = rope_instance._freqs
    else:
        raise NotImplementedError(
            f"F.0b prototype does not handle RoPE class {cls_name!r} "
            f"(only nn.RoPE and ProportionalRoPE are in use on "
            f"silica's targets per plans/P5_F_OPENING.md §4); "
            f"add explicit dispatch in F.1 if a new family routes "
            f"here. See plans/P5_F_OPENING.md §6.2 for the F.1 "
            f"in-use vs forward-compat coverage policy."
        )
    if inverse:
        freqs = -freqs
    return mx.fast.rope(
        k_block,
        dims,
        traditional=traditional,
        base=None,
        scale=scale,
        offset=base_offset,
        freqs=freqs,
    )


def _round_trip_inverse_rope_for_extract(
    detached_blocks: list[list[tuple[mx.array, mx.array]]],
    *,
    block_size: int,
    rope_instances_per_attn_layer: list[Any],
) -> list[list[tuple[mx.array, mx.array]]]:
    """Inverse-RoPE every K block extracted from a post-RoPE cache.

    Input shape per outer index: ``[block_idx][attn_pos] -> (K, V)``
    where K is post-RoPE shape ``(1, n_kv_heads, block_size, head_dim)``
    and V is the same shape unrotated.

    Output preserves the structure; K is replaced with its pre-RoPE
    counterpart (``mx.fast.rope`` with ``freqs=-freqs`` at offset
    ``block_idx * block_size``). V passes through unchanged.

    The store / codec downstream sees pre-RoPE K, matching the
    space vqbench's ``_QuantizedProj`` injects in.
    """
    return [
        [
            (
                _apply_rope_to_k_block(
                    k,
                    rope_instances_per_attn_layer[attn_pos],
                    base_offset=block_idx * block_size,
                    inverse=True,
                ),
                v,
            )
            for attn_pos, (k, v) in enumerate(per_attn_layer)
        ]
        for block_idx, per_attn_layer in enumerate(detached_blocks)
    ]


def _round_trip_forward_rope_for_admit(
    detached_blocks: list[list[tuple[mx.array, mx.array]]],
    *,
    block_size: int,
    rope_instances_per_attn_layer: list[Any],
) -> list[list[tuple[mx.array, mx.array]]]:
    """Forward-RoPE every K block decoded from the pre-RoPE store.

    Inverse of :func:`_round_trip_inverse_rope_for_extract`. K is
    re-rotated at offset ``block_idx * block_size`` so the seeded
    ``BatchKVCache`` carries post-RoPE K — matching what mlx-lm's
    attention forward expects when the cache is passed in.
    """
    return [
        [
            (
                _apply_rope_to_k_block(
                    k,
                    rope_instances_per_attn_layer[attn_pos],
                    base_offset=block_idx * block_size,
                    inverse=False,
                ),
                v,
            )
            for attn_pos, (k, v) in enumerate(per_attn_layer)
        ]
        for block_idx, per_attn_layer in enumerate(detached_blocks)
    ]


def teacher_forced_chunked_nll_with_codec_pre_rope(
    adapter: Any,
    prefix_cache: Any,
    token_ids: mx.array,
    *,
    chunk_size: int = 256,
) -> tuple[float, int]:
    """P-5-F F.0b prototype — pre-RoPE store via inverse-RoPE round-trip.

    Mirrors :func:`teacher_forced_chunked_nll_with_codec` block-for-
    block but adds two RoPE round-trips:

    - **Encode seam** (after extract, before insert_detached): every
      K block extracted from the seeded ``BatchKVCache`` (post-RoPE)
      is inverse-rotated to pre-RoPE space before the codec sees it.
    - **Decode seam** (after fetch_detached_blocks, before
      build_seeded_batch_kv): every K block decoded from the store
      (still pre-RoPE) is forward-rotated to post-RoPE space so the
      seeded cache mlx-lm consumes carries post-RoPE K as the
      forward expects.

    V is unrotated in mlx-lm attention for every silica-supported
    family (P5_F_OPENING.md §4), so V passes through unchanged.

    Math: RoPE is orthogonal, so noise ε injected by the codec in
    pre-RoPE space and then forward-rotated equals the same ε
    rotated to post-RoPE space — the same noise distribution
    vqbench's ``_QuantizedProj`` produces. The chunk-boundary cost
    the post-RoPE store pays comes from RoPE-coupled noise
    interacting with subsequent positions during refills; that
    coupling vanishes when the codec's noise lives in pre-RoPE
    space until it is rotated alongside the K it belongs to.

    Closes the production-vs-D.2a-oracle ΔPPL gap (currently +20
    PPL on Qwen3-0.6B + BlockTQ b64 b4 vs +0.51 PPL on the D.2a
    oracle) when the F.0 (b) gate (§6.1) accepts.

    Args / Returns: same contract as
    :func:`teacher_forced_chunked_nll_with_codec`.

    Raises: same shape / chunk validation, plus
    :class:`NotImplementedError` if any attention layer's RoPE
    instance is outside the F.0b-supported set
    (``nn.RoPE`` or ``ProportionalRoPE``); see
    :func:`_apply_rope_to_k_block`.
    """
    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must be 2-D (1, seq_len); got shape "
            f"{tuple(token_ids.shape)}"
        )
    B, seq_len = token_ids.shape
    if B != 1:
        raise ValueError(
            f"teacher_forced_chunked_nll_with_codec_pre_rope supports "
            f"B=1 only (streaming PPL is per-sequence); got B={B}"
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
    attn_layer_indices: list[int] | None = None
    rope_instances_per_attn_layer: list[Any] | None = None

    is_hybrid_recurrent = isinstance(adapter, RecurrentStateAdapter)
    prev_recurrent_snapshot: RecurrentSnapshot | None = None

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_tokens = token_ids[:, start:end]
        chunk_len = end - start

        if start == 0:
            cache_list = adapter.make_batch_cache([0])
            attn_layer_indices = [
                i
                for i, c in enumerate(cache_list)
                if isinstance(c, BatchKVCache)
            ]
            # Capture per-attention-layer RoPE handles up front so the
            # encode-/decode-seam wrappers can apply forward / inverse
            # rotation without re-resolving on every chunk. Reads from
            # the live mlx-lm model the adapter built; same handles
            # mlx-lm's attention forward uses, so any RoPE
            # configuration the adapter carries (theta / freqs /
            # partial-rotary-factor) flows through naturally.
            rope_instances_per_attn_layer = [
                model.layers[layer_idx].self_attn.rope
                for layer_idx in attn_layer_indices
            ]
        else:
            assert attn_layer_indices is not None
            assert rope_instances_per_attn_layer is not None
            prefix_token_list = _mx_1d_to_int_list(token_ids[0, :start])
            hit = prefix_cache.lookup(prefix_token_list)
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
                # F.0b decode seam: K is pre-RoPE in the store
                # (encode seam below stripped RoPE). Apply forward
                # RoPE so the seeded cache carries post-RoPE K as
                # mlx-lm's attention expects.
                detached_blocks = _round_trip_forward_rope_for_admit(
                    list(detached_blocks),
                    block_size=block_size,
                    rope_instances_per_attn_layer=(
                        rope_instances_per_attn_layer
                    ),
                )
                seeded_attn = build_seeded_batch_kv(
                    detached_blocks, num_layers=len(attn_layer_indices)
                )
                cache_list = adapter.make_batch_cache([0])
                for pos, src_idx in enumerate(attn_layer_indices):
                    cache_list[src_idx] = seeded_attn[pos]
            finally:
                prefix_cache.release(hit.block_ids)

            if is_hybrid_recurrent and prev_recurrent_snapshot is not None:
                assert isinstance(adapter, RecurrentStateAdapter)
                adapter.restore_recurrent_state(
                    cache_list, 0, prev_recurrent_snapshot
                )

        logits = forward_batched_full(model, chunk_tokens, cache_list)

        if prev_last_logit is not None:
            target = chunk_tokens[:, 0]
            total_nll += float(
                _cross_entropy_sum(prev_last_logit, target).item()
            )
            total_tokens += 1

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

        prev_last_logit = mx.contiguous(
            logits[:, -1, :].astype(mx.float32)
        )
        mx.eval(prev_last_logit)

        if is_hybrid_recurrent:
            assert isinstance(adapter, RecurrentStateAdapter)
            prev_recurrent_snapshot = adapter.snapshot_recurrent_state(
                cache_list, row_idx=0
            )

        aligned_end = (end // block_size) * block_size
        if aligned_end > 0:
            detached_to_insert = _extract_aligned_blocks_from_seeded_cache(
                cache_list, block_size, aligned_end
            )
            # F.0b encode seam: K extracted from the seeded cache is
            # post-RoPE (mlx-lm's attention rotated it before
            # cache.update_and_fetch). Inverse-rotate K to pre-RoPE
            # so the codec's noise-injection lives in the same
            # space as vqbench's ``_QuantizedProj``.
            detached_to_insert = _round_trip_inverse_rope_for_extract(
                detached_to_insert,
                block_size=block_size,
                rope_instances_per_attn_layer=(
                    rope_instances_per_attn_layer
                ),
            )
            prefix_tokens_to_insert = _mx_1d_to_int_list(
                token_ids[0, :aligned_end]
            )
            prefix_cache.insert_detached(
                prefix_tokens_to_insert, detached_to_insert
            )

    return total_nll, total_tokens


# =============================================================================
# P-5-F F.0b' — pre-norm store oracle (Option 3b).
#
# F.0b's "post-k_norm pre-RoPE" injection (above) verified empirically
# that codec noise injected after k_norm is ~2x larger in the
# post-k_norm space mlx-lm reads, leading to ~8x worse PPL through
# attention softmax non-linearity. (3b) routes the prefix store
# through ``adapter.install_pre_norm_capture`` —
# ``silica/models/pre_norm_capture.py`` installs a per-attention-layer
# proxy on ``self_attn.k_proj`` at adapter construction. The proxy
# captures ``k_proj(x)`` (pre-k_norm K) into a buffer the oracle
# supplies for each chunk forward, leaving the in-flight forward
# bit-identical to the unwrapped reference.
#
# On hit-path admission, K_pre blocks decode → ``adapter.apply_k_norm_then_rope``
# (k_norm + RoPE) → seeded cache. Codec noise lives in the same
# pre-k_norm space vqbench's ``_QuantizedProj`` injects in, so the
# same +0.51 PPL D.2a achieves is reachable under persistent block-
# grained encoding (F.0b' verified +0.015 PPL — even better; see
# ``plans/P5_F_OPENING.md`` §10.3).
#
# Difference vs (3a) / D.2a: the proxy does NOT modify the in-flight
# forward. ``_PreNormCaptureProxy.__call__`` returns ``k_proj(x)``
# unchanged, only side-effecting a capture for later encode. In-flight
# attention runs clean — the current chunk's logits use noise-free K.
# Codec noise affects only PRIOR chunks' K (the seeded cache
# contents), matching the production prefix-cache hit semantic.
# (3a) injects noise on the in-flight forward too; (3b) is strictly
# less noisy and matches what the F.2+ production store does.
# =============================================================================


def teacher_forced_chunked_nll_with_codec_pre_norm(
    adapter: Any,
    prefix_cache: Any,
    token_ids: mx.array,
    *,
    chunk_size: int = 256,
) -> tuple[float, int]:
    """P-5-F (3b) production-store oracle — pre-norm store via projection-output capture.

    Routes through ``adapter.install_pre_norm_capture(buffer)`` and
    ``adapter.apply_k_norm_then_rope(layer_pos, K_pre, offset=...)``
    (P-5-F F.1; see ``silica/models/pre_norm_capture.py``). The proxy
    on each attention layer's ``k_proj`` captures K_pre into the
    oracle-supplied buffer during chunk forward; the oracle reads it
    back, encodes K_pre per block into the prefix store, and reuses
    the adapter's reconstruction method on hit-path admit.

    Codec noise lives in pre-k_norm space, identical to vqbench's
    ``_QuantizedProj`` injection point. F.0b' verification reached
    +0.015 PPL on Qwen3-0.6B + BlockTQ b64 b4 — better than D.2a's
    +0.51 (see ``plans/P5_F_OPENING.md`` §10.3).

    Buffer disarm: the buffer pointer is reset to ``None`` in a
    try/finally so an exception during forward leaves the adapter
    in the disarmed (no-capture) state.
    """
    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must be 2-D (1, seq_len); got shape "
            f"{tuple(token_ids.shape)}"
        )
    B, seq_len = token_ids.shape
    if B != 1:
        raise ValueError(
            f"teacher_forced_chunked_nll_with_codec_pre_norm supports "
            f"B=1 only; got B={B}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    block_size = prefix_cache.block_size
    if chunk_size % block_size != 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be a positive multiple of "
            f"prefix_cache.block_size ({block_size})"
        )
    if seq_len == 0:
        return 0.0, 0

    model = adapter._model
    # Read the adapter's attn_layer_indices directly — set at adapter
    # __init__ alongside the proxy install (P-5-F F.1).
    attn_layer_indices: list[int] = list(adapter._attn_layer_indices)
    layout = adapter.kv_layout()
    n_kv_heads = layout.n_kv_heads
    head_dim = layout.head_dim

    capture_buffer: dict[int, mx.array] = {}

    try:
        return _run_pre_norm_oracle_inner(
            adapter=adapter,
            model=model,
            prefix_cache=prefix_cache,
            token_ids=token_ids,
            seq_len=seq_len,
            chunk_size=chunk_size,
            block_size=block_size,
            attn_layer_indices=attn_layer_indices,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            capture_buffer=capture_buffer,
        )
    finally:
        adapter.install_pre_norm_capture(None)


def _run_pre_norm_oracle_inner(
    *,
    adapter: Any,
    model: Any,
    prefix_cache: Any,
    token_ids: mx.array,
    seq_len: int,
    chunk_size: int,
    block_size: int,
    attn_layer_indices: list[int],
    n_kv_heads: int,
    head_dim: int,
    capture_buffer: dict[int, mx.array],
) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0
    prev_last_logit: mx.array | None = None

    is_hybrid_recurrent = isinstance(adapter, RecurrentStateAdapter)
    prev_recurrent_snapshot: RecurrentSnapshot | None = None

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_tokens = token_ids[:, start:end]
        chunk_len = end - start

        if start == 0:
            cache_list = adapter.make_batch_cache([0])
        else:
            prefix_token_list = _mx_1d_to_int_list(token_ids[0, :start])
            hit = prefix_cache.lookup(prefix_token_list)
            try:
                if hit.num_hit_tokens != start:
                    raise RuntimeError(
                        f"prefix_cache.lookup returned "
                        f"{hit.num_hit_tokens} hit tokens; expected "
                        f"full prefix hit of {start}"
                    )
                detached_blocks = list(
                    prefix_cache.fetch_detached_blocks(hit.block_ids)
                )
                # Convert pre-norm K + (still-pre-RoPE) V payloads
                # into the post-RoPE seeded cache mlx-lm expects.
                # K: adapter.apply_k_norm_then_rope per layer at the
                # block's absolute offset. V: pass through (V never
                # normalises or rotates in mlx-lm's supported attention
                # forwards).
                seeded_blocks: list[list[tuple[mx.array, mx.array]]] = []
                for block_idx, per_attn_layer in enumerate(detached_blocks):
                    block_post_rope: list[tuple[mx.array, mx.array]] = []
                    for attn_pos, (k_pre, v) in enumerate(per_attn_layer):
                        k_post = adapter.apply_k_norm_then_rope(
                            attn_pos,
                            k_pre,
                            offset=block_idx * block_size,
                        )
                        block_post_rope.append((k_post, v))
                    seeded_blocks.append(block_post_rope)

                seeded_attn = build_seeded_batch_kv(
                    seeded_blocks, num_layers=len(attn_layer_indices)
                )
                cache_list = adapter.make_batch_cache([0])
                for pos, src_idx in enumerate(attn_layer_indices):
                    cache_list[src_idx] = seeded_attn[pos]
            finally:
                prefix_cache.release(hit.block_ids)

            if is_hybrid_recurrent and prev_recurrent_snapshot is not None:
                assert isinstance(adapter, RecurrentStateAdapter)
                adapter.restore_recurrent_state(
                    cache_list, 0, prev_recurrent_snapshot
                )

        # Arm capture: clear any leftover entries and route the next
        # forward's k_proj outputs into ``capture_buffer``.
        capture_buffer.clear()
        adapter.install_pre_norm_capture(capture_buffer)

        logits = forward_batched_full(model, chunk_tokens, cache_list)

        # Disarm immediately; the next read on ``capture_buffer`` is
        # safe because the proxy has already populated entries during
        # the forward.
        adapter.install_pre_norm_capture(None)

        if prev_last_logit is not None:
            target = chunk_tokens[:, 0]
            total_nll += float(
                _cross_entropy_sum(prev_last_logit, target).item()
            )
            total_tokens += 1

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

        prev_last_logit = mx.contiguous(
            logits[:, -1, :].astype(mx.float32)
        )
        mx.eval(prev_last_logit)

        if is_hybrid_recurrent:
            assert isinstance(adapter, RecurrentStateAdapter)
            prev_recurrent_snapshot = adapter.snapshot_recurrent_state(
                cache_list, row_idx=0
            )

        # Encode K_pre + V per block. The wrapper captured this chunk's
        # full K_pre (output of k_proj on the chunk's hidden); split
        # into block_size slices and pair with the V slices extracted
        # from the seeded cache (V is unchanged in cache vs v_proj
        # output, so cache-extract is cheaper than wrapping v_proj
        # too).
        aligned_end = (end // block_size) * block_size
        if aligned_end > 0:
            # Validate the buffer received an entry for every attention
            # layer position (proxy fired during forward).
            for attn_pos in range(len(attn_layer_indices)):
                if attn_pos not in capture_buffer:
                    raise RuntimeError(
                        f"capture_buffer missing attn_pos={attn_pos} "
                        f"after chunk forward (start={start}, end={end}); "
                        f"adapter proxy for layer "
                        f"{attn_layer_indices[attn_pos]} did not fire."
                    )
            # On hit chunks (start > 0) the proxy captured only the
            # new chunk's K_pre — block_idx for new tokens runs from
            # (start // block_size) to (aligned_end // block_size).
            # Old blocks (0..start//block_size) are already in the
            # store from prior reclaims; insert_detached's
            # duplicate-prefix handling will skip them.
            chunk_first_new_block = start // block_size
            new_block_count = (aligned_end - start) // block_size
            new_blocks_kv: list[list[tuple[mx.array, mx.array]]] = []
            v_extracted = _extract_aligned_blocks_from_seeded_cache(
                cache_list, block_size, aligned_end
            )
            for new_block_offset in range(new_block_count):
                block_idx = chunk_first_new_block + new_block_offset
                # K_pre slice from this chunk's capture: tokens
                # [new_block_offset * block_size : (new_block_offset+1) * block_size]
                # within the chunk are absolute-position
                # [block_idx * block_size : (block_idx+1) * block_size].
                per_attn_kv: list[tuple[mx.array, mx.array]] = []
                cap_start = new_block_offset * block_size
                cap_end = cap_start + block_size
                for attn_pos in range(len(attn_layer_indices)):
                    capture = capture_buffer[attn_pos]
                    cap_per_head = capture.reshape(
                        1, chunk_len, n_kv_heads, head_dim
                    ).transpose(0, 2, 1, 3)
                    k_pre_block = mx.contiguous(
                        cap_per_head[:, :, cap_start:cap_end, :]
                    )
                    # V from the extracted-cache list at the same
                    # absolute block index.
                    _k_post_unused, v_block = v_extracted[block_idx][attn_pos]
                    per_attn_kv.append((k_pre_block, v_block))
                new_blocks_kv.append(per_attn_kv)

            # Pad with empty blocks for the prefix-tree positions
            # before the first new block; insert_detached's
            # duplicate-prefix handling skips these (the existing
            # tree nodes are reused), but we still need the outer
            # list to cover every aligned block in
            # `prefix_tokens_to_insert` per the contract.
            full_blocks: list[list[tuple[mx.array, mx.array]]] = []
            # Old blocks: provide a placeholder that won't be touched
            # (insert_detached on a duplicate-prefix branch ignores
            # the corresponding entry). Use the last new block's K/V
            # shape so the contract `len(per_layer) == num_layers`
            # holds even on the duplicate path.
            placeholder = (
                new_blocks_kv[0] if new_blocks_kv else []
            )
            for _ in range(chunk_first_new_block):
                full_blocks.append(placeholder)
            full_blocks.extend(new_blocks_kv)

            mx.eval(*[
                t for blk in new_blocks_kv for kv in blk for t in kv
            ])
            prefix_tokens_to_insert = _mx_1d_to_int_list(
                token_ids[0, :aligned_end]
            )
            prefix_cache.insert_detached(
                prefix_tokens_to_insert, full_blocks
            )

    return total_nll, total_tokens


# =============================================================================
# P-5-D.2a — vqbench-aligned PPL path (pre-RoPE projection patch).
# =============================================================================


CodecFactory = Callable[..., VectorCodec]
"""Signature-compatible with :data:`silica.bench.codec_registry.CodecFactory`.

Takes ``(block_size, n_kv_heads, head_dim, dtype, seed)`` by keyword and
returns a :class:`silica.kvcache.codec.VectorCodec`. The vqbench-aligned
oracle instantiates one codec per observed chunk length (typically the
fixed ``chunk_size``, with at most one shorter instance for the final
partial chunk) because :class:`silica.vq.BlockTurboQuantMSE` and its
siblings validate the time-axis ``block_size`` against the input's
third dimension. A Protocol-level ``VectorCodec`` that accepted a
dynamic time axis would sidestep this, but is out of scope for
P-5-D.2a — the oracle is not hot serving code, and a per-L codec
instance on a 512-token run costs at most two factory calls."""


class _WrappedProj:
    """Quantize→dequantize wrapper around ``k_proj`` / ``v_proj``.

    Installed in-place on an mlx-lm attention module by
    :func:`teacher_forced_chunked_nll_vqbench_aligned`. On each call,
    runs the original linear projection, reshapes the output into the
    per-head ``(B, nkv, L, head_dim)`` layout every codec in
    :mod:`silica.vq` consumes, pipes it through the codec's
    ``encode_tensor`` → ``decode_tensor``, then reshapes back to the
    flat ``(B, L, nkv*head_dim)`` layout mlx-lm's attention forward
    expects downstream (so RoPE + cache.update_and_fetch see the
    same shape they would have received without the wrapper).

    The codec is built per-L on first use and cached in ``_codecs``:
    a 512-token teacher-forced run chunked at 256 typically sees two
    Ls (256 and the final partial chunk, or just 256 if the sequence
    length is a clean multiple). A new codec instance is cheap
    (Haar rotation + Lloyd-Max codebook — both O(head_dim²) and
    seeded, no data dependency), and a per-call creation would trade
    a negligible allocation cost for consistency with
    :func:`teacher_forced_chunked_nll_with_codec`'s single-codec
    pattern. Caching here avoids even that allocation without
    changing behaviour — the codec state is deterministic in
    ``(seed, vq knobs, dtype, L)``.

    Not an ``mlx.nn.Module`` subclass: Module installation would
    re-trigger the parent's parameter registration on a call chain
    that already owns its params. The wrapper only needs to intercept
    ``__call__``, so a plain class sidesteps the Module lifecycle.
    Restoration is guaranteed by the installer's ``try/finally``.
    """

    __slots__ = (
        "_orig",
        "_n_kv_heads",
        "_head_dim",
        "_factory",
        "_seed",
        "_dtype",
        "_codecs",
    )

    def __init__(
        self,
        orig: Any,
        *,
        n_kv_heads: int,
        head_dim: int,
        factory: CodecFactory,
        seed: int,
        dtype: mx.Dtype,
    ) -> None:
        self._orig = orig
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._factory = factory
        self._seed = seed
        self._dtype = dtype
        self._codecs: dict[int, VectorCodec] = {}

    def _get_codec(self, length: int) -> VectorCodec:
        cached = self._codecs.get(length)
        if cached is not None:
            return cached
        codec = self._factory(
            block_size=length,
            n_kv_heads=self._n_kv_heads,
            head_dim=self._head_dim,
            dtype=self._dtype,
            seed=self._seed,
        )
        self._codecs[length] = codec
        return codec

    def __call__(self, x: mx.array) -> mx.array:
        out = self._orig(x)
        # mlx-lm Qwen3 attention applies the per-head reshape and
        # transpose right after ``self.k_proj(x)`` / ``self.v_proj(x)``.
        # Mirror the same reshape here so the codec sees the
        # ``(1, n_kv_heads, L, head_dim)`` layout it validates against,
        # then reverse the reshape so the caller sees the same flat
        # ``(B, L, n_kv_heads * head_dim)`` output the unwrapped
        # projection would have produced. The net effect is a pure
        # quantize→dequantize pass on the per-head K (or V) vectors
        # in the pre-RoPE projection space.
        B, L, _ = out.shape
        reshaped = out.reshape(B, L, self._n_kv_heads, self._head_dim).transpose(
            0, 2, 1, 3
        )
        codec = self._get_codec(L)
        decoded = codec.decode_tensor(codec.encode_tensor(reshaped))
        return decoded.transpose(0, 2, 1, 3).reshape(B, L, -1)


def _install_vqbench_wrappers(
    model: Any,
    *,
    n_kv_heads: int,
    head_dim: int,
    factory: CodecFactory,
    seed: int,
    dtype: mx.Dtype,
    wrap_v: bool,
) -> list[tuple[Any, Any, Any]]:
    """Replace ``attn.k_proj`` (and ``attn.v_proj`` when ``wrap_v``) on
    every layer of ``model`` with a :class:`_WrappedProj`.

    Returns a list of restoration tokens
    ``[(attn_module, orig_k_proj, orig_v_proj_or_None), ...]`` that
    :func:`_restore_vqbench_wrappers` consumes. The caller *must* call
    the restorer in a ``finally`` block — the mlx-lm module the
    adapter hands out is the same object the serving hot path uses,
    and leaving quantized projections installed across oracle calls
    would silently poison every subsequent forward.

    Rotation sharing: one codec per ``(projection_side, L)`` pair is
    built through ``factory(..., seed=seed)`` — the seed is the
    oracle's execution seed, shared across all layers and all heads
    within a layer. This mirrors silica's prefix-cache C.2 pattern
    (one codec instance per K/V side, not per layer) and deviates
    from vqbench's ``methods.common.monkey_patch._QuantizedProj``,
    which runs ``quant_dequant_tensor`` with a *per-head* seed.
    Probe 2 of the D.2 investigation
    (``plans/P5_D2_INVESTIGATION/p5_d2_probe2.py``) found the
    shared-vs-per-head rotation split gives a 0.977× Frobenius
    ratio on real Qwen3-0.6B post-RoPE K — below the measurement
    floor of the ΔPPL signal we are matching. Staying on silica's
    shared-rotation convention keeps the oracle's codec semantics
    identical to the production store path for the rotation axis,
    which is what lets a future port to per-head rotation (if ever
    warranted) be isolated to one code path rather than two.
    """
    restorations: list[tuple[Any, Any, Any]] = []
    for layer in model.layers:
        attn = layer.self_attn
        orig_k = attn.k_proj
        orig_v = attn.v_proj if wrap_v else None
        attn.k_proj = _WrappedProj(
            orig_k,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            factory=factory,
            seed=seed,
            dtype=dtype,
        )
        if wrap_v:
            attn.v_proj = _WrappedProj(
                orig_v,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                factory=factory,
                seed=seed,
                dtype=dtype,
            )
        restorations.append((attn, orig_k, orig_v))
    return restorations


def _restore_vqbench_wrappers(
    restorations: list[tuple[Any, Any, Any]],
) -> None:
    """Undo the in-place replacement performed by
    :func:`_install_vqbench_wrappers`."""
    for attn, orig_k, orig_v in restorations:
        attn.k_proj = orig_k
        if orig_v is not None:
            attn.v_proj = orig_v


def teacher_forced_chunked_nll_vqbench_aligned(
    adapter: Any,
    token_ids: mx.array,
    *,
    chunk_size: int = 256,
    codec_factory: CodecFactory,
    seed: int = 42,
    wrap_v: bool = True,
) -> tuple[float, int]:
    """Teacher-forced NLL with vqbench-style pre-RoPE codec injection.

    Wraps each attention layer's ``k_proj`` (and, when ``wrap_v``,
    ``v_proj``) in a quantize→dequantize pass, then delegates the
    chunk-boundary scoring / cache management to
    :func:`teacher_forced_chunked_nll`. The wrapper operates in
    pre-RoPE space, so the cache stores the lossy projection output
    *after* RoPE has been applied to it — identical to what
    vqbench's ``methods.common.monkey_patch._QuantizedProj`` does.
    Projections are restored in a ``try/finally`` so an exception
    during the forward still leaves the adapter's model in its
    original state.

    Not a re-implementation of the C.1 chunk loop — this function
    *wraps* the C.1 oracle call so the two paths stay bit-identical
    on their shared scoring logic. The only difference is whether a
    codec is installed on the projections during the delegated call.
    Under an identity codec (or a lossless round-trip) this function
    and :func:`teacher_forced_chunked_nll` return the same result to
    fp tolerance; under a lossy codec the returned NLL carries the
    accumulated error of all K and V (current + past) being
    projected through the codec on every chunk.

    Args:
        adapter: same contract as :func:`teacher_forced_chunked_nll`.
            Additionally must expose ``kv_layout()`` (for
            ``n_kv_heads`` / ``head_dim`` / ``dtype``) and
            ``_model.layers`` with ``.self_attn.{k_proj, v_proj}``
            per layer — mlx-lm's standard attention convention.
        token_ids: ``(1, seq_len)`` ``mx.array`` of token ids.
        chunk_size: forwarded to :func:`teacher_forced_chunked_nll`.
        codec_factory: callable returning a
            :class:`silica.kvcache.codec.VectorCodec`, signature
            compatible with
            :data:`silica.bench.codec_registry.CodecFactory`. One
            codec is built per observed chunk length (time-axis
            block_size), cached inside each wrapper for the duration
            of the call. An IdentityCodec factory is a valid no-op
            and used in the parity test.
        seed: codec seed (Haar rotation). Forwarded into every
            ``codec_factory`` invocation.
        wrap_v: whether to wrap ``v_proj`` in addition to ``k_proj``.
            Default ``True`` matches vqbench's ``--patch-v`` flag as
            used in the REPORT §3.1 BlockTQ production recommendation.

    Returns:
        ``(nll_sum, n_tokens_scored)`` — same shape as the other two
        entry points.
    """
    layout = adapter.kv_layout()
    restorations = _install_vqbench_wrappers(
        adapter._model,
        n_kv_heads=layout.n_kv_heads,
        head_dim=layout.head_dim,
        factory=codec_factory,
        seed=seed,
        dtype=layout.dtype,
        wrap_v=wrap_v,
    )
    try:
        return teacher_forced_chunked_nll(
            adapter, token_ids, chunk_size=chunk_size
        )
    finally:
        _restore_vqbench_wrappers(restorations)


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
