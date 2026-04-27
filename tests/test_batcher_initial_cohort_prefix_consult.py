"""P5.9 step 2(b) — Q-012 affirmative resolution.

Initial-cohort prefix-cache consultation in ``ContinuousBatcher._prepare_cohort``.

Q-012's pre-fix behaviour (PLAN.md §10 Q-012, surfaced 2026-04-21
during P-4.5-C.1 test authoring): ``_prepare_cohort`` does not
consult ``self._prefix_cache``; the prefix-hit lookup path
(``peek`` → ``_admit_single_hit_row``) only fires inside
``_admit_waiting_requests``. Cross-call prefix reuse via repeated
``Engine.generate_batch([prompt], prefix_cache=shared_pc, ...)``
is therefore effectively zero — the chat REPL pays a full prefill
on every turn even when the prior turn's prefix is fully cached
in ``shared_pc``.

Post-fix (D-021 step 2(b), v1.7.14): ``_prepare_cohort``
classifies initial-cohort rows the same way
``_admit_waiting_requests`` does for mid-run admissions —
``peek`` → max-aligned check → optional recurrent-snapshot
guard → hit rows route through ``_admit_single_hit_row``,
miss rows through ``_admit_miss_cohort``.

This file's tests demonstrate the gap pre-fix and pin the
post-fix contract:

1. ``test_initial_cohort_full_hit_routes_to_seeded_admission``:
   single-row initial cohort whose prompt fully matches a
   pre-seeded block-aligned prefix; ``batcher.prefix_hits >= 1``
   after the first ``step()``.
2. ``test_initial_cohort_no_hit_still_runs_miss_path``: regression
   guard — when the prefix cache is empty, ``_prepare_cohort``
   continues to behave like the pre-fix miss-path: cohort sealed,
   batched prefill runs, no spurious ``prefix_hits`` increment.
3. ``test_initial_cohort_no_prefix_cache_unchanged``: regression
   guard — when ``prefix_cache=None``, ``_prepare_cohort``
   behaviour is byte-identical to pre-fix (cohort sealed, rows
   transitioned PREFILL, batch_cache built, ``_prefill_phase``
   runs the forward).

Uses ``_ScriptedAdapter`` from ``tests/test_batcher.py`` for fast
test cycles — no real-model dependency.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.scheduler.batcher import ContinuousBatcher
from tests.test_batcher import _ScriptedAdapter, _ScriptedModel

BLOCK_SIZE = 4


def _per_layer_kv_block(
    n_layers: int, marker: float
) -> list[tuple[mx.array, mx.array]]:
    """One (K, V) tuple per attention-layer position for one block.

    Matches the shape ``build_seeded_batch_kv`` consumes:
    ``(1, n_kv_heads, block_size, head_dim)`` per layer.
    """
    shape = (1, _ScriptedModel.N_KV, BLOCK_SIZE, _ScriptedModel.HEAD_DIM)
    return [
        (
            mx.full(shape, marker, dtype=mx.float16),
            mx.full(shape, marker + 0.5, dtype=mx.float16),
        )
        for _ in range(n_layers)
    ]


def _seed_two_blocks(
    pc: RadixPrefixCache, prompt: list[int], n_layers: int
) -> None:
    """Pre-seed ``pc`` with two block-aligned blocks worth of K/V for
    the first ``2 * BLOCK_SIZE`` tokens of ``prompt``. All-GLOBAL
    adapters consume one (K, V) per transformer layer per block;
    ``recurrent_snapshots=[None, None]`` is benign for all-GLOBAL
    patterns because the Phase-B classifier's snapshot guard only
    fires for ``RecurrentStateAdapter`` instances.
    """
    detached = [
        _per_layer_kv_block(n_layers, marker=100.0),
        _per_layer_kv_block(n_layers, marker=200.0),
    ]
    pc.insert_detached(
        prompt[: 2 * BLOCK_SIZE],
        detached,
        recurrent_snapshots=[None, None],
    )


def test_initial_cohort_full_hit_routes_to_seeded_admission() -> None:
    """Q-012 / D-021 step 2(b) primary contract.

    Pre-seed the prefix cache with two block-aligned blocks
    matching the request's first 8 tokens. Submit a 9-token
    request via ``add_request`` BEFORE the first ``step()`` so it
    enters the initial cohort path (not the waiting queue).
    After one ``step()`` the prefix-cache hit counter must be
    non-zero — proving ``_prepare_cohort`` consulted the cache
    rather than running miss-path prefill unconditionally.
    """
    n_layers = 2
    # Script length 1: ``_admit_single_hit_row`` runs a single
    # B=1 suffix-prefill forward and samples one token; that
    # consumes one entry from the scripted token queue.
    adapter = _ScriptedAdapter(n_layers=n_layers, script=[7])
    pc = RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=BLOCK_SIZE),
    )
    # 9-token prompt: 2 full blocks (cached) + 1-token suffix.
    prompt = list(range(1, 10))
    _seed_two_blocks(pc, prompt, n_layers=n_layers)

    batcher = ContinuousBatcher(adapter, prefix_cache=pc)
    batcher.add_request(
        0, prompt, SamplingParams(temperature=0.0, max_tokens=1)
    )
    events = batcher.step()

    # Primary contract: the cohort consulted the cache.
    assert batcher.prefix_hits >= 1, (
        f"Q-012: _prepare_cohort failed to consult prefix_cache on "
        f"a full-hit prompt; prefix_hits={batcher.prefix_hits}, "
        f"events={events}"
    )
    # Secondary contract: only the suffix was prefilled. The 8
    # cached tokens must NOT be re-forwarded.
    assert batcher.forward_prompt_tokens == 1, (
        f"Q-012: prefix-hit rows must forward only the suffix; "
        f"forward_prompt_tokens={batcher.forward_prompt_tokens}"
    )
    # Tertiary: a token event was emitted (the suffix-prefill
    # produced a sample).
    token_events = [e for e in events if e.kind == "token"]
    assert len(token_events) == 1, (
        f"expected 1 token event from the seeded admission; "
        f"got {len(token_events)} events={events}"
    )


def test_initial_cohort_no_hit_still_runs_miss_path() -> None:
    """Regression guard: empty prefix cache ⇒ pre-fix miss-path
    behaviour.

    With ``prefix_cache`` installed but no nodes seeded, the
    cohort must still admit normally (cohort sealed, batched
    prefill runs, first token sampled). ``prefix_hits`` stays
    at 0 — the cohort consulted the cache and saw nothing.
    """
    n_layers = 2
    adapter = _ScriptedAdapter(n_layers=n_layers, script=[5])
    pc = RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=BLOCK_SIZE),
    )
    batcher = ContinuousBatcher(adapter, prefix_cache=pc)
    prompt = list(range(1, 10))
    batcher.add_request(
        0, prompt, SamplingParams(temperature=0.0, max_tokens=1)
    )
    events = batcher.step()

    assert batcher.prefix_hits == 0
    # Miss-cohort path: the whole prompt was forwarded.
    assert batcher.forward_prompt_tokens == len(prompt)
    token_events = [e for e in events if e.kind == "token"]
    assert len(token_events) == 1
    assert token_events[0].token_id == 5


def test_initial_cohort_no_prefix_cache_unchanged() -> None:
    """Regression guard: ``prefix_cache=None`` path is bit-identical
    to pre-fix behaviour.

    Without a prefix cache, ``_prepare_cohort`` must continue to
    seal the cohort and let ``_prefill_phase`` run the batched
    forward in the same ``step()`` call. The classification
    branch added by D-021 step 2(b) is gated on
    ``self._prefix_cache is not None``.
    """
    n_layers = 2
    adapter = _ScriptedAdapter(n_layers=n_layers, script=[3])
    batcher = ContinuousBatcher(adapter)
    prompt = list(range(1, 10))
    batcher.add_request(
        0, prompt, SamplingParams(temperature=0.0, max_tokens=1)
    )
    events = batcher.step()

    # No prefix cache; the prefix_hits attribute may exist but
    # must be zero, and a token must have been emitted.
    assert batcher.prefix_hits == 0
    token_events = [e for e in events if e.kind == "token"]
    assert len(token_events) == 1
    assert token_events[0].token_id == 3


def test_mixed_initial_cohort_hits_and_misses() -> None:
    """Q-012 generalisation: B>1 cohort with one hit + one miss
    routes the hit row through seeded admission and the miss row
    through ``_admit_miss_cohort``.

    Prompt 0 fully matches the seeded prefix; prompt 1 shares no
    tokens with the seeded prefix (different leading id, so no
    radix hit). After one ``step()``: ``prefix_hits == 1`` and
    both rows have emitted their first token.
    """
    n_layers = 2
    # Script length 2: hit row's seeded admission consumes one
    # forward (suffix prefill on prompt 0); miss cohort consumes
    # a second forward (B=1 prefill on prompt 1). Both sample
    # one token. Token order in the script doesn't matter for
    # this test — only the count and event shape.
    adapter = _ScriptedAdapter(n_layers=n_layers, script=[7, 11])
    pc = RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=BLOCK_SIZE),
    )
    prompt_hit = list(range(1, 10))  # 9 tokens, 8 cached + 1 suffix
    prompt_miss = list(range(20, 29))  # 9 tokens, no cached prefix
    _seed_two_blocks(pc, prompt_hit, n_layers=n_layers)

    batcher = ContinuousBatcher(adapter, prefix_cache=pc, max_batch_size=2)
    batcher.add_request(
        0, prompt_hit, SamplingParams(temperature=0.0, max_tokens=1)
    )
    batcher.add_request(
        1, prompt_miss, SamplingParams(temperature=0.0, max_tokens=1)
    )
    events = batcher.step()

    assert batcher.prefix_hits == 1, batcher.prefix_hits
    # Forward bytes: 1 suffix from hit row + 9 prompt from miss row.
    assert batcher.forward_prompt_tokens == 1 + 9
    token_events = [e for e in events if e.kind == "token"]
    # Exactly two token events, one per row.
    assert len(token_events) == 2
    req_indices = sorted(e.req_index for e in token_events)
    assert req_indices == [0, 1]


@pytest.mark.parametrize("max_batch_size", [1, 2])
def test_two_consecutive_generate_batch_calls_share_prefix(
    max_batch_size: int,
) -> None:
    """Q-012 motivating user case: two ``Engine.generate_batch``-
    style invocations against the same ``RadixPrefixCache``. Pre-fix
    the second call's row would run miss-path prefill regardless of
    what the first call inserted; post-fix the second call sees the
    prefix.

    This drives ``ContinuousBatcher`` directly (no ``Engine``) to
    keep the test fast and free of mlx-lm load. The producer-side
    extraction path is covered elsewhere; this test pre-seeds the
    shared ``RadixPrefixCache`` as if call 1 had already reclaimed
    and inserted, then asserts call 2 hits on its initial cohort.
    """
    n_layers = 2
    pc = RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=BLOCK_SIZE),
    )
    prompt = list(range(1, 10))  # 9 tokens
    # Pre-seed the cache as if call 1 had already extracted its
    # prefix. The unit test for `_extract_and_insert_prefix` is
    # elsewhere; here we drive the consumption side directly.
    _seed_two_blocks(pc, prompt, n_layers=n_layers)

    # "Call 2" — fresh batcher, fresh cohort, same prefix cache.
    adapter2 = _ScriptedAdapter(n_layers=n_layers, script=[42])
    batcher2 = ContinuousBatcher(
        adapter2, prefix_cache=pc, max_batch_size=max_batch_size
    )
    batcher2.add_request(
        0, prompt, SamplingParams(temperature=0.0, max_tokens=1)
    )
    events2 = batcher2.step()

    assert batcher2.prefix_hits >= 1, (
        f"Q-012: second generate_batch-style invocation failed to "
        f"hit shared prefix cache on its initial cohort; "
        f"prefix_hits={batcher2.prefix_hits}"
    )
    # Suffix-only forward on call 2.
    assert batcher2.forward_prompt_tokens == 1
    token_events = [e for e in events2 if e.kind == "token"]
    assert len(token_events) == 1
