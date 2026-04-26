"""P-3-C3c hybrid batched smoke — Qwen/Qwen3.5-0.8B, no prefix cache.

After C3c lifted the ContinuousBatcher gate for ``HYBRID_DELTANET``,
this file exercises the real-model hybrid path to prove the wiring
(C3a adapter factory + C3b mid-run factory + prefix-cache guard)
actually produces a working batched generation loop on a real
DeltaNet + GQA hybrid checkpoint. Two tests:

  * Public API smoke via ``Engine.generate_batch`` — proves the
    end-to-end stream (tokens + done events, no aborts) on a small
    batch with ``prefix_cache=None``.
  * Direct ``ContinuousBatcher`` smoke — drives ``step()`` manually
    and asserts the live ``_batch_cache`` is genuinely heterogeneous
    (``ArraysCache`` at DeltaNet layer indices, ``BatchKVCache`` at
    global-attention layer indices). This is the load-bearing
    assertion that ``adapter.make_batch_cache`` actually reached the
    scheduler rather than falling back to an all-``BatchKVCache`` list.

Token-level parity (vs ``Engine.generate`` single-request) is NOT
asserted here — smoke only guarantees "does not crash, emits
tokens, live cache is genuinely hybrid". Bit-parity / golden-token
verification is M-4 / future C3d territory.

Skipped when Qwen3.5-0.8B is not in the local HF cache — run
``scripts/acceptance_p1_mlx_lm_parity.py`` or
``scripts/probe_qwen3_5_load.py`` once to populate, or set
``SILICA_SKIP_MODEL_TESTS=1`` to skip explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from silica import Engine
from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.adapter import AttentionKind
from silica.models.factory import adapter_for_repo
from silica.scheduler.batcher import ContinuousBatcher

REPO = "Qwen/Qwen3.5-0.8B"

_HF_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3.5-0.8B"
)
_SKIP_REASON = (
    "Qwen3.5-0.8B not cached at "
    f"{_HF_CACHE}; run scripts/probe_qwen3_5_load.py or "
    "scripts/acceptance_p1_mlx_lm_parity.py to populate it."
)
_SKIP = not _HF_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_engine_generate_batch_runs_on_qwen3_5_0_8b() -> None:
    """Public API smoke: ``Engine.generate_batch`` against a
    HYBRID_DELTANET adapter with ``prefix_cache=None`` (the C3b-
    guard-compliant path) yields token + done events for every
    request and produces no aborted events."""
    adapter, kv = adapter_for_repo(REPO)
    engine = Engine(adapter, kv)
    tokenizer = adapter.tokenizer()
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=4,
        stop_token_ids=eos_ids,
    )

    prompts = ["Hello", "The capital of France"]
    # BatchEvent fields ``token_id`` and ``finish_reason`` are typed
    # Optional because the fields are unused outside their own event
    # kinds; narrow locally via assertions at the call site.
    tokens_by_req: dict[int, list[int]] = {}
    done_by_req: dict[int, str] = {}
    aborts: list[tuple[int, str]] = []
    for event in engine.generate_batch(
        prompts, params, max_batch_size=2, prefix_cache=None
    ):
        if event.kind == "token":
            assert event.token_id is not None
            tokens_by_req.setdefault(event.req_index, []).append(event.token_id)
        elif event.kind == "done":
            assert event.finish_reason is not None
            done_by_req[event.req_index] = event.finish_reason
        elif event.kind == "aborted":
            assert event.finish_reason is not None
            aborts.append((event.req_index, event.finish_reason))

    assert aborts == [], f"unexpected abort events: {aborts}"
    assert set(tokens_by_req) == {0, 1}
    assert set(done_by_req) == {0, 1}
    for req_index in (0, 1):
        assert len(tokens_by_req[req_index]) >= 1


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_batcher_batch_cache_is_genuinely_hybrid_on_qwen3_5_0_8b() -> None:
    """Direct batcher smoke: after ``_prepare_cohort`` runs on a real
    Qwen3.5-0.8B adapter, the live ``_batch_cache`` interleaves
    ``ArraysCache`` (DeltaNet layers) with ``BatchKVCache`` (global
    attention layers). This proves ``Qwen3_5Adapter.make_batch_cache``
    actually reached the scheduler rather than the ``callable()``
    fallback producing an all-``BatchKVCache`` list.

    The assertion walks ``adapter.attention_pattern().per_layer`` and
    ``_batch_cache`` in lockstep, pinning the exact per-layer mapping:
    ``HYBRID_DELTANET → ArraysCache``, ``GLOBAL → BatchKVCache``. Any
    future ``AttentionKind`` reaching this path must fail loudly here
    rather than silently degrading to the wrong cache type.
    """
    adapter, _ = adapter_for_repo(REPO)
    batcher = ContinuousBatcher(
        adapter, max_batch_size=2, prefix_cache=None
    )
    tokenizer = adapter.tokenizer()
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))
    params = SamplingParams(
        temperature=0.0, max_tokens=2, stop_token_ids=eos_ids
    )
    prompt_a = list(tokenizer.encode("Hello"))
    prompt_b = list(tokenizer.encode("The capital of France"))
    assert prompt_a and prompt_b
    batcher.add_request(0, prompt_a, params)
    batcher.add_request(1, prompt_b, params)

    # One step is enough to seal the cohort + run batched prefill +
    # emit first token; the assertions below inspect state after
    # ``_prepare_cohort`` has populated ``_batch_cache``.
    list(batcher.step())

    cache = batcher._batch_cache  # type: ignore[attr-defined]
    assert cache is not None
    assert len(cache) == adapter.config.num_layers

    pattern = adapter.attention_pattern().per_layer
    assert len(pattern) == len(cache)
    for layer_idx, (kind, layer_cache) in enumerate(
        zip(pattern, cache, strict=True)
    ):
        if kind == AttentionKind.HYBRID_DELTANET:
            assert isinstance(layer_cache, ArraysCache), (
                f"layer {layer_idx}: kind={kind.value} expected "
                f"ArraysCache, got {type(layer_cache).__name__}"
            )
        elif kind == AttentionKind.GLOBAL:
            assert isinstance(layer_cache, BatchKVCache), (
                f"layer {layer_idx}: kind={kind.value} expected "
                f"BatchKVCache, got {type(layer_cache).__name__}"
            )
        else:
            raise AssertionError(
                f"layer {layer_idx}: unexpected AttentionKind "
                f"{kind.value!r} reached the hybrid cache-alignment "
                f"smoke — either the capability gate grew a new "
                f"supported kind without a matching case here, or "
                f"Qwen3_5Adapter.make_batch_cache is producing caches "
                f"for layers it should not own."
            )


# --- P-3-C5.3.4 sibling: prefix_cache enabled, multi-token byte-exact ---

_PREFIX_CACHE_BLOCK_SIZE = 16
# Long enough to tokenize to >= 33 tokens (2 * BLOCK_SIZE + 1) so the
# subject's request 1 produces a non-trivial seeded tree (>= 2 blocks)
# and request 2's hit-admission has a non-empty suffix.
_C5_3_4_PROMPT_TEXT = (
    "The recurrent associative memory accumulates across every "
    "processed token in this sequence. Each forward pass advances "
    "the cache state without resetting until the request completes. "
    "Slice-prefill regime captures snapshots at fixed block "
    "boundaries to support prefix-cache cooperation under hybrid "
    "DeltaNet attention."
)


def _drain_until_req_done(
    batcher: ContinuousBatcher, req_index: int
) -> list[object]:
    """Drive ``step()`` calls until the given req_index emits a
    ``"done"`` event. Collects events from every step.

    Used to walk the multi-step decode sequence from admission
    through ``max_tokens`` reach. The reclaim step (where the
    DONE row is removed and prefix is extracted) typically emits
    no events, so the loop exits before reclaim runs; callers
    that need reclaim done can call ``step()`` once more.
    """
    collected: list[object] = []
    for _ in range(64):  # generous upper bound; max_tokens stays small
        events = batcher.step()
        collected.extend(events)
        for ev in events:
            if (
                ev.kind == "done"  # type: ignore[attr-defined]
                and ev.req_index == req_index  # type: ignore[attr-defined]
            ):
                return collected
    raise AssertionError(
        f"req_index={req_index} did not finish within 64 steps; "
        f"last events: {collected[-5:]}"
    )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_batcher_prefix_cache_hit_admission_multi_token_byte_exact_on_qwen3_5_0_8b() -> (
    None
):
    """C5.3.4 acceptance — multi-token byte-exact gate.

    Drives Qwen3.5-0.8B with two sequentially-admitted requests
    sharing a long prompt. Request 1 seeds the radix tree
    (prefill + decode + reclaim's extract); request 2 admits
    mid-run via Phase-B + ``_admit_single_hit_row``. The hit row
    runs ``max_tokens`` decode steps; the full token stream is
    asserted byte-equal to a slice-regime oracle running the
    same prompt + ``max_tokens`` from scratch under an empty
    ``RadixPrefixCache`` (Phase-B walks the empty tree → miss →
    slice-regime full-prompt prefill, same trajectory regime as
    the subject's hit-path restore + suffix prefill).

    Multi-token verification extends the C5.3.3b first-token gate
    (tests/test_batcher_hit_admission_byte_exact_oracle.py) to
    ensure post-restore decode stays byte-exact across multiple
    decode steps. Token equality across the full stream is
    implied by first-token equality + cache byte-equality at
    admission, plus deterministic forward; this test pins that
    chain in practice on a real model.
    """
    adapter, _ = adapter_for_repo(REPO)
    tokenizer = adapter.tokenizer()
    prompt_ids = list(tokenizer.encode(_C5_3_4_PROMPT_TEXT))
    assert len(prompt_ids) >= 2 * _PREFIX_CACHE_BLOCK_SIZE + 1, (
        f"_C5_3_4_PROMPT_TEXT tokenized to {len(prompt_ids)}; need "
        f">= {2 * _PREFIX_CACHE_BLOCK_SIZE + 1}"
    )
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))

    max_tokens_request_2 = 4

    # === Subject: prefix_cache + opt-in flag ===
    pc = RadixPrefixCache(
        block_size=_PREFIX_CACHE_BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(
            block_size=_PREFIX_CACHE_BLOCK_SIZE
        ),
    )
    subject = ContinuousBatcher(
        adapter,
        prefix_cache=pc,
    )
    assert subject._slice_prefill_active() is True
    assert (
        subject._effective_slice_block_size()
        == _PREFIX_CACHE_BLOCK_SIZE
    )

    # Request 1: short max_tokens just to drive the row to DONE so
    # reclaim's extract seeds the radix tree.
    subject.add_request(
        0,
        prompt_ids,
        SamplingParams(
            temperature=0.0,
            max_tokens=2,
            stop_token_ids=eos_ids,
        ),
    )
    _drain_until_req_done(subject, 0)
    subject.step()  # reclaim → extract → tree seeded
    assert pc.node_count() >= 2, (
        f"request 1 reclaim should have seeded >= 2 radix nodes; "
        f"got {pc.node_count()}"
    )

    # Request 2: admits mid-run via Phase-B + hit; runs
    # ``max_tokens_request_2`` decode steps total.
    subject.add_request(
        1,
        prompt_ids,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens_request_2,
            stop_token_ids=eos_ids,
        ),
    )
    subject_events = _drain_until_req_done(subject, 1)
    assert subject.prefix_hits == 1, (
        f"request 2 must hit the radix tree; "
        f"prefix_hits={subject.prefix_hits}"
    )

    subject_tokens = [
        ev.token_id  # type: ignore[attr-defined]
        for ev in subject_events
        if ev.kind == "token"  # type: ignore[attr-defined]
        and ev.req_index == 1  # type: ignore[attr-defined]
    ]
    assert len(subject_tokens) == max_tokens_request_2, (
        f"subject emitted {len(subject_tokens)} tokens for req 1; "
        f"expected {max_tokens_request_2}"
    )

    # === Oracle: prefix_cache=None + slice-regime force flag ===
    # P-3-C5.4: oracle uses an empty RadixPrefixCache for slice-regime
    # parity with the subject. Phase-B walks the empty tree → miss →
    # slice-prefill the full prompt; same trajectory regime as the
    # subject's hit-path restore + suffix prefill, only without any
    # tree state to consume.
    oracle_pc = RadixPrefixCache(
        block_size=_PREFIX_CACHE_BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(
            block_size=_PREFIX_CACHE_BLOCK_SIZE
        ),
    )
    oracle = ContinuousBatcher(adapter, prefix_cache=oracle_pc)
    assert oracle._slice_prefill_active() is True
    assert (
        oracle._effective_slice_block_size()
        == _PREFIX_CACHE_BLOCK_SIZE
    )

    oracle.add_request(
        1,
        prompt_ids,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens_request_2,
            stop_token_ids=eos_ids,
        ),
    )
    oracle_events = _drain_until_req_done(oracle, 1)
    # Oracle's prefix_cache is empty (no prior request seeded it),
    # so the single request runs through the initial cohort path
    # (prefill + decode) without ever hitting the Phase-B
    # classifier. prefix_hits stays 0.
    assert oracle.prefix_hits == 0

    oracle_tokens = [
        ev.token_id  # type: ignore[attr-defined]
        for ev in oracle_events
        if ev.kind == "token"  # type: ignore[attr-defined]
        and ev.req_index == 1  # type: ignore[attr-defined]
    ]
    assert len(oracle_tokens) == max_tokens_request_2

    # === Byte-exact token stream ===
    assert subject_tokens == oracle_tokens, (
        f"multi-token byte-exact mismatch:\n"
        f"  subject: {subject_tokens}\n"
        f"  oracle:  {oracle_tokens}\n"
        f"slice-regime greedy decode should produce identical "
        f"streams under byte-exact recurrent state at admission"
    )


# --- P-3-C5.4: production path through Engine.generate_batch ---


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_engine_generate_batch_with_prefix_cache_on_qwen3_5_0_8b() -> None:
    """C5.4 production-path smoke: ``Engine.generate_batch`` runs
    Qwen3.5-0.8B with a shared ``RadixPrefixCache`` end-to-end.

    Pre-C5.4 the ``ContinuousBatcher`` ctor rejected this combination
    (hybrid + prefix_cache) with NotImplementedError; the C5.3 test
    files relied on a private ``_allow_recurrent_prefix_cache_for_c5_3_testing``
    flag to bypass the guard. C5.4 removed the guard and the flag,
    so ``Engine.generate_batch`` — which has no awareness of the flag —
    can now drive the path.

    Two sequential ``generate_batch`` calls share the same
    ``RadixPrefixCache``: the first call admits cold and seeds the
    radix tree on reclaim; the second call admits the same prompt
    and hits via Phase-B + the slice-regime restore path. The smoke
    asserts the public API yields token + done events with no aborts
    on both calls, and that the second call observes a tree
    populated by the first (``pc.node_count() > 0`` after the first
    call drains).
    """
    adapter, kv = adapter_for_repo(REPO)
    engine = Engine(adapter, kv)
    tokenizer = adapter.tokenizer()
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))

    pc = RadixPrefixCache(
        block_size=_PREFIX_CACHE_BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(
            block_size=_PREFIX_CACHE_BLOCK_SIZE
        ),
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        stop_token_ids=eos_ids,
    )
    prompts = [_C5_3_4_PROMPT_TEXT]

    def _drain(call_idx: int) -> tuple[int, list[str], list[tuple[int, str]]]:
        tokens = 0
        dones: list[str] = []
        aborts: list[tuple[int, str]] = []
        for event in engine.generate_batch(
            prompts,
            params,
            max_batch_size=1,
            prefix_cache=pc,
        ):
            if event.kind == "token":
                assert event.token_id is not None, (
                    f"call {call_idx}: token event missing token_id"
                )
                tokens += 1
            elif event.kind == "done":
                assert event.finish_reason is not None
                dones.append(event.finish_reason)
            elif event.kind == "aborted":
                assert event.finish_reason is not None
                aborts.append((event.req_index, event.finish_reason))
        return tokens, dones, aborts

    # Cold call seeds the radix tree.
    cold_tokens, cold_dones, cold_aborts = _drain(0)
    assert cold_aborts == [], (
        f"cold call produced abort events: {cold_aborts}"
    )
    assert cold_tokens >= 1
    assert len(cold_dones) == 1
    # Reclaim happened during the cold call's drain (the row reaches
    # DONE during decode, the next step's reclaim phase fires the
    # extract). Tree should now carry at least one block-aligned
    # node from the prompt prefix.
    assert pc.node_count() >= 1, (
        f"cold call should have seeded the radix tree; "
        f"node_count={pc.node_count()}"
    )

    # Warm call observes the tree populated by the cold call.
    warm_tokens, warm_dones, warm_aborts = _drain(1)
    assert warm_aborts == [], (
        f"warm call produced abort events: {warm_aborts}"
    )
    assert warm_tokens >= 1
    assert len(warm_dones) == 1
    # The same-prompt warm call admits via the hit path; the tree
    # must remain populated (release fires after admission, but
    # eviction only happens under memory pressure).
    assert pc.node_count() >= 1
