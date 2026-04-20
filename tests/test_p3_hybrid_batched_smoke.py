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
