"""P-3-D3 Gemma4-31B-4bit batched smoke on the real checkpoint.

After P-3-D2 landed ``Gemma4Adapter.make_batch_cache`` and P-3-D3
lifted the ``ContinuousBatcher`` capability gate for
``AttentionKind.SLIDING``, this file exercises the sliding-window
batched path end-to-end on the real Gemma4-31B-4bit checkpoint. Two
tests:

  * Public API smoke via ``Engine.generate_batch`` — proves the
    stream (tokens + done events, no aborts) on a small batch with
    ``prefix_cache=None``. D3 only commits to the miss-only path;
    the constructor guard rejects ``prefix_cache != None`` when
    ``AttentionKind.SLIDING`` is present.
  * Direct ``ContinuousBatcher`` smoke — drives ``step()`` manually
    and asserts the live ``_batch_cache`` is genuinely heterogeneous
    (``BatchRotatingKVCache`` at sliding layer indices,
    ``BatchKVCache`` at full-attention layer indices). This proves
    ``Gemma4Adapter.make_batch_cache`` actually reached the scheduler
    rather than falling back to an all-``BatchKVCache`` list via the
    ``callable()`` branch.

Token-level parity (vs ``Engine.generate`` single-request) is NOT
asserted here — that is P-3-D3.1 territory, structurally mirroring
C3d on the Qwen3.5 hybrid side.

**Dual gate**: real weights cached AND
``SILICA_REAL_GEMMA4_31B=1`` must be set. Gemma4-31B batched is
more expensive than the D1.1 single-request smoke (~18 GB on disk,
~30+ GB device-memory peak, multi-token greedy decode for two
rows); the extra env-var gate avoids unexpectedly long local runs
when the weights are cached but the developer is on an unrelated
task. Single-request D1.1 uses the cache-only gate because cost
there is one short forward pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.adapter import AttentionKind
from silica.models.factory import adapter_for_repo
from silica.scheduler.batcher import ContinuousBatcher

REPO = "mlx-community/gemma-4-31b-4bit"

_HF_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--gemma-4-31b-4bit"
)
_ENV_FLAG = os.environ.get("SILICA_REAL_GEMMA4_31B") == "1"
_SKIP_REASON = (
    "Gemma4-31B-4bit batched smoke is dual-gated. Required: (1) the "
    f"checkpoint cached at {_HF_CACHE} (run "
    "scripts/probe_gemma4_31b_load.py --repo mlx-community/"
    "gemma-4-31b-4bit to populate, ~18 GB); (2) env var "
    "SILICA_REAL_GEMMA4_31B=1 to opt in — batched 31B is more "
    "expensive than single-request, so we do not run it by default "
    "even when the cache exists."
)
_SKIP = (
    not _HF_CACHE.exists()
    or not _ENV_FLAG
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_engine_generate_batch_runs_on_gemma4_31b() -> None:
    """Public API smoke: ``Engine.generate_batch`` against the
    Gemma4-31B SLIDING-bearing adapter with ``prefix_cache=None`` (the
    D3-guard-compliant path) yields token + done events for every
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
def test_batcher_batch_cache_is_genuinely_hybrid_on_gemma4_31b() -> None:
    """Direct batcher smoke: after ``_prepare_cohort`` runs on a real
    Gemma4-31B adapter, the live ``_batch_cache`` interleaves
    ``BatchRotatingKVCache`` (sliding layers) with ``BatchKVCache``
    (full-attention layers). Proves ``Gemma4Adapter.make_batch_cache``
    actually reached the scheduler rather than the ``callable()``
    fallback producing an all-``BatchKVCache`` list.

    The assertion walks ``adapter.attention_pattern().per_layer`` and
    ``_batch_cache`` in lockstep, pinning the exact per-layer mapping:
    ``SLIDING → BatchRotatingKVCache``, ``GLOBAL → BatchKVCache``. Any
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

    list(batcher.step())

    cache = batcher._batch_cache  # type: ignore[attr-defined]
    assert cache is not None
    assert len(cache) == adapter.config.num_layers

    pattern = adapter.attention_pattern().per_layer
    assert len(pattern) == len(cache)
    for layer_idx, (kind, layer_cache) in enumerate(
        zip(pattern, cache, strict=True)
    ):
        if kind == AttentionKind.SLIDING:
            assert isinstance(layer_cache, BatchRotatingKVCache), (
                f"layer {layer_idx}: kind={kind.value} expected "
                f"BatchRotatingKVCache, got {type(layer_cache).__name__}"
            )
        elif kind == AttentionKind.GLOBAL:
            assert isinstance(layer_cache, BatchKVCache), (
                f"layer {layer_idx}: kind={kind.value} expected "
                f"BatchKVCache, got {type(layer_cache).__name__}"
            )
        else:
            raise AssertionError(
                f"layer {layer_idx}: unexpected AttentionKind "
                f"{kind.value!r} reached the Gemma4 cache-alignment "
                f"smoke — either the capability gate grew a new "
                f"supported kind without a matching case here, or "
                f"Gemma4Adapter.make_batch_cache is producing caches "
                f"for layers it should not own."
            )
