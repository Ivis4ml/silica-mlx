"""P-3-C3d: Qwen3.5-0.8B hybrid batched parity against single-request.

Following P-3-C3c (gate lift + smoke-level "doesn't crash"), this
module attempts **exact token parity** between:

  - ``Engine.generate_batch(prompts, params, max_batch_size=N)`` and
  - ``Engine.generate(prompt, params)`` per row.

The attempt is deliberately strict first (the user's directive):
``test_bgt1_strict_parity_matches_single_request`` asserts byte-for-
byte token equality. If it fails empirically, the assertion message
surfaces the raw drift (single-row tokens, batch-row tokens, first
mismatch index, prompt, finish_reason) so the project can decide
whether to degrade to a weaker invariant — matching the approach
``tests/test_p2_batched_parity.py`` took when fp16 batched SDPA drift
on Apple Silicon blocked exact vs-solo parity.

Three "hard" invariants independent of that drift:

  - ``test_b1_batch_equals_single_request`` (hard gate): a batch of
    size 1 is semantically identical to a single request; any
    divergence is a bug in the hybrid batched path's B=1 case.
  - ``test_identical_prompts_yield_identical_rows``: symmetry — if
    every row sees the same tokens, every row must emit the same
    sequence regardless of any fp16 B>1 noise.
  - ``test_different_length_prompts_yield_per_row_results``:
    left-padding / row-lifecycle regression smoke — not parity.

Each run uses a freshly built ``adapter`` + ``Engine`` to keep
mlx-lm's in-place cache from polluting across invocations.

Skipped when Qwen3.5-0.8B is not in the local HF cache — same guard
pattern as ``tests/test_p2_batched_parity.py``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.factory import adapter_for_repo

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


def _fresh_engine() -> tuple[Engine, Any]:
    """Build a fresh ``(adapter, Engine)`` pair from the HF checkpoint.

    Every comparison run uses its own pair so the in-place mlx-lm
    cache from a previous call cannot pollute the next one. Weights
    are already on disk in the HF cache, so the second call is
    fast — we trade a few model-construction seconds for cleanly
    isolated test state.
    """
    adapter, kv = adapter_for_repo(REPO)
    return Engine(adapter, kv), adapter


def _params(adapter: Any, *, max_tokens: int = 8) -> SamplingParams:
    """Single source of truth for SamplingParams across all parity
    tests. Single-request and batched runs MUST share this so a drift
    cannot be attributed to a params-field difference."""
    tokenizer = adapter.tokenizer()
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop_token_ids=eos_ids,
    )


def _single_request_tokens(prompt: str, max_tokens: int) -> list[int]:
    """Fresh engine → ``generate(prompt)`` → list of token ids."""
    engine, adapter = _fresh_engine()
    return list(engine.generate(prompt, _params(adapter, max_tokens=max_tokens)))


def _batch_tokens(
    prompts: Sequence[str],
    max_batch_size: int,
    max_tokens: int,
) -> tuple[dict[int, list[int]], dict[int, str], list[tuple[int, str]]]:
    """Fresh engine → ``generate_batch`` → per-req token / done / abort
    collections.

    Returns ``(tokens_by_req, done_by_req, aborts)``. Callers are
    expected to check ``aborts == []`` and ``set(done_by_req) ==
    set(range(len(prompts)))`` *before* comparing tokens, so a parity
    mismatch that is actually an abort failure is never silently
    treated as a drift.
    """
    engine, adapter = _fresh_engine()
    tokens_by_req: dict[int, list[int]] = {}
    done_by_req: dict[int, str] = {}
    aborts: list[tuple[int, str]] = []
    for event in engine.generate_batch(
        prompts,
        _params(adapter, max_tokens=max_tokens),
        max_batch_size=max_batch_size,
        prefix_cache=None,
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
    return tokens_by_req, done_by_req, aborts


def _format_parity_mismatch(
    *,
    prompt: str,
    single_tokens: list[int],
    batch_tokens: list[int],
    finish_reason: str,
) -> str:
    first_mismatch = next(
        (
            i
            for i, (a, b) in enumerate(
                zip(single_tokens, batch_tokens, strict=False)
            )
            if a != b
        ),
        min(len(single_tokens), len(batch_tokens)),
    )
    return (
        "hybrid batched parity vs single-request failed:\n"
        f"  prompt:            {prompt!r}\n"
        f"  finish_reason:     {finish_reason!r}\n"
        f"  single tokens:     {single_tokens}\n"
        f"  batch row tokens:  {batch_tokens}\n"
        f"  first mismatch at index {first_mismatch}"
    )


# --- Hard gates ---------------------------------------------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_b1_batch_equals_single_request() -> None:
    """A batch of size 1 is semantically identical to a single request.
    Any divergence here is a bug in the hybrid batched path's B=1
    handling — never acceptable. No degradation is offered.
    """
    prompt = "The capital of France is"
    max_tokens = 8

    single_tokens = _single_request_tokens(prompt, max_tokens)
    tokens_by_req, done_by_req, aborts = _batch_tokens(
        [prompt], max_batch_size=1, max_tokens=max_tokens
    )

    assert aborts == []
    assert set(done_by_req) == {0}
    batch_tokens = tokens_by_req.get(0, [])
    assert batch_tokens == single_tokens, _format_parity_mismatch(
        prompt=prompt,
        single_tokens=single_tokens,
        batch_tokens=batch_tokens,
        finish_reason=done_by_req[0],
    )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_identical_prompts_yield_identical_rows() -> None:
    """Symmetry: two identical prompts running in the same batch must
    produce identical token sequences, regardless of any fp16 B>1
    drift that might affect batched-vs-solo comparisons. This holds
    because every row sees exactly the same inputs at every step."""
    prompt = "The capital of France is"
    tokens_by_req, done_by_req, aborts = _batch_tokens(
        [prompt, prompt], max_batch_size=2, max_tokens=8
    )
    assert aborts == []
    assert set(done_by_req) == {0, 1}
    assert tokens_by_req[0] == tokens_by_req[1], (
        "identical prompts produced divergent batch rows — "
        f"row 0: {tokens_by_req[0]}; row 1: {tokens_by_req[1]}"
    )


# --- Strict parity attempt (exact first; degrade only after empirical review) ---


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_bgt1_strict_parity_matches_single_request() -> None:
    """B>1 exact parity: each batch row's tokens equal the
    corresponding ``Engine.generate(prompt)`` single-request output.

    If this fails, the assertion message surfaces the drift for
    offline review; the project then decides whether to degrade this
    invariant (prefix match, first-N tokens, argmax agreement, etc.)
    or fix the code. See ``tests/test_p2_batched_parity.py`` for a
    precedent where fp16 batched SDPA drift on Apple Silicon blocked
    exact vs-solo parity and the invariants were weakened.
    """
    prompts = [
        "The capital of France is",
        "The capital of Japan is",
    ]
    # Empirical check at runtime (2026-04-19): strict parity holds
    # at max_tokens=16, 32, and 64 on Qwen3.5-0.8B. Unlike P-2's
    # Qwen3-0.6B fp16-SDPA batched drift, hybrid DeltaNet (fp32
    # recurrent state, 3:1 D:G ratio) appears stable across B=2.
    # The test stays at 16 — enough to catch early drift if a future
    # cache/forward change degrades stability, but fast enough to
    # keep the suite under a handful of seconds.
    max_tokens = 16

    singles = [
        _single_request_tokens(p, max_tokens) for p in prompts
    ]
    tokens_by_req, done_by_req, aborts = _batch_tokens(
        prompts, max_batch_size=2, max_tokens=max_tokens
    )

    assert aborts == []
    assert set(done_by_req) == {0, 1}

    for req_index, prompt in enumerate(prompts):
        batch_tokens = tokens_by_req.get(req_index, [])
        assert batch_tokens == singles[req_index], _format_parity_mismatch(
            prompt=prompt,
            single_tokens=singles[req_index],
            batch_tokens=batch_tokens,
            finish_reason=done_by_req[req_index],
        )


# --- Row lifecycle regression smoke (not parity) -----------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_different_length_prompts_yield_per_row_results() -> None:
    """Unequal prompt lengths exercise left-padding bookkeeping. Not
    a parity assertion — only that both rows emit tokens and finish
    cleanly. This catches regressions in ``_prepare_cohort`` /
    ``make_batch_cache(left_padding=...)`` that would manifest as
    aborts or empty token streams."""
    prompts = ["Hello", "The capital of a European country with Paris as its capital"]
    tokens_by_req, done_by_req, aborts = _batch_tokens(
        prompts, max_batch_size=2, max_tokens=8
    )
    assert aborts == []
    assert set(done_by_req) == {0, 1}
    for req_index in (0, 1):
        assert len(tokens_by_req.get(req_index, [])) >= 1, (
            f"row {req_index} produced no tokens"
        )
