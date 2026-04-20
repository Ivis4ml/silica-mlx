"""P-3-D3.1: Gemma4-31B-4bit batched invariants.

Following P-3-D3 (SLIDING gate lift + miss-only batched smoke), this
module pins the strongest batched correctness claims Gemma4 supports
empirically on the current Apple Silicon + mlx-lm toolchain:

  - B=1 ``Engine.generate_batch`` equals ``Engine.generate`` exactly.
  - Identical prompts in the same batch emit identical rows.
  - B>1 ``Engine.generate_batch`` matches a direct mlx-lm batched
    reference driven with ``Gemma4Adapter.make_batch_cache``.
  - Unequal prompt lengths exercise row lifecycle and left padding.

The original exact B>1 batched-vs-single-request attempt failed
empirically on 2026-04-20: for ``"The capital of France is"`` at
``max_tokens=16`` the first drift appeared at token index 2
(``single=600``, ``batch=529``), while B=1, same-prompt symmetry, and
unequal-length row lifecycle all passed. This mirrors the P-2 plain
Qwen3 caveat more than Qwen3.5-C3d: Gemma4-31B is pure KV attention
(50 sliding + 10 full layers) on a 4-bit checkpoint, so B>1 greedy
sequences are not claimed to equal free-running solo greedy output.
Instead, D3.1 pins the scheduler-owned contract: Silica batched output
must match a direct mlx-lm batched run with the same per-layer cache
types and left-padding convention.

Each run uses a freshly built ``adapter`` + ``Engine`` to keep
mlx-lm's in-place cache from polluting across invocations.

**Dual gate**: real weights cached AND ``SILICA_REAL_GEMMA4_31B=1``
must be set — same gate as ``tests/test_p3_gemma4_batched_smoke.py``.
Single-request parity runs plus one batched run per parity test
multiplies the decode cost of D3's smoke by roughly the number of
tests, so the opt-in env var matters more here than on the D3 smoke.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.factory import adapter_for_repo
from silica.weights.resident import ResidentWeightProvider

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
    "Gemma4-31B-4bit batched invariant suite is dual-gated. Required: "
    "(1) the "
    f"checkpoint cached at {_HF_CACHE} (run "
    "scripts/probe_gemma4_31b_load.py --repo mlx-community/"
    "gemma-4-31b-4bit to populate, ~18 GB); (2) env var "
    "SILICA_REAL_GEMMA4_31B=1 to opt in — batched 31B invariant "
    "tests run real single-request and batched forwards, so the cost "
    "compounds over the tests in this file."
)
_SKIP = (
    not _HF_CACHE.exists()
    or not _ENV_FLAG
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)


def _fresh_engine() -> tuple[Engine, Any]:
    """Build a fresh ``(adapter, Engine)`` pair from the HF checkpoint.

    Every comparison run uses its own pair so the in-place mlx-lm
    cache from a previous call cannot pollute the next one. Weights
    are already on disk in the HF cache, so the second call is
    fast compared to the initial download; we trade a few
    model-construction seconds for cleanly isolated test state.
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
    """Fresh engine -> ``generate(prompt)`` -> list of token ids."""
    engine, adapter = _fresh_engine()
    return list(engine.generate(prompt, _params(adapter, max_tokens=max_tokens)))


def _batch_tokens(
    prompts: Sequence[str],
    max_batch_size: int,
    max_tokens: int,
) -> tuple[dict[int, list[int]], dict[int, str], list[tuple[int, str]]]:
    """Fresh engine -> ``generate_batch`` -> per-req token / done / abort
    collections.

    Returns ``(tokens_by_req, done_by_req, aborts)``. Callers are
    expected to check ``aborts == []`` and ``set(done_by_req) ==
    set(range(len(prompts)))`` *before* comparing tokens, so a parity
    mismatch that is actually an abort failure is never silently
    treated as a drift.

    ``prefix_cache=None`` is non-negotiable here: the D3 constructor
    guard rejects ``prefix_cache != None`` for SLIDING-bearing
    adapters, and passing anything else would fail construction
    before producing tokens.
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


def _direct_mlx_lm_batched_tokens(
    prompts: Sequence[str],
    max_tokens: int,
) -> dict[int, list[int]]:
    """Drive the loaded Gemma4 mlx-lm model directly in batched mode.

    This is the D3.1 reference for B>1 correctness. It bypasses
    ``Engine`` / ``ContinuousBatcher`` but intentionally uses the
    adapter-owned ``make_batch_cache(left_padding)`` factory so the
    per-layer cache list matches Silica's live scheduler path:
    ``BatchRotatingKVCache`` for SLIDING layers and ``BatchKVCache``
    for GLOBAL layers.
    """
    adapter, _ = adapter_for_repo(REPO)
    model = adapter.build(ResidentWeightProvider())
    tokenizer = adapter.tokenizer()
    encoded = [list(tokenizer.encode(prompt)) for prompt in prompts]
    assert all(encoded), "test prompts must tokenize to non-empty ids"

    max_len = max(len(ids) for ids in encoded)
    left_padding = [max_len - len(ids) for ids in encoded]
    padded = [
        [0] * pad + ids for pad, ids in zip(left_padding, encoded, strict=True)
    ]
    make_batch_cache = getattr(adapter, "make_batch_cache", None)
    assert callable(make_batch_cache)
    cache = make_batch_cache(left_padding)

    tokens = mx.array(padded, dtype=mx.int32)
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    last = logits[:, -1, :]
    next_tokens = [
        int(mx.argmax(last[row_idx]).item()) for row_idx in range(len(prompts))
    ]
    out = {row_idx: [tok] for row_idx, tok in enumerate(next_tokens)}

    for _ in range(max_tokens - 1):
        decode = mx.array([[tok] for tok in next_tokens], dtype=mx.int32)
        logits = model(decode, cache=cache)
        mx.eval(logits)
        last = logits[:, -1, :]
        next_tokens = [
            int(mx.argmax(last[row_idx]).item())
            for row_idx in range(len(prompts))
        ]
        for row_idx, tok in enumerate(next_tokens):
            out[row_idx].append(tok)

    return out


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
        "Gemma4 batched parity vs single-request failed:\n"
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
    Any divergence here is a bug in the sliding-bearing batched path's
    B=1 handling — never acceptable. No degradation is offered.
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
    produce identical token sequences, regardless of any mixed-
    precision B>1 drift that might affect batched-vs-solo comparisons.
    Holds because every row sees exactly the same inputs at every
    step (including identical left-padding = 0 on both sides)."""
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


# --- B>1 direct-reference parity ---------------------------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_bgt1_matches_direct_mlx_lm_batched_reference() -> None:
    """B>1 scheduler correctness: Silica batched output matches a
    direct mlx-lm batched run using the same adapter-produced cache
    list and left-padding convention.

    This deliberately does NOT compare against ``Engine.generate`` per
    row. The exact-vs-solo attempt drifted at token index 2 on the
    real 31B-4bit checkpoint, which is a numerical / upstream batched
    kernel property unless this direct reference disagrees with
    Silica. The invariant we own is scheduler glue equivalence to the
    direct batched execution path.
    """
    prompts = [
        "The capital of France is",
        "The capital of Japan is",
    ]
    max_tokens = 16

    tokens_by_req, done_by_req, aborts = _batch_tokens(
        prompts, max_batch_size=2, max_tokens=max_tokens
    )
    direct = _direct_mlx_lm_batched_tokens(prompts, max_tokens=max_tokens)

    assert aborts == []
    assert set(done_by_req) == {0, 1}

    for req_index, prompt in enumerate(prompts):
        batch_tokens = tokens_by_req.get(req_index, [])
        direct_tokens = direct[req_index]
        assert batch_tokens == direct_tokens, (
            "Gemma4 Silica batched output diverged from direct mlx-lm "
            "batched reference:\n"
            f"  prompt:             {prompt!r}\n"
            f"  finish_reason:      {done_by_req[req_index]!r}\n"
            f"  direct tokens:      {direct_tokens}\n"
            f"  silica batch tokens: {batch_tokens}"
        )


# --- Row lifecycle regression smoke (not parity) -----------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_different_length_prompts_yield_per_row_results() -> None:
    """Unequal prompt lengths exercise left-padding bookkeeping,
    specifically through ``BatchRotatingKVCache.prepare(left_padding=
    ...)`` on the sliding layers. Not a parity assertion — only that
    both rows emit tokens and finish cleanly. Catches regressions in
    ``_prepare_cohort`` / ``make_batch_cache(left_padding=...)`` that
    would manifest as aborts or empty token streams on the sliding
    path even when they pass on the all-GLOBAL Qwen3 path."""
    prompts = [
        "Hello",
        "The capital of a European country with Paris as its capital",
    ]
    tokens_by_req, done_by_req, aborts = _batch_tokens(
        prompts, max_batch_size=2, max_tokens=8
    )
    assert aborts == []
    assert set(done_by_req) == {0, 1}
    for req_index in (0, 1):
        assert len(tokens_by_req.get(req_index, [])) >= 1, (
            f"row {req_index} produced no tokens"
        )
