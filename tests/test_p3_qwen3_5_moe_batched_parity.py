"""P-3-E4 Qwen3.5-MoE batched scheduler-glue parity.

Following P-3-E4 (capability-gate lift + B=2 smoke at
`tests/test_p3_qwen3_5_moe_batched_smoke.py`), this module pins the
strongest empirically-defensible batched correctness claim for the
real ``mlx-community/Qwen3.5-35B-A3B-4bit`` MoE checkpoint, mirroring
the dense P-3-D3.1 pattern at `tests/test_p3_gemma4_batched_parity.py`:

  - B=1 ``Engine.generate_batch`` equals ``Engine.generate`` exactly.
  - Identical prompts in the same batch emit identical rows.
  - B>1 ``Engine.generate_batch`` matches a direct mlx-lm batched
    reference driven with ``Qwen3_5MoeAdapter.make_batch_cache``.
  - Unequal-length prompts exercise row lifecycle and left padding.

What this pins (load-bearing claim):
The survey §5 E4 audit's central finding — mlx-lm's ``SwitchGLU`` +
``gather_mm`` path is **B-agnostic** under quantized
``QuantizedSwitchLinear`` weights — was previously verified only at
the smoke level. The B>1 direct-reference test in this module
upgrades that claim to "Silica's batched scheduler glue produces
the same per-row token streams as a direct mlx-lm batched forward
through the adapter's ``make_batch_cache``-produced cache list",
which is the same scheduler-glue parity gate dense Gemma4-31B uses
under D3.1.

What this explicitly does NOT claim (matches survey §5.1 deferral):
Per-row top-k expert indices stability under different right-padding
lengths. The "what counts as the same routing when row-A's token-7
pads differently from row-B's token-7" definition exercise stays
out of scope; this file pins **token-level scheduler-glue parity**,
not expert-routing parity. Token-level matching subsumes the routing
question for the purposes of "scheduler glue is correct on batched
MoE", because if the direct mlx-lm batched reference reaches the
same tokens as Silica, both must have routed through compatible
top-k expert sets at every position.

Each run uses a freshly built ``adapter`` + ``Engine`` to keep
mlx-lm's in-place cache from polluting across invocations (mirrors
D3.1's ``_fresh_engine`` pattern).

**Dual gate**: real weights cached AND ``SILICA_REAL_QWEN3_5_MOE=1``
must be set — the same gate ``test_p3_qwen3_5_moe_batched_smoke.py``
uses. Multiple full forward passes per parity test multiply the
35B-A3B cost roughly in proportion to the number of tests, so the
opt-in env var matters more here than on the smoke.

**Memory note**: 35B-A3B-4bit on disk is ~20 GB; B=2 batched forward
peaks above the single-request ~30 GB envelope on M5 Pro 48 GB. The
B=1-equals-single-request and identical-prompts tests run one
forward pass each. The direct-reference parity test runs two
sequential B=2 forward passes (Silica + direct ref). To stay inside
the 48 GB budget, the parity test does its single B=2 Silica forward,
releases the engine + adapter, then runs the direct mlx-lm reference
on a freshly loaded adapter — see ``_run_parity_under_memory_budget``.
"""

from __future__ import annotations

import gc
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

REPO = "mlx-community/Qwen3.5-35B-A3B-4bit"

_HF_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--Qwen3.5-35B-A3B-4bit"
)
_ENV_FLAG = os.environ.get("SILICA_REAL_QWEN3_5_MOE") == "1"
_SKIP_REASON = (
    "Qwen3.5-35B-A3B-4bit batched MoE parity suite is dual-gated. "
    "Required: (1) the "
    f"checkpoint cached at {_HF_CACHE} (run scripts/probe_qwen3_5_moe_load.py "
    "--repo mlx-community/Qwen3.5-35B-A3B-4bit to populate, ~20 GB); "
    "(2) env var SILICA_REAL_QWEN3_5_MOE=1 to opt in — multiple "
    "full B=2 forward passes per parity test multiply the 35B-A3B "
    "decode cost so we do not run by default even when cached."
)
_SKIP = (
    not _HF_CACHE.exists()
    or not _ENV_FLAG
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)


def _fresh_engine() -> tuple[Engine, Any]:
    """Build a fresh ``(adapter, Engine)`` pair from the HF checkpoint.

    Every comparison run uses its own pair so the in-place mlx-lm cache
    from a previous call cannot pollute the next one. Mirrors D3.1's
    ``_fresh_engine`` for Gemma4-31B.
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
    set(range(len(prompts)))`` before comparing tokens, so a parity
    mismatch that is actually an abort failure is never silently
    treated as a drift.

    ``prefix_cache=None`` is non-negotiable here because the E4
    capability lift accepts MoE only under ``prefix_cache is None``;
    the SLIDING + prefix_cache and recurrent + prefix_cache
    cooperation work is independent and outside E4 scope.
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
    """Drive the loaded Qwen3.5-MoE mlx-lm model directly in batched mode.

    This is the E4 reference for B>1 correctness, mirroring D3.1.
    Bypasses ``Engine`` / ``ContinuousBatcher`` but uses the
    adapter-owned ``make_batch_cache(left_padding)`` so the per-layer
    cache list matches Silica's live scheduler path:
    ``BatchKVCache`` for both HYBRID_DELTANET (layer-level KV is the
    same shape — see ``Qwen3_5Adapter.make_batch_cache``) and GLOBAL
    layers under the Qwen3.5-MoE adapter's inherited ``make_batch_cache``.

    The MoE-specific path runs through mlx-lm's ``SwitchGLU`` +
    ``gather_mm`` quantized fast path inside ``model(tokens, cache=cache)``;
    the survey §5 audit pins this as B-agnostic. This function just
    drives the call; the parity test asserts equality.
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
    assert callable(make_batch_cache), (
        "Qwen3_5MoeAdapter must inherit make_batch_cache from Qwen3_5Adapter "
        "— if this fires the inheritance has been broken"
    )
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


def _release_mlx_state() -> None:
    """Drop Python references to mlx tensors and clear the metal
    allocator before loading a second copy of the 35B-A3B-4bit
    weights for the direct-reference comparison.

    On 48 GB M5 Pro the Silica B=2 forward holds ~30+ GB peak; loading
    a fresh adapter for the direct reference without releasing the
    previous one would push past the device budget. This helper
    centralises the "let go before loading again" handshake.
    """
    gc.collect()
    # ``mx.metal.clear_cache`` purges the allocator's free list so the
    # next allocation does not stack on top of cached blocks. The call
    # is a no-op on non-metal backends; the test is gated to Apple
    # Silicon by SILICA_REAL_QWEN3_5_MOE in practice.
    clear = getattr(getattr(mx, "metal", None), "clear_cache", None)
    if callable(clear):
        clear()


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
        "Qwen3.5-MoE batched parity vs single-request failed:\n"
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
    Any divergence here is a bug in the MoE batched path's B=1 handling
    — never acceptable. No degradation is offered.

    This is the E4 equivalent of D3.1's identically-named test for
    Gemma4-31B. The MoE branch (mlx-lm's ``SwitchGLU`` + ``gather_mm``)
    runs at every layer regardless of B, so B=1 batched and
    single-request must produce bit-identical token streams under
    greedy sampling.
    """
    prompt = "The capital of France is"
    max_tokens = 4

    single_tokens = _single_request_tokens(prompt, max_tokens)
    _release_mlx_state()

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
    produce identical token sequences, regardless of any mixed-precision
    B>1 drift that might affect batched-vs-solo comparisons. Holds
    because every row sees exactly the same inputs at every step
    (including identical left-padding = 0 on both sides) and the
    MoE router computes deterministic top-k expert indices over
    identical hidden states.
    """
    prompt = "The capital of France is"
    tokens_by_req, done_by_req, aborts = _batch_tokens(
        [prompt, prompt], max_batch_size=2, max_tokens=4
    )
    assert aborts == []
    assert set(done_by_req) == {0, 1}
    assert tokens_by_req[0] == tokens_by_req[1], (
        "identical prompts produced divergent batch rows on Qwen3.5-MoE — "
        f"row 0: {tokens_by_req[0]}; row 1: {tokens_by_req[1]}"
    )


# --- B>1 direct-reference parity ---------------------------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_bgt1_matches_direct_mlx_lm_batched_reference() -> None:
    """B>1 scheduler correctness on real MoE weights: Silica batched
    output matches a direct mlx-lm batched run using the adapter's
    ``make_batch_cache`` factory and identical left-padding.

    This is the load-bearing E4 parity claim. The survey §5 audit
    pinned ``SwitchGLU`` + ``gather_mm`` as B-agnostic at the
    algorithm level; this test verifies the audit empirically on a
    real 35B-A3B-4bit checkpoint. If the direct mlx-lm batched
    reference disagrees with Silica, the bug is in Silica's
    scheduler glue, not in mlx-lm's MoE forward.

    Memory pattern: the Silica B=2 forward and the direct mlx-lm B=2
    forward are run sequentially with ``_release_mlx_state`` between
    them so the device sees at most one B=2 forward live at a time.
    Both use the same prompts and ``max_tokens=4`` to keep peak
    memory inside the 48 GB envelope.
    """
    prompts = [
        "The capital of France is",
        "The capital of Japan is",
    ]
    max_tokens = 4

    tokens_by_req, done_by_req, aborts = _batch_tokens(
        prompts, max_batch_size=2, max_tokens=max_tokens
    )
    _release_mlx_state()

    direct = _direct_mlx_lm_batched_tokens(prompts, max_tokens=max_tokens)

    assert aborts == []
    assert set(done_by_req) == {0, 1}

    for req_index, prompt in enumerate(prompts):
        batch_tokens = tokens_by_req.get(req_index, [])
        direct_tokens = direct[req_index]
        assert batch_tokens == direct_tokens, (
            "Qwen3.5-MoE Silica batched output diverged from direct "
            "mlx-lm batched reference:\n"
            f"  prompt:              {prompt!r}\n"
            f"  finish_reason:       {done_by_req[req_index]!r}\n"
            f"  direct tokens:       {direct_tokens}\n"
            f"  silica batch tokens: {batch_tokens}"
        )


# --- Row lifecycle regression smoke (not parity) -----------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_different_length_prompts_yield_per_row_results() -> None:
    """Unequal prompt lengths exercise left-padding bookkeeping
    through ``BatchKVCache.prepare(left_padding=...)`` on the global
    layers and the ``Qwen3_5MoeAdapter.make_batch_cache`` factory
    that backs both ``ContinuousBatcher`` and the direct reference.
    Not a parity assertion — only that both rows emit tokens and
    finish cleanly. Catches regressions in ``_prepare_cohort`` /
    ``make_batch_cache(left_padding=...)`` that would surface as
    aborts or empty token streams on the MoE path."""
    prompts = [
        "Hello",
        "The capital of a European country with Paris as its capital",
    ]
    tokens_by_req, done_by_req, aborts = _batch_tokens(
        prompts, max_batch_size=2, max_tokens=4
    )
    assert aborts == []
    assert set(done_by_req) == {0, 1}
    for req_index in (0, 1):
        assert len(tokens_by_req.get(req_index, [])) >= 1, (
            f"row {req_index} produced no tokens"
        )
