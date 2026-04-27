"""P-3-E4 Gemma4-MoE batched scheduler-glue parity.

Companion to ``tests/test_p3_qwen3_5_moe_batched_parity.py``. Following
P-3-E4 (capability-gate lift + B=2 smoke at
`tests/test_p3_gemma4_moe_batched_smoke.py`), this module pins the
strongest empirically-defensible batched correctness claim for the
real ``mlx-community/gemma-4-26b-a4b-4bit`` MoE checkpoint, mirroring
the dense P-3-D3.1 pattern at `tests/test_p3_gemma4_batched_parity.py`:

  - B=1 ``Engine.generate_batch`` equals ``Engine.generate`` exactly.
  - Identical prompts in the same batch emit identical rows.
  - B>1 ``Engine.generate_batch`` matches a direct mlx-lm batched
    reference driven with ``Gemma4MoeAdapter.make_batch_cache``.
  - Unequal-length prompts exercise row lifecycle and left padding.

What this pins (load-bearing claim):
The survey §5 E4 audit's central finding — mlx-lm's ``SwitchGLU`` +
``gather_mm`` path is **B-agnostic** under quantized
``QuantizedSwitchLinear`` weights — was previously verified only at
the smoke level on Gemma4-MoE. The B>1 direct-reference test in this
module upgrades that claim to "Silica's batched scheduler glue
produces the same per-row token streams as a direct mlx-lm batched
forward through the adapter's ``make_batch_cache``-produced cache
list (mixed sliding + global per-layer caches)", which is the same
scheduler-glue parity gate dense Gemma4-31B uses under D3.1.

Gemma4-MoE specifically adds the **always-on dense MLP additive
forward** path on top of SwitchGLU experts (E0 survey §3.2:
``has_always_on_dense_mlp=True``). The parity claim covers this
additive path implicitly: if the direct mlx-lm reference and Silica
agree at the token level, both must have summed dense + experts the
same way at every layer.

What this explicitly does NOT claim (matches survey §5.1 deferral):
Per-row top-k expert indices stability under different right-padding
lengths. Same out-of-scope boundary as the Qwen3.5-MoE parity file.

Two independent capability gates apply at construction time, same as
the smoke companion:

  * The ``has_moe=True`` gate — lifted at P-3-E4 (this commit series).
  * The ``SLIDING + prefix_cache is not None`` ctor guard — landed at
    P-3-D3 and **still in force**. The parity tests use
    ``prefix_cache=None`` to stay inside the supported batched surface.

Each run uses a freshly built ``adapter`` + ``Engine`` to keep mlx-lm's
in-place cache from polluting across invocations (mirrors D3.1's
``_fresh_engine`` pattern).

**Dual gate**: real weights cached AND ``SILICA_REAL_GEMMA4_MOE=1``
must be set — same gate as ``test_p3_gemma4_moe_batched_smoke.py``.
Multiple full forward passes per parity test multiply the 26B-A4B
cost roughly in proportion to the number of tests.

**Memory note**: 26B-A4B-4bit is ~16 GB on disk; B=2 batched forward
through the always-on dense MLP + experts path is more expensive per
token than dense Gemma4-31B. The B=1-equals-single-request and
identical-prompts tests run one forward pass each. The
direct-reference parity test runs two sequential B=2 forward passes
(Silica + direct ref) with ``_release_mlx_state`` between them so
the device sees at most one B=2 forward live at a time.
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

REPO = "mlx-community/gemma-4-26b-a4b-4bit"

_HF_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--gemma-4-26b-a4b-4bit"
)
_ENV_FLAG = os.environ.get("SILICA_REAL_GEMMA4_MOE") == "1"
_SKIP_REASON = (
    "Gemma4-26B-A4B-4bit batched MoE parity suite is dual-gated. "
    "Required: (1) the "
    f"checkpoint cached at {_HF_CACHE} (run scripts/probe_gemma4_moe_load.py "
    "--repo mlx-community/gemma-4-26b-a4b-4bit to populate, ~16 GB); "
    "(2) env var SILICA_REAL_GEMMA4_MOE=1 to opt in — the always-on "
    "dense MLP + SwitchGLU experts B=2 forward is more expensive per "
    "token than dense Gemma4-31B."
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
    ``_fresh_engine`` for Gemma4-31B and the Qwen3.5-MoE parity helper.
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
    collections. Mirrors the Qwen3.5-MoE parity helper.

    ``prefix_cache=None`` is non-negotiable here because both the E4
    capability lift (MoE) and the D3 SLIDING + prefix_cache ctor guard
    require it.
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
    """Drive the loaded Gemma4-MoE mlx-lm model directly in batched mode.

    This is the E4 reference for B>1 correctness on Gemma4-MoE. Uses the
    adapter-owned ``make_batch_cache(left_padding)`` so the per-layer
    cache list matches Silica's live scheduler path:
    ``BatchRotatingKVCache`` for SLIDING layers and ``BatchKVCache``
    for GLOBAL layers, identical to dense Gemma4-31B and inherited
    through ``Gemma4MoeAdapter``.

    The MoE-specific path runs through mlx-lm's always-on dense MLP +
    ``SwitchGLU`` + ``gather_mm`` quantized fast path inside
    ``model(tokens, cache=cache)``. The survey §5 audit pins this as
    B-agnostic; this function just drives the call so the parity test
    can assert equality.
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
        "Gemma4MoeAdapter must inherit make_batch_cache from Gemma4Adapter "
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
    allocator before loading a second copy of the 26B-A4B-4bit weights.

    On 48 GB M5 Pro the Silica B=2 forward + always-on dense MLP holds
    significant peak memory; loading a fresh adapter for the direct
    reference without releasing the previous one would push past the
    device budget. Centralises the "let go before loading again"
    handshake; mirrors the Qwen3.5-MoE parity helper.
    """
    gc.collect()
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
        "Gemma4-MoE batched parity vs single-request failed:\n"
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
    Any divergence here is a bug in the Gemma4-MoE batched path's B=1
    handling — never acceptable.

    The MoE branch (mlx-lm's always-on dense MLP + ``SwitchGLU`` +
    ``gather_mm``) runs at every layer regardless of B, so B=1
    batched and single-request must produce bit-identical token
    streams under greedy sampling.
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
    produce identical token sequences. Holds because every row sees
    exactly the same inputs at every step (including identical
    left-padding = 0 on both sides) and the always-on dense MLP +
    SwitchGLU router both compute deterministic outputs over identical
    hidden states.
    """
    prompt = "The capital of France is"
    tokens_by_req, done_by_req, aborts = _batch_tokens(
        [prompt, prompt], max_batch_size=2, max_tokens=4
    )
    assert aborts == []
    assert set(done_by_req) == {0, 1}
    assert tokens_by_req[0] == tokens_by_req[1], (
        "identical prompts produced divergent batch rows on Gemma4-MoE — "
        f"row 0: {tokens_by_req[0]}; row 1: {tokens_by_req[1]}"
    )


# --- B>1 direct-reference parity ---------------------------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_bgt1_matches_direct_mlx_lm_batched_reference() -> None:
    """B>1 scheduler correctness on real Gemma4-MoE weights: Silica
    batched output matches a direct mlx-lm batched run using the
    adapter's mixed-cache ``make_batch_cache`` factory and identical
    left-padding.

    Load-bearing E4 parity claim on Gemma4-MoE. The survey §5 audit
    pinned ``SwitchGLU`` + ``gather_mm`` + always-on dense MLP as
    B-agnostic at the algorithm level; this test verifies the audit
    empirically on a real 26B-A4B-4bit checkpoint. If the direct
    mlx-lm batched reference disagrees with Silica, the bug is in
    Silica's scheduler glue, not in mlx-lm's MoE forward.

    Memory pattern: Silica + direct mlx-lm forwards run sequentially
    with ``_release_mlx_state`` between them; both use the same
    prompts and ``max_tokens=4`` to keep peak inside the 48 GB
    envelope.
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
            "Gemma4-MoE Silica batched output diverged from direct "
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
    through ``BatchRotatingKVCache.prepare(left_padding=...)`` on the
    sliding layers and ``BatchKVCache.prepare(left_padding=...)`` on
    the global layers, both produced by
    ``Gemma4MoeAdapter.make_batch_cache``. Not a parity assertion —
    only that both rows emit tokens and finish cleanly. Catches
    regressions in ``_prepare_cohort`` / ``make_batch_cache`` that
    would surface as aborts or empty token streams on the MoE path
    even when they pass on the all-GLOBAL Qwen3 path."""
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
