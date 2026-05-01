"""Tests for D-021 step 5 sub-unit (f) — greedy parity on real models.

Pin that ``Engine.generate`` with a self-drafter (``DraftTargetEngine``
loading the same repo as both target and draft) yields byte-equal
token streams to spec-off (``NoopDraftEngine``) under greedy decoding,
within the **first speculative cycle**.

Two cache-only scenarios:

  - **Qwen3-0.6B** — plain GLOBAL attention. Validates the
    speculative engine cycle, ``decode_step_multi`` verify forward,
    bonus sampling, and the I-2 ``KVManager`` rollback contract on
    a non-recurrent target.
  - **Qwen3.5-0.8B** — hybrid (DeltaNet recurrent + GLOBAL attention).
    Validates the (e) slice 2 wiring on a real model: snapshot
    before verify, full-accept ``commit_state``, free_state cleanup.
    Self-drafter under causal greedy always hits full accept, so this
    scenario primarily exercises the full-accept path; the partial-
    reject + replay path is covered by the (i) synthetic three-rollback
    test on top of the (e) slice 2 engine tests.

**Why a single cycle?** Empirically, beyond the first speculative
cycle (``verify_k`` tokens), spec-on and spec-off diverge under fp16
because target's batched ``decode_step_multi`` and the per-step
``decode_step`` loop accumulate KV writes in different reduction
orders. ``tests/test_decode_step_multi_real.py`` pins per-position
argmax stability for short fresh-cache inputs (T <= 4); over a longer
run the noise compounds and eventually flips an argmax. One full
verify cycle (1 prefill yield + γ accepted drafts + 1 bonus =
``verify_k + 1`` tokens) is what (f) can exercise byte-exactly on
this hardware regime. Longer spec runs are validated via the bench
scenarios in sub-unit (h) and the synthetic rollback tests in (i),
not via byte equality against a reference loop.

Both tests skip when the HF cache is empty.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.models.factory import adapter_for_repo
from silica.speculative.draft_target import DraftTargetEngine

# --- cache-only gates ------------------------------------------------------

QWEN3_REPO = "Qwen/Qwen3-0.6B"
QWEN3_5_REPO = "Qwen/Qwen3.5-0.8B"

_QWEN3_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
)
_QWEN3_SKIP = not _QWEN3_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_QWEN3_SKIP_REASON = (
    f"Qwen3-0.6B not cached at {_QWEN3_CACHE}; pull via huggingface-cli "
    "or any test that loads the 0.6B fixture."
)

_QWEN3_5_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3.5-0.8B"
)
_QWEN3_5_SKIP = not _QWEN3_5_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_QWEN3_5_SKIP_REASON = (
    f"Qwen3.5-0.8B not cached at {_QWEN3_5_CACHE}; pull via huggingface-cli "
    "or any test that loads the 0.8B fixture."
)

# verify_k = 4 → γ = 3. One full speculative cycle yields γ + 1 = 4
# tokens (3 drafts + bonus). Plus the prefill's first sample: 5 total.
# Beyond this, fp16 noise between batched verify and per-step decode
# starts flipping an argmax somewhere downstream — see module docstring.
VERIFY_K = 4
PARITY_MAX_TOKENS = 5


# --- helpers ---------------------------------------------------------------


def _run_spec_off(repo: str, prompt: str, params: SamplingParams) -> list[int]:
    """Generate from a fresh adapter under spec-off (default NoopDraftEngine)."""
    adapter, kv = adapter_for_repo(repo)
    engine = Engine(adapter, kv)
    return list(engine.generate(prompt, params))


def _run_spec_on(
    repo: str, prompt: str, params: SamplingParams, *, verify_k: int
) -> list[int]:
    """Generate from a fresh target adapter with a self-drafter loaded
    from the same ``repo``. Independent adapter / KV instances vs the
    spec-off run so cache state cannot leak between configurations."""
    adapter, kv = adapter_for_repo(repo)
    drafter = DraftTargetEngine.from_repo(repo)
    engine = Engine(
        adapter, kv, draft_engine=drafter, verify_k=verify_k
    )
    return list(engine.generate(prompt, params))


def _greedy(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


# --- parity tests ----------------------------------------------------------


@pytest.mark.skipif(_QWEN3_SKIP, reason=_QWEN3_SKIP_REASON)
def test_qwen3_0_6b_self_spec_matches_spec_off_first_cycle() -> None:
    """Qwen3-0.6B: self-drafter under greedy yields a byte-equal stream
    to spec-off across one full speculative cycle (prefill + γ drafts
    + bonus = ``verify_k + 1`` tokens). Plain attention, no recurrent
    state — validates the spec cycle wiring without exercising the
    (e) recurrent rollback path."""
    prompt = "The capital of France is"
    params = _greedy(max_tokens=PARITY_MAX_TOKENS)

    spec_off = _run_spec_off(QWEN3_REPO, prompt, params)
    spec_on = _run_spec_on(QWEN3_REPO, prompt, params, verify_k=VERIFY_K)

    assert len(spec_off) == PARITY_MAX_TOKENS
    assert spec_on == spec_off, (
        f"spec-on / spec-off divergence on Qwen3-0.6B (cycle 1):\n"
        f"  off = {spec_off}\n"
        f"  on  = {spec_on}"
    )


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_qwen3_5_0_8b_self_spec_matches_spec_off_first_cycle() -> None:
    """Qwen3.5-0.8B: self-drafter under greedy yields a byte-equal
    stream to spec-off across one full speculative cycle. Hybrid model
    — exercises the (e) slice 2 wiring's snapshot / commit_state /
    free_state path under full-accept (the case self-drafter always
    hits under causal greedy). Partial-reject / replay path is covered
    by the (i) synthetic three-rollback test once it lands."""
    prompt = "The capital of France is"
    params = _greedy(max_tokens=PARITY_MAX_TOKENS)

    spec_off = _run_spec_off(QWEN3_5_REPO, prompt, params)
    spec_on = _run_spec_on(QWEN3_5_REPO, prompt, params, verify_k=VERIFY_K)

    assert len(spec_off) == PARITY_MAX_TOKENS
    assert spec_on == spec_off, (
        f"spec-on / spec-off divergence on Qwen3.5-0.8B (cycle 1):\n"
        f"  off = {spec_off}\n"
        f"  on  = {spec_on}"
    )
