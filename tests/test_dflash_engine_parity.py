"""D-021 step 6 (ε) — cycle-1 byte-exact spec-on / spec-off parity gate.

§6.2 tier-1 correctness gate from ``plans/P6_C4_DFLASH_OPENING.md``:
on cached ``Qwen/Qwen3.5-0.8B``, with the (γ) synthetic-emitter
oracle-replay design, spec-on (``--speculative dflash`` simulated by
``DFlashDrafter.for_synthetic``) yields token sequences byte-exact
with spec-off (``NoopDraftEngine``) under fixed greedy decoding.

Design (from the user's ε guidance):

- Run spec-off with ``max_tokens=N`` to produce the oracle list of
  ``N`` greedy tokens.
- For verify_k=4 (γ=3 drafts/cycle), construct a synthetic emitter
  that walks the oracle starting after the prefill argmax. Cycle-1
  proposes ``oracle[1:1+γ]`` (3 drafts); on acceptance the engine
  yields ``oracle[1:1+γ]`` plus the bonus from
  ``verify_logits[γ]``, which under greedy + the same target equals
  ``oracle[1+γ]``. Spec-on output is asserted equal to ``oracle[:N]``.

Cycle-1 has no batched-vs-sequential KV reduction-order divergence,
so byte equality is achievable at this regime per
``plans/P6_SPEC_FOUNDATION_OPENING.md`` §6.1 (f) closure. Long-run
parity is bounded by fp16 dispatch noise and is **not** asserted.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from silica.core.profiler import MetricsRegistry
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.models.qwen3_5 import Qwen3_5Adapter
from silica.speculative.dflash_drafter import DFlashDrafter
from silica.speculative.engine import NoopDraftEngine

REPO = "Qwen/Qwen3.5-0.8B"
PROMPT = "Hello, world."
VERIFY_K = 4
MAX_TOKENS = 5  # Cycle-1 yields 4 tokens (anchor + γ + bonus); +1 buffer.

_QWEN3_5_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3.5-0.8B"
)
_SKIP = not _QWEN3_5_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_SKIP_REASON = (
    f"Qwen3.5-0.8B not cached at {_QWEN3_5_CACHE}; pull via "
    "huggingface-cli or any test that loads the 0.8B fixture."
)


def _make_oracle_emitter(oracle_drafts: list[int]):  # type: ignore[no-untyped-def]
    """Synthetic emitter that walks ``oracle_drafts`` linearly. Each
    call returns the next ``k`` tokens; subsequent calls advance the
    cursor. Used to replay a spec-off-recorded sequence so
    ``greedy_verify`` accepts the full block at every cycle."""
    cursor = {"i": 0}

    def emit(target_hidden, k):  # type: ignore[no-untyped-def]
        del target_hidden
        i = cursor["i"]
        out = tuple(oracle_drafts[i : i + k])
        cursor["i"] = i + len(out)
        return out

    return emit


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_dflash_oracle_replay_produces_byte_equal_spec_on_output() -> None:
    """Spec-off oracle: run NoopDraftEngine, record N tokens.
    Spec-on: synthetic DFlashDrafter emits the recorded drafts; the
    engine output must equal the oracle byte-for-byte under fixed
    greedy decoding on cycle 1."""
    # 1. Spec-off oracle.
    adapter_off, kv_off = Qwen3_5Adapter.from_hf_repo(REPO)
    engine_off = Engine(
        adapter=adapter_off,
        kv_manager=kv_off,
        sampler=Sampler(),
        metrics=MetricsRegistry(),
        draft_engine=NoopDraftEngine(),
        verify_k=VERIFY_K,
    )
    oracle = list(
        engine_off.generate(
            prompt=PROMPT,
            params=SamplingParams(
                max_tokens=MAX_TOKENS, temperature=0.0
            ),
        )
    )
    assert len(oracle) == MAX_TOKENS, (
        f"spec-off produced {len(oracle)} tokens; expected {MAX_TOKENS}"
    )

    # 2. Spec-on with synthetic drafter that replays oracle drafts.
    # Engine cycle 0 yields oracle[0] from the prefill argmax. Cycle 1
    # then proposes drafts; the synthetic emitter walks the oracle
    # starting at index 1 (oracle[0] is already yielded as the
    # prefill anchor).
    adapter_on, kv_on = Qwen3_5Adapter.from_hf_repo(REPO)
    # Pick three drafter target_layer_ids spread across Qwen3.5-0.8B's
    # 24-layer stack. Valid range is 0..num_layers-1 = 0..23 — the +1
    # offset maps these to silica's capture-dict keys 1..num_layers.
    # The synthetic emitter ignores ``target_hidden`` so the layer
    # choice does not affect the oracle replay; the choice just
    # exercises the capture path at multiple depths.
    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0, 11, 23),
        synthetic_emit=_make_oracle_emitter(oracle[1:]),
    )
    engine_on = Engine(
        adapter=adapter_on,
        kv_manager=kv_on,
        sampler=Sampler(),
        metrics=MetricsRegistry(),
        draft_engine=drafter,
        verify_k=VERIFY_K,
    )
    spec_on = list(
        engine_on.generate(
            prompt=PROMPT,
            params=SamplingParams(
                max_tokens=MAX_TOKENS, temperature=0.0
            ),
        )
    )

    # 3. Byte-exact equality. Cycle 1 (verify_k=4 → γ=3 drafts) is in
    # the regime where fp16 batched-vs-sequential KV reduction-order
    # noise has not yet flipped any argmax — see
    # plans/P6_SPEC_FOUNDATION_OPENING.md §6.1 (f) closure. Asserting
    # the full MAX_TOKENS=5 sequence covers the prefill argmax + one
    # full spec cycle (γ accepted drafts + bonus).
    assert spec_on == oracle, (
        f"oracle replay failed:\n  spec_off = {oracle}\n  spec_on  = {spec_on}"
    )
