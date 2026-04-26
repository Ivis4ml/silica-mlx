"""P-3-E4 (Gemma4-MoE half): batched B=2 smoke on the real 26B-A4B-4bit checkpoint.

After P-3-E4 lifted the ``has_moe=True`` rejection at the
``ContinuousBatcher`` capability gate (commit B+E), this file
exercises the live MoE adapter against the real
``mlx-community/gemma-4-26b-a4b-4bit`` weights with B=2 and
**different prompts per row**. Same-prompt B=2 would degenerate
the test by collapsing both rows onto the same top-k experts
and the per-row top-k dispatch claim (the survey §5 E4 audit's
central B-agnostic finding) would not be exercised.

Two independent capability gates apply at construction time:

  * The ``has_moe=True`` gate — lifted at P-3-E4 (this commit
    series).
  * The ``SLIDING + prefix_cache is not None`` ctor guard —
    landed at P-3-D3 and **still in force**. Gemma4 is pure KV
    attention with a 5:1 sliding-window ratio; sliding-window
    cooperation with ``RadixPrefixCache`` is its own piece of
    scheduler work and is not in E4 scope. The smoke uses
    ``prefix_cache=None`` to stay inside the supported batched
    surface.

Single test:

  * ``test_engine_generate_batch_runs_on_gemma4_moe`` — runs
    ``Engine.generate_batch`` over ``["Hello", "The capital of
    France is"]`` with ``max_batch_size=2`` and
    ``prefix_cache=None``. Asserts no abort events, both rows
    emit at least one token, and both rows emit ``done``. Peak
    device memory recorded for the commit message.

No token parity claim — survey §5.1 explicitly defers parity to
post-P-5 alongside (b-static), and Gemma4-MoE additionally has
the always-on dense MLP + SwitchGLU experts additive forward
path (E0 survey §3.2: ``has_always_on_dense_mlp=True``), which
amplifies the parity-definition difficulty over Qwen3.5-MoE's
single-MoE-branch topology.

**Dual gate**: real weights cached AND ``SILICA_REAL_GEMMA4_MOE=1``
must be set. Mirrors ``tests/test_p3_gemma4_moe_smoke.py`` (the
single-request E3 smoke). The 26B-A4B-4bit checkpoint is ~16 GB
on disk; B=2 batched forward through the always-on dense MLP +
experts path is more expensive than dense Gemma4-31B, so the
env-var opt-in stays warranted.

**Memory note**: the conservative scope (short prompts,
``max_tokens=4``, B=2) is chosen so peak memory stays inside
the 48 GB target. If a future checkpoint exceeds the envelope,
the test can fall back to B=1 by changing ``max_batch_size=2``
to ``max_batch_size=1`` and using a single prompt — the
gate-lift unit tests in
``tests/test_batcher.py::test_capability_gate_accepts_has_moe_*``
remain the durable verification regardless of memory envelope.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.factory import adapter_for_repo

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
    "Gemma4-26B-A4B-4bit batched MoE smoke is dual-gated. Required: (1) "
    f"the checkpoint cached at {_HF_CACHE} (run "
    "scripts/probe_gemma4_moe_load.py --repo "
    "mlx-community/gemma-4-26b-a4b-4bit to populate, ~16 GB); (2) env "
    "var SILICA_REAL_GEMMA4_MOE=1 to opt in — the always-on dense MLP "
    "+ SwitchGLU experts B=2 forward is more expensive per token than "
    "dense Gemma4-31B, so we do not run it by default even when the "
    "cache exists."
)
_SKIP = (
    not _HF_CACHE.exists()
    or not _ENV_FLAG
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_engine_generate_batch_runs_on_gemma4_moe() -> None:
    """B=2 batched smoke through the public ``Engine.generate_batch``
    API on the real 26B-A4B-4bit MoE weights. Different prompts per
    row exercise per-row top-k expert dispatch through SwitchGLU +
    gather_mm under the always-on dense MLP additive forward path.
    Pre-E4 this raised ``NotImplementedError`` at
    ``ContinuousBatcher`` construction via the ``has_moe=True``
    branch; post-E4 the gate accepts (still under
    ``prefix_cache=None`` because the SLIDING + prefix_cache guard
    is independent of has_moe and remains in force).
    """
    mx.reset_peak_memory()

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

    prompts = ["Hello", "The capital of France is"]

    tokens_by_req: dict[int, list[int]] = {}
    done_by_req: dict[int, str] = {}
    aborts: list[tuple[int, str]] = []
    for event in engine.generate_batch(
        prompts, params, max_batch_size=2, prefix_cache=None
    ):
        if event.kind == "token":
            assert event.token_id is not None
            tokens_by_req.setdefault(event.req_index, []).append(
                event.token_id
            )
        elif event.kind == "done":
            assert event.finish_reason is not None
            done_by_req[event.req_index] = event.finish_reason
        elif event.kind == "aborted":
            assert event.finish_reason is not None
            aborts.append((event.req_index, event.finish_reason))

    peak_mb = float(mx.get_peak_memory()) / 1e6

    assert aborts == [], (
        f"unexpected abort events on B=2 MoE batched run: {aborts} "
        f"(peak={peak_mb:.0f} MB)"
    )
    assert set(tokens_by_req) == {0, 1}, (
        f"missing token events for some rows: got {sorted(tokens_by_req)} "
        f"(peak={peak_mb:.0f} MB)"
    )
    assert set(done_by_req) == {0, 1}, (
        f"missing done events for some rows: got {sorted(done_by_req)} "
        f"(peak={peak_mb:.0f} MB)"
    )
    for req_index in (0, 1):
        assert len(tokens_by_req[req_index]) >= 1, (
            f"req {req_index} emitted no tokens (peak={peak_mb:.0f} MB)"
        )
        vocab_size = adapter.config.vocab_size
        for tok in tokens_by_req[req_index]:
            assert isinstance(tok, int)
            assert 0 <= tok < vocab_size

    print(
        f"gemma4-moe batched smoke: peak={peak_mb:.0f} MB, "
        f"tokens_by_req={tokens_by_req}"
    )
