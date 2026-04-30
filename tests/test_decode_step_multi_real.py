"""Real-model greedy-equivalence tests for ``decode_step_multi``.

D-021 step 5 sub-unit (a2) plain-adapter slice (slice 2) and inherited
MoE slice (slice 3). Each test asserts that
``adapter.decode_step_multi(tokens, kv_handle)`` produces logits
position-for-position equal to a sequential ``decode_step`` loop on
the same model — within numerical tolerance — across the v0.1
verify_k=4 window plus boundary T values.

This is the load-bearing greedy-equivalence pin that the OPENING §3
sub-unit (a2) calls out: if a multi-token forward over T positions
diverges from T sequential single-token forwards, the spec verify
silently produces wrong tokens.

Two test gates with different cost profiles:

  - Plain Qwen3-0.6B (cache-only skip; no env-var gate). Always-on
    when the small fixture is available locally — runs in CI dev
    machines, exercises the shared ``forward_full`` helper that
    Qwen3 and Gemma4 both consume.
  - Gemma4-26B-A4B MoE (dual-gated: cache hit AND
    ``SILICA_REAL_GEMMA4_MOE=1``). Opt-in only — the 16 GB load
    plus dense+MoE forward per call is heavier than the dense
    plain-adapter row, so it does not run on every local test
    invocation. Same equivalence contract.

The hybrid Qwen3.5 path (with DeltaNet) lands its own sub-slice and
greedy-equivalence pin then.

Test pattern: build two independent ``(adapter, kv)`` instances
loaded from the same repo. On instance A, run ``decode_step_multi``
over T tokens. On instance B, run ``decode_step`` T times sequentially
over the same tokens. Compare per-position logits.

Two adapter instances avoids forking mlx-lm cache state — both paths
start from a fresh ``KVCache`` and consume the same tokens in order;
mlx-lm's ``model.__call__`` produces identical logits because the
attention is causal and per-position.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from silica.kvcache.manager import KVHandle
from silica.models.factory import adapter_for_repo
from silica.models.qwen3 import Qwen3Adapter

REPO = "Qwen/Qwen3-0.6B"
GEMMA4_MOE_REPO = "mlx-community/gemma-4-26b-a4b-4bit"

_QWEN3_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
)
_SKIP_REASON = (
    f"Qwen3-0.6B not cached at {_QWEN3_CACHE}; run "
    "scripts/probe_p2_preload.py to populate it."
)
_SKIP = not _QWEN3_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)


_GEMMA4_MOE_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--gemma-4-26b-a4b-4bit"
)
_GEMMA4_MOE_ENV = os.environ.get("SILICA_REAL_GEMMA4_MOE") == "1"
_GEMMA4_MOE_SKIP_REASON = (
    "Gemma4-26B-A4B MoE decode_step_multi greedy-equivalence is "
    "dual-gated. Required: (1) checkpoint cached at "
    f"{_GEMMA4_MOE_CACHE} (run scripts/probe_gemma4_moe_load.py "
    "--repo mlx-community/gemma-4-26b-a4b-4bit to populate, "
    "~16 GB); (2) env var SILICA_REAL_GEMMA4_MOE=1 to opt in."
)
_GEMMA4_MOE_SKIP = (
    not _GEMMA4_MOE_CACHE.exists()
    or not _GEMMA4_MOE_ENV
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)


def _greedy_equivalence_window(
    multi_logits: mx.array,
    loop_logits: list[mx.array],
    *,
    rtol: float = 0.02,
    atol: float = 0.5,
) -> None:
    """Assert position-for-position logit equivalence within tolerance.

    Greedy correctness is what the spec verify actually reads —
    ``argmax`` must agree exactly per position. The per-element
    numerical bound is a regression guard against off-by-one
    alignment, attention-mask corruption, and similar shape-level
    bugs (which shift entire rows and produce max_abs >> 1).

    Tolerance is sized for fp16 reduction-order noise between an
    N-token batched forward and N single-token forwards on Qwen3-0.6B:
    matmul accumulation order in attention differs between the two
    paths, producing a few-percent relative drift in the logit values
    while leaving the argmax stable. ``rtol=0.02`` (2%) + ``atol=0.5``
    accommodates that without masking real correctness bugs.
    """
    assert multi_logits.shape[0] == len(loop_logits), (
        f"position count mismatch: multi has {multi_logits.shape[0]} "
        f"rows, loop has {len(loop_logits)}"
    )
    for i, loop_pos in enumerate(loop_logits):
        multi_pos = multi_logits[i]
        # Argmax must agree exactly — that is what greedy verify reads.
        multi_top1 = int(mx.argmax(multi_pos).item())
        loop_top1 = int(mx.argmax(loop_pos).item())
        assert multi_top1 == loop_top1, (
            f"position {i}: argmax diverges (multi={multi_top1}, "
            f"loop={loop_top1})"
        )
        # Per-element numerical distance bound.
        max_abs = float(mx.max(mx.abs(multi_pos - loop_pos)).item())
        loop_max = float(mx.max(mx.abs(loop_pos)).item())
        bound = rtol * loop_max + atol
        assert max_abs <= bound, (
            f"position {i}: |multi - loop|_max={max_abs:.4e} exceeds "
            f"rtol*|loop| + atol = {bound:.4e} (loop_max={loop_max:.4e})"
        )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
@pytest.mark.parametrize("T", [1, 2, 4])
def test_qwen3_decode_step_multi_matches_decode_step_loop(T: int) -> None:
    # Two adapter instances so the two paths each get a fresh KVCache.
    # The model weights are loaded from the HF cache (no network call
    # since the cache hit is a precondition); each load takes a few
    # seconds, so this test is parametrized rather than re-loading per
    # T inside the test body.
    adapter_multi, kv_multi = Qwen3Adapter.from_hf_repo(REPO)
    adapter_loop, kv_loop = Qwen3Adapter.from_hf_repo(REPO)

    req_id = "spec-eq-test"
    handle_multi = KVHandle(req_id=req_id)
    handle_loop = KVHandle(req_id=req_id)

    kv_multi.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_loop.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    # A short, deterministic token sequence. Picking small ids keeps
    # the test independent of tokenizer details — the model's response
    # to arbitrary in-vocab tokens is what we compare.
    token_ids = [101, 202, 303, 404][:T]

    # Multi path: one decode_step_multi over all T tokens.
    tokens_arr = mx.array(token_ids, dtype=mx.int32)
    multi_logits, _ = adapter_multi.decode_step_multi(tokens_arr, handle_multi)

    # Loop path: T sequential decode_step calls.
    loop_logits: list[mx.array] = []
    for tid in token_ids:
        single_arr = mx.array([tid], dtype=mx.int32)
        per_step_logits, _ = adapter_loop.decode_step(single_arr, handle_loop)
        loop_logits.append(per_step_logits)

    # Shape contract: ``decode_step_multi`` returns ``(T, V)`` where V
    # matches the model's actual output dim. Qwen3-0.6B's embedding is
    # padded above ``config.vocab_size`` (151936 vs 151643), so we
    # assert against the loop path's per-step shape rather than the
    # config — both forwards consume the same model and must agree.
    assert multi_logits.shape[0] == T
    assert multi_logits.shape[1] == loop_logits[0].shape[0]

    _greedy_equivalence_window(multi_logits, loop_logits)


@pytest.mark.skipif(_GEMMA4_MOE_SKIP, reason=_GEMMA4_MOE_SKIP_REASON)
@pytest.mark.parametrize("T", [1, 2, 4])
def test_gemma4_moe_decode_step_multi_matches_decode_step_loop(T: int) -> None:
    # Gemma4MoeAdapter inherits decode_step_multi from Gemma4Adapter
    # (slice 3 inheritance pin). This real-model test confirms the
    # inherited path produces position-for-position equivalent logits
    # against a sequential decode_step loop on the same MoE model —
    # the always-on dense MLP + SwitchGLU experts forward must agree
    # with the per-token routing that decode_step would do separately
    # on each token, position by position.
    adapter_multi, kv_multi = adapter_for_repo(GEMMA4_MOE_REPO)
    adapter_loop, kv_loop = adapter_for_repo(GEMMA4_MOE_REPO)

    req_id = "spec-eq-test-gemma4-moe"
    handle_multi = KVHandle(req_id=req_id)
    handle_loop = KVHandle(req_id=req_id)

    kv_multi.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_loop.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    # Same deterministic in-vocab token sequence as the Qwen3 path; the
    # exact ids are unimportant — what matters is feeding identical
    # tokens through both forwards in identical order.
    token_ids = [101, 202, 303, 404][:T]

    tokens_arr = mx.array(token_ids, dtype=mx.int32)
    multi_logits, _ = adapter_multi.decode_step_multi(tokens_arr, handle_multi)

    loop_logits: list[mx.array] = []
    for tid in token_ids:
        single_arr = mx.array([tid], dtype=mx.int32)
        per_step_logits, _ = adapter_loop.decode_step(single_arr, handle_loop)
        loop_logits.append(per_step_logits)

    assert multi_logits.shape[0] == T
    assert multi_logits.shape[1] == loop_logits[0].shape[0]

    _greedy_equivalence_window(multi_logits, loop_logits)
