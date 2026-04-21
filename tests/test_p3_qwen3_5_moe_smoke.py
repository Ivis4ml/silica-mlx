"""P-3-E3 (Qwen3.5-MoE half): single-request smoke on the real 35B-A3B-4bit checkpoint.

After P-3-E1.1 landed ``silica/models/qwen3_5_moe.py`` with a
unit-tested contract on a fake 35B-A3B-shaped adapter, and P-3-E2
pinned the option (c) dispatch-observation seam against a mock
provider, this smoke file exercises the same adapter against the
real ``mlx-community/Qwen3.5-35B-A3B-4bit`` weights on-device. Two
tests:

  * ``test_adapter_for_repo_dispatches_to_qwen3_5_moe_adapter`` —
    the factory registers ``"qwen3_5_moe"`` and returns a live
    ``Qwen3_5MoeAdapter`` whose capabilities report
    ``has_moe=True``, ``has_recurrent_state=True`` (DeltaNet
    hybrid layers preserved), and ``attention_kinds`` covering
    ``{HYBRID_DELTANET, GLOBAL}``.
  * ``test_engine_generate_produces_tokens_on_qwen3_5_35b_a3b`` —
    runs ``Engine.generate("Hello", max_tokens=4)`` and asserts a
    non-empty token list with all ids in vocab. Goal is "does not
    crash on the real MoE weights"; no token parity claim — the
    MoE FFN runs through mlx-lm's SwitchGLU + gather_mm path
    inside the model forward, which Silica does not validate
    against an HF reference here (E-open-5 already noted that
    mlx-lm silently drops ``attn_output_gate``, so a hypothetical
    HF-parity attempt would fail on attention before reaching MoE).

``Engine.generate`` is single-request; batched MoE remains rejected
at the ``ContinuousBatcher`` capability gate via the
``has_moe=True`` branch (P-3-E4 will revisit).

**Dual gate**: real weights cached AND ``SILICA_REAL_QWEN3_5_MOE=1``
must be set. The 35B-A3B-4bit checkpoint is ~20 GB on disk and the
forward peaks at ~30+ GB device memory; the env-var opt-in avoids
unintended runs even when the cache exists. Mirrors the dual-gate
pattern from ``tests/test_p3_gemma4_batched_smoke.py`` rather than
the cache-only gate from D1.1 — MoE forwards are slower than dense
single-request ones, so the stricter gate is warranted.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.adapter import AttentionKind
from silica.models.capabilities import ModelCapabilities
from silica.models.factory import adapter_for_repo
from silica.models.qwen3_5_moe import Qwen3_5MoeAdapter

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
    "Qwen3.5-35B-A3B-4bit MoE smoke is dual-gated. Required: (1) the "
    f"checkpoint cached at {_HF_CACHE} (run "
    "scripts/probe_qwen3_5_moe_load.py --repo "
    "mlx-community/Qwen3.5-35B-A3B-4bit to populate, ~20 GB); (2) env "
    "var SILICA_REAL_QWEN3_5_MOE=1 to opt in — a 35B-A3B forward is "
    "~30+ GB device memory and slower than dense single-request, so "
    "we do not run it by default even when the cache exists."
)
_SKIP = (
    not _HF_CACHE.exists()
    or not _ENV_FLAG
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_adapter_for_repo_dispatches_to_qwen3_5_moe_adapter() -> None:
    """E1.1 registered ``"qwen3_5_moe"`` in ``factory._ADAPTERS``.
    Against the real 35B-A3B repo the factory must return a live
    ``Qwen3_5MoeAdapter`` whose capabilities honour both the MoE
    flag and the inherited DeltaNet hybrid attention.
    """
    adapter, _ = adapter_for_repo(REPO)
    assert isinstance(adapter, Qwen3_5MoeAdapter)

    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    # Pinned by E1.1: has_moe=True is the override; the rest comes
    # from the dense Qwen3.5 hybrid pattern.
    assert caps.has_moe is True
    assert caps.has_recurrent_state is True
    assert AttentionKind.HYBRID_DELTANET in caps.attention_kinds
    assert AttentionKind.GLOBAL in caps.attention_kinds

    # config.extra MoE metadata pinned by E1.1 — verifies the real
    # checkpoint's text_config matches what the unit tests assumed.
    extra = adapter.config.extra
    assert extra["is_moe_adapter"] is True
    assert extra["num_experts"] == 256
    assert extra["num_experts_per_tok"] == 8
    assert extra["moe_intermediate_size"] == 512
    assert extra["shared_expert_intermediate_size"] == 512
    # E-open-2 resolution: empty on the probed checkpoint.
    assert extra["mlp_only_layers"] == []
    # E-open-5 resolution: mlx-lm silently drops attn_output_gate;
    # the config sets it but Silica records the upstream divergence.
    assert extra["attn_output_gate_mlx_lm_honors"] is False


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_engine_generate_produces_tokens_on_qwen3_5_35b_a3b() -> None:
    """Single-request ``Engine.generate`` against the real 35B-A3B
    MoE weights. "Works end-to-end through the MoE FFN" is the
    claim — no token parity. Covers prefill + 3 decode steps (each
    routing through SwitchGLU + gather_mm + the shared-expert
    branch on every layer)."""
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
    tokens = list(engine.generate("Hello", params))
    assert len(tokens) >= 1
    vocab_size = adapter.config.vocab_size
    for tok in tokens:
        assert isinstance(tok, int)
        assert 0 <= tok < vocab_size
