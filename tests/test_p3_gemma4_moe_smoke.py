"""P-3-E3 (Gemma4-MoE half): single-request smoke on the real 26B-A4B-4bit checkpoint.

After P-3-E1.2 landed ``silica/models/gemma4_moe.py`` and the
factory branch in ``_build_gemma4`` that disambiguates the
``model_type="gemma4"`` collision via ``enable_moe_block``, this
smoke file exercises the same adapter against the real
``mlx-community/gemma-4-26b-a4b-4bit`` weights on-device. Two tests:

  * ``test_adapter_for_repo_dispatches_to_gemma4_moe_adapter`` — the
    factory's local enable_moe_block branch routes a real
    ``model_type="gemma4"`` MoE checkpoint to ``Gemma4MoeAdapter``
    (NOT the dense ``Gemma4Adapter`` that shares the same
    model_type key). Capabilities must report ``has_moe=True``,
    ``has_recurrent_state=False``, and ``attention_kinds``
    covering ``{SLIDING, GLOBAL}``.
  * ``test_engine_generate_produces_tokens_on_gemma4_26b_a4b`` —
    runs ``Engine.generate("Hello", max_tokens=4)`` and asserts a
    non-empty token list with all ids in vocab. Goal is "does not
    crash through the always-on dense MLP + experts additive
    forward path on the real weights"; no token parity.

``Engine.generate`` is single-request; batched MoE coverage
(B=2 with different prompts to exercise per-row top-k expert
dispatch) lives in ``tests/test_p3_gemma4_moe_batched_smoke.py``
after P-3-E4 lifted the ``has_moe=True`` rejection at the
``ContinuousBatcher`` capability gate.

**Dual gate**: real weights cached AND ``SILICA_REAL_GEMMA4_MOE=1``
must be set. Mirrors the Qwen3.5-MoE smoke's gate; the 26B-A4B
checkpoint is ~16 GB on disk and the always-on dense MLP +
SwitchGLU experts forward is more expensive per token than dense
Gemma4-31B single-request.
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
from silica.models.gemma4 import Gemma4Adapter
from silica.models.gemma4_moe import Gemma4MoeAdapter

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
    "Gemma4-26B-A4B-4bit MoE smoke is dual-gated. Required: (1) the "
    f"checkpoint cached at {_HF_CACHE} (run "
    "scripts/probe_gemma4_moe_load.py --repo "
    "mlx-community/gemma-4-26b-a4b-4bit to populate, ~16 GB); (2) env "
    "var SILICA_REAL_GEMMA4_MOE=1 to opt in — the always-on dense "
    "MLP + SwitchGLU experts forward is more expensive per token "
    "than dense Gemma4-31B, so we do not run it by default even when "
    "the cache exists."
)
_SKIP = (
    not _HF_CACHE.exists()
    or not _ENV_FLAG
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_adapter_for_repo_dispatches_to_gemma4_moe_adapter() -> None:
    """E1.2 added a local enable_moe_block branch inside
    ``factory._build_gemma4``. Against the real 26B-A4B repo the
    factory must return a live ``Gemma4MoeAdapter`` (NOT the dense
    ``Gemma4Adapter`` that shares the ``model_type="gemma4"`` key).
    """
    adapter, _ = adapter_for_repo(REPO)
    assert isinstance(adapter, Gemma4MoeAdapter)
    # Defence-in-depth: subclass relationship preserved, but a strict
    # type check would also pass — Gemma4MoeAdapter inherits from
    # Gemma4Adapter, so this isinstance succeeds via subclassing.
    assert isinstance(adapter, Gemma4Adapter)

    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert caps.has_moe is True
    # Gemma4 is pure KV attention — no recurrent state even in MoE
    # variants (unlike Qwen3.5-MoE which inherits DeltaNet).
    assert caps.has_recurrent_state is False
    assert AttentionKind.SLIDING in caps.attention_kinds
    assert AttentionKind.GLOBAL in caps.attention_kinds

    extra = adapter.config.extra
    assert extra["is_moe_adapter"] is True
    # Real 26B-A4B values pinned by E1.2 unit tests.
    assert extra["num_experts"] == 128
    assert extra["top_k_experts"] == 8
    assert extra["moe_intermediate_size"] == 704
    # Distinguishing structural fact recorded by E1.2.
    assert extra["has_always_on_dense_mlp"] is True
    assert extra["moe_expert_path"] == "layer.experts.switch_glu"

    # Per-kind KV budget (D4) inherits unchanged: 25 sliding + 5 full
    # on bfloat16 yields 225,280 bytes/token (matches E0 survey §3.2).
    layout = adapter.kv_layout()
    assert layout.bytes_per_token_total == 225_280


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_engine_generate_produces_tokens_on_gemma4_26b_a4b() -> None:
    """Single-request ``Engine.generate`` against the real 26B-A4B
    MoE weights. Covers prefill + 3 decode steps with the
    always-on dense MLP + SwitchGLU experts additive forward path
    running on every layer."""
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
