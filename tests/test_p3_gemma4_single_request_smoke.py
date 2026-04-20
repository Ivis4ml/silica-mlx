"""P-3-D1.1: Gemma4-31B-4bit single-request smoke on the real checkpoint.

After P-3-D1 landed ``silica/models/gemma4.py`` with a unit-tested
contract on a fake 31B-shaped adapter, this smoke file exercises the
same adapter against the real ``mlx-community/gemma-4-31b-4bit``
weights on-device. Two tests:

  * ``test_adapter_for_repo_dispatches_to_gemma4_adapter`` — the
    factory registers ``"gemma4"`` and returns a live
    ``Gemma4Adapter`` with the expected capabilities
    (``{SLIDING, GLOBAL}``, ``has_recurrent_state=False``,
    ``has_moe=False``). Load is ~2 s when the repo is already in the
    HF cache.
  * ``test_engine_generate_produces_tokens_on_gemma4_31b`` — runs
    ``Engine.generate("Hello", max_tokens=4)`` and asserts a non-empty
    token list. The goal is "does not crash on the real weights", not
    token parity — Gemma4 has no known parity anchor in Silica today
    (P-3-D2 / D3 expand the test surface).

``Engine.generate`` is single-request only and does not go through
``ContinuousBatcher``; the batched path for Gemma4 unlocks in
P-3-D3 and is exercised separately in
``tests/test_p3_gemma4_batched_smoke.py`` (dual-gated: cache +
``SILICA_REAL_GEMMA4_31B=1``).

Skipped when Gemma4-31B-4bit is not in the local HF cache — same
skipif pattern as ``tests/test_p2_batched_parity.py`` so new machines
/ CI never silently download the ~18 GB checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.capabilities import ModelCapabilities
from silica.models.factory import adapter_for_repo
from silica.models.gemma4 import Gemma4Adapter

REPO = "mlx-community/gemma-4-31b-4bit"

_HF_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--gemma-4-31b-4bit"
)
_SKIP_REASON = (
    "Gemma4-31B-4bit not cached at "
    f"{_HF_CACHE}; run scripts/probe_gemma4_31b_load.py --repo "
    f"{REPO} to populate it (~18 GB download)."
)
_SKIP = not _HF_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_adapter_for_repo_dispatches_to_gemma4_adapter() -> None:
    """P-3-D1 registered ``"gemma4"`` in ``factory._ADAPTERS``.
    Against the real 31B repo the factory must return a live
    ``Gemma4Adapter`` that declares ``{SLIDING, GLOBAL}`` /
    ``has_recurrent_state=False`` / ``has_moe=False``."""
    adapter, _ = adapter_for_repo(REPO)
    assert isinstance(adapter, Gemma4Adapter)

    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    # attention_kinds is a frozenset of enum members; compare by
    # membership to stay robust to enum iteration order.
    kinds = caps.attention_kinds
    # Import here to keep the smoke's module-level imports minimal.
    from silica.models.adapter import AttentionKind

    assert AttentionKind.SLIDING in kinds
    assert AttentionKind.GLOBAL in kinds
    assert caps.has_recurrent_state is False
    assert caps.has_moe is False

    # ModelConfig.extra carries the per-kind detail the single-shape
    # KVLayout summary cannot express (D-open-1 option (a)).
    extra = adapter.config.extra
    assert extra["kv_layout_summary"] == "sliding_attention"
    assert extra["sliding_kv_heads"] == 16
    assert extra["sliding_head_dim"] == 256
    assert extra["global_kv_heads"] == 4
    assert extra["global_head_dim"] == 512
    assert extra["sliding_window"] == 1024
    assert "summary" in extra["kv_layout_caveat"]


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_engine_generate_produces_tokens_on_gemma4_31b() -> None:
    """Single-request ``Engine.generate`` against the real 31B-4bit
    weights. "Works end-to-end" is the claim — no token parity.

    max_tokens=4 keeps the smoke fast; the adapter's forward path
    (prefill + 3 decode steps) is what the test is covering.
    """
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
    # Token ids must lie inside the tokenizer's vocabulary (sanity
    # check that forward / sampler didn't produce garbage / negatives).
    vocab_size = adapter.config.vocab_size
    for tok in tokens:
        assert isinstance(tok, int)
        assert 0 <= tok < vocab_size
