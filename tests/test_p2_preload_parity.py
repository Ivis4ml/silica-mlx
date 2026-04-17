"""P-2 Unit 16a real-model parity — Engine.generate_batch B=1 == Engine.generate.

Skipped when the Qwen3-0.6B checkpoint is not cached locally (fresh
clones, minimal CI). Run ``python scripts/probe_p2_preload.py`` once to
populate the cache.

The oracle token sequence is the one recorded in docs/P2_PRELOAD.md.
When mlx-lm upgrades drift greedy output, this test fails loud — that
is the intended regression guard (see the §"Unit 16a oracle
specification" block in the preload doc for why storing rather than
recomputing).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.models.qwen3 import Qwen3Adapter

REPO = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
MAX_TOKENS = 16

# Recorded in docs/P2_PRELOAD.md §"Unit 16a oracle specification".
EXPECTED_TOKENS: tuple[int, ...] = (
    12095, 13, 576, 6722, 315, 15344, 374, 21718, 13,
    576, 6722, 315, 17689, 374, 24081, 13,
)

_QWEN3_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
)
_SKIP_REASON = (
    "Qwen3-0.6B not cached at "
    f"{_QWEN3_CACHE}; run scripts/probe_p2_preload.py to populate it."
)
_SKIP = not _QWEN3_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_p1_engine_generate_matches_oracle() -> None:
    adapter, kv = Qwen3Adapter.from_hf_repo(REPO)
    engine = Engine(adapter, kv)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    tokens = tuple(engine.generate(PROMPT, params))
    assert tokens == EXPECTED_TOKENS


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_p2_generate_batch_matches_oracle_at_b1() -> None:
    """Unit 16a parity gate: batched path at B=1 must match P-1 byte-for-byte."""
    adapter, kv = Qwen3Adapter.from_hf_repo(REPO)
    engine = Engine(adapter, kv)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    events = list(engine.generate_batch([PROMPT], params))
    tokens = tuple(e.token_id for e in events if e.kind == "token")
    assert tokens == EXPECTED_TOKENS


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_p1_and_p2_agree_on_the_same_model_instance() -> None:
    """Strongest parity assertion: both paths driving the same adapter agree."""
    adapter, kv = Qwen3Adapter.from_hf_repo(REPO)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    p1_engine = Engine(adapter, kv)
    p1_tokens = tuple(p1_engine.generate(PROMPT, params))

    p2_engine = Engine(adapter, kv)
    p2_events = list(p2_engine.generate_batch([PROMPT], params))
    p2_tokens = tuple(e.token_id for e in p2_events if e.kind == "token")

    assert p1_tokens == p2_tokens
