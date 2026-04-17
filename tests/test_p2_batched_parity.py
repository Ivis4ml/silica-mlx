"""P-2 Unit 16b real-model parity for the fixed-cohort batched path.

Skipped when Qwen3-0.6B is not cached (run scripts/probe_p2_preload.py).

**Important oracle caveat (discovered during 16b implementation).** In
fp16 on Apple Silicon Metal, mlx-lm's batched SDPA at ``B > 1`` produces
per-row outputs that **do not** numerically match the corresponding
``B = 1`` solo generations, even with ``left_padding = [0, ..., 0]``
(i.e. identical prompt lengths). The first few greedy tokens typically
match, then fp16 round-off from parallel reductions flips an argmax and
the streams diverge. This is a property of mlx-lm / MLX, not a Silica
bug: we verified it by driving ``model + BatchKVCache`` directly with
the same prompts and getting the same divergence.

Consequently, the Unit 16b real-model oracle cannot be "batched row ==
solo". The acceptance tests instead pin four weaker-but-honest
properties:

1. **B=1 parity with solo holds.** 16a's guarantee still valid.
2. **Identical prompts → identical rows.** Symmetry: if every row sees
   the same tokens, all rows must emit the same sequence.
3. **Silica batched == mlx-lm direct batched.** Silica's wrapper adds
   no bugs on top of mlx-lm; any batched-vs-solo drift lives one layer
   below us.
4. **Left-padding matches direct mlx-lm.** Unequal prompt lengths use
   the same left-padding convention as a hand-driven BatchKVCache.

The "true" batched-vs-solo equivalence is a numerical-kernel concern
deferred to the paged-attention kernel track (P-2 Opening §Paged
kernel trigger-gated future track). Recording this caveat in code
rather than docs means a future kernel upgrade's effect is
immediately visible: if batched-vs-solo then becomes exact, we upgrade
the test instead of shipping a quiet regression.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache
from mlx_lm.utils import load

from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.models.qwen3 import Qwen3Adapter

REPO = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 10

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
def test_b1_parity_still_holds_after_16b_changes() -> None:
    """Regression: 16a's B=1 oracle still green on the new cohort path.

    At B=1 there is no batched SDPA drift — solo parity must hold."""
    adapter, kv = Qwen3Adapter.from_hf_repo(REPO)
    params = SamplingParams(temperature=0.0, max_tokens=16)
    prompt = "The capital of France is"

    solo = list(Engine(adapter, kv).generate(prompt, params))
    events = list(Engine(adapter, kv).generate_batch([prompt], params))
    batched = [
        e.token_id for e in events if e.kind == "token" and e.token_id is not None
    ]
    assert batched == solo


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_identical_prompts_yield_identical_rows() -> None:
    """Symmetry: B=3 with the same prompt on every row → all rows identical."""
    adapter, kv = Qwen3Adapter.from_hf_repo(REPO)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    prompt = "The capital of France is"

    engine = Engine(adapter, kv)
    events = list(engine.generate_batch([prompt, prompt, prompt], params))
    rows: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for e in events:
        if e.kind == "token" and e.token_id is not None:
            rows[e.req_index].append(e.token_id)
    assert len(rows[0]) == MAX_TOKENS
    assert rows[0] == rows[1] == rows[2]


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_silica_batched_matches_mlx_lm_direct_batched() -> None:
    """Silica's ContinuousBatcher at B=2 produces the same per-row tokens
    as driving mlx-lm's model + BatchKVCache by hand. Any batched-vs-solo
    drift is upstream; we only own the scheduling and sampling glue."""
    adapter, kv = Qwen3Adapter.from_hf_repo(REPO)
    tokenizer = adapter.tokenizer()

    prompts = ["France is", "Spain is"]
    ids = [tokenizer.encode(p) for p in prompts]
    # Guard: both prompts must be the same length here (otherwise we'd
    # need left-padding in the raw mlx-lm driver below; we stick to the
    # simpler equal-length case for this specific parity test).
    assert len(ids[0]) == len(ids[1])

    # Silica path
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    events = list(Engine(adapter, kv).generate_batch(prompts, params))
    silica: dict[int, list[int]] = {0: [], 1: []}
    for e in events:
        if e.kind == "token" and e.token_id is not None:
            silica[e.req_index].append(e.token_id)

    # mlx-lm direct path — same model instance, fresh BatchKVCache.
    model, _ = load(REPO)  # type: ignore[misc]
    num_layers = len(model.layers)
    bkv = [BatchKVCache(left_padding=[0, 0]) for _ in range(num_layers)]
    tokens = mx.array(ids, dtype=mx.int32)
    logits = model(tokens, cache=bkv)
    mx.eval(logits)
    last = logits[:, -1, :]
    t0 = int(mx.argmax(last[0]).item())
    t1 = int(mx.argmax(last[1]).item())
    direct: dict[int, list[int]] = {0: [t0], 1: [t1]}
    for _ in range(MAX_TOKENS - 1):
        dec = mx.array([[t0], [t1]], dtype=mx.int32)
        logits = model(dec, cache=bkv)
        mx.eval(logits)
        last = logits[:, -1, :]
        t0 = int(mx.argmax(last[0]).item())
        t1 = int(mx.argmax(last[1]).item())
        direct[0].append(t0)
        direct[1].append(t1)

    assert silica[0] == direct[0]
    assert silica[1] == direct[1]


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_left_padding_does_not_corrupt_any_row() -> None:
    """With unequal prompt lengths, verify that the cohort still runs
    without error and produces deterministic output equal to a
    direct-driven mlx-lm reference with matching left-padding.

    Weaker than solo parity (see module docstring) but catches any bug
    where Silica's left-padding construction drifts from what mlx-lm
    expects."""
    adapter, kv = Qwen3Adapter.from_hf_repo(REPO)
    tokenizer = adapter.tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    prompts = ["Hi", "The capital of France is"]
    ids = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(x) for x in ids)
    left_padding = [max_len - len(x) for x in ids]
    pad_id = 0
    padded = [[pad_id] * (max_len - len(x)) + list(x) for x in ids]

    # Silica path
    events = list(Engine(adapter, kv).generate_batch(prompts, params))
    silica: dict[int, list[int]] = {0: [], 1: []}
    for e in events:
        if e.kind == "token" and e.token_id is not None:
            silica[e.req_index].append(e.token_id)

    # mlx-lm direct reference — matching left-padding.
    model, _ = load(REPO)  # type: ignore[misc]
    num_layers = len(model.layers)
    bkv = [BatchKVCache(left_padding=left_padding) for _ in range(num_layers)]
    tokens = mx.array(padded, dtype=mx.int32)
    logits = model(tokens, cache=bkv)
    mx.eval(logits)
    last = logits[:, -1, :]
    t0 = int(mx.argmax(last[0]).item())
    t1 = int(mx.argmax(last[1]).item())
    direct: dict[int, list[int]] = {0: [t0], 1: [t1]}
    for _ in range(MAX_TOKENS - 1):
        dec = mx.array([[t0], [t1]], dtype=mx.int32)
        logits = model(dec, cache=bkv)
        mx.eval(logits)
        last = logits[:, -1, :]
        t0 = int(mx.argmax(last[0]).item())
        t1 = int(mx.argmax(last[1]).item())
        direct[0].append(t0)
        direct[1].append(t1)

    assert silica[0] == direct[0]
    assert silica[1] == direct[1]
