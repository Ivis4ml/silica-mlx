"""P-5-A.3c acceptance gate — BlockTQ row-1 decode tok/s ≥ 0.85 ×
IdentityCodec row-1 decode tok/s on the ``qwen3-0.6b-prefix-hit-
decode-*`` bench rows.

Model target amendment: opening §7(d) originally named Qwen3.5-0.8B,
but that checkpoint is hybrid-DeltaNet (``has_recurrent_state=True``)
and ``ContinuousBatcher`` refuses ``RadixPrefixCache`` on recurrent
adapters (DeltaNet running accumulation cannot be sliced into
block-aligned prefix K/V — see ``docs/P3_DELTANET_SURVEY.md``
C-open-3). Amended to Qwen3-0.6B (plain GLOBAL attention, 28 layers,
head_dim=128, no recurrent state). The BlockTQ-vs-identity ratio
is head-dim / layer-count independent, so the 0.85× threshold
carries over.

Opening §7(d) threshold: the gate asserts that compressed-codec
prefix-hit admission + steady-state decode is **no worse than 15 %
below** the fp16 baseline. Tighter bars are v0.2 work; 0.85× reflects
the post-hoc expectation that an fp32 matmul for the Haar rotation +
sub-byte unpack + fancy-indexed centroid lookup on Apple Silicon's
unified-memory path lands close to the fp16 reference. A larger
regression (e.g. 0.5× from an accidental NumPy round-trip or a
Metal-kernel-compile leak) would fail loudly here.

Dual gate, mirroring ``test_q010_ratio_below_threshold_on_five_runs``:

- ``Qwen/Qwen3.5-0.8B`` present in the local HF cache (cheap to
  populate by running the ``qwen3.5-0.8b-b1-parity`` bench row
  once).
- ``SILICA_PREFIX_HIT_DECODE_TIMING=1`` in the environment, OR
  ``pytest -m prefix_hit_decode_timing`` explicit marker selection
  (the ``conftest.py::pytest_collection_modifyitems`` hook skips by
  default; see that file's ``_TIMING_GATES`` table).

Measurement protocol — alternating pairs with warmup:

1. **Warmup pair** (results discarded): run fp16 then BlockTQ once.
   Covers Metal-kernel-compile costs paid on first invocation of
   each codec's hot path, plus MLX allocator warmup — without this
   the first measured BlockTQ run would carry ~200-500 ms of cold
   Metal-compile time that would not be part of steady-state decode.
2. **Measurement pairs × 3** alternating ``fp16 → BlockTQ → fp16 →
   BlockTQ ...``. Alternating guards against monotonic drift: if
   some background system load were ramping up, a back-to-back
   run-all-fp16-then-all-BlockTQ layout would attribute the drift
   to BlockTQ; alternating distributes it.
3. Median comparison: ``ratio = median(blocktq_decode_tok_s) /
   median(fp16_decode_tok_s)``. Asserting on the median rather than
   mean absorbs a single outlier without requiring a larger sample
   (3 samples × 2 codecs = 6 scenario runs; each scenario reloads
   the adapter, so total test time is O(adapter-load × 8) ≈ 40-60 s
   on this hardware).

Failure message includes per-sample numbers so a regression can be
triaged without re-running: "BlockTQ [46.3, 47.1, 45.8] median 46.3
vs fp16 [58.0, 59.1, 57.9] median 58.0 → ratio 0.799× < 0.85×".
"""

from __future__ import annotations

import statistics
from typing import Any

import pytest

from silica.bench.runner import BenchRunner
from silica.bench.scenario import ScenarioResult, hf_cache_path_for_repo
from silica.bench.scenarios import get_scenario

_REPO = "Qwen/Qwen3-0.6B"
_FP16_SCENARIO_ID = "qwen3-0.6b-prefix-hit-decode-fp16"
_BLOCK_TQ_SCENARIO_ID = "qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4"

_SKIP_HF_CACHE = not hf_cache_path_for_repo(_REPO).exists()
_SKIP_HF_CACHE_REASON = (
    f"{_REPO} not present in the local HF cache. Run any "
    f"cache-only Qwen3-0.6B bench row (e.g. qwen3-0.6b-smoke) once "
    f"to populate."
)

# Measurement protocol knobs.
_N_WARMUP_PAIRS = 1  # runs discarded — covers Metal compile + cache warmup
_N_MEASURED_PAIRS = 3  # alternating fp16 / BlockTQ samples per codec
_RATIO_THRESHOLD = 0.85  # BlockTQ median / fp16 median >= this


@pytest.fixture(scope="module")
def _adapter_engine() -> Any:
    """Load Qwen3.5-0.8B once per test-module invocation and reuse
    across all 8 scenario runs. Loading takes O(seconds) on cold
    cache; reloading per scenario would dominate total test time and
    add variance unrelated to the decode-speed claim under test.

    Shared across runs because the per-scenario state the runner
    actually cares about — a fresh ``RadixPrefixCache`` + codec-
    installed store — is constructed inside ``_run_one`` per call.
    The engine's ``MetricsRegistry`` is reused, but this oracle
    does not consume ``snap.decode_tok_s`` (metadata's
    ``row1_decode_tok_s`` is the headline number per A.3b H-1),
    so accumulation across runs is harmless.
    """
    from silica.engine import Engine
    from silica.models.factory import adapter_for_repo

    adapter, kv = adapter_for_repo(_REPO)
    engine = Engine(adapter, kv)
    return adapter, engine


def _run_scenario(runner: BenchRunner, scenario_id: str) -> ScenarioResult:
    scenario = get_scenario(scenario_id)
    return runner._run_one(scenario)


def _assert_decode_tok_s(
    result: ScenarioResult, scenario_id: str
) -> float:
    """Extract decode_tok_s from a successful scenario result or fail
    the whole test with a diagnostic message if the oracle failed."""
    assert result.status == "ok", (
        f"{scenario_id} failed: status={result.status} reason={result.reason!r}; "
        f"full result metadata={result.metadata}"
    )
    assert result.decode_tok_s is not None, (
        f"{scenario_id} produced no decode_tok_s — A.3b field-promotion "
        f"regression? metadata={result.metadata}"
    )
    return float(result.decode_tok_s)


@pytest.mark.prefix_hit_decode_timing
@pytest.mark.skipif(_SKIP_HF_CACHE, reason=_SKIP_HF_CACHE_REASON)
def test_blocktq_decode_tok_s_within_85_percent_of_fp16() -> None:
    """P-5-A.3 Acceptance (d): row 1's steady-state decode tok/s
    under block_tq_b64_b4 is at least 0.85× the IdentityCodec
    baseline on the same prefix-hit workload.

    Shared adapter + engine across runs; fresh prefix cache per run.
    See module docstring for measurement protocol rationale.
    """
    from silica.engine import Engine
    from silica.models.factory import adapter_for_repo

    # Load once — subsequent scenario runs reuse this adapter/engine
    # via the engine_factory closure below. Adapter load is the
    # dominant cost of this test; re-loading per scenario would make
    # the test unusable for acceptance gating.
    adapter, kv = adapter_for_repo(_REPO)
    engine = Engine(adapter, kv)

    def engine_factory(_scenario: Any) -> tuple[Any, Any]:
        return adapter, engine

    # Out-path None → runner does not write JSONL. Acceptance is the
    # ratio comparison, not a bench-report row.
    runner = BenchRunner(engine_factory=engine_factory, out_path=None)

    # 1. Warmup pairs — Metal-kernel-compile + MLX allocator warmup.
    for _ in range(_N_WARMUP_PAIRS):
        _run_scenario(runner, _FP16_SCENARIO_ID)
        _run_scenario(runner, _BLOCK_TQ_SCENARIO_ID)

    # 2. Measurement pairs alternating.
    fp16_samples: list[float] = []
    blocktq_samples: list[float] = []
    for _ in range(_N_MEASURED_PAIRS):
        fp16_result = _run_scenario(runner, _FP16_SCENARIO_ID)
        fp16_samples.append(
            _assert_decode_tok_s(fp16_result, _FP16_SCENARIO_ID)
        )
        blocktq_result = _run_scenario(runner, _BLOCK_TQ_SCENARIO_ID)
        blocktq_samples.append(
            _assert_decode_tok_s(blocktq_result, _BLOCK_TQ_SCENARIO_ID)
        )

    fp16_median = statistics.median(fp16_samples)
    blocktq_median = statistics.median(blocktq_samples)
    assert fp16_median > 0.0, (
        f"fp16 median decode_tok_s is non-positive — measurement "
        f"protocol is broken. samples={fp16_samples}"
    )
    ratio = blocktq_median / fp16_median

    diagnostic = (
        f"BlockTQ samples: {[f'{x:.1f}' for x in blocktq_samples]} "
        f"median {blocktq_median:.1f} tok/s; "
        f"fp16 samples: {[f'{x:.1f}' for x in fp16_samples]} "
        f"median {fp16_median:.1f} tok/s; "
        f"ratio {ratio:.3f}× (gate {_RATIO_THRESHOLD:.2f}×)"
    )
    # Print to stdout so -s runs surface the numbers even on pass;
    # acceptance runs in the regression sweep log the actual ratio
    # rather than just "passed" for trend-tracking.
    print(f"\n{diagnostic}")

    assert ratio >= _RATIO_THRESHOLD, (
        f"P-5-A.3 Acceptance (d) failed: {diagnostic}. "
        f"Threshold set by opening §7(d). A ratio this low suggests "
        f"the codec's decode hot path regressed beyond acceptable "
        f"MLX-native overhead — check for accidental NumPy round-"
        f"trips, Metal-kernel-compile leaks across runs, or a "
        f"change to the fp32-rotation / argmin centroid lookup "
        f"implementation in silica.vq.block_tq."
    )
