"""P-5-A.3c + P-5-B.3 acceptance gate — compressed-codec row-1 decode
tok/s ≥ 0.85 × IdentityCodec row-1 decode tok/s on the
``qwen3-0.6b-prefix-hit-decode-*`` bench rows.

Two codec arms share the same measurement protocol:

- ``block_tq_b64_b4`` (landed P-5-A.3c): BlockTurboQuantMSE
  production recommendation — Haar rotation + sub-byte unpack +
  fancy-indexed centroid lookup on fp32.
- ``ext_rabitq_b4`` (added P-5-B.3): ExtRaBitQ multi-bit — integer-
  grid codebook lookup + per-vector scale multiply + re-
  normalization + inverse rotation. Symmetric K+V (``v_supported=
  True`` per B.2b registry). ``rabitq_b1`` is deliberately not
  gated: it is K-only (``v_supported=False``) so the symmetric
  kv_codec= shorthand cannot install it, and its hypercube MSE is
  worse than BlockTQ at matching bit budget; adding a decode-speed
  gate on a codec that doesn't make the production ladder is
  signal-free.

Model target amendment: opening §7(d) originally named Qwen3.5-0.8B,
but that checkpoint is hybrid-DeltaNet (``has_recurrent_state=True``)
and ``ContinuousBatcher`` refuses ``RadixPrefixCache`` on recurrent
adapters (DeltaNet running accumulation cannot be sliced into
block-aligned prefix K/V — see ``docs/P3_DELTANET_SURVEY.md``
C-open-3). Amended to Qwen3-0.6B (plain GLOBAL attention, 28 layers,
head_dim=128, no recurrent state). The codec-vs-identity ratio
is head-dim / layer-count independent, so the 0.85× threshold
carries over across both codec arms.

Opening §7(d) threshold: the gate asserts that compressed-codec
prefix-hit admission + steady-state decode is **no worse than 15 %
below** the fp16 baseline. Tighter bars are v0.2 work; 0.85× reflects
the post-hoc expectation that an fp32 matmul for the Haar rotation +
sub-byte unpack + fancy-indexed codebook lookup on Apple Silicon's
unified-memory path lands close to the fp16 reference. A larger
regression (e.g. 0.5× from an accidental NumPy round-trip or a
Metal-kernel-compile leak) would fail loudly here.

Dual gate, mirroring ``test_q010_ratio_below_threshold_on_five_runs``:

- ``Qwen/Qwen3-0.6B`` present in the local HF cache (cheap to
  populate by running any cache-only Qwen3-0.6B bench row, e.g.
  ``qwen3-0.6b-smoke``, once).
- ``SILICA_PREFIX_HIT_DECODE_TIMING=1`` in the environment, OR
  ``pytest -m prefix_hit_decode_timing`` explicit marker selection
  (the ``conftest.py::pytest_collection_modifyitems`` hook skips by
  default; see that file's ``_TIMING_GATES`` table).

Measurement protocol — alternating pairs with warmup, per codec arm:

1. **Warmup pair** (results discarded): run fp16 then compressed-codec
   once. Covers Metal-kernel-compile costs paid on first invocation
   of each codec's hot path, plus MLX allocator warmup — without this
   the first measured compressed-codec run would carry ~200-500 ms
   of cold Metal-compile time that would not be part of steady-state
   decode.
2. **Measurement pairs × 3** alternating ``fp16 → codec → fp16 →
   codec ...``. Alternating guards against monotonic drift: if
   some background system load were ramping up, a back-to-back
   run-all-fp16-then-all-codec layout would attribute the drift
   to the codec; alternating distributes it.
3. Median comparison: ``ratio = median(codec_decode_tok_s) /
   median(fp16_decode_tok_s)``. Asserting on the median rather than
   mean absorbs a single outlier without requiring a larger sample.

Adapter + engine are loaded **once per module** via the
``_adapter_engine`` fixture and shared between both codec-arm tests.
Under the default ``_TIMING_GATES`` skip, the fixture is never
requested, so the cost is only paid when the gate runs. When both
tests run, total cost is ``adapter-load + (8 scenario runs per arm)
× 2 arms ≈ adapter + 16 × short-scenario-latency`` — much cheaper
than loading twice.

Failure message includes per-sample numbers so a regression can be
triaged without re-running: "codec [46.3, 47.1, 45.8] median 46.3
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
_EXT_RABITQ_SCENARIO_ID = "qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4"

_SKIP_HF_CACHE = not hf_cache_path_for_repo(_REPO).exists()
_SKIP_HF_CACHE_REASON = (
    f"{_REPO} not present in the local HF cache. Run any "
    f"cache-only Qwen3-0.6B bench row (e.g. qwen3-0.6b-smoke) once "
    f"to populate."
)

# Measurement protocol knobs.
_N_WARMUP_PAIRS = 1  # runs discarded — covers Metal compile + cache warmup
_N_MEASURED_PAIRS = 3  # alternating fp16 / codec samples per arm
_RATIO_THRESHOLD = 0.85  # codec median / fp16 median >= this


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


@pytest.fixture(scope="module")
def _adapter_engine() -> tuple[Any, Any]:
    """Load adapter + engine once per module; both codec-arm tests
    share the loaded weights. Fixture is only instantiated when a test
    actually runs (skipif + marker-filter gate happens before fixture
    request), so the adapter load cost is not paid on the default
    test run where the gate is skipped by ``conftest.py``."""
    from silica.engine import Engine
    from silica.models.factory import adapter_for_repo

    adapter, kv = adapter_for_repo(_REPO)
    engine = Engine(adapter, kv)
    return adapter, engine


def _measure_codec_ratio(
    adapter: Any,
    engine: Any,
    codec_scenario_id: str,
    codec_label: str,
) -> tuple[list[float], list[float], float]:
    """Run warmup + measurement pairs for one codec arm against the
    fp16 baseline. Returns ``(fp16_samples, codec_samples, ratio)``."""

    def engine_factory(_scenario: Any) -> tuple[Any, Any]:
        return adapter, engine

    # Out-path None → runner does not write JSONL. Acceptance is the
    # ratio comparison, not a bench-report row.
    runner = BenchRunner(engine_factory=engine_factory, out_path=None)

    # 1. Warmup pairs — Metal-kernel-compile + MLX allocator warmup.
    for _ in range(_N_WARMUP_PAIRS):
        _run_scenario(runner, _FP16_SCENARIO_ID)
        _run_scenario(runner, codec_scenario_id)

    # 2. Measurement pairs alternating.
    fp16_samples: list[float] = []
    codec_samples: list[float] = []
    for _ in range(_N_MEASURED_PAIRS):
        fp16_result = _run_scenario(runner, _FP16_SCENARIO_ID)
        fp16_samples.append(
            _assert_decode_tok_s(fp16_result, _FP16_SCENARIO_ID)
        )
        codec_result = _run_scenario(runner, codec_scenario_id)
        codec_samples.append(
            _assert_decode_tok_s(codec_result, codec_scenario_id)
        )

    fp16_median = statistics.median(fp16_samples)
    codec_median = statistics.median(codec_samples)
    assert fp16_median > 0.0, (
        f"fp16 median decode_tok_s is non-positive — measurement "
        f"protocol is broken. samples={fp16_samples}"
    )
    ratio = codec_median / fp16_median

    diagnostic = (
        f"{codec_label} samples: {[f'{x:.1f}' for x in codec_samples]} "
        f"median {codec_median:.1f} tok/s; "
        f"fp16 samples: {[f'{x:.1f}' for x in fp16_samples]} "
        f"median {fp16_median:.1f} tok/s; "
        f"ratio {ratio:.3f}× (gate {_RATIO_THRESHOLD:.2f}×)"
    )
    # Print so -s runs surface the numbers even on pass; acceptance
    # runs in the regression sweep log the actual ratio rather than
    # just "passed" for trend-tracking.
    print(f"\n{diagnostic}")

    return fp16_samples, codec_samples, ratio


@pytest.mark.prefix_hit_decode_timing
@pytest.mark.skipif(_SKIP_HF_CACHE, reason=_SKIP_HF_CACHE_REASON)
def test_blocktq_decode_tok_s_within_85_percent_of_fp16(
    _adapter_engine: tuple[Any, Any],
) -> None:
    """P-5-A.3 Acceptance (d): row 1's steady-state decode tok/s
    under block_tq_b64_b4 is at least 0.85× the IdentityCodec
    baseline on the same prefix-hit workload.

    Shared adapter + engine across runs; fresh prefix cache per run.
    See module docstring for measurement protocol rationale.
    """
    adapter, engine = _adapter_engine
    fp16_samples, blocktq_samples, ratio = _measure_codec_ratio(
        adapter, engine, _BLOCK_TQ_SCENARIO_ID, codec_label="BlockTQ"
    )

    assert ratio >= _RATIO_THRESHOLD, (
        f"P-5-A.3 Acceptance (d) failed: BlockTQ samples "
        f"{[f'{x:.1f}' for x in blocktq_samples]} "
        f"vs fp16 samples {[f'{x:.1f}' for x in fp16_samples]} "
        f"→ ratio {ratio:.3f}× < {_RATIO_THRESHOLD:.2f}×. "
        f"Threshold set by opening §7(d). A ratio this low suggests "
        f"the codec's decode hot path regressed beyond acceptable "
        f"MLX-native overhead — check for accidental NumPy round-"
        f"trips, Metal-kernel-compile leaks across runs, or a "
        f"change to the fp32-rotation / argmin centroid lookup "
        f"implementation in silica.vq.block_tq."
    )


@pytest.mark.prefix_hit_decode_timing
@pytest.mark.skipif(_SKIP_HF_CACHE, reason=_SKIP_HF_CACHE_REASON)
def test_ext_rabitq_b4_decode_tok_s_within_85_percent_of_fp16(
    _adapter_engine: tuple[Any, Any],
) -> None:
    """P-5-B.3 Acceptance: row 1's steady-state decode tok/s under
    ext_rabitq_b4 is at least 0.85× the IdentityCodec baseline on the
    same prefix-hit workload. Same measurement protocol and threshold
    as the BlockTQ arm; ExtRaBitQ's hot path differs (integer-grid
    codebook lookup + per-vector scale multiply + re-normalization +
    inverse rotation) but the 0.85× floor is head-dim / algorithm
    independent per opening §7(d).
    """
    adapter, engine = _adapter_engine
    fp16_samples, ext_rabitq_samples, ratio = _measure_codec_ratio(
        adapter,
        engine,
        _EXT_RABITQ_SCENARIO_ID,
        codec_label="ExtRaBitQ",
    )

    assert ratio >= _RATIO_THRESHOLD, (
        f"P-5-B.3 Acceptance failed: ExtRaBitQ samples "
        f"{[f'{x:.1f}' for x in ext_rabitq_samples]} "
        f"vs fp16 samples {[f'{x:.1f}' for x in fp16_samples]} "
        f"→ ratio {ratio:.3f}× < {_RATIO_THRESHOLD:.2f}×. "
        f"Threshold set by opening §7(d). A ratio this low suggests "
        f"the codec's decode hot path regressed — check "
        f"silica.vq.rabitq.rabitq_ext for an accidental NumPy "
        f"round-trip, a Metal-kernel-compile leak across runs, or a "
        f"change to the codebook-lookup / re-normalization / inverse-"
        f"rotation implementation."
    )
