"""P-5-C.2 step 3b integration: run the fp16 wikitext PPL row on
real Qwen3-0.6B weights end-to-end.

Gates (triple-gated, matching the existing prefix-hit-decode and
P3 model-smoke pattern):

  1. Qwen3-0.6B HF cache present — weak gate, same as the cache-only
     bench rows. Absent → skipped, no model load.
  2. WikiText-2 cache file present at the scenario's configured
     ``wikitext_path``. Absent → skipped; populate once via
     ``python scripts/prepare_wikitext2_cache.py``.
  3. ``SILICA_SKIP_MODEL_TESTS`` env var respected — when set to
     ``"1"`` every model-loading test is skipped regardless of
     cache presence (dev-loop knob for times when a real forward
     pass is too expensive).

Scope (per user guidance):

- Only ``qwen3-0.6b-wikitext-ppl-fp16`` is exercised. The codec
  rows' encode/decode path is already regression-locked by the
  C.2 counting-codec harness and the A.3c / B.3 prefix-hit
  decode tests — a real-model PPL run for every codec would
  multiply CI load without adding orthogonal coverage.
- The assertion set verifies the full chain works:
  real tokenizer → loaded weights → bench runner → wikitext
  gate → oracle structural validation → ``ScenarioResult``
  metadata population. If any link breaks, exactly one row's
  status becomes ``"failed"`` or ``"skipped"`` and the reason
  field says why.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from silica.bench.runner import BenchRunner
from silica.bench.scenario import Scenario, hf_cache_path_for_repo
from silica.bench.scenarios import get_scenario

_REPO = "Qwen/Qwen3-0.6B"
_SCENARIO_ID = "qwen3-0.6b-wikitext-ppl-fp16"

_SKIP_HF_CACHE = not hf_cache_path_for_repo(_REPO).exists()
_SKIP_HF_CACHE_REASON = (
    f"{_REPO} not present in the local HF cache. Run any cache-only "
    f"Qwen3-0.6B bench row (e.g. qwen3-0.6b-smoke) once to populate."
)

_SKIP_SILICA_MODEL_TESTS = bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_SKIP_SILICA_MODEL_TESTS_REASON = (
    "SILICA_SKIP_MODEL_TESTS is set; model-loading tests skipped"
)

# WikiText cache file presence is a scenario-carried path; resolve
# lazily so an absent path produces a skip, not an import-time error.
_SCENARIO = get_scenario(_SCENARIO_ID)
_WIKITEXT_PATH = Path(
    str(_SCENARIO.oracle_config["wikitext_path"])
)
_SKIP_WIKITEXT = not _WIKITEXT_PATH.is_file()
_SKIP_WIKITEXT_REASON = (
    f"WikiText cache file missing at {_WIKITEXT_PATH}. Populate "
    f"once with: python scripts/prepare_wikitext2_cache.py"
)


@pytest.mark.skipif(
    _SKIP_SILICA_MODEL_TESTS, reason=_SKIP_SILICA_MODEL_TESTS_REASON
)
@pytest.mark.skipif(_SKIP_HF_CACHE, reason=_SKIP_HF_CACHE_REASON)
@pytest.mark.skipif(_SKIP_WIKITEXT, reason=_SKIP_WIKITEXT_REASON)
def test_qwen3_0_6b_wikitext_ppl_fp16_end_to_end() -> None:
    """Real-model PPL row: loaded Qwen3-0.6B tokenizer + weights +
    bench runner + WikiText gate + oracle structural validation.

    Asserts the full chain:

    - ``status == "ok"`` (oracle structural validation passed).
    - ``metadata["ppl"]`` is finite and strictly positive (a sane
      PPL for a real language model on natural text).
    - ``metadata["n_tokens"] >= min_scored_tokens`` (the oracle's
      floor was satisfied by the real tokenized input).
    - ``total_tokens == n_tokens`` (runner-side total-token
      counter matches the oracle's scored-token count; any drift
      here would indicate the dispatch branch forgot to propagate
      one of the two numbers).
    """
    import math

    runner = BenchRunner()
    result = runner._run_one(_SCENARIO, seed=0)

    assert result.status == "ok", (
        f"{_SCENARIO_ID}: status={result.status} "
        f"reason={result.reason!r}; metadata={result.metadata}"
    )

    ppl = result.metadata.get("ppl")
    assert isinstance(ppl, float), (
        f"metadata['ppl'] missing or wrong type: {ppl!r}"
    )
    assert math.isfinite(ppl), f"ppl not finite: {ppl}"
    assert ppl > 0.0, f"ppl not positive: {ppl}"

    n_tokens = result.metadata.get("n_tokens")
    assert isinstance(n_tokens, int), (
        f"metadata['n_tokens'] missing or wrong type: {n_tokens!r}"
    )
    min_scored = int(_SCENARIO.oracle_config["min_scored_tokens"])
    assert n_tokens >= min_scored, (
        f"n_tokens={n_tokens} below oracle_config.min_scored_tokens="
        f"{min_scored}; tokenized text was too short for the "
        f"configured chunk_size + max_tokens"
    )

    assert result.total_tokens == n_tokens, (
        f"total_tokens={result.total_tokens} does not match "
        f"metadata.n_tokens={n_tokens}; dispatch-branch drift in "
        f"BenchRunner.run for OracleKind.PPL"
    )


@pytest.mark.skipif(_SKIP_HF_CACHE, reason=_SKIP_HF_CACHE_REASON)
def test_wikitext_missing_cache_skips_before_model_load() -> None:
    """Gate check: if the WikiText file is absent,
    ``_check_gates`` must return the
    ``wikitext_cache_missing:<path>`` skip reason **before** the
    engine factory is invoked (which would load ~600 MB of
    weights).

    The previous shape of this test used the default
    ``BenchRunner()`` and only asserted the skip reason. If a
    future edit moved ``_check_gates`` after ``engine_factory``
    but kept the same reason string on failure, that test would
    pass while silently triggering the model load — exactly the
    regression the docstring claims to prevent. The fix: inject a
    fail-fast ``engine_factory`` that raises on call and assert it
    is never invoked, so the pre-load invariant is actually
    pinned.

    Gated on the HF cache so the earlier ``cache_missing:``
    gate-branch does not shadow the ``wikitext_cache_missing:``
    branch this test targets. Not gated on
    ``SILICA_SKIP_MODEL_TESTS`` or ``_SKIP_WIKITEXT`` — no real
    model load ever happens (the fail-fast factory never runs)
    and the test's whole point is to assert the skip behaviour
    when the WikiText file is absent.
    """
    from dataclasses import replace

    bogus_path = "/tmp/silica_definitely_absent_wikitext_file.txt"
    scenario = replace(
        _SCENARIO,
        oracle_config={
            **_SCENARIO.oracle_config,
            "wikitext_path": bogus_path,
        },
    )

    factory_calls: list[None] = []

    def _fail_fast_factory(_scenario_arg: Scenario) -> Any:
        factory_calls.append(None)
        raise AssertionError(
            "engine_factory must not be called when the wikitext "
            "gate should skip the scenario pre-load"
        )

    runner = BenchRunner(engine_factory=_fail_fast_factory)
    result = runner._run_one(scenario, seed=0)
    assert result.status == "skipped"
    assert result.reason is not None
    assert result.reason.startswith("wikitext_cache_missing:")
    assert bogus_path in result.reason
    assert factory_calls == [], (
        "engine_factory was called despite the wikitext gate — "
        "gate moved to after the factory?"
    )
