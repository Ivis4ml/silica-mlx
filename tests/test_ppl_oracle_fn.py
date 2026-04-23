"""Unit tests for :func:`silica.bench.oracles.ppl_oracle`.

Validates the structural / type / metadata-forwarding contract without
touching the runner, adapter, or engine surfaces. ``ppl_oracle`` is a
pure function over ``(Scenario, collected, context)``; these tests
synthesize both inputs directly.
"""

from __future__ import annotations

import math
from typing import Any

from silica.bench.oracles import ORACLES, ppl_oracle
from silica.bench.scenario import OracleKind, Scenario, Workload


def _make_scenario(
    *, min_scored_tokens: int = 1, kv_codec: str | None = None
) -> Scenario:
    workload = Workload(
        name="wikitext-ppl-test",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=kv_codec is not None,
        temperature=0.0,
        top_p=1.0,
        kv_codec=kv_codec,
    )
    return Scenario(
        id="test-wikitext-ppl",
        repo="dummy/repo",
        workload=workload,
        oracle=OracleKind.PPL,
        oracle_config={
            "min_scored_tokens": min_scored_tokens,
            "chunk_size": 16,
            "max_tokens": 64,
            "wikitext_path": "/tmp/irrelevant.txt",
        },
    )


def _valid_collected(
    *, nll_sum: float = 20.0, n_tokens: int = 10, ppl: float | None = None
) -> dict[str, Any]:
    if ppl is None:
        ppl = math.exp(nll_sum / n_tokens)
    return {
        "nll_sum": nll_sum,
        "n_tokens": n_tokens,
        "ppl": ppl,
    }


def _valid_context(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chunk_size": 16,
        "max_tokens": 64,
        "wikitext_path": "/tmp/irrelevant.txt",
        "kv_codec": None,
    }
    base.update(kwargs)
    return base


# =============================================================================
# Registry + basic pass-through
# =============================================================================


class TestRegistry:
    def test_ppl_oracle_registered(self) -> None:
        assert OracleKind.PPL in ORACLES
        assert ORACLES[OracleKind.PPL] is ppl_oracle


class TestAcceptsValidInput:
    def test_accepts_well_shaped_collected(self) -> None:
        scenario = _make_scenario()
        ok, reason, metadata = ppl_oracle(
            scenario, _valid_collected(), _valid_context()
        )
        assert ok is True
        assert reason is None
        assert metadata["nll_sum"] == 20.0
        assert metadata["n_tokens"] == 10
        assert metadata["ppl"] == math.exp(2.0)

    def test_metadata_forwards_context_fields(self) -> None:
        scenario = _make_scenario()
        ctx = _valid_context(
            chunk_size=256, max_tokens=512, kv_codec="block_tq_b64_b4"
        )
        _, _, metadata = ppl_oracle(scenario, _valid_collected(), ctx)
        assert metadata["chunk_size"] == 256
        assert metadata["max_tokens"] == 512
        assert metadata["kv_codec"] == "block_tq_b64_b4"
        assert metadata["wikitext_path"] == "/tmp/irrelevant.txt"

    def test_collected_does_not_override_delta_keys_from_context(
        self,
    ) -> None:
        """Context's ``chunk_size`` etc. forward via ``setdefault``;
        collected's core numbers win if the runner ever added them to
        both dicts. Regression-lock the priority."""
        scenario = _make_scenario()
        # Deliberately pre-populate context with a value that would
        # differ from collected's ``nll_sum``; the collected number
        # must still land in metadata.
        ctx = _valid_context()
        ctx["nll_sum"] = -1.0
        _, _, metadata = ppl_oracle(scenario, _valid_collected(), ctx)
        assert metadata["nll_sum"] == 20.0


# =============================================================================
# Rejection paths
# =============================================================================


class TestRejectsMalformedCollected:
    def test_rejects_non_dict_collected(self) -> None:
        scenario = _make_scenario()
        ok, reason, _ = ppl_oracle(scenario, 42.0, _valid_context())
        assert ok is False
        assert reason == "ppl_collected_shape_mismatch"

    def test_rejects_missing_nll_sum(self) -> None:
        scenario = _make_scenario()
        bad = {"n_tokens": 10, "ppl": 1.0}
        ok, reason, _ = ppl_oracle(scenario, bad, _valid_context())
        assert ok is False
        assert reason is not None
        assert "nll_sum" in reason

    def test_rejects_missing_n_tokens(self) -> None:
        scenario = _make_scenario()
        bad = {"nll_sum": 20.0, "ppl": 1.0}
        ok, reason, _ = ppl_oracle(scenario, bad, _valid_context())
        assert ok is False
        assert reason is not None
        assert "n_tokens" in reason

    def test_rejects_missing_ppl(self) -> None:
        scenario = _make_scenario()
        bad = {"nll_sum": 20.0, "n_tokens": 10}
        ok, reason, _ = ppl_oracle(scenario, bad, _valid_context())
        assert ok is False
        assert reason is not None
        assert "ppl" in reason

    def test_rejects_non_castable_fields(self) -> None:
        scenario = _make_scenario()
        bad = {"nll_sum": "not_a_number", "n_tokens": 10, "ppl": 1.0}
        ok, reason, _ = ppl_oracle(scenario, bad, _valid_context())
        assert ok is False
        assert reason is not None
        assert reason.startswith("ppl_collected_field_not_castable")


class TestRejectsNumericViolations:
    def test_rejects_negative_n_tokens(self) -> None:
        scenario = _make_scenario()
        ok, reason, _ = ppl_oracle(
            scenario, _valid_collected(n_tokens=-1), _valid_context()
        )
        assert ok is False
        assert reason == "ppl_n_tokens_negative"

    def test_rejects_n_tokens_below_min_scored_tokens(self) -> None:
        scenario = _make_scenario(min_scored_tokens=256)
        ok, reason, metadata = ppl_oracle(
            scenario, _valid_collected(n_tokens=100), _valid_context()
        )
        assert ok is False
        assert reason == "ppl_n_tokens_below_min_scored_tokens"
        assert metadata["n_tokens"] == 100
        assert metadata["min_scored_tokens"] == 256

    def test_accepts_n_tokens_equal_min_scored_tokens(self) -> None:
        scenario = _make_scenario(min_scored_tokens=10)
        ok, _, _ = ppl_oracle(
            scenario, _valid_collected(n_tokens=10), _valid_context()
        )
        assert ok is True

    def test_rejects_non_finite_ppl(self) -> None:
        scenario = _make_scenario()
        for bad_ppl in (float("inf"), float("nan")):
            ok, reason, _ = ppl_oracle(
                scenario,
                _valid_collected(ppl=bad_ppl),
                _valid_context(),
            )
            assert ok is False, f"ppl={bad_ppl} should be rejected"
            assert reason == "ppl_not_finite_or_negative"

    def test_rejects_negative_ppl(self) -> None:
        scenario = _make_scenario()
        ok, reason, _ = ppl_oracle(
            scenario, _valid_collected(ppl=-1.0), _valid_context()
        )
        assert ok is False
        assert reason == "ppl_not_finite_or_negative"


# =============================================================================
# Delta-PPL computation when ppl_fp16 in context
# =============================================================================


class TestDeltaPplComputation:
    def test_no_delta_when_context_lacks_ppl_fp16(self) -> None:
        scenario = _make_scenario()
        _, _, metadata = ppl_oracle(
            scenario, _valid_collected(), _valid_context()
        )
        assert metadata["delta_ppl"] is None
        assert metadata["delta_ppl_pct"] is None

    def test_delta_ppl_populated_when_fp16_supplied(self) -> None:
        scenario = _make_scenario()
        collected = _valid_collected(nll_sum=20.0, n_tokens=10)
        ctx = _valid_context(ppl_fp16=math.exp(1.9))
        _, _, metadata = ppl_oracle(scenario, collected, ctx)
        assert metadata["delta_ppl"] is not None
        assert metadata["delta_ppl_pct"] is not None
        # ppl = exp(2.0); ppl_fp16 = exp(1.9); delta = exp(2) - exp(1.9)
        expected_delta = math.exp(2.0) - math.exp(1.9)
        assert metadata["delta_ppl"] == expected_delta

    def test_delta_zero_when_codec_matches_baseline(self) -> None:
        scenario = _make_scenario()
        ppl = math.exp(2.0)
        collected = _valid_collected(ppl=ppl)
        ctx = _valid_context(ppl_fp16=ppl)
        _, _, metadata = ppl_oracle(scenario, collected, ctx)
        assert metadata["delta_ppl"] == 0.0
        assert metadata["delta_ppl_pct"] == 0.0

    def test_non_finite_ppl_fp16_ignored(self) -> None:
        scenario = _make_scenario()
        ctx = _valid_context(ppl_fp16=float("inf"))
        _, _, metadata = ppl_oracle(
            scenario, _valid_collected(), ctx
        )
        assert metadata["delta_ppl"] is None

    def test_zero_ppl_fp16_ignored(self) -> None:
        """ppl_fp16 == 0 would make the pct computation divide by
        zero; the oracle treats it as "no baseline"."""
        scenario = _make_scenario()
        ctx = _valid_context(ppl_fp16=0.0)
        _, _, metadata = ppl_oracle(
            scenario, _valid_collected(), ctx
        )
        assert metadata["delta_ppl"] is None


# =============================================================================
# Non-dict context
# =============================================================================


class TestNonDictContext:
    def test_non_dict_context_does_not_crash(self) -> None:
        """Structural validation on ``collected`` still applies; a
        non-dict context yields empty forwarded metadata but no
        crash."""
        scenario = _make_scenario()
        ok, _, metadata = ppl_oracle(
            scenario, _valid_collected(), "not-a-dict"
        )
        assert ok is True
        assert metadata["nll_sum"] == 20.0
        # context-forwarded keys are absent; delta_ppl falls back to None.
        assert metadata["delta_ppl"] is None
