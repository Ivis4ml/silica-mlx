"""Tests for ``silica.bench.oracles.warm_ttft_pair_oracle``.

P-6.0.5 sub-unit 6 scope: the oracle pure function exercised with
canned collector outputs so no ``Engine`` / adapter / model load is
needed. End-to-end runner integration through
``_run_warm_ttft_pair`` is covered by the catalog tests + manual
real-model run during the on-device pass.

The oracle's job:

1. Validate the collected dict shape (required keys present, types
   coercible, TTFTs and token counts strictly positive).
2. Read ``prefix_hit_tokens`` from the runner context (defaults to
   0 when context is missing or malformed).
3. Synthesise derived metadata fields:
   * ``warm_ttft_ms`` aliases ``prompt2_ttft_ms``;
   * ``compile_amortized_ms = prompt1_ttft_ms - prompt2_ttft_ms``.

This is a measurement reporter, not a gate — the oracle does not
enforce a target on absolute TTFT values; Decision Gate 1 (D-021
step 4) reads the warm number against any §6 acceptance framing
that lands.
"""

from __future__ import annotations

from typing import Any

from silica.bench.oracles import warm_ttft_pair_oracle
from silica.bench.scenario import OracleKind, Scenario, Workload


def _stub_scenario() -> Scenario:
    return Scenario(
        id="stub-warm-ttft-pair",
        repo="stub/stub",
        workload=Workload(
            name="stub-warm-ttft-pair",
            prompts=("first prompt", "second different prompt"),
            max_tokens=4,
            max_batch_size=1,
            prefix_cache=False,
        ),
        oracle=OracleKind.WARM_TTFT_PAIR,
    )


def _happy_collected(
    *,
    prompt1_ttft_ms: float = 1800.0,
    prompt2_ttft_ms: float = 25.0,
    prompt1_tokens: int = 28,
    prompt2_tokens: int = 27,
) -> dict[str, Any]:
    """Build a collected dict with realistic warm-TTFT values.

    Defaults model the expected production shape: prompt 1 is
    cold + compile (~1.8 s on 27B per the P-6.0 baseline anchor);
    prompt 2 is warm and ttft drops by ~70× because the kernel
    cache is now populated.
    """
    return {
        "prompt1_ttft_ms": prompt1_ttft_ms,
        "prompt2_ttft_ms": prompt2_ttft_ms,
        "prompt1_tokens": prompt1_tokens,
        "prompt2_tokens": prompt2_tokens,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_reports_warm_ttft() -> None:
    collected = _happy_collected()
    context = {"prefix_hit_tokens": 0}
    ok, reason, md = warm_ttft_pair_oracle(_stub_scenario(), collected, context)
    assert ok is True
    assert reason is None
    assert md["prompt1_ttft_ms"] == 1800.0
    assert md["prompt2_ttft_ms"] == 25.0
    assert md["prompt1_tokens"] == 28
    assert md["prompt2_tokens"] == 27


def test_warm_ttft_ms_aliases_prompt2_ttft_ms() -> None:
    """``warm_ttft_ms`` must be a literal alias for ``prompt2_ttft_ms``
    so downstream consumers (and the runner's ``ScenarioResult.ttft_ms``
    promotion) can read either name interchangeably."""
    collected = _happy_collected(prompt2_ttft_ms=37.5)
    _, _, md = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert md["warm_ttft_ms"] == md["prompt2_ttft_ms"] == 37.5


def test_compile_amortized_ms_is_p1_minus_p2() -> None:
    collected = _happy_collected(prompt1_ttft_ms=1500.0, prompt2_ttft_ms=20.0)
    _, _, md = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert md["compile_amortized_ms"] == 1480.0


def test_compile_amortized_ms_can_be_negative() -> None:
    """If prompt 2 is somehow slower than prompt 1 (e.g. memory
    pressure, system noise), the oracle must still report — the
    diagnostic value is the signal that something is off, not a
    failure mode."""
    collected = _happy_collected(prompt1_ttft_ms=20.0, prompt2_ttft_ms=25.0)
    ok, reason, md = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert ok is True
    assert reason is None
    assert md["compile_amortized_ms"] == -5.0


# ---------------------------------------------------------------------------
# Context-driven prefix_hit_tokens handling
# ---------------------------------------------------------------------------


def test_prefix_hit_tokens_defaults_to_zero_when_context_missing_key() -> None:
    """Gate row pins ``prefix_cache=False`` and the runner records
    ``prefix_hit_tokens=0`` in context. If the oracle gets an empty
    context dict (collector never set the field), default to 0
    rather than failing — the gate row's contract is "structurally 0"."""
    _, _, md = warm_ttft_pair_oracle(_stub_scenario(), _happy_collected(), {})
    assert md["prefix_hit_tokens"] == 0


def test_prefix_hit_tokens_read_from_context_when_present() -> None:
    """When the collector populates the field (e.g. the deferred
    ``-shared-prefix`` variant), the oracle must surface the value
    rather than silently zeroing it."""
    collected = _happy_collected()
    context = {"prefix_hit_tokens": 42}
    _, _, md = warm_ttft_pair_oracle(_stub_scenario(), collected, context)
    assert md["prefix_hit_tokens"] == 42


def test_prefix_hit_tokens_defaults_to_zero_when_context_not_dict() -> None:
    _, _, md = warm_ttft_pair_oracle(_stub_scenario(), _happy_collected(), None)
    assert md["prefix_hit_tokens"] == 0


def test_prefix_hit_tokens_defaults_to_zero_when_value_not_int() -> None:
    """A non-int prefix_hit_tokens (e.g. string from a malformed
    collector) is coerced to 0 rather than propagated as garbage."""
    collected = _happy_collected()
    context = {"prefix_hit_tokens": "garbage"}
    _, _, md = warm_ttft_pair_oracle(_stub_scenario(), collected, context)
    assert md["prefix_hit_tokens"] == 0


# ---------------------------------------------------------------------------
# Failure paths — collector-shape validation
# ---------------------------------------------------------------------------


def test_rejects_non_dict_collected() -> None:
    ok, reason, _ = warm_ttft_pair_oracle(_stub_scenario(), [1, 2, 3], {})
    assert ok is False
    assert reason == "warm_ttft_pair_collected_shape_mismatch"


def test_rejects_missing_key_prompt1_ttft_ms() -> None:
    collected = _happy_collected()
    del collected["prompt1_ttft_ms"]
    ok, reason, md = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert ok is False
    assert reason == "warm_ttft_pair_collected_missing_key:prompt1_ttft_ms"
    assert "prompt2_ttft_ms" in md["keys_present"]


def test_rejects_missing_key_prompt2_tokens() -> None:
    collected = _happy_collected()
    del collected["prompt2_tokens"]
    ok, reason, _ = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert ok is False
    assert reason == "warm_ttft_pair_collected_missing_key:prompt2_tokens"


def test_rejects_uncoercible_ttft_value() -> None:
    """A non-numeric TTFT field (e.g. ``None`` from a runner bug) must
    surface a typed error reason, not a silent NaN downstream."""
    collected: dict[str, Any] = _happy_collected()
    collected["prompt1_ttft_ms"] = None
    ok, reason, _ = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert ok is False
    assert reason is not None
    assert reason.startswith("warm_ttft_pair_collected_type_error:")


# ---------------------------------------------------------------------------
# Failure paths — non-positive measurements
# ---------------------------------------------------------------------------


def test_rejects_non_positive_prompt1_ttft() -> None:
    collected = _happy_collected(prompt1_ttft_ms=0.0)
    ok, reason, _ = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert ok is False
    assert reason is not None
    assert reason.startswith("warm_ttft_pair_non_positive_ttft:")


def test_rejects_non_positive_prompt2_ttft() -> None:
    collected = _happy_collected(prompt2_ttft_ms=-1.0)
    ok, reason, _ = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert ok is False
    assert reason is not None
    assert reason.startswith("warm_ttft_pair_non_positive_ttft:")


def test_rejects_non_positive_prompt_tokens() -> None:
    collected = _happy_collected(prompt1_tokens=0)
    ok, reason, _ = warm_ttft_pair_oracle(_stub_scenario(), collected, {})
    assert ok is False
    assert reason is not None
    assert reason.startswith("warm_ttft_pair_non_positive_prompt_tokens:")


# ---------------------------------------------------------------------------
# Metadata shape lock-in — every field the JSONL row depends on
# ---------------------------------------------------------------------------


def test_metadata_carries_every_documented_field() -> None:
    """The OracleKind.WARM_TTFT_PAIR docstring lists seven metadata
    fields; pin all of them so a future schema drift fails here
    rather than silently breaking the JSONL consumer."""
    collected = _happy_collected()
    context = {"prefix_hit_tokens": 0}
    _, _, md = warm_ttft_pair_oracle(_stub_scenario(), collected, context)
    expected = {
        "prompt1_ttft_ms",
        "prompt2_ttft_ms",
        "warm_ttft_ms",
        "compile_amortized_ms",
        "prompt1_tokens",
        "prompt2_tokens",
        "prefix_hit_tokens",
    }
    assert set(md) == expected
