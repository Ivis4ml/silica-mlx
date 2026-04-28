"""P5.9 step 2(d) catalog + oracle-metadata tests for the sustained
4K / 8K context warm-decode probes.

Per D-021 step 2(d) (v1.7.14): four new scenarios extend the P-6.0
warm-decode shape to ~4K and ~8K total-context envelopes on the
two dense production targets (Qwen3.5-27B-4bit, Gemma4-31B-4bit).
The §6(4) RAM headroom gate's reference baseline currently rests on
inference from the 384-token P-6.0 baseline; these rows verify it
directly under sustained decode at materially longer contexts.

Tests in this file are **registration / structural** only — they do
not load any real-model weights. Real-hardware execution against
``SILICA_REAL_QWEN3_5_27B`` / ``SILICA_REAL_GEMMA4_31B`` is part of
P-6.0.5 measurement expansion (D-021 step 3); under-target outcomes
are then visible in the JSONL row via the metadata fields pinned
here.

Pinned contracts:

1. All four scenarios appear in ``BUILTIN_SCENARIOS`` keyed by the
   names listed in PLAN.md D-021 step 2(d).
2. Each scenario uses ``OracleKind.WARM_DECODE`` (no new oracle —
   per the user-stated v1.7.14 step 2(d) constraint to avoid scope
   drift on judgement logic).
3. Each scenario carries ``target_context_tokens`` and
   ``expected_total_context_floor`` in ``oracle_config`` so the
   oracle / runner can echo them into the JSONL row at runtime.
4. Each scenario is dual-gated on the matching ``SILICA_REAL_*``
   env var (mirrors the existing 27B / 31B warm-decode rows).
5. The 4K row carries a strictly higher prompt-character count than
   the v1.7.13 384-token row; the 8K row carries a strictly higher
   prompt-character count than the 4K row. (Token counts are not
   asserted at registration time per the user-stated constraint;
   character-length monotonicity is a tokenizer-independent sanity
   check.)
6. The WARM_DECODE oracle's metadata, when fed a runner-populated
   context with ``target_context_tokens`` /
   ``expected_total_context_floor`` / ``prompt_token_counts`` /
   ``max_tokens``, surfaces ``actual_total_context_per_row``,
   ``actual_total_context_min``, and ``reached_expected_floor``
   so under-target tokenizer drift is observable rather than
   silently absorbed.
7. Legacy WARM_DECODE rows (without ``target_context_tokens`` in
   their oracle_config) see no new metadata fields — byte-identical
   metadata shape for the cache-only / 27B / MoE rows landed at
   v1.7.13.
"""

from __future__ import annotations

import math

import pytest

from silica.bench.oracles import warm_decode_oracle
from silica.bench.scenario import OracleKind, Scenario, Workload
from silica.bench.scenarios import (
    _WARM_DECODE_PROMPT,
    BUILTIN_SCENARIOS,
)

EXTENDED_SCENARIO_IDS = (
    "qwen3.5-27b-warm-decode-b1-4k",
    "qwen3.5-27b-warm-decode-b1-8k",
    "gemma4-31b-warm-decode-b1-4k",
    "gemma4-31b-warm-decode-b1-8k",
)


@pytest.mark.parametrize("scenario_id", EXTENDED_SCENARIO_IDS)
def test_extended_context_scenario_registered(scenario_id: str) -> None:
    """Each of the four extended-context scenarios appears in the
    catalog and uses the WARM_DECODE oracle (no new oracle)."""
    sc = BUILTIN_SCENARIOS[scenario_id]
    assert sc.oracle == OracleKind.WARM_DECODE


@pytest.mark.parametrize("scenario_id", EXTENDED_SCENARIO_IDS)
def test_extended_context_scenario_oracle_config(
    scenario_id: str,
) -> None:
    """``oracle_config`` carries the design-target metadata that the
    runner / oracle echo into the JSONL row."""
    sc = BUILTIN_SCENARIOS[scenario_id]
    cfg = sc.oracle_config
    assert "target_context_tokens" in cfg, (
        f"{scenario_id}: missing target_context_tokens; the runner "
        f"+ oracle key off this for extended-context metadata"
    )
    assert "expected_total_context_floor" in cfg
    target = cfg["target_context_tokens"]
    floor = cfg["expected_total_context_floor"]
    assert isinstance(target, int) and target > 0
    assert isinstance(floor, int) and 0 < floor <= target


def test_4k_target_has_target_4096() -> None:
    for sid in (
        "qwen3.5-27b-warm-decode-b1-4k",
        "gemma4-31b-warm-decode-b1-4k",
    ):
        cfg = BUILTIN_SCENARIOS[sid].oracle_config
        assert cfg["target_context_tokens"] == 4096


def test_8k_target_has_target_8192() -> None:
    for sid in (
        "qwen3.5-27b-warm-decode-b1-8k",
        "gemma4-31b-warm-decode-b1-8k",
    ):
        cfg = BUILTIN_SCENARIOS[sid].oracle_config
        assert cfg["target_context_tokens"] == 8192


@pytest.mark.parametrize(
    "scenario_id,gate_env_var",
    [
        ("qwen3.5-27b-warm-decode-b1-4k", "SILICA_REAL_QWEN3_5_27B"),
        ("qwen3.5-27b-warm-decode-b1-8k", "SILICA_REAL_QWEN3_5_27B"),
        ("gemma4-31b-warm-decode-b1-4k", "SILICA_REAL_GEMMA4_31B"),
        ("gemma4-31b-warm-decode-b1-8k", "SILICA_REAL_GEMMA4_31B"),
    ],
)
def test_extended_context_scenario_gated(
    scenario_id: str, gate_env_var: str
) -> None:
    sc = BUILTIN_SCENARIOS[scenario_id]
    assert sc.gate_env_var == gate_env_var


@pytest.mark.parametrize("scenario_id", EXTENDED_SCENARIO_IDS)
def test_extended_context_workload_shape(scenario_id: str) -> None:
    """B=1, max_tokens >= 113 (warmup + window + measurement_min + 1),
    prefix_cache=False, kv_codec=None, single prompt of repeated base."""
    sc = BUILTIN_SCENARIOS[scenario_id]
    wl = sc.workload
    assert wl.max_batch_size == 1
    # >=113 is the WARM_DECODE oracle's hard minimum; D-021 step 2(d)
    # picked 600 explicitly so warm-up + measurement + headroom fit
    # comfortably even with EOS-induced early termination.
    assert wl.max_tokens >= 113
    assert wl.prefix_cache is False
    assert wl.kv_codec is None
    assert len(wl.prompts) == 1
    # Prompt is a repeated _WARM_DECODE_PROMPT (registration-time
    # calibration; runner records the actual token count).
    assert _WARM_DECODE_PROMPT in wl.prompts[0]


def test_8k_prompt_is_strictly_longer_than_4k() -> None:
    """Tokenizer-independent monotonicity: same model family's 8K
    row must have a longer prompt string than its 4K row, and both
    must be longer than the v1.7.13 384-token row."""
    base_27b = BUILTIN_SCENARIOS["qwen3.5-27b-warm-decode-b1"]
    p_4k = BUILTIN_SCENARIOS["qwen3.5-27b-warm-decode-b1-4k"]
    p_8k = BUILTIN_SCENARIOS["qwen3.5-27b-warm-decode-b1-8k"]
    base_31b = BUILTIN_SCENARIOS["gemma4-31b-warm-decode-b1"]
    g_4k = BUILTIN_SCENARIOS["gemma4-31b-warm-decode-b1-4k"]
    g_8k = BUILTIN_SCENARIOS["gemma4-31b-warm-decode-b1-8k"]
    assert (
        len(p_4k.workload.prompts[0])
        > len(base_27b.workload.prompts[0])
    )
    assert (
        len(p_8k.workload.prompts[0]) > len(p_4k.workload.prompts[0])
    )
    assert (
        len(g_4k.workload.prompts[0])
        > len(base_31b.workload.prompts[0])
    )
    assert (
        len(g_8k.workload.prompts[0]) > len(g_4k.workload.prompts[0])
    )


# --- oracle metadata echo ---


def _make_warm_decode_scenario_stub() -> Scenario:
    """Minimal Scenario stub for direct oracle invocation."""
    return Scenario(
        id="test-extended",
        repo="dummy/repo",
        workload=Workload(
            name="x",
            prompts=("dummy",),
            max_tokens=600,
        ),
        oracle=OracleKind.WARM_DECODE,
    )


def _stable_timeline(
    cold_ttft_ms: float = 100.0,
    n_decode: int = 200,
    interval_ms: float = 6.5,
) -> list[float]:
    ts = [cold_ttft_ms]
    for i in range(n_decode):
        # Tiny alternating jitter so the rolling-window stability rule
        # exits warm-up cleanly.
        ts.append(ts[-1] + interval_ms + (0.05 if i % 2 == 0 else -0.05))
    return ts


def _default_warm_context() -> dict[str, object]:
    return {
        "vocab_size": 200_000,
        "warmup_min_steps": 32,
        "warmup_rolling_window": 16,
        "warmup_rel_std_threshold": 0.05,
        "measurement_steps_min": 64,
    }


def test_oracle_echoes_extended_context_metadata_when_present() -> None:
    """When the runner populates ``target_context_tokens``,
    ``expected_total_context_floor``, ``prompt_token_counts``, and
    ``max_tokens`` in context, the WARM_DECODE oracle echoes them
    plus computes ``actual_total_context_*`` and
    ``reached_expected_floor`` from per-row token counts.
    """
    ts = _stable_timeline(n_decode=200)
    tokens = {0: [42] * (len(ts))}  # tokens-list length = number of timestamps
    token_ts_ms = {0: ts}

    context = _default_warm_context()
    context.update(
        {
            "target_context_tokens": 4096,
            "expected_total_context_floor": 3500,
            "prompt_token_counts": [3450],
            "max_tokens": 600,
        }
    )

    ok, reason, meta = warm_decode_oracle(
        _make_warm_decode_scenario_stub(),
        (tokens, token_ts_ms),
        context,
    )
    assert ok, f"reason={reason}"
    assert meta["target_context_tokens"] == 4096
    assert meta["expected_total_context_floor"] == 3500
    assert meta["prompt_token_counts"] == [3450]
    assert math.isclose(meta["prompt_token_count_per_row_mean"], 3450.0)
    assert meta["max_tokens"] == 600
    # 3450 prompt + 201 tokens (incl. ttft entry) > 3500 floor.
    assert meta["actual_total_context_per_row"] == [3450 + len(tokens[0])]
    assert meta["actual_total_context_min"] == 3450 + len(tokens[0])
    assert meta["reached_expected_floor"] is True


def test_oracle_under_target_floor_marked_false_but_still_passes() -> None:
    """Under-target outcomes (tokenizer drift below the configured
    floor) are diagnostic, not gate failures — the oracle still
    returns ok=True and surfaces ``reached_expected_floor=False``."""
    ts = _stable_timeline(n_decode=200)
    tokens = {0: [1] * len(ts)}
    token_ts_ms = {0: ts}

    context = _default_warm_context()
    context.update(
        {
            "target_context_tokens": 8192,
            "expected_total_context_floor": 7000,
            "prompt_token_counts": [
                100
            ],  # well below the 8K floor — simulates massive drift
            "max_tokens": 600,
        }
    )
    ok, _, meta = warm_decode_oracle(
        _make_warm_decode_scenario_stub(),
        (tokens, token_ts_ms),
        context,
    )
    assert ok, "the floor is diagnostic, not a hard fail"
    assert meta["reached_expected_floor"] is False


def test_oracle_legacy_context_emits_no_extended_metadata() -> None:
    """Regression guard: WARM_DECODE rows without extended-context
    keys in context (the cache-only / dense / MoE v1.7.13 rows) see
    metadata byte-identical to the pre-step-2(d) shape — no
    spurious ``target_context_tokens`` / ``actual_total_*`` keys."""
    ts = _stable_timeline(n_decode=200)
    tokens = {0: [1] * len(ts)}
    token_ts_ms = {0: ts}

    context = _default_warm_context()
    ok, _, meta = warm_decode_oracle(
        _make_warm_decode_scenario_stub(),
        (tokens, token_ts_ms),
        context,
    )
    assert ok
    for key in (
        "target_context_tokens",
        "expected_total_context_floor",
        "prompt_token_counts",
        "prompt_token_count_per_row_mean",
        "actual_total_context_per_row",
        "actual_total_context_min",
        "reached_expected_floor",
        "max_tokens",
    ):
        assert key not in meta, (
            f"legacy WARM_DECODE row should not surface "
            f"extended-context metadata key {key!r}; "
            f"got value={meta[key]!r}"
        )
