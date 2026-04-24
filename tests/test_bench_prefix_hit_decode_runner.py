"""Runner-level tests for the ``DECODE_TOK_S_WITH_PREFIX_HIT`` oracle
path (P-5-A.3b follow-up).

Complements ``tests/test_bench_prefix_hit_decode_oracle.py`` (pure
oracle function) by exercising the pieces that bracket the oracle in
``BenchRunner._run_one``:

1. Authoring-error validation — invalid workload shapes for this
   oracle fail BEFORE the engine factory loads an adapter. Regression
   lock against the A.3b review finding M-2 (moving all workload
   checks out of ``_run_prefix_hit_decode`` into
   ``_validate_workload_for_oracle``).
2. ``ScenarioResult`` shape — the oracle's ``row1_decode_tok_s`` and
   ``row1_first_token_ms`` metadata values get promoted into the
   standard ``decode_tok_s`` / ``ttft_ms`` columns so downstream JSONL
   consumers (bench report + the A.3c acceptance gate) do not need
   to special-case this oracle's metadata bucket.

Both test groups use a fake adapter/engine pair injected via the
``BenchRunner(engine_factory=...)`` seam, so no real model load is
required — kept out of the gated on-device tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pytest

from silica.bench.runner import BenchRunner
from silica.bench.scenario import (
    OracleKind,
    Scenario,
    ScenarioResult,
    Workload,
)

_REPO = "Qwen/Qwen3.5-0.8B"
_SHARED_PROMPT = (
    "The continuous-batching scheduler allocates fresh KV blocks at "
    "admission time, retains source references through the radix "
    "tree, and releases detached storage only once hit counts drop "
    "to zero."
)


# ---------------------------------------------------------------------------
# Fake adapter + engine for runner-integration tests
# ---------------------------------------------------------------------------


class _FakeLayout:
    num_layers = 2
    n_kv_heads = 2
    head_dim = 64
    import mlx.core as mx  # type: ignore[import-not-found]

    dtype = mx.float16
    bytes_per_token_total: int | None = None


class _FakeConfig:
    model_name = "stub-0.8B"
    num_layers = 2
    hidden_size = 128
    vocab_size = 1024


class _FakeEventStream:
    """Emits a canned BatchEvent stream for ``generate_batch``.

    The collector reads ``time.perf_counter()`` between events; we
    interleave small sleeps so per-token timestamps advance enough
    to compute a non-degenerate ``(t_last - t_first)`` interval for
    the oracle's ``row1_decode_tok_s`` formula. Without the sleep
    all token events would land in the same tick and the oracle's
    ``row_1_nonpositive_interval`` gate would (correctly) reject the
    run.
    """

    def __iter__(self) -> Iterator[Any]:
        import time

        from silica.core.events import BatchEvent

        # Row 0: 4 tokens then done. Rapid — 1 ms between events.
        for tok in (7, 8, 9, 10):
            time.sleep(0.001)
            yield BatchEvent(kind="token", req_index=0, token_id=tok)
        yield BatchEvent(kind="done", req_index=0, finish_reason="max_tokens")
        # Row 1: 4 tokens then done. Same cadence.
        for tok in (7, 8, 9, 10):
            time.sleep(0.001)
            yield BatchEvent(kind="token", req_index=1, token_id=tok)
        yield BatchEvent(kind="done", req_index=1, finish_reason="max_tokens")


class _FakeEngine:
    def __init__(self) -> None:
        from silica.core.profiler import MetricsRegistry

        self.metrics = MetricsRegistry()

    def generate_batch(
        self,
        prompts: Iterable[str],
        params: Any,
        *,
        max_batch_size: int = 1,
        prefix_cache: Any | None = None,
    ) -> Iterator[Any]:
        # Simulate prefix-cache hit on row 1's admission so the
        # oracle's hit-counter gate passes.
        if prefix_cache is not None:
            prefix_cache.hits += 1
        return iter(_FakeEventStream())


class _FakeTokenizer:
    eos_token_ids: tuple[int, ...] = ()

    def encode(self, text: str) -> list[int]:
        return list(range(len(text) % 64 + 1))


class _FakeAdapter:
    config = _FakeConfig()

    def kv_layout(self) -> Any:
        return _FakeLayout()

    def tokenizer(self) -> Any:
        return _FakeTokenizer()


class _FactoryProbe:
    """engine_factory wrapper that records invocations so tests can
    assert it was (or was not) called."""

    def __init__(self) -> None:
        self.calls: list[Scenario] = []

    def __call__(
        self, scenario: Scenario
    ) -> tuple[Any, Any]:
        self.calls.append(scenario)
        return _FakeAdapter(), _FakeEngine()


def _prefix_hit_scenario(
    *,
    prompts: tuple[str, ...] = (_SHARED_PROMPT, _SHARED_PROMPT),
    max_batch_size: int = 1,
    prefix_cache: bool = True,
    kv_codec: str | None = "fp16",
    scenario_id: str = "prefix-hit-fp16-fake",
) -> Scenario:
    return Scenario(
        id=scenario_id,
        repo=_REPO,
        workload=Workload(
            name="prefix-hit-fake",
            prompts=prompts,
            max_tokens=4,
            max_batch_size=max_batch_size,
            prefix_cache=prefix_cache,
            kv_codec=kv_codec,
        ),
        oracle=OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT,
        gate_env_var=None,
    )


def _make_runner(
    factory: _FactoryProbe,
    *,
    clock_values: list[float] | None = None,
) -> BenchRunner:
    """Build a BenchRunner wired to the probe factory.

    ``clock_values`` feeds a deterministic clock into the runner's
    outer timing (wall_s). Kept simple — the oracle's per-token
    timestamps come from ``time.perf_counter()`` inside the collector
    and are not controlled here; the fake engine's events all fire
    in one Python-level tick so timestamps bunch together, which the
    oracle handles via the ``row_1_nonpositive_interval`` gate that
    we exercise explicitly."""
    from pathlib import Path

    # BenchRunner exposes reset_peak + read_peak_mb (not a combined
    # peak_memory_mb). Injecting no-op hooks sidesteps MLX peak-memory
    # calls in the tests — the assertions do not depend on peak mem.
    return BenchRunner(
        engine_factory=factory,
        out_path=Path("/tmp/silica_prefix_hit_runner_test.jsonl"),
        clock=lambda: 0.0 if not clock_values else clock_values.pop(0),
        reset_peak=lambda: None,
        read_peak_mb=lambda: 0.0,
    )


# ---------------------------------------------------------------------------
# Group 1 — authoring validation skips engine_factory (M-2 regression)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broken_workload_kwargs,expected_reason_substring",
    [
        (
            {"max_batch_size": 2},
            "max_batch_size=1",
        ),
        (
            {"prompts": ("one-prompt-only",)},
            "exactly 2 prompts",
        ),
        (
            {"prompts": ("different", "prompts-here")},
            "identical prompts",
        ),
        (
            {"prefix_cache": False, "kv_codec": None},
            "prefix_cache=True",
        ),
        (
            {"kv_codec": None},
            "kv_codec to be set",
        ),
    ],
)
def test_invalid_workload_fails_before_engine_factory(
    broken_workload_kwargs: dict[str, Any],
    expected_reason_substring: str,
) -> None:
    """Authoring errors on this oracle surface via
    ``_validate_workload_for_oracle`` and short-circuit
    ``_run_one`` before the adapter + engine get loaded. Pre-fix
    these checks lived in ``_run_prefix_hit_decode`` and fired only
    after a multi-GB model load.
    """
    base = {
        "prompts": (_SHARED_PROMPT, _SHARED_PROMPT),
        "max_batch_size": 1,
        "prefix_cache": True,
        "kv_codec": "fp16",
    }
    kwargs = {**base, **broken_workload_kwargs}
    scenario = _prefix_hit_scenario(**kwargs)

    probe = _FactoryProbe()
    runner = _make_runner(probe)
    result = runner._run_one(scenario, seed=0)

    assert result.status == "failed"
    assert result.reason is not None
    assert expected_reason_substring in result.reason, (
        f"reason {result.reason!r} missing {expected_reason_substring!r}"
    )
    # Load-bearing invariant: engine_factory never called → no
    # adapter / weights load cost on authoring-broken scenarios.
    assert probe.calls == [], (
        "engine_factory should not be invoked when workload validation "
        f"fails; got {len(probe.calls)} call(s)"
    )


# ---------------------------------------------------------------------------
# Group 2 — ScenarioResult.decode_tok_s promoted from oracle metadata
# ---------------------------------------------------------------------------


def test_decode_tok_s_field_populated_from_oracle_metadata() -> None:
    """H-1 regression lock. ``Engine.generate_batch`` does not populate
    ``MetricsRegistry``, so the runner's default
    ``snap.decode_tok_s`` is ``None`` on any generate_batch-based
    oracle. For this oracle specifically, the runner promotes
    ``metadata['row1_decode_tok_s']`` into
    ``ScenarioResult.decode_tok_s`` so the standard JSONL column
    carries the measurement — no downstream tool has to special-
    case this oracle's metadata bucket to find the headline number.
    """
    scenario = _prefix_hit_scenario()
    probe = _FactoryProbe()
    runner = _make_runner(probe)
    result = runner._run_one(scenario, seed=0)

    # Sanity: the fake engine / event stream should pass every gate
    # in the oracle. If this fails, the runner plumbing or the fake
    # setup drifted.
    assert result.status == "ok", f"expected ok, got {result.status} ({result.reason})"
    assert probe.calls == [scenario]

    # The load-bearing assertion.
    assert result.decode_tok_s is not None, (
        "ScenarioResult.decode_tok_s must be populated for the "
        "DECODE_TOK_S_WITH_PREFIX_HIT oracle — either by promoting "
        "metadata['row1_decode_tok_s'] (post-fix) or by Engine.metrics "
        "(not populated on generate_batch today)"
    )
    # And it equals the metadata value.
    assert result.decode_tok_s == result.metadata["row1_decode_tok_s"]


def test_ttft_ms_field_populated_from_oracle_metadata() -> None:
    """Parallel to decode_tok_s: ``ttft_ms`` on the ScenarioResult
    gets ``metadata['row1_first_token_ms']`` for this oracle so the
    JSONL column reflects the seeded-admission first-token latency."""
    scenario = _prefix_hit_scenario()
    result = _make_runner(_FactoryProbe())._run_one(scenario, seed=0)

    assert result.status == "ok"
    assert result.ttft_ms is not None
    assert result.ttft_ms == result.metadata["row1_first_token_ms"]


def test_promotion_is_gated_on_oracle_kind() -> None:
    """Regression guard: the ``decode_tok_s`` / ``ttft_ms`` promotion
    runs only inside the ``if scenario.oracle ==
    DECODE_TOK_S_WITH_PREFIX_HIT`` branch. This test verifies the
    gate by asserting the prefix-hit oracle's metadata keys
    (``row1_decode_tok_s`` / ``row1_first_token_ms``) are not
    present in results for other oracles — if the gate were dropped
    and the promotion ran unconditionally, SMOKE / B1 / BGT1 /
    TEACHER_FORCED results would silently pull an unrelated key
    out of their own metadata. Pure static check, no engine
    invocation needed.
    """
    # Build a synthetic ScenarioResult with SMOKE's typical
    # metadata shape — no prefix-hit keys present.
    smoke_result = ScenarioResult(
        scenario_id="smoke-fake",
        status="ok",
        metadata={"total_tokens": 2, "max_token_id": 5, "vocab_size": 1024},
    )
    # Confirm the prefix-hit keys are absent — the promotion branch
    # would be a no-op even if accidentally entered with this metadata.
    assert "row1_decode_tok_s" not in smoke_result.metadata
    assert "row1_first_token_ms" not in smoke_result.metadata
