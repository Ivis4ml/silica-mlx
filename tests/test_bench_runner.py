"""Unit tests for ``silica.bench.runner.BenchRunner`` (P-4.1).

Uses a fake engine factory so every runner branch can be covered
without loading real weights. The runner's public surface is
small — skip gates, workload dispatch, oracle invocation, metrics
plumbing, JSONL emission — and these tests pin each branch with a
minimal scenario definition tailored to the branch under test.

The fake adapter / engine pair mirrors the exact methods the runner
touches (``adapter.tokenizer()``, ``adapter.config.vocab_size``,
``engine.generate``, ``engine.metrics.snapshot``). They do NOT
implement the full :class:`ModelAdapter` / :class:`Engine` surface —
the runner is the only caller, so broader mocks would be dead code.

Coverage:

  * ``test_cache_missing_gate_skips_without_loading`` — bogus repo,
    runner never calls the factory.
  * ``test_env_var_not_set_gate_skips_without_loading`` — cache
    present but gate_env_var not "1".
  * ``test_env_var_set_to_1_runs`` — same scenario with the env
    var set, factory is invoked.
  * ``test_smoke_scenario_passes_with_valid_tokens`` — happy path.
  * ``test_smoke_scenario_fails_on_out_of_vocab_token`` — oracle
    rejects, status="failed", metrics still populated.
  * ``test_engine_exception_collapses_to_failed`` — generate raises.
  * ``test_batched_workload_deferred_to_p42`` — max_batch_size>1
    scenario is skipped, not attempted.
  * ``test_jsonl_output_one_row_per_result`` — out_path contains
    every row including skipped.
  * ``test_unknown_oracle_kind_raises_via_runner`` — a scenario
    using a not-yet-implemented oracle (B1_PARITY_VS_SINGLE) runs
    the engine but fails at oracle dispatch with a NotImplementedError
    surfacing in ScenarioResult.reason.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from silica.bench import (
    BenchRunner,
    OracleKind,
    Scenario,
    ScenarioResult,
    Workload,
    hf_cache_path_for_repo,
)
from silica.core.events import BatchEvent
from silica.core.profiler import MetricsRegistry

# ---------- fakes ------------------------------------------------------


@dataclass
class _FakeConfig:
    vocab_size: int


class _FakeTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.eos_token_ids: set[int] = set()

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return ""


class _FakeAdapter:
    def __init__(self, vocab_size: int = 100) -> None:
        self.config = _FakeConfig(vocab_size=vocab_size)
        self._tokenizer = _FakeTokenizer(vocab_size)

    def tokenizer(self) -> _FakeTokenizer:
        return self._tokenizer


class _FakeEngine:
    """Stand-in for :class:`silica.engine.Engine` — yields pre-programmed tokens.

    ``generate`` drives single-request scenarios. ``generate_batch``
    drives the B=1 parity path: by default it echoes the same token
    stream as ``generate`` followed by a ``done`` event (happy-path
    parity), but ``batched_events`` can override the stream so tests
    can exercise abort / unexpected-req_index failure modes.
    """

    def __init__(
        self,
        tokens: list[int],
        *,
        metrics: dict[str, float | int] | None = None,
        raise_on_generate: Exception | None = None,
        batched_events: list[BatchEvent] | None = None,
    ) -> None:
        self._tokens = list(tokens)
        self._raise = raise_on_generate
        self._batched_events = (
            list(batched_events) if batched_events is not None else None
        )
        self.metrics = MetricsRegistry()
        for name, value in (metrics or {}).items():
            self.metrics.set_metric(name, value)

    def generate(self, prompt: str, params: Any) -> Iterator[int]:
        if self._raise is not None:
            raise self._raise
        yield from self._tokens

    def generate_batch(
        self,
        prompts: Any,
        params: Any,
        *,
        max_batch_size: int | None = None,
        prefix_cache: Any = None,
    ) -> Iterator[BatchEvent]:
        if self._batched_events is not None:
            yield from self._batched_events
            return
        for tok in self._tokens:
            yield BatchEvent.token(req_index=0, token_id=tok)
        yield BatchEvent.done(req_index=0, reason="max_tokens")


def _factory_returning(
    adapter: _FakeAdapter, engine: _FakeEngine
) -> Any:
    def factory(scenario: Scenario) -> tuple[Any, Any]:
        return adapter, engine

    return factory


# ---------- scenario builders -----------------------------------------


def _scenario(
    *,
    scenario_id: str,
    repo: str,
    oracle: OracleKind = OracleKind.SMOKE,
    gate_env_var: str | None = None,
    max_batch_size: int = 1,
    prompts: tuple[str, ...] = ("hello",),
    max_tokens: int = 4,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        repo=repo,
        workload=Workload(
            name="fake",
            prompts=prompts,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
        ),
        oracle=oracle,
        gate_env_var=gate_env_var,
    )


def _cached_repo_pointing_at(tmp_path: Path) -> str:
    """Return a repo string whose derived cache path exists under tmp_path.

    Derives the HF-style directory name from the repo, then creates
    that directory inside the user's real HF cache root (via
    monkeypatching Path.home). The test fixture below wires that up;
    here we simply pick a repo string that will not collide with a
    real one.
    """
    return "test-owner/test-fake-bench-model"


@pytest.fixture
def fake_home_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() so cache-presence checks look under tmp_path."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".cache" / "huggingface" / "hub").mkdir(
        parents=True, exist_ok=True
    )
    return tmp_path


def _create_cache_dir(repo: str) -> Path:
    """Create the HF-style cache dir for ``repo`` under the (faked) home."""
    cache = hf_cache_path_for_repo(repo)
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# ---------- tests -----------------------------------------------------


def test_cache_missing_gate_skips_without_loading(
    fake_home_cache: Path,
) -> None:
    called = {"n": 0}

    def factory(scenario: Scenario) -> Any:
        called["n"] += 1
        raise AssertionError("factory must not be called when cache missing")

    scenario = _scenario(
        scenario_id="cache-missing",
        repo="test-owner/never-cached",
    )
    runner = BenchRunner(engine_factory=factory, reset_peak=lambda: None, read_peak_mb=lambda: None)
    [result] = runner.run([scenario])

    assert result.status == "skipped"
    assert result.reason is not None
    assert result.reason.startswith("cache_missing:")
    assert called["n"] == 0


def test_env_var_not_set_gate_skips_without_loading(
    fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = "test-owner/dual-gated"
    _create_cache_dir(repo)
    monkeypatch.delenv("SILICA_BENCH_FAKE_GATE", raising=False)

    called = {"n": 0}

    def factory(scenario: Scenario) -> Any:
        called["n"] += 1
        raise AssertionError("factory must not be called when env var missing")

    scenario = _scenario(
        scenario_id="env-gated",
        repo=repo,
        gate_env_var="SILICA_BENCH_FAKE_GATE",
    )
    runner = BenchRunner(engine_factory=factory, reset_peak=lambda: None, read_peak_mb=lambda: None)
    [result] = runner.run([scenario])

    assert result.status == "skipped"
    assert result.reason == "env_var_not_set:SILICA_BENCH_FAKE_GATE"
    assert called["n"] == 0


def test_env_var_set_to_1_runs(
    fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = "test-owner/dual-gated-runs"
    _create_cache_dir(repo)
    monkeypatch.setenv("SILICA_BENCH_FAKE_GATE", "1")

    adapter = _FakeAdapter(vocab_size=50)
    engine = _FakeEngine([7, 8])
    scenario = _scenario(
        scenario_id="env-gated-runs",
        repo=repo,
        gate_env_var="SILICA_BENCH_FAKE_GATE",
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "ok"
    assert result.total_tokens == 2


def test_smoke_scenario_passes_with_valid_tokens(fake_home_cache: Path) -> None:
    repo = "test-owner/happy"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        [10, 20, 30, 40],
        metrics={
            "ttft_ms": 42.5,
            "prefill_tok_s": 1234.0,
            "decode_tok_s": 100.0,
            "resident_mb": 512.0,
        },
    )

    scenario = _scenario(scenario_id="happy", repo=repo)
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: 1024.0,
    )
    [result] = runner.run([scenario])

    assert result.status == "ok"
    assert result.reason is None
    assert result.total_tokens == 4
    assert result.ttft_ms == pytest.approx(42.5)
    assert result.prefill_tok_s == pytest.approx(1234.0)
    assert result.decode_tok_s == pytest.approx(100.0)
    assert result.resident_mb == pytest.approx(512.0)
    assert result.peak_memory_mb == pytest.approx(1024.0)
    assert result.wall_s is not None and result.wall_s >= 0
    assert result.metadata["total_tokens"] == 4
    assert result.metadata["max_token_id"] == 40
    assert result.metadata["vocab_size"] == 100


def test_smoke_scenario_fails_on_out_of_vocab_token(
    fake_home_cache: Path,
) -> None:
    repo = "test-owner/bad-token"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=50)
    engine = _FakeEngine([10, 999])  # 999 >= vocab_size

    scenario = _scenario(scenario_id="bad-token", repo=repo)
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert result.reason.startswith("smoke_token_1_out_of_vocab:")
    # Metrics still populated (engine did run, oracle just rejected).
    assert result.total_tokens == 2


def test_engine_exception_collapses_to_failed(fake_home_cache: Path) -> None:
    repo = "test-owner/boom"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        [], raise_on_generate=RuntimeError("prefill exploded")
    )

    scenario = _scenario(scenario_id="boom", repo=repo)
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "RuntimeError" in result.reason
    assert "prefill exploded" in result.reason
    # No metrics populated because we never reached snapshot.
    assert result.ttft_ms is None
    assert result.total_tokens is None


def test_smoke_b1_with_multiple_prompts_is_authoring_error(
    fake_home_cache: Path,
) -> None:
    """SMOKE at max_batch_size=1 must have exactly one prompt — a
    single-request scenario with multiple prompts is authoring-
    broken (which prompt does the run use?). Surfaces as
    status=failed before the engine factory runs."""
    repo = "test-owner/smoke-b1-multi-prompt"
    _create_cache_dir(repo)

    called = {"n": 0}

    def factory(scenario: Scenario) -> Any:
        called["n"] += 1
        raise AssertionError("factory must not be called for authoring errors")

    scenario = _scenario(
        scenario_id="smoke-b1-multi-prompt",
        repo=repo,
        max_batch_size=1,
        prompts=("a", "b"),
    )
    runner = BenchRunner(
        engine_factory=factory, reset_peak=lambda: None, read_peak_mb=lambda: None
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "max_batch_size=1 requires exactly 1 prompt" in result.reason
    assert called["n"] == 0


def test_smoke_b_gt_1_with_single_prompt_is_authoring_error(
    fake_home_cache: Path,
) -> None:
    """SMOKE at max_batch_size>1 must have at least 2 prompts — B>1
    with one prompt would degenerate to single-request hiding in a
    batched API."""
    repo = "test-owner/smoke-bgt1-single-prompt"
    _create_cache_dir(repo)

    called = {"n": 0}

    def factory(scenario: Scenario) -> Any:
        called["n"] += 1
        raise AssertionError("factory must not be called for authoring errors")

    scenario = _scenario(
        scenario_id="smoke-bgt1-single-prompt",
        repo=repo,
        max_batch_size=4,
        prompts=("only-one",),
    )
    runner = BenchRunner(
        engine_factory=factory, reset_peak=lambda: None, read_peak_mb=lambda: None
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "requires at least 2 prompts" in result.reason
    assert called["n"] == 0


def test_jsonl_output_one_row_per_result(
    fake_home_cache: Path, tmp_path: Path
) -> None:
    repo_ok = "test-owner/ok"
    repo_skip = "test-owner/never-cached-jsonl"
    _create_cache_dir(repo_ok)

    adapter = _FakeAdapter(vocab_size=50)
    engine = _FakeEngine([5])

    ok_scenario = _scenario(scenario_id="ok", repo=repo_ok)
    skip_scenario = _scenario(scenario_id="skip", repo=repo_skip)

    out_path = tmp_path / "subdir" / "bench.jsonl"
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        out_path=out_path,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    results = runner.run([ok_scenario, skip_scenario])

    assert [r.status for r in results] == ["ok", "skipped"]
    assert out_path.exists()
    lines = [ln for ln in out_path.read_text().splitlines() if ln]
    assert len(lines) == 2
    rows = [json.loads(ln) for ln in lines]
    assert rows[0]["scenario_id"] == "ok"
    assert rows[0]["status"] == "ok"
    assert rows[0]["metadata"]["total_tokens"] == 1
    assert rows[1]["scenario_id"] == "skip"
    assert rows[1]["status"] == "skipped"


def test_unknown_oracle_kind_raises_via_runner(fake_home_cache: Path) -> None:
    """A scenario naming a not-yet-implemented oracle surfaces the
    NotImplementedError via ScenarioResult.reason rather than crashing
    the runner (which would take down subsequent scenarios in the
    same run). Uses TEACHER_FORCED_ARGMAX because its implementation
    lands in P-4.3; the other oracle kinds are already implemented."""
    repo = "test-owner/unknown-oracle"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine([1, 2])

    scenario = _scenario(
        scenario_id="unknown-oracle",
        repo=repo,
        oracle=OracleKind.TEACHER_FORCED_ARGMAX,
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "NotImplementedError" in result.reason
    assert "P-4.3" in result.reason


# ---------- B1 parity oracle / runner branch --------------------------


def test_b1_parity_happy_path(fake_home_cache: Path) -> None:
    """Single reference and B=1 batched emit identical token streams;
    runner collects batched tokens, oracle sees matching reference
    via context['reference_tokens'], returns ok."""
    repo = "test-owner/b1-happy"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    # Default generate_batch echoes the single-request tokens, so
    # batch and reference match by construction.
    engine = _FakeEngine([10, 20, 30])
    scenario = _scenario(
        scenario_id="b1-happy",
        repo=repo,
        oracle=OracleKind.B1_PARITY_VS_SINGLE,
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "ok"
    assert result.reason is None
    assert result.total_tokens == 3
    assert result.metadata["reference_len"] == 3
    assert result.metadata["batch_len"] == 3
    assert result.metadata["first_mismatch_index"] == -1


def test_b1_parity_mismatch_reports_index(fake_home_cache: Path) -> None:
    """Single [1,2,3], batched [1,9,3]: oracle returns failed with
    first_mismatch_index=1 and both stream tokens in metadata."""
    repo = "test-owner/b1-mismatch"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        [1, 2, 3],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=1),
            BatchEvent.token(req_index=0, token_id=9),
            BatchEvent.token(req_index=0, token_id=3),
            BatchEvent.done(req_index=0, reason="max_tokens"),
        ],
    )
    scenario = _scenario(
        scenario_id="b1-mismatch",
        repo=repo,
        oracle=OracleKind.B1_PARITY_VS_SINGLE,
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason == "b1_parity_first_mismatch_index:1"
    assert result.metadata["first_mismatch_index"] == 1
    assert result.metadata["reference_len"] == 3
    assert result.metadata["batch_len"] == 3
    assert result.metadata["reference_token_at_mismatch"] == 2
    assert result.metadata["batch_token_at_mismatch"] == 9


def test_b1_parity_length_mismatch(fake_home_cache: Path) -> None:
    """Single stream is a strict prefix of batched: oracle reports
    length mismatch with first_mismatch_index set to the common
    prefix length (no tokens differ within the overlap)."""
    repo = "test-owner/b1-length"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        [1, 2],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=1),
            BatchEvent.token(req_index=0, token_id=2),
            BatchEvent.token(req_index=0, token_id=3),
            BatchEvent.done(req_index=0, reason="max_tokens"),
        ],
    )
    scenario = _scenario(
        scenario_id="b1-length",
        repo=repo,
        oracle=OracleKind.B1_PARITY_VS_SINGLE,
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert result.reason.startswith("b1_parity_length_mismatch:")
    assert result.metadata["reference_len"] == 2
    assert result.metadata["batch_len"] == 3
    assert result.metadata["first_mismatch_index"] == 2


def test_b1_parity_batched_aborted_fails_before_oracle(
    fake_home_cache: Path,
) -> None:
    """An ``aborted`` event in the batched stream must surface as a
    runner failure with a ``b1_batched_aborted:`` reason, NOT as an
    oracle mismatch. The user needs to know the scheduler misbehaved,
    not that the tokens disagreed."""
    repo = "test-owner/b1-aborted"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        [1, 2],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=1),
            BatchEvent.aborted(req_index=0, reason="budget-exhausted"),
        ],
    )
    scenario = _scenario(
        scenario_id="b1-aborted",
        repo=repo,
        oracle=OracleKind.B1_PARITY_VS_SINGLE,
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "b1_batched_aborted:budget-exhausted" in result.reason
    # Oracle never ran, so parity-specific metadata absent.
    assert "first_mismatch_index" not in result.metadata


def test_b1_parity_batched_unexpected_req_index_fails_before_oracle(
    fake_home_cache: Path,
) -> None:
    """An event with req_index != 0 in a B=1 stream is a scheduler
    fault (the only admitted request is row 0). Runner surfaces this
    as a structured failure before calling the oracle."""
    repo = "test-owner/b1-bad-req-index"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        [1, 2],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=1),
            BatchEvent.token(req_index=1, token_id=2),
        ],
    )
    scenario = _scenario(
        scenario_id="b1-bad-req-index",
        repo=repo,
        oracle=OracleKind.B1_PARITY_VS_SINGLE,
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "b1_batched_unexpected_req_index:1" in result.reason


# ---------- BGT1 parity oracle / runner branch ------------------------


def _bgt1_scenario(
    *,
    scenario_id: str,
    repo: str,
    prompts: tuple[str, ...] = ("prompt-a", "prompt-b"),
    max_tokens: int = 3,
    max_batch_size: int = 2,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        repo=repo,
        workload=Workload(
            name="fake-bgt1",
            prompts=prompts,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
        ),
        oracle=OracleKind.BGT1_DIRECT_BATCHED_REFERENCE,
    )


def _bgt1_events_for(per_row_tokens: dict[int, list[int]]) -> list[BatchEvent]:
    """Helper: turn a per-row token dict into the event stream
    generate_batch would emit (tokens in row-interleaved order,
    followed by done events)."""
    events: list[BatchEvent] = []
    rows = sorted(per_row_tokens)
    max_steps = max(len(per_row_tokens[r]) for r in rows)
    for step in range(max_steps):
        for row in rows:
            if step < len(per_row_tokens[row]):
                events.append(
                    BatchEvent.token(
                        req_index=row, token_id=per_row_tokens[row][step]
                    )
                )
    for row in rows:
        events.append(BatchEvent.done(req_index=row, reason="max_tokens"))
    return events


def test_bgt1_parity_happy_path(fake_home_cache: Path) -> None:
    """Silica batched and direct reference produce identical per-row
    streams. Oracle returns ok with per-row metadata."""
    repo = "test-owner/bgt1-happy"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    silica_tokens = {0: [11, 12, 13], 1: [21, 22, 23]}
    reference_tokens = {0: [11, 12, 13], 1: [21, 22, 23]}
    engine = _FakeEngine(
        tokens=[], batched_events=_bgt1_events_for(silica_tokens)
    )

    def fake_reference(
        adapter_arg: Any, prompts: list[str], params: Any
    ) -> dict[int, list[int]]:
        assert adapter_arg is adapter
        assert prompts == ["prompt-a", "prompt-b"]
        # params.stop_token_ids must be empty — runner override.
        assert params.stop_token_ids == ()
        return reference_tokens

    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        direct_batched_reference=fake_reference,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    scenario = _bgt1_scenario(scenario_id="bgt1-happy", repo=repo)
    [result] = runner.run([scenario])

    assert result.status == "ok"
    assert result.reason is None
    # total_tokens sums per-row: 3 + 3 = 6
    assert result.total_tokens == 6
    assert result.metadata["first_failure"] is None
    rows = result.metadata["rows"]
    assert len(rows) == 2
    for row_entry in rows:
        assert row_entry["first_mismatch_index"] == -1
        assert row_entry["batch_len"] == 3
        assert row_entry["reference_len"] == 3


def test_bgt1_parity_per_row_mismatch(fake_home_cache: Path) -> None:
    """Row 1 diverges at index 2; oracle reports the specific row +
    position in metadata.first_failure."""
    repo = "test-owner/bgt1-mismatch"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    silica_tokens = {0: [11, 12, 13], 1: [21, 22, 99]}
    reference_tokens = {0: [11, 12, 13], 1: [21, 22, 23]}
    engine = _FakeEngine(
        tokens=[], batched_events=_bgt1_events_for(silica_tokens)
    )

    def fake_reference(adapter_arg: Any, prompts: Any, params: Any) -> Any:
        return reference_tokens

    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        direct_batched_reference=fake_reference,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    scenario = _bgt1_scenario(scenario_id="bgt1-mismatch", repo=repo)
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason == "bgt1_parity_row_1_mismatch_index:2"
    failure = result.metadata["first_failure"]
    assert failure["row"] == 1
    assert failure["first_mismatch_index"] == 2
    assert failure["batch_token_at_mismatch"] == 99
    assert failure["reference_token_at_mismatch"] == 23


def test_bgt1_parity_batched_aborted_fails_before_oracle(
    fake_home_cache: Path,
) -> None:
    """An aborted row in the Silica batched stream must surface as a
    runner failure with a bgt1_batched_aborted:* reason, never as an
    oracle mismatch."""
    repo = "test-owner/bgt1-aborted"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        tokens=[],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=11),
            BatchEvent.aborted(req_index=1, reason="budget-exhausted"),
            BatchEvent.done(req_index=0, reason="max_tokens"),
        ],
    )

    def fake_reference(adapter_arg: Any, prompts: Any, params: Any) -> Any:
        return {0: [11], 1: [21]}

    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        direct_batched_reference=fake_reference,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    scenario = _bgt1_scenario(scenario_id="bgt1-aborted", repo=repo)
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "bgt1_batched_aborted:row=1" in result.reason
    assert "reason=budget-exhausted" in result.reason


def test_bgt1_parity_unexpected_req_index_fails_before_oracle(
    fake_home_cache: Path,
) -> None:
    """A req_index outside range(len(prompts)) is a scheduler fault;
    runner surfaces it as a structured failure with the offending
    row number embedded."""
    repo = "test-owner/bgt1-bad-req"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        tokens=[],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=11),
            BatchEvent.token(req_index=5, token_id=99),  # out of range
        ],
    )

    def fake_reference(adapter_arg: Any, prompts: Any, params: Any) -> Any:
        return {0: [11], 1: [21]}

    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        direct_batched_reference=fake_reference,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    scenario = _bgt1_scenario(scenario_id="bgt1-bad-req", repo=repo)
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "bgt1_batched_unexpected_req_index:5" in result.reason


def test_bgt1_parity_missing_done_fails_before_oracle(
    fake_home_cache: Path,
) -> None:
    """If the batched stream closes without a ``done`` event for a
    row, runner treats that row as incomplete and fails with a
    structured reason. Protects against scheduler bugs where rows
    silently drop out."""
    repo = "test-owner/bgt1-missing-done"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        tokens=[],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=11),
            BatchEvent.done(req_index=0, reason="max_tokens"),
            # Row 1 never emits anything.
        ],
    )

    def fake_reference(adapter_arg: Any, prompts: Any, params: Any) -> Any:
        return {0: [11], 1: [21]}

    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        direct_batched_reference=fake_reference,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    scenario = _bgt1_scenario(scenario_id="bgt1-missing-done", repo=repo)
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "bgt1_batched_rows_never_completed:[1]" in result.reason


def test_bgt1_parity_rejects_single_prompt_at_authoring_time(
    fake_home_cache: Path,
) -> None:
    """A BGT1 scenario with only one prompt is authoring-broken.
    Runner short-circuits to failed before building the engine."""
    repo = "test-owner/bgt1-one-prompt"
    _create_cache_dir(repo)

    called = {"n": 0}

    def factory(scenario: Any) -> Any:
        called["n"] += 1
        raise AssertionError("factory must not run for authoring errors")

    scenario = _bgt1_scenario(
        scenario_id="bgt1-one-prompt",
        repo=repo,
        prompts=("only-one",),
        max_batch_size=2,
    )
    runner = BenchRunner(
        engine_factory=factory,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "at least 2 prompts" in result.reason
    assert called["n"] == 0


def test_bgt1_parity_rejects_batch_size_one_at_authoring_time(
    fake_home_cache: Path,
) -> None:
    """A BGT1 scenario with max_batch_size=1 is authoring-broken
    (use B1_PARITY_VS_SINGLE for that shape). Runner short-circuits
    to failed before building the engine."""
    repo = "test-owner/bgt1-batch-one"
    _create_cache_dir(repo)

    called = {"n": 0}

    def factory(scenario: Any) -> Any:
        called["n"] += 1
        raise AssertionError("factory must not run for authoring errors")

    scenario = _bgt1_scenario(
        scenario_id="bgt1-batch-one",
        repo=repo,
        prompts=("a", "b"),
        max_batch_size=1,
    )
    runner = BenchRunner(
        engine_factory=factory,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "max_batch_size>=2" in result.reason
    assert called["n"] == 0


# ---------- SMOKE B>1 batched dispatch --------------------------------


def _smoke_batched_scenario(
    *,
    scenario_id: str,
    repo: str,
    prompts: tuple[str, ...] = ("prompt-a", "prompt-b"),
    max_tokens: int = 3,
    max_batch_size: int = 2,
    prefix_cache: bool = False,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        repo=repo,
        workload=Workload(
            name="fake-smoke-batched",
            prompts=prompts,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
            prefix_cache=prefix_cache,
        ),
        oracle=OracleKind.SMOKE,
    )


def test_smoke_batched_happy_path(fake_home_cache: Path) -> None:
    """Multi-prompt SMOKE at B>1 returns per-row metadata, total
    tokens summed across rows, status ok."""
    repo = "test-owner/smoke-batched-happy"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    per_row = {0: [11, 12, 13], 1: [21, 22, 23]}
    engine = _FakeEngine(
        tokens=[], batched_events=_bgt1_events_for(per_row)
    )
    scenario = _smoke_batched_scenario(
        scenario_id="smoke-batched-happy", repo=repo
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "ok"
    assert result.reason is None
    assert result.total_tokens == 6
    rows = result.metadata["rows"]
    assert len(rows) == 2
    assert rows[0]["row"] == 0 and rows[0]["token_count"] == 3
    assert rows[1]["row"] == 1 and rows[1]["token_count"] == 3
    # max_token_id surfaced for at-a-glance JSONL validation.
    assert result.metadata["max_token_id"] == 23


def test_smoke_batched_fails_on_empty_row(fake_home_cache: Path) -> None:
    """A row that emits no tokens before done is a scheduler fault
    the SMOKE oracle catches — ``smoke_row_X_no_tokens_emitted``
    with the specific row index."""
    repo = "test-owner/smoke-batched-empty-row"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        tokens=[],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=11),
            BatchEvent.done(req_index=0, reason="max_tokens"),
            # Row 1 "done" without any token events.
            BatchEvent.done(req_index=1, reason="max_tokens"),
        ],
    )
    scenario = _smoke_batched_scenario(
        scenario_id="smoke-batched-empty-row", repo=repo
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason == "smoke_row_1_no_tokens_emitted"


def test_smoke_batched_aborted_row_fails_before_oracle(
    fake_home_cache: Path,
) -> None:
    """An aborted row in the SMOKE batched stream must surface as a
    runner failure with a ``smoke_batched_aborted:*`` reason, not
    as an oracle empty-row mismatch."""
    repo = "test-owner/smoke-batched-aborted"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        tokens=[],
        batched_events=[
            BatchEvent.token(req_index=0, token_id=11),
            BatchEvent.aborted(req_index=1, reason="budget-exhausted"),
            BatchEvent.done(req_index=0, reason="max_tokens"),
        ],
    )
    scenario = _smoke_batched_scenario(
        scenario_id="smoke-batched-aborted", repo=repo
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "smoke_batched_aborted:row=1" in result.reason
    assert "reason=budget-exhausted" in result.reason


def test_smoke_batched_prefix_cache_reaches_engine(
    fake_home_cache: Path,
) -> None:
    """When workload.prefix_cache=True, runner must wire a
    RadixPrefixCache into generate_batch. Verified by recording
    the kwarg the engine received."""
    repo = "test-owner/smoke-batched-pc"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    per_row = {0: [11, 12], 1: [21, 22]}
    observed_kwargs: dict[str, Any] = {}

    class _PCSpy(_FakeEngine):
        def generate_batch(
            self, prompts: Any, params: Any, **kwargs: Any
        ) -> Any:
            observed_kwargs.update(kwargs)
            return _FakeEngine.generate_batch(
                self, prompts, params, **kwargs
            )

    engine = _PCSpy(
        tokens=[], batched_events=_bgt1_events_for(per_row)
    )
    scenario = _smoke_batched_scenario(
        scenario_id="smoke-batched-pc", repo=repo, prefix_cache=True
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "ok"
    from silica.kvcache.prefix import RadixPrefixCache

    pc = observed_kwargs.get("prefix_cache")
    assert pc is not None, "expected runner to pass a RadixPrefixCache"
    assert isinstance(pc, RadixPrefixCache)


def test_smoke_batched_prefix_cache_is_none_when_workload_flag_off(
    fake_home_cache: Path,
) -> None:
    """Workload.prefix_cache=False keeps prefix_cache=None on the
    generate_batch call — no accidental cache wiring."""
    repo = "test-owner/smoke-batched-no-pc"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    per_row = {0: [11, 12], 1: [21, 22]}
    observed_kwargs: dict[str, Any] = {}

    class _PCSpy(_FakeEngine):
        def generate_batch(
            self, prompts: Any, params: Any, **kwargs: Any
        ) -> Any:
            observed_kwargs.update(kwargs)
            return _FakeEngine.generate_batch(
                self, prompts, params, **kwargs
            )

    engine = _PCSpy(
        tokens=[], batched_events=_bgt1_events_for(per_row)
    )
    scenario = _smoke_batched_scenario(
        scenario_id="smoke-batched-no-pc", repo=repo, prefix_cache=False
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "ok"
    assert observed_kwargs.get("prefix_cache") is None


def test_result_schema_is_flat(fake_home_cache: Path) -> None:
    """Sanity check: ScenarioResult fields are JSON-serialisable.
    Regression guard in case metadata picks up non-JSON types later."""
    repo = "test-owner/schema-check"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=50)
    engine = _FakeEngine([7])
    scenario = _scenario(scenario_id="schema", repo=repo)
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: 256.5,
    )
    [result] = runner.run([scenario])

    from dataclasses import asdict

    json.dumps(asdict(result))  # must not raise


def test_scenario_result_dataclass_exposes_expected_fields() -> None:
    """Lock the ScenarioResult schema so any accidental reshape is
    caught in unit tests before a bench run notices."""
    result = ScenarioResult(scenario_id="x", status="ok")
    assert result.scenario_id == "x"
    assert result.status == "ok"
    assert result.ttft_ms is None
    assert result.peak_memory_mb is None
    assert result.metadata == {}
