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
        # One id per character, modulo vocab so all ids stay in
        # range. Gives teacher-forced tests direct control over the
        # tokenized length — e.g. encode("x" * 100) returns a
        # 100-id list.
        if not text:
            return []
        return [(ord(c) % max(1, self.vocab_size)) for c in text]

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
        length_spread_threshold: float = 2.0,
    ) -> Iterator[BatchEvent]:
        # ``length_spread_threshold`` is accepted but ignored by the
        # fake engine: the bench harness passes ``float('inf')`` on
        # the BGT1 parity path (P-4.5-B.1 opt-out, see
        # ``silica/bench/runner.py::_collect_bgt1_batched_tokens``),
        # and this test's fakes script events directly without going
        # through the admission-reorder code path. Accept the kwarg
        # so the runner's call signature matches what the real
        # Engine exposes; ignore it in the body.
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


def test_b1_parity_missing_done_fails_before_oracle(
    fake_home_cache: Path,
) -> None:
    """Stream that closes without a ``done`` event — even when the
    emitted tokens happen to equal the single-request reference —
    must surface as a runner failure, not a silent parity pass.
    Parallels the BGT1 / SMOKE-batched ``rows_never_completed``
    checks."""
    repo = "test-owner/b1-missing-done"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine(
        [1, 2, 3],
        # Same tokens as the default single-request reference, but
        # the stream omits ``done`` entirely. Without the new
        # check, the oracle would see matching tokens and pass.
        batched_events=[
            BatchEvent.token(req_index=0, token_id=1),
            BatchEvent.token(req_index=0, token_id=2),
            BatchEvent.token(req_index=0, token_id=3),
        ],
    )
    scenario = _scenario(
        scenario_id="b1-missing-done",
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
    assert "b1_batched_never_completed" in result.reason


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
    tokens summed across rows, status ok, plus a per-row
    first_token_ms_offset (runner-instrumented TTFT — Engine's
    batched path does not populate MetricsRegistry)."""
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
    # Per-row first-token wall offsets must be present and
    # monotonic non-negative — the exact values depend on wall
    # time (test uses real perf_counter) so we only pin shape.
    for row_entry in rows:
        assert "first_token_ms_offset" in row_entry
        assert row_entry["first_token_ms_offset"] >= 0


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


# ---------- TEACHER_FORCED_ARGMAX -------------------------------------


def _tf_scenario(
    *,
    scenario_id: str,
    repo: str,
    target_continuation: str | None = " Paris.",
    min_agreement_rate: float = 0.98,
    prompt: str = "The capital of France is",
) -> Scenario:
    oracle_config: dict[str, Any] = {}
    if target_continuation is not None:
        oracle_config["target_continuation"] = target_continuation
    oracle_config["min_agreement_rate"] = min_agreement_rate
    return Scenario(
        id=scenario_id,
        repo=repo,
        workload=Workload(
            name="fake-tf",
            prompts=(prompt,),
            max_tokens=1,
            max_batch_size=1,
        ),
        oracle=OracleKind.TEACHER_FORCED_ARGMAX,
        oracle_config=oracle_config,
    )


def test_teacher_forced_argmax_happy_path(fake_home_cache: Path) -> None:
    """Silica and reference both return identical argmax lists of
    equal length; oracle returns ok with agreement_rate=1.0."""
    repo = "test-owner/tf-happy"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine([])
    # target_continuation=" Paris." tokenises to 7 ids via the
    # per-character fake tokenizer, so prediction lists must
    # be length 7 to match.
    predictions = [10, 20, 30, 40, 50, 60, 70]

    def fake_silica(
        a: Any, e: Any, prompt_ids: list[int], target_ids: list[int]
    ) -> list[int]:
        assert len(target_ids) == len(predictions)
        return list(predictions)

    def fake_reference(
        a: Any, prompt_ids: list[int], target_ids: list[int]
    ) -> list[int]:
        return list(predictions)

    scenario = _tf_scenario(scenario_id="tf-happy", repo=repo)
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        teacher_forced_silica=fake_silica,
        teacher_forced_reference=fake_reference,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "ok"
    assert result.reason is None
    assert result.total_tokens == len(predictions)
    assert result.metadata["agreement_rate"] == 1.0
    assert result.metadata["first_mismatch_index"] == -1
    assert result.metadata["length"] == len(predictions)
    assert result.metadata["matches"] == len(predictions)


def test_teacher_forced_argmax_fail_below_threshold(
    fake_home_cache: Path,
) -> None:
    """With 1 match out of 4 positions, agreement_rate=0.25 which
    is below the 0.98 threshold → oracle fails with a structured
    reason."""
    repo = "test-owner/tf-fail"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine([])
    # Target "abcd" tokenises to length 4 via per-character fake.
    silica = [10, 99, 98, 97]
    reference = [10, 20, 30, 40]  # matches silica only at index 0

    def fake_silica(*args: Any) -> list[int]:
        return list(silica)

    def fake_reference(*args: Any) -> list[int]:
        return list(reference)

    scenario = _tf_scenario(
        scenario_id="tf-fail", repo=repo, target_continuation="abcd"
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        teacher_forced_silica=fake_silica,
        teacher_forced_reference=fake_reference,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert result.reason.startswith(
        "teacher_forced_argmax_agreement_below_threshold:"
    )
    assert result.metadata["agreement_rate"] == 0.25
    assert result.metadata["first_mismatch_index"] == 1
    assert result.metadata["matches"] == 1


def test_teacher_forced_argmax_exact_threshold_is_pass(
    fake_home_cache: Path,
) -> None:
    """Agreement rate exactly equal to threshold must pass (>=)."""
    repo = "test-owner/tf-threshold"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine([])
    # 98 matches / 100 positions -> 0.98 exactly
    reference = list(range(100))
    silica = list(reference)
    silica[50] = 9999
    silica[75] = 9999

    def fake_silica(*args: Any) -> list[int]:
        return list(silica)

    def fake_reference(*args: Any) -> list[int]:
        return list(reference)

    scenario = _tf_scenario(
        scenario_id="tf-threshold",
        repo=repo,
        # 100 chars -> 100 tokenised ids via per-char fake tokenizer,
        # so predictions line up with the 100 silica/reference slots.
        target_continuation="x" * 100,
        min_agreement_rate=0.98,
    )
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        teacher_forced_silica=fake_silica,
        teacher_forced_reference=fake_reference,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "ok"
    assert result.metadata["agreement_rate"] == pytest.approx(0.98)
    assert result.metadata["matches"] == 98


def test_teacher_forced_argmax_missing_oracle_config_target_fails(
    fake_home_cache: Path,
) -> None:
    """A scenario that forgot oracle_config['target_continuation']
    is authoring-broken; runner surfaces the specific missing key
    rather than crashing inside the silica hook."""
    repo = "test-owner/tf-no-target"
    _create_cache_dir(repo)

    adapter = _FakeAdapter(vocab_size=100)
    engine = _FakeEngine([])
    scenario = _tf_scenario(
        scenario_id="tf-no-target", repo=repo, target_continuation=None
    )

    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert (
        "teacher_forced_argmax_missing_oracle_config_target_continuation"
        in result.reason
    )


def test_teacher_forced_argmax_rejects_batched_workload(
    fake_home_cache: Path,
) -> None:
    """Teacher-forced is strictly single-request — B>1 is an
    authoring error, not a runner branch."""
    repo = "test-owner/tf-batched"
    _create_cache_dir(repo)

    called = {"n": 0}

    def factory(scenario: Any) -> Any:
        called["n"] += 1
        raise AssertionError("factory must not run for authoring errors")

    scenario = Scenario(
        id="tf-batched",
        repo=repo,
        workload=Workload(
            name="bad",
            prompts=("a", "b"),
            max_tokens=1,
            max_batch_size=2,
        ),
        oracle=OracleKind.TEACHER_FORCED_ARGMAX,
        oracle_config={"target_continuation": "x", "min_agreement_rate": 0.98},
    )
    runner = BenchRunner(
        engine_factory=factory,
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
    )
    [result] = runner.run([scenario])

    assert result.status == "failed"
    assert result.reason is not None
    assert "max_batch_size=1" in result.reason
    assert called["n"] == 0


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


# ---------- P-5-C.4 step 1 — multi-seed fan-out -----------------------


class TestSeedDefaultAndRangeValidation:
    def test_empty_seeds_list_rejected_at_init(
        self, fake_home_cache: Path
    ) -> None:
        """``seeds=()`` is a nonsense request — no work to do but the
        caller still expected a result list. Fail fast at
        construction time rather than silently producing [] later."""
        with pytest.raises(ValueError, match="at least one seed"):
            BenchRunner(seeds=())

    def test_negative_seed_rejected_at_init(self) -> None:
        """Direct programmatic callers (tests, future library users)
        must not be able to smuggle a seed into ``_run_one`` that
        NumPy / MLX would reject at ``seed()`` time. That call
        happens BEFORE the per-scenario ``try:`` boundary, so an
        out-of-range seed would crash the whole run rather than
        collapsing to one ``status='failed'`` row. Mirror the CLI's
        ``_parse_seeds`` range contract here."""
        with pytest.raises(ValueError, match="out of range"):
            BenchRunner(seeds=(-1,))

    def test_seed_at_2_to_32_rejected_at_init(self) -> None:
        """NumPy accepts seeds in ``[0, 2**32)``. The upper endpoint
        itself is out of range."""
        with pytest.raises(ValueError, match="out of range"):
            BenchRunner(seeds=(1 << 32,))

    def test_non_int_seed_rejected_at_init(self) -> None:
        """Floats / strings would drift silently under numpy's
        implicit conversion; reject them clearly at the API
        boundary."""
        with pytest.raises(TypeError, match="must be int"):
            BenchRunner(seeds=(3.14,))  # type: ignore[arg-type]

    def test_bool_seed_rejected_at_init(self) -> None:
        """``isinstance(True, int)`` is True in Python, so booleans
        would be silently accepted as seed 0 / seed 1 otherwise —
        almost always a caller bug to mix bools with ints in a
        seed list."""
        with pytest.raises(TypeError, match="must be int"):
            BenchRunner(seeds=(True,))  # type: ignore[list-item]

    def test_upper_bound_seed_accepted_at_init(self) -> None:
        """``2**32 - 1`` is the last legal NumPy seed and must
        construct without complaint."""
        BenchRunner(seeds=((1 << 32) - 1,))

    def test_default_seeds_tuple_is_zero(
        self, fake_home_cache: Path
    ) -> None:
        """Default ``seeds=(0,)`` keeps pre-C.4 single-row behaviour
        for every CLI invocation that does not pass ``--seeds``."""
        repo = "test-owner/default-seed"
        _create_cache_dir(repo)
        adapter = _FakeAdapter(vocab_size=50)
        engine = _FakeEngine([7, 8])
        scenario = _scenario(scenario_id="default-seed", repo=repo)

        runner = BenchRunner(
            engine_factory=_factory_returning(adapter, engine),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
        )
        [result] = runner.run([scenario])
        assert result.metadata["seed"] == 0


class TestScenarioMajorOrdering:
    def test_two_scenarios_two_seeds_expand_scenario_major(
        self, fake_home_cache: Path
    ) -> None:
        """``run([A, B])`` with ``seeds=(42, 43)`` returns
        ``[(A,42), (A,43), (B,42), (B,43)]``. Same-scenario rows stay
        adjacent so downstream aggregation can consume contiguous
        slices without a sort pass."""
        repo_a = "test-owner/scen-a"
        repo_b = "test-owner/scen-b"
        _create_cache_dir(repo_a)
        _create_cache_dir(repo_b)

        # Each call to the factory returns a fresh (adapter, engine)
        # pair so the four (scenario, seed) executions do not share
        # token-stream state from previous invocations.
        def factory(scenario: Scenario) -> Any:
            return _FakeAdapter(vocab_size=50), _FakeEngine([1, 2])

        scenarios = [
            _scenario(scenario_id="scen-a", repo=repo_a),
            _scenario(scenario_id="scen-b", repo=repo_b),
        ]
        runner = BenchRunner(
            engine_factory=factory,
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(42, 43),
        )
        results = runner.run(scenarios)

        observed = [(r.scenario_id, r.metadata["seed"]) for r in results]
        assert observed == [
            ("scen-a", 42),
            ("scen-a", 43),
            ("scen-b", 42),
            ("scen-b", 43),
        ]
        for r in results:
            assert r.status == "ok"

    def test_three_seeds_produce_three_rows_per_scenario(
        self, fake_home_cache: Path
    ) -> None:
        """Fan-out cardinality: N scenarios × M seeds → N*M rows."""
        repo = "test-owner/one-scen"
        _create_cache_dir(repo)

        def factory(scenario: Scenario) -> Any:
            return _FakeAdapter(vocab_size=50), _FakeEngine([9])

        scenario = _scenario(scenario_id="one-scen", repo=repo)
        runner = BenchRunner(
            engine_factory=factory,
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(1, 2, 3),
        )
        results = runner.run([scenario])
        assert len(results) == 3
        assert [r.metadata["seed"] for r in results] == [1, 2, 3]


class TestEarlyReturnPathsCarrySeed:
    """C.4 step 1 Q2 guard (runner-side): every early-return path in
    ``_run_one`` must set ``metadata["seed"]`` so the raw JSONL stays
    strictly per-``(scenario, seed)`` self-describing, not just the
    success path.
    """

    def test_gate_skipped_row_carries_seed(
        self, fake_home_cache: Path
    ) -> None:
        """Cache-missing skip path. Factory never runs, but the
        resulting ScenarioResult still carries the seed of the
        (scenario, seed) pair that was attempted."""
        scenario = _scenario(
            scenario_id="gate-skip",
            repo="test-owner/never-cached",
        )
        runner = BenchRunner(
            engine_factory=lambda _s: (_ for _ in ()).throw(
                AssertionError("factory must not run")
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(7, 11),
        )
        results = runner.run([scenario])
        assert len(results) == 2
        assert [r.status for r in results] == ["skipped", "skipped"]
        assert [r.metadata["seed"] for r in results] == [7, 11]

    def test_workload_validation_failed_row_carries_seed(
        self, fake_home_cache: Path
    ) -> None:
        """Authoring-error path. B1_PARITY_VS_SINGLE with two prompts
        fails ``_validate_workload_for_oracle`` before any factory
        call; each per-seed row must still carry ``metadata["seed"]``."""
        repo = "test-owner/bad-authoring"
        _create_cache_dir(repo)

        scenario = Scenario(
            id="bad-authoring",
            repo=repo,
            workload=Workload(
                name="fake",
                prompts=("one", "two"),
                max_tokens=1,
                max_batch_size=1,
            ),
            oracle=OracleKind.B1_PARITY_VS_SINGLE,
        )
        runner = BenchRunner(
            engine_factory=lambda _s: (_ for _ in ()).throw(
                AssertionError("factory must not run on authoring error")
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(5, 6, 7),
        )
        results = runner.run([scenario])
        assert [r.status for r in results] == ["failed", "failed", "failed"]
        assert [r.metadata["seed"] for r in results] == [5, 6, 7]

    def test_exception_failed_row_carries_seed(
        self, fake_home_cache: Path
    ) -> None:
        """Exception path: factory raises mid-execution, runner's
        outer ``except Exception`` catches and builds a failed row
        which must still carry the seed (the exception is orthogonal
        to which seed was in flight)."""
        repo = "test-owner/factory-explodes"
        _create_cache_dir(repo)

        def exploding_factory(_s: Scenario) -> Any:
            raise RuntimeError("synthetic engine-factory failure")

        scenario = _scenario(scenario_id="boom", repo=repo)
        runner = BenchRunner(
            engine_factory=exploding_factory,
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(100, 200),
        )
        results = runner.run([scenario])
        assert [r.status for r in results] == ["failed", "failed"]
        for r, expected_seed in zip(results, (100, 200)):
            assert r.metadata["seed"] == expected_seed
            assert "RuntimeError" in (r.reason or "")
            assert "synthetic engine-factory failure" in (r.reason or "")


class TestRngSeedingActuallyFires:
    """C.4 step 1 Q3 guard: ``_run_one`` must seed ``mx.random``,
    ``np.random``, AND ``random`` — not just label the row. A future
    oracle that consumes RNG would silently non-determinise otherwise.
    """

    def test_numpy_random_seeded_per_seed(
        self, fake_home_cache: Path
    ) -> None:
        """Observe ``np.random.random()`` inside the injected
        factory: the value is deterministic for a given seed and
        differs between seeds. If ``_run_one`` skipped
        ``np.random.seed``, both factory calls would draw from
        whatever NumPy state the process inherited — almost certainly
        identical only by accident."""
        import numpy as np

        repo = "test-owner/np-rng"
        _create_cache_dir(repo)

        observed: list[float] = []

        def factory(_s: Scenario) -> Any:
            observed.append(float(np.random.random()))
            return _FakeAdapter(vocab_size=50), _FakeEngine([0])

        scenario = _scenario(scenario_id="np-rng", repo=repo)
        runner = BenchRunner(
            engine_factory=factory,
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(42, 43, 42),  # seed 42 twice — deterministic
        )
        # Deduplication is a CLI-layer concern; BenchRunner runs
        # whatever seed tuple it receives. Feeding (42, 43, 42)
        # here is how we prove the RNG is actually re-seeded per
        # (scenario, seed) pair rather than seeded once at
        # construction.
        runner.run([scenario])
        assert len(observed) == 3
        assert observed[0] == observed[2], (
            "np.random.seed(42) must produce the same draw every "
            "time; got different values, so the runner either "
            "did not re-seed or consumed RNG state between calls"
        )
        assert observed[0] != observed[1], (
            "np.random.seed(42) and np.random.seed(43) must "
            "produce distinct draws; identical values indicate the "
            "seed is not actually being applied"
        )

    def test_stdlib_random_seeded_per_seed(
        self, fake_home_cache: Path
    ) -> None:
        """Mirror coverage for ``random.seed`` — the user's Q3
        amendment explicitly called this out because future CLI-layer
        prompt sampling is likely to reach for stdlib ``random``
        before NumPy."""
        import random as stdlib_random

        repo = "test-owner/stdlib-rng"
        _create_cache_dir(repo)

        observed: list[float] = []

        def factory(_s: Scenario) -> Any:
            observed.append(stdlib_random.random())
            return _FakeAdapter(vocab_size=50), _FakeEngine([0])

        scenario = _scenario(scenario_id="stdlib-rng", repo=repo)
        runner = BenchRunner(
            engine_factory=factory,
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(1, 2, 1),
        )
        runner.run([scenario])
        assert len(observed) == 3
        assert observed[0] == observed[2]
        assert observed[0] != observed[1]


class TestSuccessPathMetadataPrecedence:
    """Runner-injected seed wins if an oracle accidentally writes the
    same key into its metadata. Prevents authoring-dimension leakage
    via an oracle that reads ``oracle_config`` and echoes it back."""

    def test_runner_seed_overrides_oracle_metadata(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Patch the SMOKE oracle to inject a bogus ``seed`` key;
        the runner's ``{**metadata, "seed": seed}`` merge must
        overwrite it. If the merge order inverts, the bogus value
        would leak into the JSONL."""
        from silica.bench import oracles as oracles_module

        real_smoke_oracle = oracles_module.ORACLES[OracleKind.SMOKE]

        def tainted_smoke(
            scenario: Scenario, output: Any, context: Any
        ) -> tuple[bool, str | None, dict[str, Any]]:
            ok, reason, metadata = real_smoke_oracle(
                scenario, output, context
            )
            return ok, reason, {**metadata, "seed": -999}

        monkeypatch.setitem(
            oracles_module.ORACLES, OracleKind.SMOKE, tainted_smoke
        )

        repo = "test-owner/precedence"
        _create_cache_dir(repo)
        scenario = _scenario(scenario_id="precedence", repo=repo)
        runner = BenchRunner(
            engine_factory=_factory_returning(
                _FakeAdapter(vocab_size=50), _FakeEngine([1])
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(77,),
        )
        [result] = runner.run([scenario])
        assert result.status == "ok"
        assert result.metadata["seed"] == 77, (
            f"runner seed 77 must win over oracle-written seed -999; "
            f"got {result.metadata['seed']}"
        )


# ---------- P-5-C.5 step 1 — --kv-codec override --------------------


class TestCodecOverridesConstructorValidation:
    def test_empty_codec_overrides_rejected(self) -> None:
        """Empty list mirrors the seeds=() rejection shape."""
        with pytest.raises(ValueError, match="at least one codec"):
            BenchRunner(codec_overrides=())

    def test_unknown_codec_id_rejected_at_init(self) -> None:
        """Typoed codec id must fail at construction, not after N
        scenarios have already run. Mirror of the seed range-check
        contract — direct programmatic callers (tests, notebooks)
        get a clean ValueError instead of a deep
        _maybe_build_prefix_cache failure."""
        with pytest.raises(ValueError, match="not a registered codec id"):
            BenchRunner(codec_overrides=("not-a-real-codec",))

    def test_non_str_non_none_entry_rejected(self) -> None:
        """``int`` / ``bool`` / float etc. are caller bugs."""
        with pytest.raises(TypeError, match="str \\| None"):
            BenchRunner(codec_overrides=(42,))  # type: ignore[arg-type]

    def test_registered_codec_ids_accepted(self) -> None:
        """Every id in CODEC_REGISTRY constructs without complaint,
        including mixed-None entries for fan-out over 'baked + one
        override'."""
        BenchRunner(codec_overrides=("fp16", "block_tq_b64_b4"))
        BenchRunner(codec_overrides=(None, "block_tq_b64_b4"))
        BenchRunner(codec_overrides=(None,))


class TestCodecOverrideNoneDefault:
    def test_none_override_preserves_baked_codec_in_metadata(
        self, fake_home_cache: Path
    ) -> None:
        """``codec_override=None`` means 'use scenario's baked codec'.
        For a scenario whose Workload does not set kv_codec, the
        metadata carries ``codec_id=None`` — the key still exists
        so JSONL consumers have a stable field."""
        repo = "test-owner/default-codec"
        _create_cache_dir(repo)

        adapter = _FakeAdapter(vocab_size=50)
        engine = _FakeEngine([7, 8])
        scenario = _scenario(scenario_id="default-codec", repo=repo)

        runner = BenchRunner(
            engine_factory=_factory_returning(adapter, engine),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
        )
        [result] = runner.run([scenario])
        assert result.metadata["codec_id"] is None
        # seed still present — execution dimensions compose cleanly
        assert result.metadata["seed"] == 0


class TestCodecOverrideApplied:
    def _prefix_cache_scenario(self, scenario_id: str, repo: str) -> Scenario:
        """Scenario shape that ACCEPTS a codec override: prefix_cache
        enabled + workload set to the scheduler-friendly B=1,2-prompt
        shape so the override does not hit the Workload guard."""
        _create_cache_dir(repo)
        return Scenario(
            id=scenario_id,
            repo=repo,
            workload=Workload(
                name="codec-override-fake",
                prompts=("p",),
                max_tokens=0,
                max_batch_size=1,
                prefix_cache=True,
                kv_codec="fp16",  # baked baseline to be overridden
            ),
            oracle=OracleKind.STORAGE,
        )

    def test_override_codec_id_flows_into_workload_invalid_row(
        self, fake_home_cache: Path
    ) -> None:
        """Override is applied BEFORE ``_validate_workload_for_oracle``
        in ``_run_one``, so if workload validation subsequently
        fails (here: STORAGE needs 2 prompts, scenario has 1),
        ``metadata['codec_id']`` still carries the attempted
        override — a log reader chasing a failed row sees which
        codec was in flight when the row crashed.

        Complementary to ``TestCodecOverridePreservesOtherWorkloadFields``
        which verifies the override flows through the SUCCESS path;
        this test pins the identity on the validation-failed path."""
        scenario = self._prefix_cache_scenario(
            "codec-override", "test-owner/codec-override"
        )
        # The oracle won't actually run (prompts=1 fails STORAGE
        # validation); we just need the metadata to carry the
        # codec id from the override phase of _run_one, which
        # happens BEFORE workload validation.
        runner = BenchRunner(
            engine_factory=lambda _s: (_ for _ in ()).throw(
                AssertionError("factory should not be reached")
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            codec_overrides=("block_tq_b64_b4",),
        )
        [result] = runner.run([scenario])
        assert result.status == "failed"
        assert result.reason is not None
        assert "requires exactly 2 prompts" in result.reason
        # Load-bearing assertion: codec_id = override, not scenario's
        # baked "fp16". If override were applied after validation,
        # this would still be "fp16".
        assert result.metadata["codec_id"] == "block_tq_b64_b4"


class TestCodecOverrideIncompatibleScenario:
    def test_override_on_no_prefix_cache_scenario_fails_cleanly(
        self, fake_home_cache: Path
    ) -> None:
        """Applying ``--kv-codec X`` to a SMOKE scenario (no prefix
        cache) triggers ``Workload.__post_init__`` rejecting
        ``kv_codec != None`` without ``prefix_cache=True``. Runner
        collapses the ValueError to a failed row; metadata still
        carries both seed and the attempted codec_id so a log
        reader can diagnose the mismatch."""
        repo = "test-owner/no-prefix-cache"
        _create_cache_dir(repo)
        scenario = _scenario(scenario_id="no-prefix", repo=repo)  # SMOKE, prefix_cache=False

        runner = BenchRunner(
            engine_factory=lambda _s: (_ for _ in ()).throw(
                AssertionError("factory must not run when override fails")
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            codec_overrides=("block_tq_b64_b4",),
        )
        [result] = runner.run([scenario])
        assert result.status == "failed"
        assert result.reason is not None
        assert "codec_override_invalid" in result.reason
        assert "prefix_cache" in result.reason
        assert result.metadata["codec_id"] == "block_tq_b64_b4"
        assert result.metadata["seed"] == 0


class TestCodecIdMetadataOnAllReturnPaths:
    """Mirror of C.4 step 1's seed-on-every-return-path guard, for
    the new codec_id key. JSONL stays self-describing if, for
    example, one seed skips (missing cache) and the next seed
    succeeds — both rows must carry codec_id."""

    def test_gate_skipped_row_carries_codec_id(
        self, fake_home_cache: Path
    ) -> None:
        """Cache-missing gate skip path. No override is applied
        (codec_override=None default) and the scenario has no baked
        codec, so metadata['codec_id'] == None. Key still present."""
        scenario = _scenario(
            scenario_id="gate-skip-codec",
            repo="test-owner/never-cached-codec",
        )
        runner = BenchRunner(
            engine_factory=lambda _s: (_ for _ in ()).throw(
                AssertionError("factory must not run")
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
        )
        [result] = runner.run([scenario])
        assert result.status == "skipped"
        assert "codec_id" in result.metadata
        assert result.metadata["codec_id"] is None

    def test_exception_failed_row_carries_codec_id(
        self, fake_home_cache: Path
    ) -> None:
        """Factory raises mid-execution; the except-Exception
        boundary still produces a row with codec_id in metadata."""
        repo = "test-owner/exploding-codec"
        _create_cache_dir(repo)
        scenario = _scenario(scenario_id="boom-codec", repo=repo)

        def exploding_factory(_s: Scenario) -> Any:
            raise RuntimeError("synthetic factory failure for codec test")

        runner = BenchRunner(
            engine_factory=exploding_factory,
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
        )
        [result] = runner.run([scenario])
        assert result.status == "failed"
        assert "codec_id" in result.metadata


class TestCodecOverridePreservesOtherWorkloadFields:
    def test_override_does_not_alter_prompts_max_tokens_batch_size(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``replace(workload, kv_codec=...)`` edit must NOT
        clobber any other workload field. Use a stub oracle that
        echoes the workload shape back through metadata so we can
        inspect it after the run."""
        from silica.bench import oracles as oracles_module

        repo = "test-owner/workload-preservation"
        _create_cache_dir(repo)
        scenario = Scenario(
            id="preserve-fields",
            repo=repo,
            workload=Workload(
                name="preserve-workload",
                prompts=("p0",),
                max_tokens=7,
                max_batch_size=1,
                prefix_cache=True,
                kv_codec="fp16",
                temperature=0.3,
                top_p=0.9,
            ),
            oracle=OracleKind.SMOKE,
        )

        def stub_smoke(
            sc: Scenario, output: Any, context: Any
        ) -> tuple[bool, str | None, dict[str, Any]]:
            return True, None, {
                "echoed_kv_codec": sc.workload.kv_codec,
                "echoed_max_tokens": sc.workload.max_tokens,
                "echoed_prompts_len": len(sc.workload.prompts),
                "echoed_temperature": sc.workload.temperature,
                "echoed_top_p": sc.workload.top_p,
                "echoed_max_batch_size": sc.workload.max_batch_size,
                "echoed_prefix_cache": sc.workload.prefix_cache,
            }

        monkeypatch.setitem(
            oracles_module.ORACLES, OracleKind.SMOKE, stub_smoke
        )

        adapter = _FakeAdapter(vocab_size=50)
        engine = _FakeEngine([1, 2])
        runner = BenchRunner(
            engine_factory=_factory_returning(adapter, engine),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            codec_overrides=("block_tq_b64_b4",),
        )
        [result] = runner.run([scenario])

        # The override replaced only kv_codec; everything else stayed.
        md = result.metadata
        assert md["echoed_kv_codec"] == "block_tq_b64_b4"
        assert md["echoed_max_tokens"] == 7
        assert md["echoed_prompts_len"] == 1
        assert md["echoed_temperature"] == 0.3
        assert md["echoed_top_p"] == 0.9
        assert md["echoed_max_batch_size"] == 1
        assert md["echoed_prefix_cache"] is True
        # And the runner-injected codec_id reflects the override.
        assert md["codec_id"] == "block_tq_b64_b4"


class TestCodecFanOutOrdering:
    def test_scenario_major_codec_middle_seed_innermost(
        self, fake_home_cache: Path
    ) -> None:
        """Result order under mixed fan-out: for scenarios [A, B],
        codec_overrides=[None, None], seeds=[0, 1] the order is
        scenario-major with codec middle and seed innermost.

        Step 1 only exercises cardinality 1 for codec_overrides in
        the CLI, but the underlying order invariant must already
        hold so step 2 (--all-kv-codecs) can lift it without
        re-checking."""
        repo_a = "test-owner/fan-a"
        repo_b = "test-owner/fan-b"
        _create_cache_dir(repo_a)
        _create_cache_dir(repo_b)

        def factory(_s: Scenario) -> Any:
            return _FakeAdapter(vocab_size=50), _FakeEngine([1, 2])

        scenarios = [
            _scenario(scenario_id="fan-a", repo=repo_a),
            _scenario(scenario_id="fan-b", repo=repo_b),
        ]
        runner = BenchRunner(
            engine_factory=factory,
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(0, 1),
            codec_overrides=(None, None),
        )
        results = runner.run(scenarios)
        # 2 scenarios * 2 codecs * 2 seeds = 8 rows
        observed = [
            (r.scenario_id, r.metadata["codec_id"], r.metadata["seed"])
            for r in results
        ]
        # scenario-major, codec middle, seed inner
        assert observed == [
            ("fan-a", None, 0),
            ("fan-a", None, 1),
            ("fan-a", None, 0),
            ("fan-a", None, 1),
            ("fan-b", None, 0),
            ("fan-b", None, 1),
            ("fan-b", None, 0),
            ("fan-b", None, 1),
        ]


class TestJsonlMultiSeedRoundTrip:
    def test_jsonl_one_row_per_scenario_seed(
        self, fake_home_cache: Path, tmp_path: Path
    ) -> None:
        """Every row appended to the JSONL carries its seed. Reading
        the file back yields ``N * M`` rows with the same
        scenario-major ordering the runner produces in memory."""
        repo = "test-owner/jsonl-seed"
        _create_cache_dir(repo)

        def factory(_s: Scenario) -> Any:
            return _FakeAdapter(vocab_size=50), _FakeEngine([1, 2])

        scenario = _scenario(scenario_id="jsonl-seed", repo=repo)
        out_path = tmp_path / "results.jsonl"
        runner = BenchRunner(
            engine_factory=factory,
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(10, 11, 12),
            out_path=out_path,
        )
        runner.run([scenario])

        raw = out_path.read_text(encoding="utf-8").splitlines()
        assert len(raw) == 3
        rows = [json.loads(line) for line in raw]
        assert [row["metadata"]["seed"] for row in rows] == [10, 11, 12]
        assert all(row["scenario_id"] == "jsonl-seed" for row in rows)


# ---------- P-5-C.6 step 1 — vqbench xcheck plumbing ----------------


from silica.bench.scenario import VqbenchXcheckSpec  # noqa: E402
from silica.bench.vqbench_baseline import VqbenchBaselineResult  # noqa: E402


class TestScenarioVqbenchXcheckPPLGuard:
    """Scenario.__post_init__ enforces vqbench_xcheck only on PPL
    scenarios. This is authoring-time validation: mismatched oracle
    surfaces as ValueError at Scenario construction rather than a
    silent skip during a bench run."""

    def test_vqbench_xcheck_on_ppl_scenario_accepted(self) -> None:
        # The PPL oracle expects oracle_config["wikitext_path"]; not
        # required to exist for this __post_init__ test — we're
        # probing the oracle-kind guard, not running the oracle.
        Scenario(
            id="ppl-ok",
            repo="demo/repo",
            workload=Workload(
                name="w",
                prompts=(),
                max_tokens=0,
                max_batch_size=1,
                prefix_cache=True,
                kv_codec="fp16",
            ),
            oracle=OracleKind.PPL,
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/whatever.py",
                method="BlockTurboQuantMSE",
                bits=4,
            ),
        )

    @pytest.mark.parametrize(
        "oracle",
        [
            OracleKind.SMOKE,
            OracleKind.B1_PARITY_VS_SINGLE,
            OracleKind.BGT1_DIRECT_BATCHED_REFERENCE,
            OracleKind.TEACHER_FORCED_ARGMAX,
            OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT,
            OracleKind.STORAGE,
            OracleKind.ADMISSION_HEADROOM,
        ],
    )
    def test_vqbench_xcheck_on_non_ppl_scenario_rejected(
        self, oracle: OracleKind
    ) -> None:
        """Every non-PPL oracle kind must loud-fail at Scenario
        construction when vqbench_xcheck is set."""
        with pytest.raises(ValueError, match="only meaningful on OracleKind.PPL"):
            Scenario(
                id=f"bad-{oracle.value}",
                repo="demo/repo",
                workload=Workload(
                    name="w",
                    prompts=("p",),
                    max_tokens=1,
                    max_batch_size=1,
                ),
                oracle=oracle,
                vqbench_xcheck=VqbenchXcheckSpec(
                    script_path="/tmp/whatever.py",
                    method="Foo",
                    bits=4,
                ),
            )


def _ppl_scenario_with_xcheck(
    scenario_id: str,
    repo: str,
    *,
    wikitext_path: str,
    kv_codec: str = "fp16",
    vqbench_xcheck: VqbenchXcheckSpec | None = None,
) -> Scenario:
    """Build a minimal PPL scenario used by vqbench-xcheck runner
    tests. oracle_config carries wikitext_path / chunk_size /
    max_tokens so the real ``_run_ppl`` would find them — but the
    tests inject a stubbed oracle so the real path never runs."""
    return Scenario(
        id=scenario_id,
        repo=repo,
        workload=Workload(
            name=f"{scenario_id}-wl",
            prompts=(),
            max_tokens=0,
            max_batch_size=1,
            prefix_cache=True,
            kv_codec=kv_codec,
        ),
        oracle=OracleKind.PPL,
        oracle_config={
            "wikitext_path": wikitext_path,
            "chunk_size": 128,
            "max_tokens": 256,
        },
        vqbench_xcheck=vqbench_xcheck,
    )


def _stub_ppl_run(
    _scenario: Scenario,
    _adapter: Any,
    *,
    fp16_baseline_ppl: float | None = None,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stub for silica.bench.runner._run_ppl — returns a success
    payload + oracle_context matching the real shape so the runner
    can proceed through the ``ok`` branch without loading wikitext
    or a real model. Used via monkeypatch in vqbench xcheck tests.

    Accepts the C.6 step 2 ``fp16_baseline_ppl`` kwarg transparently:
    when the runner threads a baseline PPL in, the stub echoes it
    into ``oracle_context["ppl_fp16"]`` so ``ppl_oracle`` receives
    a populated context and computes ``delta_ppl`` /
    ``delta_ppl_pct`` metadata downstream — matching what the real
    ``_run_ppl`` does. Accepts the P-5-D.1 ``seed`` kwarg as well —
    the stub does not consume randomness, so the value is accepted
    for signature parity and discarded."""
    del seed
    context: dict[str, Any] = {
        "chunk_size": 128,
        "max_tokens": 256,
        "wikitext_path": "/nowhere",
        "kv_codec": _scenario.workload.kv_codec,
    }
    if fp16_baseline_ppl is not None:
        context["ppl_fp16"] = fp16_baseline_ppl
    return (
        {"nll_sum": 1.5, "n_tokens": 128, "ppl": 3.14},
        context,
    )


def _write_fake_wikitext(tmp_path: Path) -> str:
    """Create a tiny WikiText file under ``tmp_path`` so the PPL
    scenario's ``_check_gates`` passes its ``is_file()`` check.
    Returns the absolute path string the scenario authors would
    pass in ``oracle_config['wikitext_path']``."""
    p = tmp_path / "fake_wikitext.txt"
    p.write_text("hello world " * 200, encoding="utf-8")
    return str(p)


class TestVqbenchXcheckFlagOff:
    def test_no_vqbench_keys_in_metadata_when_flag_off(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default ``vqbench_xcheck_enabled=False`` — no vqbench_*
        keys appear in metadata regardless of scenario spec."""
        import silica.bench.runner as runner_module

        monkeypatch.setattr(runner_module, "_run_ppl", _stub_ppl_run)
        repo = "test-owner/xcheck-off"
        _create_cache_dir(repo)
        scenario = _ppl_scenario_with_xcheck(
            "xcheck-off",
            repo,
            wikitext_path=_write_fake_wikitext(fake_home_cache),
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/x.py", method="Foo", bits=4
            ),
        )

        adapter = _FakeAdapter(vocab_size=50)
        engine = _FakeEngine([])
        runner = BenchRunner(
            engine_factory=_factory_returning(adapter, engine),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
        )
        [result] = runner.run([scenario])
        assert result.status == "ok"
        # No vqbench_* keys should be present anywhere in metadata.
        assert not any(
            k.startswith("vqbench_") for k in result.metadata
        )


class TestVqbenchXcheckScenarioNotDeclared:
    def test_flag_on_no_spec_marks_scenario_not_declared(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag-on + scenario without ``vqbench_xcheck`` → status
        recorded as ``scenario_not_declared``. Distinguishes "no
        opt-in at scenario-author level" from "declared but silica
        failed"."""
        import silica.bench.runner as runner_module

        monkeypatch.setattr(runner_module, "_run_ppl", _stub_ppl_run)
        repo = "test-owner/xcheck-no-spec"
        _create_cache_dir(repo)
        scenario = _ppl_scenario_with_xcheck(
            "xcheck-no-spec",
            repo,
            wikitext_path=_write_fake_wikitext(fake_home_cache),
            vqbench_xcheck=None,
        )

        called: list[Any] = []

        def fake_vqbench(*args: Any, **kwargs: Any) -> VqbenchBaselineResult:
            called.append((args, kwargs))
            raise AssertionError("vqbench must not run when spec absent")

        runner = BenchRunner(
            engine_factory=_factory_returning(
                _FakeAdapter(vocab_size=50), _FakeEngine([])
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            vqbench_xcheck_enabled=True,
            vqbench_runner=fake_vqbench,
        )
        [result] = runner.run([scenario])
        assert result.status == "ok"
        assert result.metadata["vqbench_status"] == "scenario_not_declared"
        assert called == []


class TestVqbenchXcheckSilicaOracleNotOk:
    def test_silica_gate_skipped_row_records_silica_oracle_not_ok(
        self, fake_home_cache: Path
    ) -> None:
        """Scenario with xcheck declared but gate-skipped (cache
        missing) → ``silica_oracle_not_ok``. Silica never ran, so
        vqbench comparison would be meaningless."""
        # Cache missing for this repo; gate skip fires.
        scenario = _ppl_scenario_with_xcheck(
            "xcheck-cache-missing",
            "test-owner/never-cached-xcheck",
            wikitext_path="/nowhere",
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/x.py", method="Foo", bits=4
            ),
        )
        runner = BenchRunner(
            engine_factory=lambda _s: (_ for _ in ()).throw(
                AssertionError("factory must not run")
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            vqbench_xcheck_enabled=True,
            vqbench_runner=lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("vqbench must not run on silica-not-ok")
            ),
        )
        [result] = runner.run([scenario])
        assert result.status == "skipped"
        assert result.metadata["vqbench_status"] == "silica_oracle_not_ok"


class TestVqbenchXcheckOverrideDiffers:
    def test_codec_override_differs_from_baked_skips_with_diagnostic(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scenario baked with ``kv_codec='fp16'`` and vqbench_xcheck
        declared for that arm; user overrides with ``kv_codec='block_tq_b64_b4'``.
        Spec was authored for fp16's method/bits — running vqbench
        against block_tq would compare different codecs' PPL. Skip
        with ``override_differs_from_declared_arm`` + diagnostic
        fields naming both arms."""
        import silica.bench.runner as runner_module

        monkeypatch.setattr(runner_module, "_run_ppl", _stub_ppl_run)
        repo = "test-owner/xcheck-arm-mismatch"
        _create_cache_dir(repo)
        scenario = _ppl_scenario_with_xcheck(
            "xcheck-arm-mismatch",
            repo,
            wikitext_path=_write_fake_wikitext(fake_home_cache),
            kv_codec="fp16",
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/x.py",
                method="IdentityCodec",
                bits=16,
            ),
        )

        runner = BenchRunner(
            engine_factory=_factory_returning(
                _FakeAdapter(vocab_size=50), _FakeEngine([])
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            codec_overrides=("block_tq_b64_b4",),
            vqbench_xcheck_enabled=True,
            vqbench_runner=lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("vqbench must not run on arm mismatch")
            ),
        )
        [result] = runner.run([scenario])
        assert result.status == "ok"
        assert (
            result.metadata["vqbench_status"]
            == "override_differs_from_declared_arm"
        )
        assert result.metadata["vqbench_declared_arm"] == "fp16"
        assert result.metadata["vqbench_effective_arm"] == "block_tq_b64_b4"


class TestVqbenchXcheckOkPath:
    def test_all_conditions_met_invokes_vqbench_and_flattens_result(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: flag on + spec present + silica ok + arm
        matches. Runner invokes vqbench with auto-appended args and
        flattens the entire VqbenchBaselineResult into
        vqbench_-prefixed metadata."""
        import silica.bench.runner as runner_module

        monkeypatch.setattr(runner_module, "_run_ppl", _stub_ppl_run)
        repo = "test-owner/xcheck-happy"
        _create_cache_dir(repo)
        scenario = _ppl_scenario_with_xcheck(
            "xcheck-happy",
            repo,
            wikitext_path=_write_fake_wikitext(fake_home_cache),
            kv_codec="block_tq_b64_b4",
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/reproduce.py",
                method="BlockTurboQuantMSE",
                bits=4,
                extra_args=("--block-size", "64", "--patch-v"),
            ),
        )

        captured_args: list[Any] = []

        def fake_vqbench(
            script: Any,
            script_args: Any = None,
            **kwargs: Any,
        ) -> VqbenchBaselineResult:
            captured_args.append((script, list(script_args or []), kwargs))
            return VqbenchBaselineResult(
                status="ok",
                script=str(script),
                python_executable="/fake/python",
                script_args=tuple(script_args or ()),
                model="Qwen/Qwen3-0.6B",
                method="BlockTurboQuantMSE",
                bits=4,
                ppl_fp16=3.10,
                ppl_quant=3.15,
                delta_ppl=0.05,
                delta_pct=1.61,
                wall_s=42.0,
                returncode=0,
                stdout_tail="tail lines here",
            )

        runner = BenchRunner(
            engine_factory=_factory_returning(
                _FakeAdapter(vocab_size=50), _FakeEngine([])
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            seeds=(42,),
            vqbench_xcheck_enabled=True,
            vqbench_python="/fake/python",
            vqbench_runner=fake_vqbench,
        )
        [result] = runner.run([scenario])
        assert result.status == "ok"

        # Vqbench was invoked exactly once with the right shape.
        assert len(captured_args) == 1
        script, args, kwargs = captured_args[0]
        assert script == "/tmp/reproduce.py"
        # python_executable is the CLI-passed value, threaded via
        # BenchRunner's vqbench_python.
        assert kwargs.get("python_executable") == "/fake/python"
        # Auto-appended args in fixed order: model, seed, method,
        # bits, chunk, max-tokens, then extra_args.
        assert args == [
            "--model", repo,
            "--seed", "42",
            "--method", "BlockTurboQuantMSE",
            "--bits", "4",
            "--chunk", "128",
            "--max-tokens", "256",
            "--block-size", "64",
            "--patch-v",
        ]

        # All 15 VqbenchBaselineResult fields flattened with prefix.
        md = result.metadata
        assert md["vqbench_status"] == "ok"
        assert md["vqbench_script"] == "/tmp/reproduce.py"
        assert md["vqbench_python_executable"] == "/fake/python"
        assert md["vqbench_model"] == "Qwen/Qwen3-0.6B"
        assert md["vqbench_method"] == "BlockTurboQuantMSE"
        assert md["vqbench_bits"] == 4
        assert md["vqbench_ppl_fp16"] == 3.10
        assert md["vqbench_ppl_quant"] == 3.15
        assert md["vqbench_delta_ppl"] == 0.05
        assert md["vqbench_delta_pct"] == 1.61
        assert md["vqbench_wall_s"] == 42.0
        assert md["vqbench_returncode"] == 0
        assert md["vqbench_stdout_tail"] == "tail lines here"
        # reason was None on success; flattened as None, not absent.
        assert md["vqbench_reason"] is None

    def test_vqbench_auto_append_uses_oracle_context_chunk_values(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--chunk and --max-tokens must come from the PPL oracle's
        returned context, not re-read from oracle_config. This
        catches a regression where chunk drifted between
        silica oracle and vqbench subprocess."""
        import silica.bench.runner as runner_module

        def ppl_with_custom_context(
            _scenario: Scenario,
            _adapter: Any,
            *,
            fp16_baseline_ppl: float | None = None,
            seed: int = 42,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            del seed
            ctx: dict[str, Any] = {
                "chunk_size": 999,  # diverges from oracle_config
                "max_tokens": 1111,
                "wikitext_path": "/nowhere",
                "kv_codec": "fp16",
            }
            if fp16_baseline_ppl is not None:
                ctx["ppl_fp16"] = fp16_baseline_ppl
            return ({"nll_sum": 1.0, "n_tokens": 50, "ppl": 2.0}, ctx)

        monkeypatch.setattr(
            runner_module, "_run_ppl", ppl_with_custom_context
        )

        repo = "test-owner/xcheck-context-chunk"
        _create_cache_dir(repo)
        scenario = _ppl_scenario_with_xcheck(
            "xcheck-context-chunk",
            repo,
            wikitext_path=_write_fake_wikitext(fake_home_cache),
            kv_codec="fp16",
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/reproduce.py",
                method="IdentityCodec",
                bits=16,
            ),
        )

        captured_args: list[Any] = []

        def fake_vqbench(
            script: Any, script_args: Any = None, **kwargs: Any
        ) -> VqbenchBaselineResult:
            captured_args.append(list(script_args or []))
            return VqbenchBaselineResult(status="ok")

        runner = BenchRunner(
            engine_factory=_factory_returning(
                _FakeAdapter(vocab_size=50), _FakeEngine([])
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            vqbench_xcheck_enabled=True,
            vqbench_runner=fake_vqbench,
        )
        runner.run([scenario])
        assert captured_args == [
            [
                "--model", repo,
                "--seed", "0",
                "--method", "IdentityCodec",
                "--bits", "16",
                "--chunk", "999",
                "--max-tokens", "1111",
            ]
        ]


# ---------- P-5-C.6 step 2 — divergence gate + gap computation -----


from silica.bench.runner import (  # noqa: E402
    _VQBENCH_PCT_EPSILON,
    _compute_gap_fields,
)


class TestComputeGapFields:
    """Unit tests for the pure helper. All runner-level
    integration tests go through the BenchRunner path below; the
    helper gets its own tests because the arithmetic is easy to
    reason about in isolation."""

    def test_computes_signed_ppl_gap_and_pct_gap(self) -> None:
        out = _compute_gap_fields(
            silica_delta_ppl=0.050,
            silica_delta_pct=0.5,
            vqbench_delta_ppl=0.045,
            vqbench_delta_pct=0.4,
            epsilon_ppl=0.01,
        )
        assert out["vqbench_delta_ppl_gap"] == pytest.approx(0.005)
        assert out["vqbench_delta_pct_gap"] == pytest.approx(0.1)
        # |0.005| < 0.01 AND |0.1| <= 0.1 → no warning (both
        # under threshold; pct gap is exactly at epsilon, not
        # above).
        assert out["vqbench_divergence_warning"] is False

    def test_negative_gap_preserved_signed(self) -> None:
        """Silica's delta smaller than vqbench's → negative gap."""
        out = _compute_gap_fields(
            silica_delta_ppl=0.01,
            silica_delta_pct=0.1,
            vqbench_delta_ppl=0.05,
            vqbench_delta_pct=0.5,
            epsilon_ppl=0.01,
        )
        assert out["vqbench_delta_ppl_gap"] == pytest.approx(-0.04)
        # |-0.04| > 0.01 → warning
        assert out["vqbench_divergence_warning"] is True

    def test_ppl_gap_alone_triggers_warning(self) -> None:
        """|ΔPPL gap| over epsilon while pct gap is within
        0.1 still trips the warning — either metric is sufficient."""
        out = _compute_gap_fields(
            silica_delta_ppl=0.05,
            silica_delta_pct=0.3,
            vqbench_delta_ppl=0.015,
            vqbench_delta_pct=0.28,
            epsilon_ppl=0.01,
        )
        # PPL gap = 0.035 > 0.01 → warning even though pct gap
        # (0.02) is safely under the 0.1 pct epsilon.
        assert abs(out["vqbench_delta_ppl_gap"]) > 0.01
        assert abs(out["vqbench_delta_pct_gap"]) < _VQBENCH_PCT_EPSILON
        assert out["vqbench_divergence_warning"] is True

    def test_pct_gap_alone_triggers_warning(self) -> None:
        """Conversely, a big pct gap with small raw ΔPPL still
        raises a warning. Covers the case where the baselines
        scale differently between implementations."""
        out = _compute_gap_fields(
            silica_delta_ppl=0.005,
            silica_delta_pct=1.0,
            vqbench_delta_ppl=0.003,
            vqbench_delta_pct=0.2,
            epsilon_ppl=0.01,
        )
        # PPL gap tiny, pct gap 0.8 > 0.1
        assert abs(out["vqbench_delta_ppl_gap"]) < 0.01
        assert abs(out["vqbench_delta_pct_gap"]) > _VQBENCH_PCT_EPSILON
        assert out["vqbench_divergence_warning"] is True

    @pytest.mark.parametrize(
        "silica_delta_ppl,silica_delta_pct,vqbench_delta_ppl,vqbench_delta_pct",
        [
            (None, 0.5, 0.045, 0.4),  # silica ppl missing
            (0.05, None, 0.045, 0.4),  # silica pct missing
            (0.05, 0.5, None, 0.4),  # vqbench ppl missing
            (0.05, 0.5, 0.045, None),  # vqbench pct missing
            (None, None, None, None),  # all missing
        ],
    )
    def test_none_inputs_yield_none_gaps_and_warning(
        self,
        silica_delta_ppl: float | None,
        silica_delta_pct: float | None,
        vqbench_delta_ppl: float | None,
        vqbench_delta_pct: float | None,
    ) -> None:
        """Any missing delta on either side → all three fields
        ``None``. Users filtering for warnings must not see
        missing-data rows as False (within-threshold)."""
        out = _compute_gap_fields(
            silica_delta_ppl=silica_delta_ppl,
            silica_delta_pct=silica_delta_pct,
            vqbench_delta_ppl=vqbench_delta_ppl,
            vqbench_delta_pct=vqbench_delta_pct,
            epsilon_ppl=0.01,
        )
        assert out["vqbench_delta_ppl_gap"] is None
        assert out["vqbench_delta_pct_gap"] is None
        assert out["vqbench_divergence_warning"] is None

    def test_custom_epsilon_overrides_default(self) -> None:
        """Tighter epsilon flips a previously-safe gap into a
        warning — verifies the CLI --vqbench-epsilon propagation."""
        out_loose = _compute_gap_fields(
            silica_delta_ppl=0.05,
            silica_delta_pct=0.2,
            vqbench_delta_ppl=0.04,
            vqbench_delta_pct=0.18,
            epsilon_ppl=0.05,
        )
        # |0.05 - 0.04| = 0.01 <= 0.05 → no warning; pct gap 0.02 < 0.1.
        assert out_loose["vqbench_divergence_warning"] is False

        out_tight = _compute_gap_fields(
            silica_delta_ppl=0.05,
            silica_delta_pct=0.2,
            vqbench_delta_ppl=0.04,
            vqbench_delta_pct=0.18,
            epsilon_ppl=0.005,
        )
        # Same inputs, tighter epsilon → warning fires on the
        # raw gap side.
        assert out_tight["vqbench_divergence_warning"] is True


class TestComputeSilicaFp16Baseline:
    def test_baseline_replaces_workload_to_fp16_shape(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_compute_silica_fp16_baseline`` must invoke ``_run_ppl``
        with a scenario whose workload has kv_codec=None AND
        prefix_cache=False — the fp16 path signature. Captures the
        scenario the stubbed ``_run_ppl`` receives and inspects
        the workload shape."""
        import silica.bench.runner as runner_module

        captured_scenarios: list[Scenario] = []

        def capturing_run_ppl(
            scenario: Scenario,
            _adapter: Any,
            *,
            fp16_baseline_ppl: float | None = None,
            seed: int = 42,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            del seed
            captured_scenarios.append(scenario)
            return (
                {"nll_sum": 0.5, "n_tokens": 10, "ppl": 2.718},
                {"chunk_size": 128, "max_tokens": 256},
            )

        monkeypatch.setattr(runner_module, "_run_ppl", capturing_run_ppl)

        scenario = _ppl_scenario_with_xcheck(
            "baseline-probe",
            "test-owner/baseline-probe",
            wikitext_path="/nowhere",
            kv_codec="block_tq_b64_b4",
        )
        runner = BenchRunner(
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
        )
        from typing import cast

        from silica.models.adapter import ModelAdapter

        result_ppl = runner._compute_silica_fp16_baseline(
            scenario, cast(ModelAdapter, _FakeAdapter(vocab_size=50))
        )
        assert result_ppl == pytest.approx(2.718)
        # The captured scenario's workload must be the fp16 shape.
        assert len(captured_scenarios) == 1
        wl = captured_scenarios[0].workload
        assert wl.kv_codec is None
        assert wl.prefix_cache is False
        # Other fields preserved (authoring invariants downstream
        # depend on this).
        assert wl.name == scenario.workload.name
        assert wl.max_tokens == scenario.workload.max_tokens


class TestVqbenchXcheckBaselineFires:
    def test_baseline_computed_only_when_xcheck_preconditions_met(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_run_ppl`` should receive a non-None
        ``fp16_baseline_ppl`` kwarg on the main pass iff the row
        will actually cross-check (flag on + spec present + arm
        matches). Skip cases must NOT incur the extra baseline
        invocation."""
        import silica.bench.runner as runner_module

        main_calls: list[dict[str, Any]] = []

        def tracking_run_ppl(
            scenario: Scenario,
            adapter: Any,
            *,
            fp16_baseline_ppl: float | None = None,
            seed: int = 42,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            del seed
            # Distinguish baseline pass (kv_codec=None) from main
            # pass by workload shape; only record the main pass.
            if scenario.workload.kv_codec is not None:
                main_calls.append({"baseline": fp16_baseline_ppl})
            ctx: dict[str, Any] = {
                "chunk_size": 128,
                "max_tokens": 256,
                "wikitext_path": "/nowhere",
                "kv_codec": scenario.workload.kv_codec,
            }
            if fp16_baseline_ppl is not None:
                ctx["ppl_fp16"] = fp16_baseline_ppl
            return (
                {"nll_sum": 1.5, "n_tokens": 128, "ppl": 3.14},
                ctx,
            )

        monkeypatch.setattr(runner_module, "_run_ppl", tracking_run_ppl)

        repo = "test-owner/baseline-precondition"
        _create_cache_dir(repo)
        wikitext = _write_fake_wikitext(fake_home_cache)

        declared = _ppl_scenario_with_xcheck(
            "declared-match",
            repo,
            wikitext_path=wikitext,
            kv_codec="fp16",
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/x.py", method="IdentityCodec", bits=16
            ),
        )
        undeclared = _ppl_scenario_with_xcheck(
            "undeclared",
            repo,
            wikitext_path=wikitext,
            kv_codec="fp16",
            vqbench_xcheck=None,
        )

        def fake_vqbench(*a: Any, **kw: Any) -> VqbenchBaselineResult:
            return VqbenchBaselineResult(status="ok")

        runner = BenchRunner(
            engine_factory=_factory_returning(
                _FakeAdapter(vocab_size=50), _FakeEngine([])
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            vqbench_xcheck_enabled=True,
            vqbench_runner=fake_vqbench,
        )
        runner.run([declared, undeclared])

        assert len(main_calls) == 2
        # declared (xcheck will fire) → baseline was threaded.
        assert main_calls[0]["baseline"] == pytest.approx(3.14)
        # undeclared (xcheck skipped via "scenario_not_declared")
        # → baseline NOT threaded; saves the extra PPL pass.
        assert main_calls[1]["baseline"] is None


class TestVqbenchXcheckGapInMetadata:
    def test_happy_path_populates_all_three_gap_fields(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: xcheck preconditions met → vqbench returns
        a concrete ΔPPL → runner computes the 3 gap fields from
        silica's oracle-computed delta_ppl (via ppl_fp16 threaded
        through context) vs vqbench's delta_ppl."""
        import silica.bench.runner as runner_module

        def deterministic_run_ppl(
            scenario: Scenario,
            _adapter: Any,
            *,
            fp16_baseline_ppl: float | None = None,
            seed: int = 42,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            del seed
            # Baseline scenario (kv_codec=None): fp16 PPL = 3.0.
            # Main scenario (kv_codec=fp16): quant PPL = 3.05.
            if scenario.workload.kv_codec is None:
                return (
                    {"nll_sum": 1.0, "n_tokens": 100, "ppl": 3.0},
                    {},
                )
            ctx: dict[str, Any] = {
                "chunk_size": 128,
                "max_tokens": 256,
                "wikitext_path": "/nowhere",
                "kv_codec": scenario.workload.kv_codec,
            }
            if fp16_baseline_ppl is not None:
                ctx["ppl_fp16"] = fp16_baseline_ppl
            return (
                {"nll_sum": 1.1, "n_tokens": 100, "ppl": 3.05},
                ctx,
            )

        monkeypatch.setattr(runner_module, "_run_ppl", deterministic_run_ppl)

        repo = "test-owner/gap-happy"
        _create_cache_dir(repo)
        scenario = _ppl_scenario_with_xcheck(
            "gap-happy",
            repo,
            wikitext_path=_write_fake_wikitext(fake_home_cache),
            kv_codec="fp16",
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/x.py", method="IdentityCodec", bits=16
            ),
        )

        def fake_vqbench(*a: Any, **kw: Any) -> VqbenchBaselineResult:
            # vqbench reports its own ΔPPL = 0.04 (slightly lower
            # than silica's 0.05 delta). Expected ppl gap = 0.01.
            return VqbenchBaselineResult(
                status="ok",
                ppl_fp16=3.00,
                ppl_quant=3.04,
                delta_ppl=0.04,
                delta_pct=1.333,
            )

        runner = BenchRunner(
            engine_factory=_factory_returning(
                _FakeAdapter(vocab_size=50), _FakeEngine([])
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            vqbench_xcheck_enabled=True,
            vqbench_runner=fake_vqbench,
            vqbench_epsilon=0.01,
        )
        [result] = runner.run([scenario])
        md = result.metadata
        # silica delta = 3.05 - 3.0 = 0.05; vqbench delta = 0.04.
        # Gap = 0.05 - 0.04 = 0.01. Not strictly > 0.01 epsilon
        # → no warning from the ppl metric alone.
        assert md["vqbench_delta_ppl_gap"] == pytest.approx(0.01)
        # Pct check: silica pct = 0.05 / 3.0 * 100 = 1.667;
        # vqbench pct = 1.333; gap = 0.334. |0.334| > 0.1 pct
        # epsilon → warning fires.
        assert md["vqbench_delta_pct_gap"] == pytest.approx(
            (0.05 / 3.0 * 100) - 1.333, abs=1e-4
        )
        assert md["vqbench_divergence_warning"] is True

    def test_epsilon_cli_override_shifts_warning_boundary(
        self, fake_home_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass vqbench_epsilon=0.5 → the same gap (0.01) is now
        safely under threshold; no warning from the ppl metric.
        Separately verifies the epsilon plumbs through
        BenchRunner.__init__ into _compute_gap_fields."""
        import silica.bench.runner as runner_module

        def deterministic_run_ppl(
            scenario: Scenario,
            _adapter: Any,
            *,
            fp16_baseline_ppl: float | None = None,
            seed: int = 42,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            del seed
            if scenario.workload.kv_codec is None:
                return (
                    {"nll_sum": 1.0, "n_tokens": 100, "ppl": 3.0},
                    {},
                )
            ctx: dict[str, Any] = {"kv_codec": scenario.workload.kv_codec}
            if fp16_baseline_ppl is not None:
                ctx["ppl_fp16"] = fp16_baseline_ppl
            return (
                {"nll_sum": 1.1, "n_tokens": 100, "ppl": 3.05},
                ctx,
            )

        monkeypatch.setattr(runner_module, "_run_ppl", deterministic_run_ppl)

        repo = "test-owner/epsilon-loose"
        _create_cache_dir(repo)
        scenario = _ppl_scenario_with_xcheck(
            "epsilon-loose",
            repo,
            wikitext_path=_write_fake_wikitext(fake_home_cache),
            kv_codec="fp16",
            vqbench_xcheck=VqbenchXcheckSpec(
                script_path="/tmp/x.py", method="IdentityCodec", bits=16
            ),
        )

        # vqbench pct matching silica's pct so pct gap is zero —
        # isolate the ppl-gap boundary for this test.
        silica_pct = 0.05 / 3.0 * 100

        def fake_vqbench(*a: Any, **kw: Any) -> VqbenchBaselineResult:
            return VqbenchBaselineResult(
                status="ok",
                ppl_fp16=3.00,
                ppl_quant=3.04,
                delta_ppl=0.04,
                delta_pct=silica_pct,
            )

        runner = BenchRunner(
            engine_factory=_factory_returning(
                _FakeAdapter(vocab_size=50), _FakeEngine([])
            ),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            vqbench_xcheck_enabled=True,
            vqbench_runner=fake_vqbench,
            vqbench_epsilon=0.5,
        )
        [result] = runner.run([scenario])
        md = result.metadata
        assert md["vqbench_delta_ppl_gap"] == pytest.approx(0.01)
        # pct gap = 0 (matched), ppl gap 0.01 < 0.5 → no warning.
        assert md["vqbench_divergence_warning"] is False
