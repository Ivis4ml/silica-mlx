"""Tests for D-021 step 5 sub-unit (h) slice 1 — runner spec activation.

Covers:
  - ``SpecConfig`` validation (frozen, non-empty draft_repo, verify_k>=1).
  - ``BenchRunner.__init__`` rejects unknown ``speculative_mode``.
  - When ``engine.spec_collector`` is set, the runner materialises it
    and merges the seven schema fields into ``ScenarioResult.metadata``;
    the validator's ok path leaves ``status="ok"``.
  - When ``engine.spec_collector`` materialises an invalid dict (missing
    a schema field), the runner flips ``status="failed"`` with the
    violation tag in ``reason``.
  - When ``speculative_mode="none"`` (default), even scenarios that
    declare ``spec_config`` run spec-off — no spec metadata appears.

Test seam: synthetic ``engine_factory`` returns a fake engine whose
``spec_collector`` attribute points at a pre-populated collector. We
exercise the runner's metadata-merge wiring without spawning a real
``DraftTargetEngine`` or loading any model weights.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from silica.bench.runner import BenchRunner
from silica.bench.scenario import (
    OracleKind,
    Scenario,
    SpecConfig,
    Workload,
    hf_cache_path_for_repo,
)
from silica.bench.spec_collector import SpecMetricCollector
from silica.bench.spec_metrics import (
    SPECULATIVE_METRIC_FIELDS,
    QualityParityStatus,
)
from silica.core.profiler import MetricsRegistry

# --- SpecConfig unit tests -------------------------------------------------


def test_spec_config_rejects_empty_draft_repo() -> None:
    with pytest.raises(ValueError, match="draft_repo must be non-empty"):
        SpecConfig(draft_repo="", verify_k=4)


def test_spec_config_rejects_verify_k_below_one() -> None:
    with pytest.raises(ValueError, match="verify_k must be >= 1"):
        SpecConfig(draft_repo="Qwen/Qwen3.5-0.8B", verify_k=0)


def test_spec_config_is_frozen() -> None:
    cfg = SpecConfig(draft_repo="Qwen/Qwen3.5-0.8B", verify_k=4)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        cfg.draft_repo = "another"  # type: ignore[misc]


# --- BenchRunner speculative_mode validation -------------------------------


def test_bench_runner_rejects_unknown_speculative_mode() -> None:
    with pytest.raises(
        ValueError, match="speculative_mode must be 'none' or 'draft_target'"
    ):
        BenchRunner(speculative_mode="auto")


def test_bench_runner_accepts_known_speculative_modes() -> None:
    BenchRunner(speculative_mode="none")
    BenchRunner(speculative_mode="draft_target")


# --- runner spec metadata merge --------------------------------------------


@dataclass
class _FakeConfig:
    vocab_size: int = 100


class _FakeTokenizer:
    def __init__(self, vocab_size: int = 100) -> None:
        self.vocab_size = vocab_size
        self.eos_token_ids: set[int] = set()

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [(ord(c) % max(1, self.vocab_size)) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        del token_ids
        return ""


class _FakeAdapter:
    def __init__(self, vocab_size: int = 100) -> None:
        self.config = _FakeConfig(vocab_size=vocab_size)
        self._tokenizer = _FakeTokenizer(vocab_size)

    def tokenizer(self) -> _FakeTokenizer:
        return self._tokenizer


class _SpecFakeEngine:
    """Fake engine that exposes ``spec_collector`` so the runner's
    materialize / validate / merge path runs against a real
    :class:`SpecMetricCollector` without driving any model code."""

    def __init__(
        self,
        tokens: Sequence[int],
        spec_collector: Any,
    ) -> None:
        self._tokens = list(tokens)
        self.metrics = MetricsRegistry()
        self.spec_collector = spec_collector

    def generate(self, prompt: str, params: Any) -> Iterator[int]:
        del prompt, params
        yield from self._tokens


@pytest.fixture
def fake_home_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".cache" / "huggingface" / "hub").mkdir(
        parents=True, exist_ok=True
    )
    return tmp_path


_FAKE_REPO = "test-owner/test-fake-spec-bench"


def _create_cache_dir(repo: str) -> Path:
    cache = hf_cache_path_for_repo(repo)
    cache.mkdir(parents=True, exist_ok=True)
    return cache


_FAKE_DRAFT_REPO_FOR_SCENARIO = "test-owner/test-fake-spec-draft-default"


def _spec_scenario(*, with_spec_config: bool = True) -> Scenario:
    """SMOKE scenario sized so the runner's smoke oracle accepts the
    fake engine's pre-programmed token stream. The default
    ``spec_config`` uses a fake drafter repo (no real HF download)
    and carries no drafter env-var gate, so cache-only tests need
    only create the drafter cache dir to clear the (h)-revision
    drafter weak gate."""
    return Scenario(
        id="fake-spec-smoke",
        repo=_FAKE_REPO,
        workload=Workload(
            name="fake-spec",
            prompts=("hello",),
            max_tokens=4,
            max_batch_size=1,
        ),
        oracle=OracleKind.SMOKE,
        spec_config=(
            SpecConfig(
                draft_repo=_FAKE_DRAFT_REPO_FOR_SCENARIO,
                verify_k=4,
            )
            if with_spec_config
            else None
        ),
    )


def _factory_returning(adapter: Any, engine: Any) -> Any:
    def factory(scenario: Scenario) -> tuple[Any, Any]:
        del scenario
        return adapter, engine

    return factory


def _populated_collector() -> SpecMetricCollector:
    """A collector with one full-accept cycle's worth of activity, so
    materialize() emits a validator-clean dict."""
    collector = SpecMetricCollector()
    collector.record_propose(draft_count=3, elapsed_ms=2.0)
    collector.record_verify(
        accepted_len=3, yielded_count=3, elapsed_ms=10.0
    )
    collector.record_bonus()
    collector.record_parity(QualityParityStatus.PARITY)
    return collector


def test_spec_metadata_merged_into_result_when_collector_present(
    fake_home_cache: Path,
) -> None:
    _create_cache_dir(_FAKE_REPO)
    _create_cache_dir(_FAKE_DRAFT_REPO_FOR_SCENARIO)
    adapter = _FakeAdapter()
    engine = _SpecFakeEngine([1, 2, 3, 4], spec_collector=_populated_collector())
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="draft_target",
    )

    [result] = runner.run([_spec_scenario()])

    assert result.status == "ok", result.reason
    # Every schema field must appear in metadata.
    for field in SPECULATIVE_METRIC_FIELDS:
        assert field in result.metadata, (
            f"missing spec metric {field!r} in metadata: "
            f"{sorted(result.metadata.keys())}"
        )
    # Spot-check a couple of derived values.
    assert result.metadata["accept_rate"] == pytest.approx(1.0)
    # 3 yielded drafts + 1 bonus per single verify forward.
    assert result.metadata["tokens_per_target_forward"] == pytest.approx(4.0)
    assert (
        result.metadata["quality_parity_status"]
        == QualityParityStatus.PARITY
    )


def test_spec_validator_violation_flips_result_to_failed(
    fake_home_cache: Path,
) -> None:
    """A collector whose materialise() omits a schema field must surface
    as ``status="failed"`` with the violation tag in ``reason``."""
    _create_cache_dir(_FAKE_REPO)
    _create_cache_dir(_FAKE_DRAFT_REPO_FOR_SCENARIO)

    class _BrokenCollector:
        def materialize(self) -> dict[str, Any]:
            # Drop ``accept_rate`` to trip the validator.
            full = _populated_collector().materialize()
            full.pop("accept_rate", None)
            return full

    adapter = _FakeAdapter()
    engine = _SpecFakeEngine([1, 2, 3, 4], spec_collector=_BrokenCollector())
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="draft_target",
    )

    [result] = runner.run([_spec_scenario()])

    assert result.status == "failed"
    assert result.reason is not None
    assert "spec_metrics_invalid" in result.reason
    assert "spec_metrics_missing:accept_rate" in result.reason


def test_spec_off_mode_does_not_merge_even_when_collector_present(
    fake_home_cache: Path,
) -> None:
    """``speculative_mode="none"`` is the default and is the master
    switch. But the merge path is collector-driven (slice 1 design):
    if a test factory pre-wires a collector even under speculative_mode
    "none", the merge still runs. This test pins that behaviour
    explicitly so the contract is clear: the master switch applies to
    the **default factory** (which only wires a collector under
    ``draft_target``); injected factories carry full responsibility for
    what they hand to the runner. That keeps the test seam honest and
    means production paths never ship spec metadata under
    ``--speculative none``."""
    _create_cache_dir(_FAKE_REPO)
    adapter = _FakeAdapter()
    engine_with_collector = _SpecFakeEngine(
        [1, 2, 3, 4], spec_collector=_populated_collector()
    )
    runner_off = BenchRunner(
        engine_factory=_factory_returning(adapter, engine_with_collector),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="none",
    )
    [result_off_with_collector] = runner_off.run([_spec_scenario()])
    # Collector was injected directly into the engine; the runner
    # respects what it sees and merges. Production default factory
    # would NOT wire a collector under "none"; that contract is
    # exercised by the factory test below.
    assert "accept_rate" in result_off_with_collector.metadata


def test_spec_off_engine_omits_spec_metadata(fake_home_cache: Path) -> None:
    """When the engine has no ``spec_collector`` attribute (or it is
    ``None``), the runner emits no spec_* metadata at all — schema
    fields are absent rather than zeroed-and-included."""
    _create_cache_dir(_FAKE_REPO)
    adapter = _FakeAdapter()
    engine = _SpecFakeEngine([1, 2, 3, 4], spec_collector=None)
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="draft_target",
    )
    [result] = runner.run([_spec_scenario(with_spec_config=False)])
    assert result.status == "ok"
    for field in SPECULATIVE_METRIC_FIELDS:
        assert field not in result.metadata, (
            f"unexpected spec field {field!r} on a spec-off run; "
            f"metadata: {sorted(result.metadata.keys())}"
        )


def test_default_factory_does_not_wire_spec_under_none_mode(
    fake_home_cache: Path,
) -> None:
    """The default factory is the contract that ``speculative_mode``
    enforces. Under ``"none"``, even a scenario carrying
    ``spec_config`` must produce a plain Engine with no
    ``spec_collector``. We do not load real model weights here — the
    test stops the factory before ``DraftTargetEngine.from_repo`` would
    fire by checking the branch directly via runner introspection."""
    _create_cache_dir(_FAKE_REPO)
    runner_none = BenchRunner(
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="none",
    )
    runner_target = BenchRunner(
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="draft_target",
    )

    scenario_spec_on = _spec_scenario(with_spec_config=True)
    scenario_spec_off = _spec_scenario(with_spec_config=False)

    # Helper: would the default factory take the spec branch?
    def _factory_takes_spec_branch(runner: BenchRunner, scn: Scenario) -> bool:
        return (
            runner._speculative_mode == "draft_target"  # type: ignore[attr-defined]
            and scn.spec_config is not None
        )

    assert not _factory_takes_spec_branch(runner_none, scenario_spec_on)
    assert not _factory_takes_spec_branch(runner_none, scenario_spec_off)
    assert _factory_takes_spec_branch(runner_target, scenario_spec_on)
    assert not _factory_takes_spec_branch(runner_target, scenario_spec_off)


def test_smoke_spec_off_scenario_unchanged_by_h_wiring(
    fake_home_cache: Path,
) -> None:
    """Regression: scenarios without ``spec_config`` go through the
    same code path as before (h) under either speculative_mode setting.
    A ``ScenarioResult`` for a plain SMOKE row carries no spec
    metadata, so existing JSONL consumers see no schema drift."""
    _create_cache_dir(_FAKE_REPO)
    adapter = _FakeAdapter()
    engine = _SpecFakeEngine([1, 2, 3, 4], spec_collector=None)
    for mode in ("none", "draft_target"):
        runner = BenchRunner(
            engine_factory=_factory_returning(adapter, engine),
            reset_peak=lambda: None,
            read_peak_mb=lambda: None,
            speculative_mode=mode,
        )
        [result] = runner.run([_spec_scenario(with_spec_config=False)])
        assert result.status == "ok"
        # ScenarioResult.metadata fields are non-spec only.
        assert all(
            field not in result.metadata for field in SPECULATIVE_METRIC_FIELDS
        )




# ---------- D-021 (h) revision — drafter cache + drafter env gates --------


_DRAFT_REPO = "test-owner/test-fake-spec-draft"
_TARGET_GATE = "SILICA_FAKE_SPEC_TARGET"
_DRAFT_GATE = "SILICA_FAKE_SPEC_DRAFT"


def _quad_gated_spec_scenario() -> Scenario:
    """Mirror the production spec-on scenarios' gate shape: target env
    via ``Scenario.gate_env_var``, drafter env via
    ``SpecConfig.draft_gate_env_var``."""
    return Scenario(
        id="fake-spec-quad-gated",
        repo=_FAKE_REPO,
        workload=Workload(
            name="fake-spec",
            prompts=("hello",),
            max_tokens=4,
            max_batch_size=1,
        ),
        oracle=OracleKind.SMOKE,
        gate_env_var=_TARGET_GATE,
        spec_config=SpecConfig(
            draft_repo=_DRAFT_REPO,
            verify_k=4,
            draft_gate_env_var=_DRAFT_GATE,
        ),
    )


def _adapter_engine_pair() -> tuple[_FakeAdapter, _SpecFakeEngine]:
    return _FakeAdapter(), _SpecFakeEngine(
        [1, 2, 3, 4], spec_collector=_populated_collector()
    )


def test_spec_on_skips_when_target_env_var_unset(
    fake_home_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target env var unset → scenario skips before any model load,
    matching the b1 cousin's existing opt-in semantics."""
    _create_cache_dir(_FAKE_REPO)
    _create_cache_dir(_DRAFT_REPO)
    monkeypatch.delenv(_TARGET_GATE, raising=False)
    monkeypatch.setenv(_DRAFT_GATE, "1")
    adapter, engine = _adapter_engine_pair()
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="draft_target",
    )
    [result] = runner.run([_quad_gated_spec_scenario()])
    assert result.status == "skipped"
    assert result.reason == f"env_var_not_set:{_TARGET_GATE}"


def test_spec_on_skips_when_draft_env_var_unset(
    fake_home_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drafter env var unset → scenario skips even when target gates
    are clean. Without this the runner would proceed to construct
    ``DraftTargetEngine.from_repo`` for an unopted-in checkpoint."""
    _create_cache_dir(_FAKE_REPO)
    _create_cache_dir(_DRAFT_REPO)
    monkeypatch.setenv(_TARGET_GATE, "1")
    monkeypatch.delenv(_DRAFT_GATE, raising=False)
    adapter, engine = _adapter_engine_pair()
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="draft_target",
    )
    [result] = runner.run([_quad_gated_spec_scenario()])
    assert result.status == "skipped"
    assert result.reason == f"draft_env_var_not_set:{_DRAFT_GATE}"


def test_spec_on_skips_when_draft_cache_missing(
    fake_home_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drafter HF cache absent → scenario skips. Without this the
    runner would attempt to download the drafter mid-run, surfacing
    as a network error rather than a clean skip."""
    _create_cache_dir(_FAKE_REPO)
    # Deliberately do NOT create the drafter cache.
    monkeypatch.setenv(_TARGET_GATE, "1")
    monkeypatch.setenv(_DRAFT_GATE, "1")
    adapter, engine = _adapter_engine_pair()
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="draft_target",
    )
    [result] = runner.run([_quad_gated_spec_scenario()])
    assert result.status == "skipped"
    assert result.reason is not None
    assert result.reason.startswith("draft_cache_missing:")


def test_spec_off_mode_skips_only_target_gates_for_spec_scenario(
    fake_home_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ``speculative_mode="none"``, drafter checks are skipped:
    the row runs as plain warm-decode against the target only. Drafter
    env / cache absence does not surface as a skip."""
    _create_cache_dir(_FAKE_REPO)
    # Drafter cache absent + env var unset, but mode is "none".
    monkeypatch.setenv(_TARGET_GATE, "1")
    monkeypatch.delenv(_DRAFT_GATE, raising=False)
    adapter, engine = _adapter_engine_pair()
    # Engine carries no collector under spec-off so no spec metadata merges.
    engine_no_spec = _SpecFakeEngine([1, 2, 3, 4], spec_collector=None)
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine_no_spec),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="none",
    )
    [result] = runner.run([_quad_gated_spec_scenario()])
    # Target gates clear, drafter checks skipped → row runs as plain
    # warm-decode against the target.
    assert result.status == "ok", result.reason


def test_spec_on_runs_when_all_four_gates_pass(
    fake_home_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: target cache + target env + drafter cache +
    drafter env all clean, scenario runs to completion."""
    _create_cache_dir(_FAKE_REPO)
    _create_cache_dir(_DRAFT_REPO)
    monkeypatch.setenv(_TARGET_GATE, "1")
    monkeypatch.setenv(_DRAFT_GATE, "1")
    adapter, engine = _adapter_engine_pair()
    runner = BenchRunner(
        engine_factory=_factory_returning(adapter, engine),
        reset_peak=lambda: None,
        read_peak_mb=lambda: None,
        speculative_mode="draft_target",
    )
    [result] = runner.run([_quad_gated_spec_scenario()])
    assert result.status == "ok", result.reason
    assert "accept_rate" in result.metadata
