"""silica.bench.runner — P-4.1 BenchRunner.

``BenchRunner`` drives one or more :class:`Scenario` objects from
skip-check through workload execution through oracle through JSONL
emission. It is the single seam where the bench harness touches the
engine; the CLI shim in ``scripts/bench.py`` is a thin argparse layer
around this class.

Design choices this module encodes:

  * **Per-scenario isolation**. Each scenario gets a fresh
    ``adapter_for_repo`` load. Weight streaming / model caching
    across scenarios is a P-4.2+ feature; P-4.1 favours correctness
    over throughput (the first migrated scenario is cheap).
  * **Injected engine factory**. The default factory delegates to
    :func:`silica.models.factory.adapter_for_repo`, but tests pass
    a fake that returns a mock-driven ``Engine``. This is the only
    seam the test suite needs to cover every runner branch without
    real weights.
  * **JSONL from day 1**. Every result, including skipped and
    failed ones, is appended to the configured output path. A
    skipped row is still load-bearing: it records *why* a gate
    blocked the run so the bench report does not pretend the
    scenario does not exist.
  * **Batched workloads rejected in P-4.1**. Any scenario with
    ``workload.max_batch_size > 1`` returns
    ``status="skipped", reason="batched_workload_deferred_p42"``
    rather than silently running single-request on the first
    prompt. P-4.2 lands the batched dispatch branch alongside the
    D3.1 migration.

Exception handling is deliberately coarse: any exception during the
load / generate / oracle call chain collapses to
``status="failed"`` with the exception type + message as the
reason. Structured per-phase failure reporting (distinguishing a
load crash from an oracle rejection) is a P-4.2 polish item once
the second and third scenarios reveal whether the collapse is
actually confusing in practice.

The peak-memory hook is injectable so tests on fake adapters can
skip the mlx call entirely. The defaults are backed by
``mlx.core.get_peak_memory`` / ``reset_peak_memory``, which are the
same APIs ``scripts/bench_p2_baseline.py`` has been using.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

from silica.bench.oracles import ORACLES
from silica.bench.scenario import (
    Scenario,
    ScenarioResult,
    hf_cache_path_for_repo,
)
from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.models.adapter import ModelAdapter

EngineFactory = Callable[[Scenario], tuple[ModelAdapter, Engine]]


def _default_engine_factory(
    scenario: Scenario,
) -> tuple[ModelAdapter, Engine]:
    """Load adapter + KV manager for ``scenario.repo`` and build an Engine.

    Imports :mod:`silica.models.factory` lazily so ``silica.bench``
    stays cheap to import for CLI listing (--list) without pulling
    in the adapter dispatch table.
    """
    from silica.models.factory import adapter_for_repo

    adapter, kv = adapter_for_repo(scenario.repo)
    return adapter, Engine(adapter, kv)


def _mlx_reset_peak_memory() -> None:
    """Reset MLX peak-memory accounting. No-op if mlx is unavailable."""
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass


def _mlx_peak_memory_mb() -> float | None:
    """Read MLX peak memory in MB, or None if mlx is unavailable."""
    try:
        import mlx.core as mx

        return float(mx.get_peak_memory()) / 1e6
    except Exception:
        return None


def _check_gates(scenario: Scenario) -> str | None:
    """Return skip reason or ``None`` if the scenario is runnable.

    Two gates, matching the existing dual-gate pattern:

      * HF cache directory must exist (weak gate — cheap scenarios
        are gated on cache alone).
      * ``gate_env_var`` (if set) must equal ``"1"`` (strong gate
        for scenarios whose forward is expensive enough to warrant
        explicit opt-in).
    """
    cache = hf_cache_path_for_repo(scenario.repo)
    if not cache.exists():
        return f"cache_missing:{cache}"
    if scenario.gate_env_var is not None:
        if os.environ.get(scenario.gate_env_var) != "1":
            return f"env_var_not_set:{scenario.gate_env_var}"
    return None


class BenchRunner:
    """Drives scenarios and emits :class:`ScenarioResult` rows.

    Single-use but reusable: ``run`` can be called multiple times
    with different scenario iterables if the same configured
    output path should accumulate rows.
    """

    def __init__(
        self,
        *,
        engine_factory: EngineFactory | None = None,
        out_path: Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
        reset_peak: Callable[[], None] | None = None,
        read_peak_mb: Callable[[], float | None] | None = None,
    ) -> None:
        self._engine_factory: EngineFactory = (
            engine_factory if engine_factory is not None else _default_engine_factory
        )
        self._out_path = out_path
        self._clock = clock
        self._reset_peak = reset_peak or _mlx_reset_peak_memory
        self._read_peak_mb = read_peak_mb or _mlx_peak_memory_mb

    def run(self, scenarios: Iterable[Scenario]) -> list[ScenarioResult]:
        """Run every scenario in order and return the results list.

        Rows are appended to ``out_path`` as they complete (one JSON
        object per line) so a long-running bench does not lose
        progress if the process is killed mid-run.
        """
        results: list[ScenarioResult] = []
        for scenario in scenarios:
            result = self._run_one(scenario)
            results.append(result)
            if self._out_path is not None:
                self._append_jsonl(result)
        return results

    def _run_one(self, scenario: Scenario) -> ScenarioResult:
        # Gate check first: skipped rows never touch the engine, so
        # cache-missing scenarios do not accidentally block on a
        # download prompt or a model-factory KeyError.
        skip_reason = _check_gates(scenario)
        if skip_reason is not None:
            return ScenarioResult(
                scenario_id=scenario.id,
                status="skipped",
                reason=skip_reason,
            )

        wl = scenario.workload
        if wl.max_batch_size > 1:
            return ScenarioResult(
                scenario_id=scenario.id,
                status="skipped",
                reason="batched_workload_deferred_p42",
            )
        if len(wl.prompts) != 1:
            return ScenarioResult(
                scenario_id=scenario.id,
                status="failed",
                reason=(
                    f"single-request scenario expects exactly one "
                    f"prompt, got {len(wl.prompts)}"
                ),
            )

        try:
            self._reset_peak()
            t_start = self._clock()
            adapter, engine = self._engine_factory(scenario)
            tokens = _run_single_request(scenario, engine, adapter)
            wall_s = self._clock() - t_start
            peak_mb = self._read_peak_mb()
            oracle_fn = ORACLES[scenario.oracle]
            ok, oracle_reason, metadata = oracle_fn(
                scenario,
                tokens,
                {"vocab_size": adapter.config.vocab_size},
            )
        except Exception as exc:  # noqa: BLE001 — top-level scenario boundary
            return ScenarioResult(
                scenario_id=scenario.id,
                status="failed",
                reason=f"{type(exc).__name__}: {exc}",
            )

        snap = engine.metrics.snapshot()
        return ScenarioResult(
            scenario_id=scenario.id,
            status="ok" if ok else "failed",
            reason=oracle_reason,
            ttft_ms=snap.ttft_ms,
            prefill_tok_s=snap.prefill_tok_s,
            decode_tok_s=snap.decode_tok_s,
            resident_mb=snap.resident_mb,
            peak_memory_mb=peak_mb,
            total_tokens=len(tokens),
            wall_s=wall_s,
            metadata=dict(metadata),
        )

    def _append_jsonl(self, result: ScenarioResult) -> None:
        assert self._out_path is not None
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result)) + "\n")


def _run_single_request(
    scenario: Scenario, engine: Engine, adapter: ModelAdapter
) -> list[int]:
    wl = scenario.workload
    tokenizer = adapter.tokenizer()
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))
    params = SamplingParams(
        temperature=wl.temperature,
        top_p=wl.top_p,
        max_tokens=wl.max_tokens,
        stop_token_ids=eos_ids,
    )
    return list(engine.generate(wl.prompts[0], params))


__all__ = ["BenchRunner", "EngineFactory"]
