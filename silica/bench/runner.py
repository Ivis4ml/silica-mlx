"""silica.bench.runner — BenchRunner.

``BenchRunner`` drives one or more :class:`Scenario` objects from
skip-check through workload execution through oracle through JSONL
emission. It is the single seam where the bench harness touches the
engine; the CLI shim in ``scripts/bench.py`` is a thin argparse layer
around this class.

Design choices this module encodes:

  * **Per-scenario isolation**. Each scenario gets a fresh
    ``adapter_for_repo`` load. Weight streaming / model caching
    across scenarios is a later phase feature; P-4.1/4.2 favour
    correctness over throughput (migrated scenarios are cheap).
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
  * **B>1 workloads rejected through P-4.2b**. Any scenario with
    ``workload.max_batch_size > 1`` returns ``status="skipped",
    reason="batched_workload_deferred_p42"``. P-4.2c lands the
    B>1 dispatch branch alongside the direct mlx-lm batched
    reference oracle.

Oracle-driven dispatch:

The runner picks its workload-execution path by
``scenario.oracle``:

  * ``SMOKE`` — single-request via ``Engine.generate``; oracle
    receives the token stream plus ``context={'vocab_size': N}``.
  * ``B1_PARITY_VS_SINGLE`` — runs the single-request reference
    first (populates ``engine.metrics``), then collects B=1
    batched tokens via ``Engine.generate_batch`` with
    ``max_batch_size=1`` and the *same* :class:`SamplingParams`
    instance. Batch-event validation is strict: an ``aborted``
    event or a ``req_index != 0`` event raises a RuntimeError that
    collapses to ``status="failed"`` with a ``b1_batched_*``
    reason, never reaching the oracle (the user is told the scheduler
    misbehaved, not that the tokens mismatched). On success the
    oracle receives batched tokens plus ``context`` carrying
    ``reference_tokens``.

Exception handling is deliberately coarse: any exception during
load / generate / oracle collapses to ``status="failed"`` with the
exception type + message. Metrics snapshot reflects the
single-request reference for B1 parity because
``Engine.generate_batch`` does not populate the shared
``MetricsRegistry`` at present; ``wall_s`` covers reference +
batched end-to-end.

The peak-memory hook is injectable so tests on fake adapters can
skip the mlx call entirely. The defaults are backed by
``mlx.core.get_peak_memory`` / ``reset_peak_memory``, the same APIs
``scripts/bench_p2_baseline.py`` uses.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from silica.bench.oracles import ORACLES
from silica.bench.scenario import (
    OracleKind,
    Scenario,
    ScenarioResult,
    Workload,
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
            params = _build_sampling_params(wl, adapter)

            if scenario.oracle == OracleKind.B1_PARITY_VS_SINGLE:
                tokens, oracle_context = _run_b1_parity(
                    scenario, engine, adapter, params
                )
            else:
                tokens = _run_single_request(scenario, engine, params)
                oracle_context = {"vocab_size": adapter.config.vocab_size}

            wall_s = self._clock() - t_start
            peak_mb = self._read_peak_mb()
            oracle_fn = ORACLES[scenario.oracle]
            ok, oracle_reason, metadata = oracle_fn(
                scenario, tokens, oracle_context
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


def _build_sampling_params(
    workload: Workload, adapter: ModelAdapter
) -> SamplingParams:
    """Shared SamplingParams helper.

    Both the single-request and the B=1 batched paths on a
    B1_PARITY scenario pull from this one helper so the parity
    claim stays honest: divergent tokens cannot be blamed on
    accidentally drifted sampling params.
    """
    tokenizer = adapter.tokenizer()
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))
    return SamplingParams(
        temperature=workload.temperature,
        top_p=workload.top_p,
        max_tokens=workload.max_tokens,
        stop_token_ids=eos_ids,
    )


def _run_single_request(
    scenario: Scenario, engine: Engine, params: SamplingParams
) -> list[int]:
    return list(engine.generate(scenario.workload.prompts[0], params))


def _run_b1_parity(
    scenario: Scenario,
    engine: Engine,
    adapter: ModelAdapter,
    params: SamplingParams,
) -> tuple[list[int], dict[str, Any]]:
    """Drive single-request reference + B=1 batched on the same prompt.

    Returns the batched token stream (the oracle's primary input)
    and the oracle context carrying ``reference_tokens`` and
    ``vocab_size``. Running reference first means the engine's
    MetricsRegistry captures single-request timings — batch-side
    metrics are not re-populated by ``generate_batch`` at present,
    so the JSONL row reflects the reference execution.
    """
    prompt = scenario.workload.prompts[0]
    reference_tokens = list(engine.generate(prompt, params))
    batched_tokens = _collect_b1_batched_tokens(engine, prompt, params)
    return batched_tokens, {
        "vocab_size": adapter.config.vocab_size,
        "reference_tokens": reference_tokens,
    }


def _collect_b1_batched_tokens(
    engine: Engine, prompt: str, params: SamplingParams
) -> list[int]:
    """Collect row-0 token events from a B=1 ``generate_batch`` stream.

    Strict validation: any event with ``req_index != 0`` is
    treated as a scheduler fault (generated batches in a B=1 run
    must only emit against the single admitted request), and an
    ``aborted`` event short-circuits the run as a failure rather
    than letting the oracle see a truncated stream. Both conditions
    raise ``RuntimeError`` so the outer ``_run_one`` collapses them
    to ``status="failed"`` with a structured ``b1_batched_*``
    reason — never into the oracle's "tokens mismatched" branch.
    """
    tokens: list[int] = []
    for event in engine.generate_batch([prompt], params, max_batch_size=1):
        if event.req_index != 0:
            raise RuntimeError(
                f"b1_batched_unexpected_req_index:{event.req_index}"
            )
        if event.kind == "aborted":
            raise RuntimeError(
                f"b1_batched_aborted:{event.finish_reason}"
            )
        if event.kind == "token":
            if event.token_id is None:
                raise RuntimeError("b1_batched_token_event_missing_id")
            tokens.append(event.token_id)
        # "done" is terminal but carries no token data; accept
        # silently so the loop exits on stream close.
    return tokens


__all__ = ["BenchRunner", "EngineFactory"]
