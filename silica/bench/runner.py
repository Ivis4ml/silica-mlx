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
  * **Workload shape validated per oracle**. Oracle kinds dictate
    admissible ``max_batch_size`` and ``len(prompts)`` combinations
    (SMOKE / B1_PARITY require B=1 + one prompt; BGT1_PARITY
    requires B>=2 + at least two prompts). Misconfigurations
    surface as ``status="failed"`` with a structured reason, not
    ``skipped`` — the scenario is authoring-broken, not
    environmentally blocked.

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
  * ``BGT1_DIRECT_BATCHED_REFERENCE`` — builds a direct mlx-lm
    batched reference with
    ``adapter.make_batch_cache(left_padding)`` then runs Silica's
    ``Engine.generate_batch``. Both paths use
    ``stop_token_ids=()`` (overriding ``_build_sampling_params``)
    so the reference — which runs unconditionally for
    ``max_tokens`` — stays length-aligned with Silica's stream
    regardless of EOS. Event-stream validation mirrors the B=1
    path: ``aborted`` events or unexpected ``req_index`` values
    raise before the oracle runs, so scheduler faults never
    masquerade as parity mismatches.

Exception handling is deliberately coarse: any exception during
load / generate / oracle collapses to ``status="failed"`` with the
exception type + message. Metrics snapshot reflects whichever
execution path populates the shared ``MetricsRegistry``
(``Engine.generate`` sets it; ``Engine.generate_batch`` does not),
so for B1 parity the snapshot is the single-request reference and
for BGT1 parity every metric is ``None``; ``wall_s`` covers all
executions the oracle consumes end-to-end.

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

_SINGLE_REQUEST_ORACLES: frozenset[OracleKind] = frozenset(
    {OracleKind.SMOKE, OracleKind.B1_PARITY_VS_SINGLE}
)
_BGT1_ORACLES: frozenset[OracleKind] = frozenset(
    {OracleKind.BGT1_DIRECT_BATCHED_REFERENCE}
)

EngineFactory = Callable[[Scenario], tuple[ModelAdapter, Engine]]
DirectBatchedReferenceFn = Callable[
    [ModelAdapter, list[str], SamplingParams], dict[int, list[int]]
]


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
        direct_batched_reference: DirectBatchedReferenceFn | None = None,
        out_path: Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
        reset_peak: Callable[[], None] | None = None,
        read_peak_mb: Callable[[], float | None] | None = None,
    ) -> None:
        self._engine_factory: EngineFactory = (
            engine_factory if engine_factory is not None else _default_engine_factory
        )
        self._direct_batched_reference: DirectBatchedReferenceFn = (
            direct_batched_reference
            if direct_batched_reference is not None
            else _direct_mlx_lm_batched_reference
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

        workload_error = _validate_workload_for_oracle(
            scenario.oracle, scenario.workload
        )
        if workload_error is not None:
            return ScenarioResult(
                scenario_id=scenario.id,
                status="failed",
                reason=workload_error,
            )

        try:
            self._reset_peak()
            t_start = self._clock()
            adapter, engine = self._engine_factory(scenario)

            # oracle_input's concrete type varies by oracle kind
            # (list[int] for single-request / B1 parity, dict[int,
            # list[int]] for BGT1). OracleFn's second arg is Any
            # precisely to allow this, but mypy needs one explicit
            # annotation so the branches do not each try to narrow
            # a single inferred type.
            oracle_input: Any
            oracle_context: dict[str, Any]
            total_tokens: int

            if scenario.oracle == OracleKind.B1_PARITY_VS_SINGLE:
                params = _build_sampling_params(scenario.workload, adapter)
                b1_tokens, oracle_context = _run_b1_parity(
                    scenario, engine, adapter, params
                )
                oracle_input = b1_tokens
                total_tokens = len(b1_tokens)
            elif scenario.oracle == OracleKind.BGT1_DIRECT_BATCHED_REFERENCE:
                params = _build_sampling_params(
                    scenario.workload, adapter, include_eos=False
                )
                bgt1_tokens, oracle_context = self._run_bgt1_parity(
                    scenario, engine, adapter, params
                )
                oracle_input = bgt1_tokens
                total_tokens = sum(
                    len(row) for row in bgt1_tokens.values()
                )
            else:
                params = _build_sampling_params(scenario.workload, adapter)
                single_tokens = _run_single_request(scenario, engine, params)
                oracle_input = single_tokens
                oracle_context = {"vocab_size": adapter.config.vocab_size}
                total_tokens = len(single_tokens)

            wall_s = self._clock() - t_start
            peak_mb = self._read_peak_mb()
            oracle_fn = ORACLES[scenario.oracle]
            ok, oracle_reason, metadata = oracle_fn(
                scenario, oracle_input, oracle_context
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
            total_tokens=total_tokens,
            wall_s=wall_s,
            metadata=dict(metadata),
        )

    def _append_jsonl(self, result: ScenarioResult) -> None:
        assert self._out_path is not None
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def _run_bgt1_parity(
        self,
        scenario: Scenario,
        engine: Engine,
        adapter: ModelAdapter,
        params: SamplingParams,
    ) -> tuple[dict[int, list[int]], dict[str, Any]]:
        """Direct mlx-lm reference first, then Silica batched.

        The direct-reference step goes through
        ``self._direct_batched_reference`` so tests can inject a
        pre-computed row dict instead of re-running mlx-lm. The
        default implementation uses
        :func:`_direct_mlx_lm_batched_reference`, which drives the
        adapter's underlying model with the same
        ``adapter.make_batch_cache`` + left-padding convention as
        ``tests/test_p3_gemma4_batched_parity.py``.
        """
        prompts = list(scenario.workload.prompts)
        reference = self._direct_batched_reference(adapter, prompts, params)
        silica_batched = _collect_bgt1_batched_tokens(
            engine, prompts, params, scenario.workload.max_batch_size
        )
        return silica_batched, {
            "vocab_size": adapter.config.vocab_size,
            "reference_tokens": reference,
        }


def _validate_workload_for_oracle(
    oracle: OracleKind, wl: Workload
) -> str | None:
    """Return an authoring-error reason string, or ``None`` if OK.

    SMOKE / B1_PARITY scenarios must be single-request (B=1, one
    prompt); BGT1_PARITY scenarios must carry at least two prompts
    and a batch size >= 2. Misconfigurations land as
    ``status="failed"`` rather than ``skipped`` because the
    scenario is broken as authored, not blocked by the environment.
    """
    if oracle in _SINGLE_REQUEST_ORACLES:
        if wl.max_batch_size > 1:
            return (
                f"oracle {oracle.value!r} requires max_batch_size=1, "
                f"got {wl.max_batch_size}"
            )
        if len(wl.prompts) != 1:
            return (
                f"oracle {oracle.value!r} requires exactly 1 prompt, "
                f"got {len(wl.prompts)}"
            )
    elif oracle in _BGT1_ORACLES:
        if wl.max_batch_size < 2:
            return (
                f"oracle {oracle.value!r} requires max_batch_size>=2, "
                f"got {wl.max_batch_size}"
            )
        if len(wl.prompts) < 2:
            return (
                f"oracle {oracle.value!r} requires at least 2 prompts, "
                f"got {len(wl.prompts)}"
            )
    # Unknown oracle kinds (e.g. TEACHER_FORCED_ARGMAX stub) pass
    # this validation unchanged — their stub oracle raises
    # NotImplementedError once dispatched.
    return None


def _build_sampling_params(
    workload: Workload,
    adapter: ModelAdapter,
    *,
    include_eos: bool = True,
) -> SamplingParams:
    """Shared SamplingParams helper.

    Both sides of a parity oracle pull from this one helper so
    divergent tokens cannot be blamed on accidentally drifted
    sampling params. ``include_eos`` is ``True`` by default (EOS
    from the tokenizer populates ``stop_token_ids``); BGT1 parity
    sets it to ``False`` because the direct mlx-lm reference runs
    unconditionally for ``max_tokens``, so Silica's batched path
    must also ignore EOS to keep the two streams length-aligned.
    """
    if include_eos:
        tokenizer = adapter.tokenizer()
        eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))
    else:
        eos_ids = ()
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


def _collect_bgt1_batched_tokens(
    engine: Engine,
    prompts: list[str],
    params: SamplingParams,
    max_batch_size: int,
) -> dict[int, list[int]]:
    """Collect per-row token events from a B>1 ``generate_batch`` stream.

    Validates the event stream the same way :func:`_collect_b1_batched_tokens`
    does, plus the stronger B>1 invariant that ``req_index`` must
    fall in ``range(len(prompts))``. Aborts or out-of-range row ids
    raise ``RuntimeError`` so the outer ``_run_one`` collapses them
    to a structured ``bgt1_batched_*`` failure before the oracle
    runs — keeping the "the scheduler misbehaved" signal distinct
    from "per-row tokens diverged".
    """
    expected_rows = set(range(len(prompts)))
    tokens: dict[int, list[int]] = {row: [] for row in expected_rows}
    done_rows: set[int] = set()

    for event in engine.generate_batch(
        prompts, params, max_batch_size=max_batch_size
    ):
        if event.req_index not in expected_rows:
            raise RuntimeError(
                f"bgt1_batched_unexpected_req_index:{event.req_index}"
            )
        if event.kind == "aborted":
            raise RuntimeError(
                f"bgt1_batched_aborted:row={event.req_index}_"
                f"reason={event.finish_reason}"
            )
        if event.kind == "token":
            if event.token_id is None:
                raise RuntimeError(
                    f"bgt1_batched_token_event_missing_id:row={event.req_index}"
                )
            tokens[event.req_index].append(event.token_id)
        elif event.kind == "done":
            done_rows.add(event.req_index)

    missing = expected_rows - done_rows
    if missing:
        raise RuntimeError(
            f"bgt1_batched_rows_never_completed:{sorted(missing)}"
        )
    return tokens


def _direct_mlx_lm_batched_reference(
    adapter: ModelAdapter,
    prompts: list[str],
    params: SamplingParams,
) -> dict[int, list[int]]:
    """Drive ``adapter``'s underlying mlx-lm model directly in batched mode.

    Mirrors the structure of
    ``tests/test_p3_gemma4_batched_parity.py::_direct_mlx_lm_batched_tokens``:
    pad each prompt to the left with token id ``0`` so the window
    ends at the same column, build the per-layer cache via
    ``adapter.make_batch_cache(left_padding)`` so SLIDING /
    HYBRID_DELTANET adapters get the right cache types, feed the
    padded matrix, argmax row-by-row, and loop for
    ``params.max_tokens - 1`` additional decode steps.

    The reference intentionally does not early-stop on EOS: runner
    callers pass ``stop_token_ids=()`` to both sides so Silica's
    batched stream also runs the full budget and the per-row
    lengths match.
    """
    import mlx.core as mx  # noqa: PLC0415 — optional import, runner path only

    from silica.weights.resident import ResidentWeightProvider  # noqa: PLC0415

    model = adapter.build(ResidentWeightProvider())
    tokenizer = adapter.tokenizer()
    encoded = [list(tokenizer.encode(prompt)) for prompt in prompts]
    if not all(encoded):
        raise RuntimeError(
            "bgt1_batched_reference_empty_prompt_after_tokenize"
        )

    max_len = max(len(ids) for ids in encoded)
    left_padding = [max_len - len(ids) for ids in encoded]
    padded = [
        [0] * pad + ids
        for pad, ids in zip(left_padding, encoded, strict=True)
    ]
    make_batch_cache = getattr(adapter, "make_batch_cache", None)
    if not callable(make_batch_cache):
        raise RuntimeError(
            f"adapter {type(adapter).__name__} does not expose "
            "make_batch_cache — required for BGT1 direct reference"
        )
    cache = make_batch_cache(left_padding)

    tokens_arr = mx.array(padded, dtype=mx.int32)
    logits = model(tokens_arr, cache=cache)
    mx.eval(logits)
    last = logits[:, -1, :]
    next_tokens = [
        int(mx.argmax(last[row_idx]).item()) for row_idx in range(len(prompts))
    ]
    out: dict[int, list[int]] = {
        row_idx: [tok] for row_idx, tok in enumerate(next_tokens)
    }

    for _ in range(params.max_tokens - 1):
        decode = mx.array([[tok] for tok in next_tokens], dtype=mx.int32)
        logits = model(decode, cache=cache)
        mx.eval(logits)
        last = logits[:, -1, :]
        next_tokens = [
            int(mx.argmax(last[row_idx]).item())
            for row_idx in range(len(prompts))
        ]
        for row_idx, tok in enumerate(next_tokens):
            out[row_idx].append(tok)

    return out


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


__all__ = ["BenchRunner", "DirectBatchedReferenceFn", "EngineFactory"]
