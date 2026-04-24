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
    (SMOKE allows B=1 with one prompt *or* B>1 with >=2 prompts;
    B1_PARITY requires B=1 + one prompt; BGT1_PARITY requires
    B>=2 + at least two prompts). Misconfigurations surface as
    ``status="failed"`` with a structured reason, not ``skipped``
    — the scenario is authoring-broken, not environmentally
    blocked.

Oracle-driven dispatch:

The runner picks its workload-execution path by
``scenario.oracle``:

  * ``SMOKE`` — single-request via ``Engine.generate`` when
    ``max_batch_size == 1``; multi-row via
    ``Engine.generate_batch`` when ``max_batch_size > 1``. In the
    batched case the per-row token streams collapse to a
    ``dict[int, list[int]]`` passed to the oracle; single-request
    still passes a flat ``list[int]`` for compatibility. Both
    paths receive ``context={'vocab_size': N}``. If
    ``workload.prefix_cache`` is true the batched path builds a
    fresh :class:`RadixPrefixCache` (block_size=16) backed by
    :class:`SyntheticPrefixBlockStore` and passes it to
    ``generate_batch`` — enabling shared-prefix scenarios to
    reuse prefix blocks across rows in the same batch.
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
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

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

_BGT1_ORACLES: frozenset[OracleKind] = frozenset(
    {OracleKind.BGT1_DIRECT_BATCHED_REFERENCE}
)

EngineFactory = Callable[[Scenario], tuple[ModelAdapter, Engine]]
DirectBatchedReferenceFn = Callable[
    [ModelAdapter, list[str], SamplingParams], dict[int, list[int]]
]
# Teacher-forced argmax has two injectable hooks because the
# Silica path and the reference path drive different code
# (adapter.prefill/decode_step vs direct mlx-lm forward). Tests
# inject pre-computed prediction lists; on-device uses the
# module-level default implementations.
TeacherForcedSilicaFn = Callable[
    [ModelAdapter, Engine, list[int], list[int]], list[int]
]
TeacherForcedReferenceFn = Callable[
    [ModelAdapter, list[int], list[int]], list[int]
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

    ``OracleKind.PPL`` rows additionally require the pre-extracted
    WikiText text file at ``oracle_config['wikitext_path']`` to be
    present on disk. Skipping before the engine load avoids loading
    ~600 MB of Qwen3-0.6B weights only to fail on a missing
    tokenizer input; the cost asymmetry between "HF cache hit +
    wikitext missing" and "HF cache hit + wikitext present" is
    large enough to warrant a gate check, not a runtime raise.
    """
    cache = hf_cache_path_for_repo(scenario.repo)
    if not cache.exists():
        return f"cache_missing:{cache}"
    if scenario.gate_env_var is not None:
        if os.environ.get(scenario.gate_env_var) != "1":
            return f"env_var_not_set:{scenario.gate_env_var}"
    if scenario.oracle == OracleKind.PPL:
        wikitext_path = scenario.oracle_config.get("wikitext_path")
        if wikitext_path is None:
            return "ppl_wikitext_path_missing_in_oracle_config"
        wp = Path(str(wikitext_path))
        if not wp.is_file():
            return f"wikitext_cache_missing:{wp}"
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
        teacher_forced_silica: TeacherForcedSilicaFn | None = None,
        teacher_forced_reference: TeacherForcedReferenceFn | None = None,
        out_path: Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
        reset_peak: Callable[[], None] | None = None,
        read_peak_mb: Callable[[], float | None] | None = None,
        seeds: Sequence[int] = (0,),
    ) -> None:
        self._engine_factory: EngineFactory = (
            engine_factory if engine_factory is not None else _default_engine_factory
        )
        self._direct_batched_reference: DirectBatchedReferenceFn = (
            direct_batched_reference
            if direct_batched_reference is not None
            else _direct_mlx_lm_batched_reference
        )
        self._teacher_forced_silica: TeacherForcedSilicaFn = (
            teacher_forced_silica
            if teacher_forced_silica is not None
            else _default_teacher_forced_silica
        )
        self._teacher_forced_reference: TeacherForcedReferenceFn = (
            teacher_forced_reference
            if teacher_forced_reference is not None
            else _default_teacher_forced_reference
        )
        self._out_path = out_path
        self._clock = clock
        self._reset_peak = reset_peak or _mlx_reset_peak_memory
        self._read_peak_mb = read_peak_mb or _mlx_peak_memory_mb
        # P-5-C.4 step 1: multi-seed fan-out. Defaults to (0,) so an
        # omitted --seeds flag produces the same single-row output
        # shape as pre-C.4. Validate every seed here as well as at
        # the CLI parser: the CLI is one entry, but BenchRunner is a
        # public API called directly by tests and future programmatic
        # callers. Catching a bad seed at construction produces a
        # ValueError the caller sees immediately instead of a
        # numpy / mlx crash inside _run_one, which happens before
        # the per-scenario exception boundary and would abort the
        # whole run rather than collapsing to a failed row.
        if not seeds:
            raise ValueError(
                "BenchRunner requires at least one seed; pass "
                "seeds=(0,) for the single-seed default behaviour"
            )
        seeds_tuple: tuple[int, ...] = tuple(seeds)
        for seed in seeds_tuple:
            # ``isinstance(seed, bool)`` catches ``True`` / ``False``
            # being passed where an int was meant: booleans are ints
            # in Python (``isinstance(True, int)`` is True) and
            # NumPy would silently accept them as seed 0 / seed 1,
            # turning a type error into a coincidence.
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise TypeError(
                    f"BenchRunner seeds must be int; got "
                    f"{type(seed).__name__} ({seed!r})"
                )
            if not (0 <= seed < (1 << 32)):
                raise ValueError(
                    f"BenchRunner seed {seed} is out of range; "
                    f"must satisfy 0 <= seed < 2**32 "
                    f"({1 << 32})"
                )
        self._seeds: tuple[int, ...] = seeds_tuple

    def run(self, scenarios: Iterable[Scenario]) -> list[ScenarioResult]:
        """Run every scenario × seed in scenario-major order.

        For ``seeds=(s0, s1, ..., sK)`` and scenarios ``[A, B]`` the
        result order is ``[(A,s0), (A,s1), ..., (A,sK), (B,s0), ...]``
        so same-scenario rows stay adjacent — both for terminal
        readability (a watcher sees scenario A finish all seeds
        before B starts) and for bench-report grouping (downstream
        aggregation can consume contiguous slices).

        Rows are appended to ``out_path`` as they complete (one JSON
        object per line) so a long-running bench does not lose
        progress if the process is killed mid-run. Each JSONL row
        carries ``metadata["seed"]`` so the file is self-describing
        without reference to the CLI invocation.
        """
        results: list[ScenarioResult] = []
        for scenario in scenarios:
            for seed in self._seeds:
                result = self._run_one(scenario, seed=seed)
                results.append(result)
                if self._out_path is not None:
                    self._append_jsonl(result)
        return results

    def _run_one(
        self, scenario: Scenario, *, seed: int
    ) -> ScenarioResult:
        # Seed all three RNGs BEFORE the gate check so anything
        # downstream (adapter load, factory-driven prompt selection,
        # future stochastic oracles) runs deterministically for a
        # given (scenario, seed) pair. Current oracle implementations
        # are deterministic, so this is effectively a no-op + label;
        # when an oracle eventually consumes randomness this seeding
        # makes the result reproducible without a second commit.
        mx.random.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Gate check first: skipped rows never touch the engine, so
        # cache-missing scenarios do not accidentally block on a
        # download prompt or a model-factory KeyError.
        skip_reason = _check_gates(scenario)
        if skip_reason is not None:
            return ScenarioResult(
                scenario_id=scenario.id,
                status="skipped",
                reason=skip_reason,
                metadata={"seed": seed},
            )

        workload_error = _validate_workload_for_oracle(
            scenario.oracle, scenario.workload
        )
        if workload_error is not None:
            return ScenarioResult(
                scenario_id=scenario.id,
                status="failed",
                reason=workload_error,
                metadata={"seed": seed},
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
            elif scenario.oracle == OracleKind.TEACHER_FORCED_ARGMAX:
                tf_predictions, oracle_context = self._run_teacher_forced_argmax(
                    scenario, engine, adapter
                )
                oracle_input = tf_predictions
                total_tokens = len(tf_predictions)
            elif scenario.oracle == OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT:
                prefix_hit_collected, oracle_context = _run_prefix_hit_decode(
                    scenario, engine, adapter
                )
                oracle_input = prefix_hit_collected
                tokens_dict, _ = prefix_hit_collected
                total_tokens = sum(len(row) for row in tokens_dict.values())
            elif scenario.oracle == OracleKind.PPL:
                # PPL bypasses engine.generate_batch entirely — the
                # oracle is teacher-forced and needs positional logits,
                # not sampled tokens. ``_run_ppl`` drives the adapter
                # directly via silica.bench.ppl_oracle.
                ppl_collected, oracle_context = _run_ppl(scenario, adapter)
                oracle_input = ppl_collected
                total_tokens = int(ppl_collected["n_tokens"])
            elif scenario.oracle == OracleKind.STORAGE:
                # STORAGE drives the same shared-prefix 2-prompt
                # workload DECODE_TOK_S_WITH_PREFIX_HIT uses, then
                # reads resident_bytes off the store after the
                # event stream drains. No per-row token stream is
                # inspected; total_tokens reports the number of
                # detached blocks registered so the JSONL row still
                # carries a useful counter.
                storage_collected, oracle_context = _run_storage(
                    scenario, engine, adapter
                )
                oracle_input = storage_collected
                total_tokens = int(storage_collected["live_blocks"])
            elif scenario.oracle == OracleKind.ADMISSION_HEADROOM:
                # ADMISSION_HEADROOM bypasses engine.generate_batch.
                # The runner still loaded adapter via factory (v1
                # accepts the engine-construction cost); the engine
                # itself is unused. total_tokens reports the sum of
                # fp16 + compressed admission counts so the JSONL
                # row has a non-trivial counter.
                ah_collected, oracle_context = _run_admission_headroom(
                    scenario, adapter
                )
                oracle_input = ah_collected
                total_tokens = int(
                    ah_collected["n_fp16"] + ah_collected["n_block"]
                )
            else:
                # SMOKE (and any future single/batched-flexible
                # oracle). B=1 takes the single-request path,
                # B>1 goes through generate_batch with an optional
                # prefix cache.
                params = _build_sampling_params(scenario.workload, adapter)
                wl = scenario.workload
                oracle_context = {"vocab_size": adapter.config.vocab_size}
                if wl.max_batch_size == 1:
                    single_tokens = _run_single_request(
                        scenario, engine, params
                    )
                    oracle_input = single_tokens
                    total_tokens = len(single_tokens)
                else:
                    batched_tokens, first_token_ms = _run_smoke_batched(
                        scenario, engine, params, adapter
                    )
                    oracle_input = batched_tokens
                    total_tokens = sum(
                        len(row) for row in batched_tokens.values()
                    )
                    oracle_context["first_token_ms_per_row"] = first_token_ms

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
                metadata={"seed": seed},
            )

        snap = engine.metrics.snapshot()
        # For DECODE_TOK_S_WITH_PREFIX_HIT the headline metrics come
        # from the oracle's per-row timestamp analysis rather than
        # engine.metrics — ``Engine.generate_batch`` does not populate
        # the ``MetricsRegistry`` today, so ``snap.decode_tok_s`` would
        # be ``None`` on this path even though the oracle has computed
        # an honest row-1 decode tok/s in metadata. Promoting those
        # numbers into the standard ``ScenarioResult.decode_tok_s`` /
        # ``ttft_ms`` fields keeps the JSONL schema homogeneous across
        # oracles — downstream tools (bench report, A.3c acceptance
        # gate) do not need to special-case this oracle's metadata
        # bucket.
        ttft_ms = snap.ttft_ms
        decode_tok_s = snap.decode_tok_s
        if (
            ok
            and scenario.oracle == OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT
        ):
            row1_decode = metadata.get("row1_decode_tok_s")
            if isinstance(row1_decode, (int, float)):
                decode_tok_s = float(row1_decode)
            row1_first = metadata.get("row1_first_token_ms")
            if isinstance(row1_first, (int, float)):
                ttft_ms = float(row1_first)

        # Runner-injected ``seed`` wins if an oracle accidentally
        # writes the key: seed is an execution dimension (runner
        # fan-out), not an authoring dimension (oracle_config). The
        # ``{**metadata, "seed": seed}`` ordering enforces that
        # precedence without requiring every oracle to know about
        # the seed key.
        result_metadata: dict[str, Any] = {**metadata, "seed": seed}
        return ScenarioResult(
            scenario_id=scenario.id,
            status="ok" if ok else "failed",
            reason=oracle_reason,
            ttft_ms=ttft_ms,
            prefill_tok_s=snap.prefill_tok_s,
            decode_tok_s=decode_tok_s,
            resident_mb=snap.resident_mb,
            peak_memory_mb=peak_mb,
            total_tokens=total_tokens,
            wall_s=wall_s,
            metadata=result_metadata,
        )

    def _append_jsonl(self, result: ScenarioResult) -> None:
        assert self._out_path is not None
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def _run_teacher_forced_argmax(
        self,
        scenario: Scenario,
        engine: Engine,
        adapter: ModelAdapter,
    ) -> tuple[list[int], dict[str, Any]]:
        """Drive both the silica and reference teacher-forced paths.

        Pulls the target continuation text from
        ``scenario.oracle_config``, tokenises with the adapter's
        tokenizer, then invokes the injectable silica / reference
        hooks (defaults: drive ``adapter.prefill`` + ``decode_step``
        for silica; direct mlx-lm single forward for reference).
        Oracle context carries the reference predictions, the
        target token ids, and the pass-rate threshold.

        Any missing oracle_config key (``target_continuation``,
        ``min_agreement_rate``) raises a ``RuntimeError`` that the
        outer ``_run_one`` collapses to ``status="failed"`` — the
        scenario is authoring-broken, not environmentally blocked.
        """
        target_text = scenario.oracle_config.get("target_continuation")
        if not isinstance(target_text, str) or not target_text:
            raise RuntimeError(
                "teacher_forced_argmax_missing_oracle_config_target_continuation"
            )
        threshold = scenario.oracle_config.get("min_agreement_rate", 0.98)

        tokenizer = adapter.tokenizer()
        prompt = scenario.workload.prompts[0]
        prompt_ids = list(tokenizer.encode(prompt))
        target_ids = list(tokenizer.encode(target_text))
        if not prompt_ids:
            raise RuntimeError("teacher_forced_argmax_empty_prompt")
        if not target_ids:
            raise RuntimeError("teacher_forced_argmax_empty_target")

        silica_predictions = self._teacher_forced_silica(
            adapter, engine, prompt_ids, target_ids
        )
        reference_predictions = self._teacher_forced_reference(
            adapter, prompt_ids, target_ids
        )
        return silica_predictions, {
            "reference_predictions": reference_predictions,
            "target_tokens": target_ids,
            "min_agreement_rate": float(threshold),
            "vocab_size": adapter.config.vocab_size,
        }

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

    Per-oracle rules:

      * ``SMOKE`` — the flexible "does it crash" floor. Accepts
        ``max_batch_size=1`` with exactly one prompt, or
        ``max_batch_size>1`` with at least two prompts (the B>1
        shape drives :meth:`Engine.generate_batch`).
      * ``B1_PARITY_VS_SINGLE`` — single-request reference plus a
        strict B=1 batched run; B>1 would require a different
        oracle.
      * ``BGT1_DIRECT_BATCHED_REFERENCE`` — needs real batching;
        B=1 would make the claim degenerate.

    Misconfigurations land as ``status="failed"`` rather than
    ``skipped`` because the scenario is broken as authored, not
    blocked by the environment.
    """
    if oracle == OracleKind.SMOKE:
        if wl.max_batch_size < 1:
            return (
                f"max_batch_size must be >= 1, got {wl.max_batch_size}"
            )
        if wl.max_batch_size == 1 and len(wl.prompts) != 1:
            return (
                f"oracle 'smoke' at max_batch_size=1 requires exactly "
                f"1 prompt, got {len(wl.prompts)}"
            )
        if wl.max_batch_size > 1 and len(wl.prompts) < 2:
            return (
                f"oracle 'smoke' at max_batch_size={wl.max_batch_size} "
                f"requires at least 2 prompts, got {len(wl.prompts)}"
            )
    elif oracle == OracleKind.B1_PARITY_VS_SINGLE:
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
    elif oracle == OracleKind.TEACHER_FORCED_ARGMAX:
        if wl.max_batch_size != 1:
            return (
                f"oracle {oracle.value!r} requires max_batch_size=1, "
                f"got {wl.max_batch_size}"
            )
        if len(wl.prompts) != 1:
            return (
                f"oracle {oracle.value!r} requires exactly 1 prompt, "
                f"got {len(wl.prompts)}"
            )
    elif oracle == OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT:
        # Front-load the workload-shape check so scenarios with
        # authoring errors fail BEFORE the adapter + engine are
        # loaded. Previously these validations lived inside
        # ``_run_prefix_hit_decode`` and fired after a multi-GB
        # Qwen3.5-0.8B load.
        if wl.max_batch_size != 1:
            return (
                f"oracle {oracle.value!r} requires max_batch_size=1, "
                f"got {wl.max_batch_size} — row 1 must enter the "
                f"waiting queue, which only happens when the initial "
                f"cohort is size 1"
            )
        if len(wl.prompts) != 2:
            return (
                f"oracle {oracle.value!r} requires exactly 2 prompts, "
                f"got {len(wl.prompts)} — the workload drives row 1 "
                f"into the waiting queue so mid-run admission fires "
                f"the prefix-hit path"
            )
        if wl.prompts[0] != wl.prompts[1]:
            return (
                f"oracle {oracle.value!r} requires identical prompts "
                f"(shared prefix drives row 1 through the full-match "
                f"hit path); got two different prompts"
            )
        if not wl.prefix_cache:
            return (
                f"oracle {oracle.value!r} requires prefix_cache=True"
            )
        if wl.kv_codec is None:
            return (
                f"oracle {oracle.value!r} requires kv_codec to be set "
                f"explicitly (fp16 baseline vs compressed codec is the "
                f"whole point of the measurement); got kv_codec=None"
            )
    elif oracle == OracleKind.STORAGE:
        # Pre-load validation mirrors DECODE_TOK_S_WITH_PREFIX_HIT
        # verbatim: same shared-prefix 2-prompt workload shape so
        # the scheduler's _extract_and_insert_prefix fires on row 0
        # termination and populates the store. Fail-fast before the
        # engine factory loads weights — matches the A.3b review
        # M-2 pattern applied to DECODE_TOK_S_WITH_PREFIX_HIT.
        if wl.max_batch_size != 1:
            return (
                f"oracle {oracle.value!r} requires max_batch_size=1, "
                f"got {wl.max_batch_size} — row 1 must enter the "
                f"waiting queue, which only happens when the initial "
                f"cohort is size 1"
            )
        if len(wl.prompts) != 2:
            return (
                f"oracle {oracle.value!r} requires exactly 2 prompts, "
                f"got {len(wl.prompts)} — workload shape mirrors "
                f"DECODE_TOK_S_WITH_PREFIX_HIT so the same codec hot "
                f"path is exercised"
            )
        if wl.prompts[0] != wl.prompts[1]:
            return (
                f"oracle {oracle.value!r} requires identical prompts "
                f"(shared prefix drives row 1 through the full-match "
                f"hit path); got two different prompts"
            )
        if not wl.prefix_cache:
            return (
                f"oracle {oracle.value!r} requires prefix_cache=True "
                f"— the oracle's observable is resident_bytes on the "
                f"prefix cache's store"
            )
        if wl.kv_codec is None:
            return (
                f"oracle {oracle.value!r} requires kv_codec to be set "
                f"(e.g. 'fp16' for IdentityCodec baseline). The "
                f"pass-through path reports resident_bytes_per_block="
                f"None and breaks cross-row compression comparisons"
            )
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

    P-4.5-B.1 opt-out: the BGT1 oracle compares Silica's B>1 batched
    tokens against a direct mlx-lm ``B=N`` reference
    (``_direct_mlx_lm_batched_reference``) run over the unsplit
    cohort. The default ``length_spread_threshold=2.0`` would split
    heterogeneous-length BGT1 scenarios (e.g. the catalog's
    ``qwen3-0.6b-bgt1-parity`` on prompts ``("Hello", "The capital
    of Japan is")`` with tokenized lengths ``[1, 5]``, ratio ``5.0 >
    2.0``) into ``B=1 + B=1``, which is not what the reference
    exercises. Pin ``float('inf')`` so the oracle continues to
    compare like-for-like. The SMOKE B>1 path
    (``_collect_smoke_batched_tokens``) intentionally does NOT opt
    out — its Q-010 scenario is what the admission reorder is there
    to fix.
    """
    expected_rows = set(range(len(prompts)))
    tokens: dict[int, list[int]] = {row: [] for row in expected_rows}
    done_rows: set[int] = set()

    for event in engine.generate_batch(
        prompts,
        params,
        max_batch_size=max_batch_size,
        length_spread_threshold=float("inf"),
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


def _run_smoke_batched(
    scenario: Scenario,
    engine: Engine,
    params: SamplingParams,
    adapter: ModelAdapter,
) -> tuple[dict[int, list[int]], dict[int, float]]:
    """Drive SMOKE through ``generate_batch`` for multi-prompt B>1.

    Wires a fresh :class:`RadixPrefixCache` when
    ``scenario.workload.prefix_cache`` is ``True`` — each scenario
    gets its own cache instance so shared-prefix metrics reflect
    that run alone. When ``workload.kv_codec`` is set, the cache's
    store is constructed with the named codec so subsequent encode /
    decode path actually exercises it (P-5-A.3a). Returns
    ``(tokens, first_token_ms)`` — per-row token streams plus
    per-row first-token wall offsets in ms (measured from the
    moment the batched generation started, so the TTFT-under-
    concurrency scenario can show how a long-prompt prefill inflates
    short-prompt rows' first-token latency).
    """
    prompts = list(scenario.workload.prompts)
    prefix_cache = _maybe_build_prefix_cache(scenario.workload, adapter)
    return _collect_smoke_batched_tokens(
        engine,
        prompts,
        params,
        max_batch_size=scenario.workload.max_batch_size,
        prefix_cache=prefix_cache,
    )


def _run_prefix_hit_decode(
    scenario: Scenario,
    engine: Engine,
    adapter: ModelAdapter,
) -> tuple[
    tuple[dict[int, list[int]], dict[int, list[float]]],
    dict[str, Any],
]:
    """Drive the P-5-A.3b ``DECODE_TOK_S_WITH_PREFIX_HIT`` workload.

    Validates the workload shape inline (the scenario schema does
    not encode "exactly 2 prompts", only that a workload is legal;
    this oracle pins a specific shape at runtime). Returns the raw
    per-row token streams + per-row per-token wall-clock timestamps
    for the oracle to compute ``row1_decode_tok_s`` from. Context
    carries ``vocab_size`` + the prefix-cache hit counter so the
    oracle can reject the run when the hit path was never taken
    (scheduler regression to miss-path only).
    """
    wl = scenario.workload
    if len(wl.prompts) != 2:
        raise RuntimeError(
            f"decode_tok_s_with_prefix_hit requires exactly 2 prompts, "
            f"got {len(wl.prompts)} — the workload shape drives row 1 "
            f"into the waiting queue so mid-run admission fires the "
            f"hit path"
        )
    if wl.max_batch_size != 1:
        raise RuntimeError(
            f"decode_tok_s_with_prefix_hit requires max_batch_size=1, "
            f"got {wl.max_batch_size} — row 1 must enter the waiting "
            f"queue, which only happens when max_batch_size keeps the "
            f"initial cohort at 1"
        )
    if not wl.prefix_cache:
        raise RuntimeError(
            "decode_tok_s_with_prefix_hit requires prefix_cache=True"
        )
    if wl.prompts[0] != wl.prompts[1]:
        raise RuntimeError(
            "decode_tok_s_with_prefix_hit requires identical prompts "
            "(shared prefix drives row 1 through the full-match hit "
            "path); got two different prompts"
        )

    params = _build_sampling_params(wl, adapter, include_eos=False)
    prefix_cache = _maybe_build_prefix_cache(wl, adapter)
    assert prefix_cache is not None  # wl.prefix_cache=True guarantees this

    collected = _collect_prefix_hit_decode(
        engine, list(wl.prompts), params, prefix_cache=prefix_cache
    )
    context: dict[str, Any] = {
        "vocab_size": adapter.config.vocab_size,
        "prefix_cache_hits": int(prefix_cache.hits),
    }
    return collected, context


def _collect_prefix_hit_decode(
    engine: Engine,
    prompts: list[str],
    params: SamplingParams,
    *,
    prefix_cache: Any,
) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Collect per-row token streams + per-row per-token timestamps.

    Drives ``generate_batch(prompts, params, max_batch_size=1,
    prefix_cache=prefix_cache)``. Token-level timestamps are measured
    from the start of ``generate_batch`` so row 1's first timestamp
    represents cold-start + prefix-hit admission overhead, and the
    inter-token intervals after that are the decode-path throughput
    this oracle measures.

    Returns ``(tokens, timestamps_ms)`` — both dicts keyed by
    ``req_index`` (0, 1). Strict validation mirrors
    ``_collect_smoke_batched_tokens``: aborted events and unexpected
    ``req_index`` values raise ``RuntimeError`` so the outer
    ``run_scenario`` collapses them to a failed ``ScenarioResult``
    before the oracle sees the input.
    """
    expected_rows = set(range(len(prompts)))
    tokens: dict[int, list[int]] = {row: [] for row in expected_rows}
    token_ts_ms: dict[int, list[float]] = {row: [] for row in expected_rows}
    done_rows: set[int] = set()
    t_start = time.perf_counter()

    for event in engine.generate_batch(
        prompts, params, max_batch_size=1, prefix_cache=prefix_cache
    ):
        if event.req_index not in expected_rows:
            raise RuntimeError(
                f"prefix_hit_decode_unexpected_req_index:{event.req_index}"
            )
        if event.kind == "aborted":
            raise RuntimeError(
                f"prefix_hit_decode_aborted:row={event.req_index}_"
                f"reason={event.finish_reason}"
            )
        if event.kind == "token":
            if event.token_id is None:
                raise RuntimeError(
                    f"prefix_hit_decode_token_event_missing_id:"
                    f"row={event.req_index}"
                )
            ms = (time.perf_counter() - t_start) * 1000.0
            token_ts_ms[event.req_index].append(ms)
            tokens[event.req_index].append(event.token_id)
        elif event.kind == "done":
            done_rows.add(event.req_index)

    missing = expected_rows - done_rows
    if missing:
        raise RuntimeError(
            f"prefix_hit_decode_rows_never_completed:{sorted(missing)}"
        )
    return tokens, token_ts_ms


def _collect_smoke_batched_tokens(
    engine: Engine,
    prompts: list[str],
    params: SamplingParams,
    *,
    max_batch_size: int,
    prefix_cache: Any | None,
) -> tuple[dict[int, list[int]], dict[int, float]]:
    """Collect per-row token events + per-row first-token offsets.

    Event-stream validation mirrors :func:`_collect_bgt1_batched_tokens`:
    aborted events and out-of-range ``req_index`` values raise
    ``RuntimeError`` so the outer ``_run_one`` collapses them to
    ``smoke_batched_*`` reasons before the oracle runs. Rows that
    never emit ``done`` are also a scheduler fault rather than an
    oracle concern.

    ``first_token_ms`` is populated on each row's first token
    event and carries the wall-clock offset (ms) from the start
    of ``generate_batch``. This is what the TTFT-under-concurrency
    scenario needs to surface per-row first-token latency —
    ``Engine.generate_batch`` itself does not populate
    ``MetricsRegistry``, so this is the only way a SMOKE B>1 run
    records TTFT today. Oracle merges the dict into per-row
    metadata so the JSONL row is self-describing.
    """
    expected_rows = set(range(len(prompts)))
    tokens: dict[int, list[int]] = {row: [] for row in expected_rows}
    first_token_ms: dict[int, float] = {}
    done_rows: set[int] = set()
    t_start = time.perf_counter()

    for event in engine.generate_batch(
        prompts,
        params,
        max_batch_size=max_batch_size,
        prefix_cache=prefix_cache,
    ):
        if event.req_index not in expected_rows:
            raise RuntimeError(
                f"smoke_batched_unexpected_req_index:{event.req_index}"
            )
        if event.kind == "aborted":
            raise RuntimeError(
                f"smoke_batched_aborted:row={event.req_index}_"
                f"reason={event.finish_reason}"
            )
        if event.kind == "token":
            if event.token_id is None:
                raise RuntimeError(
                    f"smoke_batched_token_event_missing_id:"
                    f"row={event.req_index}"
                )
            if event.req_index not in first_token_ms:
                first_token_ms[event.req_index] = (
                    (time.perf_counter() - t_start) * 1000.0
                )
            tokens[event.req_index].append(event.token_id)
        elif event.kind == "done":
            done_rows.add(event.req_index)

    missing = expected_rows - done_rows
    if missing:
        raise RuntimeError(
            f"smoke_batched_rows_never_completed:{sorted(missing)}"
        )
    return tokens, first_token_ms


def _default_teacher_forced_silica(
    adapter: ModelAdapter,
    engine: Engine,
    prompt_ids: list[int],
    target_ids: list[int],
) -> list[int]:
    """Drive Silica's ``adapter.prefill`` + ``decode_step`` with
    teacher-forced target tokens; argmax at each position.

    Returns ``len(target_ids)`` predictions: the first from the
    prefill logits (predicting ``target[0]``) and the rest from
    decoding each ``target[i]`` into step logits (predicting
    ``target[i+1]``). The last target token is NOT decoded because
    its next-position logit would predict *past* the target window.

    Uses ``engine.kv_manager`` directly so we do not need to spin
    up a second ``SimpleKVCache`` for this run; the handle is
    local (``req_id="bench-teacher-forced"``) and freed in
    ``finally``.
    """
    import mlx.core as mx  # noqa: PLC0415 — optional import, runner path only

    from silica.kvcache.manager import KVHandle  # noqa: PLC0415

    req_id = "bench-teacher-forced"
    handle = KVHandle(req_id=req_id)
    kv_manager = engine.kv_manager
    kv_manager.reserve_for_prefill(req_id, prompt_ids)
    try:
        prompt_arr = mx.array(prompt_ids, dtype=mx.int32)
        predictions: list[int] = []
        logits, _ = adapter.prefill(prompt_arr, handle)
        predictions.append(int(mx.argmax(logits).item()))
        for i in range(len(target_ids) - 1):
            step_in = mx.array([target_ids[i]], dtype=mx.int32)
            logits, _ = adapter.decode_step(step_in, handle)
            predictions.append(int(mx.argmax(logits).item()))
        return predictions
    finally:
        kv_manager.free(req_id)


def _default_teacher_forced_reference(
    adapter: ModelAdapter,
    prompt_ids: list[int],
    target_ids: list[int],
) -> list[int]:
    """Direct mlx-lm reference: one forward over ``prompt + target[:-1]``,
    slice positional logits at the N teacher-forced positions,
    argmax each.

    Uses ``adapter.make_batch_cache([0])`` for a fresh B=1 cache
    list so the reference run has no state shared with whatever
    the engine is carrying. Returns ``len(target_ids)`` predictions
    aligned one-for-one with the silica path.
    """
    import mlx.core as mx  # noqa: PLC0415 — optional import, runner path only

    from silica.weights.resident import ResidentWeightProvider  # noqa: PLC0415

    model = adapter.build(ResidentWeightProvider())
    make_batch_cache = getattr(adapter, "make_batch_cache", None)
    if not callable(make_batch_cache):
        raise RuntimeError(
            f"adapter {type(adapter).__name__} does not expose "
            "make_batch_cache — required for teacher-forced reference"
        )
    cache = make_batch_cache([0])

    prompt_len = len(prompt_ids)
    target_len = len(target_ids)
    # Feed prompt + target[:-1]. The logit at position i predicts
    # position i+1, so positions [prompt_len-1 : prompt_len-1+target_len]
    # predict target[0 : target_len].
    input_ids = prompt_ids + target_ids[:-1]
    tokens_arr = mx.array([input_ids], dtype=mx.int32)  # (1, prompt_len + target_len - 1)
    logits = model(tokens_arr, cache=cache)  # (1, T, V)
    mx.eval(logits)
    positional = logits[0, prompt_len - 1 : prompt_len - 1 + target_len, :]
    return [int(mx.argmax(positional[i]).item()) for i in range(target_len)]


def _run_admission_headroom(
    scenario: Scenario, adapter: ModelAdapter
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drive the P-5-C.3 step 2 ``OracleKind.ADMISSION_HEADROOM``
    row (§4.7 mode (B) vs mode (C) admission comparison).

    Bypasses ``engine.generate_batch`` entirely — the signal is
    about ``MemoryBudgeter.headroom_bytes()`` arithmetic under
    different store residencies, not about token generation.
    ``_run_one`` still calls ``engine_factory`` to materialize
    the adapter; the returned engine is discarded. v1 accepts
    that weight load cost (~600 MB for Qwen3-0.6B) rather than
    inventing a metadata-only adapter loader in this commit.

    Procedure mirrors opening §7(c) verbatim:

    1. Build a prefix cache with the fp16 codec (default
       ``"fp16"`` = IdentityCodec). Synthesize zero-filled K/V
       blocks with unique token sequences and register them via
       ``prefix_cache.insert_detached`` one at a time until
       ``store.resident_bytes() >= cap_bytes * warmup_ratio``.
       Record the ordered recipe
       ``[(tokens_tuple, per_layer_kv), ...]``.
    2. Build a parallel prefix cache with the compressed codec
       (default ``"block_tq_b64_b4"``). **Replay the same
       recipe** — identical token sequences, identical K/V
       tensors. Under the compressed codec each block lands at
       smaller ``resident_bytes`` than the fp16 equivalent, so
       mode (C) ends up with more headroom at the same bound.
    3. For each prefix cache, construct a
       ``MemoryBudgeter.for_adapter`` with
       ``account_prefix_residency=True`` and call ``admit(...)``
       in a loop using ``(n_prompt, max_tokens)`` from
       ``oracle_config``. Count **consecutive**
       ``AdmitDecision`` returns and stop at the first non-
       ``AdmitDecision``. Eviction- and preemption-assisted
       admits are NOT counted — they mix admission-policy signal
       into what should be a pure-headroom comparison.

    ``oracle_config`` keys:

    - ``cap_bytes`` (int, required): headroom cap for the
      budgeter.
    - ``weights_bytes`` (int, required): weights subtracted from
      ``cap_bytes`` in ``headroom_bytes()``. ``0`` is legal and
      useful for isolating pure prefix-residency signal.
    - ``warmup_ratio`` (float, default ``0.5``): fraction of
      ``cap_bytes`` the fp16 store must reach before admissions
      start. ``0 < ratio < 1``.
    - ``n_prompt`` (int, required): synthetic prompt length per
      trial admission.
    - ``max_tokens`` (int, required): synthetic max_tokens per
      trial admission.
    - ``fp16_codec`` (str, default ``"fp16"``): registry id for
      the mode-(B) codec. IdentityCodec is the reference baseline
      §7(c) names; this knob exists so a future "double-codec
      regression" row can pin a specific fp16-pass-through
      variant without redefining the oracle.
    - ``compressed_codec`` (str, default ``"block_tq_b64_b4"``):
      registry id for the mode-(C) codec.

    Safety rails:

    - A warmup loop that would run indefinitely (per-block
      residency too small relative to the cap target) aborts at
      ``_ADMISSION_HEADROOM_WARMUP_BLOCK_CAP`` blocks and raises
      ``RuntimeError``.
    - If fp16-codec warmup never reaches the target, raises.
    - If an admission loop would run indefinitely (every call
      returns ``AdmitDecision``), bound at
      ``_ADMISSION_HEADROOM_ADMIT_CAP`` iterations and raise.
    """
    from silica.kvcache.prefix import RadixPrefixCache
    from silica.kvcache.store import SyntheticPrefixBlockStore
    from silica.scheduler.budget import (
        AdmitDecision,
        MemoryBudgeter,
    )

    cfg = scenario.oracle_config
    for required in ("cap_bytes", "weights_bytes", "n_prompt", "max_tokens"):
        if required not in cfg:
            raise RuntimeError(
                f"admission_headroom oracle_config missing required "
                f"key {required!r}"
            )
    cap_bytes = int(cfg["cap_bytes"])
    weights_bytes = int(cfg["weights_bytes"])
    warmup_ratio = float(cfg.get("warmup_ratio", 0.5))
    n_prompt = int(cfg["n_prompt"])
    max_tokens = int(cfg["max_tokens"])
    fp16_codec_id = str(cfg.get("fp16_codec", "fp16"))
    compressed_codec_id = str(
        cfg.get("compressed_codec", "block_tq_b64_b4")
    )

    if cap_bytes <= 0:
        raise RuntimeError(
            f"admission_headroom oracle_config cap_bytes must be > 0, "
            f"got {cap_bytes}"
        )
    if weights_bytes < 0:
        raise RuntimeError(
            f"admission_headroom oracle_config weights_bytes must be "
            f">= 0, got {weights_bytes}"
        )
    if not (0.0 < warmup_ratio < 1.0):
        raise RuntimeError(
            f"admission_headroom oracle_config warmup_ratio must be "
            f"in (0, 1); got {warmup_ratio}"
        )
    if n_prompt <= 0 or max_tokens <= 0:
        raise RuntimeError(
            f"admission_headroom oracle_config requires n_prompt>0 and "
            f"max_tokens>0; got n_prompt={n_prompt}, max_tokens="
            f"{max_tokens}"
        )

    block_size = 16  # matches _maybe_build_prefix_cache
    warmup_target = int(cap_bytes * warmup_ratio)
    if warmup_target <= 0:
        raise RuntimeError(
            f"admission_headroom warmup_target computed as "
            f"{warmup_target}; increase cap_bytes or warmup_ratio"
        )

    layout = adapter.kv_layout()

    def _build_prefix_cache(codec_id: str) -> Any:
        from silica.bench.codec_registry import get_codec_spec

        spec = get_codec_spec(codec_id)
        if not (spec.k_supported and spec.v_supported):
            raise RuntimeError(
                f"admission_headroom requires a symmetric codec for "
                f"shorthand install; codec {codec_id!r} has "
                f"k_supported={spec.k_supported}, "
                f"v_supported={spec.v_supported}"
            )
        codec = spec.factory(
            block_size=block_size,
            n_kv_heads=layout.n_kv_heads,
            head_dim=layout.head_dim,
            dtype=layout.dtype,
        )
        store = SyntheticPrefixBlockStore(
            block_size=block_size, codec=codec
        )
        return RadixPrefixCache(block_size=block_size, store=store)

    def _synth_block(block_idx: int) -> list[tuple[Any, Any]]:
        # Zero-filled K/V. The admission-headroom signal is about
        # budgeter arithmetic over store residency; quantization
        # quality is irrelevant here.
        per_layer: list[tuple[Any, Any]] = []
        for _ in range(layout.num_layers):
            k = mx.zeros(
                (1, layout.n_kv_heads, block_size, layout.head_dim),
                dtype=layout.dtype,
            )
            v = mx.zeros(
                (1, layout.n_kv_heads, block_size, layout.head_dim),
                dtype=layout.dtype,
            )
            per_layer.append((k, v))
        return per_layer

    # Step 1 — warm fp16 cache to the target, record recipe.
    fp16_pc = _build_prefix_cache(fp16_codec_id)
    recipe: list[tuple[tuple[int, ...], list[tuple[Any, Any]]]] = []
    fp16_store = fp16_pc.store
    while fp16_store.resident_bytes() < warmup_target:
        block_idx = len(recipe)
        if block_idx >= _ADMISSION_HEADROOM_WARMUP_BLOCK_CAP:
            raise RuntimeError(
                f"admission_headroom warmup exceeded "
                f"{_ADMISSION_HEADROOM_WARMUP_BLOCK_CAP} blocks without "
                f"reaching target={warmup_target} bytes "
                f"(current={fp16_store.resident_bytes()}). Codec "
                f"{fp16_codec_id!r} per-block residency may be too "
                f"small for the configured cap_bytes × warmup_ratio; "
                f"increase cap_bytes or reduce warmup_ratio."
            )
        # Unique token sequence per block so insert_detached actually
        # creates a new radix node (identical tokens would
        # deduplicate).
        tokens = tuple(
            block_idx * block_size + i for i in range(block_size)
        )
        per_layer = _synth_block(block_idx)
        fp16_pc.insert_detached(list(tokens), [per_layer])
        recipe.append((tokens, per_layer))

    warmup_blocks = len(recipe)
    if warmup_blocks < 1:
        raise RuntimeError(
            "admission_headroom warmup produced zero blocks; "
            "warmup_target already satisfied by empty store"
        )

    # Step 2 — replay identical recipe into the compressed prefix cache.
    block_pc = _build_prefix_cache(compressed_codec_id)
    for tokens, per_layer in recipe:
        block_pc.insert_detached(list(tokens), [per_layer])

    # Step 3 — admission loops on each.
    def _count_admissions(prefix_cache: Any) -> int:
        budgeter = MemoryBudgeter.for_adapter(
            adapter,
            prefix_cache=prefix_cache,
            weights_bytes=weights_bytes,
            cap_bytes=cap_bytes,
            account_prefix_residency=True,
        )
        count = 0
        while count < _ADMISSION_HEADROOM_ADMIT_CAP:
            req_id = f"admission-headroom-trial-{count}"
            decision = budgeter.admit(req_id, n_prompt, max_tokens)
            if not isinstance(decision, AdmitDecision):
                return count
            budgeter.apply_admit(req_id, decision.reserved_delta)
            count += 1
        raise RuntimeError(
            f"admission_headroom admit loop exceeded "
            f"{_ADMISSION_HEADROOM_ADMIT_CAP} consecutive admits "
            f"without a non-AdmitDecision; cap_bytes may be too large "
            f"relative to per-admit worst_case_bytes"
        )

    n_fp16 = _count_admissions(fp16_pc)
    n_block = _count_admissions(block_pc)

    resident_fp16 = int(fp16_pc.store.resident_bytes())
    resident_block = int(block_pc.store.resident_bytes())

    collected: dict[str, Any] = {
        "n_fp16": n_fp16,
        "n_block": n_block,
        "resident_bytes_fp16": resident_fp16,
        "resident_bytes_block": resident_block,
        "warmup_blocks": warmup_blocks,
    }
    context: dict[str, Any] = {
        "cap_bytes": cap_bytes,
        "weights_bytes": weights_bytes,
        "warmup_ratio": warmup_ratio,
        "n_prompt": n_prompt,
        "max_tokens": max_tokens,
        "fp16_codec": fp16_codec_id,
        "compressed_codec": compressed_codec_id,
    }
    return collected, context


# Safety rails for the admission-headroom loops. Chosen generously;
# real rows stay well under both.
_ADMISSION_HEADROOM_WARMUP_BLOCK_CAP: int = 4096
_ADMISSION_HEADROOM_ADMIT_CAP: int = 4096


def _run_storage(
    scenario: Scenario, engine: Engine, adapter: ModelAdapter
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drive the P-5-C.3 ``OracleKind.STORAGE`` workload.

    Workload shape mirrors the prefix-hit decode rows verbatim (two
    identical prompts, ``max_batch_size=1``, ``prefix_cache=True``)
    so the scheduler's ``_extract_and_insert_prefix`` on row-0
    termination and the ``_admit_single_hit_row`` on row-1
    admission both fire against the configured codec. After the
    ``generate_batch`` event stream drains, the store's
    ``resident_bytes()`` + ``live_block_ids()`` +
    ``resident_bytes_per_block()`` + the radix cache's own ``hits``
    counter form the row's observable.

    Workload-shape validation happens pre-load in
    :func:`_validate_workload_for_oracle`, mirroring the
    ``DECODE_TOK_S_WITH_PREFIX_HIT`` pattern — authoring errors
    fail before the engine factory runs.

    Event-stream validation mirrors
    :func:`_collect_prefix_hit_decode`: an ``aborted`` event or
    unexpected ``req_index`` or a row that never emits ``done``
    raises ``RuntimeError``, which the outer ``_run_one`` collapses
    to a failed ``ScenarioResult``. Skipping this check would let
    the store snapshot succeed against a half-completed workload
    and the STORAGE oracle would surface a residency number from a
    failed run as if it were valid.

    Raises ``RuntimeError`` if the workload completes with zero
    detached blocks in the store. An empty store after this
    workload shape means either the scheduler regressed on
    ``_extract_and_insert_prefix`` or the codec failed to register
    blocks — either way the ``resident_bytes`` reading would be
    trivially zero and the row's signal content is zero.
    """
    wl = scenario.workload
    params = _build_sampling_params(wl, adapter, include_eos=False)
    prefix_cache = _maybe_build_prefix_cache(wl, adapter)
    assert prefix_cache is not None  # wl.prefix_cache=True guarantees this

    # Drain the batched event stream with the same strictness the
    # decode-speed path uses — aborted rows, unknown req_index, and
    # rows that never complete would otherwise leave the workload
    # "half-run" yet the store snapshot might still look plausible.
    prompts = list(wl.prompts)
    expected_rows = set(range(len(prompts)))
    done_rows: set[int] = set()
    for event in engine.generate_batch(
        prompts,
        params,
        max_batch_size=wl.max_batch_size,
        prefix_cache=prefix_cache,
    ):
        if event.req_index not in expected_rows:
            raise RuntimeError(
                f"storage_unexpected_req_index:{event.req_index}"
            )
        if event.kind == "aborted":
            raise RuntimeError(
                f"storage_aborted:row={event.req_index}_"
                f"reason={event.finish_reason}"
            )
        if event.kind == "done":
            done_rows.add(event.req_index)

    missing = expected_rows - done_rows
    if missing:
        raise RuntimeError(
            f"storage_rows_never_completed:{sorted(missing)}"
        )

    store = prefix_cache.store
    # ``resident_bytes`` / ``live_block_ids`` / ``resident_bytes_per_block``
    # are not on the ``PrefixBlockStore`` Protocol — consumers use a
    # capability check. All four qwen3-0.6b-compression-* rows pin a
    # codec, so the store is always a ``SyntheticPrefixBlockStore``
    # with these methods; assert the contract loudly rather than
    # silently handle a missing attribute.
    if not hasattr(store, "live_block_ids"):
        raise RuntimeError(
            f"storage oracle: prefix cache store "
            f"{type(store).__name__} lacks live_block_ids(); "
            f"expected SyntheticPrefixBlockStore"
        )
    live_block_ids = store.live_block_ids()
    live_blocks = len(live_block_ids)
    if live_blocks == 0:
        raise RuntimeError(
            f"storage oracle: workload completed with zero "
            f"detached blocks in the prefix cache's store. Either "
            f"_extract_and_insert_prefix did not fire on row 0 "
            f"termination (scheduler regression) or the codec "
            f"({wl.kv_codec!r}) failed to register blocks. Empty "
            f"store means the reported resident_bytes would be "
            f"trivially zero — fail loud rather than emit noise."
        )

    resident_bytes = int(store.resident_bytes())
    bpb_raw = store.resident_bytes_per_block()
    resident_bytes_per_block: int | None = (
        int(bpb_raw) if bpb_raw is not None else None
    )
    prefix_cache_hits = int(prefix_cache.hits)

    collected: dict[str, Any] = {
        "resident_bytes": resident_bytes,
        "resident_bytes_per_block": resident_bytes_per_block,
        "live_blocks": live_blocks,
        "prefix_cache_hits": prefix_cache_hits,
    }
    context: dict[str, Any] = {
        "kv_codec": wl.kv_codec,
        "block_size": prefix_cache.block_size,
    }
    return collected, context


def _run_ppl(
    scenario: Scenario, adapter: ModelAdapter
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Drive the P-5-C.2 ``OracleKind.PPL`` workload.

    Does **not** go through ``engine.generate_batch`` — the PPL
    oracle is teacher-forced and needs positional logits, not
    sampled tokens. Reads the pre-extracted WikiText text file from
    ``scenario.oracle_config['wikitext_path']``, tokenizes with the
    adapter's bound tokenizer, then dispatches:

    - ``workload.kv_codec is None`` → fp16 baseline path
      (:func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll`).
      No prefix cache is built; the oracle uses the adapter's own
      ``BatchKVCache``.
    - otherwise → codec-backed path
      (:func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll_with_codec`).
      Builds a fresh :class:`RadixPrefixCache` with the named codec
      via :func:`_maybe_build_prefix_cache` (kwarg-compatible; no
      engine dependency for the oracle itself).

    ``oracle_config`` keys:

    - ``wikitext_path``: filesystem path to a UTF-8 text file.
      Required.
    - ``chunk_size``: default 256 (vqbench REPORT headline).
    - ``max_tokens``: cap on tokenized length (default 512).
    - ``min_scored_tokens``: oracle-side floor on ``n_tokens``;
      read by :func:`ppl_oracle`, not here.

    Returns:
        ``(collected, context)``. ``collected`` is
        ``{"nll_sum": float, "n_tokens": int, "ppl": float}``;
        ``context`` carries ``chunk_size`` / ``max_tokens`` /
        ``wikitext_path`` / ``kv_codec`` for the oracle to forward
        into metadata.
    """
    from silica.bench.ppl_oracle import (
        perplexity_from_nll,
        teacher_forced_chunked_nll,
        teacher_forced_chunked_nll_with_codec,
    )
    from silica.bench.wikitext import (
        load_wikitext_text,
        tokenize_for_ppl,
    )

    cfg = scenario.oracle_config
    if "wikitext_path" not in cfg:
        raise RuntimeError(
            f"scenario {scenario.id!r}: OracleKind.PPL requires "
            f"oracle_config['wikitext_path']"
        )
    wikitext_path = Path(cfg["wikitext_path"])
    chunk_size = int(cfg.get("chunk_size", 256))
    max_tokens = int(cfg.get("max_tokens", 512))

    if chunk_size < 1:
        raise RuntimeError(
            f"scenario {scenario.id!r}: oracle_config['chunk_size'] "
            f"must be >= 1; got {chunk_size}"
        )

    text = load_wikitext_text(wikitext_path)
    # ``ModelAdapter.tokenizer`` is a method, not a property (see
    # silica/models/adapter.py); bound-method-as-tokenizer would
    # trip ``tokenize_for_ppl``'s encode-attribute check. Call it.
    tokenizer = adapter.tokenizer()
    # Floor the tokenized length at one full chunk so the oracle
    # scores at least ``chunk_size - 1`` within-chunk tokens. A
    # single-chunk scenario with a short tokenized length would
    # silently produce a degenerate NLL; loud failure at tokenize
    # time is more useful than a near-zero PPL downstream.
    tokens = tokenize_for_ppl(
        tokenizer,
        text,
        max_tokens=max_tokens,
        min_tokens=chunk_size,
    )

    prefix_cache = _maybe_build_prefix_cache(scenario.workload, adapter)

    if prefix_cache is None:
        # fp16 baseline path — kv_codec is None AND prefix_cache=False.
        nll_sum, n_tokens = teacher_forced_chunked_nll(
            adapter, tokens, chunk_size=chunk_size
        )
    else:
        # Codec-backed path — build_seeded_batch_kv seeds per-chunk
        # caches from prefix_cache; encode / decode fire on every
        # chunk boundary.
        nll_sum, n_tokens = teacher_forced_chunked_nll_with_codec(
            adapter,
            prefix_cache,
            tokens,
            chunk_size=chunk_size,
        )

    ppl = perplexity_from_nll(nll_sum, n_tokens)

    collected: dict[str, float | int] = {
        "nll_sum": float(nll_sum),
        "n_tokens": int(n_tokens),
        "ppl": float(ppl),
    }
    context: dict[str, Any] = {
        "chunk_size": chunk_size,
        "max_tokens": max_tokens,
        "wikitext_path": str(wikitext_path),
        "kv_codec": scenario.workload.kv_codec,
    }
    return collected, context


def _maybe_build_prefix_cache(
    workload: Workload, adapter: ModelAdapter | None = None
) -> Any | None:
    """Build a fresh :class:`RadixPrefixCache` if the workload asks
    for it, else ``None``.

    When ``workload.kv_codec`` is set (P-5-A.3a), constructs the
    named codec via ``silica.bench.codec_registry`` and installs it
    on the synthetic store as the shorthand (both K and V sides use
    the same codec). Codec factories take ``(block_size, n_kv_heads,
    head_dim, dtype)`` — the last three come from
    ``adapter.kv_layout()``, which is why this helper now takes an
    ``adapter`` kwarg. Callers that pass a workload with
    ``kv_codec=None`` (all pre-P-5-A.3 callers) can omit the adapter.

    Block size is hard-coded to 16 — the same constant the README's
    Quickstart example uses, and the default value every pytest-
    side prefix-cache test has relied on since P-2. Exposing
    block_size as a Workload field is avoided until a scenario
    actually needs a different value.

    Imports lazily so ``silica.bench.runner`` stays cheap when the
    caller only runs B=1 SMOKE / B1 parity scenarios.
    """
    if not workload.prefix_cache:
        return None
    from silica.kvcache.prefix import RadixPrefixCache
    from silica.kvcache.store import SyntheticPrefixBlockStore

    block_size = 16
    codec: Any = None
    if workload.kv_codec is not None:
        if adapter is None:
            raise ValueError(
                f"workload {workload.name!r} has kv_codec="
                f"{workload.kv_codec!r} but no adapter was supplied "
                f"to _maybe_build_prefix_cache; codec factories need "
                f"adapter.kv_layout() for n_kv_heads / head_dim / "
                f"dtype"
            )
        from silica.bench.codec_registry import get_codec_spec

        spec = get_codec_spec(workload.kv_codec)
        # A.3b guard: the shorthand installs one codec on both sides
        # (K and V). P-5-A.1 entries are all symmetric (k_supported
        # and v_supported both True); this check becomes load-bearing
        # when P-5-B lands K-only / V-only variants so a future
        # ``kv_codec="rabitq_k_only"`` scenario fails fast rather
        # than silently installing an asymmetric codec on the V side
        # where it is not meant to run.
        if not (spec.k_supported and spec.v_supported):
            raise ValueError(
                f"workload {workload.name!r}: kv_codec="
                f"{workload.kv_codec!r} is not symmetric "
                f"(k_supported={spec.k_supported}, "
                f"v_supported={spec.v_supported}); "
                f"shorthand installs one codec on both sides. "
                f"Asymmetric K/V codecs need a split-id field "
                f"(P-5-C scope) rather than the kv_codec shorthand."
            )
        layout = adapter.kv_layout()
        codec = spec.factory(
            block_size=block_size,
            n_kv_heads=layout.n_kv_heads,
            head_dim=layout.head_dim,
            dtype=layout.dtype,
        )

    store = SyntheticPrefixBlockStore(
        block_size=block_size, codec=codec
    )
    return RadixPrefixCache(block_size=block_size, store=store)


def _collect_b1_batched_tokens(
    engine: Engine, prompt: str, params: SamplingParams
) -> list[int]:
    """Collect row-0 token events from a B=1 ``generate_batch`` stream.

    Strict validation: any event with ``req_index != 0`` is
    treated as a scheduler fault (generated batches in a B=1 run
    must only emit against the single admitted request), an
    ``aborted`` event short-circuits the run, and the stream
    MUST emit a ``done`` event for row 0 before ending — any of
    these raises ``RuntimeError`` so the outer ``_run_one``
    collapses them to ``status="failed"`` with a structured
    ``b1_batched_*`` reason, never into the oracle's "tokens
    mismatched" branch. Matching the BGT1 / SMOKE-batched
    event-validation pattern is deliberate: if a B=1 stream
    closes without ``done`` but the emitted tokens happen to
    equal the single-request reference, we would otherwise
    falsely pass parity on a silently broken scheduler.
    """
    tokens: list[int] = []
    done_seen = False
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
        elif event.kind == "done":
            done_seen = True
    if not done_seen:
        raise RuntimeError("b1_batched_never_completed")
    return tokens


__all__ = [
    "BenchRunner",
    "DirectBatchedReferenceFn",
    "EngineFactory",
    "TeacherForcedReferenceFn",
    "TeacherForcedSilicaFn",
]
