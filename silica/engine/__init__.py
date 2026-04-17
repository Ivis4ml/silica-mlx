"""silica.engine — top-level generation orchestrator (P-1).

Engine glues the pieces together:
  - ``ModelAdapter`` (I-1) — tokenizer + prefill + decode_step.
  - ``KVManager`` (I-2) — reserve / free per-request KV (``SimpleKVCache`` in
    P-1, ``PagedKVCache`` in P-2; Engine is agnostic).
  - ``Sampler`` (P-0) — processor chain → sampled token from logits.
  - ``MetricsRegistry`` (P-0) — per-instance (not global) timing + throughput
    + memory gauges. Populated during ``generate`` and readable via
    ``engine.metrics.snapshot()`` once the generator is exhausted.

P-1 scope: one request at a time. Multi-request continuous batching is P-2
(``ContinuousBatcher`` + ``MemoryBudgeter``). ``Engine.generate`` returns an
iterator of token ids so callers (CLI, tests, bench harness) can consume
streaming output without any decode coupling.

Stop policy for P-1:
  - ``max_tokens`` is a hard upper bound on yielded tokens.
  - ``stop_token_ids`` tokens are yielded-then-stopped (vLLM / mlx-lm
    convention — the caller needs to see the stop token to know *why* we
    stopped).
  - String-sequence ``stop`` patterns are a P-2 concern (they require
    incremental decoding of generated tokens into a text buffer).
  - EOS handling: the caller is responsible for populating ``stop_token_ids``
    with the tokenizer's EOS id (or omitting it if ``ignore_eos``).

Metrics populated per ``generate`` call:
  - ``ttft_ms``: wall-clock ms from prefill start to first yielded token
    (prefill forward + first sample). Greedy's sample step is essentially
    free but the schema reserves room for heavier processor chains.
  - ``prefill_tok_s``: ``len(prompt_ids) / ttft_s`` — prompt throughput.
  - ``decode_tok_s``: ``n_decode / decode_s`` where ``decode_s`` is only the
    time the generator spent inside its own loop (``perf_counter`` in a
    generator only ticks while ``__next__`` is active, so caller-side
    latency between yields is correctly excluded).
  - ``resident_mb``: ``kv_manager.budget().resident_bytes / 1e6`` at the
    end of generation (peak for the request on ``SimpleKVCache``).
  - ``logical_kv_bytes``: ``kv_manager.budget().logical_bytes`` at the end.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

import mlx.core as mx

from silica.core.events import BatchEvent
from silica.core.profiler import MetricsRegistry
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.kvcache.manager import KVHandle, KVManager
from silica.models.adapter import ModelAdapter
from silica.scheduler.batcher import ContinuousBatcher


class Engine:
    """Single-request generation orchestrator (P-1).

    Construct once per (adapter, kv_manager) pair; call ``generate`` any
    number of times. ``req_id`` is auto-assigned so callers don't manage it.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        kv_manager: KVManager,
        sampler: Sampler | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._adapter = adapter
        self._kv_manager = kv_manager
        self._sampler = sampler or Sampler()
        self.metrics = metrics or MetricsRegistry()
        self._req_counter = 0

    def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> Iterator[int]:
        """Yield generated token ids one at a time.

        Empty prompts yield nothing (mlx-lm's generate_step also requires a
        non-empty prompt / input_embeddings; we mirror that here).
        """
        effective = params or SamplingParams()
        tokenizer = self._adapter.tokenizer()
        prompt_ids: list[int] = list(tokenizer.encode(prompt))
        if not prompt_ids:
            return

        req_id = self._new_req_id()
        handle = KVHandle(req_id=req_id)
        self._kv_manager.reserve_for_prefill(req_id, prompt_ids)
        try:
            yield from self._drive(prompt_ids, handle, effective)
        finally:
            self._kv_manager.free(req_id)

    # --- private ---

    def _drive(
        self,
        prompt_ids: list[int],
        handle: KVHandle,
        params: SamplingParams,
    ) -> Iterator[int]:
        prompt_arr = mx.array(prompt_ids, dtype=mx.int32)

        # Prefill + first sample — measured as a single TTFT block.
        t0 = time.perf_counter()
        logits, _ = self._adapter.prefill(prompt_arr, handle)
        history: list[int] = list(prompt_ids)
        token_scalar = self._sampler.sample(
            logits, mx.array(history, dtype=mx.int32), params
        )
        tok_int = int(token_scalar.item())
        t_first = time.perf_counter()
        ttft_s = t_first - t0
        self.metrics.set_metric("ttft_ms", ttft_s * 1000.0)
        if ttft_s > 0:
            self.metrics.set_metric("prefill_tok_s", len(prompt_ids) / ttft_s)

        yield tok_int
        history.append(tok_int)
        if tok_int in params.stop_token_ids:
            self._record_tail_metrics(decode_count=0, decode_start=t_first)
            return

        decode_count = 0
        n = 1
        while n < params.max_tokens:
            step_in = mx.array([tok_int], dtype=mx.int32)
            logits, _ = self._adapter.decode_step(step_in, handle)
            token_scalar = self._sampler.sample(
                logits, mx.array(history, dtype=mx.int32), params
            )
            tok_int = int(token_scalar.item())
            yield tok_int
            n += 1
            decode_count += 1
            history.append(tok_int)
            if tok_int in params.stop_token_ids:
                break

        self._record_tail_metrics(
            decode_count=decode_count, decode_start=t_first
        )

    def _record_tail_metrics(
        self, *, decode_count: int, decode_start: float
    ) -> None:
        if decode_count > 0:
            decode_elapsed = time.perf_counter() - decode_start
            if decode_elapsed > 0:
                self.metrics.set_metric(
                    "decode_tok_s", decode_count / decode_elapsed
                )
        budget = self._kv_manager.budget()
        self.metrics.set_metric("resident_mb", budget.resident_bytes / 1e6)
        self.metrics.set_metric("logical_kv_bytes", budget.logical_bytes)

    def _new_req_id(self) -> str:
        rid = f"req-{self._req_counter}"
        self._req_counter += 1
        return rid

    # --- P-2 Units 16a / 16b: batched generation ---

    def generate_batch(
        self,
        prompts: Sequence[str],
        params: SamplingParams | list[SamplingParams] | None = None,
    ) -> Iterator[BatchEvent]:
        """Yield ``BatchEvent`` values driving ``ContinuousBatcher``.

        P-2 Unit 16b: all non-empty prompts admit at step 0 as a fixed
        cohort; mid-run admission (16c) and preemption (16d) are not
        yet available. Cohort size is whatever ``prompts`` contains
        after empty-prompt filtering.

        ``params`` accepts a single ``SamplingParams`` (homogeneous — the
        P-2 supported case) or a list of length ``len(prompts)``. For
        P-2 all elements of the list must be equal; heterogeneous lists
        raise ``NotImplementedError`` — the union signature is reserved
        so P-3 can land per-row logit stacks without breaking the API.

        Empty prompts are skipped silently (their indices are preserved
        — "`['', 'hi']`" yields events for req_index=1 only). Empty
        ``prompts`` iterable yields nothing.
        """
        prompts_list = list(prompts)
        effective = _resolve_batch_params(prompts_list, params)
        tokenizer = self._adapter.tokenizer()

        # Pre-tokenize and drop empties while preserving original req_index.
        admissions: list[tuple[int, list[int]]] = []
        for req_index, prompt in enumerate(prompts_list):
            prompt_ids = list(tokenizer.encode(prompt))
            if prompt_ids:
                admissions.append((req_index, prompt_ids))
        if not admissions:
            return

        batcher = ContinuousBatcher(
            self._adapter,
            sampler=self._sampler,
            max_batch_size=len(admissions),
        )
        for req_index, prompt_ids in admissions:
            batcher.add_request(req_index, prompt_ids, effective)

        # Uses has_active for now (16b semantics). Unit 16c.1 step 2
        # will flip this to has_work() once reclaim is implemented —
        # see ``docs/P2_UNIT_16C_PREP.md`` §1 I-5. Switching prematurely
        # would infinite-loop because step() currently returns [] when
        # has_active is False but has_work stays True from terminal-
        # pending-reclaim rows.
        while batcher.has_active():
            for event in batcher.step():
                yield event


def _resolve_batch_params(
    prompts: Sequence[str],
    params: SamplingParams | list[SamplingParams] | None,
) -> SamplingParams:
    """Validate and reduce generate_batch's union-typed params argument.

    Returns a single ``SamplingParams`` for P-2's homogeneous batch.
    Heterogeneous lists raise ``NotImplementedError`` naming the phase
    that will add per-row support (P-3).
    """
    if params is None:
        return SamplingParams()
    if isinstance(params, SamplingParams):
        return params
    if isinstance(params, list):
        if len(params) != len(prompts):
            raise ValueError(
                f"params list length ({len(params)}) must equal "
                f"prompts length ({len(prompts)})"
            )
        if not params:
            return SamplingParams()
        first = params[0]
        if not all(p == first for p in params):
            raise NotImplementedError(
                "Heterogeneous SamplingParams per row arrives in P-3; "
                "P-2 requires all rows to share the same params."
            )
        return first
    raise TypeError(
        f"params must be SamplingParams | list[SamplingParams] | None, "
        f"got {type(params).__name__}"
    )


__all__ = ["Engine"]
