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

import math
import time
from collections.abc import Iterator, Sequence

import mlx.core as mx

from silica.core.events import BatchEvent
from silica.core.profiler import MetricsRegistry
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.kvcache.manager import KVHandle, KVManager
from silica.kvcache.prefix import RadixPrefixCache
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

    @property
    def kv_manager(self) -> KVManager:
        """Public accessor for the KV manager.

        Exposed so the bench harness's teacher-forced-argmax path
        (P-4.3) can drive ``adapter.prefill`` / ``decode_step``
        directly without owning the KV lifecycle — otherwise it
        would need a parallel ``SimpleKVCache.from_model`` load.
        Internal callers still use ``self._kv_manager``.
        """
        return self._kv_manager

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

    # --- P-2 Units 16a / 16b / 16c.1: batched generation ---

    def generate_batch(
        self,
        prompts: Sequence[str],
        params: SamplingParams | list[SamplingParams] | None = None,
        *,
        max_batch_size: int | None = None,
        prefix_cache: RadixPrefixCache | None = None,
        length_spread_threshold: float = 2.0,
    ) -> Iterator[BatchEvent]:
        """Yield ``BatchEvent`` values driving ``ContinuousBatcher``.

        As of Unit 16c.1, the batcher supports **queue-bounded
        admission**: if ``len(prompts)`` exceeds ``max_batch_size``,
        the first ``max_batch_size`` prompts admit as the initial
        cohort, the rest sit in the waiting queue, and the admit
        phase drains the backlog as slots free via reclaim. Mid-run
        admission uses ``BatchKVCache.extend``.

        As of P-4.5-B.1 (Q-010 fix), ``generate_batch`` also reorders
        admissions by prompt length when the batch is heterogeneous
        enough to stall short-row TTFT behind long-row prefill. See
        ``length_spread_threshold`` below and
        ``docs/P4_5_CHUNKED_PREFILL_OPENING.md``.

        Args:
            prompts: one or more prompts. Empty strings are skipped
                silently (their ``req_index`` stays mapped to the
                original position in the list).
            params: a single ``SamplingParams`` (homogeneous, the P-2
                supported case) or a list of length ``len(prompts)``.
                For P-2 all elements of the list must be equal;
                heterogeneous lists raise ``NotImplementedError``.
            max_batch_size: optional cap on **active physical rows**
                (not queue length). Defaults to the number of
                non-empty prompts so all admit at step 0 — preserving
                the fixed-cohort 16b behaviour for small batches.
                Callers testing queue-bounded admission explicitly
                pass e.g. ``max_batch_size=4`` with 8 prompts.
            prefix_cache: optional ``RadixPrefixCache`` for
                16c.2 shared-prefix reuse. Ownership lives with the
                caller so the cache can persist across multiple
                ``generate_batch`` invocations. When ``None`` (default),
                behaviour is bit-identical to 16c.1 (invariant S-6 of
                the step-4 skeleton).
            length_spread_threshold: P-4.5-B.1 Q-010 fairness fix.
                When ``max(prompt_lens) / min(prompt_lens) >``
                threshold, admissions sort by length ASC and only
                the leading short-prompt cluster admits pre-step;
                the remainder queues and drains through the existing
                mid-run admission path. Default ``2.0`` fixes the
                measured TTFT defect for heterogeneous batches.
                Pass ``float('inf')`` to disable the split (needed by
                strict-parity tests that compare Silica against a
                direct mlx-lm ``B=N`` reference run over the same
                unsplit cohort — the P-2 ``test_left_padding_does_not_corrupt_any_row``
                case and the bench harness BGT1 parity path).
                ``req_index`` remains the original user-supplied
                index on every event, independent of the admission
                reorder.

        Empty ``prompts`` iterable yields nothing.
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

        effective_batch_size = (
            max_batch_size if max_batch_size is not None else len(admissions)
        )
        if effective_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1, got {max_batch_size}"
            )

        # P-4.5-B.1 Q-010: reorder admissions so heterogeneous batches
        # do not stall short-row TTFT. ``_sort_admissions_by_length`` is
        # a stable sort; ``_initial_cohort_cap`` is a pure function
        # whose spec (and reverse-example pins) lives in the opening
        # doc §6.1. The ``length_spread_threshold > 1.0`` precondition
        # raises inside ``_initial_cohort_cap``; we catch the violation
        # before any batcher construction.
        #
        # Reorder policy: the sort itself changes the ``BatchEvent``
        # emission order (event ``req_index`` stays stable on the
        # tuple, but the row index in ``_rows`` differs, which in
        # turn changes the order of per-step emit). A caller who
        # passes ``length_spread_threshold=float('inf')`` — the
        # documented opt-out for strict-parity tests — expects
        # pre-P-4.5 behaviour, not just "no split but different event
        # order". Same for homogeneous batches where
        # ``max_len / min_len <= threshold`` does not warrant a
        # split. We therefore only switch to the sorted ordering
        # when the split path actually fires. The cap helper is
        # still called in both branches so the threshold / NaN
        # preconditions surface at the same entry point.
        admissions_sorted = _sort_admissions_by_length(admissions)
        cap = _initial_cohort_cap(
            admissions_sorted, effective_batch_size, length_spread_threshold
        )
        lens = [len(ids) for _, ids in admissions]
        needs_reorder = (
            len(admissions) > 1
            and min(lens) > 0
            and max(lens) / min(lens) > length_spread_threshold
        )
        admissions_ordered = admissions_sorted if needs_reorder else list(admissions)
        pre_step = admissions_ordered[:cap]
        remainder = admissions_ordered[cap:]

        batcher = ContinuousBatcher(
            self._adapter,
            sampler=self._sampler,
            max_batch_size=effective_batch_size,
            prefix_cache=prefix_cache,
        )
        # Pre-step admits seal the initial cohort. ``req_index`` on each
        # tuple is the ORIGINAL user-supplied index (unchanged by the
        # sort — the sort reorders the admission-order only). The
        # batcher stores ``req_index`` on each ``_BatchRow`` and emits
        # it unchanged in every ``BatchEvent``, so downstream callers
        # still see events tagged with the original index.
        for req_index, prompt_ids in pre_step:
            batcher.add_request(req_index, prompt_ids, effective)
        # Remaining prompts: prepare cohort via a bootstrap step so
        # subsequent add_request calls route to the waiting queue.
        if remainder:
            # The first step() call seals the pre-step cohort; we peel
            # one step's events out for the caller and then queue the
            # backlog, which the ongoing drain loop will admit as
            # slots free.
            for event in batcher.step():
                yield event
            for req_index, prompt_ids in remainder:
                batcher.add_request(req_index, prompt_ids, effective)

        # Uses has_work (not has_active) so cohort-drain completes even
        # when the last sample phase terminates every row — the step()
        # after the last active row handles deferred reclaim, which
        # empties ``self._rows`` and flips has_work to False. See
        # ``docs/P2_UNIT_16C_PREP.md`` §1 I-5.
        while batcher.has_work():
            for event in batcher.step():
                yield event


def _sort_admissions_by_length(
    admissions: Sequence[tuple[int, list[int]]],
) -> list[tuple[int, list[int]]]:
    """Sort admissions by prompt-token length ASC, stably.

    Python's ``list.sort`` is stable, so ties preserve the user's
    original order. ``req_index`` stays on its tuple — the original
    user-facing index the event stream must emit — and only the
    *admission order* (which prompts go pre-step vs queue) changes.
    P-4.5-B.1 / `docs/P4_5_CHUNKED_PREFILL_OPENING.md` §6.1.
    """
    return sorted(admissions, key=lambda a: len(a[1]))


def _initial_cohort_cap(
    admissions_sorted: Sequence[tuple[int, list[int]]],
    effective_batch_size: int,
    spread_ratio_threshold: float,
) -> int:
    """How many leading (shortest) admissions go into the initial cohort.

    P-4.5-B.1 / `docs/P4_5_CHUNKED_PREFILL_OPENING.md` §6.1 spec.

    Preconditions enforced at call time:
      - ``admissions_sorted`` is length-ASC (caller's responsibility;
        we don't re-sort defensively since this is hot-path
        adjacent and the upstream is a known ``_sort_admissions_by_length``).
      - ``effective_batch_size >= 1``.
      - ``spread_ratio_threshold > 1.0`` — a threshold of exactly 1.0
        would require strict equality of lengths to avoid splitting,
        i.e. an always-split policy on any variation; rejected
        explicitly so callers cannot stumble into it. ``float('inf')``
        is a legal opt-out sentinel (never splits).

    Homogeneous fast path: if there is at most one admission or the
    length-spread ratio ``max_len / min_len`` is ``<=`` threshold,
    return ``min(effective_batch_size, len(admissions_sorted))`` —
    current pre-P-4.5 behaviour.

    Split path: find the smallest index ``k`` such that
    ``admissions_sorted[k].len > threshold * admissions_sorted[0].len``
    and return ``cap = max(1, min(effective_batch_size, k))``. The
    ``min(effective_batch_size, ...)`` ceiling is the hard invariant
    that the runtime respects ``max_batch_size``; the ``max(1, ...)``
    floor handles the pathological zero-index case defensively.
    """
    if effective_batch_size < 1:
        raise ValueError(
            f"effective_batch_size must be >= 1, got {effective_batch_size}"
        )
    # NaN comparisons always return False, so ``NaN <= 1.0`` slips
    # past the threshold precondition below and later ``len(ids) >
    # NaN * min_len`` is also False, which silently disables the
    # split — the exact shape of ``float('inf')`` but without the
    # caller's consent. Reject NaN explicitly so a caller who
    # accidentally passes one gets a clear error rather than a
    # silently-degraded fairness guarantee.
    if math.isnan(spread_ratio_threshold):
        raise ValueError(
            "length_spread_threshold must not be NaN (use float('inf') "
            "to disable the split)"
        )
    if spread_ratio_threshold <= 1.0:
        raise ValueError(
            f"length_spread_threshold must be > 1.0 (use float('inf') "
            f"to disable the split), got {spread_ratio_threshold}"
        )
    n = len(admissions_sorted)
    if n <= 1:
        return min(effective_batch_size, n)
    min_len = len(admissions_sorted[0][1])
    max_len = len(admissions_sorted[-1][1])
    # min_len is >= 1 because upstream filters empty prompts, but
    # guard against future callers that skip that filter.
    if min_len <= 0 or max_len / min_len <= spread_ratio_threshold:
        return min(effective_batch_size, n)
    # Split: find the first index whose prompt exceeds the threshold.
    threshold_abs = spread_ratio_threshold * min_len
    first_exceeding = next(
        (k for k, (_, ids) in enumerate(admissions_sorted) if len(ids) > threshold_abs),
        n,
    )
    return max(1, min(effective_batch_size, first_exceeding))


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
