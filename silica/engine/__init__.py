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
from typing import TYPE_CHECKING

import mlx.core as mx

from silica.core.events import BatchEvent
from silica.core.profiler import MetricsRegistry
from silica.core.request import Request, RequestState
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.kvcache.manager import KVHandle, KVManager
from silica.kvcache.prefix import RadixPrefixCache
from silica.models.adapter import ModelAdapter
from silica.models.hidden_capture import HiddenCaptureAdapter
from silica.models.recurrent import SpecRecurrentRollbackAdapter
from silica.scheduler.batcher import ContinuousBatcher
from silica.speculative.engine import (
    DraftEngine,
    NoopDraftEngine,
    TargetHiddenConsumer,
)
from silica.speculative.verify import greedy_verify, run_verify_forward

if TYPE_CHECKING:
    # Lazy import: ``silica.bench`` pulls in ``silica.bench.runner``
    # which imports ``silica.engine.Engine`` — eager import of the
    # collector here would cycle. The collector is duck-typed
    # at runtime; only the type checker resolves the symbol.
    from silica.bench.spec_collector import SpecMetricCollector


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
        *,
        draft_engine: DraftEngine | None = None,
        verify_k: int = 4,
        spec_collector: SpecMetricCollector | None = None,
    ) -> None:
        # D-021 step 5 sub-unit (b): single-request spec wiring.
        # ``draft_engine`` defaults to ``NoopDraftEngine`` so spec-off is
        # the path of zero overhead and byte-identical to the pre-spec
        # decode loop. ``verify_k`` is the target verify forward input
        # length (γ = verify_k - 1 actual draft proposals per cycle;
        # see plans/P6_SPEC_FOUNDATION_OPENING.md §4.5). v0.1 default 4
        # matches the Unit-7 microbench's regime-transition sweet spot;
        # values < 1 are rejected loud so a misconfiguration cannot
        # silently disable spec.
        if verify_k < 1:
            raise ValueError(f"verify_k must be >= 1, got {verify_k}")
        self._adapter = adapter
        self._kv_manager = kv_manager
        self._sampler = sampler or Sampler()
        self.metrics = metrics or MetricsRegistry()
        self._req_counter = 0
        self._draft_engine: DraftEngine = draft_engine or NoopDraftEngine()
        self._verify_k = verify_k
        # D-021 step 5 sub-unit (g): optional spec-metrics collector. The
        # engine emits propose / verify / rollback signals to it during
        # the spec branch; spec-off cycles (NoopDraftEngine) emit nothing.
        # Bench runners construct one collector per scenario and call
        # ``materialize`` after generation to populate the seven schema
        # fields in ``ScenarioResult.metadata``.
        self._spec_collector: SpecMetricCollector | None = spec_collector

    @property
    def spec_collector(self) -> SpecMetricCollector | None:
        """Read-only accessor for the per-engine speculative-metric
        collector. The bench runner constructs one collector per spec-on
        scenario, threads it in via the ``spec_collector`` constructor
        kwarg, and after generation reads it back through this
        property to call ``materialize`` and merge the seven schema
        fields into ``ScenarioResult.metadata``. Returns ``None`` for
        spec-off scenarios (no collector wired)."""
        return self._spec_collector

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
            # Drop any draft-side state held over from this request so
            # the next ``generate`` enters its cycle 0 cleanly.
            # ``reset`` is not on the I-5 Protocol surface (it is a
            # lifecycle convenience on ``DraftTargetEngine``); guard
            # via hasattr so ``NoopDraftEngine`` and any other
            # Protocol conformer without a reset method still work.
            reset = getattr(self._draft_engine, "reset", None)
            if reset is not None:
                reset(handle.req_id)
            # D-021 step 6 sub-unit (ε): drop the per-``req_id``
            # target_hidden + draft_caches when the drafter is a
            # ``TargetHiddenConsumer`` (DFlash class). Mirrors the
            # ``reset`` guard above — the method is on the optional
            # Protocol mixin, so non-target-hidden drafters
            # (``NoopDraftEngine`` / ``DraftTargetEngine``) skip this
            # cleanup branch without any guard cost.
            if isinstance(self._draft_engine, TargetHiddenConsumer):
                self._draft_engine.free_target_hidden(handle.req_id)
            # Drop any pending pre-draft recurrent snapshot. Idempotent
            # under ``free_state`` semantics — safe to call when no draft
            # window is open or when this request never entered the spec
            # path. D-021 step 5 sub-unit (e) slice 2.
            if isinstance(self._adapter, SpecRecurrentRollbackAdapter):
                self._adapter.free_state(handle.req_id)

    # --- private ---

    def _drive(
        self,
        prompt_ids: list[int],
        handle: KVHandle,
        params: SamplingParams,
    ) -> Iterator[int]:
        # v0.1 spec is greedy-only. Reject non-greedy spec at entry so
        # the spec branch below cannot silently produce wrong tokens
        # under a sampler the verify path does not honour.
        spec_active = not isinstance(self._draft_engine, NoopDraftEngine)
        if spec_active and params.temperature != 0.0:
            raise NotImplementedError(
                "Speculative decoding under temperature > 0 is a P-7 v0.2 "
                "feature; v0.1 D-021 step 5 is greedy-only "
                "(see plans/P6_SPEC_FOUNDATION_OPENING.md §4.3)."
            )

        # D-021 step 6 sub-unit (ε): TargetHiddenConsumer side channel
        # detection. Cached once at the top of ``_drive`` so the per-
        # cycle hot path can branch on a local rather than re-run
        # ``isinstance`` per iteration. C.1 ``DraftTargetEngine`` and
        # ``NoopDraftEngine`` do not implement this Protocol; the
        # branch below is a no-op for them and the existing engine
        # path stays byte-identical.
        target_hidden_drafter = (
            self._draft_engine
            if isinstance(self._draft_engine, TargetHiddenConsumer)
            else None
        )
        if target_hidden_drafter is not None and not isinstance(
            self._adapter, HiddenCaptureAdapter
        ):
            raise NotImplementedError(
                f"TargetHiddenConsumer drafter "
                f"({type(self._draft_engine).__name__}) requires a "
                "HiddenCaptureAdapter target. Qwen3.5 dense + MoE are "
                "supported per (αβ.1) / (αβ.2); other families are out "
                "of scope (no upstream DFlash drafter targets them per "
                "dflash_mlx.generate.DRAFT_REGISTRY)."
            )

        prompt_arr = mx.array(prompt_ids, dtype=mx.int32)

        # Prefill + first sample — measured as a single TTFT block.
        # When a TargetHiddenConsumer drafter is wired, the prefill
        # forward is routed through ``prefill_with_capture`` so the
        # captured hidden states seed cycle-1 ``target_hidden`` via
        # ``drafter.prime``. Last-position logits match the existing
        # ``prefill`` contract — the engine sampler is unchanged.
        t0 = time.perf_counter()
        if target_hidden_drafter is not None:
            assert isinstance(self._adapter, HiddenCaptureAdapter)
            logits, captured_prefill, _ = self._adapter.prefill_with_capture(
                prompt_arr,
                handle,
                target_hidden_drafter.capture_layer_ids,
            )
            target_hidden_drafter.prime(handle.req_id, captured_prefill)
        else:
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

        # Build a minimal RequestState the draft engine can consult.
        # ``NoopDraftEngine.propose`` ignores ``ctx``; ``DraftTargetEngine``
        # reads ``ctx.request.token_ids + ctx.output_token_ids`` for KV
        # catch-up bookkeeping. Both are kept in lockstep with
        # ``history`` (prompt + yielded tokens) on every cycle below.
        # ``request_id=handle.req_id`` matches the engine's allocated
        # id so slice 2a's per-``req_id`` keying inside the drafter
        # is predictable across cycles within one ``generate`` call.
        ctx = RequestState(
            request=Request(
                prompt="",
                sampling_params=params,
                request_id=handle.req_id,
                token_ids=tuple(prompt_ids),
            )
        )
        ctx.output_token_ids = [tok_int]

        gamma = self._verify_k - 1
        decode_count = 0
        n = 1
        while n < params.max_tokens:
            propose_start = (
                time.perf_counter()
                if self._spec_collector is not None
                else 0.0
            )
            drafts = self._draft_engine.propose(ctx, gamma)
            if not drafts.token_ids and target_hidden_drafter is None:
                # Spec-off (NoopDraftEngine) or non-target-hidden
                # drafter that declined this cycle. Single-token
                # decode — byte-identical to the pre-spec loop.
                #
                # ``TargetHiddenConsumer`` drafters explicitly skip
                # this branch even on empty drafts: they must route
                # through the spec verify path below so the captured
                # hidden states feed ``update_target_hidden`` and
                # ``target_hidden`` does not go stale across the
                # empty cycle. With ``draft_count = 0`` the verify
                # input is just ``[anchor]`` and the bonus emit at
                # the bottom of the cycle yields exactly one token,
                # functionally equivalent to ``decode_step`` here.
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
                ctx.output_token_ids.append(tok_int)
                if tok_int in params.stop_token_ids:
                    break
                continue

            # Spec path. The Protocol allows ``propose`` to return up to
            # γ drafts but fewer is legal (silica/speculative/engine.py
            # §I-5 docstring). All KV / commit / bonus math below keys
            # on the **actual** ``draft_count`` returned, not on the
            # propose-budget γ — using γ over-rolls back when the
            # drafter returns fewer items than asked. A drafter that
            # returns more than γ violates the Protocol; reject loud
            # rather than silently corrupting KV state.
            draft_count = len(drafts.token_ids)
            if draft_count > gamma:
                raise RuntimeError(
                    f"draft engine returned {draft_count} drafts but the "
                    f"Engine asked for at most γ = {gamma} (verify_k - 1); "
                    f"propose contract is 'up to k', see "
                    f"silica/speculative/engine.py I-5 docstring."
                )
            # verify_input = [anchor] + draft_count drafts (length
            # ``draft_count + 1``, ≤ verify_k). The verify forward
            # fills KV for all input positions in one batched call;
            # per-position logits feed greedy verification below.
            verify_input = mx.array(
                [tok_int] + list(drafts.token_ids), dtype=mx.int32
            )
            # D-021 step 5 sub-unit (g): record the propose cost. Timed
            # from before ``propose`` returned drafts (start at the top
            # of the loop) until just before the verify forward runs.
            if self._spec_collector is not None:
                self._spec_collector.record_propose(
                    draft_count=draft_count,
                    elapsed_ms=(time.perf_counter() - propose_start) * 1000.0,
                )
            # D-021 step 5 sub-unit (e) slice 2: capture the pre-draft
            # recurrent state on adapters that own one. Snapshot lives
            # under ``handle.req_id`` until ``commit_state`` (full
            # accept) or ``rollback_state`` (partial reject) closes the
            # window. Skipped on non-recurrent adapters; ``free_state``
            # in the ``generate`` finally tail covers terminal cleanup.
            if isinstance(self._adapter, SpecRecurrentRollbackAdapter):
                self._adapter.snapshot_pre_draft_state(handle.req_id)
            verify_start = (
                time.perf_counter()
                if self._spec_collector is not None
                else 0.0
            )
            # D-021 step 6 sub-unit (ε): route the verify forward
            # through the capture variant when a TargetHiddenConsumer
            # drafter is active. The capture is purely additive —
            # logits returned are bit-equivalent to the plain
            # ``run_verify_forward`` path (pinned in (αβ.3)), so the
            # verifier and bonus-emit math below are unchanged.
            captured_verify: dict[int, mx.array] | None = None
            if target_hidden_drafter is not None:
                assert isinstance(self._adapter, HiddenCaptureAdapter)
                verify_logits, captured_verify, _ = (
                    self._adapter.decode_step_multi_with_capture(
                        verify_input,
                        handle,
                        target_hidden_drafter.capture_layer_ids,
                    )
                )
            else:
                verify_logits, _ = run_verify_forward(
                    self._adapter, verify_input, handle
                )
            verify_elapsed_ms = (
                (time.perf_counter() - verify_start) * 1000.0
                if self._spec_collector is not None
                else 0.0
            )
            accepted_len = greedy_verify(drafts.token_ids, verify_logits)

            # Yield up to ``accepted_len`` drafts, but cap at the
            # ``max_tokens`` budget and bail on a stop token mid-yield.
            # ``yielded_count`` is the engine-level count (NOT the
            # verify-level ``accepted_len``) — both KV rollback below
            # and ``draft_engine.commit`` below key on it so the cache
            # state stays in sync with what was actually committed
            # to ``ctx.output_token_ids``.
            yielded_count = 0
            stop_hit = False
            for j in range(accepted_len):
                if n >= params.max_tokens:
                    stop_hit = True
                    break
                accepted_tok = drafts.token_ids[j]
                yield accepted_tok
                n += 1
                decode_count += 1
                history.append(accepted_tok)
                ctx.output_token_ids.append(accepted_tok)
                yielded_count += 1
                if accepted_tok in params.stop_token_ids:
                    stop_hit = True
                    break
            # Post-loop ``max_tokens`` check. The in-loop guard fires at
            # the **start** of an iteration, so an accept_len-token cycle
            # that exits the for loop naturally with ``n == max_tokens``
            # never trips it; the bonus emit below would otherwise
            # overshoot ``max_tokens`` by one. Reproduces under
            # spec-on at ``max_tokens == prefill_yield + γ * cycles``
            # (e.g. verify_k=4 / max_tokens=4 yields 5 tokens without
            # this guard). Spec-off has the same hard cap via the top-of-
            # loop ``while n < params.max_tokens``; spec-on must mirror it.
            if n >= params.max_tokens:
                stop_hit = True

            # KV rollback: undo every draft slot past ``yielded_count``
            # in the verify forward — the (draft_count - accepted_len)
            # drafts the verifier rejected, plus any (accepted_len -
            # yielded_count) accepted-but-not-yielded drafts (which
            # only happens when ``max_tokens`` or a stop token cut the
            # yield short). Keys on ``draft_count`` (the actual count
            # returned by ``propose``), NOT on γ — a drafter returning
            # fewer than γ would otherwise have its committed prefix
            # over-trimmed.
            un_committed = draft_count - yielded_count
            # D-021 step 5 sub-unit (g): record this verify cycle. Done
            # before the rollback path runs so a partial-reject cycle's
            # verify cost is attributed regardless of the rollback path
            # taken (recurrent vs non-recurrent share the same verify
            # forward). Rollback events are recorded separately below.
            if self._spec_collector is not None:
                self._spec_collector.record_verify(
                    accepted_len=accepted_len,
                    yielded_count=yielded_count,
                    elapsed_ms=verify_elapsed_ms,
                )
                if un_committed > 0:
                    self._spec_collector.record_rollback()
            # D-021 step 6 sub-unit (ε): update the target-hidden
            # drafter's per-``req_id`` ``target_hidden`` BEFORE the
            # target-side KV / recurrent rollback runs. The captured
            # hidden states reference distinct mx.arrays from the KV
            # cache (intermediate layer outputs, not cache writes),
            # so ordering is correctness-neutral — it is fixed at
            # "update first, rollback after" for clarity, matching
            # the F-1 state machine in §4.1 of the OPENING.
            if target_hidden_drafter is not None:
                assert captured_verify is not None
                target_hidden_drafter.update_target_hidden(
                    handle.req_id,
                    captured_verify,
                    yielded_count,
                )
            recurrent_adapter = (
                self._adapter
                if isinstance(self._adapter, SpecRecurrentRollbackAdapter)
                else None
            )
            if un_committed > 0:
                # D-021 step 5 sub-unit (e) slice 2 — recurrent path:
                # the verify forward advanced both attention KV and
                # recurrent state through ``draft_count + 1`` positions.
                # Reset attention KV to the pre-cycle offset by
                # trimming all verify writes, restore recurrent state
                # from the snapshot, then replay ``decode_step_multi``
                # over the committed prefix to drive both layer types
                # forward in lockstep through ``1 + yielded_count``
                # tokens. See orientation §3 [F-3] / [F-3a] for why
                # the simpler "trim by un_committed only and replay"
                # sequence corrupts global-attention context during
                # replay.
                if recurrent_adapter is not None:
                    self._kv_manager.rollback(
                        handle.req_id, draft_count + 1
                    )
                    recurrent_adapter.rollback_state(
                        handle.req_id, un_committed
                    )
                    replay_input = verify_input[: 1 + yielded_count]
                    # Symmetric with the verify forward above: route
                    # through ``run_verify_forward`` so adapters that
                    # have not shipped a real ``decode_step_multi``
                    # take the per-step ``decode_step`` fallback for
                    # replay too. Logits are discarded here — replay
                    # exists only for the side effects on KV +
                    # recurrent state.
                    run_verify_forward(self._adapter, replay_input, handle)
                else:
                    self._kv_manager.rollback(handle.req_id, un_committed)
            elif recurrent_adapter is not None:
                # Full accept (``yielded_count == draft_count``): no
                # rollback, the verify forward already landed both
                # attention KV and recurrent state on the committed
                # boundary. Drop the pending snapshot so the next
                # cycle's ``snapshot_pre_draft_state`` does not raise
                # the nested-window guard.
                recurrent_adapter.commit_state(handle.req_id, yielded_count)

            # Inform the draft engine. ``yielded_count`` (not
            # ``accepted_len``) is the count the draft's own KV must
            # roll back to so it tracks the engine's committed state.
            self._draft_engine.commit(ctx, yielded_count)

            if stop_hit:
                break

            # Sample the bonus token. On partial accept (``yielded_count
            # < draft_count``) the bonus comes from the rejected
            # draft's logits at index ``yielded_count``. On full accept
            # of all returned drafts it comes from the prediction past
            # the last accepted draft — the last position of the
            # verify input.
            bonus_idx = (
                yielded_count
                if yielded_count < draft_count
                else int(verify_input.size) - 1
            )
            bonus_logits = verify_logits[bonus_idx]
            bonus_scalar = self._sampler.sample(
                bonus_logits, mx.array(history, dtype=mx.int32), params
            )
            tok_int = int(bonus_scalar.item())
            yield tok_int
            n += 1
            decode_count += 1
            history.append(tok_int)
            ctx.output_token_ids.append(tok_int)
            # D-021 step 5 sub-unit (g): record the bonus emission so
            # ``tokens_per_target_forward`` counts the bonus token, not
            # only the accepted drafts. Reached only when ``stop_hit``
            # was False above (max_tokens / stop-token mid-yield paths
            # break before reaching this point and emit no bonus).
            if self._spec_collector is not None:
                self._spec_collector.record_bonus()
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
        ``plans/P4_5_CHUNKED_PREFILL_OPENING.md``.

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
            # D-021 step 5 sub-unit (c) slice 1: pass spec-decoding
            # config through to the batcher so ``Engine(...,
            # draft_engine=real)`` reaches both ``generate`` and
            # ``generate_batch`` symmetrically. Without the
            # pass-through ``generate_batch`` would silently degrade
            # to spec-off regardless of the engine-level config —
            # constructing the batcher with NoopDraftEngine.
            draft_engine=self._draft_engine,
            verify_k=self._verify_k,
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
        # ``plans/P2_UNIT_16C_PREP.md`` §1 I-5.
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
    P-4.5-B.1 / `plans/P4_5_CHUNKED_PREFILL_OPENING.md` §6.1.
    """
    return sorted(admissions, key=lambda a: len(a[1]))


def _initial_cohort_cap(
    admissions_sorted: Sequence[tuple[int, list[int]]],
    effective_batch_size: int,
    spread_ratio_threshold: float,
) -> int:
    """How many leading (shortest) admissions go into the initial cohort.

    P-4.5-B.1 / `plans/P4_5_CHUNKED_PREFILL_OPENING.md` §6.1 spec.

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
