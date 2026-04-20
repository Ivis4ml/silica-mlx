"""silica.scheduler.batcher — ContinuousBatcher step loop (Units 16a / 16b).

P-2 Opening v2.3 fixes the shape this class will take over Units 16a-d:

  - **16a** (shipped): B=1 scaffolding. Build the step-loop phases
    (admission / prefill / decode / sample / finalize), the BatchEvent
    stream, the capability gate, and the slot_table + BatchKVCache
    wiring. Output is token-identical to P-1's ``Engine.generate`` on
    Qwen3-0.6B (acceptance in ``tests/test_p2_preload_parity.py``).
  - **16b** (this revision): fixed-cohort B>1. All rows admit at step 0
    via ``add_request`` (admission closes on the first ``step`` call —
    subsequent calls raise ``NotImplementedError("... 16c")``). Prefill
    runs as **one** batched forward over the left-padded ``(B, T_max)``
    token tensor; decode as **one** batched ``(B, 1)`` forward per
    step. Unit tests pin row isolation and left-padding arithmetic;
    real-model parity is against direct mlx-lm batched execution because
    B>1 fp16 batched SDPA can drift from B=1 solo generations.
  - **16c**: mid-run admission via ``BatchKVCache.extend`` + prefix
    cache integration (``RadixPrefixCache.lookup`` + copy source K/V
    into the new row).
  - **16d**: budget-aware preemption via ``BatchKVCache.filter`` +
    ``RequestState.transition(PREEMPTED)`` + re-admission cycle; abort
    with ``BatchEvent.aborted`` on ``RejectDecision``.

Capability gate (Layer 3 of the v2.3 three-layer stack): the batcher
rejects adapters whose ``attention_pattern`` contains any kind other
than ``AttentionKind.GLOBAL``. Mamba / DeltaNet / sliding / hybrid
families refuse themselves at this gate; the batcher never checks
``isinstance``.

Scope note: 16b retains 16a's BatchKVCache-from-day-one choice and adds
per-row ``left_padding`` arithmetic. Rows that reach a terminal state
mid-cohort are kept in the batch — they continue to occupy their slot
in the batched forward (fed a placeholder pad token) but are NOT
sampled or emitted from. The row's ``generated`` list freezes on
termination. Actually filtering out finished rows (reclaiming slots /
reshuffling indices) is 16d's filter+re-index path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

from silica.core.events import BatchEvent
from silica.core.request import Request, RequestState, RequestStatus
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.mlx.runner import forward_batched
from silica.models.adapter import AttentionKind, ModelAdapter
from silica.scheduler.budget import (
    AdmitAfterEvictDecision,
    AdmitAfterPreemptDecision,
    AdmitDecision,
    MemoryBudgeter,
    RejectDecision,
)
from silica.scheduler.seed_kv import build_seeded_batch_kv
from silica.weights.provider import WeightProvider
from silica.weights.resident import ResidentWeightProvider


class _BudgetEvictUnderrun(RuntimeError):
    """Raised by ``_apply_evict`` when ``evict_until`` frees fewer blocks
    than the budgeter's decision asked for (16d-3).

    Private to this module. The ``_admit_waiting_requests`` Phase A
    branch for ``AdmitAfterEvictDecision`` catches it, aborts the
    triggering admission with ``finish_reason='budget-exhausted'``, and
    moves on. The batcher's ``self._rows`` / ``self._batch_cache`` /
    ``self._budgeter`` state stays untouched — Phase A's apply_admit
    for this pending has not yet run when the raise fires.
    """


@dataclass(frozen=True)
class _PendingAdmit:
    """One request waiting for cohort admission (16c.1).

    Only held in the batcher's waiting queue; popped into an active
    ``_BatchRow`` during the admit phase of some future ``step()``.

    ``is_replay`` (16d-4a): ``True`` when the pending was re-enqueued
    by ``_apply_preempt`` (16d-4c lands the re-enqueue flow). Replays
    are excluded from triggering further preempts — Phase A's B-9
    anti-ping-pong rule (16d-4c/d) keys off this flag. Default
    ``False`` keeps every existing ``add_request`` / ``_admit_*``
    construction site unchanged.
    """

    req_index: int
    prompt_ids: tuple[int, ...]
    params: SamplingParams
    is_replay: bool = False


@dataclass
class _BatchRow:
    """Per-row state held by the batcher (internal).

    One row = one request. The row persists from admission through
    DECODE / DONE / ABORTED / PREEMPTED. ``generated`` only stores
    tokens that were sampled and emitted — placeholder feeds used to
    keep ``_idx`` in lockstep after the row terminates do NOT append
    here (so ``len(generated)`` keeps a clean "how many tokens did this
    row actually produce" reading at any point).
    """

    req_index: int
    req_id: str
    prompt_ids: list[int]
    params: SamplingParams
    state: RequestState
    generated: list[int] = field(default_factory=list)


class ContinuousBatcher:
    """P-2 scheduler step loop — Units 16a (B=1) + 16b (fixed-cohort B>1)."""

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        sampler: Sampler | None = None,
        weight_provider: WeightProvider | None = None,
        max_batch_size: int = 1,
        prefix_cache: RadixPrefixCache | None = None,
        budgeter: MemoryBudgeter | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1, got {max_batch_size}"
            )
        # P-3-C3b: the prefix-cache extract path (``_extract_and_insert_prefix``
        # + the seeded-admission path through the radix trie) assumes
        # per-token K/V slicing on every layer's cache. DeltaNet
        # recurrent state is a running accumulation over the full
        # sequence; it cannot be sliced to an arbitrary block-aligned
        # prefix (see ``docs/P3_DELTANET_SURVEY.md`` C-open-3). Reject
        # the combination at construction time with a specific error
        # rather than crashing deep inside ``_extract_and_insert_prefix``
        # at an ``AttributeError`` on ``ArraysCache.offset`` / ``.keys``.
        # The guard sits BEFORE ``_enforce_capability_gate`` so the more
        # specific error surfaces first and the placement stays stable
        # once C3c lifts the capability gate for hybrid adapters running
        # without a prefix cache.
        if prefix_cache is not None:
            caps = adapter.capabilities()
            if caps.has_recurrent_state:
                raise NotImplementedError(
                    "ContinuousBatcher does not support a RadixPrefixCache "
                    "on adapters with has_recurrent_state=True "
                    f"(attention_kinds="
                    f"{sorted(k.value for k in caps.attention_kinds)!r}): "
                    "DeltaNet / recurrent state is a running accumulation "
                    "over the full sequence and cannot be sliced into "
                    "block-aligned prefix K/V entries (see "
                    "docs/P3_DELTANET_SURVEY.md C-open-3). Pass "
                    "prefix_cache=None to run hybrid adapters under the "
                    "miss-only admission path."
                )
            # P-3-D3: sliding-window attention needs its own guard against
            # the seeded-admission path. ``build_seeded_batch_kv`` /
            # ``_extract_and_insert_prefix`` emit ``BatchKVCache`` per
            # layer; adapting that to ``BatchRotatingKVCache``'s window
            # truncation, ``offset`` arithmetic, and ``rotated`` state
            # has not been validated. D3 only commits to the
            # ``prefix_cache=None`` path for SLIDING-bearing adapters;
            # future work widens the seed path rather than bypassing
            # this guard.
            if AttentionKind.SLIDING in caps.attention_kinds:
                raise NotImplementedError(
                    "ContinuousBatcher does not support a RadixPrefixCache "
                    "on adapters whose attention_kinds include "
                    f"AttentionKind.SLIDING "
                    f"(attention_kinds="
                    f"{sorted(k.value for k in caps.attention_kinds)!r}): "
                    "the prefix-cache seed path emits BatchKVCache per "
                    "layer, and the window-truncation / offset / rotated "
                    "semantics of BatchRotatingKVCache under seeded "
                    "admission are not yet validated (P-3-D3 local "
                    "follow-up). Pass prefix_cache=None to run "
                    "sliding-bearing adapters under the miss-only path."
                )
        self._enforce_capability_gate(adapter)
        self._adapter = adapter
        self._sampler = sampler or Sampler()
        self._model = adapter.build(weight_provider or ResidentWeightProvider())
        self._max_batch_size = max_batch_size
        self._rows: list[_BatchRow] = []
        # As of P-3-C3a / P-3-D3 the cache list is adapter-produced and
        # may mix ``BatchKVCache`` (global attention), ``ArraysCache``
        # (DeltaNet layers, Qwen3.5 hybrid), and ``BatchRotatingKVCache``
        # (sliding-window layers, Gemma4). ``list[Any]`` so the
        # annotation tracks the heterogeneous reality.
        self._batch_cache: list[Any] | None = None
        # Cohort-prep flag. First step() seals the pre-step cohort and
        # allocates the batched cache; subsequent add_request calls go
        # to the waiting queue instead of directly into self._rows
        # (16c.1 step 3 enables this mid-run admission path).
        self._cohort_prepared: bool = False
        # Waiting queue (16c.1). Populated by mid-run add_request calls;
        # drained by the admit phase of step().
        self._waiting_queue: deque[_PendingAdmit] = deque()
        # Slot table (16c.1): req_index → current row index. Rebuilt
        # after every filter (§1 I-3). Kept as an observable field so
        # tests can verify post-reclaim coherence directly.
        self._slot_table: dict[int, int] = {}
        # 16c.2 step 4: optional shared-prefix cache. When None (default)
        # every prefix-related code path short-circuits — the batcher
        # behaves bit-identically to 16c.1 (invariant S-6). Callers own
        # the cache's lifetime so it can span multiple Engine.generate_batch
        # invocations (that is the whole point of prefix reuse).
        self._prefix_cache: RadixPrefixCache | None = prefix_cache
        # 16c.2 observability counters. ``forward_prompt_tokens`` tracks
        # tokens fed through prefill forwards (excludes decode steps).
        # ``prefix_hits`` counts admissions that used the hit path. Step
        # 5 acceptance (PLAN §7 #2) asserts a closed-form reduction on
        # these counters.
        self.forward_prompt_tokens: int = 0
        self.prefix_hits: int = 0
        # 16d-2b: optional budget-aware admission policy. When None
        # (default), every admission proceeds as in 16c.2 — no budget
        # check (invariant B-1: no-budgeter bit-identical to 16c.2).
        # When set, _admit_waiting_requests consults budgeter.admit per
        # pending and routes to admit / reject / evict / preempt.
        # 16d-3 wires evict; 16d-4c wires preempt + replay requeue.
        self._budgeter: MemoryBudgeter | None = budgeter
        # 16d counters paralleling 16c.2's hits/forward counters.
        # ``aborts`` increments on RejectDecision / budget failure,
        # ``evictions`` on successful evict decisions, and ``preempts``
        # on successful fresh-pending preempts.
        self.aborts: int = 0
        self.evictions: int = 0
        self.preempts: int = 0

    # --- public admission / stepping surface ---

    def add_request(
        self,
        req_index: int,
        prompt_ids: list[int],
        params: SamplingParams,
    ) -> None:
        """Enqueue a request for admission.

        Two paths:

        - **Pre-step (cohort not yet prepared)**: request is appended
          directly to ``self._rows``, same as 16b. ``_prepare_cohort``
          will seal them as the initial cohort on the first ``step``
          call. This is what 16a/16b test fixtures rely on.
        - **Mid-run (cohort already prepared)**: request is appended to
          ``_waiting_queue``. Phase 2 of a later ``step`` pops it, runs
          its own batched prefill, and extends main via
          ``BatchKVCache.extend``.

        Raises:
            RuntimeError: when cohort is pre-step AND ``self._rows`` is
                already at ``max_batch_size``. Mid-run requests are
                accepted into the unbounded waiting backlog; active
                physical-row capacity is enforced later by the admit
                phase.
            ValueError: if ``prompt_ids`` is empty.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")

        if self._cohort_prepared:
            # Mid-run admission: enqueue as backlog. ``max_batch_size``
            # bounds **active physical rows**, not queue length — the
            # waiting queue is an unbounded backlog that the admit
            # phase drains up to whatever capacity remains after the
            # preceding reclaim. Capping the queue here would make the
            # canonical "submit N > max_batch_size prompts, admit as
            # slots free" pattern (prep doc §3 16c.1 acceptance)
            # impossible to express naturally.
            self._waiting_queue.append(
                _PendingAdmit(
                    req_index=req_index,
                    prompt_ids=tuple(prompt_ids),
                    params=params,
                )
            )
            return

        # Pre-step admission: direct append (16b-compatible path).
        if len(self._rows) >= self._max_batch_size:
            raise RuntimeError(
                f"ContinuousBatcher capacity {self._max_batch_size} reached"
            )
        request = Request(
            prompt="",  # text not carried through the batcher
            sampling_params=params,
            request_id=f"req-{req_index}",
            token_ids=tuple(prompt_ids),
        )
        state = RequestState(request=request)
        new_row_idx = len(self._rows)
        self._rows.append(
            _BatchRow(
                req_index=req_index,
                req_id=f"req-{req_index}",
                prompt_ids=list(prompt_ids),
                params=params,
                state=state,
            )
        )
        self._slot_table[req_index] = new_row_idx

    def has_active(self) -> bool:
        """True iff at least one row is not in a terminal state.

        Literal semantics — used by internal phase decisions. After
        16c.1 step 2 lands reclaim, Engine's drain loop should use
        :py:meth:`has_work` because terminal rows pending reclaim keep
        ``self._rows`` populated even when ``has_active()`` is already
        False.
        """
        return any(not r.state.is_terminal for r in self._rows)

    def has_work(self) -> bool:
        """True iff anything remains that the next ``step`` could do.

        Covers three distinct states: an active row, a terminal row
        that has not yet been reclaimed (deferred reclaim — see
        ``docs/P2_UNIT_16C_PREP.md`` §1 I-5), or a pending request in
        the waiting queue. ``Engine.generate_batch`` will loop on this
        once 16c.1 step 2 lands reclaim, so the cohort drains cleanly
        even when the last sample phase terminates every row.
        """
        return bool(self._waiting_queue) or bool(self._rows)

    def step(self) -> list[BatchEvent]:
        """Advance the cohort by one scheduler iteration.

        Phase order (16c.1 step 3 onwards):

          1. **Reclaim** — drop terminal rows from ``self._rows`` and
             from the batched cache. If all rows were terminal, reset
             the batched cache to ``None`` so Phase 2 can install a
             fresh one instead of extending into stale physical rows.
             Docs: §1 I-5 + §4 Reclaim flow.
          2. **Admit** — drain the waiting queue (respecting
             ``max_batch_size``) into newly-admitted rows; run one
             batched prefill for those rows; extend main cache with
             the result (or replace it if main is None).
          3. **Forward** — one batched prefill (if any row is PREFILL)
             or one batched decode (otherwise). Skipped when Phase 2
             ran admissions this step (prefill T vs decode T=1 can't
             mix; existing DECODE rows idle this step).
        """
        if not self._cohort_prepared:
            self._prepare_cohort()

        # Phase 1: reclaim deferred terminals.
        self._reclaim_terminated()

        # Phase 2: mid-run admission (if waiting queue non-empty).
        admit_events = self._admit_waiting_requests()
        if admit_events:
            # Admit ran its own batched prefill forward; existing DECODE
            # rows idle this step so prefill-T vs decode-T=1 don't mix.
            return admit_events

        # Phase 3: batched forward for the incumbent cohort.
        if not self.has_active():
            return []
        if any(r.state.status == RequestStatus.PREFILL for r in self._rows):
            return self._prefill_phase()
        return self._decode_phase()

    # --- phase methods ---

    def _prepare_cohort(self) -> None:
        """Seal the initial cohort; allocate per-layer cache with per-row
        left_padding.

        Rows with shorter prompts get more left padding so that, after
        the prefill forward, the ``offset[i] == T_max - left_padding[i]``
        equals each row's true prompt length.

        As of P-3-C3a the per-layer cache list is produced by the
        adapter's ``make_batch_cache(left_padding)`` method when it
        exists, falling back to an all-``BatchKVCache`` list (what the
        scheduler hardcoded pre-C3a). Letting the adapter produce
        this list is what enables hybrid adapters like
        ``Qwen3_5Adapter`` to interleave ``ArraysCache`` for DeltaNet
        layers without the scheduler having to grow family-specific
        logic. After P-3-C3c the capability gate admits both
        ``GLOBAL`` and ``HYBRID_DELTANET`` adapters (subject to the
        C3b prefix-cache guard), so the hybrid factory path is live
        for real Qwen3.5 workloads — not only the plain-Qwen3
        fallback.

        After this runs, subsequent ``add_request`` calls go to the
        waiting queue (mid-run admission path, see Phase 2).
        """
        self._cohort_prepared = True
        if not self._rows:
            return
        max_prompt_len = max(len(r.prompt_ids) for r in self._rows)
        left_padding = [
            max_prompt_len - len(r.prompt_ids) for r in self._rows
        ]
        self._batch_cache = _make_batch_cache(self._adapter, left_padding)
        for row in self._rows:
            row.state.transition(RequestStatus.PREFILL, reason="admit-cohort")

    def _reclaim_terminated(self) -> None:
        """Drop terminal rows from the batch before the forward phase.

        Per ``docs/P2_UNIT_16C_PREP.md`` §4 Reclaim flow:

          - If no row is terminal, no-op (cheap path; most steps).
          - Otherwise compute ``kept`` and either (a) ``filter(kept)`` on
            each layer's BatchKVCache + prune ``self._rows`` and
            rebuild ``self._slot_table`` (partial reclaim), or (b)
            reset ``self._batch_cache = None`` entirely if ``kept`` is
            empty — extending into a batch of all-stale rows would
            corrupt future admissions (§4 Fix 2).

        Called at the top of every ``step()`` (after ``_prepare_cohort``).

        16c.2 step 4 sub-commit 2 hooks the eager-extract path here,
        **before** filter / cache drop: for each terminating row,
        slice its block-aligned prefix K/V out of the current batch
        cache and register with ``self._prefix_cache.insert_detached``.
        The slice is materialised via ``mx.eval`` so filter's
        shift-left mutation (or the all-terminal cache drop) cannot
        corrupt it afterwards (Gate-0.75 probe B is the physical
        evidence; invariant S-3b pins the rule).
        """
        if not any(r.state.is_terminal for r in self._rows):
            return
        kept = [i for i, r in enumerate(self._rows) if not r.state.is_terminal]
        terminated = [
            i for i, r in enumerate(self._rows) if r.state.is_terminal
        ]

        # 16c.2 step 4 sub-commit 2: extract prefix K/V BEFORE filter.
        # Extract runs for every terminating row regardless of whether
        # we will filter or drop the whole cache. Gated on cache being
        # present because sub-commit 1's init left it None.
        if self._prefix_cache is not None and self._batch_cache is not None:
            for row_idx in terminated:
                self._extract_and_insert_prefix(row_idx)

        # 16d-2b: release budgeter reservations for terminal rows BEFORE
        # the filter / cache drop (B-6' terminal branch). Release is
        # bookkeeping only — ordering relative to the extract loop above
        # does not matter for correctness, but both must precede the
        # destructive filter / cache-None mutations so a mid-release
        # raise cannot leave the batcher with kept rows whose
        # reservations have been lost.
        if self._budgeter is not None:
            for row_idx in terminated:
                self._budgeter.release(self._rows[row_idx].req_id)

        if not kept:
            # Every row terminated in the last step. The batched cache
            # is full of stale K/V no live row owns; extending into it
            # would put new rows after the stale rows, corrupting
            # slot_table. Drop it — Phase 2 admit (16c.1 step 3) will
            # install a fresh cache on the next admission wave.
            self._rows = []
            self._slot_table = {}
            self._batch_cache = None
            return
        assert self._batch_cache is not None
        for layer_cache in self._batch_cache:
            layer_cache.filter(kept)
        self._rows = [self._rows[i] for i in kept]
        self._rebuild_slot_table()

    def _extract_and_insert_prefix(self, row_idx: int) -> None:
        """Slice a terminating row's block-aligned prefix K/V out of the
        current batched cache and register it with ``self._prefix_cache``
        (16c.2 step 4 sub-commit 2).

        Caller guarantees ``self._prefix_cache is not None`` and
        ``self._batch_cache is not None``. Safe to call before the
        reclaim's filter / drop step (Gate-0.75 probe B proved
        ``mx.contiguous`` + ``mx.eval`` slices survive the source's
        subsequent filter).

        Two subtleties pinned by the prep doc (S-3a, S-3b):

          - **Cache contents vs accounting.** mlx-lm's cache holds K/V
            for ``prompt_ids + generated[:-1]``. ``row.generated[-1]``
            was sampled from the previous forward's logits and has
            not yet been fed through any forward; its K/V is NOT in
            cache. Using ``prompt_ids + generated`` as the source of
            "computed ids" would push an extra token into the radix
            tree with no backing K/V, corrupting future hits.
          - **Left-padding offset.** Each layer's ``keys`` tensor is
            shaped ``(B, H, T_max, D)`` with row ``row_idx``'s logical
            token 0 at axis-2 index ``left_padding[row_idx]``. Slicing
            from axis-2 zero would pull pad values.

        No-ops when the aligned prefix is shorter than one block —
        a partial trailing block is not retainable under option B.
        """
        assert self._prefix_cache is not None
        assert self._batch_cache is not None

        row = self._rows[row_idx]
        if row.generated:
            computed_ids: list[int] = list(row.prompt_ids) + list(
                row.generated[:-1]
            )
        else:
            computed_ids = list(row.prompt_ids)

        # S-3a: cache's own authoritative accounting must agree.
        computed_len = int(self._batch_cache[0].offset[row_idx].item())
        assert len(computed_ids) == computed_len, (
            f"row {row_idx}: computed_ids len {len(computed_ids)} != "
            f"offset[row_idx] {computed_len} — sampler/forward drift"
        )

        block_size = self._prefix_cache.block_size
        aligned_tokens = (computed_len // block_size) * block_size
        if aligned_tokens < block_size:
            return

        # S-3b: every layer's BatchKVCache shares the same left_padding
        # vector (they evolve together under extend/filter), so layer 0
        # is authoritative for this row.
        base = int(self._batch_cache[0].left_padding[row_idx].item())

        tokens_prefix = computed_ids[:aligned_tokens]
        num_layers = len(self._batch_cache)
        detached_blocks: list[list[tuple[mx.array, mx.array]]] = []
        for b_idx in range(aligned_tokens // block_size):
            start = b_idx * block_size
            end = start + block_size
            per_layer: list[tuple[mx.array, mx.array]] = []
            for layer_idx in range(num_layers):
                layer_cache = self._batch_cache[layer_idx]
                keys = layer_cache.keys
                values = layer_cache.values
                if keys is None or values is None:
                    raise AssertionError(
                        f"row {row_idx} layer {layer_idx}: "
                        "cache tensors are not initialised"
                    )
                k = mx.contiguous(
                    keys[
                        row_idx : row_idx + 1, :, base + start : base + end, :
                    ]
                )
                v = mx.contiguous(
                    values[
                        row_idx : row_idx + 1, :, base + start : base + end, :
                    ]
                )
                per_layer.append((k, v))
            detached_blocks.append(per_layer)

        # S-3b eager materialisation: force each slice to its own
        # backing memory BEFORE the source cache's filter / drop.
        mx.eval(
            *[arr for pl in detached_blocks for (k, v) in pl for arr in (k, v)]
        )
        self._prefix_cache.insert_detached(tokens_prefix, detached_blocks)

    def _rebuild_slot_table(self) -> None:
        """Re-derive ``slot_table`` from the current ``self._rows``.

        Must be called after any operation that reshuffles
        ``self._rows`` — notably ``filter`` (which mlx-lm's BatchKVCache
        re-indexes to ``0..K-1``) and, in 16c.1 step 3, ``extend`` (to
        record the newly-admitted rows' indices).
        """
        self._slot_table = {row.req_index: i for i, row in enumerate(self._rows)}

    def _find_row_by_req_id(self, req_id: str) -> int | None:
        """Return the current row index for ``req_id``, or None if absent.

        Linear scan over ``self._rows``. The active row count is bounded
        by ``max_batch_size`` (single-digit in practice), so asymptotic
        cost is trivial; keeping the implementation simple beats parsing
        the ``"req-{N}"`` id format, which would couple this method to a
        naming convention only ``add_request`` / ``_admit_*`` currently
        know about.

        Used by 16d-4's preempt path (lands in 16d-4b/c) to resolve the
        victim row from the budgeter's
        ``AdmitAfterPreemptDecision.preempt_req_id`` string. The row
        index is the identity the caller actually needs — a row object
        post-filter would invite confusion about whether it still
        represents an active slot.
        """
        for row_idx, row in enumerate(self._rows):
            if row.req_id == req_id:
                return row_idx
        return None

    def _preempt_active_row(
        self, victim_req_id: str
    ) -> _BatchRow | None:
        """Strip one active row out of the batch and return it (16d-4b).

        Does steps 1-4 of prep doc ``P2_UNIT_16D_PREP.md`` §4.4:

          1. Extract victim's aligned prefix K/V into
             ``self._prefix_cache`` (if present). Safe no-op when the
             cache is absent — the row's prompt work is lost, but
             preempt still completes.
          2. Transition state PREFILL/DECODE → PREEMPTED → WAITING on
             the OLD ``RequestState`` for state-machine correctness.
             The returned row carries that state so callers can
             inspect the chain.
          3. ``filter`` the victim out of every layer's BatchKVCache,
             prune ``self._rows``, and rebuild ``self._slot_table``.
             When the victim was the only row, the batch cache is
             dropped to None (same rationale as ``_reclaim_terminated``
             §4 Fix 2).
          4. Release the victim's reservation on
             ``self._budgeter`` so the triggering admission's own
             ``apply_admit`` lands in consistent accounting.

        Re-enqueue (step 5 in prep doc) is NOT this helper's job —
        16d-4c's ``_apply_preempt`` will wrap this to build the
        composite prompt + ``appendleft`` the replay ``_PendingAdmit``.

        Returns:
            The victim ``_BatchRow`` on success (caller reads its
            ``prompt_ids`` / ``generated`` / ``params`` to construct the
            re-admission). ``None`` when the request is no longer
            active (not in ``self._rows``) — that is B-7's race path.
        """
        if self._batch_cache is None:
            # No active batched cache → no row to preempt. Distinct
            # from "victim missing" but maps to the same None result.
            return None
        victim_row_idx = self._find_row_by_req_id(victim_req_id)
        if victim_row_idx is None:
            return None
        victim_row = self._rows[victim_row_idx]

        # 1. Extract prefix K/V into the prefix cache (optional).
        if self._prefix_cache is not None:
            self._extract_and_insert_prefix(victim_row_idx)

        # 2. State transitions on the OLD RequestState.
        victim_row.state.transition(
            RequestStatus.PREEMPTED, reason="budget-preempt"
        )
        victim_row.state.transition(
            RequestStatus.WAITING, reason="re-admit"
        )

        # 3. Drop the victim from batched cache + self._rows.
        kept = [
            i for i in range(len(self._rows)) if i != victim_row_idx
        ]
        if not kept:
            self._batch_cache = None
            self._rows = []
        else:
            for layer_cache in self._batch_cache:
                layer_cache.filter(kept)
            self._rows = [self._rows[i] for i in kept]
        self._rebuild_slot_table()

        # 4. Release budget reservation for the victim.
        if self._budgeter is not None:
            self._budgeter.release(victim_req_id)

        return victim_row

    def _apply_preempt(self, victim_req_id: str) -> bool:
        """Preempt ``victim_req_id`` and re-enqueue it as a replay (16d-4c).

        Wraps ``_preempt_active_row`` with step 5 of prep doc
        ``P2_UNIT_16D_PREP.md`` §4.4:

          - ``composite_prompt = prompt_ids + generated`` (full — no
            ``[:-1]`` trim). B-5: every token the victim emitted was
            observed by the caller via ``BatchEvent.token``; the replay
            must resume at the NEXT position to avoid duplicating the
            last emitted token on the caller side.
          - ``replay_params = params.model_copy(update={"max_tokens":
            remaining})`` where ``remaining = max_tokens - len(generated)``.
            Q-2 algebra: ``(len(composite) + remaining) == (n_prompt +
            max_tokens)`` — the replay's worst-case bytes equal the
            original's, so the budgeter's admission decision at re-admit
            time matches the original's shape.
          - ``appendleft`` the replay ``_PendingAdmit`` with
            ``is_replay=True`` so Phase A's B-9 rule excludes it from
            triggering further preempts.

        Returns True on successful preempt + re-enqueue, False when
        the victim has already left the active batch (B-7 race;
        caller requeues its triggering admission).

        Raises AssertionError when ``remaining <= 0`` — the budgeter
        picked a victim at its max_tokens cap that should have already
        transitioned to DONE. The check runs BEFORE any state mutation
        so the loud-fail leaves the batcher in a coherent state (no
        half-preempted row, no dangling replay pending).
        """
        # Pre-check: find victim + validate remaining BEFORE mutating
        # state. This duplicates one _find_row_by_req_id scan vs pushing
        # the check into _preempt_active_row, but keeps the helper's
        # "does extract + filter" semantics honest and keeps the
        # AssertionError tripwire non-destructive.
        if self._batch_cache is None:
            return False
        victim_row_idx = self._find_row_by_req_id(victim_req_id)
        if victim_row_idx is None:
            return False
        victim_row = self._rows[victim_row_idx]
        remaining = (
            victim_row.params.max_tokens - len(victim_row.generated)
        )
        if remaining <= 0:
            raise AssertionError(
                f"preempt victim {victim_req_id}: remaining tokens = "
                f"{remaining} (victim should have been DONE)"
            )

        detached = self._preempt_active_row(victim_req_id)
        # The second lookup inside _preempt_active_row scans the same
        # self._rows we just read; no mutation in between so a None
        # here would be a consistency violation.
        assert detached is not None

        composite_prompt = tuple(
            list(detached.prompt_ids) + list(detached.generated)
        )
        replay_params = detached.params.model_copy(
            update={"max_tokens": remaining}
        )
        self._waiting_queue.appendleft(
            _PendingAdmit(
                req_index=detached.req_index,
                prompt_ids=composite_prompt,
                params=replay_params,
                is_replay=True,
            )
        )
        return True

    def _prefill_phase(self) -> list[BatchEvent]:
        """One batched forward over the left-padded prompt tensor."""
        assert self._batch_cache is not None
        tokens = self._build_prefill_tokens()  # (B, T_max)
        logits = forward_batched(
            self._model, tokens, list(self._batch_cache)
        )  # (B, V)
        # 16c.2 step 4: initial-cohort forward_prompt_tokens accounting.
        # Counts effective prompt tokens (not padded B*T_max) so the S-5
        # acceptance assertion reads in user-visible units.
        self.forward_prompt_tokens += sum(
            len(r.prompt_ids) for r in self._rows
        )
        return self._sample_and_emit_batched(logits, is_prefill=True)

    def _decode_phase(self) -> list[BatchEvent]:
        """One batched forward at ``T=1`` over all rows.

        Terminal rows are fed ``pad_token_id`` as a placeholder so the
        batch axis stays fixed (16d's filter path is what actually
        reclaims terminal slots). Their output logits are discarded.
        """
        assert self._batch_cache is not None
        tokens = self._build_decode_tokens()  # (B, 1)
        logits = forward_batched(
            self._model, tokens, list(self._batch_cache)
        )  # (B, V)
        return self._sample_and_emit_batched(logits, is_prefill=False)

    def _sample_and_emit_batched(
        self,
        batched_logits: mx.array,
        *,
        is_prefill: bool,
    ) -> list[BatchEvent]:
        """Sample / emit / transition over ``self._rows`` (Phase 3 forward).

        Wraps :py:meth:`_sample_and_emit_rows` with the incumbent row
        list. Phase 2 admit uses the underlying helper directly on the
        newly-admitted subset.
        """
        return self._sample_and_emit_rows(
            self._rows, batched_logits, is_prefill=is_prefill
        )

    def _sample_and_emit_rows(
        self,
        rows: list[_BatchRow],
        batched_logits: mx.array,
        *,
        is_prefill: bool,
    ) -> list[BatchEvent]:
        """Per-row sample / emit / transition over an explicit row list.

        Terminal rows are skipped — they occupied a slot in the forward
        (to keep ``_idx`` in lockstep) but contribute no token event.
        Live rows sample, append to ``row.generated``, and transition
        according to stop / max_tokens rules.
        """
        events: list[BatchEvent] = []
        for i, row in enumerate(rows):
            if row.state.is_terminal:
                continue
            row_logits = batched_logits[i]  # (V,)
            history_ids = list(row.prompt_ids) + list(row.generated)
            history = mx.array(history_ids, dtype=mx.int32)
            token_scalar = self._sampler.sample(
                row_logits, history, row.params
            )
            tok = int(token_scalar.item())
            row.generated.append(tok)
            events.append(BatchEvent.token(row.req_index, tok))

            if tok in row.params.stop_token_ids:
                row.state.transition(RequestStatus.DONE, reason="stop_token")
                events.append(BatchEvent.done(row.req_index, "stop_token"))
            elif len(row.generated) >= row.params.max_tokens:
                row.state.transition(RequestStatus.DONE, reason="max_tokens")
                events.append(BatchEvent.done(row.req_index, "max_tokens"))
            elif is_prefill:
                row.state.transition(
                    RequestStatus.DECODE, reason="prefill_done"
                )
            # Plain DECODE continuation: no transition; stays DECODE.
        return events

    def _admit_waiting_requests(self) -> list[BatchEvent]:
        """Phase 2 — drain up to ``max_batch_size − len(self._rows)``
        requests from ``_waiting_queue``.

        Two phases (prep doc docs/P2_UNIT_16D_PREP.md §4.2):

          **Phase A — decide + apply (per-pending, pop-one-at-a-time).**
          Per pending popped from the queue, consult
          ``self._budgeter.admit`` (when configured) and route:

            - ``AdmitDecision``           → ``apply_admit`` immediately,
              then push onto the accepted list. Committing the
              reservation BEFORE the next iteration's ``admit()`` runs
              is invariant B-8 — otherwise a batch of small pendings
              each sees the same initial headroom and over-admits.
            - ``RejectDecision``          → emit ``BatchEvent.aborted``,
              bump ``self.aborts``, do NOT ``apply_admit`` (Q-4 / B-6';
              aborted admissions never touch the reservation tally).
            - ``AdmitAfterEvictDecision`` → evict first, then
              ``apply_admit``. Evict underrun aborts this admission
              without reserving it (B-2).
            - ``AdmitAfterPreemptDecision`` → fresh pendings may
              preempt one active row, requeue that victim as replay,
              then ``apply_admit`` the triggering pending. Replay
              pendings never preempt again (B-9).

          When ``self._budgeter is None``, every popped pending is
          accepted without a decision — behaviour is bit-identical to
          16c.2 (invariant B-1).

          **Phase B — execute (grouped by hit/miss).**
          16c.2's hit/miss split runs over the accepted list:

            - **hit rows** (per-row): non-zero block-aligned prefix
              match in ``self._prefix_cache`` → ``_admit_single_hit_row``.
            - **miss rows** (batched): the rest → ``_admit_miss_cohort``
              (single batched prefill over K miss rows).

          Events emit in phase-grouped order (hit rows first, then miss
          cohort) — same convention as 16c.2 (S-7). Aborted events from
          Phase A prepend this stream.

          **Release-on-raise (B-6' (c)).** If ``_admit_single_hit_row``
          or ``_admit_miss_cohort`` raises after their pendings had
          ``apply_admit`` called, the ``except`` block releases only the
          reservation of the rows that did not commit to ``self._rows``.
          Successfully-committed rows in the same step keep their
          reservations (they ARE real admitted requests and will be
          released on their own terminal path via ``_reclaim_terminated``).

        Returns the emitted events. An empty list means no admission or
        rejection fired this step; a non-empty list can include any mix
        of aborted events (Phase A) and token/done events (Phase B) —
        the caller's step() early-returns on non-empty, so the incumbent
        decode forward skips this step.
        """
        capacity = self._max_batch_size - len(self._rows)
        if capacity <= 0 or not self._waiting_queue:
            return []

        events: list[BatchEvent] = []
        accepted: list[_PendingAdmit] = []

        # Phase A — decide + apply.
        while len(accepted) < capacity and self._waiting_queue:
            pending = self._waiting_queue.popleft()

            if self._budgeter is None:
                accepted.append(pending)
                continue

            req_id = f"req-{pending.req_index}"
            decision = self._budgeter.admit(
                req_id=req_id,
                n_prompt=len(pending.prompt_ids),
                max_tokens=pending.params.max_tokens,
            )
            match decision:
                case AdmitDecision():
                    # B-8: commit reservation BEFORE the next iteration's
                    # admit() call observes reserved_bytes.
                    self._budgeter.apply_admit(
                        req_id, decision.reserved_delta
                    )
                    accepted.append(pending)
                case RejectDecision():
                    events.append(
                        BatchEvent.aborted(
                            pending.req_index, decision.reason
                        )
                    )
                    self.aborts += 1
                case AdmitAfterEvictDecision():
                    try:
                        self._apply_evict(decision.n_blocks)
                    except _BudgetEvictUnderrun:
                        # Policy assumed more evictable blocks than the
                        # tree could supply (live-hit churn between peek
                        # and apply, or a disagreement between budgeter's
                        # and batcher's prefix-cache views). Surface as
                        # an aborted admission — the batcher's row /
                        # cache / budgeter state is untouched because
                        # apply_admit has NOT yet been called for this
                        # pending (B-2).
                        events.append(
                            BatchEvent.aborted(
                                pending.req_index, "budget-exhausted"
                            )
                        )
                        self.aborts += 1
                        continue
                    self.evictions += 1
                    self._budgeter.apply_admit(
                        req_id, decision.reserved_delta
                    )
                    accepted.append(pending)
                case AdmitAfterPreemptDecision():
                    if pending.is_replay:
                        # B-9 anti-ping-pong: a replay admission CANNOT
                        # trigger further preempts. Two sub-branches:
                        if self._rows:
                            # Active rows exist — wait for one to
                            # terminate naturally, then retry this
                            # pending. Only the current pending is
                            # requeued; the untouched suffix of the
                            # waiting queue is preserved by the
                            # pop-one-at-a-time loop shape.
                            self._waiting_queue.appendleft(pending)
                            break
                        # No active rows to wait on — the replay is
                        # genuinely unfittable under this cap.
                        events.append(
                            BatchEvent.aborted(
                                pending.req_index, "budget-exhausted"
                            )
                        )
                        self.aborts += 1
                        continue
                    # Fresh (non-replay) pending may trigger preempt.
                    if not self._apply_preempt(decision.preempt_req_id):
                        # B-7: victim was not in self._rows (race with
                        # reclaim, same-step termination, etc.). Requeue
                        # this triggering admission at queue front and
                        # break so next step retries with fresh state.
                        self._waiting_queue.appendleft(pending)
                        break
                    self.preempts += 1
                    self._budgeter.apply_admit(
                        req_id, decision.reserved_delta
                    )
                    accepted.append(pending)
                case _:
                    raise AssertionError(
                        f"unhandled AdmissionDecision type: "
                        f"{type(decision).__name__}"
                    )

        if not accepted:
            # Every pending rejected (events non-empty) or the queue was
            # empty after capacity check (events also empty). Either way
            # Phase B has nothing to execute; return what Phase A emitted.
            return events

        # Phase B — execute. Split accepted into hit / miss. With no
        # prefix_cache installed, every admission routes to the miss path.
        hit_rows: list[tuple[_PendingAdmit, int]] = []
        miss_rows: list[_PendingAdmit] = []
        if self._prefix_cache is None:
            miss_rows = list(accepted)
        else:
            block_size = self._prefix_cache.block_size
            for pending in accepted:
                raw = self._prefix_cache.peek(pending.prompt_ids)
                # S-5 edge 1: reserve at least one token for suffix
                # prefill so first-token logits are available.
                if len(pending.prompt_ids) <= 1:
                    max_aligned = 0
                else:
                    max_aligned = (
                        (len(pending.prompt_ids) - 1) // block_size
                    ) * block_size
                usable = min(raw.num_hit_tokens, max_aligned)
                if usable == 0:
                    miss_rows.append(pending)
                else:
                    hit_rows.append((pending, usable))

        # Phase B.1 — hit rows, per-row seeded admission.
        for pending, usable in hit_rows:
            try:
                events.extend(
                    self._admit_single_hit_row(pending, usable)
                )
            except Exception:
                # B-6' (c): release THIS row's reservation. Prior hit
                # rows already committed stay reserved; tail rows (not
                # yet executed) also leak reservations, which is
                # acceptable because a body-raise means the batcher is
                # no longer in a coherent reusable state.
                if self._budgeter is not None:
                    self._budgeter.release(f"req-{pending.req_index}")
                raise
        # Phase B.2 — miss cohort, one batched prefill over K miss rows.
        if miss_rows:
            try:
                events.extend(self._admit_miss_cohort(miss_rows))
            except Exception:
                # Miss cohort is atomic (one forward for all K rows);
                # on raise, release every miss row's reservation together.
                if self._budgeter is not None:
                    for pending in miss_rows:
                        self._budgeter.release(
                            f"req-{pending.req_index}"
                        )
                raise

        # One slot_table rebuild suffices after both phases' mutations.
        self._rebuild_slot_table()
        return events

    def _admit_single_hit_row(
        self, pending: _PendingAdmit, usable_hit_tokens: int
    ) -> list[BatchEvent]:
        """Per-row admission via prefix-cache hit (16c.2 step 4 sub-commit 3).

        Sequence (docs/P2_UNIT_16C_2_STEP_4_SKELETON.md §3.2):

          1. lookup retains hits in the store.
          2. try: assert S-1, fetch detached K/V, build seeded
             per-layer BatchKVCache list, suffix-prefill forward,
             sample + emit, extend into main, append row, counters.
          3. finally: release retained hits. Release lives in
             ``finally`` so an exception anywhere in step 2 (shape
             error, forward failure, sampler bug) still returns the
             hit refs to the store — leaking them would make those
             blocks permanently non-evictable (S-2).

        Placing the assertion INSIDE the try block is load-bearing.
        If it ran between lookup and try, a failing assertion would
        skip the finally and strand the hits that lookup just retained.
        """
        assert self._prefix_cache is not None

        # Construct the row state before the retaining lookup so the
        # try-finally window can be as tight as possible.
        request = Request(
            prompt="",
            sampling_params=pending.params,
            request_id=f"req-{pending.req_index}",
            token_ids=pending.prompt_ids,
        )
        state = RequestState(request=request)
        state.transition(RequestStatus.PREFILL, reason="admit-mid-run-hit")
        row = _BatchRow(
            req_index=pending.req_index,
            req_id=f"req-{pending.req_index}",
            prompt_ids=list(pending.prompt_ids),
            params=pending.params,
            state=state,
        )

        hit = self._prefix_cache.lookup(
            pending.prompt_ids[:usable_hit_tokens]
        )
        try:
            # S-1: the walk inside lookup must agree with what peek
            # measured above. Divergence would mean the radix tree
            # mutated between peek and lookup, which should be
            # impossible inside a single step().
            assert hit.num_hit_tokens == usable_hit_tokens, (
                f"lookup hit_tokens {hit.num_hit_tokens} != "
                f"peek-sized {usable_hit_tokens} — radix tree drift"
            )

            detached = self._prefix_cache.fetch_detached_blocks(
                list(hit.block_ids)
            )
            num_layers = self._adapter.config.num_layers
            row_cache = build_seeded_batch_kv(
                detached, num_layers=num_layers
            )

            suffix_tokens = list(pending.prompt_ids[usable_hit_tokens:])
            # S-5 edge 1 guarantees suffix_tokens is non-empty — the
            # max_aligned formula above reserves at least one token.
            suffix_arr = mx.array([suffix_tokens], dtype=mx.int32)
            logits = forward_batched(
                self._model, suffix_arr, list(row_cache)
            )  # (1, V)
            events = self._sample_and_emit_rows(
                [row], logits, is_prefill=True
            )

            # Stitch row_cache into main. Both branches preserve
            # invariant I-2 (incumbent rows keep their indices).
            if self._batch_cache is None:
                self._batch_cache = row_cache
            else:
                for layer in range(num_layers):
                    self._batch_cache[layer].extend(row_cache[layer])

            self._rows.append(row)
            self.prefix_hits += 1
            self.forward_prompt_tokens += len(suffix_tokens)
            return events
        finally:
            self._prefix_cache.release(list(hit.block_ids))

    def _admit_miss_cohort(
        self, admitted_pending: list[_PendingAdmit]
    ) -> list[BatchEvent]:
        """Cohort-prefill admission path for rows with no prefix hit.

        Renamed from the pre-step-4 body of ``_admit_waiting_requests``.
        Behaviour is unchanged: build fresh B=K BatchKVCache, run one
        batched prefill over the left-padded admit subset, sample + emit,
        extend into main. Counter bump goes at the end so it does not
        double-count if a hit row ran earlier in the same step.

        Does NOT rebuild slot_table — caller does that once per step
        after both hit and miss phases.
        """
        admitted: list[_BatchRow] = []
        for pending in admitted_pending:
            request = Request(
                prompt="",
                sampling_params=pending.params,
                request_id=f"req-{pending.req_index}",
                token_ids=pending.prompt_ids,
            )
            state = RequestState(request=request)
            state.transition(RequestStatus.PREFILL, reason="admit-mid-run")
            admitted.append(
                _BatchRow(
                    req_index=pending.req_index,
                    req_id=f"req-{pending.req_index}",
                    prompt_ids=list(pending.prompt_ids),
                    params=pending.params,
                    state=state,
                )
            )

        num_layers = self._adapter.config.num_layers
        k_prompt_lens = [len(r.prompt_ids) for r in admitted]
        k_max_len = max(k_prompt_lens)
        k_left_padding = [k_max_len - n for n in k_prompt_lens]
        # P-3-C3b: mid-run admission now routes through the adapter's
        # cache factory alongside the initial-cohort path (C3a wired
        # that into ``_prepare_cohort``). For plain Qwen3 the fallback
        # returns the same all-``BatchKVCache`` shape as the pre-C3b
        # hardcoded list, so behaviour is bit-identical for currently
        # batchable adapters; for hybrid adapters (once C3c lifts the
        # gate) the factory produces the matching ``ArraysCache`` /
        # ``BatchKVCache`` interleaving.
        k_batch_cache = _make_batch_cache(self._adapter, k_left_padding)

        pad = self._pad_token_id()
        rows_2d: list[list[int]] = []
        for r in admitted:
            n_pad = k_max_len - len(r.prompt_ids)
            rows_2d.append([pad] * n_pad + list(r.prompt_ids))
        tokens = mx.array(rows_2d, dtype=mx.int32)

        logits = forward_batched(self._model, tokens, list(k_batch_cache))
        events = self._sample_and_emit_rows(
            admitted, logits, is_prefill=True
        )

        if self._batch_cache is None:
            self._batch_cache = k_batch_cache
        else:
            for layer in range(num_layers):
                self._batch_cache[layer].extend(k_batch_cache[layer])

        self._rows.extend(admitted)
        # Counter bump AFTER the forward. Effective prompt tokens
        # (ignores pad) — user-visible units, matches S-5 formula.
        self.forward_prompt_tokens += sum(k_prompt_lens)
        return events

    def _apply_evict(self, n_blocks: int) -> None:
        """Evict ``n_blocks`` LRU leaves from the prefix cache (16d-3).

        Called by Phase A when the budgeter returns
        ``AdmitAfterEvictDecision``. Raises ``_BudgetEvictUnderrun`` if
        ``evict_until`` freed fewer blocks than asked — the caller
        converts that to an aborted admission (B-2: underrun never
        corrupts state).

        The assert on ``self._prefix_cache is not None`` is intentional:
        the budgeter can only return this decision when it observed
        evictable blocks, which requires a prefix cache. If the batcher
        and budgeter somehow disagree on cache presence, a decision got
        produced against state the batcher cannot apply — fail loudly
        rather than silently skip the eviction.
        """
        assert self._prefix_cache is not None, (
            "AdmitAfterEvictDecision reached the batcher without a "
            "prefix_cache — budgeter and batcher disagree on cache "
            "presence"
        )
        freed = self._prefix_cache.evict_until(n_blocks)
        if freed < n_blocks:
            raise _BudgetEvictUnderrun(
                f"evict_until freed {freed} of {n_blocks} blocks"
            )

    # --- tensor construction helpers ---

    def _build_prefill_tokens(self) -> mx.array:
        """Construct the left-padded ``(B, T_max)`` prompt tensor."""
        max_len = max(len(r.prompt_ids) for r in self._rows)
        pad = self._pad_token_id()
        rows_2d: list[list[int]] = []
        for row in self._rows:
            pad_amt = max_len - len(row.prompt_ids)
            rows_2d.append([pad] * pad_amt + list(row.prompt_ids))
        return mx.array(rows_2d, dtype=mx.int32)

    def _build_decode_tokens(self) -> mx.array:
        """Construct the ``(B, 1)`` next-step token tensor.

        Live rows feed their last generated token; terminal rows feed
        ``_pad_token_id`` to keep the batch axis fixed without
        polluting ``row.generated``.
        """
        pad = self._pad_token_id()
        rows_2d: list[list[int]] = []
        for row in self._rows:
            if row.state.is_terminal:
                rows_2d.append([pad])
            else:
                rows_2d.append([row.generated[-1]])
        return mx.array(rows_2d, dtype=mx.int32)

    def _pad_token_id(self) -> int:
        """Placeholder token id for left-padded and terminal-row slots.

        Hardcoded to 0 in 16b — the left-padded positions are masked
        out by the attention mask (BatchKVCache's left_padding metadata
        drives the mask) and terminal rows' logits are discarded, so
        the numerical effect is unobservable. Encapsulated as a method
        so a future ``Tokenizer`` Protocol extension that exposes
        ``pad_token_id`` can be wired in at one call site.
        """
        return 0

    # --- capability gate ---

    @staticmethod
    def _enforce_capability_gate(adapter: ModelAdapter) -> None:
        """Reject adapters whose capabilities the batcher cannot schedule.

        Layer 3 of the v2.3 three-layer stack. The batcher asks the
        adapter for its ``ModelCapabilities`` (D-016) and decides from
        that typed summary. ``AttentionPattern`` remains authoritative
        for per-layer detail — used here only to locate the offending
        layer index for the error message.

        As of P-3-D3 the accepted set is ``GLOBAL`` (plain attention,
        P-2), ``HYBRID_DELTANET`` (Qwen3.5 hybrid family, P-3-C3c), and
        ``SLIDING`` (Gemma4 sliding-window family, P-3-D3 — restricted
        to the ``prefix_cache=None`` path by a separate constructor
        guard). ``RECURRENT`` (pure-Mamba), ``HYBRID`` (single-kind
        sliding/global hybrid), and MoE routing still need their own
        scheduler plumbing before being unlocked here.
        """
        caps = adapter.capabilities()
        if (
            caps.attention_kinds.issubset(_SUPPORTED_ATTENTION_KINDS)
            and not caps.has_moe
        ):
            return
        # Locate a concrete offending layer for the error message. We
        # walk ``attention_pattern`` and skip every kind in the
        # supported set — crucial after C3c, because an adapter whose
        # kinds include ``{GLOBAL, HYBRID_DELTANET, SLIDING, RECURRENT}``
        # must report the ``RECURRENT`` layer rather than any of the
        # three (now supported) kinds.
        pattern = adapter.attention_pattern()
        for layer_idx, kind in enumerate(pattern.per_layer):
            if kind in _SUPPORTED_ATTENTION_KINDS:
                continue
            reason = _unsupported_kind_reason(kind)
            extra = (
                "; adapter also declares MoE routing (has_moe=True) — "
                "MoE adapters (e.g. qwen3_5_moe, gemma4_moe) are "
                "single-request-only until P-3-E4 lands batched MoE "
                "smoke + parity"
                if caps.has_moe
                else ""
            )
            raise NotImplementedError(
                f"ContinuousBatcher accepts AttentionKind.GLOBAL, "
                f"AttentionKind.HYBRID_DELTANET, and AttentionKind.SLIDING "
                f"only; adapter capabilities include {kind.value!r} "
                f"(layer {layer_idx}, has_recurrent_state="
                f"{caps.has_recurrent_state}) — {reason}{extra}"
            )
        # Fallback: the capability predicate rejected the adapter but
        # the attention_pattern walk found no layer outside
        # ``_SUPPORTED_ATTENTION_KINDS``. The normal path into here
        # is ``has_moe=True`` with an otherwise supported
        # ``attention_kinds`` set (e.g. pure ``{GLOBAL}``, pure
        # ``{HYBRID_DELTANET}``, or a mix of the two); a pathological
        # adapter whose ``attention_kinds`` disagree with its
        # ``attention_pattern`` would also fall here. Emit a generic
        # message that names the capabilities values explicitly, and
        # append MoE context only when ``has_moe`` is actually set —
        # no hard-coded assumption.
        moe_tail = (
            " — MoE adapters (Qwen3.5-35B-A3B, gemma-4-26B-A4B) are "
            "registered as of P-3-E1 for single-request execution "
            "only; batched MoE scheduling is P-3-E4"
            if caps.has_moe
            else (
                "; attention_pattern() reports no unsupported layer, so "
                "ModelCapabilities and AttentionPattern disagree — "
                "re-run capabilities_from_attention_pattern on the "
                "adapter's pattern to regenerate a consistent summary"
            )
        )
        kinds_display = sorted(k.value for k in caps.attention_kinds)
        raise NotImplementedError(
            f"ContinuousBatcher accepts AttentionKind.GLOBAL, "
            f"AttentionKind.HYBRID_DELTANET, and AttentionKind.SLIDING "
            f"only; adapter capabilities are unsupported: "
            f"attention_kinds={kinds_display!r}, "
            f"has_recurrent_state={caps.has_recurrent_state}, "
            f"has_moe={caps.has_moe}{moe_tail}"
        )


_SUPPORTED_ATTENTION_KINDS: frozenset[AttentionKind] = frozenset(
    {
        AttentionKind.GLOBAL,
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.SLIDING,
    }
)


def _unsupported_kind_reason(kind: AttentionKind) -> str:
    """Short explanation used by the capability-gate error message.

    HYBRID_DELTANET was dropped from this map in P-3-C3c; SLIDING was
    dropped in P-3-D3. The remaining kinds still lack scheduler support.
    """
    if kind == AttentionKind.RECURRENT:
        return (
            "pure-recurrent batching needs adapter-owned recurrent "
            "state on par with DeltaNet hybrid; add a scheduler path "
            "analogous to the HYBRID_DELTANET one from P-3-C3a/b/c"
        )
    if kind == AttentionKind.HYBRID:
        return (
            "the single-kind HYBRID enum (sliding/global combined in one "
            "AttentionKind) has no adapter on the v0.1 roadmap; real "
            "hybrids like Gemma4 emit per-layer SLIDING + GLOBAL"
        )
    return "unknown AttentionKind; no scheduler mapping defined"


def _make_batch_cache(
    adapter: ModelAdapter, left_padding: list[int]
) -> list[Any]:
    """Produce the per-layer batched cache list for ``_prepare_cohort``.

    Prefers ``adapter.make_batch_cache(left_padding)`` (P-3-C3a) so
    family adapters can inject their own per-layer cache type — hybrid
    Qwen3.5 interleaves ``ArraysCache`` for DeltaNet layers with
    ``BatchKVCache`` for global attention, whereas plain Qwen3 keeps a
    homogeneous ``BatchKVCache`` list. The method is deliberately NOT
    on the I-1 Protocol (see D-016 discussion for the "keep Protocol
    lean" rule), so callers use ``getattr`` and fall back to the
    pre-C3a hardcoded all-``BatchKVCache`` shape when the adapter does
    not implement it. The fallback preserves bit-identical behaviour
    for any adapter that passed the capability gate before C3a.
    """
    maker = getattr(adapter, "make_batch_cache", None)
    if callable(maker):
        caches: list[Any] = maker(left_padding)
        return caches
    # ``maker`` may be absent (no attribute) or explicitly set to
    # ``None`` / a non-callable sentinel; all three route to fallback.
    return [
        BatchKVCache(left_padding=left_padding)
        for _ in range(adapter.config.num_layers)
    ]
