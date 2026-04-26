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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

from silica.core.events import BatchEvent
from silica.core.request import Request, RequestState, RequestStatus
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.mlx.runner import forward_batched
from silica.models.adapter import AttentionKind, ModelAdapter
from silica.models.pre_norm_capture import PreNormCaptureAdapter
from silica.models.recurrent import RecurrentSnapshot, RecurrentStateAdapter
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

    ``recurrent_snapshot`` (P-3-C5.2): the adapter-captured
    ``RecurrentSnapshot`` for this row at preempt time, or ``None``
    when (a) the adapter does not implement ``RecurrentStateAdapter``
    (no recurrent state to capture, e.g. plain Qwen3 / Gemma4), or
    (b) this pending was created by ``add_request`` rather than by
    preempt (no preempt history to snapshot from).

    **Boundary semantics**: when set, the snapshot describes the
    recurrent state at ``T + len(generated) - 1`` tokens consumed,
    where ``T = len(prompt_ids)`` of the original (pre-preempt)
    request and ``len(generated)`` is the number of tokens emitted
    before preempt. This is the cache's natural state at preempt:
    ``T`` prompt tokens plus the first ``len(generated) - 1``
    decode-step inputs have been fed back; the latest sample (which
    became ``generated[-1]``) is observed but not yet consumed.

    **C5.2 stashes; C5.3 restores.** ``_admit_miss_cohort`` does
    NOT call ``adapter.restore_recurrent_state`` on the full-replay
    path because ``composite_prompt`` is ``prompt + generated`` (i.e.
    ``T + len(generated)`` tokens) and the post-prefill cache lands
    at ``T + len(generated)`` consumed — one token past the snapshot
    boundary. See ``docs/P3_C5_DRIFT_EXPERIMENT/README.md``
    "C5.2 acceptance" for the boundary derivation. C5.3 will enable
    restore only on admission paths whose post-prefill boundary
    matches the snapshot boundary (e.g. prefix-hit replay where the
    prefill ends at a block-aligned edge the snapshot was captured
    against). Until then, ``recurrent_snapshot`` is a captured
    record carried for future use, not actively consumed.
    """

    req_index: int
    prompt_ids: tuple[int, ...]
    params: SamplingParams
    is_replay: bool = False
    recurrent_snapshot: RecurrentSnapshot | None = None


@dataclass(frozen=True)
class _PreemptedRow:
    """Result of ``_preempt_active_row`` (P-3-C5.2).

    Carries the detached ``_BatchRow`` alongside the optional
    recurrent snapshot captured **before** the cache-filter pass. The
    snapshot is set when ``isinstance(adapter, RecurrentStateAdapter)``
    holds; otherwise ``None`` and the replay proceeds via batched
    re-prefill (today's behaviour for non-recurrent adapters).

    Returning a small dataclass instead of stuffing a side-channel
    into ``_BatchRow`` keeps the snapshot capture point inside
    ``_preempt_active_row`` (where the live ``victim_row_idx`` is
    valid against ``self._batch_cache``) and lets ``_apply_preempt``
    pass the snapshot into ``_PendingAdmit`` cleanly.
    """

    detached: _BatchRow
    recurrent_snapshot: RecurrentSnapshot | None = None


@dataclass
class _BatchRow:
    """Per-row state held by the batcher (internal).

    One row = one request. The row persists from admission through
    DECODE / DONE / ABORTED / PREEMPTED. ``generated`` only stores
    tokens that were sampled and emitted — placeholder feeds used to
    keep ``_idx`` in lockstep after the row terminates do NOT append
    here (so ``len(generated)`` keeps a clean "how many tokens did this
    row actually produce" reading at any point).

    ``recurrent_snapshots_per_block`` / ``absolute_consumed_tokens``
    (P-3-C5.3.1): per-row slice-regime accounting for hybrid adapters
    running with a ``RadixPrefixCache``. The dict maps absolute block
    index → ``RecurrentSnapshot`` captured at that block boundary by
    the slice-prefill helper or the decode-step capture path.
    Absolute index uses ``block_size``-aligned token counts (block 0
    covers tokens ``[0, block_size)``; block ``i`` covers
    ``[i * block_size, (i + 1) * block_size)``). Empty for non-slice-
    regime rows (B>1 cohort admission, non-recurrent adapters,
    ``prefix_cache=None``). ``absolute_consumed_tokens`` runs in
    lockstep with the live cache: increments by chunk size after each
    slice-prefill forward, +1 after each decode-step that consumed a
    real token (not pad). Stays ``0`` for rows admitted via contiguous
    prefill — the decode-step capture path uses a ``> 0`` precondition
    to skip those rows so cross-regime snapshots never reach the radix
    tree (P-3-C5.3.1 finding C, see ``docs/P3_C5_3_DESIGN.md`` §4.2).
    """

    req_index: int
    req_id: str
    prompt_ids: list[int]
    params: SamplingParams
    state: RequestState
    generated: list[int] = field(default_factory=list)
    recurrent_snapshots_per_block: dict[int, RecurrentSnapshot] = field(
        default_factory=dict
    )
    absolute_consumed_tokens: int = 0
    # P-5-F F.2b: per-row pre-norm K capture for the (3b) prefix-cache.
    # Populated only when ``prefix_cache.store.pre_norm`` is True. Outer
    # key is absolute block index (block 0 covers tokens
    # ``[0, block_size)``); inner list is indexed by attention-layer
    # position (the same ``attn_layer_indices`` order
    # ``_extract_and_insert_prefix`` uses), each entry shape
    # ``(1, n_kv_heads, block_size, head_dim)`` carrying ``k_proj(x)``
    # output (pre-k_norm K). At ``_extract_and_insert_prefix`` time the
    # K side of every block-aligned payload comes from this dict
    # instead of slicing the live cache; V continues to come from the
    # live cache as before. Empty for legacy ``pre_norm=False`` rows
    # so the memory cost is bounded to active (3b) deployments.
    k_pre_per_block: dict[int, list[mx.array]] = field(default_factory=dict)


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
        # P-3-D3 sliding-window guard. ``build_seeded_batch_kv`` /
        # ``_extract_and_insert_prefix`` emit ``BatchKVCache`` per layer;
        # adapting that to ``BatchRotatingKVCache``'s window truncation,
        # ``offset`` arithmetic, and ``rotated`` state has not been
        # validated. D3 only commits to the ``prefix_cache=None`` path
        # for SLIDING-bearing adapters; future work widens the seed
        # path rather than bypassing this guard.
        #
        # The earlier C3b recurrent-state guard (rejecting hybrid +
        # ``RadixPrefixCache``) was removed at P-3-C5.4 once the
        # C5.3 chain wired the slice-regime trajectory + heterogeneous
        # row-cache assembly + Phase-B classifier through end-to-end.
        if prefix_cache is not None:
            caps = adapter.capabilities()
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
            # P-5-F F.2a: pre-norm store contract. When the store
            # declares ``pre_norm=True`` the K side of every detached
            # payload is pre-k_norm K (the output of ``attn.k_proj``);
            # the admit path must call
            # ``adapter.apply_k_norm_then_rope`` per block per attn
            # position before seeding the live cache. The adapter
            # therefore must implement ``PreNormCaptureAdapter``. The
            # check uses ``getattr`` with a default so legacy
            # ``PagedPrefixBlockStore``-shaped stores (no ``pre_norm``
            # attribute) keep behaving as the post-RoPE path.
            store_is_pre_norm = bool(
                getattr(prefix_cache.store, "pre_norm", False)
            )
            if store_is_pre_norm and not isinstance(
                adapter, PreNormCaptureAdapter
            ):
                raise TypeError(
                    "ContinuousBatcher: prefix_cache.store.pre_norm is "
                    "True but the adapter does not implement "
                    "silica.models.pre_norm_capture.PreNormCaptureAdapter "
                    f"(adapter type: {type(adapter).__name__}). The "
                    "pre-norm store path requires the adapter to expose "
                    "install_pre_norm_capture / apply_k_norm_then_rope "
                    "so the batcher can capture K_pre during prefill "
                    "and reconstruct post-RoPE K on hit-path admit. "
                    "Either pass an adapter that ships P-5-F F.1 or "
                    "construct the store with pre_norm=False (legacy "
                    "post-RoPE path)."
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

    def _token_kv_layer_indices(
        self, cache_list: list[Any] | None = None
    ) -> list[int]:
        """Return source-transformer-layer indices whose layer cache is
        a token-granular K/V store (``BatchKVCache``).

        For pure-attention adapters (Qwen3, etc.), every layer is a
        ``BatchKVCache`` so this returns ``list(range(num_layers))``.
        For hybrid adapters (Qwen3.5 DeltaNet+attention), this returns
        only the attention-layer source indices; DeltaNet layers carry
        ``ArraysCache`` whose state is captured via the recurrent
        snapshot path, NOT the token-K/V slice path.

        ``cache_list`` argument (P-3-C5.3.3-het.2): when provided, the
        helper introspects that list instead of ``self._batch_cache``.
        The admit-side seed-assembly path uses this to derive attention
        positions from a freshly-built empty heterogeneous row cache
        before any forward has populated layers; passing it explicitly
        avoids depending on the main batch cache (which may be ``None``
        when the hit row arrives in an empty batcher). When omitted, the
        helper preserves the original extract-side behaviour and asserts
        ``self._batch_cache`` is set.

        Centralizes the cache-shape introspection so callers
        (``_extract_and_insert_prefix``, ``_admit_single_hit_row``)
        don't sprinkle ``isinstance`` / ``hasattr`` checks across the
        prefix-cache plumbing. Loud-fail on a pure-recurrent stack
        (zero ``BatchKVCache`` layers) — the C5.3 prefix-cache extract
        path has no work to do without token-K/V layers, and silently
        skipping would mask a misconfiguration.

        P-3-C5.3.3-het.1 (introduced); P-3-C5.3.3-het.2 (cache_list arg).
        """
        if cache_list is None:
            assert self._batch_cache is not None
            target = self._batch_cache
        else:
            target = cache_list
        indices = [
            i
            for i, layer in enumerate(target)
            if isinstance(layer, BatchKVCache)
        ]
        if not indices:
            raise RuntimeError(
                "no token-K/V layers found in batch cache; "
                "pure-recurrent stacks are not supported by the "
                "C5.3 prefix-cache extract path"
            )
        return indices

    def _pre_norm_capture_active(self) -> bool:
        """Whether the (3b) pre-norm K capture path is engaged.

        True iff the active ``RadixPrefixCache`` was constructed with a
        ``pre_norm=True`` store (P-5-F F.2a contract). Pure post-RoPE
        deployments and ``prefix_cache=None`` paths return False; the
        F.2a constructor gate has already validated the adapter
        implements ``PreNormCaptureAdapter`` when this is True.
        """
        if self._prefix_cache is None:
            return False
        return bool(getattr(self._prefix_cache.store, "pre_norm", False))

    def _capture_adapter(self) -> PreNormCaptureAdapter:
        """Typed accessor for the F.2b protocol surface.

        The F.2a constructor gate guarantees the adapter implements
        ``PreNormCaptureAdapter`` whenever ``_pre_norm_capture_active``
        is True; this helper narrows the static type so the call sites
        do not each repeat the ``isinstance`` assertion.
        """
        return cast(PreNormCaptureAdapter, self._adapter)

    def _split_capture_into_row_kpre(
        self,
        buffer: dict[int, mx.array],
        rows: list[_BatchRow],
        *,
        pad_per_row: list[int],
        pre_consumed_per_row: list[int],
        real_in_chunk_per_row: list[int],
        block_size: int,
    ) -> None:
        """Slice block-aligned K_pre per row out of a chunk's capture
        buffer and append to each row's ``k_pre_per_block`` dict.

        Buffer shape per ``attn_pos``: ``(B, chunk_len, n_kv_heads * head_dim)``.
        For row ``i``, real tokens in this chunk live at axis-1
        ``[pad_per_row[i], pad_per_row[i] + real_in_chunk_per_row[i])``
        and cover absolute positions
        ``[pre_consumed_per_row[i], pre_consumed_per_row[i] + real_in_chunk_per_row[i])``.

        Each block ``b`` whose absolute range
        ``[b*block_size, (b+1)*block_size)`` is fully contained in the
        row's real-token range gets its K_pre slice written to
        ``row.k_pre_per_block[b]``. Slices are reshaped + transposed to
        the standard per-head ``(1, n_kv_heads, block_size, head_dim)``
        layout the codec / seeded-cache pipelines consume, then forced
        to materialised storage so the next chunk forward overwriting
        the buffer cannot reach back into them.

        The ``b_idx`` keying matches the absolute-block contract used by
        ``recurrent_snapshots_per_block`` and the radix tree's
        ancestor-chain walk in ``_admit_single_hit_row``.
        """
        layout = self._adapter.kv_layout()
        n_kv_heads = layout.n_kv_heads
        head_dim = layout.head_dim
        n_attn = len(buffer)
        for row_idx, row in enumerate(rows):
            pad_i = pad_per_row[row_idx]
            pre = pre_consumed_per_row[row_idx]
            real_in = real_in_chunk_per_row[row_idx]
            if real_in <= 0:
                continue
            # Range of complete blocks in [pre, pre + real_in).
            first_b = (pre + block_size - 1) // block_size  # ceil
            last_b_excl = (pre + real_in) // block_size  # floor
            for b_idx in range(first_b, last_b_excl):
                # Map absolute block bounds back to chunk axis-1 positions
                # for this row: real tokens for row i in this chunk start
                # at axis-1 pad_i and absolute pre. So absolute pos x maps
                # to axis-1 (x - pre) + pad_i.
                chunk_start = pad_i + (b_idx * block_size - pre)
                chunk_end = chunk_start + block_size
                per_attn: list[mx.array] = []
                for attn_pos in range(n_attn):
                    cap = buffer[attn_pos]
                    slc = cap[
                        row_idx : row_idx + 1, chunk_start:chunk_end, :
                    ]
                    k_pre = slc.reshape(
                        1, block_size, n_kv_heads, head_dim
                    ).transpose(0, 2, 1, 3)
                    k_pre = mx.contiguous(k_pre)
                    per_attn.append(k_pre)
                # Force materialisation — next chunk's forward will
                # rebind buffer entries.
                mx.eval(*per_attn)
                row.k_pre_per_block[b_idx] = per_attn

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

        # P-3-C5.3.3-het.1: hybrid adapters have heterogeneous batch
        # caches — DeltaNet layers carry ``ArraysCache`` (no
        # ``.offset`` / ``.keys``), full-attention layers carry
        # ``BatchKVCache``. The pre-het code at the C5.3.3a snapshot
        # tripped on ``self._batch_cache[0].offset`` because layer 0
        # of Qwen3.5 is a DeltaNet layer. ``_token_kv_layer_indices``
        # filters to the attention-only positions; offset / left_padding
        # come from the first one (all attention layers share the same
        # values, they evolve together under extend/filter).
        attn_layer_indices = self._token_kv_layer_indices()
        first_attn_cache = self._batch_cache[attn_layer_indices[0]]

        # S-3a: cache's own authoritative accounting must agree.
        computed_len = int(first_attn_cache.offset[row_idx].item())
        assert len(computed_ids) == computed_len, (
            f"row {row_idx}: computed_ids len {len(computed_ids)} != "
            f"offset[row_idx] {computed_len} — sampler/forward drift"
        )

        block_size = self._prefix_cache.block_size
        aligned_tokens = (computed_len // block_size) * block_size
        if aligned_tokens < block_size:
            return

        # S-3b: every BatchKVCache layer shares the same left_padding
        # vector, so the first attention layer is authoritative.
        base = int(first_attn_cache.left_padding[row_idx].item())

        tokens_prefix = computed_ids[:aligned_tokens]
        n_aligned_blocks = aligned_tokens // block_size
        # detached_blocks[b][i] is the i-th attention layer's K/V for
        # block b; i is the position in attn_layer_indices, NOT the
        # source transformer layer index. Hybrid callers (het.2 admit
        # path) recover the mapping by re-deriving attn_layer_indices
        # from the empty heterogeneous cache they assemble.
        #
        # P-5-F F.2b: when the prefix-cache store carries the pre-norm
        # contract (``store.pre_norm=True``), the K side comes from
        # ``row.k_pre_per_block`` (captured at every prefill chunk
        # forward by the proxy installed in
        # ``PreNormCaptureAdapter.install_pre_norm_capture``). V
        # continues to come from the live cache as before — V is
        # unchanged across families (no normalisation, no RoPE), so
        # the live cache's V slice is directly reusable. Legacy
        # ``pre_norm=False`` deployments stay on the live-cache K
        # slice, bit-identical to today.
        use_captured_k = self._pre_norm_capture_active()
        # Hit-admit rows enter ``_extract_and_insert_prefix`` with the
        # first ``K = absolute_consumed_tokens // block_size`` blocks
        # already in the prefix tree from the prior request that
        # registered them. Those blocks have no entry in
        # ``row.k_pre_per_block`` (this row never forwarded their
        # tokens through ``k_proj``); ``insert_detached``'s
        # duplicate-prefix branch will skip them. We still need a
        # type-correct K placeholder so the per-layer list shape is
        # uniform; ``mx.zeros`` of the captured shape stands in. The
        # placeholder is GC-eligible after ``insert_detached`` returns.
        layout = self._adapter.kv_layout()
        kpre_placeholder: mx.array | None = None
        if use_captured_k:
            kpre_placeholder = mx.zeros(
                (1, layout.n_kv_heads, block_size, layout.head_dim),
                dtype=layout.dtype,
            )
        detached_blocks: list[list[tuple[mx.array, mx.array]]] = []
        for b_idx in range(n_aligned_blocks):
            start = b_idx * block_size
            end = start + block_size
            per_layer: list[tuple[mx.array, mx.array]] = []
            captured_kpre: list[mx.array] | None = None
            if use_captured_k:
                captured_kpre = row.k_pre_per_block.get(b_idx)
                if (
                    captured_kpre is not None
                    and len(captured_kpre) != len(attn_layer_indices)
                ):
                    raise AssertionError(
                        f"row {row_idx} block {b_idx}: captured "
                        f"K_pre has {len(captured_kpre)} layers but "
                        f"{len(attn_layer_indices)} attention layers "
                        "expected — proxy install drift"
                    )
            for pos, layer_idx in enumerate(attn_layer_indices):
                layer_cache = self._batch_cache[layer_idx]
                values = layer_cache.values
                if values is None:
                    raise AssertionError(
                        f"row {row_idx} layer {layer_idx}: "
                        "cache values tensor is not initialised"
                    )
                v = mx.contiguous(
                    values[
                        row_idx : row_idx + 1, :, base + start : base + end, :
                    ]
                )
                if use_captured_k:
                    if captured_kpre is not None:
                        k = captured_kpre[pos]
                    else:
                        # Hit-admit duplicate block — insert_detached
                        # skips it; placeholder content is irrelevant.
                        assert kpre_placeholder is not None
                        k = kpre_placeholder
                else:
                    keys = layer_cache.keys
                    if keys is None:
                        raise AssertionError(
                            f"row {row_idx} layer {layer_idx}: "
                            "cache keys tensor is not initialised"
                        )
                    k = mx.contiguous(
                        keys[
                            row_idx : row_idx + 1,
                            :,
                            base + start : base + end,
                            :,
                        ]
                    )
                per_layer.append((k, v))
            detached_blocks.append(per_layer)

        # P-3-C5.3.2: forward per-block recurrent snapshots that the
        # slice-prefill regime captured into ``row.recurrent_snapshots_per_block``.
        # ``b_idx`` is the absolute block index (block 0 covers tokens
        # ``[0, block_size)``, etc.) — same key space the capture path
        # writes into. ``.get`` graceful-defaults missing keys to None,
        # which covers (a) non-slice-regime rows whose dict stays empty
        # and (b) slice-regime rows whose decode-era counter never
        # crossed a particular block boundary. ``insert_detached``'s
        # C5.3.0 contract handles None entries: new nodes get
        # ``recurrent_snapshot=None``; duplicate-prefix branch's
        # backfill only fires on a non-None caller-provided snapshot.
        recurrent_snapshots = [
            row.recurrent_snapshots_per_block.get(b_idx)
            for b_idx in range(n_aligned_blocks)
        ]

        # S-3b eager materialisation: force each slice to its own
        # backing memory BEFORE the source cache's filter / drop.
        mx.eval(
            *[arr for pl in detached_blocks for (k, v) in pl for arr in (k, v)]
        )
        self._prefix_cache.insert_detached(
            tokens_prefix, detached_blocks, recurrent_snapshots
        )

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
    ) -> _PreemptedRow | None:
        """Strip one active row out of the batch and return it (16d-4b).

        Does steps 1-4 of prep doc ``P2_UNIT_16D_PREP.md`` §4.4 plus
        the P-3-C5.2 recurrent-state snapshot:

          0. **(P-3-C5.2)** Capture the victim row's recurrent
             state via ``adapter.snapshot_recurrent_state(
             self._batch_cache, victim_row_idx)`` when the adapter
             implements ``RecurrentStateAdapter``. Captured **before**
             any cache mutation (prefix extract / filter) so the
             snapshot reflects exactly the state the victim row had
             at the moment preempt was decided. The snapshot is
             returned alongside the detached row so
             ``_apply_preempt`` can stash it on the replay's
             ``_PendingAdmit``.
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
            ``_PreemptedRow`` carrying the detached ``_BatchRow`` and
            an optional ``RecurrentSnapshot`` (set when the adapter
            implements ``RecurrentStateAdapter``; ``None`` otherwise).
            Returns ``None`` when the request is no longer active
            (not in ``self._rows``) — that is B-7's race path.
        """
        if self._batch_cache is None:
            # No active batched cache → no row to preempt. Distinct
            # from "victim missing" but maps to the same None result.
            return None
        victim_row_idx = self._find_row_by_req_id(victim_req_id)
        if victim_row_idx is None:
            return None
        victim_row = self._rows[victim_row_idx]

        # 0. P-3-C5.2 — capture recurrent snapshot before any cache
        # mutation. The victim_row_idx is valid against
        # self._batch_cache here; once the filter at step 3 runs,
        # that index would point at a different (or missing) row.
        recurrent_snapshot: RecurrentSnapshot | None = None
        if isinstance(self._adapter, RecurrentStateAdapter):
            recurrent_snapshot = self._adapter.snapshot_recurrent_state(
                self._batch_cache, victim_row_idx
            )

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

        return _PreemptedRow(
            detached=victim_row, recurrent_snapshot=recurrent_snapshot
        )

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

        result = self._preempt_active_row(victim_req_id)
        # The second lookup inside _preempt_active_row scans the same
        # self._rows we just read; no mutation in between so a None
        # here would be a consistency violation.
        assert result is not None
        detached = result.detached

        composite_prompt = tuple(
            list(detached.prompt_ids) + list(detached.generated)
        )
        replay_params = detached.params.model_copy(
            update={"max_tokens": remaining}
        )
        # P-3-C5.2: stash the recurrent snapshot on the replay's
        # _PendingAdmit. The snapshot was captured at preempt time
        # against the **live** cache state, which is one token
        # earlier than the composite_prompt length:
        #
        #   len(composite_prompt) == n_prompt + len(generated)
        #   snapshot boundary    == n_prompt + len(generated) - 1
        #
        # The off-by-one is fundamental: at preempt time, the latest
        # sample was emitted from the prefill / decode logits but
        # has NOT yet been fed back into the cache. Snapshot
        # therefore describes "all of generated except the last
        # token has been consumed". This is why C5.2's
        # _admit_miss_cohort does NOT call restore_recurrent_state
        # on the full-replay path — see the boundary-mismatch
        # comment at the restore call site for the full derivation.
        # The pending owns the snapshot's lifetime; C5.3 (prefix-hit
        # admission) will be the first site to consume the snapshot
        # via restore, where the prefill ends at the matching
        # boundary by construction.
        self._waiting_queue.appendleft(
            _PendingAdmit(
                req_index=detached.req_index,
                prompt_ids=composite_prompt,
                params=replay_params,
                is_replay=True,
                recurrent_snapshot=result.recurrent_snapshot,
            )
        )
        return True

    def _slice_prefill_active(self) -> bool:
        """Predicate for the slice-prefill trajectory regime.

        Slice-prefill is the canonical regime for hybrid adapters
        running with a ``RadixPrefixCache`` so the producer (whose
        snapshots get stored) and the consumer (the prefix-hit
        admission path in ``_admit_single_hit_row``) operate in the
        same trajectory regime — see ``docs/P3_C5_3_DESIGN.md``
        §2.2 / §3.2 for why same-regime is required for byte-exact
        cooperation.

        Predicate: ``RecurrentStateAdapter`` mixin AND
        ``prefix_cache is not None``. Non-recurrent adapters and
        ``prefix_cache=None`` paths stay on contiguous prefill
        bit-for-bit.

        Returns ``True`` to enable slice-prefill in the prefill paths,
        the suffix prefill in ``_admit_single_hit_row``, and decode-
        step capture in ``_decode_phase``; the prefill callers add
        their own ``len(rows) == 1`` clamp because B>1 cohort
        slicing is post-C5.3 backlog (§5.2).
        """
        if not isinstance(self._adapter, RecurrentStateAdapter):
            return False
        return self._prefix_cache is not None

    def _effective_slice_block_size(self) -> int:
        """The block_size the slice helper / decode capture uses.

        Reads from ``self._prefix_cache.block_size``. Caller must
        guarantee ``_slice_prefill_active()`` is True — otherwise
        the helper has no business asking for a block_size.
        """
        assert self._prefix_cache is not None, (
            "_effective_slice_block_size called without a prefix_cache "
            "— _slice_prefill_active() should have been False"
        )
        return self._prefix_cache.block_size

    def _slice_prefill_with_capture(
        self,
        row: _BatchRow,
        cache: list[Any],
        tokens: Sequence[int],
    ) -> mx.array:
        """B=1 slice-prefill helper — issues ``ceil(len(tokens) / block_size)``
        forwards of up to ``block_size`` tokens each.

        Captures the per-DeltaNet-layer recurrent state at every full-
        block boundary into ``row.recurrent_snapshots_per_block`` keyed
        by absolute block index. The final chunk may be shorter than
        ``block_size`` (for non-aligned input lengths) and produces no
        snapshot — the decode-step capture path picks up snapshots at
        later block boundaries crossed by single-step decoding.

        ``tokens`` is the explicit sequence to slice-prefill. Two call
        shapes today:

        - **Initial / miss-cohort prefill** (P-3-C5.3.1):
          ``tokens = row.prompt_ids``. Counter starts at 0; absolute
          block indices align with row-relative indices.
        - **Hit-admission suffix prefill** (P-3-C5.3.3):
          ``tokens = pending.prompt_ids[usable_hit_tokens:]``. Counter
          is pre-seeded to ``K * block_size`` at admission, so this
          helper's first chunk advances to ``(K+1) * block_size`` and
          captures at absolute index ``K``.

        Block-index math: ``block_idx = absolute_consumed_tokens //
        block_size - 1`` after the per-chunk increment. Works for both
        call shapes because the seeded counter shifts the math by ``K``
        without the helper needing to know.

        Caller invariant: ``self._slice_prefill_active()`` is True and
        ``len(tokens) >= 1``. Returns the final chunk's ``(1, V)``
        logits for the sampler.
        """
        assert isinstance(self._adapter, RecurrentStateAdapter)
        # Block size comes from prefix_cache when present, or from the
        # oracle flag's int value when the test-only oracle path is
        # active (the oracle runs slice-regime with prefix_cache=None
        # and therefore has no RadixPrefixCache to read block_size
        # from; the flag's int value carries the block_size for
        # regime parity with the subject).
        block_size = self._effective_slice_block_size()
        n = len(tokens)

        # P-5-F F.2b: per-chunk K_pre capture. Allocated once outside
        # the loop and reused via ``buffer.clear()`` between chunks —
        # ``adapter.install_pre_norm_capture(buffer)`` arms before each
        # forward, ``install_pre_norm_capture(None)`` disarms after.
        # The disarm in the finally guards against the buffer leaking
        # past this scope into a subsequent decode forward.
        capture_buffer: dict[int, mx.array] | None = (
            {} if self._pre_norm_capture_active() else None
        )

        last_logits: mx.array | None = None
        start = 0
        while start < n:
            end = min(start + block_size, n)
            chunk_arr = mx.array(
                [list(tokens[start:end])], dtype=mx.int32
            )
            pre_consumed = row.absolute_consumed_tokens
            consumed = end - start
            if capture_buffer is not None:
                capture_buffer.clear()
                self._capture_adapter().install_pre_norm_capture(capture_buffer)
            try:
                last_logits = forward_batched(self._model, chunk_arr, cache)
            finally:
                if capture_buffer is not None:
                    self._capture_adapter().install_pre_norm_capture(None)
            row.absolute_consumed_tokens += consumed
            if consumed == block_size:
                snap = self._adapter.snapshot_recurrent_state(
                    cache, row_idx=0
                )
                block_idx = (
                    row.absolute_consumed_tokens // block_size - 1
                )
                row.recurrent_snapshots_per_block[block_idx] = snap
            if capture_buffer is not None:
                # B=1, no left-padding (tight chunk_arr); pad_per_row=[0].
                self._split_capture_into_row_kpre(
                    capture_buffer,
                    [row],
                    pad_per_row=[0],
                    pre_consumed_per_row=[pre_consumed],
                    real_in_chunk_per_row=[consumed],
                    block_size=block_size,
                )
            start = end

        assert last_logits is not None
        return last_logits

    def _slice_prefill_with_capture_batched(
        self,
        rows: list[_BatchRow],
        cache: list[Any],
        tokens_2d: mx.array,
    ) -> mx.array:
        """B>=1 slice-prefill helper for left-padded cohorts (P-3-C5.5).

        Companion to ``_slice_prefill_with_capture`` — same chunked
        forward + per-row snapshot capture, generalised to a batched
        cohort. Splits the existing B=1 path off from a dual-signature
        approach to avoid putting the C5.3.3b byte-exact gate at risk
        on every C5.5 edit (advisor sharpening 2).

        ``tokens_2d`` is the left-padded ``(B, max_L)`` prompt tensor
        — exactly the shape ``_build_prefill_tokens`` /
        ``_admit_miss_cohort`` produce. Each row ``i`` has
        ``pad_i = max_L - len(rows[i].prompt_ids)`` left-pad tokens at
        axis-2 ``[0, pad_i)`` and real prompt at ``[pad_i, max_L)``.

        Per-row absolute math under left-padding:

          absolute_after_chunk_c =
            max(0, min((c+1)*B, max_L) - pad_i)

        capped at ``L_i``. The capture predicate is
        ``absolute % block_size == 0 and absolute > 0`` evaluated at
        each chunk's end. **α MVP scope (design §5.2 addendum)**: this
        predicate fires reliably for every chunk only when
        ``pad_i ≡ 0 (mod block_size)`` — i.e., the longest row(s) in
        the cohort. Rows with non-aligned padding may get one capture
        at the final chunk if their ``L_i`` is a multiple of
        ``block_size``, otherwise zero captures during prefill. The
        decode-era capture path in ``_decode_phase`` picks up
        subsequent block boundaries past ``L_i``; the prefill-side
        partial-coverage limitation is documented at design §5.2.

        ``forward_batched`` returns ``(B, V)`` last-position logits
        per chunk; only the final chunk's logits feed the sampler.

        Caller invariant: ``self._slice_prefill_active()`` is True and
        every row's ``prompt_ids`` is non-empty. Returns the final
        chunk's ``(B, V)`` logits.
        """
        assert isinstance(self._adapter, RecurrentStateAdapter)
        block_size = self._effective_slice_block_size()
        b, max_len = tokens_2d.shape
        assert b == len(rows), (
            f"tokens_2d B={b} != rows={len(rows)}"
        )

        prompt_lens = [len(row.prompt_ids) for row in rows]
        pad_per_row = [max_len - n for n in prompt_lens]

        # P-5-F F.2b: per-chunk K_pre capture, same lifecycle as the
        # B=1 helper. The capture buffer holds entries shaped
        # ``(B, end - start, n_kv_heads * head_dim)`` per attn pos —
        # ``_split_capture_into_row_kpre`` slices each row's
        # block-aligned region back out using the same left-padding
        # math the snapshot-capture predicate already uses.
        capture_buffer: dict[int, mx.array] | None = (
            {} if self._pre_norm_capture_active() else None
        )

        last_logits: mx.array | None = None
        start = 0
        while start < max_len:
            end = min(start + block_size, max_len)
            chunk = tokens_2d[:, start:end]
            pre_consumed_per_row = [
                row.absolute_consumed_tokens for row in rows
            ]
            if capture_buffer is not None:
                capture_buffer.clear()
                self._capture_adapter().install_pre_norm_capture(capture_buffer)
            try:
                last_logits = forward_batched(self._model, chunk, cache)
            finally:
                if capture_buffer is not None:
                    self._capture_adapter().install_pre_norm_capture(None)

            real_in_chunk_per_row: list[int] = []
            for row_idx, row in enumerate(rows):
                pad_i = pad_per_row[row_idx]
                # Real tokens in chunk = overlap between chunk axis
                # range [start, end) and row's real range
                # [pad_i, max_len). Negative overlaps clamp to 0.
                real_start = max(start, pad_i)
                real_end = min(end, max_len)
                real_in_chunk = max(0, real_end - real_start)
                real_in_chunk_per_row.append(real_in_chunk)
                row.absolute_consumed_tokens += real_in_chunk
                if (
                    row.absolute_consumed_tokens > 0
                    and row.absolute_consumed_tokens % block_size == 0
                ):
                    snap = self._adapter.snapshot_recurrent_state(
                        cache, row_idx=row_idx
                    )
                    block_idx = (
                        row.absolute_consumed_tokens // block_size - 1
                    )
                    row.recurrent_snapshots_per_block[block_idx] = snap

            if capture_buffer is not None:
                # In the chunk's local axis-1, real tokens for row i
                # start at ``max(0, pad_i - start)``. Slice helper
                # accepts that as ``pad_per_row``.
                local_pad_per_row = [
                    max(0, pad_per_row[i] - start) for i in range(b)
                ]
                self._split_capture_into_row_kpre(
                    capture_buffer,
                    rows,
                    pad_per_row=local_pad_per_row,
                    pre_consumed_per_row=pre_consumed_per_row,
                    real_in_chunk_per_row=real_in_chunk_per_row,
                    block_size=block_size,
                )
            start = end

        assert last_logits is not None
        return last_logits

    def _prefill_phase(self) -> list[BatchEvent]:
        """One batched forward over the left-padded prompt tensor.

        Routes through the slice-prefill helpers when the C5.3
        slice-prefill regime is active:

        - B=1: ``_slice_prefill_with_capture`` (the original B=1
          helper, byte-exact-gated by C5.3.3b).
        - B>1: ``_slice_prefill_with_capture_batched`` (P-3-C5.5
          α MVP — full capture for ``pad_i ≡ 0 (mod block_size)``
          rows, partial elsewhere; design §5.2 addendum).

        Non-slice-regime falls back to today's single-batched-forward.
        """
        assert self._batch_cache is not None
        if self._slice_prefill_active():
            if len(self._rows) == 1:
                logits = self._slice_prefill_with_capture(
                    self._rows[0],
                    list(self._batch_cache),
                    self._rows[0].prompt_ids,
                )
            else:
                tokens = self._build_prefill_tokens()  # (B, T_max)
                logits = self._slice_prefill_with_capture_batched(
                    list(self._rows),
                    list(self._batch_cache),
                    tokens,
                )
        else:
            tokens = self._build_prefill_tokens()  # (B, T_max)
            # P-5-F F.2b: contiguous prefill capture. Single forward
            # over the entire left-padded cohort; the buffer holds the
            # full prompt's K_pre per attention layer. After the
            # forward completes, ``_split_capture_into_row_kpre``
            # slices each row's block-aligned region using the same
            # left-padding math the live cache uses (axis-2 tokens
            # ``[pad_i, pad_i + L_i)`` carry row i's real K_pre).
            capture_buffer: dict[int, mx.array] | None = (
                {} if self._pre_norm_capture_active() else None
            )
            if capture_buffer is not None:
                self._capture_adapter().install_pre_norm_capture(capture_buffer)
            try:
                logits = forward_batched(
                    self._model, tokens, list(self._batch_cache)
                )  # (B, V)
            finally:
                if capture_buffer is not None:
                    self._capture_adapter().install_pre_norm_capture(None)
            if capture_buffer is not None:
                # ``_pre_norm_capture_active`` implies prefix_cache.
                assert self._prefix_cache is not None
                prompt_lens = [len(r.prompt_ids) for r in self._rows]
                max_len = max(prompt_lens)
                pad_per_row = [max_len - n for n in prompt_lens]
                self._split_capture_into_row_kpre(
                    capture_buffer,
                    list(self._rows),
                    pad_per_row=pad_per_row,
                    pre_consumed_per_row=[0] * len(self._rows),
                    real_in_chunk_per_row=prompt_lens,
                    block_size=self._prefix_cache.block_size,
                )
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

        When the C5.3 slice-prefill regime is active, captures
        per-row recurrent snapshots at decode-step block boundaries.
        The capture is gated on ``row.absolute_consumed_tokens > 0``:
        a zero counter means the row was admitted via contiguous
        prefill (B>1 miss cohort, slice-prefill skipped) and its
        decode-era trajectory is "contiguous + decode" — a different
        regime from slice-prefill, so capturing here would contaminate
        the byte-exact-vs-slice-oracle gate. Pre-step terminal rows
        are also skipped (their cache fed a pad token, not a real
        sample). See design §4.2 finding C.
        """
        assert self._batch_cache is not None
        tokens = self._build_decode_tokens()  # (B, 1)
        logits = forward_batched(
            self._model, tokens, list(self._batch_cache)
        )  # (B, V)
        if self._slice_prefill_active():
            # mypy narrowing: _slice_prefill_active guarantees the
            # adapter mixin. block_size resolves through
            # _effective_slice_block_size so the oracle path (no
            # prefix_cache) reads the flag's int value instead.
            assert isinstance(self._adapter, RecurrentStateAdapter)
            block_size = self._effective_slice_block_size()
            for row_idx, row in enumerate(self._rows):
                if row.state.is_terminal:
                    continue
                if row.absolute_consumed_tokens == 0:
                    continue
                row.absolute_consumed_tokens += 1
                if row.absolute_consumed_tokens % block_size == 0:
                    snap = self._adapter.snapshot_recurrent_state(
                        list(self._batch_cache), row_idx=row_idx
                    )
                    idx = (
                        row.absolute_consumed_tokens // block_size - 1
                    )
                    row.recurrent_snapshots_per_block[idx] = snap
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
            needs_recurrent_snapshot = isinstance(
                self._adapter, RecurrentStateAdapter
            )
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
                    continue
                # P-3-C5.3.3: hybrid adapters require a recurrent
                # snapshot at the deepest USABLE node — i.e., the
                # node ``_admit_single_hit_row`` will actually
                # restore from. The raw deepest node may be deeper
                # than ``usable`` when the prompt is fully cached
                # (``raw.num_hit_tokens=8`` vs ``max_aligned=4`` for
                # an 8-token block_size=4 prompt). Re-walk via
                # ``peek_with_node(prompt[:usable])`` so the node
                # whose snapshot we inspect matches the node the
                # atomic phase will consume. ``peek_with_node`` is
                # side-effect-free; no ``retain_hit`` fires before
                # the routing decision (design §3.5).
                if needs_recurrent_snapshot:
                    _, deepest_usable = self._prefix_cache.peek_with_node(
                        pending.prompt_ids[:usable]
                    )
                    if (
                        deepest_usable is None
                        or deepest_usable.recurrent_snapshot is None
                    ):
                        miss_rows.append(pending)
                        continue
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

        hit, deepest_usable = self._prefix_cache.lookup_with_node(
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

            # P-5-F F.2b: when the store carries pre-norm K
            # (``store.pre_norm=True``, F.2a contract), the K side of
            # every fetched ``(K, V)`` is pre-k_norm — the proxy-
            # captured ``k_proj(x)``. Reconstruct post-RoPE K via the
            # adapter Protocol method ``apply_k_norm_then_rope`` per
            # block at offset ``b_idx * block_size`` before passing
            # into ``build_seeded_batch_kv``. ``b_idx`` here is dense
            # over the hit's aligned blocks (block 0 is the radix
            # tree's first hit block); RoPE offset arithmetic in the
            # capture path used the same dense indexing, so the
            # phase-rotation contract holds across the seam.
            if self._pre_norm_capture_active():
                hit_block_size = self._prefix_cache.block_size
                detached = [
                    [
                        (
                            self._capture_adapter().apply_k_norm_then_rope(
                                attn_pos,
                                k_pre,
                                offset=b_idx * hit_block_size,
                            ),
                            v,
                        )
                        for attn_pos, (k_pre, v) in enumerate(per_layer)
                    ]
                    for b_idx, per_layer in enumerate(detached)
                ]

            # P-3-C5.3.3-het.2: row_cache assembly for hybrid adapters.
            # ``detached_blocks[b]`` is indexed by **attention-layer
            # position** (the het.1 extract path stores only token-K/V
            # layers — DeltaNet positions carry no token K/V to slice).
            # For hybrid Qwen3.5 the row_cache must mirror the
            # heterogeneous shape from ``adapter.make_batch_cache``:
            # ``ArraysCache`` for DeltaNet positions, seeded
            # ``BatchKVCache`` for attention positions. The pre-het.2
            # code passed ``num_layers=adapter.config.num_layers`` to
            # ``build_seeded_batch_kv`` which (a) trips the helper's
            # ``len(detached[0]) == num_layers`` validator on hybrid
            # and (b) would yield an all-``BatchKVCache`` list that
            # ``restore_recurrent_state`` cannot splice DeltaNet state
            # into. For pure-attention adapters the assembly degenerates
            # to today's behaviour: every transformer layer is an
            # attention layer, ``empty_row_cache`` is an all-empty
            # ``BatchKVCache`` list, and seeded entries replace every
            # slot 1:1.
            empty_row_cache = _make_batch_cache(self._adapter, [0])
            attn_layer_indices = self._token_kv_layer_indices(
                empty_row_cache
            )
            seeded_attn = build_seeded_batch_kv(
                detached, num_layers=len(attn_layer_indices)
            )
            row_cache: list[Any] = list(empty_row_cache)
            for pos, src_layer_idx in enumerate(attn_layer_indices):
                row_cache[src_layer_idx] = seeded_attn[pos]
            num_layers = len(row_cache)

            # P-3-C5.3.3 Insertion A — seed row metadata for slice
            # regime: walk the ancestor chain of ``deepest_usable``
            # and copy each ancestor's ``recurrent_snapshot`` into
            # ``row.recurrent_snapshots_per_block`` keyed by its
            # absolute block index. Set ``absolute_consumed_tokens``
            # to ``usable_hit_tokens`` so the suffix slice helper's
            # block-index math starts at absolute index ``K``. Per
            # design §4.4 (relocated from §4.2). ``_Node.parent``
            # walks deepest → root; reverse to assign indices 0..K-1.
            # Ancestors with ``recurrent_snapshot is None`` (legacy
            # nodes) leave the dict key absent — the design's
            # ``dict.get`` graceful default in extract / decode-step
            # capture handles missing keys.
            if isinstance(self._adapter, RecurrentStateAdapter):
                assert deepest_usable is not None  # non-empty hit
                chain: list[Any] = []
                cursor = deepest_usable
                while cursor is not None and cursor.parent is not None:
                    chain.append(cursor)
                    cursor = cursor.parent
                chain.reverse()
                for abs_idx, node in enumerate(chain):
                    if node.recurrent_snapshot is not None:
                        row.recurrent_snapshots_per_block[abs_idx] = (
                            node.recurrent_snapshot
                        )
                row.absolute_consumed_tokens = usable_hit_tokens

            # P-3-C5.3.3 Insertion B — restore the deepest-usable
            # node's recurrent snapshot into the freshly-seeded
            # ``row_cache``. Phase-B classifier guarantees the
            # snapshot exists (route-to-miss happens upstream when
            # it's None), so the assert below is a tripwire for a
            # classifier bug, not a fallback. Restore must precede
            # the suffix prefill (Insertion C) — the suffix forward
            # advances the recurrent state from the restored
            # boundary.
            if isinstance(self._adapter, RecurrentStateAdapter):
                assert hit.block_ids, (
                    "hit path entered with empty block_ids"
                )
                assert deepest_usable is not None
                snapshot = deepest_usable.recurrent_snapshot
                assert snapshot is not None, (
                    "hit path entered with snapshotless deepest-usable "
                    "node — Phase-B classifier should have routed to miss"
                )
                self._adapter.restore_recurrent_state(
                    row_cache, 0, snapshot
                )

            # P-3-C5.3.3 Insertion C — suffix prefill routes through
            # the slice helper when slice-regime is active so capture
            # at absolute indices ``K, K+1, ...`` matches the
            # producer-side regime; otherwise stays on today's
            # contiguous suffix forward.
            suffix_tokens = list(pending.prompt_ids[usable_hit_tokens:])
            # S-5 edge 1 guarantees suffix_tokens is non-empty — the
            # max_aligned formula above reserves at least one token.
            if self._slice_prefill_active():
                # Slice helper (B=1) installs and reads the K_pre
                # capture buffer per chunk inside the loop; nothing
                # more to do here. The helper's pre_consumed / split
                # math accounts for the row's seeded
                # ``absolute_consumed_tokens = K * block_size``.
                logits = self._slice_prefill_with_capture(
                    row, list(row_cache), suffix_tokens
                )
            else:
                # P-5-F F.2b non-slice suffix capture. The seeded
                # ``row.absolute_consumed_tokens`` is 0 here (this
                # branch is reached on non-recurrent adapters where
                # the seed step did not initialise the counter); use
                # ``usable_hit_tokens`` directly as the pre-consumed
                # baseline so block-index math lines up with the hit
                # block range (blocks ``[K, K + suffix_blocks)``).
                suffix_arr = mx.array([suffix_tokens], dtype=mx.int32)
                hit_capture: dict[int, mx.array] | None = (
                    {} if self._pre_norm_capture_active() else None
                )
                if hit_capture is not None:
                    self._capture_adapter().install_pre_norm_capture(hit_capture)
                try:
                    logits = forward_batched(
                        self._model, suffix_arr, list(row_cache)
                    )  # (1, V)
                finally:
                    if hit_capture is not None:
                        self._capture_adapter().install_pre_norm_capture(None)
                if hit_capture is not None:
                    suffix_block_size = self._prefix_cache.block_size
                    self._split_capture_into_row_kpre(
                        hit_capture,
                        [row],
                        pad_per_row=[0],
                        pre_consumed_per_row=[usable_hit_tokens],
                        real_in_chunk_per_row=[len(suffix_tokens)],
                        block_size=suffix_block_size,
                    )
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

        # P-3-C5.3.1 / P-3-C5.5: route through slice-prefill when the
        # regime is active. B=1 takes the original tight-tensor path
        # (no padding); B>1 builds the (B, max_L) left-padded tensor
        # and routes to the C5.5 batched helper. Non-slice-regime
        # cohorts stay on contiguous batched prefill bit-for-bit.
        if self._slice_prefill_active():
            if len(admitted) == 1:
                logits = self._slice_prefill_with_capture(
                    admitted[0],
                    list(k_batch_cache),
                    admitted[0].prompt_ids,
                )
            else:
                pad = self._pad_token_id()
                rows_2d: list[list[int]] = []
                for r in admitted:
                    n_pad = k_max_len - len(r.prompt_ids)
                    rows_2d.append([pad] * n_pad + list(r.prompt_ids))
                tokens = mx.array(rows_2d, dtype=mx.int32)
                logits = self._slice_prefill_with_capture_batched(
                    admitted, list(k_batch_cache), tokens
                )
        else:
            pad = self._pad_token_id()
            rows_2d = []
            for r in admitted:
                n_pad = k_max_len - len(r.prompt_ids)
                rows_2d.append([pad] * n_pad + list(r.prompt_ids))
            tokens = mx.array(rows_2d, dtype=mx.int32)
            # P-5-F F.2b: same contiguous-prefill capture as
            # ``_prefill_phase`` non-slice branch. The mid-run cohort
            # uses a fresh ``k_batch_cache`` rather than the main
            # ``self._batch_cache``, but K_pre capture is independent
            # of the cache (it hooks ``attn.k_proj`` projection
            # output), so the same arm/disarm/split lifecycle applies.
            mc_capture: dict[int, mx.array] | None = (
                {} if self._pre_norm_capture_active() else None
            )
            if mc_capture is not None:
                self._capture_adapter().install_pre_norm_capture(mc_capture)
            try:
                logits = forward_batched(
                    self._model, tokens, list(k_batch_cache)
                )
            finally:
                if mc_capture is not None:
                    self._capture_adapter().install_pre_norm_capture(None)
            if mc_capture is not None:
                assert self._prefix_cache is not None
                pad_per_row = [k_max_len - n for n in k_prompt_lens]
                self._split_capture_into_row_kpre(
                    mc_capture,
                    admitted,
                    pad_per_row=pad_per_row,
                    pre_consumed_per_row=[0] * len(admitted),
                    real_in_chunk_per_row=k_prompt_lens,
                    block_size=self._prefix_cache.block_size,
                )
        events = self._sample_and_emit_rows(
            admitted, logits, is_prefill=True
        )

        # P-3-C5.2 boundary note — restore is INTENTIONALLY NOT called
        # on the full-replay admission path. The snapshot stashed on
        # ``pending.recurrent_snapshot`` describes the recurrent state
        # at boundary ``T + len(generated) - 1`` consumed (the cache's
        # natural pre-preempt state, with the latest sample held but
        # not yet fed back). The replay prefills the full
        # ``composite_prompt = prompt + generated`` (``T + len(generated)``
        # tokens), so ``k_batch_cache`` ends at ``T + len(generated)``
        # consumed — one token ahead of the snapshot. Restoring the
        # snapshot here would rewind the cache by one position and
        # erase the consumption of the last generated token, breaking
        # the replay's trajectory. C5.3 (RadixPrefixCache cooperation)
        # will enable restore only on admission paths whose post-
        # prefill boundary equals the snapshot boundary, e.g. prefix-
        # hit paths where the prefill ends at the prefix-block edge
        # the snapshot was captured against. See
        # docs/P3_C5_DRIFT_EXPERIMENT/README.md "C5.2 acceptance" for
        # the full boundary derivation.

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

        As of P-3-E4 the accepted set is ``GLOBAL`` (plain attention,
        P-2), ``HYBRID_DELTANET`` (Qwen3.5 hybrid family, P-3-C3c), and
        ``SLIDING`` (Gemma4 sliding-window family, P-3-D3 — restricted
        to the ``prefix_cache=None`` path by a separate constructor
        guard). ``has_moe=True`` is no longer rejected: the gate-lift
        landed at P-3-E4 (smoke-only, parity deferred) — mlx-lm's
        ``SwitchGLU`` + ``gather_mm`` path is B-agnostic per the
        P3_MOE_SURVEY §5 E4 audit, so a batched forward through an
        MoE adapter dispatches per-row top-k experts without further
        scheduler work. ``RECURRENT`` (pure-Mamba) and ``HYBRID``
        (single-kind sliding/global hybrid) remain rejected.
        """
        caps = adapter.capabilities()
        if caps.attention_kinds.issubset(_SUPPORTED_ATTENTION_KINDS):
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
            raise NotImplementedError(
                f"ContinuousBatcher accepts AttentionKind.GLOBAL, "
                f"AttentionKind.HYBRID_DELTANET, and AttentionKind.SLIDING "
                f"only; adapter capabilities include {kind.value!r} "
                f"(layer {layer_idx}, has_recurrent_state="
                f"{caps.has_recurrent_state}, has_moe={caps.has_moe}) — "
                f"{reason}"
            )
        # Fallback: ``ModelCapabilities`` and ``AttentionPattern``
        # disagree — ``attention_kinds`` reported an unsupported kind
        # but the per-layer walk found none. Pre-E4 this branch also
        # caught the ``has_moe=True`` path; that branch is now
        # accepted, so the only way to reach here is a pathological
        # adapter whose two capability views disagree.
        kinds_display = sorted(k.value for k in caps.attention_kinds)
        raise NotImplementedError(
            f"ContinuousBatcher accepts AttentionKind.GLOBAL, "
            f"AttentionKind.HYBRID_DELTANET, and AttentionKind.SLIDING "
            f"only; adapter capabilities are unsupported: "
            f"attention_kinds={kinds_display!r}, "
            f"has_recurrent_state={caps.has_recurrent_state}, "
            f"has_moe={caps.has_moe}; attention_pattern() reports no "
            f"unsupported layer, so ModelCapabilities and "
            f"AttentionPattern disagree — re-run "
            f"capabilities_from_attention_pattern on the adapter's "
            f"pattern to regenerate a consistent summary"
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
