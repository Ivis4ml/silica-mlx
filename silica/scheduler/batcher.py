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

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

from silica.core.events import BatchEvent
from silica.core.request import Request, RequestState, RequestStatus
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.mlx.runner import forward_batched
from silica.models.adapter import AttentionKind, ModelAdapter
from silica.weights.provider import WeightProvider
from silica.weights.resident import ResidentWeightProvider


@dataclass(frozen=True)
class _PendingAdmit:
    """One request waiting for cohort admission (16c.1).

    Only held in the batcher's waiting queue; popped into an active
    ``_BatchRow`` during the admit phase of some future ``step()``.
    """

    req_index: int
    prompt_ids: tuple[int, ...]
    params: SamplingParams


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
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1, got {max_batch_size}"
            )
        self._enforce_capability_gate(adapter)
        self._adapter = adapter
        self._sampler = sampler or Sampler()
        self._model = adapter.build(weight_provider or ResidentWeightProvider())
        self._max_batch_size = max_batch_size
        self._rows: list[_BatchRow] = []
        self._batch_cache: list[BatchKVCache] | None = None
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
        """Seal the initial cohort; allocate BatchKVCache with per-row
        left_padding.

        Rows with shorter prompts get more left padding so that, after
        the prefill forward, the ``offset[i] == T_max - left_padding[i]``
        equals each row's true prompt length.

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
        num_layers = self._adapter.config.num_layers
        self._batch_cache = [
            BatchKVCache(left_padding=left_padding) for _ in range(num_layers)
        ]
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
        16c.2 will add an eager-extract hook here, *before* ``filter``,
        to detach pinned K/V for the prefix cache.
        """
        if not any(r.state.is_terminal for r in self._rows):
            return
        kept = [i for i, r in enumerate(self._rows) if not r.state.is_terminal]
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

    def _rebuild_slot_table(self) -> None:
        """Re-derive ``slot_table`` from the current ``self._rows``.

        Must be called after any operation that reshuffles
        ``self._rows`` — notably ``filter`` (which mlx-lm's BatchKVCache
        re-indexes to ``0..K-1``) and, in 16c.1 step 3, ``extend`` (to
        record the newly-admitted rows' indices).
        """
        self._slot_table = {row.req_index: i for i, row in enumerate(self._rows)}

    def _prefill_phase(self) -> list[BatchEvent]:
        """One batched forward over the left-padded prompt tensor."""
        assert self._batch_cache is not None
        tokens = self._build_prefill_tokens()  # (B, T_max)
        logits = forward_batched(
            self._model, tokens, list(self._batch_cache)
        )  # (B, V)
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
        requests from ``_waiting_queue`` and run one batched prefill
        for them.

        Extends main batched cache with the K new rows, or replaces
        main outright when main is ``None`` (post all-terminal reclaim).

        Returns: emitted events (K token events + any done events
        from rows that hit stop / max_tokens on the first token). An
        empty list means no admission fired this step — the caller
        then proceeds to Phase 3 forward.
        """
        capacity = self._max_batch_size - len(self._rows)
        if capacity <= 0 or not self._waiting_queue:
            return []

        # Pop up to `capacity` pending admits.
        admitted: list[_BatchRow] = []
        while admitted.__len__() < capacity and self._waiting_queue:
            pending = self._waiting_queue.popleft()
            request = Request(
                prompt="",
                sampling_params=pending.params,
                request_id=f"req-{pending.req_index}",
                token_ids=pending.prompt_ids,
            )
            state = RequestState(request=request)
            # New rows start in PREFILL; _sample_and_emit_rows transitions
            # them to DECODE (or DONE) after sampling.
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

        if not admitted:
            return []

        # Build fresh B=K BatchKVCache directly (prep doc §4 Fix 5 — no
        # merge-of-empty-KVCaches). Mirrors 16b's _prepare_cohort layout
        # but sized for the admit subset.
        num_layers = self._adapter.config.num_layers
        k_prompt_lens = [len(r.prompt_ids) for r in admitted]
        k_max_len = max(k_prompt_lens)
        k_left_padding = [k_max_len - n for n in k_prompt_lens]
        k_batch_cache = [
            BatchKVCache(left_padding=k_left_padding) for _ in range(num_layers)
        ]

        # Left-padded (K, k_max_len) token tensor for the admit subset.
        pad = self._pad_token_id()
        rows_2d: list[list[int]] = []
        for r in admitted:
            n_pad = k_max_len - len(r.prompt_ids)
            rows_2d.append([pad] * n_pad + list(r.prompt_ids))
        tokens = mx.array(rows_2d, dtype=mx.int32)

        # One batched prefill forward over just the admitted rows.
        logits = forward_batched(self._model, tokens, list(k_batch_cache))
        events = self._sample_and_emit_rows(admitted, logits, is_prefill=True)

        # Stitch into main cache. Two regimes:
        #   (a) main is None (post all-terminal reclaim): replace.
        #   (b) main has live rows: extend per layer.
        if self._batch_cache is None:
            self._batch_cache = k_batch_cache
        else:
            for layer in range(num_layers):
                self._batch_cache[layer].extend(k_batch_cache[layer])

        self._rows.extend(admitted)
        self._rebuild_slot_table()
        return events

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
        """P-2 batcher only accepts adapters whose every layer is GLOBAL.

        Layer 3 of the v2.3 three-layer stack. The batcher does not
        test ``isinstance(adapter, ...)``; it asks the adapter what
        ``AttentionPattern`` it claims and decides from that alone.
        Mamba / DeltaNet / sliding-window families will each need their
        own scheduler (or an extended capability matrix) — this gate
        simply refuses to guess.
        """
        pattern = adapter.attention_pattern()
        for layer_idx, kind in enumerate(pattern.per_layer):
            if kind == AttentionKind.GLOBAL:
                continue
            reason = _unsupported_kind_reason(kind)
            raise NotImplementedError(
                f"ContinuousBatcher accepts AttentionKind.GLOBAL only in P-2; "
                f"adapter layer {layer_idx} is {kind.value!r} — {reason}"
            )


def _unsupported_kind_reason(kind: AttentionKind) -> str:
    """Short explanation used by the capability-gate error message."""
    if kind == AttentionKind.HYBRID_DELTANET:
        return (
            "hybrid DeltaNet batching arrives in P-3 via "
            "BatchRecurrentStateStore on the adapter side"
        )
    if kind == AttentionKind.RECURRENT:
        return (
            "pure-recurrent batching arrives in P-3 via "
            "BatchRecurrentStateStore"
        )
    if kind == AttentionKind.SLIDING:
        return (
            "sliding-window batching is gated on the P-2 Q-013 "
            "BatchRotatingKVCache probe"
        )
    if kind == AttentionKind.HYBRID:
        return "sliding/global hybrid needs the Q-013 probe first"
    return "unknown AttentionKind; no scheduler mapping defined"
