# P-2 Unit 16d — prep: budget-aware preemption + re-admit + abort

**Date:** 2026-04-17
**Status:** design prep — no code changes yet.
**Depends on:** Unit 16c.2 (`4460541`).
**Blocks:** PLAN §7 P-2 acceptance #3 (budget overflow clean).

## Why this doc

Unit 16c.2 wired the prefix cache into admission + reclaim. Unit 16d
adds the **third leg** of P-2's admission policy — memory budget —
and closes PLAN §7 acceptance #3 ("exceeding the memory budget aborts
/ queues cleanly without crashing").

Most of the pieces already exist independently:

- `silica.scheduler.budget.MemoryBudgeter` (Unit #15) — decision-only
  policy with an `admit → evict → preempt → reject` ladder.
- `RequestState` has `PREEMPTED` with transitions
  `PREFILL/DECODE → PREEMPTED → WAITING|ABORTED` (Unit #12).
- `BatchEvent.aborted(req_index, finish_reason)` factory (Unit 16a).
- `BatchKVCache.filter` already reshuffles the batch axis — 16c.1's
  `_reclaim_terminated` uses this shape.
- `RadixPrefixCache.evict_until` already releases LRU leaves.

What 16d adds is the **wiring**: the batcher calls the budgeter at
each admission decision, applies the decision's physical action
(evict / preempt / abort), and plumbs the resulting event stream
through. No new data structures, no new Protocol split.

---

## 1. Scope & sub-commit split

Four sub-commits, each independently testable:

1. **16d-1 — MemoryBudgeter decouple from PagedKVCache.** Today it
   takes `kv: PagedKVCache`; 16d needs it to drive an admission on a
   `BatchKVCache`-backed batcher. Replace the `kv` dependency with
   direct `block_size: int` + keep `prefix_cache: RadixPrefixCache`.
   Tests: existing `tests/test_memory_budgeter.py` fixture updated.

2. **16d-2 — admit / reject plumbing.** Batcher ctor grows a
   `budgeter: MemoryBudgeter | None` kwarg. Each admission consults
   `budgeter.admit(...)`. On `AdmitDecision`: proceed as before.
   On `RejectDecision`: emit `BatchEvent.aborted(req_index,
   "budget-exhausted")`, do not enter the prefix-hit / miss
   admission flow for that request. No preempt / evict yet.

3. **16d-3 — evict path.** On `AdmitAfterEvictDecision(n_blocks)`:
   call `prefix_cache.evict_until(n_blocks)` before continuing the
   admission. Assert freed count matches what the decision asked for;
   if underrun (eviction races), fall back to reject.

4. **16d-4 — preempt path + re-admit continuation.** On
   `AdmitAfterPreemptDecision(req_id)`: identify the victim row
   (must be in `self._rows`), extract its prefix K/V into the
   prefix cache (same mechanism as reclaim), filter it out of the
   batched cache, transition its state to PREEMPTED → WAITING, and
   re-enqueue as a `_PendingAdmit` whose `prompt_ids = original +
   generated_so_far`. The new admission (the one that triggered the
   preempt) then proceeds. Real-model acceptance
   `test_budget_overflow_aborts_cleanly` rolled in here.

Each sub-commit pauses for user confirmation before `git commit` per
feedback_commit_approval.md.

---

## 2. Physical model

### 2.1 What "cap" means in the BatchKVCache world

MemoryBudgeter's cap is **bytes of KV residency**. `PagedKVCache`
reports this via `budget().resident_bytes` — sum of claimed-block
count × bytes-per-block. Under BatchKVCache there is no "block" in
the physical sense (Option B discards paging); residency is driven
directly by the batched K/V tensors:

```text
resident_bytes = sum over layers of (
    self._batch_cache[l].keys.size * dtype_bytes
  + self._batch_cache[l].values.size * dtype_bytes
)
```

But the budgeter's `admit()` decision does NOT actually query kv
residency — it uses `reserved_bytes` (a tally of worst-case bytes per
admitted-not-released request) to decide admission. The per-request
reservation `(n_prompt + max_tokens) * bytes_per_token` is an
admission-time upper bound, not a live reading.

So: the budgeter needs `bytes_per_token` and `block_size` as static
construction inputs; it does NOT need a handle to the kv cache at
decision time. Today's `kv: PagedKVCache` ctor dependency is used
only to fetch `block_size` and `bytes_per_block` — both derivable
from `bytes_per_token * block_size`. The refactor in 16d-1 removes
this dependency (see §3).

### 2.2 Preempt K/V recycle vs prefix-cache reuse

Preempting a DECODE request means dropping its K/V bytes. The
**prompt tokens' K/V** is exactly the kind of content the prefix
cache would benefit from retaining — a future request sharing the
same prompt can hit through the usual admission path without
re-computing those tokens.

16d-4 reuses 16c.2's `_extract_and_insert_prefix` before the
preempt-side filter. Concretely, preempt's physical steps are:

```text
1. Identify victim_row_idx from req_id.
2. Call _extract_and_insert_prefix(victim_row_idx).   # 16c.2's helper
3. kept = [i for i in range(len(self._rows)) if i != victim_row_idx]
4. for layer_cache in self._batch_cache:
       layer_cache.filter(kept)
5. self._rows = [self._rows[i] for i in kept]
6. self._rebuild_slot_table()
```

This is **identical** to reclaim for a terminal row, except the
transition is PREFILL/DECODE → PREEMPTED instead of → DONE. We lean
into that similarity — one physical path, two state-level outcomes.

### 2.3 Re-admit continuation

The preempted request's "new prompt" for re-admission is
`prompt_ids + generated_so_far` where `generated_so_far` is
`row.generated` at preempt time. The `[:-1]` trim does NOT apply here
because **all tokens including the last sampled one have observable
effects** — the last generated token was emitted via a
`BatchEvent.token`. Losing it on re-admit would produce duplicate
tokens on the caller side, or force the caller to dedupe. Neither is
acceptable. So:

```text
composite_prompt = row.prompt_ids + row.generated  # include last
```

On re-admit the batcher peeks/looks up this composite prompt against
the prefix cache. With 16c.2's flow the prefix cache now holds the
K/V for `row.prompt_ids + row.generated[:-1]` (that's what
`_extract_and_insert_prefix` inserted). So peek returns a hit on that
full body minus one token, and the suffix prefill processes the
**one unprocessed token** (the last generated one, which was sampled
from logits but never fed back as a decode step).

Numerically this is the same work the original run would have done
as its next decode step — 16d's re-admit picks up exactly where
preempt left off for the **token stream**. Continuity of anything
else on the old `_BatchRow` / `RequestState` (history log,
`state_delta_snapshot`) is explicitly out of scope for 16d — see
§4.6.

---

## 3. MemoryBudgeter refactor (16d-1)

Minimal API change. Today:

```python
MemoryBudgeter(
    *,
    kv: PagedKVCache,
    prefix_cache: RadixPrefixCache,
    weights_bytes: int,
    bytes_per_token: int,
    cap_bytes: int,
)
```

After:

```python
MemoryBudgeter(
    *,
    prefix_cache: RadixPrefixCache,
    weights_bytes: int,
    bytes_per_token: int,
    block_size: int,            # NEW — was derived from kv.block_size
    cap_bytes: int,
)
```

`kv` parameter removed. Two places it was used:

- `self._kv = kv` stored for `_kv_bytes_per_block()`. Replaced by
  stored `self._block_size`; helper becomes
  `self._bytes_per_token * self._block_size`.
- Comment in docstring about `PagedKVCache.budget().resident_bytes`
  kept as a non-normative note on what canonical-residency means —
  the budgeter never reads it at decision time today.

`_count_evictable_prefix_blocks` already uses `self._pc` (the prefix
cache); unchanged.

**Existing tests.** `tests/test_memory_budgeter.py` passes a
`PagedKVCache` in its `_make` fixture. Post-refactor the fixture
constructs the budgeter with `block_size=kv.block_size` directly
and drops the `kv=` kwarg. Every behaviour assertion on the
budgeter itself is unchanged.

The `PagedKVCache` fixture stays in the test file because a
`RadixPrefixCache` with `PagedPrefixBlockStore` still wraps a
`PagedKVCache` — the prefix cache side of the tests does not change.

---

## 4. Batcher wiring (16d-2..16d-4)

### 4.1 Ctor additions

```python
def __init__(
    self,
    adapter: ModelAdapter,
    *,
    sampler: Sampler | None = None,
    weight_provider: WeightProvider | None = None,
    max_batch_size: int = 1,
    prefix_cache: RadixPrefixCache | None = None,
    budgeter: MemoryBudgeter | None = None,          # NEW
) -> None:
    ...
    self._budgeter: MemoryBudgeter | None = budgeter
    # 16d counters, paralleling 16c.2's prefix_hits / forward_prompt_tokens.
    self.preempts: int = 0
    self.evictions: int = 0
    self.aborts: int = 0
```

When `budgeter is None` (default), every admission proceeds as in
16c.2 — no budget check. Invariant **B-1: no-budgeter bit-identical
to 16c.2.** All existing tests remain green without modification.

### 4.2 Decision site + reservation timing

There is exactly one place in the batcher where we call
`budgeter.admit`: at the top of `_admit_waiting_requests`, per
pending admission, **before** it is routed to hit / miss path.

**Reservation-timing invariant (B-8).** Within one
`_admit_waiting_requests` loop, the budgeter's internal
`reserved_bytes` must reflect all prior admissions in the same loop
before the next `admit()` call runs. `MemoryBudgeter.admit()` is pure
— it does NOT mutate `reserved_bytes`; only `apply_admit()` does.
So we must call `apply_admit` **immediately after** the decision's
prerequisite (evict / preempt) runs and BEFORE routing to the hit /
miss body. Any subsequent failure (forward raise, extend mismatch)
triggers `release(req_id)` in an `except` block so the reservation
does not leak.

Without this discipline, a 4-pending admission step would see every
pending against the same initial headroom and over-admit
systematically.

**Two-phase shape — preserves 16c.2's miss-cohort batching.**
16c.2's `_admit_waiting_requests` batches K miss rows into a single
prefill forward. We must not regress that — routing every pending
through a single-row admit body (regardless of hit/miss) would
serialise the miss path and silently halve throughput. The fix is
to split admission into two phases:

  **Phase A — decide + apply (per-pending).** Pop one pending at a
  time from the queue (no pre-batch pop), run the budgeter decision,
  apply its prerequisite (evict / preempt) + `apply_admit`, and
  collect the accepted pendings into a list. Reject / underrun /
  replay-requires-wait cases emit their aborted event (or requeue
  at front + break) at this stage.

  **Phase B — execute (grouped).** With the accepted list in hand,
  route by hit/miss exactly as 16c.2 does: per-row
  `_admit_single_hit_row` for hits, batched `_admit_miss_cohort` for
  misses. Failure in either body releases only the reservations of
  the rows that did not commit to `self._rows` — successful rows in
  the same step keep their reservations and events.

**Pop-one-at-a-time (not pre-batch-pop).** Using
`while capacity and self._waiting_queue: pending = popleft()`
means a mid-loop `break` (triggered by a replay that cannot admit
or a preempt that failed) only needs to requeue the single
currently-held pending. Pre-popping a batch and breaking mid-iteration
would strand the uninspected tail.

```text
# Phase A — decide + apply.
accepted: list[_PendingAdmit] = []
while self._capacity() > 0 and self._waiting_queue:
    pending = self._waiting_queue.popleft()
    req_id = f"req-{pending.req_index}"

    if self._budgeter is None:
        accepted.append(pending)
        continue

    decision = self._budgeter.admit(
        req_id=req_id,
        n_prompt=len(pending.prompt_ids),
        max_tokens=pending.params.max_tokens,
    )
    match decision:
        case RejectDecision():
            events.append(BatchEvent.aborted(
                pending.req_index, decision.reason
            ))
            self.aborts += 1
            continue
        case AdmitAfterEvictDecision(n_blocks, delta):
            try:
                self._apply_evict(n_blocks)
            except _BudgetEvictUnderrun:
                events.append(BatchEvent.aborted(
                    pending.req_index, "budget-exhausted"
                ))
                self.aborts += 1
                continue
            self.evictions += 1
            reserved_delta = delta
        case AdmitAfterPreemptDecision(victim_req_id, delta):
            if pending.is_replay:
                # B-9 anti-ping-pong: replay cannot use preempt.
                if self._rows:
                    self._waiting_queue.appendleft(pending)
                    break  # only current pending requeued; suffix
                           # of queue untouched because we didn't
                           # pre-pop
                events.append(BatchEvent.aborted(
                    pending.req_index, "budget-exhausted"
                ))
                self.aborts += 1
                continue
            if not self._apply_preempt(victim_req_id):
                # B-7: victim missing / race. Requeue this admission
                # and break; next step retries.
                self._waiting_queue.appendleft(pending)
                break
            self.preempts += 1
            reserved_delta = delta
        case AdmitDecision(delta):
            reserved_delta = delta

    # Commit reservation BEFORE the next iteration's admit() runs (B-8).
    self._budgeter.apply_admit(req_id, reserved_delta)
    accepted.append(pending)

# Phase B — execute, preserving 16c.2's hit/miss grouping.
hit_rows: list[tuple[_PendingAdmit, int]] = []
miss_rows: list[_PendingAdmit] = []
if self._prefix_cache is None:
    miss_rows = list(accepted)
else:
    block_size = self._prefix_cache.block_size
    for pending in accepted:
        raw = self._prefix_cache.peek(pending.prompt_ids)
        max_aligned = max(
            0, ((len(pending.prompt_ids) - 1) // block_size) * block_size
        )
        usable = min(raw.num_hit_tokens, max_aligned)
        if usable == 0:
            miss_rows.append(pending)
        else:
            hit_rows.append((pending, usable))

for pending, usable in hit_rows:
    try:
        events.extend(self._admit_single_hit_row(pending, usable))
    except Exception:
        # Only this row failed; others in `accepted` remain committed.
        if self._budgeter is not None:
            self._budgeter.release(f"req-{pending.req_index}")
        raise

if miss_rows:
    try:
        events.extend(self._admit_miss_cohort(miss_rows))
    except Exception:
        # Miss cohort is atomic — one forward per cohort — so release
        # every miss row's reservation together.
        if self._budgeter is not None:
            for pending in miss_rows:
                self._budgeter.release(f"req-{pending.req_index}")
        raise

self._rebuild_slot_table()
```

The explicit try / release pairing per body guarantees **B-6'**
(see §5): every `apply_admit` has a matching `release` on any
terminal outcome, including mid-admission exceptions. Rows that
successfully enter `self._rows` keep their reservations; failures
release only their own row's reservation (single-hit) or the whole
cohort (miss).

On normal row termination, `_reclaim_terminated` gains a hook that
calls `budgeter.release(req_id)` for each terminal row **before**
the filter / cache drop (same precondition ordering as 16c.2's
`_extract_and_insert_prefix`).

### 4.3 Evict helper

```python
def _apply_evict(self, n_blocks: int) -> None:
    assert self._prefix_cache is not None  # decision implies cache present
    freed = self._prefix_cache.evict_until(n_blocks)
    if freed < n_blocks:
        # Policy assumed more evictable blocks than the tree could
        # actually supply (live-hit churn between peek and apply).
        # Surface as a runtime error — the admission cannot proceed
        # safely. 16d policy: fall back to reject for this one
        # admission; caller sees BatchEvent.aborted.
        raise _BudgetEvictUnderrun(
            f"evict_until freed {freed} of {n_blocks}"
        )
```

Wrapped in a try/except in the caller so we can convert the underrun
to a reject + aborted event for the triggering admission.
**Invariant B-2: evict underrun never corrupts state** — the caller's
try/except catches it, bumps `self.aborts`, emits aborted event, and
continues.

### 4.4 Preempt helper + anti-ping-pong

`_PendingAdmit` grows one field:

```python
@dataclass(frozen=True)
class _PendingAdmit:
    req_index: int
    prompt_ids: tuple[int, ...]
    params: SamplingParams
    is_replay: bool = False     # NEW — set when re-enqueued by preempt.
```

`is_replay=True` tells `_admit_waiting_requests` this admission MUST
NOT use the preempt mechanism (B-9). `add_request` keeps the default
`False`; only `_apply_preempt` sets it to `True`.

```python
def _apply_preempt(self, victim_req_id: str) -> bool:
    """Return True iff the victim was found, preempted, and re-queued.
    Caller must NOT apply_admit() for the triggering admission if this
    returns False — the admission has to requeue at queue front and
    wait for the next step.
    """
    if self._batch_cache is None:
        return False
    victim_row_idx = self._find_row_by_req_id(victim_req_id)
    if victim_row_idx is None:
        return False       # B-7: victim missing (raced / already terminal).
    victim_row = self._rows[victim_row_idx]

    # 1. Extract prefix K/V into the prefix cache. Safe when
    #    prefix_cache is None — no-op; preempt without prefix cache
    #    means losing the victim's prompt work (that's OK; prefix
    #    cache is an optimisation).
    if self._prefix_cache is not None:
        self._extract_and_insert_prefix(victim_row_idx)

    # 2. State transition: PREFILL/DECODE → PREEMPTED → WAITING.
    #    Per §4.6 scope note: this transition is state-machine
    #    correctness only; the _BatchRow object is dropped and the
    #    re-enqueued _PendingAdmit does not carry it forward. For
    #    global-attention adapters (P-2 scope) there is no
    #    state_delta_snapshot to preserve.
    victim_row.state.transition(
        RequestStatus.PREEMPTED, reason="budget-preempt"
    )
    victim_row.state.transition(
        RequestStatus.WAITING, reason="re-admit"
    )

    # 3. Drop the victim from the batch cache via filter.
    kept = [i for i in range(len(self._rows)) if i != victim_row_idx]
    if not kept:
        self._batch_cache = None
        self._rows = []
    else:
        for layer_cache in self._batch_cache:
            layer_cache.filter(kept)
        self._rows = [self._rows[i] for i in kept]
    self._rebuild_slot_table()

    # 4. Budgeter release — the victim's reservation goes back to
    #    the pool so the triggering admission's apply_admit lands
    #    in consistent accounting.
    if self._budgeter is not None:
        self._budgeter.release(victim_req_id)

    # 5. Re-enqueue the victim at queue front with the replay flag.
    #    Its composite prompt includes every generated token so the
    #    re-admission resumes without token duplication (B-5).
    #    max_tokens is updated to max_tokens - len(generated) so the
    #    worst-case reservation equals the original (Q-2 algebra).
    composite_prompt = (
        list(victim_row.prompt_ids) + list(victim_row.generated)
    )
    remaining = (
        victim_row.params.max_tokens - len(victim_row.generated)
    )
    if remaining <= 0:
        # Edge case: the victim was at max_tokens when preempt fired.
        # It should have terminated naturally; picking it as a victim
        # is a budgeter bug. Loud-fail.
        raise AssertionError(
            f"preempt victim {victim_req_id}: remaining tokens = "
            f"{remaining} (victim should have been DONE)"
        )
    replay_params = victim_row.params.model_copy(
        update={"max_tokens": remaining}
    )
    self._waiting_queue.appendleft(
        _PendingAdmit(
            req_index=victim_row.req_index,
            prompt_ids=tuple(composite_prompt),
            params=replay_params,
            is_replay=True,
        )
    )
    return True
```

**Invariant B-3: preempt preserves per-row event semantics.** Every
token the victim emitted before preempt stays emitted; re-admission
picks up from the next-to-sample position. Tokens are never
duplicated, never skipped, never reordered within a single
`req_index`'s event stream.

**Invariant B-4: preempt is visible only via counters, not events.**
Per skeleton Q-E (step 4), `RequestState` has a PREEMPTED status but
`BatchEvent.kind` does NOT gain a "preempted" variant. Callers see a
natural pause in token emission for that `req_index`, then resumed
emission once re-admission fires. `batcher.preempts` is the
acceptance-testable observation surface.

**Invariant B-9: replay admissions never use preempt.** When an
`AdmitAfterPreemptDecision` is returned for a pending marked
`is_replay=True`, the batcher falls back:
  - if there ARE active rows → requeue at queue front, break out of
    the admit loop (retry next step after some row terminates).
  - if there are NO active rows → emit aborted with
    `finish_reason="budget-exhausted"`. The victim genuinely cannot
    fit under the current cap; no amount of preempting will help.

This breaks the ping-pong: a fresh admission can force a preempt,
but a replay cannot. After enough fresh admissions terminate
naturally, the replay's `AdmitDecision` or `AdmitAfterEvictDecision`
branches succeed and re-admission completes.

### 4.5 Terminal handling for aborted admissions

When `budgeter.admit` returns `RejectDecision`, we emit
`BatchEvent.aborted(req_index, reason)` and drop the request. No
`_BatchRow` is ever created. The request's original
`_PendingAdmit` was popped from `_waiting_queue` earlier in the loop
and is simply not re-enqueued. Terminal from the caller's
perspective: the event stream for that `req_index` ends on the
aborted event.

---

### 4.6 Scope: `RequestState` continuity across preempt

16d's preempt-then-re-admit cycle **discards the victim's
`_BatchRow`** (and the `RequestState` it wrapped) and builds a fresh
row on re-admission from the `_PendingAdmit`. Consequence:

- `RequestState._history`, `arrival_time`, and `first_token_time` do
  not carry across the preempt boundary. Re-admission starts a fresh
  `RequestState` with `WAITING → PREFILL → ...`.
- `state_delta_snapshot` (D-015, for hybrid-DeltaNet / recurrent
  layers) is NOT preserved. For P-2's global-attention adapters
  (Qwen3, Qwen3.5-dense path) this is fine — `state_delta_snapshot`
  is always `None` for those.

The PREEMPTED → WAITING transitions on the OLD state still fire for
state-machine correctness (the old state moves cleanly through its
terminal-ish path). A separate test pins this as
**`test_preempt_transitions_victim_PREEMPTED_then_WAITING`** in §6.

**Resume representation is the composite prompt alone**, not the old
state object. Anything else — a continuity for `state_delta_snapshot`,
per-request accumulated metrics, or an observable "this is a
resumption" flag — is explicitly deferred. When P-3 wires recurrent
states, it will extend `_PendingAdmit` with a
`resume_snapshot: StateDelta | None` field and teach `_apply_preempt`
to hand it off. 16d does not pre-empt that design (pun intended).

---

## 5. Invariants (B-1 … B-9)

- **B-1 (no-budgeter bit-identical).** When `budgeter=None`, the
  batcher behaves exactly as 16c.2. Existing 466-test suite stays
  green without modification.

- **B-2 (evict underrun never corrupts state).** If
  `evict_until` frees fewer blocks than the decision asked for, the
  admission falls back to an aborted event; `self._rows` /
  `self._batch_cache` / `self._waiting_queue` are untouched.

- **B-3 (preempt preserves per-row event semantics).** Within one
  `req_index`'s BatchEvent stream, tokens are never duplicated or
  lost across a preempt → re-admit cycle. The last token emitted
  before preempt is the last real output before the pause; the first
  token after re-admission is the next in sequence.

- **B-4 (preempt is counter-observable, not event-observable).**
  `BatchEvent.kind` stays in {token, done, aborted}. Callers
  observing a single req_index's stream cannot distinguish
  "preempted and re-admitted" from "slow step" — only
  `batcher.preempts` surfaces it.

- **B-5 (re-admit uses composite prompt with generated included).**
  `composite_prompt = prompt_ids + generated_so_far` — full list, no
  `[:-1]` trim. The prefix cache holds K/V for `prompt_ids +
  generated[:-1]` (from `_extract_and_insert_prefix`); the suffix
  prefill on re-admit processes the one last generated token whose
  K/V was never computed. Matches what a continuing decode step
  would have done. `max_tokens` on the replay's params is
  `original - len(generated)` so the worst-case reservation equals
  the original (Q-2 algebra).

- **B-6' (reservation / release pair symmetry).** Every
  `budgeter.apply_admit(req_id, ...)` is paired with exactly one
  `budgeter.release(req_id)`, on one of these paths:
    (a) terminal row in `_reclaim_terminated` (DONE / ABORTED);
    (b) victim of `_apply_preempt` before re-enqueue;
    (c) `except` clause around `_admit_single_hit_row` or
        `_admit_miss_cohort` when the admit body raises after
        `apply_admit` has committed.
  No path may skip (c) or double-release. `RejectDecision` flows do
  NOT participate — they never `apply_admit` in the first place.

- **B-7 (preempt victim selection matches budgeter's choice).** The
  batcher does NOT second-guess the budgeter's victim; it uses
  `AdmitAfterPreemptDecision.preempt_req_id` as-is. If the req_id
  is not in `self._rows` (e.g. terminated same-step post-decision),
  `_apply_preempt` returns `False`; the triggering admission is
  requeued at queue front and retries on the next step. The race
  window is small because `admit` is the first step phase and
  `_rows` mutates only at specific points, but the recovery path
  is explicit rather than "assume it works".

- **B-8 (reservation-timing: apply_admit before next admit).**
  Within one `_admit_waiting_requests` loop, every admission's
  `apply_admit` commits to the budgeter's `reserved_bytes` BEFORE
  the next iteration's `admit()` call runs. Otherwise a batch of
  small pendings would each see the same initial headroom and all
  pass individually while their sum exceeds the cap. This is why
  `apply_admit` is tied to the decision site, not to the hit/miss
  admit body's completion.

- **B-9 (replay admissions never preempt).** A `_PendingAdmit` with
  `is_replay=True` (re-enqueued from a prior preempt) that receives
  an `AdmitAfterPreemptDecision` falls through to:
    - requeue at front if `self._rows` is non-empty (wait for
      natural termination);
    - abort with `"budget-exhausted"` if `self._rows` is empty
      (genuinely unfittable).
  This breaks the ping-pong cycle where two replay victims
  repeatedly evict each other. Only fresh (non-replay) admissions
  can trigger `_apply_preempt`.

---

## 6. Test plan

### 6.1 `tests/test_memory_budgeter.py` — refactor-only updates

No new tests; existing ones rewrite fixture:

```python
pc = RadixPrefixCache(
    block_size=kv.block_size, store=PagedPrefixBlockStore(kv)
)
bpt = _bytes_per_token_from_kv(kv)
b = MemoryBudgeter(
    prefix_cache=pc,
    weights_bytes=weights_bytes,
    bytes_per_token=bpt,
    block_size=kv.block_size,        # was kv=kv
    cap_bytes=cap_bytes,
)
```

All 24 existing tests stay green.

### 6.2 `tests/test_batcher.py` — new (16d-2 through 16d-4)

Grouped by sub-commit:

**Sub-commit 2 — admit + reject + reservation timing:**
```
test_budgeter_none_matches_16c2_behaviour              # B-1
test_admit_decision_proceeds_as_before
test_admit_decision_commits_reservation_before_next_admit  # B-8
test_admit_body_raise_releases_reservation                 # B-6'(c)
test_reject_decision_emits_aborted_event
test_reject_does_not_create_batch_row
test_reject_increments_aborts_counter
test_reject_does_not_call_apply_admit                      # B-6' / Q-4
```

**Sub-commit 3 — evict:**
```
test_evict_decision_calls_evict_until_with_correct_n
test_evict_then_admit_proceeds_normally
test_evict_increments_evictions_counter
test_evict_underrun_falls_back_to_reject                # B-2
test_evict_underrun_leaves_rows_and_cache_unchanged     # B-2
```

**Sub-commit 4 — preempt + re-admit:**
```
test_preempt_helper_returns_true_on_success            # helper contract
test_preempt_helper_returns_false_on_missing_victim    # B-7
test_preempt_missing_victim_requeues_triggering_admission  # B-7 flow
test_preempt_transitions_victim_PREEMPTED_then_WAITING
test_preempt_drops_victim_row_via_filter
test_preempt_extracts_prefix_before_filter              # B-3 + prefix reuse
test_preempt_reenqueues_victim_at_queue_front_with_replay_flag
test_preempt_composite_prompt_includes_generated        # B-5
test_preempt_replay_max_tokens_decremented              # Q-2 algebra
test_preempt_no_prefix_cache_still_works                # preempt is optional-prefix-safe
test_preempt_increments_preempts_counter
test_preempt_emits_no_event_kind_beyond_taxonomy        # B-4
test_preempt_victim_at_max_tokens_is_assertion_error    # loud-fail guard
test_readmit_after_preempt_resumes_token_sequence       # B-3 bit-exact
test_readmit_hits_prefix_cache_populated_by_preempt     # B-5 mechanism
test_replay_pending_never_triggers_preempt             # B-9
test_replay_with_no_active_rows_aborts                  # B-9 termination
test_replay_with_active_rows_requeues_at_front          # B-9 wait path
test_no_preempt_ping_pong_across_many_steps             # B-9 end-to-end
```

### 6.3 Real-model acceptance (`tests/test_p2_batched_parity.py`)

```
test_budget_overflow_aborts_cleanly                     # PLAN §7 #3
```

Setup: Qwen3-0.6B + tight `cap_bytes` (e.g. 2 * per-request worst-
case). Admit 3 requests with `max_tokens` picks that make the 3rd
exceed the cap. Assertions:

- Requests 0-1 produce normal token + done events.
- Request 2's admission triggers preempt (request 1 is FIFO-newest
  at that point). Request 1 gets re-queued; its token stream
  resumes after request 2 finishes.
- `batcher.preempts >= 1`, `batcher.aborts == 0`.
- OR — if cap is tight enough — requests 2 triggers reject;
  `batcher.aborts >= 1`, `aborted` event emitted with
  `finish_reason == "budget-exhausted"`.

The parametrisation of cap_bytes chooses which branch (preempt vs
reject) fires, so both paths have coverage in a single test file.
PLAN §7 #3's wording "queues cleanly" is satisfied by the preempt
branch (work is resumed, not lost); "aborts cleanly" is satisfied by
the reject branch.

---

## 7. Non-goals

- Per-row `SamplingParams` preempt priorities (heterogeneous batches
  are P-3; all 16d victims are selected by FIFO-newest).
- Preempt-then-evict compound decisions (§MemoryBudgeter policy
  §4 explicitly rejects these — "at most one mechanism per
  admission" — and 16d inherits the simplification).
- Cascaded preemption (preempting more than one victim to admit one
  request). Same rationale.
- ABORTED state surfacing in events beyond `finish_reason`
  (B-4 keeps the event taxonomy flat).
- Speculative-decode rollback integration (P-7's concern — 16d only
  ensures `BatchEvent.aborted` is wired so P-7 can reuse it).
- Dynamic `cap_bytes` tuning at runtime (budgeter construction is
  static per batcher; runtime cap changes are deferred to a future
  `BudgetPolicy` refactor).
- **Continuity of `RequestState` across preempt.** 16d discards the
  victim's `_BatchRow` + `RequestState` and builds a fresh state on
  re-admission. `state_delta_snapshot` (D-015) is not preserved;
  for P-2's global-attention scope this is fine (snapshot is always
  None). Recurrent-state preservation is P-3's concern — see §4.6.

---

## 8. Open questions

### Q-1. Where does the batcher get `bytes_per_token` for MemoryBudgeter?

The batcher knows `num_layers`, `n_kv_heads`, `head_dim`, `dtype_bytes`
via `adapter.config` + `adapter.kv_layout()`. `bytes_per_token = 2 *
num_layers * n_kv_heads * head_dim * dtype_bytes`. The budgeter also
still needs a `prefix_cache` (for `_count_evictable_prefix_blocks`)
and a `block_size` — the latter naturally comes from
`prefix_cache.block_size`, so we do NOT take it as a separate arg.

Proposed signature:

```python
@classmethod
def for_adapter(
    cls,
    adapter: ModelAdapter,
    *,
    prefix_cache: RadixPrefixCache,
    weights_bytes: int,
    cap_bytes: int,
) -> MemoryBudgeter:
    layout = adapter.kv_layout()
    dtype_bytes = layout.dtype.size
    bytes_per_token = (
        2 * layout.num_layers * layout.n_kv_heads
        * layout.head_dim * dtype_bytes
    )
    return cls(
        prefix_cache=prefix_cache,
        weights_bytes=weights_bytes,
        bytes_per_token=bytes_per_token,
        block_size=prefix_cache.block_size,
        cap_bytes=cap_bytes,
    )
```

The batcher does NOT call this itself — construction stays
caller-explicit (mirrors 16c.2's `prefix_cache` ownership rule).
The factory exists so tests and harnesses don't each recompute
`bytes_per_token`.

### Q-2. What if a preempt victim's `generated_so_far` pushes the
composite prompt length past `max_tokens`?

Composite prompt length = `len(prompt_ids) + len(generated_so_far)`.
The budgeter's worst-case math reserves `(n_prompt + max_tokens) *
bytes_per_token`. For re-admission with the composite prompt,
`n_prompt_new = len(composite_prompt)` and the remaining token
budget is `max_tokens - len(generated_so_far)`. The new reservation
is then:

```text
(len(composite_prompt) + max_tokens_remaining) * bytes_per_token
 = (len(prompt_ids) + len(generated_so_far) +
    max_tokens - len(generated_so_far)) * bytes_per_token
 = (len(prompt_ids) + max_tokens) * bytes_per_token
```

Which is **exactly the original reservation**. Nice algebraic
invariant — re-admission charges the same bytes as the original
admission, so if the pressure that caused preempt is resolved, the
victim can be re-admitted immediately without a second preempt.

Proposed: 16d-4 passes `max_tokens_remaining = params.max_tokens -
len(generated_so_far)` to the budgeter on re-admit.
`SamplingParams` is a Pydantic `BaseModel` with
`ConfigDict(frozen=True)` (not a stdlib dataclass), so the
substitute is produced via:

```python
replay_params = victim_row.params.model_copy(
    update={"max_tokens": remaining}
)
```

Store the updated params on the re-enqueued `_PendingAdmit`.

**Not sufficient alone.** The algebra proves re-admission costs the
same bytes, not that the headroom at re-admission time is enough.
Without an anti-ping-pong rule the same two victims can keep
preempting each other. See B-9 for the wait / abort fallback.

### Q-3. Should preempt transition PREFILL or DECODE rows differently?

A PREFILL row has `generated == []` (no tokens sampled yet) — the
composite prompt equals the original prompt. A DECODE row has
`generated != []`. The preempt code path is the same for both;
only B-5's math collapses trivially for PREFILL.

Proposed: no special case. The helper handles both uniformly.

### Q-4. What about the budgeter's `release` on abort?

The aborted request was never `apply_admit`'d (reject happens before
admission commits). So there's no reservation to release. 16d's
abort flow does NOT call `release`. Verified by
`test_abort_releases_no_budgeter_reservation`.

For the preempt path, the victim was `apply_admit`'d when originally
admitted. Preempt calls `release` on the victim's req_id; re-admit
later calls `apply_admit` again with the updated reservation.

---

## 9. Implementation order within 16d

Each sub-commit pauses for explicit approval:

1. **16d-1** — MemoryBudgeter ctor change + `for_adapter` factory +
   test fixture update. 5 mins; foundational.
2. **16d-2** — batcher kwarg + admit/reject wiring + 6 tests. 30 mins.
3. **16d-3** — evict wiring + 5 tests. 30 mins.
4. **16d-4** — preempt + re-admit wiring + 11 tests + real-model
   acceptance. 60 mins; biggest sub-commit.

Total estimate: ~2 hours if there are no kernel-level surprises
(no new Gate probes expected — preempt reuses the reclaim physical
path, and the state surface was already exercised by 16c.1's filter
tests).

---

## 10. Ready-to-start checklist

- [x] 16c.2 merged (`4460541`).
- [x] 466-test regression suite green on HEAD.
- [x] PLAN §7 P-2 acceptance #1 + #2 closed.
- [x] Round-2 revision: (1) reservation-timing B-8 with
      apply_admit-before-next-admit; (2) anti-ping-pong B-9 via
      `is_replay` flag on `_PendingAdmit`; (3) `_apply_preempt`
      returns bool, callers branch on it; (4) §4.6 scope note on
      RequestState discontinuity (state_delta_snapshot deferred to
      P-3). Test plan grew accordingly (2026-04-17).
- [x] Round-3 revision: (1) §4.2 rewritten as two-phase
      (decide/apply per-pending, then grouped hit/miss execute) so
      miss-cohort batching is preserved under budgeter enabled;
      (2) pop-one-at-a-time with `while ... popleft()` — no
      pre-batch pop, so mid-loop `break` only requeues the
      current pending; (3) Q-1 `for_adapter` signature takes
      `prefix_cache` (not just adapter) and derives `block_size`
      from `prefix_cache.block_size`; (4) Q-2 + §4.4 use Pydantic
      `params.model_copy(update={"max_tokens": remaining})`, not
      stdlib `dataclasses.replace` — SamplingParams is a Pydantic
      `BaseModel` with `ConfigDict(frozen=True)` (2026-04-17).
- [ ] This revised prep doc approved.
- [x] Q-1 / Q-2 / Q-3 / Q-4 resolved; Q-2 gets the B-9 supplement
      and the model_copy syntax correction.

On approval, start 16d-1. Est. first commit within 10 minutes of the
green light.
