# P-2 Unit 16c — prep: slot-table invariants + sub-split

**Date:** 2026-04-17
**Status:** design prep — no code changes yet.
**Depends on:** Unit 16b (`25c47a8`), Gate-0.5 (`64e2211`).
**Blocks:** Unit 16c implementation, Unit 16d, P-2 acceptance.

## Why this document

Unit 16c has a larger state space than 16a/16b: every step can add
rows (via `BatchKVCache.extend`) and remove rows (via
`BatchKVCache.filter`). Row indices are **not** stable under `filter`
(proven empirically by Gate-0.5). Without a written-down invariant
ledger and an explicit phase order for the step loop, 16c implementation
tends to slide into "fix-a-bug, discover-another, fix-that" loops where
each patch subtly breaks a previously-held invariant. This doc pins
the invariants in one place and proposes a sub-split that keeps the
admit-lifecycle work separate from the prefix-K/V-materialisation
work.

---

## 1. Slot-table invariants (the five promises 16c must keep)

The batcher maintains a monotonic truth:

```text
req_index (stable, assigned at admit)  ←→  row (mutable, bookkeeping)
```

The five invariants below are what the scheduler relies on. Any code
change that would violate one must update the ledger first.

### I-1. Domain: ``slot_table: dict[req_index, int]``

``slot_table[r.req_index]`` is the **current** row index of request
``r`` inside every ``BatchKVCache[layer]`` and inside ``self._rows``.
It is rebuilt after every ``filter`` and extended after every ``extend``.

``req_index`` is assigned once at ``add_request`` time and **never
changes** — it is what ``BatchEvent.req_index`` refers to through the
whole request lifetime. The row index is volatile.

### I-2. ``extend`` appends; existing rows' indices are preserved

After ``main.extend(new_b1_cache)`` where ``main`` has ``B = B_prev``
and ``new_b1_cache`` has ``B = K``:

- rows ``0..B_prev-1`` keep their row index (and slot_table entries)
- new rows occupy indices ``[B_prev, B_prev+1, …, B_prev+K-1]``

Gate-0.5 probe verified this behaviour. We rely on it so
slot_table only needs delta updates (add K entries), not full rebuild,
on admit.

### I-3. ``filter([kept])`` reshuffles to ``0..len(kept)-1``

After ``main.filter(kept)``:

- physical row at old index ``kept[i]`` is now at row ``i``
- all other rows are **dropped** (their K/V is gone — proven by
  Gate-0.5 Q5)

Consequence: slot_table must be **rebuilt from scratch** after every
``filter`` call. There is no delta form. Implementation:

```python
def _rebuild_slot_table_after_filter(self, kept: list[int]) -> None:
    old_rows = self._rows
    self._rows = [old_rows[i] for i in kept]
    self._slot_table = {r.req_index: i for i, r in enumerate(self._rows)}
```

### I-4. ``_idx`` advances monotonically; extend adjusts via left_padding

``BatchKVCache._idx`` only grows (via ``update_and_fetch``) or stays
equal (via ``extend`` when ``other._idx <= self._idx``). It is
**never reduced** except by ``filter``'s shift-left optimisation when
``min(new_left_padding) > 0`` — and even then, every row's effective
sequence length (``offset[i] = _idx - left_padding[i]``) is preserved.

On ``main.extend(new_b1)`` where ``main._idx = X`` and ``new_b1._idx
= Y ≤ X``: ``new_b1``'s rows have their ``left_padding`` increased by
``X - Y`` before concatenation, so their effective offset stays at
``Y`` while ``main._idx`` stays at ``X``. This is what lets a
freshly-prefilled row join an existing cohort that's already
``X - Y`` tokens ahead.

### I-5. Termination triggers reclaim before the next admit, not immediately

When a row hits DONE / ABORTED inside a sample phase, the row is
**not** filter()'d in the same step. It stays in the batch fed a pad
token (16b's behaviour) until the **next** step's reclaim phase runs
``filter([kept])`` and rebuilds slot_table.

Why deferred: immediate in-step filter would require re-running
subsequent phases with a reshuffled slot_table, adding an ordering
hazard. Deferred reclaim makes every step have a deterministic
linear structure (reclaim → admit → forward).

**Loop predicate distinction (resolved from review round 1).** Because
reclaim is deferred, ``has_active()`` alone is wrong as the Engine's
loop condition: the last sample phase that terminates every row
leaves ``self._rows`` populated with terminal-pending-reclaim entries;
``has_active()`` returns False and the Engine exits before reclaim
runs. The batcher therefore exposes two predicates:

```python
def has_active(self) -> bool:
    """True iff at least one row is non-terminal. Literal semantics;
    used by internal phase decisions."""
    return any(not r.state.is_terminal for r in self._rows)

def has_work(self) -> bool:
    """True iff anything remains that the next step could do:
    reclaim pending, admission pending, or active rows present.
    ``Engine.generate_batch`` loops on this."""
    return bool(self._waiting_queue) or bool(self._rows)
```

``Engine.generate_batch`` uses ``while batcher.has_work()``, so
cohort-drain completes: last sample phase terminates rows → next
step runs reclaim → self._rows becomes empty → has_work() returns
False → loop exits cleanly.

---

## 2. Step-loop structure for 16c

Three phases per ``step()`` call, in order:

```text
┌───────────────────────────────────────────────────────────┐
│ Phase 1 — Reclaim                                         │
│   if no row is terminal: skip                             │
│   16c.2 only: eager-extract pinned blocks → _detached     │
│   compute kept = [i : not self._rows[i].state.is_terminal]│
│   if kept is []:                                          │
│     self._batch_cache = None    # FIX 2 — can't extend    │
│     self._rows, self._slot_table = [], {}                 │
│   else:                                                   │
│     for each layer: batch_cache[layer].filter(kept)       │
│     self._rows = [self._rows[i] for i in kept]            │
│     self._rebuild_slot_table()                            │
├───────────────────────────────────────────────────────────┤
│ Phase 2 — Admit                                           │
│   K = pop up to (max_batch_size − len(self._rows)) from   │
│       waiting_queue                                       │
│   if K == 0: continue to Phase 3                          │
│   16c.1: no prefix cache touched.                         │
│   16c.2: peek() per row → `prefix_hit_tokens`; seed       │
│          per-layer plain KVCache via detached K/V         │
│          (state setter) for hit_tokens positions.         │
│   build BatchKVCache(left_padding=[k_max − len_i]) per    │
│     layer with K rows (FIX 5 — direct construction, no    │
│     merge-of-empty-kvcaches)                              │
│   one batched forward over left-padded (K, k_max_len)     │
│     prompt tokens (or, in 16c.2, only the non-hit suffix) │
│   sample + emit first token per admitted row; transition  │
│     state                                                 │
│   stitch into main cache:                                 │
│     if main is None: self._batch_cache = k_batch_cache    │
│     else: for each layer: main[layer].extend(k_batch_...) │
│   self._rows.extend(admitted); rebuild slot_table         │
│                                                           │
│   If any admission happened in this step, return from     │
│   step() WITHOUT running Phase 3 — existing DECODE rows   │
│   idle this step. This keeps prefill and decode in        │
│   separate forwards with compatible T.                    │
├───────────────────────────────────────────────────────────┤
│ Phase 3 — Decode                                          │
│   (only runs if Phase 2 admitted nothing this step)       │
│   if not self.has_active(): return                        │
│   tokens = self._build_decode_tokens()  # (B, 1)          │
│   logits = forward_batched(model, tokens, batch_cache)    │
│   emit per-row tokens, transition DONE as needed          │
└───────────────────────────────────────────────────────────┘
```

**Important timing property**: when Phase 2 admits, existing DECODE
rows stall for that step. Throughput cost is bounded by admission
frequency; correctness is preserved because no row's K/V advances
without its corresponding query.

**Alternative rejected**: "one forward per step regardless of mix" via
chunked prefill (vLLM v1). Requires ragged T in batch; mlx-lm's
``BatchKVCache`` does not support it. Deferred to P-4 Q-010.

---

## 3. Sub-split: 16c.1 + 16c.2

### 16c.1 — Dynamic cohort: queue + reclaim + slot_table (no prefix cache)

**One-line scope (resolved from review round 1):** dynamic waiting
queue + deferred reclaim + slot_table correctness; **prefix cache is
not exercised at all**. The PagedKVCache / RadixPrefixCache modules
remain unwired in 16c.1 — there is no lookup, no insert, no
``_live_hits`` touch. Compressing 16c.1 to this frame keeps its
failure surface purely about state management and rules out two
confounders that would otherwise muddy debugging (refcount leaks and
block-id lifecycle).

**Deliverables**:

- `ContinuousBatcher._waiting_queue`: deque[_PendingAdmit] with
  `(req_index, prompt_ids, params)` triples.
- `ContinuousBatcher._admission_closed` removed (mid-run admission is
  the whole point). `add_request` permitted at any step.
- `has_active()` + `has_work()` predicates per I-5.
- Phase 1 reclaim: `filter([kept])` + slot_table rebuild; includes the
  `kept == []` branch that fully resets `self._batch_cache = None`
  (see Fix 2 below — extend-into-stale-cache is a wrong default).
- Phase 2 admit: construct a fresh `BatchKVCache(left_padding=[...])`
  directly per the newly-admitted K rows' prompt lengths (mirrors
  16b's prefill), then extend into main (or replace main when it is
  None).
- Phase 3 decode as in 16b.
- `_BatchRow.prefix_hit_tokens` field reserved (int, default 0) but
  never populated in 16c.1. Tests assert it stays 0.

**Acceptance**:

- Mid-run admit: cohort of 4, 1 row finishes, 5th admits into the
  freed slot, batch size returns to 4 with correct row order.
- Queue-bounded admission: `generate_batch(8 prompts)` with
  `max_batch_size=4` → first 4 admit at step 0, others admit as
  earlier ones finish. All 8 eventually DONE.
- slot_table coherence across filter: after filter, every remaining
  row's req_index maps via slot_table to a row whose `generated`
  matches expected.
- Cohort-drain: after the last admitted row terminates, one more
  step runs reclaim, then `has_work() == False`.
- PLAN acceptance #1 ("8 concurrent requests run stably") achievable
  here.

**Explicitly NOT in 16c.1**: prefix cache of any kind; eager-extract;
detached K/V; hit counter; block-id synthesis. PLAN acceptance #2
waits on 16c.2.

### 16c.2 — Prefix cache integration + K/V materialisation (option B)

**Scope**: wire `RadixPrefixCache` into the admit path end-to-end
with real K/V reuse. This is the full option B from P2_OPENING v2.1.

**Deliverables**:

- **Block-id source**: synthesise monotonic block ids per
  newly-admitted row as chunks of `block_size` of its prompt. These
  ids are metadata handles — they do NOT correspond to PagedKVCache
  physical pages (which aren't used on the BatchKVCache path).
  `_BatchRow.block_ids: list[int]` records the row's ids.
- **Insert on terminate (before filter)**: for each terminating row,
  slice its batched K/V via `BatchKVCache.extract(row_idx)`, split
  into per-block K/V (one slice of `block_size` tokens per id), and
  insert into RadixPrefixCache with those ids. Also populate
  `RadixPrefixCache._detached[block_id] = list[KVCache]` (per layer).
- **Peek vs lookup**: add `RadixPrefixCache.peek(tokens) -> PrefixHit`
  — identical walk to ``lookup`` but **no refcount / `_live_hits`
  mutation**. Admit phase of 16c.2 calls `peek` to compute
  `prefix_hit_tokens` without pinning; then `lookup` at the moment
  it actually needs the K/V to copy. This keeps admit-time
  bookkeeping clean and avoids the refcount-leak trap the review
  flagged.
- **Seed flow**: when `prefix_hit_tokens > 0`, build per-layer
  plain `KVCache()` seeded with the detached K/V via
  `cache.state = (prefix_k, prefix_v)` so `cache.offset == hit_len`.
  Then run remaining prefill on `prompt_ids[hit_len:]`; then
  merge → B=1 BatchKVCache, extend into main.
- **Detached storage lifecycle (resolved from review round 1)**:
  `_detached[block_id]` is tied to the radix **tree node's lifetime**,
  NOT to the live-hit pin. `release()` decrements the live-hit
  counter (safe after a request has copied its seed K/V out); only
  `evict_until()` — which actually removes the radix node — drops
  the `_detached` entry. This keeps prefix K/V available across
  many request admissions.

**Acceptance**:

- `peek()` does not change `_live_hits` or `kv._refcount`.
- Insert-on-terminate: after row A with prompt "shared prefix ..." ends,
  its prompt's block ids are in the radix tree and their detached
  K/V is recorded in `_detached`.
- Reuse with skip-prefill: 2 prompts sharing 30-token prefix.
  Request A runs full prefill of `len(A)` tokens; request B admits,
  `peek` returns `num_hit_tokens = 30`, remaining prefill covers
  only `len(B) - 30` tokens. Verified via instrumented prompt-token
  counter.
- K/V correctness: B's emitted tokens equal direct-mlx-lm B=2 with
  prefix-seeded row (within fp16 envelope).
- Detached storage survives release: first hit pins with lookup →
  `_live_hits[block] == 1`; `release()` → `_live_hits[block] == 0`
  but `_detached[block]` still present; second hit lookup succeeds
  with the same detached K/V.
- Eviction drops detached: `evict_until(n)` removes leaf nodes
  whose `_live_hits == 0`; each eviction also drops
  `_detached[block_id]` for that node.
- PLAN acceptance #2 ("shared-prefix hit verifiable") satisfied with
  the **behavioural** skip-prefill assertion, not just metadata.

---

## 4. Algorithmic details

### Admit flow (16c.1 variant — direct BatchKVCache, no merge)

Per review round 1, we construct `BatchKVCache(left_padding=[...])`
directly with the K admitted rows' per-row left-paddings — just like
16b's single-cohort prefill, but on a subset. No `merge` of empty
`KVCache` objects; `merge` is the wrong primitive here because it
expects already-prefilled single-request caches.

```python
def _admit_waiting_requests(self) -> list[BatchEvent]:
    events: list[BatchEvent] = []
    admitted_this_step: list[_BatchRow] = []

    while (
        len(self._rows) + len(admitted_this_step) < self._max_batch_size
        and self._waiting_queue
    ):
        req_index, prompt_ids, params = self._waiting_queue.popleft()
        admitted_this_step.append(
            self._build_row(req_index, prompt_ids, params)
        )

    if not admitted_this_step:
        return events  # Phase 3 decode can run this step.

    # Build a fresh B=K BatchKVCache for the new rows, directly — same
    # pattern as 16b.__prepare_cohort but on the admit subset only.
    num_layers = self._adapter.config.num_layers
    k_prompt_lens = [len(r.prompt_ids) for r in admitted_this_step]
    k_max_len = max(k_prompt_lens)
    k_left_padding = [k_max_len - n for n in k_prompt_lens]
    k_batch_cache = [
        BatchKVCache(left_padding=k_left_padding) for _ in range(num_layers)
    ]

    # Left-padded (K, k_max_len) prompt tokens.
    pad = self._pad_token_id()
    rows_2d: list[list[int]] = []
    for r in admitted_this_step:
        n_pad = k_max_len - len(r.prompt_ids)
        rows_2d.append([pad] * n_pad + list(r.prompt_ids))
    tokens = mx.array(rows_2d, dtype=mx.int32)

    # One batched prefill for the K admitted rows.
    logits = forward_batched(self._model, tokens, k_batch_cache)  # (K, V)
    # Sample + emit first tokens; transition state per row.
    events.extend(self._sample_emit_for_admits(admitted_this_step, logits))

    # Stitch into main cache. Two regimes, per review round 1 Fix 2:
    #   (a) main is empty (None) after a reclaim-to-zero: replace directly.
    #   (b) main has live rows: extend them with the K new rows.
    if self._batch_cache is None:
        self._batch_cache = k_batch_cache
    else:
        for layer in range(num_layers):
            self._batch_cache[layer].extend(k_batch_cache[layer])

    self._rows.extend(admitted_this_step)
    self._rebuild_slot_table()
    return events
```

Note: the admitted rows are appended to `self._rows` **after**
sample+emit so the _BatchRow list stays consistent with the batch row
ordering (`self._rows[-K:]` are the newly-admitted ones, matching
`self._batch_cache[layer]`'s last K rows after extend).

### Reclaim flow (with review-round-1 Fix 2 applied)

```python
def _reclaim_terminated(self) -> None:
    if not any(r.state.is_terminal for r in self._rows):
        return
    kept = [i for i, r in enumerate(self._rows) if not r.state.is_terminal]

    # 16c.2 only: eager_extract pinned blocks + insert to RadixPrefixCache
    # BEFORE filter, so the dropped rows' K/V is captured in _detached.
    # 16c.1 skips this block entirely.

    if not kept:
        # Every row terminated in the last step. main batch cache is full
        # of stale K/V that no live row owns. ``extend`` would append new
        # rows after the stale ones; its slot_table would be off by |rows|.
        # Drop the cache entirely — Phase 2 admit's `if _batch_cache is
        # None` branch will construct a fresh one for the new cohort.
        self._rows = []
        self._slot_table = {}
        self._batch_cache = None
        return

    # Partial reclaim: filter drops terminated rows and (optionally, per
    # mlx-lm's internal shift-left optimisation) compresses _idx.
    assert self._batch_cache is not None
    for layer in range(len(self._batch_cache)):
        self._batch_cache[layer].filter(kept)
    self._rows = [self._rows[i] for i in kept]
    self._rebuild_slot_table()
```

### Step loop

```python
def step(self) -> list[BatchEvent]:
    if not self._cohort_prepared:
        self._prepare_cohort()  # 16c: seed self._waiting_queue and batch_cache

    events: list[BatchEvent] = []

    # Phase 1
    self._reclaim_terminated()

    # Phase 2
    events.extend(self._admit_waiting_requests())
    if events:
        return events  # admission ran prefill; existing rows idle this step

    # Phase 3
    if not self.has_active():
        return events
    events.extend(self._decode_phase())
    return events
```

---

## 5. Test plan

### 16c.1 — scripted unit tests (no prefix cache wired)

- `has_work` vs `has_active`: immediately after a sample phase
  terminates the last active row, `has_active() == False` but
  `has_work() == True` (pending reclaim). After the next step
  (which runs reclaim only), both return False.
- Mid-run admit: B=2 cohort, row 0 hits max_tokens=1 → DONE; admit
  3rd row into the freed slot at the next step; continue. Assert
  batch size stays 2, req_index preserved, tokens correct.
- Queue-bounded admission: 8 prompts submitted with max_batch_size=4
  → first 4 admit immediately, others admit as space frees. All 8
  emit their token streams.
- slot_table coherence: after filter, assert every row's req_index
  maps via slot_table to a row whose `generated` matches expected.
- `kept == []` resets main cache: all rows terminate simultaneously,
  next step runs reclaim, `self._batch_cache is None`, and a
  subsequent admit initialises a fresh batch cache (not extend into
  stale rows).
- Admit forward-call budget: admit of K rows → exactly 1 prefill
  forward + existing DECODE rows idle that step (`forward_calls`
  increments by 1).
- `prefix_hit_tokens` stays 0 (reserved field, untouched by 16c.1).

### 16c.1 — real-model tests

- 8 concurrent on Qwen3-0.6B, max_batch_size=4: all 8 complete;
  per-row tokens equal direct-mlx-lm driven the same way (queue
  mechanics + filter don't alter the numerical path).
- Determinism: two consecutive `generate_batch(same_prompts,
  same_params)` runs produce the same event sequence per req_index.
- Cohort-drain: after the scenario ends, `has_work() == False` and
  `self._rows == []`.

### 16c.2 — scripted unit tests (adds prefix cache)

- `peek()` is side-effect free: calling `peek(tokens)` does not
  change `_live_hits` nor `kv._refcount`. `hits` counter still
  increments for observability (or only on `lookup` — TBD per
  RadixPrefixCache's existing contract; tests assert the chosen
  rule).
- Insert-on-terminate: after row A with prompt P ends, P's block
  ids are in the radix tree and `_detached` holds per-block K/V.
- Re-admit with seeded cache: new row with matching prefix admits
  and only prefills `prompt_len - hit_len` tokens (instrumented
  prompt-token counter).
- Detached-storage longevity: first hit `lookup()` → `release()`
  leaves `_detached[block]` intact; a second admit with the same
  prefix still hits and reuses.
- Eviction drops detached: `evict_until(n)` removes a leaf node
  **and** drops its `_detached` entry.

### 16c.2 — real-model tests

- 4 prompts sharing 100-token prefix: request 0 runs full prefill;
  requests 1-3 admit with 100-token prefix hit, prefill only the
  remainder. `hits >= 3`; prompt-token forward counter reduced from
  `4 * len(prompt)` to `100 + 3 * (len(prompt) - 100)`.
- Output correctness: tokens emitted by requests 1-3 match tokens
  emitted by direct-mlx-lm B=4 driven the same way (same prefix
  seeding, same decode cadence — within fp16 envelope).

---

## 6. Open questions to settle before 16c.1 implementation

After review round 1, Q-C and Q-D are resolved; Q-A and Q-B remain
minor design choices still worth pinning before coding starts.

- **Q-A (open)**: When `generate_batch(prompts, params)` is called
  with `len(prompts) > max_batch_size`, should the default
  `max_batch_size` still be `len(prompts)` (16b behaviour) or should
  the caller specify a separate `max_batch_size` knob? →
  Proposal: keep `max_batch_size = len(prompts)` default so 16b tests
  stay green; add an optional `max_batch_size: int | None` knob on
  `generate_batch` for 16c queue-bounded testing.
- **Q-B (open)**: Should `_BatchRow` carry `prefix_hit_tokens` as a
  field now (16c.1 leaves it at 0, 16c.2 populates via `peek`) or
  only add the field at 16c.2? → Proposal: field exists from 16c.1,
  default 0, explicitly untouched; 16c.1 tests assert it stays 0.
  Avoids a dataclass change in 16c.2 and makes the "was it populated?"
  question testable.
- **Q-C (resolved round 1)**: `has_active()` keeps its literal
  meaning (any non-terminal row). New `has_work()` = non-empty
  waiting queue OR any self._rows (active OR terminal-pending-reclaim).
  Engine's drain loop uses `has_work()`.
- **Q-D (resolved round 1)**: Decode phase asserts no row has status
  `WAITING`. Admit phase prefills synchronously, so the only state
  transition into DECODE happens in admit; Phase 3 decode iterates
  only over non-terminal rows (which are guaranteed to be in
  DECODE).

---

## 7. What 16c explicitly does NOT do

- ❌ Budget-driven preemption (16d; separate trigger, separate
  event kind).
- ❌ Chunked prefill (Q-010; P-4).
- ❌ Sliding-window attention (Q-013 probe).
- ❌ BatchRotatingKVCache / DeltaNet hybrid batching (P-3).
- ❌ Abort event emission for failed admission (deferred to 16d with
  `RejectDecision` handling — for 16c, admissions either succeed or
  stay in queue).

---

## 8. Sign-off checklist

Before opening Unit 16c.1 implementation, confirm:

- [ ] Five invariants (§1) understood and accepted as-is (I-5 updated
  to add the `has_active` / `has_work` distinction).
- [ ] Sub-split 16c.1 / 16c.2 approved — 16c.1 is now
  **state-management only** (no prefix cache wiring), 16c.2 is the
  full prefix-cache+K/V-reuse unit.
- [ ] Admit-phase-then-return-skip-decode trade-off acknowledged.
- [ ] Test plan coverage accepted.
- [ ] Open questions Q-A and Q-B have chosen answers (Q-C/Q-D
  already resolved in round 1).

When all five are checked, 16c.1 is ready to start.

## 9. Review round 1 (2026-04-17) — applied changes

Five concrete fixes + one caveat were raised by external review before
the first commit. Each is now reflected in the body of the doc; this
section is a locator so future readers can find the rationale without
re-reading the whole file.

| # | Concern raised | Applied at |
| - | --- | --- |
| 1 | `has_active()` misses terminal-pending-reclaim; Engine's drain loop exits prematurely | §1 I-5 footnote; §6 Q-C resolution; §2 step-loop phase 3; §5 16c.1 `has_work` test |
| 2 | `kept == []` reclaim must clear main BatchKVCache — extending into stale physical rows is wrong | §2 step-loop Phase 1; §4 Reclaim pseudocode; §5 16c.1 `kept == []` resets main cache test |
| 3 | `RadixPrefixCache.lookup()` has refcount + live-hit side effects; 16c.1 metadata-only would leak | §3 16c.1 scope clamped to "no prefix cache wired"; §3 16c.2 adds `peek()` primitive; §5 16c.2 `peek` side-effect-free test |
| 4 | Prefix insert lifecycle + block-id source unspecified; RadixPrefixCache uses PagedKVCache ids but 16b/16c use BatchKVCache directly | §3 16c.1 explicitly NOT in scope; §3 16c.2 "Block-id source" deliverable (synthesised monotonic ids, per-row `block_ids` field) |
| 5 | `merge()` on empty `KVCache`s is the wrong primitive for admit; direct `BatchKVCache(left_padding=[...])` matches 16b's pattern | §4 Admit flow rewritten to construct `BatchKVCache` directly, mirroring 16b's `_prepare_cohort` |
| bonus | `RadixPrefixCache.release()` clears live-hit pin, not detached K/V; detached lifetime should track radix node, not release | §3 16c.2 Detached-storage lifecycle bullet; §5 16c.2 release-preserves-detached + eviction-drops-detached tests |

**Net effect**: 16c.1 shrinks from "admit + filter + metadata-only
prefix" to "admit + filter + slot_table correctness, no prefix cache".
This lets 16c.1 stand as a pure state-management milestone and lets
16c.2 land prefix cache in one integrated piece with the block-id
lifecycle finally pinned down.
