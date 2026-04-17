# P-2 Opening Proposal — mini-vLLM core

**Date:** 2026-04-16
**Phase:** P-2 (PLAN.md §7 "Phase 2 — Mini-vLLM Core")
**Status:** proposal — requires user approval before implementation begins.
**Depends on:** P-1 complete (M-2 tag, commit `16e0a6d`).

## Why a phase-opening proposal

PLAN.md §7 calls P-2 "the single most important Phase" because the skeleton
decides stub-replacement paths for every native capability downstream (VQ /
streaming / speculative). The five deliverables are tightly coupled — block
size, page-table shape, request lifecycle, and scheduler policy all cross-cut
— so an upfront design pass avoids rework when the third unit reveals the
first unit's interface is wrong. This doc records the design decisions, flags
open questions, and fixes a unit build order. Implementation begins only
after review.

## Scope statement (hard)

P-2 acceptance validates **batched execution semantics for KV-only models**,
not batched support for all adapter model families. Passing P-2 means Silica
has a working multi-request engine skeleton — admission, decode batching,
prefix bookkeeping, memory budget, preemption — **on pure-KV attention
models only**. Hybrid DeltaNet batching, paged-attention kernel fusion, and
cross-request physical KV sharing are explicitly out of scope for P-2 and
will be re-opened as distinct phase items (see §Risks / opens).

This guardrail exists so that "P-2 passed" cannot be read as "Silica has
universal batched generation". The P-1 Qwen3.5-0.8B single-request path
remains fully functional and untouched in parallel.

## Pre-P-2 probes (cheap closures of opens)

### Q-009 → closed: MLX has no native paged-attention kernel

Inventory of `mlx.core.fast` on mlx 0.31.1:

```text
cuda_kernel, layer_norm, metal_kernel, precompiled_cuda_kernel,
rms_norm, rope, scaled_dot_product_attention
```

`scaled_dot_product_attention` is the only fused attention kernel; it expects
contiguous K / V tensors. No `paged_attention`, no block-mask SDPA, no
gather-based variant. `mlx_lm` has **zero** files referencing "paged".

**Consequence:** **R-7 triggers** — P-2 cannot rely on a paged-attention
kernel for performance. The P-2 ``PagedKVCache`` must therefore use **logical
paging over a physically contiguous (or batched-contiguous) cache**. Page
ids exist as metadata for the prefix cache (COW + LRU + refcount) and for
accounting, but the per-layer storage handed to ``model(tokens, cache=...)``
is mlx-lm's ``BatchKVCache`` — a flat contiguous tensor per layer, sharded
per request along the batch axis. A "real" paged SDPA is a P-5+ item gated
on MLX upstream adding the kernel or Silica writing its own Metal kernel.

**Policy statement (imported from external review):** *Native paged-attention
on MLX is a planned backend replacement path, not a P-2 dependency. P-2
validates engine semantics on dense batched KV; kernel-level paging is
revisited only after functional acceptance and profiling show it is the
dominant bottleneck.* MLX does expose `mlx.core.fast.metal_kernel` as a
public custom-kernel entry point, and there is active community interest
(2025-05 and 2025-12 PagedAttention issues on the MLX repo report
promising Metal benchmarks), so the path exists — the judgment is timing,
not feasibility. Trigger conditions for opening a kernel track are listed
in §Risks / opens.

### New finding — mlx-lm has no batched ArraysCache (DeltaNet)

Inspection of `mlx_lm/models/cache.py`:

- `BatchKVCache` and `BatchRotatingKVCache` exist for full-attention layers.
- `ArraysCache` (DeltaNet / recurrent state holder) has **no batched variant**.
- `_make_cache` in `mlx_lm/generate.py:838-867` only constructs
  ``BatchKVCache`` / ``BatchRotatingKVCache`` — there is no code path that
  batches recurrent caches.

**Consequence:** Qwen3.5 (hybrid DeltaNet + full attention) **cannot** be
driven in a batched forward via mlx-lm's existing machinery. Batched
decoding for DeltaNet models requires a ``BatchRecurrentStateStore`` on the
**adapter side** (D-015 explicitly makes recurrent state adapter-owned), and
that is a P-3 engineering item, not a P-2 core-engine item.

**Proposed P-2 dev-loop model: Qwen3-0.6B** (pure KV attention).

- PLAN R-8 already pre-authorised this fallback: "worst case, P-1 falls
  back to Qwen3-0.6B ... DeltaNet-specific work shifts to P-3."
- P-1 did **not** need to trigger R-8 because Gate B passed for Qwen3.5-0.8B.
  We are triggering an analogous P-2-specific fallback for independent
  reasons: P-2 is about paged KV + continuous batching, which are orthogonal
  to DeltaNet. Using a pure-KV model keeps the two concerns decoupled.
- P-1's Qwen3.5-0.8B path remains unchanged — single-request Engine keeps
  working end-to-end on hybrid models. P-2's batched path will simply not
  support DeltaNet models until P-3's recurrent-state store lands.

## Architecture sketch

```text
┌───────────────────────────────────────────────────────┐
│  Engine.generate_batch([req_1, req_2, ..., req_N])    │  public API
└───────────────────────────────────────────────────────┘
                       │
┌───────────────────────────────────────────────────────┐
│  ContinuousBatcher                                     │
│  step loop:                                            │
│    1. admit WAITING requests (via MemoryBudgeter)      │
│    2. assign prefix-cache hits (via RadixPrefixCache)  │
│    3. run batched prefill on newly admitted            │
│    4. run batched decode_step on DECODE set            │
│    5. update RequestState per request                  │
│    6. preempt lowest-priority if budget violated       │
└───────────────────────────────────────────────────────┘
       │              │               │             │
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────┐
│ Request    │  │ PagedKV    │  │ RadixPrefix│  │ Memory  │
│ State      │  │ Cache (I-2)│  │ Cache      │  │ Budgeter│
│ Machine    │  └────────────┘  └────────────┘  └─────────┘
└────────────┘        │               │             │
                      ▼               │             │
           mlx_lm.models.cache        │             │
           .BatchKVCache  ◄───────────┘             │
           (physical storage;                       │
            logical paging is metadata)             │
                      │                             │
                      ▼                             │
            resident_bytes accounting  ─────────────┘
            (D-012 canonical)
```

Key ownership boundaries:

| Layer | Owns |
| --- | --- |
| ``ContinuousBatcher`` | step loop, admission / preemption timing |
| ``MemoryBudgeter`` | admission policy (yes / no / evict-first) |
| ``RadixPrefixCache`` | radix tree of (token prefix → block ids), refcount |
| ``PagedKVCache`` | block allocator, page table per req_id, LRU eviction |
| ``RequestState`` (per request) | state transitions, generated tokens, SamplingParams |
| ``Engine`` | outer lifecycle, request id assignment, metrics aggregation |

## Component specs

### 1. `silica.core.request.RequestState`

A first-class class (not a loose enum) holding the per-request lifecycle.
**Pure FSM** — the state object only validates transitions and mutates its
own status; it does **not** drive resource side-effects. Releasing KV
blocks on preempt, pinning prefix blocks on re-admit, or zeroing a batch
row on done — all of these are driven by ``ContinuousBatcher`` /
``PagedKVCache`` after observing the new status. This separation pays off
in P-7 (speculative): when tentative state changes must be rolled back,
the FSM does not need a reverse-side-effect path because side effects were
never inside it.

**Fields:** `req_id: str`, `status: RequestStatus`, `prompt_ids: list[int]`,
`generated_ids: list[int]`, `params: SamplingParams`, `metrics`
(per-request timing), `prefix_hit_tokens: int`, `state_delta_snapshot`
(for re-admission from PREEMPTED).

**States and transitions:**

```text
    ┌─ admit (budget ok) ──→ PREFILL ── prefill done ──→ DECODE
    │                          │                           │
  WAITING                      │ preempt                   │ preempt
    ▲                          ↓                           ↓
    │                        PREEMPTED ←──────────────────┘
    │                          │
    └── re-admit (reuses prefix + state_delta snapshot) ───┘

  any state  ── hit stop / max_tokens ──→ DONE
  any state  ── error / unrecoverable OOM / user cancel ──→ ABORTED
```

**Transition method:** `state.transition(to: RequestStatus, *, reason: str)`
validates the transition against a small allow-list and updates `status`.
That is all it does. Returns the previous status (useful for observers) or
raises ``InvalidTransition`` on an illegal move.

### 2. `silica.kvcache.paged.PagedKVCache` (I-2 for P-2)

Implements the I-2 ``KVManager`` Protocol for multi-request KV. Two distinct
bookkeeping layers inside one object — both must be defined before any code
is written:

#### Layer A: physical slot table (the row invariant)

The batched K / V tensor (one ``BatchKVCache`` per layer) has a fixed batch
axis size = `max_concurrent_requests`. Each integer in `0..max_batch-1` is a
**physical slot** (batch row). We maintain:

```python
slot_table:  dict[req_id, int]          # req_id → batch_row
row_state:   list[RowState]             # row index → {FREE, RESERVED, ACTIVE}
```

**Invariant for P-2 (v2.2, updated per Gate-0.5):** The ``slot_table``
mapping is **stable within a contiguous no-filter interval**. Concretely:

- `extend` (admission) preserves existing rows' indices and appends new
  rows at `[B_self, B_self+1, …]` — no reshuffle, no slot rebuild.
- `filter([kept_indices])` (preempt / release) **re-indexes** kept rows
  to `0..len(kept)-1` in the order of the indices list; the scheduler
  rebuilds `slot_table` from that ordering immediately after the call.

The earlier ``row_state`` with RESERVED/ACTIVE/FREE states is retained
for semantic clarity but **does not** imply the physical row persists
when FREE — we compact via `filter` rather than keeping row holes, which
matches mlx-lm's `BatchKVCache` behaviour (there is no in-place "empty
the row" primitive; the only way to reclaim is `filter`). Gate-0.5
confirmed this compaction is cheap (~0.6 ms / op at B=8).

#### Layer B: logical page table (the prefix-cache metadata)

Over the physical rows we lay a logical block map:

```python
page_table:  dict[req_id, list[int]]    # req_id → ordered logical block ids
refcount:    dict[int, int]             # block id → refcount
free_blocks: list[int]                  # logical ids currently unassigned
```

Logical block ids are handles — they name the block for prefix-cache
purposes, not physical memory. A block id's refcount counts **how many
prefix-cache source registrations refer to it** (see RadixPrefixCache for
the exact semantics). A `refcount > 0` means "this block cannot be evicted
from the pool of copy sources"; it does **not** mean "multiple live
requests share this physical page" — that would require option-A aliasing,
which MLX does not yet support.

**Physical-to-logical mapping in P-2:** for the block ids belonging to an
active request, their physical storage IS that request's batch row in the
`BatchKVCache`. Two live requests never point at the same physical row.

**I-2 method implementations:**

- `reserve_for_prefill(req_id, tokens) -> BlockList`: allocate a physical
  slot + `ceil(len(tokens) / block_size)` logical blocks, bind in
  `slot_table` / `page_table`. Returns the logical BlockList; batcher uses
  the slot index internally.
- `append_slot(req_id, n)`: extend the request's logical block list when
  the last block fills during decode. Physical growth is always inside the
  request's fixed row.
- `commit / rollback`: P-7 forward-compat; P-2 delegates to mlx-lm's trim
  on the affected row.
- `free(req_id)`: decrements refcounts on this request's blocks, returns
  its physical row to FREE, drops the `slot_table` / `page_table` entry.
  Any blocks whose refcount drops to 0 become eligible for LRU reclaim.
- `get_computed_blocks(tokens) -> PrefixHit`: delegates to
  ``RadixPrefixCache``.
- `available_blocks()`: `len(free_blocks)` (logical, matches PLAN contract).
- `budget()`: sum of `BatchKVCache.nbytes` across layers + page-table
  overhead.

### 3. `silica.kvcache.prefix.RadixPrefixCache`

Radix tree of `(token prefix) → list[block_id]`, plus an LRU list of
unreferenced blocks. Inspired by mini-sglang's prefix tree; no cross-
reference to runtime mini-sglang code (D-009 — no non-MLX runtime imports).

**Physical semantics — P-2 commits to option B: copy-on-admit.**

A prefix hit means: on admission, Silica copies the hit K / V from the
source blocks into the new request's freshly reserved batch row, skipping
that prefix's forward compute. It does **not** mean two live requests
share physical K / V pages. Concretely:

- `RadixPrefixCache` holds its own refcount on blocks as **copy sources**.
  A block with prefix-cache refcount > 0 cannot be reclaimed to the free
  list — it survives beyond its originating request's lifetime so that
  future admissions can copy from it.
- **Eager-extract requirement (v2.2, per Gate-0.5):** When the owning
  request of a pinned block transitions to DONE / ABORTED / PREEMPTED,
  the scheduler must call `BatchKVCache.extract(row)` to detach that
  request's K/V **before** the `filter` call that drops the row —
  otherwise the K/V is lost with the filtered row. The detached
  per-layer `KVCache` is stored inside `RadixPrefixCache` (keyed by
  block id); on a later prefix hit the batcher copies from this detached
  storage into the new request's row.
- On admit, the batcher calls `lookup(tokens)`; the returned blocks are
  copied into the new request's row (cheap — a slice assignment per
  layer), and the source blocks' refcount stays pinned.
- Memory scaling: **option B saves TTFT, not memory**. 8 concurrent
  requests sharing a 2k-token system prompt still hold 8 copies of that
  2k KV, one per row. True physical aliasing (option A) needs either a
  gather-based attention kernel or scattered-read SDPA — neither exists
  in MLX today (see Q-009 closure + §Risks). This distinction is
  load-bearing for PLAN.md §2's 48 GB budget assumptions and should be
  re-evaluated at P-5.

Options A and C (alternatives we explicitly did not choose):

- **Option A — physical aliasing.** Two requests literally point at the
  same physical K / V slots; decode reads gather from shared pages.
  Requires a paged-attention kernel; R-7 triggered → not in P-2.
- **Option C — metadata-only hit.** Hit counter goes up but engine still
  runs full prefill. Misleading half-feature; rejected because a "hit"
  that changes nothing observable is worse than no cache at all.

**Core operations:**

- `insert(tokens, block_ids)`: add the request's blocks to the tree at
  block-aligned prefix points. Only whole blocks are insertable — partial
  blocks are not retained. This keeps the copy-source invariant simple:
  a hit always returns a whole number of copy-complete blocks.
- `lookup(tokens) -> PrefixHit`: walk the tree, return the longest block-
  aligned prefix match as `(source_block_ids, num_hit_tokens)`.
  Increments the prefix-cache refcount on each source block.
- `release(block_ids)`: decrement refcounts; blocks whose refcount drops
  to zero enter the LRU eviction list.
- `evict_until(n_blocks)`: free LRU unreferenced blocks back to
  `PagedKVCache.free_blocks` when the budgeter asks for room.

**Hit counter:** `self.hits: int` — incremented on any `num_hit_tokens > 0`
lookup. P-2 acceptance reads this, but also asserts the stronger behavioural
property (see §Acceptance) that hit prefixes were not re-prefilled.

### 4. `silica.scheduler.budget.MemoryBudgeter`

Admission + preemption policy. Requires two distinct byte accountings —
conflating them is the classic budgeter bug.

**Two accountings:**

- **resident_bytes** — actual bytes currently held by cache entries, i.e.
  `PagedKVCache.budget().resident_bytes` (D-012 canonical). Grows as
  decode progresses.
- **reserved_bytes** — admission-time *upper-bound* reservation.
  `sum_over_admitted_requests(worst_case_kv)` where a request's worst case
  is `(n_prompt + max_tokens) * bytes_per_token`. A request's reservation
  is tallied at admission and released at free / abort. Always
  `resident_bytes <= reserved_bytes` for admitted, non-finished requests.

Admission decisions compare **reserved_bytes + new_request_worst_case**
against the cap, not `resident`. Without this split, a budgeter reasoning
on live `resident_bytes` is systematically over-optimistic and will admit
requests that later OOM mid-decode. Keeping both concepts explicit from
the start is cheap and prevents a whole class of "it worked until real
traffic" bugs.

**Inputs per decision:**

- `cap` — target resident budget. Default = 80% of unified memory (M5 Pro
  48 GB → 38.4 GB cap), knob `SILICA_RESIDENT_CAP_MB` overrides.
- current `reserved_bytes`
- current `resident_bytes` (used for observability + soft-preemption trigger)
- weights footprint (static, read from adapter.config + quant profile)
- incoming request's worst-case KV.

**Policy (P-2 greedy):**

1. Accept if `weights + reserved_bytes + new_worst_case <= cap`.
2. Otherwise try LRU eviction of unreferenced prefix-cache source blocks
   (``RadixPrefixCache.evict_until``). Re-check.
3. If still over cap, preempt the lowest-priority (FIFO-newest) DECODE
   request: status → PREEMPTED, its row released, reserved_bytes
   decremented; preempted request's `state_delta_snapshot` retained for
   re-admission. Re-check.
4. If nothing can be freed (e.g. single in-flight request already over
   cap), the new request transitions to ABORTED with
   `finish_reason="budget-exhausted"` — satisfies PLAN's "clean abort"
   acceptance line without crashing the engine.

### 5. `silica.scheduler.batcher.ContinuousBatcher`

The step loop. Holds references to all of the above plus the adapter.

**One step** (called by ``Engine.generate_batch``):

1. Admission phase: ask ``MemoryBudgeter`` about each WAITING request in
   FIFO order; admitted ones move to PREFILL.
2. Prefix-cache phase: for each newly admitted, call
   ``RadixPrefixCache.lookup`` and pin hit blocks; subtract hit length from
   the request's prefill work.
3. Prefill phase: batched forward over all newly admitted requests. This is
   the most delicate code — mlx-lm's `BatchKVCache` expects `left_padding`
   concatenation, so the batcher has to produce a left-padded token tensor.
4. Decode phase: batched forward over all DECODE requests (one-token each,
   concatenated to `(B, 1)`).
5. Sample + yield: Silica's ``Sampler`` applied per-row on the resulting
   `(B, V)` logits; tokens streamed out to the per-request consumer queues
   (a `dict[req_id, Queue]` that the caller reads).
6. State updates: tokens appended to each ``RequestState.generated_ids``;
   completions move to DONE; stop hits moved to DONE; evictions to
   PREEMPTED.

### 6. `silica.engine.Engine.generate_batch`

New top-level API that coexists with `generate` (single-request).

**Signature designed for forward compatibility:**

```python
def generate_batch(
    prompts: list[str],
    params: SamplingParams | list[SamplingParams],
) -> Iterator[BatchEvent]: ...
```

P-2 implements the homogeneous case (single ``SamplingParams`` applied to
all rows; a list of identical params is accepted and validated). A list
with per-row differing params raises ``NotImplementedError`` in P-2 and
becomes the entry point for per-row logit-processor stacks in P-3+. The
union type in the signature is accepted now so this promotion does not
churn the public API.

**BatchEvent** — event stream, not tuple stream, so DONE / ABORTED /
PREEMPTED are first-class rather than magic values:

```python
@dataclass(frozen=True)
class BatchEvent:
    req_index: int
    kind: Literal["token", "done", "aborted"]
    token_id: int | None = None          # set when kind == "token"
    finish_reason: str | None = None     # set when kind in {"done", "aborted"}
```

Callers bucket by `req_index`; terminal events (`done` / `aborted`) signal
that no more events for that index will arrive. The event shape is
extensible — `kind="preempted"` or `kind="progress"` can be added later
without breaking the signature. (P-2 does not surface ``preempted`` to
callers — preemption is internal, the request returns to WAITING and
eventually yields more ``token`` events on re-admission.)

## Unit order & dependencies

```text
1. RequestState          (no deps, ~80 LOC + 15 tests)
2. PagedKVCache          (deps: RequestState types, ~220 LOC + 20 tests)
3. RadixPrefixCache      (deps: PagedKVCache.BlockList, ~270 LOC + 18 tests)
4. MemoryBudgeter        (deps: above 3, ~140 LOC + 12 tests)
5. ContinuousBatcher     (deps: above 4, ~320 LOC + 20 tests)
6. Engine.generate_batch (glue, ~110 LOC + 10 tests)
7. Acceptance probe      (scripts/acceptance_p2_*.py + doc)
```

**Total estimate:** ~1,140 LOC production + ~400 LOC tests + probe + doc.
Comparable to P-1's ~1,200 LOC, but with more interlocking state.

**Gate-0 (blocking pre-P-2).** A ~30-line smoke test that constructs an
`mlx_lm.models.cache.BatchKVCache`, drives `update_and_fetch` on batched
`(B=4, T=3)` inputs against a tiny fake model, and confirms batched
forward + row-scoped `update_and_fetch` behave as expected. Same role as
Gate A/B at P-1 entry: "batched cache injection works" is a load-bearing
assumption under everything else; verify it before writing the page
table. Failure here would force either a custom batched-cache shim or
falling back to per-request serialisation (huge scope change). P-2 unit
work **does not start** until Gate-0 is green and documented.

**Early type extraction.** Before Unit 1 touches mlx-lm, promote the small
shared vocabulary (``BlockId``, ``BlockSpan``, ``PrefixHit``, ``RowIndex``)
into `silica.kvcache.manager` or a new `silica.kvcache.types` module.
RadixPrefixCache unit tests can then operate on fake `BlockId`s without
dragging in MLX or BatchKVCache — test-only decoupling that keeps the
radix-tree tests fast and deterministic.

## Acceptance test shape

From PLAN.md §7 P-2, sharpened to three hard behavioural assertions each.

**1. 8 concurrent requests run stably** — `scripts/acceptance_p2_concurrent.py`

- 8 distinct prompts, `max_tokens=64`, **greedy (`temperature=0.0`)**.
  Greedy is required because only deterministic decode lets us compare
  batched output against a single-request oracle row-for-row; top-p /
  temperature variance would mask real divergence.
- For each prompt `i`, record the single-request tokens via P-1's
  `Engine.generate` as oracle, then assert the batched run's tokens for
  `req_index == i` are identical.
- After all 8 complete: `PagedKVCache.budget().resident_bytes == 0`,
  `free_blocks` is fully populated, `slot_table` is empty, every `row_state`
  slot is FREE. Catches silent KV / row leaks that functional tests miss.

**2. Shared-prefix hit is observable at both metadata and behaviour
levels** — `scripts/acceptance_p2_prefix_hit.py`

- 4 prompts sharing a 100-token prefix + 4 unique continuations,
  submitted in sequence so request 1 populates the cache and 2–4 can
  hit.
- **Metadata assertion:** `RadixPrefixCache.hits >= 3`.
- **Behavioural assertion:** requests 2–4 must skip prefill forward on the
  shared prefix tokens. Implementation: instrument the adapter (or use an
  injectable counter) to count the actual prefill token count per
  request, assert `counted_prefill_tokens[i] == len(prompt_i) -
  hit_tokens_for_i` for `i in [2, 3, 4]`. Without this the hit counter
  is just accounting theater.

**3. Budget overflow produces clean queue or abort without engine-level
collateral** — `scripts/acceptance_p2_budget.py`

- Construct a scenario where the 8th admission would overflow the cap;
  assert one of two clean outcomes: (a) it waits in WAITING until an
  earlier request finishes and releases reserved bytes, then admits and
  completes normally; (b) it transitions to ABORTED with
  `finish_reason="budget-exhausted"`.
- **Structural assertion (most important):** after the scenario runs to
  completion, `PagedKVCache.budget().resident_bytes == 0`,
  `PagedKVCache.reserved_bytes == 0`, `free_blocks` is fully populated,
  `row_state` is all FREE, `RadixPrefixCache` refcount dict contains
  only prefix-source entries (no stale live-request references). This
  catches the leak-on-abort path that is the usual scheduler bug.

## Risks / opens

| ID | Description | Mitigation |
| --- | --- | --- |
| R-7 (triggered) | No native MLX paged-attention kernel | Logical paging over BatchKVCache; kernel track is trigger-gated (see below) |
| R-NEW (DeltaNet batching) | mlx-lm has no BatchArraysCache; Qwen3.5 hybrid cannot batch | P-2 uses Qwen3-0.6B; P-3 introduces BatchRecurrentStateStore |
| R-NEW (padded vs ragged batch) | `BatchKVCache` uses left-padding; wasted compute when request lengths vary | Accept for P-2; ragged-batch is a P-4 optimisation |
| R-NEW (prefix cache + DeltaNet) | Radix cache only works for KV-attention layers; recurrent state is per-request | Document explicitly; prefix hits for DeltaNet models are P-3 |
| R-NEW (upstream cache churn) | mlx-lm has in-flight fixes touching BatchKVCache offset behaviour (PR circa 2026-04) and a 2026-03 custom-kernel use-after-free under lazy graph composition | Gate-0 verifies the current mlx-lm 0.31.2 behaviour empirically; if the offset bug is still live, pin to a post-fix version or cherry-pick |
| PLAN impact | R-NEW rows above fold into PLAN.md as R-9 / R-10 / R-11 / R-12 and new Q entries | Separate PLAN.md update PR after this proposal is approved |

### Paged-attention kernel — trigger-gated future track

Instead of a parallel spike during P-2, a native paged-attention backend
(either an MLX upstream landing or a Silica-owned `mlx.core.fast.metal_kernel`
implementation) is opened **only** when all of the following are true:

1. P-2 dense-batch path is functionally correct (acceptance passed).
2. Profiling on a representative workload shows attention IO / KV copy
   dominates end-to-end latency — not scheduler overhead, not tokenizer
   overhead, not sampling.
3. Option-B prefix copying is demonstrably causing memory-bound rejection
   of admissible workloads (e.g. 8 agentic requests on a 2k shared prompt
   overflow the cap purely because of prefix duplication).
4. Either MLX upstream ships a paged SDPA that we can retrofit behind the
   existing Protocol, or the team is willing to own a Metal kernel
   including its stability envelope (the 2026-03 custom-kernel bug report
   suggests this envelope is still moving).

**Not starting it in P-2 is a focus decision, not a technical one.** The
three largest unknowns in P-2 are scheduler state transitions, prefix-
refcount correctness, and preemption cleanup — all *orthogonal* to the
attention kernel. Adding a Metal-kernel spike in parallel couples those
unknowns to kernel correctness and MLX lazy-execution semantics, making
any failure harder to attribute. P-4 bench exit (or P-5 entry) is the
natural point to re-evaluate against real numbers.

## What stays untouched

- `silica.core.*` (Sampler, SamplingParams, logger, profiler) — no changes
  in P-2; they are model-agnostic and batch-agnostic.
- `silica.kvcache.simple.SimpleKVCache` — stays as the single-request path.
  P-1's Engine.generate remains fully functional on Qwen3.5.
- `silica.mlx.runner` — the single-request `forward((T,), cache_list)
  -> (V,)` stays; **v2.2** adds a sibling `forward_batched((B, T),
  cache_list) -> (B, V)` returning per-row last-position logits.
  ``forward`` becomes an adapter that wraps ``forward_batched`` with
  ``tokens[None]`` / ``logits[0]`` so P-1 callers stay unchanged.
- `silica.models.qwen3.Qwen3Adapter` — gains a batched prefill / decode_step
  path, but the single-request methods stay identical.
- I-1..I-5 Protocol signatures — **frozen per P-0 exit**. No change.

## Amendment ledger

- **v2.2 (2026-04-17, after Gate-0.5)**: Layer-A row invariant changed
  from "row fixed for lifetime" to "slot_table stable within no-filter
  interval"; option-B eager-extract requirement spelled out;
  `runner.forward_batched` signature added. See `docs/P2_GATE_0_5.md`
  for the primitives and probe results that drove these changes.
- **v2.1 (2026-04-16)**: Hard scope statement; option-B prefix semantics;
  slot_table + row_state invariant (original form); pure-FSM
  RequestState; BatchEvent / union-params signature; reserved vs resident
  split; Gate-0 upgraded to blocking; acceptance sharpened.
- **v2 (2026-04-16)**: First full proposal after review of v1.

## Proposed decision-log delta (for subsequent PLAN.md update)

- **Close Q-009** with "no native MLX paged-attention kernel; logical paging
  over BatchKVCache is the P-2 strategy."
- **New D-016**: P-2 dev-loop model is Qwen3-0.6B; DeltaNet batching via
  adapter-owned ``BatchRecurrentStateStore`` shifts to P-3.
- **New Q-011**: contiguous-per-request vs physically-paged storage — when
  (and whether) MLX grows a paged-attention kernel that lets us retrofit.
  Revisit at **P-4 bench exit or P-5 entry**, whichever comes first. Opening
  the kernel track requires the four trigger conditions in §Risks / opens
  to be simultaneously true.
- **Update R-7, R-8** with current status.
- **Append R-9 / R-10 / R-11** per the table above.

---

**Ready for review.** Once approved, I will open P-2 with unit #1
(``RequestState``) — smallest, leaf dependency, no mlx-lm coupling, quick
sanity check that the state-machine shape holds up before committing to
paging.
