# P-3-C5.3 — Prefix-cache cooperation: design note

| Field       | Value |
| ----------- | ----- |
| Sub-unit    | P-3-C5.3 (RadixPrefixCache cooperation) |
| Status      | design — recon complete, implementation pending user review |
| Parent docs | `docs/P3_C5_OPENING.md` §6.4; `docs/P3_C5_DRIFT_EXPERIMENT/README.md` |
| Predecessor | C5.2 captures + stashes recurrent snapshots on preempt; C5.3 wires the capture path into in-flight prefill at block boundaries and the restore path into prefix-hit admission. |

## 1. Recon summary

### 1.1 `silica/kvcache/prefix.py` — radix tree layout

- `_Node` has `__slots__ = ("parent", "tokens", "block_id", "children", "access_tick")`. Each non-root node represents exactly `block_size` tokens and one `block_id`. Adding a per-node optional `recurrent_snapshot` field is one slot extension.
- Node lifecycle: `insert_detached` → new node + `store.allocate_id` + `store.retain_source` + `store.register_detached(id, blocks)`; `lookup` retains hits via `store.retain_hit`; `release` decrements via `store.release_hit`; `_evict_node` calls `store.release_detached` + `store.release_source` and drops the node from `parent.children`. **No external code holds a reference to `_Node`**, so dropping a node automatically releases its attached recurrent snapshot — no separate hook needed.
- `PrefixBlockStore` Protocol owns block-id-keyed K/V storage. It does NOT need to know about recurrent snapshots: snapshots are per-tree-node logical metadata (one per radix path), not per-block-id physical storage.

### 1.2 `silica/scheduler/batcher.py` — admission paths

- `_admit_single_hit_row` (line 1156): builds the seeded `row_cache` at line 1216 (`row_cache = build_seeded_batch_kv(detached, num_layers=num_layers)`), runs the suffix prefill at line 1222 (`forward_batched(model, suffix_arr, list(row_cache))`), then extends into `self._batch_cache` at line 1232 / 1235. **Restore site for C5.3 is between line 1216 and line 1222** — after `row_cache` is materialised with the hit prefix's full-attention K/V, before the suffix prefill advances the cache.
- `_extract_and_insert_prefix` (line 533): at preempt / reclaim time, builds `computed_ids = prompt + generated[:-1]` (line 565-570), aligns to `block_size`, and slices per-block K/V via axis-2 ranges. **Crucially, this function only sees the row's FINAL cache state** (`T + len(generated) - 1` consumed) — it has no record of intermediate per-block-boundary recurrent states. Capturing snapshots here is therefore not viable for C5.3's restore semantics.

## 2. Boundary alignment — token boundaries vs trajectory regimes

### 2.1 Token-boundary alignment (the easy half)

C5.2 hit a fundamental boundary mismatch on the full-replay path:

```text
preempt-time live cache:                T + len(generated) - 1 consumed
full-replay prefill of composite_prompt: T + len(generated) consumed
```

The mismatch is one token; restore at the misaligned site rewinds the cache and erases the last generated token's consumption. C5.2 therefore stashes the snapshot but does NOT call restore on the full-replay path.

**C5.3's prefix-hit admission path has aligned token boundaries by construction.**

- The radix tree stores prefixes at `block_size`-aligned token boundaries: every `_Node` represents exactly `block_size` tokens.
- A prefix hit of length `K` blocks corresponds to `K * block_size` consumed tokens at the snapshot time.
- After `build_seeded_batch_kv(detached, num_layers)`, the seeded `row_cache`'s full-attention K/V slots already represent `K * block_size` consumed tokens (the hit prefix), but the recurrent slots are `None` (lazy — DeltaNet `ArraysCache` only allocates on first forward).
- Restoring the snapshot at THIS point sets the recurrent state to the same "K * block_size consumed" boundary the K/V is already at.
- The subsequent `forward_batched(suffix_arr, row_cache)` then advances the cache from `K * block_size` to `K * block_size + len(suffix)` consumed.

```text
restore site (after build_seeded_batch_kv, before suffix prefill):
    row_cache state = K * block_size consumed (full-attn slots seeded;
                                                recurrent slots None)
snapshot must describe:
    K * block_size consumed (recurrent state at the END of the hit prefix)
post-suffix-prefill state:
    K * block_size + len(suffix) consumed
```

**Token boundaries align — but that alone is not enough for byte-exact-vs-oracle.** §2.2 below addresses why.

### 2.2 Trajectory regime — why slice-prefill is a regime choice, not a snapshot site

The C5.2 drift experiment proved that under bf16, two different segmentations of the same logical token sequence produce recurrent states that differ at fp32 precision (~1-2% rel-Frob), even though both eventually consume the same tokens. The drift sits below greedy-argmax under bf16, but it IS observable byte-by-byte.

Three relevant **trajectory regimes** through the same token sequence:

- **Contiguous prefill** (today's `_prefill_phase` and `_admit_miss_cohort`): one `forward_batched` call consumes the whole prompt.
- **Single-step decode** (today's `_decode_phase`): one token per `forward_batched` call, post-prefill.
- **Slice prefill** (the regime C5.3 needs to introduce): `ceil(T / block_size)` calls of `block_size` each.

To capture a snapshot at boundary `K * block_size` consumed, the row's forward must HAVE run a discrete operation that stops at that boundary. mlx-lm's `gated_delta_kernel` advances the recurrent state through all S timesteps inside a single layer call and writes `cache[1]` once at the end (per `docs/P3_C5_STATE_INVENTORY/README.md` §5). There is no non-intrusive way to observe the kernel's intermediate state between timesteps without either modifying the kernel itself or splitting the input so the kernel runs over `block_size` steps at a time. **Both options are equivalent to slice-prefill at some level of the stack** — there is no "pure observation" that leaves the segmentation untouched.

**Consequence for C5.3's acceptance gate.** If the producer runs **contiguous prefill** but the consumer (prefix-hit admission) restores from snapshots that were captured via slice-prefill, the consumer's recurrent state lives in a **different trajectory regime** than the producer's. The K/V seeded into `row_cache` from the radix tree was captured during the producer's contiguous-prefill regime; the recurrent snapshot to be restored was captured during a slice-prefill regime. Even at the same token boundary, the two regimes' recurrent states differ at fp32. There is no byte-exact-vs-no-prefix-cache-contiguous oracle achievable.

**The honest fix is to make slice-prefill the canonical regime for hybrid adapters with `prefix_cache=True`** (§3.2 below). Both the producer (whose snapshots get stored) and the consumer (whose restored state must match those snapshots) operate in the same regime. The oracle for the byte-exact-vs-oracle gate becomes "same hybrid adapter, slice-prefill regime, no preempt or hit" — not "no-prefix-cache contiguous regime".

This is a deliberate scope correction. C5.3 does NOT promise "prefix_cache=True hybrid adapters produce byte-identical tokens to prefix_cache=None hybrid adapters"; the drift experiment shows that under bf16 greedy these still match in practice, but it's no longer a stated gate. C5.3 promises "within the slice-prefill regime, with-hit and without-hit produce byte-identical recurrent state at the post-restore + post-suffix-prefill boundary".

## 3. Locked design decisions

### 3.1 Snapshot storage — attach to `_Node`, not `PrefixBlockStore`

Add `recurrent_snapshot: RecurrentSnapshot | None = None` as a fifth `__slots__` entry on `_Node`. Rationale:

- Snapshots are **per-tree-path logical metadata**, not per-block-id physical storage. Two radix paths sharing a `block_id` (via the duplicate-prefix branch in `insert_detached` — line 252-255 of `prefix.py`) would each want their own snapshot at their own boundary, but in practice the duplicate path's tree node is reused and the snapshot is identical (same prefix, same tokens, same recurrent state). One snapshot per tree node is the right granularity.
- Eviction is automatic: `_evict_node` removes the node from `parent.children`, and Python GC reclaims the snapshot. No extra `release_recurrent_snapshot` hook needed on the store.
- Avoids extending the `PrefixBlockStore` Protocol surface (which would require touching `PagedPrefixBlockStore` + `SyntheticPrefixBlockStore` + the Protocol definition).

### 3.2 Capture site — slice-prefill regime, conditional on `RecurrentStateAdapter` + `prefix_cache != None`

`_extract_and_insert_prefix` only sees the row's final cache state, not per-block-boundary intermediate states. Capturing snapshots there cannot yield boundary-aligned snapshots (they would be off by `consumed_tokens % block_size`, which is 0..block_size-1).

mlx-lm's `gated_delta_kernel` does not expose intermediate-step state to a non-intrusive callback (§2.2). The only way to capture state at boundary `K * block_size` is to **issue a discrete forward that stops there** — i.e., slice-prefill. There is no "pure observation that leaves segmentation untouched" alternative; that option was considered and ruled out in §2.2.

**Decision: slice-prefill is a regime choice**, not a single capture site.

When the running adapter implements `RecurrentStateAdapter` AND `self._prefix_cache is not None`, the batcher routes every prefill (initial cohort, mid-run admission, suffix prefill on prefix-hit admission) through a slice-prefill helper that issues `ceil(T / block_size)` `forward_batched` calls of `block_size` tokens each (final chunk may be shorter). Snapshots are captured between calls.

**Adapters / configs that stay on contiguous prefill (no regime change):**

- Non-recurrent adapters (Qwen3, Gemma4): no recurrent state to snapshot, no need to slice. Today's contiguous prefill is preserved bit-for-bit. The `RecurrentStateAdapter` `isinstance` check gates the regime branch.
- Recurrent adapters with `prefix_cache=None`: the producer-side trajectory stays contiguous because there is no consumer side that would consume snapshots. Avoids paying slice overhead when there's no benefit. (The C5.4 guard removal preserves this behaviour: `prefix_cache=None` hybrid runs unchanged in production.)

**Adapters / configs that go to slice-prefill regime:**

- Recurrent adapters with `prefix_cache != None` (the C5.3 target). Producer-side prefills slice; consumer-side `_admit_single_hit_row`'s suffix prefill also slices; both are in the same regime.

**Production gate predicate** (used by all production code paths, including any future P-7 / batched-MoE consumer):

```python
def _slice_prefill_regime_active(self) -> bool:
    if not isinstance(self._adapter, RecurrentStateAdapter):
        return False
    if self._prefix_cache is not None:
        return True
    # Test-only escape: see _force_recurrent_slice_prefill_for_c5_3_oracle
    # in §3.6. Production paths never set this flag.
    return self._force_recurrent_slice_prefill_for_c5_3_oracle
```

The second clause is a test-only opt-in; in production the predicate is exactly `RecurrentStateAdapter and prefix_cache is not None`.

**Acceptance oracle redefinition (per §2.2).** Tests assert byte-equality **within the slice-prefill regime**. Both subject and oracle use hybrid adapters and both run slice-prefill, so the trajectory regime is held constant; only the prefix-cache cooperation differs.

- **Subject**: `prefix_cache=RadixPrefixCache(...)` + slice-regime active by the production predicate (clause `prefix_cache is not None`). Sees the hit, restores the snapshot.
- **Oracle**: `prefix_cache=None` + slice-regime forced via the test-only flag `_force_recurrent_slice_prefill_for_c5_3_oracle=True` (per §3.6). Production never trips this flag. The oracle never sees a hit because there is no prefix cache; it slice-prefills the full prompt from scratch. This isolates the cooperation effect (snapshot-restore vs full re-prefill) from the regime effect (slice vs contiguous).

Subject's post-restore state byte-equals oracle's slice-trajectory state at every DeltaNet layer. The contiguous-prefill regime's tokens are NOT a C5.3 oracle; they remain a sub-rounding-drift sanity check (drift experiment showed greedy bf16 tokens match across regimes, so token-stream parity remains testable even though it is not the gate).

**Storage of in-flight snapshots before tree-attachment** — the snapshots are captured during forward but only attached to tree nodes when `_extract_and_insert_prefix` runs (preempt / reclaim). Side-buffer them on the `_BatchRow` itself, **keyed by absolute block index** (see §3.2.1 below for why list-position is insufficient). When `_extract_and_insert_prefix` walks `aligned_tokens // block_size` blocks for tree insertion, it reads the matching snapshot by absolute block index from this side-buffer and passes it to `insert_detached(..., recurrent_snapshots=...)`.

#### 3.2.1 Absolute block indexing in the side-buffer

A row's `recurrent_snapshots_per_block` field cannot be a plain `list` indexed by list-position. Reason:

- A miss-from-zero row (admitted via `_admit_miss_cohort`) starts the side-buffer empty. Its slice-prefill captures snapshots for blocks 0, 1, 2, ..., appending in order. Absolute block index == list position. Fine.
- A prefix-hit-admitted row (admitted via `_admit_single_hit_row`) starts AT absolute block index `K = len(hit.block_ids)`. Its suffix prefill captures snapshots for absolute blocks `K`, `K+1`, .... If we plain-append, list[0] is the absolute block-K snapshot; `_extract_and_insert_prefix` later iterating `range(aligned_blocks)` sees list[0] when iterating absolute index 0, but the radix tree node at absolute index 0 was already attached in a previous insertion — this row never captured a fresh snapshot for block 0, and treating list[0] as if it were the block-0 snapshot is silently wrong.

**Resolution: use a `dict[int, RecurrentSnapshot]` keyed by absolute block index.** Hit admission seeds the dict with the `K` ancestor nodes' snapshots (copied from the radix tree's hit chain), so the row's view of "what snapshots cover what absolute blocks" is complete. Suffix prefill adds `K`, `K+1`, ... to the dict at their absolute indices. `_extract_and_insert_prefix` reads `recurrent_snapshots_per_block.get(absolute_block_index)` for each block it inserts, gracefully passing `None` for blocks that have no snapshot in this row's view (which means: tree node already exists from another path; `insert_detached`'s duplicate-prefix branch will leave it alone).

The decode-step counter that triggers per-block-boundary capture must initialise to `(absolute_consumed_tokens) % block_size` rather than 0. For miss-from-zero rows that's 0 trivially; for hit-admitted rows it's `(K * block_size) % block_size == 0` (because hit boundary is block-aligned), so the counter is also 0 there. The wrinkle only appears if we ever admit a row with non-block-aligned consumed tokens — this is not a current code path, but the counter spec is written for absolute consumption to avoid future drift.

### 3.3 Restore site — between `build_seeded_batch_kv` and suffix prefill

In `_admit_single_hit_row`, between line 1216 and line 1222:

```python
row_cache = build_seeded_batch_kv(detached, num_layers=num_layers)

# C5.3 restore. Precondition (enforced at Phase B split per §3.5):
# this code path only runs when isinstance(adapter, RecurrentStateAdapter)
# AND the deepest hit node has recurrent_snapshot is not None. The
# Phase-B classifier routes hit pendings without a deepest-node
# snapshot to the miss cohort BEFORE retain — so we never get here
# without a snapshot to consume. The defensive assert below is a
# tripwire for a Phase-B classifier bug, not a fallback.
if isinstance(self._adapter, RecurrentStateAdapter):
    assert hit.block_ids, "hit path entered with empty block_ids"
    snapshot = deepest_node.recurrent_snapshot
    assert snapshot is not None, (
        "hit path entered with snapshotless deepest node — Phase-B "
        "classifier should have routed to miss"
    )
    self._adapter.restore_recurrent_state(
        row_cache, row_idx=0, snapshot=snapshot
    )

suffix_arr = mx.array([suffix_tokens], dtype=mx.int32)
logits = slice_prefill_or_contiguous(
    self._model, suffix_arr, list(row_cache),
    slice_block_size=self._maybe_slice_block_size(),
)
```

Only the **deepest usable** hit node's snapshot is consumed — i.e., the node at depth `usable_hit_tokens / block_size`, where `usable_hit_tokens` is the clamped value passed in by the Phase-B classifier (line 1117). Intermediate hit nodes' snapshots are not needed at restore time; they only matter for tree maintenance and future re-extraction. The "raw deepest" node (which may be deeper than `usable_hit_tokens` when the prompt is fully cached) is NOT the restore source.

`prefix_cache.find_node(block_id)` does not exist today. **Lean: extend lookup** to a sibling method `lookup_with_node(tokens) -> tuple[PrefixHit, _Node | None]` that returns the deepest matched node alongside the existing hit info; replaces the `lookup(pending.prompt_ids[:usable_hit_tokens])` call at line 1197 with `lookup_with_node(pending.prompt_ids[:usable_hit_tokens])`. The truncation by `usable_hit_tokens` is what guarantees the returned node is the deepest USABLE node, matching the Phase-B classifier's snapshot-presence check.

### 3.4 Eviction — automatic via `_Node` GC

When `_evict_node` removes a node from `parent.children`, Python GC drops the `_Node` instance and its `recurrent_snapshot` attribute. No special handling. The snapshot's `mx.array` tensors are released in the same wave as any other GC-eligible mlx array.

### 3.5 Fallback when deepest USABLE node has no snapshot — Phase-B classifier routes to miss

A hit chain may reach a deepest-usable node (the node at depth `usable_hit_tokens / block_size`, see §3.3) whose `recurrent_snapshot is None` for two reasons:

- The node was inserted before C5.3 capture wiring was active (legacy / transitional).
- The original `_extract_and_insert_prefix` insertion lacked a snapshot for that block (e.g., the row that produced this prefix was admitted via a hit path that didn't propagate snapshots far enough — a real correctness path of its own; see §3.5.1).

**Decision: route the pending to the miss cohort at Phase-B classification, BEFORE any `lookup`/`retain_hit` call.**

The Phase-B split currently lives at `silica/scheduler/batcher.py:1099-1121`:

```python
# After C5.3 (sketch):
if self._prefix_cache is None:
    miss_rows = list(accepted)
else:
    block_size = self._prefix_cache.block_size
    needs_snapshot = isinstance(self._adapter, RecurrentStateAdapter)
    for pending in accepted:
        raw = self._prefix_cache.peek(pending.prompt_ids)
        # Existing usable-aligned-token bookkeeping unchanged: clamp
        # raw.num_hit_tokens by max_aligned to reserve at least one
        # suffix token. See batcher.py:1109-1117 for the exact form.
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
        # C5.3: hybrid adapters require a recurrent snapshot at the
        # deepest USABLE node — i.e., the node `_admit_single_hit_row`
        # will actually restore from. The raw deepest node may be
        # deeper than `usable` (e.g., block_size=4 with a fully cached
        # 8-token prompt: raw.num_hit_tokens=8 but max_aligned=4, so
        # admission restores from the 4-token node, not the 8-token
        # leaf). Re-walking with the truncated prompt selects the
        # node at the actual restore depth.
        if needs_snapshot:
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
```

`peek_with_node` is a new sibling of the existing `peek` (side-effect-free walk). Returns the deepest matched node alongside a `PrefixHit`. The Phase-B classifier first uses `peek` to compute `usable` (raw `num_hit_tokens` clamped by `max_aligned`), then re-walks via `peek_with_node` on `prompt_ids[:usable]` so the node returned is the one whose snapshot the atomic-phase restore will actually consume. Two walks per hybrid hit pending — radix walk is O(prompt_len/block_size) and the overhead is negligible.

The atomic-phase counterpart in `_admit_single_hit_row` uses `lookup_with_node(pending.prompt_ids[:usable_hit_tokens])` — matching the existing `lookup(pending.prompt_ids[:usable_hit_tokens])` call at line 1197 — to materialise the same deepest-usable node under retain semantics.

This avoids the half-built-state risk: `retain_hit`, `fetch_detached_blocks`, `build_seeded_batch_kv`, and the suffix prefill all stay together as one atomic phase. The fallback decision is made before the atomic phase begins, not midway.

`prefix_hits` counter increments are unchanged — only rows that actually go through `_admit_single_hit_row` increment, so a route-to-miss does not pollute the counter.

`forward_prompt_tokens` accounting also unchanged: routes-to-miss go through `_admit_miss_cohort` and pay full prefill on the contiguous regime (or slice regime per §3.2). The miss-cohort's snapshot capture proceeds normally; the row's future `_extract_and_insert_prefix` would then carry valid snapshots, gradually backfilling the radix tree's snapshotless legacy nodes (see §3.5.1).

#### 3.5.1 Backfill via duplicate-insert path

`prefix.py`'s `insert_detached` already handles the "node already exists" case at line 252-255 by reusing the existing node and skipping K/V registration. C5.3.0 extends this branch to **backfill the existing node's `recurrent_snapshot` if and only if it's currently `None` and the caller provides a fresh non-None snapshot**:

```python
# Inside insert_detached, the duplicate-prefix branch:
existing = node.children.get(chunk)
if existing is not None:
    # C5.3 backfill: if the existing node lacks a snapshot but the
    # caller has one, attach it. This lets a row that misses on a
    # legacy snapshotless prefix and re-prefills produce snapshots
    # that the next admission can hit. When both existing and
    # caller-provided snapshots are non-None, keep existing and let
    # the new one GC. (RecurrentSnapshot today carries only `entries`
    # and `nbytes` — no boundary metadata, regime marker, or
    # consumed-token count — so a "matching" debug assert is not
    # mechanically implementable from the planned value object;
    # adding such metadata is out of scope for C5.3 and would be
    # revisited if a future sub-unit needs it.)
    new_snap = (
        recurrent_snapshots[i] if recurrent_snapshots is not None else None
    )
    if new_snap is not None and existing.recurrent_snapshot is None:
        existing.recurrent_snapshot = new_snap
    # else: keep the existing snapshot; new_snap GC-eligible after caller
    self._touch(existing)
    node = existing
    continue
```

**Self-healing:** legacy snapshotless nodes get filled in as soon as a row that re-prefills the same prefix is extracted. After enough backfill, route-to-miss rate drops and prefix-hit rate climbs back. C5.3.4 includes a smoke test for this convergence.

### 3.6 Guard removal stays C5.4 — two test-only flags

The ctor guard at `silica/scheduler/batcher.py:152-166` still raises `NotImplementedError` for `has_recurrent_state=True + prefix_cache != None` constructions. **C5.3 does NOT remove this guard.**

C5.3 introduces two private constructor flags, both intended for tests only and both planned to be removed alongside the guard at C5.4:

- `_allow_recurrent_prefix_cache_for_c5_3_testing: bool = False` — bypasses the ctor guard so a `RadixPrefixCache` may be wired alongside a hybrid adapter for the prefix-hit-admission test path. Production paths still hit `NotImplementedError`.
- `_force_recurrent_slice_prefill_for_c5_3_oracle: bool = False` — flips the slice-regime predicate for hybrid + `prefix_cache=None`. Used ONLY by the §4.4 / §4.5 oracle batchers so they run slice-regime alongside no prefix cache, isolating the cooperation effect from the regime effect. Production paths never set this flag; production hybrid + `prefix_cache=None` stays contiguous (§3.2).

The two flags are independent: subject batchers in oracle tests set `_allow_recurrent_prefix_cache_for_c5_3_testing=True` only; oracle batchers set `_force_recurrent_slice_prefill_for_c5_3_oracle=True` only. Their existence is a test-only signal that C5.3 isn't fully landed; both are removed at C5.4.

## 4. Sub-unit breakdown

C5.3 is the integrative payoff for P-3-C5; the work is sliced for incremental review.

### 4.1 C5.3.0 — `_Node` + tree-level surface (no batcher)

**Goal**: extend the radix tree's data shape, lookup, and insert surfaces without touching the batcher.

**Scope**:

- Add `recurrent_snapshot: RecurrentSnapshot | None = None` to `_Node.__slots__` and `_Node.__init__`.
- Add `RadixPrefixCache.lookup_with_node(tokens) -> tuple[PrefixHit, _Node | None]` that returns the deepest matched node alongside the existing hit info. The existing `lookup` stays unchanged for non-recurrent callers.
- Add `RadixPrefixCache.peek_with_node(tokens) -> tuple[PrefixHit, _Node | None]` (side-effect-free sibling of `peek`). Used by the Phase-B classifier per §3.5.
- Extend `RadixPrefixCache.insert_detached(tokens, detached_blocks, recurrent_snapshots=None)` with an optional per-block snapshot list. Three behaviours:
  - **New node** (no existing child): if `recurrent_snapshots[i]` is set, attach to the new `_Node`.
  - **Duplicate-prefix branch with backfill** (existing child, existing.recurrent_snapshot is None, new is non-None): backfill the existing node. Per §3.5.1.
  - **Duplicate-prefix branch with both non-None**: keep existing snapshot, drop new. No equality / boundary check today (the value object lacks the metadata to make such a check mechanical — see §3.5.1).
- `recurrent_snapshots=None` (default) preserves today's behaviour bit-for-bit.
- No batcher change yet; no in-flight capture; no restore call; no Phase-B classifier change.

**Acceptance**:

- Unit tests on `RadixPrefixCache` covering attach-on-new-node, lookup_with_node returns deepest, peek_with_node side-effect-free, eviction-clears-snapshot, duplicate-insert backfills None, duplicate-insert keeps existing when both non-None.
- No real-model load required.

### 4.2 C5.3.1 — In-flight snapshot capture in `_BatchRow` (slice-prefill regime)

**Goal**: capture per-block snapshots during the row's prefill / decode forwards, gated by `isinstance(self._adapter, RecurrentStateAdapter) and self._prefix_cache is not None` per §3.2.

**Scope**:

- Extend `_BatchRow` with two fields:
  - `recurrent_snapshots_per_block: dict[int, RecurrentSnapshot] = field(default_factory=dict)` — keyed by **absolute block index**, not list position. Per §3.2.1.
  - `absolute_consumed_tokens: int = 0` — running count of tokens this row's cache has consumed. Incremented after every successful forward call (slice or decode-step). Used by the snapshot-boundary trigger to decide when to capture.
- In `_prefill_phase` and `_admit_miss_cohort`, when slice-prefill regime is active: replace the single `forward_batched(prompt, cache)` with a slice loop:
  - Compute `slice_starts = range(0, T, block_size)`.
  - For each slice, call `forward_batched(model, prompt_2d[:, start:end], cache)`, then `adapter.snapshot_recurrent_state(cache, row_idx)`, then store at `row.recurrent_snapshots_per_block[absolute_block_index]` where `absolute_block_index = row.absolute_consumed_tokens // block_size` *immediately before* the snapshot is taken (i.e., the index of the block this slice just completed). Increment `row.absolute_consumed_tokens += block_size_actually_consumed`.
  - The final slice may be shorter than `block_size`; do NOT capture a snapshot at its end (mid-block boundary).
- Decode-step path: each decode-step consumes 1 token. Increment `absolute_consumed_tokens` by 1. When `absolute_consumed_tokens % block_size == 0` after a decode-step, capture a snapshot at the corresponding absolute block index.
- `_admit_single_hit_row` seeding: when admitting a hit row, populate `row.recurrent_snapshots_per_block[i]` for `i` in `0..K-1` by copying from the matching ancestor nodes' `_Node.recurrent_snapshot` (could be `None` for legacy nodes; that's fine, the dict just lacks that key). Set `row.absolute_consumed_tokens = K * block_size` so subsequent suffix-prefill slice counters start at the right boundary.

**Acceptance**:

- Synthetic-adapter test on a miss-from-zero row: B=1 prefill of `3*block_size` tokens, verifies `row.recurrent_snapshots_per_block.keys() == {0, 1, 2}` and each snapshot's marker reflects its absolute boundary.
- Synthetic-adapter test on a hit-admitted row: `K=2` cached prefix, suffix of `2*block_size + 5` tokens, verifies `row.recurrent_snapshots_per_block.keys() == {0, 1, 2, 3}` (K=0,1 seeded from ancestors; K=2,3 captured during suffix prefill; trailing 5-token chunk does not produce a snapshot).
- Synthetic-adapter test on decode-step counter: drive prefill that lands on a non-block-aligned end (e.g., `2*block_size + 5` tokens), then `block_size - 5` decode steps, verifies a snapshot appears at absolute block index 2 once the decode counter rolls over.

**Risk note**: B>1 batched prefill via slicing will re-issue forwards with the SAME left-padding shape per slice — needs careful per-slice rebuild of `tokens_2d` from the active subset. C5.3.1 restricts the slice-prefill regime to B=1 cohorts. Bigger cohorts continue to use contiguous prefill (correctness preserved; capture skipped; rows admitted via miss with no snapshots; route-to-miss kicks in for any future hit on these rows).

**B>1 slicing scope decision**: B>1 cohort slicing is a **post-C5.3 backlog item**, NOT a C5.3.4 prerequisite. C5.3.4's smoke test (§4.5) admits the two requests sequentially (request 1 alone in B=1, then request 2 alone in B=1 after the first preempts) so B=1-only slicing is sufficient to close the C5.3 acceptance gates. Production multi-request workloads where two simultaneous arrivals would hit the same B=2 miss cohort will get capture-skipped on that cohort and benefit from prefix hits only after one of them preempts and re-extracts — a real but non-blocking limitation, tracked under §5.2 below for a follow-on sub-unit (working name C5.5; sequencing TBD relative to P-7).

### 4.3 C5.3.2 — `_extract_and_insert_prefix` passes snapshots through

**Goal**: when `_extract_and_insert_prefix` inserts block-aligned K/V, it also passes the matching per-block snapshots to `insert_detached`.

**Scope**:

- Build a per-block snapshot list keyed by absolute block index for the blocks being inserted: `[row.recurrent_snapshots_per_block.get(i) for i in range(aligned_blocks)]`. Missing entries are `None` (graceful — `insert_detached`'s duplicate-prefix branch and new-node branch both handle `None` per §4.1's spec).
- Pass the list to `insert_detached(tokens_prefix, detached_blocks, recurrent_snapshots=...)`.
- Eviction continues to drop snapshots automatically (§3.4).

**Acceptance**:

- Synthetic-adapter test that drives prefill, preempt, inspects the radix tree's deepest node, asserts `recurrent_snapshot is not None` and matches the expected boundary marker.
- Synthetic-adapter test on hit-admitted-then-extracted shape: row admits via hit (K=2 cached), runs suffix that crosses 2 more block boundaries, gets preempted; verifies the radix tree has snapshots at all 4 absolute block indices (the first 2 already existed and were retained; the last 2 are freshly attached via this row's extract path).
- Backfill convergence test: insert a snapshotless prefix (simulating legacy), drive a fresh row through a prefix-hit-but-route-to-miss path (per §3.5), let it complete and re-insert; verify the tree now carries a snapshot at the previously-snapshotless boundary.

### 4.4 C5.3.3 — Phase-B classifier + restore in `_admit_single_hit_row`

**Goal**: prefix-hit admission consumes the deepest USABLE node's snapshot before suffix prefill; rows whose deepest-usable hit node lacks a snapshot are routed to the miss path BEFORE retain.

**Scope**:

- Phase-B classifier (line 1099-1121 region): keep the existing `peek` call to compute `raw.num_hit_tokens`, derive `usable = min(raw.num_hit_tokens, max_aligned)`, then under `isinstance(adapter, RecurrentStateAdapter)` re-walk via `peek_with_node(pending.prompt_ids[:usable])` and route to `miss_rows` when the returned node has `recurrent_snapshot is None`. Per §3.5 control-flow shape — two tree walks per hybrid hit pending, no half-built state.
- `_admit_single_hit_row`: replace `lookup(pending.prompt_ids[:usable_hit_tokens])` at line 1197 with `lookup_with_node(pending.prompt_ids[:usable_hit_tokens])` so the returned node is the deepest USABLE node (matching the Phase-B classifier's snapshot-presence check, not the raw deepest leaf). Under `isinstance(adapter, RecurrentStateAdapter)`, after `row_cache = build_seeded_batch_kv(...)` and before suffix prefill, call `adapter.restore_recurrent_state(row_cache, 0, deepest_usable.recurrent_snapshot)`. Use the assert-tripwire form per §3.3 (Phase-B classifier guarantees the snapshot exists; assert is for catching classifier bugs).
- Suffix prefill itself runs through the slice-prefill regime helper from §4.2 when slice-regime is active; otherwise unchanged. Decode-step counter for the admitted row continues from absolute block index `K` (set at admission per §4.2).
- Add the two test-only flags from §3.6 to `ContinuousBatcher.__init__` so the real-model byte-exact gate below can run in this same commit:
  - `_allow_recurrent_prefix_cache_for_c5_3_testing: bool = False` — bypasses the `caps.has_recurrent_state + prefix_cache != None` ctor guard at line 152-166.
  - `_force_recurrent_slice_prefill_for_c5_3_oracle: bool = False` — flips the slice-regime predicate for hybrid + `prefix_cache=None`, used only by the oracle batcher in the byte-exact gate below.
  Both flags default to False; production paths leave them at the default. Their existence is gated for removal at C5.4 alongside the underlying guard.

**Acceptance — within slice-prefill regime, byte-exact-vs-oracle gate**:

Real-model test on Qwen3.5-0.8B using the two test-only flags from §3.6 / §4.4. Both batchers run **slice-prefill regime** for hybrid adapters (same regime, only the prefix-cache cooperation differs):

- **Subject**: `prefix_cache=RadixPrefixCache(...)` + `_allow_recurrent_prefix_cache_for_c5_3_testing=True` (bypass guard). Slice-regime active by the production predicate's `prefix_cache is not None` clause. Two requests share a long prompt; the second hits the radix tree's snapshot, admits via `_admit_single_hit_row`, restores recurrent state, runs suffix slice-prefill.
- **Oracle**: `prefix_cache=None` + `_force_recurrent_slice_prefill_for_c5_3_oracle=True` (test-only flip). Slice-regime active via the test-only predicate clause. Same prompts, same seeds; the second request prefills via the slice-regime miss path (no hit available because no prefix cache). The flag is the only deviation from production; in production, hybrid + `prefix_cache=None` stays contiguous.
- **Comparison**: after both batchers process request 2, snapshot the row's DeltaNet recurrent slots from each `_batch_cache` and assert `mx.array_equal` per layer. **This is byte-exact within slice-prefill regime.**

A token-stream sanity row sits alongside the byte-exact gate: subject's emitted tokens (prefix_cache=True) and oracle's emitted tokens (prefix_cache=None, slice-regime) match bit-by-bit. Drift experiment context applies — both regimes are slice-prefill, so this is the SAME regime; drift across regimes (slice vs contiguous) is a separate question and is NOT a C5.3 gate. Token-stream-vs-contiguous-regime is a separate diagnostic test that may report differences within the sub-rounding band the drift experiment characterised; it is informational only.

**Acceptance — fallback path**:

Synthetic-adapter test that fakes a snapshotless deepest-usable node (insert a prefix manually with `recurrent_snapshots=[None]*K`), then admits a request whose prompt matches that prefix. Verify:

- The pending lands in `miss_rows`, not `hit_rows` (Phase-B classifier routed it).
- `prefix_hits` counter is unchanged.
- The miss-path admission completes normally; the row's slice-prefill captures fresh snapshots that backfill the snapshotless node via `_extract_and_insert_prefix`.

**Acceptance — clamp correctness (regression for the P2 #1 finding)**:

Synthetic-adapter test, `block_size=4`. Insert a 2-block (8-token) prefix manually with `recurrent_snapshots=[snap_b0, snap_b1]` — the 4-token-depth node and the 8-token-depth node both carry snapshots. Submit a request whose prompt is exactly those 8 tokens (so `raw.num_hit_tokens=8` but `max_aligned=((8-1)//4)*4=4`, hence `usable=4`). Verify:

- Phase-B classifier checks `peek_with_node(prompt_ids[:4])`, which returns the 4-token-depth node — NOT the 8-token leaf. Snapshot at the 4-token node is non-None, so the pending lands in `hit_rows`.
- `_admit_single_hit_row` calls `lookup_with_node(prompt_ids[:4])` and consumes `snap_b0` for restore — NOT `snap_b1`.
- The mirror case: insert the same 8-token prefix with `recurrent_snapshots=[None, snap_b1]` (4-token-depth node has no snapshot, 8-token leaf does). Phase-B routes the pending to `miss_rows` (because the deepest-usable node, the 4-token one, has `recurrent_snapshot is None`) — NOT to hit, even though the raw deepest leaf has a snapshot.

### 4.5 C5.3.4 — Vertical slice + smoke convergence

**Goal**: end-to-end Qwen3.5-0.8B + `prefix_cache=True` smoke test that exercises C5.3.0–.3 together. The two test-only flags landed in C5.3.3 (§4.4); this sub-unit consumes them, does NOT introduce them.

**Scope**:

- Smoke test: drive Qwen3.5-0.8B with two sequentially-admitted requests sharing a long prompt (request 1 admits, completes / preempts and seeds the radix tree; request 2 then admits via prefix-hit). Assert request 2 admits via prefix-hit, post-restore recurrent state byte-equals the slice-regime oracle, token stream byte-equals the slice-regime oracle.
- Backfill convergence smoke: pre-load the radix tree with a snapshotless prefix; drive a fresh request, confirm route-to-miss; drive a second fresh request with the same prefix, confirm route-to-hit (snapshot was backfilled by the first request's extract).

**Acceptance**:

- Smoke tests green on Qwen3.5-0.8B; HF-cache-gated.
- No regression in `prefix_cache=None` non-recurrent paths (existing `tests/test_qwen3_5_preempt_replay.py`, `tests/test_qwen3_5_recurrent_snapshot.py`, `tests/test_batcher.py` all still pass).
- Production `prefix_cache=None` hybrid path stays on contiguous prefill (no slice-regime activation, no snapshot capture). Verified by a synthetic-adapter test that constructs the batcher without either C5.3 test-only flag and confirms the row's `recurrent_snapshots_per_block` stays empty after a hybrid-adapter prefill. Tokens match the pre-C5.3 baseline bit-for-bit because no path changed.
- `_force_recurrent_slice_prefill_for_c5_3_oracle=True` hybrid path (test-only — the oracle batcher in §4.4) produces tokens that match the production `prefix_cache=None` contiguous-regime tokens within bf16 greedy. This is a sanity check on the cross-regime drift band the experiment characterised, not a C5.3 acceptance gate.

## 5. Risks + open questions

### 5.1 R-C5.3-1 — slice-prefill perf cost

`block_size` token slicing adds Python + GPU launch overhead per chunk. On Qwen3.5-0.8B with `block_size=16` and a 128-token prompt, that's 8 forward calls instead of 1. Per-call overhead might be ~5-10ms; total ~40-80ms added to prefill.

**Mitigation paths if measured cost is unacceptable** (none of these are inner-forward-callback — that option is ruled out in §2.2):

- **Larger `block_size`**: doubles chunk size, halves slicing overhead. Side effect: coarser radix-tree granularity (fewer hit opportunities on partial-prefix matches). `block_size` is already a knob on `RadixPrefixCache`; this is a measurement-driven retune, not a new mechanism.
- **Snapshot only at the FINAL block boundary** (degraded mode): capture a single snapshot at the row's last completed block boundary instead of every block. Reduces overhead but eliminates partial-prefix hit support — only exact-prompt-reuse benefits. Acceptable as a fallback if full per-boundary snapshots prove too expensive; documented as a future C5.3+ option.
- **Skip slice-prefill on rows that the budgeter knows will not be cached** (e.g., `max_tokens` very small, single-shot requests). Requires admission-time signal; out of scope for first C5.3 land.

**C5.3.1 acceptance includes a wall-clock measurement row** so the cost is concrete before we commit downstream sub-units to it.

### 5.2 R-C5.3-2 — B>1 cohort prefill slicing

A miss cohort with B=4 rows of different lengths is sliced into chunks where the SHORTEST row's prompt completes first. Subsequent slices process only the rows still active. This requires per-slice `tokens_2d` rebuilding with the active subset — non-trivial.

**Scope status**: post-C5.3 backlog (working name C5.5), NOT a C5.3.4 prerequisite. C5.3.1 lands B=1 slicing only; bigger cohorts run contiguous prefill and skip snapshot capture. C5.3.4's smoke test sequences admissions B=1-at-a-time so this gap does not block the smoke. The functional consequence in production is that simultaneous-arrival cohorts pay full prefill the first time and only enjoy prefix hits after one row preempts and re-extracts; correctness is preserved. C5.5's sequencing relative to P-7 is TBD.

### 5.3 R-C5.3-3 — opt-in flag rot

The `_allow_recurrent_prefix_cache_for_c5_3_testing` flag is a test-only escape hatch. C5.4 must remove both the flag and the underlying guard simultaneously; if C5.4 is delayed, the flag becomes a footgun for production code that finds and uses it.

**Mitigation**: name the flag explicitly `_for_c5_3_testing`. Document its temporary nature in the docstring. Add a test that fails if the flag survives past v1.x.x (where x.x is the version C5.4 lands at — TBD).

### 5.4 Open — eviction memory cap

A radix tree with deep prefixes accumulates one snapshot per node. On Qwen3.5-0.8B, ~19.5MB per snapshot (per `docs/P3_C5_STATE_INVENTORY/README.md` §6). A 16-block tree × multiple unique prefix paths could pin hundreds of megabytes.

**Open question for C5.3.4**: should snapshot retention be a separate eviction policy from K/V retention, or unified? Today the LRU evicts both together via `_evict_node`. If snapshots need a tighter budget than K/V, a separate "evict snapshot but keep K/V (forces fallback)" option may be needed.

## 6. Test-scaffold sketch

### 6.1 New test file `tests/test_radix_prefix_cache_recurrent_snapshot.py`

- `TestNodeSnapshotSlot`: `_Node` carries a `recurrent_snapshot` slot; default `None`; assignable.
- `TestInsertDetachedWithSnapshots`: `insert_detached(tokens, blocks, recurrent_snapshots=...)` attaches per-block; deepest node's snapshot is the one matching the last input.
- `TestLookupWithNode`: `lookup_with_node` returns `(PrefixHit, _Node)` where the node is the deepest hit.
- `TestEvictionDropsSnapshot`: evict a leaf, snapshot reference released (verified via weakref or by counting tree node total).

### 6.2 New test file `tests/test_qwen3_5_prefix_cache_hit_admission.py`

- `TestSliceCaptureSynthetic`: spy adapter records snapshot calls during a prefill; assert one call per `block_size` boundary; assert snapshots attached to `_BatchRow.recurrent_snapshots_per_block`.
- `TestExtractInsertPropagatesSnapshots`: drive prefill + preempt; assert the deepest tree node has `recurrent_snapshot is not None`.
- `TestHitAdmissionRestoresSnapshot` (real Qwen3.5-0.8B; HF-cache-gated): drive a prefix-hit admission and assert subject's row_cache after admission byte-equals the **slice-regime** oracle's cache at every DeltaNet layer. Subject batcher: `prefix_cache=RadixPrefixCache(...)` + `_allow_recurrent_prefix_cache_for_c5_3_testing=True`. Oracle batcher: `prefix_cache=None` + `_force_recurrent_slice_prefill_for_c5_3_oracle=True` so it runs the SAME slice-prefill regime as the subject (per §4.4). Do NOT compare against a `prefix_cache=None` contiguous-regime oracle — that gate was deliberately demoted in §2.2 / §3.2 because slice-vs-contiguous trajectories diverge at fp32 by construction.
- `TestFallbackOnMissingSnapshot`: deepest-usable node has `recurrent_snapshot=None`; admission falls back to miss path (Phase-B classifier route per §3.5); sanity tokens still match.

### 6.3 Smoke test additions

`tests/test_p3_hybrid_batched_smoke.py` (existing) currently runs Qwen3.5-0.8B with `prefix_cache=None`. C5.3.4 adds a sibling test with `prefix_cache=RadixPrefixCache(...)` + the opt-in flag. Skips on HF cache absence.

## 7. Implementation pause point

This design note is the C5.3 entry gate. User reviews:

- §3.1 storage choice (`_Node` field vs Protocol extension)
- §3.2 capture-site choice (slice-prefill vs forward callback)
- §3.3 restore-site shape (deepest-usable-node consumption)
- §3.5 fallback semantics (refuse hit + fall back to miss path)
- §3.6 opt-in flag mechanism
- §4 sub-unit decomposition (5 sub-units)

On approval, implementation starts at C5.3.0 (smallest, isolated to `RadixPrefixCache`).

C5.3.0 lands without touching the batcher; C5.3.1 / .2 / .3 land sequentially with their own pause points (C5.3.3 carries the two test-only flags alongside the classifier + restore wiring it gates); C5.3.4 lands the smoke / backfill-convergence tests and consumes the flags introduced in C5.3.3. Each sub-unit is its own commit, in keeping with the C5.0 / C5.1 / C5.2 cadence.
