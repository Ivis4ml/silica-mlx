# Unit 16c.2 step 4 — batcher surgery skeleton

**Date:** 2026-04-17
**Status:** structural outline — not code; reviewer-first.
**Depends on:** steps 2 (`a30e7bd`), 3 (`4f67e3a`), 3.5 (`6450b8a`).
**Blocks:** step 5 real-model acceptance, PLAN §7 P-2 acceptance #2.

## Why a skeleton

Step 4 touches the admission path (prefix hit / seed / suffix prefill),
the reclaim path (extract before filter / insert_detached), one new
field on `ContinuousBatcher`, and one new knob on `Engine.generate_batch`.
If we start writing code before the boundaries are pinned, the diff
merges four concerns into one giant commit and failures will not
localise. This skeleton pins:

1. Files and functions that change (signatures only).
2. Phase ordering inside `step()` and `_admit_waiting_requests`.
3. New invariants on top of 16c.1's I-1 … I-5.
4. The pytest names (not bodies) that encode the contract.
5. Open decisions that need resolution before the implementation PR.

No prose beyond what pins a decision. No code. Step 4 implementation
starts when this doc is approved.

---

## 1. Files touched

| File | Role in step 4 |
|---|---|
| `silica/scheduler/batcher.py` | Admission + reclaim surgery + one new ctor arg |
| `silica/engine/__init__.py` | `generate_batch(..., prefix_cache=...)` pass-through |
| `silica/scheduler/seed_kv.py` | Unchanged — the tested primitive from step 3.5 |
| `silica/kvcache/prefix.py` | Small public helper: `fetch_detached_blocks` (see §3.6) — keeps the batcher from reaching into `self._prefix_cache._store`. |
| `silica/kvcache/store.py` | Unchanged |
| `tests/test_batcher.py` | New tests for prefix-hit admission + reclaim-insert |
| `tests/test_engine_generate_batch.py` | New tests for `prefix_cache` kwarg |
| `tests/test_p2_batched_parity.py` | New real-model parity test (step 5) |

No new files. Step 4 is deliberately purely additive on existing
modules — if we can't express it inside `batcher.py` + `engine/__init__.py`
then the abstraction boundary is wrong and we pause.

---

## 2. New batcher state

`ContinuousBatcher.__init__` grows one optional keyword argument:

```python
def __init__(
    self,
    adapter: ModelAdapter,
    *,
    sampler: Sampler | None = None,
    weight_provider: WeightProvider | None = None,
    max_batch_size: int = 1,
    prefix_cache: RadixPrefixCache | None = None,  # NEW
) -> None: ...
```

When `None`, every prefix code path short-circuits — the batcher
behaves bit-identically to its pre-step-4 self. Existing 16a / 16b /
16c.1 tests do not pass `prefix_cache`; they continue to exercise the
no-cache path and must remain green.

One new instance field:

```python
self._prefix_cache: RadixPrefixCache | None = prefix_cache
```

Plus two counters for observability (consumed by step 5's acceptance):

```python
self.forward_prompt_tokens: int = 0   # tokens fed through prefill forwards
self.prefix_hits: int = 0             # admissions that used the hit path
```

No new enum values, no new `RequestState` transitions. 16d adds
`PREEMPTED`; step 4 does not.

---

## 3. Method changes — batcher.py

### 3.1 `_admit_waiting_requests()` — the admission rewrite

Today (16c.1): pop up to `capacity` admits → build `BatchKVCache(B=K)`
from scratch → one batched prefill over all K rows → extend into main.

Step 4: split admits into **hit rows** and **miss rows** based on
`prefix_cache.peek`. Miss rows go through the existing K-row path;
hit rows go through a new per-row path.

Phase order inside `_admit_waiting_requests`:

```text
1. compute capacity; pop up to `capacity` pending admits.
2. if self._prefix_cache is None:
       miss_rows = admitted; hit_rows = []
   else:
       for r in admitted:
           raw = prefix_cache.peek(r.prompt_ids)
           max_aligned = ((len(r.prompt_ids) - 1) // block_size) * block_size
           usable = min(raw.num_hit_tokens, max_aligned)
           if usable == 0:
               miss_rows.append(r)
           else:
               hit_rows.append((r, usable))
3. for (r, usable) in hit_rows:
       events += _admit_single_hit_row(r, usable)   # NEW helper; counter bumped inside
4. if miss_rows:
       events += _admit_miss_cohort(miss_rows)      # current 16c.1 logic, renamed;
                                                    # counter bumped inside
5. self._rebuild_slot_table(); return events
   # No counter update at this level — each helper owns its own
   # accounting (forward_prompt_tokens, prefix_hits). Double-counting
   # by bumping here would be a silent acceptance-test regression.
```

### 3.2 New helper — `_admit_single_hit_row(row, usable_hit_tokens)`

Per-row admission path (one hit at a time; K=1). Sequence is pinned
by docs/P2_UNIT_16C_2_PREP.md §4, wrapped in ``try/finally`` so a
mid-sequence exception cannot strand retained hit refs in the store:

```text
1. hit = self._prefix_cache.lookup(row.prompt_ids[:usable_hit_tokens])
   try:
2.     assert hit.num_hit_tokens == usable_hit_tokens     # invariant S-1
3.     detached = self._prefix_cache.fetch_detached_blocks(hit.block_ids)
4.     row_cache = build_seeded_batch_kv(detached, num_layers=N)
5.     suffix_tokens = row.prompt_ids[usable_hit_tokens:]
6.     suffix_arr = mx.array([suffix_tokens], dtype=mx.int32)
       logits = forward_batched(self._model, suffix_arr, list(row_cache))
7.     events = self._sample_and_emit_rows(
           [row_batch_row], logits, is_prefill=True
       )
8.     if self._batch_cache is None:
           self._batch_cache = row_cache
       else:
           for layer in range(N):
               self._batch_cache[layer].extend(row_cache[layer])
9.     self._rows.append(row_batch_row)
10.    self.prefix_hits += 1
       self.forward_prompt_tokens += len(suffix_tokens)
   finally:
11.    self._prefix_cache.release(list(hit.block_ids))
```

**Placement of the S-1 assertion inside `try`, not between `lookup`
and `try`.** ``lookup()`` already retained one hit per block; if we
assert OUT of the ``try``, a failing S-1 leaks every retained ref
because ``finally`` never runs. The ``try`` window must start on the
line immediately after `lookup` — every statement that can raise
between retain and release belongs inside.

**Why release in `finally` is safe.** `release_hit` only decrements
the hit counter — it does not touch source refs or detached storage.
So even if an exception aborts between lookup and extend, the block
remains a live source in the radix tree with intact detached K/V;
only the transient hit ref gets cleaned up. Because step 4 runs
entirely inside a single `step()` call (no concurrent eviction
possible), release can happen at any point after lookup without
racing anyone. Without the `finally`, a raising `forward_batched` or
`sampler.sample` or `extend` would leak hit refs and `evict_until`
would silently skip those blocks forever.

**Invariant S-2 (admission happy-path sequence).** Steps 3-9 execute
in order. Swapping 7↔8 (extend before sample) corrupts row_cache
mid-sample; swapping 6↔7 is incoherent. Release (step 11) runs
**after** step 9 in the normal case, but is permitted anywhere after
step 1 under exception paths because of the source/detached lifetime
split documented above.

### 3.3 `_admit_miss_cohort(admitted)` — rename of current 16c.1 logic

Lift current body of `_admit_waiting_requests` from today's line 456
onward into a helper. No behaviour change; just a renaming so the
split is readable. Counter bump:

```python
self.forward_prompt_tokens += sum(len(r.prompt_ids) for r in admitted)
```

### 3.4 `_reclaim_terminated()` — the insert-before-filter hook

Today: detect terminals → compute `kept` → `filter` + rebuild slot_table.

Step 4: before the filter, extract detached K/V from each terminal
row's block-aligned prefix and push to `insert_detached`. This is the
source of future prefix hits.

Phase order:

```text
1. if no terminal rows: return.
2. kept = [...]; terminated = [...]  # indices
3. if self._prefix_cache is not None and self._batch_cache is not None:
       for row_idx in terminated:
           _extract_and_insert_prefix(row_idx)   # NEW helper
4. (rest unchanged: filter OR batch_cache=None + rows=[]; rebuild slot_table)
```

### 3.5 New helper — `_extract_and_insert_prefix(row_idx)`

Two subtleties pinned by 16c.1's sample/forward timing and mlx-lm's
left-padding layout (see `BatchKVCache.keys` shape commentary):

- **Token/KV alignment.** At reclaim time, cache holds K/V for
  `prompt_ids + generated[:-1]`. The last `generated` entry was
  sampled from the previous step's forward logits and never fed
  back through any forward — its K/V is NOT in cache. Using
  `prompt_ids + generated` as the "computed ids" would push one
  extra token into the radix tree without backing K/V, corrupting
  every future prefix hit that lands on that block.
- **Left-padding offset.** Each layer's `keys` tensor is
  `(B, H, T_max, D)` with row `row_idx`'s logical token 0 located at
  axis-2 index `left_padding[row_idx]`. Slicing `keys[row_idx, :,
  0:block_size, :]` when `left_padding[row_idx] > 0` pulls from the
  pad zeros, not from the prompt's first block. `mlx-lm`'s internal
  shift-left after filter can lower the min-left-pad over time but
  each row's own `left_padding[row_idx]` keeps tracking where its
  content starts.

```text
1. row = self._rows[row_idx]
2. computed_ids = row.prompt_ids + row.generated[:-1] if row.generated \
                  else list(row.prompt_ids)
3. computed_len = int(self._batch_cache[0].offset[row_idx].item())
   assert len(computed_ids) == computed_len          # invariant S-3a
4. aligned_tokens = (computed_len // block_size) * block_size
5. if aligned_tokens < block_size: return            # nothing aligned to save
6. base = int(self._batch_cache[0].left_padding[row_idx].item())
   # Note: base is per-row. Every layer's BatchKVCache shares the
   # same (B, T_max) shape and the same left_padding vector (they
   # are constructed and mutated together by extend/filter), so
   # layer 0's base is authoritative for all layers.
7. tokens_prefix = computed_ids[:aligned_tokens]
   detached_blocks: list[list[(K, V)]] = []          # [block][layer]
   for b_idx in range(aligned_tokens // block_size):
       start = b_idx * block_size
       end = start + block_size
       per_layer = []
       for l in range(num_layers):
           keys_l = self._batch_cache[l].keys
           vals_l = self._batch_cache[l].values
           k = mx.contiguous(keys_l[row_idx:row_idx+1, :, base + start : base + end, :])
           v = mx.contiguous(vals_l[row_idx:row_idx+1, :, base + start : base + end, :])
           per_layer.append((k, v))
       detached_blocks.append(per_layer)
8. mx.eval(*[arr for per_layer in detached_blocks for (k, v) in per_layer for arr in (k, v)])
9. self._prefix_cache.insert_detached(tokens_prefix, detached_blocks)
```

**Invariant S-3a:** `len(prompt_ids) + len(generated[:-1]) ==
offset[row_idx]`. This is an assertion on our own accounting against
mlx-lm's authoritative `offset` field — if it ever fails, either our
`generated` list is out of sync with forward calls or a stray
`update_and_fetch` happened that we did not record. Loud-fail.

**Invariant S-3b (eager materialisation before filter):** `mx.eval`
in step 8 runs BEFORE the `filter` in `_reclaim_terminated`'s later
lines. Gate-0.75 probe B proved slices survive the source's filter
once they are materialised. Without the eval the slices are graph
references and filter's shift-left mutation silently corrupts them.

### 3.6 New helper — `RadixPrefixCache.fetch_detached_blocks(block_ids)`

Thin public method on `RadixPrefixCache` (lives in
`silica/kvcache/prefix.py`) that returns per-block detached K/V for
a sequence of hit block ids:

```python
def fetch_detached_blocks(
    self, block_ids: Sequence[int]
) -> list[Sequence[tuple[mx.array, mx.array]]]:
    """Per-block detached K/V for the given hit block ids.

    Shape matches what ``build_seeded_batch_kv`` expects:
    indexed ``[block_idx][layer_idx]`` → ``(K, V)``.

    Caller must have just obtained the ids via ``lookup`` (so the
    hits are retained). Raises ``KeyError`` if any id has no
    detached K/V — that would indicate the Paged backend is in use
    on a synthetic-only code path, which is a caller bug.
    """
    return [self._store.fetch_detached(b) for b in block_ids]
```

**Why add this method instead of letting the batcher call
`self._prefix_cache._store.fetch_detached(...)` directly.** The
Protocol + store split shipped in step 2 exists so
RadixPrefixCache is the one layer that knows about its backing
store. Breaking that encapsulation in the batcher means future
changes to the Paged backend (detached K/V model for the
trigger-gated paged-attention track) have to touch the batcher too.
The one-line helper costs nothing and keeps the layers clean.

Step 3 test file `tests/test_prefix_cache_synthetic.py` gets one
new test: `test_fetch_detached_blocks_returns_per_block_kv`.

---

## 4. Method changes — engine/\_\_init\_\_.py

Signature addition:

```python
def generate_batch(
    self,
    prompts: Sequence[str],
    params: SamplingParams | list[SamplingParams] | None = None,
    *,
    max_batch_size: int | None = None,
    prefix_cache: RadixPrefixCache | None = None,  # NEW
) -> Iterator[BatchEvent]: ...
```

Default `None` preserves 16c.1 behaviour. When passed, forwarded
1:1 to `ContinuousBatcher(prefix_cache=prefix_cache)`.

Engine does NOT construct a prefix_cache on behalf of the caller —
the cache's lifetime spans multiple `generate_batch` calls (that's the
whole point of prefix reuse), so ownership lives with whoever
constructs the Engine instance (bench harness, user code, test
fixture). Engine is transport only.

---

## 5. Invariants — additions to 16c.1's I-1 … I-5

(Labelling as `S-1 … S-6` so they stand apart from 16c.1's `I-n`.)

- **S-1 (lookup agrees with peek).** `lookup(prompt_ids[:usable_hit_tokens]).num_hit_tokens`
  must equal `usable_hit_tokens`. Divergence means the radix tree
  mutated between peek and lookup — should not happen because admission
  runs inside a single `step()` call with no concurrent mutation.
  Asserted; loud-fail.

- **S-2 (admission sequence + finally-release).** Happy path order for
  a hit row: `lookup` → `try:` (S-1 assertion, seed, suffix-prefill,
  sample, extend, counters) → `finally: release`. The `try` block
  starts on the line immediately after `lookup`; the S-1 assertion
  itself lives INSIDE the try. Placing it outside would defeat the
  finally-release guarantee — a failing assertion would abort before
  the finally is registered and leak every retained hit ref.
  `release_hit` does not touch source or detached storage, so it is
  safe to run under every exception path.

- **S-3a (cache contents vs our accounting).** At reclaim time,
  `computed_ids = prompt_ids + generated[:-1]` and
  `len(computed_ids) == BatchKVCache.offset[row_idx]`. The last entry
  in `row.generated` was sampled from logits but never fed through a
  forward — its K/V is NOT in cache. Asserting against mlx-lm's
  `offset` field catches accounting drift loudly.

- **S-3b (eager materialisation before filter, with left_padding).**
  Every `insert_detached` runs after its K/V slices have been
  `mx.eval`'d (Gate-0.75 probe B) and before the source cache's
  `filter` call. Slice indexing uses `base + start : base + end`
  where `base = left_padding[row_idx]` — slicing from axis-2 zero
  would pull pad values, not prompt content.

- **S-4 (capacity accounting includes hits and misses).** `capacity =
  max_batch_size - len(self._rows)` is checked AT POP TIME, BEFORE
  the hit/miss split. Hit rows do not cost extra capacity (they still
  occupy one `self._rows` entry), and the split does not drain more
  than `capacity` admits from the waiting queue.

- **S-5 (prefix hits skip forward tokens).** For a hit row with
  `usable_hit_tokens = U` and prompt length `L`, exactly `L - U`
  tokens are fed through the model in prefill — never 0, never L. The
  `forward_prompt_tokens` counter is the observation surface for
  acceptance test #11 (step 5).

- **S-6 (cache-None invariance).** If `prefix_cache=None` (default),
  every step-4 code path must be bit-identical to the 16c.1 code path.
  This is enforced by the existing 16c.1 test suite remaining green —
  no step-4 test may change its fixture to pass a prefix cache unless
  it's explicitly a step-4 test.

- **S-7 (intra-step event grouping, not admit-order).** When a single
  `step()`'s admission phase contains both hit and miss rows, events
  are emitted in **phase-grouped order**: all hit-row events first
  (per-row processing), then the miss cohort's events (single batched
  forward). Events are NOT interleaved in admitted-queue order.

  **Per-row ordering is preserved**: within one row the sequence is
  always `token → done` in the correct causal order; a row's events
  never interleave with its own later events. The guarantee we drop
  is cross-row: if admits arrive as `[miss_A, hit_B, miss_C]`, output
  during that step's admission is `[B's events, A's events, C's
  events]`, not `[A, B, C]`. `test_mixed_hit_and_miss_rows_in_one_admit_call`
  asserts per-row correctness + admission capacity, NOT cross-row
  interleave.

  *Why this trade-off:* miss cohort needs a batched forward over all
  miss rows (that is the whole point of 16c.1's K-row prefill). Hit
  rows are inherently per-row (each has a distinct seeded cache and
  distinct suffix length). Preserving admit-order would force K
  separate forwards for the miss side, defeating cohort batching. The
  cheap compromise — and the one every vLLM-lineage scheduler makes
  — is to group by phase and document the ordering contract.

---

## 6. Test names (bodies deferred to implementation PR)

### 6.1 `tests/test_batcher.py` — new tests

```
# Hit path admission
test_admission_with_no_cache_matches_16c1_behaviour
test_admission_with_empty_cache_routes_all_to_miss_path
test_single_hit_row_admission_skips_prefix_tokens_in_forward
test_full_hit_reservation_keeps_one_block_of_suffix_prefill
test_unaligned_prompt_with_full_aligned_hit_retains_all_aligned_blocks
test_hit_row_extends_main_cache_preserving_live_rows_I2
test_multiple_hit_rows_in_one_admit_call_each_admit_independently
test_mixed_hit_and_miss_rows_in_one_admit_call
test_hit_admission_releases_hit_refs_on_forward_error     # S-2 finally
test_hit_admission_releases_hit_refs_on_sampler_error     # S-2 finally
# See §5 S-7 for the event-order guarantee this test DOES and does NOT assert.

# Reclaim path
test_reclaim_extracts_and_inserts_detached_before_filter
test_reclaim_insert_detached_is_filter_safe
test_reclaim_with_no_cache_does_not_mutate_prefix_state
test_reclaim_unaligned_terminal_prefix_drops_partial_block
test_reclaim_full_terminal_cohort_handles_all_None_main_cache
test_reclaim_excludes_unfed_last_generated_token          # S-3a
test_reclaim_extract_respects_left_padding                # S-3b

# Invariants / counters
test_forward_prompt_tokens_counts_prefill_not_decode
test_forward_prompt_tokens_reduces_on_prefix_hit
test_prefix_hits_counter_increments_per_hit_admission
test_capacity_check_covers_hit_and_miss_split
```

### 6.2 `tests/test_engine_generate_batch.py` — new tests

```
test_prefix_cache_kwarg_accepts_none_as_default
test_prefix_cache_kwarg_forwarded_to_batcher
test_prefix_cache_lifetime_spans_multiple_generate_batch_calls
```

### 6.3 `tests/test_p2_batched_parity.py` — new real-model test (step 5)

```
test_shared_prefix_reduces_forward_tokens
```

**Driver: direct `ContinuousBatcher`, NOT `Engine.generate_batch`.**
The counter (`batcher.forward_prompt_tokens`) lives on the batcher
instance (Q-B); `Engine.generate_batch` hides the batcher inside the
method body and yields only `BatchEvent`s, so reading the counter
through the Engine surface would require either exposing an
observable (bloats the API for a test-only concern) or scraping via
a side channel. The acceptance test constructs `ContinuousBatcher`
directly, runs it through `step()` loops, and asserts on
`batcher.forward_prompt_tokens` / `radix.hits`.

`Engine` is tested separately in `test_engine_generate_batch.py` for
kwarg pass-through only (three `test_prefix_cache_kwarg_*` tests in
§6.2).

Staggered admission (`max_batch_size=1`), prefix length 128 (=8 blocks
@ block_size=16), suffix length 16. Closed-form assertion per
docs/P2_UNIT_16C_2_PREP.md §5 edge 5:

```python
aligned_prefix = (prefix_len // block_size) * block_size
usable_hit    = aligned_prefix - block_size  # S-5 + edge 1
observed = (prefix_len + suffix_len) + 3 * (prefix_len + suffix_len - usable_hit)
assert radix.hits == 3
assert batcher.forward_prompt_tokens == observed
# Per-row token sequences match no-prefix-cache baseline bit-exactly.
```

---

## 7. Implementation order (sub-commits under step 4)

Step 4 is one big commit's worth of code; we split it into four
sub-commits to keep each reviewable and bisect-able:

1. **Ctor + no-op wiring** — add `prefix_cache` kwarg to batcher and
   Engine, thread through; no admission / reclaim changes. Tests:
   the three `test_prefix_cache_kwarg_*` tests + a batcher smoke
   test pinning S-6 (no-cache path unchanged).
2. **Reclaim-path insert_detached** — add `_extract_and_insert_prefix`
   + wire into `_reclaim_terminated`. Tests: all reclaim-path tests.
   Admission still does not use the cache. This is the "fill the
   cache" half of 16c.2.
3. **Admission-path hit split** — add `_admit_single_hit_row` + rename
   current body to `_admit_miss_cohort` + split in `_admit_waiting_requests`.
   Tests: all hit-path batcher tests + counter tests. This is the
   "use the cache" half.
4. **Real-model acceptance** — `test_shared_prefix_reduces_forward_tokens`.
   PLAN §7 #2 green. This is step 5 per the original 5-step plan, but
   rolling it into step 4's sub-commits keeps the milestone local.

Each sub-commit gets its own `ask-before-commit` pause per the session
rule.

---

## 8. Open questions

**Q-A. How does the Engine/harness construct the prefix cache?**

Current plan: caller constructs
`RadixPrefixCache(block_size=B, store=SyntheticPrefixBlockStore(block_size=B))`
and passes it in. `block_size` must match the test's expectations —
we hard-code `16` in tests. For the real-model acceptance, the caller
picks a block_size that divides the Qwen3-0.6B test's prefix cleanly
(128 = 8 * 16).

No `Engine.with_prefix_cache()` convenience constructor yet — explicit
construction by the caller is clearer.

**Q-B. Do we need `forward_prompt_tokens` on `BatcherMetrics` or on
`self`?**

Resolved: direct attribute on `ContinuousBatcher` (per §2). Cleaner
than threading a metrics registry for one scalar; test code reads
`batcher.forward_prompt_tokens`. Follow-on consequence: real-model
acceptance (§6.3) drives `ContinuousBatcher` directly rather than
going through `Engine.generate_batch`, because the Engine surface
intentionally does not expose the batcher. Engine is tested only
for kwarg pass-through.

**Q-C. What happens if the prompt is ALL prefix-cached (edge-case
full-hit)?**

Covered by §5 edge 1 of the prep doc: `max_aligned = ((L-1)//B)*B`
reserves at least one block's worth of suffix (or unaligned tail).
Test: `test_full_hit_reservation_keeps_one_block_of_suffix_prefill`.
No new code path — the helper's `usable_hit_tokens` formula handles it.

**Q-D. What if multiple hit rows in one `_admit_waiting_requests`
share the same prefix blocks?**

Each row calls `lookup` independently; the store's `retain_hit`
increments per call. Release happens per-row inside the `finally`
block of §3.2 (step 11). Hit refs stay balanced because retain/release
are paired 1:1 per row. This is a consequence of L-2 being a counter
(tested in step 2's `test_hit_lifecycle_independent_of_source_count`).

**Q-E. Should we add a `RequestState` status for "hit-admitted"?**

No. The state machine stays at PREFILL → DECODE → DONE/ABORTED. A
prefix hit is a backend optimisation; no user-observable event is
emitted specifically for the hit path. The `prefix_hits` counter is
enough for acceptance-test introspection.

---

## 9. Non-goals

- Per-row `SamplingParams` on hit rows (still homogeneous per
  `generate_batch`; heterogeneous batches are P-3).
- Evicting from the prefix cache mid-`step()` to free room for a new
  admission (evict policy sits on the user; the batcher never calls
  `evict_until` in step 4).
- `prefix_cache.peek` during step()'s reclaim phase to skip
  insert_detached for already-cached prefixes — would be a small win
  and is easy to add later; not needed for PLAN §7 #2.
- Any interaction with the budget-aware preempt path — 16d's concern.

---

## 10. Ready-to-start checklist

Before starting sub-commit 1:

- [x] Step 3 merged (`4f67e3a`).
- [x] Step 3.5 merged (`6450b8a`).
- [x] Skeleton revision round 1: S-2 finally-release, S-3a/S-3b
      alignment + left_padding, real-model acceptance drives batcher
      directly, `forward_batched` call path pinned (2026-04-17).
- [x] Skeleton revision round 2: `fetch_detached_blocks` added to
      `RadixPrefixCache` (no more batcher → store reach-through),
      counter-bump removed from §3.1 aggregation path, S-7 pins
      intra-step event-grouping contract (2026-04-17).
- [x] Skeleton revision round 3: S-1 assertion moved INSIDE the
      `try` block (was leaking hit refs on assertion failure);
      Q-D and S-2 entries re-pointed from "step 7 / step 10" to
      "finally block"; step numbers in §3.2 bumped by 1 after the
      insertion (2026-04-17).
- [ ] This revised skeleton approved.
- [x] Q-A … Q-E resolved (Q-B explicitly ties acceptance to
      batcher-direct path).
- [x] Existing regression suite green on HEAD: 431 passed / 8 skipped.

Once all boxes are checked, sub-commit 1 takes ~30 minutes; sub-commit
2 ~45 minutes; sub-commit 3 ~90 minutes (the hit-split is the core
logic); sub-commit 4 depends on Qwen3-0.6B run time (~2 min per 4-row
acceptance run). Total wall-clock: ~3.5 hours for a coordinated pass.
