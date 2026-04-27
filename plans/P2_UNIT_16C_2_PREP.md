# P-2 Unit 16c.2 — prep: prefix K/V reuse over BatchKVCache

**Date:** 2026-04-17
**Status:** design prep — no code changes yet.
**Depends on:** Unit 16c.1 (`cb4a7f7`), Unit #14 RadixPrefixCache (`fd9a6f1`),
Gate-0.5 (`64e2211`).
**Blocks:** PLAN §7 P-2 acceptance #2 (prefix hit verifiable), Unit 16d
(budget-aware preemption).

## Why this document

Unit 16c.1 shipped the dynamic-cohort state machine (queue, extend, filter,
slot_table). 16c.2 adds the **physical K/V reuse path**: a newly-admitted
request with a shared prefix should skip prefill compute over that prefix
and copy existing K/V into its fresh batch row (option B from P2_OPENING
§3.2).

The hazard: `RadixPrefixCache` today takes `kv: PagedKVCache` and pins
block ids via `kv.incref/decref`. But 16c.1's batcher uses `BatchKVCache`,
not `PagedKVCache` — there is no paged-attention kernel in P-2. Without
a redesign, 16c.2 would either (a) plug a stub `PagedKVCache` in just to
satisfy the incref/decref interface, or (b) duplicate the radix tree over
a second backend. Both are bad. This doc pins the alternative:
`RadixPrefixCache` takes an abstract `PrefixBlockStore` protocol; the
batcher supplies a synthetic-id implementation that stores detached K/V
slices alongside the refcount bookkeeping.

---

## 1. Decision: `PrefixBlockStore` Protocol (α′, not α or β)

**α (rejected):** `RadixPrefixCache` takes a `refcount_backend: Callable[[int], None] | None`.
Too thin; the semantics of "is this retain for source or for hit?" are
hidden at the call site and regress readability over the current
PagedKVCache coupling.

**β (rejected):** batcher passes a `_SyntheticRefcount` helper as the `kv=`
parameter. The name `kv=` would mislead every future reader — it already
signals "PagedKVCache". Passing a non-kv helper through `kv=` is a
strictly worse naming hack than just changing the parameter.

**α′ (adopted):** `RadixPrefixCache` depends on an explicit small Protocol:

```python
from typing import Protocol

class PrefixBlockStore(Protocol):
    """Refcount + K/V-residency backend for RadixPrefixCache.

    Two independent refcount dimensions:
      - source: pinning a block as a copy-source in the radix tree
                (survives the originating request's free).
      - hit:    pinning a block because some live request is currently
                copying/consuming its K/V (lookup-release lifecycle).

    Eviction is safe only when a block has zero *hits*; source refs
    can drop to zero (via evict) to physically release the block.
    """

    block_size: int

    def retain_source(self, block_id: int) -> None: ...
    def release_source(self, block_id: int) -> None: ...
    def retain_hit(self, block_id: int) -> None: ...
    def release_hit(self, block_id: int) -> None: ...
```

Two implementations:

- **`PagedPrefixBlockStore`** — wraps `PagedKVCache`. Both `retain_source`
  and `retain_hit` call `kv.incref`; both `release_*` call `kv.decref`.
  Preserves today's PagedKVCache behaviour exactly, used by any future
  paged-attention path (P-2 Opening trigger-gated track).
- **`SyntheticPrefixBlockStore`** — owns its own `_source_refs: dict[int, int]`,
  `_hit_refs: dict[int, int]`, `_next_id: int`, and
  `_detached: dict[int, list[tuple[mx.array, mx.array]]]` (one K/V pair
  per layer). Used by 16c.2's batcher. No PagedKVCache involved.

`RadixPrefixCache.__init__(kv: PagedKVCache)` becomes
`RadixPrefixCache.__init__(store: PrefixBlockStore)`. Internally every
`kv.incref` → `store.retain_source` (in `insert`) or
`store.retain_hit` (in `lookup`); every `kv.decref` → `store.release_source`
(in `evict_until`) or `store.release_hit` (in `release`). The two refcount
dimensions become explicit in the call sites.

### Why splitting retain_source / retain_hit matters

Today, eviction looks at `self._live_hits[block_id] > 0` to decide whether
a node is evictable. That works because `_live_hits` is a prefix-cache-local
counter tracking only lookup/release pairs. With the Protocol split, the
store itself owns both dimensions: eviction can ask
`store._hit_refs[block_id] == 0` directly, and RadixPrefixCache's local
`_live_hits` bookkeeping becomes redundant. We will collapse it — one
refcount view per dimension, owned by the store.

---

## 2. Invariants (three lifetimes, must-not-conflate)

16c.2 juggles three overlapping lifetimes per block:

### L-1. **Source lifetime** — the radix node owns the block.

Starts: `RadixPrefixCache.insert` creates a new node (not on duplicate
insert of an already-present prefix).
Ends: `RadixPrefixCache.evict_until` removes the node.
While alive: `store._source_refs[block_id] >= 1` and, for the synthetic
store, `store._detached[block_id]` holds the K/V slices.

### L-2. **Hit lifetime** — a live request is copying the block.

Starts: `RadixPrefixCache.lookup` returns the block as part of its `PrefixHit`
(`retain_hit` called).
Ends: the admitting batcher calls `release(block_ids)` after it has
physically copied the K/V into the new row's prefix region (`release_hit`
called).
Invariant: `L-2 ⊆ L-1` — a hit cannot exist without a live source.

### L-3. **Detached-K/V lifetime** — physical K/V bytes held by the store.

Starts: `insert_detached` stores `_detached[block_id] = [(K_l, V_l) for l in layers]`.
Ends: `evict_until` drops the node and the store deletes `_detached[block_id]`
**in the same atomic operation** as dropping `_source_refs[block_id]`.

Critical: `L-3 == L-1` for the synthetic store. Detached K/V MUST NOT
outlive the radix node — otherwise memory leaks silently on eviction.
Similarly, a release_source that drops source refs to 0 outside of
`evict_until` is a bug (sources go to zero only via eviction in 16c.2;
the owning request's own ref is the hit-release path, not source).

---

## 3. API changes

### 3.1 `RadixPrefixCache`

```python
class RadixPrefixCache:
    def __init__(self, *, block_size: int, store: PrefixBlockStore) -> None: ...

    def peek(self, tokens: Sequence[int]) -> PrefixHit:
        """Side-effect-free lookup — no retain_hit, no _live_hits update,
        no self.hits counter increment, no LRU touch. Used by the batcher
        during admission planning before it commits to actually using the
        hit. Walks the tree, returns (block_ids, num_hit_tokens)."""

    def lookup(self, tokens: Sequence[int]) -> PrefixHit:
        """Retained walk — retain_hit per block, self.hits += 1, LRU touch.
        Use this when the batcher has decided to actually copy the K/V
        into a row. Must be paired with eventual release(block_ids)."""

    def insert_detached(
        self,
        tokens: Sequence[int],
        detached_blocks: Sequence[Sequence[tuple[mx.array, mx.array]]],
    ) -> Sequence[int]:
        """Add tokens as aligned-block nodes with attached K/V slices.

        ``detached_blocks`` is indexed **[block_idx][layer_idx]** → (K, V).
        Precondition: ``len(detached_blocks) == len(tokens) // block_size``
        and every inner sequence has length ``num_layers``. The batcher
        builds this shape by iterating block-by-block over the row
        (§4.1) and slicing per-layer inside each block.

        Returns the block_ids actually newly allocated (empty for
        all-dup insert). For each new node: allocate a fresh id via the
        store, call ``store.retain_source``, call
        ``store.register_detached(block_id, detached_blocks[block_idx])``.
        Duplicate-prefix nodes reuse the existing block_id; the caller's
        corresponding entry in ``detached_blocks`` is discarded (GC'd
        when the caller's outer list goes out of scope)."""

    def release(self, block_ids: Sequence[int]) -> None:
        """Unchanged signature. Implementation switches kv.decref →
        store.release_hit. Does NOT touch detached storage."""

    def evict_until(self, n_blocks: int) -> int:
        """Unchanged signature. Implementation, per evicted node: call
        store.release_source(block_id) AND store.release_detached(block_id)
        atomically. Returns freed count."""
```

`insert(tokens, block_ids)` — the old signature took caller-provided block
ids (because PagedKVCache allocated them). In the synthetic path the
store allocates. We deprecate the old `insert` in favour of
`insert_detached`; the PagedPrefixBlockStore path (trigger-gated future
track) can add a separate `insert_paged(tokens, block_ids)` when it
arrives.

### 3.2 `PrefixBlockStore` Protocol

```python
class PrefixBlockStore(Protocol):
    block_size: int

    def allocate_id(self) -> int: ...
    def retain_source(self, block_id: int) -> None: ...
    def release_source(self, block_id: int) -> None: ...
    def retain_hit(self, block_id: int) -> None: ...
    def release_hit(self, block_id: int) -> None: ...
    def register_detached(
        self,
        block_id: int,
        per_layer_kv: Sequence[tuple[mx.array, mx.array]],
    ) -> None: ...
    def fetch_detached(
        self, block_id: int
    ) -> Sequence[tuple[mx.array, mx.array]]: ...  # per-layer
    def release_detached(self, block_id: int) -> None: ...
```

`SyntheticPrefixBlockStore` implements all eight. `PagedPrefixBlockStore`
raises `NotImplementedError` from `register_detached` / `fetch_detached`
(paged-attention kernel track will define its own detached-K/V model).

---

## 4. Admit algorithm with prefix hit

Called from `ContinuousBatcher._admit_waiting_requests()` for one pending
admission `(req_index, prompt_ids, params)`:

```text
# 1. Plan — side-effect-free; decide how many blocks to actually retain.
raw = radix.peek(prompt_ids)
max_aligned = ((len(prompt_ids) - 1) // block_size) * block_size
usable_hit_tokens = min(raw.num_hit_tokens, max_aligned)
# Ensure at least one suffix token survives prefill (§5 edge case 1).

if usable_hit_tokens == 0:
    # No-hit path: build a fresh BatchKVCache(B=1), prefill full prompt.
    # (identical to 16c.1's admission)
    ...
    return

# 2. Commit — retained lookup over the trimmed-to-usable prefix. This
#    single call owns ALL retain_hit / _live_hits / self.hits / LRU
#    touch bookkeeping; the batcher MUST NOT call store.retain_hit
#    directly (that would bypass RadixPrefixCache's invariants).
hit = radix.lookup(prompt_ids[:usable_hit_tokens])
assert hit.num_hit_tokens == usable_hit_tokens  # same tree, same walk
hit_block_ids = hit.block_ids

# 3. Materialise — build a BatchKVCache(B=1) seeded with detached K/V.
suffix_tokens = prompt_ids[usable_hit_tokens:]
row_cache = _build_seeded_batch_kv(
    detached_blocks=[store.fetch_detached(b) for b in hit_block_ids],
    left_padding=0,
)  # row_cache._idx == usable_hit_tokens; keys/values/offset pre-filled

# 4. Prefill only the suffix.
suffix_arr = mx.array([suffix_tokens], dtype=mx.int32)
logits = model(suffix_arr, cache=row_cache)
first_token = sampler.sample(logits[:, -1, :], ...)

# 5. Release hits — the K/V is now copied into row_cache.
radix.release(hit_block_ids)

# 6. Extend into main batch cache (if any live rows) or seed it.
main_cache.extend(row_cache)  # invariant I-2 preserves existing rows
```

**Subtlety — why `lookup(prompt_ids[:usable_hit_tokens])` not
`lookup(prompt_ids)` trimmed after the fact:** the walk must stop at
the usable boundary, not walk the full raw hit and then discard the
tail. A full-raw lookup would `retain_hit` blocks we then throw away,
and we would need to `release` them immediately — extra churn with no
benefit, and an easy place to drop a retain/release pair and leak
live-hit refs. Trimming the input instead lets RadixPrefixCache's
internal loop terminate at the right block naturally.

The `_build_seeded_batch_kv` helper is the one new low-level primitive. It
constructs a `BatchKVCache(left_padding=[0])` per layer, then sets each
layer's `.keys` / `.values` / `._idx` directly from the detached slices.
Gate-Q: confirm this direct-state-injection works the same as a sequence
of `update_and_fetch` calls — Gate-0.5 probed extend/filter but not
direct seeding. We will add a small probe before implementation (§6.1).

## 4.1 Insert on terminate — before filter

On request termination (inside `_reclaim_terminated`, before the `filter`
call that drops the row):

```text
# 1. Extract detached K/V per layer from the row BEFORE filter reshuffles.
row = self._rows[row_idx]
total_tokens = len(row.prompt_ids) + len(row.generated)
aligned_tokens = (total_tokens // block_size) * block_size
# Only aligned-block prefixes are retainable (partial trailing block dropped).

if aligned_tokens >= block_size:
    detached_per_block: list[list[tuple[K, V]]] = []
    for b_idx in range(aligned_tokens // block_size):
        start, end = b_idx * block_size, (b_idx + 1) * block_size
        per_layer = [
            (main_cache[l].extract_slice(row_idx, start, end))
            for l in range(n_layers)
        ]
        detached_per_block.append(per_layer)
    tokens_prefix = (row.prompt_ids + row.generated)[:aligned_tokens]
    radix.insert_detached(tokens_prefix, detached_per_block)

# 2. Now filter can safely reshuffle — detached K/V has been copied out.
main_cache.filter(kept)
```

`extract_slice(row_idx, start, end)` — small convenience wrapper over
mlx-lm's `BatchKVCache.extract` or direct `self.keys[row_idx:row_idx+1,
:, start:end, :]` slicing + `mx.eval` to force materialisation so the
eventual `filter` doesn't GC the source memory. Gate-Q #2: confirm that
slicing with `mx.eval` truly detaches from the batched tensor (Gate-0.5
assumption, worth a 10-line probe).

---

## 5. Edge cases (pin these now, they bite in code)

### Edge 1. Full-prefix hit → reserve one suffix token for first-token logits.

If `raw.num_hit_tokens == len(prompt_ids)`, the request has K/V for every
prompt token in cache. But `KVCache` stores K/V of *already consumed*
tokens — it does NOT store logits. Without a forward pass there is no
distribution to sample from.

Guard: `max_aligned = ((len(prompt_ids) - 1) // block_size) * block_size`
ensures at least 1 suffix token remains. If `len(prompt_ids) == 1` the
whole prompt prefills normally (no hit usable at block granularity
anyway). This is not a KVCache quirk; it is a property of every LLM that
keys + values + decoder-layer outputs are only produced by running
attention over tokens.

### Edge 2. Duplicate-prefix insert → don't leak detached K/V.

`insert_detached(tokens, detached)` may find that some of the tokens'
chunks are already in the tree (e.g. two requests finishing with the
same shared prefix). The existing node's block_id stays; the caller's
detached K/V for that chunk is **not registered** with the store. Python
GC collects the slice once `detached_per_block` goes out of scope in the
batcher's reclaim path. No explicit `release_detached` is needed in this
case — the slice never entered the store.

Test must assert: after two terminations of the same prefix, `store._detached`
has exactly one entry per block_id, not two.

### Edge 3. Release on same-cohort partial-hit copy.

After the admit algorithm copies hits into `row_cache`, it calls
`radix.release(hit_block_ids)`. This drops the hit count but must NOT
drop source refs (the tree still owns the block as a source). Eviction
can only happen when `store._hit_refs[b] == 0` AND the eviction routine
walks it.

Test must assert: after a lookup/release cycle on block `b`, source
refcount is unchanged (node is still in the tree) and `_detached[b]`
still present.

### Edge 4. Eviction during live hit — must refuse.

`evict_until(n)` walks nodes in LRU order skipping any with
`store._hit_refs[block_id] > 0`. This preserves L-2 ⊆ L-1. Test
must assert: a node with a live hit is skipped even if it is the oldest
leaf.

### Edge 5. Staggered admission shape for the acceptance test.

The PLAN §7 acceptance #2 ("prefix hit verifiable") cannot use "4 prompts
with shared prefix, all admit at step 0" — at step 0 the radix tree is
empty. Prefix reuse only happens if at least one request has **finished**
and its prefix has been inserted.

Test shape:

```text
block_size = 16
prefix_len = 128  # deliberately block-aligned; 8 whole blocks
suffix_len = 16   # unique tail per request, also aligned — keeps math clean
max_batch_size = 1
admit request 0 (prefix P + unique suffix 0)
  run to completion → insert_detached registers P as 8 blocks (128/16)
admit request 1 → peek returns 8 blocks → retain 7 blocks (leave 1
  block of prompt for suffix prefill, §5 edge 1) → prefill 16 + 16 = 32
  tokens instead of 128 + 16 = 144
... requests 2, 3 similarly
```

Why `prefix_len = 128` not `100`: with `block_size = 16`, a 100-token
prefix only aligns 96 tokens (`(100 // 16) * 16 == 96`); the trailing 4
tokens are not retainable. Using a block-aligned `prefix_len` avoids the
"why is my hit 96 not 100" confusion and makes the forward-token
assertion below write cleanly in closed form.

Assertion metrics, closed-form:

```python
aligned_prefix = (prefix_len // block_size) * block_size
usable_hit    = aligned_prefix - block_size  # §5 edge 1: reserve 1 block

# Tokens forwarded through the model during prefill over all 4 requests:
baseline = 4 * (prefix_len + suffix_len)              # no prefix cache
observed = (prefix_len + suffix_len) \
         + 3 * (prefix_len + suffix_len - usable_hit)  # 3 hits
assert radix.hits == 3
assert forward_tokens == observed
assert observed < baseline  # non-trivial reduction
```

- Per-row greedy output matches the hit-free baseline (same model, same
  prompts, same sampling) bit-exactly — hits must not change arithmetic.

---

## 6. Test plan

### 6.1 Pre-implementation probes (~20 lines each, one commit)

**Probe A — direct K/V seeding into BatchKVCache.**
Two sub-probes, both required before step 2:

- **A.1 — full state surface.** `BatchKVCache.update_and_fetch` advances
  both `_idx` and `offset`. Naively setting
  `cache.keys = K; cache.values = V; cache._idx = hit_tokens` leaves
  `offset` at its construction value (`[0]`), which some downstream
  paths read. Probe must exercise the **full state surface** — either
  via the `cache.state = (keys, values, offset, left_padding)` setter
  (preferred) or by explicit `cache.offset = mx.array([hit_tokens])`.
  Recommendation in advance of probe: the eventual 16c.2 primitive is a
  `_seed_batch_cache_state(cache, keys, values, hit_tokens,
  left_padding)` helper that exercises the state setter, not raw
  field assignment. Probe A.1 validates the helper against a reference
  `update_and_fetch`-driven cache produces bit-identical decode output.
- **A.2 — seeded `extend`.** `main_cache.extend(row_cache)` where
  `row_cache._idx > 0` (i.e. seeded prefix). This is Open-Q Q1 turned
  into a test: if extend requires incoming `_idx == 0`, option B's
  compute saving collapses and we must pivot (§8 step 1 note).

**Probe B — extract-slice detachment.**
Slice a chunk out of `BatchKVCache.keys`, call `mx.eval`, then filter the
source cache dropping that row. Confirm the slice still reads correct
values afterward. If mlx lazy-materialises and GCs the source, we need
`mx.array(copy=True)` or similar.

### 6.2 Unit tests (`tests/test_prefix_store.py`, `tests/test_radix_prefix_cache.py` extensions)

1. `SyntheticPrefixBlockStore` — retain/release split, allocate_id monotonic,
   register/fetch/release_detached lifecycle.
2. `peek` is side-effect-free: pre/post snapshots of `store._hit_refs`,
   `cache._live_hits`, `cache.hits` are equal.
3. `insert_detached` on duplicate prefix: store has 1 detached entry per
   block, not 2. `_source_refs[b] == 1` (not 2).
4. `release(block_ids)` after `lookup`: `_hit_refs[b] == 0` and
   `_source_refs[b]`, `_detached[b]` unchanged.
5. `evict_until(n)` on live-hit block: skipped; returns freed < n.
6. `evict_until(n)` on dead leaf: drops tree node AND deletes
   `_detached[block_id]` atomically.
7. Full-hit reservation — two branches, both load-bearing:
   - **Aligned full-hit:** `len(prompt_ids) == 5 * block_size` with
     raw hit over all 5 blocks → `max_aligned = ((5B-1)//B)*B = 4B` →
     `usable_hit_tokens == 4 * block_size`. One block of prompt is
     reserved for suffix prefill to produce first-token logits.
   - **Unaligned sibling:** `len(prompt_ids) == 5 * block_size + 3`
     with raw hit over the aligned 5 blocks → `max_aligned =
     ((5B+2)//B)*B = 5B` → `usable_hit_tokens == 5 * block_size`.
     Here all 5 aligned blocks retain; the 3 trailing unaligned tokens
     cover the "need ≥ 1 suffix token" requirement on their own, so
     no extra block is reserved.

   Assert both. The `-1` in `max_aligned`'s formula is the load-bearing
   piece — drop it and the aligned-full-hit case will try to skip
   prefill entirely, leaving no way to sample the first decode token.

### 6.3 Batcher integration tests (`tests/test_batcher.py` extensions)

8. Admit path routes through radix when hit available; routes through
   full prefill when no hit (same observable behaviour, different
   internal forward-token count).
9. Reclaim inserts detached K/V before filter (assert `store._detached`
   has entries after reclaim, even though row is gone from main cache).
10. Two admits sharing prefix — after first finishes + reclaims, second
    sees a hit in peek.

### 6.4 Real-model acceptance (`tests/test_p2_batched_parity.py`
    extension, Qwen3-0.6B)

11. `test_shared_prefix_reduces_forward_tokens` — 4 prompts with a
    **block-aligned** shared prefix (e.g. 128 tokens at block_size=16) +
    unique suffix, staggered admission (max_batch_size=1). Use the
    closed-form assertion from §5 edge 5 for `forward_tokens` — not a
    percentage — so the test fails informatively if either the hit
    boundary or the suffix reservation drifts. `radix.hits == 3` exactly;
    per-row output matches a no-prefix-cache baseline bit-exactly.

---

## 7. Non-goals for 16c.2

- **Paged-attention kernel path** (`PagedPrefixBlockStore` detached K/V) —
  out of scope; P-2 Opening trigger-gated future track.
- **Cross-layer block sharing / selective layer prefix** — every block id
  names the same logical block across every layer, as today.
- **Prompt-side tokenization variance robustness** — two prompts whose
  first 100 tokens are "textually the same" but tokenize differently (BPE
  merges, whitespace) will not share a prefix. That is correct behaviour
  at the token layer; upper layers (ShareGPT harness, API shim) own the
  "normalise prompt for prefix reuse" decision.
- **Prefix hit during initial cohort at step 0** — when the tree is empty
  (process boot, first batch), no request can hit. 16c.2's hits only
  accumulate via `insert_detached` on reclaim.
- **Heterogeneous SamplingParams across sharing requests** — `P-3`
  concern. For 16c.2 all sharing requests still share one params object,
  since `generate_batch` itself enforces homogeneity.

---

## 8. Implementation split

16c.2 will land as five atomic commits with pauses for approval
(`feedback_commit_approval.md` + `feedback_incremental_plan_execution.md`):

1. **Probes A + B** (`probe(gate-0.75): direct K/V seeding + extract-slice lifetime`).
   Confirms or rules out the two physical assumptions above.
2. **`PrefixBlockStore` + `SyntheticPrefixBlockStore`** (`feat(kvcache): PrefixBlockStore protocol (Unit 16c.2 step 1)`).
   New file `silica/kvcache/store.py`; no RadixPrefixCache changes yet.
3. **RadixPrefixCache refactor to Protocol + `peek` + `insert_detached`**
   (`refactor(kvcache): RadixPrefixCache over PrefixBlockStore (Unit 16c.2 step 2)`).
   Existing `insert(tokens, block_ids)` deprecated in favour of
   `insert_detached`; `lookup` unchanged signature, internals swap to
   store. Existing tests updated.
4. **Batcher wires prefix admission** (`feat(scheduler): prefix-reuse admit path (Unit 16c.2 step 3)`).
   `_build_seeded_batch_kv` helper + `_admit_waiting_requests` peek/lookup/release
   logic + reclaim-time `insert_detached`. Instrumentation: forward-token
   counter exposed via `batcher.metrics` or similar.
5. **Real-model acceptance + PLAN §7 #2 green** (`test(p2): prefix hit reduces forward tokens (Unit 16c.2 acceptance)`).
   Test #11 above plus any regression catches from real-model runs.

Estimated: 2 days including probes and the real-model run. If Probe A
fails we add a one-step fake-prefill path (suffix_tokens = hit_tail + real_suffix,
K/V seeded for `hit_tail`'s block); pivot documented in a committed
decision note before step 3.

---

## 9. Open questions (resolve during step 1)

Q1. Does `BatchKVCache.extend` accept a `row_cache` whose `_idx > 0`
already, or does it require the incoming cache's rows to start at
`_idx == 0` with left_padding adjusted? (Gate-0.5 Q3 confirmed extend
shift-left works for same `_idx`; the "seeded non-zero" case was not
probed.) If seeded-extend is unsupported, the seed happens via a
per-row prefill-replay: `model(prefix_tokens, cache=empty_row_cache)`,
discard logits, then continue with suffix. That negates the compute
savings — makes prefix reuse purely a source-of-truth-for-dedupe
convenience. Unacceptable; we must confirm seeded-extend works.

Q2. Per-layer K/V shape: does mlx-lm expose K/V as `(B, H, T, D)` in
`BatchKVCache.keys`? Gate-0.5 noted the shape but did not document it
here. Needed to write `extract_slice` correctly. Read
`mlx_lm.models.cache.BatchKVCache` at step 1.

Q3. `self.hits` counter: keep as-is (increment only on retained lookup),
or expose a second counter `self.peeks`? For the acceptance test we
only need lookup hits; `peek` is meant to be invisible. Keep one
counter.

---

## 10. Success criteria

16c.2 is complete when:

- `RadixPrefixCache` depends on `PrefixBlockStore`, not `PagedKVCache`.
- `peek()` is side-effect-free (test 2 green).
- A shared-prefix run on Qwen3-0.6B shows `hits >= 3` and a measurable
  forward-token reduction (test 11 green).
- Per-row outputs with prefix cache match outputs without prefix cache
  bit-exactly (no drift from reuse).
- All existing tests (16a/16b/16c.1) still green.
- PLAN §7 P-2 acceptance #2 marked green.

PLAN §7 acceptance #3 (budget overflow clean shutdown) is 16d's concern,
not 16c.2's.
