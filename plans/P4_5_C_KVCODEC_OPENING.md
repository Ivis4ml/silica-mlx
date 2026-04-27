# P-4.5-C — KVCodec runtime integration spike: opening

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Version      | v1.0.0                                                             |
| Last updated | 2026-04-21                                                         |
| Status       | opening doc — B.1 complete, implementation (C.1) not yet started   |
| Maintainer   | Xin Zhou                                                           |
| Scope        | decide the single v0.1 integration point for `KVCodec` on the      |
|              | runtime forward path; pin its granularity, resident-bytes          |
|              | accounting, and acceptance tests; amend stale PLAN §7 P-5 text     |

## 0. TL;DR

At the P-4 exit it surfaced that `IdentityCodec.encode_block` and
`decode_block` have **zero runtime callers** — they are traversed only by
`tests/test_kvcodec.py`. The P-2 forward path holds live K/V in
`BatchKVCache`, `BatchRotatingKVCache`, and mlx-lm's per-layer
`ArraysCache`, constructed by `silica/scheduler/batcher.py::_make_batch_cache`;
the prefix-cache path stores per-block K/V in
`silica/kvcache/store.py::SyntheticPrefixBlockStore._detached`. Writing a
P-5 `BlockTQCodec` against I-3 without first pinning a hook would ship
an interface-level codec whose `resident_bytes` reduction did not
appear in actual unified-memory usage.

P-4.5-C chooses the integration point and wires `IdentityCodec` through
it end-to-end, **without** shipping any real compression. The three
candidates are:

- **(A)** Encode active `BatchKVCache` in-place on the live forward
  path — extends I-3 semantics into every decode step.
- **(B)** Encode only detached prefix-cache blocks at admission; leave
  active `BatchKVCache` untouched. This is the seam
  `SyntheticPrefixBlockStore.register_detached` / `fetch_detached` /
  `release_detached` already exposes.
- **(C)** Wrap mlx-lm's `BatchKVCache` in a codec-aware façade and hand
  the façade to forward. Changes the cache shape mlx-lm's attention
  kernels see.

**Recommendation: Option (B).** (A) and (C) both require a
compressed-domain attention kernel which D-003 explicitly excludes from
v0.1, and MLX lacks a variable-length attention primitive (Q-009 / R-7)
to absorb the decoded-on-demand shape. Option (B) turns
`IdentityCodec.encode_block` into a runtime caller via the
`register_detached` / `fetch_detached` pair already on the admission
path, keeping every active-K/V shape identical to what attention
consumes today.

P-4.5-C ships:

1. `plans/P4_5_C_KVCODEC_OPENING.md` (this doc) and a PLAN §7 P-5
   Strategy amendment replacing the stale
   `PagedKVCache(codec=...) injection-based switching` wording with
   `PrefixBlockStore(codec=...)`.
2. `SyntheticPrefixBlockStore(codec: KVCodec = IdentityCodec(...))`
   constructor wiring; `register_detached` calls `codec.encode_block`
   per layer; `fetch_detached` calls `codec.decode_block`; both
   preserve the existing external shape contract.
3. `SyntheticPrefixBlockStore.resident_bytes()` — sums the stored
   `CodedBlock.resident_bytes`. A **parallel observable**, not yet a
   replacement for the budgeter's per-block formula.
4. `tests/test_kvcodec_integration.py` — the acceptance entry point is
   a **single** `Engine.generate_batch([prompt, prompt], params,
   prefix_cache=shared_pc, max_batch_size=1)` call, not two paired
   calls. Row 0 admits into the initial cohort (miss-path prefill) and
   registers its aligned prefix during reclaim; row 1 enters the
   waiting queue and, on the next step, `_admit_waiting_requests`
   routes it through the hit path (`_admit_single_hit_row` →
   `fetch_detached_blocks` → `codec.decode_block`). Encode and decode
   counters both fire inside this one call. Two consecutive
   `generate_batch([prompt], ...)` calls with a shared `prefix_cache`
   would **not** fire `decode_block` because `_prepare_cohort` (initial
   cohort seal) does not consult the prefix cache — see §8.0 for the
   design fact and PLAN §10 Q-012 for the follow-up. The prompt must
   tokenize to ≥ `2 × block_size + 1` tokens (and not an exact
   multiple of `block_size`) so cold encode and mid-run-hit decode
   counts match under the batcher's S-5 edge 1 rule — see §6.1 / §8.1.
   Baseline-identity: the codec path's token stream matches the
   no-codec `codec=None` pass-through on the same `[p, p]` workload.
   Note: `Engine.generate(prompt, params)` is **not** the verification
   entry point — it bypasses `ContinuousBatcher` / `RadixPrefixCache`
   entirely and drives `SimpleKVCache` via `_drive`, so the
   detached-K/V hook is not reached on that path.

P-4.5-C does **not** ship `BlockTQCodec`, does not touch
`MemoryBudgeter`, does not introduce a compressed-domain attention
kernel, and does not cover heterogeneous per-layer K/V shapes (Gemma4
sliding 16×256 + full 4×512 mix). Each of those belongs to P-5 proper.

---

## 1. Problem

The P-5 codec hot-path gap, restated from §7 P-4 empirical findings
(2026-04-21):

> `grep encode_block|decode_block silica/` matches `silica/kvcache/codec.py`
> only; zero runtime callers. The real forward path builds `BatchKVCache`
> / `BatchRotatingKVCache` / `ArraysCache` via `_make_batch_cache` in
> `silica/scheduler/batcher.py`.

Three consequences if we shipped a `BlockTQCodec` directly against I-3
today:

1. **`resident_bytes` reduction would be invisible.** The scheduler
   reads `kv_manager.budget().resident_bytes`, computed in
   `PagedKVCache.budget()` as `n_claimed * self._bytes_per_block` — the
   P-2 page-table accounting, independent of any codec. The codec could
   report 4× savings and the resident figure would not move. This
   violates Principle 8 (savings must be observable by the scheduler).
2. **The admission decision would not benefit.** `MemoryBudgeter.admit()`
   uses `bytes_per_token` and `_kv_bytes_per_block` formulas derived
   from the adapter's `kv_layout()`; the codec does not appear in the
   decision path. Admission control would still gate on fp16 upper
   bounds.
3. **`IdentityCodec` would remain a test-only object.** The
   `@runtime_checkable` protocol and the P-0 interface test pin the
   shape, but no forward pass traverses them — the P-5 implementation
   would land without a backing integration test.

P-4.5-C answers: where in the forward path does a codec instance
actually sit? And what is the smallest change that makes that claim
testable, without shipping a real compression implementation?

---

## 2. Constraints

From the PLAN, recent advisor notes, and the runtime survey this doc is
built on:

### 2.1 D-003 — no compressed-domain attention fast path in v0.1

`KVCodec` has `encode_block` / `decode_block` only; **no `attend()`**.
Attention always sees fp16 K/V. Any integration that puts compressed
K/V into the tensor mlx-lm's attention kernel consumes is therefore
out of scope for v0.1.

### 2.2 Q-009 / R-7 — MLX has no variable-length attention kernel

SDPA on batched K/V requires rectangular `(B, H, T, D)` with a shared
`T`. Decode-on-demand designs that hand the attention kernel a
time-varying per-request sequence length are currently impossible
without writing a custom kernel — an explicit non-goal of P-5.

### 2.3 D-009 — MLX-native hot path

No `torch.Tensor` in the live forward path. `silica.vq.block_tq` at P-5
time will be rewritten on `mx.array`; vqbench's NumPy reference is
algorithmic source, not a runtime dependency. IdentityCodec already
satisfies this.

### 2.4 P-2 Option B — active K/V lives in `BatchKVCache`

Under P-2 Option B (the only option that actually runs today),
`PagedKVCache` is a page-table + refcount bookkeeping layer. **It does
not store K/V.** Live K/V during prefill and decode live in mlx-lm
`BatchKVCache` instances built at cohort admission; prefix-hit K/V
lives in `SyntheticPrefixBlockStore._detached`. Any codec hook on
"PagedKVCache" has nothing to encode.

### 2.5 `build_seeded_batch_kv` requires shape-preserving decode

`silica/scheduler/seed_kv.py::build_seeded_batch_kv` does
`mx.concatenate(k_parts, axis=2)` across the per-block K/V slices and
then `cache.state = (keys, values, mx.array([T]), mx.array([0]))`. The
downstream attention kernel reads this unchanged. Therefore any codec
whose `decode_block` is called inside `fetch_detached` must return
`(K, V)` in fp16 with the original `(1, n_kv_heads, block_size,
head_dim)` shape. This is D-003 restated, not a new constraint — but
it pins the exact contract the spike must preserve.

### 2.6 P-4.5-C is a spike, not an optimization

Per PLAN §7 P-4.5 Notes: "P-4.5-C is a spike whose output is 'we
verified the integration point works', not 'we optimized it'." The real
BlockTQCodec constructor and resident-bytes-driven admission lift are
P-5 work. The spike ships IdentityCodec threading, not compression.

---

## 3. Three integration-point options

The three options differ on **which K/V tensors the codec touches**,
not on the codec algorithm. All three install `IdentityCodec` as the
concrete instance during the spike.

### 3.1 Option (A) — encode active `BatchKVCache` blocks in place

Hook point: extend the cohort-admission code in
`silica/scheduler/batcher.py` to encode each layer's K/V in
`block_size`-aligned slices as they grow past the boundary; store the
encoded payload in a parallel structure; `decode_block` fires before
every attention call to materialize fp16 K/V for the current forward.

- **Codec traversal:** every decode step fires `num_blocks_active ×
  num_layers × 2 (encode + decode)` codec calls.
- **Interaction with mlx-lm attention:** attention still receives fp16
  tensors, but the step between "BatchKVCache holds encoded payload"
  and "attention kernel reads decoded payload" requires either (i) a
  scratch buffer that decodes every block on every step (thrashes
  unified memory) or (ii) a cached decode pool keyed on block id with
  its own eviction policy.
- **Scheduler footprint:** large. `BatchKVCache.update_and_fetch` is
  mlx-lm code — we cannot change it. The only place to intercept K/V
  on the active path is by replacing the `BatchKVCache` instance with
  a façade, which collapses into option (C).
- **D-003 compatibility:** requires a decode scratch buffer before
  every attention call; codec call count scales with decode tokens
  emitted, not with admissions. On Qwen3-0.6B 28-layer × 16-block-size
  this is ~1–2 decodes per step per layer. The overhead is not
  unbounded, but the bookkeeping required to keep per-request scratch
  straight across cohort changes is the bulk of the scheduler
  refactor.
- **Non-starter because:** you cannot intercept K/V inside
  `BatchKVCache.update_and_fetch` without replacing the cache — that is
  option (C). Option (A) alone is not distinguishable from (C) in
  engineering reality.

### 3.2 Option (B) — encode only detached prefix-cache blocks

Hook point: `silica/kvcache/store.py::SyntheticPrefixBlockStore`.
`register_detached(block_id, per_layer_kv)` is called on every newly
inserted prefix-cache block (`insert_detached` in
`silica/kvcache/prefix.py::RadixPrefixCache`). The store currently
stores the raw `(K, V)` tuples; swap them for `CodedBlock` objects
produced by `codec.encode_block`. `fetch_detached` calls
`codec.decode_block` to return shape-preserving fp16.

- **Codec traversal:** `register_detached` fires at admission for the
  requesting row's newly-claimed prefix blocks only. `fetch_detached`
  fires at admission of a *later* request whose prompt shares that
  prefix. Encode/decode counts are bounded by prefix-cache churn, not
  by decode-step count. Under a 57-token shared prefix with
  `block_size=16` (three aligned blocks), the first request's admission
  fires `register_detached` three times × 28 layers on Qwen3-0.6B, a
  second request with the same prefix fires `fetch_detached` three
  times × 28 layers; decode does not touch the codec again.
- **Interaction with mlx-lm attention:** zero. `build_seeded_batch_kv`
  concatenates the decoded tuples into a fresh `BatchKVCache(B=1)`
  seeded via `cache.state`; from that point forward the execution is
  exactly today's code. D-003 is not touched.
- **Scheduler footprint:** minimal. Constructor-only wiring on
  `SyntheticPrefixBlockStore`; `register_detached` and `fetch_detached`
  bodies gain one `codec.encode_block` / `codec.decode_block` call per
  layer; all call sites (`insert_detached`, `fetch_detached_blocks`)
  remain untouched. `PagedPrefixBlockStore` keeps its existing
  `NotImplementedError` because paged-attention still lacks the kernel
  (Q-009 / R-7).
- **Resident bytes:** `SyntheticPrefixBlockStore.resident_bytes()` sums
  `CodedBlock.resident_bytes` across the `_detached` dict. Under
  `IdentityCodec` this equals `num_blocks × num_layers × block_size ×
  (2 × n_kv_heads × head_dim × dtype.size)`, where `num_blocks` is the
  total count of store-resident blocks — reachable either as
  `len(store.live_block_ids())` or as `prefix_cache.node_count()`
  (1:1 correspondence via `insert_detached`). This is **not** the
  same quantity as `_count_evictable_prefix_blocks ×
  _kv_bytes_per_block` — that budgeter helper counts leaf-zero-hit
  blocks only and skips internal prefix nodes, so for any multi-block
  prefix chain it under-reports. See §6.2 for the full discussion and
  why this is a **parallel observable** in P-4.5-C, not yet a
  replacement for the budgeter's eviction-shortfall view.
- **Coverage gap:** the active (decode-time growing) `BatchKVCache` is
  not encoded. A request whose prompt does not hit the prefix cache
  traverses zero codec calls for its decode tokens. This is intentional
  for v0.1 — prefix-cache residency is where multi-request sharing
  makes savings most valuable, and it is the one place where encoded
  K/V survives the owning request's lifetime (active caches are tied
  to the request and released on termination).

### 3.3 Option (C) — codec-aware `BatchKVCache` façade

Hook point: new class `CodedBatchKVCache` that wraps an mlx-lm
`BatchKVCache` and intercepts `update_and_fetch`, encoding the
`update`-side K/V into its own internal block storage and decoding
back on fetch so the mlx-lm attention call sees fp16.

- **Codec traversal:** every `update_and_fetch` on every decode step —
  same scaling as option (A).
- **Interaction with mlx-lm attention:** requires that the façade
  implement every method the mlx-lm forward pass calls on its cache
  argument. `BatchKVCache` inherits `_BaseCache`-style behaviour with
  `step=256` growth chunks and specific fp16 dtype expectations baked
  into the SDPA call. Any deviation breaks numerical parity.
- **Scheduler footprint:** large and risky. `_make_batch_cache` in
  `silica/scheduler/batcher.py` would return the façade instead of
  `BatchKVCache`; every subsequent `extend` / `state` access across
  `silica/scheduler/seed_kv.py`, `silica/scheduler/batcher.py` /
  `silica/mlx/runner.py` would need to work against the façade's
  surface.
- **Parity risk:** P-3-D3.1 Gemma4 and P-2 Qwen3-0.6B batched parity
  findings already document fp16 SDPA drift under batch-composition
  changes. Swapping in a façade whose decoded outputs may differ from
  mlx-lm's own K/V by a single ULP compounds over hundreds of decode
  steps — not a bug the spike wants to own before P-5.

---

## 4. Trade-off matrix

| Criterion                                              | (A) active in-place | (B) detached prefix store | (C) cache façade |
| ------------------------------------------------------ | ------------------- | ------------------------- | ---------------- |
| D-003 respected (no compressed-domain attention)       | no *                | yes                       | yes              |
| Works under MLX's no-var-len-attn kernel (Q-009 / R-7) | no *                | yes                       | no               |
| Codec traversal on cold single-request                 | yes (every step)    | encode-side only          | yes (every step) |
| Codec traversal on repeat-prompt request               | yes                 | decode-side only          | yes              |
| `resident_bytes` becomes observable                    | yes                 | yes                       | yes              |
| Scheduler diff size                                    | large               | ~20 lines                 | large            |
| Parity risk vs no-codec baseline                       | high                | zero (identity)           | high             |
| Preserves existing `build_seeded_batch_kv` contract    | yes                 | yes                       | no               |
| Extends to P-5 `BlockTQCodec` with the same hook       | yes                 | yes                       | yes              |

\* Options (A) and (C) require a compressed-domain attention kernel or
a per-step decode scratch; the former is D-003-excluded, the latter
runs into MLX's no-variable-length-attention kernel (Q-009 / R-7). The
"no" entries reflect the v0.1 constraint set, not a permanent
impossibility.

---

## 5. Recommendation — Option (B)

**Option (B), prefix-store-scoped codec hook, is chosen.**

Rationale, in decreasing weight:

1. **D-003 and Q-009 / R-7 rule out (A) and (C) for v0.1.** Without a
   compressed-domain attention kernel or variable-length SDPA, any hook
   on the active path needs a full-shape fp16 scratch before every
   attention call — engineering cost far outside the P-4.5-C spike
   budget, and solving it is on the critical path for P-5 proper
   separately.
2. **The seam already exists.** `SyntheticPrefixBlockStore` has three
   detached-K/V methods (`register_detached` / `fetch_detached` /
   `release_detached`) with an explicit lifetime rule and loud-fail
   guards (L-3 ⊆ L-1 invariant in `silica/kvcache/store.py`). The spike
   plugs into an interface that was designed to be pluggable; it does
   not require inventing a new one.
3. **Numerical parity is trivially guaranteed.** `IdentityCodec`
   decode_block returns `(block.k, block.v)` with no re-quantization
   step, so the concatenated fp16 K/V fed into `build_seeded_batch_kv`
   is byte-identical to what today's code produces. The spike's
   baseline-identity acceptance clause (§8.4 below) falls out of the
   contract — no fp16 drift to reconcile.
4. **Prefix-cache residency is the right scope for the first
   observable saving.** Prefix-cache K/V outlives the owning request
   (retained until `evict_until` drops it); active K/V does not. Under
   a future BlockTQCodec, a saving on prefix-cache residency is a
   saving on persistent memory — admission headroom grows. A saving on
   active K/V would only help intra-cohort, which is the case
   `MemoryBudgeter`'s per-request `reserved_bytes` upper bound already
   covers worst-case.
5. **Extends to P-5 `BlockTQCodec` without re-architecture.** The same
   `register_detached` → `encode_block` path lights up BlockTQ; the
   only P-5-proper change is swapping `IdentityCodec` for the real
   constructor and teaching `MemoryBudgeter` to read
   `store.resident_bytes()` in place of the per-block formula. Both
   are bounded changes.

Option (B) does not solve active-K/V compression. That is the right
call for v0.1 — active-K/V compression needs either a compressed-domain
attention kernel (D-003) or variable-length SDPA (Q-009 / R-7), and
neither is in P-5 scope.

---

## 6. Touchpoints

### 6.1 Encode / decode granularity

The `PrefixBlockStore.register_detached` signature is:

```python
def register_detached(
    self,
    block_id: int,
    per_layer_kv: Sequence[tuple[mx.array, mx.array]],
) -> None
```

Each `(K, V)` pair has shape `(1, n_kv_heads, block_size, head_dim)`.
Under Option (B), the store calls `codec.encode_block(k, v)` once per
layer, producing one `CodedBlock` per layer, stored as a
`tuple[CodedBlock, ...]` keyed by `block_id`:

```python
self._detached: dict[int, tuple[CodedBlock, ...]] = {}
```

(Replacing today's `dict[int, tuple[tuple[mx.array, mx.array], ...]]`.)

**Codec call count per newly-inserted prefix-cache block =
`num_layers`.** On Qwen3-0.6B (28 transformer layers) a single-block
insert produces 28 `encode_block` calls. A `fetch_detached(block_id)`
call produces 28 `decode_block` calls, returning 28 shape-preserving
fp16 `(K, V)` tuples that match the existing return shape contract
consumed by `RadixPrefixCache.fetch_detached_blocks` →
`build_seeded_batch_kv`.

**The spike uses one shared `IdentityCodec` instance across all
layers.** Qwen3-0.6B is homogeneous — every layer has the same
`n_kv_heads` and `head_dim`. Gemma4's heterogeneous 16×256 sliding +
4×512 full mix would require either per-layer codec instances or a
shape-adaptive codec constructor; P-4.5-C scopes the spike to
homogeneous-shape models (Qwen3-0.6B is the verification target) and
defers heterogeneous-shape handling to P-5 proper where BlockTQ's own
per-layer calibration raises the same per-layer-instance question.

### 6.2 `resident_bytes` — parallel observable, not replacement

`SyntheticPrefixBlockStore.resident_bytes()` is added as:

```python
def resident_bytes(self) -> int:
    return sum(
        sum(cb.resident_bytes for cb in coded_tuple)
        for coded_tuple in self._detached.values()
    )
```

Under `IdentityCodec` this equals `num_blocks × num_layers × block_size
× bytes_per_token_per_layer` where `bytes_per_token_per_layer = 2 ×
n_kv_heads × head_dim × dtype.size` (one K plus one V at the layer's
dtype — each `CodedBlock.resident_bytes = k.nbytes + v.nbytes` for
fp16). `num_blocks` is the total count of store-resident blocks. The
same number is derivable from the radix-tree snapshot:
`RadixPrefixCache.node_count() × num_layers × block_size ×
bytes_per_token_per_layer`, because every radix node corresponds 1:1
with a `register_detached` call (see
`silica/kvcache/prefix.py::insert_detached`). This is the parity
right-hand side §8.3 uses.

Beware the name collision with
`MemoryBudgeter.bytes_per_token`: the budgeter attribute is
*all-layer total* (`2 × num_layers × n_kv_heads × head_dim ×
dtype.size`, or the adapter's precomputed
`layout.bytes_per_token_total`), so multiplying it by `num_layers`
would double-count. The P-4.5-C parity formula uses
`bytes_per_token_per_layer` — the single-layer K+V cost — which when
multiplied by `num_layers` recovers the correct total. Written out
without the shorthand: `total_resident = num_blocks × num_layers ×
block_size × 2 × n_kv_heads × head_dim × dtype.size`.

**Do not confuse this with
`MemoryBudgeter._count_evictable_prefix_blocks`.** That helper counts
*leaf* nodes with *zero live hits* — the eviction-candidate set — and
deliberately skips every internal radix node. For a prompt that
tokenizes to ≥ 2 × `block_size`, the tree is a chain (root → node1 →
node2); only node2 is a leaf, so the budgeter would report 1 block
where the store holds 2. The budgeter's value is correct for its
purpose (how many blocks can we free?) but is **not** the total
resident prefix bytes, and must not be used as the parity right-hand
side for §8.3.

This is explicitly a **parallel observable**, not a replacement. The
budgeter's `_count_evictable_prefix_blocks × _kv_bytes_per_block`
formula continues to be the authority for its own eviction-shortfall
decisions in P-4.5-C; `store.resident_bytes()` is added so that:

1. When P-5 proper introduces a non-identity codec whose
   `resident_bytes` diverges from the fp16 per-block formula, the new
   authoritative source already exists at the right semantic layer
   (the prefix store owns the coded payload; the budgeter does not).
2. The spike's numerical-identity acceptance clause (§8.3) can assert
   `store.resident_bytes()` agrees with the radix-tree-derived total
   resident bytes under IdentityCodec, catching any integration
   mistake that causes double-counting or under-counting at the
   observable level.

The budgeter → `store.resident_bytes()` transition is P-5 proper work,
out of P-4.5-C scope. The P-4.5-C commit adds the observable and
pins its value under identity; it does not mutate
`MemoryBudgeter.admit()` or `_count_evictable_prefix_blocks`.

### 6.3 PLAN §7 P-5 Strategy line 539 amendment

PLAN §7 P-5 Strategy currently reads (line 539):

> `PagedKVCache(codec=...)` injection-based switching.

Under P-2 Option B, `PagedKVCache` is a page-table + refcount
bookkeeping layer that holds no K/V. The actual codec hook is
`PrefixBlockStore` (synthetic variant holds the detached K/V; paged
variant raises `NotImplementedError` on detached methods because
paged-attention requires a compressed-domain kernel or MLX variable-
length SDPA, neither of which is in v0.1). The P-4.5-C landing commit
updates PLAN §7 P-5 Strategy in the same commit as this opening doc to:

> `PrefixBlockStore(codec=...)` injection-based switching; the active-
> K/V path is **not** codec-wrapped in v0.1 because D-003 excludes
> compressed-domain attention and MLX has no variable-length SDPA
> (Q-009 / R-7). See `plans/P4_5_C_KVCODEC_OPENING.md`.

Doing the PLAN and opening-doc edits in the same commit keeps the two
documents consistent at commit time — the alternative (amend PLAN in
the later C.1 commit) would leave one commit where PLAN contradicts
the opening doc.

### 6.4 Scope is homogeneous-shape models only

`Qwen/Qwen3-0.6B` is the single target for P-4.5-C verification. Gemma4
(heterogeneous sliding + full mix, 32 layers × two different
`(n_kv_heads, head_dim)` pairs) and Qwen3.5-MoE checkpoints are out of
spike scope. The reason is not that Option (B) is incompatible — the
store does not need to know about per-layer shape variation — but that
`IdentityCodec`'s constructor takes a single `(n_kv_heads, head_dim)`
pair. A heterogeneous model would need either per-layer `IdentityCodec`
instances (store gains a `list[KVCodec]` of length `num_layers`) or a
shape-adaptive codec constructor. Both are shape-parametrization
decisions that P-5 proper must take anyway when it introduces
per-layer BlockTQ calibration parameters. Raising the question inside
the spike would bundle two design decisions into one commit; raising
it in P-5 opening keeps them separate.

---

## 7. What this spike does NOT do

Explicit non-goals so the commit scope is bounded:

- **No `BlockTQCodec` or `RaBitQCodec`.** P-5 proper ships those. The
  spike uses `IdentityCodec` as the identity element of the codec
  space; its presence or absence changes zero bytes.
- **No change to `MemoryBudgeter.admit()` or
  `_count_evictable_prefix_blocks`.** Admission decisions continue to
  use the per-block formula. `store.resident_bytes()` is added as a
  parallel observable only; its value is asserted equal to
  `prefix_cache.node_count() × num_layers × block_size × (2 ×
  n_kv_heads × head_dim × dtype.size)` under `IdentityCodec` — i.e.
  the per-layer K+V byte-per-token cost, not the budgeter's
  all-layer-summed `bytes_per_token` attribute (not to
  `_count_evictable_prefix_blocks × _kv_bytes_per_block` either, which
  counts only leaf-zero-hit blocks and would report a smaller number).
  The store becomes the future authority for P-5 proper, not this
  spike.
- **No compressed-domain attention kernel.** D-003 excludes it; Option
  (B) does not require it.
- **No active-`BatchKVCache` codec wrapping.** Options (A) / (C) are
  explicitly excluded; active K/V continues to live unwrapped.
- **No heterogeneous per-layer-shape codec.** Qwen3-0.6B homogeneous
  only; Gemma4 and Qwen3.5-MoE remain on the no-codec path through
  P-4.5-C.
- **No `PagedPrefixBlockStore` codec integration.** The paged backend
  keeps its `NotImplementedError` stubs on the detached methods;
  paged-attention's codec story lands when the paged-attention
  kernel track itself advances (currently trigger-gated, per
  `plans/P2_OPENING.md`).
- **No bench catalog row.** The verification test lives in
  `tests/test_kvcodec_integration.py`, cache-presence gated on the
  local Qwen3-0.6B HF cache (no env-var strong gate — mirrors
  `tests/test_engine_admission_reorder.py` §5 since the 0.6B forward
  is cheap enough not to warrant one). Not in
  `silica/bench/scenarios.py`. A codec-on bench row lands with P-5
  proper when there is an actual memory or quality delta to measure.

---

## 8. Live-forward acceptance specification

Four assertions, each tied to a specific call site or observable. All
on-device assertions run on `Qwen/Qwen3-0.6B` (cached); cache-presence
gates follow the same pattern
`tests/test_engine_admission_reorder.py` §5 uses.

### 8.0 Entry point and workload shape

**`Engine.generate(prompt, params)`** drives `self._kv_manager` (= a
P-1 `SimpleKVCache`) via `self._drive` — a pure single-request path
that calls `adapter.prefill` / `adapter.decode_step` without routing
through `ContinuousBatcher`. It takes no `prefix_cache` argument and
never touches the detached-K/V hook. Do not use it for verification.

**`Engine.generate_batch(prompts, params, *, prefix_cache=...,
max_batch_size=...)`** drives `ContinuousBatcher` and is the entry
point that exercises `RadixPrefixCache.insert_detached` /
`fetch_detached_blocks` → `SyntheticPrefixBlockStore.register_detached`
/ `fetch_detached`. But not every `generate_batch` shape exercises the
hit branch — and the choice of shape is load-bearing for acceptance:

- **Initial cohort (`_prepare_cohort`) does not consult the prefix
  cache.** Every admission that goes through `batcher.add_request`
  before the first `step()` is sealed into the initial cohort and runs
  a miss-path batched prefill unconditionally. Prefix-cache lookup
  (`peek` → `_admit_single_hit_row`) only fires inside
  `_admit_waiting_requests`, i.e. mid-run admission after the initial
  cohort has already run its prefill. This is a P-2 Option B design
  fact, not a bug (cross-cohort prefix reuse would complicate the
  cohort-seal invariant and was deferred); it is recorded as open
  question **Q-012** in PLAN §10 for a follow-up look.
- **Consequence for C.1 acceptance:** two consecutive
  `generate_batch([prompt], ..., prefix_cache=shared_pc,
  max_batch_size=1)` calls would *both* run miss-path prefills — the
  second call's single admission is sealed into its own fresh initial
  cohort and never queries `shared_pc`. Encode fires on both calls
  (prefix is registered when row 0 terminates); decode never fires.
  The acceptance must instead use a **single** `generate_batch([p,
  p], ..., prefix_cache=shared_pc, max_batch_size=1)` call: prompt 0
  admits into the initial cohort and runs miss-path prefill; prompt 1
  enters the waiting queue; after row 0 terminates and reclaim
  registers its aligned prefix (via `_extract_and_insert_prefix`),
  the next `step()` invokes `_admit_waiting_requests`, which routes
  prompt 1 through `_admit_single_hit_row` → `fetch_detached_blocks`
  → `codec.decode_block`. The single-call `[p, p]` workload shape
  therefore exercises both paths in one go and is what every
  assertion below uses.

### 8.1 Encode + decode counters on paired-prompt single call

Construct a counting-wrapper `IdentityCodec` that increments
`encode_calls` / `decode_calls` on each method, and build the
`RadixPrefixCache` with a `SyntheticPrefixBlockStore(codec=...)` that
wraps this counter. Run one `generate_batch` call:

```python
from silica.engine import Engine
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)
store = SyntheticPrefixBlockStore(block_size=16, codec=counting_codec)
shared_pc = RadixPrefixCache(block_size=16, store=store)
events = list(engine.generate_batch(
    [prompt, prompt],  # two copies — see §8.0
    params,
    prefix_cache=shared_pc,
    max_batch_size=1,
))
```

(Construction pattern mirrors `scripts/bench_p2_baseline.py:149-150`
and `scripts/probe_p2_preload.py:113` — `adapter_for_repo(repo)` +
`Engine(adapter, kv)`. The `SyntheticPrefixBlockStore(codec=...)`
constructor kwarg is the C.1 deliverable.)

Row 0 flows through the miss path:
`_admit_miss_cohort` → initial-cohort prefill → decode until
`max_tokens` → terminal →
`_extract_and_insert_prefix` → `insert_detached` →
`store.register_detached` → `codec.encode_block` per layer per aligned
block.

Row 1 then enters `_admit_waiting_requests`, which sees the now-
populated prefix: `peek` → `_admit_single_hit_row` → `lookup` →
`fetch_detached_blocks` → `store.fetch_detached` →
`codec.decode_block` per layer per usable hit block.

**Encode assertion:**
`encode_calls >= floor(len(prompt_tokens) / block_size) × num_layers`.
(`_extract_and_insert_prefix` slices `aligned_tokens = (computed_len
// block_size) × block_size` with `computed_len ≥ prompt_len`, so the
lower bound uses `floor(prompt_len / block_size)` without the `- 1`
adjustment.)

**Decode assertion:**
`decode_calls >= floor((len(prompt_tokens) - 1) / block_size) × num_layers`.
`_admit_single_hit_row` in `silica/scheduler/batcher.py` around line
1011 uses `max_aligned = ((len - 1) // block_size) * block_size` to
leave at least one suffix token for first-token prefill — invariant
S-5 edge 1.

**Prompt-length requirement:** `len(prompt_tokens) >= 2 * block_size + 1`,
which equals 33 under the default Qwen3 `block_size = 16`, **and**
`len(prompt_tokens) % block_size != 0`. Under the guarded length,
`floor(len / bs) == floor((len - 1) / bs)`, so both assertions use
the same `block_count` lower bound and a count asymmetry would signal
a real integration bug. An exact multiple of `block_size` would split
them by one block and the two clauses would assert different numbers
on the same prompt — confusing rather than informative. A
catalog-test-style `test_prompt_tokenization_invariants` guards both
halves against tokenizer changes; under the `_PROMPT_FIXTURE` shipped
with C.1 the prompt measures 34 tokens (mod 16 == 2).

The original PLAN wording "≥ 1 call per active KV block on a
single-request `Engine.generate('Hello', max_tokens=4)` run"
overspecified on three axes (wrong entry point; "Hello" tokenizes to
≪ `block_size`; a single cold request never fires the decode path);
PLAN is amended in the same commit as this opening to match §8.1
above.

### 8.2 `store.resident_bytes()` parity with the radix-tree total

After the same `[p, p]` workload:

```python
store = shared_pc._store
total_blocks = len(store.live_block_ids())       # total resident prefix blocks
num_layers = engine._adapter.config.num_layers
block_size = shared_pc.block_size
layout = engine._adapter.kv_layout()
bytes_per_token_per_layer = 2 * layout.n_kv_heads * layout.head_dim * layout.dtype.size

expected = total_blocks * num_layers * block_size * bytes_per_token_per_layer
assert store.resident_bytes() == expected

# And the radix-tree view agrees:
assert shared_pc.node_count() == total_blocks
```

The right-hand side is **not** `_count_evictable_prefix_blocks ×
_kv_bytes_per_block`. That budgeter helper counts *leaf nodes with
zero live hits* — the eviction-candidate set — and deliberately skips
internal radix nodes (see `silica/scheduler/budget.py` ~ line 353).
For a 34-token prompt the tree is a chain (root → node1 → node2);
only node2 is a leaf, so the budgeter would report 1 block while the
store holds 2 and the radix tree has `node_count() == 2`.
`node_count()` is the total-resident view this §8.2 assertion
requires.

Under a future BlockTQCodec, `store.resident_bytes()` diverges from
the `total_blocks × num_layers × block_size ×
bytes_per_token_per_layer` formula (the whole point of the codec).
P-5 proper will extend §8.2 into a "store is the authority"
assertion and let the budgeter read it; the P-4.5-C test pins only
today's identity invariant.

### 8.3 Baseline-identity numerical invariant

Run the same `[p, p] max_batch_size=1` workload twice — once with a
no-codec `SyntheticPrefixBlockStore(block_size=...)` (`codec=None`,
pass-through), once with a `SyntheticPrefixBlockStore(block_size=...,
codec=IdentityCodec(...))`. Both row 0's and row 1's emitted token
sequences must be byte-identical between the two runs — not within
fp16 tolerance, but exactly equal.

Under `IdentityCodec.decode_block(block)` returning `(block.k,
block.v)` (`silica/kvcache/codec.py:101-102`), the concatenated
`mx.concatenate(k_parts, axis=2)` in `build_seeded_batch_kv` produces
a tensor that is bitwise identical to what the pass-through path
produces — `CodedBlock.__init__` at `codec.py:98-99` assigns `k` / `v`
by reference without copy. Any divergence here is an integration bug,
not an fp16 drift issue; the codec literally does nothing.

The three-layer acceptance criterion used for chunked prefill
(event-taxonomy + per-row token-count + direct-mlx-lm batched
reference) does not apply here: the Option (B) hook does not change
batch composition, so the fp16 drift problem that forced the
three-layer criterion in chunked-prefill acceptance does not arise.
Byte-identity is achievable and strictly stronger; use it.

The `[p, p]` shape matters for §8.3 specifically because row 1
exercises the hit path on both runs. A single-prompt shape would only
exercise the miss path, and §8.3 would fall back to comparing fp16
prefill outputs — which is still a valid byte-identity claim under
`IdentityCodec` but loses the hit-path coverage that catches codec
bugs inside `decode_block`.

### 8.4 Tensor-reference tripwire (C.1 implementation note)

The §8.3 token-stream byte-identity assertion is value-level. A
future codec that silently inserts a defensive copy inside
`encode_block` or `decode_block` would regress from byte-identical-
reference to byte-identical-value without failing §8.3 — the copied
tensor holds the same bytes. The C.1 test adds a tripwire that uses
`is` / `id()` on the per-block K/V returned by `fetch_detached`
after a direct `register_detached` round-trip on synthetic tensors,
asserting reference preservation. Runs without the HF cache so it
stays green on cold CI. Recorded here so future codec authors know
the tripwire exists and either match the IdentityCodec reference-
preservation contract or explicitly update this invariant.

---

## 9. References

- `silica/kvcache/codec.py` — I-3 `KVCodec` Protocol + IdentityCodec.
- `silica/kvcache/store.py` — `PrefixBlockStore` Protocol +
  `SyntheticPrefixBlockStore` (detached K/V) +
  `PagedPrefixBlockStore` (detached methods raise `NotImplementedError`).
- `silica/kvcache/prefix.py::RadixPrefixCache.insert_detached` /
  `fetch_detached_blocks` — runtime callers of the detached methods.
- `silica/scheduler/seed_kv.py::build_seeded_batch_kv` — consumes the
  detached per-block K/V tuples; D-003 / Q-009 / R-7 reason it must
  receive fp16 shape-preserving decodes.
- `silica/scheduler/batcher.py::_admit_single_hit_row` (~ line 1058) —
  end-to-end sequence: `lookup` → `fetch_detached_blocks` →
  `build_seeded_batch_kv` → `forward_batched`.
- `silica/scheduler/budget.py::MemoryBudgeter` — admission decision
  path; `_count_evictable_prefix_blocks × _kv_bytes_per_block`
  formula is the authority on the *eviction-shortfall* block count
  (leaf-zero-hit only, under-reports internal nodes). The total
  resident prefix bytes are not currently exposed at the budgeter
  layer; the P-4.5-C parallel observable at the store layer fills
  that gap (see §8.3).
- `silica/engine/__init__.py::Engine._record_tail_metrics`
  (~ line 165) — the only runtime reader of
  `kv_manager.budget().resident_bytes`; used only for the
  `resident_mb` metric output, not for admission decisions.
  (`ChatSession` lives in `silica/chat/session.py` and does not
  read this field; the earlier reference to `ChatSession` here was a
  symbol-lookup slip.)
- `plans/PLAN.md` §7 P-5 Strategy — target of the line-539 amendment
  landing in this commit.
- `plans/P4_5_CHUNKED_PREFILL_OPENING.md` — companion opening doc for
  the P-4.5-B bridge; cited for the three-layer acceptance criterion
  precedent (chunked-prefill's fp16 batch-composition drift; §8.3
  above argues this spike is not subject to that precedent because
  Option (B) does not change batch composition).
- D-003 (PLAN §9) — no compressed-domain attention fast path in v0.1.
- D-009 (PLAN §9) — MLX-native hot path; no `torch.Tensor` in
  `silica.*` runtime.
- Q-009 / R-7 — MLX variable-length attention kernel absence.
- Q-010 (PLAN §10) — the P-4.5-B.1 admission-reorder heuristic; the
  reason P-4.5 is a bridge rather than straight P-5.
