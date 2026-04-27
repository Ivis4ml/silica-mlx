# P-2 Gate-0.5 — BatchKVCache rebuild primitives (mid-run admission)

**Date:** 2026-04-17
**mlx-lm version:** 0.31.2
**Result:** **PASS** — all five sub-checks green; Unit #16 architecture locked.
**Probe:** `scripts/probe_batch_kvcache_rebuild.py` (exit 0).

## Why Gate-0.5

Gate-0 (`plans/P2_GATE_0.md`) verified only the basic prefill + decode path
on a fixed-row-set ``BatchKVCache``. The P-2 ``ContinuousBatcher`` (task
Unit 16) must also handle **mid-run admission** (new requests joining an
in-flight batch) and **preemption** (dropping rows mid-decode), which
Gate-0 explicitly flagged as out-of-scope. The original P-2 Opening
proposal (v2.1) assumed these would need a custom rebuild path; **Codex
cross-reference on 2026-04-17 surfaced that mlx-lm already ships four
primitives that do this work upstream**:

| Primitive | Operation | Use |
| --- | --- | --- |
| `BatchKVCache.extend(other)` | in-place append rows | admission |
| `BatchKVCache.filter(indices)` | in-place drop rows, shift-left padding | preempt / release |
| `BatchKVCache.extract(idx)` | one row → single-request `KVCache` | eager extract for option-B |
| `BatchKVCache.merge([caches])` | classmethod: `KVCache` list → `BatchKVCache` | full rebuild |

Before Unit #16 commits to an architecture this gate verifies those
primitives' invariants empirically.

## What the probe verifies

All five sub-checks use synthetic fp16 K/V tensors with row- and
position-encoded values (`k[b,h,t,d] = offset + b*1000 + t*10 + d*0.1 + h`),
so "K/V unchanged" is a literal element-wise equality assertion — no
logit-level indirection through a real model.

### Q1 — `extract + merge` round-trip preserves per-row K/V

Build a `BatchKVCache(left_padding=[0,2,1])` (B=3), prefill 5 tokens,
decode 1, then extract each row and rebuild via `merge`. For every row,
the `left_padding → _idx` window (the attention-visible portion) is
byte-identical to the original. **PASS.** Establishes that the full
rebuild path is lossless.

### Q2 — `filter` preserves survivors' effective K/V

`BatchKVCache(left_padding=[1,0,2,0])` (B=4), prefill 6 + decode 1, then
`filter([0, 2])`. Surviving rows' `left_padding → _idx` windows are
unchanged. The internal shift-left optimisation reduces
`_idx` when `min(left_padding) > 0`, but the effective window is
preserved. **PASS.** `filter` is the preempt / release primitive.

### Q3 — `extend` appends rows without disturbing incumbents

`main = BatchKVCache(left_padding=[0,0])` prefilled to `_idx=4`.
`new_single = BatchKVCache(left_padding=[0])` prefilled to `_idx=3`.
`main.extend(new_single)`. After: B=3, `main._idx=4` (unchanged,
`max(4,3)`), incumbent rows 0/1 byte-identical to pre-extend snapshots,
new row 2 carries the standalone cache's K/V with `left_padding=1`
(auto-padded to right-justify against `main._idx=4`). **PASS.** `extend`
is the admission primitive.

### Q4 — row index stability

Observed behaviour (by inspection rather than explicit assertion):

- **`extend`**: self's rows occupy indices `0..B_self-1` (unchanged);
  new rows appended at `[B_self, B_self+1, …]`.
- **`filter(indices)`**: kept rows re-indexed to `0..len(indices)-1` in
  the order given.

**Consequence**: Silica's `slot_table: req_id → row` **cannot remain
stable across a `filter` call**. It must be rebuilt from the kept-indices
list. This is a **material change to the P-2 Opening Layer-A invariant**
and is recorded as an amendment below.

### Q5 — `filter` drops filtered-out rows' K/V (option-B needs eager extract)

Build `BatchKVCache(B=2)`, prefill, `cache.extract(1)` → `detached`
(a single-request `KVCache`), then `cache.filter([0])`. The detached
`KVCache` still carries the dropped row's K/V element-by-element. If
extract had NOT happened before filter, row 1's data would have been
unrecoverable from the filtered cache. **PASS.** Confirms the
architectural requirement:

> For **option-B prefix reuse**, `RadixPrefixCache` must eagerly
> `extract()` every pinned block's K/V **before** the owning request's
> row is `filter()`'d. The extracted `KVCache` (or per-block slice
> thereof) is the detached storage that future hits copy from.

This is also a **material change to the P-2 Opening §RadixPrefixCache**
section and is recorded as an amendment below.

### Q6 — per-op cost under `mx.eval` barrier (B=8, `_idx=100`)

Wall-clock timing with `mx.eval` after each op, averaged over 10 iterations:

| Op | Avg latency / op |
| --- | --- |
| `extract×B + merge` (full rebuild) | ~2 ms |
| `filter([0,2,4,6])` (preempt half) | ~0.6 ms |
| `extend(+1 row)` (admission) | ~0.9 ms |

**All three ops are sub-millisecond to low-millisecond at test scale.**
Even 100 admission-plus-preempt events during a 32-token decode add
~150 ms of overhead — negligible against Qwen3-0.6B decode wall time.
**PASS.** No premature optimisation needed.

## Probe output

```text
Gate-0.5 probe — mlx-lm 0.31.2 BatchKVCache rebuild primitives
(Q1) extract + merge round-trip preserves per-row K/V:
    PASS (identity up to left_padding re-alignment)
(Q2) filter preserves survivors' effective K/V:
    PASS (filter → preempt primitive; row indices reshuffle)
(Q3) extend appends rows without disturbing incumbents:
    PASS (extend → admission primitive; self row indices stable)
(Q5) filter drops row K/V unless extract()'d first:
    PASS (option-B requires eager extract before owner filter)
(Q6) rebuild / filter / extend cost (mx.eval barrier, B=8, T=100):
    rebuild (extract×8 + merge) :   1.93 ms / op
    filter ([0,2,4,6])           :   0.62 ms / op
    extend (+1 row)              :   0.84 ms / op
    PASS (all ops < 20ms at test scale)

RESULT: PASS — Unit #16 architecture locked.
```

## Material changes to P-2 Opening (v2.1 → v2.2)

### Change 1 — Layer-A invariant (§PagedKVCache)

**Before (v2.1):** "a request's row is fixed for its lifetime ... no
mid-request row compaction — a finishing request's row may leave a hole
which future admissions fill, but live requests never move."

**After (v2.2):** "the `(req_id → row)` mapping in ``slot_table`` is
stable **within a contiguous no-filter interval**. After every
`BatchKVCache.filter` call (triggered by preempt / release), the kept
rows are re-indexed `0..K-1` in the order of the kept-indices list, and
`slot_table` rebuilds from that ordering. `extend` (triggered by
admission) does not change existing rows' indices — new rows append at
`[B_self, B_self+1, …]`."

### Change 2 — Option-B implementation (§RadixPrefixCache)

**Added** to the physical semantics discussion:

> **Eager-extract requirement**: When the owning request of a pinned
> block transitions to DONE / ABORTED / PREEMPTED, the scheduler must
> `cache.extract(row)` for every pinned block **before** the `filter`
> that removes the row. The extracted K/V is stored as
> per-block detached `KVCache` slices inside `RadixPrefixCache`. On a
> later prefix hit, the batcher copies K/V from these detached slices
> into the new request's row.
>
> Storage cost per pinned block (Qwen3-0.6B fp16):
> `2 * n_layers * n_kv_heads * block_size * head_dim * 2 bytes`
> ≈ 450 KB / block / 28 layers. For a 2k-token shared prefix at
> `block_size=16` → 128 blocks → ~60 MB detached storage. Acceptable.

### Change 3 — `runner.forward_batched` signature (§1 / runner)

`silica.mlx.runner.forward` currently accepts 1-D `(T,)` → `(V,)`. Unit
16a must add a batched variant:

```python
def forward_batched(
    model: Any,
    tokens: mx.array,           # (B, T) int
    cache_list: list[Any],      # list[BatchKVCache]
) -> mx.array:                  # (B, V) last-position logits per row
    ...
```

Single-request `forward` stays as an adaptor that wraps
`forward_batched` with `tokens[None]` and `logits[0]`.

## Decision log entries (for next PLAN.md update)

- **New Q-012**: `BatchKVCache.finalize()` + `right_padding` handling for
  ragged admit — in scope for P-4 optimisation, out of scope for P-2.
- **New Q-013**: `BatchRotatingKVCache` (sliding-window models) — P-2
  supports `AttentionKind.GLOBAL` only; `SLIDING` deferred until a
  dedicated probe confirms matching primitives.
- **Close pre-condition for Unit #16**: mid-run admission + preempt are
  supported natively by upstream mlx-lm; no custom rebuild code required.
- **Amendment to P-2 Opening v2.1**: row-index stability rule and
  option-B eager-extract requirement both upgraded in v2.2.
