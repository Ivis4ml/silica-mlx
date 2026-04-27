# P-3 DeltaNet Plumbing — Structural Survey (P-3-C0)

| Field         | Value                                                             |
| ------------- | ----------------------------------------------------------------- |
| Unit          | P-3-C0 (Qwen3.5 DeltaNet hybrid survey, read-only)                |
| Prerequisite  | P-3-B empirical finding: Qwen3.5-27B-4bit loads via `mlx_lm.load` |
| Targets       | Qwen3.5-0.8B (dev loop) and Qwen3.5-27B (primary 27B dense target) |
| Date          | 2026-04-19                                                        |
| Source commit | `cbcd7a6`                                                          |

## 1. Purpose

P-3-C is the largest single unit inside P-3 — DeltaNet recurrent-state plumbing
that unblocks batched execution for Qwen3.5-0.8B (currently `Partial`) and
Qwen3.5-27B simultaneously. Every downstream P-3-C sub-unit (C1–C6) touches
either the adapter-local state store, the scheduler's capability gate, or
both. Before writing any of that code we need to be certain about what the
*actual* DeltaNet hidden state is — its shape, dtype, per-layer count, per-
request cost, and the mlx-lm APIs already shipped for batched manipulation.

This document is the record of that structural read, with direct citations
into the mlx-lm source tree (local venv copy, read 2026-04-19). It does not
propose new interfaces — those go into C1 onward once alignment on the
facts below is reached.

## 2. Scope

Read sources under `.venv/lib/python3.13/site-packages/mlx_lm/models/`:

- `qwen3_5.py` — `Model`, `TextModel`, `Qwen3_5TextModel`, `DecoderLayer`,
  `GatedDeltaNet`, `TextModelArgs`, `make_cache`, and `sanitize`.
- `gated_delta.py` — `gated_delta_update`, `gated_delta_ops`,
  `gated_delta_kernel`, `_gated_delta_step_ops`, `compute_g`, and the Metal
  kernel source templates.
- `cache.py` — `ArraysCache`, `KVCache`, and `make_prompt_cache`.

Out of scope for C0: the MoE variant `qwen3_5_moe.py`, multi-modal vision
heads (`qwen3_vl*.py`), and the sharded / distributed paths.

## 3. Findings

### 3.1 Layer selection rule

`DecoderLayer.is_linear` (qwen3_5.py:212):

```python
self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
```

With `full_attention_interval=4` (default, and confirmed on 27B) the pattern
over layer indices 0..N-1 is `[D, D, D, G, D, D, D, G, ...]` — a strict 3:1
DeltaNet-to-global repeat. For 27B (64 layers) this gives 48 DeltaNet + 16
global, matching the P-3-B probe output byte-for-byte.

`TextModel.make_cache` (qwen3_5.py:304):

```python
return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]
```

So the per-layer cache is *heterogeneous* — `ArraysCache(size=2)` for
DeltaNet layers, regular `KVCache` for the global-attention ones. This is
exactly what `silica.kvcache.simple.SimpleKVCache.from_model` already holds
(it stores the list verbatim and passes it to the forward).

### 3.2 `ArraysCache(size=2)` slot layout for DeltaNet

Inside `GatedDeltaNet.__call__` (qwen3_5.py:132-206):

- `cache[0]` — the **1D depthwise convolution state**.
  - Allocated on the first call: `mx.zeros((B, conv_kernel_size - 1, conv_dim), dtype=inputs.dtype)` (qwen3_5.py:151).
  - Updated every step by concatenating the new `qkv` into the conv
    window and slicing the last `n_keep = conv_kernel_size - 1` rows
    (qwen3_5.py:158-166).
  - Dtype follows `inputs.dtype` (bf16/fp16 at inference).
- `cache[1]` — the **recurrent (associative-memory) hidden state**.
  - Allocated inside `gated_delta_update` (gated_delta.py:276-279):
    `mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)`.
  - **Always fp32**, independent of activation dtype. This is a load-
    bearing fact for memory accounting — fp32 doubles the per-layer cost
    versus what a naive "count tokens × bytes" estimate would suggest.
  - Advanced per-timestep by the Metal kernel in `gated_delta_kernel`
    (gated_delta.py:110-115); falls back to the ops-based
    `gated_delta_ops` when `use_kernel=False` or Metal is unavailable
    (gated_delta.py:281-283).

### 3.3 Concrete DeltaNet shape on Qwen3.5-27B (measured)

Pulled from `model.language_model.args` of `mlx-community/Qwen3.5-27B-4bit`
on 2026-04-19 via a direct `mlx_lm.load` probe:

| Symbol             | Field                      | Value |
| ------------------ | -------------------------- | ----- |
| L_deltanet         | — (48 of 64 layers)        | 48    |
| L_global           | — (16 of 64 layers)        | 16    |
| Hv                 | `linear_num_value_heads`   | 48    |
| Hk                 | `linear_num_key_heads`     | 16    |
| Dk                 | `linear_key_head_dim`      | 128   |
| Dv                 | `linear_value_head_dim`    | 128   |
| conv_kernel_size   | `linear_conv_kernel_dim`   | 4     |
| key_dim            | Hk × Dk                    | 2048  |
| value_dim          | Hv × Dv                    | 6144  |
| conv_dim           | 2 × key_dim + value_dim    | 10240 |
| head_dim (global)  | `head_dim`                 | 256   |
| n_kv_heads (global) | `num_key_value_heads`      | 4     |
| n_attn_heads (global) | `num_attention_heads`   | 24    |

### 3.4 `ArraysCache` batched-operation primitives

`cache.py:594-696`. Relevant methods for Silica's scheduler:

| Method                        | Source                     | Use                                                        |
| ----------------------------- | -------------------------- | ---------------------------------------------------------- |
| `filter(batch_indices)`       | cache.py:620               | In-place keep subset of rows. Direct analogue of `BatchKVCache.filter`. Needed for **preempt** (16d) + **reclaim**. |
| `extend(other)`               | cache.py:628               | In-place concatenate along batch axis. Analogue of `BatchKVCache.extend`. Needed for **mid-run admission** (16c.1). |
| `extract(idx)`                | cache.py:642               | Build a single-row `ArraysCache` from a batched one. Used during preempt snapshot. |
| `merge(caches)` *(classmethod)* | cache.py:670             | Combine a list of single-row `ArraysCache`s into one batched cache. Useful for prefix-cache replay and admission. |
| `prepare(lengths=...)`        | cache.py:647               | Attach per-row valid length — parallels `BatchKVCache` left-padding bookkeeping. |
| `make_mask(N)`                | cache.py:660               | Produce the per-row validity mask consumed by `GatedDeltaNet` when `cache.lengths is not None`. |
| `advance(N)`                  | cache.py:654               | Shift `lengths` / `left_padding` after a forward. Called by `GatedDeltaNet.__call__` at the end. |
| `finalize()`                  | cache.py:650               | Clear `lengths` / `left_padding` — signals that all per-row padding has been resolved. |
| `empty()`                     | cache.py:691               | True when slot 0 is still `None` (no prefill has run yet). |
| `nbytes` *(property)*         | cache.py:694               | Sum of per-slot `nbytes` — mirrors `KVCache.nbytes`, so `SimpleKVCache.budget()` already accounts for conv_state + recurrent state without modification. |

**Load-bearing implication.** Batched-execution primitives are already
built into `ArraysCache`. P-3-C3 (`BatchRecurrentStateStore` in PLAN terms)
does not need to invent a new container — it needs to *use these methods
correctly in the batcher loop*. This shrinks C3's scope materially versus
the original PLAN framing.

### 3.5 Forward-pass contract

`Qwen3_5TextModel.__call__` (qwen3_5.py:254-275):

- Builds two masks up-front: `fa_mask` for global layers and `ssm_mask`
  for DeltaNet layers. `fa_mask` is the usual causal attention mask;
  `ssm_mask` is consumed by `GatedDeltaNet` to zero out padding positions.
- Walks the cache list layer-by-layer, passing each layer its matching
  cache object. No cross-layer state; the adapter can continue to rely on
  the one-cache-per-layer shape.

`GatedDeltaNet.__call__` writes **in place** to `cache[0]` and `cache[1]`
every call, then calls `cache.advance(S)` (qwen3_5.py:164, 197, 198). The
cache object must be mutable between calls — a frozen `StateDelta`
snapshot is NOT what the forward reads; the forward reads and writes the
live `ArraysCache` bound inside `kv_handle`'s list.

This reaffirms D-015's framing: `StateDelta` exposed to the engine is a
**read-only budgeting snapshot** (`recurrent_bytes()`), not the live state.
The adapter owns the live store.

### 3.6 Sanitize path and MTP

`sanitize` in qwen3_5.py:307-331 + 384-398 on the outer `Model`:

- Skips `mtp.*` (multi-token prediction), `vision_tower.*`, and
  `model.visual.*` weights. So Silica gets a pure text-only language
  head.
- Shifts `.input_layernorm.weight`, `.post_attention_layernorm.weight`,
  `model.norm.weight`, `.q_norm.weight`, `.k_norm.weight` by `+1.0` when
  MTP is present or when `conv1d.weight` hasn't been pre-sanitized. The
  D-014 / R-8 "MTP disabled" prerequisite is already handled by mlx-lm's
  sanitize path — Silica does not need to redo it.

## 4. Memory accounting (per request, Qwen3.5-27B-4bit)

Using the shapes from §3.3:

- **Conv state** — `(1, 3, 10240)` per DeltaNet layer, activation dtype
  (fp16 in the 4-bit quantized path): 61,440 bytes/layer × 48 layers =
  **~2.8 MB**.
- **Recurrent state** — `(1, 48, 128, 128)` per DeltaNet layer, fp32:
  3,145,728 bytes/layer × 48 layers = **~150 MB**. This is the dominant
  per-request overhead introduced by DeltaNet.
- **Global-attention KV** — 16 global layers × 2 (K+V) × 4 KV heads × 256
  head_dim × activation_bytes × prompt_len. At fp16 and a 512-token
  context: 16 × 2 × 4 × 256 × 2 × 512 = **~33 MB**.

Total live state per request @ 512 tokens of context ≈ **186 MB**.

**Consequences for P-3-C:**

- `StateDelta.recurrent_bytes()` returning 0 today (P-0 stub behaviour) is
  a 150 MB underestimate per request on 27B. `MemoryBudgeter` can
  mis-schedule significantly if we turn on batching before C2 lands a
  real `recurrent_bytes()`.
- Sequential scaling at 8-way batch: 8 × 186 MB = ~1.5 GB. On a 48 GB
  M5 Pro with ~30.5 GB resident after load (P-3-B finding), that leaves
  ~17 GB headroom → 8-way batch consumes ~10% of the remaining budget
  just for live state. Feasible, but tight under longer contexts.

### 4.1 Qwen3.5-0.8B numbers

Not probed yet — 1.77 GB download deferred. When C1 tests against 0.8B we
should update this section. Expected order of magnitude: Hv/Hk/Dk/Dv are
family-wide defaults unless the 0.8B checkpoint overrides, so per-layer
recurrent cost is close to the 3 MB figure but with fewer layers (probably
24–32 instead of 64). Total live state per request is likely 15–30 MB on
0.8B.

## 5. Current Silica behaviour

- `silica.kvcache.simple.SimpleKVCache.from_model(model)` calls
  `mlx_cache.make_prompt_cache(model)` (simple.py:64-66), which delegates
  to `model.make_cache()` (cache.py:31-32) — so the heterogeneous
  18+6-like list comes through unmodified.
- `SimpleKVCache.budget()` (simple.py:114-122) sums `c.nbytes` across the
  list; `ArraysCache.nbytes` (cache.py:694) returns the actual
  conv_state + recurrent_state bytes. **This is already correct** for
  single-request accounting.
- `Qwen3_5Adapter.prefill` / `decode_step` (qwen3_5.py:101-113 in Silica)
  call `self._kv_manager.cache_list(kv_handle.req_id)` and pass the list
  to `forward(model, tokens, cache_list)` — mlx-lm's model mutates the
  list in place. No Silica-side state management beyond ownership.
- `StateDelta` returned from prefill / decode_step is currently the empty
  default (qwen3_5.py:100, 107 in Silica) — `_recurrent_bytes=0`. C2 will
  make this a real number.
- `ContinuousBatcher._enforce_capability_gate` (post-D-016) reads
  `adapter.capabilities()` and rejects this adapter because
  `has_recurrent_state=True`. C4 is what lifts that gate.

**What works today (P-3-B probe confirmed):** single-request
`Engine.generate` on 0.8B *and* 27B, using the above path, end-to-end.

## 6. Implications for P-3-C sub-units

Referenced against the proposal in the preceding chat round (C1 adapter
helpers, C2 recurrent_bytes, C3 batch store, C4 capability gate, C5
preempt/replay, C6 integration tests):

- **C1 (adapter helpers: `commit_state` / `rollback_state` /
  `state_from_prefix` / `free_state`).** Mostly scoped to single-request.
  In mlx-lm's shape `commit_state` is a no-op (ArraysCache has no separate
  speculative buffer — `advance(S)` already makes the write visible);
  `rollback_state` needs to trim the last `n_reject` positions from the
  conv window and rewind the recurrent state — but `gated_delta_update`
  writes the full new state every call, so rollback requires **snapshot
  before draft**, not "trim after draft". For P-3 we only need the
  Python signatures in place; real rollback semantics are a P-7 concern
  (Q-008 speculative).
- **C2 (`recurrent_bytes()` real value).** One-liner change inside
  `Qwen3_5Adapter.prefill` / `.decode_step`: compute total bytes across
  the ArraysCache slots and return `StateDelta(_recurrent_bytes=…)`.
  Formula: `sum(c.nbytes for layer in deltanet_layers for c in layer.cache)`.
- **C3 (batched recurrent state).** ArraysCache already supports
  `filter` / `extend` / `extract` / `merge` / `prepare`. The work is
  plumbing these into `ContinuousBatcher` — analogous to how
  `BatchKVCache.filter` / `.extend` are already wired. No new container
  class is needed. The tricky part is `prepare(lengths=...)` vs
  `BatchKVCache` left-padding: both must be coherent so `fa_mask` and
  `ssm_mask` line up across rows.
- **C4 (capability gate).** Two options: (a) extend `ModelCapabilities`
  with a new bit (e.g. `supports_batched_recurrent: bool`) that adapters
  must set to unblock batching; (b) keep `ModelCapabilities` minimal and
  let the batcher accept `HYBRID_DELTANET` unconditionally once C3 is
  done. D-016 precedent says **don't add speculative bits**, so (b) is
  probably correct — unblock when the implementation genuinely supports
  it, record it in the Changelog.
- **C5 (preempt/replay with recurrent state).** Qwen3.5 layers update
  the recurrent state in-place every call; there is no "draft buffer"
  to snapshot. Preempt/replay therefore requires either (i) explicit
  `ArraysCache.extract(idx)` on the preempted row before filtering it
  out, stashing the extracted caches in the preempt record, and feeding
  them back via `ArraysCache.merge` on replay; or (ii) accepting that
  preempted rows lose their recurrent state and must re-prefill from the
  prefix cache. Option (i) is bit-exact; option (ii) requires
  `state_from_prefix` to do the reconstruction, which per D-015 "full
  prefix only" rule is feasible when the row is replayed against the
  same prompt. **Design decision deferred to C5 itself.**
- **C6 (integration tests).** Real-model tests on Qwen3.5-0.8B exercising
  the batched path against the current single-request parity output as
  oracle. Requires 0.8B download; can be gated behind a pytest marker so
  it does not run on a clean machine.

## 7. Open items (P-3-C local)

These are open questions scoped to the P-3-C track only — they do **not**
enter the global `Q-NNN` Open Questions space in `plans/PLAN.md §10` (CRUD
convention reserves that ID series for document-wide questions). Any of
these items that turns out to shape architecture rather than C-internal
implementation is promoted to a proper `Q-NNN` at that time.

- **C-open-1** — Does `ArraysCache.prepare(lengths=...)` play well with
  `BatchKVCache(left_padding=...)` when both appear in the same step's
  cache list? Hypothesis: yes, because `Qwen3_5TextModel.__call__` builds
  `fa_mask` and `ssm_mask` independently. Verify once C3 wires this up.
- **C-open-2** — Recurrent state is fp32 (gated_delta.py:279). Can we
  safely cast it to fp16 / bf16 for batched memory savings without
  drifting the output beyond the D-015 rollback-round-trip bit-exactness
  target? Measurement-gated; tentatively parked until C5.
- **C-open-3** — `state_from_prefix(req_id, token_ids)` semantics when
  the prefix-cache hit is **non-trivial but partial**. D-015's v0.1 rule
  is "full prefix only"; C0 confirms the rule makes sense because
  DeltaNet state is a running accumulation of the full sequence, not a
  per-token K/V that can be sliced.

## 8. References

- mlx-lm `qwen3_5.py` — lines 86 (`GatedDeltaNet.__init__`), 132
  (`GatedDeltaNet.__call__`), 209 (`DecoderLayer`), 254
  (`Qwen3_5TextModel.__call__`), 304 (`make_cache`), 307 (`sanitize`).
- mlx-lm `gated_delta.py` — lines 126 (`_gated_delta_step_ops`), 214
  (`gated_delta_ops`), 262 (`gated_delta_update`), 279 (recurrent state
  init with `dtype=mx.float32`).
- mlx-lm `cache.py` — lines 594 (`ArraysCache`), 620 (`filter`), 628
  (`extend`), 642 (`extract`), 647 (`prepare`), 660 (`make_mask`), 670
  (`merge`), 694 (`nbytes`).
- Silica — `silica/models/qwen3_5.py`, `silica/kvcache/simple.py`,
  `silica/scheduler/batcher.py::_enforce_capability_gate`.
- PLAN — D-014 (P-1 scope), D-015 (recurrent state as first-class
  state_delta tenant), D-016 (ModelCapabilities).
- P-3-B probe output (2026-04-19) — confirmed 64 layers = 48 DeltaNet +
  16 global on Qwen3.5-27B; single-request `Engine.generate` runs.
