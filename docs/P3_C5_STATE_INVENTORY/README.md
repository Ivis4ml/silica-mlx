# P-3-C5.0 — Hybrid-DeltaNet state inventory

Runtime-measured per-layer cache shape / dtype / byte cost on the
three Qwen3.5 targets named in `docs/P3_C5_OPENING.md` §3.1.

The probe captures **both** cache-path shapes C5 needs to see:

- `KVManager.cache_list(req_id)` — single-request path that
  `adapter.prefill(tokens, KVHandle(req_id))` writes into. Not
  the path C5.1 / C5.2 will operate on, but a useful baseline.
- `ContinuousBatcher._batch_cache` equivalent — the `list[Any]`
  that `adapter.make_batch_cache([left_padding_per_row])` returns,
  driven through one `forward_batched(model, tokens_2d,
  cache_list)` pass at `B=2` with different-length rows. This is
  the cache shape C5.1 `snapshot_recurrent_state(cache_list,
  row_idx)` will be called on at every preempt / prefix-hit site.

The acceptance gate for C5.0 is covered by both — opening doc
§3.1 / §6.1 commits the snapshot API to the batcher-owned cache,
so a single-request-only inventory would understate what C5.1
actually has to support.

**Measurement date:** 2026-04-24 (against the adapter tree at
commit `5d67aa9`, v1.7.5 + P-3-C5 opening).

**Probe:** `docs/P3_C5_STATE_INVENTORY/probe.py` — loads one repo
via `adapter_for_repo`, runs the single-request prefill AND one
`B=2` `forward_batched` pass via `adapter.make_batch_cache`, then
walks both caches and emits one JSON with a `single_request`
section and a `batched` section.

```bash
PYTHONPATH=. uv run python docs/P3_C5_STATE_INVENTORY/probe.py \
    --repo Qwen/Qwen3.5-0.8B \
    --out docs/P3_C5_STATE_INVENTORY/inventory_qwen3_5_0_8b.json
PYTHONPATH=. uv run python docs/P3_C5_STATE_INVENTORY/probe.py \
    --repo Qwen/Qwen3.5-4B \
    --out docs/P3_C5_STATE_INVENTORY/inventory_qwen3_5_4b.json
PYTHONPATH=. uv run python docs/P3_C5_STATE_INVENTORY/probe.py \
    --repo mlx-community/Qwen3.5-35B-A3B-4bit \
    --out docs/P3_C5_STATE_INVENTORY/inventory_qwen3_5_35b_a3b.json
```

## 1. Layer pattern — all three targets use `full_attention_interval=4`

| target | hidden layers | DeltaNet layers | GLOBAL layers | GLOBAL layer indices |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-0.8B | 24 | 18 | 6 | `[3, 7, 11, 15, 19, 23]` |
| Qwen3.5-4B | 32 | 24 | 8 | `[3, 7, 11, 15, 19, 23, 27, 31]` |
| Qwen3.5-35B-A3B (4bit) | 40 | 30 | 10 | `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]` |

GLOBAL layers always sit at positions `3, 7, 11, ..., L-1` — one
GLOBAL for every 4 hidden layers. DeltaNet layers occupy the other
three positions per block. The pattern is identical across the
three scales. The classification `layer.is_linear == False` is
the runtime oracle the batcher uses today.

## 2. Single-request path — `KVManager.cache_list(req_id)`

Per-layer cache shape after a 23-token prefill under `adapter.prefill`:

Slot-0 (**conv state**, bf16, DeltaNet layers):

| target | shape | dtype | bytes / layer |
| --- | --- | --- | ---: |
| Qwen3.5-0.8B | `(1, 3, 6144)` | `bfloat16` | 36 864 |
| Qwen3.5-4B | `(1, 3, 8192)` | `bfloat16` | 49 152 |
| Qwen3.5-35B-A3B | `(1, 3, 8192)` | `bfloat16` | 49 152 |

Slot-1 (**recurrent state**, fp32, DeltaNet layers):

| target | shape | dtype | bytes / layer |
| --- | --- | --- | ---: |
| Qwen3.5-0.8B | `(1, 16, 128, 128)` | `float32` | 1 048 576 |
| Qwen3.5-4B | `(1, 32, 128, 128)` | `float32` | 2 097 152 |
| Qwen3.5-35B-A3B | `(1, 32, 128, 128)` | `float32` | 2 097 152 |

GLOBAL layer `KVCache` (K / V each, bf16):

| target | K / V shape | dtype | bytes / layer (K + V) | n_kv_heads | head_dim |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | `(1, 2, 256, 256)` | `bfloat16` | 524 288 | 2 | 256 |
| Qwen3.5-4B | `(1, 4, 256, 256)` | `bfloat16` | 1 048 576 | 4 | 256 |
| Qwen3.5-35B-A3B | `(1, 2, 256, 256)` | `bfloat16` | 524 288 | 2 | 256 |

## 3. Batched path — `adapter.make_batch_cache([left_padding_per_row])`

This is the cache shape C5.1 / C5.2 / C5.3 actually operate on.
Cache class per layer kind:

- **DeltaNet layers** → `mlx_lm.models.cache.ArraysCache(size=2,
  left_padding=left_padding)`. Per-row `left_padding` is carried
  by the cache object; slot tensors carry a leading batch
  dimension.
- **GLOBAL layers** → `mlx_lm.models.cache.BatchKVCache(left_padding=left_padding)`.
  K / V carry leading batch dim; `offset` / `left_padding` are
  per-row `int32` arrays (not scalars, as they would be on
  single-request `KVCache`).

Per-layer shapes on a probe run with `B=2`, rows of 10 / 23
tokens, `padded_n_tokens = 23` (left-padding row 0 by 13):

Slot-0 (DeltaNet conv state, bf16):

| target | shape | dtype | bytes / layer |
| --- | --- | --- | ---: |
| Qwen3.5-0.8B | `(2, 3, 6144)` | `bfloat16` | 73 728 |
| Qwen3.5-4B | `(2, 3, 8192)` | `bfloat16` | 98 304 |
| Qwen3.5-35B-A3B | `(2, 3, 8192)` | `bfloat16` | 98 304 |

Slot-1 (DeltaNet recurrent state, fp32):

| target | shape | dtype | bytes / layer |
| --- | --- | --- | ---: |
| Qwen3.5-0.8B | `(2, 16, 128, 128)` | `float32` | 2 097 152 |
| Qwen3.5-4B | `(2, 32, 128, 128)` | `float32` | 4 194 304 |
| Qwen3.5-35B-A3B | `(2, 32, 128, 128)` | `float32` | 4 194 304 |

GLOBAL `BatchKVCache` per layer (K / V each, bf16):

| target | K / V shape | offset (per-row) | left_padding (per-row) |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | `(2, 2, 256, 256)` | `int32[2] = [10, 23]` | `int32[2] = [13, 0]` |
| Qwen3.5-4B | `(2, 4, 256, 256)` | same | same |
| Qwen3.5-35B-A3B | `(2, 2, 256, 256)` | same | same |

### 3.1 Per-row indexing semantics (what the probe actually measured)

On `ArraysCache` the probe confirms:

- Each DeltaNet slot's leading dim is the batch dim. `row_idx = 0`
  is the first row; `row_idx = 1` is the second. Shapes carry the
  expected `(B, ...)` form on every target (see §3 tables).

On `BatchKVCache` (GLOBAL, not part of the C5 snapshot surface —
full-attention K / V is already routed through
`SyntheticPrefixBlockStore` by P-4.5): K / V are also batch-
leading; `offset` / `left_padding` are per-row `int32` arrays so
the batcher can track each row independently.

**Primitives available but not probed**: `ArraysCache.extract(row_idx)`
and `ArraysCache.merge([...])` exist in mlx-lm and are the
expected building blocks for C5.1's snapshot / restore
implementation (per `docs/P3_DELTANET_SURVEY.md` §3.4). The C5.0
probe does **not** call either primitive — it only records cache
shape / dtype / nbytes after one forward. A full round-trip
(`extract → mx.eval → merge → reforward → logits bit-exact`)
verification is C5.1's acceptance test, not C5.0 evidence.

## 4. Totals at the measured `B` values (per-row cost is B-invariant)

| target | B | recurrent | conv | GLOBAL K/V | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 1 | 18.87 MB | 0.66 MB | 3.15 MB | 22.67 MB |
| Qwen3.5-0.8B | 2 | 37.75 MB | 1.33 MB | 6.29 MB | 45.37 MB |
| Qwen3.5-4B | 1 | 50.33 MB | 1.18 MB | 8.39 MB | 59.90 MB |
| Qwen3.5-4B | 2 | 100.66 MB | 2.36 MB | 16.78 MB | 119.80 MB |
| Qwen3.5-35B-A3B | 1 | 62.91 MB | 1.47 MB | 5.24 MB | 69.62 MB |
| Qwen3.5-35B-A3B | 2 | 125.83 MB | 2.95 MB | 10.49 MB | 139.26 MB |

**Scaling observation.** Per-row cost is constant — the B=2 row
of each target is exactly 2× the B=1 row, within rounding. Total
live-batch residency scales linearly in `B`; the per-row figure
does not change with `B`. C5.3 memory budgeting distinguishes
these:

- Per-prefix-tree-node snapshot cost = **per-row cost at B=1**
  (a snapshot is bound to one request's prefix path, not to a
  live batch position).
- Total live-batch recurrent residency = `B × per-row cost`
  (separate concern, already accounted by
  `StateDelta.recurrent_bytes()`).

`StateDelta.recurrent_bytes()` reported by the single-request
path:

- Qwen3.5-0.8B: 19.54 MB = conv 0.66 + recurrent 18.87 + marginal
  `ArraysCache` overhead.
- Qwen3.5-4B: 51.51 MB.
- Qwen3.5-35B-A3B: 64.39 MB.

These match the sum of per-DeltaNet-layer `ArraysCache.nbytes`,
consistent with D-015 item-5. The batched path's
`StateDelta.recurrent_bytes()` is not directly probed here
(`forward_batched` does not return a `StateDelta`); it equals
`B × single-request value` by construction of the `ArraysCache`
layout.

## 5. Inside the runtime write surface — mlx-lm grep, reproducible

Claim: every runtime write to DeltaNet cache slots across the
mlx-lm source tree lands on either `cache[0]` or `cache[1]`, so
the snapshot surface covers the full state.

**Source root (reproducible; uses project venv so the import
resolves from a clean checkout):**

```bash
MLX_LM_ROOT=$(PYTHONPATH=. uv run python -c \
    'import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))')
echo "$MLX_LM_ROOT"
# -> .venv/lib/python3.13/site-packages/mlx_lm
```

**Grep for cache-slot writes:**

```bash
rg -n 'cache\[[01]\]\s*=' "$MLX_LM_ROOT/models"
```

Exit: every match is inside a linear-attention forward:
`mamba.py`, `mamba2.py`, `jamba.py`, `rwkv7.py`, `plamo2.py`,
`recurrent_gemma.py`, `nemotron_h.py`, `granitemoehybrid.py`,
`falcon_h1.py`, `kimi_linear.py`, `bailing_moe_linear.py`,
`lfm2.py`, `lfm2_moe.py`, `qwen3_5.py`, `qwen3_next.py`. No
other module writes slot indices; no match on `cache[2]` /
`cache[3]`.

**Qwen3.5 exact write sites** (via `rg -n 'cache\[[01]\]\s*=' "$MLX_LM_ROOT/models/qwen3_5.py"`):

- `mlx_lm/models/qwen3_5.py:164` — `cache[0] = mx.take_along_axis(conv_input, positions, axis=1)`.
  Guarded by `if cache.lengths is not None:` on line 161. Writes
  into `cache[0]` at per-row positions driven by
  `cache.prepare(lengths=...)`.
- `mlx_lm/models/qwen3_5.py:166` — `cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])`.
  The `else` branch when `cache.lengths is None`.
- `mlx_lm/models/qwen3_5.py:197` — `cache[1] = state`
  (recurrent state write, once per forward, taken on every call).

**Which branches this probe exercises — and which it does not.**
`rg -n '\.prepare\(|prepare\(lengths' silica/` returns zero
matches: today's silica runtime never calls
`ArraysCache.prepare(lengths=...)` on any code path, neither the
single-request `adapter.prefill` nor the batched
`forward_batched(..., make_batch_cache(left_padding))` driven
here. That means `cache.lengths` is `None` on every DeltaNet
`ArraysCache` this probe or the running batcher ever observes,
and the forward always takes the line-166 contiguous branch. The
JSON records `"lengths": null` on every DeltaNet layer in the
batched section of every committed inventory, consistent with
this. **Line 164 is a dead branch inside silica's current
hybrid-DeltaNet call surface**, kept in mlx-lm for future
prepare-with-lengths drivers that silica has not yet adopted.

**Consequence for C5.1 / C5.2 / C5.3.** The snapshot / restore
implementation only needs to handle the `cache.lengths is None`
state shape the probe measured. A snapshot taken at the times
C5.2 / C5.3 care about (preempt, prefix-hit admission) observes
the same slot layout the probe captured; restore writes tensors
back in the same layout. If a future sub-unit (possibly a P-3-C
follow-up that adopts `prepare(lengths=...)` for true batched
prefill) flips silica into the line-164 branch, C5.1's snapshot
payload remains correct by construction — the two branches write
to the same `cache[0]` slot with the same shape; only the
per-row content selection differs.

**No third slot**, no escape path: `ArraysCache(size=2)`
allocates exactly two slots; `rg -n 'cache\[2\]' "$MLX_LM_ROOT/models/qwen3_5.py"`
returns no matches. The snapshot surface therefore has a bounded
fixed shape (exactly two tensors per DeltaNet layer), not an
open-ended dictionary.

**No module-level state beyond the cache.** The Metal kernel in
`gated_delta_kernel` (`gated_delta.py:110-115`) consumes
`cache[1]` as the running state, advances it per timestep, and
writes the final state back to `cache[1]` before
`GatedDeltaNet.__call__` returns. Per-timestep intermediates
live on the GPU for the kernel call duration and are released
immediately after; not a persistent side channel.

## 6. Implications for C5.1 / C5.3

- **Snapshot payload shape (C5.1).** Each DeltaNet layer
  contributes exactly one conv-state tensor and one recurrent-
  state tensor. Under the batched path the leading dim is `B`;
  C5.1's snapshot implementation slices to a single row via
  `ArraysCache.extract(row_idx)` before materialising. The
  `_RecurrentLayerEntry` tuple-of-tuples shape from
  `docs/P3_C5_OPENING.md` §5.1 remains correct.
- **Per-row indexing in batched caches** (§3.1): `row_idx` maps
  directly to the leading batch dim on every DeltaNet / GLOBAL
  per-layer cache; `ArraysCache.extract(row_idx)` and
  `ArraysCache.merge([...])` exist and are the intended
  primitives for snapshot / restore.
- **Lazy init edge case.** Both DeltaNet slots are allocated
  lazily inside the first forward. Snapshot taken before a
  layer's first forward would see `None` in both slots. C5.1
  handles this by either returning an "empty" snapshot marker or
  taking snapshots only after first forward — decision is
  C5.1's; implication is that `restore` of an empty snapshot
  must re-trigger lazy init rather than inject fp32 zeros.
- **Memory budget for Option (b) per-block snapshots (C5.3)**,
  per-row-per-tree-node:
  - Qwen3.5-0.8B: **~19.5 MB per snapshot**. 16-block prefix
    tree → 312 MB. Feasible.
  - Qwen3.5-4B: **~51.5 MB per snapshot**. 16-block tree → 824
    MB. Feasible on 48 GB M5 Pro.
  - Qwen3.5-35B-A3B: **~64.4 MB per snapshot**. 16-block tree →
    1.03 GB. On 4-bit 20 GB checkpoint, ~5% of resident; still
    feasible.
  - Snapshots are keyed to prefix tree nodes, NOT to live batch
    positions; the tree's LRU eviction caps memory cost
    independent of batch size.
  - No C5 target forces Option (a) full-replay fallback on memory
    grounds; Option (b) lean from the opening doc stays.
- **R-C5-2 aliasing verification.** Snapshot constructor must
  `mx.array(live_tensor)` + `mx.eval` so subsequent in-place
  writes at `qwen3_5.py:164 / :166 / :197` cannot mutate the
  snapshot. C5.1 acceptance includes a "mutate-live, read-
  snapshot" test that captures slot references, runs one more
  forward to provoke in-place writes, and asserts the snapshot
  tensors are unchanged.

## 7. Artefacts committed

| file | contents |
| --- | --- |
| `probe.py` | the probe script; reproducible per target; single-request + batched in one invocation |
| `inventory_qwen3_5_0_8b.json` | 24 layers × both paths |
| `inventory_qwen3_5_4b.json` | 32 layers × both paths |
| `inventory_qwen3_5_35b_a3b.json` | 40 layers × both paths |
| `README.md` | this consolidated inventory |

All three JSON files share the schema:

```
{
  "repo": str,
  "n_hidden_layers": int,
  "single_request": { ... per-layer cache after adapter.prefill ... },
  "batched":        { ... per-layer cache after forward_batched B=2 ... }
}
```

Each path section carries `n_global_layers`, `n_deltanet_layers`,
`global_layer_indices`, `deltanet_layer_indices`, totals in
bytes, and a `layers: [...]` list with one entry per hidden
layer (shape / dtype / nbytes per slot or per K/V).

## 8. C5.0 acceptance checklist

- [x] Qwen3.5-0.8B inventory complete on **both** cache paths
  and cross-verified against real forward passes (single-request
  via `adapter.prefill`, batched via `forward_batched` on
  `adapter.make_batch_cache(...)`). 24 hidden layers, 18
  DeltaNet + 6 GLOBAL.
- [x] Qwen3.5-4B inventory captured on both paths. 32 layers
  (24 + 8).
- [x] Qwen3.5-35B-A3B (4bit) inventory captured on both paths.
  40 layers (30 + 10).
- [x] `mlx-lm` `cache[0]` / `cache[1]` write-site grep run with
  the exact command recorded in §5. No third slot; no escape
  path.
- [x] Per-block-boundary snapshot memory cost estimated per
  target — feasibility established on all three C5 targets.
- [x] Opening doc §3.3 "snapshot covers both slots" verified.
- [x] Batched path exercises the `cache_list + row_idx` shape
  C5.1 will see, with per-row indexing (`row_idx` = leading
  batch dim on every slot) confirmed on shape inspection.
  `ArraysCache.extract / merge` primitives are recorded as
  available per mlx-lm `cache.py` but NOT round-tripped by this
  probe; round-trip verification is C5.1's acceptance.
- [x] Silica-wide `rg -n '\.prepare\(|prepare\(lengths'` shows
  zero matches, so the mlx-lm `qwen3_5.py:164`
  `mx.take_along_axis` branch is dead in the current silica call
  graph. C5.1 / C5.2 / C5.3 only need to handle the
  `cache.lengths is None` state shape this probe captured.

C5.0 closed. Next: C5.1 — adapter-owned snapshot / restore
implementation on `Qwen3_5Adapter`, per opening doc §6.2.
