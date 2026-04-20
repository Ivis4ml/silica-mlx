# P-3 Batched Sliding-Window KV — mlx-lm Source Survey (P-3-D2.0)

| Field         | Value                                                             |
| ------------- | ----------------------------------------------------------------- |
| Unit          | P-3-D2.0 (mlx-lm `BatchRotatingKVCache` audit)                    |
| Prerequisite  | P-3-D1 / D1.1 (commits `ff50afe` / `9e82861`) — single-request `Gemma4Adapter` shipped; batched path intentionally blocked by the `SLIDING` capability-gate rejection |
| Targets       | Gemma4-31B dense batched path (primary); any future sliding-window family (Mistral-7B, Gemma-2, Qwen3-Next-VL) would dispatch through the same scheduler surface |
| Date          | 2026-04-20                                                        |
| Source commit | `9e82861` (probe + adapter + single-request smoke, no batched yet) |

## 1. Purpose and key correction

The P-3-D0.2 Gemma4 survey (commit `5608e97`) stated in §5.3:

> **mlx-lm ships no `BatchRotatingKVCache`** ... Gemma4 batched
> admission would have to ... implement the sliding variant of
> `BatchKVCache` with per-row rotation + left-padding alignment.
> This is the PLAN §Q-013 item; no prior work lands it today.

**That claim is incorrect.** `mlx_lm/models/cache.py` defines a complete
`BatchRotatingKVCache` at line 1100 (162 lines). This survey replaces
the §5.3 statement with an audit of what the class actually provides
and how much of P-3-D2 / D3 (Q-013) it reduces to pure wiring. The
Gemma4 survey itself carries a brief superseded-by pointer at §5.3
(added in the D2.0 commit); the authoritative picture is below.

## 2. Scope

Read under `.venv/lib/python3.13/site-packages/mlx_lm/models/cache.py`:

- `RotatingKVCache` (lines 410-592) — single-request sliding cache used
  by `Gemma4TextModel.make_cache()` today.
- `BatchKVCache` (lines 880-1099) — batched full-attention cache,
  already wired by Silica via C3a/b/c.
- `BatchRotatingKVCache` (lines 1100-1444) — batched sliding cache;
  the focus of this survey.
- `dynamic_roll` helper (line 871) — per-row shift-along-axis used by
  both batched caches for left-padding alignment.

Also re-read `mlx_lm/models/base.py::create_attention_mask` (45-55) and
`Gemma4TextModel._make_masks` (gemma4_text.py:494-506) to confirm the
mask-generation path delegates to whatever cache class is in place.

## 3. `BatchRotatingKVCache` surface

### 3.1 Construction

```python
BatchRotatingKVCache(max_size: int, left_padding: list[int])
```

Unlike single-request `RotatingKVCache(max_size, keep=0)`, the batched
class:

- Requires `left_padding` at construction time (no "mutate later"
  path for the initial per-row padding).
- Has **no `keep` parameter** — batched sliding always rotates the
  full window; retaining a prefix like Mistral's `keep=4` is not
  supported here. For Gemma4 this is fine (`keep=0` at the single-
  request level).
- Initializes `offset = mx.array([-l for l in left_padding])` —
  starts negative so that after the first `update_and_fetch(keys of
  shape [B, H, S, D])` each row's `offset` ends at `S - left_padding`,
  matching the "number of real tokens seen" convention that
  `BatchKVCache` uses.

### 3.2 `update_and_fetch(keys, values)` branches on `S`

`cache.py:1234-1237`:

```python
def update_and_fetch(self, keys, values):
    if keys.shape[2] == 1:
        return self._update_in_place(keys, values)
    return self._update_concat(keys, values)
```

- `S == 1` (decode step): `_update_in_place` writes into the ring
  buffer at `self._idx`, rotates when `_idx == max_size`, and returns
  a view over the valid prefix of the buffer.
- `S > 1` (prefill or cohort batched prefill): `_update_concat`
  temporally re-orders the existing buffer, trims to
  `max_size + S - 1` to guarantee each emitted token sees at least
  `max_size` context, appends the new slab, and updates per-row
  `offset` / `left_padding`.

Both branches call `mx.depends(self.keys, (self.left_padding, self.offset))`
before returning, binding the evaluation order so downstream
`scaled_dot_product_attention` sees consistent state.

### 3.3 `prepare` / `finalize` for left + right padding

`cache.py:1239-1259`. `prepare(left_padding=..., lengths=...,
right_padding=...)` accepts any of three kwargs:

- `left_padding`: additional left-pad applied pre-write. Can be
  called only when the cache is still empty (`self.keys is None`);
  subsequent left-pad shifts go through `extend` / `_update_concat`'s
  roll logic.
- `lengths` + `right_padding`: right-padded inputs (sequences padded
  on the right to the cohort max) need the cache to know real
  per-row lengths so trim/rotate does not evict valid tokens.
  Setting `_lengths` triggers a `dynamic_roll` in `_update_concat`
  that realigns all rows before the trim/append sequence.
- `finalize()`: called after the right-padded prefill completes; it
  does one final per-row roll to push each row's real tokens
  flush-left in the ring, then clears `_lengths`. Required before
  any decode step (`_update_in_place` asserts `_lengths is None`).

### 3.4 `make_mask` — sliding-window mask with per-row left-padding

`cache.py:1297-1324`. Produces a `(B, 1, N, total_size)` bool mask
that simultaneously encodes:

- Causal: `linds >= rinds`.
- Sliding window: `linds < rinds + window_size`.
- Per-row left-padding: `rinds >= mx.expand_dims(left_padding, ...)`
  masks out pad positions on each row independently.
- Post-rotation shift: when the buffer has rotated (S=1 path with
  `_idx >= max_size`), the mask is `mx.roll`'d by `_idx + 1` along
  the last axis so the ring layout is read correctly.

This is the right shape for `scaled_dot_product_attention` on the
batched forward. `create_attention_mask(h, cache, window_size=...)`
in `base.py:49` delegates here automatically when the cache object
has a `make_mask` attribute, so no adapter-side mask work is needed.

### 3.5 Batched primitives (filter / extend / extract / merge)

Same API as `BatchKVCache`, all present:

- `filter(batch_indices)` (`cache.py:1326`) — in-place keep subset;
  reconstructs `keys` / `values` / `offset` / `left_padding`.
- `extend(other)` (`cache.py:1336`) — in-place concat two batched
  caches along batch axis, padding the shorter ring with zeros and
  aligning `_idx` / `rotated` before concat. Requires the two caches
  to share `max_size`.
- `extract(idx)` (`cache.py:1383`) — pull row `idx` out as a
  single-request `RotatingKVCache` (matching what
  `make_cache()` would have produced for B=1).
- `merge(caches)` (`cache.py:1401`) *(classmethod)* — rebuild a
  `BatchRotatingKVCache` from a list of `RotatingKVCache`s; accepts
  heterogeneous per-row lengths by right-padding.

These are the same primitives C3a/b/c wired for `BatchKVCache`. The
ContinuousBatcher's reclaim / admit / extend paths already call
`.filter(kept)` and `.extend(other)` polymorphically — so a hybrid
cache list mixing `BatchKVCache` + `BatchRotatingKVCache` should flow
through those same sites without scheduler changes.

### 3.6 Save / restore, trimmability

- `state` / `state.setter`: round-trip `(keys, values, offset,
  left_padding)`. `meta_state`: round-trip `(max_size, _offset, _idx,
  rotated)`. These make `save_prompt_cache` / `load_prompt_cache`
  work on batched sliding caches.
- `is_trimmable()` returns `True` only before the ring rotates — once
  the cache hits `max_size`, trim (for speculative rollback) is not
  defined because older tokens have been evicted. Matches
  `RotatingKVCache.is_trimmable`.

## 4. Implications for Silica (D2 / D3 scope)

### 4.1 D2 shrinks from "implement the container" to "wire the factory"

Concretely, `Gemma4Adapter.make_batch_cache(left_padding)` today
raises `NotImplementedError` naming Q-013. The new body is:

```python
from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

def make_batch_cache(self, left_padding):
    tc = self._text_config_dict(self._model)
    sliding_window = int(tc.get("sliding_window", 0) or 0)
    caches = []
    for kind in self._attention_pattern.per_layer:
        if kind == AttentionKind.SLIDING:
            caches.append(BatchRotatingKVCache(
                max_size=sliding_window, left_padding=left_padding
            ))
        elif kind == AttentionKind.GLOBAL:
            caches.append(BatchKVCache(left_padding=left_padding))
        else:  # unreachable — _build_attention_pattern already enforces
            raise AssertionError(f"unreachable kind={kind}")
    return caches
```

Roughly ten lines. This replaces the Q-013 raise.

### 4.2 D3 gate lift for `SLIDING` looks structurally identical to C3c

The `ContinuousBatcher._enforce_capability_gate` already supports the
"add a kind to `_SUPPORTED_ATTENTION_KINDS` and delete its row from
`_unsupported_kind_reason`" move from C3c. Applying the same shape to
`SLIDING` gives:

```python
_SUPPORTED_ATTENTION_KINDS = frozenset({
    AttentionKind.GLOBAL,
    AttentionKind.HYBRID_DELTANET,
    AttentionKind.SLIDING,
})
```

Plus matching test updates (the current
`test_capability_gate_rejects_non_supported_patterns` parametrizes
over `{RECURRENT, SLIDING, HYBRID}`; `SLIDING` moves out of that list
into an accept test, mirroring C3c's handling of `HYBRID_DELTANET`).

### 4.3 D2 + D3 can land as one commit

The two pieces are mutually dependent: `make_batch_cache` producing
the right shape is only useful after the gate lifts; the gate lift
depends on the factory producing correct caches. Given C3c's
precedent of landing lift + real-model smoke together, the minimal
load-bearing commit is:

- `silica/models/gemma4.py`: rewrite `make_batch_cache`.
- `silica/scheduler/batcher.py`: extend `_SUPPORTED_ATTENTION_KINDS`;
  shrink `_unsupported_kind_reason`.
- `tests/test_gemma4_adapter.py`: replace the Q-013-raise test with
  a proper per-layer factory test; update the factory expectations.
- `tests/test_batcher.py`: remove `SLIDING` from the rejection
  parametrize; add `test_capability_gate_accepts_sliding_after_d3`
  analogous to C3c's hybrid-accept test.
- `tests/test_p3_gemma4_batched_smoke.py` (new): real-model smoke
  mirroring `test_p3_hybrid_batched_smoke.py`.

Strict parity vs single-request mirroring C3d was tested in D3.1 and
does **not** hold on the observed Gemma4-31B-4bit toolchain: B=1
parity holds, but B>1 drifts from solo greedy output by token index
2 on the `"The capital of France is"` probe at `max_tokens=16`.
The landed invariant is therefore the P-2-style degraded one:
Silica batched output must match a direct mlx-lm batched reference
using the same `Gemma4Adapter.make_batch_cache(left_padding)` cache
list.

### 4.4 `KVLayout` caveat remains in place

D-open-1 option (a) from the Gemma4 survey still applies: the
single-shape `KVLayout` captures only the sliding-layer values.
`MemoryBudgeter.bytes_per_token` under batched execution will mis-
count on Gemma4 by ~20% until D4 lands a per-kind decomposition or an
aggregate-bytes field. Not blocking for smoke-level correctness;
matters once P-4 bench numbers cite KV residency.

## 5. Open items (P-3-D2 local)

Local to the D2 / D3 arc; not promoted to the global `Q-NNN`
Open Questions space (same convention as `C-open-*` /
`D-open-*`).

- **D2-open-1** — `BatchRotatingKVCache.prepare` takes `left_padding`
  but the constructor already does. Which call site should Silica use
  when building caches? The constructor signature is cleaner; D2's
  `make_batch_cache` should pass `left_padding` to `__init__` and not
  call `prepare` unless right-padding becomes relevant. Confirm
  during D2 implementation.
- **D2-open-2** — `BatchRotatingKVCache` asserts "Left padding can
  only be added to an empty cache" (line 1241-1244). Silica's
  mid-run admission path (`_admit_miss_cohort`) builds a NEW batched
  cache for the admitted cohort, then `.extend`s it onto the
  existing `self._batch_cache`. The new cache is empty at build time
  so this assertion is satisfied — but verify once the D2 wiring
  lands.
- **D2-open-3** — Cache merge into an existing batched cache
  requires matching `max_size`. All Gemma4 sliding layers share one
  `sliding_window`, so per-layer maxes are equal inside a single
  adapter; however, mixing adapters with different sliding windows in
  one batcher would break. Not on the near-term roadmap; record and
  move on.
- **D2-open-4** — `BatchRotatingKVCache` has no `keep` parameter;
  Mistral-style "keep the first K tokens across rotation" is not
  supported batched. Gemma4 uses `keep=0`, so this does not block D2;
  flag for any future Mistral-7B family work.

## 6. Revised P-3-D roadmap (replaces P3_GEMMA4_SURVEY.md §5.4)

| Sub-unit | Scope | Estimate |
| --- | --- | --- |
| **D2** | `Gemma4Adapter.make_batch_cache` returns hybrid `[BatchRotatingKVCache / BatchKVCache]` list. Unit tests updated. No gate change. | ~50 lines of code + tests. |
| **D3** | Extend `_SUPPORTED_ATTENTION_KINDS` to include `SLIDING`; error-locator + `_unsupported_kind_reason` trim; real-model Gemma4-31B batched smoke via `Engine.generate_batch`; README row updated. | Structurally identical to C3c. |
| **D3.1** | B=1 parity vs single-request, same-prompt symmetry, B>1 parity vs direct mlx-lm batched reference, unequal-length row-lifecycle smoke. Strict B>1 batched-vs-single greedy parity was attempted and degraded after observed drift. | Same spirit as P-2's honest oracle downgrade. |
| **D4** | `KVLayout` / `MemoryBudgeter` correctness on per-kind KV shapes (option (b) aggregate-bytes or (c) per-kind decomposition). | Separate bench-driven unit. |
| **D5** | Gemma4-26B MoE variant — overlaps P-3-E. | Defer until E track starts. |

D2 and D3 **may** land as a single commit given the mutual
dependency, mirroring C3c's "gate lift + smoke" precedent. Whether
to split is a readability-over-diff-size tradeoff to decide at D2
scoping time, not here.

## 7. References

- mlx-lm `cache.py` — `BatchRotatingKVCache` (1100-1444),
  `BatchKVCache` (880-1099), `RotatingKVCache` (410-592),
  `dynamic_roll` (871).
- mlx-lm `base.py` — `create_attention_mask` (45-55) delegating to
  `cache.make_mask` when present.
- mlx-lm `gemma4_text.py:494-506` — `_make_masks` per-type mask
  build; `gemma4_text.py:653-666` — single-request `make_cache`.
- Silica — `silica/models/gemma4.py::make_batch_cache` (current
  Q-013 raise), `silica/scheduler/batcher.py::_SUPPORTED_ATTENTION_KINDS`
  (current `{GLOBAL, HYBRID_DELTANET}`),
  `docs/P3_GEMMA4_SURVEY.md` §5.3 (being superseded).
- PLAN — §Q-013 (sliding-window batching), §D-015, §P-3 deliverables.
- P-3-D0 probe (commit `8718bcd`) — Gemma4-31B structural signals.
- P-3-D1 / D1.1 commits (`ff50afe` / `9e82861`) — single-request
  Gemma4-31B dense adapter and real-model smoke.
