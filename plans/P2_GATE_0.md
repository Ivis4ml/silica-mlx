# P-2 Gate-0 — mlx-lm BatchKVCache batched injection

**Date:** 2026-04-16
**mlx-lm version:** 0.31.2
**Result:** **PASS** — all three sub-checks green.
**Probe:** `scripts/probe_batch_kvcache.py` (exit 0).

## Question

P-2 opening (`plans/P2_OPENING.md` v2.1) fixed a blocking pre-condition: the
entire P-2 paged-cache + scheduler stack relies on
``mlx_lm.models.cache.BatchKVCache`` as the physical storage layer. If its
batched semantics drift — the shared insert cursor ``_idx`` getting out of
sync with the per-row ``offset`` vector, for example — every P-2
deliverable drifts with it. Gate-0 is required to be green before any P-2
unit begins, same blocking weight as Gate A / Gate B at P-1 entry.

## What the probe verifies

### (1) Direct invariant — `_idx` vs `offset` stay in lockstep

Construct `BatchKVCache(left_padding=[1, 3, 0])` (B=3), run one prefill
step of T=4 followed by one decode step of T=1. The invariant asserted at
each phase is `offset[i] == _idx - left_padding[i]`:

| Phase | `_idx` | `offset` (observed) | Expected |
| --- | --- | --- | --- |
| pre-prefill | 0 | [-1, -3, 0] | [-1, -3, 0] |
| after prefill (T=4) | 4 | [3, 1, 4] | [3, 1, 4] |
| after decode (T=1) | 5 | [4, 2, 5] | [4, 2, 5] |

This is the targeted assertion for the 2026-04 upstream offset-bug class
flagged by external review. On the installed mlx-lm 0.31.2 the invariant
holds through prefill + decode. Additional shape assertions confirm that
the returned `(k, v)` slices match `(B, n_kv_heads, _idx, head_dim)` and
that `nbytes > 0` / `empty() == False` after writes (D-012 canonical
surface).

### (2) Model-forward integration — batched cache driven end-to-end

Mirrors the Gate A pattern at P-1 entry. A toy 1-layer duck-typed model
with `B=3` input exercises `model(tokens, cache=[BatchKVCache])`; a
traced subclass confirms `update_and_fetch` is actually invoked and
`_idx` advances as expected. Confirms the P-2 Unit-4 (`PagedKVCache`) can
hand a batched cache entry to `runner.forward` identically to how
SimpleKVCache does in P-1.

### (3) Silica does not use `mlx.core.fast.metal_kernel`

Negative check — walk the `silica/` tree for references. Zero matches
confirms the 2026-03 custom-kernel lazy-graph use-after-free bug class
(also flagged by external review) is not on our runtime path. This also
anchors the Q-009 / Q-011 decision: if a future trigger opens the
paged-attention kernel track, this check would need to flip to a
positive-form assertion (`silica.*` uses only a specific, stability-
reviewed kernel registration path).

## Probe output

```text
Gate-0 probe — mlx-lm 0.31.2
Claim: BatchKVCache batched invariants hold; model forward drives it;
Silica uses no custom Metal kernels.
(1) Direct invariant test (prefill + decode across B=3):
    PASS (offset and _idx stay in lockstep; 2026-04 bug class clear)
(2) Model-forward integration (toy B=3 model, traced cache):
    PASS (batched model(tokens, cache=[bkv]) path green)
(3) Silica does not use mlx.core.fast.metal_kernel:
    PASS (2026-03 custom-kernel lazy-graph bug class not on our path)

RESULT: PASS — P-2 Gate-0 cleared. Unit work may begin with
task #12 (RequestState).
```

## Scope of coverage / explicit gaps

Gate-0 green on the **basic prefill + decode path**. The following
BatchKVCache code paths are **not** exercised here and must be re-probed
(or at least unit-tested) when the relevant P-2 unit lands:

- `BatchKVCache.prepare(left_padding=..., right_padding=..., lengths=...)`
  — called to introduce new admits mid-run or adjust padding; its
  interaction with the shared `_idx` is exactly where upstream-offset
  bugs historically surface. **Must be re-probed before Unit #16
  (`ContinuousBatcher`)** which is the first P-2 unit to touch admission
  mid-generation.
- `BatchKVCache.finalize()` — right-padding removal via `dynamic_roll`.
  Edge case for prompts whose lengths differ at the tail. Probe before
  Unit #13 (`PagedKVCache.append_slot`) if ragged admits are supported;
  P-2 scope per the opening proposal can defer this to P-4 optimisation.
- Per-row preemption (releasing a single batch row while others continue
  advancing `_idx`). mlx-lm does not expose a documented primitive for
  this — Silica's `PagedKVCache.free(req_id)` will have to simulate via
  row zeroing + marking the slot FREE + leaving `_idx` untouched for the
  surviving rows. Worth a targeted invariant test at Unit #13.

These are flagged here so a future reviewer bisecting a scheduler bug
knows which Gate-0 scope limits were **intentional**.

## Decision log update

- **P-2 Gate-0:** cleared at 2026-04-16, mlx-lm 0.31.2.
- **R-12 sub-finding (2026-04 offset bug):** not observed on the basic
  path for 0.31.2; `prepare()` / `finalize()` paths remain probe-able
  items (see §Scope above).
- **Q-009 closure stands:** no `mlx.core.fast.metal_kernel` usage found
  in silica; no regression into the custom-kernel bug class from 2026-03.
- **P-2 Unit #12 (`RequestState`) is unblocked.**
