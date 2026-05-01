# P-6 Track C.4 DFlash spike — REPORT

D-021 step 6 spike measurement bundle. One file across sub-units; new
rows append as the spike progresses from (αβ.1) → (η).

| Sub-unit | Row | Status | Date |
| -------- | --- | ------ | ---- |
| (αβ.1) | `c_capture_hidden(k=16)` on Qwen3.5-0.8B | landed | 2026-05-01 |
| (αβ.2) | MoE capture-cost row | pending | — |
| (η)    | dense 27B `qwen3.5-27b-warm-decode-c4-dflash` | pending | — |
| (η)    | MoE 35B-A3B `qwen3.5-moe-35b-a3b-warm-decode-c4-dflash` | pending | — |

---

## (αβ.1) — `c_capture_hidden(k=16)` on cached Qwen/Qwen3.5-0.8B

Closes OQ-5 (`plans/P6_C4_DFLASH_OPENING.md` §5.5) on the 0.8B
fixture. The measurement constrains the §1 prediction band's new
`c_capture_hidden(k)` term that F-1 introduced post-α.

### Setup

- Fixture: cached `Qwen/Qwen3.5-0.8B` (HF cache hit; same path the
  step 5 (f) parity test uses).
- Workload: single-request `decode_step_multi(k=16)` vs
  `decode_step_multi_with_capture(k=16, capture_layer_ids)`.
  Capture set: `{0, num_layers // 2, num_layers}` = three layer
  outputs (embedding pre-stack, mid-stack, post-stack). 28 hidden
  layers on 0.8B → captured set `{0, 14, 28}`.
- Measurement protocol: each iteration loads a fresh `Qwen3_5Adapter`
  to start from an empty KV cache (matches the cycle-1 cost the F-1
  state-machine re-runs at every cycle's verify forward; warm-cache
  amortisation is downstream and out-of-scope for this row). 3 warmup
  iterations + 20 measurement iterations; **median** wall-clock
  reported. Materialisation forced via `mx.eval(logits, *captured)`.
- Hardware: M5 Pro 48 GB (silica's primary target).
- Script: `scripts/microbench_capture_hidden.py` —
  ```text
  python -m scripts.microbench_capture_hidden \
      --repo Qwen/Qwen3.5-0.8B --k 16 --warmup 3 --iters 20
  ```

### Result

| Quantity | Value |
| -------- | ----- |
| `decode_step_multi(k=16)` baseline median | **26.374 ms** |
| `decode_step_multi_with_capture(k=16, |L|=3)` median | **26.267 ms** |
| `c_capture_hidden(k=16)` (delta) | **-0.107 ms** |
| Ratio (capture / baseline) | 0.9959 |
| Relative cost | -0.41% |

The negative delta is within per-iteration jitter; both medians sit
at the same MLX-dispatch cost. **Capture is effectively free on the
0.8B fixture for `|capture_layer_ids| = 3`.**

### Interpretation

The §1 post-α prediction band ("1.4-2.0× at α ∈ [0.5, 0.7]") was
shifted ~10% lower than the pre-α band to budget for an unknown
`c_capture_hidden(k)`. The 0.8B measurement says that budget is
essentially zero on this fixture for a small capture set, so the
band can revert toward "1.5-2.2× at α ∈ [0.5, 0.7]" pending the
real-target measurement on dense 27B-4bit (η).

Two caveats narrow the inference from 0.8B → 27B:

1. **Bandwidth utilisation.** 0.8B's verify forward is small enough
   that any incremental cost from materialising hidden states is
   absorbed by allocator headroom. The 27B-4bit verify forward at
   k=16 sits closer to bandwidth saturation per the P-6.0.5 Unit 7
   measurement (1.494× at k=4, sublinear past that); a fixed-bytes
   capture overhead may be more visible there. (η)'s real-target
   bench attestation re-measures.
2. **Capture-set size.** 0.8B was probed at |L|=3 (three captured
   layer slices). The dflash drafter's `target_layer_ids` cardinality
   varies per checkpoint; if the upstream `z-lab/Qwen3.5-27B-DFlash`
   trained with |L| ≥ 6, the per-cycle capture write-bandwidth grows
   linearly with |L|. Sub-unit (β)'s drafter-load step reads
   `DFlashDraftModelArgs.target_layer_ids` from the checkpoint
   config; if |L| > 3, (αβ.2)'s MoE row probes the same fixture
   shape on 0.8B at the actual |L| before (η) commits.

### Decision

The §1 prediction band reverts to **1.5-2.2× at α ∈ [0.5, 0.7]**
based on this 0.8B measurement, with a footnote that 27B at large
|L| may shift it down within the same range. Sub-unit (β) opens
unblocked.
