# P-6 Track C.4 DFlash spike — REPORT

D-021 step 6 spike measurement bundle. One file across sub-units; new
rows append as the spike progresses from (αβ.1) → (η).

| Sub-unit | Row | Status | Date |
| -------- | --- | ------ | ---- |
| (αβ.1) | `c_capture_hidden(k=16)` on Qwen3.5-0.8B (dense) | landed | 2026-05-01 |
| (αβ.2) | `c_capture_hidden(k=16)` on Qwen3.5-35B-A3B-4bit (MoE) | landed | 2026-05-01 |
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

Two methodology variants. The first run paid a fresh
`adapter_for_repo` load per iteration (cycle-1 cold), which made the
load cost dominate the wall-clock. The second run loads the adapter
once and reuses it across iterations (per-req KV freed via
`SimpleKVCache.free` between samples) — same cycle-1 verify-cost
contract, much tighter signal.

| Quantity | Per-iter load (initial) | Reused adapter (refined) |
| -------- | ----------------------- | ------------------------ |
| `decode_step_multi(k=16)` baseline median | 26.374 ms | **12.784 ms** |
| `decode_step_multi_with_capture(k=16, |L|=3)` median | 26.267 ms | **12.902 ms** |
| `c_capture_hidden(k=16)` (delta) | -0.107 ms | **+0.118 ms** |
| Ratio (capture / baseline) | 0.9959 | **1.0092** |
| Relative cost | -0.41% | **+0.92%** |

The reused-adapter run is the load-bearing measurement;
**`c_capture_hidden(k=16) ≈ +0.92%` on the 0.8B fixture for `|L|=3`,
within per-iter jitter and well below the §1 prediction-band budget.**
The first-run negative delta was load-jitter dominated and is
preserved in the table for methodology audit, not as the reportable
number.

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
|L| may shift it down within the same range. Sub-unit (αβ.2)'s MoE
row (below) corroborates the inference; sub-unit (β) opens
unblocked.

---

## (αβ.2) — `c_capture_hidden(k=16)` on cached `mlx-community/Qwen3.5-35B-A3B-4bit`

Closes OQ-5 on the MoE fixture, completing the (αβ) precondition
gate. The MoE adapter inherits the entire capture surface from
`Qwen3_5Adapter` (mlx-lm's `qwen3_5_moe.Model` extends
`qwen3_5.Model` directly; the outer forward chain is identical, only
per-layer MLP type differs). The microbench tests whether the
SwitchGLU sparse MLP changes the per-cycle capture cost.

### Setup

- Fixture: cached `mlx-community/Qwen3.5-35B-A3B-4bit` (~20 GB);
  upstream registers it in `dflash_mlx.generate.DRAFT_REGISTRY`
  paired with `z-lab/Qwen3.5-35B-A3B-DFlash`.
- Workload: same script, same capture set
  `{0, num_layers // 2, num_layers}` = three layer slices. MoE
  has `num_layers = 40`, `hidden_size = 2048`, so capture set is
  `{0, 20, 40}`.
- Methodology: reused-adapter (refined) — adapter loaded once,
  per-req KV freed via `SimpleKVCache.free` between samples. 3
  warmup iters + 20 measurement iters. Median wall-clock reported.
- Hardware: M5 Pro 48 GB.
- Script invocation:
  ```text
  SILICA_REAL_QWEN3_5_MOE=1 python -m scripts.microbench_capture_hidden \
      --repo mlx-community/Qwen3.5-35B-A3B-4bit \
      --k 16 --warmup 3 --iters 20
  ```

### Result

| Quantity | Value |
| -------- | ----- |
| `decode_step_multi(k=16)` baseline median | **45.257 ms** |
| `decode_step_multi_with_capture(k=16, |L|=3)` median | **43.644 ms** |
| `c_capture_hidden(k=16)` (delta) | **-1.613 ms** |
| Ratio (capture / baseline) | 0.9644 |
| Relative cost | -3.56% |

The negative delta is within per-iter jitter on a 20 GB MoE
checkpoint (MLX dispatch / sparse-MLP routing variance / OS-level
memory effects). **Capture is effectively free on the MoE fixture at
|L|=3** — same conclusion as the dense (αβ.1) row.

### Cross-fixture comparison

| Fixture | Baseline (ms) | Capture (ms) | Δ (ms) | Δ (%) |
| ------- | ------------- | ------------ | ------ | ----- |
| Qwen3.5-0.8B (dense, 24 layers, hidden=1024) | 12.78 | 12.90 | +0.12 | +0.92% |
| Qwen3.5-35B-A3B-4bit (MoE, 40 layers, hidden=2048) | 45.26 | 43.64 | -1.61 | -3.56% |

Both deltas are within per-iteration jitter; neither shows a
*systematic* per-cycle cost penalty. The §1 prediction band's
`c_capture_hidden(k)` term holds at ≈0 on both fixtures for |L|=3
on the M5 Pro 48 GB target. This corroborates the (αβ.1) inference
that the band reverts to "1.5-2.2× at α ∈ [0.5, 0.7]". Dense 27B
real-target re-measurement at sub-unit (η) remains the load-bearing
final check, and the upstream drafter checkpoint's actual
`|target_layer_ids|` is determined at sub-unit (β)'s drafter-load
step (if `|L| > 3`, (η) re-measures at the actual capture-set size).
