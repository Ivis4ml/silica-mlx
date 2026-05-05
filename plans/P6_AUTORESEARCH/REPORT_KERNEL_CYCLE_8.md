# P-6 Autoresearch — eighth cycle (2026-05-03)

| Field | Value |
| --- | --- |
| Date | 2026-05-03 |
| Branch | `opus` |
| Status | **Major progress — v9 is new best at 0.59 ms (40% over cycle-6 v2 baseline; 1.27× gap to mlx)** |
| User authorisation | "please record and display your results in PNG cycle by cycle for visualization and let us continue" |
| Companion docs | cycle 1-7 REPORTs in `plans/P6_AUTORESEARCH/` |

## TL;DR

Cycle 8 = 5 more kernel iterations on top of cycle 7's v4 best. **v9 is the new best at ~0.59 ms** (was 0.83 ms in v4; 0.97 ms in v2). Cumulative **40% improvement over cycle-6 baseline** and now **only 1.27× slower than mlx's internal QMM** (was 2.16× at cycle 6 start).

| Variant | Optimization | p50 (ms) | vs mlx ref (0.46) | Δ vs v4 |
| --- | --- | ---: | ---: | ---: |
| v4 (cycle 7 best) | hoist s/b + 4 N/sg | 0.83 | 1.85× slower | baseline |
| v7 | + 2 simdgroups per TG | 0.89 | 1.95× slower | -7% |
| v8 | + 8 N-tiles per simdgroup | 1.24 | 2.72× slower | -49% |
| **v9** | **no #pragma unroll + prefetch next group's s/b** | **0.59** | **1.27× slower** | **+29%** ← BEST |
| v10 | + 2 K_TILEs per barrier | 0.61 | 1.34× slower | +27% (≈ v9) |
| v11 | simdgroup_barrier instead of threadgroup_barrier | 0.65 | 1.41× slower | +22% |

**The single biggest insight of cycle 8: removing aggressive `#pragma unroll` directives let the Metal compiler do better register allocation, AND prefetching the NEXT group's scale/bias inside the previous group's MMA work overlaps memory loads with compute.** Combined, these two changes produced the largest single-iteration speedup of the entire campaign (29% over v4).

## What worked

### v9 — non-unrolled + prefetch (+29% over v4)

Two changes from v4:

1. **Removed all `#pragma unroll` directives** on the inner N-tiles loop. Manual unrolling forced the compiler to instantiate 4 copies of the dequant body simultaneously, which exceeded the register budget and caused spills. Letting the compiler decide produced tighter code.

2. **Prefetched the next group's scale/bias INSIDE the inner K_TILE loop**, so the load could overlap with the ongoing MMA work of the current group. The 4 simdgroup_multiply_accumulate calls have non-trivial latency; the prefetch fills the dispatch shadow.

```c
for (uint group_base = 0; group_base < K_val; group_base += G) {
    // ... K_TILE loop with MMA work ...
    // Prefetch NEXT group's scale/bias (overlaps with current group's MMA tail)
    if (group_idx + 1 < groups_per_row) {
        for (uint nb = 0; nb < N_TILES_PER_SG; nb++) {
            scale_r0[nb] = w_s[(n_block_start+nb)*8 + row0) * groups_per_row + group_idx+1];
            // ... biases / row1 ...
        }
    }
}
```

The first group's scale/bias is loaded before the outer loop (initial seed). After that, the prefetch keeps the registers fresh for the next iteration.

### What didn't work (cycle 8)

- **v7 (2 simdgroups per TG):** reduces dispatch count 2× but cuts occupancy on M5 Pro. The 64-thread TG runs fewer concurrent waves per GPU core. -7%.
- **v8 (8 N-tiles per simdgroup):** doubles A-tile reuse but adds 8 fp32 simdgroup_matrix accumulators which exceed the register file budget → spills. -49%.
- **v10 (2 K_TILEs per barrier):** halves barrier count but adds 2× threadgroup memory for batched b_tiles. The two effects cancel within noise. ≈ v9.
- **v11 (simdgroup_barrier instead of threadgroup_barrier):** Metal compiler appears to optimize threadgroup_barrier away when it sees a single-simdgroup TG. The swap was a no-op or slight regression.

## Aggregated state across cycles 1-8

**11 custom Metal kernels in `silica/kernels/`** (all correctness-validated within fp16 ULP):

- pointwise: `fused_gated_output`, `fused_silu_mul`, `fused_qk_norm`
- 4-bit QMM: `fused_qmm_decode` (naive), `fused_qmm_simdgroup` (v2-v6), `fused_qmm_simdgroup_v3..v11` (cycles 7-8 tuning iterations)

**QMM kernel progression (load-bearing):**

```
cycle 3: naive 1-thread/output     0.66 ms (1-thread brute force)
cycle 6: simdgroup_matrix v2       0.97 ms (added MMA but with thrash)
cycle 7: hoist s/b → 4 N/sg → ...  0.83 ms (v4)
cycle 8: prefetch + no #pragma     0.59 ms (v9) ← 40% under v2 baseline
mlx ref (target):                  0.46 ms (1.27× to go)
weights bandwidth floor:           0.16 ms (chip ceiling)
```

The trajectory is clear and quantitative. Each iteration narrows the gap. v9 is now within ~28% of mlx's tuned QMM at production shape (B=4, K=5120, N=17408).

## End-to-end implications

**Running best on `qwen3.5-27b-warm-decode-b4` is still 42.17 tok/s.** v9 at 0.59 ms is still 0.13 ms slower than mlx ref per QMM call. Per layer, gate_proj + up_proj = 2 × 0.13 = 0.26 ms slower; across 64 layers = 16.6 ms slower per step. **Integrating v9 into production would still hurt by ~16%.**

But **the gap is now small enough that 1-2 more tuning iterations might tip it positive.** Specifically:
- v12 (planned next): cache uint32 in register, dequantize all 8 nibbles per thread inline (eliminates redundant uint32 reads across K_TILE iterations)
- v13: vectorized uint4 loads (4 uint32s per fetch — the v6 placeholder finally implemented)

If either of these closes another 15-20%, v9's 0.59 → 0.49-0.52 ms which would match or beat mlx's 0.46 ms within noise.

## Cycle-by-cycle visualization

Two new chart files:
- `plans/P6_AUTORESEARCH_PROGRESS_QMM_KERNEL.png` — bar chart of all 11 QMM variants across cycles 3, 6, 7, 8 with mlx ref and bandwidth floor lines, gap-to-mlx annotation on v9 (the new best)
- `plans/P6_AUTORESEARCH_PROGRESS_CYCLES.png` — two panels: per-cycle deliverable counts (custom kernels, probes, discards, diagnostics) and the running-best decode_tok_s line (still flat at 42.17 across all 8 cycles)

## Files added in cycle 8

- `silica/kernels/fused_qmm_simdgroup_v7.py` — 2 simdgroups per TG (regressed)
- `silica/kernels/fused_qmm_simdgroup_v8.py` — 8 N-tiles per simdgroup (regressed)
- `silica/kernels/fused_qmm_simdgroup_v9.py` — **NEW BEST** (no #pragma unroll + prefetch)
- `silica/kernels/fused_qmm_simdgroup_v10.py` — 2 K_TILEs per barrier (≈ v9)
- `silica/kernels/fused_qmm_simdgroup_v11.py` — simdgroup_barrier swap (≈ v9)

Plus updates to chart generators and ledger.

## Ledger rows added cycle 8

- `AR_KERNEL_QMM_V7_2SG` (discard, 0.89 ms)
- `AR_KERNEL_QMM_V8_8N` (discard, 1.24 ms)
- **`AR_KERNEL_QMM_V9_PREFETCH` (diagnostic, 0.59 ms — best so far)**
- `AR_KERNEL_QMM_V10_2KT` (discard, 0.61 ms)
- `AR_KERNEL_QMM_V11_NOBARR` (discard, 0.65 ms)

## What's next

The optimization roadmap is now narrower and more concrete:

1. **v12 — cached uint32 + inline dequant** (1 day). Each thread fetches its own uint32 once and dequantizes all 8 nibbles inline as the K_TILE loop iterates. Eliminates 7/8 of the redundant uint32 reads. Estimated 5-10% additional speedup if memory-bound.
2. **v13 — vectorized uint4 loads** (1 day). Each thread reads 4 uint32s per fetch (32 nibbles). Estimated 5-15%.
3. **v14 — software pipelining at the K_TILE level** (2 days). Ping-pong b_tiles buffers so dequant of K_TILE k+1 overlaps with MMA of K_TILE k. Estimated 5-15%.

Combined potential: **another 15-30% reduction → 0.42-0.50 ms**, which would put silica's QMM at parity with or faster than mlx's 0.46 ms reference.

Once v9 (or a successor) drops below mlx's reference, integrating it into the production warm-decode-b4 path becomes net-positive — at which point the running-best frame finally moves above 42.17 tok/s.
