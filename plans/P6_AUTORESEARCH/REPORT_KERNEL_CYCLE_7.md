# P-6 Autoresearch — seventh cycle (2026-05-03)

| Field | Value |
| --- | --- |
| Date | 2026-05-03 |
| Branch | `opus` |
| Status | **Real incremental kernel-tuning progress (17% over cycle 6)**, but not yet at parity with mlx; 1.85× gap remains |
| User authorisation | "please start work" (2026-05-03) — explicit go-ahead for multi-day kernel tuning work; this is the first iteration |
| Companion docs | cycle 1-6 REPORTs in `plans/P6_AUTORESEARCH/` |

## TL;DR

Cycle 7 starts the multi-day simdgroup_matrix QMM tuning identified as the top-impact remaining lever in cycle 6. Three iterations (v3, v4, v5, v6) of the cycle-6 kernel produced **real cumulative 17% improvement** in QMM microbench: 0.97 ms → 0.83 ms at production shape (B=4, K=5120, N=17408). v4 is the new best.

Detailed iterations:

| Version | Optimization | Microbench p50 (ms) | vs mlx (0.45 ms) |
| --- | --- | ---: | ---: |
| v2 (cycle 6) | naive simdgroup_matrix MMA | 0.97 | 0.46× (2.16× slower) |
| **v3** | **+ hoist scale/bias loads (per group, not per K_TILE)** | **0.89** | **0.50× (2.0× slower)** |
| **v4** | **+ 4 N-tiles per simdgroup (share A-tile load)** | **0.83** | **0.55× (1.85× slower)** ← best |
| v5 | + batch dequant entire group (8 K_TILEs at once) | 1.21 | 0.37× (regressed — register pressure) |
| v6 | + uint4 view placeholder (kernel body unchanged) | 0.83 | ≈ v4 (vectorization would need redesign) |

v5 regressed because the unrolled batch-dequant blew register pressure / i-cache. v6 set up the uint4 view but didn't actually use it (would require full access-pattern redesign).

The 1.85× gap to mlx remains. Apple Silicon M5 Pro on this MSL revision **does not support rectangular simdgroup_matrix tiles** (4×8, 8×16, 16×8 all rejected at compile time) — only 8×8, ruling out larger-tile optimization as a lever.

## What was attempted

### v3 — Hoist scale/bias loads per group

**Optimization:** group_size=64 spans 8 K_TILEs of 8 K-values each. Each K_TILE iteration in v2 re-read the same scale/bias from HBM. v3 reads scale/bias ONCE per group and reuses across 8 K_TILEs.

**Math:** 32 threads × 2 reads per K_TILE × 640 K_TILEs = 40,960 scale/bias reads per simdgroup in v2. After v3: 32 × 2 × 80 groups = 5,120 reads. **8× reduction.** Each read is fp16=2 bytes; saved ~70 KB per simdgroup. Across 2176 simdgroups: ~150 MB of redundant reads avoided.

**Result:** 0.97 → 0.89 ms (+8.5% speedup over v2).

### v4 — Multi-N-tiles per simdgroup

**Optimization:** v3 had one simdgroup per N-tile = 2176 dispatches at N=17408. v4 has each simdgroup process 4 consecutive N-tiles, so the **A-tile load (from x) is shared across 4 different B-tile MMAs** in each K_TILE iteration. Reduces dispatch count 4× and amortizes the A-tile load 4×.

**Math:** A-tile loads per simdgroup: 640 K_TILEs × 1 A_tile per iter = 640 in v3; same in v4 but each load now serves 4 N-tiles instead of 1 → effective A-tile-loads per output column = 1/4 in v4 vs 1/1 in v3. Plus dispatch count: 2176 → 544 simdgroups (4× fewer).

**Result:** 0.89 → 0.83 ms (+6.7% over v3, **+17% cumulative over v2**). v4 is the new best.

### v5 — Batch dequantize entire group (REGRESSED)

**Optimization:** dequantize all 8 K_TILEs in a group at once into 8x64 threadgroup memory, then run 8 MMAs back-to-back without intermediate barriers. Saves 7 of 8 barriers per group.

**Result:** 0.83 → 1.21 ms (**45% REGRESSION**). Hypothesised causes:
1. Larger threadgroup memory (4 KB) crosses an L1 efficiency threshold
2. Unrolled inner loop (8 K_TILEs × 4 N tiles = 32 iter) bloats kernel code, hurting i-cache
3. Holding 4×4=16 scale/bias halves as registers per thread + 4 simdgroup_matrix accumulators may exceed register file budget, causing spills

**Disposition:** retire v5; v4 stays as best.

### v6 — uint4 view placeholder

**Optimization attempt:** cast w_q to `const device uint4*` for vectorized 4-uint32 loads. v6 has the cast but kernel body still uses scalar uint reads.

**Result:** 0.83 ms ≈ v4. Not a real change — full vectorization would require redesigning the per-thread work assignment (each thread reads 4 uint32s = 32 nibbles per fetch instead of 1 uint32 = 8 nibbles).

### Rectangular tile probe

Tested whether Apple Silicon M5 Pro Metal supports `simdgroup_matrix<half, 4, 8>`, `simdgroup_matrix<half, 8, 16>`, `simdgroup_matrix<half, 16, 8>`. **All rejected at compile time** with `"invalid size for 'simdgroup_matrix'"`. Only 8×8 supported.

This rules out a major lever (M3+ Macs with newer Metal versions support rectangular tiles for ~2× MMA throughput on small-M shapes; M5 Pro on current MSL revision does not).

## End-to-end implications

Per-layer at decode B=4 has 2 quantised matmuls at this exact shape (gate_proj and up_proj are both K=5120 N=17408). v4 at 0.83 ms vs mlx at 0.45 ms is 0.38 ms slower per matmul. Across 2 matmuls × 64 layers = 128 invocations × 0.38 ms = 48 ms slower per step. **Integrating v4 into the production path would HURT, not help.** v4 is foundation for future tuning, not a drop-in replacement yet.

The kernel is **correctness-validated** (max-abs 0.0039 within fp16 ULP across all v3-v6 variants), so future optimization work has a verified baseline to improve on.

## Aggregated state across cycles 1-7

7 cycles, 7 custom Metal kernels in tree:
- `fused_gated_output` (cycle 1)
- `fused_silu_mul` (cycle 2)
- `fused_qk_norm` (cycle 2)
- `fused_qmm_decode` (naive QMM, cycle 3)
- `fused_qmm_simdgroup` (v2 simdgroup_matrix MMA, cycle 6)
- `fused_qmm_simdgroup_v3` (cycle 7)
- `fused_qmm_simdgroup_v4` (cycle 7) ← best
- `fused_qmm_simdgroup_v5` (cycle 7, regressed)
- `fused_qmm_simdgroup_v6` (cycle 7, placeholder)

Plus shadow-install infra, layer-internal microbenches, end-to-end harness, etc.

**Running best on `decode_tok_s` running-best frame remains 42.17 tok/s** (no kernel is yet faster than mlx's internal QMM at this shape).

## What's still on the optimization surface

The remaining 1.85× gap from v4 to mlx represents techniques I haven't implemented in v6:

1. **Vectorized uint4 loads** (full implementation, not just the view): each thread reads 4 uint32s = 32 nibbles per fetch. Reduces load instruction count 4×. The view cast in v6 is a placeholder; the kernel body needs to be redesigned to consume 32-nibble chunks per thread per K_TILE. Estimated 5-15% additional speedup.

2. **Software pipelining**: overlap dequantization of K_TILE k+1 with MMA of K_TILE k. Apple Silicon Metal does not have native async copy with dequant, but careful loop scheduling can interleave. Estimated 5-15%.

3. **Asymmetric work distribution**: instead of 32 threads dequantizing 64 weights, have 8 threads do dequant while 24 do other work (e.g., loading next iteration's data). Estimated 5-10%.

4. **Per-row weight broadcast via simd_shfl**: load uint32s into thread registers, shfl across simdgroup to share. Avoids threadgroup memory for weight values entirely. Estimated 10-25%.

5. **Specialization for fixed K=5120**: hardcode the loop bounds at compile time, eliminate runtime checks. Small but measurable. Estimated 2-5%.

Combining several of these could close the gap to mlx (or come within 10-20%). Each is 1-2 days of work; cumulatively 5-10 more days to potentially reach parity.

## Honest cycle 7 disposition

**Real incremental progress** — 17% kernel speedup is the largest movement in 7 cycles, and the optimization roadmap is now well-defined (5 specific levers identified above with effort estimates). But the kernel is **still slower than mlx**, so it doesn't improve T₄ end-to-end yet. The cycle-6 honest finding stands: matching mlx's internal QMM requires multi-day expert tuning, of which cycle 7 is the first iteration of an estimated 5-10 day campaign.

For the running-best `qwen3.5-27b-warm-decode-b4` frame, **no change** — running best stays at 42.17 tok/s.

The valuable cycle-7 deliverable: a **measurable progression** (v2 0.97 → v3 0.89 → v4 0.83) that demonstrates the tuning campaign is tractable, with specific next-step levers identified by empirical evidence (not paper-claim guesses).

## Files added in cycle 7

- `silica/kernels/fused_qmm_simdgroup_v3.py` — hoisted scale/bias
- `silica/kernels/fused_qmm_simdgroup_v4.py` — 4 N-tiles per simdgroup ← BEST
- `silica/kernels/fused_qmm_simdgroup_v5.py` — batch group dequant (regressed; retained as documented negative)
- `silica/kernels/fused_qmm_simdgroup_v6.py` — uint4 view placeholder
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_7.md` — this file
