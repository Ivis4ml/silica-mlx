# P-6 Autoresearch — ninth cycle (2026-05-03)

| Field | Value |
| --- | --- |
| Date | 2026-05-03 |
| Branch | `opus` |
| Status | **2 attempts, both honest negatives — empirical wall: M5 Pro register budget for this kernel shape** |
| User authorisation | "please record and display your results in PNG cycle by cycle for visualization and let us continue" |
| Companion docs | cycles 1-8 in `plans/P6_AUTORESEARCH/` |

## TL;DR

Cycle 9 attempted two more optimizations on the cycle-8 v9 best (0.59 ms):

| Variant | Optimization | p50 (ms) | vs v9 |
| --- | --- | ---: | ---: |
| v9 (cycle 8 best) | no #pragma + prefetch | 0.59 | baseline |
| **v12** | cache 32 uint32s per thread | **0.73** | **-19% (REGRESSED)** |
| **v13** | vectorized uint4 loads, cache 16 uint4s per thread | **0.78** | **-25% (REGRESSED)** |

**Both register-caching strategies regressed because they exceeded the Apple Silicon GPU register budget**, causing spills to threadgroup memory or HBM. The empirical conclusion: **v9's structure (no per-thread weight caching beyond scale/bias) is the sweet spot on M5 Pro for this kernel shape.**

The 1.27× gap to mlx (0.59 vs 0.46 ms) cannot be closed via the standard "cache more in registers" optimization that's typical on CUDA. Apple Silicon GPUs have a tighter register budget per thread.

## What happened

### v12 — cache 32 uint32s per thread (REGRESSED -19%)

**Hypothesis:** v9 re-fetches each thread's uint32 across 8 K_TILE iterations within a group (the same uint32 holds 8 nibbles spanning the K-stripe). Caching them in a register array `cached_packed_r0[N_TILES_PER_SG][TILES_PER_GROUP] = [4][8]` should eliminate 7/8 of the redundant fetches.

**Result:** 0.73 ms vs v9 0.59 ms. **Slower by 19%.** The 32-uint32 register array per thread (128 bytes per thread × 32 threads = 4 KB per simdgroup) exceeded the available register file, causing spills back to threadgroup memory or HBM. The "saved fetches" became spill traffic instead.

### v13 — vectorized uint4 loads (REGRESSED -25%)

**Hypothesis:** Use `uint4` (4 contiguous uint32s as a single vector load) — fewer load instructions per nibble. v6 had set up the `uint4*` view but didn't actually consume it. v13 implements the full vectorized fetch + 16-uint4 register cache.

**Result:** 0.78 ms vs v9 0.59 ms. **Slower by 25%.** Same root cause as v12 but worse — the 16 uint4 registers per thread (16 × 16 bytes = 256 bytes per thread) exceeded the budget even more decisively.

### Empirical lesson

Apple Silicon M5 Pro GPU's **register file per thread is tighter than typical CUDA expectations**. Caching strategies that work on NVIDIA hardware (where threads have 64-256 32-bit registers each comfortably) cause spills here. The compiler's choice in v9 (let it allocate registers freely without our manual caching) is empirically the best.

This rules out the "cache uint32 in registers" lever entirely as a single-session optimization. The 1.27× gap to mlx must come from techniques that DON'T add register pressure:

- **Software pipelining** with ping-pong threadgroup buffers (b_tiles_a / b_tiles_b alternating). Doesn't add registers but adds threadgroup memory.
- **Reorder dequant/MMA** to exploit Apple Silicon's instruction scheduling. Compiler already does some of this.
- **Reduce work per output**: process multiple B rows per simdgroup (we already pad B=4 → 8; can't reduce further on 8×8 MMA).

These would each likely yield 5-10% — not the 25% needed to fully close the gap. **Reaching parity with mlx on this naive Metal kernel approach may not be possible without exploiting features Apple keeps internal** (e.g., scheduled MMA pipelines, fused dequant-load instructions, etc.).

## Files added in cycle 9

- `silica/kernels/fused_qmm_simdgroup_v12.py` — cached uint32 register array (regressed)
- `silica/kernels/fused_qmm_simdgroup_v13.py` — vectorized uint4 loads (regressed)

## Ledger rows added cycle 9

- `AR_KERNEL_QMM_V12_UINT32CACHE` (discard, 0.73 ms)
- `AR_KERNEL_QMM_V13_UINT4` (discard, 0.78 ms)

## Aggregated state across cycles 1-9

**13 custom Metal kernels in `silica/kernels/`** (v2 through v13 of QMM + 4 pointwise kernels).

**QMM kernel progression (final):**

```
cycle 3: naive 1-thread/output     0.66 ms
cycle 6: simdgroup_matrix v2       0.97 ms (added MMA but with overhead)
cycle 7: hoist + 4 N/sg → v4       0.83 ms
cycle 8: prefetch + no #pragma → v9   0.59 ms ← BEST
cycle 9: uint32/uint4 caching      0.73-0.78 ms (regressed, register pressure)
mlx ref:                           0.46 ms (1.27× to go)
weights bandwidth floor:           0.16 ms (chip ceiling)
```

**Running best on `qwen3.5-27b-warm-decode-b4`: still 42.17 tok/s.** v9 is too slow to integrate.

## Empirical kernel-tuning summary

Across 13 QMM kernel iterations (cycles 3, 6, 7, 8, 9), the empirical findings:

**What helped (additive gains):**
- Hoist scale/bias loads per group (not per K_TILE): +8% (v3)
- 4 N tiles per simdgroup (share A-tile load): +7% (v4)
- Remove `#pragma unroll` (let compiler allocate): +large (v9)
- Prefetch next group's scale/bias inside K_TILE loop: +large (v9)

**What hurt or was flat:**
- Group-level batch dequant (8 K_TILEs): -45% (v5; threadgroup memory pressure)
- 8 N tiles per simdgroup: -49% (v8; register pressure)
- 2 simdgroups per TG: -7% (v7; reduced occupancy)
- Cache uint32s per thread: -19% (v12; register pressure)
- Vectorized uint4 + cache: -25% (v13; worse register pressure)
- Half barriers via 2 K_TILE batching: ≈ flat (v10)
- simdgroup_barrier swap: ≈ flat (v11)

**Empirical sweet spot:** 4 N tiles per simdgroup, 1 simdgroup per TG, no per-thread caching beyond scale/bias for current and next group, no aggressive `#pragma unroll`.

## Stop conditions

After 13 kernel iterations, the cycle-9 register-pressure findings strongly suggest **the v9 structure is at the empirical optimum for this kernel approach on M5 Pro**. Further sub-1% optimizations may exist but each requires careful empirical testing.

The realistic path to mlx parity is now narrower:
1. **Software pipelining** with ping-pong b_tiles (5-10% potential)
2. **Reordering dequant/MMA via compiler hints** (small)
3. **Larger work per simdgroup via different decomposition** (e.g., process 2 output K-strips at once rather than tile-by-tile) — risky, may regress

If none of these closes the gap, the conclusion is that **mlx's internal QMM uses Apple-internal optimizations not exposed via `mx.fast.metal_kernel`** — possibly compiler features, scheduled MMA pipelines, or shared memory configurations that user-Metal cannot access.

In that case, v9 is the empirical ceiling for user-Metal QMM on this shape, and **integrating it would require accepting the 1.27× slowdown vs mlx** — which is currently NOT net-positive for the production warm-decode path.

## Cycle-by-cycle visualization (updated)

- `plans/P6_AUTORESEARCH_PROGRESS_QMM_KERNEL.png` — now includes v12 and v13 (both red bars showing register-pressure regression)
- `plans/P6_AUTORESEARCH_PROGRESS_CYCLES.png` — 9 cycles of activity, running-best line still flat at 42.17

## Next steps

The realistic remaining options are:
1. **Software pipelining (v14)**: ping-pong threadgroup buffers to overlap dequant with MMA. Most likely to help further.
2. **Accept v9 as the user-Metal optimum**: integrate it as an opt-in path even though slower than mlx, for use-cases where the kernel infrastructure value (correctness gates, custom code paths) outweighs the small slowdown.
3. **Approach the gap from a different angle**: instead of trying to match mlx's QMM on the same shape, try to ELIMINATE the matmul via algorithmic changes (e.g., ZMLX-pattern fused conv1d + gated_delta_update for DeltaNet layers, which is a different lever entirely).

Continuing in this register-pressure direction will likely produce more honest negatives. Pivoting to software pipelining or to a different lever is the path forward.
