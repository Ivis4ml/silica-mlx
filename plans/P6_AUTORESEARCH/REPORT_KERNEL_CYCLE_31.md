# P-6 Autoresearch — thirty-first cycle (2026-05-04) — vectorized DeltaNet kernel hits parity with mlx (no headroom)

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Diagnostic — wrote `silica/kernels/gated_delta_v2.py` applying v6→v7 half4/bfloat4 vectorization to mlx's gated_delta kernel. **Microbench shows parity (0.985-1.25× across B sweep, mostly noise).** mlx's gated_delta is already at near-bandwidth limit on state R/W; vectorization doesn't unlock headroom. |
| User authorization | continued |
| Companion docs | cycle 30 (B=64 decomposition); cycle 11 (FA-decode v6→v7 trick that motivated this) |

## TL;DR

Vectorized state load + bfloat4 K/V/Q reads in DeltaNet recurrent step:

| B | err vs ops ref | mlx p50 (ms) | v2 p50 (ms) | v2/mlx ratio |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.00e+00 | 0.422 | 0.337 | **1.25×** |
| 16 | 3.05e-05 | 0.366 | 0.372 | 0.985× |
| 52 | 1.53e-05 | 0.839 | 0.828 | 1.013× |
| 64 | 3.05e-05 | 0.965 | 0.964 | 1.001× |

Correctness PASS at all B (errors match mlx's 3e-5 fp16 ULP).

The B=4 1.25× win is plausibly launch-overhead-limited (small workload,
fewer kernel cycles to amortise dispatch). At B=16/52/64 the workload
is large enough that kernel time dominates and the speedup vanishes.

## Why vectorization doesn't help here (unlike FA-decode)

The cycle-11 FA v6→v7 win came from:
1. K/V tile loads are HBM-bandwidth-bound at production B
2. Scalar half loads have lower issue rate than half4
3. Vectorization improved load throughput by 1.4-1.7×

DeltaNet's bottleneck is different:
1. State R/W (288 MB R+W per layer at fp32, 144 MB at bf16) dominates
2. Apple Silicon HBM coalesces well for 32 threads × 8 bytes (= 256 B
   transactions) — already near peak for the bf16 state path
3. Vectorization changes load instruction count but NOT data volume,
   and the load instructions are not the throughput bottleneck

mlx's kernel uses `n_per_t = Dk/32 = 4` per-thread state, with 32
threads in a simdgroup. The 32×4 = 128-element state fits one Dk row
exactly. The simdgroup reduction (`simd_sum`) is the synchronization
point, not the load.

## Implication for the autoresearch loop

The cycle 30 decomposition identified DeltaNet as the only remaining
high-leverage component at B=64. Cycle 31 confirms there's no easy
half4-style headroom in mlx's existing kernel.

**Other angles for DeltaNet that might work** (not pursued in cycle 31):

1. **Reduce state size** — fp16 state (vs bf16) or even smaller storage
   types. Risk: numerical precision over many decode steps.
2. **Algorithmic restructure** — keep partial state in TG memory instead
   of HBM round-trip per step. Substantial rewrite.
3. **Merge consecutive layers** — recurrent state passed without HBM
   serialization. Massive structural change.
4. **`mx.compile` graph trace** — fuse adjacent ops in the DeltaNet
   forward (already partially done via `compute_g` @mx.compile in
   gated_delta.py). Marginal additional gain expected.

None of these are quick probes. All require multi-day commitment with
uncertain payoff.

## What this closes

Cycles 30+31 close the question "is there a DeltaNet kernel-level lever
at production B?" with **no, mlx's existing kernel is at near-bandwidth
limit**. The cycle 27-29 corrected running-best line stands:

- Within strict 36 GB envelope: 204.5 ± ~1.5 tok/s at B=52
- Within 48 GB hardware ceiling: 231.9 ± 0.3 tok/s at B=64

The autoresearch loop's deliverable is genuinely closed on this
hardware/model. Further progress requires:
- mlx 0.32+ async-copy primitives (external)
- Algorithmic / structural model changes (out of P-6 scope)
- Different model architecture (orthogonal to dense-27B mission)

## Files added in cycle 31

- `silica/kernels/gated_delta_v2.py` — vectorized DeltaNet kernel
  (kept for reference; correctness PASS, performance at parity with
  mlx; not load-bearing)
- `scripts/bench_gated_delta_v2.py` — microbench harness
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_31.md` — this file

## Ledger row added cycle 31

- `AR_C31_DELTANET_V2_PARITY` (diagnostic — kernel parity with mlx, no
  headroom) — vectorized silica DeltaNet at production shape (Hk=16,
  Hv=48, Dk=Dv=128, T=1) microbench: B=4→1.25× / B=16→0.985× /
  B=52→1.013× / B=64→1.001× vs mlx's gated_delta kernel. Correctness
  PASS at all B (err 3e-5 fp16 ULP, matching mlx). The half4/bfloat4
  vectorization that gave 1.4-1.7× on FA-decode doesn't transfer to
  DeltaNet because mlx's existing kernel is already at near-HBM-
  bandwidth limit on state R/W. Cycle 30's identification of DeltaNet
  as the dominant cost at B=64 (87.9% share) does NOT translate to a
  reachable kernel lever — the lever is already pulled by mlx itself.
