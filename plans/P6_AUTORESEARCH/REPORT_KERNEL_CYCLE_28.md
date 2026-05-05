# P-6 Autoresearch — twenty-eighth cycle (2026-05-04) — B=64 hardware ceiling re-measured under corrected v10 path

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Hardware ceiling at B=64 = **231.9 ± 0.3 tok/s with bf16 state alone**. v10 contribution at B=64 is **-1.7 tok/s** (marginally negative, dispatch overhead exceeds kernel-level savings). Combined with cycle 27, v10 has zero or slightly negative E2E contribution at production B regardless of T_kv shape. |
| User authorization | "good let us continue" — proceeding per cycle-27 + Codex recommendation |
| Companion docs | cycle 27 correction report; Codex c25/c26 reports |

## TL;DR

3 reps each at B=64 with v10 actually firing (post-merge bf16 fix):

| Configuration | Run 1 | Run 2 | Run 3 | Mean ± std |
| --- | ---: | ---: | ---: | ---: |
| bf16+v10 (v10 firing) | 228.4 | 230.8 | 231.5 | **230.2 ± 1.6** |
| bf16-only | 231.6 | 232.1 | 231.9 | **231.9 ± 0.3** |

**Δ between v10-firing and bf16-only**: -1.7 tok/s.
Pooled σ ≈ 1.63. Δ/σ ≈ 1.04.

**v10 marginally HURTS at B=64.** The bf16-only path is cleaner.

## Combined v10 attribution picture (cycles 27 + 28)

| B | bf16 only (n=3) | bf16+v10 firing (n=3-5) | Δ |
| ---: | ---: | ---: | ---: |
| 52 | 204.2 ± 1.1 | 204.7 ± 1.2 (n=5) | +0.5 (within noise) |
| **64** | **231.9 ± 0.3** | 230.2 ± 1.6 | **-1.7 (marginal regression)** |

**v10 FA-decode kernel at production B regime: zero or slightly negative
E2E contribution.** Even though v10 microbench shows 1.28-2.14× over mlx
SDPA at the kernel level, the gain doesn't translate at B=52/64 because:

1. Attention is ~22% of step time at production B
2. Dispatch overhead from monkey-patched shadow_install Python wrapper
   adds per-call cost
3. Per-step time is dominated by DeltaNet (74%) + scheduler + allocator

At B=64 specifically, v10's dispatch overhead net-out exceeds its
kernel-level savings.

## Honest revised running-best line (cycles 27 + 28 combined)

| Frame | Value | Lever attribution |
| --- | ---: | --- |
| Within 36 GB envelope | **204.5 ± ~1.5 tok/s at B=52 bf16-only** | C10 axis-shift × C12 bf16-state peak-save |
| Hardware ceiling (B=64) | **231.9 ± 0.3 tok/s bf16-only** | same; v10 not load-bearing |
| (1b) ≥60 milestone | CLEARED 3.41× (envelope) / 3.86× (ceiling) | — |

The cycle-14 reported numbers (206.2 / 232.2) are within noise of these
revised numbers — the throughput was approximately right but the
attribution to "v10+bf16 stack" was wrong. **The system has been doing
~204 / ~232 tok/s at B=52 / B=64 since cycle 13's bf16-state work; v10
was never the lever.**

## Why this matters

The **v10 FA-decode kernel** stays in the inventory as a
correctness-validated implementation with verified microbench wins
(1.28-2.14× over mlx SDPA across T_kv ∈ {128, 256, 512, 1024}). It is
NOT load-bearing for the running-best line at production B, but remains
useful for:

- Future workloads where attention is a larger fraction of step time
  (e.g., long-context decode beyond T_kv=1024, or models with different
  compute-vs-memory profile)
- The **fused gated-output epilogue** is a uniquely Silica contribution
  (no public Apple-Silicon kernel ships sigmoid(gate) * SDPA fusion);
  remains valuable as a reference implementation

## What cycle 28 closes definitively

The "v10+bf16 stack" composition the cycle-14 commit message named is
not a real compositional lever in production decode at B=52/64. The
real composition is just **C10 axis-shift × C12 bf16-state-peak-save** =
4.85× cycle-1 baseline at envelope, 5.50× at hardware ceiling.

cycle 12 bf16 state remains the load-bearing kernel-level intervention,
because its peak-memory save (3.5 GB) is what enabled cycle 13's B-axis
extension to B=52..64.

## Files added in cycle 28

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_28.md` — this file
- `/tmp/c28_b64_v10_run{1,2,3}.jsonl` — bf16+v10 reverify artefacts
- `/tmp/c28_b64_bf16only_run{1,2,3}.jsonl` — bf16-only baseline artefacts

## Ledger row added cycle 28

- `AR_C28_B64_V10_REGRESSION` (diagnostic, KEEP-revision) — B=64 bf16+v10
  = 230.2 ± 1.6 (n=3); B=64 bf16-only = 231.9 ± 0.3 (n=3). Δ = -1.7
  tok/s. v10 marginally HURTS at B=64 (dispatch overhead > kernel
  savings). Combined with cycle 27 (B=52 v10 = +0.5 within noise),
  v10 has NO measurable positive E2E contribution at production B
  in {52, 64}. Hardware ceiling correctly attributed to **bf16-only
  at B=64 = 231.9 ± 0.3 tok/s** (5.50× cycle-1 baseline).

## What's next (cycle 29+)

Per cycle 27 plan + Codex recommendation:

1. **Cycle 29 — between-session variance characterization** at B=52.
   Codex measured 200.1 ± 6.0 in conda env vs my consistent ~205 in uv.
   Need to understand whether the difference is environmental or
   stack-specific; may need cool-down protocols documented.

2. **Cycle 30 — 40 GB cliff probe** using Codex's `scripts/bench_with_mlx_limits.py`.
   Test whether `mx.metal.set_cache_limit` can move the empirical 40 GB
   peak cliff (cycle 13 found B=66 = 166 tok/s, B=68 = 169, B=72 = 173).
   If cliff moves to 44+ GB, B=68-72 territory unlocks ~250+ tok/s.

3. **Cycle 31+ — composition probe** with the corrected v10
   understanding. Now that we know v10 doesn't contribute at high B,
   look for workloads (longer T_kv, higher attn fraction) where the
   kernel-level wins translate.

## Stop conditions per AR.md (status check)

1. ≥60 tok/s on ≥2 runs — CLEARED 3.41× / 3.86×
2. New running-best ≥3σ above 42.17 with clean attribution — **CLEARED
   at 4.85× / 5.50× under corrected attribution to C10 × C12 (NOT v10)**
3. Open-lever set cannot reach 60 — N/A (already cleared)

The autoresearch loop's deliverable, with cycle 27+28 corrections:
- Within 36 GB envelope: 204.5 ± ~1.5 tok/s at B=52 bf16
- Within 48 GB hardware ceiling: 231.9 ± 0.3 tok/s at B=64 bf16
