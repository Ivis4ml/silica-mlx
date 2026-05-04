# P-6 Autoresearch — nineteenth cycle (2026-05-04) — spec-decode research loop opens; extended coverage probe

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | **Coverage@64 = 0.4051 crosses the user's 40% accept-rate threshold.** Loop opens to push effective accept rate from ~9% (β.1 baseline) to ≥40% via b=64 tree-spec, distillation drafters, or quantization-aware training. |
| User authorisation | "we need do research to achieve at least > 40% accept ratio... Let us loop the research" — explicit auth to pursue C.5 γ.1 line + drafter research + downstream variants |
| Companion docs | `plans/P6_C5_DDTREE/REPORT.md` (β.1+β.2 baseline); `plans/P6_C4_DFLASH/REPORT.md` (DFlash 2B retired); `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_{17,18}.md` (kernel-side optima) |

## TL;DR

The previous β.2 probe truncated b at 32. **Extending to b ∈ {1, 4, 8, 16,
32, 64, 128, 256, 512, 1000} reveals the rank distribution continues
climbing past b=32**:

```
@1      0.0626  ###
@4      0.1429  #######
@8      0.1977  #########
@16     0.2583  ############
@32     0.3405  #################
@64     0.4051  ####################  ← crosses 40%
@128    0.4990  ########################
@256    0.6047  ##############################
@512    0.7025  ###################################
@1000   0.7671  ######################################
```

Rank statistics: p25 = 15, median = 129, p75 = 848. The drafter is
"directionally right within top-100 most of the time, but the heavy tail
beyond top-1000 captures 23.3% of positions" — meaning **tree-spec at
b=64 can in principle hit 40% accept rate, but a quarter of positions
remain stubbornly off-distribution regardless of b**.

## Re-reading the prior research

The C.5 OPENING decision matrix gated `b ∈ {4, 8, 16}` because tree-verify
cost scaling at T=32 was "not in scope of P-6.0.5 Unit 7" without
sub-linear-cost attestation. The β.2 probe stopped at b=32 producing
coverage=0.341. Cycle 19 lifts that gate (per user research-loop
authorisation) and finds:

- **b=64: 40.5% coverage** — first b that crosses 40%
- **b=128: 49.9% coverage**
- **b=256: 60.5% coverage**

The structural ceiling (where the curve plateaus near 0.77 at b=1000) is
~77%; there's a permanent 23% "off-distribution" tail no tree size will
recover.

## Research-loop questions opened by the extended probe

1. **Verify-cost scaling**: at b=64 tree-spec, is target-side verify cost
   sub-linear in b on M5 Pro? If verify cost is O(b^0.5) or O(log b), the
   2.4× more accepted tokens per draft sequence (40.5/16.7) translates to
   net throughput. If O(b), it's a wash or net-negative.
2. **Distillation drafter**: can we train a drafter on the 4-bit target's
   own output distribution to push coverage@1 from 6.3% upward? The fact
   that 23% of positions are beyond top-1000 of the bf16 0.8B Qwen3.5
   suggests the 4-bit-target's argmax has shifted enough that no off-the-
   shelf drafter captures it.
3. **Multi-drafter ensemble**: would running 2 drafters and accepting if
   either matches push effective coverage above 0.41 at b=4?
4. **MTP head trained against the 4-bit target**: cycle-1 found Qwen3.5
   has no shipped MTP weights, but a small MTP head trained on the 4-bit
   target's actual outputs could be near-optimal.

## Loop plan (cycle 20 onwards)

The research loop will iterate through (in priority order):

| Cycle | Probe | Question | Next-step gate |
| --- | --- | --- | --- |
| 20 | tree-verify cost microbench at k=4/8/16/32/64 | Is verify cost sub-linear in b? | If sub-linear → cycle 21; if linear → de-prioritize tree-spec |
| 21 | DDTree implementation at b=64 (if 20 passes) | Does b=64 tree-spec hit ≥40% accept on free-running corpus? | If yes → write up; if no → analyze gap |
| 22 | Drafter survey — Qwen3.5-Next-Mini (3B/1.5B) availability check on HF / mlx-community | Is there a closer-architecture drafter? | If yes → measure coverage; if no → cycle 23 |
| 23 | Distillation training — train a small Qwen3.5 drafter on 4-bit target outputs | Can KD push coverage@1 from 6.3% upward? | Substantial work; multi-day cycle |
| 24+ | MTP-head training | Last resort if drafter survey/distillation flat | — |

## Files added in cycle 19

- `scripts/probe_c5_extended_coverage.py` — extended b∈{1..1000} probe + rank stats
- `plans/P6_C5_DDTREE/extended_coverage_probe.jsonl` — measurement artefact
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_19.md` — this file

## Ledger row added cycle 19

- `AR_C19_EXTENDED_COVERAGE` (diagnostic, KEEP — opens C.5 γ.1 leg) — extends
  β.2 to b∈{1..1000}, reveals coverage@64 = 0.4051 crossing the 40% line.
  Median rank 129, p95 rank 1000, ~23% of positions outside top-1000 (the
  permanent off-distribution tail). Decision: enter research loop to test
  whether b=64 tree-spec verify cost is sub-linear; if yes, build DDTree
  implementation.

## What to investigate next

Cycle 20 will run a tree-verify cost microbench at k ∈ {4, 8, 16, 32, 64}
on the cached `mlx-community/Qwen3.5-27B-4bit` to get the verify-cost
slope. If sub-linear, b=64 tree-spec is on the table; if linear, the
research pivots to drafter-side improvements.
