# P-6.0.5 Baseline Report

| Field | Value |
| --- | --- |
| Date | 2026-04-29 |
| Hardware | Apple M5 Pro, 48 GB unified memory, 307 GB/s peak bandwidth |
| Software | silica @ commit `4429cd2`, MLX-native, no codec, no speculative |
| Scenarios | 8 measured (5 mandatory warm-decode + 2 warm-TTFT pair + 1 microbench); 2 opt-in OOM-flagged rows both completed without OOM |
| Wall time | ~5 minutes total across all 8 measurements |

This document records the **P-6.0.5 measurement expansion** landing
per `plans/P6_0_5_OPENING.md` §4. It is the data input to
**Decision Gate 1** (PLAN.md §7 D-021 step 4); P-6.0.5 produces
inputs only — the gate writeup happens after this phase exit.

The eight rows close the four §1 open questions and re-anchor the
P-6 acceptance gates with empirical batch-scaling and verify-k
data that v1.7.13 P-6.0 did not collect.

---

## 1. Measured Rows

### 1.1 Warm-decode rows (decode_tok_s as a function of batch size)

| scenario | B | aggregate tok/s | per-row | linear efficiency vs B=1 | bandwidth util | peak (GB) | runs | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3.5-27b-warm-decode-b1 | 1 | 16.05 | 16.05 | — | 79.1% | 15.36 | 1 | P-6.0 baseline |
| qwen3.5-27b-warm-decode-b2 | 2 | 31.22 | 15.61 | 97% of 2× | 76.9% | 16.26 | 1 | P-6.0.5 Unit 1 |
| **qwen3.5-27b-warm-decode-b4** | **4** | **42.17 ± 0.21** | **10.54** | **66% of 4×** | **52.0%** | **17.10** | **2** | **P-6.0.5 Unit 2 (opt-in)** |
| qwen3.5-moe-35b-a3b-warm-decode-b1 | 1 | 76.01 | 76.01 | — | 37.1% | 19.40 | 1 | P-6.0 baseline |
| qwen3.5-moe-35b-a3b-warm-decode-b2 | 2 | 120.93 | 60.47 | 80% of 2× | 59.1% | 19.68 | 1 | P-6.0 baseline |
| qwen3.5-moe-35b-a3b-warm-decode-b3 | 3 | 163.50 | 54.50 | 72% of 3× | 79.9% | 20.42 | 1 | P-6.0.5 Unit 3 |
| **qwen3.5-moe-35b-a3b-warm-decode-b4** | **4** | **188.50** | **47.13** | **62% of 4×** | **92.1%** | **20.62** | **1** | **P-6.0.5 Unit 4 (opt-in)** |
| qwen3.5-moe-35b-a3b-warm-decode-b1-4k | 1 | 85.00 | 85.00 | — | n/a (KV-aware) | 23.60 | 1 | P-6.0.5 Unit 5 |

**Bandwidth-util anchor**: dense rows above are computed against
the **runtime-measured weight footprint of 15.13 GB** (from
`mlx.utils.tree_flatten(model.parameters())` on the loaded
Qwen3.5-27B-4bit checkpoint, including 4-bit scale / zero
metadata). This corrects the v1.7.13 P-6.0 anchor of 13.5 GB
("branded 27B × 0.5 byte/param") upward by ~12%; v1.7.13 ratios
remain valid relative comparisons but absolute % values shift —
e.g. v1.7.13's 70.6% (B=1) reads **79.1%** under the corrected
anchor. The microbench JSONL field
`weight_bytes_read_estimate=10.066 GB` is a separate fallback
estimate (geometric `nl × h² × 12 / 2` formula triggered by the
checkpoint not exposing `num_parameters`) that under-counts on
both axes; it is preserved in `target_verify_microbench.jsonl`
as a reproducibility artefact, not as the utilisation anchor.
See `target_verify_microbench.md` "Weight-footprint
reconciliation" for full details. **MoE rows** retain the v1.7.13
1.5 GB active-weight anchor (no microbench-grade tree_flatten
measurement was taken on the MoE checkpoint in this phase); the
1.5 GB number reflects only the active-3B path and may
under-state effective bytes/step at higher batch sizes where
router diversity touches more experts. MoE % values should be
read as anchor-consistent with v1.7.13 rather than corrected.

**Reproducibility note**: rows with `runs > 1` have multi-run
data in their `.jsonl`. Dense 27B B=4 was run twice (41.97 /
42.38 tok/s, σ = 0.21, **0.5% rel-std**), confirming the
warm-decode oracle is highly reproducible and the headline 42.17
tok/s aggregate (66% of ideal 4× linear) is not a single-run
artefact. The dense warm-TTFT pair below was run three times
(see §1.2). Single-run rows are tight measurement-wise (the
warm-decode oracle's per-step rel-std runs ~0.5% within a
single run, see per-row `.md` metadata blocks) but cross-run
variance was not characterised on those rows.

`bandwidth util` for short-context rows is `decode_tok_s × bytes/step ÷ 307 GB/s`,
using **dense 15.13 GB** (the runtime-measured `tree_flatten` weight
footprint declared in the anchor footnote above) and **MoE 1.5 GB**
(v1.7.13 active-weight anchor; no microbench-grade tree_flatten
measurement was taken on the MoE checkpoint in this phase). The
4K-context row's util is omitted because at
4K the per-step KV read is no longer negligible relative to the
active-weights stream and a weights-only ceiling under-states
bytes/step.

### 1.2 Warm-TTFT pair rows

| scenario | warm TTFT (ms) | cold TTFT (ms) | compile-amortised (ms) | runs |
| --- | ---: | ---: | ---: | ---: |
| qwen3.5-27b-warm-ttft-pair | **316.9 ± 0.3** | 750.2 ± 143 | 433.3 ± 144 | 3 |
| qwen3.5-moe-35b-a3b-warm-ttft-pair | **169.3** | 1530.5 | 1361.2 | 1 |

The dense pair was run three times; warm TTFT is reproducible to
±0.1% while cold TTFT and compile cost have ~19% / 33% rel-std.
Cross-family: MoE warm TTFT is ~half of dense (active-3B prefill
advantage), MoE compile cost is 3.1× dense (per-expert kernel
variants).

### 1.3 Target-verify microbench (verify_k cost curve, dense Qwen3.5-27B-4bit)

| verify_k | p50 (ms) | marginal vs k=1 (ms) | bandwidth util | per-extra-tok marginal | autoregressive equiv (ms) | max speedup (perfect drafter) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 59.59 | +0.00 | 82.7% | — | — (baseline) | — |
| 2 | 62.07 | +2.48 | 79.4% | 2.48 ms/tok | 119.18 | 1.92× |
| 4 | 89.01 | +29.42 | 55.4% | 9.81 ms/tok | 238.36 | 2.68× |
| 8 | 162.96 | +103.37 | 30.2% | 14.77 ms/tok | 476.72 | **2.93×** |

Bandwidth utilisation **drops** from 83% → 30% across k=1 → k=8;
the verify forward sits in the same regime as warm-decode-b1 at
k≤2 (≈79–83% util) and leaves bandwidth-bound territory around
k=2 → k=4. Break-even drafter acceptance: ~50% (k=2), ~36% (k=4),
~33% (k=8). The 2.93× column is the **target-side / zero-drafter-cost
ceiling**, not end-to-end speculative speedup: real spec-decoding
gain falls below this by drafter forward cost, drafter
acceptance probability, and the bonus-token rule.

---

## 2. §1 Open-Question Closures

### §1 Q1 — Does dense 27B scale with batch?

**Closed: NO.** The "≥90% utilisation" arm of the question is
**decisively rejected** for dense.

Bandwidth utilisation **drops** from 79.1% (B=1) → 76.9% (B=2)
→ **52.0% (B=4, 2-run mean)** on the 15.13 GB corrected weight
anchor. Aggregate scaling efficiency falls 97% (at B=2) → 66%
(at B=4); per-row throughput drops 16.05 → 15.61 → 10.54 tok/s
(−34%). The mechanism is KV / activation traffic surfacing
faster than weight reads amortise: at B=4 the weight-read is
no longer the only meaningful bandwidth consumer, so the
weights-only ceiling formula over-estimates achievable
aggregate.

**§6(1) ≥60 tok/s gate consequence**: dead by batching alone.
Linear extrapolation from B=1 predicted B=4 = 64.2 tok/s; measured
**42.17 ± 0.21** leaves a **17.83 tok/s residual gap (30% of gate)**.
B=8 is infeasible on 48 GB without aggressive tricks. The gate is
reachable only via the composite Track A + Track B + (Track C.4
or C.5) stack.

### §1 Q2 — Does MoE saturate before B=4, and is B=4 OOM-safe?

**Closed: NO (does not saturate); YES (OOM-safe at 20.62 GB).**

MoE bandwidth utilisation **climbs** monotonically from 37.1%
(B=1) → 59.1% (B=2) → 79.9% (B=3) → **92.1% (B=4)**. The §1 Q1
"unsaturated → climbs to >90%" hypothesis applies — but on the
**MoE family, not dense**. Aggregate scaling efficiency falls
modestly (80% → 72% → 62%); B=3 → B=4 still delivers 86.5% of
ideal 1.333×. At 92% utilisation the curve is near its
weight-read ceiling; B=5 / B=6 may yield small gains, but most
remaining bandwidth has been claimed.

**§6(2) ≥100 tok/s gate consequence**: exceeded by 88% at B=4
(188.5 vs 100). The (2b) reframing branch:

- **Aggregate variant ≥150 tok/s** — already cleared at B=3
  (163.5) and at B=4 (188.5 = 1.26× stretch). Live candidate.
- **Per-row variant ≥100 tok/s at B=2** — structurally
  unreachable (per-row 76 → 60 → 54 → 47, monotonically
  decreasing). Should be retired or moved to a different
  checkpoint.

**OOM-safety**: B=4 peak 20.62 GB; ≥15 GB margin to the §6(4)
36 GB envelope. The B=2 → B=3 → B=4 marginal memory cost is
+0.28, +0.74, +0.20 GB per added batch row — sub-linear with
allocator amortisation.

### §1 Q3 — Is 4K-context safe on MoE?

**Closed: YES.**

MoE B=1 4K-context peak 23.60 GB; §6(4) RAM-headroom gate
(≤36 GB at 4K) passes with **12.4 GB margin (35% safety)**.
Decode 85.0 tok/s is +12% above the short-context B=1 baseline
76.01 — counter-intuitive but the load-bearing finding is
conservative: 4K context does **not** degrade decode, and the
(2b) reframing does not need to subtract any context-length
budget. TTFT scaled +76% (2732 → 4822 ms) in line with linear
prefill cost.

### §1 Q4 — What is the verify-k cost curve?

**Closed: target-side / zero-drafter-cost ceiling 2.93× at k=8 linear.**

This is the speedup achievable when (a) the drafter incurs zero
forward cost, (b) drafter acceptance is 100% on all k tokens, and
(c) candidate shape stays linear; real end-to-end speculative
gain falls below this ceiling by drafter-forward cost, drafter
acceptance probability, and the bonus-token rule.

Bandwidth utilisation drops from **83% (k=1) to 30% (k=8)** on
the corrected 15.13 GB weight anchor (v1.7.13 anchor reads 74% →
27%); the curve has a clear regime transition between k=2 (same
regime as warm-decode-b1, ~79% util) and k=4 (entering
compute-bound). Per-extra-tok marginal cost rises 2.48 → 9.81 →
14.77 ms across k=2/4/8.

**ROI ceiling implications for PLAN.md §1.3 stack**:

- **C.4 DFlash** (claimed 6× on GPU; PLAN.md conservative MLX
  band 2.0–4.0×): upper-band 4.0× **provably unreachable** at
  k=8 linear; realistic landing 2.0–2.9×.
- **C.5 DDTree** (claimed 8.2× on GPU; PLAN.md conservative MLX
  band 2.5–5.0×): upper-band 5.0× **unreachable** at k=8 linear;
  ≥3.0× requires the **tree-shape amortisation** structurally
  beyond what this linear-k microbench measures.

Sweet spot for linear-shaped speculative on this hardware:
**k=4** (55% util on the corrected anchor, break-even acceptance
36%). k=8 reserved for high-acceptance drafters where the 2.93×
target-side ceiling is approachable.

---

## 3. Cross-family contrast (the load-bearing finding)

The same hypothesis "does utilisation climb with batch?" splits
the two families onto opposite arms:

| family | B=1 util | B=4 util | per-row B=1 → B=4 | regime | gate consequence |
| --- | ---: | ---: | ---: | --- | --- |
| Dense 27B | 79.1% | **52% (drops)** | 16.05 → 10.54 (−34%) | KV-traffic-bound at B≥4 | §6(1) gate dead by batching alone |
| MoE 35B-A3B | 37.1% | **92% (climbs)** | 76.01 → 47.13 (−38%) | weight-read-bound throughout | §6(2) cleared by 88% at B=4 |

**Cross-family % comparison is approximate, not strict.** The
two utilisation columns above use different bytes/step composition
rules under batching:

- **Dense**: `util = aggregate_tok_s × 15.13 GB ÷ (B × 307 GB/s)` —
  weights amortise across batch rows (one weight stream per step,
  produces B tokens). Standard dense-batching assumption.
- **MoE**: `util = aggregate_tok_s × 1.5 GB ÷ 307 GB/s` —
  inherited from v1.7.13's MoE ceiling formula, which assumes
  each batch row reads its own 1.5 GB active-expert set with no
  expert overlap across rows. Equivalent to "bytes/step grows
  linearly with B".

Applying the dense-style amortisation to MoE B=4 would give
`188.5 × 1.5 ÷ (4 × 307) = 23%` rather than 92%, which is also
not right — MoE active-set composition is somewhere between full
amortisation (one shared expert set) and full per-row reads (B
disjoint sets), depending on routing diversity, and v1.7.13
chose the latter as the conservative ceiling. The 92% figure
reads as "MoE B=4 fills 92% of the v1.7.13-style aggregate
ceiling"; cross-family interpretation should treat the dense and
MoE columns as **regime indicators** (climbing vs dropping)
rather than directly-comparable percentages.

Dense had little bandwidth headroom at B=1 (already 79.1% util
on the corrected 15.13 GB weight footprint), so adding batch
surfaces KV/activation traffic faster than weight-read
amortisation can absorb. MoE had abundant headroom at B=1
(37.1% util on a 1.5 GB active-weight read under the v1.7.13
ceiling formula), so batch parallelism consumes the slack
cleanly until weight reads saturate near the ceiling. Per-row
degradation percentages are similar (−34% / −38%) but the
underlying mechanisms differ.

**This contrast is the load-bearing input to Decision Gate 1**:
the dense and MoE acceptance reframings cannot share a single
pattern — they must be argued separately because the chip puts
each family in a structurally different regime.

---

## 4. Decision Gate 1 input summary

The four §1 questions resolve to a constrained decision space.

### Dense 27B — §6(1) primary gate (≥60 tok/s)

- **Batch-only path: dead.** B=4 = 42.17 ± 0.21 tok/s (2 runs); B=8 infeasible on 48 GB.
- **Verify-k ceiling: 2.93×** at k=8 perfect (Unit 7).
- **Composite stack required.** PLAN.md §1.3 mid-band arithmetic
  (A 1.05–1.15× × B 1.30× × C.4/C.5 2.0–4.0×) projects
  16.05 → 33–96 tok/s; the lower band is now empirically
  anchored at ~30 tok/s (A+B alone), the upper band requires
  C.4/C.5 to land in the upper half of their MLX-conservative
  ranges. The 60 tok/s gate is **reachable** but **only with
  C.4 / C.5 landing in the upper half of their bands** on top
  of A and B.
- **Decision arms** (gate-1 closure work, post-this-phase):
  - **(1a) lower the gate** to a band reachable from A+B alone
    (e.g. ≥40 tok/s) — anchors the dense engineering target on
    A+B, decouples C-track ROI from gate-clearing
  - **(1b) keep ≥60, accept** that C.4 / C.5 must land in the
    upper half of MLX-conservative bands — anchors gate on the
    full stack including speculative

### MoE 35B-A3B — §6(2) stretch gate (originally ≥100 tok/s aggregate)

- **Already cleared at v1.7.13 baseline** (B=2 = 120.93).
- **B=4 = 188.5** = 1.88× original gate, 1.26× of the (2b) ≥150
  stretch.
- **Per-row arm structurally unreachable** (per-row falls with B).
- **Decision**: re-anchor on **aggregate-only**, retire per-row
  variant; consider raising the threshold to ≥175 or ≥200 tok/s
  aggregate to keep the gate informative beyond B=4. (2b)
  reframing is empirically warranted.

### §6(4) RAM-headroom gate (≤36 GB at 4K context)

- **Not stressed at any measured shape.** Maxima:
  - Dense 27B B=4: 17.10 GB (53% margin to 36 GB)
  - MoE 35B-A3B B=4: 20.62 GB (43% margin)
  - MoE 35B-A3B B=1 4K: 23.60 GB (35% margin)
- **No ratification needed**; gate stands.

### §6(3) TTFT-under-concurrency gate (currently unmeasured at v1.7.13)

- **Steady-state warm TTFT anchors** (B=1, ~115-token prompt):
  - Dense: **317 ms** (3-run mean, σ ≈ 0.3 ms — reproducible)
  - MoE: **169 ms** (1 run; cross-family delta ≈ ½ × dense)
- **Compile-amortised cost** (paid once per process boot):
  - Dense: 433 ms (3-run mean, high run-to-run variance)
  - MoE: 1361 ms (per-expert kernel compile)
- **B>1 / shared-prefix concurrency TTFT** remains unmeasured;
  Track D scope, deferred.

---

## 5. Acceptance gate assessment

Mirroring the v1.7.13 P-6.0 §2 table, with P-6.0.5 data folded
in:

| gate | required | best P-6.0.5 measurement | gap | status |
| --- | --- | --- | --- | --- |
| (1) Dense 27B ≥60 tok/s | 60 | 42.17 ± 0.21 (B=4, 2-run, this phase) | **−17.83** (70% of gate) | open — see §4 dense arms |
| (2) MoE 35B-A3B ≥100 tok/s aggregate | 100 | 188.50 (B=4, this phase) | +88.50 (188% of gate) | **already satisfied; (2b) reframing live** |
| (3) TTFT under concurrency | not measured | n/a (B=1 anchors only: 317 / 169 ms) | new B>1 scenario needed | track D — deferred |
| (4) RAM headroom 27B B=1 4K ≤36 GB | 36 GB | 17.10 GB (dense B=4) / 23.60 GB (MoE 4K B=1) / 20.62 GB (MoE B=4) | ≥12 GB margin everywhere | **comfortably satisfied** |

---

## 6. Open issues / deferred work

- **B≥5 batch sizes** for MoE not measured. At 92% utilisation
  the marginal aggregate gain is bounded above by ~10%; B=5/6
  might yield a small absolute improvement but is not load-bearing
  for the (2b) reframing.
- **B>1 + 4K-context combination** not measured. Would intersect
  §6(2) and §6(4) jointly; out of P-6.0.5 scope per opening §2.2.
- **Tree-shape (C.5) verify cost curve** not measured by Unit 7
  (linear-k only). The 2.93× ceiling is the linear-shape upper
  bound; tree shape may exceed it via tree-amortisation.
- **B>1 shared-prefix TTFT** not measured (Track D dependency).
  The warm-TTFT-pair rows establish a B=1 anchor but do not
  test concurrency.
- **`silica.bench.microbench.target_verify.estimate_weight_bytes_read`
  fallback path under-counts on Qwen3.5-27B-4bit.** The function
  prefers `model.args.num_parameters × 0.5`, which is unset on
  this checkpoint, so it falls back to the geometric formula
  `nl × h² × 12 ÷ 2` and emits 10,066,329,600 bytes (10.07 GB) —
  ~33% below the runtime-measured 15.13 GB. The JSONL field is
  preserved as a reproducibility artefact and the .md / REPORT
  bandwidth-util columns use the corrected 15.13 GB; future
  microbench runs will keep emitting the under-counted estimate
  until the fallback is upgraded to a `mlx.utils.tree_flatten`
  path or the `num_parameters` config field is populated. Filed
  as a known issue for the next microbench-touching phase, not a
  P-6.0.5 blocker.

---

## 7. Cross-references

- **PLAN.md §7 D-021 step 3** — P-6.0.5 scope statement
- **PLAN.md §7 D-021 step 4** — Decision Gate 1 (consumes this report)
- **plans/P6_0_5_OPENING.md** — opening / scope / acceptance / sub-unit specs
- **plans/P6_0_BASELINE/REPORT.md** — v1.7.13 anchors used for ratios (16.05 / 76.01 / 120.93 tok/s)
- **plans/P6_0_5_BASELINE/\*.md** — per-row interpretation paragraphs for full numerics
- **plans/P6_0_5_BASELINE/logs/** — raw stdout/stderr per-run logs
