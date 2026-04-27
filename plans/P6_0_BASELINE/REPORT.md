# P-6.0 Baseline Report

| Field | Value |
| --- | --- |
| Date | 2026-04-27 |
| Hardware | Apple M5 Pro, 48 GB unified memory, 307 GB/s bandwidth |
| Software | silica @ commit 2730dee, MLX-native, no codec, no speculative |
| Scenarios | 8 (3 cache-only + 5 dual-gated; B=4 MoE deliberately skipped per OOM risk) |
| Wall time | ~3 minutes total across all 8 runs |

This document records the P-6.0 measurement gate landing per
`plans/P6_OPENING.md` §2. Every later P-6 sub-unit's success criterion
is a ratio against the numbers below, not an absolute target.

---

## 1. Measured Baselines

| scenario | B | bytes/step (GB) | ceiling tok/s | measured | util % | TTFT (ms) | peak (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6B (cache-only) | 1 | 0.3 | 1023.3 | 161.16 | 15.7% | 39 | 1.44 |
| 0.6B B=2 (cache-only) | 2 | 0.3 | 1023.3 | 208.72 | 20.4% | 54 | 1.72 |
| 0.8B hybrid (cache-only) | 1 | 0.4 | 767.5 | 123.28 | 16.1% | 43 | 1.77 |
| **27B dense (P-6 primary)** | 1 | 13.5 | 22.7 | **16.05** | **70.6%** | 1920 | 15.36 |
| 31B dense (Gemma4) | 1 | 15.5 | 19.8 | 13.63 | 68.8% | 950 | 17.51 |
| MoE 35B-A3B B=1 | 1 | 1.5 | 204.7 | 76.01 | 37.1% | 2732 | 19.40 |
| **MoE 35B-A3B B=2 (P-6 stretch)** | 2 | 1.5 | 204.7 | **120.93** | **59.1%** | 2056 | 19.68 |
| MoE 26B-A4B (Gemma4) | 1 | 2.0 | 153.5 | 68.62 | 44.7% | 454 | 14.38 |

`bytes/step` uses naive `params × 0.5 B/param` for 4-bit; the realistic
number is ~5-15% higher (group-quant scale/zero metadata) and so the
true ceiling sits just below the column above. Numbers here come from
`plans/P6_0_BASELINE/<scenario>.jsonl` files; the per-run logs are in
`logs/p6_0_step{N}_<scenario>.log`.

---

## 2. Acceptance Gate Assessment vs §6

| gate | required | baseline | gap | status |
| --- | --- | --- | --- | --- |
| (1) Dense Qwen3.5-27B-4bit ≥60 tok/s | 60 | 16.05 | **−43.95** (27% of gate) | open — see §3 |
| (2) MoE Qwen3.5-35B-A3B-4bit ≥100 tok/s aggregate | 100 | 120.93 | **+20.93** (121% of gate) | **already satisfied at baseline** |
| (3) TTFT under concurrency | not measured | n/a | new scenario needed | track D |
| (4) RAM headroom 27B B=1 4K ctx ≤36 GB | 36 GB | 15.36 GB peak (384 ctx) | +20.6 GB headroom | likely safe; re-validate at 4K context |

**Headline:** the MoE stretch validator passed on bare baseline
without any P-6 track work. The dense primary gate is far short of
the target — see §3 for the analysis.

---

## 3. Dense 27B — Gap Analysis

**Measured:** 16.05 tok/s at 70.6% bandwidth utilization.
**Required:** 60 tok/s.
**Gap multiplier:** **3.74×**.

### What the §1.3 plan arithmetic predicted

| stack | claimed multiplier | reachable from 16.05 |
| --- | --- | --- |
| Track A engine fusion (sync collapse + sampler + lazy snapshot) | 1.05 – 1.15× | ~17 – 18 tok/s |
| + Track B 3-bit weights | × 1.30× | ~22 – 24 tok/s |
| + Track C.1 draft-target speculative (50-70% accept) | × 1.4 – 1.8× | ~31 – 42 tok/s |
| + Track C.4 DFlash (claimed 6× over autoregressive on GPU; conservative MLX 2-3×) | × 2.0 – 4.0× | ~32 – 96 tok/s |
| + Track C.5 DDTree (claimed 8.2× on GPU; conservative MLX 2.5-5×) | × 2.5 – 5.0× | ~40 – 120 tok/s |

**Interpretation:** the 60 tok/s gate is **reachable only with C.4 or
C.5 landing in the upper half of their MLX-conservative bands**, OR
with Track A clawing back substantially more than its mid-estimate
(unlikely on a bandwidth-bound regime). C.1/C.2/C.3 alone, even
stacked with A and B, do not get there.

### Why bandwidth utilization is already high (70.6%)

The dense 27B path is bandwidth-saturated, not overhead-saturated.
Per the research finding (vllm-metal RFC #188), sync-barrier overhead
is largely *hidden* in the bandwidth wait when bytes/step is large.
This means Track A's win on dense 27B is at the low end of its
estimated band — the +5-15% figure in `plans/P6_OPENING.md` §3 was
correct; the higher +30-80% figure applies to MoE active-param paths
where the chip has bandwidth slack to expose dispatch overhead.

### Implication for the dense gate (re-framed at v1.7.14 per D-021)

The v1.7.14 contract sync (D-021 in `plans/PLAN.md` §9) splits the
dense gate into a two-tier shape that aligns with what the
bandwidth math actually supports:

- **(1a) Dense engineering gate ≥40 tok/s — must pass.** Reachable
  envelope from the 16.05 baseline: A 1.10-1.15× × B 1.30× × C.1
  1.40-1.80× → 32-50 tok/s. This is the gate the phase exits on.
- **(1b) Dense stretch gate ≥60 tok/s — contingent on Track C.4 / C.5
  ≥2.5× silica-integrated speedup.** Reaches 60-75 tok/s only when
  block-diffusion drafters deliver in the upper half of their
  MLX-conservative bands. Decision Gate 1 (D-021 step 4) measures
  the C.4 spike against a ≥1.8× lower bound; below that, (1b) is
  retired to a Decisions Log entry naming the empirical floor.

**Three resolution paths from this baseline, ranked by likelihood
under D-021's foundation-first ordering:**

1. **Stretch-on path: C.4 / C.5 deliver as claimed on MLX.** DFlash
   claims 6× and DDTree 8.2× on GPU; the MLX ports report 1.5× over
   autoregressive on real silicon. If silica's engine integration
   delivers ≥2.5× combined with Track A + B, (1b) clears at
   ~60-75 tok/s. The fastest way to know is the C.4 spike at D-021
   step 6, run after the spec foundation lands at step 5.

2. **Engineering-floor path: only (1a) is met.** This is the
   dominant outcome the bandwidth math supports without C.4/C.5
   best-case. Phase exits successfully on ≥40 tok/s engineering
   gate; (1b) is recorded as out-of-reach with the measured C.4
   speedup pinned in the Decisions Log.

3. **Re-target path: (1a) itself comes in below 40 tok/s.** Under
   D-021's Decision Gate 1, P-6.0.5 evidence (especially the
   target-verification microbench) would surface this before any
   Track work. Phase pauses for an explicit re-target Decision
   Log entry rather than auto-failing.

**Action item:** D-021 step ordering puts the spec foundation
(step 5) ahead of the C.4 spike (step 6). The C.4 spike's gate
threshold is ≥1.8× silica-integrated speedup over C.1; ≥2.5×
justifies pursuing (1b); below 1.8× retires (1b) per path 2 above.

---

## 4. MoE 35B-A3B — Bandwidth Has Slack, Engine Has Room to Grow

**B=1 baseline:** 76.01 tok/s at 37.1% utilization. **B=2 baseline:**
120.93 tok/s at 59.1% utilization (1.59× scaling from B=1).

### Why B=2 already clears the 100 tok/s gate

Active 3B at 4-bit reads only 1.5 GB/step against a 307 GB/s pipe,
leaving 80%+ bandwidth slack on B=1. Going to B=2 doubles the
per-step useful work without doubling the per-step bandwidth need
(activations are tiny vs weights, and weights are read once per
batched step regardless of B). Hence the 1.59× scaling, hence
clearing 100 tok/s at the bare baseline.

### Track A leverage on MoE

At 59.1% utilization B=2, there's still ~40% of the bandwidth pipe
unused. Engine fusion (Track A) is the lever that turns dispatch
overhead into bandwidth utilization. On the +30-80% band from the
research, MoE B=2 could reach 150-200 tok/s after Track A — well
into the territory where vllm-mlx's M4 Max number (127.7 tok/s)
becomes the conservative comparison.

### Implication for the MoE stretch

The MoE 100 tok/s gate is **safely cleared at baseline by 21%**.
Track work on MoE should aim higher — the natural stretch is
"≥150 tok/s aggregate at B=2" or "≥100 tok/s per-row at B=2"
(currently 60.5 tok/s per-row). These would validate that silica's
optimization stack is competitive with the GPU-class numbers vllm-mlx
publishes; they are stretch goals, not phase-exit gates.

---

## 5. Other Baselines — Side Notes

- **Gemma4-31B dense:** 13.63 tok/s at 68.8% utilization. Same
  bandwidth regime as Qwen3.5-27B; the §6 gate of "≥55 tok/s"
  faces the same physics challenge as the 27B gate. C.4/C.5
  decisions apply.
- **Gemma4-MoE 26B-A4B:** 68.62 tok/s at 44.7% utilization. Slightly
  different active-param ratio than Qwen3.5-MoE (4B vs 3B), slightly
  lower utilization — engine overhead is comparable. Track work
  applies similarly.
- **Cache-only sanity rows (0.6B / 0.8B):** all in the 123-209 tok/s
  range, confirming the WARM_DECODE oracle and runner work end-to-end
  on the dev box. Low utilization (15-20%) is expected for small
  models where Python overhead dominates.

---

## 6. TTFT Observations

| scenario | cold TTFT (ms) | comment |
| --- | --- | --- |
| 0.6B B=1 | 39 | minimal kernel compile |
| 0.6B B=2 | 54 | slight increase for batched compile |
| 0.8B hybrid | 43 | hybrid DeltaNet + GQA path |
| 27B dense | 1920 | dominant kernel compile (~2 sec) |
| 31B dense (Gemma4) | 950 | smaller — sliding/full hybrid attention compiles cheaper |
| MoE 35B-A3B B=1 | 2732 | block-level + expert FFN compile |
| MoE 35B-A3B B=2 | 2056 | second forward partially warm |
| MoE 26B-A4B | 454 | smaller layer count + smaller experts |

The §6(3) TTFT-under-concurrency gate is not yet measured (needs a
new scenario shape: 1 long + 3 short prompts at B=4 on dense 27B).
That work belongs to Track D.1 (chunked prefill).

---

## 7. Recommended Next Steps (v1.7.14 contract — D-021 ten-step path)

Given the baseline data, the recommended ordering follows the
foundation-first sequence committed at v1.7.14 (D-021). C.4 is the
dense-gate decider but it is **not** the first step — it lands at
step 6, after the spec foundation (step 5) and P5.9 hardening
(step 2):

1. **P5.9 hardening pass** (D-021 step 2). Eight bounded
   deliverables: probe double-load fix, Q-012 affirmative
   resolution, Qwen3.5 recurrent rollback, sustained 4K/8K
   memory probe, D-009 hot-path audit, speculative metrics
   schema, P-5 quality regression as P-6 per-track gate, and a
   full toolchain re-run. No new optimization features; only
   load-bearing crack repair.

2. **P-6.0.5 measurement expansion** (D-021 step 3). Add 27B B=2,
   MoE B=3, 27B 4K-context peak, warm-TTFT scenario, and the
   target-verification microbench (one target forward verifying
   2 / 4 / 8 candidate tokens). The microbench is the
   prerequisite for credible Track C ROI estimation.

3. **Decision Gate 1** (D-021 step 4). P-6.0.5 evidence either
   confirms (1a) ≥40 tok/s and (1b) ≥60 tok/s remain in pursuit,
   or records a re-target Decision Log entry. **No Track A-E PR
   opens until this gate records its re-confirmation.**

4. **Spec foundation + C.1** (D-021 step 5). DraftEngine wired
   into the engine main loop with greedy spec-on / spec-off
   parity gate; spec metadata schema populated; recurrent +
   KV rollback bound by a dedicated test. C.1 draft-target serves
   as the spec baseline every later C.x is compared against.

5. **C.4 DFlash spike** (D-021 step 6). **The dense gate decider —
   but lands fifth, not first.** Minimal closed loop: drafter
   wired, fixed P-6.0 prompt / scenario, output speedup +
   acceptance + draft overhead + peak memory + quality parity.
   Gate: ≥1.8× silica-integrated speedup over C.1 continues;
   ≥2.5× justifies pursuing the (1b) ≥60 tok/s stretch; below
   1.8× retires (1b) per D-021.

6. **Track B 3-bit** (D-021 step 7). Loader + PPL oracle first,
   pass quality gate, then 27B 3-bit warm-decode. If 3-bit lifts
   dense from 16 → 21-24 tok/s, stack with spec; otherwise ship
   opt-in.

7. **C.5 / C.2 / C.3 selection** (D-021 step 8). C.5 reuses the
   DFlash drafter and adds tree verification; C.2 ReDrafter pursued
   only if C.1 / C.4 are insufficient and (1b) is still in pursuit;
   C.3 MTP head if architecture compatibility is favorable. C.6
   QuantSpec-like self-spec is exploratory — only if C.4 / C.5
   land below 2× silica-integrated speedup.

8. **Track A sync collapse** (D-021 step 9). Repositioned as
   "general efficiency + MoE amplifier" — small lift on dense
   (bandwidth-bound), large lift on MoE B=2 (40% bandwidth slack).
   Explicitly **not** the dense-gate cracker; lands here so its
   wins are visible against an already-spec-running baseline.

9. **Track D / E** (D-021 step 10). D.1 chunked prefill + decode
   merging (resolves the TTFT-under-concurrency §6(3) gate);
   D.2 mlx-mfa kernel measurement-gated; E.1 MoE per-expert
   streaming preserves the original P-6 deliverable; E.2 active
   fp16 + cold compressed prefix tier.

**MoE stretch is met at baseline (120.93 tok/s aggregate at B=2);
phase exit on MoE is no longer in question.** Use MoE measurements
as the "clean demonstration" surface for every Track A-C win, since
the path has bandwidth slack.

---

## 8. Files

- `plans/P6_0_BASELINE/<scenario>.jsonl` — JSONL per scenario
  (8 files); machine-readable.
- `plans/P6_0_BASELINE/<scenario>.md` — Markdown report per
  scenario (8 files); human-readable detail.
- `logs/p6_0_step{N}_<scenario>.log` — full stdout/stderr with
  command at the head.
- `logs/p6_0_run_summary.md` — run metadata header.
- `logs/p6_0_baseline_summary.md` — table aggregating all 8 rows.
