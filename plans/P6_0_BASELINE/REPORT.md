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

### Implication for the dense-60 gate

The honest reading: **the dense 60 tok/s gate may be unreachable on
M5 Pro 48 GB even with the full Track A + B + C stack.** Three
contingencies, ranked by likelihood:

1. **C.4 / C.5 deliver as claimed on MLX.** DFlash claims 6× and
   DDTree 8.2× on GPU; the MLX ports report 1.5× over autoregressive
   on real silicon. If a fresh measurement under silica's engine
   delivers ≥3× combined with Track A + B, the gate clears at
   ~60-75 tok/s. **The fastest way to know: land C.4 first as a
   measurement.**

2. **Re-target to ≥40 tok/s for dense at 48 GB.** This is what the
   bandwidth math actually supports without C.4/C.5 best-case. Per
   the §6 phase-exit clause, missing the 60 tok/s gate triggers a
   Decision Log entry naming the measured engine-overhead floor —
   that path is already documented.

3. **Dual-target re-confirmation.** The MoE 100 tok/s gate is
   already met; the dense gate is the one stuck. Per Q-A's
   resolution, the dense gate is the phase-failing primary; missing
   it means the phase exits via re-target rather than success.

**Action item:** track-order should put C.4 (DFlash) first to test
contingency 1 quickly. If C.4 lands ≥2.5× over baseline, contingency
1 holds. If C.4 lands <1.8×, contingency 2/3 trigger.

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

## 7. Recommended Next Steps

Given the baseline data:

1. **Confirm or re-target the dense 60 tok/s gate before committing
   significant Track work.** The bandwidth math says the gate is at
   the edge; DFlash (C.4) is the cheapest test of whether speculative
   buys enough headroom to clear it. Land C.4 first as a measurement,
   then decide.

2. **Track A is high leverage on MoE, low leverage on dense.** If
   Track A lands first, it will move MoE B=2 from ~121 to ~150-180
   tok/s aggregate but barely move dense 27B from ~16. Order tracks
   accordingly: A is a "free win" but not the dense-gate-cracker.

3. **Track B (3-bit) is the cheapest engine win for dense.** It
   re-applies the bandwidth math at smaller bytes/step. If 3-bit
   Qwen3.5-27B holds quality, it lifts the dense ceiling from 22.7
   to ~30 tok/s at no engine cost — and combined with C.1 spec
   (1.4-1.8×), reaches 40-50 tok/s on dense without C.4/C.5.

4. **Track C.4 (DFlash) is the gate-decider for dense 60 tok/s.**
   The other C variants (C.1/C.2/C.3) cannot get dense to 60 alone.
   Land C.4 measurement before deciding whether to re-target the gate.

5. **MoE stretch is met; raise the bar.** Phase exit on MoE is no
   longer in question. Use MoE measurements as the "clean
   demonstration" track for sync collapse / sampler fusion / spec
   variants — every Track A-C win shows up cleanly on MoE because
   the path has bandwidth slack.

6. **Defer Track D.1 measurement until concurrency scenarios exist.**
   The TTFT-under-concurrency gate requires a new scenario shape
   that doesn't exist yet; not a baseline gap.

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
