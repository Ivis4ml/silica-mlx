# β.1 — `mx.compile` post-cache attention microbench, sonnet refresh

Reproduces cycle-16's "attention forward without cache mutation: 1.08×"
claim on the sonnet stack at production attention shapes.
Methodology and gate per `plans/P6_SMALL_B_BETA_OPENING.md` §4.1.

## Setup

- Repo: `mlx-community/Qwen3.5-27B-4bit` (cache-warm).
- Toolchain: mlx 0.31.1, mlx-lm 0.31.2, mlx-metal 0.31.1.
- Hardware: M5 Pro 48 GB unified memory.
- Layer: `model.layers[3]` (first `is_linear=False` full-attention block).
- Shape config: B=4, num_q_heads=24, num_kv_heads=4, head_dim=256,
  scale=1/√256, dtype=bfloat16.
- Three timing arms per shape: uncompiled / `mx.compile(fn, shapeless=True)`
  / `mx.compile(fn)` (fixed-shape upper bound).
- 3 warmup iters + 20 measurement iters, median wall ms.
- Two sessions back-to-back (per α caveat; cross-day separation deferred).

## Results — combined across 2 sessions

| T_kv | uncompiled mean (ms) | shapeless mean (ms) | shapeless× ± σ | fixed× ± σ |
| ---: | ---: | ---: | ---: | ---: |
| 128  | 0.768 | 0.310 | **2.471 ± 0.331** | 2.526 ± 0.288 |
| 1024 | 0.404 | 0.385 | **1.052 ± 0.047** | 1.006 ± 0.003 |
| 4096 | 0.705 | 0.673 | **1.047 ± 0.021** | 1.042 ± 0.010 |

Raw rows: `plans/P6_SMALL_B/BETA/microbench/20260506_093246/` (session 1)
and `.../20260506_093327/` (session 2), `compiled_attn_postcache_b4.jsonl`
each (4 rows: 3 shape_result + 1 summary).

## β.1 gate (microbench)

| Gate | Threshold | Result | Status |
| --- | --- | --- | --- |
| Best shapeless speedup | ≥ 1.05× | 2.471× (T_kv=128) / 1.052× (T_kv=1024) / 1.047× (T_kv=4096) | local signal present |
| Combined σ on speedup ratio | ≤ 0.03 | T_kv=128: 0.331; T_kv=1024: 0.047; T_kv=4096: 0.021 | T_kv=4096 PASS, T_kv=1024 marginal, T_kv=128 FAIL |
| Same-shape combined line gate | speedup ≥1.05× and σ_ratio ≤0.03 on at least one T_kv | none: T_kv=1024 clears speedup but not σ; T_kv=4096 clears σ but not speedup | **ESCALATE_BAND** |

**Verdict on the β.1 combined line gate: ESCALATE_BAND, not clean
PASS_SHAPELESS.** Each raw session's JSONL summary reports
`PASS_SHAPELESS` because the microbench module records a single-session
pre-variance screen. The opening's line-level gate is stricter: the
same T_kv shape must clear both ≥1.05× and σ_ratio ≤0.03 across
sessions. No shape does.

Cycle 16's 1.08× claim is still directionally reproduced on the current
sonnet stack: the mid-T_kv ≈ 1024-4096 regime gives a noisy ≈1.05×
microbench signal. The T_kv=128 super-win (2.47×) is a real reading on
the data but its variance (σ_ratio 13%) makes it not load-bearing on
its own; the underlying mechanism is that uncompiled-call overhead at
very small caches is dominated by Python dispatch, which `mx.compile`
traces away.

## E2E projection — does β.1 microbench transfer to β.4 KEEP?

The β.4 gate requires ≥3% E2E p50 improvement at B=4 in paired ON/OFF.
α's sonnet-side step_total median is 115-134 ms (mean ≈ 125 ms across
2 α sessions). Each step has 16 full-attention layers (Qwen3.5-27B
hybrid: 16 full + 48 linear).

If β.2 integrates only the post-cache compile (`mx.compile` of the
SDPA + transpose + reshape + sigmoid + o_proj region), per-step time
saved is `(uncompiled - shapeless) × 16`:

| T_kv | per-call ms saved | × 16 layers | % of 125 ms step | vs β.4 3% gate |
| ---: | ---: | ---: | ---: | --- |
| 128  | 0.458 | 7.33 | **5.86%** | clears |
| 1024 | 0.020 | 0.31 | **0.25%** | **fails by 12×** |
| 4096 | 0.032 | 0.51 | **0.41%** | **fails by 7×** |

The realistic chat / serving regime (T_kv ≥ ~256 for any prompt with a
system message + first user turn) projects 0.25-0.41% E2E gain.

**Even granting the local microbench signal, the math projection at
production T_kv falls 6-12× short of β.4's 3% per-row gate.** Cycle-17
saw the same shape (1.027× synthetic / 0.5% E2E for MLP `mx.compile`)
and discarded the lever as below noise floor.

## Why so much smaller than the bucket headline

α's "full-attn 21.6% of step" (decode_step_attribution) measures
**whole full-attention layer blocks** including `input_layernorm` (2%),
`self_attn` (6.7%), `post_attention_layernorm` (2.1%), and `mlp`
(11.4%) — confirmed by α's layer-internal data.

β targets only `self_attn`'s post-cache half (SDPA + sigmoid + o_proj).
That post-cache half is ≈ half of `self_attn` ≈ **3-4% of step time**.
A 1.05× speedup on a 3-4% bucket is 0.15-0.20% E2E — consistent with
the empirical 0.25-0.41% projection.

The 22% bucket headline is misleading for β specifically. The
cycle-1 framing implicitly assumed a larger compile-reachable share
than actually exists once cache mutation and pre-cache are split out.

## T_kv=128 anomaly note

The 5.86% projection at T_kv=128 would clear β.4 gate, but:

1. **σ_ratio = 13%** (uncompiled mean 0.681 → 0.855 ms across the two
   sessions, 25% session-to-session swing) — not load-bearing under
   the cycle-27 variance discipline.
2. **T_kv=128 is unrealistic for chat / serving.** Default chat-CLI
   system prompts plus the first user turn typically exceed 256
   tokens; production prompts cluster in the 256-2048 range for
   short-form interactive use.
3. The uncompiled instability at small T_kv suggests the mechanism is
   first-call dispatch overhead being traced away, not a structural
   compile win that scales.

A speculative β extension to support very-short-prompt cases via
shape-bucketed compile is possible but is not currently a single-customer
priority.

## β.1 disposition — recommendation

This measurement falls in the **escalate band**: the single-session
screen sees ≥1.05× shapeless signal, but the stricter combined
σ_ratio gate does not cleanly pass; independently, the math projection
against β.4's E2E gate clearly FAILS at production T_kv. Neither a
clean KEEP nor a clean retire of the line.

Per the `feedback_escalate_band_fact_bundle_pattern` discipline, this
report is committed first as the fact-bundle. Final disposition for
the β line — proceed with β.2/β.3/β.4 to confirm empirically, or
close β with a measurement-anchored negative on math projection — is
deferred to a separate later commit after user verdict.

**Author recommendation: close β with measurement-anchored negative.**

Reasoning:

1. The math projection at production T_kv (0.25-0.41% E2E) is 6-12×
   below the β.4 gate. β.2/β.3/β.4 work cannot bridge that gap; the
   underlying physics is that `mx.compile`'s reachable scope (post-cache
   self_attn = ~3-4% of step) is too narrow to amplify a 5% per-call
   gain into a single-customer KEEP.
2. Cycle-17 / cycle-18 lessons apply directly: synthetic microbench
   wins of 1.02-1.08× on `mx.compile` have repeatedly projected
   ≤ 1% E2E and have been retired. β.1's own microbench data is
   numerically consistent with that pattern at production T_kv.
3. The v1.7.25 memory rule on single-customer gates states: a
   microbench-only win is not a KEEP; the win must be observable in
   E2E B=4 measurements. Math projection makes the empirical E2E
   measurement decision-making redundant in this case.

**Alternative path if user wants empirical confirmation: run β.2 +
β.3 + a single β.4 session.** Cost: ~1-2 hours wall, spans a code
change to `silica/kernels/shadow_install.py` that would need to be
reverted on the close path. Expected outcome: ~0.2-0.5% E2E gain at
B=4 paired ON/OFF, β.4 close-with-negative.

If close path is taken: γ (`mx.compile` on `Qwen3NextMLP`) is the
next sub-unit per the v1.7.25 ordering. γ's bucket is larger
(linear.mlp 34.9% + full.mlp 11.4% = 46.3% step), but the same
post-cache-vs-bucket gap may apply; γ.1 microbench should explicitly
project E2E before any γ.2 integration.

## Cross-references

- `plans/P6_SMALL_B_BETA_OPENING.md` — β line opening, gates,
  sub-units.
- `silica/bench/microbench/compiled_attn_postcache.py` — β.1
  measurement code.
- `plans/P6_SMALL_B/REPORT.md` — α report; layer-internal data is the
  source for the post-cache bucket fraction estimate.
- `plans/P6_AUTORESEARCH_NOTES.md` cycle 16 — original 1.08×
  microbench claim. β.1 reproduces.
- `plans/P6_AUTORESEARCH_NOTES.md` cycles 17 / 18 — prior
  `mx.compile` discards (1.027× synthetic / 0.5% E2E and 1.019× / 1%
  E2E). Same shape as β.1 projection.
- v1.7.25 memory `feedback_p6_small_b_single_customer_gates.md` —
  microbench-only wins are not KEEPs.
- v1.7.20 memory `feedback_escalate_band_fact_bundle_pattern.md` —
  split-commit discipline for escalate-band measurements.
