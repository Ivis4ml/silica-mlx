# P-6 small-B sub-unit γ — `mx.compile` on `Qwen3NextMLP` forward

| Field | Value |
| --- | --- |
| Opening date | 2026-05-06 |
| Predecessor | β closed-negative at v1.7.26 (`plans/P6_SMALL_B/BETA/microbench/REPORT.md`). γ opens per the v1.7.25 sub-unit ordering. |
| Decision anchor | D-022 (`plans/PLAN.md` §9 + §13 v1.7.26 changelog). |
| Sibling sub-units | δ retains boundary-pass status (overhead 3.6%); ε waitlist. |
| Branch | `sonnet` |
| Toolchain pin | `mlx==0.31.1`, `mlx-lm==0.31.2`, `mlx-metal==0.31.1`. `tests/test_p2_preload_parity.py` is the determinism gate. |
| Targets | Dense Qwen3.5-27B-4bit (primary). |

This is the lightweight opening — γ borrows β's harness pattern. The
v1.7.26 *bucket × reachable-scope × per-call-gain* discipline is
applied **before** measurement so γ.1 is decision-shaped from the
first run.

## 1. TL;DR

`Qwen3NextMLP.__call__` is the compile target. Unlike β's `self_attn`
post-cache half (which excluded cache-mutation), the MLP forward has
**no cache or shape-mutating internal state** — gate / up / down
projections plus a SwiGLU activation. The full forward is in scope
for `mx.compile`, so γ has a cleaner reachable-scope multiplier than
β did.

**Hypothesis:** γ.1 microbench gives ≥ 1.10× on `Qwen3NextMLP`
forward at production shape (B=4, T_q=1, hidden_size, intermediate_size,
bf16). 1.10× × 46.3% bucket × 100% reachable = 4.2% E2E, clearing
β.4's ≥ 3% gate. Gain below 1.10× projects below the gate; γ closes.

**Key contrast with β.1.** β closed because the reachable scope
(post-cache half of `self_attn` ≈ half of 6.7% step = 3-4% step)
was too narrow. γ's reachable scope (full MLP forward = 46.3% step)
is ~12× larger, so γ's per-call gain threshold for E2E pass is
correspondingly looser. Cycle-17's 1.027× synthetic number on this
same target was below the floor — γ.1 measures whether the sonnet
0.31.x toolchain does better, with current shapes and quantised
weights, than cycle-17's earlier finding.

## 2. Pre-microbench projection (v1.7.26 discipline)

**Bucket:** 46.3% step time.
- linear.mlp: 34.9% (across 48 DeltaNet layers).
- full.mlp: 11.4% (across 16 full-attention layers).
- Same `Qwen3NextMLP` class in both layer types.

**Compile-reachable scope:** ≈ 100% of the MLP forward.
- `gate_proj(x)` — quantized matmul (B, T=1, hidden) → (B, T=1, intermediate).
- `up_proj(x)` — quantized matmul, same shape.
- `silu(gate) * up` — element-wise.
- `down_proj(...)` — quantized matmul (B, T=1, intermediate) → (B, T=1, hidden).
- No cache, no offset-dependent rope, no Python control flow inside
  the forward. `mx.compile` traces the entire path cleanly.

**Per-call gain:** unknown; γ.1 measures.
- Cycle-17 (synthetic, opus, mlx 0.31.x earlier): 1.027× → ~0.5%
  E2E. Below floor; retired at the time.
- Sonnet stack (mlx 0.31.1 pinned) on production shape with
  quantized weights: data point is missing.

**Projection table:**

| Per-call gain | × 100% reachable | × 46.3% bucket | E2E % step | vs β.4 3% gate |
| ---: | ---: | ---: | ---: | --- |
| 1.02× | 0.020 | 0.0093 | 0.93% | fails |
| 1.05× | 0.048 | 0.0220 | 2.20% | fails (1.36×) |
| 1.07× | 0.065 | 0.0303 | 3.03% | clears narrowly |
| 1.10× | 0.091 | 0.0420 | 4.20% | clears |
| 1.15× | 0.130 | 0.0604 | 6.04% | clears |

**γ KEEP threshold via projection: per-call gain ≥ 1.07×.** Below
that, the math projects below β.4's 3% E2E gate and γ closes
without integration.

## 3. γ.1 — microbench spike

| Field | Value |
| --- | --- |
| Goal | Measure mx.compile speedup on `Qwen3NextMLP.__call__` at production shape. Gate β.2 (γ-side: γ.2 integration) on per-call gain ≥ 1.07× with σ_ratio ≤ 0.03. |
| Output | `silica/bench/microbench/compiled_mlp.py` (parallels `compiled_attn_postcache.py`), JSONL + Markdown summary in `plans/P6_SMALL_B/GAMMA/microbench/`. |
| Input shape | B=4, T_q=1, hidden_size and intermediate_size from the loaded layer's MLP module (read at runtime; do not hardcode). dtype bfloat16 (the production decode dtype on this stack). |
| Three timing arms | (1) uncompiled, (2) `mx.compile(fn, shapeless=True)`, (3) `mx.compile(fn)`. |
| Measurement | n=20 iters, 3 warmup, median wall ms. |
| Variance protocol | n=2 back-to-back sessions per the α / β.1 cadence. Combined σ_ratio reported. |
| Pass gate | Best-of-arms speedup ≥ 1.07× AND σ_ratio ≤ 0.03 on the same arm. |
| Close-with-negative gate | Best-of-arms speedup < 1.07×, OR σ_ratio > 0.03 on every arm that hits the speedup. **Close γ. δ becomes the next sub-unit.** |
| Stop conditions | (i) Compile crashes — investigate before reporting. (ii) Numerical drift > 1e-4 max abs vs reference — investigate. |

`compiled_mlp.py` reuses the harness pattern from
`compiled_attn_postcache.py` (the find-layer + synth-inputs +
three-arm timing template); only the inner callable changes.

## 4. γ.2 – γ.5 — conditional, lightweight

Same shape as β.2-β.5 from `plans/P6_SMALL_B_BETA_OPENING.md` §4,
applied to `Qwen3NextMLP.__call__` instead of attention. Defined
inline only after γ.1 PASS to avoid sketching pipeline that does
not run.

If γ.1 PASSES the per-call gate AND the projection clears 3% E2E,
γ.2 wires `SILICA_USE_COMPILED_MLP` into `silica/kernels/shadow_install.py`
as a fourth env-flag patch site (default OFF). γ.3 attests greedy
parity. γ.4 does paired ON/OFF same-session E2E at B=4. γ.5 closes.

If γ.1 FAILS, γ closes; δ next.

## 5. Non-goals

Same as β (binding):

1. No new Metal kernels.
2. No spec-decode reopen.
3. No raising B beyond 12.
4. No `mx.compile` of attention (β closed it).
5. No relaxation of the determinism gate.
6. No relaxation of the variance discipline.
7. No β reopen without specific user authorization.

## 6. Acceptance gates (line-level)

γ closes via one of three terminations:

1. **γ.1 NEGATIVE.** Per-call gain < 1.07× or σ violation. Close;
   δ next. Most likely outcome per cycle-17 prior.
2. **γ.4 KEEP.** Paired ON/OFF B=4 per-row uplift ≥ 3% under
   variance discipline. γ.5 lands flag default-ON for chat-CLI
   warm path.
3. **γ.4 NEGATIVE.** Microbench passed but E2E fails the 3% gate.
   Close; δ next.

There is **no soft KEEP**, same as β.

## 7. Stop conditions (line-level)

Pause and surface to user before continuing if any of:

1. γ.1 result lands in escalate band (1.05×-1.07× with high σ).
   Run a third session before deciding; if escalate persists,
   commit fact-bundle (microbench JSONL + REPORT) per the
   escalate-band split-commit pattern.
2. γ.1 surprises with > 1.20× per-call (well above
   cycle-17's prior). Investigate before claiming KEEP — could
   reflect a measurement artefact rather than a real win.
3. γ.2 integration uncovers a quantized-weight compile interaction
   that produces parity drift > 1e-4 max abs. Investigate; do
   not declare γ.3 PASS.

## 8. Load-bearing references

- `plans/P6_SMALL_B_OPENING.md` §4 γ — original opening
  definition; this doc supersedes for actionable plan.
- `plans/P6_SMALL_B/REPORT.md` — α report; layer-internal data is
  the source for the bucket fraction.
- `plans/P6_SMALL_B/BETA/microbench/REPORT.md` — β.1 close
  precedent; γ.1 reuses the harness pattern and the bucket-scope
  projection discipline.
- `plans/PLAN.md` §9 D-022 + §13 v1.7.26 — disposition state.
- `plans/P6_AUTORESEARCH_NOTES.md` cycle 17 — original 1.027×
  synthetic claim on `Qwen3NextMLP` (~0.5% E2E, retired). γ.1
  reproduces on sonnet stack at production shape.
- `silica/bench/microbench/compiled_attn_postcache.py` — β.1
  harness; γ.1 borrows pattern.
