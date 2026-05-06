# γ.1 — `mx.compile` on `Qwen3NextMLP` forward, sonnet refresh

Re-measures cycle-17's "Qwen3NextMLP `mx.compile` 1.027× synthetic ≈
0.5% E2E (discard)" claim on the sonnet stack at production shape.
Methodology and gate per `plans/P6_SMALL_B_GAMMA_OPENING.md` §3.

## Setup

- Repo: `mlx-community/Qwen3.5-27B-4bit` (cache-warm).
- Toolchain: mlx 0.31.1, mlx-lm 0.31.2, mlx-metal 0.31.1.
- Hardware: M5 Pro 48 GB unified memory.
- Layer: `model.layers[3]` (first `is_linear=False` full-attention block).
- Module: `Qwen3NextMLP` (gate / up / down `QuantizedLinear` 4-bit + SwiGLU).
- Shape config: B=4, T_q=1, hidden_size=5120, intermediate_size=17408,
  dtype=bfloat16.
- Three timing arms per session: uncompiled / `mx.compile(fn, shapeless=True)`
  / `mx.compile(fn)` (fixed-shape).
- 3 warmup iters + 20 measurement iters, median wall ms.
- Two back-to-back sessions per the α / β.1 cadence.

## Results — combined across 2 sessions

| arm | mean ms | σ ms | speedup mean ± σ | σ_ratio |
| --- | ---: | ---: | ---: | ---: |
| uncompiled | 1.103 | 0.023 | — | — |
| shapeless compile | 1.094 | 0.017 | **1.008 ± 0.006** | 0.005 |
| fixed-shape compile | 1.091 | 0.003 | **1.011 ± 0.024** | 0.024 |

Raw rows: `plans/P6_SMALL_B/GAMMA/microbench/20260506_095243/` (session 1)
and `.../20260506_095319/` (session 2), `compiled_mlp_b4.jsonl` each.

## γ.1 gate

| Gate | Threshold | Result | Status |
| --- | --- | --- | --- |
| Best per-call speedup | ≥ 1.07× | 1.011× (fixed arm) | **FAIL** |
| σ_ratio on best arm | ≤ 0.03 | 0.024 | PASS |
| Same-shape combined gate | speedup AND σ_ratio | speedup fails by 5× | **FAIL** |

**Verdict: clean close.** γ.1 cannot reach the per-call gate. σ is
tight (well under 0.03), so the failure is structural, not noise.
Cycle-17 reported 1.027× synthetic on an earlier stack; the sonnet
0.31.x toolchain at production shape with quantized weights gives
even less (1.008-1.011× depending on arm). The MLP forward is already
near-optimal at the kernel level: three quantized matmuls with their
own scales/biases tables that cannot fuse into one another, plus a
small element-wise SwiGLU that is bandwidth-trivial. `mx.compile` has
nothing to amortise.

## E2E projection — does γ.1 transfer to γ.4 KEEP?

The β.4 / γ.4 gate requires ≥ 3% E2E p50 improvement at B=4 paired
ON/OFF. MLP layer-block bucket per α: 46.3% of step time
(linear.mlp 34.9% + full.mlp 11.4%; `Qwen3NextMLP` is the same class
in both layer kinds). Reachable scope: ≈ 100% of MLP forward
(no cache, no shape mutation).

| Best per-call speedup | × 100% reachable | × 46.3% bucket | E2E % step | vs 3% gate |
| ---: | ---: | ---: | ---: | --- |
| 1.011× (γ.1 best) | 0.011 | 0.0051 | **0.51%** | **fails by 5.7×** |

The math projection is decisive: even granting full MLP-bucket
coverage, a 1.011× per-call gain produces 0.51% E2E. β.4 gate at 3%
is 5.7× higher.

## How γ relates to β

β closed because the reachable scope (post-cache `self_attn` ≈ 3-4%
step) was too narrow despite a respectable 1.05× per-call gain. γ
closes for the inverse reason: the reachable scope is wide (46.3%
step, full MLP forward) but the per-call gain is too small (1.01×).

Both close at single-customer E2E projection ≤ 1% step, both for
physics reasons that integration would not have repaired:

| sub-unit | reachable scope | per-call gain | E2E projection |
| --- | ---: | ---: | ---: |
| β | ~3-4% step | 1.05× | 0.25-0.41% |
| γ | 46.3% step | 1.01× | 0.51% |

Both fall in the "compile cannot move dense decode at small B"
attractor. The shared mechanism: production decode at B=4 is dominated
by quantized matmul kernels (`qmv_quad`, cycle-16-confirmed loading 4
uint32 per thread, near-optimal). `mx.compile`'s wins come from kernel
fusion + Python overhead reduction; quantized matmuls cannot fuse, and
the per-step Python overhead is small (3.6%). Different sub-unit
shapes, same physics ceiling.

## γ.1 disposition — recommendation

**Close γ with measurement-anchored negative.**

Reasoning:

1. The per-call gate fails by 5×; σ is tight, so failure is
   structural. γ.2/γ.3/γ.4 cannot bridge this gap.
2. E2E projection 0.51% × 46.3% bucket is at the same noise-floor
   regime as cycle-17's earlier 0.5% E2E discard, and the sonnet
   stack measured here is even smaller per-call than cycle-17's
   1.027×.
3. Per the v1.7.26 *bucket × reachable-scope × per-call-gain*
   discipline: with reachable-scope already maximised at 100%, the
   only remaining lever is per-call gain — and γ.1 directly measured
   that as 1.01×, structurally bounded by the kernel-level optimality
   of `qmv_quad`.

γ.1 closes without γ.2 (shadow_install integration). The β.1 module
(`silica/bench/microbench/compiled_attn_postcache.py`) and the γ.1
module (`silica/bench/microbench/compiled_mlp.py`) are retained as
documentation probes; future revisits will start from these harnesses.

## Next sub-unit: δ

Per the v1.7.25 sub-unit ordering, δ (`mx.eval` cadence / per-layer
loop sync hygiene) is the only D-022 lever that remains testable:

- δ targets the 3.6% overhead bucket (α dispatch attribution).
- Mathematical ceiling is 3.6% E2E ≈ 4 tok/s on the 41 tok/s baseline
  (per-row 10.29 → ~10.7 tok/s/row).
- Even at full theoretical recovery, δ does not clear β.4's 3% gate
  with margin; success would require recovering ≥ 83% of the overhead
  bucket.
- δ is Python-side cleanup, not a compile lever; the v1.7.26
  *bucket × reachable × gain* projection does not apply directly
  (the work is removing redundant `mx.eval`s, not amortising compute).

δ should open with explicit acknowledgement that even success
clears the gate by a thin margin; a δ pre-projection should
characterise which `mx.eval` sites are the load-bearing dispatchers
before any code change.

After δ closes (KEEP or NEGATIVE), the D-022 line itself closes per
`plans/P6_SMALL_B_OPENING.md` §6: with α complete + β closed +
γ closed + δ resolved, every conditionally-opened sub-unit reaches
either (a) measurement-anchored KEEP or (b) measurement-anchored
NEGATIVE.

## Cross-references

- `plans/P6_SMALL_B_GAMMA_OPENING.md` — γ opening, gates, sub-units.
- `silica/bench/microbench/compiled_mlp.py` — γ.1 measurement code.
- `plans/P6_SMALL_B/REPORT.md` — α report; layer-internal data is
  the source for the bucket fraction.
- `plans/P6_SMALL_B/BETA/microbench/REPORT.md` — β.1 close
  precedent; identical workflow.
- `plans/PLAN.md` §9 D-022 + §13 v1.7.26 — disposition state for β
  / γ pre-this-commit.
- `plans/P6_AUTORESEARCH_NOTES.md` cycle 17 — original 1.027×
  synthetic claim. γ.1 confirms with sharper per-call number on
  sonnet 0.31.x.
- v1.7.26 memory `feedback_p6_small_b_single_customer_gates.md` —
  the bucket × reachable-scope × per-call-gain rule that γ.1
  applies.
