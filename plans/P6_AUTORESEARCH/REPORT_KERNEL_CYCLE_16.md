# P-6 Autoresearch — sixteenth cycle (2026-05-04) — diminishing-returns confirmation; remaining levers identified

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Diagnostic — 4 probes against 3 unauthorized levers (mx.compile, QMM half4 retrofit, B=4 stack regression check). Confirms cycle-15 conclusion: local optimum on existing lever set. Further breakthrough requires speculative decoding (gated) or substantial multi-layer integration (mx.compile with cache rerouting). |
| User authorisation | "looks good and let us continue" |
| Companion docs | `REPORT_KERNEL_CYCLE_{10..15}.md` |

## TL;DR

| Probe | Result | Verdict |
| --- | --- | --- |
| mlx QMM source inspection | mlx's `qmv_quad` already uses 4-uint32-per-thread vectorization (`packs_per_thread = values_per_thread / pack_factor` = 4) | **QMM half4 retrofit cannot beat mlx** — same conclusion as cycles 7-9 |
| mx.compile small-op micro | 0.91-0.94× (slowdown) | mx.compile adds overhead at small op count |
| mx.compile attention forward (no cache mutation) | 1.08× | modest gain on synthetic, blocked by cache.update_and_fetch in real layer |
| v10+bf16 stack at B=4 | 41.1 tok/s vs cycle-1 baseline 42.17 | confirms stack helps only at high-B regime |

## Findings

**1. mlx's QMM kernels are already maximally vectorized.** Inspecting
`mlx/include/mlx/backend/metal/kernels/quantized.h` line 707:
`constexpr int packs_per_thread = values_per_thread / pack_factor;` —
each thread already loads 4 uint32 packs (= 16 bytes = 32 packed 4-bit
weights). This is the half4-equivalent vectorization the cycle-11 FA
work used to beat mlx. **mlx's QMM has no missing optimization for us
to exploit.** Cycles 7-9's "1.27× from mlx parity" was a real wall, not
a missed half4 opportunity.

**2. mx.compile produces marginal gains.** On synthetic 3-op chains it
is 0.91-0.94× (slower than uncompiled — compile overhead exceeds the
savings). On a Qwen3NextAttention forward without cache mutation it is
1.08× (modest gain, comes from fusing post-cache elementwise ops:
sigmoid + multiply + transpose). The real cache-aware decode layer
mutates `cache.update_and_fetch` in the middle of the forward; mx.compile
cannot trace through that mutation. Splitting the forward into pre-cache
and post-cache halves and compiling each is a substantial 4-6 hour
integration effort for a maybe-2% E2E gain.

**3. The cycles 11-14 stack is regime-specific.** Running v10+bf16 at
B=4 (cycle-1 baseline) gives 41.1 tok/s vs the original baseline 42.17.
A 2.5% regression. The kernel + state optimizations are net-positive at
the high-B regime where their savings dominate, but slightly net-negative
at low B where their dispatch overhead dominates. This is consistent
with cycle-12's "v10 alone at B=48 = no E2E delta" finding — v10
becomes load-bearing only at the cycles 13-14 axis-extension to
B=52..64 where the bottleneck shifts.

## Remaining levers and their cost / risk

| Lever | Expected gain | Cost | Authorization |
| --- | --- | --- | --- |
| Speculative decoding (D-021 framework) | Up to 2× per-step throughput at high accept rate; cycle-1 αprobes showed α≈0.09 with current drafters | small framework integration; needs C.5 γ.1 read-only survey decision | **gated** |
| mx.compile with cache rerouting | ~2-5% E2E based on micro | 4-6 hour integration + per-Qwen3-layer correctness validation | unauth |
| QMM half4 retrofit | ≤0% (mlx already vectorized) | substantial rewrite | unauth (and unprofitable) |
| Profile mlx memory cliff at 40 GB | 0% direct (diagnostic) | 1-2 hours | unauth |
| Custom mx-graph-traced decode loop | unknown — not tested | substantial (full re-implementation of decode loop) | unauth |

## Cycles 11-15 retrospective: what produced the wins

The 4.60× cycle-1 → cycle-10 jump (axis-shift to B=48) and the 1.07×
cycle-10 → cycle-14 lift (composition of v10 + bf16 + axis-extension)
came from **structural moves at the right level of abstraction**:

- **Cycle 10**: re-read P6_AUTORESEARCH.md's metric definition; pulled the axis-shift
  lever P6_AUTORESEARCH.md actually specified ("B chosen to maximise aggregate within
  envelope"). The cycle-1-9 kernel work was solving the wrong frame.
- **Cycles 11-12**: discovered new resources (FA-decode kernel, bf16-state
  correctness, peak-memory headroom, shadow-install wiring) that looked
  flat in isolation.
- **Cycle 13**: re-composed cycle-12's resources with cycle-10's lever to
  unlock B=52..64 — the "indirect win" pattern.
- **Cycle 14**: stacked cycle-11's v10 on top once the bottleneck
  relocated to a regime where attention savings show up E2E.

What did NOT produce wins: kernel-level tuning at fixed B (cycles 1-9,
+5% probes within cycle 15), QMM optimization (cycles 7-9 retired),
chunked decode (cycles 4/5/12/15 — fails stability gate every time).

The lesson: at this stage the autoresearch loop's productivity is
concentrated in **finding compositions** of existing resources, not
producing new individual ones. Cycle 15 + 16 confirm we've found the
local optimum compositions on the current resource set.

## Files added in cycle 16

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_16.md` — this file
- `/tmp/probe_mx_compile.py`, `/tmp/probe_compile_layer.py`, `/tmp/b4_combined_run1.jsonl`

## Ledger rows added cycle 16

- `AR_C16_QMM_HALF4_RETIRED` (diagnostic) — confirmed mlx's QMM is
  already vectorized; half4 retrofit cannot beat mlx. Cycles 7-9
  retirement of QMM kernels stands.
- `AR_C16_MX_COMPILE_PROBE` (diagnostic) — mx.compile on 3-op chain is
  0.91-0.94×; on full attention without cache mutation is 1.08×.
  Cache-mutation pattern blocks broader application; would need pre/post
  split + integration effort beyond the cycle-15-16 probe round.
- `AR_C16_B4_STACK_REGRESSION` (diagnostic, 41.1 vs 42.17) — v10+bf16
  stack regresses 2.5% at B=4. Confirms the stack is regime-specific to
  high-B aggregate-amortization.

## Cycle 16 takeaway

The autoresearch loop has reached a stable local optimum:
- **Within strict 36 GB envelope: 206.2 ± 0.5 tok/s (4.89× cycle-1)**
- **Within 48 GB hardware ceiling: 232.2 ± 0.3 tok/s (5.51× cycle-1)**

(1b) ≥60 milestone: CLEARED 3.44× (envelope) / 3.87× (ceiling).

The "search and probe" phase of the autoresearch loop is closing. The
remaining levers all require larger-than-cycle commitments: speculative
decoding (framework integration + accept-rate tuning), mx.compile graph
trace (multi-layer integration + cache rerouting), or external upgrade
(MLX 0.32+ async-copy primitives become accessible — not yet in
0.31.x).

The autoresearch loop's deliverable is in a clean state for hand-off:
9 KEEPs on the running-best ladder, 30 diagnostic-class probes, 21
discards, 0 crashes; charts and ledger up-to-date; tests at 2779 pass.
