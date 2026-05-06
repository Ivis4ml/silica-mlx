# P-6 small-B sub-unit β — attention `mx.compile` graph-trace with cache reroute

| Field | Value |
| --- | --- |
| Opening date | 2026-05-06 |
| Predecessor | D-022 sub-unit α complete at v1.7.25 (`plans/P6_SMALL_B/REPORT.md`) — full-attn 21.6% of B=4 step time clears the open-condition (≥15%). |
| Decision anchor | D-022 (`plans/PLAN.md` §9 + §13 v1.7.25 changelog). |
| Sibling sub-units | γ (`mx.compile` on `Qwen3NextMLP`) and δ (`mx.eval` cadence / per-layer loop sync) are also open. β goes first per the v1.7.25 sub-unit ordering. ε (mlx 0.32+ async-copy) remains waitlist. |
| Branch | `sonnet` |
| Toolchain pin | `mlx==0.31.1`, `mlx-lm==0.31.2`, `mlx-metal==0.31.1`. `tests/test_p2_preload_parity.py` is the determinism gate. |
| Targets | Dense Qwen3.5-27B-4bit (primary). Qwen3.5-0.8B (parity / smoke; cached locally). |

This document is **actionable**, not orientation. The cycle-1 frame
applies; α confirmed the bucket distribution transfers. Sub-units below
name commands, gates, and stop conditions directly.

## 1. TL;DR

Full-attention layers own 21.6% of B=4 step time per α (decode_step
attribution, mean across 2 sessions). The cycle-1 microbench
(cycle 16, opus) showed 1.08× on attention forward when cache
mutation is excluded from the traced region. β tests whether that
microbench win translates to single-customer E2E gain.

**Hypothesis:** splitting `Qwen3NextAttention.__call__` into a
pre-cache half (Q/K/V projections + norms + transpose), a cache-mutation
section (`rope(offset)` + `update_and_fetch`), and a post-cache half
(SDPA + transpose + reshape + sigmoid gate + `o_proj`), then `mx.compile`-ing
only the post-cache half, yields ≥3% E2E p50 improvement at B=4 with
no parity regression.

**Strategic frame:** β is evaluated on **single-customer metrics** per
the v1.7.25 D-022 framing: B=4 per-row tok/s, decode-step `step_total`
ms, correctness/PPL/parity, and `mx.compile` warmup cost. Aggregate
tok/s is diagnostic context only; it is not load-bearing for this
sub-unit.

**Five sub-sub-units (β.1–β.5) gated sequentially.** β.1 closes β with
a measurement-anchored negative if the standalone microbench fails to
reach ≥1.05×. Otherwise β.2-β.5 land integration, parity, KEEP
measurement, and closure.

## 2. Framing

**Why this lever, why now.**

α confirmed cycle-1's step decomposition transfers to sonnet (DeltaNet
75% / full-attn 22% / overhead 4%). DeltaNet is bandwidth-saturated on
mlx 0.31 (cycle 31 `gated_delta_v2` = 1.001× vs mlx). The full-attn
22% bucket is the largest reachable lever at B=4; β attacks it via
`mx.compile` graph-tracing with a cache-aware split.

The split point exists because cache mutation is the obstacle to
end-to-end attention compilation. `cache.update_and_fetch` mutates a
mutable Python object and depends on `cache.offset`, which `mx.compile`
cannot trace cleanly. Cycle 16's microbench resolved this by splitting
attention into a "pre-cache" pure-compute region, a "cache touch"
Python-side region, and a "post-cache" pure-compute region. The cycle-16
result was 1.08× on the post-cache half alone (no E2E integration).

β picks up cycle 16's microbench claim at the integration point: does
the 1.08× transfer to E2E B=4 single-customer per-row throughput, after
paying the `mx.compile` warmup cost and routing through
`Qwen3NextAttention.__call__`?

**The split, in current code (`silica/kernels/shadow_install.py:80-136`
v10 path; same shape in the original mlx-lm `qwen3_next.py`).**

```
def __call__(self, x, mask, cache):
    # --- pre-cache: pure compute on x, weights -----------------------
    q_proj_output = self.q_proj(x)
    queries, gate = mx.split(q_proj_output.reshape(...), 2, axis=-1)
    gate_flat = gate.reshape(B, L, -1)
    keys, values = self.k_proj(x), self.v_proj(x)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(...)).transpose(0, 2, 1, 3)
    values = values.reshape(...).transpose(0, 2, 1, 3)

    # --- cache touch: Python-side, not compiled ----------------------
    queries = self.rope(queries, offset=cache.offset)
    keys = self.rope(keys, offset=cache.offset)
    keys, values = cache.update_and_fetch(keys, values)

    # --- post-cache: pure compute on (q, K, V, gate, mask) -----------
    output = scaled_dot_product_attention(queries, keys, values, ...)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    gated = output * mx.sigmoid(gate_flat)
    return self.o_proj(gated)
```

The post-cache half is the cycle-16 target. Its inputs at B=4 decode
(T_q=1) have stable shapes per axis except T_kv (= prefill_len + step
index), which grows by 1 per step. Compile-shape recompilation
behaviour at variable T_kv is the second-order concern β.1 must
characterise before claiming a microbench win is real.

**Why compile only the post-cache half, not the whole `__call__`.**

1. Pre-cache compute is small relative to post-cache (matmul on (B, L,
   d_model) vs SDPA over (B, h, T_q, T_kv) plus o_proj). Post-cache
   dominates the bucket.
2. Pre-cache shapes are static at T_q=1; pre-cache compilation is a β
   stretch, not a primary lever, and a separate compile boundary
   means we can land β.2 on post-cache only without touching pre-cache.
3. The cache touch's Python control flow (`if cache is not None`) and
   its read of mutable `cache.offset` cannot be traced as-is. Pulling
   it into the compiled region requires `mx.compile`-friendly cache
   protocol changes that are out of scope here.

## 3. Non-goals

**Binding for the duration of β.**

1. **No new Metal kernels.** β is `mx.compile`-only. The existing
   `silica/kernels/` surface (v10 FA-decode + slim shadow_install)
   stays unchanged. Any Metal-kernel idea found while writing β is
   filed; it does not enter β's commit train.
2. **No spec-decode reopen.** Track C remains closed (v1.7.20-22).
3. **No raising B.** β's measurement frame is B=4. B=8 / B=12 numbers
   from α are diagnostic context; they are not β's gate.
4. **No `mx.compile` of pre-cache or cache-touch.** Out of scope; the
   compile boundary is the post-cache half.
5. **No γ / δ work in β commits.** γ and δ have their own openings.
6. **No Tier-2 opus kernel imports** (e.g., v6/v7 FA, `gated_delta_v2`,
   fused-op kernels). The kernel surface stays at v10 + shadow_install.
7. **No relaxation of the determinism gate.** Every β commit must
   preserve `tests/test_p2_preload_parity.py` 3/3 PASS.
8. **No relaxation of the variance discipline.** Cycle-27's lesson
   (within-session σ underestimated cross-session drift, which led to
   the cycle-14 retraction) applies. KEEP claims need n=3 × ≥2 sessions
   with combined σ ≤ 1.5 tok/s.

## 4. Sub-sub-units (β.1 – β.5)

Sequential gating. Each sub-sub-unit's gate must clear before the next
opens. β.1 has the close-β-with-negative authority.

### β.1 — standalone microbench spike

| Field | Value |
| --- | --- |
| Goal | Reproduce cycle 16's 1.08× microbench claim on sonnet, with current toolchain (`mlx==0.31.1`), at production attention shapes (B=4, num_q_heads=24, num_kv_heads=4, head_dim=256, T_q=1, T_kv ∈ {128, 1024, 4096}). Characterise compile-shape recompilation behaviour at variable T_kv. |
| Output | `silica/bench/microbench/compiled_attn_postcache.py` (new module, parallels `decode_step_attribution.py`'s harness pattern), JSONL + Markdown summary in `plans/P6_SMALL_B/BETA/microbench/`. |
| Measurement | n=20 iters per shape, 3 warmup iters, median wall ms. Compare uncompiled vs `mx.compile`-wrapped post-cache callable. |
| Toolchain | `SILICA_REAL_QWEN3_5_27B=1` for shape resolution. Both shadow flags default OFF. |
| Variance protocol | n=3 reps per session, ≥2 sessions, combined σ check on the speedup ratio. |
| Pass gate | Median speedup ≥1.05× on at least one of T_kv ∈ {128, 1024, 4096} with combined σ_ratio ≤ 0.03. |
| Close-with-negative gate | Median speedup < 1.05× across all three T_kv values, OR compile-shape recompilation cost exceeds break-even at decode cadence. **Close β.** Document and move to γ. |
| Stop conditions | (i) Compile crashes on the post-cache callable at any production shape — investigate before β.2; (ii) numerical drift > 1e-4 max abs vs reference — investigate before β.2. |

### β.2 — shadow_install integration

Open only on β.1 PASS.

| Field | Value |
| --- | --- |
| Goal | Wire the compiled post-cache callable into `silica.kernels.shadow_install` as a third opt-in patch, gated on `SILICA_USE_COMPILED_ATTN_POSTCACHE`. Cache the compiled callable per shape signature; first call pays compile cost. |
| Output | Edit `silica/kernels/shadow_install.py` (add the env flag + patch), `tests/test_shadow_install_compiled_attn.py` (new tests parallel to existing shadow_install tests). |
| Default | OFF. Mutually compatible with `SILICA_USE_FA_DECODE_V10` and `SILICA_USE_BF16_DELTANET_STATE` (orthogonal axes). |
| Measurement | Single-shape smoke: bench `qwen3.5-27b-warm-decode-b1` with the flag ON, confirm no crash + reasonable wall time. |
| Pass gate | Smoke completes in ≤1.5× the OFF wall time (warmup-cost ceiling), no parity drift > 1e-4 vs OFF, `tests/test_p2_preload_parity.py` 3/3 PASS, `ruff` + `mypy` clean. |
| Stop conditions | (i) Compile recompilation triggers more than once per (shape, layer) tuple in steady state — fix the cache-key before β.3; (ii) memory regression > 5% at peak — investigate before β.3. |

### β.3 — parity attestation

Open only on β.2 PASS.

| Field | Value |
| --- | --- |
| Goal | Cached-checkpoint greedy parity (token streams identical) + slice-PPL parity (no regression). |
| Workloads | `qwen3.5-0.8b-b1-parity` smoke (cached `Qwen/Qwen3.5-0.8B`, 0.8B is faster to validate than 27B), then `qwen3.5-27b-smoke` greedy parity at B=1. WikiText-2 PPL on a 16-row slice via `qwen3.5-27b-wikitext-ppl-4bit --seeds 0` (one rep). |
| Pass gate | Token-by-token identical greedy output ON vs OFF on both checkpoints; |ΔPPL| < 0.01 on the slice; `test_p2_preload_parity.py` 3/3 PASS. |
| Stop conditions | (i) Any token divergence — **immediate close β with parity-FAIL**, file under `plans/P6_SMALL_B/BETA/parity_close.md`; (ii) PPL drift ≥ 0.01 — same. The 0.5-PPL P-5 codec bound does NOT apply here; β is a compute-graph rewrite, not a quantisation change, and any drift signals a real arithmetic divergence. |

### β.4 — full-stack KEEP measurement

Open only on β.3 PASS.

| Field | Value |
| --- | --- |
| Goal | Decide whether β earns a KEEP via single-customer E2E gain at B=4. |
| Workloads | `qwen3.5-27b-warm-decode-b4` paired ON vs OFF in the same sessions, with `decode_step_attribution --b 4` to break out where the time went. Optional: B=8, B=12 for diagnostic context. |
| Variance protocol | n=3 reps per session via `--seeds 0,1,2`, ≥2 sessions, ideally separated cross-day per α's caveat. Combined σ ≤ 1.5 tok/s on aggregate; per-row σ proportional. |
| Pass gate (single-customer) | **B=4 per-row tok/s rises ≥3% in paired ON/OFF comparison, AND combined σ check passes.** The v1.7.25 sonnet baseline 10.29 ± 0.16 tok/s/row gives the absolute sanity target (≥10.60 tok/s/row) when OFF reproduces α within variance. Step_total median (from attribution) drops by a corroborating margin. |
| Close-with-negative gate | <3% per-row improvement after compile warmup is amortised, OR step_total median does not corroborate, OR combined σ > 1.5 tok/s. Document, close β, move to γ. |
| Compile warmup budget | First-call compile cost is acceptable up to 30 s on cold start. If cold-start latency exceeds 30 s, β.4 closes; warmup is too expensive for serving cold-start SLA. |
| Stop conditions | (i) Compile recompilation per step — bug, fix before declaring; (ii) memory regression > 5% — bug, fix before declaring; (iii) variance protocol failure — re-run before declaring. |

### β.5 — closure

Open after β.4 verdict (KEEP or NEGATIVE).

| Field | Value |
| --- | --- |
| KEEP path | Land `SILICA_USE_COMPILED_ATTN_POSTCACHE=1` as the chat-CLI warm-path default (REPL spawns sub-process with the env set; serving cold-start uses default OFF). Update `plans/P6_SMALL_B/BETA/REPORT.md`, sync `plans/PLAN.md` §9 D-022 Status block + §13 changelog v1.7.26, update memory to mark β CLOSED-KEEP. |
| NEGATIVE path | Land `plans/P6_SMALL_B/BETA/REPORT.md` with the measurement-anchored negative; sync PLAN + memory; do NOT land the env flag. γ becomes the next sub-unit. |
| Decision artefact | `plans/P6_SMALL_B/BETA/REPORT.md` is the single source of truth for the verdict; it cites β.1 microbench, β.3 parity, and β.4 E2E rows by file path + measurement. |

## 5. Acceptance gates (line-level summary)

The β line CLOSES via one of four terminations:

1. **β.1 measurement-anchored NEGATIVE.** Microbench < 1.05× on every
   tested shape. Cycle 16's claim does not transfer; close β; γ next.
2. **β.3 parity FAIL.** Any token divergence or PPL drift ≥ 0.01.
   Close β with parity report; γ next.
3. **β.4 KEEP.** B=4 paired ON/OFF per-row uplift ≥3% under the
   variance discipline, with ≥10.60 tok/s as the absolute sanity target
   when OFF reproduces α; β.5 lands the flag default-ON for chat-CLI
   warm path.
4. **β.4 measurement-anchored NEGATIVE.** <3% per-row uplift or
   step_total / σ corroboration fails. Close β with E2E negative
   report; γ next.

There is **no soft KEEP**. Aggregate gain that does not reach the
single-customer per-row gate is not load-bearing for D-022 and does
not earn a KEEP at this line.

## 6. Stop conditions (line-level)

Pause and surface to user before continuing if any of:

1. β.1 microbench result lands in escalate band (1.03×-1.05× with
   high σ). Run a third session before deciding; if escalate persists,
   commit fact-bundle (microbench JSONL + REPORT) per the
   escalate-band split-commit pattern, and pause for user verdict.
2. β.2 integration uncovers a `Qwen3NextAttention` shape that the
   cycle-1 frame did not anticipate (e.g., MoE has a different
   attention internal). β does not target MoE; if MoE shapes appear in
   the patch path, gate the patch on dense-only and document.
3. β.3 surfaces a parity drift > 1e-4 max abs that cannot be
   attributed to fp16 reduction-order noise. Investigate before
   declaring parity FAIL — the failure mode is informative.
4. β.4 cold-start compile cost > 30 s. Document and pause; β-default-ON
   for chat-CLI may need a different deployment path (e.g., warm-up
   on REPL launch).

## 7. Load-bearing references

- `plans/P6_SMALL_B_OPENING.md` §4 β — opening definition, gate, stop
  conditions.
- `plans/P6_SMALL_B/REPORT.md` — α report (sonnet baseline +
  decomposition + sub-unit verdict block).
- `plans/PLAN.md` §9 D-022 + §13 v1.7.25 changelog — decision context.
- `plans/P6_AUTORESEARCH_NOTES.md` cycle 16 — original 1.08× microbench
  claim on attention forward without cache mutation. β.1 reproduces.
- `silica/kernels/shadow_install.py:77-138` — current
  `Qwen3NextAttention.__call__` shape under `SILICA_USE_FA_DECODE_V10`;
  β.2 patches the same call site under a new env flag.
- `silica/bench/microbench/decode_step_attribution.py` — harness
  pattern β.1 should mirror.
- `silica/bench/microbench/layer_internal_attribution.py` — same.
- `plans/P6_SMALL_B/aggregate_variance.py` — combined-session aggregator
  reused for β.4.
- `plans/P6_SMALL_B/RUNBOOK.md` — per-session command pattern; β.4
  follows the same shape with the env flag toggled.
