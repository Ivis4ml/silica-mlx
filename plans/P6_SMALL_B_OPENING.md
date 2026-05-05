# P-6 small-B dispatch / latency line — opening

| Field | Value |
| --- | --- |
| Opening date | 2026-05-05 |
| Predecessor | P-6 (1a) and (1b) cleared at v1.7.23 via opus autoresearch C10 axis-shift x C12 bf16 DeltaNet recurrent state composition (`plans/P6_AUTORESEARCH_NOTES.md`) |
| Decision anchor | D-022 (`plans/PLAN.md` §9) |
| Branch | `sonnet` |
| Hardware | Apple M5 Pro 48 GB unified memory, 307 GB/s peak bandwidth |
| Targets | Dense Qwen3.5-27B-4bit (primary). MoE Qwen3.5-35B-A3B-4bit secondary, deferred — small-B work first lands on the dense path. |
| Toolchain pin | `mlx==0.31.1`, `mlx-lm==0.31.2`, `mlx-metal==0.31.1` (Darwin); see `pyproject.toml`. `tests/test_p2_preload_parity.py` is the determinism gate. |

This document is **actionable**, not orientation. Step 4 of the v1.7.24
opus -> sonnet integration imported the executable surface
(warm-decode-b{4,8,12} scenarios, attribution microbenches, slim
`silica.kernels.shadow_install` with the bf16 DeltaNet state hook); no
tool debt blocks α. Sub-units below name commands, gates, and stop
conditions directly.

## 1. TL;DR

After P-6 (1a) ≥40 tok/s cleared and (1b) ≥60 tok/s cleared 3.40-3.87×
(dense cycle-1 uplift is 4.85× within the 36 GB envelope) via the C10
axis-shift × C12 bf16 DeltaNet state composition, the next live
research direction is **small-B interactive quality-of-experience**:
closing the dispatch-overhead and attention-forward buckets that
dominate per-step time at small batch.

The cycle-1 B=4 step-share decomposition names two reachable buckets —
full-attention (22%) via `mx.compile` graph-trace with cache reroute,
and dispatch overhead (4%) via `mx.compile` MLP plus `mx.eval` cadence
hygiene. DeltaNet (74%) is documented as bandwidth-saturated on
mlx 0.31.x and out of scope (cycle 31 silica `gated_delta_v2`
microbench = 1.001x vs mlx).

α (sonnet-side baseline refresh) is unconditional. β / γ / δ open only
on α's evidence per the gates in §4.

## 2. Framing

**Goal: interactive single-row quality-of-experience.** "Small-B
latency / dispatch" means closing the per-step wall time at
B ∈ {1, 2, 4, 8, 12}. Throughput parity (B=4 user reaching the B=52
envelope rate of ~204 tok/s aggregate) is **not** the goal — per-row
throughput at B=4 (~10.5 tok/s/row from cycle-1 baseline 42.17 / 4) is
already higher than per-row at B=52 (~3.92 tok/s/row from 204 / 52).
The throughput-parity frame is structurally inverted; the honest target
is closing per-step time on a single-row workload.

**Cycle-1 B=4 step-share decomposition (load-bearing anchor):**

| Bucket | Share | Reachable lever | Notes |
| --- | ---: | --- | --- |
| DeltaNet | 74% | none on this stack | cycle 31 confirmed mlx `gated_delta` at HBM-bandwidth limit; vectorisation = 1.001x |
| Full-attention | 22% | `mx.compile` graph-trace with cache reroute (cycle 16: 1.08x microbench on attention forward without cache mutation) | E2E projection is **hypothesis**, not gate promise; depends on whether sonnet's bucket distribution matches cycle-1's |
| Dispatch overhead | 4% | `mx.compile` MLP (cycle 17: 1.027x synthetic ≈ 0.5% E2E), per-layer loop / `mx.eval` cadence | Mathematical ceiling ≈ 4% E2E (~1.7 tok/s on 42.17 baseline) |

**Cycle-30 B=64 decomposition (DeltaNet 88% / full-attn 12.5% /
overhead 0.3%) is not the small-B anchor.** That was the v10+bf16 stack
at hardware ceiling and explains why high-B kernel work did not move
the needle. Small-B has a fundamentally different bucket distribution.

**Variance discipline.** Cycle 27's 14-cycle attribution error (small
n=3 within-session σ underestimated the true ~±1.5 tok/s run-to-run
variance) is a **process** lesson. Every sub-unit in this line carries:
≥3 reps per session, ≥2 sessions, combined σ check before declaring a
baseline or a KEEP. Cycle 33 sharpened combined σ at B=52 to 0.83 tok/s
with n=6 across 2 sessions; small-B work treats that as the protocol
standard, not the exception.

## 3. Non-goals

1. **No new Metal kernels.** Per cycle 16 / 27 / 31, mlx 0.31.x is at
   bandwidth limit on dominant ops; source-string custom kernels on
   dense 27B do not pay back. Re-opening kernel work requires either
   an mlx 0.32+ async-copy upstream or measurement evidence that the
   bottleneck shifted.
2. **No spec-decode re-open.** Cycle 23 closed spec-decode at
   production B with measurement (B=52 k=64 verify cost = 8105 ms;
   tree-spec produces ~10 tok/s aggregate vs plain decode ~206 at the
   same B). Re-opening requires a fundamentally different verifier
   with measured sub-linear cost at production batch.
3. **No high-B axis extension as primary objective.** Cycles 28-29
   confirmed the 40 GB cliff at B=66 is architectural (allocator-hint
   probes leave it in place). Dense B>64 is not pursued under this
   line. (MoE B-axis is a separate question, deferred until / unless
   small-B work lands on the dense path.)
4. **No Tier-2 opus kernel imports without explicit user
   authorization.** v6 / v7 / `gated_delta_v2` /
   `fused_gated_output` / `fused_silu_mul` / `fused_qk_norm_rope` stay
   on opus. Step 4 deliberately landed only `flash_attention_decode_v10`
   (with v8 as private K-split dependency) and the slim
   `silica.kernels.shadow_install` surface. Future expansion of
   `silica/kernels/` requires a separate decision and explicit user
   ack.

## 4. Sub-units

α is unconditional. β / γ / δ open only on α's evidence per the
open-conditions below. ε is waitlist only.

### α — sonnet-side baseline refresh (unconditional)

| Field | Value |
| --- | --- |
| Bucket target | None (diagnostic; establishes sonnet timestamped reference) |
| Why | Cycle-30 step-share data is opus-side, taken under the v10+bf16 stack at B=64. sonnet at v1.7.24 has different commit ancestry and a different operating point (B=4). β / γ / δ engineering decisions must be grounded in fresh sonnet-side numbers, not extrapolated from opus. |
| Workloads | `qwen3.5-27b-warm-decode-b4`, `-b8`, `-b12` (registered in `silica.bench.scenarios`, dual-gated on `SILICA_REAL_QWEN3_5_27B`). |
| Attribution probes | `silica.bench.microbench.decode_step_attribution` at B=4 (mandatory) and B=8 (optional). `silica.bench.microbench.layer_internal_attribution` at B=4 (mandatory). |
| Variance protocol | n=3 reps per session, ≥2 sessions, combined σ check. **Decline to declare a baseline if combined σ > 1.5 tok/s without documented explanation.** |
| Shadow flags | Both `SILICA_USE_FA_DECODE_V10` and `SILICA_USE_BF16_DELTANET_STATE` default OFF for α's first measurement. Enabling either is a different baseline; record them as separate rows if compared. |
| Commands | See block below. |
| Gate | Establish a timestamped sonnet B=4 baseline plus per-step bucket decomposition. **No speed-up expected; this is a diagnostic.** Bucket distribution informs which sub-unit (β / γ / δ) makes sense to open next. |
| Stop conditions | (i) any combined σ > 1.5 tok/s across sessions — investigate environmental drift before declaring; (ii) bucket distribution shows DeltaNet ≥ 95% of step time (no reachable lever) — close P-6 small-B line entirely and document as bandwidth-saturated; (iii) bucket distribution shows full-attn < 10% AND overhead < 2% — same as (ii). |

```
SILICA_REAL_QWEN3_5_27B=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-27b-warm-decode-b4 \
        --out plans/P6_SMALL_B/baseline_b4.jsonl

SILICA_REAL_QWEN3_5_27B=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-27b-warm-decode-b8 \
        --out plans/P6_SMALL_B/baseline_b8.jsonl

SILICA_REAL_QWEN3_5_27B=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-27b-warm-decode-b12 \
        --out plans/P6_SMALL_B/baseline_b12.jsonl

SILICA_REAL_QWEN3_5_27B=1 \
    uv run python -m silica.bench.microbench.decode_step_attribution \
        --b 4 --warmup 3 --iters 20 \
        --out plans/P6_SMALL_B/decode_step_attr_b4.jsonl

SILICA_REAL_QWEN3_5_27B=1 \
    uv run python -m silica.bench.microbench.layer_internal_attribution \
        --b 4 --warmup 3 --iters 10 \
        --out plans/P6_SMALL_B/layer_internal_attr_b4.jsonl
```

Repeat per session. Aggregate combined σ across ≥2 sessions before
declaring α complete.

### β — attention `mx.compile` graph-trace with cache reroute (conditional)

| Field | Value |
| --- | --- |
| Bucket target | Full-attention (~22% at B=4 per cycle-1; sonnet-side share to be confirmed by α) |
| Why | Cycle 16 measured 1.08x synthetic on `Qwen3NextAttention.__call__` forward without cache mutation. Cache mutation (`cache.update_and_fetch`) is the integration friction that prevents naive `mx.compile` wrap. Lever: split `Qwen3NextAttention.__call__` into pre-cache (mutates) / post-cache (reads-only) halves and `mx.compile` only the post-cache half. Estimated 4-6 hour integration. |
| Open condition | α shows full-attn ≥ 15% of B=4 step time AND combined σ low enough that a 3% movement is detectable (combined σ ≤ ~1 tok/s). |
| Acceptance gate | (a) microbench ≥1.05x over un-compiled reference at B=4 attention forward with plausible E2E projection; OR (b) ≥3% E2E p50 improvement over α B=4 with combined σ check. **5-10% E2E is hypothesis, not gate promise.** |
| Stop conditions | Microbench < 1.02x (below cycle-1 noise floor) → close β as no-go and document. Cache-reroute integration breaks correctness on the spec-on warm-decode parity rows or on `tests/test_p2_preload_parity.py` → close β; document the friction. |

### γ — `mx.compile` on `Qwen3NextMLP` (low priority, negative-confirmation reverify)

| Field | Value |
| --- | --- |
| Bucket target | MLP within dispatch-overhead bucket (subset of 4%) |
| Why | Cycle 17 measured 1.027x synthetic ≈ 0.5% E2E (below cycle-1 noise floor). Closing this with a sonnet-side reverify converts "deferred per cycle 17" into "measured no-go on sonnet". |
| Open condition | α shows MLP-attributable share materially larger than cycle-17's frame, OR user explicitly requests negative-confirmation reverify. |
| Acceptance gate | E2E ≥1% p50 improvement on B=4 with combined σ check; otherwise document as no-go matching cycle 17. |
| Stop conditions | Reverify matches cycle-17 within noise (most likely outcome) → document and close. |

### δ — `mx.eval` cadence / per-layer loop sync hygiene

| Field | Value |
| --- | --- |
| Bucket target | Dispatch-overhead 4% bucket (Python-side sync barrier reduction) |
| Why | 64-layer Python loop with implicit `mx.eval` per attention block / per MLP. Lazy-graph snapshot capture (cycle-30 attribution probe pattern) might let several layers chain before `mx.eval`. Risk: low (Python-side change, no kernel surface); benefit: bounded by the 4% dispatch ceiling. |
| Open condition | α shows overhead bucket ≥ 3% of B=4 step time AND β / γ are exhausted or determined no-go. |
| Acceptance gate | E2E ≥1% p50 improvement on B=4. Strict variance discipline. |
| Stop conditions | Improvement below 1% or breaks `tests/test_p2_preload_parity.py` determinism → close. |

### ε — mlx 0.32+ async-copy primitives (waitlist; do not open)

| Field | Value |
| --- | --- |
| Bucket target | Potentially DeltaNet bucket, depending on which primitives upstream exposes. |
| Why | Cycle 24 pinned to mlx 0.31.1 because mlx-metal 0.31.2 broke `tests/test_p2_preload_parity`. mlx 0.33+ (when released) may unblock new async-copy capabilities relevant to DeltaNet state R/W or attention KV update. |
| Open condition | Upstream mlx ≥ 0.33 release that passes the `test_p2_preload_parity` gate AND exposes documented async-copy primitives. |
| Status | Tracking only; not active work. Delete this block if upstream determinism question remains unresolved at end-of-line. |

## 5. Load-bearing references

| Item | Path / form |
| --- | --- |
| Strategic anchor | `plans/PLAN.md` §9 D-022; §13 v1.7.24 changelog. |
| Predecessor evidence | `plans/P6_AUTORESEARCH_NOTES.md` § Take-home lessons §1-§7. Cycle-1 B=4 step-share is captured as `plans/P6_AUTORESEARCH/decode_step_attribution_b4.jsonl` (raw) and summarised in the take-home notes. Cycle-27 corrections (variance / attribution discipline) are at `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_27.md`. Cycle-30 contrast (B=64 step-share) is at `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_30.md`. Cycle-31 DeltaNet bandwidth-bound proof is at `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_31.md`. Cycle-33 variance protocol is at `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_33.md`. |
| Bench scenarios | `silica.bench.scenarios.BUILTIN_SCENARIOS` rows: `qwen3.5-27b-warm-decode-b{4,8,12}` (registered at v1.7.24 path-level extract from opus); each dual-gated on `SILICA_REAL_QWEN3_5_27B`. Higher-B rows up to B=80 also available for any sweep work that crosses out of the small-B band, but **the small-B line itself stays at B ∈ {1, 2, 4, 8, 12}** per §2 framing. |
| Attribution microbenches | `silica/bench/microbench/decode_step_attribution.py` (per-layer block barrier; cycle-30 pattern). `silica/bench/microbench/layer_internal_attribution.py` (per-child-module barrier; finer-grained intra-layer breakdown). Both gated on `SILICA_REAL_QWEN3_5_27B=1`; module imports do not load any model. |
| Shadow flags (already imported, default OFF) | `SILICA_USE_FA_DECODE_V10`, `SILICA_USE_BF16_DELTANET_STATE` via `silica.kernels.shadow_install`. The slim shadow-install surface (only these two flags, no fused / compiled / retired flags) is the v1.7.23 narrowing recorded in `plans/PLAN.md` §13. **α's first measurement runs both flags off** to match the cycle-1 frame. |
| Toolchain pin | `pyproject.toml` `mlx==0.31.1`, `mlx-lm==0.31.2`, `mlx-metal==0.31.1` (Darwin). The pin is tested by `tests/test_p2_preload_parity.py` — every sub-unit's acceptance gate carries that test in its toolchain attestation. |
| Non-goal #4 enforcement | `silica/kernels/__init__.py` exports only `flash_attention_decode_v10` and `shadow_install`. Any future sub-unit that imports a kernel beyond this surface must first land an explicit decision update naming the import. |
