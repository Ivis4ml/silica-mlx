# Silica-MLX P-6 Autoresearch — take-home notes

| Field | Value |
| --- | --- |
| Mission opened | 2026-05-02 |
| Mission closed (high-B exploration) | 2026-05-04 (cycle 35) |
| Cycles run | 35 |
| Branch | `opus` |
| Hardware | Apple M5 Pro 48 GB unified memory, 307 GB/s peak bandwidth |
| Targets | Dense Qwen3.5-27B-4bit (primary); MoE Qwen3.5-35B-A3B-4bit (secondary) |
| Companion files | `AR.md` (directive); `plans/P6_AUTORESEARCH_LOG.tsv` (ledger); `plans/P6_AUTORESEARCH_PROGRESS.md` (index); `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_*.md` (per-cycle reports); `plans/P6_AUTORESEARCH_PROGRESS_*.png` (charts) |

This file is the durable take-home companion to `AR.md`. The mission was to push Silica-MLX as close as possible to the M5 Pro hardware limit on the production checkpoints. The note covers what we measured, what the levers are, what failed, and what's left for the next phase.

## TL;DR

| Track | Frame | Best measurement | Composition |
| --- | --- | ---: | --- |
| Dense 27B (primary) | within strict 36 GB envelope | **204 ± 1 tok/s at B=52** (4.85× cycle-1 baseline 42.17; n=6 across 2 sessions per cycle 33) | C10 axis-shift × C12 bf16 DeltaNet state (peak save) |
| Dense 27B (primary) | within 48 GB hardware ceiling | **231.9 ± 0.3 tok/s at B=64 bf16-only** (5.50× cycle-1; n=3 per cycle 28) | same stack at higher B; v10 attention kernel is within-noise to slightly negative at B=64 |
| MoE 35B-A3B (secondary) | within strict 36 GB envelope | **464.1 ± 0.7 tok/s at B=64** (peak 33.8 GB; 2.46× MoE B=4 baseline 188.5; n=3) | C12 bf16 state + C10 axis-shift transfer cleanly via shared `gated_delta` shadow patch |
| MoE 35B-A3B (secondary) | within 48 GB hardware ceiling | **791.8 ± 5.2 tok/s at B=128** (peak 47.96 GB; 4.20× MoE C1; n=3) | same stack; expert routing amortisation crosses utilisation threshold near B=128 |

The (1b) ≥60 tok/s milestone cleared 3.40× within strict envelope and 3.87× at hardware ceiling on the dense primary track. The 791.8 tok/s MoE secondary measurement is the largest absolute throughput observed in the 35-cycle effort.

## Mission framing (from `AR.md`)

- Push the production targets toward the hardware limit, not toward gate compliance. The (1b) milestone is a step on the way, not the ceiling.
- Decode throughput is the single primary metric. `B` (max batch size) is chosen to maximise aggregate per the AR.md metric definition. Improvements must clear ≥3σ on ≥2 reproductions before they enter the running-best line.
- Karpathy-style autoresearch ledger: every measurement is a TSV row. `plans/P6_AUTORESEARCH_LOG.tsv` is appended only by the main agent; subagents return findings, the main agent merges.
- Chunked-decode is structurally incompatible with the warm-decode oracle stability gate (cycles 4/5/12/15 — chunk=2 still fails). Do not propose chunked-decode again as a small-B lever.

## Cycle log (one-line per cycle)

| Cycle | Date | What we did | Outcome |
| ---: | --- | --- | --- |
| 1 | 2026-05-02 | Orientation: per-step decomposition, simple-kernel probe, baseline anchoring | Decomposition at B=4: DeltaNet 74.2% / full-attn 21.9% / overhead 4.0%. Fused gated-output kernel correctness PASS but 1.006× speedup. 42.17 tok/s formally proven NOT to be the chip ceiling. |
| 2 | 2026-05-02 | Architecture verification (48 DeltaNet + 16 full-attn; head_dim=256, GQA 24:4, attn_output_gate=true) | Hybrid pattern verified. Retires §6 hypothesis #6 (MTP weights ABSENT in production safetensors). |
| 3 | 2026-05-02 | QMM naive kernel | 0.66 ms vs mlx 0.45 ms (discard). |
| 4 | 2026-05-03 | Lazy-chain / chunked-decode probe | Chunked-decode fails warm-decode oracle stability gate. |
| 5 | 2026-05-03 | Batcher experiments | All discard. |
| 6 | 2026-05-03 | simdgroup MMA QMM v2 | 0.97 ms (discard). |
| 7 | 2026-05-03 | QMM v3-v6 tuning iterations | Best v6 = 0.83 ms; mlx 0.45 ms still wins. |
| 8 | 2026-05-03 | QMM v7-v11 (incl. v9 = 0.59 ms BEST silica) | Even silica's best (v9=0.59) loses to mlx 0.45. Cycle 16 confirms mlx qmv_quad already loads 4 uint32 per thread; QMM kernel arm closed. |
| 9 | 2026-05-03 | QMM v12-v13 final tuning | All discard. QMM kernel arm formally retired. |
| **10 ⭐** | 2026-05-03 | **BREAKTHROUGH** — re-read AR.md metric definition ("B chosen to maximise aggregate"); pulled the axis-shift lever | 42.17 → 193.9 tok/s at B=48 (4.60×). (1b) milestone cleared 3.23×. |
| 11 | 2026-05-04 | FA-decode kernel port (`flash_attention_decode_v10.py`): K-axis split + GQA-aware tile sharing + streaming online softmax + half4 vectorised loads + fused sigmoid output gate | Kernel-level beats mlx by 1.25-1.81× across T_kv ∈ {128, 256, 512, 1024}. **0% E2E delta at B=48** because attention is only 22% of step time. Cycle-12 dtype defect introduced here (silently skipped bf16 path). |
| 12 | 2026-05-04 | bf16 DeltaNet state probe; shadow_install wiring fix in `qwen3_5.py:from_hf_repo` | Greedy-decode token-ID parity verified. Direct E2E save at fixed B=48: 0%. **Indirect save: 3.5 GB peak-memory headroom.** Defect: shadow_install dtype check was `mx.float16` only — discovered 14 cycles later by Codex review. |
| **13 ⭐** | 2026-05-04 | Re-composed cycle-12's peak save with cycle-10's B-axis lever | **B=52 = 200.8 ± 1.5 envelope KEEP**; B=64 = 229.8 ± 2.0 hardware-ceiling KEEP. 18σ above C10. The "193 wall" was a B=48 cap, not a hardware wall. |
| 14 | 2026-05-04 | v10 + bf16 stack at B=52 / B=64; cycle-12 wiring fix should make v10 fire at last | Claimed +5.4 tok/s = 3.4σ KEEP at B=52 (206.2). **CYCLE 27 CORRECTION**: v10 was not firing due to the cycle-12 dtype defect; the small n=3 σ underestimated true variance ~±1.5 tok/s. Honest revised running-best: 204.5 at B=52 bf16-only. |
| 15 | 2026-05-04 | B=53 / chunk=2 / fused SwiGLU / SPLIT_K {64, 256} ablations on top of v10+bf16 stack | All within noise or below baseline. Local optimum at C14. |
| 16 | 2026-05-04 | Inspect mlx QMM source; mx.compile probe on synthetic chain and on Qwen3NextAttention forward | Confirmed mlx qmv_quad already loads 4 uint32 per thread (line 707). mx.compile synthetic 3-op chain: 0.91-0.94×; attention forward without cache mutation: 1.08×. Real cache-aware decode mutates `cache.update_and_fetch` which mx.compile can't trace. |
| 17 | 2026-05-04 | mx.compile on Qwen3NextMLP forward | 1.027× synthetic, projects to ~0.5% E2E (below noise floor). Discard. |
| 18 | 2026-05-04 | mx.compile on `mx.quantized_matmul` directly with weight/scales/biases as compile arguments | 1.019× synthetic, projects to ~1% E2E. Discard. |
| 19 | 2026-05-04 | C.5 / DDTree top-b coverage probe extended to b ∈ {1, 4, 8, 16, 32, 64, 128, 256, 512, 1000}; user authorised research loop to push accept rate from 9% to 40% | Coverage@64 = **0.4051** crosses user threshold. @1000 plateau 0.767 indicates ~23% off-distribution tail. |
| 20 | 2026-05-04 | Verify-cost vs k probe on Qwen3.5-27B-4bit at warm cache T_kv=128 | Highly sub-linear: k=1 → 60 ms, k=64 → 189 ms = 3.14× cost for 64× tokens. Tree-spec at b=64 looked computationally feasible. |
| 21 | 2026-05-04 | Drafter survey (Qwen3.5-{0.8B, 4B, 27B-3bit}) | All produce flat coverage curves; @1 ranges 6.3-8.0%. The 4-bit-target's argmax distribution is the structural ceiling regardless of drafter. Drafter arm closed. |
| 22 | 2026-05-04 | Design-only report — three paths: tree-spec build (5 days, +30% projected); distillation (multi-day); close research | Projection used B=1 verify cost (190 ms). Ungrounded. |
| 23 | 2026-05-04 | **CRITICAL NEGATIVE** — measure verify cost at B ∈ {1, 4, 16, 52} × k ∈ {1, 4, 16, 64} | B=52 k=64 = **8105 ms** (42× of B=1 k=64 = 189 ms). B and k cost dimensions multiply, not add. Tree-spec at B=52 produces 10 tok/s aggregate vs 206 plain. **Spec-decode arm closed with negative.** |
| 24 | 2026-05-04 | Codex (GPT-5.5) cross-review opened; uv.lock found resolving to known-bad mlx 0.31.2 / mlx-lm 0.31.3 / mlx-metal 0.31.2 stack | Pinned `pyproject.toml` to mlx 0.31.1 / mlx-lm 0.31.2 / mlx-metal 0.31.1. test_p2_preload_parity passes 3/3 after pin. |
| 25 | 2026-05-04 | B=52 reverify under uv path | 185.3 ± 1.9 tok/s, ~10% below conda mean. Codex's 185 was likely cold-start / env anomaly. |
| 26 | 2026-05-04 | Codex review found cycle-12 dtype defect: `shadow_install` checked `queries.dtype == mx.float16` but Qwen3.5-27B-4bit decode is bf16. Codex landed bf16-native v8/v10 sources (string substitution `half4` → `bfloat4`, `metal::dot(half4, half4)` → `metal::dot(float4(...), float4(...))`); separate kernel cache by dtype | v10 fires correctly post-merge (16 calls per 2-token decode). bf16 microbench: v10 vs mlx SDPA = 1.28-2.14×. |
| **27 ⭐** | 2026-05-04 | **CORRECTION CYCLE** — verify v10 firing on production decode; isolate v10 contribution at B=52 (n=8 reproductions) | bf16+v10 = 204.7 ± 1.2 (n=5); bf16-only = 204.2 ± 1.1 (n=3). v10 E2E contribution = **+0.5 tok/s = within noise**. Cycle-14's claimed 3.4σ KEEP retracted; honest running-best at B=52 is bf16 state alone (204.5 ± ~1.5). |
| 28 | 2026-05-04 | Hardware ceiling at B=64 re-measured under corrected v10 path (n=3 each for bf16+v10 and bf16-only) | bf16+v10 = 230.2 ± 1.6; bf16-only = **231.9 ± 0.3**. v10 marginally hurts (-1.7 tok/s, within noise). Hardware ceiling attribution corrected to bf16-only. |
| 29 | 2026-05-04 | Probe whether 40 GB peak cliff is movable via `mx.metal.set_cache_limit` / `set_memory_limit` / `set_wired_limit` | Three allocator-hint probes leave the cliff in place. **The cliff is architectural** (likely M5 Pro SLC threshold or unified memory bandwidth contention near 48 GB cap), not allocator policy. |
| 30 | 2026-05-04 | Per-step decomposition at B=64 with v10+bf16 stack | DeltaNet **87.9%** / full-attn 12.5% / overhead 0.3%. DeltaNet share grew from 74% (B=4) to 88% (B=64) because state R/W scales with B. Explains why FA-decode kernel had no E2E impact at production B. |
| 31 | 2026-05-04 | silica `gated_delta_v2` — bfloat4 K/V/Q + float4 state R/W vectorisation at production shape (Hk=16, Hv=48, Dk=Dv=128) | Microbench at B=64: **1.001× vs mlx**. Correctness PASS (3e-5 fp16 ULP). mlx's existing kernel is at near-HBM-bandwidth limit on state R/W; vectorisation changes load instruction count, not data volume. The cycle-11 v6→v7 trick does not transfer to DeltaNet. |
| 32 | 2026-05-04 | Re-render progress charts with cycles 25-31 + cycle 27 corrections | Chart bundle update only. |
| 33 | 2026-05-04 | Variance characterisation at B=52 — 3 additional bf16-only reps in fresh session, combined with cycle 27's 3 reps | n=6 across 2 sessions: combined 203.75 ± 0.83 tok/s. Within-session σ 0.4-1.1; between-session drift 0.9. **Honest running-best: 204 ± 1 tok/s.** |
| **34 ⭐** | 2026-05-04 | MoE 35B-A3B portability test: apply cycles 12+13 levers (bf16 state + axis-shift) via inherited `Qwen3_5MoeAdapter` and shared `gated_delta` shadow patch | NEW MoE secondary-track within-envelope KEEP: **464.4 tok/s at B=64** (peak 33.8 GB; +146% vs MoE B=4 baseline 188.5). Cycles 12+13 methodology generalises across the Qwen3.5 family. |
| **35 ⭐⭐** | 2026-05-04 | MoE B-axis push past cycle-34 cap with bf16 state lever; n=3 reps at B=64 (tightening to 464.1 ± 0.7) and B=128 | NEW MoE secondary-track running-best at hardware ceiling: **791.8 ± 5.2 tok/s at B=128** (peak 47.96 GB; n=3 reps 785.8/794.5/795.0). 4.20× MoE C1 baseline; 1.71× cycle-34 B=64 KEEP. Per-row throughput non-monotonic (7.25 → 4.87 → 6.18) reflects expert routing utilisation crossing amortisation threshold near B=128 (8 of 256 experts active per token; ~4 activations/expert/step at B=128 vs 2 at B=64). MoE has fundamentally different B-scaling than dense 27B because expert sparsity bypasses dense activation pressure that produces the 40 GB cliff. |

Stars (⭐) mark cycles that delivered or revised a running-best. Of the 35 cycles, 5 produced lasting load-bearing changes: C10, C13, C27 (correction), C34, C35.

## Take-home lessons

### 1. The right unit of analysis is *peak-memory ceiling × B-axis lever*, not isolated kernel bandwidth.

Cycles 1-9 spent at fixed B=4 with 0 keeps — wrong frame. Cycle 10 re-read AR.md's "B chosen to maximise aggregate" definition and gained 4.60× by extending B alone. The 23-cycle journey shows kernel/state probes that look flat in isolation (cycles 11/12 = 0% at fixed B=48) can be *resource feeders* that pay off when re-composed at a higher B (cycle 13 = +4.76×).

### 2. Project from measured baselines, not extrapolated ones.

Cycle 22 projected 270 tok/s with tree-spec from B=1 verify cost (190 ms). Cycle 23 measured B=52 verify cost and found 8105 ms — 42× higher. The B and k cost dimensions multiply, not add. Future loops must measure cost models at the *actual operating point* before designing on top of them.

### 3. Compositional wins dominate atomic wins.

Every single-lever probe in cycles 10-14 either landed flat or moved the running-best by a small margin in isolation. The big wins came from composition:
- C10 alone: +4.60×.
- C11 alone: 0%.
- C12 alone: 0% (but produced 3.5 GB peak save + shadow_install wiring fix).
- C13 = C12 peak save + C10 axis-shift extension: +4.76× envelope, +5.45× hardware.
- C14 = C13 + (claimed) C11 v10 stack: +4.89× envelope, +5.51× hardware (later retracted; v10 contribution within noise).

### 4. Take cross-reviews seriously; small n=3 σ underestimates run-to-run variance.

Codex's review caught the cycle-12 shadow_install dtype defect that had been silently skipping the v10 path for 14 cycles. The cycle-14 "+5.4 tok/s = 3.4σ KEEP" was attribution error: v10 wasn't firing, AND the n=3 within-session σ of 0.5 tok/s underestimated the true ~±1 tok/s run-to-run variance. **Within-session σ is not the same as combined σ.** Cycle 33 sharpened combined σ at B=52 to 0.83 tok/s with n=6 across 2 sessions.

### 5. Microbench wins do not automatically translate to E2E wins; check kernel-share at the production operating point.

Cycle 11 FA-decode kernel beats mlx 1.25-1.81× microbench. E2E delta at B=48: 0%. Cycle 30 explained: at B=4, full-attn is 22% of step; at B=64, full-attn is 12.5%. DeltaNet share grew from 74% to 88%. The kernel that wins is whichever owns the dominant share at the operating point.

Cycle 31 confirmed the inverse: DeltaNet at 88% step share is the bottleneck at B=64, but mlx's existing `gated_delta` kernel is already at near-HBM-bandwidth limit on state R/W. Vectorisation changes instruction count, not data volume. **Identifying the dominant cost is necessary but not sufficient to find a reachable lever.**

### 6. Architectural cliffs differ between dense and sparse architectures.

Dense 27B has a sharp 26% throughput drop at the B=64 → B=66 transition (40 GB peak boundary). Cycle 29 confirmed allocator hints don't move it — likely M5 Pro SLC threshold or unified memory bandwidth contention. The cliff is a property of dense activation pressure.

MoE 35B-A3B does not have the same cliff in the same place because only 8 of 256 experts are active per token; the active-weight footprint per token is much smaller. Cycle 35 found the opposite curve: per-row throughput is non-monotonic, with an amortisation threshold near B=128 where each expert sees ~4 activations per step. MoE rewards going larger when peak memory allows.

### 7. Some research arms genuinely close with a negative.

Spec-decode (cycles 19-23) is a textbook negative: the 40% accept rate is structurally feasible (cycle 19), but the B×k verify-cost wall makes it infeasible to translate into throughput at any B in {1, 4, 16, 52}. This is a measurement-anchored close, not a "we ran out of ideas" close. Re-opening requires a fundamentally different verify mechanism that breaks the B×k product, or mlx 0.32+ async-copy primitives that change the verify-cost shape.

## Closed threads (do not re-open without specific user authorization)

- **QMM custom kernels** — mlx's `qmv_quad` already loads 4 uint32 per thread (cycle 16 inspection of `mlx/include/mlx/backend/metal/kernels/quantized.h:707`). Even silica's best (v9 = 0.59 ms) loses to mlx 0.45 ms.
- **Spec-decode** (cycles 19-23) — closed by B×k verify-cost wall. The 270 tok/s aspirational target is not reachable on this stack via spec-decode at any B in {1, 4, 16, 52}.
- **Drafter swap** — Qwen3.5-{0.8B, 4B, 27B-3bit} all produce flat coverage curves; @1 ranges 6.3-8.0%. The 4-bit-target's argmax distribution is the structural ceiling regardless of drafter capacity / scale / family.
- **B-axis extension on dense 27B** — closed cycles 28-29 (40 GB architectural cliff). Hardware ceiling at B=64 = 231.9 ± 0.3 tok/s.
- **MoE B=144+ stretch probes** — explicitly retired by user 2026-05-04 in favor of small-B work.
- **Chunked-decode** — structurally incompatible with warm-decode oracle stability gate (cycles 4/5/12/15; chunk=2 still fails).
- **DeltaNet kernel vectorisation** — cycle 31 silica `gated_delta_v2` produced 1.001× vs mlx at B=64. mlx is already at bandwidth limit on state R/W. The cycle-11 v6→v7 trick does not transfer.

## Open levers / TODOs (carried forward)

| Lever | Expected gain | Cost | Status |
| --- | --- | --- | --- |
| **mx.compile graph-trace with cache rerouting** | ~5-10% E2E (cycle-16 microbench was 1.08× on attention without cache mutation) | 4-6 hours integration; split attention/DeltaNet `__call__` into pre-cache / post-cache halves | unauth, candidate for next phase |
| **Per-step dispatch overhead reduction at small B** | TBD — cycle-1 decomposition shows overhead is ~4% at B=4 vs 0.3% at B=64 | TBD | **Next research direction per 2026-05-04 user redirect.** |
| **MLX 0.32+ async-copy primitives** | unknown — `simdgroup_async_copy` not yet in source-string interface | 0 (external) | blocked on upstream; 0.31.2 broke determinism, wait for ≥0.33 |
| **Profile the 40 GB cliff under different mlx versions** | unknown | 1-2 hours | low priority since cycle 29 ruled out allocator hints |
| **Per-step decomposition at B=4 v10+bf16 stack** | identifies remaining time bucket at small B | 2-3 hours | candidate for next phase |
| **Distillation drafter training** | KD on 4-bit target → coverage@1 ~60-80% | multi-day | NOT recommended per cycle-23 (B×k wall blocks E2E gain regardless of drafter quality) |

The user's next research direction (2026-05-04) is **small-B speed**, not B-axis extension. Frame the next thread around the cycle-1 baseline (B=4 dense = 42.17 tok/s, B=4 MoE = 188.5 tok/s), not against the B=52 / B=128 keeps.

## Reproducibility recipes

### Dense 27B within-envelope running-best

```bash
SILICA_USE_BF16_DELTANET_STATE=1 \
  uv run --extra bench python -m silica.bench.cli run qwen3.5-27b-warm-decode-b52 \
  --output /tmp/repro_b52_bf16.jsonl
# Expect: ~204 tok/s, peak ~35.5 GB, status ok.
# n=3 reps recommended; combined σ ~1 tok/s.
```

### Dense 27B hardware-ceiling

```bash
SILICA_USE_BF16_DELTANET_STATE=1 \
  uv run --extra bench python -m silica.bench.cli run qwen3.5-27b-warm-decode-b64 \
  --output /tmp/repro_b64_bf16.jsonl
# Expect: ~232 tok/s, peak ~40 GB, status ok.
# v10 contribution at B=64 is within-noise to slightly negative; do NOT enable
# SILICA_USE_FA_DECODE_V10=1 for ceiling reproduction (cycle 28 finding).
```

### MoE 35B-A3B within-envelope KEEP

```bash
SILICA_USE_BF16_DELTANET_STATE=1 \
  uv run --extra bench python -m silica.bench.cli run qwen3.5-moe-35b-a3b-warm-decode-b64 \
  --output /tmp/repro_moe_b64.jsonl
# Expect: ~464 tok/s, peak ~33.8 GB, status ok.
# n=3 combined σ 0.7 tok/s.
```

### MoE 35B-A3B hardware-ceiling

```bash
SILICA_USE_BF16_DELTANET_STATE=1 \
  uv run --extra bench python -m silica.bench.cli run qwen3.5-moe-35b-a3b-warm-decode-b128 \
  --output /tmp/repro_moe_b128.jsonl
# Expect: ~791.8 tok/s, peak ~47.96 GB, status ok.
# Sits right at 48 GB hardware ceiling. n=3 combined σ 5.2 tok/s.
```

### Environment

- mlx 0.31.1 + mlx-lm 0.31.2 + mlx-metal 0.31.1 (pinned in `pyproject.toml` per cycle 24).
- mlx-metal 0.31.2 (or mlx-lm 0.31.3) breaks `tests/test_p2_preload_parity` argmax determinism. Do not upgrade until bisect or ≥0.33 release.
- Verified-good gates: `uv run pytest tests/test_p2_preload_parity.py -q` passes 3/3.

### Variance protocol (per cycle 33)

For any future re-verification of these numbers:
- Same `uv` environment (warm cache).
- ≥3 reps per session.
- ≥2 sessions for between-session drift estimate.
- Combined σ ≈ 1 tok/s on dense B=52 in same environment with proper warm cache.
- Cross-environment variance can be much larger (cycle 25 codex 185.3 anomaly).

## Files index

| File | Role |
| --- | --- |
| `AR.md` | The autoresearch directive + 2026-05-04 addendums (cycle-23 closure + cycles-28-35 update). |
| `plans/P6_AUTORESEARCH_NOTES.md` | This file — durable take-home companion. |
| `plans/P6_AUTORESEARCH_FINAL_REPORT.md` | Comprehensive 23-cycle write-up (written at cycle-23 closure; cycles 28-35 documented in this note). |
| `plans/P6_AUTORESEARCH_PROGRESS.md` | Short index over the ledger; running counts and last cycle. |
| `plans/P6_AUTORESEARCH_LOG.tsv` | Karpathy-style ledger; 110 rows. Authoritative measurement record. |
| `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_*.md` | Per-cycle reports (15 of 35 cycles have dedicated REPORT files; the rest are captured in the ledger). |
| `plans/P6_AUTORESEARCH_SUMMARY.png` | Multi-panel summary chart through cycle 35. |
| `plans/P6_AUTORESEARCH_PROGRESS_CYCLES.png` | Per-cycle deliverables + running-best trajectory (35 cycles, dense + MoE). |
| `plans/P6_AUTORESEARCH_PROGRESS_DECODE_TOK_S.png` | Karpathy-style ledger plot for dense `decode_tok_s` (38 experiments, 9 kept, running best 232.20). |
| `plans/P6_AUTORESEARCH_PROGRESS_MOE_DECODE_TOK_S.png` | Karpathy-style ledger plot for MoE `moe_decode_tok_s` (11 experiments, 2 kept, running best 791.80). |
| `plans/P6_AUTORESEARCH_PROGRESS_FA_KERNEL.png` | Cycle 11 FA-decode kernel ablation (silica vs mlx across T_kv). |
| `plans/P6_AUTORESEARCH_PROGRESS_QMM_KERNEL.png` | Cycle 7-9 QMM kernel tuning bars. |
| `plans/P6_AUTORESEARCH_PROGRESS_FA_BF16_SPEEDUP_T512.png` | Cycle 26 bf16 FA microbench chart. |
| `silica/kernels/flash_attention_decode_v10.py` | Production FA-decode entry; bf16-native after cycle 26 fix. |
| `silica/kernels/gated_delta_v2.py` | Cycle 31 silica DeltaNet kernel (parity with mlx; correctness-validated, not load-bearing). |
| `silica/kernels/shadow_install.py` | Env-flag-gated kernel installs. Hooks: `SILICA_USE_FA_DECODE_V10`, `SILICA_USE_BF16_DELTANET_STATE`. |
| `silica/bench/scenarios.py` | Scenario registry; warm-decode-b{N} factories for dense and MoE. |

## Closing notes

- The autoresearch loop reached a legitimate stop on the dense 27B primary track (cycle 28-29) and a legitimate stop on the MoE secondary track (cycle 35).
- Two of three AR.md "Stop conditions" cleared: (1) ≥60 tok/s milestone cleared 3.40-3.87×; (2) new running-best ≥3σ above 42.17 with clean lever attribution after cycle-27 correction.
- The cycle-12+13 methodology (bf16 DeltaNet state + axis-shift) generalises across architectures within the Qwen3.5 family, demonstrated by cycles 34-35 on MoE 35B-A3B.
- The next research direction is small-B speed per 2026-05-04 user redirect; B-axis extension and spec-decode threads are closed.
