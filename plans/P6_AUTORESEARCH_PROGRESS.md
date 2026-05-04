# P-6 Autoresearch Progress

Short human-readable index over `plans/P6_AUTORESEARCH_LOG.tsv`. Updated by the main agent after each experiment.

| Field | Value |
| --- | --- |
| Date opened | 2026-05-02 |
| Last cycle | 2026-05-04 cycle 23 — **spec-decode arm closes with a critical negative**. Verify cost at B=52 k=64 = 8105 ms (42× B=1 k=64 of 189 ms). Cycle-22's 270 tok/s projection used B=1 cost; reality is B×k product scaling. Spec-decode at NO B regime (1..52) beats plain decode on this stack. The 40% accept rate is structurally feasible (cycle 19) but doesn't translate to throughput gain. |
| Loop status | (1b) milestone CLEARED 3.44× / 3.87× by cycle 14. Cycles 15-18 confirmed local optimum on existing kernel/state levers. Cycles 19-23 ran user-authorized spec-decode research loop: identified coverage@64 = 40.5% (feasibility), then found B×k verify-cost wall makes spec-decode infeasible at production B. The C14 v10+bf16 stack at 206/232 stands as the load-bearing deliverable. Further breakthrough beyond 232 needs mlx 0.32+ async-copy (blocked by 0.31.2 determinism break) or mx.compile + cache rerouting (4-6 hour integration). |
| Last experiment id | `AR_C23_BATCHED_VERIFY_COST_DEAD_END` (discard — closes spec-decode arm). Cycles 19-23 spec-decode thread: `AR_C19_EXTENDED_COVERAGE`, `AR_C20_VERIFY_COST_SUBLINEAR`, `AR_C21_DRAFTER_SURVEY_CLOSED`, `AR_C22_TREE_SPEC_DESIGN`, `AR_C23_BATCHED_VERIFY_COST_DEAD_END`. |
| Total experiments | 70 (cycle 10..23 + mlx upgrade discard) |
| Kept improvements on `decode_tok_s` running-best line | **9** (C10: B=12/32/40/44/48; C13: B=52 envelope, B=64 ceiling; C14: B=52 v10+bf16 envelope, B=64 v10+bf16 ceiling). Cycle 15: zero new keeps — local optimum confirmed. |
| Discarded experiments | 21 (cycles 7-9 kernel regressions, mlx 0.31.2 upgrade, cliff regressions, plus cycle-15 chunk2/silu/SPLIT_K) |
| Diagnostic-only experiments | 38 (cycles 11-22 + cycle 18 compiled-MLP variant). Cycle 23 = discard. |
| Discarded experiments | 22 (cycles 7-9 kernels, mlx 0.31.2 upgrade, B=72/B=66/B=68 cliff, cycle-15 chunk2/silu/SPLIT_K, cycle-23 spec-decode) |
| Crashes | 0 |
| Running best (`decode_tok_s` on dense `qwen3.5-27b-warm-decode-*` row family) | **206.2 ± 0.5 tok/s at B=52 with v10+bf16 stack** (within 36 GB envelope, 4.89× cycle-1). Demonstrated ceiling **232.2 ± 0.3 tok/s at B=64 v10+bf16** within 48 GB hardware (5.51× cycle-1). |
| Stretch milestone | 60 tok/s ((1b)) — **CLEARED by 3.44× (within envelope) / 3.87× (at hardware ceiling)** |
| 3σ floor for next keep | 207.7 tok/s on ≥2 reproductions for strict-envelope ladder; 233.1 tok/s on ≥2 reproductions for hardware-ceiling ladder. |
| Demonstrated envelope | 232.2 tok/s achieved at B=64 v10+bf16 / 40 GB peak (was projected 67-100 in `plans/P6_AUTORESEARCH_NOT_LIMIT_PROOF.md`; reality exceeded by 2.32-3.47×). |
| Charts | `plans/P6_AUTORESEARCH_SUMMARY.png` — comprehensive 11-cycle multi-panel summary. `plans/P6_AUTORESEARCH_PROGRESS_CYCLES.png` — per-cycle deliverables + running-best trajectory. `plans/P6_AUTORESEARCH_PROGRESS_FA_KERNEL.png` — cycle 11 FA-decode kernel ablation (silica beats mlx 1.25-1.81×). `plans/P6_AUTORESEARCH_PROGRESS_QMM_KERNEL.png` — cycle 7-9 QMM kernel tuning. `plans/P6_AUTORESEARCH_PROGRESS_DECODE_TOK_S.png` — Karpathy-style running-best ledger plot. Render via `python scripts/render_summary_chart.py` and `python scripts/render_qmm_progress_chart.py`. |
| Orientation-cycle findings (2026-05-02) | (i) Qwen3.5-27B-4bit hybrid layer pattern verified: 48 linear-attention + 16 full-attention (3:1 interleaved); head_dim=256, GQA 24:4, attn_output_gate=true. (ii) MTP weights ABSENT in production target's safetensors index; retires §6 ranked hypothesis #6. |
| First-kernel-cycle findings (2026-05-03) | (i) Per-step decomposition: 48 DeltaNet layers cost **74.2%** of step, 16 full-attn cost **21.9%**, overhead **4.0%**; per-layer cost nearly equal (1.73 ms DeltaNet vs 1.53 ms full). (ii) Trivial fused output-gate kernel `silica.kernels.fused_gated_output(x, g)` correctness PASS but speedup 1.006× — honest negative; two-op elementwise fusion doesn't reduce HBM traffic; real speedup needs FA-style fused gated SDPA, gate+o_proj fusion, or conv1d+gated_delta_update fusion. (iii) 42.17 tok/s formally proven NOT to be the chip ceiling; demonstrated 82.7% utilisation at k=1 baseline projects to 67 tok/s at B=4. |
| Cycle-19/20/21/22/23 spec-decode research thread (2026-05-04) | (i) **Cycle 19** extended β.2 coverage probe to b∈{1..1000}: coverage@64 = **0.4051** crosses user's 40% threshold; @1000 plateau at 0.767 indicates ~23% off-distribution tail. (ii) **Cycle 20** measured target verify cost at B=1 vs k: highly sub-linear (k=1→60ms, k=64→189ms = 3.14× cost for 64× tokens). Tree-spec at b=64 looked feasible. (iii) **Cycle 21** drafter survey: Qwen3.5-{0.8B, 4B, 27B-3bit} all produce flat coverage curves; @1 ranges 6.3-8.0%. The 4-bit-target's argmax distribution is the structural ceiling regardless of drafter. Drafter arm closed. (iv) **Cycle 22** design proposed 270 tok/s at B=52 with tree-spec. (v) **Cycle 23 CRITICAL NEGATIVE**: B×k batched verify cost reveals B=52 k=64 = **8105 ms** (42× B=1 k=64 of 189 ms). Cycle-22 projection used B=1 cost; reality is B×k product scaling. Spec-decode at NO B value (1..52) beats plain decode. The 40% accept rate is structurally feasible (cycle 19) but the B×k verify-cost wall makes it infeasible to translate into throughput. **Spec-decode research thread closed with negative.** The user's 270 tok/s goal is not reachable via spec-decode on this stack. C14 v10+bf16 stack stands as the load-bearing deliverable. |
| Cycle-14 v10+bf16 composition KEEPs both ladders (2026-05-04) | (i) **NEW within-envelope RUNNING-BEST**: B=52 with `SILICA_USE_FA_DECODE_V10=1` + `SILICA_USE_BF16_DELTANET_STATE=1` = **206.2 ± 0.5 tok/s** (n=3, peak 35.52 GB, within 36 GB envelope). +5.4 tok/s vs cycle-13 B=52 bf16-only = 3.4σ keep. 4.89× cycle-1 baseline. (ii) **NEW demonstrated ceiling**: B=64 v10+bf16 = **232.2 ± 0.3 tok/s** (n=3, peak 40.01 GB, within 48 GB hardware). +2.4 vs cycle-13 B=64 bf16-only. 5.51× cycle-1 baseline. (iii) Cliff bracketed sharp at 40 GB peak: B=64 = 229.8/232.2 (good), **B=66 = 166.8** (bad), B=68 = 169.1, B=72 = 173.2. The transition past 40 GB peak triggers a ~26% throughput drop, likely M5 Pro SLC or allocator threshold. (iv) v10's attention savings DID stack with bf16 state — cycle 12's "v10 alone at B=48 = no E2E delta" finding was right at that B but the kernel becomes load-bearing once cycle-13's axis-shift relocates the bottleneck. (v) Pattern across cycles 10-14: every individual probe in isolation either landed flat or moved running-best by small margin; the big wins are compositional. C10 alone +4.60×; C11 alone 0%; C12 alone 0% but produced peak save + wiring fix; C13 = C12 peak save + C10 axis-shift extension = +4.76× envelope, +5.45× hardware; C14 = C13 + C11 v10 = **+4.89× envelope, +5.51× hardware**. |
| Cycle-13 bf16-state-headroom B-axis unlock (2026-05-04) | (i) NEW RUNNING-BEST: B=52 with bf16 DeltaNet state = **200.8 ± 1.5 tok/s** (n=3, peak 35.5 GB, within 36 GB envelope). +6.9 tok/s vs cycle-10 baseline 193.9 ± 0.6 = 4.3σ keep. (ii) Demonstrated ceiling beyond strict envelope: **B=64 = 229.8 ± 2.0 tok/s** (n=3, peak 40.0 GB, within 48 GB hardware). 18σ above cycle-10 baseline. 5.45× cycle-1 baseline. (iii) Cycle 12's "193 wall" was a B=48 cap, NOT a hardware wall. The bf16 state's value isn't direct bandwidth save (cycle 12 confirmed 0% E2E at fixed B=48) — it's the 3.5 GB peak-memory headroom that enables the same axis-shift lever cycle 10 used. (iv) B=72 regressed to 173.2 tok/s at peak 43.4 GB — empirical regime change past ~40 GB on M5 Pro 48 GB (allocator pressure, SLC cache spill, or VM behaviour). The practical hardware ceiling is B≈64. (v) Cycles 11+12 didn't move E2E in isolation, but their compositional value showed up in cycle 13: shadow_install wiring fix (cycle 12) + bf16 state correctness probe (cycle 12) became the gate that unlocked the B-axis-extension move. The right unit of analysis was peak-memory ceiling × B-axis lever, not isolated kernel bandwidth. |
| Cycle-12 DeltaNet bf16-state + 193 tok/s wall confirmation (2026-05-04) | (i) Found a defect: cycle 11's E2E v10 measurement was running un-patched mlx because `shadow_install.install` was never called from the production load path. Fixed in `silica/models/qwen3_5.py:from_hf_repo` so env-flag-gated kernel swaps apply through the bench harness. (ii) DeltaNet recurrent state at bf16 (instead of fp32) preserves greedy-decode token-ID parity vs fp32 baseline on a 20-token Qwen3.5-27B sample — kernel templates already support arbitrary `StT`. Wired through `SILICA_USE_BF16_DELTANET_STATE=1` env flag. (iii) Theoretical bandwidth save: 144 MB × 48 layers / 307 GB/s ≈ 22 ms/step (~9% E2E if linear). **Observed E2E save: 0%.** 3 reproductions at warm-decode-b48 produced 192.5 ± 1.1 tok/s — statistically indistinguishable from cycle-10 baseline 193.9 ± 0.6. (iv) v10 alone (with the wiring fix), bf16 alone, v10+bf16 combined, and chunk=32 (which fails the warmup-stability oracle, re-confirming cycle 4/5) all hit the same plateau. (v) **The 193 tok/s wall is real at warm-decode-b48 in mlx 0.31.1 eager execution.** Kernel-level optimisations cannot break it; the bottleneck is dispatch / sync overhead or some non-bandwidth limit not visible from individual kernel measurements. To move past 193, the lever family must be graph-traced compilation (`mx.compile`), speculative decoding (multi-token-per-step), or scheduler-level changes. |
| Cycle-11 FA-decode port (2026-05-04) | (i) Ported FA-2/FlashDecoding to native MLX via `mx.fast.metal_kernel`: 8 variants (v1, v3, v4, v5, v6, v7, v8, v10) plus 1 abandoned (v2, register spill) and 1 deleted (v9, exceeds 32 KB TG-mem). v10 production entry: K-axis split + GQA-aware K/V tile sharing + streaming online-softmax + fused Qwen3.5 sigmoid output gate + half4 vectorized HBM loads + half4 vectorized inner ops + single-pass fast path for T_kv≤128. (ii) v8/v10 BEAT mlx `mx.fast.scaled_dot_product_attention` by **1.25-1.81×** across the production B=48 T_kv∈{128,256,512,1024} sweep at the kernel level, both plain and gated. (iii) Bandwidth utilisation reaches **60% of theoretical 307 GB/s peak** at T_kv=1024 (vs mlx's 46%) — explicit GQA tile sharing across 6 simdgroups in one TG outperforms mlx's L2-cache-dependent design once K/V exceeds L2 capacity. (iv) Two design choices made the difference: explicit GQA-tile sharing at the threadgroup level (1 TG per (b, h_kv)), and half4 vectorised loads (`device half4 const*` + `metal::dot(q4, k4)`). (v) **E2E shadow-installed measurement on real Qwen3.5-27B-4bit at warm-decode-b48 produced 193.3 ± 1.4 tok/s (n=3) — statistically indistinguishable from cycle-10 baseline 193.9 ± 0.6.** Kernel-level 1.81× attention speedup did NOT translate to E2E because attention is only 21.9% of step time per cycle-1 decomposition (74.2% DeltaNet). To push past 193.9 in cycle 12+ requires attacking the dominant DeltaNet cost or the inter-kernel sync overhead, not the full-attention layers. (vi) The fused-gate epilogue is a uniquely Silica contribution; no public Apple-Silicon kernel ships sigmoid(gate) * SDPA fusion. (vii) **MLX upgrade 0.31.1 → 0.31.2 was attempted then ROLLED BACK**: mlx-metal 0.31.2 introduces a deterministic argmax flip in greedy decode that breaks 2/3 of the preload-parity gate tests. Pinned 0.31.1 + mlx-lm 0.31.2 + mlx-metal 0.31.1 stays as the project baseline. |

---

## Running-best line (text representation)

```
decode_tok_s
    ^
60  |  ........... (1b) stretch milestone .................
    |
50  |
    |
42.17  ┃ ──────  running best (BASELINE_27B_B4, P-6.0.5 Unit 2)
    |
30  |
    |
20  |  ........... B=1 weights-only ceiling 20.29 ........
16.05 ●  BASELINE_27B_B1
    |
10  |
 7.74 ●  C4_DFLASH_RETIRE   (discard, -0.518 vs B=1 spec-off)
 6.54 ●  C5_DDTREE_BETA1    (discard, -0.593 vs B=1 spec-off)
    |
  0 +--+----+----+----+-----+----+----+----+----+----+
       baseline                                  next
```

(ASCII rendering — replace with .png when matplotlib is authorised.)

## What the ledger entries mean

- **Diagnostic** — the row reports a measurement that informs decisions but is not itself an optimisation against the running-best line. Examples: `BASELINE_27B_B4` (anchors the line); `P605_VERIFY_K_8` (caps spec-only speedups at 2.93×); `C5_DDTREE_BETA2_COV16` (escalate signal but not on tok/s axis); `AR_MTP_KEY_INSPECTION` (resolves OQ-4 negative — retires MTP probe); `AR_ARCH_VERIFICATION` (verifies the load-bearing 48:16 layer pattern + head_dim=256 + attn_output_gate=true).
- **Discard** — the row was an experiment whose pass/fail threshold did not clear. Examples: `C4_DFLASH_RETIRE` (0.482× vs ≥1.8× engineering floor); `TRACK_B_3BIT_RETIRE` (PPL drift breach); `C5_DDTREE_BETA1` (0.408× vs ≥1.0× minimum).
- **Keep** — the row improved the running-best on a primary metric by ≥3σ on ≥2 reproductions. **Zero entries today.**
- **Crash** — instrumentation / environment failure, not an optimisation result. **Zero entries today.**

## Next action

Per `plans/P6_AUTORESEARCH_REORIENTATION.md` §7, the recommended next probe is the **per-step decode time decomposition microbench** on dense Qwen3.5-27B-4bit warm-decode B=4. Diagnostic-class probe; opens or retires custom-kernel candidates. No download, no commit, no destructive op required for execution; awaits user authorisation per AR.md "creating any git commit" before any artefacts are committed to git.

## Contributing

Per AR.md "Ledger ownership", `plans/P6_AUTORESEARCH_LOG.tsv` is appended only by the main agent. Sub-agents return findings; the main agent merges and appends. Two agents writing concurrently can corrupt the file.

After each experiment:
1. Append one TSV row to `plans/P6_AUTORESEARCH_LOG.tsv`.
2. Update this file's summary line and running-best section.
3. Regenerate the .png chart if matplotlib is authorised.
4. Update the relevant `plans/P6_AUTORESEARCH/<EXPERIMENT_ID>_REPORT.md` (one short report per experiment, mirroring `plans/P6_C5_DDTREE/REPORT.md` shape).
5. If the experiment is a breakthrough (new running-best, decisive negative result that retires a path, or a measurement that changes the ranked plan), label it on the chart.
