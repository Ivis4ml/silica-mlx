# Plans and design

`plans/` is the project's design and acceptance archive. `PLAN.md` is
the single source of truth — phases, decisions, open questions. The
remaining files are per-phase opening / prep / survey / acceptance
documents and the raw measurement artifacts that closed each phase.

This page links into the archive without rendering it inline. The
files are intentionally kept as Markdown in the repo so they can be
read in-editor while iterating.

## Master plan

- [`PLAN.md`](../plans/PLAN.md) — phases, decisions log, open questions,
  changelog. Single source of truth.

## Phase 1 — interfaces and skeleton

- [`P1_ACCEPTANCE.md`](../plans/P1_ACCEPTANCE.md)
- [`P1_DAY1_GATE_A.md`](../plans/P1_DAY1_GATE_A.md)
- [`P1_DAY1_GATE_B.md`](../plans/P1_DAY1_GATE_B.md)

## Phase 2 — continuous batcher + radix prefix cache

- [`P2_OPENING.md`](../plans/P2_OPENING.md) — architectural opening
  doc: continuous-batcher design under invariant tables S-1..S-7,
  B-1..B-9, L-1..L-3.
- [`P2_GATE_0.md`](../plans/P2_GATE_0.md), [`P2_GATE_0_5.md`](../plans/P2_GATE_0_5.md)
- [`P2_PRELOAD.md`](../plans/P2_PRELOAD.md) — Unit 16a oracle.
- [`P2_UNIT_16C_PREP.md`](../plans/P2_UNIT_16C_PREP.md) — RadixPrefixCache
  prefix admission and reclaim flow.
- [`P2_UNIT_16C_2_PREP.md`](../plans/P2_UNIT_16C_2_PREP.md)
- [`P2_UNIT_16C_2_STEP_4_SKELETON.md`](../plans/P2_UNIT_16C_2_STEP_4_SKELETON.md)
- [`P2_UNIT_16D_PREP.md`](../plans/P2_UNIT_16D_PREP.md)

## Phase 3 — model adapters (Qwen3.5, Gemma4, MoE, hybrid recurrent)

- [`P3_DELTANET_SURVEY.md`](../plans/P3_DELTANET_SURVEY.md) — hybrid
  DeltaNet survey + the C-open-3 finding that recurrent adapters do
  not pair with `RadixPrefixCache`.
- [`P3_GEMMA4_SURVEY.md`](../plans/P3_GEMMA4_SURVEY.md) — Gemma4
  sliding/full attention layout.
- [`P3_BATCH_ROTATING_KV_SURVEY.md`](../plans/P3_BATCH_ROTATING_KV_SURVEY.md)
  — `BatchRotatingKVCache` audit.
- [`P3_MOE_SURVEY.md`](../plans/P3_MOE_SURVEY.md) — Qwen3.5-MoE +
  Gemma4-MoE adapter design.
- [`P3_C5_OPENING.md`](../plans/P3_C5_OPENING.md), [`P3_C5_3_DESIGN.md`](../plans/P3_C5_3_DESIGN.md)
  — recurrent-state snapshot α-MVP.

## Phase 4.5 — exit bridge (chunked prefill + KV codec spike)

- [`P4_5_CHUNKED_PREFILL_OPENING.md`](../plans/P4_5_CHUNKED_PREFILL_OPENING.md)
- [`P4_5_C_KVCODEC_OPENING.md`](../plans/P4_5_C_KVCODEC_OPENING.md) —
  the three integration-point options against D-003.

## Phase 5 — KV codec stack (BlockTQ + RaBitQ)

- [`P5_OPENING.md`](../plans/P5_OPENING.md) — payload schemas, codec
  catalogue, scalar-equivalence invariant.
- [`P5_A_REAL_OPENING.md`](../plans/P5_A_REAL_OPENING.md) —
  real-activation Frobenius cross-check.
- [`P5_A_U4_STORE_MIGRATION.md`](../plans/P5_A_U4_STORE_MIGRATION.md)
- [`P5_C2_STEP3_PPL_ROWS.md`](../plans/P5_C2_STEP3_PPL_ROWS.md)
- [`P5_F_OPENING.md`](../plans/P5_F_OPENING.md) — pre-RoPE production
  routing via the (3b) projection-output capture path.

### Operational gate (P5.9 step 2(g))

- [`P5_REGRESSION_GATE.md`](../plans/P5_REGRESSION_GATE.md) —
  operator's how-to for the (4-b) two-part aggregated gate as
  a P-6 per-track regression contract. Documents both
  evaluation modes (cheap silica-only pre-merge gate / full
  silica-vs-vqbench phase-exit attestation), the canonical
  bench command for each, the v1.7.3 pinned reference values,
  the per-mode running frequency table, and the
  drift-investigation playbook.

### Acceptance evidence

- [`P5_ACCEPTANCE_SWEEP/`](../plans/P5_ACCEPTANCE_SWEEP/) — `bench.py`
  sweep results: codec-swap neutrality, `--all-kv-codecs` 924-row
  report, admission-headroom verification, real-activation xcheck,
  Qwen3.5-4B b-static (default + per-head) 3-seed verification.
- [`P5_D2_INVESTIGATION/`](../plans/P5_D2_INVESTIGATION/) — D.2 / D.2a
  pre-RoPE projection-patch oracle development and per-head Haar
  rotation 3-seed re-measurement.

### State inventory and drift experiments

- [`P3_C5_DRIFT_EXPERIMENT/`](../plans/P3_C5_DRIFT_EXPERIMENT/) —
  recurrent-state drift probe data + README.
- [`P3_C5_STATE_INVENTORY/`](../plans/P3_C5_STATE_INVENTORY/) —
  Qwen3.5-{0.8B, 4B, 35B-A3B} state inventories.

## Phase 6 — performance phase (re-scoped at v1.7.13)

- [`P6_OPENING.md`](../plans/P6_OPENING.md) — re-scope opening:
  bandwidth physics, Step 0 measurement gate, five orthogonal tracks
  (A sync-barrier collapse, B 3-bit weights, C speculative decoding,
  D TTFT levers, E weight streaming + SSD prefix tier), dual-target
  acceptance (dense Qwen3.5-27B-4bit ≥60 tok/s primary + MoE
  Qwen3.5-35B-A3B-4bit ≥100 tok/s stretch on 48 GB M5 Pro),
  proposed PLAN.md edits applied via D-017 / D-018 / D-019.
- [`P6_0_BASELINE/REPORT.md`](../plans/P6_0_BASELINE/REPORT.md) —
  P-6.0 measurement-gate baseline interpretation: 8 scenarios
  measured on M5 Pro 48 GB; dense 27B at 16.05 tok/s (70.6%
  bandwidth utilization, gap 3.74× to the §6 gate), MoE 35B-A3B
  B=2 at 120.93 tok/s aggregate (already clears the §6 stretch
  gate at baseline). Per-scenario `.jsonl` + `.md` reports plus
  `logs/p6_0_step*.log` reproduce the numbers.
- [`P6_REVIEW_HANDOFF.md`](../plans/P6_REVIEW_HANDOFF.md) —
  self-contained handoff for an external reviewer (e.g. GPT-5.5
  xhigh): mission + hard constraints, phase recap with evidence,
  the open question (dense 60 tok/s reachability), eight specific
  review questions Q-R1..Q-R8 ranked by priority, layered
  reading-order navigation, and the verification map for
  re-deriving any cited number independently.

### P-6.0.5 measurement expansion (D-021 step 3)

- [`P6_0_5_OPENING.md`](../plans/P6_0_5_OPENING.md) — opening for
  the eight-row measurement expansion (5 mandatory warm-decode +
  2 warm-TTFT-pair + 1 target-verify microbench).
- [`P6_0_5_BASELINE/`](../plans/P6_0_5_BASELINE/) — closed
  artefacts; `REPORT.md` aggregates the cross-row reading and
  feeds Decision Gate 1.

### Decision Gate 1 (D-021 step 4)

- [`P6_0_DECISION_GATE_1_OPENING.md`](../plans/P6_0_DECISION_GATE_1_OPENING.md)
  — closed at v1.7.18. (1a) ≥40 tok/s primary unchanged, (1b)
  ≥60 tok/s reframed as stretch with two-condition survival rule
  (full-stack measurement OR Track C.5 tree-shape spike), (2a)
  ≥100 tok/s aggregate cleared, (2b) reduced to ≥175 tok/s
  aggregate at B≥3.

### D-021 step 5 — speculative-decoding foundation (closed at v1.7.19)

- [`P6_SPEC_FOUNDATION_OPENING.md`](../plans/P6_SPEC_FOUNDATION_OPENING.md)
  — opening for step 5: ten sub-units (a, a2, b..i) with §6.1
  acceptance gates, §6.2 ≥1.2× tracked-not-blocking note, §6.3
  toolchain attestation. **Closure status (v1.7.19)** in the §6.1
  block lists every sub-unit's commit, the (e) recurrent rollback
  trim → restore → replay arithmetic, and the (c) slice 3 deferral
  (multi-request hybrid batched-spec, non-blocking for foundation
  correctness).
- [`P6_SPEC_FOUNDATION_C_ORIENTATION.md`](../plans/P6_SPEC_FOUNDATION_C_ORIENTATION.md)
  — orientation for sub-unit (c) (multi-request batcher integration);
  decisions log + slice plan with [F-1]..[F-7] findings.
- [`P6_SPEC_FOUNDATION_E_ORIENTATION.md`](../plans/P6_SPEC_FOUNDATION_E_ORIENTATION.md)
  — orientation for sub-unit (e) (recurrent-state rollback);
  decisions log including [F-3a] full-trim-before-replay rationale
  (a partial-trim variant pollutes attention context during replay).

### D-021 step 6 — C.4 DFlash spike (orientation)

- [`P6_C4_DFLASH_OPENING.md`](../plans/P6_C4_DFLASH_OPENING.md) —
  opening for the C.4 block-diffusion spike. PLAN.md §13 step 6 gate
  quoted verbatim (≥1.8× engineering / ≥2.5× one component of (1b)
  survival rule); spike shape B=1 mirroring step 5 (h); `DraftEngine`
  Protocol-signature unchanged. (α) closed favourably (MIT, no torch,
  Python API + `DRAFT_REGISTRY` 4-bit-target pairing confirmed) and
  surfaced F-1: upstream's drafter is **target-conditioned** with a
  per-layer streaming `ContextOnlyDraftKVCache` and
  `commit`-updates-`target_hidden` semantics — not a stateless small
  LM. Eight sub-units (α, αβ, β..η); (αβ) added between α and β to
  install the target-hidden capture path on `Qwen3_5Adapter` /
  `qwen3_5_moe.py` (analogous to P-5-F (3b)). Upstream tape-replay
  verify, `verify_qmm` int4 Metal kernel, and target speculative
  hooks remain explicit non-goals (deferred full-DFlash port).
  **Closed at v1.7.20 with gate FAILED** — η.1 measured 0.482×
  silica-integrated speedup (vs ≥1.8× floor) due to drafter cost
  domination (35.7 ms drafter vs 2.45 ms verify) + accept-rate
  collapse (0.088 vs predicted 0.5-0.7) on the 4-bit target. C.4
  dense path retires; (1b) ≥60 tok/s now hinges on C.5 alone (D-021
  step 8, **not auto-queued**).
- [`P6_C4_DFLASH/REPORT.md`](../plans/P6_C4_DFLASH/REPORT.md) — the
  spike's measurement bundle; (αβ.1) / (αβ.2) `c_capture_hidden(k=16)`
  microbenches at noise level; (η.1) dense 27B real-checkpoint
  attestation with the gate-FAILED call + three findings + follow-up
  open questions (quantize-draft, full-DFlash kernel port, upstream
  baseline).

### D-021 step 7 — Track B 3-bit weights (native candidate retired at v1.7.21)

- [`P6_TRACK_B_3BIT_OPENING.md`](../plans/P6_TRACK_B_3BIT_OPENING.md) —
  opening doc for Track B 3-bit weight option. PLAN.md §13 step 7 gate
  quoted verbatim (loader + PPL oracle first, then 27B 3-bit
  warm-decode; pass quality gate; if 3-bit lifts dense from 16 → 21-24
  tok/s, stack with spec; if quality or MLX path is unstable, ship
  opt-in). Three sub-units B.1 / B.2 / B.3 (loader smoke + bench
  registration; WikiText-2 PPL cross-check; 27B 3-bit warm-decode
  attestation). Quality gate is **both-pass**: ΔPPL_abs ≤ 0.5 AND
  ΔPPL_rel ≤ 5% (the user-confirmed reading; the opening's earlier
  "OR" wording was tightened during pre-B.1 review). Performance gate
  for B.3: ≥ 21 tok/s = 1.31× over the 16.05 b1 anchor.

- [`P6_TRACK_B/REPORT.md`](../plans/P6_TRACK_B/REPORT.md) —
  measurement bundle. **Track B native 3-bit candidate retired after
  the B.2 quality gate failed; B.3 not run; gate not relaxed.**
  - **B.1 PASS** — `NexVeridian/Qwen3.5-27B-3bit` loaded cleanly via
    `silica.models.factory.adapter_for_repo`; 4-token smoke generate
    succeeded; peak `11.16 GiB` clears both forms of the §6.1 B.1
    memory gate (`≤ 13 GiB` absolute and `≥ 20%` relative reduction
    against the 4-bit anchor 15.34 GiB; measured 27.2% reduction).
  - **B.2 FAIL** — paired WikiText-2 chunked-NLL PPL rows
    (`qwen3.5-27b-wikitext-ppl-{4bit,3bit}`) measured `ppl = 6.9082`
    on the 4-bit anchor and `ppl = 8.0719` on the 3-bit candidate
    over the same 511 scored positions. ΔPPL_abs = +1.1637 (gate
    ≤ 0.5 — FAIL by 0.66 PPL); ΔPPL_rel = +16.85% (gate ≤ 5% — FAIL
    by ≈12 percentage points). Both bounds breached on a both-pass
    gate.
  - **B.3 not run** — Track B's both-pass acceptance over (B.1
    memory, B.2 quality, B.3 speedup) is already retired at B.2;
    speedup numbers cannot rescue a candidate whose pre-declared
    quality bound is breached. Per the user-confirmed framing
    ("速度数字没有决策价值"), B.3 is not authorised against this
    checkpoint. The `qwen3.5-27b-warm-decode-b1-3bit` scenario
    registered at B.1 stays in the catalog as the load-bearing
    artefact for any future re-attempt — only `repo` + gate envs
    swap when a better candidate appears.
  - **Gate not relaxed.** 17% PPL drift in exchange for 27% memory
    headroom + projected 1.31× speed lift does not meet the
    mainline-performance-lever bar this step declared.
  - **Follow-up gated on a small read-only candidate survey** for
    activation-aware / smaller-group-size / AWQ-style Qwen3.5-27B
    3-bit MLX checkpoints. **No auto re-conversion of the ~52 GB
    full-precision weights** — survey first, conversion only on a
    documented motivation. Mainline next move stays the C.5 tree-
    shape spike (D-021 step 8) for the (1b) ≥60 tok/s survival path.
  - **Survey closed empty on 2026-05-01.** No viable better-
    calibrated MLX-native 3-bit matched-family candidate exists on
    HF Hub: the only matched-family MLX 3-bit checkpoints
    (`NexVeridian/Qwen3.5-27B-3bit`,
    `RepublicOfKorokke/Qwen3.5-27B-mlx-lm-3bit`) both use
    `mlx_lm.convert -q --bits 3` at default `group_size=64`
    without any activation-aware step; activation-aware methods
    (AWQ / GPTQ / AutoRound / DWQ / OptiQ) for Qwen3.5-27B exist
    in MLX at 4 bits but not at 3 bits. Track B native 3-bit
    lever is therefore fully retired pending a future checkpoint.

- [`P6_TRACK_B_FOLLOWUP_SURVEY.md`](../plans/P6_TRACK_B_FOLLOWUP_SURVEY.md) —
  read-only HF Hub survey doc; per-candidate verdicts table,
  acceptance criteria for any future re-attempt, and re-look
  triggers (MLX-native activation-aware 3-bit sib to the existing
  4-bit DWQ / OptiQ / GPTQ / AutoRound line; mlx-lm 3-bit
  calibration step; explicit model-card PPL evidence).

  Sub-unit commits in order: `9299294` opening / `adb52cd` orientation
  three-fix / `aa150d2` OQ-1 favourable close (`NexVeridian` found) /
  `62e36c3` (B.1) loader smoke + bench scenario / `eba7e26` (B.2)
  negative-result closure + ΔPPL measurement / `3d52e00`
  (PLAN + plans-index sync) / `44bd928` (follow-up survey).

### D-021 step 8 — C.5 tree-shape decision spike (closed)

- [`P6_C5_DDTREE_OPENING.md`](../plans/P6_C5_DDTREE_OPENING.md) —
  orientation document for the C.5 tree-shape spike. **Reframed
  as a decision spike, not a DDTree implementation
  continuation.** Post-C.4-retirement (v1.7.20) and Track B
  retirement (v1.7.21), C.5 is the only surviving leg of the
  v1.7.18 (1b) ≥60 tok/s two-condition rule; this orientation
  decides whether to spend implementation effort *before* any
  `silica.speculative.ddtree` code is written.

  Load-bearing concept correction: the gating empirical metric
  is **measured `coverage@b`** = Pr(target argmax ∈ drafter
  top-b) for b ∈ {4, 8, 16}, **not** linear top-1 accept rate
  α. C.4 measured α=0.088 against the 4-bit Qwen3.5-27B target;
  the independence-bound envelope `1 - (1-α)^b` is only an
  optimistic upper bound assuming b independent draws, but
  real greedy tree branches are the same drafter's top-b
  candidates and are correlated. Coverage@b must therefore be
  measured directly, not inferred from α.

  Planned sub-units at opening time: α (this orientation) → β.1 (single bench run on
  the v1.7.19 (h) `qwen3.5-27b-warm-decode-spec-on` scenario,
  yields linear α + drafter cost + rollback for the cached
  `mlx-community/Qwen3.5-27B-4bit` × `Qwen/Qwen3.5-0.8B`
  pairing) → β.2 (stand-alone read-only top-b coverage probe
  with tokenizer-alignment guard; bypasses the spec engine; no
  new download) → γ.1 upstream `humanrouter/ddtree-mlx` survey
  → γ.2 minimal `silica.speculative.ddtree` implementation → δ
  real-checkpoint attestation. The closure report supersedes this
  plan after β.1 / β.2 plus the production-B verify-cost matrix.

- [`P6_C5_DDTREE/REPORT.md`](../plans/P6_C5_DDTREE/REPORT.md) —
  β.1 / β.2 measurement bundle and final closure. The pre-declared
  matrix initially escalated: `coverage@4=0.14`, `coverage@8=0.20`,
  `coverage@16=0.26`, with `coverage@32=0.34` outside the
  implementation gate. Opus cycle 23 then measured production-B
  verify cost and closed the escape hatch: B=52 k=64 costs
  **8105 ms** vs B=1 k=64 at `190 ms`, while plain decode at the
  same B is roughly `252 ms` per step (`~206 tok/s` aggregate).
  A viable DDTree path would need to break B-axis scaling at
  production batch, not only k-axis tree width. **Final
  disposition:** C.5 clean-retired; no γ.1 survey and no
  `silica.speculative.ddtree` port.

  The original decision matrix is retained in the report as audit
  trail: β.2 alone escalated rather than auto-retired, but cycle 23
  supplied the missing production-cost evidence and converted the
  disposition to clean retire. No gate relaxation — the (1b) ≥60
  tok/s bar is the bar.

  Sub-unit commits in order: `eed5204` (orientation), `03226d7`
  (β.2 probe + tests), `309af8c` (β.1 / β.2 measurement bundle),
  then this closure sync.

### D-022 — P-6 small-B dispatch / latency line (closed at v1.7.28)

- [`P6_SMALL_B_OPENING.md`](../plans/P6_SMALL_B_OPENING.md) —
  actionable opening for the small-B latency / dispatch attack
  declared as next P-6 research direction at v1.7.24, recorded
  formally as D-022 in `PLAN.md` §9. Replaces "next direction
  hint" with sub-unit-level commands, gates, and stop conditions.
- [`P6_SMALL_B/DELTA/PRE_PROJECTION.md`](../plans/P6_SMALL_B/DELTA/PRE_PROJECTION.md) —
  final closure audit for δ and the D-022 line.

  Five sub-units (α–ε) reached terminal state: α completed the
  sonnet baseline refresh at v1.7.25; β closed negative at v1.7.26
  because post-cache attention compile had too little reachable
  scope; γ closed negative at v1.7.27 because `Qwen3NextMLP`
  compile had only 1.011× per-call gain; δ closed negative-on-audit
  at v1.7.28 because the 3.6% overhead bucket was mostly real
  compute, with ≤0.6% recoverable Python-hygiene headroom. ε remains
  an upstream mlx async-copy waitlist trigger, not an open sub-unit.

  Goal framing: interactive single-row latency / TTFT, **not**
  throughput parity (per-row at B=4 already exceeds per-row at
  B=52). Non-goals: no new Metal kernels, no spec-decode reopen,
  no high-B axis extension as primary objective, no Tier-2 opus
  kernel imports without explicit user authorization (the
  `silica/kernels/` public surface stays at v10 + slim
  shadow_install per the v1.7.23 narrowing).

  Tools landed in v1.7.24 Step 4 are α's executable surface:
  warm-decode-b{4,8,12} scenarios in `silica.bench.scenarios`,
  attribution microbenches at
  `silica/bench/microbench/{decode_step,layer_internal}_attribution.py`,
  slim `silica.kernels.shadow_install` with the
  `SILICA_USE_BF16_DELTANET_STATE` / `SILICA_USE_FA_DECODE_V10`
  flags, and the `mlx==0.31.1 / mlx-lm==0.31.2 / mlx-metal==0.31.1`
  pin attested by `tests/test_p2_preload_parity.py`. No tool
  debt blocked the first measurement. D-022's closure artefacts live
  under `plans/P6_SMALL_B/{BETA,GAMMA,DELTA}/`; re-opening requires
  new evidence such as mlx async-copy primitives, a different kernel
  path, or a different model architecture.

### D-023 — Gemma 4 MTP drafter pre-projection (Track C external reopen probe)

- [`MTP_GEMMA4_PRE_PROJECTION.md`](../plans/MTP_GEMMA4_PRE_PROJECTION.md) —
  half-day external spike against Google's Gemma 4 multi-token-prediction
  (MTP) drafter, opened as D-023 in `PLAN.md` §9 at v1.7.29 and running
  before P-8 OpenAI HTTP server work begins. External evidence on the
  v1.7.20-22 closed Track C line; D-023 is a new entry, not a C.3
  reopen, because the current Silica production target
  `mlx-community/Qwen3.5-27B-4bit` ships no MTP weights and Gemma 4 is
  a new family + new public drafter. Spike runs entirely outside
  `silica.*` through `mlx_vlm` (canonical CLI form `python -m
  mlx_vlm.generate --model … --draft-model … --draft-block-size … --temp 0`);
  native silica integration only triggers on a measured B=1 per-row ≥
  1.3× speedup gate at the decision row.

  **Pairing + feasibility caveat at the top.** The advertised MTP pair
  is target `mlx-community/gemma-4-31B-it-bf16` (~62.5 GB BF16,
  exceeds M5 Pro 48 GB unified memory) + drafter
  `mlx-community/gemma-4-31B-it-assistant-bf16` (~939 MB BF16), so the
  spike is not runnable as-advertised on this hardware. Gate (i) is now
  resolved as outcome A\*: `mlx-community/gemma-4-31b-it-4bit` exists as
  the 4-bit IT target and pairs with the BF16 assistant drafter for a
  hardware-feasible but mixed-precision / undocumented spike. The cached
  `mlx-community/gemma-4-31b-4bit` (17 GB) is the non-IT 4-bit variant
  and remains **not used**. The precision mismatch is empirical and must
  be carried into the verdict.

  Sections in the spike doc: provenance (D-023 vs C.3 reopen with
  narrowed Qwen3.5 production-target framing), license-verify (待核查
  — official Apache-2.0 vs `mlx-community` conversion-metadata
  `License: gemma`, reuse decision deferred), supported-pairing +
  hardware-feasibility verification (the load-bearing gate; outcomes
  A/A\*/B/C/D; gate (i) resolved A\* on 2026-05-06), runtime deps
  (`mlx-vlm 0.5.0` requires `mlx>=0.31.2 / mlx-lm>=0.31.3` and would
  force-bump the v1.7.21 determinism anchor, so install lands in an
  isolated venv at `~/.cache/silica-d023-mtp/.venv`; silica project
  pin `mlx==0.31.1 / mlx-lm==0.31.2 / mlx-metal==0.31.1` stays
  untouched), HF cache check (outcome A\* target snapshot `dcb78c3`
  and drafter snapshot `28e9227` cached; cached non-IT 4-bit base is
  not used; advertised BF16 target exceeds the 48 GB ceiling),
  measurement plan (`draft_block_size ∈ {2, 3, 6, 9}` ×
  `B ∈ {1, 4}`, `k_candidates = block_size − 1`; `block_size = 6` is
  the card's single-request recommendation, `block_size = 3` the
  batched recommendation, `block_size = 2` is the verify-cost-floor
  diagnostic row only), decision-row vs diagnostic-row distinction,
  variance discipline with B=1 noise-floor caveat, rough projection
  (intuition only — does not contribute to gate decision), measured-gate
  matrix (top-down: PAIR-INFEASIBLE / GREEDY-PARITY-FAIL /
  DRAFT-VERIFY-WALL hard blocks first, then B=1-PASS / B=4-ONLY-PASS /
  NEGATIVE pass-or-close rows), native-integration-gap analysis
  (`Gemma4Adapter` does not implement `HiddenCaptureAdapter`; runtime
  gate at `silica/bench/runner.py:537` raises `NotImplementedError`;
  native MTP wiring requires extending `decode_step_multi_with_capture`
  + `prefill_with_capture` to `Gemma4Adapter` first), and a verdict
  template populated only after the spike runs. P-8 opens cleanly
  after D-023 settles in any direction.

## Side track: chat CLI redesign

- [`CHAT_CLI_OPENING.md`](../plans/CHAT_CLI_OPENING.md) — the design
  doc behind the bundled REPL client. The {doc}`chat-cli` page
  summarises usage; this doc explains *why* the rewrite happened and
  documents the toolbar field set, palette mode detection, thinking
  parser, and persistence schema.

### CHAT-CLI-HARDENING — F1..F6 closure

- [`CHAT_CLI_HARDENING.md`](../plans/CHAT_CLI_HARDENING.md) — F1..F6
  GPT-5.5 review findings closed across eight commits: live `/system`
  propagation, `enable_thinking` template threading,
  `RadixPrefixCache.stats()` / `PrefixCacheStats` public surface,
  `/regenerate` rollback, `/model --keep-history`, swappable
  live-toolbar backend, non-interactive `chat_bench` harness,
  app-layer unit tests. Decision D records why a full
  `prompt_toolkit.Application` was deferred.
- [`CHAT_CLI_HARDENING_ACCEPTANCE.md`](../plans/CHAT_CLI_HARDENING_ACCEPTANCE.md)
  — manual real-model acceptance template (HARDENING-9). Pending a
  user-driven on-device run.

### CHAT-CLI-RESPONSE-POLICY — RP-1..RP-3 closure

- [`CHAT_CLI_RESPONSE_POLICY.md`](../plans/CHAT_CLI_RESPONSE_POLICY.md)
  — `thinking_history=strip` (RP-1, history-side `<think>` strip
  with deferred-finalise contract on `max_tokens`), `/continue`
  (RP-2, append-in-place continuation with implicit-leading
  restoration on the prompt and three-tier finalise fallback),
  truncation marker + `finish=` toolbar field + per-turn
  reasoning/visible char split (RP-3). Decision E pins the
  three-axis thinking model (model / display / history), G / H
  document RP-1's mixed-scope and RP-2's single-commit landing,
  I logs the interim exit back to D-021 step 3 / P-6.0.5.
  RP-4..RP-6 are listed for sequencing visibility but explicitly
  deferred behind P-6.0.5.
