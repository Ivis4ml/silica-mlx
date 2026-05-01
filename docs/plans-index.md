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
