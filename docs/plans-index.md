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

## Side track: chat CLI redesign

- [`CHAT_CLI_OPENING.md`](../plans/CHAT_CLI_OPENING.md) — the design
  doc behind the bundled REPL client. The {doc}`chat-cli` page
  summarises usage; this doc explains *why* the rewrite happened and
  documents the toolbar field set, palette mode detection, thinking
  parser, and persistence schema.
