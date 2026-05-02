# P-6 Step 8 — C.5 Tree-Shape Spike Orientation

**Status:** orientation — doc only. No code. No drafter conversion.
**Date opened:** 2026-05-01.
**Closes:** post-C.4 reframe of D-021 step 8.
**Audit chain:** v1.7.20 C.4 closure (FAIL @ 0.482×) → v1.7.21
Track B candidate retired → this doc.

---

## TL;DR

This is a **decision spike**, not the start of a DDTree port. The
v1.7.18 Decision Gate 1 reframe made C.5 the **only surviving**
path to (1b) ≥60 tok/s after C.4 retired. Before any
`silica.speculative.ddtree` code is written, this orientation
answers one question:

> Given the C.4 measured `accept_rate = 0.088` against a 4-bit
> Qwen3.5-27B target, is there *any* tree-shape configuration
> that can credibly project ≥60 tok/s on the b1 16.05 anchor —
> i.e. ≥ 3.74× silica-integrated speedup?

The arithmetic in §3 gives an **independent-branch optimistic
upper bound**: at the C.4-measured top-1 accept rate of α=0.088,
even a fat tree (b=16) tops out around 4.82 tokens/cycle in the
*idealised* envelope. The real gate is **not** that envelope —
real greedy tree branches are the same drafter's top-b
candidates and are *correlated*, not independent samples. The
load-bearing empirical metric for tree-shape lift is therefore
**`coverage@b` = Pr(target's argmax lies in the drafter's top-b
candidates)** — measured directly, not inferred from α via the
independence formula.

This reframes the gating empirical question. It is **not** "can
we implement tree-verify" and it is **not** "is α ≥ 0.30." It
is:

> For some accessible drafter-target pairing against the 4-bit
> Qwen3.5-27B target, does **measured `coverage@b`** for any
> b ∈ {4, 8, 16} reach ≥ 0.30?

Tree-shape extracts lift exactly when target argmax often
*sits inside* the drafter's top-b set — even if the drafter's
top-1 is usually wrong (low α). Conversely, if the drafter is
fundamentally mispredicting and target argmax is broadly
distributed across drafter's tail (low coverage at every b),
no tree depth or branching rescues the run.

C.5 sub-units therefore split β into a two-step probe:

- **β.1 — existing spec-on row.** Runs
  `qwen3.5-27b-warm-decode-spec-on` on cached
  `mlx-community/Qwen3.5-27B-4bit` × `Qwen/Qwen3.5-0.8B`.
  Yields linear α, `draft_cost_ms`, `verify_cost_ms`, and
  `rollback_count` for the 0.8B-drafter pairing — directly
  comparable to C.4's η.1 numbers.
- **β.2 — top-b coverage probe.** Stand-alone read-only script
  that, for N decode positions on a fixed prompt corpus,
  records the rank of the target's argmax in the drafter's
  sorted logits and reports `coverage@b` for b ∈ {1, 4, 8,
  16, 32}. Bypasses the spec engine entirely (no rollback, no
  verify path) so the metric is clean and decision-relevant.

If β.2 returns `coverage@4`, `coverage@8`, **and** `coverage@16`
all below 0.15 across the corpus, **(1b) retires** and §13 step
8 closes without a DDTree port. (If `coverage@16` is high but
`coverage@8` is low, the disposition is *escalate*, not retire
— see §9; depth alone may still admit a tree-shape projection
under specific kernel-cost assumptions.) If `coverage@b`
reaches ≥ 0.30 for some b ≤ 16 *and* β.1's drafter-cost /
rollback profile is materially better than C.4's DFlash
baseline, γ.1 (upstream DDTree survey) opens with explicit
user authorisation.

The "α=0.088 cannot rescue" arithmetic in §3 is true **only
under C.4's DFlash cost model** (35.7 ms drafter + ~80 ms
rollback/replay). It does **not** carry across to the C.1 0.8B
pairing — the 0.8B drafter is roughly 2.5× cheaper per propose
forward than the 2B BF16 DFlash drafter, and rollback frequency
is alpha-dependent. β.1 measures both before any C.5 cost
projection is admissible.

---

## 1. Why C.5 is now (1b)'s only survival path

**(1b) ≥60 tok/s** was reframed at v1.7.18 (D-021 step 4
Decision Gate 1, `plans/P6_0_DECISION_GATE_1_OPENING.md`) as a
two-condition survival rule:

> Reaching (1b) requires either
> (i) a **measured full stack** on the (1a) workload (Track A ×
> Track B × Track C with C.4 *or* C.5 landed) clearing ≥60
> tok/s, or
> (ii) a **Track C.5 tree-shape spike** demonstrating headroom
> over the linear k=8 verify ceiling (P-6.0.5 Unit 7, 2.93×
> target-side / zero-drafter-cost) sufficient to make the
> full-stack projection ≥60 credible.

Two posterior facts close (i) at v1.7.21:

- **Track B native 3-bit retired** (v1.7.21, B.2 quality gate
  FAIL on `NexVeridian/Qwen3.5-27B-3bit`; survey closed empty).
  Track B is no longer a multiplicative leg in (i)'s "full
  stack".
- **C.4 retired** (v1.7.20, η.1 measured 0.482× silica-integrated
  speedup). C.4 cannot contribute to (i)'s "full stack" either.

So (1b) survives only on (ii). If (ii) fails or is not pursued,
**(1b) retires entirely.** (1a) ≥40 tok/s primary stays at
v1.7.17's 42.17 tok/s P-6.0.5 baseline regardless.

This makes C.5 the load-bearing decision. It does **not** make
C.5 a deliverable on autopilot. The gate this orientation must
clear is: tree-shape under realistic drafter-pairing
**`coverage@b`** (measured, not inferred from α) produces a
≥ 3.74× projection that survives drafter cost + rollback
overhead + verify-kernel cost.

## 2. C.4's findings carried forward

Three C.4 (η.1) findings constrain anything that pairs the same
drafter with the 4-bit target:

1. **Drafter-target argmax mismatch.** The
   `z-lab/Qwen3.5-27B-DFlash` drafter trains against the
   **full-precision** Qwen3.5-27B target; the 4-bit-quantised
   target's argmax distribution diverges materially —
   measured `accept_rate = 0.0881`, ≈6× below the predicted α
   ∈ [0.5, 0.7] band. **Tree-shape samples from the same
   drafter distribution multiple times; it does not change
   what the drafter predicts.** If the drafter's argmax is
   wrong, the drafter's top-b are correlated with the same
   wrong direction.
2. **Drafter cost dominates verify cost by 15×.** The 2B BF16
   drafter takes 35.7 ms per propose call vs the 4-bit
   target's 2.45 ms verify forward. Tree-shape adds verify
   work but does not reduce drafter cost; if the drafter cost
   stays at 35.7 ms, tree-shape's tokens/cycle lift has to
   amortise that over a single-cycle propose.
3. **Rollbacks dominate decode time.** With α=0.088 every
   cycle effectively rolls back; per-cycle overhead beyond
   propose+verify is ~80 ms (rollback + replay). Tree-shape
   shifts where rejection happens within the verify window
   but does **not** reduce rollback frequency — that comes
   only from raising α.

Findings 1 and 3 inherit unchanged into C.5 if the same drafter
is paired with the 4-bit target. Finding 2 is drafter-specific
and would change with a different drafter pairing.

## 3. Tree-shape lift envelope — independent-branch optimistic upper bound

The table below is **only an optimistic upper bound under the
random-sampling, independent-branches assumption.** It is *not*
a prediction of real greedy tree-shape behaviour and *not* the
gate the spike clears. The real gate is measured `coverage@b`
from β.2, which the formula below cannot infer from α alone.

```
Linear k=8       tokens/cycle = (1 - α^8) / (1 - α) + 1   (incl. bonus)
Tree b, k=8      per-pos surv  = 1 - (1-α)^b              ← assumes b independent samples
                 expected depth = (1 - p^k) / (1 - p)        of α-each from drafter dist
                 tokens/cycle  = depth + 1
```

| α     | lin k=8 | tree b=2 | tree b=4 | tree b=8 | tree b=16 |
| ----- | ------- | -------- | -------- | -------- | --------- |
| 0.088 |   2.10  |   2.20   |   2.45   |   3.08   |    4.82   |
| 0.150 |   2.18  |   2.38   |   2.91   |   4.38   |    7.20   |
| 0.200 |   2.25  |   2.56   |   3.41   |   5.59   |    8.25   |
| 0.300 |   2.43  |   3.03   |   4.70   |   7.56   |    8.91   |
| 0.500 |   2.99  |   4.60   |   7.45   |   8.89   |    9.00   |
| 0.700 |   4.14  |   6.89   |   8.78   |   9.00   |    9.00   |

**Why this is only an upper bound.** Real greedy tree-shape uses
the drafter's **top-b logit candidates** at each position —
deterministically — not b independent samples from the
drafter's distribution. The b candidates therefore share the
*same* underlying logit vector and are strongly correlated. The
per-position survival probability is the empirical
`coverage@b` = `Pr(target_argmax ∈ top-b drafter candidates)`,
which:

- equals α exactly at b=1 (definition);
- is bounded above by `1 - (1-α)^b` only if the drafter is
  drawing b independent samples (it is not);
- can be *substantially lower* than the formula if drafter's
  top-2…top-b are highly correlated with top-1's wrong direction
  (typical mispredict pattern: drafter assigns mass to a
  semantically related cluster that target rejects);
- can also be *higher* than α suggests if target argmax sits
  consistently in drafter's rank 2-8 (typical of pairings where
  the drafter is "directionally right but greedy-wrong").

The shape of `coverage@b` vs b is therefore an empirical
question that this formula cannot answer. **β.2 measures it
directly** by recording per-position rank of target argmax in
drafter logits and reporting `coverage@b` for b ∈ {1, 4, 8,
16, 32}.

**P-6.0.5 Unit 7 anchor:** linear k=8 zero-drafter-cost ceiling
measured at **2.93×** target-side. Tree-shape's value is the
lift *above* 2.93×. The (1b) gate requires effective speedup ≥
3.74×. Crossing 3.74× requires `coverage@b` materially above
linear α — a condition the independence formula assumes
generously and the data may not honour.

**Cost-model carry-forward:** the table above is *zero-drafter-
cost zero-rollback* envelope. Translating it to silica-integrated
speedup requires the drafter's cost-per-propose and rollback
frequency, both of which are pairing-specific:

- **DFlash pairing (C.4 (η.1) measured):** 35.7 ms drafter +
  ~80 ms rollback/replay = ≈120 ms cycle. Even at the b=16
  optimistic-envelope 4.82 tokens/cycle, the silica-integrated
  speedup is ≈40 tok/s — *below* (1b)'s 60 tok/s. **The DFlash
  pairing at α=0.088 cannot rescue (1b) regardless of tree
  shape.** This is the only conclusion §3 supports without
  β.1's measurement.
- **C.1 / 0.8B pairing:** drafter cost is unknown but expected
  to be ~2.5× cheaper than the 2B BF16 DFlash drafter on a
  per-forward-pass basis (smaller weight footprint, fewer
  parameter loads). Rollback frequency is α-dependent so it
  cannot be projected without measuring α first. **Suspended
  until β.1.**

The crossover where **independence-bound** tree envelope first
exceeds the linear k=8 ceiling is around **α ≈ 0.20-0.30**;
below it the *upper-bound* envelope is bounded by per-position
survival. But this α-vs-coverage-confusion is itself the
load-bearing concept correction this orientation makes: at α =
0.088 with `coverage@8` = 0.40 (hypothetical), tree-shape may
still extract a credible (1b)-clearing lift. β.2 is the only
way to distinguish this from the C.4-style structural mismatch.

## 4. Where could `coverage@b ≥ 0.30` come from?

C.5 does not have to reuse C.4's DFlash drafter. The drafter is
pluggable. Three accessible pairings against the 4-bit target:

1. **C.1 small-Qwen drafter** (`Qwen/Qwen3.5-0.8B` or
   `Qwen/Qwen3-0.6B` against `mlx-community/Qwen3.5-27B-4bit`).
   The v1.7.19 (h) closure registered
   `qwen3.5-27b-warm-decode-spec-on` (target = 27B-4bit, drafter
   = 0.8B Qwen3.5) and the foundation-test suite validates the
   wiring — but **this row has not been executed against the
   real cached 27B target.** The `coverage@b` profile from
   this pairing is the single most decision-relevant unmeasured
   distribution in P-6 today; even if linear α (= `coverage@1`)
   is low, the `coverage@8` / `coverage@16` shape may still
   admit a (1b)-clearing tree-shape projection. Acquiring both
   numbers costs one bench run (β.1) plus one read-only probe
   (β.2) on existing cached checkpoints; no new download, no
   spec-engine code, no DDTree code.
2. **C.4 DFlash drafter** (`z-lab/Qwen3.5-27B-DFlash`). C.4
   measured `coverage@1 = 0.088` and was retired. Tree-shape
   *might* extract lift from this drafter if `coverage@8` or
   `coverage@16` is materially higher than `coverage@1` — i.e.
   if the DFlash drafter is "directionally right but greedy-
   wrong." This is testable by a separate β.2-shaped probe
   against the DFlash drafter, but is **out of scope here**:
   even if the DFlash `coverage@8` is high, finding 2 from §2
   (drafter cost 35.7 ms vs verify 2.45 ms) would still gate
   the silica-integrated projection unless the drafter cost
   itself can be reduced. Documented for completeness.
3. **Future trained-against-quantised-target drafter.** Out of
   scope here — would require drafter conversion / training
   work, which is exactly the implementation pit the user
   asked to avoid pre-decision. C.6 QuantSpec-like self-spec
   is a structurally adjacent option (drafter = quantised
   draft of the target itself); same gate would apply.

The β sub-units below acquire (1) and stop there.

## 5. Verify-kernel cost scaling

P-6.0.5 Unit 7 measured **linear k=8 verify** at 2.93×
target-side / zero-drafter-cost. Tree-verify with total token
count T = b · k requires either:

- A **non-causal attention mask** (ancestor-only mask: each
  draft token attends only to its tree-path ancestors). The
  upstream `humanrouter/ddtree-mlx` repo claims a Metal kernel
  for this; the C.5 entry in `plans/P6_OPENING.md` §C.5 (line
  395) records the claim. **Not verified by silica.**
- A fall-back of running the verify forward sequentially per
  branch, which destroys the parallelism that makes tree-shape
  worth doing.

If the upstream tree-attention kernel works under MLX without
torch (D-009) and produces verify-cost scaling sub-linear in
T (e.g. T=32 verify ≈ 1.5-2× single-token verify rather than
32×), then tree-shape's lift is preserved. **Survey of this
kernel is γ.1, not orientation scope.**

## 6. Sub-unit re-frame

Original P-6 / P-6.0 wording placed C.5 as "DDTree integration
on top of C.4 DFlash drafter." Post-C.4-retirement, this
ordering is wrong: it commits to implementation before
verifying that any drafter pairing carries enough top-b
coverage to make tree-shape worth doing. The re-framed
sub-units below run **coverage-checking** ahead of integration:

- **α (theoretical sufficiency check — this doc).** Establishes
  that the gating empirical metric is `coverage@b`, not linear
  α; documents the independence-bound envelope as a *non-gate*
  upper bound; carries the C.4 cost-model finding forward only
  for the DFlash pairing. Closes when the orientation lands.
  **Status: open this commit.**

- **β.1 — existing spec-on row, single bench run, no new code.**
  Run the cached `qwen3.5-27b-warm-decode-spec-on` scenario
  registered at v1.7.19 (h):

  ```text
  SILICA_REAL_QWEN3_5_27B=1 SILICA_REAL_QWEN3_5_0_8B_DRAFT=1 \
      uv run python -m scripts.bench --speculative draft_target \
          --scenario qwen3.5-27b-warm-decode-spec-on \
          --out plans/P6_C5_DDTREE/spec_on_b1_alpha_probe.jsonl
  ```

  Outputs `SpecMetricCollector` metadata: linear `accept_rate`,
  `draft_cost_ms`, `verify_cost_ms`, `tokens_per_target_forward`,
  `rollback_count`, `peak_memory_mb`. **Decision-relevant
  numbers:**
  - linear α (= `accept_rate`, = `coverage@1`) — first data
    point on the coverage curve.
  - `draft_cost_ms` — replaces C.4's 35.7 ms in the
    silica-integrated cost projection. If 0.8B drafter cost is
    materially lower, the (1b) gate envelope opens.
  - `rollback_count` / `tokens_per_target_forward` — informs
    realistic per-cycle overhead the way C.4 (η.1) §
    Interpretation did.

- **β.2 — top-b coverage probe, stand-alone read-only script,
  no spec engine.** A dedicated probe at
  `scripts/probe_c5_top_b_coverage.py` (or analogous) that:

  1. Loads cached `mlx-community/Qwen3.5-27B-4bit` (target) and
     `Qwen/Qwen3.5-0.8B` (drafter).
  2. **Tokenizer alignment guard.** Encode the corpus with one
     tokenizer and assert
     `target.tokenizer.encode(text) == drafter.tokenizer.encode(text)`
     for every prompt; assert
     `target.tokenizer.vocab_size == drafter.tokenizer.vocab_size`
     and the underlying merge rules match (Qwen3 / Qwen3.5
     family share the same `tokenizer.json`, but the assertion
     stays — a silent vocab mismatch would make `coverage@b`
     meaningless because rank comparisons would be over
     different token id spaces). The probe **must fail loud**
     on any tokenizer divergence rather than silently coerce.
  3. For a fixed prompt corpus (suggested: WikiText-2 first
     ~512 tokens, same fixture B.2 used; alternatively a small
     synthetic prompt set) and N decode positions (suggested:
     N ≥ 256 across 2-3 prompts):
     a. Run the target greedy-forward to obtain
        `target_argmax[i]` at each position.
     b. Run the drafter teacher-forced over the **same token-id
        prefix** (the alignment guard above ensures this is
        well-defined) to obtain the drafter's full logit
        vector at each position.
     c. Compute `rank[i]` = position of `target_argmax[i]` in
        the drafter's sorted logits (descending). Both
        argmax-id and rank-id are interpreted in the shared
        token id vocabulary.
  4. Aggregates `coverage@b` = (1/N) · Σ I[rank[i] < b] for
     b ∈ {1, 4, 8, 16, 32}.
  5. Emits a JSONL row with the coverage values plus per-
     position rank histogram (for distribution shape) and the
     tokenizer-alignment attestation (vocab size, sample-text
     encoding hash, tokenizer config hash).

  No spec engine, no rollback, no verify path — just two
  parallel teacher-forced forwards with rank comparison. Reads
  only existing cached weights (no new download). The probe
  script is the only new code in β.2; it does not touch
  `silica.speculative.*` and uses no DDTree primitives.

  **Decision-relevant numbers:**
  - `coverage@1` is a **same-order sanity check** against
    β.1's linear `accept_rate`; large divergence is a
    diagnostic signal (likely tokenizer drift, prompt-corpus
    mismatch, free-running-vs-teacher-forced regime difference,
    or rollback-induced sampling drift in β.1) rather than an
    automatic failure of either probe. Equality is **not
    expected** — β.1 is free-running through the spec engine
    with rollback / bonus-token / context drift, while β.2 is
    teacher-forced on a fixed corpus.
  - `coverage@4`, `coverage@8`, `coverage@16` are the load-
    bearing tree-shape gate inputs.

- **β decision gate.** Combines β.1 + β.2 across the full
  `b ∈ {4, 8, 16}` coverage profile (not a single b — see §9
  for the matrix):
  - **`coverage@b` ≥ 0.30 for some b ≤ 16** AND drafter cost
    from β.1 admits a credible (1b)-clearing projection
    (per-cycle envelope ≥ 3.74× under measured cost) → γ.1
    proceeds with explicit user authorisation.
  - **`coverage@4`, `coverage@8`, `coverage@16` all in
    [0.15, 0.30)** → escalate; user decides whether a smaller
    drafter, a different drafter family, or C.6 (QuantSpec-
    like self-spec) is worth pursuing as a C.5 substitute.
    Default: retire (1b).
  - **`coverage@4` AND `coverage@8` AND `coverage@16` all <
    0.15** → **(1b) retires entirely.** Close §13 step 8
    without a DDTree port. Asymmetric outcomes — e.g.
    `coverage@16` ≥ 0.15 but `coverage@4` and `coverage@8` < 0.15
    — also escalate rather than retire: depth alone may admit a
    tree-shape projection under specific kernel-cost
    assumptions, so the retirement bar is "no usable coverage
    at any branching factor."

- **γ.1 (DDTree upstream survey — read-only).** Only if β
  passes: read `humanrouter/ddtree-mlx` README, license,
  no-torch attestation, MLX-native kernel layout, drafter
  registry. Same shape as C.4's α sub-unit
  (`plans/P6_C4_DFLASH/REPORT.md` α). No download.
- **γ.2 (tree-verify kernel + propose API).** Only if γ.1
  passes: implement minimal `silica.speculative.ddtree` that
  routes through the existing `Engine.generate` spec path
  with a tree-shape verify forward.
- **δ (real-checkpoint attestation).** Only if γ.2 passes:
  re-run the spec-on warm-decode rows with tree-verify
  enabled; gate on ≥3.74× silica-integrated speedup
  projection.

Each sub-unit closes with PASS / FAIL on a pre-declared
criterion. Implementation does not start until α + β.1 + β.2 +
γ.1 all return PASS. **Total commitment from this commit alone
is α + β.1 + β.2** (this doc + one cached-only bench run + one
cached-only read-only probe script).

## 7. Hard non-goals at orientation

- Do not write any `silica.speculative.ddtree` code.
- Do not download `humanrouter/ddtree-mlx` weights or kernels.
- Do not retrain or re-quantise the C.1 0.8B drafter.
- Do not propose a custom MLX tree-attention kernel; if the
  upstream MLX kernel doesn't work, that's a finding, not a
  silica engineering task.
- Do not write a custom MLX kernel for the β.2 coverage probe.
  Use the existing `silica.models.factory.adapter_for_repo`
  forward calls (the same primitives that drive
  `tests/test_qwen3_5_*` real-target verification).
- Do not relax the (1b) ≥60 tok/s gate to fit a partial
  measurement. If β.2 returns `coverage@4`, `coverage@8`, AND
  `coverage@16` all below 0.15, retire (1b); the user-
  declared bar is the bar. Asymmetric outcomes (e.g. depth
  high, narrow branches low) escalate rather than auto-retire,
  per §9.
- Do not collapse β.1 and β.2 into a single bench row by
  threading rank-recording into the spec engine. β.2 is
  intentionally outside the spec path so the metric is clean
  and decision-relevant before any spec-engine integration cost
  has been spent.

## 8. Open questions

- **OQ-1: What is `coverage@b` for b ∈ {4, 8, 16} on the cached
  `mlx-community/Qwen3.5-27B-4bit` × `Qwen/Qwen3.5-0.8B`
  pairing across a representative decode corpus?** This is the
  single most decision-relevant unmeasured number in P-6 today.
  Resolved by β.2 — read-only probe on existing cached weights;
  no new download, no spec engine, no DDTree code. Linear α
  (= `coverage@1`) appears in both β.1 (free-running spec
  metric) and β.2 (teacher-forced fixed corpus). The two are a
  **same-order sanity check on each other; equality is not
  expected** — β.1 includes bonus tokens, rollback / replay
  effects, and free-running context drift; β.2 is teacher-
  forced on a fixed prompt set. Large divergence between β.1's
  `accept_rate` and β.2's `coverage@1` is a diagnostic signal
  (likely tokenizer / corpus mismatch or rollback-induced
  sampling drift) rather than an automatic failure of either
  probe.
- **OQ-2: What does β.1 measure for `draft_cost_ms` and
  `rollback_count` on the 0.8B-drafter pairing?** The C.4
  conclusion "α=0.088 cannot rescue (1b)" assumed DFlash's
  35.7 ms drafter and ≈80 ms rollback. The 0.8B drafter is
  expected to be ~2.5× cheaper per propose forward, but
  rollback frequency depends on linear α and was never
  measured. Resolved by β.1 — single bench run.
- **OQ-3: Does upstream `humanrouter/ddtree-mlx` ship a
  no-torch MLX-native tree-attention kernel that produces
  sub-linear verify-cost scaling at T = 32?** Read-only
  survey; gated behind OQ-1 + OQ-2 passing.
- **OQ-4: If C.5 retires under OQ-1 / OQ-2, does any other
  (1b) survival path exist?** Standing answer per v1.7.18
  reframe: no — (1a) ≥40 tok/s primary is the floor, (1b) ≥60
  tok/s becomes a closed stretch entry in the changelog, and
  dense Track A engine fusion (1.10-1.15× compounding) is
  downgraded from "(1b) lever" to "compounding polish on (1a)".
- **OQ-5: Is C.6 QuantSpec-like self-spec a viable C.5
  substitute under low coverage?** The C.6 design uses a
  quantised draft against the full-precision target; reverse
  of C.4's pairing. Out of orientation scope; documented for
  future reference. Note that C.6 would need its *own* β.2
  coverage probe to justify implementation under the same gate.

## 9. Decision matrix

The gate is **measured `coverage@b` from β.2 across the full
profile `b ∈ {4, 8, 16}`** (not the independence-bound formula
in §3, and not linear α from β.1 alone). β.1 supplies the
cost-model leg of the projection (`draft_cost_ms`,
`rollback_count`); β.2 supplies the coverage leg. The
profile-shape rules below are deliberate: depth-only highs
(`coverage@16` ≥ 0.15 but `@4`/`@8` < 0.15) escalate rather
than retire because deeper trees can compensate under specific
kernel-cost assumptions.

| β.2 `coverage@b` profile (b ∈ {4, 8, 16}) | β.1 cost profile | γ kernel | C.5 disposition | (1b) disposition |
| ----------------------------------------- | ---------------- | -------- | --------------- | ---------------- |
| **`coverage@b` ≥ 0.30** for some b ≤ 16 | per-cycle envelope projects ≥3.74× | OK | implement γ.1 → γ.2 → δ | survives via leg-B; δ is the final attestation |
| **`coverage@b` ≥ 0.30** for some b ≤ 16 | per-cycle envelope < 3.74× even at high coverage | n/a | retire impl — drafter too expensive or rollback too frequent | retire (1b) |
| **`coverage@b` ≥ 0.30** for some b ≤ 16 | OK | broken / not no-torch | retire impl; flag as "kernel-blocked" | retire (1b) leg-B; survival path closes |
| any `coverage@b` ∈ [0.15, 0.30) at b ∈ {4, 8, 16}; nothing reaches 0.30 | any | n/a | escalate; user decides among smaller drafter, C.6, or retirement | contingent on user decision |
| `coverage@4` AND `coverage@8` AND `coverage@16` **all < 0.15** | any | n/a | retire — drafter is structurally mispredicting at every branching factor we can afford | **retire entirely** — close §13 step 8 without a DDTree port |

## 10. References

- `plans/P6_0_DECISION_GATE_1_OPENING.md` §2.3 — (1b)
  two-condition survival rule.
- `plans/P6_0_5_BASELINE/REPORT.md` Unit 7 — linear k=8 verify
  ceiling 2.93× target-side / zero-drafter-cost.
- `plans/P6_C4_DFLASH/REPORT.md` (η.1) — C.4 drafter cost +
  α + rollback findings carried into §2 above.
- `plans/P6_OPENING.md` §C.5 — original DDTree scope reference
  (`humanrouter/ddtree-mlx`, arxiv 2604.12989, claimed 8.2×
  upstream on Qwen3); reframed by this doc to "α-conditioned
  spike, not autopilot integration."
- `plans/P6_TRACK_B/REPORT.md` (B.2) + `plans/P6_TRACK_B_FOLLOWUP_SURVEY.md`
  — Track B retirement closes the (i) full-stack leg of (1b)
  survival.
