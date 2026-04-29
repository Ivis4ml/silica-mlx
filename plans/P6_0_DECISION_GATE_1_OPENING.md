# P-6 Decision Gate 1 — opening doc (D-021 step 4)

| Field | Value |
| --- | --- |
| Phase | D-021 step 4 (Decision Gate 1) |
| Status | Open — opening doc only; PLAN.md `§6` / `§7` / `§13` sync lands as a separate commit after review |
| Last updated | 2026-04-29 |
| Predecessor | D-021 step 3 (P-6.0.5 measurement expansion) closed at v1.7.17 (commit `d640420`) |
| Successor | D-021 step 5 (Speculative foundation) |
| Source | this file (`plans/P6_0_DECISION_GATE_1_OPENING.md`) |
| Inputs | `plans/P6_0_5_BASELINE/REPORT.md` (cross-row data) + per-row `.md` interpretations |

---

## 0. TL;DR

P-6.0.5 produced enough evidence to **resolve the dense and MoE
acceptance reframings** that D-021 v1.7.14 left contingent on
empirical data. The recommended call:

- **Dense (1a) ≥40 tok/s remains the must-pass primary gate.** No
  change needed; the v1.7.14 framing already names this as the
  phase-exit anchor.
- **Dense (1b) ≥60 tok/s is reframed as a stretch gate with two
  survival conditions, either of which suffices**: (i) a later
  measured full stack (Track A × Track B × Track C with C.4 or
  C.5 landed) clears ≥60 tok/s on the (1a) workload, or (ii) a
  Track C.5 tree-shape spike demonstrates enough headroom over
  the linear k=8 verify ceiling that the full-stack projection
  including C.5 would credibly clear ≥60. C.4 alone — even at
  the upper end of its conservative MLX 2.0–2.9× band — does
  not settle (1b) survival; only the full-stack measurement or
  the C.5 spike does. The v1.7.14 wording's "C.4 or C.5 ≥2.5×
  alone clears (1b)" is what we tighten away from. **Not
  retired entirely** — see §4 for why premature retirement
  would discard live exploration ROI.
- **MoE (2a) ≥100 tok/s aggregate stays as the cleared anchor**
  (`[x]` in §6).
- **MoE (2b) is reduced to a single variant: ≥175 tok/s
  aggregate at B≥3.** The v1.7.14 OR-clause `≥150 aggregate OR
  ≥100 per-row at B=2` is replaced; the per-row arm is
  structurally unreachable on this checkpoint (per-row falls
  monotonically with B: 76 → 60 → 54 → 47), and the ≥150
  threshold has been cleared at B=3 already (163.5 tok/s) with
  thin information value remaining.
- **PLAN.md `§6` live contract is updated** in this phase, not
  only `§13` history. (1b) gating language and (2b) variant text
  are the load-bearing edits.

The opening proposes the calls; the actual PLAN.md edits land as
a separate commit after review. This doc is the audit trail for
why the gate was reframed this way given the P-6.0.5 numbers.

---

## 1. P-6.0.5 input anchors

This section names only the data points that constrain the
gate-1 decision. Full numerics are in
`plans/P6_0_5_BASELINE/REPORT.md`.

### 1.1 Dense Qwen3.5-27B-4bit

| anchor | value | source |
| --- | --- | --- |
| B=1 baseline | 16.05 tok/s @ 79.1% util | P-6.0 baseline (corrected anchor) |
| B=2 batch row | 31.22 tok/s, 97% of ideal 2× linear | P-6.0.5 Unit 1 |
| B=4 batch row | 42.17 ± 0.21 tok/s, 66% of ideal 4× linear | P-6.0.5 Unit 2 (2-run) |
| Verify-k k=8 ceiling | 2.93× target-side / zero-drafter-cost | P-6.0.5 Unit 7 |
| Warm TTFT | 317 ms (3-run, σ ≈ 0.3 ms) | P-6.0.5 Unit 6a |

Bandwidth utilisation **drops** with batch on dense (79% B=1 →
52% B=4); the regime is bandwidth-bound at B=1 and KV-traffic-
bound at B≥4. Batch-only path to 60 tok/s is dead. The verify-k
2.93× is a target-side ceiling with three structural assumptions
(zero drafter cost, 100% acceptance, linear shape); real
end-to-end speculative gain falls below by drafter forward
cost, drafter acceptance probability, and the bonus-token rule.

### 1.2 MoE Qwen3.5-35B-A3B-4bit

| anchor | value | source |
| --- | --- | --- |
| B=1 / B=2 baselines | 76.01 / 120.93 tok/s aggregate | P-6.0 baseline |
| B=3 row | 163.50 tok/s, 90% of B=2→B=3 linear | P-6.0.5 Unit 3 |
| B=4 row | 188.50 tok/s, 92% bandwidth util | P-6.0.5 Unit 4 |
| 4K-context B=1 | 85.0 tok/s, peak 23.6 GB | P-6.0.5 Unit 5 |
| Warm TTFT | 169 ms (1 run) | P-6.0.5 Unit 6b |

Bandwidth utilisation **climbs** with batch on MoE (37% B=1 →
92% B=4); regime is weight-read-bound throughout. **B=1 / B=2
had substantial headroom to the v1.7.13 active-weight ceiling;
B=4 has consumed most of it** (92% util, ≤8% remaining), so
B=5 / B=6 sit on the diminishing-returns side of the curve
(see §4.2). §6(2) ≥100 tok/s gate cleared by 88% at B=4.
Per-row throughput falls monotonically (76 → 60 → 54 → 47
across B=1/2/3/4).

### 1.3 What the data does not anchor

Three Decision Gate 1 inputs remain **unmeasured**; each shapes
the recommended call:

- **C.5 tree-shape verify cost.** P-6.0.5 Unit 7 measured linear
  k=1/2/4/8; tree-shape candidates (C.5 DDTree) were
  out-of-scope per the opening doc §2.2. The 2.93× ceiling is
  therefore a **linear-shape** bound, not a universal upper
  bound on speculative speedup against the dense target.
- **End-to-end speculative ROI.** The microbench measures only
  target verify cost. Drafter forward cost (e.g. 0.6B drafter
  on dense 27B target ≈ 1-3 ms/tok) and drafter acceptance
  probability (claimed bands 50-70% in the literature, MLX
  unmeasured) are step 5+6 deliverables.
- **B>1 + 4K-context combination.** Peak memory at MoE B=4 4K
  could intersect §6(4); not measured in P-6.0.5 (out-of-scope
  per opening §2.2).

---

## 2. Decision space

Three sub-decisions to resolve. Each lists arms with explicit
rejection rationales rather than a "recommended-only" framing,
so future re-readers can see what was considered and discarded.

### 2.1 Dense gate framing

(1a) ≥40 tok/s as must-pass primary is uncontested. The live
question is what to do with (1b).

| arm | call | rationale |
| --- | --- | --- |
| (D1) keep v1.7.14 wording (1b gated on generic "C.4 or C.5 ≥2.5×") | **reject** | too generic — C.4 alone at its upper-band 2.9× clears the threshold but does not settle (1b) reachability (PLAN.md §1.3 mid-stack `1.10 × 1.30 × 2.5 × 16.05 = 57.4 tok/s` falls short; upper-stack `1.15 × 1.30 × 2.9 × 16.05 = 69.6` clears). The trigger needs to bind to evidence of the actual reachable path, not a single component speedup. |
| (D2) retire (1b) entirely | **reject** | premature; C.5 tree-shape unmeasured and the upper-stack A+B+C.4 path is also still open (see §4.1) |
| (D3) demote (1b) to stretch with a **two-condition survival rule**: full-stack measurement clears ≥60 OR C.5 tree-shape spike shows headroom beyond the linear ceiling | **accept** | preserves both data-supported reachability paths (engineering full-stack and tree-shape exploration); phase exit anchors on (1a) regardless of (1b) outcome |

### 2.2 MoE (2b) variant resolution

The v1.7.14 framing offered `≥150 aggregate at B=2 OR ≥100
per-row at B=2`. P-6.0.5 data resolves the OR:

| arm | call | rationale |
| --- | --- | --- |
| (M1) keep both arms | **reject** | per-row 60.47 at B=2, monotonically falling with B — structurally unreachable; leaving the arm misleads future readers |
| (M2) aggregate ≥150 only | thin | cleared by B=3 (163.5) already; minimal information value |
| (M3) **aggregate ≥175 at B≥3** | **accept** | B=4 = 188.5 leaves 13.5 tok/s margin (7.7%); informative beyond v1.7.13 anchor |
| (M4) aggregate ≥200 at B≥3 | **reject** | requires B≥5 (thin marginal at 92% util) or unscheduled C-on-MoE work — see §4.2 |

The recommended (M3) retires the per-row arm explicitly rather
than leaving it as a hidden alternative.

### 2.3 (1b) survival trigger mechanism

If (1b) is reframed under (D3), what specific evidence triggers
survival vs retirement?

| arm | call | rationale |
| --- | --- | --- |
| (C1) (1b) gated by §7 step 6 generic ≥1.8× C-track speedup | **reject** | too permissive — C.4 alone at 2.0× clears step 6 but does not establish a path to 60 tok/s |
| (C2) **(1b) survives if either (a) a measured full stack (A × B × C with C.4 or C.5 landed) clears ≥60 on the (1a) workload, or (b) a C.5 tree-shape spike shows headroom beyond the linear k=8 ceiling sufficient to make the full-stack projection ≥60 credible** | **accept** | binds (1b) to the two reachability paths the data leaves open: empirical full-stack measurement, or the only verify-k regime not bounded by the 2.93× linear ceiling |
| (C3) retire (1b) now; reopen later if C.5 lands | **reject** | operationally identical to (D2); loses the "celebrate when met" framing in §6 |

(C2) is the practical refinement of (D3): the (1b) §6 entry's
"contingent on" language names two triggers explicitly rather
than the v1.7.14 single generic threshold.

### 2.4 PLAN.md update scope

(S1) §13 changelog only — **reject**, leaves §6 live contract
stale. (S2) §6 (1b) + (2b) rewrite + §7 D-021 step 4/6/8
language + §13 changelog v1.7.18 — **accept**. Full edit list
in §5.

---

## 3. Recommended call (consolidated)

The set of calls below summarises §2 with no new content; this
section exists so reviewers and PLAN.md editors have one
canonical paragraph to translate into the §6 / §7 / §13 edits.

1. **Dense (1a) ≥40 tok/s** remains the must-pass primary gate.
   No §6 wording change.

2. **Dense (1b) ≥60 tok/s** is preserved as a stretch gate
   with a **two-condition survival rule**, either of which
   suffices: **(i)** a measured full stack on the (1a) workload
   clears ≥60 tok/s (Track A × Track B × Track C with C.4 or
   C.5 landed; PLAN.md §1.3 upper-band stack
   `1.15 × 1.30 × 2.9 × 16.05 ≈ 69.6 tok/s` shows this is
   reachable in principle), **or (ii)** a Track C.5 tree-shape
   spike demonstrates enough headroom over the linear k=8
   verify ceiling that the full-stack projection including C.5
   would credibly clear ≥60. C.4 alone — even at the upper end
   of its conservative MLX 2.0–2.9× band — does not settle
   (1b). If neither trigger fires by the end of P-6, (1b)
   retires with a Decision Log entry naming the empirical floor.

3. **MoE (2a) ≥100 tok/s aggregate** stays as the cleared
   anchor (already `[x]` in §6 at the v1.7.13 B=2 baseline).

4. **MoE (2b)** is reduced to **`≥175 tok/s aggregate at B≥3`**.
   The v1.7.14 OR-clause and the per-row variant are removed.
   The §6 entry is simplified to a single threshold + workload
   shape statement.

5. **§7 D-021 step 6 (C.4 spike) gate language** is tightened
   from "`≥2.5× justifies pursuing the (1b) stretch`" to a
   dual-trigger formulation: ≥2.5× silica-integrated speedup is
   one component of the (1b) two-condition survival rule
   (feeding the full-stack measurement leg) **and** also
   motivates the C.5 tree-shape spike (the second leg); ≤1.8×
   retires (1b) only if no C.5 spike is pursued. Step 6 itself
   stays a C.4-only gate; the (1b) survival rule is owned by
   the C.5 spike step + the eventual full-stack measurement.

6. **PLAN.md §13 changelog v1.7.18** records the call, links
   this opening, and references the P-6.0.5 REPORT for the data.

---

## 4. Counter-arguments

The recommended call has three weak points. Each is worth
naming explicitly so that if future evidence changes the
trade-off, the reasoning re-opens cleanly.

### 4.1 Why not retire (1b) entirely

The strongest argument for outright retirement is the linear
k=8 ceiling at 2.93×: spec-only at the realistic upper band
gives `2.93 × 16.05 ≈ 47 tok/s`, well below 60.

**Why we do not retire**:

- The retirement argument is spec-only math. PLAN.md §1.3
  full-stack arithmetic at the upper band — `A 1.15× × B 1.30×
  × C.4 2.9× × 16.05` ≈ **69.6 tok/s** — clears 60 in
  principle on existing tracks already in scope. Mid-stack
  `1.10 × 1.30 × 2.5 × 16.05 ≈ 57.4 tok/s` falls short by 2.6
  tok/s; whether the path lands above or below 60 is an
  empirical question the C.4 spike + A landing answers, not a
  question the linear ceiling settles.
- C.5 tree-shape verify cost is **structurally different** from
  linear-k. A tree of width w and depth d with branch sharing
  can amortise the prefix forward cost across more candidate
  positions than a linear k=w·d slice; the literature claims
  8.2× on GPU (DDTree) and we do not know the MLX
  tree-amortisation factor. Even if the full stack on C.4 lands
  short of 60, C.5 tree shape may push the projection above.
- Outright retirement removes the "celebrate when met" framing
  that motivated split-gate framing in D-021 v1.7.14. The (1a)
  primary + (1b) stretch architecture exists *precisely* so the
  phase can exit cleanly on (1a) while preserving headroom for
  (1b) without blocking Track C ROI exploration.
- The cost of keeping (1b) as a contingent stretch is one
  paragraph of §6 wording. The cost of retiring it and later
  reopening is a Plan-level CRUD operation (`D-NNN` entry to
  un-retire) and the loss of explicit motivation for the C.5
  spike.

The retirement trigger we *do* accept: if the C.5 spike (or any
future tree-shape evidence) lands below the linear ceiling on
this checkpoint, (1b) retires with a Decision Log entry. That
keeps the test falsifiable; "retire on no evidence" is not
falsifiable, "retire on negative evidence" is.

### 4.2 Why not raise (2b) threshold to ≥200

B=4 measured 188.5 tok/s. Setting the threshold at ≥200 would
require either (a) MoE B=5/B=6 work, (b) speculative on MoE, or
(c) a Track A optimisation cleanly applicable to MoE.

- **(a)** MoE is at 92% bandwidth utilisation at B=4; the
  marginal aggregate gain from B=5 is bounded above by ~10%
  (188.5 × 1.08 ≈ 204 best case, more likely 195–200). Memory
  is not the constraint (≥15 GB margin at B=4); diminishing
  returns are. Setting the gate at the bandwidth ceiling pushes
  optimisation toward the steepest part of the diminishing-
  returns curve, which is poor engineering ROI for stretch
  validation.
- **(b)** Track C-on-MoE is currently unscheduled (no PLAN.md
  step lists it); making the gate require it forces a scope
  expansion that v0.1 has not committed to.
- **(c)** Track A's MoE applicability is partial — sync collapse
  helps both families but the +30-80% leverage band assumes
  compute-bound regimes, and MoE at B=4 is *not* compute-bound
  (92% bandwidth util). Track A on MoE B=4 is at the lower end
  of its band, ~10-15%, which gets to ~205-215 only with stack
  alignment.

≥175 leaves a 13.5 tok/s margin (188.5 vs 175, 7.7%). That is
enough to absorb run-to-run variance and modest regression
without immediately failing the gate. ≥200 leaves no margin and
turns the stretch from a validation row into a hard engineering
target — a different phase deliverable than v1.7.14 envisaged.

### 4.3 Why update §6 / §7 contract, not just §13 history

PLAN.md treats §6 (Acceptance Gates) and §7 (Phase 0 / Phase 6
sequencing) as **live contracts** — future Track A / B / C work
reads them as the current target set, not as historical
artefacts. §13 changelog records *why* a contract changed;
§6 / §7 record *what the contract is now*.

Leaving the v1.7.14 (1b) wording ("contingent on C.4/C.5
≥2.5×") in §6 while writing "we now bind (1b) to C.5 tree-shape
specifically" only in §13 silently de-syncs the phase: a
reader landing on §6 sees one contract, a reader of §13 sees
another. Track C.4 work in particular reads §7 step 6 to know
its gate; if step 6 still says "justifies pursuing (1b)" with
the old binding, the spike outcome cannot cleanly trigger the
new (1b) retirement rule.

The cost of doing the live-contract update is one PLAN.md edit;
the cost of *not* doing it is a documented inconsistency that
each downstream phase has to re-resolve from §13 history.

---

## 5. PLAN.md change list

The opening lands as one commit by itself (this file). The
PLAN.md sync follows as a separate commit after review. No
silica.* code change.

| Section | Edit |
| --- | --- |
| §1 status header | Append v1.7.18 clause: "Decision Gate 1 (D-021 step 4) closed at v1.7.18 — (1a) ≥40 primary unchanged; (1b) ≥60 reframed as stretch with two-condition survival (full-stack measurement clears ≥60, or C.5 tree-shape spike shows headroom beyond the linear k=8 ceiling); (2b) ≥175 aggregate at B≥3 (per-row retired)". |
| §6 (1a) entry | No wording change. |
| §6 (1b) entry | Rewrite "Reaching this requires Track C.4 DFlash or C.5 DDTree to land in the upper half of their MLX-conservative bands (≥2.5× silica-integrated...)" → "**Reaching this requires either (i) a measured full stack on the (1a) workload (Track A × Track B × Track C with C.4 or C.5 landed) clearing ≥60 tok/s, or (ii) a Track C.5 tree-shape spike demonstrating headroom over the linear k=8 verify ceiling (P-6.0.5 Unit 7, 2.93× target-side / zero-drafter-cost) sufficient to make the full-stack projection ≥60 credible. C.4 alone — even at the upper end of its conservative MLX 2.0–2.9× band — does not settle (1b); only the full-stack measurement or the C.5 spike does.**" Update the "Per Q-C resolution..." sentence to reference the two-condition rule. |
| §6 (2a) entry | No wording change; optional cross-ref to Unit 4 confirming anchor still holds at B=4. |
| §6 (2b) entry | Rewrite "≥150 tok/s aggregate at B=2 OR ≥100 tok/s per-row at B=2..." → "**≥175 tok/s aggregate at B≥3**. P-6.0.5 Unit 4 measured B=4 = 188.5 at 92% bandwidth utilisation; ≥175 leaves margin for run-to-run variance while remaining informative beyond the cleared v1.7.13 (2a) anchor." |
| §7 D-021 step 4 | Append closure note pointing at this opening; record decision summary. |
| §7 D-021 step 6 | Tighten "≥2.5× justifies pursuing the (1b) stretch" → "≥2.5× silica-integrated speedup is one component of the (1b) two-condition survival rule (full-stack measurement) and also motivates the C.5 tree-shape spike that is the second component; ≤1.8× retires (1b) only if no C.5 spike is pursued". |
| §7 D-021 step 8 | Add a sentence identifying the C.5 tree-shape spike as the second-condition (1b) survival trigger (alongside the full-stack measurement triggered by step 6 / 7); existing C.5 description becomes prerequisite framing for the spike, not the gate itself. |
| §13 changelog v1.7.18 | ~30-line entry: §3 consolidated decision, four-arm rejection summary, sources (this opening + REPORT.md), toolchain attestation (no code change, tests stay green). |

---

## 6. Acceptance / phase exit

D-021 step 4 exits when:

1. This opening doc is committed.
2. PLAN.md sync per §5 is committed (separate commit; reviewer
   pass between).
3. `§7 D-021 step 4` shows status closure with a back-reference
   to this opening.

The Decision Gate 1 has **no on-device measurement** and **no
silica.* code change**. The phase is doc-only and does not run
the bench suite or change tests.

---

## 7. Open questions

These do not block step 4 closure but flag the next places the
gate may need to re-open:

- **OQ-1 — C.5 spike ROI threshold (second-leg trigger).** The
  recommended (1b) survival rule's second leg ties to "C.5
  tree-shape spike shows headroom over the linear k=8 ceiling
  sufficient to make the full-stack projection ≥60 credible".
  What does "sufficient" mean precisely — a fixed multiplier
  over the linear ceiling (e.g. ≥1.1× of 2.93× = 3.22×), a
  threshold relative to the measured C.4 outcome (e.g. ≥1.3×
  of C.4), or a back-computed minimum that — folded into the
  full-stack arithmetic with whatever A and B landed — pushes
  the projection above 60? The C.5 spike opening doc (when it
  lands) needs to fix this; the present opening only commits to
  the qualitative "headroom beyond the linear ceiling" framing.

- **OQ-2 — (2b) ratchet.** If MoE B=5 work happens later
  (Track A optimisation, additional measurement), should ≥175
  ratchet up? Recommendation: do not ratchet automatically; let
  any future Decision Gate that re-opens (2b) make the call
  explicitly.

- **OQ-3 — §7 step 6 vs C.5 sub-step ordering.** D-021 step 6
  is the C.4 spike gate; the C.5 spike is currently inside step
  8 (selection). The recommended call moves the (1b) binding to
  the C.5 spike specifically, which may justify promoting the
  C.5 spike from step 8 sub-item to its own step (between step
  6 and step 7 / between step 6 and step 8). This is a §7
  re-ordering question, not a gate-1 question; flagged so the
  step 5+6 work knows to revisit.

---

## 8. Cross-references

- **PLAN.md §6 Acceptance Gates** — the live contract this
  phase updates ((1a)/(1b)/(2a)/(2b) checklist).
- **PLAN.md §7 D-021 step 4** — this phase; closes step 4 with
  back-reference to this opening.
- **PLAN.md §7 D-021 step 6** — C.4 spike gate; tightened by
  this phase (see §5.2).
- **PLAN.md §7 D-021 step 8** — C.5/C.2/C.3 selection; carries
  the (1b) gate post-this-phase (see §5.2 and OQ-3).
- **PLAN.md §13 Changelog v1.7.18** — added by the PLAN.md
  sync commit.
- **`plans/P6_0_5_BASELINE/REPORT.md`** — input data; the four
  §1 question closures it documents are the empirical anchors
  for this gate.
- **`plans/P6_0_5_OPENING.md`** — predecessor phase opening;
  cross-references §1 questions answered.
- **`plans/P6_OPENING.md`** — original P-6 framing; the v0.1
  user-stated 60 tok/s target this phase reframes as stretch.

---

## 9. Non-goals (recapped, for the record)

- No silica.* code change.
- No new bench scenario or oracle.
- No on-device measurement; this is a doc-only phase.
- No retirement of (1b) — the call is reframe, not delete.
- No (1a) threshold change.
- No commitment to a specific C.5 spike landing date or scope.
- No Track A / B / C-on-MoE scope expansion to clear (2b)
  ≥200 (see §4.2).
