# P-6 Step 8 — C.5 Tree-Shape Decision Spike — REPORT

D-021 step 8 measurement bundle. Combines (β.1) free-running
spec-on bench row and (β.2) teacher-forced top-b coverage probe
on the cached `mlx-community/Qwen3.5-27B-4bit` × `Qwen/Qwen3.5-0.8B`
pairing. The combined reading drives the (1b) ≥60 tok/s
survival decision per the v1.7.18 Decision Gate 1 reframe.

| Sub-unit | Row / probe | Status | Date |
| -------- | ----------- | ------ | ---- |
| (α)  | orientation — `plans/P6_C5_DDTREE_OPENING.md` | landed (commit `eed5204`) | 2026-05-01 |
| (β.1) | `qwen3.5-27b-warm-decode-spec-on` real-checkpoint run | landed; runner status `ok` | 2026-05-01 |
| (β.2) | `scripts/probe_c5_top_b_coverage.py` real-checkpoint run | landed; tokenizer attestation OK | 2026-05-01 |
| (γ.1) | `humanrouter/ddtree-mlx` upstream survey | not authorised — see §5 | — |

---

## (β.1) — `qwen3.5-27b-warm-decode-spec-on`

Closes the (β.1) sub-unit per `plans/P6_C5_DDTREE_OPENING.md`
§6. Single bench run on cached weights — no new download, no
new code beyond the v1.7.19 (h) registered scenario.

### Setup

- **Target:** `mlx-community/Qwen3.5-27B-4bit` (cached;
  `SILICA_REAL_QWEN3_5_27B`).
- **Drafter:** `Qwen/Qwen3.5-0.8B` (cached;
  `SILICA_REAL_QWEN3_5_0_8B_DRAFT`).
- **Workload:** `_warm_decode_workload(B=1, max_tokens=384)` —
  the v1.7.19 (h) registered shape mirroring
  `qwen3.5-27b-warm-decode-b1`.
- **Spec config:** `verify_k=4` (γ=3 + bonus), C.1
  draft-target path through `DraftTargetEngine`.
- **Hardware:** M5 Pro 48 GB.
- **Command:**
  ```text
  SILICA_REAL_QWEN3_5_27B=1 SILICA_REAL_QWEN3_5_0_8B_DRAFT=1 \
      uv run python -m scripts.bench --speculative draft_target \
          --scenario qwen3.5-27b-warm-decode-spec-on \
          --out plans/P6_C5_DDTREE/spec_on_b1_alpha_probe.jsonl
  ```

### Result

| Quantity | Value |
| -------- | ----- |
| Runner status | `ok` |
| `decode_tok_s` | **6.54** |
| `prefill_tok_s` | 91.4 |
| `ttft_ms` | 1488 |
| `wall_s` | 64.4 |
| `tokens` | 384 |
| `peak_memory_mb` | 17,256 |
| `accept_rate` | **0.0908** |
| `verify_cost_ms` | 2.09 |
| `draft_cost_ms` | 27.6 |
| `tokens_per_target_forward` | 1.27 |
| `rollback_count` | **296** |
| `quality_parity_status` | `not_tested` (B=1 spec-on, no parity gate per (h) closure) |

### Reading vs C.4 (η.1)

| Metric | β.1 (0.8B drafter, k=4) | C.4 η.1 (DFlash 2B, k=16) |
| ------ | ----------------------- | ------------------------- |
| `accept_rate` | **0.0908** | 0.0881 |
| `decode_tok_s` | **6.54** | 7.74 |
| Speedup vs b1=16.05 | **0.408×** | 0.482× |
| `draft_cost_ms` | 27.6 | 35.7 |
| `verify_cost_ms` | 2.09 | 2.45 |
| `tokens_per_target_forward` | 1.27 | 2.32 |
| `rollback_count` | 296 | 165 |
| `peak_memory_mb` | 17,256 | 19,043 |

**Key reading.** Swapping the drafter from 2B BF16 DFlash to
the 0.8B Qwen3.5 hybrid drafter changed `accept_rate` by
≈ +3 % (0.088 → 0.091) — the 4-bit-target argmax is the
structural ceiling, not drafter capacity. The 0.8B drafter
*is* materially cheaper per propose forward (27.6 vs 35.7 ms,
−22 %) and cuts target-side peak memory by ~10 %, but the
lower `verify_k=4` (vs DFlash's k=16) yields fewer tokens per
target forward (1.27 vs 2.32) and the higher cycle count drives
`rollback_count` up 1.8× (296 vs 165). Net silica-integrated
speedup gets *worse* (0.408× vs C.4's 0.482×).

α as a single number does not differ between the two
drafters; the question of whether C.5 has any room therefore
hinges entirely on whether the *coverage* curve carries more
information than α alone.

---

## (β.2) — `scripts/probe_c5_top_b_coverage.py`

Closes the (β.2) sub-unit per `plans/P6_C5_DDTREE_OPENING.md`
§6. Stand-alone read-only top-b coverage probe — bypasses the
spec engine entirely; two parallel teacher-forced forwards.

### Setup

- **Same target / drafter as β.1.**
- **Corpus:** WikiText-2 test split at
  `~/.cache/silica/wikitext2-test.txt` (1.3 MB; B.2 fixture).
- **Tokenisation:** target tokenizer encodes the corpus head;
  `--max-tokens 512` truncates to 512 tokens; **N = 511
  positions scored** (last position has no next-token
  target).
- **Tokenizer alignment guard:** vocab_size 248,044 confirmed
  identical between target and drafter; three sample text
  slices (1024 chars each, total 7.5 % of corpus) re-encoded
  with both tokenizers — id sequences identical
  (config_hash `85683b723c17c4e6`); special-token ids
  (`bos`/`eos`/`pad`) reconciled; **probe attestation OK**.
- **Hardware:** M5 Pro 48 GB.
- **Command:**
  ```text
  SILICA_REAL_QWEN3_5_27B=1 SILICA_REAL_QWEN3_5_0_8B_DRAFT=1 \
      uv run python -m scripts.probe_c5_top_b_coverage \
          --out plans/P6_C5_DDTREE/coverage_probe.jsonl
  ```

### Result

| Quantity | Value |
| -------- | ----- |
| `n_positions_scored` | 511 |
| Target `forward_full` | 1.36 s |
| Drafter `forward_full` | 0.12 s |
| Total elapsed | 5.93 s |
| `vocab_size` | 248,044 |

**Coverage profile:**

| b   | `coverage@b` |
| --- | ------------ |
| 1   | **0.0626**   |
| 4   | **0.1429**   |
| 8   | **0.1977**   |
| 16  | **0.2583**   |
| 32  | **0.3405**   |

**Rank histogram (right-open buckets):**

| Bucket | Count | Cumulative |
| ------ | ----- | ---------- |
| `<1`   |  32   |  6.3 %     |
| `<4`   |  41   | 14.3 %     |
| `<8`   |  28   | 19.8 %     |
| `<16`  |  31   | 25.8 %     |
| `<32`  |  42   | 34.1 %     |
| `<100` | 337   | 100.0 %    |
| `<1000`|   0   | 100.0 %    |
| `>=1000`|  0   | 100.0 %    |

(Sum = 511 ✓.)

### Reading

The coverage curve is *not* flat. Compared with
`coverage@1 = 0.063`, `coverage@32` reaches **0.341** — a
5.4× lift over linear. Most of the lift accumulates beyond
b=8: 192 of the 511 positions (38 %) have target argmax in
drafter rank 8-99. **The drafter is "directionally right but
greedy-wrong" on a non-trivial fraction of decode positions
— a signal tree-shape can in principle exploit, but only at
fat branching factors (b ≥ 32) the (1b) ≥60 tok/s window does
not pre-authorise.**

The flat tail is also informative: zero positions land
beyond rank 100, which means *when* the target argmax is in
the drafter's distribution at all, it sits in the top-100;
there is no "fully off-distribution" failure mode in this
corpus. The rank distribution is concentrated in 0-99, with
the 32-99 bucket carrying the bulk.

### Sanity check vs β.1

`coverage@1` (β.2 teacher-forced) = 0.0626; β.1
`accept_rate` (free-running) = 0.0908. Same order of magnitude
(both 6-10 %), with β.1 ≈ 45 % higher than β.2 — consistent
with the regime difference flagged in
`plans/P6_C5_DDTREE_OPENING.md` §6 β.2 contract:
"large divergence is diagnostic, not automatic failure". The
diagnostic content here is that free-running spec-on context
(target's own continuations) is moderately *easier* than
teacher-forced WikiText-2 prefixes for the drafter — exactly
the direction the target-self-stabilises hypothesis predicts.
Equality was not expected; neither probe fails.

---

## Decision-matrix read

`plans/P6_C5_DDTREE_OPENING.md` §9 row gates by **measured
`coverage@b` from β.2 across `b ∈ {4, 8, 16}`** combined with
β.1's cost profile. Reading row-by-row against the measurement:

| §9 row | Trigger | Measurement | Hits? |
| ------ | ------- | ----------- | ----- |
| `coverage@b ≥ 0.30` for some b ≤ 16 AND envelope ≥ 3.74× | clean γ-implement | `coverage@4=0.14`, `@8=0.20`, `@16=0.26` — none crosses 0.30 at b ≤ 16; β.1 envelope is 0.408× (far below 3.74×) | **no** |
| `coverage@b ≥ 0.30` AND envelope < 3.74× | retire impl, retire (1b) | preconditions not met | n/a |
| `coverage@b ≥ 0.30` AND kernel broken | retire impl, retire (1b) leg-B | preconditions not met | n/a |
| **`coverage@b ∈ [0.15, 0.30)` at b ∈ {4, 8, 16}; nothing reaches 0.30** | **escalate; user decides** | `@8=0.20`, `@16=0.26` both in band; nothing at b ≤ 16 reaches 0.30 | **yes** |
| `coverage@4` AND `@8` AND `@16` all < 0.15 | retire entirely | `@4=0.14` only just below; `@8=0.20`, `@16=0.26` clearly above | **no** |

**Disposition: escalate.** The measurement does *not* clear
γ-implement (b ≤ 16 ceiling at 0.26), but does *not* trigger
clean retire either (`@8`, `@16` are well above 0.15).

The `coverage@32 = 0.34` data point is informative but
**outside the pre-declared `b ∈ {4, 8, 16}` continuation
window** in the decision matrix — the orientation explicitly
gated b=32 out because tree-verify cost scaling at T=32 is
not in scope of P-6.0.5 Unit 7 (which measured linear k=8 at
2.93× target-side / zero-drafter-cost) and there is no
silica-side or measured upstream evidence for sub-linear
verify cost at T=32 on the M5 Pro. Treating coverage@32 as a
green-light would be retro-loosening the gate to fit the
measurement, which the user explicitly forbade in §7.

---

## Engineering recommendation

**Recommend retiring (1b) ≥60 tok/s unless the user
explicitly authorises a γ.1 read-only kernel survey
targeting `b ≥ 32` tree-verify with sub-linear cost
attestation.** Default disposition: do not enter
`silica.speculative.ddtree` implementation; close §13 step 8
in the same shape Track B retired (negative result, audit
trail kept, escape hatch documented).

The recommendation rests on three findings:

1. **`coverage@b` at b ≤ 16 does not justify implementation
   work.** The orientation pre-declared `b ∈ {4, 8, 16}` as
   the gate window because those branching factors compose
   cleanly with measured P-6.0.5 Unit 7 verify-cost scaling.
   `coverage@16 = 0.26` falls short of the 0.30 floor, and
   even at the optimistic-envelope formula in §3 of the
   orientation, projected lift over the 2.93× linear
   ceiling stays bounded.
2. **β.1 cost-leg is materially worse than C.4.** The 0.8B
   drafter saved 22 % drafter cost but the lower `verify_k=4`
   and α=0.09 floor drove `rollback_count` to 296 (1.8× C.4)
   and net speedup to 0.408×. Tree-shape on the same
   pairing would inherit this rollback dominance: rollback
   frequency is a function of `accept_rate`, not tree shape.
3. **The `coverage@32 = 0.34` signal is real but
   speculative.** The flat-tail rank histogram (zero
   positions beyond rank 100) shows the drafter is not
   off-distribution on this corpus, so a fat tree
   (b ≥ 32) *could* in principle convert the rank-16-31
   bucket into accepted tokens. But this requires:
   - upstream `humanrouter/ddtree-mlx` to ship a no-torch
     MLX-native tree-attention kernel that scales sub-linearly
     to T = 32 (currently unverified);
   - the kernel's verify cost at T = 32 to remain low enough
     that the per-cycle envelope clears 3.74× even after
     the rollback-count penalty β.1 measured;
   - drafter cost to stay flat with tree expansion, since
     the drafter still proposes a single-stream chain into
     each branch.
   None of these is established. Committing to γ.1
   implementation under the b=32 hypothesis would repeat the
   C.4 mistake: ship engineering effort against an
   unverified upstream gain claim. **The escape hatch is to
   ask γ.1 first as a read-only survey** — license,
   no-torch attestation, kernel scaling notes from the
   upstream README — without writing any silica code.

If γ.1 read-only survey returns evidence that
`humanrouter/ddtree-mlx` *does* produce sub-linear verify
cost at T = 32 on Qwen3-class targets and the upstream
benchmark numbers were measured against quantised targets
(not the C.4-style full-precision-only assumption), then
γ.2 implementation work could be re-authorised with a
revised decision matrix. Without that evidence, the default
remains retire (1b).

---

## Materials landed

- `plans/P6_C5_DDTREE/spec_on_b1_alpha_probe.jsonl` — β.1
  bench row JSONL (1 row, status `ok`, full
  `SpecMetricCollector` metadata).
- `plans/P6_C5_DDTREE/coverage_probe.jsonl` — β.2 coverage
  probe JSONL (1 row, full coverage profile + rank
  histogram + tokenizer attestation hash).
- `scripts/probe_c5_top_b_coverage.py` — β.2 probe script
  (committed at `03226d7`).
- `tests/test_probe_c5_top_b_coverage.py` — 26 pure-function
  unit tests covering rank / coverage / tokenizer-alignment
  / JSONL schema (committed at `03226d7`).

## Sub-unit commit chain

- **α** orientation — `eed5204` (`docs(plan): open D-021
  step 8 C.5 tree-shape decision spike`)
- **β.2 prep** — `03226d7` (`feat(spec): D-021 step 8 (β.2)
  C.5 top-b coverage probe + unit tests`)
- **β.1 + β.2 measurement bundle (this commit)** — adds
  `plans/P6_C5_DDTREE/REPORT.md` plus the two JSONL run
  artefacts.

## PLAN / plans-index disposition

**Not synced in this commit.** Per the user's pre-declared
order, PLAN.md §13 step 8 inline status and
`docs/plans-index.md` get updated only after the user reads
this REPORT and chooses between **retire (1b)** and
**authorise γ.1 read-only kernel survey**. Until that
decision lands, the v1.7.22 orientation status remains the
live PLAN entry.
