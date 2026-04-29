# P-6.0.5 Opening — Measurement Expansion for Decision Gate 1

| Field         | Value                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| Phase         | P-6 (Performance Phase) sub-step 0.5; D-021 step 3                                                           |
| Status        | drafted; pending user review                                                                                 |
| Last updated  | 2026-04-28                                                                                                   |
| Scope owner   | Xin Zhou                                                                                                     |
| Predecessors  | P-5 complete; P-6.0 measurement gate landed at v1.7.13; P5.9 hardening complete at v1.7.15                   |
| Successors    | D-021 step 4 (Decision Gate 1) — fixes (1a)/(1b) dense gate framing and the MoE stretch shape                |

This document opens P-6.0.5 — the measurement-expansion sub-step
between P5.9 hardening (closed at v1.7.15) and Decision Gate 1
(D-021 step 4). P-6.0.5 fills the **specific data gaps** that Decision
Gate 1 needs in order to commit to either retaining or retiring the
dense (1b) ≥60 tok/s stretch gate. It adds **no new algorithmic
content** — every deliverable is a measurement on existing models
and the existing engine.

The phase exits when seven artefacts are on disk under
`plans/P6_0_5_BASELINE/`. No code on the engine hot path changes.

---

## 0. TL;DR

P-6.0.5 lands seven measurement artefacts:

1. `qwen3.5-27b-warm-decode-b2.{jsonl,md}` — dense 27B B=2.
2. `qwen3.5-27b-warm-decode-b4.{jsonl,md}` — dense 27B B=4 (opt-in,
   OOM-flagged; mirrors the MoE B=4 run sequence).
3. `qwen3.5-moe-35b-a3b-warm-decode-b3.{jsonl,md}` — MoE B=3.
4. `qwen3.5-moe-35b-a3b-warm-decode-b4.{jsonl,md}` — MoE B=4
   (opt-in, OOM-flagged). **No new code** — scenario already
   registered at v1.7.13 (`silica/bench/scenarios.py:2032`); only
   the data row is missing from `plans/P6_0_BASELINE/` and lands
   here.
5. `qwen3.5-moe-35b-a3b-warm-decode-b1-4k.{jsonl,md}` — MoE B=1
   sustained 4K-context probe.
6. `qwen3.5-27b-warm-ttft-pair.{jsonl,md}` (and one MoE counterpart)
   — new `WARM_TTFT_PAIR` oracle, two-prompt warm-TTFT measurement.
7. `target_verify_microbench.{jsonl,md}` — independent harness; not
   a `Scenario`. Measures one verify forward (input length =
   `verify_k`) over a primed prefix KV for
   `verify_k ∈ {1, 2, 4, 8}` on dense Qwen3.5-27B-4bit.

Existing dense 27B B=1 / B=1-4K / B=1-8K (registered under P-6.0 +
P5.9 step 2(d)) are reused as-is; **dense 27B B=2 4K is explicitly
out of scope** (see §2). The microbench is the prerequisite for
credible Track C ROI and is the load-bearing input to D-021 step 4.

---

## 1. Motivation — what data Decision Gate 1 needs

D-021 step 4 (Decision Gate 1) commits the phase to one of:

- **retain (1b) ≥60 tok/s dense stretch** — pursue C.4 / C.5
  block-diffusion drafters with a ≥1.8× silica-integrated speedup
  lower bound; or
- **retire (1b) to stretch-only**, anchor the dense gate on
  (1a) ≥40 tok/s engineering, and update the MoE stretch from
  ≥100 tok/s aggregate (already met at P-6.0 baseline) to
  ≥150 tok/s aggregate or ≥100 tok/s per-row at B=2.

The current data on disk does not let us choose between those
branches. The four open questions:

1. **Does dense 27B scale with batch?** P-6.0 only measured B=1.
   If B=2 at 70.6% utilisation jumps to >90%, KV/activation
   bandwidth is unsaturated and the C.x ROI estimate must be
   adjusted; if B=2 utilisation falls, the chip is bandwidth-bound
   even with batch and (1b) is harder to reach.
2. **Does MoE saturate before B=4, and is B=4 OOM-safe?** P-6.0
   baseline showed MoE 59.1% utilisation at B=2 and 121% of the
   §6(2) gate; B=4 was deliberately skipped at v1.7.13 over OOM
   risk. B=3 (a low-risk row) tells whether the MoE stretch can
   be re-anchored to a per-row quantity. B=4 (opt-in) directly
   answers whether the (2b) ≥150 tok/s aggregate stretch is
   reachable at all on a 48 GB envelope.
3. **Is 4K-context safe on MoE?** The §6(4) RAM-headroom gate is
   currently evaluated only on dense 27B at 4K. If MoE 4K is
   uncomfortably close to the 48 GB envelope, the (2b) MoE
   stretch reframing has to account for context length.
4. **What is the verify-k cost curve?** Block-diffusion drafters
   (C.4 / C.5) move work from the autoregressive path to a
   target-side multi-token verify forward. Without a measured
   cost curve, the C.4 spike's reported speedup cannot be
   attributed between drafter quality and verify amortisation,
   and ROI estimates are unfalsifiable.

P-6.0.5 closes all four with measurement only; no engine change.

---

## 2. Scope and out-of-scope

### 2.1 In scope (seven artefacts under `plans/P6_0_5_BASELINE/`)

| # | Artefact                                                  | New code surface                                                            |
| - | --------------------------------------------------------- | --------------------------------------------------------------------------- |
| 1 | `qwen3.5-27b-warm-decode-b2.{jsonl,md}`                   | scenario registration only                                                  |
| 2 | `qwen3.5-27b-warm-decode-b4.{jsonl,md}` (opt-in)          | scenario registration only                                                  |
| 3 | `qwen3.5-moe-35b-a3b-warm-decode-b3.{jsonl,md}`           | scenario registration only                                                  |
| 4 | `qwen3.5-moe-35b-a3b-warm-decode-b4.{jsonl,md}` (opt-in)  | **none** — scenario exists at `silica/bench/scenarios.py:2032`; data only   |
| 5 | `qwen3.5-moe-35b-a3b-warm-decode-b1-4k.{jsonl,md}`        | scenario registration only (reuses extended-ctx workload)                   |
| 6 | `qwen3.5-27b-warm-ttft-pair.{jsonl,md}` + MoE counterpart | new `WARM_TTFT_PAIR` oracle + scenario registration                         |
| 7 | `target_verify_microbench.{jsonl,md}`                     | independent harness; not a `Scenario`                                       |

### 2.2 Out of scope (deliberately deferred)

- **Dense 27B B=2 4K-context.** Reusing the user's framing for the
  record:

  > P-6.0.5 reuses existing dense 27B B=1 4K data for the
  > RAM-headroom gate. The new 4K-context expansion is MoE B=1 4K.
  > Dense 27B B=2 4K is explicitly out-of-scope unless B=2 / B=4
  > scaling or memory data reveals a contradiction.

  Rationale: the §6(4) RAM-headroom gate already has its anchor in
  `qwen3.5-27b-warm-decode-b1-4k` (registered v1.7.15 at
  `silica/bench/scenarios.py:2098`). The unmeasured surface is
  MoE 4K (per the inline comment at the same file lines 2092-2096).
  P-6.0.5's purpose is to fill measurement gaps that block Decision
  Gate 1, not to widen the dense long-context matrix.

- **MoE B=3 4K.** Sustained MoE long-context at B>1 stays deferred;
  if MoE B=3 short-context shows any §6(4) tension, we revisit.
- **Gemma4-31B B=2 / 4K extensions.** Gemma4 is a corroboration
  family for §6, not a primary gate target; new B=2 / 4K rows
  on Gemma4 are not on the gate path.
- **Engine changes.** Sampler, sync, snapshot — none of these are
  touched in P-6.0.5. Track A landings happen post-Decision-Gate-1.
- **Prefix-cache reuse measurement under HTTP.** Belongs to P-8;
  the `WARM_TTFT_PAIR` oracle below records `prefix_hit_tokens`
  for diagnostic purposes only, not as an engine validation.

### 2.3 Out of scope (explicit non-goals)

- No PPL / quality oracle additions. P-6.0.5 is timing + memory
  only; quality is owned by P-5 oracles.
- No new family. Gemma4-MoE B=3 / B=4 not added; if (2b) MoE
  reframing demands it post-Gate, that lands in a follow-up unit.

---

## 3. Sub-units

### 3.1 Sub-unit 1 — Dense 27B B=2

**Scenario id:** `qwen3.5-27b-warm-decode-b2`
**Repo:** `mlx-community/Qwen3.5-27B-4bit`
**Workload:** `_warm_decode_workload(max_batch_size=2, max_tokens=384)`
**Oracle:** `WARM_DECODE` (existing)
**Gate env:** `SILICA_REAL_QWEN3_5_27B`
**Description (registration text):** dense 27B B=2 — fills the
P-6.0 batch-scaling gap. Reads against the B=1 baseline at
70.6% utilisation, 16.05 tok/s. If aggregate tok/s rises with
B (utilisation rises), KV/activation traffic was the slack; if
aggregate stalls, the chip is bandwidth-bound even with batch.

**Artefact:** `plans/P6_0_5_BASELINE/qwen3.5-27b-warm-decode-b2.{jsonl,md}`.

**Acceptance:** the scenario completes without OOM on a 48 GB
M5 Pro and the JSONL row contains the standard `WARM_DECODE`
fields — top-level `ttft_ms`, `prefill_tok_s`, `decode_tok_s`
(= warm aggregate), `peak_memory_mb`, `total_tokens`, plus
`metadata.decode_tok_s_warm_aggregate`,
`metadata.decode_tok_s_warm_per_row_mean`, and
`metadata.rows[*].decode_tok_s_warm` (per-row warm decode rate).
All field names and shapes follow the v1.7.13 baseline rows
under `plans/P6_0_BASELINE/qwen3.5-27b-warm-decode-b1.jsonl`.

### 3.2 Sub-unit 2 — Dense 27B B=4 (opt-in)

**Scenario id:** `qwen3.5-27b-warm-decode-b4`
**Repo:** `mlx-community/Qwen3.5-27B-4bit`
**Workload:** `_warm_decode_workload(max_batch_size=4, max_tokens=384)`
**Oracle:** `WARM_DECODE` (existing)
**Gate env:** `SILICA_REAL_QWEN3_5_27B`
**Description text mirrors `_QWEN3_5_MOE_WARM_DECODE_B4`:** dense 27B
B=4 is opt-in stretch and OOM-flagged. 13.5 GB weights + 4× KV /
activations sit uncomfortably close to the 48 GB envelope. **Run
sequence:** validate B=2 first; only then run B=4 on a freshly
booted Mac with no other GPU consumers, ideally with
`mx.metal.set_memory_limit(~42 GB)` set to fail fast rather than
let macOS swap. If B=4 OOMs, the row is recorded as `oom=true` in
the JSONL and the §6 gate falls back to B=2.

**Artefact:** `plans/P6_0_5_BASELINE/qwen3.5-27b-warm-decode-b4.{jsonl,md}`.

### 3.3 Sub-unit 3 — MoE 35B-A3B B=3

**Scenario id:** `qwen3.5-moe-35b-a3b-warm-decode-b3`
**Repo:** `mlx-community/Qwen3.5-35B-A3B-4bit`
**Workload:** `_warm_decode_workload(max_batch_size=3, max_tokens=384)`
**Oracle:** `WARM_DECODE` (existing)
**Gate env:** `SILICA_REAL_QWEN3_5_MOE`
**Description text:** intermediate row between MoE B=2 (120.93
tok/s aggregate, 59.1% utilisation, 19.68 GB peak) and MoE B=4
(opt-in, OOM-flagged at v1.7.13). B=3 measures whether MoE
saturates before B=4 (i.e. whether per-row throughput continues
to grow with B or plateaus around B=2-3). If B=3 lifts aggregate
above ~150 tok/s, the (2b) stretch can be reframed to a
per-row quantity; if B=3 stalls near B=2, the MoE acceptance shape
is set by B=2.

**Artefact:** `plans/P6_0_5_BASELINE/qwen3.5-moe-35b-a3b-warm-decode-b3.{jsonl,md}`.

### 3.4 Sub-unit 4 — MoE 35B-A3B B=4 (opt-in, no new code)

**Scenario id:** `qwen3.5-moe-35b-a3b-warm-decode-b4` (already
registered at `silica/bench/scenarios.py:2032`, v1.7.13).
**Code surface:** **none** in this sub-unit. The scenario was
landed at v1.7.13 with the OOM-flagged description text already
in place; it was deliberately not run as part of the P-6.0
baseline. P-6.0.5 captures the missing data row.

**Why it sits in P-6.0.5 and not in the v1.7.13 baseline.** The
v1.7.13 P-6.0 baseline skipped this row over OOM risk on a fresh
48 GB envelope; D-021 step 3 in PLAN.md §7 names "MoE B=3 / B=4
(B=4 OOM-flagged)" explicitly, so closing the P-6.0.5 exit set
without this row would leave Decision Gate 1 reading from an
incomplete MoE batch curve. Capturing it now — even as a
recorded OOM — is the audit trail the gate needs.

**Run sequence (mirrors `_QWEN3_5_MOE_WARM_DECODE_B4` description):**
validate B=2 first (already in `plans/P6_0_BASELINE/`); validate
B=3 (sub-unit 3) next; only then run B=4 on a freshly booted Mac
with no other GPU consumers, ideally with
`mx.metal.set_memory_limit(~42_000_000_000)` so the run fails
fast rather than letting macOS swap.

**Acceptance:** identical shape to sub-unit 3 — the JSONL row
either records standard `WARM_DECODE` fields and an aggregate
tok/s number, or records `oom=true` (with a non-empty `reason`
field) if the run does not fit. Either outcome closes the
artefact slot for this sub-unit.

**Artefact:** `plans/P6_0_5_BASELINE/qwen3.5-moe-35b-a3b-warm-decode-b4.{jsonl,md}`.

### 3.5 Sub-unit 5 — MoE 35B-A3B B=1 4K-context

**Scenario id:** `qwen3.5-moe-35b-a3b-warm-decode-b1-4k`
**Repo:** `mlx-community/Qwen3.5-35B-A3B-4bit`
**Workload:** `_warm_decode_workload_extended(max_batch_size=1,`
`max_tokens=600, target_context_tokens=4096)`
**Oracle:** `WARM_DECODE` (existing); `oracle_config = {
"target_context_tokens": 4096,
"expected_total_context_floor": 3500 }`
**Gate env:** `SILICA_REAL_QWEN3_5_MOE`
**Description text:** sustained 4K-context probe on MoE
35B-A3B-4bit. Mirrors the dense 27B 4K row registered at v1.7.15
(P5.9 step 2(d), `qwen3.5-27b-warm-decode-b1-4k`) but on the
active-3B MoE checkpoint. Surfaces the §6(4) RAM-headroom gate
under the MoE routing-state + KV-growth path, which has a
materially different memory profile from dense (256 experts ×
top-8, 19.4 GB peak at 384 tokens). Validates that the MoE
B=1 baseline path is safe at 4K context before any (2b) stretch
reframing on B>1 is contemplated.

**Artefact:** `plans/P6_0_5_BASELINE/qwen3.5-moe-35b-a3b-warm-decode-b1-4k.{jsonl,md}`.

### 3.6 Sub-unit 6 — Warm-TTFT pair oracle

**New oracle:** `OracleKind.WARM_TTFT_PAIR`.
**Module:** `silica/bench/oracles.py` (extend existing module).
**Contract:** the runner issues two prompts in the same engine /
session. Prompt 1 amortises one-time compile / kernel-cache
warmup; prompt 2's TTFT is the warm number reported by the §6
TTFT scenarios. The oracle returns one row per pair (not per
prompt), with the following JSONL fields:

| Field                    | Meaning                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| `prompt1_ttft_ms`        | TTFT of the first prompt (cold + compile cost folded in)           |
| `prompt2_ttft_ms`        | TTFT of the second prompt (warm; the §6 TTFT report value)         |
| `warm_ttft_ms`           | alias = `prompt2_ttft_ms` (explicit for downstream consumers)      |
| `compile_amortized_ms`   | `prompt1_ttft_ms - prompt2_ttft_ms` — diagnostic, not gate         |
| `prompt1_tokens`         | tokenised length of prompt 1 (prevents length-asymmetry pollution) |
| `prompt2_tokens`         | tokenised length of prompt 2                                       |
| `prefix_hit_tokens`      | radix-prefix-cache hit count on prompt 2 (0 if no shared prefix)   |

**Why those four diagnostic fields are mandatory.** A warm TTFT
number alone is uninterpretable: if `prompt2_tokens` differs from
`prompt1_tokens` the result is not a warm-vs-cold comparison but a
prompt-length comparison; if `prefix_hit_tokens > 0` the warm
TTFT is being inflated downwards by prefix reuse and not by
compile amortisation alone. Both diagnostics surface
contamination rather than silently absorbing it.

**Scenario registrations (two):**

- `qwen3.5-27b-warm-ttft-pair` — dense 27B, two distinct
  paragraphs hand-calibrated to ~128 BPE tokens each on the
  Qwen3 / Qwen3.5 tokenizer. The "128-token target" is a design
  anchor, not a hard equality: the tokenizer drift between the
  dense Qwen3.5-27B and the MoE Qwen3.5-35B-A3B checkpoints
  prevents an exact match across families. The catalog
  tokenizer-gated test asserts both prompts fall within ±15%
  (i.e. 109–147 tokens) on the dense Qwen3 tokenizer; the runner
  surfaces the **actual** measured `prompt1_tokens` /
  `prompt2_tokens` into every JSONL row so downstream consumers
  read ground truth rather than the design anchor. Distinct
  content keeps `prefix_hit_tokens` structurally 0 on the gate
  row.
- `qwen3.5-moe-35b-a3b-warm-ttft-pair` — MoE counterpart;
  identical prompt pair so cross-family warm-TTFT delta is a
  clean architecture-only signal modulo the tokenizer-induced
  prompt-length drift the JSONL row records.

A single optional **`-shared-prefix`** variant per family records
the case `prompt2 = prompt1` to expose the prefix-cache effect
under WARM_TTFT_PAIR. This is diagnostic, not gate-bearing.

**Artefacts:**
- `plans/P6_0_5_BASELINE/qwen3.5-27b-warm-ttft-pair.{jsonl,md}`
- `plans/P6_0_5_BASELINE/qwen3.5-moe-35b-a3b-warm-ttft-pair.{jsonl,md}`

### 3.7 Sub-unit 7 — Target-verification microbench

**Path:** independent harness, NOT a `Scenario` registration. Two
candidate placements; pick during implementation:

- `silica/bench/microbench/target_verify.py` (preferred — keeps
  bench tooling under `silica.bench.*` and lets
  `python -m scripts.bench` reach it later if catalogued); or
- `scripts/microbench_target_verify.py` (acceptable — mirrors
  existing `scripts/probe_*` placement convention).

**What it measures.** Silica's `ModelAdapter` (see
`silica/models/adapter.py:172`) exposes `prefill` + `decode_step`,
not a verify-shaped forward. There is no `target.forward(...,
candidate_tokens=verify_k)` API. The microbench therefore
synthesises the speculative-decoding verify shape directly:

1. Load `mlx-community/Qwen3.5-27B-4bit` once.
2. Build a fixed prefix (128 tokens) and a candidate slice of
   length `verify_k`. The full input id sequence is
   `prefix + candidate_ids` (length `128 + verify_k`).
3. Time **one full-sequence forward** over `prefix + candidate_ids`
   from a cold KV state, in the same shape a target would receive
   when verifying `verify_k` proposed tokens. The `verify_k=1`
   row is the single-token baseline; the marginal cost of
   multi-token verify at k=2/4/8 is recovered as
   `verify_k_marginal_ms = forward_ms[k] - forward_ms[k=1]`.

**Cache isolation policy.** Each timed forward starts from a
freshly built KV cache (`_release_mlx_state` + a clean
`prefill` of the prefix is acceptable) so successive k values
do not share warm state across runs and `forward_ms` is a
deterministic function of `verify_k` alone. The cost of the
prefix prefill is not folded into the timed call — only the
forward over the candidate slice (issued via the same low-level
forward used by `decode_step`) is timed. Implementation may
specialise this via `silica.mlx.forward.forward_batched_full`
or an equivalent multi-position call; whichever path the
adapter exposes at land time, the timed segment is the verify
forward only.

**JSONL fields per row:**

| Field                        | Meaning                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| `verify_k`                   | candidate tokens being verified in this forward                          |
| `forward_ms_p50`             | median wall time of one verify forward (N=10 reps after 2 warmups)       |
| `forward_ms_p95`             | p95 (variance signal)                                                    |
| `peak_memory_mb`             | peak memory during the verify forward                                    |
| `verify_k_marginal_ms`       | `forward_ms_p50[k] - forward_ms_p50[k=1]` — derived at write             |
| `kv_bytes_read_estimate`     | derived from `seqlen × per-layer KV bytes` (analytic, not measured)      |
| `weight_bytes_read_estimate` | derived from model weight footprint (analytic)                           |
| `prefix_token_count`         | actual tokenised prefix length (drift signal)                            |
| `seed`                       | sampler-irrelevant; recorded for reproducibility                         |

**Why `verify_k=1` is mandatory.** Without it, the verification
overhead at k=2/4/8 cannot be attributed cleanly between (a)
target single-token cost (which exists with or without
speculative decoding) and (b) the marginal cost of multi-token
verify. The C.4 / C.5 ROI estimate is
`(drafter_accept_rate × verify_k) / (verify_k_marginal +
drafter_overhead)`; without `verify_k_marginal_ms`, the
denominator is unmeasured and the speedup claim is
unfalsifiable. With it, the C.4 spike's reported speedup can be
decomposed into drafter quality vs verify amortisation and
pinned in the Decisions Log either way.

**Artefact:** `plans/P6_0_5_BASELINE/target_verify_microbench.{jsonl,md}`.

---

## 4. Acceptance — when is P-6.0.5 done

The phase exits when **all seven artefacts** under
`plans/P6_0_5_BASELINE/` are on disk and the directory contains a
`REPORT.md` mirroring the v1.7.13 P-6.0 baseline pattern (one
table summarising every row, plus per-sub-unit interpretation).

Each artefact must satisfy:

1. JSONL row(s) parse with the expected oracle fields populated
   (no NaN / null in load-bearing columns).
2. The corresponding `.md` file exists with a one-paragraph
   interpretation paragraph (what the number tells us about
   Decision Gate 1).
3. The §1 question this artefact answers is named in the
   `.md` interpretation paragraph; this is the audit trail for
   D-021 step 4.

Opt-in artefacts 2 (`qwen3.5-27b-warm-decode-b4`) and 4
(`qwen3.5-moe-35b-a3b-warm-decode-b4`) are allowed to record
`oom=true`; in those cases the JSONL row stands as
"attempted, OOM" rather than missing data, and the §6 gate
falls back to B=2 / B=3 (per §3.2 and §3.4).

The Decision Gate 1 (D-021 step 4) writeup happens **after**
this phase exit. P-6.0.5 produces the inputs only.

---

## 5. Artefact path layout

```
plans/P6_0_5_BASELINE/
  README.md                                          # one-line index
  REPORT.md                                          # cross-row table + per-row interpretation
  qwen3.5-27b-warm-decode-b2.jsonl
  qwen3.5-27b-warm-decode-b2.md
  qwen3.5-27b-warm-decode-b4.jsonl
  qwen3.5-27b-warm-decode-b4.md
  qwen3.5-moe-35b-a3b-warm-decode-b3.jsonl
  qwen3.5-moe-35b-a3b-warm-decode-b3.md
  qwen3.5-moe-35b-a3b-warm-decode-b4.jsonl
  qwen3.5-moe-35b-a3b-warm-decode-b4.md
  qwen3.5-moe-35b-a3b-warm-decode-b1-4k.jsonl
  qwen3.5-moe-35b-a3b-warm-decode-b1-4k.md
  qwen3.5-27b-warm-ttft-pair.jsonl
  qwen3.5-27b-warm-ttft-pair.md
  qwen3.5-moe-35b-a3b-warm-ttft-pair.jsonl
  qwen3.5-moe-35b-a3b-warm-ttft-pair.md
  target_verify_microbench.jsonl
  target_verify_microbench.md
  logs/                                              # raw run stdout/stderr per run
```

Mirror-of: `plans/P6_0_BASELINE/`.

---

## 6. Code touchpoints

| File                                                        | Change                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `silica/bench/scenarios.py`                                 | Register 6 new `Scenario`s (sub-units 1, 2, 3, 5 + the two pair scenarios from sub-unit 6)    |
| `silica/bench/oracles.py`                                   | Add `OracleKind.WARM_TTFT_PAIR`                                                               |
| `silica/bench/runner.py`                                    | Wire `WARM_TTFT_PAIR` two-prompt loop                                                         |
| `silica/bench/microbench/target_verify.py` (new)            | Microbench harness for sub-unit 7 (or `scripts/microbench_target_verify.py` — pick at land)   |
| `tests/test_bench_warm_ttft_pair_oracle.py` (new)           | Test surface for `WARM_TTFT_PAIR` (no real-model dep; mirrors `test_bench_prefix_hit_decode_oracle.py`) |
| `tests/test_bench_scenarios_catalog.py`                     | Append cases that the 6 new scenarios appear under `--list`                                   |

Sub-unit 4 (MoE B=4) has **no code** in this column — its
scenario already exists at `silica/bench/scenarios.py:2032` from
v1.7.13; only the data row under `plans/P6_0_5_BASELINE/`
lands in P-6.0.5. Sub-unit 7 (microbench) likewise registers no
`Scenario`; its harness is independent.

No engine, scheduler, KV, codec, or model code changes. No P-5
oracle changes.

---

## 7. Run sequence (on real hardware)

The order is set by OOM risk and by which artefacts feed which
acceptance question.

1. **Unit 1 — dense 27B B=2.** Lowest OOM risk, highest gate
   value. Run first.
2. **Unit 3 — MoE 35B-A3B B=3.** Independent of dense; B=2 was
   already validated at v1.7.13.
3. **Unit 5 — MoE 35B-A3B B=1 4K.** Same checkpoint as unit 3;
   tokeniser + workload are the only deltas.
4. **Unit 6 — warm-TTFT pair (both families).** Cheap (two
   short prompts), independent of B>1 risk.
5. **Unit 7 — target-verify microbench.** Independent harness;
   can run any time after the model is downloaded.
6. **Unit 2 — dense 27B B=4 (opt-in).** Run **only after** unit 1
   succeeds and a fresh Mac boot is available. Set
   `mx.metal.set_memory_limit(~42_000_000_000)` so the run fails
   fast on OOM rather than letting macOS swap.
7. **Unit 4 — MoE 35B-A3B B=4 (opt-in).** Run **only after**
   units 3 and 5 succeed and a second fresh Mac boot is
   available. Same `mx.metal.set_memory_limit` guard as unit 2.

If unit 2 or unit 4 OOMs, record `oom=true` in the JSONL and
stop — do not retry with smaller settings; the OOM is itself
the answer.

---

## 8. Open questions

- **OQ-1 — Microbench harness placement.** Settle
  `silica/bench/microbench/target_verify.py` vs
  `scripts/microbench_target_verify.py` at sub-unit 7 implementation
  time. The decision criterion is whether
  `python -m scripts.bench --microbench target-verify` (catalogue
  integration) is desired; if yes → `silica/bench/microbench/`,
  if not → `scripts/`.
- **OQ-2 — `WARM_TTFT_PAIR` prompt 2 distinct-from-prompt-1
  guarantee.** The two-prompt distinct-prompt anchor must keep
  `prefix_hit_tokens=0` on the gate row. Sub-unit 6 documents the
  guard, but the runner needs an explicit assertion (or the
  oracle silently records `prefix_hit_tokens > 0` and the
  diagnostic surface absorbs it). Pick at land.
- **OQ-3 — `WARM_TTFT_PAIR` and chunked-prefill interaction.**
  D.1 chunked-prefill is post-Gate; if any P-6.0.5 measurement
  inadvertently hits a chunked path, surface the chunking flag
  in the JSONL so the row is interpretable when D.1 lands.
- **OQ-4 — Verify-k microbench under MoE.** Sub-unit 7 anchors
  on dense 27B because that is where (1b) lives. If MoE Track A
  amplifier becomes load-bearing post-Gate, a MoE counterpart
  microbench may be needed; not in P-6.0.5.

None of OQ-1..4 block the opening; OQ-1 / OQ-2 are decided at
implementation time.

---

## 9. Cross-references

- **PLAN.md §7 D-021 step 3** — the canonical P-6.0.5 scope
  statement (lines 1451-1458 in the v1.7.16 tree).
- **PLAN.md §7 D-021 step 4** — Decision Gate 1; consumes the seven
  artefacts above.
- **`plans/P6_OPENING.md` §2** — original P-6.0 measurement gate;
  P-6.0.5 mirrors its data-on-disk acceptance pattern.
- **`plans/P6_0_BASELINE/REPORT.md`** — v1.7.13 baseline numbers
  P-6.0.5 reads against (dense 27B 16.05 tok/s @ 70.6%; MoE
  120.93 tok/s aggregate @ 59.1%).
- **`silica/bench/scenarios.py:2098-2196`** — P5.9 step 2(d)
  extended-context registrations (`-b1-4k`, `-b1-8k`); sub-unit 5
  reuses the same `_warm_decode_workload_extended` helper.
- **`silica/bench/scenarios.py:2031-2052`** — `_QWEN3_5_MOE_WARM_DECODE_B4`;
  sub-unit 2 description text mirrors this OOM-flagged shape.
- **`docs/plans-index.md`** — P-6 section; add a P-6.0.5 row at
  doc-sync time (not in this opening).

---

## 10. Sub-unit landing order

Following the same incremental-execution pattern used for
CHAT-CLI-RESPONSE-POLICY: each sub-unit lands as a separate
commit, the user reviews before the next sub-unit starts.

| Order | Unit                                            | Approx. effort         |
| ----- | ----------------------------------------------- | ---------------------- |
| 1     | This opening doc — current commit               | doc only               |
| 2     | Sub-unit 1 (dense 27B B=2) — scenario register  | small, ~30 lines code  |
| 3     | Sub-unit 3 (MoE B=3) — scenario register        | small, ~30 lines code  |
| 4     | Sub-unit 5 (MoE B=1 4K) — scenario register     | small, ~30 lines code  |
| 5     | Sub-unit 6 (`WARM_TTFT_PAIR` oracle + 2 scens)  | medium, oracle + tests |
| 6     | Sub-unit 7 (target-verify microbench)           | medium, new harness    |
| 7     | Sub-unit 2 (dense 27B B=4 opt-in) — register    | small; user runs       |
| 8     | All seven on-device runs (user-driven; sub-unit 4 MoE B=4 captures data only — no register step) | ~45-90 min wall time   |
| 9     | `plans/P6_0_5_BASELINE/REPORT.md` writeup       | doc only               |
| 10    | Phase-exit commit + PLAN.md status sync         | doc only               |

Steps 2-7 are pure code (scenario registration + new oracle +
microbench harness + tests); they can land before any real-model
run. Sub-unit 4 (MoE B=4) carries no code-landing step — the
scenario already exists at `silica/bench/scenarios.py:2032`, so
it appears only inside step 8 as a data row. Step 9 is the
cross-row interpretation that D-021 step 4 reads.

---

## 11. Non-goals (recapped, for the record)

- No engine / scheduler / KV / codec / model changes.
- No new family additions.
- No quality / PPL oracle additions.
- No Decision Gate 1 writeup (that is D-021 step 4, post-exit).
- No Track A / B / C / D / E work; those are gated on the
  Decision Gate 1 outcome.
