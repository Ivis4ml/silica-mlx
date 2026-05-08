# P-6 Track B 3-bit Weights — D-021 Step 7 Opening

| Field         | Value                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------- |
| Phase         | P-6 (Performance Phase) — D-021 step 7                                                                  |
| Status        | orientation drafted; no code on disk; awaits user review before B.1 begins                              |
| Last updated  | 2026-05-01                                                                                              |
| Scope owner   | xxzhou                                                                                                |
| Predecessors  | D-021 step 6 closed at v1.7.20 with C.4 retired (gate FAILED at 0.482×; see `plans/P6_C4_DFLASH/REPORT.md` (η.1)) |
| Successors    | D-021 step 8 (C.5 tree-shape spike — **NOT auto-queued; product decision pending after Track B lands**) |

This document opens **D-021 step 7 — Track B 3-bit weights**. With C.4
retired at v1.7.20, Track B is the next mainline lever in the
foundation-first execution order. Unlike the speculative tracks,
Track B is a bytes-per-step reduction: 4-bit → 3-bit shrinks
weight reads by ~25%, which on dense 27B at the bandwidth ceiling
translates directly into proportional decode tok/s.

The phase is bounded: mlx-lm already ships 3-bit kernel support
(`mlx_lm.convert -q --bits 3` for offline conversion;
`_mlx_lm_load` accepts 3-bit checkpoints unchanged at runtime).
Silica's job is to (a) verify a real 3-bit Qwen3.5-27B checkpoint
loads and runs cleanly, (b) cross-check quality via WikiText-2 PPL
against the 4-bit baseline, and (c) measure the warm-decode tok/s
lift on the b1-cousin shape. No silica/* runtime kernel work; the
heavy lifting is mlx-lm's.

---

## 0. TL;DR

Track B converts the dense 27B target from 4-bit to 3-bit weights
to lift its bandwidth ceiling. PLAN.md §13 step 7 verbatim:

> Track B 3-bit weights — loader + PPL oracle first (no runtime
> change), pass quality gate, then 27B 3-bit warm-decode. If 3-bit
> lifts dense from 16 → 21-24 tok/s, stack with spec; if quality
> or MLX path is unstable, ship opt-in.

Three sub-units (B.1 / B.2 / B.3) measured independently. Two
acceptance gates:

- **Quality (B.2)**: `ΔPPL ≤ 0.5 absolute AND ≤ 5% relative` on
  WikiText-2 chunked-NLL — **both must pass**. Failing either
  drops 3-bit to "available with caveats / opt-in"; quality gate
  is not declared passed. The two thresholds are not redundant:
  the absolute bound catches drift on smaller PPL values where
  5% relative is laxer; the relative bound catches drift on
  larger PPL values where 0.5 absolute is laxer.
- **Performance (B.3)**: `≥21 tok/s` on `qwen3.5-27b-warm-decode-b1-3bit`
  = ≥1.31× over the v1.7.13 b1 anchor (16.05 tok/s). PLAN's 21-24
  band is a prediction; the gate is the lower bound.

If both pass, 3-bit lands as a first-class quantization tier in
the bench catalog and the (1a) ≥40 tok/s primary gate (already
cleared at 42.17 tok/s in P-6.0.5) gets an additional lever. If
quality passes but performance falls short, 3-bit ships **opt-in**
(no default flip) for users who care about RAM headroom over
throughput. If quality fails, 3-bit is documented and shelved.

---

## 1. Motivation

The v1.7.13 P-6.0 baseline pinned dense 27B-4bit at 16.0 tok/s
(`qwen3.5-27b-warm-decode-b1`), which P-6.0.5 Unit 7 confirmed is
bandwidth-bound on the corrected 15.34 GB weight footprint:
`bandwidth_util = 16.0 × 15.34 GB / 307 GB/s ≈ 80%`. The
unified-memory ceiling at 100% utilisation is ~22.7 tok/s on
4-bit; reaching 60+ tok/s requires either spec (which C.4 just
failed at) or shrinking bytes/step.

**3-bit math.** A 3-bit linear quantization with the same group
size as the 4-bit checkpoint reduces weight bytes by `3/4 = 0.75×`.
At the same bandwidth utilisation, decode tok/s scales by `1/0.75
= 1.33×`. Applied to the 16.0 tok/s anchor: predicted 21.3 tok/s
floor at the same utilisation. The PLAN's 21-24 tok/s band brackets
this prediction (with `+0..+12%` headroom for slightly higher
utilisation if the 3-bit path also reduces metadata/scratch).

**Why this is mainline.** Track B is independent of every
speculative track — it's a target-side bytes/step reduction that
**multiplies** with any later spec lever. If silica eventually
opens a working spec path (full-DFlash port, retrained drafter,
C.5, C.6), the 3-bit target multiplies through. So even though
1.31× alone does not clear (1b) ≥60 tok/s, it widens the headroom
every other lever has to work with.

**Why this is achievable.** mlx-lm already supports 3-bit weights
at the kernel level (`mlx.fast.quantized_matmul` with
`bits=3`). `unsloth/Qwen3.6-27B-UD-MLX-3bit` is a published
production-grade precedent. Silica's loader path (`_mlx_lm_load`)
takes any quantization tier mlx-lm understands. The risk is
**quality** (does Qwen3.5-27B-3bit's PPL hold up?), not
**runtime** (the kernels exist).

---

## 2. Scope and out-of-scope

### 2.1 In scope

- **Use the matched-family native 3-bit Qwen3.5-27B fixture
  identified at B.1 read-only HF lookup (OQ-1 closure).**
  Primary: `NexVeridian/Qwen3.5-27B-3bit` (≈11 GB on disk; MLX
  safetensors; created 2026-02-25 from `Qwen/Qwen3.5-27B` via
  `mlx_lm.convert -q --bits 3`). Backup:
  `RepublicOfKorokke/Qwen3.5-27B-mlx-lm-3bit` (≈11 GB; same
  base; only used if NexVeridian's checkpoint exhibits a loader
  / layout issue). The 52 GB full-precision pull + offline
  convert path is **retired** — no in-house conversion needed.
- **B.1 loader smoke.** Load the 3-bit checkpoint via
  `silica.models.factory.adapter_for_repo`; run
  `Engine.generate("Hello", max_tokens=4)`; assert no exceptions,
  output is non-empty, peak resident memory `≤ 13 GB OR ≥ 20%
  reduction vs the 4-bit's 15.34 GB anchor` (whichever is looser
  — the relative-reduction form avoids brittle absolute thresholds
  if scale/zero metadata or activation scratch don't shrink
  linearly with weight bits).
- **B.2 quality cross-check.** Register **two** `OracleKind.PPL`
  scenarios — `qwen3.5-27b-wikitext-ppl-4bit` and
  `qwen3.5-27b-wikitext-ppl-3bit` — each carrying its own
  `Scenario.repo` (the 4-bit and 3-bit checkpoints respectively)
  and matching the existing 0.6B PPL fixture's chunked-NLL config
  scaled to 27B's chunk size. Each row is gated independently
  (4-bit row on `SILICA_REAL_QWEN3_5_27B`; 3-bit row on
  `SILICA_REAL_QWEN3_5_27B_3BIT`). The runner is **not**
  extended to consume two `repo`s in one row — that would
  reshape `Scenario.repo` semantics for one consumer. Instead
  the (B.2) REPORT.md section reads both rows' `ScenarioResult`
  outputs from the JSONL and computes ΔPPL against the gate
  thresholds. This keeps the harness change small and the
  `Scenario.repo` invariant ("one repo per scenario row") intact.
- **B.3 27B 3-bit warm-decode.** Add bench scenario
  `qwen3.5-27b-warm-decode-b1-3bit` mirroring `qwen3.5-27b-warm-decode-b1`
  shape exactly (B=1, 128-token prompt, 384-token generation,
  max_tokens=384). Run it under the existing scenario-cache + env
  gate pattern; record warm `decode_tok_s` and peak memory.
  Speedup against the 4-bit b1 anchor is the headline number.

### 2.2 Out of scope (deferred to later sub-units or v0.2)

- **Default-flip to 3-bit.** Even if both gates pass, 3-bit lands
  as an additional bench scenario, not as silica's recommended
  quantization tier. Promotion would require multi-workload
  attestation (chat, code, math) and is a separate decision.
- **3-bit MoE.** `Qwen3.5-35B-A3B-3bit` is not in scope for step 7
  — MoE acceptance (2a) is already cleared at the 188.5 tok/s
  baseline, and (2b) ≥175 tok/s aggregate at B≥3 is also unaffected
  by the 3-bit decision since MoE is bandwidth-bound on the
  active 3.5B/expert path, not on the full weight set. Add to
  backlog if the dense Track B clears both gates and MoE
  bandwidth becomes the next mainline question.
- **3-bit + spec stacking.** Composing 3-bit weights with the
  retired C.4 path is out of scope: C.4's 0.48× failure was
  drafter-cost-domination (35.7 ms drafter vs 2.45 ms verify);
  shrinking the verify cost further widens that gap, not narrows
  it. If a future spec path lands (full-DFlash port, retrained
  drafter, C.5), Track B lifts uniformly — no special composition
  work.
- **Custom 3-bit kernel work.** mlx-lm's `quantized_matmul` with
  `bits=3` is the runtime path. If it underperforms the 22.7
  tok/s prediction by more than 10%, that is an mlx-lm-side issue
  and silica documents the gap rather than ports a custom kernel.
- **3-bit on smaller fixtures (4B / 9B).** Not load-bearing for
  the (1a) gate decision; can be added later if the 27B row's
  numbers warrant a sweep.

### 2.3 Explicitly preserved from prior phases

- **D-009 native-runtime constraint.** mlx-lm's 3-bit kernels run
  through stock MLX; no torch in the runtime path.
- **Bench shape parity with b1 cousin.** The B.3 scenario mirrors
  `qwen3.5-27b-warm-decode-b1` exactly so the speedup ratio is
  comparable. Same as the (h) `*-spec-on` and (ζ) `*-c4-dflash`
  rows did against b1.
- **Quality-gate-before-runtime-promotion.** B.2 must pass before
  B.3 reports a "1.31× win" — measuring tok/s on a model that
  produces garbage is not a meaningful win.

---

## 3. Sub-unit decomposition (preview)

Three sub-units, each landing as one commit, with a user pause
between each per the standing incremental-execution rule.

1. **(B.1) 3-bit loader smoke + bench scenario registration.**
   - Acquire `NexVeridian/Qwen3.5-27B-3bit` (≈11 GB MLX
     safetensors; OQ-1 closed at the read-only HF lookup that
     preceded this sub-unit). Gate env var:
     `SILICA_REAL_QWEN3_5_27B_3BIT`.
   - Verify `silica.models.factory.adapter_for_repo` loads it
     cleanly; spot-check `Engine.generate("Hello", max_tokens=4)`.
     No silica code change expected unless the loader chokes on
     a 3-bit checkpoint detail (group size, scale dtype).
   - Register `qwen3.5-27b-warm-decode-b1-3bit` scenario in
     `silica/bench/scenarios.py`, mirroring `b1` shape.
   - Tests pin three invariants on the new scenario:
     - **Workload shape parity** with `qwen3.5-27b-warm-decode-b1`:
       same `prompts`, `max_tokens`, `max_batch_size`, `oracle`.
     - **Repo differs**: 3-bit row's `repo` is the 3-bit
       checkpoint id, **not** the 4-bit cousin's repo.
     - **Gate differs**: 3-bit row's `gate_env_var` is
       `SILICA_REAL_QWEN3_5_27B_3BIT`, **not** the 4-bit cousin's
       `SILICA_REAL_QWEN3_5_27B`. The two checkpoints are gated
       independently so a user with only the 4-bit cached cannot
       trigger an unintended 12 GB load.
     Plus: `--list` enumerates the new scenario; catalog count
     rises from 67 → 68.
2. **(B.2) WikiText-2 PPL cross-check.**
   - Register two `OracleKind.PPL` scenarios — one per repo:
     - `qwen3.5-27b-wikitext-ppl-4bit` (repo
       `mlx-community/Qwen3.5-27B-4bit`, gate
       `SILICA_REAL_QWEN3_5_27B`).
     - `qwen3.5-27b-wikitext-ppl-3bit` (repo from B.1, gate
       `SILICA_REAL_QWEN3_5_27B_3BIT`).
     Both reuse the existing 0.6B PPL fixture's chunked-NLL
     oracle config scaled to 27B's chunk size; **no runner
     change**, no new oracle code.
   - Run each under its own gate env. Each emits a standard
     `ScenarioResult` row with `ppl` in metadata.
   - Append (B.2) section to `plans/P6_TRACK_B/REPORT.md`,
     reading both rows from the bench JSONL and computing
     `ΔPPL_abs = ppl_3bit − ppl_4bit` + `ΔPPL_rel = ΔPPL_abs /
     ppl_4bit`. Gate evaluation is REPORT-side, not runner-side.
   - Tests: scenario shape (each row has its own correct repo +
     gate); oracle config matches existing 0.6B PPL rows;
     `--list` enumerates both rows. The actual PPL run is
     environment-affecting (loads two 27B checkpoints back-to-back
     across the two scenarios) and is the sub-unit's
     load-bearing deliverable.
3. **(B.3) 27B 3-bit warm-decode attestation.**
   - Run `qwen3.5-27b-warm-decode-b1-3bit` under the four-gate-
     active conditions: target HF cache + 3-bit env + WikiText-2
     not required (warm-decode oracle, not PPL).
   - Append (B.3) section to `plans/P6_TRACK_B/REPORT.md` with
     warm `decode_tok_s`, peak memory, ratio against the v1.7.13
     b1 anchor (16.05 tok/s).
   - Make the §13 step 7 gate call: ≥21 tok/s = 1.31× clears the
     "lifts dense from 16 → 21-24 tok/s" prediction's lower
     bound; <21 tok/s but >16 tok/s ships opt-in; ≤16 tok/s
     means the 3-bit kernel is slower than 4-bit (would be
     unexpected — likely an mlx-lm-side issue).
   - Update PLAN.md §13 with the v1.7.21 changelog entry.

(B.1) and (B.2) are pure code + measurement; (B.3) is the
load-bearing decision row. (B.1) lands without env-affecting
runs in CI (the smoke test is a 4-token generate; the b1-3bit
bench row is registered but not executed). (B.2) and (B.3) each
require explicit user go-ahead before the env-affecting bench
runs (mirroring the (η.1) approval pattern).

---

## 4. Architecture / interface contracts

### 4.1 No silica/* runtime change expected

mlx-lm's 3-bit kernels are activated by the checkpoint's quant
config; `_mlx_lm_load` reads the config and dispatches the right
`quantized_matmul` variant. silica's `Qwen3_5Adapter` consumes
the loaded model unchanged — `decode_step`, `decode_step_multi`,
`prefill`, `prefill_with_capture` (the αβ.3 capture surface) all
route through `forward_full` / `forward_batched_full`, which call
`model(tokens, cache=cache_list)` — the model handles its own
quantization internally.

If a load-time issue surfaces (group-size mismatch, scale dtype
issue), it is mlx-lm's bug, not silica's. The B.1 sub-unit
records the issue + minimal-repro and either upstreams a fix or
documents the workaround; silica does **not** patch around an
mlx-lm bug in its own runtime path.

### 4.2 PPL oracle reuse

The existing `OracleKind.PPL` oracle (used by the 0.6B
`qwen3-0.6b-wikitext-ppl-*` rows) is the load-bearing PPL
machinery. B.2's new scenario reuses it with a 27B target — no
new oracle code. The chunked-NLL config (chunk size 64, sequence
length 128) carries over from the 0.6B rows; tuning to 27B's
context is unnecessary because the comparison is between two
checkpoints, not between an absolute PPL and a published number.

### 4.3 Bench scenario shape

`qwen3.5-27b-warm-decode-b1-3bit` differs from `qwen3.5-27b-warm-decode-b1`
only in the `repo` field (the 3-bit checkpoint id) and the
`gate_env_var` (`SILICA_REAL_QWEN3_5_27B_3BIT` instead of
`SILICA_REAL_QWEN3_5_27B`). Workload, oracle, max_tokens,
max_batch_size all match exactly so the speedup ratio is read
without normalisation.

---

## 5. Open questions

### 5.1 OQ-1 — 3-bit Qwen3.5-27B checkpoint availability (closed favourably at B.1 read-only HF lookup, 2026-05-01)

**Closed favourably.** The B.1 read-only HF lookup found two
matched-family native MLX 3-bit Qwen3.5-27B checkpoints, both
created by running `mlx_lm.convert -q --bits 3` against
`Qwen/Qwen3.5-27B`:

| Repo | Created | Downloads | mlx-lm version | Storage |
| ---- | ------- | --------- | -------------- | ------- |
| **`NexVeridian/Qwen3.5-27B-3bit`** (primary) | 2026-02-25 (1 day after Qwen3.5-27B release) | 309 | 0.30.8 | ≈11.0 GB |
| `RepublicOfKorokke/Qwen3.5-27B-mlx-lm-3bit` (backup) | 2026-03-08 | 230 | 0.30.7 | ≈11.0 GB |

Both have identical layout: 3 safetensor shards + `config.json`
+ `model.safetensors.index.json` + tokenizer files;
`quantization_config.bits = 3`; `model_type = "qwen3_5"`. Total
storage parameters 26.9B (= 843 M BF16 + 2.52 B U32-packed
quantized + a handful of F32 RMSNorm scales). The two
checkpoints differ by ≈12 KB in `used_storage` and a few
thousand F32 params (likely RMSNorm vs LayerNorm metadata) —
not load-bearing.

**Decision.** B.1 uses **`NexVeridian/Qwen3.5-27B-3bit`** as the
primary fixture (earliest, most-downloaded, mlx-lm 0.30.8).
`RepublicOfKorokke/Qwen3.5-27B-mlx-lm-3bit` is reserved as a
backup for layout / loader issues only; not pulled at B.1 time.

**52 GB convert fallback retired.** The orientation's earlier
"if no native 3-bit, run `mlx_lm.convert` against the 52 GB
full-precision Qwen3.5-27B" plan is no longer needed. OQ-5
(absolute fp16 PPL anchor) remains documented as a known caveat
for the relative-only B.2 gate, but does not block step 7.

### 5.2 OQ-2 — PPL gate threshold

**Question.** PLAN.md §13 step 7 says "pass quality gate" without
naming a number. The §6.2 acceptance gate above pins
`ΔPPL ≤ 0.5 absolute AND ≤ 5% relative` — both must pass. Is
that the right pair?

**Resolution method.** B.2 records both the absolute ppl values
and the deltas; the gate evaluation reads the OPENING's threshold
unchanged. If the measured delta is in the band (0.3-0.5
absolute or 3-5% relative), B.2's REPORT entry calls out the
borderline outcome and the gate decision uses the tighter
threshold. The 0.5 / 5% pair is conservative; a stricter ≤ 0.3
absolute gate is reachable for many quantization sweeps and
worth checking against once we have a number.

### 5.3 OQ-3 — peak RAM gate form

**Question.** The B.1 acceptance gate is "≤ 13 GB **OR** ≥ 20%
reduction vs 15.34 GB anchor". The looser form picks up at ~12.3
GB minimum. Is that the right pass criterion?

**Resolution method.** B.1's measurement reports both forms; the
gate evaluation picks whichever passes (ditto OPENING phrasing).
Edge case: if the 3-bit weights fit in 11.5 GB (the naive 25%
reduction), embeddings + lm_head + scale/zero metadata + scratch
push the total above 13 GB, the relative-reduction form (15.34 ×
0.8 = 12.27 GB) catches the win without over-tightening on the
absolute number.

### 5.4 OQ-4 — does 3-bit perform proportionally on M5 Pro?

**Question.** The 22.7 → 30.3 tok/s prediction assumes the
3-bit `quantized_matmul` kernel saturates the same fraction of
the M5 Pro's 307 GB/s memory bandwidth that the 4-bit kernel
does (≈80% per Unit 7). Apple Silicon's quantization-kernel
performance scales differently across bit-widths — historical
mlx-lm benches have shown 3-bit sometimes lagging 4-bit on
absolute throughput on smaller models due to dispatch overhead
not amortising over the smaller weight read. If 27B is large
enough to amortise, 21-24 tok/s holds; if not, the win
may be smaller.

**Resolution method.** B.3 measures it directly. If the warm
`decode_tok_s` is `≥ 21` tok/s, 3-bit clears the gate and the
v1.7.18 step 7 expectation holds. If it's `< 21` but `> 16`,
3-bit ships opt-in. If it's `< 16`, mlx-lm-side issue — escalate
upstream.

### 5.5 OQ-5 — does the 3-bit checkpoint's WikiText-2 PPL track the published Qwen3.5-27B fp16 PPL?

**Question.** The PPL gate compares 3-bit against 4-bit silica
runs of the same fixture; both anchor to a relative number, not
to the absolute Qwen3.5-27B fp16 PPL. If the 4-bit baseline is
itself off by 0.5+ PPL from fp16 (which would be unusual but
possible), the relative gate fails to detect a 3-bit-only
quality drop.

**Resolution method.** Out of scope for B.2's relative-only
gate. Documented in the (B.2) REPORT as a known caveat. An fp16
anchor cross-check would require pulling the 52 GB
`Qwen/Qwen3.5-27B` full-precision checkpoint, which exceeds 48
GB unified memory at residency on M5 Pro and is out of scope
for step 7 — relative drift between the two cached MLX
quantizations is the load-bearing comparison.

---

## 6. Acceptance gates

### 6.1 B.1 — loader smoke

- 3-bit Qwen3.5-27B checkpoint loads via
  `silica.models.factory.adapter_for_repo` with no exceptions.
- `Engine.generate("Hello", max_tokens=4)` produces 4 non-empty
  tokens.
- Peak resident memory **≤ 13 GB OR ≥ 20% reduction vs the
  v1.7.14 P-6.0 4-bit anchor (15.34 GB)** — whichever passes.

### 6.2 B.2 — PPL quality cross-check

- Both `qwen3.5-27b-wikitext-ppl-4bit` and
  `qwen3.5-27b-wikitext-ppl-3bit` run to completion under the
  WikiText-2 fixture, each under its own gate env. The (B.2)
  REPORT.md section reads both `ScenarioResult.metadata.ppl`
  values from the bench JSONL.
- Quality gate: `ΔPPL_abs = ppl_3bit − ppl_4bit ≤ 0.5`
  **AND** `ΔPPL_rel = ΔPPL_abs / ppl_4bit ≤ 0.05` (5%).
  **Pass requires both.** Failing either drops 3-bit to
  "available with caveats / opt-in" — consumers can use the
  explicit scenario id but the gate is **not** declared
  passed and runtime promotion to default 3-bit is **blocked**.

### 6.3 B.3 — 27B 3-bit warm-decode performance

- `qwen3.5-27b-warm-decode-b1-3bit` runs to completion (no
  warmup-stability failure of the kind that flagged the (η.1)
  C.4 row — 3-bit decode is bandwidth-bound, no rollback
  variance to worry about).
- Warm `decode_tok_s ≥ 21` (= 1.31× over 16.05 anchor) → **3-bit
  ships as a first-class scenario.**
- `16 < decode_tok_s < 21` → 3-bit ships **opt-in** (no default
  flip, no PLAN claim of 1.31×).
- `decode_tok_s ≤ 16` → mlx-lm-side issue; escalate upstream
  before considering a runtime promotion.

### 6.4 Toolchain attestation

- ruff clean (silica + tests + scripts);
- mypy clean (no new errors over the v1.7.20 baseline);
- full non-real-model test suite passes (target: ≥2630 — the
  v1.7.20 baseline; B.1 adds at minimum scenario-registration
  tests, B.2 adds oracle-row tests);
- `python -m scripts.bench --list` enumerates **at least the
  v1.7.20 catalog (67 scenarios) plus the new B.1 + B.2 rows**
  — i.e. ≥69 scenarios expected at step 7 close.

---

## 7. PLAN.md change list (preview; lands at step 7 close)

- **§7 P-6 step 7 status row** — flip from "in progress" to
  "closed at v1.7.21" with the measured B.2 ΔPPL + B.3
  decode_tok_s and the gate-pass / -fail call.
- **§13 D-021 changelog** — new v1.7.21 entry summarising B.1 /
  B.2 / B.3 closures and the (1a) ≥40 tok/s primary lever
  status (whether 3-bit's lift to ~21 tok/s makes 27B 4-bit
  ↦ 27B 3-bit a recommended default vs an opt-in row).
- **§6 deliverable B** — flip B.1 / B.2 from `[ ]` to `[x]` with
  the commit refs.
- **§7 P-6 acceptance gate (1a)** — record the additional
  bandwidth headroom 3-bit provides; (1a) at 42.17 tok/s is
  unchanged but the gap-to-60 narrows.

---

## 8. Cross-references

- `plans/PLAN.md` §13 D-021 step 7 — the verbatim contract this
  spike satisfies.
- `plans/PLAN.md` §6 P-6 deliverable B — Track B taxonomy.
- `plans/P6_OPENING.md` §3 "Track B — 3-bit Weight Option" —
  prior orientation (B.1 / B.2 hypothesis, sub-unit sketch);
  this opening replaces and expands it.
- `plans/P6_C4_DFLASH/REPORT.md` (η.1) — the C.4 retirement that
  put Track B on the mainline path.
- `plans/P6_0_5_BASELINE/REPORT.md` Unit 7 — the bandwidth-
  ceiling math (22.7 tok/s on 4-bit; 30.3 tok/s on 3-bit at the
  same utilisation).
- `mlx_lm.convert` — upstream's offline conversion path
  (`-q --bits 3 --group-size 64 --mlx-path qwen3_5_27b_3bit
  Qwen/Qwen3.5-27B`).
- `unsloth/Qwen3.6-27B-UD-MLX-3bit` — published 3-bit precedent
  (different family, used as runtime smoke fallback if
  Qwen3.5-27B-3bit conversion is too costly).

---

## 9. Non-goals (for the avoidance of doubt)

- Track B is **not** the (1b) ≥60 tok/s settlement. The 1.31×
  prediction on top of 16 tok/s = ~21 tok/s; (1b)'s 60 tok/s
  needs another lever stacked on top. Track B's job is to widen
  the headroom, not close the gap alone.
- Track B is **not** a 3-bit kernel-engineering effort. mlx-lm's
  `quantized_matmul(bits=3)` is the runtime; silica plumbs the
  format through.
- Track B is **not** a multi-family quantization sweep. Only
  Qwen3.5-27B is in scope; MoE 35B-A3B / Gemma4 / Qwen3-0.6B
  3-bit decisions come later if the dense row clears both gates.
- Track B is **not** a recommendation to re-open C.4 with a
  3-bit drafter. C.4 retired at v1.7.20 because of accept-rate
  collapse against the 4-bit target, not because of drafter
  cost alone. Quantizing the drafter further would change the
  cost equation but not the accept-rate math; a separate
  proposal (drafter retraining or full-DFlash port) is required
  to revisit C.4.
- Track B does **not** require pulling MoE drafter weights or
  any DFlash assets beyond what (η.1) already cached.

---

*Spike opens with sub-unit (B.1) — checkpoint acquisition + loader
smoke + scenario registration — pending user review of this
document.*
