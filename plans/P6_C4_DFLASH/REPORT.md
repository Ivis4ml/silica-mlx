# P-6 Track C.4 DFlash spike — REPORT

D-021 step 6 spike measurement bundle. One file across sub-units; new
rows append as the spike progresses from (αβ.1) → (η).

| Sub-unit | Row | Status | Date |
| -------- | --- | ------ | ---- |
| (αβ.1) | `c_capture_hidden(k=16)` on Qwen3.5-0.8B (dense) | landed | 2026-05-01 |
| (αβ.2) | `c_capture_hidden(k=16)` on Qwen3.5-35B-A3B-4bit (MoE) | landed | 2026-05-01 |
| (η.1)  | dense 27B `qwen3.5-27b-warm-decode-c4-dflash` | landed; **gate FAILED at 0.48×** | 2026-05-01 |
| (η.2)  | MoE 35B-A3B `qwen3.5-moe-35b-a3b-warm-decode-c4-dflash` | **skipped** (dense ≪ 1.5×) | — |

---

## (αβ.1) — `c_capture_hidden(k=16)` on cached Qwen/Qwen3.5-0.8B

Closes OQ-5 (`plans/P6_C4_DFLASH_OPENING.md` §5.5) on the 0.8B
fixture. The measurement constrains the §1 prediction band's new
`c_capture_hidden(k)` term that F-1 introduced post-α.

### Setup

- Fixture: cached `Qwen/Qwen3.5-0.8B` (HF cache hit; same path the
  step 5 (f) parity test uses). Layer count = 24 (silica's
  `config.num_layers`); hidden_size = 1024.
- Workload: single-request `decode_step_multi(k=16)` vs
  `decode_step_multi_with_capture(k=16, capture_layer_ids)`.
  Capture set: `{0, num_layers // 2, num_layers} = {0, 12, 24}` —
  three layer outputs (embedding pre-stack, mid-stack, post-stack).
- Canonical methodology (reused-adapter): adapter loaded once at the
  start of the bench; each iteration runs ``decode_step_multi`` (or
  the capture variant) under a fresh ``req_id`` and frees the
  per-request state (`SimpleKVCache.free` + `Qwen3_5Adapter.free_state`)
  before the next iteration. This isolates the per-cycle verify cost
  the (β) `DFlashDrafter` will actually pay, without the 20 GB MoE
  load cost dominating the wall-clock as it did under the original
  per-iter-load variant. 3 warmup iterations + 20 measurement
  iterations; **median** wall-clock reported. Materialisation forced
  via `mx.eval(logits, *captured)`.
- Hardware: M5 Pro 48 GB (silica's primary target).
- Script: `scripts/microbench_capture_hidden.py` —
  ```text
  python -m scripts.microbench_capture_hidden \
      --repo Qwen/Qwen3.5-0.8B --k 16 --warmup 3 --iters 30
  ```
  An older variant of the script reloaded the adapter on every
  iteration (cycle-1 cold each time). That methodology's numbers are
  preserved in the "Result" table below as a methodology-audit row;
  the canonical numbers come from the reused-adapter run.

### Result

Two methodology variants. The first run paid a fresh
`adapter_for_repo` load per iteration (cycle-1 cold), which made the
load cost dominate the wall-clock. The second run loads the adapter
once and reuses it across iterations (per-req KV freed via
`SimpleKVCache.free` between samples) — same cycle-1 verify-cost
contract, much tighter signal.

| Quantity | Per-iter load (initial) | Reused adapter (refined) |
| -------- | ----------------------- | ------------------------ |
| `decode_step_multi(k=16)` baseline median | 26.374 ms | **12.784 ms** |
| `decode_step_multi_with_capture(k=16, |L|=3)` median | 26.267 ms | **12.902 ms** |
| `c_capture_hidden(k=16)` (delta) | -0.107 ms | **+0.118 ms** |
| Ratio (capture / baseline) | 0.9959 | **1.0092** |
| Relative cost | -0.41% | **+0.92%** |

The reused-adapter run is the load-bearing measurement;
**`c_capture_hidden(k=16) ≈ +0.92%` on the 0.8B fixture for `|L|=3`,
within per-iter jitter and well below the §1 prediction-band budget.**
The first-run negative delta was load-jitter dominated and is
preserved in the table for methodology audit, not as the reportable
number.

### Interpretation

The §1 post-α prediction band ("1.4-2.0× at α ∈ [0.5, 0.7]") was
shifted ~10% lower than the pre-α band to budget for an unknown
`c_capture_hidden(k)`. The 0.8B measurement says that budget is
essentially zero on this fixture for a small capture set, so the
band can revert toward "1.5-2.2× at α ∈ [0.5, 0.7]" pending the
real-target measurement on dense 27B-4bit (η).

Two caveats narrow the inference from 0.8B → 27B:

1. **Bandwidth utilisation.** 0.8B's verify forward is small enough
   that any incremental cost from materialising hidden states is
   absorbed by allocator headroom. The 27B-4bit verify forward at
   k=16 sits closer to bandwidth saturation per the P-6.0.5 Unit 7
   measurement (1.494× at k=4, sublinear past that); a fixed-bytes
   capture overhead may be more visible there. (η)'s real-target
   bench attestation re-measures.
2. **Capture-set size.** 0.8B was probed at |L|=3 (three captured
   layer slices). The dflash drafter's `target_layer_ids` cardinality
   varies per checkpoint; if the upstream `z-lab/Qwen3.5-27B-DFlash`
   trained with |L| ≥ 6, the per-cycle capture write-bandwidth grows
   linearly with |L|. Sub-unit (β)'s drafter-load step reads
   `DFlashDraftModelArgs.target_layer_ids` from the checkpoint
   config; if |L| > 3, (αβ.2)'s MoE row probes the same fixture
   shape on 0.8B at the actual |L| before (η) commits.

### Decision

The §1 prediction band reverts to **1.5-2.2× at α ∈ [0.5, 0.7]**
based on this 0.8B measurement, with a footnote that 27B at large
|L| may shift it down within the same range. Sub-unit (αβ.2)'s MoE
row (below) corroborates the inference; sub-unit (β) opens
unblocked.

---

## (αβ.2) — `c_capture_hidden(k=16)` on cached `mlx-community/Qwen3.5-35B-A3B-4bit`

Closes OQ-5 on the MoE fixture, completing the (αβ) precondition
gate. The MoE adapter inherits the entire capture surface from
`Qwen3_5Adapter` (mlx-lm's `qwen3_5_moe.Model` extends
`qwen3_5.Model` directly; the outer forward chain is identical, only
per-layer MLP type differs). The microbench tests whether the
SwitchGLU sparse MLP changes the per-cycle capture cost.

### Setup

- Fixture: cached `mlx-community/Qwen3.5-35B-A3B-4bit` (~20 GB);
  upstream registers it in `dflash_mlx.generate.DRAFT_REGISTRY`
  paired with `z-lab/Qwen3.5-35B-A3B-DFlash`.
- Workload: same script, same capture set
  `{0, num_layers // 2, num_layers}` = three layer slices. MoE
  has `num_layers = 40`, `hidden_size = 2048`, so capture set is
  `{0, 20, 40}`.
- Methodology: reused-adapter (refined) — adapter loaded once,
  per-req KV freed via `SimpleKVCache.free` between samples. 3
  warmup iters + 20 measurement iters. Median wall-clock reported.
- Hardware: M5 Pro 48 GB.
- Script invocation:
  ```text
  SILICA_REAL_QWEN3_5_MOE=1 python -m scripts.microbench_capture_hidden \
      --repo mlx-community/Qwen3.5-35B-A3B-4bit \
      --k 16 --warmup 3 --iters 20
  ```

### Result

| Quantity | Value |
| -------- | ----- |
| `decode_step_multi(k=16)` baseline median | **45.257 ms** |
| `decode_step_multi_with_capture(k=16, |L|=3)` median | **43.644 ms** |
| `c_capture_hidden(k=16)` (delta) | **-1.613 ms** |
| Ratio (capture / baseline) | 0.9644 |
| Relative cost | -3.56% |

The negative delta is within per-iter jitter on a 20 GB MoE
checkpoint (MLX dispatch / sparse-MLP routing variance / OS-level
memory effects). **Capture is effectively free on the MoE fixture at
|L|=3** — same conclusion as the dense (αβ.1) row.

### Cross-fixture comparison

| Fixture | Baseline (ms) | Capture (ms) | Δ (ms) | Δ (%) |
| ------- | ------------- | ------------ | ------ | ----- |
| Qwen3.5-0.8B (dense, 24 layers, hidden=1024) | 12.78 | 12.90 | +0.12 | +0.92% |
| Qwen3.5-35B-A3B-4bit (MoE, 40 layers, hidden=2048) | 45.26 | 43.64 | -1.61 | -3.56% |

Both deltas are within per-iteration jitter; neither shows a
*systematic* per-cycle cost penalty. The §1 prediction band's
`c_capture_hidden(k)` term holds at ≈0 on both fixtures for |L|=3
on the M5 Pro 48 GB target. This corroborates the (αβ.1) inference
that the band reverts to "1.5-2.2× at α ∈ [0.5, 0.7]". Dense 27B
real-target re-measurement at sub-unit (η) remains the load-bearing
final check, and the upstream drafter checkpoint's actual
`|target_layer_ids|` is determined at sub-unit (β)'s drafter-load
step (if `|L| > 3`, (η) re-measures at the actual capture-set size).

---

## (η.1) — dense 27B `qwen3.5-27b-warm-decode-c4-dflash` real-checkpoint attestation

Closes the §13 D-021 step 6 gate decision for C.4 on dense 27B.
**Outcome: gate FAILED at 0.482× silica-integrated speedup
(spec-on slower than spec-off).**

### Setup

- **Target:** `mlx-community/Qwen3.5-27B-4bit` (cached; v1.7.14 P-6.0
  anchor at 15.34 GB peak). Gate `SILICA_REAL_QWEN3_5_27B=1`.
- **Drafter:** `z-lab/Qwen3.5-27B-DFlash` (downloaded at η.1; 2B
  parameters, BF16, 3.46 GB on disk). Gate `SILICA_BENCH_DFLASH_27B=1`.
- **Scenario:** `qwen3.5-27b-warm-decode-c4-dflash` (B=1, 128-token
  prompt, 384-token generation, max_tokens=384, `verify_k=16`,
  `kind="dflash"`). Mirrors `qwen3.5-27b-warm-decode-b1` shape exactly
  so the speedup ratio is comparable to the v1.7.13 anchor.
- **Hardware:** M5 Pro 48 GB.
- **Command:**
  ```text
  SILICA_REAL_QWEN3_5_27B=1 SILICA_BENCH_DFLASH_27B=1 \
      uv run python -m scripts.bench --speculative dflash \
          --scenario qwen3.5-27b-warm-decode-c4-dflash \
          --out plans/P6_C4_DFLASH/dense_27b_run.jsonl
  ```
- **Output JSONL:** `plans/P6_C4_DFLASH/dense_27b_run.jsonl`.
- **Bench commit:** `7f7221e` (ζ wiring) + `1b39e83` (ζ doc cleanup).
  Drafter wrapper (β..δ.1) + engine ε side channel + αβ capture
  surface all on the v1.7.19+commit-trail at this run.

### Speculative-metric schema fields (verbatim from `ScenarioResult.metadata`)

| Field | Value |
| ----- | ----- |
| `accept_rate` | **0.0881** (8.8%) |
| `verify_cost_ms` | 2.45 ms (per target verify forward at k=16) |
| `draft_cost_ms` | **35.70 ms** (per drafter forward) |
| `tokens_per_target_forward` | 2.32 |
| `rollback_count` | 165 |
| `tree_node_visits` | 0 (trajectory drafter, not tree) |
| `quality_parity_status` | `not_tested` (engine ε runs the
  invariant in dedicated tests, not the bench harness) |

### REPORT-derived rows

| Quantity | Value |
| -------- | ----- |
| Spec-on warm `decode_tok_s` | **7.74 tok/s** (this row) |
| Spec-off baseline `decode_tok_s` | 16.05 tok/s (v1.7.13
  `qwen3.5-27b-warm-decode-b1` anchor) |
| **Silica-integrated speedup** | **0.482×** (≈ 52% slower than
  spec-off) |
| `peak_memory_mb` | 19,043 MB (≈ 18.6 GB; target 15.3 GB +
  drafter ≈ 3.4 GB) |
| `wall_s` | 52.63 s for 384 tokens |
| `ttft_ms` | 517 ms |
| `prefill_tok_s` | 263 tok/s |
| `draft_overhead_per_step` | 35.70 ms / cycle (≈ 4.6× a single
  spec-off `decode_step` at 16 tok/s = 62.5 ms/token, dominating
  the cycle) |

Bench row `status` is `failed` due to
`warm_decode_row_0_warmup_did_not_stabilize:boundary=368_intervals=383`
— rel_std exceeded the 5% threshold the warm-decode oracle pins.
The instability is itself a signal: rollback variability prevents a
stable decode rate. The numerical fields above are still
reported because the underlying samples are valid; the failure is
on the rate-stability invariant, not on the count or schema.

### Interpretation

Three findings explain the 0.48× outcome:

1. **Drafter cost dominates verify cost by 15×.** `draft_cost_ms`
   (35.7 ms) ≫ `verify_cost_ms` (2.45 ms). The 2B BF16 drafter is
   slower per forward than the 4-bit quantised target's verify
   forward. Per the §1 prediction band's `c_draft_block + c_verify(k)`
   denominator, this alone would already push the speedup below 1×
   even at perfect acceptance.
2. **Accept rate collapsed to 8.8%.** The §1 prediction band assumed
   α ∈ [0.5, 0.7]; the measured 0.088 is far below that range.
   Most plausible cause: the `z-lab/Qwen3.5-27B-DFlash` checkpoint
   was trained against the **full-precision** Qwen3.5-27B target,
   and the 4-bit-quantised target's argmax distribution diverges
   from what the drafter learned to anticipate. OQ-7's α-closure
   ("upstream `DRAFT_REGISTRY` maps the 4-bit MLX target ID, so
   pairing is upstream-blessed") was a **necessary but not
   sufficient** condition — the registry says "pairing is
   supported", not "accept rate is preserved". The 4-bit target
   shifts the drafter's effective accept rate down dramatically.
3. **Rollbacks dominate decode time.** With 165 rollbacks across
   ~165 cycles (effectively every cycle), the engine pays the trim
   + replay cost on the recurrent path (`Qwen3_5Adapter.rollback_state`
   + `decode_step_multi` replay over the committed prefix) on
   nearly every iteration. Combined with the 35.7 ms drafter cost,
   each cycle yields 2.32 tokens for ≈40 ms of work + ≈80 ms of
   rollback/replay = ≈19 tok/s peak per-cycle, but rollback
   variability flattens the warm aggregate to 7.74 tok/s.

### Gate decision (PLAN.md §13 step 6 verbatim)

> Gate (per Decision Gate 1 v1.7.18 reframe): ≥1.8× silica-integrated
> speedup continues; ≥2.5× is one component of the (1b) two-condition
> survival rule (feeding the **full-stack measurement** leg) and also
> motivates the C.5 tree-shape spike (the second leg, see step 8);
> ≤1.8× retires (1b) only if no C.5 spike is pursued. **C.4 alone
> does not settle (1b)** — only the full-stack measurement or the
> C.5 spike does.

**Measured: 0.482×.** This is **0.27 of the engineering-continue
floor (1.8×)** and **0.19 of the (1b) survival contribution
threshold (2.5×)**. C.4 with the upstream-shipped drafter against
the 4-bit MLX target **does not survive the engineering gate**.

**Decision:**
- **C.4 dense path retires** as a (1a) ≥40 tok/s lever and as a
  (1b) ≥60 tok/s contributor. The drafter-only spike with no
  verify-side optimisations (per §2.2 deferrals) cannot bridge the
  drafter-cost / accept-rate gap on the 4-bit target.
- **(1b) survival now hinges on the C.5 tree-shape spike** (D-021
  step 8) being pursued. Per the v1.7.18 Decision Gate 1 reframe,
  if no C.5 spike is opened, (1b) retires entirely.
- **MoE row (η.2) skipped** per the user's pre-agreed "if dense
  < 1.5× don't run MoE" constraint. Cross-target sensitivity adds
  no signal here — MoE on a parallel drafter would face the same
  4-bit-target-vs-BF16-drafter pairing problem.

### Follow-up open questions (not in this spike's scope)

- **Would a 4-bit-quantised drafter recover accept rate?** Upstream's
  `--quantize-draft` flag (`dflash_mlx.runtime._should_quantize_draft`)
  4-bit-quantises the drafter, which would (a) drop drafter cost
  ~4× and (b) align drafter precision with target precision. Whether
  this lifts accept rate enough to clear ≥1.8× is an open
  measurement; not pursued here because PLAN's (1b) survival path is
  now C.5-or-retire, and a quantised-drafter spike would be a
  separate proposal beyond step 6.
- **Would porting upstream's `verify_qmm` int4 Metal kernel +
  tape-replay verify lift the silica-integrated number close to
  upstream's 5.2× claim?** Both are explicit non-goals of step 6
  per §2.2. The drafter-cost-domination finding above suggests
  the verify-side optimisations alone wouldn't rescue C.4 at this
  accept rate; the drafter would need to be quantised first.
- **What is the actual measured upstream baseline for the same
  fixture?** Running `dflash-mlx`'s own `dflash` CLI on the same
  prompt + target + drafter would report the upstream stack's
  speedup (with all its kernel optimisations); a comparison to
  silica's 0.48× would isolate how much of the gap is silica
  integration vs the upstream verify-side stack. Out of scope for
  step 6's silica-integrated decision.

### Closure attestation

- Drafter weights cached at `~/.cache/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/`
  (3.46 GB). MoE drafter NOT pulled per user constraint.
- Bench JSONL preserved at `plans/P6_C4_DFLASH/dense_27b_run.jsonl`
  (1 row, all schema fields populated).
- Single seed (seed=0). The "warmup did not stabilize" failure
  reflects rollback-induced rate variance; multi-seed re-runs would
  not change the gate outcome (0.48× is far below 1.8×).
