# Silica-MLX Progress Review — Handoff for External Reviewer

| Field | Value |
| --- | --- |
| Date | 2026-04-27 |
| Project | silica-mlx — MLX-native LLM serving framework |
| Status | P-1 .. P-5 complete (v1.7.13); P-6 measurement gate landed; Tracks A-E pending |
| Reviewer use | GPT-5.5 xhigh (or any other model with code-reading bandwidth) |
| Maintainer | Xin Zhou |
| Read time | ~30 min for the full handoff; 5 min for the headline + open questions |

This document is a self-contained handoff so a reviewer who has not
seen the project before can form an independent opinion in one
sitting. It exists alongside, not instead of, `plans/PLAN.md` (the
single source of truth) and `plans/P6_OPENING.md` (the current
phase's opening).

The structure: §1-§3 give context; §4 lists what is done with
evidence pointers; §5 frames the open question; §6 is the **review
ask** — specific questions ranked by importance; §7 is a reading-
order navigation by depth; §8 is the verification map (how to
re-derive any number cited here independently).

---

## 1. Mission and Hard Constraints

**Mission (PLAN §2):** "On a single Apple Silicon Mac, let developers
run 27B-31B-class models locally with the same fluidity they get from
vLLM in the cloud." Target users are Mac developers running large
models for apps / experiments / privacy-sensitive work; **not** VQ
algorithm researchers, **not** distributed-serving researchers,
**not** cloud providers.

**Hard constraints (cannot be relaxed without breaking design intent):**

1. **MLX-native hot path (D-009).** Every tensor in
   `silica.engine` / `silica.mlx` / `silica.kvcache` /
   `silica.models` / `silica.scheduler` / `silica.vq` is `mx.array`;
   `torch.Tensor` and `numpy.ndarray` are not allowed. `vllm/`,
   `mini-sglang/`, `vqbench/` are gitignored reference-only
   checkouts; transcribe their algorithms, never import them at
   runtime.
2. **Single Mac chip, single process (PLAN §3.3, §5.3).** No
   distribution, no PD disaggregation, no tensor parallelism, no
   tokenizer/detokenizer worker split.
3. **No hand-rolled Metal kernels from scratch (PLAN §3.2 non-goals).**
   Existing kernels (mlx-lm primitives, mlx-flash, mlx-mfa, ddtree-mlx)
   are allowed. Writing new ones is not.
4. **Platform is the product, VQ is the means (D-006).** Silica-MLX
   itself is the product. KV codec compression / weight streaming /
   speculative decoding are levers to make the platform run big
   models well; they are not research subjects. Any decision that
   elevates VQ to the central deliverable is scope drift.
5. **Native capabilities, swappable implementations (Principle 9).**
   The five `Protocol` interfaces (I-1..I-5: ModelAdapter / KVManager /
   VectorCodec / WeightProvider / DraftEngine) are silica's own
   capability boundaries, not third-party plugin extension points.
   Integration points are fixed; implementations are interchangeable
   (BlockTQ ↔ RaBitQ; Resident ↔ Streaming; Noop ↔ DraftTarget ↔ EAGLE).

A reviewer's first sanity check: any suggestion that violates one of
these five constraints is on the wrong axis. We have rejected such
suggestions before; do not be polite about flagging them now.

---

## 2. Architecture in Two Sentences

Silica-MLX is a **vLLM-style scheduler core (continuous batching +
paged KV + radix prefix cache + memory-budget admission +
preempt/replay)** with a **mini-SGLang outer shell (chat session,
prefix-aware reuse, OpenAI HTTP)**, all running on the **MLX**
runtime against five frozen `Protocol` interfaces. The scheduler is
unaware of concrete adapter / codec / draft implementations; native
capabilities (VQ KV compression, weight streaming, speculative) are
built into the main loop as stubs from P-0 and progressively
replaced with real implementations through P-5 / P-6 / P-7.

**Reference map (PLAN §5.4 / §5.5):**

- `silica.engine.Engine` ↔ vLLM v1 `LLMEngine`
- `silica.scheduler.batcher.ContinuousBatcher` ↔ vLLM v1 `Scheduler`
- `silica.kvcache.paged.PagedKVCache` ↔ vLLM v1 `BlockPool`
- `silica.kvcache.prefix.RadixPrefixCache` ↔ mini-sglang radix
- `silica.vq` ↔ vqbench algorithmic reference (NumPy → MLX-native)
- `silica.chat` ↔ mini-sglang serving shell

---

## 3. Why "P-6 Performance Phase" Now

The user's stated v0.1 launch goal: **Qwen3.5-27B at ≥100 tokens/sec
on M5 Pro 48 GB**, with TTFT and RAM also better. The bandwidth
arithmetic in `plans/P6_OPENING.md` §1.2 shows this is impossible
on the dense path:

| Quantity | Value |
| --- | --- |
| M5 Pro unified-memory bandwidth | 307 GB/s (verified against Apple spec sheet) |
| Qwen3.5-27B-4bit weight bytes per autoregressive step | ~13.5 GB |
| Pure bandwidth ceiling | 307 / 13.5 ≈ 22.7 tok/s |
| Realistic upper bound (with engine overhead) | ~20 tok/s |
| Public benchmark for comparable shape | Qwen3.6-27B Q4_K_M @ 25.57 tok/s on flagship Mac (Simon Willison) |
| 100 tok/s requires | weight-read amortization across N≥4 accepted tokens (i.e. speculative decoding) |

Therefore the user accepted a **dual-target reframing** (D-017,
Q-016 Resolution):

- **Dense primary:** Qwen3.5-27B-4bit ≥**60 tok/s** (B=1, warm-start,
  with full Track A + B + C stack enabled).
- **MoE stretch validator:** Qwen3.5-35B-A3B-4bit ≥**100 tok/s
  aggregate** (B=2 primary; B=4 opt-in stretch with documented OOM
  risk).

P-6 is the phase that delivers these two numbers via five tracks:
A sync-barrier collapse, B 3-bit weights, C speculative decoding
(pulled in from P-7, promoted to T1), D TTFT levers, E weight
streaming + SSD prefix tier preserving the original P-6 scope.

---

## 4. What Is Done — Phase Recap with Evidence

| Phase | Title | Status | Evidence |
| --- | --- | --- | --- |
| P-0 | Skeleton | ✅ | `tests/test_interfaces.py`; 5 `Protocol`s + 5 stubs |
| P-1 | Baseline Engine | ✅ | `silica.engine.Engine.generate`; greedy parity vs mlx-lm |
| P-2 | Mini-vLLM Core | ✅ | `plans/P2_OPENING.md` invariant tables; 8-concurrent + prefix-cache hit demo |
| P-3 | Model Adapters | ✅ | 5 adapter families; v1.7.9 closes batched MoE scheduler-glue parity |
| P-4 | Bench Unification | ✅ | `silica.bench.runner.BenchRunner`; 9 OracleKinds; JSONL + Markdown |
| P-4.5 | Bridge (chunked-prefill + KV codec spike) | ✅ | `plans/P4_5_*.md`; slice-prefill regime α-MVP |
| P-5 | VQ KV Compression | ✅ | `plans/P5_ACCEPTANCE_SWEEP/` (924-row codec sweep + 144-row real-activation Frobenius + b-static 3-seed) |
| P-6 | Performance Phase | 🟡 | P-6.0 measurement gate landed (`plans/P6_0_BASELINE/REPORT.md`); Tracks A-E pending |
| P-7 | Speculative Decoding | ⏸ | T1 priority; sub-units land under P-6 Track C (C.1 .. C.6 — C.4 DFlash and C.5 DDTree per D-020; C.6 QuantSpec-like self-spec exploratory per v1.7.14 round-2 review). EAGLE / Medusa full-port stay v0.2 in the standalone P-7 phase. |
| P-8 | Mini-SGLang HTTP server | ⏸ | T2; sits behind chat-CLI which is already in tree |

### 4.1 P-2 — what's load-bearing

The scheduler decides the integration story for everything that follows.
Key files:

- `silica/scheduler/batcher.py` — 2706 lines; `ContinuousBatcher.step()` runs
  reclaim → admit → forward; per-row token+timestamp emission via
  `BatchEvent`.
- `silica/scheduler/budget.py` — admission policy reading
  `KVCodec.logical_bytes` / `resident_bytes`; **Principle 8 in code**:
  savings must be observable to the scheduler.
- `silica/kvcache/paged.py` + `prefix.py` + `store.py` — paged KV +
  radix prefix cache with detached-block store.

Invariants (`plans/P2_OPENING.md`): S-1..S-7 (state-machine safety),
B-1..B-9 (batched-forward shape), L-1..L-3 (lifetime).

### 4.2 P-3 — model adapters

Five families covering hybrid recurrent + sliding/full + MoE routing:

- **Qwen3.5 hybrid DeltaNet + GQA**: `silica/models/qwen3_5.py` —
  hybrid pattern `[D, D, D, G]` repeating; recurrent state is
  adapter-owned via `state_delta` (D-015); per-row snapshot at
  block-aligned boundaries (P-3-C5 α-MVP closed).
- **Gemma4-31B sliding/full attention**: `silica/models/gemma4.py` —
  hybrid `[S, S, S, S, S, F]` repeating; heterogeneous KV layout
  (sliding `BatchRotatingKVCache` + full `BatchKVCache` in same
  layer-list); `KVLayout.bytes_per_token_total` per-kind sum
  (D-3 / D-4 closed).
- **Qwen3.5-MoE 35B-A3B**: `silica/models/qwen3_5_moe.py` —
  256 experts × top-8 + sigmoid-gated shared expert; thin
  wrapper over `Qwen3_5Adapter`; `install_dispatch_proxy` seam
  preserves the quantized SwitchGLU fast path while making D-011's
  per-expert call-path testable.
- **Gemma4-MoE 26B-A4B**: `silica/models/gemma4_moe.py` —
  128 experts × top-8 + always-on dense MLP additive sum;
  factory-level `enable_moe_block` branch.
- **Plain Qwen3** (P-2 dev model): `silica/models/qwen3.py`.

The "degraded oracle" decision (P-3-D3.1 / E4 changelog
2026-04-26): batched B>1 greedy parity vs single-request drifts
under fp16, so we measure against a **direct mlx-lm batched
reference** driven with the adapter's `make_batch_cache` factory
instead. This is intentional, recorded, and re-validated on every
new adapter family.

### 4.3 P-5 — VQ KV compression

The most recent and most rigorously gated work. Replaced the
`IdentityCodec` stub with three real codec families
(`silica/vq/block_tq.py`, `silica/vq/rabitq/`) integrated into the
prefix store via the **(3b) projection-output capture path**
(P-5-F.3): pre-k_norm K is captured by a per-family proxy on
`attn.k_proj`, written to the store, reconstructed via
`apply_k_norm_then_rope` on hit-path admission.

Acceptance is a **two-part numeric cross-check** vs vqbench
(PLAN §7 P-5 Acceptance (4)):

- (a) per-block Frobenius — algorithmic parity. Synthetic-Gaussian
  half closed at P-5-A.1c, tolerance `5e-3` across (B, b)
  combinations, tighter `1e-3` on production-recommended (B=64, b=4).
  Real-activation half closed at v1.7.5 — worst gap `1.15e-4`,
  ~43× tolerance headroom.
- (b) end-to-end PPL mean-over-seeds. silica vs vqbench subprocess
  on identical chunks/seeds/codec, **two-part aggregated gate**:
  `|mean_gap| <= 2 * SEM_diff` AND `|mean_gap| < 1.0` PPL. Closed
  at v1.7.3 with `mean_gap = -0.150`, `2 * SEM_diff = 0.572` (~3.8×
  headroom).

Per-head Haar rotation landed as opt-in (default OFF) at v1.7.8;
re-measurements at v1.7.10 / v1.7.11 show 56% mean_gap reduction
on D.2a path and 5.3× std tightening on production path with no
mean shift outside SEM. **Default flip is now an administrative
landing, not an empirical question.**

Things deliberately deferred from P-5:

- Compressed-domain attention fast path (D-003): codecs do
  encode/decode, scheduler does admission accounting, `attend()`
  does not exist on the interface.
- `PagedPrefixBlockStore` codec injection: stubbed
  `NotImplementedError`; revisits when paged-attention kernel track
  matures.

### 4.4 P-6.0 — measurement gate (current)

Eight scenarios, three cache-only + five dual-gated (B=4 MoE
deliberately skipped per OOM risk):

| scenario | B | measured | util % |
| --- | --- | --- | --- |
| qwen3-0.6b (cache) | 1 | 161.16 | 15.7% |
| qwen3-0.6b (cache) | 2 | 208.72 | 20.4% |
| qwen3.5-0.8b hybrid | 1 | 123.28 | 16.1% |
| **qwen3.5-27b dense (P-6 primary)** | 1 | **16.05** | **70.6%** |
| gemma4-31b dense | 1 | 13.63 | 68.8% |
| qwen3.5-moe-35b-a3b | 1 | 76.01 | 37.1% |
| **qwen3.5-moe-35b-a3b (P-6 stretch)** | 2 | **120.93** | **59.1%** |
| gemma4-moe-26b-a4b | 1 | 68.62 | 44.7% |

**Headlines:**

- §6(2) MoE stretch ≥100 tok/s: **already cleared at baseline**
  (+20.93). Track work is bonus.
- §6(1) dense primary ≥60 tok/s: **gap 3.74×** at 70.6% bandwidth
  utilization. Engine fusion (Track A) buys at most ~1.4×
  (1/0.706); reaching 60 needs Track C.4 DFlash or C.5 DDTree to
  land in the upper half of their MLX-conservative bands. The
  gate's reachability is the open question for the phase.

Full interpretation in `plans/P6_0_BASELINE/REPORT.md` §3-§7.

---

## 5. The Open Question

**Is the dense Qwen3.5-27B-4bit ≥60 tok/s gate reachable on
M5 Pro 48 GB given silica's MLX-native constraint, or should the
gate be re-targeted at phase exit?**

The arithmetic in `plans/P6_0_BASELINE/REPORT.md` §3 lays out three
contingencies:

1. **C.4 DFlash and/or C.5 DDTree deliver ≥3× combined with Track
   A + B.** DFlash claims 6× over autoregressive on GPU, DDTree
   8.2×; the MLX ports report ~1.5× over autoregressive on real
   silicon. If silica's engine integration delivers ≥3× combined,
   the gate clears at 60-75 tok/s.
2. **Re-target to ≥40 tok/s for dense at 48 GB.** This is what the
   bandwidth math supports without the C.4/C.5 best case. Per
   `plans/P6_OPENING.md` §6 phase-exit clause, missing 60 triggers
   a Decision Log entry naming the measured engine-overhead floor.
3. **Dual-target re-confirmation.** MoE stretch is met; dense gate
   is the one stuck. Per Q-A's resolution, the dense gate is the
   phase-failing primary; missing it forces re-target.

Per Q-C's resolution the phase is **paused for user review** at
this point. Track work does not begin until target re-confirmation.

---

## 6. Review Ask — Specific Questions Ranked

The reviewer is asked to weigh in on the following, in roughly this
priority order:

### Q-R1 (highest priority): Is the dense 60 tok/s gate's reachability assessment correct?

Read `plans/P6_0_BASELINE/REPORT.md` §3 first. The arithmetic stacks
multipliers (Track A 1.05-1.15× × Track B 1.30× × Track C variants
1.4-5×) on the 16.05 tok/s baseline. Two specific places to push:

- Is the Track A multiplier honest? `plans/P6_OPENING.md` §3 Track A
  says +5-15% on bandwidth-bound dense, +30-80% on
  compute-bound MoE. The 70.6% bandwidth utilization argument says
  Track A's leverage on dense is small; **the reviewer should
  challenge this** — we are about to commit Track A as the
  cheapest first-win regardless. If our utilization analysis is
  wrong, Track A could be the gate-cracker we don't expect.
- Are the MLX-port discounts on C.4 / C.5 too aggressive or too
  generous? GPU benchmarks for DFlash claim 6×; the only available
  MLX number is "1.5× over autoregressive" from the
  humanrouter/ddtree-mlx README. We are budgeting 2-4× on C.4 and
  2.5-5× on C.5. If real silica-integrated numbers come in at
  1.2-1.5×, the dense gate is unreachable regardless.

### Q-R2: Did we draw the dual-target reframing correctly?

`plans/P6_OPENING.md` §1.3 / §1.3a justifies "dense 60 primary, MoE
100 stretch" rather than the inverse. The user explicitly named
"Qwen3.5-27B 100 tok/s" as the goal; we kept dense in the primary
slot to honor the original framing. **Should MoE be the primary
instead?** The MoE path is more reachable, exercises every
native-capability subsystem (hybrid attention + per-expert dispatch +
paged KV + prefix cache + codec), and the 100 tok/s number lands
near where vllm-mlx already publishes. Promoting MoE to primary
would let dense be the optional stretch that forces a re-target
question rather than a phase-failure question.

### Q-R3: Is the bandwidth-physics framing the right central design constraint?

The whole P-6 plan rests on §1.2's claim that "100 tok/s on dense
27B-4bit on M5 Pro is not credibly reachable." We anchor on
1000 ms / (1000 ms / token × 13.5 GB/step / 307 GB/s). Possible
errors a reviewer might catch:

- GQA / sliding-window / per-expert sparsity may reduce effective
  weight reads below `params × 0.5 B/param`. If true, the ceiling
  is higher than 22.7 tok/s.
- Group-quant metadata adds bytes; if `bits_per_param` is closer to
  4.5 than 4.0, the ceiling is ~10% lower.
- L2/SLC cache absorbs some inter-layer reads on small layers,
  shifting the effective bandwidth higher than the nominal pipe.
- The 70.6% measured utilization might already include the GQA
  benefit, so the "real" ceiling at our actual bytes/step is much
  closer to our measurement than the formula suggests.

The reviewer's best leverage: derive the ceiling from a different
direction and see if our number holds up.

### Q-R4: Track scope and order — is C-ahead-of-A defensible?

`plans/P6_0_BASELINE/REPORT.md` §7 recommends landing C.4 first to
test contingency 1 quickly, ahead of Track A. This is unusual: A is
the cheapest engine work and lands on every model; C.4 needs a new
draft engine path and an MLX kernel integration. **Reviewer's call:
should we land A first because it's "free," or land C.4 first
because it's the gate-decider?** A counter-argument is that A's
~1.15× win on dense is nearly invisible without speculative on top,
so landing A in isolation produces no meaningful rate change on the
phase-defining target.

### Q-R5: The "degraded oracle" pattern in P-3 batched parity

Real model fp16 batched-vs-single-request greedy parity drifts
because batched SDPA has different floating-point ordering. We
swapped to "vs direct mlx-lm batched reference driven with the
adapter's make_batch_cache" instead. This is documented and
re-validated on every adapter family, but **is it the right
correctness contract for a "MLX-native serving framework that
matches mlx-lm"?** A stricter reading would reject any new adapter
that diverges from single-request mlx-lm; we accept the divergence
as a fp16 fact of life. A reviewer may have a stronger position
either way.

### Q-R6: D-009 MLX-native compliance audit

Spot-check `silica/engine/`, `silica/scheduler/`, `silica/mlx/`,
`silica/kvcache/`, `silica/models/`, `silica/vq/` for any
`torch.Tensor` / `numpy.ndarray` reaching the hot path. The
constraint says these are not allowed; CI does not enforce it
beyond a grep at PR review time. **A clean audit confirms we held
the line; any leak is a real regression.**

### Q-R7: Anything we are missing from the 2026 SOTA?

Track C now has 5 sub-units after the user added DFlash and DDTree.
Beyond these and the older draft-target / ReDrafter / MTP head, are
there other 2026-published speculative or efficiency techniques the
plan should consider? Specific candidates worth checking:

- EAGLE-3 (NeurIPS '25 6.5×) — we explicitly excluded as "no MLX
  port"; if a port has appeared since the plan was written, that
  changes the calculus.
- New sampling-side optimizations (entropix, fast samplers).
- Anything in the Mac/MLX-specific space (Apple ML research
  publications, M5 Tensor Operations).

### Q-R8: Is the project's overall framing honest?

The README claims silica-mlx is a "vLLM-core architecture, MLX-native"
serving framework. We have continuous batching, paged KV, prefix
cache, and codec compression shipped, but **no HTTP server, no
real speculative, no real weight streaming**. Is the framing
overclaiming, or are the "planned" qualifiers in the table strong
enough? The user-facing positioning matters because P-8 (HTTP
server) is the deliverable that turns silica from "an engine
library" into "a serving platform."

---

## 7. Reading Order — Layered Navigation

### Layer 1 — Orient (5 minutes)

1. `README.md` — capability table vs mlx-lm / vLLM / SGLang
2. `plans/PLAN.md` §1-§5 + §13 changelog — mission, scope, status

### Layer 2 — Phase recap (15 minutes)

1. `plans/PLAN.md` §7 P-1..P-5 phase blocks (skip non-relevant
   empirical findings; come back to them in Layer 3)
2. `plans/PLAN.md` §9 D-001..D-020 — every accepted decision
3. `plans/PLAN.md` §10 Q-001..Q-016 — open and resolved questions

### Layer 3 — Current focus (20 minutes)

1. `plans/P6_OPENING.md` — the performance phase opening (836
   lines; §1-§3 + §6 + §11 are the load-bearing sections)
2. `plans/P6_0_BASELINE/REPORT.md` — the baseline interpretation
   (242 lines)

### Layer 4 — Drill-down by topic (30+ minutes)

- Scheduler design: `plans/P2_OPENING.md` (invariant tables)
- Hybrid DeltaNet + GQA: `plans/P3_DELTANET_SURVEY.md`,
  `plans/P3_C5_OPENING.md`
- MoE routing: `plans/P3_MOE_SURVEY.md`
- KV codec architecture: `plans/P5_OPENING.md`,
  `plans/P5_F_OPENING.md`
- KV codec evidence: `plans/P5_ACCEPTANCE_SWEEP/all_kv_codecs.md`,
  `plans/P5_ACCEPTANCE_SWEEP/admission_headroom.md`,
  `plans/P5_D2_INVESTIGATION/`

### Layer 5 — Code (open-ended)

- Top-level: `silica/__init__.py` re-exports `Engine`, `LLM`
- Hot path: `silica/engine/__init__.py` (drive loop),
  `silica/scheduler/batcher.py` (the heaviest file at 2706 lines;
  see the docstring at top), `silica/mlx/runner.py` (mlx-lm wrap)
- Adapters: `silica/models/{qwen3_5,gemma4,qwen3_5_moe,gemma4_moe,
  recurrent,capabilities}.py`
- Codecs: `silica/vq/block_tq.py`, `silica/vq/rabitq/`
- Bench: `silica/bench/{runner,oracles,scenarios,scenario}.py`

### Layer 6 — Tests

- `tests/test_interfaces.py` — Protocol shape pinning
- `tests/test_p3_*.py` — per-adapter functional tests (some
  dual-gated on real checkpoints)
- `tests/test_block_tq_*.py`, `tests/test_rabitq_*.py` —
  codec correctness incl. vqbench cross-check
- `tests/test_warm_decode_oracle.py` — P-6.0 oracle (13 tests
  including aggregate-window math regression)
- 2026 passed / 7 skipped on full non-real-model run as of
  commit `fbce8e7`

---

## 8. Verification Map — How to Re-derive Any Number

The reviewer should not have to trust the numbers in this document.
Every claim is reproducible.

### Bandwidth ceiling (§3)

```text
M5 Pro spec sheet: https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max
  -> 307 GB/s for M5 Pro

27B-4bit weights: model card reports 16.1 GB on disk
  -> bytes per autoregressive step ~13.5 GB (lower than disk because
     non-quantized embedding/lm_head are read but small)

Ceiling: 307 / 13.5 = 22.74 tok/s
```

### Baseline measurements (§4.4)

```bash
# Cache-only (any dev box that has run silica before)
python -m scripts.bench --scenario qwen3-0.6b-warm-decode-b1 \
    --out /tmp/check.jsonl
# expect: decode_tok_s ~155-165 tok/s; cold_ttft_ms ~40

# Dense 27B (needs ~16 GB cached + SILICA_REAL_QWEN3_5_27B=1)
SILICA_REAL_QWEN3_5_27B=1 python -m scripts.bench \
    --scenario qwen3.5-27b-warm-decode-b1 --out /tmp/check.jsonl
# expect: decode_tok_s ~16 tok/s; peak_mb ~15400

# MoE 35B-A3B B=2
SILICA_REAL_QWEN3_5_MOE=1 python -m scripts.bench \
    --scenario qwen3.5-moe-35b-a3b-warm-decode-b2 --out /tmp/check.jsonl
# expect: decode_tok_s ~120 tok/s aggregate; peak_mb ~20000
```

The original P-6.0 run produced files in `plans/P6_0_BASELINE/`
(JSONL + Markdown per scenario) and `logs/p6_0_step*.log` (full
stdout/stderr with command at the head).

### Test suite

```bash
# Lint + types
ruff check silica/ tests/
mypy silica/

# Full non-real-model suite
pytest tests/ -q \
    --ignore=tests/test_p3_qwen3_5_27b_smoke.py \
    --ignore=tests/test_p3_gemma4_31b_smoke.py \
    --ignore=tests/test_p3_qwen3_5_moe_smoke.py \
    --ignore=tests/test_p3_gemma4_moe_smoke.py \
    --ignore=tests/test_p3_qwen3_5_moe_batched_parity.py \
    --ignore=tests/test_p3_gemma4_moe_batched_parity.py \
    --ignore=tests/test_p3_gemma4_batched_smoke.py \
    --ignore=tests/test_p3_gemma4_batched_parity.py
# expect: 2026 passed, 7 skipped at commit fbce8e7
```

### Acceptance evidence — P-5

```bash
# Re-render the 924-row codec sweep report
cat plans/P5_ACCEPTANCE_SWEEP/all_kv_codecs.md

# Re-derive the (4-b) two-part gate
cat plans/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl

# Real-activation Frobenius xcheck
cat plans/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.md
```

---

## 9. What Would Change This Plan

The reviewer should feel free to say "your central assumption is
wrong" if any of the following turn out:

- **Bandwidth-physics math has a 2× error** (e.g. effective bytes/step
  is half of 13.5 GB due to GQA + sparsity). This would lift the
  dense ceiling to ~45 tok/s and make the 60 tok/s gate reachable
  with C.1-C.3 alone.
- **MLX has shipped a paged-attention primitive between PLAN's
  research date and now.** This would change Q-009 status and
  potentially open a different acceleration path.
- **DFlash's MLX port has a published silica-comparable benchmark
  number.** If someone has measured DFlash-MLX on a similar shape and
  reports a real number, the §3 contingency-1 estimate becomes
  data-driven instead of speculation.
- **A new draft technique lands publicly between plan and
  implementation that displaces DDTree.** Track C is designed to
  swap variants without affecting the rest of the phase, but a
  significantly better technique would re-rank Q-B sub-units.

---

## 10. Honesty Inventory

Things this document deliberately does not gloss over:

- **The dense 60 tok/s gate may not be reachable.** The plan accepts
  this as Q-C's pause point and does not commit to the gate as a
  hard pre-condition.
- **MoE 100 tok/s rests on the most-recently-validated code path.**
  Per-expert `get_expert` is stubbed under `ResidentWeightProvider`;
  Track E.1 has work to do before MoE streaming claims hold.
- **B=4 MoE has not been validated on real hardware.** v1.7.9 only
  pinned B=2; the b4 scenario is opt-in with explicit OOM-risk
  documentation.
- **DFlash and DDTree MLX ports are independent community work**,
  not Apple-blessed. Upstream stability (R-P6-8) is a documented
  risk; if they break, Track C still meets its gates via the older
  variants.
- **Some original P-5 acceptance items were re-anchored at
  re-measurement, not closed.** Per-head Haar rotation default flip
  is an "administrative landing" rather than an empirical question;
  the data supports flipping but the flip itself is deferred.
- **Original P-6 24 GB budget gate was retired (D-018).** v0.1 no
  longer independently validates dense residency relief; we trust
  the corrected ~15.3 GB peak measurement (v1.7.14 P5.9 step 2(a)
  supersedes the inflated 30.5 GB v1.6.1 figure that was caused by
  probe double-load).

If any of these turn out to be more load-bearing than the plan
treats them as, the reviewer's job is to flag it.

---

## 11. How to Respond

The reviewer can respond in any form (markdown notes, line-by-line
critique, a competing plan). The maintainer will integrate the
review by:

1. Recording each disagreement as either a new Decisions Log entry
   in PLAN.md (if accepted) or an entry in this handoff's appendix
   (if archived for future revisit).
2. Updating `plans/P6_OPENING.md` if track shape changes.
3. Re-running `plans/P6_0_BASELINE/REPORT.md` §3 arithmetic if any
   multiplier estimate shifts.
4. Running fresh Q-R6 (D-009 audit) if the reviewer flags a leak.

The fastest-impact response format is bullet points keyed to Q-R1
.. Q-R8 with the reviewer's position. A reviewer disagreement on
any one of those ripples through the rest cleanly.

---

## 12. External Review Round 1 — Response Integration (2026-04-27)

The first external review (a structured 10-step path) was integrated
into PLAN.md as **D-021** at the same v1.7.14 revision. The two-tier
dense gate (1a) ≥40 / (1b) ≥60 with C.4/C.5 contingency, the
foundation-first execution order, and Track A repositioned as
"general efficiency + MoE amplifier" all come from this review.

A second external review (recorded inline below) added concrete
file:line deliverables for P5.9, an exploratory C.6 sub-unit, and
an extension to Track E.2. These were folded into the same v1.7.14
revision rather than opening v1.7.15.

### Integrated additions from Round 2

- **(2a) / (2b) split on the MoE acceptance gate.** Original (2)
  ≥100 tok/s aggregate is preserved as **(2a)** — the
  already-cleared baseline anchor (120.93 tok/s at v1.7.13). New
  **(2b)** ≥150 tok/s aggregate at B=2 OR ≥100 tok/s per-row at
  B=2 is added as the MoE stretch the optimization stack actually
  has to deliver to demonstrate silica's competitiveness with
  vllm-mlx's 127.7 tok/s on M4 Max.
- **D-021 step 2 (P5.9 hardening) expanded** from 6 unkeyed bullets
  to 8 bullets (a..h) each with concrete file:line citation. The
  most load-bearing additions:
  - (a) probe double-load fix using the existing
    `silica.models.factory.adapter_from_loaded_model` (line 108)
    instead of `_mlx_lm_load` + `adapter_for_repo` chain
    (`scripts/probe_qwen3_5_27b_load.py:107` /
    `scripts/probe_gemma4_31b_load.py:151`).
  - (c) Qwen3.5 recurrent rollback path landed at P5.9 instead of
    deferred to spec-foundation step. `Qwen3_5Adapter` now exposes
    `snapshot_pre_draft_state(req_id)` and `rollback_state` restores
    that snapshot when `n_reject > 0`; `commit_state` / `free_state`
    clear pending snapshots. This prevents C.x speculative variants
    (especially C.4 / C.5 with higher reject rates than vanilla
    autoregressive draft) from corrupting recurrent state on rejection.
    P5.9 restores the pre-draft boundary; partial-accept verifier
    policy remains a C.1 / C.4 integration responsibility.
    Evidence lives in `tests/test_qwen3_5_adapter.py`; full
    non-real-model suite at landing: 2037 passed / 25 skipped.
  - (d) sustained 4K/8K context memory probe — moved up from
    P-6.0.5 to P5.9 because the §6(4) RAM headroom gate currently
    rests on inference from the 384-token P-6.0 baseline rather
    than direct measurement.
  - (e) D-009 hot-path audit promoted to a regression-locked
    check (CI hook or pinned grep test), not just PR-time review.
  - (f) speculative-metrics schema definition before any C.x
    landing — accept_rate, verify_cost_ms, draft_cost_ms,
    tokens_per_target_forward, rollback_count, tree_node_visits,
    quality_parity_status — so the C.4 spike's gate threshold
    (≥1.8× silica-integrated speedup) is evaluated on the same
    axes as C.1 / C.2 / C.3.
- **Track C.6 — QuantSpec-like same-model self-spec.** Added as
  exploratory 6th sub-unit. ICML 2025 (Tiwari et al.) reports
  ~2.5× speedup + ~1.3× memory reduction by drafting with
  hierarchical 4-bit weights + quantized KV against a full-precision
  target. Composes naturally with silica's existing surfaces
  (Track B 3-bit weights as draft tier, `silica.vq` BlockTQ /
  RaBitQ as quantized-KV draft). Only pursued if C.4 / C.5 land
  below 2× silica-integrated speedup.
- **Track E.2 extension.** From "SSD-tiered prefix cache (oMLX
  pattern)" to "active fp16 + cold compressed tier reusing the
  P-5-F (3b) capture path". Cold prefix nodes pass through
  BlockTQ / RaBitQ on eviction (memory-mapped SSD blob or
  compressed-resident depending on SSD speed); hits reconstruct
  via the existing `apply_k_norm_then_rope` route. Composes with
  silica's codec + capture proxy without violating D-003
  (no compressed-domain attention).
- **Primary-source citations.** Verified for currency at
  2026-04-27: DFlash arxiv 2602.06036 (2026-02), DDTree arxiv
  2604.12989 (2026-04), QuantSpec proceedings.mlr.press/v267/tiwari25b.html
  (ICML 2025), Mirror-SD arxiv 2510.13161 (2025), STree arxiv
  2505.14969 (2025). Mirror-SD informs C.5's heterogeneous draft
  pattern as a v0.2 follow-up; STree informs C.5's interaction
  with Qwen3.5 hybrid recurrent stacks.

### Reviewer judgments accepted in full

- "P6 的核心问题不是某个小 bug，而是物理上 dense 27B 已经卡在内存
  带宽墙" — accepted; encoded in (1a) / (1b) split and the
  bandwidth-physics framing of D-021.
- "MoE 已经在 baseline 清了 100 tok/s … 把 MoE 变成产品演示/压力
  路径" — accepted; (2a) preserves the baseline as anchor, (2b)
  raises the stretch.
- "speculative 前必须补 Qwen3.5 recurrent rollback" — accepted;
  P5.9 step 2(c).
- "Track A 别期待它单独让 dense 27B 从 16 到 60" — accepted;
  documented at the head of `plans/P6_OPENING.md` §3 and
  D-021 step 9.
- "P5 要变成 P6 每步回归门" — accepted; P5.9 step 2(g) makes the
  (4-b) two-part aggregated gate operational on every later
  Track PR.

### Reviewer judgments noted but not committed

- **"加一个 QuantSpec-like C.6"** — added as exploratory in step 8
  with a concrete pursuit threshold (only if C.4/C.5 land below
  2×). Not pre-committed because the bandwidth math says C.4 /
  C.5 are the higher-leverage paths first.
- **"做 speculative-aware scheduler"** — folded into D-021 step 5
  framing ("DraftEngine wired into the engine main loop") rather
  than spun out as a separate work item. The scheduler already
  has the integration point; the new work is wiring spec-decode
  events through `BatchEvent`.
- **"adaptive speculation"** — noted in step 8 as a C.5 follow-up,
  deferred to v0.2 unless C.5 acceptance variance shows a clear
  ROI signal. The v0.1 plan does not budget for adaptive policy
  development.
- **"MTP-head + DDTree hybrid"** — noted in step 8 as a future
  C.5 design refinement (MTP head as tree-node priority signal
  for DDTree's best-first heap). Not a separate sub-unit.
- **"session-first prefix cache"** — partially captured by Q-012
  promotion (P5.9 step 2(b)) and E.2 extension. The full
  user-facing latency story belongs in P-8 (HTTP server + session
  manager) and is not P-6 scope.

The integration is **complete as of v1.7.14**, **after the
v1.7.14 stale-text cleanup** that absorbed Round 3 review findings
(P6_OPENING.md §5 / §4a / Track C scope text; PLAN.md §3.2 + §7
P-6 canonical block + §7 P-7 block; REPORT.md §7 recommended-order
text; this HANDOFF self-recap row). The next code-touching PR is
the P5.9 hardening pass with the eight deliverables enumerated in
D-021 step 2; P5.9 step 2(a) (probe double-load fix) landed at
commit `0bd931a` ahead of the cleanup.

### Round 3 cleanup (2026-04-27, post-v1.7.14)

GPT-5.5's review against `a670a1d` flagged six stale-text issues
that survived v1.7.14 because the new decisions in D-021 / D-020
were not propagated to all pre-existing entry points:

- **High-1.** P6_OPENING.md §5 listed DFlash-MLX as "experimental;
  revisit later" while Track C.4 (P6_OPENING.md §3) and D-021 step 6
  already commit DFlash as the dense-gate-decider spike. Replaced
  the §5 entry with a v0.2-EAGLE-Medusa-Mirror-SD-STree-only line
  that no longer contradicts Track C.
- **High-2.** PLAN.md §7 P-6 canonical Scope bullet only enumerated
  C.1 / C.2 / C.3 (the v1.7.13 list) and the canonical Deliverables
  list only had a C.1 checkbox. Both updated to the v1.7.14 scope:
  C.1 .. C.6 in Scope, separate checkboxes for C.1 / C.4 / C.5 /
  C.6 in Deliverables, and E.2 changed from "SSD-tiered prefix
  cache" to "active fp16 + cold compressed two-tier" per Round 2.
- **High-3.** PLAN.md §3.2 non-goals + §7 P-7 Strategy + §7 P-7
  Notes still listed "DFlash deferred to v0.2." All three updated
  to acknowledge DFlash (C.4) and DDTree (C.5) are pulled forward
  into P-6, with EAGLE / Medusa / Mirror-SD / STree remaining the
  v0.2 candidates that the standalone P-7 phase block exists to
  catch.
- **Medium-4.** P6_OPENING.md §4a phase-exit text still said
  "exits when either dual-target gates land or user accepts
  re-target." Rewritten to match §6's (1a)+(3)+(4)+(5)+(6) anchor
  rule with the at-least-three-sub-units count; (1b)/(2b)
  explicitly noted as stretch slots that do not gate phase exit.
- **Medium-5.** REPORT.md §7 still recommended "land C.4 first as
  a measurement." Rewritten to a 9-step ordered list matching
  D-021's foundation-first sequence (P5.9 → P-6.0.5 → Decision
  Gate 1 → spec foundation + C.1 → C.4 spike at step 5, **not**
  step 1 → B → C.5/C.2/C.3 → A → D/E).
- **Medium/Low-6.** This HANDOFF's §4 Phase Recap row for P-7
  said "C.1 .. C.5"; updated to C.1 .. C.6 with C.6 explicitly
  exploratory.

The cleanup is doc-only and does not change any v1.7.14 technical
decision — it propagates D-020 / D-021 to the entry points that
v1.7.14 missed. v1.7.14 still names these contracts canonically;
now every reader-facing entry point is consistent with them.
