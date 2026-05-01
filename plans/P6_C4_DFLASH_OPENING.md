# P-6 Track C.4 DFlash Spike — D-021 Step 6 Opening

| Field         | Value                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| Phase         | P-6 (Performance Phase) — D-021 step 6                                                                       |
| Status        | orientation drafted; no code on disk; awaits user review before sub-unit (a) begins                          |
| Last updated  | 2026-05-01                                                                                                   |
| Scope owner   | Xin Zhou                                                                                                     |
| Predecessors  | D-021 step 5 spec foundation closed at v1.7.19 (`plans/P6_SPEC_FOUNDATION_OPENING.md` §6.1)                  |
| Successors    | D-021 step 7 (Track B 3-bit weights); D-021 step 8 (C.5 tree-shape spike, conditional on C.4 outcome)        |

This document opens **D-021 step 6 — the C.4 DFlash spike**. Step 5 closed
with the autoregressive C.1 draft-target baseline on disk: single-request
`Engine.generate` spec path, three rollback paths bound, spec-metrics
emitted into `ScenarioResult.metadata`, and two real-model spec-on
warm-decode bench scenarios (`qwen3.5-27b-warm-decode-spec-on`,
`qwen3.5-moe-35b-a3b-warm-decode-spec-on`). C.1 is the **baseline** every
later C.x compares against — its predicted ≈1.14× under the
(α≈0.5, k=4) operating point in `plans/P6_SPEC_FOUNDATION_OPENING.md`
§6.2 sits below the (1b) ≥60 tok/s gate, which is precisely why
Track C.4 / C.5 exist. Step 6 measures whether a **block-diffusion
drafter** lifts that ratio.

The spike is intentionally minimal: one scenario family, one drafter
checkpoint per target (dense and MoE), B=1 single-request shape that
mirrors step 5's (h) bench rows. The goal is a measurement, not a
production integration.

---

## 0. TL;DR

C.4 is a **drafter-only spike**: it wires the
[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx)
block-diffusion drafter into silica's existing
`silica.speculative.DraftEngine` Protocol. The I-5 protocol does **not**
change; the only thing that changes inside silica is the drafter's
**cost model** — one drafter forward emits a K-token block instead of γ
autoregressive forwards.

Two upstream-DFlash mechanisms are **explicitly out of scope** for this
spike: (a) the **tape-replay rollback** is upstream's *target-side*
GatedDeltaNet replacement, not a drafter-side concern (see upstream
README quote in §4.1); silica's existing recurrent rollback path
(`Qwen3_5Adapter.rollback_state`, step 5 sub-unit (e)) and target-side
KV rollback (`PagedKVCache.rollback`, step 5 sub-unit (d)) handle
rejection unchanged. (b) The upstream `verify_qmm` int4 Metal kernel
that accelerates the M=16 quantised matmul during target verification
is also not ported — silica's verify path runs through stock MLX
`mx.quantized_matmul`. Both deferrals shrink the upstream "5.2×
HumanEval" claim band considerably for the silica-integrated number;
see §1 for the adjusted prediction.

The spike's exit gate is the Decision Gate 1 v1.7.18 reframe quoted
from PLAN.md §13 D-021 step 6 verbatim:

> ≥1.8× silica-integrated speedup continues; ≥2.5× is one component of
> the (1b) two-condition survival rule (feeding the **full-stack
> measurement** leg) and also motivates the C.5 tree-shape spike (the
> second leg, see step 8); ≤1.8× retires (1b) only if no C.5 spike is
> pursued. **C.4 alone does not settle (1b)** — only the full-stack
> measurement or the C.5 spike does.

The output is the seven-field
`silica.bench.spec_metrics.SPECULATIVE_METRIC_FIELDS` schema
(`accept_rate`, `verify_cost_ms`, `draft_cost_ms`,
`tokens_per_target_forward`, `rollback_count`, `tree_node_visits`,
`quality_parity_status`) plus REPORT-derived fields the spike's own
`plans/P6_C4_DFLASH/REPORT.md` computes (silica-integrated speedup,
peak memory, draft-overhead-per-step). The schema is **not** bumped at
the spike — derived fields live in the report, not in the schema.

The spike does **not** open multi-request hybrid spec — that remains
deferred under step 5 (c) slice 3 — and does **not** change the
`DraftEngine` Protocol surface.

---

## 1. Motivation — what step 6 measures

Step 5's foundation gives silica a working speculative-decoding path
where a small autoregressive draft model (Qwen3.5-0.8B) proposes tokens
to a large target. The expected speedup at the (α≈0.5, k=4) operating
point measured in P-6.0.5 Unit 7 is ≈1.14× — below the standard
Leviathan-et-al. ≥1.2× threshold and well below the (1a) → (1b) gap
silica needs to close (40 → 60 tok/s requires ≈1.5× full-stack uplift).

C.1 is bandwidth-bound on the target verify (verify forward at k=4 is
1.494× a single target forward at 55% utilisation; raising k diminishes
returns sublinearly under the corrected 15.13 GB anchor). The single
lever C.4's **drafter-only** scope introduces is **drafter cost
reduction**: a block-diffusion drafter that emits K tokens in one
forward instead of γ = K - 1 autoregressive forwards turns the speedup
denominator from `γ * c_draft + c_verify(k)` into
`c_draft_block + c_verify(k)` for the same yielded-tokens expectation
`(1 - α^k) / (1 - α)`. The two upstream verify-side mechanisms
(tape-replay and the verify_qmm int4 Metal kernel; see §0 and §4.1)
are deferred — the spike does **not** harvest them, so silica's
`c_verify(k)` and recurrent rollback cost stay at their step-5 anchors.

Under that scope, the predicted speedup band is bounded by what
drafter-cost reduction alone delivers: **roughly 1.5-2.2×** at α ∈
[0.5, 0.7], not the 2-3× a full-DFlash port would predict. The band
also assumes the **stateless-drafter** wrapper (§4.1) — every
`propose` re-runs the block-diffusion forward from the committed
prefix, so `c_draft_block` is paid in full per cycle regardless of
partial accept. A future stateful re-drafting hook (OQ-5) would
*widen* this band, not narrow it; the spike's gate decision uses the
worst-case estimate. Upstream's "5.2× single-request HumanEval" claim
is on a stack that includes both the verify_qmm kernel and tape-replay
verify on a CUDA target; the silica-integrated number for a
drafter-only spike is materially lower and lands closer to the C.1
baseline than the upstream headline suggests.

The spike is the smallest experiment that turns "predicted band" into
"silica-integrated number." If the integrated number lands ≥1.8×, C.4
becomes a viable dense (1a)-survival lever and step 7 (Track B 3-bit)
stacks on it. If it lands ≥2.5× *with this drafter-only scope*, that is
a strongly positive surprise and triggers a separate full-DFlash-port
proposal beyond step 6. If it lands ≤1.8×, (1b) retires unless the C.5
spike rescues it independently — and a "port the verify-side kernels
too" follow-up may be reconsidered if the drafter-only number sits
just below the gate (≥1.5×).

---

## 2. Scope and out-of-scope

### 2.1 In scope

- **Wire `bstnxbt/dflash-mlx` as an opt-in dependency** under a new
  `silica[dflash]` extras-marker so the runtime stays slim by default.
  No changes to `silica.*` for users without `dflash-mlx` installed —
  spec-on `--speculative draft_target` continues to work via the
  autoregressive C.1 path.
- **Add `silica.speculative.dflash_drafter.DFlashDrafter`** implementing
  the existing `DraftEngine` Protocol: `propose(ctx, k) -> DraftTokens`,
  `commit(ctx, accepted_len) -> None`. The K=16 block forward is hidden
  behind `propose`; what changes is `propose` cost (one drafter forward),
  not the protocol surface. Accept-rule + verification continue to use
  `silica.speculative.verify.greedy_verify`.
- **Drafter-stateless `commit` semantics.** The wrapper treats the
  block-diffusion drafter as **stateless across `propose` calls** — each
  `propose(ctx, k)` re-runs the block-diffusion forward conditioned on
  the committed prefix, regardless of which tokens of the previous
  block survived verification. Silica's existing rollback paths
  (`PagedKVCache.rollback` for target KV; `Qwen3_5Adapter.rollback_state`
  for target recurrent state) handle rejection on the target side
  unchanged from step 5. `DFlashDrafter.commit(ctx, accepted_len)` is a
  no-op in the spike. If upstream `dflash-mlx` exposes a stateful
  re-drafting hook that lets a partially-accepted block resume from
  position `accepted_len` instead of re-forwarding from scratch, that
  becomes a step-6 follow-up optimisation, not a spike requirement.
  See OQ-5 in §5 and §4.1 for the rationale.
- **Add `--speculative dflash` to `python -m scripts.bench`** alongside
  the existing `none` / `draft_target` choices. Quad-gate the two
  spec-on scenarios on (target HF cache, target env, drafter HF cache,
  drafter env) using the same `SpecConfig.draft_gate_env_var` pattern
  step 5 (h) installed.
- **One paired bench scenario family — `qwen3.5-27b-warm-decode-c4-dflash`
  and `qwen3.5-moe-35b-a3b-warm-decode-c4-dflash`** — with drafters
  `z-lab/Qwen3.5-27B-DFlash` and `z-lab/Qwen3.5-35B-A3B-DFlash`. Same
  scenario shape (same prompt, same `max_tokens`, same warm-up cycles)
  as step 5 (h)'s `*-spec-on` rows, so the speedup ratio is comparable.
- **Two-tier correctness**: (i) **synthetic-drafter cycle-1 byte-exact
  parity test** — a scripted block-diffusion drafter that emits a
  deterministic K-token block; spec-on tokens must be byte-equivalent
  to spec-off greedy on cycle 1. This is achievable because cycle-1
  generation has no batched-vs-sequential KV reduction noise yet. (ii)
  **real-model attestation invariant — "no unverified token emitted"**
  per upstream's lossless guarantee: every token the spec-on path
  yields must equal `argmax(target_logits)` at its verification step,
  recorded into the schema's `quality_parity_status` field. Long
  spec-on vs spec-off sequence equality is **not** required —
  `plans/P6_SPEC_FOUNDATION_OPENING.md` §6.1 (f) closure already
  established that fp16 batched-vs-sequential KV reduction-order
  divergence makes long-run byte-exact parity an unsuitable gate, and
  upstream `dflash-mlx`'s README explicitly notes "Output can still
  differ from pure AR because of MLX dispatch divergence, but no
  unverified token is ever emitted." The bound is the verifier
  invariant, not full sequence equality.
- **Real-model bench attestation** — run the two new scenarios under
  the four-gate active conditions on a host with both checkpoints
  cached, record the seven schema fields plus REPORT-derived
  silica-integrated speedup / peak memory / draft-overhead-per-step,
  and append a §6 closure block to this document.

### 2.2 Out of scope (deferred to step 7+ or v0.2)

- **Tape-replay verify port (target-side).** Upstream `dflash-mlx`'s
  tape-replay is a target-side recurrent-state mechanism that replaces
  full GatedDeltaNet snapshot/restore with selective accepted-step
  replay through a custom Metal kernel. Porting it would require
  replacing silica's existing `Qwen3_5Adapter.snapshot_pre_draft_state`
  / `rollback_state` path (step 5 sub-unit (e)) and writing a Metal
  innovation-tape kernel — a much larger surface than a spike. The
  spike inherits silica's existing recurrent rollback unchanged. A
  future port is a separate proposal that would only be opened if the
  drafter-only spike lands ≥1.5× and the verify-side leverage looks
  worth the kernel effort.
- **`verify_qmm` int4 Metal kernel port (target-side).** Upstream's
  custom simdgroup-MMA kernel for the M=16 quantised matmul during
  target verification is also out of scope. Silica's verify path runs
  through stock MLX `mx.quantized_matmul`. The cost-side gap between
  silica's stock-MLX `c_verify(k=16)` and upstream's
  `verify_qmm`-accelerated `c_verify(k=16)` is a real component of the
  prediction-band reduction in §1; the spike measures the
  stock-MLX-verify integrated speedup, not the upstream stack number.
- **Multi-request batched DFlash spec.** Step 5 (c) slice 3 deferral
  remains in force — the `ContinuousBatcher` GLOBAL-only gate at
  `silica/scheduler/batcher.py:268-279` is preserved. C.4 spike runs
  through `Engine.generate`, B=1, same as step 5's `*-spec-on` rows.
  Lifting the batcher gate is a separate slice for either C.1 or C.4
  after the spike measurement clears.
- **DDTree tree-verification path.** That is D-021 step 8 (C.5),
  conditional on C.4 acceptance landing high enough to motivate it.
  C.5 reuses C.4's drafter — only the verification path changes from
  single-trajectory to tree.
- **Track B 3-bit weights stacking.** That is D-021 step 7. C.4 spike
  measures the dense 27B-4bit silica-integrated speedup standalone.
- **Dynamic K (block-size) adaptation.** Upstream `dflash-mlx` appears
  to fix K=16. Spike treats K as a constructor-time choice, not a
  per-step adaptive parameter. Adaptive K under DDTree node-budget
  control is a C.5 follow-up at most.
- **Stateful drafter re-drafting after partial accept.** If upstream
  exposes a hook letting the drafter resume from `accepted_len < K`
  instead of re-forwarding from scratch, the spike does **not** wire
  it — `commit` stays no-op and `propose` re-forwards from the
  committed prefix. Stateful re-drafting is a step-6 follow-up
  optimisation that would land only after the spike measurement
  clears the gate.
- **DFlash drafter training / KD pass.** The spike consumes pre-trained
  `z-lab/Qwen3.5-*-DFlash` checkpoints. If those checkpoints' chat
  acceptance is poor, the spike records the number and exits with
  recommendation; it does not open a training sub-unit. KD/training is
  C.2 ReDrafter scope.
- **Compressed-domain attention or KV codec composition with DFlash.**
  D-003 still holds; codec stays orthogonal.

### 2.3 Explicitly preserved from prior phases

- **D-009 native-runtime constraint.** `dflash-mlx` README states
  "stock MLX plus a small number of targeted kernels"; PyTorch is not
  in the runtime path. Verified at orientation time (§5.1 OQ-1
  closure-eligible after a pip install + grep). Silica's hot path
  remains MLX-only; `dflash-mlx` runs in the same regime.
- **The `DraftEngine` Protocol surface.** `propose(ctx, k) -> DraftTokens`
  with token_ids being `tuple[int, ...]` of length **up to** k. A
  block-diffusion drafter that fills exactly K tokens in one forward
  fits the existing protocol — what changes is propose-cost and (when
  k < K) whether DFlash supports a non-block draft length.
- **Three rollback paths — all inherited from step 5.** Target-side
  KV via `PagedKVCache.rollback` / `SimpleKVCache` per-layer trim;
  target recurrent state via `Qwen3_5Adapter.snapshot_pre_draft_state`
  / `rollback_state`; draft-side via the drafter's `commit`. The
  drafter-only spike sets `DFlashDrafter.commit` to a no-op and
  relies on `propose` re-forwarding from the committed prefix — see
  §4.1 stateless-drafter invariant. The DFlash upstream tape-replay
  mechanism is *target-side*, not drafter-side, and is out of scope
  per §2.2.
- **Lossless-verifier invariant (two tiers).** Per upstream's
  guarantee, every emitted token must equal the target's greedy
  argmax at verification time. The spike enforces this in two tiers
  per §6.2: tier 1 is byte-exact spec-on / spec-off equality on the
  synthetic cycle-1 row (achievable, no batched-vs-sequential KV
  drift); tier 2 is the "no unverified token emitted" invariant on
  the real-model row, recorded into the schema's
  `quality_parity_status` field. Long real-model spec-on / spec-off
  byte equality is **not** required, per the step 5 (f) closure.

---

## 3. Sub-unit decomposition (preview)

The spike decomposes into seven sub-units. Each lands as one commit and
pauses for user review per the project's incremental-execution rule.

1. **(α) Native-runtime + license verification of `bstnxbt/dflash-mlx`.**
   `pip install dflash-mlx` into a throwaway venv; grep package source
   for `import torch`; verify MIT license matches Apache-2.0
   compatibility; record installed version. Lands as a §5.1 closure
   block in this document, not a code commit. Closes OQ-1.
2. **(β) `silica.speculative.dflash_drafter` skeleton.** A
   `DFlashDrafter` class implementing `DraftEngine` whose `__init__`
   takes a drafter checkpoint identifier and a target tokenizer
   reference; `propose` and `commit` raise `NotImplementedError` with
   wiring docstrings. Empty test asserting Protocol conformance via
   `runtime_checkable`. Lands the package extras marker
   (`pyproject.toml`) and the import-gating skipif marker for tests.
3. **(γ) `propose` implementation — synthetic drafter (§6.2 tier 1).**
   A scripted block-diffusion drafter that, given a fixed prefix,
   emits a deterministic K=16 token block. Drives the **tier-1
   correctness gate** (§6.2): on a cached small target (e.g.
   `Qwen/Qwen3-0.6B`), spec-on cycle-1 token sequences must be
   byte-equivalent to spec-off greedy under fixed seed. Cycle 1 has
   no batched-vs-sequential KV reduction-order divergence, so byte
   equality is achievable. This sub-unit is **not** a fallback for
   absent real-model checkpoints — it is the structural-correctness
   tier of the gate in its own right and runs unconditionally as part
   of the test suite.
4. **(δ) `propose` implementation — real DFlash forward (§6.2 tier 2).**
   Wires the `dflash-mlx` package's actual block-diffusion forward
   (Python API discovery is sub-unit (α)'s second deliverable; if no
   Python API exists, this sub-unit becomes a thin wrapper around
   the package's internal module path — see OQ-2). Returns
   `DraftTokens(token_ids=tuple, draft_logprobs=...)`. Drives the
   **tier-2 correctness invariant** (§6.2): under the drafter
   HF-cache gate, every emitted token of the spec-on bench row must
   equal `argmax(target_logits)` at its verification step. **Long
   spec-on vs spec-off sequence equality is not asserted** — that
   gate is unsuitable per `plans/P6_SPEC_FOUNDATION_OPENING.md` §6.1
   (f) closure. The tier-2 invariant is the verifier guarantee the
   schema's `quality_parity_status` field records.
5. **(ε) `commit` no-op + stateless-drafter invariant test.**
   `DFlashDrafter.commit(ctx, accepted_len)` is a no-op; the spike
   relies on `propose` always re-forwarding from the committed prefix.
   A synthetic test asserts that for any partial-accept length
   `accepted_len ∈ [0, K)`, the next `propose` returns tokens
   conditioned only on the committed prefix (no leakage from the
   rejected K-tail). Silica's existing target-side rollback paths
   (step 5 sub-units (d) and (e)) handle the corresponding KV /
   recurrent rejection on the target side unchanged. This sub-unit
   does **not** touch tape-replay — that path is a deferred follow-up
   per §2.2.
6. **(ζ) Bench wiring.** `--speculative dflash` CLI arg, two new
   scenario rows `qwen3.5-27b-warm-decode-c4-dflash` and
   `qwen3.5-moe-35b-a3b-warm-decode-c4-dflash`, quad-gating per
   step 5 (h) pattern. `python -m scripts.bench --list` count rises
   from 65 → 67.
7. **(η) Real-model attestation + closure.** Run the two new rows on
   a host with both checkpoints cached, record the seven
   `SPECULATIVE_METRIC_FIELDS` schema fields plus REPORT-derived
   silica-integrated speedup, peak memory, and draft-overhead-per-step
   into `plans/P6_C4_DFLASH/REPORT.md` (mirroring
   `plans/P6_0_5_BASELINE/REPORT.md`'s structure), and append §6
   closure to this document with the gate-decision callout.

This breakdown is preview-only — sub-unit boundaries may change once
(α) lands. The user pauses after each sub-unit per the standing
incremental-execution rule.

---

## 4. Architecture / interface contracts

### 4.1 `DraftEngine` Protocol unchanged; drafter is stateless across calls

The I-5 Protocol surface stays as-is:

```python
class DraftEngine(Protocol):
    def propose(self, ctx: RequestState, k: int) -> DraftTokens: ...
    def commit(self, ctx: RequestState, accepted_len: int) -> None: ...
```

`DFlashDrafter` implements this Protocol. Two facts shape the wrapper:

- **Cost model difference (internal to `propose`):** `propose` runs
  **one** drafter forward (block diffusion over K positions) instead
  of γ = K - 1 autoregressive forwards. The returned
  `DraftTokens.token_ids` is a `tuple[int, ...]` of length **up to** k
  (typically equal to min(K, k); see OQ-3 on the K vs k relationship).
- **Stateless-drafter invariant:** `commit(ctx, accepted_len)` is a
  no-op. The drafter has no internal state that survives between
  `propose` calls in the spike — each `propose` re-runs the
  block-diffusion forward conditioned on whatever prefix
  `RequestState` carries at call time, which silica's engine has
  already truncated to the committed length via the existing
  target-side rollback paths. The truncation site is
  `silica/engine/__init__.py:362+` (`un_committed = draft_count -
  yielded_count`; per-cycle `RequestState.history` and KV / recurrent
  state are rolled back to the committed prefix before the next
  `propose` runs). This invariant is what the C.1 path already relies
  on; C.4 inherits it without modification.

The reason the wrapper does **not** translate `accepted_len` into a
DFlash tape-replay state advance is that upstream's tape-replay
mechanism is **target-side**, not drafter-side. Quoting upstream
`bstnxbt/dflash-mlx` README verbatim:

> "Tape-replay rollback": instead of snapshotting and restoring the
> full GatedDeltaNet state, dflash-mlx records an innovation tape
> during verify and replays only the accepted steps through a custom
> Metal kernel.

That is upstream's replacement for the full-state snapshot-restore
silica's step 5 sub-unit (e) already implements via
`Qwen3_5Adapter.snapshot_pre_draft_state` /
`Qwen3_5Adapter.rollback_state`. Both serve the same purpose — restore
the *target's* recurrent state on rejection. The spike does not port
upstream's tape-replay; it inherits silica's snapshot-restore. (See
§2.2 for the deferral rationale and §1 for the speedup-band impact.)

### 4.2 Rollback path table (vs step 5 foundation)

| Path             | C.1 (autoregressive draft)                                  | C.4 (DFlash block drafter; drafter-only spike)                                                                          |
| ---------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Target-side KV   | `PagedKVCache.rollback(req_id, n_reject)`                   | unchanged from step 5                                                                                                   |
| Target recurrent | `Qwen3_5Adapter.rollback_state(req_id)`                     | unchanged from step 5 (does **not** use upstream tape-replay; that is the deferred verify-side port — see §2.2)         |
| Draft-side state | `DraftTargetEngine.commit(ctx, n_acc)` advances draft cache | `DFlashDrafter.commit` is a no-op; next `propose` re-forwards from the committed prefix (stateless-drafter invariant)   |

All three paths are within silica's existing surfaces. This is what
makes C.4 a *drafter-only* spike: the engine integration is one new
`DraftEngine` implementation that re-forwards on every `propose`, not
a re-engineering of the rollback foundation.

### 4.3 Drafter / target tokenizer compatibility

`z-lab/Qwen3.5-*-DFlash` checkpoints are derived from the Qwen3.5 family
and (per upstream README listing) are intended as drafters for the
matching Qwen3.5 base targets. Silica's spec foundation already requires
draft-target tokenizer equivalence (step 5 (a) pins this for the C.1
path); C.4 inherits the same constraint. The opening assumption is that
`z-lab/Qwen3.5-27B-DFlash`'s tokenizer matches
`mlx-community/Qwen3.5-27B-4bit`'s tokenizer; (α) verifies this via a
vocabulary-equality check before (β) commits to the Protocol wiring.

### 4.4 Drafter K vs verify k

DFlash's K (block size, upstream-claimed K=16) and silica's verify k
(input length to `decode_step_multi(k)`) are independent. The verify
forward pads or truncates the drafter's K-block to k positions:

- **K ≤ k:** verify forward eats the whole block. Trivial case. Spike
  defaults to `k = K = 16` for the bench rows.
- **K > k:** verify forward consumes the first k tokens; remainder
  discarded. Wastes drafter cost. Spike avoids this configuration.
- **K < k:** drafter pads with autoregressive single-token forwards
  to fill k. Defeats DFlash's amortisation; only used for parity
  experiments, not bench rows.

The default operating point is **k = 16** for both bench rows. P-6.0.5
Unit 7 measured `c_verify(k=4)` = 1.494× single target forward; the
spike re-measures `c_verify(k=16)` as a microbench row before bench
attestation, since the bandwidth-utilisation curve at k=16 may differ
from the k=4 anchor.

### 4.5 Spec-metrics emission

The schema is `silica.bench.spec_metrics.SPECULATIVE_METRIC_FIELDS`
(v1.7.15, frozen) — C.4 emits the same seven fields the C.1 baseline
does:

| Field                       | Type                  | C.4 source                                                              |
| --------------------------- | --------------------- | ----------------------------------------------------------------------- |
| `accept_rate`               | `float ∈ [0, 1]`      | empirical, measured over the bench window                               |
| `verify_cost_ms`            | `float`               | per target forward (k=K positions); same path as C.1                    |
| `draft_cost_ms`             | `float`               | per drafter forward — one block forward, not γ AR forwards              |
| `tokens_per_target_forward` | `float`               | mean accepted tokens per target verification                            |
| `rollback_count`            | `int`                 | rejection events during the measurement window                          |
| `tree_node_visits`          | `int`                 | 0 (trajectory drafter, not tree)                                        |
| `quality_parity_status`     | `QualityParityStatus` | see "parity-status emission rule" below                                 |

**Parity-status emission rule.** Set `quality_parity_status = PARITY`
on the synthetic cycle-1 row. On the real-model row, set `PARITY` iff
every emitted token equals `argmax(target_logits)` at its verification
step (the lossless invariant per upstream `dflash-mlx` README); else
`DIVERGED`. This decouples the schema row's text width from the
correctness rule.

The schema is **not** bumped for C.4. Three derived numbers the spike's
gate references — **silica-integrated speedup** (vs the v1.7.13 P-6.0
B=1 baseline), **peak resident memory**, and
**draft-overhead-per-step** — are computed by the spike's own
`plans/P6_C4_DFLASH/REPORT.md` from the schema fields plus
`ScenarioResult.warm_decode_tok_s` and the `scripts.bench` resident-memory
read. Defining them as REPORT-derived rather than schema fields
preserves the schema-freeze invariant set at v1.7.15.

---

## 5. Open questions

### 5.1 OQ-1 — `bstnxbt/dflash-mlx` runtime composition

**Question.** Is the installed package's hot path strictly MLX (no
`torch` runtime dep), and what is its exact PyPI version, dep tree, and
license at the time C.4 spike opens?

**Resolution method.** Sub-unit (α): `pip install dflash-mlx` into a
throwaway venv, run `pip show dflash-mlx`, run
`grep -rn 'import torch\|from torch' $(python -c 'import dflash_mlx; print(dflash_mlx.__path__[0])')`,
record license string, append closure block here.

**Risk if positive (no torch).** Spike proceeds as planned.

**Risk if negative (torch in runtime).** Spike scope changes from
"port + integrate" to "clean-room re-implementation per the
`feedback_vqbench_reference_pattern.md` reference-only pattern" — a
materially larger sub-unit (γ) and possibly a new sub-unit before (δ).
The DFlash paper (arxiv 2602.06036) provides a fallback implementation
spec. WebFetch on the upstream README at orientation time reported MLX-only,
so this is highly unlikely to flip — but (α) confirms it before code
lands.

### 5.2 OQ-2 — Python API surface

**Question.** Does `dflash-mlx` expose a Python API for invoking a
block draft from code (i.e. not via the `dflash` CLI)? If so, what are
the module / class / function names?

**Resolution method.** Sub-unit (α): import `dflash_mlx`, list
top-level attributes, locate the block-draft entry point. Worst case:
read the package's `dflash` CLI implementation to extract the
internal Python call sequence and call it directly.

**Risk.** No documented Python API per upstream README. If the package
internals are unstable / private, silica's wrapper either pins a
specific version + extracts the call path, or re-implements the
drafter forward against the published `z-lab/Qwen3.5-*-DFlash` weights
using the paper's algorithm. The latter is a clean-room path with
larger surface but no upstream-version coupling. (α)'s second
deliverable picks between these.

### 5.3 OQ-3 — Block size K configurability

**Question.** Is K configurable on the drafter, or fixed at the
checkpoint? Upstream README says "16 tokens in one pass" without
elaborating.

**Resolution method.** (α) third deliverable. If K is fixed, k = K = 16
becomes a hard constant in the bench scenario; if K is configurable,
k = 8 / 16 / 24 makes a useful sensitivity row.

**Why it matters.** The expected speedup `(1 - α^k) / (1 - α) /
(c_draft_block + c_verify(k))` is non-monotone in k. At α = 0.7,
optimal k is around 8-12; at α = 0.5, around 6-8. K = 16 is fine if
α ≥ 0.6, suboptimal otherwise.

### 5.4 OQ-4 — Drafter chat acceptance on Qwen3.5-27B target

**Question.** What is the empirical accept rate of
`z-lab/Qwen3.5-27B-DFlash` on chat-style prompts against
`mlx-community/Qwen3.5-27B-4bit` greedy-decoded targets?

**Resolution method.** Sub-unit (η): bench attestation produces
`accept_rate` directly. The opening prediction band is α ∈ [0.4,
0.7]; everything in step 6's gate decision flows from where the
measurement lands.

**Why it matters.** Within the drafter-only scope (§1, §2.2):
below α = 0.5 with K = 16 the predicted speedup sits below the
band, ≤1.5× regardless of `c_verify(k)` movement, and the row likely
fails the ≥1.8× engineering gate; at α ≈ 0.6 the row sits mid-band
near the gate threshold; above α = 0.7 the row approaches the upper
end of the 1.5-2.2× band. **A measurement ≥2.5× under drafter-only
scope would be a positive surprise** — the prediction does not budget
for it because the verify-side optimisations (tape-replay verify +
`verify_qmm`) are out of scope per §2.2 — and would trigger a
separate full-DFlash-port proposal beyond step 6. The spike is the
experiment that turns this prediction band into a number.

### 5.5 OQ-5 — Stateful re-drafting hook (optional optimisation)

**Question.** Does upstream `dflash-mlx` expose a Python-level hook
that lets the drafter resume from `accepted_len` after partial accept,
instead of re-running the block-diffusion forward from scratch on the
next `propose`?

**Resolution method.** Sub-unit (α): inspect the package's drafter
class for any `partial_accept` / `resume_from` API. If absent, the
spike runs with stateless `commit`-as-no-op and the next-`propose`
re-forward — that is the documented spike scope.

**Why it matters.** Stateless re-forward costs roughly an extra
`c_draft_block` per partial-accept event; if α is high (most blocks
fully accepted) the cost is negligible, if α is low it accumulates.
A stateful hook turns this into a step-6 follow-up optimisation; the
spike's gate decision is unaffected because the integrated speedup
the gate measures already includes whichever path the spike takes.

### 5.6 OQ-6 — Drafter HF-cache gating

**Question.** What environment variable guards the dflash drafter's HF
cache for the bench scenarios? Step 5 (h) used
`SpecConfig.draft_gate_env_var` for `Qwen/Qwen3.5-0.8B`; the C.4
scenarios need separate gates for `z-lab/Qwen3.5-27B-DFlash` and
`z-lab/Qwen3.5-35B-A3B-DFlash`.

**Resolution method.** Sub-unit (ζ): extend the gate-env naming
convention. Proposed: `SILICA_BENCH_DFLASH_27B` and
`SILICA_BENCH_DFLASH_35B_A3B` alongside the existing target-cache
envs. Two checkpoints, two envs — no redundancy with the C.1
`Qwen3.5-0.8B` env.

### 5.7 OQ-7 — Drafter / target precision pairing

**Question.** Upstream HF model cards pair `z-lab/Qwen3.5-27B-DFlash`
with the **full-precision** `Qwen/Qwen3.5-27B` target ("must be used
in conjunction with the target model `Qwen/Qwen3.5-27B`"), and
`z-lab/Qwen3.5-35B-A3B-DFlash` with `Qwen/Qwen3.5-35B-A3B`. Step 5's
spec-on bench scenarios pair against the **4-bit** targets
(`mlx-community/Qwen3.5-27B-4bit`, MoE 4-bit). Does the drafter's
accept-rate hold up against the 4-bit target, or does step 6 need a
different target shape?

**Resolution method.** Sub-unit (η): the bench attestation directly
measures `accept_rate`. If acceptance against the 4-bit target lands
in the predicted band (α ≥ 0.5), the pairing is fine and the spike
proceeds. If acceptance collapses well below 0.4 — suggesting
target-quantisation drift broke the drafter's assumed argmax
distribution — the spike has two fallbacks:

1. Switch the dense bench row to a smaller dense Qwen3.5 (4B / 9B)
   where the matching `z-lab/Qwen3.5-{4,9}B-DFlash` drafter exists
   and a non-4-bit target fits 48 GB. This loses comparability with
   the v1.7.13 P-6.0 27B-4bit anchor but recovers a clean drafter /
   target precision pairing.
2. Document the drift, retire the dense (1a) C.4 path, and recommend
   a follow-up that retrains the drafter against the 4-bit target —
   which is C.2 ReDrafter scope, not C.4 spike scope.

**Why it matters.** The 27B target at full precision is ~52 GB BF16
on disk and exceeds 48 GB unified memory at residency. There is no
"just use the matching-precision target" option for the 27B row;
either the 4-bit target retains acceptance, or the row retargets
smaller. The spike's first sub-unit (α) verifies tokenizer equality
(see §4.3) but acceptance-rate is a runtime measurement, not an
orientation-time check.

---

## 6. Acceptance gates

### 6.1 Spike gate (PLAN.md §13 D-021 step 6 verbatim)

> Gate (per Decision Gate 1 v1.7.18 reframe): ≥1.8× silica-integrated
> speedup continues; ≥2.5× is one component of the (1b) two-condition
> survival rule (feeding the **full-stack measurement** leg) and also
> motivates the C.5 tree-shape spike (the second leg, see step 8);
> ≤1.8× retires (1b) only if no C.5 spike is pursued. C.4 alone does
> not settle (1b) — only the full-stack measurement or the C.5 spike
> does.

**Operationalisation for this spike:**

The "silica-integrated speedup" the gate references is REPORT-derived
(see §4.5) — not a schema field. The spike's REPORT.md computes it as
the ratio of the C.4 spec-on row's wall-clock decode tok/s to the
spec-off baseline row's wall-clock decode tok/s, both at the **same
batch size and same scenario shape**. Source-of-truth fields:
`accept_rate`, `verify_cost_ms`, `draft_cost_ms`,
`tokens_per_target_forward` from `ScenarioResult.metadata`;
`warm_decode_tok_s` from the bench scenario row itself.

- **Engineering continuation (≥1.8×) — dense 27B B=1:** the C.4
  spec-on row `qwen3.5-27b-warm-decode-c4-dflash` mirrors
  `qwen3.5-27b-warm-decode-b1` exactly (same prompt, same
  `max_tokens`, same warm-up cycles), in the same way step 5's
  `qwen3.5-27b-warm-decode-spec-on` already does at
  `silica/bench/scenarios.py:2506`. The denominator for
  silica-integrated speedup is therefore the existing
  `qwen3.5-27b-warm-decode-b1` row's `warm_decode_tok_s` (the v1.7.13
  P-6.0 anchor, ≈16.0 tok/s). No new spec-off baseline scenario is
  added in step 6. Pass: ≥1.8× → continue to step 7. Fail: ≤1.8× →
  retire (1b) unless C.5 is pursued.
- **(1b) survival contribution (≥2.5×) — dense only:** if the dense
  ratio lands ≥2.5×, this satisfies the **C.4 spike** half of the
  (1b) two-condition survival rule (full-stack measurement leg). The
  rule still requires a downstream full-stack measurement that
  combines C.4 with Track A and Track B to clear ≥60 tok/s, OR the
  C.5 tree-shape spike to show headroom over the linear k=8 verify
  ceiling.
- **MoE row reporting:** `qwen3.5-moe-35b-a3b-warm-decode-c4-dflash`
  is run at B=1 and reports **absolute** `warm_decode_tok_s`,
  `accept_rate`, and `draft_cost_ms` only. The row does **not** report
  a silica-integrated speedup ratio because no MoE B=1 warm-decode
  spec-off baseline exists at v1.7.19 (the v1.7.17 P-6.0.5 MoE
  baseline is B=4 = 188.5 tok/s; comparing B=1 spec-on to B=4
  spec-off would systematically depress the ratio under the same
  drafter, since dense-MoE batched amortisation is the dominant
  signal at B=4). If the MoE row is to contribute a ratio in a
  later phase, a paired `qwen3.5-moe-35b-a3b-warm-decode-spec-off`
  B=1 row must be added first; that addition is **not** in the
  spike's scope. The MoE C.4 row's purpose is cross-target
  sensitivity (does the block-diffusion drafter generalise from
  dense to MoE at all), not gate-deciding.

### 6.2 Correctness gate (must pass at spike close)

Two-tier correctness, mirroring step 5 (f) + (i):

**Tier 1 — synthetic cycle-1 byte-exact parity (sub-unit γ).** A
scripted block-diffusion drafter that emits a deterministic K-token
block must produce, on a cached small target (e.g. `Qwen/Qwen3-0.6B`),
spec-on token sequences byte-equivalent to spec-off greedy on **cycle
1**. Cycle 1 has no batched-vs-sequential KV reduction-order
divergence yet, so byte equality is achievable. This pins that the
verify path through `silica.speculative.verify.greedy_verify` is
unaffected by the new drafter.

**Tier 2 — real-model "no unverified token emitted" invariant (sub-unit δ).**
On the real `qwen3.5-27b-warm-decode-c4-dflash` row, every token the
spec-on path yields must equal `argmax(target_logits)` at its
verification step. The schema field `quality_parity_status` is set
to `PARITY` if the invariant holds across the bench window, `DIVERGED`
otherwise. **Long spec-on vs spec-off sequence equality is not
required** — `plans/P6_SPEC_FOUNDATION_OPENING.md` §6.1 (f) already
established that fp16 batched-vs-sequential KV reduction-order
divergence makes long-run byte-exact parity an unsuitable gate, and
upstream `dflash-mlx` README states verbatim: "Output can still
differ from pure AR because of MLX dispatch divergence, but no
unverified token is ever emitted." The tier-2 invariant is the
verifier guarantee, not the sequence guarantee.

A failure at either tier fails the spike regardless of the speedup
number.

### 6.3 Toolchain attestation (at spike close)

- ruff clean (silica + tests + scripts; same baseline as v1.7.19);
- mypy clean (no new errors over the v1.7.19 baseline);
- full non-real-model test suite passes (target: ≥2616 — the v1.7.19
  baseline; spike adds at minimum two new test files
  `test_dflash_drafter_protocol.py` and `test_dflash_rollback.py` plus
  bench scenario registration tests);
- `python -m scripts.bench --list` enumerates **at least the v1.7.19
  catalog (65 scenarios) plus the two new `-c4-dflash` rows from
  sub-unit (ζ)** — i.e. ≥67 scenarios expected at spike close.
- Spike does not regress any v1.7.19 metric (`*-spec-on` rows continue
  to clear under the C.1 path; `dflash-mlx` is not a runtime dep
  unless the user opts in).

### 6.4 Memory accounting

Drafter resident bytes (the dflash drafter's MLX weights) add to the
target's footprint when spec-on. HF model cards state both upstream
checkpoints ship at **BF16**, not 4-bit:

| Component                                           | Precision  | Rough resident bytes      |
| --------------------------------------------------- | ---------- | ------------------------- |
| Dense 27B target (`mlx-community/Qwen3.5-27B-4bit`) | 4-bit      | 15.3 GB (v1.7.14 anchor)  |
| `z-lab/Qwen3.5-27B-DFlash` drafter                  | 2B BF16    | ≈4 GB                     |
| MoE 35B-A3B target (4-bit, B=4)                     | 4-bit      | 20.6 GB (v1.7.17 anchor)  |
| `z-lab/Qwen3.5-35B-A3B-DFlash` drafter              | 0.5B BF16  | ≈1 GB                     |

Spec-on totals (orientation-time conservative estimates, unchecked
until sub-unit (η) measures):

- **Dense 27B + DFlash drafter ≈ 19.3 GB** + KV growth + system
  overhead. Fits 48 GB M5 Pro with substantial margin.
- **MoE 35B-A3B B=1 + DFlash drafter ≈ 21.6 GB** at the lower B=1
  point (the 20.6 GB B=4 anchor is an upper bound). Also fits with
  margin.

A 4-bit-quantising of the drafter would drop the dense addition from
~4 GB to ~1 GB but is **not** in spike scope — the upstream-shipped
checkpoint is BF16, and re-quantising it could perturb the drafter's
accept distribution in ways that make the C.4 spike's accept-rate
measurement (OQ-4) unrepresentative of upstream behaviour. If the
spike clears the gate and memory becomes a constraint at higher B,
quantising the drafter is a step-7 follow-up.

---

## 7. PLAN.md change list (preview; lands at spike close)

At spike close, PLAN.md updates:

- **§7 P-6 step 6 status row** — flip from "in progress" to "closed at
  v1.7.20" with the derived silica-integrated speedup measurements
  (REPORT-derived per §4.5; not a schema field) quoted.
- **§13 D-021 history block** — add v1.7.20 entry summarising the
  spike outcome and the gate decision (continue to step 7 / retire
  (1b) / motivate C.5).
- **§7 P-6 acceptance gate (1b)** — flip the C.4 contribution from
  "pending" to "settled at X.YY×" and update the (1b) two-condition
  survival rule status accordingly.
- **§6 deliverables — Track C.4** entry — add the commit list (sub-units
  α..η), the `dflash-mlx` extras-marker addition to `pyproject.toml`,
  and the two new bench scenario IDs.

No `§7` or `§13` history changes land before spike close — the
opening doc is the live source of truth until then.

---

## 8. Cross-references

- `plans/PLAN.md` §13 D-021 step 6 — the verbatim contract this
  spike measures.
- `plans/PLAN.md` §6 P-6 deliverable C.4 — Track C taxonomy + impact
  estimate (note: the §6 ≥2.0× table row is **superseded** by the
  Decision Gate 1 v1.7.18 reframe quoted in §6.1 above; treat the
  §6 table as background, the §13 quote as live contract).
- `plans/P6_OPENING.md` §3 / §6 — the original Track C planning
  document; C.4 description at lines 374-388 is background, the
  ≥2.0× gate at line 414 is **stale framing absorbed by Decision
  Gate 1**.
- `plans/P6_SPEC_FOUNDATION_OPENING.md` — step 5 closure context;
  §6.2 contains the speedup-formula reference C.4's prediction band
  rests on; §4.4 contains the draft-model-selection convention C.4
  inherits.
- `plans/P6_0_5_BASELINE/REPORT.md` — Unit 7 verify-microbench data
  (k=4 ceiling 2.93× target-side / zero-drafter-cost) the C.4
  prediction band uses as the upper bound.
- `plans/P6_0_DECISION_GATE_1_OPENING.md` §2.3 — the (1b) two-condition
  survival rule's formal statement.
- Upstream `bstnxbt/dflash-mlx` (MIT) — the block-diffusion drafter
  package this spike consumes.
- Upstream `z-lab/Qwen3.5-27B-DFlash` and `z-lab/Qwen3.5-35B-A3B-DFlash`
  HF checkpoints — the drafter weights the bench scenarios load.
- `arxiv:2602.06036` — DFlash paper (z-lab, Feb 2026); the algorithm
  reference if OQ-1 falls back to clean-room re-implementation.

---

## 9. Non-goals (for the avoidance of doubt)

- C.4 is **not** the (1b) settlement. Only the full-stack measurement
  combining C.4 + Track A + Track B against ≥60 tok/s, OR the C.5
  tree-shape spike showing headroom over the linear k=8 verify
  ceiling, settles (1b). The C.4 spike's role is to deliver one of
  those two legs' inputs.
- C.4 is **not** a drafter-quality study. Acceptance rate is a
  measurement output, not a tunable. If `z-lab/Qwen3.5-27B-DFlash`'s
  chat acceptance is poor, the spike records the number and exits
  with the gate decision; it does not iterate on drafter weights.
  Drafter retraining is C.2 ReDrafter scope.
- C.4 is **not** a multi-request / batched-spec slice. Step 5 (c)
  slice 3 deferral is preserved.
- C.4 is **not** a `DraftEngine` Protocol change. The Protocol surface
  stays as defined in `silica/speculative/engine.py`.
- C.4 is **not** an optional dep removal. `dflash-mlx` becomes an
  opt-in extra (`silica[dflash]`) that the user installs only if they
  intend to run `--speculative dflash`. Default install stays slim.
- C.4 is **not** a full-DFlash port. Upstream's tape-replay verify
  rollback (target-side GatedDeltaNet replacement via custom Metal
  innovation-tape kernel) and the `verify_qmm` int4 simdgroup-MMA
  Metal kernel for the M=16 quantised matmul during target verify
  are both deferred. The spike measures only the contribution of
  drafter-cost reduction (block-diffusion `propose` replacing γ
  autoregressive forwards). A full-DFlash port that includes the
  verify-side kernels is a separate proposal that lands only if (i)
  the drafter-only spike clears the engineering gate ≥1.8× and (ii)
  the verify-side leverage justifies the kernel-engineering surface
  beyond what stock MLX delivers.

---

*Spike opens with sub-unit (α) — `pip install dflash-mlx`, native-runtime
verification, and Python-API discovery — pending user review of this
document.*
