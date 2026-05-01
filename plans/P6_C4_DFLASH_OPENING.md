# P-6 Track C.4 DFlash Spike — D-021 Step 6 Opening

| Field         | Value                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| Phase         | P-6 (Performance Phase) — D-021 step 6                                                                       |
| Status        | (α) closed favourably (see §5.8); F-1 architecture finding triggered §0 / §1 / §2 / §3 / §4 revision; new sub-unit (αβ) added between α and β; awaits user review before (αβ) begins |
| Last updated  | 2026-05-01 (post-α revision)                                                                                 |
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

C.4 is a **drafter-only spike** in the sense that the upstream
verify-side optimisations (`verify_qmm` int4 Metal kernel +
tape-replay verify) are explicitly out of scope (see below). However,
the (α) closure findings (§5.8) established a structural fact that
the pre-α framing missed: **`bstnxbt/dflash-mlx`'s drafter is
target-conditioned**. `DFlashDraftModel.__call__` requires the
target's hidden states at specific captured layer ids as input, plus
a per-layer streaming `ContextOnlyDraftKVCache` that accumulates
those hidden states across cycles. The drafter is closer to a
C.3-style MTP head with a block-diffusion forward than to a small
autoregressive draft LM.

Concretely, the drafter state machine the spike must support is:

1. **`propose(ctx, k)`** — read the stored `target_hidden` for this
   `req_id`; call `DFlashDraftModel(noise_embedding=…, target_hidden=…,
   cache=draft_cache)`. The draft forward internally appends only
   the *target-hidden-derived context keys/values* (length `ctx_len`)
   to `draft_cache` via `ContextOnlyDraftKVCache.append_context`. The
   K mask tokens' noise keys/values are used for attention but never
   written to the cache. Returns the K - 1 drafted token ids.
2. **target verify** — silica's verify forward runs `decode_step_multi(k)`
   AND simultaneously captures hidden states at the layer ids the
   drafter consumes. This capture path is the new architectural
   surface (see sub-unit (αβ) in §3).
3. **`update_target_hidden(req_id, captured, yielded_count)`** —
   the `TargetHiddenConsumer` side channel ((β)). Slice the captured
   verify hiddens to `1 + yielded_count` positions and store as the
   new `target_hidden` for this `req_id`. The standard
   `DraftEngine.commit(ctx, accepted_len)` Protocol method stays a
   no-op for DFlash — draft-side "rollback" is implicit (rejected
   positions were noise keys/values, never written to the draft
   cache).

Three upstream-DFlash mechanisms remain **explicitly out of scope**
for this spike: (a) the **tape-replay verify rollback** is upstream's
*target-side* GatedDeltaNet snapshot/restore replacement (see
upstream README quote in §4.1); silica's existing recurrent rollback
path (`Qwen3_5Adapter.rollback_state`, step 5 sub-unit (e)) and
target-side KV rollback (`PagedKVCache.rollback`, step 5 sub-unit
(d)) handle target rejection unchanged. (b) The upstream `verify_qmm`
int4 Metal kernel that accelerates the M=16 quantised matmul during
target verification is also not ported — silica's verify path runs
through stock MLX `mx.quantized_matmul`. (c) The upstream-default
`load_target_bundle(...)` patches the target with
`_install_target_speculative_hooks(model)` and optionally with
packed weights / a custom full-attention split — silica does **not**
install those patches; it runs its own target adapters unchanged
from step 5. All three deferrals shrink the upstream "5.2× HumanEval"
claim band considerably for the silica-integrated number; see §1
for the adjusted prediction.

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
lever C.4's spike scope introduces is **drafter cost reduction**: a
block-diffusion drafter that emits K tokens in one forward instead of
γ = K - 1 autoregressive forwards turns the speedup denominator from
`γ * c_draft + c_verify(k)` into
`c_draft_block + c_capture_hidden(k) + c_verify(k)` for the same
yielded-tokens expectation `(1 - α^k) / (1 - α)`. The new
`c_capture_hidden(k)` term is the cost of capturing target hidden
states at the drafter-consumed layer ids during the verify forward
(per the F-1 finding in §5.8); whether it is materially additive or
amortises into the existing verify forward depends on MLX's ability
to expose intermediate hidden states without a second pass, which
sub-unit (αβ) microbenches. The three upstream verify-side
mechanisms (tape-replay, the `verify_qmm` int4 Metal kernel, and the
`_install_target_speculative_hooks` target patches; see §0 and §4.1)
are deferred — the spike does **not** harvest them, so silica's
`c_verify(k)` and recurrent rollback cost stay at their step-5 anchors.

Under that scope, the predicted speedup band is bounded by what
drafter-cost reduction alone delivers. The pre-α framing was
"1.5-2.2× at α ∈ [0.5, 0.7]"; F-1 surfaced the new
`c_capture_hidden(k)` denominator term, which provisionally shifted
the band to "1.4-2.0×" pending measurement. **Sub-unit (αβ.1)
measured `c_capture_hidden(k=16)` at -0.4% on cached Qwen3.5-0.8B**
(within jitter; see `plans/P6_C4_DFLASH/REPORT.md`), so the band
reverts to **1.5-2.2×** for the post-(αβ.1) prediction. A 27B-target
measurement at the actual `|target_layer_ids|` from the upstream
drafter checkpoint may shift this within the band, not below it
(see §5.5 OQ-5 closure note + REPORT.md "Interpretation"). The
drafter is **stateful per `req_id`** by design (target-conditioned
with a streaming draft KV cache; see §4.1) — `c_draft_block` is paid
in full per cycle, but the cache amortises target-hidden context
across cycles so the cumulative drafter cost stays sublinear in
cycle count.
Upstream's "5.2× single-request HumanEval" claim is on a stack that
includes both the `verify_qmm` kernel and tape-replay verify on a
CUDA target; the silica-integrated number for a *drafter-only*
spike (no upstream verify-side optimisations) is materially lower
and lands closer to the C.1 baseline than the upstream headline
suggests.

The spike is the smallest experiment that turns "predicted band" into
"silica-integrated number." If the integrated number lands ≥1.8×, C.4
becomes a viable dense (1a)-survival lever and step 7 (Track B 3-bit)
stacks on it. If it lands ≥2.5× under the drafter-only-with-hidden-capture
scope, that is a strongly positive surprise and triggers a separate
full-DFlash-port proposal beyond step 6. If it lands ≤1.8×, (1b)
retires unless the C.5 spike rescues it independently — and a "port
the verify-side kernels too" follow-up may be reconsidered if the
drafter-only-with-hidden-capture number sits just below the gate
(≥1.5×).

---

## 2. Scope and out-of-scope

### 2.1 In scope

- **Wire `bstnxbt/dflash-mlx` as an opt-in dependency** under a new
  `silica[dflash]` extras-marker so the runtime stays slim by default.
  No changes to `silica.*` for users without `dflash-mlx` installed —
  spec-on `--speculative draft_target` continues to work via the
  autoregressive C.1 path.
- **Extend silica's target adapters with intermediate hidden-state
  capture (sub-unit (αβ)).** `Qwen3_5Adapter` and `qwen3_5_moe.py`
  grow a capture-enabled variant of `decode_step_multi(k)` that
  returns both verify logits and selected-layer hidden states (at
  the layer ids the dflash drafter consumes, configurable per
  drafter checkpoint). Pattern is analogous to the P-5-F (3b)
  projection-output capture path step 5 inherited. Other adapters
  (`qwen3.py`, `gemma4.py`, `gemma4_moe.py`) gain a
  `NotImplementedError` stub for the capture path — they are out of
  scope for C.4 because no dflash drafter exists for those families
  in the upstream registry.
- **Add `silica.speculative.dflash_drafter.DFlashDrafter`** implementing
  the existing `DraftEngine` Protocol: `propose(ctx, k) -> DraftTokens`,
  `commit(ctx, accepted_len) -> None`. Per-`req_id` state holds the
  stored `target_hidden` and the per-layer `ContextOnlyDraftKVCache`s.
  The K=16 block forward is hidden behind `propose`; the
  `TargetHiddenConsumer` side channel (`update_target_hidden`) feeds
  `target_hidden` for the next cycle. Accept-rule + verification
  continue to use `silica.speculative.verify.greedy_verify`.
- **Target-conditioned drafter with `update_target_hidden`-side-channel
  semantics.** The wrapper holds, per `req_id`, the stored
  `target_hidden` and the per-layer `ContextOnlyDraftKVCache`s. The
  `target_hidden` shape is **`(1, ctx_len, |L| * hidden_size)`**
  where `L = drafter.target_layer_ids` is the (checkpoint-fixed) list
  of layer indices the drafter consumes — upstream's
  `dflash_mlx.runtime.extract_context_feature_from_dict` produces it
  as `mx.concatenate([captured_dict[layer_id + 1] for layer_id in
  target_layer_ids], axis=-1)`, where each per-layer slice has shape
  `(1, ctx_len, hidden_size)`. Each `propose(ctx, k)` reads the
  stored `target_hidden` and calls `DFlashDraftModel(noise_embedding=…,
  target_hidden=…, cache=draft_caches)`; the draft forward internally
  appends the target-hidden-derived context to the draft caches via
  `append_context`. Each
  `update_target_hidden(req_id, captured_dict, yielded_count)` (the
  `TargetHiddenConsumer` side channel; `DraftEngine.commit` stays a
  no-op) slices the verify forward's captured hidden states to
  `1 + yielded_count` positions, runs the same dict→concat aggregation
  over `target_layer_ids`, and stores the result as the new
  `target_hidden` for the next cycle. Rejected drafts were never
  written to the draft cache (only the noise keys/values for them
  existed, and those are not appended) — so draft-side "rollback" is
  implicit: simply slicing the new `target_hidden` to the committed
  length keeps the drafter's state consistent with what the engine
  emitted. Silica's existing
  target-side rollback paths (`PagedKVCache.rollback`,
  `Qwen3_5Adapter.rollback_state`) handle target rejection unchanged
  from step 5. See §4.1 for the full state-machine.
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
  / `rollback_state`; draft-side via the drafter's
  `TargetHiddenConsumer.update_target_hidden`. The C.4 spike's
  draft-side `update_target_hidden(req_id, captured, yielded_count)`
  updates the per-`req_id` stored `target_hidden` to
  `aggregate(captured)[:, :1 + yielded_count, :]`;
  `DraftEngine.commit(ctx, accepted_len)` is a no-op. The
  `ContextOnlyDraftKVCache` does not require trimming because
  rejected drafts' noise keys/values were never appended to it (see
  §4.1 state-machine). The DFlash upstream tape-replay mechanism is
  *target-side*, not drafter-side, and is out of scope per §2.2.
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

The spike decomposes into eight sub-units (post-α revision: added
(αβ) before β). Each lands as one commit and pauses for user review
per the project's incremental-execution rule.

1. **(α) Native-runtime + license + Python-API verification of
   `bstnxbt/dflash-mlx`.** `pip install dflash-mlx` into a throwaway
   venv; grep package source for `import torch`; verify MIT license
   matches Apache-2.0 compatibility; record installed version;
   identify the public Python API for invoking the drafter forward;
   discover the target-conditioning architecture. Lands as the §5.8
   closure block in this document, not a code commit. **Closed
   2026-05-01** — closes OQ-1, OQ-2, OQ-3, OQ-7 favourably; surfaces
   F-1 architecture finding that triggered this opening revision.
2. **(αβ) Target-hidden capture path on silica's target adapters.**
   Three slices:
   - **(αβ.1)** Dense `Qwen3_5Adapter`: `decode_step_multi_with_capture`
     sibling method routing through a custom forward helper that
     captures requested layer-output slices into a dict; +
     `c_capture_hidden(k=16)` microbench. **Closed.**
   - **(αβ.2)** MoE `Qwen3_5MoeAdapter`: inheritance pin (the dense
     helper lifts unchanged via `Qwen3_5MoeAdapter(Qwen3_5Adapter)`
     because mlx-lm's `qwen3_5_moe.Model` extends `qwen3_5.Model`
     directly); cached-MoE microbench row. **Closed.**
   - **(αβ.3)** Prefill capture seed + cached-prefix regression:
     adapter-side `prefill_with_capture` returning last-position logits
     + `(1, prompt_len, hidden_dim)` hidden slices so the (β) wrapper
     can seed cycle-1 `target_hidden`; tests pinning capture-enabled
     verify after a real `prefill` call (non-empty cache state, real
     mask offsets); doc cleanup of the αβ.1 / αβ.2 measurement residue.
     **Closed.**
   Other adapters (`qwen3.py`, `gemma4.py`, `gemma4_moe.py`) do **not**
   need the capture path — no upstream DFlash drafter targets them in
   `dflash_mlx.generate.DRAFT_REGISTRY`. The
   `isinstance(adapter, HiddenCaptureAdapter)` check at the (β)
   wrapper-construction site is the gate that prevents
   silently-wrong fallback.
3. **(β) `silica.speculative.dflash_drafter` skeleton.** A
   `DFlashDrafter` class implementing `DraftEngine` whose `__init__`
   takes a drafter checkpoint identifier, a target adapter handle
   (for the hidden-state capture callback), and a target tokenizer
   reference; per-`req_id` state holds stored `target_hidden` and
   per-layer `ContextOnlyDraftKVCache`s. `propose` and `commit`
   raise `NotImplementedError` with wiring docstrings referencing
   the §4.1 state-machine. Empty test asserting Protocol conformance
   via `runtime_checkable`. Lands the package extras marker
   (`pyproject.toml`) and the import-gating skipif marker for tests.
4. **(γ) `propose` against a synthetic drafter — spec-off oracle replay (§6.2 tier 1).**
   A scripted "drafter" whose `propose(ctx, k)` emits a deterministic
   K-token block. To pass the tier-1 gate the emitted tokens must be
   accepted by silica's existing `greedy_verify` against the target's
   actual argmax — this rules out hash-of-`target_hidden` schemes that
   produce arbitrary in-vocab ids, which the verifier would reject.
   The chosen design is **spec-off oracle replay**:

   - First, run spec-off greedy on the target for K cycle-1 tokens
     and record the resulting token sequence as the oracle.
   - Then, install the synthetic drafter to emit those exact recorded
     tokens; spec-on runs through `Engine.generate` with
     `--speculative dflash` (synthetic mode), and the verifier
     accepts the full drafted block because every token equals the
     target's argmax by construction.

   The fixture is **`Qwen/Qwen3.5-0.8B`** — the smallest cached
   checkpoint silica's αβ surface covers (`Qwen3Adapter` does **not**
   ship `prefill_with_capture`; the Qwen3 family is out of scope
   per `dflash_mlx.generate.DRAFT_REGISTRY`'s lack of a Qwen3
   drafter). Cycle 1 has no batched-vs-sequential KV reduction-order
   divergence, so spec-on / spec-off byte equality on the recorded
   tokens is achievable. This sub-unit is **not** a fallback for
   absent real-model checkpoints — it is the structural-correctness
   tier of the gate in its own right and runs unconditionally as
   part of the test suite. Also bound: an
   `update_target_hidden(req_id, captured, yielded_count)` test
   that asserts the next `propose` uses a `target_hidden` of length
   `1 + yielded_count`, regardless of what the previous block
   proposed.
5. **(δ) `propose` implementation — real DFlash forward (§6.2 tier 2).**
   Wires `dflash_mlx.model.DFlashDraftModel` and
   `dflash_mlx.model.ContextOnlyDraftKVCache` from the upstream
   package (per OQ-2 closure: `dflash_mlx.runtime.load_draft_bundle`
   provides the loader; the wrapper holds the loaded `DFlashDraftModel`
   instance per `req_id`). Returns
   `DraftTokens(token_ids=tuple, draft_logprobs=...)`. Drives the
   **tier-2 correctness invariant** (§6.2): under the drafter
   HF-cache gate, every emitted token of the spec-on bench row must
   equal `argmax(target_logits)` at its verification step. **Long
   spec-on vs spec-off sequence equality is not asserted** — that
   gate is unsuitable per `plans/P6_SPEC_FOUNDATION_OPENING.md` §6.1
   (f) closure. The tier-2 invariant is the verifier guarantee the
   schema's `quality_parity_status` field records.
6. **(ε) Engine integration — capture path + drafter wiring.**
   Extend `silica.engine.Engine`'s spec-on path to call the
   capture-enabled `decode_step_multi(k)` variant from (αβ) when
   the active drafter is `DFlashDrafter`, route the captured
   hidden states into `commit`, and assert under test that the
   per-`req_id` `target_hidden` is correctly sliced to
   `1 + yielded_count` after every commit. The `c_capture_hidden(k=16)`
   microbench from (αβ) re-runs end-to-end here as a sanity check
   that capture-on engine throughput tracks the microbench cost
   prediction within ±10%.
7. **(ζ) Bench wiring.** `--speculative dflash` CLI arg, two new
   scenario rows `qwen3.5-27b-warm-decode-c4-dflash` and
   `qwen3.5-moe-35b-a3b-warm-decode-c4-dflash`, quad-gating per
   step 5 (h) pattern. `python -m scripts.bench --list` count rises
   from 65 → 67.
8. **(η) Real-model attestation + closure.** Run the two new rows on
   a host with both checkpoints cached, record the seven
   `SPECULATIVE_METRIC_FIELDS` schema fields plus REPORT-derived
   silica-integrated speedup, peak memory, and draft-overhead-per-step
   into `plans/P6_C4_DFLASH/REPORT.md` (mirroring
   `plans/P6_0_5_BASELINE/REPORT.md`'s structure), and append §6
   closure to this document with the gate-decision callout.

This breakdown is preview-only — sub-unit boundaries may change once
(αβ) lands. The user pauses after each sub-unit per the standing
incremental-execution rule.

---

## 4. Architecture / interface contracts

### 4.1 `DraftEngine` Protocol surface unchanged; drafter is target-conditioned and stateful per `req_id`

The I-5 Protocol surface stays as-is:

```python
class DraftEngine(Protocol):
    def propose(self, ctx: RequestState, k: int) -> DraftTokens: ...
    def commit(self, ctx: RequestState, accepted_len: int) -> None: ...
```

`DFlashDrafter` implements this Protocol. The wrapper holds, per
`req_id`:

- **`target_hidden`** — an `mx.array` of shape
  **`(1, ctx_len, |L| * hidden_size)`** where
  `L = DFlashDraftModelArgs.target_layer_ids` (a fixed list per
  drafter checkpoint). Built by extracting per-layer slices of shape
  `(1, ctx_len, hidden_size)` from the captured-hiddens dict at keys
  `[layer_id + 1 for layer_id in target_layer_ids]` and concatenating
  along `axis=-1` — see upstream
  `dflash_mlx.runtime.extract_context_feature_from_dict`. Updated by
  `commit` (or its side-channel — see below); consumed by the next
  `propose` as the drafter's conditioning input.
- **`draft_caches`** — a list of `ContextOnlyDraftKVCache`s, one per
  drafter layer, with sink + sliding window. Mutated *inside*
  `DFlashDraftModel.__call__` via `append_context`, which writes
  only the target-hidden-derived context keys/values (length
  `ctx_len`) to the cache. The K mask tokens' noise keys/values are
  used for attention but never written to cache.

The state-machine for one `propose` / `commit` cycle is:

```text
state at entry: target_hidden_n (shape (1, ctx_n, |L|*hidden_size)),
                draft_caches_n

propose(ctx, k):
    block_token_buffer[:k] = mask_token_id
    block_token_buffer[:1] = staged_first  # from prior commit
    noise_embedding = target_embed_tokens(block_token_buffer)
    draft_hidden = DFlashDraftModel(
        noise_embedding=noise_embedding,
        target_hidden=target_hidden_n,
        cache=draft_caches_n,            # mutated: append_context(ctx_n)
    )
    drafted_logits = lm_head(draft_hidden[:, 1:, :])
    drafted_tokens = greedy_argmax(drafted_logits)
    return DraftTokens(token_ids=(staged_first, *drafted_tokens))

silica engine: target verify forward over k positions via
               decode_step_multi_with_capture, returning verify_logits
               and a captured_dict whose entries each have shape
               (1, k, hidden_size). greedy_verify(drafts,
               verify_logits) → accepted_len. yielded_count =
               min(accepted_len, max_tokens_remaining,
                   first_stop_token_position).

# Side channel separate from DraftEngine.commit (which only carries
# accepted_len). The (β) wrapper exposes update_target_hidden;
# the engine routes captured_dict + yielded_count to it.
update_target_hidden(req_id, captured_dict, yielded_count):
    selected = [captured_dict[i + 1] for i in target_layer_ids]
    full = mx.concatenate(selected, axis=-1)  # (1, k, |L|*hidden)
    self._target_hidden[req_id] = full[:, : 1 + yielded_count, :]
    # No work on draft_caches: rejected drafts' noise keys/values
    # were never written; only the next propose's append_context
    # advances the cache.

commit(ctx, accepted_len):  # standard DraftEngine surface
    return None  # no-op; state advances via update_target_hidden.

state at exit: target_hidden_{n+1} of shape
               (1, 1 + yielded_count, |L|*hidden_size),
               draft_caches_{n+1} (= draft_caches_n with ctx_n appended)
```

**Why `update_target_hidden` is split from `commit`.** The
`DraftEngine` Protocol's `commit(ctx, accepted_len)` carries only
the accepted count. Routing the verify-forward's captured hiddens
to the drafter requires a side channel; rather than widening the
Protocol surface (which would break the C.1 baseline's existing
`DraftTargetEngine` + `NoopDraftEngine`), the (β) wrapper exposes a
separate `TargetHiddenConsumer` Protocol mixin that the engine
checks via `isinstance` before calling. C.1 stays Protocol-conformant
unchanged; only DFlash-class drafters opt into the side channel.

Two consequences of this state-machine:

- **Draft-side "rollback" is implicit.** Rejected drafts were never
  persisted in `draft_caches` — only the *target-hidden-derived
  context* is appended, and the next cycle's `target_hidden` is
  pre-trimmed to the committed length before the next `propose`.
  There is no `cache.rollback(...)` call to make.
- **Silica's engine must hand the captured hidden states to
  `update_target_hidden`** (the side-channel method on the
  `TargetHiddenConsumer` Protocol; see §4.5 / sub-unit (β)). The
  current spec-on path at `silica/engine/__init__.py:362+`
  (`un_committed = draft_count - yielded_count`) handles target-side
  rollback unchanged from step 5; what (ε) adds is a side-channel
  that surfaces the verify-forward's captured dict +
  `yielded_count` to `DFlashDrafter.update_target_hidden(...)`
  alongside the existing `DraftEngine.commit(ctx, accepted_len)`
  call. `DraftEngine.commit` is unchanged in surface and stays a
  no-op for DFlash per the F-1 state machine.

The reason the wrapper does **not** plug into upstream's tape-replay
state advance is that upstream's tape-replay mechanism is
**target-side**, not drafter-side. Quoting upstream
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

| Path             | C.1 (autoregressive draft)                                  | C.4 (DFlash block drafter; drafter-only-with-hidden-capture spike)                                                                                                                               |
| ---------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Target-side KV   | `PagedKVCache.rollback(req_id, n_reject)`                   | unchanged from step 5                                                                                                                                                                            |
| Target recurrent | `Qwen3_5Adapter.rollback_state(req_id)`                     | unchanged from step 5 (does **not** use upstream tape-replay; that is the deferred verify-side port — see §2.2)                                                                                  |
| Draft-side state | `DraftTargetEngine.commit(ctx, n_acc)` advances draft cache | `DFlashDrafter.update_target_hidden(req_id, captured, yielded_count)` (TargetHiddenConsumer side channel; (β)) updates per-`req_id` `target_hidden = aggregate(captured)[:, :1 + yielded_count, :]`; `commit(ctx, accepted_len)` is a no-op; the `ContextOnlyDraftKVCache` requires no trim (rejected drafts were never written, see §4.1) |

The first two paths are inherited verbatim from step 5. The
draft-side path is new shape but stays within `DFlashDrafter` and
silica's existing per-`req_id` request bookkeeping; no I-5 Protocol
change. The new architectural surface introduced by the spike is the
hidden-state capture path on the target adapters (sub-unit (αβ));
that surface is reusable by any future target-conditioned drafter
(C.3 MTP head, C.6 self-spec, etc.) and is treated as an
infrastructure addition rather than a C.4-specific hook.

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

### 5.5 OQ-5 — `c_capture_hidden(k)` cost on silica's stock-MLX verify (closed favourably at αβ.1 / αβ.2)

**Closed across (αβ.1) and (αβ.2) on 2026-05-01.** Refined dense
Qwen3.5-0.8B: `c_capture_hidden(k=16)` = **+0.92%** (12.78 ms
baseline → 12.90 ms capture; reused-adapter methodology, the
canonical row in `plans/P6_C4_DFLASH/REPORT.md`). MoE
Qwen3.5-35B-A3B-4bit: `c_capture_hidden(k=16)` = **-3.56%** (45.26
ms → 43.64 ms; negative delta within per-iter jitter on a 20 GB
sparse-MoE checkpoint). Both deltas are in measurement-noise range
at `|capture_layer_ids| = 3`; capture is effectively free on both
fixtures. The §1 prediction band reverts to "1.5-2.2× at α ∈
[0.5, 0.7]" pending the dense 27B real-target row at (η). An earlier
"-0.41%" figure on the 0.8B fixture was load-dominated under the
per-iter-load methodology and is preserved in the REPORT as a
methodology-audit row.

**Question (preserved as historical context).** What is the additive
cost of capturing target hidden states at the drafter-consumed layer
ids during the verify forward, relative to silica's existing
`c_verify(k=16)` baseline (1.494× a single target forward at k=4 per
P-6.0.5 Unit 7)?

**Why it mattered.** The C.4 speedup denominator includes
`c_capture_hidden(k)` — see §1. If MLX's compute graph fuses the
intermediate-layer outputs into the same forward pass without a
second eval, the additive cost is small (<5% of baseline verify);
if MLX requires materialising the intermediate states with
`mx.eval(...)` separately, the additive cost is larger and shifts
the §1 prediction band. The 0.8B measurement says the former is what
MLX does; (αβ.2) and (η) re-measure on MoE / 27B respectively to
confirm the inference scales. The dense 27B at upstream's actual
`|target_layer_ids|` (read from `z-lab/Qwen3.5-27B-DFlash`'s
checkpoint config in sub-unit (β)) is the load-bearing
re-measurement.

This OQ replaced the pre-α "stateful re-drafting hook" question:
upstream's drafter is **stateful by design** (per F-1 finding), so
that question was never coherent.

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

### 5.7 OQ-7 — Drafter / target precision pairing (closed at α)

**Closed favourably at sub-unit (α), 2026-05-01.** See §5.8 for
evidence: `dflash_mlx.generate.DRAFT_REGISTRY` explicitly maps
`mlx-community/Qwen3.5-27B-4bit → z-lab/Qwen3.5-27B-DFlash` and
`mlx-community/Qwen3.5-35B-A3B-4bit → z-lab/Qwen3.5-35B-A3B-DFlash`.
Upstream supports the 4-bit MLX targets directly; the HF model
card's "must be used with `Qwen/Qwen3.5-27B`" wording is a
quality-pairing recommendation, not a precision-coupling constraint.
The spike runs against the 4-bit targets as planned, with the
upstream-blessed drafter pairing.

The pre-α fallback options below are retained as historical context
for the gate-decision audit trail, in case (η)'s `accept_rate`
measurement collapses below 0.4 anyway (e.g. due to chat-style prompts
that diverge from the upstream training distribution; this is a
*workload* concern, not a precision-coupling concern):

1. Switch the dense bench row to a smaller dense Qwen3.5 (4B / 9B)
   where the matching `z-lab/Qwen3.5-{4,9}B-DFlash` drafter exists
   per the upstream `DRAFT_REGISTRY`. This loses comparability with
   the v1.7.13 P-6.0 27B-4bit anchor but tests the drafter family
   against a different target.
2. Document the drift, retire the dense (1a) C.4 path, and recommend
   a follow-up that retrains the drafter against silica's chat-style
   workload — which is C.2 ReDrafter scope, not C.4 spike scope.

---

### 5.8 (α) closure findings — recorded on 2026-05-01

Sub-unit (α) installed `dflash-mlx==0.1.0` from PyPI into a throwaway
venv at `/tmp/dflash-probe` and inspected the package source. Findings
below close OQ-1, OQ-2, OQ-3, and OQ-7 favorably; OQ-4, OQ-5, OQ-6
remain open. **A new finding F-1 surfaces an architecture mismatch
with the opening's drafter-only-spike framing in §0 / §1 / §2.1 /
§4.1; (β) pauses pending opening revision.**

#### OQ-1 closure — runtime composition

- **Version:** `dflash-mlx 0.1.0` (PyPI metadata; package homepage
  `https://github.com/bstnxbt/dflash-mlx`).
- **License:** MIT, "Copyright (c) 2026 bstnxbt"
  (`dflash_mlx-0.1.0.dist-info/licenses/LICENSE`). Compatible with
  silica's Apache-2.0.
- **Top-level deps (`Requires:`):** `mlx`, `mlx-lm` — exactly two,
  both already in silica's runtime.
- **Transitively-installed runtime deps:** `mlx-0.31.2`,
  `mlx-lm-0.31.3`, `mlx-metal-0.31.2`, `transformers-5.7.0`,
  `tokenizers-0.22.2`, `safetensors-0.7.0`, `huggingface-hub-1.13.0`,
  `numpy-2.4.4` and standard text/HTTP packages. **No `torch`** in
  the dependency tree.
- **Source-grep confirmation:**
  `grep -rn '^import torch\|^from torch\| import torch\|, torch'
  /tmp/dflash-probe/lib/python3.13/site-packages/dflash_mlx`
  returns no matches across the six source files (`__init__.py`,
  `generate.py`, `kernels.py`, `model.py`,
  `recurrent_rollback_cache.py`, `runtime.py`, `serve.py`).

D-009 native-runtime constraint passes. Sub-unit (β)'s
`silica[dflash]` extras-marker can list `dflash-mlx>=0.1.0,<0.2`
without adding a torch dependency.

#### OQ-2 closure — Python API surface

`dflash_mlx.runtime` exports a public Python API:

- `load_target_bundle(model_ref, *, lazy, pack_target_weights, ...) -> (model, tokenizer, meta)`
- `load_draft_bundle(model_ref, *, lazy, quantize_draft) -> (drafter, ...)`
- `generate_baseline_once(...)` / `generate_dflash_once(...)`
  — high-level whole-generation entry points; return result dicts
  including `acceptance_ratio`, `phase_timings_us` (prefill / draft
  / verify / replay / commit / commit_wall), `cycles_completed`,
  `tokens_per_cycle`.
- `stream_baseline_generate(...)` / `stream_dflash_generate(...)`
  — streaming variants.

Lower-level building blocks usable from a thin wrapper:

- `dflash_mlx.model.DFlashDraftModel` — the block-diffusion drafter
  `nn.Module`. Forward signature:
  `__call__(*, noise_embedding, target_hidden, cache) -> mx.array`
  (returns hidden states; `_lm_head_logits(target_model,
  hidden_states[:, 1:, :])` then yields the K-1 drafted logits).
- `dflash_mlx.model.ContextOnlyDraftKVCache` — streaming draft KV
  cache with sink + sliding window. Per-layer; one instance per
  drafter layer per request.
- `dflash_mlx.model.DFlashDraftModelArgs.block_size` — drafter's
  maximum block size (read from the checkpoint config).
- `dflash_mlx.runtime._target_embed_tokens(target_model)` — exposes
  the target's input embedding for noise-embedding init.

The package does ship a Python API. Sub-unit (β) is not blocked on
"reverse-engineer the CLI" — the wrapper can build directly on
`load_draft_bundle` + `DFlashDraftModel` + `ContextOnlyDraftKVCache`.

#### OQ-3 closure — Block size K

- `block_tokens` is a runtime parameter on `generate_dflash_once`
  (default 16); the upper bound is the drafter checkpoint's
  `block_size` (from `DFlashDraftModelArgs`). The runtime computes
  `effective_block_tokens = max(1, min(int(block_tokens or 1),
  int(draft_model.block_size)))`.
- The default `block_tokens=16` is what upstream's CLI uses; the
  drafter checkpoint's `block_size` is the architectural maximum.
  The spike defaults to `block_tokens=16` and treats it as
  constructor-time configurable; sensitivity sweeps over k =
  8 / 12 / 16 are a step-7 follow-up if the spike clears the gate.

#### OQ-7 closure — drafter / target precision pairing (favorable)

`dflash_mlx.generate.DRAFT_REGISTRY` explicitly maps the 4-bit MLX
target IDs to the upstream drafter checkpoints:

```python
DRAFT_REGISTRY = {
    "Qwen/Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen/Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
    "Qwen/Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
    "mlx-community/Qwen3.5-27B-8bit": "z-lab/Qwen3.5-27B-DFlash",
    "mlx-community/Qwen3.5-27B-4bit": "z-lab/Qwen3.5-27B-DFlash",
    "Qwen/Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
    "mlx-community/Qwen3.5-35B-A3B-4bit": "z-lab/Qwen3.5-35B-A3B-DFlash",
}
```

The HF model card's "must be used in conjunction with `Qwen/Qwen3.5-27B`"
language is the upstream-author's pairing-by-quality recommendation,
not a precision-coupling constraint. The package itself supports the
4-bit MLX targets directly. **OQ-7's "retarget to a smaller dense"
fallback is not needed**; the spike runs against
`mlx-community/Qwen3.5-27B-4bit` and `mlx-community/Qwen3.5-35B-A3B-4bit`
as planned, with the upstream-blessed drafter pairing.

#### F-1 (NEW) — drafter is target-conditioned, contradicting opening's stateless-drafter framing

The opening's §0 / §1 / §2.1 / §4.1 framing — "stateless drafter +
no-op `commit` + each `propose` re-forwards from committed prefix" —
**does not match upstream's architecture**. Three concrete contradictions:

1. **`DFlashDraftModel.__call__` requires `target_hidden: mx.array`**
   as input (in addition to `noise_embedding` and per-layer `cache`).
   The drafter conditions on the *target model's* hidden states at
   specific captured layer ids (`draft_model.target_layer_ids`). It
   is **architecturally a target-conditioned block-diffusion head**,
   structurally similar to a C.3 MTP head, not a small-model
   autoregressive draft.
2. **The drafter carries per-layer streaming KV state**
   (`ContextOnlyDraftKVCache` with sink + window). The cache stores
   only the target-hidden-derived context keys/values; each cycle's
   `propose` call internally invokes `append_context(context_keys,
   context_values, ctx_len)` *during* the drafter forward, where
   `context_keys`/`context_values` are projections of the stored
   `target_hidden`. The K mask tokens' noise keys/values are used
   for cross-attention but **never appended** to the cache. A
   "stateless re-forward from committed prefix" would either
   (a) discard this cache (accept rate collapses), or (b) rebuild
   it from scratch each cycle (γ × layer-count cost per cycle,
   dominates the denominator).
3. **The next cycle's `target_hidden` is the verify forward's
   captured hidden states sliced to `1 + yielded_count`** — so the
   drafter depends on a target-side hidden-state output silica's
   current `decode_step_multi(k)` adapter API does not surface.
   `Qwen3_5Adapter.decode_step_multi(k)` returns logits only; to
   feed `target_hidden` to `DFlashDraftModel`, silica's target
   adapter must expose intermediate-layer hidden states at the
   layer ids the drafter was trained against (a pattern analogous
   to the P-5-F (3b) projection-output capture path step 5
   inherited).

#### F-1 disposition

This opening (post-α) absorbs F-1 in §0 / §1 / §2.1 / §2.3 / §3 /
§4.1 / §4.2 / §5.5. Concretely:

- §0 / §4.1 carry the corrected state-machine: stored `target_hidden`
  with a per-layer `ContextOnlyDraftKVCache` per `req_id`; `propose`
  consumes the stored `target_hidden` and appends only the
  target-hidden-derived context internally;
  `update_target_hidden(req_id, captured, yielded_count)` (the
  `TargetHiddenConsumer` side channel; (β)) updates the stored
  `target_hidden` to `aggregate(captured)[:, :1 + yielded_count, :]`;
  `DraftEngine.commit` is a no-op; rejected drafts' noise keys/values
  were never written, so draft-side "rollback" is implicit.
- §3 adds sub-unit (αβ) between α and β: target-hidden capture path
  on `Qwen3_5Adapter` and `qwen3_5_moe.py`, with a microbench for
  `c_capture_hidden(k=16)` (now an explicit term in the §1 speedup
  denominator) before the drafter wrapper lands.
- §5.5 OQ-5 retires the pre-α "stateful re-drafting hook" question
  (the drafter is stateful by design) and replaces it with the
  `c_capture_hidden(k)` measurement question that the §1 prediction
  band now hinges on.
- §1 shifts the predicted band from "1.5-2.2× at α ∈ [0.5, 0.7]"
  to "1.4-2.0×" pending (αβ)'s `c_capture_hidden(k)` measurement.
- The verify-side deferrals (`verify_qmm`, tape-replay verify,
  `_install_target_speculative_hooks`) remain unchanged; F-1 only
  reshapes the *drafter* integration, not the verify-side scope.

#### Resume point

Sub-unit (αβ) opens after this opening revision commits. The user
pauses per the standing incremental-execution rule between (αβ),
(β), and each subsequent sub-unit. The throwaway venv at
`/tmp/dflash-probe` is preserved for (αβ) reuse and for (β) when
the wrapper imports `dflash_mlx.model.DFlashDraftModel` /
`ContextOnlyDraftKVCache`.

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
  baseline; spike adds at minimum three new test files
  `test_qwen3_5_capture_hidden.py` (sub-unit (αβ)),
  `test_dflash_drafter_protocol.py` (sub-unit (β)), and
  `test_dflash_state_machine.py` (sub-unit (γ) / (ε)), plus bench
  scenario registration tests);
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
- C.4 is **not** a `DraftEngine` Protocol-signature change. The
  Protocol surface stays as defined in
  `silica/speculative/engine.py`. (The `DFlashDrafter` constructor
  takes new arguments — a target-adapter handle for the capture
  callback, a drafter checkpoint id — but `propose(ctx, k)` and
  `commit(ctx, accepted_len)` keep their signatures.)
- C.4 is **not** a stateless-drafter spike. Per F-1, the drafter is
  target-conditioned and stateful per `req_id`: it holds a stored
  `target_hidden` and per-layer `ContextOnlyDraftKVCache`s. The
  pre-α "stateless drafter + no-op `commit`" framing was wrong;
  see §4.1 for the corrected state-machine.
- C.4 is **not** an optional dep removal. `dflash-mlx` becomes an
  opt-in extra (`silica[dflash]`) that the user installs only if they
  intend to run `--speculative dflash`. Default install stays slim.
- C.4 is **not** a full-DFlash port. Upstream's tape-replay verify
  rollback (target-side GatedDeltaNet replacement via custom Metal
  innovation-tape kernel), the `verify_qmm` int4 simdgroup-MMA
  Metal kernel for the M=16 quantised matmul during target verify,
  and the `_install_target_speculative_hooks` target patches
  installed by `dflash_mlx.runtime.load_target_bundle` are all
  deferred. The spike measures only the contribution of
  drafter-cost reduction (block-diffusion `propose` replacing γ
  autoregressive forwards) plus the new `c_capture_hidden(k)` term
  on silica's stock-MLX verify path. A full-DFlash port that
  includes the verify-side kernels is a separate proposal that
  lands only if (i) the drafter-only-with-hidden-capture spike
  clears the engineering gate ≥1.8× and (ii) the verify-side
  leverage justifies the kernel-engineering surface beyond what
  stock MLX delivers.

---

*Spike resumes with sub-unit (αβ) — target-hidden capture path on
`Qwen3_5Adapter` and `qwen3_5_moe.py` plus `c_capture_hidden(k=16)`
microbench — pending user review of this opening revision.*
