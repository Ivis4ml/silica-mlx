# P-6 Speculative Foundation Opening — D-021 Step 5 (C.1 Draft-Target Baseline)

| Field         | Value                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| Phase         | P-6 (Performance Phase) — D-021 step 5                                                                       |
| Status        | drafted; pending user review                                                                                 |
| Last updated  | 2026-04-29                                                                                                   |
| Scope owner   | Xin Zhou                                                                                                     |
| Predecessors  | P-5 complete (v1.7.13); P-6.0.5 measurement expansion closed (v1.7.17); Decision Gate 1 closed (v1.7.18)     |
| Successors    | D-021 step 6 (C.4 DFlash spike); P-7 closure-in-v0.1-scope is settled here (see §7 OQ-2)                     |

This document opens D-021 step 5 — the first **implementation phase**
since P5.9 hardening. Every prior P-6 sub-step (P-6.0 measurement,
P5.9 hardening, P-6.0.5 measurement expansion, Decision Gate 1) was
documentation, measurement, or schema-only. Step 5 lands real engine
hot-path code: `silica.speculative.draft_target.DraftTargetEngine`,
the missing engine main-loop integration the I-5 docstring already
claims is fixed, three distinct rollback paths bound by tests, and
emission of the v1.7.15 `silica.bench.spec_metrics` schema into
`ScenarioResult.metadata` so every later C.x track lands on the same
axes.

The phase exits when ten sub-units (a, a2, b..i) are on disk and the
acceptance gates in §6 pass. By the established incremental cadence
the user pauses at each sub-unit boundary.

---

## 0. TL;DR

Step 5 lands the speculative-decoding **foundation** — not a
performance result. The exit criterion is **byte-exact greedy
parity** between spec-on and spec-off (correctness invariant), with
the v1.7.15 spec-metrics schema fully populated and the three
rollback paths bound by dedicated tests. The ≥1.2× decode-throughput
gate from PLAN §7 P-7 acceptance is **tracked but does not block step
5 closure** — it requires C.4 / C.5 drafter quality and is not
expected from the C.1 baseline (see §6.2 for rationale).

The ten sub-units (a..i, with a2 between a and b) decompose as:

1. **(a) `DraftTargetEngine` minimal implementation** — a draft model
   running the same `ModelAdapter` interface as the target, exposing
   `propose(ctx, gamma)` / `commit(ctx, accepted_len)` where γ = k - 1
   is the per-cycle draft window (k = verify forward input length;
   see §4.5 for the convention). Backed by `Qwen/Qwen3.5-0.8B`
   (canonical draft, see §4.4) against the dense
   `mlx-community/Qwen3.5-27B-4bit` target.
2. **(a2) `ModelAdapter.decode_step_multi(k)` extension across five
   adapters** — `qwen3.py`, `qwen3_5.py`, `qwen3_5_moe.py`,
   `gemma4.py`, `gemma4_moe.py`. The `(B, T)` forward over
   T = k tokens (matching `verify_k` from
   `silica.bench.microbench.target_verify`) is the
   single-target-forward verify the spec speedup depends on; running
   it as T loop iterations of `decode_step()` defeats the
   amortisation. Greedy-equivalence test pins
   `decode_step_multi(tokens)[i]` against a `decode_step()` loop on
   the same input. Cross-adapter, lands before sub-unit (b).
3. **(b) Engine main-loop integration (single-request path)** —
   `silica.engine.Engine.generate` calls `propose` → target verify →
   `commit` instead of single-token decode. Currently a no-op claim
   in the docstring; this is the actual wiring.
4. **(c) Batcher integration (multi-request path)** —
   `silica.scheduler.batcher.ContinuousBatcher.step()` extended to
   verify k tokens per spec-on row per step (anchor + γ drafts).
   Per-row draft state lives in `_BatchRow`; verification is a single
   batched target forward across rows.
5. **(d) Target-side KV rollback** — `PagedKVCache` shrinks the row's
   block list when the target rejects drafts past `accepted_len`.
   Uses the **existing** I-2 frozen interface
   `KVManager.rollback(req_id, n_reject)` (silica/kvcache/manager.py:94);
   no Protocol surface change. The spec engine / batcher computes
   `n_reject = γ - accepted_len` and calls into the existing API.
6. **(e) Recurrent state rollback (Qwen3.5)** — adapter restores the
   pre-draft `RecurrentSnapshot` (already captured in
   `_pre_draft_snapshots` per P5.9 step 2(c)) when verification
   rejects drafts past the snapshot point. New
   `ModelAdapter.rollback_recurrent_state(req_id, n_reject)` method
   on adapters that have recurrent state; default no-op on plain-KV
   adapters.
7. **(f) Greedy parity test** — temperature 0, fixed seed,
   `--speculative {none,draft_target}` produces byte-equal token
   sequences across at least one dense and one MoE scenario.
8. **(g) Spec-metrics emission** — every spec-enabled bench row
   populates the seven `silica.bench.spec_metrics` fields in
   `ScenarioResult.metadata`. Validator runs in the bench runner.
9. **(h) Bench switch + new scenarios** — `--speculative
   {none,draft_target}` flag in `scripts/bench.py`; one dense and
   one MoE warm-decode scenario registered as
   `qwen3.5-27b-warm-decode-spec-on` and
   `qwen3.5-moe-35b-a3b-warm-decode-spec-on`.
10. **(i) Recurrent + KV rollback test** — `tests/test_spec_rollback.py`
    binds the three rollback paths against synthetic accept patterns
    (full accept, partial accept, full reject); covers Qwen3.5-0.8B
    recurrent state and the paged KV block-list shrink. Cached
    `Qwen/Qwen3-0.6B` real-model row exercises target-side KV
    rollback only (plain KV; no recurrent path); the recurrent
    rollback path's only real-model exercise is the 0.8B parity row
    (sub-unit f) plus the 27B / MoE real-model acceptance rows.

PLAN §7 changes recorded in §7 of this opening.

---

## 1. Motivation — what step 5 unblocks

D-021 step 5 is the **load-bearing prerequisite** for every Track C
sub-step that follows. Specifically:

- **D-021 step 6 (C.4 DFlash spike)** measures C.4's silica-integrated
  speedup against a baseline. The Decision Gate 1 v1.7.18 reframe
  pinned the ≥1.8× / ≥2.5× gates as ratios over **C.1 draft-target**
  (PLAN §6 deliverable C.1: "spec baseline every later C.x is compared
  against"). Without C.1 in tree, C.4's spike has no anchor and its
  gate cannot be evaluated.
- **D-021 step 8 (C.5 / C.2 / C.3 selection, C.6 exploratory)**
  similarly compares against C.1. The C.5 tree-shape spike — second
  leg of the (1b) two-condition survival rule — needs C.1's accept
  rate / verify-cost / draft-cost on the same workload to argue
  whether tree-shape amortisation actually beats the linear k=8
  ceiling measured in P-6.0.5 Unit 7.
- **PLAN §7 P-7 phase block.** Acceptance bullets list exactly the
  step-5 deliverables (`DraftTargetEngine` + decode-loop integration
  + accept/rollback metrics + bench switch + ≥1.2× decode). Step 5
  closure is therefore the **v0.1-scope closure of the standalone P-7
  phase block**; remaining P-7 candidates (EAGLE / Medusa / Mirror-SD
  / STree) move to v0.2 per D-020. See §7 OQ-2 for the wording change.
- **The "fixed from P-0" docstring drift.**
  `silica/speculative/engine.py:9-11` claims the integration point —
  the decode loop calling `propose` / `commit` — is fixed from P-0.
  This is **not actually true**: neither
  `silica.engine.Engine.generate` nor
  `silica.scheduler.batcher.ContinuousBatcher.step()` invokes
  `DraftEngine` at all. The Protocol exists, `NoopDraftEngine` exists
  as the v0.1 default, but nothing calls them. Step 5 closes this
  drift by landing the integration the docstring claims and the
  PLAN.md §6 I-5 contract has presumed for two phases.

The four open data questions Decision Gate 1 left for step 5+:

1. **What is C.1's accept rate on dense Qwen3.5-27B-4bit chat?** PPL-
   findings memory shows hybrid Qwen3.5 0.6B–4B span has decent
   capacity at small sizes; whether the 0.8B draft's accept rate on
   27B target chat workloads exceeds 40% is unknown.
2. **What is the verify-cost / draft-cost ratio?** P-6.0.5 Unit 7
   measured target-side verify cost up to k=8 with **zero drafter
   cost** assumed. C.1 measures the real ratio for the first time.
3. **Does recurrent-state rollback hold under partial reject?** P5.9
   step 2(c) installed the pre-draft snapshot dict but never
   exercised the restore path against a real spec workload.
4. **MoE compatibility.** Tokenizer match between Qwen3.5 dense draft
   and Qwen3.5-MoE-A3B target is hypothesised but unverified.

Step 5 closes (1)–(3) by construction (the bench rows produce them);
(4) closes via §3 sub-unit (h) MoE bench row.

---

## 2. Scope and out-of-scope

### 2.1 In scope (ten sub-units; engine hot-path code lands)

| #  | Sub-unit                                          | New files / modifications                                                                |
| -- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| a  | `DraftTargetEngine` minimal impl                  | `silica/speculative/draft_target.py` (new)                                                |
| a2 | `ModelAdapter.decode_step_multi(k)` extension     | `silica/models/adapter.py` (Protocol signature only); `qwen3.py` / `qwen3_5.py` / `qwen3_5_moe.py` / `gemma4.py` / `gemma4_moe.py` (impls); `silica/speculative/verify.py` (free-function fallback) |
| b  | Engine main-loop integration (single-request)     | `silica/engine/__init__.py` (modify `Engine.generate`)                                    |
| c  | Batcher integration (multi-request)               | `silica/scheduler/batcher.py` (modify `ContinuousBatcher.step()` and `_BatchRow`)          |
| d  | Target-side KV rollback                           | `silica/kvcache/paged.py` (concrete `rollback(req_id, n_reject)` impl; I-2 Protocol unchanged)          |
| e  | Recurrent-state rollback (Qwen3.5)                | `silica/models/qwen3_5.py` + `silica/models/recurrent.py` (new `rollback_recurrent_state(req_id, n_reject)`) |
| f  | Greedy parity test                                | `tests/test_spec_parity.py` (new); `--speculative` on cached `Qwen/Qwen3-0.6B` smoke      |
| g  | Spec-metrics emission                             | `silica/bench/runner.py` + `silica/bench/oracles.py` (populate seven canonical fields)    |
| h  | Bench switch + two new scenarios                  | `scripts/bench.py`; `silica/bench/scenarios.py` (`-spec-on` rows, dense + MoE)            |
| i  | Three-rollback test                               | `tests/test_spec_rollback.py` (new); cached 0.6B smoke + 0.8B-as-draft synthetic harness  |

### 2.2 Out of scope (deferred to step 6+ or v0.2)

- **C.4 DFlash drafter.** Different drafter family (block-diffusion,
  not autoregressive); lands in step 6.
- **Tree-verification path (C.5).** Linear-k verify only; trees land
  in step 8.
- **Adaptive draft length.** k is a fixed scenario parameter in step
  5 (default k=4 per PLAN §6 I-5 v0.1 stub framing); per-step
  adaptation is a C.5 follow-up.
- **EAGLE / Medusa / MTP-head drafters.** Per D-020, deferred to v0.2
  P-7 (the standalone phase block), with C.3 MTP-head exploratory in
  step 8 only.
- **C.6 QuantSpec self-spec.** Exploratory; only pursued if C.4 / C.5
  land below 2× per D-021 step 8.
- **Performance gate.** ≥1.2× decode is tracked (§6.2) but does not
  block step 5 closure. The C.1 baseline is **reasonably expected to
  fall below 1.2×** on dense 27B given the 0.8B draft's likely
  accept-rate ceiling and the target's bandwidth-bound regime — this
  is what motivates Track C.4 / C.5 in the first place. See §6.2.
- **Memory hardening.** Loading a second model (the draft) on top of
  the target costs ~0.4 GB on `Qwen3.5-0.8B-4bit` (estimate; verify
  in §6.4). Step 5 measures and reports; tightening the §6(4) RAM
  gate to account for the draft is a Decision Gate 2 consideration.

### 2.3 Explicitly preserved from prior phases

- **`NoopDraftEngine`** stays as the default `DraftEngine` so spec-off
  remains the path of zero overhead. Step 5 changes nothing about its
  semantics.
- **The Protocol surface in `silica.speculative.engine`** is not
  modified. `DraftTargetEngine` is a Protocol-conforming
  implementation, not a Protocol revision.
- **Spec-metrics schema (`silica.bench.spec_metrics`).** Pinned at
  v1.7.15; step 5 is the first consumer, not the schema author.

---

## 3. Sub-unit decomposition (work breakdown)

Each sub-unit is independently committable. The user pauses at each
boundary (per `feedback_incremental_plan_execution`).

### (a) `DraftTargetEngine` minimal implementation

`silica/speculative/draft_target.py`. The draft holds a small
`ModelAdapter` instance (separate from the target's adapter; both
share the bench's `mlx.runner` if loaded in-process) plus its own
**separate** `kv_manager` — `SimpleKVCache` is sufficient for v0.1
since the draft is small and single-request-per-cycle (no need for
paging at 0.8B scale). The draft's KV grows independently of the
target's paged pool, so spec rollback never touches the target's
allocator. `propose(ctx, gamma)` runs γ = k - 1 autoregressive
forwards on the draft starting from `ctx.token_ids[-1]` and returns
the γ drafted ids plus their per-token logprobs.
`commit(ctx, accepted_len)` rolls the draft's own KV back if
`accepted_len < γ` (handled internally to the draft's `kv_manager`
instance — not the target's). The draft's own recurrent state
(Qwen3.5-0.8B is hybrid) is rolled back through the same
`snapshot_recurrent_state` / `restore_recurrent_state` mechanism
used for the target — but on the draft's adapter, fully encapsulated
inside `DraftTargetEngine.commit`.

Key contract: `DraftTargetEngine` does **not** see the target's KV
cache. The decode loop is responsible for verifying drafts against
the target and computing `accepted_len`; the draft only knows what
it proposed and how much survived.

### (a2) `ModelAdapter.decode_step_multi(k)` extension

Five concrete adapters (`qwen3.py`, `qwen3_5.py`, `qwen3_5_moe.py`,
`gemma4.py`, `gemma4_moe.py`) implement

```
decode_step_multi(
    self, tokens: mx.array, kv_handle: KVHandle
) -> tuple[mx.array, StateDelta]
```

mirroring the existing `prefill` / `decode_step` shape
(`silica/models/adapter.py:172-178`): a `(T,)` int array of input
token ids plus the request's `KVHandle`, returning logits at all T
input positions plus the post-forward `StateDelta`. Each adapter
implements the method directly; the I-1 Protocol declares the
method signature but **does not** provide a default implementation
in the Protocol body — Python's `runtime_checkable` Protocols are
structural, default method bodies declared on a `Protocol` class do
not propagate to structural conformers, so the call would
`AttributeError` on adapters that don't implement explicitly.
(Confirmed against `silica/speculative/engine.py:38`'s existing
`runtime_checkable Protocol` pattern, which has no defaults and
where every conformer ships a real implementation.)

For adapters not yet ready to ship the amortised path, the
spec-engine layer provides a free-function fallback (no Protocol
default; explicit `hasattr` dispatch):

```
# silica/speculative/verify.py
def run_verify_forward(
    adapter: ModelAdapter,
    verify_input: mx.array,        # shape (T,) — anchor + γ drafts
    kv_handle: KVHandle,
) -> tuple[mx.array, StateDelta]:
    if hasattr(adapter, "decode_step_multi"):
        return adapter.decode_step_multi(verify_input, kv_handle)
    # Correct-but-slow path — preserves output shape but defeats
    # verify amortisation. Internal accumulators stitch per-step
    # logits / StateDelta into the (T,) shape the caller expects.
    logits_acc, deltas = [], []
    for tok in verify_input:
        l, d = adapter.decode_step(tok[None], kv_handle)
        logits_acc.append(l)
        deltas.append(d)
    return mx.stack(logits_acc), _merge_state_deltas(deltas)
```

The fallback is correct but defeats verify amortisation. Sub-unit
(h)'s real-model bench rows therefore require explicit
`decode_step_multi` on Qwen3.5 dense + Qwen3.5-MoE-A3B; the
free-function path stays for tests / stub adapters / families not
on the v0.1 critical path.

Implementation differs by family for the five concrete adapters:

- **Plain KV families (Qwen3 0.6B, Gemma4 dense full-attention
  layers).** Reuses the existing `(B, T)` forward already used by
  prefill. The MLX kernel handles arbitrary T. The KV is appended
  in one shot.
- **Hybrid Qwen3.5 (DeltaNet + GQA).** DeltaNet recurrent layers'
  `step` semantics are token-by-token by construction — the hybrid
  path must run T forward steps internally on the recurrent layers
  while running a single batched forward on the full-attention
  layers. The pre-draft `RecurrentSnapshot` provides the initial
  state; T-step recurrent advancement produces the final state and
  T per-step intermediate logits.
- **MoE families (Qwen3.5-MoE-A3B, Gemma4-MoE).** Routing is
  per-token; the (B, T) forward routes T tokens independently. The
  fused `SwitchGLU` + `gather_mm` path handles arbitrary T already
  (it must — prefill uses it).

Greedy-equivalence test (`tests/test_decode_step_multi.py`) pins:
for any adapter that implements the override and any `input_ids` of
length T, `decode_step_multi(input_ids)[i]` for `i in 0..T-1` equals
the sequential `decode_step(input_ids[i])` logits at the same KV
depth, modulo numerical tolerance. The test runs on the cached 0.6B
for CI and on 0.8B / 27B at real-model acceptance time.

### (b) Engine main-loop integration (single-request)

Convention used throughout the rest of this document: **`k` = verify
forward input length**, matching `verify_k` from
`silica.bench.microbench.target_verify` directly. Of the k input
tokens to a verify forward, the first slot is the **anchor**
(previous cycle's bonus token, whose K/V is filled by this verify
forward), and the remaining `γ = k - 1` slots are **draft proposals**
from the draft engine. Yielded per cycle: up to `γ + 1 = k` tokens
(γ accepted drafts + 1 bonus). On cycle 0 the anchor is the first
sampled token from prefill (its K/V is also filled by cycle 0's
verify forward, not by prefill).

`silica.engine.Engine.generate` currently does:

```
prefill -> sample first token -> loop { decode_step -> sample -> yield }
```

After step 5:

```
# kv_handle is the request's KVHandle from kv_manager.reserve_for_prefill
prefill_logits, _ = adapter.prefill(prompt_tokens, kv_handle)
anchor = sample(prefill_logits[-1])               # first yielded token
gamma = verify_k - 1                              # draft window
loop {
    drafts = draft_engine.propose(ctx, gamma)    # gamma = k - 1 drafts
    if drafts.token_ids:
        # Verify forward consumes [anchor] + gamma drafts = k tokens.
        # Fills KV for all k positions in a single target forward.
        # Output: k logits — verify_logits[i] predicts position past
        # the i-th input position (i.e., predicts the next token
        # following input slot i).
        verify_input = mx.array([anchor] + list(drafts.token_ids))
        verify_logits, _ = adapter.decode_step_multi(verify_input, kv_handle)
        accepted_len = greedy_verify(drafts, verify_logits)  # in [0, gamma]
        for tok in drafts.token_ids[:accepted_len]:
            yield tok
        if accepted_len < gamma:
            n_reject = gamma - accepted_len
            # Same family guard as 4.1: only Qwen3.5 hybrids carry
            # recurrent state worth rolling back.
            if has_recurrent_state(adapter):
                adapter.rollback_recurrent_state(req_id, n_reject=n_reject)
            kv_manager.rollback(req_id, n_reject=n_reject)
        # Bonus: for partial accept, sample from logits at the rejected
        # draft's position (= verify_logits[accepted_len]). For full
        # accept, sample from verify_logits[k-1] (prediction past the
        # last accepted draft).
        bonus_logits = verify_logits[
            accepted_len if accepted_len < gamma else k - 1
        ]
        bonus = sample(bonus_logits)
        yield bonus
        anchor = bonus                            # carry into next cycle
        draft_engine.commit(ctx, accepted_len)
    else:
        # NoopDraftEngine path — single-token decode loop.
        # decode_step processes [anchor], filling its KV; sampler
        # produces the next token.
        token_logits, _ = adapter.decode_step(
            mx.array([anchor]), kv_handle
        )
        next_tok = sample(token_logits)
        yield next_tok
        anchor = next_tok
}
```

`greedy_verify(drafts, verify_logits)` (γ drafts to verify):

```
# verify_logits has length k = gamma + 1.
# verify_logits[0] predicts the token at the position following the
# anchor — i.e., the position of drafts[0]. So drafts[0] is verified
# against argmax(verify_logits[0]). Generalising:
# drafts[i] (for i in 0..gamma-1) is verified against
# verify_logits[i].
for i in range(gamma):
    if argmax(verify_logits[i]) != drafts.token_ids[i]:
        return i
return gamma
```

The `decode_step_multi(k)` API on `ModelAdapter` runs **one** target
forward over k tokens with prefix KV pre-primed and returns logits
for all k positions. Critical: this is **one** target forward, not k
forwards — that is the speedup mechanism. P-6.0.5 Unit 7 measured
the cost curve for this exact shape: at k=4, `forward_ms_p50 = 89.01`
vs `k=1 baseline = 59.59` — verify cost factor c_verify(k=4) ≈ 1.49.

### (c) Batcher integration (multi-request)

`ContinuousBatcher.step()` extends to per-row draft state. The
`_BatchRow` dataclass (silica/scheduler/batcher.py:157) gains
`pending_anchor: int | None` and `pending_drafts: tuple[int, ...] | None`
fields. The verify path batches across rows — one target forward per
step. Rows with empty drafts (NoopDraftEngine) contribute one token
(the anchor) to the batched forward; rows in spec mode contribute
`k = γ + 1` tokens (anchor + γ drafts). Total batched verify input
length = sum of per-row contributions.

This is the harder integration. The variable per-row draft length
breaks the existing "every row contributes one token per step"
invariant. Two options for step 5:

- **Option C.1** — pad rows with empty drafts to the max k in the
  batch, accept the wasted tokens. Simpler; loses some batch
  efficiency but correctness-clean.
- **Option C.2** — split the batch into "spec-on rows" and "spec-off
  rows" and run two target forwards per step. More efficient, but
  doubles the per-step Metal kernel launch count.

Step 5 lands Option C.1 (simpler, correctness-first); a follow-up may
revisit if batched spec efficiency matters in practice.

### (d) Target-side KV rollback

The frozen I-2 interface (`silica/kvcache/manager.py:94`) already
declares `KVManager.rollback(req_id: str, n_reject: int) -> None`
alongside `commit(req_id, n_accepted)`. Sub-unit (d) lands the
**concrete behaviour** on `PagedKVCache` (and tightens any
`SimpleKVCache` / `NullKVManager` stubs as needed): the row's
logical token count shrinks by `n_reject`. If the new tail falls in
the middle of a block, the block stays but logical count drops; if
the new tail falls on a block boundary or earlier, trailing blocks
are released back to the free pool via existing `release_blocks` /
`decrement_logical_count` primitives.

The spec-engine / batcher layer is responsible for translating
`accepted_len` into `n_reject = γ - accepted_len`; the I-2 surface
sees only `n_reject`. No new method on `KVManager`.

Critical: target KV blocks holding **already-rolled-back tokens are
not visible to the prefix cache**. The radix tree only sees committed
tokens. This means the prefix cache is unaffected by speculative
rollback.

### (e) Recurrent-state rollback (Qwen3.5)

`silica/models/qwen3_5.py:94` already declares the
`_pre_draft_snapshots: dict[str, RecurrentSnapshot]` dict — installed
in P5.9 step 2(c) but never read. Step 5 wires:

- **Capture point.** Just before `decode_step_multi(verify_input)`
  runs, the adapter calls `snapshot_recurrent_state([row_idx])` and
  stores the result keyed by `request_id`. The snapshot reflects
  the recurrent state at the position of the anchor (cycle's first
  input slot).
- **Restore point.** After verify, if `accepted_len < γ`, the
  adapter calls `restore_recurrent_state(snapshot, [row_idx])` to
  roll the recurrent state back to the pre-anchor point, then
  re-runs the recurrent path forward exactly `1 + accepted_len`
  steps (anchor + accepted drafts) before the next draft cycle.
- **Eviction.** On `commit(accepted_len == γ)`, the snapshot is
  dropped (no rollback needed). On request termination, the snapshot
  is dropped via `_pre_draft_snapshots.pop(request_id, None)`.

Open question: whether to capture per-step or per-draft-cycle.
Per-draft-cycle is sufficient for byte-exact parity but incurs O(γ)
re-execution on partial reject. Per-step (k snapshots per draft
cycle) is faster on partial reject but increases peak memory by the
snapshot footprint. Step 5 lands **per-draft-cycle** (simpler;
correctness-first); a perf follow-up revisits.

### (f) Greedy parity test

`tests/test_spec_parity.py`. Two scenarios:

1. Cached `Qwen/Qwen3-0.6B` (plain KV; no recurrent path) — runs
   spec-off vs spec-on, asserts byte-equal token streams. Cheap;
   cache-only gate.
2. `Qwen/Qwen3.5-0.8B` standalone (draft used as target; minimal
   recurrent exercise) — runs spec-off vs spec-on, asserts byte-equal
   under temperature 0. Cache-only gate.

Real 27B target parity is a **manual acceptance row** (§6.3) gated
on `SILICA_REAL_QWEN3_5_27B`, not a CI test.

### (g) Spec-metrics emission

`silica/bench/runner.py` passes a `SpecMetricCollector` into the
oracle's measurement window. The collector accumulates:

- accept count / draft count → `accept_rate`
- target forward wall time → `verify_cost_ms` (mean per step)
- draft forward wall time → `draft_cost_ms` (mean per step)
- accepted tokens / target forward count → `tokens_per_target_forward`
- partial-accept events → `rollback_count`
- 0 → `tree_node_visits` (linear; trees in step 8)
- parity check result → `quality_parity_status`

Validator (`validate_speculative_metrics`) runs at scenario completion
and fails loud on missing fields.

### (h) Bench switch + two new scenarios

`scripts/bench.py` gains `--speculative {none,draft_target}` flag.
Default `none` (NoopDraftEngine; backwards-compatible). When
`draft_target`, the runner constructs a `DraftTargetEngine` whose
draft repo is configured per scenario (default
`Qwen/Qwen3.5-0.8B`).

Two new bench scenarios:

- `qwen3.5-27b-warm-decode-spec-on` — same workload as the existing
  `qwen3.5-27b-warm-decode` (B=1, 128-token prompt, 384-token
  generation, max_tokens=384), with `draft_repo=Qwen/Qwen3.5-0.8B`,
  `verify_k=4`. Dual-gated on `SILICA_REAL_QWEN3_5_27B` +
  `SILICA_REAL_QWEN3_5_0_8B_DRAFT`.
- `qwen3.5-moe-35b-a3b-warm-decode-spec-on` — analogous against the
  MoE target. Dual-gated on `SILICA_REAL_QWEN3_5_MOE_A3B` +
  `SILICA_REAL_QWEN3_5_0_8B_DRAFT`.

### (i) Three-rollback test

`tests/test_spec_rollback.py`. Bound the three rollback paths
**without** real-model gates, using a synthetic
`AcceptPatternDraftEngine` that emits scripted token sequences and a
mock target adapter that returns scripted accept patterns:

- **Pattern A — full accept (accepted_len == γ).** No rollback
  fires; the verify forward fills KV for all k = γ+1 input positions
  (the anchor and the γ drafts). All γ drafts are committed; the
  bonus is yielded but its KV slot is filled by the next cycle's
  verify forward (where it becomes the anchor). Yielded this cycle =
  γ + 1 = k tokens.
- **Pattern B — partial accept (0 < accepted_len < γ).** KV and
  recurrent rollback fire with `n_reject = γ - accepted_len`. After
  rollback, KV growth this cycle = 1 anchor + accepted_len drafts;
  the rejected drafts past `accepted_len` are removed. Yielded =
  accepted_len + 1 (drafts + bonus from the rejected position).
- **Pattern C — full reject (accepted_len == 0).** KV / recurrent
  rollback with `n_reject = γ`. Net KV growth this cycle = 1
  position (the anchor only); all γ drafts removed. Yielded = 1
  (the bonus, sampled from `verify_logits[0]` — the prediction at
  the anchor's position).

Plus one **real-model rollback row** on cached `Qwen/Qwen3-0.6B`
gated on `--cache-only`: drives ~50 decode steps with synthetic
drafts, asserts spec-on yields the same tokens as spec-off.

---

## 4. Architecture / interface contracts

### 4.1 Three rollback paths — owners and APIs

| Path                       | Owner                                              | Trigger                    | API                                                                                |
| -------------------------- | -------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------- |
| Draft-side state           | `DraftEngine.commit`                               | After every spec cycle     | `commit(ctx, accepted_len)` — already in Protocol                                  |
| Target-side KV             | `KVManager` (concrete: `PagedKVCache`, `SimpleKVCache`) | `accepted_len < γ`         | `rollback(req_id, n_reject)` — already in I-2 frozen interface; sub-unit (d) lands concrete behaviour, no Protocol change |
| Target-side recurrent      | `ModelAdapter` (Qwen3.5 family — 0.8B / 27B / MoE) | `accepted_len < γ`         | `rollback_recurrent_state(req_id, n_reject)` — new method on adapters with recurrent state; plain-KV adapters do not implement it |

`n_reject = γ - accepted_len`; the spec-engine / batcher layer
performs the conversion before calling into I-2 / I-1. Plain-KV
adapters (Qwen3 0.6B, dense Gemma4 non-MoE) do not implement
`rollback_recurrent_state` at all; the spec-engine guards the call
with an explicit `has_recurrent_state(adapter)` check (matching the
existing `ModelCapabilities.has_recurrent_state` flag in
`silica/models/capabilities.py`) before dispatching. The §3 (b)
pseudocode shows the guard at the call site. Both Qwen3.5 dense 27B
(48 HYBRID_DELTANET + 16 GLOBAL per PLAN §13 v1.6.1) and MoE
35B-A3B (30 HYBRID_DELTANET + 10 GLOBAL) carry recurrent state and
require the non-trivial restore path on every partial reject; the
recurrent rollback test coverage on real hardware is therefore the
0.8B parity row plus the 27B and MoE acceptance rows.

### 4.2 `decode_step_multi(k)` adapter API

New method on `ModelAdapter` (sub-unit a2):

```
decode_step_multi(
    self, tokens: mx.array, kv_handle: KVHandle
) -> tuple[mx.array, StateDelta]
```

Same `KVHandle` + `StateDelta` shape as the existing `prefill` and
`decode_step` methods (`silica/models/adapter.py:172-178`). The
Protocol declares the signature; **no Protocol-level default
implementation** (Python's `runtime_checkable Protocol` does not
propagate defaults to structural conformers; see §3 sub-unit (a2)).
Adapters that don't yet implement the override use the
`silica.speculative.verify.run_verify_forward` free-function
fallback, which `hasattr`-dispatches and otherwise loops over
`decode_step` (correct but defeats verify amortisation). Existing
`decode_step()` is unchanged in semantics.

For dense models the override is straightforward — the existing
forward pass already accepts a `(B, T)` input shape; T was always 1
in pure decode but the layers handle T > 1 (prefill is just a longer
T). For hybrid Qwen3.5 the recurrent layers' `step` semantics differ
from `forward(T > 1)`; the adapter must route the verify forward
through the prefill-shaped path with the captured pre-draft snapshot
as the recurrent initial state. Full implementation contract per
adapter in §3 sub-unit (a2).

### 4.3 Greedy verify algorithm

For greedy decoding (temperature 0), verify is straightforward
(γ = k - 1 drafts proposed; verify forward output `verify_logits` has
length k; `verify_logits[i]` predicts the token at the position
following input slot `i`):

```
for i in range(gamma):
    target_top1 = argmax(verify_logits[i])
    if target_top1 != drafts.token_ids[i]:
        accepted_len = i
        break
else:
    accepted_len = gamma
```

The bonus token is sampled from `verify_logits[accepted_len]` on
partial reject (replacing the rejected draft at slot `accepted_len`)
or from `verify_logits[k - 1]` on full accept (the prediction past
the last accepted draft). Up-to-(γ + 1) = up-to-k yielded tokens for
one target forward of length k — that is the spec speedup mechanism.

Non-greedy verify (importance sampling against draft logprobs) is a
P-7 v0.2 feature. v0.1 step 5 is greedy-only; sampling temperature
> 0 with `--speculative draft_target` raises `NotImplementedError`.

### 4.4 Draft-model selection

**Canonical draft: `Qwen/Qwen3.5-0.8B`** (fp16; 4-bit MLX checkpoint
availability flagged as OQ-1). Rationale:

- **Tokenizer match.** Same Qwen3.5 family → vocabulary and chat
  template identical to both `mlx-community/Qwen3.5-27B-4bit` (dense)
  and `mlx-community/Qwen3.5-35B-A3B-4bit` (MoE) targets. Tokenizer
  divergence is the highest-impact correctness bug in spec decoding;
  same-family selection eliminates it by construction.
- **Architecture match.** Same hybrid DeltaNet structure as the
  dense 27B target. PLAN §13 v1.6.1 records the empirical 27B
  config: 64 layers = 48 HYBRID_DELTANET + 16 GLOBAL (3:1 D-D-D-G
  pattern); MoE 35B-A3B is 40 layers = 30 HYBRID_DELTANET + 10
  GLOBAL. Qwen3.5-0.8B shares the hybrid family at smaller scale
  (6 full-attention layers per `plans/P3_C5_OPENING.md`:292). So
  C.1 spec exercises the recurrent-state rollback path against
  real DeltaNet layers on every real-model row, not just synthetic
  harnesses.
- **Already in test fixtures.** `Qwen/Qwen3.5-0.8B` is referenced in
  `silica/bench/scenarios.py:296` for the P-3 hybrid B=1 parity row;
  no new model download infrastructure required for CI.
- **Size ratio.** 0.8B / 27B = 1/34. Standard spec heuristic
  (draft cost ≤ 5–10% of target verify cost) puts the 0.8B draft at
  the conservative end of "draft fast enough not to dominate";
  whether real measurement supports this is a step-5 finding.

**Fallback: `Qwen/Qwen3-0.6B`.** Plain KV (no recurrent path), so
loses the architecture-match property, but the smallest available
Qwen-family model with a guaranteed cached fixture for CI. Used in
the parity test (sub-unit f) where the target / draft are both 0.6B.
Not used as a draft for 27B real-model rows.

**Out of scope: cross-family drafts** (e.g. Llama-3-1B drafting for
Qwen3.5-27B). Tokenizer mismatch makes them correctness-fragile and
they offer no advantage over the same-family option.

### 4.5 v0.1 verify_k default

`verify_k = k = 4` (so γ = 3 actual draft proposals per cycle).
The convention is: k is the **target verify forward input length** —
identical to `verify_k` from `silica.bench.microbench.target_verify`,
so Unit 7's measured cost curve applies directly without
interpolation. Of the k input tokens, the first is the anchor (its
KV is filled by this verify) and the remaining γ = k - 1 are draft
proposals from the draft engine. Per-cycle yield is up to γ + 1 = k
tokens.

Justified by P-6.0.5 Unit 7: at k=4 the target-side bandwidth
utilisation has already dropped from 82.7% (k=1) to 55% on the
corrected 15.13 GB anchor, marking the regime transition between
k=2 and k=4. Beyond k=4 the cost curve flattens sharply (k=8 = 30%
util, 2.93× ceiling) — most of the verify amortisation gain is
harvested by k=4, with diminishing returns beyond. k=8 is too
aggressive for a baseline; the C.5 tree spike explores higher
effective k via tree shape, not linear.

Alternative conventions ruled out:

- **`k` as draft window with verify input = k+1 (Leviathan
  classic).** Would require c_verify(k+1) for the math; Unit 7
  measured k ∈ {1, 2, 4, 8} and a γ=4-draft choice would land at
  k+1=5 between two measured points, forcing interpolation. The
  benefit is conceptual familiarity (γ matches some literature
  notation), but the cost is direct disagreement with the bench
  data already on disk. Rejected.
- **`k` as draft window with verify input = k (no anchor in
  input).** Requires the bonus token's K/V to be filled by an extra
  single-token forward at the start of the next cycle (or
  equivalently to be re-fed at the next cycle's first input slot,
  reducing effective new drafts to k-1). Either complicates the
  loop or recovers Convention A under different labelling. Rejected
  for being awkward to specify.

---

## 5. Risks

### 5.1 Correctness risks

- **Off-by-one in verify alignment.** The drafted token at position i
  must be compared against `verify_logits[i]`, not `verify_logits[i+1]`.
  vLLM and reference SpecDec implementations have shipped this bug.
  Mitigation: the parity test (sub-unit f) catches any off-by-one;
  spec-on producing different output than spec-off under temperature
  0 fails the test loud.
- **Recurrent state desync on partial reject.** If the recurrent
  rollback restores to the wrong snapshot (e.g. the draft's snapshot
  instead of the target's), subsequent decode produces silently wrong
  tokens. Mitigation: dedicated recurrent rollback test (sub-unit i
  Pattern B) plus the parity test against a hybrid draft.
- **KV block-list shrink corner case.** If `accepted_len` falls
  exactly on a block boundary, the trailing block is released; if it
  falls in the middle, the block stays but logical count shrinks.
  Both paths exist in `PagedKVCache` already (preempt path uses
  them); the rollback case is new only in the dispatch shape.
- **Bonus token consistency.** vLLM's bonus-token rule samples from
  `verify_logits[accepted_len]` on partial reject (the rejected
  draft's slot) and from `verify_logits[k - 1]` on full accept (one
  past the last accepted draft). Other implementations use
  `verify_logits[accepted_len + 1]` for partial reject. For greedy
  decoding with temperature 0 the two differ by one position on
  partial reject. Step 5 follows the vLLM rule (sample at
  `accepted_len`); the parity test pins this choice.

### 5.2 Performance risks

- **Draft-model load increases peak memory.** Loading
  `Qwen/Qwen3.5-0.8B` fp16 costs ~1.6 GB on top of the 17 GB target
  (4-bit costs ~0.4 GB if checkpoint exists — OQ-1). At B=4 dense 27B
  the peak was 17.10 GB; +1.6 GB → 18.7 GB. §6(4) RAM gate (36 GB at
  4K) is unaffected at any feasible spec-on batch size, but the
  measurement should be recorded.
- **Batcher Option C.1 (pad to max k) wastes tokens.** A batch with
  one spec-off row and three spec-on rows runs the spec-off row at
  k=4 (4 tokens forward instead of 1 under our convention; γ=3
  drafts per spec-on row plus 1 anchor). This is the simpler path
  chosen in §3 sub-unit (c); a perf follow-up may switch to Option
  C.2.
- **Per-draft-cycle recurrent snapshot peaks memory.** Per the §3
  sub-unit (e) decision (per-draft-cycle, not per-step), peak
  memory adds one `RecurrentSnapshot` per active spec request. This
  is small for Qwen3.5 hybrid (recurrent state is a single
  `(N_layers, hidden)` per row), order ~MB.

### 5.3 MoE-specific risks

- **Tokenizer compatibility unverified.** `Qwen3.5-MoE-A3B` shares
  the Qwen3.5 dense tokenizer family — but this is hypothesised, not
  measured. Mitigation: parity test on the MoE bench scenario
  (sub-unit h) catches divergence loud. Discovered tokenizer mismatch
  is a step-5 blocker, not a sub-task.
- **Routing under verify forward.** MoE expert routing is per-token;
  k verify tokens may activate different experts than the
  autoregressive path's individual k forwards. For correctness this
  doesn't matter — the routing decision per token is a property of
  the input, not the kernel shape. For perf, expert-cache hit rate
  may differ. Step 5 reports; tightening is downstream.

---

## 6. Acceptance gates

### 6.1 Foundation gate (must pass for step 5 closure)

- **(a)** `silica/speculative/draft_target.py` lands; the module
  exports `DraftTargetEngine` conforming to the I-5 `DraftEngine`
  Protocol.
- **(b)** `silica.engine.Engine.generate` invokes
  `draft_engine.propose` / `commit` unconditionally; spec-off
  (`NoopDraftEngine`) is the no-op default, byte-equal to the
  pre-step-5 single-token decode path.
- **(c)** `ContinuousBatcher.step()` integrates per-row draft state
  with Option C.1 (§3) padding; spec-off rows under spec-on batch
  produce identical tokens to a fully-spec-off batch.
- **(d)** `KVManager.rollback(req_id, n_reject)` (existing I-2
  surface, `silica/kvcache/manager.py:94`) gets the concrete spec
  rollback behaviour on `PagedKVCache`; cached 0.6B real-model
  rollback row in sub-unit (i) exercises target-side KV rollback
  under partial accept (Qwen3-0.6B is plain KV per
  `silica/bench/scenarios.py:146`, so this row covers the KV path
  only — no recurrent state).
- **(e)** `adapter.rollback_recurrent_state` lands; the recurrent
  rollback path's only real-model exercise is the 0.8B parity row
  (sub-unit f) plus the 27B / MoE real-model acceptance rows. The
  synthetic patterns (A / B / C) in sub-unit (i) exercise the
  contract on a mock recurrent adapter.
- **(f) Greedy parity** — temperature 0, fixed seed,
  `--speculative draft_target` produces byte-equal token sequences
  vs `--speculative none` on:
  - `Qwen/Qwen3-0.6B` cached smoke (CI gate);
  - `Qwen/Qwen3.5-0.8B` standalone (CI gate);
  - `mlx-community/Qwen3.5-27B-4bit` (`SILICA_REAL_QWEN3_5_27B` real
    row; manual acceptance, recorded under
    `plans/P6_SPEC_FOUNDATION_BASELINE/`);
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (`SILICA_REAL_QWEN3_5_MOE_A3B`
    real row; manual acceptance).
- **(g) Spec-metrics schema** — every spec-enabled bench row populates
  all seven `silica.bench.spec_metrics` fields;
  `validate_speculative_metrics` returns empty list. Schema-mismatch
  rows fail loud at oracle exit.
- **(h)** `--speculative {none,draft_target}` flag works on
  `scripts/bench.py`; the two new scenarios (`spec-on` dense + MoE)
  are registered and runnable.
- **(i)** `tests/test_spec_rollback.py` passes (synthetic patterns
  A/B/C plus the cached 0.6B real-model rollback row).

### 6.2 Performance gate — tracked, **not** blocking

PLAN §7 P-7 acceptance lists "Decode tok/s ≥ 1.2× the draft-disabled
baseline … on a fixed standard scenario." Step 5's foundation gate
**does not include** ≥1.2×; the foundation closure only requires
correctness (parity) and metric-schema completeness.

Rationale: C.1 is the **baseline** every later C.x compares against
(PLAN §6 deliverable C.1). The whole point of Track C.4 / C.5 is
that C.1 alone may not clear ≥1.2× — small-model autoregressive
drafts on dense bandwidth-bound targets are not where the speedup
budget lives. P-6.0.5 Unit 7's 2.93× target-side ceiling at k=8
linear is the **upper bound** assuming zero drafter cost and
zero verify-cost amortisation overhead; real C.1 falls below by
both terms.

Quantitatively, the standard Leviathan-et-al. expected speedup with
greedy verify and bonus-token rule, under the convention pinned in
§4.5 (γ = k - 1 drafts per cycle; verify forward input length =
k; up to k tokens yielded per cycle):

```
speedup = E[yielded tokens per cycle] / per-cycle target-equiv cost
        = ((1 - α^k) / (1 - α)) / (gamma * c_draft + c_verify(k))
```

where α is the per-token accept probability, c_draft is the
single-token draft forward cost, and c_verify(k) is the k-token
target verify forward cost — both in units of one single-token
target forward. P-6.0.5 Unit 7 measured c_verify(k=4) directly:
89.01 ms vs 59.59 ms baseline = **1.494×** a single target forward.
Bandwidth utilisation drops from 82.7% at k=1 to ~55% at k=4 on
the corrected 15.13 GB anchor — cost grows sublinearly because
weight reads amortise across the verify's k input positions. With
α≈0.5, k=4 (γ=3), c_draft≈0.05 (0.8B / 27B draft-to-target forward
ratio), c_verify(k=4)≈1.49:

```
E[yielded per cycle] = (1 - 0.5^4) / 0.5  = 0.9375 / 0.5  = 1.875
denominator          = 3 * 0.05 + 1.49    = 1.64
speedup              ≈ 1.875 / 1.64       ≈ 1.143×
```

The `(1 - α^k)` numerator counts up-to-k yielded tokens per cycle
(γ = k-1 drafts that may be accepted plus 1 bonus). The denominator
counts the per-cycle target-equivalent compute: γ single-token
draft forwards plus one k-token target verify forward.

That is **below** the ≥1.2× gate by ~5%. α is the dominant lever:
α=0.6 lifts the prediction to ~1.33× (clearing the gate); α=0.4
drops it to ~0.99× (essentially no speedup). A real measurement
could land on either side of 1.2× depending on the 0.8B draft's
accept rate on 27B chat workloads — which is exactly what step 5
measures and exactly what makes a hard ≥1.2× gate at step-5-closure
inappropriate.

The ≥1.2× gate is therefore tracked at step 5 (recorded as
`integrated_speedup_vs_off` in `ScenarioResult.metadata`) but its
evaluation moves to Decision Gate 2 — alongside C.4's spike. If the
foundation **does** clear ≥1.2× on dense 27B, that is a positive
surprise narrowing the (1b) survival problem; if it does not (the
expected case), Track C.4 / C.5 is the path forward.

### 6.3 Toolchain attestation (at step 5 close)

- ruff clean (silica + tests + scripts);
- mypy clean (no new errors over v1.7.18 baseline);
- full non-real-model test suite passes (target: ≥2108 — was 2108 at
  v1.7.15; step 5 lands at minimum two new test files
  `test_spec_parity.py` and `test_spec_rollback.py` plus oracle /
  runner test additions);
- `python -m scripts.bench --list` enumerates **at least the v1.7.18
  catalog (63 scenarios as measured at gate-1 close) plus the two new
  `-spec-on` rows from sub-unit (h)** — i.e. ≥65 scenarios expected at
  step 5 close. Phrasing the gate against the v1.7.18 anchor rather
  than a hard-coded number avoids drift if any unrelated scenario
  lands in the same window.
- Real-model acceptance row produces a `plans/P6_SPEC_FOUNDATION_BASELINE/`
  directory mirroring `plans/P6_0_5_BASELINE/` with at minimum: dense
  27B spec-on / MoE spec-on warm-decode rows plus a brief REPORT.md
  recording integrated speedup, accept rate, draft cost ratio.

### 6.4 Memory accounting

Record the draft-model resident footprint in the spec-on bench row
metadata as `draft_resident_bytes`. Validate that
`peak_mb + draft_resident_bytes_mb` stays under 24 GB on the dense
27B B=1 spec-on row (well under the §6(4) 36 GB gate but
diagnostically useful as a check on the §5.2 ~1.6 GB estimate).

---

## 7. PLAN.md change list (preview; lands at step 5 close)

| # | Section                                  | Change                                                                                                                                              |
| - | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | §1 status header                         | Append v1.7.X line for D-021 step 5 closure.                                                                                                        |
| 2 | §6 deliverable C.1                       | Mark complete; cross-reference `plans/P6_SPEC_FOUNDATION_BASELINE/REPORT.md`.                                                                       |
| 3 | §7 D-021 step 5                          | Closure note (analogous to step 3 / step 4 closures): ten sub-units (a, a2, b..i) landed; foundation gate passes; performance gate deferred to Decision Gate 2. |
| 4 | §7 P-7 phase block                       | **OQ-2 below.** Status changes from `planned` to `complete-in-v0.1-scope` (deliverables 1–4 met by step 5); v0.2 candidates (EAGLE / Medusa / Mirror-SD / STree) listed explicitly as deferred. |
| 5 | §13 changelog                            | New v1.7.X entry: ten sub-units, parity gate, tracked-not-blocking ≥1.2×, three-rollback test, MoE compatibility verified, memory accounting.        |
| 6 | `silica/speculative/engine.py` docstring | Tighten "fixed from P-0" claim to "fixed from D-021 step 5"; remove the drift between contract and reality.                                          |

---

## 8. Open questions (carried forward to step 5 implementation)

- **OQ-1 — Qwen3.5-0.8B 4-bit MLX checkpoint availability.** The
  draft is currently fp16 (~1.6 GB). If `mlx-community/Qwen3.5-0.8B-4bit`
  exists, switching to 4-bit drops the draft footprint to ~0.4 GB and
  lowers the draft forward cost. Verify before sub-unit (a) lands;
  if no 4-bit, fp16 is acceptable for foundation closure (peak still
  well under §6(4) gate). **Action:** check `mlx-community` HF org at
  step start; record outcome in `plans/P6_SPEC_FOUNDATION_BASELINE/`
  README. Also at step start, verify `mlx-community/Qwen3.5-27B-4bit`
  layer composition matches the v1.6.1 record (48 HYBRID_DELTANET +
  16 GLOBAL) — if the checkpoint shape has drifted, the §4.4
  architecture-match rationale weakens and sub-unit (a2)'s
  Qwen3.5-specific recurrent path needs re-validation.

- **OQ-2 — P-7 phase block status wording.** PLAN §7 P-7 currently
  says `planned`. Step 5 closure moves the standalone P-7 deliverables
  (DraftTargetEngine + integration + metrics + bench switch) to
  `complete`, but the ≥1.2× decode acceptance bullet is not met by
  C.1 alone (see §6.2). Three options:
  - **(2a)** Move P-7 to `complete-in-v0.1-scope` and note that the
    ≥1.2× bullet is satisfied through Track C.4 / C.5 in P-6 rather
    than C.1 alone. v0.2 reopens P-7 for EAGLE / Medusa / etc.
  - **(2b)** Leave P-7 `in progress` until C.4 lands, then close.
  - **(2c)** Split P-7 into `P-7.1 foundation (complete at step 5)`
    and `P-7.2 ≥1.2× (closure with C.4)`.
  Recommendation: **(2a)** — cleanest given D-020 / D-021 already
  pulled DFlash and DDTree into P-6 Track C. The P-7 phase block
  exists as the v0.2 EAGLE / Medusa entry point; v0.1 closure of its
  acceptance is satisfied through the composite P-6 Track C.4 / C.5,
  not C.1 standalone.

- **OQ-3 — Per-step vs per-draft-cycle recurrent snapshot.** §3
  sub-unit (e) lands per-draft-cycle (simpler). If profiling shows
  partial-accept rollback is a hot path (≥10% of decode time), revisit
  with per-step capture. Tracked but not blocking step 5.

- **OQ-4 — Batcher Option C.1 (pad) vs Option C.2 (split forward).**
  §3 sub-unit (c) lands C.1 (pad). If batched spec-on workloads
  become a measured target (currently B=1 spec-on is the canonical
  scenario), revisit. Tracked but not blocking step 5.

---

## 9. Cross-references

- **PLAN.md §6** — I-5 `DraftEngine` interface contract; C.1 deliverable.
- **PLAN.md §7 P-6 D-021 step 5** — this opening's authoritative scope.
- **PLAN.md §7 P-7** — standalone phase block; step 5 closure settles
  v0.1 scope (OQ-2).
- **`plans/P6_OPENING.md`** — Track C parent definition; C.1 entry.
- **`plans/P6_0_5_BASELINE/REPORT.md`** — Decision Gate 1 input set;
  Unit 7 verify-k cost curve constrains step 5's `verify_k=4` choice
  (§4.5).
- **`plans/P6_0_DECISION_GATE_1_OPENING.md`** — Gate 1 closure;
  reframes (1b) as two-condition survival rule. Step 5 lights up the
  **C.1 baseline** that step 6 (C.4 spike) and step 8 (C.5 spike)
  measure their gate ratios against.
- **`silica/speculative/engine.py`** — Protocol + `NoopDraftEngine`
  stub; docstring drift fixed at step 5 (§7 row 6).
- **`silica/bench/spec_metrics.py`** — schema authored at v1.7.15;
  step 5 is first consumer.
- **`tests/test_qwen3_5_preempt_replay.py:48`** — flags P-7
  speculative draft-rollback wiring as an independent unit; step 5
  satisfies it via sub-unit (i).

---

## 10. Non-goals (for the avoidance of doubt)

- **Not a speedup result.** Step 5 closes on correctness (parity) and
  schema completeness, not on ≥1.2× decode. The expected C.1 result
  on dense 27B is **at or below 1.0× integrated** given drafter cost
  on bandwidth-bound targets; a positive surprise narrows the (1b)
  survival problem but is not budgeted for.
- **Not a tree-verify implementation.** Linear k=4 verify only.
- **Not a sampling-mode generalisation.** Greedy (temperature 0)
  only; non-greedy importance-sampling verify is v0.2.
- **Not an adaptive-k mechanism.** Fixed k per scenario.
- **Not a draft-model search.** `Qwen/Qwen3.5-0.8B` is the canonical
  draft for v0.1; alternative drafts (Qwen3.5-1.7B if it lands on
  mlx-community, MTP heads, distilled custom drafts) are
  step-8-and-beyond work.
- **Not a P-7 reopening.** Step 5 closure folds P-7's v0.1 scope into
  P-6 Track C cleanly; the standalone P-7 phase block becomes the
  v0.2 entry point for EAGLE / Medusa / Mirror-SD / STree per D-020.
