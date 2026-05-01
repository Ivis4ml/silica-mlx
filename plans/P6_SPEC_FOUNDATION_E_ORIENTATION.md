# P-6 Spec Foundation — Sub-unit (e) Orientation

| Field | Value |
| --- | --- |
| Status | round-1 review absorbed (replay sequence corrected, terminal path unified); awaiting sign-off |
| Last updated | 2026-04-30 |
| Slice scope | (e) — Qwen3.5 recurrent-state rollback + hybrid attention KV trim |
| Predecessors | (a) / (a2) / (b) / (c) sub-slices 1, 2a, 2b / (d) PagedKVCache.rollback |
| Successors | (c) slice 3 (hybrid + sliding batcher path) and (f) parity test |

This is the orientation pass for D-021 step 5 sub-unit (e). The
OPENING describes the recurrent-state side in full; this document
records the implicit prerequisite the OPENING does not call out
(`SimpleKVCache.rollback` is currently a no-op for hybrid caches),
the replay arithmetic the OPENING glosses, and the slice plan that
keeps each step independently reviewable.

---

## 1. OPENING contract recap

`plans/P6_SPEC_FOUNDATION_OPENING.md` §3 (e) commits to:

- **Capture** before `decode_step_multi(verify_input)`: adapter
  records a `RecurrentSnapshot` keyed by `request_id`.
- **Restore** after verify when `accepted_len < γ`: adapter restores
  the snapshot, then re-runs the recurrent path forward
  `1 + accepted_len` steps (anchor + accepted drafts).
- **Eviction**: drop the snapshot on full-accept commit and on
  request termination.
- **Granularity**: per-draft-cycle (not per-step). Correctness-first;
  the per-step optimisation is a perf follow-up.

The `Qwen3_5Adapter` already exposes the four helpers required
(`snapshot_pre_draft_state`, `commit_state`, `rollback_state`,
`free_state`); the dict `_pre_draft_snapshots` was installed in P5.9
step 2(c) but no consumer reads it. (e) wires `Engine.generate`
through those helpers.

---

## 2. Implicit scope extension — `SimpleKVCache.rollback` on hybrid

`silica/kvcache/simple.py:93-98`:

```python
def rollback(self, req_id: str, n_reject: int) -> None:
    self._require_owner(req_id)
    if n_reject <= 0:
        return
    if mlx_cache.can_trim_prompt_cache(self._cache_list):
        mlx_cache.trim_prompt_cache(self._cache_list, n_reject)
```

`mlx_cache.can_trim_prompt_cache` returns `all(c.is_trimmable() for c
in cache)`. Qwen3.5's per-layer list mixes 18 `ArraysCache` (DeltaNet
recurrent slots) with 6 `KVCache` (full-attention) entries.
`ArraysCache.is_trimmable()` returns `False`, so the guard short-
circuits and the **entire** rollback no-ops on hybrid — including
the trimmable `KVCache` layers. `tests/test_simple_kvcache.py:205`
pins this as the current contract.

The defensive pin made sense before (e): trimming attention KV while
recurrent state advanced through rejected drafts would have created
cross-layer position divergence with no remedy. (e) is exactly when
that defensiveness becomes obsolete: snapshot/restore handles
recurrent state explicitly, so attention KV must trim independently
to keep the I-2 surface contract honest. This aligns with OPENING
§4.1: "Target-side KV" rollback owner is `KVManager`, "Recurrent
state" rollback owner is the adapter — two decoupled paths.

**Decision**: `(e)` reframes `SimpleKVCache.rollback` to per-layer
trim. Trimmable caches (`KVCache`) are trimmed individually;
non-trimmable caches (`ArraysCache`) are passed over with no
mutation, leaving recurrent rollback to the adapter side. The pinned
test is rewritten to reflect the new partial-trim semantic.

[F-1] No external caller relies on the all-or-nothing no-op. The
only non-test call site of `kv_manager.rollback` for the target side
is `silica/engine/__init__.py:289`, which is exactly the consumer we
are fixing. Internal `silica/speculative/draft_target.py` rollbacks
operate on the **draft** model's KV manager, not the target.

---

## 3. Replay arithmetic

OPENING §3 (e) writes "re-runs the recurrent path forward exactly
`1 + accepted_len` steps". Read literally that suggests a recurrent-
only forward; no such method exists today (mlx-lm's hybrid forward
runs all layers in sequence, with DeltaNet `gated_delta_update` and
full-attention attention sharing the same call graph). The OPENING
is therefore stating the **intent** ("advance recurrent state by
`1 + accepted_len` positions"), not prescribing a recurrent-only
kernel. v0.1 takes the simplest correct realisation: replay the
committed prefix through the existing full forward
(`decode_step_multi`), with attention KV reset to pre-cycle so the
replay drives **both** layer types in lockstep over the committed
prefix.

[F-2] Replay uses `yielded_count`, not `accepted_len`, to stay
consistent with the engine's existing KV rollback formula at
`silica/engine/__init__.py:287` (`un_committed = draft_count -
yielded_count`). This matters when `max_tokens` or a stop-token cut
the yield short of the verifier's accept count — both rollback paths
must converge on the same committed boundary.

[F-3] Per-cycle KV / state arithmetic for partial reject
(`yielded_count < draft_count`) on a recurrent adapter:

```
Position L = pre-cycle KV / recurrent offset
verify_input length = draft_count + 1 (anchor + draft_count drafts)

Phase A (verify forward, pre-rollback):
  attn KV: [L, L + draft_count + 1)
  recurrent state advanced to L + draft_count + 1

Phase B (KV rollback by draft_count + 1 — drop ALL verify writes):
  attn KV: [L, L)   trimmable layers back to pre-cycle offset
  recurrent state: still at L + draft_count + 1

Phase C (recurrent restore via adapter.rollback_state):
  attn KV: unchanged (still at L)
  recurrent state: back to L (pre-cycle snapshot)

Phase D (replay decode_step_multi(verify_input[:1 + yielded_count])):
  attn KV: [L, L + 1 + yielded_count)   driven by replay forward
  recurrent state advanced to L + 1 + yielded_count

End state: both at L + 1 + yielded_count; replay observes the
correct attention context (only the committed prefix is in the
cache, no leftover verify-time writes).
```

[F-3a] **Why the full trim, not a partial one.** An earlier draft of
this orientation proposed trimming attention KV by `un_committed`
only, leaving the committed-prefix KV in place, then replaying
`1 + yielded_count` tokens, then re-trimming by `1 + yielded_count`
to undo replay's KV writes. That sequence pollutes the replay:
`decode_step_multi` is a full-model forward, so when the replay
processes the anchor, the surviving committed prefix at
`[L, L + 1 + yielded_count)` is visible to global-attention layers,
making each replay token attend to the wrong context window. The
hidden states fed into DeltaNet's `gated_delta_update` therefore
diverge from the original verify forward's, so the recurrent state
advance during replay disagrees with the verify-forward path. The
correct sequence trims **all** verify-forward attention writes
before replay so the attention context window matches the
pre-cycle position exactly.

[F-3b] Phase B is only meaningful for adapters whose KV manager
exposes per-layer trim (slice 1 prerequisite). Phase D's replay
uses the engine's existing `verify_input` slice — no new buffer.
Phase C is the sole new adapter call beyond the existing snapshot
capture.

[F-4] Non-recurrent adapters (`Qwen3Adapter`, `Gemma4Adapter`,
`Gemma4MoeAdapter`): the existing engine path already converges on
the committed boundary via `kv_manager.rollback(req_id,
un_committed)` — KV trims by `un_committed`, no replay needed
(there is no recurrent state to advance). Behaviour is identical
to today.

[F-5] Full-accept path (`yielded_count == draft_count`,
`un_committed == 0`): no rollback, no replay. The snapshot is
dropped via `commit_state(req_id, yielded_count)`. The verify
forward has already advanced both attention KV and recurrent state
through all committed positions.

[F-6] Terminal-cycle uniformity. The engine takes the same recurrent
rollback path for every cycle with `un_committed > 0`, including
the cycle that triggered `stop_hit` mid-yield or the `max_tokens`
cut. Reasons:

- **Correctness invariant**: at the end of every cycle, recurrent
  state is at `L + 1 + yielded_count`. Skipping replay on stop_hit
  would leave recurrent state at `L + draft_count + 1`, breaking
  the invariant — and the next call into `Engine.generate` for the
  same request (chat-style, not implemented today but plausible)
  would resume from a stale state.
- **Test simplicity**: one path, not two. The synthetic adapter's
  invocation log is uniform.
- **Cost**: one extra full forward over `1 + yielded_count` tokens
  on stop_hit. Negligible relative to the prefill / decode cost
  the request already paid.

`free_state(req_id)` runs in the engine's cleanup tail regardless,
idempotent under `_pre_draft_snapshots.pop(..., None)`.

---

## 4. Capability detection — Protocol vs duck typing

The four helpers (`snapshot_pre_draft_state`, `commit_state`,
`rollback_state`, `free_state`) live on `Qwen3_5Adapter` and are
inherited by `Qwen3_5MoeAdapter` (`silica/models/qwen3_5_moe.py`
defines `class Qwen3_5MoeAdapter(Qwen3_5Adapter)`). `Qwen3Adapter`
and `Gemma4Adapter` / `Gemma4MoeAdapter` (no recurrent state) do
not implement them. Two options for engine dispatch:

- **(a)** Add a `runtime_checkable` Protocol mixin
  `SpecRecurrentRollbackAdapter` to `silica/models/recurrent.py`
  alongside the existing `RecurrentStateAdapter`.
- **(b)** Duck-type via `getattr(adapter, '...', None)` at each
  call site.

[F-7] (e) takes (a). Mirrors the existing `RecurrentStateAdapter`
shape (P-3-C5.1) so engine and batcher dispatch through
`isinstance` rather than scattered `hasattr` checks. The mixin is
optional — adapters without recurrent state remain shape-compliant
without implementing it.

---

## 5. Engine vs Batcher matrix

`Engine.generate` is the single-request spec entry point (B=1).
`ContinuousBatcher._decode_phase_spec` is the multi-request spec
entry point (B>=1). Sub-unit (e)'s wiring lands in `Engine.generate`
only:

| Path | Spec-on adapters supported |
| --- | --- |
| `Engine.generate` (B=1) | GLOBAL + HYBRID_DELTANET (this slice unblocks hybrid) |
| `ContinuousBatcher._decode_phase_spec` (B>=1) | GLOBAL only |

[F-8] The batcher's GLOBAL-only gate at `silica/scheduler/batcher.py:268-279`
stays in place. (c) slice 3 — multi-request hybrid — depends on (e)
landing the adapter snapshot/restore wiring AND on a per-row
recurrent rollback path that walks the batched cache row-by-row.
(e)'s engine wiring uses `row_idx=0` (single-request) and does not
attempt per-row dispatch; that is (c) slice 3 territory.

---

## 6. Slice decomposition

### Slice 1 — `SimpleKVCache.rollback` per-layer trim

Files:
- `silica/kvcache/simple.py` — replace the all-or-nothing
  `can_trim_prompt_cache` guard with a per-cache `is_trimmable()`
  walk. Trimmable caches receive `c.trim(n_reject)`; non-trimmable
  caches are skipped.
- `tests/test_simple_kvcache.py` — rewrite
  `test_rollback_when_not_trimmable_is_noop` as
  `test_rollback_trims_only_trimmable_entries_in_hybrid_list`
  (asserts `KVCache.offset` shrinks while `ArraysCache` is
  untouched). Adds a sanity test for the all-recurrent case (no
  KVCache present → entire call is a no-op).

No call-site changes; no engine wiring. This slice can land
independently.

### Slice 2 — Engine.generate recurrent rollback wiring

Files:
- `silica/models/recurrent.py` — add `SpecRecurrentRollbackAdapter`
  Protocol mixin (runtime_checkable) covering the four helpers.
  `__all__` updated.
- `silica/engine/__init__.py` — `Engine.generate` spec path:
  - Before `run_verify_forward` (line ~249): if adapter satisfies
    `SpecRecurrentRollbackAdapter`, call
    `snapshot_pre_draft_state(req_id)`.
  - On `un_committed > 0` (line ~287), branch on the mixin:
    - **Recurrent path** (mixin satisfied): trim attention KV by
      `draft_count + 1` (drop the entire verify forward), call
      `rollback_state(req_id, un_committed)`, then
      `decode_step_multi(verify_input[:1 + yielded_count], handle)`
      to drive both attention KV and recurrent state forward
      through the committed prefix in lockstep. No second trim.
    - **Non-recurrent path**: existing
      `kv_manager.rollback(req_id, un_committed)` only.
  - On `un_committed == 0`: if adapter satisfies the mixin, call
    `commit_state(req_id, yielded_count)` (idempotent regardless of
    stop_hit).
  - In the `try/finally` cleanup tail: `free_state(req_id)` if the
    mixin is satisfied. Idempotent under the existing
    `_pre_draft_snapshots.pop(..., None)` semantics.
- `tests/test_engine_spec_decode.py` — new tests via a synthetic
  `_RecurrentScriptedAdapter` that records snapshot / restore /
  commit / free invocation order, the replay token slice, and
  the rollback `n_reject` arguments. Cover: full accept (commit
  only, no replay), partial reject (snapshot → trim by
  `draft_count + 1` → restore → replay over `1 + yielded_count`,
  no second trim), stop-hit mid-yield (same recurrent path runs;
  `free_state` runs on cleanup), max-tokens cut (replay slice
  driven by `yielded_count`, not `accepted_len`), missing
  recurrent mixin (no extra calls, attention KV path only).

### Slice 3 — deferred

No third slice in (e). The cached-Qwen3.5-0.8B real-model
spec-rollback test belongs to sub-unit (f) (greedy parity). The
multi-request hybrid path belongs to (c) slice 3. The bench scenario
that exercises real-model recurrent rollback at decode scale belongs
to (h).

---

## 7. Decisions log

| ID | Decision | Reason |
| --- | --- | --- |
| F-1 | Flip `SimpleKVCache.rollback` to per-layer trim | All-or-nothing guard blocks hybrid attention trim; only consumer is the engine spec path we are wiring |
| F-2 | Replay uses `yielded_count`, not `accepted_len` | Match existing KV rollback formula at `engine/__init__.py:287`; converge both rollback paths on the same committed boundary |
| F-3 | Full-trim → restore → replay sequence (no re-trim) | Replay is full-model forward; surviving committed-prefix KV would corrupt attention context for replay tokens. Trim all `draft_count + 1` verify-forward writes, then replay drives both KV and recurrent forward in lockstep |
| F-4 | Non-recurrent adapters keep existing partial trim | `un_committed`-only trim already converges attention to committed boundary; no replay needed |
| F-5 | Drop snapshot on full accept | `commit_state` evicts; verify forward already at correct position |
| F-6 | Recurrent rollback runs on every cycle with `un_committed > 0`, including stop_hit / max-tokens | Uniform invariant ("end-of-cycle state at `L + 1 + yielded_count`") and one test path. Negligible cost vs request-level prefill / decode |
| F-7 | New `SpecRecurrentRollbackAdapter` Protocol | Mirror existing `RecurrentStateAdapter` shape; isinstance dispatch over scattered hasattr |
| F-8 | Engine wiring only — batcher gate stays | (c) slice 3 needs per-row dispatch which is out of (e) scope |

---

## 8. Byte-identical surface (spec-off)

No spec-off behaviour changes. The four code-path additions all gate
on either `self._draft_engine` not being a `NoopDraftEngine` or
`isinstance(adapter, SpecRecurrentRollbackAdapter)`. With both
checks satisfied (the only path the new wiring is reachable), spec
is by definition active.

The `SimpleKVCache.rollback` semantic change is observable on
hybrid lists when `n_reject > 0`. This is only reachable through
the engine spec path, which today is dormant on hybrid (no consumer
exercises it). v0.2 callers that may eventually rollback hybrid
without recurrent-state cooperation should consult the new
docstring; the new partial-trim semantic remains the I-2-correct
behaviour regardless of consumer.

---

## 9. PLAN.md change list (preview; lands at full step 5 closure)

Step 5 closure already aggregates ten sub-units' rows under
`plans/P6_SPEC_FOUNDATION_OPENING.md` §7. (e) does not require its
own PLAN.md row beyond that closure entry.

---

## 10. Cross-references

- `plans/P6_SPEC_FOUNDATION_OPENING.md` §3 (e) — authoritative scope
- `plans/P6_SPEC_FOUNDATION_C_ORIENTATION.md` — sibling pattern for
  slice decomposition + decisions log
- `silica/models/qwen3_5.py:290-387` — adapter helpers awaiting consumer
- `silica/models/recurrent.py` — `RecurrentStateAdapter` Protocol pattern
- `silica/engine/__init__.py:218-321` — engine spec cycle insertion points
- `silica/scheduler/batcher.py:268-279` — GLOBAL-only gate (stays)
