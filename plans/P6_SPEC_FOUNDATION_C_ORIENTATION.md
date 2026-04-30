# P-6 Spec Foundation — Sub-unit (c) Orientation

| Field | Value |
| --- | --- |
| Status | round-2 decisions locked; ready for slice-1 implementation |
| Last updated | 2026-04-30 |
| Slice scope | (c) sub-slice 1 — surface area + B=1 spec-on synthetic; no real model |
| Predecessors | (a) DraftTargetEngine + (a2) decode_step_multi + (b) Engine spec |
| Successors | (c) sub-slice 2a — per-`req_id` drafter refactor; 2b — multi-row batcher wiring |

This is a read-only orientation pass over `silica/scheduler/batcher.py`
to plan sub-unit (c) of D-021 step 5 — multi-request batcher
integration. No code changes are proposed here. Round-2 incorporates
the user's review findings on ctx synchronisation, shared-drafter
scope, rollback formula coherence with sub-unit (b), and the
capability gate for the right-trim primitive.

---

## 1. ContinuousBatcher.step() phase map (incumbent)

`silica/scheduler/batcher.py:433` is the `step()` entry point and
walks five phases. All five must remain reachable on the spec-off
path with byte-identical behaviour after sub-unit (c) lands.

| # | Phase | Trigger | Implementation |
| --- | --- | --- | --- |
| 0 | `_prepare_cohort` | First `step()` call before cohort sealed | `batcher.py:483` |
| 1 | `_reclaim_terminated` | Top of every `step()` after cohort prep | `batcher.py:611` |
| 2 | `_admit_waiting_requests` | Non-empty waiting queue + headroom in `_rows` | `batcher.py:1665` |
| 3 | `_prefill_phase` | Phase 2 empty AND any row PREFILL | `batcher.py:1490` |
| 3 | `_decode_phase` | Phase 2 empty AND no row PREFILL | `batcher.py:1562` |
| 4 | `_sample_and_emit_*` | Tail of either Phase-3 forward | `batcher.py:1608` / `:1624` |

**Spec-on attaches to Phase 3 decode only.** Prefill, admit, reclaim,
and sample-and-emit do not change shape. The verify forward replaces
the existing `(B, 1)` decode-batched call with a `(B, k)` verify-batched
call, and the per-row sample loop expands from "yield 1 token" to
"yield up-to-k tokens with rollback".

The two phase paths that must NOT shift under spec-off:

- **Phase 2 admit** — `_admit_single_hit_row` / `_admit_miss_cohort`
  classification keys on prefix-cache `peek`. Spec is irrelevant
  here. Slice-1 contract: per-row spec fields populate only after
  the row enters DECODE, never at admission.
- **Phase 3 prefill** — prefill stays `(B, T_max)` per the existing
  left-padding arithmetic. Spec wiring branches inside
  `_decode_phase` only.

---

## 2. `_BatchRow` field inventory (incumbent)

`silica/scheduler/batcher.py:157`. Live fields:

| Field | Owner / lifetime |
| --- | --- |
| `req_index: int` | Identity; emitted on every `BatchEvent` |
| `req_id: str` | KV-handle key — `f"req-{req_index}"` |
| `prompt_ids: list[int]` | Prompt; size used by left-padding |
| `params: SamplingParams` | Per-row sampling params (P-2 homogeneous) |
| `state: RequestState` | State machine (PREFILL / DECODE / DONE / ABORTED / PREEMPTED) |
| `generated: list[int]` | Sampled tokens; consumed by `_build_decode_tokens` |
| `recurrent_snapshots_per_block` | P-3-C5.3.1 hybrid prefix snapshots |
| `absolute_consumed_tokens: int` | P-3-C5.3.1 lockstep counter for slice-prefill snapshot capture |
| `k_pre_per_block` | P-5-F pre-norm K capture buffer for the (3b) prefix store |

Sampler "history" is **not** a row field — `_sample_and_emit_rows`
reconstructs it as `prompt_ids + generated` on each call
(`batcher.py:1643`). The KV handle is implicit via `req_id`; the
per-layer batched cache lives on `self._batch_cache: list[Any]`,
indexed by row position. The slot table `self._slot_table` maps
`req_index → row_idx`.

Crucially for (c): **`row.state.output_token_ids` is not maintained
by the batcher today.** `Engine._drive` keeps it in lockstep with
`history` (`silica/engine/__init__.py:196-197`, `:217`, `:268`,
`:315`); the batcher does not. This matters for ctx
synchronisation — see §3 finding [F-1].

---

## 3. Spec state — `_BatchRow` additions and ctx synchronisation

To wire `(B, k)` verify forward + per-row variable accept, three
new fields are minimally sufficient. All three live only on
spec-active rows; spec-off rows leave them at default-empty.

| Field | Lifetime |
| --- | --- |
| `pending_anchor: int \| None` | Set after the first sampled token; carried as the cycle's anchor; updated on each verify cycle's bonus token |
| `pending_drafts: tuple[int, ...]` | Filled by `propose(ctx, γ)` at the start of each cycle; cleared after verify + rollback |
| `last_propose_count: int` | `len(pending_drafts)`; mirrors `DraftTargetEngine`'s same-named bookkeeping (`silica/speculative/draft_target.py:75`) so the two layers diff-clean against one another in tests |

### [F-1] ctx synchronisation contract — round-2 fix

Round-1 elided this and the user flagged it. `DraftTargetEngine.propose`
reads `ctx.request.token_ids + ctx.output_token_ids`
(`silica/speculative/draft_target.py:118-120`). The batcher today
does not maintain `row.state.output_token_ids` — it maintains
`row.generated` only. If `_decode_phase_spec` passes `row.state` raw,
the cycle-0 drafter sees only the prompt without the anchor, and
its `_draft_kv_pos` lands one position too low from the first cycle.

**Slice-1 contract (decided per O-5: transient ctx).** Each cycle,
`_decode_phase_spec` builds a fresh, throwaway `RequestState` and
hands it to `propose`. The persistent `row.state` is NOT mutated:

```python
# Each spec cycle, before propose():
ctx = RequestState(
    request=Request(
        prompt="",
        sampling_params=row.params,
        request_id=row.req_id,
        token_ids=tuple(row.prompt_ids),
    ),
)
ctx.output_token_ids = list(row.generated)
```

Why transient over in-place mutation of `row.state`: avoids
introducing an implicit "spec path half-maintains
`row.state.output_token_ids`" contract that the spec-off path does
not honour. The batcher's `row.state` continues to track only what
the existing scheduler logic needs (PREFILL / DECODE / DONE / etc.);
spec-only history reconstruction stays inside the spec helper.
`row.generated` remains the single source of truth for emitted
tokens; the transient ctx is just the per-cycle reformatting the
drafter Protocol expects.

`request_id=row.req_id` matters for slice 2a's per-`req_id`
drafter — once `DraftTargetEngine.propose` keys its bookkeeping
on `ctx.request.request_id`, slice 2b's batcher gets correct
multi-row dispatch with no further ctx-shape change.

**Test pins (slice-1 must include).**

- Cycle 0 ctx assertion: a scripted drafter records its received
  `ctx.request.token_ids` and `ctx.output_token_ids`; the test
  asserts `prompt_ids + [anchor_token]` at cycle 0, not just
  `prompt_ids`.
- `row.state.output_token_ids` non-mutation pin: assert that
  `row.state.output_token_ids` stays at its construction-time
  value (today: `[]`) across an entire spec-on `step()` sequence,
  proving the spec helper does not silently grow `RequestState`
  state.

### [F-2] Drafter ownership — round-2 fix

Round-1 claimed "one shared `DraftEngine` per `ContinuousBatcher`,
with `DraftTargetEngine` extended to key bookkeeping by `req_id`."
The user correctly observed that this is not a self-contained slice-1
change: `DraftTargetEngine` today carries a single `_req_id`,
`_handle`, `_reserved`, `_draft_kv_pos`, `_last_propose_count`,
`_cached_last_logits` (`silica/speculative/draft_target.py:63-90`).
Multi-row support requires keying ALL of those by `req_id`, plus
per-request `reserve_for_prefill / free / reset` lifecycle, plus
careful A→B→A switching tests.

**Slice-1 scope (revised).** Slice 1 is **B=1 spec-on only** and
makes NO shared-multi-row drafter claim:

- One `DraftEngine` instance per `ContinuousBatcher`, just like
  `Engine`.
- The batcher rejects spec-active cohorts with more than one
  spec-on row at runtime, raising `NotImplementedError` with an
  explicit slice-2 marker in the message.
- `DraftTargetEngine` is unchanged at the API surface; its single-
  request bookkeeping is what slice 1 exercises.

**Slice-2 prerequisite (separate sub-slice, NOT bundled with slice
1).** Refactor `DraftTargetEngine` so every per-request field
becomes a per-`req_id` dict:

| Field today | Slice-2 shape |
| --- | --- |
| `self._req_id: str` | `self._handles: dict[str, KVHandle]` — populated in `propose`'s cycle-0 branch when a new `req_id` first appears |
| `self._handle: KVHandle` | (folded into `_handles`) |
| `self._reserved: bool` | `self._reserved: set[str]` |
| `self._draft_kv_pos: int` | `self._draft_kv_pos: dict[str, int]` |
| `self._last_propose_count: int` | `self._last_propose_count: dict[str, int]` |
| `self._cached_last_logits: mx.array \| None` | `self._cached_last_logits: dict[str, mx.array]` |

Plus `propose(ctx, k)` deriving `req_id` from `ctx.request.request_id`
on entry, and `commit / reset` taking the same key. `reset` becomes
a per-`req_id` operation; a separate `reset_all` could exist for
batcher-level teardown if needed.

**Why not bundle this into slice 1?** Two reasons:

1. The refactor's correctness is independent of the batcher
   integration — it can be validated in isolation with A→B→A
   switching tests on `DraftTargetEngine` alone before slice 2's
   batcher wiring lands. Bundling them couples two different
   regression surfaces.
2. Slice 1 is meant to de-risk the verify-forward shape and the
   spec-off byte-identical guarantee. The drafter refactor adds
   diff bulk that distracts from those two guarantees.

`ContinuousBatcher.__init__` gains:

```python
draft_engine: DraftEngine | None = None,    # default Noop
verify_k: int = 4,                          # validates >= 1 like Engine
```

mirroring `Engine.__init__` (`silica/engine/__init__.py:69-89`) so
`Engine.generate_batch` can pass them through unchanged.

---

## 4. Option C.1 padding under the current cache surface

### [F-3] Architectural finding — `BatchKVCache` per-row right-trim

mlx-lm's `BatchKVCache` already supplies the per-row variable-trim
primitive Option C.1 needs. The OPENING §3 (c) discussion treats
this as "correctness-clean but wasteful" — that wording is correct,
and the cleanness rests on this primitive that the OPENING does not
call out:

`BatchKVCache.prepare(right_padding=[n_i, ...])` followed by
`BatchKVCache.finalize()` performs a per-row right-trim:

- `prepare(right_padding=...)` registers per-row `_right_padding`.
- `finalize()` applies `dynamic_roll(keys, padding[:, None], axis=2)`
  per-row, then `offset -= padding` (per-row) and
  `left_padding += padding` (per-row).
- Net effect: row `i`'s effective valid range shifts from
  `[left_padding_old_i, _idx)` to
  `[left_padding_old_i + n_i, _idx)`. The trimmed `n_i` positions
  are masked out via the row's left-padding bump, not physically
  deleted.

Empirically verified `2026-04-30` against
`mlx_lm/models/cache.py:880-1097`:

```text
B=2 cache, both rows fed 6 K/V positions uniformly.
prepare(right_padding=[0, 3]) + finalize():
  offset=[6,3], left_padding=[0,3], _idx=6
  row 0 valid range [0, 6) — 6 positions kept
  row 1 valid range [3, 6) — 3 positions kept (3 rolled to mask)
```

### [F-4] Capability gate — round-2 fix

The user correctly observed that the right-trim primitive lives on
`BatchKVCache` only. The batcher today admits hybrid (DeltaNet
`ArraysCache`) and sliding (`BatchRotatingKVCache`) caches via
`adapter.make_batch_cache(left_padding)` (`batcher.py:290-294`).
The two non-`BatchKVCache` paths have shape problems:

- **`ArraysCache`** (DeltaNet recurrent) — `prepare(lengths=...)`
  only; no `right_padding` parameter
  (`mlx_lm/models/cache.py:647-648`). The recurrent state is a
  snapshot, not a spatial K/V; per-row trim needs
  `adapter.rollback_recurrent_state(req_id, n_reject)` from
  sub-unit (e), not `prepare(right_padding=...)`.
- **`BatchRotatingKVCache`** (sliding) — `prepare(right_padding=...)`
  exists but couples to `lengths` and rotation state
  (`mlx_lm/models/cache.py:1239`). Audit deferred — slice 1 / 2
  do not touch this path.

**Spec-active capability gate (proposal).** At construction time
when `draft_engine` is non-None and not `NoopDraftEngine`:

```python
caps = adapter.capabilities()
if caps.attention_kinds != {AttentionKind.GLOBAL}:
    raise NotImplementedError(
        "ContinuousBatcher spec-active path supports GLOBAL-only "
        "adapters in slice 1; HYBRID_DELTANET / SLIDING land in "
        "later slices once sub-unit (e) recurrent rollback and a "
        "BatchRotatingKVCache audit are in place."
    )
```

This is a stricter subset of `_enforce_capability_gate` (which admits
`{GLOBAL, HYBRID_DELTANET, SLIDING}`) — only fires when the
constructor receives a real draft engine. Spec-off batchers stay
admissible across the full capability set unchanged.

### Per-row trim arithmetic — coherent with sub-unit (b) [F-5]

Round-1 wrote `n_reject = γ - accept_len`. The user pointed out
that this regresses the sub-unit (b) fix (`silica/engine/__init__.py:283`):
rollback must key on `yielded_count`, not `accept_len`, because
`max_tokens` / stop-token / fewer-than-γ paths can leave
`yielded_count < accept_len`. The OPENING math under the new
convention also has to account for the (k-1)-pad slots that
spec-off / drafter-declined rows consume.

**Per-row right_padding formula (revised).** Let
`verify_width = verify_k = γ + 1 = k`. After a `(B, k)` verify
forward, every row's cache offset has advanced by `k`. Per row `i`:

- `draft_count_i = len(pending_drafts_i)` — actual count returned
  by `propose`. Spec-off / declined rows: `draft_count_i = 0`.
  Drafter-violating rows (`> γ`): caught at `propose` call site
  with `RuntimeError`, mirroring `silica/engine/__init__.py:230-237`.
- `accept_len_i = greedy_verify(drafts_i, verify_logits[i, :draft_count_i + 1])`
  — `_greedy_verify` reused from
  `silica/engine/__init__.py:617-643` for diff-clean parity.
- `yielded_count_i ∈ [0, accept_len_i]` — accepted drafts the row
  actually emitted before `max_tokens` / stop-token cut the stream.
- `bonus_idx_i = yielded_count_i if yielded_count_i < draft_count_i
  else draft_count_i` — the same expression as
  `silica/engine/__init__.py:301-305`, modulo the rename
  (`int(verify_input.size) - 1` becomes `draft_count_i` here
  because the per-row verify input within the padded `(B, k)`
  forward is `[anchor] + drafts_i + pad`; the bonus prediction at
  full accept lives at draft-final position, not at the trailing
  pad).
- `right_padding_i = (verify_width - 1) - yielded_count_i`. This
  unifies three cases:
  - Spec-off / declined (`draft_count = 0`, `yielded = 0`):
    `right_padding = k - 1` — correct, all pad slots roll out.
  - Spec-on full accept (`yielded = draft_count = γ`):
    `right_padding = (k-1) - γ = 0` — no rollback.
  - Spec-on partial / mid-stop (`yielded < draft_count ≤ γ`):
    `right_padding = (k-1) - yielded ≥ 1` — trims rejected drafts
    plus any stop-truncated tail.

The draft engine sees `commit(ctx, yielded_count_i)` (matching
`silica/engine/__init__.py:290`). KV trim is one
`cache.prepare(right_padding=per_row_list) + cache.finalize()` call
per layer — `max(per_row_list) > 0` short-circuits the no-op case.

This is the formula slice 1 must implement; slice-1 tests pin the
three cases above plus fewer-than-γ and stop / max-tokens
mid-yield edge cases (mirroring sub-unit (b)'s round-1 regression
test set).

### Verdict

With [F-3] confirmed and [F-4] gated, Option C.1 is the
correctness-clean path for sub-unit (c). Option C.2 (split-forward)
adds a Metal kernel launch per step without narrowing the per-row
trim challenge (which exists inside the spec-on cohort regardless).
Slice 1 lands C.1 under the GLOBAL-only gate.

---

## 5. Slice-1 scope (synthetic only; no real model)

Slice 1 is the first of an expected **four-slice** (c) sequence
(round-1 said three; round-2 splits drafter refactor out per [F-2]):

| Slice | Scope |
| --- | --- |
| 1 | Surface area + spec-off byte-identical guard + B=1 spec-on synthetic happy path; GLOBAL-only gate. No drafter refactor. |
| 2a | `DraftTargetEngine` per-`req_id` refactor (standalone; A→B→A switching tests; `Engine.generate` regression-clean) |
| 2b | B>1 spec-on cohort + per-row right-padding rollback + drafter dispatched per-row by `req_id` |
| 3 | Hybrid (DeltaNet) + Sliding cache primitives — gated on (e) landing |

### Slice-1 deliverables

Per O-4 sign-off, the right-trim primitive gets its own isolated
test landed **before** any batcher patch consumes it.
Sub-deliverable 0 below is that pin.

0. **Pre-slice — isolated `BatchKVCache` right-trim pin.**
   `tests/test_batch_kv_cache_right_trim.py` lands as a standalone
   commit before any batcher patch. Pins:
   - Two-row roundtrip: feed K=6 positions uniformly,
     `prepare(right_padding=[0, 3]) + finalize()` ⇒
     `offset == [6, 3]`, `left_padding == [0, 3]`, `_idx == 6`,
     row 1's tokens at axis-2 `[0, 3)` are mask positions and
     `[3, 6)` carry the original first-three tokens of row 1
     (the dynamic-roll signature).
   - All-zero right-padding short-circuit: `prepare(right_padding=
     [0, 0]) + finalize()` is a no-op (`_right_padding` stays None
     past `prepare` because `max(...) == 0`).
   - Subsequent `update_and_fetch` from the trimmed state advances
     the per-row `offset` correctly (so the spec helper's "verify
     forward, trim, next-step decode" sequence is sound).
   This test is the load-bearing external contract. If a future
   mlx-lm change quietly alters trim semantics, this pin fails
   before any spec-active path reaches its tests.

1. `ContinuousBatcher.__init__` accepts `draft_engine: DraftEngine |
   None = None` and `verify_k: int = 4`. Defaults preserve
   byte-identical spec-off. `verify_k < 1` raises at construction.
   Spec-active GLOBAL-only gate ([F-4], O-6 hard) fires here.
2. `_BatchRow` gains `pending_anchor`, `pending_drafts`,
   `last_propose_count` with default-empty values. **No
   `output_token_ids` on `row.state`** — see [F-1] / O-5 transient
   ctx decision.
3. `_decode_phase` branches on `_spec_active()` (helper returning
   False when `draft_engine` is None / Noop). Spec-off path
   reaches the existing `(B, 1)` code unchanged; spec-on enters
   the new `_decode_phase_spec` helper.
4. `_decode_phase_spec` (slice 1 scope) handles **B=1 spec-on**:
   - Multi-row spec-active in the same cohort raises
     `NotImplementedError("D-021 (c) slice 2b — multi-row spec
     cohort")`.
   - Per cycle: (i) build a transient `RequestState` per [F-1] /
     O-5 (`request_id=row.req_id`, `token_ids=tuple(row.prompt_ids)`,
     `output_token_ids=list(row.generated)`); (ii) `propose(ctx, γ)`
     on the spec-on row, validate `draft_count ≤ γ` and raise on
     overflow; (iii) build `verify_input = [anchor] + drafts + pad`
     length k; (iv) one
     `forward_batched_full(model, verify_input[None], cache)` →
     `(1, k, V)`; (v) per-row greedy verify and yield with the
     `yielded_count` bookkeeping from
     `silica/engine/__init__.py:257-272`; (vi) cache
     `prepare(right_padding=[(k-1) - yielded_count]) + finalize()`
     ([F-5]); (vii) `draft_engine.commit(ctx, yielded_count)` and
     bonus sample.
5. New `_decode_phase_spec` lives next to `_decode_phase` in
   `batcher.py` (per O-3) — keeps cache + adapter access on one
   object.
6. Test files (synthetic only, no cached / real-model):
   - `tests/test_batcher_spec_decode.py` — B=1 spec-on contract
     pinning [F-1] (cycle-0 transient ctx), [F-3] right-trim,
     [F-5] formula across full-accept / partial / full-reject /
     fewer-than-γ / stop-token mid-yield / max_tokens mid-yield.
     Plus the [F-1] non-mutation pin: `row.state.output_token_ids`
     stays at construction-time value (today: `[]`) across the
     entire spec-on `step()` sequence. Spec-off byte-identical
     regression on a scripted adapter. `verify_k < 1` raises.
     Multi-row spec-active raises with the slice-2b marker.
     GLOBAL-only gate raises for HYBRID_DELTANET / SLIDING
     capability sets.
   - All existing P-2 / P-3 / P-4.5 fixtures stay green
     (regression net — see §6).

### Out of slice 1 (deferred)

- **Drafter refactor to per-`req_id` keying** ([F-2]) — slice 2a
  prerequisite, not bundled.
- **B>1 spec-on per-row variable accept** — slice 2b.
- **Hybrid (DeltaNet) + Sliding cache primitives for spec-on** —
  slice 3.
- **Real-model parity** — sub-unit (f).
- **Spec metric emission** — sub-unit (g).
- **Bench `--speculative` flag** — sub-unit (h).

---

## 6. Spec-off byte-identical surface — what must NOT shift

OPENING §6.1 (c) acceptance phrasing:

> spec-off rows under spec-on batch produce identical tokens to a
> fully-spec-off batch.

The cleanest enforcement: keep the `_decode_phase` path literally
unchanged when `draft_engine` is None / Noop. The new spec-on path
is a separate helper. Existing tests serve as the regression net:

- **P-2 left-padding arithmetic** —
  `tests/test_p2_left_padding_does_not_corrupt_any_row.py` (and
  related). Left-padding unchanged; spec-off cache state shape
  unchanged.
- **P-2 hit/miss admission** — `tests/test_radix_prefix_*.py`
  (cohort prep + mid-run admit). Sub-unit (c) does not touch
  admit-phase code.
- **P-3 hybrid + MoE batched parity** —
  `tests/test_p3_qwen3_5_moe_batched_parity.py` etc. Hybrid /
  sliding caches not touched at slice 1; the GLOBAL-only gate
  ([F-4]) means a spec-active batcher could not even construct
  on those adapters, so cross-contamination is structurally
  impossible.
- **P-4.5 fairness (length-spread reorder)** —
  `tests/test_p4_5_*.py`. Reorder logic is in `Engine`, not
  `ContinuousBatcher`; sub-unit (c) inherits Engine's spec-off
  byte-identical guarantee through the same default-Noop pattern.

Any of these tests turning red under spec-off means slice 1 leaked
surface into the spec-off path. Slice 1 design must not touch
helpers outside the new `_decode_phase_spec` and the constructor
gate.

---

## 7. Decisions (round-2 review sign-off, 2026-04-30)

The round-2 OQ table was resolved by the user as follows. Slice-1
implementation enters with these answers fixed:

| # | Question | Decision |
| --- | --- | --- |
| O-1 | Slice 1 = B=1 spec-on synthetic, multi-row raises with slice-2b marker, GLOBAL-only gate | Yes |
| O-2 | Drafter per-`req_id` refactor as standalone slice 2a | Yes |
| O-3 | `_decode_phase_spec` next to `_decode_phase` in `batcher.py` | Yes; revisit once helper genuinely bloats |
| O-4 | Lean directly on `BatchKVCache.prepare(right_padding) + finalize()` or pin first | **Pin first** — `tests/test_batch_kv_cache_right_trim.py` lands as deliverable 0 before any batcher patch consumes the primitive |
| O-5 | In-place `row.state.output_token_ids` mutation vs. transient ctx | **Transient ctx** — each spec cycle builds a throwaway `RequestState` carrying `request_id=row.req_id`, `token_ids=tuple(row.prompt_ids)`, `output_token_ids=list(row.generated)`. `row.state` is never mutated by `_decode_phase_spec`. Avoids creating an implicit "spec path half-maintains output history" contract |
| O-6 | GLOBAL-only capability gate hard vs. soft | **Hard** — HYBRID_DELTANET / SLIDING are rejected at constructor when `draft_engine` is non-Noop. Soft warn-but-allow defers errors to runtime |

Slice 1 implementation order, locked:

1. Land `tests/test_batch_kv_cache_right_trim.py` (deliverable 0).
   Standalone commit, no `silica/` changes.
2. Land slice-1 surface area + `_decode_phase_spec` + tests
   (deliverables 1–6). Single commit at sub-unit (c) slice-1 grain.
