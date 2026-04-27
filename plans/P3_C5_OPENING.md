# P-3-C5 — Hybrid-DeltaNet recurrent-state snapshot / restore: opening

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Version      | v1.0.0 (draft)                                                     |
| Last updated | 2026-04-24                                                         |
| Status       | opening doc — implementation not yet started                       |
| Maintainer   | Xin Zhou                                                           |
| Parent unit  | P-3-C (DeltaNet recurrent-state plumbing)                          |
| Parent docs  | `plans/PLAN.md` §7 P-3; `plans/P3_DELTANET_SURVEY.md`                |
| Scope        | Design and land the adapter-owned recurrent-state snapshot /      |
|              | restore surface that unblocks `ContinuousBatcher` +               |
|              | `RadixPrefixCache` cooperation on hybrid-DeltaNet adapters        |
|              | (Qwen3.5-0.8B / 4B / 35B-A3B). Exit criterion: the                |
|              | "``has_recurrent_state=True`` refuses ``prefix_cache``" guard at   |
|              | `silica/scheduler/batcher.py:154-166` is removed and Qwen3.5      |
|              | runs under `RadixPrefixCache` with correctness demonstrated by    |
|              | smoke tests.                                                       |

## 1. Problem statement

### 1.1 What is broken on the current tree (v1.7.5)

`silica/scheduler/batcher.py:152-166` refuses
`ContinuousBatcher(prefix_cache=RadixPrefixCache(...))` whenever
the adapter reports `capabilities().has_recurrent_state=True`. This
gate is defensive, not aspirational: the admission path
(`_extract_and_insert_prefix` / `_admit_single_hit_row` →
`build_seeded_batch_kv`) assumes per-token K/V slicing on every
layer's cache, and DeltaNet's recurrent state is a running
accumulation over the full sequence (`cache[1]`, fp32
`(B, Hv, Dv, Dk)` per layer — `plans/P3_DELTANET_SURVEY.md` §3.2)
that cannot be sliced at an arbitrary block-aligned prefix.

Current consequence:

- **All Qwen3.5 variants** (0.8B / 4B / 35B-A3B) can run through
  `ContinuousBatcher` **only** with `prefix_cache=None`. The miss-
  only admission path is the fallback.
- **P-5 benches routed around it.** `plans/P5_OPENING.md` §6.5
  migrated every codec-backed PPL bench row off Qwen3.5 onto the
  non-recurrent Qwen3-0.6B target so the prefix-cache store path
  could be exercised. The price was giving up real-target Qwen3.5
  PPL numbers.
- **(b-static) backlog item is blocked.** PLAN §7 P-5 Notes
  (b-static) — "Qwen3.5-4B end-to-end codec PPL vs
  `vqbench/REPORT.md` static baseline (`PPL_fp16 ≈ 10.3866`)" — is
  v1.7.2-deferred as post-P-5 follow-up, specifically because it
  needs either `RadixPrefixCache` cooperation on Qwen3.5-4B or a
  monkey-patch route. P-3-C5 lands the first option.
- **Preempt/replay on hybrid adapters.** Without an explicit
  snapshot pathway, a preempted Qwen3.5 row loses its recurrent
  state; on replay, it must re-prefill from scratch (assuming a
  prefix-cache hit serves the prompt). Even the miss-only path
  therefore has a correctness-vs-performance dilemma under
  preemption.

### 1.2 What this unit unblocks

- **Qwen3.5-0.8B / 4B / 35B-A3B under `RadixPrefixCache`.** The
  primary production payoff: hybrid-DeltaNet models participate in
  prefix-cache-backed admission just like dense Qwen3 does.
- **P-5 (b-static) PPL baseline.** Unblocks the Qwen3.5-4B end-to-
  end PPL cross-check against `vqbench/REPORT.md` §3.1 without
  the monkey-patch fallback.
- **Preempt/replay bit-exactness on hybrid adapters.** The
  recurrent side gets the same round-trip guarantee that the
  full-attention side has had since P-2.
- **P-7 precedent.** `Qwen3_5Adapter.rollback_state` currently
  raises `NotImplementedError` and hints that the "pre-draft
  snapshot of the recurrent state" pathway lands with P-7
  speculative decoding (`silica/models/qwen3_5.py:224-233`). C5's
  snapshot/restore surface, if designed right, can be the single
  adapter-owned API that serves both preempt (this unit) and
  draft-snapshot (P-7).

### 1.3 Why this is a large unit, not a quick fix

C5 is not "flip a bit in the batcher". The recurrent-state
surface has three separate consumers that must all agree:

1. **Preempt/replay** (batcher evicts a live row and replays
   later) — needs a snapshot of state at the preempt moment and a
   restore on replay.
2. **Prefix-hit admission** (new row admitted with a block-aligned
   prefix match) — needs a recurrent state that corresponds to
   that exact prefix boundary. DeltaNet state at block boundary
   `k × block_size` is not recoverable from per-block K/V; it must
   be stored explicitly or re-derived.
3. **Speculative rollback** (P-7; draft mispredict causes
   `n_reject > 0` tokens to be rolled back) — needs a snapshot
   before the draft and a restore on rejection.

All three want "a snapshot of the recurrent state at some token
position, restorable later". The difference is who takes the
snapshot, who owns it, and when it is discarded. A unified
adapter-owned snapshot API is the leverage point.

## 2. Locked design decisions (2026-04-24)

These are fixed before sub-unit drafting. Subsequent revisions may
only tighten or reinterpret; they may not flip these axes.

### 2.1 Adapter-owned snapshot, NOT a new `RecurrentStateManager`

The recurrent state is model semantics, not container semantics.
Shape, dtype, layer-to-state mapping, update rules, slicability,
and clone semantics all live inside the adapter / mlx-lm model
implementation. A new top-level manager Protocol would create a
supposedly-generic surface that, in practice, forwards every
method through to the adapter — plus it would expand I-1..I-5 to
I-1..I-6, which D-003 / D-013 discipline wants to avoid without
a concrete second implementation pressure.

**Placement:** snapshot / restore are **adapter-owned hooks**.
`ContinuousBatcher` consumes an opaque `RecurrentSnapshot` value
object and never reaches inside it. `silica.kvcache` (KV
containers) is **not** touched — recurrent state is not KV, is
never stored inside a `KVCache` / `ArraysCache` slot that a KV
manager owns outside the adapter, and is not serialised as a
"KV block" for `SyntheticPrefixBlockStore`.

**Promotion path:** introduce the snapshot surface as an
**optional capability / mixin** first. Adapters that support
recurrent state (`Qwen3_5Adapter`, later `Qwen3_5MoEAdapter`)
implement it; adapters that don't (`Qwen3Adapter`, `Gemma4Adapter`)
don't. After C5.4 lands and shapes are stable, promote to the
core `ModelAdapter` Protocol in a separate revision — the same
pattern D-015 used for `StateDelta`.

### 2.2 Deep integration is the exit criterion

"Shallow integration" — preempt/replay snapshot only, keep
`prefix_cache` refused on hybrid adapters — does NOT close this
unit. That path leaves Qwen3.5 unable to use `RadixPrefixCache`,
leaves (b-static) blocked, and only solves half the preempt/replay
problem (the half where the prefix-cache replay wouldn't help
anyway).

**Exit criterion for C5:** `ContinuousBatcher(
prefix_cache=RadixPrefixCache(...))` runs on Qwen3.5-0.8B to
token-sequence bit-exactness against the no-prefix-cache path, AND
the `has_recurrent_state` guard at
`silica/scheduler/batcher.py:154-166` is removed.

### 2.3 Staged rollout — guard removal is the LAST step, not the first

Sub-unit order:

1. **C5.0 State inventory.** Survey-grade catalogue of which
   tensors, which dtypes, which lifetimes, which per-request
   isolation edges. Mostly doc work; no code.
2. **C5.1 Adapter-owned snapshot / restore surface.** Introduce
   `RecurrentSnapshot` value object + adapter methods; Qwen3.5
   adapter implements. **Guard stays** — snapshot exists as a
   callable API, but nothing in the batcher calls it yet.
3. **C5.2 Preempt/replay integration.** Batcher's preempt path
   calls `adapter.snapshot_recurrent_state(self._batch_cache,
   victim_row_idx)` before filtering the row out; replay calls
   `adapter.restore_recurrent_state(row_cache, row_idx, snapshot)`
   on the rebuilt row cache before stitching back into the
   active batch. **Guard stays** — this is a correctness
   improvement on the `prefix_cache=None` path; hybrid adapters
   are still refused on `prefix_cache != None`.
4. **C5.3 RadixPrefixCache cooperation.** Design the mechanism
   that gives `_admit_single_hit_row` a recurrent state consistent
   with the full-attention K/V prefix it seeds. Two options
   (§6.3). **Guard still stays** — the admission path handles
   hybrid adapters but the top-level construction check remains
   defensive until C5.4's end-to-end smoke passes.
5. **C5.4 Guard removal + smoke.** Remove the `has_recurrent_state
   + prefix_cache` guard; add Qwen3.5-0.8B /-4B /-35B-A3B smoke
   tests gated on HF cache; bit-exactness vs the
   `prefix_cache=None` path is the correctness oracle; bench
   infrastructure extension (Qwen3.5-backed PPL / compression /
   prefix-hit rows) lands here too.

Rationale for guard-last: removing the guard early would let
untested code paths throw mid-inference rather than at
construction. The guard is the single explicit surface that says
"this combination is not yet validated"; removing it is the
ratification step, not an enabling step.

### 2.4 Value-object, not manager-of-state

`RecurrentSnapshot` is a frozen value object: construct once, pass
around, discard. It is **not** a live state container with
mutation methods. The live state continues to live inside the
cache object that owns the current execution: `KVManager`'s
single-request `cache_list(req_id)` on the adapter prefill /
decode path, or `ContinuousBatcher._batch_cache` on the batched
path. The adapter owns **interpretation** of those cache objects
(which layers are recurrent, which `ArraysCache` slots are
`cache[0]` conv state vs `cache[1]` recurrent state, and how to
clone / restore them), not long-lived storage for in-flight
batch rows.

This matches D-015's `StateDelta` framing — read-only snapshot,
adapter-owned semantics, caller-owned live storage — and keeps the
batcher's mental model simple: "I hold a cache list / row index
and an opaque snapshot; when the row resumes, I ask the adapter to
restore that snapshot into the supplied cache row."

## 3. Scope

### 3.1 In scope (must land inside C5)

- Adapter-owned `snapshot_recurrent_state(cache_list, row_idx) ->
  RecurrentSnapshot` and
  `restore_recurrent_state(cache_list, row_idx, RecurrentSnapshot)
  -> None` on the Qwen3.5 adapter (optional capability / mixin —
  `Qwen3Adapter` / `Gemma4Adapter` do not implement). The batcher
  owns the live cache; the adapter interprets it.
- `ContinuousBatcher` preempt/replay wired to the snapshot API on
  hybrid adapters.
- `RadixPrefixCache` cooperation: prefix-hit admission provides
  the adapter with a recurrent state consistent with the prefix
  boundary, via one of the two C5.3 options (§6.3).
- Removal of the `has_recurrent_state=True → refuse prefix_cache`
  guard.
- Qwen3.5-0.8B bit-exactness smoke test (full hybrid exercise).
- Qwen3.5-4B / 35B-A3B HF-cache-gated smoke tests (correctness
  only; larger models optional if cache missing).
- `plans/PLAN.md` §7 P-3 Status sync; `plans/P3_DELTANET_SURVEY.md`
  C-open-3 close.

### 3.2 Out of scope (stays in later phases / units)

- **P-7 speculative draft-rollback snapshot pathway.** C5's
  snapshot API should be **shape-compatible** with draft-rollback
  (i.e. not paint P-7 into a corner) but the actual wiring lands
  at P-7. `rollback_state` continues to raise
  `NotImplementedError` for `n_reject > 0` — C5 only fixes the
  "adapter has no way to snapshot at all" precondition.
- **Partial-prefix recurrent-state reconstruction.** D-015's
  "full prefix only" rule stays. If a request's prompt matches a
  block-aligned prefix in `RadixPrefixCache`, we restore state at
  that block boundary (§6.3). If the match is non-block-aligned,
  the row falls back to miss-path prefill. No interpolation, no
  partial-block recovery.
- **Recurrent-state dtype compression.** Survey C-open-2 is the
  fp32 → fp16 / bf16 question on the recurrent state; that is a
  separate measurement-gated experiment. C5 stores and restores
  whatever dtype the model already uses.
- **Pre-RoPE production routing (P-5-F).** Independent workstream
  that comes after C5 in the backlog order; not entangled with
  recurrent state.
- **`PagedPrefixBlockStore`** cooperation. `RadixPrefixCache`
  currently runs against `SyntheticPrefixBlockStore`; paged-store
  cooperation is unchanged by C5.

### 3.3 Deferred or open for sub-unit resolution

- **Snapshot storage site** (§6.2) — where exactly the snapshot
  lives between snapshot and restore: adapter-local dict, batcher
  preempt record, or prefix-cache-tree node metadata. C5.1
  decides adapter-local; C5.3 decides prefix-cache-tree node.
- **Per-block snapshot cost budget** — the memory cost of option
  (b) in §6.3. C5.3 measures and decides whether a coarser
  snapshot granularity is needed (e.g. every `N` blocks rather
  than every block).
- **Speculative rollback API shape reuse** — whether the
  `RecurrentSnapshot` value object is reused verbatim by P-7 or
  whether P-7 introduces its own variant. C5.1 writes the API
  with P-7 in mind; P-7 decides at its own opening.
- **Conv-state treatment under preempt.** `cache[0]` (1D conv
  window) and `cache[1]` (recurrent state) live in the same
  `ArraysCache`; snapshotting one-without-the-other is incorrect
  (the conv window is required to produce the next token's
  `qkv`). `RecurrentSnapshot` must cover **both** slots, not
  just the recurrent state — C5.1 names them in the value object.

## 4. State inventory (summary; C5.0 produces the full catalogue)

Condensed from `plans/P3_DELTANET_SURVEY.md` §3.2–§3.3:

Per `GatedDeltaNet` layer (`ArraysCache(size=2)`):

| Slot | Name | Shape | Dtype | Owner / lifetime |
| --- | --- | --- | --- | --- |
| `cache[0]` | 1D conv state | `(B, conv_kernel_size - 1, conv_dim)` | input dtype (bf16 / fp16) | Allocated first call inside `GatedDeltaNet.__call__`; advanced in-place every forward. |
| `cache[1]` | Recurrent hidden state | `(B, Hv, Dv, Dk)` | **fp32 always** | Allocated inside `gated_delta_update`; advanced per timestep by `gated_delta_kernel`. |

Per-row state bytes on Qwen3.5-27B (measured 2026-04-19):

- `cache[0]` per layer: `B × 3 × 10240 × 2 ≈ B × 60 KB`.
- `cache[1]` per layer: `B × 48 × 128 × 128 × 4 ≈ B × 3.0 MB` (fp32).
- 48 DeltaNet layers → **~145 MB recurrent + ~3 MB conv per row**.
- For Qwen3.5-0.8B the numbers scale down proportionally (fewer
  layers, smaller heads); C5.0 measures concretely.

Full-attention layers (6 on Qwen3.5-0.8B, 16 on Qwen3.5-27B) carry
standard K/V via `KVCache` and are already covered by
`SyntheticPrefixBlockStore`. C5 does **not** change their storage.

Primitives already on `ArraysCache` (mlx-lm `cache.py`):
`filter(indices)` / `extend(other)` / `extract(idx)` /
`merge([...])` / `prepare(lengths=...)` / `advance(N)` /
`nbytes`. The snapshot / restore implementation can stand on top
of `extract` + `merge` without introducing a new tensor-level
primitive.

## 5. Protocol surface sketch

This is the C5.1 design target, not the final spec. Two placement
options; C5.1 locks between them.

### 5.1 Option A — runtime-checkable Protocol mixin

```python
# silica/models/recurrent.py  (new)

@dataclass(frozen=True)
class _RecurrentLayerEntry:
    """One DeltaNet layer's frozen state pair.

    ``conv_state`` and ``recurrent_state`` are ``mx.array``
    instances that MUST be fully materialised (``mx.eval``-ed)
    and detached from any in-place live cache before the
    enclosing snapshot is constructed — otherwise a later
    `GatedDeltaNet.__call__` that writes in place to the
    originating ``ArraysCache`` slots will silently mutate the
    arrays this entry holds. See R-C5-2 for the verification
    obligation."""
    layer_idx: int
    conv_state: mx.array
    recurrent_state: mx.array


@dataclass(frozen=True)
class RecurrentSnapshot:
    """Opaque, read-only snapshot of one request's recurrent
    state at a point in time. External consumers
    (``ContinuousBatcher``, prefix cache) handle whole snapshots;
    they never inspect the inside.

    Immutability layers:

    - The dataclass is ``frozen=True`` — no attribute reassignment.
    - ``entries`` is a ``tuple``, not a ``dict`` or ``list`` — no
      in-place insertion / deletion / reordering.
    - Each ``_RecurrentLayerEntry`` is itself ``frozen=True`` with
      tuple-safe fields.
    - Element ``mx.array`` tensors carry the standard mlx
      copy-on-write / share-on-read semantics; the snapshot
      constructor's obligation is to ensure the arrays it stores
      do NOT alias live cache buffers that will be written in
      place by subsequent forwards. The recommended
      implementation uses ``mx.array(live_tensor)`` (explicit
      copy) plus ``mx.eval`` before construction — concrete shape
      is C5.1's decision, verified by the R-C5-2 "mutate one, read
      the other" test."""
    entries: tuple[_RecurrentLayerEntry, ...]
    nbytes: int


@runtime_checkable
class RecurrentStateAdapter(Protocol):
    """Mixin implemented by adapters that carry recurrent state.
    Absent on ``Qwen3Adapter`` / ``Gemma4Adapter``; present on
    ``Qwen3_5Adapter`` / ``Qwen3_5MoEAdapter``.

    The ``@runtime_checkable`` decorator lets ``ContinuousBatcher``
    dispatch via ``isinstance(adapter, RecurrentStateAdapter)``
    without knowing the concrete adapter class.

    **Storage / interpretation split.** The batcher owns the live
    cache (``ContinuousBatcher._batch_cache`` is a per-layer list
    of ``KVCache`` / ``ArraysCache`` objects, indexed within each
    layer by batch row). The adapter does NOT own the live state
    of in-flight batched rows — ``KVManager.cache_list(req_id)``
    is the single-request prefill / decode path, separate from
    the batched cache. The snapshot / restore methods therefore
    take the cache list + row index the batcher supplies and let
    the adapter interpret which slots are recurrent (DeltaNet
    layers' ``ArraysCache``) and which are not (full-attention
    ``KVCache``). The adapter does not retain a back-reference to
    the cache list after the call returns."""

    def snapshot_recurrent_state(
        self, cache_list: list[Any], row_idx: int
    ) -> RecurrentSnapshot:
        """Return a frozen snapshot of recurrent state for batch
        row ``row_idx`` across all DeltaNet layers in
        ``cache_list``. Implementation walks the per-layer cache
        objects, applies ``ArraysCache.extract(row_idx)`` (or
        equivalent slice) on the recurrent layers, and ensures
        the captured arrays are detached from the live cache via
        ``mx.eval`` + explicit copy so subsequent in-place writes
        in the live cache cannot alias into the snapshot
        (R-C5-2). Full-attention layers are skipped — their state
        lives in the batcher-owned K/V containers, not here."""
        ...

    def restore_recurrent_state(
        self,
        cache_list: list[Any],
        row_idx: int,
        snapshot: RecurrentSnapshot,
    ) -> None:
        """Write ``snapshot``'s contents into ``cache_list`` at
        batch row ``row_idx``. Implementation walks the
        snapshot's per-layer entries and uses
        ``ArraysCache.merge`` / per-row-write semantics to splice
        the snapshot back into the recurrent layers of the
        supplied cache. The supplied ``cache_list`` may be the
        batcher's active ``self._batch_cache`` (replay path) or a
        freshly-built per-row cache that will be merged into the
        active batch (admission path); the adapter does not need
        to know which."""
        ...
```

Decorate `RecurrentStateAdapter` with
`@typing.runtime_checkable`; `ContinuousBatcher` sniff-checks via
`isinstance(adapter, RecurrentStateAdapter)`. On adapters that
don't implement the mixin (`Qwen3Adapter`, `Gemma4Adapter`),
preempt continues the current "lose state, re-prefill from
scratch" behaviour on the miss-only path, and the prefix-cache
cooperation does not invoke the recurrent-side work. The
isinstance check is localised to preempt entry and admission
entry — no hot-path isinstance-per-forward.

### 5.2 Option B — capability bit on `ModelCapabilities`

Add `has_recurrent_state_snapshot: bool = False` to
`silica/models/capabilities.py`. Methods on the adapter are
still needed at runtime; the capability bit only signals their
presence, it does not provide a callable contract. Adopting
Option B therefore also requires an accessor helper that gives
`ContinuousBatcher` a typed handle on the methods without an
`isinstance` check on a Protocol class. Concrete shape:

```python
# silica/models/recurrent.py  (Option-B variant)

class _RecurrentSnapshotOps(Protocol):
    def snapshot_recurrent_state(
        self, cache_list: list[Any], row_idx: int
    ) -> RecurrentSnapshot: ...
    def restore_recurrent_state(
        self,
        cache_list: list[Any],
        row_idx: int,
        snapshot: RecurrentSnapshot,
    ) -> None: ...


def recurrent_snapshot_ops(
    adapter: ModelAdapter,
) -> _RecurrentSnapshotOps | None:
    """Return a typed handle on the adapter's snapshot methods
    when ``adapter.capabilities().has_recurrent_state_snapshot``
    is ``True``; otherwise ``None``. Asserts that the declared
    capability matches the adapter's actual method surface —
    raises at call time rather than silently degrading if an
    adapter declares the capability without implementing the
    methods."""
    ...
```

Under Option B, `ContinuousBatcher` consults the capability bit
first, then grabs the typed handle via `recurrent_snapshot_ops`
and calls through it. Declared-but-unimplemented is a loud
failure rather than a silent fall-through. Without the helper,
Option B quietly degrades to duck-typing with no static check at
all, which is not acceptable for a load-bearing snapshot path.

### 5.3 Lean: Option A — runtime-checkable mixin Protocol

**Lean: Option A.** Option A provides a direct callable contract
(`isinstance(adapter, RecurrentStateAdapter)` is runtime-checkable
when `@runtime_checkable` decorates the Protocol, and mypy sees
the method surface statically); Option B requires the
`_RecurrentSnapshotOps` helper above plus a capability bit, which
is two coordinated surfaces where one would do. The existing
capability flags on `ModelCapabilities` (`has_recurrent_state`,
`has_moe`) are declarative-only signals — they inform scheduler
gating but don't carry method contracts; the snapshot API needs a
method contract. Option A is the idiomatic fit.

Option B remains a valid fallback if Option A tangles with D-016
capability-semantics or if a future adapter needs the capability
signal without the methods (none is known). C5.1 revisits this
choice with the API in hand; if it flips, the §5.2 sketch is the
concrete shape that must accompany the flip.

### 5.4 P-7 speculative-rollback compatibility check

`Qwen3_5Adapter.rollback_state(self, req_id: str, n_reject: int)`
today raises for any `n_reject > 0` and points at P-7. C5's
`RecurrentSnapshot` must be shape-reusable by P-7's "pre-draft
snapshot then collapse on commit" pattern:

- Snapshot taken before the draft rollout → same API call as C5.2
  preempt snapshot.
- Restore on draft reject → same API call as C5.2 replay restore.
- Accept on commit → discard the snapshot (no restore needed); P-7
  does not need a distinct "commit" method because discarding the
  snapshot is a no-op on the adapter side.

C5.1 writes the API accordingly; P-7 consumes it without
modification. If P-7 finds a genuinely different requirement, P-7
can extend the API — C5 just commits to not painting P-7 into a
corner.

## 6. Sub-unit breakdown

### 6.1 C5.0 — State inventory

**Goal:** produce a concrete catalogue of DeltaNet state on all
three Qwen3.5 targets (0.8B, 4B, 35B-A3B), plus any other hybrid
adapter (Qwen3.5-MoE). Catalogues shape / dtype / bytes per row
per layer, and confirms the mlx-lm `ArraysCache` primitives
cover the snapshot / restore operations.

**Deliverable:** `plans/P3_C5_STATE_INVENTORY.md` — per-model layer
pattern, per-state tensor shape, concrete byte cost at batch
size 1, and a grep-verified list of every site that writes
`cache[0]` / `cache[1]` in the mlx-lm codebase (confirms the
snapshot includes every written slot).

**Acceptance:** inventory table for Qwen3.5-0.8B is complete and
cross-verified against a real forward pass (capture shapes at
runtime, not just read off `args`).

**Blocks:** C5.1 (snapshot design needs to know what it wraps).

### 6.2 C5.1 — Adapter-owned snapshot / restore

**Goal:** land the snapshot surface on `Qwen3_5Adapter` with a
frozen `RecurrentSnapshot` value object. Lock §5.1 vs §5.2
placement choice. No batcher changes — the guard stays.

**Deliverable:**

- `silica/models/recurrent.py` (new) with `RecurrentSnapshot` +
  (either) `RecurrentStateAdapter` Protocol mixin or
  `ModelCapabilities.has_recurrent_state_snapshot` bit.
- `Qwen3_5Adapter.snapshot_recurrent_state` / `.restore_recurrent_state`
  implementations walking `cache_list` and `extract`-ing per-row
  `ArraysCache`s.
- Unit tests: round-trip `snapshot → restore → forward → logits`
  bit-exact against `forward → logits` on the same input (no
  drift introduced by the snapshot mechanism).
- `plans/P3_DELTANET_SURVEY.md` C-open-3 status update to reflect
  the API landing (doesn't close — full close comes with C5.3).

**Acceptance:** 32 bit-exact round-trip tests across
`(layer_count, batch_size, step_count) ∈ {full-hybrid, one-layer,
B=1, B=4, 0-step, 1-step, 4-step}` on Qwen3.5-0.8B. HF-cache-gated.

**Blocks:** C5.2 (needs the API to call).

### 6.3 C5.2 — Preempt-side snapshot capture / stash (post-pivot)

**Goal:** `ContinuousBatcher._apply_preempt` captures the
victim row's recurrent snapshot before filtering and stashes it
on the replay's `_PendingAdmit.recurrent_snapshot` field.
**Restore is NOT enabled on the full-replay path** — see the
boundary-mismatch derivation in `plans/P3_C5_DRIFT_EXPERIMENT/README.md`
"Implication for C5.2 acceptance" (added at C5.2 landing time).
The natural snapshot boundary (`T + len(generated) - 1` consumed)
is one token earlier than the replay's post-prefill cache state
(`T + len(generated)` consumed); restoring at that misaligned
site would rewind the cache and erase the last generated
token's consumption. Restore enablement is deferred to C5.3
(prefix-hit admission, where the boundary aligns naturally).

**Deliverable:**

- **`_PendingAdmit` schema** extends by one optional field
  `recurrent_snapshot: RecurrentSnapshot | None`, default
  `None`. Plain (`add_request`) admissions and adapters without
  `RecurrentStateAdapter` leave the field `None` and behave as
  today.
- **`_PreemptedRow` result dataclass** carries the detached
  `_BatchRow` plus the captured `RecurrentSnapshot | None`.
  Returned by `_preempt_active_row`; consumed by
  `_apply_preempt` to compose the replay record.
- **`_preempt_active_row` capture site.** After `victim_row_idx`
  is resolved and BEFORE any prefix-extract / state-transition /
  `layer_cache.filter(kept)` call: under
  `isinstance(self._adapter, RecurrentStateAdapter)`, call
  `adapter.snapshot_recurrent_state(self._batch_cache,
  victim_row_idx)`. Snapshot the cache while `victim_row_idx`
  still indexes a valid row in `self._batch_cache`.
- **`_apply_preempt` stash.** Unpack `_PreemptedRow.detached` /
  `.recurrent_snapshot`; pass the snapshot into the replay's
  `_PendingAdmit(..., recurrent_snapshot=snap)`.
- **`_admit_miss_cohort` does NOT call `restore_recurrent_state`**
  on the full-replay path. The restore site is reserved for
  C5.3. A pinned in-source comment marks the restore-call
  position with the boundary-mismatch rationale, so a future
  reader doesn't accidentally enable it.

**Acceptance gates** (verified by
`tests/test_qwen3_5_preempt_replay.py`):

1. Snapshot-before-filter call order (spy-adapter ordering test
   on a B=2 cohort that actually exercises `filter`).
2. `_PendingAdmit.recurrent_snapshot` stashed on the replay
   record (synthetic test).
3. Restore is NOT called on the full-replay admission path
   (synthetic spy-adapter test directly).
4. Snapshot survives pending-lifetime aliasing across
   intervening `step()` calls (synthetic test).
5. Boundary metadata: snapshot describes
   `len(prompt_ids of replay pending) - 1` consumed
   (= `T + len(generated) - 1`).

Guard (`has_recurrent_state + prefix_cache`) unchanged. C5.2
runs every test under `prefix_cache=None`. Bit-exact restore vs
no-preempt oracle is intentionally NOT a C5.2 gate; that lands
at C5.3 where the prefix-hit boundary makes restore correct.

**Blocks:** C5.3 (consumes the snapshot capture surface; restore
enablement happens there).

### 6.4 C5.3 — RadixPrefixCache cooperation

**Goal:** `_admit_single_hit_row` gives the admitted row a
recurrent state consistent with its prefix boundary.

Two implementation options; C5.3 picks one based on measurement.
The naive "run only DeltaNet layers on prefix tokens" is NOT a
viable shortcut — DeltaNet layer inputs are hidden states
produced by all preceding layers' embedding → GLOBAL attention →
MLP → residual path on every token, so "recurrent-only replay"
would require caching per-layer hidden states for every prefix
token, which defeats any memory saving the scheme is supposed to
provide. Both options below reflect this constraint honestly.

**Option (a) — "full prefix replay on admission".** When a
prefix-hit row is admitted, run the full model forward on the
prefix tokens for that row so the recurrent state accumulates
naturally. The replay is driven from inside the batcher (NOT via
`adapter.prefill`, which is the single-request `KVManager`
path and does not touch `self._batch_cache` / the seeded
`row_cache` that `_admit_single_hit_row` is building). Concrete
shape: build a temporary or extended `row_cache` for the
admitting row, call `forward_batched(self._model, prefix_tokens,
row_cache)` with `insert_detached` suppressed for this run (the
prefix K/V is already resident in the store from the original
emit; re-inserting would double-count). The full-attention K/V
produced by the replay forward is discarded; the only retained
output is the post-replay recurrent state inside the row's
DeltaNet `ArraysCache` slots, which then merges into
`self._batch_cache` along with the seeded full-attention K/V.

**No prefill compute saving** relative to the miss path on the
recurrent + MLP + attention sides — the whole point of a prefix
cache is to skip that compute on the prefix, and (a) skips none
of it. The one saving is memory: no per-block recurrent
snapshot storage on the prefix-cache tree node. Option (a) is
effectively "disable prefix-cache compute reuse on hybrid
adapters but keep the K/V slot-pool plumbing" — a correctness
fallback when (b) is memory-infeasible.

**Option (b) — "per-block boundary snapshot"** in the prefix
cache. The snapshot **must be captured when prefill / decode
crosses each block boundary, NOT when `_extract_and_insert_prefix`
runs**. The latter timing is unworkable: at extraction time the
DeltaNet recurrent state in the live cache reflects the **full
processed prefix**, not the per-block intermediate states; there
is no way to recover the recurrent state at block boundary `k ×
block_size` from the live state at boundary `N × block_size`
(the operation `gated_delta_update` is not invertible without
full per-step memory). If snapshots were captured only at extract
time, a future prefix-hit on a strictly-shorter block-aligned
prefix (which is the common case the prefix cache is designed
for) would restore an incorrect recurrent state corresponding to
the wrong boundary.

The correct shape is in-flight capture: a hook inside the
batcher's prefill / decode loop detects every time the running
processed-token count crosses a multiple of `block_size`, calls
`adapter.snapshot_recurrent_state(...)`, and stores the snapshot
in a side buffer keyed by (block id, row). When
`_extract_and_insert_prefix` later inserts the block into the
tree, it pulls the matching snapshot from the side buffer and
attaches it to the tree node alongside the K/V handle.

Two viable implementations of the in-flight capture:

- **(b.i) Slice prefill into `block_size`-aligned segments.** The
  batcher splits a `T`-token prefill into `ceil(T / block_size)`
  forward calls, each ending exactly on a block boundary, and
  captures a snapshot between calls. Simpler, but pays the
  per-call overhead `ceil(T / block_size)` times. The pattern is
  similar to but separate from P-4.5 chunked prefill — the
  chunked-prefill chunk size and the prefix-cache block size
  are independent knobs.
- **(b.ii) Hook the inner forward.** A callback on the model
  forward intercepts at every `block_size`-th step and snapshots
  inline. Lower per-call overhead but more invasive (touches
  mlx-lm forward).

Lean: **(b.i)**. The slicing approach keeps the snapshot
machinery confined to the batcher; mlx-lm forward stays
untouched. Per-call overhead is small relative to the per-block
recurrent-update cost (one extra `mx.eval` boundary per
block_size tokens). C5.3 measures and confirms; if (b.i)
overhead exceeds an acceptable bar, fall back to (b.ii).

**Full compute saving on hit** — once snapshots exist on tree
nodes, prefix-hit admission restores both K/V and recurrent
state at the matched boundary; the recurrent side is skipped on
the prefix just like the full-attention side is. This is what a
prefix cache is supposed to deliver. Memory cost: per-tree-node
DeltaNet snapshot bytes (Qwen3.5-0.8B: TBD at C5.0; Qwen3.5-27B:
~145 MB per block-boundary per tree node). The prefix-cache's
existing eviction policy governs the total memory cost.

**Lean: Option (b)** — it is the only option that delivers the
production payoff a prefix cache is supposed to deliver on
hybrid-DeltaNet models. Option (a) becomes the fallback only
when C5.3's measurement shows (b) is memory-infeasible for the
target at hand. Measurement target at C5.3: per-tree-node
snapshot bytes on Qwen3.5-0.8B (expected to be small enough to
choose (b) unconditionally for 0.8B), then Qwen3.5-4B (the
(b-static) target — (b) is strongly preferred to match vqbench's
prefix-cache-backed benchmark shape), then Qwen3.5-27B / 35B-A3B
(where (a) may need to kick in as a per-model fallback if the
per-block cost exceeds a configurable cap).

**Deliverable:**

- `_admit_single_hit_row` handles `has_recurrent_state=True`
  adapters under the chosen option (decision locked in the C5.3
  design note after measurement).
- Infrastructure the chosen option needs:
  - **Option (a) full prefix replay**: in
    `_admit_single_hit_row`, after the seeded `row_cache` is
    built from `fetch_detached_blocks`, run a batcher-level
    `forward_batched(self._model, prefix_tokens, row_cache)`
    with a per-admission flag suppressing
    `SyntheticPrefixBlockStore.insert_detached` (the prefix K/V
    is already resident from the original emit; double-insertion
    would corrupt residency accounting). Drives the replay
    through the same forward function the batched decode loop
    uses, NOT through `adapter.prefill` — `adapter.prefill` is
    the single-request `KVManager.cache_list(req_id)` path and
    does not see the seeded `row_cache` the batcher just built.
    No new adapter method is required.
  - **Option (b) block-boundary snapshot**: install the
    in-flight capture hook (b.i slicing or b.ii forward callback
    per the choice in the design note); extend
    `RadixPrefixCache` tree nodes to store a
    `RecurrentSnapshot | None` keyed alongside the existing block
    handle; in `_extract_and_insert_prefix`, look up the matching
    snapshot in the side buffer and attach to the node; in
    `_admit_single_hit_row`, after the seeded `row_cache` is
    built, call
    `adapter.restore_recurrent_state(row_cache, row_idx=0,
    snapshot)` before merging into `self._batch_cache`.
- `plans/P3_DELTANET_SURVEY.md` C-open-3 resolved (either "state
  reconstructed via full-prefix replay" under (a), or "state
  restored from in-flight per-block-boundary snapshot" under
  (b)).

**Acceptance:** Qwen3.5-0.8B with `RadixPrefixCache` admits
prefix-hit rows and produces bit-exact tokens vs the
`prefix_cache=None` path on the same prompt. Memory cost
measured and recorded.

**Blocks:** C5.4 (guard removal needs the hit path to work).

### 6.5 C5.4 — Guard removal + end-to-end smoke

**Goal:** remove the construction-time guard, add smoke tests for
all three Qwen3.5 targets, close P-3-C5.

**Deliverable:**

- Delete lines 152-166 in `silica/scheduler/batcher.py` (the
  `has_recurrent_state + prefix_cache` rejection). The `SLIDING`
  guard at lines 167-189 is independent and stays.
- `tests/test_qwen3_5_prefix_cache_smoke.py` — Qwen3.5-0.8B
  end-to-end via `ContinuousBatcher(prefix_cache=...)`:
  - bit-exact tokens vs `prefix_cache=None` baseline
  - prefix-hit admission executes the C5.3 code path at least once
  - preempt/replay executes the C5.2 code path at least once
- HF-cache-gated smoke tests on Qwen3.5-4B and Qwen3.5-35B-A3B
  (correctness only; larger models skipped when cache missing).
- Bench scenarios: add `qwen3.5-0.8b-wikitext-ppl-*` and
  `qwen3.5-4b-wikitext-ppl-*` rows that actually exercise the
  codec + prefix cache path (unblocks P-5 (b-static) by
  construction).

**Acceptance:**

- **Qwen3.5-0.8B** smoke passes unconditionally on any machine
  with the 0.8B HF cache (0.8B is small enough to keep in the
  routine smoke set; this is the hard local gate).
- **Qwen3.5-4B** smoke passes whenever the 4B HF cache is
  present; it is the direct downstream unblocker for P-5
  (b-static) which pins 4B specifically against
  `vqbench/REPORT.md`, so a passing 35B run does NOT substitute
  for it. If the 4B cache is absent on the machine running CI,
  the test skips cleanly; if it is present, the test must pass.
- **Qwen3.5-35B-A3B** smoke is an additional HF-cache-gated
  coverage target that exercises the MoE × hybrid-DeltaNet
  combination; it does not substitute for the 4B gate.
- Full existing test suite stays green.
- `plans/P3_DELTANET_SURVEY.md` C-open-3 closed with the adopted
  C5.3 option named.

## 7. Acceptance — phase exit

P-3-C5 closes when all of the following hold:

1. Sub-unit acceptance (C5.0–C5.4) green.
2. `silica/scheduler/batcher.py` no longer refuses the
   `has_recurrent_state=True + prefix_cache != None` combination.
3. Qwen3.5-0.8B bit-exactness smoke via `ContinuousBatcher(
   prefix_cache=RadixPrefixCache(...))` passes on the committed
   tree.
4. `plans/PLAN.md` §7 P-3 Status reflects C5 close; `plans/P3_DELTANET_SURVEY.md`
   C-open-3 resolved with the adopted C5.3 option named.
5. P-5 (b-static) backlog item is unblocked (the monkey-patch
   fallback is no longer the only path); the actual (b-static)
   landing is a separate post-P-3-C5 unit under P-5's backlog.

## 8. Risks and open questions

### 8.1 R-C5-1 — fp32 recurrent state multiplies memory under option (b)

At Qwen3.5-27B scale, per-block recurrent snapshots cost ~145 MB.
A small prefix-cache with 10 blocks would pay ~1.5 GB per row of
snapshots. Mitigation: option (a) or coarser snapshot granularity.
C5.3 measures first.

### 8.2 R-C5-2 — `gated_delta_update` in-place semantics

`GatedDeltaNet.__call__` writes in place to `cache[0]` / `cache[1]`
every step. `extract(idx)` returns a new `ArraysCache`, but
whether that copy is truly independent of the original
batched-cache allocation needs a runtime verification (a simple
"mutate one, read the other" test). If the copy shares storage,
snapshot / restore silently corrupts the live batch. C5.1
acceptance includes this verification explicitly.

### 8.3 R-C5-3 — partial-block prefix matches

`RadixPrefixCache` is block-granular. If a request's prompt has a
prefix match that is not an exact multiple of `block_size` tokens,
the current miss-path handles it. Under C5.3 option (b), the
block-boundary snapshot works only at exact block boundaries —
which matches the existing cache's own block-alignment rule. No
regression, but worth documenting.

### 8.4 Open — sub-unit ordering between C5.2 and C5.3

Current plan is C5.2 before C5.3. An alternative is to do C5.3
first (prefix-hit cooperation is the larger unblock) and do C5.2
after (preempt/replay of hybrid models). The advantage of
C5.2-first is that the simpler snapshot-on-evict case validates
the Protocol choice before the more complex prefix-cooperation
case builds on it. C5.1 opening revisits this question with the
snapshot API in hand.

### 8.5 Open — C5 test cost on CI

Qwen3.5-4B is ~8 GB of weights; Qwen3.5-35B-A3B is ~20 GB. Smoke
tests are HF-cache-gated per the `test_engine_admission_reorder.py`
pattern, so machines without the cache skip cleanly. Machines
with the cache will pay the load cost once per pytest session —
acceptable per existing P-3 test conventions.

## 9. References

- `plans/PLAN.md` §7 P-3 (parent phase), §7 P-5 Notes (b-static)
  (dependent item), D-015 (StateDelta framing), D-016
  (ModelCapabilities).
- `plans/P3_DELTANET_SURVEY.md` — §3.2 state layout, §3.4
  ArraysCache primitives, §6 C-sub-unit decomposition including
  the earlier C5 sketch, §7 C-open-1 / C-open-2 / C-open-3 local
  open items, §8 source-code pointers.
- `silica/scheduler/batcher.py:152-166` — the guard this unit
  removes.
- `silica/models/qwen3_5.py:142-253` — current
  prefill / decode_step / rollback_state / state_from_prefix /
  free_state implementations. `rollback_state` hints at the P-7
  snapshot pathway C5 shares the API with.
- `silica/models/adapter.py:141-155` — `StateDelta` framing (the
  read-only byte-accounting shape C5 follows for
  `RecurrentSnapshot`).
- mlx-lm `cache.py:594-696` — `ArraysCache` with `filter` /
  `extend` / `extract` / `merge` / `prepare` / `advance` / `nbytes`
  (C5 snapshot / restore stands on `extract` + `merge`).
- mlx-lm `qwen3_5.py` + `gated_delta.py` — the GatedDeltaNet
  forward contract (§3.2 / §3.5 of the survey for specifics).

## 10. Implementation pause point

Ship this opening doc. User reviews §2 locked decisions, §3 scope
boundaries, §5 Protocol placement (Option A vs B), and the C5.3
option (a)-vs-(b) framing. On approval, implementation starts at
C5.0 — produce `plans/P3_C5_STATE_INVENTORY.md` against the
loaded Qwen3.5-0.8B adapter. Each sub-unit (C5.1 → C5.4) is a
separate landing commit with its own review pause, mirroring the
P-5 cadence.
